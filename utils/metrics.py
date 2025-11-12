import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import wandb
from tqdm import tqdm
import argparse

from config import ModelConfig, TrainingConfig
from models.imis_net import IMISNet
from data.dataset import MedicalSegmentationDataset, collate_fn
from data.transforms import MedicalImageTransform
from utils.loss import CombinedLoss
from utils.metrics import compute_metrics
from utils.prompt_simulation import PromptSimulator


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, prompt_simulator,
                   config, device, epoch):
    """Execute one training epoch with mixed precision and gradient accumulation."""
    model.train()

    total_loss = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        gt_masks = batch['masks'].to(device)

        B = images.shape[0]

        prompts_list = []
        for i in range(B):
            mask = gt_masks[i, 0]
            prompts = prompt_simulator.generate_prompts(mask)

            if 'points' in prompts:
                prompts['points']['coords'] = prompts['points']['coords'].to(device)
                prompts['points']['labels'] = prompts['points']['labels'].to(device)
            if 'boxes' in prompts:
                prompts['boxes'] = prompts['boxes'].to(device)

            prompts_list.append(prompts)

        batched_prompts = {}
        if 'points' in prompts_list[0]:
            batched_prompts['points'] = {
                'coords': torch.cat([p['points']['coords'] for p in prompts_list]),
                'labels': torch.cat([p['points']['labels'] for p in prompts_list])
            }
        if 'boxes' in prompts_list[0]:
            batched_prompts['boxes'] = torch.stack([p['boxes'] for p in prompts_list])

        with autocast(device_type='cuda', enabled=config.use_amp):
            masks_per_iter, iou_per_iter = model(images, batched_prompts, iterative=True)

            final_masks = masks_per_iter[-1]
            final_masks_resized = torch.nn.functional.interpolate(
                final_masks,
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            loss, loss_dict = criterion(final_masks_resized, gt_masks)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            metrics = compute_metrics(final_masks_resized, gt_masks)
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]

        total_loss += loss.item() * config.gradient_accumulation_steps

        pbar.set_postfix({
            'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
            'iou': f"{metrics['iou']:.4f}",
            'dice': f"{metrics['dice']:.4f}"
        })

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, prompt_simulator, config, device):
    """Validation loop with mixed precision inference."""
    model.eval()

    total_loss = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].to(device)
        gt_masks = batch['masks'].to(device)

        B = images.shape[0]

        prompts_list = []
        for i in range(B):
            mask = gt_masks[i, 0]
            prompts = prompt_simulator.generate_prompts(mask)

            if 'points' in prompts:
                prompts['points']['coords'] = prompts['points']['coords'].to(device)
                prompts['points']['labels'] = prompts['points']['labels'].to(device)
            if 'boxes' in prompts:
                prompts['boxes'] = prompts['boxes'].to(device)

            prompts_list.append(prompts)

        batched_prompts = {}
        if 'points' in prompts_list[0]:
            batched_prompts['points'] = {
                'coords': torch.cat([p['points']['coords'] for p in prompts_list]),
                'labels': torch.cat([p['points']['labels'] for p in prompts_list])
            }
        if 'boxes' in prompts_list[0]:
            batched_prompts['boxes'] = torch.stack([p['boxes'] for p in prompts_list])

        with autocast(device_type='cuda', enabled=config.use_amp):
            masks_per_iter, iou_per_iter = model(images, batched_prompts, iterative=True)

            final_masks = masks_per_iter[-1]
            final_masks_resized = torch.nn.functional.interpolate(
                final_masks,
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            loss, _ = criterion(final_masks_resized, gt_masks)

        total_loss += loss.item()

        metrics = compute_metrics(final_masks_resized, gt_masks)
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def main(args):
    model_config = ModelConfig(
        img_size=args.img_size,
        inter_num=args.inter_num
    )
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        data_path=args.data_path
    )

    device = torch.device(train_config.device)

    if args.use_wandb:
        wandb.init(project="imis-net", config={
            **vars(args),
            'model_config': vars(model_config),
            'train_config': vars(train_config)
        })

    train_transform = MedicalImageTransform(img_size=args.img_size, is_train=True)
    val_transform = MedicalImageTransform(img_size=args.img_size, is_train=False)

    train_dataset = MedicalSegmentationDataset(
        data_path=Path(args.data_path) / 'train',
        transform=train_transform,
        max_masks_per_image=train_config.mask_num_train,
        mode='train'
    )

    val_dataset = MedicalSegmentationDataset(
        data_path=Path(args.data_path) / 'val',
        transform=val_transform,
        max_masks_per_image=1,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = IMISNet(model_config).to(device)

    if args.pretrained_path:
        print(f"Loading pretrained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    criterion = CombinedLoss(
        bce_weight=train_config.bce_weight,
        dice_weight=train_config.dice_weight,
        focal_weight=train_config.focal_weight
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.num_epochs
    )

    scaler = GradScaler(enabled=train_config.use_amp)

    prompt_simulator = PromptSimulator(
        num_points=train_config.num_points_train,
        point_prob=train_config.point_prob,
        box_prob=train_config.box_prob
    )

    best_val_iou = 0.0
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{train_config.num_epochs}")
        print(f"{'='*50}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            prompt_simulator, train_config, device, epoch
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}")

        if (epoch + 1) % train_config.validate_every_n_epochs == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, prompt_simulator,
                train_config, device
            )

            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}")

            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_iou': train_metrics['iou'],
                    'train_dice': train_metrics['dice'],
                    'val_loss': val_loss,
                    'val_iou': val_metrics['iou'],
                    'val_dice': val_metrics['dice'],
                    'lr': optimizer.param_groups[0]['lr']
                })

            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_val_iou,
                    'config': model_config
                }, checkpoint_dir / 'best_model.pth')
                print(f"✓ Saved best model (IoU: {best_val_iou:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

        scheduler.step()

    print("\n" + "="*50)
    print(f"Training completed. Best Val IoU: {best_val_iou:.4f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMIS-Net')

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--inter_num', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()
    main(args)

