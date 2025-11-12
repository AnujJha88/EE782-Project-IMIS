import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def train_one_epoch(model, dataloader, criterion, optimizer, prompt_simulator,
                   config, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        gt_masks = batch['masks'].to(device)

        B = images.shape[0]

        # Simulate interactive prompts
        prompts_list = []
        for i in range(B):
            mask = gt_masks[i, 0]  # (H, W)
            prompts = prompt_simulator.generate_prompts(mask)

            # Move to device
            if 'points' in prompts:
                prompts['points']['coords'] = prompts['points']['coords'].to(device)
                prompts['points']['labels'] = prompts['points']['labels'].to(device)
            if 'boxes' in prompts:
                prompts['boxes'] = prompts['boxes'].to(device)

            prompts_list.append(prompts)

        # Batch prompts
        batched_prompts = {}
        if 'points' in prompts_list[0]:
            batched_prompts['points'] = {
                'coords': torch.cat([p['points']['coords'] for p in prompts_list]),
                'labels': torch.cat([p['points']['labels'] for p in prompts_list])
            }
        if 'boxes' in prompts_list[0]:
            batched_prompts['boxes'] = torch.stack([p['boxes'] for p in prompts_list])

        # Forward pass with iterative refinement
        masks_per_iter, iou_per_iter = model(
            images, batched_prompts, iterative=True
        )

        # Compute loss on final iteration
        final_masks = masks_per_iter[-1]  # (B, num_mask_tokens, H, W)
        final_iou = iou_per_iter[-1]      # (B, num_mask_tokens)

        # Resize predictions to match GT
        final_masks_resized = torch.nn.functional.interpolate(
            final_masks,
            size=gt_masks.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Compute loss
        loss, loss_dict = criterion(final_masks_resized, gt_masks)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(final_masks_resized, gt_masks)
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{metrics['iou']:.4f}",
            'dice': f"{metrics['dice']:.4f}"
        })

    # Average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, prompt_simulator, config, device):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].to(device)
        gt_masks = batch['masks'].to(device)

        B = images.shape[0]

        # Simulate prompts
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

        # Batch prompts
        batched_prompts = {}
        if 'points' in prompts_list[0]:
            batched_prompts['points'] = {
                'coords': torch.cat([p['points']['coords'] for p in prompts_list]),
                'labels': torch.cat([p['points']['labels'] for p in prompts_list])
            }
        if 'boxes' in prompts_list[0]:
            batched_prompts['boxes'] = torch.stack([p['boxes'] for p in prompts_list])

        # Forward
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
    # Initialize configs
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

    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="imis-net", config={
            **vars(args),
            'model_config': vars(model_config),
            'train_config': vars(train_config)
        })

    # Create datasets
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

    # Initialize model
    model = IMISNet(model_config).to(device)

    # Load pretrained weights if available
    if args.pretrained_path:
        print(f"Loading pretrained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Loss and optimizer
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

    # Prompt simulator
    prompt_simulator = PromptSimulator(
        num_points=train_config.num_points_train,
        point_prob=train_config.point_prob,
        box_prob=train_config.box_prob
    )

    # Training loop
    best_val_iou = 0.0
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{train_config.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            prompt_simulator, train_config, device, epoch
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, prompt_simulator,
            train_config, device
        )

        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}")

        # Log to wandb
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

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'config': model_config
            }, checkpoint_dir / 'best_model.pth')
            print(f"✓ Saved best model with IoU: {best_val_iou:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['iou'],
                'config': model_config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

        scheduler.step()

    print("\n" + "="*50)
    print(f"Training completed! Best Val IoU: {best_val_iou:.4f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMIS-Net')

    # Data args
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='Input image size')

    # Model args
    parser.add_argument('--inter_num', type=int, default=3,
                       help='Number of iterative refinements')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained weights')

    # Training args
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')

    args = parser.parse_args()

    main(args)

