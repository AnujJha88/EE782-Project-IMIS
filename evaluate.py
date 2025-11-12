import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import ModelConfig, TrainingConfig
from models.imis_net import IMISNet
from data.dataset import MedicalSegmentationDataset, collate_fn
from data.transforms import MedicalImageTransform
from utils.metrics import compute_metrics
from utils.prompt_simulation import PromptSimulator


@torch.no_grad()
def evaluate(model, dataloader, prompt_simulator, device, use_amp=True,
             save_vis=False, output_dir=None):
    """Evaluate model on test set"""
    model.eval()

    all_metrics = []

    if save_vis and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch['images'].to(device)
        gt_masks = batch['masks'].to(device)
        names = batch['names']

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

        with autocast(device_type='cuda', enabled=use_amp):
            masks_per_iter, iou_per_iter = model(images, batched_prompts, iterative=True)

        for iter_idx, masks in enumerate(masks_per_iter):
            masks_resized = F.interpolate(
                masks,
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            metrics = compute_metrics(masks_resized, gt_masks)
            metrics['iteration'] = iter_idx
            all_metrics.append(metrics)

        if save_vis and batch_idx < 10:
            final_masks = masks_per_iter[-1]
            final_masks_resized = F.interpolate(
                final_masks,
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            for i in range(B):
                visualize_prediction(
                    images[i],
                    gt_masks[i],
                    final_masks_resized[i],
                    prompts_list[i],
                    names[i],
                    output_dir
                )

    num_iters = len(masks_per_iter)
    iter_metrics = {i: {'iou': [], 'dice': [], 'accuracy': []}
                   for i in range(num_iters)}

    for m in all_metrics:
        iter_idx = m['iteration']
        iter_metrics[iter_idx]['iou'].append(m['iou'])
        iter_metrics[iter_idx]['dice'].append(m['dice'])
        iter_metrics[iter_idx]['accuracy'].append(m['accuracy'])

    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    for i in range(num_iters):
        avg_iou = np.mean(iter_metrics[i]['iou'])
        avg_dice = np.mean(iter_metrics[i]['dice'])
        avg_acc = np.mean(iter_metrics[i]['accuracy'])

        print(f"\nIteration {i+1}:")
        print(f"  IoU:      {avg_iou:.4f}")
        print(f"  Dice:     {avg_dice:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")

    return iter_metrics


def visualize_prediction(image, gt_mask, pred_mask, prompts, name, output_dir):
    """Visualize and save prediction"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    gt_mask = gt_mask[0].cpu().numpy()

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = pred_mask.max(dim=0)[0].cpu().numpy()
    pred_mask_binary = (pred_mask > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(pred_mask_binary, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    axes[3].imshow(image)
    axes[3].imshow(pred_mask_binary, alpha=0.3, cmap='jet')

    if 'points' in prompts:
        coords = prompts['points']['coords'][0].cpu().numpy()
        labels = prompts['points']['labels'][0].cpu().numpy()

        for coord, label in zip(coords, labels):
            color = 'green' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[3].plot(coord[0], coord[1], marker,
                        color=color, markersize=10, markeredgewidth=2)

    if 'boxes' in prompts:
        box = prompts['boxes'][0].cpu().numpy()
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor='yellow', linewidth=2)
        axes[3].add_patch(rect)

    axes[3].set_title('Prediction + Prompts')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = checkpoint['config']

    model = IMISNet(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation IoU: {checkpoint['val_iou']:.4f}")

    transform = MedicalImageTransform(img_size=model_config.img_size, is_train=False)

    test_dataset = MedicalSegmentationDataset(
        data_path=Path(args.data_path) / 'test',
        transform=transform,
        max_masks_per_image=1,
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    prompt_simulator = PromptSimulator(
        num_points=3,
        point_prob=0.7,
        box_prob=0.3
    )

    metrics = evaluate(
        model,
        test_loader,
        prompt_simulator,
        device,
        use_amp=True,
        save_vis=args.save_visualizations,
        output_dir=args.output_dir
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate IMIS-Net')

    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()
    main(args)

