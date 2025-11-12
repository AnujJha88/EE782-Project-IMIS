import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Multi-component loss function combining binary cross-entropy and Dice loss.
    Optionally includes focal loss for handling class imbalance.
    """
    def __init__(self, bce_weight=2.0, dice_weight=0.5, focal_weight=0.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred_masks, gt_masks, iou_predictions=None, gt_ious=None):
        """
        Compute combined loss.

        Args:
            pred_masks: (B, N, H, W) predicted logits
            gt_masks: (B, 1, H, W) ground truth binary masks
            iou_predictions: (B, N) predicted IoU scores (optional)
            gt_ious: (B, N) ground truth IoU scores (optional)

        Returns:
            total_loss: Weighted combination of loss components
            loss_dict: Dictionary of individual loss values
        """
        gt_masks_expanded = gt_masks.expand_as(pred_masks)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, gt_masks_expanded, reduction='mean'
        )

        dice_loss = self.dice_loss(pred_masks, gt_masks_expanded)

        focal = 0.0
        if self.focal_weight > 0:
            focal = self.focal_loss(pred_masks, gt_masks_expanded)

        iou_loss = 0.0
        if iou_predictions is not None and gt_ious is not None:
            iou_loss = F.mse_loss(iou_predictions, gt_ious)

        total_loss = (self.bce_weight * bce_loss +
                     self.dice_weight * dice_loss +
                     self.focal_weight * focal +
                     iou_loss)

        return total_loss, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal if isinstance(focal, float) else focal.item(),
            'iou': iou_loss.item() if isinstance(iou_loss, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }

    def dice_loss(self, pred, target, smooth=1.0):
        """Sørensen–Dice coefficient loss."""
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def focal_loss(self, pred, target):
        """Focal loss for addressing class imbalance."""
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)

        focal = alpha_t * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal.mean()

