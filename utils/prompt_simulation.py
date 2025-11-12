import torch
import numpy as np

class PromptSimulator:
    """
    Simulates user interactions for interactive segmentation training.
    Generates point clicks and bounding boxes based on ground truth masks.
    """
    def __init__(self, num_points=3, point_prob=0.7, box_prob=0.3):
        self.num_points = num_points
        self.point_prob = point_prob
        self.box_prob = box_prob

    def simulate_clicks(self, mask, prev_pred=None, positive_ratio=0.7):
        """
        Generate simulated user clicks on mask regions.

        Args:
            mask: Ground truth binary mask
            prev_pred: Previous iteration prediction for error-guided sampling
            positive_ratio: Proportion of positive vs negative clicks

        Returns:
            points: (N, 2) coordinates in (x, y) format
            labels: (N,) binary labels (1=foreground, 0=background)
        """
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

        if prev_pred is not None:
            prev_pred_np = prev_pred.cpu().numpy() if torch.is_tensor(prev_pred) else prev_pred
            prev_pred_binary = (prev_pred_np > 0.5).astype(np.float32)

            fn_mask = (mask_np == 1) & (prev_pred_binary == 0)
            fp_mask = (mask_np == 0) & (prev_pred_binary == 1)
        else:
            fn_mask = mask_np
            fp_mask = 1 - mask_np

        points = []
        labels = []

        num_positive = int(self.num_points * positive_ratio)
        num_negative = self.num_points - num_positive

        positive_coords = self._sample_from_mask(fn_mask, num_positive)
        points.extend(positive_coords)
        labels.extend([1] * len(positive_coords))

        negative_coords = self._sample_from_mask(fp_mask, num_negative)
        points.extend(negative_coords)
        labels.extend([0] * len(negative_coords))

        if len(points) == 0:
            h, w = mask_np.shape
            points = [[w // 2, h // 2]]
            labels = [1 if mask_np[h // 2, w // 2] > 0.5 else 0]

        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return points, labels

    def _sample_from_mask(self, mask, num_samples):
        """Sample random points from binary mask region."""
        coords = np.argwhere(mask > 0.5)

        if len(coords) == 0:
            return []

        if len(coords) < num_samples:
            indices = np.arange(len(coords))
        else:
            indices = np.random.choice(len(coords), num_samples, replace=False)

        sampled = coords[indices]
        return [[int(x), int(y)] for y, x in sampled]

    def get_bbox_from_mask(self, mask):
        """
        Extract axis-aligned bounding box from mask.

        Returns:
            bbox: (4,) tensor [x1, y1, x2, y2]
        """
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        coords = np.argwhere(mask_np > 0.5)

        if len(coords) == 0:
            return torch.tensor([0, 0, 1, 1], dtype=torch.float32)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(mask_np.shape[1] - 1, x_max + margin)
        y_max = min(mask_np.shape[0] - 1, y_max + margin)

        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        return bbox

    def add_jitter_to_bbox(self, bbox, jitter=10):
        """Add Gaussian noise to bbox coordinates for augmentation."""
        bbox = bbox.clone()
        noise = torch.randn(4) * jitter
        bbox = bbox + noise
        bbox[2:] = torch.max(bbox[2:], bbox[:2] + 1)
        return bbox

    def generate_prompts(self, mask, prev_pred=None):
        """
        Generate random combination of prompts for training.

        Returns:
            Dictionary containing 'points' and/or 'boxes' keys
        """
        prompts = {}

        use_points = np.random.random() < self.point_prob
        use_boxes = np.random.random() < self.box_prob

        if not use_points and not use_boxes:
            use_points = True

        if use_points:
            points, labels = self.simulate_clicks(mask, prev_pred)
            prompts['points'] = {
                'coords': points.unsqueeze(0),
                'labels': labels.unsqueeze(0)
            }

        if use_boxes:
            bbox = self.get_bbox_from_mask(mask)
            bbox = self.add_jitter_to_bbox(bbox, jitter=5)
            prompts['boxes'] = bbox.unsqueeze(0)

        return prompts

