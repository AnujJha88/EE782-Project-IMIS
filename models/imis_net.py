import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder

class IMISNet(nn.Module):
    """
    Interactive Medical Image Segmentation Network
    Implements iterative refinement for interactive segmentation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(config)
        self.prompt_encoder = PromptEncoder(config)
        self.mask_decoder = MaskDecoder(config)

        self.mask_threshold = 0.0

    def forward(self, images, prompts, iterative=True):
        """
        Args:
            images: (B, 3, H, W) input images
            prompts: dict containing:
                - 'points': dict with 'coords' and 'labels'
                - 'boxes': (B, 4) optional
                - 'prev_masks': (B, 1, H, W) optional
            iterative: whether to use iterative refinement
        Returns:
            masks: list of (B, num_mask_tokens, H, W) per iteration
            iou_predictions: list of (B, num_mask_tokens) per iteration
        """
        image_embeddings = self.image_encoder(images)

        image_pe = self.prompt_encoder.get_dense_pe().to(images.device)

        if iterative:
            masks, iou_preds = self.mask_decoder.predict_masks_iterative(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                prompt_encoder=self.prompt_encoder,
                points=prompts.get('points'),
                boxes=prompts.get('boxes'),
                prev_masks=prompts.get('prev_masks')
            )
        else:
            sparse_emb, dense_emb = self.prompt_encoder(
                points=prompts.get('points'),
                boxes=prompts.get('boxes'),
                masks=prompts.get('prev_masks')
            )

            masks, iou_preds = self.mask_decoder(
                image_embeddings, image_pe, sparse_emb, dense_emb
            )

            masks = [masks]
            iou_preds = [iou_preds]

        return masks, iou_preds

    def postprocess_masks(self, masks, original_size):
        """
        Upscale masks to original image size
        """
        masks = F.interpolate(
            masks,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        return masks

