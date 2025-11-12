import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PromptEncoder(nn.Module):
    """
    Encodes prompts (points, boxes, masks) into embeddings
    for the mask decoder
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.prompt_embed_dim
        self.input_image_size = (config.img_size, config.img_size)

        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, self.embed_dim) for _ in range(2)
        ])

        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, config.mask_in_chans, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(config.mask_in_chans, self.embed_dim, kernel_size=1),
        )

        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)

    def _embed_points(self, points, labels, pad=False):
        """
        Embed point prompts
        Args:
            points: (B, N, 2) point coordinates (x, y) in [0, img_size]
            labels: (B, N) point labels (1=positive, 0=negative, -1=padding)
        Returns:
            point_embeddings: (B, N, embed_dim)
        """
        points = points + 0.5

        points = points / torch.tensor([self.input_image_size[1],
                                       self.input_image_size[0]],
                                      device=points.device)

        point_embedding = self.pe_layer(points)

        labels = labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = torch.where(
            labels == -1,
            self.not_a_point_embed.weight.unsqueeze(0),
            point_embedding
        )

        for i in range(len(self.point_embeddings)):
            mask = (labels == i).to(torch.float32)
            point_embedding = point_embedding + \
                            self.point_embeddings[i].weight * mask

        return point_embedding

    def _embed_boxes(self, boxes):
        """
        Embed box prompts
        Args:
            boxes: (B, 4) boxes in format (x1, y1, x2, y2)
        Returns:
            box_embeddings: (B, 2, embed_dim) corner embeddings
        """
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)

        coords = coords / torch.tensor([self.input_image_size[1],
                                       self.input_image_size[0]],
                                      device=boxes.device)

        corner_embedding = self.pe_layer(coords)

        corner_embedding[:, 0] += self.point_embeddings[0].weight
        corner_embedding[:, 1] += self.point_embeddings[1].weight

        return corner_embedding

    def _embed_masks(self, masks):
        """
        Embed mask prompts (from previous iteration)
        Args:
            masks: (B, 1, H, W) previous mask predictions
        Returns:
            mask_embeddings: (B, embed_dim, H', W') dense embeddings
        """
        return self.mask_downscaling(masks)

    def get_dense_pe(self):
        """
        Get dense positional encoding for the image
        Returns:
            pe: (1, embed_dim, H', W')
        """
        size = (self.input_image_size[0] // self.config.patch_size,
                self.input_image_size[1] // self.config.patch_size)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(size[0], dtype=torch.float32),
            torch.arange(size[1], dtype=torch.float32),
            indexing='ij'
        )

        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0)

        grid = grid / torch.tensor([size[1], size[0]])

        pe = self.pe_layer(grid)
        pe = pe.permute(0, 3, 1, 2)

        return pe

    def forward(self, points=None, boxes=None, masks=None):
        """
        Args:
            points: dict with 'coords' (B,N,2) and 'labels' (B,N)
            boxes: (B, 4) box coordinates
            masks: (B, 1, H, W) previous masks
        Returns:
            sparse_embeddings: (B, N, embed_dim) or None
            dense_embeddings: (B, embed_dim, H', W')
        """
        sparse_embeddings = []

        if points is not None:
            point_emb = self._embed_points(points['coords'], points['labels'])
            sparse_embeddings.append(point_emb)

        if boxes is not None:
            box_emb = self._embed_boxes(boxes)
            sparse_embeddings.append(box_emb)

        if len(sparse_embeddings) > 0:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = None

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            B = points['coords'].shape[0] if points is not None else boxes.shape[0]
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(
                B, -1,
                self.input_image_size[0] // self.config.patch_size // 4,
                self.input_image_size[1] // self.config.patch_size // 4
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies
    """
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        """
        Args:
            coords: (..., 2) coordinates in [0, 1]
        Returns:
            pe: (..., num_pos_feats*2)
        """
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords):
        """
        Args:
            coords: (B, ..., 2) normalized coordinates
        Returns:
            pe: (B, ..., num_pos_feats*2)
        """
        return self._pe_encoding(coords)

