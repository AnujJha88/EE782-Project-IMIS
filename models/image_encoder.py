import torch
import torch.nn as nn
import timm
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

class ImageEncoder(nn.Module):
    """
    Vision Transformer-based Image Encoder
    Extracts dense feature representations from input images
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.embed_dim)
        )

        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        self.neck = nn.Sequential(
            nn.Conv2d(config.embed_dim, config.out_chans,
                     kernel_size=1, bias=False),
            nn.LayerNorm(config.out_chans),
            nn.Conv2d(config.out_chans, config.out_chans,
                     kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(config.out_chans)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: (B, out_chans, H/patch_size, W/patch_size)
        """
        B, C, H, W = x.shape

        x = self.patch_embed.proj(x)
        _, _, H_p, W_p = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed

        for blk in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_p, w=W_p)
        x = self.neck(x)

        return x

