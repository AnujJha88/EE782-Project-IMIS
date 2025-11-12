import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """IMIS-Net Model Configuration"""
    img_size: int = 512
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    out_chans: int = 128

    prompt_embed_dim: int = 128
    mask_in_chans: int = 8

    transformer_dim: int = 128
    decoder_depth: int = 1
    decoder_num_heads: int = 4
    mlp_dim: int = 512
    inter_num: int = 1

    num_mask_tokens: int = 1
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 128

    use_gradient_checkpointing: bool = True

@dataclass
class TrainingConfig:
    """Training Configuration"""
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3

    bce_weight: float = 2.0
    dice_weight: float = 0.5
    focal_weight: float = 0.0

    num_workers: int = 2
    mask_num_train: int = 1

    num_points_train: int = 3
    point_prob: float = 0.7
    box_prob: float = 0.3

    data_path: str = "dataset/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"

    use_amp: bool = True
    validate_every_n_epochs: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

