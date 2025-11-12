import torch
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MedicalImageTransform:
    """
    Augmentation pipeline for medical image segmentation.
    Supports synchronized image-mask transformations.
    """
    def __init__(self, img_size=512, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def __call__(self, image, mask):
        """
        Apply transformations to image and mask.

        Args:
            image: (H, W, 3) numpy array
            mask: (H, W) numpy array

        Returns:
            image: (3, H', W') tensor
            mask: (1, H', W') tensor
        """
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = (mask > 0.5).float()

        return image, mask

