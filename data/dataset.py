import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class MedicalSegmentationDataset(Dataset):
    """
    Dataset loader for medical image segmentation with support for
    multiple masks per image.
    """
    def __init__(self, data_path, transform=None, max_masks_per_image=1, mode='train'):
        """
        Args:
            data_path: Root directory containing images/ and masks/ folders
            transform: Augmentation pipeline
            max_masks_per_image: Maximum number of masks to load per image
            mode: One of 'train', 'val', 'test'
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.max_masks_per_image = max_masks_per_image
        self.mode = mode

        self.image_dir = self.data_path / 'images'
        self.mask_dir = self.data_path / 'masks'

        self.image_files = sorted(list(self.image_dir.glob('*.png')) +
                                 list(self.image_dir.glob('*.jpg')))

        self.mask_mapping = {}
        for img_file in self.image_files:
            img_name = img_file.stem
            mask_folder = self.mask_dir / img_name

            if mask_folder.exists():
                mask_files = sorted(list(mask_folder.glob('*.png')) +
                                  list(mask_folder.glob('*.jpg')))
                self.mask_mapping[img_name] = mask_files
            else:
                single_mask = self.mask_dir / f"{img_name}.png"
                if single_mask.exists():
                    self.mask_mapping[img_name] = [single_mask]

        self.image_files = [f for f in self.image_files
                           if f.stem in self.mask_mapping]

        print(f"[{mode}] Loaded {len(self.image_files)} images with masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_name = img_file.stem

        image = Image.open(img_file).convert('RGB')
        image = np.array(image)

        mask_files = self.mask_mapping[img_name]

        if len(mask_files) > self.max_masks_per_image:
            mask_files = np.random.choice(
                mask_files,
                self.max_masks_per_image,
                replace=False
            ).tolist()

        masks = []
        for mask_file in mask_files:
            mask = Image.open(mask_file).convert('L')
            mask = np.array(mask)

            if mask.max() > 1:
                mask = mask / 255.0

            if self.transform is not None:
                img_transformed, mask_transformed = self.transform(image.copy(), mask.copy())
            else:
                img_transformed = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask_transformed = torch.from_numpy(mask).unsqueeze(0).float()

            masks.append(mask_transformed)

        if self.transform is None:
            img_transformed = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            'image': img_transformed,
            'masks': masks,
            'image_name': img_name
        }


def collate_fn(batch):
    """Custom collation function for batching variable-length mask lists."""
    images = []
    all_masks = []
    names = []

    for item in batch:
        images.append(item['image'])
        all_masks.append(item['masks'][0])
        names.append(item['image_name'])

    images = torch.stack(images)
    masks = torch.stack(all_masks)

    return {
        'images': images,
        'masks': masks,
        'names': names
    }

