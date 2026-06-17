"""
dataset.py — SLiMIA segmentation dataset

Loads image-mask pairs from a pre-built metadata CSV.
Grayscale images are expanded to 3 channels so that
ImageNet-pretrained encoders can be used without modification.

CSV columns required:
    full_path  : absolute path to the .ome.tiff image
    mask_path  : absolute path to the binary segmentation mask (.tiff)

All other metadata columns (microscope, cell_line, etc.) are ignored here
and consumed by the downstream IPP modules.
"""

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset


class SLIMIADataset(Dataset):
    """
    Dataset for SLiMIA spheroid segmentation.

    Normalisation:
        Per-image percentile clipping (p1 / p99) handles the wide intensity
        variation across the nine microscope types in SLiMIA without requiring
        global statistics.

    Augmentation:
        Passed in via the `transform` argument (an albumentations Compose).
        No augmentation is applied at inference time — pass `transform=None`
        or a resize-only pipeline.
    """

    def __init__(self, csv_path: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = row["full_path"]
        mask_path = row["mask_path"]

        # Load images
        image = tifffile.imread(img_path).astype(np.float32)
        mask  = tifffile.imread(mask_path).astype(np.float32)

        # Multi-channel masks → take first channel
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Percentile normalisation — robust to outlier pixels
        p1, p99 = np.percentile(image, (1, 99))
        image   = np.clip(image, p1, p99)
        image   = (image - p1) / (p99 - p1 + 1e-8)
        image   = np.nan_to_num(image).astype(np.float32)

        # Binary mask
        mask = (mask > 0).astype(np.float32)

        # Grayscale → 3-channel (required by ImageNet-pretrained encoders)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask