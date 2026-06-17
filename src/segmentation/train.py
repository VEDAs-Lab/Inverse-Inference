"""
train.py — Segmentation training script

Trains any of the 8 SLiMIA segmentation models with:
  - 3-seed averaging (seeds 42, 123, 999)
  - ReduceLROnPlateau scheduling
  - Early stopping on val Dice
  - AMP mixed-precision
  - Checkpoint saving (best val Dice)

Usage (from repo root):
    python src/segmentation/train.py --model refinenet --csv data/slimia_metadata.csv
    python src/segmentation/train.py --model unetpp    --csv data/slimia_metadata.csv
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import SLIMIADataset
from evaluate import dice_score, iou_score
from models import build_model


# Losses

class FocalTverskyLoss(nn.Module):
    """α=0.7, β=0.3, γ=0.75 — used by UNet++, DeepLabV3+, RefineNet, SegNet."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, preds, targets):
        preds  = torch.sigmoid(preds)
        smooth = 1e-6
        TP = (preds * targets).sum(dim=(1, 2, 3))
        FP = ((1 - targets) * preds).sum(dim=(1, 2, 3))
        FN = (targets * (1 - preds)).sum(dim=(1, 2, 3))
        tv = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return ((1 - tv) ** self.gamma).mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice (equal weights) — used by DeepLabV3, AttentionUNet."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce   = self.bce(preds, targets)
        p     = torch.sigmoid(preds)
        smooth = 1e-6
        inter = (p * targets).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice  = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
        return 0.5 * bce + 0.5 * dice


# Map model name → loss (matches Table 1 in the paper)
LOSS_MAP = {
    "unetpp":        FocalTverskyLoss(),
    "deeplabv3":     BCEDiceLoss(),
    "deeplabv3plus": FocalTverskyLoss(),
    "attention_unet":BCEDiceLoss(),
    "segnet":        FocalTverskyLoss(),
    "refinenet":     FocalTverskyLoss(),
    "swin_unet":     nn.BCEWithLogitsLoss(),
    "transunet":     nn.BCEWithLogitsLoss(),
}


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# Augmentations

def get_transforms(img_size: int):
    train = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])
    val = A.Compose([A.Resize(img_size, img_size), ToTensorV2()])
    return train, val


# Training

def train_one_seed(model_name, csv_path, ckpt_dir, seed,
                   num_epochs=200, batch_size=8, lr=1e-4, patience=40):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model, img_size = build_model(model_name)
    model           = model.to(device)
    loss_fn         = LOSS_MAP[model_name]
    optimizer       = optim.Adam(model.parameters(), lr=lr)
    scheduler       = ReduceLROnPlateau(optimizer, mode="max",
                                        patience=5, factor=0.5)
    scaler          = GradScaler(enabled=device.type == "cuda")

    train_tfm, val_tfm = get_transforms(img_size)
    full_ds = SLIMIADataset(csv_path, transform=train_tfm)
    val_ds  = SLIMIADataset(csv_path, transform=val_tfm)
    indices = list(range(len(full_ds)))
    t_idx, v_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    train_loader = DataLoader(Subset(full_ds, t_idx), batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,  v_idx), batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    save_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}.pth")
    best_dice = 0.0
    patience_count = 0
    history = dict(train_loss=[], val_loss=[], val_dice=[], val_iou=[])

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        t_loss = 0.0
        for imgs, masks in tqdm(train_loader,
                                desc=f"[{model_name}|seed{seed}] Epoch {epoch+1:03d}",
                                leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(imgs)
                loss  = loss_fn(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
        history["train_loss"].append(t_loss / len(train_loader))

        # ── Validate ──
        model.eval()
        v_loss = v_dice = v_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds  = model(imgs)
                v_loss += loss_fn(preds, masks).item()
                v_dice += dice_score(preds, masks).item()
                v_iou  += iou_score(preds, masks).item()
        n = len(val_loader)
        history["val_loss"].append(v_loss / n)
        history["val_dice"].append(v_dice / n)
        history["val_iou"].append(v_iou  / n)

        scheduler.step(history["val_dice"][-1])
        print(f"  Epoch {epoch+1:03d} | "
              f"train {history['train_loss'][-1]:.4f} | "
              f"val {history['val_loss'][-1]:.4f} | "
              f"dice {history['val_dice'][-1]:.4f} | "
              f"iou  {history['val_iou'][-1]:.4f}")

        if history["val_dice"][-1] > best_dice:
            best_dice = history["val_dice"][-1]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best (Dice {best_dice:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    return history, save_path


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Train SLiMIA segmentation model")
    parser.add_argument("--model",  required=True,
                        choices=["unetpp", "deeplabv3", "deeplabv3plus",
                                 "attention_unet", "segnet", "refinenet",
                                 "swin_unet", "transunet"])
    parser.add_argument("--csv",    default="data/slimia_metadata.csv")
    parser.add_argument("--ckpt",   default="checkpoints/segmentation/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--seeds",  nargs="+", type=int, default=[42, 123, 999])
    args = parser.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)

    all_seed_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}\n  {args.model.upper()} — Seed {seed}\n{'='*60}")
        history, ckpt_path = train_one_seed(
            model_name=args.model,
            csv_path=args.csv,
            ckpt_dir=args.ckpt,
            seed=seed,
            num_epochs=args.epochs,
            batch_size=args.batch,
        )
        all_seed_results.append({
            "seed": seed,
            "best_dice": max(history["val_dice"]),
            "best_iou":  max(history["val_iou"]),
            "ckpt":      ckpt_path,
        })

    print("\n──── Summary ────")
    for r in all_seed_results:
        print(f"  Seed {r['seed']}: "
              f"best Dice {r['best_dice']:.4f} | "
              f"best IoU  {r['best_iou']:.4f}")

    dices = [r["best_dice"] for r in all_seed_results]
    ious  = [r["best_iou"]  for r in all_seed_results]
    print(f"\n  Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.5f}")
    print(f"  Mean IoU:  {np.mean(ious):.4f} ± {np.std(ious):.5f}")


if __name__ == "__main__":
    main()