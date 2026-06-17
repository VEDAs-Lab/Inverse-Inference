"""
evaluate.py — Segmentation evaluation utilities

Provides:
  - dice_score / iou_score (used during training)
  - compute_all_metrics   (full confusion-matrix stats)
  - evaluate_checkpoint   (load a saved model and score the full val set)

Usage:
    python src/segmentation/evaluate.py \
        --model refinenet \
        --ckpt  checkpoints/segmentation/refinenet_seed42.pth \
        --csv   data/slimia_metadata.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import SLIMIADataset
from models  import build_model

# Metric functions

def dice_score(pred, target, threshold=0.5, eps=1e-6):
    """
    Differentiable Dice computed over a batch.
    Safe to call with raw logits — applies sigmoid internally.
    """
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean()


def compute_all_metrics(pred, target, threshold=0.5, eps=1e-6) -> dict:
    """
    Full confusion-matrix metrics for a single batch.
    Returns dict: dice, iou, accuracy, precision, recall, f1.

    Note: Dice / IoU denominators differ slightly here (scalar sums rather than
    batched mean) — consistent with how Table 4 in the paper is computed.
    """
    probs = torch.sigmoid(pred)
    p     = (probs > threshold).float()
    t     = target.float()

    TP = (p * t).sum().item()
    TN = ((1 - p) * (1 - t)).sum().item()
    FP = (p * (1 - t)).sum().item()
    FN = ((1 - p) * t).sum().item()

    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * TP / (2 * TP + FP + FN + eps)
    dice      = dice_score(pred, target, threshold, eps).item()
    iou       = iou_score(pred, target, threshold, eps).item()

    return dict(dice=dice, iou=iou, accuracy=accuracy,
                precision=precision, recall=recall, f1=f1)


# Full dataset evaluation
def evaluate_checkpoint(model_name, ckpt_path, csv_path,
                        seed=42, batch_size=8) -> pd.DataFrame:
    """
    Load a saved checkpoint and compute per-batch metrics on the val split.
    Returns a DataFrame with one row per batch and summary statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_size = build_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    val_tfm = A.Compose([A.Resize(img_size, img_size), ToTensorV2()])
    full_ds  = SLIMIADataset(csv_path, transform=val_tfm)
    indices  = list(range(len(full_ds)))
    _, v_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    loader   = DataLoader(Subset(full_ds, v_idx), batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    rows = []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Evaluating {model_name}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            rows.append(compute_all_metrics(preds, masks))

    df = pd.DataFrame(rows)
    print(f"\n{'─'*50}")
    print(f"Model: {model_name}  |  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"{'─'*50}")
    print(df.mean().to_string())
    print(f"{'─'*50}\n")
    return df


# Per-image inference (used to generate masks for morphometry extraction)

def run_inference(model_name, ckpt_path, csv_path,
                  output_dir, img_size=256):
    """
    Runs inference over the entire dataset and saves predicted binary masks.
    These predicted masks (not ground-truth) feed into IPP training.

    Args:
        model_name  : e.g. "refinenet"
        ckpt_path   : path to saved .pth checkpoint
        csv_path    : metadata CSV
        output_dir  : directory where predicted .tiff masks are saved
        img_size    : resize resolution (256 for CNN models, 224 for transformers)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    model, _ = build_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    tfm = A.Compose([A.Resize(img_size, img_size), ToTensorV2()])
    ds  = SLIMIADataset(csv_path, transform=tfm)

    summary = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Inference"):
            img, mask = ds[i]
            img_path  = ds.df.iloc[i]["full_path"]

            pred      = model(img.unsqueeze(0).to(device))
            pred_bin  = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            fname     = os.path.splitext(os.path.basename(img_path))[0] + "_pred.tiff"
            pred_path = os.path.join(output_dir, fname)
            tifffile.imwrite(pred_path, pred_bin)

            m           = compute_all_metrics(pred, mask.unsqueeze(0).to(device))
            m["img_path"]  = img_path
            m["pred_path"] = pred_path
            summary.append(m)

    out_df = pd.DataFrame(summary)
    out_df.to_csv(os.path.join(output_dir, "inference_summary.csv"), index=False)
    print(f"\nInference complete.")
    print(f"Mean Dice : {out_df['dice'].mean():.4f}")
    print(f"Mean IoU  : {out_df['iou'].mean():.4f}")
    print(f"Saved to  : {output_dir}")
    return out_df


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument("--model",  required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--csv",    default="data/slimia_metadata.csv")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--mode",   choices=["eval", "infer"], default="eval")
    parser.add_argument("--outdir", default="data/predicted_masks/")
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_checkpoint(args.model, args.ckpt, args.csv, seed=args.seed)
    else:
        _, img_size = build_model(args.model)   # get img_size from registry
        run_inference(args.model, args.ckpt, args.csv, args.outdir, img_size)


if __name__ == "__main__":
    main()