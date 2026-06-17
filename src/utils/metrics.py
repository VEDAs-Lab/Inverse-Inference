"""
src/utils/metrics.py — All evaluation metrics used in the SLiMIA-IPP paper

Segmentation (Table 4):
    dice_score, iou_score, pixel_accuracy, compute_all_seg_metrics

IPP (Table 5 / 6):
    compute_ipp_metrics    — per-label and macro acc / prec / rec / F1
    exact_match_accuracy   — subset accuracy (all K labels must be correct)

Temporal (Table 7):
    batch_temporal_metrics — MSE, SSIM, PSNR per batch

Statistical (Section D):
    mean_std_over_seeds    — aggregate mean ± std over a list of metric dicts
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

try:
    from skimage.metrics import (
        structural_similarity as sk_ssim,
        peak_signal_noise_ratio as sk_psnr,
        mean_squared_error as sk_mse,
    )
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


# Segmentation 

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Batch-averaged Dice. Accepts raw logits — sigmoid applied internally."""
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Batch-averaged Intersection over Union."""
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()
    return (pred == target).float().mean()


def compute_all_seg_metrics(pred: torch.Tensor, target: torch.Tensor,
                             threshold: float = 0.5,
                             eps: float = 1e-6) -> dict:
    """
    Full confusion-matrix segmentation metrics for one batch.
    Excludes edge cases where both pred and GT are empty (Dice=1, IoU=0),
    consistent with Table 4 reporting.

    Returns:
        dict with keys: dice, iou, accuracy, precision, recall, f1
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


# IPP 

def compute_ipp_metrics(targets: dict, preds: dict,
                        label_columns: list) -> dict:
    """
    Per-label and macro-averaged IPP metrics.

    Args:
        targets: dict {label: list of int ground-truth indices}
        preds:   dict {label: list of int predicted indices}
        label_columns: ordered list of label names

    Returns:
        {
          'per_label': {label: {acc, prec, rec, f1}},
          'macro':     {acc, prec, rec, f1}
        }
    """
    per_label = {}
    for col in label_columns:
        t = np.array(targets[col])
        p = np.array(preds[col])
        per_label[col] = dict(
            acc  = float(accuracy_score(t, p)),
            prec = float(precision_score(t, p, average="macro", zero_division=0)),
            rec  = float(recall_score(t, p, average="macro", zero_division=0)),
            f1   = float(f1_score(t, p, average="macro", zero_division=0)),
        )
    macro = {k: float(np.mean([per_label[c][k] for c in label_columns]))
             for k in ["acc", "prec", "rec", "f1"]}
    return {"per_label": per_label, "macro": macro}


def exact_match_accuracy(targets: dict, preds: dict,
                          label_columns: list) -> float:
    """
    Exact-match (subset) accuracy — an image is correct only if ALL K
    predicted attributes match ground truth simultaneously.
    Reported separately from mean per-attribute accuracy in the paper.
    """
    n = len(next(iter(targets.values())))
    correct = np.ones(n, dtype=bool)
    for col in label_columns:
        correct &= (np.array(targets[col]) == np.array(preds[col]))
    return float(correct.mean())


# Temporal 

def batch_temporal_metrics(pred: torch.Tensor,
                            target: torch.Tensor) -> tuple:
    """
    Compute MSE, SSIM, PSNR for a batch of predicted frames.

    Args:
        pred:   (B, 1, H, W) float tensor in [0, 1]
        target: (B, 1, H, W) float tensor in [0, 1]

    Returns:
        (mean_mse, mean_ssim, mean_psnr) as Python floats
    """
    if not HAVE_SKIMAGE:
        mse = F.mse_loss(pred, target).item()
        return mse, 0.0, 0.0

    p_np = pred.detach().cpu().numpy()
    t_np = target.detach().cpu().numpy()
    mse_v, ssim_v, psnr_v = [], [], []

    for i in range(p_np.shape[0]):
        pi = np.clip(p_np[i, 0], 0, 1)
        ti = t_np[i, 0]
        mse_v.append(float(sk_mse(ti, pi)))
        try:
            ssim_v.append(float(sk_ssim(ti, pi, data_range=1.0)))
        except Exception:
            ssim_v.append(0.0)
        try:
            psnr_v.append(float(sk_psnr(ti, pi, data_range=1.0)))
        except Exception:
            psnr_v.append(20.0)

    return float(np.mean(mse_v)), float(np.mean(ssim_v)), float(np.mean(psnr_v))


# Statistical aggregation 

def mean_std_over_seeds(seed_metrics: list) -> dict:
    """
    Aggregate a list of per-seed metric dicts into mean ± std.

    Args:
        seed_metrics: list of dicts, one per seed, each {metric: float}

    Returns:
        dict {metric: (mean, std)}

    Example:
        results = [{"f1": 0.94}, {"f1": 0.96}, {"f1": 0.95}]
        mean_std_over_seeds(results)
        # → {"f1": (0.9500, 0.0082)}
    """
    keys = seed_metrics[0].keys()
    return {
        k: (float(np.mean([m[k] for m in seed_metrics])),
            float(np.std( [m[k] for m in seed_metrics])))
        for k in keys
    }


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format (mean, std) as 'X.XXXX ± Y.YYYYY' — matches paper table style."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"