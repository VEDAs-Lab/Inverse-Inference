"""
src/utils/losses.py — All loss functions used across the SLiMIA-IPP pipeline

Segmentation losses (Table 1):
    FocalTverskyLoss   — UNet++, DeepLabV3+, RefineNet, SegNet
    BCEDiceLoss        — DeepLabV3, AttentionUNet
    nn.BCEWithLogitsLoss — SwinUNet, TransUNet (standard PyTorch)

IPP losses (Table 2):
    nn.CrossEntropyLoss         — ConvNeXt, ViT, CoAtNet (standard PyTorch)
    nn.CrossEntropyLoss(weight) — ImageShapeFusion, HMTT (class-weighted)
    FocalLoss                   — HMTT (combined with weighted CE)

Temporal losses (Table 3):
    nn.L1Loss                   — ConvLSTM, PredRNN++, PhyDNet (standard PyTorch)
    MSESSIMLoss                 — MetadataFusion (0.8·MSE + 0.2·(1−SSIM))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Segmentation 

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss (Salehi et al. 2017).

    α=0.7, β=0.3 weights FN more than FP — suitable for sparse spheroid
    foreground where missing the spheroid region is costlier than a false alarm.
    γ=0.75 applies focal modulation to focus training on hard examples.

    Used by: UNet++, DeepLabV3+, RefineNet, SegNet
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        preds  = torch.sigmoid(preds)
        smooth = 1e-6
        TP = (preds * targets).sum(dim=(1, 2, 3))
        FP = ((1 - targets) * preds).sum(dim=(1, 2, 3))
        FN = (targets * (1 - preds)).sum(dim=(1, 2, 3))
        tv = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return ((1 - tv) ** self.gamma).mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice combined loss, equal weights (0.5 each).
    Dice smooth = 1e-6 (Milletari et al. 2016 V-Net convention).

    Used by: DeepLabV3, AttentionUNet
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce    = self.bce(preds, targets)
        p      = torch.sigmoid(preds)
        smooth = 1e-6
        inter  = (p * targets).sum(dim=(1, 2, 3))
        union  = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice   = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
        return 0.5 * bce + 0.5 * dice


# IPP 

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017).  γ=2.0, α=0.25 (paper defaults).
    Used by HMTT alongside class-weighted cross-entropy.

    Args:
        gamma: focusing parameter (higher → more focus on hard examples)
        alpha: per-class weight tensor (shape [C]) or scalar, or None
    """
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, target, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            a    = (self.alpha[target]
                    if isinstance(self.alpha, torch.Tensor) and self.alpha.ndim == 1
                    else self.alpha)
            loss = a * loss
        return loss.mean()


def build_class_weights(train_df, label_columns: list,
                        label_dims: dict, device: str) -> dict:
    """
    Compute balanced class weights for each label from the training split.

    Returns:
        dict mapping label_column → torch.Tensor of shape [n_classes]
    """
    weights = {}
    n = len(train_df)
    for col in label_columns:
        y      = train_df[col + "_enc"].values
        nc     = label_dims[col]
        counts = np.bincount(y, minlength=nc).astype(np.float32)
        w      = np.where(counts > 0,
                          n / (nc * np.maximum(counts, 1.0)),
                          0.0).astype(np.float32)
        weights[col] = torch.tensor(w, device=device)
    return weights


# Temporal 

class MSESSIMLoss(nn.Module):
    """
    0.8·MSE + 0.2·(1−SSIM) combined loss.
    Used by MetadataFusion temporal model.
    Falls back to 0.6·MSE + 0.4·L1 if pytorch-msssim is not installed.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        try:
            from pytorch_msssim import SSIM
            self._ssim = SSIM(data_range=1.0, channel=1)
            self._use_ssim = True
        except ImportError:
            self._l1 = nn.L1Loss()
            self._use_ssim = False

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._use_ssim:
            self._ssim = self._ssim.to(pred.device)
            return 0.8 * self.mse(pred, target) + \
                   0.2 * (1 - self._ssim(pred, target))
        return 0.6 * self.mse(pred, target) + 0.4 * self._l1(pred, target)