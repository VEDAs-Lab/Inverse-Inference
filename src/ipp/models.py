"""
src/ipp/models.py — All 5 IPP model architectures (Table 2 in paper)

    Model                    Backbone              Key Design
    MultiTaskBackbone        timm backbone         Generic multi-head wrapper
      └─ ConvNeXt-Tiny       convnext_tiny         Convolutional locality priors
      └─ ViT-B/16            vit_base_patch16_224  Global self-attention via CLS
      └─ CoAtNet-0           coatnet_0_224         Hybrid convolution + attention
    ImageShapeFusionTransformer  ConvNeXt + shape  Explicit morphometric priors
    HMTT                     ViT-B/16 encoder      Causal label conditioning

Usage:
    from src.ipp.models import MultiTaskBackbone, ImageShapeFusionTransformer, HMTT

    # ConvNeXt
    model = MultiTaskBackbone("convnext_tiny", label_dims, pretrained=True)

    # CoAtNet (trained from scratch)
    model = MultiTaskBackbone("coatnet_0_224", label_dims, pretrained=False)

    # Fusion (needs shape stats from training set)
    model = ImageShapeFusionTransformer(label_dims, shape_mean, shape_std)

    # HMTT
    model = HMTT(label_dims, label_order=HMTT_ORDER)
"""

import numpy as np
import torch
import torch.nn as nn
import timm


# Shared constants 

LABEL_COLUMNS = [
    "microscope", "cell_line", "culture_medium", "formation_method",
    "seeding_density", "timepoint", "biological_rep", "magnification",
]

SHAPE_FEATURES = [
    "area", "perimeter", "eccentricity", "solidity", "extent",
    "equivalent_diameter", "major_axis_length", "minor_axis_length", "circularity",
]

# Causal label ordering for HMTT (Section: Implementation Details of IPP Models)
# Progresses from stable upstream determinants to condition-dependent variables
HMTT_ORDER = [
    "cell_line", "culture_medium", "seeding_density", "magnification",
    "microscope", "timepoint", "biological_rep",
]

# Fusion Transformer hyperparameters (Table 2)
FUSION_D_MODEL  = 256
FUSION_N_HEADS  = 4
FUSION_N_LAYERS = 3
FUSION_FF_DIM   = 512
FUSION_DROPOUT  = 0.1


# (1/3) MultiTaskBackbone — ConvNeXt, ViT, CoAtNet 

class MultiTaskBackbone(nn.Module):
    """
    Generic multi-head wrapper for any timm backbone.
    One independent nn.Linear head per protocol label.

    Used for:
      - ConvNeXt-Tiny  (pretrained=True,  backbone='convnext_tiny')
      - ViT-B/16       (pretrained=True,  backbone='vit_base_patch16_224')
      - CoAtNet-0      (pretrained=False, backbone='coatnet_0_224')

    Args:
        backbone_name: timm model identifier
        label_dims:    {label_col: n_classes}
        pretrained:    whether to load ImageNet weights via timm
    """
    def __init__(self, backbone_name: str, label_dims: dict,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=pretrained,
                                          num_classes=0)
        D = self.backbone.num_features
        self.heads = nn.ModuleDict({
            label: nn.Linear(D, n_cls)
            for label, n_cls in label_dims.items()
        })

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        return {label: head(feat) for label, head in self.heads.items()}


# (2/3) ImageShapeFusionTransformer

class ImageShapeFusionTransformer(nn.Module):
    """
    ConvNeXt-Tiny backbone + 9 per-feature shape tokens fused via a
    Transformer encoder (d=256, 3 layers, 4 heads).

    Each of the 9 morphometric features (area, perimeter, eccentricity, …)
    is projected independently into a D-dimensional token.
    Shape features are z-normalised using training-set statistics stored as
    buffers so they are saved with the checkpoint and transferred automatically.

    Args:
        label_dims:  {label_col: n_classes}
        shape_mean:  (9,) float32 array — per-feature mean on training set
        shape_std:   (9,) float32 array — per-feature std  on training set
    """
    def __init__(self, label_dims: dict,
                 shape_mean: np.ndarray, shape_std: np.ndarray):
        super().__init__()
        n_shape = len(SHAPE_FEATURES)
        D       = FUSION_D_MODEL

        # Image branch
        self.backbone   = timm.create_model("convnext_tiny",
                                            pretrained=True, num_classes=0)
        self.image_proj = nn.Linear(self.backbone.num_features, D)

        # Shape branch — one scalar-to-D projection per feature
        self.shape_proj = nn.ModuleList([nn.Linear(1, D)
                                          for _ in range(n_shape)])
        self.n_shape    = n_shape

        # Positional embeddings: 1 image token + n_shape shape tokens
        self.pos_embed  = nn.Parameter(
            torch.randn(1, 1 + n_shape, D) * 0.02)

        # Transformer encoder
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=D, nhead=FUSION_N_HEADS,
            dim_feedforward=FUSION_FF_DIM,
            dropout=FUSION_DROPOUT, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=FUSION_N_LAYERS)
        self.norm        = nn.LayerNorm(D)

        # Classification heads
        self.heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(D, n_cls)
            ) for label, n_cls in label_dims.items()
        })

        # Normalisation stats — saved with model, auto-moved with .to(device)
        self.register_buffer("shape_mean",
                             torch.tensor(shape_mean, dtype=torch.float32))
        self.register_buffer("shape_std",
                             torch.tensor(shape_std,  dtype=torch.float32))

    def forward(self, images: torch.Tensor,
                shape_feats: torch.Tensor) -> dict:
        # Image token
        img_tok = self.image_proj(
            self.backbone(images)).unsqueeze(1)              # (B, 1, D)

        # Shape tokens (z-normalised)
        sn = (shape_feats - self.shape_mean) / (self.shape_std + 1e-6)
        s_toks = torch.cat(
            [self.shape_proj[i](sn[:, i:i+1]).unsqueeze(1)
             for i in range(self.n_shape)], dim=1)           # (B, n, D)

        seq   = torch.cat([img_tok, s_toks], dim=1) + self.pos_embed   # (B,1+n,D)
        fused = self.norm(
            self.transformer(seq.permute(1, 0, 2))[0])      # (B, D)
        return {label: head(fused) for label, head in self.heads.items()}


# (3/3) HMTT — Hierarchical Multi-Task Transformer

class HMTT(nn.Module):
    """
    ViT-B/16 encoder + hierarchical label conditioning.

    Training:  teacher forcing — each head receives ground-truth embeddings of
               all preceding labels as context (efficient, stable gradients).
    Inference: autoregressive — greedy predictions from earlier heads are
               embedded and passed as context for later heads.

    Causal order (from paper):
        cell_line → culture_medium → seeding_density → magnification →
        microscope → timepoint → biological_rep

    Any labels not listed in label_order are predicted flat (no context).

    Args:
        label_dims:   {label_col: n_classes}
        label_order:  causal ordering list (subset of label_dims keys)
    """
    def __init__(self, label_dims: dict, label_order: list):
        super().__init__()
        self.label_order = label_order
        self.all_labels  = list(label_dims.keys())

        self.encoder  = timm.create_model("vit_base_patch16_224",
                                          pretrained=True, num_classes=0)
        D = self.encoder.num_features

        # Per-label context embeddings
        self.embeds = nn.ModuleDict({
            lab: nn.Embedding(label_dims[lab], D)
            for lab in self.all_labels
        })

        # Per-label heads — input is [feature ‖ context], both D-dim
        self.heads = nn.ModuleDict({
            lab: nn.Sequential(
                nn.LayerNorm(D * 2),
                nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(D, label_dims[lab])
            ) for lab in self.all_labels
        })

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None,
                teacher_forcing: bool = True) -> dict:
        """
        Args:
            x:               (B, 3, H, W) input images
            labels:          (B, L) ground-truth label indices (needed for teacher forcing)
            teacher_forcing: True during training, False at evaluation

        Returns:
            {label_col: logits_tensor (B, n_cls)}
        """
        B    = x.size(0)
        feat = self.encoder(x)                                # (B, D)
        zeros = torch.zeros(B, feat.size(1), device=x.device)

        outputs = {}
        chosen  = {}   # stores greedy predictions for autoregressive eval

        for i, lab in enumerate(self.label_order):
            # Build context from all preceding causal labels
            if i == 0:
                ctx = zeros
            else:
                ctx = zeros.clone()
                for j in range(i):
                    prev = self.label_order[j]
                    if teacher_forcing and labels is not None:
                        idx = labels[:, self.all_labels.index(prev)]
                    else:
                        idx = chosen[prev]
                    ctx = ctx + self.embeds[prev](idx)

            logits       = self.heads[lab](torch.cat([feat, ctx], dim=1))
            outputs[lab] = logits
            chosen[lab]  = logits.argmax(dim=1)

        # Labels not in causal order: predict flat
        for lab in self.all_labels:
            if lab not in self.label_order:
                outputs[lab] = self.heads[lab](
                    torch.cat([feat, zeros], dim=1))

        return outputs


# Factory

def build_ipp_model(name: str, label_dims: dict,
                    shape_mean=None, shape_std=None) -> nn.Module:
    """
    Instantiate an IPP model by name.

    Args:
        name:        one of 'convnext', 'vit', 'coatnet', 'fusion', 'hmtt'
        label_dims:  {label_col: n_classes}
        shape_mean:  required for 'fusion'
        shape_std:   required for 'fusion'

    Returns:
        nn.Module instance
    """
    name = name.lower()
    if name == "convnext":
        return MultiTaskBackbone("convnext_tiny",        label_dims, pretrained=True)
    if name == "vit":
        return MultiTaskBackbone("vit_base_patch16_224", label_dims, pretrained=True)
    if name == "coatnet":
        return MultiTaskBackbone("coatnet_0_224",        label_dims, pretrained=False)
    if name == "fusion":
        if shape_mean is None or shape_std is None:
            raise ValueError("'fusion' model requires shape_mean and shape_std")
        return ImageShapeFusionTransformer(label_dims, shape_mean, shape_std)
    if name == "hmtt":
        order = [l for l in HMTT_ORDER if l in label_dims]
        return HMTT(label_dims, label_order=order)
    raise ValueError(f"Unknown IPP model '{name}'. "
                     f"Choose from: convnext, vit, coatnet, fusion, hmtt")