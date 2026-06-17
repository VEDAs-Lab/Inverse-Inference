# Architecture Overview

This document describes every model used in the SLiMIA-IPP paper,
their design choices, and how the components connect.

---

## Pipeline

```
Raw spheroid image (.ome.tiff)
        │
        ▼
┌─────────────────────────────┐
│  Segmentation (RefineNet)   │  → predicted binary mask
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Morphometry Extraction     │  → 9 shape descriptors per image
│  (src/morphometry/)         │    (area, perimeter, circularity…)
└─────────────────────────────┘
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
┌──────────────────┐                         ┌─────────────────────────┐
│  IPP Models      │                         │  Temporal Models        │
│  (src/ipp/)      │                         │  (src/temporal/)        │
└──────────────────┘                         └─────────────────────────┘
```

---

## Segmentation Models (Table 1)

All models take `(B, 3, H, W)` RGB input and output `(B, 1, H, W)` binary logits.
CNN backbones are ImageNet-pretrained. Transformer models require 224×224 input.

| Model | Backbone | Loss | Input |
|-------|----------|------|-------|
| U-Net++ | ResNet-50 | Focal Tversky (α=0.7, β=0.3, γ=0.75) | 256×256 |
| DeepLabV3 | ResNet-34 | BCE + Dice | 256×256 |
| DeepLabV3+ | ResNet-50 | Focal Tversky | 256×256 |
| Attention U-Net | Custom | BCE + Dice | 256×256 |
| SegNet | VGG-style | Focal Tversky | 256×256 |
| **RefineNet** ★ | ResNet-34 | Focal Tversky | 256×256 |
| Swin-UNet | Swin-Tiny | BCEWithLogits | 224×224 |
| TransUNet | ViT-B/16 + CNN | BCEWithLogits | 224×224 |

★ Best model — used to generate predicted masks for all downstream IPP experiments.

### RefineNet design

Four RefineBlock stages, each containing:
1. **RCU** (Residual Conv Unit) — two 3×3 convs with residual connection
2. **CRP** (Chained Residual Pooling) — multi-scale context via chained max-pooling + conv
3. **Fusion** with bilinearly upsampled output from the next stage

This refinement-focused design excels at faint or irregular spheroid edges.

---

## IPP Models (Table 2)

All models take `(B, 3, 224, 224)` images and output one logit tensor per label.

### MultiTaskBackbone

Generic wrapper: `timm backbone → num_classes=0 → one nn.Linear head per label`.

- **ConvNeXt-Tiny** (pretrained) — best macro-F1 (0.946); local convolutional priors suit fine-grained texture cues like seeding density and timepoint progression
- **ViT-B/16** (pretrained) — CLS token captures global geometry; best on culture medium (F1=0.964)
- **CoAtNet-0** (from scratch) — best macro accuracy (98.0%); hybrid conv+attention balances texture and structure

### ImageShapeFusionTransformer

```
Image → ConvNeXt-Tiny backbone → proj → image token (1×D)
                                                        │
Shape feats (9) → per-feature Linear → shape tokens (9×D) → cat → +pos_embed
                                                                        │
                                              Transformer Encoder (3L, 4H, d=256)
                                                                        │
                                                    CLS output → per-label heads
```

Shape features are z-normalised using training-set statistics stored as model buffers.

### HMTT

```
Image → ViT-B/16 → feat (B×D)

For each label in causal order:
    ctx = sum of preceding label embeddings
    logits = head( [feat ‖ ctx] )
    chosen = argmax(logits)  # used as context for the next label
```

**Training:** teacher forcing (GT context) for stability.  
**Inference:** autoregressive greedy decoding.

**Causal order:** `cell_line → culture_medium → seeding_density → magnification → microscope → timepoint → biological_rep`

---

## Domain-Adversarial Training (DANN)

Applied to **CoAtNet-0** (Section: Domain-Adversarial Training).

```
Input → CoAtNet backbone → features
                │                   └──→ bio heads  (L_bio = CrossEntropy)
                │
                └──→ GRL(λ) → discriminator MLP → domain logits (L_domain = CrossEntropy)

L_total = L_bio + L_domain
```

**GRL schedule:** λ(p) = 2 / (1 + exp(−10p)) − 1,  p ∈ [0,1] = training progress  
**Domain variable:** technical replicate ID (T1–T24) — represents acquisition session

The GRL forces the backbone to learn features that are discriminative for biology
but indistinguishable across acquisition sessions.

---

## Temporal Prediction Models (Table 3)

All models take `(B, T=2, 1, 128, 128)` context frames and predict the next frame `(B, 1, 128, 128)`.

| Model | Architecture | Key Component |
|-------|-------------|--------------|
| ConvLSTM | 1 ConvLSTMCell (32ch) + Conv decoder | Convolutional gates |
| PredRNN++ | 4-layer ST-LSTM + GHU | Dual-memory (H spatial, M temporal) |
| MetadataFusion | CNN enc-dec + FiLM | Metadata-conditioned generation |
| PhyDNet | PhyCell + residual ConvLSTM | Physics operator bank |

### MetadataFusion FiLM conditioning

Categorical metadata (microscope, cell_line, etc.) are embedded and combined
with continuous metadata (seeding_density_num, Δt) via a 2-layer MLP to produce
per-channel γ (scale) and β (shift) applied at the encoder bottleneck:

```
h_conditioned = h · (1 + γ) + β
```

### PhyDNet physics cell

Learns a bank of `num_ops=3` spatial convolutional operators per channel.
Approximates du/dt = F(h) via Euler integration with learnable coefficients:

```
du = tanh( Σ_k coeff_k · op_k(h) )
h_tilde = h_prev + dt · du
```

A gating mechanism blends the physics prediction with the previous hidden state.