# Training Guide

Step-by-step instructions for running every component of the SLiMIA-IPP pipeline.

---

## Prerequisites

```bash
git https://github.com/VEDAs-Lab/Inverse-Inference.git
cd SLIMIA-IPP

# Option A — pip
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yml
conda activate slimia-ipp
```

Download SLiMIA and place (or symlink) the images and `slimia_metadata.csv`
according to `data/README.md`. Update all `CSV_PATH` / `csv_path` variables in
the notebooks or configs to point to your local paths.

---

## Step 1 — Segmentation (notebook 02)

Run `notebooks/02_segmentation_training.ipynb` or use the CLI:

```bash
# Train RefineNet (best model) with all 3 seeds
python src/segmentation/train.py \
    --model refinenet \
    --csv   data/slimia_metadata.csv \
    --ckpt  checkpoints/segmentation/ \
    --seeds 42 123 999

# Evaluate a checkpoint
python src/segmentation/evaluate.py \
    --model refinenet \
    --ckpt  checkpoints/segmentation/refinenet_seed42.pth \
    --csv   data/slimia_metadata.csv

# Generate predicted masks for the full dataset (required for IPP)
python src/segmentation/evaluate.py \
    --model refinenet \
    --ckpt  checkpoints/segmentation/refinenet_seed42.pth \
    --csv   data/slimia_metadata.csv \
    --mode  infer \
    --outdir data/predicted_masks/
```

All 8 models can be trained by swapping `--model`:
`unetpp | deeplabv3 | deeplabv3plus | attention_unet | segnet | refinenet | swin_unet | transunet`

### Hyperparameters (Table 1)

| Setting | Value |
|---------|-------|
| Optimizer | Adam / AdamW (model-dependent, see `configs/segmentation.yaml`) |
| LR | 1e-4 (CNN) / 3e-4 (transformer) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 8 |
| Max epochs | 200 |
| Early stopping | patience=40 on val Dice |
| Seeds | 42, 123, 999 |

---

## Step 2 — Morphometry Extraction (notebook 03)

Run `notebooks/03_morphometry_extraction.ipynb`.
This reads `data/predicted_masks/` and writes `data/shape_features_with_metadata.csv`.

```python
from src.morphometry import extract_dataset_features

extract_dataset_features(
    metadata_csv  = "data/slimia_metadata.csv",
    pred_mask_dir = "data/predicted_masks/",
    output_csv    = "data/shape_features_with_metadata.csv",
    mode          = "predicted",   # or "otsu" if no predicted masks
)
```

> **Important:** ground-truth masks are never used here — only predicted masks.
> This ensures the IPP pipeline has no annotation leakage.

---

## Step 3 — IPP Training (notebook 04)

Run `notebooks/04_ipp_all_models.ipynb`.
Trains all 5 models with 3 seeds and saves checkpoints to `checkpoints/ipp/`.

### Hyperparameters (Table 2)

| Model | Optimizer | LR | Weight Decay | Loss |
|-------|-----------|-----|--------------|------|
| ConvNeXt-Tiny | Adam | 1e-4 | — | CrossEntropy |
| ViT-B/16 | Adam | 1e-4 | — | CrossEntropy |
| CoAtNet-0 | Adam | 1e-4 | — | CrossEntropy |
| ImageShapeFusion | AdamW | 1e-4 | 1e-2 | Weighted CrossEntropy |
| HMTT | AdamW | 1e-4 | 1e-2 | Weighted CE + Focal (γ=2, α=0.25) |

All models: batch size 32, max 200 epochs, early stopping patience=40,
ReduceLROnPlateau (patience=5, factor=0.5) monitoring val macro-F1.

### Data split (Table in paper)

| Split | Technical Replicates | Images |
|-------|---------------------|--------|
| Train | T1–T4 | ~6,418 |
| Val | T5, T8 | ~714 |
| Test | T6, T7, T9–T24 | ~7,999 |

No random image-level splitting — entire acquisition sessions stay together.

### Augmentation

Conservative (morphology-preserving): resize 224×224, RandomHorizontalFlip,
RandomRotation(10°), normalize mean=std=[0.5].
No elastic deformation, scale jitter, colour perturbation, or random cropping.

---

## Step 4 — Domain-Adversarial Training (notebook 08)

Run `notebooks/08_domain_adversarial.ipynb`.
Trains CoAtNet-0 with DANN and saves to `checkpoints/ipp/CoAtNet_DANN_seed*.pth`.

The GRL lambda is updated at the start of each epoch:
```python
p = (epoch - 1) / total_epochs
model.grl.set_lambda(p)
```

---

## Step 5 — Ablation Study (notebook 07)

Run `notebooks/07_ablation_study.ipynb`.
Uses the **biological-only** label set (5 labels, no microscope/magnification/replicate).
Each of the 5 variants trains to completion under identical conditions.

---

## Step 6 — Grad-CAM (notebook 09)

Run `notebooks/09_gradcam_analysis.ipynb`.
Requires a trained CoAtNet-0 checkpoint at `checkpoints/ipp/CoAtNet-0_seed42.pth`
and the label encoder `.pkl` files in `results/ipp/`.

---

## Step 7 — Temporal Prediction (notebook 10)

Run `notebooks/10_temporal_prediction.ipynb`.
All 4 models trained with seed=42, group-wise 70/15/15 split.

### Hyperparameters (Table 3)

| Setting | Value |
|---------|-------|
| Image size | 128×128 |
| Context frames (T) | 2 |
| Frame gaps (Δt) | 1, 2, 3 |
| Batch size | 8 |
| Optimizer | Adam (L1 models) / AdamW (MetadataFusion) |
| LR | 1e-4 |
| Early stopping | patience=40 on val SSIM |

---

## Step 8 — Cross-Dataset Validation (notebook 11)

Run `notebooks/11_cross_dataset_validation.ipynb`.
Download RxRx1 channel-1 images and set `CFG.rxrx1_csv` accordingly.
Models are transferred **without fine-tuning** to isolate representation robustness.

---

## Checkpoint directory layout

```
checkpoints/
├── segmentation/
│   ├── refinenet_seed42.pth
│   ├── refinenet_seed123.pth
│   ├── refinenet_seed999.pth
│   └── ...
├── ipp/
│   ├── CoAtNet-0_seed42.pth
│   ├── FusionTransformer_seed42.pth
│   ├── HMTT_seed42.pth
│   ├── CoAtNet_DANN_seed42.pth
│   └── ablation/
│       ├── Full_Model.pth
│       └── ...
└── temporal/
    ├── ConvLSTM_best.pth
    ├── PredRNN++_best.pth
    ├── MetadataFusion_best.pth
    └── PhyDNet_best.pth
```