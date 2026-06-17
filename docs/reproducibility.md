# Reproducibility Notes

This document explains every design decision made to ensure that results
reported in the paper can be reproduced from this repository.

---

## Random seed protocol

All experiments use **three independent seeds: 42, 123, 999**.
Results are reported as mean ± standard deviation across these seeds.

The `set_seed()` function in `src/utils/seed.py` pins:
- Python `random`
- NumPy RNG
- PyTorch CPU and CUDA RNGs
- `PYTHONHASHSEED` environment variable
- `torch.backends.cudnn.deterministic = True`

Seeds are always set before model instantiation, data shuffling, and training.

---

## Data split design

Splits are performed at the **technical replicate level**, not image level.
All images from a given acquisition session (T1–T24) appear in exactly one split.

| Split | Replicates | Purpose |
|-------|------------|---------|
| Train | T1, T2, T3, T4 | Optimisation |
| Val | T5, T8 | Hyperparameter tuning / early stopping |
| Test | T6, T7, T9–T24 | Final evaluation only |

This prevents acquisition-specific information leakage — the model cannot
memorise microscope artefacts from a session seen during training.

### Class coverage audit (Section: Attribute-Level Class Coverage Analysis)

All attributes except `timepoint` have 100% class coverage across splits.
Two timepoint classes (`000h`, `032h`) appear in the test set only — this
minor open-set condition is flagged in the paper and does not affect aggregate metrics significantly.

---

## Metric computation

### Segmentation (Table 4)

- Dice and IoU are computed per-image and averaged.
- Images where **both** predicted and GT masks are empty (Dice=1, IoU=0 edge case) are **excluded** from averages.
- 17 images with unavailable GT masks are excluded.
- 95% confidence intervals: µ ± 1.96 · (σ / √n), n = test images.

### IPP (Table 5)

- **Mean per-attribute accuracy**: average of per-head top-1 accuracy across all K labels.
- **Exact-match accuracy**: fraction of images where all K predictions are correct simultaneously.
- Table 5 reports mean per-attribute accuracy. Exact-match is reported separately.
- 95% CI computed over 3 seeds: 1.96 × SD / √3.

### Temporal (Table 7)

- MSE, SSIM, PSNR computed per frame pair and averaged over the test set.
- 95% CI: 1.96 × SD / √N, N ≈ 8,000 (sample-level).

---

## No ground-truth mask leakage

The morphometric feature extraction pipeline uses **only predicted masks**
(from RefineNet) during IPP training and evaluation. Ground-truth masks are
used exclusively to train the segmentation models.

This is enforced in `03_morphometry_extraction.ipynb` by:
1. Running RefineNet inference to generate `data/predicted_masks/*.tiff`
2. Using only those predicted masks in `extract_dataset_features(mode="predicted")`

---

## Mixed precision

Mixed-precision (AMP) training is used where indicated in the model configs.
Results should be numerically identical between AMP and full-precision runs
because `GradScaler` prevents gradient underflow and the final metrics are
computed in float32.

---

## Environment

Tested on:
- Python 3.10
- PyTorch 2.1 + CUDA 12.1
- timm 0.9.12
- segmentation-models-pytorch 0.3.3

Known version-sensitive behaviour:
- `timm` model API changed in 0.9.x — `num_classes=0` to get features.
- `torch.amp.autocast(device_type=...)` replaces the deprecated `torch.cuda.amp.autocast` in PyTorch 2.0+. All notebooks use the new API.
- `einops.rearrange` is used in `TransUNet` to reshape ViT patch tokens — ensure `einops >= 0.7`.

---

## Kaggle reproducibility

The notebooks are Kaggle-ready — paths use `/kaggle/input/slimia-metadata/...`.
To run locally, update `CSV_PATH` / `CFG.csv_path` at the top of each notebook.

Kaggle GPU T4 × 2 was used for segmentation and IPP training.
Expected runtimes per model per seed:
- Segmentation (RefineNet): ~3–4 hours
- IPP (CoAtNet-0): ~2–3 hours
- Temporal (PredRNN++): ~1–2 hours

---

## Statistical tests (Section D)

Paired two-sided t-tests and Wilcoxon signed-rank tests were used to compare
CoAtNet against each other IPP model across 8 per-label F1 scores (n=8 pairs, df=7).

Holm–Bonferroni correction was applied across 4 pairwise comparisons.
All p-values exceeded α=0.05 after correction — no model showed statistically
significant uniform superiority over CoAtNet.

Effect sizes reported as Cohen's d_z with 95% confidence intervals.