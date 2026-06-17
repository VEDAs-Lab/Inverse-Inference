"""
src/morphometry/feature_extractor.py — Morphometric feature extraction pipeline

Extracts 9 classical geometric shape descriptors from binary segmentation masks
(predicted by RefineNet) for each image in the SLiMIA dataset.

These features are used exclusively during IPP training — never ground-truth masks,
ensuring a fully automatic pipeline with no annotation leakage.

Used by: notebooks/03_morphometry_extraction.ipynb
Output:  data/shape_features_with_metadata.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm

try:
    from skimage import measure, morphology
    from skimage.filters import threshold_otsu
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

from .shape_metrics import extract_shape_features

# Minimum foreground area in pixels — smaller regions are treated as noise
MIN_AREA_PX = 50


def load_tiff_gray(path: str) -> np.ndarray:
    """Load any TIFF as a 2D float32 array normalised to [0, 1]."""
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in [1, 3] and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.mean(axis=-1)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.nan_to_num(arr).astype(np.float32)


def load_binary_mask(path: str) -> np.ndarray:
    """Load a pre-saved binary mask (uint8, values 0 or 1/255)."""
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in [1, 3] else arr.mean(-1)
    return (arr > 0).astype(np.uint8)


def otsu_mask(img_arr: np.ndarray,
              min_area: int = MIN_AREA_PX) -> np.ndarray:
    """
    Fallback segmentation via Otsu thresholding + keep largest component.
    Used when no predicted mask is available for a given image.
    """
    if not HAVE_SKIMAGE:
        return (img_arr > 0.5).astype(np.uint8)
    try:
        thresh = threshold_otsu(img_arr)
        binary = img_arr > thresh
    except Exception:
        binary = img_arr > 0.5
    binary  = morphology.remove_small_objects(binary, min_size=min_area)
    labeled = measure.label(binary)
    if labeled.max() == 0:
        return binary.astype(np.uint8)
    largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return (labeled == largest).astype(np.uint8)


def build_pred_mask_lookup(pred_mask_dir: str) -> dict:
    """
    Build a filename-stem → path lookup for predicted masks.

    Args:
        pred_mask_dir: directory containing *_pred.tiff files

    Returns:
        {image_stem: mask_path}
    """
    lookup = {}
    if not os.path.isdir(pred_mask_dir):
        return lookup
    for p in Path(pred_mask_dir).glob("*_pred.tiff"):
        stem = p.stem.replace("_pred", "")
        lookup[stem] = str(p)
    return lookup


def extract_dataset_features(metadata_csv: str,
                              pred_mask_dir: str,
                              output_csv: str,
                              mode: str = "predicted") -> pd.DataFrame:
    """
    Run morphometric feature extraction over the full SLiMIA dataset.

    Args:
        metadata_csv:   path to slimia_metadata.csv
        pred_mask_dir:  directory with RefineNet-predicted masks (*_pred.tiff)
        output_csv:     where to save the output CSV
        mode:           'predicted' (use RefineNet masks) or
                        'otsu' (fallback Otsu thresholding)

    Returns:
        DataFrame with original metadata + 9 shape feature columns
    """
    df   = pd.read_csv(metadata_csv)
    df["full_path"] = df["full_path"].astype(str).str.strip()

    lookup    = build_pred_mask_lookup(pred_mask_dir) if mode == "predicted" else {}
    records   = []
    n_pred, n_otsu, n_fail = 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Extracting morphometric features"):
        img_path = row["full_path"]
        stem     = Path(img_path).stem
        record   = row.to_dict()
        record["image_path"] = img_path

        try:
            if mode == "predicted" and stem in lookup:
                mask = load_binary_mask(lookup[stem])
                src  = "predicted"
                n_pred += 1
            else:
                img  = load_tiff_gray(img_path)
                mask = otsu_mask(img)
                src  = "otsu"
                n_otsu += 1

            feats = extract_shape_features(mask, min_area=MIN_AREA_PX)

        except Exception:
            feats = {f: np.nan for f in [
                "area", "perimeter", "eccentricity", "solidity", "extent",
                "equivalent_diameter", "major_axis_length",
                "minor_axis_length", "circularity"]}
            src = "failed"
            n_fail += 1

        record.update(feats)
        record["mask_source"] = src
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"\nExtraction complete:")
    print(f"  From predicted masks : {n_pred}")
    print(f"  From Otsu fallback   : {n_otsu}")
    print(f"  Failed               : {n_fail}")
    print(f"  Saved to             : {output_csv}")
    return out_df