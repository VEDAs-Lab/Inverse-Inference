"""
src/morphometry/shape_metrics.py — Shape descriptor computation

Computes the 9 morphometric descriptors used in the ImageShapeFusionTransformer
(Table 2 in paper) from a binary segmentation mask.

Feature definitions:
    area                 — foreground pixel count
    perimeter            — boundary pixel length
    eccentricity         — focal distance / major axis (0=circle, 1=line)
    solidity             — area / convex hull area (compactness measure)
    extent               — area / bounding box area
    equivalent_diameter  — diameter of circle with same area
    major_axis_length    — major ellipse axis length
    minor_axis_length    — minor ellipse axis length
    circularity          — 4π·area / perimeter² (1.0 = perfect circle)

All values are computed on the largest connected component to avoid
contamination from staining artefacts or background debris.
"""

import numpy as np

SHAPE_FEATURES = [
    "area", "perimeter", "eccentricity", "solidity", "extent",
    "equivalent_diameter", "major_axis_length", "minor_axis_length",
    "circularity",
]

_NAN_ROW = {f: np.nan for f in SHAPE_FEATURES}


def extract_shape_features(mask: np.ndarray,
                            min_area: int = 50) -> dict:
    """
    Extract 9 shape descriptors from a binary mask.

    Args:
        mask:     2D uint8 array, foreground = 1
        min_area: minimum foreground pixels; returns NaNs if below this

    Returns:
        dict {feature_name: float_value}
        All NaN if the mask is empty or below min_area.
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image is required: pip install scikit-image")

    if mask.sum() < min_area:
        return _NAN_ROW.copy()

    labeled = measure.label(mask)
    if labeled.max() == 0:
        return _NAN_ROW.copy()

    # Use the largest connected component only
    props  = measure.regionprops(labeled)
    region = max(props, key=lambda r: r.area)

    area       = float(region.area)
    perimeter  = float(region.perimeter) if region.perimeter > 0 else 1e-6
    circularity = float(4 * np.pi * area / (perimeter ** 2 + 1e-8))

    return {
        "area":                area,
        "perimeter":           perimeter,
        "eccentricity":        float(region.eccentricity),
        "solidity":            float(region.solidity),
        "extent":              float(region.extent),
        "equivalent_diameter": float(region.equivalent_diameter),
        "major_axis_length":   float(region.major_axis_length),
        "minor_axis_length":   float(region.minor_axis_length),
        "circularity":         circularity,
    }


def normalise_shape_features(df, feature_cols=None,
                              mean=None, std=None):
    """
    Z-normalise shape features using training-set statistics.

    Args:
        df:           DataFrame containing shape feature columns
        feature_cols: list of column names (default: SHAPE_FEATURES)
        mean:         pre-computed mean array or None (computed from df)
        std:          pre-computed std  array or None (computed from df)

    Returns:
        (normalised_df, mean_array, std_array)
    """
    feature_cols = feature_cols or SHAPE_FEATURES
    if mean is None:
        mean = df[feature_cols].mean().values.astype(np.float32)
    if std is None:
        std  = df[feature_cols].std().replace(0, 1).values.astype(np.float32)
    out = df.copy()
    out[feature_cols] = (df[feature_cols].values - mean) / (std + 1e-6)
    return out, mean, std