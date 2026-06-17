"""
src/utils/visualization.py — Shared visualisation helpers

Covers:
  - Segmentation overlays (used in 02 / 03)
  - IPP training curves (used in 04)
  - Grad-CAM overlay (used in 09)
  - Temporal prediction grids (used in 10)
  - Results table pretty-print
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from PIL import Image


# Image loading 

def load_tiff_gray_norm(path: str, size: int = None) -> np.ndarray:
    """Load a TIFF file as a float32 grayscale array normalised to [0, 1]."""
    import tifffile
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in [1, 3] and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.mean(axis=-1)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if size:
        arr = np.array(Image.fromarray((arr * 255).astype(np.uint8))
                       .resize((size, size))) / 255.0
    return arr


# Segmentation

def plot_seg_overlay(img: np.ndarray, pred_mask: np.ndarray,
                     gt_mask: np.ndarray = None,
                     ax=None, title: str = "") -> None:
    """
    Show a spheroid image with the predicted mask contour overlaid in green.
    If gt_mask is supplied, GT contour is shown in red for comparison.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.contour(pred_mask, colors=["lime"],   linewidths=1.0, label="Pred")
    if gt_mask is not None:
        ax.contour(gt_mask, colors=["red"], linewidths=0.8, linestyles="--",
                   label="GT")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_seg_comparison(images, pred_masks, gt_masks=None,
                        model_names=None, n=4,
                        save_path: str = None) -> None:
    """
    Grid comparison of segmentation results across multiple models.
    Matches Fig. 4 layout in the paper.

    Args:
        images:      list of (H, W) grayscale arrays
        pred_masks:  list of (H, W) binary arrays (one list per model)
        gt_masks:    list of (H, W) GT binary arrays (optional)
        model_names: list of model name strings
        n:           number of sample images to show
    """
    n_models = len(pred_masks)
    n_cols   = n_models + (2 if gt_masks else 1)
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.2, n * 2.2))

    for i in range(n):
        col = 0
        axes[i, col].imshow(images[i], cmap="gray")
        if i == 0: axes[i, col].set_title("Input", fontsize=8)
        axes[i, col].axis("off"); col += 1

        if gt_masks is not None:
            axes[i, col].imshow(gt_masks[i], cmap="gray")
            if i == 0: axes[i, col].set_title("GT Mask", fontsize=8)
            axes[i, col].axis("off"); col += 1

        for mi, masks in enumerate(pred_masks):
            axes[i, col].imshow(masks[i], cmap="gray")
            if i == 0:
                lbl = model_names[mi] if model_names else f"Model {mi+1}"
                axes[i, col].set_title(lbl, fontsize=8)
            axes[i, col].axis("off"); col += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# IPP training curves

def plot_training_curves(histories: dict, metric: str = "val_f1",
                          save_path: str = None) -> None:
    """
    Plot training curves for all IPP models on one figure.

    Args:
        histories: {model_name: {metric: [epoch_values]}}
        metric:    which history key to plot on y-axis
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, h) in zip(axes, histories.items()):
        if metric in h:
            ax.plot(h[metric], color="steelblue", linewidth=1.5)
        if "train_f1" in h:
            ax.plot(h["train_f1"], color="orange", linewidth=1.0,
                    linestyle="--", alpha=0.7)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric.replace("_", " ").title())
    plt.suptitle("IPP Training Curves", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Grad-CAM 

def overlay_gradcam(img_pil: Image.Image, cam: np.ndarray,
                    size: int = 224, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        img_pil: PIL RGB image
        cam:     (H', W') float array normalised to [0, 1]
        size:    output resolution
        alpha:   heatmap blend weight

    Returns:
        (size, size, 3) float array in [0, 1]
    """
    img    = np.array(img_pil.resize((size, size))) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    cam_r  = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                      .resize((size, size), Image.BILINEAR)) / 255.0
    heat   = plt.cm.jet(cam_r)[..., :3]
    blend  = (1 - alpha) * img + alpha * heat
    return np.clip(blend, 0, 1)


def plot_gradcam_grid(img_pil: Image.Image, cams: dict,
                      display_names: dict = None,
                      save_path: str = None) -> None:
    """
    Reproduce Fig. 5: original spheroid on the left, 8 raw CAM heatmaps
    in a 2×4 grid on the right.

    Args:
        img_pil:       original PIL image
        cams:          {label: (H, W) cam array}
        display_names: {label: display string}
        save_path:     optional file path
    """
    labels = list(cams.keys())
    fig    = plt.figure(figsize=(14, 7))
    gs     = gridspec.GridSpec(2, 5, figure=fig,
                                width_ratios=[1.2, 1, 1, 1, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(img_pil, cmap="gray")
    ax0.set_title("Original\nSpheroid", fontsize=10)
    ax0.axis("off")

    positions = [(r, c) for r in range(2) for c in range(1, 5)]
    for (r, c), lab in zip(positions, labels):
        ax  = fig.add_subplot(gs[r, c])
        cam = np.array(Image.fromarray((cams[lab] * 255).astype(np.uint8))
                       .resize((224, 224)))
        ax.imshow(cam, cmap="jet")
        title = (display_names or {}).get(lab, lab)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# Temporal

def plot_temporal_grid(ctx_frames: np.ndarray, gt_frame: np.ndarray,
                       pred_frames: dict, n: int = 4,
                       save_path: str = None) -> None:
    """
    Side-by-side comparison of temporal predictions from all models.
    Matches Fig. 6 layout in the paper.

    Args:
        ctx_frames:  (T, H, W) input context frames
        gt_frame:    (H, W) ground-truth next frame
        pred_frames: {model_name: (H, W) prediction}
        n:           number of samples (batch dimension assumed already sliced to n)
    """
    T       = ctx_frames.shape[0]
    n_cols  = T + 1 + len(pred_frames)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 2.5))

    col = 0
    for t in range(T):
        axes[col].imshow(ctx_frames[t], cmap="gray", vmin=0, vmax=1)
        axes[col].set_title(f"Input t-{T - t}", fontsize=8)
        axes[col].axis("off"); col += 1

    axes[col].imshow(gt_frame, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title("Ground\nTruth", fontsize=8)
    axes[col].axis("off"); col += 1

    for name, pred in pred_frames.items():
        axes[col].imshow(np.clip(pred, 0, 1), cmap="gray")
        axes[col].set_title(name, fontsize=8)
        axes[col].axis("off"); col += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Results table

def print_results_table(results: dict, metrics: list = None) -> None:
    """
    Pretty-print a results dict as an aligned table.

    Args:
        results: {model_name: {metric: value_or_(mean,std)}}
        metrics: which metrics to show (default: all)
    """
    if not results:
        return
    all_metrics = metrics or list(next(iter(results.values())).keys())
    col_w = max(len(k) for k in all_metrics) + 2

    header = f"{'Model':<28}" + "".join(f"{m:>{col_w}}" for m in all_metrics)
    print(header)
    print("─" * len(header))

    for name, vals in results.items():
        row = f"{name:<28}"
        for m in all_metrics:
            v = vals.get(m, "—")
            if isinstance(v, tuple):
                cell = f"{v[0]:.4f}±{v[1]:.4f}"
            elif isinstance(v, float):
                cell = f"{v:.4f}"
            else:
                cell = str(v)
            row += f"{cell:>{col_w}}"
        print(row)