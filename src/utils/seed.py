"""
src/utils/seed.py — Reproducibility helpers

Sets all RNGs (Python, NumPy, PyTorch, CUDA) to a fixed seed so that
results are reproducible across independent runs.

Used consistently across all notebooks and training scripts with
seeds [42, 123, 999] (three independent seeds, results averaged).
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Pin all RNGs for a fully reproducible run.

    Args:
        seed: integer seed — paper uses 42, 123, 999
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


SEEDS = [42, 123, 999]   # three independent seeds used throughout the paper