"""
src/domain_adaptation/grl.py — Gradient Reversal Layer

Implements the GRL from Ganin & Lempitsky (2015) as a custom autograd Function.

In the forward pass the GRL acts as an identity.
In the backward pass it multiplies gradients by −λ, causing the feature
extractor to learn domain-invariant representations.

λ follows the standard DANN schedule (Equation 1 in paper):
    λ(p) = 2 / (1 + exp(−10·p)) − 1
where p ∈ [0, 1] is normalised training progress (epoch / total_epochs).
"""

import numpy as np
import torch
import torch.nn as nn


class _GradientReversalFn(torch.autograd.Function):
    """Internal autograd function — use the GRL wrapper below."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lam, = ctx.saved_tensors
        return -lam.item() * grad_output, None


class GRL(nn.Module):
    """
    Gradient Reversal Layer.

    Usage:
        grl = GRL()
        grl.set_lambda(p)          # call before each forward pass
        reversed_features = grl(features)
    """
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def set_lambda(self, p: float) -> None:
        """
        Update λ according to training progress.

        Args:
            p: normalised training progress in [0, 1]
               (e.g.  p = (epoch - 1) / total_epochs)
        """
        self.lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lam)