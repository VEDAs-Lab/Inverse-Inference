"""
src/domain_adaptation/discriminator.py — Domain discriminator + full DANN model

Architecture (paper Section: Domain-Adversarial Training):
    Feature extractor : CoAtNet-0 backbone
    Biological heads  : one nn.Linear per biological attribute
    Domain discriminator: Linear(D→256) → ReLU → Dropout(0.3) → Linear(256→N_domain)
                         attached via GRL

The domain variable is the technical replicate ID (T1–T24), representing
distinct acquisition sessions.
"""

import torch
import torch.nn as nn
import timm

from .grl import GRL


class DomainDiscriminator(nn.Module):
    """
    Two-layer MLP domain classifier.
    Attached via GRL so gradients are reversed when flowing to the backbone.

    Args:
        in_features: backbone feature dimension (D)
        n_domains:   number of domain classes (= number of technical replicates)
        hidden_dim:  hidden layer width (default 256, per paper)
        dropout:     dropout probability (default 0.3, per paper)
    """
    def __init__(self, in_features: int, n_domains: int,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANNCoAtNet(nn.Module):
    """
    Domain-Adversarial Neural Network built on CoAtNet-0.

    Forward returns (bio_outputs, domain_logits) where:
      - bio_outputs:    {label_col: logits (B, n_cls)}
      - domain_logits:  (B, n_domains)

    Loss (Equation 2 in paper):
        L_total = L_bio + L_domain
    Both are standard CrossEntropyLoss, equally weighted.

    The GRL λ is updated externally before each forward pass via:
        model.grl.set_lambda(p)

    Args:
        label_dims: {label_col: n_classes} — biological attributes only
        n_domains:  number of technical replicates (domain classes)
    """
    def __init__(self, label_dims: dict, n_domains: int):
        super().__init__()
        self.backbone = timm.create_model("coatnet_0_224",
                                          pretrained=False, num_classes=0)
        D = self.backbone.num_features

        # Biological prediction heads
        self.bio_heads = nn.ModuleDict({
            lab: nn.Linear(D, n_cls)
            for lab, n_cls in label_dims.items()
        })

        # Domain discriminator via GRL
        self.grl           = GRL()
        self.discriminator = DomainDiscriminator(D, n_domains)

    def forward(self, x: torch.Tensor):
        feat     = self.backbone(x)
        bio_outs = {lab: head(feat) for lab, head in self.bio_heads.items()}
        dom_out  = self.discriminator(self.grl(feat))
        return bio_outs, dom_out
    