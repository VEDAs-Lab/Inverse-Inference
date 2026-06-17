"""
models/__init__.py — All 8 segmentation architectures used in the SLiMIA-IPP paper.

    Model          Backbone        Loss             Input size
    ─────────────────────────────────────────────────────────
    UNetPlusPlus   ResNet-50       FocalTversky     256×256
    DeepLabV3      ResNet-34       BCE + Dice       256×256
    DeepLabV3Plus  ResNet-50       FocalTversky     256×256
    AttentionUNet  custom          BCE + Dice       256×256
    SegNet         VGG-style       FocalTversky     256×256
    RefineNet      ResNet-34       FocalTversky     256×256  ← best (Dice 0.9665)
    SwinUNet       Swin-Tiny       BCEWithLogits    224×224
    TransUNet      ViT-B/16        BCEWithLogits    224×224

Usage:
    from src.segmentation.models import build_model
    model = build_model("refinenet")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from timm.models.vision_transformer import vit_base_patch16_224
from einops import rearrange


# SMP-based models (U-Net++, DeepLabV3, DeepLabV3+)

def _build_unetpp() -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name="resnet50", encoder_weights="imagenet",
        in_channels=3, classes=1, activation=None)


def _build_deeplabv3() -> nn.Module:
    return smp.DeepLabV3(
        encoder_name="resnet34", encoder_weights="imagenet",
        in_channels=3, classes=1, activation=None)


def _build_deeplabv3plus() -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights="imagenet",
        in_channels=3, classes=1, activation=None)


# Attention U-Net

class _AttentionBlock(nn.Module):
    """Additive attention gate (Oktay et al. 2018)."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class AttentionUNet(nn.Module):
    """5-level U-Net with attention gates on every skip connection."""
    def __init__(self):
        super().__init__()

        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2)
        self.e1 = cb(3, 64);    self.e2 = cb(64, 128)
        self.e3 = cb(128, 256); self.e4 = cb(256, 512)
        self.e5 = cb(512, 1024)
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.at5 = _AttentionBlock(512, 512, 256); self.d5 = cb(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.at4 = _AttentionBlock(256, 256, 128); self.d4 = cb(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.at3 = _AttentionBlock(128, 128, 64);  self.d3 = cb(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.at2 = _AttentionBlock(64, 64, 32);    self.d2 = cb(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        x5 = self.e5(self.pool(x4))
        d = self.up5(x5);  d = torch.cat([self.at5(d, x4), d], 1); d = self.d5(d)
        d = self.up4(d);   d = torch.cat([self.at4(d, x3), d], 1); d = self.d4(d)
        d = self.up3(d);   d = torch.cat([self.at3(d, x2), d], 1); d = self.d3(d)
        d = self.up2(d);   d = torch.cat([self.at2(d, x1), d], 1); d = self.d2(d)
        return self.out(d)


# SegNet

def _conv_block(in_ch, out_ch, n=2):
    layers = []
    for _ in range(n):
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """
    VGG-style encoder-decoder with max-pool unpooling.
    Pooling indices are reused in the decoder for precise spatial reconstruction.
    """
    def __init__(self, in_ch=3, out_ch=1, ch=(64, 128, 256, 512, 512)):
        super().__init__()
        self.enc1 = _conv_block(in_ch, ch[0], 2); self.p1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc2 = _conv_block(ch[0], ch[1], 2); self.p2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc3 = _conv_block(ch[1], ch[2], 3); self.p3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc4 = _conv_block(ch[2], ch[3], 3); self.p4 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc5 = _conv_block(ch[3], ch[4], 3); self.p5 = nn.MaxPool2d(2, 2, return_indices=True)
        self.u5 = nn.MaxUnpool2d(2, 2); self.dec5 = _conv_block(ch[4], ch[3], 3)
        self.u4 = nn.MaxUnpool2d(2, 2); self.dec4 = _conv_block(ch[3], ch[2], 3)
        self.u3 = nn.MaxUnpool2d(2, 2); self.dec3 = _conv_block(ch[2], ch[1], 3)
        self.u2 = nn.MaxUnpool2d(2, 2); self.dec2 = _conv_block(ch[1], ch[0], 2)
        self.u1 = nn.MaxUnpool2d(2, 2); self.dec1 = _conv_block(ch[0], ch[0], 2)
        self.clf = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x);  p1, i1 = self.p1(e1)
        e2 = self.enc2(p1); p2, i2 = self.p2(e2)
        e3 = self.enc3(p2); p3, i3 = self.p3(e3)
        e4 = self.enc4(p3); p4, i4 = self.p4(e4)
        e5 = self.enc5(p4); p5, i5 = self.p5(e5)
        d = self.dec5(self.u5(p5, i5, output_size=e5.size()))
        d = self.dec4(self.u4(d,  i4, output_size=e4.size()))
        d = self.dec3(self.u3(d,  i3, output_size=e3.size()))
        d = self.dec2(self.u2(d,  i2, output_size=e2.size()))
        d = self.dec1(self.u1(d,  i1, output_size=e1.size()))
        return self.clf(d)


# RefineNet

class _CRPBlock(nn.Module):
    """Chained Residual Pooling — captures multi-scale context."""
    def __init__(self, in_ch, out_ch, n_stages=2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 3, padding=1)
                                    for _ in range(n_stages)])
        self.relu  = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(x)
        path = x
        for conv in self.convs:
            path = conv(F.max_pool2d(path, 5, stride=1, padding=2))
            x    = x + path
        return x


class _RefineBlock(nn.Module):
    """RCU → CRP → fuse with upsampled residual."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.rcu  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.crp  = _CRPBlock(out_ch, out_ch)
        self.proj = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x, residual=None):
        x = self.proj(self.crp(self.rcu(x)))
        if residual is not None:
            x = x + F.interpolate(residual, size=x.shape[2:],
                                  mode="bilinear", align_corners=False)
        return x


class RefineNet(nn.Module):
    """
    ResNet-34 encoder + 4 RefineBlocks for multi-path refinement.
    Best-performing model in the paper (Dice 0.9665, IoU 0.9437).
    Used to generate predicted masks for the full dataset before IPP training.
    """
    def __init__(self):
        super().__init__()
        resnet       = torch.hub.load("pytorch/vision:v0.10.0", "resnet34",
                                      pretrained=True)
        self.layer0  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1   # 64 ch
        self.layer2  = resnet.layer2   # 128 ch
        self.layer3  = resnet.layer3   # 256 ch
        self.layer4  = resnet.layer4   # 512 ch
        self.r4   = _RefineBlock(512, 256)
        self.r3   = _RefineBlock(256, 256)
        self.r2   = _RefineBlock(128, 256)
        self.r1   = _RefineBlock(64,  256)
        self.head = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(self.maxpool(l0))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        r  = self.r1(l1, self.r2(l2, self.r3(l3, self.r4(l4))))
        return F.interpolate(self.head(r), size=x.shape[2:],
                             mode="bilinear", align_corners=False)


# Swin-UNet

class SwinUNet(nn.Module):
    """
    Swin-Tiny backbone + lightweight conv decoder.
    Input size: 224×224.
    Extracts last feature map (B, 768, 7, 7) and upsamples ×32.
    """
    def __init__(self):
        super().__init__()
        self.backbone = swin_tiny_patch4_window7_224(pretrained=True,
                                                     features_only=True)
        self.decoder  = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,  32,  3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32,  16,  3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16,  1,   1),
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        if feats.shape[1] != 768:          # NHWC → NCHW
            feats = feats.permute(0, 3, 1, 2)
        return self.decoder(feats)


# TransUNet

class TransUNet(nn.Module):
    """
    ViT-B/16 encoder + 4-stage CNN decoder.
    Input size: 224×224.
    Uses patch tokens (196 patches, 768-dim) reshaped to a 14×14 feature map.
    """
    def __init__(self):
        super().__init__()
        self.encoder      = vit_base_patch16_224(pretrained=True)
        self.encoder.head = nn.Identity()
        self.decoder      = nn.Sequential(
            nn.Conv2d(768, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(384, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(192, 96,  3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(96,  48,  3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(48,  1,   1),
        )

    def forward(self, x):
        tokens = self.encoder.patch_embed(x)
        tokens = tokens + self.encoder.pos_embed[:, 1:, :]
        tokens = self.encoder.pos_drop(tokens)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)
        fmap   = rearrange(tokens, "b (h w) c -> b c h w", h=14, w=14)
        return self.decoder(fmap)


# Factory

_REGISTRY = {
    "unetpp":       (_build_unetpp,       256),
    "deeplabv3":    (_build_deeplabv3,    256),
    "deeplabv3plus":(_build_deeplabv3plus,256),
    "attention_unet":(AttentionUNet,      256),
    "segnet":       (SegNet,              256),
    "refinenet":    (RefineNet,           256),
    "swin_unet":    (SwinUNet,            224),
    "transunet":    (TransUNet,           224),
}


def build_model(name: str):
    """
    Instantiate a segmentation model by name.

    Args:
        name: one of 'unetpp', 'deeplabv3', 'deeplabv3plus', 'attention_unet',
              'segnet', 'refinenet', 'swin_unet', 'transunet'

    Returns:
        (model, img_size) — img_size is 224 for transformer models, else 256
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)}")
    factory, img_size = _REGISTRY[name]
    return factory(), img_size