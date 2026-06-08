"""
models/efficientnet_v2.py
=========================
EfficientNetV2-S dilatih DARI NOL (no pretrained weights).
Arsitektur: EfficientNetV2-S backbone → GAP → FC → Dropout → Softmax

Tidak menggunakan:
    - pretrained ImageNet weights
    - Vision Transformer (ViT)
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

from configs.config import ModelConfig


# ─────────────────────────────────────────────
# 1. Custom Classifier Head
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Head classifier: GAP → Flatten → FC → Dropout → FC (logits)
    GAP sudah ada di dalam EfficientNetV2, tapi kita ganti head-nya.
    """

    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.SiLU(),                          # Swish activation (konsisten dengan EfficientNet)
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ─────────────────────────────────────────────
# 2. BUSIClassifier: EfficientNetV2-S dari Nol
# ─────────────────────────────────────────────

class BUSIClassifier(nn.Module):
    """
    Model klasifikasi BUSI menggunakan EfficientNetV2-S.

    Inisialisasi:
        - pretrained=False  → bobot random (dari nol)
        - Head baru dengan 3 kelas output
        - Global Average Pooling bawaan EfficientNetV2

    Architecture flow:
        Input (B, 3, 224, 224)
        → EfficientNetV2-S features (B, 1280, 7, 7)
        → AdaptiveAvgPool2d → (B, 1280)    [GAP]
        → Dropout(0.4)
        → Linear(1280 → 512) + SiLU + BN
        → Dropout(0.2)
        → Linear(512 → 3)
        → Output logits (B, 3)
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.num_classes = cfg.num_classes
        self.dropout_rate = cfg.dropout_rate

        # ── Load EfficientNetV2-S tanpa pretrained weights ──
        # weights=None → random initialization
        backbone = efficientnet_v2_s(weights=None)

        # Ambil feature extractor (semua layer kecuali classifier asli)
        self.features = backbone.features          # Conv stem + MBConv blocks
        self.avgpool  = backbone.avgpool           # AdaptiveAvgPool2d(1,1)

        # Dapatkan dimensi output dari backbone
        in_features = backbone.classifier[-1].in_features   # 1280

        # Ganti classifier head dengan head kustom
        self.classifier = ClassifierHead(
            in_features=in_features,
            num_classes=cfg.num_classes,
            dropout_rate=cfg.dropout_rate,
        )

        # Inisialisasi bobot (Xavier/He initialization)
        self._init_weights()

        print(f"\n[Model] BUSIClassifier initialized")
        print(f"  Backbone  : EfficientNetV2-S (from scratch, pretrained=False)")
        print(f"  Input     : 3 × 224 × 224")
        print(f"  Features  : {in_features}")
        print(f"  Classes   : {cfg.num_classes}")
        print(f"  Dropout   : {cfg.dropout_rate}")
        print(f"  Params    : {self.count_parameters():,}")

    def _init_weights(self):
        """Inisialisasi bobot semua layer secara eksplisit."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor (B, 3, 224, 224)
        Returns:
            logits: Tensor (B, num_classes)  ← raw logits, bukan softmax
        """
        # Feature extraction
        x = self.features(x)         # (B, 1280, 7, 7)
        x = self.avgpool(x)          # (B, 1280, 1, 1)  ← GAP
        x = torch.flatten(x, 1)      # (B, 1280)

        # Classification head
        x = self.classifier(x)       # (B, num_classes)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Ekstrak feature map sebelum GAP (untuk Grad-CAM)."""
        return self.features(x)

    def count_parameters(self) -> int:
        """Hitung jumlah trainable parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 3. Factory Function
# ─────────────────────────────────────────────

def build_model(cfg: Optional[ModelConfig] = None) -> BUSIClassifier:
    """
    Factory function untuk membuat model.

    Usage:
        model = build_model(cfg.model)
    """
    return BUSIClassifier(cfg)


# ─────────────────────────────────────────────
# 4. Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"\n[Sanity Check] Input: {dummy.shape} → Output: {out.shape}")
    assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"
    print("[Sanity Check] PASSED ✓")
