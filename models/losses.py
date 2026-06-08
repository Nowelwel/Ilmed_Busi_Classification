"""
models/losses.py
================
Focal Loss untuk menangani class imbalance pada BUSI.

Sesuai pipeline revisi:
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    dengan α = 0.9, γ = 2

Referensi: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss untuk multi-class classification.

    Formula:
        FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Args:
        alpha    : Scalar weighting factor. Default 0.9 sesuai pipeline revisi.
        gamma    : Focusing parameter. Default 2.0 sesuai pipeline revisi.
        reduction: 'mean' | 'sum' | 'none'

    Behaviour:
        - Sampel yang mudah diklasifikasikan (p_t tinggi) mendapat bobot kecil
        - Sampel yang sulit (p_t rendah) mendapat bobot besar
        - α mengatur kontribusi keseluruhan loss terhadap class imbalance
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        print(f"[FocalLoss] α={alpha}, γ={gamma}, reduction={reduction}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : Logits (B, C) — raw output model, belum di-softmax
            targets: Ground truth label (B,) berisi integer kelas
        Returns:
            Scalar loss (jika reduction='mean')
        """
        # Cross-entropy per-sample (log-softmax + NLLLoss)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t: probabilitas kelas yang benar = exp(-CE)
        p_t = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^γ
        # Saat p_t mendekati 1 (mudah), focal_weight → 0 (loss dikecilkan)
        # Saat p_t mendekati 0 (sulit), focal_weight → 1 (loss dipertahankan)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Focal loss: α * (1 - p_t)^γ * CE
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_loss(alpha: float = 0.9, gamma: float = 2.0) -> nn.Module:
    """
    Factory function untuk membuat FocalLoss.

    Args:
        alpha: Weighting factor (default 0.9 sesuai pipeline revisi)
        gamma: Focusing parameter (default 2.0 sesuai pipeline revisi)

    Usage:
        criterion = build_loss(alpha=0.9, gamma=2.0)
    """
    return FocalLoss(alpha=alpha, gamma=gamma)
