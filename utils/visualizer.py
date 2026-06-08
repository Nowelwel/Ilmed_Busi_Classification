"""
utils/visualizer.py
===================
Visualisasi hasil training dan evaluasi.

Sesuai pipeline revisi, output visualisasi adalah:
    1. Training history (loss & accuracy curves)
    2. Confusion Matrix (normalized + raw count)

Grad-CAM diimplementasikan terpisah di utils/gradcam.py.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── Tema visual konsisten ──
DARK_BG   = "#0f0f1a"
CARD_BG   = "#1a1a2e"
TEXT_COL  = "#e0e0e0"
CLASS_COLORS = ["#4CC9F0", "#F72585", "#4AD66D"]   # Benign, Malignant, Normal


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor" : DARK_BG,
        "axes.facecolor"   : CARD_BG,
        "axes.edgecolor"   : "#333355",
        "axes.labelcolor"  : TEXT_COL,
        "xtick.color"      : TEXT_COL,
        "ytick.color"      : TEXT_COL,
        "text.color"       : TEXT_COL,
        "grid.color"       : "#333355",
        "grid.alpha"       : 0.4,
        "font.family"      : "DejaVu Sans",
    })


def _save_fig(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Visualizer] Saved → {output_path}")


# ─────────────────────────────────────────────
# 1. Training History
# ─────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List],
    output_path: str = "outputs/figures/training_history.png",
):
    """
    Plot loss dan accuracy curves selama training.

    Panel kiri : Train Loss vs Val Loss
    Panel kanan: Train Accuracy vs Val Accuracy
    """
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss Curve ──
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#4CC9F0", lw=2.0,
            label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"],   color="#F72585", lw=2.0,
            label="Val Loss",   marker="s", markersize=3, ls="--")
    ax.set_title("Loss Curve", fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
    ax.legend(framealpha=0.3, facecolor=CARD_BG, labelcolor=TEXT_COL)
    ax.grid(True, alpha=0.3)

    # Annotate best val loss
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val   = min(history["val_loss"])
    ax.axvline(best_epoch, color="#FFD700", lw=1.2, ls=":", alpha=0.7)
    ax.annotate(
        f"Best\nepoch {best_epoch}\n{best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + max(1, len(epochs) * 0.05), best_val),
        color="#FFD700", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.0),
    )

    # ── Accuracy Curve ──
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in history["train_acc"]], color="#4CC9F0",
            lw=2.0, label="Train Acc", marker="o", markersize=3)
    ax.plot(epochs, [v * 100 for v in history["val_acc"]],   color="#F72585",
            lw=2.0, label="Val Acc",   marker="s", markersize=3, ls="--")
    ax.set_title("Accuracy Curve", fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(framealpha=0.3, facecolor=CARD_BG, labelcolor=TEXT_COL)
    ax.grid(True, alpha=0.3)

    # Annotate best val acc
    best_acc_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val_acc   = max(history["val_acc"]) * 100
    ax.axvline(best_acc_epoch, color="#FFD700", lw=1.2, ls=":", alpha=0.7)

    plt.suptitle(
        "Training History — EfficientNetV2-S (From Scratch)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, output_path)


# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/figures/confusion_matrix.png",
):
    """
    Plot confusion matrix: normalized (kiri) + raw count (kanan).

    Normalized menunjukkan recall per kelas;
    Raw count memudahkan pembacaan jumlah sampel.
    """
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    # Normalisasi per baris (recall-oriented)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.where(row_sums > 0, row_sums, 1.0)

    panels = [
        (axes[0], cm_norm, "Normalized (Row %)", ".2%"),
        (axes[1], cm,      "Raw Count",           "d"),
    ]

    for ax, data, title, fmt in panels:
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.8,
            linecolor="#0f0f1a",
            cbar_kws={"shrink": 0.82},
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    # Per-class accuracy baris footer
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    footer = "Per-class Recall:  " + "   |   ".join(
        f"{cls} = {r*100:.1f}%"
        for cls, r in zip(class_names, per_class_recall)
    )
    fig.text(
        0.5, -0.03, footer,
        ha="center", fontsize=10, color="#4CC9F0", fontweight="bold",
    )

    plt.suptitle(
        "Confusion Matrix — Independent Test Set",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, output_path)
