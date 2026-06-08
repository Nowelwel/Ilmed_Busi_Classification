"""
utils/evaluator.py
==================
Evaluasi model pada independent test set.

Metrik sesuai pipeline revisi:
    - Accuracy
    - Precision (per kelas + macro + weighted)
    - Recall    (per kelas + macro + weighted)
    - F1-Score  (per kelas + macro + weighted)
    - Confusion Matrix (untuk visualisasi)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# 1. Inference: Kumpulkan prediksi dari DataLoader
# ─────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """
    Jalankan inference pada seluruh DataLoader.

    Returns:
        all_preds : list prediksi kelas (int)
        all_labels: list ground truth (int)
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


# ─────────────────────────────────────────────
# 2. Hitung metrik evaluasi
# ─────────────────────────────────────────────

def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
    verbose: bool = True,
) -> Dict:
    """
    Hitung Accuracy, Precision, Recall, F1-Score.

    Args:
        all_preds  : Prediksi model (N,)
        all_labels : Ground truth label (N,)
        class_names: Nama kelas ['benign', 'malignant', 'normal']
        verbose    : Print hasil ke konsol

    Returns:
        metrics: dict berisi semua metrik + confusion_matrix
    """
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)

    # ── Accuracy ──
    accuracy = accuracy_score(labels_arr, preds_arr)

    # ── Precision, Recall, F1 per kelas ──
    precision_per, recall_per, f1_per, support = precision_recall_fscore_support(
        labels_arr, preds_arr,
        average=None,
        zero_division=0,
    )

    # ── Macro averages ──
    precision_macro = float(np.mean(precision_per))
    recall_macro    = float(np.mean(recall_per))
    f1_macro        = float(np.mean(f1_per))

    # ── Weighted averages ──
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels_arr, preds_arr,
        average="weighted",
        zero_division=0,
    )

    # ── Confusion Matrix ──
    cm = confusion_matrix(labels_arr, preds_arr)

    metrics = {
        # Overall
        "accuracy"          : float(accuracy),
        "precision_macro"   : precision_macro,
        "recall_macro"      : recall_macro,
        "f1_macro"          : f1_macro,
        "precision_weighted": float(precision_w),
        "recall_weighted"   : float(recall_w),
        "f1_weighted"       : float(f1_w),
        # Confusion matrix
        "confusion_matrix"  : cm,
        # Per-class breakdown
        "per_class": {
            cls: {
                "precision": float(precision_per[i]),
                "recall"   : float(recall_per[i]),
                "f1"       : float(f1_per[i]),
                "support"  : int(support[i]),
            }
            for i, cls in enumerate(class_names)
        },
    }

    if verbose:
        _print_metrics(metrics, class_names)

    return metrics


def _print_metrics(metrics: Dict, class_names: List[str]):
    """Print ringkasan metrik evaluasi ke konsol."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"{'EVALUATION RESULTS — INDEPENDENT TEST SET':^60}")
    print(sep)
    print(f"  Accuracy            : {metrics['accuracy']*100:>6.2f}%")
    print(f"  Precision (macro)   : {metrics['precision_macro']*100:>6.2f}%")
    print(f"  Recall    (macro)   : {metrics['recall_macro']*100:>6.2f}%")
    print(f"  F1-Score  (macro)   : {metrics['f1_macro']*100:>6.2f}%")
    print(f"  Precision (weighted): {metrics['precision_weighted']*100:>6.2f}%")
    print(f"  Recall    (weighted): {metrics['recall_weighted']*100:>6.2f}%")
    print(f"  F1-Score  (weighted): {metrics['f1_weighted']*100:>6.2f}%")
    print(sep)
    print(f"\n  Per-Class Breakdown:")
    header = f"  {'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1-Score':>9}  {'Support':>8}"
    print(header)
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}")
    for cls in class_names:
        pc = metrics["per_class"][cls]
        print(
            f"  {cls:>12}  {pc['precision']*100:>9.2f}%  "
            f"{pc['recall']*100:>7.2f}%  {pc['f1']*100:>8.2f}%  "
            f"{pc['support']:>8}"
        )
    print(sep)


# ─────────────────────────────────────────────
# 3. Full Evaluation Pipeline
# ─────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Entry point evaluasi: inference + compute semua metrik.

    Args:
        model      : Model PyTorch yang sudah dilatih
        test_loader: DataLoader untuk independent test set
        class_names: ['benign', 'malignant', 'normal']
        device     : torch.device (auto-detect jika None)

    Returns:
        results: dict {
            'metrics'   : dict semua metrik,
            'all_preds' : list prediksi,
            'all_labels': list ground truth,
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Evaluator] Running inference on test set (device={device})...")

    all_preds, all_labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(all_preds, all_labels, class_names)

    return {
        "metrics"   : metrics,
        "all_preds" : all_preds,
        "all_labels": all_labels,
    }
