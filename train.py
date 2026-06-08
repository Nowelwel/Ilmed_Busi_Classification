"""
train.py
========
Entry point utama untuk pipeline klasifikasi BUSI.

Alur pipeline sesuai revisi:
    BUSI Dataset → Preprocessing → Augmentation (train only)
    → Stratified Split (80/20) → EfficientNetV2-S (from scratch)
    → GAP → FC → Dropout → Softmax
    → Evaluation: Accuracy, Precision, Recall, F1
    → Visualisasi: Confusion Matrix + Grad-CAM

Usage:
    python train.py
    python train.py --dataset_root /path/to/BUSI --epochs 50
    python train.py --skip_train   # evaluasi dari checkpoint
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import cfg
from data.dataset import build_dataloaders
from models.efficientnet_v2 import build_model
from models.losses import build_loss
from utils.trainer import Trainer
from utils.evaluator import evaluate_model
from utils.visualizer import plot_training_history, plot_confusion_matrix
from utils.gradcam import visualize_gradcam


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set semua random seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BUSI Breast Cancer Classification — EfficientNetV2-S from Scratch"
    )
    parser.add_argument(
        "--dataset_root", type=str, default=cfg.data.dataset_root,
        help="Path ke folder BUSI (berisi subdir benign/, malignant/, normal/)"
    )
    parser.add_argument("--epochs",      type=int,   default=cfg.training.num_epochs)
    parser.add_argument("--batch_size",  type=int,   default=cfg.training.batch_size)
    parser.add_argument("--lr",          type=float, default=cfg.training.learning_rate)
    parser.add_argument("--seed",        type=int,   default=cfg.data.random_seed)
    parser.add_argument("--num_workers", type=int,   default=cfg.training.num_workers)
    parser.add_argument("--output_dir",  type=str,   default=cfg.eval.output_dir)
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Lewati training, langsung evaluasi menggunakan checkpoint tersimpan"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 0. Setup konfigurasi & seed ──
    set_seed(args.seed)

    cfg.data.dataset_root       = args.dataset_root
    cfg.training.num_epochs     = args.epochs
    cfg.training.batch_size     = args.batch_size
    cfg.training.learning_rate  = args.lr
    cfg.data.random_seed        = args.seed
    cfg.training.num_workers    = args.num_workers
    cfg.eval.output_dir         = args.output_dir
    cfg.eval.figures_dir        = os.path.join(args.output_dir, "figures")
    cfg.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    cfg.create_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BUSI Breast Cancer Classification Pipeline")
    print(f"  Model    : EfficientNetV2-S (from scratch, no pretrained)")
    print(f"  Device   : {device}")
    print(f"  Dataset  : {cfg.data.dataset_root}")
    print(f"  Epochs   : {cfg.training.num_epochs}")
    print(f"  Batch    : {cfg.training.batch_size}")
    print(f"  LR       : {cfg.training.learning_rate}")
    print(f"{'='*60}")

    # ── 1. Data: scan, split, build DataLoaders ──
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg_data=cfg.data,
        cfg_aug=cfg.augmentation,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # ── 2. Model: EfficientNetV2-S dari nol ──
    model = build_model(cfg.model)
    model = model.to(device)

    # ── 3. Loss: Focal Loss α=0.9, γ=2 ──
    criterion = build_loss(
        alpha=cfg.training.focal_alpha,   # 0.9
        gamma=cfg.training.focal_gamma,   # 2.0
    )

    # ── 4. Training ──
    history = {}

    if not args.skip_train:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            cfg=cfg.training,
            device=device,
        )
        history = trainer.train()

        # Simpan training history ke JSON
        history_path = os.path.join(cfg.eval.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[Main] Training history saved → {history_path}")

        # Visualisasi loss & accuracy curves
        plot_training_history(
            history,
            os.path.join(cfg.eval.figures_dir, "training_history.png"),
        )

    else:
        # Load checkpoint best model untuk evaluasi langsung
        ckpt_path = os.path.join(cfg.training.checkpoint_dir, cfg.training.best_model_name)
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            history = ckpt.get("history", {})
            print(f"[Main] Checkpoint loaded → {ckpt_path}")
            if history:
                plot_training_history(
                    history,
                    os.path.join(cfg.eval.figures_dir, "training_history.png"),
                )
        else:
            print(f"[Main] WARNING: Checkpoint tidak ditemukan di {ckpt_path}")

    # ── 5. Evaluasi pada Independent Test Set ──
    print(f"\n{'='*60}")
    print(f"  EVALUASI — INDEPENDENT TEST SET")
    print(f"{'='*60}")

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
    )
    metrics = results["metrics"]

    # Simpan metrik ke JSON
    metrics_serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }
    metrics_serializable["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    metrics_path = os.path.join(cfg.eval.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n[Main] Metrics saved → {metrics_path}")

    # ── 6. Visualisasi ──
    print(f"\n{'='*60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # Confusion Matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        os.path.join(cfg.eval.figures_dir, "confusion_matrix.png"),
    )

    # Grad-CAM pada layer konvolusional terakhir EfficientNetV2-S
    visualize_gradcam(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        target_layer_name=cfg.eval.gradcam_target_layer,
        num_samples=cfg.eval.gradcam_num_samples,
        output_path=os.path.join(cfg.eval.figures_dir, "gradcam.png"),
        device=device,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    # ── 7. Summary ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE SELESAI")
    print(f"  Output dir   : {cfg.eval.output_dir}")
    print(f"  Figures dir  : {cfg.eval.figures_dir}")
    print(f"  Accuracy     : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision (M): {metrics['precision_macro']*100:.2f}%")
    print(f"  Recall    (M): {metrics['recall_macro']*100:.2f}%")
    print(f"  F1-Score  (M): {metrics['f1_macro']*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
