"""
utils/trainer.py
================
Training engine yang lengkap:
    - Train loop dengan mixed precision (AMP)
    - Validation loop
    - Early stopping
    - Checkpoint saving/loading
    - Logging training history
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from configs.config import TrainingConfig


# ─────────────────────────────────────────────
# 1. Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Monitor validation loss dan hentikan training jika tidak ada improvement
    selama `patience` epoch berturut-turut.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns True jika training harus dihentikan.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# ─────────────────────────────────────────────
# 2. Metrics per Epoch
# ─────────────────────────────────────────────

def compute_epoch_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int = 3,
) -> Dict[str, float]:
    """Hitung accuracy per epoch (simple, cepat)."""
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    return {"accuracy": accuracy}


# ─────────────────────────────────────────────
# 3. Trainer Class
# ─────────────────────────────────────────────

class Trainer:
    """
    Engine training full-featured untuk BUSIClassifier.

    Features:
        - AMP (Automatic Mixed Precision) untuk GPU
        - AdamW + CosineAnnealingLR scheduler
        - Early stopping
        - Best model checkpoint
        - History logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        cfg: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.cfg = cfg

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)

        print(f"\n[Trainer] Device: {self.device}")

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.t_max,
            eta_min=cfg.eta_min,
        )

        # AMP Scaler (hanya aktif di CUDA)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
        )

        # History
        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr":         [],
        }

        # Checkpoint
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    # ── Train one epoch ──

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping untuk stabilitas
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_epoch_metrics(all_preds, all_labels)
        return avg_loss, metrics["accuracy"]

    # ── Validate one epoch ──

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_epoch_metrics(all_preds, all_labels)
        return avg_loss, metrics["accuracy"]

    # ── Save checkpoint ──

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }
        if is_best:
            path = self.checkpoint_dir / self.cfg.best_model_name
            torch.save(checkpoint, path)

        # Simpan juga checkpoint terbaru
        torch.save(checkpoint, self.checkpoint_dir / "last_model.pth")

    # ── Load best checkpoint ──

    def load_best_model(self):
        path = self.checkpoint_dir / self.cfg.best_model_name
        if not path.exists():
            print(f"[Trainer] Checkpoint tidak ditemukan: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Trainer] Best model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    # ── Main train loop ──

    def train(self) -> Dict[str, List]:
        """
        Jalankan training loop lengkap.
        Returns history dict.
        """
        print(f"\n{'='*60}")
        print(f"[Trainer] Start training | Epochs: {self.cfg.num_epochs}")
        print(f"[Trainer] LR: {self.cfg.learning_rate} | Batch: {self.cfg.batch_size}")
        print(f"[Trainer] Early stop patience: {self.cfg.early_stopping_patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.cfg.num_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss, is_best)

            # Print
            best_marker = " ← BEST" if is_best else ""
            print(
                f"Epoch [{epoch:>3}/{self.cfg.num_epochs}] "
                f"| Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%"
                f"  |  Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%"
                f"  |  LR: {current_lr:.2e}  [{elapsed:.1f}s]{best_marker}"
            )

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n[EarlyStopping] Stopped at epoch {epoch}. Best val_loss: {self.best_val_loss:.4f}")
                break

        print(f"\n[Trainer] Training finished. Best val_loss: {self.best_val_loss:.4f}")
        self.load_best_model()
        return self.history
