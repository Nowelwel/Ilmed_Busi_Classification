"""
configs/config.py
=================
Centralized configuration untuk pipeline klasifikasi BUSI.
Semua hyperparameter, path, dan setting ada di sini.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class DataConfig:
    """Konfigurasi dataset dan preprocessing."""
    dataset_root: str = "dataset/BUSI"          # Root folder dataset BUSI
    class_names: List[str] = field(default_factory=lambda: ["benign", "malignant", "normal"])
    image_size: Tuple[int, int] = (224, 224)    # Input size model
    num_channels: int = 3                        # RGB (grayscale → 3ch)

    # Data split
    test_size: float = 0.20                      # 20% independent test
    val_size: float = 0.15                       # 15% dari train+val = validation
    random_seed: int = 42

    # Normalisasi (ImageNet-style tapi dihitung ulang dari BUSI)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)


@dataclass
class AugmentationConfig:
    """
    Konfigurasi augmentasi data training.
    Sesuai pipeline revisi: horizontal flip, random rotation,
    affine transformation, brightness-contrast adjustment.
    Augmentasi HANYA diterapkan pada training set.
    """
    # Horizontal flip
    horizontal_flip_p: float = 0.5

    # Random rotation
    rotation_degrees: int = 15

    # Random affine (translate + scale)
    affine_translate: Tuple[float, float] = (0.05, 0.05)  # max ±5% shift
    affine_scale: Tuple[float, float] = (0.9, 1.1)         # scale 90–110%

    # Brightness & contrast (ColorJitter — tanpa saturation/hue untuk grayscale)
    brightness: float = 0.2
    contrast: float = 0.2


@dataclass
class ModelConfig:
    """Konfigurasi arsitektur model."""
    backbone: str = "efficientnet_v2_s"          # EfficientNetV2-S from scratch
    num_classes: int = 3
    dropout_rate: float = 0.4
    pretrained: bool = False                      # Dilatih dari nol!
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Konfigurasi proses training."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"                    # CosineAnnealingLR
    t_max: int = 50
    eta_min: float = 1e-6

    # Training loop
    batch_size: int = 32
    num_epochs: int = 50
    num_workers: int = 4

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    monitor_metric: str = "val_loss"

    # Focal Loss — α=0.9, γ=2 sesuai pipeline revisi
    focal_alpha: float = 0.9
    focal_gamma: float = 2.0

    # Checkpoint
    checkpoint_dir: str = "outputs/checkpoints"
    best_model_name: str = "best_model.pth"


@dataclass
class EvalConfig:
    """
    Konfigurasi evaluasi dan visualisasi.
    Sesuai pipeline revisi: Confusion Matrix + Grad-CAM.
    Metrik: Accuracy, Precision, Recall, F1-Score.
    """
    # Grad-CAM
    gradcam_target_layer: str = "features.7"     # Layer konv terakhir EfficientNetV2-S
    gradcam_num_samples: int = 12                # Jumlah sampel visualisasi Grad-CAM

    # Output dirs
    output_dir: str = "outputs"
    figures_dir: str = "outputs/figures"


@dataclass
class Config:
    """Master config yang menggabungkan semua sub-config."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def create_dirs(self):
        """Buat semua direktori output yang diperlukan."""
        dirs = [
            self.training.checkpoint_dir,
            self.eval.output_dir,
            self.eval.figures_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("[Config] Output directories created.")


# Singleton config
cfg = Config()
