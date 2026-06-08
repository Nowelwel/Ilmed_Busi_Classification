# BUSI Breast Cancer Classification
### EfficientNetV2-S · PyTorch · From Scratch · 3-Class

Pipeline klasifikasi ultrasound payudara berbasis **BUSI Dataset** menggunakan **EfficientNetV2-S** yang dilatih dari nol (tanpa pretrained weights, tanpa ViT).

---

## Struktur Project

```
busi_classification/
├── configs/
│   └── config.py             # Semua hyperparameter & konfigurasi
├── data/
│   └── dataset.py            # Dataset loader, augmentasi, stratified split
├── models/
│   ├── efficientnet_v2.py    # EfficientNetV2-S dari scratch
│   └── losses.py             # Focal Loss (α=0.9, γ=2)
├── utils/
│   ├── trainer.py            # Training engine (AMP, early stopping, checkpoint)
│   ├── evaluator.py          # Metrik: Accuracy, Precision, Recall, F1
│   ├── visualizer.py         # Training history + Confusion Matrix
│   └── gradcam.py            # Grad-CAM visualization
├── train.py                  # Entry point utama
├── predict.py                # Inference single image / folder
└── requirements.txt
```

---

## Dataset Setup

Unduh BUSI Dataset dan atur struktur folder:

```
dataset/
└── BUSI/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png    ← mask otomatis diabaikan
    │   └── ...
    ├── malignant/
    │   └── ...
    └── normal/
        └── ...
```

> Dataset BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Training dengan konfigurasi default
python train.py

# Custom path dan parameter
python train.py \
  --dataset_root dataset/BUSI \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir outputs/

# Evaluasi saja (skip training, gunakan checkpoint)
python train.py --skip_train
```

---

## Output

```
outputs/
├── checkpoints/
│   ├── best_model.pth        # Model dengan val_loss terbaik
│   └── last_model.pth        # Checkpoint epoch terakhir
├── figures/
│   ├── training_history.png  # Loss & accuracy curves
│   ├── confusion_matrix.png  # Normalized + raw count
│   └── gradcam.png           # Grad-CAM heatmap overlay
├── training_history.json
└── test_metrics.json
```

---

## Inference

```bash
# Satu gambar
python predict.py --image path/to/ultrasound.png

# Batch folder
python predict.py --folder dataset/BUSI/malignant/ --output_csv results.csv
```

---

## Arsitektur Model

```
Input (B × 3 × 224 × 224)
    │
    ▼  Grayscale → RGB via channel replication
EfficientNetV2-S Feature Extractor
(Fused-MBConv + MBConv blocks, weights random init)
    │
    ▼
AdaptiveAvgPool2d(1,1)   ← Global Average Pooling
    │
    ▼  (B × 1280)
Dropout(0.4)
    │
Linear(1280 → 512) + SiLU + BatchNorm1d
    │
Dropout(0.2)
    │
Linear(512 → 3)
    │
    ▼
Softmax → [P(benign), P(malignant), P(normal)]
```

**Inisialisasi bobot:** Kaiming Normal (Conv2d), Truncated Normal (Linear).
**Tidak ada pretrained weights. Tidak ada ViT.**

---

## Pipeline Lengkap

```
BUSI Dataset (780 images PNG)
    │
    ▼
Load Image → convert("RGB") → Resize 224×224
    │
    ├─ Training set only:
    │    RandomHorizontalFlip(p=0.5)
    │    RandomRotation(±15°)
    │    RandomAffine(translate=±5%, scale=90–110%)
    │    ColorJitter(brightness=0.2, contrast=0.2)
    │
    ▼ Stratified Split
┌───────────────────────────────┐
│  80%  Train + Validation      │
│  20%  Independent Test Set    │ ← No augmentasi
└───────────────────────────────┘
    │
    ▼
EfficientNetV2-S (from scratch, random init)
    ↓ GAP → FC → Dropout → Softmax
    │
    ▼ AdamW + CosineAnnealingLR
Focal Loss (α=0.9, γ=2) + Early Stopping
    │
    ▼
Evaluation: Accuracy · Precision · Recall · F1-Score
    │
    ▼
Confusion Matrix + Grad-CAM Visualization
```

---

## Konfigurasi Default

| Parameter              | Nilai                |
|------------------------|----------------------|
| Input size             | 224 × 224 × 3        |
| Batch size             | 32                   |
| Learning rate          | 1 × 10⁻⁴             |
| Optimizer              | AdamW                |
| Scheduler              | CosineAnnealingLR    |
| Loss function          | Focal Loss           |
| Focal α                | 0.9                  |
| Focal γ                | 2.0                  |
| Max epochs             | 50                   |
| Early stop patience    | 10 epochs            |
| Dropout                | 0.4                  |
| Split (train/val/test) | ~65% / ~15% / 20%    |

---

## Kelas Dataset

| Label | Kelas     | Jumlah (BUSI) |
|-------|-----------|---------------|
| 0     | Benign    | 437           |
| 1     | Malignant | 210           |
| 2     | Normal    | 133           |

---

## Reproducibility

Semua random seed dikontrol melalui `set_seed(42)`:
- Python `random`, NumPy, PyTorch (CPU + CUDA)
- `cudnn.deterministic = True`
- Stratified split dengan `random_state=42`
