# BUSI Breast Cancer Classification
## EfficientNetV2-S · Focal Loss · Grad-CAM

Pipeline klasifikasi kanker payudara berbasis ultrasound menggunakan
**EfficientNetV2-S dilatih dari nol** (tanpa pretrained weights).

Dataset di-download **otomatis dari Kaggle** — tidak perlu download manual.

---

## Arsitektur Pipeline

```
BUSI Dataset (Kaggle)
    ↓ kagglehub (auto-download)
Preprocessing (224×224, grayscale→RGB)
    ↓
Augmentasi Training (flip, rotation, affine, brightness-contrast)
    ↓
Stratified Split — Train 65% / Val 15% / Test 20%
    ↓
EfficientNetV2-S (from scratch)
    → features → GAP → Dropout → FC(1280→512) → FC(512→3)
    ↓
Focal Loss (α=0.9, γ=2)
    ↓
Evaluasi: Accuracy, Precision, Recall, F1-Score (macro)
    ↓
Visualisasi: Confusion Matrix + Grad-CAM
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API (WAJIB untuk download dataset)

Dataset diambil dari:
> https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

**Opsi A — File `kaggle.json` (direkomendasikan untuk lokal):**
```bash
# 1. Buka https://www.kaggle.com/settings
# 2. Scroll ke bagian "API" → klik "Create New Token"
# 3. File kaggle.json akan terdownload otomatis

# Linux / Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Letakkan di: C:\Users\<username>\.kaggle\kaggle.json
```

**Opsi B — Environment variables:**
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

**Opsi C — Google Colab:**
```python
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## Menjalankan Training

```bash
# Training default (dataset auto-download dari Kaggle)
python train.py

# Custom path, epoch, batch size
python train.py --epochs 100 --batch_size 16 --lr 5e-5

# Skip training, langsung evaluasi dari checkpoint
python train.py --skip_train

# Gunakan dataset lokal yang sudah ada (opsional)
python train.py --dataset_root /path/to/BUSI
```

Saat pertama kali dijalankan, dataset akan otomatis di-download ke `dataset/BUSI/`.
Download berikutnya akan di-skip (cached).

---

## Prediksi Gambar Baru

```bash
# Single image
python predict.py --image path/to/image.png --checkpoint outputs/checkpoints/best_model.pth

# Batch prediction (folder)
python predict.py --image path/to/folder/ --checkpoint outputs/checkpoints/best_model.pth
```

---

## Struktur Proyek

```
busi_classification/
├── configs/
│   └── config.py              # Semua hyperparameter & konfigurasi
├── data/
│   ├── dataset.py             # Dataset, transforms, DataLoaders
│   └── kaggle_downloader.py   # ★ Auto-download dari Kaggle
├── models/
│   ├── efficientnet_v2.py     # BUSIClassifier (EfficientNetV2-S from scratch)
│   └── losses.py              # Focal Loss (α=0.9, γ=2)
├── utils/
│   ├── trainer.py             # Training loop + early stopping
│   ├── evaluator.py           # Metrics evaluation
│   ├── visualizer.py          # Plot training curves & confusion matrix
│   └── gradcam.py             # Grad-CAM visualization
├── train.py                   # ★ Entry point training
├── predict.py                 # Inference / prediksi gambar baru
├── requirements.txt
└── README.md
```

---

## Hyperparameter Default

| Parameter         | Value                    |
|-------------------|--------------------------|
| Backbone          | EfficientNetV2-S         |
| Pretrained        | False (from scratch)     |
| Input size        | 224 × 224                |
| Batch size        | 32                       |
| Optimizer         | AdamW                    |
| Learning rate     | 1e-4                     |
| Scheduler         | CosineAnnealingLR        |
| Epochs            | 50                       |
| Early stopping    | patience=10              |
| Loss              | Focal Loss (α=0.9, γ=2)  |
| Dropout           | 0.4                      |

---

## Output

Setelah training selesai, hasil tersimpan di `outputs/`:

```
outputs/
├── checkpoints/
│   └── best_model.pth          # Model terbaik (val_loss terendah)
├── figures/
│   ├── training_history.png    # Loss & accuracy curves
│   ├── confusion_matrix.png    # Confusion matrix test set
│   └── gradcam.png             # Grad-CAM visualizations
├── training_history.json       # Riwayat training (loss, acc per epoch)
└── test_metrics.json           # Metrik evaluasi test set
```

---

## Dataset

**BUSI — Breast Ultrasound Images Dataset**
- Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- 780 gambar ultrasound payudara (+ mask segmentasi)
- 3 kelas: benign (437), malignant (210), normal (133)
- Format: PNG grayscale
- Sumber: Al-Dhabyani W. et al., *Data in Brief* (2020)
