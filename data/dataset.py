"""
data/dataset.py
===============
Dataset loader, preprocessing, augmentasi, dan data splitting
untuk BUSI (Breast Ultrasound Images Dataset).

Pipeline revisi:
    - Augmentasi training: horizontal flip, random rotation,
      affine transformation, brightness-contrast adjustment
    - Validation & test: TANPA augmentasi
    - Split: 80% train+val / 20% independent test (stratified)

Dataset diambil otomatis dari Kaggle via kagglehub.
Tidak perlu download manual — cukup setup Kaggle API credentials.

Struktur folder BUSI yang diharapkan (auto-created oleh KaggleDownloader):
    dataset/BUSI/
        benign/
            benign (1).png
            benign (1)_mask.png
            ...
        malignant/
            malignant (1).png
            ...
        normal/
            normal (1).png
            ...
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from configs.config import AugmentationConfig, DataConfig
from data.kaggle_downloader import download_busi_dataset


# ─────────────────────────────────────────────
# 1. HELPER: Scan dan filter file gambar BUSI
# ─────────────────────────────────────────────

def scan_busi_dataset(root: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """
    Scan folder BUSI dan kembalikan daftar path + label.
    File mask (mengandung '_mask') diabaikan secara otomatis.

    Returns:
        image_paths: list path gambar
        labels     : list integer label (sesuai urutan class_names)
    """
    image_paths: List[str] = []
    labels: List[int] = []

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset root tidak ditemukan: {root}\n"
            "Pastikan struktur folder: dataset/BUSI/benign/, malignant/, normal/"
        )

    for label_idx, cls in enumerate(class_names):
        class_dir = root_path / cls
        if not class_dir.exists():
            print(f"[WARN] Folder tidak ditemukan: {class_dir}")
            continue

        files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        img_files = [f for f in files if "_mask" not in f.name.lower()]

        print(f"  [{cls:>10}] {len(img_files):>4} images found")
        image_paths.extend([str(f) for f in img_files])
        labels.extend([label_idx] * len(img_files))

    assert len(image_paths) > 0, "Tidak ada gambar yang ditemukan! Periksa path dataset."
    return image_paths, labels


# ─────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────

def build_train_transform(cfg_data: DataConfig, cfg_aug: AugmentationConfig) -> transforms.Compose:
    """
    Transform untuk training dengan augmentasi sesuai pipeline revisi:
        1. Resize 224×224
        2. Horizontal flip
        3. Random rotation
        4. Random affine (translate + scale)
        5. Brightness-contrast adjustment
        6. ToTensor
        7. Normalize
    """
    return transforms.Compose([
        transforms.Resize(cfg_data.image_size),
        transforms.RandomHorizontalFlip(p=cfg_aug.horizontal_flip_p),
        transforms.RandomRotation(degrees=cfg_aug.rotation_degrees),
        transforms.RandomAffine(
            degrees=0,
            translate=cfg_aug.affine_translate,
            scale=cfg_aug.affine_scale,
        ),
        transforms.ColorJitter(
            brightness=cfg_aug.brightness,
            contrast=cfg_aug.contrast,
            saturation=0.0,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_data.mean, std=cfg_data.std),
    ])


def build_eval_transform(cfg_data: DataConfig) -> transforms.Compose:
    """
    Transform untuk validation dan test — TANPA augmentasi.
    Hanya resize, ToTensor, dan normalize.
    """
    return transforms.Compose([
        transforms.Resize(cfg_data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_data.mean, std=cfg_data.std),
    ])


# ─────────────────────────────────────────────
# 3. PYTORCH DATASET
# ─────────────────────────────────────────────

class BUSIDataset(Dataset):
    """
    PyTorch Dataset untuk BUSI.
    Gambar grayscale otomatis dikonversi ke RGB (3 channel)
    melalui Image.convert("RGB") — channel replication.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ["benign", "malignant", "normal"]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load dan konversi grayscale → RGB (channel replication)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Hitung jumlah sampel per kelas."""
        counts = {}
        for i, cls in enumerate(self.class_names):
            counts[cls] = self.labels.count(i)
        return counts


# ─────────────────────────────────────────────
# 4. DATA SPLITTING (Stratified)
# ─────────────────────────────────────────────

def split_dataset(
    image_paths: List[str],
    labels: List[int],
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List, List, List, List, List, List]:
    """
    Stratified split sesuai pipeline revisi:
        Step 1: 80% train+val  /  20% independent test
        Step 2: Dari 80%, pisahkan train dan validation

    Proporsi final (dari total dataset):
        Train : ~65%
        Val   : ~15%
        Test  : ~20%

    Returns:
        (train_paths, val_paths, test_paths,
         train_labels, val_labels, test_labels)
    """
    # Step 1: Pisahkan 20% sebagai independent test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed,
    )

    # Step 2: Dari sisa 80%, pisahkan validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=adjusted_val_size,
        stratify=train_val_labels,
        random_state=random_seed,
    )

    total = len(image_paths)
    print(
        f"\n[DataSplit] Total: {total} | "
        f"Train: {len(train_paths)} ({len(train_paths)/total*100:.0f}%) | "
        f"Val: {len(val_paths)} ({len(val_paths)/total*100:.0f}%) | "
        f"Test: {len(test_paths)} ({len(test_paths)/total*100:.0f}%)"
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


# ─────────────────────────────────────────────
# 5. DATALOADER BUILDER
# ─────────────────────────────────────────────

def build_dataloaders(
    cfg_data: DataConfig,
    cfg_aug: AugmentationConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Entry point utama: download dataset (Kaggle) → scan → split → build DataLoaders.

    Dataset di-download otomatis dari Kaggle menggunakan kagglehub.
    Jika sudah ada di lokal, download akan di-skip.

    Setup Kaggle API sebelum menjalankan:
        Letakkan ~/.kaggle/kaggle.json
        ATAU set env: KAGGLE_USERNAME + KAGGLE_KEY

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # ── Auto-download dari Kaggle jika belum ada ──
    dataset_root = download_busi_dataset(target_dir=cfg_data.dataset_root)
    cfg_data.dataset_root = dataset_root  # update path jika berubah

    print("\n[Dataset] Scanning BUSI dataset...")
    image_paths, labels = scan_busi_dataset(cfg_data.dataset_root, cfg_data.class_names)

    # Stratified split
    (train_paths, val_paths, test_paths,
     train_labels, val_labels, test_labels) = split_dataset(
        image_paths, labels,
        test_size=cfg_data.test_size,
        val_size=cfg_data.val_size,
        random_seed=cfg_data.random_seed,
    )

    # Transforms: augmentasi hanya untuk train
    train_tf = build_train_transform(cfg_data, cfg_aug)
    eval_tf  = build_eval_transform(cfg_data)

    # Datasets
    train_ds = BUSIDataset(train_paths, train_labels, train_tf, cfg_data.class_names)
    val_ds   = BUSIDataset(val_paths,   val_labels,   eval_tf,  cfg_data.class_names)
    test_ds  = BUSIDataset(test_paths,  test_labels,  eval_tf,  cfg_data.class_names)

    # Print distribusi kelas per split
    print(f"\n[Dataset] Class distribution per split:")
    for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        counts = ds.get_class_counts()
        print(f"  {split_name:>5}: {counts}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, cfg_data.class_names
