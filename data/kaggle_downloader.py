"""
data/kaggle_downloader.py
=========================
Utility untuk download dataset BUSI langsung dari Kaggle
menggunakan Kaggle API (kagglehub) — tanpa perlu download manual.

Dataset : aryashah2k/breast-ultrasound-images-dataset
Link    : https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

Struktur dataset setelah download:
    <cache_dir>/
        benign/
            benign (1).png
            benign (1)_mask.png
            ...
        malignant/
        normal/

Setup Kaggle API (pilih salah satu):
    Opsi A — File kaggle.json (direkomendasikan):
        1. Buka https://www.kaggle.com/settings
        2. Scroll ke "API" → klik "Create New Token"
        3. File kaggle.json akan terdownload
        4. Letakkan di: ~/.kaggle/kaggle.json
        5. Jalankan: chmod 600 ~/.kaggle/kaggle.json

    Opsi B — Environment variables:
        export KAGGLE_USERNAME="your_username"
        export KAGGLE_KEY="your_api_key"

    Opsi C — Colab / Kaggle Notebook:
        from google.colab import files
        files.upload()  # upload kaggle.json
        !mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
"""

import os
import shutil
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Konstanta dataset Kaggle
# ─────────────────────────────────────────────
KAGGLE_DATASET_HANDLE = "aryashah2k/breast-ultrasound-images-dataset"
EXPECTED_CLASSES      = ["benign", "malignant", "normal"]
DEFAULT_TARGET_DIR    = "dataset/BUSI"


# ─────────────────────────────────────────────
# Helper: cek ketersediaan kagglehub
# ─────────────────────────────────────────────
def _check_kagglehub() -> bool:
    """Return True jika kagglehub terinstall dan bisa diimport."""
    try:
        import kagglehub  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────
# Helper: temukan root folder dataset dari cache
# ─────────────────────────────────────────────
def _find_busi_root(base_path: str) -> Optional[str]:
    """
    Cari folder yang mengandung ketiga subfolder BUSI
    (benign, malignant, normal) secara rekursif.

    Args:
        base_path: Root hasil download kagglehub

    Returns:
        Path string ke folder BUSI, atau None jika tidak ditemukan
    """
    base = Path(base_path)

    # Cek direktori saat ini
    subdirs = {d.name.lower() for d in base.iterdir() if d.is_dir()}
    if all(cls in subdirs for cls in EXPECTED_CLASSES):
        return str(base)

    # Cari satu level lebih dalam
    for child in base.iterdir():
        if child.is_dir():
            subdirs2 = {d.name.lower() for d in child.iterdir() if d.is_dir()}
            if all(cls in subdirs2 for cls in EXPECTED_CLASSES):
                return str(child)

    return None


# ─────────────────────────────────────────────
# Main: download_busi_dataset
# ─────────────────────────────────────────────
def download_busi_dataset(
    target_dir: str = DEFAULT_TARGET_DIR,
    force_download: bool = False,
) -> str:
    """
    Download dataset BUSI dari Kaggle menggunakan kagglehub.

    Jika dataset sudah ada di target_dir dan force_download=False,
    fungsi ini langsung return path tanpa re-download.

    Args:
        target_dir    : Folder tujuan (default: "dataset/BUSI")
        force_download: Paksa re-download meski dataset sudah ada

    Returns:
        str: Path absolut ke folder BUSI yang siap digunakan

    Raises:
        ImportError   : Jika kagglehub belum terinstall
        RuntimeError  : Jika download gagal atau struktur folder tidak sesuai
    """
    target_path = Path(target_dir).resolve()

    # ── Cek apakah dataset sudah ada ──
    if not force_download and target_path.exists():
        existing_subdirs = {d.name.lower() for d in target_path.iterdir() if d.is_dir()}
        if all(cls in existing_subdirs for cls in EXPECTED_CLASSES):
            # Hitung jumlah gambar
            counts = {}
            for cls in EXPECTED_CLASSES:
                cls_dir = target_path / cls
                imgs = [f for f in cls_dir.glob("*.png") if "_mask" not in f.name.lower()]
                imgs += [f for f in cls_dir.glob("*.jpg") if "_mask" not in f.name.lower()]
                counts[cls] = len(imgs)

            print(f"\n[KaggleDownloader] Dataset sudah ada di: {target_path}")
            print(f"  Distribusi: {counts}")
            print(f"  Total gambar: {sum(counts.values())} (tanpa mask)")
            print(f"  Tip: Gunakan force_download=True untuk re-download.\n")
            return str(target_path)

    # ── Cek kagglehub terinstall ──
    if not _check_kagglehub():
        raise ImportError(
            "\n[KaggleDownloader] kagglehub belum terinstall!\n"
            "Jalankan: pip install kagglehub\n\n"
            "Atau install semua requirements:\n"
            "    pip install -r requirements.txt\n"
        )

    import kagglehub

    # ── Cek Kaggle API credentials ──
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json    = kaggle_json.exists()
    has_env     = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if not has_json and not has_env:
        raise RuntimeError(
            "\n[KaggleDownloader] Kaggle API credentials tidak ditemukan!\n\n"
            "Setup (pilih salah satu):\n"
            "  A) File ~/.kaggle/kaggle.json\n"
            "     1. Buka https://www.kaggle.com/settings\n"
            "     2. Scroll ke 'API' → 'Create New Token'\n"
            "     3. Letakkan kaggle.json di ~/.kaggle/\n"
            "     4. chmod 600 ~/.kaggle/kaggle.json\n\n"
            "  B) Environment variables:\n"
            "     export KAGGLE_USERNAME='your_username'\n"
            "     export KAGGLE_KEY='your_api_key'\n"
        )

    # ── Download via kagglehub ──
    print(f"\n[KaggleDownloader] Mendownload dataset dari Kaggle...")
    print(f"  Dataset : {KAGGLE_DATASET_HANDLE}")
    print(f"  Target  : {target_path}")

    try:
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        print(f"  Cache   : {cache_path}")
    except Exception as e:
        raise RuntimeError(
            f"\n[KaggleDownloader] Download gagal: {e}\n\n"
            "Kemungkinan penyebab:\n"
            "  1. Kredensial Kaggle salah/expired → buat token baru\n"
            "  2. Belum accept Terms of Use dataset\n"
            "     → Buka https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset\n"
            "       dan klik 'I Understand and Accept' jika ada\n"
            "  3. Tidak ada koneksi internet\n"
        ) from e

    # ── Temukan root folder BUSI di cache ──
    busi_root = _find_busi_root(cache_path)
    if busi_root is None:
        raise RuntimeError(
            f"\n[KaggleDownloader] Struktur folder tidak sesuai!\n"
            f"  Cache path: {cache_path}\n"
            f"  Subfolder yang diharapkan: {EXPECTED_CLASSES}\n"
            f"  Isi cache:\n"
            + "\n".join(f"    {p}" for p in sorted(Path(cache_path).rglob("*")) if p.is_dir())
        )

    # ── Copy/symlink ke target_dir ──
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if str(Path(busi_root).resolve()) != str(target_path):
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(busi_root, str(target_path))
        print(f"  Copied  : {busi_root} → {target_path}")

    # ── Verifikasi dan tampilkan statistik ──
    counts = {}
    total_imgs  = 0
    total_masks = 0

    for cls in EXPECTED_CLASSES:
        cls_dir = target_path / cls
        all_pngs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        imgs  = [f for f in all_pngs if "_mask" not in f.name.lower()]
        masks = [f for f in all_pngs if "_mask" in f.name.lower()]
        counts[cls] = len(imgs)
        total_imgs  += len(imgs)
        total_masks += len(masks)

    print(f"\n[KaggleDownloader] ✓ Download selesai!")
    print(f"  Path    : {target_path}")
    print(f"  Kelas   : {counts}")
    print(f"  Total gambar (tanpa mask) : {total_imgs}")
    print(f"  Total mask                : {total_masks}\n")

    return str(target_path)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    path = download_busi_dataset()
    print(f"Dataset siap di: {path}")
