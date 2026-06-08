"""
predict.py
==========
Script inference untuk gambar tunggal atau batch folder.

Usage:
    # Prediksi satu gambar
    python predict.py --image path/to/image.png

    # Prediksi semua gambar dalam folder
    python predict.py --folder path/to/folder/ --output_csv results.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import cfg
from models.efficientnet_v2 import build_model


CLASS_NAMES = ["benign", "malignant", "normal"]

EMOJI_MAP = {
    "benign":     "🟢",
    "malignant":  "🔴",
    "normal":     "⚪",
}


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model dari checkpoint."""
    model = build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"[Predict] Model loaded from: {checkpoint_path}")
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
) -> Dict:
    """Prediksi satu gambar, return dict hasil."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())

    return {
        "image_path":   str(image_path),
        "prediction":   CLASS_NAMES[pred],
        "confidence":   float(probs[pred]),
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)},
    }


def predict_folder(
    model: torch.nn.Module,
    folder_path: str,
    transform: transforms.Compose,
    device: torch.device,
    output_csv: str = None,
) -> List[Dict]:
    """Prediksi semua gambar dalam folder."""
    folder = Path(folder_path)
    img_files = sorted(
        [f for ext in ("*.png", "*.jpg", "*.jpeg")
         for f in folder.glob(ext)
         if "_mask" not in f.name.lower()]
    )

    if not img_files:
        print(f"[Predict] Tidak ada gambar ditemukan di: {folder}")
        return []

    print(f"\n[Predict] Processing {len(img_files)} images...")
    results = []

    for img_path in img_files:
        result = predict_single(model, img_path, transform, device)
        results.append(result)

        emoji = EMOJI_MAP[result["prediction"]]
        print(
            f"  {emoji} {img_path.name:<40} "
            f"{result['prediction']:>12}  "
            f"({result['confidence']*100:.1f}%)"
        )

    # Simpan ke CSV
    if output_csv and results:
        fieldnames = ["image_path", "prediction", "confidence",
                      "prob_benign", "prob_malignant", "prob_normal"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "image_path":     r["image_path"],
                    "prediction":     r["prediction"],
                    "confidence":     f"{r['confidence']:.4f}",
                    "prob_benign":    f"{r['probabilities']['benign']:.4f}",
                    "prob_malignant": f"{r['probabilities']['malignant']:.4f}",
                    "prob_normal":    f"{r['probabilities']['normal']:.4f}",
                })
        print(f"\n[Predict] Results saved to: {output_csv}")

    return results


def print_single_result(result: Dict):
    """Pretty print hasil prediksi satu gambar."""
    emoji = EMOJI_MAP[result["prediction"]]
    print(f"\n{'─'*45}")
    print(f"  Image     : {Path(result['image_path']).name}")
    print(f"  {emoji} Prediction : {result['prediction'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"\n  Probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"    {cls:>12} {bar} {prob*100:>5.1f}%")
    print(f"{'─'*45}\n")


def main():
    parser = argparse.ArgumentParser(description="BUSI Model Inference")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--image",  type=str, default=None, help="Path satu gambar")
    parser.add_argument("--folder", type=str, default=None, help="Path folder gambar")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    tf     = build_transform()

    if args.image:
        result = predict_single(model, args.image, tf, device)
        print_single_result(result)

    elif args.folder:
        predict_folder(model, args.folder, tf, device, args.output_csv)

    else:
        print("Gunakan --image <path> atau --folder <path>")


if __name__ == "__main__":
    main()
