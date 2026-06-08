"""
utils/gradcam.py
================
Implementasi Grad-CAM untuk EfficientNetV2-S.

Grad-CAM menggunakan gradient dari output kelas tertentu
terhadap activation map layer konvolusional terakhir
untuk menghasilkan heatmap lokalisasi.

Referensi: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import EvalConfig


# ─────────────────────────────────────────────
# 1. GradCAM Class
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM untuk model berbasis CNN.

    Usage:
        gradcam = GradCAM(model, target_layer_name="features.7")
        heatmap = gradcam(input_tensor, target_class=None)
        gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _find_layer(self) -> nn.Module:
        """Temukan layer berdasarkan nama (dot-notation)."""
        parts = self.target_layer_name.split(".")
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _register_hooks(self):
        """Daftarkan forward dan backward hooks."""
        target_layer = self._find_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Hapus semua hooks setelah selesai."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input (1, C, H, W)
            target_class: Indeks kelas target. None = kelas prediksi terbesar.
        Returns:
            heatmap: Array (H, W) dengan nilai 0-1.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Pilih kelas target
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pada kelas target
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM: GAP dari gradient × activation maps
        gradients  = self.gradients[0]    # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global Average Pool gradients
        weights = gradients.mean(dim=(1, 2))   # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (kita hanya peduli yang positif)
        cam = F.relu(cam)

        # Normalisasi 0–1
        cam = cam.numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, int(target_class)


# ─────────────────────────────────────────────
# 2. Overlay heatmap pada gambar asli
# ─────────────────────────────────────────────

def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap Grad-CAM pada gambar asli.

    Args:
        original_image: Array (H, W, 3) uint8 atau float [0,1]
        heatmap       : Array (h, w) float [0, 1]
        alpha         : Transparansi overlay heatmap
    Returns:
        Overlaid image (H, W, 3) uint8
    """
    # Resize heatmap ke ukuran gambar
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Konversi ke uint8 dan colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Pastikan original image dalam uint8
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_uint8 = (original_image * 255).clip(0, 255).astype(np.uint8)
    else:
        original_uint8 = original_image.astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ─────────────────────────────────────────────
# 3. Denormalize tensor ke numpy image
# ─────────────────────────────────────────────

def tensor_to_numpy_image(
    tensor: torch.Tensor,
    mean: Tuple = (0.485, 0.456, 0.406),
    std:  Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Konversi tensor (C, H, W) ke numpy image (H, W, 3) float [0,1]."""
    img = tensor.clone().cpu().numpy()           # (C, H, W)
    img = img.transpose(1, 2, 0)                 # (H, W, C)
    img = img * np.array(std) + np.array(mean)   # Denormalize
    img = img.clip(0, 1)
    return img


# ─────────────────────────────────────────────
# 4. Visualisasi Grad-CAM untuk beberapa sampel
# ─────────────────────────────────────────────

def visualize_gradcam(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    target_layer_name: str = "features.7",
    num_samples: int = 12,
    output_path: str = "outputs/figures/gradcam.png",
    device: Optional[torch.device] = None,
    mean: Tuple = (0.485, 0.456, 0.406),
    std:  Tuple = (0.229, 0.224, 0.225),
):
    """
    Visualisasikan Grad-CAM untuk `num_samples` gambar dari test_loader.
    Menampilkan: gambar asli | heatmap | overlay — per baris.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    gradcam = GradCAM(model, target_layer_name)

    # Kumpulkan sampel (balanced per kelas jika memungkinkan)
    samples_by_class: Dict[int, List] = {i: [] for i in range(len(class_names))}
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            cls = lbl.item()
            if len(samples_by_class[cls]) < num_samples // len(class_names) + 1:
                samples_by_class[cls].append((img, cls))

    all_samples = []
    per_class = max(1, num_samples // len(class_names))
    for cls_samples in samples_by_class.values():
        all_samples.extend(cls_samples[:per_class])
    all_samples = all_samples[:num_samples]

    n = len(all_samples)
    ncols = 3   # Original | Heatmap | Overlay
    nrows = n

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    fig.patch.set_facecolor("#1a1a2e")
    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)

    for i, (img_tensor, true_label) in enumerate(all_samples):
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Generate Grad-CAM
        heatmap, pred_class = gradcam(input_tensor)

        # Denormalize untuk display
        original_img = tensor_to_numpy_image(img_tensor, mean, std)
        overlay_img  = overlay_gradcam(original_img, heatmap)

        # Heatmap colored (untuk tampilan terpisah)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]

        # Plot
        row_label = f"True: {class_names[true_label]}\nPred: {class_names[pred_class]}"
        correct = true_label == pred_class
        label_color = "#00ff88" if correct else "#ff4444"

        axes[i, 0].imshow(original_img)
        axes[i, 0].set_ylabel(row_label, color=label_color, fontsize=9, labelpad=5)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap_colored)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_img)
        axes[i, 2].axis("off")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")

    plt.suptitle(
        "Grad-CAM Visualization — EfficientNetV2-S",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout(pad=1.5)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    gradcam.remove_hooks()
    print(f"[GradCAM] Saved → {output_path}")
