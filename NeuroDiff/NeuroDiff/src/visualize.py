"""
NeuroDiff - visualize.py
All rendering helpers: heatmap overlays, attention colormaps,
denoising strip, counterfactual side-by-side, and GIF export.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless - no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import io
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image


# ─────────────────────────────────────────────── colourmap helpers

ATTENTION_CMAP = cm.get_cmap("inferno")
SHAP_CMAP      = cm.get_cmap("RdYlGn")
DIFF_CMAP      = cm.get_cmap("coolwarm")


def apply_heatmap(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    cmap=ATTENTION_CMAP,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a 2D heatmap (values in [0,1]) on a uint8 RGB image.
    Returns a uint8 RGB image of the same size.
    """
    h, w = base_image.shape[:2]
    # Resize heatmap to match image if needed
    if heatmap.shape != (h, w):
        from PIL import Image as PILImage
        hm_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        hm_pil = hm_pil.resize((w, h), PILImage.BILINEAR)
        heatmap = np.array(hm_pil) / 255.0

    colored = cmap(heatmap)[:, :, :3]   # drop alpha channel
    colored_uint8 = (colored * 255).astype(np.uint8)
    blended = (alpha * colored_uint8 + (1 - alpha) * base_image).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────── denoising strip

def denoising_strip(
    latent_frames: List[Dict],
    attention_snapshots: List[Dict],
    output_path: str,
    upscale: int = 6,
) -> str:
    """
    Create a horizontal strip showing the denoising progression,
    with attention heatmap overlaid below each frame.
    Saves to output_path and returns it.
    """
    n = min(len(latent_frames), len(attention_snapshots))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 4.8))
    fig.patch.set_facecolor("#0d0d0d")

    for i in range(n):
        frame = latent_frames[i]
        snap  = attention_snapshots[i]

        raw_img = frame["image"]
        attn    = snap["map"]

        # top row: raw denoised frame
        ax_top = axes[0, i] if n > 1 else axes[0]
        _show_image(ax_top, raw_img, upscale)
        ax_top.set_title(f"t={frame['timestep']}", color="#aaaaaa", fontsize=8, pad=3)

        # bottom row: attention overlay
        ax_bot = axes[1, i] if n > 1 else axes[1]
        overlay = apply_heatmap(raw_img, attn, cmap=ATTENTION_CMAP, alpha=0.55)
        _show_image(ax_bot, overlay, upscale)
        if i == 0:
            ax_bot.set_ylabel("attention", color="#888888", fontsize=8)

    _style_figure(fig)
    plt.suptitle("Denoising progression  ·  attention maps", color="white", fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────── counterfactual comparison

def counterfactual_comparison(
    original_frames: List[Dict],
    counterfactual_frames: List[Dict],
    edit_window: Tuple[int, int],
    output_path: str,
    upscale: int = 6,
) -> str:
    """
    Side-by-side comparison of original vs counterfactual denoising.
    Highlights the edit window in red.
    """
    n = min(len(original_frames), len(counterfactual_frames), 8)
    orig_frames  = original_frames[:n]
    cf_frames    = counterfactual_frames[:n]

    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7.0))
    fig.patch.set_facecolor("#0d0d0d")

    for i in range(n):
        step = orig_frames[i]["step"]
        in_window = edit_window[0] <= step <= edit_window[1]

        # row 0: original
        ax0 = axes[0, i]
        _show_image(ax0, orig_frames[i]["image"], upscale)
        ax0.set_title(f"s={step}", color="#aaaaaa", fontsize=8)
        if i == 0:
            ax0.set_ylabel("original", color="#61bfad", fontsize=8)

        # row 1: counterfactual
        ax1 = axes[1, i]
        _show_image(ax1, cf_frames[i]["image"], upscale)
        if in_window:
            for sp in ax1.spines.values():
                sp.set_edgecolor("#ff4d4d")
                sp.set_linewidth(2)
        if i == 0:
            ax1.set_ylabel("counterfactual", color="#ff9a7c", fontsize=8)

        # row 2: pixel diff
        ax2 = axes[2, i]
        diff = np.abs(
            orig_frames[i]["image"].astype(float) -
            cf_frames[i]["image"].astype(float)
        ).mean(axis=-1)
        ax2.imshow(diff, cmap="hot", interpolation="nearest")
        ax2.axis("off")
        if i == 0:
            ax2.set_ylabel("pixel diff", color="#ffd580", fontsize=8)

    _style_figure(fig)
    edit_label = f"edit window: steps {edit_window[0]}–{edit_window[1]}"
    plt.suptitle(f"Counterfactual generation  ·  {edit_label}", color="white", fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────── SHAP attribution panel

def shap_panel(
    original_image: np.ndarray,
    shap_map: np.ndarray,
    blended: np.ndarray,
    output_path: str,
    upscale: int = 6,
) -> str:
    """
    3-panel figure: original | SHAP heatmap | blended overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.patch.set_facecolor("#0d0d0d")

    titles = ["Generated image", "SHAP attribution", "Overlay"]
    imgs   = [original_image, shap_map, blended]
    cmaps  = [None, "RdYlGn", None]

    for ax, title, img, cmap_name in zip(axes, titles, imgs, cmaps):
        if cmap_name:
            ax.imshow(img, cmap=cmap_name, interpolation="nearest")
        else:
            _show_image(ax, img, upscale)
        ax.set_title(title, color="#cccccc", fontsize=9)
        ax.axis("off")

    _style_figure(fig)
    plt.suptitle("Concept attribution  ·  Kernel SHAP", color="white", fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────── GIF export

def export_gif(
    frames: List[np.ndarray],
    output_path: str,
    duration_ms: int = 120,
    upscale: int = 8,
) -> str:
    """
    Convert a list of uint8 RGB arrays to an animated GIF.
    Each frame is upscaled by `upscale` for visibility.
    """
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        new_size = (frame.shape[1] * upscale, frame.shape[0] * upscale)
        img = img.resize(new_size, Image.NEAREST)
        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


# ─────────────────────────────────────────────── internal helpers

def _show_image(ax, img: np.ndarray, upscale: int):
    if img.ndim == 3:
        display_img = Image.fromarray(img)
        display_img = display_img.resize(
            (img.shape[1] * upscale, img.shape[0] * upscale), Image.NEAREST
        )
        ax.imshow(np.array(display_img), interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")


def _style_figure(fig):
    for ax in fig.axes:
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
