"""
NeuroDiff - attribution.py
SHAP-based concept attribution: given the final generated image,
which pixel regions contributed most to the overall prediction signal?

Uses Kernel SHAP over a simple surrogate model (pixel-mean predictor)
so it runs without a GPU and without a labelled dataset.
"""

import numpy as np
import shap
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


def _superpixel_segments(h: int, w: int, grid: int = 4) -> np.ndarray:
    """
    Divide image into a grid of (grid x grid) superpixel segments.
    Returns a (h, w) array of integer segment IDs.
    """
    seg = np.zeros((h, w), dtype=int)
    cell_h = h // grid
    cell_w = w // grid
    idx = 0
    for r in range(grid):
        for c in range(grid):
            r0, r1 = r * cell_h, (r + 1) * cell_h if r < grid - 1 else h
            c0, c1 = c * cell_w, (c + 1) * cell_w if c < grid - 1 else w
            seg[r0:r1, c0:c1] = idx
            idx += 1
    return seg


class ConceptAttributor:
    """
    Computes SHAP values over superpixel segments of the generated image.

    The 'model' we explain is a simple surrogate:
      f(mask) = mean brightness of un-masked pixels (normalised).

    This is a deliberately lightweight proxy that lets us demonstrate
    SHAP attribution without needing a labelled classifier — the
    explanation shows which regions the model 'lit up' most during
    generation.
    """

    def __init__(self, grid_size: int = 4, n_samples: int = 64):
        self.grid_size = grid_size
        self.n_samples = n_samples

    def _make_surrogate(self, image: np.ndarray, segments: np.ndarray):
        """
        Returns a function f: (N, num_segments) binary mask → (N,) scores.
        Score = mean pixel intensity of selected segments, normalised to [0,1].
        """
        num_seg = segments.max() + 1
        gray = image.mean(axis=-1)   # (H, W)

        def f(masks: np.ndarray) -> np.ndarray:
            scores = np.zeros(len(masks))
            for i, mask in enumerate(masks):
                selected = np.zeros_like(gray, dtype=bool)
                for seg_id in range(num_seg):
                    if mask[seg_id]:
                        selected |= (segments == seg_id)
                if selected.sum() > 0:
                    scores[i] = gray[selected].mean() / 255.0
                else:
                    scores[i] = 0.0
            return scores

        return f, num_seg

    def compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kernel SHAP on the image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3), dtype uint8

        Returns
        -------
        shap_map   : (H, W) float array — SHAP value per pixel
        segments   : (H, W) int array  — superpixel segment IDs
        """
        h, w = image.shape[:2]
        segments = _superpixel_segments(h, w, self.grid_size)
        num_seg = segments.max() + 1

        f, _ = self._make_surrogate(image, segments)

        # Background = all segments masked OFF
        background = np.zeros((1, num_seg))
        explainer = shap.KernelExplainer(f, background)

        # Explain the fully-unmasked input (all ones)
        test_input = np.ones((1, num_seg))
        shap_values = explainer.shap_values(test_input, nsamples=self.n_samples, silent=True)
        shap_values = np.array(shap_values).flatten()   # (num_seg,)

        # Map per-segment SHAP values back to pixel space
        shap_map = np.zeros((h, w), dtype=np.float32)
        for seg_id in range(num_seg):
            shap_map[segments == seg_id] = shap_values[seg_id]

        # Normalise to [0, 1]
        mn, mx = shap_map.min(), shap_map.max()
        if mx > mn:
            shap_map = (shap_map - mn) / (mx - mn)

        return shap_map, segments

    def top_segments(
        self, shap_map: np.ndarray, segments: np.ndarray, top_k: int = 4
    ) -> List[int]:
        """Return IDs of the top_k most important segments."""
        num_seg = segments.max() + 1
        seg_scores = [shap_map[segments == i].mean() for i in range(num_seg)]
        ranked = sorted(range(num_seg), key=lambda i: seg_scores[i], reverse=True)
        return ranked[:top_k]

    def highlight_mask(
        self,
        image: np.ndarray,
        shap_map: np.ndarray,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Overlay the SHAP heatmap on the original image.
        High SHAP → green tint; low → red tint.
        """
        import cv2  # only needed here

        heatmap_uint8 = (shap_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_RdYlGn)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
        blended = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
        return blended
