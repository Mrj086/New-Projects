"""
NeuroDiff - tests/test_smoke.py
Fast unit tests that run without downloading the model or GPU.
Tests the pure-python logic: visualizations, attribution math, tensor utils.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest


# ─────────────────────────────────────── visualize helpers

class TestVisualize:

    def test_apply_heatmap_shape(self):
        from src.visualize import apply_heatmap
        img     = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        heatmap = np.random.rand(32, 32).astype(np.float32)
        result  = apply_heatmap(img, heatmap, alpha=0.5)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_apply_heatmap_rescales(self):
        """Heatmap smaller than image should be upscaled automatically."""
        from src.visualize import apply_heatmap
        img     = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        heatmap = np.random.rand(8, 8).astype(np.float32)
        result  = apply_heatmap(img, heatmap, alpha=0.5)
        assert result.shape == (64, 64, 3)

    def test_export_gif(self, tmp_path):
        from src.visualize import export_gif
        frames  = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        outfile = str(tmp_path / "test.gif")
        result  = export_gif(frames, outfile, duration_ms=100, upscale=2)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# ─────────────────────────────────────── attribution

class TestAttribution:

    def test_superpixel_segments_shape(self):
        from src.attribution import _superpixel_segments
        seg = _superpixel_segments(32, 32, grid=4)
        assert seg.shape == (32, 32)
        assert seg.min() == 0
        assert seg.max() == 15   # 4x4 = 16 segments

    def test_attributor_returns_normalised(self):
        from src.attribution import ConceptAttributor
        img        = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        attributor = ConceptAttributor(grid_size=2, n_samples=8)
        shap_map, segments = attributor.compute(img)
        assert shap_map.shape == (32, 32)
        assert shap_map.min() >= 0.0 - 1e-6
        assert shap_map.max() <= 1.0 + 1e-6

    def test_top_segments(self):
        from src.attribution import ConceptAttributor, _superpixel_segments
        img        = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        attributor = ConceptAttributor(grid_size=2, n_samples=8)
        shap_map, segments = attributor.compute(img)
        top = attributor.top_segments(shap_map, segments, top_k=2)
        assert len(top) == 2
        assert all(isinstance(s, int) for s in top)

    def test_highlight_mask_shape(self):
        from src.attribution import ConceptAttributor
        img        = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        shap_map   = np.random.rand(128, 128).astype(np.float32)
        attributor = ConceptAttributor()
        try:
            result = attributor.highlight_mask(img, shap_map)
            assert result.shape == (128, 128, 3)
        except ImportError:
            pytest.skip("opencv not installed")


# ─────────────────────────────────────── model utils (no download)

class TestModelUtils:

    def test_tensor_to_numpy(self):
        """_tensor_to_numpy should convert [-1,1] tensor to uint8 (H,W,3)."""
        import torch
        from src.model import NeuroDiffModel
        t = torch.rand(1, 3, 32, 32) * 2 - 1   # fake [-1,1] tensor
        result = NeuroDiffModel._tensor_to_numpy(t)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_tensor_to_numpy_grayscale(self):
        """Single-channel tensors should be broadcast to 3 channels."""
        import torch
        from src.model import NeuroDiffModel
        t = torch.rand(1, 1, 32, 32) * 2 - 1
        result = NeuroDiffModel._tensor_to_numpy(t)
        assert result.shape == (32, 32, 3)


# ─────────────────────────────────────── attention store

class TestAttentionStore:

    def test_clear_resets(self):
        from src.model import AttentionStore
        import torch
        store = AttentionStore()
        store.maps["test"] = [torch.zeros(4, 4)]
        store.timestep_maps.append({"test": torch.zeros(4, 4)})
        store.clear()
        assert store.maps == {}
        assert store.timestep_maps == []

    def test_snapshot(self):
        from src.model import AttentionStore
        import torch
        store = AttentionStore()
        store.maps["layer1"] = [torch.rand(2, 4, 4)]
        store.snapshot()
        assert len(store.timestep_maps) == 1
        assert "layer1" in store.timestep_maps[0]
