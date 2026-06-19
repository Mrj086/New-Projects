# 🔬 NeuroDiff — Diffusion Model Explainability Engine

> **What does a diffusion model *think* at each denoising step?**  
> NeuroDiff hooks into UNet attention layers, extracts maps at every timestep,
> supports counterfactual generation, and computes SHAP concept attribution —
> all in an interactive Gradio UI.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.27+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/NeuroDiff/blob/main/notebooks/explore.ipynb)

---

## What it does

Most people treat diffusion models as **black boxes**: noise in, image out.  
NeuroDiff gives you a **microscope**.

| Feature | Description |
|---|---|
| **Attention map extraction** | Hooks into every UNet attention block; captures maps at each denoising timestep |
| **Denoising strip** | Visual timeline: raw frame + attention overlay side-by-side at every captured step |
| **Counterfactual generation** | Inject noise mid-denoising and observe how the generation path diverges |
| **SHAP attribution** | Kernel SHAP over superpixel segments — shows which image regions drove the model's output |
| **GIF export** | Animated denoising progression, ready for your README or portfolio |

---

## Demo

### Denoising progression + attention maps
![denoising strip](assets/strip_demo.png)

### Counterfactual: original vs edited generation path
![counterfactual](assets/counterfactual_demo.png)

### SHAP concept attribution
![shap](assets/shap_demo.png)

---

## Architecture

```
NeuroDiff/
├── app.py                  # Gradio UI (3 tabs)
├── src/
│   ├── model.py            # NeuroDiffModel: UNet hooks + generation + counterfactual
│   ├── attribution.py      # ConceptAttributor: Kernel SHAP over superpixels
│   └── visualize.py        # Heatmaps, strips, comparison panels, GIF export
├── notebooks/
│   └── explore.ipynb       # Colab-ready exploration notebook
└── tests/
    └── test_smoke.py       # Fast unit tests (no GPU, no model download)
```

**Diffusion backbone:** `google/ddpm-cifar10-32` — 32×32 CIFAR-10 unconditional DDPM.  
Runs on **CPU** (≈2 min/generation) or **GPU** (≈10 s).

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/NeuroDiff.git
cd NeuroDiff

# 2. Install
pip install -r requirements.txt

# 3. Launch UI
python app.py
# → open http://localhost:7860
```

The model (~200 MB) downloads automatically from HuggingFace Hub on first run.

---

## How it works

### 1. Attention map extraction

```python
# Simplified — see src/model.py for full implementation
for name, module in self.unet.named_modules():
    if "attn" in name.lower() and hasattr(module, "to_q"):
        module.register_forward_hook(store_attention(name))
```

Every forward pass through the UNet stores attention outputs.  
These are aggregated (mean over channels, resized to 32×32) into a single heatmap per timestep.

### 2. Counterfactual generation

```python
for i, t in enumerate(scheduler.timesteps):
    noise_pred = unet(image, t).sample
    image = scheduler.step(noise_pred, t, image).prev_sample

    if edit_start <= i <= edit_end:
        image = image + torch.randn_like(image) * noise_scale  # ← perturbation
```

By injecting extra noise during a specific window, we create a **divergent generation path** —
simulating "what would the model have generated if it took a different route?"

### 3. SHAP attribution

```python
# Surrogate model: mean brightness of un-masked superpixels
explainer = shap.KernelExplainer(surrogate_f, background)
shap_values = explainer.shap_values(test_input, nsamples=64)
```

Kernel SHAP assigns importance scores to each superpixel segment.
We map them back to pixel space and normalise to [0, 1].

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover: tensor utilities, attention store, SHAP attribution math,
heatmap rendering, and GIF export. No GPU or model download required.

---

## Extending NeuroDiff

| Want to... | How |
|---|---|
| Use Stable Diffusion instead of DDPM | Replace `MODEL_ID` in `model.py`, adjust hook targets for the cross-attention layers |
| Add text-conditioned generation | Pass a text prompt through a CLIP encoder; inject as UNet cross-attention conditioning |
| Export attention maps as numpy arrays | Call `model.generate(...)` and read `result["attention_snapshots"]` |
| Swap to CLIP-based attribution | Replace the surrogate in `attribution.py` with a CLIP similarity scorer |

---

## Citation

If you use NeuroDiff in research or coursework, please cite:

```bibtex
@misc{neurodiff2024,
  title   = {NeuroDiff: Diffusion Model Explainability Engine},
  author  = {YOUR NAME},
  year    = {2024},
  url     = {https://github.com/YOUR_USERNAME/NeuroDiff}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with 🤗 [Diffusers](https://github.com/huggingface/diffusers),
[SHAP](https://github.com/slundberg/shap),
[Gradio](https://gradio.app/)*
