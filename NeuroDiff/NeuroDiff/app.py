"""
NeuroDiff - app.py
Gradio UI for the diffusion model explainability engine.

Tabs:
  1. Generate & Explore   — run denoising, view attention maps frame by frame
  2. Counterfactual       — edit the generation midway, compare divergence
  3. SHAP Attribution     — concept importance heatmap on the final image
"""

import os
import sys
import gradio as gr
import numpy as np
from PIL import Image

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.model import NeuroDiffModel
from src.attribution import ConceptAttributor
from src import visualize as viz

# ── global model (loaded once at startup) ──────────────────────────────────
print("Loading NeuroDiff model … (first run downloads ~200 MB)")
MODEL = NeuroDiffModel()
ATTRIBUTOR = ConceptAttributor(grid_size=4, n_samples=64)

os.makedirs("outputs", exist_ok=True)

# ──────────────────────────────────────────────────────── Tab 1: Generate

def run_generate(seed, steps, capture_every, show_step_idx):
    """Run full denoising and return visuals for the chosen step."""
    result = MODEL.generate(
        num_inference_steps=int(steps),
        seed=int(seed),
        capture_every=int(capture_every),
    )

    n_snaps = len(result["latent_frames"])
    idx = min(int(show_step_idx), n_snaps - 1)

    frame_img = result["latent_frames"][idx]["image"]
    attn_map  = result["attention_snapshots"][idx]["map"]
    overlay   = viz.apply_heatmap(frame_img, attn_map, alpha=0.55)

    # denoising strip figure
    strip_path = "outputs/strip.png"
    viz.denoising_strip(
        result["latent_frames"],
        result["attention_snapshots"],
        strip_path,
    )

    # GIF of raw frames
    gif_path = "outputs/denoising.gif"
    viz.export_gif([f["image"] for f in result["latent_frames"]], gif_path)

    # Store result in state dict (via Gradio State)
    state = {
        "latent_frames": result["latent_frames"],
        "attention_snapshots": result["attention_snapshots"],
        "final_image": result["final_image"],
    }

    step_label = (
        f"Step {result['latent_frames'][idx]['step']} / "
        f"t={result['latent_frames'][idx]['timestep']}"
    )

    n_choices = list(range(n_snaps))

    return (
        Image.fromarray(frame_img),
        Image.fromarray(overlay),
        Image.fromarray(result["final_image"]),
        strip_path,
        gif_path,
        step_label,
        gr.update(maximum=n_snaps - 1, value=idx),
        state,
    )


def update_step(state, step_idx):
    """Swap which denoising step is shown without re-running the model."""
    if state is None:
        return None, None, "Run generation first."
    idx = min(int(step_idx), len(state["latent_frames"]) - 1)
    frame_img = state["latent_frames"][idx]["image"]
    attn_map  = state["attention_snapshots"][idx]["map"]
    overlay   = viz.apply_heatmap(frame_img, attn_map, alpha=0.55)
    label = f"Step {state['latent_frames'][idx]['step']} / t={state['latent_frames'][idx]['timestep']}"
    return Image.fromarray(frame_img), Image.fromarray(overlay), label


# ──────────────────────────────────────────────────────── Tab 2: Counterfactual

def run_counterfactual(seed, steps, edit_start, edit_end, noise_scale):
    """Generate both original and counterfactual, compare them."""
    steps = int(steps)
    seed  = int(seed)

    # Original run
    orig = MODEL.generate(num_inference_steps=steps, seed=seed, capture_every=5)

    # Counterfactual run
    cf = MODEL.counterfactual_generate(
        seed=seed,
        edit_timestep_start=int(edit_start),
        edit_timestep_end=int(edit_end),
        noise_scale=float(noise_scale),
        num_inference_steps=steps,
    )

    comp_path = "outputs/counterfactual.png"
    viz.counterfactual_comparison(
        orig["latent_frames"],
        cf["frames"],
        cf["edit_window"],
        comp_path,
    )

    diff_img = np.abs(
        orig["final_image"].astype(float) - cf["final_image"].astype(float)
    ).mean(axis=-1)
    diff_norm = (diff_img / diff_img.max() * 255).astype(np.uint8)

    summary = (
        f"Edit window: steps {edit_start}–{edit_end}  |  "
        f"Noise scale: {noise_scale}  |  "
        f"Mean pixel divergence: {diff_img.mean():.1f}"
    )

    return (
        Image.fromarray(orig["final_image"]),
        Image.fromarray(cf["final_image"]),
        Image.fromarray(diff_norm),
        comp_path,
        summary,
    )


# ──────────────────────────────────────────────────────── Tab 3: SHAP

def run_shap(state):
    """Compute SHAP attribution on the last generated image."""
    if state is None or "final_image" not in state:
        return None, None, "Please run generation first (Tab 1)."

    img = state["final_image"]
    shap_map, segments = ATTRIBUTOR.compute(img)

    # Upscale for visibility (32→128)
    from PIL import Image as PILImage
    big_img = np.array(PILImage.fromarray(img).resize((128, 128), PILImage.NEAREST))
    big_shap = np.array(
        PILImage.fromarray((shap_map * 255).astype(np.uint8)).resize((128, 128), PILImage.NEAREST)
    ) / 255.0

    try:
        blended = ATTRIBUTOR.highlight_mask(big_img, big_shap)
    except ImportError:
        # cv2 not available — use matplotlib blend
        blended = viz.apply_heatmap(big_img, big_shap, cmap=viz.SHAP_CMAP, alpha=0.55)

    panel_path = "outputs/shap_panel.png"
    viz.shap_panel(big_img, big_shap, blended, panel_path)

    top_segs = ATTRIBUTOR.top_segments(shap_map, segments, top_k=4)
    top_str = f"Top influential segments: {top_segs}"

    return (
        Image.fromarray(blended),
        panel_path,
        top_str,
    )


# ──────────────────────────────────────────────────────── Gradio layout

CSS = """
body { background: #0f0f0f !important; }
.gr-button-primary { background: #5c6bc0 !important; }
h1 { font-size: 1.7rem !important; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="NeuroDiff") as demo:

    state = gr.State(None)   # holds latest generation result

    gr.Markdown(
        """
        # 🔬 NeuroDiff — Diffusion Model Explainability Engine
        Visualise what a diffusion model *thinks* at each denoising step.
        Explore attention maps, generate counterfactuals, and compute SHAP concept attributions.
        """
    )

    # ─────────────────────────────────────── Tab 1
    with gr.Tab("🎲 Generate & Explore"):
        with gr.Row():
            with gr.Column(scale=1):
                seed_input    = gr.Slider(0, 9999, value=42, step=1, label="Random seed")
                steps_input   = gr.Slider(20, 100, value=50, step=5, label="Denoising steps")
                capture_input = gr.Slider(1, 10, value=5, step=1, label="Capture every N steps")
                gen_btn       = gr.Button("▶ Generate", variant="primary")

            with gr.Column(scale=2):
                step_slider   = gr.Slider(0, 10, value=0, step=1, label="Explore step")
                step_label    = gr.Textbox(label="Current step", interactive=False)
                with gr.Row():
                    frame_out   = gr.Image(label="Denoised frame", width=180, height=180)
                    attn_out    = gr.Image(label="Attention overlay", width=180, height=180)
                    final_out   = gr.Image(label="Final image", width=180, height=180)

        with gr.Row():
            strip_out = gr.Image(label="Denoising strip", type="filepath")
            gif_out   = gr.Image(label="Denoising GIF", type="filepath")

        gen_btn.click(
            run_generate,
            inputs=[seed_input, steps_input, capture_input, step_slider],
            outputs=[frame_out, attn_out, final_out, strip_out, gif_out, step_label, step_slider, state],
        )
        step_slider.change(
            update_step,
            inputs=[state, step_slider],
            outputs=[frame_out, attn_out, step_label],
        )

    # ─────────────────────────────────────── Tab 2
    with gr.Tab("🔀 Counterfactual"):
        gr.Markdown("Inject noise during a specific window of the denoising process and see how the final image diverges.")
        with gr.Row():
            with gr.Column(scale=1):
                cf_seed   = gr.Slider(0, 9999, value=42, step=1, label="Seed (same as Tab 1 for fair comparison)")
                cf_steps  = gr.Slider(20, 100, value=50, step=5, label="Total steps")
                cf_start  = gr.Slider(0, 40, value=10, step=1, label="Edit window start")
                cf_end    = gr.Slider(5, 49, value=30, step=1, label="Edit window end")
                cf_noise  = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label="Noise scale")
                cf_btn    = gr.Button("▶ Run counterfactual", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    cf_orig = gr.Image(label="Original final", width=200, height=200)
                    cf_gen  = gr.Image(label="Counterfactual final", width=200, height=200)
                    cf_diff = gr.Image(label="Pixel divergence", width=200, height=200)
                cf_strip  = gr.Image(label="Full comparison strip", type="filepath")
                cf_summary = gr.Textbox(label="Summary", interactive=False)

        cf_btn.click(
            run_counterfactual,
            inputs=[cf_seed, cf_steps, cf_start, cf_end, cf_noise],
            outputs=[cf_orig, cf_gen, cf_diff, cf_strip, cf_summary],
        )

    # ─────────────────────────────────────── Tab 3
    with gr.Tab("🧩 SHAP Attribution"):
        gr.Markdown("Compute **Kernel SHAP** over superpixel segments of your generated image. Shows which regions drove the model's output signal.")
        with gr.Row():
            shap_btn = gr.Button("▶ Compute SHAP (requires Tab 1 run first)", variant="primary")

        with gr.Row():
            shap_overlay = gr.Image(label="SHAP overlay", width=256, height=256)
            shap_panel   = gr.Image(label="Full attribution panel", type="filepath")

        shap_info = gr.Textbox(label="Top segments", interactive=False)

        shap_btn.click(
            run_shap,
            inputs=[state],
            outputs=[shap_overlay, shap_panel, shap_info],
        )

    gr.Markdown(
        """
        ---
        **NeuroDiff** · Built with 🤗 Diffusers, SHAP, Gradio · MIT License
        Model: `google/ddpm-cifar10-32` (CIFAR-10, 32×32)
        """
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
