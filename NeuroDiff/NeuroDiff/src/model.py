"""
NeuroDiff - model.py
Loads a diffusion model and hooks into UNet attention layers
to extract attention maps at every denoising timestep.
"""

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class AttentionStore:
    """
    Stores attention maps from hooked UNet layers.
    Accumulates maps across all timesteps during a diffusion run.
    """

    def __init__(self):
        self.maps: Dict[str, List[torch.Tensor]] = {}
        self.timestep_maps: List[Dict[str, torch.Tensor]] = []
        self._hooks = []

    def clear(self):
        self.maps = {}
        self.timestep_maps = []

    def snapshot(self):
        """Save current maps as one timestep snapshot, then clear buffer."""
        if self.maps:
            self.timestep_maps.append({k: v[-1].cpu() for k, v in self.maps.items()})

    def register_hook(self, name: str):
        """Returns a hook function that stores attention output for `name`."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            if name not in self.maps:
                self.maps[name] = []
            self.maps[name].append(attn_output.detach())
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class NeuroDiffModel:
    """
    Wraps a pretrained DDPM model with:
      - attention map extraction at every denoising step
      - counterfactual generation (guided noise editing)
      - per-step latent snapshots for visualization
    """

    MODEL_ID = "google/ddpm-cifar10-32"   # 32x32, fast to run, no GPU required

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NeuroDiff] Loading model on {self.device} …")
        self.pipeline = DDPMPipeline.from_pretrained(self.MODEL_ID)
        self.pipeline.to(self.device)
        self.unet: UNet2DModel = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        self.unet.eval()
        self.attention_store = AttentionStore()
        self._register_attention_hooks()
        print("[NeuroDiff] Model ready ✓")

    # ------------------------------------------------------------------ hooks

    def _register_attention_hooks(self):
        """Hook every Attention block inside the UNet."""
        self.attention_store.remove_hooks()
        for name, module in self.unet.named_modules():
            # diffusers UNet2DModel uses AttentionProcessor layers
            if "attn" in name.lower() and hasattr(module, "to_q"):
                h = module.register_forward_hook(
                    self.attention_store.register_hook(name)
                )
                self.attention_store._hooks.append(h)

    # -------------------------------------------------------- generation core

    @torch.no_grad()
    def generate(
        self,
        num_inference_steps: int = 50,
        seed: int = 42,
        capture_every: int = 5,
    ) -> Dict:
        """
        Run full denoising, capturing:
          - latent frames (every `capture_every` steps)
          - attention maps (every `capture_every` steps)

        Returns a dict with everything the UI needs.
        """
        self.scheduler.set_timesteps(num_inference_steps)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Start from pure noise
        image = torch.randn(
            (1, self.unet.config.in_channels,
             self.unet.config.sample_size,
             self.unet.config.sample_size),
            generator=generator,
            device=self.device,
        )

        latent_frames = []
        attention_snapshots = []
        self.attention_store.clear()

        for i, t in enumerate(self.scheduler.timesteps):
            self.attention_store.maps = {}   # reset per-step buffer
            noise_pred = self.unet(image, t).sample
            image = self.scheduler.step(noise_pred, t, image).prev_sample

            if i % capture_every == 0 or i == num_inference_steps - 1:
                frame = self._tensor_to_numpy(image)
                latent_frames.append({"step": i, "timestep": int(t), "image": frame})
                self.attention_store.snapshot()
                attn_map = self._aggregate_attention()
                attention_snapshots.append({"step": i, "timestep": int(t), "map": attn_map})

        final_image = self._tensor_to_numpy(image)
        return {
            "final_image": final_image,
            "latent_frames": latent_frames,
            "attention_snapshots": attention_snapshots,
            "num_steps": num_inference_steps,
        }

    # ------------------------------------------------------- counterfactual

    @torch.no_grad()
    def counterfactual_generate(
        self,
        seed: int = 42,
        edit_timestep_start: int = 10,
        edit_timestep_end: int = 30,
        noise_scale: float = 0.3,
        num_inference_steps: int = 50,
    ) -> Dict:
        """
        Generate a 'counterfactual' by injecting extra noise between
        edit_timestep_start and edit_timestep_end, then continuing
        denoising. This creates a divergent generation path — simulating
        'what if the model took a different route?'
        """
        self.scheduler.set_timesteps(num_inference_steps)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        image = torch.randn(
            (1, self.unet.config.in_channels,
             self.unet.config.sample_size,
             self.unet.config.sample_size),
            generator=generator,
            device=self.device,
        )

        frames = []
        for i, t in enumerate(self.scheduler.timesteps):
            noise_pred = self.unet(image, t).sample
            image = self.scheduler.step(noise_pred, t, image).prev_sample

            # ---- inject perturbation in the edit window ----
            if edit_timestep_start <= i <= edit_timestep_end:
                edit_noise = torch.randn_like(image) * noise_scale
                image = image + edit_noise

            if i % 5 == 0 or i == num_inference_steps - 1:
                frames.append({"step": i, "image": self._tensor_to_numpy(image)})

        return {
            "final_image": self._tensor_to_numpy(image),
            "frames": frames,
            "edit_window": (edit_timestep_start, edit_timestep_end),
            "noise_scale": noise_scale,
        }

    # -------------------------------------------------- attention aggregation

    def _aggregate_attention(self) -> np.ndarray:
        """
        Average all stored attention tensors into a single 2D heatmap.
        Returns a (H, W) numpy array normalised to [0, 1].
        """
        if not self.attention_store.timestep_maps:
            return np.zeros((32, 32))

        last_snapshot = self.attention_store.timestep_maps[-1]
        all_maps = []
        for name, tensor in last_snapshot.items():
            t = tensor.float()
            # reshape from (B, C, ...) → average over channel dim
            if t.dim() == 4:
                t = t.mean(dim=1)   # (B, H, W)
            elif t.dim() == 3:
                t = t.mean(dim=0)   # (H, W)
            t = t.squeeze()
            if t.dim() == 2:
                all_maps.append(t.numpy())

        if not all_maps:
            return np.zeros((32, 32))

        # Resize all to 32x32 and average
        resized = []
        for m in all_maps:
            m_tensor = torch.tensor(m).unsqueeze(0).unsqueeze(0)
            m_resized = F.interpolate(m_tensor, size=(32, 32), mode="bilinear", align_corners=False)
            resized.append(m_resized.squeeze().numpy())

        avg = np.mean(resized, axis=0)
        mn, mx = avg.min(), avg.max()
        if mx > mn:
            avg = (avg - mn) / (mx - mn)
        return avg.astype(np.float32)

    # ------------------------------------------------------------ utilities

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a (1, C, H, W) tensor in [-1,1] to a (H, W, 3) uint8 array."""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        img = (img + 1.0) / 2.0          # [-1,1] → [0,1]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.shape[-1] == 1:           # grayscale → RGB
            img = np.repeat(img, 3, axis=-1)
        return img
