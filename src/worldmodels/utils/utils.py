import cv2
import numpy as np
import torch
from torchvision import utils as vutils


def save_grid(t: torch.Tensor, fp: str, nrow: int = 8):
    vutils.save_image(t.clamp(0, 1).cpu(), fp, nrow=nrow)


def _looks_like_image(arr: np.ndarray) -> bool:
    if arr.ndim == 3 and (arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4)):
        return True
    if arr.ndim == 2:
        return True
    return False


def _obs_to_frame(env, obs: np.ndarray) -> np.ndarray:
    """Return an HWC image frame for the current state."""
    if isinstance(obs, np.ndarray) and _looks_like_image(obs):
        return obs
    # Otherwise, fall back to renderer (requires render_mode="rgb_array")
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Create env with render_mode='rgb_array'."
        )
    return frame


def _encode_frame(img: np.ndarray, vae, device: torch.device) -> torch.Tensor:
    """
    Accepts a frame in common layouts and returns z_t (latent from VAE).

    Accepts:
      - HWC RGB/RGBA  (H, W, 3/4)
      - CHW RGB/RGBA  (3/4, H, W)
      - Grayscale     (H, W)       -> replicated to 3 channels

    Resizes to 64x64 (if your VAE was trained on 64x64) and converts to [0,1] float.
    """

    img = np.asarray(img)

    # ---- Normalize layout to HWC, 3 channels
    if img.ndim == 2:
        # grayscale HxW -> HxWx3
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.ndim == 3:
        # CHW -> HWC
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        # RGBA -> RGB
        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
    else:
        raise ValueError(f"_encode_frame: unsupported input shape {img.shape}")

    # ---- Ensure uint8 HWC 64x64x3
    if img.dtype != np.uint8:
        # If 0..1 floats, scale to 0..255; otherwise clip
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if 0.0 <= vmin and vmax <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        z_mu, _ = vae.encode(x)  # adjust if your VAE API differs
    return z_mu.squeeze(0)
