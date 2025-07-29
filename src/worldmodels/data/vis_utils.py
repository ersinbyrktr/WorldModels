# src/worldmodels/data/vis_utils.py
from __future__ import annotations
import numpy as np
import cv2

def resize64(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)

def frames_to_rgb8(frames: np.ndarray) -> np.ndarray:
    """
    Convert a frame stack to HWC RGB uint8.
    Supports:
      - (T, H, W, 3)   RGB
      - (T, H, W, 4)   RGBA -> drop alpha
      - (T, 3, H, W)   CHW  -> HWC
      - (T, H, W)      grayscale -> replicate to RGB
    Values in [0,1] are scaled to [0,255]. Others are clipped to [0,255].
    """
    arr = np.asarray(frames)

    if arr.ndim == 4 and arr.shape[-1] in (3, 4):     # HWC
        x = arr[..., :3]
    elif arr.ndim == 4 and arr.shape[1] in (3, 4):    # CHW
        x = np.moveaxis(arr, 1, -1)[..., :3]
    elif arr.ndim == 3:                                # grayscale H W
        x = np.repeat(arr[..., None], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported frame stack shape: {arr.shape}")

    if x.dtype != np.uint8:
        # If clearly 0–1, scale up; otherwise clip to [0,255]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if 0.0 <= x_min and x_max <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def to_rgb64_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a single frame to 64×64 RGB uint8."""
    x = frames_to_rgb8(frame[None, ...])[0]
    return resize64(x)
