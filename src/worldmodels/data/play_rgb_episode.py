#!/usr/bin/env python
# scripts/play_rgb_episode.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.worldmodels.envs import EnvKind


# ────────────────────────────────────────────────────────────────────────────────
# Config (derive episode path from EnvKind + collection_name + index)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodePlayerCfg:
    env: EnvKind = EnvKind.CARRACING
    collection_name: str = "default"  # data folder: ../../../data/{env}_{collection_name}
    index: int = 1  # episode_{index:05d}_obs.npy
    fps: int = 50
    data_root: Path = Path("../../../data")
    # Optional: direct path override (if set, env/collection/index are ignored)
    npy_path: Optional[Path] = None
    window_name: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _env_short(env: EnvKind) -> str:
    return getattr(env, "short_name", env.name.lower())


def _episode_path(cfg: EpisodePlayerCfg) -> Path:
    if cfg.npy_path is not None:
        return Path(cfg.npy_path).expanduser()
    root = cfg.data_root / f"{_env_short(cfg.env)}_{cfg.collection_name}"
    return root / f"episode_{cfg.index:05d}_obs.npy"


# ────────────────────────────────────────────────────────────────────────────────
# Player
# ────────────────────────────────────────────────────────────────────────────────

def play_episode(cfg: EpisodePlayerCfg) -> None:
    npy_path = _episode_path(cfg)
    frames = np.load(npy_path)  # (N, H, W, 3) RGB uint8
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{npy_path} does not contain an RGB frame stack (N,H,W,3).")

    win_name = cfg.window_name or f"{npy_path.stem} ({len(frames)} frames)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    delay = max(1, int(1000 / max(1, cfg.fps)))

    print(f"[play] file={npy_path} | frames={len(frames)} | fps={cfg.fps}")
    for f in frames:
        cv2.imshow(win_name, f[..., ::-1])  # RGB -> BGR
        k = cv2.waitKey(delay) & 0xFF
        if k in (ord('q'), 27):  # q or Esc
            break
    cv2.destroyWindow(win_name)


# ────────────────────────────────────────────────────────────────────────────────
# Example
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = EpisodePlayerCfg(
        env=EnvKind.BIPEDAL_WALKER,
        collection_name="baseline",
        index=1,
        fps=50,
    )
    play_episode(cfg)
