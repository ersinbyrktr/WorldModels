#!/usr/bin/env python
# play_rgb_episode.py
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def play_episode(npy_path: str | Path, fps: int = 50) -> None:
    """
    Visualise a saved RGB episode stored as a .npy file.

    Parameters
    ----------
    npy_path : str | Path
        Path to the .npy file which contains the rgb frames.
        (shape == (N, H, W, 3), dtype=uint8, RGB).
    fps : int, default 50
        Playback speed in frames‑per‑second. 50 matches the game's metadata.
    Press q or Esc at any time to quit.
    """
    npy_path = Path(npy_path).expanduser()
    frames = np.load(npy_path)  # (N, 64, 64, 3) RGB
    num_frames = len(frames)
    print(f"Number of frames in '{npy_path.name}': {num_frames}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{npy_path} does not contain an RGB frame stack")

    win_name = f"{npy_path.stem}  ({len(frames)}frames)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    delay = max(1, int(1000 / fps))  # ms between frames

    for f in frames:
        bgr = f[..., ::-1]  # OpenCV expects BGR
        cv2.imshow(win_name, bgr)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            break

    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play a recorded episode (.npy)")
    parser.add_argument("--npy", type=str, help="Path to episode_XXXXX.npy",
                        default="../../../data/bipedal/episode_00000_obs.npy")
    parser.add_argument("--fps", type=int, default=50,
                        help="Playback FPS (default: 50)")
    args = parser.parse_args()

    play_episode(args.npy, fps=args.fps)
