#!/usr/bin/env python
# play_carracing_episode.py
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def play_episode(npy_path: str | Path, fps: int = 50, loop: bool = False) -> None:
    """
    Visualise a saved CarRacing episode stored as a .npy file.

    Parameters
    ----------
    npy_path : str | Path
        Path to the .npy file produced by collect_carracing_episodes.py
        (shape == (N, H, W, 3), dtype=uint8, RGB).
    fps : int, default 50
        Playback speed in frames‑per‑second. 50 matches the game's metadata.
    loop : bool, default False
        If True, restart from the first frame when the episode ends.
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

    playing = True
    while playing:
        for f in frames:
            bgr = f[..., ::-1]  # OpenCV expects BGR
            cv2.imshow(win_name, bgr)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):  # q or Esc
                playing = False
                break
        if not loop:
            break

    cv2.destroyWindow(win_name)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play a recorded CarRacing episode (.npy)")
    parser.add_argument("--npy", type=str, help="Path to episode_XXXXX.npy",
                        default="../../../data/carracing/episode_00000.npy")
    parser.add_argument("--fps", type=int, default=50,
                        help="Playback FPS (default: 140)")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Loop the video until you press q / Esc")
    args = parser.parse_args()

    play_episode(args.npy, fps=args.fps, loop=args.loop)
