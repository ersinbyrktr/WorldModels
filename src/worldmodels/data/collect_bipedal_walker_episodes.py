#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import os

import numpy as np
from tqdm import trange

from src.worldmodels.data.vis_utils import to_rgb64_uint8
from src.worldmodels.envs.bipedal_walker import get_envs


def looks_like_image(obs: np.ndarray) -> bool:
    """Heuristic to detect image-like observations."""
    if obs.ndim == 3 and (obs.shape[-1] in (1, 3, 4) or obs.shape[0] in (1, 3, 4)):
        return True
    if obs.ndim == 2:
        return True
    return False


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")  # .npy.tmp

    with open(tmp_path, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def main(out: Path, episodes: int, workers: int) -> None:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    envs = get_envs(workers)
    obs, _ = envs.reset()

    use_obs_as_frames = looks_like_image(obs[0])

    episode_buffers = [[] for _ in range(workers)]
    action_buffers  = [[] for _ in range(workers)]
    episode_count = 0

    with trange(episodes, desc="Episodes", smoothing=0.1) as pbar:
        while episode_count < episodes:
            actions = np.stack(
                [envs.single_action_space.sample() for _ in range(workers)],
                axis=0
            )

            obs, _, term, trunc, _ = envs.step(actions)
            done = np.logical_or(term, trunc)

            frames_list = None
            if not use_obs_as_frames:
                frames_list = envs.call("render")  # list of frames, len == workers

            for i in range(workers):
                frame_src = obs[i] if use_obs_as_frames else frames_list[i]
                frame64 = to_rgb64_uint8(frame_src)

                episode_buffers[i].append(frame64)
                action_buffers[i].append(actions[i].astype(np.float32))

                if done[i]:
                    ep_obs     = np.asarray(episode_buffers[i], dtype=np.uint8)   # (T, 64, 64, 3)
                    ep_actions = np.asarray(action_buffers[i],  dtype=np.float32) # (T, A)

                    obs_path = out / f"episode_{episode_count:05d}_obs.npy"
                    act_path = out / f"episode_{episode_count:05d}_actions.npy"

                    _save_npy_atomic(obs_path, ep_obs)
                    _save_npy_atomic(act_path, ep_actions)

                    episode_buffers[i].clear()
                    action_buffers[i].clear()

                    episode_count += 1
                    pbar.update(1)

            if episode_count >= episodes:
                break

    envs.close()


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Collect Bipedal/CarRacing frames in parallel.")
    parser.add_argument("--out", type=Path, default=Path("../../../data/bipedal"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(out=args.out, episodes=args.episodes, workers=args.workers)
