#!/usr/bin/env python
# collect_carracing_episodes.py
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

from src.worldmodels.envs.carracing import get_envs


def resize64(arr: np.ndarray) -> np.ndarray:
    """Fast 64×64 resize without type-checker complaints."""
    return cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)


def main(
        out: Path,
        episodes: int,
        workers: int
) -> None:
    # ------------- vector env ------------------------------------------------
    envs = get_envs(workers)
    obs, _ = envs.reset()

    # ------------- bookkeeping ----------------------------------------------
    step_counters = np.zeros(workers, dtype=np.int32)
    episode_buffers = [[] for _ in range(workers)]  # NEW
    episode_count = 0
    frame_count = 0
    # ------------- rollout loop ---------------------------------------------
    with trange(episodes, desc="Episodes") as pbar:
        steer_bias = random.uniform(-0.1, 0.5)
        gas_bias = random.uniform(0, 0.8)
        brake_bias = random.uniform(0, 0.8)
        print(f"Steering bias: {steer_bias}")
        print(f"Gas bias: {gas_bias}")
        print(f"Brake bias: {brake_bias}")
        while episode_count < episodes:
            actions = []
            for _ in range(workers):
                action = envs.action_space.sample()  # np array with shape (1, 3)
                # Extract the flat action to work with it
                flat_action = np.array([action[0, 0], action[0, 1], action[0, 2]])
                # Apply the biases
                flat_action[0] = np.maximum(flat_action[0] - steer_bias, -1)  # steering
                flat_action[1] = np.minimum(flat_action[1] + gas_bias, 1)  # gas
                flat_action[2] = np.maximum(flat_action[2] - brake_bias, 0)  # brake
                actions.append(flat_action)

            # Convert to numpy array with shape (workers, 3)
            actions = np.array(actions)

            obs, _, term, trunc, _ = envs.step(actions)
            done = np.logical_or(term, trunc)

            # -------- collect frames ----------------------------------------
            for i, ob in enumerate(obs):
                episode_buffers[i].append(resize64(ob))
                step_counters[i] += 1

                if done[i]:
                    # -------- flush this entire episode in one shot ----------
                    ep_arr = np.asarray(episode_buffers[i], dtype=np.uint8)
                    np.save(out / f"episode_{episode_count:05d}.npy", ep_arr)
                    frame_count += len(ep_arr)
                    # reset per-env counters & buffer
                    episode_buffers[i].clear()
                    step_counters[i] = 0
                    episode_count += 1
                    pbar.update(1)

            if episode_count >= episodes:
                break
    envs.close()


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Collect CarRacing frames in parallel.")
    parser.add_argument("--out", type=Path, default=Path("../../../data/carracing"),
                        help="output directory (PNGs) or prefix for .npy")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes to record (each ≤ max-steps)")
    parser.add_argument("--workers", type=int, default=12, help="# parallel environments / CPU workers")

    args = parser.parse_args()

    main(
        out=args.out,
        episodes=args.episodes,
        workers=args.workers
    )
