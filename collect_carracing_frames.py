#!/usr/bin/env python
# collect_carracing_frames.py
# -------------------------------------------------------------------
# Collect 64×64 RGB frames from many parallel CarRacing environments.
# -------------------------------------------------------------------
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from tqdm import trange

# -------------------------------------------------------------------

from stable_baselines3 import PPO  # RL algo implementation
from huggingface_sb3 import load_from_hub  # HF helper


def load_pretrained_carracing() -> PPO:
    """
    Downloads and returns a PPO agent trained on CarRacing‑v3.
    The weights are hosted on the Hugging Face Hub (≈ 26 MB).
    """
    if PPO is None or load_from_hub is None:
        raise RuntimeError(
            "Install stable‑baselines3 and huggingface‑sb3 first "
            "(pip install stable-baselines3 huggingface-sb3)"
        )

    # model card: https://huggingface.co/Rinnnt/ppo-CarRacing-v3
    repo_id = "Rinnnt/ppo-CarRacing-v3"
    filename = "ppo-CarRacing-v3.zip"
    return PPO.load(load_from_hub(repo_id=repo_id, filename=filename))


def make_env(seed: int):
    """Factory that creates ONE CarRacing env (called in each worker)."""

    def _thunk():
        env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
        env.reset(seed=seed)
        return env

    return _thunk


def resize64(arr: np.ndarray) -> np.ndarray:
    """Fast 64×64 resize without type-checker complaints."""
    return cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)


def save_png(img: np.ndarray, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# -------------------------------------------------------------------


def main(
        out: Path,
        episodes: int,
        max_steps: int,
        skip: int,
        start_frame: int,
        workers: int,
        seed: int,
        save_npy: bool,
        policy: str
) -> None:
    """Run many CarRacing episodes in parallel and save frames.

    Parameters
    ----------
    start_frame : int
        Number of initial frames to ignore **per episode** before any frames
        qualify for saving. Defaults to 40.
    skip : int
        After the *start_frame* offset has been reached, save every
        ``skip + 1``‑th frame.
    """

    random.seed(seed)
    np.random.seed(seed)

    # ------------- vector env ------------------------------------------------
    env_fns = [make_env(seed + i) for i in range(workers)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    obs, _ = envs.reset()

    # ------------- bookkeeping ----------------------------------------------
    step_counters = np.zeros(workers, dtype=np.int32)
    episode_buffers = [[] for _ in range(workers)]  # NEW
    episode_count = 0
    frame_count = 0
    # ------------- policy picker ----------------------------------------------
    policy_model = None
    if policy == "ppo":
        policy_model = load_pretrained_carracing()
    # ------------- rollout loop ---------------------------------------------
    with trange(episodes, desc="Episodes") as pbar:
        while episode_count < episodes:
            if policy == "random":
                actions = [envs.single_action_space.sample() for _ in range(workers)]
            else:  # "ppo"
                # The PPO checkpoint was trained with one env, so predict for each worker
                actions = []
                for ob in obs:  # `obs` is (H,W,C) per worker
                    act, _ = policy_model.predict(ob, deterministic=True)
                    actions.append(act)
            print(actions)
            obs, _, term, trunc, _ = envs.step(actions)
            done = np.logical_or(term, trunc)

            # -------- collect frames ----------------------------------------
            for i, ob in enumerate(obs):
                if step_counters[i] >= start_frame and (
                        (step_counters[i] - start_frame) % (skip + 1) == 0
                ):
                    episode_buffers[i].append(resize64(ob))

                step_counters[i] += 1

                if done[i]:
                    # -------- flush this entire episode in one shot ----------
                    if save_npy:
                        ep_arr = np.asarray(episode_buffers[i], dtype=np.uint8)
                        np.save(out / f"episode_{episode_count:05d}.npy", ep_arr)
                        frame_count += len(ep_arr)
                    else:
                        for img in episode_buffers[i]:
                            save_png(img, out / f"{frame_count:08d}.png")
                            frame_count += 1
                    # reset per-env counters & buffer
                    episode_buffers[i].clear()
                    step_counters[i] = 0
                    episode_count += 1
                    pbar.update(1)

            if episode_count >= episodes:
                break

    envs.close()

    # ------------- write single .npy (optional) ------------------------------
    envs.close()
    print(
        f"[info] Saved {frame_count} "
        f"{'frames (as individual episode .npy files)' if save_npy else 'PNGs'} "
        f"under {out}")

    # -------------------------------------------------------------------


if __name__ == "__main__":
    mp.freeze_support()  # Essential on Windows; no-op elsewhere

    parser = argparse.ArgumentParser(
        description="Collect CarRacing frames in parallel."
    )
    parser.add_argument("--out", type=Path, default=Path("data/carracing"),
                        help="output directory (PNGs) or prefix for .npy")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes to record (each ≤ max-steps)")
    parser.add_argument("--max-steps", type=int, default=1_000,
                        help="truncate episode after this many steps (unused)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="save every (skip + 1)th frame after start-frame offset")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="number of initial frames to ignore in each episode")
    parser.add_argument("--workers", type=int, default=1,
                        help="# parallel environments / CPU workers")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--npy", action="store_true", default=True,
                        help="store all frames into one .npy instead of PNGs")
    parser.add_argument("--policy", choices=["random", "ppo"], default="ppo",
                        help="which policy drives the car")

    args = parser.parse_args()

    main(
        out=args.out,
        episodes=args.episodes,
        max_steps=args.max_steps,
        skip=args.skip_frames,
        start_frame=args.start_frame,
        workers=args.workers,
        seed=args.seed,
        save_npy=args.npy,
        policy=args.policy
    )
