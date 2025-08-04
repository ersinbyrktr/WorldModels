# src/worldmodels/envs/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np


def resize64(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)


def flatten_action(a: np.ndarray | Sequence[float]) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0]
    return a.reshape(-1)


def bounds_from_space(space: gym.Space) -> Optional[list[Tuple[float, float]]]:
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low).reshape(-1)
        high = np.asarray(space.high).reshape(-1)
        return [(float(l), float(h)) for l, h in zip(low, high)]
    return None


@dataclass
class CollectorEnv:
    """Base adapter for episode collection."""
    render_mode: str = "rgb_array"
    seed: Optional[int] = None

    # ---- overridables --------------------------------------------------------
    def make_env(self):
        raise NotImplementedError

    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        return resize64(obs)

    def on_run_start(self) -> None:
        """Optional hook (e.g., sample per-run action biases)."""
        pass

    def postprocess_action(self, a1d: np.ndarray) -> np.ndarray:
        """Optional hook (e.g., apply env-specific action shaping/bias)."""
        return a1d

    # ---- default vectorization & action sampling -----------------------------
    def vectorize(self, workers: int):
        env_fns = [self.make_env() for _ in range(workers)]
        return gym.vector.AsyncVectorEnv(env_fns)

    def sample_action(self, space: gym.Space) -> np.ndarray:
        return flatten_action(space.sample())

    def sample_batch_actions(self, space: gym.Space, workers: int) -> np.ndarray:
        acts = []
        for _ in range(workers):
            a = self.sample_action(space)
            a = self.postprocess_action(a)
            acts.append(a)
        return np.asarray(acts, dtype=np.float32)

    # ---- convenience ---------------------------------------------------------
    def action_bounds(self) -> Optional[list[Tuple[float, float]]]:
        # instantiate one env to inspect space if needed
        thunk = self.make_env()
        env = thunk()
        try:
            return bounds_from_space(env.action_space)
        finally:
            env.close()

    # ---- keyboard play --------------------------------------------------------
    def keys_to_action(self) -> dict:
        """
        Override per env. Return a mapping suitable for gymnasium.utils.play.play
        where keys are str or tuple[str, ...] and values are 1-D np.float32 actions.
        """
        return {}

    def noop_action(self, space: gym.Space) -> np.ndarray:
        """
        Default 'no-op' action: zeros in the correct shape/range.
        Override if a different neutral command is preferred.
        """
        # shape as per a sampled action but fill with zeros
        a = np.asarray(space.sample(), dtype=np.float32)
        a[...] = 0.0
        return flatten_action(a)
