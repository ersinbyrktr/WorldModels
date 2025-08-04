# src/worldmodels/envs/carracing.py
from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from src.worldmodels.envs.base import CollectorEnv, resize64


@dataclass
class CarRacingBias:
    steer: float
    gas: float
    brake: float


def apply_carracing_bias(a: np.ndarray, bias: CarRacingBias) -> np.ndarray:
    """[steer∈(-1,1), gas∈(0,1), brake∈(0,1)]"""
    a = a.copy()
    a[0] = np.maximum(a[0] - bias.steer, -1.0)
    a[1] = np.minimum(a[1] + bias.gas, 1.0)
    a[2] = np.maximum(a[2] - bias.brake, 0.0)
    return a


class CarRacingAdapter(CollectorEnv):
    """Env adapter for CarRacing-v3 with optional per-run random action bias."""

    def __post_init__(self):
        self._bias: CarRacingBias | None = None

    def make_env(self):
        def _thunk():
            env = gym.make(
                "CarRacing-v3",
                render_mode=self.render_mode,
                lap_complete_percent=0.95,
                domain_randomize=False,
            )
            if self.seed is not None:
                env.reset(seed=self.seed)
            else:
                env.reset()
            return env

        return _thunk

    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        # CarRacing frames are HWC uint8; default 64x64 resize is OK
        return resize64(obs)

    def on_run_start(self) -> None:
        # sample per-run biases
        self._bias = CarRacingBias(
            steer=random.uniform(-0.1, 0.5),
            gas=random.uniform(0.0, 0.8),
            brake=random.uniform(0.0, 0.8),
        )
        print(f"[CarRacing bias] steer={self._bias.steer:.3f} gas={self._bias.gas:.3f} brake={self._bias.brake:.3f}")

    def postprocess_action(self, a1d: np.ndarray) -> np.ndarray:
        if self._bias is None:
            return a1d
        return apply_carracing_bias(a1d, self._bias)

    def keys_to_action(self) -> dict:
        # [steer ∈ (-1,1), gas ∈ (0,1), brake ∈ (0,1)]
        return {
            "w": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # throttle
            "s": np.array([0.0, 0.0, 0.8], dtype=np.float32),  # brake
            "a": np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # steer left
            "d": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # steer right
            ("w", "a"): np.array([-1.0, 1.0, 0.0], dtype=np.float32),
            ("w", "d"): np.array([1.0, 1.0, 0.0], dtype=np.float32),
        }

    # Optional: explicit noop for clarity (all zeros already matches base)
    # def noop_action(self, space) -> np.ndarray:
    #     return np.array([0.0, 0.0, 0.0], dtype=np.float32)


# Compatibility exports (if other code still imports these)
def make_env(render_mode: str = "rgb_array"):
    return CarRacingAdapter(render_mode=render_mode).make_env()


def get_envs(workers: int, render_mode: str = "rgb_array", seed: int | None = None):
    return CarRacingAdapter(render_mode=render_mode, seed=seed).vectorize(workers)


if __name__ == "__main__":
    actions = CarRacingAdapter(render_mode="human").action_bounds()
    print(actions)
