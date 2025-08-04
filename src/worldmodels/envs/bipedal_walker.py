# src/worldmodels/envs/bipedal_walker.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.worldmodels.envs import CollectorEnv
from src.worldmodels.envs.base import resize64


class BipedalWalkerAdapter(CollectorEnv):
    """Env adapter for BipedalWalker-v3; no action bias."""

    def make_env(self):
        def _thunk():
            env = gym.make("BipedalWalker-v3", render_mode=self.render_mode)
            if self.seed is not None:
                env.reset(seed=self.seed)
            else:
                env.reset()
            return env

        return _thunk

    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        # If you record RGB frames, resize to 64×64; tweak if using state obs.
        return resize64(obs)

    # postprocess_action inherited (no-op)

    def keys_to_action(self) -> dict:
        """
        Action: 4 floats in [-1, 1] → [hip1, knee1, hip2, knee2]
        These bindings are heuristic; adjust to taste.
        """
        F = 0.6  # hip magnitude
        K = 0.6  # knee magnitude
        return {
            # forward/back lean
            "w": np.array([+F, +0.0, +F, +0.0], dtype=np.float32),
            "s": np.array([-F, +0.0, -F, +0.0], dtype=np.float32),
            # lateral balance (hip-knee alternation)
            "a": np.array([-F, +K, -F, +K], dtype=np.float32),
            "d": np.array([+F, -K, +F, -K], dtype=np.float32),
            # combos
            ("w", "a"): np.array([-F, +K, +F, +0.0], dtype=np.float32),
            ("w", "d"): np.array([+F, -K, +F, +0.0], dtype=np.float32),
        }

    # Optional: explicit noop (zeros already provided by base)
    # def noop_action(self, space) -> np.ndarray:
    #     return np.zeros(4, dtype=np.float32)


if __name__ == "__main__":
    actions = BipedalWalkerAdapter(render_mode="human").action_bounds()
    print(actions)
