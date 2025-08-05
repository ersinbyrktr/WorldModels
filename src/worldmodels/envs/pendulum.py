# src/worldmodels/envs/pendulum.py
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np

from src.worldmodels.envs import CollectorEnv


# -------------------------------------------------------------------------- #
# Action-scaling: accept [-1, 1], send [-2, 2] to the underlying env
# -------------------------------------------------------------------------- #
class _ScaleAction(gym.ActionWrapper):
    def __init__(self, env, scale: float = 2.0):
        super().__init__(env)
        self.scale = scale
        low = np.full(env.action_space.shape, -1.0, dtype=np.float32)
        high = np.full(env.action_space.shape, 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32) * self.scale


# -------------------------------------------------------------------------- #
# Observation-wrapper: replace state vector with 96×96 RGB frame
# -------------------------------------------------------------------------- #
class _ImageObs(gym.ObservationWrapper):
    def __init__(self, env, out_size: int = 96):
        super().__init__(env)
        self.out_size = out_size
        shape = (out_size, out_size, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, _state_obs):
        frame = self.env.render()  # RGB frame from inner env (usually 400×400)
        if frame is None:
            raise RuntimeError(
                "Env.render() returned None — make sure render_mode='rgb_array'."
            )
        frame = cv2.resize(frame, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        return frame.astype(np.uint8)


# -------------------------------------------------------------------------- #
# Adapter itself
# -------------------------------------------------------------------------- #
class PendulumAdapter(CollectorEnv):
    """Pendulum-v1 adapter
       * actions exposed as [-1, 1]
       * obs returned as 96×96 RGB frames (uint8) like CarRacing
    """

    def make_env(self):
        def _thunk():
            env = gym.make("Pendulum-v1", render_mode=self.render_mode)
            env = _ScaleAction(env, scale=2.0)  # shrink/ext. action range
            env = _ImageObs(env, out_size=64)  # replace obs with images
            if self.seed is not None:
                env.reset(seed=self.seed)
            else:
                env.reset()
            return env

        return _thunk

    # ------------------------------------------------------------------ #
    # Pre/​post-processing
    # ------------------------------------------------------------------ #
    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        # Already a 96×96 uint8 RGB frame, so just pass it through
        return obs

    # No change to postprocess_action — _ScaleAction handles scaling.

    # ------------------------------------------------------------------ #
    # Keyboard helpers
    # ------------------------------------------------------------------ #
    def keys_to_action(self) -> dict:
        T = 1.0
        return {
            "a": np.array([-T], dtype=np.float32),  # push left
            "d": np.array([+T], dtype=np.float32),  # push right
            "s": np.array([0.0], dtype=np.float32),  # hold
        }


# -------------------------------------------------------------------------- #
# Smoke-test
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    adapter = PendulumAdapter()
    print("action_bounds():", adapter.action_bounds())  # [(-1.0, 1.0)]
    env = adapter.vectorize(workers=1)
    try:
        obs, _ = env.reset()
        print("obs shape / dtype:", obs.shape, obs.dtype)  # (1, 96, 96, 3) uint8
        step_obs, _, _, _, _ = env.step(np.array([[0.5]], dtype=np.float32))
        print("step OK, next obs shape:", step_obs.shape)
    finally:
        env.close()
