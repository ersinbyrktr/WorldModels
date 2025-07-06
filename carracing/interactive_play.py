import numpy as np
from gymnasium.envs.box2d import car_racing   # pulls in the module, *not* an env
from gymnasium.utils.play import play


car_racing.TRACK_TURN_RATE   = 0.03          # default 0.31  → curlier

import gymnasium as gym
env = gym.make("CarRacing-v3", render_mode="rgb_array",)   # now uses the new numbers

keys_to_action = {
    # single-key presses
    "w":  np.array([ 0.0, 1.0, 0.0], dtype=np.float32),   # accelerate
    "s":  np.array([ 0.0, 0.0, 0.8], dtype=np.float32),   # brake
    "a":  np.array([-1.0, 0.0, 0.0], dtype=np.float32),   # steer left
    "d":  np.array([ 1.0, 0.0, 0.0], dtype=np.float32),   # steer right
    # combined presses (e.g. hold W + A or W + D to corner under throttle)
    ("w", "a"): np.array([-1.0, 1.0, 0.0], dtype=np.float32),
    ("w", "d"): np.array([ 1.0, 1.0, 0.0], dtype=np.float32),
}

play(
    env,
    keys_to_action=keys_to_action,
    fps=60,
    zoom=1.0,  # ⇧ for pixel-perfect; >1.0 zooms in
    noop=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # when no key is pressed
)