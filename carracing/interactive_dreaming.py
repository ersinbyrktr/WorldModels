# play_dream_carracing.py
"""Play World Models 'dream' CarRacing with keyboard controls.

Controls (same as OpenAI gym demo):
    ← / →    steer
    ↑        gas
    ↓        brake
    ESC      quit
"""

from collections import defaultdict

import cv2
import numpy as np
import torch

from src.worldmodels.envs.dream_carracing import DreamCarRacingEnv
from src.worldmodels.models.rnn import MDN_LSTM
from src.worldmodels.models.vae import VAE

# --------------------------------------------------------------------- #
#  Load models
# --------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_PATH = "../trained_model/vae_latest.pt"
RNN_PATH = "../trained_model/rnn_model_512_2.pt"
TAU = 1.0  # dream temperature

vae = VAE.load_model(VAE_PATH, device=DEVICE).eval()
rnn = MDN_LSTM.load_model(RNN_PATH, device=DEVICE).eval()

env = DreamCarRacingEnv(model=rnn, tau=TAU, deterministic=True)
CURRENT_ACTION = np.zeros(3, dtype=np.float32)  # steer, gas, brake
# --------------------------------------------------------------------- #
#  Key-mapping helper
# --------------------------------------------------------------------- #
KEYS = defaultdict(lambda: False)


def _update_keys():
    global CURRENT_ACTION
    key = cv2.waitKey(1) & 0xFF
    if key == 255:
        return  # no key this frame

    if key == 27:  # ESC
        raise KeyboardInterrupt

    # Map key-down to a *persistent* action
    if key in (ord('a'), 81, 2424832):  # ←
        CURRENT_ACTION[:] = [-1.0, 0.0, 0.0]
    elif key in (ord('d'), 83, 2555904):  # →
        CURRENT_ACTION[:] = [+1.0, 0.0, 0.0]
    elif key in (ord('w'), 82, 2490368):  # ↑
        CURRENT_ACTION[:] = [0.0, 1.0, 0.0]
    elif key in (ord('s'), 84, 2621440):  # ↓
        CURRENT_ACTION[:] = [0.0, 0.0, 1.0]
    elif key == ord(' '):  # space = coast
        CURRENT_ACTION[:] = [0.0, 0.0, 0.0]


def _action_from_keys() -> np.ndarray:
    steer = 0.0
    if KEYS[ord('a')] or KEYS[81] or KEYS[2424832]:  # ← or 'a'
        steer = -1.0
    elif KEYS[ord('d')] or KEYS[83] or KEYS[2555904]:  # → or 'd'
        steer = +1.0
    gas = 1.0 if KEYS[ord('w')] or KEYS[82] or KEYS[2490368] else 0.0  # ↑ or 'w'
    brake = 1.0 if KEYS[ord('s')] or KEYS[84] or KEYS[2621440] else 0.0  # ↓ or 's'
    return np.array([steer, gas, brake], dtype=np.float32)


# --------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------- #
obs, _ = env.reset()
total_reward = 0.0
frame_id = 0

try:
    while True:
        # 1. decode z_t (first 32 dims of observation) → RGB 64×64
        z = torch.from_numpy(obs[:64]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = vae.decode(z).squeeze(0)  # (3,64,64)
        img = (recon.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # 2. enlarge 4× for nicer viewing
        img_bgr = cv2.cvtColor(cv2.resize(img, (256, 256), cv2.INTER_NEAREST),
                               cv2.COLOR_RGB2BGR)
        # overlay step / return
        cv2.putText(img_bgr, f"step {frame_id}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"R={total_reward:.0f}", (180, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Dream Car Racing", img_bgr)
        _update_keys()
        action = CURRENT_ACTION.copy()

        # 3. env step
        obs, r, done, _, _ = env.step(action)
        total_reward += r
        frame_id += 1

        if done:
            print(f"Episode finished. Return = {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward, frame_id = 0.0, 0

except KeyboardInterrupt:
    print("\n[quit] bye!")

env.close()
cv2.destroyAllWindows()
