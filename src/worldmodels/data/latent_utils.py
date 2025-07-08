# latent_utils.py ------------------------------------------------------------
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _matching_actions_file(obs_file: Path) -> Path:
    return obs_file.with_name(obs_file.name.replace("_obs.npy", "_actions.npy"))


def encode_episodes_to_latents_actions(vae, root: Path, device, batch=64) \
        -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns two lists (latents, actions) with equal length == #episodes.
      • latents[i]  : (Ti, latent_dim)
      • actions[i]  : (Ti, action_dim)
    """
    root = Path(root)
    L, A = [], []

    for obs_file in sorted(root.glob("*_obs.npy")):
        act_file = _matching_actions_file(obs_file)
        frames = np.load(obs_file)  # (T,H,W,3) uint8
        actions = np.load(act_file).astype(np.float32)  # (T, action_dim)

        frames = torch.as_tensor(frames).permute(0, 3, 1, 2).float() / 255.
        dl = DataLoader(frames, batch_size=batch, shuffle=False)

        ep_lat = []
        with torch.no_grad():
            for imgs in dl:
                imgs = imgs.to(device)
                mu, _ = vae.encode(imgs)
                ep_lat.append(mu.cpu().numpy())

        lat_arr = np.concatenate(ep_lat, axis=0)  # (T, latent_dim)
        assert lat_arr.shape[0] == actions.shape[0], f"Length mismatch in {obs_file}"

        L.append(lat_arr)
        A.append(actions)
    return L, A
