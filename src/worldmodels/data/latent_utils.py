# latent_utils.py ------------------------------------------------------------
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def encode_episodes_to_latents(vae, root: Path, device, batch=64) -> list[np.ndarray]:
    """
    Returns a list whose i-th element is a (Ti, latent_dim) array –
    the VAE latents for episode i.
    """
    root = Path(root)
    latents_per_episode: list[np.ndarray] = []

    for npy in sorted(root.glob("*.npy")):          # ← 1 file == 1 episode
        frames = np.load(npy)                       # (T,H,W,3) uint8
        frames = torch.as_tensor(frames).permute(0,3,1,2).float() / 255.
        dl = DataLoader(frames, batch_size=batch, shuffle=False)

        episode_latents = []
        with torch.no_grad():
            for imgs in dl:
                imgs = imgs.to(device)
                mu, _ = vae.encode(imgs)            # (B, latent_dim)
                episode_latents.append(mu.cpu().numpy())

        latents_per_episode.append(np.concatenate(episode_latents, axis=0))

    return latents_per_episode          # length = #episodes
