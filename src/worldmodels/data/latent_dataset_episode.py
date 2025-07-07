# data/latent_dataset.py  (add or replace)

from torch.utils.data import Dataset
import torch
import numpy as np

class EpisodicLatentDataset(Dataset):
    """
    Each item is the entire latent trajectory of one episode.

    Args
    ----
    latents_per_episode : list[np.ndarray]
        List length = #episodes.
        Each array has shape (T, latent_dim) — *raw* VAE latents.
    """
    def __init__(self, latents_per_episode: list[np.ndarray]):
        super().__init__()
        self.seqs = [np.asarray(seq, dtype=np.float32) for seq in latents_per_episode]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx):
        z = self.seqs[idx]              # (T, C)
        x = torch.from_numpy(z[:-1])    # (T-1, C)
        y = torch.from_numpy(z[1:])     # (T-1, C)
        L = x.shape[0]                  # real length
        return x, y, L                  # -> collate_fn
