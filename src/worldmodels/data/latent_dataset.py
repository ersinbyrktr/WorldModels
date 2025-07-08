# latent_dataset.py
# ----------------------------------------------------------
# Dataset for working with VAE latent sequences for RNN training
# ----------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.worldmodels.utils.collate import pad_collate

class EpisodicLatentActionDataset(Dataset):
    """
    Item = (x, y, L) where
        x : (T-1, latent_dim + action_dim)  =  concat[z_t , a_t]
        y : (T-1, latent_dim)              =  z_{t+1}
        L : int  real length (T-1)
    """
    def __init__(self, latents, actions):
        assert len(latents) == len(actions)
        self.zs = [np.asarray(z, dtype=np.float32) for z in latents]
        self.as_ = [np.asarray(a, dtype=np.float32) for a in actions]

    def __len__(self): return len(self.zs)

    def __getitem__(self, idx):
        z, a = self.zs[idx], self.as_[idx]
        x = np.concatenate([z[:-1], a[:-1]], axis=1)   # concat along channel
        y = z[1:]
        L = x.shape[0]
        return torch.from_numpy(x), torch.from_numpy(y), L


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
        z = self.seqs[idx]  # (T, C)
        x = torch.from_numpy(z[:-1])  # (T-1, C)
        y = torch.from_numpy(z[1:])  # (T-1, C)
        L = x.shape[0]  # real length
        return x, y, L  # -> collate_fn


def create_rnn_latent_dataloaders(vae,
                                  data_root,
                                  *,
                                  batch_size=32,
                                  train_split=0.9,
                                  num_workers=4,
                                  device=None):
    # ---- produce *list of arrays*, one per episode -----------------------

    from .latent_utils import encode_episodes_to_latents_actions
    zs, acts = encode_episodes_to_latents_actions(vae, data_root, device=device)
    split = int(len(zs) * train_split)
    train_ds = EpisodicLatentActionDataset(zs[:split], acts[:split])
    val_ds = EpisodicLatentActionDataset(zs[split:], acts[split:])

    action_dim = acts[0].shape[1]

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=pad_collate)

    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=pad_collate)

    return train_dl, val_dl, action_dim
