# src/worldmodels/data/latent_dataset.py
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.worldmodels.utils.collate import pad_collate


class EpisodicLatentActionDataset(Dataset):
    """
    x = concat[z_t , a_t]     (T-1,  latent_dim + action_dim)
    y = z_{t+1}               (T-1,  latent_dim)
    L = real length (T-1)
    """

    def __init__(self,
                 latents: list[np.ndarray],
                 actions: list[np.ndarray],
                 action_dim: int):  # NEW
        assert len(latents) == len(actions)
        self.zs = [np.asarray(z, dtype=np.float32) for z in latents]

        # Slice actions to the **true** env dimension
        self.as_ = [np.asarray(a, dtype=np.float32)[..., :action_dim]  # NEW
                    for a in actions]
        self._action_dim = action_dim

    def __len__(self): return len(self.zs)

    def __getitem__(self, idx):
        z, a = self.zs[idx], self.as_[idx]  # (T, C_lat) , (T, C_act)
        x = np.concatenate([z[:-1], a[:-1]], axis=1)  # (T-1, latent+act)
        y = z[1:]  # (T-1, latent)
        return torch.from_numpy(x), torch.from_numpy(y), x.shape[0]


def create_rnn_latent_dataloaders(vae,
                                  data_root: str | os.PathLike,
                                  *,
                                  action_dim_expected: int,  # NEW
                                  batch_size: int = 32,
                                  train_split: float = 0.9,
                                  num_workers: int = 4,
                                  device=None):
    from .latent_utils import encode_episodes_to_latents_actions
    zs, acts = encode_episodes_to_latents_actions(vae, data_root, device=device)

    split = int(len(zs) * train_split)
    train_ds = EpisodicLatentActionDataset(zs[:split], acts[:split], action_dim_expected)
    val_ds = EpisodicLatentActionDataset(zs[split:], acts[split:], action_dim_expected)

    # We now **trust** it
    action_dim = action_dim_expected

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
