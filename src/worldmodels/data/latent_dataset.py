# latent_dataset.py
# ----------------------------------------------------------
# Dataset for working with VAE latent sequences for RNN training
# ----------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.worldmodels.data.latent_dataset_episode import EpisodicLatentDataset
from src.worldmodels.models.vae import VAE
from src.worldmodels.utils.collate import pad_collate


class LatentSequenceDataset(Dataset):
    """Dataset for sequences of VAE-encoded latent vectors."""

    def __init__(self, latent_sequences: np.ndarray, block_size: int):
        """
        Args:
            latent_sequences: Array of shape (N, latent_dim) where N is total timesteps
            block_size: Length of each sequence chunk
        """
        self.latents = torch.as_tensor(latent_sequences, dtype=torch.float32)
        self.block_size = block_size

    def __len__(self):
        return len(self.latents) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.latents[idx: idx + self.block_size]  # Input sequence
        y = self.latents[idx + 1: idx + self.block_size + 1]  # Target sequence (shifted by 1)
        return x, y


class LatentSequenceDatasetV2(Dataset):
    """Dataset for sequences of VAE-encoded latent vectors."""

    def __init__(self, latent_sequences: np.ndarray):
        """
        Args:
            latent_sequences: Array of shape (N, latent_dim) where N is total timesteps
        """
        self.latents = torch.as_tensor(latent_sequences, dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = self.latents[idx: idx + len(self.latents) - 1]  # Input sequence
        y = self.latents[idx + 1: idx + len(self.latents)]  # Target sequence (shifted by 1)
        return x, y


def encode_image_sequences_to_latents(vae_model: VAE, image_dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Encode all images in the dataloader to latent representations.

    Args:
        vae_model: Trained VAE model
        image_dataloader: DataLoader with image sequences
        device: torch device

    Returns:
        np.ndarray: Array of latent vectors, shape (N, latent_dim)
    """
    vae_model.eval()
    all_latents = []

    with torch.no_grad():
        for batch_images in tqdm(image_dataloader, desc="Encoding images to latents"):
            batch_images = batch_images.to(device)
            mu, _ = vae_model.encode(batch_images)
            # Use mean of latent distribution (deterministic encoding)
            latents = mu  # Shape: (batch_size, latent_dim)
            all_latents.append(latents.cpu().numpy())

    return np.concatenate(all_latents, axis=0)


def create_rnn_latent_dataloaders(vae,
                                  data_root,
                                  *,
                                  batch_size=32,
                                  train_split=0.9,
                                  num_workers=4,
                                  device=None):
    # ---- produce *list of arrays*, one per episode -----------------------
    from .latent_utils import encode_episodes_to_latents  # or your own helper
    episode_latents = encode_episodes_to_latents(vae,
                                                 data_root,
                                                 device=device)

    split = int(len(episode_latents) * train_split)
    train_ds = EpisodicLatentDataset(episode_latents[:split])
    val_ds = EpisodicLatentDataset(episode_latents[split:])

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

    return train_dl, val_dl
