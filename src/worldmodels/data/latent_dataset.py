# latent_dataset.py
# ----------------------------------------------------------
# Dataset for working with VAE latent sequences for RNN training
# ----------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.worldmodels.data.data_loader import make_dataloaders
from src.worldmodels.models.vae import VAE


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


def create_latent_dataloaders(
        vae_model: VAE,
        data_root: str | Path,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        cached_latents_path: Optional[str] = None,
        device: Optional[torch.device] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with latent sequences encoded by the VAE.

    Args:
        vae_model: Trained VAE model
        data_root: Path to image data directory
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader
        train_val_split: Split ratio between train and validation sets
        cached_latents_path: Optional path to save/load cached latent vectors
        device: Device to use for encoding (defaults to CUDA if available)

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if we can load cached latents
    if cached_latents_path and os.path.exists(cached_latents_path):
        print(f"Loading cached latent vectors from {cached_latents_path}")
        latent_vectors = np.load(cached_latents_path)
    else:
        # Load image data with sequential ordering (no shuffle)
        print(f"Loading image data from {data_root}")
        image_loader, _ = make_dataloaders(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=False  # Important: keep temporal order for sequences
        )

        # Encode images to latent space
        print("Encoding images to latent space...")
        latent_vectors = encode_image_sequences_to_latents(vae_model, image_loader, device)

        # Save latents if path provided
        if cached_latents_path:
            os.makedirs(os.path.dirname(cached_latents_path), exist_ok=True)
            print(f"Saving latent vectors to {cached_latents_path}")
            np.save(cached_latents_path, latent_vectors)

    # Split into train and validation sets
    n_samples = len(latent_vectors)
    split_idx = int(n_samples * train_val_split)

    train_latents = latent_vectors[:split_idx]
    val_latents = latent_vectors[split_idx:]

    # Create datasets
    train_dataset = LatentSequenceDatasetV2(train_latents)
    val_dataset = LatentSequenceDatasetV2(val_latents)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created train dataloader with {len(train_dataset):,} sequences")
    print(f"Created val dataloader with {len(val_dataset):,} sequences")

    return train_loader, val_loader
