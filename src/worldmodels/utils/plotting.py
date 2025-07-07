# plotting.py
# ----------------------------------------------------------
# Visualization utilities for World Models
# ----------------------------------------------------------
from __future__ import annotations

from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_latent_space_2d(latent_vectors: np.ndarray, labels: Optional[np.ndarray] = None,
                         dims: Tuple[int, int] = (0, 1), figsize=(10, 8),
                         alpha=0.7, s=10):
    """Plot 2D projection of latent space with optional coloring by labels.

    Args:
        latent_vectors: Array of shape (N, latent_dim)
        labels: Optional array of shape (N,) for coloring points
        dims: Which two latent dimensions to plot
        figsize: Figure size
        alpha: Point transparency
        s: Point size
    """
    plt.figure(figsize=figsize)

    if labels is not None:
        plt.scatter(latent_vectors[:, dims[0]], latent_vectors[:, dims[1]],
                    c=labels, alpha=alpha, s=s, cmap='viridis')
        plt.colorbar(label='Frame index')
    else:
        plt.scatter(latent_vectors[:, dims[0]], latent_vectors[:, dims[1]],
                    alpha=alpha, s=s)

    plt.xlabel(f'Latent dimension {dims[0]}')
    plt.ylabel(f'Latent dimension {dims[1]}')
    plt.title('2D projection of VAE latent space')
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_latent_trajectory(latent_sequence: np.ndarray, dims: Tuple[int, int] = (0, 1),
                           figsize=(10, 8), window=100):
    """Plot trajectory in latent space for a sequence.

    Args:
        latent_sequence: Array of shape (T, latent_dim)
        dims: Which two latent dimensions to plot
        figsize: Figure size
        window: Number of timesteps to show in trajectory (for long sequences)
    """
    if isinstance(latent_sequence, torch.Tensor):
        latent_sequence = latent_sequence.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    # If sequence is too long, plot the last `window` steps
    if len(latent_sequence) > window:
        seq = latent_sequence[-window:]
        start_idx = len(latent_sequence) - window
    else:
        seq = latent_sequence
        start_idx = 0

    # Plot points with color gradient by time
    points = plt.scatter(seq[:, dims[0]], seq[:, dims[1]],
                         c=np.arange(len(seq)), cmap='viridis',
                         alpha=0.7, s=10)
    plt.colorbar(points, label='Time step')

    # Draw trajectory line
    plt.plot(seq[:, dims[0]], seq[:, dims[1]], 'k-', alpha=0.3, linewidth=1)

    # Mark start and end points
    plt.plot(seq[0, dims[0]], seq[0, dims[1]], 'ro', markersize=8, label=f'Start (t={start_idx})')
    plt.plot(seq[-1, dims[0]], seq[-1, dims[1]], 'go', markersize=8, label=f'End (t={start_idx + len(seq) - 1})')

    plt.xlabel(f'Latent dimension {dims[0]}')
    plt.ylabel(f'Latent dimension {dims[1]}')
    plt.title('Trajectory in VAE latent space')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_latent_components(latent_sequence: np.ndarray, dims: List[int] = None, figsize=(12, 8)):
    """Plot the evolution of selected latent dimensions over time.

    Args:
        latent_sequence: Array of shape (T, latent_dim)
        dims: Which latent dimensions to plot (defaults to first 6)
        figsize: Figure size
    """
    if isinstance(latent_sequence, torch.Tensor):
        latent_sequence = latent_sequence.detach().cpu().numpy()

    # Default to first 6 dimensions if not specified
    if dims is None:
        dims = list(range(min(6, latent_sequence.shape[1])))

    plt.figure(figsize=figsize)

    time_steps = np.arange(len(latent_sequence))

    for i, dim in enumerate(dims):
        plt.plot(time_steps, latent_sequence[:, dim], label=f'z_{dim}')

    plt.xlabel('Time step')
    plt.ylabel('Latent value')
    plt.title('Latent dimensions over time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
