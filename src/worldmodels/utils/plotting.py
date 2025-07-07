# plotting.py
# ----------------------------------------------------------
# Visualization utilities for World Models
# ----------------------------------------------------------
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


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
