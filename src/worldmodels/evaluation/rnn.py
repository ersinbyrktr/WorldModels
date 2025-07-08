# evaluation/rnn.py
# ----------------------------------------------------------
# Evaluation utilities for the MDN-RNN model
# ----------------------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from torch.utils.data import DataLoader

from src.worldmodels.models.rnn import MDN_LSTM, _np


@torch.no_grad()
def evaluate_latent_prediction(model: MDN_LSTM, val_loader: DataLoader, device: torch.device = None):
    """Evaluate the RNN on a validation set of latent sequences."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    count = 0

    for x_batch, y_batch, lens in val_loader:
        x_batch, y_batch, lens = (
            x_batch.to(device),
            y_batch.to(device),
            lens.to(device),
        )
        batch_size, seq_len, _ = x_batch.shape
        latent_dim = y_batch.shape[2]

        # Initialize hidden state
        h = (
            torch.zeros(model.cfg.num_layers, batch_size, model.hidden_size, device=device),
            torch.zeros(model.cfg.num_layers, batch_size, model.hidden_size, device=device)
        )

        # Prediction phase
        out = {k: [] for k in ("wl", "mu", "ls")}
        for t in range(seq_len):
            wl, mu, ls, h = model(x_batch[:, t], h)
            out["wl"].append(wl)
            out["mu"].append(mu)
            out["ls"].append(ls)

        # Calculate loss
        wl = torch.stack(out["wl"], 1)  # (B,T,K)
        mu = torch.stack(out["mu"], 1)  # (B,T,K,C)
        ls = torch.stack(out["ls"], 1)

        wl = wl.reshape(batch_size * seq_len, -1)
        mu = mu.reshape(batch_size * seq_len, model.cfg.num_gaussians, latent_dim)
        ls = ls.reshape_as(mu)
        target = y_batch.reshape(batch_size * seq_len, latent_dim)

        # mask padded timesteps
        mask = (torch.arange(seq_len, device=device)[None, :] < lens[:, None]).reshape(-1)

        loss = model.loss(
            target[mask],
            weight_logits=wl[mask],
            means=mu[mask],
            log_stds=ls[mask],
        )

        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / count


@torch.no_grad()
def visualize_latent_predictions(model: MDN_LSTM, x_sequence: torch.Tensor, y_sequence: torch.Tensor | None = None,
                                 dim_indices=(0, 1, 2, 32, 33, 34)):
    """Visualize the prediction of the RNN on a latent sequence.

    Args:
        model: The trained MDN_LSTM model
        latent_sequence: A single latent sequence of shape (T, latent_dim)
        dim_indices: Indices of latent dimensions to visualize
    """
    model.eval()
    device = next(model.parameters()).device

    # Ensure (T, latent_dim) float32 on device
    x_sequence = torch.as_tensor(x_sequence, dtype=torch.float32, device=device)
    if y_sequence is not None:
        y_sequence = torch.as_tensor(y_sequence, dtype=torch.float32, device=device)

    T, Cin = x_sequence.shape
    latent_dim = model.cfg.output_size

    # Get predictions
    preds = []
    h0 = (
        torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=device),
        torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=device)
    )
    h = tuple(x.clone() for x in h0)
    for t in range(T):
        wl, mu, ls, h = model(x_sequence[t:t + 1], h)
        y_hat = model.predict(wl, mu, ls, deterministic=True)  # (1, latent_dim)
        preds.append(y_hat.squeeze(0).cpu())

    preds = torch.stack(preds)  # (T, latent_dim)

    # Plotting
    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(10, 3 * len(dim_indices)), sharex=True)
    if len(dim_indices) == 1:
        axes = [axes]

    t = np.arange(T)

    for i, dim_idx in enumerate(dim_indices):
        ax = axes[i]

        # Plot ground truth
        if y_sequence is not None:
            ax.plot(t, _np(y_sequence.cpu())[:, dim_idx], label="ground truth", lw=1.2)
        else:
            ax.plot(t, _np(x_sequence.cpu())[:, dim_idx], label="ground truth", lw=1.2)

        # Plot prediction
        ax.plot(t, _np(preds)[:, dim_idx], label="prediction", lw=1.2)

        ax.set_ylabel(f"z_{dim_idx}")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("time step")
    fig.suptitle(f"Latent Space Predictions")
    plt.tight_layout()
    plt.show()

    return preds  # (T, latent_dim)
