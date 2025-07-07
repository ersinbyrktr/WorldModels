"""Minimal‑docs refactor of the WorldModels MDN‑RNN/LSTM demo.
Keeps original behaviour but moves all user‑tunable knobs into
plain dataclass configs or CLI args.

Only tricky bits are commented inline ✔️.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import ticker
from torch.utils.data import DataLoader, Dataset

from src.worldmodels.training.rnn import TrainCfg, train


def _np(x):
    """Torch → NumPy (1‑D or 2‑D) & squeeze the last dim if singleton."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def sliding_window_predict_and_plot(
        model: MDN_LSTM,
        series: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = True,
        title: str = "MDN‑LSTM fit",
        ax: plt.Axes | None = None,
):
    """
    One‑step‑ahead predictions with a sliding window burn‑in,
    then a side‑by‑side plot of ground truth vs. forecasts.
    """
    model.eval()
    dev = next(model.parameters()).device

    # ensure (T, C) float32 on device
    series = torch.as_tensor(series, dtype=torch.float32, device=dev)
    if series.ndim == 1:
        series = series.unsqueeze(-1)
    T, C = series.shape

    preds = []
    h0 = (torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=dev),
          torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=dev))

    for start in range(T):
        # fresh hidden each window
        h = tuple(x.clone() for x in h0)

        # 2) one‑step forecast
        wl, mu, ls, _ = model(series[start:start + 1], h)
        y_hat = model.predict(wl, mu, ls, deterministic=deterministic)  # (1,C)
        preds.append(y_hat.squeeze(0).cpu())

    preds = torch.stack(preds)  # (T, C)

    # ────── plotting ───────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(T)
    ax.plot(t, _np(series)[:, 0], label="ground truth", lw=1.2)
    ax.plot(t, _np(preds)[:, 0], label="prediction", lw=1.2)
    ax.set_xlabel("time step")
    ax.set_ylabel("xₜ")
    ax.set_title(f"{title} ({'det' if deterministic else 'sampled'})")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout();
    plt.show()

    return preds  # (T, C)


# ────────────────────────────────────────────────────────────────────────────────
#  Configs
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    input_size: int = 32  # Default to VAE latent dimension
    output_size: int = 32  # Default to VAE latent dimension
    hidden_size: int = 768  # 512 + 256
    num_layers: int = 1  # LSTM only
    num_gaussians: int = 5  # World Models uses 5 mixtures


# ────────────────────────────────────────────────────────────────────────────────
#  Synthetic data helpers (unchanged API)
# ────────────────────────────────────────────────────────────────────────────────

def sine_regime_switch(
        T: int = 10_000,
        amplitudes: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
        dwell_range: Tuple[int, int] = (200, 400),
        omega: float = 0.05,
        phi: Optional[float] = None,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi) if phi is None else phi
    sin_part = np.sin(omega * np.arange(T) + phi)

    A = np.empty(T)
    idx = 0
    while idx < T:
        A_cur = rng.choice(amplitudes)
        seg_len = rng.integers(*dwell_range)
        seg_end = min(idx + seg_len, T)
        A[idx:seg_end] = A_cur
        idx = seg_end

    x = A * sin_part + rng.normal(0, noise_std, size=T)
    return x.astype(np.float32)[:, None], A.astype(np.float32)


def sine_multi_component(
        T: int = 10_000,
        amplitudes: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
        omegas: Optional[Sequence[float]] = None,
        phi: Optional[Sequence[float]] = None,
        noise_std: float = 0.1,
        return_parts: bool = False,
        seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    K = len(amplitudes)
    amplitudes = np.asarray(amplitudes, dtype=np.float32)

    if omegas is None:
        omegas = rng.uniform(0.02, 0.10, size=K)
    omegas = np.broadcast_to(omegas, K).astype(np.float32)

    if phi is None:
        phi = rng.uniform(0, 2 * np.pi, size=K)
    phi = np.broadcast_to(phi, K).astype(np.float32)

    t = np.arange(T, dtype=np.float32)
    parts = np.stack(
        [A * np.sin(om * t + ph) for A, om, ph in zip(amplitudes, omegas, phi)], axis=1
    )
    x = parts.sum(axis=1) + rng.normal(0, noise_std, size=T)

    if return_parts:
        return x.astype(np.float32)[:, None], parts.astype(np.float32)
    return x.astype(np.float32)[:, None]


def multivariate_regime_switch(
        T: int = 12_000,
        C: int = 3,
        n_components: int = 4,
        dwell_range: Tuple[int, int] = (120, 300),
        mean_scale: float = 3.0,
        noise_range: Tuple[float, float] = (0.05, 0.2),
        seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-mean_scale, mean_scale, size=(n_components, C))
    sigmas = rng.uniform(*noise_range, size=(n_components, C))

    z = np.empty(T, dtype=np.int64)
    x = np.empty((T, C), dtype=np.float32)

    idx = 0
    while idx < T:
        k = rng.integers(n_components)
        seg_len = rng.integers(*dwell_range)
        seg_end = min(idx + seg_len, T)
        eps = rng.normal(size=(seg_end - idx, C))
        x[idx:seg_end] = means[k] + sigmas[k] * eps
        z[idx:seg_end] = k
        idx = seg_end
    return x, z, means.astype(np.float32), sigmas.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────────
#  Dataset + tiny plotting util (unchanged logic, minor tidy‑ups)
# ────────────────────────────────────────────────────────────────────────────────

class ToyDataset(Dataset):
    def __init__(self, seq: np.ndarray, block_size: int):
        self.x = torch.as_tensor(seq, dtype=torch.float32)
        self.block_size = block_size

    def __len__(self):
        return len(self.x) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.x[idx: idx + self.block_size]
        y = self.x[idx + 1: idx + self.block_size + 1]
        return x, y


# ────────────────────────────────────────────────────────────────────────────────
#  Base MDN helper (shared by RNN & LSTM)
# ────────────────────────────────────────────────────────────────────────────────

def mdn_loss(
        target: torch.Tensor,
        weight_logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
):
    """Negative log‑likelihood for a diagonal Gaussian mixture.

    Args:
        target: Shape (B, D) where D is latent dimension
        weight_logits: Shape (B, K) where K is number of mixtures
        means: Shape (B, K, D)
        log_stds: Shape (B, K, D)
    """
    target = target.unsqueeze(1)  # (B, D) → (B, 1, D)
    inv_var = torch.exp(-2 * log_stds)
    # Calculate log probability for each dimension and sum across dimensions
    log_prob = (
            -0.5 * (target - means).pow(2) * inv_var - log_stds - 0.5 * math.log(2 * math.pi)
    ).sum(dim=-1)  # Sum across dimensions, result: (B, K)
    log_mix = F.log_softmax(weight_logits, dim=1)  # (B, K)
    # Log-sum-exp trick for numerical stability
    return -torch.logsumexp(log_prob + log_mix, dim=-1).mean()  # Mean over batch


def mdn_predict(
        weight_logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        deterministic: bool = True,
):
    if deterministic:
        probs = F.softmax(weight_logits, dim=-1)
        return (probs.unsqueeze(-1) * means).sum(dim=1)
    probs = F.softmax(weight_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    idx = idx.unsqueeze(-1).expand(-1, -1, means.size(-1))
    mu = torch.gather(means, 1, idx)
    ls = torch.gather(log_stds, 1, idx)
    return mu.squeeze(1) + torch.randn_like(mu) * torch.exp(ls).clamp(1e-6, 1e2)


# ────────────────────────────────────────────────────────────────────────────────
#  Models
# ────────────────────────────────────────────────────────────────────────────────

class _MDN_Core(nn.Module):
    """Linear heads shared by both RNN flavours."""

    def __init__(self, hidden: int, out: int, k: int):
        super().__init__()
        self.weight_logits = nn.Linear(hidden, k)
        self.means = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, k * out))
        self.log_stds = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, k * out))
        self.k, self.out = k, out

    def forward(self, h: torch.Tensor):
        wl = self.weight_logits(h)
        mu = self.means(h).view(-1, self.k, self.out)
        ls = self.log_stds(h).view(-1, self.k, self.out)
        return wl, mu, ls


class MDN_LSTM(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            cfg.input_size, cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True
        )
        self.mdn = _MDN_Core(cfg.hidden_size, cfg.output_size, cfg.num_gaussians)

    @property
    def hidden_size(self):
        return self.cfg.hidden_size

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        out, hidden = self.lstm(x.unsqueeze(1), hidden)  # (B,1,C)
        wl, mu, ls = self.mdn(out.squeeze(1))
        return wl, mu, ls, hidden

    loss = staticmethod(mdn_loss)
    predict = staticmethod(mdn_predict)

    def save_model(self, path):
        """Save model state to disk"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.cfg
        }, path)

    @classmethod
    def load_model(cls, path, device=None):
        """Load a saved model from disk"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


# ────────────────────────────────────────────────────────────────────────────────
#  Entry‑point (CLI or import)
# ────────────────────────────────────────────────────────────────────────────────

def main(args: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Train MDN‑RNN/LSTM on VAE latent space")
    # basic hyper‑params
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    # VAE integration params
    p.add_argument("--vae-path", type=str, default="../../../trained_model/vae_latest.pt",
                   help="Path to trained VAE model")
    p.add_argument("--data-path", type=str, default="../../../data/carracing",
                   help="Path to image data directory")
    p.add_argument("--sequence-length", type=int, default=100,
                   help="Length of sequences for RNN training")
    p.add_argument("--latent-cache", type=str, default=None,
                   help="Optional path to cache/load encoded latent vectors")
    p.add_argument("--demo-mode", action="store_true", default=False,
                   help="Run with toy data instead of VAE latents (for testing)")
    p.add_argument("--model-save-path", type=str, default="../../../trained_models/rnn_model.pt",
                   help="Path to save the trained model")

    cfg_ns = p.parse_args(args)

    torch.manual_seed(cfg_ns.seed)
    np.random.seed(cfg_ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model configuration
    if cfg_ns.demo_mode:
        # Use toy data for testing
        print("Running in demo mode with synthetic data")
        model_cfg = ModelCfg(
            input_size=1,  # Toy data is 1D
            output_size=1,
            hidden_size=cfg_ns.hidden,
            num_layers=cfg_ns.layers,
        )
        train_cfg = TrainCfg(epochs=cfg_ns.epochs, lr=cfg_ns.lr, batch_size=cfg_ns.bs)

        # Generate synthetic data
        x_sum, _ = sine_multi_component(
            T=1500,
            amplitudes=(0.8, 2.5, 4.0, 1.2),
            omegas=(0.025, 0.050, 0.085, 0.140),
            seed=cfg_ns.seed,
            return_parts=True
        )
        ds = ToyDataset(x_sum, block_size=cfg_ns.sequence_length)
        loader = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = loader  # In demo mode, use same loader for validation
    else:
        # Use VAE latent space
        from src.worldmodels.models.vae import VAE
        from src.worldmodels.data.latent_dataset import create_latent_dataloaders

        # Load trained VAE
        print(f"Loading VAE model from {cfg_ns.vae_path}")
        vae_model = VAE.load_model(cfg_ns.vae_path, device)
        print(f"VAE loaded with latent dimension: {vae_model.latent}")

        # Create model config with VAE latent dimensions
        model_cfg = ModelCfg(
            input_size=vae_model.latent,
            output_size=vae_model.latent,
            hidden_size=cfg_ns.hidden,
            num_layers=cfg_ns.layers,
        )
        train_cfg = TrainCfg(epochs=cfg_ns.epochs, lr=cfg_ns.lr, batch_size=cfg_ns.bs)

        # Create latent sequence dataloaders
        loader, val_loader = create_latent_dataloaders(
            vae_model=vae_model,
            data_root=cfg_ns.data_path,
            batch_size=cfg_ns.bs,
            cached_latents_path=cfg_ns.latent_cache,
            device=device
        )

    # Create and train model
    model = MDN_LSTM(model_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train(model, loader, train_cfg)

    # Save the trained model
    import os
    os.makedirs(os.path.dirname(cfg_ns.model_save_path), exist_ok=True)
    model.save_model(cfg_ns.model_save_path)
    print(f"Model saved to {cfg_ns.model_save_path}")

    # Visualize results
    if cfg_ns.demo_mode:
        # For toy data, use the existing plot function
        sliding_window_predict_and_plot(
            model,
            ds.x,  # full raw sequence
            deterministic=True,
            title="Demo fit on toy 1-D data",
        )
    else:
        # For VAE latents, visualize a few validation samples
        from src.worldmodels.evaluation.rnn import visualize_latent_predictions
        from src.worldmodels.utils.plotting import plot_latent_components

        # Get a sample batch from validation
        for x_batch, y_batch in val_loader:
            sample_sequence = x_batch[0].to(device)  # Take first sequence
            break

        # Plot predictions for selected dimensions
        visualize_latent_predictions(model, sample_sequence, dim_indices=(0, 1, 2))

        # Also show the latent components over time
        plot_latent_components(sample_sequence.cpu().numpy())


if __name__ == "__main__":
    main()
