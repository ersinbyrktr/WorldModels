# models/rnn.py

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.worldmodels.evaluation.batch_eval import evaluate
from src.worldmodels.training.rnn import TrainCfg, train


def _np(x):
    """Torch → NumPy (1‑D or 2‑D) & squeeze the last dim if singleton."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    # VAE integration params
    p.add_argument("--vae-path", type=str, default="../../../trained_model/vae_latest.pt",
                   help="Path to trained VAE model")
    p.add_argument("--data-path", type=str, default="../../../data/carracing",
                   help="Path to image data directory")
    p.add_argument("--sequence-length", type=int, default=100,
                   help="Length of sequences for RNN training")
    p.add_argument("--model-save-path", type=str, default="../../../trained_models/rnn_model.pt",
                   help="Path to save the trained model")

    cfg_ns = p.parse_args(args)

    torch.manual_seed(cfg_ns.seed)
    np.random.seed(cfg_ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use VAE latent space
    from src.worldmodels.models.vae import VAE
    from src.worldmodels.data.latent_dataset import create_rnn_latent_dataloaders

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
    loader, val_loader = create_rnn_latent_dataloaders(
        vae=vae_model,
        data_root=cfg_ns.data_path,
        num_workers=cfg_ns.workers,
        # batch_size=cfg_ns.bs,
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

    # For VAE latents, visualize a few validation samples
    from src.worldmodels.utils.plotting import plot_latent_components
    from src.worldmodels.evaluation.rnn import visualize_latent_predictions
    val_nll = evaluate(model, val_loader)
    print(f"Validation NLL / frame: {val_nll:.4f}")
    # Get a sample batch from validation
    xb, _, _ = next(iter(val_loader))  # xb : (B,Tmax,C) with padding
    sample_sequence = xb[0].to(device)  # pick first episode in the batch

    # Plot predictions for selected dimensions
    visualize_latent_predictions(model, sample_sequence, dim_indices=(0, 1, 2))

    # Also show the latent components over time
    plot_latent_components(sample_sequence.cpu().numpy())


if __name__ == "__main__":
    main()
