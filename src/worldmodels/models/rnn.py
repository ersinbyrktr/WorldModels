# models/rnn.py

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional

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

    def save_model(self, path: str | os.PathLike):
        """Save model state to disk"""
        import os, dataclasses
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cfg_dict = dataclasses.asdict(self.cfg)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': cfg_dict
        }, path)

    @classmethod
    def load_model(cls, path: str | os.PathLike, device=None):
        """
        Load an MDN-LSTM checkpoint saved in either OLD (pickled dataclass)
        or NEW (dict-only) format, regardless of where `ModelCfg` lived.

        • First we inject `ModelCfg` into `__main__` so legacy pickles that
          reference  __main__.ModelCfg  can resolve it.
        • Then we *try* the new safer route  (weights_only=True)  with an
          explicit allow-list (`safe_globals`).
        • If that still fails, we fall back to  weights_only=False  (full
          unpickle) – but only after allow-listing the class.
        """
        import __main__, torch.serialization as _ser

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Make sure legacy pickles can resolve  __main__.ModelCfg
        setattr(__main__, "ModelCfg", ModelCfg)

        safe = {
            "ModelCfg": ModelCfg,
            "__main__.ModelCfg": ModelCfg,
            "src.worldmodels.models.rnn.ModelCfg": ModelCfg,
        }

        # 1) Preferred: dict-only checkpoint (no arbitrary code)
        try:
            ckpt = torch.load(
                path,
                map_location=device,
                weights_only=True,
                safe_globals=safe,
            )
        except Exception:
            # 2) Legacy: full pickle (we still keep it safe by allow-listing)
            _ser.add_safe_globals(safe)
            ckpt = torch.load(
                path,
                map_location=device,
                weights_only=False,  # full pickle
            )

        cfg_raw = ckpt["model_config"]
        cfg = ModelCfg(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

        model = cls(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model


# ────────────────────────────────────────────────────────────────────────────────
#  Entry‑point (CLI or import)
# ────────────────────────────────────────────────────────────────────────────────

def main(args: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Train MDN‑RNN/LSTM on VAE latent space")
    # basic hyper‑params
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=0)
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
    p.add_argument("--model-save-path", type=str, default="../../../trained_model/rnn_model.pt",
                   help="Path to save the trained model")
    p.add_argument("--load-model", type=str, default="../../../trained_model/rnn_model.pt",
                   help="Path to a previously-trained RNN checkpoint (.pt). "
                        "If given, the model will be loaded instead of "
                        "initialised from scratch.")

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
    train_dl, val_dl, action_dim = create_rnn_latent_dataloaders(
        vae=vae_model,
        data_root=cfg_ns.data_path,
        num_workers=cfg_ns.workers,
        device=device
    )
    if cfg_ns.load_model:
        from pathlib import Path
        ckpt_path = Path(cfg_ns.load_model).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--load-model file not found: {ckpt_path}")
        print(f"Loading RNN model from {ckpt_path}")
        model = MDN_LSTM.load_model(str(ckpt_path), device)
    else:
        model_cfg = ModelCfg(
            input_size=vae_model.latent + action_dim,
            output_size=vae_model.latent,
            hidden_size=cfg_ns.hidden,
            num_layers=cfg_ns.layers,
        )
        model = MDN_LSTM(model_cfg).to(device)
        print("Initialised new RNN model")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_cfg = TrainCfg(epochs=cfg_ns.epochs, lr=cfg_ns.lr, batch_size=cfg_ns.bs)
    loader, val_loader = train_dl, val_dl

    if train_cfg.epochs > 0:
        train(model, loader, train_cfg)

        # Save the trained model
        import os
        os.makedirs(os.path.dirname(cfg_ns.model_save_path), exist_ok=True)
        model.save_model(cfg_ns.model_save_path)
        print(f"Model saved to {cfg_ns.model_save_path}")
    else:
        print("Skipping training")

    # Visualize results

    # For VAE latents, visualize a few validation samples
    from src.worldmodels.evaluation.rnn import visualize_latent_predictions
    val_nll = evaluate(model, loader)
    print(f"Validation NLL / frame: {val_nll:.4f}")
    # Get a sample batch from validation
    xb, yb, _ = next(iter(loader))  # xb : (B,Tmax,C) with padding
    sample_x = xb[0].to(device)
    sample_y = yb[0].to(device)

    visualize_latent_predictions(model, sample_x, sample_y, dim_indices=(0, 1, 2, 3, 4, 5, 6))


if __name__ == "__main__":
    main()
