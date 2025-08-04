#!/usr/bin/env python
# models/vae.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.worldmodels.data.data_loader import make_vae_dataloaders
from src.worldmodels.envs import EnvKind
from src.worldmodels.utils.paths import (
    get_data_path,
    get_vae_model_dir,
    get_viz_path,
)
from src.worldmodels.utils.utils import save_grid


# ────────────────────────────────────────────────────────────────────────────────
# Config (paths derived from EnvKind, collection_name, name)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class VaeRunCfg:
    # All paths are derived from these:
    #   data_root  = ../../../data/{env_short}_{collection_name}
    #   model_dir  = ../../../trained_{env_short}_model/{name}
    #   viz_dir    = ../../../trained_{env_short}_model/{name}/viz
    env: EnvKind = EnvKind.BIPEDAL_WALKER
    collection_name: str = "default"
    name: str = "baseline"

    # model
    latent: int = 32
    recon_loss: str = "L2"  # "L2" | "BCE"

    # train
    epochs: int = 1
    lr: float = 3e-4
    beta: float = 1.0
    kl_on: bool = True
    warm: int = 1
    batch_size: int = 512
    num_workers: int = field(default_factory=lambda: min(16, os.cpu_count() or 16))

    # system
    device: str = "auto"  # "auto" | "cpu" | "cuda"


# ────────────────────────────────────────────────────────────────────────────────
# Eval
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: "VAE", loader, beta: float) -> float:
    """
    Average total loss (reconstruction + beta * KL) over the loader.
    """
    model.eval()
    dev = next(model.parameters()).device
    tot = 0.0
    for x in loader:
        x = x.to(dev)
        xr, mu, lv = model(x)
        rc, kl = model.loss_terms(x, xr, mu, lv)
        tot += rc.item() + beta * kl.item()
    return tot / len(loader)


# ────────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────────

def train(model: "VAE", train_loader, test_loader, cfg: VaeRunCfg) -> None:
    dev = _resolve_device(cfg)
    model.to(dev)

    model_dir = get_vae_model_dir(cfg.env, cfg.name)
    viz_dir = get_viz_path(cfg.env, cfg.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, fused=(dev.type == "cuda"))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    fixed = next(iter(test_loader))[:64].to(dev)
    real_row = fixed[:8]

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = run_rc = run_kl = 0.0
        beta_t = cfg.beta * min(1.0, ep / max(1, cfg.warm))

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {ep}/{cfg.epochs}")
        for idx, x in pbar:
            x = x.to(dev, non_blocking=True)

            with torch.amp.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                xr, mu, lv = model(x)
                rc, kl = model.loss_terms(x, xr, mu, lv)
                loss = rc + (beta_t * kl if cfg.kl_on else 0.0)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()
            run_rc += rc.item()
            run_kl += kl.item()
            pbar.set_postfix(loss=run_loss / idx)

        # validation (reporting only)
        val = evaluate(model, test_loader, beta_t)
        avg_loss = run_loss / len(train_loader)
        avg_rc = run_rc / len(train_loader)
        avg_kl = run_kl / len(train_loader)

        print(f"Epoch {ep}: train={avg_loss:.4f} val={val:.4f} β={beta_t:.4f}")
        print(f"           rc={avg_rc:.4f} kl={avg_kl:.4f}")

        # samples & reconstructions
        save_grid(model.sample(64), str(viz_dir / f"sample_ep{ep}.png"))
        with torch.no_grad():
            recon_row = model.reconstruct(real_row)
            recon_grid = torch.cat([real_row, recon_row], dim=0)
        save_grid(recon_grid, str(viz_dir / f"recon_ep{ep}.png"), nrow=8)

        # final checkpoint
        if ep == cfg.epochs:
            latest_path = model_dir / "vae_latest.pt"
            model.save_model(str(latest_path))
            print(f"VAE Model saved to {latest_path}")


# ────────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, *, latent: int = 20, recon_loss: str = "BCE"):
        super().__init__()
        assert recon_loss in {"L2", "BCE"}
        self.latent = latent
        self.recon_loss = recon_loss

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),  # 64 → 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 32 → 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 16 → 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True)  # 8 → 4
        )
        self.flat = 256 * 4 * 4
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)
        self.fc = nn.Linear(latent, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 4 → 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 8 → 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),  # 16 → 32
            nn.ConvTranspose2d(32, 3, 4, 2, 1)  # 32 → 64
        )
        if recon_loss == "L2":
            self.dec.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        return self.mu(h), torch.clamp(self.logvar(h), -10, 10)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(-1, 256, 4, 4)
        return self.dec(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl_per_dim.sum(dim=1).mean()

    def loss_terms(self, x: torch.Tensor, xr: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        rc_fn = F.mse_loss if self.recon_loss == "L2" else F.binary_cross_entropy_with_logits
        rc = rc_fn(xr, x, reduction="sum") / x.size(0)
        return rc, self._kl(mu, logvar)

    def sample(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n, self.latent, device=next(self.parameters()).device)
            x = self.decode(z)
            return torch.sigmoid(x) if self.recon_loss == "BCE" else x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xr, _, _ = self(x)
            return torch.sigmoid(xr) if self.recon_loss == "BCE" else xr

    def per_sample_terms(self, x: torch.Tensor, xr: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        rc = F.binary_cross_entropy_with_logits(xr, x, reduction="none").flatten(1).sum(dim=1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
        return rc, kl

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"model_state_dict": self.state_dict(), "latent": self.latent, "recon_loss": self.recon_loss},
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> "VAE":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        model = cls(latent=checkpoint["latent"], recon_loss=checkpoint["recon_loss"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

    @classmethod
    def load_latest(cls, env: EnvKind, name: str, device: Optional[torch.device] = None) -> "VAE":
        """
        Load the latest VAE checkpoint for a given (env, name) run:
            path = ../../../trained_{env_short}_model/{name}/vae_latest.pt
        """
        ckpt = get_vae_model_dir(env, name) / "vae_latest.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
        return cls.load_model(str(ckpt), device=device)


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _resolve_device(cfg: VaeRunCfg) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloaders(cfg: VaeRunCfg):
    data_root = get_data_path(cfg.env, cfg.collection_name)
    return make_vae_dataloaders(
        str(data_root),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )


def _init_or_load_model(cfg: VaeRunCfg) -> VAE:
    """
    Try loading an existing checkpoint via `VAE.load_latest`; otherwise init a new model.
    """
    try:
        model = VAE.load_latest(cfg.env, cfg.name)
        print(f"Loaded VAE: latent={model.latent}, recon_loss={model.recon_loss}")
        return model
    except FileNotFoundError:
        print("No existing VAE found — initializing new model")
        return VAE(latent=cfg.latent, recon_loss=cfg.recon_loss)


# ────────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────────

def run(cfg: Optional[VaeRunCfg] = None) -> None:
    cfg = cfg or VaeRunCfg()
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = _build_dataloaders(cfg)
    model = _init_or_load_model(cfg)

    print(f"Initial val {evaluate(model, test_loader, cfg.beta):.4f}")
    train(model, train_loader, test_loader, cfg)


if __name__ == "__main__":
    cfg = VaeRunCfg(
        env=EnvKind.CARRACING,  # or EnvKind.BIPEDAL_WALKER
        collection_name="baseline",  # matches collector's name used for data
        name="vae_small",  # run/model name
        epochs=1,
    )
    run(cfg)
