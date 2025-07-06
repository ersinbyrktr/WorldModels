#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm.auto import tqdm

from worldmodels_data import make_dataloaders


# ———————————————————— utils ————————————————————

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_grid(t: torch.Tensor, fp: str, nrow: int = 8):
    vutils.save_image(t.clamp(0, 1).cpu(), fp, nrow=nrow)


# ———————————————————— model ————————————————————
class VAE(nn.Module):
    def __init__(self, *, latent: int = 20, recon_loss: str = "BCE"):
        super().__init__()
        assert recon_loss in {"L2", "BCE"}
        self.latent = latent
        self.recon_loss = recon_loss
        self.enc = nn.Sequential(  # output size
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 → 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 → 4
            nn.ReLU(inplace=True),
        )
        self.flat = 256 * 4 * 4
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)
        self.fc = nn.Linear(latent, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 → 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)  # 32 → 64
        )
        if recon_loss == "L2": self.dec.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), torch.clamp(self.logvar(h), -10, 10)

    def decode(self, z):
        h = self.fc(z).reshape(-1, 256, 4, 4)
        return self.dec(h)

    def reparam(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL[q(z|x) || N(0,1)]   (nats per image, averaged over the batch)

        mu, logvar : shape  [B, latent]
        returns     : scalar
        """
        # element-wise KL for each latent dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # shape [B, latent]

        # nats *per image*  (sum over latent dims)
        kl_per_image = kl_per_dim.sum(dim=1)  # shape [B]

        # average over the batch
        return kl_per_image.mean()

    def loss_terms(self, x, xr, mu, logvar):
        rc = (F.mse_loss if self.recon_loss == "L2" else F.binary_cross_entropy_with_logits)(xr, x,
                                                                                             reduction="sum") / x.size(
            0)
        return rc, self._kl(mu, logvar)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent, device=next(self.parameters()).device)
            x = self.decode(z)
            return torch.sigmoid(x) if self.recon_loss == "BCE" else x

    def reconstruct(self, x):
        with torch.no_grad(): xr, _, _ = self(x); return torch.sigmoid(xr) if self.recon_loss == "BCE" else xr

    def per_sample_terms(self, x, xr, mu, logvar):
        """
        Returns reconstruction loss  *and*  KL **per data-point** (no batch
        averaging) so we can build the IWAE bound.

        Shapes
        -------
        x , xr : [N, 1, 28, 28]
        mu, logvar : [N, latent]

        Returns
        -------
        rc : [N]  reconstruction in nats / image
        kl : [N]  KL in nats / image
        """
        # reconstruction (sum over pixels)
        rc = F.binary_cross_entropy_with_logits(
            xr, x, reduction="none"
        ).flatten(1).sum(dim=1)  # [N]

        # KL (sum over latent dims)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1. - logvar).sum(dim=1)  # [N]
        return rc, kl


# ———————————————————— loop helpers ————————————————————

def evaluate(model, loader, beta):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x in loader:
            x = x.to(next(model.parameters()).device)
            xr, mu, lv = model(x)
            rc, kl = model.loss_terms(x, xr, mu, lv)
            tot += rc.item() + beta * kl.item()
    return tot / len(loader)


def train(model, train_loader, test_loader, *, epochs: int, lr: float,
          beta: float, kl_on: bool, warm: int, outdir: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    os.makedirs(outdir, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(dev.type == "cuda"))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    fixed = next(iter(test_loader))[:64].to(dev)
    real_row = fixed[:8]

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = run_rc = run_kl = 0.0
        beta_t = beta * min(1.0, ep / warm)

        pbar = tqdm(enumerate(train_loader, 1),
                    total=len(train_loader),
                    desc=f"Epoch {ep}/{epochs}")

        for idx, x in pbar:
            x = x.to(dev, non_blocking=True)

            with torch.amp.autocast(device_type=dev.type,
                                    enabled=dev.type == "cuda",
                                    dtype=torch.float16):
                xr, mu, lv = model(x)
                rc, kl = model.loss_terms(x, xr, mu, lv)
                loss = rc + (beta_t * kl if kl_on else 0.0)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()
            run_rc += rc.item()
            run_kl += kl.item()

            pbar.set_postfix(loss=run_loss / idx)

        # validation (total loss only – unchanged)
        val = evaluate(model, test_loader, beta_t)

        # averages for this epoch
        avg_loss = run_loss / len(train_loader)
        avg_rc = run_rc / len(train_loader)
        avg_kl = run_kl / len(train_loader)

        # original print
        print(f"Epoch {ep}: train={avg_loss:.4f} val={val:.4f} β={beta_t:.4f}")
        # extra line with rc & kl
        print(f"           rc={avg_rc:.4f} kl={avg_kl:.4f}")

        # sample & recon grids
        save_grid(model.sample(64), f"{outdir}/sample_ep{ep}.png")
        with torch.no_grad():
            recon_row = model.reconstruct(real_row)     # [8, 3, 64, 64]
            recon_grid = torch.cat([real_row, recon_row], dim=0)  # [16, 3, 64, 64]

        save_grid(recon_grid, f"{outdir}/recon_ep{ep}.png", nrow=8)


# ———————————————————— CLI ————————————————————

def main():
    ap = argparse.ArgumentParser(description="Convolutional VAE on World Models (Py 3.13 safe)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--loss", choices=["L2", "BCE"], default="L2")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="world_models_elbo_outputs")
    args = ap.parse_args()
    set_seed(args.seed)
    train_loader, test_loader = make_dataloaders(
        "data/carracing",
        batch_size=args.batch,
        num_workers=min(16, os.cpu_count()),
        seed=args.seed,
    )
    model = VAE(latent=args.latent, recon_loss=args.loss)
    print(f"Initial val {evaluate(model, test_loader, args.beta):.4f}")
    train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, beta=args.beta, kl_on=not args.no_kl,
          warm=args.warm, outdir=args.outdir)


if __name__ == "__main__":
    main()
