#!/usr/bin/env python
# models/vae.py
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.worldmodels.data.data_loader import make_vae_dataloaders
from src.worldmodels.evaluation.vae import evaluate
from src.worldmodels.training.vae import train


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

    def save_model(self, path):
        """
        Save model state to disk

        Parameters
        ----------
        path : str
            Path where the model will be saved
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'latent': self.latent,
            'recon_loss': self.recon_loss
        }, path)

    @classmethod
    def load_model(cls, path, device=None):

        """
        Load a saved model from disk

        Parameters
        ----------
        path : str
            Path to the saved model
        device : torch.device, optional
            Device to load the model to

        Returns
        -------
        model : VAE
            Loaded model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)
        model = cls(latent=checkpoint['latent'], recon_loss=checkpoint['recon_loss'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


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
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="world_models_elbo_outputs")
    ap.add_argument("--model-path", type=str, default="../../../trained_model",
                    help="Path to save/load model checkpoints")
    ap.add_argument("--load-model", type=str, default="",
                    help="Path to load a pretrained model")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader = make_vae_dataloaders(
        "../../../data/carracing",
        batch_size=args.batch,
        num_workers=min(16, os.cpu_count())
    )

    # Load model if specified, otherwise create a new one
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = VAE.load_model(args.load_model)
        print(f"Model loaded with latent dim={model.latent}, recon_loss={model.recon_loss}")
    else:
        model = VAE(latent=args.latent, recon_loss=args.loss)

    print(f"Initial val {evaluate(model, test_loader, args.beta):.4f}")
    train(model, train_loader, test_loader,
          epochs=args.epochs,
          lr=args.lr,
          beta=args.beta,
          kl_on=not args.no_kl,
          warm=args.warm,
          outdir=args.outdir,
          model_path=args.model_path)


if __name__ == "__main__":
    main()
