# training/rnn.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR


@dataclass
class TrainCfg:
    epochs: int = 12
    lr: float = 1.0e-4
    step_size: int = 4
    gamma: float = 0.5
    batch_size: int = 32


def init_hidden(model: nn.Module, batch: int):
    h = torch.zeros(model.cfg.num_layers, batch, model.hidden_size, device=next(model.parameters()).device)
    return (h, h.clone())  # (hidden, cell)


def train(model, loader, cfg):
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)

    for epoch in range(cfg.epochs):
        model.train()
        tot, seen = 0.0, 0

        for xb, yb, lens in loader:  # xb : (B,Tmax,C)
            xb, yb, lens = xb.to(dev), yb.to(dev), lens.to(dev)
            B, Tmax, Cin = xb.shape
            Cout = yb.shape[2]

            # sort lengths ↓ for pack; restore order later if you want
            lens, sort_idx = lens.sort(descending=True)
            xb, yb = xb[sort_idx], yb[sort_idx]

            h0 = (
                torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
                torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev)
            )

            packed_in = pack_padded_sequence(xb, lens.cpu(), batch_first=True)
            packed_out, _ = model.lstm(packed_in, h0)  # (∑lens, H)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B,Tmax,H)

            wl, mu, ls = model.mdn(out.reshape(-1, model.hidden_size))
            wl = wl.reshape(B, Tmax, -1)
            mu = mu.reshape(B, Tmax, model.cfg.num_gaussians, Cout)
            ls = ls.reshape_as(mu)

            # ---------- mask away the padding -----------------------------
            mask = (torch.arange(Tmax, device=dev)[None, :] < lens[:, None])  # (B,T)
            mask_f = mask.reshape(-1)  # (B*T)

            target = yb.reshape(-1, Cout)[mask_f]
            out_kw = {
                "weight_logits": wl.reshape(-1, model.cfg.num_gaussians)[mask_f],
                "means": mu.reshape(-1, model.cfg.num_gaussians, Cout)[mask_f],
                "log_stds": ls.reshape(-1, model.cfg.num_gaussians, Cout)[mask_f],
            }
            loss = model.loss(target, **out_kw)

            # --------------------------------------------------------------
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item() * B
            seen += B

        sched.step()
        print(f"Epoch {epoch + 1}/{cfg.epochs} | loss {tot / seen:.4f}")
