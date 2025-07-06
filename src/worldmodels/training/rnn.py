from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.worldmodels.models.rnn import MDN_LSTM


@dataclass
class TrainCfg:
    epochs: int = 12
    lr: float = 1.0e-4
    step_size: int = 4
    gamma: float = 0.5
    batch_size: int = 32


def init_hidden(model: MDN_LSTM, batch: int):
    h = torch.zeros(model.cfg.num_layers, batch, model.hidden_size, device=next(model.parameters()).device)
    return (h, h.clone())  # (hidden, cell)


def train(model: MDN_LSTM, loader: DataLoader, cfg: TrainCfg):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    model.train()

    for epoch in range(cfg.epochs):
        tot, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B, T, C = xb.shape
            h = init_hidden(model, B)

            # burn‑in (⚠ detach after loop)
            for t in range(model.t_warm_up):
                _, _, _, h = model(xb[:, t], h)
            if isinstance(h, tuple):
                h = tuple(i.detach() for i in h)
            else:
                h = h.detach()

            out = {k: [] for k in ("wl", "mu", "ls")}
            for t in range(model.t_warm_up, T):
                wl, mu, ls, h = model(xb[:, t], h)
                out["wl"].append(wl)
                out["mu"].append(mu)
                out["ls"].append(ls)

            L = T - model.t_warm_up
            out = {
                "weight_logits": torch.stack(out["wl"], 1).reshape(B * L, -1),
                "means": torch.stack(out["mu"], 1).reshape(B * L, model.cfg.num_gaussians, C),
                "log_stds": torch.stack(out["ls"], 1).reshape(B * L, model.cfg.num_gaussians, C),
            }
            target = yb[:, model.t_warm_up:].reshape(B * L, C)
            loss = model.loss(target, **out)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * B
            seen += B
        sched.step()
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | LR {sched.get_last_lr()[0]:.1e} | loss {tot / seen:.4f}"
        )
