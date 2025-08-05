# models/rnn.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import ticker
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR

from src.worldmodels.envs import EnvKind
from src.worldmodels.utils.paths import (
    get_data_path,  # ../../../data/{env_short}_{collection}
    get_rnn_model_dir,  # ../../../trained_{env_short}_model/{name}
)


# ────────────────────────────────────────────────────────────────────────────────
# Config (VAE object is passed directly; paths derived from EnvKind & names)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RNNRunCfg:
    # Derive dataset/checkpoint paths from these
    env: EnvKind = EnvKind.BIPEDAL_WALKER
    collection_name: str = "default"  # -> data path
    name: str = "rnn_baseline"  # -> RNN checkpoint folder

    # Provide the VAE *instance* directly (loaded by caller via VAE.load_latest)
    vae: "VAE" | None = None

    # Model
    hidden: int = 256
    layers: int = 1

    # Train
    epochs: int = 20
    lr: float = 1e-4
    step_size: int = 4
    gamma: float = 0.5
    batch_size: int = 8  # kept for compatibility; dataset builder controls batching

    # System
    seed: int = 1
    workers: int = 16
    device: str = "auto"  # "auto" | "cpu" | "cuda"


# ────────────────────────────────────────────────────────────────────────────────
# Eval / viz
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: "MDN_LSTM", loader):
    dev = next(model.parameters()).device
    model.eval()
    tot_nll, tot_frames = 0.0, 0
    for xb, yb, lens in loader:
        xb, yb, lens = xb.to(dev), yb.to(dev), lens.to(dev)
        B, Tmax, Cin = xb.shape
        Cout = yb.shape[2]

        lens, sort_idx = lens.sort(descending=True)
        xb, yb = xb[sort_idx], yb[sort_idx]

        h0 = (
            torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
            torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
        )
        packed_in = pack_padded_sequence(xb, lens.cpu(), batch_first=True)
        packed_out, _ = model.lstm(packed_in, h0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=Tmax)

        wl, mu, ls = model.mdn(out.reshape(-1, model.hidden_size))
        wl = wl.reshape(B, Tmax, -1)
        mu = mu.reshape(B, Tmax, model.cfg.num_gaussians, Cout)
        ls = ls.reshape_as(mu)

        mask = (torch.arange(Tmax, device=dev)[None, :] < lens[:, None]).reshape(-1)
        target = yb.reshape(-1, Cout)[mask]
        out_kw = {
            "weight_logits": wl.reshape(-1, model.cfg.num_gaussians)[mask],
            "means": mu.reshape(-1, model.cfg.num_gaussians, Cout)[mask],
            "log_stds": ls.reshape(-1, model.cfg.num_gaussians, Cout)[mask],
        }
        loss = model.loss(target, **out_kw)
        tot_nll += loss.item() * lens.sum().item()
        tot_frames += lens.sum().item()
    return tot_nll / tot_frames


@torch.no_grad()
def visualize_latent_predictions(
        model: "MDN_LSTM",
        x_sequence: torch.Tensor,
        y_sequence: torch.Tensor | None = None,
        dim_indices=(0, 1, 2, 32, 33, 34),
):
    model.eval()
    device = next(model.parameters()).device
    x_sequence = torch.as_tensor(x_sequence, dtype=torch.float32, device=device)
    if y_sequence is not None:
        y_sequence = torch.as_tensor(y_sequence, dtype=torch.float32, device=device)

    T, _ = x_sequence.shape
    preds = []
    h = (
        torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=device),
        torch.zeros(model.cfg.num_layers, 1, model.hidden_size, device=device),
    )
    for t in range(T):
        wl, mu, ls, h = model(x_sequence[t:t + 1], h)
        y_hat = model.predict(wl, mu, ls, deterministic=True)
        preds.append(y_hat.squeeze(0).cpu())
    preds = torch.stack(preds)

    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(10, 3 * len(dim_indices)), sharex=True)
    if len(dim_indices) == 1:
        axes = [axes]
    t = np.arange(T)
    for i, d in enumerate(dim_indices):
        ax = axes[i]
        if y_sequence is not None:
            ax.plot(t, _np(y_sequence.cpu())[:, d], label="ground truth", lw=1.2)
        else:
            ax.plot(t, _np(x_sequence.cpu())[:, d], label="ground truth", lw=1.2)
        ax.plot(t, _np(preds)[:, d], label="prediction", lw=1.2)
        ax.set_ylabel(f"z_{d}")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[-1].set_xlabel("time step")
    fig.suptitle("Latent Space Predictions")
    plt.tight_layout()
    plt.show()
    return preds


# ────────────────────────────────────────────────────────────────────────────────
# Loss / predict
# ────────────────────────────────────────────────────────────────────────────────

def mdn_loss(
        target: torch.Tensor,
        weight_logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
):
    target = target.unsqueeze(1)
    inv_var = torch.exp(-2 * log_stds)
    log_prob = (-0.5 * (target - means).pow(2) * inv_var - log_stds - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
    log_mix = F.log_softmax(weight_logits, dim=1)
    return -torch.logsumexp(log_prob + log_mix, dim=-1).mean()


def mdn_predict(
        weight_logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        deterministic: bool = True,
):
    probs = F.softmax(weight_logits, dim=-1)
    if deterministic:
        return (probs.unsqueeze(-1) * means).sum(dim=1)
    idx = torch.multinomial(probs, 1).unsqueeze(-1).expand(-1, -1, means.size(-1))
    mu = torch.gather(means, 1, idx)
    ls = torch.gather(log_stds, 1, idx)
    return mu.squeeze(1) + torch.randn_like(mu) * torch.exp(ls).clamp(1e-6, 1e2)


# ────────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    input_size: int = 32
    output_size: int = 32
    hidden_size: int = 768
    num_layers: int = 1
    num_gaussians: int = 5


class _MDN_Core(nn.Module):
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
        self.lstm = nn.LSTM(cfg.input_size, cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True)
        self.mdn = _MDN_Core(cfg.hidden_size, cfg.output_size, cfg.num_gaussians)

    @property
    def hidden_size(self):
        return self.cfg.hidden_size

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        out, hidden = self.lstm(x.unsqueeze(1), hidden)
        wl, mu, ls = self.mdn(out.squeeze(1))
        return wl, mu, ls, hidden

    loss = staticmethod(mdn_loss)
    predict = staticmethod(mdn_predict)

    def save_model(self, path: str | os.PathLike):
        import dataclasses
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "model_config": dataclasses.asdict(self.cfg)}, path)

    @classmethod
    def load_model(cls, path: str | os.PathLike, device=None):
        import __main__, torch.serialization as _ser
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setattr(__main__, "ModelCfg", ModelCfg)
        safe = {"ModelCfg": ModelCfg, "__main__.ModelCfg": ModelCfg, "src.worldmodels.models.rnn.ModelCfg": ModelCfg}
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True, safe_globals=safe)
        except Exception:
            _ser.add_safe_globals(safe)
            ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_raw = ckpt["model_config"]
        cfg = ModelCfg(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw
        m = cls(cfg).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        return m

    @classmethod
    def load_latest(cls, env: EnvKind, name: str, device: Optional[torch.device] = None) -> "MDN_LSTM":
        """
        Load the latest RNN checkpoint for a given (env, name) run:
            path = ../../../trained_{env_short}_model/{name}/rnn_latest.pt
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = get_rnn_model_dir(env, name) / "rnn_latest.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"RNN checkpoint not found: {ckpt}")
        return cls.load_model(str(ckpt), device=device)


# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────

def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_device(cfg: RNNRunCfg) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(cfg: RNNRunCfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)


def _build_dataloaders(cfg: RNNRunCfg, device: torch.device):
    from src.worldmodels.data.latent_dataset import create_rnn_latent_dataloaders

    if cfg.vae is None:
        raise ValueError("cfg.vae must be provided")

    # --- true action-dim comes from the adapter, not from the raw .npy arrays
    adapter = cfg.env.adapter_cls(render_mode="rgb_array", seed=None)
    action_dim_expected = len(adapter.action_bounds())

    data_root = get_data_path(cfg.env, cfg.collection_name)
    return create_rnn_latent_dataloaders(
        vae=cfg.vae,
        data_root=str(data_root),
        action_dim_expected=action_dim_expected,  # ← NEW
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        device=device,
    )


def _ensure_header_matches(model: "MDN_LSTM", vae_latent: int, action_dim: int) -> None:
    """Fail fast if a loaded checkpoint has incompatible header."""
    exp_in = int(vae_latent + action_dim)
    exp_out = int(vae_latent)
    got_in = int(getattr(model.cfg, "input_size", -1))
    got_out = int(getattr(model.cfg, "output_size", -1))
    if got_out != exp_out or got_in != exp_in:
        raise RuntimeError(
            "Loaded RNN checkpoint is incompatible with current VAE/dataset.\n"
            f"  expected input_size = vae.latent + action_dim = {vae_latent} + {action_dim} = {exp_in}\n"
            f"  got      input_size = {got_in}\n"
            f"  expected output_size (latent) = {exp_out}\n"
            f"  got      output_size          = {got_out}\n"
            "Please retrain the RNN for this env / VAE (use a new cfg.name or delete the old run folder)."
        )


def _init_or_load_model(cfg: RNNRunCfg, device: torch.device, action_dim: int) -> MDN_LSTM:
    if cfg.vae is None:
        raise ValueError("RNNRunCfg.vae must be provided (pass an already loaded VAE).")

    rnn_dir = get_rnn_model_dir(cfg.env, cfg.name)
    rnn_ckpt = rnn_dir / "rnn_latest.pt"

    # If checkpoint exists, *validate header matches* expected dims.
    if rnn_ckpt.is_file():
        print(f"Loading RNN from {rnn_ckpt}")
        m = MDN_LSTM.load_model(str(rnn_ckpt), device)
        _ensure_header_matches(m, cfg.vae.latent, action_dim)  # ← FAIL FAST if mismatched
        return m

    # Otherwise initialise a *new* compatible model
    model_cfg = ModelCfg(
        input_size=cfg.vae.latent + action_dim,
        output_size=cfg.vae.latent,
        hidden_size=cfg.hidden,
        num_layers=cfg.layers,
    )
    print("Initialised new RNN model")
    rnn_dir.mkdir(parents=True, exist_ok=True)
    return MDN_LSTM(model_cfg).to(device)


def _train_loop(model: MDN_LSTM, loader, cfg: RNNRunCfg):
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    for epoch in range(cfg.epochs):
        model.train()
        tot, seen = 0.0, 0
        for xb, yb, lens in loader:
            xb, yb, lens = xb.to(dev), yb.to(dev), lens.to(dev)
            B, Tmax, Cin = xb.shape
            Cout = yb.shape[2]

            lens, sort_idx = lens.sort(descending=True)
            xb, yb = xb[sort_idx], yb[sort_idx]

            h0 = (
                torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
                torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
            )
            packed_in = pack_padded_sequence(xb, lens.cpu(), batch_first=True)
            packed_out, _ = model.lstm(packed_in, h0)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)

            wl, mu, ls = model.mdn(out.reshape(-1, model.hidden_size))
            wl = wl.reshape(B, Tmax, -1)
            mu = mu.reshape(B, Tmax, model.cfg.num_gaussians, Cout)
            ls = ls.reshape_as(mu)

            mask = (torch.arange(Tmax, device=dev)[None, :] < lens[:, None]).reshape(-1)
            target = yb.reshape(-1, Cout)[mask]
            out_kw = {
                "weight_logits": wl.reshape(-1, model.cfg.num_gaussians)[mask],
                "means": mu.reshape(-1, model.cfg.num_gaussians, Cout)[mask],
                "log_stds": ls.reshape(-1, model.cfg.num_gaussians, Cout)[mask],
            }
            loss = model.loss(target, **out_kw)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item() * B
            seen += B
        sched.step()
        print(f"Epoch {epoch + 1}/{cfg.epochs} | loss {tot / seen:.4f}")


def _save(model: MDN_LSTM, cfg: RNNRunCfg):
    rnn_dir = get_rnn_model_dir(cfg.env, cfg.name)
    rnn_dir.mkdir(parents=True, exist_ok=True)
    ckpt = rnn_dir / "rnn_latest.pt"
    model.save_model(str(ckpt))
    print(f"Model saved to {ckpt}")


def _final_eval_and_viz(model: MDN_LSTM, loader):
    val_nll = evaluate(model, loader)
    print(f"Validation NLL / frame: {val_nll:.4f}")
    xb, yb, _ = next(iter(loader))
    device = next(model.parameters()).device
    visualize_latent_predictions(model, xb[0].to(device), yb[0].to(device), dim_indices=(0, 1, 2, 3, 4, 5, 6))


# ────────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────────

def run(cfg: Optional[RNNRunCfg] = None):
    cfg = cfg or RNNRunCfg()
    if cfg.vae is None:
        raise ValueError("RNNRunCfg.vae must be provided (use VAE.load_latest and pass it in).")

    _set_seed(cfg)
    device = _resolve_device(cfg)

    train_dl, val_dl, action_dim = _build_dataloaders(cfg, device)
    model = _init_or_load_model(cfg, device, action_dim)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if cfg.epochs > 0:
        _train_loop(model, train_dl, cfg)
        _save(model, cfg)
    else:
        print("Skipping training")

    _final_eval_and_viz(model, train_dl)


if __name__ == "__main__":
    # Example usage: load VAE externally and pass it in.
    from src.worldmodels.models.vae import VAE

    env = EnvKind.PENDULUM
    vae_name = "baseline"

    vae = VAE.load_latest(env=env, name=vae_name)  # loads ../../../trained_{env}_model/{vae_name}/vae_latest.pt

    cfg = RNNRunCfg(
        env=env,
        collection_name="baseline",
        name="baseline",  # separate RNN run folder
        vae=vae,
        epochs=5,
        hidden=512,
        step_size=5,
        lr=1e-6,
    )
    run(cfg)
