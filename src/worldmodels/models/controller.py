"""
Controller for a World-Models CarRacing agent.

The policy receives the VAE latent vector **z_t** (dim = VAE.latent) and the
top-layer LSTM hidden state **h_t** (dim = RNN.hidden_size).  It outputs a
continuous three-dimensional action:

    [steer  ∈ (-1,1),   gas ∈ (0,1),   brake ∈ (0,1)]

The file also provides helpers to flatten / unflatten the parameter vector so
that the controller can be optimised by CMA-ES.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.worldmodels.models.rnn import MDN_LSTM
from src.worldmodels.models.vae import VAE


# ────────────────────────────────────────────────────────────────────────────────
#  Weight-vector helpers
# ────────────────────────────────────────────────────────────────────────────────


def _params_to_vector(net: nn.Module) -> np.ndarray:
    """Flatten all parameters into a single 1-D NumPy array."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def _vector_to_params(net: nn.Module, vec: np.ndarray) -> None:
    """Load a flat parameter vector back into a network *in-place*."""
    idx = 0
    for p in net.parameters():
        numel = p.numel()
        block = vec[idx: idx + numel].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        idx += numel


# ─────────────────────────────────────────────────────────────────────────
#  Config dataclass  (avoids pickling issues)
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ControllerCfg:
    input_size: int


# ────────────────────────────────────────────────────────────────────────────────
#  Policy network
# ────────────────────────────────────────────────────────────────────────────────


class PolicyNet(nn.Module):
    """
    Minimal 2-layer MLP controller:

        (z_t, h_t)  →  (steer, gas, brake)
    """

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_size, 3)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, input_size) – concatenated [z_t , h_t]

        Returns
        -------
        actions : (B, 3)  with correct squashing:
            steer  ∈ (-1,1)   via tanh
            gas    ∈ (0,1)    via sigmoid
            brake  ∈ (0,1)    via sigmoid
        """
        raw = self.fc(x)

        steer = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])

        return torch.cat([steer, gas, brake], dim=1)

    # ------------------------------------------------------------------ #
    def act(self, controller_input: np.ndarray) -> np.ndarray:
        """Greedy action for a *single* concatenated input (1-D NumPy)."""
        x = torch.from_numpy(controller_input).float().unsqueeze(0)  # (1,D)
        with torch.no_grad():
            a = self.forward(x).squeeze(0).cpu().numpy()
        return a

    # ------------------------------------------------------------------ #
    #  CMA-ES rollout
    # ------------------------------------------------------------------ #
    @classmethod
    def rollout(
            cls,
            policy_params: np.ndarray,
            rollouts_per_eval: int,
            vae_path: str,
            rnn_path: str,
            device: str = "cpu",
            env_seed: int | None = None,
    ) -> float:
        """
        Evaluate one controller in CarRacing-v3.

        Returns **negative** average reward so that CMA-ES (which minimises)
        can be used directly.
        """
        dev = torch.device(device)
        vae, rnn = _load_models(vae_path, rnn_path, dev)
        latent_dim = vae.latent
        hidden_dim = rnn.hidden_size

        policy = cls(input_size=latent_dim + hidden_dim)
        _vector_to_params(policy, policy_params)
        policy.eval()

        rewards: list[float] = []
        for _ in range(rollouts_per_eval):
            env = gym.make("CarRacing-v3", render_mode=None)
            if env_seed is not None:
                env.reset(seed=env_seed)

            obs, _ = env.reset()
            # LSTM hidden state (h, c) each: (layers=1, batch=1, hidden_dim)
            h = (
                torch.zeros(rnn.cfg.num_layers, 1, hidden_dim, device=dev),
                torch.zeros(rnn.cfg.num_layers, 1, hidden_dim, device=dev),
            )

            ep_reward, done = 0.0, False
            while not done:
                # (1) frame → latent z_t
                with torch.no_grad():
                    z_t = _encode_frame(obs, vae, dev)

                # (2) controller input & action
                h_flat = h[0][-1, 0]  # take last LSTM layer, batch-dim squeezed → (hidden_dim,)
                ctrl_in = torch.cat([z_t, h_flat], dim=0).detach().cpu().numpy()
                a = policy.act(ctrl_in)  # (3,)

                # (3) env step
                obs, reward, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_reward += reward

                # (4) update RNN hidden:  (z_t, a_t) → h_{t+1}
                za_t = torch.cat([z_t, torch.from_numpy(a).to(dev)], dim=0).unsqueeze(0)
                _, _, _, h = rnn(za_t, h)

            env.close()
            rewards.append(ep_reward)

        # CMA-ES minimises the objective
        return -float(np.mean(rewards))

    def save_model(self, path: str | os.PathLike) -> None:
        """Save weights and config as a pure-dict checkpoint (pickle-safe)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cfg = ControllerCfg(
            input_size=self.fc.in_features
        )
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": asdict(cfg),
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str | os.PathLike, device: str | torch.device = "cpu"):
        """
        Load a controller checkpoint saved by `save_model`.
        Works even under PyTorch safe-unpickling (weights_only=True).
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg_dict = ckpt["config"]
        model = cls(**cfg_dict).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model


# ────────────────────────────────────────────────────────────────────────────────
#  Per-process model cache & helpers
# ────────────────────────────────────────────────────────────────────────────────

_CACHED_MODELS: dict[tuple[str, str, str], tuple[VAE, MDN_LSTM]] = {}


def _load_models(vae_path: str, rnn_path: str, device: torch.device):
    """
    Load (or retrieve cached) VAE & RNN on the requested device.
    """
    key = (vae_path, rnn_path, device.type)
    if key not in _CACHED_MODELS:
        vae = VAE.load_model(vae_path, device)
        rnn = MDN_LSTM.load_model(rnn_path, device)
        vae.eval()
        rnn.eval()
        _CACHED_MODELS[key] = (vae, rnn)
    return _CACHED_MODELS[key]


def _encode_frame(frame: np.ndarray, vae: VAE, device: torch.device) -> torch.Tensor:
    """
    Resize a raw CarRacing frame to 64×64, normalise to [0,1],
    encode via the VAE, and return μ as a 1-D tensor on *device*.
    """
    img = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        mu, _ = vae.encode(x)
    return mu.squeeze(0)  # (latent_dim,)


# Expose helpers for the training script
__all__ = [
    "PolicyNet",
    "_params_to_vector",
    "_vector_to_params",
    "_load_models",
    "_encode_frame",
]
