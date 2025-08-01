"""
Controller for a World-Models agent.

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
import numpy as np
import torch
import torch.nn as nn

import src.worldmodels.envs.bipedal_walker as BipedalWalkerEnv
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


def _looks_like_image(arr: np.ndarray) -> bool:
    if arr.ndim == 3 and (arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4)):
        return True
    if arr.ndim == 2:
        return True
    return False


def _obs_to_frame(env, obs: np.ndarray) -> np.ndarray:
    """Return an HWC image frame for the current state."""
    if isinstance(obs, np.ndarray) and _looks_like_image(obs):
        return obs
    # Otherwise, fall back to renderer (requires render_mode="rgb_array")
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Create env with render_mode='rgb_array'."
        )
    return frame


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

    def __init__(self, input_size: int, action_bounds: list[tuple[float, float]]):
        """
        Parameters
        ----------
        input_size : int
            Size of the concatenated [z_t , h_t] vector.
        env : gymnasium.Env
            A single (not vector‑wrapped) environment instance.
        """
        super().__init__()

        self.action_bounds = action_bounds
        act_dim = len(self.action_bounds)

        # simple 1‑layer MLP head — replace with your own architecture if needed
        self.fc = nn.Linear(input_size, act_dim)

        # build masks once, keep them on whatever device the model lives on
        tanh_mask = [low == -1 and high == 1 for low, high in self.action_bounds]
        sigmoid_mask = [low == 0 and high == 1 for low, high in self.action_bounds]

        # register as buffers so they follow `.to(device)` / `.cuda()` calls
        self.register_buffer("_tanh_mask", torch.tensor(tanh_mask, dtype=torch.bool))
        self.register_buffer("_sigmoid_mask", torch.tensor(sigmoid_mask, dtype=torch.bool))

        if not (self._tanh_mask.any() or self._sigmoid_mask.any()):
            raise ValueError("No dimensions matched (-1,1) or (0,1) bounds.")

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, input_size)
            Concatenated features [z_t , h_t].

        Returns
        -------
        actions : torch.Tensor, shape (B, action_dim)
            Each dimension is squashed to its correct range:
            • (-1, 1) → tanh
            • ( 0, 1) → sigmoid
        """
        raw = self.fc(x)  # (B, action_dim)
        actions = torch.empty_like(raw)

        if self._tanh_mask.any():
            actions[:, self._tanh_mask] = torch.tanh(raw[:, self._tanh_mask])
        if self._sigmoid_mask.any():
            actions[:, self._sigmoid_mask] = torch.sigmoid(raw[:, self._sigmoid_mask])

        return actions

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
        Evaluate one controller

        Returns **negative** average reward so that CMA-ES (which minimises)
        can be used directly.
        """
        dev = torch.device(device)
        vae, rnn = _load_models(vae_path, rnn_path, dev)
        latent_dim = vae.latent
        hidden_dim = rnn.hidden_size

        policy = cls(input_size=latent_dim + hidden_dim, action_bounds=BipedalWalkerEnv.action_space)
        _vector_to_params(policy, policy_params)
        policy.eval()

        rewards: list[float] = []
        for _ in range(rollouts_per_eval):
            # OPTIONAL but clearer: ensure render_mode returns RGB arrays
            env = BipedalWalkerEnv.make_env(render_mode="rgb_array")()  # ← change (optional)
            if env_seed is not None:
                env.reset(seed=env_seed)

            obs, _ = env.reset()
            h = (
                torch.zeros(rnn.cfg.num_layers, 1, hidden_dim, device=dev),
                torch.zeros(rnn.cfg.num_layers, 1, hidden_dim, device=dev),
            )

            ep_reward, done = 0.0, False
            while not done:
                # (1) frame → latent z_t
                frame = _obs_to_frame(env, obs)  # ← NEW
                with torch.no_grad():
                    z_t = _encode_frame(frame, vae, dev)  # ← replace obs → frame

                # (2) controller input & action
                h_flat = h[0][-1, 0]
                ctrl_in = torch.cat([z_t, h_flat], dim=0).detach().cpu().numpy()
                a = policy.act(ctrl_in)

                # (3) env step
                obs, reward, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_reward += reward

                # (4) update RNN hidden
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


def _encode_frame(img: np.ndarray, vae, device: torch.device) -> torch.Tensor:
    """
    Accepts a frame in common layouts and returns z_t (latent from VAE).

    Accepts:
      - HWC RGB/RGBA  (H, W, 3/4)
      - CHW RGB/RGBA  (3/4, H, W)
      - Grayscale     (H, W)       -> replicated to 3 channels

    Resizes to 64x64 (if your VAE was trained on 64x64) and converts to [0,1] float.
    """

    img = np.asarray(img)

    # ---- Normalize layout to HWC, 3 channels
    if img.ndim == 2:
        # grayscale HxW -> HxWx3
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.ndim == 3:
        # CHW -> HWC
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        # RGBA -> RGB
        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
    else:
        raise ValueError(f"_encode_frame: unsupported input shape {img.shape}")

    # ---- Ensure uint8 HWC 64x64x3
    if img.dtype != np.uint8:
        # If 0..1 floats, scale to 0..255; otherwise clip
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if 0.0 <= vmin and vmax <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        z_mu, _ = vae.encode(x)  # adjust if your VAE API differs
    return z_mu.squeeze(0)


# Expose helpers for the training script
__all__ = [
    "PolicyNet",
    "_params_to_vector",
    "_vector_to_params",
    "_load_models",
    "_encode_frame",
]
