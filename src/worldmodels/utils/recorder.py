#!/usr/bin/env python
# scripts/record_episode.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from gymnasium import wrappers

from src.worldmodels.envs import EnvKind
from src.worldmodels.models.controller import PolicyNet
from src.worldmodels.models.rnn import MDN_LSTM
from src.worldmodels.models.vae import VAE
from src.worldmodels.utils.paths import get_controller_model_dir
from src.worldmodels.utils.utils import _encode_frame, _obs_to_frame


# ────────────────────────────────────────────────────────────────────────────────
# Config (paths derived from EnvKind, vae_name, rnn_name, name)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RecorderCfg:
    env: EnvKind
    vae_name: str
    rnn_name: str
    controller_name: str
    device: str = "cpu"
    render_mode: str = "rgb_array"
    seed: Optional[int] = None

    # Optional overrides
    out_dir: Optional[Path] = None  # default: ../../../trained_{env}_model/{name}/videos
    name_prefix: Optional[str] = None  # default: "{name}_play"


# ────────────────────────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────────────────────────

def _video_paths(cfg: RecorderCfg) -> Tuple[Path, str]:
    """
    Returns (video_dir, file_prefix)
    """
    ctrl_dir = get_controller_model_dir(cfg.env, cfg.controller_name)
    video_dir = cfg.out_dir if cfg.out_dir is not None else (ctrl_dir / "videos")
    prefix = cfg.name_prefix if cfg.name_prefix is not None else f"{cfg.controller_name}_play"
    return video_dir, prefix


# ────────────────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────────────────

def record_single_episode(cfg: RecorderCfg) -> None:
    video_dir, prefix = _video_paths(cfg)
    video_dir.mkdir(parents=True, exist_ok=True)

    # env (via adapter)
    adapter = cfg.env.adapter_cls(render_mode=cfg.render_mode, seed=cfg.seed)
    env = adapter.make_env()()

    rec_env = wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=prefix,
        episode_trigger=lambda ep: True,  # first episode only
    )

    # models (use new load_latest helpers)
    dev = torch.device(cfg.device)
    vae = VAE.load_latest(cfg.env, cfg.vae_name, device=dev)
    rnn = MDN_LSTM.load_latest(cfg.env, cfg.rnn_name, device=dev)
    policy = PolicyNet.load_latest(cfg.env, cfg.controller_name, device=dev)

    vae.eval()
    rnn.eval()
    policy.eval()

    # rollout
    h = (
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
    )
    obs, _ = rec_env.reset(seed=cfg.seed)
    done, total_reward = False, 0.0

    while not done:
        frame = _obs_to_frame(env, obs)
        with torch.no_grad():
            z = _encode_frame(frame, vae, dev)  # (latent,)
        ctrl_in = torch.cat([z, h[0][-1, 0]], dim=0)
        action = policy.act(ctrl_in.detach().cpu().numpy()).astype(np.float32)

        obs, r, term, trunc, _ = rec_env.step(action)
        done = bool(term or trunc)
        total_reward += float(r)

        za = torch.cat([z, torch.from_numpy(action).to(dev)], dim=0).unsqueeze(0)
        with torch.no_grad():
            _, _, _, h = rnn(za, h)

    rec_env.close()
    print(f"[record] saved episode to {Path(video_dir).resolve()}")
    print(f"[record] episode reward = {total_reward:.1f}")


# ────────────────────────────────────────────────────────────────────────────────
# Example
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = RecorderCfg(
        env=EnvKind.CARRACING,  # or EnvKind.CARRACING
        vae_name="baseline",
        rnn_name="baseline",
        controller_name="baseline",
        device="cpu",
    )
    record_single_episode(cfg)
