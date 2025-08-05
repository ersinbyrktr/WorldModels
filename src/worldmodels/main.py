#!/usr/bin/env python
# main.py
"""
End-to-end, single-machine World-Models pipeline.

Steps
-----
1. Collect latent-training episodes **once** (skipped if dataset folder exists)
2. Train / resume a VAE on the collected dataset
3. Train / resume an MDN-RNN using the frozen VAE
4. Train / resume a CMA-ES controller that plugs the VAE+RNN into the env
"""

from __future__ import annotations

import multiprocessing as mp

from models.controller import run as ctrl_run, ControllerCfg
from models.rnn import run as rnn_run, RNNRunCfg, MDN_LSTM
from models.vae import run as vae_run, VaeRunCfg, VAE
# ── pipeline stages (all expose run(cfg)) ────────────────────────────────────
from src.worldmodels.data.collect_episodes import run as collect_run, CollectCfg
from src.worldmodels.envs import EnvKind
from src.worldmodels.utils.config_helper import _write_pipeline_config
from src.worldmodels.utils.paths import (
    get_data_path,
    get_vae_model_dir,
    get_rnn_model_dir,
    get_controller_model_dir,
)

# ════════════════════════════════════════════════════════════════════════════
#  User-editable parameters
# ════════════════════════════════════════════════════════════════════════════
ENV = EnvKind.PENDULUM  # or EnvKind.BIPEDAL_WALKER
RUN_NAME = "baseline"

NUM_EPISODES = 2_000  # for collector (only if needed)
NUM_WORKERS = 12  # env subprocesses for collector

VAE_EPOCHS = 5
RNN_EPOCHS = 50


# Controller CMA-ES params sit inside ControllerCfg below
# ════════════════════════════════════════════════════════════════════════════


def configure():
    mp.set_start_method("spawn", force=True)
    collect_cfg = CollectCfg(
        env=ENV,
        name=RUN_NAME,
        episodes=NUM_EPISODES,
        workers=NUM_WORKERS,
    )
    vae_cfg = VaeRunCfg(
        env=ENV,
        collection_name=RUN_NAME,
        name=RUN_NAME,
        epochs=VAE_EPOCHS,
        batch_size=8,
    )
    rnn_cfg = RNNRunCfg(
        env=ENV,
        collection_name=RUN_NAME,
        name=RUN_NAME,
        vae=None,  # assign later
        epochs=RNN_EPOCHS,
        batch_size=8,
        step_size=10,
        hidden=512,
    )
    ctrl_cfg = ControllerCfg(
        env=ENV,
        name=RUN_NAME,
        vae=None,
        rnn=None,
        vae_name=RUN_NAME,  # so that worker processes can lazy-load
        rnn_name=RUN_NAME,
        device="cpu",
        maxiter=20,
        popsize=16,
        sigma0=0.3,
        rollouts=4,
        workers=8,
    )
    # Write a single JSON file with all configs into the controller folder
    _ = _write_pipeline_config(
        ENV,
        RUN_NAME,
        collect_cfg=collect_cfg,
        vae_cfg=vae_cfg,
        rnn_cfg=rnn_cfg,
        ctrl_cfg=ctrl_cfg,
    )
    return collect_cfg, ctrl_cfg, rnn_cfg, vae_cfg


def main() -> None:
    # ---------------------------------------------------------------------- #
    # 0. Configs
    # ---------------------------------------------------------------------- #
    collect_cfg, ctrl_cfg, rnn_cfg, vae_cfg = configure()
    # ---------------------------------------------------------------------- #
    # 1. Episodes – collect once
    # ---------------------------------------------------------------------- #
    data_path = get_data_path(ENV, RUN_NAME)
    if not data_path.is_dir():
        print(f"[main] Collecting {NUM_EPISODES} episodes → {data_path}")
        collect_run(collect_cfg)
    else:
        print(f"[main] Dataset folder already exists – skipping collection: {data_path}")

    # ---------------------------------------------------------------------- #
    # 2. VAE – train / resume
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── VAE stage ─────────────────────────────────────────")
    vae_run(vae_cfg)

    # Load the trained VAE for downstream stages
    vae_ckpt_dir = get_vae_model_dir(ENV, RUN_NAME)
    vae = VAE.load_latest(ENV, RUN_NAME)
    print(f"[main] Loaded VAE from {vae_ckpt_dir}")

    # ---------------------------------------------------------------------- #
    # 3. RNN – train / resume
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── RNN stage ─────────────────────────────────────────")
    rnn_cfg.vae = vae
    rnn_run(rnn_cfg)

    # Load the trained RNN for the controller
    rnn_ckpt_dir = get_rnn_model_dir(ENV, RUN_NAME)
    rnn = MDN_LSTM.load_latest(ENV, RUN_NAME)
    print(f"[main] Loaded RNN from {rnn_ckpt_dir}")

    # ---------------------------------------------------------------------- #
    # 4. Controller – CMA-ES
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── Controller stage ──────────────────────────────────")
    ctrl_cfg.vae = vae
    ctrl_cfg.rnn = rnn
    ctrl_run(ctrl_cfg)

    # ---------------------------------------------------------------------- #
    print("\n[main] Pipeline completed successfully.")
    print(f"  • Episodes folder : {data_path}")
    print(f"  • VAE dir         : {vae_ckpt_dir}")
    print(f"  • RNN dir         : {rnn_ckpt_dir}")
    print(f"  • CTRL dir        : {get_controller_model_dir(ENV, RUN_NAME)}")


if __name__ == "__main__":
    main()
