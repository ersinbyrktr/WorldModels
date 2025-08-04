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
from src.worldmodels.utils.paths import (
    get_data_path,
    get_vae_model_dir,
    get_rnn_model_dir,
    get_controller_model_dir,
)

# ════════════════════════════════════════════════════════════════════════════
#  User-editable parameters
# ════════════════════════════════════════════════════════════════════════════
ENV = EnvKind.CARRACING  # or EnvKind.BIPEDAL_WALKER
COLLECTION = "baseline"  # episode folder suffix
VAE_NAME = "vae_small"
RNN_NAME = "rnn_small"
CTRL_NAME = "ctrl_small"

NUM_EPISODES = 1_00  # for collector (only if needed)
NUM_WORKERS = 12  # env subprocesses for collector

VAE_EPOCHS = 0
RNN_EPOCHS = 0


# Controller CMA-ES params sit inside ControllerCfg below
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    mp.set_start_method("spawn", force=True)

    # ---------------------------------------------------------------------- #
    # 1. Episodes – collect once
    # ---------------------------------------------------------------------- #
    data_path = get_data_path(ENV, COLLECTION)
    if not data_path.is_dir():
        print(f"[main] Collecting {NUM_EPISODES} episodes → {data_path}")
        collect_cfg = CollectCfg(
            env=ENV,
            name=COLLECTION,
            episodes=NUM_EPISODES,
            workers=NUM_WORKERS,
        )
        collect_run(collect_cfg)
    else:
        print(f"[main] Dataset folder already exists – skipping collection: {data_path}")

    # ---------------------------------------------------------------------- #
    # 2. VAE – train / resume
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── VAE stage ─────────────────────────────────────────")
    vae_cfg = VaeRunCfg(
        env=ENV,
        collection_name=COLLECTION,
        name=VAE_NAME,
        epochs=VAE_EPOCHS,
    )
    vae_run(vae_cfg)

    # Load the trained VAE for downstream stages
    vae_ckpt_dir = get_vae_model_dir(ENV, VAE_NAME)
    vae = VAE.load_latest(ENV, VAE_NAME)
    print(f"[main] Loaded VAE from {vae_ckpt_dir}")

    # ---------------------------------------------------------------------- #
    # 3. RNN – train / resume
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── RNN stage ─────────────────────────────────────────")
    rnn_cfg = RNNRunCfg(
        env=ENV,
        collection_name=COLLECTION,
        name=RNN_NAME,
        vae=vae,  # pass the instance (faster & avoids path fiddling)
        epochs=RNN_EPOCHS,
    )
    rnn_run(rnn_cfg)

    # Load the trained RNN for the controller
    rnn_ckpt_dir = get_rnn_model_dir(ENV, RNN_NAME)
    rnn = MDN_LSTM.load_latest(ENV, RNN_NAME)
    print(f"[main] Loaded RNN from {rnn_ckpt_dir}")

    # ---------------------------------------------------------------------- #
    # 4. Controller – CMA-ES
    # ---------------------------------------------------------------------- #
    print("\n[main] ─── Controller stage ──────────────────────────────────")
    ctrl_cfg = ControllerCfg(
        env=ENV,
        name=CTRL_NAME,
        vae=vae,
        rnn=rnn,
        vae_name=VAE_NAME,  # so that worker processes can lazy-load
        rnn_name=RNN_NAME,
        device="cpu",
        maxiter=25,
        popsize=16,
        sigma0=0.1,
        rollouts=8,
        workers=16,
    )
    ctrl_run(ctrl_cfg)

    # ---------------------------------------------------------------------- #
    print("\n[main] Pipeline completed successfully.")
    print(f"  • Episodes folder : {data_path}")
    print(f"  • VAE dir         : {vae_ckpt_dir}")
    print(f"  • RNN dir         : {rnn_ckpt_dir}")
    print(f"  • CTRL dir        : {get_controller_model_dir(ENV, CTRL_NAME)}")


if __name__ == "__main__":
    main()
