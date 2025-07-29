"""
Parallel CMA-ES training script for the World-Models CarRacing controller.

Each worker process constructs its own VAE + RNN (loaded from checkpoints)
and a PolicyNet whose parameters are supplied by CMA-ES.  Fitness = negative
average episode reward (because CMA-ES minimises).

Run, for example:

    python -m src.worldmodels.training.controller \
        --vae-path ../../../trained_model/vae_latest.pt \
        --rnn-path ../../../trained_models/rnn_model.pt \
        --popsize 64 --maxiter 400
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import cma
import numpy as np
import torch
from gymnasium import Env

import src.worldmodels.envs.carracing as CarracingEnv
from src.worldmodels.models.controller import (
    PolicyNet,
    _params_to_vector,
    _vector_to_params,
    _load_models,
    _encode_frame,
)


def run() -> None:
    # ────────────────────────────────────────────────────────────────────────────────
    #  CLI
    # ────────────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Parallel CMA-ES on CarRacing-v3")
    parser.add_argument("--vae-path", default="../../../trained_model/vae_latest.pt",
                        help="Path to trained VAE (.pt)")
    parser.add_argument("--rnn-path", default="../../../trained_model/rnn_model.pt",
                        help="Path to trained RNN (.pt)")
    parser.add_argument("--popsize", type=int, default=16, help="CMA population λ")
    parser.add_argument("--sigma0", type=float, default=0.1, help="Initial CMA σ₀")
    parser.add_argument("--rollouts", type=int, default=4, help="Episodes per evaluation")
    parser.add_argument("--maxiter", type=int, default=0, help="Max CMA generations")
    parser.add_argument("--workers", type=int, default=16, help="# CPU workers")
    parser.add_argument("--render", action="store_true", default=True, help="Render a final greedy run")
    parser.add_argument("--save-model", default="../../../trained_model/controller_best.pt",
                        help="Path to save best controller (.pt)")
    parser.add_argument("--load-model", default="../../../trained_model/controller_best_891.pt",
                        help="Path to an existing controller to resume / evaluate")

    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────────────────────
    #  Determine controller input size (latent + hidden)
    # ────────────────────────────────────────────────────────────────────────────────
    from src.worldmodels.models.vae import VAE
    from src.worldmodels.models.rnn import MDN_LSTM

    vae_tmp = VAE.load_model(args.vae_path, device="cpu")
    rnn_tmp = MDN_LSTM.load_model(args.rnn_path, device="cpu")
    CTRL_IN_DIM = vae_tmp.latent + rnn_tmp.hidden_size
    del vae_tmp, rnn_tmp

    # ────────────────────────────────────────────────────────────────────────────────
    #  CMA-ES initialisation
    # ────────────────────────────────────────────────────────────────────────────────
    if args.load_model:
        print(f"[init] Loading controller from {args.load_model}")
        seed_policy = PolicyNet.load_model(args.load_model, device="cpu")
    else:
        seed_policy = PolicyNet(input_size=CTRL_IN_DIM, action_bounds=CarracingEnv.action_space)
    x0 = _params_to_vector(seed_policy)

    cma_opts: dict[str, int] = {}
    if args.popsize is not None:
        cma_opts["popsize"] = args.popsize

    es = cma.CMAEvolutionStrategy(x0, args.sigma0, cma_opts)

    best_reward, best_vec_global = -float("inf"), x0.copy()
    t0 = time.time()
    # ────────────────────────────────────────────────────────────────────────────────
    #  Parallel optimisation loop
    # ────────────────────────────────────────────────────────────────────────────────
    mp_context = mp.get_context("spawn")  # safer with PyTorch
    with mp_context.Pool(processes=args.workers) as pool:
        print(f"[CMA-ES] Using {args.workers} worker processes")
        for generation in range(args.maxiter):
            # 1) sample offspring
            offspring = es.ask()
            # 2) evaluate in parallel
            fitnesses = pool.starmap(
                PolicyNet.rollout,
                [
                    (
                        np.asarray(vec, dtype=np.float32),
                        args.rollouts,
                        args.vae_path,
                        args.rnn_path,
                        "cpu",
                        None,
                    )
                    for vec in offspring
                ],
            )
            # 3) update CMA-ES
            es.tell(offspring, fitnesses)

            if generation % 10 == 0:
                es.disp()
            rewards = [-f for f in fitnesses]  # convert back to reward
            gen_best, gen_mean = max(rewards), np.mean(rewards)
            if gen_best > best_reward:
                best_reward, best_vec_global = gen_best, offspring[int(np.argmax(rewards))]
            elapsed = time.time() - t0
            print(
                f"[Gen {generation:04d}/{args.maxiter}] "
                f"best {gen_best:7.1f} | mean {gen_mean:7.1f} | "
                f"global {best_reward:7.1f} | {elapsed / 60:5.1f} min elapsed",
                flush=True,
            )
            if es.stop():
                print("[CMA-ES] Stopping criteria met.")
                break

    # ------------------------------------------------------------------ determine the best policy ---
    print(f"\n[save] Writing best controller to {args.save_model}")
    best_policy = PolicyNet(input_size=CTRL_IN_DIM, action_bounds=CarracingEnv.action_space)
    _vector_to_params(best_policy, best_vec_global)

    if args.maxiter > 0:
        # ------------------------------------------------------------------ save ---
        print(f"\n[CMA-ES] Best average reward over {args.rollouts} rollouts: {best_reward:.1f}")
        best_policy.save_model(args.save_model)

    # ────────────────────────────────────────────────────────────────────────────────
    #  Retrieve & test best controller
    # ────────────────────────────────────────────────────────────────────────────────

    env_test: Env = CarracingEnv.make_env(render_mode="human" if args.render else None)()
    obs, _ = env_test.reset()
    vae_eval, rnn_eval = _load_models(args.vae_path, args.rnn_path, torch.device("cpu"))
    h_eval = (
        torch.zeros(rnn_eval.cfg.num_layers, 1, rnn_eval.hidden_size),
        torch.zeros(rnn_eval.cfg.num_layers, 1, rnn_eval.hidden_size),
    )

    tot_reward, done = 0.0, False
    while not done:
        z_eval = _encode_frame(obs, vae_eval, torch.device("cpu"))
        h_flat_eval = h_eval[0][-1, 0]
        ctrl_in_eval = torch.cat([z_eval, h_flat_eval], dim=0).detach().numpy()
        action = best_policy.act(ctrl_in_eval)

        obs, reward, term, trunc, _ = env_test.step(action)
        done = term or trunc
        tot_reward += reward

        za_eval = torch.cat([z_eval, torch.from_numpy(action)], dim=0).unsqueeze(0)
        _, _, _, h_eval = rnn_eval(za_eval, h_eval)

    env_test.close()
    print(f"[CMA-ES] Greedy reward in single episode: {tot_reward:.1f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # good practice on Windows
    run()  # ← only executed in the parent
