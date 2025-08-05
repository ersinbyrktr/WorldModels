"""
World-Models Controller (paths derived from EnvKind; VAE/RNN can be passed directly).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cma
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env

from src.worldmodels.envs import EnvKind  # enum with .adapter_cls and .short_name
from src.worldmodels.models.rnn import MDN_LSTM
from src.worldmodels.models.vae import VAE
from src.worldmodels.utils.paths import get_controller_model_dir
from src.worldmodels.utils.utils import _encode_frame, _obs_to_frame


# ────────────────────────────────────────────────────────────────────────────────
# Config (derive controller paths from EnvKind + controller `name`)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerCfg:
    # Provide VAE/RNN instances directly (recommended).
    # main process (evaluation). Worker processes will *not* receive them.
    vae: Optional[VAE]
    rnn: Optional[MDN_LSTM]

    # Checkpoint folders:
    #   CTRL  → ../../../trained_{env}_model/{name}/controller_latest.pt
    env: EnvKind = EnvKind.CARRACING
    name: str = "ctrl_baseline"

    # If you run CMA-ES with multiple processes, workers will lazily load models
    # using the standard `load_latest` helpers (no need to pass paths).
    vae_name: str = "vae_baseline"
    rnn_name: str = "rnn_baseline"

    # runtime
    device: str = "cpu"
    env_seed: Optional[int] = None

    # CMA-ES
    popsize: int = 16
    sigma0: float = 0.3
    rollouts: int = 8
    maxiter: int = 25
    workers: int = 16

    action_bounds: Optional[List[Tuple[float, float]]] = None
    input_size: Optional[int] = None


# ────────────────────────────────────────────────────────────────────────────────
# Policy
# ────────────────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, cfg: ControllerCfg):
        super().__init__()
        input_dim = _computed_input_size(cfg)  # ← new helper (below)
        if cfg.action_bounds is None:
            raise ValueError("cfg.action_bounds must be set.")
        self.cfg = cfg
        # self.fc = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, len(cfg.action_bounds)))
        self.fc = nn.Linear(input_dim, len(cfg.action_bounds))  # input_dim = latent_dim + hidden_dim
        tanh_mask = [lo == -1 and hi == 1 for lo, hi in cfg.action_bounds]
        sigmoid_mask = [lo == 0 and hi == 1 for lo, hi in cfg.action_bounds]
        self.register_buffer("_tanh_mask", torch.tensor(tanh_mask, dtype=torch.bool))
        self.register_buffer("_sigmoid_mask", torch.tensor(sigmoid_mask, dtype=torch.bool))
        if not (self._tanh_mask.any() or self._sigmoid_mask.any()):
            raise ValueError("No action dims matched (-1,1) or (0,1).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.fc(x)
        a = torch.empty_like(raw)
        if self._tanh_mask.any():
            a[:, self._tanh_mask] = torch.tanh(raw[:, self._tanh_mask])
        if self._sigmoid_mask.any():
            a[:, self._sigmoid_mask] = torch.sigmoid(raw[:, self._sigmoid_mask])
        return a

    def act(self, controller_input: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(controller_input).float().unsqueeze(0)
        with torch.no_grad():
            return self.forward(x).squeeze(0).cpu().numpy()

    # ------------- rollout used by CMA-ES (returns negative reward) -------------
    @classmethod
    def rollout(cls, policy_params: np.ndarray, cfg: ControllerCfg) -> float:
        # NOTE: This method runs in *worker processes*. We *do not* rely on
        # cfg.vae/cfg.rnn here (they are not pickled across processes).
        dev = torch.device(cfg.device)
        vae, rnn = _load_models_for_worker(cfg, dev)
        policy = cls(cfg)
        _vector_to_params(policy, policy_params)
        policy.eval()

        rewards: list[float] = []
        env_thunk = _env_thunk(cfg)
        for _ in range(cfg.rollouts):
            env = env_thunk()
            if cfg.env_seed is not None:
                env.reset(seed=cfg.env_seed)
            rewards.append(_episode_return(env, policy, vae, rnn, dev))
            env.close()
        return -float(np.mean(rewards))

    # ------------- persistence ---------------------------------------------------
    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "input_size": int(self.fc.in_features),
                    "action_bounds": [(float(l), float(h)) for (l, h) in self.cfg.action_bounds or []],
                },
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: str | torch.device = "cpu") -> "PolicyNet":
        """
        Strict loader: expects a checkpoint saved by `save_model` with:
          {
            "model_state_dict": ...,
            "config": {"input_size": int, "action_bounds": List[Tuple[float, float]]}
          }
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt["model_state_dict"]
        cfg_dict = ckpt["config"]
        cfg = ControllerCfg(
            input_size=int(cfg_dict["input_size"]),
            action_bounds=[(float(l), float(h)) for (l, h) in cfg_dict["action_bounds"]],
        )
        model = cls(cfg).to(device)
        model.load_state_dict(state)
        return model

    @classmethod
    def load_latest(cls, env: EnvKind, name: str, device: Optional[torch.device] = None) -> "PolicyNet":
        """
        Load the latest controller checkpoint for a given (env, name) run:
            path = ../../../trained_{env_short}_model/{name}/controller_latest.pt
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = get_controller_model_dir(env, name) / "controller_latest.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Controller checkpoint not found: {ckpt}")
        return cls.load_model(str(ckpt), device=device)


def _action_dim(cfg: ControllerCfg) -> int:
    adapter = cfg.env.adapter_cls(render_mode="rgb_array", seed=cfg.env_seed)
    return len(adapter.action_bounds())


_HEADER_CACHE: dict[tuple[str, str], int] = {}  # env / run-names → input dim


def _computed_input_size(cfg: ControllerCfg) -> int:
    """
    latent_dim + rnn.hidden_size  (loads headers lazily if objects absent)
    """
    if cfg.vae is not None and cfg.rnn is not None:
        return cfg.vae.latent + cfg.rnn.hidden_size

    key = (cfg.env.name, f"{cfg.vae_name}|{cfg.rnn_name}")
    if key not in _HEADER_CACHE:
        # lightweight: read only checkpoint headers on CPU
        vae = VAE.load_latest(cfg.env, cfg.vae_name, device="cpu")
        rnn = MDN_LSTM.load_latest(cfg.env, cfg.rnn_name, device="cpu")
        _HEADER_CACHE[key] = vae.latent + rnn.hidden_size
        del vae, rnn
    return _HEADER_CACHE[key]


def _validate_models(cfg: ControllerCfg, vae: VAE, rnn: MDN_LSTM) -> None:
    act_dim = _action_dim(cfg)
    exp_in = vae.latent + act_dim
    # rnn.cfg.<...> comes from the saved ModelCfg
    got_in = getattr(rnn.cfg, "input_size", None)
    got_out = getattr(rnn.cfg, "output_size", None)

    if got_out is not None and got_out != vae.latent:
        raise RuntimeError(
            "VAE/RNN latent mismatch:\n"
            f"  VAE.latent         = {vae.latent}\n"
            f"  RNN.cfg.output_size= {got_out}\n"
            "The RNN must be trained with the same VAE latent size."
        )
    if got_in is not None and got_in != exp_in:
        raise RuntimeError(
            "RNN input-size mismatch:\n"
            f"  expected (vae.latent + action_dim) = {vae.latent} + {act_dim} = {exp_in}\n"
            f"  RNN.cfg.input_size                 = {got_in}\n"
            "This usually means the RNN was trained for a different env or VAE run.\n"
            f"Check env={cfg.env.name}, vae_name={cfg.vae_name}, rnn_name={cfg.rnn_name}."
        )


# ────────────────────────────────────────────────────────────────────────────────
# Per-process model cache for workers
# ────────────────────────────────────────────────────────────────────────────────

_WORKER_CACHE: dict[tuple[str, str, str], tuple[VAE, MDN_LSTM]] = {}


def _load_models_for_worker(cfg: ControllerCfg, device: torch.device) -> tuple[VAE, MDN_LSTM]:
    key = (cfg.env.name, cfg.vae_name, cfg.rnn_name + f"@{device.type}")
    if key not in _WORKER_CACHE:
        vae = VAE.load_latest(cfg.env, cfg.vae_name, device=device).eval()
        rnn = MDN_LSTM.load_latest(cfg.env, cfg.rnn_name, device=device).eval()
        _validate_models(cfg, vae, rnn)  # ← add this
        _WORKER_CACHE[key] = (vae, rnn)
    return _WORKER_CACHE[key]


# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────

def _env_thunk(cfg: ControllerCfg):
    adapter = cfg.env.adapter_cls(render_mode="rgb_array", seed=cfg.env_seed)
    return adapter.make_env()


def _get_models_main(cfg: ControllerCfg, device: torch.device) -> tuple[VAE, MDN_LSTM]:
    if cfg.vae is not None and cfg.rnn is not None:
        vae = cfg.vae.to(device).eval()
        rnn = cfg.rnn.to(device).eval()
    else:
        vae, rnn = _load_models_for_worker(cfg, device)
    _validate_models(cfg, vae, rnn)  # ← add this
    return vae, rnn


def _fill_runtime(cfg: ControllerCfg) -> ControllerCfg:
    # 1) Action bounds from adapter if missing (uses CollectorEnv.action_bounds()).
    if cfg.action_bounds is None:
        adapter = cfg.env.adapter_cls(render_mode="rgb_array", seed=cfg.env_seed)
        cfg.action_bounds = [
            (float(lo), float(hi)) for (lo, hi) in adapter.action_bounds()
        ]
    return cfg


def _cfg_for_workers(cfg: ControllerCfg) -> ControllerCfg:
    """Create a light copy of cfg to send to worker processes (drop model refs)."""
    return replace(cfg, vae=None, rnn=None)


def _params_to_vector(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def _vector_to_params(net: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    for p in net.parameters():
        numel = p.numel()
        block = vec[idx: idx + numel].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        idx += numel


def _vector_to_policy(vec: np.ndarray, cfg: ControllerCfg) -> PolicyNet:
    p = PolicyNet(cfg)
    _vector_to_params(p, vec)
    return p


def _episode_return(env: Env, policy: PolicyNet, vae: VAE, rnn: MDN_LSTM, dev: torch.device) -> float:
    obs, _ = env.reset()
    h = (
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
    )
    done, ret = False, 0.0
    while not done:
        frame = _obs_to_frame(env, obs)
        with torch.no_grad():
            z_t = _encode_frame(frame, vae, dev)
        h_flat = h[0][-1, 0]
        ctrl_in = torch.cat([z_t, h_flat], dim=0).detach().cpu().numpy()
        a = policy.act(ctrl_in)
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        ret += r
        za_t = torch.cat([z_t, torch.from_numpy(a).to(dev)], dim=0).unsqueeze(0)
        _, _, _, h = rnn(za_t, h)
    return float(ret)


def _evaluate_greedy(policy: PolicyNet, cfg: ControllerCfg) -> float:
    dev = torch.device(cfg.device)
    vae, rnn = _get_models_main(cfg, dev)
    env = _env_thunk(cfg)()
    try:
        total = _episode_return(env, policy, vae, rnn, dev)
    finally:
        env.close()
    return total


def _cma_optimize(x0: np.ndarray, cfg: ControllerCfg) -> Tuple[np.ndarray, float]:
    es = cma.CMAEvolutionStrategy(x0, cfg.sigma0, {"popsize": int(cfg.popsize)})
    best_reward, best_vec = -float("inf"), x0.copy()
    t0 = time.time()

    # Send a light cfg (no model objects) to workers
    worker_cfg = _cfg_for_workers(cfg)
    mp_context = mp.get_context("spawn")
    with mp_context.Pool(processes=cfg.workers) as pool:
        print(f"[CMA-ES] Using {cfg.workers} worker processes")
        for gen in range(cfg.maxiter):
            offspring = es.ask()
            fitnesses = pool.starmap(PolicyNet.rollout, [(np.asarray(v, np.float32), worker_cfg) for v in offspring])
            es.tell(offspring, fitnesses)
            if gen % 10 == 0:
                es.disp()

            rewards = [-f for f in fitnesses]
            gen_best = float(np.max(rewards))
            if gen_best > best_reward:
                best_reward = gen_best
                best_vec = offspring[int(np.argmax(rewards))]
            gen_mean = float(np.mean(rewards))
            elapsed = (time.time() - t0) / 60
            print(
                f"[Gen {gen:04d}/{cfg.maxiter}] best {gen_best:7.1f} | mean {gen_mean:7.1f} | "
                f"global {best_reward:7.1f} | {elapsed:5.1f} min",
                flush=True,
            )
            if es.stop():
                print("[CMA-ES] Stopping criteria met.")
                break
    return best_vec, best_reward


# ────────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────────

def run(cfg: Optional[ControllerCfg]) -> None:
    cfg = _fill_runtime(cfg)

    # If a previous controller exists, start from it
    ctrl_dir = get_controller_model_dir(cfg.env, cfg.name)
    ctrl_ckpt = ctrl_dir / "controller_latest.pt"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    if ctrl_ckpt.is_file():
        print(f"[init] Loading controller seed from {ctrl_ckpt}")
        seed_policy = PolicyNet.load_model(str(ctrl_ckpt), device="cpu")
        seed_policy.cfg = cfg
    else:
        seed_policy = PolicyNet(cfg)

    x0 = _params_to_vector(seed_policy)
    best_vec, best_reward = _cma_optimize(x0, cfg)

    print(f"\n[save] Writing best controller to {ctrl_ckpt}")
    best_policy = _vector_to_policy(best_vec, cfg)
    if cfg.maxiter > 0:
        print(f"\n[CMA-ES] Best avg reward over {cfg.rollouts} rollouts: {best_reward:.1f}")
        best_policy.save_model(str(ctrl_ckpt))

    tot = _evaluate_greedy(best_policy, cfg)
    print(f"[CMA-ES] Greedy reward in single episode: {tot:.1f}")


if __name__ == "__main__":
    # Load VAE/RNN *first* and pass them directly into the controller config.
    from src.worldmodels.models.vae import VAE
    from src.worldmodels.models.rnn import MDN_LSTM

    mp.set_start_method("spawn", force=True)

    env = EnvKind.PENDULUM
    vae_name = "baseline"
    rnn_name = "baseline"
    ctrl_name = "baseline"

    vae = VAE.load_latest(env=env, name=vae_name)  # ../../../trained_{env}_model/{vae_name}/vae_latest.pt
    rnn = MDN_LSTM.load_latest(env=env, name=rnn_name)  # ../../../trained_{env}_model/{rnn_name}/rnn_latest.pt

    cfg = ControllerCfg(
        env=env,
        name=ctrl_name,
        vae=vae,  # used in main process
        rnn=rnn,  # used in main process
        vae_name=vae_name,  # used by worker processes to lazy-load
        rnn_name=rnn_name,  # used by worker processes to lazy-load
        device="cpu",
        maxiter=20,
        popsize=16,
        sigma0=0.3,
        rollouts=10,
        workers=8,
    )
    run(cfg)
