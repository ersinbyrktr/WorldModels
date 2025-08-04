"""
Dream-world Controller for World Models – Car Racing
----------------------------------------------------
* Observation: [ z_t (32) , h_t (256) , c_t (256) ]  →  544-D
* Action     : [ steer ∈ (-1,1) , gas ∈ (0,1) , brake ∈ (0,1) ]
The network is a single linear layer with tanh / sigmoid squashes,
exactly like the original Linear controller – just a bigger input.

Designed for CMA-ES optimisation (provides param ↔ vector helpers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from src.worldmodels.envs.dream_carracing import DreamCarRacingEnv
from src.worldmodels.models.rnn import MDN_LSTM


# ----------------------------------------------------------------------------- #
#  Helpers to flatten / unflatten parameters
# ----------------------------------------------------------------------------- #
def params_to_vector(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def vector_to_params(net: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    for p in net.parameters():
        n = p.numel()
        block = vec[idx: idx + n].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        idx += n


# ----------------------------------------------------------------------------- #
#  Config dataclass  (pickle-safe)
# ----------------------------------------------------------------------------- #
@dataclass
class ControllerCfg:
    input_size: int
    action_bounds: List[Tuple[float, float]]  # e.g. [(-1,1),(0,1),(0,1)]


# ----------------------------------------------------------------------------- #
#  Linear dream-policy  (tiny & CMA-ES-friendly)
# ----------------------------------------------------------------------------- #
class DreamPolicyLinear(nn.Module):
    """
    Minimal one-layer controller:

        x = concat[ z_t , h_t , c_t ]  (544-D)
        a = fc(x)  →  steer,gas,brake  (squashed)
    """

    def __init__(
            self,
            input_size: int = 1088,  # 32 + 256 + 256
            action_bounds: List[Tuple[float, float]] | None = None,
    ):
        super().__init__()
        if action_bounds is None:
            action_bounds = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # Car Racing default

        self.action_bounds = action_bounds
        self.fc = nn.Linear(input_size, len(action_bounds))

        tanh_mask = [lo == -1.0 and hi == 1.0 for lo, hi in action_bounds]
        sigm_mask = [lo == 0.0 and hi == 1.0 for lo, hi in action_bounds]
        self.register_buffer("_tanh_mask", torch.tensor(tanh_mask, dtype=torch.bool))
        self.register_buffer("_sigm_mask", torch.tensor(sigm_mask, dtype=torch.bool))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.fc(x)
        out = torch.empty_like(raw)
        if self._tanh_mask.any():
            out[:, self._tanh_mask] = torch.tanh(raw[:, self._tanh_mask])
        if self._sigm_mask.any():
            out[:, self._sigm_mask] = torch.sigmoid(raw[:, self._sigm_mask])
        return out

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            a = self.forward(x).squeeze(0).cpu().numpy()
        return a

    # ------------------------------------------------------------------ #
    #  CMA-ES rollout in dream world
    # ------------------------------------------------------------------ #
    @classmethod
    def rollout(
            cls,
            policy_vec: np.ndarray,
            rnn_path: str,
            rollouts: int = 5,
            tau: float = 1.15,
            device: str | torch.device = "cpu",
            deterministic_model: bool = False,
            env_seed: Optional[int] = None,
    ) -> float:
        """
        Evaluate a controller **inside the DreamCarRacingEnv**.

        Returns negative average reward (CMA-ES minimises).
        """
        dev = torch.device(device)
        rnn: MDN_LSTM = MDN_LSTM.load_model(rnn_path, device=dev).eval()

        # build policy & load params
        inp = rnn.cfg.output_size + 2 * rnn.hidden_size
        policy = cls(input_size=inp).to(dev)
        vector_to_params(policy, policy_vec)
        policy.eval()

        rewards = []
        for ep in range(rollouts):
            env = DreamCarRacingEnv(
                model=rnn,
                tau=tau,
                deterministic=deterministic_model,
            )
            if env_seed is not None:
                obs, _ = env.reset(seed=env_seed + ep)
            else:
                obs, _ = env.reset()

            done, ret = False, 0.0
            while not done:
                a = policy.act(obs)
                obs, r, done, _, _ = env.step(a)
                ret += r
            env.close()
            rewards.append(ret)

        return -float(np.mean(rewards))  # CMA-ES expects *minimisation*

    # ------------------------------------------------------------------ #
    #  (De)serialisation helpers
    # ------------------------------------------------------------------ #
    def save_model(self, path: str | os.PathLike) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cfg = ControllerCfg(
            input_size=self.fc.in_features,
            action_bounds=[(float(l), float(h)) for l, h in self.action_bounds],
        )
        torch.save(
            {"state_dict": self.state_dict(), "config": asdict(cfg)},
            path,
        )

    @classmethod
    def load_model(cls, path: str | os.PathLike, device: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt["config"]
        model = cls(**cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ----------------------------------------------------------------------------- #
#  Convenience wrapper for CMA-ES scripts
# ----------------------------------------------------------------------------- #
def make_seed_policy() -> DreamPolicyLinear:
    """
    Returns an **un-trained** dream-policy with correct shapes.
    Useful for generating CMA-ES x0.
    """
    return DreamPolicyLinear(input_size=1088)
