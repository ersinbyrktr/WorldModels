"""
Minimal Gym environment that *dreams* Car Racing inside an MDN-RNN latent world.

Observation  o_t  =  concat( z_t , h_t , c_t )           shape = 32 + 256 + 256 = 544
Action       a_t  =  np.array([steer, gas, brake])      range = [-1,1]×[0,1]×[0,1]
Reward       r_t  =  +1  (survival time-step)            — simple but works for CMA-ES
Done         done =  t >= max_steps                      (optional: model-based flag)

You can **tune the temperature τ** at construction time to make
dreams more or less stochastic.

Author: World Models “learning-in-the-dream” refactor
--------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch

from src.worldmodels.models.rnn import MDN_LSTM


class DreamCarRacingEnv(gym.Env):
    metadata = {"render.modes": []}  # no visualisation yet

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
            self,
            model: MDN_LSTM,
            tau: float = 1.0,
            max_steps: int = 1_000,
            deterministic: bool = False,
            device: Optional[torch.device] = None,
            use_done_head: bool = False,
    ):
        """
        Args
        ----
        model          trained MDN_LSTM (latent→latent)
        tau            temperature applied at sampling (σ_scaled = σ * τ)
        max_steps      hard horizon (Car Racing default = 1 000 frames)
        deterministic  if True, use mixture mean instead of sampling
        device         cuda / cpu (defaults to model’s own device)
        use_done_head  if the model has a Bernoulli done head, set True
                       and make  done = p_done > .5
        """
        super().__init__()
        self.model = model.eval()
        self.tau = tau
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.use_done_head = use_done_head

        self.device = (
            device if device is not None else next(model.parameters()).device
        )

        # ---- Gym spaces -------------------------------------------------- #
        # Actions = { steer ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1] }
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = model.cfg.output_size + 2 * model.hidden_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self._z: torch.Tensor | None = None  # (1,32)
        self._h: Tuple[torch.Tensor, torch.Tensor] | None = None  # (2,1,256)
        self._t: int = 0

    # --------------------------------------------------------------------- #
    #  Gym API
    # --------------------------------------------------------------------- #
    def reset(self, z0: Optional[np.ndarray] = None, seed: Optional[int] = None):
        super().reset(seed=seed)
        self._t = 0

        # latent start state — zeros or user-supplied
        if z0 is None:
            z0 = np.zeros(self.model.cfg.output_size, dtype=np.float32)
        z0 = torch.as_tensor(z0, dtype=torch.float32, device=self.device).unsqueeze(0)

        # hidden & cell zeros
        h0 = torch.zeros(
            self.model.cfg.num_layers, 1, self.model.hidden_size, device=self.device
        )
        self._z, self._h = z0, (h0.clone(), h0.clone())

        return self._get_obs(), {}  # Gymnasium-style (obs, info)

    def step(self, action: np.ndarray):
        assert self._z is not None, "Call reset() first"
        self._t += 1

        # 1. Forward through MDN-RNN
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_in = torch.cat([self._z, act], dim=-1)  # (1, Cin)
        wl, mu, ls, self._h = self.model(x_in, self._h)

        # 2. Sample / mean latent
        z_next = self.model.predict(
            wl, mu, ls,
            deterministic=self.deterministic,
            tau=self.tau,
        )
        self._z = z_next

        # 3. Decide termination
        done = self._t >= self.max_steps
        if self.use_done_head:
            # Model must return Bernoulli parameter in wl; assume last logit is done
            p_done = torch.sigmoid(wl[..., -1]).item()
            done = done or (p_done > 0.5)

        # 4. Reward = survival time-step (+1)
        reward = 1.0
        obs = self._get_obs()
        info = {"t": self._t}

        return obs, reward, done, False, info  # Gymnasium return-signature

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _get_obs(self):
        """Concatenate latent z, hidden h and cell c → np.float32."""
        h_out, c_out = self._h
        vec = torch.cat([self._z.squeeze(0), h_out[-1, 0], c_out[-1, 0]], dim=-1)
        return vec.detach().cpu().numpy().astype(np.float32)

    # --------------------------------------------------------------------- #
    #  No rendering (yet)
    # --------------------------------------------------------------------- #
    def render(self, mode="human"):
        raise NotImplementedError("Dream environment has no renderer.")

    def close(self):
        pass
