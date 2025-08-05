#!/usr/bin/env python
# scripts/play_env_keyboard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium.utils.play import play

from src.worldmodels.envs import EnvKind  # your Enum with .adapter_cls


@dataclass
class KeyboardPlayCfg:
    env: EnvKind = EnvKind.CARRACING  # or EnvKind.BIPEDAL_WALKER
    render_mode: str = "rgb_array"  # required by gym.utils.play
    fps: int = 60
    zoom: float = 1.0
    seed: Optional[int] = None


def run(cfg: KeyboardPlayCfg) -> None:
    adapter = cfg.env.adapter_cls(render_mode=cfg.render_mode, seed=cfg.seed)
    env = adapter.make_env()()

    try:
        keymap = adapter.keys_to_action()
        noop = adapter.noop_action(env.action_space).astype(np.float32)

        play(
            env,
            keys_to_action=keymap,
            fps=cfg.fps,
            zoom=cfg.zoom,
            noop=noop,
        )
    finally:
        env.close()


if __name__ == "__main__":
    run(KeyboardPlayCfg(env=EnvKind.PENDULUM, fps=60, zoom=1.0))
