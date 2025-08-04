#!/usr/bin/env python
# scripts/collect_episodes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import trange

from src.worldmodels.envs import EnvKind
from src.worldmodels.envs.base import CollectorEnv
from src.worldmodels.utils.paths import get_data_path


# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class CollectCfg:
    env: EnvKind = EnvKind.CARRACING
    name: str = "default"  # output dir = ../../../data/{env}_{name}
    episodes: int = 1000
    workers: int = 12
    render_mode: str = "rgb_array"
    seed: Optional[int] = None


# ────────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────────

def run(cfg: CollectCfg) -> None:
    out = get_data_path(cfg.env, cfg.name)
    out.mkdir(parents=True, exist_ok=True)

    adapter: CollectorEnv = cfg.env.adapter_cls(render_mode=cfg.render_mode, seed=cfg.seed)
    vec = adapter.vectorize(cfg.workers)
    vec.reset()
    adapter.on_run_start()

    episode_buffers = [[] for _ in range(cfg.workers)]
    action_buffers = [[] for _ in range(cfg.workers)]
    reward_buffers = [[] for _ in range(cfg.workers)]
    episode_count = 0

    with trange(cfg.episodes, desc="Episodes") as pbar:
        while episode_count < cfg.episodes:
            actions = adapter.sample_batch_actions(vec.action_space, cfg.workers)
            obs, rewards, term, trunc, _ = vec.step(actions)
            done = np.logical_or(term, trunc)

            for i, ob in enumerate(obs):
                episode_buffers[i].append(adapter.preprocess_obs(ob))
                action_buffers[i].append(actions[i])
                reward_buffers[i].append(float(np.asarray(rewards)[i]))

                if done[i]:
                    np.save(out / f"episode_{episode_count:05d}_obs.npy",
                            np.asarray(episode_buffers[i], dtype=np.uint8))
                    np.save(out / f"episode_{episode_count:05d}_actions.npy",
                            np.asarray(action_buffers[i], dtype=np.float32))
                    np.save(out / f"episode_{episode_count:05d}_rewards.npy",
                            np.asarray(reward_buffers[i], dtype=np.float32))

                    episode_buffers[i].clear()
                    action_buffers[i].clear()
                    reward_buffers[i].clear()

                    episode_count += 1
                    pbar.update(1)

            if episode_count >= cfg.episodes:
                break

    vec.close()


if __name__ == "__main__":
    cfg = CollectCfg(
        env=EnvKind.CARRACING,  # or EnvKind.BIPEDAL_WALKER
        name="baseline",
        episodes=100,
        workers=12,
        render_mode="rgb_array",
        seed=None,
    )
    run(cfg)
