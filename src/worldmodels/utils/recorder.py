import gymnasium as gym
from pathlib import Path

import torch

from src.worldmodels.models.controller import PolicyNet, _load_models, _encode_frame


def record_single_episode(
    policy: PolicyNet,
    vae_path: str,
    rnn_path: str,
    out_dir: str = "videos",
    name_prefix: str = "controller_play",
    device: str = "cpu",
):
    """
    Runs ONE greedy episode and saves an MP4 in ``out_dir`` (created if missing).

    The file name will look like  controller_play-episode-0000.mp4
    """
    # --------------------------------------------------------------------- #
    # 1) Build & wrap the env so that it writes video frames automatically.
    # --------------------------------------------------------------------- #
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",          # required for RecordVideo
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=out_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda ep: True,  # record the very first episode only
    )

    # --------------------------------------------------------------------- #
    # 2) Prepare World-Models components exactly as you already do.
    # --------------------------------------------------------------------- #
    dev  = torch.device(device)
    vae, rnn = _load_models(vae_path, rnn_path, dev)

    h = (
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
        torch.zeros(rnn.cfg.num_layers, 1, rnn.hidden_size, device=dev),
    )

    obs, _ = env.reset(seed=None)
    done, total_reward = False, 0.0

    # --------------------------------------------------------------------- #
    # 3) Greedy rollout (unchanged, apart from calling env.render()         #
    #    implicitly through the wrapper).                                   #
    # --------------------------------------------------------------------- #
    while not done:
        z = _encode_frame(obs, vae, dev)                # (latent,)
        ctrl_in = torch.cat([z, h[0][-1, 0]], dim=0)     # (latent+hidden,)
        action = policy.act(ctrl_in.cpu().detach().numpy())

        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += r

        za = torch.cat([z, torch.from_numpy(action).to(dev)], dim=0).unsqueeze(0)
        _, _, _, h = rnn(za, h)

    env.close()
    print(f"[record] saved episode to {Path(out_dir).resolve()}")
    print(f"[record] episode reward = {total_reward:.1f}")

if __name__ == "__main__":
    vae_path = "../../../trained_model/vae_latest.pt"
    rnn_path = "../../../trained_model/rnn_model.pt"
    best_policy = PolicyNet.load_model("../../../trained_model/controller_best_891.pt")
    record_single_episode(best_policy, vae_path, rnn_path)