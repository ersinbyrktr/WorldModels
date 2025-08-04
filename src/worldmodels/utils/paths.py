"""
Centralised, location-independent path helpers.

Every path is anchored to _this_ file’s directory, so running scripts from
any working directory will produce the same layout:

    <repo-root>/
        data/
        trained_<env>_model/
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.worldmodels.envs import EnvKind

# ────────────────────────────────────────────────────────────────────────────────
# Resolve the repository root *once* (3 levels up from this file)
# src/worldmodels/utils/paths.py  →  parents[0]=utils, [1]=worldmodels, [2]=src, [3]=<repo-root>
# ────────────────────────────────────────────────────────────────────────────────
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

# Base folders
_BASE_DATA_ROOT: Final[Path] = _REPO_ROOT / "data"


# ────────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────────
def _env_short(env: EnvKind) -> str:
    """Return a short, filesystem-friendly name for the environment enum."""
    return getattr(env, "short_name", env.name.lower())


def _trained_root(env: EnvKind) -> Path:
    """
    Root folder for all trained artefacts of a given environment, e.g.:

        <repo-root>/trained_bipedal_walker_model
        <repo-root>/trained_carracing_model
    """
    return _REPO_ROOT / f"trained_{_env_short(env)}_model"


# ────────────────────────────────────────────────────────────────────────────────
# Public helpers
# ────────────────────────────────────────────────────────────────────────────────
def get_data_path(env: EnvKind, name: str) -> Path:
    """
    Dataset / collection folder:
        <repo-root>/data/{env_short}_{name}
    """
    return _BASE_DATA_ROOT / f"{_env_short(env)}_{name}"


def get_vae_model_dir(env: EnvKind, name: str) -> Path:
    """
    VAE training run directory:
        <repo-root>/trained_{env_short}_model/{name}
    (latest checkpoint:  .../vae_latest.pt)
    """
    return _trained_root(env) / name


def get_rnn_model_dir(env: EnvKind, name: str) -> Path:
    """
    RNN (MDN-LSTM) run directory:
        <repo-root>/trained_{env_short}_model/{name}
    (latest checkpoint:  .../rnn_latest.pt)
    """
    return _trained_root(env) / name


def get_controller_model_dir(env: EnvKind, name: str) -> Path:
    """
    Controller optimisation run directory:
        <repo-root>/trained_{env_short}_model/{name}
    (latest checkpoint:  .../controller_latest.pt)
    """
    return _trained_root(env) / name


def get_viz_path(env: EnvKind, name: str) -> Path:
    """
    Visualisation output directory:
        <repo-root>/trained_{env_short}_model/{name}/viz
    """
    return get_vae_model_dir(env, name) / "viz"


__all__ = [
    "get_data_path",
    "get_vae_model_dir",
    "get_rnn_model_dir",
    "get_controller_model_dir",
    "get_viz_path",
]
