import json
from dataclasses import asdict, replace, is_dataclass
from pathlib import Path
from typing import Dict, Any

from src.worldmodels.data.collect_episodes import CollectCfg
from src.worldmodels.envs import EnvKind
from src.worldmodels.models.controller import ControllerCfg
from src.worldmodels.models.rnn import RNNRunCfg
from src.worldmodels.models.vae import VaeRunCfg
from src.worldmodels.utils.paths import get_data_path, get_vae_model_dir, get_rnn_model_dir, get_controller_model_dir


# ────────────────────────────────────────────────────────────────────────────
# JSON helpers (make dataclasses/Enums/Paths serialisable)
# ────────────────────────────────────────────────────────────────────────────
def _env_to_dict(env: EnvKind) -> Dict[str, Any]:
    return {
        "name": env.name,
        "short": getattr(env, "short_name", env.name.lower()),
    }


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, paths, tuples to JSON-able forms."""
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, EnvKind):
        return _env_to_dict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # primitives pass through
    return obj


def _sanitise_cfg_dict(d: Dict[str, Any], *, drop_keys: tuple[str, ...] = ()) -> Dict[str, Any]:
    """Drop non-serialisable or redundant keys and convert to JSON-friendly types."""
    out = {k: v for k, v in d.items() if k not in drop_keys}
    # Normalise env if present
    if "env" in out and isinstance(out["env"], EnvKind):
        out["env"] = _env_to_dict(out["env"])
    return _to_jsonable(out)


def _write_pipeline_config(
        env: EnvKind,
        collection_name: str,
        *,
        collect_cfg: CollectCfg,
        vae_cfg: VaeRunCfg,
        rnn_cfg: RNNRunCfg,
        ctrl_cfg: ControllerCfg,
) -> Path:
    """
    Write a single JSON snapshot with all 4 configs to the controller run folder.
    """
    ctrl_dir = get_controller_model_dir(env, ctrl_cfg.name)
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctrl_dir / "pipeline_config.json"

    # Make JSON-friendly copies (drop live model refs etc.)
    collect_dict = _sanitise_cfg_dict(asdict(collect_cfg))
    vae_dict = _sanitise_cfg_dict(asdict(vae_cfg))
    rnn_dict = _sanitise_cfg_dict(asdict(replace(rnn_cfg, vae=None)), drop_keys=("vae",))
    ctrl_dict = _sanitise_cfg_dict(asdict(replace(ctrl_cfg, vae=None, rnn=None)), drop_keys=("vae", "rnn"))

    payload = {
        "env": _env_to_dict(env),
        "collection": collection_name,
        "paths": {
            "episodes_dir": str(get_data_path(env, collection_name)),
            "vae_dir": str(get_vae_model_dir(env, vae_cfg.name)),
            "rnn_dir": str(get_rnn_model_dir(env, rnn_cfg.name)),
            "controller_dir": str(ctrl_dir),
        },
        "configs": {
            "collector": collect_dict,
            "vae": vae_dict,
            "rnn": rnn_dict,
            "controller": ctrl_dict,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[main] Wrote pipeline config → {out_path}")
    return out_path
