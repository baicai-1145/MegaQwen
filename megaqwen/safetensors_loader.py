from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import safe_open


def _load_index(model_dir: Path) -> dict[str, str] | None:
    """Return key->filename map from `model.safetensors.index.json` if present."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return {str(k): str(v) for k, v in weight_map.items()}


def load_tensors(
    model_dir: str | Path,
    keys: Iterable[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load selected tensors from a (possibly sharded) safetensors checkpoint.

    Supports:
    - `model.safetensors`
    - `model.safetensors.index.json` + shards
    """
    model_dir = Path(model_dir)
    keys = list(dict.fromkeys(keys))  # stable-dedup
    if not keys:
        return {}

    index = _load_index(model_dir)
    if index is None:
        # Single-file checkpoint.
        ckpt = model_dir / "model.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(ckpt), framework="pt", device=str(device)) as f:
            available = set(f.keys())
            missing = [k for k in keys if k not in available]
            if missing:
                raise KeyError(f"Missing tensors in {ckpt.name}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            for k in keys:
                out[k] = f.get_tensor(k)
        return out

    # Sharded checkpoint: group by shard filename.
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    missing_in_index = []
    for k in keys:
        fname = index.get(k)
        if fname is None:
            missing_in_index.append(k)
        else:
            keys_by_file[fname].append(k)
    if missing_in_index:
        raise KeyError(
            f"Missing tensors in index: {missing_in_index[:5]}{'...' if len(missing_in_index) > 5 else ''}"
        )

    out: dict[str, torch.Tensor] = {}
    for fname, file_keys in keys_by_file.items():
        ckpt = model_dir / fname
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing shard: {ckpt}")
        with safe_open(str(ckpt), framework="pt", device=str(device)) as f:
            available = set(f.keys())
            missing = [k for k in file_keys if k not in available]
            if missing:
                raise KeyError(f"Missing tensors in shard {fname}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            for k in file_keys:
                out[k] = f.get_tensor(k)
    return out

