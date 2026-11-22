"""Resumable checkpoints for trainable parameters and optimizer state."""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _atomic_save(path: Path, arrays: dict) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    mx.save_safetensors(str(tmp), arrays)
    os.replace(tmp, path)


def save(directory: Path, step: int, params: dict, optimizer=None, extra: dict | None = None) -> None:
    """Save a flat or nested parameter tree plus optimizer state, then the step marker."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(params))
    _atomic_save(directory / "params.safetensors", flat)
    if optimizer is not None:
        state = {k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)}
        _atomic_save(directory / "optimizer.safetensors", state)
    marker = dict(step=int(step), **(extra or {}))
    tmp = directory / f".state.{os.getpid()}.json"
    tmp.write_text(json.dumps(marker))
    os.replace(tmp, directory / "state.json")


def _assign(tree, path, value):
    for i, part in enumerate(path[:-1]):
        nxt = tree[int(part)] if isinstance(tree, list) else tree.get(part)
        if nxt is None:
            tree[part] = nxt = [] if path[i + 1].isdigit() else {}
        tree = nxt
    last = path[-1]
    if isinstance(tree, list):
        tree[int(last)] = value
    else:
        tree[last] = value


def load(directory: Path, optimizer=None):
    """Return (step, params tree, extra) or (0, None, {}) when no checkpoint exists."""
    directory = Path(directory)
    if not (directory / "state.json").exists():
        return 0, None, {}
    marker = json.loads((directory / "state.json").read_text())
    params = tree_unflatten(list(mx.load(str(directory / "params.safetensors")).items()))
    if optimizer is not None and (directory / "optimizer.safetensors").exists():
        # Call optimizer.init(trainable params) first. Arrays are written into the existing
        # state tree in place, so empty subtrees for frozen modules keep their structure.
        for key, value in mx.load(str(directory / "optimizer.safetensors")).items():
            _assign(optimizer.state, key.split("."), value)
    step = marker.pop("step")
    return step, params, marker
