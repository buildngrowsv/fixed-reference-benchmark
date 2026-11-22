"""Vendored copy of net-atlas/atlas/bundle.py. Write a net-atlas bundle: one JSON manifest plus one gzip-compressed binary blob.

Every experiment that wants the shared viewer saves the same kinds of arrays:

* ``states/<run>``  residual (or layer) states, shape [layers, items, width]
* ``detail/<run>/<part>``  internals of one zoomed block, shape [items, width]
* weights for a few selected units, metrics, and per-item predictions

Only NumPy is required, so the writer runs inside any experiment environment.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

DTYPES = {"f16": np.float16, "f32": np.float32, "i32": np.int32, "u8": np.uint8, "i8": np.int8}


class Bundle:
    def __init__(self, title: str, description: str = ""):
        self.meta: dict = {"format": "net-atlas/1", "title": title, "description": description,
                           "arrays": {}}
        self._chunks: list[bytes] = []
        self._offset = 0

    def __setitem__(self, key, value):
        self.meta[key] = value

    def __getitem__(self, key):
        return self.meta[key]

    def array(self, name: str, values, dtype: str = "f16") -> None:
        """Store an array. ``q8`` stores int8 with one float32 scale per last-axis column."""
        a = np.ascontiguousarray(np.asarray(values))
        spec = {"shape": list(a.shape), "dtype": dtype}
        if dtype == "q8":
            flat = a.reshape(-1, a.shape[-1]).astype(np.float32)
            scale = np.abs(flat).max(0)
            scale[scale == 0] = 1
            q = np.clip(np.rint(flat / scale * 127), -127, 127).astype(np.int8)
            spec["scale"] = self._put(scale.astype(np.float32).tobytes())
            spec["data"] = self._put(q.tobytes())
        else:
            if dtype in ("f16", "f32") and not np.all(np.isfinite(a.astype(np.float64))):
                raise ValueError(f"{name} contains non-finite values")
            if dtype == "f16" and a.size and np.abs(a).max() > 6e4:
                dtype = spec["dtype"] = "f32"  # keep very large activations exact
            spec["data"] = self._put(a.astype(DTYPES[dtype]).tobytes())
        self.meta["arrays"][name] = spec

    def _put(self, data: bytes) -> list[int]:
        pad = (-len(data)) % 8
        where = [self._offset, len(data)]
        self._chunks.append(data + b"\0" * pad)
        self._offset += len(data) + pad
        return where

    def save(self, directory) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "blob.bin.gz").write_bytes(gzip.compress(b"".join(self._chunks), compresslevel=6))
        self.meta["blob_bytes"] = self._offset
        (out / "manifest.json").write_text(json.dumps(self.meta, separators=(",", ":")))
        return out


def grid_order(features: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Place similar units near each other on a rows x cols grid.

    ``features`` has one row per unit. Units are ordered into horizontal bands by the
    first principal component and within each band by the second. The result lists the
    unit index shown at each grid cell in row-major order. The arrangement is only a
    display choice; the viewer can switch back to raw index order.
    """
    n = rows * cols
    x = np.asarray(features, dtype=np.float64)
    assert x.shape[0] == n, (x.shape, rows, cols)
    x = x - x.mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    p1, p2 = u[:, 0] * s[0], u[:, 1] * s[1]
    # Make the orientation deterministic.
    if p1[np.argmax(np.abs(p1))] < 0:
        p1 = -p1
    if p2[np.argmax(np.abs(p2))] < 0:
        p2 = -p2
    cells = np.argsort(p1, kind="stable").reshape(rows, cols)
    for r in range(rows):
        cells[r] = cells[r][np.argsort(p2[cells[r]], kind="stable")]
    return cells.ravel().astype(np.int32)


def grid_shape(n: int) -> tuple[int, int]:
    """Choose a near-square rows x cols factorization, wider than tall."""
    best = (1, n)
    for r in range(1, int(n ** 0.5) + 1):
        if n % r == 0:
            best = (r, n // r)
    return best
