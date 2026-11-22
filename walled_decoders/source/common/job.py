"""The contract between a program job and the Studio queue worker.

The worker sets STUDIO_JOB_DIR and treats a job as successful only when it exits 0 and
``result.json`` names a ``decision``. ``progress.json`` drives the live dashboard. A failed
process is retried once, so long loops save checkpoints and resume from them.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_default) + "\n")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


class Job:
    """Output directory, progress, metrics log and final decision for one queued job."""

    def __init__(self, name: str, config: dict | None = None):
        scratch = Path.home() / "research" / "program-scratch" / name
        self.dir = Path(os.environ.get("STUDIO_JOB_DIR", scratch))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.queued = "STUDIO_JOB_DIR" in os.environ
        self.config = config or {}
        self._last_progress = 0.0
        self.started = time.time()
        try:  # keep freed GPU buffers from piling up; the Studio froze when processes oversubscribed RAM
            import mlx.core as mx
            mx.set_cache_limit(int(float(os.environ.get("PROGRAM_MLX_CACHE_GB", "16")) * 1e9))
        except Exception:  # noqa: BLE001
            pass
        write_json(self.dir / "environment.json", dict(
            job=name, argv=sys.argv, config=self.config, started=datetime.now(timezone.utc).isoformat(),
            python=sys.version, platform=platform.platform(), source=self._source(), packages=self._packages()))
        self.log(f"job {name} -> {self.dir}")

    @staticmethod
    def _source():
        here = Path(__file__).resolve().parents[1]
        try:
            rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        except OSError:
            rev = ""
        return dict(path=str(here), git=rev or here.name)

    @staticmethod
    def _packages():
        out = {}
        for name in ("mlx", "mlx_lm", "numpy", "transformers", "tokenizers"):
            try:
                module = __import__(name)
                out[name] = getattr(module, "__version__", None)
            except Exception:  # noqa: BLE001
                out[name] = None
        try:
            import mlx.core as mx
            out["mlx"] = mx.__version__
        except Exception:  # noqa: BLE001
            pass
        return out

    def log(self, *parts) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}]", *parts, flush=True)

    def progress(self, arm: str, step: int, target: int, tokens_per_second: float = 0.0, force=False, **extra) -> None:
        now = time.time()
        if not force and now - self._last_progress < 20:
            return
        self._last_progress = now
        write_json(self.dir / "progress.json", dict(arm=arm, step=int(step), target=int(target),
                                                   tokens_per_second=float(tokens_per_second), **extra))

    def metrics(self, record: dict, name: str = "metrics.jsonl") -> None:
        record = {"time": round(time.time() - self.started, 1), **record}
        with (self.dir / name).open("a") as f:
            f.write(json.dumps(record, default=_default) + "\n")

    def read_metrics(self, name: str = "metrics.jsonl") -> list[dict]:
        path = self.dir / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def save(self, name: str, data) -> Path:
        write_json(self.dir / name, data)
        return self.dir / name

    def load(self, name: str, default=None):
        return read_json(self.dir / name, default)

    def finish(self, decision: str, **report) -> None:
        try:
            import mlx.core as mx
            report.setdefault("peak_memory_gb", round(mx.get_peak_memory() / 1e9, 1))
        except Exception:  # noqa: BLE001
            pass
        report = dict(decision=decision, job=self.name, hours=round((time.time() - self.started) / 3600, 3), **report)
        write_json(self.dir / "result.json", report)
        self.log("decision:", decision)


class Budget:
    """Wall-clock cap for adaptive loops."""

    def __init__(self, hours: float):
        self.end = time.time() + hours * 3600

    def left(self) -> float:
        return max(0.0, self.end - time.time()) / 3600

    def spent(self) -> bool:
        return time.time() >= self.end
