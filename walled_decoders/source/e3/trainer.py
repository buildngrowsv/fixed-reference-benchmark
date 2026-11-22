"""One resumable training loop shared by every E3 and E5 run."""
from __future__ import annotations

import math
import os
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from common import ckpt
from e3.model import loss_fn

CHECKPOINT_SECONDS = float(os.environ.get("E3_CHECKPOINT_SECONDS", 20 * 60))


def lr_at(step: int, steps: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(steps - warmup, 1)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def evaluate(model, val: dict, batch: int = 16, remove=None) -> dict:
    out = {}
    for name, windows in val.items():
        total = count = 0.0
        for i in range(0, len(windows), batch):
            w = mx.array(windows[i:i + batch])
            logits, _ = model(w[:, :-1], remove)
            ce = nn.losses.cross_entropy(logits.astype(mx.float32), w[:, 1:], reduction="sum")
            total += float(ce.item())
            count += w[:, 1:].size
        out[name] = total / count
    return out


def probe_states(model, probe_ids: np.ndarray, remove=None) -> np.ndarray:
    """Residual states on the probe texts: [L + 1, P * T, D] float16."""
    s = model.states(mx.array(probe_ids), remove)
    L1, P, T, D = s.shape
    return np.array(s.reshape(L1, P * T, D).astype(mx.float16))


def train(job, model, *, arm: str, steps: int, peak_lr: float, batches, val: dict, out: Path,
          tokens_per_step: int, warmup: int | None = None, weight_decay: float = 0.1, eval_every: int = 500,
          snapshots: int = 6, probe_ids=None, grad_mask=None, after_update=None, flops_per_token: float = 0.0,
          on_eval=None, micro: int = 1) -> dict:
    """Train the trainable parameters of ``model`` for ``steps`` updates; resumes from ``out``.

    ``batches(step) -> [B, T + 1]`` supplies data. ``grad_mask`` multiplies gradients (a tree
    matching ``model.trainable_parameters()``). ``after_update(model, lr)`` runs after each
    update (the topographic proximal step). Returns the evaluation history and snapshots.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    warmup = max(1, int(0.02 * steps)) if warmup is None else warmup
    optimizer = optim.AdamW(learning_rate=peak_lr, betas=[0.9, 0.95], weight_decay=weight_decay)
    optimizer.init(model.trainable_parameters())
    start, params, extra = ckpt.load(out / "ckpt", optimizer)
    if params is not None:
        model.update(params)
        job.log(f"{arm}: resumed at step {start}")
    history = extra.get("history", [])
    snap_steps = sorted(set([0] + [round(steps * (k + 1) / snapshots) for k in range(snapshots)]))
    snaps = {}
    if probe_ids is not None and (out / "snapshots.npz").exists():
        with np.load(out / "snapshots.npz") as f:
            snaps = {int(k): f[k] for k in f.files}

    lag = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=[model.state], outputs=[model.state])
    def grads_of(batch):
        (loss, ce), grads = lag(model, batch)
        return loss, ce, grads

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def apply(grads, lr):
        optimizer.learning_rate = lr
        if grad_mask is not None:
            grads = tree_map(lambda g, m: g * m, grads, grad_mask)
        optimizer.update(model, grads)

    def update(batch: np.ndarray, lr):
        """Average gradients over ``micro`` equal chunks of the batch (bounded activation memory)."""
        chunks = np.array_split(batch, micro)
        acc = None
        loss_sum = ce_sum = 0.0
        for c in chunks:
            loss, ce, g = grads_of(mx.array(c))
            g = tree_map(lambda x: x * (len(c) / len(batch)), g)
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(acc, loss, ce)
            loss_sum += float(loss.item()) * len(c) / len(batch)
            ce_sum += float(ce.item()) * len(c) / len(batch)
        apply(acc, mx.array(lr, dtype=mx.float32))
        return loss_sum, ce_sum

    def record(step, train_ce):
        losses = evaluate(model, val)
        rec = dict(arm=arm, step=step, tokens=step * tokens_per_step, flops=step * tokens_per_step * flops_per_token,
                   train_ce=train_ce, lr=lr_at(step, steps, peak_lr, warmup), **{f"val_{k}": v for k, v in losses.items()})
        if on_eval is not None:
            rec.update(on_eval(model) or {})
        history.append(rec)
        job.metrics(rec)
        job.log(arm, step, {k: round(v, 4) for k, v in rec.items() if k.startswith("val_") or k == "train_ce"})

    def snap(step):
        if probe_ids is None or step not in snap_steps or step in snaps:
            return
        snaps[step] = probe_states(model, probe_ids)
        tmp = out / "snapshots.tmp.npz"
        np.savez(tmp, **{str(k): v for k, v in snaps.items()})
        tmp.replace(out / "snapshots.npz")

    def save(step):
        ckpt.save(out / "ckpt", step, model.parameters(), optimizer, dict(history=history))

    if start == 0 and not history:
        record(0, float("nan"))
    snap(start)
    last_ckpt = time.time()
    t0, tok0, ce_acc, n_acc = time.time(), 0, 0.0, 0
    for step in range(start, steps):
        lr = lr_at(step, steps, peak_lr, warmup)
        loss, ce = update(batches(step), lr)
        if after_update is not None:
            after_update(model, lr)
        mx.eval(model.parameters(), optimizer.state)
        ce_acc += ce
        n_acc += 1
        tok0 += tokens_per_step
        done = step + 1
        elapsed = time.time() - t0
        job.progress(arm, done, steps, tok0 / max(elapsed, 1e-6))
        if done % eval_every == 0 or done == steps:
            record(done, ce_acc / max(n_acc, 1))
            ce_acc, n_acc = 0.0, 0
        snap(done)
        if time.time() - last_ckpt > CHECKPOINT_SECONDS and done < steps:
            save(done)
            last_ckpt = time.time()
        if not math.isfinite(loss):
            raise FloatingPointError(f"{arm}: non-finite loss at step {done}")
    save(steps)
    job.progress(arm, steps, steps, tok0 / max(time.time() - t0, 1e-6), force=True)
    return dict(history=history, snapshots=snaps, snap_steps=snap_steps)


def save_model(model, cfg, path: Path, extra: dict | None = None) -> None:
    import json
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(dict(config=cfg.to_dict(), layers=len(model.blocks), **(extra or {})), indent=2))


def load_model(path: Path, layers: int | None = None):
    import json
    from e3.model import Config, Model
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    cfg = Config(**meta["config"])
    model = Model(cfg, layers or meta["layers"])
    model.update(tree_unflatten(list(mx.load(str(path / "model.safetensors")).items())))
    mx.eval(model.parameters())
    return model, cfg, meta
