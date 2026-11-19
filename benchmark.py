# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.4", "numpy==2.5.5"]
# ///
"""Run on an Apple GPU: uv run benchmark.py prepare|pilot|run|summarize [--width 64]."""
import argparse
from functools import partial
import hashlib
import itertools
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEEDS = [101, 211, 307, 401, 503]
VARIANTS = ["ordinary", "endpoint", "writable", "fixed"]
STEPS, SLOTS, COLORS, REF, BATCH = 6, 8, 4, 4, 250
SWAPS = list(itertools.combinations(range(SLOTS), 2))
COUNT = len(SWAPS) + SLOTS
LEFT = np.array([a for a, b in SWAPS] + list(range(SLOTS)))
RIGHT = np.array([b for a, b in SWAPS] + list(range(SLOTS)))


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def trace_sources(cmd, skip=-1):
    rows = np.arange(len(cmd))
    origin = np.tile(np.arange(SLOTS), (len(cmd), 1))
    for step, codes in enumerate(cmd.T):
        if step == skip:
            continue
        a, b = LEFT[codes], RIGHT[codes]
        prev_a = origin[rows, a].copy()
        swap = codes < len(SWAPS)
        origin[rows, a] = np.where(swap, origin[rows, b], a)
        origin[rows, b] = np.where(swap, prev_a, b)
    return origin


def examples(rng, size):
    board = rng.integers(COLORS, size=(size, SLOTS), dtype=np.int32)
    shape = (size, STEPS)
    cmd = np.where(rng.random(shape) < .5, rng.integers(len(SWAPS), size=shape),
                   rng.integers(len(SWAPS), COUNT, size=shape)).astype(np.int32)
    q = rng.integers(SLOTS, size=size, dtype=np.int32)
    row = np.arange(size)
    target_src = trace_sources(cmd)[row, q]
    start, final = board[row, q], board[row, target_src]
    sensitive = np.zeros(size, dtype=bool)
    for step in range(STEPS):
        skipped = trace_sources(cmd, step)[row, q]
        sensitive |= (cmd[:, step] >= len(SWAPS)) & (board[row, skipped] != final)
    return dict(board=board, cmd=cmd, q=q, start=start, final=final, src=target_src,
                sensitive=sensitive)


def keys(batch):
    # 16 board bits and six 6-bit commands fit exactly in uint64.
    key = np.zeros(len(batch["board"]), dtype=np.uint64)
    for col in batch["board"].T:
        key = key * 4 + col.astype(np.uint64)
    for col in batch["cmd"].T:
        key = key * 64 + col.astype(np.uint64)
    return key


def prepare():
    DATA.mkdir(exist_ok=True)
    seen = set()
    for split, size, seed in [("train", 50000, 773), ("val", 5000, 991),
                               ("test", 10000, 1301), ("sensitive", 10000, 1601)]:
        rng = np.random.default_rng(seed)
        chunks, used = [], 0
        bins = np.zeros(COLORS, dtype=int)
        while used < size:
            gen = examples(rng, 8000)
            selected = []
            for i, key in enumerate(keys(gen).tolist()):
                color = gen["final"][i]
                if key in seen:
                    continue
                if split == "sensitive" and (not gen["sensitive"][i] or bins[color] >= size//COLORS):
                    continue
                selected.append(i)
                seen.add(key)
                bins[color] += 1
                if len(selected) + used == size:
                    break
            if selected:
                chunks.append({k: v[selected] for k, v in gen.items()})
                used += len(selected)
        joined = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        np.savez(DATA / (split + ".npz"), **joined)
        print(split, used, "sensitive", joined["sensitive"].sum(), "colors", bins.tolist(), flush=True)
    assert len(seen) == 75000
    # Independent interpreter operating on colors, including counterfactual no-ops.
    ex = examples(np.random.default_rng(6113), 512)
    for i, cmds in enumerate(ex["cmd"]):
        board, q = ex["board"][i].tolist(), ex["q"][i]
        answers = []
        for omit in range(-1, STEPS):
            work = board.copy()
            for j, code in enumerate(cmds):
                if j == omit:
                    continue
                if code < len(SWAPS):
                    a, b = SWAPS[code]
                    work[a], work[b] = work[b], work[a]
                else:
                    a = code - len(SWAPS)
                    work[a] = board[a]
            answers.append(work[q])
        assert answers[0] == ex["final"][i]
        changed = any(cmds[j] >= len(SWAPS) and answers[1+j] != answers[0] for j in range(STEPS))
        assert changed == ex["sensitive"][i]
    print("Independent simulator and split checks passed.", flush=True)


def array(rng, shape, scale):
    return mx.array(rng.normal(0., scale, shape).astype("float32"))


class Edit(nn.Module):
    def __init__(self, rng, width, variant):
        super().__init__()
        h = width * 2
        n_in = width - REF if variant == "endpoint" else width
        n_out = width if variant == "ordinary" else width - REF
        self.in_w = array(rng, (width, h), width**-.5)[:n_in]
        self.offset = array(rng, (COUNT, h), .25)
        self.gain = array(rng, (COUNT, h), .25) + 1.
        out = array(rng, (h, width), .4 * h**-.5)
        self.out_w = out[:, :n_out]
        if variant == "writable":
            self.ref_w = mx.zeros((h, REF))

    def __call__(self, x, cmd):
        hidden = nn.silu(x @ self.in_w + self.offset[cmd]) * self.gain[cmd]
        delta = hidden @ self.out_w
        if "ref_w" in self:
            delta = mx.concatenate([delta, hidden @ self.ref_w], axis=1)
        return delta


class BoardNet(nn.Module):
    def __init__(self, width, variant, seed):
        super().__init__()
        self.variant = variant
        rng = np.random.default_rng(seed)
        self.encoder = array(rng, (SLOTS*COLORS, width), SLOTS**-.5)
        self.bias = mx.zeros(width)
        self.edits = [Edit(rng, width, variant) for _ in range(STEPS)]
        self.readout = array(rng, (width, 2*SLOTS*COLORS), width**-.5)
        self.out_bias = mx.zeros(2*SLOTS*COLORS)

    def encode(self, board):
        return board @ self.encoder + self.bias

    def step(self, state, ref, code, block):
        if self.variant == "endpoint":
            return state + block(state, code)
        if self.variant == "fixed":
            inp = mx.concatenate([state, ref], axis=1)
            return state + block(inp, code)
        return state + block(state, code)

    def __call__(self, board, cmds, q, override=None):
        orig = self.encode(board)
        ref = orig[:, -REF:] if override is None else override
        limited = self.variant in ("fixed", "endpoint")
        state = orig[:, :-REF] if limited else orig
        for i, block in enumerate(self.edits):
            state = self.step(state, ref, cmds[:, i], block)
        if limited:
            state = mx.concatenate([state, ref], axis=1)
        logits = (state @ self.readout + self.out_bias).reshape(-1, 2, SLOTS, COLORS)
        return logits[mx.arange(q.size)[:, None], mx.arange(2)[None, :], q[:, None]]


def load(split):
    with np.load(DATA / (split + ".npz")) as f:
        return dict(f)


def boards(x):
    return mx.array(np.eye(COLORS, dtype="float32")[x].reshape(len(x), -1))


def tensors(d):
    y = np.stack([d["start"], d["final"]], axis=1)
    return boards(d["board"]), mx.array(d["cmd"]), mx.array(d["q"]), mx.array(y)


def check_networks(width):
    probe = examples(np.random.default_rng(4019), 40)
    x, c, q, y = tensors(probe)
    fixed, writable, endpoint = [BoardNet(width, v, 17) for v in ("fixed","writable","endpoint")]
    f, w = fixed(x, c, q), writable(x, c, q)
    assert np.allclose(np.array(f), np.array(w), atol=3e-5)
    s = fixed.encode(x)
    ref = s[:, -REF:]
    for j, block in enumerate(fixed.edits):
        s = fixed.step(s[:, :-REF], ref, c[:, j], block)
        s = mx.concatenate([s, ref], axis=1)
        assert np.array_equal(np.array(s[:, -REF:]), np.array(ref))
    # Perturbing only the reference can influence the fixed model's working values.
    # The endpoint implementation cannot read it in any of the six update blocks.
    for model in (fixed, endpoint):
        state = model.encode(x)[:, :-REF]
        a = model.step(state, ref, c[:, 0], model.edits[0])
        b = model.step(state, ref + 1, c[:, 0], model.edits[0])
        assert np.array_equal(np.array(a), np.array(b)) == (model.variant == "endpoint")
    print("Architecture checks passed, including equal fixed/writable logits at initialization.", flush=True)


def evaluate(model, d, patch=False):
    x, c, q, y = tensors(d)
    predicts, losses, patched, donors = [], [], [], []
    for off in range(0, len(d["q"]), 500):
        sl = slice(off, off+500)
        logits = model(x[sl], c[sl], q[sl])
        predicts.append(np.array(mx.argmax(logits, axis=-1)))
        losses.append(np.array(nn.losses.cross_entropy(logits, y[sl], reduction="none")))
        if patch:
            alt = d["board"][sl].copy()
            idx = np.arange(len(alt)), d["src"][sl]
            alt[idx] = (alt[idx]+1) % COLORS
            other = boards(alt)
            donor_ref = model.encode(other)[:, -REF:]
            donors.append(np.array(mx.argmax(model(other,c[sl],q[sl])[:,1], axis=-1)))
            patched.append(np.array(mx.argmax(model(x[sl],c[sl],q[sl],donor_ref)[:,1], axis=-1)))
    pred = np.concatenate(predicts).astype(np.uint8)
    acc = pred == np.stack([d["start"], d["final"]],axis=1)
    metrics = {"original": float(acc[:,0].mean()), "current": float(acc[:,1].mean()),
               "sensitive": float(acc[d["sensitive"],1].mean()),
               "changed_current": float(acc[d["start"]!=d["final"],1].mean()),
               "log_loss": np.concatenate(losses).mean(axis=0).tolist()}
    extras = {}
    if patch:
        donor, edit = np.concatenate(donors), np.concatenate(patched)
        alt_target = (d["final"]+1) % COLORS
        ok = acc[:,1] & (donor == alt_target)
        metrics["patch"] = {"both_correct": int(ok.sum()), "toward_donor": int((ok & (edit==alt_target)).sum()),
                            "stayed_original": int((ok & (edit==d["final"])).sum())}
        extras = {"patched":edit.astype(np.uint8), "donor":donor.astype(np.uint8)}
    return metrics, pred, extras


def train(variant, seed, width, epochs, out, pilot=False):
    out.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    model = BoardNet(width, variant, seed)
    optimizer = optim.Adam(learning_rate=.002)
    def loss_fn(m, x, c, q, y):
        return nn.losses.cross_entropy(m(x,c,q), y, reduction="mean")
    value_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def update(x,c,q,y):
        loss, grads = value_grad(model,x,c,q,y)
        optimizer.update(model,grads)
        return loss
    tr, val = load("train"), load("val")
    x,c,q,y = tensors(tr)
    mx.eval(x,c,q,y,model.parameters(),optimizer.state)
    rng = np.random.default_rng(50000 + seed)
    curve = []
    train_seconds = 0.
    mx.reset_peak_memory()
    for ep in range(1, epochs + 1):
        t = time.perf_counter()
        perm = mx.array(rng.permutation(len(tr["q"])).astype(np.int32))
        for k in range(0,len(tr["q"]),BATCH):
            rows = perm[k:k+BATCH]
            loss = update(x[rows], c[rows], q[rows], y[rows])
            mx.eval(loss, state)
        train_seconds += time.perf_counter()-t
        vm = evaluate(model,val)[0]
        curve.append(vm)
        if pilot or ep in (1, epochs):
            print(f"{variant} seed {seed} epoch {ep} validation current {vm['current']:.3f} restore {vm['sensitive']:.3f}",flush=True)
        if pilot and ep in (4,8,12,16) and vm["current"]>=.75 and vm["sensitive"]>=.72:
            break
    result = dict(variant=variant, seed=seed, width=width, epochs=ep, batch=BATCH,
                  examples_presented=ep*len(tr["q"]), parameters=sum(v.size for _,v in tree_flatten(model.trainable_parameters())),
                  training_seconds=train_seconds, validation=curve, train_peak_mib=mx.get_peak_memory()/2**20)
    if not pilot:
        predictions = {}
        for split in ("test","sensitive"):
            metrics, pred, extra = evaluate(model,load(split), patch=variant=="fixed" and split=="sensitive")
            result[split] = metrics
            predictions[split] = pred
            predictions.update(extra)
        np.savez_compressed(out/"predictions.npz", **predictions)
    result["elapsed_seconds"] = time.perf_counter()-start
    write_json(out/"result.json",result)
    return result


def run(args):
    mx.set_default_device(mx.gpu)
    stamp = dict(host=socket.gethostname(),device=mx.device_info()["device_name"],os=platform.platform(),
                 python=sys.version.split()[0],mlx=mx.__version__,numpy=np.__version__,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.mode == "prepare":
        prepare()
        check_networks(args.width)
        write_json(ROOT/"environment.json",stamp)
        return
    area = ROOT / f"width-{args.width}"
    area.mkdir(exist_ok=True)
    if args.mode == "pilot":
        check_networks(args.width)
        info=train("ordinary",19,args.width,16,area/"pilot",pilot=True)
        print("Locked common epoch budget:",info["epochs"],flush=True)
        return
    budget = json.loads((ROOT/"width-64"/"pilot"/"result.json").read_text())["epochs"]
    if args.mode == "run":
        for i, seed in enumerate(SEEDS):
            # Rotate the order to distribute first-run and host-load effects.
            for variant in VARIANTS[i%4:] + VARIANTS[:i%4]:
                dest = area / f"{seed}-{variant}"
                if (dest/"result.json").exists() and (dest/"predictions.npz").exists():
                    continue
                res=train(variant,seed,args.width,budget,dest)
                print(f"DONE {variant} {seed}: restore {res['sensitive']['current']:.4f} in {res['training_seconds']:.1f}s",flush=True)
                mx.clear_cache()
    else:
        from analyze import summarize
        summarize(ROOT, area, SEEDS, VARIANTS)
        print("Summary written",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","pilot","run","summarize"])
    parser.add_argument("--width",type=int,default=64)
    run(parser.parse_args())
