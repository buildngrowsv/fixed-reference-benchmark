"""Retention benchmarks scored by log-likelihood: ARC-Easy and HellaSwag.

Each item is a context plus candidate continuations. The prediction is the candidate with
the highest length-normalized log-likelihood (acc_norm style: mean per token). Items are
a fixed, seeded subset so every arm is scored on the same questions.
"""
from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

from common.paths import RAW


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def items(name: str, count: int = 500, seed: int = 0):
    import pyarrow.parquet as pq
    if name == "arc_easy":
        t = pq.read_table(RAW / "ai2_arc/ARC-Easy/test-00000-of-00001.parquet").to_pylist()
        out = []
        for r in t:
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            out.append(dict(context=f"Question: {r['question']}\nAnswer:",
                            choices=[" " + c for c in r["choices"]["text"]],
                            gold=labels.index(r["answerKey"])))
    elif name == "hellaswag":
        t = pq.read_table(RAW / "hellaswag/data/validation-00000-of-00001.parquet").to_pylist()
        out = []
        for r in t:
            ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
            out.append(dict(context=_hellaswag_clean(r["activity_label"] + ": " + ctx),
                            choices=[" " + _hellaswag_clean(e) for e in r["endings"]], gold=int(r["label"])))
    else:
        raise KeyError(name)
    order = np.random.default_rng(seed).permutation(len(out))[:count]
    return [out[i] for i in sorted(order)]


def multiple_choice(logits_fn, encode, data, batch: int = 16) -> float:
    """Accuracy of ``logits_fn(ids [B, T]) -> logits [B, T, V]`` on ``items`` output."""
    rows, owners = [], []
    for k, it in enumerate(data):
        ctx = encode(it["context"])
        for c in it["choices"]:
            cont = encode(c)
            rows.append((ctx, cont))
            owners.append(k)
    scores = np.zeros(len(rows))
    order = np.argsort([len(a) + len(b) for a, b in rows])
    for s in range(0, len(order), batch):
        chunk = [rows[i] for i in order[s:s + batch]]
        width = max(len(a) + len(b) for a, b in chunk)
        ids = np.zeros((len(chunk), width), dtype=np.int32)
        for r, (a, b) in enumerate(chunk):
            ids[r, :len(a) + len(b)] = a + b
        logp = logits_fn(mx.array(ids)).astype(mx.float32)
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        logp = np.array(logp)
        for r, (a, b) in enumerate(chunk):
            pos = np.arange(len(a) - 1, len(a) + len(b) - 1)
            scores[order[s + r]] = logp[r, pos, np.asarray(b)].mean()
    correct = 0
    for k, it in enumerate(data):
        mine = [scores[i] for i, o in enumerate(owners) if o == k]
        correct += int(np.argmax(mine) == it["gold"])
    return correct / len(data)
