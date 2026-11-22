"""Loading and layer-by-layer running of mlx-lm decoder models (Qwen2 family and relatives)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from common.paths import model_path


def load(name: str, lazy: bool = False, model_config: dict | None = None):
    """Return (model, tokenizer) for a pinned checkpoint name from ``common.paths.MODELS``.

    ``model_config`` overrides config.json fields (OLMoE's config omits rms_norm_eps)."""
    from mlx_lm.utils import load as mlx_load
    model, tok = mlx_load(str(model_path(name)), lazy=lazy, model_config=model_config or {})
    return model, tok


def core(model):
    return model.model


def layers(model):
    return model.model.layers


def embed(model, ids):
    return model.model.embed_tokens(ids)


def logits_from_hidden(model, h):
    """Apply the final norm and output projection to residual states."""
    h = model.model.norm(h)
    if getattr(model.args, "tie_word_embeddings", False) or not hasattr(model, "lm_head"):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def run_layers(model, x, start=0, stop=None, mask=None, hooks=None, keep=None):
    """Run blocks ``start:stop`` on residual ``x``.

    ``hooks`` maps a state index to ``f(h) -> h`` applied to the residual entering that block
    (state index i = output of block i-1; 0 is the embedding). ``keep`` lists state indices
    to return. Returns (x, {index: state}).
    """
    blocks = model.model.layers
    stop = len(blocks) if stop is None else stop
    if mask is None:
        mask = create_attention_mask(x, None)
    kept = {}
    hooks = hooks or {}
    for i in range(start, stop):
        if i in hooks:
            x = hooks[i](x)
        if keep is not None and i in keep:
            kept[i] = x
        x = blocks[i](x, mask, None)
    if stop in hooks:
        x = hooks[stop](x)
    if keep is not None and stop in keep:
        kept[stop] = x
    return x, kept


def forward(model, ids, hooks=None, keep=None):
    """Full forward pass returning (logits, kept states)."""
    x = embed(model, ids)
    x, kept = run_layers(model, x, 0, None, hooks=hooks, keep=keep)
    return logits_from_hidden(model, x), kept


def token_loss(logits, targets, mask=None):
    """Mean cross-entropy over positions where ``mask`` is 1 (all positions if None)."""
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    if mask is None:
        return ce.mean()
    mask = mask.astype(mx.float32)
    return (ce * mask).sum() / mx.maximum(mask.sum(), 1)


def mean_loss(logits_fn, windows: np.ndarray, batch: int = 8) -> float:
    """Mean next-token loss of ``logits_fn(ids)`` over fixed windows [N, T+1]."""
    total, count = 0.0, 0
    for i in range(0, len(windows), batch):
        w = mx.array(windows[i:i + batch])
        logits = logits_fn(w[:, :-1])
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), w[:, 1:], reduction="sum")
        total += float(ce.item())
        count += int(w[:, 1:].size)
    return total / count


def bf16(model):
    model.set_dtype(mx.bfloat16)
    return model


def param_count(tree) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(tree) if isinstance(v, mx.array)))
