"""From-scratch decoder models for E3 (walls) and E5 (bottom-up growth).

Four arms share one pre-norm transformer recipe (RoPE, SwiGLU, untied embeddings):

* ``dense``        width d, all modules read and write the whole residual stream.
* ``walled``       R rooms of width d/2 plus G shared door channels. A room's attention and
                   feed-forward read only that room and the doors and write only that room and
                   the doors. Rooms are a batched leading dimension, not masks on dense matrices.
* ``topographic``  the dense model with fixed positions on a circle for every residual channel,
                   attention dimension and hidden unit, plus a wiring-length penalty applied as
                   a proximal step (see ``wiring_prox``).
* ``routed``       dense attention; the feed-forward is R experts of half the dense hidden size
                   with a learned top-2 router, so active parameters equal the dense model.

Parameters are float32; matrix products run in bfloat16.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

DT = mx.bfloat16


@dataclass
class Config:
    arm: str = "dense"
    vocab: int = 8192
    d: int = 768
    layers: int = 12
    head_dim: int = 64
    rooms: int = 4
    doors: int = 32
    seq: int = 512
    top_k: int = 2
    balance: float = 0.01
    topo_lambda: float = 0.0

    @property
    def hidden(self) -> int:
        """SwiGLU hidden size matched to a 4d two-matrix MLP (8d/3, rounded to 64)."""
        return int(round(8 * self.d / 3 / 64)) * 64

    @property
    def room_width(self) -> int:
        return self.d // 2

    @property
    def room_hidden(self) -> int:
        return int(round(8 * self.room_width / 3 / 64)) * 64

    @property
    def width(self) -> int:
        if self.arm == "walled":
            return self.rooms * self.room_width + self.doors
        return self.d

    def to_dict(self):
        return asdict(self)


def _normal(shape, std):
    return mx.random.normal(shape) * std


def _rope(x, head_dim):
    return mx.fast.rope(x, head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)


class DenseBlock(nn.Module):
    def __init__(self, c: Config, depth_scale: float):
        super().__init__()
        d, h = c.d, c.hidden
        self.n1 = mx.ones((d,))
        self.n2 = mx.ones((d,))
        self.wqkv = _normal((d, 3 * d), 0.02)
        self.wo = _normal((d, d), 0.02 * depth_scale)
        if c.arm == "routed":
            he = h // 2
            self.router = _normal((d, c.rooms), 0.02)
            self.wgu = _normal((c.rooms, d, 2 * he), 0.02)
            self.wd = _normal((c.rooms, he, d), 0.02 * depth_scale)
        else:
            self.wgu = _normal((d, 2 * h), 0.02)
            self.wd = _normal((h, d), 0.02 * depth_scale)
        self.c = c

    def __call__(self, x, remove=None):
        """``remove`` is a room index whose units write nothing (None keeps everything)."""
        c = self.c
        B, T, d = x.shape
        H = d // c.head_dim
        u = mx.fast.rms_norm(x, self.n1, 1e-5).astype(DT)
        qkv = (u @ self.wqkv.astype(DT)).reshape(B, T, 3, H, c.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = _rope(qkv[0], c.head_dim), _rope(qkv[1], c.head_dim), qkv[2]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=c.head_dim ** -0.5, mask="causal")
        a = a.transpose(0, 2, 1, 3).reshape(B, T, d)
        if remove is not None:
            a = a * arc_mask(d, c.rooms, remove).astype(DT)
        x = x + (a @ self.wo.astype(DT)).astype(x.dtype)
        u = mx.fast.rms_norm(x, self.n2, 1e-5).astype(DT)
        aux = mx.array(0.0)
        if c.arm == "routed":
            m, aux = self._experts(u, remove)
        else:
            g = u @ self.wgu.astype(DT)
            g1, g2 = mx.split(g, 2, axis=-1)
            act = nn.silu(g1) * g2
            if remove is not None:
                act = act * arc_mask(act.shape[-1], c.rooms, remove).astype(DT)
            m = act @ self.wd.astype(DT)
        return x + m.astype(x.dtype), aux

    def _experts(self, u, remove):
        c = self.c
        B, T, d = u.shape
        N, R, k = B * T, c.rooms, c.top_k
        flat = u.reshape(N, d)
        logits = (flat @ self.router.astype(DT)).astype(mx.float32)
        probs = mx.softmax(logits, axis=-1)
        idx = mx.stop_gradient(mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k])
        top = mx.take_along_axis(logits, idx, axis=-1)
        weights = mx.softmax(top, axis=-1)
        if remove is not None:
            weights = weights * (idx != remove).astype(weights.dtype)
        # Switch-style balance loss: fraction of assignments times mean router probability.
        counts = mx.stop_gradient((idx[..., None] == mx.arange(R)).astype(mx.float32).sum(axis=(0, 1)) / (N * k))
        aux = R * (counts * probs.mean(axis=0)).sum()
        e = idx.reshape(-1)
        order = mx.argsort(e)
        inv = mx.argsort(order)
        xs = mx.take(flat, order // k, axis=0)[:, None, :]
        es = mx.take(e, order)
        h = mx.gather_mm(xs, self.wgu.astype(DT), rhs_indices=es, sorted_indices=True)
        h1, h2 = mx.split(h, 2, axis=-1)
        y = mx.gather_mm(nn.silu(h1) * h2, self.wd.astype(DT), rhs_indices=es, sorted_indices=True)
        y = mx.take(y[:, 0, :], inv, axis=0).reshape(N, k, d)
        out = (y * weights[..., None].astype(DT)).sum(axis=1)
        return out.reshape(B, T, d), aux


class WalledBlock(nn.Module):
    def __init__(self, c: Config, depth_scale: float):
        super().__init__()
        R, w, g, h = c.rooms, c.room_width, c.doors, c.room_hidden
        i = w + g
        self.n1 = mx.ones((R, i))
        self.n2 = mx.ones((R, i))
        self.wqkv = _normal((R, i, 3 * w), 0.02)
        self.wo = _normal((R, w, i), 0.02 * depth_scale)
        self.wgu = _normal((R, i, 2 * h), 0.02)
        self.wd = _normal((R, h, i), 0.02 * depth_scale)
        self.c = c

    def _normed(self, rooms, doors, n):
        """RMS-normalise each room together with the shared doors: [R, B, T, w + g] bf16."""
        c = self.c
        w, g = c.room_width, c.doors
        ss = (rooms * rooms).sum(-1, keepdims=True) + (doors * doors).sum(-1, keepdims=True)[None]
        inv = mx.rsqrt(ss / (w + g) + 1e-5)
        r = rooms * inv * n[:, None, None, :w]
        d = mx.broadcast_to(doors[None], rooms.shape[:-1] + (g,)) * inv * n[:, None, None, w:]
        return mx.concatenate([r, d], axis=-1).astype(DT)

    def _write(self, rooms, doors, y, remove):
        """Rooms add their own channels; door writes from all rooms add up."""
        R, B, T, w = rooms.shape
        if remove is not None:
            y = y * (mx.arange(R) != remove).astype(y.dtype)[:, None, None]
        y = y.reshape(R, B, T, -1).astype(mx.float32)
        return rooms + y[..., :w], doors + y[..., w:].sum(axis=0)

    def __call__(self, state, remove=None):
        """``state`` is (rooms [R, B, T, w], doors [B, T, g]), both float32."""
        c = self.c
        rooms, doors = state
        R, B, T, w = rooms.shape
        hd = c.head_dim
        H = w // hd
        u = self._normed(rooms, doors, self.n1).reshape(R, B * T, -1)
        qkv = mx.matmul(u, self.wqkv.astype(DT)).reshape(R * B, T, 3, H, hd).transpose(2, 0, 3, 1, 4)
        q, k, v = _rope(qkv[0], hd), _rope(qkv[1], hd), qkv[2]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal")
        a = a.transpose(0, 2, 1, 3).reshape(R, B * T, w)
        rooms, doors = self._write(rooms, doors, mx.matmul(a, self.wo.astype(DT)), remove)
        u = self._normed(rooms, doors, self.n2).reshape(R, B * T, -1)
        g1, g2 = mx.split(mx.matmul(u, self.wgu.astype(DT)), 2, axis=-1)
        m = mx.matmul(nn.silu(g1) * g2, self.wd.astype(DT))
        rooms, doors = self._write(rooms, doors, m, remove)
        return (rooms, doors), mx.array(0.0)


def _rms(x, eps=1e-5):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def arc_mask(n: int, rooms: int, remove: int) -> mx.array:
    """1 everywhere except the contiguous arc ``remove`` of ``n`` units split into ``rooms``."""
    i = mx.arange(n)
    lo, hi = remove * n // rooms, (remove + 1) * n // rooms
    return 1 - ((i >= lo) & (i < hi)).astype(mx.float32)


class Model(nn.Module):
    def __init__(self, c: Config, layers: int | None = None):
        super().__init__()
        self.c = c
        L = layers or c.layers
        scale = 1 / math.sqrt(2 * c.layers)
        Block = WalledBlock if c.arm == "walled" else DenseBlock
        self.embed = _normal((c.vocab, c.width), 0.02)
        self.blocks = [Block(c, scale) for _ in range(L)]
        self.norm = mx.ones((c.width,))
        self.unembed = _normal((c.width, c.vocab), 0.02)

    def _split(self, x):
        c = self.c
        B, T, _ = x.shape
        R, w = c.rooms, c.room_width
        return x[..., :R * w].reshape(B, T, R, w).transpose(2, 0, 1, 3), x[..., R * w:]

    @staticmethod
    def _join(state):
        rooms, doors = state
        R, B, T, w = rooms.shape
        return mx.concatenate([rooms.transpose(1, 2, 0, 3).reshape(B, T, R * w), doors], axis=-1)

    def hidden(self, ids, remove=None, keep_states=False, upto=None):
        x = self.embed[ids]
        states = [x] if keep_states else None
        aux = mx.array(0.0)
        walled = self.c.arm == "walled"
        if walled:
            x = self._split(x)
        for b in self.blocks[:upto]:
            x, a = b(x, remove)
            aux = aux + a
            if keep_states:
                states.append(self._join(x) if walled else x)
        if walled:
            x = self._join(x)
        return x, aux, states

    def logits(self, x):
        return mx.fast.rms_norm(x, self.norm, 1e-5).astype(DT) @ self.unembed.astype(DT)

    def __call__(self, ids, remove=None):
        x, aux, _ = self.hidden(ids, remove)
        return self.logits(x), aux

    def states(self, ids, remove=None):
        """Residual stream after the embedding and after every block: [L + 1, B, T, D]."""
        _, _, s = self.hidden(ids, remove, keep_states=True)
        return mx.stack(s)


def loss_fn(model, batch, remove=None):
    ids, targets = batch[:, :-1], batch[:, 1:]
    logits, aux = model(ids, remove)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")
    total = ce + (model.c.balance * aux / len(model.blocks) if model.c.arm == "routed" else 0.0)
    return total, ce


# ---------- topographic wiring ----------

def _circ(a, b):
    d = mx.abs(a[:, None] - b[None, :])
    return mx.minimum(d, 1 - d)


def wiring_distances(c: Config) -> dict:
    """Circular distance between the endpoints of every block weight, keyed by parameter name."""
    d, h = c.d, c.hidden
    res = mx.arange(d) / d
    hid = mx.arange(h) / h
    qkv_out = mx.concatenate([res, res, res])
    return {"wqkv": _circ(res, qkv_out), "wo": _circ(res, res),
            "wgu": _circ(res, mx.concatenate([hid, hid])), "wd": _circ(hid, res)}


def wiring_prox(model, dist: dict, lam: float, lr: float) -> None:
    """Proximal step for lam * sum |W_ij| * dist_ij: shrink each weight toward zero by lr*lam*dist."""
    for b in model.blocks:
        for name, dmat in dist.items():
            w = getattr(b, name)
            t = (lr * lam) * dmat
            setattr(b, name, mx.sign(w) * mx.maximum(mx.abs(w) - t, 0))


def wiring_cost(model, dist: dict) -> float:
    """Mean distance-weighted magnitude per weight (lower = more local wiring)."""
    num = den = 0.0
    for b in model.blocks:
        for name, dmat in dist.items():
            w = mx.abs(getattr(b, name))
            num += float((w * dmat).sum().item())
            den += float(w.sum().item())
    return num / max(den, 1e-12)


def count_params(model) -> dict:
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(model.parameters()))
    total = sum(v.size for v in flat.values())
    emb = model.embed.size + model.unembed.size
    return dict(total=int(total), embedding=int(emb), blocks=int(total - emb))
