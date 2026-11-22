"""Model sizes, batch shape and compute accounting shared by E3 and E5 jobs."""
from __future__ import annotations

from e3.model import Config

SEQ = 512
PER_DOMAIN = 40                      # sequences per training domain in every batch
BATCH = 3 * PER_DOMAIN               # 120 sequences, 61,440 tokens per update
TOKENS_PER_STEP = BATCH * SEQ

SIZES = {
    "screen": dict(d=384, layers=6, head_dim=32, doors=16),
    "main": dict(d=768, layers=12, head_dim=64, doors=32),
}
PEAK_LR = {"screen": 1.5e-3, "main": 6e-4}
MICRO = {"screen": 2, "main": 3}     # gradient-accumulation chunks per update (bounds activation memory)
TOPO_DEFAULT = 0.05


def config(scale: str, arm: str, topo_lambda: float = 0.0) -> Config:
    return Config(arm=arm, seq=SEQ, topo_lambda=topo_lambda if arm == "topographic" else 0.0, **SIZES[scale])


def flops_per_token(cfg: Config, layers: int, trainable_layers: int | None = None, train_head: bool = True) -> float:
    """Training FLOPs per token: 2N forward for every layer, 4N backward for trainable layers.

    N counts active block parameters (top-2 experts for the routed arm) plus the attention
    score and value products (2 * T * attention width per layer, forward).
    """
    d, T = cfg.d, cfg.seq
    if cfg.arm == "walled":
        R, w, g, h = cfg.rooms, cfg.room_width, cfg.doors, cfg.room_hidden
        block = R * (4 * w * (w + g) + 3 * (w + g) * h)
        attn = 2 * T * R * w
    else:
        block = 4 * d * d + 3 * d * cfg.hidden + (d * cfg.rooms if cfg.arm == "routed" else 0)
        attn = 2 * T * d
    head = cfg.width * cfg.vocab
    trainable_layers = layers if trainable_layers is None else trainable_layers
    forward = 2 * (layers * (block + attn) + head)
    backward = 4 * (trainable_layers * (block + attn) + (head if train_head else 0))
    return float(forward + backward)
