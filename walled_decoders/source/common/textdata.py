"""Pinned text sources, tokenized once into shared uint32 shards.

Sources (downloaded by ``common/fetch.py``; dataset revisions are pinned there):

* ``english``   FineWeb-Edu sample-10BT shard 000. Every 100th document is validation.
* ``welsh``     FineWeb-2 cym_Latn. Train file for training, the published test file for validation.
* ``icelandic`` FineWeb-2 isl_Latn, split the same way.
* ``code``      GitHub Python files from codeparrot-clean (files 1-2 train, the published
                validation file 54 for validation). Large enough to pretrain on.
* ``python``    CPython source files (.py) from the archive used by fixed-reference-benchmark.
                Files whose CRC32 of the path is divisible by 20 are validation.

Documents are separated by the tokenizer's end-of-text token. Shards are built on first use
under ``program-data/tokens/<tokenizer>/`` and reused by every later job.
"""
from __future__ import annotations

import fcntl
import io
import json
import tarfile
import zlib
from pathlib import Path

import numpy as np

from common.paths import DATA, RAW, TOKENS, model_path

SOURCES = {
    "english": dict(train=RAW / "fineweb-edu/sample/10BT/000_00000.parquet", val=None, every=100),
    "welsh": dict(train=RAW / "fineweb-2/data/cym_Latn/train/000_00000.parquet",
                  val=RAW / "fineweb-2/data/cym_Latn/test/000_00000.parquet"),
    "icelandic": dict(train=RAW / "fineweb-2/data/isl_Latn/train/000_00000.parquet",
                      val=RAW / "fineweb-2/data/isl_Latn/test/000_00000.parquet"),
    "python": dict(archive=DATA.parent / "routed-expert-splice/cache/cpython.tar.gz"),
    "code": dict(train=[RAW / "codeparrot-clean/file-000000000001.json.gz", RAW / "codeparrot-clean/file-000000000002.json.gz"],
                 val=[RAW / "codeparrot-clean-valid/file-000000000054.json.gz"]),
}

TOKENIZERS = {
    "qwen": lambda: model_path("qwen0.5b") / "tokenizer.json",
    "olmoe": lambda: model_path("olmoe") / "tokenizer.json",
    "bpe8k": lambda: DATA / "tokenizers" / "bpe8k.json",
}
EOS = {"qwen": 151643}


def documents(source: str, split: str):
    """Yield raw text documents in a fixed order."""
    spec = SOURCES[source]
    if source == "python":
        with tarfile.open(spec["archive"]) as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(".py")),
                             key=lambda m: m.name)
            for m in members:
                is_val = zlib.crc32(m.name.encode()) % 20 == 0
                if is_val != (split == "val"):
                    continue
                data = tar.extractfile(m).read()
                text = data.decode("utf-8", errors="replace")
                if text.strip():
                    yield text
        return
    if source == "code":
        import gzip
        for path in spec[split]:
            with gzip.open(path, "rt") as f:
                for line in f:
                    text = json.loads(line)["content"]
                    if text.strip():
                        yield text
        return
    import pyarrow.parquet as pq
    path = spec["train"] if (split == "train" or spec.get("every")) else spec["val"]
    every = spec.get("every")
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2048):
        for text in batch.column(0).to_pylist():
            if every:
                is_val = index % every == 0
                index += 1
                if is_val != (split == "val"):
                    continue
            yield text


def tokenizer(name: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZERS[name]()))


def eos_id(name: str) -> int:
    if name in EOS:
        return EOS[name]
    tok = tokenizer(name)
    for candidate in ("<|endoftext|>", "</s>", "<eos>", "<|eos|>"):
        i = tok.token_to_id(candidate)
        if i is not None:
            return i
    raise ValueError(f"no end-of-text token in {name}")


def shard(source: str, tok: str, split: str, tokens: int) -> np.ndarray:
    """Return at least ``tokens`` token ids (fewer if the source runs out) as a read-only memmap."""
    out = TOKENS / tok / f"{source}.{split}.u32"
    meta_path = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if not meta or (meta["tokens"] < tokens and not meta["exhausted"]):
            meta = _build(source, tok, split, tokens, out)
            meta_path.write_text(json.dumps(meta, indent=2))
    data = np.memmap(out, dtype=np.uint32, mode="r", shape=(meta["tokens"],))
    return data[:tokens] if meta["tokens"] > tokens else data


def _build(source, tok, split, tokens, out):
    t = tokenizer(tok)
    eos = eos_id(tok)
    tmp = out.with_suffix(".tmp")
    n, docs, exhausted = 0, 0, True
    batch = []
    with open(tmp, "wb") as f:
        def flush(batch):
            nonlocal n, docs
            for enc in t.encode_batch(batch, add_special_tokens=False):
                ids = np.asarray(enc.ids + [eos], dtype=np.uint32)
                f.write(ids.tobytes())
                n += len(ids)
                docs += 1
        for text in documents(source, split):
            batch.append(text)
            if len(batch) == 512:
                flush(batch)
                batch = []
                if n >= tokens:
                    exhausted = False
                    break
        if batch:
            flush(batch)
    tmp.replace(out)
    return dict(source=source, tokenizer=tok, split=split, tokens=n, documents=docs, exhausted=exhausted)


class Windows:
    """Fixed-order training windows: every arm that uses the same seed sees the same tokens."""

    def __init__(self, data: np.ndarray, seq: int, batch: int, seed: int = 0):
        self.data, self.seq, self.batch = data, seq, batch
        self.count = (len(data) - 1) // seq
        self.order = np.random.default_rng(seed).permutation(self.count)

    def __len__(self):
        return self.count // self.batch

    def get(self, step: int) -> np.ndarray:
        """Batch ``step`` as [batch, seq + 1] ids. Wraps around after one epoch."""
        idx = [self.order[(step * self.batch + i) % self.count] for i in range(self.batch)]
        return np.stack([np.asarray(self.data[j * self.seq: j * self.seq + self.seq + 1], dtype=np.int32) for j in idx])


def val_windows(data: np.ndarray, seq: int, count: int) -> np.ndarray:
    """The first ``count`` non-overlapping windows, [count, seq + 1]."""
    count = min(count, (len(data) - 1) // seq)
    return np.stack([np.asarray(data[i * seq: i * seq + seq + 1], dtype=np.int32) for i in range(count)])
