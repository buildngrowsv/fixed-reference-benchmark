"""prog-e3-data: train the shared 8,192-entry tokenizer and build the E3/E5 token shards."""
from __future__ import annotations

import argparse
import hashlib
import shutil

import numpy as np

from common import textdata
from common.job import Job
from common.paths import TOKENS
from e3 import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=150_000_000, help="per training domain")
    ap.add_argument("--growth-tokens", type=int, default=40_000_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.train_tokens, a.growth_tokens = 2_000_000, 1_000_000
    job = Job("prog-e3-data", vars(a))
    meta = data.build_tokenizer(log=job.log)
    sha = hashlib.sha256(data.TOKENIZER.read_bytes()).hexdigest()
    assert sha == meta["sha256"], "tokenizer changed after its metadata was written"
    marker = TOKENS / data.TOK / "TOKENIZER_SHA256"
    if marker.exists() and marker.read_text().strip() != sha:
        job.log("tokenizer changed: removing stale shards")
        shutil.rmtree(TOKENS / data.TOK)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(sha)

    shards = {}
    plan = [(d, "train", a.train_tokens) for d in data.TRAIN_DOMAINS] + [(data.GROWTH_DOMAIN, "train", a.growth_tokens)]
    plan += [(d, "val", 2_000_000) for d in data.TRAIN_DOMAINS + (data.GROWTH_DOMAIN,)]
    for k, (domain, split, n) in enumerate(plan):
        job.progress("tokenize", k, len(plan), force=True)
        arr = textdata.shard(domain, data.TOK, split, n)
        sample = np.asarray(arr[:200_000])
        text = textdata.tokenizer(data.TOK).decode(sample.tolist(), skip_special_tokens=False)
        shards[f"{domain}.{split}"] = dict(tokens=int(len(arr)), bytes_per_token=len(text.encode()) / len(sample))
        job.log(domain, split, f"{len(arr):,} tokens", f"{shards[f'{domain}.{split}']['bytes_per_token']:.2f} bytes/token")
    job.finish("data_ready", tokenizer=meta, shards=shards)


if __name__ == "__main__":
    main()
