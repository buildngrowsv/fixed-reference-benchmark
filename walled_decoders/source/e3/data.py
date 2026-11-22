"""Tokenizer, training mixture, validation sets and probe texts for E3 and E5."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from common import textdata
from common.paths import DATA

TOK = "bpe8k"
TRAIN_DOMAINS = ("english", "code", "welsh")
GROWTH_DOMAIN = "icelandic"
TOKENIZER = DATA / "tokenizers" / "bpe8k.json"


def build_tokenizer(chars_per_domain: int = 60_000_000, log=print) -> dict:
    """Byte-level BPE with 8,192 entries trained on an equal-character mix of the three domains.

    Documents are taken in file order, so the result is deterministic for pinned inputs.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if TOKENIZER.exists():
        return json.loads(TOKENIZER.with_suffix(".meta.json").read_text())

    def texts():
        for domain in TRAIN_DOMAINS:
            n = 0
            for doc in textdata.documents(domain, "train"):
                doc = doc[: chars_per_domain - n]
                n += len(doc)
                yield doc
                if n >= chars_per_domain:
                    break
            log(f"tokenizer text: {domain} {n:,} characters")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts(), trainer=trainer)
    TOKENIZER.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKENIZER.with_suffix(".tmp")
    tok.save(str(tmp))
    tmp.replace(TOKENIZER)
    meta = dict(vocab=tok.get_vocab_size(), chars_per_domain=chars_per_domain, domains=TRAIN_DOMAINS,
                sha256=hashlib.sha256(TOKENIZER.read_bytes()).hexdigest())
    TOKENIZER.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class Mixture:
    """Equal token shares per batch: ``per_domain`` sequences from each training domain."""

    def __init__(self, tokens_per_domain: int, seq: int, per_domain: int, seed: int, domains=TRAIN_DOMAINS):
        self.windows = [textdata.Windows(textdata.shard(d, TOK, "train", tokens_per_domain), seq, per_domain,
                                         seed=seed * 1000 + k) for k, d in enumerate(domains)]
        self.domains = domains

    def get(self, step: int) -> np.ndarray:
        return np.concatenate([w.get(step) for w in self.windows])


def validation(seq: int, count: int, domains=TRAIN_DOMAINS + (GROWTH_DOMAIN,)) -> dict:
    return {d: textdata.val_windows(textdata.shard(d, TOK, "val", 2_000_000), seq, count) for d in domains}


def probes(tokens: int = 32) -> tuple[np.ndarray, list[str]]:
    """One fixed text per training domain, ``tokens`` long: [3, tokens] ids and labels."""
    rows = []
    for d in TRAIN_DOMAINS:
        data = textdata.shard(d, TOK, "val", 2_000_000)
        eos = np.flatnonzero(np.asarray(data[:200_000]) == textdata.eos_id(TOK))
        # Start at the beginning of the third validation document, a fixed choice.
        start = int(eos[1]) + 1
        rows.append(np.asarray(data[start:start + tokens], dtype=np.int32))
    return np.stack(rows), list(TRAIN_DOMAINS)


def vocab_strings(ids) -> dict:
    tok = textdata.tokenizer(TOK)
    return {str(int(i)): tok.decode([int(i)]) for i in sorted(set(int(x) for x in np.asarray(ids).ravel()))}
