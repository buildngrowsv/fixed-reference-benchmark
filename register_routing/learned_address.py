"""Transfer a payload-blind attention router trained on versioned register recall."""
from functools import partial
from pathlib import Path
from typing import NamedTuple
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

OUT = Path(os.environ["STUDIO_JOB_DIR"])
ALPHABET = np.indices((8, 8)).reshape(2, -1).T
PAYLOADS = 64
PAIRS = 1024


class Batch(NamedTuple):
    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    kind: np.ndarray
    address: np.ndarray
    alternate: np.ndarray
    replacement: np.ndarray


def make_batch(seed, count, held_out=False, offset=0):
    rng = np.random.default_rng(seed)
    size = 20 if held_out else 12
    pool = ALPHABET[(ALPHABET.sum(axis=1) % 2) == int(held_out)]
    keys = pool[np.argsort(rng.random((count, len(pool))), axis=1)[:, :size]]
    order = np.tile(np.arange(size), 3)[np.argsort(rng.random((count, 3*size)), axis=1)]
    selected = np.argsort(rng.random((count, size)), axis=1)[:, :8]
    row = np.arange(count)[:, None]
    positions = np.arange(3*size)[None, None, :]
    match = order[:, None, :] == selected[:, :, None]
    first = np.where(match, positions, 3*size).min(2)
    last = np.where(match, positions, -1).max(2)
    addresses = np.stack((first, last), axis=2).reshape(count, 16)
    values = np.argsort(rng.random((count, PAYLOADS)), axis=1) + offset
    paired = np.repeat(keys[row, selected], 2, axis=1)
    kind = np.tile([0, 1], (count, 8))
    return Batch(keys[row, order], values[:, :3*size], paired, kind, addresses,
                 np.stack((last, first), axis=2).reshape(count, 16), values[:, -1:])


class Locator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Embedding(8, 16)
        self.b = nn.Embedding(8, 16)
        self.query = nn.Linear(32, 32, bias=False)
        self.key = nn.Linear(32, 32, bias=False)
        self.direction = nn.Embedding(2, 1)
        self.direction.weight = mx.zeros((2, 1))

    def symbols(self, tokens):
        return mx.concatenate((self.a(tokens[..., 0]), self.b(tokens[..., 1])), axis=-1)

    def __call__(self, keys, queries, kind):
        q = self.query(self.symbols(queries))
        k = self.key(self.symbols(keys))
        ranks = mx.linspace(-1., 1., keys.shape[1])
        return (q @ k.swapaxes(1, 2)) / (32**.5) + self.direction(kind) * ranks


class Reader(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.locator = Locator()
        self.payload = nn.Embedding(PAYLOADS, 64)
        self.output = nn.Linear(64, PAYLOADS)

    def route(self, keys, queries, kind):
        return self.locator(keys, queries, kind)

    def decode(self, scores, values):
        context = mx.softmax(scores, axis=-1) @ self.payload(values - self.offset)
        return self.output(context)

    def __call__(self, keys, values, queries, kind):
        return self.decode(self.route(keys, queries, kind), values)


def atomic_json(name, data):
    p = OUT/name
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(p)


def labels(batch):
    return np.take_along_axis(batch.values, batch.address, axis=1)


def fit(model, length, split_seed, arm):
    opt = optim.Adam(0.003, bias_correction=True)

    def objective(m, k, v, q, op, y):
        z = m(k, v, q, op)
        return nn.losses.cross_entropy(z, y, reduction="mean")

    gradient = nn.value_and_grad(model, objective)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(k, v, q, op, y):
        loss, grad = gradient(model, k, v, q, op, y)
        opt.update(model, grad)
        return loss

    start = time.monotonic()
    for i in range(1, length + 1):
        b = make_batch([split_seed, i], 64, offset=model.offset)
        value = step(*(mx.array(a, dtype=mx.int32) for a in b[:4]), mx.array(labels(b)-model.offset, dtype=mx.int32))
        mx.eval(value, state)
        if i == 1 or i % 32 == 0:
            x = value.item()
            if not np.isfinite(x):
                raise ValueError(f"{arm} nonfinite at {i}")
            atomic_json("progress.json", dict(arm=arm, step=i, target=length, training_loss=x,
                time=time.time(), tokens_per_second=i*64*36/(time.monotonic()-start)))
    return time.monotonic() - start


def independent_gold(batch):
    """Interpreter independent of generator's vectorized first/last selection."""
    rows = []
    for writes, keys, operations in zip(batch.keys, batch.queries, batch.kind):
        occurrences = {}
        for i, key in enumerate(writes):
            occurrences.setdefault(tuple(key), []).append(i)
        rows.append([occurrences[tuple(k)][-1 if op else 0] for k, op in zip(keys, operations)])
    return np.array(rows)


def swap_rows(scores):
    return mx.concatenate((scores[..., -7:], scores[..., :-7]), axis=-1)


def ties(a):
    return (a == a.max(axis=-1, keepdims=True)).sum(axis=-1) > 1


def predict(model, batch, permute=False):
    k, v, q, op = (mx.array(a, dtype=mx.int32) for a in batch[:4])
    scores = model.route(k, q, op)
    if permute:
        scores = swap_rows(scores)
    result = np.array(model.decode(scores, v))
    idx = np.array(scores)
    return idx.argmax(-1), result.argmax(-1) + model.offset, ties(idx), ties(result)


def key_pairs(query_keys):
    """Find two matched query pairs per program, one for each shared symbol."""
    keys = query_keys[:, ::2]
    equal = keys[:, :, None] == keys[:, None, :]
    upper = np.triu(np.ones((8, 8), dtype=bool), 1)
    pairs = []
    for symbol in (0, 1):
        valid = equal[..., symbol] & ~equal[..., 1-symbol] & upper
        choice = valid.reshape(len(keys), -1).argmax(1)
        pairs.append(np.stack((2*(choice//8), 2*(choice%8)), axis=-1))
        # Rows without a pair on this symbol do not enter this denominator.
        pairs.append(valid.any(axis=(1, 2)))
    return pairs


def payload_intervention(model, b, chosen, clean_route):
    k, v, q, op = (mx.array(a, dtype=mx.int32) for a in b[:4])
    qa = np.arange(q.shape[1])
    via = model.route(k, q, op)
    np.testing.assert_array_equal(np.array(via), clean_route)
    row_ids = np.arange(len(b.values))[:, None]
    v = np.broadcast_to(b.values[:, None, :], (len(b.values), q.shape[1], v.shape[1])).copy()
    v[row_ids, qa, chosen] = b.replacement
    # One query per changed world. The address computation receives exactly the same inputs.
    probs = mx.softmax(via, axis=-1)
    hidden = mx.sum(probs[..., None] * model.payload(mx.array(v-model.offset, dtype=mx.int32)), axis=2)
    return np.array(model.output(hidden).argmax(-1)) + model.offset


def ci(d):
    rng = np.random.default_rng(771921)
    samples = []
    for _ in range(20):
        ix = rng.integers(len(d), size=(200, len(d)))
        samples.extend(d[ix].mean(axis=1))
    return np.quantile(samples, [.025, .975]).tolist()


def score(models, panel_seed, held_out, write_arrays=None):
    entries = []
    for chunk in range(PAIRS//64 if held_out else 4):
        host = make_batch([panel_seed, chunk], 64, held_out, offset=64)
        np.testing.assert_array_equal(independent_gold(host), host.address)
        np.testing.assert_array_equal(independent_gold(host._replace(kind=1-host.kind)), host.alternate)
        donor = host._replace(values=host.values-64)
        predictions = {k:predict(m, donor if k=="donor" else host, permute=k=="permuted")
                       for k,m in models.items()}
        correct = {}
        for name, (p, answer, route_ties, answer_ties) in predictions.items():
            correct[name] = dict(source=p==host.address, accuracy=answer==labels(host),
                route_ties=route_ties, answer_ties=answer_ties)
            if name=="donor":
                correct[name]["accuracy"] = answer == labels(donor)
        d = dict(p={k:v[0] for k,v in predictions.items()},answers={k:v[1] for k,v in predictions.items()})
        d.update(correct=correct, address=host.address, values=host.values, keys=host.keys, queries=host.queries)
        if held_out:
            m = models["graft"]
            args = [mx.array(a, dtype=mx.int32) for a in host[:4]]
            route = np.array(m.route(args[0], args[2], args[3]))
            target = payload_intervention(m, host, host.address, route)
            stale = payload_intervention(m, host, host.alternate, route)
            clean = correct["graft"]["source"] & correct["graft"]["accuracy"]
            ix = np.arange(len(clean))[:, None]
            left, have_left, right, have_right = key_pairs(host.queries)
            key_ok = np.stack((clean[ix,left].all(1),clean[ix,right].all(1)),axis=-1)
            key_have = np.stack((have_left,have_right),axis=-1)
            d.update(key_ok=key_ok,key_have=key_have,time_ok=clean.reshape(-1,8,2).all(2),
                gold_changed=host.replacement,replace_actual=target,stale_actual=stale,
                replace_correct=target==host.replacement,stale_correct=stale==labels(host))
        entries.append(d)

    def cat(key):
        return np.concatenate([r[key] for r in entries])

    arms = {}
    for k in models:
        parts = {m:np.concatenate([r["correct"][k][m] for r in entries]) for m in entries[0]["correct"][k]}
        arms[k] = {m:float(v.mean()) for m,v in parts.items()}
        for name,i in (("first",0),("latest",1)):
            arms[k][name] = float(parts["accuracy"][:,i::2].mean())
    report = dict(arms=arms,programs=len(entries)*64)
    if held_out:
        d = {k:cat(k) for k in entries[0] if k not in ("p","answers","correct")}
        for name in models:
            for key in ("p","answers"):
                d[name+"_"+key] = np.concatenate([r[key][name] for r in entries])
        np.savez_compressed(OUT/write_arrays, **d)
        gold = np.take_along_axis(d["values"],d["address"],axis=1)
        correct = d["graft_answers"] == gold
        r = {}
        for name in ("permuted","native","untrained"):
            diff = correct.mean(1)-(d[name+"_answers"]==gold).mean(1)
            r[name]=dict(delta=float(diff.mean()),interval=ci(diff))
        target,stale = d["replace_correct"],d["stale_correct"]
        counter = dict(key_switched=int((d["key_ok"] & d["key_have"]).sum()),key_pairs=int(d["key_have"].sum()),
            time_switched=int(d["time_ok"].sum()),time_pairs=d["time_ok"].size,clean_correct=int(correct.sum()),
            queries=correct.size,target_replaced=int((target & correct).sum()),stale_preserved=int((stale & correct).sum()),
            target_unconditional=int(target.sum()),stale_unconditional=int(stale.sum()))
        report.update(contrast=r,counterfactual=counter)
    return report


def passes(v):
    g=v["arms"]["graft"]
    c=v["counterfactual"]
    return (g["source"]>=.90 and g["accuracy"]>=.90 and v["contrast"]["permuted"]["interval"][0]>.10
        and c["key_switched"]>=.8*c["key_pairs"] and c["time_switched"]>=.8*c["time_pairs"]
        and c["target_replaced"]>=.9*c["clean_correct"] and c["stale_preserved"]>=.9*c["clean_correct"])


def main():
    if os.environ.get("STUDIO_ADDRESS_DEVICE") == "cpu":
        mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64 * 1024 * 1024)
    all_runs = {}
    for donor_seed, host_seed in ((101,503),(211,607)):
        mx.random.seed(donor_seed)
        donor = Reader(0)
        timings = {"donor":fit(donor,768,79121+donor_seed,f"{donor_seed}: donor")}
        frozen = donor.locator
        mx.random.seed(donor_seed+732)
        sham = Locator()
        models = {"donor":donor}
        for arm in ("graft","native","untrained"):
            mx.random.seed(host_seed)
            model=Reader(64)
            if arm!="native":
                model.locator = frozen if arm=="graft" else sham
                model.locator.freeze()
            timings[arm]=fit(model,128,42918+host_seed,f"{donor_seed}: {arm}")
            models[arm]=model
        models["permuted"]=models["graft"]
        report=dict(timing=timings,
            held_out=score(models,810731+donor_seed,True,f"paired-{donor_seed}.npz"),
            familiar=score(models,319721+donor_seed,False))
        all_runs[str(donor_seed)]=report
        atomic_json(f"seed-{donor_seed}.json",report)
    passed=all(passes(r["held_out"]) for r in all_runs.values())
    advantage=all(r["held_out"]["contrast"]["native"]["interval"][0]>.05 for r in all_runs.values())
    atomic_json("result.json",dict(decision="complete",location_transfer=passed,
        native_advantage=passed and advantage,hardware=mx.device_info()["device_name"],
        device=str(mx.default_device()),seeds=all_runs))


if __name__=="__main__":
    main()
