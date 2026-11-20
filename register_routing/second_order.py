"""A preregistered second paired fit on fresh register programs."""
from functools import partial
import time
import numpy as np
from unpredictable import random_mask, token_mask
from run import RoutedModel, OUT, batch, mx, nn, optim, optimizer, restore, routing, snapshot, scores, uncertainty, write

TRAIN_SEED = 211
TEST_SEED = 816239

def edges(ids, arm, salt):
    if arm == "random":
        return random_mask(ids, salt)
    events = ids.reshape(len(ids), 64, 4)
    ops, keys = events[:, :, 0], events[:, :, 1]
    query, position = np.arange(64)[:, None], np.arange(64)[None, :]
    source = (position - 1) % np.maximum(query, 1)
    ok = (position < query) & (ops[:, source] < 2) & (keys[:, source] == keys[:, :, None])
    return mx.array(ok)

def fit(arm):
    mx.random.seed(TRAIN_SEED)
    model, opt = RoutedModel(), optimizer()
    path = OUT / (arm + ".safetensors")
    start = restore(model, opt, path) if path.exists() else 0
    def objective(m, tokens, allowed, labels, ops):
        logits = m(tokens, mask=token_mask(tokens, allowed))
        loss = nn.losses.cross_entropy(logits, labels, reduction="none")
        queries = ops >= 2
        return (loss * queries).sum() / mx.maximum(queries.sum(), 1)
    value_grad = nn.value_and_grad(model, objective)
    state = [model.state, opt.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def step(ids, allowed, y, op):
        loss, grad = value_grad(model, ids, allowed, y, op)
        grad, _ = optim.clip_grad_norm(grad, 1.)
        opt.update(model, grad)
        return loss
    then = time.monotonic()
    for t in range(start + 1, 513):
        ids, y, op, _, _ = batch([TRAIN_SEED, 27491, t])
        allowed = edges(ids, arm, [7949, TRAIN_SEED, t])
        if t == 1:
            check_mask(ids, allowed, arm)
        loss = step(mx.array(ids), allowed, mx.array(y), mx.array(op))
        mx.eval(loss, state)
        if t % 64 == 0:
            if not np.isfinite(loss.item()):
                raise ValueError("nonfinite training loss")
            snapshot(model, opt, t, path)
            write("progress.json", dict(arm=arm, step=t, target=512, time=time.time(),
                  tokens_per_second=(t-start)*8192/(time.monotonic()-then), loss=loss.item()))
    return model

def check_mask(ids, allowed, arm):
    x = mx.array(ids)
    a = np.array(token_mask(x, allowed))
    ref = np.array(routing(x, True))
    np.testing.assert_array_equal(a.sum(-1), ref.sum(-1))
    assert not np.triu(a, 1).any()
    if arm == "stable":
        np.testing.assert_array_equal(a, ref)

def main():
    models = {name: fit(name) for name in ("stable", "random")}
    counts = {name: [] for name in models}
    for j in range(64):
        ids, y, op, changed, guess = batch([TEST_SEED, j])
        x = mx.array(ids)
        for arm, model in models.items():
            mask = token_mask(x, edges(ids, arm, [56711, TEST_SEED, j]))
            hit = np.array(model(x, mask=mask).argmax(-1)) == y
            row = []
            for kind in (2,3):
                use = changed & (op == kind)
                row.extend((use.sum(1), (hit & use).sum(1), ((guess == y) & use).sum(1)))
            counts[arm].append(np.stack(row, axis=1))
    arrays = {k: np.concatenate(v) for k,v in counts.items()}
    np.testing.assert_array_equal(arrays["stable"][:,[0,2,3,5]],arrays["random"][:,[0,2,3,5]])
    np.savez_compressed(OUT/"test.npz", **arrays)
    accuracy = {k:scores(v) for k,v in arrays.items()}
    delta = float(np.mean(accuracy["stable"])-np.mean(accuracy["random"]))
    ci = uncertainty(arrays["stable"],arrays["random"])
    write("result.json",dict(decision="complete",train_seed=TRAIN_SEED,test_seed=TEST_SEED,
          programs=len(arrays["stable"]),accuracy=accuracy,stable_minus_random=delta,interval=ci,
          replicated=bool(delta >= .10 and ci[0] > 0)))
if __name__=="__main__":main()
