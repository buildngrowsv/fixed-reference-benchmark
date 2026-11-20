"""Transplant the localized memory head and train only the following host block."""
from functools import partial
import sys
import time

import numpy as np
from mlx.utils import tree_flatten
from run import (RoutedModel, OUT, RUNS, batch, evaluate, mx, nn, optim,
                 optimizer, restore, scores, snapshot, uncertainty, write)

HOST = RUNS / "reference-transformer-pilot/writable.safetensors"
DONOR = RUNS / "register-addressing/rotated.safetensors"
WIDTH, HEAD = 384, 64


def load_model(path):
    model = RoutedModel()
    assert restore(model, optimizer(), path) == 512
    return model


def install_head(target, source):
    block, expert = target.blocks[3], source.blocks[3]
    new_qkv = []
    for host, donor in zip(mx.split(block.qkv.weight, 3), mx.split(expert.qkv.weight, 3)):
        new_qkv.append(mx.concatenate([donor[:HEAD], host[HEAD:]]))
    block.qkv.weight = mx.concatenate(new_qkv)
    block.out.weight = mx.concatenate([expert.out.weight[:, :HEAD], block.out.weight[:, HEAD:]], axis=1)


def freeze_except_receiver(model):
    model.freeze()
    model.blocks[4].unfreeze()
    params = tree_flatten(model.trainable_parameters())
    assert all(k.startswith("blocks.4.") for k, _ in params)
    count = sum(v.size for _, v in params)
    assert count == 12 * WIDTH**2 + 2 * WIDTH
    return count


def smoke():
    mx.set_default_device(mx.cpu)
    host, donor = load_model(HOST), load_model(DONOR)
    x = mx.array(batch([48371, 0], 2)[0])
    before = np.array(host(x, wrong=True))
    originals = [np.array(host.blocks[3].qkv.weight), np.array(host.blocks[3].out.weight)]
    backup = load_model(HOST)
    install_head(host, donor)
    q = np.array(host.blocks[3].qkv.weight).reshape(3, WIDTH, WIDTH)
    np.testing.assert_array_equal(q[:, HEAD:], originals[0].reshape(3, WIDTH, WIDTH)[:, HEAD:])
    np.testing.assert_array_equal(q[:, :HEAD], np.array(donor.blocks[3].qkv.weight).reshape(3, WIDTH, WIDTH)[:, :HEAD])
    np.testing.assert_array_equal(np.array(host.blocks[3].out.weight)[:, HEAD:], originals[1][:, HEAD:])
    np.testing.assert_array_equal(np.array(host.blocks[3].out.weight)[:, :HEAD], np.array(donor.blocks[3].out.weight)[:, :HEAD])
    assert np.any(np.array(host(x, wrong=True)) != before)
    install_head(host, backup)
    np.testing.assert_array_equal(np.array(host(x, wrong=True)), before)
    print("Verified head identity, reversibility and trainable weights", freeze_except_receiver(host))


def main():
    if sys.argv[1:] == ["smoke"]:
        smoke()
        return
    gate = __import__("json").loads((RUNS/"displaced-register-interchange/result.json").read_text())
    if not gate["supported"]:
        write("result.json", dict(decision="stop", reason="donor intervention missed its threshold"))
        return
    donor, native = load_model(DONOR), load_model(HOST)
    counts = {"donor_full": evaluate(donor, "rotated", 482731)}
    for arm in ("native", "graft"):
        model = load_model(HOST)
        if arm == "graft":
            install_head(model, donor)
        opt = optimizer()
        count = freeze_except_receiver(model)
        path = OUT / (arm + ".safetensors")
        start = restore(model, opt, path) if path.exists() else 0
        def objective(m, x, y, op):
            logits = m(x, wrong=True)
            loss = nn.losses.cross_entropy(logits, y, reduction="none")
            use = op >= 2
            return (loss * use).sum() / mx.maximum(use.sum(), 1)
        grad = nn.value_and_grad(model, objective)
        state = [model.state, opt.state]
        @partial(mx.compile, inputs=state, outputs=state)
        def update(x, y, op):
            value, grads = grad(model, x, y, op)
            grads, _ = optim.clip_grad_norm(grads, 1.)
            opt.update(model, grads)
            return value
        since = time.monotonic()
        for t in range(start + 1, 1025):
            ids, targets, operations, _, _ = batch([717139, t])
            loss = update(mx.array(ids), mx.array(targets), mx.array(operations))
            mx.eval(loss, state)
            if t % 64 == 0:
                if not np.isfinite(loss.item()):
                    raise ValueError("nonfinite training loss")
                snapshot(model, opt, t, path)
                write("progress.json", dict(arm=arm, step=t, target=1024, loss=loss.item(), time=time.time(),
                      tokens_per_second=(t-start)*8192/(time.monotonic()-since)))
        counts[arm] = evaluate(model, "rotated", 482731)
        if arm == "graft":
            install_head(model, native)
            counts["donor_removed"] = evaluate(model, "rotated", 482731)
    np.savez_compressed(OUT/"test.npz", **counts)
    accuracy = {k:scores(v) for k,v in counts.items()}
    effects = {}
    for k in ("native", "donor_removed"):
        np.testing.assert_array_equal(counts[k][:,[0,2,3,5]], counts["graft"][:,[0,2,3,5]])
        difference = float(np.mean(accuracy["graft"]) - np.mean(accuracy[k]))
        effects[k] = dict(delta=difference, interval=uncertainty(counts["graft"], counts[k]))
    passed = all(v["delta"] >= .05 and v["interval"][0] > 0 for v in effects.values())
    write("result.json", dict(decision="replicate" if passed else "stop", accuracy=accuracy,
          trainable=count, transplanted=4*WIDTH*HEAD, programs=len(counts["graft"]), effects=effects))


if __name__ == "__main__":
    main()
