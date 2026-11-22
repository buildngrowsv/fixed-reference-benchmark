"""prog-e3-screen: five small runs that decide whether walls or wiring cost make rooms.

Arms: dense, walled, topographic at two penalty strengths, routed. Each trains on the same
mixture order. The job picks the topographic strength for the main runs and applies the
protocol's stop rule (selectivity index above 0.5 for walls or the topographic penalty).
"""
from __future__ import annotations

import argparse

from common.job import Job
from e3 import configs
from e3.arm import run_arm

LAMBDAS = (0.01, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tokens = 20 * configs.TOKENS_PER_STEP
    job = Job("prog-e3-screen" + ("-smoke" if a.smoke else ""), vars(a))
    runs = [("dense", "dense", 0.0), ("walled", "walled", 0.0)]
    runs += [(f"topographic-{lam}", "topographic", lam) for lam in LAMBDAS] + [("routed", "routed", 0.0)]
    summaries = {}
    for name, arm, lam in runs:
        summaries[name] = run_arm(job, scale="screen", arm=arm, seed=1, tokens=a.tokens, out=job.dir / name, lam=lam,
                                  eval_every=10 if a.smoke else None, val_windows=8 if a.smoke else 48,
                                  label=f"{name} screen model")
        s = summaries[name]
        job.log(name, "mean val", round(s["train_mean_val"], 4), "max selectivity", round(s["removal"]["max_selectivity"], 3))

    topo = {lam: summaries[f"topographic-{lam}"] for lam in LAMBDAS}
    selective = [lam for lam, s in topo.items() if s["removal"]["max_selectivity"] > 0.5]
    if selective:
        chosen = min(selective, key=lambda lam: topo[lam]["train_mean_val"])
    else:
        chosen = max(LAMBDAS, key=lambda lam: topo[lam]["removal"]["max_selectivity"])
    walled_sel = summaries["walled"]["removal"]["max_selectivity"]
    topo_sel = max(s["removal"]["max_selectivity"] for s in topo.values())
    decision = "screen_selective" if max(walled_sel, topo_sel) > 0.5 else "screen_no_selectivity"
    dense = summaries["dense"]["train_mean_val"]
    table = {k: dict(mean_val=s["train_mean_val"], val=s["val"], cost_vs_dense=s["train_mean_val"] / dense - 1,
                     max_selectivity=s["removal"]["max_selectivity"],
                     selective_domains=s["removal"]["domains_with_selective_room"],
                     rooms=[dict(room=r["room"], domain=r["domain"], selectivity=r["selectivity"], ratio=r["ratio"])
                            for r in s["removal"]["rooms"]],
                     tokens_per_second=s["tokens_per_second"], params=s["params"]) for k, s in summaries.items()}
    job.finish(decision, topo_lambda=chosen, walled_max_selectivity=walled_sel, topographic_max_selectivity=topo_sel,
               arms=table)


if __name__ == "__main__":
    main()
