"""Verify room-removal, transplant and growth calculations from the Studio records."""

import json
from math import isclose
from pathlib import Path
from statistics import mean


DOMAINS = ("english", "code", "welsh")
REVISION = "4982b3395c2cace1c4295be8e0a47d93b099218a"
MAIN_ARMS = {"dense": "dense", "walled_seed_1": "walled", "walled_seed_2": "walled_s2",
             "topographic": "topographic", "routed": "routed"}


def verify_removal(removal):
    highest = 0.0
    selected = set()
    for room in removal["rooms"]:
        base = removal["base"]
        increases = {d: max(0, room["loss"][d] - base[d]) for d in DOMAINS}
        assert increases == room["increase"]
        largest = max(DOMAINS, key=increases.get)
        other = mean(v for d, v in increases.items() if d != largest)
        top = increases[largest]
        score = (top - other) / max(top + other, 1e-9)
        assert isclose(score, room["selectivity"], abs_tol=1e-12)
        assert largest == room["domain"]
        meaningful = top >= .01 * base[largest]
        assert meaningful == room["meaningful"]
        if meaningful:
            highest = max(highest, score)
            if score >= .5:
                selected.add(largest)
    assert isclose(highest, removal["max_selectivity"], abs_tol=1e-12)
    assert selected == set(removal["domains_with_selective_room"])
    return highest, selected


def main():
    out = json.loads(Path(__file__).with_name("evidence.json").read_text())
    for arm in out.values():
        receipt = arm["receipt"]
        assert receipt["revision"] == REVISION and receipt["status"] == "done" and receipt["exit"] == 0
    screen = out["screen"]["result"]
    assert screen["decision"] == "screen_no_selectivity"
    assert max(screen["arms"][name]["max_selectivity"]
               for name in ("walled", "topographic-0.01", "topographic-0.05")) <= .5
    surgery = out["surgery"]["result"]
    results = {}
    for arm, label in MAIN_ARMS.items():
        r = out[arm]["result"]
        assert r["steps"] == 6510 and r["tokens"] == r["steps"] * 120 * 512
        loss = mean(r["val"][d] for d in DOMAINS)
        assert isclose(loss, r["train_mean_val"], abs_tol=1e-12)
        assert r["removal"] == surgery["removal"][label]
        score, domains = verify_removal(r["removal"])
        assert not domains
        results[arm] = (loss, score)
        print(f"{arm}: mean loss {loss:.4f}, highest selectivity {score:.4f}")

    cost = results["walled_seed_1"][0] / results["dense"][0] - 1
    assert isclose(cost, surgery["walls_cost"], abs_tol=1e-12)
    t = surgery["transplant"]
    chosen = max(out["walled_seed_1"]["result"]["removal"]["rooms"],
                 key=lambda row: (row["meaningful"], row["selectivity"]))
    assert (t["room"], t["domain"]) == (chosen["room"], chosen["domain"])
    d = t["domain"]
    removed = t["removed_before_refit"][d]
    damage = removed - t["intact"][d]
    assert damage > 0
    recovery = (removed - t["transplant_after_refit"][d]) / damage
    control = (removed - t["removed_after_refit"][d]) / damage
    assert isclose(recovery, t["recovery"]) and isclose(control, t["control_recovery"])
    assert recovery < .9 and recovery - control < .25 and surgery["decision"] == "walls_null"

    for name in ("grow_sealed", "grow_doors", "full_finetune"):
        g = surgery["growth"][name]
        assert isclose(g["icelandic_gain"], g["before"]["icelandic"] - g["after"]["icelandic"])
        assert g["old_domain_change"] == {d: g["after"][d] - g["before"][d] for d in DOMAINS}
    print(f"Verified the wall null, transplant recovery {recovery:.2%}, and door-only recovery {control:.2%}.")


if __name__ == "__main__":
    main()
