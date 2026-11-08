"""MENDEL MN-17 — the REFEREE low-dose retest at matched realized motion (queue item 25).

THE ANOMALY. REFEREE's low-dose IN-CORE arms showed task gains (+0.023 / +0.021) that went
unadjudicated on core-specificity, because the Haar control was recorded as having "under-dosed
0.28–0.31x and VOIDed". The charter budgets a GPU dose-cell job to re-run the same doses with
controls matched on REALIZED motion.

IT DOES NOT NEED THE GPU RUN, AND THE PREMISE IS INVERTED. Every REFEREE cell banks a
`motion_audit` — per-layer `tot`, `in_sub` and `in_core` displacement at 20 checkpoints — and an
`evals` block with per-item task scores. Read at those artifacts:

  * The "0.28–0.31x under-dose" is the Haar arm's motion measured IN THE CORE. That is not
    under-dosing; it is a Haar control correctly NOT moving the core, which is what it is for.
  * On the quantity a matched control must match — TOTAL realized motion — the Haar arm
    OVER-dosed: 19.97 against 1.86 for incore@1 (10.8x) and 13.34 for incore@4 (1.50x).

So the banked control was never under-dosed on the matched quantity; it moved more, not less.

THE A FORTIORI ARGUMENT. If a control that moves 1.5x–10.8x MORE, in a random subspace, produces
the SAME gain, then the gain is not core-specific — and the direction of the mismatch cannot
rescue core-specificity. A control moving less could only produce a smaller effect under any
monotone reading of dose, which is also not a rescue. The residual (an exactly motion-matched Haar
cell) is named as an optional confirmation rather than pretended to be necessary.

KILL, verbatim: "matched controls show the same gains → the low-dose gain is generic motion, not
core-specific, and the anomaly closes."

STATISTICS. Gains are computed against the SAME-SEED baseline cell, paired at the ITEM level over
the 400-item eval, so the seed's own difficulty draw cancels. The arm-vs-arm contrast is then a
paired difference of gains.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "MN-17"
R = MC.RESULTS_S3
BASELINES = {s: f"{R}/referee_mde_s{s}/out/baseline/metrics.json" for s in (0, 1, 2)}
CELLS = {
    ("incore", 1.0, 0): f"{R}/referee_ladder_L1/out/incore1_s0/metrics.json",
    ("incore", 1.0, 1): f"{R}/referee_ladder_L1/out/incore1_s1/metrics.json",
    ("incore", 1.0, 2): f"{R}/referee_ladder_L1/out/incore1_s2/metrics.json",
    ("incore", 4.0, 0): f"{R}/referee_ladder_L2/out/incore4_s0/metrics.json",
    ("incore", 4.0, 1): f"{R}/referee_ladder_L2/out/incore4_s1/metrics.json",
    ("incore", 4.0, 2): f"{R}/referee_ladder_L2/out/incore4_s2/metrics.json",
    ("haar", 16.0, 0): f"{R}/referee_ladder_L3/out/haar16_s0/metrics.json",
    ("haar", 16.0, 1): f"{R}/referee_ladder_L3/out/haar16_s1/metrics.json",
    ("haar", 16.0, 2): f"{R}/referee_ladder_L3/out/haar16_s2/metrics.json",
}
REFEREE_BANKED_GAINS = {1.0: 0.023, 4.0: 0.021}     # POINTERS; recomputed below, never quoted


def motion(cell: dict) -> dict:
    """Final realized displacement, split into the three channels the audit records.

    Summed in quadrature over the audited matrices, which is how a displacement over a set of
    matrices composes — summing the norms themselves would over-count by treating them as collinear.
    """
    last = cell["motion_audit"][-1]["audit"]
    assert last, "empty motion audit"
    out = {}
    for ch in ("tot", "in_sub", "in_core"):
        # The BASELINE arm has no target subspace, so its audit records `tot` alone. Reporting an
        # absent channel as 0.0 would make it look like a measured zero; it is recorded as absent.
        present = [v[ch] for v in last.values() if ch in v]
        out[ch] = float(np.sqrt(sum(x ** 2 for x in present))) if present else None
    assert out["tot"] is not None, f"motion audit has no `tot` channel: {sorted(next(iter(last.values())))}"
    out["channels_present"] = sorted(k for k in ("tot", "in_sub", "in_core") if out[k] is not None)
    out["step"] = int(cell["motion_audit"][-1]["step"])
    return out


def per_item(cell: dict, phase: str = "end") -> np.ndarray:
    v = np.asarray(cell["evals"][phase]["task_per_item"], dtype=np.float64)
    assert v.size > 50, f"{cell.get('cell_id')}: only {v.size} eval items"
    return v


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-17 — REFEREE at matched realized motion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn17", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn17"), out)

    base = {s: json.loads(man.require(f"baseline_s{s}", u).read_text())
            for s, u in BASELINES.items()}
    cells = {k: json.loads(man.require(f"{k[0]}{k[1]:g}_s{k[2]}", u).read_text())
             for k, u in CELLS.items()}
    n_items = {per_item(c).size for c in list(base.values()) + list(cells.values())}
    assert len(n_items) == 1, f"eval item counts differ across cells: {n_items}"
    n_items = n_items.pop()

    # ---- the motion audit: is the control under- or over-dosed, and on WHICH channel? ----
    mot = {k: motion(c) for k, c in cells.items()}
    mot_base = {s: motion(c) for s, c in base.items()}
    by_arm = {}
    for (arm, lam, _s), m in mot.items():
        by_arm.setdefault((arm, lam), {"tot": [], "in_sub": [], "in_core": []})
        for ch in ("tot", "in_sub", "in_core"):
            if m[ch] is not None:
                by_arm[(arm, lam)][ch].append(m[ch])
    mot_mean = {k: {ch: float(np.mean(v)) for ch, v in d.items() if v} for k, d in by_arm.items()}
    dose = {}
    for lam in (1.0, 4.0):
        dose[f"haar16_over_incore{lam:g}"] = {
            ch: mot_mean[("haar", 16.0)][ch] / mot_mean[("incore", lam)][ch]
            for ch in ("tot", "in_sub", "in_core")}
    for (arm, lam), m in sorted(mot_mean.items()):
        for ch in [c for c in ("tot", "in_sub", "in_core") if c in m]:
            at.add("referee_realized_motion", value=m[ch], item_id=f"{arm}@{lam:g}",
                   bank_id=ch, control_id=MC.STRUCTURED if arm == "haar" else MC.TREATMENT,
                   flags={"channel": ch, "n_seeds": 3})

    # ---- gains against the SAME-SEED baseline, paired at the item level ----
    gains, rows = {}, []
    for (arm, lam, s), c in sorted(cells.items()):
        g = per_item(c) - per_item(base[s])
        gains[(arm, lam, s)] = g
        rows.append({"arm": arm, "lam": lam, "seed": s, "gain_mean": float(g.mean()),
                     "arm_acc": float(per_item(c).mean()),
                     "base_acc": float(per_item(base[s]).mean()),
                     "total_motion": mot[(arm, lam, s)]["tot"],
                     "base_total_motion": mot_base[s]["tot"]})
    arm_gain = {}
    for (arm, lam) in sorted({(k[0], k[1]) for k in gains}):
        pooled = np.concatenate([gains[(arm, lam, s)] for s in (0, 1, 2)])
        ci = MC.paired_delta_ci(pooled, np.zeros_like(pooled), seed=a.seed)
        arm_gain[(arm, lam)] = ci
        at.add_ci("referee_task_gain", pooled, item_id=f"{arm}@{lam:g}",
                  control_id=MC.STRUCTURED if arm == "haar" else MC.TREATMENT, seed=a.seed,
                  corpus_id="textvqa",
                  flags={"n_seeds": 3, "n_items": n_items,
                         "total_motion": mot_mean[(arm, lam)]["tot"]})

    # ---- the chance floor: what a "gain" looks like when NOTHING was done differently ----
    # Two baseline seeds differenced against each other. This is the HAAR slot of §5's trinity and
    # it is the number that says whether a +0.02 gain is large: if seed-to-seed noise alone spans
    # that, no arm here has cleared anything.
    import itertools
    pairs = list(itertools.combinations(sorted(base), 2))
    # The comparable statistic is the SIGNED accuracy difference between two baseline seeds — the
    # same shape as a "gain", measured where nothing was done differently. Its SPREAD is the floor:
    # pooling the signed item differences across pairs would let opposite-signed pairs cancel and
    # report a fake-tight interval, so the per-pair means are kept and their range reported.
    pair_means = [float((per_item(base[i]) - per_item(base[j])).mean()) for i, j in pairs]
    seed_noise = np.concatenate([per_item(base[i]) - per_item(base[j]) for i, j in pairs])
    noise_ci = MC.paired_delta_ci(np.abs(seed_noise), np.zeros_like(seed_noise), seed=a.seed)
    at.add_ci("referee_task_gain", seed_noise, item_id="baseline_vs_baseline",
              control_id=MC.HAAR, seed=a.seed, corpus_id="textvqa",
              flags={"what": "two baseline seeds differenced — the seed-noise floor for a gain",
                     "n_pairs": len(pairs), "per_pair_mean_gain": pair_means,
                     "mean_abs": noise_ci["delta_mean"]})

    # ---- the KILL: does the control show the SAME gain? ----
    contrasts = {}
    for lam in (1.0, 4.0):
        a_in = np.concatenate([gains[("incore", lam, s)] for s in (0, 1, 2)])
        a_hr = np.concatenate([gains[("haar", 16.0, s)] for s in (0, 1, 2)])
        ci = MC.paired_delta_ci(a_in, a_hr, seed=a.seed)
        contrasts[f"incore{lam:g}_minus_haar16"] = ci
        at.add_ci("referee_task_gain", a_in - a_hr, item_id=f"incore{lam:g}_minus_haar16",
                  control_id=MC.SHUFFLE, seed=a.seed, corpus_id="textvqa",
                  flags={"rule": "KILL if the control shows the same gain (CI includes 0)"})

    same = all(not c["clears_zero"] and c["ci_lo"] < 0 < c["ci_hi"]
               for c in contrasts.values())
    repro = {f"incore@{lam:g}": {"ours": arm_gain[("incore", lam)]["delta_mean"],
                                 "referee_banked": REFEREE_BANKED_GAINS[lam]}
             for lam in (1.0, 4.0)}
    over = all(dose[k]["tot"] > 1.0 for k in dose)

    verdict = MP.check_verdict(ARM, "GENERIC-MOTION" if same else "GAIN-SURVIVES-MATCHING")
    gates.record("mn17_control_is_overdosed_not_underdosed", over,
                 dose_ratios=dose,
                 note=("the queue records the Haar control as under-dosed 0.28-0.31x; that is its "
                       "IN-CORE channel. On TOTAL realized motion — the quantity a matched control "
                       "must match — it moved MORE, not less"))
    gates.record("mn17_gains_exceed_baseline_seed_noise",
                 min(v["delta_mean"] for v in arm_gain.values()) > max(abs(x) for x in pair_means),
                 per_pair_baseline_gain=pair_means,
                 largest_baseline_pair_gain=float(max(abs(x) for x in pair_means)),
                 arm_gains={f"{k[0]}@{k[1]:g}": v["delta_mean"] for k, v in arm_gain.items()},
                 note=("RECORDED. If two BASELINE seeds differ by as much as an arm gains, no arm "
                       "here has cleared anything — this is prior to the core-specificity question"))
    gates.record("mn17_control_shows_the_same_gain", same,
                 contrasts={k: {"delta": v["delta_mean"], "ci": [v["ci_lo"], v["ci_hi"]]}
                            for k, v in contrasts.items()})
    at.check_control_trinity(["referee_task_gain"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "n_items": int(n_items), "n_seeds": 3,
        "realized_motion_mean": {f"{k[0]}@{k[1]:g}": v for k, v in mot_mean.items()},
        "baseline_motion": {f"s{s}": m for s, m in mot_base.items()},
        "dose_ratios_haar_over_incore": dose,
        "control_overdosed_on_total_motion": bool(over),
        "gain_by_arm": {f"{k[0]}@{k[1]:g}": v for k, v in arm_gain.items()},
        "seed_noise_floor": {"mean_abs_item_difference": noise_ci["delta_mean"],
                             "ci": [noise_ci["ci_lo"], noise_ci["ci_hi"]],
                             "n": noise_ci["n"],
                             "per_pair_signed_mean_gain": pair_means,
                             "spread": [float(min(pair_means)), float(max(pair_means))],
                             "largest_abs": float(max(abs(x) for x in pair_means)),
                             "what": ("signed accuracy difference between two BASELINE seeds — the "
                                      "same shape as a gain, where nothing was done differently")},
        "gains_exceed_seed_noise": bool(
            min(v["delta_mean"] for v in arm_gain.values())
            > max(abs(x) for x in pair_means)),
        "reproduction_of_referee_headline": repro,
        "kill_contrasts": contrasts, "control_shows_same_gain": bool(same),
        "cells": rows,
        "residual": ("an exactly motion-matched Haar cell (eta_scale ~0.67 to match incore@4, "
                     "~0.09 to match incore@1) would confirm directly; it is not required, because "
                     "an OVER-dosed control showing the same gain already rules out "
                     "core-specificity and a lower-dosed one could only show less"),
        "atlas": info})
    print("MN17_OK " + json.dumps({"verdict": verdict, "same_gain": bool(same),
                                   "overdosed": bool(over),
                                   "gains": {f"{k[0]}@{k[1]:g}": round(v["delta_mean"], 4)
                                             for k, v in arm_gain.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
