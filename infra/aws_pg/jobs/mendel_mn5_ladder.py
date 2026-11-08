"""MENDEL MN-5 — the survival-predictor ladder against the PLENUM matrix (queue item 17, tiers 0/0.5/1).

THE QUESTION. PLENUM sealed four survival predictors before measuring and none of them worked.
Does a WEIGHT-SPACE feature do better — and does each rung of the ladder beat the rung below, OUT
OF SAMPLE?

THE FLOOR IS NOT THE SEALED PREDICTORS (freeze A12, from G3). Over PLENUM's 112 adjudicated cells
the outcome mix is retained 94 / consumed 11 / partial 7, so a constant "always retained"
predictor scores 0.8393 while the best sealed predictor (H_fragility) scores 0.4821. All four
sealed predictors are BELOW the trivial baseline — they over-call "consumed" 56-91 times out of
112 when 11 cells were consumed. So the ladder is read against the MAJORITY-CLASS rate, recomputed
on this arm's own stratum. Comparing a tier only against the sealed predictors would let a tier
scoring 0.6 read as a win while being far worse than a constant.

THE STRATUM. The 24 official `ft-*` fine-tunes only. PLENUM's other 26 units are film checkpoints,
stitched decoders and lineage members — pooling them is exactly the "extrapolation outside the
intervention family" risk the spec names, and it would also make the tier-0 feature incomparable
(a checkpoint's dW is measured from a different base than a fine-tune's).

THE SPLIT IS BY UNIT, not by cell. A fine-tune contributes up to four faculty cells and they all
share one dW; splitting by cell would put the same weight-delta on both sides of the firewall and
every tier would leak. sha1 content hash of the unit id, so the split is deterministic and
independent of any measured value.

THE OUTCOME is binary — retained vs not-retained (consumed union partial) — matching the two-class
convention the sealed predictors were scored on, so the floor and the tiers are commensurable.

THE LADDER
    tier -1   majority class from the TRAIN half                       the floor (freeze A12)
    tier 0    total ||dW||_F per fine-tune, banked as dW_fro2_by_ft    one threshold fit on train
    tier 0.5  QUORUM curvature footprint                               BLOCKED-WITH-UNLOCK
    tier 1    realized-motion / v-hat statistics                       BLOCKED-WITH-UNLOCK

Tiers 0.5 and 1 are blocked on FEATURES, not on compute: `quorum_verdict` banks capability-level
quantities (rho_break, per_capability, redundancy) and no per-fine-tune footprint, and no
realized-motion series exists for the official fine-tunes. Both unlocks are named in the metrics.

KILL per tier: fails to beat the tier below out-of-sample.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mendel_core as MC
from infra.pg import mendel_critic as CR
from infra.pg import mendel_program as MP

ARM = "MN-5"
PLENUM = f"{MC.RESULTS_S3}/plenum_agg_final2/out"
HUMBOLDT_A1 = f"s3://{MC.BUCKET}/humboldt/humboldt_a1"
LABELLED_FACULTIES = ("caption", "ocr", "reading", "vqa")     # grounding is 49/49 unadjudicated
RETAINED = "retained"


def unit_split(unit: str, *, salt: str = "mendel-mn5") -> str:
    """Deterministic 50/50 train/test assignment from a sha1 of the unit id.

    Content-hash rather than rng-based so the split is reproducible from the unit name alone and
    cannot drift with a library's PRNG. Salted with the arm name so MN-5's split is not accidentally
    the same partition another arm used on the same units.
    """
    h = hashlib.sha1(f"{salt}|{unit}".encode()).hexdigest()
    return "train" if int(h[:8], 16) % 2 == 0 else "test"


def majority_rule(train_labels):
    """Tier -1: predict the train half's most common class, always."""
    c = collections.Counter(train_labels)
    assert c, "majority rule needs a non-empty training set"
    return c.most_common(1)[0][0]


def threshold_rule(train_feature, train_labels):
    """Tier 0: ONE threshold on ONE feature, chosen on train only.

    Every candidate split point is the midpoint between two adjacent observed values, and both
    polarities are tried; the best train accuracy wins, ties broken toward the lower threshold so
    the choice is deterministic. Deliberately the simplest possible learner: with ~12 training
    units anything richer would fit the split rather than the phenomenon.
    """
    x = np.asarray(train_feature, dtype=np.float64)
    y = np.asarray(train_labels)
    assert x.size == y.size and x.size >= 2, "threshold rule needs >=2 training units"
    classes = sorted(set(y.tolist()))
    if len(classes) == 1:
        return {"threshold": float("inf"), "polarity": 1, "high_class": classes[0],
                "low_class": classes[0], "train_acc": 1.0, "degenerate": True}
    xs = np.unique(x)
    cands = [(xs[i] + xs[i + 1]) / 2.0 for i in range(xs.size - 1)] or [float(xs[0])]
    best = None
    for thr in cands:
        for hi, lo in ((classes[0], classes[1]), (classes[1], classes[0])):
            pred = np.where(x >= thr, hi, lo)
            acc = float((pred == y).mean())
            key = (-acc, thr)
            if best is None or key < best[0]:
                best = (key, {"threshold": float(thr), "high_class": hi, "low_class": lo,
                              "train_acc": acc, "degenerate": False})
    return best[1]


def apply_threshold(rule, feature):
    x = np.asarray(feature, dtype=np.float64)
    if rule["degenerate"]:
        return np.full(x.shape, rule["high_class"], dtype=object)
    return np.where(x >= rule["threshold"], rule["high_class"], rule["low_class"])


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-5 — the survival-predictor ladder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn5", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn5"), out)

    # ---- G3 gate ----
    g3 = json.loads((pathlib.Path(a.out).parent / "g3_metrics.json").read_text()) \
        if (pathlib.Path(a.out).parent / "g3_metrics.json").exists() else None
    plenum = json.loads(man.require("plenum_metrics", f"{PLENUM}/metrics.json").read_text())
    a1_atlas = man.require("humboldt_a1_atlas", f"{HUMBOLDT_A1}/atlas.jsonl")
    a1_summary = json.loads(man.require("humboldt_a1_summary",
                                        f"{HUMBOLDT_A1}/a1_summary.json").read_text())
    repos = a1_summary["manifest"]["cloud_repos"]
    n_ft = len(repos)

    # ---- tier 0's feature: total write energy per fine-tune, summed over all 126 cells ----
    rows = [json.loads(x) for x in a1_atlas.read_text().splitlines() if x.strip()]
    fro_rows = [r for r in rows if r["statistic"] == "dW_fro2_by_ft"]
    assert len(fro_rows) == 126, f"expected 126 dW_fro2_by_ft rows, got {len(fro_rows)}"
    vec_dir = man.require("humboldt_a1_vectors", f"{HUMBOLDT_A1}/vectors", recursive=True)
    total_fro2 = np.zeros(n_ft, dtype=np.float64)
    for r in fro_rows:
        v = np.load(vec_dir / pathlib.Path(r["vector_ref"]).name)
        assert v.shape == (n_ft,), f"{r['vector_ref']}: shape {v.shape} != ({n_ft},)"
        total_fro2 += v
    dw_norm = np.sqrt(total_fro2)
    # `ft-ai2d` <-> `google/paligemma-3b-ft-ai2d-224`
    unit_of = {r: "ft-" + r.split("paligemma-3b-ft-")[1].rsplit("-224", 1)[0] for r in repos}
    feature = {unit_of[r]: float(dw_norm[i]) for i, r in enumerate(repos)}

    # ---- the labelled cells on this stratum ----
    cells = [c for c in plenum["cells"]
             if c.get("faculty") in LABELLED_FACULTIES and c.get("label") is not None
             and str(c.get("unit", "")).startswith("ft-")]
    assert cells, "no labelled ft-* cells in the PLENUM matrix"
    missing = sorted({c["unit"] for c in cells} - set(feature))
    assert not missing, f"units with no banked dW: {missing}"

    data = [{"unit": c["unit"], "faculty": c["faculty"],
             "y": RETAINED if c["label"] == RETAINED else "not_retained",
             "x": feature[c["unit"]], "split": unit_split(c["unit"])} for c in cells]
    tr = [d for d in data if d["split"] == "train"]
    te = [d for d in data if d["split"] == "test"]
    assert tr and te, f"degenerate split: {len(tr)} train / {len(te)} test"

    # ---- tier -1: the majority-class floor, refit on THIS stratum's train half ----
    maj = majority_rule([d["y"] for d in tr])
    acc_floor = float(np.mean([d["y"] == maj for d in te]))

    # ---- tier 0: one threshold on ||dW|| ----
    rule = threshold_rule([d["x"] for d in tr], [d["y"] for d in tr])
    pred0 = apply_threshold(rule, [d["x"] for d in te])
    acc_t0 = float(np.mean(pred0 == np.array([d["y"] for d in te], dtype=object)))

    # A threshold on 12 units is a coarse use of the feature. The rank correlation between ||dW||
    # and a unit's survival RATE uses the same feature without discretising, so a null there is a
    # much stronger statement than a null on the threshold alone.
    by_unit = collections.defaultdict(list)
    for d in data:
        by_unit[d["unit"]].append(d["y"] == RETAINED)
    units = sorted(by_unit)
    surv = [float(np.mean(by_unit[u])) for u in units]
    xs = [feature[u] for u in units]
    rho = CR.spearman(xs, surv)
    perm = MC.permutation_pvalue(
        surv, ["hi" if feature[u] >= float(np.median(xs)) else "lo" for u in units],
        seed=a.seed)

    for name, acc, cid in (("tier_-1_majority", acc_floor, MC.STRUCTURED),
                           ("tier_0_dW_norm", acc_t0, MC.TREATMENT)):
        at.add("survival_ladder_test_accuracy", value=acc, n=len(te), item_id=name,
               control_id=cid, corpus_id="plenum_ft_stratum",
               flags={"n_train": len(tr), "n_test": len(te), "majority_class": maj,
                      "rule": rule if name.startswith("tier_0") else None})
    at.add("survival_dW_spearman", value=rho, n=len(units), control_id=MC.TREATMENT,
           corpus_id="plenum_ft_stratum",
           flags={"p_permutation": perm["p_value"], "method": perm["method"],
                  "what": "rank correlation of ||dW|| with the unit's faculty-survival rate"})

    beats = acc_t0 > acc_floor
    blocked = {
        "tier_0.5_quorum_footprint": (
            "BLOCKED-WITH-UNLOCK — quorum_verdict banks capability-level quantities "
            "(rho_break, per_capability, redundancy_ratio) and no per-fine-tune curvature "
            "footprint. UNLOCK: bank a per-unit footprint over the 24 ft-* deltas."),
        "tier_1_realized_motion": (
            "BLOCKED-WITH-UNLOCK — no realized-motion / v-hat series exists for the official "
            "fine-tunes (they are published endpoints, not trajectories). UNLOCK: MN-1's FT-family "
            "stage produces per-fine-tune column write-energy profiles, which are a motion proxy "
            "on exactly these 24 units."),
    }
    # Only tier 0 is runnable, so exactly two frozen tokens are reachable: NO-TIER-BEATS-DW if the
    # one runnable rung fails its own floor, and BLOCKED-WITH-UNLOCK if it clears the floor but the
    # rungs above it cannot be evaluated. TIER1-WINS / TIER0.5-WINS are unreachable in this build
    # and are NOT written, rather than being reached by a path that does not test them.
    verdict = MP.check_verdict(ARM, "NO-TIER-BEATS-DW" if not beats else "BLOCKED-WITH-UNLOCK")

    gates.record("mn5_tier0_beats_majority_class", beats, tier0=acc_t0, floor=acc_floor,
                 majority_class=maj, n_test=len(te),
                 note=("freeze A12: the floor is the majority-class rate, NOT the sealed "
                       "predictors' 0.4821 — all four of those score below a constant"))
    gates.record("mn5_ladder_complete", False, blocked=sorted(blocked),
                 note="tiers 0.5 and 1 are blocked on FEATURES, not compute; unlocks named")
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "stratum": "the 24 official ft-* fine-tunes", "faculties": list(LABELLED_FACULTIES),
        "excluded_faculty": {"grounding": "all 49 cells unadjudicated (label=None) in PLENUM"},
        "n_cells": len(data), "n_train": len(tr), "n_test": len(te),
        "n_units": len(units), "split": "sha1 content hash of the unit id, salted 'mendel-mn5'",
        "outcome": "binary: retained vs not-retained (consumed union partial)",
        "tier_-1_majority": {"class": maj, "test_accuracy": acc_floor},
        "tier_0_dW_norm": {"rule": rule, "test_accuracy": acc_t0, "beats_floor": bool(beats)},
        "dW_spearman": {"rho": rho, "p_permutation": perm["p_value"],
                        "method": perm["method"], "n_units": len(units)},
        "blocked_tiers": blocked,
        "g3_floor_reference": (g3 or {}).get("majority_class_null"),
        "atlas": info})
    print("MN5_OK " + json.dumps({"verdict": verdict, "floor": round(acc_floor, 4),
                                  "tier0": round(acc_t0, 4), "rho": round(rho, 4),
                                  "p": round(perm["p_value"], 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
