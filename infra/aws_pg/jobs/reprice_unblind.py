"""REPRICE unblind — fills the prereg §8 grid from banked cells (CPU, no GPU).

Pure aggregation (`aggregate`) is separated from file I/O so it is unit-tested
weight-free. Reads:
  * cap cells   : decoder=pg, arm=cap, n ∈ budgets (the recovery curve, both axes)
  * null cell   : decoder=pg, arm=cap, n=0  (matched random UNTRAINED adapter — the §5 null)
  * ceiling     : pg_before per-axis per-item (preflight baselines.json)
all on the SAME eval split, aligned by `item_hash`.

Emits, per axis: S_x(b), C_x (min budget reaching τ·ceiling), then the two headline
readouts with paired cluster-bootstrap CI:
  * C_cloze / C_MCQ          (recovery-cost ratio)
  * J at the MCQ-hits-τ rung (concordance, effect − random-adapter null)
and the §3 verdict (frozen rules; NO post-unblind rule swap).

    python -m infra.aws_pg.jobs.reprice_unblind --cells <dir> --baselines <preflight.json> \
        --tau 0.9 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def aggregate(cap_by_budget: dict, null_cell: dict, ceiling: dict, *, budgets: list,
              tau: float, n_boot: int = 10000, seed: int = 0) -> dict:
    """Pure §8 aggregation. **Alignment is by POSITION in the canonical (sorted-
    subject, row) order**, NOT by hash — the `[:64]` qa_item_hash has up to 895-way
    collisions (templated subjects e.g. moral_scenarios), so hash-keying would
    collapse items. Every cell + ceiling + null is the SAME split → identical item
    SEQUENCE; the hash sequence is asserted equal only as a reorder tripwire.

    cap_by_budget: {budget: {"hash":[...], "mcq":[0/1...], "cloze":[0/1...]}}
    null_cell:     {"hash":[...], "mcq":[...], "cloze":[...]}  (n=0 untrained adapter)
    ceiling:       {"hash":[...], "mcq":[...], "cloze":[...]}  (pg_before, split-filtered)
    budgets:       ascending list, == sorted(cap_by_budget keys)
    """
    import numpy as np
    from infra.pg import reprice_stats as RS

    assert budgets == sorted(budgets), "budgets must be ascending"
    assert set(budgets) == set(cap_by_budget), "budgets must match cap cells"

    ref = list(ceiling["hash"])
    assert ref, "empty ceiling sequence"
    for b in budgets:
        assert list(cap_by_budget[b]["hash"]) == ref, \
            f"budget {b} item SEQUENCE != ceiling (reorder/version drift) — HALT"
    assert list(null_cell["hash"]) == ref, "null item SEQUENCE != ceiling — HALT"

    # null_cell (n=0 untrained adapter) is BOTH the recovery floor AND the J null.
    ceil_mcq_pi = np.array(ceiling["mcq"], float); ceil_cloze_pi = np.array(ceiling["cloze"], float)
    floor_mcq_pi = np.array(null_cell["mcq"], float); floor_cloze_pi = np.array(null_cell["cloze"], float)
    ceil_mcq = float(ceil_mcq_pi.mean()); ceil_cloze = float(ceil_cloze_pi.mean())
    mcq_bb = np.array([cap_by_budget[b]["mcq"] for b in budgets], dtype=float).T   # (N,B)
    clz_bb = np.array([cap_by_budget[b]["cloze"] for b in budgets], dtype=float).T
    s_mcq = mcq_bb.mean(axis=0).tolist()
    s_cloze = clz_bb.mean(axis=0).tolist()
    maxb = len(budgets) - 1

    # ---- PRIMARY (pre-frozen): per-axis recovery FRACTION at the max budget ----
    rf_mcq = RS.bootstrap_recovery_fraction(mcq_bb[:, maxb], floor_mcq_pi, ceil_mcq_pi, n=n_boot, seed=seed)
    rf_cloze = RS.bootstrap_recovery_fraction(clz_bb[:, maxb], floor_cloze_pi, ceil_cloze_pi, n=n_boot, seed=seed)

    # ---- SECONDARY: recovery-cost (flagged degenerate when floor ≥ τ·ceiling) ----
    c_mcq = RS.recovery_cost(budgets, s_mcq, tau, ceil_mcq)
    c_cloze = RS.recovery_cost(budgets, s_cloze, tau, ceil_cloze)
    cx_degenerate_cloze = bool(float(floor_cloze_pi.mean()) >= tau * ceil_cloze)
    cloze_reaches_tau_at_mcq_rung = None
    if c_mcq["reached"]:
        cloze_reaches_tau_at_mcq_rung = bool(s_cloze[budgets.index(int(c_mcq["cost"]))] >= tau * ceil_cloze)
    ratio_ci = RS.bootstrap_cost_ratio(mcq_bb, clz_bb, budgets, tau, ceil_mcq, ceil_cloze,
                                       n=n_boot, seed=seed)

    # ---- J = P(cloze recovered | MCQ recovered) at the max budget, minus the n=0 null ----
    mcq_b, clz_b = mcq_bb[:, maxb], clz_bb[:, maxb]
    jci = RS.bootstrap_concordance(mcq_b, clz_b, floor_mcq_pi.astype(int), floor_cloze_pi.astype(int),
                                   n=n_boot, seed=seed)
    J = {"rung": budgets[maxb], "point": jci["point"], "lo": jci["lo"], "hi": jci["hi"],
         "raw": RS.concordance(mcq_b, clz_b)["J"]}

    verdict = _verdict(rf_mcq, rf_cloze, J)
    return {"n_items": len(ref), "tau": tau,
            "ceiling": {"mcq": ceil_mcq, "cloze": ceil_cloze},
            "floor": {"mcq": float(floor_mcq_pi.mean()), "cloze": float(floor_cloze_pi.mean())},
            "S_mcq": s_mcq, "S_cloze": s_cloze, "budgets": budgets,
            "recovery_fraction": {"mcq": rf_mcq, "cloze": rf_cloze},   # PRIMARY
            "C_MCQ": c_mcq, "C_cloze": c_cloze, "cost_ratio": ratio_ci,  # SECONDARY
            "cx_degenerate_cloze": cx_degenerate_cloze,
            "cloze_reaches_tau_at_mcq_rung": cloze_reaches_tau_at_mcq_rung,
            "J": J, "verdict": verdict}


def _verdict(rf_mcq, rf_cloze, J) -> dict:
    """§3 verdict on the PRE-FROZEN primary statistic (per-axis recovery fraction)
    + the J concordance. H_know: MCQ recovers AND cloze ALSO recovers (lockstep,
    its CI excludes 0 from below) AND J significantly above the n=0 null. H_head:
    MCQ recovers AND cloze does NOT (its recovery-fraction CI is entirely ≤ 0) AND
    J is not significantly above null. Else graded.

    (Replaces the recovery-COST plateau rule, which is degenerate here: the cloze
    floor already exceeds τ·ceiling because deletion barely touched content — see
    `cx_degenerate_cloze`. The recovery fraction was frozen as primary BEFORE
    confirmation, journal Step 4.)"""
    mcq_recovers = bool(rf_mcq["point"] > 0.5 and rf_mcq["lo"] > 0)
    cloze_recovers = bool(rf_cloze["lo"] > 0)
    cloze_no_recover = bool(rf_cloze["hi"] <= 0)
    j_lo = J.get("lo")
    j_above_null = bool(j_lo is not None and j_lo > 0)
    j_not_above_null = bool(j_lo is None or j_lo <= 0)

    if mcq_recovers and cloze_recovers and j_above_null:
        v = "H_know"
    elif mcq_recovers and cloze_no_recover and j_not_above_null:
        v = "H_head"
    else:
        v = "graded"
    return {"verdict": v, "mcq_recovers": mcq_recovers, "cloze_recovers": cloze_recovers,
            "cloze_no_recover": cloze_no_recover, "J_above_null": j_above_null,
            "rf_mcq": rf_mcq["point"], "rf_cloze": rf_cloze["point"]}


# --------------------------------------------------------------------------- #
# File I/O glue.
# --------------------------------------------------------------------------- #
def _cell_seq(cell):
    """Ordered {hash, mcq, cloze} sequences for one banked cell (split canonical order)."""
    return {"hash": cell["item_hash"], "mcq": cell["mcq"]["per_item"],
            "cloze": cell["cloze"]["per_item"]}


def load_and_aggregate(cells_dir, baselines_path, *, tau, split, rank=512,
                       budgets=(1000, 4000, 16000), n_boot=10000, seed=0):
    from infra.pg import w16_probes as W
    pred = W.is_grid_subset if split == "bucket" else W.is_confirmation

    cells = {}
    for p in sorted(pathlib.Path(cells_dir).rglob("cell_*.json")):   # pipeline writes per-step subdirs
        c = json.loads(p.read_text())
        if c["eval_split"] != split:
            continue
        cells[(c["decoder"], c["arm"], c["rank"], c["n"])] = c

    cap_by_budget = {b: _cell_seq(cells[("pg", "cap", rank, b)]) for b in budgets}
    null_cell = _cell_seq(cells[("pg", "cap", rank, 0)])

    base = json.loads(pathlib.Path(baselines_path).read_text())
    cb = base["baselines_blankimg"]["ceiling_seq"]    # full-canonical ordered {hash,mcq,cloze}
    keep = [i for i, h in enumerate(cb["hash"]) if pred(h)]    # filter to the split, PRESERVE order
    ceiling = {"hash": [cb["hash"][i] for i in keep],
               "mcq": [cb["mcq"][i] for i in keep],
               "cloze": [cb["cloze"][i] for i in keep]}
    return aggregate(cap_by_budget, null_cell, ceiling, budgets=list(budgets),
                     tau=tau, n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--split", default="confirmation", choices=["bucket", "confirmation"])
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = load_and_aggregate(args.cells, args.baselines, tau=args.tau, split=args.split,
                             rank=args.rank)
    (out / "reprice_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    rf = res["recovery_fraction"]
    print(f"[REPRICE-UNBLIND] verdict={res['verdict']['verdict']}  (n={res['n_items']})\n"
          f"  recovery fraction @{res['budgets'][-1]}:  MCQ {rf['mcq']['point']:+.3f} "
          f"[{rf['mcq']['lo']:+.3f},{rf['mcq']['hi']:+.3f}]   "
          f"cloze {rf['cloze']['point']:+.3f} [{rf['cloze']['lo']:+.3f},{rf['cloze']['hi']:+.3f}]\n"
          f"  J−null={res['J']['point']:+.4f} [{res['J']['lo']:+.4f},{res['J']['hi']:+.4f}]   "
          f"cx_degenerate_cloze={res['cx_degenerate_cloze']}")


if __name__ == "__main__":
    main()
