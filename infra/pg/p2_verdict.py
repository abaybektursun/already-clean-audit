# P2 — cross-seed verdict reader (LOCAL, JSON-only; no model compute).
"""Read the seed-1337 Q1 battery JSON + the P2 per-seed JSONs and render the
≥3-seed replication verdict for the 2 causal hits (RefCOCO + COCO-Caption, ablate).

For each (task, seed): the per-seed beats-random paired |Δ| bootstrap CI (significant
iff CI95 lower bound > 0 AND target loses more than random) and the target Spearman ρ
vs dose (monotone iff |ρ|>=0.9) — the SAME pre-set gate W5/Q1 used. Then the CROSS-SEED
picture per task: how many of the N seeds pass the gate, the across-seed mean ± spread
of (target_minus_random effect, beats-random Δ, ρ), and BH-FDR (q=0.05) over the
per-(task,seed) beats-random family. Reuses w5_verdict._bh_fdr / _one_sided_p_from_ci
verbatim (same statistics as W5/Q1).

A task's hit is FIRM iff it passes the gate (monotone AND beats-random, sign preserved)
on ALL N seeds. If it passes at <N seeds, flips the target-minus-random sign, or shows
huge across-seed variance → the hit is SINGLE-SEED / NOT REPLICATED.

Both schemas are accepted: the Q1 battery JSON (5 tasks, key '<task>|all|ablate' in
prereg_analysis) and the P2 JSON (2 tasks, same key shape) — we read only the two
ablate cells from each, keyed by the file's `seed`.

Pure: argv = paths to the seed JSONs (Q1 battery + P2 per-seed). No side effects.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from infra.pg.w5_verdict import _bh_fdr, _one_sided_p_from_ci, fmt_curve

P2_TASKS = ("refcoco", "coco_captions")
MODE = "ablate"


def _rank_avg(a):
    """Average ranks with proper TIE handling (1-based), no scipy dep."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho_tiecorrect(x, y) -> float:
    """Tie-correct Spearman ρ (average ranks). The W5/Q1 in-harness spearman_rho uses
    argsort(argsort(...)) which breaks ties by index order — this INFLATES |ρ| exactly
    at the monotonicity gate when a dose curve has tied values (e.g. RefCOCO seed-0
    target curve [0.612, 0.612, 0.600, 0.610, 0.594]: buggy ρ=-0.900 PASSES the |ρ|>=0.9
    gate; tie-correct ρ=-0.872 FAILS it). P2 recomputes ρ tie-correctly from the stored
    curve and uses THIS for the monotonicity gate; the stored value is kept for the audit
    trail. This is a real, gate-flipping fix, not a cosmetic one."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    rx = _rank_avg(x) - _rank_avg(x).mean()
    ry = _rank_avg(y) - _rank_avg(y).mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _extract_seed_cells(d: dict) -> tuple[int, dict]:
    """From one steer JSON (Q1 battery or P2), pull (seed, {task: ablate-analysis})
    for the 2 hit tasks. Fails loud if a hit task's ablate cell is absent."""
    seed = d.get("seed", d.get("config", {}).get("seed"))
    if seed is None:
        raise RuntimeError(f"JSON has no seed (keys={list(d.keys())[:8]}).")
    A = d["prereg_analysis"]
    cells = {}
    for t in P2_TASKS:
        key = f"{t}|all|{MODE}"
        if key not in A:
            raise RuntimeError(f"seed {seed}: missing analysis cell '{key}' "
                               f"(have {sorted(A.keys())[:8]}).")
        cells[t] = A[key]
    return int(seed), cells


def _gate(cell: dict) -> dict:
    """The pre-set W5/Q1 gate for one (task,seed): monotone (|ρ|>=0.9) AND beats-random
    (paired |Δ| CI lower bound > 0 AND target loses more than random). Returns the
    per-cell row including the one-sided p for the FDR family."""
    beats = cell.get("beats_random_paired_abs_delta") or {}
    lo, hi = beats.get("ci95_low"), beats.get("ci95_high")
    tmr = cell.get("target_minus_random_effect")
    beats_ci_pos = lo is not None and lo > 0
    # beats-random requires BOTH the |Δ| CI to exclude 0 AND the target to lose MORE
    # than random (the |Δ| contrast is two-sided; a cell where random is more
    # disruptive must not read as "target beats random").
    beats_sig = bool(beats_ci_pos and tmr is not None and tmr > 0)
    # RECOMPUTE ρ tie-correctly from the stored target curve (the harness value uses a
    # tie-breaking that inflates |ρ| at the gate — see spearman_rho_tiecorrect). The
    # tie-correct ρ is the one that decides monotonicity here; keep the stored value for
    # the audit trail.
    rho_stored = cell.get("target_spearman_rho_vs_dose", float("nan"))
    rho = spearman_rho_tiecorrect(cell.get("doses"), cell.get("target_curve"))
    monotone = bool(np.isfinite(rho) and abs(rho) >= 0.9)
    monotone_stored = bool(cell.get("monotone", False))
    p1 = _one_sided_p_from_ci(beats.get("delta_mean"), lo, hi)
    return {
        "rho": rho, "rho_stored": rho_stored, "monotone": monotone,
        "monotone_stored": monotone_stored,
        "target_minus_random": tmr,
        "beats_delta_mean": beats.get("delta_mean"),
        "beats_ci_low": lo, "beats_ci_high": hi,
        "beats_sig": beats_sig, "beats_p": p1,
        "target_effect_maxdose": cell.get("target_effect_maxdose"),
        "random_effect_maxdose": cell.get("random_effect_maxdose"),
        "gate_pass": bool(monotone and beats_sig),
    }


def _spread(xs: list[float]) -> dict:
    a = np.asarray([x for x in xs if x is not None and x == x], float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def p2_verdict(seed_jsons: list[dict]) -> dict:
    # Collect per-seed cells, dedup by seed (first wins; warn on dup).
    by_seed: dict[int, dict] = {}
    provenance = {}
    for d in seed_jsons:
        seed, cells = _extract_seed_cells(d)
        if seed in by_seed:
            print(f"[warn] duplicate seed {seed} — keeping the first.")
            continue
        by_seed[seed] = cells
        provenance[seed] = {
            "steer_repo": d.get("steer_repo"), "floor": d.get("floor"),
            "n_eval": d.get("config", {}).get("steer_n_eval"),
            "set_sizes": d.get("target_provenance", {}).get("set_sizes"),
            "source_schema": "Q1_battery" if len(d.get("task_order", [])) > 2 else "P2",
        }
    seeds = sorted(by_seed.keys())
    N = len(seeds)

    print("=" * 92)
    print(f"P2 — ≥3-SEED REPLICATION of the 2 causal hits  (mode={MODE}, position=all)")
    print(f"seeds (n={N}): {seeds}")
    for s in seeds:
        pr = provenance[s]
        print(f"  seed {s}: {pr['source_schema']}  repo={pr['steer_repo']} floor={pr['floor']} "
              f"n_eval={pr['n_eval']} sets={pr['set_sizes']}")
    print("=" * 92)

    # --- per-(task,seed) rows + gate ---
    rows = []  # one per (task, seed)
    for t in P2_TASKS:
        for s in seeds:
            cell = by_seed[s][t]
            g = _gate(cell)
            g.update({"task": t, "seed": s,
                      "target_curve": cell.get("target_curve"),
                      "random_curve": cell.get("random_curve"),
                      "doses": cell.get("doses")})
            rows.append(g)

    # --- BH-FDR over the per-(task,seed) beats-random family (only cells where the
    # target loses MORE than random enter the family; same rule as W5/Q1) ---
    fam = [(f"{r['task']}|s{r['seed']}", r["beats_p"]) for r in rows
           if r["target_minus_random"] is not None and r["target_minus_random"] > 0]
    fdr_keep = _bh_fdr(fam, q=0.05)
    for r in rows:
        rkey = f"{r['task']}|s{r['seed']}"
        r["beats_random_fdr"] = (rkey in fdr_keep) and (r["target_minus_random"] is not None
                                                        and r["target_minus_random"] > 0)
        # The replication gate uses FDR-corrected beats-random + monotonicity.
        r["gate_pass_fdr"] = bool(r["monotone"] and r["beats_random_fdr"])

    # --- per-task × per-seed table ---
    print("\nPER-TASK × PER-SEED  (gate = monotone |ρ|>=0.9 [tie-correct]  AND  beats-random under BH-FDR):")
    print("  (ρ = tie-correct Spearman; ρ* = stored harness value — differs only when the dose curve has ties)")
    hdr = (f"  {'task':14s} {'seed':5s} {'ρ':7s} {'ρ*':7s} {'mono':5s} {'tgt-rnd':8s} "
           f"{'beatsΔ':8s} {'CI95':18s} {'beatsFDR':8s} {'GATE':5s}")
    print(hdr)
    for t in P2_TASKS:
        for s in seeds:
            r = next(x for x in rows if x["task"] == t and x["seed"] == s)
            lo, hi = r["beats_ci_low"], r["beats_ci_high"]
            ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "[na,na]"
            print(f"  {t:14s} {s:<5d} {r['rho']:+.3f} {r['rho_stored']:+.3f}  {str(r['monotone'])[:5]:5s} "
                  f"{(r['target_minus_random'] or float('nan')):+.4f} "
                  f"{(r['beats_delta_mean'] or float('nan')):+.4f} {ci:18s} "
                  f"{str(r['beats_random_fdr'])[:5]:8s} {str(r['gate_pass_fdr'])[:5]:5s}")

    # --- cross-seed roll-up per task ---
    print("\n" + "=" * 92)
    print("CROSS-SEED ROLL-UP per task (mean ± across-seed std; consistent = gate passes on ALL seeds):")
    per_task = {}
    for t in P2_TASKS:
        tr = [r for r in rows if r["task"] == t]
        n_pass = sum(r["gate_pass_fdr"] for r in tr)
        tmr_vals = [r["target_minus_random"] for r in tr]
        beats_vals = [r["beats_delta_mean"] for r in tr]
        rho_vals = [r["rho"] for r in tr]
        signs = {np.sign(v) for v in tmr_vals if v is not None and v == v}
        sign_consistent = len(signs) == 1
        consistent = (n_pass == N) and sign_consistent
        tmr_s = _spread(tmr_vals); beats_s = _spread(beats_vals); rho_s = _spread(rho_vals)
        per_task[t] = {
            "grounding": (t == "refcoco"),
            "n_seeds": N, "n_gate_pass": n_pass,
            "seeds_passing": [r["seed"] for r in tr if r["gate_pass_fdr"]],
            "seeds_failing": [r["seed"] for r in tr if not r["gate_pass_fdr"]],
            "target_minus_random": tmr_s,
            "beats_delta": beats_s,
            "rho": rho_s,
            "sign_consistent": bool(sign_consistent),
            "consistent_all_seeds": bool(consistent),
            "per_seed": {r["seed"]: {"rho": r["rho"], "rho_stored": r["rho_stored"],
                                     "monotone": r["monotone"],
                                     "target_minus_random": r["target_minus_random"],
                                     "beats_delta_mean": r["beats_delta_mean"],
                                     "beats_ci": [r["beats_ci_low"], r["beats_ci_high"]],
                                     "beats_random_fdr": r["beats_random_fdr"],
                                     "gate_pass": r["gate_pass_fdr"]} for r in tr},
        }
        verdict = ("FIRM (replicated on all seeds)" if consistent
                   else f"NOT REPLICATED ({n_pass}/{N} seeds pass"
                        + ("" if sign_consistent else ", target-minus-random sign FLIPS") + ")")
        print(f"\n  [{t}]  grounding={t == 'refcoco'}   → {verdict}")
        print(f"    gate passes on {n_pass}/{N} seeds  passing={per_task[t]['seeds_passing']} "
              f"failing={per_task[t]['seeds_failing']}")
        print(f"    target-minus-random: mean={tmr_s['mean']:+.4f} ± {tmr_s['std']:.4f} "
              f"(min {tmr_s['min']:+.4f}, max {tmr_s['max']:+.4f})  sign_consistent={sign_consistent}")
        print(f"    beats-random Δ:      mean={beats_s['mean']:+.4f} ± {beats_s['std']:.4f} "
              f"(min {beats_s['min']:+.4f}, max {beats_s['max']:+.4f})")
        print(f"    ρ(target,dose):      mean={rho_s['mean']:+.3f} ± {rho_s['std']:.3f} "
              f"(min {rho_s['min']:+.3f}, max {rho_s['max']:+.3f})")

    # --- overall verdict ---
    firm = [t for t in P2_TASKS if per_task[t]["consistent_all_seeds"]]
    fragile = [t for t in P2_TASKS if not per_task[t]["consistent_all_seeds"]]
    if N < 3:
        overall = (f"INCOMPLETE — only {N} seeds present (<3); need ≥3 to adjudicate. "
                   f"firm-so-far={firm} fragile-so-far={fragile}")
    elif len(firm) == 2:
        overall = "BOTH HITS FIRM — RefCOCO + COCO-Caption replicate on all seeds (enter lineage map as seed-replicated)."
    elif firm:
        overall = (f"PARTIAL — {firm} FIRM (seed-replicated); {fragile} NOT replicated "
                   f"(downgrade {fragile} to single-seed-fragile, flag).")
    else:
        overall = ("NEITHER HIT REPLICATES — the causal leg is NOT robust across seeds; "
                   "downgrade both to 'single-seed, not replicated' and flag.")

    print("\n" + "=" * 92)
    print(f"P2 VERDICT: {overall}")
    print("=" * 92)
    return {"verdict": overall, "n_seeds": N, "seeds": seeds,
            "firm": firm, "fragile": fragile, "per_task": per_task,
            "rows": rows, "fdr_keep": sorted(fdr_keep), "provenance": provenance}


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m infra.pg.p2_verdict <seed1337_q1_battery.json> <p2_seedX.json> ...")
        return
    seed_jsons = []
    for p in paths:
        d = json.load(open(p))
        if "prereg_analysis" not in d:
            print(f"[skip] {p}: not a steer JSON (keys {list(d.keys())[:8]})")
            continue
        seed_jsons.append(d)
    if seed_jsons:
        p2_verdict(seed_jsons)


if __name__ == "__main__":
    main()
