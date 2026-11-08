# Q1 — verdict reader (LOCAL, JSON-only; no model compute).
"""Read the Q1 steer_battery_*.json and render the per-task generalisation verdict.

Mirrors infra.pg.w5_verdict.steer_verdict but over ALL 5 battery tasks at position
'all'. For each (task, mode): target/random dose curves, target monotonicity
(Spearman ρ vs dose), target-minus-random max-dose effect, the beats-random paired
|Δ| bootstrap CI. Then BH-FDR (q=0.05) across the family of beats-random tests, and
the GENERAL-vs-RefCOCO-SPECIFIC call: a task "perturbs" iff (monotone AND beats-random
under FDR). Reuses w5_verdict._bh_fdr / _one_sided_p_from_ci verbatim (same statistics).

Pure: argv = path(s) to steer_battery_*.json. No side effects beyond print.
"""
from __future__ import annotations

import json
import sys

from infra.pg.w5_verdict import _bh_fdr, _one_sided_p_from_ci, fmt_curve


def battery_verdict(steer: dict) -> dict:
    A = steer["prereg_analysis"]
    base = steer["results"]["baseline"]
    grounding = steer.get("grounding", {})
    task_order = steer.get("task_order", sorted({A[k]["task"] for k in A}))
    print("=" * 84)
    print(f"Q1 STEER  model={steer.get('steer_repo')}  floor={steer.get('floor')}  "
          f"position={steer.get('config', {}).get('position')}")
    tp = steer.get("target_provenance", {})
    print(f"reused target: L{tp.get('layer')}/{tp.get('width')} {tp.get('sae_rel_path')}  "
          f"set_sizes={tp.get('set_sizes')}  w4_reproduced={tp.get('w4_reproduced_counts')}")
    print(f"pooled img-mass: {tp.get('pooled_image_mass_fraction')}")
    print("baseline: " + "  ".join(f"{t}={base[t]['mean']:.3f}" for t in task_order))
    print("=" * 84)

    rows = []
    for key in sorted(A.keys()):
        a = A[key]
        t, mode = a["task"], a["mode"]
        beats = a.get("beats_random_paired_abs_delta") or {}
        lo, hi = beats.get("ci95_low"), beats.get("ci95_high")
        beats_sig = lo is not None and lo > 0
        p1 = _one_sided_p_from_ci(beats.get("delta_mean"), lo, hi)
        print(f"\n[{key}]  grounding={a['grounding']}")
        print(f"  target  {fmt_curve(a['target_curve'])}  (doses {a['doses']})")
        print(f"  random  {fmt_curve(a['random_curve'])}")
        print(f"  ρ(target,dose)={a['target_spearman_rho_vs_dose']:.3f}  monotone={a['monotone']}")
        print(f"  effect@max: target={a['target_effect_maxdose']:.4f} "
              f"random={a['random_effect_maxdose']:.4f} "
              f"target-random={a['target_minus_random_effect']:+.4f}")
        print(f"  beats-random paired |Δ|: mean={beats.get('delta_mean')} "
              f"CI95=[{lo},{hi}] excludes0(>0)={beats_sig}")
        rows.append({"key": key, "task": t, "mode": mode, "grounding": a["grounding"],
                     "monotone": a["monotone"], "rho": a["target_spearman_rho_vs_dose"],
                     "target_minus_random": a["target_minus_random_effect"],
                     "beats_random_sig": beats_sig, "beats_p": p1})

    # BH-FDR over the beats-random family (queue guardrail: FDR across tasks). Only a
    # cell where the target ALSO loses more than random counts (the |Δ| contrast is
    # two-sided; a cell where random is MORE disruptive must not read as "beats random").
    fam = [(r["key"], r["beats_p"]) for r in rows if r["target_minus_random"] > 0]
    fdr_keep = _bh_fdr(fam, q=0.05)
    for r in rows:
        r["beats_random_fdr"] = (r["key"] in fdr_keep) and (r["target_minus_random"] > 0)
        r["perturbs"] = bool(r["monotone"] and r["beats_random_fdr"])

    # Per-task roll-up: a task "perturbs" iff ANY mode is monotone + beats-random(FDR).
    per_task = {}
    for t in task_order:
        tr = [r for r in rows if r["task"] == t]
        per_task[t] = {
            "grounding": grounding.get(t),
            "perturbs": any(r["perturbs"] for r in tr),
            "best_rho": max((abs(r["rho"]) for r in tr if r["rho"] == r["rho"]), default=float("nan")),
            "best_target_minus_random": max((r["target_minus_random"] for r in tr), default=float("nan")),
            "any_beats_random_fdr": any(r["beats_random_fdr"] for r in tr),
            "modes": {r["mode"]: {"monotone": r["monotone"], "rho": r["rho"],
                                  "target_minus_random": r["target_minus_random"],
                                  "beats_random_fdr": r["beats_random_fdr"],
                                  "perturbs": r["perturbs"]} for r in tr},
        }

    print("\n" + "=" * 84)
    print("PER-TASK (perturbs = monotone AND beats-random under BH-FDR q=0.05):")
    print(f"  {'task':14s} {'ground':6s} {'perturbs':8s} {'best|ρ|':8s} "
          f"{'tgt-rnd':8s} {'beatsFDR':8s}")
    for t in task_order:
        d = per_task[t]
        print(f"  {t:14s} {str(d['grounding']):6s} {str(d['perturbs']):8s} "
              f"{d['best_rho']:.3f}    {d['best_target_minus_random']:+.4f}  "
              f"{str(d['any_beats_random_fdr']):8s}")

    grounding_tasks = [t for t in task_order if grounding.get(t)]
    nongrounding = [t for t in task_order if not grounding.get(t)]
    g_pert = [t for t in grounding_tasks if per_task[t]["perturbs"]]
    ng_pert = [t for t in nongrounding if per_task[t]["perturbs"]]
    n_pert = sum(per_task[t]["perturbs"] for t in task_order)

    # Verdict logic (honest, no spin):
    #  - GENERAL: ≥2 tasks perturb (esp. grounding-heavy), not just RefCOCO.
    #  - GROUNDING-SELECTIVE: only grounding tasks perturb (RefCOCO + ≥1 of VSR/TextVQA).
    #  - REFCOCO-SPECIFIC: only RefCOCO perturbs.
    #  - NULL: nothing perturbs (effect was RefCOCO-only in W5 and does not even
    #    replicate here) — a real, fine result.
    if n_pert == 0:
        verdict = "NULL (no task beats random under FDR — not even RefCOCO replicates here)"
    elif set(g_pert) == {"refcoco"} and not ng_pert and len(g_pert) == 1:
        verdict = "REFCOCO-SPECIFIC (only RefCOCO perturbs)"
    elif g_pert and not ng_pert and len(g_pert) >= 2:
        verdict = f"GROUNDING-SELECTIVE (grounding tasks {g_pert} perturb; non-grounding do not)"
    elif n_pert >= 2:
        verdict = f"GENERAL (≥2 tasks perturb: {[t for t in task_order if per_task[t]['perturbs']]})"
    else:
        verdict = f"SINGLE-TASK ({[t for t in task_order if per_task[t]['perturbs']][0]} only)"

    print("\n" + "=" * 84)
    print(f"grounding tasks perturbing: {g_pert}   non-grounding perturbing: {ng_pert}")
    print(f"Q1 VERDICT: {verdict}")
    print("=" * 84)
    return {"verdict": verdict, "per_task": per_task,
            "grounding_perturbing": g_pert, "nongrounding_perturbing": ng_pert,
            "n_perturbing": n_pert, "rows": rows, "fdr_keep": sorted(fdr_keep)}


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m infra.pg.q1_verdict steer_battery_*.json")
        return
    for p in paths:
        d = json.load(open(p))
        if "prereg_analysis" in d and "task_order" in d:
            battery_verdict(d)
        else:
            print(f"[skip] {p}: not a Q1 battery steer JSON (keys {list(d.keys())[:8]})")


if __name__ == "__main__":
    main()
