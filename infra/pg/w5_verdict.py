# W5 — verdict reader (LOCAL, JSON-only; no model compute).
"""Read the W5 steer/patch result JSONs and render the pre-registered verdict.

Runs locally on the result JSONs pulled from the pg-results volume (no GPU, no model).
Renders, for the steering prong, per (task,position,mode): the target/random/amplified_txt
dose curves, the target monotonicity (Spearman ρ vs dose), the target-vs-random max-dose
effect, the beats-random paired contrast (CI), and the pre-reg PASS/FAIL flags — then the
overall SUPPORTED / NOT-SUPPORTED call (pre-reg §3). For patching, the per-layer recovery
peak + lower/upper-half gradient (W4 predicts the lower LM).

Pure: argv = paths to steer_*.json (+ optional patch_*.json). No side effects beyond print.
"""
from __future__ import annotations

import json
import sys


import math


def fmt_curve(xs):
    return "[" + ", ".join(f"{v:.3f}" if v is not None and v == v else "nan" for v in xs) + "]"


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _one_sided_p_from_ci(mean, lo, hi):
    """One-sided p for H1: contrast>0, from a bootstrap mean + 95% CI (normal approx:
    SE = (hi-lo)/(2*1.96); p = P(Z <= 0 | N(mean,SE)) = Φ(-mean/SE)). Approximation used
    only for the FDR family-control over the beats-random tests (CIs are the primary read)."""
    if mean is None or lo is None or hi is None:
        return float("nan")
    se = (hi - lo) / (2 * 1.959964)
    if se <= 0:
        return 0.0 if mean > 0 else 1.0
    return _norm_cdf(-mean / se)


def _bh_fdr(pairs, q=0.05):
    """Benjamini-Hochberg over [(key, p)] → set of keys rejected at FDR=q."""
    valid = [(k, p) for k, p in pairs if p == p]
    if not valid:
        return set()
    valid.sort(key=lambda kp: kp[1])
    n = len(valid)
    crit = None
    for i, (k, p) in enumerate(valid, start=1):
        if p <= (i / n) * q:
            crit = p
    if crit is None:
        return set()
    return {k for k, p in valid if p <= crit}


def steer_verdict(steer: dict) -> dict:
    """Render the pre-reg success criterion (§3) over the steering analysis. The PRIMARY
    test is position 'all' on RefCOCO (image-grounded, S_B=0.00). Returns a verdict dict."""
    A = steer["prereg_analysis"]
    base = steer["results"]["baseline"]
    print("=" * 78)
    print(f"STEER  model={steer.get('steer_repo')}  floor={steer.get('floor')}  "
          f"sets={steer.get('set_sizes')}")
    print(f"baseline: " + "  ".join(f"{t}={base[t]['mean']:.3f}[{base[t]['ci95_low']:.3f},{base[t]['ci95_high']:.3f}]"
                                    for t in base))
    print("=" * 78)
    rows = []
    for key in sorted(A.keys()):
        a = A[key]
        tname, pos, mode = key.split("|")
        beats = a.get("beats_random_paired_abs_delta") or {}
        beats_ci = (beats.get("ci95_low"), beats.get("ci95_high"))
        # beats-random is significant iff the paired |Δ_target|-|Δ_random| CI excludes 0 (>0).
        beats_sig = beats_ci[0] is not None and beats_ci[0] > 0
        print(f"\n[{key}]")
        print(f"  target_curve     {fmt_curve(a['target_curve'])}  (doses {a['doses']})")
        print(f"  random_curve     {fmt_curve(a['random_curve'])}")
        if any(v == v for v in a.get('amplified_txt_curve', [])):
            print(f"  amplified_txt    {fmt_curve(a['amplified_txt_curve'])}")
        print(f"  target Spearman ρ vs dose = {a['target_spearman_rho_vs_dose']:.3f}  "
              f"(monotone if |ρ|>=0.9 → {a['monotone']})")
        print(f"  effect@maxdose: target={a['target_effect_maxdose']:.4f}  "
              f"random={a['random_effect_maxdose']:.4f}  "
              f"target-random={a['target_minus_random_effect']:+.4f}")
        print(f"  beats-random paired |Δ| contrast: mean={beats.get('delta_mean')}  "
              f"CI95=[{beats_ci[0]},{beats_ci[1]}]  excludes0(>0)={beats_sig}")
        p1 = _one_sided_p_from_ci(beats.get("delta_mean"), beats_ci[0], beats_ci[1])
        rows.append({"key": key, "task": tname, "pos": pos, "mode": mode,
                     "monotone": a["monotone"], "beats_random": a["target_beats_random"],
                     "beats_random_sig": beats_sig, "beats_p": p1,
                     "rho": a["target_spearman_rho_vs_dose"],
                     "target_minus_random": a["target_minus_random_effect"]})

    # FDR (BH q=0.05) over the beats-random family (pre-reg §5). Only count a cell as a
    # genuine beats-random when the target ALSO loses more than random (target_minus_random>0)
    # — the |Δ| contrast is two-sided, so a cell where random is MORE disruptive must not
    # be read as "target beats random" even if |Δ|>0.
    fam = [(r["key"], r["beats_p"]) for r in rows if r["target_minus_random"] > 0]
    fdr_keep = _bh_fdr(fam, q=0.05)
    for r in rows:
        r["beats_random_fdr"] = (r["key"] in fdr_keep) and (r["target_minus_random"] > 0)

    # PRE-REG PRIMARY (§3): RefCOCO, position 'all', either mode: monotone (|ρ|>=0.9) AND
    # beats random (target loses more) AND survives FDR.
    primary = [r for r in rows if r["task"] == "refcoco" and r["pos"] == "all"]
    primary_pass = any(r["monotone"] and r["beats_random_fdr"] for r in primary)
    any_refcoco_pass = any(r["task"] == "refcoco" and r["monotone"] and r["beats_random_fdr"] for r in rows)
    print("\n" + "=" * 78)
    print("FDR-surviving beats-random cells (target loses MORE than random, BH q=0.05):")
    for r in rows:
        if r["beats_random_fdr"]:
            print(f"  {r['key']:28s} ρ={r['rho']:+.3f} monotone={r['monotone']} "
                  f"target-random={r['target_minus_random']:+.4f} p≈{r['beats_p']:.4f}")
    print(f"\nPRE-REG PRIMARY (RefCOCO, pos=all): monotone + beats-random(FDR) → "
          f"{'PASS' if primary_pass else 'FAIL'}")
    print(f"ANY RefCOCO position cell passing (monotone + beats-random FDR): {any_refcoco_pass}")
    verdict = "SUPPORTED" if primary_pass else ("SUPPORTED-secondary-only" if any_refcoco_pass else "NOT SUPPORTED")
    print(f"H_gc (gain-control) STEERING VERDICT: {verdict}")
    print("=" * 78)
    return {"verdict": verdict, "primary_pass": primary_pass,
            "any_refcoco_pass": any_refcoco_pass, "rows": rows, "fdr_keep": sorted(fdr_keep)}


def patch_summary(patch: dict):
    print("\n" + "#" * 78)
    print(f"PATCH  model={patch.get('patch_repo')}  n_lm_layers={patch['config']['n_lm_layers']}")
    print("#" * 78)
    for tname, byk in patch["tasks"].items():
        for key, d in byk.items():
            ap = d["by_position"]
            print(f"\n[{key}]  clean_lp={d['clean_logprob_mean']:.3f} corrupt_lp={d['corrupt_logprob_mean']:.3f} "
                  f"n_pairs={d['n_pairs']}")
            for pos in ("all", "image", "text"):
                p = ap[pos]
                lm = p["layer_recovery_mean"]
                print(f"  {pos:5s}: best_L={p['best_layer']} rec={p['best_layer_recovery']} "
                      f"lower_half={p['lower_half_mean']:.3f} upper_half={p['upper_half_mean']:.3f} "
                      f"n_valid={p['n_valid_pairs']}")
                # show the per-layer curve compactly (every 2nd layer)
                if lm:
                    print(f"         curve[::2]={fmt_curve(lm[::2])}")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m infra.pg.w5_verdict steer_*.json [patch_*.json]")
        return
    for p in paths:
        d = json.load(open(p))
        if "prereg_analysis" in d:
            steer_verdict(d)
        elif "tasks" in d and "config" in d and "n_lm_layers" in d.get("config", {}):
            patch_summary(d)
        else:
            print(f"[skip] {p}: unrecognized schema (keys {list(d.keys())[:8]})")


if __name__ == "__main__":
    main()
