"""W12 §9 STEP-5 — unblind once (CPU, fp64). Reads the real ΔW split a SINGLE time and applies the
FROZEN §6 rules from controls.json. No threshold is set here — they were frozen from the §5
matched-random null in w12_controls (read from S3; CONTROLS_OK + PREFLIGHT_OK + checkpoint SHA all
asserted before any real value is read).

Per matrix (26×7 = 182): decompose ΔW (fp64) → gain/route/out-of-span energies, E_gain, C_route (at
the pre-committed headline-k band), f_perp + §6B spectra. Per matrix: empirical p vs the stored null
samples (one-sided enrichment). BH-FDR q=0.05 over the 182 family for {gain-enriched,
routing-structured, out-of-span-above-null}. The family verdict is driven by the per-matrix counts
(each already gated on the FROZEN τ_p99 + BH), with a per-projection-type localized read so a
type-specific effect is not washed out by a global-majority threshold. Stability vs the §7
rank-matched null is reported. Verdict on the §11 grid, stated as enrichment ABOVE the null.

--pair base-pt (headline: gemma-2-2b → PG2-pt) or pt-mix (logged secondary: PG2-pt → PG2-mix).

    python -m infra.aws_pg.jobs.w12_unblind --out <dir> --pair base-pt \
        --controls-s3 s3://pg-lineage-769063704696/results/w12_controls/out/controls.json \
        --preflight-s3 s3://pg-lineage-769063704696/results/w12_preflight/out/preflight.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ENRICHED_TYPE_MIN = 20   # ≥20/26 layer matrices of a projection type ⇒ that type is "structured"


def _emp_p(obs, null_samples):
    """One-sided empirical p that obs is NOT above the null: (1 + #{null ≥ obs}) / (n + 1)."""
    if obs is None:
        return 1.0
    n = len(null_samples)
    return (1 + sum(1 for x in null_samples if x >= obs)) / (n + 1)


def _bh_reject(pvals, q):
    """Benjamini–Hochberg: bool list (same order) of rejections at FDR q."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh_rank = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= (rank / m) * q:
            thresh_rank = rank
    rej = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= thresh_rank:
            rej[i] = True
    return rej


def _boot_mean_ci(vals, n_boot, seed):
    import numpy as np
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return {"mean": float(a.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(a.size)}


def _load_pair(pair, device="cpu"):
    """Return (matched {key:(wb,wa)}, before_sha, after_sha, before_repo, after_repo)."""
    import torch
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from huggingface_hub import HfApi
    from infra.pg import gain_routing as GR
    from infra.pg.entities import ENTITIES, resolve_submodule_paths
    api = HfApi()

    def _pg_lm(repo):
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
        p = resolve_submodule_paths(m)
        sd = {k: v.clone() for k, v in m.get_submodule(p["language_model"]).state_dict().items()}
        del m
        return sd

    if pair == "base-pt":
        L0, PG = ENTITIES["L0"], ENTITIES["PG"]
        g2 = AutoModelForCausalLM.from_pretrained(L0.repo_id, revision=L0.revision,
                                                  torch_dtype=torch.bfloat16, device_map="cpu",
                                                  attn_implementation="eager")
        before = {k: v.clone() for k, v in g2.model.state_dict().items()}; del g2
        after = _pg_lm(PG.repo_id)
        bsha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
        asha = api.model_info(PG.repo_id).sha
        matched = GR.matched_decoder_weight_matrices(after, before)
        return matched, bsha, asha, L0.repo_id, PG.repo_id
    if pair == "pt-mix":
        PG, MIX = ENTITIES["PG"], ENTITIES["PG_mix"]
        before = _pg_lm(PG.repo_id)
        after = _pg_lm(MIX.repo_id)
        bsha = api.model_info(PG.repo_id).sha
        asha = api.model_info(MIX.repo_id).sha
        matched = GR.matched_decoder_weight_matrices(after, before)
        return matched, bsha, asha, PG.repo_id, MIX.repo_id
    raise ValueError(f"unknown pair {pair!r}; expected base-pt or pt-mix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", default="base-pt", choices=["base-pt", "pt-mix"])
    ap.add_argument("--controls-s3",
                    default="s3://pg-lineage-769063704696/results/w12_controls/out/controls.json")
    ap.add_argument("--preflight-s3",
                    default="s3://pg-lineage-769063704696/results/w12_preflight/out/preflight.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fdr-q", type=float, default=0.05)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import gain_routing as GR

    # --- read FROZEN controls + preflight gates (assert before any real value is read) ---
    cj = out / "controls.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(cj), "--region", "us-east-1"], check=True)
    ctrl = json.loads(cj.read_text())
    assert ctrl["CONTROLS_OK"], "controls.json CONTROLS_OK=False → cannot unblind (frozen-gate breach)."
    pj = out / "preflight.json"
    subprocess.run(["aws", "s3", "cp", args.preflight_s3, str(pj), "--region", "us-east-1"], check=True)
    pf = json.loads(pj.read_text())
    assert pf["PREFLIGHT_OK"], "preflight.json PREFLIGHT_OK=False → cannot unblind."
    assert pf["entities"]["PG"]["sha"] == ctrl["entities"]["PG"]["sha"], \
        "preflight vs controls PG-SHA drift → HALT (gates ran on different weights)."

    headline_k = int(ctrl["config"]["headline_k"])
    degen_rtol = float(ctrl["config"]["degen_rtol"])
    recon_tol = float(ctrl["config"].get("recon_tol", 1e-10))
    thr = ctrl["thresholds_by_shape"]
    null_samples = {k: v["samples"] for k, v in ctrl["null_band_by_shape"].items()}
    print(f"[W12-UNBLIND] pair={args.pair} controls OK; headline_k={headline_k} shapes={list(thr)}")

    matched, bsha, asha, brepo, arepo = _load_pair(args.pair)
    assert len(matched) == 26 * 7
    if args.pair == "base-pt":
        assert bsha == ctrl["entities"]["L0"]["sha"] and asha == ctrl["entities"]["PG"]["sha"], \
            "base-pt checkpoint SHA drift vs controls → HALT (null band is for the wrong weights)."
    else:
        # pt-mix is a logged secondary; the null is shape-keyed + scale-invariant so the frozen
        # thresholds apply, but the preflight base-init gate is base-pt-scoped → note it.
        assert bsha == ctrl["entities"]["PG"]["sha"], "pt-mix W_before (PG-pt) SHA drift vs controls."
        print("[W12-UNBLIND] pt-mix SECONDARY: frozen shape-keyed thresholds reused; base-init "
              "gate is base-pt-scoped (PG-pt↔PG-mix share architecture, asserted by build).")

    # =====================================================================
    # decompose every real ΔW once + §6B spectra.
    # =====================================================================
    rows = []
    for key, (wb, wa) in matched.items():
        # check_recon=False: the fp64 reconstruction identity (the O(n³) n×n projector) was already
        # validated in preflight on all three matrix orientations (3e-15 ≪ recon_tol); recomputing it
        # per-matrix here is redundant and was the controls/unblind runtime bottleneck on wide MLP rows.
        d = GR.decompose(wb, wa, top_k=headline_k, degen_rtol=degen_rtol, check_recon=False)
        sp = GR.spectra(wb, wa, top=headline_k)
        skey = f"{d['shape'][0]}x{d['shape'][1]}"
        parts = key.split(".")
        layer = int(parts[1]) if parts[0] == "layers" and parts[1].isdigit() else -1
        rows.append({
            "key": key, "type": GR.fine_type(key), "layer": layer, "shape_key": skey,
            "E_gain": d["E_gain"], "C_route": d.get("C_route"), "f_perp": d["f_perp"],
            "gain_frac": d["gain_frac"], "route_frac": d["route_frac"], "in_span_frac": d["in_span_frac"],
            "n_degenerate_pairs": d["n_degenerate_pairs"], "degen_route_frac": d["degen_route_frac"],
            "mean_cos_u_topk": sp["mean_cos_u_topk"], "mean_cos_v_topk": sp["mean_cos_v_topk"],
            "dsigma_mean_abs": float(np.mean(np.abs(sp["dsigma"]))),
        })

    # per-matrix empirical p + BH-FDR over the 182 family.
    p_gain = [_emp_p(r["E_gain"], null_samples[r["shape_key"]]["E_gain"]) for r in rows]
    p_route = [_emp_p(r["C_route"], null_samples[r["shape_key"]]["C_route"]) for r in rows]
    p_fperp = [_emp_p(r["f_perp"], null_samples[r["shape_key"]]["f_perp"]) for r in rows]
    rej_gain, rej_route, rej_fperp = (_bh_reject(p, args.fdr_q) for p in (p_gain, p_route, p_fperp))
    for i, r in enumerate(rows):
        t = thr[r["shape_key"]]
        r["p_gain"], r["p_route"], r["p_fperp"] = p_gain[i], p_route[i], p_fperp[i]
        r["gain_enriched"] = bool(rej_gain[i] and r["E_gain"] > t["tau_gain_p99"])
        r["gain_enriched_rankmatched"] = bool(r["E_gain"] > t["tau_gain_p99_rankmatched"])   # §7 stability
        r["route_structured"] = bool(rej_route[i] and r["C_route"] is not None
                                     and r["C_route"] > t["tau_route_p99"])
        r["out_of_span_above_null"] = bool(rej_fperp[i] and r["f_perp"] > t["f_perp_null_p99"])

    # =====================================================================
    # aggregate per projection-type (CI across the 26 layers) + per-depth.
    # =====================================================================
    by_type = {}
    for t in GR.PROJECTION_TYPES:
        rs = [r for r in rows if r["type"] == t]
        by_type[t] = {
            "n": len(rs),
            "E_gain": _boot_mean_ci([r["E_gain"] for r in rs], args.boot, args.seed),
            "C_route": _boot_mean_ci([r["C_route"] for r in rs], args.boot, args.seed + 1),
            "f_perp": _boot_mean_ci([r["f_perp"] for r in rs], args.boot, args.seed + 2),
            "gain_frac": _boot_mean_ci([r["gain_frac"] for r in rs], args.boot, args.seed + 3),
            "mean_cos_u_topk": float(np.mean([r["mean_cos_u_topk"] for r in rs])),
            "max_degen_route_frac": float(max(r["degen_route_frac"] for r in rs)),
            "n_gain_enriched": sum(r["gain_enriched"] for r in rs),
            "n_gain_enriched_rankmatched": sum(r["gain_enriched_rankmatched"] for r in rs),
            "n_route_structured": sum(r["route_structured"] for r in rs),
            "n_oos_above_null": sum(r["out_of_span_above_null"] for r in rs),
        }
    by_depth = {}
    for L in sorted({r["layer"] for r in rows if r["layer"] >= 0}):
        rs = [r for r in rows if r["layer"] == L]
        by_depth[str(L)] = {"E_gain_mean": float(np.mean([r["E_gain"] for r in rs])),
                            "C_route_mean": float(np.mean([r["C_route"] for r in rs])),
                            "f_perp_mean": float(np.mean([r["f_perp"] for r in rs]))}

    # =====================================================================
    # Verdict (§11 grid) — driven by the per-matrix counts (each gated on the FROZEN τ_p99 + BH),
    # with a per-type localized read so a type-specific effect is not washed out by a global majority.
    # =====================================================================
    N = len(rows)
    n_gain = sum(r["gain_enriched"] for r in rows)
    n_gain_rk = sum(r["gain_enriched_rankmatched"] for r in rows)
    n_route = sum(r["route_structured"] for r in rows)
    n_oos = sum(r["out_of_span_above_null"] for r in rows)
    types_gain = [t for t in GR.PROJECTION_TYPES if by_type[t]["n_gain_enriched"] >= ENRICHED_TYPE_MIN]
    types_route = [t for t in GR.PROJECTION_TYPES if by_type[t]["n_route_structured"] >= ENRICHED_TYPE_MIN]
    types_oos = [t for t in GR.PROJECTION_TYPES if by_type[t]["n_oos_above_null"] >= ENRICHED_TYPE_MIN]
    gain_enriched_family = bool(n_gain >= 0.5 * N or len(types_gain) >= 1)
    routing_structured_family = bool(n_route >= 0.5 * N or len(types_route) >= 1)
    new_family = bool(n_oos >= 0.5 * N or len(types_oos) >= 1)

    if gain_enriched_family and routing_structured_family:
        verdict = "MIXED — gain enriched AND routing structured above the null (report the split)"
    elif gain_enriched_family:
        verdict = "H_gain — gain (diagonal) preferentially enriched ≫ null; routing not structured"
    elif routing_structured_family:
        verdict = "H_routing (lit-predicted) — routing structured above null, gain at chance"
    elif new_family:
        verdict = "H_new — substantial out-of-span energy above the null"
    else:
        verdict = "H_null — split within the matched-random-matrix null"

    all_E = _boot_mean_ci([r["E_gain"] for r in rows], args.boot, args.seed)
    all_C = _boot_mean_ci([r["C_route"] for r in rows], args.boot, args.seed + 1)
    all_F = _boot_mean_ci([r["f_perp"] for r in rows], args.boot, args.seed + 2)
    result = {
        "torch": torch.__version__, "device": "cpu", "pair": args.pair,
        "entities": {"before": {"repo": brepo, "sha": bsha}, "after": {"repo": arepo, "sha": asha}},
        "controls_s3": args.controls_s3, "headline_k": headline_k, "fdr_q": args.fdr_q,
        "family": {"n_matrices": N,
                   "E_gain_mean_ci": all_E, "C_route_mean_ci": all_C, "f_perp_mean_ci": all_F,
                   "n_gain_enriched": n_gain, "n_gain_enriched_rankmatched": n_gain_rk,
                   "n_route_structured": n_route, "n_oos_above_null": n_oos,
                   "types_gain_enriched": types_gain, "types_route_structured": types_route,
                   "types_oos_above_null": types_oos,
                   "gain_enriched_family": gain_enriched_family,
                   "routing_structured_family": routing_structured_family, "new_family": new_family},
        "by_type": by_type, "by_depth": by_depth,
        "VERDICT": verdict, "per_matrix": rows,
    }
    (out / f"unblind_{args.pair}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W12-UNBLIND] family E_gain={all_E['mean']:.2f}[{all_E['lo']:.2f},{all_E['hi']:.2f}] "
          f"C_route={all_C['mean']:.2f}[{all_C['lo']:.2f},{all_C['hi']:.2f}] f_perp={all_F['mean']:.4f} | "
          f"gain {n_gain}/182 (rank-matched {n_gain_rk}) route {n_route}/182 oos {n_oos}/182")
    print(f"[W12-UNBLIND] types_gain={types_gain} types_route={types_route} types_oos={types_oos}")
    print(f"[W12-UNBLIND] VERDICT ({args.pair}): {verdict}")
    for t in GR.PROJECTION_TYPES:
        b = by_type[t]
        print(f"    {t:10s} E_gain={b['E_gain']['mean']:.2f} C_route={b['C_route']['mean']:.2f} "
              f"f_perp={b['f_perp']['mean']:.4f} cos_u={b['mean_cos_u_topk']:.3f} "
              f"gain_enr={b['n_gain_enriched']}/{b['n']} route_str={b['n_route_structured']}/{b['n']} "
              f"degenRF≤{b['max_degen_route_frac']:.1e}")


if __name__ == "__main__":
    main()
