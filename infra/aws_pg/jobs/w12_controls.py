"""W12 §9 STEP-2/4 — develop-on-control + controls-live gate (CPU, fp64).

NO peek at the real ΔW split (§9-2 firewall). Reads ONLY:
  * matrix SHAPES (public architecture) → the §5 matched-random-matrix null BAND per shape;
  * the published W1 per-type eff_rank (public reference) → the §7 RANK-MATCHED null variant;
  * synthetic PLANTED splits (positive control) recovered to frac ≈ 1 (arithmetic validity, §5);
  * the N1 identity (decompose(W,W) ⇒ ΔW = 0, all metrics null).

The decision RULES are frozen here (τ_gain / τ_route = the 99th-pct of the matched-random null;
degeneracy tol; the pre-committed headline-k); the live NUMERIC bands + raw null samples (for the
unblind's empirical-p BH-FDR) are measured on the infra and written to controls.json. EVERY W12
conclusion is then stated as enrichment ABOVE this band (§5/§6) — never a raw fraction.

    python -m infra.aws_pg.jobs.w12_controls --out <dir> [--n-draws 256] [--headline-k 64]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-s3",
                    default="s3://pg-lineage-769063704696/reference/w1_real_dw/w1_real_dw_summary.json")
    ap.add_argument("--n-draws", type=int, default=256, help="§5 null draws per shape (band + empirical p)")
    ap.add_argument("--headline-k", type=int, default=64, help="pre-committed top-k band for C_route")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--degen-rtol", type=float, default=1e-3)
    ap.add_argument("--ctrl-tol", type=float, default=1e-6, help="§5 positive-control planted-split tol")
    ap.add_argument("--recon-tol", type=float, default=1e-10)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration

    from infra.pg import gain_routing as GR
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    PG, L0 = ENTITIES["PG"], ENTITIES["L0"]
    api = HfApi()
    pg_sha = api.model_info(PG.repo_id).sha
    l0_sha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
    print(f"[W12-CTRL] PG={PG.repo_id}@{pg_sha} L0={L0.repo_id}@{l0_sha}")

    # published W1 per-type eff_rank (public reference) → rank-matched null (§7). NOT a split peek.
    ref_local = out / "reference_w1_real_dw_summary.json"
    subprocess.run(["aws", "s3", "cp", args.reference_s3, str(ref_local), "--region", "us-east-1"], check=True)
    ref_eff = {t: v["eff_rank_mean"] for t, v in json.loads(ref_local.read_text())["lm"]["by_matrix_type_fine"].items()}

    # --- load decoders (CPU); enumerate the 182 weight matrices + shapes (no split peek) ---
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG.repo_id, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    paths = resolve_submodule_paths(pg)
    pg_lm_sd = {k: v.clone() for k, v in pg.get_submodule(paths["language_model"]).state_dict().items()}
    del pg
    g2 = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager")
    l0_sd = {k: v.clone() for k, v in g2.model.state_dict().items()}
    del g2

    matched = GR.matched_decoder_weight_matrices(pg_lm_sd, l0_sd)
    assert len(matched) == 26 * 7, f"expected 182 decoder matrices, got {len(matched)}"

    # shape → representative key + the projection types sharing that shape (for the rank-matched K).
    shape_to_key, shape_types = {}, {}
    for k, (wb, _wa) in matched.items():
        s = tuple(wb.shape)
        shape_to_key.setdefault(s, k)
        shape_types.setdefault(s, set()).add(GR.fine_type(k))
    print(f"[W12-CTRL] {len(matched)} matrices, {len(shape_to_key)} unique shapes: {sorted(shape_to_key)}")

    # =====================================================================
    # §5 matched-random-matrix null band per shape (norm-matched primary + rank-matched §7 variant).
    # =====================================================================
    null_by_shape, null_rankmatched, thresholds = {}, {}, {}
    for shape in sorted(shape_to_key):
        key = f"{shape[0]}x{shape[1]}"
        rank_k = int(round(sum(ref_eff[t] for t in shape_types[shape]) / len(shape_types[shape])))
        nb = GR.null_band(shape, n_draws=args.n_draws, seed=args.seed, top_k=args.headline_k)
        nr = GR.null_band(shape, n_draws=args.n_draws, seed=args.seed + 1, top_k=args.headline_k, rank_k=rank_k)
        null_by_shape[key] = nb
        null_rankmatched[key] = nr
        thresholds[key] = {
            "tau_gain_p99": nb["E_gain"]["p99.0"], "tau_route_p99": nb["C_route"]["p99.0"],
            "f_perp_null_p99": nb["f_perp"]["p99.0"], "f_perp_analytic": nb["analytic_f_perp"],
            "E_gain_null_mean": nb["E_gain"]["mean"], "E_gain_null_std": nb["E_gain"]["std"],
            "C_route_null_mean": nb["C_route"]["mean"], "C_route_null_std": nb["C_route"]["std"],
            "rank_matched_k": rank_k, "tau_gain_p99_rankmatched": nr["E_gain"]["p99.0"],
            "tau_route_p99_rankmatched": nr["C_route"]["p99.0"],
        }
        print(f"[W12-CTRL] null {key} r={nb['rank_r']} K={rank_k}: E_gain mean={nb['E_gain']['mean']:.3f} "
              f"p99={nb['E_gain']['p99.0']:.3f}(rk {nr['E_gain']['p99.0']:.3f}) | C_route mean={nb['C_route']['mean']:.3f} "
              f"p99={nb['C_route']['p99.0']:.3f} | f_perp mean={nb['f_perp']['mean']:.4f} (analytic {nb['analytic_f_perp']:.4f})")

    # null sanity: E_gain ≈ 1, C_route ≈ 1, f_perp mean ≈ analytic (1−r²/mn) — all three headline nulls.
    null_sane = all(
        abs(nb["E_gain"]["mean"] - 1.0) < 0.5
        and (nb["C_route"] is None or abs(nb["C_route"]["mean"] - 1.0) < 0.2)
        and abs(nb["f_perp"]["mean"] - nb["analytic_f_perp"]) < 0.02
        for nb in null_by_shape.values())

    # =====================================================================
    # §5 positive control (planted splits → frac ≈ 1) + N1 identity (ΔW = 0).
    # =====================================================================
    wb_rect = matched[shape_to_key[next(s for s in shape_to_key if s[0] != s[1])]][0]
    gen = torch.Generator().manual_seed(args.seed)
    cg = GR.decompose(wb_rect, wb_rect + GR.plant_gain(wb_rect, gen))
    cr = GR.decompose(wb_rect, wb_rect + GR.plant_routing(wb_rect, gen))
    co = GR.decompose(wb_rect, wb_rect + GR.plant_out_of_span(wb_rect, gen))
    pos_ctrl = {nm: {"gain_frac": d["gain_frac"], "route_frac": d["route_frac"], "f_perp": d["f_perp"],
                     "recon_residual": d["recon_residual"]}
                for nm, d in (("pure_gain", cg), ("pure_routing", cr), ("pure_out_of_span", co))}
    pos_ok = bool(abs(cg["gain_frac"] - 1) < args.ctrl_tol and cg["route_frac"] < args.ctrl_tol
                  and abs(cr["route_frac"] - 1) < args.ctrl_tol and cr["gain_frac"] < args.ctrl_tol
                  and abs(co["f_perp"] - 1) < args.ctrl_tol
                  and max(cg["recon_residual"], cr["recon_residual"], co["recon_residual"]) < args.recon_tol)

    ident = GR.decompose(wb_rect, wb_rect.clone())
    ident_ok = bool(ident["frob_dW_sq"] == 0.0 and ident["gain_energy"] == 0.0 and ident["route_energy"] == 0.0)
    print(f"[W12-CTRL] pos_ctrl gain={cg['gain_frac']:.8f} route={cr['route_frac']:.8f} oos={co['f_perp']:.8f} "
          f"ok={pos_ok} | identity frob_dW²={ident['frob_dW_sq']} ok={ident_ok} | null_sane={null_sane}")

    controls_ok = bool(null_sane and pos_ok and ident_ok)
    result = {
        "torch": torch.__version__, "device": "cpu",
        "entities": {"PG": {"repo": PG.repo_id, "sha": pg_sha},
                     "L0": {"repo": L0.repo_id, "sha": l0_sha, "revision": L0.revision}},
        "config": {"n_draws": args.n_draws, "headline_k": args.headline_k, "seed": args.seed,
                   "degen_rtol": args.degen_rtol, "recon_tol": args.recon_tol},
        "n_matrices": len(matched), "unique_shapes": [f"{s[0]}x{s[1]}" for s in sorted(shape_to_key)],
        "decision_rule": ("per-shape: real E_gain > tau_gain_p99 (matched-random 99th pct) ⇒ gain "
                          "enriched; real C_route > tau_route_p99 ⇒ routing structured; real f_perp "
                          "> f_perp_null_p99 ⇒ out-of-span above null. BH-FDR q=0.05 over the 182 "
                          "family on empirical p from the stored null samples. Stability vs the "
                          "rank-matched null reported (§7)."),
        "null_band_by_shape": null_by_shape,            # norm-matched (primary) + raw samples for FDR
        "null_band_rankmatched": null_rankmatched,      # §7 rank-matched robustness variant
        "thresholds_by_shape": thresholds,
        "null_sane": null_sane,
        "positive_control": {"ctrl_tol": args.ctrl_tol, **pos_ctrl, "ok": pos_ok},
        "n1_identity": {"frob_dW_sq": ident["frob_dW_sq"], "ok": ident_ok},
        "CONTROLS_OK": controls_ok,
    }
    (out / "controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W12-CTRL] CONTROLS_OK={controls_ok} (null_sane={null_sane} pos_ctrl={pos_ok} identity={ident_ok})")
    if not controls_ok:
        raise RuntimeError(f"W12 CONTROLS HALT: null_sane={null_sane} pos_ctrl={pos_ok} "
                           f"identity={ident_ok}. See {out / 'controls.json'}.")
    print("[W12-CTRL] controls passed → thresholds frozen; safe to FREEZE the tag then unblind once.")


if __name__ == "__main__":
    main()
