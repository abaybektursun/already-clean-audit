"""W12 §9 STEP-1 PREFLIGHT GATE (fail-fast, before any real ΔW decomposition).

Three sub-gates (prereg §9 step 1). CPU only — state_dict math + tokenizer parity, NO forward
pass (the decomposition is a pure weight-space measurement; gate 1a's parity loads tokenizers).

  (1a) base-init match — base gemma-2-2b is PG2-pt's EXACT LM init: PG language_model decoder
       keys == gemma-2-2b (g2.model) decoder keys with ZERO shape mismatch over the 26×7 weight
       matrices + norms (embed_tokens excluded — extended multimodal vocab, §2), AND tokenizer
       parity (gemma-2-2b vs PG2 processor on shared-vocab clean text). Records both repo SHAs.
       Mismatch → HALT: a wrong BEFORE contaminates ΔW + every split (the dominant W1/W12 confound, §8).
  (1b) pipeline-port — recompute the W1 LM ΔW (PG language_model vs gemma-2-2b g2.model,
       embed_tokens handled separately) via compute_weight_diff (the W1 path VERBATIM) and match
       the cached W1 reference summary (rel_frob_mean, eff_rank_mean) + per-type (q_proj/down_proj)
       READ FROM S3 (not transcribed — CLAUDE.md provenance) within tol. Mismatch → HALT (port invalid).
  (1c) fp64 reconstruction identity — gain_routing.decompose on real LM matrices:
       ‖U M Vᵀ − ΔW_∥‖/‖ΔW‖ < recon_tol (the §9-1c fp64 spectral-roundoff gate). PLUS a §5
       synthetic positive-control VALIDATION (planted pure-gain / pure-routing / pure-out-of-span
       splits must be recovered to frac ≈ 1) — asserts the decomposition arithmetic is correct
       before any real split is trusted. (PREFLIGHT_OK keys on the 3 registered gates; the
       positive control is reported under `validations` and fail-loud asserted.)

Any fail → HALT (raise; no try/except swallowing).

    python -m infra.aws_pg.jobs.w12_preflight --out <dir> \
        --reference-s3 s3://pg-lineage-769063704696/reference/w1_real_dw/w1_real_dw_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def _build_matched_lm(pg_lm_sd, l0_sd):
    """Mirror modal_pg_w1_real's LM-ΔW key alignment: shared NON-embed keys, shape-equal.
    Raises on any non-embed shape mismatch or unexpected-side-only key (architecture regression)."""
    pg_keys, l0_keys = set(pg_lm_sd), set(l0_sd)
    unexpected_pg = [k for k in (pg_keys - l0_keys) if "embed_tokens" not in k and "lm_head" not in k]
    unexpected_l0 = [k for k in (l0_keys - pg_keys) if "embed_tokens" not in k and "lm_head" not in k]
    if unexpected_pg or unexpected_l0:
        raise RuntimeError(f"LM key mismatch beyond embed_tokens: pg_only={unexpected_pg} "
                           f"l0_only={unexpected_l0}. Architecture misalignment → HALT.")
    pg_matched, l0_matched, shape_bad = {}, {}, []
    for k in sorted(pg_keys & l0_keys):
        if "embed_tokens" in k or "lm_head" in k:
            continue
        if pg_lm_sd[k].shape != l0_sd[k].shape:
            shape_bad.append({"key": k, "pg": list(pg_lm_sd[k].shape), "l0": list(l0_sd[k].shape)})
            continue
        pg_matched[k] = pg_lm_sd[k]
        l0_matched[k] = l0_sd[k]
    return pg_matched, l0_matched, shape_bad


def _by_type_means(per_tensor):
    """Bucket per_tensor records by gain_routing.fine_type → {type: {rel_frob_mean, eff_rank_mean, n}}."""
    from collections import defaultdict
    from infra.pg.gain_routing import fine_type
    buckets = defaultdict(list)
    for r in per_tensor:
        buckets[fine_type(r["key"])].append(r)
    out = {}
    for t, recs in buckets.items():
        rfs = [r["rel_frob"] for r in recs]
        eff = [r["eff_rank"] for r in recs if r.get("eff_rank") is not None]
        out[t] = {"n": len(recs), "rel_frob_mean": sum(rfs) / len(rfs),
                  "eff_rank_mean": (sum(eff) / len(eff)) if eff else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-s3",
                    default="s3://pg-lineage-769063704696/reference/w1_real_dw/w1_real_dw_summary.json")
    ap.add_argument("--tol-frob", type=float, default=1e-3, help="abs tol on rel_frob_mean reproduction")
    ap.add_argument("--tol-effrank", type=float, default=0.5, help="abs tol on eff_rank_mean reproduction")
    ap.add_argument("--recon-tol", type=float, default=1e-10, help="§9-1c fp64 reconstruction-identity tol")
    ap.add_argument("--ctrl-tol", type=float, default=1e-6, help="§5 positive-control planted-split tol (|frac−target|)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # MUST precede any huggingface_hub/transformers import (run_job.sh sets HF_HUB_OFFLINE=1, which
    # hf caches into a module constant at import → OfflineModeIsEnabled on the gated metadata HEAD).
    # diff_harvest's top is light (no hf).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from huggingface_hub import HfApi
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              PaliGemmaForConditionalGeneration, PaliGemmaProcessor)

    from infra.pg import gain_routing as GR
    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES, resolve_submodule_paths
    from infra.pg.weight_diff import compute_weight_diff

    device = "cpu"   # W12 is a pure weight-space measurement; no GPU needed.
    print(f"[W12-PRE] torch={torch.__version__} device={device}")

    api = HfApi()
    PG, L0 = ENTITIES["PG"], ENTITIES["L0"]
    pg_sha = api.model_info(PG.repo_id).sha
    l0_sha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
    print(f"[W12-PRE] PG={PG.repo_id}@{pg_sha}  L0={L0.repo_id}@{l0_sha}")

    # --- read cached W1 reference (instance-role aws cli; fail loud, no transcription) ---
    ref_local = out / "reference_w1_real_dw_summary.json"
    subprocess.run(["aws", "s3", "cp", args.reference_s3, str(ref_local), "--region", "us-east-1"],
                   check=True)
    ref_lm = json.loads(ref_local.read_text())["lm"]
    ref_frob = float(ref_lm["summary"]["rel_frob_mean"])
    ref_eff = float(ref_lm["summary"]["eff_rank_mean"])
    ref_by_type = ref_lm["by_matrix_type_fine"]
    print(f"[W12-PRE] reference LM rel_frob_mean={ref_frob:.6f} eff_rank_mean={ref_eff:.4f}")

    # --- load the two LM decoders (CPU) ---
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG.repo_id, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    paths = resolve_submodule_paths(pg)
    pg_lm_sd = {k: v.clone() for k, v in pg.get_submodule(paths["language_model"]).state_dict().items()}
    del pg
    g2 = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager")
    l0_sd = {k: v.clone() for k, v in g2.model.state_dict().items()}   # g2.model = decoder, aligns w/ PG LM
    del g2

    pg_matched, l0_matched, shape_bad = _build_matched_lm(pg_lm_sd, l0_sd)

    # =====================================================================
    # (1a) base-init match — decoder key/shape alignment + tokenizer parity.
    # =====================================================================
    n_weight_matrices = sum(1 for k in pg_matched if GR.fine_type(k) in GR.PROJECTION_TYPES
                            and pg_matched[k].ndim == 2)
    matrices_ok = bool(n_weight_matrices == 26 * 7)   # 26 layers × {q,k,v,o,gate,up,down}
    shape_ok = bool(not shape_bad)

    tok_g = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    tok_pg = PaliGemmaProcessor.from_pretrained(PG.repo_id).tokenizer
    parity = TE.tokenizer_parity(tok_g, tok_pg)

    gate_1a = bool(shape_ok and matrices_ok and parity["match"])
    print(f"[W12-PRE][1a] decoder matrices={n_weight_matrices} (expect 182) shape_mismatch={len(shape_bad)} "
          f"tokenizer_parity={parity['match']} → ok={gate_1a}")

    # =====================================================================
    # (1b) pipeline-port — reproduce the W1 LM ΔW summary (compute_weight_diff VERBATIM).
    # =====================================================================
    lm_diff = compute_weight_diff(l0_matched, pg_matched, assert_zero=False)
    rec_frob = float(lm_diff["summary"]["rel_frob_mean"])
    rec_eff = float(lm_diff["summary"]["eff_rank_mean"])
    d_frob, d_eff = abs(rec_frob - ref_frob), abs(rec_eff - ref_eff)
    by_type = _by_type_means(lm_diff["per_tensor"])
    type_checks = {}
    for t in ("q_proj", "down_proj"):
        rt = ref_by_type[t]
        df = abs(by_type[t]["rel_frob_mean"] - rt["rel_frob_mean"])
        de = abs(by_type[t]["eff_rank_mean"] - rt["eff_rank_mean"])
        type_checks[t] = {"d_rel_frob": df, "d_eff_rank": de,
                          "ok": bool(df <= args.tol_frob and de <= args.tol_effrank)}
    gate_1b = bool(d_frob <= args.tol_frob and d_eff <= args.tol_effrank
                   and all(c["ok"] for c in type_checks.values()))
    print(f"[W12-PRE][1b] rel_frob {rec_frob:.6f} vs {ref_frob:.6f} |Δ|={d_frob:.2e}; "
          f"eff_rank {rec_eff:.4f} vs {ref_eff:.4f} |Δ|={d_eff:.2e}; "
          f"per_type_ok={all(c['ok'] for c in type_checks.values())} → ok={gate_1b}")

    # =====================================================================
    # (1c) fp64 reconstruction identity on real LM matrices + §5 positive-control validation.
    # =====================================================================
    probe_keys = ["layers.0.self_attn.q_proj.weight",    # registered 1c primary (2048×2304)
                  "layers.0.self_attn.o_proj.weight",     # other orientation (2304×2048)
                  "layers.0.mlp.down_proj.weight"]        # strongly rectangular (2304×9216)
    recon = {}
    for k in probe_keys:
        assert k in pg_matched, f"1c probe key absent: {k}"
        d = GR.decompose(l0_matched[k], pg_matched[k])
        # NOTE (blind protocol): split values (E_gain/f_perp) are stored in the JSON artifact only,
        # NOT printed — the operator must not see real-ΔW split values pre-freeze (§9-2 firewall).
        recon[k] = {"shape": d["shape"], "recon_residual": d["recon_residual"],
                    "f_perp": d["f_perp"], "E_gain": d["E_gain"],
                    "ok": bool(d["recon_residual"] < args.recon_tol)}
        print(f"[W12-PRE][1c] {k} shape={d['shape']} recon_residual={d['recon_residual']:.2e} "
              f"(tol {args.recon_tol}) → ok={recon[k]['ok']}")
    gate_1c = bool(all(r["ok"] for r in recon.values()))

    # §5 positive-control validation (asserted fail-loud; not part of the 3-gate PREFLIGHT_OK key).
    wb = l0_matched["layers.0.self_attn.q_proj.weight"]   # rectangular ⇒ out-of-span control valid
    gen = torch.Generator().manual_seed(args.seed)
    cg = GR.decompose(wb, wb + GR.plant_gain(wb, gen))
    cr = GR.decompose(wb, wb + GR.plant_routing(wb, gen))
    co = GR.decompose(wb, wb + GR.plant_out_of_span(wb, gen))
    validations = {
        "pure_gain":        {"gain_frac": cg["gain_frac"], "route_frac": cg["route_frac"],
                             "f_perp": cg["f_perp"], "recon_residual": cg["recon_residual"]},
        "pure_routing":     {"gain_frac": cr["gain_frac"], "route_frac": cr["route_frac"],
                             "f_perp": cr["f_perp"], "recon_residual": cr["recon_residual"]},
        "pure_out_of_span": {"gain_frac": co["gain_frac"], "route_frac": co["route_frac"],
                             "f_perp": co["f_perp"], "recon_residual": co["recon_residual"]},
    }
    print(f"[W12-PRE][1c-ctrl] gain→gain_frac={cg['gain_frac']:.6f} | "
          f"routing→route_frac={cr['route_frac']:.6f} | oos→f_perp={co['f_perp']:.6f}")
    assert abs(cg["gain_frac"] - 1.0) < args.ctrl_tol and cg["route_frac"] < args.ctrl_tol, \
        f"§5 pure-gain control failed: {validations['pure_gain']}"
    assert abs(cr["route_frac"] - 1.0) < args.ctrl_tol and cr["gain_frac"] < args.ctrl_tol, \
        f"§5 pure-routing control failed: {validations['pure_routing']}"
    assert abs(co["f_perp"] - 1.0) < args.ctrl_tol, \
        f"§5 pure-out-of-span control failed: {validations['pure_out_of_span']}"

    ok = bool(gate_1a and gate_1b and gate_1c)
    result = {
        "torch": torch.__version__, "device": device,
        "entities": {"PG": {"repo": PG.repo_id, "sha": pg_sha},
                     "L0": {"repo": L0.repo_id, "sha": l0_sha, "revision": L0.revision}},
        "reference_s3": args.reference_s3,
        "gate_1a_base_init_match": {
            "n_decoder_weight_matrices": n_weight_matrices, "expected": 26 * 7,
            "matrices_ok": matrices_ok, "shape_mismatch": shape_bad, "shape_ok": shape_ok,
            "tokenizer_parity": parity, "ok": gate_1a},
        "gate_1b_w1_port": {
            "reference_rel_frob_mean": ref_frob, "recomputed_rel_frob_mean": rec_frob, "d_rel_frob": d_frob,
            "reference_eff_rank_mean": ref_eff, "recomputed_eff_rank_mean": rec_eff, "d_eff_rank": d_eff,
            "tol_frob": args.tol_frob, "tol_effrank": args.tol_effrank,
            "n_tensors": lm_diff["summary"]["n_tensors_total"], "per_type": type_checks, "ok": gate_1b},
        "gate_1c_fp64_recon_identity": {"recon_tol": args.recon_tol, "probes": recon, "ok": gate_1c},
        "validations_positive_control": {"ctrl_tol": args.ctrl_tol, **validations},
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W12-PRE] PREFLIGHT_OK={ok}  (1a={gate_1a} 1b={gate_1b} 1c={gate_1c})")
    if not ok:
        raise RuntimeError(
            f"W12 PREFLIGHT HALT: 1a_base_init={gate_1a} 1b_w1_port={gate_1b} "
            f"1c_recon_identity={gate_1c}. See {out / 'preflight.json'}.")
    print("[W12-PRE] gate passed → safe to develop the gain/routing decomposition on control data.")


if __name__ == "__main__":
    main()
