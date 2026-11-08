"""E2b — geometry spine, the PRIMARY verdict (pi-trajectory spec §E2b). CPU, fp64, fail-loud.

On the only tensor diffable at both hops (the Gemma-1-2B LM tower), measure — SIGNED — whether FT#2
(π0) extends / undoes / builds-disjoint-from / partially-reuses the directions FT#1 (PaliGemma-1) built,
in ΔW₁'s top-r basis at r∈{8,16,32,64} vs the architecture-intrinsic Haar null, plus the §3.1 form split.
Settles the GEOMETRY of the primary claim and NOTHING about function (every verdict stamped geometry-only).

Gated on: E0a PASS ∧ E0b-load PASS ∧ E1 green ∧ the E1.5 sealed sd. Refuses to start without the three
green STATUS artifacts. Consumes the E0a init verdict + the E1.5 sealed π0 backbone sd — does NOT re-slice.

  pg run infra.aws_pg.jobs.e2_lm_geometry --type r7i.4xlarge --no-weights \
      --args "--r-grid 8,16,32,64 --n-null-draws 1000 --seed 1337" --budget-min 120
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _aws_json(key):
    """Fetch an S3 JSON via the `aws` CLI (boto3 is NOT in the pg-env job venv)."""
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--region", "us-east-1"],
                       check=True, capture_output=True)
    return json.loads(r.stdout)


def _aws_file(key, dest):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dest), "--region", "us-east-1"], check=True)


def _bh_by(pairs, q=0.05):
    """Benjamini-Hochberg-YEKUTIELI: dependency-robust FDR (the 4 estimands × 4 r are nested-projector
    dependent → plain BH is anti-conservative). BY uses q' = q / Σ_{i=1}^n (1/i). Returns rejected keys."""
    from infra.pg.w5_verdict import _bh_fdr
    valid = [(k, p) for k, p in pairs if p == p]
    n = len(valid)
    if n == 0:
        return set()
    c_n = sum(1.0 / i for i in range(1, n + 1))
    return _bh_fdr(valid, q=q / c_n)


def _seed(base, key, est, r):
    """Per-call null seed (critic minor): independent Haar draws per (tensor, estimand, r) so the empirical-p
    family is not built from one shared Gaussian sequence."""
    return int(hashlib.md5(f"{key}|{est}|{r}|{base}".encode()).hexdigest(), 16) % (2 ** 31)


def _empirical_p(value, samples, side):
    """One-sided empirical p of `value` vs the null `samples` (upper: P[null ≥ value]; lower: P[null ≤])."""
    import numpy as np
    s = np.asarray(samples, dtype=float)
    n = s.size
    if side == "upper":
        return float((np.sum(s >= value) + 1) / (n + 1))
    return float((np.sum(s <= value) + 1) / (n + 1))


def _norm_keys(sd):
    return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}


def _matched_18x7(w0, w1, w2, n_layers):
    """The 18×7 decoder matrices present in ALL THREE checkpoints (NOT the 26×7 Gemma-2 set). Asserts the
    layer count + per-(type,layer) presence explicitly (key-intersection alone would not catch a gap)."""
    keys = set(w0) & set(w1) & set(w2)
    out = {}
    for k in sorted(keys):
        if any(t in k for t in PROJ) and w0[k].ndim == 2:
            assert w0[k].shape == w1[k].shape == w2[k].shape, f"shape mismatch {k}"
            out[k] = (w0[k], w1[k], w2[k])
    expect = {f"layers.{i}.{('self_attn' if p.startswith(('q','k','v','o')) else 'mlp')}.{p}.weight"
              for i in range(n_layers) for p in PROJ}
    missing = sorted(expect - set(out))
    assert not missing, f"E2b HALT: {len(missing)} decoder matrices missing (e.g. {missing[:5]})"
    assert len(out) == n_layers * 7, f"expected {n_layers*7}, got {len(out)} (lineage 18×7 assert)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--r-grid", default="8,16,32,64")
    ap.add_argument("--n-null-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r_grid = [int(x) for x in a.r_grid.split(",")]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from infra.pg import gain_routing as gr
    from infra.pg import reuse_estimators as re
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.diff_metrics import bootstrap_ci
    from infra.pg.w8_weight_activation import subspace_overlap

    # 1) DEPENDENCY GATE — refuse without three green STATUS artifacts; record which passed.
    rep = {"dependency_gate": {}}
    e0a = _aws_json("results/e0a_initfamily_gate/out/e0a_initfamily_gate.json")
    e0b = _aws_json("results/e0b_load_gate/out/e0b_load_gate.json")
    e1 = _aws_json("results/e1_calibration/out/calibration_e1.json")
    e15 = _aws_json("results/e1_5_port/out/e1_5_port_manifest.json")
    rep["dependency_gate"] = {"e0a_verdict": e0a["verdict"], "e0b_verdict": e0b["verdict"],
                              "e1_all_pass": e1["all_pass"], "e15_ok": e15.get("gate_ok")}
    assert e0a["verdict"].startswith("PASS"), f"E0a not PASS: {e0a['verdict']} — HALT (π0-FAST fallback)"
    assert e0b["verdict"].startswith("PASS"), f"E0b-load not PASS: {e0b['verdict']} — HALT"
    assert e1["all_pass"], "E1 calibration not green — HALT the numeric program"
    assert e15.get("gate_ok"), "E1.5 port artifact gate not OK — HALT"
    rep["instrument_sha256"] = e1["provenance"]["reuse_estimators_sha256"]
    # G-power resolvability is PINNED by the FROZEN E1 calibration table (isotropic, pre-unblind) — NOT
    # re-derived from live ΔW₂ draws (that would un-pin the pre-registered gate). Keyed by int r.
    e1_gpower = {int(k): v for k, v in e1["gates"]["G-power"]["observed"]["half_width"].items()}
    rep["e1_gpower_halfwidth"] = e1_gpower

    # 2) PROVENANCE — load the three checkpoints' Gemma-1 decoder state_dicts (fp32 CPU).
    base = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.float32,
                                                device_map="cpu", attn_implementation="eager")
    w0 = _norm_keys(base.model.state_dict())
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224", torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    w1 = _norm_keys(lm.state_dict())
    # W2 = the SEALED π0 backbone sd from E1.5 (Gemma-1 decoder keys) — consume, do NOT re-slice.
    sealed_path = out / "pi0_backbone_lm_sd.safetensors"
    _aws_file("results/e1_5_port/out/pi0_backbone_lm_sd.safetensors", sealed_path)
    w2 = _norm_keys(load_file(str(sealed_path)))
    n_layers = int(base.config.num_hidden_layers)
    assert n_layers == 18, f"expected Gemma-1-2B 18 layers, got {n_layers} — wrong base model loaded"

    # 3) TENSOR SELECTION — 18×7, NOT the 26×7 Gemma-2 set; explicit layer-count assert.
    matched = _matched_18x7(w0, w1, w2, n_layers)
    rep["n_matched"] = len(matched)
    # vocab-padding issue is INERT for this set (touches only embed/lm_head) — stated, not silently skipped.
    rep["vocab_pad_note"] = "embed/lm_head excluded from the 18×7 set; vocab padding inert here"

    # 11) PREFLIGHT (first compute, fail-fast) — recon identity on a SMALL tensor (q_proj 2048×2048, NOT
    #     an FF down_proj whose n×n projector is 8.6 GB); planted round-trips; isotropic straddle.
    q0 = matched["layers.0.self_attn.q_proj.weight"]
    pf_dec = gr.decompose(q0[0], q0[1], top_k=8, check_recon=True)
    assert pf_dec["recon_residual"] < 1e-10, f"preflight recon FAIL {pf_dec['recon_residual']}"
    g = torch.Generator().manual_seed(a.seed)
    planted_gain = gr.decompose(q0[0], q0[0] + gr.plant_gain(q0[0], g), check_recon=False)
    planted_route = gr.decompose(q0[0], q0[0] + gr.plant_routing(q0[0], g), check_recon=False)
    iso = re.haar_null(torch.linalg.svdvals((q0[1] - q0[0]).double()).tolist(),
                       [1.0] * 32, 32, q0[0].shape[0], q0[0].shape[1], "rho_L", a.n_null_draws, a.seed)
    rep["preflight"] = {"recon_residual": pf_dec["recon_residual"],
                        "planted_gain_frac": planted_gain["gain_frac"], "planted_route_frac": planted_route["route_frac"],
                        "isotropic_mean": iso["mean"], "isotropic_analytic": iso["analytic"]}
    assert planted_gain["gain_frac"] > 0.95 and planted_route["route_frac"] > 0.95, "planted controls FAIL"
    iso_se = iso["std"] / (iso["n_draws"] ** 0.5)
    assert abs(iso["mean"] - iso["analytic"]) < 3 * iso_se, \
        f"isotropic straddle FAIL: MC mean {iso['mean']:.4f} vs analytic {iso['analytic']:.4f} (>3 SE) — Haar miscalibrated"

    # 4) ΔW extraction (fp64) + N2 identity (compute_weight_diff self-diff is exact-0 by construction;
    #    the 18×7 set above is the lineage-correct unit, NOT gain_routing's 26×7 Gemma-2 default).
    per_tensor = []
    for key, (a0, a1, a2) in matched.items():
        W0 = a0.double(); W1 = a1.double(); W2 = a2.double()
        dW1 = W1 - W0
        dW2 = W2 - W1
        ftype = next(t for t in PROJ if t in key)
        layer = int(key.split("layers.")[1].split(".")[0])
        rec = {"key": key, "type": ftype, "layer": layer, "shape": list(dW1.shape),
               "frob_dW1": dW1.norm().item(), "frob_dW2": dW2.norm().item()}

        # 5) FORM arm (§3.1) — gain_routing.decompose per hop in its OWN W_before basis (never one edit in
        #    two frames). z-score vs its OWN null_band; degen gate.
        d1 = gr.decompose(W0, W1, top_k=64, check_recon=False)   # FT#1 form in W0's basis
        d2 = gr.decompose(W1, W2, top_k=64, check_recon=False)   # FT#2 form in W1's basis
        rec["form_ft1"] = {"E_gain": d1["E_gain"], "C_route": d1.get("C_route"), "f_perp": d1["f_perp"],
                           "degen_route_frac": d1["degen_route_frac"]}
        rec["form_ft2"] = {"E_gain": d2["E_gain"], "C_route": d2.get("C_route"), "f_perp": d2["f_perp"],
                           "degen_route_frac": d2["degen_route_frac"]}

        # 6) REUSE arm (§3.2-3.3) — per r: rho_L/R/J + signed a, INDEPENDENT perp identity, forced floor,
        #    dW1 truncation-boundary degeneracy gate (distinct from the form in-block gate).
        sig2 = torch.linalg.svdvals(dW2).tolist()
        U, s, Vh, V, rank = gr._svd_fp64(dW1)
        per_r = {}
        for r in r_grid:
            st = re.reuse_stats(dW1, dW2, r)
            assert st["recon_resid"] < 1e-9, f"ρ_L+ρ_⊥ identity FAIL {key} r={r}: {st['recon_resid']}"
            do, di, st_top = dW1.shape[0], dW1.shape[1], s[:r].tolist()
            hL = re.haar_null(sig2, st_top, r, do, di, "rho_L", a.n_null_draws, _seed(a.seed, key, "rho_L", r))
            hR = re.haar_null(sig2, st_top, r, do, di, "rho_R", a.n_null_draws, _seed(a.seed, key, "rho_R", r))
            hJ = re.haar_null(sig2, st_top, r, do, di, "rho_J", a.n_null_draws, _seed(a.seed, key, "rho_J", r))
            hA = re.haar_null(sig2, st_top, r, do, di, "a", a.n_null_draws, _seed(a.seed, key, "a", r))
            a_hw = e1_gpower[r]                              # FROZEN E1 G-power half-width (NOT live-derived)
            cls = re.classify_reuse(st["rho_L"], st["a"], hL, hA, st["forced_floor_L"], a_hw)
            per_r[str(r)] = {
                "rho_L": st["rho_L"], "rho_R": st["rho_R"], "rho_J": st["rho_J"], "a": st["a"],
                "rho_perp_L": st["rho_perp_L"], "recon_resid": st["recon_resid"],
                "forced_floor_L": st["forced_floor_L"], "trunc_gap_rel": st["trunc_gap_rel"],
                "dw1_inblock_frac_flagged": st["dw1_inblock_frac_flagged"], "tau_gap": st["tau_gap"],
                "p_rho_L": _empirical_p(st["rho_L"], hL["samples"], "upper"),
                "p_rho_R": _empirical_p(st["rho_R"], hR["samples"], "upper"),
                "p_rho_J": _empirical_p(st["rho_J"], hJ["samples"], "upper"),
                "p_a_pos": _empirical_p(st["a"], hA["samples"], "upper"),
                "p_a_neg": _empirical_p(st["a"], hA["samples"], "lower"),
                "haar_rho_L": {k2: hL["pcts"][k2] for k2 in ("p95.0", "p99.5")} | {"mean": hL["mean"], "std": hL["std"]},
                "haar_rho_R_p95": hR["pcts"]["p95.0"], "haar_rho_J_p95": hJ["pcts"]["p95.0"],
                "a_band": [hA["pcts"]["p2.5"], hA["pcts"]["p97.5"]], "a_halfwidth_frozen": a_hw,
                "bin": cls["bin"], "a_resolvable": cls["a_resolvable"], "rho_margin_sigma": cls["rho_margin_sigma"],
            }
        rec["reuse"] = per_r

        # 8) OVERLAP diagnostic — do the (declared-independent) Form (W0 basis) and Reuse (ΔW₁ basis) frames
        #    share directions? principal-angle overlap of W0's vs ΔW₁'s top-r LEFT subspaces vs the random floor.
        Ub0, _, _, _, _ = gr._svd_fp64(W0)
        k = min(32, rank)
        rec["frame_overlap_top32"] = {"overlap": subspace_overlap(Ub0[:, :k].numpy(), U[:, :k].numpy()),
                                      "random_floor": float(k) / float(W0.shape[0])}
        per_tensor.append(rec)

    # 7) NULLS (synthetic controls, beat-the-null) — planted-UNDO + shuffle on a representative tensor.
    qk = matched["layers.9.self_attn.q_proj.weight"]
    W0q, W1q, W2q = qk[0].double(), qk[1].double(), qk[2].double()
    dW1q = W1q - W0q
    Uq, sq, Vhq, Vq, rq = gr._svd_fp64(dW1q)                               # ONE SVD (was three)
    kq = min(32, rq)
    planted_undo = -0.5 * (Uq[:, :kq] @ torch.diag(sq[:kq]) @ Vhq[:kq, :])  # ΔW₂ = −β·in-span(ΔW₁)
    undo = re.reuse_stats(dW1q, planted_undo, 32)
    gsh = torch.Generator().manual_seed(a.seed + 1)
    dq = W2q - W1q
    shuf = gr.matched_random_delta(dq.shape, dq.norm().item(), gsh)        # Haar-random matched ΔW₂
    shuf_st = re.reuse_stats(dW1q, shuf, 32)
    rep["controls"] = {"planted_undo_a": undo["a"], "planted_undo_rho_L": undo["rho_L"],
                       "shuffle_rho_L": shuf_st["rho_L"],
                       "note": "planted-UNDO must give a<0 high-ρ_L; shuffle collapses ρ_L to the band"}
    # beat-the-null controls must actually fire (else a reuse_stats sign/projection bug would pass silently).
    assert undo["a"] < -0.5, f"planted-UNDO a={undo['a']:.3f} not negative — undo bin untestable"
    assert undo["rho_L"] > 0.8, f"planted-UNDO rho_L={undo['rho_L']:.3f} not high (in-span recovery failed)"
    assert shuf_st["rho_L"] < 0.1, f"shuffle rho_L={shuf_st['rho_L']:.3f} not near the null band"

    # 9) AGGREGATION + MULTIPLICITY — per-tensor ρ_L/ρ_R/ρ_J/a are deterministic scalars → empirical-p vs N1
    #    (no bootstrap CI per tensor); BH-BY over the full {ρ_L,ρ_R,ρ_J,a±}×r×tensor family (spec §E2b-9);
    #    per-layer/pooled get bootstrap (the random axis there is the tensor population).
    fam = []
    for rec in per_tensor:
        for r in r_grid:
            pr = rec["reuse"][str(r)]
            fam.append((f"{rec['key']}|rho_L|{r}", pr["p_rho_L"]))
            fam.append((f"{rec['key']}|rho_R|{r}", pr["p_rho_R"]))
            fam.append((f"{rec['key']}|rho_J|{r}", pr["p_rho_J"]))
            fam.append((f"{rec['key']}|a_pos|{r}", pr["p_a_pos"]))
            fam.append((f"{rec['key']}|a_neg|{r}", pr["p_a_neg"]))
    rejected = _bh_by(fam, q=0.05)
    rep["family_size"] = len(fam)
    rep["n_rejected_bh_by"] = len(rejected)
    rep["rejected_by_test"] = {t: sum(1 for k in rejected if f"|{t}|" in k)
                               for t in ("rho_L", "rho_R", "rho_J", "a_pos", "a_neg")}

    # per-type + pooled (energy-weighted) aggregations with bootstrap CIs (the genuinely-random axis is
    # the population of tensors, so bootstrap over tensors is valid here).
    w = np.array([rec["frob_dW2"] ** 2 for rec in per_tensor], dtype=float)
    wsum = float(w.sum())

    def _ew(field_fn):
        return float(np.average([field_fn(rec) for rec in per_tensor], weights=w))

    agg, headline = {}, {}
    for r in r_grid:
        by_type = {}
        for t in PROJ:
            vals = [rec["reuse"][str(r)]["rho_L"] for rec in per_tensor if rec["type"] == t]
            by_type[t] = bootstrap_ci(vals) if vals else None
        all_rhoL = [rec["reuse"][str(r)]["rho_L"] for rec in per_tensor]
        all_a = [rec["reuse"][str(r)]["a"] for rec in per_tensor]
        bins = [rec["reuse"][str(r)]["bin"] for rec in per_tensor]
        # HEADLINE = energy-weighted POOLED scalars run through the SAME §4 waterfall (spec §E2b-9/10) —
        # NOT a plurality vote of per-tensor bins (which low-energy tensors would dominate, possibly
        # flipping the Tier-2 GO/NO-GO). G-power uses the FROZEN E1 half-width.
        pooled_rhoL = _ew(lambda rc: rc["reuse"][str(r)]["rho_L"])
        pooled_a = _ew(lambda rc: rc["reuse"][str(r)]["a"])
        pooled_haar_L = {"pcts": {"p95.0": _ew(lambda rc: rc["reuse"][str(r)]["haar_rho_L"]["p95.0"])},
                         "mean": _ew(lambda rc: rc["reuse"][str(r)]["haar_rho_L"]["mean"]),
                         "std": _ew(lambda rc: rc["reuse"][str(r)]["haar_rho_L"]["std"])}
        pooled_haar_a = {"pcts": {"p2.5": _ew(lambda rc: rc["reuse"][str(r)]["a_band"][0]),
                                  "p97.5": _ew(lambda rc: rc["reuse"][str(r)]["a_band"][1])}}
        pooled_ff = _ew(lambda rc: rc["reuse"][str(r)]["forced_floor_L"])
        cls = re.classify_reuse(pooled_rhoL, pooled_a, pooled_haar_L, pooled_haar_a, pooled_ff, e1_gpower[r])
        headline[str(r)] = cls["bin"]
        agg[str(r)] = {"by_type_rho_L": by_type, "pooled_energy_weighted_rho_L": pooled_rhoL,
                       "pooled_energy_weighted_a": pooled_a, "pooled_haar_L_p95": pooled_haar_L["pcts"]["p95.0"],
                       "pooled_classify": cls, "rho_L_mean_ci": bootstrap_ci(all_rhoL),
                       "a_mean_ci": bootstrap_ci(all_a),
                       "per_tensor_bin_counts": {b: bins.count(b) for b in set(bins)}}  # secondary diagnostic
    rep["aggregation"] = agg

    # 10) DECISION — pooled-energy-weighted headline bin per r. build_disjoint at ALL r ⇒ Tier-1 falsifies reuse.
    disjoint_all_r = all(headline[str(r)] == "build_disjoint" for r in r_grid)
    rep["headline_bin_per_r"] = headline
    rep["tier1_verdict"] = ("build_disjoint (reuse FALSIFIED at Tier-1 — do NOT spend Tier-2 GPU; "
                            "negative result is publishable, geometry-only)" if disjoint_all_r
                            else f"non-disjoint: {headline} — GEOMETRY-ONLY; Tier-2 function arm gates promotion")
    rep["GEOMETRY_ONLY"] = True

    (out / "e2_geometry.json").write_text(json.dumps({"summary": rep, "per_tensor": per_tensor}, indent=2))
    print("=" * 78)
    for r in r_grid:
        print(f"  r={r:3d}  pooled ρ_L={agg[str(r)]['pooled_energy_weighted_rho_L']:.4f} "
              f"(haar p95 {agg[str(r)]['pooled_haar_L_p95']:.4f})  pooled a={agg[str(r)]['pooled_energy_weighted_a']:+.3f}  "
              f"headline={headline[str(r)]}  per-tensor={agg[str(r)]['per_tensor_bin_counts']}")
    print("=" * 78)
    print("E2B_OK", json.dumps({"tier1": rep["tier1_verdict"][:60], "n_rejected": len(rejected)}))


if __name__ == "__main__":
    main()
