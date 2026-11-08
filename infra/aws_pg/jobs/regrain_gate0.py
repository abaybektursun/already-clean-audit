"""REGRAIN Gate-0 (spec §12.0; GPU) — WRITE-FIRST at every stage. Verifies at the artifacts,
on real weights, BEFORE freeze:

  A. Ledger: every NEW regrain_core surface exists; every REUSED grain_bsf surface still does.
  B. Read banks located + valid; sites resolved from artifacts (L0 primary, L14 secondary).
  C. Bank sha: the banked m=1 L0 dictionary loads, its data-manifest sha matches the harvest,
     and its v2 gate verdict is read from grain_t3diag (A5a) — never from the v1 grain_t2 gates.
  D. SEALED-EXPOSURE AUDIT (§12.0): no GRAIN job ever evaluated the sealed half. Proven three
     ways: (i) no grain_g_sealed* prefix exists in S3; (ii) grain_gdisc_saction ran in
     `grain1_descriptive` mode with no sealed cells; (iii) the grain| halves rebuild bit-exactly
     from the live item stream. Clean ⇒ REUSE the halves (§2); dirty ⇒ re-salt `regrain|`.
  E. PG1 loads EAGER at the invcore-FUNC-pinned sha; ppl gate reproduces the artifact at the
     PROGRAM-FROZEN n_seq=120.
  F. Hook inertness (bitwise) + rotated-error hook NORM-exactness on REAL activations.
  G. C0 pre-gate: the banked m=1 dictionary re-gated on a fresh discovery slice at n_seq=120
     (GRAIN gated at n_seq=60 — its Δppl is not comparable and is never quoted), with the
     rotated-error null cell measured beside the true splice.
  H. S_vl re-estimation draws present (Arm D adjudicates there, A5b).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_gate0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_gate0 --budget-min 180 --args "--n-probe 192"
"""
import argparse
import json
import os
import pathlib
import subprocess


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-probe", type=int, default=192)
    ap.add_argument("--n-gate-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "regrain_gate0_v1", "seed": a.seed}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    RS.install_pg()

    # ---------------- A: ledger ----------------
    import dictionary_learning
    new_surfaces = ("GrassmannBSF", "train_grassmann_bsf", "rate_bits", "log2_binom",
                    "mdl_bits_per_token", "grassmann_free_params", "free_decoder_params",
                    "haar_rotation", "rotate_error", "r2_of", "quantize_step",
                    "make_pre_hook_rotated_splice", "make_pre_hook_quantize",
                    "make_pre_hook_gaussian", "hook_at", "usage_strata", "usage_matched_union",
                    "uniform_random_union", "surrogate_at_canonicity", "top_aligned_atoms",
                    "canonicity_matched_jaccard_null", "adjudicate_atom_handle",
                    "adjudicate_draws", "adjudicate_sign", "adjudicate_stream", "stable_rank",
                    "stable_rank_haar_band", "gen_planted_circle", "blocks_covering_usage",
                    "b_dec_offset_ratio", "sqrt_d_scale", "ScaledAE", "decoder_matrix",
                    "save_bank_any", "load_bank_any")
    for s in new_surfaces:
        assert hasattr(RC, s), f"regrain_core missing NEW surface {s}"
    for s in ("BlockTopKSAE", "train_block_sae", "block_quality", "union_Q", "block_alignment",
              "block_usage", "block_stable_rank", "jaccard", "pairwise_jaccard", "jaccard_null",
              "make_pg1_pre_hook_splice", "make_pg1_pre_hook_passthrough", "load_dict_bank"):
        assert hasattr(GB, s), f"grain_bsf lost REUSED surface {s}"
    assert "signed" in GB.BlockTopKSAE.__init__.__code__.co_varnames, "signed path absent"
    rep["ledger"] = {"dictionary_learning": getattr(dictionary_learning, "__version__",
                                                    "unpinned-git"),
                     "regrain_core_surfaces": len(new_surfaces), "grain_bsf_reused": True}
    _w()

    # ---------------- B: read banks + sites ----------------
    banks = RS.load_read_banks("/tmp/regrain")
    Lp, Ls = banks["saction"]["L"], banks["svl"]["L"]
    rep["sites"] = {"primary": Lp, "secondary": Ls}
    rep["banks"] = {r: {"L": banks[r]["L"], "shape": list(banks[r]["V"].shape)} for r in banks}
    _w()

    # ---------------- C: bank sha + the v2 gate artifact ----------------
    gates_v2 = RS.load_v2_gates("/tmp/regrain")
    meta = json.loads(RS.fetch(f"{RS.GRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    bank_name = RS.dict_bank_name("primary", 1, 0)
    ae_m1, cfg_m1, q_m1, msha = GB.load_dict_bank(
        RS.fetch(f"{RS.GRAIN_DICTS_L0}/{bank_name}", f"/tmp/regrain/{bank_name}"))
    assert msha == meta["manifest_sha"], \
        f"banked dict manifest {msha[:12]} != harvest {meta['manifest_sha'][:12]}"
    assert gates_v2["manifest_sha"] == meta["manifest_sha"]
    assert cfg_m1["m"] == 1 and cfg_m1.get("signed", False) is False
    seeds_present = [s for s in (0, 1, 2)
                     if subprocess.run(["aws", "s3", "ls",
                                        f"{RS.GRAIN_DICTS_L0}/{RS.dict_bank_name('primary',1,s)}",
                                        "--region", "us-east-1"],
                                       capture_output=True).returncode == 0]
    rep["c0_banks"] = {"m1_seeds_present": seeds_present, "manifest_sha": msha,
                       "fvu_m1_s0": q_m1["fvu"], "threshold": cfg_m1["threshold"],
                       "gate_pass_v2": gates_v2["configs"]["m1_s0"]["gate_pass"],
                       "gate_pass_v1": gates_v2["configs"]["m1_s0"].get("gate_pass_v1"),
                       "seeds_to_retrain": [s for s in (1, 2) if s not in seeds_present]}
    assert seeds_present == [0], f"A5a assumed only seed 0 exists; found {seeds_present}"
    _w()

    # ---------------- D: sealed-exposure audit ----------------
    ls = subprocess.run(["aws", "s3", "ls", f"{RS.S3_BUCKET}/results/", "--region", "us-east-1"],
                        capture_output=True, text=True)
    grain_prefixes = [l.split()[-1].rstrip("/") for l in ls.stdout.splitlines()
                      if "grain_" in l and "regrain" not in l]
    sealed_prefixes = [p for p in grain_prefixes if "sealed" in p]
    gdisc = json.loads(RS.fetch(RS.GRAIN_GDISC, "/tmp/regrain/g_disc.json").read_text())
    vqa, coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the GRAIN gate0 bank — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    halves_match = sorted(int(i) for i in split["conf_idx"]) == sorted(it["sealed_idx"])
    clean = (not sealed_prefixes) and gdisc["mode"] == "grain1_descriptive" \
        and "promoted_cells" not in gdisc and halves_match
    rep["sealed_exposure_audit"] = {
        "grain_prefixes": grain_prefixes, "sealed_prefixes_found": sealed_prefixes,
        "gdisc_mode": gdisc["mode"], "gdisc_has_sealed_cells": "promoted_cells" in gdisc,
        "halves_rebuild_exact": halves_match, "clean": bool(clean),
        "decision": "REUSE grain| halves" if clean else "RE-SALT regrain|",
        "n_disc": int(split["n_disc"]), "n_sealed": int(split["n_conf"])}
    assert clean, "sealed half was exposed by GRAIN — the re-salt rule fires (§12 pivot)"
    _w()

    # ---------------- E: PG1 + ppl gate @ the program-frozen n_seq ----------------
    sha, ppl_ref = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    rep["pg1"] = {"repo": IP.PG1_REPO, "sha": sha, "n_layers": len(layers), "d": IP.D}
    pp = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
    rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref,
                       "n_seq": RS.N_PPL_SEQ, "tol": RS.PPL_TOL}
    assert abs(pp["ppl"] - ppl_ref) <= RS.PPL_TOL, \
        f"ppl gate RED: {pp['ppl']:.3f} vs artifact {ppl_ref:.3f} (sdpa trap / wiring)"
    base_ppl = pp["ppl"]
    _w()

    # floors + intact on a discovery probe slice
    disc_rows = [rows[i] for i in split["disc_idx"]]
    probe_rows = disc_rows[: a.n_probe]
    stats = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[Lp]["hbar"]
    intact = IP.eval_pg1(model, proc, vqa, probe_rows, "cuda")
    floor_eval = IP.eval_pg1(model, proc, vqa, probe_rows, "cuda",
                             ctx=IP.replace_at(layers, Lp, hbar))
    floor = float(np.asarray(floor_eval["per_item"], float).mean())
    dyn = intact["score"] - floor
    assert dyn > 0.05, f"no dynamic range vs dead floor ({dyn:.4f}) — halt"
    rep["vqa_probe"] = {"intact": intact["score"], "intact_ci": intact["ci95"],
                        "dead_floor_L0": floor, "dynamic_range": dyn, "n": len(probe_rows)}
    _w()

    # ---------------- F: inertness + rotated-error exactness on REAL activations ----------------
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    inp = vqa_forward_batches(proc, vqa, probe_rows[:8], "cuda")[0]
    with torch.inference_mode():
        base_logits = model(**inp).logits[:, -1, :].float().cpu()
    h = layers[Lp].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(), with_kwargs=True)
    with torch.inference_mode():
        hooked = model(**inp).logits[:, -1, :].float().cpu()
    h.remove()
    with torch.inference_mode():
        after = model(**inp).logits[:, -1, :].float().cpu()
    assert torch.equal(base_logits, hooked), "pass-through hook NOT a bitwise no-op"
    assert torch.equal(base_logits, after), "hook not inert after removal"

    # rotated-error NORM exactness measured on the real entering residual at L0
    cap = []
    hc = layers[Lp].register_forward_pre_hook(RS.make_row_capture(cap, IP.D), with_kwargs=True)
    with torch.inference_mode():
        model(**inp)
    hc.remove()
    Xr = cap[0].float()
    ae_dev = ae_m1.to("cuda").eval()
    Q = RC.haar_rotation(IP.D, RS.ROT_SEED, device="cuda", dtype=torch.float32)
    # exactness of the rotation is an fp64 property; the OPERATIONAL claim is that the fp32 hook
    # preserves each row's error norm, which is measured directly on real activations below.
    Qd = Q.double()
    orth_err = float((Qd.T @ Qd - torch.eye(IP.D, device="cuda", dtype=torch.float64)).abs().max())
    with torch.inference_mode():
        xr = Xr.to("cuda")
        rec = ae_dev.decode(ae_dev.encode(xr, use_threshold=True))
        rot = RC.rotate_error(xr, rec, Q)
    n_true = (rec - xr).norm(dim=1)
    n_rot = (rot - xr).norm(dim=1)
    norm_err = float(((n_true - n_rot).abs() / n_true.clamp_min(1e-6)).max())
    del Qd
    assert orth_err < 1e-8, f"Haar rotation not orthogonal in fp64 ({orth_err:.2e})"
    assert norm_err < 1e-3, f"rotated-error changes the per-row error norm ({norm_err:.2e})"
    rep["rotated_error_hook"] = {"haar_orth_err": orth_err, "rel_norm_err": norm_err,
                                 "seed": RS.ROT_SEED, "n_rows": int(Xr.shape[0])}
    _w()

    # ---------------- G: C0 pre-gate at n_seq=120 (true splice vs rotated-error null) ---------
    gate_rows = disc_rows[: a.n_gate_items]
    intact_g = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda")
    cell = gates_v2["configs"]["m1_s0"]
    r_e = int(cell["em_band"]["r_e"])
    em_dvqa, em_dppl = [], []
    for s in RS.EM_SEEDS:
        Qem = RS.em_span(r_e, s, IP.D)
        r = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=IP.ablate_at(layers, Lp, Qem, hbar))
        em_dvqa.append(r["score"] - intact_g["score"])
        with IP.ablate_at(layers, Lp, Qem, hbar):
            em_dppl.append(IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"] - base_ppl)
    spl = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=GB.splice_at(layers, Lp, ae_dev))
    with GB.splice_at(layers, Lp, ae_dev):
        spl_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    rot_hook = RC.make_pre_hook_rotated_splice(ae_dev, Q)
    rot = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=RC.hook_at(layers, Lp, rot_hook))
    with RC.hook_at(layers, Lp, rot_hook):
        rot_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]

    dvqa, dppl = spl["score"] - intact_g["score"], spl_ppl - base_ppl
    vband = {"mean": float(np.mean(em_dvqa)), "halfwidth": float((max(em_dvqa) - min(em_dvqa)) / 2),
             "deltas": em_dvqa}
    pband = {"mean": float(np.mean(em_dppl)), "halfwidth": float((max(em_dppl) - min(em_dppl)) / 2),
             "deltas": em_dppl}
    vqa_ok = dvqa >= vband["mean"] - max(vband["halfwidth"], 0.01)
    ppl_ok = dppl <= pband["mean"] + max(pband["halfwidth"], RS.PPL_TOL)
    rep["c0_gate"] = {
        "n_seq": RS.N_PPL_SEQ, "n_items": len(gate_rows), "r_e": r_e,
        "intact": intact_g["score"], "base_ppl": base_ppl,
        "splice": {"vqa": spl["score"], "dvqa": dvqa, "ppl": spl_ppl, "dppl": dppl},
        "rotated_error_null": {"vqa": rot["score"], "dvqa": rot["score"] - intact_g["score"],
                               "ppl": rot_ppl, "dppl": rot_ppl - base_ppl},
        "em_vqa_band": vband, "em_ppl_band": pband,
        "gate_parts": {"vqa_ok": bool(vqa_ok), "ppl_ok": bool(ppl_ok)},
        "gate_pass": bool(vqa_ok and ppl_ok),
        "grain_v1_numbers_not_quoted": "GRAIN T3 used n_seq=60; recomputed here at 120"}
    _w()
    ae_dev = ae_m1.to("cpu")
    del Q
    torch.cuda.empty_cache()

    # ---------------- H: S_vl draws (Arm D adjudicates there) ----------------
    draws = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/svl_draws.pt", "/tmp/regrain/svl_draws.pt"),
                       map_location="cpu", weights_only=False)
    rep["svl_draws"] = {"n": len(draws), "n_pos": {str(k): int(draws[k]["n_pos"]) for k in draws},
                        "arm_d_adjudicates_here": len(draws) >= 3,
                        "saction_draws": 2,
                        "note": "A5b: a 3rd S_action draw needs the pi0.5 ACTION loss; "
                                "accumulate_gram_pre is a VL-loss Gram. S_action stays n=2 "
                                "DESCRIPTIVE; Arm D adjudicates on S_vl."}
    assert len(draws) >= 3, "S_vl draws < 3 — Arm D cannot adjudicate anywhere"

    # Gate-0 greenness is a PROGRAM property. A failing C0 re-gate is an Arm-C finding
    # (§12 pivot: retrain from banked shards, one retry, else INSTRUMENT-DRIFT) — recorded
    # here, never silently fatal.
    print("REGRAIN_GATE0_OK", json.dumps(
        {"sites": rep["sites"], "ppl": rep["ppl_gate"]["ppl"], "dyn": dyn,
         "c0_gate_pass": rep["c0_gate"]["gate_pass"], "c0_dvqa": dvqa, "c0_dppl": dppl,
         "rot_dvqa": rep["c0_gate"]["rotated_error_null"]["dvqa"],
         "sealed_clean": clean, "seeds_to_retrain": rep["c0_banks"]["seeds_to_retrain"]},
        default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
