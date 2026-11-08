"""INHERIT Gate-0 (§9.0, wiring + provenance; GPU) — WRITE-FIRST at every stage. Verifies at the
artifacts, on real weights, BEFORE freeze:

  A. V-bank valid + covers the frozen k-grid; layer/k selection resolved from CTRLIFACE artifacts;
  B. U_unt derived at the selected layers from the invcore scratch ΔWs (orthonormal; bottom band
     near-zero relative to the spectrum top);
  C. PG1 loads EAGER; 18 decoder layers, d=2048 (the PG1 compartment facts); resolved sha recorded;
  D. eager blank-image wikitext ppl REPRODUCES the invcore-FUNC artifact value (|Δ| ≤ tol; the
     sdpa-trap tripwire) — the collateral instrument is wired;
  E. feasibility probe on discovery-half items: intact VQA + CIDEr ≫ scrambled floor (dynamic
     range), sd(intact) → the frozen §4 power rule fixes n per task;
  F. hook inertness + teeth smoke: install/remove leaves logits bit-identical; a rank-16 random
     ablation at L* runs the generate path without error.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_gate0 --type p5.4xlarge,g6e.8xlarge,g6e.4xlarge \
      --no-weights --s3-prefix results/inherit_gate0 --budget-min 120 --args "--n-probe 256"
"""
import argparse
import json
import pathlib

PPL_TOL = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--n-coco", type=int, default=128)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"ppl_tol": PPL_TOL, "seed": a.seed}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import inherit_core as IC
    from infra.pg import inherit_pg1 as IP

    SH.install_pg()

    # --- A: bank + selection ---
    bases, _pi_hbars, pi_traces = SH.load_bank()
    sel = SH.resolve_selection()
    rep["bank"] = {"layers": len(bases), "shape": list(bases[0].shape)}
    rep["selection"] = sel
    assert max(sel["k_grid"]) <= bases[0].shape[1]
    # audit item 7: the em-rank the CTRLIFACE controls/confirm jobs ACTUALLY used — E_target from a
    # small activation capture over tr(C) from the 48-batch bank (mixed rows). Recorded here from
    # the artifacts verbatim so the CTRLIFACE disclosure can quote a number, not a suspicion.
    cc = json.loads(pathlib.Path("/tmp/inherit/ctrliface_controls.json").read_text())
    from infra.pg.ctrliface_core import energy_matched_rank
    r_e_used = energy_matched_rank(cc["E_target_Lk"], pi_traces[int(cc["L"])], 2048)
    rep["ctrliface_em_rank_audit"] = {"E_target_Lk": cc["E_target_Lk"],
                                      "bank_trace_L": pi_traces[int(cc["L"])],
                                      "k_intended": int(cc["k"]), "r_e_used": int(r_e_used)}
    _w()

    # --- B: U_unt at the selected layers ---
    L_sel = sorted({sel["primary"], sel["secondary"], *sel["anchors"]})
    uunt = SH.derive_uunt(L_sel, k=128)
    rep["uunt"] = {}
    for L, (U, evals) in uunt.items():
        err = float((U.T @ U - torch.eye(128, dtype=torch.float64)).abs().max())
        assert err < 1e-8, f"U_unt L{L} not orthonormal"
        rep["uunt"][str(L)] = {"orthonormal_err": err, "bottom4": [float(x) for x in evals[:4]]}
    torch.save({L: uunt[L][0] for L in uunt}, out / "uunt_selected.pt")
    _w()

    # --- C: PG1 loads (eager) at the invcore-FUNC-pinned sha (exact instrument reproduction) ---
    func = json.loads(SH.fetch(SH.INVCORE_FUNC_JSON, "/tmp/inherit/invcore_func.json").read_text())
    pg1_sha = func["base_sha"]
    model, proc = IP.load_pg1(sha=pg1_sha)
    layers = IP.decoder_layers(model)
    d_probe = int(model.get_input_embeddings().weight.shape[1])
    assert d_probe == IP.D, f"PG1 hidden {d_probe} != {IP.D}"
    rep["pg1"] = {"repo": IP.PG1_REPO, "sha": pg1_sha,
                  "n_layers": len(layers), "d": d_probe}
    _w()

    # --- D: eager ppl gate vs the invcore-FUNC artifact ---
    ppl_ref = float(func["capability_baseline"]["ppl"])
    pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref,
                       "n_windows": pp["n_windows"]}
    assert abs(pp["ppl"] - ppl_ref) <= PPL_TOL, \
        f"ppl gate RED: {pp['ppl']:.3f} vs artifact {ppl_ref:.3f} (sdpa trap / wiring)"
    _w()

    # --- E: feasibility probe (discovery half only) ---
    vqa, coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, a.n_probe * 2)
    split = IC.sha1_split(hashes, name="gate0_vqa")
    disc_rows = [rows[i] for i in split["disc_idx"]][: a.n_probe]
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    floor = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", scramble_seed=a.seed)
    dyn = intact["score"] - floor["score"]
    sd = float(np.std(np.asarray(intact["per_item"], float)))
    n_frozen = IC.choose_n_items(sd, dyn)
    rep["vqa_probe"] = {"intact": intact["score"], "intact_ci": intact["ci95"],
                        "floor": floor["score"], "floor_ci": floor["ci95"],
                        "dynamic_range": dyn, "sd_intact": sd, "n_frozen": n_frozen}
    assert dyn > 0.05, f"VQA dynamic range {dyn:.4f} too small — range-failure pivot (§9)"
    _w()
    crows, chashes = IP.load_items(coco, a.n_coco * 2)
    csplit = IC.sha1_split(chashes, name="gate0_coco")
    cdisc = [crows[i] for i in csplit["disc_idx"]][: a.n_coco]
    ci_int = IP.eval_pg1(model, proc, coco, cdisc, "cuda")
    ci_flr = IP.eval_pg1(model, proc, coco, cdisc, "cuda", scramble_seed=a.seed)
    rep["coco_probe"] = {"intact": ci_int["score"], "intact_ci": ci_int["ci95"],
                         "floor": ci_flr["score"], "floor_ci": ci_flr["ci95"]}
    assert ci_int["score"] > ci_flr["score"], "COCO CIDEr has no dynamic range"
    _w()

    # --- F: hook inertness + teeth smoke ---
    smoke = disc_rows[:8]
    inp = SH.vqa_forward_batches(proc, vqa, smoke, "cuda")[0]
    with torch.inference_mode():
        base_logits = model(**inp).logits[:, -1, :].float().cpu()
    Q16 = SH.build_arm_Q("random_k", sel["primary"], 16, bases, seed=3)
    hprobe = IP.harvest(layers, lambda b: model(**b), [inp], [sel["primary"]], x_cap=0)
    hbarL = hprobe[sel["primary"]][0]
    with IP.ablate_at(layers, sel["primary"], Q16, hbarL):
        pass                                              # install + remove
    with torch.inference_mode():
        after_logits = model(**inp).logits[:, -1, :].float().cpu()
    assert torch.equal(base_logits, after_logits), "hook not inert after removal"
    abl = IP.eval_pg1(model, proc, vqa, smoke, "cuda",
                      ctx=IP.ablate_at(layers, sel["primary"], Q16, hbarL))
    rep["hook_smoke"] = {"inert_after_removal": True, "ablated_generate_ran": abl["n"] == len(smoke)}
    rep["green"] = True
    _w()
    print("GATE0_OK", json.dumps({"ppl": rep["ppl_gate"], "vqa": rep["vqa_probe"],
                                  "selection": {kk: sel[kk] for kk in
                                                ("primary", "secondary", "anchors", "k_star")}},
                                 default=str))


if __name__ == "__main__":
    main()
