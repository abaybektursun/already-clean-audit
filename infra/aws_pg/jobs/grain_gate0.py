"""GRAIN Gate-0 (spec §11.0; GPU) — WRITE-FIRST at every stage. Verifies at the artifacts,
on real weights, BEFORE freeze:

  A. §3 ledger: every REUSED surface imports + every NEW grain_bsf surface exists;
     dictionary_learning present, version pinned.
  B. Read banks located + valid (S_action bases.pt via validate_bank; S_vl svl_bases.pt);
     sites resolved from artifacts (endow_o_controls primary + inherit_svl layer).
  C. S_action re-estimation draw banks LOCATED + counted (≥3 needed for the adjudicated S2
     branch; fewer → that branch pre-recorded UNRUN-adjudicated, descriptive-only).
  D. GRAIN item halves built + hashed (grain| salt; VQAv2 + COCO) → items_*.json banked.
  E. PG1 loads EAGER at the invcore-FUNC-pinned sha; ppl gate reproduces the artifact value
     (|Δ| ≤ 0.5); dead-model floor + intact + scramble recomputed on a discovery slice.
  F. Splice-glue inertness: the pass-through pre-hook is a bitwise no-op on real logits.
  G. COMWRITE builder + anatomy artifacts asserted (X-arm feasibility).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_gate0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_gate0 --budget-min 120 --args "--n-probe 192"
"""
import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-probe", type=int, default=192)
    ap.add_argument("--n-items-vqa", type=int, default=2400, help="total streamed VQA items to split")
    ap.add_argument("--n-items-coco", type=int, default=1024)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "grain_gate0_v1", "seed": a.seed}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP

    GS.install_pg()

    # --- A: ledger + library pin ---
    import dictionary_learning
    from dictionary_learning.dictionary import BatchTopKSAE            # noqa: F401
    from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer  # noqa: F401
    from infra.pg.comwrite_surgery import build_decoder                # noqa: F401
    from infra.pg.inherit_core import sha1_split                       # noqa: F401
    for surface in ("BlockTopKSAE", "train_block_sae", "block_quality", "union_Q",
                    "block_alignment", "block_usage", "block_stable_rank", "support_spread",
                    "adjudicate_grain", "adjudicate_s2", "jaccard_null", "gen_planted",
                    "make_pg1_pre_hook_splice", "make_pg1_pre_hook_passthrough",
                    "make_pg1_pre_hook_steer", "save_dict_bank", "load_dict_bank"):
        assert hasattr(GB, surface), f"grain_bsf missing NEW surface {surface}"
    rep["ledger"] = {"dictionary_learning": getattr(dictionary_learning, "__version__", "unpinned-git"),
                     "grain_bsf_surfaces": "all-present"}
    _w()

    # --- B: read banks + sites ---
    banks = GS.load_read_banks()
    Lp, Ls = banks["saction"]["L"], banks["svl"]["L"]
    rep["sites"] = {"primary": Lp, "secondary": Ls}
    rep["banks"] = {r: {"L": banks[r]["L"], "shape": list(banks[r]["V"].shape)} for r in banks}
    _w()

    # --- C: S_action draw banks (S2 branch feasibility, frozen §11.3 pivot) ---
    draws_found = ["ctrliface_discovery/out/bases.pt"]
    rc = subprocess.run(["aws", "s3", "ls", GS.FULL_BANK, "--region", "us-east-1"],
                        capture_output=True)
    if rc.returncode == 0:
        draws_found.append("inherit_pi05_bank/out/bases_full.pt")
    rep["saction_draws"] = {"found": draws_found, "n": len(draws_found),
                            "s2_adjudicated": len(draws_found) >= 3,
                            "note": ("<3 banked S_action draws → S2(S_action) pre-recorded "
                                     "UNRUN-adjudicated (re-estimation needs the pi0.5 lerobot "
                                     "stack, out of GRAIN budget); descriptive Jaccard on the "
                                     "available draws only. S2 adjudication runs on S_vl draws "
                                     "(re-estimated on disjoint discovery subsets, T1 stage 2).")}
    _w()

    # --- D: GRAIN item halves (fresh grain| salt), banked with hashes ---
    model, proc = None, None
    vqa, coco = IP.pg1_tasks()
    items = {}
    for task, n_total, slug in ((vqa, a.n_items_vqa, "vqa"), (coco, a.n_items_coco, "coco")):
        rows, hashes = IP.load_items(task, n_total)
        split = GS.grain_split(hashes, name=f"grain_{slug}")
        items[slug] = {"rows": rows, "hashes": hashes, "split": split}
        (out / f"items_{slug}.json").write_text(json.dumps(
            {"task": task.name, "n": len(hashes), "hashes": hashes,
             "sealed_idx": [int(i) for i in split["conf_idx"]],
             "salt": "grain|", "stream_window": n_total}, indent=2))
        rep[f"items_{slug}"] = {"n": len(hashes), "n_disc": int(split["n_disc"]),
                                "n_sealed": int(split["n_conf"])}
    _w()

    # --- E: PG1 (pinned sha, eager) + ppl gate + floors on a discovery slice ---
    sha, ppl_ref = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    rep["pg1"] = {"repo": IP.PG1_REPO, "sha": sha, "n_layers": len(layers), "d": IP.D}
    pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref,
                       "tol": GS.PPL_TOL}
    assert abs(pp["ppl"] - ppl_ref) <= GS.PPL_TOL, \
        f"ppl gate RED: {pp['ppl']:.3f} vs artifact {ppl_ref:.3f} (sdpa trap / wiring)"
    _w()

    disc_rows = [items["vqa"]["rows"][i] for i in items["vqa"]["split"]["disc_idx"]][: a.n_probe]
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    scram = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", scramble_seed=a.seed)
    # PG1-native h̄ at both sites (probe-sized; the T1 harvest recomputes over the full set)
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    batches = vqa_forward_batches(proc, vqa, disc_rows[:64], "cuda")
    hv = IP.harvest(layers, lambda b: model(**b), batches, [Lp, Ls])
    floors = {}
    for L in (Lp, Ls):
        fl = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda",
                         ctx=IP.replace_at(layers, L, hv[L][0]))
        floors[str(L)] = {"score": fl["score"], "ci": fl["ci95"]}
    rep["vqa_probe"] = {"intact": intact["score"], "intact_ci": intact["ci95"],
                        "scramble": scram["score"], "dead_floor_by_L": floors}
    dyn = intact["score"] - min(v["score"] for v in floors.values())
    assert dyn > 0.05, f"no dynamic range vs dead floor ({dyn:.4f}) — halt"
    rep["dynamic_range_vs_dead"] = dyn
    _w()

    # --- F: splice-glue inertness (bitwise) on real logits ---
    inp = batches[0]
    with torch.inference_mode():
        base_logits = model(**inp).logits[:, -1, :].float().cpu()
    h = layers[Lp].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(),
                                             with_kwargs=True)
    with torch.inference_mode():
        hooked_logits = model(**inp).logits[:, -1, :].float().cpu()
    h.remove()
    with torch.inference_mode():
        after_logits = model(**inp).logits[:, -1, :].float().cpu()
    assert torch.equal(base_logits, hooked_logits), "pass-through hook NOT a bitwise no-op"
    assert torch.equal(base_logits, after_logits), "hook not inert after removal"
    rep["splice_glue"] = {"passthrough_bitwise_noop": True}
    _w()

    # --- G: COMWRITE anatomy artifacts present (X-arm needs metrics.json k_star/excluded_lt
    # + the donor LOO scratch bases; X1 items are GRAIN's own — journal E1 #13) ---
    anat = json.loads(GS.fetch(f"{GS.COMWRITE_ANATOMY}/metrics.json",
                               "/tmp/grain/anatomy.json").read_text())
    assert "k_star" in anat and "excluded_lt" in anat and "per_matrix" in anat, \
        f"anatomy artifact schema drift: {sorted(anat.keys())[:8]}"
    assert anat["base_sha"] == sha, "anatomy base sha != GRAIN pin — provenance fork"
    loo_ok = subprocess.run(["aws", "s3", "ls",
                             f"{GS.COMWRITE_ANATOMY}/scratch/loo__down_proj__L0__0.npy",
                             "--region", "us-east-1"], capture_output=True).returncode == 0
    assert loo_ok, "anatomy LOO scratch missing — X-arm infeasible"
    rep["comwrite_x_feasible"] = {"anatomy2": True, "k_star": int(anat["k_star"]),
                                  "excluded_n": len(anat["excluded_lt"]), "loo_scratch": True}

    rep["green"] = True
    _w()
    print("GRAIN_GATE0_OK", json.dumps({"sites": rep["sites"], "ppl": rep["ppl_gate"]["ppl"],
                                        "dyn": dyn, "s2_saction_adjudicated":
                                        rep["saction_draws"]["s2_adjudicated"]}, default=str))


if __name__ == "__main__":
    main()
