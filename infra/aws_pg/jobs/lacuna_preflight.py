"""LACUNA preflight + controls (spec §5, §9 step 4) — every gate that must pass BEFORE any contrast.

  P1  the July multimodal collateral read (`_cleave_shared.mm_battery`) and LACUNA's own ladder read
      are run on PG1 over the SAME items IN THE SAME JOB and must agree inside the paired-bootstrap
      CI. FAILURE = harness mismatch → halt-fix. (Both recomputed; nothing quoted from a prior run.)
  P2  base gemma-2b MMLU + wikitext ppl reproduce the VLAT artifact anchors (read FROM
      `results/vlat/metrics.json`, not from prose). FAILURE = text-battery wiring drift.
  G2  the base-ppl anchor, digit-exact, on the PG1 shell.
  G4  Δ_perception(R3) on VQAv2 — the shell must demonstrably USE the image, else every mm verdict
      is VOID (a gray-image forward that scores like a real one means the instrument is blind).
  S1  answer-shuffle: rescoring the SAME generations against permuted ground truth must collapse
      to the chance level of each scorer.
  S2  caption-target permutation must collapse CIDEr.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_preflight --type g6e.8xlarge --no-weights \
      --s3-prefix results/lacuna_preflight --budget-min 150
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PPL_ANCHOR = 22.42583
# `pg` ships ONLY `infra/` to the instance, so the VLAT artifact is read from its S3 copy.
# Uploaded verbatim from results/vlat/metrics.json (sha256 966490d9…) — the anchors are READ from
# the artifact, never transcribed into this file (baseline discipline).
VLAT_ARTIFACT_S3 = "s3://pg-lineage-769063704696/artifacts/vlat/metrics.json"
VLAT_ARTIFACT_SHA256 = "966490d983866d40de92bbc4bc798d582dda1db0089b5969187512d69f2c8420"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-p1-vqa", type=int, default=200)      # the July battery's own size
    ap.add_argument("--n-p1-coco", type=int, default=64)
    ap.add_argument("--n-g4", type=int, default=600)
    ap.add_argument("--n-shuffle", type=int, default=600)
    ap.add_argument("--mmlu-n-per-subject", type=int, default=20)
    ap.add_argument("--shuffle-seed", type=int, default=4242)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-p2", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs._inherit_shared import install_pg
    install_pg()               # pycocoevalcap (CIDEr) is not in the AMI
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS

    assert torch.cuda.is_available(), "lacuna_preflight needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "lacuna_preflight_v1", "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "seeds": {"shuffle": a.shuffle_seed}}
    ctrl = {}
    rep["controls"] = ctrl

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    model, proc = LS.load_pg1(device=device)
    tasks = LS.mm_tasks()

    # ---- G2: the ppl anchor, digit-exact --------------------------------------------------
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    ctrl["G2_ppl_anchor"] = {"ppl": val, "anchor": PPL_ANCHOR, "abs_diff": abs(val - PPL_ANCHOR),
                             "passes": abs(val - PPL_ANCHOR) < 1e-3}
    print(f"[PRE] G2 ppl={val} anchor={PPL_ANCHOR} pass={ctrl['G2_ppl_anchor']['passes']}", flush=True)
    _w()
    assert ctrl["G2_ppl_anchor"]["passes"], f"G2 FAIL: {val} vs {PPL_ANCHOR}"

    # ---- P1: the July read vs the LACUNA read, same model, same items ---------------------
    from infra.aws_pg.jobs import _cleave_shared as CS
    from infra.aws_pg.jobs import _stencil_shared as SS
    bctx = SS.make_battery_ctx(proc, {"broad_hashes": []}, n_vqa=a.n_p1_vqa, n_coco=a.n_p1_coco)
    july = CS.mm_battery(model, proc, device, bctx)
    lac_vqa = IP.eval_pg1(model, proc, tasks["vqav2"], bctx["vqa_rows"][:a.n_p1_vqa], device,
                          batch_size=a.batch_size)
    lac_coco = IP.eval_pg1(model, proc, tasks["coco_captions"], bctx["coco_rows"][:a.n_p1_coco],
                           device, batch_size=a.batch_size)
    d_vqa = abs(july["vqav2"] - lac_vqa["score"])
    d_coco = abs(july["coco_cider"] - lac_coco["score"])
    ctrl["P1_harness"] = {
        "july_vqav2": july["vqav2"], "lacuna_vqav2": lac_vqa["score"], "abs_diff_vqa": d_vqa,
        "lacuna_vqa_ci95": lac_vqa["ci95"], "n_vqa": a.n_p1_vqa,
        "july_coco_cider": july["coco_cider"], "lacuna_coco_cider": lac_coco["score"],
        "abs_diff_cider": d_coco, "n_coco": a.n_p1_coco,
        "passes": bool(d_vqa <= (lac_vqa["ci95"][1] - lac_vqa["ci95"][0]) / 2 and d_coco <= 0.05)}
    print(f"[PRE] P1 vqa july={july['vqav2']:.4f} lacuna={lac_vqa['score']:.4f} Δ={d_vqa:.5f} | "
          f"cider july={july['coco_cider']:.4f} lacuna={lac_coco['score']:.4f} Δ={d_coco:.5f}",
          flush=True)
    _w()
    assert ctrl["P1_harness"]["passes"], f"P1 FAIL — harness mismatch: {ctrl['P1_harness']}"

    # ---- G4: does the shell USE the image? (Δ_perception on PG1) --------------------------
    rows, hashes = IP.load_items(tasks["vqav2"], a.n_g4)
    real = IP.eval_pg1(model, proc, tasks["vqav2"], rows, device, batch_size=a.batch_size)
    gray = IP.eval_pg1(model, proc, LS.gray_task(tasks["vqav2"]), rows, device,
                       batch_size=a.batch_size)
    dp = LC.delta_perception(real["per_item"], gray["per_item"], n_boot=10000, seed=1)
    ctrl["G4_shell_perception"] = {"real": real["score"], "gray": gray["score"], **dp,
                                   "passes": bool(dp["lo95"] > 0), "n": a.n_g4}
    print(f"[PRE] G4 real={real['score']:.4f} gray={gray['score']:.4f} "
          f"Δ={dp['diff']:.4f} lo95={dp['lo95']:.4f}", flush=True)
    _w()
    assert ctrl["G4_shell_perception"]["passes"], "G4 FAIL — the shell never reads the image"

    # ---- S1: answer-shuffle → chance (rescore the SAME generations) -----------------------
    rng = np.random.default_rng(a.shuffle_seed)
    s1 = {}
    ctrl["S1_answer_shuffle"] = s1          # bound BEFORE the loop so a partial run is visible
    S1_TASKS = ("vqav2", "textvqa", "refcoco")
    for name in S1_TASKS:
        task = tasks[name]
        r_rows, _h = IP.load_items(task, a.n_shuffle)
        ev = IP.eval_pg1(model, proc, task, r_rows, device, batch_size=a.batch_size)
        perm = rng.permutation(len(r_rows))
        while any(perm == np.arange(len(r_rows))):
            perm = rng.permutation(len(r_rows))
        shuffled = [float(task.score_fn(p, r_rows[perm[i]])) for i, p in enumerate(ev["preds"])]
        s1[name] = {"true": ev["score"], "shuffled": float(np.mean(shuffled)),
                    "shuffled_ci": LC.boot_mean_ci(shuffled, n_boot=2000, seed=2),
                    "collapsed": bool(np.mean(shuffled) < 0.5 * ev["score"])}
        print(f"[PRE] S1 {name}: true={ev['score']:.4f} shuffled={s1[name]['shuffled']:.4f}",
              flush=True)
        _w()
    # `all()` over an empty dict is True — a control that cannot fail. Require every task present.
    assert len(s1) == len(S1_TASKS), f"S1 incomplete: {sorted(s1)} vs {list(S1_TASKS)}"
    assert all(v["collapsed"] for v in s1.values()), f"S1 FAIL — scoring does not track content: {s1}"

    # ---- S2: caption-target permutation → CIDEr collapse ----------------------------------
    from infra.pg.projector_ablation import _cider_boot
    cand, ref = lac_coco["cand"], lac_coco["ref"]
    keys = list(cand)
    p2 = rng.permutation(len(keys))
    while any(p2 == np.arange(len(keys))):
        p2 = rng.permutation(len(keys))
    ref_perm = {k: ref[keys[p2[i]]] for i, k in enumerate(keys)}
    sc, lo, hi = _cider_boot(cand, ref_perm, n_boot=200)
    ctrl["S2_caption_permutation"] = {"true_cider": lac_coco["score"], "permuted_cider": sc,
                                      "permuted_ci95": [lo, hi],
                                      "collapsed": bool(sc < 0.25 * lac_coco["score"])}
    print(f"[PRE] S2 cider true={lac_coco['score']:.4f} permuted={sc:.4f}", flush=True)
    _w()
    assert ctrl["S2_caption_permutation"]["collapsed"], "S2 FAIL — CIDEr does not collapse"

    del model
    torch.cuda.empty_cache()

    # ---- P2: base gemma-2b text battery reproduces the VLAT artifact anchors --------------
    if not a.skip_p2:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from infra.pg import text_eval as TE
        from infra.pg.entities import ENTITIES
        L0 = ENTITIES["L0_g1"]
        import hashlib
        import subprocess
        lp = pathlib.Path("/tmp/vlat_metrics.json")
        subprocess.run(["aws", "s3", "cp", VLAT_ARTIFACT_S3, str(lp), "--region", "us-east-1"],
                       check=True, capture_output=True)
        got = hashlib.sha256(lp.read_bytes()).hexdigest()
        assert got == VLAT_ARTIFACT_SHA256, f"VLAT artifact sha mismatch: {got}"
        art = json.loads(lp.read_text())
        anchor_mmlu = art["controls"]["positive_base"]["mmlu_acc_pct"]
        anchor_ppl = art["controls"]["positive_base"]["ppl"]
        tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
        g1 = AutoModelForCausalLM.from_pretrained(
            L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        mc = TE.mmlu_choice_logits(g1, tok, device, n_per_subject=a.mmlu_n_per_subject)
        pp = TE.clean_text_perplexity(g1, tok, device, max_tokens=120_000)
        ci = LC.boot_mean_ci(mc["per_item"], n_boot=4000, seed=3)
        ctrl["P2_text_wiring"] = {
            "artifact": VLAT_ARTIFACT_S3, "artifact_sha256": VLAT_ARTIFACT_SHA256, "anchor_mmlu_acc_pct": anchor_mmlu, "anchor_ppl": anchor_ppl,
            "recomputed_mmlu_acc_pct": mc["acc_pct"], "n": mc["n"],
            "recomputed_mmlu_ci95_pct": [100 * ci["lo95"], 100 * ci["hi95"]],
            "recomputed_ppl": pp["ppl"],
            "mmlu_in_ci": bool(100 * ci["lo95"] <= anchor_mmlu <= 100 * ci["hi95"]),
            "ppl_rel_diff": abs(pp["ppl"] - anchor_ppl) / anchor_ppl,
            "note": "the sub-sampled MMLU (n_per_subject) is a WIRING gate, not a power claim"}
        ctrl["P2_text_wiring"]["passes"] = bool(ctrl["P2_text_wiring"]["mmlu_in_ci"]
                                                and ctrl["P2_text_wiring"]["ppl_rel_diff"] < 0.05)
        print(f"[PRE] P2 mmlu={mc['acc_pct']:.2f} (anchor {anchor_mmlu}) ppl={pp['ppl']:.3f} "
              f"(anchor {anchor_ppl}) pass={ctrl['P2_text_wiring']['passes']}", flush=True)
        _w()
        assert ctrl["P2_text_wiring"]["passes"], f"P2 FAIL: {ctrl['P2_text_wiring']}"

    rep["preflight_pass"] = True
    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_PREFLIGHT_OK", json.dumps({k: v.get("passes") for k, v in ctrl.items()},
                                            default=str), flush=True)


if __name__ == "__main__":
    main()
