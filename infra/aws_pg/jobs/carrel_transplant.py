"""CARREL — weight-level transplant on the HELMHOLTZ-1/BROCA pair.

Spec: studies/lineage/cross-lineage/research/2026-09-02_carrel_spec.md. One job, server-side:
load the pair ONCE (BEFORE = healthy, AFTER = broken; eager attention, the validated
construction) -> snapshot the broken LM -> baselines ONCE -> for each frozen config copy the
healthy tensors for that config's parameter names INTO the broken model, evaluate end to end
(no donor at runtime), restore the snapshot, and assert the restored model reproduces
acc_after EXACTLY (wiring gate) -> verdict (carrel_core) -> cloze at best band -> RefCOCO
skill-retention at best band. Fail-loud; metrics.json incremental after every config.

  pg run infra.aws_pg.jobs.carrel_transplant --type g6e.2xlarge,g6e.xlarge,g6e.4xlarge,g6e.8xlarge \
      --s3-prefix results/carrel_transplant --budget-min 240
  # tail re-run (roster banked in results/carrel_transplant; cloze CI crashed on degenerate draws):
  pg run infra.aws_pg.jobs.carrel_transplant --args " --tail-only --best-band band_16_22" \
      --type g6e.2xlarge,g6e.xlarge,g6e.4xlarge,g6e.8xlarge --s3-prefix results/carrel_tail --budget-min 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-subject", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--vision-n", type=int, default=200)
    ap.add_argument("--vision-batch", type=int, default=8)
    ap.add_argument("--tail-only", action="store_true",
                    help="skip the roster (already banked); run cloze + vision at --best-band")
    ap.add_argument("--best-band", default=None, help="roster verdict's best_band (tail-only mode)")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import carrel_core as CC
    from infra.pg import text_eval as TE
    from infra.pg.diff_harvest import build_before_after
    from infra.pg.gemma_scope import resolve_gemma2_layers

    assert torch.cuda.is_available()
    device = "cuda"
    rep = {"object": "CARREL weight transplant (healthy layers into the broken model)",
           "n_per_subject": a.n_per_subject, "seed": a.seed, "n_boot": a.n_boot,
           "roster": [n for n, _ in CC.CONFIGS], "configs": {}}
    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- pair (validated HELMHOLTZ-1 construction) + anatomy bind ----
    pg_after, pg_before, proc, paths, _i, gate = build_before_after(device, pg_key="PG",
                                                                    attn_impl="eager")
    lm_b = resolve_gemma2_layers(pg_before.get_submodule(paths["language_model"]))
    lm_a = resolve_gemma2_layers(pg_after.get_submodule(paths["language_model"]))
    n_layers = len(lm_b); d_model = lm_b[0].mlp.down_proj.out_features
    rep["bind"] = {"n_layers": n_layers, "d_model": d_model, "lm_path": paths["language_model"],
                   "pair_gate": gate}
    assert n_layers == 26 and d_model == 2304, rep["bind"]

    healthy_sd = pg_before.state_dict()          # references (GPU); copied on load
    broken_sd = pg_after.state_dict()
    assert list(healthy_sd) == list(broken_sd)
    part = CC.partition_keys(list(broken_sd), paths["language_model"] + ".")
    universe = CC.lm_keys(part)
    for k in universe:
        assert healthy_sd[k].shape == broken_sd[k].shape, (k, healthy_sd[k].shape, broken_sd[k].shape)
        assert healthy_sd[k].dtype == broken_sd[k].dtype, (k, healthy_sd[k].dtype, broken_sd[k].dtype)
    # embed / lm_head tying + pair-identity (kept from PG in BOTH halves by construction)
    emb_k = part["embed"][0]
    tie_cfg = getattr(pg_after.config, "tie_word_embeddings", None)
    tie_txt = getattr(getattr(pg_after.config, "text_config", None), "tie_word_embeddings", None)
    tied_ptr = bool(part["lm_head"]) and (broken_sd[part["lm_head"][0]].data_ptr() == broken_sd[emb_k].data_ptr())
    for k in part["embed"] + part["lm_head"]:
        assert torch.equal(healthy_sd[k], broken_sd[k]), f"embed/lm_head differs across the pair: {k}"
    rep["anatomy"] = {"n_lm_keys": len(universe), "n_layer_keys": 26 * 11,
                      "lm_head_keys": part["lm_head"], "embed_key": emb_k,
                      "final_norm_key": part["final_norm"][0],
                      "tie_word_embeddings_config": tie_cfg, "tie_word_embeddings_text_config": tie_txt,
                      "lm_head_shares_storage_with_embed": tied_ptr,
                      "embed_lm_head_identical_across_pair": True,
                      "note": "head config's effective content = final norm only (embed/lm_head pair-identical)"}
    print(f"[CARREL] anatomy: {json.dumps(rep['anatomy'])}", flush=True)
    _w()

    # ---- broken snapshot (CPU) ----
    snap = {k: broken_sd[k].detach().clone().cpu() for k in universe}

    def transplant(names):
        res = pg_after.load_state_dict({k: healthy_sd[k] for k in names}, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        assert set(res.missing_keys) == set(broken_sd) - set(names), "missing-key set drifted"
        for k in names:
            assert torch.equal(broken_sd[k], healthy_sd[k]), f"transplant did not land: {k}"

    def restore(names):
        res = pg_after.load_state_dict({k: snap[k].to(device) for k in names}, strict=False)
        assert not res.unexpected_keys
        assert set(res.missing_keys) == set(broken_sd) - set(names)
        for k in names:
            assert torch.equal(broken_sd[k].cpu(), snap[k]), f"restore did not land: {k}"

    def mcq(model):
        return TE.mmlu_blankimg(model, proc, device, n_per_subject=a.n_per_subject)

    # ---- baselines ONCE ----
    rb = mcq(pg_before); ra = mcq(pg_after)
    acc_before, acc_after = rb["acc"], ra["acc"]
    assert rb["per_item_hash"] == ra["per_item_hash"], "item pairing broke"
    rep["baselines"] = {"acc_before": acc_before, "acc_after": acc_after, "n": ra["n"],
                        "n_correct_before": int(sum(rb["per_item"])),
                        "n_correct_after": int(sum(ra["per_item"]))}
    print(f"[CARREL] baselines: before {acc_before:.4f} after {acc_after:.4f} n={ra['n']}", flush=True)
    _w()

    def wiring_gate(tag):
        rg = mcq(pg_after)
        ok = (rg["per_item"] == ra["per_item"])
        assert ok, f"WIRING VOID after {tag}: restored model acc {rg['acc']} != acc_after {acc_after}"
        return True

    def _ci(st):
        return "VOID-CI" if st["ci_void"] else f"[{st['lo']:+.3f},{st['hi']:+.3f}]"

    if a.tail_only:
        assert a.best_band in CC.BAND_NAMES, a.best_band
        best = a.best_band
        rep["tail_only"] = True; rep["best_band_given"] = best
        best_names = CC.config_keys(part, dict(CC.CONFIGS)[best])
        # bind this job's hybrid to the roster artifact: MCQ of the best-band hybrid re-measured
        transplant(best_names)
        r = mcq(pg_after)
        restore(best_names); wiring_gate("mcq_check@" + best)
        rep["mcq_check_best_band"] = {"config": best, "acc": r["acc"], "n_correct": int(sum(r["per_item"])),
                                      "R": CC.restoration_fraction(r["acc"], acc_after, acc_before)}
        print(f"[CARREL][tail] MCQ re-check @{best}: {rep['mcq_check_best_band']}", flush=True)
        _w()
    else:
        # ---- the roster ----
        R = {}
        sanity_ok = True
        for name, spec in CC.CONFIGS:
            names = CC.config_keys(part, spec)
            tc = time.time()
            transplant(names)
            r = mcq(pg_after)
            assert r["per_item_hash"] == ra["per_item_hash"]
            st = CC.boot_R(r["per_item"], ra["per_item"], rb["per_item"], n_boot=a.n_boot, seed=a.seed)
            R[name] = st["R"]
            row = {"acc": r["acc"], "n_correct": int(sum(r["per_item"])), "n_keys": len(names),
                   "R": st["R"], "R_lo": st["lo"], "R_hi": st["hi"], "ci_void": st["ci_void"],
                   "frac_degenerate": st["frac_degenerate"], "per_item": r["per_item"],
                   "eval_s": time.time() - tc}
            if name == "none":
                row["sanity_equals_after"] = bool(r["per_item"] == ra["per_item"])
                sanity_ok &= row["sanity_equals_after"]
            if name == "all":
                row["sanity_equals_before"] = bool(r["per_item"] == rb["per_item"])
                sanity_ok &= row["sanity_equals_before"]
            restore(names)
            row["wiring_gate_after_restore"] = wiring_gate(name)
            rep["configs"][name] = row
            print(f"[CARREL] {name:16s} keys={len(names):3d} acc={r['acc']:.4f} R={st['R']:+.3f} "
                  f"{_ci(st)} restore-gate OK ({row['eval_s']:.0f}s)", flush=True)
            if name == "all":
                assert sanity_ok, "SANITY FAILED: none!=AFTER or all!=BEFORE — pair/transplant wiring void"
            if name == "ctrl_0_15":
                rep["replication_ctrl_0_15"] = CC.replication_check(st["R"])
                print(f"[CARREL] replication vs BROCA R(15): {rep['replication_ctrl_0_15']}", flush=True)
            if name == CC.CONTROL_NAMES[-1]:
                band = CC.null_band(R); T = CC.threshold(band)
                rep["null_band"] = band; rep["threshold"] = T
                print(f"[CARREL] null band={band:.4f} -> T={T:.4f} (FROZEN rule, computed before any band)", flush=True)
            _w()

        # ---- verdict ----
        verdict = CC.carrel_verdict(R, sanity_ok)
        rep["verdict"] = verdict
        print(f"[CARREL] VERDICT: {json.dumps(verdict)}", flush=True)
        _w()
        best = verdict["best_band"]; best_names = CC.config_keys(part, dict(CC.CONFIGS)[best])

    # ---- cloze (content axis): none, all, best band — point values banked BEFORE any bootstrap ----
    clz_a = TE.mmlu_cloze(pg_after, proc.tokenizer, device, processor=proc, n_per_subject=a.n_per_subject)
    clz_b = TE.mmlu_cloze(pg_before, proc.tokenizer, device, processor=proc, n_per_subject=a.n_per_subject)
    assert clz_a["per_item_hash"] == clz_b["per_item_hash"]
    transplant(best_names)
    clz_h = TE.mmlu_cloze(pg_after, proc.tokenizer, device, processor=proc, n_per_subject=a.n_per_subject)
    restore(best_names); wiring_gate("cloze@" + best)
    rep["cloze"] = {"config": best, "acc_after": clz_a["acc"], "acc_before": clz_b["acc"],
                    "acc_hybrid": clz_h["acc"], "n": clz_a["n"],
                    "n_correct_after": int(sum(clz_a["per_item"])), "n_correct_before": int(sum(clz_b["per_item"])),
                    "n_correct_hybrid": int(sum(clz_h["per_item"])),
                    "R": CC.restoration_fraction(clz_h["acc"], clz_a["acc"], clz_b["acc"]),
                    "per_item_after": clz_a["per_item"], "per_item_before": clz_b["per_item"],
                    "per_item_hybrid": clz_h["per_item"]}
    _w()
    cst = CC.boot_R(clz_h["per_item"], clz_a["per_item"], clz_b["per_item"], n_boot=a.n_boot, seed=a.seed)
    rep["cloze"].update({"R_lo": cst["lo"], "R_hi": cst["hi"], "ci_void": cst["ci_void"],
                         "frac_degenerate": cst["frac_degenerate"]})
    print(f"[CARREL] cloze @{best}: after {clz_a['acc']:.4f} before {clz_b['acc']:.4f} "
          f"hybrid {clz_h['acc']:.4f} R={cst['R']:+.3f} {_ci(cst)}", flush=True)
    _w()

    # ---- skill retention: RefCOCO acc@0.5 (e4_capability's validated zero-shot readout) ----
    from infra.aws_pg.jobs import e4_capability as E4
    from infra.pg import battery_tasks as B
    task = B.REFCOCO_TASK
    rows = E4._eval_rows(task, a.vision_n, a.seed)
    def vis(model):
        return E4._per_item_scores(task, rows, E4._generate(model, proc, task, rows, device, a.vision_batch))
    s_after = vis(pg_after); s_before = vis(pg_before)
    rep["vision"] = {"task": task.name, "dataset": task.dataset, "split": task.eval_split,
                     "metric": task.metric_name, "n": len(rows), "config": best,
                     "acc_after": float(s_after.mean()), "acc_before": float(s_before.mean()),
                     "per_item_after": s_after.tolist(), "per_item_before": s_before.tolist()}
    _w()
    transplant(best_names)
    s_h = vis(pg_after)
    restore(best_names); wiring_gate("vision@" + best)
    rep["vision"].update({"acc_hybrid": float(s_h.mean()), "per_item_hybrid": s_h.tolist(),
                          "retention_point": CC.retention(float(s_h.mean()), float(s_after.mean()),
                                                          float(s_before.mean()))})
    _w()
    ret = CC.boot_retention(s_h, s_after, s_before, n_boot=a.n_boot, seed=a.seed)
    rep["vision"].update({"ci_after": E4._boot_ci(s_after, a.n_boot, a.seed),
                          "ci_before": E4._boot_ci(s_before, a.n_boot, a.seed),
                          "ci_hybrid": E4._boot_ci(s_h, a.n_boot, a.seed), "retention": ret})
    print(f"[CARREL] vision @{best}: after {s_after.mean():.4f} before {s_before.mean():.4f} "
          f"hybrid {s_h.mean():.4f} retention={ret}", flush=True)
    _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("CARREL_OK " + json.dumps({"tail_only": a.tail_only, "best_band": best,
                                     "cloze_R": rep["cloze"]["R"], "vision": rep["vision"]["retention"]}),
          flush=True)


if __name__ == "__main__":
    main()
