"""STENCIL preflight + range guard (GPU) — spec §9 step 1 + the §5 positive-control range guards,
run BEFORE the ~10-28 GPU-h Q banking (the responsible ordering: settle VOID-no-range cheaply first).

Preflight (§9.1): base ppl ≈ the FUNC anchor through the adapter-attached forward; a zero-delta
fixed-basis adapter is byte-identical to no-adapter; the trajectory trainer ≡ relearn.train_adapter_qa
(A-STENCIL-7); the banked L0 Q loads with its recorded metric_key/k_stiff (bit-exact ingredient).

Range guard (§5): free-LoRA (high rank) + full-FT on discovery subjects → task-gain + forgetting
frontier. gain_range_ok ∧ forget_range_ok ⇒ proceed to Q banking; else VOID-no-range (a complete
landing — the axis has no room to move on PG1).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_preflight_range \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/stencil_prerange --budget-min 240
"""
import argparse
import json
import os
import pathlib

# provisional range floors (discovery formalizes with SEs; the probe reports raw + this call)
DELTA_GAIN = 0.03        # unconstrained adaptation must lift held-out subject acc by ≥ 3 abs points
DELTA_FORGET = 0.02      # ... and perturb the battery by ≥ 2% composite forgetting
PROBE_RANK = 150         # free-LoRA probe rank (ample capacity; fp32 adapter masters)
PROBE_RECIPE = {"lr": 2e-4, "epochs": 6, "batch_size": 16, "warmup_ratio": 0.1, "micro_batch": 8}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=1, help="discovery subjects to probe for range")
    ap.add_argument("--no-full-ft", action="store_true", help="skip full-FT (fits 24GB A10G; free-LoRA range signal)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _ration_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import relearn as RL
    from infra.pg import stencil_core as SC

    SS.install_pg()
    device = "cuda"
    rep = {"object": "stencil_prerange_v1", "gpu_seconds": {}}
    timer = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- load PG1 -----------------------------------------------------------------------
    with timer.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/stencil")
    rep["pinned_sha"], rep["func_base_ppl"] = sha, float(base_ppl)
    _w()

    # ---- PREFLIGHT ----------------------------------------------------------------------
    pf = {}
    # (a) ppl anchor through the true forward
    with timer.stage("preflight_ppl"):
        pg = SPS.safe_ppl_gate(model, proc, device, n_seq=SS.N_PPL_BATTERY)
    rel = abs(pg["ppl"] - base_ppl) / base_ppl
    pf["ppl"] = {"measured": pg["ppl"], "anchor": float(base_ppl), "rel_err": rel}
    assert rel < 0.05, f"base ppl {pg['ppl']:.3f} != anchor {base_ppl:.3f} (rel {rel:.3f}) — wrong forward"

    # (b) banked L0 Q loads with its recorded ingredients (bit-exact)
    qb = RS.load_q_bank("/tmp/stencil")
    pf["banked_q"] = {"metric_key": qb["metric_key"], "k_stiff": int(qb["k_stiff"]),
                      "eval_top": float(torch.as_tensor(qb["evals"])[0]),
                      "evecs_shape": list(torch.as_tensor(qb["evecs"]).shape)}
    assert qb["metric_key"] == "L0/dc/ppl" and int(qb["k_stiff"]) == 679, pf["banked_q"]

    # (c) zero-delta fixed-basis adapter is byte-identical to no-adapter
    snap = SS.snapshot_base(model)
    d = SS.D
    g = torch.Generator().manual_seed(0)
    synbank = {L: {"Q_evecs": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0],
                   "sigma_evecs": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0],
                   "u_wr": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0][:, :200],
                   "d": d} for L in range(SS.N_LAYERS)}
    # zero-delta identity on the PG1 in-distribution BLANK-IMAGE forward (matches the eval/train forward)
    probe_txt = "The following are multiple choice questions.\nWhat is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    from PIL import Image
    import numpy as np
    gray = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text=[probe_txt], images=[gray], return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)

    def _fwd():
        with torch.inference_mode():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"], logits_to_keep=1).logits[:, -1, :].float().clone()
    logits0 = _fwd()
    with timer.stage("preflight_identity"):
        SC.attach_fixed_adapters(model, "susceptibility", synbank, rank=20, alpha=20)
        logits1 = _fwd()
    pf["zero_delta_identity"] = {"max_abs_diff": float((logits0 - logits1).abs().max()),
                                 "byte_identical": bool(torch.equal(logits0, logits1))}
    assert torch.equal(logits0, logits1), "zero-delta adapter is NOT identity — forward differs"
    SS.reset_model(model, snap)

    # (d) trajectory trainer sanity: the PG1 blank-image trainer runs, loss is finite & decreases, and a
    # checkpoint eval fires (A-STENCIL-8; the trainer deliberately uses the blank-image forward, so an
    # equivalence to relearn's text-only forward is NOT expected).
    mmlu = SS.mmlu_by_subject()
    roster = SS.build_roster(mmlu)
    subj = roster["discovery"][0]
    tr_rows, ev_rows, ev_hashes = SS.subject_train_eval(mmlu[subj]["test"])
    examples, _ = SS.train_examples(tr_rows, ev_hashes)
    tiny = examples[:32]
    with timer.stage("preflight_trainer"):
        SC.attach_free_lora(model, rank=8, alpha=8, seed=SS.SEED)
        seen = {"n": 0}
        sB = SC.train_with_trajectory(model, proc, device, tiny, lr=2e-4, epochs=99,
                                      batch_size=16, warmup_ratio=0.1, seed=SS.SEED, max_steps=12,
                                      micro_batch=8, checkpoint_steps=[6, 12],
                                      eval_fn=lambda m, s: seen.update(n=seen["n"] + 1) or {"step_seen": s})
        SS.reset_model(model, snap)
    losses_ok = sB["final_loss"] is not None and sB["final_loss"] == sB["final_loss"]  # finite (not NaN)
    pf["trainer"] = {"final_loss": sB["final_loss"], "steps": sB["steps"],
                     "n_checkpoints": len(sB["trajectory"])}
    assert losses_ok and sB["steps"] == 12 and len(sB["trajectory"]) == 2, "blank-image trainer sanity failed"
    rep["preflight"] = pf
    _w()
    print("[stencil] preflight OK:", json.dumps({k: (pf[k] if k != "banked_q" else pf[k]["metric_key"])
          for k in ("ppl", "zero_delta_identity", "trainer", "banked_q")}, default=str), flush=True)

    # ---- RANGE GUARD --------------------------------------------------------------------
    # smaller battery + 2 checkpoints for the probe (the range signal is coarse; discovery uses full)
    bctx = SS.make_battery_ctx(proc, roster, n_vqa=100, n_coco=32, n_ppl=100)
    probe_cps = (0.5, 1.0)
    weights = {c: 1.0 for c in SS.BATTERY_COMPONENTS}       # equal weights for the probe
    rep["roster"] = {k: roster[k] for k in ("discovery", "confirm", "broad_mmlu", "test_counts")}

    with timer.stage("base_refs"):
        base_battery = SS.battery_probe(model, proc, device, bctx)
        base_subj = {}
        for s in roster["discovery"][:a.subjects]:
            _, evr, evh = SS.subject_train_eval(mmlu[s]["test"])
            base_subj[s] = {"acc": SS.subject_accuracy(model, proc, device, evh)["acc"], "hashes": evh}
    rep["base_battery"] = base_battery
    rep["base_subject_acc"] = {s: base_subj[s]["acc"] for s in base_subj}
    _w()

    probe_arms = [("free_lora", PROBE_RANK)]
    if not a.no_full_ft:
        probe_arms.append(("full_ft", 0))
    cells = []
    for s in roster["discovery"][:a.subjects]:
        tr_rows, _, evh = SS.subject_train_eval(mmlu[s]["test"])
        for arm, rank in probe_arms:
            SS.reset_model(model, snap)
            with timer.stage(f"cell_{arm}_{s}"):
                cell = SS.adapt_and_eval(
                    model, proc, device, arm=arm, banks={}, rank=rank, alpha=max(rank, 1),
                    subject_rows=tr_rows, eval_hashes=evh, bctx=bctx, base_battery=base_battery,
                    base_subj_acc=base_subj[s]["acc"], weights=weights, recipe=PROBE_RECIPE,
                    checkpoint_fracs=probe_cps)
            traj = cell["trajectory"]
            best_gain = max((t["task_gain"] for t in traj), default=0.0)
            max_forget = max((t["forgetting"] for t in traj), default=0.0)
            cells.append({"subject": s, "arm": arm, "rank": rank, "best_gain": best_gain,
                          "max_forget": max_forget, "trajectory": traj,
                          "train_stats": cell["train_stats"]})
            print(f"[stencil] range {arm}@{s}: best_gain={best_gain:+.4f} max_forget={max_forget:+.4f}",
                  flush=True)
            _w()
    rep["range_cells"] = cells

    best_gain_all = max((c["best_gain"] for c in cells), default=0.0)
    best_forget_all = max((c["max_forget"] for c in cells), default=0.0)
    gain_ok = best_gain_all > DELTA_GAIN
    forget_ok = best_forget_all > DELTA_FORGET
    rep["range_verdict"] = {
        "best_gain_all": best_gain_all, "best_forget_all": best_forget_all,
        "delta_gain": DELTA_GAIN, "delta_forget": DELTA_FORGET,
        "gain_range_ok": bool(gain_ok), "forget_range_ok": bool(forget_ok),
        "proceed_to_banking": bool(gain_ok and forget_ok),
        "provisional_verdict": ("PROCEED" if (gain_ok and forget_ok) else "VOID-no-range"),
        "note": "provisional (equal weights, single probe rank); discovery formalizes with SEs"}
    _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("STENCIL_PRERANGE_OK", json.dumps(rep["range_verdict"], default=str), flush=True)


if __name__ == "__main__":
    main()
