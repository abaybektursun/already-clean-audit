"""W12 §6C — causal reconstruction (GPU). Which IN-SPAN component of the LM ΔW carries capability:
gain (rescale existing pairings) or routing (recombine existing directions)?

Reuses the GraftLM-E4 no-retrain generate-and-score harness with a GAIN/ROUTING truncation instead
of rank-k. Four arms differ ONLY in the 26×7 decoder WEIGHT MATRICES (norms / embed / vision /
projector are ALL held at PG2 = CEILING, so the contrast is purely the weight-matrix component):

  FLOOR        = W_before (base gemma-2-2b decoder matrices)            R_cap ≡ 0
  gain-only    = W_before + U diag(M) Vᵀ   (freeze routing)
  routing-only = W_before + U offdiag(M) Vᵀ (freeze gain)
  full         = W_after  (real PG2-pt matrices)                        R_cap ≡ 1  (= CEILING)

R_cap = (arm − FLOOR)/(CEILING − FLOOR). Primary axis = RefCOCO acc@0.5 (the installed-capability
axis with clean dynamic range, e4 preflight). Secondary = blank-image MMLU (the W11-A text-forgetting
axis; CEILING is at chance there, so it reads which component carries the forgetting). δ rule (§6C):
gain-only R_cap ≥ 1−δ ⇒ gain-carried (SVF); routing-only ≥ 1−δ ⇒ routing-carried (SVFT/Jin); both
partial ⇒ split. Capability metric PRIMARY (the program's KL≠capability lesson); R_KL deferred.

    pg run infra.aws_pg.jobs.w12_causal --args="--n-refcoco 300 --mmlu-n-per-subject 20" \
        --type g6e.xlarge --s3-prefix results/w12_causal --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import pathlib

DELTA = 0.10   # §6C frozen "recovers ≥ (1−δ)·full" margin


def _rcap_ci(arm_scores, floor_scores, ceil_scores, n_boot, seed, min_range=0.05):
    """Paired bootstrap of R_cap = (arm − FLOOR)/(CEILING − FLOOR) over the shared eval items.
    Returns R_cap=None (INVALID) when |CEILING − FLOOR| < min_range — the axis has no dynamic range
    on which to normalise (e.g. CEILING ≈ FLOOR), so R_cap is meaningless rather than a spurious nan."""
    import numpy as np
    a, f, c = (np.asarray(x, float) for x in (arm_scores, floor_scores, ceil_scores))
    assert a.size == f.size == c.size, "arms must be scored on the SAME items (paired R_cap)"
    arm_m, floor_m, ceil_m = float(a.mean()), float(f.mean()), float(c.mean())
    if abs(ceil_m - floor_m) < min_range:
        return {"R_cap": None, "INVALID": f"|CEILING−FLOOR|={abs(ceil_m-floor_m):.3f} < {min_range} "
                f"(no dynamic range to normalise)", "arm_mean": arm_m, "floor_mean": floor_m,
                "ceil_mean": ceil_m}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        den = c[idx].mean() - f[idx].mean()
        draws.append((a[idx].mean() - f[idx].mean()) / den if den != 0 else np.nan)
    draws = np.asarray(draws, float)
    return {"R_cap": (arm_m - floor_m) / (ceil_m - floor_m),
            "lo": float(np.nanpercentile(draws, 2.5)), "hi": float(np.nanpercentile(draws, 97.5)),
            "arm_mean": arm_m, "floor_mean": floor_m, "ceil_mean": ceil_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-refcoco", type=int, default=300, help="RefCOCO eval items (primary axis)")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=20, help="blank-image MMLU cap (secondary)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import gain_routing as GR
    from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, _per_item_scores
    from infra.aws_pg.jobs.w11_legA_diagnostic import mmlu_blankimg

    assert torch.cuda.is_available(), "W12 §6C needs a GPU (generation)."
    device = "cuda"
    print(f"[W12-CAUSAL] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    # CEILING = PG2-pt (pg_after); FLOOR = base-decoder PG2 (pg_before). Bit-identical to E4/W11.
    pg_after, pg_before, proc, paths, _img_tok, gate = H.build_pair("base-pt", device)
    lm = pg_after.get_submodule(paths["language_model"])           # the model we mutate per arm
    after_sd = {k: v.detach().clone() for k, v in lm.state_dict().items()}            # W_after (PG2)
    before_sd = {k: v.detach().clone() for k, v in
                 pg_before.get_submodule(paths["language_model"]).state_dict().items()}  # W_before (base)
    del pg_before
    torch.cuda.empty_cache()

    matched = GR.matched_decoder_weight_matrices(after_sd, before_sd)   # {key:(wb=base, wa=PG2)}
    assert len(matched) == 26 * 7

    # Precompute the non-full arms' 182 decoder matrices (fp64 reconstruct → bf16). Norms / embed /
    # vision / projector are kept at PG2 (= CEILING) for every arm. 'in_span' = gain+routing (drops
    # only the out-of-span energy) — the proper ceiling for the gain-vs-routing question, since
    # gain-only and routing-only can NEVER reach full when f_perp>0 (out-of-span is in neither).
    print(f"[W12-CAUSAL] reconstructing floor/gain/routing/in_span decoder matrices ({len(matched)}) …")
    arm_mats = {m: {} for m in ("floor", "gain", "routing", "in_span")}
    for key, (wb, wa) in matched.items():
        arm_mats["floor"][key] = wb.detach().to(after_sd[key].dtype)
        arm_mats["gain"][key] = GR.reconstruct(wb, wa, "gain").to(after_sd[key].dtype)
        arm_mats["routing"][key] = GR.reconstruct(wb, wa, "routing").to(after_sd[key].dtype)
        arm_mats["in_span"][key] = GR.reconstruct(wb, wa, "in_span").to(after_sd[key].dtype)

    task = B.REFCOCO_TASK
    rc_rows = _eval_rows(task, args.n_refcoco, args.seed)
    print(f"[W12-CAUSAL] RefCOCO n={len(rc_rows)} split={task.eval_split}; MMLU n/subj={args.mmlu_n_per_subject}")

    def _load_arm(name):
        """Set the 182 decoder matrices to arm `name` (norms/embed kept at PG2); 'full' restores PG2."""
        sd = {k: v.clone() for k, v in after_sd.items()}
        if name != "full":
            for key, t in arm_mats[name].items():
                sd[key] = t.to(device)
        miss, unexp = lm.load_state_dict(sd, strict=True)
        assert not miss and not unexp, f"arm {name} load not strict-clean: {miss} {unexp}"

    refcoco_scores, mmlu_acc = {}, {}
    for arm in ("full", "floor", "gain", "routing", "in_span"):
        _load_arm(arm)
        pg_after.eval()
        _preds = _generate(pg_after, proc, task, rc_rows, device, args.batch)
        sc = _per_item_scores(task, rc_rows, _preds)
        samples = [{"score": float(s), "pred": p} for s, p in zip(sc, _preds)]
        refcoco_scores[arm] = sc
        mm = mmlu_blankimg(pg_after, proc, device, args.mmlu_n_per_subject, batch_size=4)
        mmlu_acc[arm] = mm
        print(f"[W12-CAUSAL] arm={arm:8s} RefCOCO acc={sum(sc)/len(sc):.4f} | MMLU={mm['acc_pct']:.2f}")
        if arm == "full":
            for smp in samples[:4]:
                print(f"    full sample score={smp['score']:.0f} pred={smp['pred']!r}")

    def _rcap_table(per_arm, ceil_arm, seed):
        """R_cap of gain/routing/in_span normalised against ceil_arm ('full' or 'in_span')."""
        return {arm: _rcap_ci(per_arm[arm], per_arm["floor"], per_arm[ceil_arm], args.boot, seed)
                for arm in ("gain", "routing", "in_span") if arm != ceil_arm}

    mmlu_per = {arm: mmlu_acc[arm]["per_item"] for arm in mmlu_acc}
    rcap = {
        # primary gain-vs-routing question: normalise against IN-SPAN (the reachable ceiling for
        # in-span-only arms). Also report vs FULL (shows how much out-of-span contributes).
        "refcoco_vs_inspan": _rcap_table(refcoco_scores, "in_span", args.seed),
        "refcoco_vs_full": _rcap_table(refcoco_scores, "full", args.seed + 3),
        "mmlu_vs_inspan": _rcap_table(mmlu_per, "in_span", args.seed + 7),
        "mmlu_vs_full": _rcap_table(mmlu_per, "full", args.seed + 11),
    }

    def _carrier(tbl, axis):
        g, r = tbl["gain"].get("R_cap"), tbl["routing"].get("R_cap")
        pre = f"({axis}) "
        if g is None or r is None:
            return pre + f"INVALID: {tbl['gain'].get('INVALID') or tbl['routing'].get('INVALID')}"
        if g >= 1 - DELTA and r < 1 - DELTA:
            return pre + f"gain-carried (gain R_cap={g:.2f} ≥ {1-DELTA}, routing={r:.2f})"
        if r >= 1 - DELTA and g < 1 - DELTA:
            return pre + f"routing-carried (routing R_cap={r:.2f} ≥ {1-DELTA}, gain={g:.2f})"
        if g >= 1 - DELTA and r >= 1 - DELTA:
            return pre + f"both recover (gain={g:.2f}, routing={r:.2f}) — redundant across components"
        return pre + f"split / neither alone (gain R_cap={g:.2f}, routing R_cap={r:.2f})"

    result = {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "delta": DELTA, "swap_gate": gate,
        "n_refcoco": len(rc_rows), "mmlu_n_per_subject": args.mmlu_n_per_subject,
        "axis_directions": {
            "refcoco": "installed-capability: R_cap=1 ⇒ arm reaches PG2 localisation; R_cap=0 ⇒ base decoder",
            "mmlu": "FORGETTING: CEILING=PG2-pt is at chance (W11-A) and FLOOR=base decoder is competent, "
                    "so R_cap=1 ⇒ arm forgot text like PG2-pt; R_cap=0 ⇒ arm retained base text capability"},
        "refcoco_raw_acc": {arm: sum(refcoco_scores[arm]) / len(refcoco_scores[arm]) for arm in refcoco_scores},
        "mmlu_raw_acc": {arm: mmlu_acc[arm]["acc_pct"] for arm in mmlu_acc},
        "R_cap": rcap,
        "carrier_refcoco_primary": _carrier(rcap["refcoco_vs_inspan"], "RefCOCO vs in_span"),
        "carrier_mmlu_secondary_forgetting": _carrier(rcap["mmlu_vs_inspan"], "MMLU forgetting vs in_span"),
        "inspan_sufficiency": {
            "refcoco_R_cap_inspan_vs_full": rcap["refcoco_vs_full"]["in_span"].get("R_cap"),
            "mmlu_R_cap_inspan_vs_full": rcap["mmlu_vs_full"]["in_span"].get("R_cap"),
            "note": "in_span vs full < 1 ⇒ out-of-span energy contributes to capability (¬negligible)"},
    }
    (out / "causal.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W12-CAUSAL] RefCOCO primary: {result['carrier_refcoco_primary']}")
    print(f"[W12-CAUSAL] MMLU secondary: {result['carrier_mmlu_secondary_forgetting']}")
    print(f"[W12-CAUSAL] in-span sufficiency: refcoco={result['inspan_sufficiency']['refcoco_R_cap_inspan_vs_full']} "
          f"mmlu={result['inspan_sufficiency']['mmlu_R_cap_inspan_vs_full']}")


if __name__ == "__main__":
    main()
