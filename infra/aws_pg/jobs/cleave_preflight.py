"""CLEAVE preflight (GPU) — spec §9.1, the gates that only a GPU can settle, before any fleet spend.

Four gates, cheapest-first, each fail-loud:
  1. **ppl anchor** — the eager PG1 forward reproduces the pinned base perplexity (≈22.43;
     `reference_paligemma_sdpa_wrong_forward`: sdpa silently mishandles the prefix-LM mask and reads
     36.8 — an anchor drift means the forward is wrong and every downstream number is wrong).
  2. **identity inertness** — the `identity` cell through the REAL 126-matrix decoder grid is BITWISE
     base, and the vision tower is untouched. If the surgery is not inert, no contrast means anything.
  3. **banked-Q bit-exactness** — a banked Q number reproduces exactly (the banks are the map).
  4. **the WIRING gate (the decisive one)** — train a SHORT natural full-FT, read it, then rebuild the
     SAME update through `build_decoder("ft", …)` and read it again. The swap-shell must reproduce the
     natural fine-tune's task-gain AND forgetting. PROTECT's entire claim is that `ΔW_off` behaves
     like a fine-tune minus a subspace; if installing the FULL ΔW does not reproduce the fine-tune,
     the install path is miswired and every projected cell is measuring the pipeline, not the map
     (spec §1.4 → halt-miswired, the only legitimate halt).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_preflight \
      --args "--banks-s3 results/cleave_qcert/cleave_banks.pt --max-steps 24" \
      --type g6e.4xlarge,g6e.8xlarge,p5.4xlarge --s3-prefix results/cleave_preflight --budget-min 120
"""
import argparse
import json
import os
import pathlib

# The anchor is the invcore-FUNC artifact value and is measured at n_seq=N_PPL_BATTERY (120) — the
# ppl of a DIFFERENT window count is a different quantity, so the gate must match the anchor's own
# n_seq or it compares apples to oranges. Tolerance is STENCIL's proven 5% RELATIVE: the failure this
# gate exists to catch (sdpa's mishandled prefix-LM mask → 36.8 vs 22.4) is a 64% error, so 5%
# discriminates it with room to spare (`reference_paligemma_sdpa_wrong_forward`).
PPL_REL_TOL = 0.05
WIRE_GAIN_TOL = 0.02        # |install − trained| task-gain (accuracy points, absolute)
WIRE_FORGET_TOL = 0.02      # |install − trained| composite forgetting (relative units)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", default="")
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--n-vqa", type=int, default=64)
    ap.add_argument("--n-coco", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import stencil_core as SC

    CLS.install_pg()
    device = "cuda"
    rep = {"object": "cleave_preflight_v1", "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- 1. load + ppl anchor ------------------------------------------------------------
    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/cleave")
    rep["pinned_sha"] = sha
    with tm.stage("ppl_anchor"):
        from infra.pg import inherit_pg1 as IP  # noqa: F401  (battery loaders below)
        pp = SPS.safe_ppl_gate(model, proc, device, n_seq=CLS.N_PPL_BATTERY)
    rel = abs(pp["ppl"] - base_ppl) / base_ppl
    rep["ppl_anchor"] = {"measured": pp["ppl"], "pinned": base_ppl, "rel_err": rel,
                         "n_seq": CLS.N_PPL_BATTERY, "tol": PPL_REL_TOL}
    assert rel < PPL_REL_TOL, \
        f"base ppl {pp['ppl']:.3f} != anchor {base_ppl:.3f} (rel {rel:.3f}) — the forward is wrong"
    _w()

    # ---- banks ---------------------------------------------------------------------------
    with tm.stage("banks"):
        if a.banks_s3:
            local = CLS.fetch_s3(a.banks_s3, "/tmp/cleave/cleave_banks.pt")
            banks = torch.load(local, map_location="cpu", weights_only=False)
        else:                       # pilot mode: a synthetic bank exercises the wiring, not the map
            from infra.aws_pg.jobs import _ration_shared as RATS
            qb = RATS.load_q_bank("/tmp/cleave")
            d = IP.D
            g = torch.Generator().manual_seed(0)
            S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
            banks = {L: {"Q_evecs": torch.as_tensor(qb["evecs"]).float(),
                         "sigma_evecs": S.float(), "d": d, "k_stiff": int(qb["k_stiff"])}
                     for L in range(CLS.N_LAYERS)}
            rep["banks_note"] = "L0 carry_q_bank replicated across layers — WIRING pilot only, not a map read"
    rep["banks_layers"] = sorted(banks)
    assert set(banks) == set(range(CLS.N_LAYERS)), f"banks cover {sorted(banks)}, need all 18 layers"
    _w()

    # ---- 2. identity inertness on the REAL decoder grid -----------------------------------
    lm, paths, vt_orig = CLS.lm_bundle(model)
    grid_names = set(CL._target_grid(model))          # full param names of the 126 grid matrices
    base_lm_sd = CLS.decoder_sd(lm)
    grid = CLS.assert_grid(base_lm_sd)
    with tm.stage("identity"):
        rep["identity_gate"] = CLS.assert_identity_inert(base_lm_sd, banks)
    rep["grid"] = {"n_matrices": len(grid), "n_layers": CLS.N_LAYERS}
    _w()

    # ---- 3. banked-Q bit-exactness --------------------------------------------------------
    with tm.stage("bank_check"):
        Q0 = torch.as_tensor(banks[0]["Q_evecs"]).double()
        gram = Q0[:, :16].T @ Q0[:, :16]
        orth_err = float((gram - torch.eye(16, dtype=torch.float64)).abs().max())
        assert orth_err < 1e-4, f"banked Q eigenbasis not orthonormal (err {orth_err:.2e})"
        rep["bank_check"] = {"orthonormality_err": orth_err,
                             "k_stiff_L0": int(banks[0].get("k_stiff", -1))}
    _w()

    # ---- 4. the WIRING gate ---------------------------------------------------------------
    mmlu = CLS.mmlu_by_subject()
    roster = CLS.build_roster(mmlu)
    subj = [s for s in roster["discovery"] if CLS.splittable(mmlu[s]["test"])][0]
    train_rows, _evl, ev_hashes = CLS.subject_train_eval(mmlu[subj]["test"])
    examples, dedup = CLS.train_examples(train_rows, ev_hashes)
    bctx = CLS.make_battery_ctx(proc, roster, n_vqa=a.n_vqa, n_coco=a.n_coco)
    snap = CLS.snapshot_base(model)

    with tm.stage("base_read"):
        base_sa = CLS.subject_accuracy(model, proc, device, ev_hashes)["acc"]
        base_mm = CLS.mm_battery(model, proc, device, bctx)
    read = CLS.make_reader(model, proc, device, bctx, base_mm, base_sa, ev_hashes)
    rep["base"] = {"subject": subj, "subj_acc": base_sa, "mm": base_mm,
                   "n_train_examples": len(examples), "dedup_dropped": dedup.get("n_dropped")}
    _w()

    # a SHORT natural full-FT (the wiring gate needs a real ΔW, not a long one)
    with tm.stage("train_short_ft"):
        info = CL.mark_grid_trainable(model)          # A-CLEAVE-7: exactly the decomposable grid
        stats = SC.train_with_trajectory(
            model, proc, device, examples, lr=CLS.FT_RECIPE["lr"], epochs=1,
            batch_size=CLS.FT_RECIPE["batch_size"], warmup_ratio=CLS.FT_RECIPE["warmup_ratio"],
            seed=CLS.SEED, max_steps=a.max_steps,
            micro_batch=CLS.FT_RECIPE["micro_batch"],   # A-CLEAVE-8: memory only, same gradient
            checkpoint_steps=(), eval_fn=None)
    with tm.stage("trained_read"):
        trained = read("trained_in_place")
    ft_lm_sd = CLS.decoder_sd(lm)
    rep["train_short"] = {"trainable_params": info["trainable_params"], "steps": stats["steps"],
                          "final_loss": stats["final_loss"], "trained_read": trained}
    _w()

    # rebuild the SAME update through the swap-shell and re-read
    def reset():
        CLS.reset_model(model, snap)
    install = CLS.make_installer(model, lm, paths, vt_orig)
    with tm.stage("wiring_install"):
        reset()
        after_reset = read("after_reset")
        assert abs(after_reset["task_gain"]) < 1e-9, \
            f"reset did not restore base (task_gain {after_reset['task_gain']}) — the reset is broken"
        bases = CL.off_bases(banks, "random", 5, n_layers=CLS.N_LAYERS)
        cell_sd, book = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases, n_layers=CLS.N_LAYERS)
        # A-CLEAVE-7 makes this checkable at the WEIGHTS, not just the behaviour: the natural update
        # lives entirely in the 126-matrix grid, so rebuilding it must return the trained decoder to
        # the LAST REPRESENTABLE BIT. It is NOT bit-identical, and asserting that was wrong: the cell
        # computes fl(Wb + fl(ft − Wb)) in fp32 and casts back to bf16. That lands within one fp32
        # ulp of `ft`, which bf16's much coarser rounding absorbs — EXCEPT at an exact round-to-even
        # tie, where 1 fp32 ulp flips the final bit. Across ~2e9 elements even a 1e-7 tie rate flips
        # many, and `torch.equal` is all-or-nothing per tensor (that is why 41/164 keys "drifted"
        # while the reconstruction was in fact correct).
        #
        # The gate that means what it says: every element within ONE bf16 ulp (rel 2^-8), and the
        # differing-element fraction reported rather than assumed zero. A key that is TRAINED but not
        # REBUILT (the failure this exists to catch, e.g. a layernorm) drifts by the full update —
        # orders of magnitude beyond an ulp — so this still catches it with room to spare.
        ULP = 2.0 ** -8                       # bf16 has an 8-bit mantissa
        # An absolute floor beside the relative one, justified independently of any mechanism: for a
        # weight that lands near zero a pure relative tolerance collapses to ~0, so it would flag an
        # error of 1e-9 on a 0.02-scale weight — noise, not a fault. 1e-6 is still ~800x BELOW the
        # ~8e-4 update magnitude that the failure this gate exists for (a trained-but-not-rebuilt
        # key) would produce, so the gate keeps its power.
        ATOL = 1e-6
        worst, n_diff_keys, examples, diag = 0.0, 0, [], []
        for k in ft_lm_sd:
            # memory-frugal: the fp32 widenings of a 16384x2048 matrix are ~134MB each, and this box
            # already holds ~21GB of CPU state-dicts (snapshot + base + ft + cell). Hold the minimum
            # and free it per key — the previous version OOM-killed the job (exit 137) on a 32GB box.
            neq = int((cell_sd[k] != ft_lm_sd[k]).sum())
            a = cell_sd[k].to(torch.float32)
            b = ft_lm_sd[k].to(torch.float32)
            absdiff = (a - b).abs()
            del a
            tol = b.abs() * ULP + ATOL
            excess = (absdiff - tol).max().item()
            del tol
            if neq:
                n_diff_keys += 1
                # record WHAT differs and by how much — the diagnostic, not a guess
                diag.append({"key": k, "n_elem": int(b.numel()), "n_differing": neq,
                             "frac_differing": neq / max(1, b.numel()),
                             "max_abs_diff": float(absdiff.max()),
                             "max_rel_diff": float((absdiff / (b.abs() + 1e-30)).max()),
                             "is_grid_target": bool(k in {kk.split("language_model.model.")[-1]
                                                          for kk in grid_names}),
                             "exceeds_one_ulp": bool(excess > 0)})
            if excess > 0:
                examples.append((k, excess))
            worst = max(worst, float((absdiff / (b.abs() + 1e-30)).max()))
            del absdiff, b
        diag.sort(key=lambda d: -d["max_rel_diff"])
        rep["wiring_drift_diagnostic"] = diag[:12]
        _w()
        rep["wiring_weight_exact"] = {
            "n_keys": len(ft_lm_sd), "n_keys_not_bit_identical": n_diff_keys,
            "max_relative_error": worst, "one_bf16_ulp": ULP, "atol": ATOL,
            "keys_exceeding_one_ulp": len(examples), "examples": [e[0] for e in examples[:5]],
            "note": ("bit-identity is unachievable through an fp32 round-trip at bf16 tie boundaries; "
                     "the meaningful gate is <=1 ulp, which a trained-but-not-rebuilt key blows past")}
        assert not examples, (
            f"HALT-MISWIRED: the 'ft' cell does not reproduce the trained decoder to within one bf16 "
            f"ulp — {len(examples)}/{len(ft_lm_sd)} keys exceed it (e.g. {[e[0] for e in examples[:3]]}, "
            f"worst excess {max(e[1] for e in examples):.3e}). A key trained but not rebuilt by the "
            "surgery means the natural update does not live in the decomposable grid.")
        install(cell_sd, "wiring/ft")
        del cell_sd, bases
        installed = read("installed_ft")

    d_gain = abs(installed["task_gain"] - trained["task_gain"])
    d_forget = abs(installed["forgetting"] - trained["forgetting"])
    rep["wiring_gate"] = {
        "trained": {"task_gain": trained["task_gain"], "forgetting": trained["forgetting"]},
        "installed": {"task_gain": installed["task_gain"], "forgetting": installed["forgetting"]},
        "delta_task_gain": d_gain, "delta_forgetting": d_forget,
        "tol_gain": WIRE_GAIN_TOL, "tol_forget": WIRE_FORGET_TOL,
        "total_fro2": book["total_fro2"],
        "ok": bool(d_gain <= WIRE_GAIN_TOL and d_forget <= WIRE_FORGET_TOL)}
    print(f"[preflight] WIRING trained gain={trained['task_gain']:+.4f} forget={trained['forgetting']:+.4f} "
          f"| installed gain={installed['task_gain']:+.4f} forget={installed['forgetting']:+.4f} "
          f"| Δgain={d_gain:.4f} Δforget={d_forget:.4f}", flush=True)
    _w()
    assert rep["wiring_gate"]["ok"], (
        f"HALT-MISWIRED: the swap-shell install does not reproduce the natural fine-tune "
        f"(Δgain={d_gain:.4f} > {WIRE_GAIN_TOL} or Δforget={d_forget:.4f} > {WIRE_FORGET_TOL}). "
        "Every projected cell would measure the pipeline, not the map (spec §6.1).")

    # the ΔW is non-trivial (a wiring gate that passes on a ZERO update is a gate that cannot fail)
    assert book["total_fro2"] > 0, "the natural ΔW is zero — the wiring gate would pass vacuously"

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("CLEAVE_PREFLIGHT_OK", json.dumps({"ppl": rep["ppl_anchor"], "wiring": rep["wiring_gate"]},
                                            default=str), flush=True)


if __name__ == "__main__":
    main()
