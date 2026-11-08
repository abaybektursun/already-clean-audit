"""W16 UNBLIND (single read; frozen rules of prereg §6 applied mechanically).

Reads the frozen w16_controls.json (CONTROLS_OK gate), evaluates AFTER once,
and emits verdicts under the post-critic frozen procedure — BH-FDR at q=0.05
over the tests actually run; FLOOR is the null side everywhere:

  Arm A   : paired per-item AFTER−FLOOR margin/AUC differences, confirmation
            half (`arm_a_paired_diff`) → two BH family members.
  B/C scr : 14 AFTER cells on the grid bucket; observed max ΔS vs the FROZEN
            FLOOR-side `screening_null_band` draws → promotion iff p < 0.05.
  B/C conf: (only if promoted) confirmation-half reads at the promoted cell on
            AFTER + FLOOR sides (3 seeds at σ if noise; len-1 if quant);
            statistic `seed_mean_delta_s`; null `cell_null_band` (FLOOR side,
            data collected here mechanically — control-side, frozen form);
            one-sided p → third BH member. Stability rule: seed spread
            (max−min) > band_width ⇒ non-result regardless of p.

Cycle integrity per prereg §5: full-scope bitwise restore assert per cell +
embed/1-D-norm spot checks + pre-confirmation capped unprobed re-read equality.

    python -m infra.aws_pg.jobs.w16_unblind --out <dir> \
        --controls-s3 s3://.../w16_controls.json [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

NOISE_SIGMAS = (1e-3, 3e-3, 1e-2, 3e-2)
NOISE_SEEDS = (1337, 1338, 1339)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W16 unblind needs GPU."
    device = "cuda"
    n_per = 2 if args.smoke else None
    print(f"[W16-UNB] torch={torch.__version__} smoke={args.smoke}")

    # ---- frozen controls (s3://… or a local path when chained via pipeline.py) ----
    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "w16_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local), "--region", "us-east-1"],
                       check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True, "controls did not pass — fix harness, do not unblind."
    assert bool(C["smoke"]) == bool(args.smoke), \
        f"smoke mismatch: controls={C['smoke']} unblind={args.smoke}"
    hashes = C["item_hashes"]
    b_pi = C["BEFORE"]["per_item"]
    f_pi = C["FLOOR"]["per_item"]
    f_logits = C["FLOOR"]["choice_logits"]
    correct = C["correct_idx"]
    band = C["screening_null_band"]
    bucket_hashes = W.mask_by(hashes, hashes, W.is_grid_subset)
    print(f"[W16-UNB] controls: BEFORE={C['BEFORE']['acc_pct']:.2f} FLOOR={C['FLOOR']['acc_pct']:.2f} "
          f"band_hi={band['band_hi']:.4f}")

    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- build pair: AFTER is the treatment; keep BEFORE only to rebuild FLOOR ----
    pg_after, pg_before, _proc, paths, _it, build_gate = H.build_before_after(device)
    lm_after = pg_after.get_submodule(paths["language_model"])

    # ---- AFTER full-split logits (THE single treatment read) ----
    after_cl = TE.mmlu_choice_logits(pg_after, tok, device, n_per_subject=n_per)
    assert after_cl["per_item_hash"] == hashes, "AFTER item order != frozen controls — HALT"
    assert after_cl["correct_idx"] == correct, "AFTER correct_idx != frozen controls — HALT"
    a_pi = after_cl["per_item"]
    a_logits = after_cl["choice_logits"]
    print(f"[W16-UNB] AFTER acc={after_cl['acc_pct']:.2f}")

    # ---- Arm A: paired AFTER−FLOOR diffs on the confirmation half ----
    conf = W.is_confirmation
    armA = W.arm_a_paired_diff(W.mask_by(a_logits, hashes, conf),
                               W.mask_by(f_logits, hashes, conf),
                               W.mask_by(correct, hashes, conf),
                               n_boot=args.n_boot, seed=1337)
    print(f"[W16-UNB] ArmA margin_diff={armA['margin_diff_point']:.4f} p={armA['margin_p_two_sided']:.4f} | "
          f"auc_diff={armA['auc_diff_point']:.4f} p={armA['auc_p_two_sided']:.4f}")

    # ---- B/C screening: 14 AFTER cells on the grid bucket ----
    cells_q = [8, 4] if not args.smoke else [8]
    cells_n = ([(s, sd) for s in NOISE_SIGMAS for sd in NOISE_SEEDS]
               if not args.smoke else [(1e-3, 1337)])
    snap = W.snapshot_probe_params(lm_after)
    scope = W.probe_param_names(lm_after)
    emb_name = [n for n, p in lm_after.named_parameters() if "embed_tokens" in n][0]
    norm_name = [n for n, p in lm_after.named_parameters() if p.ndim == 1][0]
    emb_ref = lm_after.get_parameter(emb_name).detach().cpu().clone()
    norm_ref = lm_after.get_parameter(norm_name).detach().cpu().clone()

    def integrity():
        for nm in scope:
            assert torch.equal(lm_after.get_parameter(nm).detach().cpu(), snap[nm]), \
                f"restore not bitwise on {nm} — HALT"
        assert torch.equal(lm_after.get_parameter(emb_name).detach().cpu(), emb_ref), "embed drifted"
        assert torch.equal(lm_after.get_parameter(norm_name).detach().cpu(), norm_ref), "norm drifted"

    after_cells = {}
    for bits in cells_q:
        W.quantize_decoder_(lm_after, bits)
        ev = TE.mmlu_choice_logits(pg_after, tok, device, n_per_subject=n_per,
                                   item_filter=W.is_grid_subset)
        W.restore_probe_params_(lm_after, snap)
        integrity()
        assert ev["per_item_hash"] == bucket_hashes or args.smoke, f"int{bits} bucket order mismatch"
        after_cells[f"int{bits}"] = ev["per_item"]
        print(f"[W16-UNB] AFTER int{bits}: acc={ev['acc_pct']:.2f}")
    for sigma, sd in cells_n:
        W.noise_decoder_(lm_after, sigma, seed=sd)
        ev = TE.mmlu_choice_logits(pg_after, tok, device, n_per_subject=n_per,
                                   item_filter=W.is_grid_subset)
        W.restore_probe_params_(lm_after, snap)
        integrity()
        assert ev["per_item_hash"] == bucket_hashes or args.smoke, f"σ={sigma} bucket order mismatch"
        after_cells[f"noise_{sigma:g}_s{sd}"] = ev["per_item"]
        print(f"[W16-UNB] AFTER σ={sigma:g} s{sd}: acc={ev['acc_pct']:.2f}")

    gb = W.is_grid_subset
    a_unp_bucket = W.mask_by(a_pi, hashes, gb)
    b_bucket = W.mask_by(b_pi, hashes, gb)
    f_bucket = W.mask_by(f_pi, hashes, gb)
    per_cell, max_pt, argmax_cell = W.grid_points(after_cells, a_unp_bucket, b_bucket, f_bucket)
    p_screen = W.one_sided_p(max_pt, band["draws"])
    promoted = bool(p_screen < 0.05)
    print(f"[W16-UNB] screening: max ΔS={max_pt:.4f} @ {argmax_cell} p={p_screen:.4f} promoted={promoted}")

    # ---- B/C confirmatory (mechanical, only if promoted) ----
    confirm = None
    if promoted:
        # pre-confirmation integrity: capped unprobed re-read equality
        cap_first = TE.mmlu_choice_logits(pg_after, tok, device, n_per_subject=2)
        cap_again = TE.mmlu_choice_logits(pg_after, tok, device, n_per_subject=2)
        assert cap_again["per_item"] == cap_first["per_item"], "pre-confirmation re-read drifted — HALT"

        if argmax_cell.startswith("int"):
            probe_list = [("q", int(argmax_cell[3:]), None)]
        else:
            sigma = float(argmax_cell.split("_")[1])
            probe_list = [("n", sigma, sd) for sd in (NOISE_SEEDS if not args.smoke else (1337,))]

        def conf_reads(model, lm, side_snap):
            outs = []
            for kind, a, sd in probe_list:
                if kind == "q":
                    W.quantize_decoder_(lm, a)
                else:
                    W.noise_decoder_(lm, a, seed=sd)
                ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per,
                                           item_filter=W.is_confirmation)
                W.restore_probe_params_(lm, side_snap)
                outs.append(ev["per_item"])
            return outs

        after_conf_probed = conf_reads(pg_after, lm_after, snap)
        integrity()

        pg_floor = H.build_random_after(pg_before, paths, device, seed=int(C["FLOOR"]["decoder_seed"]))
        lm_floor = pg_floor.get_submodule(paths["language_model"])
        floor_snap = W.snapshot_probe_params(lm_floor)
        floor_conf_probed = conf_reads(pg_floor, lm_floor, floor_snap)

        a_unp_conf = W.mask_by(a_pi, hashes, conf)
        b_conf = W.mask_by(b_pi, hashes, conf)
        f_conf = W.mask_by(f_pi, hashes, conf)
        stat = W.seed_mean_delta_s(after_conf_probed, a_unp_conf, b_conf, f_conf)
        cell_band = W.cell_null_band(floor_conf_probed, f_conf, b_conf, f_conf,
                                     n_boot=args.n_boot, seed=1337)
        p_conf = W.one_sided_p(stat, cell_band["draws"])
        per_seed = [W.delta_s(a, a_unp_conf, b_conf, f_conf) for a in after_conf_probed]
        spread = max(per_seed) - min(per_seed)
        unstable = bool(spread > cell_band["band_width"])
        confirm = {"cell": argmax_cell, "stat_seed_mean": stat, "per_seed": per_seed,
                   "seed_spread": spread, "band_width": cell_band["band_width"],
                   "unstable_non_result": unstable, "p_one_sided": p_conf,
                   "cell_band_hi": cell_band["band_hi"],
                   "n_draws_dropped": cell_band["n_draws_dropped"]}
        print(f"[W16-UNB] confirm @ {argmax_cell}: stat={stat:.4f} p={p_conf:.4f} "
              f"spread={spread:.4f} vs width={cell_band['band_width']:.4f} unstable={unstable}")

    # ---- governing procedure: BH over the family actually run ----
    family = {"armA_margin": armA["margin_p_two_sided"], "armA_auc": armA["auc_p_two_sided"]}
    if confirm is not None:
        family["bc_promoted"] = confirm["p_one_sided"]
    bh = W.bh_fdr(family, q=args.q)

    metric_induced = bool(bh["armA_margin"]["significant"] or bh["armA_auc"]["significant"])
    if confirm is None:
        bc_verdict = "H_deletion stands (no promotion — registered-prediction branch)"
    elif confirm["unstable_non_result"]:
        bc_verdict = "non-result (seed spread > band width; house stability rule)"
    elif bh["bc_promoted"]["significant"]:
        bc_verdict = "H_suppression supported (confirmation BH-significant) — W14 re-grade gate fires"
    else:
        bc_verdict = "H_deletion stands (discovery spike unreplicated)"
    armA_verdict = ("metric-induced (graded)" if metric_induced
                    else "H_zero-real stands (bounded-difference read)")

    result = {
        "torch": torch.__version__, "smoke": args.smoke, "controls_s3": args.controls_s3,
        "build_gate": build_gate,
        "AFTER": {"acc_pct": after_cl["acc_pct"], "per_item": a_pi, "choice_logits": a_logits},
        "arm_a": armA,
        "screening": {"per_cell_delta_s": per_cell, "max": max_pt, "argmax_cell": argmax_cell,
                      "p_one_sided": p_screen, "promoted": promoted,
                      "band_hi_frozen": band["band_hi"]},
        "confirmation": confirm,
        "bh_family": bh, "q": args.q,
        "VERDICT_arm_a": armA_verdict,
        "VERDICT_bc": bc_verdict,
        "prediction_grid": {
            "armA_margin_diff": f"{armA['margin_diff_point']:.4f} (p_adj={bh['armA_margin']['p_adj']:.4f})",
            "armA_auc_diff": f"{armA['auc_diff_point']:.4f} (p_adj={bh['armA_auc']['p_adj']:.4f})",
            "bc_screening": f"max={max_pt:.4f} @ {argmax_cell}, p={p_screen:.4f}",
            "bc_confirmation": ("n/a (not promoted)" if confirm is None
                                 else f"stat={confirm['stat_seed_mean']:.4f} p={confirm['p_one_sided']:.4f}"),
        },
    }
    (out / "w16_unblind.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[W16-UNB] VERDICT arm A: {armA_verdict}")
    print(f"[W16-UNB] VERDICT B/C : {bc_verdict}")


if __name__ == "__main__":
    main()
