"""W9 §9.2-3 CONTROLS + FREEZE-OF-RECORD calibration (control data ONLY — no real Δa readout).

Builds the control trinity and measures the thresholds that §6 freezes BEFORE unblinding:
  * N3 random-delta floor: Δa_random = a(random-decoder PG2) − a_before → Diff-SAE → the 95th
    percentile of decoder out-of-span energy (the floor a real direction must beat). Computed per
    (layer, width) so unblind can apply the matching frozen floor.
  * N4 shuffle: Δa_shuf = shuffle(after) − before → Diff-SAE → out-of-span energy must COLLAPSE to
    the N3 floor (the dictionary fits position identity, not a rewrite).
  * Positive control: plant a known unit direction into a_after → Diff-SAE MUST recover it
    (max-cosine of some decoder direction to the planted direction near 1). Fail → instrument
    broken, HALT.

NO real Δa is touched here (the real readout is w9_unblind, run once after the freeze tag).

    python -m infra.aws_pg.jobs.w9_controls --out <dir> [--widths 16,32] [--seed 1337]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _pick_layer_sae(layer, width="16k", target_l0=70):
    """Exact-layer Gemma-Scope residual SAE (nearest L0 to target); fall to nearest layer if the
    exact layer has no SAE at this width. Returns (SAEChoice, exact_layer_bool)."""
    from infra.pg.gemma_scope import list_residual_saes
    cands = [c for c in list_residual_saes() if c.width == width]
    if not cands:
        raise RuntimeError(f"no Gemma-Scope residual SAE at width={width}")
    exact = [c for c in cands if c.layer == layer]
    if exact:
        return min(exact, key=lambda c: abs(c.average_l0 - target_l0)), True
    layers = sorted({c.layer for c in cands})
    nearest = min(layers, key=lambda L: abs(L - layer))
    at = [c for c in cands if c.layer == nearest]
    return min(at, key=lambda c: abs(c.average_l0 - target_l0)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="16,32", help="comma list of dict expansions")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-base-dict", action="store_true",
                    help="skip ρ_novel method (ii) (Gemma-Scope base dict not seeded)")
    ap.add_argument("--layers", default="", help="comma list to restrict layers (smoke); default all")
    ap.add_argument("--steps", type=int, default=0, help="override SAE train steps (smoke); 0=default")
    ap.add_argument("--pair", default="base-pt", choices=["base-pt", "pt-mix"],
                    help="before/after axis: base-pt (W9 Gemma→PT) or pt-mix (W6 PT→Mix)")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    steps_kw = {"steps": args.steps} if args.steps else {}

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    from infra.pg import diff_sae as DS
    from infra.pg import diff_metrics as DM
    H.hf_online_setup()                    # HF_TOKEN + online (gated models + dataset streaming)
    H.ensure_wandb()                       # dictionary_learning.training imports wandb at top

    layers_run = [int(x) for x in args.layers.split(",")] if args.layers else list(H.LAYERS)

    device = "cuda"
    print(f"[W9-CTL] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"widths={widths} layers={layers_run} steps={'default' if not args.steps else args.steps}")

    pg_after, pg_before, proc, paths, img_tok, gate = H.build_pair(args.pair, device)
    dec_delta = gate.get("decoder_max_abs_delta_vs_PG", gate.get("decoder_max_abs_delta_mix_vs_pt"))
    print(f"[W9-CTL] pair={args.pair} gate ok dec_delta={dec_delta:.3g}")
    mm = H.build_mm_batch(proc, img_tok, device)

    result = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "pair": args.pair, "swap_gate": gate, "widths": widths, "seed": args.seed,
        "frozen_rules": {
            "novel_criterion": "out_of_span_energy(method_i) > N3_pct95_floor AND survives §2.2 cross-check (FDR q=0.05)",
            "fvu_gate_max": 0.6, "dead_latent_frac_max": 0.5,
            "pca_var_threshold": 0.95, "crosscheck_n_perm": 1000, "crosscheck_max_candidates": 256,
            "fdr_q": 0.05,
        },
        "layers": {},
    }

    for L in layers_run:
        print(f"\n[W9-CTL] ===== LAYER {L} =====")
        acts_before, acts_after, is_img, held = H.harvest_paired_layer(pg_after, pg_before, L, mm, device)
        N, d = acts_before.shape
        print(f"[W9-CTL][L{L}] real paired tokens N={N} d={d} img_frac={float(is_img.float().mean()):.3f}")

        # before-PCA basis (the 'pre-existing feature space', method i reference).
        U, pca_info = DM.before_pca_basis(acts_before, var_threshold=0.95, seed=args.seed)
        print(f"[W9-CTL][L{L}] before-PCA r={pca_info['r']} expl_var={pca_info['explained_var']:.3f}")

        # base dict (method ii) — loaded once per layer.
        base_dirs = None; base_info = None
        if not args.no_base_dict:
            choice, is_exact = _pick_layer_sae(L)
            from infra.pg.gemma_scope import JumpReLUSAE
            sae = JumpReLUSAE.from_repo(choice)
            base_dirs = sae.W_dec.detach().cpu().float()          # (d_sae, d_model)
            base_info = {"rel_path": choice.rel_path, "exact_layer": bool(is_exact),
                         "d_sae": int(base_dirs.shape[0])}
            print(f"[W9-CTL][L{L}] base dict {choice.rel_path} exact={is_exact} d_sae={base_dirs.shape[0]}")

        # N3 random-delta: harvest a random-decoder after, build Δa_random.
        pg_random = H.build_random_after(pg_before, paths, device, seed=args.seed)
        ab_r, aa_r, _, _ = H.harvest_paired_layer(pg_random, pg_before, L, mm, device)
        # ab_r is before from pg_before-side hook = same before; aa_r is the random-decoder after.
        delta_n3 = (aa_r - ab_r)
        del pg_random; torch.cuda.empty_cache()

        layer_rec = {"n_tokens": int(N), "d_model": int(d),
                     "image_fraction": float(is_img.float().mean()),
                     "pca": pca_info, "base_dict": base_info, "widths": {}}

        # ---- N3 floor PER WIDTH (the only width-specific threshold unblind reads) ----
        n3_floors = {}
        for expansion in widths:
            ae_n3, cfg_n3 = DS.train_diff_sae(delta_n3, L, seed=args.seed, device=device, expansion=expansion, **steps_kw)
            q_n3 = DS.diff_sae_quality(ae_n3, delta_n3, device)
            dirs_n3 = DS.decoder_directions(ae_n3)
            e_n3 = DM.out_of_span_energy(dirs_n3, U).numpy()
            floor95 = DM.n3_floor_pct95(e_n3)
            n3_floors[expansion] = floor95
            print(f"[W9-CTL][L{L} x{expansion}] N3 fvu={q_n3['fvu']:.3f} dead={q_n3['dead_latent_frac']:.3f} "
                  f"floor95={floor95:.4f}")
            layer_rec["widths"][str(expansion)] = {
                "n3_fvu": q_n3["fvu"], "n3_dead_frac": q_n3["dead_latent_frac"],
                "n3_mean_l0": q_n3["mean_l0"], "n3_dict_size": cfg_n3["dict_size"],
                "n3_out_of_span_floor_pct95": floor95, "n3_out_of_span_mean": float(e_n3.mean()),
            }
            del ae_n3; torch.cuda.empty_cache()

        # ---- N4 collapse + positive recovery ONCE at the base width (width-independent
        #      instrument checks; running them at every width is wasteful). ----
        w0 = widths[0]; floor0 = n3_floors[w0]
        delta_n4 = DM.make_shuffled_delta(acts_before, acts_after, seed=args.seed)
        ae_n4, _ = DS.train_diff_sae(delta_n4, L, seed=args.seed, device=device, expansion=w0, **steps_kw)
        e_n4 = DM.out_of_span_energy(DS.decoder_directions(ae_n4), U).numpy()
        n4_above_floor = float((e_n4 > floor0).mean())          # must be ≈0.05 (collapse to floor)
        del ae_n4; torch.cuda.empty_cache()
        print(f"[W9-CTL][L{L}] N4 frac_above_N3floor(x{w0})={n4_above_floor:.4f} (collapse→≈0.05)")

        planted, u, plant_info = DM.plant_direction(acts_after, seed=args.seed)
        ae_pos, _ = DS.train_diff_sae(planted - acts_before, L, seed=args.seed, device=device, expansion=w0, **steps_kw)
        pos_dirs = DS.decoder_directions(ae_pos)
        recov_subspace, recov_cos = DM.recovered_planted_subspace(pos_dirs, u, topm=8)
        del ae_pos; torch.cuda.empty_cache()
        print(f"[W9-CTL][L{L}] positive recovery(x{w0}): subspace(top8)={recov_subspace:.3f} "
              f"max|cos|={recov_cos:.3f} (→1 = recovered; gate on subspace)")

        layer_rec["n4_positive"] = {
            "width": w0, "n4_frac_above_n3_floor": n4_above_floor,
            "positive_recovered_subspace": recov_subspace,
            "positive_recovered_max_cos": recov_cos, "positive_plant": plant_info,
        }
        result["layers"][str(L)] = layer_rec
        del acts_before, acts_after, delta_n3; torch.cuda.empty_cache()

    # ---- control gates (instrument-works verdict) ----
    # Gate on the SUBSPACE recovery (robust to BatchTopK feature-splitting); max|cos| kept for
    # transparency only.
    pos_ok = all(L["n4_positive"]["positive_recovered_subspace"] >= 0.8
                 for L in result["layers"].values())
    n4_ok = all(L["n4_positive"]["n4_frac_above_n3_floor"] <= 0.15
                for L in result["layers"].values())
    result["control_gates"] = {
        "positive_recovers": bool(pos_ok), "n4_collapses": bool(n4_ok),
        "instrument_works": bool(pos_ok and n4_ok),
    }
    (out / "controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[W9-CTL] control gates: positive_recovers={pos_ok} n4_collapses={n4_ok}")
    if not (pos_ok and n4_ok):
        raise RuntimeError(
            f"CONTROLS HALT: instrument failed its own controls "
            f"(positive_recovers={pos_ok}, n4_collapses={n4_ok}). Fix harness before unblinding.")
    print("[W9-CTL] controls passed → safe to freeze + unblind.")


if __name__ == "__main__":
    main()
