"""W10 ρ_novel^vis controls + freeze-calibration (control data ONLY — no real Δa_vis readout).

The saev leg of the vision emergent-fraction (prereg §6.2): is PG training adding vision
features SigLIP-base lacked, or is the tower featurally preserved? Operationalised as the W9
Diff-SAE yardstick on the PAIRED vision delta —

    before = siglip-base pre-projector patch acts (V0)
    after  = pg2-vision  pre-projector patch acts (V1)        (same SigLIP processor + images
    Δa_vis = after − before                                     + patch order ⇒ row-paired)

Builds the control trinity + the per-width N3 floor that §6 freezes BEFORE unblinding:
  * N3 random-encoder floor: Δa_vis_n3 = a(random-init SigLIP) − a(SigLIP-base) → Diff-SAE → the
    95th pct of decoder out-of-span energy vs the SigLIP-base PCA span (the floor a real PG2
    direction must beat). Per width (C4).
  * N4 shuffle: shuffle(after) − before → out-of-span energy must COLLAPSE to the N3 floor.
  * Positive control: plant a known unit direction into a_after → Diff-SAE MUST recover its span
    (recovered_planted_subspace ≥ 0.8). Fail → instrument broken, HALT.

NO real Δa_vis is read here (that is w10_unblind, once, after the freeze tag).

    python -m infra.aws_pg.jobs.w10_controls --out <dir> [--widths 16,32] [--n-per-probe 384]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="16,32", help="comma list of dict expansions (C4)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-per-probe", type=int, default=384)
    ap.add_argument("--steps", type=int, default=0, help="override SAE train steps (smoke); 0=default")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    steps_kw = {"steps": args.steps} if args.steps else {}

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # hf_online_setup MUST precede any hf/transformers import (offline constant cached at import).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    H.ensure_wandb()                       # dictionary_learning.training imports wandb at top

    import numpy as np
    import torch
    from infra.pg import diff_metrics as DM
    from infra.pg import diff_sae as DS
    from infra.pg import vision_harvest as V

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[W10-CTL] device={device} widths={widths} n_per_probe={args.n_per_probe} "
          f"steps={'default' if not args.steps else args.steps}")

    # --- harvest the three pre-projector encoders on the SAME images (row-paired) ---
    probe = V.load_probe_images(n_per_probe=args.n_per_probe)
    images, held_img = probe["images"], probe["held"]
    print(f"[W10-CTL] probe unique images={len(images)} held={int(held_img.sum())}")
    h_sig = V.harvest_encoder("siglip-base", images, held_img, device)
    h_pg = V.harvest_encoder("pg2-vision", images, held_img, device)
    h_rnd = V.harvest_encoder("random", images, held_img, device)
    before = h_sig["pre"]; after = h_pg["pre"]; rnd = h_rnd["pre"]      # [N,1152] each, row-paired
    held_patch = h_sig["held_patch"]
    N, d = before.shape
    assert after.shape == before.shape == rnd.shape, "encoder pre-act shapes must match for pairing"

    # discovery↔confirmation firewall (per-patch, from per-image held) via the tested utility.
    split = DM.enforce_split(held_patch, name="w10-vis")
    disc = torch.from_numpy(split["disc_mask"])
    print(f"[W10-CTL] N={N} d={d} disc={split['n_disc']} conf={split['n_conf']}")

    # before-PCA span (the pre-existing SigLIP-base feature space; ρ_novel^vis method-i reference).
    U, pca_info = DM.before_pca_basis(before[disc], var_threshold=0.95, seed=args.seed)
    print(f"[W10-CTL] before-PCA r={pca_info['r']} expl_var={pca_info['explained_var']:.3f}")

    delta_n3 = (rnd - before)                                          # N3 random-encoder delta
    result = {
        "device": device, "pair": "siglip-base→pg2-vision", "point": "pre-projector",
        "widths": widths, "seed": args.seed, "n_tokens": int(N), "d_model": int(d),
        "n_disc": split["n_disc"], "n_conf": split["n_conf"], "pca": pca_info,
        "encoders": {k: v["info"] for k, v in
                     {"siglip-base": h_sig, "pg2-vision": h_pg, "random": h_rnd}.items()},
        "frozen_rules": {
            "novel_criterion": "out_of_span_energy(method_i) > N3_pct95_floor AND survives §2.2 cross-check (FDR q=0.05)",
            "fvu_gate_max": 0.6, "dead_latent_frac_max": 0.5,
            "pca_var_threshold": 0.95, "crosscheck_n_perm": 1000, "crosscheck_max_candidates": 256,
            "fdr_q": 0.05, "positive_subspace_gate": 0.8, "n4_collapse_max": 0.15,
        },
        "widths_detail": {},
    }

    # ---- N3 floor PER WIDTH (the only width-specific threshold unblind reads) ----
    n3_floors = {}
    for expansion in widths:
        ae_n3, cfg_n3 = DS.train_diff_sae(delta_n3[disc], 0, seed=args.seed, device=device,
                                          expansion=expansion, **steps_kw)
        q_n3 = DS.diff_sae_quality(ae_n3, delta_n3[disc], device)
        e_n3 = DM.out_of_span_energy(DS.decoder_directions(ae_n3), U).numpy()
        floor95 = DM.n3_floor_pct95(e_n3)
        n3_floors[expansion] = floor95
        print(f"[W10-CTL][x{expansion}] N3 fvu={q_n3['fvu']:.3f} dead={q_n3['dead_latent_frac']:.3f} "
              f"floor95={floor95:.4f}")
        result["widths_detail"][str(expansion)] = {
            "n3_fvu": q_n3["fvu"], "n3_dead_frac": q_n3["dead_latent_frac"],
            "n3_mean_l0": q_n3["mean_l0"], "n3_dict_size": cfg_n3["dict_size"],
            "n3_out_of_span_floor_pct95": floor95, "n3_out_of_span_mean": float(e_n3.mean()),
        }
        del ae_n3
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- N4 collapse + positive recovery ONCE at base width (width-independent checks) ----
    w0 = widths[0]; floor0 = n3_floors[w0]
    delta_n4 = DM.make_shuffled_delta(before, after, seed=args.seed)
    ae_n4, _ = DS.train_diff_sae(delta_n4[disc], 0, seed=args.seed, device=device, expansion=w0, **steps_kw)
    e_n4 = DM.out_of_span_energy(DS.decoder_directions(ae_n4), U).numpy()
    n4_above_floor = float((e_n4 > floor0).mean())                    # must collapse to ≈0.05
    del ae_n4
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[W10-CTL] N4 frac_above_N3floor(x{w0})={n4_above_floor:.4f} (collapse→≈0.05)")

    planted, u, plant_info = DM.plant_direction(after, seed=args.seed)
    ae_pos, _ = DS.train_diff_sae((planted - before)[disc], 0, seed=args.seed, device=device,
                                  expansion=w0, **steps_kw)
    recov_subspace, recov_cos = DM.recovered_planted_subspace(DS.decoder_directions(ae_pos), u, topm=8)
    del ae_pos
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[W10-CTL] positive recovery(x{w0}): subspace(top8)={recov_subspace:.3f} max|cos|={recov_cos:.3f}")

    result["n4_positive"] = {
        "width": w0, "n4_frac_above_n3_floor": n4_above_floor,
        "positive_recovered_subspace": recov_subspace,
        "positive_recovered_max_cos": recov_cos, "positive_plant": plant_info,
    }

    pos_ok = recov_subspace >= 0.8
    n4_ok = n4_above_floor <= 0.15
    result["control_gates"] = {
        "positive_recovers": bool(pos_ok), "n4_collapses": bool(n4_ok),
        "instrument_works": bool(pos_ok and n4_ok),
    }
    (out / "controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[W10-CTL] control gates: positive_recovers={pos_ok} n4_collapses={n4_ok}")
    if not (pos_ok and n4_ok):
        raise RuntimeError(
            f"W10 CONTROLS HALT: instrument failed its own controls (positive_recovers={pos_ok}, "
            f"n4_collapses={n4_ok}). Fix harness before unblinding.")
    print("[W10-CTL] ρ_novel^vis controls passed → safe to freeze + unblind this leg.")


if __name__ == "__main__":
    main()
