"""W10 ρ_novel^vis UNBLIND (§9 step 5 — the real Δa_vis readout, run ONCE after the freeze).

The saev emergent-fraction (prereg §6.2): does PG training install vision directions SigLIP-base
lacked? Real paired delta —
    before = siglip-base pre-projector patch acts (V0)
    after  = pg2-vision  pre-projector patch acts (V1)
    Δa_vis = after − before
ρ_novel^vis = fraction of Diff-SAE(Δa_vis) decoder directions whose out-of-span energy vs the
SigLIP-base PCA span exceeds the FROZEN N3 floor (read from controls.json) AND survives the §2.2
direct-activation cross-check on the SEALED confirmation half (out-of-sample). ρ_novel^vis ≤ floor
→ vision tower featurally preserved (coherent with W1 ΔW 0.042 + A1 S̄_A); ≫ floor + cross-check
→ H_vision-repurposed (report it).

Reuses diff_sae + diff_metrics (all W9-validated); firewall via DM.enforce_split. The N3 floor is
NOT recomputed — it is read from the frozen controls.json (provenance: match-not-transcribe).

    python -m infra.aws_pg.jobs.w10_unblind --out <dir> \
        --controls-s3 s3://pg-lineage-769063704696/results/w10_controls/out/controls.json \
        --freeze-sha <w10-vision-prereg sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3",
                    default="s3://pg-lineage-769063704696/results/w10_controls/out/controls.json")
    ap.add_argument("--freeze-sha", default="")
    ap.add_argument("--widths", default="16,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-per-probe", type=int, default=384)
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    H.ensure_wandb()

    import numpy as np
    import torch
    from infra.pg import diff_metrics as DM
    from infra.pg import diff_sae as DS
    from infra.pg import vision_harvest as V

    # frozen N3 floors (read, NOT recomputed — CLAUDE.md provenance discipline).
    ctl_local = out / "controls_ref.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctl_local), "--region", "us-east-1"], check=True)
    controls = json.loads(ctl_local.read_text())
    floors = {int(w): float(d["n3_out_of_span_floor_pct95"]) for w, d in controls["widths_detail"].items()}
    if not controls["control_gates"]["instrument_works"]:
        raise RuntimeError("controls.json says instrument_works=False — do not unblind a broken instrument.")
    print(f"[W10-UNB] frozen N3 floors={floors} freeze_sha={args.freeze_sha}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = V.load_probe_images(n_per_probe=args.n_per_probe)
    images, held_img = probe["images"], probe["held"]
    h_sig = V.harvest_encoder("siglip-base", images, held_img, device)
    h_pg = V.harvest_encoder("pg2-vision", images, held_img, device)
    before, after = h_sig["pre"], h_pg["pre"]                  # [N,1152] row-paired
    held_patch = h_sig["held_patch"]
    N, d = before.shape
    delta = after - before                                    # REAL Δa_vis (first observation)

    split = DM.enforce_split(held_patch, name="w10-unb")
    disc = torch.from_numpy(split["disc_mask"]); conf_idx = split["conf_idx"]
    delta_disc = delta[disc]
    U, pca_info = DM.before_pca_basis(before[disc], var_threshold=0.95, seed=1337)
    print(f"[W10-UNB] N={N} d={d} disc={split['n_disc']} conf={conf_idx.size} pca_r={pca_info['r']}")

    # sealed confirmation-half acts for the §2.2 cross-check (subsampled for permutation cost).
    rng = np.random.default_rng(1337)
    sel = conf_idx[rng.choice(conf_idx.size, min(20000, conf_idx.size), replace=False)]
    ab_conf = before.numpy()[sel]; aa_conf = after.numpy()[sel]

    result = {"device": device, "pair": "siglip-base→pg2-vision", "point": "pre-projector",
              "freeze_sha": args.freeze_sha, "controls_s3": args.controls_s3,
              "n_tokens": int(N), "d_model": int(d), "n_disc": split["n_disc"],
              "n_conf": int(conf_idx.size), "pca": pca_info, "frozen_floors": floors, "grid": {}}

    for expansion in widths:
        floor95 = floors[expansion]
        rho_by_seed = []
        for seed in seeds:
            ae, _ = DS.train_diff_sae(delta_disc, 0, seed=seed, device=device, expansion=expansion)
            q = DS.diff_sae_quality(ae, delta_disc, device)
            dirs = DS.decoder_directions(ae)
            dirs_np = dirs.numpy()
            e_real = DM.out_of_span_energy(dirs, U).numpy()
            cand_idx = np.where(e_real > floor95)[0]
            cand_idx = cand_idx[np.argsort(-e_real[cand_idx])][:256]
            survive_full = np.zeros(dirs_np.shape[0], dtype=bool)
            cc = {"n_candidates": int(cand_idx.size), "n_survive": 0}
            if cand_idx.size > 0:
                ccr = DM.crosscheck_directions(dirs_np[cand_idx], ab_conf, aa_conf,
                                               n_perm=1000, seed=1337, fdr_q=0.05, max_candidates=256)
                survived = ccr["survives_crosscheck"]
                survive_full[cand_idx[:survived.size]] = survived
                cc = {"n_candidates": int(ccr["n_candidates"]), "n_survive": int(ccr["n_survive"]),
                      "rand_ratio_median": float(ccr["rand_ratio_median"])}
            mask, n_novel, rho = DM.novel_flags(e_real, floor95, survive_full)
            rho_by_seed.append(rho)
            np.savez_compressed(out / f"features_x{expansion}_s{seed}.npz",
                                out_of_span_energy=e_real, novel_flag=mask)
            result["grid"][f"x{expansion}_s{seed}"] = {
                "rho_novel_vis": rho, "n_novel": n_novel, "dict_size": int(e_real.shape[0]),
                "fvu": q["fvu"], "dead_latent_frac": q["dead_latent_frac"], "mean_l0": q["mean_l0"],
                "out_of_span_floor_pct95": floor95, "crosscheck_conf": cc,
                "out_of_span_mean_ci": DM.bootstrap_ci(e_real, n_boot=1000, seed=1337)}
            print(f"[W10-UNB][x{expansion} s{seed}] rho_novel_vis={rho:.5f} n_novel={n_novel} "
                  f"cc={cc['n_survive']}/{cc['n_candidates']} fvu={q['fvu']:.3f}")
            del ae
            if device == "cuda":
                torch.cuda.empty_cache()
        rs = np.array(rho_by_seed)
        result["grid"][f"x{expansion}_rho_seed_mean"] = float(rs.mean())
        result["grid"][f"x{expansion}_rho_seed_ci"] = DM.bootstrap_ci(rs, n_boot=1000, seed=1337)

    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    verdict = "preserved (ρ_novel^vis ≤ floor)" if all(
        result["grid"][f"x{w}_rho_seed_mean"] <= 1e-6 for w in widths) else "emergent directions present"
    print(f"[W10-UNB] DONE → {verdict}")


if __name__ == "__main__":
    main()
