"""W10 §9 STEP-1 PREFLIGHT GATE (fail-fast, before any vision-instrument training).

Three sub-gates (prereg §9 step 1), all CPU — weights + state_dicts only, NO forward pass:

  (1a) SigLIP-init match — confirm the public SigLIP-So400m-patch14-224 is the ACTUAL init of
       PG2-3b-pt-224's vision_tower: shared keys present, ZERO shape mismatch, matching
       position-embedding shape. Record the SigLIP + PG repo SHAs. Mismatch → HALT (the vision
       "before" would be wrong → ρ_novel^vis + Λ would measure the wrong thing). Also closes the
       base-init confound on the vision side.
  (1b) Pipeline-port check — recompute the W1 vision ΔW (rel-Frobenius of V1=PG2-vision vs
       V0=SigLIP) on AWS and match the cached Modal reference rel_frob_mean, READ FROM S3 (not
       transcribed — CLAUDE.md provenance), within tol. Mismatch → HALT (model-load/port invalid).
  (1c) Split assertion — DM.enforce_split returns a non-empty, disjoint held set on a dummy mask
       (the §9-1c firewall utility that kills the "held captured but never used" leak).

    python -m infra.aws_pg.jobs.w10_preflight --out <dir> \
        --reference-s3 s3://pg-lineage-769063704696/reference/w1_real_dw/w1_real_dw_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-s3",
                    default="s3://pg-lineage-769063704696/reference/w1_real_dw/w1_real_dw_summary.json")
    ap.add_argument("--tol", type=float, default=1e-3, help="abs tolerance on vision rel_frob_mean")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # MUST run before any huggingface_hub/transformers import: hf caches HF_HUB_OFFLINE into a
    # module constant AT IMPORT, so run_job.sh's HF_HUB_OFFLINE=1 sticks if hf is imported first
    # (→ model_info raises OfflineModeIsEnabled). diff_harvest's top is light (no hf), safe to import.
    from infra.pg import diff_harvest as H
    H.hf_online_setup()                    # HF_TOKEN + online mode (gated PG2 / SigLIP)

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, SiglipVisionModel

    from infra.pg import diff_metrics as DM
    from infra.pg.contrast_a1 import V0_SIGLIP_REPO, probe_swap_structure
    from infra.pg.entities import ENTITIES, resolve_submodule_paths
    from infra.pg.weight_diff import compute_weight_diff

    # --- read cached W1 reference (instance-role aws cli; fail loud, no transcription) ---
    ref_local = out / "reference_w1_real_dw_summary.json"
    subprocess.run(["aws", "s3", "cp", args.reference_s3, str(ref_local),
                    "--region", "us-east-1"], check=True)
    ref = json.loads(ref_local.read_text())
    ref_rel_frob_mean = float(ref["vision"]["summary"]["rel_frob_mean"])
    print(f"[W10-PRE] reference vision rel_frob_mean={ref_rel_frob_mean:.6f}")

    # =====================================================================
    # (1a) SigLIP-init match — structural probe + recorded SHAs.
    # =====================================================================
    api = HfApi()
    siglip_sha = api.model_info(V0_SIGLIP_REPO).sha
    pg_sha = api.model_info(ENTITIES["PG"].repo_id).sha
    probe = probe_swap_structure()                       # loads PG2 + SigLIP (+ Gemma2) on CPU
    vt = probe["vision_tower"]
    pe_pg, pe_sig = vt["pg_pos_embed"], vt["siglip_pos_embed"]
    pos_ok = bool(pe_pg and pe_sig and pe_pg["shape"] == pe_sig["shape"])
    siglip_match = bool(vt["n_shared"] > 0 and vt["n_shape_mismatch"] == 0 and pos_ok)
    print(f"[W10-PRE][1a] siglip={V0_SIGLIP_REPO}@{siglip_sha} pg={ENTITIES['PG'].repo_id}@{pg_sha}")
    print(f"[W10-PRE][1a] vt shared={vt['n_shared']} shape_mismatch={vt['n_shape_mismatch']} "
          f"pos_embed pg={pe_pg} siglip={pe_sig} → match={siglip_match}")

    # =====================================================================
    # (1b) Pipeline-port: recompute vision ΔW (V1=PG2-vision vs V0=SigLIP), match the S3 reference.
    # =====================================================================
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa")
    paths = resolve_submodule_paths(pg)
    pg_vt_sd = {k: v.clone() for k, v in pg.get_submodule(paths["vision_tower"]).state_dict().items()}
    del pg
    siglip = SiglipVisionModel.from_pretrained(V0_SIGLIP_REPO, torch_dtype=torch.bfloat16, device_map="cpu")
    v0_sd = {k: v.clone() for k, v in siglip.state_dict().items()}
    del siglip
    pg_vt_keys = set(pg_vt_sd)
    pg_only = pg_vt_keys - set(v0_sd)                    # PG keys must be ⊆ SigLIP keys
    assert not pg_only, f"Unexpected PG-only vision keys (architecture regression): {sorted(pg_only)}"
    v0_sd_matched = {k: v0_sd[k] for k in pg_vt_keys}   # intersection (drops SigLIP head.* PG stripped)
    vision_diff = compute_weight_diff(v0_sd_matched, pg_vt_sd, assert_zero=False)
    rel_frob_mean = float(vision_diff["summary"]["rel_frob_mean"])
    delta_dw = abs(rel_frob_mean - ref_rel_frob_mean)
    dw_ok = bool(delta_dw <= args.tol)
    print(f"[W10-PRE][1b] recomputed vision rel_frob_mean={rel_frob_mean:.6f} "
          f"|Δ|={delta_dw:.2e} (tol {args.tol}) → ok={dw_ok}")

    # =====================================================================
    # (1c) Split assertion — the §9-1c firewall utility on a dummy mask.
    # =====================================================================
    rng = np.random.default_rng(1337)
    dummy_held = rng.random(1000) < 0.25
    dummy_text = rng.random(1000) < 0.5
    split = DM.enforce_split(dummy_held, name="preflight", text=dummy_text)
    split_ok = bool(split["n_conf"] > 0 and split["n_disc"] > 0
                    and split["n_conf"] + split["n_disc"] == split["n"])
    print(f"[W10-PRE][1c] enforce_split disc={split['n_disc']} conf={split['n_conf']} "
          f"conf_text={split['conf_text_idx'].size} → ok={split_ok}")

    ok = bool(siglip_match and dw_ok and split_ok)
    result = {
        "torch": torch.__version__,
        "reference_s3": args.reference_s3,
        "gate_1a_siglip_match": {
            "siglip_repo": V0_SIGLIP_REPO, "siglip_sha": siglip_sha,
            "pg_repo": ENTITIES["PG"].repo_id, "pg_sha": pg_sha,
            "vision_tower": vt, "match": siglip_match},
        "gate_1b_dw_port": {
            "reference_rel_frob_mean": ref_rel_frob_mean,
            "recomputed_rel_frob_mean": rel_frob_mean,
            "abs_delta": delta_dw, "tolerance": args.tol,
            "eff_rank_mean": float(vision_diff["summary"]["eff_rank_mean"]),
            "ok": dw_ok},
        "gate_1c_split": {"n": int(split["n"]), "n_disc": int(split["n_disc"]),
                          "n_conf": int(split["n_conf"]), "ok": split_ok},
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W10-PRE] PREFLIGHT_OK={ok}  (1a={siglip_match} 1b={dw_ok} 1c={split_ok})")
    if not ok:
        raise RuntimeError(
            f"W10 PREFLIGHT HALT: 1a_siglip_match={siglip_match} 1b_dw_port={dw_ok} "
            f"1c_split={split_ok}. See {out / 'preflight.json'}.")
    print("[W10-PRE] gate passed → safe to develop the vision-instrument pipeline.")


if __name__ == "__main__":
    main()
