"""W16 CONTROLS (prereg §5/§6 — develop-on-control; AFTER is SEALED, never scored).

Builds the FLOOR-side null and every comparator the unblind job reads, per the
post-critic prereg v2 (null ≡ FLOOR random-decoder PG2; BEFORE ≡ damage
reference, descriptive only):

  1. build pair; record ΔW rel-Frobenius over the probe scope (weights only,
     context never threshold); DELETE AFTER before any scoring.
  2. BEFORE + FLOOR(seed 1337) full-split `mmlu_choice_logits` (text-only,
     gemma tokenizer). Gates: item_hashes == W14 frozen hashes; per-item
     exact-match ≥ 0.98 + acc |Δ| ≤ 0.5 pt vs W14 mmlu_text BEFORE/FLOOR
     (extraction stability); denom > 0.05 + FLOOR ≤ 35% on grid bucket AND
     confirmation half (scale validity).
  3. FLOOR null grid: 14 cells (int8, int4, 4σ × 3 seeds) on the grid bucket
     (`item_filter=is_grid_subset`), snapshot/restore bitwise per cell.
  4. BEFORE descriptive grid: 6 cells (int8, int4, 4σ × seed 1337).
  5. FLOOR filtered unprobed bucket read — masking-convention consistency gate
     (filtered vs masked acc |Δ| ≤ 0.5 pt).
  6. FLOOR decoder seeds 1338/1339: bucket `mmlu_choice_logits` (Arm-A margin
     seed context).
  7. `max_null_band` over the 14 FLOOR cells (shared-index bootstrap, den≤0
     dropped+counted) → screening band + draws, FROZEN into controls.json.

    python -m infra.aws_pg.jobs.w16_controls --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

W14_CONTROLS_S3 = "s3://pg-lineage-769063704696/results/w14_arm23_controls/out/legA_controls.json"
NOISE_CELLS = [(s, seed) for s in (1e-3, 3e-3, 1e-2, 3e-2) for seed in (1337, 1338, 1339)]


def _cell_name(kind, a, b=None):
    return f"int{a}" if kind == "q" else f"noise_{a:g}_s{b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w14-controls-s3", default=W14_CONTROLS_S3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="2 items/subject, 2 cells, wiring only")
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

    assert torch.cuda.is_available(), "W16 controls need GPU."
    device = "cuda"
    n_per = 2 if args.smoke else None
    print(f"[W16-CTRL] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")

    # ---- W14 frozen cross-check arrays ----
    w14_local = out / "w14_controls_cached.json"
    subprocess.run(["aws", "s3", "cp", args.w14_controls_s3, str(w14_local), "--region", "us-east-1"],
                   check=True)
    w14 = json.loads(w14_local.read_text())
    w14_hashes = w14["item_hashes"]
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- build pair; ΔW context from weights; SEAL AFTER ----
    pg_after, pg_before, _proc, paths, _it, build_gate = H.build_before_after(device)
    lm_after = pg_after.get_submodule(paths["language_model"])
    lm_before = pg_before.get_submodule(paths["language_model"])
    scope = W.probe_param_names(lm_before)
    rels = []
    with torch.no_grad():
        for n in scope:
            a = lm_after.get_parameter(n).detach().float()
            b = lm_before.get_parameter(n).detach().float()
            rels.append(float((a - b).norm() / b.norm().clamp_min(1e-12)))
    dw_ctx = {"rel_frob_mean": sum(rels) / len(rels), "rel_frob_max": max(rels),
              "n_tensors": len(scope), "note": "context only, never a threshold (prereg §1)"}
    del lm_after, pg_after
    torch.cuda.empty_cache()
    print(f"[W16-CTRL] ΔW rel-Frob mean={dw_ctx['rel_frob_mean']:.4f} max={dw_ctx['rel_frob_max']:.4f} "
          f"— AFTER sealed (weights compared, never scored)")

    # ---- BEFORE full-split logits ----
    before_cl = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=n_per)
    hashes = before_cl["per_item_hash"]
    print(f"[W16-CTRL] BEFORE acc={before_cl['acc_pct']:.2f} n={before_cl['n']}")

    # ---- FLOOR (binding seed 1337) full-split logits ----
    pg_floor = H.build_random_after(pg_before, paths, device, seed=1337)
    lm_floor = pg_floor.get_submodule(paths["language_model"])
    floor_cl = TE.mmlu_choice_logits(pg_floor, tok, device, n_per_subject=n_per)
    print(f"[W16-CTRL] FLOOR acc={floor_cl['acc_pct']:.2f}")

    # ---- identity + extraction-stability gates ----
    assert floor_cl["per_item_hash"] == hashes, "BEFORE/FLOOR item order differs — HALT"
    assert floor_cl["correct_idx"] == before_cl["correct_idx"], "correct_idx differs — HALT"
    if not args.smoke:
        assert hashes == w14_hashes, "item_hashes differ from W14 frozen controls — HALT"
        gates_ext = {}
        for name, cl, w14key in (("BEFORE", before_cl, "BEFORE"), ("FLOOR", floor_cl, "FLOOR")):
            ref_pi = w14["mmlu_text"][w14key]["per_item"]
            match = float(np.mean(np.array(cl["per_item"]) == np.array(ref_pi)))
            d_acc = abs(cl["acc_pct"] - float(w14["mmlu_text"][w14key]["acc_pct"]))
            ok = bool(match >= 0.98 and d_acc <= 0.5)
            gates_ext[name] = {"exact_match": match, "abs_acc_delta_pct": d_acc, "ok": ok}
            print(f"[W16-CTRL] extraction {name}: match={match:.4f} |Δacc|={d_acc:.2f} ok={ok}")
        assert all(g["ok"] for g in gates_ext.values()), f"extraction stability HALT: {gates_ext}"
    else:
        gates_ext = {"smoke": True}

    # ---- scale validity on both decision item sets ----
    b_pi, f_pi = before_cl["per_item"], floor_cl["per_item"]
    scale = {}
    for set_name, pred in (("grid_bucket", W.is_grid_subset), ("confirmation", W.is_confirmation)):
        bm = W.mask_by(b_pi, hashes, pred)
        fm = W.mask_by(f_pi, hashes, pred)
        denom = float(np.mean(bm) - np.mean(fm))
        floor_pct = float(np.mean(fm) * 100)
        ok = bool(denom > 0.05 and floor_pct <= 35.0)
        scale[set_name] = {"n": len(bm), "denom": denom, "floor_pct": floor_pct, "ok": ok}
        print(f"[W16-CTRL] scale {set_name}: n={len(bm)} denom={denom:.4f} floor={floor_pct:.1f}% ok={ok}")
    if not args.smoke:
        assert all(s["ok"] for s in scale.values()), f"scale validity HALT: {scale}"

    # ---- FLOOR null grid (14 cells) + BEFORE descriptive grid (6 cells) ----
    cells_q = [8, 4] if not args.smoke else [8]
    cells_n = NOISE_CELLS if not args.smoke else [(1e-3, 1337)]

    def run_grid(model, lm, side, quant_bits, noise_cells):
        snap = W.snapshot_probe_params(lm)
        names = W.probe_param_names(lm)
        cells = {}
        for bits in quant_bits:
            rep = W.quantize_decoder_(lm, bits)
            ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per,
                                       item_filter=W.is_grid_subset)
            W.restore_probe_params_(lm, snap)
            for nm in names:
                assert torch.equal(lm.get_parameter(nm).detach().cpu(), snap[nm]), \
                    f"{side} int{bits}: restore not bitwise on {nm}"
            cells[_cell_name("q", bits)] = {"per_item": ev["per_item"],
                                            "per_item_hash": ev["per_item_hash"],
                                            "acc_pct": ev["acc_pct"], "rel_frob_max": rep["rel_frob_max"]}
            print(f"[W16-CTRL] {side} int{bits}: acc={ev['acc_pct']:.2f}")
        for sigma, seed in noise_cells:
            rep = W.noise_decoder_(lm, sigma, seed=seed)
            ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per,
                                       item_filter=W.is_grid_subset)
            W.restore_probe_params_(lm, snap)
            for nm in names:
                assert torch.equal(lm.get_parameter(nm).detach().cpu(), snap[nm]), \
                    f"{side} σ={sigma} s{seed}: restore not bitwise on {nm}"
            cells[_cell_name("n", sigma, seed)] = {"per_item": ev["per_item"],
                                                   "per_item_hash": ev["per_item_hash"],
                                                   "acc_pct": ev["acc_pct"],
                                                   "rel_frob_max": rep["rel_frob_max"]}
            print(f"[W16-CTRL] {side} σ={sigma:g} s{seed}: acc={ev['acc_pct']:.2f}")
        return cells

    floor_cells = run_grid(pg_floor, lm_floor, "FLOOR", cells_q, cells_n)

    # ---- masking-convention consistency: filtered unprobed FLOOR vs masked full-split ----
    floor_filt = TE.mmlu_choice_logits(pg_floor, tok, device, n_per_subject=n_per,
                                       item_filter=W.is_grid_subset)
    floor_masked_pi = W.mask_by(f_pi, hashes, W.is_grid_subset)
    mask_gate = {
        "filtered_acc_pct": floor_filt["acc_pct"],
        "masked_acc_pct": float(np.mean(floor_masked_pi) * 100),
        "hash_match": floor_filt["per_item_hash"] == W.mask_by(hashes, hashes, W.is_grid_subset),
    }
    mask_gate["abs_delta_pct"] = abs(mask_gate["filtered_acc_pct"] - mask_gate["masked_acc_pct"])
    mask_gate["ok"] = bool(mask_gate["hash_match"] and mask_gate["abs_delta_pct"] <= 0.5)
    print(f"[W16-CTRL] masking gate: filtered={mask_gate['filtered_acc_pct']:.2f} "
          f"masked={mask_gate['masked_acc_pct']:.2f} ok={mask_gate['ok']}")
    if not args.smoke:
        assert mask_gate["ok"], f"masking-convention HALT: {mask_gate}"

    # ---- FLOOR decoder seeds 1338/1339 — Arm-A margin seed context (bucket) ----
    floor_seed_ctx = {}
    del pg_floor, lm_floor
    torch.cuda.empty_cache()
    for fs in ([1338, 1339] if not args.smoke else [1338]):
        pg_fs = H.build_random_after(pg_before, paths, device, seed=fs)
        ev = TE.mmlu_choice_logits(pg_fs, tok, device, n_per_subject=n_per,
                                   item_filter=W.is_grid_subset)
        m = W.margin_per_item(ev["choice_logits"], ev["correct_idx"])
        floor_seed_ctx[fs] = {"acc_pct": ev["acc_pct"], "margin_mean": float(m.mean()),
                              "rank_auc": W.rank_auc(ev["choice_logits"], ev["correct_idx"])}
        print(f"[W16-CTRL] FLOOR seed {fs}: acc={ev['acc_pct']:.2f} "
              f"margin={floor_seed_ctx[fs]['margin_mean']:.3f} auc={floor_seed_ctx[fs]['rank_auc']:.3f}")
        del pg_fs
        torch.cuda.empty_cache()

    before_cells = run_grid(pg_before, pg_before.get_submodule(paths["language_model"]),
                            "BEFORE", cells_q,
                            [(s, 1337) for s, seed in NOISE_CELLS if seed == 1337] if not args.smoke
                            else [(1e-3, 1337)])

    # ---- screening null band (FROZEN) ----
    bucket_hashes = W.mask_by(hashes, hashes, W.is_grid_subset)
    f_unp_bucket = W.mask_by(f_pi, hashes, W.is_grid_subset)
    b_bucket = W.mask_by(b_pi, hashes, W.is_grid_subset)
    for c, d in floor_cells.items():
        assert d["per_item_hash"] == bucket_hashes, f"FLOOR cell {c} bucket order mismatch"
    # FLOOR side: the unprobed array IS the floor scale array (same model, same mask)
    band = W.max_null_band({c: d["per_item"] for c, d in floor_cells.items()},
                           f_unp_bucket, b_bucket, f_unp_bucket,
                           n_boot=args.n_boot, seed=1337)
    print(f"[W16-CTRL] screening band: hi={band['band_hi']:.4f} width={band['band_width']:.4f} "
          f"kept={band['n_draws_kept']} dropped={band['n_draws_dropped']}")

    controls_ok = bool((args.smoke or (all(g.get("ok", True) for g in gates_ext.values())
                                       and all(s["ok"] for s in scale.values())
                                       and mask_gate["ok"])))
    result = {
        "torch": torch.__version__, "smoke": args.smoke,
        "build_gate": build_gate, "dw_context": dw_ctx,
        "item_hashes": hashes,
        "correct_idx": before_cl["correct_idx"],
        "BEFORE": {"acc_pct": before_cl["acc_pct"], "per_item": b_pi,
                   "choice_logits": before_cl["choice_logits"]},
        "FLOOR": {"acc_pct": floor_cl["acc_pct"], "per_item": f_pi,
                  "choice_logits": floor_cl["choice_logits"], "decoder_seed": 1337},
        "gates_extraction": gates_ext, "scale_validity": scale, "masking_gate": mask_gate,
        "floor_cells_bucket": floor_cells, "before_cells_bucket": before_cells,
        "floor_seed_context": floor_seed_ctx,
        "screening_null_band": band,
        "n_boot": args.n_boot,
        "CONTROLS_OK": controls_ok,
    }
    (out / "w16_controls.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[W16-CTRL] CONTROLS_OK={controls_ok}")
    if not controls_ok:
        raise RuntimeError(f"W16 CONTROLS HALT — see {out / 'w16_controls.json'}")


if __name__ == "__main__":
    main()
