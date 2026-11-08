"""W9 §9.1 PREFLIGHT GATE (fail-fast, before any Diff-SAE training).

Port-validation: re-harvest the paired before/after activations on AWS, retrain ONE crosscoder
at L20 in the frozen W4 regime, recompute its FVU, and compare to the cached Modal reference
`reference/w4_crosscoder/delta_crosscoder.json` (read from S3 at runtime — NOT transcribed,
CLAUDE.md provenance discipline). Match within tolerance → the activation pipeline survived the
Modal→AWS port, proceed. Mismatch → the harness is invalid → HALT (raise; STATUS exit=1).

    python -m infra.aws_pg.jobs.w9_preflight --out <dir> \
        --reference-s3 s3://pg-lineage-769063704696/reference/w4_crosscoder/delta_crosscoder.json
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
                    default="s3://pg-lineage-769063704696/reference/w4_crosscoder/delta_crosscoder.json")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.05, help="abs tolerance on fvu_after")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    H.hf_online_setup()                    # HF_TOKEN + online (gated models + dataset streaming)
    H.ensure_wandb()                       # dictionary_learning.training imports wandb at top

    # --- read the cached reference (download via the instance-role aws cli; fail loud) ---
    ref_local = out / "reference_delta_crosscoder.json"
    subprocess.run(["aws", "s3", "cp", args.reference_s3, str(ref_local),
                    "--region", "us-east-1"], check=True)
    ref = json.loads(ref_local.read_text())
    ref_fvu_after = float(ref["layers"][str(args.layer)]["crosscoder_fvu"]["fvu_after"])
    ref_fvu_before = float(ref["layers"][str(args.layer)]["crosscoder_fvu"]["fvu_before"])

    device = "cuda"
    print(f"[W9-PRE] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"[W9-PRE] reference L{args.layer} fvu_after={ref_fvu_after:.4f} fvu_before={ref_fvu_before:.4f}")

    pg_after, pg_before, proc, paths, img_tok, gate = H.build_before_after(device)
    print(f"[W9-PRE] swap_gate={json.dumps(gate, default=str)}")
    mm = H.build_mm_batch(proc, img_tok, device)
    acts_before, acts_after, is_img, held = H.harvest_paired_layer(
        pg_after, pg_before, args.layer, mm, device)
    N, d = acts_before.shape
    print(f"[W9-PRE][L{args.layer}] paired tokens N={N} d={d} img_frac={float(is_img.float().mean()):.3f}")

    ae, cfg = H.train_crosscoder(acts_before, acts_after, args.layer, seed=0, device=device)
    fvu = H.crosscoder_fvu(ae, acts_before, acts_after, device)
    print(f"[W9-PRE][L{args.layer}] recomputed FVU before={fvu['fvu_before']:.4f} after={fvu['fvu_after']:.4f}")

    delta_after = abs(fvu["fvu_after"] - ref_fvu_after)
    ok = delta_after <= args.tol
    result = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "layer": args.layer, "n_paired_tokens": int(N), "d_model": int(d),
        "image_fraction": float(is_img.float().mean()),
        "swap_gate": gate, "crosscoder_cfg": cfg,
        "reference_s3": args.reference_s3,
        "reference_fvu_after": ref_fvu_after, "reference_fvu_before": ref_fvu_before,
        "recomputed_fvu": fvu,
        "abs_delta_fvu_after": delta_after, "tolerance": args.tol,
        "PREFLIGHT_OK": bool(ok),
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W9-PRE] PREFLIGHT_OK={ok}  |Δfvu_after|={delta_after:.4f} (tol {args.tol})")
    if not ok:
        raise RuntimeError(
            f"PREFLIGHT HALT: recomputed FVU@L{args.layer} after={fvu['fvu_after']:.4f} "
            f"deviates from reference {ref_fvu_after:.4f} by {delta_after:.4f} > tol {args.tol}. "
            f"The Modal→AWS activation pipeline did NOT reproduce — harness invalid.")
    print("[W9-PRE] gate passed → safe to develop the Diff-SAE pipeline.")


if __name__ == "__main__":
    main()
