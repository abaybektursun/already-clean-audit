"""VLAT §9.4 — Phase 3 MECHANISM (weight-space corroboration; DESCRIPTIVE/exploratory). CPU, fail-loud.

Per VLA hop, characterise the backbone ΔW against the dimension-counting matched-random null:
  ΔW(PG1→π0)   and   ΔW(PG1→π0.5)
paired by decoder state_dict key (matched_decoder_weight_matrices), then per matrix:
  * f_perp        — gain_routing.decompose out-of-span fraction of the RAW ΔW (the W12-sense object).
                    NB: distinct from PG2's W15 install-gradient cotangent f_perp=0.81 — NEVER compared
                    numerically (different object) and PG2 is label-only anyway (firewall).
  * rel-Frob      — pi0_port._rel_frob = ‖ΔW‖_F / ‖W_PG1‖_F (the insulation-signature magnitude).
  * eff-rank      — weight_diff._participation_ratio of ΔW's fp64 singular values.
  * null          — gain_routing.null_band(shape): observed f_perp vs the matched-random p95/p99 band.

Expectation per the boundary theory: a rewritten backbone (π0, naive joint FT) → large rel-Frob, high
out-of-span f_perp above null; a shielded/co-trained backbone (π0.5) → smaller rel-Frob / lower f_perp.
Within-Gemma-1 numerics only (all 18L) — firewall-legal to compare π0 vs π0.5 to each other.

  pg run infra.aws_pg.jobs.vlat_mechanism --args "--pi05-sealed-s3 s3://.../pi05_backbone_lm_sd.safetensors" \
      --type r7i.4xlarge --no-weights --budget-min 90
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"


def _hop(before_dec, after_dec, n_draws, seed):
    import numpy as np
    import torch

    from infra.pg import gain_routing as GR
    from infra.pg import pi0_port
    from infra.pg.gain_routing import fine_type
    from infra.pg.weight_diff import _participation_ratio

    pairs = GR.matched_decoder_weight_matrices(after_dec, before_dec)   # (pg=AFTER, l0=BEFORE)→(wb,wa)
    assert len(pairs) == 126, f"expected 126 paired decoder matrices (18×7), got {len(pairs)}"
    per_matrix = {}
    for k, (wb, wa) in pairs.items():
        d = GR.decompose(wb, wa, check_recon=True)
        dW = (wa.to(torch.float64) - wb.to(torch.float64))
        _, s, _, _, _ = GR._svd_fp64(dW)
        shape = tuple(d["shape"])
        nb = GR.null_band(shape, n_draws=n_draws, seed=seed, top_k=min(8, min(shape)))
        per_matrix[k] = {
            "type": fine_type(k), "shape": list(shape),
            "f_perp": float(d["f_perp"]), "rel_frob": float(pi0_port._rel_frob(wb, wa)),
            "eff_rank": float(_participation_ratio(s)), "rank_r": int(d["rank_r"]),
            "null_f_perp_p95": float(nb["f_perp"]["p95.0"]), "null_f_perp_p99": float(nb["f_perp"]["p99.0"]),
            "analytic_null_f_perp": float(nb["analytic_f_perp"]),
            "f_perp_above_null_p95": bool(d["f_perp"] > nb["f_perp"]["p95.0"])}

    def med(field):
        return float(np.median([m[field] for m in per_matrix.values()]))
    by_type = {}
    for t in pi0_port.PROJ:
        vals = [m for m in per_matrix.values() if m["type"] == t]
        by_type[t] = {"median_f_perp": float(np.median([m["f_perp"] for m in vals])),
                      "median_rel_frob": float(np.median([m["rel_frob"] for m in vals])),
                      "median_eff_rank": float(np.median([m["eff_rank"] for m in vals]))}
    return {
        "n_matrices": len(per_matrix),
        "median_f_perp": med("f_perp"), "median_rel_frob": med("rel_frob"), "median_eff_rank": med("eff_rank"),
        "frac_above_null_p95": float(np.mean([m["f_perp_above_null_p95"] for m in per_matrix.values()])),
        "by_type": by_type, "per_matrix": per_matrix}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=200, help="null_band draws per matrix shape")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pi05-sealed-s3", default="", help="optional s3:// sealed pi05 decoder")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder

    pi05_local = ""
    if a.pi05_sealed_s3:
        pi05_local = str(out / "pi05_backbone_lm_sd.safetensors")
        subprocess.run(["aws", "s3", "cp", a.pi05_sealed_s3, pi05_local, "--region", "us-east-1"], check=True)

    pg1, sha_pg1 = _extract_decoder("PG1", "cpu", "")
    pi0, sha_pi0 = _extract_decoder("pi0", "cpu", "")
    pi05, sha_pi05 = _extract_decoder("pi05", "cpu", pi05_local)
    print(f"[VLAT-MECH] extracted PG1({len(pg1)}) pi0({len(pi0)}) pi05({len(pi05)})")

    hops = {
        "PG1__pi0": _hop(pg1, pi0, a.n_draws, a.seed),
        "PG1__pi05": _hop(pg1, pi05, a.n_draws, a.seed)}
    for name, h in hops.items():
        print(f"[VLAT-MECH] {name}: med f_perp={h['median_f_perp']:.4f} "
              f"med rel_frob={h['median_rel_frob']:.4f} med eff_rank={h['median_eff_rank']:.1f} "
              f"above_null_p95={h['frac_above_null_p95']:.2f}")

    payload = {"shas": {"PG1": sha_pg1, "pi0": sha_pi0, "pi05": sha_pi05},
               "n_draws": a.n_draws, "seed": a.seed,
               "f_perp_object": "raw-ΔW decompose (W12 sense); NOT the PG2 W15 cotangent f_perp",
               "hops": hops}
    (out / "mechanism.json").write_text(json.dumps(payload, indent=2, default=str))
    print("[VLAT-MECH] MECHANISM_OK", json.dumps(
        {n: {"f_perp": round(h["median_f_perp"], 4), "rel_frob": round(h["median_rel_frob"], 4)}
         for n, h in hops.items()}))


if __name__ == "__main__":
    main()
