"""RELAW Arm-1 unblind — the σ-rank-matched mechanism gate (content vs top-σ; prereg §3.2/§6).

Loads the re-entry cells for {content, top-σ, randspan} × seeds {1337,1338,1339} plus the LATMECH
deleted/random_floor anchors, computes RF_MCQ(16k) per condition, the PRIMARY discriminator
RF_content − RF_topσ (paired item cluster bootstrap, `latmech.rf_diff_contrast`), the reported
RF_content − RF_randspan, the randspan band-recovery, seed-stability, and the §6 verdict:
  H-spec iff RF_content − RF_topσ CI < 0 (seed-stable); else H-prin (content ≈ top-σ).
Pure CPU, numpy, fail-loud.

    python -m infra.aws_pg.jobs.relaw_arm1_unblind --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

B = "s3://pg-lineage-769063704696/results"
SEEDS = (1337, 1338, 1339)

# per-seed S3 paths for each condition's confirmation cell.
def paths(seed):
    g = f"{B}/latmech-grid-s{seed}/out"
    topσ = (f"{B}/latmech-topsig/out/topsig_ablated/cell_content_cap_r512_n16000_s1337_confirmation.json"
            if seed == 1337 else
            f"{B}/relaw-arm1-topsig-s{seed}/out/cell_content_cap_r512_n16000_s{seed}_confirmation.json")
    rs = (f"{B}/latmech-randspan/out/randspan_ablated/cell_content_cap_r512_n16000_s1337_confirmation.json"
          if seed == 1337 else
          f"{B}/relaw-arm1-randspan-s{seed}/out/randspan_ablated/cell_content_cap_r512_n16000_s{seed}_confirmation.json")
    return {
        "content": f"{g}/content_n16000/cell_content_cap_r512_n16000_s{seed}_confirmation.json",
        "topsig": topσ,
        "randspan": rs,
        "deleted": f"{g}/deleted_n16000/cell_deleted_cap_r512_n16000_s{seed}_confirmation.json",
        "random_floor": f"{g}/random_floor_n16000/cell_random_floor_cap_r512_n16000_s{seed}_confirmation.json",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import latmech as Lm
    from infra.pg import reprice_stats as RS

    def load(s3, dest):
        subprocess.run(["aws", "s3", "cp", s3, str(dest), "--region", "us-east-1"], check=True)
        return np.array(json.loads(pathlib.Path(dest).read_text())["mcq"]["per_item"], float)

    per_seed = {}
    for seed in SEEDS:
        P = paths(seed)
        arrs = {c: load(P[c], out / f"{c}_s{seed}.json") for c in P}
        floor, ceil = arrs["random_floor"], arrs["deleted"]
        assert all(arrs[c].shape == floor.shape for c in arrs), f"seed {seed}: per-item length mismatch"

        def rf(S):
            return RS.recovery_fraction(S, floor, ceil)
        per_seed[seed] = {
            "RF_content": rf(arrs["content"]), "RF_topsig": rf(arrs["topsig"]),
            "RF_randspan": rf(arrs["randspan"]), "RF_deleted": rf(ceil), "RF_floor": rf(floor),
            "content_minus_topsig": Lm.rf_diff_contrast(arrs["content"], arrs["topsig"], floor, ceil,
                                                        n_boot=a.n_boot, seed=seed),
            "content_minus_randspan": Lm.rf_diff_contrast(arrs["content"], arrs["randspan"], floor, ceil,
                                                          n_boot=a.n_boot, seed=seed),
        }
        d = per_seed[seed]
        print(f"[ARM1] seed {seed}: RF content={d['RF_content']:+.3f} topσ={d['RF_topsig']:+.3f} "
              f"randspan={d['RF_randspan']:+.3f} | content−topσ=[{d['content_minus_topsig']['lo']:+.3f},"
              f"{d['content_minus_topsig']['hi']:+.3f}]", flush=True)

    # pooled across seeds (concat per-item) for the headline CI.
    def cat(cond):
        xs = []
        for seed in SEEDS:
            xs.append(np.array(json.loads((out / f"{cond}_s{seed}.json").read_text())["mcq"]["per_item"], float))
        return np.concatenate(xs)
    cont, tops, rspn, fl, ce = (cat("content"), cat("topsig"), cat("randspan"),
                                cat("random_floor"), cat("deleted"))
    pooled = {
        "content_minus_topsig": Lm.rf_diff_contrast(cont, tops, fl, ce, n_boot=a.n_boot, seed=1337),
        "content_minus_randspan": Lm.rf_diff_contrast(cont, rspn, fl, ce, n_boot=a.n_boot, seed=1337),
    }
    signs_ct = [1 if per_seed[s]["content_minus_topsig"]["point"] > 0 else -1 for s in SEEDS]
    seed_stable = len(set(signs_ct)) == 1
    ct = pooled["content_minus_topsig"]
    # §6 corrected rule: H-spec iff content craters MORE than top-σ (CI<0), seed-stable; else H-prin.
    named = "H-spec" if (ct["hi"] < 0 and seed_stable) else "H-prin"

    rep = {"object": "relaw_arm1_unblind_v1", "seeds": list(SEEDS), "per_seed": per_seed,
           "pooled": pooled, "seed_signs_content_minus_topsig": signs_ct, "seed_stable": seed_stable,
           "named": named,
           "grid": {"Arm1_content_minus_topsig": [round(ct["lo"], 4), round(ct["hi"], 4)],
                    "Arm1_content_minus_randspan": [round(pooled["content_minus_randspan"]["lo"], 4),
                                                    round(pooled["content_minus_randspan"]["hi"], 4)],
                    "Arm1_RF_randspan_per_seed": [round(per_seed[s]["RF_randspan"], 3) for s in SEEDS],
                    "named_arm1_verdict": named},
           "frozen_rule": ("H-spec iff RF_content−RF_topσ CI<0 seed-stable (content craters MORE than the "
                           "same-rank pure-σ removal); else H-prin (content ≈ top-σ). randspan reported "
                           "as the band-recovery control (NOT the discriminator).")}
    (out / "relaw_arm1_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("=" * 78, flush=True)
    print(f"[ARM1] pooled content−topσ=[{ct['lo']:+.3f},{ct['hi']:+.3f}] seed-stable={seed_stable} "
          f"signs={signs_ct} | randspan RF/seed={[round(per_seed[s]['RF_randspan'],2) for s in SEEDS]}",
          flush=True)
    print(f"[ARM1] VERDICT: {named}", flush=True)
    print(f"RELAW_ARM1_OK verdict={named}", flush=True)


if __name__ == "__main__":
    main()
