"""N5c verdict job (pi-trajectory N5 prereg §5/§8). Reads the FROZEN cloud table from S3 and runs ONLY the
re-aggregation + the §5 dual-cut (INSIDE / OUTSIDE / INCONCLUSIVE-N). NO ΔW re-extraction, no model, no GPU,
no hub, no boto3 (aws CLI; absent from the pg-env job venv). Mirrors e2_reagg.py. Fail-loud, geometry-only.

  pg run infra.aws_pg.jobs.n5c_reagg --type r7i.2xlarge --no-weights --args "--seed 1337" --budget-min 30
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
INSTR_SHA = "693a7d04"          # frozen reuse_estimators.py sha256 (pi-e1-instrument-freeze-v2)


def _aws_file(key, dest):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dest), "--region", "us-east-1"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # read the frozen cloud extraction (members + π0) — read-only, NOT recomputed.
    src = out / "n5c_cloud.json"
    _aws_file("results/n5c_ft_cloud/out/n5c_cloud.json", src)
    d = json.loads(src.read_text())
    summ, members, pi0 = d["summary"], d["members"], d["pi0_member"]

    # provenance gates — frozen instrument, π0 bit-identical to its E2b freeze, a real cloud.
    assert summ["instrument_sha256"].startswith(INSTR_SHA), \
        f"cloud not the v2 freeze: {summ['instrument_sha256'][:12]} != {INSTR_SHA}"
    assert summ["pi0_freeze_max_abs_diff"] < 1e-9, \
        f"π0 cloud member drifted from the E2b freeze: {summ['pi0_freeze_max_abs_diff']}"
    assert len(members) >= 5, f"cloud too small for a meaningful rank: N={len(members)}"

    # cross-check: π0's N5c a_pooled_exL17mlp reproduces the published E2b reagg (pi-e2-reagg-v3) value.
    e2reagg = out / "e2_reagg.json"
    _aws_file("results/e2_reagg/out/e2_reagg.json", e2reagg)
    e2 = json.loads(e2reagg.read_text())
    for r in ("32", "64"):
        frozen = e2["A"][r]["a_pooled_exL17mlp"]
        here = pi0["A"][r]["a_pooled_exL17mlp"]
        assert abs(frozen - here) < 1e-9, \
            f"r={r} π0 a_pooled_exL17mlp mismatch vs e2_reagg: {frozen} vs {here}"

    from infra.pg import n5c_aggregate
    verdict = n5c_aggregate.decide_n5c(pi0, members, r_grid=(32, 64))
    verdict["provenance"] = {
        "cloud_extraction": "results/n5c_ft_cloud (frozen instrument 693a7d04 / git pi-n5c-prereg)",
        "instrument_sha256": summ["instrument_sha256"], "gain_routing_sha256": summ["gain_routing_sha256"],
        "base_sha": summ["base_sha"], "pg1_sha": summ["pg1_sha"], "cloud_shas": summ["cloud_shas"],
        "pi0_freeze_max_abs_diff": summ["pi0_freeze_max_abs_diff"],
        "e2_reagg_tag": "pi-e2-reagg-v3", "reagg_prereg": "pi-n5c-prereg",
        "n_dropped": len(summ.get("dropped", {})), "dropped": summ.get("dropped", {}),
        "GEOMETRY_ONLY": True, "seed": a.seed,
    }
    (out / "n5c_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("=" * 78)
    for r in ("32", "64"):
        pr = verdict["per_r"][r]
        print(f"  r={r}: π0 a_exL17mlp={pr['pi0_a_pooled_exL17mlp']:+.4f}  ft median={pr['ft_a_median']:+.4f}  "
              f"rank={pr['a_rank_fraction']:.3f}  band={[round(x,4) for x in pr['a_central80_band']]}  "
              f"inside={pr['a_inside']} outside={pr['a_outside_strict_gap']}")
    print(f"  N={verdict['N']} (floor 1/(N+1)={verdict['rank_floor']:.3f})  "
          f"ρ_L axis downgraded={verdict['rhoL_axis_downgraded']}  "
          f"survives-outlier={verdict['survives']['survives_outlier']}")
    print(f"  task-distance: near vs distant {verdict['task_distance_gradient']}")
    print("=" * 78)
    print(f"N5C_VERDICT {verdict['verdict']}")
    print(verdict["verdict_note"])


if __name__ == "__main__":
    main()
