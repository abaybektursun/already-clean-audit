"""E2b re-aggregation job (pi-trajectory follow-up). Reads the FROZEN per-tensor table from S3 and re-does
ONLY the aggregation (Deliverable A decisive L17 check + Deliverable B by_type_a CIs). NO ΔW re-extraction,
no model, no GPU, no hub, no boto3 (aws CLI for S3; absent from the pg-env job venv). Fail-loud.

  pg run infra.aws_pg.jobs.e2_reagg --type r7i.2xlarge --no-weights --args "--seed 1337" --budget-min 30
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
V2_SHA_PREFIX = "693a7d04"          # frozen reuse_estimators.py sha256 (pi-e1-instrument-freeze-v2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # fetch the frozen extraction output (the per_tensor table) — read-only, NOT recomputed.
    src = out / "e2_geometry.json"
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/results/e2_lm_geometry/out/e2_geometry.json",
                    str(src), "--region", "us-east-1"], check=True)
    d = json.loads(src.read_text())
    summ, per_tensor = d["summary"], d["per_tensor"]

    # discipline gate 2: assert the frozen v2 extraction provenance (no silent re-extraction).
    assert summ["instrument_sha256"].startswith(V2_SHA_PREFIX), \
        f"source not the v2 freeze: {summ['instrument_sha256'][:12]} != {V2_SHA_PREFIX}"
    assert len(per_tensor) == 126, f"expected 126 frozen per-tensor rows, got {len(per_tensor)}"

    from infra.pg import e2_aggregate
    reagg = e2_aggregate.reaggregate(per_tensor, r_grid=(32, 64), seed=a.seed)

    # cross-check: re-aggregated full pooled a reproduces the frozen summary value (bit-identical inputs).
    for r in ("32", "64"):
        frozen = summ["aggregation"][r]["pooled_energy_weighted_a"]
        recomputed = reagg["A"][r]["a_pooled_full"]
        assert abs(frozen - recomputed) < 1e-9, \
            f"r={r} full a_pooled mismatch: frozen {frozen} vs reagg {recomputed} — NOT bit-identical inputs"

    reagg["provenance"] = {
        "extraction_freeze": "pi-e1-instrument-freeze-v2 / git 0ca8dc16",
        "instrument_sha256": summ["instrument_sha256"],
        "reagg_tag": "pi-e2-reagg-v3",
        "reagg_prereg": "pi-e2-reagg-prereg",
        "GEOMETRY_ONLY": True,
        "seed": a.seed,
    }
    (out / "e2_reagg.json").write_text(json.dumps(reagg, indent=2))

    v = reagg["verdict"]
    print("=" * 78)
    for r in ("32", "64"):
        A = reagg["A"][r]
        print(f"  r={r}: a_pooled_full={A['a_pooled_full']:+.4f}  a_pooled_exL17mlp={A['a_pooled_exL17mlp']:+.4f}  "
              f"a_ub_exL17mlp={A['a_ub_exL17mlp']:+.4f}  survives={A['survives_exL17mlp']}")
    print(f"  MLP-a>0 in {v['n_early_layers_pos']}/17 of layers 0..16 (rule ≥13)")
    print("=" * 78)
    print(f"E2_REAGG_VERDICT {v['bin']}", json.dumps({"surv32": v["surv32"], "surv64": v["surv64"],
                                                      "early_pos": v["early_pos_rule_13of17"]}))


if __name__ == "__main__":
    main()
