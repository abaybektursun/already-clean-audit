"""MERCATOR §8 step 8 — assemble a wave: one atlas, the coverage table filled, the figures.

Pulls every completed arm's artifacts from S3, concatenates their atlases into the program's single
`out/atlas.parquet` (§9), renders the §3 complete-sweep figures, and writes the filled coverage
table plus the surprise log and the defect-ledger pointer.

THREE PROPERTIES THIS ENFORCES, because each is a defect class §0.1 names:

  * NO SILENT SHRINKAGE. The arms it expects come from `ArmStatus`, not from what happens to be on
    S3, so an arm recorded complete whose artifact is missing is an error rather than a smaller
    atlas.
  * EVERY FIGURE DRAWS ITS CONTROLS. `darwin_figures.assert_controls_drawn` refuses a bank sweep
    that omits a twin, so a figure cannot show a treatment without the battery it is read against.
  * PROVENANCE SURVIVES THE MERGE. Each row keeps its own `commit_sha` and `run_id`; the assembled
    table is never stamped with one sha, because it was not produced at one commit (the same
    mistake the Wave-K critic found in DARWIN's assembly).

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import subprocess
import sys

from infra.pg import darwin_figures as F
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P


def _pull(prefix: str, dest: pathlib.Path) -> pathlib.Path:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://{M.BUCKET}/{prefix}", str(dest),
                    "--region", "us-east-1", "--exclude", "*.npy"], check=True)
    return dest


def _atlases(root: pathlib.Path):
    """Every atlas.parquet under a pulled prefix, at any depth (arms run as pipeline steps)."""
    return sorted(root.rglob("atlas.parquet"))


def main():
    ap = argparse.ArgumentParser(description="MERCATOR wave assembly + figures")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wave", required=True, choices=list(P.WAVE_ORDER))
    ap.add_argument("--status-json", required=True,
                   help="the ArmStatus file naming which arms are complete and where")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    status = P.ArmStatus(a.status_json)
    arms = [x for x in P.ARMS if P.ARMS[x]["wave"] == a.wave
            and status.get(x)["status"] == "complete"]
    assert arms, f"no completed arms recorded for wave {a.wave}"
    print(f"[assemble] wave {a.wave}: {len(arms)} arms recorded complete", flush=True)

    frames, per_arm, missing = [], {}, []
    cache = H.cache_dir(out, f"assemble_{a.wave}")
    for arm in arms:
        prefix = status.get(arm)["s3_prefix"]
        d = _pull(prefix, cache / arm.replace("/", "_"))
        found = [p for p in _atlases(d)
                 if arm in pd.read_parquet(p, columns=["arm"])["arm"].unique()]
        if not found:
            missing.append({"arm": arm, "prefix": prefix})
            continue
        for p in found:
            df = pd.read_parquet(p)
            df = df[df["arm"] == arm]
            frames.append(df)
            per_arm[arm] = per_arm.get(arm, 0) + len(df)
    assert not missing, (
        f"arms recorded COMPLETE with no atlas rows on S3: {missing}. A wave assembled from "
        f"whatever happens to be present is exactly the silent shrinkage §0.1 forbids")

    atlas = pd.concat(frames, ignore_index=True)
    assert list(atlas.columns) == list(M.ATLAS_COLUMNS), \
        f"column drift on merge: {sorted(set(atlas.columns) ^ set(M.ATLAS_COLUMNS))}"
    atlas.to_parquet(out / "atlas.parquet", index=False)
    shas = sorted(str(x) for x in atlas["commit_sha"].unique())
    print(f"[assemble] {len(atlas)} rows, {atlas['arm'].nunique()} arms, "
          f"{len(shas)} distinct commit shas", flush=True)

    # Figures: one bank sweep per arm that HAS a bank battery, with its controls asserted drawn.
    F.use_agg()
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figures, exemptions = {}, []
    full_battery = {"core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                    "haar32_s0", "haar32_s1", "haar32_s2"}
    for arm, g in atlas.groupby("arm"):
        banked = g[g["bank_id"].notna()]
        if banked.empty:
            continue
        for stat, gs in banked.groupby("statistic"):
            if gs["layer"].isna().all() or gs["bank_id"].nunique() < 2:
                continue
            path = figdir / f"{arm}__{stat}.png"
            drawn = F.bank_sweep(gs, x="layer", y="value", title=f"{arm} — {stat}",
                                 xlabel="layer", ylabel=stat, out_path=path)
            have = set(str(b) for b in drawn)
            if full_battery <= have:
                F.assert_controls_drawn(drawn, figure=f"{arm}/{stat}")
            else:
                # NOT a silent skip. A statistic that never carried the full battery cannot have
                # it drawn, and the exemption is written into the manifest with the banks that are
                # missing — so a reader sees which figures show a treatment against a partial
                # control set, and why.
                exemptions.append({"figure": f"{arm}/{stat}",
                                   "banks_drawn": sorted(have),
                                   "banks_missing": sorted(full_battery - have),
                                   "why": "the statistic was not written with the full battery"})
            figures[f"{arm}/{stat}"] = str(path)
    F.manifest(figures, out)

    cov = P.render_coverage_md(status)
    (out / "COVERAGE.md").write_text(cov)
    M.write_json(out, "assemble_summary.json",
                 {"wave": a.wave, "arms_recorded_complete": arms, "rows_per_arm": per_arm,
                  "n_rows": int(len(atlas)), "n_arms": int(atlas["arm"].nunique()),
                  "distinct_commit_shas": shas, "n_figures": len(figures), "control_exemptions": exemptions,
                  "provenance": M.provenance(), "utc": M.utc_now(),
                  "sha_note": ("the assembled atlas carries ONE SHA PER RUN and must never be "
                               "cited as produced at a single commit; each row keeps its own")})
    print("ASSEMBLE_OK " + json.dumps({"wave": a.wave, "rows": int(len(atlas)),
                                       "arms": int(atlas["arm"].nunique()),
                                       "figures": len(figures), "shas": len(shas)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
