"""E1 — instrument-freeze job (pi-trajectory spec §E1). Zero weights, CPU, fail-loud.

Runs the 12-item synthetic calibration battery and FREEZES the reuse-estimator instrument: records
sha256(reuse_estimators.py) + sha256(calibration_e1.py) (the real on-box freeze anchors — the launcher
tars only infra/, so there is no .git to `git rev-parse` here; the frozen commit SHA is passed in via
--git-head from the local dispatch) + torch/numpy versions + master seed into calibration_e1.json. Any
edit to either module changes its sha256 and re-triggers E1. The prereg cites this JSON + the module SHAs.

  pg run infra.aws_pg.jobs.e1_calibration --type r7i.2xlarge --no-weights \
      --args "--seed 1337 --n_draws 2000 --git-head <frozen-sha>" --budget-min 30
"""
import argparse
import hashlib
import json
import pathlib


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--git-head", default="unset")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    assert a.git_head != "unset", "--git-head must be the frozen commit SHA (pass via --args)"
    import numpy
    import torch
    from infra.pg import calibration_e1, gain_routing, reuse_estimators

    result = calibration_e1.run_calibration(seed=a.seed, n_draws=a.n_draws)
    result["provenance"] = {
        "git_head": a.git_head,
        "reuse_estimators_sha256": _sha256(reuse_estimators.__file__),
        "calibration_e1_sha256": _sha256(calibration_e1.__file__),
        "gain_routing_sha256": _sha256(gain_routing.__file__),  # _svd_fp64 sign gauge dependency
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "master_seed": a.seed,
    }
    (out / "calibration_e1.json").write_text(json.dumps(result, indent=2))

    print("=" * 78)
    for name, g in result["gates"].items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {name:16s} {g['note']}")
    print("=" * 78)
    print(("E1_OK" if result["all_pass"] else "E1_FAIL"),
          json.dumps({"all_pass": result["all_pass"],
                      "reuse_estimators_sha256": result["provenance"]["reuse_estimators_sha256"][:12]}))
    assert result["all_pass"], "E1 calibration FAILED — instrument not frozen (HALT the numeric program)"


if __name__ == "__main__":
    main()
