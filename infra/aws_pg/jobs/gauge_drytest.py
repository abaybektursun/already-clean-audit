"""GAUGE weight-free dry-run gate (§12.1) — run the 7 pure pytest suites on a CPU box.

This is the FIRST real gate in the §12 protocol: prove every probe's pure logic (concentration +
Haar null, symmetry-catalogue function-equivalence + aligner, MP fit + edge buffer + permutation
ESD, CCC + triple + null cloud, the 4 metric kernels through law_verdict, bisection + premium on a
toy quadratic, and the shared spine) BEFORE any weight download or GPU minute. No model, no weights
(`--no-weights`), CPU only. Fail-loud: a single failing test → non-zero exit → STATUS exit=1.

  pg run infra.aws_pg.jobs.gauge_drytest --type r7i.4xlarge --no-weights \
      --s3-prefix results/gauge_drytest --budget-min 30
"""
import argparse
import json
import pathlib
import subprocess
import sys

SUITES = [
    "infra/pg/tests/test_gauge_common.py",
    "infra/pg/tests/test_gauge_cs.py",
    "infra/pg/tests/test_gauge_fix.py",
    "infra/pg/tests/test_gauge_rmt.py",
    "infra/pg/tests/test_gauge_tree.py",
    "infra/pg/tests/test_gauge_fun.py",
    "infra/pg/tests/test_gauge_dist.py",
]


def _ensure(mod, pip_name=None):
    """Idempotent import-or-install for a control-plane test dep (pytest/scipy) not guaranteed on the
    env AMI — mirrors the wandb shim pattern (skill gotcha 12). Fail-loud if the install itself fails."""
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)
        __import__(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    _ensure("pytest")
    _ensure("scipy")           # gauge_fix aligner (linear_sum_assignment) + its test
    import numpy, scipy        # noqa: F401  (report versions; fail loud if absent)

    # run each suite separately so a crash localises to one probe (not one opaque pytest exit).
    results = {}
    all_ok = True
    for suite in SUITES:
        assert pathlib.Path(suite).exists(), f"missing suite {suite} (code package incomplete)"
        proc = subprocess.run([sys.executable, "-m", "pytest", suite, "-q", "--no-header",
                               "--tb=short", "-p", "no:cacheprovider"], capture_output=True, text=True)
        full = proc.stdout + proc.stderr
        # full log per suite so a truncated 8-line tail never hides a traceback (the fix-suite lesson).
        (out / (pathlib.Path(suite).stem + ".log")).write_text(full)
        tail = "\n".join(full.splitlines()[-8:])
        ok = proc.returncode == 0
        results[suite] = {"returncode": proc.returncode, "ok": ok, "tail": tail}
        all_ok = all_ok and ok
        print(f"[gauge-drytest] {suite}: {'PASS' if ok else 'FAIL'}\n{tail}", flush=True)

    rep = {"object": "gauge_drytest", "all_ok": all_ok, "n_suites": len(SUITES),
           "python": sys.version.split()[0], "numpy": numpy.__version__,
           "scipy": scipy.__version__, "per_suite": results}
    (out / "drytest.json").write_text(json.dumps(rep, indent=2))
    print("=" * 70)
    print(f"GAUGE_DRYTEST_{'OK' if all_ok else 'FAIL'} "
          f"{sum(r['ok'] for r in results.values())}/{len(SUITES)} suites pass")
    assert all_ok, "GAUGE dry-run gate FAILED — see per_suite tails in drytest.json"


if __name__ == "__main__":
    main()
