"""REGRAIN weight-free dry-run job (CPU, --no-weights) — runs BOTH suites (the inherited GRAIN
instrument suite, which must still be 17/17 green after the signed-path refactor, and the new
REGRAIN suite) and fails loud on any red test. Freeze is blocked until this job is green.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_dryrun --type r7i.4xlarge --no-weights \
      --s3-prefix results/regrain_dryrun --budget-min 60
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

SUITES = ("infra/pg/tests/test_grain_bsf.py", "infra/pg/tests/test_regrain_core.py",
          "infra/pg/tests/test_regrain_job_invariants.py")
PASSED_RE = re.compile(r"(\d+) passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    summary = {"object": "regrain_dryrun_v1", "suites": {}}
    rc_all = 0
    for suite in SUITES:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-v", "--tb=short", "-p", "no:cacheprovider"],
            cwd="/opt/pg/repo", capture_output=True, text=True)
        slug = pathlib.Path(suite).stem
        (out / f"pytest_{slug}.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
        tail = proc.stdout.strip().splitlines()[-12:]
        m = PASSED_RE.search(proc.stdout)
        summary["suites"][slug] = {"returncode": proc.returncode, "tail": tail,
                                   "n_passed": int(m.group(1)) if m else 0}
        print(f"\n===== {suite} (rc={proc.returncode}) =====")
        print(proc.stdout[-6000:])
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
        rc_all |= proc.returncode
    summary["n_passed_total"] = sum(v["n_passed"] for v in summary["suites"].values())
    summary["green"] = rc_all == 0
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    assert rc_all == 0, f"dry-run suites RED (rc={rc_all}) — freeze blocked"
    print("REGRAIN_DRYRUN_OK", json.dumps(
        {k: v["n_passed"] for k, v in summary["suites"].items()}
        | {"total": summary["n_passed_total"]}))


if __name__ == "__main__":
    main()
