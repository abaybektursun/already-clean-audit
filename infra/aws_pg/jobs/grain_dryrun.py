"""GRAIN §11.1 weight-free dry-run job (CPU, --no-weights) — runs the mandatory tiny-tensor
suite (m=1 exact reduction vs the house BatchTopKSAE, block-norm selection, planted recovery,
QR unions, splice pass-through no-op, adjudicator tables, Jaccard null) and fails loud on any
red test. Freeze is blocked until this job is green.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_dryrun --type r7i.4xlarge --no-weights \
      --s3-prefix results/grain_dryrun --budget-min 60
"""
import argparse
import json
import pathlib
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "infra/pg/tests/test_grain_bsf.py", "-v", "-x",
         "--tb=short"],
        cwd="/opt/pg/repo", capture_output=True, text=True)
    (out / "pytest.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    summary = {"returncode": proc.returncode,
               "tail": proc.stdout.strip().splitlines()[-15:]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(proc.stdout[-4000:])
    assert proc.returncode == 0, f"dry-run suite RED (rc={proc.returncode}) — freeze blocked"
    print("GRAIN_DRYRUN_OK")


if __name__ == "__main__":
    main()
