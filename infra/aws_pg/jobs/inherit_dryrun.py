"""INHERIT §9.1 weight-free dry-run gate — run the INHERIT tiny-tensor suite + the CTRLIFACE
27-test suite (the reused instrument) on a CPU box; record pass counts + content hashes. The
freeze tag lands only after this is green (spec §9.3).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/inherit_dryrun --budget-min 30
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

SUITES = ("infra/pg/tests/test_inherit_core.py", "infra/pg/tests/test_ctrliface_core.py")
HASHED = ("infra/pg/inherit_core.py", "infra/pg/inherit_pg1.py",
          "infra/aws_pg/jobs/_inherit_shared.py", "infra/pg/tests/test_inherit_core.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # pytest is not guaranteed in the AMI; ensure it (idempotent, cheap). numpy/torch are baked.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    rep = {"suites": {}, "sha256": {}}
    for f in HASHED:
        rep["sha256"][f] = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
    ok = True
    for suite in SUITES:
        r = subprocess.run([sys.executable, "-m", "pytest", suite, "-q", "--tb=short"],
                           capture_output=True, text=True)
        rep["suites"][suite] = {"exit": r.returncode, "tail": r.stdout[-3000:] + r.stderr[-2000:]}
        ok = ok and r.returncode == 0
    rep["green"] = ok
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
    print("DRYRUN_OK" if ok else "DRYRUN_FAIL", json.dumps({s: v["exit"] for s, v in rep["suites"].items()}))
    assert ok, "dry-run gate RED — do not freeze"


if __name__ == "__main__":
    main()
