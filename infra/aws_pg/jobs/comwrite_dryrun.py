"""COMWRITE §10.1 weight-free dry-run gate — run the three COMWRITE tiny-tensor suites (core /
surgery / adjudicator) on a CPU box; record pass counts + content hashes. The per-arm freeze
tags land only after this is green (spec §10, freeze checklist). No model, no GPU, no datasets
— seconds on r7i (endow_dryrun precedent).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/comwrite_dryrun --budget-min 30
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

SUITES = {
    "core": "infra/pg/tests/test_comwrite_core.py",
    "surgery": "infra/pg/tests/test_comwrite_surgery.py",
    "adjudicate": "infra/pg/tests/test_comwrite_adjudicate.py",
    "b_analysis": "infra/pg/tests/test_comwrite_b_analysis.py",
}
HASHED = ("infra/pg/comwrite_core.py", "infra/pg/comwrite_surgery.py",
          "infra/pg/comwrite_adjudicate.py",
          "infra/aws_pg/jobs/comwrite_b_analysis.py") + tuple(SUITES.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="core,surgery,adjudicate,b_analysis")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    rep = {"suites": {}, "sha256": {}}
    for f in HASHED:
        rep["sha256"][f] = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
    ok = True
    for name in a.suites.split(","):
        suite = SUITES[name]
        r = subprocess.run([sys.executable, "-m", "pytest", suite, "-q", "--tb=short"],
                           capture_output=True, text=True)
        rep["suites"][name] = {"suite": suite, "exit": r.returncode,
                               "tail": r.stdout[-3000:] + r.stderr[-2000:]}
        ok = ok and r.returncode == 0
    rep["green"] = ok
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
    print("DRYRUN_OK" if ok else "DRYRUN_FAIL",
          json.dumps({n: v["exit"] for n, v in rep["suites"].items()}))
    assert ok, "COMWRITE dry-run gate RED — do not freeze"


if __name__ == "__main__":
    main()
