"""ENDOW §9.1 weight-free dry-run gate — run the four ENDOW tiny-tensor suites (O / R / F / J) on a
CPU box; record pass counts + content hashes. Each workstream's freeze tag lands only after its
suite is green (spec §9, freeze checklist). No model, no GPU, no datasets — seconds on r7i.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/endow_dryrun --budget-min 30
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

# Per-workstream suites (freeze each separately): O, R, F, J.
SUITES = {
    "O": "infra/pg/tests/test_endow_o.py",
    "R": "infra/pg/tests/test_endow_transplant.py",
    "F": "infra/pg/tests/test_endow_atlas.py",
    "J": "infra/pg/tests/test_endow_writefrac.py",
}
HASHED = ("infra/pg/endow_o.py", "infra/pg/endow_transplant.py",
          "infra/pg/endow_atlas.py", "infra/pg/endow_writefrac.py") + tuple(SUITES.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ws", default="ORFJ", help="which workstream suites to run (subset of O,R,F,J)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    rep = {"suites": {}, "sha256": {}}
    for f in HASHED:
        rep["sha256"][f] = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
    ok = True
    for ws, suite in SUITES.items():
        if ws not in a.ws:
            continue
        r = subprocess.run([sys.executable, "-m", "pytest", suite, "-q", "--tb=short"],
                           capture_output=True, text=True)
        rep["suites"][ws] = {"suite": suite, "exit": r.returncode,
                             "tail": r.stdout[-3000:] + r.stderr[-2000:]}
        ok = ok and r.returncode == 0
    rep["green"] = ok
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
    print("DRYRUN_OK" if ok else "DRYRUN_FAIL",
          json.dumps({ws: v["exit"] for ws, v in rep["suites"].items()}))
    assert ok, "dry-run gate RED — do not freeze"


if __name__ == "__main__":
    main()
