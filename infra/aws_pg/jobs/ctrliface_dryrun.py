"""CTRLIFACE §9.1 weight-free dry-run gate — run the pure-logic unit tests on a cheap CPU
box, NO weights, NO GPU, NO lerobot. Gates the FREEZE: the spec is frozen only at the commit
that passes this and records the recorded content-hash of the core + test in the journal.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/ctrliface_dryrun --budget-min 20

Fail-loud: a failing test → non-zero pytest rc → job STATUS exit=1 (the launcher surfaces it).
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def _sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # pytest is not guaranteed in the AMI; ensure it (idempotent, cheap). numpy/torch are baked.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

    core = "infra/pg/ctrliface_core.py"
    test = "infra/pg/tests/test_ctrliface_core.py"
    hashes = {core: _sha256(core), test: _sha256(test)}

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test, "-v", "--no-header"],
        cwd="/opt/pg/repo", capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)

    res = {
        "gate": "ctrliface_dryrun_9.1",
        "pytest_rc": proc.returncode,
        "passed": proc.returncode == 0,
        "content_hashes": hashes,
        "python": sys.version.split()[0],
        "tail": proc.stdout.strip().splitlines()[-25:],
    }
    (out / "dryrun.json").write_text(json.dumps(res, indent=2))
    print("DRYRUN", json.dumps({"passed": res["passed"], "hashes": hashes}))
    # fail-loud: propagate the pytest rc so a red test blocks the freeze
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
