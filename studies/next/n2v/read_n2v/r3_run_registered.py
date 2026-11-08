"""N2V reader, step 4: run the registered verdict code (infra.pg.n2v verdict, SPEC section 5) under python -I.

    python -I r3_run_registered.py --worktree <worktree> --bank <dir laid out as banks/score-02> --out <scratchpad>/verdict_n2v.json

-I drops the current directory from sys.path, so `python -I -m infra.pg.n2v` cannot import the repo; this file puts the
worktree root on sys.path explicitly and calls n2v.main with the arguments SPEC section 5 writes. N12's banks and
thresholds are n2v.main's defaults (the worktree's studies/next/n12/...).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--worktree", required=True)
ap.add_argument("--bank", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.worktree)
from infra.pg import n2v  # noqa: E402

n2v.main(["verdict", "--bank", a.bank, "--thresholds", f"{a.worktree}/studies/next/n2v/thresholds.json", "--out", a.out])
