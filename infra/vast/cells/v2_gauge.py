"""V2 cell: run the M-Q2 gauge job (REUSED instrument, verbatim) as a rental cell.

Thin wrapper: the science lives in infra/aws_pg/jobs/mercator_mq2.py — the five-arm
gauge machinery, transported banks, delivery statistic, ExactAdam, and its own
function-preservation gates (which act as this cell's in-run instrument gate). The
wrapper only sets arguments and lets the job write its house artifacts (atlas.jsonl,
gates.json, result json) into out_dir, which run_cells banks per §9.5.

Cell params: gauges, seeds, steps (smoke cells pass tiny values — §9.9's smallest
unit is 1 gauge x 1 seed x 2 steps).
"""
import subprocess
import sys


def run(cell, out_dir):
    cmd = [sys.executable, "-m", "infra.aws_pg.jobs.mercator_mq2",
           "--out", out_dir,
           "--gauges", cell["gauges"],
           "--seeds", cell["seeds"],
           "--steps", str(cell["steps"])]
    for flag in ("lr", "n-train", "n-eval", "n-seq", "n-gate-seq"):
        key = flag.replace("-", "_")
        if key in cell:
            cmd += [f"--{flag}", str(cell[key])]
    print(f"[v2_gauge] exec: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
