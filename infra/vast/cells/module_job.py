"""One Vast cell = one python module command, `python -m <job> <args> --out <out_dir>`, then out_dir synced to S3.

Manifest cell: {"id": "q4b", "module": "infra.vast.cells.module_job",
                "job": "infra.aws_pg.jobs.<module>", "args": "--repo Qwen/Qwen3-4B-Base ...",
                "s3_out": "s3://pg-lineage-769063704696/results/mdl/neurons/show/<n>/<model>/",
                "pg_git_sha": "<short sha of the commit that holds the shipped code>"}

The job runs as a subprocess in the box environment, so lifecycle.sh's env contract holds: PYTHONPATH=/app,
HF_HUB_OFFLINE cleared when the manifest env sets HF_ONLINE=1, HF_TOKEN shipped by controller.py from SSM
/pg-lineage/hf-token, HF_HOME / HF_HUB_CACHE from the manifest env. The shipped code is `git archive HEAD infra
configs scripts` (no .git on the box), so the job reads its commit from PG_GIT_SHA, as pipeline_job does.

out_dir is synced to s3_out whether the job exits 0 or not (a gate abort still banks its metrics.json and STATUS);
the cell then raises on a nonzero exit, so run_cells.py writes no done-marker and lifecycle.sh destroys the
instance. run_cells.py banks the same files again under the rental prefix (cells/<id>/), and the onstart streamer
copies /workspace/out to live_out/ every 60 s while the job runs.

S3 credentials: the box holds the scoped IAM user named by program.json box_profile (mendeleev-box), which writes
only results/mdl/*, so s3_out must sit under results/mdl/ (an AccessDenied fails the sync loudly). Copy it to
results/neurons/... server side afterwards (infra/vast/operations/2026-10-26_neuronshow_path.md).
"""
import os
import shlex
import subprocess
import sys


def run(cell, out_dir):
    os.environ["PG_GIT_SHA"] = cell["pg_git_sha"]
    cmd = [sys.executable, "-m", cell["job"], *shlex.split(cell["args"]), "--out", out_dir]
    print(f"[module_job] {cell['id']}: {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd).returncode
    subprocess.run(["aws", "s3", "sync", out_dir, cell["s3_out"]], check=True)
    print(f"[module_job] {cell['id']}: exit {code}; {out_dir} synced to {cell['s3_out']}", flush=True)
    assert code == 0, f"{cell['job']} exited {code}"
