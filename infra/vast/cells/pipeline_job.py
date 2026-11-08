"""One Vast cell = the AWS pipeline runner (infra.aws_pg.jobs.pipeline) on a steps.json, unchanged.

Manifest cell: {"id": "probe_r5", "module": "infra.vast.cells.pipeline_job",
                "steps_s3": "s3://.../steps.json", "pg_git_sha": "<short sha of the shipped commit>"}
The shipped code is `git archive HEAD infra configs scripts`, so pg_git_sha is exact (never -dirty);
the jobs record it in manifest.json as on AWS. Each step writes <out_dir>/<step>/, which run_cells.py banks.

Env contract on the box: PYTHONPATH=/app and HF_HUB_OFFLINE=1 from lifecycle.sh (weights_mode "s3" syncs
the cache to /workspace/hf); HF_HUB_CACHE=/workspace/hf/hub from the manifest env, because
neurons_corpus.local_snapshot reads it; HF_TOKEN from controller.py. Inputs are fetched by the jobs
themselves (aws s3 cp), so every S3 path in the steps must sit under the box user's results/mdl/*.
"""
import os
import subprocess
import sys


def run(cell, out_dir):
    os.environ["PG_GIT_SHA"] = cell["pg_git_sha"]
    cmd = [sys.executable, "-m", "infra.aws_pg.jobs.pipeline", "--steps-s3", cell["steps_s3"], "--out", out_dir]
    print(f"[pipeline_job] {cell['id']}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
