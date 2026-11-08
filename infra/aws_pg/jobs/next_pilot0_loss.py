"""studies/next/pilot0 GPU loss check, second check (J8, pilot0/GPU_CHECK2.md): five runs of pilot0/loss_check.py.

The first check (commit ed3e72006, pilot0/GPU_CHECK.md) ran batch_60001 eager and sdpa and batch_60000 sdpa at
stage1-step60000. This version runs the J8 pairs. Trainer step s trains the 0-based batch s - 1 with the weights
after s - 1 updates, so the target batch of step s and its neighbour (the batch of step s - 1) are both scored
under checkpoint stage1-step<s - 1>, and both are compared with the W&B value at step s:
  1. batch_61001,  stage1-step61000,  target    W&B step 61,001  2.1173441410064697
  2. batch_61000,  stage1-step61000,  neighbour same W
  3. batch_100001, stage1-step100000, target    W&B step 100,001 2.0262222290039062
  4. batch_100000, stage1-step100000, neighbour same W
  5. batch_60001,  stage1-step60000,  known number: the first 32 rows, against the banked pilot-0 per-row values
All runs use sdpa. Inputs, uploaded before launch to s3://<bucket>/<PG_S3_PREFIX>/inputs/: loss_check.py and
batch_<s>/{tokens.npy,meta.json} for each run. The checkpoints are downloaded in-job from the HF hub (the S3 weight
cache holds no OLMo-3 weights, infra/pg/matrix_repo_facts.json), one at a time in a background thread, in order of
first use, so later downloads overlap the GPU runs. Each run writes out/batch_<s>/loss_check_sdpa[_rows<n>].json,
rewritten every 16 rows. This parent is the only writer of out/metrics.json, rewritten every 10 s while a run is in
flight, with the per-row values copied from the run files. A non-zero child exit or a failed download raises.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.next_pilot0_loss --no-weights --type g6e.2xlarge \\
      --s3-prefix results/next_pilot0/<attempt> --budget-min 49 --detach
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "pg-lineage-769063704696"
REPO = "allenai/Olmo-3-1025-7B"
SD = 0.0248
RUNS = (  # (batch dir, role, revision, W&B value compared against, rows scored)
    ("batch_61001", "target step 61,001", "stage1-step61000", 2.1173441410064697, 512),
    ("batch_61000", "neighbour of step 61,001", "stage1-step61000", 2.1173441410064697, 512),
    ("batch_100001", "target step 100,001", "stage1-step100000", 2.0262222290039062, 512),
    ("batch_100000", "neighbour of step 100,001", "stage1-step100000", 2.0262222290039062, 512),
    ("batch_60001", "known number, pilot-0 batch", "stage1-step60000", 2.1240015029907227, 32))


def imds(path):
    req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
    tok = urllib.request.urlopen(req, timeout=5).read().decode()
    req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                 headers={"X-aws-ec2-metadata-token": tok})
    return urllib.request.urlopen(req, timeout=5).read().decode()


def write_metrics(out, state):
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, out / "metrics.json")


def read_progress(run, path):
    if path.exists():
        r = json.loads(path.read_text())
        run.update(rows_done=r["rows_done"], complete=r["complete"], mean_ce_so_far=r["mean_ce"],
                   mean_ce_excluding_pad_targets_so_far=r["pad"]["mean_ce_excluding_pad_targets"],
                   pad_target_positions_so_far=r["pad"]["pad_target_positions"],
                   per_row_mean_ce_excluding_pad_targets=r["per_row_mean_ce_excluding_pad_targets"],
                   per_row_pad_target_positions=r["pad"]["per_row_pad_target_positions"],
                   per_row_nonpad_target_positions=r["per_row_nonpad_target_positions"])


def download(revision):
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snap = snapshot_download(REPO, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"],
                             max_workers=8)
    nbytes = sum(f.stat().st_size for f in pathlib.Path(snap).rglob("*") if f.is_file())
    print(f"downloaded {revision}: {nbytes:,} B in {time.time() - t0:.1f} s -> {snap}", flush=True)
    return {"revision": revision, "seconds": round(time.time() - t0, 1), "bytes": nbytes, "snapshot": str(snap)}


def run_all(script, in_dir, out, state, instance_type, downloads, runs=RUNS, poll_s=10, attn="sdpa"):
    for name, role, revision, target, rows in runs:
        t_wait = time.time()
        state["downloads"][revision] = downloads[revision].result()  # blocks until this checkpoint is on disk
        rel = f"{name}/loss_check_{attn}.json" if rows == 512 else f"{name}/loss_check_{attn}_rows{rows}.json"
        run = {"run": rel, "role": role, "revision": revision, "attn": attn, "target": target, "rows": rows,
               "status": "running", "rows_done": 0, "mean_ce_so_far": None, "complete": False,
               "download_wait_seconds": round(time.time() - t_wait, 1), "seconds": 0.0}
        state["runs"].append(run)
        cmd = [sys.executable, str(script), "--tokens", str(in_dir / name / "tokens.npy"), "--revision", revision,
               "--target", repr(target), "--sd", repr(SD), "--attn", attn, "--out", str(out / rel),
               "--instance-type", instance_type, "--rows", str(rows)]
        print("$", " ".join(cmd), flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd)
        while p.poll() is None:
            read_progress(run, out / rel)
            run["seconds"] = round(time.time() - t0, 1)
            write_metrics(out, state)
            time.sleep(poll_s)
        read_progress(run, out / rel)
        run["seconds"] = round(time.time() - t0, 1)
        run["status"] = "done" if p.returncode == 0 and run["complete"] else f"failed rc={p.returncode}"
        write_metrics(out, state)
        if run["status"] != "done":
            raise RuntimeError(f"{rel}: {run['status']}, complete={run['complete']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="1,2,3,4,5", help="1-based indices into RUNS, run in the given order")
    a = ap.parse_args()
    runs = [RUNS[int(i) - 1] for i in a.runs.split(",")]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = os.environ["PG_S3_PREFIX"]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()  # HF token from SSM + online mode, before any huggingface import
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the house setting (reference_hf_xet_wedge)
    import torch
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    assert "olmo3" in CONFIG_MAPPING_NAMES, f"transformers {transformers.__version__} does not register olmo3"
    assert torch.cuda.is_available(), "no CUDA on this instance"

    state = {"instance_id": imds("instance-id"), "instance_type": imds("instance-type"),
             "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
             "transformers": transformers.__version__, "s3_prefix": prefix, "stage": "inputs", "runs": []}
    write_metrics(out, state)

    in_dir = pathlib.Path("/work/in")
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{prefix}/inputs/", str(in_dir), "--recursive",
                    "--region", "us-east-1"], check=True)
    for name, _, _, _, _ in runs:
        assert (in_dir / name / "tokens.npy").stat().st_size == 16777344, name
        assert (in_dir / name / "meta.json").exists(), name

    revisions = list(dict.fromkeys(r[2] for r in runs))  # order of first use
    pool = ThreadPoolExecutor(max_workers=1)  # one download at a time; .result() re-raises a failed download
    downloads = {rev: pool.submit(download, rev) for rev in revisions}
    state.update(stage="runs", runs_requested=a.runs, downloads={})
    write_metrics(out, state)
    try:
        run_all(in_dir / "loss_check.py", in_dir, out, state, state["instance_type"], downloads, runs)
    finally:  # on a raise, do not let queued downloads keep the process alive; the error still propagates
        pool.shutdown(wait=False, cancel_futures=True)
    state["stage"] = "complete"
    write_metrics(out, state)


if __name__ == "__main__":
    main()
