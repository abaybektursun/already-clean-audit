"""One Vast cell = one N13 launch: infra.aws_pg.jobs.pipeline on a steps.json, with the AWS lane's out/ layout and S3 prefix.

Manifest cell: {"id": "n13", "module": "infra.vast.cells.n13_pipeline", "attempt": "s2smoke-01v",
                "steps_s3": "s3://pg-lineage-769063704696/results/next_n13/s2smoke-01v/steps.json", "budget_min": 60,
                "upload_probe_gb": 1}   (written by infra/vast/n13_manifest.py; upload_probe_gb > 0 times an S3 upload first)

It does on the box what infra/aws_pg/run_job.sh does on EC2, so the laptop reads a Vast attempt exactly as an AWS one:
  out dir /work/out (the steps name /work/out/setup and /work/out/r1_rank), cwd = the code root;
  the job env: HF_HOME=/opt/hf-cache/hf (where the image bakes the three revisions, the AWS lane's HF_HOME), HF_HUB_OFFLINE=1,
    PYTHONUNBUFFERED=1, TORCHINDUCTOR_CACHE_DIR=/work/jit, PG_S3_PREFIX; TRANSFORMERS_OFFLINE removed (run_job.sh does not set it);
  `timeout --signal=KILL <budget_min> min` around the pipeline (the same hard clock pipeline_status.json's started_utc measures);
  nvidia-smi dmon -s u -d 30 into out/dmon.log (spec §11.5);
  every 30 s: the rental log as job.log and out/ to s3://<bucket>/results/next_n13/<attempt>/; at the end a final sync and
    STATUS ("exit=<code>").
The lane's own record (manifest, code, job_live.log, live_out/, the cell marker) stays under results/mdl/mendeleev_<rental>/.
Output progress for the watchdog: /workspace/out/progress.json is rewritten whenever a file under /work/out changes (the
lane's 60 s streamer uploads it to live_out/progress.json).
A non-zero pipeline exit raises after the final sync, so the lane classifies it as a science failure (no respawn).
"""
import json
import os
import subprocess
import sys
import threading
import time

BUCKET = "pg-lineage-769063704696"
OUT = "/work/out"
JOB_LOG = "/workspace/job.log"          # lifecycle.sh tees the whole rental, this cell's subprocesses included, here
HF = "/opt/hf-cache/hf"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def newest_mtime(root):
    return max((os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs), default=0.0)


def sync(s3, check):
    """job.log and out/ to the attempt prefix. The 30 s streamer passes check=False, as run_job.sh's streamer does: one failed
    upload must not end the streaming, and the next tick retries; each failure is printed. The final sync passes check=True."""
    for cmd in (["aws", "s3", "cp", JOB_LOG, f"{s3}/job.log", "--only-show-errors"],
                ["aws", "s3", "sync", OUT, f"{s3}/out/", "--only-show-errors"]):
        rc = subprocess.run(cmd, check=check).returncode
        if rc != 0:
            print(f"[n13_pipeline] upload failed rc={rc}: {' '.join(cmd[:3])}; the next tick retries", flush=True)


def run(cell, out_dir):
    attempt = cell["attempt"]
    s3 = f"s3://{BUCKET}/results/next_n13/{attempt}"
    assert cell["steps_s3"] == f"{s3}/steps.json", (cell["steps_s3"], s3)
    os.makedirs(OUT)                      # no exist_ok: a restarted rental must not write into a half-filled out/
    os.makedirs("/work/jit", exist_ok=True)
    env = dict(os.environ)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.update({"HF_HOME": HF, "HF_HUB_CACHE": f"{HF}/hub", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/app", "TORCHINDUCTOR_CACHE_DIR": "/work/jit", "PG_S3_PREFIX": f"results/next_n13/{attempt}"})
    rec = {"attempt": attempt, "s3": s3, "steps_s3": cell["steps_s3"], "budget_min": cell["budget_min"], "started_utc": utc()}
    if cell.get("upload_probe_gb"):
        # the lane's reliability check: an S3 upload of N GB from this host, timed, into the lane's own prefix (not the attempt's)
        n = int(cell["upload_probe_gb"])
        subprocess.run(["head", "-c", str(n << 30), "/dev/urandom"], stdout=open("/workspace/upload_probe.bin", "wb"), check=True)
        dst = f"{os.environ['S3_BASE']}/{os.environ['PROGRAM']}_{os.environ['RENTAL']}/upload_probe.bin"
        t0 = time.time()
        subprocess.run(["aws", "s3", "cp", "/workspace/upload_probe.bin", dst, "--only-show-errors"], check=True)
        dt = time.time() - t0
        rec["upload_probe"] = {"bytes": n << 30, "seconds": round(dt, 2), "MB_per_s": round((n << 30) / 1e6 / dt, 1), "dst": dst}
        os.remove("/workspace/upload_probe.bin")
        print(f"[n13_pipeline] upload probe: {rec['upload_probe']}", flush=True)
    print(f"[n13_pipeline] {attempt}: out {OUT} -> {s3}/out/", flush=True)

    dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "30"], stdout=open(f"{OUT}/dmon.log", "w"),
                            stderr=subprocess.STDOUT)
    stop = threading.Event()

    def streamer():
        seen = 0.0
        while not stop.wait(30):
            sync(s3, check=False)
            t = newest_mtime(OUT)
            if t > seen:
                seen = t
                st = f"{OUT}/pipeline_status.json"
                note = {"cell": cell["id"], "state": "running", "newest_out_utc": time.strftime("%FT%TZ", time.gmtime(t)),
                        "pipeline": json.load(open(st)) if os.path.exists(st) else None, "utc": utc()}
                json.dump(note, open("/workspace/out/progress.json", "w"))

    th = threading.Thread(target=streamer, daemon=True)
    th.start()
    cmd = ["timeout", "--signal=KILL", f"{int(cell['budget_min'] * 60)}s", sys.executable, "-m", "infra.aws_pg.jobs.pipeline",
           "--steps-s3", cell["steps_s3"], "--out", OUT]
    print(f"[n13_pipeline] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env, cwd="/app").returncode
    stop.set()
    th.join()
    dmon.terminate()
    sync(s3, check=True)
    open("/tmp/STATUS", "w").write(f"exit={rc}\n")
    subprocess.run(["aws", "s3", "cp", "/tmp/STATUS", f"{s3}/STATUS", "--only-show-errors"], check=True)
    rec.update({"returncode": rc, "finished_utc": utc()})
    json.dump(rec, open(os.path.join(out_dir, "n13_cell.json"), "w"), indent=1)
    print(f"[n13_pipeline] {attempt}: pipeline exit {rc}", flush=True)
    if rc != 0:
        raise SystemExit(f"[n13_pipeline] {attempt}: pipeline exit {rc} (STATUS on S3; fail-loud)")
