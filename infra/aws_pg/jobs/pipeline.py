"""Server-side pipeline runner — the durable fix for the 'silent overnight idle' hole.

Runs a SEQUENCE of pg job steps inside ONE ephemeral instance, fail-loud, so a multi-arm
pipeline drives itself to completion ON EC2 — independent of the laptop, the agent loop, or
any background monitor. The per-arm sequencing lives on the instance (which self-terminates at
the end via the existing run_job.sh teardown), not in a laptop-tethered control plane. Before
this, "launch arm N+1 after arm N" was a laptop poll-loop that re-invoked the agent; if that
chain went quiet (laptop/agent idle) the pipeline stalled for hours with nobody watching. Here
the instance just keeps working until every step is done, then dies.

Steps come from an S3 JSON (avoids user-data quoting): a list of {name, module, args}. Each
step runs `<venv python> -m <module> <args> --out out/<name>/` as a subprocess, sharing the
instance's weights cache + the run_job.sh env + the single budget. Any non-zero step ABORTS
the pipeline (fail-loud → STATUS exit!=0 → the EventBridge reaper + the launcher's `pg watch`
fire). `out/pipeline_status.json` is rewritten after EVERY step transition (durable progress:
`pg watchdog` reads it to tell 'running' from 'stalled' from 'complete', so a stall is visible).
Its `started_utc` is the time this process started (whole seconds, UTC). run_job.sh starts it under `timeout --signal=KILL <budget>s`,
so it is the hard kill's clock to within the interpreter's start-up; steps read it for their elapsed time (n13_plan.launch_t0).

    # 1) upload the manifest, 2) launch the pipeline as one job:
    aws s3 cp steps.json s3://pg-lineage-769063704696/results/<prefix>/steps.json --profile fuelos
    pg run infra.aws_pg.jobs.pipeline \
        --args "--steps-s3 s3://pg-lineage-769063704696/results/<prefix>/steps.json" \
        --type g6e.8xlarge,p5.4xlarge --s3-prefix results/<prefix> --budget-min <sum-of-steps>

steps.json = [{"name": "retention_pt", "module": "infra.aws_pg.jobs.w15_retention",
               "args": "--pg-variant base-pt --seeds 0,1,2 --grad-n 256 --mmlu-per-subject 4"}, ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time


def _read_steps(steps_s3, out):
    local = out / "steps.json"
    subprocess.run(["aws", "s3", "cp", steps_s3, str(local), "--region", "us-east-1"], check=True)
    steps = json.loads(local.read_text())
    assert isinstance(steps, list) and steps, f"steps must be a non-empty JSON list, got {steps!r}"
    for s in steps:
        assert {"name", "module"} <= set(s), f"each step needs name+module, got {s!r}"
    return steps


def main():
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # the job's start on the instance (see the docstring)
    ap = argparse.ArgumentParser(description="server-side sequential pipeline runner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-s3", required=True, help="S3 path to a JSON list of {name, module, args}")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = _read_steps(a.steps_s3, out)

    status = {"started_utc": started, "n_steps": len(steps), "state": "running", "completed": 0, "steps": []}

    def _dump():
        (out / "pipeline_status.json").write_text(json.dumps(status, indent=2))

    _dump()
    print(f"[pipeline] {len(steps)} steps: {[s['name'] for s in steps]}", flush=True)

    for i, s in enumerate(steps):
        name, module, args = s["name"], s["module"], s.get("args", "")
        sout = out / name
        sout.mkdir(parents=True, exist_ok=True)
        rec = {"i": i, "name": name, "module": module, "args": args, "state": "running"}
        status["steps"].append(rec)
        _dump()
        cmd = [sys.executable, "-m", module, *shlex.split(args), "--out", str(sout)]
        print(f"[pipeline] step {i + 1}/{len(steps)} START {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode      # inherits env (PYTHONPATH, HF cache, venv); fail-loud
        rec["state"] = "ok" if rc == 0 else "failed"
        rec["returncode"] = rc
        if rc == 0:
            status["completed"] += 1
        else:
            status["state"] = "failed"
        _dump()
        if rc != 0:
            raise SystemExit(f"[pipeline] ABORT — step {name} failed rc={rc} (fail-loud; "
                             f"{status['completed']}/{len(steps)} steps done before failure)")
        print(f"[pipeline] step {i + 1}/{len(steps)} OK {name}", flush=True)

    status["state"] = "complete"
    _dump()
    print("PIPELINE_OK", json.dumps({"completed": status["completed"],
                                     "steps": [r["name"] for r in status["steps"]]}), flush=True)


if __name__ == "__main__":
    main()
