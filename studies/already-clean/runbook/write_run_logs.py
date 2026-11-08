"""Write the two per-run files SPEC_D1 §10 names, run.log and events.jsonl, beside every harvest run (closeout, 2026-10-24).

usage: AWS_PROFILE=fuelos AWS_REGION=us-east-1 python3 -I write_run_logs.py <harvest bank dir, e.g. .../bank/neurons> [--dry]

The jobs never wrote files with those names: the AWS lane wrote job.log and STATUS per run, the vast lane vast/job.log,
vast/timings.tsv, vast/_rental_complete.json, and both lanes out/pipeline_status.json and out/<cell>/manifest.json with
started_utc, finished_utc and timing_s. For every run prefix under s3://…/results/neurons/ that holds a job.log:
  run.log      = the job's job.log, copied verbatim server-side (vast lane: vast/job.log; attempts' logs stay where they are)
  events.jsonl = one JSON line per lifecycle event read from those files, with a first line that says so (derived, sources,
                 written_utc); nothing is invented: STATUS gives the exit code, timings.tsv the lifecycle marks, manifests the
                 per-stage start, finish and timings.
Runs whose vast-lane files live under another prefix (the center probe, run under results/mdl/) are listed in EXTRA.
Write-once: a prefix that already has run.log or events.jsonl is left alone and reported; --regenerate rewrites events.jsonl
(never run.log) after a change to this script (2026-10-24: corpus-stage compute added, EXTRA runs' source fields name their real prefix). Runs present in the local bank get
the same two files beside their out/.
"""
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile

BUCKET = "s3://pg-lineage-769063704696/results/neurons"
ROOT = "s3://pg-lineage-769063704696/results"
# runs whose vast-lane files (job.log, timings.tsv, _rental_complete.json) live under another prefix: run -> that prefix (under results/)
EXTRA = {"x/center_probe/1": "mdl/mendeleev_neurons_x_center_probe"}


def s3(*args, capture=True):
    r = subprocess.run(["aws", "s3", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"aws s3 {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout if capture else None


def listing():
    keys = [l.split()[-1] for l in s3("ls", BUCKET + "/", "--recursive").splitlines() if l.strip()]
    keys = [k[len("results/neurons/"):] for k in keys]
    runs = {}
    for k in keys:
        if k.endswith("/job.log") and "/attempts/" not in k:
            run = k[:-len("/vast/job.log")] if k.endswith("/vast/job.log") else k[:-len("/job.log")]
            runs[run] = "vast" if k.endswith("/vast/job.log") else "aws"
    return runs, set(keys)


def get_text(key, base=BUCKET):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pass
    s3("cp", f"{base}/{key}", f.name, "--only-show-errors")
    return pathlib.Path(f.name).read_text(encoding="utf-8", errors="replace")


def iso(epoch):
    return datetime.datetime.fromtimestamp(float(epoch), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def events_for(run, lane, keys, now, vast_prefix=None, vast_keys=None):
    """vast_prefix/vast_keys: where the vast-lane files live when not under <run>/vast (EXTRA runs); keys relative to results/"""
    src = []
    ev = []
    vp, vk, vbase = (vast_prefix, vast_keys, ROOT) if vast_prefix else (f"{run}/vast", keys, BUCKET)
    if lane == "aws" and f"{run}/STATUS" in keys:
        st = get_text(f"{run}/STATUS").strip(); src.append("STATUS")
        ev.append({"event": "status", "detail": st, "source": "STATUS"})
    if lane == "vast":
        if f"{vp}/timings.tsv" in vk:
            src.append(f"{vp}/timings.tsv" if vast_prefix else "vast/timings.tsv")
            for line in get_text(f"{vp}/timings.tsv", vbase).splitlines():
                if line.strip():
                    t, name = line.split("\t", 1)
                    ev.append({"utc": iso(t), "event": name.strip(), "source": f"{vp}/timings.tsv" if vast_prefix else "vast/timings.tsv"})
        if f"{vp}/_rental_complete.json" in vk:
            src.append(f"{vp}/_rental_complete.json" if vast_prefix else "vast/_rental_complete.json")
            rc = json.loads(get_text(f"{vp}/_rental_complete.json", vbase))
            ev.append({"utc": rc.get("utc"), "event": "rental_complete", "rental": rc.get("rental"), "cells": rc.get("cells"),
                       "total_seconds": rc.get("total_seconds"), "source": f"{vp}/_rental_complete.json" if vast_prefix else "vast/_rental_complete.json"})
    if f"{run}/out/pipeline_status.json" in keys:
        src.append("out/pipeline_status.json")
        ps = json.loads(get_text(f"{run}/out/pipeline_status.json"))
        ev.append({"event": "pipeline_status", "detail": ps if len(json.dumps(ps)) < 4000 else {"keys": sorted(ps)}, "source": "out/pipeline_status.json"})
    stages = sorted(k for k in keys if k.startswith(run + "/out/") and k.endswith("/manifest.json") and k.count("/") == run.count("/") + 3)
    for k in stages:
        m = json.loads(get_text(k)); cell = k.split("/")[-2]; src.append(f"out/{cell}/manifest.json")
        if m.get("started_utc"):
            ev.append({"utc": m["started_utc"], "event": "stage_start", "stage": cell, "source": f"out/{cell}/manifest.json"})
        if m.get("finished_utc"):
            ev.append({"utc": m["finished_utc"], "event": "stage_finish", "stage": cell, "timing_s": m.get("timing_s"),
                       "load_s": m.get("load_s"), "tokens_per_second": m.get("tokens_per_second"),
                       "n_tokens_scored": m.get("n_tokens_scored"), "device_name": m.get("device_name"), "source": f"out/{cell}/manifest.json"})
    for k in sorted(k for k in keys if k.startswith(run + "/out/corpus/") and k.endswith("/manifest.json")):
        m = json.loads(get_text(k)); rel = k[len(run) + 1:]; src.append(rel)
        e = {"utc": m.get("finished_utc") or m.get("created_utc") or m.get("built_utc") or m.get("utc"), "event": "corpus_stage", "stage": "corpus", "source": rel}
        e.update({k: m[k] for k in ("read_seconds", "n_tokens", "n_docs", "sha256", "manifest_sha256") if m.get(k) is not None})   # only fields the manifest has (F38)
        ev.append(e)
    ev.sort(key=lambda e: e.get("utc") or "")
    head = {"derived": True, "written_utc": now, "run": run, "lane": lane, "sources": src,
            "note": "SPEC_D1 §10 names events.jsonl and run.log per run; the jobs wrote the sources listed here instead. run.log is the "
                    "job's job.log copied verbatim; every event line below is read from the named source (runbook/write_run_logs.py, 2026-10-24)."}
    return [head] + ev


def main(bank, dry, regenerate=False):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs, keys = listing()
    extra_keys = {}
    for run, pref in EXTRA.items():
        runs[run] = "vast"
        extra_keys[run] = {l.split()[-1][len("results/"):] for l in s3("ls", f"{ROOT}/{pref}/", "--recursive").splitlines() if l.strip()}
    done, skipped = [], []
    for run, lane in sorted(runs.items()):
        if (f"{run}/run.log" in keys or f"{run}/events.jsonl" in keys) and not regenerate:
            skipped.append(run); continue
        if run in EXTRA:
            src_log = f"{ROOT}/{EXTRA[run]}/job.log"
            lines = events_for(run, lane, keys, now, EXTRA[run], extra_keys[run])
        else:
            src_log = f"{BUCKET}/" + (f"{run}/vast/job.log" if lane == "vast" else f"{run}/job.log")
            lines = events_for(run, lane, keys, now)
        if dry:
            print("DRY", run, lane, "events", len(lines) - 1, "sources", lines[0]["sources"][:4]); continue
        if f"{run}/run.log" not in keys:
            s3("cp", src_log, f"{BUCKET}/{run}/run.log", "--only-show-errors")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for e in lines:
                f.write(json.dumps(e) + "\n")
        s3("cp", f.name, f"{BUCKET}/{run}/events.jsonl", "--only-show-errors")
        local = pathlib.Path(bank) / run
        if local.is_dir():
            if not (local / "run.log").exists():
                s3("cp", src_log, str(local / "run.log"), "--only-show-errors")
            if regenerate or not (local / "events.jsonl").exists():
                (local / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
        done.append((run, lane, len(lines) - 1, local.is_dir()))
    for d in done:
        print("WROTE", *d)
    print(json.dumps({"runs": len(runs), "written": len(done), "skipped_existing": skipped}))


if __name__ == "__main__":
    main(sys.argv[1], "--dry" in sys.argv[2:], "--regenerate" in sys.argv[2:])
