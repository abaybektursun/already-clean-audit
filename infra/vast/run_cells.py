#!/usr/bin/env python3
"""Resume-aware cell dispatcher (SHACKLETON §9.5).

Each cell: skip if its S3 done-marker exists; else run its module, checksum
every artifact, upload, re-read the S3 listing and compare (§9.3), then write
the marker. A crash mid-cell leaves no marker, so a restarted rental redoes at
most one cell. After the last cell, writes the rental completion marker the
controller watchdog keys on.

Cell contract: {"id": ..., "module": "infra.vast.cells.<name>", ...params}
with the module exposing run(cell: dict, out_dir: str) -> None.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time


def sh(*args):
    subprocess.run(args, check=True)


def stamp(step):
    path = os.environ.get("TIMINGS")
    if path:
        with open(path, "a") as f:
            f.write(f"{time.time():.3f}\t{step}\n")


def s3_has(url):
    r = subprocess.run(["aws", "s3", "ls", url], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def s3_listing(prefix_url):
    r = subprocess.run(["aws", "s3", "ls", "--recursive", prefix_url],
                       capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            out[parts[3]] = int(parts[2])
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--s3", required=True, help="s3://bucket/results/shackleton_<rental>")
    args = p.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    man = json.load(open(args.manifest))
    cells = man["cells"]
    t_start = time.time()

    def progress(cid, state, idx):
        # boundary progress signal for the watchdog (uploaded immediately —
        # log-GROWTH covers within-cell liveness, this covers between-cell)
        note = {"cell": cid, "state": state, "idx": idx, "n": len(cells),
                "utc": time.strftime("%FT%TZ", time.gmtime())}
        ppath = "/workspace/out/progress.json"
        os.makedirs("/workspace/out", exist_ok=True)
        json.dump(note, open(ppath, "w"))
        sh("aws", "s3", "cp", ppath, f"{args.s3}/live_out/progress.json")

    for ci, cell in enumerate(cells):
        cid = cell["id"]
        marker = f"{args.s3}/cells/{cid}/_done.json"
        if s3_has(marker):
            print(f"[cells] {cid}: marker exists, skipping (resume)", flush=True)
            continue
        print(f"[cells] {cid}: running module {cell['module']}", flush=True)
        stamp(f"cell_{cid}_start")
        progress(cid, "start", ci)
        out_dir = f"/workspace/out/{cid}"
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        mod = importlib.import_module(cell["module"])
        mod.run(cell, out_dir)

        files = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(out_dir) for f in fs
        )
        assert files, f"cell {cid} produced no artifacts"
        sums = {}
        for fp in files:
            rel = os.path.relpath(fp, out_dir)
            sums[rel] = {"sha256": sha256(fp), "bytes": os.path.getsize(fp)}
            sh("aws", "s3", "cp", fp, f"{args.s3}/cells/{cid}/{rel}")

        # verify-then-mark: re-read the listing, compare names and sizes
        listed = s3_listing(f"{args.s3}/cells/{cid}/")
        for rel, meta in sums.items():
            hits = [k for k in listed if k.endswith(f"cells/{cid}/{rel}")]
            assert hits and listed[hits[0]] == meta["bytes"], \
                f"S3 verify failed for {cid}/{rel}"

        # D16: cells share one process — free the previous cell's model so
        # device_map=auto never sees phantom-full GPUs and CPU-offloads
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        done = {"cell": cid, "utc": time.strftime("%FT%TZ", time.gmtime()),
                "seconds": round(time.time() - t0, 1), "artifacts": sums}
        dpath = f"/workspace/out/{cid}_done.json"
        json.dump(done, open(dpath, "w"), indent=2)
        sh("aws", "s3", "cp", dpath, marker)
        stamp(f"cell_{cid}_banked")
        progress(cid, "banked", ci)
        print(f"[cells] {cid}: banked {len(files)} artifacts in {done['seconds']}s",
              flush=True)

    comp = {"rental": man["rental"], "cells": [c["id"] for c in cells],
            "utc": time.strftime("%FT%TZ", time.gmtime()),
            "total_seconds": round(time.time() - t_start, 1)}
    cpath = "/workspace/out/_rental_complete.json"
    json.dump(comp, open(cpath, "w"), indent=2)
    sh("aws", "s3", "cp", cpath, f"{args.s3}/_rental_complete.json")
    print("[cells] rental complete marker written", flush=True)


if __name__ == "__main__":
    main()
