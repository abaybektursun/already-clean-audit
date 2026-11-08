"""Filesystem client for infra/pg/probe_server.py: write one request, wait for its response.

    python -m infra.pg.probe_client --requests DIR --responses DIR --json '{"op": "vocab", "model": "G4", "pole": "L18N3736-"}'
    python -m infra.pg.probe_client --requests DIR --responses DIR --file request.json [--timeout 600]

The request gets an id (its "id" field, else a new one), is written to a hidden temp file and renamed to
DIR/<id>.json (the server reads only *.json, so it never sees a partial file); the client then waits for
responses/<id>.json and prints it. Exit code 0 when the response says ok, 1 when it carries an error, 2 on timeout.
Any transport that moves these two directories works unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import uuid


def submit(req_dir, req: dict) -> str:
    req = dict(req)
    req.setdefault("id", f"{req.get('op', 'req')}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")
    d = pathlib.Path(req_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{req['id']}.json.tmp"
    tmp.write_text(json.dumps(req))
    os.replace(tmp, d / f"{req['id']}.json")
    return req["id"]


def wait(resp_dir, rid: str, timeout: float = 600.0, poll: float = 0.2) -> dict | None:
    f = pathlib.Path(resp_dir) / f"{rid}.json"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if f.exists():
            return json.loads(f.read_text())
        time.sleep(poll)
    return None


def ask(req_dir, resp_dir, req: dict, timeout: float = 600.0, poll: float = 0.2) -> dict | None:
    return wait(resp_dir, submit(req_dir, req), timeout, poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--responses", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--json")
    g.add_argument("--file")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    req = json.loads(a.json if a.json is not None else pathlib.Path(a.file).read_text())
    rid = submit(a.requests, req)
    resp = wait(a.responses, rid, a.timeout)
    if resp is None:
        print(f"timeout after {a.timeout} s waiting for {rid}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(resp, indent=1, ensure_ascii=False))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
