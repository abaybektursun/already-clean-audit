#!/usr/bin/env python3
"""Probe station client: a request JSON file in, the server's response JSON out, over one SSH command per probe.

Laptop (stdlib only; finds the instance labelled <program>_<rental> through the vastai CLI):
  python3 infra/vast/probe.py send REQUEST.json [--timeout 600]  response on stdout, timings on stderr
  python3 infra/vast/probe.py log [-n 50]                         tail of the server log (+ server_exit.json)
  python3 infra/vast/probe.py deploy --ref REF --server "CMD"     ship `git archive REF infra configs scripts` into
                                                                  /app, restart the server as CMD (PG_GIT_SHA = REF)
  python3 infra/vast/probe.py stop                                end the station now: banked, then destroyed
  python3 infra/vast/probe.py status                              credential-free instance summary
  python3 infra/vast/probe.py url                                 the ssh URL; `export PROBE_SSH=<url>` skips the two
                                                                  vastai lookups on every later call
  --rental NAME (default neuronshow_probe) goes before the subcommand.
Box (run by `send` over ssh):
  python3 /app/infra/vast/probe.py box RID TIMEOUT < request      write requests/RID.json, print responses/RID.json

Exit codes of send: 0 response printed; 3 the server process has exited (log tail on stderr); 4 no response within
--timeout; 255 ssh failed. The directory protocol is in infra/vast/cells/probe_station.py.
"""
import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
BOX = pathlib.Path("/workspace/probe")
CTRL = os.path.expanduser("~/.ssh/vast-probe-%C")


def tail(path, n):
    return "".join(path.read_text().splitlines(keepends=True)[-n:]) if path.exists() else f"({path} absent)\n"


def box(rid, timeout):
    body = sys.stdin.buffer.read()
    tmp = BOX / "requests" / f"{rid}.tmp"
    tmp.write_bytes(body)
    tmp.rename(BOX / "requests" / f"{rid}.json")
    res, down = BOX / "responses" / f"{rid}.json", BOX / "server_exit.json"
    t_end = time.time() + timeout
    while not res.exists():
        if down.exists():
            sys.stderr.write(f"server process exited: {down.read_text()}\n" + tail(BOX / "server.log", 30))
            sys.exit(3)
        if time.time() > t_end:
            sys.stderr.write(f"no response to {rid} after {timeout} s\n" + tail(BOX / "server.log", 30))
            sys.exit(4)
        time.sleep(0.02)
    sys.stdout.write(res.read_text())


def status(rental):
    """Never print raw instance objects: Vast includes credentials in extra_env."""
    label = f"{json.loads((HERE / 'program.json').read_text())['program']}_{rental}"
    result = subprocess.run(["vastai", "show", "instances", "--raw"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Vast instance lookup failed (exit {result.returncode}); raw output withheld")
    fields = ("id", "label", "actual_status", "gpu_name", "dph_total", "duration", "gpu_util", "vmem_usage")
    return [{k: row.get(k) for k in fields} for row in json.loads(result.stdout) if row.get("label") == label]


def target(rental):
    if os.environ.get("PROBE_SSH"):
        return os.environ["PROBE_SSH"]
    label = f"{json.loads((HERE / 'program.json').read_text())['program']}_{rental}"
    out = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True, check=True).stdout
    ids = [i["id"] for i in json.loads(out) if i.get("label") == label]
    assert len(ids) == 1, f"{len(ids)} live instances labelled {label}"
    return subprocess.run(["vastai", "ssh-url", str(ids[0])], capture_output=True, text=True, check=True).stdout.strip()


def ssh(url, remote, **kw):
    u = urllib.parse.urlparse(url)
    return subprocess.run(["ssh", "-p", str(u.port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                           "-o", "LogLevel=ERROR", "-o", "ControlMaster=auto", "-o", f"ControlPath={CTRL}",
                           "-o", "ControlPersist=600", f"{u.username}@{u.hostname}", remote], **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rental", default="neuronshow_probe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send"); s.add_argument("request"); s.add_argument("--timeout", type=int, default=600)
    s = sub.add_parser("log"); s.add_argument("-n", type=int, default=50)
    s = sub.add_parser("deploy"); s.add_argument("--ref", required=True); s.add_argument("--server", required=True)
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("url")
    s = sub.add_parser("box"); s.add_argument("rid"); s.add_argument("timeout", type=int)
    a = p.parse_args()

    if a.cmd == "box":
        return box(a.rid, a.timeout)
    if a.cmd == "status":
        print(json.dumps(status(a.rental), indent=2))
        return
    t0 = time.time()
    url = target(a.rental)
    if a.cmd == "url":
        print(url)
    elif a.cmd == "send":
        body = open(a.request, "rb").read()
        json.loads(body)
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}_" + \
            re.sub(r"[^A-Za-z0-9_-]", "_", pathlib.Path(a.request).stem)
        t1 = time.time()
        r = ssh(url, f"python3 /app/infra/vast/probe.py box {rid} {a.timeout}", input=body, capture_output=True)
        sys.stdout.write(r.stdout.decode())
        sys.stderr.write(r.stderr.decode())
        sys.stderr.write(f"[probe] {rid}: lookup {t1 - t0:.2f} s, round trip {time.time() - t1:.2f} s, exit {r.returncode}\n")
        sys.exit(r.returncode)
    elif a.cmd == "log":
        ssh(url, f"tail -n {a.n} {BOX}/server.log; cat {BOX}/server_exit.json 2>/dev/null; true", check=True)
    elif a.cmd == "stop":
        ssh(url, f"touch {BOX}/stop", check=True)
        print("[probe] stop requested: the station banks its files and the instance destroys itself")
    elif a.cmd == "deploy":
        root = HERE.parent.parent
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", a.ref], capture_output=True, text=True,
                             check=True).stdout.strip()
        paths = [d for d in ("infra", "configs", "scripts")
                 if subprocess.run(["git", "-C", root, "cat-file", "-e", f"{a.ref}:{d}"]).returncode == 0]
        tar = subprocess.run(["git", "-C", root, "archive", "--format=tar.gz", a.ref, *paths], capture_output=True,
                             check=True).stdout
        ssh(url, f"tar xzf - -C /app && printf %s {shlex.quote(a.server)} > {BOX}/server_cmd && "
                 f"echo {sha} > {BOX}/server_sha && touch {BOX}/restart", input=tar, check=True)
        print(f"[probe] deployed {a.ref} = {sha} ({len(tar)} bytes, {' '.join(paths)}) into /app; "
              f"the station restarts the server within 1 s as: {a.server}")


if __name__ == "__main__":
    main()
