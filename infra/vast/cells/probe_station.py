"""Probe station cell: supervise one long-lived server process and return after `idle_min` minutes without a
request, so run_cells.py banks the session and lifecycle.sh destroys the instance (the trap, the watchdog, the
manifest watch bounds and the Lambda reaper stay exactly as for any other rental).

Manifest cell: {"id": "station", "module": "infra.vast.cells.probe_station",
                "server": "python3 -m infra.vast.probe_echo", "idle_min": 30}

Directory protocol. /workspace/probe is a symlink to this cell's out_dir, so every file below is copied to S3
live_out/<id>/ by the 60 s onstart streamer and banked under cells/<id>/ when the station ends.
  requests/<rid>.json   written by infra/vast/probe.py (a .tmp file, then rename: a reader never sees half a file)
  responses/<rid>.json  written by the server, same name, also .tmp + rename; the server never deletes requests
  server.log            the server's stdout and stderr; server_exit.json is written here when it exits
  server_cmd/server_sha written by `probe.py deploy`: they replace the manifest's command and set PG_GIT_SHA
  restart / stop        touched by `probe.py deploy` / `probe.py stop`
The server is started with PROBE_DIR=/workspace/probe and the box environment (HF_TOKEN, HF_HOME, PYTHONPATH=/app).

Idle: the newest request file (or the station start, or the last deploy) is older than idle_min. A server that
exits does not end the station: server_exit.json records the code, probe.py prints it with the log tail, a fixed
server can be deployed, and the idle timeout still ends the rental.
"""
import json
import os
import pathlib
import shlex
import subprocess
import time

PROBE = pathlib.Path("/workspace/probe")


def utc():
    return time.strftime("%FT%TZ", time.gmtime())


def log(msg):
    print(f"[probe_station] {utc()} {msg}", flush=True)


def start(d, cell):
    cmd = (d / "server_cmd").read_text().strip() if (d / "server_cmd").exists() else cell["server"]
    env = dict(os.environ, PROBE_DIR=str(PROBE))
    if (d / "server_sha").exists():
        env["PG_GIT_SHA"] = (d / "server_sha").read_text().strip()
    (d / "server_exit.json").unlink(missing_ok=True)
    log(f"server start: {cmd} (PG_GIT_SHA={env.get('PG_GIT_SHA')})")
    return subprocess.Popen(shlex.split(cmd), stdout=open(d / "server.log", "a"), stderr=subprocess.STDOUT, env=env)


def halt(proc):
    if proc.poll() is None:
        proc.terminate()
        for _ in range(30):
            if proc.poll() is not None:
                return
            time.sleep(1)
        proc.kill()
        proc.wait()


def run(cell, out_dir):
    d = pathlib.Path(out_dir)
    (d / "requests").mkdir(parents=True, exist_ok=True)
    (d / "responses").mkdir(exist_ok=True)
    if PROBE.is_symlink():
        PROBE.unlink()
    PROBE.symlink_to(d)
    idle_s = 60 * cell["idle_min"]
    t_start = time.time()
    proc = start(d, cell)
    reported = False
    while True:
        if (d / "restart").exists():
            (d / "restart").unlink()
            halt(proc)
            proc, reported = start(d, cell), False
        if proc.poll() is not None and not reported:
            json.dump({"code": proc.returncode, "utc": utc()}, open(d / "server_exit.json", "w"))
            log(f"server exited with code {proc.returncode}; see server.log; the station stays up until idle")
            reported = True
        if (d / "stop").exists():
            log("stop requested")
            break
        marks = [t_start] + [p.stat().st_mtime for p in (d / "requests").glob("*.json")]
        marks += [(d / f).stat().st_mtime for f in ("server_cmd",) if (d / f).exists()]
        idle = time.time() - max(marks)
        if idle > idle_s:
            log(f"idle {idle / 60:.1f} min > idle_min {cell['idle_min']}")
            break
        time.sleep(1)
    halt(proc)
    n_req = len(list((d / "requests").glob("*.json")))
    n_res = len(list((d / "responses").glob("*.json")))
    log(f"station ends after {(time.time() - t_start) / 60:.1f} min: {n_req} requests, {n_res} responses")
