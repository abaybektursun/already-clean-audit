#!/usr/bin/env python3
"""Tracked deterministic Vast CLI subset with real-shaped offline responses."""

from __future__ import annotations

import fcntl
import json
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".pg1-fake-services"
        if candidate.is_dir():
            return candidate
    raise SystemExit("tracked fake Vast service root absent")


ROOT = _root(); STATE = ROOT / "vast-state.json"; LOCK = ROOT / "vast.lock"


def _load() -> dict[str, Any]:
    try: return json.loads(STATE.read_text(encoding="utf-8"))
    except FileNotFoundError: return {"instances": {}}


def _save(value: dict[str, Any]) -> None:
    STATE.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _instance_id() -> str:
    for value in sys.argv:
        if value.isdigit() and int(value) > 700000:
            return value
    return "778899"


def main() -> int:
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); state = _load()
        if sys.argv[1:3] == ["create", "instance"]:
            offer = int(sys.argv[3]); label = sys.argv[sys.argv.index("--label") + 1]
            onstart = sys.argv[sys.argv.index("--onstart-cmd") + 1]
            row = {"id": 778899, "machine_id": 771133,
                   "container_id": "future-container-abc", "label": label,
                   "actual_status": "running", "offer_id": offer,
                   "onstart_cmd": onstart}
            stdout_path = ROOT / "onstart.stdout"
            stderr_path = ROOT / "onstart.stderr"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    onstart, shell=True, executable="/bin/sh", stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True,
                )
            row["onstart_pid"] = process.pid
            state["instances"]["778899"] = row; _save(state)
            print(json.dumps({"success": True, "new_contract": 778899,
                              "instance_api_key": "1" * 64}, sort_keys=True)); return 0
        if sys.argv[1:3] == ["show", "instances"]:
            print(json.dumps(list(state["instances"].values()), sort_keys=True)); return 0
        if sys.argv[1:3] == ["show", "instance"]:
            row = state["instances"].get(_instance_id())
            print(json.dumps([] if row is None else [row], sort_keys=True)); return 0
        if sys.argv[1] == "copy":
            source = Path(sys.argv[2]); instance, target_text = sys.argv[3].split(":", 1)
            if instance not in state["instances"]: return 2
            worker = ROOT / "worker"
            target = worker / Path(target_text).relative_to("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); return 0
        if sys.argv[1:3] in (["stop", "instance"], ["destroy", "instance"]):
            instance = _instance_id()
            if sys.argv[1] == "stop" and instance in state["instances"]:
                state["instances"][instance]["actual_status"] = "stopped"
            elif sys.argv[1] == "destroy":
                state["instances"].pop(instance, None)
            _save(state); print("{}") ; return 0
        if sys.argv[1] == "rest" and sys.argv[2] in {"stop", "destroy"}:
            instance = sys.argv[3]
            if sys.argv[2] == "stop" and instance in state["instances"]:
                state["instances"][instance]["actual_status"] = "stopped"
            elif sys.argv[2] == "destroy":
                state["instances"].pop(instance, None)
            _save(state)
            print(json.dumps({"channel": "REST", "action": sys.argv[2],
                              "http_status": 200}, sort_keys=True)); return 0
    raise SystemExit(f"fake Vast argv rejected: {sys.argv[1:]}")


if __name__ == "__main__":
    raise SystemExit(main())
