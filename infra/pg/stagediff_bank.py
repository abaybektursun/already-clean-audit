"""STAGEDIFF artifact banking — spec §10.

One `Bank` per job: `out/metrics.json` written incrementally (temp + `os.replace`) after every readout,
`out/per_item/<name>.json`, `out/samples/<readout>__<arm>.jsonl`, `out/events.jsonl`, `out/manifest.json`.
Sealed (confirmation-half) numbers live under `metrics["sealed"]`; the jobs never print them. Fail loud,
no try/except.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


class Bank:
    def __init__(self, out_dir: str, job: str, args: dict, resume: bool = False):
        self.out = out_dir
        for d in ("", "samples", "per_item"):
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        path = os.path.join(out_dir, "metrics.json")
        if resume and os.path.exists(path):
            self.metrics = json.load(open(path))
            assert self.metrics.get("job") == job, (self.metrics.get("job"), job)
            self.metrics.setdefault("stages", []).append({"args": args, "utc": utc_now()})
        else:
            self.metrics = {"job": job, "args": args, "started_utc": utc_now(), "readouts": {}, "sealed": {},
                            "controls": {}, "timing_s": {}, "void": {}}
        self.write()
        self.event("start" if not resume else "resume", job=job, args=args)

    # ---- metrics -------------------------------------------------------------------------
    def write(self):
        self.metrics["updated_utc"] = utc_now()
        _write_json(os.path.join(self.out, "metrics.json"), self.metrics)

    def put(self, section: str, key: str, value):
        """metrics[section][key] = value, persisted. section ∈ readouts | sealed | controls | timing_s | void."""
        assert section in self.metrics and isinstance(self.metrics[section], dict), section
        self.metrics[section][key] = value
        self.write()

    def top(self, key: str, value):
        self.metrics[key] = value
        self.write()

    def stamp_void(self, readout: str, reason: str):
        self.metrics["void"][readout] = {"reason": reason, "utc": utc_now()}
        self.write()
        self.event("VOID", readout=readout, reason=reason)

    # ---- per item / samples / events ------------------------------------------------------
    def per_item(self, name: str, obj):
        _write_json(os.path.join(self.out, "per_item", f"{name}.json"), obj)

    def sample(self, readout: str, arm: str, rec: dict):
        rec = {"utc": utc_now(), "readout": readout, "arm": arm, **rec}
        with open(os.path.join(self.out, "samples", f"{readout}__{arm}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def event(self, kind: str, **kw):
        with open(os.path.join(self.out, "events.jsonl"), "a") as fh:
            fh.write(json.dumps({"utc": utc_now(), "kind": kind, **kw}, default=str) + "\n")
        print(f"[{kind}] " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    def manifest(self, extra: dict):
        m = {"utc": utc_now(), "python": sys.version.split()[0], "platform": platform.platform(),
             "git_sha": os.environ.get("PG_GIT_SHA"), "s3_prefix": os.environ.get("PG_S3_PREFIX"),
             "harness_shift": os.environ.get("MDL_HARNESS_SHIFT"), **extra}
        for mod in ("torch", "transformers", "numpy", "datasets"):
            try:
                m[f"{mod}_version"] = __import__("importlib.metadata").metadata.version(mod)
            except Exception as e:                       # noqa: BLE001 — a missing optional pin is data, recorded as such
                m[f"{mod}_version"] = f"unavailable ({type(e).__name__})"
        m["gpu"] = None
        if shutil.which("nvidia-smi"):
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True)
            m["gpu"] = smi.stdout.strip() if smi.returncode == 0 else None
        _write_json(os.path.join(self.out, "manifest.json"), m)
        return m


def _dryrun():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        b = Bank(d, "dry", {"x": 1})
        b.put("readouts", "B1-beh", {"acc": 0.5})
        b.put("sealed", "B1-beh", {"acc": 0.4})
        b.sample("B1-beh", "clean", {"hash": "h", "correct": True})
        b.per_item("B1-beh_clean", {"h": 1})
        b.stamp_void("I3", "positive control did not fire")
        m = json.load(open(os.path.join(d, "metrics.json")))
        assert m["readouts"]["B1-beh"]["acc"] == 0.5 and m["void"]["I3"]["reason"]
        assert os.path.exists(os.path.join(d, "samples", "B1-beh__clean.jsonl"))
        assert len(open(os.path.join(d, "events.jsonl")).read().splitlines()) == 2
        b.manifest({"repo": "x"})
        assert json.load(open(os.path.join(d, "manifest.json")))["repo"] == "x"
    print("PASS stagediff_bank dry-run")


if __name__ == "__main__":
    _dryrun()
