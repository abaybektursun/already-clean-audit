"""N13 — 100 % of the hardware (spec §11.5): batch-size probe + `nvidia-smi dmon` sampler.

choose_batch(fwd, prompts, start=4, max_bs=64): double the shell's batch until CUDA OOM (ONLY
torch.cuda.OutOfMemoryError is caught — anything else raises; on OOM the cache is freed and the
last good batch is kept) or until the throughput gain over the previous batch is < 10 %.
Graft evaluations stay at the fixed batch 4 (numerics); everything else runs at the probed batch.

DmonSampler: runs `nvidia-smi dmon -s u -d 30` in a subprocess and appends {t, sm, mem} per
sample to `out/events.jsonl`; `mean_sm_last(minutes)` is the §11.5 rule input (mean SM-active
< 60 % over any 10-minute window → finish the stage, stop, do not relaunch until fixed + journaled).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

MIN_GAIN = 0.10


def _throughput(fwd, prompts, bs):
    """Prompts per second at batch `bs` (one warm-up chunk, then one timed pass over `prompts`)."""
    import torch
    fwd.batch_size = bs
    fwd.next_token_logits(prompts[:bs])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    fwd.next_token_logits(prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(prompts) / (time.time() - t0)


def choose_batch(fwd, prompts, start: int = 4, max_bs: int = 64) -> dict:
    """Returns {"batch": int, "trace": [{bs, prompts_per_s | oom}], "stop": reason} and leaves
    `fwd.batch_size` at the chosen value."""
    import torch
    prompts = list(prompts)
    assert len(prompts) >= start, f"need ≥ {start} prompts to probe, got {len(prompts)}"
    trace = []
    best_bs, best_tp, stop = None, 0.0, None
    bs = start
    while bs <= max_bs:
        try:
            tp = _throughput(fwd, prompts, bs)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trace.append({"bs": bs, "oom": True})
            stop = f"OOM at {bs}"
            break
        trace.append({"bs": bs, "prompts_per_s": tp})
        if best_bs is not None and tp < best_tp * (1 + MIN_GAIN):
            stop = f"gain < {MIN_GAIN:.0%} at {bs}"
            break
        best_bs, best_tp = bs, tp
        bs *= 2
    if stop is None:
        stop = f"max_bs {max_bs} reached"
    assert best_bs is not None, f"batch {start} does not fit: {trace}"
    fwd.batch_size = best_bs
    return {"batch": best_bs, "prompts_per_s": best_tp, "trace": trace, "stop": stop}


def parse_dmon_line(line: str):
    """`nvidia-smi dmon -s u` prints `# gpu sm mem enc dec ...` headers and `0 87 41 0 0` rows.
    Returns (sm, mem) as ints or None for header/blank/'-' rows."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) < 3 or parts[1] == "-" or parts[2] == "-":
        return None
    return int(parts[1]), int(parts[2])


class DmonSampler:
    """`nvidia-smi dmon -s u -d <interval>` → events.jsonl lines {"t", "sm", "mem", "kind": "dmon"}."""

    def __init__(self, events_path, interval_s: int = 30, cmd=None):
        self.events_path = str(events_path)
        self.interval_s = interval_s
        self.cmd = cmd or ["nvidia-smi", "dmon", "-s", "u", "-d", str(interval_s)]
        self.samples = []                      # (t, sm, mem)
        self._proc = None
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, bufsize=1)
        self._thread = threading.Thread(target=self._pump, args=(self._proc.stdout,), daemon=True)
        self._thread.start()
        return self

    def _pump(self, stream):
        for line in stream:
            self.ingest(line)

    def ingest(self, line: str, t: float | None = None):
        parsed = parse_dmon_line(line)
        if parsed is None:
            return
        sm, mem = parsed
        t = time.time() if t is None else t
        with self._lock:
            self.samples.append((t, sm, mem))
            with open(self.events_path, "a") as f:
                f.write(json.dumps({"kind": "dmon", "t": t, "sm": sm, "mem": mem}) + "\n")

    def mean_sm_last(self, minutes: float):
        """Mean SM-active over the trailing window; None when no sample falls inside it."""
        cutoff = time.time() - 60.0 * minutes
        with self._lock:
            xs = [sm for t, sm, _ in self.samples if t >= cutoff]
        return (sum(xs) / len(xs)) if xs else None

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


def log_event(events_path, **fields):
    """Append one structured line (stage / t / gpu_util ...) to events.jsonl."""
    rec = {"t": time.time(), **fields}
    with open(events_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _dryrun():
    import tempfile
    import torch

    class FakeFwd:
        """Throughput grows with batch, saturates at 16, OOMs at 64."""
        batch_size = 4
        calls = []

        def next_token_logits(self, prompts):
            bs = self.batch_size
            self.calls.append(bs)
            if bs >= 64:
                raise torch.cuda.OutOfMemoryError("fake OOM")
            time.sleep(0.002 * len(prompts) / min(bs, 16))
            return torch.zeros(len(prompts), 4)

    f = FakeFwd()
    r = choose_batch(f, [f"p{i}" for i in range(64)], start=4, max_bs=128)
    assert r["batch"] in (16, 32), r
    assert f.batch_size == r["batch"]
    assert all(("oom" in t) or ("prompts_per_s" in t) for t in r["trace"])

    class OomAt8(FakeFwd):
        def next_token_logits(self, prompts):
            if self.batch_size >= 8:
                raise torch.cuda.OutOfMemoryError("fake OOM")
            return torch.zeros(len(prompts), 4)
    r2 = choose_batch(OomAt8(), [f"p{i}" for i in range(16)], start=4)
    assert r2["batch"] == 4 and r2["stop"].startswith("OOM"), r2

    class Bad(FakeFwd):
        def next_token_logits(self, prompts):
            raise ValueError("not an OOM — must propagate")
    raised = False
    try:
        choose_batch(Bad(), [f"p{i}" for i in range(8)], start=4)
    except ValueError:
        raised = True
    assert raised, "non-OOM errors must propagate out of choose_batch"

    with tempfile.TemporaryDirectory() as d:
        ev = os.path.join(d, "events.jsonl")
        s = DmonSampler(ev, interval_s=30, cmd=["true"])      # never started; fed by hand
        now = time.time()
        for i, line in enumerate(["# gpu    sm   mem   enc   dec", "# Idx     %     %     %     %",
                                  "    0    80    40     0     0", "    0    -     -     -     -",
                                  "    0    40    20     0     0"]):
            s.ingest(line, t=now - 30 * (5 - i))
        assert len(s.samples) == 2, s.samples
        assert abs(s.mean_sm_last(10) - 60.0) < 1e-9
        assert s.mean_sm_last(0.0001) is None
        log_event(ev, stage="manifest", gpu_util=s.mean_sm_last(10))
        lines = [json.loads(l) for l in open(ev)]
        assert len(lines) == 3 and lines[0]["kind"] == "dmon" and lines[-1]["stage"] == "manifest"
        assert parse_dmon_line("") is None and parse_dmon_line("0 12 3 0 0") == (12, 3)
    print(f"PASS util_probe dry-run (batch={r['batch']} stop='{r['stop']}'; OOM backoff; dmon parse)")


if __name__ == "__main__":
    _dryrun()
