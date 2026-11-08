"""The liveness classifier, exercised on the two cases that actually cost this program time.

Both incidents had a job.log that was byte-identical for hours. In M-D97 that read as a stall and a
healthy census was killed 11/16 through its base. In the recensus7 case it read as a wedge and a
redundant hedge was launched; the job finished on its own. In BOTH the process was working — one
computing, one downloading — and the log simply had nothing to say.

These tests pin the property that makes the sensor worth having: a FROZEN LOG must never by itself
produce a not-working verdict. `log_bytes` is held constant in every case below.
"""
from infra.aws_pg.pg import classify_heartbeats


def hb(utc, cpu=0, rx=0, rb=0, wb=0, log=10311):
    return {"utc": utc, "pid": "123", "cpu_seconds": cpu, "rss_kb": 4096,
            "read_bytes": rb, "write_bytes": wb, "net_rx_bytes": rx, "net_tx_bytes": 0,
            "log_bytes": log}


def test_downloading_with_a_frozen_log():
    """recensus7: >2.5h of byte-identical log while shards streamed in. Diagnosed as wedged; it
    was downloading and finished on its own."""
    r = classify_heartbeats([hb("T0", rx=1_000_000), hb("T1", rx=1_400_000)])
    assert r["state"] == "DOWNLOADING", r
    assert r["delta"]["net_rx_bytes"] == 400_000


def test_computing_with_a_frozen_log():
    """M-D97 / the wide row-Grams: a 16384-wide eigendecomposition prints nothing for 35 minutes."""
    r = classify_heartbeats([hb("T0", cpu=1200), hb("T1", cpu=1230)])
    assert r["state"] == "COMPUTING", r
    assert r["delta"]["cpu_seconds"] == 30


def test_cpu_wins_over_network():
    """A job that both computes and downloads is COMPUTING — the strongest evidence reported."""
    r = classify_heartbeats([hb("T0", cpu=10, rx=1), hb("T1", cpu=11, rx=99)])
    assert r["state"] == "COMPUTING", r


def test_io_only():
    r = classify_heartbeats([hb("T0", wb=5), hb("T1", wb=9)])
    assert r["state"] == "IO", r


def test_genuinely_stuck():
    """Nothing moves anywhere: the ONLY configuration that may read as not-working."""
    r = classify_heartbeats([hb("T0", cpu=99, rx=5, rb=5, wb=5), hb("T1", cpu=99, rx=5, rb=5, wb=5)])
    assert r["state"] == "NO-PROGRESS", r


def test_one_sample_is_not_a_verdict():
    """A single heartbeat has no delta. Reporting NO-PROGRESS from it would recreate the original
    bug at a 30-second timescale."""
    assert classify_heartbeats([hb("T0")])["state"] == "UNKNOWN"
    assert classify_heartbeats([])["state"] == "UNKNOWN"


def test_a_frozen_log_alone_never_means_stalled():
    """The property under test, stated directly: log_bytes is constant in every case above, and
    three of them are working states."""
    working = 0
    for rows in ([hb("T0", rx=1), hb("T1", rx=2)],
                 [hb("T0", cpu=1), hb("T1", cpu=2)],
                 [hb("T0", rb=1), hb("T1", rb=2)]):
        assert rows[0]["log_bytes"] == rows[1]["log_bytes"]
        if classify_heartbeats(rows)["state"] != "NO-PROGRESS":
            working += 1
    assert working == 3


def test_unresolved_pid_is_unknown_not_stuck():
    """The fail-open case. The first heartbeat matched the `timeout` wrapper instead of the
    interpreter (both command lines contain the venv python path, and `head -1` takes the parent),
    so every per-process counter read 0. A compute-bound job with no network — the M-D97 case —
    would then classify NO-PROGRESS and `pg kill` would have terminated the exact job the guard
    exists to protect. Unmeasured must never present as measured-zero."""
    rows = [hb("T0"), hb("T1")]
    for r in rows:
        r["pid"] = "none"
    r = classify_heartbeats(rows)
    assert r["state"] == "UNKNOWN", r
    assert "pid" in r["why"]


def test_compute_bound_with_no_network_is_not_stuck():
    """M-D97 restated as a test: CPU climbing, network flat, log frozen."""
    r = classify_heartbeats([hb("T0", cpu=100, rx=7), hb("T1", cpu=131, rx=7)])
    assert r["state"] == "COMPUTING", r
    assert r["delta"]["net_rx_bytes"] == 0
