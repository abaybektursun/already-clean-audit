"""DEPTHOFF offline gate — the whole pipeline minus the model forward, green before any GPU.

CLAUDE.md "Before any compute" item 4. What this exercises with synthetic rollout rows:

  * the pairing rule: an episode counts only if EVERY paired condition produced it, so a job killed
    mid-episode contributes whole pairs or nothing
  * BROKEN handled as its own unpaired sample
  * latency and realized-scalar aggregation
  * the frozen rules wired to the summary, on planted outcomes that force no_loss AND loss
  * the rules' own 21 planted branches

    .pgvenv/bin/python -m infra.aws_pg.jobs.depthoff_dryrun
"""
from __future__ import annotations

import json
from typing import Dict, List

import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))

from infra.aws_pg.jobs import depthoff_loop as job
from infra.pg import depthoff_rules as rules

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import depthoff_report as report  # noqa: E402


def _row(cond: str, task: int, ep: int, success: bool, *, seconds: float = 45.0, calls: int = 30,
         gate: float | None = 0.018, suite: str = "libero_object") -> Dict[str, object]:
    return {"suite": suite, "task_id": task, "episode": ep, "arm": cond, "condition": cond,
            "pair_id": f"{suite}/{task}/ep{ep}", "episode_index_used": ep,
            "success": success, "n_steps": 120, "seconds": seconds,
            "realized_gate_mean": gate, "meta": {"calls": calls}}


def build_rows(*, n_tasks: int = 10, n_eps: int = 5, zero_flips: int = 0, nogen_flips: int = 30,
               on2_flips: int = 2, drop_last_pair_for: str | None = None,
               suite: str = "libero_object", with_broken: bool = True) -> List[Dict[str, object]]:
    """Synthetic rows: ON succeeds everywhere except a fixed handful, and each other condition flips a
    chosen number of episodes against it. `drop_last_pair_for` simulates a job killed mid-episode."""
    rows: List[Dict[str, object]] = []
    pairs = [(t, e) for t in range(n_tasks) for e in range(n_eps)]
    n_on_fail = max(0, round(0.08 * len(pairs)))   # ON itself misses ~8%, like the released policy
    on_fail = set(pairs[-n_on_fail:]) if n_on_fail else set()
    flips = {"ON2": set(pairs[:on2_flips]),
             "ZERO": set(pairs[:zero_flips]),
             "NOGEN": set(pairs[:nogen_flips])}
    for i, (t, e) in enumerate(pairs):
        for cond in job.CONDITIONS:
            if drop_last_pair_for == cond and (t, e) == pairs[-1]:
                continue                            # the pair this condition never got to
            base = (t, e) not in on_fail
            success = (not base) if (cond != "ON" and (t, e) in flips[cond]) else base
            gate = 0.0 if cond == "ZERO" else (None if cond == "NOGEN" else 0.018)
            secs = 30.0 if cond == "NOGEN" else 45.0
            calls = 30
            rows.append(_row(cond, t, e, success, seconds=secs, calls=calls, gate=gate,
                             suite=suite))
    # the positive control: its own unpaired sample, 1 of 10
    if with_broken:
        for t in range(10):
            rows.append(_row("BROKEN", t, 0, t == 0, gate=0.018, suite=suite))
    return rows


def main() -> int:
    res: Dict[str, object] = {}
    failures: List[str] = []

    def check(name: str, got, want) -> None:
        res[name] = got
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # ---- 1. the rules' own planted branches
    st = rules.selftest()
    check("rules selftest ok", st["ok"], True)
    if not st["ok"]:
        failures.append(f"rules selftest failures: {st['FAILURES']}")

    # ---- 2. ONE SUITE CANNOT DECIDE THE MARGIN. 50 pairs with PERFECT agreement still leaves a 7.1-point
    # bound (Wilson upper on 0 of 50), wider than the frozen 5. A per-suite verdict must read
    # `underpowered` so nobody quotes a single job as the answer.
    s = job.summarize(build_rows(zero_flips=0, nogen_flips=30), suite="libero_object")
    check("one suite paired episodes", s["n_paired_episodes"], 50)
    check("ON rate", s["rates"]["ON"]["rate"], 0.92)
    check("one suite cannot decide", s["decided"]["headline"]["ZERO"], "underpowered")
    check("one suite still catches a big loss", s["decided"]["headline"]["NOGEN"], "loss")
    check("G-instrument", s["decided"]["gates"]["G-instrument"]["verdict"], "pass")
    check("G-control", s["decided"]["gates"]["G-control"]["verdict"], "pass")
    check("noise floor discordant", s["decided"]["gates"]["G-floor"]["n_discordant"], 2)
    check("ZERO realized scalar", s["realized_gate_median"]["ZERO"], 0.0)
    check("NOGEN realized scalar", s["realized_gate_median"]["NOGEN"], None)
    check("NOGEN faster", s["latency"]["NOGEN"]["median_seconds_per_episode"]
          < s["latency"]["ON"]["median_seconds_per_episode"], True)

    # ---- 2b. THE POOLED PATH, which is the one that decides. Four suites x 50 = 200 pairs.
    pooled_rows: List[Dict[str, object]] = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        pooled_rows += build_rows(zero_flips=0, nogen_flips=30, suite=suite,
                                  with_broken=(suite == "libero_object"))
    paired = report.pooled_outcomes(pooled_rows)
    check("pooled paired episodes", len(paired["pair_ids"]), 200)
    decided = rules.decide(paired["outcomes"])
    check("pooled ZERO verdict", decided["headline"]["ZERO"], "no_loss")
    check("pooled NOGEN verdict", decided["headline"]["NOGEN"], "loss")
    lat = report.latency(pooled_rows)
    check("depth cost measured", lat["_depth_cost"]["multiple"], 1.5)
    r5 = report.steps_to_success(pooled_rows)
    check("steps-to-success computed", r5["ZERO"]["n_both_succeed"] > 0, True)

    # ---- 3. the pairing rule: a condition missing the last episode drops that pair from EVERY condition
    s2 = job.summarize(build_rows(drop_last_pair_for="NOGEN"), suite="libero_object")
    check("pairing drops the incomplete episode", s2["n_paired_episodes"], 49)
    check("every condition same length",
          len({len(v) for k, v in s2["decided"]["rates"].items() if k != "BROKEN"}), 1)

    # ---- 4. a broken instrument cancels every verdict, however clean the arms look
    bad = [r for r in build_rows(zero_flips=0)
           if not (r["condition"] == "ON" and r["task_id"] < 5)]
    bad += [_row("ON", t, e, False) for t in range(5) for e in range(5)]
    s3 = job.summarize(bad, suite="libero_object")
    check("broken instrument cancels", s3["decided"]["headline"]["ZERO"], "cancelled_instrument")

    # ---- 5. an insensitive positive control cancels every verdict too
    rows = [r for r in build_rows(zero_flips=0) if r["condition"] != "BROKEN"]
    rows += [_row("BROKEN", t, 0, True) for t in range(10)]     # the control did NOT collapse
    s4 = job.summarize(rows, suite="libero_object")
    check("insensitive control cancels", s4["decided"]["headline"]["ZERO"], "cancelled_control")

    # ---- 6. too few episodes must read underpowered, never no_loss
    s5 = job.summarize(build_rows(n_tasks=10, n_eps=1, zero_flips=0, nogen_flips=0, on2_flips=0),
                       suite="libero_object")
    check("small n is underpowered", s5["decided"]["headline"]["ZERO"], "underpowered")
    check("small n is not no_loss", s5["decided"]["headline"]["ZERO"] != "no_loss", True)

    # ---- 7. a partial job (no ON2 yet) says so instead of deciding
    only_on = [r for r in build_rows() if r["condition"] in ("ON", "BROKEN")]
    s6 = job.summarize(only_on, suite="libero_object")
    check("partial job not decidable", s6["decided"]["verdict"], "not_decidable_yet")

    res["FAILURES"] = failures
    res["ok"] = not failures
    print(json.dumps(res, indent=1, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
