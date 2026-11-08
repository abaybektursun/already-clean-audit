"""Table tests for policy.py — each row is a scenario that must never be
mis-handled. Run: python3 infra/vast/test_policy.py (no pytest needed)."""

from policy import (backstop_decide, budget_action, decide, loading_doomed,
                    parse_layer_progress, should_respawn)

CASES = [
    # (name, obs, want_action, want_classify)
    ("genuine 20h training run survives",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 20.0, "liveness_age_s": 30, "output_stall_min": 2},
     "none", None),

    ("quiet-but-computing cell only warns (no author kill opt-in)",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 3.0, "liveness_age_s": 30, "output_stall_min": 45},
     "warn", None),

    ("author-declared progress kill fires",
     {"phase": "running", "declared": {"max_billed_h": 24,
                                       "progress_kill_min": 120},
      "run_elapsed_h": 5.0, "liveness_age_s": 30, "output_stall_min": 130},
     "kill", "infra"),

    ("author-declared progress_stall_min raises the default warn bar",
     {"phase": "running", "declared": {"max_billed_h": 24,
                                       "progress_stall_min": 90},
      "run_elapsed_h": 5.0, "liveness_age_s": 30, "output_stall_min": 45},
     "none", None),

    ("slow-but-finishing pull survives",
     # 2 GB/min on a 40 GB layer: finishes in ~14 min, well inside backstop
     {"phase": "loading", "declared": {"load_backstop_min": 120},
      "load_elapsed_min": 60, "same_fp_polls": 0,
      "pull_prog": (10.0, 12.0, 1.0, 40.0)},
     "none", None),

    ("doomed pull early-killed under its own backstop",
     # 0.05 GB/min with 138 GB of the layer left: certain backstop miss
     {"phase": "loading", "declared": {"load_backstop_min": 60},
      "load_elapsed_min": 30, "same_fp_polls": 0,
      "pull_prog": (1.0, 1.05, 1.0, 140.0)},
     "kill", "infra"),

    ("new-layer counter reset is never treated as doom",
     {"phase": "loading", "declared": {"load_backstop_min": 60},
      "load_elapsed_min": 30, "same_fp_polls": 0,
      "pull_prog": (39.0, 0.5, 1.0, 140.0)},
     "none", None),

    ("cold start is never projected",
     {"phase": "loading", "declared": {"load_backstop_min": 60},
      "load_elapsed_min": 5, "same_fp_polls": 0,
      "pull_prog": (0.0, 0.01, 1.0, 140.0)},
     "none", None),

    ("wedged load (frozen fingerprint) killed",
     {"phase": "loading", "declared": {"load_backstop_min": 120},
      "load_elapsed_min": 40, "same_fp_polls": 25},
     "kill", "infra"),

    ("early fingerprint freeze only warns (grouped-layer extraction, D15)",
     {"phase": "loading", "declared": {"load_backstop_min": 120},
      "load_elapsed_min": 12, "same_fp_polls": 6},
     "warn", None),

    ("declared load backstop exceeded",
     {"phase": "loading", "declared": {"load_backstop_min": 90},
      "load_elapsed_min": 95, "same_fp_polls": 0},
     "kill", "infra"),

    ("no declared bounds -> liveness-only, long load survives",
     {"phase": "loading", "declared": {},
      "load_elapsed_min": 500, "same_fp_polls": 2},
     "none", None),

    ("no declared bounds -> liveness-only, long run survives",
     {"phase": "running", "declared": {},
      "run_elapsed_h": 100.0, "liveness_age_s": 30, "output_stall_min": 5},
     "none", None),

    ("science crash: killed, classified science",
     {"phase": "running", "has_joblog": True, "traceback": True},
     "kill", "science"),

    ("lifecycle died without traceback: killed, classified infra",
     {"phase": "running", "has_joblog": True, "traceback": False},
     "kill", "infra"),

    ("complete marker: cleanup kill",
     {"phase": "running", "has_complete": True},
     "kill", "complete"),

    ("net probe abort (exit 9) is infra: respawn on another host, never DONE",
     {"phase": "running", "has_complete": True, "exit_code": 9},
     "kill", "infra"),

    ("complete marker with a normal exit code stays complete",
     {"phase": "running", "has_complete": True, "exit_code": 0},
     "kill", "complete"),

    ("dead streamer killed",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 2.0, "liveness_age_s": 500},
     "kill", "infra"),

    ("stale-ish streamer warns",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 2.0, "liveness_age_s": 300, "output_stall_min": 1},
     "warn", None),

    ("streamer never seen early in running: no false kill",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 0.05, "liveness_age_s": None},
     "none", None),

    ("streamer never came up at all: killed via running-start clock",
     {"phase": "running", "declared": {"max_billed_h": 24},
      "run_elapsed_h": 0.2, "liveness_age_s": None},
     "kill", "infra"),

    ("unknown-phase wedge bounded by declared contract",
     {"phase": "offline", "declared": {"max_billed_h": 1,
                                       "load_backstop_min": 20},
      "load_elapsed_min": 120},
     "kill", "infra"),

    ("unknown phase inside declared contract survives",
     {"phase": "offline", "declared": {"max_billed_h": 1,
                                       "load_backstop_min": 20},
      "load_elapsed_min": 60},
     "none", None),

    ("declared ceiling exceeded: killed, classified ceiling (no respawn)",
     {"phase": "running", "declared": {"max_billed_h": 6},
      "run_elapsed_h": 6.1, "liveness_age_s": 30},
     "kill", "ceiling"),

    ("90% of ceiling warns",
     {"phase": "running", "declared": {"max_billed_h": 6},
      "run_elapsed_h": 5.5, "liveness_age_s": 30, "output_stall_min": 1},
     "warn", None),
]


def run():
    fails = 0
    for name, obs, want_action, want_classify in CASES:
        got = decide(obs)
        ok = got["action"] == want_action and got["classify"] == want_classify
        if not ok:
            fails += 1
            print(f"FAIL {name}: want ({want_action},{want_classify}) "
                  f"got ({got['action']},{got['classify']}) — {got['reason']}")
    # budget: hard only at the ceiling; soft on projection; quiet below
    assert budget_action(50.0, 0.0, 50.0) == "hard"
    assert budget_action(39.5, 1.0, 50.0) == "soft"
    assert budget_action(10.0, 2.0, 50.0) is None
    # respawn: infra only, bounded; science and ceiling never
    assert should_respawn("infra", 0, 3) and should_respawn("infra", 2, 3)
    assert not should_respawn("infra", 3, 3)
    assert not should_respawn("science", 0, 3)
    assert not should_respawn("ceiling", 0, 3)
    assert not should_respawn("complete", 0, 3)
    # server-side backstop: generous, lifetime-based, never kills a genuine
    # job inside its declared contract
    W = {"load_backstop_min": 90, "max_billed_h": 8}
    bs = lambda **kw: backstop_decide(**{
        "phase": "running", "age_min": 0, "declared": W, "has_complete": False,
        "has_joblog": False, "has_manifest": True, **kw})
    # genuine 8h run: lifetime 500min < (90+480)*1.2=684 -> survives
    assert bs(age_min=500, liveness_age_s=60)["action"] == "none"
    # wedge the laptop never saw: loading 140min > 90*1.5 -> reaped
    assert bs(phase="loading", age_min=140)["action"] == "kill"
    # loading within margin survives (laptop layer handles precision)
    assert bs(phase="loading", age_min=120)["action"] == "none"
    # lifetime blowout past declared total -> reaped as ceiling
    assert bs(age_min=700)["classify"] == "ceiling"
    # dead machinery: live upload stale 40min while "running" -> reaped
    assert bs(age_min=100, liveness_age_s=2400)["action"] == "kill"
    # asleep-laptop cleanup on markers
    assert bs(age_min=5, has_complete=True)["classify"] == "complete"
    assert bs(age_min=5, has_joblog=True)["classify"] == "infra"
    # foreign labeled instance without banked manifest: grace then reap
    assert bs(age_min=10, has_manifest=False)["action"] == "none"
    assert bs(age_min=40, has_manifest=False)["action"] == "kill"
    # no declared bounds + no markers: backstop never time-kills
    assert bs(age_min=10000, declared={}, liveness_age_s=60)["action"] == "none"
    # progress parser
    assert parse_layer_progress("pulling fs layer 12.3GB/40GB") == (12.3, 40.0)
    assert parse_layer_progress("extracting 512 MB / 2 GB extra")[0] == 0.512
    assert parse_layer_progress("waiting") is None
    assert parse_layer_progress(None) is None
    # doom projection is a lower bound: generous rate -> not doomed
    assert not loading_doomed(30, 60, 10.0, 20.0, 1.0, 140.0)
    print(f"{'OK' if fails == 0 else 'FAILED'}: "
          f"{len(CASES) - fails}/{len(CASES)} scenarios + invariants")
    assert fails == 0


if __name__ == "__main__":
    run()
