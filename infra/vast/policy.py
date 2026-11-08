"""Kill/warn/respawn/budget policy — the ONLY place a destroy decision is made.

Shared by watchdog.py (laptop layer: rich + fast) and reaper_lambda.py
(server-side backstop: laptop-independent). Pure functions, no I/O —
table-tested in test_policy.py against the scenarios that must never be
mis-handled (the genuine 20 h training run, the quiet-but-computing cell,
the slow-but-finishing pull).

The anti-blind-rule contract:
1. NO global wall-clock kill exists. Every time bound is DECLARED per rental
   by the job's author in the manifest "watch" block; rent refuses a manifest
   without one.
2. Liveness detectors are size- and duration-independent: they fire on ABSENT
   MACHINERY (dead streamer, dead lifecycle), never on "the job is slow".
3. An early kill is allowed only when the rental is already doomed under its
   OWN declared bound, computed from an UNDERESTIMATE of remaining work — a
   projected miss is then a certain miss, so this can never kill a rental
   that would have survived its backstop.
4. Output-progress stalls WARN by default; they KILL only if the author
   declared progress_kill_min. Silence alone never kills a computing job.
5. Respawn only on INFRA failures (wedge, host death, dead streamer) — never
   SCIENCE failures (traceback in job.log; a deterministic bug replays
   identically) and never CEILING kills (raising a bound is the author's
   call). Attempts are bounded.
"""

import re

# Machinery-liveness windows (size-independent by design — see contract #2).
LOAD_WARN_POLLS = 5      # identical pull fingerprint, ~1 poll/min
LOAD_KILL_POLLS = 25     # generous: grouped 40 GB layers freeze status_msg (D15)
LIVENESS_WARN_S = 240    # streamer re-uploads every 60 s regardless of content
LIVENESS_KILL_S = 480    # so staleness == instance/network dead, not "quiet job"
OUTPUT_STALL_WARN_MIN = 30   # default warn-only; kill requires author opt-in
CEILING_WARN_FRAC = 0.9
DOOM_MARGIN = 1.25       # projection must overshoot the backstop by 25 %
DOOM_MIN_OBS_MIN = 10    # never project from a cold start
NET_PROBE_EXIT = 9       # lifecycle.sh net probe < 10 MB/s (D21): the HOST is slow, not the job

_PROG_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([GM])i?B\s*/\s*(\d+(?:\.\d+)?)\s*([GM])i?B")


def parse_layer_progress(status_msg):
    """(done_gb, total_gb) from a docker-pull style 'x GB / y GB', else None."""
    m = _PROG_RE.search(status_msg or "")
    if not m:
        return None
    done = float(m.group(1)) * (1.0 if m.group(2) == "G" else 1e-3)
    total = float(m.group(3)) * (1.0 if m.group(4) == "G" else 1e-3)
    return (done, total)


def loading_doomed(elapsed_min, backstop_min, done0_gb, done1_gb,
                   dt_min, total_gb):
    """True only when finishing just the CURRENT layer already misses the
    declared backstop. Remaining work >= remaining current layer, so this is
    a lower bound — a True here is a certain backstop miss (contract #3)."""
    if backstop_min is None or elapsed_min < DOOM_MIN_OBS_MIN:
        return False
    if done1_gb <= done0_gb or dt_min <= 0:
        return False   # new layer started or no rate signal — not our call
    rate = (done1_gb - done0_gb) / dt_min
    finish_layer_min = (total_gb - done1_gb) / rate
    return elapsed_min + finish_layer_min > backstop_min * DOOM_MARGIN


def decide(obs):
    """obs -> {"action": none|warn|kill, "reason": str, "classify": ...}

    obs keys (all plain values; None = unknown):
      phase             "loading" | "running" | other/None
      has_complete      _rental_complete.json banked
      has_joblog        job.log banked (lifecycle finished or died)
      traceback         job.log contains a Python traceback
      declared          manifest watch block: max_billed_h, load_backstop_min,
                        progress_stall_min, progress_kill_min (any may be absent)
      load_elapsed_min  minutes since creation (loading phase)
      same_fp_polls     consecutive polls with identical pull fingerprint
      pull_prog         (done0_gb, done1_gb, dt_min, total_gb) or None
      run_elapsed_h     hours since running began
      liveness_age_s    age of the streamer's newest upload (None early on)
      output_stall_min  minutes since job output last advanced (None = unknown)
      exit_code         from _rental_complete.json when the lifecycle wrote one
                        (NET_PROBE_EXIT = the net probe aborted before any cell)
    """
    d = obs.get("declared") or {}
    phase = obs.get("phase")

    if obs.get("has_complete"):
        if obs.get("exit_code") == NET_PROBE_EXIT:
            return _kill("net probe < 10 MB/s (exit 9) — slow host, respawn "
                         "elsewhere", "infra")
        return _kill("complete marker present", "complete")
    if obs.get("has_joblog"):
        if obs.get("traceback"):
            return _kill("job.log has traceback (science failure — "
                         "a rerun replays the same bug)", "science")
        return _kill("job.log without completion marker (lifecycle died)",
                     "infra")

    if phase in ("loading", "None", None):
        el = obs.get("load_elapsed_min") or 0.0
        bk = d.get("load_backstop_min")
        if bk is not None and el > bk:
            return _kill(f"declared load backstop {bk}min exceeded "
                         f"({el:.0f}min)", "infra")
        pp = obs.get("pull_prog")
        if pp is not None and loading_doomed(el, bk, *pp):
            return _kill(f"pull certain to miss declared backstop {bk}min "
                         f"(lower-bound projection from layer rate)", "infra")
        fp = obs.get("same_fp_polls") or 0
        if fp >= LOAD_KILL_POLLS:
            return _kill(f"no pull progress {fp} polls", "infra")
        if fp >= LOAD_WARN_POLLS:
            return _warn(f"no pull progress {fp} polls")
        return _none()

    if phase == "running":
        rh = obs.get("run_elapsed_h") or 0.0
        mx = d.get("max_billed_h")
        if mx is not None and rh > mx:
            return _kill(f"declared max_billed_h {mx}h exceeded ({rh:.1f}h)",
                         "ceiling")
        age = obs.get("liveness_age_s")
        if age is None:
            age = rh * 3600   # streamer never seen: running-start is the clock
        if age > LIVENESS_KILL_S:
            return _kill(f"live upload stale {age:.0f}s (streamer/instance "
                         f"dead)", "infra")
        stall = obs.get("output_stall_min")
        pk = d.get("progress_kill_min")
        if stall is not None and pk is not None and stall > pk:
            return _kill(f"output static {stall:.0f}min > declared "
                         f"progress_kill_min {pk} (author opt-in)", "infra")
        if mx is not None and rh > mx * CEILING_WARN_FRAC:
            return _warn(f"at {rh:.1f}h of declared max_billed_h {mx}h")
        if age > LIVENESS_WARN_S:
            return _warn(f"live upload {age:.0f}s stale")
        ps = d.get("progress_stall_min", OUTPUT_STALL_WARN_MIN)
        if stall is not None and ps is not None and stall > ps:
            return _warn(f"output static {stall:.0f}min (warn-only: no "
                         f"progress_kill_min declared)")
        return _none()

    # unknown phase (created/offline/exited...): machinery unobservable —
    # bound total lifetime by the declared contract with backstop margins
    bk, mx = d.get("load_backstop_min"), d.get("max_billed_h")
    if bk is not None and mx is not None:
        life = (bk + mx * 60) * BACKSTOP_LIFE_MARGIN
        if (obs.get("load_elapsed_min") or 0.0) > life:
            return _kill(f"lifetime in phase '{phase}' > declared total "
                         f"{life:.0f}min", "infra")
    return _none()


# ---- server-side backstop (stateless, generous by design) ----
BACKSTOP_LOAD_MARGIN = 1.5    # laptop watchdog kills precisely; this layer
BACKSTOP_LIFE_MARGIN = 1.2    # only guarantees bounded bleed when it's asleep
BACKSTOP_LIVE_STALE_S = 1800  # streamer refreshes 60s; 30 min = long dead
ORPHAN_MIN = 30               # rent banks the manifest BEFORE create


def backstop_decide(phase, age_min, declared, has_complete, has_joblog,
                    has_manifest, liveness_age_s=None):
    """Stateless per-invocation rules for the Lambda reaper (fires every
    5 min with no memory). Uses only instance LIFETIME — an overestimate of
    run time — and generous margins over the author's declared bounds, so a
    genuine job inside its own contract is never killed here."""
    if has_complete:
        return _kill("complete marker present (cleanup)", "complete")
    if has_joblog:
        return _kill("job.log without completion marker (lifecycle died)",
                     "infra")
    if not has_manifest:
        # every genuine rental banks its manifest before the instance exists
        if age_min > ORPHAN_MIN:
            return _kill("labeled instance with no banked manifest", "infra")
        return _none()
    d = declared or {}
    bk, mx = d.get("load_backstop_min"), d.get("max_billed_h")
    loading = phase in ("loading", "None", None)
    if loading and bk is not None and age_min > bk * BACKSTOP_LOAD_MARGIN:
        return _kill(f"lifetime {age_min:.0f}min > load backstop {bk}min "
                     f"x{BACKSTOP_LOAD_MARGIN}", "infra")
    if not loading and liveness_age_s is not None \
            and liveness_age_s > BACKSTOP_LIVE_STALE_S:
        return _kill(f"live upload stale {liveness_age_s / 60:.0f}min "
                     f"(machinery long dead)", "infra")
    if bk is not None and mx is not None:
        life = (bk + mx * 60) * BACKSTOP_LIFE_MARGIN
        if age_min > life:
            return _kill(f"lifetime {age_min:.0f}min > declared total "
                         f"{life:.0f}min", "ceiling")
    return _none()


def budget_action(realized_usd, open_rate_usd_h, ceiling_usd, soft_frac=0.8):
    """'hard' -> destroy everything (the program's own declared ceiling);
    'soft' -> loud warning while real work continues; None -> fine."""
    if realized_usd >= ceiling_usd:
        return "hard"
    if realized_usd + open_rate_usd_h >= ceiling_usd * soft_frac:
        return "soft"
    return None


def should_respawn(classify, attempts, max_attempts):
    return classify == "infra" and attempts < max_attempts


def _kill(reason, classify):
    return {"action": "kill", "reason": reason, "classify": classify}


def _warn(reason):
    return {"action": "warn", "reason": reason, "classify": None}


def _none():
    return {"action": "none", "reason": "", "classify": None}
