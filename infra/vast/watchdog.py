#!/usr/bin/env python3
"""vast watchdog daemon — detection layer 2 (see ARCHITECTURE.md).

One persistent process watching ALL instances of THIS program
(program.json names it; the label prefix and S3 prefix follow). Emits one stdout
line per EVENT (phase change, warn, action, heartbeat) — run it with
`python3 -u` under a session Monitor so every event wakes the operator.

Every destroy decision comes from policy.decide() (pure, table-tested in
test_policy.py) — this file only builds observations and acts:

  observe   phase, pull progress (parsed from status_msg), streamer liveness
            (S3 job_live.log age — refreshed every 60 s regardless of content,
            so staleness == dead machinery, never "quiet job"), output
            progress (job_live.log size + progress.json age), banked markers.
  act       warn (deduped) | destroy | destroy+respawn. Respawn only on
            INFRA-classified kills (policy.should_respawn), bounded by
            MAX_RESPAWNS counted durably in the spend ledger, excluding the
            failed machine_id. Science failures and ceiling kills never
            respawn.
  race      if a rental has 2+ live instances (controller rent --race) and
            one reached running, the still-loading siblings lost: destroyed,
            no respawn, verdict "race_lost".
  budget    policy.budget_action(realized + open estimate) per poll; soft ->
            loud warning, real work continues; hard (program ceiling) ->
            destroy everything, respawns disabled.
  ledger    machine_id backfilled on first sight; kills stamp host_verdict
            (wedged/clean/race_lost) — controller plan excludes wedged hosts.

No try/except (house rule): transient poll failures are handled by returncode
and shape checks — a failed poll is a skipped tick, never a crash.
"""
import json
import os
import fcntl
import subprocess
import time

import policy

POLL_S = 60
HEARTBEAT_S = 1800                  # operator-comfort cadence, not detection
MAX_RESPAWNS = 3                    # per rental, lifetime (counted in ledger)
AWS = ["aws", "--profile", "fuelos", "--region", "us-east-1"]
HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = json.load(open(os.path.join(HERE, "budget.json")))
# same source of truth as controller.py — see program.json
CFG = json.load(open(os.path.join(HERE, "program.json")))
PROGRAM = CFG["program"]
S3_BASE = CFG["s3_results"]                       # s3://bucket/<key base>
BUCKET, S3_KEY_BASE = S3_BASE[len("s3://"):].split("/", 1)
LEDGER = os.path.join(HERE, "..", "..", CFG["ledger"])


def acquire_watchdog_lock(path=None):
    """One supervisor per program across worktrees, held until process exit."""
    path = path or f"/tmp/vast-watchdog-{os.getuid()}-{PROGRAM}.lock"
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        raise SystemExit(f"Another watchdog owns {path}: {owner}")
    handle.seek(0)
    handle.truncate()
    json.dump({"pid": os.getpid(), "cwd": os.getcwd()}, handle)
    handle.flush()
    return handle


def sh(args, timeout=60):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def instances():
    r = sh(["vastai", "show", "instances", "--raw"])
    if r.returncode != 0 or not r.stdout.strip().startswith("["):
        return None  # poll miss
    return [i for i in json.loads(r.stdout)
            if str(i.get("label", "")).startswith(f"{PROGRAM}_")]


def destroy(iid, why):
    print(f"ACTION destroy {iid}: {why}")
    sh(["bash", "-c", f"echo y | vastai destroy instance {iid}"])


def s3_head(rental, name):
    """(age_s, size_bytes) of an S3 artifact, or None if absent."""
    r = sh([*AWS, "s3api", "head-object", "--bucket", BUCKET,
            "--key", f"{S3_KEY_BASE}/{PROGRAM}_{rental}/{name}"])
    if r.returncode != 0 or not r.stdout.strip().startswith("{"):
        return None
    h = json.loads(r.stdout)
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(h["LastModified"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (time.time() - dt.timestamp(), int(h["ContentLength"]))


def s3_has(rental, name):
    r = sh([*AWS, "s3", "ls", f"{S3_BASE}/{PROGRAM}_{rental}/{name}"])
    return r.returncode == 0 and bool(r.stdout.strip())


def s3_json(rental, name):
    """Parsed JSON artifact, or None if absent / not JSON."""
    r = sh([*AWS, "s3", "cp", f"{S3_BASE}/{PROGRAM}_{rental}/{name}", "-"])
    ok = r.returncode == 0 and r.stdout.strip().startswith("{")
    return json.loads(r.stdout) if ok else None


def archive_attempt(rental, n):
    """Move the dead attempt's terminal markers aside before a respawn. The
    respawn reuses the rental's S3 dir, so a left-over job.log / completion
    marker / stale job_live.log would make policy kill the NEW instance on
    its first poll (has_joblog, has_complete, liveness_age). The bucket is
    versioned + deny-delete-version: nothing is lost, `mv` only re-keys."""
    for name in ("job.log", "_rental_complete.json", "job_live.log", "timings.tsv"):
        src = f"{S3_BASE}/{PROGRAM}_{rental}/{name}"
        if s3_has(rental, name):
            sh([*AWS, "s3", "mv", src,
                f"{S3_BASE}/{PROGRAM}_{rental}/attempts/{n}/{name}"], timeout=120)


def joblog_traceback(rental):
    r = sh([*AWS, "s3", "cp", f"{S3_BASE}/{PROGRAM}_{rental}/job.log", "-"])
    return r.returncode == 0 and "Traceback (most recent call last)" in r.stdout


def watch_cfg(rental, cache={}):
    """Author-declared bounds from the rent-time manifest banked on S3."""
    if rental not in cache:
        r = sh([*AWS, "s3", "cp", f"{S3_BASE}/{PROGRAM}_{rental}/manifest.json", "-"])
        ok = r.returncode == 0 and r.stdout.strip().startswith("{")
        cache[rental] = json.loads(r.stdout).get("watch", {}) if ok else {}
    return cache[rental]


def track_pull(st, status_msg, now):
    """Update per-instance pull tracking; return pull_prog tuple or None."""
    cur = policy.parse_layer_progress(status_msg)
    prev = st.get("pull_prev")
    st["pull_prev"] = (cur[0], cur[1], now) if cur else prev
    if cur is None or prev is None:
        return None
    done0, total0, t0 = prev
    if abs(total0 - cur[1]) > 0.01 or now <= t0:
        return None   # layer changed or same poll — no rate signal
    return (done0, cur[0], (now - t0) / 60.0, cur[1])


def track_output(st, live_head, prog_head, now):
    """Minutes since job output last advanced (live-log size growth or
    progress.json refresh), or None before any signal exists."""
    if live_head is not None:
        size = live_head[1]
        if size != st.get("out_size"):
            st["out_size"], st["out_t"] = size, now
    if prog_head is not None and prog_head[0] < 90:
        st["out_t"] = now - prog_head[0]
    if "out_t" not in st:
        return None
    return (now - st["out_t"]) / 60.0


def ledger_rows():
    p = LEDGER
    return (json.load(open(p)), p) if os.path.exists(p) else ([], p)


def ledger_stamp(iid, **fields):
    rows, p = ledger_rows()
    for r in rows:
        if r.get("instance_id") == iid:
            r.update(fields)
    json.dump(rows, open(p, "w"), indent=2)


def respawn_count(rental):
    rows, _ = ledger_rows()
    return sum(1 for r in rows
               if r.get("rental") == rental and r.get("auto_respawn"))


def respawn(rental, machine_id):
    if not watch_cfg(rental).get("auto_respawn", True):
        print(f"RESPAWN-DISABLED {rental}: operator will inspect the failure")
        return
    n = respawn_count(rental)
    if not policy.should_respawn("infra", n, MAX_RESPAWNS):
        print(f"RESPAWN-EXHAUSTED {rental}: {n}/{MAX_RESPAWNS} attempts used "
              f"— operator decision needed")
        return
    archive_attempt(rental, n + 1)
    r = sh(["python3", os.path.join(HERE, "controller.py"), "respawn", rental,
            "--exclude-machine", str(machine_id or 0)], timeout=180)
    ok = r.returncode == 0
    print(f"RESPAWN {rental} attempt {n + 1}/{MAX_RESPAWNS}: "
          f"{'launched' if ok else 'FAILED'}")
    if not ok:
        print((r.stderr or r.stdout).strip()[-500:])


def budget_check(insts, states, now, last_state):
    rows, _ = ledger_rows()
    realized = sum(r.get("realized_usd") or 0 for r in rows)
    open_est = open_rate = 0.0
    for i in insts:
        st = states.get(i["id"], {})
        if str(i.get("actual_status")) == "running":
            open_rate += float(i.get("dph_total", 0))
            open_est += float(i.get("dph_total", 0)) * \
                max(0.0, now - st.get("run_t0", now)) / 3600
    act = policy.budget_action(realized + open_est, open_rate,
                               BUDGET["program_ceiling_usd"],
                               BUDGET["soft_frac"])
    if act != last_state and act is not None:
        print(f"BUDGET-{act.upper()} realized+open=${realized + open_est:.2f} "
              f"rate=${open_rate:.2f}/h ceiling=${BUDGET['program_ceiling_usd']}")
    return act


def main():
    lock_handle = acquire_watchdog_lock()  # keep the descriptor alive for this process
    states = {}   # iid -> tracking dict
    last_beat, budget_state = time.time(), None
    print(f"WATCHDOG up pid={os.getpid()} poll={POLL_S}s "
          f"ceiling=${BUDGET['program_ceiling_usd']}")
    while True:
        insts = instances()
        now = time.time()
        if insts is None:
            print("WARN poll miss (vast API)")
        else:
            budget_state = budget_check(insts, states, now, budget_state)
            if budget_state == "hard":
                for i in insts:
                    destroy(i["id"], "program budget ceiling — destroy all")
                    ledger_stamp(i["id"], status="killed_budget")
                time.sleep(POLL_S)
                continue

            # race resolution: one sibling running -> loading siblings lost
            by_rental = {}
            for i in insts:
                by_rental.setdefault(i["label"], []).append(i)
            race_losers = set()
            for group in by_rental.values():
                if len(group) > 1 and any(
                        str(g.get("actual_status")) == "running" for g in group):
                    for g in group:
                        if str(g.get("actual_status")) != "running":
                            race_losers.add(g["id"])

            live_ids = set()
            for i in insts:
                iid = i["id"]
                live_ids.add(iid)
                rental = i["label"].removeprefix(f"{PROGRAM}_")
                phase = str(i.get("actual_status"))
                st = states.setdefault(iid, {"fp": "", "same": 0, "t0": now,
                                             "phase": None})
                st["rental"] = rental
                if i.get("machine_id") and not st.get("mid_stamped"):
                    ledger_stamp(iid, machine_id=i["machine_id"])
                    st["mid_stamped"] = True
                    st["machine_id"] = i["machine_id"]
                if phase != st["phase"]:
                    print(f"EVENT {iid} {rental}: phase {st['phase']} -> "
                          f"{phase} (+{(now - st['t0']) / 60:.0f}min)")
                    st["phase"] = phase
                    if phase == "running":
                        st["run_t0"] = now
                        ledger_stamp(iid, load_min=round((now - st["t0"]) / 60, 1))
                if iid in race_losers:
                    destroy(iid, "race lost — sibling reached running first")
                    ledger_stamp(iid, host_verdict="race_lost", status="closed")
                    continue

                # --- build observation ---
                fp = f"{phase}|{i.get('status_msg', '')}"
                st["same"] = st["same"] + 1 if fp == st["fp"] else 0
                st["fp"] = fp
                has_joblog = s3_has(rental, "job.log")
                live_head = s3_head(rental, "job_live.log")
                prog_head = s3_head(rental, "live_out/progress.json")
                comp = s3_json(rental, "_rental_complete.json")
                obs = {
                    "phase": phase,
                    "has_complete": comp is not None,
                    "exit_code": comp.get("exit_code") if comp else None,
                    "has_joblog": has_joblog,
                    "traceback": has_joblog and joblog_traceback(rental),
                    "declared": watch_cfg(rental),
                    "load_elapsed_min": (now - st["t0"]) / 60.0,
                    "same_fp_polls": st["same"],
                    "pull_prog": track_pull(st, i.get("status_msg", ""), now),
                    "run_elapsed_h": (now - st.get("run_t0", now)) / 3600.0,
                    "liveness_age_s": live_head[0] if live_head else None,
                    "output_stall_min": track_output(st, live_head, prog_head,
                                                     now),
                }
                dec = policy.decide(obs)

                # --- act ---
                if dec["action"] == "warn":
                    if dec["reason"] != st.get("warned"):
                        print(f"WARN {iid} {rental}: {dec['reason']}")
                        st["warned"] = dec["reason"]
                elif dec["action"] == "kill":
                    destroy(iid, dec["reason"])
                    cls = dec["classify"]
                    if cls == "complete":
                        ledger_stamp(iid, host_verdict="clean", status="done")
                        print(f"DONE {rental}: results banked — run "
                              f"controller ledger")
                    elif cls == "science":
                        ledger_stamp(iid, host_verdict="clean",
                                     status="failed_science")
                        print(f"FAILED {rental}: science failure (traceback "
                              f"in job.log) — NO respawn, fix the bug")
                    elif cls == "ceiling":
                        ledger_stamp(iid, host_verdict="clean",
                                     status="killed_ceiling")
                        print(f"FAILED {rental}: exceeded its declared "
                              f"ceiling — NO respawn, raising it is the "
                              f"author's call")
                    elif cls == "infra":
                        wedged = phase in ("loading", "None")
                        slow = obs["exit_code"] == policy.NET_PROBE_EXIT
                        ledger_stamp(iid, host_verdict="slow_net" if slow else
                                     "wedged" if wedged else "died",
                                     status="killed_infra")
                        print(f"FAILED {rental}: infra ({dec['reason']}) — "
                              f"respawning on a different host")
                        respawn(rental, i.get("machine_id"))
            for iid in list(states):
                if iid not in live_ids:
                    # a healthy rental self-destroys from inside (lifecycle trap)
                    # and can vanish between two polls, so the DONE branch above
                    # never sees it. Close the ledger row here instead of leaving
                    # every clean rental stamped "open" forever.
                    rental = states[iid].get("rental")
                    comp = s3_json(rental, "_rental_complete.json") if rental else None
                    done = comp is not None
                    if done and comp.get("exit_code") == policy.NET_PROBE_EXIT:
                        # the net probe self-destroyed in seconds (lifecycle trap) — the
                        # live branch never saw it; the host is slow, not the job
                        mid = states[iid].get("machine_id")
                        ledger_stamp(iid, host_verdict="slow_net", status="killed_infra")
                        print(f"FAILED {rental}: net probe {comp.get('mb_per_s')} MB/s "
                              f"on m{mid} — respawning on a different host")
                        respawn(rental, mid)
                    elif done:
                        ledger_stamp(iid, status="done", host_verdict="clean")
                    print(f"EVENT {iid}: instance gone"
                          f"{' (complete marker present -> ledger closed)' if done else ''}")
                    del states[iid]
        if now - last_beat >= HEARTBEAT_S:
            n = len(insts) if insts is not None else -1
            print(f"HEARTBEAT watching {n} instances")
            last_beat = now
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
