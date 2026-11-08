"""S4 poller, one per prefix, every 120 s: out/pipeline_status.json, STATUS, the EC2 instance state, and the running gen step's
metrics.json status fields (status, finished, cap hits, screen fired; never the correct count or the accuracy: blind protocol, spec
9.5). Never job.log. One line per poll into <log>. Exits when a step's state changes, STATUS appears, the instance leaves 'running',
the screen fires, or a step's engine block first appears (the head's engine rule, received 18:35Z), so the executor wakes; else after <max> polls.
usage: python3 -I watch_s4.py <log> <max polls> <prefix> <instance id>"""
import json, subprocess, sys, time
B = "s3://pg-lineage-769063704696/results/next_n13"
log, mx, p, iid = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
AWS = ["--profile", "fuelos", "--region", "us-east-1"]


def cat(key):
    r = subprocess.run(["aws", "s3", "cp", f"{B}/{key}", "-", *AWS], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def inst():
    r = subprocess.run(["aws", "ec2", "describe-instances", "--instance-ids", iid, "--query", "Reservations[0].Instances[0].State.Name",
                        "--output", "text", *AWS], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"describe-failed:{r.stderr.strip()[:80]}"


import os
SEEN = log + ".engines"          # steps whose engine block was already reported (kept across restarts)
last, last_fired = None, None
engine_seen = set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
for n in range(mx):
    ps, st, es = cat(f"{p}/out/pipeline_status.json"), cat(f"{p}/STATUS"), inst()
    if ps:
        d = json.loads(ps)
        key = f"{d['state']} {d['completed']}/{d['n_steps']} " + " ".join(f"{s['name']}={s['state']}" for s in d["steps"])
        run = [s["name"] for s in d["steps"] if s["state"] == "running"]
    else:
        key, run = "no pipeline_status", []
    met, fired, new_engine = "", False, []
    if run and run[0] != "setup":
        for b in ("math500", "gsm8k"):
            m = cat(f"{p}/out/{run[0]}/{run[0]}/{b}/metrics.json")
            if m:
                m = json.loads(m)
                c = m.get("counts", {})
                sc = m.get("screen", {})
                fired |= bool(sc.get("fired"))
                if "engine" in m and run[0] not in engine_seen:
                    new_engine.append(run[0])
                met += (f" {b} {m.get('status')} finished {c.get('finished')}/{m.get('submitted')} cap {c.get('cap_hits')}"
                        f" screen {'fired' if sc.get('fired') else ('ready' if sc.get('ready') else '-')} ({m.get('updated_utc')});")
    key += f" | STATUS={st.strip() if st else 'none'} | ec2={es}"
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {p} {key} |{met}"
    open(log, "a").write(line + "\n")
    print(line, flush=True)
    ev = []
    if last is not None and last != key:
        ev.append("state change")
    if st:
        ev.append("STATUS")
    if es != "running" and not es.startswith("describe-failed"):
        ev.append(f"ec2 {es}")
    if new_engine:
        engine_seen.update(new_engine)
        open(SEEN, "w").write("\n".join(sorted(engine_seen)) + "\n")
        ev.append(f"engine loaded {new_engine[0]}")
    if last_fired is False and fired:
        ev.append("screen fired")
    if ev:
        print(f"EVENT {p}: " + ", ".join(ev), flush=True)
        sys.exit(0)
    last, last_fired = key, fired
    time.sleep(120)
print(f"NO EVENT {p} after {mx} polls", flush=True)
