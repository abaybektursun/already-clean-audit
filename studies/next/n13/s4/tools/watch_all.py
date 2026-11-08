"""S4 poller for every open prefix, every 120 s: out/pipeline_status.json, STATUS, the EC2 state, the running gen step's metrics.json
status fields (status, finished, cap hits, screen; never the correct count or the accuracy: blind protocol, spec 9.5), and the B2-02
launch log. Never job.log. Each poll appends one line per prefix to <dir>/w_<prefix>.log. State is kept in <dir>/watch_state.json across
runs. Exits after the poll that sees an event (a step's state change, STATUS, the instance leaving 'running', a step's engine block first
seen, the screen firing, a LAUNCHED line), printing EVENT lines; else after <max> polls, printing one status line per prefix.
usage: python3 -I watch_all.py <max polls> <prefix>=<instance id or 'pending'> ..."""
import json, os, subprocess, sys, time
B = "s3://pg-lineage-769063704696/results/next_n13"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]
D = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(D, "watch_state.json")
mx = int(sys.argv[1])
targets = dict(a.split("=") for a in sys.argv[2:])
state = json.load(open(ST)) if os.path.exists(ST) else {}


def cat(key):
    r = subprocess.run(["aws", "s3", "cp", f"{B}/{key}", "-", *AWS], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def inst(iid):
    r = subprocess.run(["aws", "ec2", "describe-instances", "--instance-ids", iid, "--query", "Reservations[0].Instances[0].State.Name",
                        "--output", "text", *AWS], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"describe-failed:{r.stderr.strip()[:80]}"


lines = {}
for n in range(mx):
    ev = []
    for p, iid in targets.items():
        s = state.setdefault(p, {"key": None, "fired": False, "engines": [], "launched": False})
        if iid == "pending":
            lg = os.path.join(D, f"{p}.log")
            txt = open(lg).read() if os.path.exists(lg) else ""
            tries = txt.count(" start ")
            cur = txt[max(txt.rfind("loop restart"), 0):]          # only the running loop's lines
            stop = "\nSTOP " in cur
            txt = cur if "LAUNCHED" not in cur else txt
            if ("LAUNCHED" in txt or stop) and not s["launched"]:
                s["launched"] = True
                ev.append(f"{p}: launch loop {'STOPPED' if stop else 'LAUNCHED'}")
            lines[p] = f"{p} pending, {tries} tries, last: {txt.strip().splitlines()[-1][:120] if txt.strip() else '-'}"
            continue
        ps, stt, es = cat(f"{p}/out/pipeline_status.json"), cat(f"{p}/STATUS"), inst(iid)
        if ps:
            d = json.loads(ps)
            key = f"{d['state']} {d['completed']}/{d['n_steps']} " + " ".join(f"{x['name']}={x['state']}" for x in d["steps"])
            run = [x["name"] for x in d["steps"] if x["state"] == "running"]
        else:
            key, run = "no pipeline_status", []
        met, fired = "", False
        if run and run[0] != "setup":
            for b in ("math500", "gsm8k"):
                m = cat(f"{p}/out/{run[0]}/{run[0]}/{b}/metrics.json")
                if m:
                    m = json.loads(m)
                    c, sc = m.get("counts", {}), m.get("screen", {})
                    fired |= bool(sc.get("fired"))
                    if "engine" in m and run[0] not in s["engines"]:
                        s["engines"].append(run[0])
                        ev.append(f"{p}: engine loaded {run[0]}")
                    met += f" {b} {m.get('status')} {c.get('finished')}/{m.get('submitted')} cap {c.get('cap_hits')} screen {'fired' if sc.get('fired') else ('ready' if sc.get('ready') else '-')};"
        full = f"{key} | STATUS={stt.strip() if stt else 'none'} | ec2={es}"
        lines[p] = f"{p} {full} |{met}"
        open(os.path.join(D, f"w_{p}.log"), "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + lines[p] + "\n")
        if s["key"] is not None and s["key"] != full and not full.startswith("no pipeline_status"):
            ev.append(f"{p}: state change -> {key}")
        if stt:
            ev.append(f"{p}: STATUS {stt.strip()}")
        if es not in ("running",) and not es.startswith("describe-failed"):
            ev.append(f"{p}: ec2 {es}")
        if fired and not s["fired"]:
            ev.append(f"{p}: screen fired at {run[0]}")
        s["key"], s["fired"] = full, fired or s["fired"]
    json.dump(state, open(ST, "w"), indent=1)
    if ev:
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "EVENT")
        print("\n".join(ev))
        print("\n".join(lines.values()))
        sys.exit(0)
    if n < mx - 1:
        time.sleep(120)
print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), f"NO EVENT after {mx} polls")
print("\n".join(lines.values()))
