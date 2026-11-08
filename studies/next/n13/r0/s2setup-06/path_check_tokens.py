"""Token counts and timing of the s2setup-06 path check (olmes --limit 20, batch size 1), for N13_SPEC.md A19.

    python3 -I path_check_tokens.py <dir with the 7 task-*-predictions.jsonl of out/r0_known/olmes_limit20> <this dir> > path_check_tokens.json

Reads: the olmes predictions (num_tokens, context_tokens per generation; copied from
s3://pg-lineage-769063704696/results/next_n13/s2setup-06/out/r0_known/olmes_limit20/), and from this directory the task metrics
(current_date = the time each subject's generation began), setup/metrics.json (finished_utc, when r0_known began) and
r0_known/metrics.json (path_check.elapsed_s). The full-task token count is an estimate: the path check's mean length per subject times
the subject's test size times 4 repeats.
"""
import datetime as dt
import glob
import json
import pathlib
import sys

SIZES = {"algebra": 1187, "counting_and_probability": 474, "geometry": 479, "intermediate_algebra": 903, "number_theory": 540,
         "prealgebra": 871, "precalculus": 546}
pred_dir, here = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
utc = lambda s: dt.datetime.strptime(s.replace(" UTC", "").replace("Z", ""), "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%dT%H:%M:%S")
start = utc(json.loads((here / "setup/metrics.json").read_text())["finished_utc"])
wall = json.loads((here / "r0_known/metrics.json").read_text())["path_check"]["elapsed_s"]
end = start + dt.timedelta(seconds=wall)
begun = {}
for f in sorted((here / "r0_known/olmes_limit20").glob("task-*-metrics.json")):
    m = json.loads(f.read_text())
    begun[m["task_name"].replace("minerva_math_", "")] = utc(m["current_date"])
order = sorted(begun, key=begun.get)
per, tot, n_all, cap_all = {}, 0, 0, 0
for s in SIZES:
    (f,) = glob.glob(str(pred_dir / f"task-*-minerva_math_{s}-predictions.jsonl"))
    outs = [o for l in open(f) for o in json.loads(l)["model_output"]]
    nt = [o["num_tokens"] for o in outs]
    ct = [o["context_tokens"] for o in outs]
    i = order.index(s)
    t_end = begun[order[i + 1]] if i + 1 < len(order) else end
    secs = (t_end - begun[s]).total_seconds()
    per[s] = {"generations": len(nt), "tokens": sum(nt), "mean_tokens": sum(nt) / len(nt), "max_tokens": max(nt),
              "at_1024": sum(x >= 1024 for x in nt), "context_mean": sum(ct) / len(ct), "context_max": max(ct),
              "began_utc": begun[s].isoformat() + "Z", "seconds_to_next_subject": secs, "tokens_per_s": sum(nt) / secs,
              "full_task_tokens_estimate": SIZES[s] * 4 * sum(nt) / len(nt)}
    tot, n_all, cap_all = tot + sum(nt), n_all + len(nt), cap_all + per[s]["at_1024"]
gen_s = (end - begun[order[0]]).total_seconds()
print(json.dumps({"source": "s2setup-06 path check, olmes --limit 20 at batch size 1", "r0_known_began_utc": start.isoformat() + "Z",
                  "path_check_wall_s": wall, "path_check_end_utc_estimate": end.isoformat() + "Z",
                  "startup_s": (begun[order[0]] - start).total_seconds(), "generation_phase_s": gen_s,
                  "generations": n_all, "tokens": tot, "mean_tokens": tot / n_all, "at_1024": cap_all, "tokens_per_s_batch1": tot / gen_s,
                  "seconds_per_generation_wall": wall / n_all,
                  "full_task_tokens_estimate": sum(v["full_task_tokens_estimate"] for v in per.values()), "per_subject": per}, indent=1))
