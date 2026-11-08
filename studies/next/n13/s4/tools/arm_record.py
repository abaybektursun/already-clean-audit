"""Per-arm record of one banked S4 step, from the scratch bank (full files). No accuracy and no correct count is read or written
(blind protocol, N13_SPEC.md §9.5): gates, G-COMPLETE recomputed from generations.jsonl, mask, DARE kept counts, energies, the
per-tensor hashes against R1 stage 6 (s3gate-03 arm_hashes.json), cap hits, lengths, engine, wall time against the projection.
Writes studies/next/n13/s4/<prefix>/<step>/arm_record.json and prints it.
usage: PYTHONPATH=<worktree> PYTHONDONTWRITEBYTECODE=1 <n13 venv python> arm_record.py <prefix> <step>
Fail-loud: no try/except."""
import datetime as dt, json, pathlib, sys

import numpy as np

from infra.pg import n13_plan as P
from infra.pg import n13_score as S

REPO = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad")
prefix, step = sys.argv[1], sys.argv[2]
root = SCR / "s4banks" / prefix / step
arm = step
man = json.loads((REPO / f"infra/aws_pg/steps/n13_{prefix}.json").read_text())
args = next(s["args"] for s in man if s["name"] == step).split()
g = lambda f: args[args.index(f) + 1]
benches = g("--bench").split(",")
seed_set = P.seed_set(arm)
proj = P.gen_minutes(500, 1319 if "gsm8k" in benches else 0, float(g("--proj-tok-s")), float(g("--proj-mean-math")),
                     float(g("--proj-mean-gsm")), float(g("--proj-load-min")))
R = {"prefix": prefix, "step": step, "arm": arm, "seed_set": seed_set, "benches": benches, "projection_min": round(proj, 1),
     "allowance_min": float(g("--allowance-min"))}
bj = root / arm / "build.json"
if bj.exists():
    b = json.loads(bj.read_text())
    ref = json.loads((SCR / "s4bank_ref/arm_hashes.json").read_text()).get(arm)
    eq = None
    if ref is not None:
        eq = {"n_tensors": len(b["sha256"]), "n_equal": sum(b["sha256"].get(k) == v for k, v in ref["sha256"].items()),
              "same_names": set(b["sha256"]) == set(ref["sha256"])}
        eq["all_equal"] = eq["same_names"] and eq["n_equal"] == eq["n_tensors"]
    R["build"] = {"build_seconds": b.get("build_seconds"), "n_neurons_in_set": b.get("n_neurons_in_set"),
                  "spec": {k: v for k, v in b["spec"].items() if k not in ("neurons",)},
                  "families": b["families"], "G-MASK": b.get("mask", {}).get("pass"),
                  "kept": b.get("kept"), "kept_expected": b.get("kept_expected"), "kept_rel_err": b.get("kept_rel_err"),
                  "kept_sd_units": b.get("kept_sd_units"), "kept_pass": b.get("kept_pass"),
                  "intended_mlp_energy": b.get("intended_mlp_energy"), "energy_rel_err": b.get("energy_rel_err"),
                  "energy_flag": b.get("energy_flag"), "energy_passes": b.get("energy_passes"), "c_built": b.get("c_built"), "a31": b.get("a31"),
                  "hashes_vs_R1_stage6": eq, "R1_stage6_note": "rndE hashes changed by A30 (stochastic rounding); R1 at s3gate-03 ran before A30"
                  if arm.startswith("rndE") else None}
R["bench"] = {}
for bn in benches:
    d = root / arm / bn
    m = json.loads((d / "metrics.json").read_text())
    recs = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip()]
    n = 500 if bn == "math500" else 1319
    gc = S.g_complete(recs, n, seed_set) if m["status"] == "done" else None
    gc_fi = (not S.g_complete(recs[:-1], n, seed_set)["pass"]) if gc else None
    tp = m.get("throughput", {})
    t0 = dt.datetime.fromisoformat(m["started_utc"].replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(m["finished_utc"].replace("Z", "+00:00")) if m.get("finished_utc") else None
    R["bench"][bn] = {"status": m["status"], "started_utc": m["started_utc"], "finished_utc": m.get("finished_utc"),
                      "wall_min": round((t1 - t0).total_seconds() / 60, 1) if t1 else None,
                      "template": m.get("template"), "cap": m.get("cap"), "park": m.get("park"), "condition": m.get("condition"),
                      "screen_fired": m.get("screen", {}).get("fired"), "screen_cap_hits_0_19": m.get("screen", {}).get("cap_hits"),
                      "G-COMPLETE_job": m.get("G-COMPLETE"), "G-COMPLETE_recount": gc, "G-COMPLETE_failing_input_failed": gc_fi,
                      "n_records": len(recs), "cap_hits": sum(r["cap_hit"] for r in recs),
                      "mean_new_tokens": float(np.mean([r["n_new"] for r in recs])) if recs else None,
                      "modes": {k: sum(r["mode"] == k for r in recs) for k in ("answer_line", "boxed", "none")},
                      "metrics_build": m.get("build"),
                      "engine": {k: m.get("engine", {}).get(k) for k in ("vllm", "max_num_seqs", "max_num_batched_tokens", "num_gpu_blocks",
                                                                         "attention_backend", "env", "load_s")},
                      "throughput": {k: tp.get(k) for k in ("generation_seconds", "step_s", "r_sat", "mean_new_tokens", "median_new_tokens",
                                                            "max_new_tokens", "n_cap_hits", "mean_output_tokens", "n_answers", "load_s")}}
out = REPO / "studies/next/n13/s4" / prefix / step / "arm_record.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
print(json.dumps(R, indent=1))
