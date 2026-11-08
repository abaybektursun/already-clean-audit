"""N13 plan (studies/next/07_PICKS/N13_SPEC.md §3.2, §3.3, §9.1 S3 layout, §9.4 sessions, §9.7 budget and stop rule): the arm roster
with its priority and instance, the seed sets, the S3 attempt prefixes, the manifests, the budget projector and the park check.

Arm names: the spec's "top_1%" is "top_k1" here (k tags k0.1, k0.3, k1, k3), "rnd0_1%" is "rnd0_k1", "top_1%_B" is "top_k1_B",
"rndT_1%_d2" is "rndT_k1_d2", "dare_1%_d2" is "dare_k1_d2", "topF_1%" is "topF_k1".

A22 (N13_SPEC.md): the S3 and S4 projection from the smoke's engine-step logs (step_rates, sat_phase, project_smoke), run on the
laptop as

    python -m infra.pg.n13_plan --smoke-dir <dir with smoke_base/ and smoke_rl/> --cap C --spent <GPU-h> [--rate min]
        [--killed rl [--killed-reading prefix|all]] [--out projection.json] [--manifest-dir infra/aws_pg/steps]

A24 (2026-11-06): --killed names a smoke cut by the hard kill (no throughput.json); its L comes from its finished answers over the
admitted prefix with the cut answers at C, its rate and tail from the other smoke (killed_smoke).

The rate is min(r_full, r_sat) (A22 addition, 2026-11-06); the S3 and S4 table is also given at r_full and at r_sat alone. It exits
non-zero with a message when the rule's rate is undefined (no engine step in the class the rate needs); no manifest is written then.
A27 (2026-11-06, the user: "Remove the GPU limits."): no study ceiling; the stop rule's check of spent plus projection against 35 GPU-h
is removed. Each launch's allowance (the manifest's --allowance-min and pg's --budget-min) is 1.3 times its own projection.
A28 (2026-11-06, the head): S3 runs as two launches, s3gate1 (setup with stage 2, R1, base, rl) and s3gate2 (setup, zero, id); zero
and id read no R1 output. S4 runs as four launches, each instance's priority-ordered list split by turns (S4_SPLIT), so that the
first steps of all four are the highest priorities. --manifest-dir writes the six manifests.
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import calendar
import json
import pathlib
import time

BUCKET = "pg-lineage-769063704696"
ROOT = "results/next_n13"
ATTEMPT_KINDS = ("s2setup", "s2smoke", "s3gate", "s4armsA", "s4armsB", "s5pair2",
                 "s3gate1", "s3gate2", "s4armsA1", "s4armsA2", "s4armsB1", "s4armsB2")          # A28: the last six
N_MATH, N_GSM = 500, 1319
TARGET_GPU_H = (20.0, 25.0)

# (arm, seed set, carries GSM8K, priority, instance, conditional)
GATE = [("base", "A", True), ("rl", "A", True), ("zero", "B", True), ("id", "B", True)]
S4 = {
    "A": [("top_k1", 1, True), ("dare_k1", 1, True), ("mlp", 2, True), ("top_k3", 3, False), ("top_k0.3", 3, False), ("top_k0.1", 3, False),
          ("dare_k3", 4, False), ("dare_k0.3", 4, False), ("dare_k0.1", 4, False), ("comp_k1", 5, True), ("comp_k3", 5, False),
          ("rnd0_k1", 6, False), ("top_k1_B", 6, False), ("dare_k1_d2", 6, False)],
    "B": [("rndE_k1", 1, True), ("rndT_k1", 1, True), ("attn", 2, True), ("rndE_k3", 3, False), ("rndT_k3", 3, False),
          ("rndE_k0.3", 4, False), ("rndT_k0.3", 4, False), ("rndE_k0.1", 4, False), ("rndT_k0.1", 4, False), ("comp_k0.3", 5, False),
          ("comp_k0.1", 5, False), ("rndT_k1_d2", 6, False), ("topF_k1", 7, False)],
}
CONDITIONAL = {"topF_k1": "J_floor(k1) < 0.9"}
# A28: four S4 launches. Each instance's list (priority order) is split by turns: the 1st, 3rd, 5th ... arm to manifest 1, the 2nd, 4th
# ... to manifest 2, so every manifest starts with its instance's highest priorities and the four together run P1 first.
S4_SPLIT = {f"s4arms{i}{h + 1}": S4[i][h::2] for i in ("A", "B") for h in (0, 1)}
S3_SPLIT = {"s3gate1": GATE[:2], "s3gate2": GATE[2:]}   # A28: base and rl with R1; zero and id (built from base and rl, no R1 input)
LAUNCHES = (*S3_SPLIT, *S4_SPLIT)


def seed_set(arm: str) -> str:
    return "B" if arm in ("zero", "id") or arm.endswith("_B") else "A"


def roster() -> list:
    """Every arm run of the full roster: 31 MATH-500 runs (30 unconditional, 1 conditional), 11 GSM8K runs."""
    rows = [{"arm": a, "seed_set": s, "gsm8k": g, "priority": 0, "launch": k, "conditional": False} for k, arms in S3_SPLIT.items()
            for a, s, g in arms]
    for k, arms in S4_SPLIT.items():
        rows += [{"arm": a, "seed_set": seed_set(a), "gsm8k": g, "priority": p, "launch": k, "conditional": a in CONDITIONAL}
                 for a, p, g in arms]
    return rows


def check_roster() -> dict:
    r = roster()
    arms = [x["arm"] for x in r]
    v = {"math500_runs": len(r), "unconditional": sum(not x["conditional"] for x in r), "gsm8k_runs": sum(x["gsm8k"] for x in r),
         "distinct": len(set(arms)) == len(arms), "steps_A": len(S4["A"]), "steps_B": len(S4["B"])}
    v["pass"] = v["math500_runs"] == 31 and v["unconditional"] == 30 and v["gsm8k_runs"] == 11 and v["distinct"] \
        and v["steps_A"] == 14 and v["steps_B"] == 13
    return v


def prefix(kind: str, nn: int) -> str:
    assert kind in ATTEMPT_KINDS and 1 <= nn <= 99, (kind, nn)
    return f"{ROOT}/{kind}-{nn:02d}"


def check_fresh(listing: list, pfx: str) -> None:
    """Fresh S3 prefix per attempt: the caller passes `aws s3 ls --recursive` keys; any key under the prefix stops the launch."""
    used = [k for k in listing if k.startswith(pfx + "/")]
    assert not used, f"{pfx} is not fresh: {used[:3]}"


# ------------------------------------------------------------------------------------------------ budget (§9.7)
ASSUME = {"tok_s": (800.0, 500.0), "mean_math": 2500.0, "mean_gsm": 940.0, "load_min": 5.0, "setup_min": 20.0,
          "olmes_venv_min": 10.0, "olmes_load_min": 5.0, "r0_tokens": 5.0e6, "r1_min": 75.0, "smoke_problems": 20}
# A22: the smoke of the s2smoke launch runs MATH-500 problems 0:300, so that more requests are in the engine than it runs at once.
# (ASSUME["smoke_problems"] stays the spec's 20: it is the §9.7 R0-launch row, reproduced by the offline gate.)
SMOKE_PROBLEMS_A22 = 300
# A22: r_full counts the engine steps with at least CONC_FULL running sequences. 128 is the n13 engine's own limit: LLMEngine.from_engine_args
# uses UsageContext.ENGINE_CONTEXT, which is not in vLLM 0.32.0's default table, so max_num_seqs = SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128
# (vllm/engine/arg_utils.py lines 3123-3127, vllm/config/scheduler.py line 43, at v0.32.0). n13_gen records the effective value.
CONC_FULL = 128
CAPS = (8192, 16384)                                   # the two values of C (§2.3)


def gen_minutes(n_math: int, n_gsm: int, tok_s: float, mean_math: float, mean_gsm: float, load_min: float) -> float:
    return load_min + (n_math * mean_math + n_gsm * mean_gsm) / tok_s / 60.0


def arm_load(arm: str, A: dict) -> float:
    """Fixed minutes per engine load. A22: built arms (every arm but base and rl) add A["build_min"] for building and writing the
    14.6 GB arm, which no smoke measures (absent from ASSUME, so the §9.7 rows are unchanged)."""
    return A["load_min"] + (0.0 if arm in ("base", "rl") else A.get("build_min", 0.0))


def project_launch(kind: str, tok_s: float, A: dict = ASSUME, smoke_mean=None, max_priority: int = 99, only=None) -> float:
    """Projected minutes of one launch (instance uptime), from the throughput and the mean lengths."""
    mm, mg, ld = A["mean_math"], A["mean_gsm"], A["load_min"]
    if kind == "r0":                                   # setup, R0 (olmes), smoke (base and rl, 20 problems each)
        sm = smoke_mean if smoke_mean is not None else mm
        return (A["setup_min"] + A["olmes_venv_min"] + A["olmes_load_min"] + A["r0_tokens"] / tok_s / 60.0
                + 2 * gen_minutes(A["smoke_problems"], 0, tok_s, sm, 0, ld))
    if kind == "s2":                                   # the spec's S2: the r0 launch plus R1
        return project_launch("r0", tok_s, A, smoke_mean) + A["r1_min"]
    if kind == "s3gate":
        return A["setup_min"] + sum(gen_minutes(N_MATH, N_GSM, tok_s, mm, mg, arm_load(a, A)) for a, _, _ in GATE)
    if kind == "s3gate_r1":                            # the S3 launch when R1 moved out of the R0 launch
        return project_launch("s3gate", tok_s, A) + A["r1_min"]
    if kind in ("s4armsA", "s4armsB"):
        arms = [x for x in S4[kind[-1]] if x[1] <= max_priority]
        return A["setup_min"] + sum(gen_minutes(N_MATH, N_GSM if g else 0, tok_s, mm, mg, arm_load(a, A)) for a, _, g in arms)
    if kind in S3_SPLIT:                               # A28: setup (+ R1 on s3gate1) and two gate arms, each with GSM8K
        return A["setup_min"] + (A["r1_min"] if kind == "s3gate1" else 0.0) + sum(
            gen_minutes(N_MATH, N_GSM, tok_s, mm, mg, arm_load(a, A)) for a, _, _ in S3_SPLIT[kind] if only is None or a in only)
    if kind in S4_SPLIT:                               # A28
        arms = [x for x in S4_SPLIT[kind] if x[1] <= max_priority]
        return A["setup_min"] + sum(gen_minutes(N_MATH, N_GSM if g else 0, tok_s, mm, mg, arm_load(a, A)) for a, _, g in arms)
    raise ValueError(kind)


def project_total(tok_s: float, A: dict = ASSUME, first: str = "s2") -> dict:
    gate = "s3gate" if first == "s2" else "s3gate_r1"
    parts = {k: project_launch(k, tok_s, A) / 60.0 for k in (first, gate, "s4armsA", "s4armsB")}
    return {"gpu_h": parts, "total_gpu_h": sum(parts.values()), "tok_s": tok_s}


# ------------------------------------------------------------------------------------------------ A22: rates from the engine-step log
# The log's class "wait" (unfinished > running) is the head's saturated phase (A22 addition, 2026-11-06): requests still waiting to be
# admitted (or preempted), waiting = (submitted - finished so far) - running > 0. Its rate is r_sat. Logs written before the addition
# (attempt 8) hold [seconds, tokens, t_last] per class; later logs add [run_dt, ctx_dt]. OLD_NAMES maps the attempt-8 throughput.json keys.
OLD_NAMES = {"r_wait": "r_sat", "wait_s": "sat_s", "wait_tokens": "sat_tokens", "tail_wait_s": "tail_sat_s"}
TAIL_OF = {"r_full": "tail_s", "r_sat": "tail_sat_s"}
WHY_UNDEFINED = {"r_full": f"no engine step had at least {CONC_FULL} running sequences",
                 "r_sat": "no engine step had requests waiting (unfinished > running)"}
RATES = ("min", "r_full", "r_sat")                     # "min" = min(r_full, r_sat), the rule (A22 addition)


def step_rates(rows: list) -> dict:
    """Sums over n13_gen's steps.jsonl rows (one row per block of engine steps; each block carries its sums over the steps with at
    least CONC_FULL running sequences, "full", and over the steps with requests waiting, "wait": unfinished > running).
    r_full = output tokens of the full steps / their wall seconds; tail_s = generation end - end of the last full step;
    r_all = all output tokens / generation wall seconds; r_sat and tail_sat_s are the same over the waiting steps. When the log holds
    run_dt and ctx_dt per class: <class>_run_mean = sum(running x dt) / seconds, <class>_ctx_mean = sum(context x dt) / sum(running x dt)
    (the time-weighted mean context per running sequence). A quantity with no step in its class is None."""
    gen_s = rows[-1]["t"] if rows else 0.0
    dt, tok = sum(r["dt"] for r in rows), sum(r["tok"] for r in rows)

    def part(key):
        xs = [r[key] for r in rows if r[key] is not None]
        widths = {len(x) for x in xs}
        assert widths <= {3} or widths == {5}, f"A22: mixed {key} formats in one log: {widths}"
        s, k = sum(x[0] for x in xs), sum(x[1] for x in xs)
        rd = sum(x[3] for x in xs) if widths == {5} else None
        cd = sum(x[4] for x in xs) if widths == {5} else None
        return ((k / s if s > 0 else None), s, k, (gen_s - max(x[2] for x in xs) if xs else None),
                (rd / s if rd is not None and s > 0 else None), (cd / rd if cd is not None and rd > 0 else None))

    r_full, full_s, full_tok, tail_s, full_run, full_ctx = part("full")
    r_sat, sat_s, sat_tok, tail_sat_s, sat_run, sat_ctx = part("wait")
    return {"conc_full": CONC_FULL, "steps": sum(r["n"] for r in rows), "rows": len(rows), "gen_s": gen_s, "dt_sum": dt,
            "output_tokens": tok, "r_all": tok / dt if dt > 0 else None, "r_full": r_full, "full_s": full_s, "full_tokens": full_tok,
            "tail_s": tail_s, "full_run_mean": full_run, "full_ctx_mean": full_ctx, "r_sat": r_sat, "sat_s": sat_s, "sat_tokens": sat_tok,
            "tail_sat_s": tail_sat_s, "sat_run_mean": sat_run, "sat_ctx_mean": sat_ctx,
            "run_max": max((r["run"][1] for r in rows), default=None),
            "run_mean_time_weighted": sum(r["run_dt"] for r in rows) / dt if dt > 0 else None}


def sat_phase(rows: list, answers: list, n: int) -> dict:
    """A22 addition: the saturated phase's mean running count and mean context, reconstructed for a log without run_dt and ctx_dt per
    class (attempt 8). answers: the n answers in finish order (generations.jsonl line order), each with prompt_n_tokens and n_generated.
    Finished so far at the end of block b = n - the block's smallest unfinished count (no request is added after the start, so the
    count only falls); the answers at finish-order positions [fin(b-1), fin(b)) finished in block b and are placed at its last step.
    Admission step a = f - n_generated + 1: one token per running sequence per step, the first at the prefill step. Context of a running
    sequence at step s = prompt + tokens generated through s. A step is saturated when (n - finished through s) - running(s) > 0. Steps
    in a block share its wall seconds equally. Limits: the finish step is known to within the block (10 steps); a preempted request's
    residency is longer than n_generated steps, which this does not see (the running-count cross-check below shows it)."""
    import numpy as np
    assert len(answers) == n, (len(answers), n)
    ends = np.array([r["i"] + r["n"] - 1 for r in rows])
    fin = n - np.array([r["unf"][0] for r in rows])
    assert fin[-1] == n and np.all(np.diff(fin) >= 0) and fin[0] >= 0, "A22: the unfinished counts do not fall from n to 0"
    f = ends[np.searchsorted(fin, np.arange(n), side="right")]
    L = np.array([x["n_generated"] for x in answers])
    Pl = np.array([x["prompt_n_tokens"] for x in answers])
    a = f - L + 1
    assert a.min() >= 0, "A22: an admission step before the first step (more than one token per step for some request)"
    S = int(ends[-1]) + 1
    cnt, base, fcnt = np.zeros(S + 1), np.zeros(S + 1), np.zeros(S + 1)
    np.add.at(cnt, a, 1)
    np.add.at(cnt, f + 1, -1)
    np.add.at(base, a, Pl - a + 1)
    np.add.at(base, f + 1, -(Pl - a + 1))
    np.add.at(fcnt, f, 1)
    s_idx = np.arange(S)
    R = np.cumsum(cnt)[:S]
    C = np.cumsum(base)[:S] + s_idx * R
    unf = n - np.cumsum(fcnt)[:S]
    sat = (unf - R) > 0
    dts = np.concatenate([np.full(r["n"], r["dt"] / r["n"]) for r in rows])
    blk = np.concatenate([np.full(r["n"], j) for j, r in enumerate(rows)])
    assert len(dts) == S
    lo, hi = np.array([r["run"][0] for r in rows])[blk], np.array([r["run"][1] for r in rows])[blk]
    sat_s = float(dts[sat].sum())
    log_sat_s = sum(r["wait"][0] for r in rows if r["wait"] is not None)
    return {"method": "reconstructed from the finish order and the per-block unfinished counts (n13_plan.sat_phase)",
            "sat_s": sat_s, "sat_run_mean": float((dts * R)[sat].sum() / sat_s) if sat_s > 0 else None,
            "sat_ctx_mean": float((dts * C)[sat].sum() / (dts * R)[sat].sum()) if sat_s > 0 else None,
            "check": {"log_sat_s": log_sat_s, "share_of_steps_running_within_block_range": float(np.mean((R >= lo) & (R <= hi))),
                      "run_mean_time_weighted": float((dts * R).sum() / dts.sum()),
                      "log_run_mean_time_weighted": sum(r["run_dt"] for r in rows) / sum(r["dt"] for r in rows),
                      "tokens": int(L.sum()), "log_tokens": sum(r["tok"] for r in rows)}}


def read_jsonl(path) -> list:
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]


def pick(sr: dict, rate: str):
    """The rate and its tail for one smoke; None when undefined. "min" takes the smaller of r_full and r_sat with its own tail."""
    k = (("r_sat" if sr["r_sat"] <= sr["r_full"] else "r_full") if sr["r_full"] is not None and sr["r_sat"] is not None else None) \
        if rate == "min" else rate
    return None if k is None or sr[k] is None else {"rate": sr[k], "from": k, "tail_s": sr[TAIL_OF[k]]}


def launch_table(per: dict, cap: int, A: dict) -> dict:
    """The §9.7 arithmetic for S3 and S4 at one rate. per = {m: {rate, tail_s, mean_math, load_s}}; the smoke with the larger
    mean_math / rate sets r, L_math and F = load_s + tail_s for every arm."""
    ms = max(per, key=lambda m: per[m]["mean_math"] / per[m]["rate"])
    p = per[ms]
    B = dict(A, mean_math=p["mean_math"], load_min=(p["load_s"] + p["tail_s"]) / 60.0, build_min=A["load_min"])
    r = p["rate"]
    s3 = project_launch("s3gate_r1", r, B)
    by_p = [{"through_priority": q, "gpu_h": sum(project_launch(k, r, B, max_priority=q) for k in ("s4armsA", "s4armsB")) / 60.0}
            for q in range(1, 8)]
    launches = {k: {"min": project_launch(k, r, B), "gpu_h": project_launch(k, r, B) / 60.0, "allowance_min": budget_min(project_launch(k, r, B)),
                    "arms": [x[0] for x in (S3_SPLIT.get(k) or S4_SPLIT[k])]} for k in LAUNCHES}
    by_p4 = [{"through_priority": q, "gpu_h": sum(project_launch(k, r, B, max_priority=q) for k in S4_SPLIT) / 60.0} for q in range(1, 8)]
    return {"sets_the_rate": ms, "rate_from": p["from"], "tok_s": r, "assumptions": B, "s3_min": s3, "s3_gpu_h": s3 / 60.0,
            "s3_allowance_min": budget_min(s3), "s4_gpu_h": {k: project_launch(k, r, B) / 60.0 for k in ("s4armsA", "s4armsB")},
            "s4_by_priority": by_p, "launches": launches, "s4_by_priority_four_launches": by_p4,
            "s3_two_launches_gpu_h": sum(launches[k]["gpu_h"] for k in S3_SPLIT), "s4_four_launches_gpu_h": sum(launches[k]["gpu_h"] for k in S4_SPLIT)}


KILLED_READINGS = ("prefix", "all")


def killed_smoke(rows: list, answers: list, n: int, load_s: float, cap: int, other: dict, reading: str = "prefix") -> dict:
    """A24 (N13_SPEC.md; the head's brief of 2026-11-06): a smoke cut by the hard kill has no throughput.json. Its step log (flushed
    per line) and its finished answers (generations.jsonl, finish order) are real; the rest did not finish. Requests are submitted in
    problem order and the scheduler admits first-come first-served, so every problem up to the largest finished index was admitted.
    reading "prefix" (the rule): L at C is the mean over that admitted prefix, each finished answer at min(n_generated, C) and each
    prefix answer cut by the kill at C (it had not stopped, and C bounds it: an upper bound); the problems after the prefix are left
    out. reading "all": every unfinished answer of the n at C, also those waiting at the kill (the brief's words read literally; an
    upper bound that counts answers with few or no tokens at C). The rate and its tails are the other smoke's (flagged): the cut log has no
    tail. The cut log's own rates are recorded beside (own_steps)."""
    assert reading in KILLED_READINGS, reading
    sr = step_rates(rows)
    fin = sum(x["n_generated"] for x in answers)
    assert len(answers) < n and sr["output_tokens"] >= fin, ("A24: not a cut smoke", len(answers), n, sr["output_tokens"], fin)
    clip = sum(min(x["n_generated"], cap) for x in answers)
    n_pre = max(x["problem_index"] for x in answers) + 1
    u_pre, u = n_pre - len(answers), n - len(answers)
    means = {"prefix": (clip + u_pre * cap) / n_pre, "all": (clip + u * cap) / n}
    return {"steps": other["steps"], "mean_math": means[reading], "load_s": load_s, "sat_phase": {**{k: other["sat_phase"][k] for k in
            ("sat_run_mean", "sat_ctx_mean")}, "from": "the other smoke (A24)"}, "sat_phase_reconstructed": None,
            "killed": {"reading": reading, "n_submitted": n, "n_finished": len(answers), "n_unfinished": u, "prefix_n": n_pre,
                       "prefix_unfinished": u_pre, "mean_at_cap": means, "finished_tokens": fin,
                       "partial_tokens_of_unfinished": sr["output_tokens"] - fin,
                       "prefix_mean_if_every_partial_token_is_in_the_prefix": (clip + sr["output_tokens"] - fin) / n_pre,
                       "own_steps": sr}}


def project_smoke(smokes: dict, cap: int, A: dict = ASSUME, rate: str = "min", killed_reading: str = "prefix") -> dict:
    """A22: the S3 and S4 projection from the two smokes. smokes = {"base": (throughput.json, steps rows, answers or None), "rl": ...}.
    Per smoke m: its rate (the rule: min(r_full, r_sat), A22 addition), L_m = its mean output tokens per MATH-500 answer at the cap C
    (each answer's n_generated clipped at C), F_m = load_s + the chosen rate's tail (seconds per engine load). The smoke with the larger
    L_m / r_m sets r, L_math and F for every arm. L_gsm stays §9.7's 940 (not measured). Built arms add §9.7's 5 min for the build (not
    measured). Then the §9.7 arithmetic (launch_table), at the rule's rate and at r_full and r_sat alone (by_rate). Refuses
    (SystemExit, a message) when the rule's rate is undefined for a smoke."""
    assert rate in RATES, rate
    per = {}
    for m in ("base", "rl"):
        tp, rows, answers = smokes[m][:3]
        if tp is None:                                 # A24: cut by the hard kill; built from the other smoke below
            continue
        sr = step_rates(rows)
        diff = {k: (v, sr[OLD_NAMES.get(k, k)]) for k, v in tp["steps"].items() if sr[OLD_NAMES.get(k, k)] != v}
        assert not diff, f"A22: the {m} steps log does not recount to its throughput.json: {diff}"
        assert sr["output_tokens"] == tp["output_tokens"], (f"A22: {m}: steps-log tokens differ from the answers' tokens",
                                                           sr["output_tokens"], tp["output_tokens"])
        assert tp["steps"]["conc_full"] == CONC_FULL, tp["steps"]["conc_full"]
        rec = sat_phase(rows, answers, tp["n_answers"]) if answers is not None else None
        sat = {"sat_run_mean": sr["sat_run_mean"], "sat_ctx_mean": sr["sat_ctx_mean"], "from": "the step log"} \
            if sr["sat_run_mean"] is not None or sr["sat_s"] == 0 else None
        if sat is None:
            assert rec is not None, f"A22 addition: the {m} log has no run_dt or ctx_dt per class; the saturated-phase means need {m}'s answers"
            sat = {"sat_run_mean": rec["sat_run_mean"], "sat_ctx_mean": rec["sat_ctx_mean"], "from": rec["method"]}
        per[m] = {"steps": sr, "mean_math": tp["mean_output_tokens_at"][str(cap)], "load_s": tp["load_s"], "sat_phase": sat,
                  "sat_phase_reconstructed": rec}
    for m in ("base", "rl"):
        if smokes[m][0] is None:
            other = next(x for x in per if x != m)
            per[m] = killed_smoke(smokes[m][1], smokes[m][2], smokes[m][3], smokes[m][4], cap, per[other], killed_reading)
    tables = {}
    for rt in RATES:
        picks = {m: pick(per[m]["steps"], rt) for m in per}
        tables[rt] = None if any(v is None for v in picks.values()) else launch_table(
            {m: {**picks[m], "mean_math": per[m]["mean_math"], "load_s": per[m]["load_s"]} for m in per}, cap, A)
    if tables[rate] is None:
        need = ("r_full", "r_sat") if rate == "min" else (rate,)
        m, k = next((m, k) for m in per for k in need if per[m]["steps"][k] is None)
        raise SystemExit(f"A22: the {m} smoke's {k} is undefined: {WHY_UNDEFINED[k]} (largest running count "
                         f"{per[m]['steps']['run_max']}); the projection refuses")
    return {"amendment": "N13_SPEC.md A22 and its addition of 2026-11-06", "rate": rate, "cap": cap,
            "per_smoke": {m: {"r_full": v["steps"]["r_full"], "r_sat": v["steps"]["r_sat"], "r_all": v["steps"]["r_all"],
                              "tail_s": v["steps"]["tail_s"], "tail_sat_s": v["steps"]["tail_sat_s"], "full_s": v["steps"]["full_s"],
                              "sat_s": v["steps"]["sat_s"], "gen_s": v["steps"]["gen_s"], "mean_math": v["mean_math"], "load_s": v["load_s"],
                              **{k: v["sat_phase"][k] for k in ("sat_run_mean", "sat_ctx_mean")}, "sat_phase_from": v["sat_phase"]["from"],
                              "sat_phase_reconstructed": v["sat_phase_reconstructed"], "killed": v.get("killed")} for m, v in per.items()},
            "flags": ["mean_gsm is the spec's 940 (not measured)", "build_min is the spec's 5 min per built arm (not measured)",
                      "setup_min 20 and r1_min 75 are the spec's (A14)"]
            + [f"A24: the {m} smoke was cut by the hard kill: its rate and tails are the other smoke's; its L is read '{v['killed']['reading']}' "
               f"({v['killed']['prefix_unfinished'] if v['killed']['reading'] == 'prefix' else v['killed']['n_unfinished']} unfinished "
               f"answers at C over {v['killed']['prefix_n'] if v['killed']['reading'] == 'prefix' else v['killed']['n_submitted']})"
               for m, v in per.items() if v.get("killed")],
            **tables[rate], "by_rate": tables}


def budget_min(projected_min: float) -> int:
    return int(round(1.3 * projected_min))


# ------------------------------------------------------------------------------------------------ the stop rule inside a launch
def launch_t0(out_dir) -> float:
    """The job's start on the instance: started_utc in the pipeline's out/pipeline_status.json, which infra/aws_pg/jobs/pipeline.py
    takes when its process starts and writes at its first write. run_job.sh starts that process under `timeout --signal=KILL
    <budget>s`, so this is the hard kill's clock, to within the interpreter's start-up (later) and the truncation to whole seconds
    (earlier). A missing started_utc stops the step (KeyError). Outside the pipeline (the offline gate) there is no
    pipeline_status.json and the step's out dir mtime is used.
    Until the A19 revision (N13_SPEC.md, 2026-11-05) this read the mtime of out/steps.json, which `aws s3 cp` sets to the object's
    upload time: 9.0 min before the launch in s2setup-06, and more with every launch retry."""
    out = pathlib.Path(out_dir)
    s = out.parent / "pipeline_status.json"
    if s.exists():
        return float(calendar.timegm(time.strptime(json.loads(s.read_text())["started_utc"], "%Y-%m-%dT%H:%M:%SZ")))
    return out.stat().st_mtime


def park_check(out_dir, allowance_min, projected_min: float, now=None) -> dict:
    """PARKED when the launch's elapsed time plus this step's projection exceeds the allowance (§9.7). A parked step parks every later one."""
    out = pathlib.Path(out_dir)
    marker = out.parent / "PARKED"
    now = time.time() if now is None else now
    elapsed = (now - launch_t0(out)) / 60.0
    v = {"allowance_min": allowance_min, "elapsed_min": elapsed, "projected_min": projected_min, "earlier_parked": marker.exists()}
    v["parked"] = bool(allowance_min is not None and (marker.exists() or elapsed + projected_min > allowance_min))
    if v["parked"]:
        marker.write_text(json.dumps(v))
    return v


# ------------------------------------------------------------------------------------------------ manifests
def manifest(kind: str, nn: int, A: dict = ASSUME, allowance_min=None, tok_s=None, mean_math=None, cap: int = 16384, only=None) -> list:
    """steps.json for one launch: a list of {name, module, args} that infra/aws_pg/jobs/pipeline.py runs in order."""
    W = "/work/out"
    tok_s = tok_s or A["tok_s"][1]
    mm = mean_math or A["mean_math"]
    al = f" --allowance-min {allowance_min}" if allowance_min is not None else ""

    def pr(arm):                                       # the step's own park projection (§9.7); A22: built arms add build_min
        return f" --proj-tok-s {tok_s} --proj-mean-math {mm} --proj-mean-gsm {A['mean_gsm']} --proj-load-min {arm_load(arm, A)}"

    def smoke(n):
        return [{"name": f"smoke_{arm}", "module": "infra.aws_pg.jobs.n13_gen",
                 "args": f"--arm {arm} --bench math500 --problems 0:{n} --cap 16384 --smoke --setup-dir {W}/setup{al}{pr(arm)}"}
                for arm in ("base", "rl")]
    if kind == "s2setup":
        return [
            {"name": "setup", "module": "infra.aws_pg.jobs.n13_setup", "args": "--pair pair1 --models base,rl --olmes"},
            {"name": "r0_known", "module": "infra.aws_pg.jobs.n13_known", "args": f"--setup-dir {W}/setup{al}"},   # A19: measured gate
        ] + smoke(A["smoke_problems"])
    if kind == "s2smoke":                              # R0 is banked (s2setup-07, G-KNOWN passed): setup without the olmes venv, the smoke
        return [{"name": "setup", "module": "infra.aws_pg.jobs.n13_setup", "args": "--pair pair1 --models base,rl"}] + smoke(SMOKE_PROBLEMS_A22)
    inputs = f"s3://{BUCKET}/{prefix(kind, nn)}/inputs"
    if kind in S3_SPLIT:                               # A28: s3gate1 holds R1 (A14), s3gate2 the built gate arms; C is decided from
        steps = [{"name": "setup", "module": "infra.aws_pg.jobs.n13_setup", "args": "--pair pair1 --models base,rl,stage2"},
                 {"name": "r1_rank", "module": "infra.aws_pg.jobs.n13_rank", "args": f"--setup-dir {W}/setup --delete-stage2"}] \
            if kind == "s3gate1" else [{"name": "setup", "module": "infra.aws_pg.jobs.n13_setup", "args": "--pair pair1 --models base,rl"}]
        g = f"--setup-dir {W}/setup --cap {cap}{al}"   # the smoke (§2.3) and passed here; no gate arm reads sets.json (A28)
        return steps + [{"name": a, "module": "infra.aws_pg.jobs.n13_gen",
                         "args": f"--arm {a} --bench math500,gsm8k --problems 0:500 {g}{pr(a)}"} for a, _, _ in S3_SPLIT[kind]
                        if only is None or a in only]
    if kind in S4_SPLIT:
        steps = [{"name": "setup", "module": "infra.aws_pg.jobs.n13_setup", "args": "--pair pair1 --models base,rl"}]
        for a, _, g in S4_SPLIT[kind]:
            b = "math500,gsm8k" if g else "math500"
            cond = f" --only-if {inputs}/controls.json" if a in CONDITIONAL else ""
            steps.append({"name": a, "module": "infra.aws_pg.jobs.n13_gen", "args": f"--arm {a} --bench {b} --problems 0:500 --sets "
                          f"{inputs}/sets.json --setup-dir {W}/setup --cap-from {inputs}/thresholds.json{al}{pr(a)}{cond}"})
        return steps
    raise ValueError(kind)


# A28: the S3 launches keep the head's prefixes (results/next_n13/s3gate-01 and s3gate-02); the S4 launches take prefix(kind, 1).
MANIFEST_FILE = {"s3gate1": "n13_s3gate-01.json", "s3gate2": "n13_s3gate-02.json", **{k: f"n13_{k}-01.json" for k in S4_SPLIT}}


def check_manifest(steps: list) -> dict:
    """The assertions of pipeline.py's _read_steps, plus: each arm once, every module importable by name."""
    assert isinstance(steps, list) and steps, steps
    for s in steps:
        assert {"name", "module"} <= set(s), s
    names = [s["name"] for s in steps]
    return {"n": len(steps), "names": names, "distinct": len(set(names)) == len(names), "modules": sorted({s["module"] for s in steps})}


# ------------------------------------------------------------------------------------------------ A22: the laptop projection command
def main(argv=None):
    ap = argparse.ArgumentParser(description="A22: the S3 and S4 projection from the smoke; writes the S3 manifest when it fits")
    ap.add_argument("--smoke-dir", required=True, help="holds smoke_base/ and smoke_rl/, each with throughput.json and steps.jsonl "
                    "(and lengths.json, the answers in finish order, for a log without per-class run_dt and ctx_dt)")
    ap.add_argument("--cap", type=int, required=True, choices=CAPS)
    ap.add_argument("--spent", type=float, required=True, help="GPU-h spent by N13 before S3 (the operations note; recorded, no ceiling: A27)")
    ap.add_argument("--rate", default="min", choices=RATES, help="the rule is min(r_full, r_sat) (A22 addition); the others are shown")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest-dir", default=None, help="A28: writes n13_s3gate-01.json (s3gate1), n13_s3gate-02.json (s3gate2) and "
                    "n13_<kind>-01.json for the four S4 launches, each at allowance 1.3 x its projection (A27)")
    ap.add_argument("--killed", default=None, choices=("base", "rl"), help="A24: this smoke was cut by the hard kill (no throughput.json; "
                    "lengths.json holds its finished answers and n_submitted; load_s from its metrics.json)")
    ap.add_argument("--killed-reading", default="prefix", choices=KILLED_READINGS, help="A24: the cut smoke's L over its admitted prefix "
                    "(the rule) or with every unfinished answer at C")
    ap.add_argument("--r1-min", type=float, default=ASSUME["r1_min"], help="R1's minutes in the s3gate1 projection (the spec's 75; the "
                    "relaunch after A29 adds stage 6, measured at s3gate-01)")
    ap.add_argument("--relaunch", nargs="*", default=None, help="S3 relaunch manifests instead of the six of --manifest-dir: entries "
                    "kind=file or kind=file:arm,arm (only those gate arms), e.g. s3gate1=n13_s3gate-03.json s3gate2=n13_s3gate-04.json:id; "
                    "each at allowance round(1.3 x its own projection) (A27)")
    a = ap.parse_args(argv)
    d = pathlib.Path(a.smoke_dir)
    smokes = {}
    for m in ("base", "rl"):
        so = d / f"smoke_{m}"
        ln = so / "lengths.json"
        answers = None
        if ln.exists():
            L = json.loads(ln.read_text())
            assert L["order"] == "finish", f"{ln}: the answers must be in finish order (generations.jsonl line order)"
            answers = L["answers"]
        if m == a.killed:
            assert not (so / "throughput.json").exists() and answers is not None, f"A24: {so} is not a cut smoke with lengths.json"
            load_s = json.loads((so / m / "math500/metrics.json").read_text())["engine"]["load_s"]
            smokes[m] = (None, read_jsonl(so / "steps.jsonl"), answers, L["n_submitted"], load_s)
            continue
        smokes[m] = (json.loads((so / "throughput.json").read_text()), read_jsonl(so / "steps.jsonl"), answers)
    pj = project_smoke(smokes, a.cap, A=dict(ASSUME, r1_min=a.r1_min), rate=a.rate, killed_reading=a.killed_reading)
    for t in [pj] + [v for v in pj["by_rate"].values() if v is not None]:
        t["cumulative_after_s3_gpu_h"] = a.spent + t["s3_two_launches_gpu_h"]
        t["cumulative_after_s4_gpu_h"] = a.spent + t["s3_two_launches_gpu_h"] + t["s4_four_launches_gpu_h"]
    pj.update({"spent_gpu_h": a.spent, "ceiling_gpu_h": None, "ceiling": "none (A27)"})
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(pj, indent=1) + "\n")
    keys = ("rate_from", "sets_the_rate", "tok_s", "s3_two_launches_gpu_h", "s4_four_launches_gpu_h", "cumulative_after_s4_gpu_h", "launches")
    print(json.dumps({"rule": a.rate, "cap": a.cap, "spent_gpu_h": a.spent, "per_smoke": {m: {k: v[k] for k in ("r_full", "r_sat", "r_all",
                      "sat_run_mean", "sat_ctx_mean", "mean_math")} for m, v in pj["per_smoke"].items()},
                      "by_rate": {r: (None if t is None else {k: t[k] for k in keys}) for r, t in pj["by_rate"].items()}}, indent=1))
    if a.relaunch is not None:
        assert a.manifest_dir, "--relaunch writes into --manifest-dir"
        B, pj["relaunch"] = pj["assumptions"], {}
        for e in a.relaunch:
            k, f = e.split("=")
            f, only = (f.split(":")[0], tuple(f.split(":")[1].split(","))) if ":" in f else (f, None)
            assert k in S3_SPLIT and (only is None or set(only) <= {x[0] for x in S3_SPLIT[k]}), e
            mn = project_launch(k, pj["tok_s"], B, only=only)
            steps = manifest(k, 1, B, allowance_min=budget_min(mn), tok_s=pj["tok_s"], mean_math=B["mean_math"], cap=a.cap, only=only)
            (pathlib.Path(a.manifest_dir) / f).write_text(json.dumps(steps, indent=1) + "\n")
            pj["relaunch"][f] = {"kind": k, "only": only, "min": mn, "gpu_h": mn / 60.0, "allowance_min": budget_min(mn),
                                 "r1_min": B["r1_min"] if k == "s3gate1" else None, "steps": [x["name"] for x in steps]}
            print(f"wrote {f}: {[x['name'] for x in steps]}, projection {mn:.1f} min, allowance {budget_min(mn)} min")
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(pj, indent=1) + "\n")
    elif a.manifest_dir:
        B = pj["assumptions"]
        for k, v in pj["launches"].items():
            steps = manifest(k, 1, B, allowance_min=v["allowance_min"], tok_s=pj["tok_s"], mean_math=B["mean_math"], cap=a.cap)
            f = pathlib.Path(a.manifest_dir) / MANIFEST_FILE[k]
            f.write_text(json.dumps(steps, indent=1) + "\n")
            print(f"wrote {f}: {len(steps)} steps, cap {a.cap}, projection {v['min']:.1f} min, allowance {v['allowance_min']} min, rate "
                  f"{pj['tok_s']:.1f} ({pj['rate_from']}, smoke {pj['sets_the_rate']})")


if __name__ == "__main__":
    main()
