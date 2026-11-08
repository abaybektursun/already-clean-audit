"""Attempt 9 (s2smoke-09) smoke reading, from the files copied from S3 after the job ended (N13_SPEC.md §9.4 S2 close-out, §2.3 + A17
+ A22 item 1, A22 prediction, A24 projection inputs).

    python3 studies/next/n13/r0/s2smoke-09/analysis.py --raw <copy of s3://.../s2smoke-09/out> --bank studies/next/n13/r0/s2smoke-09
    (then the projection command of STATE.md, then the same line with --estimates for projection_estimates.json)

Writes into --bank: smoke_<m>/lengths.json (the finished answers in finish order = generations.jsonl line order, and the unfinished
problem indexes), read_answers.json (10 answers per model, problems 0-9, full text), smoke_analysis.json (the C rule, the lengths with
the censoring stated, the A22 prediction, the saturated-phase context estimate). Fail-loud: no try/except.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

N, CAPS, RULE_N, RULE_T = 300, (8192, 16384), 20, 8192
KEEP = ("problem_index", "n_new", "n_generated", "prompt_n_tokens", "cap_hit", "finish_reason", "mode", "score")


def jl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def ctx_estimate(prompts, lengths) -> float:
    """Time-weighted mean context per running sequence when each answer is resident for its length: a sequence of length L with
    prompt p is resident for L steps at mean context p + (L + 1) / 2, so the mean is sum L (p + (L + 1) / 2) / sum L. It ignores
    the opening (all start together) and the tail; it is checked against the logged sat_ctx_mean at the measured cap."""
    p, L = np.asarray(prompts, float), np.asarray(lengths, float)
    return float((L * (p + (L + 1) / 2)).sum() / L.sum())


def one(raw: pathlib.Path, bank: pathlib.Path, m: str) -> dict:
    so = raw / f"smoke_{m}"
    recs = jl(so / m / "math500/generations.jsonl")
    met = json.loads((so / m / "math500/metrics.json").read_text())
    rows = jl(so / "steps.jsonl")
    tp_path = so / "throughput.json"
    idx = [r["problem_index"] for r in recs]
    assert len(set(idx)) == len(idx) and set(idx) <= set(range(N)), "duplicate or foreign problem index"
    unf = sorted(set(range(N)) - set(idx))
    log_tok = sum(r["tok"] for r in rows)
    fin_tok = sum(r["n_generated"] for r in recs)
    assert log_tok >= fin_tok, (m, log_tok, fin_tok)
    (bank / f"smoke_{m}").mkdir(parents=True, exist_ok=True)
    lengths = {"order": "finish", "n_submitted": N, "n_finished": len(recs), "unfinished_problem_index": unf,
               "partial_tokens_of_unfinished": log_tok - fin_tok if unf else 0,
               "note": "finished answers in generations.jsonl line order; the unfinished were cut by the hard kill (censored)",
               "answers": [{k: r[k] for k in KEEP} for r in recs]}
    (bank / f"smoke_{m}/lengths.json").write_text(json.dumps(lengths, indent=0) + "\n")
    by = {r["problem_index"]: r for r in recs}
    first = [by.get(i) for i in range(RULE_N)]
    n_new = np.array([r["n_new"] for r in recs])
    n_gen = np.array([r["n_generated"] for r in recs])
    u = len(unf)
    out = {"n_finished": len(recs), "n_unfinished": u, "unfinished_problem_index": unf, "throughput_json": tp_path.exists(),
           "status": met["status"], "counts": met.get("counts"), "engine": {k: met["engine"].get(k) for k in
           ("load_s", "max_num_seqs", "max_num_batched_tokens", "num_gpu_blocks", "env", "attention_backend", "vllm")},
           "score_finished": float(np.mean([r["score"] for r in recs])),
           "modes_finished": {k: sum(r["mode"] == k for r in recs) for k in ("answer_line", "boxed", "none")},
           "finish_reason": {k: sum(r["finish_reason"] == k for r in recs) for k in sorted({r["finish_reason"] for r in recs})},
           "cap_hits_finished": int(sum(r["cap_hit"] for r in recs)),
           "log_tokens": log_tok, "finished_tokens": fin_tok, "partial_tokens_of_unfinished": log_tok - fin_tok,
           "finished_n_new": {"mean": float(n_new.mean()), "median": float(np.median(n_new)), "p95": float(np.percentile(n_new, 95)),
                              "max": int(n_new.max())},
           "rule_problems_0_19": {"finished": sum(r is not None for r in first),
                                  "n_new": [None if r is None else r["n_new"] for r in first],
                                  "longer_than_8192": sum(r is not None and r["n_new"] > RULE_T for r in first),
                                  "unfinished": [i for i, r in enumerate(first) if r is None]},
           "over_300_longer_than_8192": {"finished": int((n_new > RULE_T).sum()), "upper_bound_with_unfinished": int((n_new > RULE_T).sum()) + u},
           "mean_at_cap": {}}
    for c in CAPS:
        clip_new, clip_gen = np.minimum(n_new, c).sum(), np.minimum(n_gen, c).sum()
        out["mean_at_cap"][str(c)] = {
            "n_new_unfinished_at_C": float((clip_new + u * c) / N), "n_generated_unfinished_at_C": float((clip_gen + u * c) / N),
            "n_generated_unfinished_at_kill_length": float((clip_gen + (log_tok - fin_tok)) / N) if c == max(CAPS) else None,
            "finished_only_n_generated": float(clip_gen / len(recs))}
    prompts = [r["prompt_n_tokens"] for r in recs]
    out["ctx_estimate"] = {str(c): ctx_estimate(prompts, np.minimum(n_gen, c)) for c in CAPS} if not u else None
    from infra.pg import n13_plan as P
    sr = P.step_rates(rows)
    out["steps"] = sr
    full_steps = sr["full_tokens"] / 128 if sr["full_tokens"] else 0
    out["A22_prediction"] = {"full_s_over_gen_s": sr["full_s"] / sr["gen_s"], "r_full_over_r_sat": (sr["r_full"] / sr["r_sat"])
                             if sr["r_full"] and sr["r_sat"] else None, "steps_at_128_running": full_steps,
                             "last_full_step_end_s": sr["gen_s"] - sr["tail_s"] if sr["tail_s"] is not None else None}
    if tp_path.exists():
        tp = json.loads(tp_path.read_text())
        assert P.step_rates(rows) == tp["steps"], f"{m}: the steps log does not recount to throughput.json"
        assert log_tok == tp["output_tokens"] and len(recs) == tp["n_answers"] == N
    return out


def read10(raw: pathlib.Path, m: str) -> list:
    recs = {r["problem_index"]: r for r in jl(raw / f"smoke_{m}" / m / "math500/generations.jsonl")}
    return [{k: recs[i][k] for k in ("problem_index", "unique_id", "seed", "prompt", "prompt_n_tokens", "prompt_first20", "prompt_last20",
                                      "text", "n_new", "n_generated", "cap_hit", "finish_reason", "mode", "extracted", "gold", "parsed_answer",
                                      "parsed_gold", "score", "answer_prob", "answer_tokens")} for i in range(10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--estimates", action="store_true", help="after the projection command: write projection_estimates.json")
    a = ap.parse_args()
    raw, bank = pathlib.Path(a.raw), pathlib.Path(a.bank)
    if a.estimates:
        estimates(bank)
        return
    res = {m: one(raw, bank, m) for m in ("base", "rl")}
    (bank / "read_answers.json").write_text(json.dumps({m: read10(raw, m) for m in ("base", "rl")}, indent=1) + "\n")
    (bank / "smoke_analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({m: {k: v[k] for k in ("n_finished", "n_unfinished", "score_finished", "rule_problems_0_19", "over_300_longer_than_8192",
                                            "mean_at_cap", "ctx_estimate", "A22_prediction")} for m, v in res.items()}, indent=1))



def estimates(bank: pathlib.Path) -> dict:
    """Rows beside the projection command's (projection.json, the A24 prefix reading), marked as estimates: (1) L at the lower bound of
    the rl prefix (each cut answer at its length at the kill, all partial tokens assumed in the prefix); (2) the C rule's context
    scaling (N13 brief: rate x 5,513 / mean context at the chosen C; C = 16,384 is the smoke's cap, so the factor is 1); (3) the same
    scaling with the rl prefix's context estimate (ctx_estimate, cut answers at C) against the base's by the same estimator."""
    from infra.pg import n13_plan as P
    pj = json.loads((bank / "projection.json").read_text())
    an = json.loads((bank / "smoke_analysis.json").read_text())
    base = json.loads((bank / "smoke_base/lengths.json").read_text())["answers"]
    rl = json.loads((bank / "smoke_rl/lengths.json").read_text())
    pr = {a["problem_index"]: a["prompt_n_tokens"] for a in base}
    k = pj["per_smoke"]["rl"]["killed"]
    cut = [i for i in rl["unfinished_problem_index"] if i < k["prefix_n"]]
    cap = pj["cap"]
    ctx_rl = ctx_estimate([a["prompt_n_tokens"] for a in rl["answers"]] + [pr[i] for i in cut],
                          [min(a["n_generated"], cap) for a in rl["answers"]] + [cap] * len(cut))
    ctx_base = an["base"]["ctx_estimate"][str(cap)]
    A = {kk: v for kk, v in P.ASSUME.items()}
    per = {m: {"rate": pj["tok_s"], "from": pj["rate_from"], "tail_s": pj["per_smoke"]["base"]["tail_sat_s"],
               "mean_math": pj["per_smoke"][m]["mean_math"], "load_s": pj["per_smoke"][m]["load_s"]} for m in ("base", "rl")}
    lower = dict(per, rl=dict(per["rl"], mean_math=k["prefix_mean_if_every_partial_token_is_in_the_prefix"]))
    sat_ctx = pj["per_smoke"]["base"]["sat_ctx_mean"]
    f_rl = ctx_base / ctx_rl
    rows = {"lower_L": P.launch_table(lower, cap, A),
            "rate_scaled_C_rule": {"factor": sat_ctx / sat_ctx, "note": "C = 16,384 equals the smoke's cap: 5,513 against 5,513; the table is projection.json's"},
            "rate_scaled_rl_context": dict(P.launch_table({m: dict(v, rate=v["rate"] * f_rl) for m, v in per.items()}, cap, A),
                                           factor=f_rl, ctx_estimate_rl_prefix=ctx_rl, ctx_estimate_base=ctx_base)}
    spent = pj["spent_gpu_h"]
    for key in ("lower_L", "rate_scaled_rl_context"):
        t = rows[key]
        t["left_for_s4_gpu_h"] = P.CEILING_GPU_H - spent - t["s3_gpu_h"]
        t["s4_fits_through_priority"] = max([x["through_priority"] for x in t["s4_by_priority"] if x["gpu_h"] <= t["left_for_s4_gpu_h"]], default=0)
    out = {"estimate": True, "spent_gpu_h": spent, "rows": rows}
    (bank / "projection_estimates.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    main()
