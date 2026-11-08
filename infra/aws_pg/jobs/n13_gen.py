"""N13 generation step: one arm, one or two benchmarks, one engine load (studies/next/07_PICKS/N13_SPEC.md §2.2-2.5, §3.1-3.4, §6.3
damage screen, §10.1 samples).

    python -m infra.aws_pg.jobs.n13_gen --arm <arm> --bench math500[,gsm8k] --problems 0:500 --setup-dir <setup out>
        (--cap C | --cap-from thresholds.json) [--sets sets.json] [--smoke] [--allowance-min M --proj-tok-s T ...] --out <dir>

The pg venv parent fetches the S3 inputs, then re-executes this module under /opt/n13/bin/python (vLLM 0.32.0); the child does the work.
  1. Park check (§9.7): elapsed launch time plus this step's projection against the allowance; a parked step writes status "parked".
  2. Prompts: the RL tokenizer and template (§2.2); every rendered text asserted equal to the registered bytes, the MATH-500 ids equal
     to those banked by the setup step (G-TEMPLATE in every generation step).
  3. Weights: base and rl from the HF snapshots; every other arm built by n13_arms.build_arm into <arm-root>/<arm>/ and audited
     (build.json, written before the engine starts): G-HASH for zero and id, G-MASK for mask arms, DARE kept count, realized energy.
  4. The loop of §2.5: all requests submitted in order (MATH-500 rows, then GSM8K rows), per-problem seeds of the seed set; each finished
     answer scored and appended to <out>/<arm>/<bench>/generations.jsonl (flush + fsync); metrics.json and samples.jsonl rewritten every
     25 answers and at least every 60 s; the damage screen once MATH-500 problems 0-19 are done (abort all requests when it fires on an
     arm other than base, rl, zero, id).
  5. Final metrics: status done or screened, G-COMPLETE per benchmark, the throughput record (output tokens per second, seconds per
     problem, output-token histogram, GPU memory, host peak RSS, free disk); --smoke also writes <out>/throughput.json.
  A22 (N13_SPEC.md): every engine step's wall time, running sequences, unfinished requests and output tokens go into <out>/steps.jsonl,
     aggregated per STEP_BLOCK steps with the sums that keep r_full, r_all and tail_s exact (n13_plan.step_rates); the engine's load
     time (construction plus weight load) is written into metrics.json -> engine.load_s before the first request; the throughput
     record adds r_full, r_all, tail_s, load_s, the mean, median and maximum output length and the count at the cap. A22 addition
     (2026-11-06): r_sat over the steps with requests waiting (unfinished > running), and, per class, the time-weighted running count and
     context per running sequence (prompt + tokens so far), from the engine's per-step context sum.
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from infra.pg import n13_arms as A
from infra.pg import n13_plan as P
from infra.pg import n13_score as S

N13_PY = "/opt/n13/bin/python"
# A21 (N13_SPEC.md): vLLM 0.32.0's V2 sampler takes FlashInfer's top-k/top-p kernel whenever flashinfer-python imports, without checking
# for nvcc (vllm/v1/sample/ops/topk_topp_sampler.py lines 93-125 at v0.32.0); the sampler warmup's unseeded requests then JIT-compile it,
# and the AMI has no CUDA toolkit (s2setup-07 smoke_base: "Could not find nvcc"). "0" returns False at lines 95-100, so the sampler takes
# apply_top_k_top_p + gumbel_sample (vllm/v1/worker/gpu/sample/sampler.py lines 344-355), the path every seeded request takes anyway
# (lines 307-312: any explicit seed makes the fused sampler ineligible).
ENGINE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
# A21: vLLM's default on SM 8.9 (vllm/platforms/cuda.py line 174), pinned so that an invalid FLASH_ATTN raises (cuda.py lines 428-444)
# instead of falling back to FLASHINFER, which would JIT-compile. vLLM 0.32.0 has no VLLM_ATTENTION_BACKEND variable (vllm/envs.py);
# the pin is the engine argument (vllm/engine/arg_utils.py lines 791, 2686-2695).
ATTENTION_BACKEND = "FLASH_ATTN"
BENCHES = ("math500", "gsm8k")
ENDPOINTS = ("base", "rl", "zero", "id")
HIST_EDGES = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
EVERY_N, EVERY_S = 25, 60.0
STEP_BLOCK = 10       # A22: engine steps per steps.jsonl line (one line per step would pass 1 MB in a gate arm: about 10^5 steps)
SAMPLING = {"n": 1, "temperature": 0.6, "top_p": 0.95, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "stop_token_ids": [S.STOP_ID], "logprobs": 0}
NEUTRAL = {"ignore_eos": (False,), "min_tokens": (0,), "bad_words": (None, []), "logits_processors": (None, []), "allowed_token_ids": (None,),
           "logit_bias": (None, {}), "truncate_prompt_tokens": (None,), "stop": (None, []), "best_of": (None, 1),
           "guided_decoding": (None,), "structured_outputs": (None,), "prompt_logprobs": (None,)}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class Out:
    def __init__(self, rid, finished, token_ids, logprobs, finish_reason):
        self.rid, self.finished, self.token_ids, self.logprobs, self.finish_reason = rid, finished, token_ids, logprobs, finish_reason


class VllmEngine:
    """The four engine calls the loop uses, on vLLM's LLMEngine (§0.3 item 4: verified at the S2 smoke, its first run)."""

    def __init__(self, model_dir, tokenizer_dir, cap):
        wrong = {k: os.environ.get(k) for k, v in ENGINE_ENV.items() if os.environ.get(k) != v}
        assert not wrong, f"A21: the engine environment differs from {ENGINE_ENV}: {wrong}"     # checked before vllm is imported
        import vllm
        from vllm import EngineArgs, LLMEngine, SamplingParams
        self.SP, self.cap = SamplingParams, cap
        args = EngineArgs(model=str(model_dir), tokenizer=str(tokenizer_dir), dtype="bfloat16", tensor_parallel_size=1,
                          max_model_len=2048 + cap, gpu_memory_utilization=0.90, enable_prefix_caching=True, generation_config="vllm", seed=0,
                          attention_backend=ATTENTION_BACKEND)
        self.engine = LLMEngine.from_engine_args(args)
        sc = self.engine.vllm_config.scheduler_config
        self.info = {"vllm": vllm.__version__, "engine_args": {k: str(v) for k, v in vars(args).items()},
                     "env": {k: os.environ[k] for k in ENGINE_ENV}, "attention_backend": ATTENTION_BACKEND, "amendment": "N13_SPEC.md A21",
                     "max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,     # A22: the concurrency limit
                     "num_gpu_blocks": getattr(self.engine.vllm_config.cache_config, "num_gpu_blocks", None)}
        self.seen, self.plen, self.last = {}, {}, None

    def params(self, seed):
        sp = self.SP(**SAMPLING, max_tokens=self.cap, seed=seed)
        bad = {k: getattr(sp, k) for k, ok in NEUTRAL.items() if hasattr(sp, k) and getattr(sp, k) not in ok}
        assert not bad, f"sampling fields not at their neutral value: {bad}"
        return sp

    def effective(self, seed) -> dict:
        sp = self.params(seed)
        return {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in
                ((k, getattr(sp, k)) for k in dir(sp) if not k.startswith("_") and not callable(getattr(sp, k)))}

    def add(self, rid, ids, seed):
        self.plen[rid] = len(ids)
        self.engine.add_request(rid, {"prompt_token_ids": ids}, self.params(seed))

    def step(self):
        """The finished answers. A22: self.last = this step's running sequences (requests that got new tokens; the default output kind
        is cumulative, vllm/v1/engine/output_processor.py lines 307-373 and 428-441 at v0.32.0), output tokens, unfinished requests and
        (A22 addition) ctx, the sum over the running sequences of prompt + tokens so far."""
        res, run, tok, ctx = [], 0, 0, 0
        for o in self.engine.step():
            c = o.outputs[0]
            n = len(c.token_ids)
            new = n - self.seen.get(o.request_id, 0)
            run += new > 0
            tok += new
            ctx += (self.plen[o.request_id] + n) if new > 0 else 0
            if o.finished:
                self.seen.pop(o.request_id, None)
                self.plen.pop(o.request_id)
                tids = list(c.token_ids)
                res.append(Out(o.request_id, True, tids, [lp[t].logprob for lp, t in zip(c.logprobs, tids)], c.finish_reason))
            else:
                self.seen[o.request_id] = n
        self.last = {"run": run, "tok": tok, "unf": self.engine.get_num_unfinished_requests(), "ctx": ctx}
        return res

    def has_unfinished(self):
        return self.engine.has_unfinished_requests()

    def abort(self, rids):
        self.engine.abort_request(list(rids))

    def memory(self):
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"], check=True,
                           capture_output=True, text=True).stdout.strip().split(", ")
        return {"gpu_memory_used_mib": int(q[0]), "gpu_memory_total_mib": int(q[1]), "gpu": q[2]}


class FakeEngine:
    """Offline gate only: canned texts keyed by request id, tokenized with the real tokenizer; per_step answers finish per step; the clock
    advances dt seconds per step. A22: min(queued, conc) requests run in each step, and a finished answer's tokens count in its step."""

    def __init__(self, tok, canned: dict, cap, clock, per_step=3, dt=30.0, seed=0, conc=P.CONC_FULL):
        self.tok, self.canned, self.cap, self.clock, self.per_step, self.dt, self.conc = tok, canned, cap, clock, per_step, dt, conc
        self.queue, self.rng = [], np.random.default_rng(seed)
        self.info = {"vllm": None, "fake": True, "max_num_seqs": conc}
        self.last = None

    def effective(self, seed):
        return {**SAMPLING, "max_tokens": self.cap, "seed": seed}

    def add(self, rid, ids, seed):
        self.queue.append((rid, seed, len(ids)))

    def step(self):
        self.clock.advance(self.dt)
        run = min(len(self.queue), self.conc)
        ctx = sum(q[2] for q in self.queue[:run])       # the fake's running sequences hold their prompts only
        done, self.queue = self.queue[: self.per_step], self.queue[self.per_step:]
        res = []
        for rid, seed, _ in done:
            c = self.canned[rid]
            if c == "CAP":
                ids = (self.tok("step ", add_special_tokens=False)["input_ids"] * self.cap)[: self.cap]
                res.append(Out(rid, True, ids, list(self.rng.uniform(-2, 0, len(ids))), "length"))
            else:
                ids = self.tok(c, add_special_tokens=False)["input_ids"][: self.cap] + [S.STOP_ID]
                res.append(Out(rid, True, ids, list(self.rng.uniform(-2, 0, len(ids))), "stop"))
        self.last = {"run": run, "tok": sum(len(o.token_ids) for o in res), "unf": len(self.queue), "ctx": ctx}
        return res

    def has_unfinished(self):
        return bool(self.queue)

    def abort(self, rids):
        rids = set(rids)
        self.queue = [q for q in self.queue if q[0] not in rids]

    def memory(self):
        return {"gpu_memory_used_mib": None, "gpu_memory_total_mib": None, "gpu": "fake"}


class Clock:
    def __init__(self, t0=None):
        self.t = time.time() if t0 is None else t0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class StepLog:
    """A22: the engine-step log. One line per STEP_BLOCK steps: i (first step), n (steps), t (seconds from the generation start to the end
    of the block's last step), dt (the block's wall seconds; a step's wall time runs from the end of the previous step to the end of
    this one), tok (output tokens), run and unf ([min, max] running sequences and unfinished requests), run_dt (sum of running x dt),
    full and wait ([seconds, tokens, end time of the last such step, sum of running x dt, sum of ctx x dt] over the steps with at
    least n13_plan.CONC_FULL running, and over the steps with unfinished > running; None when the block has none; the last two from
    the A22 addition, absent in attempt 8's log). Each line is flushed when written."""

    def __init__(self, path, block=STEP_BLOCK, conc=P.CONC_FULL):
        path = pathlib.Path(path)
        assert not path.exists(), f"{path} exists: one fresh steps log per run"
        self.f, self.block, self.conc, self.rows, self.cur, self.i = open(path, "w"), block, conc, [], None, 0

    def add(self, t, dt, run, tok, unf, ctx):
        b = self.cur or {"i": self.i, "n": 0, "t": 0.0, "dt": 0.0, "tok": 0, "run": [run, run], "unf": [unf, unf], "run_dt": 0.0,
                         "full": None, "wait": None}
        b["n"] += 1
        b["t"] = round(t, 6)
        b["dt"] = round(b["dt"] + dt, 6)
        b["tok"] += tok
        b["run"] = [min(b["run"][0], run), max(b["run"][1], run)]
        b["unf"] = [min(b["unf"][0], unf), max(b["unf"][1], unf)]
        b["run_dt"] = round(b["run_dt"] + run * dt, 6)
        for key, cond in (("full", run >= self.conc), ("wait", unf > run)):
            if cond:
                x = b[key] or [0.0, 0, None, 0.0, 0.0]
                b[key] = [round(x[0] + dt, 6), x[1] + tok, b["t"], round(x[3] + run * dt, 6), round(x[4] + ctx * dt, 6)]
        self.i += 1
        self.cur = b
        if b["n"] == self.block:
            self._write()

    def _write(self):
        self.f.write(json.dumps(self.cur) + "\n")
        self.f.flush()
        self.rows.append(self.cur)
        self.cur = None

    def close(self):
        if self.cur is not None:
            self._write()
        self.f.close()
        return self.rows


# ------------------------------------------------------------------------------------------------ records
def record(tok, arm, pair, bench, prob, seed, prompt, ids_prompt, o: Out, cap) -> dict:
    j = next((i for i, t in enumerate(o.token_ids) if t == S.STOP_ID), None)
    ids = o.token_ids if j is None else o.token_ids[:j]
    assert j is not None or len(ids) == cap, f"{arm} {bench} {prob['index']}: no stop id in {len(ids)} ids, cap {cap}"
    text = tok.decode(ids, skip_special_tokens=False)
    mode, ext, span = S.extract(text)
    g = S.grade(prob["answer"], ext)
    lp = S.span_logprob(tok, ids, list(o.logprobs[: len(ids)]), text, span)
    return {"arm": arm, "pair": pair, "bench": bench, "problem_index": prob["index"], "unique_id": prob["unique_id"], "seed": seed,
            "prompt": prompt, "prompt_n_tokens": len(ids_prompt), "prompt_first20": ids_prompt[:20], "prompt_last20": ids_prompt[-20:],
            "text": text, "n_new": len(ids), "n_generated": len(o.token_ids), "cap_hit": j is None, "finish_reason": o.finish_reason,
            "mode": mode, "extracted": ext, "gold": prob["answer"], **g, **lp, "utc": utc()}


def pick_samples(recs: list) -> list:
    by = sorted(recs, key=lambda r: r["problem_index"])
    roles = {}
    for r in by:
        if r["problem_index"] in (0, 1, 2):
            roles.setdefault(r["problem_index"], []).append(f"fixed_{r['problem_index']}")
    for name, cond in (("lowest_right", lambda r: r["score"] == 1), ("lowest_wrong", lambda r: r["score"] == 0), ("cap_hit", lambda r: r["cap_hit"])):
        x = next((r for r in by if cond(r)), None)
        if x is not None:
            roles.setdefault(x["problem_index"], []).append(name)
    return [{**r, "roles": roles[r["problem_index"]]} for r in by if r["problem_index"] in roles]


def counts(recs: list) -> dict:
    n = len(recs)
    modes = {m: sum(r["mode"] == m for r in recs) for m in ("answer_line", "boxed", "none")}
    return {"finished": n, "correct": sum(r["score"] for r in recs), "cap_hits": sum(r["cap_hit"] for r in recs), "modes": modes,
            "output_tokens": sum(r["n_generated"] for r in recs), "acc": S.acc([r["score"] for r in recs]) if n else None}


def throughput(arm, recs_all: list, t_gen0: float, t_last: float, engine, rows: list, load_s: float, step_s: float, cap: int) -> dict:
    el = max(t_last - t_gen0, 1e-9)
    n_new = [r["n_new"] for r in recs_all]
    n_gen = [r["n_generated"] for r in recs_all]
    toks = sum(n_gen)
    h, _ = np.histogram(n_new, bins=HIST_EDGES)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sr = P.step_rates(rows)
    return {"arm": arm, "n_answers": len(recs_all), "output_tokens": toks, "generation_seconds": el, "output_tokens_per_s": toks / el,
            "seconds_per_problem": el / max(len(recs_all), 1), "mean_new_tokens": float(np.mean(n_new)) if n_new else None,
            "max_new_tokens": int(max(n_new)) if n_new else None, "histogram_edges": HIST_EDGES, "histogram_counts": h.tolist(),
            **engine.memory(), "host_peak_rss_kb": ru if sys.platform != "darwin" else ru // 1024,
            "free_disk_gb": shutil.disk_usage("/").free / 1e9,
            # A22 (N13_SPEC.md)
            "amendment": "N13_SPEC.md A22", "r_full": sr["r_full"], "r_all": sr["r_all"], "tail_s": sr["tail_s"], "load_s": load_s,
            "r_sat": sr["r_sat"], "tail_sat_s": sr["tail_sat_s"], "sat_run_mean": sr["sat_run_mean"], "sat_ctx_mean": sr["sat_ctx_mean"],
            "step_s": step_s, "median_new_tokens": float(np.median(n_new)) if n_new else None, "cap": cap,
            "n_cap_hits": sum(r["cap_hit"] for r in recs_all), "mean_output_tokens": toks / len(recs_all) if recs_all else None,
            "mean_output_tokens_at": {str(c): float(np.mean([min(x, c) for x in n_gen])) if n_gen else None for c in P.CAPS},
            "steps_file": "steps.jsonl", "steps_block": STEP_BLOCK, "steps": sr}


# ------------------------------------------------------------------------------------------------ the step
def run(a, engine_factory=None, clock=time.time) -> dict:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    benches = a.bench.split(",")
    assert all(b in BENCHES for b in benches) and benches == [b for b in BENCHES if b in benches], benches
    lo, hi = (int(x) for x in a.problems.split(":"))
    seed_set = P.seed_set(a.arm)
    arm_out = out / a.arm
    t_run0 = clock()
    assert not (out / "steps.jsonl").exists(), f"{out / 'steps.jsonl'} exists: one fresh steps log per run"
    for b in benches:
        (arm_out / b).mkdir(parents=True, exist_ok=True)
        assert not any((arm_out / b).iterdir()), f"{arm_out / b} is not empty: one fresh directory per run"
    base_m = {"step": "gen", "arm": a.arm, "pair": a.pair, "seed_set": seed_set, "status": "running", "started_utc": utc(), "smoke": a.smoke,
              "grade_rule": S.GRADE_RULE}
    ms = {b: {**base_m, "bench": b} for b in benches}

    def write_all():
        for b in benches:
            ms[b]["updated_utc"] = utc()
            dump(arm_out / b / "metrics.json", ms[b])

    setup = json.loads((pathlib.Path(a.setup_dir) / "setup.json").read_text())
    probs_all = json.loads(pathlib.Path(setup["problems"]).read_text())
    probs = {b: probs_all[b][lo:hi] if b == "math500" else probs_all[b] for b in benches}
    proj = P.gen_minutes(len(probs.get("math500", [])), len(probs.get("gsm8k", [])), a.proj_tok_s, a.proj_mean_math, a.proj_mean_gsm,
                         a.proj_load_min)
    park = P.park_check(out, a.allowance_min, proj, now=clock())
    for b in benches:
        ms[b]["park"] = park
    if park["parked"]:
        for b in benches:
            ms[b]["status"] = "parked"
        write_all()
        return ms
    if a.only_if:
        ctl = json.loads(pathlib.Path(a.only_if).read_text())
        jf = ctl["J_floor"][A.K_TAGS[A.PRIMARY]]
        cond = {"rule": P.CONDITIONAL.get(a.arm), "J_floor_k1": jf, "run": bool(jf < 0.9)}
        for b in benches:
            ms[b]["condition"] = cond
        if not cond["run"]:
            for b in benches:
                ms[b]["status"] = "not_run_condition"
            write_all()
            return ms
    cap = a.cap if a.cap else json.loads(pathlib.Path(a.cap_from).read_text())["C"]
    for b in benches:
        ms[b]["cap"] = cap
    write_all()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(setup["tokenizer_dir"])
    banked = json.loads(pathlib.Path(setup["prompt_ids"]).read_text())
    assert banked["pair"] == a.pair, (banked["pair"], a.pair)
    prompts, pids = {}, {}
    for b in benches:
        prompts[b] = [S.render_text(tok, p["problem"]) for p in probs[b]]
        bad = [p["index"] for p, t in zip(probs[b], prompts[b]) if t != S.TEMPLATE[a.pair].format(problem=p["problem"])]
        assert not bad, (f"G-TEMPLATE: {b} renders differ from the registered bytes", bad[:5])
        pids[b] = [S.render_ids(tok, p["problem"]) for p in probs[b]]
        assert max(len(x) for x in pids[b]) <= 2048, f"a {b} prompt is longer than 2,048 tokens"
    if "math500" in benches:
        assert pids["math500"] == banked["ids"][lo:hi], "G-TEMPLATE: the MATH-500 prompt ids differ from the setup step's"
    for b in benches:
        ms[b]["template"] = {"pass": True, "first_id_is_bos": sorted({x[0] == S.BOS_ID for x in pids[b]}), "n": len(pids[b])}

    # weights -------------------------------------------------------------------------------------------------------------------
    snaps = setup["snapshots"]
    if a.arm in ("base", "rl"):
        model_dir, audit = snaps[a.arm], {"arm": a.arm, "from": "HF snapshot", "dir": snaps[a.arm]}
    else:
        sets = None if a.arm in ("zero", "id") else json.loads(pathlib.Path(a.sets).read_text())   # A28: zero and id read no R1 set
        cfg = json.loads((pathlib.Path(snaps["rl"]) / "config.json").read_text())
        before, after = A.Weights(snaps["base"]), A.Weights(snaps["rl"])
        model_dir = pathlib.Path(a.arm_root) / a.arm
        spec = A.arm_spec(a.arm, sets)
        t0 = time.time()
        audit = A.build_arm(before, after, cfg, spec, out_dir=model_dir, kept_sd=a.kept_sd)
        audit["build_seconds"] = round(time.time() - t0, 1)
        if a.arm in ("zero", "id"):
            ref = A.snapshot_hashes(before if a.arm == "zero" else after)
            audit["G-HASH"] = A.g_hash(audit["sha256"], ref)
            fi = dict(audit["sha256"])
            if a.arm == "id":
                rev = A.build_arm(before, after, cfg, {"name": "id_rev00", "kind": "complement", "neurons": [0], "c": 1.0, "non_mlp": "after"})
                fi = rev["sha256"]
            else:
                fwd = A.build_arm(before, after, cfg, {"name": "zero_add00", "kind": "neurons", "neurons": [0], "c": 1.0, "non_mlp": "before"})
                fi = fwd["sha256"]
            audit["G-HASH"]["failing_input"] = {"what": "neuron (0, 0) reverted (id) or added (zero)", "failed": not A.g_hash(fi, ref)["pass"]}
    dump(arm_out / "build.json", audit)
    if "G-HASH" in audit:
        assert audit["G-HASH"]["pass"] and audit["G-HASH"]["failing_input"]["failed"], ("G-HASH", audit["G-HASH"])
    if "pass" in audit.get("mask", {}):
        assert audit["mask"]["pass"], "G-MASK: the realized changed rows and columns differ from the set"
    if "kept_pass" in audit:
        assert audit["kept_pass"], ("DARE kept count outside tolerance (A4)", audit["kept_sd_units"], audit["kept_tol_entries"])
    for b in benches:
        ms[b]["build"] = {k: audit.get(k) for k in ("arm", "from", "G-HASH", "kept_rel_err", "kept_sd_units", "energy_rel_err", "energy_flag",
                                                    "energy_passes", "c_built", "a31", "build_seconds")}
        ms[b]["build"]["G-MASK"] = audit.get("mask", {}).get("pass")
    write_all()

    # engine and loop -----------------------------------------------------------------------------------------------------------
    t_load0 = clock()
    engine = engine_factory(model_dir, setup["tokenizer_dir"], cap, tok) if engine_factory else VllmEngine(model_dir, setup["tokenizer_dir"], cap)
    load_s = clock() - t_load0                      # A22: engine construction plus weight load
    eff = engine.effective(S.SEED_BASE[seed_set] + 0)
    for b in benches:
        ms[b]["engine"] = {**engine.info, "load_s": load_s}
        ms[b]["sampling_effective"] = eff
        ms[b]["submitted"] = len(probs[b])
    write_all()
    for b in benches:
        for p, ids in zip(probs[b], pids[b]):
            engine.add(f"{b}:{p['index']}", ids, S.SEED_BASE[seed_set] + p["index"])
    t_gen0 = clock()
    slog, t_prev = StepLog(out / "steps.jsonl"), t_gen0
    recs = {b: [] for b in benches}
    by_idx = {b: {p["index"]: (p, t, i) for p, t, i in zip(probs[b], prompts[b], pids[b])} for b in benches}
    files = {b: open(arm_out / b / "generations.jsonl", "a") for b in benches}
    last_write, n_since, screened, t_last = clock(), 0, None, clock()
    n_writes = 0

    def refresh():
        nonlocal n_writes
        n_writes += 1
        for b in benches:
            ms[b]["counts"] = counts(recs[b])
            ms[b]["elapsed_s"] = clock() - t_gen0
            ms[b]["n_metrics_writes"] = n_writes
            with open(arm_out / b / "samples.jsonl.tmp", "w") as f:
                for r in pick_samples(recs[b]):
                    f.write(json.dumps(r) + "\n")
            os.replace(arm_out / b / "samples.jsonl.tmp", arm_out / b / "samples.jsonl")
        write_all()

    while engine.has_unfinished():
        outs = engine.step()
        t_step = clock()
        slog.add(t_step - t_gen0, t_step - t_prev, **engine.last)
        t_prev = t_step
        for o in outs:
            b, idx = o.rid.split(":")
            p, prompt, ids_p = by_idx[b][int(idx)]
            r = record(tok, a.arm, a.pair, b, p, S.SEED_BASE[seed_set] + p["index"], prompt, ids_p, o, cap)
            files[b].write(json.dumps(r) + "\n")
            files[b].flush()
            os.fsync(files[b].fileno())
            recs[b].append(r)
            n_since += 1
            t_last = clock()
        if "math500" in benches and screened is None:
            sc = S.damage_screen(recs["math500"]) if lo == 0 else {"ready": False}
            if sc["ready"]:
                screened = sc
                for b in benches:
                    ms[b]["screen"] = sc
                if sc["fired"] and a.arm not in ENDPOINTS:
                    done = {(b, r["problem_index"]) for b in benches for r in recs[b]}
                    engine.abort([f"{b}:{p['index']}" for b in benches for p in probs[b] if (b, p["index"]) not in done])
        if n_since >= EVERY_N or clock() - last_write >= EVERY_S:
            refresh()
            n_since, last_write = 0, clock()
    for f in files.values():
        f.close()
    rows = slog.close()
    tp = throughput(a.arm, [r for b in benches for r in recs[b]], t_gen0, t_last, engine, rows, load_s, clock() - t_run0, cap)
    for b in benches:
        ms[b]["G-COMPLETE"] = S.g_complete(recs[b], len(probs[b]), seed_set) | {"expected_n": len(probs[b])}
        ms[b]["throughput"] = tp
        ms[b]["status"] = "screened" if (screened and screened["fired"] and a.arm not in ENDPOINTS) else "done"
        ms[b]["finished_utc"] = utc()
    refresh()
    if a.smoke:
        dump(out / "throughput.json", tp)
    print(f"ARM {a.arm}: " + ", ".join(f"{b} {ms[b]['counts']['correct']}/{ms[b]['counts']['finished']} ({ms[b]['status']})" for b in benches)
          + f"; {tp['output_tokens_per_s']:.0f} output tok/s; A22 r_full {tp['r_full']}, r_sat {tp['r_sat']}, r_all {tp['r_all']}, tail_s {tp['tail_s']}, "
          f"load_s {load_s:.1f}", flush=True)
    return ms


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bench", default="math500")
    ap.add_argument("--problems", default="0:500")
    ap.add_argument("--pair", default="pair1")
    ap.add_argument("--setup-dir", required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--cap-from", default=None)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--only-if", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm-root", default="/work/arm")
    ap.add_argument("--kept-sd", type=float, default=4.0, help="DARE kept-count tolerance in binomial sd (N13_SPEC.md A4)")
    ap.add_argument("--allowance-min", type=float, default=None)
    ap.add_argument("--proj-tok-s", type=float, default=P.ASSUME["tok_s"][1])
    ap.add_argument("--proj-mean-math", type=float, default=P.ASSUME["mean_math"])
    ap.add_argument("--proj-mean-gsm", type=float, default=P.ASSUME["mean_gsm"])
    ap.add_argument("--proj-load-min", type=float, default=P.ASSUME["load_min"])
    ap.add_argument("--inner", action="store_true")
    return ap


def fetch_inputs(a, argv: list) -> list:
    """Parent side: S3 inputs to local files; returns argv with local paths."""
    loc = pathlib.Path("/work/inputs")
    loc.mkdir(parents=True, exist_ok=True)
    new = list(argv)
    for flag in ("--sets", "--cap-from", "--only-if"):
        if flag in new and new[new.index(flag) + 1].startswith("s3://"):
            src = new[new.index(flag) + 1]
            dst = loc / pathlib.Path(src).name
            subprocess.run(["aws", "s3", "cp", src, str(dst), "--region", "us-east-1"], check=True)
            new[new.index(flag) + 1] = str(dst)
    return new


def child_env(environ) -> dict:
    """The --inner child's environment, which vLLM's engine process inherits: the venv first on PATH, as for olmes (s2setup-03), and
    ENGINE_ENV (A21)."""
    env = dict(environ)
    env["PATH"] = f"{pathlib.Path(N13_PY).parent}:{env['PATH']}"
    env.update(ENGINE_ENV)
    return env


def main():
    a = parser().parse_args()
    if not a.inner:
        from infra.pg.humboldt_core import hf_setup
        hf_setup()
        argv = fetch_inputs(a, sys.argv[1:])
        built = pathlib.Path(a.arm_root) / a.arm
        assert not built.exists(), f"{built} exists before the build"
        subprocess.run([N13_PY, "-m", "infra.aws_pg.jobs.n13_gen", *argv, "--inner"], check=True, env=child_env(os.environ))
        if built.exists():
            shutil.rmtree(built)          # 14.6 GB per arm; the audit and the hashes are in build.json
        return
    run(a)


if __name__ == "__main__":
    main()
