"""N13 R0, the known number (studies/next/07_PICKS/N13_SPEC.md §6.1 G-KNOWN, §9.4 S2 step 2; studies/next/07_PICKS/N13_MATH_TASK.md).

    python -m infra.aws_pg.jobs.n13_known --setup-dir <setup out> --out <dir> [--allowance-min M]

Runs olmes (allenai/olmes at 5a51f502…, its own venv with vllm 0.11.0 and lm_eval 0.4.3) on the base at main, as amended by
N13_SPEC.md A18 (b) and A19:
    olmes --model allenai/Olmo-3-1025-7B --revision a81bae42… --model-type vllm --model-args '{"trust_remote_code": false}'
          --task minerva_math::olmes:n4:v2 --gpus 1 --batch-size auto --output-dir <out>/olmes
--batch-size auto (A19): without it olmes runs at batch size 1 and lm_eval sends one prompt per LLM.generate call (s2setup-06: 560
calls of 1 prompt, 3,120 s for 560 generations); with "auto" lm_eval sends all prompts of one subject in one call, and vLLM schedules
them together. The sampling parameters are the same under both.
The olmes subprocess gets n13_olmes_shim/ first on PYTHONPATH and N13_VLLM_SHIM_LOG set, so its sitecustomize.py converts lm_eval's
generate(prompt_token_ids=...) call for vLLM 0.11.0; the number of converted calls and the shim's sha256 are written to metrics.json.
First a path check (A18 condition 3 and A19, not a gate): the same command with --limit 100 into <out>/olmes_limit100 (A19 second
revision: 100 problems x 4 repeats = one call of 400 prompts per subject, above vLLM's 256 running sequences, so the engine runs at its
limit for most of each call; 2,800 generations in all). While it runs, every poll reads the shim log's first line; if that call has
n_prompts < 2 (olmes at batch size 1), olmes' process group is killed at once and the step stops (exit 4). At the end it must have
exited 0, give 100 scored rows per subject and log at least one shim call with n_prompts >= 2, and the reader and the rescorer must run
on it; its score is not read against 40.0. If it fails, the step stops (exit 4) before the full task.
Then the projection gate (A19, second revision 2026-11-05): the shim logs one call per subject in olmes' task order (= SUBJECTS); for
subject s, tokens_s and n_s are its call's output tokens and prompts. T = sum_s 4 x size_s x tokens_s / n_s (the full task's output
tokens at each subject's mean length), r = all output tokens / all generation seconds (the summed time inside vLLM's generate),
F = path-check wall seconds - generation seconds (olmes start-up and per-subject set-up and scoring, carried over unchanged);
projected = F + T / r. The step parks (status "parked", the PARKED marker, exit 0) before the full task when the projection exceeds
85 % of the minutes left, i.e. the allowance minus the job's elapsed time on the instance (n13_plan.launch_t0).
While olmes runs, out/metrics.json is rewritten every 60 s (elapsed, olmes files written). Then:
  - read_primary: the aggregate task minerva_math::olmes:n4:v2 in metrics-all.jsonl, field primary_score (= primary_score_macro over the
    7 subjects); pass_at_1_micro is recorded and never read as the score; a directory without the aggregate primary_score stops the job;
  - G-KNOWN: |100 * primary_score - 40.0| <= 1.6;
  - the failing input: n13_olmes_rescore under the olmes venv rescores the same predictions with olmes' own Minerva scorer against a
    derangement of the golds within each subject (seed 0); it must score < 10.0. The same rescorer with the true golds must reproduce
    olmes' per-document pass_at_1 exactly (the identity pair of the rescorer);
  - samples.jsonl: 5 documents scored right and 5 scored wrong, with the prompt as the model saw it, the four continuations and the gold.
Exit 0 only when G-KNOWN passes, its failing input fails and the identity rescore matches; exit 4 on a failed path check; exit 3 otherwise,
so the pipeline stops
before the smoke (the stop rule of N13_SPEC.md §6.5: "stop before S3").
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

from infra.pg import n13_plan as P

SUITE = "minerva_math::olmes:n4:v2"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")
SUBJECT_SIZES = {"algebra": 1187, "counting_and_probability": 474, "geometry": 479, "intermediate_algebra": 903, "number_theory": 540,
                 "prealgebra": 871, "precalculus": 546}
TARGET, TOL, DERANGED_MAX = 40.0, 1.6, 10.0
BASE_REPO, BASE_REV = "allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"
OLMES_BIN = "/opt/olmes-venv/bin/olmes"
OLMES_PY = "/opt/olmes-venv/bin/python"
OLMES_COMMIT = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
MODEL_ARGS = '{"trust_remote_code": false}'          # A18 (b): vLLM 0.11.0 rejects olmes' default None; False is what None means here
BATCH_SIZE = "auto"                                   # A19: olmes passes it to lm_eval's VLLM; "auto" = one LLM.generate call per subject
MARGIN = 0.85                                         # A19: the projected full task must fit in 85 % of the minutes left
SHIM_DIR = pathlib.Path(__file__).resolve().parent / "n13_olmes_shim"
PATH_LIMIT = 100                                      # A19 second revision: 400 prompts per subject call (was 20: 80 prompts)
EXPECTED = {
    "suite_tasks": [f"minerva_math_{t}::olmes:n4:v2" for t in SUBJECTS],
    "suite_primary_metric": "macro",
    "task": {"split": "test", "primary_metric": "pass_at_1", "num_shots": 4, "fewshot_source": "Minerva:MATH:fixed",
             "generation_kwargs": {"max_gen_toks": 1024, "temperature": 0.6, "top_p": 0.6, "do_sample": True,
                                   "stop_sequences": ["Problem:", "\n\n"], "repeats": 4},
             "metric_kwargs": {"pass_at_ks": [1, 2, 4], "maj_at_ks": [1, 2, 4]}},
    "math_task_types": list(SUBJECTS),
    "dataset_path": "EleutherAI/hendrycks_math",
    "pins": {"vllm": "0.11.0", "lm_eval": "0.4.3", "antlr4-python3-runtime": "4.11"},
    "model_defaults": {"chat_model": False, "max_length": 2048},
}
REPEATS = EXPECTED["task"]["generation_kwargs"]["repeats"]
FULL_GENERATIONS = sum(SUBJECT_SIZES.values()) * REPEATS          # 5,000 x 4 = 20,000 (A19)
PATH_PROMPTS = len(SUBJECTS) * PATH_LIMIT * REPEATS                # 7 x 100 x 4 = 2,800 (A19 second revision)


def command(out_dir, model_type: str = "vllm", limit=None) -> list:
    """The G-KNOWN command of §6.1 as amended by A18 (b) and A19. model_type "hf" only under §0.3 item 1's branch; limit only for the
    path check."""
    return [OLMES_BIN, "--model", BASE_REPO, "--revision", BASE_REV, "--model-type", model_type, "--model-args", MODEL_ARGS, "--task", SUITE,
            "--gpus", "1", "--batch-size", BATCH_SIZE, "--output-dir", str(out_dir)] + (["--limit", str(limit)] if limit else [])


def engine() -> dict:
    """The engine named in the artifact (A18 condition 1); the pins are asserted by the setup step."""
    shim = SHIM_DIR / "sitecustomize.py"
    return {"vllm": EXPECTED["pins"]["vllm"], "lm_eval": EXPECTED["pins"]["lm_eval"], "olmes_commit": OLMES_COMMIT, "model_args": MODEL_ARGS,
            "batch_size": BATCH_SIZE, "shim": str(shim), "shim_sha256": hashlib.sha256(shim.read_bytes()).hexdigest(),
            "amendment": "N13_SPEC.md A18 (b), A19"}


def release(v: str) -> tuple:
    """A numeric version's release tuple with trailing zeros dropped, so "4.11" equals "4.11.0" (PEP 440). olmes pins
    antlr4-python3-runtime==4.11 and pip installs 4.11.0 (s2setup-02 stopped on the string comparison)."""
    t = [int(x) for x in v.split(".")]
    while t and t[-1] == 0:
        t.pop()
    return tuple(t)


def tolerance() -> dict:
    """One reading's sampling SE bound: p(1-p) <= 1/4 per sample, 4 samples, subject weights 1/7; the difference of two readings; x3."""
    var = sum(1 / (16 * n) for n in SUBJECT_SIZES.values()) / 49
    se = 100 * math.sqrt(var)
    return {"se_one_reading": se, "se_difference": math.sqrt(2) * se, "three_se_difference": 3 * math.sqrt(2) * se, "tol": TOL}


# ------------------------------------------------------------------------------------------------ the olmes task config
def _literal(node):
    return ast.literal_eval(node)


def static_config(olmes_dir) -> dict:
    """Read the Minerva task configs from the olmes source without running it (ast only)."""
    d = pathlib.Path(olmes_dir)
    suites = ast.parse((d / "oe_eval/configs/task_suites.py").read_text())
    tasks = ast.parse((d / "oe_eval/configs/tasks.py").read_text())
    types = ast.parse((d / "oe_eval/data/math_task_types.py").read_text())
    cfg = {}
    for n in ast.walk(suites):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].slice, ast.Constant) \
                and n.targets[0].slice.value == SUITE:
            keys = {k.value: v for k, v in zip(n.value.keys, n.value.values)}
            comp = keys["tasks"]
            fmt = "".join(p.value if isinstance(p, ast.Constant) else "{" + p.value.id + "}" for p in comp.elt.values)
            cfg["suite_task_format"] = fmt
            cfg["suite_task_iter"] = comp.generators[0].iter.id
            cfg["suite_primary_metric"] = _literal(keys["primary_metric"])
            cfg["suite_line"] = n.lineno
    for n in ast.walk(tasks):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].slice, ast.JoinedStr):
            parts = [p.value if isinstance(p, ast.Constant) else "{}" for p in n.targets[0].slice.values]
            if "".join(parts) == "minerva_math_{}::olmes:n4:v2":
                v = _literal(n.value) if all(not isinstance(x, ast.JoinedStr) for x in ast.walk(n.value)) else None
                if v is None:
                    v = {k.value: (None if isinstance(val, ast.JoinedStr) else _literal(val)) for k, val in zip(n.value.keys, n.value.values)}
                cfg["task"] = {k: v[k] for k in v if k != "task_name"}
                cfg["task_line"] = n.lineno
    for n in ast.walk(types):
        if isinstance(n, ast.Assign) and n.targets[0].id == "MATH_TASK_TYPES":
            cfg["math_task_types"] = _literal(n.value)
    mm = (d / "oe_eval/tasks/oe_eval_tasks/minerva_math.py").read_text()
    cfg["dataset_path"] = re.search(r'"dataset_path":\s*"([^"]+)"', mm).group(1)
    pp = (d / "pyproject.toml").read_text()
    cfg["pins"] = {k: re.search(rf'"{re.escape(k)}==([^"]+)"', pp).group(1) for k in EXPECTED["pins"]}
    dc = ast.parse((d / "oe_eval/default_configs.py").read_text())
    for n in ast.walk(dc):
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "MODEL_DEFAULTS":
            md = {k.value: val for k, val in zip(n.value.keys, n.value.values)}
            cfg["model_defaults"] = {k: _literal(md[k]) for k in EXPECTED["model_defaults"]}
    lp = (d / "oe_eval/launch.py").read_text()
    cfg["launch_flags"] = sorted(set(re.findall(r'_parser\.add_argument\(\s*"(--[a-z0-9-]+)"', lp)))
    return cfg


def dynamic_config() -> dict:
    """The same fields read from the installed olmes package (run under the olmes venv on the instance)."""
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS
    from oe_eval.data.math_task_types import MATH_TASK_TYPES
    from oe_eval.default_configs import MODEL_DEFAULTS
    import importlib.metadata as md
    s = TASK_SUITE_CONFIGS[SUITE]
    t = dict(TASK_CONFIGS[f"minerva_math_{SUBJECTS[0]}::olmes:n4:v2"])
    t.pop("task_name")
    every = all({k: v for k, v in TASK_CONFIGS[x].items() if k != "task_name"} == t for x in s["tasks"])
    return {"suite_tasks": s["tasks"], "suite_primary_metric": s["primary_metric"], "task": t, "all_subtasks_equal": every,
            "math_task_types": MATH_TASK_TYPES, "model_defaults": {k: MODEL_DEFAULTS[k] for k in EXPECTED["model_defaults"]},
            "pins": {k: md.version(k) for k in EXPECTED["pins"]}}


def check_config(cfg: dict) -> dict:
    """G-KNOWN's configuration half: the olmes task, line by line against EXPECTED (= N13_MATH_TASK.md)."""
    rows = {}
    if "suite_tasks" in cfg:
        rows["suite_tasks"] = cfg["suite_tasks"] == EXPECTED["suite_tasks"]
    else:
        rows["suite_tasks"] = [cfg["suite_task_format"].format(task_type=t) for t in cfg["math_task_types"]] == EXPECTED["suite_tasks"]
    rows["suite_primary_metric"] = cfg["suite_primary_metric"] == EXPECTED["suite_primary_metric"]
    for k, v in EXPECTED["task"].items():
        rows[f"task.{k}"] = cfg["task"].get(k) == v
    rows["math_task_types"] = cfg["math_task_types"] == EXPECTED["math_task_types"]
    if "dataset_path" in cfg:
        rows["dataset_path"] = cfg["dataset_path"] == EXPECTED["dataset_path"]
    rows["pins"] = set(cfg["pins"]) == set(EXPECTED["pins"]) and all(release(cfg["pins"][k]) == release(v) for k, v in EXPECTED["pins"].items())
    rows["model_defaults"] = cfg["model_defaults"] == EXPECTED["model_defaults"]
    if "all_subtasks_equal" in cfg:
        rows["all_subtasks_equal"] = cfg["all_subtasks_equal"]
    if "launch_flags" in cfg:
        used = [x for x in command("X") if x.startswith("--")]
        rows["command_flags_accepted"] = all(f in cfg["launch_flags"] for f in used)
    return {"rows": rows, "pass": all(rows.values()), "failed_rows": [k for k, v in rows.items() if not v]}


# ------------------------------------------------------------------------------------------------ reading olmes outputs
def read_primary(olmes_dir) -> dict:
    rows = [json.loads(l) for l in (pathlib.Path(olmes_dir) / "metrics-all.jsonl").read_text().splitlines() if l.strip()]
    agg = [r for r in rows if r["task_name"] == SUITE]
    assert len(agg) == 1, f"expected one aggregate row for {SUITE}, found {len(agg)}"
    m = agg[0]["metrics"]
    assert "primary_score" in m, f"the aggregate row holds no primary_score (keys: {sorted(m)})"
    assert agg[0]["task_config"].get("primary_metric") == "macro", agg[0]["task_config"].get("primary_metric")
    subs = {r["task_name"]: r for r in rows if r["task_name"] in [f"minerva_math_{t}" for t in SUBJECTS]}
    assert sorted(subs) == sorted(f"minerva_math_{t}" for t in SUBJECTS), sorted(subs)
    per = {t: subs[f"minerva_math_{t}"]["metrics"]["primary_score"] for t in SUBJECTS}
    sizes = {t: subs[f"minerva_math_{t}"]["num_instances"] for t in SUBJECTS}
    macro = sum(per.values()) / len(per)
    assert abs(macro - m["primary_score"]) < 1e-9, (macro, m["primary_score"])
    return {"primary_score": m["primary_score"], "primary_score_macro": m.get("primary_score_macro"), "pass_at_1_micro": m.get("pass_at_1_micro"),
            "per_subject": per, "num_instances": sizes, "sizes_match": sizes == SUBJECT_SIZES, "n_total": agg[0]["num_instances"]}


def g_known(primary: float, deranged: float, identity_ok: bool) -> dict:
    v = 100 * primary
    return {"value": v, "target": TARGET, "tol": TOL, "pass": bool(abs(v - TARGET) <= TOL),
            "failing_input": {"what": "the same predictions rescored by olmes' scorer against a derangement of the golds within each subject "
                                      "(seed 0)", "value": deranged, "failed": bool(deranged < DERANGED_MAX)},
            "identity": {"what": "the same rescorer with the true golds reproduces olmes' per-document pass_at_1", "pass": identity_ok}}


def samples(olmes_dir, n_each: int = 5) -> list:
    d = pathlib.Path(olmes_dir)
    out = []
    for t in SUBJECTS:
        pf = sorted(d.glob(f"task-*-minerva_math_{t}-predictions.jsonl"))
        rf = sorted(d.glob(f"task-*-minerva_math_{t}-requests.jsonl"))
        assert len(pf) == 1 and len(rf) == 1, (t, pf, rf)
        ctx = {}
        for l in rf[0].read_text().splitlines():
            r = json.loads(l)
            ctx.setdefault(r["doc_id"], r["request"]["context"])
        for l in pf[0].read_text().splitlines():
            r = json.loads(l)
            out.append({"subject": t, "doc_id": r["doc_id"], "native_id": r.get("native_id"), "pass_at_1": r["metrics"]["pass_at_1"],
                        "label": r["label"], "prompt": ctx[r["doc_id"]], "continuations": [x["continuation"] for x in r["model_output"]],
                        "model_answers": [x.get("model_answer") for x in r["model_output"]]})
    right = [s for s in out if s["pass_at_1"] == 1][:n_each]
    wrong = [s for s in out if s["pass_at_1"] == 0][:n_each]
    return [{**s, "scored": "right"} for s in right] + [{**s, "scored": "wrong"} for s in wrong]


def write(out: pathlib.Path, m: dict):
    m["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, out / "metrics.json")


def shim_lines(log: pathlib.Path) -> list:
    """The shim's log, one dict per converted LLM.generate call, in call order."""
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def first_call(log: pathlib.Path):
    """The shim log's first complete line, or None before the first converted call has returned (A19 second revision)."""
    if not log.exists():
        return None
    with open(log) as f:
        line = f.readline()
    return json.loads(line) if line.endswith("\n") else None


def shim_calls(log: pathlib.Path) -> dict:
    """The shim's log (one line per converted LLM.generate call): calls, prompts, the largest call, and (A19 revision) generate_s, the
    summed t_end - t_start of the calls, and output_tokens, the summed len(token_ids) of every returned completion."""
    lines = shim_lines(log)
    return {"log": str(log), "calls": len(lines), "prompts": sum(x["n_prompts"] for x in lines),
            "max_n_prompts": max((x["n_prompts"] for x in lines), default=0),
            "generate_s": sum(x["t_end"] - x["t_start"] for x in lines), "output_tokens": sum(x["output_tokens"] for x in lines)}


def project(wall_s: float, calls: list, allowance_min, elapsed_min: float) -> dict:
    """A19 projection gate, second revision 2026-11-05. calls: the path check's shim log lines, one per subject in olmes' task order,
    which is SUBJECTS (s2setup-06: task-000 to task-006 began in that order; olmes runs the suite's tasks one after another). For
    subject s, tokens_s and n_s are its call's output tokens and prompts. T = sum_s REPEATS x size_s x tokens_s / n_s, the full task's
    output tokens at each subject's own mean length (the path check samples PATH_LIMIT problems per subject; the full task weights the
    subjects by size); r = all output tokens / all generation seconds (t_end - t_start inside vLLM's generate); F = wall_s - generation
    seconds, the olmes start-up and the per-subject set-up and scoring, carried over unchanged. projected = F + T / r. At 400 prompts
    per call the engine runs at its concurrency limit for most of each call, so r is close to the full task's rate. It parks when
    projected > MARGIN x (allowance - elapsed). Without an allowance it is recorded and never parks."""
    assert len(calls) == len(SUBJECTS), f"{len(calls)} shim calls in the path check; expected one per subject, {len(SUBJECTS)}"
    per = {s: {"n_prompts": c["n_prompts"], "output_tokens": c["output_tokens"], "generate_s": c["t_end"] - c["t_start"],
               "mean_tokens": c["output_tokens"] / c["n_prompts"], "full_generations": REPEATS * SUBJECT_SIZES[s],
               "T_s": REPEATS * SUBJECT_SIZES[s] * c["output_tokens"] / c["n_prompts"]} for s, c in zip(SUBJECTS, calls)}
    gen = sum(v["generate_s"] for v in per.values())
    tokens = sum(v["output_tokens"] for v in per.values())
    assert 0 < gen <= wall_s, f"generation seconds {gen} outside (0, wall {wall_s}]: the timing is broken"
    T = sum(v["T_s"] for v in per.values())
    r = tokens / gen
    F = wall_s - gen
    full_s = F + T / r
    left = None if allowance_min is None else allowance_min - elapsed_min
    return {"rule": f"park when projected_full_min > {MARGIN} x (allowance_min - elapsed_min); projected = F + T / r, "
                    "T = sum over subjects of 4 x size_s x tokens_s / n_s",
            "path_check_wall_s": wall_s, "path_check_generate_s": gen, "path_check_output_tokens": tokens,
            "path_check_generations": sum(v["n_prompts"] for v in per.values()), "full_generations": FULL_GENERATIONS,
            "per_subject": per, "T_full_output_tokens": T, "r_tokens_per_s": r, "F_fixed_s": F, "projected_full_s": full_s,
            "projected_full_min": full_s / 60, "allowance_min": allowance_min, "elapsed_min": elapsed_min, "left_min": left,
            "fits_min": None if left is None else MARGIN * left, "parked": bool(left is not None and full_s / 60 > MARGIN * left)}


def scored_rows(olmes_dir) -> dict:
    """Rows with a pass_at_1 per subject in the predictions files (0 when a subject has no file)."""
    d = pathlib.Path(olmes_dir)
    return {t: sum("pass_at_1" in json.loads(l)["metrics"] for f in sorted(d.glob(f"task-*-minerva_math_{t}-predictions.jsonl"))
                   for l in f.read_text().splitlines() if l.strip()) for t in SUBJECTS}


def run_olmes(cmd, od: pathlib.Path, log: pathlib.Path, rec: dict, m: dict, out: pathlib.Path, poll_s: float, stop_batch1=False) -> int:
    """One olmes run with the A18 shim; rec (a dict inside m) gets the command, elapsed time, files, return code and shim count.
    stop_batch1 (the path check, A19 second revision): olmes runs in its own process group, and every poll reads the shim log's first
    line; if that call has n_prompts < 2 (olmes at batch size 1), the whole group (olmes' launcher, the run_eval it starts through
    /bin/sh, vLLM's engine process) is killed, rec["early_stop"] records the call and when it was seen, and the run returns. With
    batching the first line appears when the first subject's call returns; at batch size 1, seconds after the first generation."""
    env = os.environ.copy()
    # olmes' launcher runs "python -m oe_eval.run_eval" through /bin/sh; the AMI has no "python" on PATH (s2setup-03: exit 127).
    env["PATH"] = f"{pathlib.Path(OLMES_BIN).parent}:{env['PATH']}"
    env["PYTHONPATH"] = f"{SHIM_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SHIM_DIR)
    env["N13_VLLM_SHIM_LOG"] = str(log)
    rec["command"] = cmd
    rec["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(out, m)
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, start_new_session=stop_batch1)
    while p.poll() is None:
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["olmes_files"] = sorted(x.name for x in od.glob("*")) if od.exists() else []
        c = first_call(log) if stop_batch1 else None
        if c is not None and c["n_prompts"] < 2:
            seen = time.time()
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rec["early_stop"] = {"what": "the first shim call has n_prompts < 2 (olmes at batch size 1); olmes' process group killed",
                                 "first_call": c, "seen_s": round(seen - t0, 3), "lag_after_first_call_s": seen - c["t_end"],
                                 "poll_s": poll_s, "killed_s": round(time.time() - t0, 3)}
            break
        write(out, m)
        time.sleep(poll_s)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["olmes_files"] = sorted(x.name for x in od.glob("*")) if od.exists() else []
    rec["olmes_returncode"] = p.returncode
    rec["shim"] = shim_calls(log)
    write(out, m)
    return p.returncode


def rescore(a, od, rs, rescore_py) -> dict:
    subprocess.run([rescore_py or OLMES_PY, "-m", "infra.aws_pg.jobs.n13_olmes_rescore", "--dir", str(od), "--seed", "0", "--out", str(rs),
                    "--scorer", a.scorer, "--tokenizer", a.tokenizer or ""], check=True, env=os.environ.copy())
    res = json.loads(pathlib.Path(rs).read_text())
    return {k: res[k] for k in ("deranged_macro", "identity_macro", "identity_docs_equal", "n_docs", "scorer", "context_tokens")}


def run(a, olmes_cmd=None, rescore_py=None) -> int:
    """olmes_cmd(out_dir, limit) replaces the real command in the offline gate."""
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = {"step": "r0_known", "status": "running", "suite": SUITE, "target": TARGET, "tol": TOL, "tolerance": tolerance(), "engine": engine()}
    cmd_for = olmes_cmd or (lambda d, limit: command(d, a.model_type, limit))

    # the path check (A18 condition 3, A19): not a gate; its score is recorded and not read against 40.0
    odp = out / f"olmes_limit{PATH_LIMIT}"
    plog = out / f"shim_calls_limit{PATH_LIMIT}.jsonl"
    pc = m["path_check"] = {"what": f"olmes --limit {PATH_LIMIT} on the same task, model and model-args; a path check, not a gate"}
    rc = run_olmes(cmd_for(odp, PATH_LIMIT), odp, plog, pc, m, out, a.poll_s, stop_batch1=True)
    pc["rows_per_subject"] = scored_rows(odp)
    fc = first_call(plog)
    pc["batched"] = {"what": "A19: while olmes runs, the first shim call must have n_prompts >= 2, else olmes is killed within one poll "
                             "interval; at the end, at least one call with n_prompts >= 2. Failing input: batch size 1, every call n_prompts 1",
                     "first_n_prompts": None if fc is None else fc["n_prompts"], "max_n_prompts": pc["shim"]["max_n_prompts"],
                     "stopped_early": "early_stop" in pc, "pass": pc["shim"]["max_n_prompts"] >= 2 and "early_stop" not in pc}
    pc["pass"] = rc == 0 and all(v == PATH_LIMIT for v in pc["rows_per_subject"].values()) and pc["batched"]["pass"]
    write(out, m)
    if not pc["pass"]:
        es = pc.get("early_stop")
        pc["reason"] = (f"stopped early: the first shim call had n_prompts {es['first_call']['n_prompts']} (olmes at batch size 1), seen "
                        f"{es['seen_s']} s after olmes started and {es['lag_after_first_call_s']:.2f} s after that call returned" if es else
                        f"olmes exit {rc}, scored rows {pc['rows_per_subject']}, largest shim call {pc['shim']['max_n_prompts']} prompts")
        m["status"] = "failed_path_check"
        write(out, m)
        print(f"PATH CHECK FAILED: {pc['reason']}; the full task is not run", flush=True)
        return 4
    assert pc["shim"]["prompts"] == PATH_PROMPTS, pc["shim"]
    pc["read"] = read_primary(odp)
    pc["rescore"] = rescore(a, odp, out / f"rescore_limit{PATH_LIMIT}.json", rescore_py)
    write(out, m)
    print(f"PATH CHECK passed: {PATH_LIMIT} scored rows per subject, shim calls {pc['shim']['calls']} (largest {pc['shim']['max_n_prompts']} "
          f"prompts), {pc['elapsed_s']} s, primary_score {pc['read']['primary_score']:.4f} (not read against {TARGET})", flush=True)

    # the projection gate (A19): decided here, before the full task
    pj = m["projection"] = project(pc["elapsed_s"], shim_lines(plog), a.allowance_min, (time.time() - P.launch_t0(out)) / 60)
    write(out, m)
    print(f"PROJECTION: full task {pj['projected_full_min']:.1f} min (F {pj['F_fixed_s']:.0f} s + T {pj['T_full_output_tokens']:.0f} tokens "
          f"/ r {pj['r_tokens_per_s']:.1f} tokens/s); fits in {pj['fits_min']} min; {'PARKED' if pj['parked'] else 'go'}", flush=True)
    if pj["parked"]:
        (out.parent / "PARKED").write_text(json.dumps(pj))
        m["status"] = "parked"
        write(out, m)
        return 0

    # the full task
    od = out / "olmes"
    rc = run_olmes(cmd_for(od, None), od, out / "shim_calls.jsonl", m, m, out, a.poll_s)
    assert rc == 0, f"olmes exited {rc}"
    r = read_primary(od)
    m["read"] = r
    write(out, m)
    m["rescore"] = rescore(a, od, out / "rescore.json", rescore_py)
    m["G-KNOWN"] = g_known(r["primary_score"], 100 * m["rescore"]["deranged_macro"], m["rescore"]["identity_docs_equal"])
    with open(out / "samples.jsonl", "w") as f:
        for s in samples(od):
            f.write(json.dumps(s) + "\n")
    g = m["G-KNOWN"]
    ok = g["pass"] and g["failing_input"]["failed"] and g["identity"]["pass"] and r["sizes_match"]
    m["status"] = "done" if ok else "failed_gate"
    write(out, m)
    print(f"GATE G-KNOWN: {g['value']:.2f} vs {TARGET} +- {TOL}: {'pass' if g['pass'] else 'FAIL'}; deranged {g['failing_input']['value']:.2f} "
          f"({'failed' if g['failing_input']['failed'] else 'DID NOT FAIL'}); identity rescore {g['identity']['pass']}; shim calls "
          f"{m['shim']['calls']}", flush=True)
    return 0 if ok else 3


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--setup-dir")
    ap.add_argument("--allowance-min", type=float, default=None)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--scorer", default="olmes", choices=("olmes", "string"))
    ap.add_argument("--model-type", default="vllm", choices=("vllm", "hf"))
    ap.add_argument("--tokenizer", default=None, help="a tokenizer dir for the context-length count (the base snapshot)")
    ap.add_argument("--print-olmes-config", action="store_true")
    return ap


def main():
    a = parser().parse_args()
    if a.print_olmes_config:
        print(json.dumps(dynamic_config()))
        return
    from infra.pg.humboldt_core import hf_setup
    hf_setup()
    if a.setup_dir and a.tokenizer is None:
        a.tokenizer = json.loads((pathlib.Path(a.setup_dir) / "setup.json").read_text())["snapshots"]["base"]
    sys.exit(run(a))


if __name__ == "__main__":
    main()
