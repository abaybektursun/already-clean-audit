"""REALPROMPT job (studies/neuronrw/side_realprompt/PREREG.md): the NEURONRW residual push from domain prompts.

    python -m infra.aws_pg.jobs.realprompt --out D --model-key M --repo R --tokname T [--revision B] --rms S3/rms.json

For each prompt of each domain the model's poles use (infra/pg/realprompt_plan.py), one batch of rows: 10 unsteered
rows, then every direction of that domain x every positive dose x 10 samples, in plan order, chunked into generate
calls of at most MAX_ROWS rows; torch.manual_seed(SEED + 1000 * prompt_index + c) before call c of the prompt
(the Arm 2 rule). Row b is steered by neuron_rw_core.ResidualSteer with alpha_abs = dose * rms[layer] (Arm 0
rms.json) from position P - 1 on. Every call appends its rows to out/samples.jsonl and rewrites out/metrics.json.

Before generation: the identity check (alpha 0 logits equal the unhooked logits exactly on the first prompt; failing
input alpha 1, which must differ) and the realized-change failing input (the hook keyed per forward, which never
steers a decode step: its step-1 cosine must fall below COS_MIN). Either failing to behave aborts the job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

from infra.aws_pg.jobs.neuron_off import append_jsonl, dump
from infra.pg import realprompt_plan as P

SCHEMA = "realprompt/metrics/v1"
FAIL_ALPHA = 1.0


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_calls(model_key: str, max_rows: int = P.MAX_ROWS) -> list:
    """[(prompt_index, domain, prompt, c, rows)], rows = [(direction or None, dose, k)] in plan order."""
    dirs = P.directions(model_key)
    calls = []
    for dom in P.DOMAINS:
        ds = [d for d in dirs if d["domain"] == dom]
        if not ds:
            continue
        for j, prompt in enumerate(P.PROMPTS[dom]):
            rows = [(None, 0, k) for k in range(P.N_SAMPLES)]
            rows += [(d, dose, k) for d in ds for dose in P.DOSES[model_key] if dose > 0 for k in range(P.N_SAMPLES)]
            for c, s in enumerate(range(0, len(rows), max_rows)):
                calls.append((P.prompt_index(dom, j), dom, prompt, c, rows[s:s + max_rows]))
    return calls


def run(model, tok, model_key: str, rms_abs: list, out, meta: dict, max_rows: int = P.MAX_ROWS,
        sampling: dict = P.SAMPLING) -> dict:
    import torch
    from infra.pg import neuron_rw_core as R
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_layers = len(R.text_tower(model).layers)
    assert len(rms_abs) == n_layers, f"rms has {len(rms_abs)} layers, the model {n_layers}"
    width = R.text_tower(model).layers[0].mlp.down_proj.weight.shape[1]
    if model_key in P.D_MLP:
        assert width == P.D_MLP[model_key], f"MLP width {width}, plan drew controls with {P.D_MLP[model_key]}"
    dirs = {d["pole"]: d for d in P.directions(model_key)}
    w = {}
    for p in dirs:
        layer, neuron, sign = P.parse_pole(p)
        w[p] = (layer, R.pole_direction(model, layer, neuron, sign))
    calls = build_calls(model_key, max_rows)
    dump(out / "plan.json", {"model_key": model_key, "directions": list(dirs.values()), "doses": P.DOSES[model_key],
                             "rms_abs": rms_abs, "sampling": sampling, "max_rows": max_rows, "seed": P.SEED,
                             "calls": [{"prompt_index": i, "domain": dom, "prompt": pr, "c": c, "n_rows": len(rows)}
                                       for i, dom, pr, c, rows in calls]})
    m = {"schema": SCHEMA, "status": "running", "started_utc": utc(), "updated_utc": utc(), **meta,
         "gates": {}, "progress": {"calls_planned": len(calls), "calls_done": 0, "rows_written": 0},
         "throughput": {"gen_s": 0.0, "gen_rows": 0}}
    dump(out / "metrics.json", m)

    # identity: alpha 0 on the first prompt's first direction; failing input alpha FAIL_ALPHA
    i0, dom0, prompt0 = calls[0][:3]
    p0 = next(d for d in dirs.values() if d["domain"] == dom0)["pole"]
    layer0, w0 = w[p0]
    ids = torch.tensor([R.prompt_ids(tok, prompt0)])
    start = ids.shape[1] - 1
    base = R.logits_with(model, ids)
    zero = R.logits_with(model, ids, R.steer_one(model, layer0, w0, 0.0, start))
    fail = R.logits_with(model, ids, R.steer_one(model, layer0, w0, FAIL_ALPHA * rms_abs[layer0], start))
    ident = {"prompt": prompt0, "pole": p0, "max_abs_diff": float((zero.float() - base.float()).abs().max()),
             "failing_input": {"alpha": FAIL_ALPHA, "max_abs_diff": float((fail.float() - base.float()).abs().max())}}
    ident["pass"] = ident["max_abs_diff"] == 0.0
    ident["failing_input"]["fired"] = ident["failing_input"]["max_abs_diff"] > 0.0

    # realized change failing input: keyed per forward, top dose, 4 rows, 2 new tokens
    top = max(P.DOSES[model_key])
    res = R.generate_rows(model, tok, prompt0, [(layer0, w0, top * rms_abs[layer0])] * 4, P.SEED,
                          {**sampling, "max_new_tokens": 2}, keyed="per_forward")
    cos1 = [c for r in res for (st, c, _) in r["realized_steps"] if st == 1]
    assert len(cos1) == 4, f"step 1 read on {len(cos1)} of 4 rows"
    gen_fail = {"pole": p0, "dose": top, "hook": "per-forward index", "cos_step1": cos1,
                "fired": max(cos1) < P.COS_MIN}
    m["gates"] = {"identity": ident, "realized_failing_input": gen_fail}
    if not (ident["pass"] and ident["failing_input"]["fired"] and gen_fail["fired"]):
        m["status"] = "aborted_gate"
        dump(out / "metrics.json", {**m, "updated_utc": utc()})
        raise AssertionError(f"gate failed: identity {ident}, realized failing input {gen_fail}")
    dump(out / "metrics.json", {**m, "updated_utc": utc()})

    samples = out / "samples.jsonl"
    assert not samples.exists(), f"{samples} exists; use a fresh output directory"
    for i, dom, prompt, c, rows in calls:
        spec = [(None, None, 0.0) if d is None else (w[d["pole"]][0], w[d["pole"]][1], dose * rms_abs[w[d["pole"]][0]])
                for d, dose, _ in rows]
        seed = P.SEED + 1000 * i + c
        t0 = time.time()
        res = R.generate_rows(model, tok, prompt, spec, seed, sampling)
        dt = time.time() - t0
        recs = []
        for (d, dose, k), (_, _, a_abs), r in zip(rows, spec, res):
            s1 = [(cs, ra) for st, cs, ra in r["realized_steps"] if st == 1]
            assert len(s1) <= 1
            ids_ = r["completion_token_ids"]
            recs.append({"model_key": model_key, "pole": None if d is None else d["pole"],
                         "role": "unsteered" if d is None else d["role"], "target": None if d is None else d["target"],
                         "domain": dom, "prompt_index": i, "prompt": prompt, "dose": dose, "alpha_abs": a_abs, "k": k,
                         "seed": seed, "call": c, "completion": r["completion"],
                         "head60": tok.decode(ids_[:P.SHOW_TOKENS], skip_special_tokens=True),
                         "head64": tok.decode(ids_[:P.CODE_HEAD], skip_special_tokens=True),
                         "completion_token_ids": ids_, "n_new_tokens": r["n_new_tokens"],
                         "stopped_eos": r["stopped_eos"], "realized_cos": r["realized_cos"],
                         "cos_step1": s1[0][0] if s1 else None, "ratio_step1": s1[0][1] if s1 else None})
        append_jsonl(samples, recs)
        m["progress"]["calls_done"] += 1
        m["progress"]["rows_written"] += len(recs)
        m["throughput"]["gen_s"] = round(m["throughput"]["gen_s"] + dt, 2)
        m["throughput"]["gen_rows"] += len(recs)
        dump(out / "metrics.json", {**m, "updated_utc": utc()})
    m["status"] = "complete"
    dump(out / "metrics.json", {**m, "updated_utc": utc()})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True, choices=sorted(P.MODELS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokname", required=True, choices=["olmo3", "qwen3", "gemma3"])
    ap.add_argument("--revision", default=None, help="Hub branch to download (O7); absent = the cached snapshot")
    ap.add_argument("--rms", required=True, help="the model's Arm 0 out/arm0/rms.json")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    import transformers
    from infra.aws_pg.jobs.neurons_harvest import download, fetch
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    assert torch.cuda.is_available(), "no CUDA on this instance"
    spec = P.MODELS[a.model_key]
    assert (a.repo, a.revision, a.tokname, a.rms) == (spec["repo"], spec["branch"], spec["tokname"], spec["rms"]), \
        "arguments differ from realprompt_plan.MODELS"
    raw = fetch(a.rms, "rms.json").read_bytes()
    rms = json.loads(raw)
    assert rms["model_key"] == a.model_key, f"--rms is {rms['model_key']}'s file"
    meta = {"model_key": a.model_key, "repo": a.repo, "branch": a.revision, "device_name": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__, "torch": torch.__version__, "pg_git_sha": os.environ.get("PG_GIT_SHA"),
            "rms_source": {"path": a.rms, "sha256": hashlib.sha256(raw).hexdigest()}}
    if a.revision:
        local, sha, meta["download"] = download(a.repo, a.revision)
    else:
        local, sha = NC.local_snapshot(a.repo)
    assert sha == spec["revision_expected"], f"{a.repo}: snapshot {sha}, plan expects {spec['revision_expected']}"
    meta["revision"] = sha
    t0 = time.time()
    model, tok = NH.load_model(local, a.model_key)
    meta["load_s"] = round(time.time() - t0, 1)
    m = run(model, tok, a.model_key, [x["rms"] for x in rms["layers"]], a.out, meta)
    print("REALPROMPT_OK", json.dumps({"model_key": a.model_key, "status": m["status"], "progress": m["progress"]}),
          flush=True)


if __name__ == "__main__":
    main()
