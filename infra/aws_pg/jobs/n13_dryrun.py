"""N13 offline gate (studies/next/07_PICKS/N13_SPEC.md §9.3). Laptop, no model forward, no weights downloaded.

    python -m infra.aws_pg.jobs.n13_dryrun --out <dir> --hf <dir of small HF text files> --olmes <olmes source at 5a51f502>
        --wandb <dir of saved W&B run JSON> --task-doc studies/next/07_PICKS/N13_MATH_TASK.md

Prints one line per check and N13_DRYRUN_OK only when all hold. Writes every gate record to <out>/evidence/*.json.
  A. selfchecks of n13_arms and n13_score.
  B. Gates on real small files: G-GRADER (500 MATH-500 golds, both grading rules), G-TEMPLATE (3.1 and 3.0 templates, 500 problems),
     G-INIT offline half (LFS oids, the logged init revision), G-PARTITION (the 3.1 index and config), G-KNOWN configuration (olmes
     source read by ast, N13_MATH_TASK.md lines, the tolerance), S-BAND (the band from the 15 logged readings), 3.1 main vs step branches.
  C. Stand-in weights (4 layers, hidden 64, MLP width 96, 4 heads, 4 key-value heads, vocabulary 512, bf16, the real parameter names):
     setup (local), R1 (n13_rank.run), every arm built and audited; PC-RANK at real ratios on a wide stand-in; PC-ART on a stand-in with
     neuron-dependent weight scale; the rndT short branch; A20's per-block weight-change profile (recount, a planted scaled block, the
     identity pair).
  D. Scoring path: 16 fabricated problems through the generation loop with a fake engine (one with an empty Answer: line, A10; two with
     the label inside an emphasis pair, A26; one empty answer, A25); the damage screen; G-COMPLETE and its A25 failing inputs; parking.
  E. The whole job from its entry points: R0 on fabricated olmes directories (the --limit 100 path check, its failing input, the A18
     shim environment and count; A19: the early batch-size-1 stop inside the path check and the end-of-check assertion, each with its
     batch-size-1 failing input, the subject-weighted projection gate on the s2setup-06 numbers, on planted --limit 100 rates and end to
     end; the elapsed clock from the pipeline's started_utc), the A18 shim on a fake LLM
     with vLLM 0.11.0's generate signature and return type (and the same call without the shim, which must raise TypeError), A21's engine
     environment (VLLM_USE_FLASHINFER_SAMPLER=0 in the --inner child; VllmEngine stops without it), the 4
     gate arms and the 27 S4 arms on the stand-in with a fake engine (500 real MATH-500 problems, 60 fabricated GSM8K rows),
     freeze_thresholds, verdict_n13 (the verdict table), figure_n13, samples_coverage, recount_n13; the smoke's throughput record and
     the budget projection from it. A22: the smoke at 300 problems writes the engine-step log (recounted, its tokens equal to the
     answers'), the laptop projection command (n13_plan main) writes an S3 manifest from it; planted step logs: concurrency that never
     reaches 128 (the projection refuses), concurrency 300 with a known token count and wall time (the expected r_full, tail_s and
     S3 minutes), and the stop rule (spent + S3 over 35 GPU-h: no manifest). A22 addition: waiting at 24 running (the known r_sat,
     the rule min(r_full, r_sat)), and the attempt-8 reconstruction of the saturated phase on an exact FCFS engine.
  F. freeze_thresholds room true and false; verdict_n13 on planted banks: each hypothesis branch, each gate's failing input, the exit
     when a failing input does not fail.
  G. Manifests: parse with each module's own parser; each arm once; the committed s2smoke-08 manifest equals n13_plan's (A22).
  A24: the projection command on a smoke cut by the hard kill (--killed): L over the admitted prefix with the cut answers at C (or,
     --killed-reading all, every unfinished answer at C), the other smoke's rate.
  A23: R1's stage-2 deletion on a planted Hugging Face cache with two revisions under one models-- folder: main stays whole; the old
     code path (rmtree of the repo folder) is the registered wrong input and must remove main.
  A27, A28: no study ceiling (spent 34.5 GPU-h still writes the manifests); six launches (two S3, four S4) with allowance 1.3 x each
     launch's projection; zero and id are built without R1's sets.json.
  A31: a stochastic-rounding (rndE) arm on a 32-layer stand-in whose MLP magnitudes span 30 binades, in 3 shards: the written build
     equals the matched in-memory build bit for bit while the two float64 MLP energies differ in their last bits (the pre-A31 exact
     check fails, as at s4armsB1-01 and s4armsB2-01); the A31 check passes; one written entry one bf16 step up fails it.
Fail-loud: no try/except; a check that cannot run stops the dry run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

from infra.aws_pg.jobs import n13_gen as G
from infra.aws_pg.jobs import n13_known as K
from infra.aws_pg.jobs import n13_rank as R
from infra.aws_pg.jobs import n13_setup as SU
from infra.pg import n13_arms as A
from infra.pg import n13_plan as P
from infra.pg import n13_score as S

REPO = pathlib.Path(__file__).resolve().parents[3]
CHECKS = []
STANDIN_KS = [2, 4, 8, 24]
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json")


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"CHECK {name}: {'ok' if ok else 'FAIL'}" + (f"  {json.dumps(detail, default=str)[:300]}" if detail is not None else ""), flush=True)


def info(name, detail):
    print(f"INFO  {name}: {json.dumps(detail, default=str)[:400]}", flush=True)


def save(ev, name, obj):
    (ev / name).write_text(json.dumps(obj, indent=1, default=str))


def jl(p):
    return json.loads(pathlib.Path(p).read_text())


# ------------------------------------------------------------------------------------------------ B. real small files
def real_gates(a, ev):
    hf = pathlib.Path(a.hf)
    rows = [json.loads(l) for l in (hf / "math500/test.jsonl").read_text().splitlines()]
    golds = [r["answer"] for r in rows]
    t = jl(hf / "math500/tree.json")
    oid = next(x["oid"] for x in t if x["path"] == "test.jsonl")
    check("MATH-500 file: git blob sha1 equals the HF tree oid at 6e4ed1a2", SU.git_blob_sha1((hf / "math500/test.jsonl").read_bytes()) == oid,
          {"oid": oid, "rows": len(rows)})

    gr = {r: S.g_grader(golds, r) for r in ("registered", "dollar_both", "boxed")}
    save(ev, "G-GRADER.json", gr)
    for r in ("registered", "boxed"):
        info(f"G-GRADER rule '{r}'", {k: gr[r][k] for k in ("pass", "planted_eq_true", "planted_neq_false", "self_equal", "derangement_rate", "eq_failed")})
    check("G-GRADER active rule passes (24/24, 16/16, 500/500, derangement bound)", gr[S.GRADE_RULE]["pass"],
          {k: gr[S.GRADE_RULE][k] for k in ("rule", "planted_eq_true", "planted_neq_false", "self_equal", "derangement_rate", "derangement_bound")})
    check("G-GRADER failing input fails under every rule (the derangement scored as the identity)",
          all(gr[r]["failing_input"]["failed"] for r in gr),
          [gr[r]["failing_input"]["value"] for r in gr])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf / "rl31")
    probs = [r["problem"] for r in rows]
    g1, ids = S.g_template(tok, probs, "pair1")
    tok30 = AutoTokenizer.from_pretrained(hf / "rl31")
    tok30.chat_template = (hf / "rl30/chat_template.jinja").read_text()
    g1b, _ = S.g_template(tok30, probs, "pair1b")
    tmpl_sha = hashlib.sha256((hf / "rl31/chat_template.jinja").read_text().encode()).hexdigest()
    save(ev, "G-TEMPLATE.json", {"pair1": g1, "pair1b": g1b, "chat_template_sha256_pair1": tmpl_sha,
                                 "max_prompt_tokens": max(map(len, ids)), "min_prompt_tokens": min(map(len, ids))})
    check("G-TEMPLATE pair 1: 500/500 renders equal the §2.2 bytes", g1["pass"], {"n_mismatched": g1["n_mismatched"], "first_id_is_bos": g1["first_id_is_bos"]})
    check("G-TEMPLATE pair 1 failing input fails (add_generation_prompt=False differs)", g1["failing_input"]["failed"], g1["failing_input"]["tail"])
    check("chat template sha256 equals the spec's 4380e622…", tmpl_sha == "4380e62276b7ce7e28e30be381840f7cef65f20f43173437cae5d2d6908a6155", tmpl_sha)
    check("G-TEMPLATE pair 1b: 500/500 renders equal the 'math problem' bytes", g1b["pass"] and g1b["failing_input"]["failed"], g1b["n_mismatched"])

    trees = {k: {x["path"]: x["lfs"]["oid"] for x in jl(hf / f"tree_base_{k}.json") if x["path"].endswith(".safetensors")}
             for k in ("a81bae42", "39ff1a89", "c3c800dc")}
    gi = A.g_init_hash(trees["a81bae42"], trees["39ff1a89"])
    gi_f = A.g_init_hash(trees["c3c800dc"], trees["39ff1a89"])
    wb = (pathlib.Path(a.wandb) / "run_ydkgbnai.json").read_text()
    init_rev = "39ff1a89dc6950c25f4a435a993904f8587c6a3a"
    named = f"allenai/Olmo-3-1025-7B/{init_rev}" in wb
    save(ev, "G-INIT_offline.json", {"a": gi, "failing_input": gi_f, "oids": trees, "wandb_run": "ydkgbnai", "wandb_names_init": named,
                                     "wandb_field": "model_name_or_path = /gs/ai2-llm/post-training/deletable_cache_models/allenai/Olmo-3-1025-7B/" + init_rev})
    check("G-INIT (a) offline: base main shard oids equal those at the logged init 39ff1a89", gi["pass"], trees["a81bae42"])
    check("G-INIT (a) failing input fails: stage2-step47684 oids in place of main's", not gi_f["pass"])
    check("W&B run ydkgbnai names Olmo-3-1025-7B revision 39ff1a89 as model_name_or_path", named)

    cfg = jl(hf / "rl31/config.json")
    ix = jl(hf / "rl31/model.safetensors.index.json")
    names = sorted(ix["weight_map"])
    gp = A.partition(names, cfg, 7_298_011_136)
    drop = next(n for n in names if A.family(n) == "mlp")
    gpf = A.partition([n for n in names if n != drop], cfg, 7_298_011_136)
    base_names = sorted(jl(hf / "base/model.safetensors.index.json")["weight_map"])
    cfg_b = jl(hf / "base/config.json")
    same = {k: cfg.get(k) == cfg_b.get(k) for k in R.SHAPE_FIELDS}
    save(ev, "G-PARTITION_offline.json", {"pass": gp, "failing_input": gpf, "index_total_parameters": ix["metadata"]["total_parameters"],
                                          "base_names_equal": base_names == names, "config_shape_fields_equal": same,
                                          "config_differences": {k: [cfg.get(k), cfg_b.get(k)] for k in set(cfg) | set(cfg_b) if cfg.get(k) != cfg_b.get(k)}})
    check("G-PARTITION offline: 355 names in four families sum to 7,298,011,136 (3.1 index and config)", gp["pass"] and
          ix["metadata"]["total_parameters"] == 7_298_011_136, gp["per_family"])
    check("G-PARTITION failing input fails: the index without one MLP tensor", not gpf["pass"], {"missing": gpf["missing"]})
    check("base and 3.1 configs: names and shape fields equal", base_names == names and all(same.values()), same)

    st = K.static_config(a.olmes)
    ck = K.check_config(st)
    bad = json.loads(json.dumps(st))
    bad["task"]["num_shots"] = 0
    bad["model_defaults"]["chat_model"] = True
    ckf = K.check_config(bad)
    doc = pathlib.Path(a.task_doc).read_text().splitlines()
    want = {93: 'TASK_CONFIGS[f"minerva_math_{task_type}::olmes:n4:v2"] = {', 96: '"primary_metric": "pass_at_1",', 97: '"num_shots": 4,',
            98: '"fewshot_source": "Minerva:MATH:fixed",', 100: '"max_gen_toks": 1024,', 101: '"temperature": 0.6,', 102: '"top_p": 0.6,',
            104: '"stop_sequences": ["Problem:", "\\n\\n"],', 105: '"repeats": 4,', 82: 'TASK_SUITE_CONFIGS["minerva_math::olmes:n4:v2"] = {',
            84: '"primary_metric": "macro",', 285: 'olmes --model allenai/Olmo-3-1025-7B --model-type vllm \\', 286: '--task minerva_math::olmes:n4:v2 --output-dir <fresh dir>',
            288: 'Read the aggregate task `minerva_math::olmes:n4:v2`, field `primary_score` (= `pass_at_1_macro`), times 100.'}
    doc_rows = {ln: doc[ln - 1].strip() == w for ln, w in want.items()}
    tol = K.tolerance()
    cmd = K.command("out/r0_known/olmes")
    spec_cmd = ["--model", "allenai/Olmo-3-1025-7B", "--revision", "a81bae42db3975be1671e27b9c9a56da1a9f980f", "--model-type", "vllm",
                "--model-args", '{"trust_remote_code": false}', "--task", "minerva_math::olmes:n4:v2", "--gpus", "1", "--batch-size", "auto",
                "--output-dir", "out/r0_known/olmes"]
    save(ev, "G-KNOWN_config.json", {"static_config": st, "check": ck, "failing_config_check": ckf, "doc_lines": doc_rows, "tolerance": tol,
                                     "command": cmd, "spec_command_equal": cmd[1:] == spec_cmd, "target": K.TARGET, "tol": K.TOL})
    check("G-KNOWN config: olmes at 5a51f502 equals N13_MATH_TASK.md field by field (alias, shots, decoding, metric, pins, flags)", ck["pass"], ck["failed_rows"])
    check("G-KNOWN config failing input fails (num_shots 0, chat_model true)", not ckf["pass"], ckf["failed_rows"])
    check("N13_MATH_TASK.md quoted lines equal the expected config text", all(doc_rows.values()), doc_rows)
    check("G-KNOWN command equals the spec §6.1 command as amended by A18 (b) and A19 (--batch-size auto), and the path check adds only "
          "--limit 100 (A19 second revision)", cmd[1:] == spec_cmd and K.PATH_LIMIT == 100
          and K.command("out/r0_known/olmes", limit=K.PATH_LIMIT)[1:] == spec_cmd + ["--limit", "100"], " ".join(cmd))
    check("G-KNOWN tolerance recomputed: 3 x sqrt(2) x SE rounds to 1.6", abs(tol["se_one_reading"] - 0.374) < 0.001 and round(tol["three_se_difference"], 1) == 1.6, tol)
    info("olmes default max_length (default_configs.py) and the unset --model-args", st["model_defaults"])

    w = pathlib.Path(a.wandb)
    vals = []
    for run in ("ydkgbnai", "kl07u3lf"):
        h = jl(w / f"hist_m500_{run}.json")["data"]["project"]["run"]["sampledHistory"][0]
        vals += [r[k] for r in h for k in r if k.endswith("exact_match_flex") and r[k] is not None]
    sd1 = 100 * (0.89 * 0.11 / 500) ** 0.5
    band = [100 * min(vals) - 3 * 2 ** 0.5 * sd1, 100 * max(vals) + 3 * 2 ** 0.5 * sd1]
    save(ev, "S-BAND_offline.json", {"n_readings": len(vals), "readings": vals, "min": min(vals), "max": max(vals), "margin": 3 * 2 ** 0.5 * sd1,
                                     "band_exact": band, "band_registered": [81.5, 96.7]})
    check("S-BAND: 15 logged 3.1 MATH-500 readings span 0.874-0.908; band [81.46, 96.74] rounds to [81.5, 96.7]",
          len(vals) == 15 and abs(min(vals) - 0.874) < 1e-9 and abs(max(vals) - 0.908) < 1e-9 and [round(x, 1) for x in band] == [81.5, 96.7],
          {"n": len(vals), "band": band})

    main = {x["path"]: x["lfs"]["oid"] for x in jl(hf / "tree_rl31_main.json") if x["path"].endswith(".safetensors")}
    eq = sorted(p.stem[5:] for p in (hf / "rl31_steps").glob("tree_*.json")
                if {x["path"]: x["lfs"]["oid"] for x in jl(p) if x["path"].endswith(".safetensors")} == main)
    save(ev, "rl31_main_vs_steps.json", {"main_shard_oids": main, "branches_with_equal_shards": eq})
    info("3.1 main shard oids equal those of branch", eq)


# ------------------------------------------------------------------------------------------------ C. stand-in weights
def standin_cfg(hf, L=4, H=64, W=96, heads=4, kv=4, V=512):
    cfg = jl(pathlib.Path(hf) / "rl31/config.json")
    cfg.update(num_hidden_layers=L, hidden_size=H, intermediate_size=W, num_attention_heads=heads, num_key_value_heads=kv, vocab_size=V,
               layer_types=cfg["layer_types"][:L])
    return cfg


def write_snapshot(d, cfg, tensors: dict, tok_src):
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    shapes = {n: tensors[n].shape for n in names}
    chunks = [names[i::3] for i in range(3)]
    wm = {}
    for i, ch in enumerate(chunks):
        f = f"model-0000{i + 1}-of-00003.safetensors"
        A.write_safetensors(d / f, sorted(ch), shapes, lambda n: tensors[n])
        wm |= {n: f for n in ch}
    total = sum(int(np.prod(s)) for s in shapes.values())
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_parameters": total, "total_size": 2 * total}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps(cfg))
    for f in TOK_FILES:
        shutil.copyfile(pathlib.Path(tok_src) / f, d / f)
    return total


def standin_base(cfg, seed, hetero=False):
    rng = np.random.default_rng(seed)
    out = {}
    for n, s in A.expected_shapes(cfg).items():
        if A.family(n) == "norm":
            x = 1 + 0.1 * rng.standard_normal(s)
        else:
            x = 0.02 * rng.standard_normal(s)
            if hetero and A.family(n) == "mlp":
                sc = np.exp(0.5 * rng.standard_normal(cfg["intermediate_size"]))
                x = x * (sc[:, None] if "down_proj" not in n else sc[None, :])
        out[n] = A.bf16(x)
    return out


def standin_rl(base, cfg, seed, planted):
    rng = np.random.default_rng(seed)
    W = cfg["intermediate_size"]
    out = {}
    for n, b in base.items():
        fam = A.family(n)
        if fam == "mlp":
            mv = rng.random(b.shape) < 0.3
            sg = np.where(rng.random(b.shape) < 0.5, -1, 1)
            t = A.from_ordered(A.ordered(b) + mv * sg)
            l, w = A.layer_of(n), A.which_mlp(n)
            x = A.f32(t).copy()
            for p in planted:
                if p // W == l:
                    k = int(p % W)
                    if w < 2:
                        x[k, :] += 0.05 * rng.standard_normal(x.shape[1])
                    else:
                        x[:, k] += 0.05 * rng.standard_normal(x.shape[0])
            out[n] = A.bf16(x)
        elif fam in ("attn", "norm"):
            out[n] = A.bf16(A.f32(b) + 0.002 * rng.standard_normal(b.shape))
        else:
            out[n] = b.copy()
    return out


def standin_dense(base, seed):
    rng = np.random.default_rng(seed)
    return {n: A.bf16(A.f32(b) + 0.02 * rng.standard_normal(b.shape)) for n, b in base.items()}


def weights_gates(a, ev, root):
    cfg = standin_cfg(a.hf)
    tok_src = pathlib.Path(a.hf) / "rl31"
    W = cfg["intermediate_size"]
    nn = A.neuron_count(cfg)
    planted = sorted(np.random.default_rng(5).choice(nn, 8, replace=False).tolist())
    b = standin_base(cfg, 1)
    r = standin_rl(b, cfg, 2, planted)
    s2 = standin_dense(b, 3)
    snap = {k: root / "weights" / k for k in ("base", "rl", "stage2")}
    total = write_snapshot(snap["base"], cfg, b, tok_src)
    write_snapshot(snap["rl"], cfg, r, tok_src)
    write_snapshot(snap["stage2"], cfg, s2, tok_src)
    import safetensors
    from safetensors import safe_open
    with safe_open(str(snap["rl"] / "model-00001-of-00003.safetensors"), framework="numpy") as f:
        keys = list(f.keys())
    check("stand-in shards written by n13_arms.write_safetensors open with the safetensors library", len(keys) > 0,
          {"safetensors": safetensors.__version__, "keys": len(keys)})
    return cfg, snap, planted, total


def local_setup(a, out, snap, math_file, gsm_file, models="base,rl,stage2"):
    hb = {p.name: A.file_sha256(p) for p in sorted(pathlib.Path(snap["base"]).glob("*.safetensors"))}
    h2 = {p.name: A.file_sha256(p) for p in sorted(pathlib.Path(snap["stage2"]).glob("*.safetensors"))}
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    oids = pathlib.Path(out).parent / "init_oids.json"
    oids.write_text(json.dumps({"init": hb, "stage2": h2}))
    ns = argparse.Namespace(out=str(out), pair="pair1", models=models, olmes=False, local=True,
                            local_snapshots=",".join(f"{k}={v}" for k, v in snap.items()), local_math500=str(math_file),
                            local_gsm8k=str(gsm_file), local_init_oids=str(oids))
    SU.run(ns)
    return jl(pathlib.Path(out) / "metrics.json")


def rank_checks(ev, out_r1, setup_dir, cfg, snap, planted, total):
    ns = argparse.Namespace(out=str(out_r1), setup_dir=str(setup_dir), n_plant=8, n_redraw=20, ks=STANDIN_KS, total=total,
                            hash_arms="roster", kept_sd=4.0, delete_stage2=False)
    C = R.run(ns)
    save(ev, "R1_standin_controls.json", C)
    check("G-PARTITION (stand-in) passes and its failing input fails", C["G-PARTITION"]["pass"] and C["G-PARTITION"]["failing_input"]["failed"])
    gi = C["G-INIT"]
    check("G-INIT (stand-in): base chosen over stage 2; swapped candidates fail", gi["b"]["pass"] and gi["b"]["failing_input"]["failed"]
          and gi["a"]["pass"] and gi["a"]["failing_input"]["failed"], {k: gi["b"][k] for k in ("share_main", "share_other", "difference")})
    rk = np.load(pathlib.Path(out_r1) / "ranking.npz")
    check("the ranker recovers the 8 planted neurons at k = 8", sorted(rk["order"][:8].tolist()) == planted, planted)
    pc = C["PC-RANK"]
    check("PC-RANK (stand-in) passes (recovery >= 0.99)", pc["pass"], pc["recovery"])
    info("PC-RANK failing input at stand-in size (top 16 of 384 neurons, chance 0.042)", pc["failing_input"])
    sets = jl(pathlib.Path(out_r1) / "sets.json")
    check("J_floor = 1 and J_round = 1 on the planted large change at k = 8", C["J_floor"]["k1"] == 1.0 and C["J_round"]["k1"] == 1.0,
          {"J_floor": C["J_floor"], "J_round": C["J_round"]})
    art = C["PC-ART"]
    check("PC-ART (A2, iid stand-in): the H_artifact rule gives SUPPORTED on P+; its failing input P- (8 neurons moved 8 steps) gives REFUTED",
          art["pass"] and art["rule_P_plus"] == "SUPPORTED" and art["failing_input"]["failed"] and art["rule_P_minus"] == "REFUTED",
          {"P_plus": {k: art["P_plus"][k]["k1"] for k in ("J_floor", "J_round", "J_sp")}, "P_minus": {k: art["P_minus"][k]["k1"] for k in
           ("J_floor", "J_round")}, "n_planted": art["n_planted_P_minus"]})
    gh = C["G-HASH"]
    check("G-HASH (R1 hash-only builds): zero = base and id = RL, every tensor; failing input (id with (0,0) reverted) fails",
          gh["pass"] and gh["failing_input"]["failed"])
    check("G-MASK (R1): every mask arm passes; failing input (down_proj rows) fails", C["G-MASK"]["pass"] and C["G-MASK"]["failing_input"]["failed"],
          C["G-MASK"])
    hashes = jl(pathlib.Path(out_r1) / "arm_hashes.json")
    dare = {k: v["kept_rel_err"] for k, v in hashes.items() if k.startswith("dare")}
    sd = {k: 1 / np.sqrt(sets["dare_q"][k.split("_")[1]] * 4 * 3 * 64 * 96) for k in dare}
    check("DARE realized kept count within 4 binomial sd of P_k (A4), as the builder asserts", all(dare[k] <= 4 * sd[k] for k in dare)
          and all(v["kept_pass"] for k, v in hashes.items() if k.startswith("dare")), dare)
    en = {k: v["energy_rel_err"] for k, v in hashes.items() if k.startswith("rndE")}
    check("rndE realized MLP energy within 5% of E(S_k)", all(v <= 0.05 for v in en.values()), en)
    rt = {t: (sets["rndT"][t]["E"] >= sets["rndT"][t]["E_S"], sets["rndT"][t]["m"], sets["rndT"][t]["short"]) for t in A.K_TAGS}
    check("rndT reaches E(S_k) at every k (or is labeled short)", all(x[0] or x[2] for x in rt.values()), rt)
    flat = np.array([100.0, 1, 1, 1, 1, 1])
    sh = A.rnd_t(flat, np.array([0]), np.arange(1, 6), 0)
    check("rndT short branch fires when the top set holds most energy", sh["short"] and sh["m"] == 5 and sh["shortfall"] == 95.0, sh)
    return C, sets


def build_checks(ev, root, cfg, snap, sets):
    before, after = A.Weights(snap["base"]), A.Weights(snap["rl"])
    ref_b, ref_r = A.snapshot_hashes(before), A.snapshot_hashes(after)
    audits = {}
    for r in P.roster():
        arm = r["arm"]
        if arm in ("base", "rl"):
            continue
        d = root / "arms" / arm
        aud = A.build_arm(before, after, cfg, A.arm_spec(arm, sets), out_dir=d)
        (root / "arms" / f"{arm}.build.json").write_text(json.dumps(aud))
        aud = jl(root / "arms" / f"{arm}.build.json")
        rb = A.Weights(d)
        reread = {n: A.tensor_sha256(rb.get(n)) for n in rb.names()}
        audits[arm] = {"mask": aud["mask"].get("pass"), "reread_equal": reread == aud["sha256"], "kept": aud.get("kept_rel_err"),
                       "energy": aud.get("energy_rel_err")}
    check("every §3.2 arm built at stand-in size; written shards re-read hash equal to the audit", all(v["reread_equal"] for v in audits.values()),
          len(audits))
    check("G-MASK passes for every mask arm (built and written)", all(v["mask"] in (True, None) for v in audits.values()),
          {k: v["mask"] for k, v in audits.items() if v["mask"] is not None})
    zb, ib = A.Weights(root / "arms/zero"), A.Weights(root / "arms/id")
    check("G-HASH (written arms): zero = base, id = RL", A.g_hash(A.snapshot_hashes(zb), ref_b)["pass"] and A.g_hash(A.snapshot_hashes(ib), ref_r)["pass"])
    rev = A.build_arm(before, after, cfg, {"name": "id_rev00", "kind": "complement", "neurons": [0], "c": 1.0, "non_mlp": "after"})
    check("G-HASH failing input fails: id with neuron (0, 0) reverted", not A.g_hash(rev["sha256"], ref_r)["pass"], rev["families"]["mlp"])
    bad = A.build_arm(before, after, cfg, A.arm_spec("top_k1", sets), down_axis="rows")
    check("G-MASK failing input fails: top_k1 built with down_proj rows", bad["mask"]["pass"] is False)
    t1, c1 = A.Weights(root / "arms/top_k1"), A.Weights(root / "arms/comp_k1")
    merged_ok = True
    for n in after.names():
        bt, tt, ct, it = before.get(n), t1.get(n), c1.get(n), ib.get(n)
        m = np.where(tt != bt, tt, ct) if A.family(n) == "mlp" else ct
        merged_ok &= bool(np.array_equal(m, it))
    check("comp_k + top_k reproduces id bitwise (k = 8)", merged_ok)
    save(ev, "arm_builds_standin.json", audits)
    shutil.rmtree(root / "arms")


def power_checks(ev, a, root):
    """PC-RANK at the real ratios (1,000 / 352,256 planted; top 1% + planted) on a wide stand-in; PC-ART on neuron-dependent scale."""
    tok_src = pathlib.Path(a.hf) / "rl31"
    cfg = standin_cfg(a.hf, L=4, H=32, W=2048, heads=4, kv=4, V=256)
    nn = A.neuron_count(cfg)
    planted = sorted(np.random.default_rng(6).choice(nn, 8, replace=False).tolist())
    b = standin_base(cfg, 11)
    r = standin_rl(b, cfg, 12, planted)
    d = root / "wide"
    write_snapshot(d / "base", cfg, b, tok_src)
    write_snapshot(d / "rl", cfg, r, tok_src)
    wb, wr = A.Weights(d / "base"), A.Weights(d / "rl")
    s2, _, _ = A.weight_stats(wb, wr, cfg)
    k1, npl = int(round(0.01 * nn)), int(round(1000 / 352256 * nn))
    pc = A.pc_rank(wb, wr, cfg, s2, k1, npl)
    fi = A.pc_rank(wb, wr, cfg, s2, k1, npl, along="down_rows")
    save(ev, "PC-RANK_real_ratio.json", {"neurons": nn, "k_ref": k1, "n_plant": npl, "pass": pc, "failing_input": fi})
    check("PC-RANK at real ratios (8,192 neurons, 23 planted, top 105): recovery >= 0.99", pc["recovery"] >= 0.99, pc["recovery"])
    check("PC-RANK failing input at real ratios: recovery <= 0.05", fi["recovery"] <= 0.05, {"recovery": fi["recovery"], "chance": fi["chance"]})
    out = {}
    for label, het in (("iid", False), ("neuron_scale_lognormal_0.5", True)):
        cfg2 = standin_cfg(a.hf)
        bb = standin_base(cfg2, 21, hetero=het)
        dd = root / f"art_{label}"
        write_snapshot(dd, cfg2, bb, tok_src)
        w = A.Weights(dd)
        syn = A.Overlay(w, A.synthetic_rounding(w, cfg2))
        s2s, s2fs, _ = A.weight_stats(w, syn, cfg2)
        jr = A.j_round(w, syn, cfg2, s2s, STANDIN_KS)
        sp2 = np.zeros((cfg2["num_hidden_layers"], cfg2["intermediate_size"]))
        for l in range(cfg2["num_hidden_layers"]):
            sp2[l] = A.neuron_sq(*(A.spacing(x) for x in A.layer_triplet(w, l)))
        o, osp = A.order(s2s), A.order(sp2)
        out[label] = {"J_round_synthetic": jr["J_round"], "J_synthetic_vs_spacing_only": {t: A.jaccard(o[:k], osp[:k]) for t, k in zip(A.K_TAGS, STANDIN_KS)}}
    save(ev, "PC-ART_power.json", out)
    info("PC-ART: J_round(synthetic) by weight-scale model", out)
    shutil.rmtree(d)
    return out


# ------------------------------------------------------------------------------------------------ D. scoring path
FAB = [  # (problem, gold, canned answer, expected score, expected mode)
    ("What is 1/2 as a decimal?", "0.5", "Half is \\frac{1}{2}.\nAnswer: $\\frac{1}{2}$", 1, "answer_line"),
    ("Simplify 2/4.", "\\frac{1}{2}", "Work.\nThe answer is \\boxed{\\dfrac{1}{2}}.", 1, "boxed"),
    ("Compute 3+4.", "7", "3+4=8.\nAnswer: 8", 0, "answer_line"),
    ("Compute 9-2.", "7", "I am not sure what to do here.", 0, "none"),
    ("Count to many.", "10", "CAP", 0, "none"),
    ("What is 20/2?", "10", "Answer: 10.0", 1, "answer_line"),
    ("Solve x^2<9 for x in R.", "(-3,3)", "Answer: (-3, 3)", 1, "answer_line"),
    ("Each angle of an equilateral triangle?", "60^\\circ", "Answer: 60", 1, "answer_line"),
    ("Price of the ticket?", "\\$36", "Answer: 36", 1, "answer_line"),
    ("Write 3/4 as a fraction.", "\\frac{3}{4}", "  Answer: \\frac34", 1, "answer_line"),
    ("Temperature in degrees? (°)", "45^\\circ", "Answer: 45°", 1, "answer_line"),
    ("What is 2^{10}?", "1024", "Answer: 1000", 0, "answer_line"),
    ("Compute 2+3.", "5", "Work.\nAnswer:\n\\boxed{5}", 1, "boxed"),          # A10: an empty Answer: line is skipped
    ("What is 6 times 7?", "42", "Work.\n**Answer:** 42", 1, "answer_line"),    # A26: the label inside an emphasis pair
    ("Compute 1+4.", "5", "Work.\n**Answer:**\n\\boxed{5}", 1, "boxed"),    # A26 with A10: an empty emphasised label is skipped
    ("Say nothing.", "1", "", 0, "none"),                                     # A25: the stop token alone (empty text, n_generated 1)
]
A26_ROWS = {13: "42", 14: "5"}
A25_EMPTY = 15


def derived_setup(src, dst, math_rows):
    """A copy of a setup step's output whose MATH-500 rows are replaced (prompt ids re-banked with the same tokenizer)."""
    from transformers import AutoTokenizer
    shutil.copytree(src, dst)
    sj = jl(pathlib.Path(dst) / "setup.json")
    probs = jl(sj["problems"])
    probs["math500"] = math_rows
    (pathlib.Path(dst) / "problems.json").write_text(json.dumps(probs))
    tok = AutoTokenizer.from_pretrained(sj["tokenizer_dir"])
    (pathlib.Path(dst) / "prompt_ids.json").write_text(json.dumps({"pair": "pair1", "ids": [S.render_ids(tok, r["problem"]) for r in math_rows]}))
    sj |= {"problems": str(pathlib.Path(dst) / "problems.json"), "prompt_ids": str(pathlib.Path(dst) / "prompt_ids.json")}
    (pathlib.Path(dst) / "setup.json").write_text(json.dumps(sj))
    return pathlib.Path(dst)


def fab_rows(n):
    return [{"index": i, "problem": FAB[i % len(FAB)][0], "answer": FAB[i % len(FAB)][1], "unique_id": f"fab/{i}", "row_sha256": ""} for i in range(n)]


def gen_ns(out, arm, setup_dir, bench="math500", problems="0:12", cap=64, sets=None, smoke=False, allowance=None, arm_root=None):
    return argparse.Namespace(out=str(out), arm=arm, bench=bench, problems=problems, pair="pair1", setup_dir=str(setup_dir), cap=cap,
                              cap_from=None, sets=sets, only_if=None, smoke=smoke, arm_root=str(arm_root or pathlib.Path(out).parent / "arms"),
                              kept_sd=4.0, allowance_min=allowance, proj_tok_s=500.0, proj_mean_math=2500.0, proj_mean_gsm=940.0, proj_load_min=5.0,
                              inner=True)


def fake_factory(canned, clock, per_step=3, dt=30.0):
    return lambda model_dir, tok_dir, cap, tok: G.FakeEngine(tok, canned, cap, clock, per_step=per_step, dt=dt)


def scoring_checks(ev, root, a, setup_src):
    sd = derived_setup(setup_src, root / "fab/out/setup", fab_rows(len(FAB)))
    canned = {f"math500:{i}": c for i, (_p, _g, c, _s, _m) in enumerate(FAB)}
    clock = G.Clock()
    ms = G.run(gen_ns(root / "fab/out/s_base", "base", sd, problems=f"0:{len(FAB)}"), engine_factory=fake_factory(canned, clock), clock=clock)
    d = root / "fab/out/s_base/base/math500"
    recs = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines()]
    got = {r["problem_index"]: (r["score"], r["mode"]) for r in recs}
    exp = {i: (s, m) for i, (_p, _g, _c, s, m) in enumerate(FAB)}
    save(ev, "scoring_fabricated.json", {"expected": exp, "got": got, "records": recs})
    check(f"scoring path: {len(FAB)} fabricated answers scored and moded as expected (one empty Answer: line, A10; two emphasised "
          "labels, A26; one empty answer, A25)", got == exp, {i: (got[i], exp[i]) for i in got if got[i] != exp[i]})
    by = {r["problem_index"]: r for r in recs}
    a26 = {i: {"text": by[i]["text"], "mode": by[i]["mode"], "extracted": by[i]["extracted"], "score": by[i]["score"],
               "answer_tokens": by[i]["answer_tokens"]} for i in A26_ROWS}
    check("A26 planted: '**Answer:** 42' extracts '42' (answer_line, right); '**Answer:**' with nothing after it falls to the box (A10)",
          all(by[i]["extracted"] == x for i, x in A26_ROWS.items()) and by[13]["mode"] == "answer_line" and by[14]["mode"] == "boxed"
          and by[13]["answer_tokens"] >= 1, a26)
    cap_rec = next(r for r in recs if r["problem_index"] == 4)
    check("cap hit recorded (n_new = C, finish_reason length)", cap_rec["cap_hit"] and cap_rec["n_new"] == 64 and cap_rec["finish_reason"] == "length")
    sm = [json.loads(l) for l in (d / "samples.jsonl").read_text().splitlines()]
    roles = {x for s in sm for x in s["roles"]}
    check("samples.jsonl holds problems 0-2, one right, one wrong, one cap hit", {"fixed_0", "fixed_1", "fixed_2", "lowest_right", "lowest_wrong",
                                                                                   "cap_hit"} <= roles, sorted(roles))
    m = jl(d / "metrics.json")
    check("metrics.json rewritten at least three times inside the loop", m["n_metrics_writes"] >= 3, m["n_metrics_writes"])
    check("answer-line log-probability banked beside every extracted answer", all(r["answer_logprob"] is not None for r in recs if r["mode"] != "none"))
    check("G-COMPLETE passes on the bank and fails with one answer removed", m["G-COMPLETE"]["pass"] and not S.g_complete(recs[:-1], len(FAB), "A")["pass"])
    e = by[A25_EMPTY]
    bank_a25 = {"empty_record": {k: e[k] for k in ("text", "n_new", "n_generated", "finish_reason", "mode", "score")}}
    missing = [r for r in recs if r["problem_index"] != 3]
    gen0 = [dict(r, n_generated=0) if r["problem_index"] == 3 else r for r in recs]
    nofin = [dict(r, finish_reason=None) if r["problem_index"] == 3 else r for r in recs]
    bank_a25 |= {"bank": m["G-COMPLETE"], "missing_record": S.g_complete(missing, len(FAB), "A"),
                 "n_generated_0": S.g_complete(gen0, len(FAB), "A"), "finish_reason_none": S.g_complete(nofin, len(FAB), "A"),
                 "last_removed": S.g_complete(recs[:-1], len(FAB), "A")}
    save(ev, "A25_g_complete.json", bank_a25)
    check("A25 G-COMPLETE: the bank with one empty answer (n_generated 1, finish_reason stop) passes with n_empty 1; a missing record, "
          "a record with n_generated 0 and a record with no finish_reason each fail",
          e["text"] == "" and e["n_generated"] == 1 and e["finish_reason"] == "stop" and m["G-COMPLETE"]["pass"]
          and m["G-COMPLETE"]["n_empty"] == 1 and not any(bank_a25[k]["pass"] for k in ("missing_record", "n_generated_0", "finish_reason_none")),
          {k: v["pass"] for k, v in bank_a25.items() if k != "empty_record"})
    big = root / "screen"
    ss = derived_setup(setup_src, big / "out/setup", fab_rows(40))
    canned20 = {f"math500:{i}": ("CAP" if i < 16 else "Answer: nothing") for i in range(40)}
    shutil.copyfile(root / "e2e/out/r1_rank/sets.json", big / "sets.json")
    res = {}
    for arm in ("dare_k0.1", "base"):
        clock = G.Clock()
        ms = G.run(gen_ns(big / "out" / arm, arm, ss, problems="0:40", cap=32, sets=str(big / "sets.json")),
                   engine_factory=fake_factory(canned20, clock, per_step=4), clock=clock)
        res[arm] = {"status": ms["math500"]["status"], "screen": ms["math500"]["screen"], "finished": ms["math500"]["counts"]["finished"]}
    save(ev, "damage_screen.json", res)
    check("damage screen fires on a non-endpoint arm (16/20 cap hits, none correct) and aborts it", res["dare_k0.1"]["status"] == "screened"
          and res["dare_k0.1"]["screen"]["fired"] and res["dare_k0.1"]["finished"] < 40, res["dare_k0.1"])
    check("damage screen is only recorded on base (an endpoint): every answer finishes", res["base"]["status"] == "done" and res["base"]["finished"] == 40,
          res["base"])
    # parking
    pk = root / "park/out"
    (pk / "setup").mkdir(parents=True)
    (pk / "pipeline_status.json").write_text(json.dumps({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))}))
    m1 = G.run(gen_ns(pk / "step_a", "base", sd, allowance=60.0), engine_factory=fake_factory(canned, G.Clock()))
    m2 = G.run(gen_ns(pk / "step_b", "rl", sd, allowance=1e6), engine_factory=fake_factory(canned, G.Clock()))
    check("stop rule: a step over its allowance is PARKED, and so is every later step", m1["math500"]["status"] == "parked"
          and m2["math500"]["status"] == "parked", [m1["math500"]["park"], m2["math500"]["park"]["earlier_parked"]])


# ------------------------------------------------------------------------------------------------ E. the whole job
P_ARM = {"base": 0.30, "zero": 0.30, "rl": 0.92, "id": 0.92, "top_k0.1": 0.33, "top_k0.3": 0.45, "top_k1": 0.80, "top_k3": 0.88, "top_k1_B": 0.80,
         "topF_k1": 0.78, "comp_k0.1": 0.40, "comp_k0.3": 0.40, "comp_k1": 0.40, "comp_k3": 0.38, "rndE_k0.1": 0.32, "rndE_k0.3": 0.32, "rndE_k1": 0.32,
         "rndE_k3": 0.33, "rndT_k0.1": 0.33, "rndT_k0.3": 0.33, "rndT_k1": 0.33, "rndT_k3": 0.34, "rnd0_k1": 0.31, "rndT_k1_d2": 0.33,
         "dare_k0.1": None, "dare_k0.3": 0.30, "dare_k1": 0.36, "dare_k3": 0.50, "dare_k1_d2": 0.36, "mlp": 0.85, "attn": 0.31}


def canned_for(arm, probs, gsm, seed_set):
    D = np.random.default_rng(7).random(len(probs))
    Dg = np.random.default_rng(8).random(len(gsm))
    p = P_ARM[arm]
    out = {}
    for bench, rows, dd in (("math500", probs, D), ("gsm8k", gsm, Dg)):
        for r in rows:
            i = r["index"]
            if p is None:
                out[f"{bench}:{i}"] = "CAP" if i < 16 else "Answer: nothing"
                continue
            e = np.random.default_rng([S.SEED_BASE[seed_set] + i, 1 if bench == "gsm8k" else 0]).random()
            ok = (dd[i] < p) if e >= 0.1 else (e < 0.05)
            fmt = int(1000 * e) % 10
            if ok:
                out[f"{bench}:{i}"] = f"Reasoning.\nAnswer: {r['answer']}" if fmt else f"So the result is \\boxed{{{r['answer']}}}."
            else:
                out[f"{bench}:{i}"] = "Reasoning.\nAnswer: wrong" if fmt != 3 else "I cannot finish."
    return out


def fab_olmes(d, primary=0.40, only_micro=False, n=10):
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, s in enumerate(K.SUBJECTS):
        preds, reqs = [], []
        for i in range(n):
            lab = f"{j}{i}"
            right = i < 4 * n // 10
            conts = [f"Work. Final Answer: The final answer is ${lab if right else 'x'}$. I hope it is correct."] * 4
            preds.append({"doc_id": i, "native_id": i, "label": lab, "model_output": [{"continuation": c, "model_answer": None} for c in conts],
                          "metrics": {"pass_at_1": 1.0 if right else 0.0}})
            reqs += [{"doc_id": i, "request": {"context": f"Problem:\nshot\n\nSolution: s\n\nProblem:\nQ{j}{i}\n\nSolution:"}, "idx": k} for k in range(4)]
        (d / f"task-{j:03d}-minerva_math_{s}-predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in preds))
        (d / f"task-{j:03d}-minerva_math_{s}-requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reqs))
        (d / f"task-{j:03d}-minerva_math_{s}-metrics.json").write_text(json.dumps({"task_config": {"context_kwargs": {"use_cot": True, "cot_style": "minerva"}}}))
        rows.append({"task_name": f"minerva_math_{s}", "metrics": {"primary_score": primary}, "num_instances": K.SUBJECT_SIZES[s], "task_config": {}})
    agg = {"pass_at_1_micro": primary} if only_micro else {"primary_score": primary, "primary_score_macro": primary, "pass_at_1_micro": primary + 0.01}
    rows.insert(0, {"task_name": K.SUITE, "metrics": agg, "num_instances": 5000, "task_config": {"primary_metric": "macro"}})
    (d / "metrics-all.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


FAKE_TOKENS = 245          # output tokens per prompt in a batch-size-1 fake log line (the s2setup-06 path check's mean was 245.3)
A6 = REPO / "studies/next/n13/r0/s2setup-06/path_check_tokens.json"     # s2setup-06's path check, per subject (path_check_tokens.py)


def batched_calls(n=K.PATH_LIMIT * K.REPEATS) -> list:
    """The batched path check's shim calls, [n_prompts, output_tokens] per subject in SUBJECTS order: n prompts at s2setup-06's mean
    output length for that subject (path_check_tokens.json -> per_subject.*.mean_tokens; 400 x mean = 5 x that subject's tokens)."""
    a6 = jl(A6)["per_subject"]
    return [[n, round(n * a6[s]["mean_tokens"])] for s in K.SUBJECTS]


BATCH1_CALLS = [[1, FAKE_TOKENS]] * (len(K.SUBJECTS) * K.PATH_LIMIT * K.REPEATS)     # batch size 1: 2,800 calls of 1 prompt
FAKE_RUN_EVAL = (
    "import json, os, shutil, sys, time\n"
    "src, od, calls, sleep_s, call_s = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])\n"
    "time.sleep(sleep_s)\n"
    "for n, tok in calls:\n"
    "    t0 = time.time(); time.sleep(call_s); t1 = time.time()\n"
    "    open(os.environ['N13_VLLM_SHIM_LOG'], 'a').write(json.dumps({'pid': os.getpid(), 'n_prompts': n, 't_start': t0, 't_end': t1, "
    "'output_tokens': tok, 'pythonpath': os.environ['PYTHONPATH']}) + '\\n')\n"
    "shutil.copytree(src, od, dirs_exist_ok=True)\n")
FAKE_LAUNCHER = "import subprocess, sys; sys.exit(subprocess.run([sys.executable, '-I', '-c', *sys.argv[1:]]).returncode)"


def fake_olmes(src_path, src_full, calls=None, sleep_s=0.2, call_s=0.01):
    """A stand-in for the olmes command, in two processes as olmes' launcher runs run_eval in a child: the child sleeps sleep_s
    seconds (start-up), logs one shim line per call in the shim's format with the PYTHONPATH it saw (t_start and t_end around a sleep of
    call_s seconds, n_prompts, output_tokens), then copies a fabricated output directory into place. Path check: calls, a list of
    [n_prompts, output_tokens], default batched_calls() (7 calls of 400 prompts); BATCH1_CALLS is batch size 1, as in s2setup-06.
    Full run: one call of 2 prompts. Run with -I, so the shim's sitecustomize (which imports vllm) is not loaded on the laptop."""
    calls = batched_calls() if calls is None else calls
    return lambda od, limit: [sys.executable, "-I", "-c", FAKE_LAUNCHER, FAKE_RUN_EVAL, str(src_path if limit else src_full), str(od),
                              json.dumps(calls if limit else [[2, 2 * FAKE_TOKENS]]), str(sleep_s if limit else 0.0),
                              str(call_s if limit else 0.0)]


FAKE_LLM = """
import json, sys, time, types, typing
from infra.aws_pg.jobs.n13_olmes_shim import sitecustomize as SC     # inert here: N13_VLLM_SHIM_LOG is not set
class LLM:                                                             # vLLM 0.11.0's signature (vllm/entrypoints/llm.py line 335)
    def generate(self, prompts, sampling_params=None, *, use_tqdm=True, lora_request=None, priority=None):
        self.seen = {"prompts": prompts, "sampling_params": sampling_params, "use_tqdm": use_tqdm, "lora_request": lora_request, "priority": priority}
        time.sleep(0.05)
        # vLLM 0.11.0 returns list[RequestOutput]; RequestOutput.outputs is a list of CompletionOutput, each with token_ids (vllm/outputs.py).
        # Planted: completions of 3 tokens for the first prompt, of 5 and 2 tokens for the second (two completions): 10 in all.
        self.ret = [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[11, 12, 13])]),
                    types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[21, 22, 23, 24, 25]), types.SimpleNamespace(token_ids=[31, 32])])
                    ][:len(prompts)]
        return self.ret
class TokensPrompt(typing.TypedDict):
    prompt_token_ids: list
class SamplingParams:
    pass
sp = SamplingParams()
"""
SHIMMED = FAKE_LLM + """
SC.install(LLM, TokensPrompt, sys.argv[1])
llm, llm2 = LLM(), LLM()
out = llm.generate(prompt_token_ids=[[100257, 5, 6], [7]], sampling_params=sp, use_tqdm=False)
out2 = llm2.generate([TokensPrompt(prompt_token_ids=[1])], sp)
r, r2 = llm.seen, llm2.seen
print(json.dumps({"prompts": r["prompts"], "sp_same": r["sampling_params"] is sp, "use_tqdm": r["use_tqdm"], "lora": r["lora_request"],
                  "returned_unchanged": out is llm.ret, "new_signature": [r2["prompts"], r2["sampling_params"] is sp, out2 is llm2.ret]}))
"""
UNSHIMMED = FAKE_LLM + """
LLM().generate(prompt_token_ids=[[100257, 5, 6], [7]], sampling_params=sp, use_tqdm=False)
"""


def shim_checks(ev):
    """A18 condition 2: the shim on a fake LLM whose generate accepts only vLLM 0.11.0's signature; the failing input is the same call
    without the shim, which must raise TypeError."""
    d = pathlib.Path(ev).parent / "shim"
    d.mkdir()
    log = d / "calls.jsonl"
    p = subprocess.run([sys.executable, "-c", SHIMMED, str(log)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)
    calls = [json.loads(l) for l in log.read_text().splitlines()]
    ok = (r["prompts"] == [{"prompt_token_ids": [100257, 5, 6]}, {"prompt_token_ids": [7]}] and r["sp_same"] and r["use_tqdm"] is False
          and r["lora"] is None and r["returned_unchanged"] and r["new_signature"] == [[{"prompt_token_ids": [1]}], True, True]
          and [(c["n_prompts"], c["output_tokens"]) for c in calls] == [(2, 10)] and calls[0]["t_end"] - calls[0]["t_start"] >= 0.049
          and "N13 vLLM shim installed" in p.stderr)
    f = subprocess.run([sys.executable, "-c", UNSHIMMED], capture_output=True, text=True)
    sha = hashlib.sha256((K.SHIM_DIR / "sitecustomize.py").read_bytes()).hexdigest()
    save(ev, "A18_shim.json", {"shimmed": r, "calls": calls, "stderr": p.stderr, "unshimmed_returncode": f.returncode,
                               "unshimmed_error": f.stderr.strip().splitlines()[-1], "shim": str(K.SHIM_DIR / "sitecustomize.py"), "shim_sha256": sha})
    check("A18 shim: a two-prompt generate(prompt_token_ids=...) call becomes generate([TokensPrompt x 2], ...), the sampling params pass "
          "through unchanged (same object), use_tqdm kept, the return value passed back unchanged, a new-signature call untouched; one "
          "converted call logged with its wall time (>= the fake's 0.05 s) and output_tokens 10 (planted: 3 + 5 + 2 token ids over three "
          "completions, A19 revision)", ok, {"seen": r, "calls": calls})
    check("A18 shim failing input: the same call without the shim raises TypeError", f.returncode != 0 and "TypeError" in f.stderr,
          f.stderr.strip().splitlines()[-1])
    info("A18 shim file and sha256", {"path": str(K.SHIM_DIR / "sitecustomize.py"), "sha256": sha})


def shim_lines_from(per: list) -> list:
    """Shim log lines from [n_prompts, output_tokens, seconds] per call, the calls back to back."""
    out, t = [], 0.0
    for n, tok, sec in per:
        out.append({"n_prompts": n, "t_start": t, "t_end": t + sec, "output_tokens": tok})
        t += sec
    return out


def projection_table(ev):
    """A19 second revision (2026-11-05): the subject-weighted projection gate on the s2setup-06 path check (registered failing input;
    its 560 one-prompt calls summed into one call per subject, the subject's generation phase as its seconds) and on planted --limit 100
    path checks at 300, 500 and 1,000 tokens/s with s2setup-06's per-subject mean lengths (400 prompts per subject call); on each row
    also the first revision's unweighted value, F + 20,000 / generations x generation seconds. Sources:
    studies/next/n13/r0/s2setup-06/path_check_tokens.json (per_subject tokens, generations, mean_tokens, seconds_to_next_subject;
    path_check_wall_s 3,120; full_task_tokens_estimate 4,643,969.85). Planted, not measured: F = 132 s (90 s start-up + 7 x 6 s
    set-up, the third executor's estimate); elapsed at the decision = 15.9 min setup (s2setup-06) + the path-check wall + 3.5 min of
    rescore (s2setup-06's about 0.7 min for 140 documents, scaled by 100 / 20 to 700 documents)."""
    a6 = jl(A6)
    ps = [a6["per_subject"][s] for s in K.SUBJECTS]
    A, F7, setup_min, rescore_min = 280.0, 132.0, 15.9, 0.7 * K.PATH_LIMIT / 20
    rows = {"s2setup06_replay": dict(wall_s=a6["path_check_wall_s"], elapsed_min=0.0,
                                     calls=shim_lines_from([[v["generations"], v["tokens"], v["seconds_to_next_subject"]] for v in ps]))}
    for r in (300.0, 500.0, 1000.0):
        per = [[n, tok, tok / r] for n, tok in batched_calls()]
        g = sum(x[2] for x in per)
        rows[f"limit100_{int(r)}_tok_s"] = dict(wall_s=F7 + g, elapsed_min=setup_min + (F7 + g) / 60 + rescore_min, calls=shim_lines_from(per))
    out = {}
    for k, v in rows.items():
        pj = K.project(v["wall_s"], v["calls"], A, v["elapsed_min"])
        unw = (pj["F_fixed_s"] + K.FULL_GENERATIONS / pj["path_check_generations"] * pj["path_check_generate_s"]) / 60
        out[k] = pj | {"first_revision_unweighted_min": unw, "first_revision_parked": bool(unw > pj["fits_min"])}
    # the slowest rate that goes at F = 132 s: F + T / r <= 0.85 x (60 (A - setup - rescore) - F - tok / r), in seconds
    tok = out["limit100_500_tok_s"]["path_check_output_tokens"]
    T = out["limit100_500_tok_s"]["T_full_output_tokens"]
    r_min = (T + K.MARGIN * tok) / (K.MARGIN * 60 * (A - setup_min - rescore_min) - (1 + K.MARGIN) * F7)
    out["boundary_at_F_132_s"] = {"slowest_r_tokens_per_s_that_goes": r_min, "path_check_generate_s": tok / r_min,
                                  "per_subject_s": tok / r_min / 7, "projected_full_min": (F7 + T / r_min) / 60,
                                  "elapsed_min": setup_min + rescore_min + (F7 + tok / r_min) / 60}
    out["inputs"] = {"allowance_min": A, "source": str(A6.relative_to(REPO)), "s2setup06_full_task_tokens_estimate": a6["full_task_tokens_estimate"],
                     "planted_F_s": F7, "setup_min": setup_min, "rescore_min": rescore_min, "path_check_calls": batched_calls(),
                     "elapsed_rule": "planted rows: setup + the path-check wall + rescore; s2setup-06 row: 0, the whole allowance left"}
    save(ev, "A19_projection.json", out)
    a6r, b3, b5, b10 = out["s2setup06_replay"], out["limit100_300_tok_s"], out["limit100_500_tok_s"], out["limit100_1000_tok_s"]
    check("A19 full-task generations: the 7 subject test sizes (sum 5,000) x 4 repeats = 20,000; the path check 7 x 100 x 4 = 2,800",
          sum(K.SUBJECT_SIZES.values()) == 5000 and K.REPEATS == 4 and K.FULL_GENERATIONS == 20000 and K.PATH_PROMPTS == 2800, K.SUBJECT_SIZES)
    check("A19 projection gate failing input (second revision): the s2setup-06 path check, one call per subject (137,386 tokens in 3,030 s "
          "of generation, 45.3 tokens/s, F 90 s), parks at allowance 280 with nothing elapsed; its subject-weighted T equals "
          "path_check_tokens.py's full-task estimate 4,643,969.85", a6r["parked"] and abs(a6r["r_tokens_per_s"] - 45.342) < 0.001
          and abs(a6r["F_fixed_s"] - 90.0) < 1e-9 and abs(a6r["T_full_output_tokens"] - a6["full_task_tokens_estimate"]) < 1e-3,
          {x: a6r[x] for x in ("T_full_output_tokens", "r_tokens_per_s", "F_fixed_s", "projected_full_min", "fits_min", "parked")})
    check("A19 projection on planted --limit 100 path checks (400 prompts per subject at s2setup-06's per-subject mean lengths, F 132 s, "
          "elapsed = setup + wall + rescore): 300 tokens/s parks, 500 and 1,000 tokens/s go", b3["parked"] and not b5["parked"] and not b10["parked"],
          {k: {x: v[x] for x in ("projected_full_min", "fits_min", "elapsed_min", "parked")} for k, v in (("300", b3), ("500", b5), ("1000", b10))}
          | {"boundary": out["boundary_at_F_132_s"]})
    p = subprocess.run([sys.executable, "-c", "from infra.aws_pg.jobs.n13_known import project; "
                        "project(100.0, [{'n_prompts': 400, 't_start': 0.0, 't_end': 120.0 / 7, 'output_tokens': 1000}] * 7, 280.0, 0.0)"],
                       capture_output=True, text=True)
    check("A19 projection stops on broken timing: generation seconds above the wall seconds raise", p.returncode != 0 and "AssertionError" in p.stderr,
          p.stderr.strip().splitlines()[-1])
    p = subprocess.run([sys.executable, "-c", "from infra.aws_pg.jobs.n13_known import project; "
                        "project(100.0, [{'n_prompts': 400, 't_start': 0.0, 't_end': 1.0, 'output_tokens': 1000}] * 6, 280.0, 0.0)"],
                       capture_output=True, text=True)
    check("A19 projection stops unless the path check logged exactly 7 calls, one per subject (6 calls raise)",
          p.returncode != 0 and "AssertionError" in p.stderr and "expected one per subject" in p.stderr, p.stderr.strip().splitlines()[-1])


CLOCK_PROBE = """import argparse, json, pathlib, time
from infra.pg import n13_plan as P
ap = argparse.ArgumentParser()
ap.add_argument("--out")
out = pathlib.Path(ap.parse_args().out)
(out / "clock.json").write_text(json.dumps({"launch_t0": P.launch_t0(out), "now": time.time(),
                                             "steps_json_mtime": (out.parent / "steps.json").stat().st_mtime}))
"""
PIPELINE_LOCAL = """import json, pathlib, shutil, sys
from infra.aws_pg.jobs import pipeline as PL
src = sys.argv[1]
def read_local(steps_s3, out):
    local = out / "steps.json"
    shutil.copy2(src, local)                  # keeps the source's old mtime, as aws s3 cp sets the object's LastModified
    return json.loads(local.read_text())
PL._read_steps = read_local
sys.argv = ["pipeline", "--out", sys.argv[2], "--steps-s3", "s3://not-read"]
PL.main()
"""


def clock_checks(ev, root):
    """A19 revision: the job's elapsed time counts from the pipeline's own start (out/pipeline_status.json -> started_utc), not from the
    mtime of out/steps.json. Registered failing input: a steps.json whose mtime is 1 h old (aws s3 cp gives it the upload time); the old
    clock read 60 min there. The real pipeline.main runs one probe step; only its S3 download is replaced by a local copy."""
    root.mkdir(parents=True)
    (root / "mod").mkdir()
    (root / "mod" / "n13_clock_probe.py").write_text(CLOCK_PROBE)
    steps = root / "steps_uploaded.json"
    steps.write_text(json.dumps([{"name": "probe", "module": "n13_clock_probe", "args": ""}]))
    os.utime(steps, (time.time() - 3600, time.time() - 3600))
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{root / 'mod'}"}
    t_launch = time.time()
    p = subprocess.run([sys.executable, "-c", PIPELINE_LOCAL, str(steps), str(root / "out")], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    ps = jl(root / "out/pipeline_status.json")
    c = jl(root / "out/probe/clock.json")
    rec = {"pipeline_status": ps, "probe": c, "dryrun_time_before_launch": t_launch, "elapsed_min_new_clock": (c["now"] - c["launch_t0"]) / 60,
           "elapsed_min_old_clock": (c["now"] - c["steps_json_mtime"]) / 60, "launch_t0_minus_launch_s": c["launch_t0"] - t_launch}
    save(ev, "A19_clock.json", rec)
    check("A19 elapsed clock: pipeline.py writes started_utc at its first write and a step's launch_t0 reads it (within 2 s of the "
          "pipeline's start); the 1-h-old steps.json mtime (the old clock: 60 min elapsed) does not move it", ps["state"] == "complete"
          and "started_utc" in ps and abs(rec["launch_t0_minus_launch_s"]) < 2.0 and rec["elapsed_min_new_clock"] < 0.1
          and abs(rec["elapsed_min_old_clock"] - 60.0) < 0.5, {k: rec[k] for k in ("elapsed_min_new_clock", "elapsed_min_old_clock",
                                                                                 "launch_t0_minus_launch_s")} | {"started_utc": ps["started_utc"]})


def block_profile_checks(ev, out_r1, cfg, snap):
    """A20: R1 writes block_profile into metrics.json and sets.json (informational; no rule reads it). Three readings: the stand-in RL
    change recounted from the snapshots by a separate float64 sum; a planted input (block PB's attention and MLP tensors scaled by 2, so
    each of its two groups changes by exactly ||2w - w|| / ||w|| = 1 and every other group by 0); the identity pair (base against itself,
    0 everywhere), on which the planted expectation fails."""
    base, rl = A.Weights(snap["base"]), A.Weights(snap["rl"])
    out_r1 = pathlib.Path(out_r1)
    bp_m, bp_s = jl(out_r1 / "metrics.json")["block_profile"], jl(out_r1 / "sets.json")["block_profile"]
    L = cfg["num_hidden_layers"]
    groups = {g: [[f"model.layers.{l}.{k}.weight" for k in keys] for l in range(L)] for g, keys in (("attn", A.ATTN), ("mlp", A.MLP))}

    def rel(after, names):
        d = sum(float(np.sum((A.f32(after.get(n)).astype(np.float64) - A.f32(base.get(n)).astype(np.float64)) ** 2)) for n in names)
        b = sum(float(np.sum(A.f32(base.get(n)).astype(np.float64) ** 2)) for n in names)
        return float(np.sqrt(d / b))

    def flat(bp):
        return bp["attn"] + bp["mlp"] + [bp["embed"], bp["lm_head"]]

    rc = {g: [rel(rl, names) for names in groups[g]] for g in groups} | {"embed": rel(rl, ["model.embed_tokens.weight"]),
                                                                        "lm_head": rel(rl, ["lm_head.weight"])}
    diff = max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(flat(bp_m), flat(rc)))
    check("A20: R1 writes block_profile (attention and MLP per block, embedding, head) into metrics.json and sets.json, equal in both, "
          f"{L} blocks; a separate float64 recount from the stand-in snapshots matches (largest relative difference < 1e-9)",
          bp_m == bp_s and bp_m["n_blocks"] == L and len(bp_m["attn"]) == len(bp_m["mlp"]) == L and diff < 1e-9,
          {"attn": [round(x, 5) for x in bp_m["attn"]], "mlp": [round(x, 5) for x in bp_m["mlp"]], "embed": bp_m["embed"],
           "lm_head": bp_m["lm_head"], "max_rel_diff": diff})
    PB, FAC = 2, 2.0
    over = A.Overlay(base, {n: A.bf16(FAC * A.f32(base.get(n))) for g in groups for n in groups[g][PB]})
    pl = A.weight_stats(base, over, cfg)[2]["block_profile"]
    idp = A.weight_stats(base, base, cfg)[2]["block_profile"]
    want = [abs(FAC - 1) if l == PB else 0.0 for l in range(L)]
    err = max(abs(x - y) for x, y in zip(flat(pl), want + want + [0.0, 0.0]))
    fails_on_identity = max(abs(x - y) for x, y in zip(flat(idp), want + want + [0.0, 0.0])) > 0.5
    save(ev, "A20_block_profile.json", {"standin_rl": bp_m, "recount": rc, "max_rel_diff": diff, "planted": {"block": PB, "factor": FAC,
                                        "profile": pl, "expected_attn_mlp": want, "max_abs_err": err}, "identity": idp,
                                        "planted_expectation_fails_on_identity": fails_on_identity})
    check(f"A20 planted input: block {PB}'s attention and MLP tensors scaled by {FAC:g} give attn[{PB}] = mlp[{PB}] = {abs(FAC - 1):g} and 0 at "
          "every other block, the embedding and the head (largest error < 1e-12); the identity pair (base against itself) gives 0 everywhere, "
          "so the planted expectation fails on it", err < 1e-12 and all(x == 0.0 for x in flat(idp)) and fails_on_identity,
          {"planted_attn": pl["attn"], "planted_mlp": pl["mlp"], "max_abs_err": err})


ENGINE_PROBE = "from infra.aws_pg.jobs import n13_gen as G; G.VllmEngine('model', 'tokenizer', 16)"


def engine_env_checks(ev):
    """A21: the --inner child's environment sets VLLM_USE_FLASHINFER_SAMPLER=0, and VllmEngine refuses to start without it (checked before
    vllm is imported). The engine start itself, which on the instance raises "Could not find nvcc" without the variable, cannot run here
    (no vLLM, no GPU): this check is on the environment only."""
    env = G.child_env({"PATH": "/usr/bin", "OTHER": "1"})
    main_src = __import__("inspect").getsource(G.main)
    runs = {}
    for name, val in (("set", "0"), ("unset", None), ("set_to_1", "1")):
        e = dict(os.environ)
        e.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        if val is not None:
            e["VLLM_USE_FLASHINFER_SAMPLER"] = val
        r = subprocess.run([sys.executable, "-c", ENGINE_PROBE], capture_output=True, text=True, env=e)
        runs[name] = {"returncode": r.returncode, "last_line": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}
    save(ev, "A21_engine_env.json", {"ENGINE_ENV": G.ENGINE_ENV, "ATTENTION_BACKEND": G.ATTENTION_BACKEND, "child_env_sample": env,
                                     "main_uses_child_env": "env=child_env(os.environ)" in main_src, "probe": runs})
    check("A21: the --inner child's environment (n13_gen.child_env, used by main) sets VLLM_USE_FLASHINFER_SAMPLER=0, keeps the venv first "
          "on PATH and every other variable; the engine pins attention_backend FLASH_ATTN", G.ENGINE_ENV == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
          and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["PATH"] == f"{pathlib.Path(G.N13_PY).parent}:/usr/bin" and env["OTHER"] == "1"
          and "env=child_env(os.environ)" in main_src and G.ATTENTION_BACKEND == "FLASH_ATTN", {k: env[k] for k in ("VLLM_USE_FLASHINFER_SAMPLER", "PATH")})
    check("A21 failing input: VllmEngine without VLLM_USE_FLASHINFER_SAMPLER, or with it at 1, stops on the A21 assertion before importing "
          "vllm; with it at 0 it passes the assertion and reaches the vllm import (no vLLM on the laptop: ModuleNotFoundError). The engine "
          "start without the variable, which raises 'Could not find nvcc' on the instance, cannot run here",
          all(runs[k]["returncode"] != 0 and "AssertionError: A21" in runs[k]["last_line"] for k in ("unset", "set_to_1"))
          and runs["set"]["returncode"] != 0 and "No module named 'vllm'" in runs["set"]["last_line"], runs)


def a22_cli(smoke_dir, cap, spent, out_json, manifest_dir, rate="min"):
    """manifest_dir: created here; A28: the command writes six manifests into it (MANIFEST_FILE)."""
    pathlib.Path(manifest_dir).mkdir(parents=True)
    r = subprocess.run([sys.executable, "-m", "infra.pg.n13_plan", "--smoke-dir", str(smoke_dir), "--cap", str(cap), "--spent", str(spent),
                        "--out", str(out_json), "--manifest-dir", str(manifest_dir), "--rate", rate], capture_output=True, text=True)
    return {"returncode": r.returncode, "stdout": r.stdout, "last_stderr_line": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}


def a22_parse(steps: list) -> dict:
    """The S3 manifest's gate steps, parsed with n13_gen's own parser."""
    return {s["name"]: G.parser().parse_args(shlex.split(s["args"]) + ["--out", "/work/out/" + s["name"]]) for s in steps
            if s["module"].endswith("n13_gen")}


def a22_smoke_checks(ev, smoke_root, tp):
    """A22 on the fake smoke (300 problems, 3 answers per step, at most 128 running, 0.05 s per step): the steps log is written per
    10 steps, recounts to throughput.json, and holds the answers' tokens; the projection command writes an S3 manifest from it.
    The fake clock counts from time.time() (about 1.7e9 s), so step times relative to the generation start carry float error of
    about 2e-6 s; the time bounds are 1e-4 s."""
    rec = {}
    for t in tp:
        so = smoke_root / f"smoke_{t['arm']}"
        rows = P.read_jsonl(so / "steps.jsonl")
        gens = [json.loads(x) for x in (so / t["arm"] / "math500/generations.jsonl").read_text().splitlines()]
        m = jl(so / t["arm"] / "math500/metrics.json")
        rec[t["arm"]] = {"rows": len(rows), "blocks": sorted({r["n"] for r in rows}), "recount_equal": P.step_rates(rows) == t["steps"],
                         "step_tokens": sum(r["tok"] for r in rows), "answer_tokens": sum(g["n_generated"] for g in gens), "steps": t["steps"],
                         "load_s_in_metrics": m["engine"].get("load_s"), "fields": {k: t.get(k) for k in ("r_full", "r_all", "tail_s", "load_s",
                         "mean_new_tokens", "median_new_tokens", "max_new_tokens", "n_cap_hits", "mean_output_tokens_at", "r_sat", "tail_sat_s",
                         "sat_run_mean", "sat_ctx_mean")}}
    ok = all(v["rows"] == 10 and v["blocks"] == [10] and v["recount_equal"] and v["step_tokens"] == v["answer_tokens"]
             and v["steps"]["steps"] == 100 and abs(v["steps"]["full_s"] - 58 * 0.05) < 1e-4 and abs(v["steps"]["tail_s"] - 42 * 0.05) < 1e-4
             and v["steps"]["run_max"] == 128 and v["load_s_in_metrics"] is not None and abs(v["steps"]["sat_s"] - 57 * 0.05) < 1e-4
             and all(v["fields"][k] is not None for k in v["fields"]) for v in rec.values())
    pj_path, mdir = smoke_root / "projection.json", smoke_root / "manifests"
    cli = a22_cli(smoke_root, 16384, 4.843, pj_path, mdir)
    pj = jl(pj_path) if cli["returncode"] == 0 else {}
    parsed = {k: a22_parse(jl(mdir / P.MANIFEST_FILE[k])) for k in P.S3_SPLIT if (mdir / P.MANIFEST_FILE[k]).exists()}
    F = pj.get("assumptions", {}).get("load_min", -1)
    man_ok = sorted(parsed) == ["s3gate1", "s3gate2"] and sorted(parsed["s3gate1"]) == ["base", "rl"] and sorted(parsed["s3gate2"]) == ["id", "zero"] \
        and all(x.cap == 16384 and x.allowance_min == pj["launches"][k]["allowance_min"] and x.proj_tok_s == pj["tok_s"] and x.problems == "0:500"
                and x.sets is None and abs(x.proj_load_min - (F + (0.0 if n in ("base", "rl") else 5.0))) < 1e-9
                for k in parsed for n, x in parsed[k].items())
    save(ev, "A22_smoke_fake.json", {"smokes": rec, "cli": cli, "projection": pj})
    check("A22 smoke (fake engine, 300 problems, at most 128 running): steps.jsonl written per 10 steps (10 lines for 100 steps), its "
          "recount equals throughput.json, its tokens equal the answers' n_generated; 58 steps at 128 running (2.9 s), tail 2.1 s; "
          "57 steps with requests waiting (2.85 s; A22 addition); throughput.json carries r_full, r_sat, r_all, tail_s, load_s, the "
          "saturated phase's mean running and context, mean, median and maximum length and the cap count; metrics.json carries "
          "engine.load_s", ok, {k: {"rows": v["rows"], "tokens": [v["step_tokens"], v["answer_tokens"]], "r_full": v["fields"]["r_full"],
                                            "tail_s": v["fields"]["tail_s"]} for k, v in rec.items()})
    check("A22 projection command (python -m infra.pg.n13_plan, rule min(r_full, r_sat)) on the fake smoke: exit 0, writes projection.json and "
          "the two S3 manifests (A28: base and rl in s3gate-01, zero and id in s3gate-02) whose gate steps parse with n13_gen's parser and "
          "carry --cap 16384, no --sets, their launch's allowance round(1.3 x its minutes) (A27), --proj-tok-s = the rate and --proj-load-min "
          "= F (base, rl) or F + 5 (zero, id)", cli["returncode"] == 0 and man_ok,
          {"exit": cli["returncode"], "launches": {k: v["allowance_min"] for k, v in pj.get("launches", {}).items()}, "tok_s": pj.get("tok_s"),
           "stderr": cli["last_stderr_line"]})


def a27_a28_checks(ev, smoke_root):
    """A27 and A28 on the fake smoke's projection and the six manifests written by a22_smoke_checks. A28: the S3 split (base and rl with
    setup with stage 2 and R1; zero and id with setup only), the S4 split by turns (each manifest in priority order, the first arm P1,
    every P1 and P2 arm among the first two arms of its manifest, the union equal to the roster's S4 arms). A27: every manifest's
    allowance is round(1.3 x its projection), and the s3gate2 projection equals 20 + 2 x (F + 5 + (500 L + 1,319 x 940) / r / 60) by hand."""
    pj, mdir = jl(smoke_root / "projection.json"), smoke_root / "manifests"
    man = {k: jl(mdir / P.MANIFEST_FILE[k]) for k in P.LAUNCHES}
    pri = {a: q for inst in P.S4.values() for a, q, _ in inst}
    arms = {k: [x["name"] for x in v if x["module"].endswith("n13_gen")] for k, v in man.items()}
    s4 = {k: [pri[a] for a in arms[k]] for k in P.S4_SPLIT}
    B = pj["assumptions"]
    hand = 20 + 2 * (B["load_min"] + 5.0 + (500 * B["mean_math"] + 1319 * 940) / pj["tok_s"] / 60)
    allow = {k: [G.parser().parse_args(shlex.split(x["args"]) + ["--out", "/x"]).allowance_min for x in v if x["module"].endswith("n13_gen")]
             for k, v in man.items()}
    rec = {"arms": arms, "priorities": s4, "allowance": allow, "launches": pj["launches"], "s3gate2_hand_min": hand,
           "setup_args": {k: v[0]["args"] for k, v in man.items()}}
    save(ev, "A27_A28_manifests.json", rec)
    ok_s3 = arms["s3gate1"] == ["base", "rl"] and arms["s3gate2"] == ["zero", "id"] and [x["name"] for x in man["s3gate1"][:2]] == ["setup", "r1_rank"] \
        and "stage2" in man["s3gate1"][0]["args"] and "stage2" not in man["s3gate2"][0]["args"] and man["s3gate2"][1]["name"] == "zero"
    ok_s4 = all(q == sorted(q) and q[0] == 1 for q in s4.values()) and all(arms[k].index(a) < 2 for k in P.S4_SPLIT for a in arms[k] if pri[a] <= 2) \
        and sorted(a for k in P.S4_SPLIT for a in arms[k]) == sorted(a for inst in P.S4.values() for a, _, _ in inst)
    ok_al = all(set(allow[k]) == {pj["launches"][k]["allowance_min"]} and pj["launches"][k]["allowance_min"] == int(round(1.3 * pj["launches"][k]["min"]))
                for k in P.LAUNCHES) and abs(pj["launches"]["s3gate2"]["min"] - hand) < 1e-6
    check("A28: S3 in two launches (s3gate-01: setup with stage 2, R1, base, rl; s3gate-02: setup, zero, id) and S4 in four, split by turns: "
          "each in priority order starting at P1, every P1 and P2 arm among the first two arms of its launch, the union the roster's 27 S4 arms",
          ok_s3 and ok_s4, {"arms": arms, "priorities": s4})
    check("A27: every launch's allowance is round(1.3 x its own projection), in each gen step's --allowance-min; s3gate2's projection "
          "equals the hand computation", ok_al, {k: [pj["launches"][k]["min"], pj["launches"][k]["allowance_min"]] for k in P.LAUNCHES} | {"hand": hand})
    # A29 relaunch: --r1-min and --relaunch on the same fake smoke (s3gate1 whole with R1 at 100 min; s3gate2 with id only)
    rd = smoke_root / "relaunch"
    rd.mkdir()
    r = subprocess.run([sys.executable, "-m", "infra.pg.n13_plan", "--smoke-dir", str(smoke_root), "--cap", "16384", "--spent", "8.0",
                        "--r1-min", "100", "--out", str(rd / "projection.json"), "--manifest-dir", str(rd), "--relaunch",
                        "s3gate1=g3.json", "s3gate2=g4.json:id"], capture_output=True, text=True)
    g3, g4 = (jl(rd / f) if (rd / f).exists() else [] for f in ("g3.json", "g4.json"))
    id_min = 20 + B["load_min"] + 5.0 + (500 * B["mean_math"] + 1319 * 940) / pj["tok_s"] / 60
    s3_min = pj["launches"]["s3gate1"]["min"] + 25.0
    al = lambda st: {G.parser().parse_args(shlex.split(x["args"]) + ["--out", "/x"]).allowance_min for x in st if x["module"].endswith("n13_gen")}
    ok_rl = r.returncode == 0 and [x["name"] for x in g3] == ["setup", "r1_rank", "base", "rl"] and [x["name"] for x in g4] == ["setup", "id"] \
        and "stage2" not in g4[0]["args"] and al(g3) == {int(round(1.3 * s3_min))} and al(g4) == {int(round(1.3 * id_min))}
    save(ev, "A29_relaunch_manifests.json", {"returncode": r.returncode, "stderr": r.stderr[-400:], "g3": g3, "g4": g4, "id_min_hand": id_min,
                                             "s3gate1_min_hand": s3_min})
    check("A29 relaunch manifests (n13_plan --r1-min 100 --relaunch s3gate1=... s3gate2=...:id): the first is setup with stage 2, R1, base, "
          "rl at allowance round(1.3 x (its projection with R1 at 100 min)); the second is setup without stage 2 and id only, at allowance "
          "round(1.3 x (20 + F + 5 + (500 L + 1,319 x 940) / r / 60))", ok_rl, {"g3": [x["name"] for x in g3], "g4": [x["name"] for x in g4],
                                                                                "allow": [sorted(al(g3)), sorted(al(g4))], "hand": [s3_min, id_min]})


def a29_checks(ev):
    """A29: the c = 1 builder check on the planted pairs of n13_arms.a29_planted (inexact entries and an after value of -0, the cases
    that stopped R1 at s3gate-01). The built tensor equals the after bits; the pre-A29 check fails on the same input. Registered wrong
    input for the A29 check itself: a delta() off by a factor 1 + 2^-6 on every entry must make build_tensor raise its assertion."""
    r = A.a29_planted()
    code = ("import numpy as np; from infra.pg import n13_arms as A; "
            "A.delta = lambda b, a: (A.f32(a) - A.f32(b)) * np.float32(1 + 2 ** -6); "
            "b = A.bf16(np.array([[x[0] for x in A.A29_PLANTED]], np.float32)); a = A.bf16(np.array([[x[1] for x in A.A29_PLANTED]], np.float32)); "
            "A.build_tensor('model.layers.0.mlp.down_proj.weight', b, a, {'kind': 'all', 'c': 1.0, 'non_mlp': 'after'}, 7)")
    w = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    msg = w.stderr.strip().splitlines()[-1] if w.returncode != 0 and w.stderr.strip() else ""
    r["wrong_input_delta_scaled"] = {"returncode": w.returncode, "last_stderr_line": msg}
    save(ev, "A29_builder_check.json", r)
    check("A29: on planted bf16 pairs (3 inexact, 1 after = -0, 3 exact) the c = 1 builder writes the after bits; the pre-A29 check "
          "(bits on every entry with Delta != 0) fails on them (3 entries off), as on the real weights at s3gate-01; a delta() scaled by "
          "1 + 2^-6 makes the A29 check raise", r["built_equals_after"] and not r["old_assertion_holds"] and r["formula_off_entries"] == 3
          and w.returncode != 0 and "AssertionError" in msg and "does not return theta_a" in msg, {k: r[k] for k in ("built_equals_after", "old_assertion_holds", "formula_off_entries")} | {"raised": msg})


def a30_checks(ev):
    """A30: the planted all-one-step arm at c = 1.3 (n13_arms.a30_planted). Deterministic rounding realizes 1 / 1.69 of the intended
    energy; with energy_match it still fails after three builds (the registered wrong input, run in a subprocess); stochastic rounding
    with energy_match ends within 5 %. The stochastic cast is unbiased in value on a planted array, and its streams are per tensor."""
    q = A.a30_planted()
    code = ("from infra.pg import n13_arms as A; once, spec, _ = A.a30_tensor('deterministic'); A.energy_match(once, spec)")
    w = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    last = w.stderr.strip().splitlines()[-1] if w.returncode != 0 and w.stderr.strip() else ""
    x = np.full(200_000, 1.0 + 0.3 * 2 ** -7, np.float32)              # 0.3 of the way from 1 to its bf16 neighbour above
    r = A.f32(A.bf16_stochastic(x, np.random.default_rng(0))).astype(np.float64)
    share_up = float((r > 1.0).mean())
    q["deterministic_with_energy_match"] = {"returncode": w.returncode, "last_stderr_line": last}
    q["stochastic_cast_share_up_at_0.3"] = share_up
    save(ev, "A30_energy.json", q)
    check("A30: an arm of one-step changes at c = 1.3 realizes 0.592 of its intended energy under deterministic rounding (outside 5 %), "
          "and energy_match cannot fix it in three builds (AssertionError); stochastic rounding with energy_match ends within 5 % "
          "(pass 1 at 1.12, the variance of the rounding; pass 2 after c x sqrt(intended / realized)); the stochastic cast rounds up 0.3 "
          "of the time at 0.3 of a step (within 0.005)",
          abs(q["deterministic_ratio_at_1.3"] - 1 / 1.69) < 1e-9 and w.returncode != 0 and "AssertionError" in last
          and "not within 0.05" in last and abs(q["stochastic_final_ratio"] - 1) <= 0.05 and abs(share_up - 0.3) < 0.005,
          {"det": q["deterministic_ratio_at_1.3"], "passes": [round(p["ratio"], 4) for p in q["stochastic_passes"]], "share_up": share_up,
           "det_loop": last[-120:]})


A31_BUMP = "model.layers.0.mlp.gate_proj.weight"


def a31_patch():
    """The A31 wrong input: in a written build (out_dir given), the first changed entry of A31_BUMP one bf16 step up."""
    orig_arm, orig_tensor = A._build_arm, A.build_tensor

    def tensor(name, b16, a16, spec, W, down_axis="columns"):
        T, M = orig_tensor(name, b16, a16, spec, W, down_axis)
        if name == A31_BUMP:
            T = T.copy()
            i = int(np.flatnonzero((T != np.asarray(b16)).ravel())[0])
            T.ravel()[i] = A.from_ordered(A.ordered(T.ravel()[i:i + 1]) + 1)[0]
        return T, M

    def arm(before, after, cfg, spec, out_dir=None, *x, **k):
        if out_dir is None:
            return orig_arm(before, after, cfg, spec, out_dir, *x, **k)
        A.build_tensor = tensor
        audit = orig_arm(before, after, cfg, spec, out_dir, *x, **k)
        A.build_tensor = orig_tensor
        return audit
    A._build_arm = arm


def a31_build(root, out, bump=False) -> dict:
    """A31: build_arm (energy_match, the written build, a31_compare) on the stand-in case in <root>; bump: the registered wrong input."""
    root = pathlib.Path(root)
    case = jl(root / "case.json")
    if bump:
        a31_patch()
    return A.build_arm(A.Weights(root / "base"), A.Weights(root / "rl"), case["cfg"], case["spec"], out_dir=root / out)


def a31_checks(ev, a):
    """A31 (N13_SPEC.md): the written rndE arm against its matched in-memory build. Stand-in: 32 layers (96 MLP tensors, as the real
    model), MLP magnitudes spread over 30 binades so that float64 sums of squares round, 3 shards with the names dealt round-robin (the
    in-memory build sums the MLP energy in sorted-name order, the written build shard by shard)."""
    root = pathlib.Path(a.out) / "a31"
    cfg = standin_cfg(a.hf, L=32)
    rng = np.random.default_rng(7)
    b = {n: (A.bf16(A.f32(t) * np.float32(2.0) ** rng.integers(-30, 1, t.shape).astype(np.float32)) if A.family(n) == "mlp" else t)
         for n, t in standin_base(cfg, 1).items()}
    tok = pathlib.Path(a.hf) / "rl31"
    write_snapshot(root / "base", cfg, b, tok)
    write_snapshot(root / "rl", cfg, standin_rl(b, cfg, 2, sorted(np.random.default_rng(5).choice(A.neuron_count(cfg), 8, replace=False).tolist())), tok)
    before, after = A.Weights(root / "base"), A.Weights(root / "rl")
    spec = {"name": "rndE_a31", "kind": "neurons", "neurons": sorted(np.random.default_rng(11).choice(A.neuron_count(cfg), 192, replace=False).tolist()),
            "c": 1.3, "non_mlp": "before", "rounding": "stochastic", "round_seed": 142}
    matched = A._build_arm(before, after, cfg, spec, None)
    spec["intended_mlp_energy"] = matched["families"]["mlp"]["realized_sq"]
    old = A._build_arm(before, after, cfg, spec, root / "old")
    r = {"pre_A31": {"matched_sq": matched["families"]["mlp"]["realized_sq"], "written_sq": old["families"]["mlp"]["realized_sq"],
                     "exact_equal": matched["families"]["mlp"]["realized_sq"] == old["families"]["mlp"]["realized_sq"],
                     "tensors_equal": matched["sha256"] == old["sha256"],
                     "reread_equal": {n: A.tensor_sha256(A.Weights(root / "old").get(n)) for n in after.names()} == old["sha256"]}}
    (root / "case.json").write_text(json.dumps({"cfg": cfg, "spec": spec}))
    r["A31"] = a31_build(root, "arm")["a31"]
    code = f"from infra.aws_pg.jobs import n13_dryrun as D; D.a31_build({str(root)!r}, 'bumped', bump=True)"
    w = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    last = w.stderr.strip().splitlines()[-1] if w.returncode != 0 and w.stderr.strip() else ""
    bumped = ast.literal_eval(last.split("AssertionError: ", 1)[1])[1] if last.startswith("AssertionError: ('A31") else None
    r["wrong_input_one_step"] = {"tensor": A31_BUMP, "returncode": w.returncode, "last_stderr_line": last[:600], "a31": bumped,
                                 "energy_part_alone_at_1e-6_fails": None if bumped is None else bool(bumped["rel_diff"] > 1e-6)}
    save(ev, "A31_written_vs_matched.json", r)
    shutil.rmtree(root)
    p, q = r["pre_A31"], r["A31"]
    check("A31: on a 32-layer stand-in with MLP magnitudes over 30 binades in 3 shards, the written rndE build equals the matched build "
          "bit for bit (sha256 and re-read) while the two float64 MLP energies differ (the pre-A31 exact check fails, as at s4armsB1-01 "
          "and B2-01); the A31 check passes (relative difference > 0 and <= 1e-12, tensors equal); one written entry one bf16 step up "
          "makes build_arm raise the A31 assertion with 1 tensor differing",
          not p["exact_equal"] and p["tensors_equal"] and p["reread_equal"] and q["pass"] and 0 < q["rel_diff"] <= A.A31_REL_TOL
          and q["n_tensors_differ"] == 0 and w.returncode != 0 and bumped is not None and not bumped["pass"] and bumped["n_tensors_differ"] == 1,
          {"pre_A31_rel": abs(p["matched_sq"] - p["written_sq"]) / p["matched_sq"], "a31_rel": q["rel_diff"],
           "bumped": None if bumped is None else {k: bumped[k] for k in ("rel_diff", "energy_pass", "n_tensors_differ", "tensors_differ")}})


def a22_plant(d, steps):
    """A planted smoke: steps = [(n, dt, run, tok, unf, ctx), ...] per model, written through n13_gen.StepLog; a throughput.json from
    the same rows (300 answers, load 60 s)."""
    for m in ("base", "rl"):
        so = d / f"smoke_{m}"
        so.mkdir(parents=True)
        log, t = G.StepLog(so / "steps.jsonl"), 0.0
        for n, dt, run, tok, unf, ctx in steps:
            for _ in range(n):
                t += dt
                log.add(t, dt, run, tok, unf, ctx)
        rows = log.close()
        sr = P.step_rates(rows)
        L = sr["output_tokens"] / 300
        (so / "throughput.json").write_text(json.dumps({"arm": m, "n_answers": 300, "output_tokens": sr["output_tokens"], "load_s": 60.0,
                                                        "mean_output_tokens_at": {"8192": L, "16384": L}, "steps": sr}))


def fcfs_engine(prompts, lengths, conc, dt):
    """A22 addition: an exact FCFS engine for the reconstruction check. Requests in order; at each step the waiting ones are admitted
    while fewer than conc run; every running request gets one token (the first at its admission step) and finishes after its length.
    Returns the per-step (run, tok, unf, ctx) and the answers in finish order."""
    n, q, running, done, steps = len(prompts), list(range(len(prompts))), {}, [], []
    while q or running:
        while q and len(running) < conc:
            running[q.pop(0)] = 0
        for i in running:
            running[i] += 1
        ctx = sum(prompts[i] + k for i, k in running.items())
        fin = [i for i, k in running.items() if k == lengths[i]]
        steps.append((len(running), len(running), n - len(done) - len(fin), ctx))
        for i in fin:
            del running[i]
            done.append({"problem_index": i, "prompt_n_tokens": prompts[i], "n_generated": lengths[i]})
    return steps, done


def a22_planted_checks(ev, root):
    """A22's registered inputs. (1) Concurrency that never reaches 128: the projection refuses. (2) Concurrency 300 with a known token
    count and wall time, no request waiting: at --rate r_full the expected r_full, tail_s, r_all and S3 minutes; at the rule
    min(r_full, r_sat) it refuses (r_sat undefined). (3) The stop rule: spent + S3 > 35 GPU-h writes no manifest. A22 addition:
    (4) waiting stays positive at 24 running with a known token count: the known r_sat, mean running and mean context, and the rule
    takes r_sat; (5) the attempt-8 reconstruction (n13_plan.sat_phase) on an exact FCFS engine's log equals the log's own means."""
    never, c300, sat24 = root / "never128", root / "conc300", root / "sat24"
    a22_plant(never, [(2000, 0.1, 100, 100, 300, 100 * 500)])
    a22_plant(c300, [(1005, 0.1, 300, 300, 300, 300 * 200), (200, 0.05, 10, 10, 10, 10 * 900)])   # 1,005 steps: block 100 holds 5 of each
    a22_plant(sat24, [(20, 0.04, 128, 128, 300, 128 * 200), (2000, 0.05, 24, 24, 200, 24 * 1500), (100, 0.02, 10, 10, 10, 10 * 3000)])
    r_never = a22_cli(never, 16384, 4.843, never / "projection.json", never / "manifest")
    r_300 = a22_cli(c300, 16384, 4.843, c300 / "projection.json", c300 / "manifest", rate="r_full")
    r_300min = a22_cli(c300, 16384, 4.843, c300 / "projection_min.json", c300 / "manifest_min")
    r_stop = a22_cli(c300, 16384, 34.5, c300 / "projection_stop.json", c300 / "manifest_stop", rate="r_full")
    r_24 = a22_cli(sat24, 16384, 4.843, sat24 / "projection.json", sat24 / "manifest")
    pj = jl(c300 / "projection.json")
    L, F = 301500 * 1.0 / 300 + 2000 / 300, (60.0 + 10.0) / 60
    arm = F + (500 * L + 1319 * 940) / 3000.0 / 60
    s3_want = 20 + 75 + 4 * arm + 2 * 5.0
    rl = pj["per_smoke"]["rl"]
    # (4): waiting in the 20 opening steps (300 unfinished, 128 running) and the 2,000 steps at 24 running (200 unfinished)
    sat_tok, sat_s = 20 * 128 + 2000 * 24, 20 * 0.04 + 2000 * 0.05
    run_mean = (128 * 0.8 + 24 * 100.0) / sat_s
    ctx_mean = (128 * 200 * 0.8 + 24 * 1500 * 100.0) / (128 * 0.8 + 24 * 100.0)
    L24 = (sat_tok + 100 * 10) / 300
    F24 = (60.0 + 100 * 0.02) / 60
    s3_24 = 20 + 75 + 4 * (F24 + (500 * L24 + 1319 * 940) / (sat_tok / sat_s) / 60) + 2 * 5.0
    p24 = jl(sat24 / "projection.json") if r_24["returncode"] == 0 else {}
    q = p24.get("per_smoke", {}).get("rl", {})
    rec = {"never128": r_never, "conc300": r_300, "conc300_min": r_300min, "stop_rule": r_stop, "sat24": r_24, "projection": pj,
           "projection_sat24": p24, "s3_want_min": s3_want, "r_full_want": 3000.0, "tail_s_want": 10.0, "r_all_want": 303500 / 110.5,
           "sat24_want": {"r_sat": sat_tok / sat_s, "r_full": 3200.0, "sat_run_mean": run_mean, "sat_ctx_mean": ctx_mean, "s3_min": s3_24}}
    check("A22 failing input: a planted steps log whose concurrency never reaches 128 (2,000 steps at 100 running) makes the projection "
          "command refuse: exit non-zero, the message names r_full, no manifest written",
          r_never["returncode"] != 0 and "r_full is undefined" in r_never["last_stderr_line"] and not any((never / "manifest").iterdir()),
          r_never["last_stderr_line"])
    check("A22 planted input: concurrency 300 for 1,005 steps of 0.1 s and 300 tokens, then 200 steps of 0.05 s at 10 running: at --rate "
          "r_full, r_full = 301,500 / 100.5 s = 3,000 tokens/s, tail_s = 10 s, r_all = 303,500 / 110.5 s; the S3 projection equals 20 + 75 "
          "+ 4 x (F + (500 L + 1,319 x 940) / r / 60) + 2 x 5 min with F = (60 + 10) / 60 min; the manifest is written. No request waits "
          "(unfinished = running), so the rule min(r_full, r_sat) refuses (r_sat undefined) and writes no manifest. A27 (no study ceiling): "
          "with 34.5 GPU-h spent the command exits 0 and writes the six manifests, as with 4.843",
          r_300["returncode"] == 0 and abs(rl["r_full"] - 3000.0) < 1e-6 and abs(rl["tail_s"] - 10.0) < 1e-6
          and abs(rl["r_all"] - 303500 / 110.5) < 1e-6 and abs(pj["s3_min"] - s3_want) < 1e-6 and pj["s3_allowance_min"] == int(round(1.3 * s3_want))
          and len(list((c300 / "manifest").iterdir())) == 6 and r_300min["returncode"] != 0 and "r_sat is undefined" in r_300min["last_stderr_line"]
          and not any((c300 / "manifest_min").iterdir()) and r_stop["returncode"] == 0
          and sorted(x.name for x in (c300 / "manifest_stop").iterdir()) == sorted(P.MANIFEST_FILE.values()),
          {"r_full": rl["r_full"], "tail_s": rl["tail_s"], "r_all": rl["r_all"], "s3_min": pj["s3_min"], "s3_want": s3_want,
           "min_rule": r_300min["last_stderr_line"], "stop": r_stop["last_stderr_line"]})
    w = rec["sat24_want"]
    check("A22 addition, planted input (the head's): 20 opening steps of 0.04 s at 128 running with 300 unfinished, then 2,000 steps of 0.05 s "
          "at 24 running with 200 unfinished (waiting 176), then 100 steps of 0.02 s at 10 running with none waiting: r_sat = 50,560 / "
          "100.8 s = 501.59 tokens/s, r_full = 2,560 / 0.8 s = 3,200; saturated-phase mean running 24.83 and mean context 1,446.8 "
          "tokens; the rule takes r_sat (the smaller), its tail 2.0 s, and gives the S3 minutes computed by hand; the S3 table is also "
          "given at r_full (smaller S3)",
          r_24["returncode"] == 0 and abs(q.get("r_sat", 0) - w["r_sat"]) < 1e-6 and abs(q.get("r_full", 0) - 3200.0) < 1e-6
          and abs(q.get("sat_run_mean", 0) - run_mean) < 1e-6 and abs(q.get("sat_ctx_mean", 0) - ctx_mean) < 1e-4
          and p24.get("rate_from") == "r_sat" and abs(p24.get("tok_s", 0) - w["r_sat"]) < 1e-6 and abs(p24.get("s3_min", 0) - s3_24) < 1e-6
          and p24["by_rate"]["r_full"]["s3_min"] < p24["by_rate"]["r_sat"]["s3_min"] == p24["s3_min"] and (sat24 / "manifest" / P.MANIFEST_FILE["s3gate1"]).exists(),
          {"r_sat": q.get("r_sat"), "want": w["r_sat"], "run_mean": q.get("sat_run_mean"), "ctx_mean": q.get("sat_ctx_mean"),
           "s3_min": p24.get("s3_min"), "s3_want": s3_24, "s3_at_r_full": (p24.get("by_rate") or {}).get("r_full", {}).get("s3_min"),
           "stderr": r_24["last_stderr_line"]})
    # (5) the reconstruction for a log without per-class run_dt and ctx_dt (attempt 8): an exact FCFS engine, 300 requests, at most 40
    # running, lengths 50 to 1,500, prompts 60 to 300; logged once per step (block 1, exact) and per 10 steps (the attempt-8 block)
    rng = np.random.default_rng(22)
    prompts, lengths = rng.integers(60, 301, 300).tolist(), rng.integers(50, 1501, 300).tolist()
    steps, done = fcfs_engine(prompts, lengths, 40, 0.03)
    rr = {}
    for blk in (1, 10):
        f = root / f"fcfs_block{blk}.jsonl"
        log, t = G.StepLog(f, block=blk), 0.0
        for run, tok, unf, ctx in steps:
            t += 0.03
            log.add(t, 0.03, run, tok, unf, ctx)
        rows = log.close()
        sr = P.step_rates(rows)
        old = [{**r, **{k: (r[k][:3] if r[k] is not None else None) for k in ("full", "wait")}} for r in rows]   # attempt-8 format
        rc = P.sat_phase(old, done, 300)
        rr[blk] = {"log": {k: sr[k] for k in ("sat_s", "sat_run_mean", "sat_ctx_mean", "r_sat")}, "reconstructed": rc}
    save(ev, "A22_planted.json", {**rec, "fcfs": {"conc": 40, "steps": len(steps), "by_block": rr}})
    e1, e10 = rr[1], rr[10]
    check("A22 addition, reconstruction (n13_plan.sat_phase, for attempt 8's log): on an exact FCFS engine (300 requests, at most 40 "
          "running, one token per step) logged per step, the means rebuilt from the finish order and the unfinished counts equal the "
          "log's own saturated seconds, mean running and mean context; logged per 10 steps, within 1 % (the finish step is known to "
          "within the block)",
          abs(e1["reconstructed"]["sat_s"] - e1["log"]["sat_s"]) < 1e-6 and abs(e1["reconstructed"]["sat_run_mean"] - e1["log"]["sat_run_mean"]) < 1e-6
          and abs(e1["reconstructed"]["sat_ctx_mean"] - e1["log"]["sat_ctx_mean"]) < 1e-3
          and e1["reconstructed"]["check"]["share_of_steps_running_within_block_range"] == 1.0
          and abs(e10["reconstructed"]["sat_run_mean"] / e10["log"]["sat_run_mean"] - 1) < 0.01
          and abs(e10["reconstructed"]["sat_ctx_mean"] / e10["log"]["sat_ctx_mean"] - 1) < 0.01,
          {b: {"log": {k: round(v, 3) for k, v in rr[b]["log"].items()}, "rec": {k: round(rr[b]["reconstructed"][k], 3) for k in
               ("sat_s", "sat_run_mean", "sat_ctx_mean")}} for b in rr})


def a24_killed_checks(ev, root):
    """A24: a smoke cut by the hard kill (no throughput.json; lengths.json holds its finished answers and n_submitted). Planted: the
    head's sat24 log for both models; the rl smoke then loses its throughput.json and holds 200 finished answers of 100 tokens out of
    300 submitted, at problems 0-219 except the 20 with index mod 11 = 5 (cut), load 90 s. With --killed rl (reading "prefix"): the
    admitted prefix is problems 0-219, L at C = (200 x 100 + 20 x C) / 220; with --killed-reading all: L = (200 x 100 + 100 x C) / 300.
    Either way the rate and tail are the base's (r_sat 501.59, 2.0 s), F = (90 + 2.0) / 60 min, and S3 is the hand computation.
    Wrong inputs: --killed on the unchanged smoke (it has throughput.json) and the cut smoke without --killed both exit non-zero."""
    d, ok_dir = root / "cut", root / "uncut"
    steps = [(20, 0.04, 128, 128, 300, 128 * 200), (2000, 0.05, 24, 24, 200, 24 * 1500), (100, 0.02, 10, 10, 10, 10 * 3000)]
    a22_plant(d, steps)
    a22_plant(ok_dir, steps)
    so = d / "smoke_rl"
    (so / "throughput.json").unlink()
    done = [i for i in range(220) if i % 11 != 5]
    (so / "lengths.json").write_text(json.dumps({"order": "finish", "n_submitted": 300, "answers": [
        {"problem_index": i, "n_generated": 100, "prompt_n_tokens": 50} for i in done]}))
    (so / "rl/math500").mkdir(parents=True)
    (so / "rl/math500/metrics.json").write_text(json.dumps({"engine": {"load_s": 90.0}}))

    def cli(sd, extra, tag):
        r = subprocess.run([sys.executable, "-m", "infra.pg.n13_plan", "--smoke-dir", str(sd), "--cap", "16384", "--spent", "4.843",
                            "--out", str(sd / f"projection_{tag}.json"), *extra], capture_output=True, text=True)
        return {"returncode": r.returncode, "last_stderr_line": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}
    runs = {"prefix": cli(d, ["--killed", "rl"], "prefix"), "all": cli(d, ["--killed", "rl", "--killed-reading", "all"], "all"),
            "wrong_uncut": cli(ok_dir, ["--killed", "rl"], "uncut"), "wrong_flagless": cli(d, [], "flagless")}
    r = (20 * 128 + 2000 * 24) / (20 * 0.04 + 2000 * 0.05)
    F = (90.0 + 2.0) / 60
    want = {}
    for tag, L in (("prefix", (200 * 100 + 20 * 16384) / 220), ("all", (200 * 100 + 100 * 16384) / 300)):
        want[tag] = {"L": L, "s3_min": 20 + 75 + 4 * (F + (500 * L + 1319 * 940) / r / 60) + 2 * 5.0}
    pj = {t: jl(d / f"projection_{t}.json") if runs[t]["returncode"] == 0 else {} for t in ("prefix", "all")}
    save(ev, "A24_killed.json", {"runs": runs, "want": want, "r": r, "F_min": F, "projection": pj, "done": len(done)})
    ok = all(runs[t]["returncode"] == 0 and abs(pj[t]["per_smoke"]["rl"]["mean_math"] - want[t]["L"]) < 1e-9
             and abs(pj[t]["tok_s"] - r) < 1e-9 and pj[t]["sets_the_rate"] == "rl" and abs(pj[t]["assumptions"]["load_min"] - F) < 1e-9
             and abs(pj[t]["s3_min"] - want[t]["s3_min"]) < 1e-6 and pj[t]["per_smoke"]["rl"]["killed"]["prefix_n"] == 220
             and pj[t]["per_smoke"]["rl"]["killed"]["prefix_unfinished"] == 20 for t in ("prefix", "all"))
    check("A24: a smoke cut by the hard kill (planted: 200 of 300 answers finished at 100 tokens over problems 0-219, 20 of them cut, "
          "no throughput.json, load 90 s): --killed rl gives L over the admitted prefix (200 x 100 + 20 x 16,384) / 220 = 1,580.36 and "
          "--killed-reading all gives (200 x 100 + 100 x 16,384) / 300 = 5,528.0; the rate and tail are the base's (r_sat 501.59, 2.0 s) "
          "and S3 equals the hand computation for both; --killed on an uncut smoke and the cut smoke without --killed both exit non-zero",
          ok and runs["wrong_uncut"]["returncode"] != 0 and runs["wrong_flagless"]["returncode"] != 0,
          {t: {"L": pj[t].get("per_smoke", {}).get("rl", {}).get("mean_math"), "s3_min": pj[t].get("s3_min"), "want": want[t]}
           for t in ("prefix", "all")})


def headers_checks(ev):
    """s2smoke-08: the n13 engine's torch.compile builds Triton's CUDA driver stub with gcc and needs Python.h; the install lived in the
    olmes stage only. Source check (apt-get cannot run here): venv_n13 installs and asserts the headers before it builds the venv, and
    setup runs venv_n13 before venv_olmes. Its failing input is the instance run without the headers (s2smoke-08, banked log lines)."""
    import inspect
    src_n13, src_run, src_hdr = inspect.getsource(SU.venv_n13), inspect.getsource(SU.run), inspect.getsource(SU.python_headers)
    rec = {"venv_n13_calls_python_headers_first": src_n13.index("python_headers()") < src_n13.index('"venv", "/opt/n13"'),
           "python_headers_installs_and_asserts": "python3.12-dev" in src_hdr and 'Path(inc, "Python.h").exists()' in src_hdr,
           "venv_olmes_no_longer_installs": "apt-get" not in inspect.getsource(SU.venv_olmes),
           "run_order_n13_before_olmes": src_run.index("venv_n13(out)") < src_run.index("venv_olmes(out)"),
           "failing_input": "studies/next/n13/r0/s2smoke-08/smoke_base/job_log_lines_900-1414.txt"}
    save(ev, "headers_setup.json", rec)
    check("s2smoke-08 job-file fix: venv_n13 installs python3.12-dev and asserts Python.h before building the n13 venv (the engine's "
          "torch.compile needs it); setup runs venv_n13 before venv_olmes, which only asserts it now", all(v for k, v in rec.items()
                                                                                                              if k != "failing_input"), rec)


def plant_cache(hub: pathlib.Path) -> dict:
    """A23: a Hugging Face cache as the setup step's snapshot_download lays it out (relative symlinks snapshots/<commit>/<file> ->
    ../../blobs/<hash>; refs/<branch> holds a commit): two revisions of allenai/Olmo-3-1025-7B, base main and stage2-step47684, that
    share the blobs of config.json and tokenizer.json and differ in the three shards, and one revision of the RL repo."""
    revs = {"base": SU.MODELS["base"], "stage2": SU.MODELS["stage2"], "rl": SU.MODELS["rl"]}
    files = {}
    for k, (repo_id, rev) in revs.items():
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        fs = {f"model-0000{i}-of-00003.safetensors": f"{k} shard {i}".encode() for i in (1, 2, 3)}
        fs |= {"config.json": f"config of {repo_id}".encode(), "tokenizer.json": f"tokenizer of {repo_id}".encode()}
        for name, data in fs.items():
            b = repo / "blobs" / hashlib.sha256(data).hexdigest()
            b.parent.mkdir(parents=True, exist_ok=True)
            b.write_bytes(data)
            ln = repo / "snapshots" / rev / name
            ln.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(b, ln.parent), ln)
        files[k] = {"snap": repo / "snapshots" / rev, "files": fs}
    refs = hub / "models--allenai--Olmo-3-1025-7B" / "refs"
    refs.mkdir()
    (refs / "main").write_text(SU.MODELS["base"][1])
    (refs / "stage2-step47684").write_text(SU.MODELS["stage2"][1])
    return files


def cache_state(hub: pathlib.Path, files: dict) -> dict:
    """Per planted revision: every file present and equal to its planted bytes; the refs entries; the blob count; the library's own
    reading of the cache (huggingface_hub.scan_cache_dir) as revisions per repo."""
    from huggingface_hub import scan_cache_dir
    repo = hub / "models--allenai--Olmo-3-1025-7B"
    intact = {k: v["snap"].is_dir() and all((v["snap"] / n).exists() and (v["snap"] / n).read_bytes() == d for n, d in v["files"].items())
              for k, v in files.items()}
    scan = {r.repo_id: sorted(x.commit_hash for x in r.revisions) for r in scan_cache_dir(hub).repos} if any(hub.iterdir()) else {}
    return {"intact": intact, "stage2_snapshot_exists": files["stage2"]["snap"].exists(),
            "refs": sorted(r.name for r in (repo / "refs").iterdir()) if (repo / "refs").exists() else None,
            "base_repo_blobs": len(list((repo / "blobs").iterdir())) if (repo / "blobs").exists() else 0, "scan_cache_dir": scan}


def a23_cache_checks(ev, root):
    """A23: R1's --delete-stage2 removed the whole models--allenai--Olmo-3-1025-7B folder (shutil.rmtree of the stage-2 snapshot's
    parents[1]), base main with it, which the S3 steps base, zero and id read afterwards. The offline gate ran R1 with
    delete_stage2=False on stand-in snapshots in separate folders, so it never ran the deletion. Planted: two revisions under one
    models-- folder (plant_cache). n13_rank.delete_snapshot must leave base main and the RL repo whole and remove stage 2's snapshot,
    its refs entry and its three shard blobs, keeping the two blobs it shares with main. The registered wrong input, the old code path
    on a fresh copy, must remove main."""
    import inspect
    hub = root / "hub_fixed" / "hub"
    hub.mkdir(parents=True)
    files = plant_cache(hub)
    before = cache_state(hub, files)
    rec = R.delete_snapshot(files["stage2"]["snap"])
    after = cache_state(hub, files)
    hub_old = root / "hub_old" / "hub"
    hub_old.mkdir(parents=True)
    files_old = plant_cache(hub_old)
    shutil.rmtree(pathlib.Path(files_old["stage2"]["snap"]).parents[1])          # the code before A23 (n13_rank.py line 169 at 7fb507bd3)
    old = cache_state(hub_old, files_old)
    src = inspect.getsource(R.run)
    out = {"before": before, "delete_snapshot": rec, "after": after, "old_code_path": old,
           "run_calls_delete_snapshot": 'delete_snapshot(sn["stage2"])' in src and "parents[1]" not in src}
    save(ev, "A23_cache.json", out)
    b1, b2 = SU.MODELS["base"][1], SU.MODELS["stage2"][1]
    ok_before = before["intact"] == {"base": True, "stage2": True, "rl": True} and before["base_repo_blobs"] == 8 \
        and before["scan_cache_dir"]["allenai/Olmo-3-1025-7B"] == sorted([b1, b2])
    ok_fixed = after["intact"] == {"base": True, "stage2": False, "rl": True} and not after["stage2_snapshot_exists"] \
        and after["refs"] == ["main"] and after["base_repo_blobs"] == 5 and rec["blobs_removed"] == 3 and rec["blobs_shared_kept"] == 2 \
        and after["scan_cache_dir"] == {"allenai/Olmo-3-1025-7B": [b1], "allenai/Olmo-3.1-7B-RL-Zero-Math": [SU.MODELS["rl"][1]]}
    check("A23: on a planted cache with base main and stage2-step47684 under one models-- folder (sharing 2 of 5 blobs), "
          "n13_rank.delete_snapshot removes stage 2's snapshot, its refs entry and its 3 shard blobs, and leaves base main (5 files, "
          "same bytes) and the RL repo whole; huggingface_hub.scan_cache_dir then lists main only; R1's run calls it",
          ok_before and ok_fixed and out["run_calls_delete_snapshot"], {"after": after, "removed": rec["blobs_removed"]})
    check("A23 registered wrong input: the old code path (rmtree of the stage-2 snapshot's parents[1]) on the same planted cache removes "
          "base main", old["intact"]["base"] is False and old["intact"]["rl"] is True, old["intact"])


def e2e(ev, root, a, snap, cfg, planted, total):
    hf = pathlib.Path(a.hf)
    out = root / "out"
    gsm = root / "gsm60.json"
    rng = np.random.default_rng(9)
    gx = [(int(rng.integers(1, 50)), int(rng.integers(1, 50))) for _ in range(60)]
    gsm.write_text(json.dumps([{"question": f"What is {x} + {y}?", "answer": f"{x}+{y}={x + y}\n#### {x + y:,}"} for x, y in gx]))
    sm = local_setup(a, out / "setup", snap, hf / "math500/test.jsonl", gsm)
    check("setup (local): data, G-GRADER, G-INIT (a), G-TEMPLATE pass; setup.json written", sm["status"] == "done", {k: sm["stages"][k].get("pass") if
          isinstance(sm["stages"][k], dict) else None for k in ("G-GRADER", "G-INIT_a")})
    # R0 on fabricated olmes directories: a fake olmes logs shim calls, as the real shim would, and copies one into place
    L = K.PATH_LIMIT
    src = {n: root / f"olmes_src_{n}" for n in ("path", "short", "full", "off")}
    fab_olmes(src["path"], n=L)
    fab_olmes(src["short"], n=L - 1)
    fab_olmes(src["full"])
    fab_olmes(src["off"], primary=0.30)
    ns = argparse.Namespace(out=str(out / "r0_known"), setup_dir=str(out / "setup"), allowance_min=280.0, poll_s=0.05,
                            scorer="string", model_type="vllm", tokenizer=str(snap["base"]))
    rc = K.run(ns, olmes_cmd=fake_olmes(src["path"], src["full"]), rescore_py=sys.executable)
    km = jl(out / "r0_known/metrics.json")
    save(ev, "R0_fabricated.json", km)
    check("R0 reader returns the aggregate primary_score (40.0) and G-KNOWN passes; derangement rescore < 10 fails it", rc == 0 and
          abs(km["G-KNOWN"]["value"] - 40.0) < 1e-9 and km["G-KNOWN"]["failing_input"]["failed"] and km["G-KNOWN"]["identity"]["pass"],
          {k: km["G-KNOWN"][k] for k in ("value", "pass")} | {"deranged": km["G-KNOWN"]["failing_input"]["value"]})
    logs = [json.loads(l) for f in (f"shim_calls_limit{L}.jsonl", "shim_calls.jsonl") for l in (out / "r0_known" / f).read_text().splitlines()]
    shim_sha = hashlib.sha256((K.SHIM_DIR / "sitecustomize.py").read_bytes()).hexdigest()
    check("R0 path check (A18, A19 second revision): --limit 100 run first, 100 scored rows per subject, reader and rescorer run; the "
          "olmes environment puts the shim first on PYTHONPATH; shim calls counted in metrics.json (7 calls, 2,800 prompts); the engine and "
          "the shim sha256 named",
          km["path_check"]["pass"] and set(km["path_check"]["rows_per_subject"].values()) == {L} and km["path_check"]["read"]["n_total"] == 5000
          and km["path_check"]["shim"]["calls"] == 7 and km["path_check"]["shim"]["prompts"] == 2800 and km["path_check"]["batched"]["pass"]
          and km["shim"]["calls"] == 1 and all(x["pythonpath"].split(":")[0] == str(K.SHIM_DIR) for x in logs)
          and km["engine"]["shim_sha256"] == shim_sha and km["engine"]["vllm"] == "0.11.0" and km["engine"]["lm_eval"] == "0.4.3"
          and km["engine"]["olmes_commit"] == SU.OLMES_COMMIT and not (out / f"r0_known/olmes_limit{L}").samefile(out / "r0_known/olmes"),
          {"rows": km["path_check"]["rows_per_subject"], "shim": [km["path_check"]["shim"]["calls"], km["shim"]["calls"]], "shim_sha256": shim_sha})
    kpj, ksh = km["projection"], km["path_check"]["shim"]
    want = batched_calls()
    T_want = sum(K.REPEATS * K.SUBJECT_SIZES[s] * tok / n for s, (n, tok) in zip(K.SUBJECTS, want))
    check("R0 path check (A19 second revision): batched calls pass (first and largest call 400 prompts, no early stop); the projection is "
          "built before the full task from the shim log, each call mapped to its subject in task order (per-subject tokens equal the "
          "planted s2setup-06 lengths), T = sum 4 x size_s x tokens_s / n_s, r = tokens / generation seconds, F = wall - generation "
          "seconds > 0; it goes at allowance 280", km["path_check"]["batched"]["pass"] and km["path_check"]["batched"]["first_n_prompts"] == 400
          and km["path_check"]["batched"]["max_n_prompts"] == 400 and not km["path_check"]["batched"]["stopped_early"]
          and [[kpj["per_subject"][s]["n_prompts"], kpj["per_subject"][s]["output_tokens"]] for s in K.SUBJECTS] == want
          and kpj["path_check_generations"] == 2800 and kpj["full_generations"] == 20000 and abs(kpj["T_full_output_tokens"] - T_want) < 1e-6
          and kpj["path_check_generate_s"] == ksh["generate_s"] and 0.069 <= ksh["generate_s"] < kpj["path_check_wall_s"] and kpj["F_fixed_s"] > 0
          and abs(kpj["projected_full_s"] - (kpj["F_fixed_s"] + kpj["T_full_output_tokens"] / kpj["r_tokens_per_s"])) < 1e-6 and not kpj["parked"]
          and km["engine"]["batch_size"] == "auto", {"batched": km["path_check"]["batched"], "T": kpj["T_full_output_tokens"], "T_want": T_want,
                                                      "projected_full_min": kpj["projected_full_min"], "parked": kpj["parked"]})
    check("R0 banks 5 right and 5 wrong samples with the prompt", len((out / "r0_known/samples.jsonl").read_text().splitlines()) == 10)
    # A19 second revision, registered failing input: batch size 1 must stop inside the path check, within one poll interval
    b1 = root / "r0_b1"
    t_b1 = time.time()
    rcb1 = K.run(argparse.Namespace(**{**vars(ns), "out": str(b1)}), olmes_cmd=fake_olmes(src["path"], src["full"], BATCH1_CALLS, call_s=0.01),
                 rescore_py=sys.executable)
    t_b1 = time.time() - t_b1
    n_b1 = len((b1 / f"shim_calls_limit{L}.jsonl").read_text().splitlines())
    time.sleep(0.5)
    n_b1_later = len((b1 / f"shim_calls_limit{L}.jsonl").read_text().splitlines())
    kb1 = jl(b1 / "metrics.json")
    es = kb1["path_check"].get("early_stop", {})
    rec = {"metrics": kb1, "step_seconds": t_b1, "log_lines_at_exit": n_b1, "log_lines_0.5_s_later": n_b1_later,
           "unstopped_seconds_at_least": len(BATCH1_CALLS) * 0.01}
    save(ev, "A19_batch1_early.json", rec)
    check("A19 early stop, registered failing input (batch size 1: 2,800 calls of 1 prompt, at least 28 s unstopped): the first logged "
          "call (n_prompts 1) is seen within one poll interval (0.05 s) of its return; olmes' process group is killed (the log stops "
          "growing, so the run_eval child died too); status failed_path_check with the reason; exit 4; no projection, no full task",
          rcb1 == 4 and kb1["status"] == "failed_path_check" and kb1["path_check"]["batched"]["stopped_early"]
          and kb1["path_check"]["batched"]["first_n_prompts"] == 1 and es.get("lag_after_first_call_s", 99) <= ns.poll_s + 0.2
          and "stopped early" in kb1["path_check"]["reason"] and n_b1 < 100 and n_b1_later == n_b1 and t_b1 < 10
          and not (b1 / "olmes").exists() and "projection" not in kb1 and "G-KNOWN" not in kb1,
          {"exit": rcb1, "reason": kb1["path_check"].get("reason"), "lag_s": es.get("lag_after_first_call_s"), "log_lines": [n_b1, n_b1_later],
           "step_s": round(t_b1, 2)})
    # the end-of-check assertion, kept: a batch-1 run that ends before any poll sees its log (poll 3 s, the fake ends in about 1 s)
    b1e = root / "r0_b1_end"
    rcb1e = K.run(argparse.Namespace(**{**vars(ns), "out": str(b1e), "poll_s": 3.0}),
                  olmes_cmd=fake_olmes(src["path"], src["full"], BATCH1_CALLS, sleep_s=0.3, call_s=0.0), rescore_py=sys.executable)
    kb1e = jl(b1e / "metrics.json")
    save(ev, "A19_batch1_end.json", kb1e)
    check("A19 end-of-check assertion (kept), failing input: a batch-1 path check that exits before the first poll sees its log "
          "(2,800 calls of 1 prompt, olmes exit 0) fails at the end: largest call 1 prompt, exit 4, no early stop, no full task",
          rcb1e == 4 and kb1e["status"] == "failed_path_check" and kb1e["path_check"]["olmes_returncode"] == 0
          and not kb1e["path_check"]["batched"]["stopped_early"] and kb1e["path_check"]["batched"]["max_n_prompts"] == 1
          and kb1e["path_check"]["shim"]["calls"] == 2800 and not (b1e / "olmes").exists() and "projection" not in kb1e,
          {"exit": rcb1e, "reason": kb1e["path_check"].get("reason"), "calls": kb1e["path_check"]["shim"]["calls"]})
    projection_table(ev)
    pk = root / "r0_park" / "r0_known"
    pk.mkdir(parents=True)
    (pk.parent / "pipeline_status.json").write_text(json.dumps({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    rcp = K.run(argparse.Namespace(**{**vars(ns), "out": str(pk), "allowance_min": 0.25}), olmes_cmd=fake_olmes(src["path"], src["full"], call_s=0.3),
                rescore_py=sys.executable)
    kp = jl(pk / "metrics.json")
    save(ev, "A19_park_e2e.json", kp)
    check("A19 projection gate end to end: a path check whose projection exceeds 85 % of the minutes left (allowance 0.25 min; 7 fake "
          "calls of 0.3 s at s2setup-06's per-subject lengths, so T / r = 6.76 x 2.1 s) writes the projection, parks (PARKED marker) and "
          "exits 0 before the full task", rcp == 0 and kp["status"] == "parked" and kp["projection"]["parked"] and kp["projection"]["elapsed_min"] < 1.0
          and (pk.parent / "PARKED").exists() and not (pk / "olmes").exists() and "G-KNOWN" not in kp,
          {k: kp["projection"][k] for k in ("path_check_wall_s", "path_check_generate_s", "F_fixed_s", "projected_full_min", "fits_min",
                                             "elapsed_min", "parked")})
    clock_checks(ev, root / "clock")
    rc99 = K.run(argparse.Namespace(**{**vars(ns), "out": str(root / "r0_short")}), olmes_cmd=fake_olmes(src["short"], src["full"]),
                 rescore_py=sys.executable)
    k99 = jl(root / "r0_short/metrics.json")
    check("R0 path check failing input: 99 scored rows per subject stops the step (exit 4) before the full task", rc99 == 4
          and k99["status"] == "failed_path_check" and not (root / "r0_short/olmes").exists() and "G-KNOWN" not in k99,
          {"exit": rc99, "status": k99["status"], "rows": k99["path_check"]["rows_per_subject"]["algebra"]})
    bad = root / "olmes_micro_only"
    fab_olmes(bad, only_micro=True)
    p = subprocess.run([sys.executable, "-c", f"from infra.aws_pg.jobs.n13_known import read_primary; read_primary({str(bad)!r})"], capture_output=True, text=True)
    check("R0 reader stops on a directory that holds only pass_at_1_micro", p.returncode != 0 and "primary_score" in p.stderr, p.stderr.strip().splitlines()[-1])
    rc_off = K.run(argparse.Namespace(**{**vars(ns), "out": str(root / "r0_off")}), olmes_cmd=fake_olmes(src["path"], src["off"]),
                   rescore_py=sys.executable)
    check("G-KNOWN fails (exit 3, the stop rule) on a fabricated base score of 30.0", rc_off == 3)
    shim_checks(ev)
    engine_env_checks(ev)
    headers_checks(ev)
    a23_cache_checks(ev, root / "a23")
    # R1
    C, sets = rank_checks(ev, out / "r1_rank", out / "setup", cfg, snap, planted, total)
    block_profile_checks(ev, out / "r1_rank", cfg, snap)
    # smoke (A22: problems 0:300, the s2smoke manifest's) and every arm
    probs = jl(out / "setup/problems.json")
    smoke_root = root / "s2smoke_out"                      # the smoke is its own attempt (s2smoke), not in the gate banks
    tp = []
    for arm in ("base", "rl"):
        clock = G.Clock()
        so = smoke_root / f"smoke_{arm}"
        G.run(gen_ns(so, arm, out / "setup", problems=f"0:{P.SMOKE_PROBLEMS_A22}", cap=16384, smoke=True),
              engine_factory=fake_factory(canned_for(arm, probs["math500"], probs["gsm8k"], "A"), clock, dt=0.05), clock=clock)
        tp.append(jl(so / "throughput.json"))
    save(ev, "smoke_throughput_fake.json", tp)
    check("smoke writes the throughput record (tokens/s, s/problem, histogram, memory, RSS, disk)", all(
        {"output_tokens_per_s", "seconds_per_problem", "histogram_counts", "host_peak_rss_kb", "free_disk_gb"} <= set(t) for t in tp))
    a22_smoke_checks(ev, smoke_root, tp)
    a27_a28_checks(ev, smoke_root)
    a22_planted_checks(ev, root / "a22")
    a24_killed_checks(ev, root / "a24")
    for r in P.roster():
        arm = r["arm"]
        bench = "math500,gsm8k" if r["gsm8k"] else "math500"
        clock = G.Clock()
        sets = None if arm in ("zero", "id") else str(out / "r1_rank/sets.json")      # A28: as in the s3gate2 manifest
        G.run(gen_ns(out / arm, arm, out / "setup", bench=bench, problems="0:500", cap=256, sets=sets),
              engine_factory=fake_factory(canned_for(arm, probs["math500"], probs["gsm8k"], r["seed_set"]), clock, per_step=25), clock=clock)
        if (out / "arms" / arm).exists():
            shutil.rmtree(out / "arms" / arm)
    from studies.next.n13.scripts import freeze_thresholds as FT
    T = FT.freeze(out, 256, "pair1", 500, 60)
    (root / "thresholds.json").write_text(json.dumps(T, indent=1))
    check("freeze_thresholds on the gate banks: room true; G-GATE, G-ID, S-BAND pass; each failing input fails",
          T["room"]["pass"] and T["G-GATE"]["pass"] and T["G-ID"]["pass"] and T["S-BAND"]["pass"] and T["G-GATE"]["failing_input"]["failed"]
          and T["G-ID"]["failing_input"]["failed"] and T["S-BAND"]["failing_input"]["failed"],
          {k: T[k] for k in ("sigma", "delta", "B", "L", "G", "L_prime")} | {"S-BAND_deranged": T["S-BAND"]["failing_input"]["value"]})
    vr = subprocess.run([sys.executable, "-m", "studies.next.n13.scripts.verdict_n13", "--banks", str(out), "--thresholds", str(root / "thresholds.json"),
                         "--setup", str(out / "setup"), "--known", str(out / "r0_known/metrics.json"), "--controls", str(out / "r1_rank/controls.json"),
                         "--out", str(root / "verdict_n13.json"), "--n-gsm", "60"], capture_output=True, text=True)
    (root / "verdict_stdout.txt").write_text(vr.stdout + vr.stderr)
    V = jl(root / "verdict_n13.json")
    check("verdict_n13 on the dry-run banks: COMPLETE, verdict table written", vr.returncode == 0 and V["verdict"] == "COMPLETE" and (root / "verdict_n13.md").exists(),
          {"exit": vr.returncode, "verdict": V["verdict"], "hypotheses": V.get("hypotheses")})
    return V, vr.returncode


# ------------------------------------------------------------------------------------------------ F. planted banks
def write_bank(root, arm, bench, correct: set, n, seed_set, status="done", screened_n=None):
    d = pathlib.Path(root) / arm / arm / bench
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    rng_n = range(screened_n) if screened_n else range(n)
    for i in rng_n:
        ok = i in correct
        cap = bool(screened_n and i < 16)
        rows.append({"problem_index": i, "score": int(ok), "seed": S.SEED_BASE[seed_set] + i, "text": "x", "cap_hit": cap,
                     "mode": "answer_line", "n_new": 100, "n_generated": 101, "finish_reason": "stop", "extracted": str(i) if ok else "-1",
                     "gold": str(i), "unique_id": f"p/{i}"})
    (d / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "metrics.json").write_text(json.dumps({"status": status}))


def planted(root, accs: dict, n=500, ng=60):
    root = pathlib.Path(root)
    perm = np.random.default_rng(3).permutation(n).tolist()
    pg = np.random.default_rng(4).permutation(ng).tolist()
    sets = {"base": set(perm[:150]), "zero": set(perm[10:160]), "rl": set(perm[:440]), "id": set(perm[10:450])}
    gsets = {"base": set(pg[:18]), "zero": set(pg[2:20]), "rl": set(pg[:48]), "id": set(pg[2:50])}
    for r in P.roster():
        arm = r["arm"]
        if arm in sets:
            c, cg = sets[arm], gsets[arm]
        else:
            ac = accs[arm]
            c = set(perm[: int(round(n * ac / 100))]) if ac is not None else set()
            cg = set(pg[: int(round(ng * (ac or 0) / 100))])
        if arm in accs and accs[arm] is None:
            write_bank(root, arm, "math500", set(), n, r["seed_set"], status="screened", screened_n=20)
        else:
            write_bank(root, arm, "math500", c, n, r["seed_set"])
        if r["gsm8k"]:
            write_bank(root, arm, "gsm8k", cg, ng, r["seed_set"])


BASE_ACCS = {"top_k0.1": 31, "top_k0.3": 40, "top_k1": 70, "top_k3": 78, "dare_k0.1": None, "dare_k0.3": 32, "dare_k1": 32, "dare_k3": 32,
             "rndE_k0.1": 31, "rndE_k0.3": 31, "rndE_k1": 31, "rndE_k3": 31, "rndT_k0.1": 32, "rndT_k0.3": 32, "rndT_k1": 32, "rndT_k3": 32,
             "mlp": 75, "attn": 31, "comp_k0.1": 35, "comp_k0.3": 35, "comp_k1": 35, "comp_k3": 35, "rnd0_k1": 31, "top_k1_B": 70,
             "rndT_k1_d2": 32, "dare_k1_d2": 32, "topF_k1": 69}
SCEN = {"local_supported": {}, "krec_none": {"top_k1": 40, "top_k3": 41, "mlp": 41, "top_k1_B": 40, "topF_k1": 40},
        "dare_within": {"dare_k1": 69.6}, "attention_supported": {"attn": 70, "mlp": 31}}
EXPECT = {"local_supported": ("H_local", "SUPPORTED"), "krec_none": ("H_local", "REFUTED (k_rec NONE)"),
          "dare_within": ("H_local", "REFUTED (a control within delta)"), "attention_supported": ("H_attention", "SUPPORTED")}


def planted_gates(root):
    d = pathlib.Path(root)
    (d / "setup").mkdir(parents=True, exist_ok=True)
    (d / "setup/grader.json").write_text(json.dumps({"pass": True, "active_rule": "boxed", "rules": {"boxed": {"failing_input": {"failed": True}, "self_equal": 500}}}))
    (d / "setup/template.json").write_text(json.dumps({"pass": True, "failing_input": {"failed": True}}))
    (d / "known.json").write_text(json.dumps({"G-KNOWN": {"pass": True, "value": 40.1, "failing_input": {"failed": True, "value": 0.4}}}))
    ok = {"pass": True, "failing_input": {"failed": True, "value": 0.0}}
    C = {"G-INIT": {**ok, "b": {"difference": 0.5}}, "G-PARTITION": ok, "PC-RANK": {**ok, "recovery": 1.0}, "G-HASH": {**ok, "id": {"pass": True}},
         "G-MASK": ok, "PC-ART": {"pass": True, "rule_P_plus": "SUPPORTED", "rule_P_minus": "REFUTED", "failing_input": {"failed": True},
                                  "P_plus": {"J_round": {"k1": 0.3}, "J_sp": {"k1": 0.2}}},
         "J_floor": {"k1": 0.95}, "J_round": {"k1": 0.9}, "J_sp": {"k1": 0.1}}
    (d / "controls.json").write_text(json.dumps(C))
    C2 = json.loads(json.dumps(C))
    C2["PC-RANK"]["failing_input"]["failed"] = False
    (d / "controls_fi_not_failed.json").write_text(json.dumps(C2))
    C3 = json.loads(json.dumps(C))
    C3["PC-ART"] |= {"pass": False, "rule_P_minus": "SUPPORTED", "failing_input": {"failed": False}}
    (d / "controls_pcart_minus_supported.json").write_text(json.dumps(C3))


def run_verdict(root, banks, th, controls, out, fi=None):
    cmd = [sys.executable, "-m", "studies.next.n13.scripts.verdict_n13", "--banks", str(banks), "--thresholds", str(th), "--setup", str(root / "setup"),
           "--known", str(root / "known.json"), "--controls", str(controls), "--out", str(out), "--n-gsm", "60"]
    if fi:
        cmd += ["--failing-input", fi]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert "Traceback" not in p.stderr, p.stderr
    return p.returncode, jl(out), p.stdout


def verdict_checks(ev, root):
    from studies.next.n13.scripts import freeze_thresholds as FT
    root = pathlib.Path(root)
    planted_gates(root)
    res = {}
    for name, ch in SCEN.items():
        bk = root / name
        planted(bk, {**BASE_ACCS, **ch})
        T = FT.freeze(bk, 8192, "pair1", 500, 60)
        (bk / "thresholds.json").write_text(json.dumps(T))
        rc, V, _ = run_verdict(root, bk, bk / "thresholds.json", root / "controls.json", bk / "verdict.json")
        h, want = EXPECT[name]
        res[name] = {"exit": rc, "verdict": V["verdict"], "hyp": V.get("hypotheses")}
        check(f"verdict on planted banks '{name}': {h} = {want}", rc == 0 and V["hypotheses"][h] == want, V.get("hypotheses"))
    hd = res["dare_within"]["hyp"]["H_diffuse"]
    check("verdict on planted banks 'dare_within': H_diffuse SUPPORTED", hd.startswith("SUPPORTED"), hd)
    ha = res["local_supported"]["hyp"]["H_attention"]
    check("verdict on planted banks 'local_supported': H_attention REFUTED", ha == "REFUTED", ha)
    bk = root / "local_supported"
    fi_res = {}
    for g in ("G-GRADER", "G-TEMPLATE", "G-INIT", "G-PARTITION", "PC-RANK", "G-KNOWN", "G-HASH", "G-MASK", "G-COMPLETE", "G-GATE", "G-ID", "S-BAND", "G-GATE-GSM"):
        rc, V, so = run_verdict(root, bk, bk / "thresholds.json", root / "controls.json", root / f"fi_{g}.json", fi=g)
        fi_res[g] = {"exit": rc, "verdict": V["verdict"], "stamps": V["stamps"]}
    save(ev, "verdict_failing_inputs.json", fi_res)
    check("verdict --failing-input <gate>: each of 13 gates gives its registered effect (CANCELLED, or the S-BAND / GSM8K stamp)",
          all(v["exit"] == 0 for v in fi_res.values()), {k: v["verdict"] for k, v in fi_res.items()})
    rc, V, _ = run_verdict(root, bk, bk / "thresholds.json", root / "controls_fi_not_failed.json", root / "fi_not_failed.json")
    check("verdict exits non-zero (5) when a gate's failing input did not fail", rc == 5, V["verdict"])
    ha_ok = res["local_supported"]["hyp"]["H_artifact"]
    rc, V, _ = run_verdict(root, bk, bk / "thresholds.json", root / "controls_pcart_minus_supported.json", root / "pcart_minus_supported.json")
    check("verdict (A2): H_artifact is decided by the rule with PC-ART passing, and CANCELLED when rule(P-) is not REFUTED",
          ha_ok == "REFUTED" and rc == 0 and V["hypotheses"]["H_artifact"].startswith("CANCELLED (PC-ART"), [ha_ok, V["hypotheses"]["H_artifact"]])
    # room false on banks where G = 2 delta
    rb = root / "room_false"
    n = 500
    perm = np.random.default_rng(3).permutation(n).tolist()
    base_c = set(perm[:150])
    zero_c = set(perm[10:160])
    sig = 100 * np.sqrt(20 / 2) / n
    g_pts = 2 * 2 * np.sqrt(2) * sig
    rl_n = 150 + int(round(g_pts * n / 100))
    for arm, c, ss in (("base", base_c, "A"), ("zero", zero_c, "B"), ("rl", set(perm[:rl_n]), "A"), ("id", set(perm[10:rl_n + 10]), "B")):
        write_bank(rb, arm, "math500", c, n, ss)
    T = FT.freeze(rb, 8192, "pair1", 500, with_gsm=False)
    check("freeze_thresholds: room false on banks where G = 2 delta", not T["room"]["pass"], T["room"])
    save(ev, "verdict_planted.json", res)


# ------------------------------------------------------------------------------------------------ G. manifests
def manifest_checks(ev, a):
    out = {}
    arms = []
    for kind in ("s2setup", "s2smoke", *P.LAUNCHES):
        steps = P.manifest(kind, 1, allowance_min={"s2setup": 279, "s2smoke": 110}.get(kind, 600))
        c = P.check_manifest(steps)
        mods = {"infra.aws_pg.jobs.n13_setup": SU.parser(), "infra.aws_pg.jobs.n13_known": K.parser(), "infra.aws_pg.jobs.n13_gen": G.parser(),
                "infra.aws_pg.jobs.n13_rank": R.parser()}
        for s in steps:
            mods[s["module"]].parse_args(shlex.split(s["args"]) + ["--out", "/work/out/" + s["name"]])
        arms += [s["name"] for s in steps if s["module"].endswith("n13_gen") and not s["name"].startswith("smoke")]
        out[kind] = {"check": c, "steps": steps}
    rc = P.check_roster()
    check("manifests parse with pipeline.py's checks and each module's own argparse parser", all(v["check"]["distinct"] for v in out.values()),
          {k: v["check"]["n"] for k, v in out.items()})
    check("each roster arm appears once across the six launches of A28, s3gate1, s3gate2, s4armsA1, A2, B1, B2 (31 = 30 + 1 conditional; 11 GSM8K)", sorted(arms) == sorted(r["arm"] for r in P.roster())
          and len(arms) == len(set(arms)) and rc["pass"], rc)
    committed = jl(REPO / "infra/aws_pg/steps/n13_s2smoke-08.json")
    c9 = jl(REPO / "infra/aws_pg/steps/n13_s2smoke-09.json")
    check("A22: the committed attempt-8 and attempt-9 manifests infra/aws_pg/steps/n13_s2smoke-08.json and -09.json equal "
          "n13_plan.manifest('s2smoke', nn, allowance_min=110): setup, then smoke_base and smoke_rl on MATH-500 problems 0:300 with "
          "--allowance-min 110", committed == P.manifest("s2smoke", 8, allowance_min=110) == c9 == P.manifest("s2smoke", 9, allowance_min=110)
          and [s["name"] for s in committed] == ["setup", "smoke_base", "smoke_rl"]
          and all("--problems 0:300 " in s["args"] and "--allowance-min 110 " in s["args"] for s in committed[1:]),
          [s["args"][:60] for s in committed[1:]])
    proj = {"r0": [P.project_launch("r0", t) / 60 for t in P.ASSUME["tok_s"]], "full_s2_first": [P.project_total(t)["total_gpu_h"] for t in P.ASSUME["tok_s"]],
            "full_r0_first_r1_in_s3": [P.project_total(t, first="r0")["total_gpu_h"] for t in P.ASSUME["tok_s"]],
            "parts_800": P.project_total(800.0)["gpu_h"], "parts_500": P.project_total(500.0)["gpu_h"]}
    out["projection"] = proj
    save(ev, "manifests_and_budget.json", out)
    check("budget projector reproduces the spec §9.7 totals (25.5 and 37.7 GPU-h, within 0.3)", abs(proj["full_s2_first"][0] - 25.5) < 0.3 and abs(proj["full_s2_first"][1] - 37.7) < 0.3,
          proj["full_s2_first"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--olmes", required=True)
    ap.add_argument("--wandb", required=True)
    ap.add_argument("--task-doc", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    assert not root.exists() or not any(root.iterdir()), f"{root} is not empty"
    ev = root / "evidence"
    ev.mkdir(parents=True)
    os.environ["PYTHONPATH"] = str(REPO)
    t0 = time.time()
    check("selfchecks: n13_arms, n13_score", A.selfcheck() and S.selfcheck())
    a29_checks(ev)
    a30_checks(ev)
    a31_checks(ev, a)
    real_gates(a, ev)
    cfg, snap, planted_n, total = weights_gates(a, ev, root / "e2e")
    V, vrc = e2e(ev, root / "e2e", a, snap, cfg, planted_n, total)
    build_checks(ev, root / "e2e", cfg, snap, jl(root / "e2e/out/r1_rank/sets.json"))
    power_checks(ev, a, root)
    scoring_checks(ev, root, a, root / "e2e/out/setup")
    from studies.next.n13.scripts import figure_n13, samples_coverage
    if V["verdict"] == "COMPLETE":
        figs = figure_n13.render(root / "e2e/verdict_n13.json", root / "figures", footer="offline dry run, fake engine, stand-in weights")
        check("figure_n13 renders PNG and PDF from the verdict file", all(pathlib.Path(f).stat().st_size > 1000 for f in figs), figs)
    cov = samples_coverage.coverage(root / "e2e/out")
    check("samples_coverage: every roster arm x benchmark present with a right and a wrong sample (or screened)",
          all(r[2].startswith("present") for r in cov), sum(r[2] == "present" for r in cov))
    rc = subprocess.run([sys.executable, "-m", "studies.next.n13.scripts.recount_n13", "--banks", str(root / "e2e/out"), "--verdict",
                         str(root / "e2e/verdict_n13.json")], capture_output=True, text=True)
    check("recount_n13 reproduces every arm's accuracy from the banks", "RECOUNT_OK" in rc.stdout, rc.stdout.strip().splitlines()[-1] if rc.stdout else rc.stderr[-300:])
    verdict_checks(ev, root / "planted")
    manifest_checks(ev, a)
    n_ok = sum(c["ok"] for c in CHECKS)
    save(ev, "checks.json", {"checks": CHECKS, "n": len(CHECKS), "ok": n_ok, "seconds": round(time.time() - t0, 1)})
    print(f"{n_ok}/{len(CHECKS)} checks ok in {time.time() - t0:.0f} s")
    if n_ok == len(CHECKS):
        print("N13_DRYRUN_OK")
    else:
        print("N13_DRYRUN_FAILED: " + "; ".join(c["check"] for c in CHECKS if not c["ok"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
