"""Weight-free dry run of the NEURONOFF job (SCHEMA v4) and its analysis: a tiny random model of one architecture
(built from its transformers config class; nothing downloaded), a byte-level tokenizer trained on a few lines, a
fake corpus, a fake plan and pairs file (64 ONE/UNK pairs, 22 null pairs, 1 Arm B pole with 2 controls and 3
prompts), `neuron_off.Job.run` writing every output file to <out>/job, a check of every SCHEMA v3 key, then freeze,
unblind, armb_counts and unblind --study (neuron_off_analysis) in a scratch git repo at <out>/study. Plan and
pairs files follow SCHEMA v4 (neuronoff/plan/v2, neuronoff/pairs/v2).

    python -m infra.aws_pg.jobs.neuron_off_dryrun --arch gemma3 --out <dir>

Independent references built here, not by the job: the known number (a plain cross-entropy loop over whole
windows) and every held-out row's a_banked (MLPTap's act(g) * u on whole windows, as the harvest computes it).
A random model has learned nothing, so zeroing a layer need not raise NLL: the fake plan's positive-control floor
is -1 nats, and the known-number tolerance is 1e-4 nats (one window shift moves a random model's mean NLL by more).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

ARCHS = ("qwen3", "gemma3", "olmo3")
BOS = {"qwen3": False, "gemma3": True, "olmo3": True}        # as Q4B, G4 and O7's tokenizers
VOCAB = 384
WINDOW = 128
DRY_KNOWN_TOL = 1e-4
LINES = ["import numpy as np", "import pandas as pd", "from torch import nn", "def f(x):\n    return x + 1",
         "SELECT * FROM t WHERE id = 1;", "<a title=\"home\" href=\"/\">", "The cat sat on the mat."]


def tiny_model(arch: str, seed: int = 0, dtype=torch.bfloat16, n_layers: int = 2):
    import transformers as T
    assert arch in ARCHS, arch
    torch.manual_seed(seed)
    common = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                  bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if arch == "qwen3":
        model = T.Qwen3ForCausalLM(T.Qwen3Config(**common))
    elif arch == "olmo3":
        model = T.Olmo3ForCausalLM(T.Olmo3Config(**common))
    else:
        vision = T.SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                      image_size=32, patch_size=8)
        cfg = T.Gemma3Config(text_config=T.Gemma3TextConfig(**common, sliding_window=16), vision_config=vision,
                             mm_tokens_per_image=4, boi_token_index=VOCAB - 3, eoi_token_index=VOCAB - 2,
                             image_token_index=VOCAB - 1)
        model = T.Gemma3ForConditionalGeneration(cfg)
    return model.to(dtype).eval()


def tiny_tokenizer(bos: bool = True):
    """Byte-level BPE trained on LINES; <pad>=0, <bos>=1, <eos>=2; with bos, a BOS is prepended (as Gemma's)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(LINES * 4, trainers.BpeTrainer(vocab_size=VOCAB - 3, special_tokens=["<pad>", "<bos>", "<eos>"],
                                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if bos:
        t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<bos>" if bos else None, eos_token="<eos>",
                                  pad_token="<pad>")
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1 if bos else None, 2)
    assert len(tok) <= VOCAB - 3, len(tok)
    return tok


def fake_corpus(tok, n_windows: int = 40) -> np.ndarray:
    ids = tok(" ".join(LINES * 200), add_special_tokens=False).input_ids
    assert len(ids) >= n_windows * WINDOW + 1, len(ids)
    return np.asarray(ids[:n_windows * WINDOW + 1], dtype=np.int64)


def reference_known(model, tok, tokens, n_targets: int) -> float:
    """The harvest's rule: per window, F.cross_entropy(logits[:-1], x[1:]), windows in order, first n targets."""
    import torch.nn.functional as F
    from infra.pg.neurons_harvest import forward
    prefix = [] if tok.bos_token_id is None else [tok.bos_token_id]
    s, n, w = 0.0, 0, 0
    while n < n_targets:
        x = torch.tensor([prefix + tokens[w * WINDOW:(w + 1) * WINDOW].tolist()])
        with torch.no_grad():
            nll = F.cross_entropy(forward(model, x, True).logits[0, :-1].float(), x[0, 1:], reduction="none")
        take = min(n_targets - n, nll.numel())
        s, n, w = s + float(nll[:take].double().sum()), n + take, w + 1
    return s / n


def reference_a(model, tok, tokens, windows) -> dict:
    """{layer: a [n_windows, W, neurons]} on whole windows (window positions only, BOS dropped), from MLPTap's
    act(g) * u, as the harvest computes it."""
    from infra.pg import neuron_off_core as C
    from infra.pg.neurons_harvest import MLPTap, forward
    layers = C.decoder_layers(model)
    tap, P = MLPTap(layers), (0 if tok.bos_token_id is None else 1)
    out = {l: [] for l in range(len(layers))}
    for w in windows:
        x = torch.tensor([([tok.bos_token_id] if P else []) + tokens[w * WINDOW:(w + 1) * WINDOW].tolist()])

        def fn(lyr, g, u, a_model):
            out[lyr].append((layers[lyr].mlp.act_fn(g.float()) * u.float())[0, P:])
        with torch.no_grad():
            tap.run(fn, lambda: forward(model, x, False))
    return {l: torch.stack(v) for l, v in out.items()}


def heldout(a_ref: dict, windows, layer, neuron, sign, k=12) -> list:
    """The pole's k strongest positions (window position 0 excluded) over the reference windows, ranks 30.. ."""
    a = a_ref[layer][:, 1:, neuron]
    order = torch.argsort(-(sign * a).flatten(), stable=True)[:k]
    rows = []
    for i, f in enumerate(order.tolist()):
        wi, pos = divmod(f, WINDOW - 1)
        assert sign * float(a[wi, pos]) > 0, "pole fires fewer than k times"
        rows.append({"rank": 30 + i, "tok": windows[wi] * WINDOW + pos + 1, "a_banked": float(a[wi, pos])})
    return rows


def pid(l, n, s):
    return f"L{l:02d}N{n}{'+' if s == 1 else '-'}"


def fake_plan(model, tok, tokens, arch: str, n_pairs: int, n_null: int) -> tuple[dict, dict]:
    """(plan, pairs) in the SCHEMA §1 formats. Per layer, neurons in order: pair k takes poles 3k (ONE), 3k+1 (control)
    and, for the first n_null pairs, 3k+2 (the null partner); Arm B uses neurons 120-127."""
    windows = list(range(10, 30))
    a_ref = reference_a(model, tok, tokens, windows)
    b_pole, b_ctrls = (1, 120, 1), [(1, 121, -1), (0, 122, 1)]

    def match_vars(l, n, s, rows):
        a = a_ref[l][:, 1:, n] * s
        return {"density": float((a >= 0.01 * a.max()).float().mean()), "meanabs": float(a[a > 0].mean()),
                "ho_med": float(np.median([abs(r["a_banked"]) for r in rows])) if rows else None,
                "rate": float((a > 0).float().mean())}
    poles = {}
    for l, n, s in [b_pole] + b_ctrls:
        poles[pid(l, n, s)] = {"layer": l, "neuron": n, "sign": s, "arm": "B",
                               "banked_max_abs": float(a_ref[l][:, 1:, n].mul(s).max()),
                               "match_vars": match_vars(l, n, s, []), "heldout": []}
    pairs, nulls = [], []
    for k in range(n_pairs):
        l, j = k % 2, k // 2
        trio = [(l, 3 * j + r, 1 if (a_ref[l][:, 1:, 3 * j + r] > 0).sum() >= (a_ref[l][:, 1:, 3 * j + r] < 0).sum()
                 else -1) for r in range(3)]                          # each pole takes its neuron's more frequent sign
        for q in trio[:3 if k < n_null else 2]:
            rows = heldout(a_ref, windows, *q)
            poles[pid(*q)] = {"layer": q[0], "neuron": q[1], "sign": q[2], "arm": "A",
                              "banked_max_abs": float(a_ref[q[0]][:, 1:, q[1]].mul(q[2]).max()),
                              "match_vars": match_vars(*q, rows), "heldout": rows}
        mv1, mv2 = poles[pid(*trio[0])]["match_vars"], poles[pid(*trio[1])]["match_vars"]
        pairs.append({"pair_id": f"T-A{k + 1:03d}", "layer": l, "one": pid(*trio[0]), "ctrl": pid(*trio[1]), "d": 0.0,
                      "log_density_ratio": float(np.log(mv1["density"] / mv2["density"])), "class_stable": k % 3 != 0,
                      "explainer_one": ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"][k % 2],
                      "explainer_ctrl": "anthropic/claude-opus-5-5", "vars_one": mv1, "vars_ctrl": mv2})
        if k < n_null:
            nulls.append({"pair_id": f"T-N{k + 1:03d}", "layer": l, "ctrl": pid(*trio[1]), "extra": pid(*trio[2]),
                          "d": 0.0, "source": "extra"})
    n_targets = 300
    plan = {"schema": "neuronoff/plan/v2", "model_key": f"tiny-{arch}", "repo": f"tiny-random-{arch}",
            "revision_expected": None, "branch": None, "tokname": "tiny", "window": WINDOW, "bos_id": tok.bos_token_id,
            "corpus": "dry run: fake_corpus",
            "corpus_tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "known": {"nll_corpus_first_120k": reference_known(model, tok, tokens, n_targets), "n_targets": n_targets,
                      "source": "dry run: reference_known"},
            "collateral": {"n_windows": 3},
            "positive_control": {"n_contexts": 3, "floor_nats": -1.0, "ceiling_nats": 15.0},
            "reproduction": {"median_rel_tol": 0.03, "p95_rel_tol": 0.15},
            "floor_slices": [128, 64, 32, 16, 8],
            "order": list(poles),
            "poles": poles,
            "arm_b": [{"pole": pid(*b_pole), "ctrl1": pid(*b_ctrls[0]), "ctrl2": pid(*b_ctrls[1]),
                       "prompts": [{"i": 0, "prompt": "import numpy", "target": " as", "target_position": "end",
                                    "continuation": " as np"},
                                   {"i": 1, "prompt": "SELECT * FROM", "target": " t", "target_position": "end",
                                    "continuation": " t WHERE"},
                                   {"i": 2, "prompt": "from torch", "target": " import", "target_position": "end",
                                    "continuation": " import nn"}]}],
            "sampling": {"n_samples": 3, "temperature": 0.7, "top_k": 0, "top_p": 1.0, "max_new_tokens": 16,
                         "seed_rule": "torch.manual_seed(base_seed + 1000 * i)", "base_seed": 20261025}}
    pairs = {"schema": "neuronoff/pairs/v2", "model_key": plan["model_key"], "seed": 20261025, "rule": "dry run",
             "caliper_log": 0.4054651081081644, "inputs": {}, "match_vars": ["density", "meanabs", "ho_med"],
             "estimand": {"statement": "dry run", "n_one_eligible": n_pairs + 5, "n_one_feasible": n_pairs,
                          "n_one_unmatched": 5, "n_one_kept": n_pairs}, "pairs": pairs, "null_pairs": nulls,
             "arm_b": [{"pole": pid(*b_pole), "label": "dry-run label", "ctrl1": pid(*b_ctrls[0]),
                        "ctrl2": pid(*b_ctrls[1]), "d1": 0.0, "d2": 0.0}]}
    return plan, pairs


def build(arch: str, out: pathlib.Path, edit_plan=None, n_pairs: int = 64, n_null: int = 22):
    """The Job on the tiny model, its plan written to and read back from out/inputs/plan.json, beside the pairs file
    and a fake parent-verdict file; `edit_plan(plan)` may change the plan first (a planted failure)."""
    from infra.aws_pg.jobs import neuron_off as J
    model, tok = tiny_model(arch), tiny_tokenizer(BOS[arch])
    tokens = fake_corpus(tok)
    plan, pairs = fake_plan(model, tok, tokens, arch, n_pairs, n_null)
    if edit_plan:
        edit_plan(plan)
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    ones = [x["one"] for x in pairs["pairs"]]
    verdict = [{"pole": q, "class": "ONE" if q in ones or q == pairs["arm_b"][0]["pole"] else "UNK",
                "tercile": plan["poles"][q]["layer"] * 2,
                "explainer_model": ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"][i % 2]}
               for i, q in enumerate(plan["poles"])]
    (inp / "verdict_poles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in verdict))
    pairs["inputs"]["verdict_poles"] = {"path": str((inp / "verdict_poles.jsonl").resolve()),
                                        "sha256": hashlib.sha256((inp / "verdict_poles.jsonl").read_bytes()).hexdigest()}
    (inp / "pairs.json").write_text(json.dumps(pairs))
    (inp / "plan.json").write_text(json.dumps(plan))
    raw = (inp / "plan.json").read_bytes()
    plan = json.loads(raw)
    meta = {"model_key": plan["model_key"], "repo": plan["repo"], "branch": None, "revision": None, "dry_run": True}
    return J.Job(out / "job", model=model, tok=tok, tokens=tokens, plan=plan,
                 plan_sha256=hashlib.sha256(raw).hexdigest(), meta=meta, known_tol=DRY_KNOWN_TOL)


SCHEMA_KEYS = {   # SCHEMA v4 §3: keys every output must carry
    "metrics": {"schema", "status", "started_utc", "updated_utc", "model_key", "repo", "revision", "branch",
                "device_name", "transformers", "torch", "pg_git_sha", "plan_sha256", "gates", "progress", "poles", "arm_b"},
    "progress": {"n_poles_planned", "n_poles_done", "arm_b_done", "tokens_forwarded", "forward_s", "tokens_per_second"},
    "metrics_pole": {"file", "own_dnll_mean", "own_dnll_median", "own_clean_nll_mean", "own_n_contexts",
                     "own_removed_norm_median", "w_down_norm", "coll_dnll", "coll_removed_norm_sum",
                     "coll_n_ablated_positions", "pc_dnll_mean", "identity_max_abs_dlogit", "repro_median_rel_err",
                     "repro_p95_rel_err", "repro_n_sign_mismatch", "gates"},
    "metrics_arm_b": {"file_teacher", "file_samples", "mean_logp_sum", "mean_logp_first", "n_samples_written", "done"},
    "controls": {"known_number", "collateral_clean", "determinism", "identity_failing_input", "floor"},
    "pole": {"pole", "layer", "neuron", "sign", "w_down_norm", "own", "positive_control", "identity", "reproduction",
             "collateral", "timing_s"},
    "context": {"rank", "tok", "window", "pos", "target_id", "target_str", "a_banked", "a_ours", "rel_err",
                "clean_nll", "ablated_nll", "dnll", "removed_norm"},
    "teacher": {"pole", "i", "condition", "prompt_n_tokens", "prefix_property", "cont_token_ids", "cont_tokens",
                "first_token_str", "target_matches_first_token", "logp_per_token", "logp_sum", "logp_first", "top5_end",
                "max_abs_dlogit_vs_clean"},
    "sample": {"pole", "i", "condition", "seed", "k", "completion", "completion_token_ids", "n_new_tokens", "stopped_eos"},
    "own_sample": {"pole", "rank", "tok", "text_before", "token", "target_str", "clean", "ablated", "dnll"},
}


def analyse(out: pathlib.Path, job_out: pathlib.Path, model_key: str, b_pole: str):
    """freeze -> commit -> unblind -> armb_counts -> unblind --study, in a scratch git repo under out/study."""
    import subprocess
    from infra.pg import neuron_off_analysis as A
    st = out / "study"
    (st / "results").mkdir(parents=True, exist_ok=True)

    def git(*args):
        subprocess.run(["git", "-C", str(st), "-c", "user.name=dryrun", "-c", "user.email=dryrun@localhost", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    (st / "JOURNAL.md").write_text("# dry run\n\n## 2026-10-25T00:00Z — PREDICTIONS\n\nnone (dry run)\n")
    git("add", "JOURNAL.md")
    git("commit", "-q", "-m", "predictions")
    thr = st / "results" / f"thresholds_{model_key}.json"
    inp = out / "inputs"
    A.main(["freeze", "--model", model_key, "--bank", str(job_out), "--pairs", str(inp / "pairs.json"), "--out", str(thr)])
    git("add", str(thr))
    git("commit", "-q", "-m", "thresholds")
    A.main(["unblind", "--model", model_key, "--bank", str(job_out), "--pairs", str(inp / "pairs.json"),
            "--plan", str(inp / "plan.json"), "--thresholds", str(thr), "--journal", str(st / "JOURNAL.md"),
            "--out", str(st / "results" / f"verdict_{model_key}.json")])
    (inp / "patterns.json").write_text(json.dumps({b_pole: r"\bas\b"}))
    A.main(["armb_counts", "--bank", str(job_out), "--plan", str(inp / "plan.json"), "--patterns",
            str(inp / "patterns.json"), "--out", str(st / "results" / f"armb_counts_{model_key}.json")])
    A.main(["unblind", "--study", "--results", str(st / "results")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--n-null", type=int, default=22)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    job = build(a.arch, out, n_pairs=a.n_pairs, n_null=a.n_null)
    plan, o = job.plan, job.out
    seen = []                                 # n_poles_done as read back from disk after every write
    save = job.save

    def save_and_read():
        save()
        seen.append(json.loads((o / "metrics.json").read_text())["progress"]["n_poles_done"])
    job.save = save_and_read
    m = job.run()

    from infra.aws_pg.jobs.neuron_off import finite, stem
    disk = json.loads((o / "metrics.json").read_text())
    assert disk["status"] == "complete" and disk == json.loads(json.dumps(finite(m)))
    a_poles = [p for p, v in plan["poles"].items() if v["arm"] == "A"]
    b = plan["arm_b"][0]
    n_b = len(b["prompts"])
    assert seen == [0, 0] + [0] * n_b + [3] + list(range(4, 4 + len(a_poles))) + [3 + len(a_poles)], seen
    assert sorted(x.name for x in (o / "poles").iterdir()) == sorted(f"{stem(p)}.json" for p in a_poles)
    K = SCHEMA_KEYS
    ctl = json.loads((o / "controls.json").read_text())
    pole0 = json.loads((o / "poles" / f"{stem(a_poles[0])}.json").read_text())
    teach = [json.loads(x) for x in open(o / f"arm_b/{stem(b['pole'])}_teacher.jsonl")]
    samp = [json.loads(x) for x in open(o / f"arm_b/{stem(b['pole'])}_samples.jsonl")]
    own_s = [json.loads(x) for x in open(o / "samples/own_ablation.jsonl")]
    checks = [(K["metrics"], disk), (K["progress"], disk["progress"]), (K["metrics_pole"], disk["poles"][a_poles[0]]),
              (K["metrics_arm_b"], disk["arm_b"][b["pole"]]), (K["controls"], ctl), (K["pole"], pole0),
              (K["context"], pole0["own"]["contexts"][0]), (K["own_sample"], own_s[0])] + \
        [(K["teacher"], t) for t in teach] + [(K["sample"], x) for x in samp]
    for keys, obj in checks:
        assert keys <= set(obj), sorted(keys - set(obj))
    assert all(("pole_a_inputs" in t) == (t["condition"] == "clean") for t in teach)
    assert all(t["max_abs_dlogit_vs_clean"] == 0.0 for t in teach if t["condition"] == "identity")
    assert len(teach) == n_b * 6 and len(samp) == (n_b * 3 + 1) * plan["sampling"]["n_samples"]
    clean0 = [x["completion_token_ids"] for x in samp if x["i"] == 0 and x["condition"] == "clean"]
    ident0 = [x["completion_token_ids"] for x in samp if x["i"] == 0 and x["condition"] == "identity"]
    poles = [json.loads((o / "poles" / f"{stem(p)}.json").read_text()) for p in a_poles]
    summary = {"arch": a.arch, "n_arm_a_poles": len(a_poles), "writes": len(seen), "gates": m["gates"],
               "known": {k: ctl["known_number"][k] for k in ("abs_diff", "pass")},
               "known_failing_input": {k: ctl["known_number"]["failing_input"][k] for k in ("abs_diff", "pass")},
               "floor": {l: (f["max_abs_dlogit"], f["failing_input"]["max_abs_dlogit"]) for l, f in ctl["floor"].items()},
               "G_ident_max": max(p["identity"]["max_abs_dlogit_identity"] for p in poles),
               "G_fit_pass": sum(p["reproduction"]["pass"] for p in poles),
               "G_fit_failing_input_failed": sum(not p["reproduction"]["failing_input"]["pass"] for p in poles),
               "G_fit_failing_input_min_median_rel_err": min(p["reproduction"]["failing_input"]["median_rel_err"]
                                                             for p in poles),
               "G_fit_failing_input_min_sign_mismatch": min(p["reproduction"]["failing_input"]["n_sign_mismatch"]
                                                            for p in poles),
               "G_det": (ctl["determinism"]["max_abs_dlogit_clean_repeat"],
                         ctl["determinism"]["failing_input"]["max_abs_dlogit"]),
               "G_ident_failing_input": ctl["identity_failing_input"]["max_abs_dlogit"],
               "G_pc_failing_input_failed": sum(not p["positive_control"]["failing_input"]["pass"] for p in poles),
               "clean_equals_identity_samples_prompt0": clean0 == ident0}
    print("NEURON_OFF_DRYRUN_OK", json.dumps(finite(summary)), flush=True)
    analyse(out, o, plan["model_key"], b["pole"])


if __name__ == "__main__":
    main()
