"""Weight-free dry run of the NEURONRW job (studies/neuronrw/SCHEMA.md v5): a tiny random model of one architecture
(neuron_off_dryrun.tiny_model; nothing downloaded), its byte-level tokenizer, a fake corpus, fake arm0 / arm1 / arm2
plans in the SCHEMA §1 formats, and `neuron_rw.Job.run` for each stage writing every output under <out>/<stage>.
After every metrics.json write the file is read back from disk, so the check sees what a poller would see.

    python -m infra.aws_pg.jobs.neuron_rw_dryrun --arch gemma3 --out <dir>

Independent references built here, not by the job: the known number's a_banked (MLPTap's act(g) * u on whole
windows, as the harvest computes it) and its clean and ablated NLLs (a plain cross-entropy over whole windows with
a separate zeroing pre-hook). A random model has learned nothing, so the dry plans differ from the real ones in three
numbers: the positive-control floor is 1e-4 nats (above the alpha = 0 failing input's exact 0), the pc poles are
planted (`plant`), and the known number's failing shift is 1e-5 nats (zeroing tiny Gemma's layer-0 MLP moves its
near-uniform NLL by about 3e-5; the real plans use 0.5, and NEURONOFF measured 1.7-1.9 nats). The tolerances below
absorb bf16 differences between whole-window and window-prefix forwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from infra.aws_pg.jobs.neuron_off_dryrun import ARCHS, BOS, VOCAB, WINDOW, fake_corpus, reference_a, tiny_model, \
    tiny_tokenizer

NEUTRAL = ["Here is some text:\n", "import numpy", "SELECT * FROM"]      # [0] is SPEC §4.7 prompt 0
DRY = {"a_rel_tol": 0.05, "nll_abs_tol": 0.01, "failing_min_shift_nats": 1e-5, "pc_floor": 1e-4}


def pid(l, n, s):
    return f"L{l:02d}N{n}{'+' if s == 1 else '-'}"


def pole(l, n, s):
    return {"pole": pid(l, n, s), "layer": l, "neuron": n, "sign": s}


def ref_nll(model, tok, tokens, w: int, layer=None, neuron=None, sign=None) -> torch.Tensor:
    """Whole-window next-token NLL [W] (positions 0..W-1 of the window, BOS prepended when defined), optionally with
    a_i := 0 where sign * a_i > 0 from window position 1 on (an independent zeroing pre-hook)."""
    from infra.pg import neuron_off_core as C
    from infra.pg.neurons_harvest import forward
    pre = C.bos_prefix(tok)
    x = torch.tensor([pre + tokens[w * WINDOW:(w + 1) * WINDOW + 1].tolist()])
    h = None
    if layer is not None:
        def zero(m, args):
            a = args[0].float().clone()
            v = a[0, len(pre) + 1:, neuron]
            a[0, len(pre) + 1:, neuron] = torch.where(v * sign > 0, torch.zeros_like(v), v)
            return (a.to(args[0].dtype),)
        h = C.decoder_layers(model)[layer].mlp.down_proj.register_forward_pre_hook(zero)
    with torch.no_grad():
        logits = forward(model, x[:, :-1], True).logits[0].float()
    if h is not None:
        h.remove()
    return F.cross_entropy(logits[len(pre):], x[0, len(pre) + 1:], reduction="none")


def contexts(model, tok, tokens, a_ref, windows, l, n, s, k=6, rank0=30) -> list:
    """The pole's k strongest positions (window position 0 excluded) with banked a, clean NLL and ablated ΔNLL."""
    a = a_ref[l][:, 1:, n] * s
    order = torch.argsort(-a.flatten(), stable=True)[:k].tolist()
    clean = {w: ref_nll(model, tok, tokens, w) for w in windows}
    abl = {w: ref_nll(model, tok, tokens, w, l, n, s) for w in windows}
    rows = []
    for i, f in enumerate(order):
        wi, p = divmod(f, WINDOW - 1)
        w, pos = windows[wi], p + 1
        assert float(a[wi, p]) > 0, "pole fires fewer than k times"
        t = w * WINDOW + pos
        rows.append({"rank": rank0 + i, "tok": t, "target_id": int(tokens[t + 1]), "a": float(a_ref[l][wi, pos, n]),
                     "clean_nll": float(clean[w][pos]), "dnll": float(abl[w][pos] - clean[w][pos])})
    return rows


KEY = {"qwen3": "Q4B", "gemma3": "G4", "olmo3": "O7"}      # the real key of the same architecture (model_index rngs)


def top_positions(a_ref, windows, q, k=1000) -> list:
    """Global token indices of the pole's k strongest positions (window position 0 excluded), strongest first."""
    a = a_ref[q["layer"]][:, 1:, q["neuron"]] * q["sign"]
    out = []
    for f in torch.argsort(-a.flatten(), stable=True)[:k].tolist():
        wi, p = divmod(f, WINDOW - 1)
        out.append(windows[wi] * WINDOW + p + 1)
    return out


def model_fields(model, arch, tok, tokens) -> dict:
    from infra.pg.neurons_harvest import text_tower
    return {"model_key": KEY[arch], "repo": f"tiny-random-{arch}", "revision_expected": None, "branch": None,
            "tokname": "tiny", "window": WINDOW, "bos_id": tok.bos_token_id, "corpus": "dry run: fake_corpus",
            "corpus_tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "d_model": text_tower(model).norm.weight.shape[0]}


def fake_plans(model, tok, tokens, arch: str, tok_sha256: str = "dry") -> dict:
    """arm0, arm1, arm2 plans in the shapes make_plan.py writes (natural-span pc texts, place, floor.alpha_fail,
    firing check in arm0, the per-call seed rule; judge0_pc with top-1,000 positions, n_distinct and the A20 rule,
    `tok_sha256` the sha256 of the tokenizer file the job decodes with)."""
    from infra.aws_pg.jobs.neuron_rw import attested_ids, firing_set, ids_sha256, special_ids
    from infra.pg.neurons_harvest import text_tower
    windows = list(range(4, 10))
    a_ref = reference_a(model, tok, tokens, windows)

    def major(l, n):
        return 1 if (a_ref[l][:, 1:, n] > 0).sum() >= (a_ref[l][:, 1:, n] < 0).sum() else -1
    kn_pole = pole(1, 9, major(1, 9))
    kc = contexts(model, tok, tokens, a_ref, windows, kn_pole["layer"], kn_pole["neuron"], kn_pole["sign"])
    known = {**kn_pole, "source": "dry run: reference_a and ref_nll",
             "contexts": [{"rank": c["rank"], "tok": c["tok"], "target_id": c["target_id"], "a_banked": c["a"],
                           "clean_nll_banked": c["clean_nll"], "dnll_banked": c["dnll"]} for c in kc],
             "own_clean_nll_mean_banked": float(np.mean([c["clean_nll"] for c in kc])),
             "own_dnll_mean_banked": float(np.mean([c["dnll"] for c in kc])),
             "a_rel_tol": DRY["a_rel_tol"], "nll_abs_tol": DRY["nll_abs_tol"], "nll_corpus_first_120k": None,
             "failing_input": "layer 0 whole-MLP output zeroed on the mask domain",
             "failing_min_shift_nats": DRY["failing_min_shift_nats"]}
    repair = []
    for l, n in ((0, 20), (1, 33), (1, 47)):
        cs = contexts(model, tok, tokens, a_ref, windows, l, n, major(l, n), k=4)
        repair.append({**pole(l, n, major(l, n)), "contexts": [
            {"rank": c["rank"], "tok": c["tok"], "target_id": c["target_id"], "a_off": c["a"],
             "clean_nll_off": c["clean_nll"], "dnll_off": c["dnll"]} for c in cs]})
    att = attested_ids(tokens, special_ids(tok))
    poles = sorted([pole(l, n, s) for l in (0, 1) for n, s in ((3, 1), (17, -1), (58, 1), (101, -1))], key=lambda p: p["pole"])
    pc = {"poles": [pole(*x) for x in PC_POLES], "alpha": 4, "texts": "natural_spans calibration spans 0..3",
          "floor_nats": DRY["pc_floor"], "ceiling_nats": 15.0}
    spans = {"calibration_windows": [0, 1], "test_windows": [2, 3], "spans_per_window": 4, "prompt_tokens": 8,
             "cont_tokens": 24}
    common = {"pc": pc, "place": {"alphas": [1, 4], "cos_min": 0.95, "ratio_band": [0.9, 1.1]},
              "gen_gate": {"alphas": [1, 4], "steps": [1, 2, 3], "cos_min": 0.95, "ratio_band": [0.9, 1.1],
                           "poles": "first 3 pc poles", "prompt": "neutral prompt 0", "rows": 4},
              "coherence": {"split_seed": "default_rng([20261025, model_index, 12]).permutation; first half calibrates, "
                                          "second half tests", "lo_pct": 2.5, "hi_pct": 97.5, "min_new_tokens": 8,
                            "passing_min_share": 0.9},
              "floor": {"random_dir_seed": 20261025, "alpha_fail": 4}, "natural_spans": spans}
    sampling = {"temperature": 0.7, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 12,
                "seed_rule": "torch.manual_seed(base_seed + 1000 * prompt_i + c) before the c-th generate call of "
                             "prompt_i (c counts from 0 in plan row order)", "base_seed": 20261025, "max_rows": 16}
    mf = model_fields(model, arch, tok, tokens)
    t, c = pole(1, 9, 1), pole(1, 61, -1)
    pset = [{"i": 0, "prompt": "import numpy", "continuation": " as np"},
            {"i": 1, "prompt": "from torch", "continuation": " import nn"},
            {"i": 2, "prompt": "SELECT * FROM", "continuation": " t WHERE id = 1;"}]
    known_sub = {**known, "contexts": known["contexts"][:N_KNOWN_STAGE],       # make_plan.known_subset: Arm 1 and Arm 2
                 "own_clean_nll_mean_banked": float(np.mean([c["clean_nll_banked"] for c in known["contexts"][:N_KNOWN_STAGE]])),
                 "own_dnll_mean_banked": float(np.mean([c["dnll_banked"] for c in known["contexts"][:N_KNOWN_STAGE]]))}
    arm0 = {"schema": "neuronrw/plan-arm0/v1", **mf, "known": known, "n_layers": len(text_tower(model).layers),
            "attested_vocab": {"rule": "dry run", "n_ids": int(len(att)), "sha256": ids_sha256(att)}, "topn": 20,
            "poles": poles, "rms": {"windows": [0, 1, 2, 3], "exclude_first": 1}, **common, "repair": repair,
            "sweep": {"poles": [pole(0, 12, 1), pole(1, 44, -1)], "doses": [-4, -1, 1, 4], "n_samples": 2,
                      "n_samples_alpha0": 4},
            "firing_check": [{**t, "banked_max_abs": float(a_ref[1][:, 1:, 9].mul(t["sign"]).max()), "prompt_set": "L01N9p"}],
            "firing_rule": {"threshold_frac": 0.3, "min_prompts": 2}, "prompt_sets": {"L01N9p": pset},
            "repair_rule": {"total_t_only": "one extra forward per context", "direct_primary": "both scales frozen",
                            "direct_secondary": ["post-MLP norm recomputed, final frozen", "final norm recomputed"]},
            "throughput_probe": {"prompt_set": "L01N9p", "prompt_i": 0, "rows": 16, "calls": 2, "prompt": pset[0]["prompt"]},
            "judge0_pc": [{**q, "top_tok": top_positions(a_ref, windows, q),
                           "n_distinct": firing_set([tok.backend_tokenizer.decode([int(tokens[t])], skip_special_tokens=False)
                                                     for t in top_positions(a_ref, windows, q)])[1]} for q in poles[:2]],
            "judge0_pc_rule": {"positions": "dry top 1,000", "min_distinct": 10, "max_tokens": 20,
                               "tokenizer": {"path": "inputs/tokenizer.json", "sha256": tok_sha256}},
            "judge0_pc_replacements": [],
            "neutral_prompts": NEUTRAL, "sampling": sampling}
    arm1 = {"schema": "neuronrw/plan-arm1/v1", **mf, **common, "known": known_sub, "dose_scale": {"alpha_c": 4, "source": "brief default"},
            "doses": [-4, 0, 4], "poles": [{**t, "prompt_set": "L01N9p"}, {**c, "prompt_set": "L01N9p"}],
            "prompt_sets": {"L01N9p": [{"i": x["i"], "prompt": x["prompt"]} for x in pset]},
            "rms_source": "dry arm0 out/arm0/rms.json", "sampling": {**sampling, "n_samples": 3}}
    arm2 = {"schema": "neuronrw/plan-arm2/v1", **mf, **common, "known": known_sub, "dose_scale": {"alpha_c": 8, "source": "dry"},
            "doses": [2.0, 4.0], "primary_dose": 4.0,
            "poles": [pole(0, 7, 1), pole(0, 8, -1), pole(1, 7, 1), pole(1, 8, -1)], "neutral_prompts": NEUTRAL,
            "label_prompts": [{**pole(0, 7, 1), "prompt": "Here is some text about the following: imports\n\n"},
                              {**pole(1, 7, 1), "prompt": "Here is some text about the following: tables\n\n"}],
            "sampling": {**sampling, "n_samples": 2}, "draw": {"rule_sha256": None, "arm0_agreement_sha256": None}}
    return {"arm0": arm0, "arm1": arm1, "arm2": arm2}


N_KNOWN_STAGE = 3          # Arm 1 and Arm 2 plans carry the first 3 of the 6 known contexts (make_plan keeps 20 of 100)
PLANT_GAIN = 8.0
PC_POLES = [(1, 5, 1), (1, 6, -1), (1, 70, 1), (1, 90, -1)]          # last layer: the direct path only


def plant_token(tokens: np.ndarray) -> int:
    """The smallest non-special id that never occurs in the fake corpus."""
    return min(set(range(3, VOCAB - 3)) - set(np.unique(tokens).tolist()))


@torch.no_grad()
def plant(model, token_id: int, poles: list = PC_POLES, gain: float = PLANT_GAIN):
    """Scale W_U[token_id] by `gain` (token_id absent from the corpus) and set each pole's W_down column (poles in the
    last layer) to sign * ||column|| * W_U[token_id] / ||W_U[token_id]||: pushing the pole raises that token's logit at
    every steered position, so the steered NLL of the corpus targets must rise (the dry run's known positive control;
    a random tiny model's own columns move NLL by a few millinats either way). Applied before any dry reference is
    computed."""
    from infra.pg.neurons_harvest import text_tower
    W_U = model.get_output_embeddings().weight
    W_U[token_id] = (W_U[token_id].float() * gain).to(W_U.dtype)
    e = W_U[token_id].float() / W_U[token_id].float().norm()
    for l, n, sg in poles:
        col = text_tower(model).layers[l].mlp.down_proj.weight
        col[:, n] = (sg * col[:, n].float().norm() * e).to(col.dtype)


def build(arch: str, out: pathlib.Path, stage: str, edit_plan=None, smoke: bool = False, rms: dict | None = None):
    """The Job on the tiny model for one stage (the pc poles planted, see `plant`), its plan written to and read back
    from out/inputs/; `edit_plan(plan)` may change the plan first (a planted failure); arm1 and arm2 take `rms`
    (arm0's rms.json)."""
    from infra.aws_pg.jobs import neuron_rw as J
    from tokenizers import Tokenizer
    model, tok = tiny_model(arch), tiny_tokenizer(BOS[arch])
    tokens = fake_corpus(tok)
    plant(model, plant_token(tokens))
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    tok.backend_tokenizer.save(str(inp / "tokenizer.json"))                 # A20: the file the firing tokens decode with
    tk_sha = hashlib.sha256((inp / "tokenizer.json").read_bytes()).hexdigest()
    plans = fake_plans(model, tok, tokens, arch, tk_sha)
    plan = plans[stage]
    if edit_plan:
        edit_plan(plan)
    if stage != "arm0" and rms is None:                             # a standalone arm1 / arm2 dry run: its own rms.json
        from infra.pg import neuron_rw_core as R
        pre = [] if tok.bos_token_id is None else [tok.bos_token_id]
        wins = [torch.tensor([pre + tokens[w * WINDOW:(w + 1) * WINDOW].tolist()]) for w in (0, 1, 2, 3)]
        r = R.residual_rms(model, wins, 1)
        rms = {"windows": [0, 1, 2, 3], "exclude_first": 1, "n_positions": r["n_positions"], "layers": r["layers"],
               "model_key": plan["model_key"], "n_layers": len(r["layers"])}
    (inp / f"{stage}.json").write_text(json.dumps(plan))
    raw = (inp / f"{stage}.json").read_bytes()
    meta = {"model_key": plan["model_key"], "repo": plan["repo"], "branch": None, "revision": None, "dry_run": True,
            "rms_source": None if rms is None else "dry arm0 rms.json"}
    return J.Job(out / stage, stage=stage, model=model, tok=tok, tokens=tokens, plan=json.loads(raw),
                 plan_sha256=hashlib.sha256(raw).hexdigest(), meta=meta, rms_banked=rms, smoke=smoke,
                 firing_tok=Tokenizer.from_file(str(inp / "tokenizer.json")), firing_tok_sha256=tk_sha,
                 known_arm0=None if stage == "arm0" else plans["arm0"]["known"])


def run_watched(job) -> list:
    """Run the job; after every metrics.json write, read it back from disk. Returns the progress dicts seen."""
    seen, save = [], job.save

    def save_and_read():
        save()
        m = json.loads((job.out / "metrics.json").read_text())
        seen.append({"status": m["status"], **m["progress"]})
    job.save = save_and_read
    job.run()
    return seen


SCHEMA_KEYS = {   # SCHEMA v5 §3: keys every output must carry
    "metrics": {"schema", "stage", "status", "started_utc", "updated_utc", "model_key", "repo", "revision", "branch",
                "device_name", "transformers", "torch", "pg_git_sha", "plan_sha256", "gates", "throughput", "progress"},
    "gates": {"determinism", "identity", "place", "gen", "known", "floor_random_dir", "floor_position", "pc", "coherence"},
    "throughput": {"forward_tokens", "forward_s", "forward_tokens_per_s", "gen_rows", "gen_new_tokens", "gen_s",
                   "gen_rows_per_s", "score_tokens", "score_s", "score_tokens_per_s", "gen_rows_per_s_arm1probe",
                   "arm1_mean_prompt_tokens"},
    "controls": {"identity", "known", "floor", "pc", "place", "gen", "coherence", "determinism"},
    "gen": {"pole", "alpha", "steps", "pass", "failing_input"},
    "place": {"pole", "alpha", "cos_median", "ratio_median", "pass", "failing_input", "reported"},
    "coherence": {"band", "failing_inputs", "passing_inputs", "pass"},
    "band": {"lo", "hi", "hi_alpha0", "hi_natural", "n_alpha0", "n_natural"},
    "sample": {"arm", "condition", "pole", "prompt_set", "prompt_i", "alpha", "alpha_abs", "layer", "k", "seed", "completion",
               "completion_token_ids", "n_new_tokens", "stopped_eos", "nll_clean_mean", "nll_clean_n"},
    "projection": {"pole", "layer", "neuron", "sign", "w_down_norm", "top", "bottom", "top_full", "bottom_full", "neighbour"},
    "repair": {"pole", "layer", "neuron", "sign", "w_down_norm", "failing_input", "contexts"},
    "repair_context": {"k", "rank", "tok", "target_id", "a_ours", "a_off", "clean_nll_ours", "clean_nll_off", "clean_nll_manual",
                       "direct_dnll", "direct_dnll_postrenorm", "direct_dnll_finalrenorm", "tonly_clean_nll",
                       "tonly_ablated_nll", "tonly_dnll", "tonly_identity_max_abs_dlogit", "dnll_off", "delta_r_norm",
                       "final_rms", "clean_nll_manual_early", "clean_nll_manual_model_dtype",
                       "clean_nll_manual_model_dtype_wrong_eps"},
    "natural_span": {"set", "window", "span", "nll_clean_mean"},
    "firing": {"pole", "prompt_set", "i", "prefix_property", "max_sign_a", "argmax_pos", "frac_of_banked_max", "fires"},
    "firing_tokens": {"pole", "tokens", "n_ranks_read", "n_distinct"},
    "rms": {"windows", "exclude_first", "n_positions", "layers"},
}
MODEL_GATES = ("determinism", "identity", "place", "gen", "known", "floor_random_dir", "floor_position", "pc")


def schema_check(out: pathlib.Path, stage: str) -> int:
    """Every SCHEMA v5 key on every output of the stage; returns the number of objects checked."""
    from infra.pg.neuron_rw_core import read_jsonl
    K = SCHEMA_KEYS
    m, ctl = json.loads((out / "metrics.json").read_text()), json.loads((out / "controls.json").read_text())
    checks = [(K["metrics"], m), (K["gates"], m["gates"]), (K["throughput"], m["throughput"]), (K["controls"], ctl),
              (K["coherence"], ctl["coherence"]), (K["band"], ctl["coherence"]["band"])] + [(K["place"], x) for x in ctl["place"]] + \
        [(K["gen"], x) for x in ctl["gen"]]
    samples = {"arm0": ["sweep/samples.jsonl"], "arm2": ["arm2/samples.jsonl"],
               "arm1": [str(f.relative_to(out)) for f in sorted((out / "arm1" / "samples").glob("*.jsonl"))]}[stage]
    rows = [r for f in samples for r in read_jsonl(out / f)]
    checks += [(K["sample"], r) for r in rows]
    assert all(("realized_cos" in r) == (r["pole"] is not None and r["condition"] == "steer") for r in rows)
    if stage == "arm0":
        checks += [(K["rms"], json.loads((out / "arm0" / "rms.json").read_text()))]
        checks += [(K["projection"], r) for r in read_jsonl(out / "arm0" / "projections.jsonl")]
        for f in sorted((out / "arm0" / "repair").glob("*.json")):
            rec = json.loads(f.read_text())
            checks += [(K["repair"], rec)] + [(K["repair_context"], c) for c in rec["contexts"]]
        checks += [(K["natural_span"], r) for r in read_jsonl(out / "arm0" / "natural_spans.jsonl")]
        checks += [(K["firing"], r) for r in read_jsonl(out / "arm0" / "firing.jsonl")]
        checks += [(K["firing_tokens"], r) for r in read_jsonl(out / "arm0" / "firing_tokens.jsonl")]
    for keys, obj in checks:
        assert keys <= set(obj), sorted(keys - set(obj))
    for f in ("samples/gates.jsonl", "samples/steer.jsonl", "samples/coherence_inputs.jsonl", "manifest.json"):
        assert (out / f).exists(), f
    return len(checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--stage", default="arm0,arm1,arm2", help="one stage, or a comma list run in order")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    summary = {"arch": a.arch}
    rms = None
    for stage in a.stage.split(","):
        job = build(a.arch, out, stage, rms=rms)
        seen = run_watched(job)
        m = json.loads((job.out / "metrics.json").read_text())
        ctl = json.loads((job.out / "controls.json").read_text())
        assert m["status"] == "complete" and all(m["gates"][g] for g in MODEL_GATES), m["gates"]
        assert isinstance(m["gates"]["coherence"], bool)      # recorded; a random model cannot pass G-coh (P8)
        n_checked = schema_check(job.out, stage)
        summary[stage] = {"n_metrics_writes": len(seen), "n_schema_objects_checked": n_checked, "gates": m["gates"],
                          "identity": (ctl["identity"]["max_abs_dlogit"], ctl["identity"]["failing_input"]["max_abs_dlogit"]),
                          "floor_random_dir": (ctl["floor"]["random_dir"]["max_abs_dlogit"],
                                               ctl["floor"]["random_dir"]["failing_input"]["max_abs_dlogit"]),
                          "floor_position": (ctl["floor"]["position"]["max_abs_dlogit_before_P_minus_1"],
                                             ctl["floor"]["position"]["failing_input"]["max_abs_dlogit_before_P_minus_1"]),
                          "gen": [(x["steps"][0]["cos"], x["steps"][-1]["ratio"], x["failing_input"]["steps"][0]["cos"])
                                  for x in ctl["gen"][:2]],
                          "place": [(x["cos_median"], x["ratio_median"], x["failing_input"]["cos_median"],
                                     x["failing_input"]["ratio_median"], x["reported"] and x["reported"]["ratio_median"])
                                    for x in ctl["place"][:2]],
                          "pc": (ctl["pc"]["mean_dnll"], ctl["pc"]["failing_input"]["mean_dnll"]),
                          "known_a_median_rel_err": ctl["known"]["a"]["median_rel_err"],
                          "known_failing_shift": ctl["known"]["failing_input"]["shift"],
                          "determinism": (ctl["determinism"]["max_abs_dlogit_clean_repeat"],
                                          ctl["determinism"]["failing_input"]["max_abs_dlogit"]),
                          "coherence": {k: v for k, v in ctl["coherence"].items() if k != "band"},
                          "last_progress": seen[-1], "throughput": m["throughput"]}
        if stage == "arm0":
            rms = json.loads((job.out / "arm0" / "rms.json").read_text())
    print("NEURON_RW_DRYRUN_OK", json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
