"""NEURONRW job (studies/neuronrw/SCHEMA.md v5 §2-3; SPEC.md §2, §6, §9.1.2; plans as built by plan/make_plan.py).

    python -m infra.aws_pg.jobs.neuron_rw --stage arm0|arm1|arm2 --model-key Q4B --repo Qwen/Qwen3-4B-Base \
        --corpus s3://pg-lineage-769063704696/results/neurons/w1/q4b/1/out/corpus/corpus --tokname qwen3 \
        --plan s3://pg-lineage-769063704696/results/neurons/rw/plans/<attempt>/<plan file> \
        [--rms <s3 path of the model's arm0/rms.json> --known-plan <s3 path of the model's arm0 plan>]  # arm1, arm2
        [--revision stage1-step1413814] [--smoke] --out <dir>

Order: load -> residual RMS (J1; the gates' doses are alpha * RMS, so it runs before them) -> model gates G-det,
G-ident, G-place, G-gen, G-known, G-floor-dir, G-floor-pos, G-pc (controls.json; a gate failing or its failing input
passing writes status "aborted_gate" and raises) -> stage work (arm0: sweep, Arm 1 throughput probe, final-norm gain,
projections, natural spans, firing check, judge0_pc firing tokens, repair) -> G-coh once the alpha = 0 samples exist (controls.json rewritten with `coherence`; never aborts, SPEC P8) -> status
complete. metrics.json is rewritten (temp file, then rename) after the gates, every projection chunk, every repair
pole, every generate call and G-coh. Samples are appended as each generate call returns. NaN and inf -> null.

Rules this file fixes where SCHEMA and SPEC name a quantity but not every step (for the spec writer to ratify):
- J1 RMS: arm0 measures it on plan.rms (each window's input BOS + the 2048 window tokens) and writes arm0/rms.json
  with model_key and n_layers added. arm1 and arm2 require --rms (that file): its model_key and n_layers are
  asserted, the RMS is re-measured on the file's own windows (<stage>/rms_check.json, every layer within RMS_REL_TOL
  relative, else the job raises) and the banked values are used. alpha_abs = alpha * rms of the pole's layer.
- J2 Gate inputs (SPEC §6): natural calibration spans 0 and 1 (BOS + 32 prompt + 96 cont tokens), P = len(BOS) +
  prompt_tokens, steering from P - 1; pole = plan.poles[0]. SCHEMA's identity block names "prompt_i": 0; the
  inputs are corpus spans, banked under "input".
- J3 G-ident: alpha 0, max |dlogit| over every position of both spans against the unhooked forward, pass == 0;
  failing input alpha 1.0.
- J4 G-floor-dir: numpy default_rng(floor.random_dir_seed) normal direction, normalized, at the pole's layer on both
  spans, alpha 0, pass == 0; failing input alpha = floor.alpha_fail. G-floor-pos: the pole at floor.alpha_fail from
  P - 1, max |dlogit| over the prediction columns before P - 1, pass == 0; failing input start 0.
- J5 G-place: one controls.json entry per (first 3 pc poles, alpha of place.alphas) on natural span 0: the realized
  change of the layer output (neuron_rw_core.realized_change) passes when cos_min <= median cos <= 1 + 1e-6 and the
  median norm ratio is inside ratio_band; failing input the same vector at the MLP input (each must fail); reported
  for Gemma 3 and Olmo 3: the down_proj-output placement. The gate passes when every entry passes and every
  failing input fails. A21: per G-place pole one more entry at alpha = -1 (cos against the signed intended vector):
  the real hook must pass, and its failing input, the hook adding -v, gives cos about -1.0 and must fail. G-gen (A2): one entry per (first 3 pc poles, alpha of gen_gate.alphas): gen_gate.rows rows on
  neutral prompt 0, one cached generate of max(gen_gate.steps) + 1 tokens, median over rows of cos and norm ratio of
  the realized change at each decode step of gen_gate.steps; every step inside the bounds passes; failing input the
  same call with the hook keyed on the index within the current forward (decode steps get no vector; must fail).
- J6 G-pc: texts = natural spans named by pc.texts ("natural_spans calibration spans a..b", calibration spans in
  window order then span order); per pc pole the mean over every cont token of NLL(steered from P - 1 at pc.alpha) -
  NLL(unhooked); mean_dnll = mean over poles; ci95 = percentile bootstrap over poles (10,000,
  default_rng(20261025)); pass = floor <= mean <= ceiling; failing input alpha 0 on the same poles and texts.
- J7 G-known: contexts grouped by corpus window, each window's input BOS + window tokens up to its last context
  (NEURONOFF Arm A); arm1 and arm2 read the Arm 0 plan's known block (--known-plan, A26: their own 20-context block
  cuts some windows shorter, and the bf16 forward of a shorter window drifts), with the reading on their own block
  printed beside as "truncated_windows_reading"; the pole hook is neuron_off_core.PoleAblator (mode pole, mask domain from window position 1);
  a.pass = median |a / a_banked - 1| <= a_rel_tol (exact matches and sign mismatches counted); own_clean_nll_mean,
  own_dnll_mean and nll_corpus_first_120k against their banked values within nll_abs_tol, a null banked value giving
  a null block. Failing input (A24): the MLP outputs of layers 0 .. n_layers // 4 zeroed on the mask domain (layer 0
  alone moved OLMo 3's 120k-token NLL by 0.114 nats); the per-model shift is banked; the readout it moves is
  nll_corpus_first_120k when banked (O7), else own_clean_nll_mean; it passes (a defect) when the shift is below
  failing_min_shift_nats.
- J8 G-det as NEURONOFF: windows 0 and 1 forwarded twice, max |dlogit| == 0; failing input token 100 of window 1
  replaced by (id + 1) mod V, compared from position 100 on.
- J9 Placement: every layer the stage steers passes neuron_rw_core.placement_check (module names and child call
  order) before the gates, else the job raises; banked in controls.json "placement".
- J10 Generation (sweep, arm1, arm2): per prompt, the alpha = 0 rows (pole null; n_samples_alpha0 in the sweep,
  n_samples otherwise) once, then every pole x nonzero dose x n_samples rows in plan order, chunked at max_rows;
  torch.manual_seed(base_seed + 1000 * prompt_i + c) before the c-th call of the prompt. k counts the rows of one
  (pole, alpha, prompt). realized_cos (steered rows only; A2) is the minimum over prefill P - 1 and decode steps 1-3
  of cos(bf16(h + v) - h, v), v = alpha * RMS * w the signed intended vector (A21); the analysis reads the cell median.
- J11 Firing check (arm0, SPEC §5.5): per plan.firing_check entry and prompt of its set, max over positions >= 1 of
  sign * a_i in one clean forward of BOS + prompt + continuation (neuron_rw_core.firing_max); fires =
  frac_of_banked_max >= firing_rule.threshold_frac; prefix_property as NEURONOFF A7. The selection of the TeX
  replacement from these lines is `neuron_rw_analysis replacement` (SPEC §5.5; the job plan carries no roles).
- J12 Natural spans (arm0): every calibration and test span k of window w = tokens [w*2048 + 128k, +128), BOS + the
  first 32 as context, nll_clean_mean over the next 96; arm1 and arm2 recompute both sets for G-coh.
- J13 G-coh (SPEC §5.2, §6), every generating stage: samples/coherence_inputs.jsonl holds failing 1 = neutral prompt
  0 + the attested top-1 projection token of the first sweep pole (arm0) or plan.poles[0] (arm1, arm2, which carry no
  sweep) repeated 96 times; failing 2 = neutral prompt 0 + 96 attested ids by default_rng([20261025, model_index,
  10]).choice; passing 1 = the natural test spans; passing 2 = the test half of the stage's alpha = 0 rows
  (neuron_rw_core.split_halves, default_rng([20261025, model_index, 12]).permutation; A7); passing 3 (arm2) = the
  label-prompted rows. The band: L = p2.5 of the calibration half, U = max(p97.5 calibration half, p97.5 natural
  calibration spans). The job's band uses the stage's own alpha = 0 rows (arm2: Arm 2's only; the analysis pools
  the sweep's, SPEC §5.2).
- J14 arm2 plan.label_prompts: per entry n_samples unhooked rows (condition "label_prompt", the entry's pole,
  alpha 0, prompt_i null) in arm2/samples.jsonl, seed base_seed + 1000 * (100 + entry index) + c.
- J15 Firing tokens (arm0, SPEC §5.1, A20): per plan.judge0_pc entry, the distinct strings decoded (skip_special_tokens
  False) with the corpus tokenizer file whose sha256 the plan records, over all its top_tok positions (top 1,000),
  ranked by frequency, ties by best rank, top 20 (firing_set); the job asserts the distinct count equals the plan's
  n_distinct and is at least 10; the row banks n_distinct and n_ranks_read = len(top_tok).
- J16 Repair (arm0, A6): per window one clean forward (neuron_rw_core.repair_readout: direct path with both norm
  scales frozen, the two recomputed variants, clean NLL, the manual NLL in the model dtype, the failing-input NLL from
  the residual entering layer n_layers // 2), then per context one t-only ablation forward
  (neuron_off_core.PoleAblator, mode pole, exclude_first = t) read at the same positions; tonly_dnll = ablated -
  clean NLL. At k = 0 of each pole the same forward with the identity hook gives tonly_identity_max_abs_dlogit
  (G-repair-fit needs 0); other contexts carry null there. Each repair file and controls.json "repair_failing_input"
  bank the pole's failing-input layer and median |clean_nll_manual_early - clean_nll_ours| (the margin against the
  analysis' 0.5-nat rule) and the median |clean_nll_manual - clean_nll_ours|.
- J17 Throughput probe (arm0, A9): plan.throughput_probe.calls generate calls of .rows rows on the Arm 1 prompt, rows
  steered at the sweep poles x sweep doses in turn; gen_rows_per_s_arm1probe = rows / generation seconds;
  arm1_mean_prompt_tokens over plan.prompt_sets (BOS included). Completions are not banked.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time

import numpy as np

from infra.aws_pg.jobs.neuron_off import append_jsonl, dump, finite, stem
from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus

METRICS_SCHEMA = "neuronrw/metrics/v1"
PLAN_SCHEMA = {"arm0": "neuronrw/plan-arm0/v1", "arm1": "neuronrw/plan-arm1/v1", "arm2": "neuronrw/plan-arm2/v1"}
SEED = 20261025
N_BOOT = 10_000
IDENTITY_FAIL_ALPHA = 1.0
DET_FAIL_POS = 100
GATE_TOKENS = 512
RMS_REL_TOL = 1e-4
N_BEFORE = 64
SMOKE = {"projections": 20, "repair": 3, "sweep": 2}
PC_TEXTS = re.compile(r"natural_spans calibration spans (\d+)\.\.(\d+)")
MODEL_INDEX = {"Q4B": 0, "G4": 1, "O7": 2}          # arm2_rule.json model_index
NEUTRAL_PROMPT_0 = "Here is some text:\n"           # SPEC §4.7, the context of the registered coherence inputs
COH_LEN = 96                                         # SPEC §5.2 registered failing inputs: 96 tokens
N_FIRING_TOKENS, N_FIRING_MIN = 20, 10      # A20: top 20 distinct decoded strings by frequency, floor 10
NEG_ALPHA = -1.0                            # A21: G-place's negative-dose registered inputs


def firing_set(strings: list) -> tuple:
    """A20 (J15): the distinct decoded strings of a pole's top-1,000 positions (in rank order), ranked by frequency over
    the positions, ties by best rank; (the top N_FIRING_TOKENS, the number of distinct strings). Same rule as
    plan/make_plan.py:firing_set (tests assert they agree)."""
    count, first = {}, {}
    for k, s in enumerate(strings):
        count[s] = count.get(s, 0) + 1
        first.setdefault(s, k)
    return sorted(count, key=lambda s: (-count[s], first[s]))[:N_FIRING_TOKENS], len(count)


def sstr(sign: int) -> str:
    assert sign in (1, -1), f"plan sign must be +1 or -1, got {sign!r}"
    return "+" if sign == 1 else "-"


def special_ids(tok) -> list:
    """The ids of the tokenizer's added tokens marked special (tokenizer.json added_tokens with special true)."""
    return sorted(i for i, t in tok.added_tokens_decoder.items() if t.special)


def attested_ids(tokens: np.ndarray, special: list) -> np.ndarray:
    """The plan's attested_vocab rule: ids with count >= 1 in tokens.npy, minus the special ids, sorted, int32."""
    return np.setdiff1d(np.unique(tokens), np.asarray(special, dtype=np.int64)).astype("<i4")


def ids_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(ids).astype("<i4").tobytes()).hexdigest()


def boot_ci(x, seed: int = SEED, n_boot: int = N_BOOT) -> list:
    x = np.asarray(x, dtype=np.float64)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


class Job:
    def __init__(self, out, *, stage: str, model, tok, tokens, plan: dict, plan_sha256: str, meta: dict,
                 rms_banked: dict | None = None, smoke: bool = False, firing_tok=None, firing_tok_sha256: str | None = None,
                 known_arm0: dict | None = None):
        import torch
        from infra.pg import neuron_off_core as C
        from infra.pg import neuron_rw_core as R
        from infra.pg.neurons_harvest import utc
        assert stage in PLAN_SCHEMA, stage
        assert plan["schema"] == PLAN_SCHEMA[stage], f"plan schema {plan['schema']} for stage {stage}"
        assert not smoke or stage == "arm0", "--smoke runs stage arm0 only"
        assert (rms_banked is None) == (stage == "arm0"), "arm1 and arm2 take --rms (arm0's rms.json); arm0 does not"
        assert tok.bos_token_id == plan["bos_id"], f"tokenizer BOS {tok.bos_token_id} != plan bos_id {plan['bos_id']}"
        if stage == "arm0":                       # A20: firing tokens decode with the plan's tokenizer file
            want = plan["judge0_pc_rule"]["tokenizer"]["sha256"]
            assert firing_tok is not None and firing_tok_sha256 == want, \
                f"firing-token tokenizer sha256 {firing_tok_sha256} != plan {want}"
        self.firing_tok = firing_tok
        if stage == "arm0":                       # A26: G-known reads the Arm 0 known windows in every stage
            assert known_arm0 is None, "arm0 reads its own plan's known block"
            self.known = plan["known"]
        else:
            assert known_arm0 is not None, "arm1 and arm2 take --known-plan, the Arm 0 plan whose known block G-known reads (A26)"
            k1 = plan["known"]
            assert all(k1[x] == known_arm0[x] for x in ("pole", "layer", "neuron", "sign", "nll_abs_tol", "a_rel_tol")), \
                "the Arm 0 known block names another pole or tolerance than this plan's"
            r0 = [c["rank"] for c in known_arm0["contexts"]]
            assert [c["rank"] for c in k1["contexts"]] == r0[:len(k1["contexts"])], "this plan's known contexts are not Arm 0's first ones"
            self.known = known_arm0
        assert R.d_model(model) == plan["d_model"], f"model d_model {R.d_model(model)} != plan {plan['d_model']}"
        if "n_layers" in plan:
            assert len(R.text_tower(model).layers) == plan["n_layers"], "model and plan n_layers differ"
        if model.device.type == "cuda":
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "set CUBLAS_WORKSPACE_CONFIG before CUDA starts"
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        self.torch, self.C, self.R, self.utc = torch, C, R, utc
        self.stage, self.model, self.tok, self.tokens, self.plan, self.smoke = stage, model, tok, tokens, plan, smoke
        self.rms_banked = rms_banked
        self.out = pathlib.Path(out)
        for d in ("samples", stage) + (("arm0/repair", "sweep") if stage == "arm0" else ()) + \
                (("arm1/samples",) if stage == "arm1" else ()):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.W = int(plan["window"])
        self.prefix = C.bos_prefix(tok)
        self.excl = len(self.prefix) + 1
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.S = plan["sampling"]
        for p in self.steered_poles():
            assert p["pole"] == C.pole_id(p["layer"], p["neuron"], sstr(p["sign"])), f"pole id {p['pole']} vs its fields"
        self.tp = {"forward_tokens": 0, "forward_s": 0.0, "gen_rows": 0, "gen_new_tokens": 0, "gen_s": 0.0,
                   "score_tokens": 0, "score_s": 0.0}
        self.stage_s, self.rms_abs = {}, None
        self.probe = {"gen_rows_per_s_arm1probe": None, "arm1_mean_prompt_tokens": None}
        gen = C.generation_config(tok, self.S["temperature"], self.S["max_new_tokens"], self.S["top_k"], self.S["top_p"])
        self.m = {"schema": METRICS_SCHEMA, "stage": stage, "status": "running", "started_utc": utc(), "updated_utc": None,
                  "model_key": plan["model_key"], **{k: meta.get(k) for k in ("repo", "revision", "branch", "device_name",
                                                                              "transformers", "torch", "pg_git_sha")},
                  "plan_sha256": plan_sha256, "gates": None, "throughput": None, "progress": {}}
        self.manifest = {**meta, "stage": stage, "smoke": smoke, "plan_sha256": plan_sha256, "dtype": str(model.dtype),
                         "attn_implementation": model.config._attn_implementation,
                         "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "argv": sys.argv,
                         "generation_config": gen.to_dict(), "deterministic_algorithms": True, "seed": SEED,
                         "rms_source": "measured in this stage (arm0/rms.json)" if stage == "arm0" else meta.get("rms_source"),
                         "s3_prefix": os.environ.get("PG_S3_PREFIX"), "started_utc": self.m["started_utc"],
                         "finished_utc": None, "stage_s": self.stage_s}

    # ---------------------------------------------------------------------------------- plumbing
    def steered_poles(self) -> list:
        p = self.plan
        return list(p["poles"]) + list(p["pc"]["poles"]) + (list(p["sweep"]["poles"]) if self.stage == "arm0" else [])

    def save(self):
        t = dict(self.tp)
        for k in ("forward", "score"):
            t[f"{k}_tokens_per_s"] = t[f"{k}_tokens"] / t[f"{k}_s"] if t[f"{k}_s"] > 0 else None
        t["gen_rows_per_s"] = t["gen_rows"] / t["gen_s"] if t["gen_s"] > 0 else None
        t.update(self.probe)
        for k in ("forward_s", "gen_s", "score_s"):
            t[k] = round(t[k], 3)
        self.m["throughput"] = t
        self.m["updated_utc"] = self.utc()
        dump(self.out / "metrics.json", self.m)

    def sync(self):
        if self.model.device.type == "cuda":
            self.torch.cuda.synchronize()

    def timed(self, kind: str, n_tokens: int, fn):
        t0 = time.time()
        r = fn()
        self.sync()
        self.tp[f"{kind}_s"] += time.time() - t0
        self.tp[f"{kind}_tokens"] += n_tokens
        return r

    def stage_t(self, name, t0):
        self.stage_s[name] = round(self.stage_s.get(name, 0.0) + time.time() - t0, 2)

    def ids(self, start: int, n: int):
        assert 0 <= start and start + n <= self.tokens.size, f"tokens [{start}, {start + n}) outside the corpus"
        return self.torch.tensor([self.prefix + self.tokens[start:start + n].tolist()])

    def dec(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])

    def alpha_abs(self, layer: int, alpha: float) -> float:
        return float(alpha) * self.rms_abs[layer]

    def direction(self, p: dict):
        return self.R.pole_direction(self.model, p["layer"], p["neuron"], p["sign"])

    def steer(self, p: dict, alpha: float, start: int, w=None):
        return self.R.steer_one(self.model, p["layer"], self.direction(p) if w is None else w,
                                self.alpha_abs(p["layer"], alpha), start)

    def fwd(self, ids, hook=None, keep_logits=False, positions=None, targets=None):
        return self.timed("forward", ids.numel(), lambda: self.C.next_token(self.model, ids, hook, keep_logits, positions, targets))

    def dlogit(self, a, b) -> float:
        return float(self.C.max_abs_dlogit(a, b).max())

    def logits(self, ids: list, hook=None):
        t = self.torch.tensor([ids])
        return self.timed("forward", len(ids), lambda: self.R.logits_with(self.model, t, hook))

    def tnll(self, ids: list, P: int, hook=None):
        return self.timed("forward", len(ids), lambda: self.R.steered_nll(self.model, ids, P, hook))

    def seed(self, i: int, c: int) -> int:
        return self.S["base_seed"] + 1000 * i + c

    def generate(self, prompt: str, rows: list, seed: int) -> list:
        """One generate call (rows: (layer, w, alpha_abs)) plus clean scoring of every row."""
        R = self.R
        t0 = time.time()
        res = R.generate_rows(self.model, self.tok, prompt, rows, seed, self.S)
        self.sync()
        self.tp["gen_s"] += time.time() - t0
        self.tp["gen_rows"] += len(rows)
        self.tp["gen_new_tokens"] += sum(r["n_new_tokens"] for r in res)
        p = R.prompt_ids(self.tok, prompt)
        comps = [r["completion_token_ids"] for r in res]
        nll = self.timed("score", sum(len(p) + len(c) for c in comps if c), lambda: R.score_rows(self.model, p, comps, self.pad))
        for r, v in zip(res, nll):
            r["nll_clean_mean"], r["nll_clean_n"] = v, r["n_new_tokens"]
        return res

    def record(self, arm, pole, prompt_set, i, alpha, k, seed, r, condition="steer") -> dict:
        """SCHEMA §3.4; realized_cos only on steered rows."""
        steered = pole is not None and condition == "steer"
        r = {k_: v for k_, v in r.items() if k_ != "realized_steps" and (k_ != "realized_cos" or steered)}
        return {"arm": arm, "condition": condition, "pole": None if pole is None else pole["pole"], "prompt_set": prompt_set,
                "prompt_i": i, "alpha": float(alpha), "alpha_abs": self.alpha_abs(pole["layer"], alpha) if steered else 0.0,
                "layer": pole["layer"] if steered else None, "k": k, "seed": seed, **r}

    def chunks(self, rows: list) -> list:
        n = self.S["max_rows"]
        return [rows[s:s + n] for s in range(0, len(rows), n)]

    def span(self, w: int, j: int) -> dict:
        ns = self.plan["natural_spans"]
        L = ns["prompt_tokens"] + ns["cont_tokens"]
        assert ns["spans_per_window"] * L <= self.W, "natural spans overflow the window"
        s = w * self.W + j * L
        return {"window": w, "start": j * L, "prompt_ids": self.tokens[s:s + ns["prompt_tokens"]].tolist(),
                "cont_ids": self.tokens[s + ns["prompt_tokens"]:s + L].tolist()}

    def calibration_spans(self) -> list:
        ns = self.plan["natural_spans"]
        return [(w, j) for w in ns["calibration_windows"] for j in range(ns["spans_per_window"])]

    # ---------------------------------------------------------------------------------- RMS (J1)
    def measure_rms(self):
        t0 = time.time()
        src = self.plan["rms"] if self.rms_banked is None else self.rms_banked
        wins = [self.ids(w * self.W, self.W) for w in src["windows"]]
        r = self.timed("forward", sum(w.numel() for w in wins),
                       lambda: self.R.residual_rms(self.model, wins, src["exclude_first"]))
        rec = {"windows": list(src["windows"]), "exclude_first": src["exclude_first"], "n_positions": r["n_positions"],
               "layers": r["layers"]}
        if self.rms_banked is None:
            dump(self.out / "arm0" / "rms.json", {**rec, "model_key": self.plan["model_key"], "n_layers": len(r["layers"])})
            self.rms_abs = [x["rms"] for x in r["layers"]]
        else:
            assert (self.rms_banked["model_key"], self.rms_banked["n_layers"]) == \
                (self.plan["model_key"], len(self.R.text_tower(self.model).layers)), "--rms names another model or depth"
            banked = [x["rms"] for x in self.rms_banked["layers"]]
            ours = [x["rms"] for x in r["layers"]]
            assert len(banked) == len(ours), f"--rms has {len(banked)} layers, the model {len(ours)}"
            rel = max(abs(a / b - 1.0) for a, b in zip(ours, banked))
            dump(self.out / self.stage / "rms_check.json", {**rec, "banked_rms": banked, "max_rel_diff": rel,
                                                            "tol": RMS_REL_TOL})
            assert rel <= RMS_REL_TOL, f"re-measured RMS differs from --rms by {rel:.3g} relative"
            self.rms_abs = banked
        assert all(math.isfinite(x) and x > 0 for x in self.rms_abs), f"residual RMS not finite and positive: {self.rms_abs}"
        self.stage_t("rms", t0)

    # ---------------------------------------------------------------------------------- gates (SPEC §6, SCHEMA §3.1)
    def gate_spans(self) -> list:
        """J2: natural calibration spans 0 and 1 as (ids, P)."""
        cal, out = self.calibration_spans(), []
        for k in (0, 1):
            s = self.span(*cal[k])
            out.append((self.prefix + s["prompt_ids"] + s["cont_ids"], len(self.prefix) + len(s["prompt_ids"])))
        return out

    def g_identity(self, p0, spans, clean):
        mx = max(self.dlogit(self.logits(ids, self.steer(p0, 0.0, P - 1)), c) for (ids, P), c in zip(spans, clean))
        fx = max(self.dlogit(self.logits(ids, self.steer(p0, IDENTITY_FAIL_ALPHA, P - 1)), c) for (ids, P), c in zip(spans, clean))
        return {"alpha": 0.0, "max_abs_dlogit": mx, "pass": mx == 0.0,
                "failing_input": {"alpha": IDENTITY_FAIL_ALPHA, "max_abs_dlogit": fx, "pass": fx == 0.0}, "pole": p0["pole"]}

    def g_floor(self, p0, spans, clean):
        fa = self.plan["floor"]["alpha_fail"]
        rnd = self.R.random_direction(self.model, self.plan["floor"]["random_dir_seed"])

        def rd(alpha):
            return max(self.dlogit(self.logits(ids, self.steer(p0, alpha, P - 1, rnd)), c) for (ids, P), c in zip(spans, clean))

        def before(start_at_zero):
            out = 0.0
            for (ids, P), c in zip(spans, clean):
                assert P - 1 >= 1, f"prompt too short for the position floor (P = {P})"
                lg = self.logits(ids, self.steer(p0, fa, 0 if start_at_zero else P - 1))
                out = max(out, self.dlogit(lg[:, :P - 1], c[:, :P - 1]))
            return out
        mx, fx, pm, pf = rd(0.0), rd(fa), before(False), before(True)
        return {"random_dir": {"alpha": 0.0, "max_abs_dlogit": mx, "pass": mx == 0.0,
                               "failing_input": {"alpha": fa, "max_abs_dlogit": fx, "pass": fx == 0.0}},
                "position": {"alpha": fa, "max_abs_dlogit_before_P_minus_1": pm, "pass": pm == 0.0,
                             "failing_input": {"start": 0, "max_abs_dlogit_before_P_minus_1": pf, "pass": pf == 0.0}}}

    def g_place(self, span0) -> list:
        """J5 / SPEC §6 G-place: the first 3 pc poles on natural span 0 at each alpha of plan.place; SCHEMA §3.1 entries."""
        R, pl, (ids, P) = self.R, self.plan["place"], span0
        lo, hi = pl["ratio_band"]

        def ok(r):
            return pl["cos_min"] <= r["cos_median"] <= 1 + 1e-6 and lo <= r["ratio_median"] <= hi
        post = R.POST_MLP_NORM[R.model_type(self.model)] is not None
        out = []
        for q in self.plan["pc"]["poles"][:3]:
            w = self.direction(q)
            for a in pl["alphas"]:
                a_abs = self.alpha_abs(q["layer"], a)
                read = lambda where: self.timed("forward", 2 * len(ids), lambda: R.realized_change(  # noqa: E731
                    self.model, ids, P, q["layer"], w, a_abs, where))
                good, bad = read("layer_output"), read("mlp_input")
                out.append({"pole": q["pole"], "alpha": a, **{k: good[k] for k in ("cos_median", "ratio_median", "cos_min",
                                                                                     "ratio_min", "ratio_max", "n_positions")},
                            "pass": ok(good), "failing_input": {**bad, "pass": ok(bad)},
                            "reported": read("down_proj_output") if post else None})
            a_abs = self.alpha_abs(q["layer"], NEG_ALPHA)             # A21: alpha = -1, real hook and wrong-sign hook
            good, bad = read("layer_output"), read("layer_output_wrong_sign")
            out.append({"pole": q["pole"], "alpha": NEG_ALPHA, **{k: good[k] for k in ("cos_median", "ratio_median", "cos_min",
                                                                                         "ratio_min", "ratio_max", "n_positions")},
                        "pass": ok(good), "failing_input": {**bad, "hook": "wrong sign (adds -v), A21", "pass": ok(bad)},
                        "reported": None})
        return out

    def g_gen(self) -> list:
        """SPEC §6 G-gen (A2): the first 3 pc poles, neutral prompt 0, gen_gate.rows rows, each alpha of gen_gate; the
        realized change at decode steps 1-3 of a cached generate (median over rows); every step must have cos_min <=
        cos <= 1 + 1e-6 and ratio inside ratio_band. Failing input: the same call with the hook keyed on the index
        within the current forward (it never steers a decode step)."""
        R, gg = self.R, self.plan["gen_gate"]
        lo, hi = gg["ratio_band"]
        if "neutral_prompts" in self.plan:
            assert self.plan["neutral_prompts"][0] == NEUTRAL_PROMPT_0, "neutral prompt 0 differs from SPEC §4.7"
        S = {**self.S, "max_new_tokens": max(gg["steps"]) + 1}

        def read(q, alpha, keyed):
            rows = [(q["layer"], self.direction(q), self.alpha_abs(q["layer"], alpha))] * gg["rows"]
            res = self.timed("forward", gg["rows"] * S["max_new_tokens"],
                             lambda: R.generate_rows(self.model, self.tok, NEUTRAL_PROMPT_0, rows, SEED, S, keyed))
            steps = []
            for st in gg["steps"]:
                vals = [(c, r_) for row in res for (k, c, r_) in row["realized_steps"] if k == st]
                assert len(vals) == gg["rows"], f"decode step {st} read on {len(vals)} of {gg['rows']} rows"
                steps.append({"step": st, "cos": float(np.median([c for c, _ in vals])), "ratio": float(np.median([r_ for _, r_ in vals]))})
            ok = all(gg["cos_min"] <= x["cos"] <= 1 + 1e-6 and lo <= x["ratio"] <= hi for x in steps)
            return steps, ok
        out = []
        for q in self.plan["pc"]["poles"][:3]:
            for a in gg["alphas"]:
                good, ok = read(q, a, "absolute")
                bad, bad_ok = read(q, a, "per_forward")
                out.append({"pole": q["pole"], "alpha": a, "steps": good, "pass": ok,
                            "failing_input": {"hook": "per-forward index", "steps": bad, "pass": bad_ok}})
        return out

    def pc_texts(self) -> list:
        m = PC_TEXTS.fullmatch(self.plan["pc"]["texts"])
        assert m, f"pc.texts {self.plan['pc']['texts']!r} is not 'natural_spans calibration spans a..b'"
        cal = self.calibration_spans()
        a, b = int(m[1]), int(m[2])
        assert 0 <= a <= b < len(cal), f"pc spans {a}..{b} outside the {len(cal)} calibration spans"
        return [self.span(*cal[k]) for k in range(a, b + 1)]

    def pc_mean(self, texts: list, alpha: float) -> dict:
        """J6: per pc pole, mean steered - clean NLL over every cont token of every text."""
        rows = []
        for s in texts:
            ids, P = self.prefix + s["prompt_ids"] + s["cont_ids"], len(self.prefix) + len(s["prompt_ids"])
            rows.append((ids, P, self.tnll(ids, P)))
        n = sum(len(ids) - P for ids, P, _ in rows)
        per = {}
        for q in self.plan["pc"]["poles"]:
            d = [float((self.tnll(ids, P, self.steer(q, alpha, P - 1)) - c).double().sum()) for ids, P, c in rows]
            per[q["pole"]] = sum(d) / n
        return {"per": per, "n": n}

    def g_pc(self):
        pc = self.plan["pc"]
        texts = self.pc_texts()
        r, r0 = self.pc_mean(texts, pc["alpha"]), self.pc_mean(texts, 0.0)
        v = list(r["per"].values())
        mean, mean0 = float(np.mean(v)), float(np.mean(list(r0["per"].values())))
        lo_hi = (pc["floor_nats"], pc["ceiling_nats"])
        return {"alpha": pc["alpha"], "n_poles": len(v), "n_positions": r["n"] * len(v), "texts": pc["texts"],
                "mean_dnll": mean, "ci95": boot_ci(v), "per_pole_mean_dnll": r["per"], "floor_nats": lo_hi[0],
                "ceiling_nats": lo_hi[1], "pass": lo_hi[0] <= mean <= lo_hi[1],
                "failing_input": {"alpha": 0.0, "mean_dnll": mean0, "pass": lo_hi[0] <= mean0 <= lo_hi[1]}}

    def by_window(self, contexts: list):
        """(contexts, ids, positions) per corpus window, each window's ids cut at its last context of `contexts`."""
        W, P = self.W, len(self.prefix)
        groups = {}
        for c in contexts:
            groups.setdefault(c["tok"] // W, []).append(c)
        for w in sorted(groups):
            cs = groups[w]
            ids = self.ids(w * W, max(c["tok"] % W for c in cs) + 1)
            pos = [P + c["tok"] % W for c in cs]
            for c, q in zip(cs, pos):
                assert int(ids[0, q]) == int(self.tokens[c["tok"]]), f"rank {c['rank']}: context ids misaligned"
                assert c["target_id"] == int(self.tokens[c["tok"] + 1]), f"rank {c['rank']}: target is not the next token"
            yield cs, ids, pos

    def known_zero(self, last: int | None = None):
        """A24: the G-known failing input's hook, MLP outputs of layers 0 .. n_layers // 4 (or 0 .. last) zeroed on the
        mask domain."""
        return self.R.MlpZeroLayers(self.model, self.R.known_zero_last(self.model) if last is None else last, self.excl)

    def known_read(self, kn: dict, zero_early: bool = False, last: int | None = None) -> dict:
        """J7: per context rank, the clean NLL; without zero_early also the hook's clean a_i and the ablated NLL."""
        C = self.C
        a, clean, abl, wins = {}, {}, {}, []
        for cs, ids, pos in self.by_window(kn["contexts"]):
            wins.append((cs[0]["tok"] // self.W, int(ids.shape[1])))
            tgt = [c["target_id"] for c in cs]
            r0 = self.fwd(ids, self.known_zero(last) if zero_early else None, positions=pos, targets=tgt)
            for k, c in enumerate(cs):
                clean[c["rank"]] = float(r0["nll"][0, k])
            if zero_early:
                continue
            pole = C.PoleAblator(self.model, "pole", kn["layer"], kn["neuron"], sstr(kn["sign"]), exclude_first=self.excl)
            r1 = self.fwd(ids, pole, positions=pos, targets=tgt)
            for k, c in enumerate(cs):
                a[c["rank"]], abl[c["rank"]] = float(r1["a"][0, pos[k]]), float(r1["nll"][0, k])
        return {"a": a, "clean": clean, "ablated": abl, "windows": wins}

    def corpus_nll(self, n_targets: int, zero_early: bool = False, last: int | None = None) -> float:
        s, n, w = 0.0, 0, 0
        while n < n_targets:
            nll = self.fwd(self.ids(w * self.W, self.W), self.known_zero(last) if zero_early else None)["nll"][0].double()
            take = min(n_targets - n, nll.numel())
            s, n, w = s + float(nll[:take].sum()), n + take, w + 1
        return s / n

    def known_reading(self, kn: dict, corpus) -> dict:
        """J7: the G-known readings of one known block (contexts grouped by window, each window cut at its last context
        of this block) against its banked values; `windows` = [window, forward length] as read."""
        tol = kn["nll_abs_tol"]
        r, cs = self.known_read(kn), kn["contexts"]
        rel = [abs(r["a"][c["rank"]] / c["a_banked"] - 1.0) for c in cs]
        a_blk = {"median_rel_err": float(np.median(rel)), "max_rel_err": float(max(rel)), "n_contexts": len(cs),
                 "n_exact": sum(x == 0.0 for x in rel), "n_sign_mismatch": sum(r["a"][c["rank"]] * kn["sign"] <= 0 for c in cs),
                 "a_rel_tol": kn["a_rel_tol"], "pass": float(np.median(rel)) <= kn["a_rel_tol"]}
        ours_clean = float(np.mean([r["clean"][c["rank"]] for c in cs]))
        ours_dnll = float(np.mean([r["ablated"][c["rank"]] - r["clean"][c["rank"]] for c in cs]))

        def blk(ours, banked):
            return None if banked is None else \
                {"ours": ours, "banked": banked, "abs_diff": abs(ours - banked), "pass": abs(ours - banked) <= tol}
        out = {"a": a_blk, "own_clean_nll_mean": blk(ours_clean, kn["own_clean_nll_mean_banked"]),
               "own_dnll_mean": blk(ours_dnll, kn["own_dnll_mean_banked"]),
               "nll_corpus_first_120k": None if corpus is None else blk(corpus, kn["nll_corpus_first_120k"]),
               "ours": {"own_clean_nll_mean": ours_clean, "own_dnll_mean": ours_dnll}, "nll_abs_tol": tol,
               "windows": [[w, n] for w, n in r["windows"]]}
        blocks = [out[k] for k in ("own_clean_nll_mean", "own_dnll_mean", "nll_corpus_first_120k") if out[k] is not None]
        assert blocks, "G-known has no banked NLL to compare against"
        out["pass"] = a_blk["pass"] and all(b["pass"] for b in blocks)
        return out

    def g_known(self):
        """J7 / SPEC §6 G-known on self.known: this plan's block in arm0, the Arm 0 plan's block in arm1 and arm2 (A26),
        so every stage reads the same windows as Arm 0; arm1 and arm2 print the reading on their own plan's block (its
        windows cut at its last context) beside, reported and not gating."""
        kn, tol = self.known, self.known["nll_abs_tol"]
        corpus = None if kn.get("nll_corpus_first_120k") is None else self.corpus_nll(120_000)
        out = self.known_reading(kn, corpus)
        ours_clean = out["ours"]["own_clean_nll_mean"]

        def failed(last=None):
            if corpus is None:
                return float(np.mean([v for v in self.known_read(kn, zero_early=True, last=last)["clean"].values()]))
            return self.corpus_nll(120_000, zero_early=True, last=last)
        name, ours = ("own_clean_nll_mean", ours_clean) if corpus is None else ("nll_corpus_first_120k", corpus)
        fail, last = failed(), self.R.known_zero_last(self.model)
        shift = abs(fail - ours)
        fail0 = fail if last == 0 else failed(0)                    # A24: the layer-0-only shift, a reported reading
        out["failing_input"] = {"rule": f"MLP outputs of layers 0..{last} (n_layers // 4) zeroed on the mask domain (A24; the "
                                        f"plan's text: {kn['failing_input']!r})", "layers_zeroed": [0, last], "readout": name,
                                name: fail, "shift": shift, "min_shift_nats": kn["failing_min_shift_nats"],
                                "pass": shift < kn["failing_min_shift_nats"],
                                "layer0_only": {name: fail0, "shift": abs(fail0 - ours), "reported": True,
                                                "inside_tolerance": abs(fail0 - ours) <= tol}}
        out["known_block"] = "this plan" if self.stage == "arm0" else "the Arm 0 plan's known block (--known-plan, A26)"
        if self.stage != "arm0":
            out["truncated_windows_reading"] = {**self.known_reading(self.plan["known"], corpus), "reported": True,
                                                "rule": "this plan's known contexts, each window cut at its last context "
                                                        "(the pre-A26 gate; reported, not gating)"}
        return out

    def g_det(self):
        W, rep, n_cmp, fail = self.W, 0.0, 0, None
        for w in (0, 1):
            ids = self.ids(w * W, W)
            c1, c2 = self.fwd(ids, keep_logits=True), self.fwd(ids, keep_logits=True)
            rep, n_cmp = max(rep, self.dlogit(c1["logits"], c2["logits"])), n_cmp + c1["logits"].shape[1]
            if w == 1:
                j = len(self.prefix) + DET_FAIL_POS
                assert j < ids.shape[1] - 1, f"window {W} too short for the G-det failing input"
                bad = ids.clone()
                bad[0, j] = (bad[0, j] + 1) % self.model.config.get_text_config().vocab_size
                fail = self.dlogit(self.fwd(bad, keep_logits=True)["logits"][:, j:], c1["logits"][:, j:])
            del c1, c2
        return {"max_abs_dlogit_clean_repeat": rep, "positions_compared": n_cmp, "pass": rep == 0.0,
                "failing_input": {"rule": f"token {DET_FAIL_POS} of window 1 replaced by (id + 1) mod V",
                                  "max_abs_dlogit": fail, "pass": fail == 0.0}}

    def gates(self):
        """SPEC §9.1 item 2 order: G-det, G-ident, G-place, G-gen, G-known, G-floor-dir, G-floor-pos, G-pc (G-coh after
        the alpha = 0 samples, `coherence`)."""
        t0 = time.time()
        layers = sorted({p["layer"] for p in self.steered_poles()})
        probe = self.torch.tensor([self.prefix + self.tokens[:16].tolist()])
        placement = {str(l): self.R.placement_check(self.model, l, probe) for l in layers}
        bad = [l for l, x in placement.items() if not x["pass"]]
        assert not bad, f"steering hook is not after the post-MLP norm at layers {bad}: {placement[bad[0]]}"
        p0 = self.plan["poles"][0]
        spans = self.gate_spans()
        det = self.g_det()
        clean = [self.logits(ids) for ids, _ in spans]
        ident = self.g_identity(p0, spans, clean)
        place = self.g_place(spans[0])
        gen = self.g_gen()
        known = self.g_known()
        floor = self.g_floor(p0, spans, clean)
        del clean
        pc = self.g_pc()
        gin = {"natural_calibration_spans": [0, 1], "P": [P for _, P in spans], "bos": bool(self.prefix)}
        self.controls = {"identity": {**ident, "input": gin}, "known": known, "floor": {**floor, "input": gin},
                         "pc": pc, "place": place, "gen": gen, "determinism": det, "coherence": None,
                         "placement": {l: {k: x[k] for k in ("hook_module", "post_mlp_norm", "call_order", "last_child", "pass")}
                                       for l, x in placement.items()}}
        dump(self.out / "controls.json", self.controls)
        g = {"determinism": det["pass"] and not det["failing_input"]["pass"],
             "identity": ident["pass"] and not ident["failing_input"]["pass"],
             "place": all(r["pass"] and not r["failing_input"]["pass"] for r in place),
             "gen": all(r["pass"] and not r["failing_input"]["pass"] for r in gen),
             "known": known["pass"] and not known["failing_input"]["pass"],
             "floor_random_dir": floor["random_dir"]["pass"] and not floor["random_dir"]["failing_input"]["pass"],
             "floor_position": floor["position"]["pass"] and not floor["position"]["failing_input"]["pass"],
             "pc": pc["pass"] and not pc["failing_input"]["pass"], "coherence": None}
        append_jsonl(self.out / "samples" / "gates.jsonl", [
            {"gate": "determinism", "input": {"windows": [0, 1]}, "readout": det},
            {"gate": "identity", "input": {**gin, "text_span0": self.dec(spans[0][0][len(self.prefix):])}, "readout": ident},
            {"gate": "place", "input": {**gin, "natural_calibration_spans": [0]}, "readout": place},
            {"gate": "gen", "input": {"prompt": NEUTRAL_PROMPT_0, "rows": self.plan["gen_gate"]["rows"]}, "readout": gen},
            {"gate": "known", "input": {"pole": self.known["pole"],
                                        "windows": sorted({c["tok"] // self.W for c in self.known["contexts"]})},
             "readout": known},
            {"gate": "floor", "input": gin, "readout": floor},
            {"gate": "pc", "input": {"texts": pc["texts"]}, "readout": {k: v for k, v in pc.items() if k != "per_pole_mean_dnll"}}])
        self.m["gates"] = g
        self.stage_t("gates", t0)
        if not all(v for k, v in g.items() if k != "coherence"):
            self.m["status"] = "aborted_gate"
            self.save()
            raise SystemExit(f"[neuron_rw] gate failure: {json.dumps(finite(g))}")
        self.save()
        print("[neuron_rw] gates", json.dumps(finite(g)), flush=True)

    def coherence(self, alpha0: list, calibration: list, test_spans: list, att_t, pole: dict, label_rows=None):
        """G-coh (SPEC §5.2, §6, A7): the stage's alpha = 0 rows split into halves (neuron_rw_core.split_halves); band from
        the calibration half and the natural calibration spans; the registered inputs (written to
        samples/coherence_inputs.jsonl) classified; controls.json rewritten with `coherence`; never aborts (P8)."""
        R = self.R
        cal, test = R.split_halves(alpha0, self.plan["model_key"])
        rows = self.coherence_inputs(pole, test, att_t, test_spans, label_rows)
        by = {r["input"]: r for r in rows}
        gc = R.g_coh(R.coherence_band(cal, calibration),
                     {"repeated": by["failing_1_repeated_token"], "random": by["failing_2_random_attested"]}, test_spans, test,
                     label_rows)
        gc["split"] = {"n_calibration": len(cal), "n_test": len(test), "rule": self.plan["coherence"]["split_seed"]}
        self.controls["coherence"] = gc
        dump(self.out / "controls.json", self.controls)
        self.m["gates"]["coherence"] = gc["pass"]
        self.save()

    # ---------------------------------------------------------------------------------- generation (J10)
    def steer_sample(self, arm, pole, alpha, prompt, rec):
        """SCHEMA §3.5 samples/steer.jsonl line."""
        p = self.R.prompt_ids(self.tok, prompt)
        pr = self.logits(p, self.steer(pole, alpha, len(p) - 1))[0, -1].float().softmax(-1)
        v, i = pr.topk(5)
        shown = ("<bos>" if self.prefix else "") + self.tok.decode(p[len(self.prefix):])
        append_jsonl(self.out / "samples" / "steer.jsonl", [
            {"arm": arm, "pole": pole["pole"], "alpha": float(alpha), "alpha_abs": self.alpha_abs(pole["layer"], alpha),
             "prompt_shown": shown, "completion": rec["completion"], "nll_clean_mean": rec["nll_clean_mean"],
             "top5_next": [[self.dec([int(t)]), float(x)] for t, x in zip(i, v)]}])

    def grid(self, poles: list, doses: list, n0: int) -> list:
        n = self.S["n_samples"] if "n_samples" in self.S else self.plan["sweep"]["n_samples"]
        return [(None, 0.0, k) for k in range(n0)] + [(q, a, k) for q in poles for a in doses if a != 0 for k in range(n)]

    def run_prompt(self, arm: str, prompt_set, i: int, prompt: str, grid: list, path) -> list:
        recs = []
        for c, rows in enumerate(self.chunks(grid)):
            seed = self.seed(i, c)
            spec = [(None, None, 0.0) if q is None else (q["layer"], self.direction(q), self.alpha_abs(q["layer"], a))
                    for q, a, _ in rows]
            got = [self.record(arm, q, prompt_set, i, a, k, seed, r)
                   for (q, a, k), r in zip(rows, self.generate(prompt, spec, seed))]
            append_jsonl(path, got)
            recs += got
            pr = self.m["progress"]
            if "n_generate_calls_done" in pr:
                pr["n_generate_calls_done"] += 1
                pr["n_samples_written"] += len(got)
            self.save()
        if i == 0:
            poles = {q["pole"]: q for q, _, _ in grid if q is not None}
            for rec in recs:
                if rec["k"] == 0 and rec["pole"] is not None:
                    self.steer_sample(arm, poles[rec["pole"]], rec["alpha"], prompt, rec)
        return recs

    def n_calls(self, grid: list) -> int:
        return math.ceil(len(grid) / self.S["max_rows"])

    # ---------------------------------------------------------------------------------- stage arm0
    def arm0(self):
        """SPEC §9.1 item 2 (A9): the sweep first, then the Arm 1 throughput probe, projections, natural spans, firing
        check, judge0_pc firing tokens, repair; G-coh once the natural spans exist."""
        torch, R, plan = self.torch, self.R, self.plan
        pr = self.m["progress"]
        poles = plan["poles"][:SMOKE["projections"]] if self.smoke else plan["poles"]
        repair = plan["repair"][:SMOKE["repair"]] if self.smoke else plan["repair"]
        sweep = plan["sweep"]["poles"][:SMOKE["sweep"]] if self.smoke else plan["sweep"]["poles"]
        pr.update({"n_projection_poles": len(poles), "projections_done": 0, "repair_poles_planned": len(repair),
                   "repair_poles_done": 0, "sweep_poles_done": 0, "rms_done": True, "natural_spans_done": False,
                   "firing_check_done": False})
        t0 = time.time()
        sw, fs = plan["sweep"], self.out / "sweep" / "samples.jsonl"
        assert 0 not in sw["doses"], "alpha 0 is not a sweep dose (J10)"
        fs.write_text("")
        a0 = []
        for i, prompt in enumerate(plan["neutral_prompts"]):
            recs = self.run_prompt("sweep", None, i, prompt, self.grid(sweep, sw["doses"], sw["n_samples_alpha0"]), fs)
            a0 += [r for r in recs if r["pole"] is None]
        pr["sweep_poles_done"] = len(sweep)
        self.save()
        self.stage_t("sweep", t0)
        self.throughput_probe(sweep, sw["doses"])
        g = R.final_norm_gain(self.model)
        dump(self.out / "arm0" / "final_norm_gain.json", {"d_model": int(g.numel()), "gain": g.cpu().tolist(),
                                                          "rule": "final_norm(ones(d)) in fp32"})
        t0 = time.time()
        att = attested_ids(self.tokens, special_ids(self.tok))
        av = plan["attested_vocab"]
        assert (len(att), ids_sha256(att)) == (av["n_ids"], av["sha256"]), "attested vocabulary differs from the plan's"
        att_t = torch.from_numpy(att.astype(np.int64))
        f = self.out / "arm0" / "projections.jsonl"
        f.write_text("")
        first = []
        for s in range(0, len(poles), 50):
            part = poles[s:s + 50]
            vp = R.vocab_projection(self.model, self.tok, [(q["pole"], q["layer"], q["neuron"], q["sign"]) for q in part],
                                    att_t, plan["topn"])
            nxt = lambda q: (q["neuron"] + 1) % R.text_tower(self.model).layers[q["layer"]].mlp.down_proj.in_features  # noqa: E731
            vn = R.vocab_projection(self.model, self.tok, [(q["pole"], q["layer"], nxt(q), q["sign"]) for q in part],
                                    att_t, 10)                     # A24: G-proj's failing input, neuron i + 1, same layer and sign
            lines = [{"pole": q["pole"], "layer": q["layer"], "neuron": q["neuron"], "sign": q["sign"], **vp[q["pole"]],
                      "neighbour": {"neuron": nxt(q), "top_full": [x["str"] for x in vn[q["pole"]]["top_full"]]}} for q in part]
            append_jsonl(f, lines)
            first += lines[:max(0, 5 - len(first))]
            pr["projections_done"] += len(part)
            self.save()
        append_jsonl(self.out / "samples" / "projections.jsonl", first)
        self.stage_t("projections", t0)
        t0 = time.time()
        self.natural_spans()
        pr["natural_spans_done"] = True
        self.save()
        self.stage_t("natural_spans", t0)
        t0 = time.time()
        self.firing_check()
        pr["firing_check_done"] = True
        self.save()
        self.stage_t("firing_check", t0)
        self.firing_tokens()
        t0 = time.time()
        for q in repair:
            self.repair_pole(q)
            pr["repair_poles_done"] += 1
            self.save()
        self.stage_t("repair", t0)
        self.coherence(a0, self.cal_spans, self.test_spans, att_t, plan["sweep"]["poles"][0])

    def throughput_probe(self, sweep: list, doses: list):
        """SPEC §12 (A9): plan.throughput_probe.calls generate calls of plan.throughput_probe.rows rows on the Arm 1 prompt,
        rows steered at the sweep poles x sweep doses in turn (the per-row dose path of Arms 1 and 2), seed base_seed +
        1000 * prompt_i + c; generation time only. Also the mean Arm 1 prompt length in tokens (BOS included) over the
        prompts of plan.prompt_sets. Completions are not banked (a timing readout)."""
        pb = self.plan["throughput_probe"]
        t = self.probe
        if pb is None:
            return
        t0 = time.time()
        cyc = [(q, a) for q in sweep for a in doses]
        rows = [(cyc[k % len(cyc)][0]["layer"], self.direction(cyc[k % len(cyc)][0]),
                 self.alpha_abs(cyc[k % len(cyc)][0]["layer"], cyc[k % len(cyc)][1])) for k in range(pb["rows"])]
        n = 0
        for c in range(pb["calls"]):
            self.R.generate_rows(self.model, self.tok, pb["prompt"], rows, self.seed(pb["prompt_i"], c), self.S)
            self.sync()
            n += len(rows)
        t["gen_rows_per_s_arm1probe"] = n / (time.time() - t0)
        lens = [len(self.R.prompt_ids(self.tok, x["prompt"])) for ps in self.plan["prompt_sets"].values() for x in ps]
        t["arm1_mean_prompt_tokens"] = float(np.mean(lens)) if lens else None
        self.stage_t("throughput_probe", t0)
        self.save()

    def repair_pole(self, q: dict):
        """SPEC §2.3 (A6), SCHEMA §3.3: per window one clean pass (neuron_rw_core.repair_readout), then per context the
        t-only total effect: the same window input with neuron_off_core.PoleAblator (mode pole, exclude_first = the
        context's position, so position t is the first edited and nothing after it reaches t's logits), read at the same
        positions; at k = 0 of each pole the same forward with the identity hook (G-repair-fit, J16)."""
        R, C = self.R, self.C
        out, keep = [], {}
        kidx = {c["rank"]: k for k, c in enumerate(q["contexts"])}
        for cs, ids, pos in self.by_window(q["contexts"]):
            tgt = [c["target_id"] for c in cs]
            rr = self.timed("forward", ids.numel(), lambda: R.repair_readout(self.model, ids, pos, tgt, q["layer"], q["neuron"]))
            for j, (c, t_pos, r) in enumerate(zip(cs, pos, rr["rows"])):
                k = kidx[c["rank"]]
                abl = C.PoleAblator(self.model, "pole", q["layer"], q["neuron"], sstr(q["sign"]), exclude_first=t_pos)
                ablated = float(self.fwd(ids, abl, positions=pos, targets=tgt)["nll"][0, j])
                ident = None
                if k == 0:
                    idn = C.PoleAblator(self.model, "identity", q["layer"], q["neuron"], sstr(q["sign"]), exclude_first=t_pos)
                    li = self.fwd(ids, idn, keep_logits=True, positions=pos, targets=tgt)["logits"]
                    ident = float((li[0, j].float() - rr["logits"][j].float()).abs().max())
                out.append({"k": k, "rank": c["rank"], "tok": c["tok"], "target_id": c["target_id"], "a_ours": r["a_ours"],
                            "a_off": c["a_off"], "clean_nll_ours": r["clean_nll_ours"], "clean_nll_off": c["clean_nll_off"],
                            **{x: r[x] for x in ("clean_nll_manual", "direct_dnll", "direct_dnll_postrenorm",
                                                 "direct_dnll_finalrenorm")},
                            "tonly_clean_nll": r["clean_nll_ours"], "tonly_ablated_nll": ablated,
                            "tonly_dnll": ablated - r["clean_nll_ours"], "tonly_identity_max_abs_dlogit": ident,
                            "dnll_off": c["dnll_off"], "delta_r_norm": r["delta_r_norm"], "final_rms": r["final_rms"],
                            "clean_nll_manual_early": r["clean_nll_manual_early"],
                            "clean_nll_manual_model_dtype": r["clean_nll_manual_model_dtype"],
                            "clean_nll_manual_model_dtype_wrong_eps": r["clean_nll_manual_model_dtype_wrong_eps"]})
                keep[c["rank"]] = ids[0, :t_pos + 1].tolist()
        out.sort(key=lambda x: x["k"])
        fi = {"layer": R.early_layer(self.model),            # G-repair-fit failing input: its shift banked per pole
              "median_abs_shift_nats": float(np.median([abs(x["clean_nll_manual_early"] - x["clean_nll_ours"]) for x in out])),
              "manual_median_abs_nats": float(np.median([abs(x["clean_nll_manual"] - x["clean_nll_ours"]) for x in out]))}
        dump(self.out / "arm0" / "repair" / f"{stem(q['pole'])}.json",
             {"pole": q["pole"], "layer": q["layer"], "neuron": q["neuron"], "sign": q["sign"],
              "w_down_norm": float(R.w_down_column(self.model, q["layer"], q["neuron"]).norm()), "failing_input": fi,
              "contexts": out})
        self.controls.setdefault("repair_failing_input", []).append({"pole": q["pole"], **fi})
        dump(self.out / "controls.json", self.controls)
        hi, lo = max(out, key=lambda x: x["dnll_off"]), min(out, key=lambda x: x["dnll_off"])
        append_jsonl(self.out / "samples" / "repair.jsonl", [
            {"pole": q["pole"], "which": which, "text_before": self.dec(keep[x["rank"]][-N_BEFORE:]),
             "target_str": self.dec([x["target_id"]]), **x} for which, x in (("largest_dnll_off", hi), ("smallest_dnll_off", lo))])

    def natural_spans(self):
        """J12: SCHEMA §3.3 lines; keeps both sets for G-coh."""
        ns = self.plan["natural_spans"]
        f = self.out / "arm0" / "natural_spans.jsonl"
        f.write_text("")
        self.cal_spans = self.span_nlls(ns["calibration_windows"], "calibration")
        self.test_spans = self.span_nlls(ns["test_windows"], "test")
        append_jsonl(f, self.cal_spans + self.test_spans)

    def span_nlls(self, windows: list, which: str) -> list:
        ns, rows = self.plan["natural_spans"], []
        for w in windows:
            for j in range(ns["spans_per_window"]):
                s = self.span(w, j)
                nll = self.tnll(self.prefix + s["prompt_ids"] + s["cont_ids"], len(self.prefix) + len(s["prompt_ids"]))
                rows.append({"set": which, "window": w, "span": j, "nll_clean_mean": float(nll.double().mean()),
                             "n_new_tokens": len(s["cont_ids"])})
        return rows

    def firing_check(self):
        """J11 / SPEC §5.5: SCHEMA §3.3 arm0/firing.jsonl."""
        R, plan = self.R, self.plan
        rule = plan["firing_rule"]
        f = self.out / "arm0" / "firing.jsonl"
        f.write_text("")
        for e in plan["firing_check"]:
            rows = []
            for pr in plan["prompt_sets"][e["prompt_set"]]:
                p_txt = self.tok(pr["prompt"], add_special_tokens=False).input_ids
                f_txt = self.tok(pr["prompt"] + pr["continuation"], add_special_tokens=False).input_ids
                ids = R.prompt_ids(self.tok, pr["prompt"] + pr["continuation"])
                mx, pos = self.timed("forward", len(ids), lambda: R.firing_max(self.model, ids, e["layer"], e["neuron"],
                                                                                e["sign"], 1))
                frac = mx / e["banked_max_abs"]
                rows.append({"pole": e["pole"], "prompt_set": e["prompt_set"], "i": pr["i"],
                             "prefix_property": len(f_txt) > len(p_txt) and f_txt[:len(p_txt)] == p_txt,
                             "max_sign_a": mx, "argmax_pos": pos, "argmax_token": self.dec([ids[pos]]),
                             "frac_of_banked_max": frac, "fires": frac >= rule["threshold_frac"]})
            append_jsonl(f, rows)

    def firing_tokens(self):
        """J15 (A20): per judge0_pc pole, firing_set over the decoded strings of every top_tok position, decoded with the
        plan's tokenizer file; the count must equal the plan's n_distinct and be at least N_FIRING_MIN."""
        rows = []
        for e in self.plan["judge0_pc"]:
            strings = [self.firing_tok.decode([int(self.tokens[t])], skip_special_tokens=False) for t in e["top_tok"]]
            out, n = firing_set(strings)
            assert n == e["n_distinct"] and n >= N_FIRING_MIN and len(out) == min(n, N_FIRING_TOKENS), \
                f"{e['pole']}: {n} distinct firing tokens in {len(e['top_tok'])} positions, plan {e['n_distinct']}"
            rows.append({"pole": e["pole"], "tokens": out, "n_ranks_read": len(e["top_tok"]), "n_distinct": n})
        f = self.out / "arm0" / "firing_tokens.jsonl"
        f.write_text("")
        append_jsonl(f, rows)

    def coherence_inputs(self, pole: dict, a0_test: list, att_t, test_spans: list, label_rows=None) -> list:
        """J13."""
        R = self.R
        if "neutral_prompts" in self.plan:
            assert self.plan["neutral_prompts"][0] == NEUTRAL_PROMPT_0, "neutral prompt 0 differs from SPEC §4.7"
        p = R.prompt_ids(self.tok, NEUTRAL_PROMPT_0)
        top1 = R.vocab_projection(self.model, self.tok, [(pole["pole"], pole["layer"], pole["neuron"], pole["sign"])],
                                  att_t, 1)[pole["pole"]]["top"][0]
        rng = np.random.default_rng([SEED, MODEL_INDEX[self.plan["model_key"]], 10])
        rand = rng.choice(att_t.numpy(), COH_LEN).tolist()
        rows = []
        for name, cont, extra in (("failing_1_repeated_token", [top1["id"]] * COH_LEN, {"pole": pole["pole"], "token": top1}),
                                  ("failing_2_random_attested", rand, {"rng": f"default_rng([{SEED}, model_index, 10])"})):
            rows.append({"input": name, "role": "failing", "prompt": NEUTRAL_PROMPT_0, "cont_ids": cont,
                         "cont_text": self.tok.decode(cont), "n_new_tokens": len(cont),
                         "nll_clean_mean": float(self.tnll(p + cont, len(p)).double().mean()), **extra})
        rows.append({"input": "passing_1_natural_test_spans", "role": "passing", "n": len(test_spans),
                     "nll_clean_mean": [x["nll_clean_mean"] for x in test_spans],
                     "n_new_tokens": [x["n_new_tokens"] for x in test_spans]})
        for name, rr in (("passing_2_alpha0_test_half", a0_test), ("passing_3_label_prompted", label_rows)):
            if rr is not None:
                rows.append({"input": name, "role": "passing", "n": len(rr), "nll_clean_mean": [x["nll_clean_mean"] for x in rr],
                             "n_new_tokens": [x["n_new_tokens"] for x in rr]})
        append_jsonl(self.out / "samples" / "coherence_inputs.jsonl", rows)
        return rows

    def stage_coherence(self, a0: list, label_rows=None):
        """J13 for arm1 and arm2: attested ids and natural calibration and test spans recomputed in the stage."""
        att = attested_ids(self.tokens, special_ids(self.tok))
        ns = self.plan["natural_spans"]
        cal, test = self.span_nlls(ns["calibration_windows"], "calibration"), self.span_nlls(ns["test_windows"], "test")
        self.coherence(a0, cal, test, self.torch.from_numpy(att.astype(np.int64)), self.plan["poles"][0], label_rows)

    # ---------------------------------------------------------------------------------- stages arm1, arm2
    def arm1(self):
        plan, pr = self.plan, self.m["progress"]
        sets, by_set = plan["prompt_sets"], {}
        for q in plan["poles"]:
            assert q["prompt_set"] in sets, f"{q['pole']}: prompt set {q['prompt_set']} not in the plan"
            by_set.setdefault(q["prompt_set"], []).append(q)
        assert 0 in plan["doses"], "arm1 doses must include 0 (the shared alpha = 0 rows)"
        n0 = self.S["n_samples"]
        pr.update({"n_generate_calls_planned": sum(len(sets[s]) * self.n_calls(self.grid(qs, plan["doses"], n0))
                                                   for s, qs in by_set.items()),
                   "n_generate_calls_done": 0, "n_samples_written": 0, "poles_done": []})
        t0, a0 = time.time(), []
        for s, qs in by_set.items():
            path = self.out / "arm1" / "samples" / f"{s}.jsonl"
            path.write_text("")
            for p in sets[s]:
                recs = self.run_prompt("arm1", s, p["i"], p["prompt"], self.grid(qs, plan["doses"], n0), path)
                a0 += [r for r in recs if r["pole"] is None]
            pr["poles_done"] += [q["pole"] for q in qs]
            self.save()
        self.stage_t("arm1", t0)
        self.stage_coherence(a0)

    def arm2(self):
        plan, pr = self.plan, self.m["progress"]
        n0, prompts = self.S["n_samples"], plan["neutral_prompts"]
        grid = self.grid(plan["poles"], plan["doses"], n0)
        jpc = plan["label_prompts"]
        pr.update({"n_generate_calls_planned": len(prompts) * self.n_calls(grid) + len(jpc) * math.ceil(n0 / self.S["max_rows"]),
                   "n_generate_calls_done": 0, "n_samples_written": 0, "poles_done": []})
        path = self.out / "arm2" / "samples.jsonl"
        path.write_text("")
        t0, a0 = time.time(), []
        for i, prompt in enumerate(prompts):
            a0 += [r for r in self.run_prompt("arm2", None, i, prompt, grid, path) if r["pole"] is None]
        pr["poles_done"] = [q["pole"] for q in plan["poles"]]
        self.save()
        labelled = []
        for e_i, e in enumerate(plan["label_prompts"]):                     # J14
            for c, rows in enumerate(self.chunks([(None, None, 0.0)] * n0)):
                seed = self.seed(100 + e_i, c)
                got = [self.record("arm2", e, None, None, 0.0, c * self.S["max_rows"] + k, seed, r, "label_prompt") | {"prompt": e["prompt"]}
                       for k, r in enumerate(self.generate(e["prompt"], rows, seed))]
                append_jsonl(path, got)
                labelled += got
                pr["n_generate_calls_done"] += 1
                pr["n_samples_written"] += len(got)
                self.save()
        self.stage_t("arm2", t0)
        self.stage_coherence(a0, labelled)

    # ---------------------------------------------------------------------------------- run
    def run(self):
        dump(self.out / "manifest.json", self.manifest)
        self.save()
        self.measure_rms()
        self.gates()
        {"arm0": self.arm0, "arm1": self.arm1, "arm2": self.arm2}[self.stage]()
        self.m["status"] = "complete"
        self.save()
        self.manifest["finished_utc"] = self.utc()
        dump(self.out / "manifest.json", self.manifest)
        return self.m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(PLAN_SCHEMA))
    ap.add_argument("--model-key", required=True, choices=["Q4B", "G4", "O7"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokname", required=True, choices=["olmo3", "qwen3", "gemma3"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--rms", default=None, help="arm1, arm2: arm0's out/arm0/rms.json for this model (J1)")
    ap.add_argument("--known-plan", default=None, help="arm1, arm2: the model's Arm 0 plan; G-known reads its known block (A26)")
    ap.add_argument("--revision", default=None, help="Hub branch to download (O7); absent = the cached snapshot")
    ap.add_argument("--smoke", action="store_true", help="preflight: arm0 with the SCHEMA §2 subset and every gate")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"           # before CUDA starts; deterministic cuBLAS

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    import transformers
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    from infra.pg.matrix_models import vocab_sha256
    assert torch.cuda.is_available(), "no CUDA on this instance"

    raw = fetch(a.plan, "plan.json").read_bytes()
    plan = json.loads(raw)
    assert (plan["model_key"], plan["repo"], plan["branch"], plan["tokname"], plan["corpus"]) == \
        (a.model_key, a.repo, a.revision, a.tokname, a.corpus), "plan and arguments name different models or corpora"
    meta = {"model_key": a.model_key, "repo": a.repo, "branch": a.revision, "device_name": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__, "torch": torch.__version__, "pg_git_sha": os.environ.get("PG_GIT_SHA")}
    rms = None
    if a.rms:
        rraw = fetch(a.rms, "rms.json").read_bytes()
        rms = json.loads(rraw)
        meta["rms_source"] = {"path": a.rms, "sha256": hashlib.sha256(rraw).hexdigest(), "plan": plan.get("rms_source")}
    known_arm0 = None
    assert (a.known_plan is None) == (a.stage == "arm0"), "arm1 and arm2 take --known-plan (A26); arm0 does not"
    if a.known_plan:
        kraw = fetch(a.known_plan, "known_plan.json").read_bytes()
        kp = json.loads(kraw)
        assert (kp["schema"], kp["model_key"]) == (PLAN_SCHEMA["arm0"], a.model_key), "--known-plan is not this model's Arm 0 plan"
        known_arm0 = kp["known"]
        meta["known_plan"] = {"path": a.known_plan, "sha256": hashlib.sha256(kraw).hexdigest()}
    if a.revision:
        local, sha, meta["download"] = download(a.repo, a.revision)
    else:
        local, sha = NC.local_snapshot(a.repo)
    assert sha == plan["revision_expected"], f"{a.repo}: snapshot {sha}, plan expects {plan['revision_expected']}"
    meta["revision"] = sha
    corpus = NC.load_tok(fetch_corpus(a.corpus, a.tokname), a.tokname)
    assert corpus["tokens_sha256"] == plan["corpus_tokens_sha256"], "corpus tokens differ from the plan's"
    meta["corpus_tokens_sha256"] = corpus["tokens_sha256"]
    t0 = time.time()
    model, tok = NH.load_model(local, a.model_key)
    meta["load_s"] = round(time.time() - t0, 1)
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "model and corpus tokenizer vocabularies differ"
    from tokenizers import Tokenizer
    tk_path = pathlib.Path(corpus["tokenizer_dir"]) / "tokenizer.json"          # A20: the file the plan counted with
    m = Job(a.out, stage=a.stage, model=model, tok=tok, tokens=corpus["tokens"], plan=plan,
            plan_sha256=hashlib.sha256(raw).hexdigest(), meta=meta, rms_banked=rms, smoke=a.smoke,
            firing_tok=Tokenizer.from_file(str(tk_path)), firing_tok_sha256=hashlib.sha256(tk_path.read_bytes()).hexdigest(),
            known_arm0=known_arm0).run()
    print("NEURON_RW_OK", json.dumps({"model_key": a.model_key, "stage": a.stage, "status": m["status"],
                                      "progress": m["progress"]}), flush=True)


if __name__ == "__main__":
    main()
