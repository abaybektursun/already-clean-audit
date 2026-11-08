# MENDELEEV build contract (2026-09-03) — every NEW module binds to this, nothing else

> **Instrument note (2026-09-03).** Blank-image MMLU numbers: the readout scores an off-format token
> surface in the VLM shell, but re-measurement on both surfaces gives identical accuracies, so nothing is
> retracted; the shell itself costs the healthy model ~0.16 accuracy vs text-only, which is the caveat that
> matters for interpretation. Crosscoder "features preserved" statements on the gemma-2→PaliGemma-2 pair
> are VOID (C6, FVU 8.6). See `studies/lineage/2026-09-03_INSTRUMENT_INCIDENT.md` and
> `studies/lineage/CLAIMS_LEDGER.md`.

Spec: `studies/lineage/cross-lineage/research/2026-09-02_mendeleev_spec.md` (read §1.2, §2, §5, §6, §10–§12
before writing a line). Rules of the house: fail loud (no try/except that hides), simplest code that
works, no commits (the lead commits by file name), no cloud launches unless your brief says so, never
run models locally (dry-runs are weight-free: stubs, tiny tensors). Every module has `_dryrun()` and
`python -m <module>` runs it. Pinned env on the box: torch 2.12.0, transformers 5.9.0, peft 0.19.1,
datasets 4.8.5, accelerate 1.13.0, numpy, scipy 1.17, sklearn 1.8. Jobs run under
`infra/aws_pg/run_job.sh` (HF cache at `/opt/hf-cache/hf`, `HF_HUB_OFFLINE=1` by default) — every job
calls `diff_harvest.hf_online_setup()` FIRST (before importing transformers/datasets) and sets
`HF_HUB_DISABLE_XET=1`; models not in the S3 cache are downloaded from HF in-job.

## 1. File map (who writes what)
| file | item | owner |
|---|---|---|
| `infra/pg/matrix_items.py` | item sets + hashes (B5) | agent READOUTS |
| `infra/pg/format_readouts.py` | N2: S (k-shot), F1–F5, H, K-shuffle-null, A, TQ, tokenizer gate | agent READOUTS |
| `infra/pg/two_hop_items.py` | N4: U builder + shortcut control | agent READOUTS |
| `infra/pg/matrix_forward.py` | `Forward` shells (text / PaliGemma / Molmo2 / π0-in-PG1) | agent MODELS |
| `infra/pg/matrix_models.py` | row registry, `load_pair`, binds B1/B3/B4 (hashes, tokenizer identity, LM-delta) | agent MODELS |
| `infra/pg/probe_readout.py` | N3: harvest + ridge-logistic probe + shuffle + random-init bound | agent MODELS |
| `infra/pg/util_probe.py` | N13: batch-doubling probe + `nvidia-smi dmon` sampler | agent MODELS |
| `infra/aws_pg/jobs/matrix_behaviour.py` | N7: all behaviour columns for one row | agent MODELS |
| `infra/pg/family_resolvers.py` | N5: decoder-layer lists, hook names, key remaps per family | agent GRAFT |
| `infra/pg/cached_donor.py` | N6: per-layer donor cache + `PatchedLM` from cache | agent GRAFT |
| `infra/aws_pg/jobs/matrix_graft.py` | N9: G (R_G, D_G, cuts) + W (transplant) for one row | agent GRAFT |
| `infra/aws_pg/jobs/matrix_recovery.py` | N8: REC/scratch/healthy/REH/permuted/GB/SUB/IM for one (row, recipe) | agent RECOVERY |
| `infra/pg/vlm_skill_eval.py` | N14: RefCOCO/VQAv2/TextVQA via each model's processor | agent ROW3 |
| `infra/aws_pg/jobs/molmoact2_libero.py` | N15: LIBERO for MolmoAct2 (gated) | agent ROW3 |
| `infra/vast/manifests/mdl_*.json`, `infra/vast/cells/mdl_*.py`, image, IAM policy, net probe | N11 | agent INFRA |
| `infra/aws_pg/pg.py` `--weights`, `run_job.sh` selective sync + dmon heartbeat, `config.INSTANCE_TYPES`, N16 | harness | agent INFRA |
| `infra/pg/matrix_core.py`, `scripts/matrix_verdicts.py` | rules + offline verdicts | the lead |

## 2. Items (`matrix_items.py`) — every list content-hashed; hash recipe fixed
- `item_hash(question) = sha1(question[:64].encode()).hexdigest()[:12]` (identical to `text_eval`).
- `mmlu_eval(half)`; half ∈ {"discovery" (test rows[:5] per subject, 285), "confirmation" (rows[5:10], 285)};
  returns list of `{subject, question, choices[4], answer:int, hash}` in `text_eval` order (sorted subject,
  row order). `mmlu_dev_shots(subject, n)` = first n dev rows (5-shot prefix exactly as `text_eval._q_block`).
- `held_out_subjects()` = 14 subjects, `random.Random(1337).sample(sorted(all_subjects), 14)`.
- `mmlu_train(n, seed, exclude_hashes)` = `auxiliary_train` + dev, minus held-out subjects, minus any
  hash in `exclude_hashes` (ALL eval hashes, both halves, asserted disjoint), first n after
  `ladder_subset`-style deterministic shuffle (`relearn.ladder_subset` semantics, salt "mdl").
- `rfp_items()` = the frozen `read_from_passage` set (call its builder with its frozen seed).
- `wmdp_items(split)`: `cais/wmdp` bio + cyber; per-config 60/40 split with `random.Random(1337)`;
  split ∈ {"train", "eval"}.
- `arith_items()` = 200 GSM-style two-step word problems from a fixed template family, integers only,
  `random.Random(1337)`; each `{question, answer:str, hash}`; also a 4-option version (gold + 3 distractors
  = gold ± small offsets) for the candidate-likelihood readout.
- `triviaqa_items()` = 500 items from `mandarjoshi/trivia_qa` `rc.nocontext` validation (first 500 by
  a `random.Random(1337)` permutation) with 4 options = gold + 3 other items' golds (fixed).
- `vision_index_lists()` = RefCOCO testA / VQAv2 val / TextVQA val, n=200 each: REUSE `battery_tasks`'s
  fixed lists; `vqav2_train(n, seed)` for REH (hash-disjoint from the eval list, asserted).
- `manifest()` returns `{name: {n, sha256 of the json-serialised list}}` for every set above.

## 3. Forward shells (`matrix_forward.py`)
```python
class Forward:                       # one per (model, shell); ALL readouts go through this
    n_layers: int; family: str; shell: str   # shell ∈ {"text","paligemma","molmo2","pi0-pg1"}
    def next_token_logits(self, prompts: list[str]) -> Tensor[B, V]     # left-pad; last real token; logits_to_keep=1
    def continuation_logprobs(self, prompts, continuations) -> list[(sum_logprob, n_tok)]  # text_eval.mmlu_cloze semantics
    def generate(self, prompts, max_new_tokens) -> list[str]           # greedy; VLM shells via _blank_image_generate
    def letter_ids(self, letters=("A","B","C","D"), space=True) -> list[int]  # asserts single-token
    def batch_size: int                                                # set by util_probe at job start (graft: fixed 4)
```
`TextForward(model, tok)`, `PaliGemmaForward(model, processor, gray=128)` (blank 224² gray image, exactly
`mmlu_blankimg`'s call: input_ids/attention_mask/pixel_values only), `Molmo2Forward` (its processor,
blank image at its native size), `Pi0Forward` (π0 backbone remapped into the PG1 shell via
`pi0_port.remap_backbone_lm`; own tied head primary, PG1 head secondary — both reported).
**Parity rule (§12.10):** `S` computed through `Forward.next_token_logits` must reproduce
`text_eval.mmlu_blankimg`'s `per_item` on row 2 (rows[:5]) at ≥ 99 % item agreement and |Δacc| ≤ 0.005.

## 4. Readouts (`format_readouts.py`) — each returns the SAME record shape
```python
{"col": "F2", "acc": float, "n": int, "floor": float, "per_item": [0/1], "per_item_hash": [str],
 "per_item_scores": [[...4 floats]] or None, "pred": [int], "extra": {...}}
```
- `S(fwd, items, shots: int)` → letter logits at " A".." D" after "Answer:"; prompt = `text_eval._subject_header`
  + k `_q_block` shots + `_q_block(item)`; k ∈ {0,1,5} (S-k). `extra` = `{"p_any_letter": mean prob mass
  on the 4 letters over the full vocab, "argmax_in_letters_frac"}` — computed for BOTH models (D14).
- `F1` permuted letters: fixed derangement of option ORDER per item, `random.Random(1337)`; letters A–D
  unchanged; gold tracked. `F2` numbered "1. .. 4." answers " 1".." 4". `F3` verification: per item 4
  prompts "Question … Proposed answer: <opt>. Is the proposed answer correct? Answer (Yes/No):";
  balanced accuracy over (tp, fn, tn, fp); floor 0.5. `F4` "(A)"-style labels, readout on the letter
  token after "(". `F5` answer-word: options shown with letters, score the 4 option TEXTS after
  "Answer:" by per-token-mean log-likelihood (continuation_logprobs), argmax; floor 0.25.
- `H(pred)` → `matrix_core.letter_entropy`; report beside the key's entropy.
- `K(fwd, items, shots)` = call `text_eval.mmlu_cloze` semantics THROUGH `fwd.continuation_logprobs`
  (choices hidden; per-token mean; 4-way argmax); `K_shuffle_null(fwd, items, shots, seed=1337)` = the
  four option texts of item i replaced by the option texts of item π(i) (fixed derangement), gold index
  kept; must land ≤ 0.30 (D6).
- `RFP(fwd, items)` = wrap `read_from_passage.passage_cloze` (text) / its shell path.
- `A(fwd, items)` → `{"em": exact match via generate (max_new_tokens 8, first integer in output),
  "cand": 4-option candidate-likelihood acc}`; report both; `em` is the column, `cand` beside it.
- `TQ(fwd, items)` → 4-way option-text scoring like K; floor 0.25.
- `tokenizer_gate(fwd_before, fwd_after)` → single-token letters, `argmax_in_letters_frac` for both,
  `p_any_letter` for both → `matrix_core.tokenizer_gate(...)` (v3 signature takes both models).
- `ppl(fwd)` = `text_eval.clean_text_perplexity` through the shell's model.

## 5. Behaviour job (`matrix_behaviour.py`) — one row, both models, both halves
`--row <id> --cols S,S-k,F1,F2,F3,F4,F5,H,K,K-null,RFP,U,ppl,A,TQ,P --halves discovery,confirmation --out DIR`
Stages in order, each appended to `out/metrics.json` on completion (incremental, `json.dump` to a temp
file then `os.replace`): `manifest` (repos, config sha256, safetensors index sha256, tokenizer sha256,
item manifest, env pins, git SHA, instance type) → `binds` (B1/B3/B4: tokenizer identity, shared-vocab
hash, per-layer LM delta ‖ΔW‖/‖W‖) → `gates` (tokenizer gate; text-only-vs-shell parity for S AND K on the
healthy LM, |Δ| ≤ 0.05 — D15) → `throughput` (N13) → each column for `before` then `after`, per half →
`lost` (`matrix_core.margin_lost` + `bootstrap_fraction` per column per half). Per-item vectors always
saved. `--smoke` = 1 item/subject, `n_boot` 10. Exit 0 only when every requested column is present.
Artifacts: `out/metrics.json`, `out/per_item/<col>_<model>_<half>.json`, `out/manifest.json`,
`out/events.jsonl` (stage, t, gpu_util), `out/controls.json`.

## 6. Graft job (`matrix_graft.py`) — one row
`--row <id> --half discovery --pass 1 --stride 2` then `--pass 2 --layers 14,15,16,...` (bracketed) `--half confirmation`.
Order: identity gate (self-graft 3 layers, item match 100 %, max|Δlogit| ≤ 1e-2, batch 4) → positive hook
test (last-layer prefix graft; wiring pass ⟺ donor-residual + DONOR norm/head reproduces donor logits to
1e-2 per item; recipient-head reading reported as `R_G(last)`, low value labelled "head-stage damage",
never VOID — D2) → per-layer hook-name assertion (resolver names must match `named_modules`) → unpatched
baselines (both halves) → ≥ 5 mismatched-item null cuts → pass-1 R_G(L) stride 2 → D_G(L) converse →
extra cuts on the best layer (final-position-only; top-k directions k ∈ {8,64,512} of healthy−broken
residual difference) → W transplant configs via `carrel_core` (band, band+tail, singles, attn-only,
MLP-only, final norm, 2 outside-band controls, prefix control; restore-after-each exact; sanity none/all
reproduce baselines per item; |R_prefix(L) − R_G(L)| ≤ 0.05 mandatory — D17). Cached-donor mode
`--cached-donor` (N6) for rows where two copies do not fit; equivalence gate on row 2 (`--equivalence`).
Artifacts: per layer/config per-item correctness `out/per_item/G_L<L>_<half>.json`, `metrics.json` with
curves, `controls.json` (identity, hook, nulls, T = `matrix_core.threshold_T(null_spread)`).

## 7. Recovery job (`matrix_recovery.py`) — one (row, recipe, arm)
`--row 2 --recipe r512|r64|full --arm rec|scratch|healthy|reh|permuted --doses 250,1000,4000,16000 --out DIR`
Recipe constants ONLY from `configs/matrix_recipes.json` (hash recorded). Train items from
`matrix_items.mmlu_train` (held-out subjects excluded; eval-hash-disjoint asserted). Per dose: train
(fresh adapter / fresh copy per dose, seed 0), then evaluate S (5-shot), K, O on BOTH halves + the
held-out subjects; `dose 0` evaluation FIRST and asserted equal per item to the behaviour cell's `after`
vectors (D18); answer-token CE logged per step; adapters saved (`out/adapters/<dose>/`). `reh` = 50/50
MMLU + VQAv2-train mix (VLM rows). `permuted` = `relearn.shuffle_labels` (H-2 recipe) at 4k only.
Sub-modes: `--gb` (graft-back at the first dose with O_lost ≥ 0.3; donor = damaged model; sweep layers;
≥ 5 mismatched cuts on O for its own null — D12), `--sub` (top-k left-singular subspace of the install
ΔW vs ΔW_rec per layer; nulls: Haar AND an unrelated adapter on the same base — D9), `--im <task>`
(one r512 run on a task's train set at 4k; all IM tasks measured before/after). Full-FT path =
`helmholtz2_arm.train_text_ce`-style AdamW with exact fp32 Adam (CPU-offload flag). LoRA path =
`relearn.make_adapter/train_adapter_qa/attach_adapters` with `alpha = r`.
Artifacts: `metrics.json` per dose (`S_rec`, `K_rec`, `O_lost` with per-item vectors), `train_curve.json`.

## 8. Resolvers (`family_resolvers.py`)
`resolve(model) -> Resolver(layers: list[nn.Module], layer_names: list[str], final_norm, lm_head,
embed, family, n_layers, key_remap(donor_key)->recipient_key, lm_prefix)` for: gemma-1/2/3/4 (text +
PaliGemma-1/2 shells: `model.language_model.model.layers` etc.; verify against `named_modules` on THIS
transformers 5.9.0), Qwen3, Molmo2 (text tower only; vocab prefix 151 936 rows; the action expert and
vision modules excluded), Llama-3, Mistral, DeepSeek-V4 (`hc_mult=4`: `layers[L]` output is the 4-stream
tuple; the graft replaces all four — document the actual output structure from the modeling file before
claiming it). Every resolver: `assert len(layers) == config.num_hidden_layers`.

## 9. Placement + launch (INFRA)
AWS: `pg run <module> --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/mdl/<session>/<cell>/<attempt>
--budget-min N --weights models--google--gemma-2-2b,models--google--paligemma2-3b-pt-224` (selective S3
sync via `--exclude '*' --include`; absent models download from HF in-job). Vast: manifests
`mdl_<cell>.json` with mandatory `watch`, `weights_mode: download`, cells calling the SAME job modules
(`cells/mdl_job.py` shim: `run(cell, out_dir)` → `module.main(argv)`), image = pinned deps + repo code, no
weights. `config.INSTANCE_TYPES` on this branch = `["g6e.2xlarge", "g6e.4xlarge"]`.

## 10. Gates every module ships with
`_dryrun()` weight-free (stub model with a `Forward`-shaped fake); `py_compile`; import smoke without
CUDA; and for jobs a `--smoke` path that a real GPU cell runs first (1 item/subject, n_boot 10).
