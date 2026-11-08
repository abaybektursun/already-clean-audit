# N13 — What RL wrote: pre-registration draft

> **Amendment, 2026-11-05.** No model was run. The facts are from `studies/next/07_PICKS/N13_MATH_TASK.md`.
> Line numbers below are those of the draft before this note was added.
>
> 1. **G-KNOWN.** Table 3's MATH 40.0 is Minerva MATH (the 5,000-problem MATH test, olmes
>    `minerva_math::olmes:n4:v2`), not MATH-500. G-KNOWN now runs that task on base `main` with vLLM 0.11.0 and
>    requires the suite `primary_score` within 1.6 points of 40.0 (three times the sampling standard error of
>    the difference of two readings, 0.53). MATH-500 stays the readout for every arm, under our registered
>    harness. Draft lines 65, 78, 82, 221, 244, 462, 505, 518–519, 542, 552 (paragraphs added after it), 645,
>    692, 761, 795 (a sentence added after it), 805–806, 809–813, 875–886, 956–957, 981–982.
> 2. **Stage pair.** Pair 1 is now base `main` → `Olmo-3.1-7B-RL-Zero-Math` `main`; the 3.0 pair becomes the
>    fallback pair 1b (new §3.8, which states the reason). The published 3.1 MATH-500 band, 0.874 to 0.908, is
>    used only as a sanity band for the identity transplant (new check S-BAND, [81.5, 96.7]), not as a
>    reproduction target, because its prompt is not public. G-INIT for pair 1 is the hash check against the
>    logged init `39ff1a89…` plus a share comparison. The 3.1 template lacks the word "math"; §2.2 gives its
>    bytes. The budget is recomputed for longer 3.1 answers (to verify at run time) and for the 20,000
>    generations of R0. Draft lines 3, 5–6, 19, 43, 47, 51–53, 55–56, 59–60, 63 (two rows added after it),
>    83–84, 89 (two rows added after it), 221 (a row added after it), 244, 258, 263, 270 (a paragraph added
>    after it), 426–429, 451 (§3.8 added after it), 462, 505, 518–519, 539, 548 (a row added after it), 552
>    (paragraphs added after it), 574 and 578 (lines added after each), 644, 692, 805–806, 809–813, 825, 827,
>    875–895, 934, 956, 981.
> 3. **Table 49.** Its "7B RL-Zero" column matches the 3.1 run `ydkgbnai`, not the 3.0 run `zim8gcyv` (16 × 16,
>    response length 12,000). Draft lines 61–62 and 281–282.
>
> Also changed: line 101 now names the realized-perturbation study and its path by a glob, so that no word
> banned by the house rules appears in this file.

This run decides whether the MATH-500 gain of `allenai/Olmo-3.1-7B-RL-Zero-Math` over its pre-RL base is
carried by its 352 to 10,568 most-changed MLP neurons (0.1% to 3% of 352,256), or whether a DARE-rescaled
random subset of the same parameter count does as well. Estimated cost: 25.5 to 37.7 GPU-h on `g6e.2xlarge`,
with a ceiling of 35 GPU-h; above the ceiling the stop rule parks the lowest-priority arms (§9.7).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | **Frozen at freeze 1** (2026-11-06T04:00:49Z, `n13-freeze-1`). Written 2026-11-05; amended A1–A28 before the tag. |
| freeze 1 | `n13-freeze-1`: this spec, the code, the offline gate green. **Tagged 2026-11-06T04:00:49Z** on commit `79e9a28e580fad161272d66647277ec9d475027e` (annotated tag object `d200535394e7318aad2d24ef844cb48aa01dfdc3`), with amendments A1–A28 and `N13_DRYRUN_OK` 107 of 107 (`studies/next/n13/offline_gate/rerun_a27/`). Authorized by the head, 2026-11-06. |
| freeze 1a | `n13-freeze-1a`: freeze 1 plus A29, the first amendment after freeze 1 (approved by the head, 2026-11-06). **Tagged 2026-11-06T05:48:57Z** on commit `b8ae07996134b94cdd4e126a987bfbff1e4641cd` (annotated tag object `711449f4b38263bf8422db2d33021543db44b2fe`), `N13_DRYRUN_OK` 109 of 109 (`studies/next/n13/offline_gate/rerun_a29b/`). The frozen rules of freeze 1 are unchanged. |
| freeze 2 | `n13-freeze-2`: `studies/next/n13/thresholds.json`, written from the gate runs and before any arm runs (§6.2). **Tagged 2026-11-06T18:09:51Z** on commit `ad8a3da392acb30e9e0465180a0036066ed88d24` (annotated tag object `31a432346c504b713fb764930cc73c68e3598151`). C 16,384; σ 1.924, δ 5.441, B 57.2, L 92.0, G 34.8, H 74.6, Q 65.9 (MATH-500). Every S3 gate passes. A30 is in code. No S4 arm has run. |
| freeze 2a | `n13-freeze-2a`: freeze 2 plus A31, the first amendment after freeze 2 (accepted by the head, 2026-11-07). **Tagged 2026-11-07T15:47:29Z** on commit `e69f2b2b72edfe0aca4fc13dc206e2e965627ecb` (annotated tag object `25c270e171c104285f1c98b8feff0883fcbc06f3`), `N13_DRYRUN_OK` 111 of 111 (`studies/next/n13/offline_gate/rerun_a31/`). The frozen rules of freeze 2 are unchanged. `s4armsB1-02` and `s4armsB2-02` launch from this commit. |
| method | `~/.claude/skills/research-spec/SKILL.md`; repo `CLAUDE.md` ("Before any compute", "Instruments and controls", "Provenance") |
| parents | `studies/next/03_IDEAS.md` N13; `studies/next/06_CRITIC.md` Part 2 N13 (lines 567–612) and Part 6 (verdict "RUN WITH CHANGES"); `studies/next/04_PRACTICAL.md` §1(d) R1 and §5 item 2; `studies/next/04_LITERATURE.md` N13; `studies/next/04_RECENT.md` §6; `studies/next/00_ASSETS.md` §2, §4 |
| scope | one stage pair (pair 1: base `main` → `Olmo-3.1-7B-RL-Zero-Math` `main`), its fallback (pair 1b: base `main` → `Olmo-3-7B-RL-Zero-Math` `main`, §3.8), one fallback pair for the noise gate (pair 2, §3.7), MLP neurons, two benchmarks. The neuron job labels of the card's demo are not in this run (§3.6). |
| vocabulary | The house rules of `studies/next/README.md`. A run whose gate fails is **CANCELLED**. Chance level is the random-arm reading. |

### 0.1 What the reviewer changed, and where each change sits

The critic's verdict row for N13 (`06_CRITIC.md` Part 6) lists these changes. Each is placed here.

| critic's change | where |
|---|---|
| DARE-rescaled random set | arm `dare_k`, §3.2 |
| tail-energy-matched random set | arm `rndT_k`, §3.2 |
| bf16 floor | §0.5, H_artifact, §6.3 |
| attention-and-norm arm and complement arm | arms `attn`, `comp_k`, §3.2 |
| gain gated against MATH-500 noise | gate G-GATE, §6.1 |
| absolute accuracies | §2.1, §7, §10.3 |
| jobs from harvests at `main` and each stage | not in this run (§3.6); it serves the demo, not the decision |
| budget 20 to 25 GPU-h | §9.7 |

### 0.2 Facts read for this draft

Read on 2026-11-05 through the Hugging Face API (`/api/models/<repo>`, `/refs`, `/tree/<rev>`), the raw model
cards, the arXiv HTML of the Olmo 3 report, and PyPI JSON. The API responses are saved in
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/n13/`
(`refs_*.json`, `info_*.json`, `trees.json`, `readme_*.md`, `tmpl_rlzm.jinja`, `dl/olmo3.txt`). That folder is
session scratch. The executor re-reads every fact in §0.3 at run time. The facts added by the amendment of
2026-11-05 come from `studies/next/07_PICKS/N13_MATH_TASK.md` and from the saved responses in the same
scratchpad under `mathtask_hf/` (HF model API for `Olmo-3.1-7B-RL-Zero-Math`, base commit list) and
`mathtask_wandb/` (W&B run configs and histories).

| fact | value | source |
|---|---|---|
| pre-RL base, pairs 1 and 1b | `allenai/Olmo-3-1025-7B`, branch `main`, commit `a81bae42db3975be1671e27b9c9a56da1a9f980f` | HF refs |
| base shards (sha256) | `0490d6668e61…c4e2`, `e127ea479fb6…c641`, `f3ddff10052f…3c70` | HF tree API, LFS oid |
| `main` equals the end of stage 3 | `stage3-step11921` (commit `4b32e100e6d94c5903b9680b6662c45089c534e4`) has the same three shard hashes | HF tree API |
| end of stage 2 (the other init candidate) | `stage2-step47684`, commit `c3c800dc900f3fecc112e8cd6b2a13edabc096e1`, shards `822ceb78d3fa…`, `7551e03e9c1d…`, `28e89370d6ab…` (differ from `main`) | HF tree API |
| RL model, pair 1 (registered) | `allenai/Olmo-3.1-7B-RL-Zero-Math`, `main`, commit `364820b002fc8e81d11a81c7fc9dd6b49c83cd05`; shard LFS oids read at freeze 1 (§0.3 item 9) | HF model API, `sha` field (read 2026-11-05) |
| RL model, pair 1b (fallback, §3.8) | `allenai/Olmo-3-7B-RL-Zero-Math`, `main`, commit `dee26a4074b88215cd1326472e1ab14d0259d892`, shards `5d43c3f9c8c1…e8a1`, `56eea2882ea4…7633`, `9336a00f7478…71c9` | HF refs, tree API |
| the cards | 3.1: `base_model: allenai/Olmo-3-1025-7B`, dataset `allenai/Dolci-RL-Zero-Math-7B`, no evaluation section, no `model-index`. 3.0: `base_model: allenai/Olmo-3-1025-7B`; `new_version: allenai/Olmo-3.1-7B-RL-Zero-Math`; no evaluation section | raw README; HF model API; `N13_MATH_TASK.md` |
| the 3.1 init | the W&B config of run `ydkgbnai` ("olmo3_7b_rlzero_math") names `Olmo-3-1025-7B` revision `39ff1a89dc6950c25f4a435a993904f8587c6a3a` (a base commit of 2025-11-24); its three safetensors LFS oids equal those of base `main` (`0490d6668e61`, `e127ea479fb6`, `f3ddff10052f`). The restart run `kl07u3lf` starts from `ydkgbnai` `step_2000`. | W&B GraphQL, project `ai2-llm/Olmo-3-7B-RL-Zero`; HF tree API; `N13_MATH_TASK.md` |
| intermediate checkpoints | 3.1: branches `step_0100` to `step_2800`; which training step 3.1 `main` is, is not established. 3.0: 19 branches `step_100` to `step_1900`; `main` differs from `step_1900` (commit `8182367150cef52ddf00dd5259ea94eaa330918e`, other shard hashes) | HF refs, tree API; agrees with `00_ASSETS.md` §2; `N13_MATH_TASK.md` |
| pair 2 | before: `allenai/Olmo-3-7B-Think-DPO`, `main`, commit `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`, shards `c3021a8d3a08…`, `40693e47833b…`, `d54321b82037…`. After: `allenai/Olmo-3-7B-Think`, `main`, commit `d97e442d7cc678210054dbcc9b440894d62c89a4`, shards `611c266f99a1…`, `a032a27f1768…`, `50ba91f88796…`; card `base_model: allenai/Olmo-3-7B-Think-DPO`; 55 branches `step_0025` to `step_1375`; `main` differs from `step_1375` | HF refs, tree API, raw README |
| parameter count | 7,298,011,136, all BF16, in each of the six repos above | HF API `safetensors` field (3.1 read 2026-11-05) |
| shapes | 32 layers; hidden 4,096; MLP width 11,008; 32 attention heads and 32 key-value heads; vocabulary 100,278; untied embeddings; `sliding_attention` on three of every four layers (window 4,096); YaRN factor 8. 3.1: architecture `Olmo3ForCausalLM` (HF API); its `config.json` is compared at freeze 1 (§0.3 item 9). | `config.json` at `main` of base and 3.0 RL (identical when parsed) |
| parameter partition (derived) | MLP 32 × 3 × 4,096 × 11,008 = 4,328,521,728; attention projections 32 × 4 × 4,096² = 2,147,483,648; norm weights 32 × 4 × 4,096 + 4,096 = 528,384; embeddings and LM head 2 × 100,278 × 4,096 = 821,477,376. The sum equals 7,298,011,136 exactly. | arithmetic on `config.json` |
| MLP neurons | 32 × 11,008 = 352,256 | `config.json`; `00_ASSETS.md` §1.1 |
| RL-Zero math prompt, pair 1 | the 3.1 repo's chat template; sha256 of its text `4380e62276b7ce7e28e30be381840f7cef65f20f43173437cae5d2d6908a6155` (425 characters). It differs from the 3.0 template by one word: "Solve the following problem step by step" against "Solve the following math problem step by step". So it differs from Figure 37 by that word. | HF model API, `config.chat_template_jinja` (read 2026-11-05); the same hashing of the 3.0 text gives `bbff3bab…` |
| RL-Zero math prompt, pair 1b | the 3.0 repo's `chat_template.jinja` (HF blob `63b0c16e0cc63e6a9d5485b718fa2161110a322f`; sha256 of its text `bbff3bab94f2df1534d5971cb5ef47bfa3b06f1fb56965cc70dbcf261e72d3fa`). It equals Figure 37 of the report. | raw file; https://arxiv.org/abs/2512.13961 Figure 37 |
| tokenizers differ | base `tokenizer.json` blob `5fe172127988…`, 3.0 RL `060ff36bb4f8…`. The 3.0 RL config sets `bos_token` `<|endoftext|>` (base: none) and renames added tokens 100256 to 100275 (for example `<|extra_id_3|>` → `<think>`). Neither sets `add_bos_token`. The 3.1 `tokenizer_config` sets no `bos_token`, as the base; its `tokenizer.json` blob is read at freeze 1 (§0.3 item 9). | raw `tokenizer_config.json`; HF model API for 3.1 |
| RL-Zero training (Table 49) | Table 49, column "7B RL-Zero": 13,314 prompts; 2,000 steps; learning rate 1.0e-6 constant; response length 16,384; sampling temperature 1.0; 32 prompts × group 8 per step; TIS cap 2.0. This column matches the **3.1** run `ydkgbnai` (`allenai/Dolci-RLZero-Math-7B`, 32 × 8, response length 16,384, 2,001 steps, learning rate 1e-6 constant, `mask_truncated_completions` false), not the 3.0 run `zim8gcyv` (16 × 16, response length 12,000, `mask_truncated_completions` true, DAPO-Math-17k and MATH_3000 subsets, 2,000 steps). The 3.1 branches run past 2,000 steps, to `step_2800`, through the restart run `kl07u3lf`. | Olmo 3 report, Table 49; W&B configs (`N13_MATH_TASK.md`, side finding 1) |
| RL-Zero 3.0 vs 3.1 | "setting completion length to 16k instead of 12k" and "not masking truncated sequences" are named as 3.1 changes | report, Appendix A.6.4 |
| RL-Zero evaluation in the report | AIME 2024 and 2025, pass@1 as a bootstrapped average over 32 samples, at 32K tokens and temperature 1.0 (§6.1, Figure 24). The numbers are only in the figure. No MATH-500 number for RL-Zero-Math is in the text. | report §6.1, §6.2 |
| published math numbers of the RL models | 3.1: MATH-500 `exact_match_flex` 0.874 to 0.908 over 15 logged evaluations (runs `ydkgbnai`, `kl07u3lf`), task `minerva_math_500::hamish_zs_reasoning_rlzero`: 0-shot, chat format, temperature 1.0, top-p 0.95, 32,768 tokens, one sample. That task's prompt text and answer extraction are not public (not in public olmes). 3.0: no MATH or MATH-500 number anywhere; AIME 2024 pass@1 0.416 and pass@32 0.800 at step 2000 (W&B run `zim8gcyv`, 0-shot chat, temperature 1.0, top-p 0.95, 32,768 tokens, 32 samples). | W&B GraphQL, project `ai2-llm/Olmo-3-7B-RL-Zero`; `N13_MATH_TASK.md` |
| training-rollout lengths at the end of each run | mean response length of the last logged rollout batch, on training prompts: 3.1 `ydkgbnai` (step 1,999) 12,756 tokens; 3.1 restart `kl07u3lf` (step 1,280) 10,689; 3.0 `zim8gcyv` (step 2,000) 8,181. Not MATH-500 lengths. | W&B run summaries, `val/sequence_lengths` |
| post-trained MATH 500 protocol | CoT, metric "EM Flex", temperature 0.6, top-p 0.95, Minerva answer extraction, 32,768 max tokens, N = 1; thinking traces stripped before scoring | report Table 16 and its caption |
| base reference number | Table 3 ("OlmoBaseEval Main suite"), row MATH, column Olmo 3 7B: **40.0**. It is **Minerva MATH**, the full 5,000-problem Hendrycks MATH test (`EleutherAI/hendrycks_math`, 7 subjects), olmes task `minerva_math::olmes:n4:v2`: 4 fixed shots (`Minerva:MATH:fixed`) in the `Problem:`/`Solution:` text format, no chat template, temperature 0.6, top-p 0.6, 1,024 tokens, stop at `Problem:` or a blank line, 4 samples, pass@1 under EM Flex, macro mean over the 7 subjects (`primary_score`). It is not MATH-500. Table 13 gives 39.8 for the same released checkpoint ("Olmo 3 7B Stage 3"). No MATH-500 number for the base is published. Row "OlmoBaseEval Math": 54.7, the mean of GSM8k 75.5, GSM Symbolic 48.6 and MATH 40.0 (equals the base card). | report Tables 3, 13, 46; olmes at `5a51f502…`; `N13_MATH_TASK.md` |
| Think chain, "MATH" row | SFT 94.4, DPO 92.4, Think 95.1; AIME 2024 69.6 / 74.6 / 71.6; AIME 2025 57.6 / 62.7 / 64.6; IFEval 77.9 / 75.9 / 88.2. "All numbers are the mean of three runs." | report Table 15; https://huggingface.co/allenai/Olmo-3-7B-Think |
| run-to-run spread in the report | MATH: 0.2522 (the mean over 14 models of the standard deviation of 3 runs) | report §4.1.1 |
| Think decoding | `generation_config.json`: temperature 0.6, top-p 0.95, max_new_tokens 32,768, eos ids [100265, 100257] | raw file |
| datasets | `HuggingFaceH4/MATH-500` commit `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` (`test.jsonl`); `openai/gsm8k` commit `740312add88f781978c0658806c59bc2815b9866` (`main/test-00000-of-00001.parquet`) | HF dataset API |
| packages | `vllm` 0.32.0, uploaded 2026-10-18, requires `torch==2.13.0` and `transformers>=5.16.1,<5.20.0`; `math-verify` 0.9.0, uploaded 2026-01-10 (https://github.com/huggingface/math-verify) | PyPI JSON |
| AMI | `pg-env-v1`, torch 2.12 + cu130; no vLLM in `bake_env.sh` | `infra/aws_pg/config.py`, `infra/aws_pg/bake_env.sh` |
| launcher limits | default `g6e.2xlarge` (1× L40S 48 GB, 8 vCPU, 64 GB host RAM); preference list `["g6e.2xlarge", "g6e.4xlarge"]`; 32-vCPU G quota, so 4 at once; "`p5`/`p4de` are not used" | `infra/aws_pg/config.py` lines 26–34; `.claude/skills/aws-pg/SKILL.md` |
| weight cache | no Olmo-3 repo in `s3://pg-lineage-769063704696/weights/hf/`; checkpoints download in-job (14.6 GB each) | `infra/pg/matrix_repo_facts.json` `_meta.bucket_dirs`; `00_ASSETS.md` §1.6 |

### 0.3 Facts to verify at run time

Each line names where the fact is read and what the run does if it differs. None of them needs a human
mid-run. Item 1 was settled on 2026-11-05 (`07_PICKS/N13_MATH_TASK.md`); S1 re-reads it at the pinned commit.

| # | fact | read at | rule |
|---|---|---|---|
| 1 | The task behind Table 3's MATH 40.0: olmes `minerva_math::olmes:n4:v2` (§0.2), in https://github.com/allenai/olmes at `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e` (its math configs equal those of `7bebf0eb148cd89d1d8a8096505f99d4fc48c209`, 2025-10-18). Install: `pip install -e ".[gpu]"`, which pins `vllm==0.11.0`, `lm_eval==0.4.3`, `antlr4-python3-runtime==4.11`, `torch>=2.8.0,<2.9`, Python 3.10 to 3.12. | olmes `oe_eval/configs/task_suites.py` lines 342–345, `oe_eval/configs/tasks.py` lines 4696–4714, `pyproject.toml`, at that commit | S1 re-reads the three files; a difference from G-KNOWN (§6.1) stops freeze 1. In S2, if vLLM 0.11.0 cannot load the base, R0 runs the same task with olmes `--model-type hf`; the tolerance is unchanged and the artifact names the engine. |
| 2 | The pre-RL init. Pair 1 (3.1): the logged init is base revision `39ff1a89…`, whose safetensors equal base `main`'s (§0.2, "the 3.1 init"). This settles which checkpoint the 3.1 run started from. Pair 1b (3.0): not logged publicly (run `zim8gcyv` names a private path, `/weka/oe-adapt-default/michaeln/checkpoints/olmo3-7b-base`). | HF tree API at `39ff1a89…` and `a81bae42…`; weight test G-INIT (§6.1); the RL-Zero launch script in https://github.com/allenai/open-instruct (`scripts/train/olmo3/`) for pair 1b | G-INIT decides. Pair 1 fails G-INIT → pair 1b (§3.8). Pair 1b: if `stage2-step47684` passes instead of `main`, it becomes the base for every arm; if neither passes: CANCELLED. |
| 3 | A published MATH-500 number for the RL model. Pair 1 (3.1): 0.874 to 0.908 under `minerva_math_500::hamish_zs_reasoning_rlzero`, whose prompt text and extraction are not public (§0.2). Pair 1b (3.0): none. | W&B project `ai2-llm/Olmo-3-7B-RL-Zero` (runs `ydkgbnai`, `kl07u3lf`, `zim8gcyv`); the report; the cards | Pair 1: the band is used only as the sanity band S-BAND for the identity transplant (§6.1), not as an exact reproduction target, because the prompt is not public. Pair 1b: the artifact says "no published MATH-500 number for the RL side"; no check. |
| 4 | vLLM 0.32.0 loads `Olmo3ForCausalLM` with sliding window; `LLMEngine.add_request`, `step`, `abort_request`, `has_unfinished_requests`; `SamplingParams(seed=…, logprobs=0)`; `LLM(..., generation_config="vllm")` | the installed package, in the S2 smoke | If an API differs, fix the call before any arm runs (dated amendment). If vLLM cannot load the model, use the newest vLLM release at least 14 days old that lists `Olmo3ForCausalLM`; if none, pivot P-ENGINE (§6.5). |
| 5 | Olmo-3 parameter names: `self_attn.{q,k,v,o}_proj`, `self_attn.q_norm`, `self_attn.k_norm`, `post_attention_layernorm`, `post_feedforward_layernorm`, `mlp.{gate,up,down}_proj`, `model.norm`, `model.embed_tokens`, `lm_head` | `model.safetensors.index.json` at the pinned commits | Asserted. The four families must sum to 7,298,011,136. A mismatch stops R1. |
| 6 | Which split Table 15's "MATH" row is | report Table 16 and OLMo-Eval | Pair 2 only. If it is not MATH 500, pair 2 has no published reference; the artifact says so. |
| 7 | The user message used for MATH 500 in the post-training suite | OLMo-Eval task config | Pair 2 only (§3.7); settled before pair 2 launches. |
| 8 | Host RAM and disk suffice to build an arm while the base and RL are memory-mapped | S2 smoke (peak RSS, free disk) | If not, `--type g6e.4xlarge` (same GPU, 128 GB host RAM per AWS listing, to verify). |
| 9 | The 3.1 repo at `364820b0…`: shard LFS oids, `config.json`, `tokenizer.json`, `generation_config.json`, `chat_template.jinja`; whether `main` equals a `step_*` branch | HF tree API and raw files, in S1 | Recorded in §0.2 before freeze 1. A `config.json` that differs from the base's in the §0.2 shape fields fails G-PARTITION → pair 1b. The template must render the bytes of §2.2 (G-TEMPLATE); if it does not, pair 1b. Which step `main` is changes no rule; it is recorded. |
| 10 | MATH-500 answer lengths of 3.1 under §2.2–2.3 (the budget of §9.7 assumes a mean of 2,500 tokens) | S2 smoke | The projection is redone before S3; the stop rule decides what runs. |

### 0.4 Already-done audit

**In this repo.** Every earlier measurement on the cells this plan touches:

| cell | prior measurement | path | what it gives this run |
|---|---|---|---|
| OLMo-3 post-training checkpoints | none run | `00_ASSETS.md` §6 gap 3 ("none of the Think/Instruct chain") | nothing to reuse |
| OLMo-3 7B neurons | all-neuron top-50 harvests at 16 stage-1 points, last `stage1-step1413814` | `s3://pg-lineage-769063704696/results/neurons/w3/`; `00_ASSETS.md` §1 | not used: the RL chain starts from `main`, after stages 2 and 3, not from `stage1-step1413814` (`06_CRITIC.md` N13 item 5, citing `01_JLENS.md` §7) |
| OLMo-3 load and loss | pilot-0 loss check at `stage1-step60000` | `studies/next/pilot0/GPU_CHECK.md`; `infra/aws_pg/jobs/next_pilot0_loss.py` | the in-job HF download pattern for this model |
| MATH-500 generation | ALWAYSON SPEC2 and INSTRUMENTS E2 on Qwen3.5-0.8B (200 and 300 problems, `model.generate`) | `studies/alwayson/SPEC2_edit.md`; `infra/aws_pg/jobs/alwayson_edit.py`; `studies/instruments/results/e2/verdict_e2.json` (frozen room gate failed, every rule cancelled, exit 4) | the problem hashing (`alwayson_edit.row_sha256`) and `boxed_last`. Its grader (`answer_flags`: string equality after whitespace removal) is too strict for MATH-500 equivalence; this run needs a new grader. E2's failure is the reason this run has a noise gate with room checked before any arm. |
| matched random controls | the realized-perturbation study: a random subspace matched on the realized perturbation | `SPEC.md` in the one directory `studies/realizednull/SPEC.md` | the rule that a control is matched on realized, not intended, energy |
| transplants of named structure | STAGEDIFF S-4 head-and-norm transplant: UNDECIDED (the carry clause could not pass); SINKPATCH residual patching; ENDOW builder (gemma-2b into a random init) | `studies/stagediff/CLAIMS.md` S-4; `00_ASSETS.md` §3 SINKPATCH row; `infra/pg/endow_transplant.py` | S-4's lesson: check that the pass clause can fire before freezing it (§6.4) |
| matched-mass rollback literature | 2605.20296 App. I.3 (TIES, WiSE-FT, DARE-Linear at matched Frobenius); 2505.16703 (keep the top M% changed neurons) | `studies/unite/LITERATURE_2026-10-31.md` M3 | DARE-Linear's realized ratio fell short of target (0.354 for 0.5): audit realized energy |

**Literature.** Query phrases are recorded in `04_LITERATURE.md` N13 (six phrases) and `04_RECENT.md` "Overlap
with our cards", N5/N13/N16 (seven phrases). For this draft I ran two more searches on 2026-11-05: "DARE drop
and rescale applied to RLVR reinforcement learning weight delta random sparsification keeps reasoning gain" and
"transplant top-k changed MLP neurons from RL model into base model recovers math accuracy random neurons
control". Neither found the experiment below. Read levels: TEXT = full text read for the claim; ABSTRACT; the
level of the cited file otherwise.

| work | claim used here | read level |
|---|---|---|
| Mukherjee, Yuan, Hakkani-Tur, Peng, "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models", https://arxiv.org/abs/2505.11711 | 68.5% to 96.0% of bf16 parameters unchanged after RL at tolerance 1e-5; "nearly all parameter matrices receive similarly sparse updates"; updates nearly full-rank | TEXT (`04_LITERATURE.md` N13) |
| Zhu et al., "The Path Not Taken: RLVR Provably Learns Off the Principals", https://arxiv.org/abs/2511.08567 | Def. 2.1: unchanged iff \|ŵ − w\| ≤ η·max(\|w\|, \|ŵ\|), η = 1e-3, which is equivalent to bitwise equality in bf16; RL sparsity 36% to 92%; "bfloat16 precision further accentuates the apparent sparsity by attenuating micro-updates" | TEXT (read for this draft, §2.1 and App. E) |
| Shenfeld, Pari, Agrawal, "RL's Razor", https://arxiv.org/abs/2509.04259 | the sparse updates came from bfloat16; "Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates"; all algorithms give full-rank updates | TEXT (read for this draft, the paragraph "Sparsity and rank of updates") |
| Yu et al., "Language Models are Super Mario" (DARE), https://arxiv.org/abs/2311.03099 | randomly drops delta parameters with ratio p and rescales the rest by 1/(1 − p); SFT deltas "typically small (within 0.002)"; "DARE can effortlessly eliminate 90% or even 99% of them" | ABSTRACT (read for this draft) |
| Deng et al., "DARE the Extreme", https://arxiv.org/abs/2410.09344 | "DARE fails when either the pruning rate or the magnitude of the delta parameters is large", from "an excessively large rescaling factor" | ABSTRACT (read for this draft) |
| Panigrahi, Saunshi, Zhao, Arora, "Task-Specific Skill Localization in Fine-tuned Language Models", https://arxiv.org/abs/2302.06600 | a learned graft of about 0.01% of parameters recovers more than 95% of fine-tuned performance (RoBERTa); regions chosen by movement do worse than learned regions at low sparsity (Fig. 2); AdamW fine-tuning has no sparse graft without an ℓ1 penalty | TEXT (`04_LITERATURE.md` N5) |
| Sun et al., "Weight Patching", https://arxiv.org/abs/2604.13694 | the top 5% of heads and neurons patched from instruct into base recover "a substantial portion" of instruction following (figure only) | TEXT (`04_LITERATURE.md` N13) |
| Du, Tang, Duan, Liu, https://arxiv.org/abs/2609.15064 | steering the changed SAE features into the base recovers about 80% of an RL gain, Qwen2.5-1.5B | full (`04_RECENT.md` §4) |
| Shiromani, "What a Weight Update Buys", https://arxiv.org/abs/2610.37719 | RL-Zero and RLVR updates on Olmo-3 have 388 to 1,508 times the Fisher leverage of random directions of the same size; whole-model only | full (`04_RECENT.md` §2) |
| Gupta, Zhang, Jin, https://arxiv.org/abs/2610.22937 | swapping attention patterns of six middle layers between RL and DPO checkpoints (OLMo-2-7B, Tulu-3-8B) moves a cue effect "most of the way" | full (`04_RECENT.md` §6) |
| Kim et al., https://arxiv.org/abs/2611.04141 | "Same-energy random directions often nearly match" a grafted LoRA thinking subspace | ABSTRACT (`04_RECENT.md` §6) |
| Wu et al., "Sparsity Curse", https://arxiv.org/abs/2606.18521 | "RLVR models suffer severe degradation under standard merging methods" | ABSTRACT (read for this draft) |
| Altaf, https://arxiv.org/abs/2610.22438 | the counted rank of updates is only a little below a random matrix of the same shape | ABSTRACT (`04_RECENT.md` §6) |
| Wang et al., https://arxiv.org/abs/2610.06851 | ".\n\nOkay" raises base Olmo-3-7B MATH-500 pass@1 from 42 to 78 percent: the base score depends strongly on the prompt | full (`04_RECENT.md` §2) |
| Team Olmo, "Olmo 3", https://arxiv.org/abs/2512.13961 | every Olmo 3 fact in §0.2 | TEXT for the sections cited (read for this draft) |

**What this spec adds.** No study found combines these on one model: a neuron-basis transplant of an RL
gain at 7B, from one released stage into its own init; DARE at the same parameter count as the neuron set;
count-and-energy-matched and tail-energy-matched random neurons; the attention-and-norm arm and the
complement on the same chart; a ranking on exact float32 differences of bf16 checkpoints with a defined
one-step floor; absolute accuracies behind a measured noise gate. `04_LITERATURE.md` N13 says PARTLY;
`04_RECENT.md` finds no neuron transplant between OLMo-3 stages with a count- and energy-matched control.

### 0.5 Objects, defined once

- **Neuron** (l, n): row n of `gate_proj.weight` [11,008 × 4,096] and of `up_proj.weight` [11,008 × 4,096],
  and column n of `down_proj.weight` [4,096 × 11,008], in layer l. 12,288 parameters.
- **Change** Δ = float32(W_after) − float32(W_before), per entry. Both inputs are bf16. The float32 difference
  of two bf16 numbers is exact when their exponents differ by at most 16; R1 counts the entries where this
  does not hold and reports the count.
- **Changed entry**: Δ ≠ 0, equivalently the bf16 bit patterns differ (±0 treated as equal). This equals
  Def. 2.1 of 2511.08567 with η = 1e-3.
- **Bit distance** d: map each bf16 value to an ordered integer o (o = i if i ≥ 0, else −(i & 0x7FFF), on the
  int16 view i). d = \|o_after − o_before\|. d = 0 unchanged; **d = 1 one-step entry** (the after value is the
  bf16 neighbor of the before value); **d ≥ 2 above-floor entry**.
- **bf16 floor (spacing)** sp(w) = 2^(⌊log2\|w\|⌋ − 7) for normal w. A one-step entry has \|Δ\| = sp(w_before),
  except at a power of two, where the step down is half that.
- **Neuron change score** s_(l,n) = sqrt(‖Δg_n‖² + ‖Δu_n‖² + ‖Δd_n‖²), sums in float64. **Energy** of a set
  S: E(S) = Σ_(n∈S) s_n². **Above-floor score** s'_n: the same sum over entries with d ≥ 2 only.
- **Top-k set** S_k: the k_n neurons with the largest s, over all 32 layers together. Ties broken by (layer,
  index) ascending. **Tail** T_k: all other neurons.
- **k grid** (fraction of MLP neurons → k_n): 0.1% → 352; 0.3% → 1,057; 1% → 3,523; 3% → 10,568
  (k_n = round(f × 352,256)). Parameters touched: 4,325,376; 12,988,416; 43,290,624; 129,859,584, which
  is 0.059%, 0.178%, 0.593%, 1.779% of all 7,298,011,136. **Primary k: 1%.**
- **Arm**: a set of bf16 weights built from the before and after checkpoints (§3.1).
- **Seed set**: A uses per-problem seed 1,000,000 + problem index; B uses 2,000,000 + problem index.
- **acc(a)**: percent of the 500 MATH-500 problems that arm a answers correctly under seed set A.
- **B, L, G**: B = acc(base-A); L = acc(rl-A); G = L − B. B' = acc(zero-B); L' = acc(id-B).
- **σ**: the standard deviation of one single-seed accuracy reading, from the two seed runs of the same
  weights: σ_m = 100 × sqrt(Σ_i (x_iA − x_iB)² / 2) / 500, where x_i ∈ {0, 1}; σ = max(σ_base, σ_rl). The term
  (x_A − x_B)²/2 is an unbiased estimate of p_i(1 − p_i).
- **δ** = 2·√2·σ: the 95% band for the difference of two single-seed readings of identical weights.
- **H** = B + G/2 (halfway). **Q** = B + G/4 (quarter).

---

## 1. Question, decision, hypotheses

### 1.1 The question

When RL post-training raises MATH-500 accuracy, is the gain carried by a small set of MLP neurons (their gate
rows, up rows and down columns) that can be copied into the pre-RL model? The comparison is against random
neurons matched on count and on energy, and against DARE at the same parameter count. The alternative is that
the gain is spread, so that any rescaled random subset of the change does as well (2311.03099).

### 1.2 The decision this run makes

| outcome | what is done next |
|---|---|
| H_local supported at k_rec ≤ 3% | The gain ships as a k-neuron delta (at most 129,859,584 parameters). Next: job harvests at `main` and RL for those neurons (the card's demo, about 1.2 to 1.8 GPU-h each per `06_CRITIC.md` N13 item 5); N16 neuron-level merging. |
| H_diffuse supported | The RL change is not localized in the neuron basis at these counts. N16 is set aside. Next: DARE-style compression of RL deltas as the practical result. |
| H_attention supported | Move the question to heads and norms (N3, 2610.22937). |
| H_artifact supported | "Most-changed neuron" rankings on bf16 checkpoints are not trusted; claims about which neurons RL changed are withdrawn; the transplant numbers stand as transplants of released values. |
| CANCELLED | No claim. The cancelled gate and its value are reported. |

### 1.3 Hypotheses and the observation that refutes each

All accuracies are absolute MATH-500 accuracies under seed set A; labels are defined in §6.3.

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_local** | A small top-k set carries the gain; random neurons and DARE do not. | top_k reaches H at some k ≤ 3%, and there it beats dare_k, rndE_k and rndT_k by more than δ | top_3% stays below H; or, at k_rec, dare_k, rndE_k or rndT_k comes within δ of top_k or above it |
| **H_diffuse** | The gain is spread: DARE-rescaled random entries at the same parameter count do as well. | dare_k reaches H and is within δ of top_k at some k | at every k, dare_k stays below H (or is DAMAGED), or top_k beats it by more than δ |
| **H_attention** | The gain sits in attention and norm weights, not in MLP neurons. | `attn` reaches H; `mlp` stays at or below Q | `mlp` reaches H while `attn` stays at or below Q; or both reach H |
| **H_artifact** | The "changed" set and the ranking are bf16 rounding artifacts. | the top-1% set changes when one-step entries are removed or when sub-floor values are redrawn | the top-1% set from above-floor entries (d ≥ 2) and the mean over 20 sub-floor redraws both overlap the raw top-1% set with Jaccard ≥ 0.7 |

The hypotheses are not all exclusive. H_local and H_diffuse can both hold at different k (DARE often fails at
large rescaling factors, 2410.09344). The question is answered at k_rec, the smallest k where top_k reaches
H. At k_rec, H_local and H_diffuse predict opposite outcomes of one comparison: top_k − dare_k > δ against
\|top_k − dare_k\| ≤ δ.

### 1.4 Refutation table: the observation that separates each pair

| pair | separating observation |
|---|---|
| H_local vs H_diffuse | at k_rec: top_k − dare_k > δ (local) against dare_k ≥ top_k − δ (diffuse) |
| H_local vs energy alone | at k_rec: top_k − rndE_k > δ and top_k − rndT_k > δ (local) against either within δ (energy carries it) |
| H_local vs H_attention | `mlp` reaches H and `attn` stays at or below Q, and top_k reaches H (local) against `attn` reaching H and `mlp` at or below Q |
| H_artifact vs the ranking being real | Jaccard of top-1% sets: raw against above-floor, and raw against the sub-floor redraws (≥ 0.7 real, < 0.4 artifact) |
| any hypothesis vs a broken harness | gates G-INIT, G-KNOWN, G-HASH, G-ID (§6.1) |

### 1.5 The author's prior, logged before any data (2026-11-05)

These are not hypotheses. They are written down so the result can be compared with them.

| quantity | prior | basis |
|---|---|---|
| G (pair 1) | 15 to 35 points (logged for the 3.0 RL model, before the amendment) | Table 3 base Minerva MATH 40.0 (5,000 problems, 4-shot; a different task from MATH-500, §0.2); base MATH-500 pass@1 42% under another prompt (2610.06851) |
| L (pair 1, 3.1; added at the amendment, 2026-11-05) | 80 to 92 | the published 3.1 band 87.4 to 90.8 under a prompt that is not public, temperature 1.0 and 32,768 tokens (§0.2); our temperature is 0.6 and our cap is at most 16,384 |
| share of MLP entries unchanged | 0.4 to 0.9 | 2511.08567 reports 36% to 92% for RL |
| H_artifact refuted (J ≥ 0.7) | probability 0.6 | none strong |
| top_3% reaches H | 0.6 | Weight Patching recovers with 5% of heads and neurons (2604.13694) |
| top_1% reaches H | 0.35 | |
| dare_3% reaches H and within δ of top_3% | 0.5 | DARE removes 90% or 99% of SFT deltas (2311.03099); dare_3% drops 97% |
| dare_0.1% DAMAGED | 0.7 | rescale factor 1,000.7; 2410.09344 |
| rndT within δ of top at k_rec | 0.5 | same-energy random directions nearly match (2611.04141) |
| `mlp` reaches H | 0.7 | MLP holds 59% of parameters (§0.2) |
| `attn` reaches H | 0.3 | 2610.22937 shows attention matters for one cue effect |
| pair 2 passes G-GATE on MATH-500 | 0.2 | published gain +2.7 points (Table 15) |

---

## 2. The measurement and the instrument

### 2.1 The measurement, in plain words

- **Quantity**: MATH-500 accuracy, in percent of 500 problems. A problem counts as correct when the answer
  extracted from the model's text is judged equal to the gold answer by `math_verify.verify`.
- **Items**: the 500 rows of `HuggingFaceH4/MATH-500` `test.jsonl` at commit `6e4ed1a2…`, in file order.
  problem index = row index. Each row's sha256 (`alwayson_edit.row_sha256`) is banked in `problems.json`.
- **Before-model**: `allenai/Olmo-3-1025-7B` at `a81bae42…` (subject to G-INIT). **After-model**:
  `allenai/Olmo-3.1-7B-RL-Zero-Math` at `364820b0…` (pair 1b: `allenai/Olmo-3-7B-RL-Zero-Math` at `dee26a40…`,
  §3.8). **Arms**: hybrids of the two (§3).
- **Readout for every arm**: MATH-500 under this registered harness (§2.2–2.4). The Minerva MATH run of
  G-KNOWN (§6.1) checks the base weights against the report; no arm is scored on Minerva MATH.
- **Samples**: one sample per problem per arm, seed set A. Base and RL also run under seed set B (as `zero`
  and `id`, §3.2). `top_1%` also runs under seed set B.
- **Pairing**: by problem index and seed. The same problem gets the same sampling seed in every arm.
- **Summary**: absolute accuracy with a Wilson 95% interval (n = 500). Arm-against-arm: the difference in
  points, the exact two-sided McNemar test on discordant problems, Benjamini–Hochberg at q = 0.05 over the
  primary family (§7), and a paired bootstrap 95% interval (2,000 resamples of problems, seed 0).
- **Held-out readout**: the same measurement on the 1,319 rows of `openai/gsm8k` `main` test at commit
  `740312ad…`, for 11 arms (§3.3). Gold = the text after `#### ` in the `answer` field, commas removed.
- **Units on both sides**: every comparison is between two accuracies in percent of the same problem set,
  under the same prompt, decoding and grader. Never between a fraction of a margin and an accuracy.

### 2.2 The prompt, exactly

All arms use the RL repo's tokenizer and template at commit `364820b0…` (3.1). The prompt is
`tok.apply_chat_template([{"role": "user", "content": problem}], add_generation_prompt=True, tokenize=True)`.
For one message this renders to the bytes below (`\n` is a newline):

```
Solve the following problem step by step. The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after "Answer:"
```

The job asserts that the rendered text equals this string with `{problem}` substituted, for all 500 problems.
It banks the token ids of every prompt and asserts that they are identical in every arm. It records whether
the first id is 100257 (a BOS). Failing input: the render with `add_generation_prompt=False` must differ
(the final line is missing).

Pair 1b uses the 3.0 repo's tokenizer and template at `dee26a40…`, which render the same string with "Solve the
following math problem step by step." as the first sentence; the same assertions apply to that string.

### 2.3 Decoding, exactly

- Engine: vLLM 0.32.0 in its own venv on the instance (§9.2), `dtype=bfloat16`, `tensor_parallel_size=1`,
  `max_model_len = 2,048 + C`, `gpu_memory_utilization=0.90`, `enable_prefix_caching=True`,
  `generation_config="vllm"` (the model's own generation config is ignored), engine `seed=0`.
- Sampling: `n=1`, `temperature=0.6`, `top_p=0.95`, top-k off, `min_p=0`, `repetition_penalty=1.0`,
  `presence_penalty=0`, `frequency_penalty=0`, `max_tokens=C`, `stop_token_ids=[100257]`,
  `seed=<per-problem seed>`, `logprobs=0` (the sampled token's log-probability). Temperature and top-p match
  the report's post-training MATH 500 settings (Table 16); the prompt and the cap do not.
- Cap C: 8,192 if at most 1 of the 20 RL smoke answers (S2) is longer than 8,192 tokens; else 16,384 (the
  3.1 training response length, Table 49 and W&B run `ydkgbnai`; the 3.0 run trained at 12,000). Frozen in
  `thresholds.json` before the gate runs. The same C for every arm and both benchmarks.
- Every effective `SamplingParams` field is recorded in `metrics.json`. Any field not listed above that changes
  the sampling distribution must be at its neutral value (asserted).
- Requests are submitted at the start in problem order (MATH-500 rows 0–499, then GSM8K rows 0–1,318 for the
  arms that carry GSM8K). The scheduler is first-come first-served.

### 2.4 Extraction and grading, exactly

1. Text = the generated ids up to the first stop id, decoded with `skip_special_tokens=False`.
2. Extraction mode `answer_line`: the text after the last line that begins with `Answer:` (leading spaces
   allowed), stripped, up to the end of that line. If none, mode `boxed`: `alwayson_edit.boxed_last(text)`.
   If none, mode `none` (scored wrong).
3. Score: `math_verify.verify(math_verify.parse(gold), math_verify.parse(extracted))` with default
   configurations, `math-verify` 0.9.0. The extracted string is wrapped as `$…$` before parsing when it has no
   `$` or `\boxed`.
4. Recorded per answer: `n_new`, `cap_hit` (n_new = C), `finish_reason`, extraction mode, the extracted
   string, the parsed gold, the score, the sum of sampled-token log-probabilities over the answer line's
   tokens, and the mean over all tokens.
5. Format rate per arm: the share of answers in each extraction mode. It is a secondary readout: the base may
   miss the `Answer:` format, and a gain in format is not a gain in mathematics (§8 threat 5).

### 2.5 The generation loop and where `metrics.json` is written

One arm is one subprocess (`infra/aws_pg/jobs/n13_gen.py`). It builds the arm (§3.1) into
`/work/arm/<arm>/` as three safetensors shards, with `config.json` and tokenizer files copied from the RL
snapshot, then starts vLLM on that directory. The loop is:

```
for each problem in order: engine.add_request(request_id, prompt_token_ids, SamplingParams(seed=...))
while engine.has_unfinished_requests():
    for out in engine.step():
        if out.finished:
            score; append one line to out/<arm>/<bench>/generations.jsonl (flush + fsync)
            update counts; maybe append to samples.jsonl (§10.1)
    every 25 finished answers, and at least every 60 s:
        rewrite out/<arm>/<bench>/metrics.json (write to .tmp, then rename)
    when problems 0-19 have all finished: evaluate the damage screen (§6.3); if it fires on an arm
        (never on base, rl, zero, id: there it is only recorded), abort all requests
rewrite metrics.json with status "done"; exit 0
```

`metrics.json` holds: status, arm, benchmark, pair, cap C, counts (submitted, finished, correct, cap hits, each
extraction mode), output tokens, elapsed seconds, output tokens per second, the screen result, the
arm-building audit (§3.4), the effective sampling fields, and `updated_utc`. `run_job.sh` uploads `out/` to S3
every 30 s. The executor polls `out/<step>/<arm>/<bench>/metrics.json` and never `job.log`.

---

## 3. Design

### 3.1 The transplant operation

For every parameter tensor θ, with the before and after checkpoints θ_b and θ_a in bf16:

```
θ_T = bf16( float32(θ_b) + c ⊙ M ⊙ ( float32(θ_a) − float32(θ_b) ) )
```

M is a 0/1 mask of the tensor's shape; c is a scale (a scalar or 1). For a neuron set S in layer l:

```
M_gate[l][n, :] = M_up[l][n, :] = 1   and   M_down[l][:, n] = 1   for every (l, n) in S; 0 elsewhere.
```

When c = 1 and M = 1, and the float32 difference is exact (§0.5), the float32 sum equals θ_a, and the bf16
cast returns θ_a. The builder
asserts this: for c = 1 arms, the realized tensor equals θ_a on M = 1 and θ_b on M = 0, bitwise. Tensors
are read from memory-mapped safetensors one at a time, so peak host memory stays near one tensor in float32
(the largest, the embedding, is 1.64 GB in float32).

### 3.2 Arms

k ∈ {0.1%, 0.3%, 1%, 3%}; "MLP" means the 96 `gate_proj`, `up_proj`, `down_proj` matrices.

| arm | MLP mask M | scale c | non-MLP | parameters taken from the change | seed set |
|---|---|---|---|---|---|
| `base` | 0 | — | before | 0 | A (loaded from the HF files) |
| `rl` | 1 | 1 | after | all 7,298,011,136 | A (loaded from the HF files) |
| `zero` | 0 (built by the builder) | — | before | 0; bitwise equal to `base` | B |
| `id` | 1 (built by the builder) | 1 | after | all; bitwise equal to `rl` | B |
| `top_k` | S_k | 1 | before | k_n × 12,288 | A |
| `comp_k` | all neurons except S_k | 1 | **after** (attention, norms, embeddings, LM head from RL) | everything except S_k | A |
| `rndE_k` | k_n neurons drawn uniformly from T_k | one scalar c_k = sqrt(E(S_k)/E(R)) | before | k_n × 12,288, rescaled to the top-k energy | A |
| `rndT_k` | a random prefix of a random order of T_k, the shortest with E ≥ E(S_k); count m_k | 1 | before | m_k × 12,288 (m_k ≥ k_n) | A |
| `dare_k` | each MLP entry kept with probability q_k = k_n / 352,256 (Bernoulli) | 1/q_k: 1,000.73, 333.26, 99.99, 33.33 | before | about k_n × 12,288 entries; drop rates 99.90%, 99.70%, 99.00%, 97.00% | A |
| `mlp` | all neurons | 1 | before | 4,328,521,728 | A |
| `attn` | 0 | 1 | attention projections and all norm weights from RL; embeddings and LM head from before | 2,148,012,032 | A |
| `rnd0_1%` | 3,523 neurons drawn uniformly from T_1% | 1 | before | 43,290,624, natural scale (the plain floor) | A |
| `top_1%_B` | S_1% | 1 | before | as `top_1%` | B |
| `rndT_1%_d2`, `dare_1%_d2` | second random draws | as above | before | as above | A |
| `topF_1%` (conditional) | top 3,523 by the above-floor score s' | 1 | before | 43,290,624 | A |

Notes on the random arms.

- **On the brief's two random-neuron arms.** "k random neurons matched on change energy" is `rndE_k`: the
  count is k_n and one scalar brings their summed change energy up to E(S_k). "k random neurons matched on
  the tail's energy" is `rndT_k`: tail neurons at their own change, added until their summed energy reaches
  E(S_k). When the change is concentrated this needs more than k_n neurons (`06_CRITIC.md` N13 item 1.1),
  so the count m_k is reported, not fixed. The count-k random set at its own scale runs once, as `rnd0_1%`.
- **rndT not constructible.** If E(T_k) < E(S_k), the arm takes all of T_k, is labeled `rndT short`, and the
  energy shortfall is recorded.
- **DARE pool.** DARE as published drops from every delta parameter. Here the pool is the MLP only, so that
  `dare_k` and `top_k` differ only in which MLP entries move. This is a stated deviation.
- **Seeds of the draws** (numpy `default_rng`, torch `Generator` for DARE masks on CPU): rndE 10 + j,
  rndT 20 + j, DARE 30 + j, rnd0 50, second draws rndT 120 + j and DARE 130 + j, with j = 0, 1, 2, 3 for
  k = 0.1%, 0.3%, 1%, 3%. Planted control 40. Sub-floor redraws 60 to 79.
- **Realized, not intended.** For every arm the builder records the realized ‖θ_T − θ_b‖² per family, the
  realized count of changed entries, and for DARE the realized kept count (asserted within 0.1% of P_k).
  For rescaled arms the realized energy must lie within 5% of the intended energy, else the arm is flagged.

### 3.3 Which arms carry the held-out readout

GSM8K runs in the same engine load as MATH-500 for: `base`, `rl`, `zero`, `id` (the held-out gate), and
`top_1%`, `dare_1%`, `rndE_1%`, `rndT_1%`, `comp_1%`, `mlp`, `attn`. Eleven arms, one sample per problem.

### 3.4 The arm-building audit (banked per arm, in `metrics.json` and `sets.json`)

Per tensor sha256 of the built arm; for `id` and `zero` these must equal the after and before checkpoints'
(G-HASH). Per family: realized squared norm of θ_T − θ_b, count of changed entries, count inside M. For mask
arms: the realized changed rows of gate and up and the realized changed columns of down must equal S ∩
{neurons with any nonzero change} (G-MASK). Every number is written before the engine starts.

### 3.5 What each piece adds

| piece | adds |
|---|---|
| `top_k` over four k | the sufficiency curve; k_rec |
| `dare_k` | the H_diffuse test at the same parameter count |
| `rndE_k` | rules out "the top set only holds more energy", at equal count |
| `rndT_k` | the same, at natural scale, with no rescaling to damage the model |
| `rnd0_1%` | the plain matched-random floor the card named |
| `comp_k` | necessity: does removing the top set from RL remove the gain |
| `mlp`, `attn` | H_attention; the k = 100% point of the top-k curve |
| `zero`, `id` | bitwise checks of the builder; the seed-B readings of the endpoints (σ, identity pair) |
| `top_1%_B` | σ for an intermediate model (exploratory check of δ) |
| second draws at 1% | draw-to-draw spread of the random arms |
| `topF_1%` | the transplant under the floor-aware ranking (runs only if J_floor(1%) < 0.9) |
| GSM8K | does the decisive contrast at 1% hold on a second benchmark |

### 3.6 Not in this run, and why

- **Job labels before and after** (the card's demo). They need all-neuron harvests at base `main` and at RL,
  1.2 to 1.8 GPU-h each (`06_CRITIC.md` N13 item 5), and they do not decide the question. Follow-up if
  H_local is supported.
- **Other stages** (Think-SFT, Instruct, other RL-Zero domains, and `Olmo-3-7B-RL-Zero-Math` unless pair 1b
  replaces pair 1). One pair answers the question; replication is a follow-up spec.
- **The RL-Zero intermediate checkpoints** (3.1: `step_0100` to `step_2800`). A ranking-stability check across them
  is weight-only and cheap; it is a follow-up.
- **AIME.** The report's RL-Zero readout, but 30 problems per year cannot resolve a transplant
  (`06_CRITIC.md` N13 item 2), and at 8 samples it would cost about 1 GPU-h per arm if AIME answers average
  6,000 tokens (an assumption) at the §9.7 throughput.
- **A learned graft** (2302.06600) as an upper reference: training, out of budget.
- **DAREx-q** (2410.09344): a different rescale; follow-up if `dare_k` is DAMAGED at small k.

### 3.7 Pair 2 (fallback, runs only if pair 1 fails G-GATE)

Before `allenai/Olmo-3-7B-Think-DPO` at `7b18bf92…`; after `allenai/Olmo-3-7B-Think` at `d97e442d…`. Changes
from pair 1:

- G-INIT on (DPO `main`, Think `main`). Failing input: (base `main`, Think `main`).
- Prompt: the Think repo's chat template at its pinned commit, with its default system message; user message
  = the problem plus the MATH 500 instruction of the OLMo-Eval post-training task (§0.3 item 7). Extraction:
  the text after `</think>`, then the last `\boxed{}`, then an `Answer:` line.
- Decoding: temperature 0.6, top-p 0.95; stop ids [100265, 100257]; cap 16,384 if at most 1 of 20 Think smoke
  answers exceeds 16,384 tokens, else 32,768.
- Reference numbers: if Table 15's MATH row is MATH 500 (§0.3 item 6), DPO-A and Think-A must each lie within
  max(2.0, 2·SE) of 92.4 and 95.1. Otherwise the stamp reads "pair 2: no published MATH-500 reference".
- No GSM8K for pair 2 (not budgeted).
- The same arms, gates, labels and priority order. The stop rule (§9.7) decides how many arms run.

### 3.8 Why pair 1 is the 3.1 pair, and pair 1b (its fallback)

Pair 1 is base `main` (`a81bae42…`) → `allenai/Olmo-3.1-7B-RL-Zero-Math` `main` (`364820b0…`). Reason: it is the
only RL-Zero-Math run whose init is logged (base revision `39ff1a89…`, whose safetensors equal base `main`), and
the only RL-Zero-Math model with any published MATH-500 number (0.874 to 0.908, §0.2); Table 49 describes its
training run. The 3.0 model has no published MATH or MATH-500 number, only AIME, and its init is not logged
publicly. The 3.1 band is used only as a sanity band (S-BAND, §6.1), because the prompt behind it is not public.

Pair 1b is base `main` → `allenai/Olmo-3-7B-RL-Zero-Math` `main` (`dee26a40…`). It replaces pair 1 only if pair 1
fails G-INIT or G-PARTITION at R1, if the 3.1 template fails G-TEMPLATE, or if vLLM loads the 3.0 checkpoint but
not the 3.1 one. It does not replace pair 1 on G-GATE, G-ID or S-BAND; a G-GATE failure goes to pair 2 (§3.7).
Changes from pair 1:

- G-INIT as first registered: the chosen before-model leaves ≥ 0.20 of MLP entries bitwise unchanged in the
  3.0 model, every other candidate ≤ 0.05; failing input: the other candidate (`stage2-step47684` if `main`
  passes). If `stage2-step47684` passes instead, it becomes the base for every arm. If neither passes: CANCELLED
  ("pre-RL model not identified").
- Prompt: the 3.0 tokenizer and template at `dee26a40…` (§2.2, last paragraph).
- No S-BAND. The stamp reads "pair 1b: no published MATH-500 number for the RL side".
- Launch: a second S2 launch (`s2setup-NN`) with the 3.0 download, R1 and the smoke, about 1.8 to 2.0 GPU-h
  (the first registration's S2 without R0). R0 is not repeated; it reads only the base.
- The same arms, gates, labels, cap rule and priority order. Budget: §9.7, the pair 1b totals.

---

## 4. Data

### 4.1 Manifest

| object | identity | size |
|---|---|---|
| base | `allenai/Olmo-3-1025-7B` @ `a81bae42db3975be1671e27b9c9a56da1a9f980f`, three shards (hashes §0.2) | 14.6 GB (`00_ASSETS.md` §1.6) |
| other init candidate | `allenai/Olmo-3-1025-7B` @ `c3c800dc900f3fecc112e8cd6b2a13edabc096e1` (`stage2-step47684`) | 14.6 GB |
| RL, pair 1 | `allenai/Olmo-3.1-7B-RL-Zero-Math` @ `364820b002fc8e81d11a81c7fc9dd6b49c83cd05` (shard hashes recorded at freeze 1, §0.3 item 9) | 14.6 GB |
| RL, pair 1b (only if it runs, §3.8) | `allenai/Olmo-3-7B-RL-Zero-Math` @ `dee26a4074b88215cd1326472e1ab14d0259d892` | 14.6 GB |
| Minerva MATH (R0 only) | `EleutherAI/hendrycks_math`, split `test`, 5,000 rows over 7 subjects (dataset sha `21a5633873b6a120296cce3e2df9d5550074f4a3` on 2026-11-05), loaded by olmes itself; R0 records the revision it loaded and asserts 5,000 rows | small |
| pair 2 | `Olmo-3-7B-Think-DPO` @ `7b18bf92…`; `Olmo-3-7B-Think` @ `d97e442d…` | 14.6 GB each |
| MATH-500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2…`, `test.jsonl`, 500 rows | small |
| GSM8K | `openai/gsm8k` @ `740312ad…`, `main/test-00000-of-00001.parquet`, 1,319 rows | small |

Every downloaded file's sha256 is checked against the HF tree API's LFS oid at the pinned commit. A mismatch
stops the job.

### 4.2 Splits

There is no discovery half. No arm, k or threshold is chosen from benchmark data. The only data-dependent
choices are the cap C (from 20 RL smoke answers) and σ, δ, H, Q (from the four gate runs). All are frozen in
`thresholds.json` before any arm runs. GSM8K is the held-out readout of the primary contrast.

### 4.3 Forward-condition parity

One engine, one tokenizer (the RL repo's), one template, one sampling configuration, one cap, one grader
version for every arm of a pair. The only thing that differs between arms is the weight files. The `base`
and `rl` endpoints are loaded from the HF snapshots; `zero` and `id` from builder output. G-HASH shows that
these are the same tensors.

### 4.4 Contamination

The report decontaminates Dolci RL-Zero against evaluation data and finds that RL with random rewards gives
no gain on its benchmarks (§6.2, Figure 27). That is evidence, not proof, that MATH-500 is not memorized.
The transplant compares arms on the same problems, so contamination would raise all arms that carry the
relevant weights; it cannot by itself make top_k beat rndT_k. Stated as a limitation.

---

## 5. Controls, each tied to a claim

| control | type | claim it serves | expected | failing input |
|---|---|---|---|---|
| G-GATE: rl-A − base-A ≥ 5√2σ | positive control of the readout: it detects the known RL gain | every arm reading | passes | (rl-A, id-B) as if it were a stage pair: must fail |
| PC-RANK: 1,000 planted neurons | positive control of the ranker | S_k is the set it claims | ≥ 0.99 of planted neurons in the top 4,523 | planted along rows of `down_proj` (hidden dimensions, not neurons): recovery ≤ 0.05 |
| PC-ART: a synthetic rounding-only change | positive control of the H_artifact instrument | H_artifact verdict | J_round(synthetic) < 0.4 | the real change: reported, not a gate |
| `zero`, `id` | identity pairs, weight level | the builder | bitwise equal to base and RL | `id` with neuron (0, 0) reverted: hash mismatch |
| (rl-A, id-B), (base-A, zero-B) | identity pairs, readout level | the harness is reproducible | \|difference\| ≤ 3√2σ | (base-A, rl-A) as if an identity pair: must fail |
| `rndE_k`, `rndT_k`, `rnd0_1%` | structured random controls from the same operator, matched on count and energy | H_local's "special set" | below top_k if H_local holds | — |
| `dare_k` | the strongest random-entry control at the same parameter count | H_diffuse | within δ of top_k if H_diffuse holds | — |
| second draws at 1% | draw-to-draw spread | the random arms are typical | within δ of the first draw | — |
| `top_1%_B` | seed spread of an intermediate model | δ | σ_mid ≤ 1.5σ | — |
| G-KNOWN | known number on real weights, under the report's own harness (olmes `minerva_math::olmes:n4:v2`, vLLM 0.11.0) | the base weights, the download and the GPU | base Minerva MATH `primary_score` within 1.6 points of 40.0 | the same predictions rescored against a derangement of the golds: must fall below 10.0 |
| S-BAND (pair 1) | sanity band for the identity transplant under our harness | the RL side of the study's harness (§2.2–2.4) | L' (id-B) within [81.5, 96.7] | the `id-B` generations scored against a derangement of the golds: must fall below 81.5 |

PC-RANK construction. Draw 1,000 neurons (seed 40) uniformly from those with s below the median. In an
independent copy of the change, multiply each planted neuron's gate row, up row and down column by
f_n = 1.5 · s_(3,523) / s_n, written with explicit slicing (`dg[n, :]`, `du[n, :]`, `dd[:, n]`), not with the
builder's helper. Rank the copy with the ranker.

PC-ART construction. A synthetic change on the real base weights: each MLP entry moves one bf16 step, with
random sign, with probability 0.3, independent of its neuron. Its ranking is set by the spacing, so the
sub-floor redraws must scramble it.

**The three numbers stamped before unblinding** (`verdict_n13.json` → `three_numbers`):

1. Fit on both sides: G-KNOWN (base side, Minerva MATH `primary_score` against 40.0); the RL side, \|L − L'\|
   against 3√2σ with the `id` hash equal to RL's, plus S-BAND (pair 1: L' against the published 3.1 band; pair
   1b: stamped "no published MATH-500 number", §0.3 item 3).
2. A positive control that fired: G against 5√2σ; PC-RANK recovery.
3. The value on an identity pair: L − L' and B − B'.

If any of the three is missing, the verdict is CANCELLED and the missing item is named.

---

## 6. Gates, frozen rules and the verdict

### 6.1 Gates

Each gate has a pass condition, a failing input that must make it fail, and the branch taken when it fails.
The arm readout has a floor and a ceiling: the floor is B (the zero transplant reads B'), the ceiling is L (the
identity transplant reads L'). A reading more than δ below the floor is DAMAGED; more than δ above the
ceiling is ABOVE (§6.3). The cancelled branches are the last column.

| gate | where | pass | failing input | if it fails |
|---|---|---|---|---|
| G-GRADER | offline gate and the first CPU step of S2 | planted set (§9.3): all 24 equivalent pairs true, all 16 non-equivalent pairs false; every MATH-500 gold judged equal to itself (500/500); golds against a derangement (seed 0) judged equal at a rate ≤ the rate of identical normalized strings + 0.01 | the derangement itself, scored as if it were the identity: must fail | before freeze: fix and record. After freeze: CANCELLED. |
| G-INIT (pair 1) | R1, weight-only | (a) hash check: the three downloaded base `main` shards have sha256 equal to the LFS oids at revision `39ff1a89…` (the logged 3.1 init), read from the HF tree API at run time; **and** (b) the share of MLP entries bitwise unchanged between base `main` and 3.1 `main` exceeds the share between `stage2-step47684` and 3.1 `main` by ≥ 0.10 | (a) the `stage2-step47684` shard hashes in place of `main`'s; (b) the two candidates swapped | pair 1 → pair 1b (§3.8), which runs G-INIT as first registered; pair 1b also fails → CANCELLED ("pre-RL model not identified") |
| G-PARTITION | R1 | the parameter names of §0.3 item 5 cover every tensor; the four families sum to 7,298,011,136; the shapes match §0.5 | an index with one MLP tensor removed | stop; fix the code |
| PC-RANK | R1 | ≥ 0.99 | rows of `down_proj`: ≤ 0.05 | CANCELLED until fixed (ranker defect) |
| G-KNOWN | R0 | olmes `minerva_math::olmes:n4:v2` on base `main` with vLLM 0.11.0: \|100 × `primary_score` − 40.0\| ≤ 1.6 points (`primary_score` of the aggregate task, the macro mean over 7 subjects; never `pass_at_1_micro`) | the R0 predictions rescored by the same olmes scorer against a derangement of the golds within each subject (seed 0): must score < 10.0 and so fail | CANCELLED ("base does not reproduce Table 3") |
| G-TEMPLATE | every generation step | §2.2 assertions | `add_generation_prompt=False`: must differ | stop; fix |
| G-HASH | every built arm | `zero` = base and `id` = RL, every tensor sha256 | `id` with neuron (0, 0) reverted | CANCELLED until fixed; no arm data is read before it passes |
| G-MASK | every built mask arm | realized changed rows and columns equal S ∩ {nonzero change} | `top_1%` built with `down_proj` rows instead of columns | stop; fix |
| G-COMPLETE | every arm | 500 (or 1,319) answers, every index once, seeds as registered, no empty text | a bank with one answer removed | the arm is re-run under a new attempt |
| G-GATE | S3 | G ≥ 5√2σ **and** G' = L' − B' ≥ 5√2σ **and** exact McNemar (rl-A vs base-A) p < 0.001 | (rl-A, id-B) treated as the pair | pair 1 → pair 2. Pair 2 also fails → CANCELLED ("gain under the noise gate") |
| G-ID | S3 | \|L − L'\| ≤ 3√2σ and \|B − B'\| ≤ 3√2σ | (base-A, rl-A) treated as an identity pair | CANCELLED ("identity transplant does not reproduce the released score") |
| S-BAND | S3, pair 1 only | 81.5 ≤ L' ≤ 96.7 (L' = acc(id-B)); L is reported beside it | the `id-B` generations scored against a derangement of the golds (seed 0): must fall below 81.5 | not CANCELLED (the prompt behind the band is not public). Stop before freeze 2; read 20 `id` answers in full; check G-TEMPLATE, extraction modes, cap-hit share and the grader against §2.2–2.4; write the finding in the journal. A defect found: fix under a dated amendment and relaunch S3 once. None found: the run goes on, and every artifact carries the stamp "RL side outside the published 3.1 band under this harness" with L' and the band. |
| G-GATE-GSM | S3 | the G-GATE formula on GSM8K, with its own σ | as G-GATE | GSM8K reported as "no room"; the MATH-500 verdict stands |

Ceilings. G-GATE has no upper limit on G, but if L ≥ 98 or B ≤ 2, the ceiling or floor of the scale is
reached and the stamp says so; the rules still run.

G-KNOWN, its quantity and tolerance. G-KNOWN runs the task behind the 40.0 itself: Minerva MATH on 5,000
problems, 4 shots, 4 samples, 20,000 generations of at most 1,024 tokens (§0.2). It does not run MATH-500, and its
score is never compared with a MATH-500 accuracy. Install and command (olmes at `5a51f502…`, its own venv
`/opt/olmes`, separate from `/opt/n13`):

```
pip install -e ".[gpu]"      # vllm==0.11.0, lm_eval==0.4.3, antlr4-python3-runtime==4.11
olmes --model allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f --model-type vllm \
      --task minerva_math::olmes:n4:v2 --gpus 1 --output-dir out/r0_known/olmes
```

The read is `primary_score` of the aggregate task `minerva_math::olmes:n4:v2`, times 100. The tolerance is 1.6
points: three times the sampling standard error of the difference of two independent readings of this score.
One reading has standard error at most 0.374 points (p(1 − p) ≤ 1/4 per problem, 4 samples, subject weights 1/7,
subject sizes 1,187, 474, 479, 903, 540, 871, 546); the difference has at most √2 × 0.374 = 0.53; 3 × 0.53 = 1.59,
rounded to 1.6. The report's second reading of the same checkpoint, 39.8 (Table 13), lies 0.2 from 40.0. The
tolerance covers sampling only (olmes sets no generation seed); a larger shift from a wrong alias, wrong shots, a
chat template or engine numerics is what the gate exists to catch.

MATH-500 is the study's readout for every arm, under the harness of §2.2–2.4. G-KNOWN checks the base weights, the
download and the GPU under the report's harness. It does not check the study's harness. That is checked by
G-GRADER, G-TEMPLATE, G-ID and, for pair 1, S-BAND.

S-BAND, the band. The published 3.1 MATH-500 readings span 87.4 to 90.8 (15 evaluations, §0.2). The band is that
span widened by 5.9 points on each side, [81.5, 96.7]: three times the standard error of the difference of two
single-sample readings at p = 0.89 (one reading: 100 × √(0.89 × 0.11 / 500) = 1.40 points; the difference:
1.98; × 3 = 5.94). The margin covers sampling only. The published readings also differ from ours in temperature
(1.0 against 0.6), length cap (32,768 against C), grader (`exact_match_flex` against `math-verify`) and the user
message (not public). So a miss is a reason to inspect the harness, not proof of a defect, and the band is not an
exact reproduction target.

G-INIT for pair 1, the margin in (b). The logged init settles which checkpoint the 3.1 run started from; (a)
checks that the released base `main` is that checkpoint, and (b) checks it at the weight level. Between
`stage2-step47684` and the RL model lies all of stage 3 training as well as RL, so that share is expected near the
level of bf16 coincidences. RL alone leaves 36% to 92% of entries unchanged (2511.08567), so the right init is
expected to give a difference far above 0.10. The 3.1 run is longer than the 3.0 run (branches to `step_2800`),
so the absolute floor of 0.20 in the first registration (kept for pair 1b) is replaced here by this comparison.

### 6.2 Thresholds, frozen at freeze 2

Written by `studies/next/n13/scripts/freeze_thresholds.py` from the four gate banks into
`studies/next/n13/thresholds.json`: C, σ_base, σ_rl, σ, δ, B, L, G, B', L', G', H, Q, the GSM8K counterparts,
and the sha256 of each gate bank. Committed and tagged `n13-freeze-2` before the first arm launches. No arm
may launch from a commit without that tag.

Room. G ≥ 5√2σ = 2.5δ, so H − B = G/2 ≥ 1.25δ. An arm at H is more than δ above the base, and the halfway
rule can separate "recovers" from "base". A comparison between two arms at B and at L spans 2.5δ, so "beats
by more than δ" has room. This is checked by the script at freeze 2 and written to `thresholds.json` →
`room`.

`top_1%_B` gives σ_mid after freeze 2. If σ_mid > 1.5σ, both readings (δ from σ and δ from σ_mid) are reported
under a dated exploratory deviation. The frozen δ decides.

### 6.3 Labels and the verdict, as the script runs it

```
# studies/next/n13/verdict_n13.py; inputs: the banks, thresholds.json (tag n13-freeze-2), R1 outputs
for gate in every gate of section 6.1:
    if not gate.failing_input_failed: return CANCELLED(gate, "its failing input did not fail")
# pair 1 failing G-INIT or G-PARTITION at R1, or G-TEMPLATE, was replaced by pair 1b before S3 (section 3.8);
# the tree below runs on the pair that passed
for gate in [G-GRADER, G-INIT, G-PARTITION, PC-RANK, G-KNOWN, G-HASH, G-MASK, G-COMPLETE]:
    if not gate.pass: return CANCELLED(gate)
if not G-GATE.pass: run the same tree on pair 2; if pair 2 fails G-GATE: return CANCELLED("gain under the noise gate")
if not G-ID.pass: return CANCELLED("identity transplant does not reproduce the released score")
if pair 1 and not S-BAND.pass: stamp("RL side outside the published 3.1 band under this harness")   # no cancel
if pair 1b: stamp("pair 1b: no published MATH-500 number for the RL side")

label(a):                                    # a = one arm on MATH-500, seed set A; not the four endpoints
    DAMAGED  if screen fired (>= 15 of problems 0-19 hit the cap and <= 1 of them correct) or acc(a) < B - delta
    ABOVE    if acc(a) > L + delta
    REC      if acc(a) >= H
    PART     if Q < acc(a) < H
    NONE     if acc(a) <= Q (and not DAMAGED)
BEATS(a, b):   acc(a) - acc(b) > delta  and  BH-adjusted McNemar p < 0.05 (primary family, section 7)
WITHIN(b, a):  acc(b) >= acc(a) - delta                     # b does as well as a, or better

k_rec = smallest k in [0.1, 0.3, 1, 3] % with label(top_k) in {REC, ABOVE}; else NONE

H_local:
    REFUTED       if k_rec is NONE
    REFUTED       elif WITHIN(dare_k, top_k) or WITHIN(rndE_k, top_k) or WITHIN(rndT_k, top_k)   at k = k_rec
    INDETERMINATE elif label(rndT_k) is DAMAGED                                                  at k = k_rec
    SUPPORTED     elif BEATS(top_k, dare_k) and BEATS(top_k, rndE_k) and BEATS(top_k, rndT_k)    at k = k_rec
    INDETERMINATE otherwise
    # a BEATS over a DAMAGED dare_k or rndE_k counts, flagged "control damaged"
H_diffuse:
    DW_k  = label(dare_k) in {REC, ABOVE} and WITHIN(dare_k, top_k)
    DF_k  = label(dare_k) not in {REC, ABOVE} or BEATS(top_k, dare_k)        # DW_k and DF_k cannot both hold
    SUPPORTED if any DW_k (report the smallest such k);  REFUTED if DF_k for all four k;  else INDETERMINATE
H_attention:
    SUPPORTED     if label(attn) in {REC, ABOVE} and label(mlp) in {NONE, DAMAGED}
    REFUTED       if label(mlp) in {REC, ABOVE} and label(attn) in {NONE, DAMAGED}
    REFUTED       if both in {REC, ABOVE}          # "not MLP neurons" is false; labeled BOTH
    INDETERMINATE otherwise                        # NEITHER (both NONE) is labeled "interaction"; else PARTIAL
H_artifact (weight-only, k = 1%; reported at all four k):
    CANCELLED     if PC-ART J_round(synthetic) >= 0.4  # the instrument cannot see an artifact
    REFUTED       if J_floor >= 0.7 and J_round >= 0.7
    SUPPORTED     if J_floor < 0.4 or J_round < 0.4
    INDETERMINATE otherwise
Necessity (secondary): NECESSARY_k if label(comp_k) in {NONE, PART, DAMAGED}; NOT NECESSARY_k if WITHIN(comp_k, rl)
GSM8K at 1% (only if G-GATE-GSM passed), for top-dare, top-rndE, top-rndT:
    REPLICATES   same sign as MATH-500 and paired bootstrap 95% CI excludes 0
    CONSISTENT   same sign, CI includes 0
    CONTRADICTS  opposite sign, CI excludes 0
```

J_floor(k) = Jaccard(S_k by s, S_k by s'). J_round(k) = the mean over 20 redraws (seeds 60 to 79) of
Jaccard(S_k by s, S_k by s*), where s* is computed from U* = Δ + uniform(−sp/2, +sp/2) per entry, with sp the
spacing of the before value. U* is a draw of the float32 change consistent with the released bf16 values, if
the RL run started from the bf16 base and rounded to nearest at release.

Exploratory readouts, labeled as such: the layer histogram of S_k (the card's "near the output" claim: the
share of S_1% in layers 24 to 31 against 0.25); the share of change energy in attention, norms, embeddings and
LM head; the counts of unchanged, one-step and above-floor entries per family; accuracy among non-capped
answers; mean output length per arm; format rates.

### 6.4 Room to fire, checked before freeze 1

| rule | can it fire? |
|---|---|
| H_local REFUTED via k_rec NONE | yes: top_3% touches 1.8% of parameters; its accuracy can sit at B |
| H_local SUPPORTED | yes, after G-GATE: G ≥ 2.5δ (§6.2) |
| H_diffuse both ways | yes: DARE can be DAMAGED (2410.09344) or match |
| H_attention both ways | yes: `mlp` and `attn` partition 89% of parameters between them |
| H_artifact both ways | the offline gate shows J_round < 0.4 on the synthetic rounding change and J = 1 on a planted large change; R1 repeats it at real shapes (PC-ART) |
| DAMAGED | yes: a rescale of 1,000.7 on 0.1% of entries can break generation |

### 6.5 Pre-mortem: failures and the response fixed now

| failure | likely cause | response |
|---|---|---|
| G-INIT fails on pair 1 | the released base `main` is not the logged init `39ff1a89…`; or the released 3.1 weights do not descend from it | pair 1b (§3.8) |
| G-INIT on pair 1b: neither candidate leaves ≥ 0.20 unchanged | RL started from another checkpoint; or RL training saved float32-derived weights with no bitwise ties | read the open-instruct launch script (§0.3 item 2); if it names another revision, run G-INIT on it (dated amendment, weight-only); else CANCELLED |
| G-KNOWN misses | wrong task alias, wrong shots, a chat template applied, `pass_at_1_micro` read in place of `primary_score`, engine numerics | stop before S3; compare the olmes run with `07_PICKS/N13_MATH_TASK.md` line by line (task id, shots, stop sequences, no chat template, vLLM 0.11.0, the score field); one relaunch with the fix; a second miss → CANCELLED |
| vLLM will not load Olmo3 (P-ENGINE) | version | older or newer release per §0.3 item 4; if none: HF `transformers` `generate` (pin from the RL card, ≥ 4.57.0) with batch 16; the arm list is then cut by the priority order (§9.7) |
| the base ignores the `Answer:` format | base models do not follow instructions | boxed fallback already in the grader; format rate reported; the primary rule unchanged |
| G-GATE fails on pair 1 | small gain or large σ | pair 2 (§3.7) |
| G-GATE fails on pair 2 | +2.7 points published | CANCELLED; follow-up spec on IFEval (published gain +12.3, Table 15) |
| DARE DAMAGED at every k | the rescale | a result for H_diffuse at these counts; DAREx-q is the follow-up |
| `rndT` not constructible at k_rec | the top set holds more energy than the tail | labeled `rndT short`; rule unchanged |
| throughput below 400 tok/s | KV cache of a 32-KV-head model | projection recomputed; arms cut by priority |
| long or looping generations | damaged arms | damage screen (§6.3); cap C |
| an instance dies mid-arm | capacity or reaper | the arm re-runs under a new attempt; partial banks are kept and not used |
| HF download hangs | network | `HF_HUB_DOWNLOAD_TIMEOUT=60` and `retry_transport` (§9.2); a hang raises; relaunch |

---

## 7. Statistical plan

- **Absolute accuracy**: Wilson score interval, 95%, z = 1.959964, n = 500 (MATH-500) or 1,319 (GSM8K).
  Damaged arms stopped by the screen: Wilson on n = 20.
- **Arm against arm**: difference in points; exact two-sided McNemar test (binomial on discordant problems,
  p = 0.5); paired bootstrap 95% interval over problems (2,000 resamples, numpy `default_rng(0)`).
- **Primary family for Benjamini–Hochberg at q = 0.05** (14 tests): top_k vs dare_k, top_k vs rndE_k, top_k vs
  rndT_k at four k (12); `mlp` vs `attn` (1); `top_1%` vs `comp_1%` (1). Every other p-value is reported
  unadjusted and labeled exploratory.
- **Noise**: σ from the seed pairs (§0.5); δ = 2√2σ; the binomial upper bound 100·sqrt(p(1 − p)/500) is
  reported beside it (2.24 points at p = 0.5).
- **Random draws**: one draw per random arm; a second draw at 1% for `rndT` and `dare`; draw-to-draw
  difference reported against δ.
- **Effect against the strongest control**: every top_k is reported against the best of dare_k, rndE_k and
  rndT_k at the same k, never against the base alone.
- **Reported, whatever the outcome**: every arm's accuracy and interval, every label, every gate value with its
  failing-input value.

---

## 8. Threats to validity

| # | threat | effect on the result | mitigation |
|---|---|---|---|
| 1 | wrong pre-RL model (stage 3 against stage 2) | every neuron looks changed; the base line is wrong | G-INIT, weight-only, with the other candidate as failing input |
| 2 | bf16 rounding sets the ranking | "most changed" means "crossed a rounding step" | H_artifact readouts; `topF_1%` if J_floor < 0.9; PC-ART |
| 3 | energy alone | top_k beats random only because it holds more change energy (`06_CRITIC.md` N13 item 1.1) | `rndE_k` (count and energy), `rndT_k` (energy at natural scale) |
| 4 | rescaling damage | `dare_k`, `rndE_k` break the model, so top_k looks special | DAMAGED label; H_local requires beating `rndT_k`, which is not rescaled |
| 5 | format against mathematics | the base misses the `Answer:` line; the gain is format | boxed fallback; format rate per arm; accuracy among formatted answers |
| 6 | termination | an arm recovers accuracy by stopping in time (the lesson of the public Qwen3.5 edit, `JOURNAL.md` 00:45 entry) | cap-hit share and length per arm; accuracy among non-capped answers |
| 7 | σ underestimated for intermediate arms | false BEATS | BH-adjusted McNemar as a second condition; `top_1%_B` |
| 8 | one random draw | one draw lands far from the typical random set | second draws at 1% |
| 9 | grader errors | systematic mis-scoring of one arm's style | G-GRADER; manual check of 10 answers per model in the smoke; extraction modes recorded |
| 10 | engine numerics | scores differ from the published protocol | G-KNOWN on real weights under the report's harness (vLLM 0.11.0); S-BAND for the study's harness (pair 1); one engine (vLLM 0.32.0) for every arm |
| 11 | MATH-500 in the RL data | recall, not reasoning | the report's decontamination and random-reward control; GSM8K; stated |
| 12 | tokenizer and template differ between base and RL | the base sees a different prompt | one tokenizer and template for all arms; ids asserted identical |
| 13 | MLP and attention interact | neither `mlp` nor `attn` recovers while `id` does | labeled "interaction"; `comp_k` shows necessity |
| 14 | embedding and LM-head changes are in no partial arm | an unassigned share | their energy share reported; gap between `id` and `mlp`, `attn` reported |
| 15 | one stage pair, one model | scope | stated; replication is a follow-up |
| 16 | budget overrun | arms parked | cap C, damage screen, stop rule, priority order |
| 17 | the fp32 master weights are not released | the true sub-floor change is unknown | the sub-floor redraws bound its effect on the ranking; stated |

---

## 9. Runbook

### 9.1 Infrastructure checklist (read before the first launch)

- Read `.claude/skills/aws-pg/SKILL.md` and `infra/aws_pg/jobs/pipeline.py` (docstring). Run
  `./infra/aws_pg/pg ps` (profile `fuelos`, region `us-east-1` are hard-coded in `infra/aws_pg/config.py`).
- Work in worktree `../interp_research-n13`, branch `n13`. Never on the main worktree. Commit named files only.
- Launch: `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_n13/<attempt>/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n13/<attempt> --budget-min <1.3 × projected minutes> --json`.
  `--no-weights`: Olmo-3 weights are not in the bucket; jobs download them from HF at pinned commits.
- Watch: `./infra/aws_pg/pg watch results/next_n13/<attempt>`. Liveness: an instance ID from `pg ps` or an
  artifact timestamp. Poll `out/<step>/.../metrics.json` with `aws s3 cp … --profile fuelos --region us-east-1`.
  Never poll `job.log`.
- Kill: `./infra/aws_pg/pg kill <instance-id>`. The reaper kills at budget + 10 min.
- **Instance type.** `g6e.2xlarge` is enough in memory: 14.6 GB of weights and about 26 GB of KV cache on the
  48 GB L40S. It is slow for this model. Olmo-3-7B has 32 key-value heads (no grouped-query attention), so the
  cache costs 2 × 32 × 32 × 128 × 2 = 524,288 bytes per token and holds about 50,000 tokens. Decode is then
  bound by memory traffic. A larger GPU would be faster, but `p5`/`p4de` are not used by this launcher
  (aws-pg skill), and `g6e.4xlarge` has the same single L40S. So the plan stays on `g6e.2xlarge`, and wall-clock
  is reduced by running two instances in parallel in S4. That does not change GPU-hours.
- Cost and incidents go to `infra/aws_pg/operations/2026-11-<dd>_n13_<run>.md`, not to the journal.

**S3 layout.** One attempt id per launch, never reused:
`s3://pg-lineage-769063704696/results/next_n13/<attempt>/` with `<attempt>` ∈ {`s2setup-NN`, `s3gate-NN`,
`s4armsA-NN`, `s4armsB-NN`, `s5pair2-NN`}, NN = 01 for the first launch and +1 for each relaunch. Under it:

```
steps.json                      # the pipeline manifest, uploaded before launch
STATUS  job.log  heartbeat.jsonl
out/pipeline_status.json        # rewritten by pipeline.py at every step transition
out/r0_known/metrics.json  out/r0_known/samples.jsonl
out/r1_rank/{init_id.json, weight_stats.json, ranking.npz, sets.json, controls.json, metrics.json}
out/<step>/<arm>/<bench>/{generations.jsonl, samples.jsonl, metrics.json}     # bench in {math500, gsm8k}
out/<step>/<arm>/build.json     # the arm-building audit (section 3.4)
```

Inputs that later launches read (asserted by sha256): `results/next_n13/s2setup-NN/out/r1_rank/ranking.npz` and
`sets.json`; `studies/next/n13/thresholds.json` uploaded to `results/next_n13/<attempt>/inputs/`.

### 9.2 NEW and REUSED code (to be verified against the tree at freeze 1)

REUSED, checked to exist on `main` on 2026-11-05:

| object | path | use |
|---|---|---|
| launcher | `infra/aws_pg/pg`, `infra/aws_pg/pg.py`, `infra/aws_pg/config.py` | every launch |
| multi-step runner | `infra/aws_pg/jobs/pipeline.py` | one instance runs a manifest of steps |
| HF environment | `infra/pg/humboldt_core.hf_setup` (sets `HF_HUB_DISABLE_XET=1`, `HF_HUB_DOWNLOAD_TIMEOUT=60`, calls `diff_harvest.hf_online_setup`) | first call in every job |
| bounded transport retry | `infra/pg/mercator_core.retry_transport` | around `snapshot_download` |
| problem hashing | `infra/pg/alwayson_edit.row_sha256` | `problems.json` |
| last balanced box | `infra/pg/alwayson_edit.boxed_last` | extraction fallback |
| in-job pinned install pattern | `infra/aws_pg/jobs/alwayson_edit.ensure_transformers` | pattern for the vLLM venv install |

NEW (each with a `selfcheck()` on tiny tensors, run by the offline gate):

| object | path | content |
|---|---|---|
| weights instrument | `infra/pg/n13_arms.py` | bf16 ordered integers, bit distance, spacing; Δ and its exactness count; s, s', components; S_k; c_k; rndT prefix; DARE masks; sub-floor redraws; PC-RANK and PC-ART; the builder of §3.1 with the G-HASH and G-MASK audits; per-tensor sha256 |
| scoring | `infra/pg/n13_score.py` | prompt render and assertions; extraction; `math_verify` wrapper; Wilson; McNemar; BH; paired bootstrap; σ; damage screen |
| jobs | `infra/aws_pg/jobs/n13_setup.py` (venv install of `vllm==0.32.0 math-verify==0.9.0` into `/opt/n13`; olmes at `5a51f502…` with `pip install -e ".[gpu]"` into `/opt/olmes`, whose `pip freeze` is banked because olmes also pulls `ai2-olmo` from a git `main`; downloads, hash checks), `n13_known.py` (R0: the olmes command of §6.1, then reads `primary_score`, banks samples and runs the derangement rescore), `n13_rank.py` (R1), `n13_gen.py` (one arm, one or two benchmarks, the loop of §2.5), `n13_dryrun.py` (offline gate) | |
| manifests | `infra/aws_pg/steps/n13_<attempt>.json` | step lists |
| analysis | `studies/next/n13/scripts/freeze_thresholds.py`, `verdict_n13.py` (with `--failing-input <gate>` for every gate), `figure_n13.py`, `samples_coverage.py`, `recount_n13.py` (for the independent reader) | laptop only, banked JSON in |

### 9.3 The offline gate (laptop, no model forward, no download of weights)

`python -m infra.aws_pg.jobs.n13_dryrun --out /tmp/<scratch>/n13_dry` prints one line per check and
`N13_DRYRUN_OK` only when all hold. It runs in the local environment used by
`infra/aws_pg/jobs/instruments_dryrun_e2.py` (CPU torch).

1. **Stand-in weights.** A tiny random state dict with the real Olmo-3 parameter names: 4 layers, hidden 64,
   MLP width 96 (384 neurons), 4 heads and 4 key-value heads, vocabulary 512, bf16. A stand-in "RL" = base
   plus (a) 8 planted neurons with large multi-step changes, (b) one-step changes on 30% of all MLP entries,
   (c) changes to attention and norms. A stand-in "stage 2" = base plus dense large changes everywhere.
2. **Weights path.** G-INIT picks the stand-in base over the stand-in stage 2, and fails when given the stand-in stage 2. G-PARTITION passes and fails
   on its failing input. The ranker recovers the 8 planted neurons at k = 8. PC-RANK passes and its failing
   input fails. J_floor and J_round equal 1 on the planted large change and J_round < 0.4 on the synthetic
   rounding change. Every arm of §3.2 is built at stand-in sizes; G-HASH and G-MASK pass, and each failing
   input fails; `comp_k` + `top_k` reproduces `id`; DARE realized count within tolerance; `rndE` realized
   energy within 5%; `rndT` reaches E(S_k), and its "short" branch fires on a stand-in where the top set holds
   most energy. `sets.json` and `build.json` are written and read back.
3. **Scoring path.** A fake engine object with `add_request`, `step`, `has_unfinished_requests` and
   `abort_request` returns canned texts for 12 fabricated MATH-style problems: answer-line right, boxed-only
   right, wrong, no answer, cap hit, and equivalent forms (`\frac{1}{2}` against `0.5`, `\dfrac`, `10.0` against
   `10`, an interval, a degree sign). The loop writes `generations.jsonl`, `samples.jsonl` (one right and one
   wrong example present) and rewrites `metrics.json` at least three times. The damage screen fires on a
   fake arm with 16 of 20 cap hits and none correct, and not on a fake base. G-COMPLETE passes and fails.
4. **Grader.** If `math-verify` 0.9.0 is in the local environment, G-GRADER's planted set (24 equivalent, 16
   non-equivalent pairs, listed in `n13_score.PLANTED`) runs here. If not, it runs as the first CPU step of S2,
   before any download, and S2 stops if it fails.
5. **Statistics and verdict.** Wilson, McNemar, BH and bootstrap match hand-computed values on fixed inputs.
   `freeze_thresholds.py` on fabricated gate banks writes `thresholds.json` with `room` true, and with `room`
   false on banks where G = 2δ. `verdict_n13.py` on planted banks produces each of: H_local SUPPORTED,
   H_local REFUTED (k_rec NONE), H_local REFUTED (dare within δ), H_diffuse SUPPORTED, H_attention SUPPORTED,
   H_attention REFUTED, each CANCELLED branch of §6.1, and exits non-zero when a failing input does not fail.
   The R0 reader of `n13_known.py`, on a fabricated olmes output directory, returns the aggregate
   `primary_score` and stops on a directory that holds only `pass_at_1_micro`.
6. **Figure.** `figure_n13.py` renders from the planted verdict file (PNG and PDF).
7. **Manifests.** Every `steps.json` parses with `pipeline.py`'s parser; each arm appears once; the budget
   projection function returns a number for the S2 measured rates.

### 9.4 Sessions

The first line of every status report is the run/unrun/parked count over the full roster: 31 MATH-500 arm runs
(30 unconditional, 1 conditional), 11 GSM8K arm runs, R0, R1, the smoke.

**S1 — build and freeze 1 (laptop).** Create the worktree. Re-read §0.3 item 1 at olmes `5a51f502…` and record
the 3.1 repo facts of §0.3 item 9 in §0.2. Write the NEW code. Run the offline gate to `N13_DRYRUN_OK`. Verify each REUSED line against the tree. Commit
named files; tag `n13-freeze-1`; record tag, SHA and UTC in §0. Write `studies/next/n13/STATE.md`.

**S2 — setup, known number, ranking, smoke (one launch, `s2setup-01`, about 3.8 to 5.0 GPU-h).** Steps:
1. `setup`: the two venv installs (`/opt/n13`, `/opt/olmes`); G-GRADER if not run offline; downloads of base
   `main`, `stage2-step47684` and 3.1 RL `main` with hash checks.
2. `r0_known` (**the first GPU run, one known number**): olmes `minerva_math::olmes:n4:v2` on base `main` with
   vLLM 0.11.0, with that task's own prompt, shots, decoding and scorer (the command of §6.1). G-KNOWN.
   Samples: 5 right, 5 wrong.
3. `r1_rank` (CPU on the instance): G-PARTITION; G-INIT; Δ; the weight statistics; s, s'; S_k; PC-RANK; PC-ART;
   J_floor, J_round at all k; c_k, m_k, DARE masks' realized counts; the per-tensor sha256 of every arm
   (built once, then discarded); `sets.json`. Then `stage2-step47684` is deleted from disk.
4. `smoke`: `base` and `rl`, MATH-500 problems 0–19, seed set A, cap 16,384; tokens per second; peak RSS;
   free disk.

The executor then, before S3: opens the smoke samples and writes in `studies/next/n13/JOURNAL.md` the answers
to (a) is the input what §2.2 says, (b) is the output in the expected format, (c) is the scored quantity what
the model produced, with 10 answers per model read in full. Decides C by the rule of §2.3. Recomputes the
budget projection from the measured tokens per second and lengths. Writes `STATE.md`.

**S3 — gate (one launch, `s3gate-01`, about 4.1 to 6.2 GPU-h).** Steps: `base` (from HF files, seed A), `rl`
(from HF files, seed A), `zero` (seed B), `id` (seed B), each on MATH-500 and GSM8K. Then, on the laptop:
G-HASH, G-COMPLETE, G-GATE, G-ID, S-BAND (pair 1), G-GATE-GSM; `freeze_thresholds.py`; commit `thresholds.json`; tag
`n13-freeze-2`. If G-GATE fails: launch pair 2's S2 and S3 equivalents under `s5pair2-NN` (§3.7).

**S4 — arms (two launches in parallel, `s4armsA-01` and `s4armsB-01`).** Each step is one arm; the order within
each manifest is the priority order. GSM8K runs in the same engine load for the arms of §3.3.

| priority | instance A | instance B |
|---|---|---|
| P1 (decisive at 1%) | `top_1%` (+GSM8K), `dare_1%` (+GSM8K) | `rndE_1%` (+GSM8K), `rndT_1%` (+GSM8K) |
| P2 (H_attention) | `mlp` (+GSM8K) | `attn` (+GSM8K) |
| P3 (the top-k curve) | `top_3%`, `top_0.3%`, `top_0.1%` | `rndE_3%`, `rndT_3%` |
| P4 (controls at other k) | `dare_3%`, `dare_0.3%`, `dare_0.1%` | `rndE_0.3%`, `rndT_0.3%`, `rndE_0.1%`, `rndT_0.1%` |
| P5 (necessity) | `comp_1%` (+GSM8K), `comp_3%` | `comp_0.3%`, `comp_0.1%` |
| P6 (spread) | `rnd0_1%`, `top_1%_B`, `dare_1%_d2` | `rndT_1%_d2` |
| P7 (conditional) | — | `topF_1%` if J_floor(1%) < 0.9 |

Instance A has 14 steps; instance B has 12, or 13 with the conditional arm. While they run, the executor opens the first finished
`samples.jsonl` of each instance and records the three answers of S2 in the journal. This does not change any
rule.

> **Head's note, 2026-11-06 (sent about 18:30Z; restated in the head's message received 18:35Z, the source; recorded by the tenth N13
> executor; not an amendment: no rule changes).** S4 runs in the four launches of A28 from commit `9cf5e73c6` (code of `n13-freeze-2`).
> The n13 environment of `s4armsA2-01` differs from S3's in three transitive packages that `n13_setup` does not pin: click 8.5.0 →
> 8.5.1, jiter 0.18.0 → 0.19.0, typer 0.27.3 → 0.28.0. A1, B1 and B2 may install the same.
> 1. The environment is left as it is. Pinning now would make A1, B1 and B2 run a different code commit than A2, which is the larger
>    difference. click, typer and jiter are not on the LLMEngine generation path.
> 2. Every arm's engine block is compared with S3's as it lands: vllm, torch, triton, transformers and tokenizers versions,
>    `max_num_seqs`, `max_num_batched_tokens`, `num_gpu_blocks`, attention backend, sampler environment, and the `engine_args` (every
>    key but `model`, the weights path, which differs by arm by design). Any difference is reported to the head at once, and the
>    instance keeps running: the executor never terminates an instance; the head decides whether to terminate it. An arm with a
>    differing engine does not count, whatever its output (the head, message received 18:37Z).
>    The check is `studies/next/n13/s4/tools/engine_check.py` (reference: `s3gate-03` base). It flags a planted difference
>    (`num_gpu_blocks` 12,816, seed 1, torch 2.13.1; exit 1) and passes the S3 identity pair (`zero` against `base`; exit 0). Each
>    comparison is in `studies/next/n13/s4/<launch>/<arm>/engine_check.json`. Operations: `infra/aws_pg/operations/2026-11-06_next_n13_s4.md`.

**S5 — unblind once, close (laptop).** Run `verdict_n13.py` once on the complete banks (or on the banks at the
stop rule, with parked arms listed). Run `figure_n13.py`, `samples_coverage.py`. Write the claims record and
the registry. Then an independent reader, a model other than the executor (repo Provenance rule 3), recounts
the numbers that matter with `recount_n13.py` and writes `studies/next/n13/READ_n13.md`. The result document
is written after that read.

A session ends only at a launch boundary and only after `STATE.md` holds: the run/unrun/parked counts, open
instance IDs, the exact next action, and the resume steps.

### 9.5 Blind protocol

- Thresholds and the cap are frozen (freeze 2) before any arm runs.
- During S4 the executor reads samples and `metrics.json` status fields, not arm-against-arm comparisons.
  No rule, arm or order changes after the first arm's accuracy is written.
- `verdict_n13.py` runs once. A rule found wrong afterwards gets a dated exploratory deviation in the journal;
  the frozen rule still decides.

### 9.6 Journal and handoff

`studies/next/n13/JOURNAL.md` starts at the first launch: one line per milestone (date, what changed, why, what
is open). The author's prior (§1.5) is copied into it before S3. Never backfilled. `STATE.md` is the handoff
file. Infrastructure and cost go to `infra/aws_pg/operations/`.

### 9.7 Budget and the stop rule

Assumptions, each marked. Throughput is my estimate, not a sourced number: about 800 output tokens per second
for vLLM on one L40S for this model (from the KV-cache size in §9.1; `04_PRACTICAL.md` §0 assumed 2,500 for a
generic 7B, which does not account for 32 key-value heads); the low case is 500. Mean output lengths are
assumptions. For pair 1 (3.1): 2,500 tokens on MATH-500 and 940 on GSM8K. These are the first registration's
1,600 and 600 (set for 3.0) times 1.56, the ratio of the end-of-run training-rollout lengths of 3.1 and 3.0
(12,756 against 8,181 tokens, §0.2; the 3.1 restart run gives 1.31). No MATH-500 length is published for either
model, so the 3.1 lengths are to verify at run time: S2 measures lengths and throughput and the projection is
redone before S3. R0: 20,000 generations at a mean of 250 tokens (an assumption; the cap is 1,024 and a blank line
ends a sample), 5.0 M tokens, at the same throughput figures.

| launch | runs | answers | output tokens | GPU-h at 800 tok/s | GPU-h at 500 tok/s |
|---|---|---|---|---|---|
| S2 setup, R0, R1, smoke | 1 instance; R0 (one olmes load), R1, 2 smoke loads | 20,000 R0 generations; 40 smoke | 5.0 M (R0); smoke small | 3.8 | 5.0 |
| S3 gate | 4 loads | 2,000 MATH-500 + 5,276 GSM8K | 5.0 M + 5.0 M | 4.1 | 6.2 |
| S4 arms | 2 instances, 27 loads | 13,500 MATH-500 + 9,233 GSM8K | 33.8 M + 8.7 M | 17.6 | 26.5 |
| **total** | 34 loads | 50,049 | 57.4 M | **25.5** | **37.7** |

S2 is the first registration's 1.8 and 2.0 GPU-h (setup, R1, smoke) plus R0: 5.0 M tokens at the throughput, 10
minutes for the olmes venv (an assumption) and 5 minutes for its engine load, so 2.0 and 3.0 GPU-h. Load overhead
is 5 minutes per arm (build, write 14.6 GB, start the engine). Instance setup is about 20 minutes (venv install,
two 14.6 GB downloads). If MATH-500 and GSM8K answers are 25% longer than assumed, the totals become about 30 and
45. At the low throughput the projection (37.7, or 45 with longer answers) is above the 35 GPU-h ceiling, and the
stop rule parks the last steps of each manifest, which are the lowest priorities. For pair 1b, with the first registration's
lengths (1,600 and 600) and R0 included, the totals are about 19 and 27 GPU-h.
**Target: 20 to 25 GPU-h. Ceiling: 35 GPU-h.** Both are unchanged by the amendment; the pair 1 projection lies
above the target.

Stop rule. Before each launch the executor checks that spent GPU-h (the sum of instance uptimes, recorded in
the operations note) plus the launch's projection stays ≤ 35. Each launch gets an allowance, its share of what
remains, written into `steps.json`; `--budget-min` is set to that allowance. Before each step, the job compares
its own elapsed time plus the step's projection (from the measured tokens per second and lengths) with the
allowance. A step that would exceed it is PARKED, with every later step in that manifest. The verdict runs on
what exists. A comparison whose arms are parked is INDETERMINATE and listed.

Pair 2, if it runs, starts with whatever remains under 35 after pair 1's S2 and S3 (about 7.9 to 11.2 GPU-h
spent). Think answers are longer, so fewer arms fit. The priority order decides which.

---

## 10. Artifacts

### 10.1 Samples, at a fixed path beside each `metrics.json`

S3: `results/next_n13/<attempt>/out/<step>/<arm>/<bench>/samples.jsonl`. Local mirror after each launch:
`studies/next/n13/results/<attempt>/<arm>/<bench>/samples.jsonl`, next to that arm's `metrics.json`.

Each file holds, for that arm and benchmark:
- problems 0, 1 and 2: the same three items through every arm;
- the lowest-index problem scored right and the lowest-index problem scored wrong;
- one cap-hit answer, if any;
- for R0: 5 right and 5 wrong.

Each row: arm, pair, benchmark, problem index, `unique_id`, the prompt exactly as the model saw it (text, number
of tokens, first and last 20 ids), the full generated text, `n_new`, `cap_hit`, `finish_reason`, extraction
mode, the extracted answer, the gold, the `math_verify` result, the score, the answer line's summed
log-probability and its exponential, the mean token log-probability, the seed. `generations.jsonl` holds every
answer with the same fields, for the recount.

`studies/next/n13/results/SAMPLES_COVERAGE.md` lists every arm × benchmark of the roster as present or
PENDING. No number from an arm is written down before its samples were opened (§9.4).

### 10.2 Registry and records

- `studies/next/n13/REGISTRY.md`: every number and figure → generating script, input sha256, CONFIRMATORY
  (the rules of §6.3) or EXPLORATORY (§6.3 last paragraph, σ_mid, layer histograms).
- `studies/next/n13/CLAIMS.md`: one line per load-bearing claim, with ID and status; prose cites the ID.
- `verdict_n13.json`: every gate with its value and its failing input's value; `three_numbers`; every arm's
  accuracy, Wilson interval, label, counts, format rates, mean length, cap-hit share; every comparison with
  difference, McNemar p, BH-adjusted p, bootstrap interval; the hypothesis verdicts; parked arms.
- Regenerate every figure and table after any correction.

### 10.3 The picture

One chart. MATH-500, pair 1 (or pair 1b or pair 2 if it ran instead).

- x axis: k, as a percent of MLP neurons, log scale, ticks 0.1, 0.3, 1, 3, then a separate tick "all MLP"
  for `mlp`, and after a gap a separate tick "attention + norms" for `attn`.
- y axis: absolute MATH-500 accuracy in percent, 0 to 100.
- One line per arm type: `top_k`, `comp_k`, `rndE_k`, `rndT_k` (each point annotated with m_k), `dare_k`.
  Each point has its Wilson 95% bar. Damaged arms are drawn as × at their screen accuracy.
- Horizontal lines: `base` (B) and `rl` (L), seed set A, with their Wilson bands shaded; seed-B readings
  (B', L') as thin lines. A thin dotted line at H, labeled "halfway (frozen rule)".
- Title: the finding, one sentence, filled at closeout. Caption: n = 500, one sample per problem, temperature
  0.6, top-p 0.95, cap C, seed set A, grader `math-verify` 0.9.0, the before and after commits. Footer: attempt
  IDs, commit, `figure_n13.py`, sha256 of `verdict_n13.json`.
- `studies/next/n13/figures/fig_n13.png` and `.pdf`, rendered from `verdict_n13.json` only.

GSM8K and the weight statistics go in tables, not in a second chart.

---

## 11. Prediction grid and results template (filled at unblinding, whatever the result)

Filled 2026-11-08 by the twelfth N13 executor from `studies/next/n13/verdict_n13.json` only (sha256 `cae55c04ebf8…`; verdict COMPLETE, run once at 10:51Z on the S5 tree whose manifest sha256 is `56c87b3c06c4…`). Generator: `studies/next/n13/s5/grid11.py`. Values not in the verdict file are marked so. c_k and m_k are in each arm's `build.json` (`studies/next/n13/s4/<launch>/<arm>/arm_record.json`), not in the verdict file. Differences are percentage points of accuracy (first arm minus second).

| item | prediction or rule | observed | verdict |
|---|---|---|---|
| G-INIT | `main` is the init (pair 1: hash check against `39ff1a89…` and the share comparison) | pass; share difference 0.557; failing input failed | PASS |
| G-KNOWN | base Minerva MATH (`minerva_math::olmes:n4:v2`, `primary_score`) within 1.6 points of 40.0 | 39.96 (failing input 1.12 failed) | PASS |
| S-BAND (pair 1) | L' within [81.5, 96.7] (prior: L 80 to 92) | L' 91.2; L 92.0 | PASS |
| G-GATE | G ≥ 5√2σ (prior: G 15 to 35 points) | G 34.8 ≥ 13.60 (σ 1.924) | PASS |
| G-ID | \|L − L'\| ≤ 3√2σ | 0.8 ≤ 8.16; B − B' 1.8 | PASS |
| unchanged share of MLP entries | prior 0.4 to 0.9 | not in `verdict_n13.json` (R1 `weight_stats.json`); not filled here | — |
| H_artifact | J_floor, J_round at 1% | J_floor 0.642, J_round 0.534; PC-ART rule(P+) SUPPORTED, rule(P−) REFUTED | INDETERMINATE |
| k_rec |  | NONE: no top_k labeled REC or ABOVE | NONE |
| H_local | §1.3 | k_rec NONE | REFUTED (k_rec NONE) |
| H_diffuse | §1.3 | dare_k REC at every k; top − dare -27.6, -25.4, -24.0, -22.4 (k 0.1, 0.3, 1, 3) | SUPPORTED (smallest k k0.1) |
| H_attention | §1.3 | mlp 86.4 REC, attn 83.0 REC | REFUTED (BOTH) |
| necessity at k_rec |  | k_rec NONE; at all four k: k0.1 NOT NECESSARY, k0.3 NOT NECESSARY, k1 NOT NECESSARY, k3 NOT NECESSARY | NOT NECESSARY (all k) |
| GSM8K at 1% |  | G-GATE-GSM 28.7, pass; top_k1 − dare_k1: -23.7 [-26.5, -21.1] REPLICATES; top_k1 − rndE_k1: +2.1 [+0.2, +3.9] REPLICATES; top_k1 − rndT_k1: +1.9 [+0.1, +3.6] REPLICATES | REPLICATES (3 of 3) |

| arm | acc (%) | Wilson 95% | label | cap-hit share | `answer_line` share | mean tokens |
|---|---|---|---|---|---|---|
| base | 57.2 | [52.8, 61.5] | - | 0.14 | 0.44 | 3350 |
| rl | 92.0 | [89.3, 94.1] | - | 0.03 | 0.00 | 4499 |
| zero | 55.4 | [51.0, 59.7] | - | 0.13 | 0.39 | 3280 |
| id | 91.2 | [88.4, 93.4] | - | 0.03 | 0.00 | 4333 |
| top_k0.1 | 57.6 | [53.2, 61.9] | NONE | 0.15 | 0.43 | 3545 |
| top_k0.3 | 61.4 | [57.1, 65.6] | NONE | 0.13 | 0.45 | 3347 |
| top_k1 | 61.8 | [57.5, 66.0] | NONE | 0.16 | 0.43 | 3715 |
| top_k3 | 63.8 | [59.5, 67.9] | NONE | 0.17 | 0.44 | 4004 |
| comp_k0.1 | 90.4 | [87.5, 92.7] | REC | 0.06 | 0.00 | 4671 |
| comp_k0.3 | 90.8 | [87.9, 93.0] | REC | 0.05 | 0.00 | 4570 |
| comp_k1 | 91.6 | [88.8, 93.7] | REC | 0.05 | 0.00 | 4523 |
| comp_k3 | 90.6 | [87.7, 92.9] | REC | 0.05 | 0.00 | 4499 |
| rndE_k0.1 | 58.8 | [54.4, 63.0] | NONE | 0.12 | 0.46 | 3148 |
| rndE_k0.3 | 59.8 | [55.4, 64.0] | NONE | 0.13 | 0.46 | 3287 |
| rndE_k1 | 59.2 | [54.8, 63.4] | NONE | 0.13 | 0.42 | 3324 |
| rndE_k3 | 63.8 | [59.5, 67.9] | NONE | 0.13 | 0.43 | 3363 |
| rndT_k0.1 | 59.4 | [55.0, 63.6] | NONE | 0.12 | 0.46 | 3133 |
| rndT_k0.3 | 58.0 | [53.6, 62.2] | NONE | 0.14 | 0.43 | 3366 |
| rndT_k1 | 60.0 | [55.6, 64.2] | NONE | 0.14 | 0.45 | 3438 |
| rndT_k3 | 62.8 | [58.5, 66.9] | NONE | 0.13 | 0.45 | 3346 |
| dare_k0.1 | 85.2 | [81.8, 88.0] | REC | 0.10 | 0.02 | 4607 |
| dare_k0.3 | 86.8 | [83.6, 89.5] | REC | 0.10 | 0.02 | 4569 |
| dare_k1 | 85.8 | [82.5, 88.6] | REC | 0.10 | 0.01 | 4667 |
| dare_k3 | 86.2 | [82.9, 88.9] | REC | 0.09 | 0.01 | 4456 |
| mlp | 86.4 | [83.1, 89.1] | REC | 0.08 | 0.01 | 4469 |
| attn | 83.0 | [79.5, 86.0] | REC | 0.12 | 0.04 | 4428 |
| rnd0_k1 | 59.6 | [55.2, 63.8] | NONE | 0.15 | 0.43 | 3482 |
| top_k1_B | 60.0 | [55.6, 64.2] | NONE | 0.15 | 0.41 | 3653 |
| rndT_k1_d2 | 59.6 | [55.2, 63.8] | NONE | 0.14 | 0.44 | 3399 |
| dare_k1_d2 | 87.8 | [84.6, 90.4] | REC | 0.09 | 0.01 | 4478 |
| topF_k1 | 64.0 | [59.7, 68.1] | NONE | 0.13 | 0.44 | 3435 |

---

## 12. References

Every claim in this file with a URL is listed in §0.4 with its read level. The Olmo 3 facts are from
https://arxiv.org/abs/2512.13961 (HTML read 2026-11-05), the model cards at https://huggingface.co/allenai/Olmo-3-1025-7B,
https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math, https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math,
https://huggingface.co/allenai/Olmo-3-7B-Think-DPO and
https://huggingface.co/allenai/Olmo-3-7B-Think, and the HF API reads of §0.2. The Minerva MATH task, the 3.1 init and
the published RL-Zero numbers are from `studies/next/07_PICKS/N13_MATH_TASK.md`, which cites
https://github.com/allenai/olmes at `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e` and the W&B project
`ai2-llm/Olmo-3-7B-RL-Zero`. Packages: https://pypi.org/project/vllm/0.32.0/,
https://pypi.org/project/math-verify/0.9.0/.

---

## Amendments before freeze 1, 2026-11-05

Written 2026-11-05 by the N13 executor, from the 17 spec issues of `studies/next/07_PICKS/N13_OFFLINE_GATE.md`
§"Spec issues". The text above this section is not edited. Line numbers are those of this file before this section
was appended (1,116 lines). Where a row amends the text, its new text below replaces the cited lines from freeze 1
on; the old text stays above for the record. Rows marked **head** change a registered rule or tolerance, or settle
an open decision, and wait for the head's approval. Until it is given, the registered text stands, except rows A1
and A2, which the head has already decided to amend.

**The head's decisions, 2026-11-05 13:00 CST** (recorded by the second N13 executor). A1, A2, A4, A10 and A14 are approved as
written. A3, A5 to A9, A11 to A13 and A15 to A17 are accepted. A18 is decided for option (b), under five conditions, in its
row below. The code of A2, A4, A10 and A18 is in; the offline gate was rerun after it (`N13_DRYRUN_OK`, 79 of 79;
`studies/next/07_PICKS/N13_OFFLINE_GATE.md` §"Offline gate rerun, 2026-11-05, A2 A4 A10 A18"). No row is open.

### Summary

| row | issue | spec lines | decision | needs the head | code |
|---|---|---|---|---|---|
| A1 | the grading rule parses the gold without delimiters | 337–339 | amend | approved by the head, 2026-11-05 13:00 CST | in code: `n13_score.GRADE_RULE = "dollar_both"` |
| A2 | PC-ART has no room on realistic weights and no failing input | 566, 581–583, 721, 750 | amend: replace the PC-ART rule | approved by the head, 2026-11-05 13:00 CST | in code: `n13_rank.py` stage 5, `n13_arms.plant_multistep`, `verdict_n13.py` H_artifact, `n13_dryrun.py` |
| A3 | G-KNOWN runs at olmes' default context length | 631–634 | accept as a threat; name the relaunch fix now | no | in code: R0 counts contexts over 1,024 tokens |
| A4 | the DARE kept-count tolerance fails on chance | 432 | amend: four binomial standard deviations | approved by the head, 2026-11-05 13:00 CST | in code: `n13_arms.build_arm` (`kept_sd=4.0`) |
| A5 | REUSED paths moved | 865–867, 881–883 | amend | no | in code |
| A6 | signed zeros in the builder | 380, 389–391 | amend | no | in code |
| A7 | PC-RANK's failing input is undefined | 565 | amend | no (the bound 0.05 is unchanged) | in code |
| A8 | random-number conventions | 428–430, 581 | amend | no | in code |
| A9 | MATH-500 `test.jsonl` is not an LFS file | 535–536 | amend | no | in code |
| A10 | an `Answer:` line with nothing after it | 334–336 | amend: skip empty answer lines | approved by the head, 2026-11-05 13:00 CST | in code: `n13_score.extract` |
| A11 | GSM8K label missing for one case | 726–729 | amend | no | in code |
| A12 | G-HASH has no failing input for `zero` | 567, 614 | amend | no | in code |
| A13 | G-TEMPLATE missing from the cancel list | 686 | amend | no (adds a cancel condition, no threshold) | in code |
| A14 | R1's time is not stated; where R1 runs | 931–933, 1001 | amend: R1 in the S3 launch | approved by the head, 2026-11-05 13:00 CST | in code: `n13_plan.manifest("s3gate")` |
| A15 | facts for §0.2 and §0.3 item 9 | 84, 87, 92, 525 | amend (facts) | no | — |
| A16 | S-BAND rounded inward by 0.04 | 574, 648–651 | accept: no reachable L' is judged differently | no | — |
| A17 | C is frozen before `thresholds.json` exists | 324–325, 665–667 | amend | no (the rule for C is unchanged) | in code: `--cap` in the S3 manifest |
| A18 | the registered olmes vLLM command cannot run on olmes' own pins (found at R0, after the 17 gate issues) | 612, 630–634; §0.3 item 1 | option (b), an engine amendment, under five conditions | approved by the head, 2026-11-05 13:00 CST | in code: `n13_known.command`, `n13_known.run` (path check), `infra/aws_pg/jobs/n13_olmes_shim/sitecustomize.py` |
| A19 | R0 ran at olmes' batch size 1 (found at R0 attempt 6), and the R0 park projection assumed batching | 630–634, 996–997, 1016–1022; A18 | amend: `--batch-size auto`; a batched-call assertion in the path check; a measured projection gate before the full task; revised 2026-11-05: projected = F + T / r from the shim's per-call times and tokens, elapsed from the job's start on the instance; second revision 2026-11-05: the path check at `--limit 100` (400 prompts per subject call), an early stop at batch size 1 inside the path check, T weighted by subject size | directed by the head, 2026-11-05 (briefs to the third, fourth and fifth N13 executors) | in code: `n13_known.command`, `n13_known.PATH_LIMIT`, `n13_known.project`, `n13_known.shim_calls`, `n13_known.first_call`, `n13_known.run_olmes`, `n13_known.run`, `n13_olmes_shim/sitecustomize.py`, `n13_plan.launch_t0`, `infra/aws_pg/jobs/pipeline.py` (`started_utc`), `n13_plan.manifest("s2setup")`, `n13_dryrun.py` |
| A20 | the per-block relative weight change ‖θ_RL − θ_base‖ / ‖θ_base‖ (attention and MLP per block, embedding, head), after Fel et al. 2026 (`studies/next/12_SHAPING_NOTE.md` §4 item 2) | §9.4 S2 step 3 (R1 outputs) | add, informational: no rule reads it | decided by the head, 2026-11-05 | in code: `n13_arms.weight_stats`, `n13_arms.block_profile`, `n13_rank.run` (`block_profile` in metrics.json and sets.json), `n13_dryrun.block_profile_checks` |
| A21 | the n13 engine (vLLM 0.32.0) stops at start-up: its sampler warmup JIT-compiles FlashInfer's sampling module and the AMI has no `nvcc` (found at the smoke of R0 attempt 7) | §2.3, §9.2 | amend: `VLLM_USE_FLASHINFER_SAMPLER=0` in the engine's environment; `attention_backend` pinned to FLASH_ATTN (vLLM's default on SM 8.9); no change to any n13 draw | directed by the head, 2026-11-05 (00:26Z on 2026-11-06) | in code: `n13_gen.ENGINE_ENV`, `n13_gen.ATTENTION_BACKEND`, `n13_gen.child_env`, `n13_gen.VllmEngine`, `n13_dryrun.engine_env_checks`; attempt 8 manifest `n13_plan.manifest("s2smoke")` |
| A22 | a smoke of 20 requests measures a latency-bound rate: the engine runs far below its concurrency limit and waits for its longest answer, while a gate arm sends 1,819 requests at once (found by the sixth executor before attempt 8) | §9.4 S2 step 4, §9.7 | amend: the smoke runs MATH-500 problems 0:300 per model; an engine-step log; `throughput.json` gains r_full, r_all, tail_s, load_s, lengths and the cap count; the S3 and S4 projection uses r_full, the measured MATH-500 length and load_s + tail_s per engine load | directed by the head, 2026-11-05 18:50 CST | in code: `n13_gen.StepLog`, `n13_gen.VllmEngine.step`, `n13_gen.throughput`, `n13_plan.step_rates`, `n13_plan.project_smoke`, `n13_plan.main`, `n13_plan.arm_load`, `n13_plan.SMOKE_PROBLEMS_A22`, `n13_plan.CONC_FULL`, `n13_dryrun.a22_*`; attempt 8 manifest `n13_plan.manifest("s2smoke", 8, allowance_min=110)` |
| A23 | R1's `--delete-stage2` removed the whole `models--allenai--Olmo-3-1025-7B` cache folder, base `main` with it, before the S3 arms `base`, `zero` and `id` read it (found by the Vast-lane agent, 2026-11-06) | §9.4 S2 step 3; A14 | amend (job-file fix): delete only the stage-2 snapshot, its refs entries and the blobs no other revision links to | directed by the head, 2026-11-06 | in code: `n13_rank.delete_snapshot`, `n13_rank.links`, `n13_rank.run` (`stage2_deleted` in metrics.json), `n13_dryrun.plant_cache`, `n13_dryrun.cache_state`, `n13_dryrun.a23_cache_checks` |
| A24 | the projection needs a `throughput.json` per smoke, and attempt 9's RL smoke was cut by the hard kill (142 of 300 answers) | §9.7; A22 item 4 | amend: `--killed`: the other smoke's rate and tail (flagged); L over the admitted prefix with the cut answers at C (reading `prefix`), or every unfinished answer at C (`all`) | directed by the head, 2026-11-06; the `prefix` reading waits for the head | in code: `n13_plan.killed_smoke`, `n13_plan.KILLED_READINGS`, `n13_plan.project_smoke`, `n13_plan.main`, `n13_dryrun.a24_killed_checks` |
| A25 | G-COMPLETE's "no empty text" fails on the base's real output (7 of 300 smoke answers are the stop token alone) | §6.1 line 616; §6.3 line 686 | amend: every problem has a record with n_generated ≥ 1 and a finish_reason; empty text is counted per arm (`n_empty`), not gated | decided by the head, 2026-11-06 03:50Z | in code: `n13_score.g_complete`, `verdict_n13.gather_gates` (`n_empty` per arm), `n13_dryrun.scoring_checks` (FAB row 15 and the three failing inputs) |
| A26 | mode `answer_line` misses `**Answer:** …`: right base answers scored wrong (smoke reading (c)) | §2.4 item 2 (lines 334–336, as amended by A10) | amend: the label may sit inside one markdown emphasis pair (`**Answer:**`, `*Answer:*`, `__Answer:__`) as well as alone (`Answer:`); nothing else changes | decided by the head, 2026-11-06 03:50Z | in code: `n13_score.ANSWER_LABEL`, `n13_score.extract`, `n13_score.selfcheck`, `n13_dryrun.FAB` rows 13 and 14; re-score of the smoke `studies/next/n13/amendments/a26_rescore.py` |
| A27 | the 35 GPU-h ceiling leaves no S4 tier after S3 at the measured rate | §9.7 (target, ceiling, stop rule, pair 2) | amend: no study ceiling; each launch's allowance (manifest and `--budget-min`) is 1.3 × its projection; the park order applies only to that allowance; the spent + projection ≤ 35 check removed | decided by the user, 2026-11-05 21:42 CST ("Remove the GPU limits."), passed on by the head | in code: `n13_plan` (`CEILING_GPU_H` removed, `main`, `launch_table` → `launches`), `n13_dryrun.a22_planted_checks`, `a27_a28_checks` |
| A28 | S3 as one launch and S4 as two take longer than the quota requires | §9.4 S3 and S4; §3.3; A14 | amend (layout): S3 in two launches (`s3gate-01`: setup with stage 2, R1, base, rl; `s3gate-02`: setup, zero, id), S4 in four (`s4armsA1`, `A2`, `B1`, `B2`, split by turns in priority order) | directed by the head, 2026-11-06 | in code: `n13_plan.S3_SPLIT`, `S4_SPLIT`, `LAUNCHES`, `MANIFEST_FILE`, `roster`, `project_launch`, `manifest`, `main` (`--manifest-dir`); `n13_gen.run` (zero and id read no sets file); `n13_dryrun` |

### A1. The grading rule (§2.4 item 3, lines 337–339). Amend.

**Approved by the head, 2026-11-05 13:00 CST.**

Evidence. As written, the gold is parsed as stored. On the 500 MATH-500 golds this rule judges 108 golds unequal to
themselves (392 of 500 equal), and 51 golds parse to nothing. It misses 12 of 24 planted equivalent pairs:
`[1,3)`/`[1, 3)`, `(-\infty, 2]`/`(-\infty,2]`, `2\sqrt{3}`/`\sqrt{12}`, `\frac{\sqrt{2}}{2}`/`\frac{1}{\sqrt{2}}`,
`\frac{\pi}{2}`/`\pi/2`, `(x+1)^2`/`x^2+2x+1`, `(2,3)`/`(2, 3)`, `\{1,2,3\}`/`\{3,2,1\}`, `1,2,3`/`1, 2, 3`,
`\text{(C)}`/`C`, `2^{10}`/`1024`, `\$36`/`36`. It fails G-GRADER. The amended rule passes G-GRADER: 500 of 500 golds
equal to themselves, 24 of 24 planted equivalent pairs, 16 of 16 non-equivalent pairs, a derangement rate of 0.006
against the bound 0.016. Its failing input, the derangement scored as the identity, gives 3 of 500 and fails, as it
must. A `\boxed{}` wrap of both sides also gives 500 of 500, but judges `45°` unequal to `45^\circ` (23 of 24).
Paths: `studies/next/n13/offline_gate/evidence/G-GRADER.json`; recount with the 51 empty parses,
`studies/next/n13/amendments/grader_recount.json` (script `grader_recount.py` beside it).

New text of item 3:

> 3. Score: `math_verify.verify(math_verify.parse(wrap(gold)), math_verify.parse(wrap(extracted)))` with default
>    configurations, `math-verify` 0.9.0. `wrap(x)` returns x unchanged when x holds a `$` that is not preceded by
>    `\`, or holds `\boxed`; otherwise it returns `$x$`. The same wrap is applied to the gold and to the extracted
>    string. An escaped `\$` (a currency sign in some MATH-500 golds) is not a delimiter. Code:
>    `infra/pg/n13_score.py`, `wrap(x, "dollar_both", side)`, `GRADE_RULE = "dollar_both"`.

### A2. PC-ART (§5 line 566, lines 581–583, §6.3 line 721, §6.4 line 750). Amend: replace the rule.

**Approved by the head, 2026-11-05 13:00 CST.** Code: `infra/aws_pg/jobs/n13_rank.py` stage 5 builds P− from P+ in place (`n13_arms.plant_multistep`) and runs
the rule on both (`n13_arms.artifact_rule`); `studies/next/n13/scripts/verdict_n13.py` reads `rule_P_plus` and `rule_P_minus`;
`infra/aws_pg/jobs/n13_dryrun.py` checks the new pass and failing-input fields, and that the verdict returns CANCELLED when rule(P−) is not
REFUTED.

Evidence. The registered rule is J_round(synthetic, k1) < 0.4. Each redraw adds a uniform value of width one spacing
to each entry, so a ranking set by the spacing survives the redraws. Lines 582–583 assert the opposite. With a
neuron-dependent weight scale (log-normal, sd 0.5) the rule fails on the stand-in: J_round(synthetic, k1) = 0.718
against 0.4. On iid stand-ins it gives 0.234 and 0.388 (`studies/next/n13/offline_gate/evidence/PC-ART_power.json`,
`N13_OFFLINE_GATE.md` gate table). At 3,072 entries per neuron (closer to the real 12,288), the log-normal stand-in
gives 0.903 (`studies/next/n13/amendments/pcart_room.json`). Real neuron norms vary, so the registered rule would
CANCEL H_artifact at R1. It also has no registered failing input ("the real change: reported, not a gate").

The amended control runs the whole H_artifact rule, J_floor and J_round together, on two registered changes of the
real base. Room check, offline, weight-free, at 4,096 neurons of 3,072 entries
(`studies/next/n13/amendments/pcart_room.json`, script `pcart_room.py` beside it):

| stand-in | change | J_floor(k1) | J_round(k1) | J_sp(k1) | rule |
|---|---|---|---|---|---|
| iid | P+ | 0.000 | 0.342 | 0.065 | SUPPORTED |
| iid | P− | 1.000 | 1.000 | 0.000 | REFUTED |
| log-normal neuron scale, sd 0.5 | P+ | 0.000 | 0.903 | 0.783 | SUPPORTED |
| log-normal neuron scale, sd 0.5 | P− | 1.000 | 1.000 | 0.000 | REFUTED |

New §5 row (replaces line 566):

> | PC-ART: the H_artifact rule on two registered changes of the real base | positive control of the H_artifact
> instrument | H_artifact verdict | the rule returns SUPPORTED on P+, the synthetic rounding-only change | P−, the
> same change with 3,523 planted neurons moved 8 bf16 steps on every entry: the rule must return REFUTED |

New construction text (replaces lines 581–583):

> PC-ART construction. **P+**: a synthetic change on the real base weights. Each MLP entry moves one bf16 step,
> with random sign, with probability 0.3, independent of its neuron (NumPy `default_rng([40, 1, layer, w])`, w = 0,
> 1, 2 for gate, up, down). **P−**: P+ with 3,523 neurons (k_n at 1%, drawn uniformly over all 352,256 with
> `default_rng(41)`) replaced: every entry of each planted neuron's gate row, up row and down column moves 8 bf16
> steps from its base value, with random sign (`default_rng(42)`). J_floor, J_round (20 redraws, seeds 60 to 79)
> and J_sp are computed on P+ and on P− exactly as on the real change. PC-ART passes when the H_artifact rule
> returns SUPPORTED on P+ and REFUTED on P−. J_round(P+) is the reference value of J_round on this base; it is
> printed beside J_round of the real change. A redraw does not scramble a ranking that is set by the spacing,
> because the redraw width is the spacing itself; J_floor carries the separation of P+ from P−.
>
> J_sp(k) = Jaccard(S_k by s, S_k by sp_n), where sp_n² is the sum of sp(w_before)² over the neuron's 12,288
> entries: a ranking computed from the base weights alone. Reported for the real change and for P+ at all four k.
> Exploratory; it decides nothing.

New H_artifact block of §6.3 (replaces lines 720–724):

```
H_artifact (weight-only, k = 1%; reported at all four k):
    rule(x)       = REFUTED if J_floor(x) >= 0.7 and J_round(x) >= 0.7
                    SUPPORTED elif J_floor(x) < 0.4 or J_round(x) < 0.4
                    INDETERMINATE otherwise
    CANCELLED     if rule(P+) != SUPPORTED or rule(P-) != REFUTED   # PC-ART: the instrument does not separate the two
    otherwise     rule(the real change)
```

New §6.4 row (replaces line 750):

> | H_artifact both ways | PC-ART at R1: SUPPORTED on P+ and REFUTED on P−. Offline at 3,072 entries per
> neuron: SUPPORTED (J_floor 0.000) and REFUTED (J_floor 1.000, J_round 1.000) under iid and under log-normal
> neuron scale (`studies/next/n13/amendments/pcart_room.json`) |

What H_artifact's verdict means under the amended control.

- **SUPPORTED**: the top-1% set is not kept when the one-step entries are dropped (J_floor < 0.4), or when the
  sub-floor part of the change is redrawn (J_round < 0.4). P+ shows that the rule gives this on a change made only
  of one-step entries.
- **REFUTED**: the top-1% set is carried by entries that moved two or more bf16 steps (J_floor ≥ 0.7), and it is
  kept under the redraws (J_round ≥ 0.7). P− shows that the rule gives this on a planted multi-step change.
- **What REFUTED does not show.** It does not show that the ranking is independent of the base weights' scale. On a
  rounding-only change with neuron-dependent scale, J_round is high (0.718 and 0.903 on the stand-ins above). So
  J_round ≥ 0.7 alone does not separate a ranking set by the spacing from a real one; J_floor does. J_sp(real) and
  J_round(P+) are printed beside the verdict, so a reader sees how far the real ranking follows the base weight
  scale.
- **CANCELLED**: P+ is not SUPPORTED or P− is not REFUTED. The instrument is defective; no claim about which
  neurons RL changed is made from J_floor or J_round.

Cost: one more synthetic change and 20 more redraws on the real base at R1, about the cost of the registered P+
part again (R1 is projected at 75 min in A14). Code after approval: `infra/aws_pg/jobs/n13_rank.py` stage 5 (build
P−, the rule on both, J_sp), `studies/next/n13/scripts/verdict_n13.py` (the CANCELLED line reads both rule values),
`infra/aws_pg/jobs/n13_dryrun.py` (the check reads the new pass and failing-input fields), then the offline gate is
rerun.

### A3. G-KNOWN context length (lines 631–634). Accept as a threat.

The command passes no `--model-args`. olmes then uses `max_length` 2,048 (`oe_eval/default_configs.py` line 9 at
`5a51f502`). Its vLLM path keeps only the last 2,048 − 1,024 = 1,024 context tokens
(`oe_eval/models/eleuther_vllm_causallms.py` lines 199–204). The `max_length` behind Table 3 is not public. The
registered command stays as it is; it is the olmes default. Added after line 634:

> R0 counts the 4-shot contexts longer than 1,024 tokens (`out/r0_known/rescore.json` → `context_tokens.over_1024`).
> On a G-KNOWN miss with that count above 0, the one relaunch of §6.5 adds
> `--model-args '{"max_length": 4096}'` and changes nothing else. The tolerance is unchanged.

### A4. DARE kept-count tolerance (line 432). Amend. **Head.**

**Approved by the head, 2026-11-05 13:00 CST.** Code: `infra/pg/n13_arms.build_arm(..., kept_sd=4.0)`; `n13_gen --kept-sd` and `n13_rank --kept-sd`, default 4.

The registered assertion "within 0.1% of P_k" is 2.08 binomial standard deviations at k = 0.1% (P_k = 4,325,376,
sd 2,078.7). One draw then fails by chance with probability 0.037. The seeds are fixed (30 + j, 130 + j), so a draw
that fails fails at every relaunch, and `n13_gen` asserts the tolerance before the engine starts
(`infra/aws_pg/jobs/n13_gen.py` lines 308–309). The arm could then never run. At 0.3%, 1% and 3% the same tolerance
is 3.6, 6.6 and 11.6 sd. New text (replaces "asserted within 0.1% of P_k" on line 432):

> asserted |K − P_k| ≤ 4·sqrt(P_k·(1 − q_k)), four binomial standard deviations, where K is the realized kept
> count and P_k = q_k × 4,328,521,728 = k_n × 12,288 the expected one (tolerances 8,315, 14,394, 26,186 and 44,893
> entries at k = 0.1%, 0.3%, 1%, 3%)

A draw then fails by chance with probability 6.3e-5. A kept rate off by 1% of q_k is still caught at every k (20.8
sd at k = 0.1%).

### A5. Moved REUSED paths (lines 865–867, 881–883). Amend.

Commit `97085e7d8` moved the files. New paths for the three rows of §9.2:

> | problem hashing | `studies/reproductions/alwayson/code/pg/alwayson_edit.row_sha256` | `problems.json` |
> | last balanced box | `studies/reproductions/alwayson/code/pg/alwayson_edit.boxed_last` | extraction fallback |
> | in-job pinned install pattern | `studies/reproductions/alwayson/code/jobs/alwayson_edit.ensure_transformers` | pattern for the vLLM venv install |

New text of lines 881–883:

> The offline gate prints one line per check and `N13_DRYRUN_OK` only when all hold. Its command and its local venv
> (Python 3.12.12, `math-verify` 0.9.0, NumPy, transformers 5.19.0, no torch) are in
> `studies/next/07_PICKS/N13_OFFLINE_GATE.md` §"How to rerun".

### A6. Signed zeros in the builder (lines 380, 389–391). Amend.

New text of lines 389–391:

> For c = 1 arms the builder copies the after tensor's bits where M = 1 and the before tensor's bits where M = 0.
> Wherever M = 1 and Δ ≠ 0 it asserts that the formula above gives the same bits. The formula alone maps a pair of
> zeros of opposite sign (Δ = 0) to +0, so `id` would not be bitwise RL. R1 counts these signed-zero swaps per family
> (`weight_stats.json`). For c ≠ 1 the formula is applied where M = 1 and Δ ≠ 0, and the before bits are kept
> elsewhere.

### A7. PC-RANK failing input (line 565). Amend.

New failing-input cell of line 565:

> each planted neuron n's factor f_n applied to row (n mod 4,096) of the `down_proj` change only, gate and up rows
> left as they are: recovery ≤ 0.05 (chance at the real ratios: 4,523 / 352,256 = 0.013)

The stand-in at the real ratios gives recovery 1.0 on the registered input and 0.0 on this one
(`studies/next/n13/offline_gate/evidence/PC-RANK_real_ratio.json`).

### A8. Random-number conventions (lines 428–430, 581). Amend.

New text of lines 428–430:

> **Seeds of the draws** (NumPy `default_rng`): neuron sets draw from `default_rng(seed)` over the whole tail;
> DARE masks draw one stream per tensor, `default_rng([seed, layer, w])` with w = 0, 1, 2 for gate, up, down, so a
> mask does not depend on the order in which tensors are visited. Seeds: rndE 10 + j, rndT 20 + j, DARE 30 + j,
> rnd0 50, second draws rndT 120 + j and DARE 130 + j, with j = 0, 1, 2, 3 for k = 0.1%, 0.3%, 1%, 3%. Planted
> control 40. Sub-floor redraws 60 to 79, one stream per tensor, `default_rng([seed, layer, w])`. PC-ART: P+
> `default_rng([40, 1, layer, w])`; P− neurons `default_rng(41)`, signs `default_rng(42)` (A2).

### A9. MATH-500 is not an LFS file (lines 535–536). Amend.

New text:

> Every downloaded weight file's sha256 is checked against the HF tree API's LFS oid at the pinned commit. MATH-500
> `test.jsonl` is not an LFS file; its git blob sha1 is checked against the tree oid (`2376b9a1…`). GSM8K's parquet
> is checked by its LFS sha256. A mismatch stops the job.

### A10. An empty `Answer:` line (lines 334–336). Amend. **Head.**

**Approved by the head, 2026-11-05 13:00 CST.** Code: `infra/pg/n13_score.extract`; the planted answer `Answer:` (empty), then `\boxed{5}`, is in the offline gate's
scoring check (`FAB` row 12, counted from 0; scored 1 in mode `boxed`).

As written, an `Answer:` line with nothing after it gives an empty extraction, scored wrong, and the boxed fallback
is skipped. The prompt asks for the answer "on its own line after "Answer:"", so a model may write `Answer:` and put
the answer below it. Skipping such lines and falling back to the last box applies to every arm the same way. The 40
smoke answers of R0 are counted for this pattern before freeze 1 (`N13_OFFLINE_GATE.md`, R0 record). New text of
item 2:

> 2. Extraction mode `answer_line`: the text after `Answer:` on the last line that begins with `Answer:` (leading
>    spaces allowed) and has a non-empty remainder after stripping, up to the end of that line. An `Answer:` line
>    with nothing after it is skipped. If no line qualifies, mode `boxed`: `boxed_last(text)`. If none, mode `none`
>    (scored wrong).

Code after approval: `infra/pg/n13_score.extract` records a hit only when `raw.strip()` is non-empty; one planted
answer (`Answer:` empty, then `\boxed{5}`) is added to the offline gate's scoring check.

### A11. GSM8K labels (lines 726–729). Amend.

Added after line 729:

```
    UNDEFINED    opposite sign, CI includes 0        # no claim either way
```

### A12. G-HASH failing input for `zero` (lines 567, 614). Amend.

New failing-input cells of lines 567 and 614:

> `id` with neuron (0, 0) reverted to the base, and `zero` with neuron (0, 0) taken from RL: each a hash mismatch

### A13. G-TEMPLATE in the cancel list (line 686). Amend.

New line 686:

```
for gate in [G-GRADER, G-INIT, G-PARTITION, PC-RANK, G-KNOWN, G-TEMPLATE, G-HASH, G-MASK, G-COMPLETE]:
```

### A14. R1's time and where it runs (lines 931–933, 1001). Amend. **Head.**

**Approved by the head, 2026-11-05 13:00 CST.**

The S2 row's 3.8 and 5.0 GPU-h less the R0 launch's 2.52 and 3.58 leave 1.25 GPU-h for R1 at either throughput, so
R1 is projected at 75 min. A laptop timing of one 11,008 × 4,096 tensor gives about 50 min of compute. R1 is not in
the R0 launch (`s2setup-01` holds setup, R0 and the smoke). The code puts it first in the S3 launch. New text after
line 933:

> R1 runs at the start of the S3 launch: `setup --models base,rl,stage2`, then `r1_rank --delete-stage2`, then the
> four gate arms. A failure of G-INIT, G-PARTITION or PC-RANK stops that launch before any gate arm. R1 is projected
> at 75 min (1.25 GPU-h).

New S2 and S3 rows of the §9.7 table (replace line 1001 and the S3 row; totals unchanged at 25.5 and 37.5):

> | R0 launch: setup, R0, smoke | 1 instance; R0 (one olmes load), 2 smoke loads | 20,000 R0 generations; 40 smoke | 5.0 M (R0); smoke small | 2.52 | 3.58 |
> | S3: setup, R1, gate | 4 loads | 2,000 MATH-500 + 5,276 GSM8K | 5.0 M + 5.0 M | 5.37 | 7.45 |

The figures are from `n13_plan.project_total` (`studies/next/n13/offline_gate/evidence/manifests_and_budget.json`)
and are replaced by the projection from the measured smoke (R0 record).

### A15. Facts for §0.2 and §0.3 item 9 (lines 84, 87, 92, 525). Amend (facts).

Read 2026-11-05 through the HF API (`studies/next/n13/offline_gate/evidence/hf_fetched.json`,
`rl31_main_vs_steps.json`, `G-PARTITION_offline.json`):

> - 3.1 `main` (`364820b0…`) shard LFS oids: `c2cc7984cd63…`, `bc03dcc0a929…`, `cad83510bc38…`. They equal those of
>   branch `step_2800` and of no other `step_*` branch. So 3.1 `main` is `step_2800`.
> - The 3.1 `config.json` equals the base's in every shape field. It differs in `transformers_version` (4.57.1
>   against 4.57.0) and `use_cache` (false against true).
> - The 3.1 `tokenizer.json` blob is `5fe172127988…`, the base's blob.

### A16. S-BAND rounding (lines 574, 648–651). Accept.

The exact band is [81.46, 96.74]; the registered [81.5, 96.7] is rounded inward by 0.04 on each side. L' is a
multiple of 0.2 points (n = 500), and no multiple of 0.2 lies in [81.46, 81.5) or (96.7, 96.74]. Both bands judge
every reachable L' the same way. No change.

### A17. When C is frozen (lines 324–325 against 665–667). Amend.

New text of lines 324–325 (from "Frozen in"):

> Decided from the 20 RL smoke answers of the R0 launch by the rule above, written into `STATE.md` and the journal,
> passed to every generation step of the S3 and S4 manifests as `--cap C`, and recorded again in
> `thresholds.json` at freeze 2. The same C for every arm and both benchmarks.

### A18. The registered olmes vLLM command cannot run (lines 612, 630–634; §0.3 item 1). Decided: option (b). **Head.**

Found at R0 attempt `s2setup-05` (`infra/aws_pg/operations/2026-11-05_next_n13_r0.md` incident 7;
`studies/next/n13/r0/s2setup-05/r0_known/metrics.json` → `olmes_returncode` 1). Two defects of olmes `5a51f502` on its
own pins (vllm 0.11.0, lm_eval 0.4.3):

1. Observed. olmes passes its default `trust_remote_code` None to vLLM (no olmes model config names
   `allenai/Olmo-3-1025-7B`). vLLM 0.11.0's `ModelConfig` declares `trust_remote_code: bool = False` in a pydantic
   dataclass and rejects None (`ValidationError`, olmes exit 1).
2. Read from source, not yet observed. lm_eval 0.4.3 calls `LLM.generate(prompt_token_ids=requests, ...)`
   (`lm_eval/models/vllm_causallms.py` lines 252, 268, 275 of the PyPI wheel). vLLM 0.11.0's `LLM.generate` has no
   `prompt_token_ids` parameter (`vllm/entrypoints/llm.py` line 335 at `v0.11.0`). Generation would raise `TypeError`.

§0.3 item 1 registers a branch for this: "if vLLM 0.11.0 cannot load the base, R0 runs the same task with olmes
`--model-type hf`; the tolerance is unchanged and the artifact names the engine." The choice:

| option | what changes | R0 cost (estimate, not measured) | status |
|---|---|---|---|
| (a) the registered branch | `--model-type hf` in the `r0_known` args; olmes' HF path, batch 16 (`oe_eval/launch.py` line 339) | about 5 to 9 GPU-h for 20,000 samples: a batch decodes until its longest sample stops, at about 50 ms per step on one L40S. Above the R0 launch's `--budget-min 280`, so it needs a larger budget, and the N13 total rises by the same amount | registered |
| (b) an engine amendment | vLLM 0.11.0 and lm_eval 0.4.3 kept. olmes gets `--model-args '{"trust_remote_code": false}'` (False is what None means for a model with no remote code). A compatibility wrapper maps lm_eval's `generate(prompt_token_ids=ids)` to `generate([TokensPrompt(prompt_token_ids=x) for x in ids])`, the conversion vLLM itself made for that argument through v0.10.1 (`LLM._convert_v1_inputs`, `vllm/entrypoints/llm.py` line 1589 at `v0.10.1`; removed at `v0.10.2`). Same model, task, prompts, sampling and scorer | about 0.5 to 1 GPU-h for R0, plus setup and the smoke as registered | not registered; needs this amendment |

The executor's recommendation is (b), because (a) costs more than the whole registered R0 launch for the same number.
The tolerance (1.6 points) and the read (`primary_score`) are unchanged under either option. Under either, the artifact
names the engine and the change. A third defect of the same kind on either path would stop R0 again.

**Decision: option (b). Approved by the head, 2026-11-05 13:00 CST, under five conditions.** What was done:

1. **Engine and shim.** `trust_remote_code` goes to vLLM through olmes' own `--model-args` as false. No olmes or lm_eval file
   is edited. The new G-KNOWN command (replaces lines 631–633):

   > ```
   > olmes --model allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f --model-type vllm \
   >       --model-args '{"trust_remote_code": false}' --task minerva_math::olmes:n4:v2 --gpus 1 --output-dir out/r0_known/olmes
   > ```

   The compatibility shim is `infra/aws_pg/jobs/n13_olmes_shim/sitecustomize.py` (2,103 bytes, sha256
   `f39d888a3c4e688464cc6df26dd17bc42d46d00008dc25a5e17bd4011415d148`). `n13_known` puts its directory first on PYTHONPATH and
   sets `N13_VLLM_SHIM_LOG` only for the olmes subprocess. Python imports `sitecustomize` at start-up, so each olmes process
   imports vllm and replaces `LLM.generate` by a wrapper. The wrapper turns `generate(prompt_token_ids=ids, ...)` into
   `generate([TokensPrompt(prompt_token_ids=x) for x in ids], ...)` and passes every other argument through unchanged; a call
   without `prompt_token_ids` is not touched. It prints one line when installed and logs one line per converted call.
   `out/r0_known/metrics.json` holds `engine` (vllm 0.11.0, lm_eval 0.4.3, olmes `5a51f502…`, the model args, the shim path and
   sha256) and `shim.calls`, the number of converted calls, for the path check and for the full run. A directory first on
   PYTHONPATH comes before the standard library on `sys.path`, so a system `sitecustomize`, if the AMI's Python has one, does not
   shadow this one.
   **2026-11-05, A19 revision:** the shim now writes its log line after vLLM returns and adds the call's wall times and output
   tokens. The conversion is unchanged. The file is 2,738 bytes, sha256
   `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6` (A19 §"Revision, 2026-11-05", item 2). Attempts up to
   `s2setup-06` ran the `f39d888a…` file.
2. **Offline gate.** A fake LLM class whose `generate` accepts only vLLM 0.11.0's signature: with the shim, a two-prompt call
   `generate(prompt_token_ids=[[100257, 5, 6], [7]], sampling_params=sp, use_tqdm=False)` reaches it as two `TokensPrompt`s, with
   the same `sp` object and `use_tqdm` false, and one converted call is logged; a call in the new signature is untouched. The
   registered failing input, the same call without the shim, raises `TypeError` ("unexpected keyword argument
   'prompt_token_ids'"). `N13_DRYRUN_OK`, 79 of 79 (`N13_OFFLINE_GATE.md` §"Offline gate rerun, 2026-11-05, A2 A4 A10 A18").
3. **Path check before the full task.** The `r0_known` step first runs the same command with `--limit 20` (olmes applies it per
   subject: 20 problems in each of the 7, 140 in all, 560 generations) into `out/r0_known/olmes_limit20`. It must exit 0 and give 20
   scored rows per subject; the reader and the rescorer must run on it. It is a path check, not a gate: its score is recorded
   (`path_check.read`) and not read against 40.0. If it fails, the step stops with exit 4 and the full task does not run. Its
   failing input in the offline gate (19 rows per subject) stops the step. [A19 second revision, 2026-11-05: `--limit 100`, 100 rows
   per subject, 2,800 generations, into `out/r0_known/olmes_limit100`; the failing input is 99 rows per subject.]
4. **The gate is unchanged:** `primary_score` within 1.6 of 40.0; `--budget-min 280`; the 35 GPU-h ceiling covers all of N13,
   including the 1.222 GPU-h of attempts 1 to 5; the park order as registered.
5. **One attempt.** If attempt 6 (`results/next_n13/s2setup-06`) fails for any reason, R0 stops and the cause goes to the head;
   no fix and relaunch.

The A3 relaunch, if it is ever run, puts its `max_length` into the same `--model-args` object as `trust_remote_code`.

### A19. R0 runs at olmes' batch size 1, and its park projection assumed batching (lines 630–634, 996–997, 1016–1022; A18). Amend. **Head.**

Directed by the head on 2026-11-05 (brief to the third N13 executor). Written by the third N13 executor, 2026-11-05. The frozen
gate is unchanged: `primary_score` within 1.6 of 40.0.

**Found at R0 attempt `s2setup-06`** (`studies/next/07_PICKS/N13_OFFLINE_GATE.md` §"R0 attempt 6"; files copied to
`studies/next/n13/r0/s2setup-06/`). The A18 path check passed: exit 0, 20 scored rows per subject, primary_score 0.3339 at
n = 140, derangement 0.0018, identity rescore equal (`r0_known/metrics.json` → `path_check`). But it took 3,120 s for 560
generations. All 560 lines of `r0_known/shim_calls_limit20.jsonl` read `n_prompts` 1, and every
`r0_known/olmes_limit20/task-*-metrics.json` reads `compute_config.batch_size` 1. The path check generated 137,386 tokens at
45.3 tokens per second (`studies/next/n13/r0/s2setup-06/path_check_tokens.json`). At that rate the full task (20,000
generations) takes about 31 hours. The job's park check had projected 171.67 minutes and did not park
(`r0_known/metrics.json` → `park`). The full task started; the head terminated the instance.

**Cause, read from source** (quoted lines and file hashes: `studies/next/n13/offline_gate/rerun_batch/sources_batching.txt`):

1. olmes' launcher defines `--batch-size` with default None (`oe_eval/launch.py` line 109 at `5a51f502`).
   `args_dict.get("batch_size", 16)` (line 339) returns None, because the key exists. `make_cli_command` drops None values
   (`oe_eval/utils.py` lines 266–267), so `run_eval` gets no `--batch-size` and uses its own default, 1 (`oe_eval/run_eval.py`
   lines 138–142).
2. `run_eval` passes the batch size into the model config (`run_eval.py` lines 337, 651) and on to `VLLM_Verbose`
   (`oe_eval/utilities/model_utils.py` lines 89–101), which passes it to lm_eval's `VLLM` (`oe_eval/models/eleuther_vllm_causallms.py`
   line 85). lm_eval keeps the string `"auto"` (`lm_eval/models/vllm_causallms.py` lines 97–101 at `v0.4.3`).
3. olmes' `generate_until_verbose` groups the requests by generation arguments and cuts them into chunks of
   `int(self.batch_size)`, or of 0 when the batch size is `"auto"` (`eleuther_vllm_causallms.py` lines 152–155). Chunk size 0
   yields the whole group as one chunk (`lm_eval/models/utils.py` lines 612–621: `len(arr) == 0` never holds after an append).
   olmes calls `generate_until_verbose` once per subject with all of that subject's requests (`run_eval.py` lines 252–254), and
   the 4 repeats are 4 separate requests per problem (`oe_eval/tasks/base_task.py` lines 251, 307).
4. Each chunk is one `_model_generate` call, so one `LLM.generate(prompt_token_ids=chunk, ...)` (`vllm_causallms.py` lines 267–278),
   which the A18 shim turns into one `LLM.generate([TokensPrompt × n], ...)`. vLLM 0.11.0 adds every prompt of the call to its
   engine and steps until all finish (`vllm/entrypoints/llm.py` lines 393–401, 1502–1523, 1599–1600 at `v0.11.0`), up to 256
   sequences at a time on a GPU with less than 70 GiB (`vllm/engine/arg_utils.py` lines 1605, 1617–1624), with prefix caching on
   by default (lines 1560–1561).

So at batch size 1 every `generate` call holds one prompt and vLLM decodes one sequence at a time. The registered projection
(§9.7, line 996: 5.0 M tokens at 500 to 800 output tokens per second) assumed batched decoding. The token count was close (the
path check's mean length gives about 4.64 M tokens, `path_check_tokens.json`); the rate was 11 times lower (45.3 against 500).

**Amendment.**

1. **The batch size.** The G-KNOWN command (replaces A18's command, which replaced lines 631–633):

   > ```
   > olmes --model allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f --model-type vllm \
   >       --model-args '{"trust_remote_code": false}' --task minerva_math::olmes:n4:v2 --gpus 1 --batch-size auto \
   >       --output-dir out/r0_known/olmes
   > ```

   The path check adds only `--limit 20` [`--limit 100` since the second revision below: 400 prompts per call]. With `"auto"`, each subject is one `generate` call: 80 prompts in the path check, 4 × the
   subject size in the full task (4,748 for algebra). `"auto"` is chosen over an integer because olmes passes the string through
   (`launch.py` lines 109, 351; `run_eval.py` line 140, `type=str`), and one call per subject leaves the scheduling to vLLM. A
   fixed chunk such as 256 would wait for the longest sample of each chunk before the next one starts. The S3 and S4 arm runners do
   not build an olmes command: `n13_gen.py` submits every request of a step to one vLLM 0.32.0 `LLMEngine` before stepping it
   (lines 322–324, 345–346). The shim is not changed (sha256 `f39d888a…`); it already accepts a list of token lists. [The revision
   below changes the shim's log line, not the conversion: sha256 `1f6ccee3…`.]
2. **What batching does and does not change.** The sampling parameters are built per chunk from the same generation arguments:
   temperature 0.6, top-p 0.6, `max_tokens` 1,024, stop sequences `Problem:`, a blank line and the EOS text, `logprobs` 1
   (`eleuther_vllm_causallms.py` lines 186–190, 233, 243–249; `vllm_causallms.py` lines 234–235, 527–540). The context is
   truncated per request to its last 2,048 − 1,024 = 1,024 tokens, whatever the batch (`truncate_context` defaults to True and the
   Minerva config does not set it: lines 196, 200–202, 231; `oe_eval/configs/tasks.py` lines 4696–4714). The 4 repeats stay 4
   requests. What changes: no request carries a seed (the task sets no `generation_seeds`; lm_eval seeds only the engine, 1234,
   `vllm_causallms.py` line 94), so the individual draws depend on which requests share a step, and batched kernels round
   differently. The sampled distribution is the same. The batch-1 path-check score (0.3339) is not expected to repeat exactly;
   it is not a gate.
3. **The path-check assertion.** The path check now also requires at least one logged shim call with `n_prompts` ≥ 2
   (`path_check.batched` in `out/r0_known/metrics.json`). If it fails, the step stops with exit 4 before the full task. After a
   passing path check the step also asserts that the logged prompts equal 7 × 20 × 4 = 560, the projection's denominator (an
   assertion: the step stops with exit 1). **Registered failing input: batch size 1**, 560 shim calls of one
   prompt each, as in `s2setup-06`. In the offline gate it stops the step with exit 4. [Second revision below: the path check also
   stops early, within one poll interval, when its first logged call has one prompt; the end assertion is 7 × 100 × 4 = 2,800.]
4. **The projection gate.** After the path check passes, and before the full task:
   projected full-task seconds = (path-check wall seconds / path-check generations) × full-task generations.
   Full-task generations = 20,000: the test-split sizes of the 7 subjects (algebra 1,187, counting_and_probability 474, geometry
   479, intermediate_algebra 903, number_theory 540, prealgebra 871, precalculus 546; sum 5,000; line 639, `N13_MATH_TASK.md`
   lines 131–133, and the Hugging Face datasets-server record banked as
   `studies/next/n13/offline_gate/rerun_batch/hf_math_test_sizes.json`) times the 4 repeats. The step parks when the projection
   exceeds 85 % of the minutes left, where minutes left = the allowance (`--allowance-min`, 280) minus the launch's elapsed time
   (`n13_plan.launch_t0`). A parked step writes `projection` and status `parked` to `out/r0_known/metrics.json`, writes the
   `PARKED` marker (so the smoke steps park), exits 0 and does not start the full task. **Registered failing input: the
   `s2setup-06` numbers** (560 generations in 3,120 s; 20,000; allowance 280 min; 42 min elapsed) give 1,857.1 min against
   0.85 × 238 = 202.3 min and park. A planted batched rate (560 generations in 300 s) gives 178.6 min and goes. The old
   projection (olmes load 5 min + 5.0 M tokens at `--proj-tok-s`) is removed, with `--proj-tok-s` from the `r0_known` arguments;
   the R0 manifest `infra/aws_pg/steps/n13_s2setup-01.json` is regenerated from `n13_plan.manifest("s2setup", 1,
   allowance_min=280)`. [Replaced by the revision of 2026-11-05 below: projected = F + T / r, and elapsed from the job's start on
   the instance.]
5. **Unchanged:** the gate (`primary_score` within 1.6 of 40.0) and its failing input and identity pair; the read; the shim
   [its log line changed in the revision below]; `--budget-min 280`; the 35 GPU-h ceiling; the park order. The §9.7 projector (`n13_plan.project_launch`) is not changed.
6. **Room to pass, for the head.** The rule scales the path check's fixed costs by 20,000 / 560 = 35.7: the olmes start-up (90 s
   in `s2setup-06`, `path_check_tokens.json` → `startup_s`), the per-subject set-up, and the per-subject wait for the longest sample
   at only 80 prompts. At 42 min elapsed the largest path-check wall time that goes is 339.9 s
   (`rerun_batch/A19_projection.json`). The executor's estimate of a healthy batched path check is about 400 s (an estimate from
   source and the batch-1 rate, not a measurement; `N13_OFFLINE_GATE.md` §"R0 attempt 6"), against a bound of about 365 s at the
   24.5 min elapsed that estimate implies. So the rule may park a run whose full task would fit. [The head accepted this; the
   revision below replaces the rule.]
   `run_olmes` now records `started_utc`, so the start-up can be read after the run (olmes' `task-000-…-metrics.json` →
   `current_date`).

Offline gate after the change: `N13_DRYRUN_OK`, 85 of 85 (`studies/next/n13/offline_gate/rerun_batch/checks.json`).

#### Revision, 2026-11-05: the projection from measured throughput, and the elapsed clock. **Head.**

Directed by the head on 2026-11-05 (brief to the fourth N13 executor). Written by the fourth N13 executor. The frozen gate is
unchanged: `primary_score` within 1.6 of 40.0. Nothing else in A19 changes: `--batch-size auto`, the batched-call assertion and
its failing input, the 560 assertion, the margin 0.85, the allowance 280 and the park behaviour stay as above.

**Reason.** The first form would have parked a healthy batched run. It multiplied the whole path-check wall time by
20,000 / 560 = 35.7, so the olmes start-up and the per-subject set-up were scaled as if they recurred per generation. On a planted
batched path check (500 output tokens per second, F = 132 s, 406.8 s wall, 23.4 min elapsed) the first form projects 242.1 min
against 218.1 min allowed and parks; the revised form projects 165.8 min and goes (`studies/next/n13/offline_gate/rerun_batch2/
A19_projection.json` → `batched_500_tok_s`).

1. **The projection** (`n13_known.project`, called from `n13_known.run` after the path check passes):
   - T = 20,000 × (path-check output tokens / 560): the full task's output tokens at the path check's mean length.
   - r = path-check output tokens / path-check generation seconds.
   - F = path-check wall seconds − generation seconds: the olmes start-up and the 7 subjects' set-up and scoring, carried over
     unchanged.
   - projected = F + T / r. The step parks when projected > 0.85 × (allowance − elapsed).
   [Replaced by the second revision below: T is weighted by subject size, and the path check runs at `--limit 100`.]

   Sources. Output tokens and generation seconds: the shim log `out/r0_known/shim_calls_limit20.jsonl`, summed by
   `n13_known.shim_calls` (`output_tokens`, `generate_s`). Wall seconds: `path_check.elapsed_s` (`n13_known.run_olmes`, the olmes
   subprocess from start to exit). 560: the logged prompts, asserted equal to 7 × 20 × 4. 20,000: `n13_known.FULL_GENERATIONS`.
   Elapsed: `n13_plan.launch_t0` (item 3).

   The denominator of r, generation seconds, is the sum over the shim's calls of t_end − t_start: the wall time inside vLLM's own
   `LLM.generate` (request intake, prefill, decode, detokenization, collecting the outputs). It excludes the olmes start-up, the vLLM
   engine construction (model load, cache profiling, graph capture), each subject's data loading, prompt building and tokenization,
   and olmes' scoring and file writes; those are in F. The option chosen is (a) of the brief, the shim's own timer, because it is
   per call and needs no olmes file timestamps.

   T / r equals 35.7 × generation seconds: the token count cancels. The decision reads generation seconds and F; T and r are
   recorded in `projection` for the reader. So the one change from the first form is that F is not multiplied by 35.7. T follows
   the brief's formula, the unweighted mean length; the subject-weighted estimate of `path_check_tokens.py` gives 4.64 M for
   `s2setup-06` against 4.91 M unweighted.

   r is measured at 80 prompts per call, and each call waits for its longest sample. The full task sends 1,896 to 4,748 prompts per
   call (4 × the subject size), up to 256 running at once. So r is below the full task's rate and the projection errs toward
   parking.
   `n13_known.project` stops (assertion) unless 0 < generation seconds ≤ wall seconds.
2. **The shim** (`infra/aws_pg/jobs/n13_olmes_shim/sitecustomize.py`) writes one line per converted call after vLLM returns:
   `n_prompts`, `t_start`, `t_end` and `output_tokens`, the sum of `len(token_ids)` over every completion of every returned
   `RequestOutput`. vLLM 0.11.0's `LLM.generate` returns `list[RequestOutput]` (`vllm/entrypoints/llm.py` line 344);
   `RequestOutput.outputs` is a list of `CompletionOutput`, each with `token_ids` (`vllm/outputs.py` lines 22, 42, 79, 109 and 131
   at `v0.11.0`, fetched raw from GitHub, sha256 `ee60aa7563b7db029c5cc352fcdbd63015575e2d5f898933dba294fb69afd389`). olmes counts
   each generation's `num_tokens` the same way, `len(completion_output.token_ids)` of `outputs[0]`
   (`oe_eval/models/eleuther_vllm_causallms.py` lines 253 and 258 at `5a51f502`, sha256 `1f96bb32…`). Quoted lines:
   `studies/next/n13/offline_gate/rerun_batch2/sources_shim.txt`. The return value is passed back unchanged. The conversion is unchanged. New file: 2,738
   bytes, sha256 `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6` (was `f39d888a…`).
3. **The elapsed clock.** `n13_plan.launch_t0` now reads `started_utc` from `out/pipeline_status.json`.
   `infra/aws_pg/jobs/pipeline.py` takes that time when its process starts (whole seconds, UTC) and writes it at its first write,
   before any step runs. `run_job.sh` starts the pipeline under `timeout --signal=KILL <budget>s` (line 275), the hard kill. So the
   marker sits after the kill clock's start by the interpreter's start-up (0.05 s for the pipeline's imports on the laptop; not
   measured on the instance) and before it by up to 1 s of truncation. The launcher's EventBridge reaper is armed at the launch call
   plus the budget plus 600 s (`infra/aws_pg/pg.py` `_arm_deadline`); with 69 s from the launch to the setup start in `s2setup-06`
   it fires about 9 min after the timeout. Before: `launch_t0` read the mtime of `out/steps.json`, which `aws s3 cp` sets to the
   object's upload time. That was 9.0 min before the launch in `s2setup-06` (upload 19:13:43Z, launch 19:22:44Z) and grows with
   every launch retry. At `s2setup-06`'s decision point (about 20:32:28Z) the new clock would have read 68.6 to 69.0 min (the
   pipeline started between the first heartbeat at 19:23:27Z and the setup start at 19:23:53Z) instead of 78.8. The smoke steps'
   park check (`n13_plan.park_check`) uses the same clock. Outside the pipeline (the offline gate) the step's out-directory mtime
   is used, as before.
4. **Registered failing input: the `s2setup-06` path check.** 137,386 output tokens in 3,030 s of generation (45.3 tokens per
   second), wall 3,120 s, so F = 90 s (`studies/next/n13/r0/s2setup-06/path_check_tokens.json`), allowance 280 min, nothing elapsed:
   projected 1,805.1 min against 238.0 allowed. It parks. With the brief's figures (T = 4.64 M at 45.3 tokens per second, F 90 s):
   1,708.5 min. It parks. Nothing elapsed is the most permissive case.
5. **Planted passing input: a batched rate.** The same tokens at 500 tokens per second, F = 132 s (90 s start-up + 7 × 6 s set-up,
   the third executor's estimate), elapsed 15.9 min setup + the path-check wall + 0.7 min rescore: projected 165.8 min against
   218.1. It goes.

| input | wall s | generation s | F s | r tokens/s | elapsed min | projected min | allowed min | revised | first form |
|---|---|---|---|---|---|---|---|---|---|
| `s2setup-06` replay | 3,120.0 | 3,030.0 | 90.0 | 45.3 | 0 | 1,805.1 | 238.0 | parks | 1,857.1, parks |
| `s2setup-06`, the brief's figures (T 4.64 M) | 2,957.8 | 2,867.8 | 90.0 | 45.3 | 0 | 1,708.5 | 238.0 | parks | 1,760.6, parks |
| planted, 300 tokens/s | 590.0 | 458.0 | 132.0 | 300 | 26.4 | 274.8 | 215.5 | parks | 351.2, parks |
| planted, 500 tokens/s | 406.8 | 274.8 | 132.0 | 500 | 23.4 | 165.8 | 218.1 | goes | 242.1, parks |
| planted, 800 tokens/s | 303.7 | 171.7 | 132.0 | 800 | 21.7 | 104.4 | 219.6 | goes | 180.8, goes |

   Source: `studies/next/n13/offline_gate/rerun_batch2/A19_projection.json` (`n13_dryrun.projection_table`). T is 4,906,643
   tokens in every row but the second.
6. **Room to pass.** At F = 132 s and 16.6 min of setup and rescore, the largest generation time that goes is 360.7 s, 51.5 s per
   subject, which is r ≥ 380.9 tokens per second for 137,386 tokens (`A19_projection.json` → `boundary_at_F_132_s`). A path check
   below about 381 tokens per second parks. The third executor's estimate of a batched path check is about 37 s per subject (an
   estimate, not a measurement; `N13_OFFLINE_GATE.md` §"R0 attempt 6").

Offline gate after the revision: `N13_DRYRUN_OK`, 87 of 87 (`studies/next/n13/offline_gate/rerun_batch2/checks.json`).

#### Second revision, 2026-11-05: a path check at the engine's concurrency limit, an early stop at batch size 1, and the subject-weighted projection. **Head.**

Directed by the head on 2026-11-05 (brief to the fifth N13 executor). Written by the fifth N13 executor. The frozen gate is
unchanged: `primary_score` within 1.6 of 40.0. Also unchanged from the revision above: `--batch-size auto`, the margin 0.85, the
allowance 280, r = output tokens / generation seconds, F = path-check wall − generation seconds, the elapsed clock
(`started_utc`), the park behaviour, and the shim (sha256 `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6`).

**Reason.** In the revision above, T / r = (20,000 / 560) × generation seconds. The token count cancels. So a batched
`--limit 20` path check went only if its 7 calls generated for at most about 361 s in all, about 51 s per subject. A call of 80
prompts does not keep vLLM full. Most of its samples stop after a few hundred tokens. The call then waits for its longest sample
with few sequences running. So its mean rate is below the full task's rate. The full task's calls hold 1,896 to 4,748 prompts and
keep the engine at its concurrency limit. The head put the gap at a factor of about two to three (not measured) and put a batched
80-prompt call at 40 to 60 s per subject (an estimate). At those numbers the gate parks a healthy run about as often as it passes
one. The pre-check has to measure the quantity the projection needs: the output rate with the engine at its concurrency limit.

1. **The path check runs at `--limit 100`** (`n13_known.PATH_LIMIT`). 100 problems × 4 repeats is one call of 400 prompts per
   subject, 2,800 generations in all (`n13_known.PATH_PROMPTS`). 400 is above vLLM 0.11.0's limit of 256 running sequences on a
   GPU with less than 70 GiB (`vllm/engine/arg_utils.py` lines 1605, 1617–1624, quoted in
   `studies/next/n13/offline_gate/rerun_batch/sources_batching.txt`). The KV cache may bind first: about 50,000 cached tokens on
   the L40S (§9.1; an estimate, not measured). Either way 400 prompts exceed the limit, so the engine runs at its limit for most of
   each call and only the end of each call drains. Every subject has at least 474 test problems. The path check must give 100
   scored rows per subject. Its directory is `out/r0_known/olmes_limit100` and its log `out/r0_known/shim_calls_limit100.jsonl`.
   The end assertion on the logged prompts is 7 × 100 × 4 = 2,800.
2. **An early stop at batch size 1** (`n13_known.run_olmes`, `n13_known.first_call`). At 2,800 generations a batch-size-1 path
   check would run about 4.3 hours at `s2setup-06`'s 5.57 s per generation before the end assertion could fire. So during the
   path check every poll (`--poll-s`, 60 s on the instance) reads the shim log's first line. If the first logged call has
   `n_prompts` < 2, the step kills olmes' process group with SIGKILL. For the path check olmes is started in its own session, so the
   kill reaches olmes' launcher, the `run_eval` it starts through `/bin/sh` and vLLM's engine process. The step then records
   `path_check.early_stop` (the call, when it was seen, the lag after the call returned) and `path_check.reason`, sets status
   `failed_path_check` and exits 4. With batching the first line appears when the first subject's 400-prompt call returns. At batch
   size 1 it appears after the first generation, a few seconds after olmes' start-up (90 s in `s2setup-06`). So a batch-size-1 run
   stops within about 2.5 min of the path check's start, at most one poll interval after its first call. The end assertion is kept:
   at least one call with `n_prompts` ≥ 2, then 2,800 prompts. The full run is started as before, in the pipeline's process group.
   **Registered failing input: batch size 1** (2,800 calls of one prompt; the first has `n_prompts` 1). In the offline gate the
   first call is seen 0.048 s after it returned (poll interval 0.05 s), the step exits 4 after 0.31 s against at least 28 s
   unstopped, and the log stops growing at 4 lines, so the child process died too (`rerun_batch3/A19_batch1_early.json`). A second
   failing input exercises the kept end assertion: a batch-size-1 run that exits before any poll sees its log fails at the end
   (exit 4, no early stop; `rerun_batch3/A19_batch1_end.json`).
3. **The subject-weighted projection** (`n13_known.project`). The path check takes 100 problems from each subject. The full task
   weights the subjects by size. The shim logs one call per subject, in olmes' task order. That order is the suite's: algebra,
   counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus. In `s2setup-06` the files
   task-000 to task-006 began in this order (`studies/next/n13/r0/s2setup-06/path_check_tokens.json` → `began_utc`). `project`
   asserts exactly 7 calls (else the step stops, exit 1) and maps call i to subject i. For subject s, tokens_s and n_s are its
   call's output tokens and prompts.
   - T = Σ_s 4 × size_s × tokens_s / n_s: the full task's output tokens at each subject's own mean length.
   - r = Σ_s tokens_s / Σ_s generation seconds_s.
   - F = path-check wall seconds − Σ_s generation seconds_s.
   - projected = F + T / r. The step parks when projected > 0.85 × (allowance − elapsed).
   `projection.per_subject` records each subject's `n_prompts`, `output_tokens`, `generate_s`, `mean_tokens` and `T_s`. On
   `s2setup-06`'s lengths the weighted T is 4,643,970 tokens and the unweighted T 4,906,643 (5.4 % lower). The largest subject,
   algebra (1,187 problems), has the shortest samples (138.4 tokens on average). The longest, precalculus (394.9), has 546 problems.
4. **Registered failing input of the projection: the `s2setup-06` path check.** Its 560 one-prompt calls are summed into one call
   per subject: the subject's tokens, 80 prompts, and its generation phase as the call's seconds (`path_check_tokens.json`). T is
   4,643,969.85 tokens, equal to `path_check_tokens.py`'s own full-task estimate. r is 45.3 tokens per second and F 90 s. With
   nothing elapsed it projects 1,708.5 min against 238.0 allowed. It parks.
5. **Planted inputs at `--limit 100`.** 400 prompts per subject at `s2setup-06`'s per-subject mean lengths, 686,930 output tokens
   in all; F = 132 s (90 s start-up + 7 × 6 s set-up, the third executor's estimate); elapsed = 15.9 min setup (`s2setup-06`) +
   the path-check wall + 3.5 min rescore (`s2setup-06`'s 0.7 min for 140 documents, times 5 for 700; an estimate).

| input | wall s | generation s | r tokens/s | elapsed min | projected min | allowed min | decision | unweighted T (revision above) |
|---|---|---|---|---|---|---|---|---|
| `s2setup-06`, one call per subject | 3,120.0 | 3,030.0 | 45.3 | 0 | 1,708.5 | 238.0 | parks | 1,805.1, parks |
| planted, 300 tokens/s | 2,421.8 | 2,289.8 | 300 | 59.8 | 260.2 | 187.2 | parks | 274.8, parks |
| planted, 500 tokens/s | 1,505.9 | 1,373.9 | 500 | 44.5 | 157.0 | 200.2 | goes | 165.8, goes |
| planted, 1,000 tokens/s | 818.9 | 686.9 | 1,000 | 33.0 | 79.6 | 209.9 | goes | 84.0, goes |

   Source: `studies/next/n13/offline_gate/rerun_batch3/A19_projection.json` (`n13_dryrun.projection_table`). At these planted
   inputs the slowest rate that goes is 400.7 tokens per second (`boundary_at_F_132_s`). With the path check at the engine's limit,
   r measures the full task's rate except for the drain at the end of each call. So the boundary sits near the rate at which the
   full task stops fitting: at 400.7 tokens per second the decision comes at 50.2 min elapsed with 195.4 min projected, which leaves
   about 34 min of the 280 for the full rescore and the smoke.
6. **The larger path check's time** (an estimate from `s2setup-06`'s lengths, not a measurement). Its generation takes 686,930
   tokens / r: 38.2 min at 300, 22.9 min at 500, 11.4 min at 1,000 and 8.8 min at 1,300 tokens per second. The 80-prompt check
   was estimated at 40 to 60 s per subject, 4.7 to 7.0 min. The difference, about 2 to 18 min at 1,300 to 500 tokens per second,
   plus about 2.8 more min of rescore, counts in the elapsed time, so the projection accounts for it. GPU-hours and cost are in
   `infra/aws_pg/operations/2026-11-05_next_n13_r0.md`.
7. **What the offline gate cannot show.** That olmes really runs the subjects in suite order on the instance (read from
   `s2setup-06`, not checked inside the step), and how fast a 400-prompt call runs. The path check writes both into
   `out/r0_known/metrics.json` → `projection.per_subject` before the full task. Each subject's `output_tokens` can be checked after
   the run against the sum of `num_tokens` in that subject's `olmes_limit100/task-*-predictions.jsonl`.

The R0 manifest `infra/aws_pg/steps/n13_s2setup-01.json` is unchanged: the path check's limit lives in `n13_known.PATH_LIMIT`, not in
the step arguments, and the file equals `n13_plan.manifest("s2setup", 1, allowance_min=280)`. Offline gate after the second
revision: `N13_DRYRUN_OK`, 89 of 89 (`studies/next/n13/offline_gate/rerun_batch3/checks.json`).

### A20. The per-block weight-change profile in R1 (§9.4 S2 step 3; R1's outputs). Add; informational. **Head.**

Decided by the head on 2026-11-05 after reading Fel et al. 2026 ("Shaping & Monitoring Models through their Internal
Representations", Goodfire preprint), summarized in `studies/next/12_SHAPING_NOTE.md` §4 item 2 (commit `52ade986b`, on branches
`clean-neurons` and `m1x`, not on `n13`). Written by the sixth N13 executor, 2026-11-05.

**Reason.** In that preprint, five consecutive OLMo-3-7B blocks copied from one DPO run into another recovered most of a behaviour
difference, and they were the blocks whose weights differed most between the runs (pp. 60–61, as read in the note). Whether the
RL-Zero-Math change has a block-level peak or is flat informs a later decision on block-window arms (note §4 item 7). R1 already
reads every tensor, so the profile costs no GPU run.

**What R1 now writes.** `n13_arms.weight_stats` keeps, per tensor, the sum of squared change and the sum of squared before-weights
(float64), and `n13_arms.block_profile` groups them. For each block l in 0–31: `attn[l]` = ‖θ_RL − θ_base‖ / ‖θ_base‖ over the q, k, v
and o projections (the Frobenius norm over the four tensors), `mlp[l]` the same over gate, up and down. `embed` and `lm_head` are the
same quantity for `model.embed_tokens.weight` and `lm_head.weight`. The norm weights (q_norm, k_norm, the two block layer norms, the
final norm) are in no group. `n13_rank` writes the result as `block_profile` into `out/r1_rank/metrics.json` and `sets.json` (and,
through the weight statistics, `weight_stats.json`), with the sums for a recount (`attn_sq`, `mlp_sq`, `embed_sq`, `lm_head_sq`, each
`delta` and `before`).

**No rule reads it.** It is not a gate. It has no threshold and changes no arm, set, cap or verdict.

**Offline gate** (`studies/next/n13/offline_gate/rerun_a20/A20_block_profile.json`). On the 4-block stand-in, R1's profile is equal in
`metrics.json` and `sets.json` and equals a separate float64 recount from the snapshots (largest relative difference 0.0). Planted
input: block 2's attention and MLP tensors scaled by 2 (exact in bf16) give `attn[2]` = `mlp[2]` = 1.0 and 0 at every other block, the
embedding and the head (largest error 0.0). Identity pair: base against itself gives 0 everywhere, so the planted expectation fails on
it.

**Cost.** One more float64 sum of squares over each base tensor in each `weight_stats` call (three calls in R1: the real change, P+
and P−). Not timed.

### A21. The n13 engine stops at start-up in FlashInfer's sampling JIT (§2.3; §9.2; found at the smoke of R0 attempt 7). Amend. **Head.**

Directed by the head on 2026-11-05 (message of 00:26Z on 2026-11-06). Written by the sixth N13 executor. No gate, threshold, prompt,
sampling field or cap changes.

**Found at `s2setup-07`, step `smoke_base`**, the first start of the n13 venv's vLLM 0.32.0 on an instance (the earlier attempts
stopped before it). The engine core died during start-up with `RuntimeError: Could not find nvcc and default
cuda_home='/usr/local/cuda' doesn't exist`, raised by `flashinfer/jit/cpp_ext.py` `get_cuda_path` while building FlashInfer's sampling
module (`gen_sampling_module().build_and_load()`). `n13_gen` exited 1 at 00:23:41Z and the pipeline stopped before `smoke_rl`. The
step wrote only `base/build.json` and `base/math500/metrics.json` (status `running`, `template.pass` true). The log lines are banked:
`studies/next/n13/r0/s2setup-07/smoke_base/job_log_lines_11883-12185.txt` (job.log copied once after the job ended, sha256
`46db440f…`). The olmes venv's vLLM 0.11.0 ran R0 on the same instance. The AMI has no CUDA toolkit (no `nvcc`, no `/usr/local/cuda`;
attempt 1 found no `Python.h`).

**Cause, read from source.** vLLM at tag `v0.32.0` and flashinfer-python 0.7.0.post1 (the venv's versions,
`studies/next/n13/r0/s2setup-07/setup/pip_freeze_n13.txt`). Every quoted line, with each file's sha256:
`studies/next/n13/offline_gate/rerun_a20/sources_a21.txt`. The fetched files match the traceback at every line it names.

1. The engine ran the V2 model runner (job.log: "Using V2 Model Runner"). Its sampler sets
   `use_flashinfer = not return_sampling_mask and flashinfer_sampler_supported()` (`vllm/v1/worker/gpu/sample/sampler.py` lines 88–90).
2. `flashinfer_sampler_supported` returns True on CUDA when `VLLM_USE_FLASHINFER_SAMPLER` is not 0, the compute capability is
   supported, the GPU has more than 16 SMs, and FlashInfer's `check_cuda_arch` passes (`vllm/v1/sample/ops/topk_topp_sampler.py`
   lines 93–125). It does not ask whether `nvcc` exists. vLLM's own `has_flashinfer()` does ask (`vllm/utils/flashinfer.py` lines
   87–105), and it returned False on the instance: job.log shows its warning (line 99), "FlashInfer kernels are disabled:
   flashinfer-cubin is not installed and nvcc ... is missing". The sampler does not consult it. The log then shows "Using FlashInfer for
   top-p & top-k sampling" (`topk_topp_sampler.py` line 124).
3. Start-up warms the sampler with `SamplingParams.for_sampler_warmup()` (`vllm/v1/worker/gpu/warmup.py` lines 324 and 381;
   `vllm/sampling_params.py` lines 1474–1490: top-p 0.9, top-k 50, no seed). With top-k or top-p set and no seed, the fused sampler is
   eligible (`sampler.py` lines 307–315), so `_sample_random` calls `flashinfer_sample` (lines 337–341). FlashInfer then JIT-compiles its
   sampling module, and with no CUDA toolkit `get_cuda_path` raises.

**The head's three questions.**

- (a) **`VLLM_USE_FLASHINFER_SAMPLER=0` avoids the call.** `vllm/envs.py` lines 878–882 parse it to False, and
  `flashinfer_sampler_supported` returns False at `topk_topp_sampler.py` lines 95–100. That is before the JIT check and before the raise
  at lines 126–130, which fires only when the variable is set and FlashInfer is unavailable, and is not reached at 0. With
  `use_flashinfer` False, `_sample_random` takes `apply_top_k_top_p` (Triton when available, lines 460–470) and `gumbel_sample`
  (`sampler.py` lines 344–355). No other file of `vllm/v1/worker/gpu/sample/` (19 files) and not `vllm/v1/worker/gpu/warmup.py` calls
  FlashInfer. **It changes no n13 draw.** Every n13 request carries a seed (`n13_score.SEED_BASE[set] + index`). Any request with an
  explicit seed makes the fused sampler ineligible (`sampler.py` lines 307–312), so the n13 requests take `apply_top_k_top_p` and
  `gumbel_sample` with or without the variable. The variable changes only the kernel that samples the warmup's unseeded requests.
- (b) **Attention.** On an L40S (SM 8.9), for a model without MLA, vLLM's priority list is FLASH_ATTN, FLASHINFER, TRITON_ATTN,
  FLEX_ATTENTION, TURBOQUANT (`vllm/platforms/cuda.py` lines 168–179). On the instance it chose FLASH_ATTN with FlashAttention version
  2 (job.log), and start-up then passed model load, torch.compile, CUDA-graph capture and KV-cache profiling. So this backend needed no
  JIT there. vLLM 0.32.0 has no `VLLM_ATTENTION_BACKEND` variable (not in `vllm/envs.py`). The backend is pinned by the engine argument
  `attention_backend` (`vllm/engine/arg_utils.py` line 791; lines 2686–2695 move it into `attention_config.backend`). A pinned
  backend that is not valid raises `ValueError` (`cuda.py` lines 428–444) instead of falling down the list to FLASHINFER, which would
  JIT-compile. `n13_gen` now pins `attention_backend="FLASH_ATTN"`: the same backend as the default, now fail-loud.
- (c) **Other FlashInfer entries on `n13_gen`'s path.** Every start-up stage before the sampler warmup ran on the instance (job.log
  lines 11884–11937): model load, torch.compile, CUDA graphs, a KV cache of 87,790 tokens at `max_model_len` 18,432 (= 2,048 +
  16,384), and vLLM's "JIT kernel warmup finished in 0.00s". After the sampler warmup come the warmup's decode steps and the loop.
  These use the same sampler and attention backend. Sampling at temperature 0.6, top-p 0.95, top-k off, with seeds, is the Triton path
  of (a). `n13_gen` sets no stop strings, only `stop_token_ids` (`n13_gen.SAMPLING`; `NEUTRAL` asserts `stop` is None or empty).
  The cap of 16,384 sets only `max_tokens` and `max_model_len`. The logprob code (`sample/logprob.py`) calls no FlashInfer. What is
  not shown offline: the remaining warmup steps and the loop on the instance. The smoke (attempt 8) shows them.

Also recorded, not changed: vLLM 0.32.0 serves `Olmo3ForCausalLM` only through its Transformers modeling backend
(`vllm/model_executor/models/registry.py` line 724, in `_TRANSFORMERS_SUPPORTED_MODELS`; job.log: "Using Transformers modeling
backend"). Every S3 and S4 arm runs on this engine path.

**Amendment (code, `infra/aws_pg/jobs/n13_gen.py`).**

1. `ENGINE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}`. `child_env` puts it into the `--inner` child's environment, with the venv
   first on PATH as before. `main` starts the child with it, and vLLM's engine-core process inherits it.
2. `VllmEngine` asserts that `os.environ` holds `ENGINE_ENV` before it imports vllm. It passes `attention_backend="FLASH_ATTN"`
   (`ATTENTION_BACKEND`) to `EngineArgs`. It records `env`, `attention_backend` and `amendment` in the `engine` block of every
   generation step's `metrics.json`, beside the engine arguments.

**Registered failing input:** the engine start without the variable, which raises "Could not find nvcc" on the instance
(`s2setup-07`). It cannot run on the laptop (no vLLM, no GPU). So the offline gate checks the environment only. `child_env` sets the
variable and keeps PATH and every other variable. `main` uses `child_env`. `VllmEngine` stops on the A21 assertion when the variable is
unset or 1. At 0 it passes the assertion and reaches the vllm import, which fails on the laptop because vLLM is absent
(`studies/next/n13/offline_gate/rerun_a20/A21_engine_env.json`). The first run of the fix is the smoke of attempt 8.

**Alternative, not taken.** Install FlashInfer's precompiled kernels instead: vLLM looks for `flashinfer-cubin`
(`vllm/utils/flashinfer.py` lines 76–84) and names `flashinfer-jit-cache`, published for CUDA 12.9 and later (lines 115–121); both must
match flashinfer-python 0.7.0.post1. The source shows the variable is enough, and it adds no package.

**Attempt 8** (`results/next_n13/s2smoke-08`): `n13_plan.manifest("s2smoke", 8, allowance_min=90)`, written to
`infra/aws_pg/steps/n13_s2smoke-08.json`: setup without the olmes venv, then `smoke_base` and `smoke_rl`. R0 is banked and is not
rerun.

Offline gate after A20 and A21: `N13_DRYRUN_OK`, 93 of 93 (89 before, 2 new for A20, 2 new for A21; the manifest check now also parses
the `s2smoke` manifest), `studies/next/n13/offline_gate/rerun_a20/checks.json`.

### A22. The smoke measures a latency-bound rate (§9.4 S2 step 4; §9.7; found before attempt 8). Amend. **Head.**

Directed by the head on 2026-11-05 at 18:50 CST (brief to the seventh N13 executor). Written by the seventh N13 executor,
2026-11-06. No gate, threshold, prompt, sampling field or cap rule changes.

**Reason** (the sixth executor's finding; the head agrees). A smoke of 20 requests runs the engine far below its concurrency limit
and then waits for the longest answer. A gate arm sends 1,819 requests at once. A projection from the 20-request rate would
overstate S3 and S4 by several times and park arms wrongly. R0 had the same defect; A19 fixed it by measuring at the engine's limit.

1. **The smoke runs MATH-500 problems 0:300** for `base` and for `rl`: seed set A, per-problem seeds unchanged (`n13_score.SEED_BASE["A"]
   + index`), cap 16,384. So 300 requests are in the engine at once. The reading of §9.4 is unchanged: 10 answers per model, read in
   full. The rule for C (§2.3, A17) reads "the 20 RL smoke answers". It is applied to problems 0–19, the registered 20, unchanged. The
   count over all 300 is recorded beside it and decides nothing.
2. **The engine-step log** (`n13_gen.StepLog`). At every engine step, `n13_gen` logs the step's wall time, the number of running
   sequences, the number of unfinished requests and the output tokens the step produced. A step's wall time runs from the end of the
   previous `step()` call to the end of this one, so the step times sum to the generation's wall time. Running sequences are the
   requests that received new tokens in the step. vLLM 0.32.0's default output kind is cumulative, so `step()` returns every request
   with new tokens (`vllm/v1/engine/output_processor.py` lines 307–323, 440–441); vLLM's own running count also counts requests in
   prefill. Unfinished requests are the output processor's count (lines 526–527). The file is `<out>/steps.jsonl`, **aggregated per
   10 steps**: one line per step would pass 1 MB in a gate arm (about 10^5 steps). Each line holds the block's first step, step count,
   end time, wall seconds, tokens, [min, max] running and unfinished, and two sums that keep the rates exact: over the steps with at
   least 128 running (`full`) and over the steps with requests waiting, unfinished > running (`wait`), each as [seconds, tokens, end
   time of the last such step]. Each line is flushed when written, so a step killed by the budget leaves its log on S3.
3. **The throughput record** (`throughput.json` for the smoke; `metrics.json` → `throughput` in every arm), from the log by
   `n13_plan.step_rates`:
   - r_full = output tokens of the steps with at least 128 running / the wall seconds of those steps;
   - r_all = all output tokens / the generation's wall seconds;
   - tail_s = the wall seconds from the end of the last step with at least 128 running to the end of generation;
   - load_s = engine construction plus weight load (`VllmEngine(...)`), also written into `metrics.json` → `engine.load_s` before the
     first request;
   - the mean, median and maximum output length (`n_new`), the count at the cap, and the mean output tokens per answer with each
     answer clipped at 8,192 and at 16,384 (for the projection at either C);
   - informational, decides nothing: r_wait and tail_wait_s, the same quantities over the steps with requests waiting; the
     time-weighted mean running count.
   The engine block also records `max_num_seqs`, `max_num_batched_tokens` and `num_gpu_blocks` as vLLM resolved them.
4. **The projection** (`python -m infra.pg.n13_plan`, `n13_plan.project_smoke`; the command in `STATE.md`). For each smoke m:
   r_m = its r_full; L_m = its mean output tokens per MATH-500 answer at C; F_m = load_s + tail_s, seconds per engine load. It first
   recounts each `steps.jsonl` and asserts that the recount equals `throughput.json` and that the log's tokens equal the answers'. The
   smoke with the larger L_m / r_m (seconds per answer at the engine's limit) sets r, L_math and F for every arm. Then the §9.7
   arithmetic:
   - arm minutes = F / 60 + B + (500 × L_math + n_GSM × L_GSM) / r / 60, with n_GSM = 1,319 for the arms that carry GSM8K and 0
     otherwise;
   - B = 5 min for every arm built from the sets (`zero`, `id` and all S4 arms): §9.7's per-arm overhead, kept for the build and the
     14.6 GB write, which no smoke measures; B = 0 for `base` and `rl`;
   - L_GSM = 940, §9.7's assumption, **flagged**: no GSM8K length is measured before S3;
   - S3 minutes = 20 (setup, §9.7) + 75 (R1, A14) + the four gate arms; S4 minutes per instance = 20 + its arms;
   - S3 allowance = round(1.3 × S3 minutes) (`n13_plan.budget_min`).
   The command exits non-zero with a message when r_full is undefined for a smoke (no step with at least 128 running), and when
   spent + the S3 projection exceeds 35 GPU-h (§9.7's stop rule). It writes the S3 manifest only when both hold, with `--cap C`, the
   allowance, `--proj-tok-s r`, `--proj-mean-math L_math` and `--proj-load-min` F / 60 (+ 5 for built arms) on every gate step.
5. **Registered failing input** (`n13_dryrun.a22_planted_checks`, `studies/next/n13/offline_gate/rerun_a22/A22_planted.json`): a
   planted log of 2,000 steps at 100 running. The projection command exits 1: "A22: the base smoke's r_full is undefined: no engine
   step had at least 128 running sequences (largest running count 100); the projection refuses". No manifest is written. **Planted
   input with a known answer:** 1,005 steps of 0.1 s at 300 running and 300 tokens each, then 200 steps of 0.05 s at 10 running and 10
   tokens (block 100 holds 5 steps of each). It gives r_full 3,000.0 tokens/s (301,500 / 100.5 s), tail_s 10.0 s, r_all 2,746.61
   (303,500 / 110.5 s), and S3 148.46 min, equal to 20 + 75 + 4 × (70/60 + (500 × 1,011.67 + 1,319 × 940) / 3,000 / 60) + 2 × 5. The
   stop rule: the same input with 34.5 GPU-h spent exits 1 and writes no manifest. On the fake engine, the 300-problem smoke (3 answers
   per step, at most 128 running, 0.05 s per step) writes 10 lines for 100 steps; its recount equals `throughput.json`; its tokens
   equal the answers' `n_generated` (2,321 and 2,944); 58 steps ran at 128 (2.9 s), and the tail is 2.1 s.
6. **The attempt-8 manifest** `infra/aws_pg/steps/n13_s2smoke-08.json` = `n13_plan.manifest("s2smoke", 8, allowance_min=110)`: setup
   without the olmes venv, then `smoke_base` and `smoke_rl` on problems 0:300 with `--allowance-min 110`. Launched with
   `--budget-min 110`.

**Expected from source, before the run (the executor's prediction, not a measurement).** 128 is the n13 engine's own limit.
`LLMEngine.from_engine_args` uses `UsageContext.ENGINE_CONTEXT`, which is not in vLLM 0.32.0's default table, so `max_num_seqs` =
`SchedulerConfig.DEFAULT_MAX_NUM_SEQS` = 128 (`vllm/engine/arg_utils.py` lines 3123–3127; `vllm/config/scheduler.py` line 43). The KV
cache may bind first. Attempt 7's engine start logged 25.03 GiB of KV memory in 4 groups of 8 layers, "87,790 tokens, maximum
concurrency for 18,432 tokens per request: 4.76x" (`studies/next/n13/r0/s2setup-07/smoke_base/job_log_lines_11883-12185.txt` lines
51–54). The model has 32 key-value heads of 128 and 24 of its 32 layers use a 4,096-token sliding window (`config.json`). So one token
costs 16 KiB per layer, 128 KiB per group, and the pool holds 205,046 token slots. A sequence with context c ≤ 4,096 takes 4c of them.
So 128 sequences fit only while their mean context (prompt plus answer so far) is at most about 400 tokens. **Prediction:** in each
smoke the steps with 128 running are the first few hundred, about 275 decode steps after the prompts (the prompts of problems 0:300
average 125.1 tokens, counted on the laptop with the RL tokenizer; maximum 755). r_full therefore measures short
contexts, and it lies above the rate of the rest of the run. After that the running count falls to about 51,000 / mean context, about
20 to 50 sequences for answers of 1,000 to 3,000 tokens. **Refutation:** the RL smoke's steps with 128 running hold more than half of
its generation time (`steps.full_s` / `steps.gen_s` > 0.5), or r_full / r_wait < 1.5. If the prediction holds, a projection at r_full
understates S3 and S4. r_wait is the rate over the steps with requests waiting, which is a gate arm's state for most of its run. It is
recorded for the head, and `--rate r_wait` gives that projection. The frozen choice stays r_full until the head decides.

**Cost** (the head's estimate). About 300 × 2,500 tokens per model at a few hundred to about 1,500 tokens per second: 6 to 25 min per
model, plus setup. That is 0.5 to 1.2 GPU-h, against the arms it sizes, 21 to 34 GPU-h. A risk, not an estimate: if the RL answers
average far above 2,500 tokens (3.1's training rollouts ended at a mean of 12,756 on training prompts, §0.2), `smoke_rl` can reach the
110 min hard kill. The steps log, `generations.jsonl` and `metrics.json` are written during the run, so a killed step leaves them.

**Offline gate after A22:** `N13_DRYRUN_OK`, 97 of 97 (93 before; 4 new: the fake 300-problem smoke's log, the failing input, the
planted input with the stop rule, the committed manifest's equality; 1 replaced: the projection command on the fake smoke replaces the
check of the removed `project_from_smoke`), `studies/next/n13/offline_gate/rerun_a22/checks.json`.
The source lines behind A22, with each file's sha256: `studies/next/n13/offline_gate/rerun_a22/sources_a22.txt`.

#### A22, addition after the head's read of the prediction, 2026-11-06. **Head.**

Directed by the head on 2026-11-06, while attempt 8 ran (no change to the running job). Written by the seventh N13 executor before any
attempt-8 smoke data was read.

**Reason** (the head's). The prediction above follows from the source. With answers of thousands of tokens, the KV pool bounds the
running count, not `max_num_seqs`. So the steps with 128 running are only the opening of a run. A gate arm with 1,819 requests
spends almost all of its time bound by the KV pool, at a few dozen running sequences. So r_full overstates the sustained rate and
makes S3 and S4 look cheaper than they are.

1. **r_sat** = output tokens / wall seconds over the saturated steps. A step is saturated when requests are still waiting to be
   admitted: waiting = (submitted − finished so far) − running > 0. **How "finished so far" is derived.** It is the engine's own
   count, read after every step: `LLMEngine.get_num_unfinished_requests()` = the number of request states in vLLM's output
   processor (`vllm/v1/engine/output_processor.py` lines 526–527). A request leaves that set when its final output is returned or
   when it is aborted, and the endpoint arms never abort. So unfinished = submitted − finished so far, and the log's `unf` is that
   count. This is the log's class `wait` (unfinished > running) of item 2 above, so r_sat is exact from the log as written,
   attempt 8 included. Attempt 8's `throughput.json` names it `steps.r_wait`; `n13_plan.OLD_NAMES` maps the old names.
   **Limitations.** (a) "Running" is the count of requests that received tokens in the step. A request in the middle of a chunked
   prefill, or preempted, gets no token, so it counts as waiting. That matches the head's definition. (b) The count is read after
   the step, so a request that finished in the step counts as running but no longer as unfinished. At the end of the saturated
   phase, a step with no more waiting requests than finishers is classed as not saturated. This affects only the boundary.
2. **Recorded beside r_full and r_all**, with the mean running count and the mean context in the saturated phase. Mean running =
   Σ running × dt / Σ dt over the saturated steps. Mean context = Σ context × dt / Σ running × dt, where a running sequence's context
   is its prompt plus the tokens generated through the step. That is the time-weighted mean context per running sequence. **From the
   next run on**, `n13_gen` computes both in the job. `VllmEngine.step` sums prompt + tokens so far over the running sequences.
   `StepLog` keeps, per class, [seconds, tokens, end of the last such step, Σ running × dt, Σ context × dt]. `throughput.json` and
   `metrics.json` → `throughput` gain `r_sat`, `tail_sat_s`, `sat_run_mean` and `sat_ctx_mean`. **Attempt 8's log lacks the
   context sum** (it ran the code of commit `95bc1a775`). Its two means are reconstructed by `n13_plan.sat_phase` from the per-block
   unfinished counts and the answers in finish order (`generations.jsonl` line order, banked as `lengths.json`):
   - finished by the end of block b = 300 − the block's smallest unfinished count;
   - the answers at finish positions [fin(b−1), fin(b)) finished in block b and are placed at its last step;
   - the admission step is f − n_generated + 1 (one token per running sequence per step);
   - the context and the saturation flag then follow per step.
   **Its limitations:** the finish step is known only to within the block (10 steps). A preempted request's residency is longer than
   n_generated steps, and the reconstruction does not see it. The cross-checks it reports show both: the share of steps whose
   reconstructed running count lies within the block's logged [min, max], the reconstructed against the logged saturated seconds, and
   the time-weighted mean running count.
3. **The projection** (`n13_plan.project_smoke`, `python -m infra.pg.n13_plan`):
   - The rate for each smoke is min(r_full, r_sat), with the chosen rate's own tail: tail_s or tail_sat_s, the seconds after the
     last step of that class. So F = load_s + that tail.
   - The rest of item 4 above is unchanged: the smoke with the larger L / r sets the rate, L_GSM is 940 and flagged, B is 5 min.
   - The command also gives the S3 and S4 table at r_full alone and at r_sat alone (`by_rate`), so the difference is visible.
   - It refuses when the rule's rate is undefined for a smoke, which covers both "no step with 128 running" and "no step with a
     request waiting".
   - The registered A22 inputs keep their answers at `--rate r_full`. The 300-running input has no waiting step, so the rule refuses
     on it: "r_sat is undefined".
4. **Planted input** (the head's; `n13_dryrun.a22_planted_checks`, `studies/next/n13/offline_gate/rerun_a22add/A22_planted.json`). The
   planted log runs in three phases:
   - 20 opening steps of 0.04 s at 128 running, with 300 unfinished and 128 tokens per step;
   - 2,000 steps of 0.05 s at 24 running, with 200 unfinished (176 waiting) and 24 tokens per step;
   - 100 steps of 0.02 s at 10 running, with none waiting.

   The result:
   - r_sat = 50,560 / 100.8 s = 501.59 tokens/s; r_full = 2,560 / 0.8 s = 3,200;
   - mean running in the saturated phase 24.83, mean context 1,446.8 tokens (planted context 200 and 1,500 per sequence);
   - the rule takes r_sat, with a tail of 2.0 s;
   - S3 is 285.35 min, equal to the hand computation. At r_full alone S3 would be 143.42 min.

   **Reconstruction check:** an exact first-come-first-served engine (300 requests, at most 40 running, lengths 50 to 1,500, prompts
   60 to 300, one token per step), with its log's per-class sums stripped to attempt 8's format. Logged once per step, the
   reconstruction equals the log's own saturated seconds (154.83), mean running (40.0) and mean context (653.77). Logged per 10 steps,
   it is within 1 %.
5. **Offline gate after the addition** (and after the `s2smoke-08` setup fix, a job-file fix recorded in `N13_OFFLINE_GATE.md`
   §"R0 attempt 8"): `N13_DRYRUN_OK`, 100 of 100.
   - 97 before.
   - 2 new: the head's planted input and the reconstruction check.
   - 1 new for the setup fix.
   - Changed: the fake smoke's log check also asserts 57 waiting steps (2.85 s); the A22 planted check runs at `--rate r_full` and
     asserts that the rule refuses on it; the manifest check also covers `n13_s2smoke-09.json`.

   Record: `studies/next/n13/offline_gate/rerun_a22add/checks.json`.
6. **Attempt 8 gave no smoke data** (`s2smoke-08`: the engine stopped in torch.compile for want of `Python.h`; `N13_OFFLINE_GATE.md`
   §"R0 attempt 8"). So items 1 and 2 are in code and checked offline, but r_sat, the saturated phase's means and the table under both
   rates have not been computed on a real log. The prediction above is not yet tested.

### A23. R1's stage-2 deletion removed base `main` (§9.4 S2 step 3; A14; found before S3). Amend (job-file fix). **Head.**

Found on 2026-11-06 by the agent on the Vast lane and passed on by the head (brief to the eighth N13 executor). Written by the eighth N13
executor. No gate, threshold, prompt, sampling field or cap changes.

**The defect.** `infra/aws_pg/jobs/n13_rank.py` line 169 at `7fb507bd3` ran `shutil.rmtree(pathlib.Path(sn["stage2"]).parents[1])`
under `--delete-stage2`. The setup step downloads with `snapshot_download`, which lays a repo out as
`models--<org>--<name>/{blobs/<hash>, snapshots/<commit>/<file> -> ../../blobs/<hash>, refs/<branch>}`. Attempt 9's setup recorded
`/opt/hf-cache/hf/hub/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42…` for base `main`
(`studies/next/n13/r0/s2smoke-09/setup/setup.json`). Stage 2 (`c3c800dc…`) is a second snapshot of the same repo, so `parents[1]` of
its snapshot is the shared `models--allenai--Olmo-3-1025-7B` folder. The line removed base `main` with it. In the S3 manifest R1 runs
before `base`, `zero` and `id` (A14), and all three read base `main`. S3 would have failed at its first gate arm, after setup and R1
(about 95 min).

**The offline gate's miss.** The dry run ran R1 with `delete_stage2=False`, on stand-in snapshots in three separate folders
(`n13_dryrun.rank_checks`). So it never ran the deletion, and its layout could not show a shared repo folder. The line came in with
the offline gate's first commit (`017d3c88b`).

**Amendment (code).**

1. `n13_rank.delete_snapshot(snap)` removes one revision and nothing of another: `snapshots/<commit>`, the `refs` entries that name the
   commit, and the blobs that no other snapshot of the repo links to. Blobs shared with another revision stay (`config.json` and
   `tokenizer.json` can have equal content in base `main` and stage 2). It asserts the snapshot path has the cache layout, that every
   link resolves into the repo's `blobs/`, and that every other snapshot's links still resolve afterwards. `snapshot_download` writes a
   `refs` entry only for a branch or tag (`huggingface_hub.file_download._cache_commit_hash_for_specific_revision`), and the setup step
   passes commits, so on the instance the refs list is expected empty.
2. `run` calls `delete_snapshot(sn["stage2"])` and records its result in `metrics.json` → `stage2_deleted` (refs removed, blobs
   removed and kept, bytes freed).

**Registered wrong input and the check** (`n13_dryrun.a23_cache_checks`; `studies/next/n13/offline_gate/rerun_a23/A23_cache.json`).
A planted cache with relative symlinks as `snapshot_download` makes them: base `main` and stage 2 under one
`models--allenai--Olmo-3-1025-7B` folder, 5 files each, sharing 2 of their blobs (8 blobs in all), `refs/main` and
`refs/stage2-step47684`, and the RL repo beside it. After `delete_snapshot` on stage 2: base `main`'s 5 files read their planted bytes,
stage 2's snapshot, its refs entry and its 3 shard blobs are gone, the 2 shared blobs stay (5 blobs), the RL repo is whole, and
`huggingface_hub.scan_cache_dir` lists `main` only. The old code path on a fresh copy of the same cache removes base `main`. The check
also reads `run`'s source: it calls `delete_snapshot(sn["stage2"])` and has no `parents[1]`.

**Offline gate after A23 and A24:** `N13_DRYRUN_OK`, 103 of 103 (100 before; 2 new for A23, 1 new for A24),
`studies/next/n13/offline_gate/rerun_a23/checks.json`.

### A24. The projection from a smoke cut by the hard kill (§9.7; A22 item 4; found at attempt 9). Amend. **Head.**

Directed by the head on 2026-11-06 (brief to the eighth N13 executor, task 4: the rate from the base smoke when the RL smoke has no
`throughput.json`, flagged; the RL lengths with the cut answers at C). Written by the eighth N13 executor, who adds one reading for the
head's decision (item 3). No gate, threshold, prompt, sampling field or cap changes.

**Found at attempt 9** (`studies/next/n13/r0/s2smoke-09/`). The 110 min hard kill cut `smoke_rl` (STATUS `exit=137`, 03:21:26Z)
after 2,111.5 s of generation. 142 of 300 answers had finished; there is no `throughput.json`. A22 item 4 needs one per smoke.
Requests go in in problem order and the scheduler is first-come first-served, so every problem up to the largest finished index (161)
was admitted. Of the 158 unfinished answers, 20 lie in problems 0–161: 14 running and 6 waiting at the kill (vLLM's own log line
at 03:21:21Z: "Running: 14 reqs, Waiting: 144 reqs"; `smoke_rl/job_log_lines_1760-1772.txt`). The other 138 are problems 162–299,
all waiting at the kill. The step log counts 79,193 output tokens for all 158 together (log tokens 642,298 minus the finished
answers' 563,105; `smoke_analysis.json`).

**Amendment (code, `infra/pg/n13_plan.py`).**

1. `--killed rl` names a smoke cut by the kill. It must have no `throughput.json` and a `lengths.json` with its finished answers and
   `n_submitted`. Its `load_s` is read from its `metrics.json` → `engine.load_s`.
2. Its rate and tail are the other smoke's (flagged in `flags`). The cut log has no tail. Its own rates over the cut log are recorded
   beside (`killed.own_steps`).
3. Its L at C (`killed_smoke`), in two readings:
   - `prefix` (the default, proposed as the rule): the mean over the admitted prefix, problems 0 to the largest finished index. Each
     finished answer counts at min(n_generated, C), each cut answer at C (it had not stopped, and C bounds it). The problems after the
     prefix are left out. Attempt 9: (563,105 + 20 × 16,384) / 162 = 5,498.7, an upper bound. If every one of the 79,193 partial
     tokens belongs to the 20 cut answers, the lower bound is 3,964.8.
   - `all` (`--killed-reading all`; the brief's words read literally): every one of the 158 unfinished answers at C, over 300:
     10,505.9. This counts 138 answers about which the run gives almost no information (they hold at most 79,193 tokens among all
     158) as 16,384 tokens each.
   The prefix is a fair sample of MATH-500 only if the problem order is unrelated to answer length. On the base smoke, which
   finished all 300: mean 2,990.7 over problems 0–161, 2,786.0 over 162–299, 2,896.6 over all (`smoke_base/lengths.json`).
4. The S3 projection at C = 16,384, 7.062 GPU-h spent (`projection.json`, `projection_all.json`):
   - `prefix`: S3 21.11 GPU-h, allowance 1,647 min; it fits (7.062 + 21.11 ≤ 35). The command writes the S3 manifest.
   - `all`: S3 32.87 GPU-h; 7.062 + 32.87 > 35, so the stop rule refuses and no manifest is written.

**Registered wrong inputs and the planted check** (`n13_dryrun.a24_killed_checks`; `studies/next/n13/offline_gate/rerun_a23/A24_killed.json`).
Both smokes carry the head's sat24 log. Then the RL smoke loses its `throughput.json` and holds 200 answers of 100 tokens, finished
at problems 0–219 except the 20 with index mod 11 = 5, out of 300, with a load of 90 s. Results:
- `prefix`: L = (20,000 + 20 × 16,384) / 220 = 1,580.36.
- `all`: L = 5,528.0.
- Both take the base's rate (501.59) and tail (2.0 s), and S3 equals the hand computation (380.95 and 643.29 min).
- `--killed` on an uncut smoke exits non-zero, and so does the cut smoke without `--killed`.

### A25. G-COMPLETE fails on the base's real output: empty answers (§6.1 G-COMPLETE, line 616; §6.3 line 686). Amend. **Head.**

Found by the eighth N13 executor at the attempt-9 smoke reading and proposed by it (the text down to "The decision" is that
proposal, kept for the record). **Decided by the head, 2026-11-06 03:50Z**, before freeze 1. Written into the code by the ninth N13
executor.

**Found.** 7 of the base's 300 smoke answers are the stop token alone (n_new 0, n_generated 1, finish_reason `stop`; problems 67,
113, 141, 165, 167, 178, 274; `smoke_base/lengths.json`). G-COMPLETE requires "no empty text"
(`n13_score.g_complete`: `empty == 0`). It reads `pass: false` in the smoke's `metrics.json`. The RL smoke has no empty answer among
its 142 finished ones. A gate arm on base (500 + 1,319 answers) will very likely have some. `verdict_n13.py` then returns
CANCELLED(G-COMPLETE) (`GATES_PASS`). The registered response, "the arm is re-run under a new attempt", uses the same per-problem
seeds. A rerun is expected to reproduce the empty answers, though this has not been checked across engine loads.

**Proposed, the smallest change.** The emptiness condition reads "every answer holds at least one generated id (n_generated ≥ 1)". An
answer that is the stop token alone is the model's output. It is extracted as mode `none`, scored wrong, and still counted in
G-COMPLETE's `empty` field. The failing input is unchanged: a bank with one answer removed. A second failing input would be a record
with n_generated 0.

**The decision (the head, 2026-11-06 03:50Z).** The base did produce an answer in each of the 7 cases: an empty one, scored wrong. As
registered, those 7 would cancel the verdict. New text of the G-COMPLETE row of §6.1, from freeze 1 on:

> | G-COMPLETE | every arm | every problem (500, or 1,319) has exactly one record, every index once, seeds as registered, and every
> record has n_generated ≥ 1 and a finish_reason. Empty text (the stop token alone) is counted and reported per arm (`n_empty`), not
> gated: it is extracted as mode `none` and scored wrong | (1) a bank with one problem's record removed; (2) a record with n_generated 0;
> (3) a record with no finish_reason | the arm is re-run under a new attempt |

The cancel list of §6.3 is unchanged: G-COMPLETE still cancels when it fails.

**Code** (`infra/pg/n13_score.g_complete`). The record of G-COMPLETE holds `n`, `distinct`, `covers`, `seeded`, `generated` (every
n_generated ≥ 1), `finish_reason` (every finish_reason a non-empty string) and `n_empty`. `pass` reads the first six; `n_empty` is
reported only. `n13_gen` writes this record into each arm's `metrics.json` (the call is unchanged). `verdict_n13.gather_gates` adds
`n_empty` per arm × benchmark beside the pass values. The planted banks of the verdict check carry `n_generated` 101 and
`finish_reason` `stop`.

**Registered failing inputs and the planted check** (`n13_dryrun.scoring_checks`; `studies/next/n13/offline_gate/rerun_a26/A25_g_complete.json`).
The fabricated bank of 16 answers goes through the generation loop with the fake engine and the real tokenizer. Its row 15 is the
stop token alone: text "", n_new 0, n_generated 1, finish_reason `stop`, mode `none`, score 0, like the 7 smoke answers.
- The bank passes, with `n_empty` 1. Under the registered rule it would fail.
- A missing record (problem 3's record removed): fails.
- A record with n_generated 0 (problem 3): fails.
- A record with finish_reason None (problem 3; added to give the second half of the condition its own failing input): fails.
- The registered failing input, the bank with its last answer removed: fails.

**On the smoke's real records** (`studies/next/n13/amendments/a26_rescore.json` → `G-COMPLETE_on_300_submitted`):
- base, 300 records: the registered rule fails (`empty` 7); the amended rule passes, with `n_empty` 7.
- RL, 142 of 300 records (cut by the kill): both rules fail on `covers`, as they must; `n_empty` 0.

### A26. The `Answer:` label inside markdown emphasis (§2.4 item 2, lines 334–336, as amended by A10). Amend. **Head.**

**Decided by the head, 2026-11-06 03:50Z**, before freeze 1, from the eighth executor's smoke reading (`studies/next/n13/JOURNAL.md`,
2026-11-06 03:40Z, (c)). Written by the ninth N13 executor. This is a format variant decided from the smoke reading before freeze 1.
Its effect on the smoke scores is recorded below. No gate, threshold, prompt, sampling field, cap or grading rule changes.

**Found.** Mode `answer_line` requires a line to begin with `Answer:` after optional spaces. The base writes `**Answer:** …` on 45
lines of its 300 smoke answers (a laptop survey of lines that begin with a label; plain `Answer:` lines: 102). Those lines are missed.
Read answers 0 and 8 are right answers scored wrong for this reason (`**Answer:** (3, π/2)` and `**Answer:** $3\sqrt{13}$`). No
smoke answer writes `*Answer:*`, `__Answer:__`, `**Answer**:` or `***Answer:***`.

**New text of item 2** (replaces A10's text from freeze 1 on):

> 2. Extraction mode `answer_line`: a label line is a line that begins, after optional spaces, with one of `Answer:`, `**Answer:**`,
>    `*Answer:*` or `__Answer:__` (the label alone or inside one markdown emphasis pair, case as written). The extracted string is the
>    text after the label on the last label line whose remainder is non-empty after stripping, up to the end of that line. A label
>    line with nothing after it is skipped (A10). If no line qualifies, mode `boxed`: `boxed_last(text)`. If none, mode `none` (scored
>    wrong).

Not accepted, as before: `**Answer**:`, `**Final Answer:**`, `Final Answer:`, `### Answer:`, `answer:`, `ANSWER:`, `***Answer:***`, and
mismatched pairs such as `**Answer:*`. Emphasis around the answer text itself (`**Answer:** **42**`) is not removed.

**Code** (`infra/pg/n13_score.py`). `ANSWER_LABEL = re.compile(r" *(?:(\*\*|\*|__)Answer:\1|Answer:)")`, matched at the start of
each line; the backreference requires the closing marker to equal the opening one. `extract` takes the text after the match. For a
line that begins with plain `Answer:`, the match ends where `len("Answer:")` ended before, so every extraction the old code made from
such a line is unchanged. A new label line can still be the last label line, and so replace an earlier plain one.

**Planted check** (`n13_score.selfcheck`; `n13_dryrun.FAB` rows 13 and 14, through the generation loop):
- `**Answer:** 42` extracts `42` in mode `answer_line`, scored right, with its answer-line log-probability over 1 token.
- `**Answer:**` with nothing after it, then `\boxed{5}`, falls to the box: mode `boxed`, `5` (A10's rule on the new label).
- Selfcheck: `*Answer:* 7` and `  __Answer:__ 8` extract; the last label line wins over an earlier plain one; the span of
  `  **Answer:**  (3, \pi)  ` is exactly `(3, \pi)`; the 6 forms listed as not accepted give mode `none`.

**Grading checks of A1 and A10, rerun** (the grading rule is unchanged; `studies/next/n13/offline_gate/rerun_a26/G-GRADER.json`,
`scoring_fabricated.json`): 500 of 500 golds equal to themselves; 24 of 24 planted equivalent pairs true; 16 of 16 non-equivalent pairs
false; derangement rate 0.006 against the bound 0.016; the derangement scored as the identity gives 3 of 500 and fails, as it must.
A10's planted row 12 (`Answer:` empty, then `\boxed{5}`) is scored 1 in mode `boxed`.

**Effect on the smoke** (`studies/next/n13/amendments/a26_rescore.py`, output `a26_rescore.json`; inputs the two S3
`generations.jsonl` files, sha256 `4ab1220b…` (base) and `a01c3965…` (RL)). The code before A26 (`f55da49b6`), re-run on the laptop,
reproduces the recorded mode, extracted string and score of all 300 base and all 142 RL records.
- **Base: 153 → 177 of 300 right (0.510 → 0.590).** 24 answers change from wrong to right: problems 0, 8, 16, 37, 44, 71, 117, 158, 180,
  185, 199, 206, 215, 229, 231, 241, 249, 251, 257, 259, 261, 262, 269, 297. None changes from right to wrong.
- 17 more base answers change their extraction but not their score: 15 right answers move from mode `boxed` to `answer_line`
  (problems 14, 22, 61, 69, 80, 92, 95, 106, 112, 190, 202, 220, 247, 256, 273), and 2 wrong ones from `none` to `answer_line`
  (20: `$117 + 41i$` against `6+9i`; 50: `$Answer: 256$` against `203`).
- Base modes: `answer_line` 97 → 138, `boxed` 102 → 87, `none` 101 → 75.
- **RL: 131 of 142 right, unchanged.** No RL record changes: RL writes no label line (modes `boxed` 139, `none` 3).

This change raises the base's MATH-500 accuracy and so lowers G (§8 threat 5: format against mathematics). On the smoke the
lowering is 24 answers of 300, 8.0 points.

### A27. No study ceiling; each launch's allowance is 1.3 times its projection (§9.7). Amend. **Head.**

**Decided by the user, 2026-11-05 21:42 CST (2026-11-06 03:42Z), in the words "Remove the GPU limits."**, passed on by the head
(extension of the brief to the ninth N13 executor, 2026-11-06). Written by the ninth N13 executor, before freeze 1.

**Why.** At the measured rate (236.6 output tokens per second, A22 and A24), S3 alone takes 21.1 GPU-h and S4's first priority tier
20 to 21 GPU-h. Under the 35 GPU-h ceiling no S4 tier could follow S3 (`STATE.md` of 2026-11-06 03:50Z).

**The rule, from freeze 1 on** (replaces the "Target", "Ceiling" and "Stop rule" paragraphs of §9.7 and its last paragraph on pair 2):
1. There is no study ceiling. The full roster runs, on AWS L40S instances (`g6e.2xlarge`, `g6e.4xlarge`). Vast stays reserved for the
   user's other agent.
2. Each launch keeps a hard kill as runaway protection. Its allowance is 1.3 times its own projection at the measured rate:
   `n13_plan.budget_min(projected_min) = round(1.3 × projected_min)`, the same factor the eighth executor's table used. The allowance
   is written into every gen step of the manifest (`--allowance-min`) and passed to pg as `--budget-min`.
3. The park check inside a launch is unchanged and reads only that allowance: before each gen step, the launch's elapsed time plus
   the step's projection is compared with the allowance; a step that would exceed it is PARKED, with every later step of that
   manifest. A comparison whose arms are parked is INDETERMINATE and listed.
4. The check "spent GPU-h plus the launch's projection ≤ 35" is removed. `n13_plan` no longer has `CEILING_GPU_H`, and its projection
   command no longer refuses or withholds a manifest for it. It records `--spent` and the cumulative GPU-h after S3 and S4.
5. Pair 2, if it runs, is projected and launched the same way.

**Code.** `infra/pg/n13_plan.py`: `CEILING_GPU_H` removed; `main` writes the manifests whenever the rule's rate is defined;
`projection.json` gains `ceiling: "none (A27)"`, `cumulative_after_s3_gpu_h`, `cumulative_after_s4_gpu_h` and, per launch,
`launches.<kind>.{min, gpu_h, allowance_min, arms}`; it loses `left_for_s4_gpu_h`, `s3_fits` and `s4_fits_through_priority`.

**Registered input and checks** (`n13_dryrun.a22_planted_checks`, `a27_a28_checks`; `studies/next/n13/offline_gate/rerun_a27/`).
- The planted input that tested the stop rule (34.5 GPU-h spent, S3 over 35) now must exit 0 and write all six manifests. Before A27
  it had to exit non-zero and write none.
- Every launch's allowance equals round(1.3 × its projection), in every gen step's `--allowance-min`.
- The s3gate2 projection equals the hand computation 20 + 2 × (F + 5 + (500 L + 1,319 × 940) / r / 60).

### A28. S3 in two launches, S4 in four (§9.4 S3 and S4; §3.3; A14). Amend (layout). **Head.**

Directed by the head, 2026-11-06 (the same extension as A27), so that S3 takes about half the wall clock and S4 uses the four g6e
instances the AWS quota allows at once. Written by the ninth N13 executor, before freeze 1. No gate, threshold, prompt, sampling field,
cap, arm, seed set or priority changes.

**Can `zero` and `id` be built without R1's output?** Yes, as read from the code at `e897d9b53`:
- `n13_arms.arm_spec("zero", sets)` returns `{"kind": "none", "non_mlp": "before"}` and `arm_spec("id", sets)` returns
  `{"kind": "all", "non_mlp": "after"}`; neither reads `sets`.
- `n13_arms.build_arm` builds them from the base and RL snapshots alone (§3.1: M = 0 or M = 1), and G-HASH compares them with the
  same snapshots.
- The only use of R1's output was that `n13_gen.run` read the `--sets` file for every built arm before calling `arm_spec`. It now
  reads it only for arms other than `zero` and `id`. The S3 manifests pass no `--sets`; `base` and `rl` never read it.
- Setup needs only base `main` and RL `main` for them (`--models base,rl`); stage 2 is needed only by R1.

**The layout.**
- S3: `s3gate-01` = setup with stage 2 (`--models base,rl,stage2`), `r1_rank --delete-stage2`, `base`, `rl`; `s3gate-02` = setup
  (`--models base,rl`), `zero`, `id`. Each gate arm on MATH-500 and GSM8K (§3.3). Code kinds `s3gate1` and `s3gate2`
  (`n13_plan.S3_SPLIT`); the launch prefixes are the head's, `results/next_n13/s3gate-01` and `s3gate-02` (`n13_plan.MANIFEST_FILE`).
  A relaunch of either takes the next free number.
- S4: `s4armsA1`, `s4armsA2`, `s4armsB1`, `s4armsB2` (`n13_plan.S4_SPLIT`). Each instance's priority-ordered list of §9.4 is split by
  turns: its 1st, 3rd, 5th … arm to manifest 1, its 2nd, 4th … to manifest 2. Every manifest starts with a P1 arm, keeps priority
  order, and holds its P1 and P2 arms among its first two arms; with four instances at once, the four P1 arms run first, then the
  two P2 arms beside two P3 arms.

| launch | arms in order (priority) |
|---|---|
| s4armsA1 | `top_k1` (1, +GSM8K), `mlp` (2, +GSM8K), `top_k0.3` (3), `dare_k3` (4), `dare_k0.1` (4), `comp_k3` (5), `top_k1_B` (6) |
| s4armsA2 | `dare_k1` (1, +GSM8K), `top_k3` (3), `top_k0.1` (3), `dare_k0.3` (4), `comp_k1` (5, +GSM8K), `rnd0_k1` (6), `dare_k1_d2` (6) |
| s4armsB1 | `rndE_k1` (1, +GSM8K), `attn` (2, +GSM8K), `rndT_k3` (3), `rndT_k0.3` (4), `rndT_k0.1` (4), `comp_k0.1` (5), `topF_k1` (7, conditional) |
| s4armsB2 | `rndT_k1` (1, +GSM8K), `rndE_k3` (3), `rndE_k0.3` (4), `rndE_k0.1` (4), `comp_k0.3` (5), `rndT_k1_d2` (6) |

**Projection per launch** (`n13_plan` on the banked smoke, A24 `prefix`, rate r_sat 236.6, L_math 5,498.7, L_gsm 940 flagged, F 9.35 min
per load, build 5 min per built arm; `studies/next/n13/r0/s2smoke-09/projection.json` → `launches`):

| launch | projection, min | GPU-h | allowance, min (1.3 ×) |
|---|---|---|---|
| s3gate-01 (setup 20, R1 75, base, rl) | 675.8 | 11.26 | 879 |
| s3gate-02 (setup 20, zero, id) | 610.8 | 10.18 | 794 |
| s4armsA1 | 1,651.0 | 27.52 | 2,146 |
| s4armsA2 | 1,651.0 | 27.52 | 2,146 |
| s4armsB1 (with the conditional arm) | 1,651.0 | 27.52 | 2,146 |
| s4armsB2 | 1,355.6 | 22.59 | 1,762 |

S3 in two launches is 21.44 GPU-h (one setup more than the single launch's 21.11). S4 in four is 105.14 GPU-h. With the 7.062 spent,
the cumulative is 28.50 after S3 and 133.65 after S4.

**What the split adds, stated.** The seed-set-B gate arms (`zero`, `id`) now run on a different instance than `base` and `rl`. σ is
read from base-A against zero-B (§0.5), so any instance-to-instance difference in the engine enters σ. S4 already compares arms on other
instances with base and rl, so this kind of comparison is in the design. Both launches use the same AMI and the same setup step; each
banks its pip freeze and its engine block (`metrics.json` → `engine`), and the two are compared before freeze 2.

**Code.** `infra/pg/n13_plan.py`: `S3_SPLIT`, `S4_SPLIT`, `LAUNCHES`, `MANIFEST_FILE`; `roster` names each arm's launch;
`project_launch` and `manifest` take the six kinds (the single-launch kinds stay in `project_launch` only, to reproduce §9.7's table);
`--manifest-dir` replaces `--manifest`. `infra/aws_pg/jobs/n13_gen.py`: `zero` and `id` read no sets file.
`infra/aws_pg/jobs/n13_dryrun.py`: the roster run builds `zero` and `id` without `--sets`; `a27_a28_checks`; `manifest_checks` over the
six kinds.

**Checks** (`studies/next/n13/offline_gate/rerun_a27/A27_A28_manifests.json`): the S3 split (base and rl after setup with stage 2 and
R1; zero and id after setup without stage 2); the S4 split (each manifest in priority order from P1, every P1 and P2 arm among the
first two arms, the union the roster's 27 S4 arms); every roster arm once across the six launches; `zero` and `id` built and G-HASH
passed without `--sets`.

**Offline gate after A27 and A28:** `N13_DRYRUN_OK`, 107 of 107 (105 before, 2 new), `studies/next/n13/offline_gate/rerun_a27/checks.json`.

## Amendments after freeze 1

### A29. The c = 1 builder check on entries whose float32 difference is not exact (§3.1 lines 389–391 as amended by A6; §0.5). Amend. **Head.**

**Approved by the head on 2026-11-06 as built.** It is the first amendment after freeze 1 (`n13-freeze-1`, 2026-11-06T04:00:49Z). The
code is tagged `n13-freeze-1a` (§0). The frozen rules of freeze 1 are unchanged: no gate, threshold, arm, set, prompt, sampling field
or cap changes. **No bit of any built arm changes.** Scale-1 arms copy the RL bits wherever M = 1, and the formula was only a check on
them. G-HASH is the gate on that (`id` against the RL snapshot, `zero` against the base snapshot, every tensor by sha256).

Found by the ninth N13 executor on 2026-11-06 at 05:35Z, when R1 stopped at `s3gate-01`. **Cause, in one line:** the offline gate's
stand-in weights had no entries with an inexact Δ, and R1's own statistics count 152 in the MLP, 144 in attention and 17 in the
embedding on the real weights.

**Found.** `s3gate-01` (`i-0f87b6f6fad1ff3aa`) passed setup and R1 stages 1 to 5. These are G-PARTITION, G-INIT (a) and (b), the
weight statistics, PC-RANK, PC-ART, J_floor, J_round and `sets.json`. Stage 6 builds every roster arm in memory for its hashes. It built
`zero`, then stopped on the first MLP tensor of `id`:
`AssertionError: model.layers.0.mlp.down_proj.weight: the formula does not return theta_a` (job.log lines 1030–1048, banked in
`studies/next/n13/s3/s3gate-01/job_log_lines_1027-1050.txt`). STATUS `exit=1` at 05:33:52Z. `base` and `rl` did not run.

**Cause.** For c = 1 arms `n13_arms.build_tensor` writes the after bits where M = 1 (`T = np.where(M, a16, b16)`). It then asserts
that bf16(float32(θ_b) + Δ) gives those bits on every entry with Δ ≠ 0 (A6's text). §3.1 claims this only "when the float32
difference is exact (§0.5)", and §0.5 says the difference is exact only when the two exponents differ by at most 16. The real weights
hold such entries: R1's `weight_stats.json` counts 152 inexact MLP entries, 144 in attention and 17 in the embedding. Where
|θ_b| ≫ |θ_a|, the rounded Δ leaves the sum one or more bf16 steps from θ_a. A second case also breaks the check: θ_a = −0 with
θ_b ≠ 0, where the sum is +0. R1 does not count that case. Planted pairs reproduce both (`studies/next/n13/offline_gate/rerun_a29/A29_builder_check.json`):
- before 0.0123 (`0x3c4a`), after 1.3e-8 (`0x325f`): the formula gives `0x3260`;
- before 0.5, after 2.9e-6 (exponents 18 apart): the formula gives `0x3644` against `0x3643`;
- before 0.0123, after −0: the formula gives `0x0000` against `0x8000`.
The offline gate's stand-in weights had none of these entries, so the check never fired there.

**The built arms were never wrong.** T copies the after bits wherever M = 1. G-HASH, which compares every tensor's sha256 with the
RL snapshot, is the gate on `id`. The check that stopped R1 only tests the formula.

**New text of §3.1 lines 389–391** (replacing A6's text):

> For c = 1 arms the builder copies the after tensor's bits where M = 1 and the before tensor's bits where M = 0. Wherever M = 1,
> Δ ≠ 0 and the float32 difference is exact (§0.5), it asserts that the formula above gives the same value (+0 and −0 equal). The
> other entries (inexact Δ, counted by R1 per family in `weight_stats.json`; an after value of −0) are written from the after bits
> and are not checked by the formula. R1 counts signed-zero swaps per family. For c ≠ 1 the formula is applied where M = 1 and Δ ≠ 0,
> and the before bits are kept elsewhere.

**Code** (`infra/pg/n13_arms.py`). `inexact(b16, a16)` returns the mask that `inexact_count` already summed. The c = 1 branch of
`build_tensor` asserts on `M & (Δ ≠ 0) & ~inexact`, comparing values (float32 of the bf16 bits) rather than bit patterns.
`a29_planted` holds the planted check, which `selfcheck` runs. The c ≠ 1 path is unchanged.

**Registered inputs** (`n13_dryrun.a29_checks`).
- The 7 planted pairs: 3 exact, 3 inexact, and 1 after value of −0. The c = 1 builder writes the after bits on all 7. The pre-A29
  check fails on them, with 3 entries off, as it failed on the real weights.
- Wrong input for the A29 check itself: `delta()` scaled by 1 + 2^-6 on every entry. `build_tensor` must raise its assertion, and it
  does.

**Offline gate after A29:** `N13_DRYRUN_OK`, 108 of 108 (`studies/next/n13/offline_gate/rerun_a29/checks.json`). With the relaunch
manifests' check below: 109 of 109 (`rerun_a29b/`).

**Prediction, logged before `s3gate-02` reaches `id`.** `s3gate-02` runs the pre-A29 code. After `zero` finishes, its `id` step will
stop in the build with the same AssertionError on `model.layers.0.mlp.down_proj.weight`, before the engine loads. Refuted if `id`
builds.

**What the gate cannot show.** Whether R1 stage 6 and the later R1 stages (G-HASH, G-MASK, the DARE kept counts, the energy check)
pass on the real weights. They did not run.

**The relaunch (the head, 2026-11-06).** There are two launches with fresh prefixes. Each allowance is round(1.3 × its projection)
(A27). The rate is 236.6 tokens per second, L_math 5,498.7 and F 9.35 min, as in A28. The manifests come from
`n13_plan --r1-min 127.2 --relaunch s3gate1=n13_s3gate-03.json s3gate2=n13_s3gate-04.json:id` (projection
`studies/next/n13/s3/relaunch_projection.json`).
- `s3gate-03` = setup with stage 2, R1, `base`, `rl`, each with GSM8K. Projection 728.0 min, allowance 946 min. R1 is projected at
  127.2 min, not the spec's 75. That is 75.7 min measured at `s3gate-01` (setup's end at 04:18:10Z to STATUS at 05:33:52Z: stages 1–5,
  `zero`'s in-memory build, the start of `id`), plus the rest of stage 6. The rest is 32 builds or snapshot hashes (29 roster arms, the
  G-HASH and G-MASK failing inputs, the two snapshot hashes, less `zero`), each at `zero`'s measured build time of 96.6 s.
- `s3gate-04` = setup without stage 2, then `id` with GSM8K. Projection 315.4 min, allowance 410 min.
- `zero` from `s3gate-02` stands if it completes with G-HASH and G-COMPLETE passing (G-HASH passed at its build). `s3gate-02`'s `id`
  step is left to fail on its own and is recorded.
- **R1 reproducibility.** The R1 rerun in `s3gate-03` must reproduce `s3gate-01`'s `sets.json` bit for bit: sha256
  `49a1065c57050f9a0069a7a9a08a3aa2a846c5e26463664a7381d9339ee61557`, 740,846 bytes (`studies/next/n13/s3/s3gate-01/r1_rank/sets.json`).
  A29 touches no code that writes it. A difference stops S3 and goes to the head.
- Relaunch check (`n13_dryrun.a27_a28_checks`, `rerun_a29b/A29_relaunch_manifests.json`): on the fake smoke, `--r1-min 100` gives a
  first manifest of setup with stage 2, R1, `base` and `rl`, at round(1.3 × (its projection with R1 at 100 min)). The second holds setup
  without stage 2 and `id` only, at round(1.3 × (20 + F + 5 + (500 L + 1,319 × 940) / r / 60)).

### A30. The rndE arms realize their intended MLP change energy within 5 % (§3.2 line 434; §3.1). Amend. **Head.**

**Decided by the head, 2026-11-06**, after R1 at `s3gate-03`. It is the second amendment after freeze 1 and comes before freeze 2. **Accepted as built by
the head, 2026-11-06**: the square-root adjustment is the right reading, the pass-1 overshoot and the second pass inside the band are
recorded, and rndT stays report-only. It
touches no S3 arm (`base`, `rl`, `zero`, `id`), so freeze 1 is not re-tagged. Written by the ninth N13 executor.

**Found.** R1's arm audits on the real weights (`studies/next/n13/s3/s3gate-03/r1_rank/reproduce_s3gate01.json`) show the four `rndE`
arms realizing 0.700 (k0.1), 0.822 (k0.3), 0.774 (k1) and 0.811 (k3) of their intended MLP energy c² E(R) = E(S_k). That is outside
§3.2's 5 %, so the arms were flagged. **Cause:** c lies in [1.15, 1.47], and a change of one bf16 step times c rounds back to one step
under round-to-nearest. One-step entries are 1.08e9 of the 1.89e9 changed MLP entries (R1 `weight_stats.json`: one_step 1,079,297,769,
above_floor 813,964,493).

**The rule, from freeze 2 on** (added to §3.2's "Realized, not intended"):

> For `rndE_k` the builder rounds θ_b + c·Δ to bf16 stochastically on the entries of the mask with Δ ≠ 0. It goes to the upper
> neighbouring bf16 value (away from zero) with probability equal to the value's fractional position between its two neighbours, and to
> the lower one (toward zero) otherwise. One random stream per tensor, `default_rng([140 + j, layer, w])` with j = 0, 1, 2, 3 for
> k = 0.1 %, 0.3 %, 1 %, 3 % and w = 0, 1, 2 for gate, up, down (A8's convention; seed 140 + j is new). After a build the realized
> MLP energy is computed. If it is not within 5 % of c₀² E(R) (c₀ from `sets.json`), c is multiplied by sqrt(intended / realized)
> and the arm is built again, at most three builds. Otherwise the build fails loud. The c that built the arm is recorded
> (`c_built`, `energy_passes`).

The factor is the square root of the energy ratio because energy scales as c². "Adjust c once by the ratio" in the head's text is read
that way.

**Not changed.**
- `sets.json` is unchanged: R1 stages 1–5 write it (`n13_arms.select_sets`, `rnd_e`, `rnd_t`, `weight_stats`), and none of them calls
  the builder.
- The bits of the `top`, `comp`, `rnd0`, `topF`, `rndT`, `mlp`, `attn`, `zero` and `id` arms are unchanged: they are c = 1 arms, and
  stochastic rounding applies only where the spec says `rounding: stochastic`, which only `arm_spec("rndE_*")` sets.
- The bits of the `dare` arms are unchanged. They are c ≠ 1 but keep round-to-nearest: c = 1/q is 33 to 1,000, so the realized energy
  is within 0.7 % already (R1: 0.0002 to 0.0065).
- Only the four `rndE` arm hashes change.
- Checked on the offline gate's stand-in (`studies/next/n13/amendments/a30_compare.py`, `a30_compare.json`): against the A29 dry
  run, `sets.json` is equal byte for byte, and of the 29 arm hashes only the four `rndE` ones differ.

**rndT, checked and reported, not asserted.** `arm_spec("rndT_*")` now carries the intended energy E(S_k), and the audit records the
ratio and its flag. It is not asserted. An `rndT` arm is c = 1 and is defined as the shortest tail prefix with E ≥ E(S_k), so its
overshoot is bounded by one neuron's energy, not by 5 %. On the stand-in the overshoot is 33 % at k0.1, where few tail neurons carry
much energy, and the short branch fires at k0.3, k1 and k3. On the real weights (R1 at `s3gate-03`, bit copies) realized / E(S_k) is:
- k0.1: 1.0011 (765 neurons);
- k0.3: 1.0003 (1,861);
- k1: 1.0001 (5,221);
- k3: 1.0000 (13,910);
- `rndT_k1_d2`: 1.0001 (5,203).
None is short.

**Planted check** (`n13_arms.a30_planted`, `n13_dryrun.a30_checks`; `studies/next/n13/offline_gate/rerun_a30/A30_energy.json`). The
planted arm is one 64 × 96 `down_proj` tensor whose every change is one bf16 step up, all columns in the set, c = 1.3 and intended
energy 1.69 E_R.
- Deterministic rounding realizes 0.5917 = 1/1.69 of the intended. With the c adjustment it still fails after three builds
  (AssertionError, run in a subprocess as the registered wrong input).
- Stochastic rounding: pass 1 at c = 1.3 realizes **1.120**, also outside 5 %. Stochastic rounding is unbiased in value, so its
  variance adds f(1 − f) step² per entry to the energy. Pass 2, at c = 1.2284, realizes 0.9921, inside the band.
- So the stochastic rule meets 5 % through the c adjustment, not at c₀. On the real `rndE` arms (one-step entries 57 % of the changed
  ones) pass 1 is expected above 1 by a few percent, and pass 2 within the band.
- The stochastic cast rounds up 0.2998 of the time at 0.3 of a step (200,000 draws).

**Offline gate after A30:** `N13_DRYRUN_OK`, 110 of 110 (`studies/next/n13/offline_gate/rerun_a30/checks.json`). The stand-in `rndE`
arms build in one pass each (ratios 0.9995 to 1.0004; their changes are not one-step dominated).

**Where it acts.** The S4 `rndE` arms (P1 `rndE_k1`, P3 `rndE_k3`, P4 `rndE_k0.3` and `rndE_k0.1`) and R1's stage-6 hashes when R1
runs again. Each `rndE` build takes up to three in-memory builds plus the written one, about 1.5 min each.

## Amendments after freeze 2

### A31. The written `rndE` arm against its matched build: every tensor bit for bit, the MLP energy within 1e-12 relative (A30's check; §3.2 "Realized, not intended"). Amend. **Head.**

**Decided by the head on 2026-11-07** (its brief to the eleventh N13 executor: compare the two energies within a registered relative
tolerance, both sides recorded; an arm whose written tensors differ by one bf16 step in one entry must still fail). It is the first
amendment after freeze 2 and the third after freeze 1. No gate, threshold, arm, set, prompt, sampling field or cap changes. **No bit of
any arm changes:** A31 replaces a check, not a build. Written by the eleventh N13 executor.

**Found.** Two S4 launches stopped in their first `rndE` build, at A30's check
`assert audit["families"]["mlp"]["realized_sq"] == rec[-1]["realized"], "A30: the written arm differs from the matched build"`
(`infra/pg/n13_arms.py` line 611 at `628f61b2b`):
- `s4armsB2-01` (`i-0acc7de3158361b23`) at `rndE_k3`, STATUS `exit=1` at 2026-11-07T02:44:04Z, after `rndT_k1` (with GSM8K) had
  completed. job.log lines 2319–2347 are in `studies/next/n13/s4/s4armsB2-01/job_log_lines_2319-2347.txt`.
- `s4armsB1-01` (`i-07f12939f48abecc4`) at `rndE_k1`, its first arm, STATUS `exit=1` at 2026-11-07T03:46:52Z. job.log lines 1028–1056
  are in `studies/next/n13/s4/s4armsB1-01/job_log_lines_1028-1056.txt`.
- **What it cost:** the two launches, 3.705 h and 0.265 h from launch to STATUS on g6e.2xlarge, 3.970 GPU-h, about $8.89 at $2.24 per
  hour. `rndT_k1` from `s4armsB2-01` is kept; it is not rerun.

**Cause.** The two energies are float64 sums over the same 96 MLP tensors taken in different orders. `_build_arm` adds each tensor's
sum of squares to `families["mlp"]["realized_sq"]` as it visits the tensor. The in-memory build that `energy_match` runs (no `out_dir`)
visits the tensors in sorted-name order (`Weights.names()`). The written build visits them shard by shard (`sorted(by_shard.items())`),
and the three shards of the real snapshot do not hold the tensors in sorted-name order. Each tensor's bits and its own sum are the same
in both builds: the stochastic rounding stream is `default_rng([140 + j, layer, w])`, one per tensor, independent of the visiting
order. On the real weights each tensor's sum has rounded mantissa bits (tens of millions of squares over a wide range of magnitudes),
so the two orders round differently in the last bits, and the exact comparison fails. The offline gate missed it. On its stand-in (4
layers, MLP magnitudes within a few binades) every float64 sum of squares is exact, so the order cannot matter; 32 layers alone stay
exact. With MLP magnitudes spread over 20 or 30 binades the two sums differ by 1.3e-16 to 7.5e-16 relative while every tensor is equal.
Neither side was written on the instances before the assertion, so the size of the difference on the real weights is not known. It is
reasoned from the code and reproduced on the stand-in.

**The bound.** A float64 recursive sum of n non-negative terms is within (n − 1) · 2^-53 of the exact sum, relative. Two orders of the
96 MLP tensors therefore differ by at most 2 × 95 × 2^-53 = 2.1e-14, relative.

**The rule, from A31 on** (replacing A30's exact comparison). The written `rndE` arm passes when:
1. every tensor's sha256 equals the matched in-memory build's; and
2. the two MLP energies agree within **1e-12, relative** (`A31_REL_TOL`, about 50 times the bound).

Otherwise the build fails loud. Both energies, their relative difference, the tolerance and the count of differing tensors are
recorded in `build.json` → `a31` and in each benchmark's `metrics.json` → `build.a31`.

**Why the tensors are compared too.** An energy comparison alone cannot make the one-step input fail. A one-step change in a small
entry moves the energy by less than the summation noise. On the planted input below it moves it by nothing visible: the relative
difference is 7.5e-16 with the bump and without it. The sha256 comparison catches any changed bit, and it costs nothing: both builds
already hash every tensor (§3.4). The tolerance 1e-6 (the head's figure for a summation difference) would leave the energy part as
blind as 1e-12 here, and blind to larger changes as well; 1e-12 is what the bound supports.

**Code.** `infra/pg/n13_arms.py`: `A31_REL_TOL = 1e-12`; `a31_compare(written, matched)`; `build_arm` keeps the matched build's audit,
records `audit["a31"]` and asserts its `pass`. `infra/aws_pg/jobs/n13_gen.py` copies `a31` into the `metrics.json` build block.

**Registered inputs** (`n13_dryrun.a31_checks`; `studies/next/n13/offline_gate/rerun_a31/A31_written_vs_matched.json`). A 32-layer
stand-in (96 MLP tensors, as the real model) whose MLP magnitudes are spread over 30 binades, written in 3 shards with the names dealt
round-robin; one stochastic-rounding arm (192 neurons, c = 1.3, seed 142):
- The pre-A31 check: matched 1.1252463099126716e-05, written 1.1252463099126724e-05, not equal, while every tensor is equal (sha256,
  and the written shards re-read). It fails, as on the two instances.
- A31: passes; relative difference 7.5e-16 ≤ 1e-12; 0 of 355 tensors differ.
- Wrong input: in the written build only, the first changed entry of `model.layers.0.mlp.gate_proj.weight` one bf16 step up.
  `build_arm` raises the A31 assertion with 1 tensor differing. The energy part alone passes it (relative difference 7.5e-16).

**Offline gate after A31:** `N13_DRYRUN_OK`, 111 of 111 (`studies/next/n13/offline_gate/rerun_a31/checks.json`). The four stand-in
`rndE` arms of the roster run pass A31.

**What the gate cannot show.** The size of the summation difference on the real weights. The first `rndE` build after A31 records both
sums.

**The relaunch (the head's brief).** `s4armsB1-02`: B1-01's seven arms (`rndE_k1` and `attn` with GSM8K, `rndT_k3`, `rndT_k0.3`,
`rndT_k0.1`, `comp_k0.1`, `topF_k1` under its `--only-if`); projection 1,651.0 min, allowance 2,146. `s4armsB2-02`: `rndE_k3`,
`rndE_k0.3`, `rndE_k0.1`, `comp_k0.3`, `rndT_k1_d2`; projection 1,060.2 min, allowance 1,378. Allowance round(1.3 × projection) (A27);
projection with the assumptions of `studies/next/n13/r0/s2smoke-09/projection.json`. Fresh prefixes; inputs equal to the -01
launches' (`sets.json` `49a1065c…`, `thresholds.json` `60bf6d4e…`, `controls.json` `f98c30e4…`). Manifests
`infra/aws_pg/steps/n13_s4armsB1-02.json` and `n13_s4armsB2-02.json`, written by `studies/next/n13/s4/tools/manifests_02.py`
(`n13_plan.manifest(kind, 2, …)`; it first checks that `manifest(kind, 1, …)` reproduces the committed -01 manifests). No freeze tag is
made for A31 at first; the head then had it tagged: `n13-freeze-2a` on `e69f2b2b7`, 2026-11-07T15:47:29Z (§0).
