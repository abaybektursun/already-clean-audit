# M1 assets: the OLMo-3-7B stage-2 mix runs

**Answer.** The study can run with changes. The shared start is released as `stage1-step476838`, which is
2,000,003,530,752 tokens. Both documented 2T mix configs load exactly that step. The three mix ends exist
only as endpoints. The plan as written needs six harvests, about 11 GPU-h on an L40S, not five at about 8
GPU-h. It also needs a loader fix and a different control.

Read-only check, 2026-11-05. No model was run and no weights were downloaded. Sources are tagged [S#] and
listed at the end. Recount scripts are in the scratchpad (listed at the end).

## Changes the plan needs

1. **Loader.** The mix-end configs say `"model_type": "olmo2-retrofit"` and
   `"architectures": ["Olmo2RetrofitForCausalLM"]` [S4]. transformers has no such type: a code search of
   huggingface/transformers for "retrofit" returns 0 hits [S18]. The harvest asserts
   `cfg.model_type == "olmo3"` for O7 (`infra/pg/neurons_harvest.py` lines 49, 137–139) [S17]. So the config
   must be rewritten to `olmo3` / `Olmo3ForCausalLM` before loading. The 355 tensor names are identical to
   the `olmo3` branches. The only other config difference is `sliding_window` 4097 against 4096 [S4]. The
   harvest window is 2,048 tokens, so the sliding window has no effect on its forward pass.
2. **Shards.** Each mix branch holds two weight sets [S2]. The index points to six fp32 shards
   (`model-0000k-of-00006`, `total_size` 29,192,044,544). The branch also keeps three bf16 shards
   (`model-0000k-of-00003`) whose LFS oids equal those of `main` and `stage3-step11921` (`0490d6668e613a29`,
   `e127ea479fb6e208`, `f3ddff10052ffe5d`). Those three are the final OLMo-3 base model, not the mix model.
   `from_pretrained` follows the index, so it loads the right set. But `HF_ALLOW = ["*.safetensors", …]`
   (`infra/pg/matrix_models.py` line 109) downloads all nine shards (43.8 GB). Restrict the download to
   `model-*-of-00006.safetensors`. Add a gate that fails when the loaded weights hash equal to `main`. Its
   registered failing input is loading the three stale shards.
3. **Harvest count.** Start, plus three ends, plus two identity checkpoints, is six harvests, not five. An
   identity pair that includes the start (`stage1-step476000` with `stage1-step476838`, 838 steps apart)
   makes it five.
4. **Control.** The written refutation compares the between-mix difference with an identity pair of
   adjacent main-run stage-2 checkpoints. That pair cannot measure run-to-run variance, for three reasons:
   - It lies on a different trajectory: it starts at 5.93T, not 2T.
   - At the end of the main run the learning rate is near 0: 2.97e-6 at `stage2-step47000` and 0 at
     `stage2-step47684` (lr_check).
   - Its spread will sit near the harvest's own repeatability: on the T_14 re-harvest, 0 of 640 poles
     flipped presence (JOURNAL.md line 878) [S16].

   So the condition has almost no room to fire. No two runs of one mix from one start with different seeds
   are released. The test with room to fail is an ordering written down in advance: the code + math share
   of training tokens is 0% in gen-mc-only, 40% in round5 and 65% in math-code-reasoning-web, so the shift
   should order the same way. Any identity pair serves only as a harvest floor.
5. **Lead contrast.** gen-mc-only against math-code-reasoning-web is the contrast whose settings are all
   public, and they match except for the data (§4). The round5-from-2T run has no public config.
6. **Category definition.** Freeze "code and math documents" as corpus directories `stack_edu-*` (4) and
   `finemath-3plus` (1). No finer per-document label exists (§6).

## 1. Branches, steps, files

The refs list has 1,487 branches: 1,421 `stage1-*`, 52 `stage2-*`, 13 `stage3-*` and `main` [S1]. The 52
stage-2 branches are `stage2-step1000` … `stage2-step47000` every 1,000 steps (47), plus `stage2-step47600`
and `stage2-step47684`, plus three mix branches. Exact mix branch strings [S1]:

- `stage2-step47684-mix-round5-from-2T-ckpt`
- `stage2-step47684-mix-gen-mc-only-from-2T-ckpt`
- `stage2-step47684-mix-math-code-reasoning-web-from-2T-ckpt`

| branch | commit [S1] | uploaded [S3] | weights the index loads [S2, S4] | branch safetensors total [S2] | config [S4] |
|---|---|---|---|---|---|
| `stage1-step476838` (shared start) | `31bf90b3` | 2025-10-01 | 3 bf16 shards, 4,969,984,976 + 4,981,161,496 + 4,644,917,240 = 14,596,063,712 B | 14,596,063,712 B | `Olmo3ForCausalLM`, `olmo3`, window 4096 |
| `…mix-round5-from-2T-ckpt` | `34e13d5f` | 2026-04-15 | 6 fp32 shards, 4,948,333,768 + 4,991,622,816 + 4,924,546,896 + 4,857,404,928 + 4,857,404,928 + 4,612,772,376 = 29,192,085,712 B; index `total_parameters` 7,298,011,136 | 43,788,149,424 B (adds main's 3 bf16 shards) | `Olmo2RetrofitForCausalLM`, `olmo2-retrofit`, window 4097 |
| `…mix-gen-mc-only-from-2T-ckpt` | `45786899` | 2026-04-15 | same sizes, distinct oids | same | same config blob (sha1 `387607fd…`) |
| `…mix-math-code-reasoning-web-from-2T-ckpt` | `ab9f4b07` | 2026-04-15 | same sizes, distinct oids | same | same config blob |
| `stage2-step47684` (main end) | `c3c800dc` | 2025-10-01 | 3 bf16 shards, 14,596,063,712 B | same | `olmo3`, window 4096; config identical to `stage1-step476838` |
| `stage2-step47600`, `stage2-step47000` | `e6304548`, `ea5d6377` | 2025-10-01 | 3 bf16 shards each | same | `olmo3` |

- **Steps.** Every mix run is named step 47,684. That is 100B tokens at 2^21 = 2,097,152 tokens per step
  (100e9 / 2^21 = 47,683.7) [S6, S9].
- **Distinct weights.** Across the 52 stage-2 branches plus the 2T start, `stage1-step1413814`,
  `stage1-step476000`, `stage1-step477000`, `stage3-step11921` and `main`, there are 57 distinct weight sets
  among 58 branches. The only duplicate is `main` = `stage3-step11921` [S2].
- **Tokenizer.** The mix branches' `tokenizer.json` blob (`5fe172127988`) equals that of the stage-1 and
  stage-2 branches [S2]. So the corpus `tok/olmo3/tokens.npy` applies unchanged.
- **Branch history.** The mix branches were cut from `main`; their history carries main's README commits
  [S3]. Their README is the OLMo 2 7B card (blob `39b8d2a67444`).
- **CSV token column.** The OLMo-core checkpoint CSV gives `stage2_midtraining,1000,5929965715456`, which
  is the stage-1 end count [S8]. Its stage-2 token column therefore lags by 1,000 steps. Compute token
  positions as step × 2,097,152.

## 2. The shared start

`stage1-step476838` is a released branch. 476,838 × 4,194,304 = 2,000,003,530,752 tokens (lr_check). It
sits between `stage1-step476000` and `stage1-step477000` [S1]. It is one of seven stage-1 steps that are not
multiples of 1,000: 39500, 189900, 476838, 596047, 597046, 736730 and 1413814.

Load paths:

- **The two skewed-mix configs** [S9], files
  `anneal-proposed-mix-gen-mc-only-100B-olmo25_7b-2T.yaml` and
  `anneal-math-code-reasoning-superswarm-ratios-olmo25_7b-100B-anneal-decon-2T.yaml`, line 23 of each:
  `load_path: gs://ai2-llm/checkpoints/OLMo25/step476838`
- **The main stage-2 run.** `OLMo-3-1025-7B-midtrain.py` line 101 [S6]:
  `load_path="https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step1413814/",`
  The W&B config agrees: `trainer/load_path gs://ai2-llm/checkpoints/OLMo25/step1413814` [S12].
- **`OLMo25` is the stage-1 run.** `OLMo-3-1025-7B-pretrain-2.py` resumes from
  `…/checkpoints/OLMo25/step596047/` [S7]. The CSV lists `stage1/step1413814` as the stage-1 end [S8].
- **Round5 from 2T.** No public config was found. The model card, line 150 [S5], reads: "These experiments
  were run earlier in the Stage 1 pretraining process, and therefore represent training on 100B
  midtraining tokens starting from a Stage 1 checkpoint that had been trained to 2T tokens." The one
  public round-5 config, `anneal-round5-olmo3_7b-anneal-decon-12T.yaml`, loads
  `gs://ai2-llm/checkpoints/OLMo3-7B-swafix/step756906`. Its description is "OLMo3 7b anneal to 100B Tokens
  from 12T ckpt with decon -- round 5 data mix". So it is a different run [S9].
- **Official scripts.** `src/scripts/official/OLMo3/` holds only the main stage-2 script; it contains no
  mix-run script [S6]. The OLMo-core CSV has no row for step 476838 and no mix rows [S8].

**Not verified.** That the HF `stage1-step476838` weights are byte-identical to
`gs://ai2-llm/checkpoints/OLMo25/step476838`. This rests on the run name and the step number.

## 3. Mix compositions

The table gives the share of the 100B training tokens. Group totals were summed from `target_ratio` (mix_groups).

| group | config source names | gen-mc-only (2T yaml) | math-code-reasoning-web (2T yaml) | round5 / final mix |
|---|---|---|---|---|
| code | `code_fim` (Stack-Edu, FIM), `swallowcode` (CraneCode) | 0 | 43.81% | 20.00% |
| math | `megamatt`, `swallowmath` (CraneMath), `dolminomath`, OMR full-thoughts rewrite, tinyMATH MIND/PoT (round5 groups the last four under `dolminos2math`) | 0 | 21.19% | 20.00% |
| thinking | r1, qwq, gemini, llama-nemotron and openthoughts2 traces; math/code meta-reasoning; verifiable (round5: `faeze_reasoning`) | 0 | 15.00% | 7.50% |
| QA | `reddit` (flashcards), `rcqa` (Wiki RCQA), `nemotron-synth-qa` | 22.32% | 0 | 13.90% |
| instruction | `instruction-new-format` (Tulu 3 SFT), `flan` | 7.36% | 0 | 6.10% |
| web, Common Crawl | `web-all_dressed-snazzy2-v18-v20--*` (24 topics; paths `vigintile_0018`, `vigintile_0020`) | 55.56% | 16.97% | 22.50% |
| web, crawl | `sponge` | 4.84% | 0 | 5.00% |
| PDFs | `pdf-quality-*` | 9.92% | 3.03% | 5.00% |
| **code + math** | | **0** | **65.00%** | **40.00%** |

Sources:

- **gen-mc-only.** `anneal-proposed-mix-gen-mc-only-100B-olmo25_7b-2T.yaml`, 49 sources. Description: "OLMo2.5 7b anneal to 100B Tokens on proposed midtraining mix with gen/MC only, from 2T ckpt" [S9].
- **math-code-reasoning-web.** `anneal-math-code-reasoning-superswarm-ratios-olmo25_7b-100B-anneal-decon-2T.yaml`, 58 sources. Description: "OLMo2.5 7b anneal to 100B Tokens on superswarm ratios, math/code/reasoning + web (.65 math/code, .15 reasoming, .2 web) from 2T ckpt" [S9]. The typo is in the source.
- **round5 / final mix.** The main stage-2 W&B config (run `j9nypgrc`, 58 sources) [S12] and the dolma3
  round-5 12T yaml [S9] give the same group totals. Per-source values differ by at most 0.00028 (`megamatt`
  0.01726 against 0.01698). These agree with the paper's Table 5: code 10.0% + 10.0%, Common Crawl HQ 22.5%,
  STEM-Heavy Crawl 5.0%, PDFs 5.0%, total "99.95B (100%)" [S13, §3.5]. They also agree with the model card's
  "20% code, 28% web pages, 19% math, 14% QA, 8% thinking, 6% instruction, and 5% PDFs" [S5].
- **Round5-from-2T itself.** Its own ratios are not public. The model card [S5] and the paper's Table 7
  [S13] call it "Round 5 (final mix)".
- **`sponge`.** Mapped to the paper's STEM-Heavy Crawl by its 5% share; the name itself is not mapped
  anywhere.
- **Paper wording** [S13, §3.5.4, "Performance shows substantial domain tradeoffs"]: "Our Gen-QA mix
  increases proportions of web, QA, and instruction data while omitting math, code, and thinking, and our
  math-code-thinking mix increases proportions of math, code, and thinking data while omitting QA and
  instruction data". The math-code-thinking mix also keeps web data.
- **Behavior-level result, Table 7** [S13], in the order Gen-QA / math-code-thinking / Round 5:
  - Math 27.5 / 60.8 / 57.3
  - Code 11.9 / 35.6 / 31.2
  - MC STEM 66.3 / 62.5 / 66.4

## 4. Settings: mix against mix, and mix against main

| setting | gen-mc-only [S9] | math-code-reasoning-web [S9] | round5 from 2T | main stage-2 [S6, S12] |
|---|---|---|---|---|
| start | `OLMo25/step476838` | `OLMo25/step476838` | 2T per model card; path not public | `OLMo25/step1413814` |
| tokens / steps | 100B / 47,684 | 100B / 47,684 | 47,684 (branch name) | 100B / 47,684 |
| global batch | 2,097,152 | 2,097,152 | not public | 2,097,152 (`2**21`) |
| sequence length | 8,192 | 8,192 | not public | 8,192 |
| start LR | `initial_lr: 0.00020712352850360292` | same | not public | `LR = 0.00020712352850360292` |
| schedule | `scheduler_type: linear`, `warmup_steps: 0`; the cookbook sets `alpha_f=0.0` for anneals [S10, builder.py line 462] | same | not public | `LinearWithWarmup(warmup=0, alpha_f=0.0)` |
| seed | 1337 | 1337 | not public | 1337 |
| weight decay | no public record | no public record | not public | 0.1 |
| trainer / optimizer state | `load_state: false`, which the cookbook passes as `load_trainer_state` [S10, config.py line 203]; optimizer default not traced | same | not public | `load_trainer_state=False`, `load_optim_state=True` |
| model string | `olmo25_7b` | `olmo25_7b` | not public | OLMo-core TransformerConfig |
| nodes | 16 × 8 GPUs | 16 × 8 | not public | 128 H100s (README) |

**LR at the start, relative to the checkpoint's own LR (lr_check)** [S7, S11, S12]:

- **Mixes.** The 2T mixes start at the LR stage 1 had at step 476,838. Cosine over 5T with 2,000 warm-up
  steps gives 2.0712339e-4 against the config's 2.0712353e-4, so there is no jump.
- **Main run.** It starts at the same 2.0712e-4 from a checkpoint whose stage-1 LR was 3.93e-5. The
  half-cosine over 7T gives 3.9325e-5, and W&B run `ej79z38y` logs 3.932527e-5 at step 1,413,814. So the
  main run raises the LR 5.3-fold at step 0.

**Confounds, mix against mix**

1. **Round5 settings unknown.** Round5-from-2T has no public load path, LR, seed, batch or weight decay.
   Any round5 contrast carries that unknown.
2. **Every group changes at once.** Web is 60.4% against 17.0%, QA 22.3% against 0, PDFs 9.9% against
   3.0%. A shift toward code and math documents under math-code-reasoning-web is also a shift away from web
   and QA. The reading is directional only.
3. **Possible document overlap.** Corpus code documents are Stack-Edu originals; math-code-reasoning-web
   trains on a Stack-Edu FIM sample at 22.4%. Corpus math documents are FineMath-3+ originals; the mixes
   train on CraneMath, which rewrites FineMath-4+. All three mixes draw web from `vigintile_0018` and
   `vigintile_0020`. 10 corpus web directories carry the suffix `-0018`; that this suffix is the same
   vigintile index is not verified. A shift can therefore partly be exposure to the very documents. An
   exact-text hash check against the training pools is needed before unblinding. The public
   `allenai/dolma3_dolmino_mix-100B-1025` is the final mix only, not the skewed samples.
4. **Different code samples.** math-code-reasoning-web uses `stack-edu/sample-fim-weighted-pl-edu-score-decon-50BT`.
   Round5 and the main run use `stack-edu/sample-fim-weighted-pl-edu-score-decon` [S9, S12].
5. **No seed variance.** Seeds are equal (1337) where recorded. So no seed variance is available for any
   mix (see change 4).
6. **Weight decay and optimizer state** are not recorded for the 2T runs. The current cookbook
   `builder.py` hard-codes `weight_decay=0.033` (line 477). But the main run was also launched through the
   cookbook (W&B `_CLASS_` `cookbook.model.config.ModelTrainConfig`) and logs 0.1. So that code line is not
   evidence for the 2T runs.

**Confounds, mix against main.** These bear on the main-run identity pair and on any mix-against-main
reading.

1. Start at 2.0T against 5.93T.
2. LR continues at the start for the mixes; the main run jumps 5.3-fold.
3. HF label and storage differ: `olmo2-retrofit`, window 4097, fp32 for the mix ends, against `olmo3`,
   window 4096, bf16 for the start and the main run. The harvest casts every checkpoint to bf16. The window
   size has no effect at a 2,048-token window. Whether training differed at positions beyond 4,096 is not
   established. A third-party loader states that `olmo2-retrofit` is "a pre-release name for the
   architecture that was released as `olmo3`" [S19].

## 5. Intermediate checkpoints of the mix runs

None are released. The refs list has exactly one branch per mix, the step-47,684 endpoint [S1]. The OLMo-core CSV
lists no mix checkpoints [S8]. An HF search of author `allenai` for "gen-mc", "math-code-reasoning" and
"round5" returns no models. Only the main stage-2 run has checkpoints every 1,000 steps.

## 6. The fixed corpus

The corpus manifest is `s3://pg-lineage-769063704696/results/neurons/w1/q4b/1/out/corpus/corpus/manifest.json`
(55,200 B, sha256 `0c99a8e8…`) [S14]. This equals the `corpus_manifest_sha256` of the W3 T_15 harvest
manifest [S15]. So the 20.57M corpus of the W3 grid is bands 1–2 of this corpus.

| field | value [S14 unless noted] |
|---|---|
| source | `dataset_repo` `allenai/dolma3_mix-6T-1025-7B`, `dataset_commit` `2ca900fb…`: OLMo-3-7B's stage-1 mix (SPEC_D1.md §4.2). It is not the stage-2 data (`allenai/dolma3_dolmino_mix-100B-1025` [S5]), so it is a stand-in for the stage-2 distribution. |
| directories | 50 of `n_dirs_total` 165, `dir_rule` "sorted data/* directories at indices floor(i * n_dirs_total / 50), i = 0..49" |
| documents | `n_docs` 34,789; `n_duplicates_skipped` 44,807 |
| 20M corpus | bands 1–2: documents 0–14,106 (7,193 + 6,913); olmo3 tokens 0–20,570,743. The W3 harvest scored 20,560,068 (`n_tokens_scored`; BOS and window position 0 excluded; 10,044 windows of 2,048) [S15]. |
| per-document label | `docs.jsonl.gz` rows `{"i", "shard", "band", "text"}` (DATA_CONTRACT.md §1) [S16]. The manifest's `shards[]` has `dir`, `n_docs_selected` per band and `olmo3_tokens_selected`, so document → directory can be rebuilt in corpus order without opening the 81.7 MB `docs.jsonl.gz`. There is no topic, quality or language field per document. The source is the directory. |

Directory groups in bands 1–2, recounted from the manifest:

| group | directories | documents |
|---|---|---|
| code | `stack_edu-CSharp`, `-JavaScript`, `-Python`, `-SQL` (4) | 2,154 |
| math | `finemath-3plus` (1) | 242 |
| PDFs | `olmocr_science_pdfs-*` (7) | 446 |
| web, `common_crawl-science_math_and_technology-*` | 3 | 758 |
| web, `common_crawl-software*` | 5 | 1,612 |
| web, other `common_crawl-*` | 30 | 8,894 |

- **Code and math are identifiable only by directory.** The `common_crawl` topic directories for science,
  math and software are web pages that may hold math or code. The frozen definition must say whether they
  count; change 6 says they do not.
- **Token share.** The band rule gives each directory about 400k tokens in bands 1–2 (`band_rule`; each
  band's budget is exceeded by at most one document). So code + math is about 2.0M of 20.57M tokens, about
  10%. The exact count needs `tok/olmo3/doc_offsets.npy`, which was not fetched.

## 7. Cost

Measured harvests on the 20.57M corpus (O7, bf16, window 2,048):

| run | device | download | harvest loop | wall | file |
|---|---|---|---|---|---|
| T_15 `stage1-step1413814` | L40S | 14.61 GB in 62.3 s | 6,166 s at 3,334 tok/s | 6,476 s | [S15] |
| T_0, T_6 | L40S | 14.61 GB in 57 s and 58 s | — | 6,601 s, 6,738 s | [S15] |
| T_7, T_14 | RTX PRO 5000 Blackwell | — | 4,551–4,620 tok/s | 4,954 s, 4,772 s | [S15] |
| T_14 re-harvest | L40S | 59 s | 3,327 tok/s | 6,772 s; 1.893 GPU-h billed | JOURNAL.md line 878 [S16] |
| W3, 16 checkpoints | mixed | — | — | 26.9 GPU-h billed = 1.68 per checkpoint | JOURNAL.md line 660 [S16] |

**Per mix end.** A mix end downloads 29.2 GB with the shard pattern fixed (43.8 GB without). At 234 MB/s
that adds 1–2 minutes over the 62 s bf16 download. The forward is unchanged after the bf16 cast.

| plan | harvests | GPU-h |
|---|---|---|
| as written (start, 3 ends, 2 adjacent main-run checkpoints) | 6 | 10.1 (W3 average) to 11.4 (1.89 each, L40S billed) |
| identity pair `stage1-step476000` / `stage1-step476838` (shares the start) | 5 | 8.4 to 9.5 |

- **Free contrast.** T_15 (`stage1-step1413814`) is already harvested on this corpus. If
  `stage2-step47684` is one of the identity checkpoints, the main run's own stage-2 shift (round 5 from
  5.93T) comes at no extra harvest.
- **Comparison with 06_CRITIC.md.** The critic's "five harvests, about 8 GPU-h" is one harvest short for
  the plan as written.

## 8. Prior work

- **The search.** One extended web search: "OLMo 3 midtraining data mix neurons features interpretability
  "mix-round5" OR "gen-mc-only" OR "math-code-reasoning-web" checkpoints comparison". It found no
  neuron-level or feature-level comparison of these mixes. The closest hits are the OLMo 3 report itself
  (Table 7, benchmark level) [S13] and a LessWrong post on eval-awareness across OLMo 3 training
  (behavior level).
- **Nearest prior work.** Per `04_RECENT.md` §2: https://arxiv.org/abs/2610.06851, which retrains
  OLMo-3-7B mid-training on edited data and reads behavior.
- **GitHub code search for the branch names.** These repositories use the mix branches, none at neuron
  level:
  - **parsa-ra/lmbehave.** arXiv 2610.24029, which reads layerwise residual-stream signatures. Its
    `experiments/olmo3_longitudinal_v1.json` names `mix-2t-math-code-reasoning-web`. Only the config was
    read, not the paper.
  - **JirkoRu/ValueEmergence**, `docs/WP0_RESULTS.md` lines 462–469. This reads value profiles at behavior
    level. Its claim "Their shared 2 T ancestor is **not released**" is contradicted by `stage1-step476838`
    [S1].
  - **mariagrandury/snr-multilingual**, `src/evals/CLAUDE.md` §8. It records the `olmo2-retrofit` load
    failure in vLLM and transformers 5.1.0.
  - **Branch inventories only:** johannn05/mech-interp (a neuron-lineage study of post-training checkpoints
    that only lists the mix branches), deepsteer/deepsteer, quadrismegistus/malign-logits and
    anima-research/truebasebench.
- **Novelty.** No study was found that compares neurons or features across these three mixes.

## Not checked

- Byte identity of HF `stage1-step476838` with `OLMo25/step476838` (§2).
- The round5-from-2T settings (§4).
- The 2T runs' weight decay and optimizer-state loading (§4).
- Overlap between corpus documents and the training pools (confound 3).
- Whether the `common_crawl` directory suffix is the quality vigintile.
- The exact code and math token count of the 20.57M corpus (`doc_offsets.npy` not fetched).
- The full text of arXiv 2610.24029.

## Sources

- [S1] https://huggingface.co/api/models/allenai/Olmo-3-1025-7B/refs, read 2026-11-05.
- [S2] https://huggingface.co/api/models/allenai/Olmo-3-1025-7B/tree/<branch>, for each branch named.
- [S3] https://huggingface.co/api/models/allenai/Olmo-3-1025-7B/commits/<branch>.
- [S4] https://huggingface.co/allenai/Olmo-3-1025-7B/resolve/<branch>/config.json and
  `model.safetensors.index.json`.
- [S5] Model card https://huggingface.co/allenai/Olmo-3-1025-7B/blob/main/README.md, lines 146–150.
- [S6] https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py
  (repo HEAD `ccf4ef4b`, 2026-10-23; file last changed `66f768b2`, 2026-08-11), lines 41–44, 57, 74, 80,
  101–104.
- [S7] Same directory: `OLMo-3-1025-7B-pretrain-1.py` (LR 3e-4, `CosWithWarmup(warmup_steps=2000)`,
  5T) and `OLMo-3-1025-7B-pretrain-2.py` (`HalfCosWithWarmup`, 7T, `load_path …/OLMo25/step596047/`).
- [S8] https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv.
- [S9] https://github.com/allenai/dolma3/tree/main/datasets/configs/midtraining (HEAD `1a9daced`,
  2026-04-20; the configs were added in `59ee1ba7`, 2026-04-14, "add round 1-5 and skewed mix midtraining
  configs").
- [S10] https://github.com/allenai/olmo-cookbook/blob/main/src/cookbook/model/builder.py lines 461–477
  (last changed 2025-09-18) and `src/cookbook/utils/config.py` line 203.
- [S11] https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/optim/scheduler.py
  (`LinearWithWarmup`, `CosWithWarmup`, `HalfCosWithWarmup`).
- [S12] W&B `ai2-llm/Olmo-3-1025-7B`: run `j9nypgrc` config through https://api.wandb.ai/graphql; run
  `ej79z38y` last logged `optim/LR (group 0)`, from the critic's banked
  `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/crit06/wandb_runs.json`.
- [S13] OLMo 3 report, https://arxiv.org/html/2512.13961 (v2): §3.5 Table 5; §3.5.1 "Integration tests";
  §3.5.4 Tables 6–7.
- [S14] `s3://pg-lineage-769063704696/results/neurons/w1/q4b/1/out/corpus/corpus/manifest.json`.
- [S15] `s3://pg-lineage-769063704696/results/neurons/w3/grid/1/out/{t15_step1413814,t00_step0,t06_step12000}/manifest.json`
  and `…/grid/1b/out/{t07_step20000,t14_step1000000}/manifest.json`.
- [S16] `studies/already-clean/JOURNAL.md` lines 535, 660 and 878; `studies/already-clean/DATA_CONTRACT.md`
  §1, §2, §10; `studies/already-clean/SPEC_D1.md` §4.2.
- [S17] `infra/aws_pg/jobs/neurons_harvest.py` line 48; `infra/pg/matrix_models.py` line 109;
  `infra/pg/neurons_harvest.py` lines 49 and 137–142.
- [S18] https://github.com/huggingface/transformers: code search for "retrofit" returns 0 hits;
  `models/olmo3/configuration_olmo3.py` has `model_type = "olmo3"` and `sliding_window … = 4096`.
- [S19] `aryamanarora/matryoshka-attribution-parameters/scripts/olmo3_post/olmo3_rlzero_patch.py` docstring.

Recount scripts and saved reads are in
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/m1/`:

- `lr_check.py` and `lr_check_out.txt`
- `mix_groups.py` and `mix_groups_out.txt`
- `refs.json`, `tree_*.json`, `commits_*.json`, `idx_*.json`, `cfg_*.json`
- `corpus_manifest.json`, `t15_manifest.json`, `wandb_main_s2.json`
- `dolma3cfg/`, `olmocore/`, `paper/olmo3.txt`
