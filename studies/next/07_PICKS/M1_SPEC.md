# M1: the three stage-2 data mixes in the neuron basis. Pre-registration draft

**Answer.** M1 is ready for its first GPU step, the known number K0: about 0.26 GPU-h on one L40S. The full
run is five harvests of all 704,512 poles on the fixed 20.56M-token corpus, plus K0 and the banked
references. It is projected at 9.73 GPU-h at the measured L40S rate and 10.15 at the lowest banked L40S rate,
against a ceiling of 12 GPU-h (`M1_OFFLINE_GATE.md` §PROJECT). The test is an ordering written before the
data. Per layer band, the band-mean code+math share of the poles' top-50 positions must order
gen-mc-only < round5 < math-code-reasoning-web. That follows the code+math share of their training tokens:
0 %, 40 % and 65 % (`M1_ASSETS.md` §3). The lead contrast is math-code-reasoning-web against gen-mc-only. It
must exceed a bar set before any mix is read. The bar is the larger of two numbers: the identity pair's spread
(the floor) and the largest band-mean change between neighbouring banked stage-1 checkpoints (the drift
reference). On the 640 banked grid poles, both the pass branch and the fail branch fire on real checkpoint
pairs (§3.4).

Written 2026-11-05 by an Opus worker in worktree `interp_research-m1`, branch `m1`. No model was run. No
weights were downloaded. The only network reads were small text files from Hugging Face, listed in
`M1_OFFLINE_GATE.md` §Fetched.

---

## 0. Status and provenance

| field | value |
|---|---|
| status | **DRAFT, not frozen.** Freeze 1 and freeze 2 tags, SHAs and UTC are recorded here when made. |
| freeze 1 | `m1-freeze-1`: this spec, `infra/pg/m1.py`, `infra/aws_pg/jobs/m1_harvest.py`, `infra/aws_pg/jobs/m1_dryrun.py`, the three step files. The offline gate must print `M1_DRYRUN_OK` first. **Made:** tag `m1-freeze-1` (a lightweight tag) on commit `8f34b6629` (`8f34b66297a6ba4b351f79961ed3782f984a6fac`), committed 2026-11-05T16:11:06Z. |
| freeze 2 | `m1-freeze-2`: `studies/next/m1/thresholds.json`. It is written by `python -m infra.pg.m1 freeze` from the K0 launch's banked references, after K0 passes and before any mix harvest launches. **Made:** `M1_FREEZE_OK`, run 2026-11-05T16:43:13Z on the six references of `k0-01`; `thresholds.json` sha256 `4bb56ded950396cd…`; τ = 0.0082, 0.0066, 0.0023, 0.0035 and κ = 0.183, 0.177, 0.192, 0.202 by band; tag `m1-freeze-2` (a lightweight tag) on the commit that adds the file and this row (`M1_OFFLINE_GATE.md` §Freeze 2). |
| parents | `studies/next/06_CRITIC.md` Part 5 M1 (lines 818–850); `studies/next/07_PICKS/M1_ASSETS.md` (all); `studies/next/05_SCORING.md` lines 110–136 (picks v2, item 2); `studies/next/00_ASSETS.md` §1, §4; the head's plan review relayed by the coordinator on 2026-11-05 (§0.1) |
| method | repo `CLAUDE.md` ("Before any compute", "Instruments and controls", "Provenance"); the format of `07_PICKS/N13_SPEC.md` |
| vocabulary | `studies/next/README.md` house rules. A run whose gate fails is labelled **DOES NOT COUNT**. Chance level for a shift is the identity pair's value. |

### 0.1 Design decisions from the head's plan review, and where each sits

| decision (coordinator, 2026-11-05) | where |
|---|---|
| Five harvests: the shared start, the three mix ends, and one identity pair (the same weights harvested twice on the same instance type) as the floor of every shift measure | §2.1, §3.1; the identity pair is the start harvested on two instances of one type |
| Ceiling 12 GPU-h; a park order after the identity pair and the shared start | §9.7 |
| One GPU model for all harvests; AWS L40S by default, a Vast RTX PRO 5000 class card if the user tops up Vast | §9.6; the K0 point follows the lane (§6) |
| The refutation must have room, shown on control data before the freeze; write the does-not-count branch | §3.4, §5 |
| The known number: the cheapest re-harvest of one banked grid point through the full code path, at a stated tolerance, against a banked value whose path is verified | §6; every compared file exists on the laptop with its sha256 registered (`M1_OFFLINE_GATE.md` §ROOM, K0 lines) |
| On Vast, the first GPU step doubles as the lane's reliability check | §9.6, §9.8 |

### 0.2 Facts read for this draft

The Hugging Face API reads were made on 2026-11-05. They are saved under
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/m1_gate/hf/`
(called `HF/` below). The same values are registered in `infra/pg/m1.py` (`BRANCHES`, `STALE`, `REF_CONFIG`).
The offline gate checks the registry against these reads (section REAL).

| fact | value | source |
|---|---|---|
| repo | `allenai/Olmo-3-1025-7B` | |
| shared start | `stage1-step476838`, commit `31bf90b3211621ce8154e55e7fec5b5eda08e620`; 3 bf16 shards, sha256 `6005c96a…`, `da953fe9…`, `c5696461…`; 476,838 × 4,194,304 = 2,000,003,530,752 tokens | `HF/refs.json`, `HF/tree_stage1-step476838.json`; `M1_ASSETS.md` §2 |
| G (0 % code+math) | `stage2-step47684-mix-gen-mc-only-from-2T-ckpt`, commit `45786899c6c427d17f9291f3d97c7e4fcbd4962b`; 6 fp32 shards `model-0000k-of-00006`, sha256 `e8e8abbf…`, `453c2ae4…`, `982c106e…`, `3ec0366c…`, `e50e3dfb…`, `81c6f17d…` | `HF/tree_…gen-mc-only….json`, `HF/index_…json` |
| R (40 %) | `stage2-step47684-mix-round5-from-2T-ckpt`, commit `34e13d5fae4d1aacf4aa019c377a332ed09ff136`; 6 fp32 shards `8e8c22d2…`, `89b4cbf3…`, `8d2219ce…`, `a0e4d0ef…`, `f081c891…`, `55509870…` | same |
| M (65 %) | `stage2-step47684-mix-math-code-reasoning-web-from-2T-ckpt`, commit `ab9f4b070e9125de257bfbe31bf8cc777daadfb4`; 6 fp32 shards `7c4231e0…`, `fe69ddc3…`, `717bb447…`, `dd44c4be…`, `9bbe71d5…`, `ebc7c8f5…` | same |
| stale shards in every mix branch | three bf16 `model-0000k-of-00003` shards, sha256 `0490d666…`, `e127ea47…`, `f3ddff10…`, equal to `main` (commit `a81bae42…`) | `HF/tree_main.json`; offline gate REAL |
| shard dtypes | `model-00001-of-00006` of G and `model-00006-of-00006` of M hold only `F32` tensors; `model-00001-of-00003` of G and of the start hold only `BF16` | safetensors headers read by HTTP range (header bytes only), `HF/headers/` |
| mix index | 355 tensors over the six fp32 shards; `total_parameters` 7,298,011,136; `total_size` 29,192,044,544 | `HF/index_…json` |
| mix config | `model_type` `olmo2-retrofit`, `architectures` `["Olmo2RetrofitForCausalLM"]`, `sliding_window` 4097, `torch_dtype` `bfloat16`; every other key equals the start's `config.json` | `HF/config_*.json`; offline gate REAL |
| transformers 5.9.0 on the as-released mix config | `AutoConfig.from_pretrained` raises `ValueError` ("does not recognize this architecture"); on the rewritten config it returns `Olmo3Config`, equal to the start's except `sliding_window` | offline gate REAL |
| model from the rewritten config, on the meta device | exactly the 355 tensor names of the mix index; 7,298,011,136 parameters | offline gate REAL |
| tokenizer | the mix branches' `tokenizer.json` is git blob `5fe172127988`, as on the stage-1 branches; its vocabulary sha256 `2f3c015e…` equals the corpus manifest's `vocab_sha256` | `HF/tok/`; offline gate REAL |
| K0 point, AWS lane | `stage1-step1413814` (T_15), commit `373bad25002f1624757a73235c5ca844c6375c25`, harvested on an L40S (`00_ASSETS.md` line 46) | `HF/tree_stage1-step1413814.json` |
| K0 point, Vast lane | `stage1-step1000000` (T_14), commit `cb286dcdaaf3f92b3c419dd77439078fdebffe53`, harvested on an RTX PRO 5000 Blackwell (`00_ASSETS.md` line 45) | `HF/tree_stage1-step1000000.json` |
| the banked grid settings | `--max-tokens 20570743 --snapshot-bands 1 --poles <640 grid poles>`; T_15 with its own top-50 as positions, T_1…T_14 with `--positions` = T_15's `poles/positions.npz`; window 2,048, batch 8, 10,044 windows, 20,560,068 scored tokens; sdpa, bf16, transformers 5.9.0, torch 2.12.0+cu130 | `studies/already-clean/runbook/steps/steps_grid_{a,b}_mdl.json`; the T_15 manifest (copy at `scratchpad/m1/t15_manifest.json`) |
| harvest timing, T_15 on an L40S | download 14.61 GB in 62.3 s (234.3 MB/s); harvest loop 6,166 s at 3,334 tokens/s; band-1 snapshot write 155 s; final write 145 s; wall 6,476 s | T_15 manifest `timing_s`, `download` |
| tokens per second | L40S 3,215–3,334 (T_0–T_6, T_15); RTX PRO 5000 Blackwell 4,551–4,623 (T_7–T_14) | `00_ASSETS.md` lines 29–46 (this branch's copy; the coordinator cited lines 22–39) |
| prices | AWS g6e.2xlarge $2.24 per hour, quoted in an operations note (`00_ASSETS.md` lines 222–223); Vast realized $22.68 for 28.9 billed GPU-h over the W3 grid = $0.785 per GPU-h (`studies/already-clean/SPEC_D1.md` lines 707–708) | |
| corpus | manifest sha256 `0c99a8e8…` (equals the T_15 harvest's `corpus_manifest_sha256`); 50 directories: code `stack_edu-{CSharp,JavaScript,Python,SQL}`, math `finemath-3plus`, 7 `olmocr_science_pdfs-*`, 38 `common_crawl-*` | `scratchpad/m1/corpus_manifest.json`; offline gate REAL |

### 0.3 Facts to verify at run time

None needs a human mid-run. Each is read by a named command or gate.

| # | fact | unverified because | verified by | rule if it differs |
|---|---|---|---|---|
| 1 | The S3 corpus `s3://pg-lineage-769063704696/results/mdl/neurons/w1/q4b/1/out/corpus/corpus/` holds `manifest.json` and `tok/olmo3/{tokens.npy,doc_offsets.npy}` | no AWS call was allowed | §9.5 step S1.3 (`aws s3 ls`); in-job G-CORPUS (manifest sha256 `0c99a8e8…`, tokens sha256 `2aca80ca…`) | stop before launch; find the copy whose sha256 matches |
| 2 | The six reference harvests exist on S3 with `topk/`, `manifest.json`, `controls.json`: `results/neurons/w3/grid/1/out/{t00_step0,t15_step1413814}`, `…/1b/out/{t12_step320000,t13_step560000,t14_step1000000}`, `…/1c/out/t14_step1000000` | as 1; the `1c` path is quoted from `studies/already-clean/JOURNAL.md` 2026-10-21T19:31:14Z | S1.3; in-job: each reference manifest's `tokens_sha256` and `corpus_manifest_sha256` must equal the registered values | stop before launch; correct the path in `next_m1_k0-01.json` (a dated amendment, before freeze 2) |
| 3 | The AMI (`pg-env-v1`) has torch `2.12.0+cu130` and transformers `5.9.0`. The pilot-0 run had them on 2026-11-05 (`pilot0/GPU_CHECK.md`) | not re-read | G-SAME at job start, before any download | the job stops in seconds; no harvest runs on other versions |
| 4 | `accelerate` is installed on the AMI (`device_map` needs it) | not read | the pilot-0 load used `device_map="cuda"` on this AMI (`GPU_CHECK.md`, changes list) | a load error stops K0 |
| 5 | The cross-device numbers behind the K0 tolerance (median 0.43 %, p99 2.22 %) | the artifact `results/neurons/w3/device_check/t14_1b_vs_1c.json` is on S3 only; the numbers are quoted from the journal entry | not needed for the gate; the tolerance stays as registered | none |
| 6 | The mix ends were cast to bf16 the way AI2 cast its own bf16 releases (round to nearest even) | AI2's conversion script was not read | not resolvable here | none: all three ends use one cast, so it cannot make a between-mix difference (§8 threat 4) |
| 7 | The code+math token share of the 20.57M corpus (about 10 %, `M1_ASSETS.md` §6) | needs `doc_offsets.npy` | every harvest records `token_share_per_dir` in `m1/metrics.json` | reported |
| 8 | On Vast, a GPU query that rents only RTX PRO 5000 Blackwell cards | the query syntax was not tested | `python -m infra.vast.controller plan <manifest>` lists offers | if no query pins the model, use the AWS lane |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives this run |
|---|---|---|---|
| OLMo-3-7B all-neuron harvests | 16 stage-1 points, the same corpus and code; 640 grid poles on the laptop | `BANK/neurons/w3/grid/{1,1b}/out/` (BANK = `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid`); S3 `results/neurons/w3/grid/` | K0's bank; the references; the room check (§3.4) |
| harvest repeatability | T_14 re-harvested on an L40S against its Blackwell harvest: 0 of 640 poles flip presence; exemplar relative difference median 0.43 %, p99 2.22 % | `studies/already-clean/JOURNAL.md` 2026-10-21T19:31:14Z; S3 `results/neurons/w3/device_check/t14_1b_vs_1c.json` (not read) | the K0 tolerance; the banked identity reference `t14b`/`t14c` |
| stage-2 mixes, any level | none in this repo; behavior level in the OLMo 3 report, Table 7: Math 27.5 / 60.8 / 57.3 and Code 11.9 / 35.6 / 31.2 for Gen-QA / math-code-thinking / Round 5 | `M1_ASSETS.md` §3, §8 | the positive control's expected direction |
| OLMo-3 load and loss on AWS | pilot-0 loss check, `stage1-step60000`, sdpa 1.26 s per 8,192-token row | `studies/next/pilot0/GPU_CHECK.md` | the in-job download pattern, the AMI versions |
| literature | one search, no neuron-level comparison of these mixes found | `M1_ASSETS.md` §8 | novelty claim, as far as that search goes |

### 0.5 Objects, defined once

- **Pole** p = (layer l, neuron n, sign s), as in the house vocabulary. 32 × 11,008 × 2 = 704,512 poles.
- **Top-50 positions** of p at checkpoint c: `topk/layer_LL.npz` → `tok[n, s, :]` written by
  `infra/pg/neurons_harvest.py`. They are the global token indices of the 50 corpus positions with the largest
  a (+ pole) or −a (− pole), over the 20,560,068 scored tokens. a = act(g)·u, recomputed in fp32. Window
  position 0 and BOS are excluded.
- **Directory of a position**: its document's corpus directory. Documents map to directories by the manifest
  band rule (`m1.doc_dirs`). Tokens map to documents by `doc_offsets.npy`.
- **counts[l, n, s, d]**: the number of p's filled top-50 slots in directory d (`m1/counts.npz`).
- **cm_share(p, c)** = (slots in the 5 code and math directories) / (filled slots). Its unit is a fraction of
  50 slots.
- **Layer band** b: layers 0–7, 8–15, 16–23, 24–31.
- **C_c(b)**: the mean of cm_share over the 176,128 poles of band b at checkpoint c.
- **Contrast** Δ_XY(b) = mean over poles in b of [cm_share(p, X) − cm_share(p, Y)], paired by pole. The
  interval is a 95 % bootstrap over neurons: the two poles of a neuron are resampled together, 2,000
  resamples, numpy `default_rng(0)`.
- Checkpoint keys: **start_a**, **start_b** (the start, harvested twice), **G**, **R**, **M**, and the
  references **t00, t12, t13, t14b, t14c, t15**.

---

## 1. Question, decision, hypotheses

### 1.1 The question

AI2 trained three 100B-token stage-2 runs from one stage-1 checkpoint. The runs differ in data: 0 %, 40 % and
65 % code+math. Does the data mix decide which documents a neuron fires most on? Per neuron, the measure is
the share of its top-50 positions on the fixed corpus that lie in code and math documents.

### 1.2 The decision this run makes

| outcome | next |
|---|---|
| H_lead SUPPORTED and H_order SUPPORTED | A held-out prediction landed at 7B: the training mix sets the neurons' document preference in proportion to its code+math share. Next: the picture (§10), then per-pole analysis of which neurons move (exploratory). |
| H_lead SUPPORTED, H_order REFUTED or UNDECIDED | The two documented mixes separate. Round5 does not sit between them; its settings are not public (§8 threat 2). Report as such. |
| H_lead REFUTED | 100B tokens of a skewed mix move the band-mean code+math share by less than ordinary stage-1 drift, or in the wrong direction. This is a result: the neuron basis is set by stage 1. |
| DOES NOT COUNT | No claim. The failed gate and its value are reported. |

### 1.3 Hypotheses and the observation that refutes each

All quantities are band means of cm_share, in fractions of top-50 slots (§0.5). bar(b) and κ(b) are frozen
before any mix is read (§3).

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_lead** (primary) | The math-code mix moves neurons toward code and math documents more than the general mix does. | In every band b: the lower 95 % bound of Δ_MG(b) exceeds bar(b). | In any band b, the upper 95 % bound of Δ_MG(b) is below bar(b): the contrast is smaller than drift, or reversed. |
| **H_order** (the full ordering) | The shifts order with the code+math share of training: G < R < M. | H_lead in every band, and in every band the lower bounds of Δ_RG(b) and Δ_MR(b) exceed bar(b). | H_lead refuted; or in any band the upper bound of Δ_RG(b) or of Δ_MR(b) is below −bar(b) (R outside the G–M interval by more than the bar). |
| **H_specific** (secondary) | The contrast is specific to code and math, not a reshuffle over all directories. | In every band, Δ_MG for the 5 code+math directories exceeds the 99th percentile of \|Δ_MG\| over 1,000 random sets of 5 other directories. | In any band, the code+math Δ_MG is at or below the median of the random sets' \|Δ_MG\|. |

Labels per band are SUPPORTED, REFUTED, UNDECIDED (interval straddles the line) and ABOVE_CEILING (Δ_MG(b) >
κ(b)). The band labels combine as follows. All SUPPORTED gives SUPPORTED. Any ABOVE_CEILING gives ABOVE_CEILING.
Otherwise, any REFUTED gives REFUTED, and anything else gives UNDECIDED (`m1.combine`).

### 1.4 The author's prior, logged before any data (2026-11-05)

These are not hypotheses. They are written down so the result can be compared with them.

| quantity | prior |
|---|---|
| Δ_MG(b), every band | +0.01 to +0.06 (C at the nearest banked point, T_13, is 0.170, 0.135, 0.057, 0.053 by band on the 640 poles, `M1_OFFLINE_GATE.md` §ROOM) |
| H_lead SUPPORTED | probability 0.65 |
| H_order SUPPORTED | 0.4; R's position λ = Δ_RG / Δ_MG near 0.6 if the shift is linear in the code+math fraction (40 / 65 = 0.62) |
| G-ID: start_a and start_b counts identical | 0.8 (same software, same GPU model, deterministic kernels) |
| PC_NLL fires | 0.95 |
| H_specific SUPPORTED | 0.6 |

---

## 2. The measurement and the instrument

### 2.1 The measurement, in plain words

- **Quantity.** cm_share(p, c) for every pole p (§0.5). It is computed from the harvest's own `topk/` arrays by
  `m1_harvest.counts_and_quantiles`, then `m1.share`.
- **Items.** All 704,512 poles. The 640 grid poles also carry exemplar values, as in the banked grid.
- **Before-model and after-models.** Before: `allenai/Olmo-3-1025-7B` @ `stage1-step476838` (commit `31bf90b3…`),
  harvested twice (start_a, start_b). After: G, R, M at the commits in §0.2.
- **Pairing.** By pole identity (l, n, s). All three ends continue from the same start, so coordinate n is
  the same neuron in all of them.
- **Summary.** The band mean of the per-pole difference, Δ_XY(b) (§0.5).
- **Interval.** 95 % bootstrap over neurons, 2,000 resamples, numpy `default_rng(0)` (`m1.band_diff`).
- **Secondary, exploratory, no rule.** The shift of each mix from the start, Δ_{G,start}, Δ_{R,start} and
  Δ_{M,start}. The activation quantiles `q99`, `q999`, `q9999`, `max_abs` and `density` of
  `stats/layer_LL.npz` are banked in `m1/quantiles.npz`. They are compared only as within-checkpoint ranks
  among a layer's poles, because activation scale changes across checkpoints (`06_CRITIC.md` Finding 0.5).

### 2.2 Harvest settings: identical to the banked grid

`m1_harvest` calls `neurons_harvest.load_model` and `neurons_harvest.run` unchanged, with the grid's arguments.
G-SAME compares the harvest's `manifest.json` with the registered T_15 values (`m1.SAME`) and stops on any
difference.

| setting | banked grid (T_15 manifest) | M1 |
|---|---|---|
| corpus | manifest `0c99a8e8…`, tokens `2aca80ca…`, olmo3 | same, asserted (G-CORPUS) |
| tokens | `max_tokens` 20,570,743 → 10,044 windows of 2,048 → 20,560,068 scored | same |
| poles | the 640 grid poles | the same list, read from the banked T_15 `samples/contexts.jsonl` (sha256 `a4d46cd8…`) |
| positions | T_15: its own top-50; T_1–T_14: T_15's `positions.npz` | T_15's positions, rebuilt from the same contexts file (32,000 of 32,000 filled) |
| top-K, histogram, snapshot | K 50, top-1,000 for the grid poles, 400 bins, snapshot band 1 | same (module constants) |
| batch, window, attention, dtype | 8, 2,048, sdpa, bf16 | same |
| versions | transformers 5.9.0, torch 2.12.0+cu130 | asserted at job start |
| code | `infra/pg/neurons_harvest.py` at `17057bd0` (`pg_git_sha` of the T_15 manifest) | the same file; `git diff 17057bd0 HEAD` changes only a parser refactor, an added `O2` key and a `qwen3_5` branch of `text_tower`, none on the O7 path |
| GPU | T_15: L40S | one GPU model for every harvest of a lane (§9.6) |

### 2.3 The two loader fixes (smallest change that works)

1. **Architecture name.** `m1_harvest.load_dir` writes a new directory. Its `config.json` is the mix config with
   `model_type` → `olmo3` and `architectures` → `["Olmo3ForCausalLM"]`. Its other files are symlinks into the
   HF cache, so the cached blob is never written. `sliding_window` 4097 is kept: on a 2,049-position forward
   it gives logits identical to 4096 (offline gate WIN, a tiny Olmo 3 on CPU; the control window 1,000 does
   change them). G-CFG requires the result to equal the registered start config except `sliding_window`. The
   as-released config is its failing input.
2. **fp32 shards.** Only the registered shard files are downloaded (`allow_patterns` = the non-weight files
   plus exactly the index's six `model-0000k-of-00006.safetensors`). The three stale bf16 shards are never
   fetched: 29.19 GB instead of 43.79 GB. The cast is `neurons_harvest.load_model`'s existing
   `from_pretrained(…, dtype=torch.bfloat16)`. This is enough, because transformers casts each fp32 tensor
   to bf16 on load, and the bf16 model is the same 14.6 GB on the GPU as a bf16 checkpoint. A separate
   cast-and-write step would add 14.6 GB of disk writes and change nothing. G-CAST verifies it. Seven
   registered tensors (MLP first) are read back from the shard files with `safetensors`, and the loaded bf16
   tensor must equal `file_tensor.to(torch.bfloat16)` bit for bit. torch's cast rounds to nearest even.
   Every parameter must be bf16. On the tiny model, a truncating cast differs in 48 % to 51 % of entries, so
   the gate can tell the two casts apart (`M1_OFFLINE_GATE.md` §JOB).

---

## 3. Controls and references

Rule 4 of "Instruments and controls" asks for three numbers before any readout is unblinded. They are fit on
both sides (G-FIT on every harvest), a positive control that fired (PC_NLL) and the value on an identity pair
(G-ID).

### 3.1 The identity pair: the floor

start_a and start_b are the same weights (`stage1-step476838`), harvested on two instances of one instance
type. I(b) = C_{start_b}(b) − C_{start_a}(b). The share of poles whose counts differ in any directory is also
reported. The floor enters every bar: bar(b) ≥ |I(b)|. The prior expectation is I(b) = 0 exactly
(deterministic kernels, the same software and GPU model). The banked cross-device repeat of T_14 changed no
pole's presence (`JOURNAL.md` 2026-10-21T19:31:14Z).

### 3.2 The drift reference τ

The identity floor alone gives the refutation almost no room (`M1_ASSETS.md` lines 31–42). A between-mix
difference of 0.001 would beat a floor of 0. So the bar also contains τ(b). It is the largest band-mean
change of C between neighbouring banked stage-1 checkpoints that share a GPU type. These points were trained
on one unchanged mix:

τ(b) = max(|C_t13 − C_t12|, |C_t14b − C_t13|, |C_t15 − C_t14c|)(b), on all poles (`m1.freeze`, `REF_PAIRS`).

The pairs span 1.0T, 1.85T and 1.74T tokens of stage-1 training (`00_ASSETS.md` §1.1). A 100B-token mix
contrast below τ cannot be told apart from what ordinary training does in either direction.

**The bar.** bar(b) = max(τ(b), ε_dev(b), |I(b)|). Here ε_dev(b) = |C_t14c(b) − C_t14b(b)| is the banked
cross-device identity spread.

### 3.3 The ceiling κ

κ(b) = the largest |C_i(b) − C_j(b)| over every pair of the six references, including random init (t00). A
lead contrast above κ(b) is larger than any change this model's stage-1 run showed, from initialization on.
It is labelled ABOVE_CEILING. The verdict then reads "DOES NOT COUNT until inspected". The executor reads 20
grid poles' top-50 contexts at G and M from `samples/contexts.jsonl`, checks the directory mapping against
the decoded text, and writes the finding as a dated note. The frozen rule is not changed.

### 3.4 Room to fire, shown on control data before the freeze

On the 640 banked grid poles over the 16 banked points (`M1_OFFLINE_GATE.md` §ROOM, run 2026-11-05):

- C by band at T_13 (2.35T tokens, the banked point nearest the 2.0T start): 0.1704, 0.1353, 0.0566, 0.0530.
- τ_640 = 0.0134, 0.0025, 0.0035, 0.0098 and κ_640 = 0.195, 0.162, 0.2274, 0.2055.
- The frozen labelling function `m1.label_lead` ran on each of the 15 neighbouring pairs in both
  orientations, as if the pair were (G, M): 120 band labels. **SUPPORTED fired 9 times** (for example T_01 as G,
  T_00 as M: bands 0–2). **REFUTED fired 29 times** (for example T_14 as G, T_15 as M: band 3). The rest were
  UNDECIDED. So both branches are reachable on real checkpoints.
- With 80 neurons per band, the interval is about ±0.013. The full run has 88,064 neurons per band, so the
  interval shrinks by about 33-fold, to about ±0.0004 (sd of a neuron's mean shift about 0.06, from the same
  data). Far fewer readings will be UNDECIDED.
- G-ID on banked data: T_15 against itself gives I = 0 and passes. Its failing input, T_13 in place of the
  twin, gives I = −0.0171, −0.0036, −0.0034, +0.0086 and fails against τ_640,min / 4 = 0.0006.

At freeze 2, `m1.freeze` repeats this check on all poles of the six references. The CLI refuses to write
`thresholds.json` unless SUPPORTED and REFUTED are both reachable and τ(b) < κ(b) in every band.

### 3.5 The positive control PC_NLL

The question is whether the mixes did what their names say, as seen on this corpus's own code and math
documents. Each harvest computes the mean next-token NLL, in nats per token, on the same windows: 32 windows
entirely inside code+math documents and 32 entirely inside web documents. The windows are drawn by
`m1.pc_windows` with `default_rng(0)` from the harvest's own windows, using the harvest's window and BOS rule.

- Δcm = mean over cm windows of (NLL_M − NLL_G); Δdd = Δcm − Δweb. Both have paired bootstrap 95 % intervals
  over windows.
- Fires if the upper bound of Δcm < −f, the upper bound of Δdd < −f, and |Δcm| < 2.0 nats. Here f =
  max(0.01, |identity NLL difference|), where the identity difference is start_b − start_a over the 64
  windows.
- Failing input: G and M swapped. It must fail.
- If PC_NLL does not fire, the run DOES NOT COUNT. In that case the mix labels, the shard identity or the
  directory map is not what it is named.

### 3.6 The specificity control (secondary)

This control checks that the contrast is not a general reshuffle of the top-50 positions over directories. It
compares Δ_MG of the 5 code+math directories with Δ_MG of 1,000 random sets of 5 other directories. The sets
are matched on directory count. The band rule gives each directory about 0.4M tokens of bands 1–2. The
realized token counts are recorded as `token_share_per_dir`. The shares sum to 1, so a random set moves by
about −Δ_cm × 5/45 on average.

---

## 4. Gates

Every gate has a pass condition, a failing input that must make it fail, and the branch taken when it fails.
Instance gates are asserted in the job: the step exits non-zero, `pipeline.py` aborts the launch, and STATUS
shows `exit≠0`. Each instance gate also records its failing input's value in `m1/metrics.json` or
`m1/download.json`. The verdict reads every gate again.

| gate | where | pass | registered failing input | if it fails |
|---|---|---|---|---|
| G-SAME (versions) | job start, before download | torch, transformers equal `2.12.0+cu130`, `5.9.0` | registry `transformers` = `0.0.0` (offline FAIL) | stop; no harvest on other software |
| inputs | job start | the five laptop bank copies have their registered sha256 | one sha256 changed (offline FAIL) | stop; re-upload from the laptop |
| G-CORPUS | after the corpus fetch | manifest sha256 `0c99a8e8…`, tokens sha256 `2aca80ca…` | the manifest with one byte appended (offline FAIL) | stop |
| G-REV | after download | snapshot commit = registered commit | registered commit replaced by zeros (offline FAIL) | stop |
| G-SHARD | after download | the weight files on disk are exactly the registered ones, with their LFS sha256; the index names exactly those; none is a stale shard of `main` | main's three bf16 shards in place of the mix shards (in-job and offline); one oid changed (offline) | stop |
| G-CFG | before load | loaded config = registered start config except `sliding_window` ∈ {4096, 4097} | the as-released mix config (in-job and offline, on the real configs); a width change (offline) | stop |
| G-CAST | after load | all parameters bf16; 7 registered tensors equal the file tensor's round-to-nearest-even cast, bit for bit | each tensor against the next layer's tensor of the same name (in-job); one entry changed by 1.0 (offline) | stop |
| G0 | first batch, inside `neurons_harvest.run` | recomputed a equals the model's down-projection input within 1e-2 | the wrong activation; u × 1.1 (existing, in-job) | stop |
| G-FIT | after harvest | `complete`, `G0_pass`, `dead_poles` 0, 20,560,068 scored tokens; NLL on the first 120,000 tokens in [2.40, 3.20] nats; G-SAME on every registered manifest field | `complete` false; NLL 12.223 (banked T_0, `00_ASSETS.md` line 54); batch 16 (offline FAIL) | stop |
| K0 | the K0 step (§6) | values and subset rules hold, and both failing inputs fail | T_14's banked values; positions + 1 | stop the lane; no mix launches |
| G-ID | verdict | max_b \|I(b)\| ≤ min_b τ(b) / 4 | start_b replaced by the t15 reference (3.9T tokens later); offline on banked data: T_13 for T_15, fails | DOES NOT COUNT ("harvest not repeatable at the scale of the bar") |
| PC_NLL | verdict | §3.5 | G and M swapped | DOES NOT COUNT |
| room | freeze 2 | SUPPORTED and REFUTED reachable on the references; τ < κ | (the CLI asserts) | no freeze; no mix launches |
| complete set | verdict | start_a, start_b, G, M present | a missing G | DOES NOT COUNT (missing) |

The verdict reads every gate's failing input. A gate that passed while its own failing input also passed has
no room, and the run DOES NOT COUNT (`m1.verdict`, `no_room`).

---

## 5. Labels and the verdict, as the script runs it

`python -m infra.pg.m1 verdict --banks studies/next/m1/banks --thresholds studies/next/m1/thresholds.json --out studies/next/m1/verdict_m1.json`

1. Read every harvest's gates (G-FIT, G-SAME, and the instance gates' `gates_pass`).
2. If start_a, start_b, G or M is missing: **DOES NOT COUNT (missing …)**.
3. G-ID and its failing input. bar(b) = max(τ(b), ε_dev(b), |I(b)|).
4. PC_NLL and its failing input.
5. Lead contrast per band: ABOVE_CEILING, SUPPORTED, REFUTED or UNDECIDED (`m1.label_lead`); H_lead =
   `combine`.
6. If R is present: R between G and M, per band where the lead is SUPPORTED (`m1.label_between`); H_order.
   If R is parked: H_order = NOT_RUN.
7. Specificity; H_specific.
8. Exploratory: shifts from the start, C per checkpoint.
9. Headline: **DOES NOT COUNT** if any harvest gate, G-ID or PC_NLL failed, or a failing input passed.
   **DOES NOT COUNT until inspected** if H_lead is ABOVE_CEILING. Otherwise
   `H_lead …; H_order …; H_specific …`.

Every value is written to `verdict_m1.json`, whatever the headline.

The dry run produced each branch on synthetic banks (`M1_OFFLINE_GATE.md` §VERDICT):
- the satisfying case: H_lead, H_order and H_specific all SUPPORTED;
- R above M in band 2: H_order REFUTED;
- M equal to G: H_lead REFUTED;
- the identity pair differing: DOES NOT COUNT;
- the positive control failing: DOES NOT COUNT;
- M far above G: ABOVE_CEILING, "until inspected".

---

## 6. The known number K0

**What it is.** The cheapest re-harvest that runs the whole instance path: registry, download, G-SHARD,
G-CFG, load, G-CAST, `neurons_harvest.run`, the poles/ outputs. It covers the first 2^20 = 1,048,576 corpus
tokens (512 windows, 64 batches, 1,048,064 scored tokens) of one banked grid point. The grid's 640 poles are
used, with T_15's top-50 positions inside that prefix as exemplar positions. That is 1,535 (pole, slot) entries
on 439 poles, and every layer has 17 to 75 of them. The prefix is the shortest that still covers the
harvest's fixed 120,000-token NLL control, which the code asserts.

| lane | point | bank compared (exists on the laptop; sha256 registered in `m1.BANK_SHA`) | failing input |
|---|---|---|---|
| AWS L40S (default) | T_15 `stage1-step1413814`, banked on an L40S | `BANK/1/out/t15_step1413814/poles/exemplar_a.npz` (`84bd59fd…`) and `…/samples/contexts.jsonl` (`a4d46cd8…`) | T_14's banked values at the same positions, `BANK/1b/out/t14_step1000000/poles/exemplar_a.npz` (`9fabd78b…`) |
| Vast RTX PRO 5000 | T_14 `stage1-step1000000`, banked on an RTX PRO 5000 Blackwell | `BANK/1b/out/t14_step1000000/poles/exemplar_a.npz` and `…/samples/contexts.jsonl` (`692e8585…`) | T_13's banked values, `BANK/1b/out/t13_step560000/poles/exemplar_a.npz` (`7976ca71…`) |

The executor uploads these laptop files to the inputs prefix before launch. The job checks their sha256, so
the compared values are exactly the files verified here.

**The rules (K0 passes only if all four hold).**
1. Values: on the 1,535 entries, the relative difference |a_new − a_bank| / |a_bank| has median ≤ 0.0043 and
   99th percentile ≤ 0.0222. These are the banked cross-device values (§0.4), an upper bound for a repeat on
   the same device type.
2. Subset: ≥ 0.99 of the bank's own top-50 positions inside the prefix appear in the prefix run's top-50 of
   that pole. A global top-50 position inside a prefix is always a top-50 position of the prefix.
3. The values rule fails on its failing input. Offline on the laptop, T_14 against T_15 gives median 0.3144
   and p99 1.5021, so it fails by 73-fold at the median.
4. The subset rule fails on positions + 1. Offline, that gives 0.038.

The job also banks the closest and the farthest entry (pole, slot, token, a_bank, a_new) as samples.

**Cost.** Boot 164 s, download 14.61 GB at 234 MB/s (62 s), prefix loop 1,048,064 / 3,334 = 314 s, final write
145 s and gates 60 s make about 0.21 GPU-h. With the references step in the same launch, it is 0.26 GPU-h, or
0.28 at the lowest banked rates. On the Vast RTX PRO 5000 it is 0.24.

---

## 7. Unblinding order

1. Freeze 1: the offline gate prints `M1_DRYRUN_OK`; commit; tag `m1-freeze-1`.
2. L1 (`k0-01`): K0, then the references. K0 must pass. The executor reads K0's two samples and the
   reference NLLs (`m1/metrics.json` → `refs.*.nll_first_120k`) before anything else.
3. Freeze 2: `python -m infra.pg.m1 freeze` on the references only. It writes `thresholds.json`: τ, ε_dev, κ,
   the room check and the references' sha256. Commit; tag `m1-freeze-2`.
4. L2 (`mix-a-01`, `mix-b-01`): the five harvests. During the run, the executor reads only phase and status
   fields and the gates. No cm_share is computed before all harvests are banked.
5. Verdict, once. A rule found wrong afterwards gets a dated exploratory deviation in the journal. The frozen
   rule still decides the headline.
6. An independent reader (a model other than the executor) recomputes the band contrasts from `counts.npz`
   and writes `studies/next/m1/READ_m1.md`. The result document comes after that read.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | Every group of the mix changes at once (web 60 % against 17 %, QA 22 % against 0) | a shift toward code+math is also a shift away from web and QA; direction only | stated; H_specific; the per-directory contrasts are banked |
| 2 | Round5-from-2T has no public settings | R's position is confounded | H_order is secondary; H_lead uses only the two documented mixes |
| 3 | Corpus documents may be in the mixes' training pools (Stack-Edu originals, FineMath) | a shift can be exposure to the very documents, not to the domain | not resolvable here: the skewed mixes' samples are not public (`M1_ASSETS.md` confound 3); stated in every reading |
| 4 | The ends are cast by us; the start was cast by AI2 | at most one bf16 rounding step per entry, common to all three ends | the lead contrast compares ends cast one way; G-CAST verifies our cast |
| 5 | The corpus is the stage-1 distribution, not the stage-2 data | measures preference on stage-1-like documents | stated; it is the fixed corpus of the grid |
| 6 | Harvest nondeterminism across instances | identity spread above zero | G-ID; the bar includes \|I\| |
| 7 | One end of each mix, no seed replicate | the between-mix difference has no run-to-run reference | τ as the drift reference; stated as a limit |
| 8 | Top-50 positions cluster in one document | cm_share per pole jumps between 0 and 1 | band means over 176,128 poles; per-pole distributions reported |
| 9 | Directory is the only label | code inside web pages counts as web | frozen definition (`M1_ASSETS.md` change 6) |
| 10 | Wrong weights loaded | no shift, or a wrong one | G-REV, G-SHARD, G-CAST, PC_NLL |

---

## 9. Runbook

### 9.1 Checklist before the first launch

- Read `.claude/skills/aws-pg/SKILL.md` and the docstrings of `infra/aws_pg/jobs/pipeline.py` and
  `infra/aws_pg/jobs/m1_harvest.py`. Profile `fuelos`, region `us-east-1` (hard-coded in `infra/aws_pg/config.py`).
- Work in worktree `../interp_research-m1`, branch `m1`. Commit named files only.
- Liveness: an instance ID from `./infra/aws_pg/pg ps`, or an artifact timestamp. Poll
  `out/<step>/metrics.json` (the harvest loop rewrites it every 50 batches, about every 246 s on an L40S) and
  `out/<step>/m1/metrics.json` (one write per phase). Never poll `job.log`.
- Cost and incidents go to `infra/aws_pg/operations/2026-11-<dd>_next_m1.md`, not to the journal.

### 9.2 Code

| file | role |
|---|---|
| `infra/pg/m1.py` (NEW) | registry; gates; directory counts; references, freeze, verdict (CLI) |
| `infra/aws_pg/jobs/m1_harvest.py` (NEW) | the job: modes `known`, `harvest`, `refs` |
| `infra/aws_pg/jobs/m1_dryrun.py` (NEW) | the offline gate |
| `infra/aws_pg/steps/next_m1_k0-01.json`, `next_m1_mix-a-01.json`, `next_m1_mix-b-01.json` (NEW) | step lists |
| `infra/pg/neurons_harvest.py`, `infra/aws_pg/jobs/neurons_harvest.py` (`fetch`, `fetch_corpus`), `infra/pg/neurons_corpus.py`, `infra/aws_pg/jobs/pipeline.py` (REUSED, unchanged) | the banked grid's code path |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-m1
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m1_dryrun --hf $SP/m1_gate/hf \
    --grid /Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid \
    --corpus-manifest $SP/m1/corpus_manifest.json --t15-manifest $SP/m1/t15_manifest.json --out $SP/m1_gate/dry
```
It must end with `M1_DRYRUN_OK`. The results are in `M1_OFFLINE_GATE.md`.

### 9.4 S3 layout

One attempt id per launch, never reused: `s3://pg-lineage-769063704696/results/next_m1/<attempt>/`, with
`<attempt>` ∈ {`k0-NN`, `mix-a-NN`, `mix-b-NN`}. NN is 01 for the first launch and +1 for each relaunch;
relaunching means copying the step file under the new number. The inputs are shared and checked by sha256:
`results/next_m1/inputs/`.

```
results/next_m1/inputs/{t15_contexts.jsonl,t15_exemplar_a.npz,t14_exemplar_a.npz,t14_contexts.jsonl,t13_exemplar_a.npz}
results/next_m1/<attempt>/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_m1/<attempt>/out/pipeline_status.json
results/next_m1/<attempt>/out/<step>/{manifest.json,metrics.json,controls.json,topk/,stats/,snapshots/,poles/,samples/}
results/next_m1/<attempt>/out/<step>/m1/{metrics.json,download.json,counts.npz,quantiles.npz}
results/next_m1/k0-NN/out/refs/<t00|t12|t13|t14b|t14c|t15>/m1/{counts.npz,metrics.json}
```

### 9.5 Sessions (AWS lane, the default)

```
B=s3://pg-lineage-769063704696
BANK=/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad   # any Python with numpy serves
AWS="--profile fuelos --region us-east-1"
```

**S1. Freeze 1 and pre-launch checks (laptop).**
1. Run §9.3 to `M1_DRYRUN_OK`. Commit the files of §9.2, this spec and `M1_OFFLINE_GATE.md`. Tag
   `m1-freeze-1`. Record the tag, SHA and UTC in §0.
2. `./infra/aws_pg/pg ps` and `./infra/aws_pg/pg audit`: no stray instance.
3. Existence checks, each of which must list objects:
   `aws s3 ls $B/results/mdl/neurons/w1/q4b/1/out/corpus/corpus/tok/olmo3/ $AWS`;
   for each of `1/out/t00_step0 1b/out/t12_step320000 1b/out/t13_step560000 1b/out/t14_step1000000 1c/out/t14_step1000000 1/out/t15_step1413814`:
   `aws s3 ls $B/results/neurons/w3/grid/<it>/topk/ $AWS | wc -l` (must be 32) and
   `aws s3 ls $B/results/neurons/w3/grid/<it>/controls.json $AWS`.
   The attempt prefixes must be empty: `aws s3 ls $B/results/next_m1/ $AWS`.
4. Upload the inputs from the laptop bank:
   ```
   aws s3 cp $BANK/1/out/t15_step1413814/samples/contexts.jsonl  $B/results/next_m1/inputs/t15_contexts.jsonl $AWS
   aws s3 cp $BANK/1/out/t15_step1413814/poles/exemplar_a.npz    $B/results/next_m1/inputs/t15_exemplar_a.npz $AWS
   aws s3 cp $BANK/1b/out/t14_step1000000/poles/exemplar_a.npz   $B/results/next_m1/inputs/t14_exemplar_a.npz $AWS
   aws s3 cp $BANK/1b/out/t14_step1000000/samples/contexts.jsonl $B/results/next_m1/inputs/t14_contexts.jsonl $AWS
   aws s3 cp $BANK/1b/out/t13_step560000/poles/exemplar_a.npz    $B/results/next_m1/inputs/t13_exemplar_a.npz $AWS
   ```

**S2. L1, the known number and the references (`k0-01`, 0.26 GPU-h projected, budget 40 min).**
```
aws s3 cp infra/aws_pg/steps/next_m1_k0-01.json $B/results/next_m1/k0-01/steps.json $AWS
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_m1/k0-01/steps.json" \
    --no-weights --type g6e.2xlarge --s3-prefix results/next_m1/k0-01 --budget-min 40 --json
./infra/aws_pg/pg watch results/next_m1/k0-01
```
Then, on the laptop:
```
aws s3 sync $B/results/next_m1/k0-01/out/ studies/next/m1/banks/k0-01/ $AWS --exclude "*" \
    --include "k0/m1/*" --include "k0/manifest.json" --include "k0/metrics.json" --include "refs/*/m1/*"
mkdir -p studies/next/m1/banks/refs && cp -R studies/next/m1/banks/k0-01/refs/* studies/next/m1/banks/refs/
```
Read `k0/m1/metrics.json` → `K0` (pass, values, subset, samples). Read `k0/manifest.json` → `tokens_per_second`,
`download.MB_per_s`, `instance type` (from `pg ps`). Read the reference NLLs. Record the instance type T1.

**S3. Freeze 2 (laptop).**
```
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m1 freeze --banks studies/next/m1/banks --out studies/next/m1/thresholds.json
```
It prints `M1_FREEZE_OK` or stops on a failed room check. Then the stop rule (§9.7) with the measured rates.
Commit `thresholds.json`; tag `m1-freeze-2`. No mix launches from a commit without that tag.

**S4. L2, the five harvests (two instances in parallel, the same instance type T1).**
```
aws s3 cp infra/aws_pg/steps/next_m1_mix-a-01.json $B/results/next_m1/mix-a-01/steps.json $AWS
aws s3 cp infra/aws_pg/steps/next_m1_mix-b-01.json $B/results/next_m1/mix-b-01/steps.json $AWS
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_m1/mix-a-01/steps.json" \
    --no-weights --type <T1> --s3-prefix results/next_m1/mix-a-01 --budget-min 390 --json --detach
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_m1/mix-b-01/steps.json" \
    --no-weights --type <T1> --s3-prefix results/next_m1/mix-b-01 --budget-min 270 --json --detach
```
Instance A runs start_a, M, R. Instance B runs start_b, G. If one of the two cannot launch on T1, terminate the
other only if it has not yet finished its start harvest, and relaunch both as `-02` on the other L40S type
(`g6e.4xlarge`). The identity pair must share an instance type. Watch both with `pg watch`. Poll
`out/<step>/metrics.json`.

**S5. Pull and unblind (laptop).**
```
for at in mix-a-01 mix-b-01; do aws s3 sync $B/results/next_m1/$at/out/ studies/next/m1/banks/ $AWS --exclude "*" \
    --include "*/m1/*" --include "*/metrics.json" --include "*/controls.json" --include "*/manifest.json" \
    --include "*/samples/contexts.jsonl"; done
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m1 verdict --banks studies/next/m1/banks --thresholds studies/next/m1/thresholds.json \
    --out studies/next/m1/verdict_m1.json
```
The laptop needs numpy only. `counts.npz` is a few MB per harvest. Run the verdict once. Then the independent
read (§7 step 6).

### 9.6 The Vast lane (only if the user tops up Vast)

- One GPU model for every harvest: RTX PRO 5000 Blackwell. Pin it in `gpu_query` (§0.3 item 8).
- K0 point T_14 (§6). In `next_m1_k0-01.json`, replace the k0 args'
  `--branch stage1-step1413814 --bank-exemplar t15_exemplar_a.npz --fail-exemplar t14_exemplar_a.npz --bank-contexts t15_contexts.jsonl`
  with `--branch stage1-step1000000 --bank-exemplar t14_exemplar_a.npz --fail-exemplar t13_exemplar_a.npz --bank-contexts t14_contexts.jsonl`.
- The box user reads only `results/mdl/*` (`infra/vast/cells/pipeline_job.py` docstring). So every S3 path in
  the three step files becomes `results/mdl/next_m1/…`, and the references become `results/mdl/neurons/w3/grid/…`
  (unverified for `1c`; check with `aws s3 ls` before launch). Copy the step files to `next_m1_vast_*.json` and
  upload them to `results/mdl/next_m1/<attempt>/steps.json`.
- Manifest `infra/vast/manifests/next_m1_<attempt>.json`, after `neurons_w3_grid_b.json`. Image
  `769063704696.dkr.ecr.us-east-1.amazonaws.com/shackleton:mdl-v1`, `weights_mode` `download`, env
  `HF_ONLINE=1 HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub HF_HUB_DISABLE_XET=1`, one cell
  `{"module": "infra.vast.cells.pipeline_job", "steps_s3": "s3://pg-lineage-769063704696/results/mdl/next_m1/<attempt>/steps.json", "pg_git_sha": "<the m1-freeze-1 commit>"}`,
  `watch.max_billed_h` = 1.4 × the projection. The shipped code is `git archive HEAD`, so freeze 1 must be committed.
- `python -m infra.vast.controller plan <manifest>`, then `rent <manifest> <offer_id>`, as in the W3 grid.
  `infra/vast/budget.json` allows $42.0 in total and records $33.29 spent. The projected Vast run is 7.34 GPU-h
  × $0.785 = $5.76, which fits under the remaining $8.71 only if the realized price holds.
- **The reliability check, recorded for the K0 rental** in `infra/vast/operations/2026-11-<dd>_next_m1_k0.md`:
  rent-to-running time (controller rent UTC to the first `heartbeat`); download rate (`m1/download.json` →
  `MB_per_s`); `metrics.json` cadence (the `updated_utc` stamps, against the expected 178 s at 4,612 tokens/s);
  billed hours from the Vast invoice against uptime (first to last heartbeat). L2 launches on Vast only if
  K0 passed and the record shows no stall longer than 10 minutes.

### 9.7 Budget, stop rule, park order

| launch | contents | GPU-h, measured L40S rate (3,334 tok/s, 234 MB/s) | GPU-h, lowest banked L40S rate (3,215 tok/s, 143 MB/s) | Vast RTX PRO 5000 (4,612 tok/s) | `--budget-min` |
|---|---|---|---|---|---|
| L1 `k0-01` | boot, K0, references | 0.26 | 0.28 | 0.24 | 40 |
| L2 `mix-a-01` | boot, start_a (bf16), M, R (fp32) | 5.68 | 5.92 | 4.26 | 390 |
| L2 `mix-b-01` | boot, start_b (bf16), G (fp32) | 3.79 | 3.95 | 2.84 | 270 |
| **total** | 5 harvests, K0, 6 references | **9.73** | **10.15** | **7.34** | |

Per harvest: download (14.61 GB bf16, 29.19 GB fp32), the 20,560,068-token loop, 155 s snapshot write, 145 s
final write, 60 s for counts and quantiles, 60 s for the positive control, and 30 s (bf16) or 90 s (fp32) for
sha256, load and G-CAST. The two load figures are estimates; the others are from the T_15 manifest. Boot is
164 s per instance, from the 2026-11-04 mechanics screen (launch to STATUS 650 s minus job wall 486 s). At
$2.24 per hour the AWS total is about $21.8.

**Ceiling: 12 GPU-h.** The kill limits (`--budget-min`) sum to 11.67 h plus boot, so no combination of
timeouts passes 12.

**Stop rule.** Before L2, recompute the two L2 projections from K0's measured `tokens_per_second` and
`MB_per_s`. If the GPU-h spent on L1 plus the L2 projections exceed 11.7, apply the park order.

**Park order.** The identity pair and the shared start are never parked. Parked first: **R**, the last step of
`mix-a-01`. Remove its line from the step file and lower `--budget-min` by 2.0 h; H_order becomes NOT_RUN.
Then **G**, then **M**. Parking either leaves H_lead DOES NOT COUNT (missing), so it is a decision for the
user, not the executor. A timeout during R loses only R, because every earlier step's outputs are already
synced.

### 9.8 Artifacts

- Per harvest: `out/<step>/m1/metrics.json` holds the gates with their failing-input values, `counts`,
  `token_share_per_dir`, the positive control's windows and NLLs, and `fit`. Also
  `out/<step>/m1/{counts.npz,quantiles.npz,download.json}`, plus the harvest's own files as in the grid.
- Samples beside every score: `samples/contexts.jsonl` of each harvest (640 grid poles, top-50 contexts with
  text, document and a), and K0's closest and farthest entries. Before any number is quoted, the executor
  opens for each harvest one top-50 slot counted as code+math and one counted as other, decodes the text, and
  checks the directory against it.
- Laptop: `studies/next/m1/{thresholds.json,verdict_m1.json,banks/,JOURNAL.md,READ_m1.md}`.
- Operations: `infra/aws_pg/operations/2026-11-<dd>_next_m1.md` (instance ids, types, launch and STATUS UTC,
  uptime, billed GPU-h against projection); on Vast, the reliability record of §9.6.

---

## 10. Prediction grid and the picture (filled at unblinding, whatever the result)

| item | rule or prior | observed | verdict |
|---|---|---|---|
| K0 | §6 rules | `k0-01`, 2026-11-05: 1,535 of 1,535 values bit-identical (median and p99 relative difference 0.0); subset 1.000; failing inputs 0.3144 / 1.5021 and 0.1355; corpus NLL first 120k 2.675372037760417 = bank (`M1_OFFLINE_GATE.md` §K0 record) | PASS |
| τ, ε_dev, κ | frozen at freeze 2 | τ 0.0082, 0.0066, 0.0023, 0.0035; ε_dev ≤ 2.8e-5; κ 0.183, 0.177, 0.192, 0.202 (`thresholds.json` `4bb56ded…`) | frozen |
| G-ID | ≤ τ_min / 4 (prior: exactly 0) | I = 0, 0, 0, 0; changed-pole share 0.0 (start_a and start_b `counts.npz` byte-identical); line 0.000563; failing input (start_a against t15) fails | PASS |
| PC_NLL | §3.5 (prior: Δcm < −0.05 nats) | Δcm = −0.148 [−0.204, −0.098] nats per token; Δcm − Δweb = −0.240 [−0.296, −0.187]; identity NLL difference 0.0, floor 0.01; failing input (G and M swapped) fails | FIRES |
| Δ_MG(b), 4 bands | > bar(b) (prior +0.01 to +0.06) | +0.0109 [0.0104, 0.0114], +0.0100 [0.0095, 0.0105], +0.0174 [0.0170, 0.0178], +0.0196 [0.0192, 0.0200]; bar 0.0082, 0.0066, 0.0023, 0.0035; ceiling 0.183 to 0.202 | SUPPORTED in every band; H_lead SUPPORTED. At the low end of the prior. |
| Δ_RG(b), Δ_MR(b) | > bar(b) (prior λ near 0.6) | Δ_RG +0.0077, +0.0068, +0.0115, +0.0150; Δ_MR +0.0032 [0.0028, 0.0037], +0.0032 [0.0028, 0.0035], +0.0059 [0.0057, 0.0062], +0.0046 [0.0044, 0.0049]; λ = Δ_RG / Δ_MG 0.70, 0.68, 0.66, 0.76 | bands 2–3 SUPPORTED; bands 0–1 UNDECIDED (Δ_MR is positive but its lower bound is below the bar); H_order UNDECIDED |
| H_specific | §1.3 | code+math Δ_MG 0.0109, 0.0100, 0.0174, 0.0196 against the 99th percentile of \|Δ_MG\| over 1,000 random 5-directory sets: 0.0031, 0.0034, 0.0051, 0.0068 | SUPPORTED in every band |

Filled 2026-11-05 from `studies/next/m1/verdict_m1.json` (sha256 `a329ce9c…`), the single verdict run. Every harvest's gates passed and no step was parked. The independent read (`studies/next/m1/READ_m1.md`, an Opus agent on branch `m1-read`, merged as `a464c376e`) reproduces the headline and all 131 deterministic fields exactly; it lists eight code–spec differences, none of which changes the headline (READ §3). Fable's read, the different-model read, is pending. The picture: `studies/next/m1/figures/fig_m1.png`, rendered by `figures/fig_m1.py` from `verdict_m1.json` only.

The picture is one chart, x = layer band, y = C in fractions of top-50 slots (absolute, not normalized). Each
band shows the start (start_a), G, R and M as points with 95 % intervals. The shaded band is start ± bar(b).
The identity pair is drawn as a second start point. The title states the finding in one sentence. The footer
lists the attempt ids, the commit and the sha256 of `verdict_m1.json`. The figure is rendered from
`verdict_m1.json` only and regenerated after any correction.

## 11. References

`M1_ASSETS.md` sources [S1]–[S19]; the Hugging Face reads of §0.2; the OLMo 3 report
(https://arxiv.org/abs/2512.13961, Tables 5 and 7, as read in `M1_ASSETS.md`). No other citation is used.
