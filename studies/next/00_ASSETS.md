# 00 — Asset inventory for the NEXT brainstorm

Written 2026-11-04 (CST evening) by the inventory agent. Read-only pass. No model was run.

Path conventions. A path with no prefix is under `/Users/abaybektursun/projects/interp_research/` (main worktree).
`BANK/` is `/Users/abaybektursun/projects/interp_research_banks/`. `s3://B/` is `s3://pg-lineage-769063704696/`
(listed with `--profile fuelos --region us-east-1`). `branch:path` is a file read with `git show` from a branch that
has no worktree. Every number below was read from the file or URL named beside it on 2026-11-04.

**The answer in one line.** We hold exact top-50 contexts and activation histograms for every MLP neuron of
`allenai/Olmo-3-1025-7B` (352,256 neurons, 704,512 poles) at 16 stage-1 checkpoints on one fixed 20.56M-token
corpus drawn from that model's own training mix: 16 directories of about 1.9 GB each on S3 (`s3://B/results/neurons/w3/`,
30,949,522,014 bytes, 2,399 objects). Only the 640-pole subset is on the laptop.

---

## 1. Training-checkpoint access we already have

### 1.1 Model and the 16 points

Model `allenai/Olmo-3-1025-7B` (key O7): 32 layers, MLP width 11,008, hidden 4,096, SiLU, 352,256 neurons
(`studies/already-clean/SPEC_D1.md` §4.1, line 347). `n_poles` 704,512 at every point (`BANK/neurons/w3/grid/1/out/t00_step0/metrics.json`).

Tokens seen come from the official OLMo-core table `Tokens Seen` column
(https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv). Every row equals
step × 4,194,304. HF revision sha and GPU are from each point's `manifest.json` on S3
(`s3://B/results/neurons/w3/grid/{1,1b}/out/<dir>/manifest.json`, `revision`, `branch`, `device_name`).

| id | HF branch | revision sha | tokens seen | harvest GPU | tokens/s |
|---|---|---|---|---|---|
| T_0 | `stage1-step0` | f129a5d1… | 0 | L40S | 3,307 |
| T_1 | `stage1-step1000` | d82e70ec… | 4,194,304,000 | L40S | 3,234 |
| T_2 | `stage1-step2000` | 6a238f66… | 8,388,608,000 | L40S | 3,215 |
| T_3 | `stage1-step3000` | 9de48909… | 12,582,912,000 | L40S | 3,216 |
| T_4 | `stage1-step5000` | 825e84ed… | 20,971,520,000 | L40S | 3,224 |
| T_5 | `stage1-step8000` | ebfe7b62… | 33,554,432,000 | L40S | 3,234 |
| T_6 | `stage1-step12000` | a1cbaea4… | 50,331,648,000 | L40S | 3,246 |
| T_7 | `stage1-step20000` | 04eeee4f… | 83,886,080,000 | RTX PRO 5000 Blackwell | 4,551 |
| T_8 | `stage1-step35000` | d1dbb0e2… | 146,800,640,000 | RTX PRO 5000 Blackwell | 4,575 |
| T_9 | `stage1-step60000` | 08b5da69… | 251,658,240,000 | RTX PRO 5000 Blackwell | 4,602 |
| T_10 | `stage1-step100000` | 1d074dbc… | 419,430,400,000 | RTX PRO 5000 Blackwell | 4,612 |
| T_11 | `stage1-step180000` | c515906c… | 754,974,720,000 | RTX PRO 5000 Blackwell | 4,615 |
| T_12 | `stage1-step320000` | af83fc98… | 1,342,177,280,000 | RTX PRO 5000 Blackwell | 4,616 |
| T_13 | `stage1-step560000` | db8c654b… | 2,348,810,240,000 | RTX PRO 5000 Blackwell | 4,623 |
| T_14 | `stage1-step1000000` | cb286dcd… | 4,194,304,000,000 | RTX PRO 5000 Blackwell | 4,620 |
| T_15 | `stage1-step1413814` | 373bad25… | 5,929,965,715,456 | L40S | 3,334 |

Tokens/s is `tokens_per_second` in each point's `BANK/neurons/w3/grid/*/out/<dir>/metrics.json`. All 16 have `complete: true`,
`G0_pass: true`, `dead_poles: 0` (same files). Jobs a (T_15, T_0–T_6) and b (T_7–T_14) ran on vast.ai rentals
(`studies/already-clean/DATA_CONTRACT.md` §10). A device check re-harvested T_14 on an L40S: 0 of 640 poles change
presence (`DATA_CONTRACT.md` §10; `s3://B/results/neurons/w3/device_check/t14_1b_vs_1c.json`, not re-read).

Mean next-token NLL on the first 120,000 corpus tokens, per point (`nll_corpus_first_120k.mean_nll` in
`s3://B/results/neurons/w3/grid/*/out/<dir>/controls.json`): T_0 12.223, T_1 4.563, T_2 3.936, T_3 3.657, T_4 3.443,
T_5 3.288, T_6 3.188, T_7 3.088, T_8 3.013, T_9 2.940, T_10 2.896, T_11 2.850, T_12 2.816, T_13 2.771, T_14 2.715, T_15 2.675.

### 1.2 Corpus scored

Source `allenai/dolma3_mix-6T-1025-7B` at commit `2ca900fb…`, 50 of its 165 `data/*` directories, documents deduplicated
by text hash, three nested bands (`s3://B/results/neurons/w1/q4b/1/out/corpus/corpus/manifest.json`, 55 KB).

- 50M corpus: 34,789 documents, 50,540,791 OLMo-3 tokens (52,468,404 Qwen3, 53,540,348 Gemma3). Same manifest, `bands`, `tokenizers`.
- 20M corpus (bands 1–2, the grid corpus): 14,106 documents, 20,570,743 OLMo-3 tokens. Scored per point: 20,560,068 tokens
  in 10,044 windows of 2,048 (`n_tokens_scored`, `n_windows` in each grid `manifest.json`; window position 0 and BOS excluded).
- 10M prefix (band 1): 10,432,539 OLMo-3 tokens; banked as `snapshots/band1/` inside every grid point.
- Arrays on S3: `docs.jsonl.gz` 81,702,089 bytes; `tok/olmo3/tokens.npy` 202,163,292 bytes; also `qwen3`, `gemma3` (same prefix).

### 1.3 What is banked per point

On S3, per grid point (`s3://B/results/neurons/w3/grid/1/out/t03_step3000/`, 139 objects, 1,901,097,102 bytes):
`manifest.json`, `metrics.json`, `controls.json`; `topk/layer_LL.npz` and `stats/layer_LL.npz` for all 32 layers
(every pole: exact top-50 token positions with a, g, u; max |a|; 400-bin log-|a| histogram; q99/q999/q9999; density;
sign counts; `pos0_share`; schema in `DATA_CONTRACT.md` §2, §8); the same two directories at the 10M band
(`snapshots/band1/`); `poles/` for the 640 grid poles (`exemplar_a.npz`, `positions.npz`, `top1000.npz`, `vocab.json`);
`samples/contexts.jsonl` (16,516,677 bytes), `first_batch.json`, `sample_draw.json`.

On the laptop (`BANK/neurons/`, 516 MB; `w3` 253 MB, `w4` 3.3 MB, `w5` 259 MB; `du -sh`):

- 16 point directories `BANK/neurons/w3/grid/{1,1b}/out/<tNN_stepN>/`, each with only `metrics.json`,
  `samples/contexts.jsonl` (640 lines; each line one pole with its top-50 contexts: 64 tokens before, 16 after, a, g, u)
  and `poles/exemplar_a.npz` (`a`, `g`, `u`, each [640, 50]: the activation at T_15's top-50 positions of each pole, read at that point).
- `BANK/neurons/w3/o7/draw/S.json` (600 poles) and `S_gate.json` (40). Its README says both were rebuilt on 2026-11-04 from
  T_15's `contexts.jsonl` because the original draw files were not found on S3 (`BANK/neurons/w3/o7/draw/README.md`).
- `BANK/neurons/w4/o7_s_v2/explain/`: `clusters.jsonl`, `judgments.jsonl`, `manifest.json` (clusterer and judge on
  `anthropic/claude-sonnet-5-5`, seeds 4111–4113; `manifest.json` `batches`).
- `BANK/neurons/w5/verdict/`: `metrics.json` (created 2026-10-24T04:06:05Z) and `samples/` (12 jsonl files).

### 1.4 The 300-neuron selection

O7 audit sample: 300 neurons drawn uniformly at random with a logged seed, stratified over layers, both poles, 600 poles
(`SPEC_D1.md` §4.4, line 427). The development set S was to be drawn from the audit sample's classified poles in class
order; at 600 of 600 the order has no effect (`DATA_CONTRACT.md` §10), so S is the whole audit sample. Read from
`BANK/neurons/w3/o7/draw/S.json`: 600 poles = 300 neurons × 2 signs, 16–20 poles per layer. S_gate: 40 poles = 20 more
neurons × 2 signs. S ∪ S_gate: 640 poles = 320 neurons, exactly 20 poles (10 neurons) per layer, which is the grid pole list.

### 1.5 Other harvests on S3 (all neurons, final weights)

| model | prefix | tokens scored | size |
|---|---|---|---|
| O7 T_15, 50M corpus | `s3://B/results/neurons/w1/o7_t15/1/` | 50,515,866 (`manifest.json`) | 912,384,867 B, 84 objects |
| Qwen3-4B-Base, 50M | `s3://B/results/neurons/w1/q4b/1/` | not read | 3,945,050,912 B (includes the corpus) |
| gemma-3-4b-pt, 50M | `s3://B/results/neurons/w1/g4/1/` | not read | 918,874,598 B |
| OLMo-2-1124-7B (revision 7df9a825…), 50M | `s3://B/results/instruments/e6b/1/out/harvest/` | 50,515,866 (`studies/instruments/results/e6/b/1/harvest/manifest.json`) | 2,619,801,921 B, 306 objects |

### 1.6 S3 bucket, top two levels

`aws s3 ls s3://B/` gives 17 prefixes. Second level (count of entries; sizes from `--summarize` where taken):

| prefix | second level |
|---|---|
| `artifacts/` | `vlat/` (one `metrics.json`) |
| `bake/` | `DONE`, `env_bake.log`, `pip-freeze.txt` |
| `code/` | 5,436 code tarballs `<hash>.tgz` plus 1 prefix; `--summarize`: 5,560 objects, 21,283,084,383 B |
| `darwin/` | 89 prefixes (`banks/`, `darwin_*` runs) |
| `datasets/` | `eval_subsets/` (refcoco, textvqa, vqav2), `hf/` (4,148,946,491 B, 359 objects), `seed.log` |
| `depthanchor/` | `camera/ code/ figures/ frames/ gates/ ladder/ libero_assets/ liberods/ seed/` |
| `depthopen/` | `arms/ liberoplus_assets/ r5_sample/ recon/ seed/ src/ subset/ subset_depth/` |
| `humboldt/` | 21 prefixes |
| `mercator/` | `banks/ recensus/`, `expectations.jsonl` |
| `meta/` | 2 json files |
| `parallax/` | 7 prefixes (`cloud-builder/ control-plane/ input-staging/ metadata-probes/ results/ source-manifests/ source/`) |
| `postcheck/` | `code/ frames/ seed/` |
| `reference/` | `relaw/ w1_real_dw/ w4_crosscoder/` |
| `republic/` | `battery/ runs/` |
| `results/` | 1,401 prefixes. Largest families by prefix count: plenum 179, mercator 132, tessera 57, copernicus 57, darwin 43, gestalt 41, erlangen 39. Current-program prefixes: `neurons/` (80,337,194,515 B, 11,219 objects), `stagediff/` (71,140,396,741 B), `instruments/` (3,153,011,527 B), `depthoff/` (39,685,170 B), `reuse/`, `mechanics/`, `unite/`, `alwayson/`, `depthanchor/`, `depthopen/`, `postcheck/`, `public_check/`, `mdl/` |
| `seed/` | `stagediff/`, `eval_subsets_code.tgz`, `seed.log` |
| `weights/` | `hf/` (`hub/` 644,673,619,231 B, 1,254 objects; `xet/`), `lacuna_units/`, `plenum_film/` |

`results/neurons/` third level: `manual/ off/ rw/ show/ transcripts/ w0/ w1/ w2c/ w2d/ w3/ w4/ w5/`, six earlier `w5_run<n>_<utc>/`, `x/`
(`center_probe/ evilqwen/ realprompt/`). Weight cache `weights/hf/hub/` holds 37 model repos: Qwen3-4B (Base, Instruct-2507, plain),
gemma-3-4b (pt, it), gemma-2-2b, `google/gemma-scope-2-4b-pt` (61,977,044,786 B, 46 objects), Molmo2-4B, Molmo2-ER, MolmoAct2 (Pretrain,
base, Think, LIBERO, Think-LIBERO, DROID, others), MolmoAct-7B-D variants, paligemma, siglip, zephyr. **No OLMo repo is in the cache**;
O7 checkpoints were downloaded in-job and deleted after each harvest, 14.6 GB each (`SPEC_D1.md` §4.1).

---

## 2. Public model suites with intermediate training checkpoints

Method: branch and tag lists from `https://huggingface.co/api/models/<repo>/refs`; license and creation date from
`https://huggingface.co/api/models/<repo>`; statements from the model card `https://huggingface.co/<repo>/raw/main/README.md`;
file lists from `.../api/models/<repo>/tree/<branch>`; optimizer files by HTTP HEAD (no download). All read 2026-11-04.
"Count" = distinct checkpoint refs; "spacing" = the most common gap in steps.

| suite (repo) | sizes | count and spacing | data order released | optimizer states | license |
|---|---|---|---|---|---|
| **OLMo 3 7B** `allenai/Olmo-3-1025-7B` | 7B | stage1 1,421 (0–1,413,814, every 1,000; 4,194,304 tokens/step from the OLMo-core CSV); stage2 49 + 3 mix variants; stage3 13 | mix manifests and pre-tokenized `.npy` shards are public (OLMo-core `src/scripts/official/OLMo3/README.md` lines 30–46); a per-step document index is not a published file; order reconstruction not verified by running | OLMo-core checkpoints listed in `OLMo-3-1025-7B.csv` (1,475 rows). `stage3/step11921/model_and_optim/.metadata` HTTP 200 (2,388,519 B). Stage-1 directories gave 404 on every file name tried: **not verified for stage 1** | apache-2.0 |
| **OLMo 3 32B** `allenai/Olmo-3-1125-32B` | 32B | stage1 655 (0–656,000, every 1,000); stage2 two ingredients × 24 + 1 merge; stage3 12; 2 `nooptim` branches; CSV 742 rows | as 7B; pretraining used parameter averaging at several points (OLMo3 README line 71) | not verified | apache-2.0 |
| OLMo 3 post-training, 7B | 7B | Think-SFT 43 (every 1,000 to 43,000); Think-DPO 1; Think 55 (every 25 to 1,375); Instruct-SFT 1; Instruct 8 (every 50 to 400); RL-Zero-Math 19 (every 100) | not verified | not verified | apache-2.0 |
| OLMo 3 / 3.1 post-training, 32B | 32B | 3-32B-Think-SFT 22 refs (11 steps to 10,790); 3-32B-Think 15 (every 50 to 750); 3.1-32B-Think 46 (every 50 to 2,300); 3.1-32B-Instruct 1 | not verified | not verified | apache-2.0 |
| Olmo Hybrid `allenai/Olmo-Hybrid-7B` (created 2026-01-28) | 7B | stage1 1,419 (0–1,414,078, every 1,000); stage2 25 + 25; stage3 25 | not verified | not verified | apache-2.0 |
| **OLMo 2** `allenai/OLMo-2-0425-1B`, `-1124-7B`, `-1124-13B`, `-0325-32B` | 1B, 7B, 13B, 32B | 1B stage1 195 (every 10,000 to 1,907,359) + 3×24 stage2, plus `OLMo-2-0425-1B-early-training` 37 (every 1,000 to 36,000); 7B stage1 928 (150–928,646, every 1,000) + 3×12; 13B stage1 646 (0–596,057) + 12/9/11/36; 32B stage1 689 (0–721,901) + 13/12/37/1 | `allenai/OLMo` README: stage-2 runs "with different data order" per seed and a provenance CSV (lines 108–136); stage-1 order not verified | 7B verified: `https://olmo-checkpoints.org/ai2-llm/peteish7/step1000-unsharded/optim.safetensors` 58,389,056,611 B (model 29,194,508,992 B). 1B and 13B: same file name 404, not verified | apache-2.0 |
| **Pythia** `EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b,12b}` (+ `-deduped`) | 70M–12B | 154 each: 0, 1, 2, 4, …, 512, then every 1,000 to 143,000; 2,097,152 tokens/step (README line 130) | yes: "exact same data, in the exact same order", pre-tokenized data and `utils/batch_viewer.py` (`EleutherAI/pythia` README lines 8, 49, 67, 222) | README line 132: final step only, others on request. But `EleutherAI/neox-ckpt-pythia-12b` branch `step1000` lists 255 `*_optim_states.pt` files (tree API), 155 branches | apache-2.0 |
| **LLM360**, now org `IFM` (HF redirects `LLM360/*` to `IFM/*`): Amber, K2, CrystalCoder, K2-V2 | 7B, 65B, 7B, 70B | Amber 360 (`ckpt_000`–`ckpt_359`); K2 120 `ckpt_N` + 20 `ministage2_ckpt_N`; CrystalCoder 93 distinct steps (0–214,387, gaps 1,500 and 3,000); K2-V2 (2025-12-01) 29 `base_N` tags + 9 midtraining tags | yes per cards: Amber "the full data sequence"; K2 "The full data sequence" at `LLM360/K2Datasets` | Amber `ckpt_100` lists 14 files, none optimizer. K2 and CrystalCoder not verified (tree API: invalid revision / auth error) | apache-2.0 |
| IFM **K2-Horizon** (created 2026-09-01) | 0.9B, 3.7B, 7B, 32B, MoVA-36B-A4B, 375B-A23B | 7B: 11 pretrain tags (every 100,000 to 1,100,000), 42 midtraining, 10 SFT, 2 RL; 375B: 23 pretrain tags | training data public (card: datasets `IFM/K2-Horizon-Pretrain-Data`, `-Midtrain-Data`; "Fully open"); order not verified | not verified | apache-2.0 |
| **SmolLM3** `HuggingFaceTB/SmolLM3-3B-checkpoints` | 3B | 118 pretraining (every 40,000 steps = 94.4B tokens; GBS 2,359,296), 10 long-context (every 4,000), post-training refs | not stated in card | none on `main`; not verified | apache-2.0 |
| **SmolLM2** `HuggingFaceTB/SmolLM2-{135M,360M,1.7B}-intermediate-checkpoints` | 135M, 360M, 1.7B | 8 (every 240,000), 16 (every 160,000), 41 (every 125,000; GBS 2,097,152) | not stated | not verified | apache-2.0 |
| **Apertus** `swiss-ai/Apertus-8B-2509`, `-70B-2509` | 8B, 70B | 8B: 44 pretraining (125–2,627,139, mostly every 50,000) + 27 long-context; 70B: 47 + 34 | card: "Training data reconstruction scripts: github.com/swiss-ai/pretrain-data" (not opened) | not verified (AdEMAMix optimizer per card) | apache-2.0 |
| Apertus v1.5 `swiss-ai/Apertus-v1.5-*` | 8B, 70B | API returned 401 | not verified | not verified | not verified |
| **Marin** `marin-community/marin-8b-base`, `marin-32b-base` | 8B, 32B | 8B: 44 `step-N` tags (20,000–1,419,999, mostly every 40,000). Card: "On November 3, 2026, we pruned 32 intermediate checkpoints"; 32B: 6 phase branches | not verified | not verified | apache-2.0 |
| **DCLM** `apple/DCLM-7B`, `TRI-ML/DCLM-1B` | 7B, 1B | 1 branch each (`main`): no intermediate checkpoints on HF | — | — | apple-ascl; apache-2.0 |
| **Zyphra** `Zyphra/Zamba-7B-v1-phase1` | 7B | 191 `iterN` (every 2,500 iterations); `Zamba2-7B` 401; `ZAYA1-base` main only | not verified | `iter230000` lists 10 files, none optimizer | apache-2.0 |
| **TinyLlama** `TinyLlama/tinyLlama-intermediate-checkpoints`, `-after-1T-token`, `tinyLLaMA-v1.1-checkpoints` | 1.1B | 144 (5k–720k steps, every 5k); 190 (485k–1,430k, every 5k); 218 (5,000–1,089,914, every 5,000) | not verified | not verified | apache-2.0 |
| Trillion Labs `trillionlabs/{0.5B,1.9B,Tri-70B}-Intermediate-Checkpoints` (2025-08/09) | 0.5B, 1.9B, 70B | 60, 33, 12 (every 20,000 steps); 7B repo 401 | not verified | not verified | apache-2.0 |

2026 search. Queries: `"intermediate checkpoints" open language model released 2026 training dynamics`; `"training checkpoints"
released fully open LLM 2026 data order optimizer states`; an extended search on 2026 fully open releases; and a Hugging Face API
name search (`Intermediate-Checkpoints`, `training-checkpoints`, created after 2025-10). New suites found and verified:
Olmo-Hybrid-7B (2026-01-28) and IFM K2-Horizon (2026-09-01). The API name search found only small personal repos.

Sources: [OLMo-3 7B refs](https://huggingface.co/api/models/allenai/Olmo-3-1025-7B/refs),
[OLMo-core OLMo3 CSV](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv),
[OLMo-core OLMo3 README](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/README.md),
[allenai/OLMo README](https://github.com/allenai/OLMo/blob/main/README.md),
[Pythia README](https://github.com/EleutherAI/pythia/blob/main/README.md),
[Amber](https://huggingface.co/IFM/Amber), [K2](https://huggingface.co/IFM/K2), [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B),
[K2-Horizon-375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B), [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B),
[SmolLM3 checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints), [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-2509),
[marin-8b-base](https://huggingface.co/marin-community/marin-8b-base), [Tri-70B checkpoints](https://huggingface.co/trillionlabs/Tri-70B-Intermediate-Checkpoints).

---

## 3. Our closed results, one line each

Source list: `studies/unite/INVENTORY.md` §1 and `studies/unite/PROPOSAL.md` §1. The worktrees they name
(`-neurons-explain`, `-neuronoff`, `-neuronrw`, `-evilqwen`, `-realprompt`) no longer exist under `/Users/abaybektursun/projects/`;
the files were found merged into main or in `BANK/`. Every number is read from the JSON named.

| study | result | artifact (checked to exist) |
|---|---|---|
| ALREADY-CLEAN D1, packing | O7 late-born jobs judged unrelated: p_u 0.265 [0.211, 0.321], rule H_SP, 1,238 clusters on 600 poles; positive control a_ctrl 0.924 | `BANK/neurons/w5/verdict/metrics.json` → `q2.p_u`. Named path `-neurons-explain/bank/w5/verdict/metrics.json`: **path not found** (worktree gone) |
| D1, birth order | main job oldest: b 0.476 [0.436, 0.517] on 574 poles; the permutation reference has mean 0.414 and 97.5th percentile 0.449; p 0.0005 | same file → `q2.birth_order` |
| D1, births | main-job births per point T_0…T_15: 0,1,1,5,8,19,17,34,42,50,74,89,97,109,47,7; non-main 1,854 jobs; share born at T_1 0.0004 | same file → `q2.births`, `q2.share_born_T_1` |
| D1, one-job shares | ONE: Q4B 0.597 (358/600), G4 0.563, O7 0.518. Not understood (UNK): 0.320, 0.365, 0.415. Second-reader kappa 0.180, 0.167, 0.070 | same file → `q1.<M>.fractions`, `q1.<M>.kappa_label` |
| NEURONOFF | Arm A cancelled: the positive control failed on 0.417 (Q4B) and 0.371 (G4) of poles. Under the cancelled label M −0.000101 [−0.0045, 0.0108] nats (n 26) and −0.0000348 (n 40) | `studies/neuronoff/results/verdict_{Q4B,G4}_v2.json` → `pc_cancel_share`, `stats.own` |
| NEURONRW | exploratory: label agrees with the pole's top output tokens when literal: D_w 0.553 (n 19), 0.618 (17), 0.556 (18); non-literal 0.086 (81), 0.096 (83), 0.030 [−0.055, 0.116] (82) | `studies/neuronrw/results/literal_split_{Q4B,G4,O7}.json` → `arm0` |
| STAGEDIFF | robot fine-tunes lose B1 next-token accuracy: MolmoAct2→LIBERO −0.096 [−0.192, −0.014], →DROID −0.123 [−0.219, −0.027], n 73 | `studies/stagediff/results/verdict/verdict.json` → `hops.H5.dB1`, `hops.H5p.dB1` |
| SINKPATCH | wikitext NLL recovery R: parent residual at all positions, layers 12–31: LIBERO 0.629 [0.611, 0.646], DROID 0.431 [0.414, 0.449]; logit bias: LIBERO −0.008, DROID 0.378; BOS-only patches ≈ 0 | `studies/sinkpatch/results/verdict/verdict.json` → `hops.{H5,H5p}.B3.arms.{all_parent,calib_bias,bos_parent}` |
| DEPTHANCHOR | shuffling all 100 depth cells moves the action by 0.000868 [0.000800, 0.000970] of an image swap (median over 120 frames); label H_epi | `depthanchor:studies/depthanchor/results/contrast/1/out/aggregate.json` → `ratios.shuffle`, `decision.label` |
| DEPTHOFF | LIBERO success 195/200 with the depth block not generated against 194/200 with it; zeroed 191/200; wrong image 0/10; headline `no_loss` (NOGEN), inconclusive (ZERO). Errata of 2026-11-04 change the timing claim, not this | `depthoff:studies/depthoff/results/verdict.json` → `pooled_rates`, `headline`; errata `/Users/abaybektursun/projects/interp_research-walkthrough-slider/studies/depthoff/ERRATA_2026-11-04.md` |
| DEPTHOPEN | shuffle effect at gate 0.018, 0.5 and no gate: 0.000963, 0.001122, 0.001037; no-gate minus 0.018 paired +0.000077 [+0.000028, +0.000170] on 40 frames; these arms score 0/200 on LIBERO (CLOSEOUT) | `depthopen:studies/depthopen/results/arm/{A0,A3,A4}/1/out/metrics.json` → `stages.r2.absolute.shuffle`; `depthopen:studies/depthopen/results/paired_A4_vs_A0.json`; `depthopen:studies/depthopen/CLOSEOUT.md` |
| POSTCHECK | re-measured depth claims; not re-read here | `postcheck:studies/postcheck/results/RESULTS.md` (exists) |
| INSTRUMENTS E1 (O7 T_15) | zero replacement with the global top-1% exclusion: two-neuron chance level 0.0074 [−0.0019, 0.0321], 95th percentile 0.210 (n 56); real cases 0.293 [0.216, 0.356], 150 cases on 104 neurons; mean replacement 0.067, p95 0.384 | `studies/instruments/results/e1/verdict_e1.json` → `reported.A1.zero__e`, `reported.cases_I.zero__e`, `reported.A1.mean__e` |
| INSTRUMENTS E2 (Qwen3.5-0.8B) | frozen room gate failed, every rule cancelled (`exit` 4). Repetition share on 300 MATH-500 problems: base 0.760, editor's edit 0.443, direction-only 0.453, product-64 0.507, product-256 0.383, random 0.667/0.680; product-64 recovers 72 of the editor's 128 columns (chance 8.7) | `studies/instruments/results/e2/verdict_e2.json` → `gates_failed`, `reported.per_arm.*.rep`, `reported.recovery` |
| INSTRUMENTS E6b (OLMo-2-1124-7B) | the generic knock-out term is present: N_O2 0.123 [0.087, 0.169], 56 pairs, against line 0.10 | `studies/instruments/results/e6/verdict_e6b.json` → `reading.N_O2` |
| EVILQWEN | steering two neurons (B) against none (A): hostility 3.46 [3.04, 3.88] against 1.00 on 1–5; coherence 2.29 [2.14, 2.45], below the 2.5 line (incoherent branch) | `studies/x/evilqwen/bank/summary.json` → `groups.run1/instruct.conditions.{A,B}` |
| REALPROMPT | frozen: Q4B 3 NOT SCORABLE (code-rule gate failed), G4 3 REFUTED | `studies/x/realprompt/results/score.json` → `<M>.predictions[].verdict`, `Q4B.gate_failures` |
| HIGH-ENERGY LOW-GAIN (draft, PaliGemma lineage) | energy-matched deletion is 2.90× [2.74, 3.08] costlier than deleting the band, cheaper in 98/98 windows; spectrum-matched 2.44×; reader gain 0.13×, 0.26×, 0.72× (draft table) | `studies/lineage/paligemma1-pi0/research/gutenberg_gb1_verdict.json` → `sites.q.E_over_C`, `S_over_C`, `core_cheaper_in_n_windows`; reader gain from `papers/highenergy-lowgain/DRAFT.md` line 77 (JSON `amundsen_reader_gain.json` not re-read) |

Other closed studies seen in main but not read for this file: `studies/reuse/README.md` (SPEC to SPEC12, ablation of upstream
drivers at birth checkpoints, 11.655 GPU-h), `studies/mechanics/README.md` (2026-11-04), `studies/alwayson/`, `studies/neuronshow/`.

---

## 4. Compute and harnesses

**AWS launcher** (`infra/aws_pg/pg.py` docstring and `infra/aws_pg/config.py`):

- One job = one ephemeral EC2 instance from AMI `ami-0cee78ee4b1c24614` (pg-env-v1: Python 3.12, torch 2.12+cu130), root 200 GB,
  self-terminating; an EventBridge schedule `pg-reap-<instance>` kills it at budget + 600 s.
- Instance types: preference list `INSTANCE_TYPES = ["g6e.2xlarge", "g6e.4xlarge"]`, default `g6e.2xlarge` (1× L40S 48 GB,
  8 vCPU, 64 GB host RAM); up to 4 at once under the 32-vCPU G quota; `--type p5.4xlarge,p4de.24xlarge` for jobs over 80 GB (config.py lines 26–34).
- Price: `ONDEMAND_USD_PER_HR = None` in config.py (not recorded). Operations records quote $2.24/h for g6e.2xlarge
  (`infra/aws_pg/operations/2026-11-03_alwayson_edit_run.md`, "quoted here, not banked") and $3.00424/h for g6e.4xlarge
  (AWS Price List API, read 2026-11-04, `infra/aws_pg/operations/2026-11-04_mechanics_screen_run.md`).
- Launch: `./infra/aws_pg/pg run <job_module> --args " ..." --budget-min 240 [--type ...] [--weights models--org--name,...] [--s3-prefix ...]`;
  subcommands `run watch map ps kill audit liveness logs collect` (pg.py lines 670–740). The working tree's `infra/`, `configs/`,
  `scripts/` are tarred to `s3://B/code/<hash>.tgz`. Weights sync from `s3://B/weights/hf/` at launch.
- Results land at `s3://B/results/<short>/` by default, or `--s3-prefix` (pg.py line 213), with `STATUS`, `metrics.json`, `job.log`, `heartbeat.jsonl`.
- Vast lane: `infra/vast/` (controller, cells); ceiling $42.0, "spent to date $33.29" (`infra/vast/budget.json`). The W3 grid ran there.

**Robot evaluation harness (MolmoAct2 on LIBERO).** DEPTHOFF ran `allenai/MolmoAct2-Think-LIBERO` on standard LIBERO through
`depthoff:infra/aws_pg/jobs/depthoff_loop.py` (hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1, horizon 220;
`depthoff:studies/depthoff/results/loop/libero_goal/1/out/metrics.json` → `stages.env`, `stages.inputs`). Four suites × 10 tasks × 5
episodes, paired across conditions; 810 rollouts in all, 200 paired episodes per condition (`depthoff:studies/depthoff/results/verdict.json`
→ `n_rows`, `n_paired_episodes`). Median seconds per episode: ON 27.62, ON2 27.41, ZERO 28.93, NOGEN 13.0; per action call 2.18 s with depth,
0.98 s without, on one L40S (`verdict.json` → `latency`). About 7 GPU-hours on 4 × g6e.2xlarge (`depthoff:studies/depthoff/CLOSEOUT.md` line 141).
The loop code is only on branches `depthoff` and `depthanchor` (`depthanchor:infra/aws_pg/jobs/depthanchor_loop.py`). Main holds
`infra/aws_pg/jobs/molmoact2_libero.py` ("LIBERO success rate for MolmoAct2 … through the released LeRobot checkpoint", docstring) and
`infra/aws_pg/jobs/n5a_pi0_libero_eval.py`. LIBERO assets: `s3://B/depthanchor/libero_assets/`, `s3://B/depthopen/liberoplus_assets/`.
`studies/vla-trace/README.md` §6 adds the loaders: `infra/pg/matrix_models.py` (Molmo2 on transformers 4.57.1, MolmoAct2 on 5.9.0).

**Weight-edit replication (E2).** On `Qwen/Qwen3.5-0.8B` (revision `2fc06364…`) it rebuilds the released loop edit: in blocks 15 and 19,
64 `down_proj` columns each get W ← W − 6 u(uᵀW), with u the top singular vector of the released ΔW (`studies/instruments/SPEC_instruments.md`
lines 21, 166, 552). It then picks other 16/64/256-column sets by |u·W[:, n]|, by the loop-minus-productive activation contrast, by their
product, or at random (norm- or KL-matched), and generates 300 MATH-500 answers per arm to read the repetition share (same file, line 172).
The rebuilt edit differs from the released tensor by relative Frobenius 0.0012 (line 666).

---

## 5. Code we can reuse

First docstring line of each module, `infra/pg/` unless noted.

1. `neurons_harvest.py` — "D1 neuron harvest (N1) … One checkpoint per call": exact top-K, histograms, quantiles, G0 recompute gate.
2. `neurons_corpus.py` — "D1 corpus (N2)": the banded Dolma 3 corpus under three tokenizers.
3. `neurons_explain.py` + `neurons_llm.py` — "D1 explain side … Laptop only"; "one request/response format … and three backends" (anthropic, openai_compat, manual).
4. `neurons_verdict.py` — "D1 verdict … pure functions plus a CLI" (θ_s, classes, p_u, birth order).
5. `neurons_device_check.py` — "one checkpoint harvested on two GPUs (LOCAL, banked arrays only)".
6. `reuse_ablate.py` — "mean-replacement ablation of upstream MLP neurons and the fall of a pole's activation".
7. `reuse_birth.py`, `infra/aws_pg/jobs/reuse_dating.py` — the same instrument at a job's birth checkpoint; "one O7 grid checkpoint per call".
8. `unite_route.py` — "first-order upstream attribution of a pole's activation, per context".
9. `instruments_values.py`, `instruments_linear.py`, `instruments_adapter.py` — written values (zero, mean, median, resample), the first-order prediction, a model adapter for Olmo, gemma and Qwen.
10. `alwayson_edit.py` + `instruments_e2.py` — "the column edit, its Frobenius norm and the strength match".
11. `neuron_off_core.py` — "turn off one pole of one MLP neuron and measure what changes".
12. `neuron_rw_core.py` — "push poles' W_down columns into the residual stream, read what the" model writes.
13. `stagediff_lens.py` — lenses over `lm_anatomy`, using the vendored Jacobian lens `infra/pg/vendor/jlens` (anthropics/jacobian-lens @ 581d3986).
14. `sinkpatch.py` — "residual-stream position patching".
15. `cached_donor.py`, `lm_patching.py`, `head_patching.py` — co-resident patched model with a per-layer donor cache; layer/position and head patching.
16. `gemma_scope.py`, `scope2_sae.py` — "Direct loader + JumpReLU forward + residual-stream harvest for Gemma Scope SAEs"; Gemma Scope 2 loader.
17. `diff_sae.py`, `delta_crosscoder.py`, `tessera_crosscoder.py` — "a BatchTopK sparse autoencoder trained DIRECTLY on the activation delta"; crosscoder harvest/train/stats.
18. `mechanics.py` — random-token sequences, fp32 accumulators for induction excess and per-head induction scores, per-head ablation hook.
19. `probe_server.py` — "load Q4B, G4 and O7 once on one GPU, then answer probe requests from a directory".
20. `infra/aws_pg/jobs/molmoact2_libero.py`, `depthoff:infra/aws_pg/jobs/depthoff_loop.py` — LIBERO rollouts for MolmoAct2.

---

## 6. Gaps (only what was checked)

1. **No dense activations.** The harvest banks top-50 positions and per-pole histograms only, plus top-1,000 and exemplar values
   for the 640 grid poles (`DATA_CONTRACT.md` §2). No per-token activation arrays, no residual-stream vectors, at any checkpoint.
2. **MLP neurons only.** Hooks are on `gate_proj`, `up_proj`, `down_proj` (`DATA_CONTRACT.md` §2). No attention-head banks at the 16 points.
3. **16 of 1,421 stage-1 checkpoints**, none of stage 2 or 3 and none of the Think/Instruct chain. No OLMo weights in
   `s3://B/weights/hf/hub/`; each new checkpoint is a 14.6 GB in-job download.
4. **Laptop holds only the 640-pole subset.** The all-neuron `topk/` and `stats/` are on S3 only (30.9 GB for the grid).
5. **Jobs and birth dates exist only for S (600 poles) and S_gate (40).** The draw files were rebuilt on 2026-11-04 (`BANK/neurons/w3/o7/draw/README.md`).
6. **No SAE or transcoder dictionaries for OLMo-3 or OLMo-2.** The only dictionary in the weight cache is `google/gemma-scope-2-4b-pt`
   (for G4). `s3://B/reference/w4_crosscoder/` exists, contents not read.
7. **No data-order index.** Our corpus is a 50.5M-token rule-based sample of the stage-1 mix, not in training order. We cannot say which
   documents the model saw between two checkpoints. OLMo-core publishes the mix manifests and pre-tokenized shards, so one could be built.
8. **No optimizer states banked.** Public optimizer state verified only at OLMo-3 7B `stage3/step11921` and OLMo-2 7B `step1000`
   (58.4 GB per checkpoint), and Pythia-12B NeoX `step1000` (255 files).
9. **Per-checkpoint loss is one number.** `controls.json` holds mean NLL on the first 120,000 tokens; no per-token loss is banked.
10. **Hourly price not banked.** `ONDEMAND_USD_PER_HR = None` (`infra/aws_pg/config.py`); the prices above are quoted in operations notes.

**The asset nobody else has.** Public suites give weights. We add, for OLMo-3 7B, the all-neuron exact top-50 contexts and activation
histograms at 16 stage-1 checkpoints on one fixed corpus from the model's own training mix, with 2,454 LLM-clustered jobs dated by birth
checkpoint on 600 poles (`BANK/neurons/w5/verdict/metrics.json` → `q2.n_clusters`), and the same all-neuron harvest at the final
weights of OLMo-2-1124-7B, Qwen3-4B-Base and gemma-3-4b-pt.

## Corrections, 2026-11-05 (Fable, after `00b_DATA_ORDER.md`)

- §2 table, OLMo 3 7B row ("order reconstruction not verified by running"; stage-1 optimizer "not verified"): the
  stage-1 run's own checkpoint files are public in the Hugging Face bucket under `checkpoints/OLMo25/stepN/`
  (`config.json`, `data_paths.txt`, `train/rank0.pt`), and the batch order is a seeded NumPy PCG64 shuffle over
  8,192-token windows (seed 34,521 + epoch 1), batch k = instances 512k to 512k+511. See `00b_DATA_ORDER.md`.
- §6 gap 7 ("No data-order index"): superseded. The index can be computed from public files; the loss check on one
  reconstructed batch (expected 2.1240015 at step 60,001) has not been run.
