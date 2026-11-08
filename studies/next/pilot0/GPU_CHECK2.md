# pilot0 GPU loss check 2 (J8): two new steps, pad targets excluded

**State, 2026-11-05T22:05Z: both steps are in the pass branch under the frozen rule.** Step 61,001 from b1 (below).
Step 100,001 from rerun b2: N = 1.9779166 on 512 rows, |N − W| = 0.0483 > |S − W| = 0.0094 (section "Rerun b2
result" at the end). The known number holds: max |diff| 0 on 32 rows. The b1 result below is kept as written. Its
does not count branch for step 100,001 is replaced by b2's pass branch. The verdict is Fable's.

**Head's verdict (Fable, 2026-11-05 22:05Z):** the second data-order check is closed. Step 61,001 (b1) and step 100,001 (S from b1, N from b2) are both in the pass branch under the frozen pad-excluded rule, and the known-number gate holds (max |diff| 0 on 32 rows against 1e-4). The head recounted N = 1.9779166 from the per-row sums over 4,169,956 non-pad positions, |S − W| = 0.0093759, |N − W| = 0.0483056 and the known-number gate from the S3 files at 21:49Z, before this record was written, and agrees with every number in it; that recount is the different-model read. The open items listed under "What stays open" stay open. J8 cost at most 1.164 GPU-hours against the deviation ceiling of 1.70.

## Result (written 2026-11-05T16:50Z, after the run)

**Step 61,001 is in the pass branch.** `batch_61001/` forwarded through `stage1-step61000` gives 2.1252322 nats per
token with pad targets excluded, against the logged 2.1173441: +0.0079, inside the 0.0124 band. Its neighbour
`batch_61000/` gives 2.0350325, which is 0.0823 from the same logged value. **Step 100,001 is in the does not count
branch.** Its target is complete: 2.0355981 against the logged 2.0262222, +0.0094. But the job's time limit stopped
its neighbour at 320 of 512 rows, below the 400-row minimum. No step is in the fail branch. The known number
(run 5) did not run. The verdict is Fable's. The branches above are the output of the frozen rule code
(`$SP/j8/scripts/rule.py`, output `$SP/j8/verdict_rule_output.txt`, run once at 16:42:12Z).

### Per step

Units: nats per token, mean next-token cross-entropy over every target position that is not pad (100277), token-weighted.

| s | checkpoint | S (target) | rows | N (neighbour) | rows | W (log at s) | S − W | N − W | branch |
|---|---|---|---|---|---|---|---|---|---|
| 61,001 | `stage1-step61000` | 2.1252322 | 512 | 2.0350325 | 512 | 2.1173441 | +0.0078880 | −0.0823116 | pass |
| 100,001 | `stage1-step100000` | 2.0355981 | 512 | 1.9455010 | 320 | 2.0262222 | +0.0093759 | −0.0807213 | does not count (rows) |

Files: `batch_<s>/loss_check_sdpa.json` → `pad.mean_ce_excluding_pad_targets` and `rows_done`. Both readings of
"also" (see the rule) give the same branch at both steps, because dN > 0.0124 at both. The neighbour of 100,001
is a partial mean over its first 320 rows (`complete: false`). Rows are in batch order, which is the shuffled
order, so those rows are not selected by content. That value does not enter the rule.

A recount from the per-row fields (`per_row_ce_sum_excluding_pad_targets` summed, divided by the non-pad targets
counted from `tokens.npy`) equals all four file values to the printed digit. The pad-target counts from
`tokens.npy` equal the files' counts (`$SP/j8/recount.txt`).

### Three readings per batch

| file | rows | pad targets excluded | every position | rows without pad (n) | pad targets | top-1 |
|---|---|---|---|---|---|---|
| `batch_61001/loss_check_sdpa.json` | 512 | 2.1252322 | 2.2835233 | 2.1269371 (504) | 45,901 | 0.53440 |
| `batch_61000/loss_check_sdpa.json` | 512 | 2.0350325 | 2.2059702 | 2.0353505 (502) | 48,228 | 0.54834 |
| `batch_100001/loss_check_sdpa.json` | 512 | 2.0355981 | 2.1765206 | 2.0343336 (505) | 38,816 | 0.54943 |
| `batch_100000/loss_check_sdpa.json` | 320 | 1.9455010 | 2.0572999 | 1.9486520 (316) | 18,625 | 0.56798 |

With every position counted, both targets lie far above their logs (+0.166 and +0.150), as in pilot 0. The two
exclusion readings, whole pad rows dropped and pad targets dropped, differ by 0.0017 (61,001) and 0.0013 (100,001).
Both lie inside the band, so these data still cannot separate them.

### Against the prediction

- Predicted S − W about +0.005 at both steps. Measured +0.0079 and +0.0094. Pilot 0 measured +0.0055 at step 60,001.
  All three offsets are positive. The largest leaves 0.0030 of room below the 0.0124 band edge. Their cause was
  not measured.
- Predicted N about 2.03 for step 61,001. Measured 2.0350. Its drop below its own log W(61,000) = 2.0699108 is
  −0.0349. Pilot 0's neighbour dropped −0.0389. This is the expected effect of update s − 1, which trained on the
  neighbour batch. For step 100,001, the 320-row partial gives 1.9455 against W(100,000) = 2.0171299. The predicted
  value was about 1.98. A 320-row partial mean is not the logged 512-row quantity, so this comparison is exploratory.

### Known number (run 5): missing

Run 5 (`batch_60001/`, first 32 rows, `stage1-step60000`, against the banked pilot-0 values within 1e-4) did not
start, because the job was killed during run 4. So this run has no reproduction of a banked number. The recorded
stack is the same as pilot 0's: torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S, bfloat16 parameters, sdpa
(fields `torch`, `transformers`, `gpu`, `dtype` in each file). That is a record of versions, not a reproduction.
The checkpoint `stage1-step60000` did download (448.0 s, snapshot `08b5da69…`, `$SP/j8/b1/job.log`).

### Spread of the per-row loss (pad targets excluded)

| bin (nats per token) | `batch_61001/` | `batch_61000/` | `batch_100001/` | `batch_100000/` (320 rows) |
|---|---|---|---|---|
| [0, 0.5) | 3 | 6 | 3 | 6 |
| [0.5, 1.0) | 16 | 25 | 23 | 22 |
| [1.0, 1.5) | 45 | 40 | 43 | 30 |
| [1.5, 2.0) | 123 | 124 | 139 | 96 |
| [2.0, 2.5) | 199 | 239 | 215 | 117 |
| [2.5, 3.0) | 108 | 71 | 79 | 45 |
| [3.0, 4.0) | 12 | 6 | 7 | 2 |
| [4.0, 6.0) | 3 | 0 | 0 | 0 |
| ≥ 6.0 | 0 | 0 | 1 | 0 |
| no non-pad target (8,192 pad tokens) | 3 | 1 | 2 | 2 |

Field: `per_row_mean_ce_excluding_pad_targets`. The standard error of the batch mean over the rows without pad is
0.0252, 0.0238, 0.0232 and 0.0321 (sd of `per_row_mean_ce` over those rows ÷ √n). The same caveat as pilot 0
applies: the forward is deterministic on a fixed batch, so S − W is a difference between the HF forward and the
training forward, not sampling error.

### Twenty raw rows

The ten rows with the largest pad-excluded row loss among the four files, and ten random rows
(`numpy.random.default_rng(20261105)`, fixed before any output existed), drawn from all 1,856 scored rows. Full first
200 decoded tokens, pad runs shown as `<|pad|>×k`, in `$SP/j8/report_out.txt` (script `$SP/j8/scripts/report.py`).
First characters here, `\n` for a line break.

| set | file, row | row loss | pad tokens | source file | text |
|---|---|---|---|---|---|
| largest | `batch_100001/` 22 | 6.1571 | 7,936 | `olmocr_science_pdfs/science_math_and_technology/000001.npy` | starts with 200+ pad; first non-pad: `We are very pleased to announce that five Associate Editors have joined` |
| largest | `batch_61001/` 269 | 5.6412 | 0 | `stack_edu/Python/000000.npy` | `JvKT2Dhmw+eMlbPe+pa752VfO9oV2MvmdV\ngT2YeJS+MZtbvqZjET66TUw8` (base64) |
| largest | `batch_61001/` 91 | 4.0913 | 0 | `common_crawl/games/000000.npy` | ` like this with no studio experience and go up against studios that were on their 3rd` |
| largest | `batch_61001/` 351 | 4.0812 | 0 | `common_crawl/literature/000024.npy` | `; he deliberately charms us, by the incantation of his verse, into an unreal world` |
| largest | `batch_100000/` 271 | 3.9870 | 0 | `common_crawl/science_math_and_technology/000005.npy` | ` Lahaul A Spili dialnula,\nHimauhal Pu-inn. Gaol Surv.` (OCR errors) |
| largest | `batch_61001/` 45 | 3.7645 | 0 | `common_crawl/entertainment/000028.npy` | ` Goolies is a 1970 English-language series directed by Hal Sutherland` |
| largest | `batch_61000/` 163 | 3.6715 | 0 | `common_crawl/health/000002.npy` | ` of !ndia.ground. and their conseHuences in multiple "ital organs.` (OCR errors) |
| largest | `batch_100000/` 237 | 3.6270 | 0 | `common_crawl/literature/000013.npy` | ` angel guided to the coal, Friedl, and she carried her stuttering into fluency later.` |
| largest | `batch_61001/` 189 | 3.4758 | 0 | `common_crawl/science_math_and_technology/000000.npy` | `Electromagnetic Radiation. You don’t see it, but that doesn’t mean it’s not there.` |
| largest | `batch_100001/` 287 | 3.4600 | 0 | `common_crawl/entertainment/000026.npy` | `Don’t doubt it so quickly. The editors of this site are very close to Akon` |
| random | `batch_100000/` 239 | 1.8297 | 0 | `olmocr_science_pdfs/science_math_and_technology/000001.npy` | `47 FR 7235 for ref. to this S.O.)</td>\n<td>Applied to lands currently under Tracts 1 and 1b` |
| random | `batch_100000/` 205 | 0.8508 | 0 | `stack_edu/CSharp/000006.npy` | ` cálculo: ");\n            double n2 = double.Parse(Console.ReadLine());` |
| random | `batch_100001/` 61 | 2.1340 | 0 | `common_crawl/software/000031.npy` | ` access broker (IAB) known to be used in ransomware attacks.` |
| random | `batch_100000/` 211 | 2.5317 | 0 | `common_crawl/art_and_design/000008.npy` | `\n  3. Oversee how they translate their observations into their art.` |
| random | `batch_61000/` 492 | 2.0376 | 0 | `common_crawl/education_and_jobs/000023.npy` | ` relations.\n  4. Understand business goals and objectives.\n  5. Policy management.` |
| random | `batch_100000/` 182 | 2.5355 | 0 | `common_crawl/entertainment/000013.npy` | ` they were a sports agent that told me he was one of 100 assistant DAs.` |
| random | `batch_100001/` 88 | 1.2175 | 0 | `stack_edu/JavaScript/000006.npy` | `];\n\n    var slided = sidebar[s];\n\n    if (slided == true) {` |
| random | `batch_61001/` 371 | 1.8416 | 0 | `common_crawl/industrial/000017.npy` | ` \n\nProperties: Colorless transparent monoclinic crystal or white powder` |
| random | `batch_100001/` 384 | 1.7509 | 0 | `common_crawl/software_development/000006.npy` | `Speed Insights, GTmetrix etc. Your bid should include a summary of your relevant experience` |
| random | `batch_100001/` 446 | 2.6118 | 0 | `common_crawl/software_development/000010.npy` | ` using a propriety based JavaScript MVC architecture built on top of a command queue.` |

### Banked samples (row 0, field `row0_first32`)

| file | position | last context token | target, p(target) | top-1, p(top-1) | scored |
|---|---|---|---|---|---|
| `batch_61001/loss_check_sdpa.json` | 0 | `Ġparty` | `,`, 0.1003 | `,`, 0.1003 | right |
| `batch_61001/loss_check_sdpa.json` | 1 | `,` | `Ġor`, 0.0142 | `Ġand`, 0.1119 | wrong |
| `batch_100001/loss_check_sdpa.json` | 1 | `Ġhe` | `Ġwas`, 0.0933 | `Ġwas`, 0.0933 | right |
| `batch_100001/loss_check_sdpa.json` | 0 | `Anyway` | `Ġhe`, 0.0008 | `,`, 0.8103 | wrong |

### Runs, time, cost

| run | file | rows | wall s (`metrics.json` → `runs[].seconds`) | forward s | load s |
|---|---|---|---|---|---|
| 1 | `batch_61001/loss_check_sdpa.json` | 512 | 730.2 | 640.2 | 7.9 |
| 2 | `batch_61000/loss_check_sdpa.json` | 512 | 640.2 | 631.1 | 2.6 |
| 3 | `batch_100001/loss_check_sdpa.json` | 512 | 640.2 | 631.9 | 2.7 |
| 4 | `batch_100000/loss_check_sdpa.json` | 320 (killed) | 410.2 | 392.5 | 2.5 |
| 5 | — | 0 (not started) | — | — | — |

Downloads of 14,605,764,269 B each: `stage1-step61000` 435.7 s (blocking), `stage1-step100000` 351.8 s and
`stage1-step60000` 448.0 s (both in the background, during runs 1–3) (`$SP/j8/b1/job.log`). Peak GPU memory
21.19 GB in every run. Instance `i-03e36825423328376`, g6e.2xlarge, NVIDIA L40S: launched 15:50:08Z; STATUS
`exit=137` at 16:40:19Z (the 2,940 s `timeout` in `run_job.sh`); shutting-down by 16:41:11Z. At most 0.851 GPU-hours,
about $1.91 at $2.24 per hour, inside the 1.0 ceiling. Operations record:
`infra/aws_pg/operations/2026-11-05_next_j8_loss_check2_run.md`. S3: `s3://pg-lineage-769063704696/results/next_pilot0/2026-11-05_b1/`.

### What is left

Step 100,001 needs its neighbour `batch_100000/` scored on all 512 rows under `stage1-step100000`. Run 5, the known
number, also needs to run. Per the head's decision, the step is rerun later under the same rule. One instance for
those two runs costs one checkpoint download (6–8 min at today's rate) plus 640 s plus about 65 s, or two downloads
if run 5 goes on the same instance: about 0.35–0.5 GPU-hours.

## Rule, written 2026-11-05T15:48:23Z (file time of this document's first version), before the launch at 15:50:08Z

Source: `../09_PLAN.md`, "Head's decisions" item 4. Copied here unchanged, with the quantities named.

Convention, kept from pilot 0 (`README.md`, "Which directory the GPU check uses"): trainer step s trains the
0-based batch index s − 1, with the weights after s − 1 updates. HF revision `stage1-step<a>` holds the weights
after a updates. Directories are named by trainer step: `batch_<s>/` holds batch index s − 1.

| symbol | what it is | file and field |
|---|---|---|
| S | target: pad-excluded mean CE of the batch trainer step s consumed (`batch_<s>/`), checkpoint `stage1-step<s−1>`, sdpa | `batch_<s>/loss_check_sdpa.json` → `pad.mean_ce_excluding_pad_targets` |
| N | neighbour: pad-excluded mean CE of the batch of step s − 1 (`batch_<s−1>/`), the same checkpoint, sdpa | `batch_<s−1>/loss_check_sdpa.json` → `pad.mean_ce_excluding_pad_targets` |
| W | W&B run `zckrmllo`, `train/CE loss` at step s | the W&B slice named per step below |
| rows | rows scored | `rows_done` in the same files |

Pad-excluded mean: summed fp32 next-token cross-entropy over every target position whose target is not 100277,
divided by the number of those positions, over all scored rows (token-weighted). Units: nats per token.

The two steps:

| s | checkpoint | target | neighbour | W (W&B `zckrmllo`, `train/CE loss` at s) | file of W |
|---|---|---|---|---|---|
| 61,001 | `stage1-step61000` (sha `6931d9fe…`) | `batch_61001/` (index 61,000) | `batch_61000/` (index 60,999) | 2.1173441410064697 | `batch_61001/wandb_zckrmllo_steps_60999_61002.json`; the same value at `_step` 61,001 in `$SP/dataorder_in/wandb/zckrmllo_60001_61001.json` and in `../00b_DATA_ORDER.md` line 124 |
| 100,001 | `stage1-step100000` (sha `1d074dbc…`) | `batch_100001/` (index 100,000) | `batch_100000/` (index 99,999) | 2.0262222290039062 | `batch_100001/wandb_zckrmllo_steps_99000_101002.json` |

`$SP` is `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.
The W&B slices were fetched on 2026-11-05 from `https://api.wandb.ai/graphql` without a key, project
`ai2-llm/Olmo-3-1025-7B`, run `zckrmllo` (display name `OLMo25-20250723T222053+0000`), keys `_step`,
`train/CE loss`, `optim/step skipped` (script `$SP/j8/scripts/wb_range.py`). Neither step s nor s − 1 is a skipped
update (`optim/step skipped` 0 at 60,999–61,002 and at 99,999–100,002).

Per step s, with dS = |S − W| and dN = |N − W|:
- **does not count branch**: fewer than 400 of the 512 rows scored in either batch; or dS ≤ 0.0124 and dN ≤ 0.0124.
- **pass branch**: otherwise, dS ≤ 0.0124 and dN > dS.
- **fail branch**: otherwise. That is dS > 0.0124, or dN ≤ dS.

The rule as a whole: fail if either step is in the fail branch. The verdict is Fable's, read from the JSON files.

One reading had to be fixed to write the rule as code. The decision says "does not count if the neighbour batch
is also within 0.0124". I read "also" as: the target and the neighbour are both within 0.0124. A target outside
0.0124 is the fail branch whatever the neighbour gives. The other reading (neighbour within 0.0124, target
anywhere, gives does not count) changes the branch only when dS > 0.0124 and dN ≤ 0.0124. If the data land
there, the result section says so. Code: `$SP/j8/scripts/rule.py`.

0.0124 is half of 0.0248, the per-step sd of `train/CE loss` over steps 60,001–61,000. Over 100,002–101,001 the
sd is 0.024837 (`batch_100001/wandb_zckrmllo_steps_99000_101002.json`). The band is not rescaled.

## Room check and the failing input (run 2026-11-05T15:45Z, before the launch)

Inputs: the banked pilot-0 files, read by the rule code, not typed. Output: `$SP/j8/room_check.txt`.

| input | S | N | W | dS | dN | branch |
|---|---|---|---|---|---|---|
| as registered: target `batch_60001/loss_check_sdpa.json`, neighbour `batch_60000/loss_check_sdpa.json` | 2.1294733 | 2.0447261 | 2.1240015 | 0.0054718 | 0.0792754 | pass branch |
| failing input: the two files swapped | 2.0447261 | 2.1294733 | 2.1240015 | 0.0792754 | 0.0054718 | fail branch |

Both rows have 512 rows scored. So the rule can pass on a batch that reproduced its log under this reading, and it
fails when the neighbour is put in the target's place. Synthetic inputs reach the other branches (W = 2.0):
(S 2.005, N 2.003) and (S 2.005, N 2.010) give does not count; S 2.02 gives fail; 399 neighbour rows give does
not count (same file).

Chance level. A wrong batch lands within 0.0124 of W about as often as another step's logged loss does:

| s | other steps compared | within ±0.0124 of W | W(s) − W(s − 1) |
|---|---|---|---|
| 61,001 | 60,001–61,000 | 162 of 1,000 (16.2 %) | +0.0474 |
| 100,001 | 100,002–101,001 | 275 of 1,000 (27.5 %) | +0.0091 |

Output: `$SP/j8/band_fractions.txt`. W(100,001) = 2.0262 lies near the local mean 2.0446, so step 100,001 has less
power than step 61,001. Its own neighbour's log, W(100,000) = 2.0171, is 0.0091 from W. The neighbour part of the
rule therefore depends on the neighbour batch landing below its own log. Pilot 0 measured that drop once: −0.0389
(`batch_60000/loss_check_sdpa.json`, pad-excluded mean against 2.0835855).

## Why the second step is 100,001

- No OLMo-3 checkpoint is in the S3 weight cache. `infra/pg/matrix_repo_facts.json` → `_meta.bucket_dirs` lists 22
  model directories and none is OLMo-3. A listing of `s3://pg-lineage-769063704696/weights/hf/hub/` on 2026-11-05
  shows no OLMo-3 directory either. So every candidate costs one in-job download of 14,596,063,712 B of
  safetensors (HF API, `?blobs=true`, for each of the three revisions used).
- With no cached option, the brief's named preference stands: `stage1-step100000` is grid point T_10 of the
  study (`../00_ASSETS.md` line 41). Its HF sha `1d074dbc15d91544abe47d5508ac1670ef8c7129` matches the grid
  manifest's `1d074dbc…`. Step 100,001 lies inside run `zckrmllo`, which covers steps 39,520–168,989
  (`../00b_DATA_ORDER.md` line 245). No other grid point after 60,000 lies inside that run.
- The three revisions have identical `config.json` files (`stage1-step60000`, `stage1-step61000`,
  `stage1-step100000`; compared on 2026-11-05, files in `$SP/j8/hf/`).

## Offline gate (no money), 2026-11-05

### Reconstruction

Command, once per pair (one input directory per trainer state, because cached files keep fixed names):

```
cd $SP/j8 && /usr/bin/time -l $SP/venvs/np1/bin/python -I studies/next/pilot0/reconstruct_batch.py \
    --in-dir $SP/j8/in_61000 --state-step 61000 61000 61001
cd $SP/j8 && /usr/bin/time -l $SP/venvs/np1/bin/python -I studies/next/pilot0/reconstruct_batch.py \
    --in-dir $SP/j8/in_100000 --state-step 100000 100000 100001
```

Python 3.12.12, NumPy 1.26.4. Logs: `batch_61001/reconstruct_61000_61001.log` (59.3 s real, peak RSS
3,032,973,312 B) and `batch_100001/reconstruct_100000_100001.log` (58.6 s real, peak RSS 3,035,054,080 B).

The one change to `reconstruct_batch.py`: `--state-step A` reads `config.json`, `data_paths.txt` and
`train/rank0.pt` from `checkpoints/OLMo25/step<A>/` instead of always `step60000/`, and asserts
`batches_processed` = A. Every other assert is unchanged.

| directory | trainer step | batch index | trainer state | its `batches_processed` | fingerprint equal | sha256 of token bytes | pad tokens | rows with pad | EOS | files |
|---|---|---|---|---|---|---|---|---|---|---|
| `batch_61000/` | 61,000 | 60,999 | `step61000` | 61,000 | yes | `400f4260…2d28` | 48,234 | 10 | 2,733 | 324 |
| `batch_61001/` | 61,001 | 61,000 | `step61000` | 61,000 | yes | `a67370a9…323d` | 45,904 | 8 | 2,682 | 318 |
| `batch_100000/` | 100,000 | 99,999 | `step100000` | 100,000 | yes | `d8ebbc0b…63de` | 23,840 | 6 | 2,825 | 327 |
| `batch_100001/` | 100,001 | 100,000 | `step100000` | 100,000 | yes | `86d5c16f…a561` | 38,820 | 7 | 2,808 | 311 |

Fields: `meta.json` → `trainer_state`, `trainer_state_data_loader`, `fingerprint_recomputed`,
`fingerprint_trainer_state`, `token_bytes_sha256`, `pad_tokens_total`, `instances_with_pad`, `eos_total`,
`distinct_files`. The fingerprint is `ddc8c024ea19a0b8f2bbf62c4fd90ab2dd0231d379f8d4e1fb0f14de8ddb8a27` in both
trainer states, equal to the value recomputed from the 1,016 public sizes and to pilot 0's. Its two failing inputs
(one size + 4 bytes; dtype written `uint32`) changed the hash in both runs (asserts in the script).

An independent read of the files (`$SP/j8/scripts/verify_batches.py`): each `tokens.npy` is uint32, shape
(512, 8192), 4,194,304 tokens = the global batch of 4,194,304 tokens, 16,777,344 B on disk; its sha256 equals
`meta.json`; the largest id is 100,277; `instances.jsonl` has 512 distinct instances. No instance is shared by any
two of the six batches 60,000, 60,001, 61,000, 61,001, 100,000, 100,001. Rows that are 8,192 pad tokens:
1 in `batch_61000/`, 3 in `batch_61001/`, 2 in `batch_100000/`, 2 in `batch_100001/`.

### Known number

The plan's J8 card asks that the first 64 rows of `batch_60001/` reproduce the banked `per_row_mean_ce` within
1e-4. This run scores the first 32 rows, last in the job, so it cannot take time from the registered batches
under the 1.0 GPU-hour ceiling (see the launch plan). Gate: max over rows 0–31 of |new − banked| ≤ 1e-4, banked
file `batch_60001/loss_check_sdpa.json`. Code: `$SP/j8/scripts/known_gate.py`. Output:
`$SP/j8/known_gate_failing_input.txt`.

| input | max \|diff\| over rows 0–31 | outcome |
|---|---|---|
| failing input: banked eager file `batch_60001/loss_check_eager.json` as the new run | 7.39e-3 (row 29, the row with 3,795 pad tokens) | fails |
| identity: the banked sdpa file against itself | 0 | holds |

### Job module and `loss_check.py`

Changes to `loss_check.py`: `--rows N` scores the first N rows; per row, the JSON now holds
`per_row_nonpad_target_positions`, `per_row_ce_sum_excluding_pad_targets` and
`per_row_mean_ce_excluding_pad_targets` (empty entry for a row whose 8,191 targets are all pad); the `pad.handling`
string no longer says the training masked nothing.

Changes to `infra/aws_pg/jobs/next_pilot0_loss.py`: the five runs below, one revision per run; checkpoints
downloaded one at a time in one background thread in order of first use, so the second and third downloads overlap
GPU runs; a failed download raises in the main thread; `metrics.json` carries `updated_utc` and, per run, the
pad-excluded mean so far and the per-row pad-excluded loss, pad-target count and non-pad target count, copied from
the run file at each 10 s rewrite inside the polling loop.

Local gate (`$SP/j8/gate/gate.py`, output `$SP/j8/gate/gate_output.txt`): `run_all` and `read_progress` of the
job module ran all five runs against a stub child that executes the real `write()` of `loss_check.py` on synthetic
per-row sums (no torch, no model). The batch pad-excluded mean in `metrics.json` equals the hand value
2.3384752059441123 to 1e-12; the per-row entries for a no-pad row, a half-pad row and an all-pad row are 2.0, 3.0
and empty. Failing inputs: a child that exits 3 makes `run_all` raise; a download that raises makes `run_all`
raise before any run starts.

## Prediction, written before the launch

- S − W about +0.005 at both steps, the pilot-0 offset (+0.0055 at step 60,001). The cause of that offset was not
  measured, so the sign and size may differ at other checkpoints.
- N below its own log by about 0.04, as in pilot 0: N about 2.03 for step 61,001 (dN about 0.09) and about 1.98
  for step 100,001 (dN about 0.05).
- Both steps in the pass branch.

## Launch plan

Runs, in order, all sdpa, one g6e.2xlarge:

| # | batch | checkpoint | rows | output |
|---|---|---|---|---|
| 1 | `batch_61001/` (target) | `stage1-step61000` | 512 | `out/batch_61001/loss_check_sdpa.json` |
| 2 | `batch_61000/` (neighbour) | `stage1-step61000` | 512 | `out/batch_61000/loss_check_sdpa.json` |
| 3 | `batch_100001/` (target) | `stage1-step100000` | 512 | `out/batch_100001/loss_check_sdpa.json` |
| 4 | `batch_100000/` (neighbour) | `stage1-step100000` | 512 | `out/batch_100000/loss_check_sdpa.json` |
| 5 | `batch_60001/` (known number) | `stage1-step60000` | 32 | `out/batch_60001/loss_check_sdpa_rows32.json` |

Every run is compared with W of its step: runs 1–2 with 2.1173441410064697, runs 3–4 with 2.0262222290039062,
run 5 with 2.1240015029907227 (field `target_wandb_ce`, `diff`).

Time estimate from pilot 0 (`GPU_CHECK.md` lines 120–131; `$SP/pilot0_a2/out/metrics.json`): boot and setup about
190 s; first download 75–102 s; first run 760 s (it includes about 100 s of first-start cost); later 512-row runs
about 640–670 s; run 5 about 65 s. Job time about 2,870 s (47.8 min); instance time about 51 min.

Ceiling: `--budget-min 49`. `run_job.sh` kills the job 49 min after it starts, and the launcher's reaper terminates
the instance 49 + 10 = 59 min after launch, so the instance cannot outlive 1.0 h. If the budget cuts run 4, it
keeps the rows written every 16 rows; 400 rows of run 4 need about 500 s of its 645 s.

### Note written 2026-11-05T15:58Z, during the run, before any loss value was read

The first checkpoint download is slower than in pilot 0. The heartbeat (`heartbeat.jsonl` at the prefix) shows
about 6.9 GB written in the 196 s after 15:52:33Z, 22–55 MB/s per 30 s sample. Pilot 0 downloaded 14.6 GB in 75–102 s.
The job process started by 15:51:29Z, so the 49-minute `timeout` ends it at about 16:40Z. Projection: download 1
ends about 15:59:30Z; runs 1–3 end about 16:32–16:35Z; run 4 (`batch_100000/`, the neighbour of step 100,001)
gets about 250–350 of its 512 rows; run 5 (the known number) does not start. Under the frozen rule, step 100,001
would then be in the does not count branch (fewer than 400 rows). The instance is not SSM-managed
(`aws ssm describe-instance-information` returns no entry), so the budget cannot be changed in flight. A second
instance for runs 4–5 would add about 0.4 GPU-hours and take J8 over its 1.0 ceiling, so none is launched.

### Deviation, 2026-11-05 (16:20Z): extension decided, not applied

The head decided at about 16:18Z to extend the self-termination of `i-03e36825423328376` by 15 minutes (J8 ceiling
1.0 → 1.25 GPU-hours, an operational deviation; the measurement rule and its 400-row minimum unchanged), only if
the launcher can change the reaper or budget of a running job without a relaunch. It cannot, so nothing was
changed:
- The binding limit is on the instance. The user-data passes budget `2940` to `run_job.sh`, which runs the job
  under `timeout --signal=KILL 2940s` (`aws ec2 describe-instance-attribute --attribute userData`). When it fires, the
  exit trap writes STATUS and runs `shutdown -h now`.
- No `pg` command changes it: `watchdog` and `reconcile` only classify jobs, and `register` only writes the S3
  registry record. The instance has no key pair (`KeyName` None) and no SSM registration, so there is no way to it.
- The reaper `pg-reap-i-03e36825423328376`, `at(2026-11-05T16:49:09)`, could be moved with the scheduler API. That
  alone would not keep the job alive past the in-instance timeout, so it was left as it is.

Kill times are unchanged: the job ends at about 16:40Z (2,940 s after the job process started, by 15:51:29Z per
the heartbeat), and the reaper is at 16:49:09Z. Neither the job nor its code was touched. If run 4 ends with fewer
than 400 rows, step 100,001 does not count and is rerun later under the same rule (head's decision).

## Rerun b2, 2026-11-05

### Before the launch (written 2026-11-05T18:37:54Z)

Why: run 4 of b1 (`batch_100000/`, the neighbour of step 100,001) stopped at 320 of 512 rows, so step 100,001 is
in the does not count branch. Run 5 (the known number) never started. Head's decision, 2026-11-05 10:55 CST: a step
under 400 rows does not count and is rerun under the same rule. The rule above is unchanged. The target of step
100,001 is not rerun: S = 2.0355981 on 512 rows stays from b1 (`batch_100001/loss_check_sdpa.json`).

Runs, in order, one g6e.2xlarge, sdpa, `--runs 4,5` of `infra/aws_pg/jobs/next_pilot0_loss.py`:

| # | batch | checkpoint | rows | output |
|---|---|---|---|---|
| 4 | `batch_100000/` (neighbour of 100,001) | `stage1-step100000` | 512 | `out/batch_100000/loss_check_sdpa.json` |
| 5 | `batch_60001/` (known number) | `stage1-step60000` | 32 | `out/batch_60001/loss_check_sdpa_rows32.json` |

Inputs: the same `loss_check.py` as b1 (md5 `02d2ebdd…`, equal to the b1 upload's ETag). `tokens.npy` of both
batches: uint32 (512, 8192), sha256 of the token bytes equal to `meta.json` → `token_bytes_sha256`.

Prediction:
- N (`batch_100000/`, 512 rows, pad targets excluded) lands below W = 2.0262222 by about 0.04 to 0.09, so N in
  about [1.936, 1.986]. Its 320-row partial in b1 was 1.9455010. If so, dN ≈ 0.04–0.09 > dS = 0.0093759, and step
  100,001 lands in the pass branch.
- The 32 known-number rows of `batch_60001/` match the banked pilot-0 `per_row_mean_ce`
  (`batch_60001/loss_check_sdpa.json`) within 0.001 nats per row. In pilot 0, eager and sdpa differed by 0.0001 in
  the mean. The registered gate stays the tighter one written above: max |diff| over rows 0–31 ≤ 1e-4
  (`$SP/j8/scripts/known_gate.py`).

Budget, sized on today's HF download rate (25 to 42 MB/s; 14.6 GB per checkpoint, 350 to 585 s each), not on
pilot 0: setup about 190 s; download 1 blocking (350–585 s); run 4 about 645–730 s (b1 run 1, the first run on its
instance, took 730.2 s); download 2 in the background during run 4; run 5 about 65 s. Job about 1,100–1,400 s.
`--budget-min 40`: the job is killed 2,400 s after it starts and the reaper terminates the instance 50 min after
launch. Ceiling of this rerun: 0.85 GPU-hours. J8 as a whole then reaches at most 1.70 GPU-hours against the
registered 1.0. That is an operational deviation, decided by the head on 2026-11-05 at 10:55 CST. The measurement
rule is unchanged.

File names, for when the rerun lands: b1's 320-row neighbour file is renamed
`batch_100000/loss_check_sdpa_b1_320rows.json` (content unchanged: `rows_done` 320, `complete` false, pad-excluded
1.9455010), and the 512-row file goes in at `batch_100000/loss_check_sdpa.json`. b2 produced no file, so the rename
was undone. The b1 file is still at `batch_100000/loss_check_sdpa.json`, the name the b1 section uses.

### Launch attempts: no instance (2026-11-05, 18:38–19:13Z)

**b2 did not run.** All nine launch attempts failed. Each attempt tried g6e.2xlarge in every subnet and then in an
EC2-chosen AZ, and the last error was `InsufficientInstanceCapacity`. No instance started, no reaper was armed and
no job was registered. So b2 spent 0 GPU-hours, and step 100,001 stays in the does not count branch from b1. Run 5,
the known number, has still not run. J8 stands at 0.851 GPU-hours.

| attempt | start (UTC) | end (UTC) | rc | last error |
|---|---|---|---|---|
| 1 | 18:38:23 | 18:39:32 | 1 | `InsufficientInstanceCapacity` |
| 2 | 18:42:49 | 18:44:06 | 1 | `InsufficientInstanceCapacity` |
| 3 | 18:47:06 | 18:48:14 | 1 | `InsufficientInstanceCapacity` |
| 4 | 18:51:14 | 18:52:26 | 1 | `InsufficientInstanceCapacity` |
| 5 | 18:55:26 | 18:56:27 | 1 | `InsufficientInstanceCapacity` |
| 6 | 18:59:27 | 19:00:44 | 1 | `InsufficientInstanceCapacity` |
| 7 | 19:03:44 | 19:04:49 | 1 | `InsufficientInstanceCapacity` |
| 8 | 19:07:49 | 19:09:02 | 1 | `InsufficientInstanceCapacity` |
| 9 | 19:12:02 | 19:13:08 | 1 | `InsufficientInstanceCapacity` |

Log: `$SP/j8/b2/launch_attempts.log`. Checks at 19:13:15Z:
- No instance carries the tag `s3prefix=results/next_pilot0/2026-11-05_b2` (`describe-instances`).
- No registry record names the prefix.
- The prefix holds only the five input objects uploaded at 18:38Z.

Nothing in this section is a loss value. The prediction above has not been tested.

Ready for the next attempt:
- The inputs are on S3 under `results/next_pilot0/2026-11-05_b2/inputs/`. Under the fresh-prefix rule, the next
  attempt copies them to a new prefix first.
- `$SP/j8/b2/recompute.py` recounts N from the per-row fields and `tokens.npy`, applies the frozen `branch()` of
  `$SP/j8/scripts/rule.py`, and computes the known-number gate. Its test inputs, run on banked files, are in
  `$SP/j8/b2/test/test_output.txt`:
  - b1's 320-row file gives N = 1.9455010 and the does not count branch (rows).
  - eager rows as the new known-number run make the gate fail (7.39e-3).
  - the identity pair gives 0, and the gate holds.
  - a wrong `tokens.npy` raises.
- `$SP/j8/b2/poll.sh` prints instance state, STATUS presence and the progress fields of `metrics.json`, with no
  loss values.

### Rerun b2 result, 2026-11-05 (written 2026-11-05T22:05Z, after the run)

**Step 100,001 is in the pass branch.** `batch_100000/` on all 512 rows under `stage1-step100000` gives
N = 1.9779166 nats per token with pad targets excluded. The target from b1 is S = 2.0355981 on 512 rows. Against
W = 2.0262222: |S − W| = 0.0093759 ≤ 0.0124, and |N − W| = 0.0483056 > |S − W|. **The known number holds.** The 32
rows of `batch_60001/` reproduce the banked pilot-0 `per_row_mean_ce` exactly: max |diff| 0.00e+00 against the bound
1e-4. The branch is the output of the frozen `branch()` (`scripts/rule.py`, a byte-for-byte copy of
`$SP/j8/scripts/rule.py`), called by `scripts/recompute.py` at 21:55:12Z. Output: `scripts/recompute_b2_output.txt`.
The verdict is Fable's.

#### Step 100,001 under the frozen rule

Units: nats per token, pad targets excluded, token-weighted.

| s | checkpoint | S (target) | rows | N (neighbour) | rows | W (log at s) | S − W | N − W | branch |
|---|---|---|---|---|---|---|---|---|---|
| 100,001 | `stage1-step100000` | 2.0355981 (b1) | 512 | 1.9779166 (b2) | 512 | 2.0262222 | +0.0093759 | −0.0483056 | pass |

Both readings of "also" give the same branch, because dN = 0.0483 > 0.0124.

N two ways:
- file field `pad.mean_ce_excluding_pad_targets`: 1.9779166304530469.
- recount: the sum of `per_row_ce_sum_excluding_pad_targets` over 512 rows, divided by the sum of
  `per_row_nonpad_target_positions` (4,169,956): 1.9779166. |diff| 0.0e+00.

The per-row non-pad target counts, and the pad-target total 23,836, equal the counts taken from
`batch_100000/tokens.npy`. Those are asserts in `recompute.py`.

The file field was first printed before 21:50Z, in a listing of the file's fields, before `recompute.py` ran. The
rule, its code, S and W were fixed before the launch and did not change.

#### Files

| quantity | repo file | S3 key under `s3://pg-lineage-769063704696/` | sha256 |
|---|---|---|---|
| N | `batch_100000/loss_check_sdpa.json` (512 rows, `complete` true) | `results/next_pilot0/2026-11-05_b2/out/batch_100000/loss_check_sdpa.json` | `3d927cec…7896` |
| known number | `batch_60001/loss_check_sdpa_rows32.json` (32 rows) | `results/next_pilot0/2026-11-05_b2/out/batch_60001/loss_check_sdpa_rows32.json` | `6bf0a4e8…5726` |
| job metrics | `batch_100000/b2_metrics.json` | `results/next_pilot0/2026-11-05_b2/out/metrics.json` | `271ac5b1…7b2e` |
| S | `batch_100001/loss_check_sdpa.json` (b1, unchanged) | `results/next_pilot0/2026-11-05_b1/` | — |
| W | `batch_100001/wandb_zckrmllo_steps_99000_101002.json`, `_step` 100,001 | — | — |

Each repo file is the S3 object, downloaded again before 21:51Z. Its sha256 equals that of the earlier scratchpad
copy. b1's 320-row file is renamed `batch_100000/loss_check_sdpa_b1_320rows.json` (`git mv`; sha256
`27dcab74…7b90`, unchanged; `rows_done` 320, `complete` false, pad-excluded 1.9455010). The b1 sections above call
it `batch_100000/loss_check_sdpa.json`. That name now holds the 512-row b2 file.

`token_bytes_sha256` in both new files equals `meta.json` of the same batch: `d8ebbc0b…63de` (`batch_100000/`) and
`c389cebe…b80e` (`batch_60001/`). `loss_check.py` asserts the tokens' hash against `meta.json` before it scores. The
two `meta.json` files under `results/next_pilot0/2026-11-05_b2/inputs/` equal the repo's byte for byte. The S3 ETags
of the two `tokens.npy` inputs equal the 8 MiB multipart MD5 of the repo files.

#### Three readings, `batch_100000/`

| file | rows | pad targets excluded | every position | rows without pad (n) | pad targets | top-1 |
|---|---|---|---|---|---|---|
| `batch_100000/loss_check_sdpa.json` (b2) | 512 | 1.9779166 | 2.0627823 | 1.9806015 (506) | 23,836 | 0.56114 |
| `batch_100000/loss_check_sdpa_b1_320rows.json` (b1) | 320 | 1.9455010 | 2.0572999 | 1.9486520 (316) | 18,625 | 0.56798 |

The full batch lies 0.0324 above b1's 320-row partial. The partial was not the logged quantity, as the b1 section
said. The standard error of the batch mean over the 506 rows without pad is 0.0240 (sd of `per_row_mean_ce` ÷ √n).

Spread of the per-row loss, pad targets excluded (`per_row_mean_ce_excluding_pad_targets`), b2, 512 rows: [0, 0.5) 7;
[0.5, 1.0) 28; [1.0, 1.5) 48; [1.5, 2.0) 145; [2.0, 2.5) 215; [2.5, 3.0) 65; [3.0, 4.0) 2; ≥ 4.0 0; no non-pad
target 2. The largest row is row 271, 3.9870, the OCR row already listed in the b1 table above. Output:
`scripts/recompute_b2_output.txt`, last block.

The first 320 rows of b2 are bit-identical to b1's 320 rows. `per_row_mean_ce`, `per_row_ce_sum_excluding_pad_targets`
and `row0_first32` are all equal, and the max |diff| of `per_row_mean_ce` is 0 over 320 rows (`scripts/known_gate.py`,
the last `known_gate.py` block of `scripts/recompute_b2_output.txt`). This comparison is exploratory, not a registered gate.

Banked samples, `batch_100000/loss_check_sdpa.json`, row 0 (`row0_first32`):

| position | last context token | target, p(target) | top-1, p(top-1) | scored |
|---|---|---|---|---|
| 8 | `Ġused` | `Ġhis`, 0.2942 | `Ġhis`, 0.2942 | right |
| 0 | `146` | `0`, 0.0447 | `.`, 0.0737 | wrong |

#### Known-number gate

The registered gate is max over rows 0–31 of |new − banked| ≤ 1e-4, on the field `per_row_mean_ce`
(`scripts/known_gate.py`). That field counts every position, pad targets included. New:
`batch_60001/loss_check_sdpa_rows32.json` (b2, `stage1-step60000`, sdpa). Banked: `batch_60001/loss_check_sdpa.json`
(pilot 0).

| field | rows | max \|diff\| | outcome |
|---|---|---|---|
| `per_row_mean_ce` (the registered gate) | 32 | 0.00e+00 | holds |
| pad-excluded row mean | 31 rows with no pad target | 0.00e+00 | beside the gate |

All 32 values are bit-identical, including row 29 (3,794 pad targets, 11.1903605). `per_row_top1` and
`row0_first32` are identical too. The banked pilot-0 file has no per-row pad-excluded field. On a row with no pad
target, the pad-excluded row mean is the same number as `per_row_mean_ce`, so the pad-excluded comparison exists only
on the 31 rows without pad. Row 29 has no banked pad-excluded value (b2 gives 2.4833021 over 4,397 non-pad targets).
`recompute.py` and `known_gate.py` give the same result. The prediction (within 0.001 per row) also holds.

The gate's failing input and identity pair were rerun from the repo copies (`scripts/test_recompute.sh`, output
`scripts/test_output.txt`). Eager rows as the new run fail at 7.39e-3 (row 29). The identity pair gives 0 and holds.
A wrong `tokens.npy` raises. b1's 320-row file gives N = 1.9455010 and the does not count branch (rows). The numbers
equal the scratchpad run's.
`scripts/rule.py` and `scripts/known_gate.py` are unchanged copies of the scratchpad files. `scripts/recompute.py`
differs from `$SP/j8/b2/recompute.py` in three places: it loads `rule.py` from its own directory, it prints the
pad-excluded comparison beside the gate, and it prints the stack fields of each file.

#### Stack on both sides

| file | GPU | instance | torch | transformers | parameters | attention |
|---|---|---|---|---|---|---|
| b2: `batch_100000/loss_check_sdpa.json`, `batch_60001/loss_check_sdpa_rows32.json` | NVIDIA L40S | g6e.4xlarge | 2.12.0+cu130 | 5.9.0 | bfloat16 | sdpa |
| b1: `batch_100001/loss_check_sdpa.json` (S) | NVIDIA L40S | g6e.2xlarge | 2.12.0+cu130 | 5.9.0 | bfloat16 | sdpa |
| pilot 0: `batch_60001/loss_check_sdpa.json` (banked known number) | NVIDIA L40S | g6e.2xlarge | 2.12.0+cu130 | 5.9.0 | bfloat16 | sdpa |

Fields `gpu`, `instance_type`, `torch`, `transformers`, `dtype.parameters`, `attn_implementation`. Both instance types
carry one L40S; g6e.4xlarge has more vCPU and host memory. On these inputs the change of instance type did not
change the forward: the known number and b1's 320 rows came out bit-identical. So S (b1, g6e.2xlarge) and N
(b2, g6e.4xlarge) are compared across instance types with no measured difference from the hardware.

#### Against the prediction (written 18:37:54Z, before the launch)

| quantity | predicted | measured |
|---|---|---|
| N, 512 rows | about [1.936, 1.986] | 1.9779166, inside |
| dN | about 0.04–0.09 | 0.0483056 |
| step 100,001 | pass branch | pass branch |
| known number, per row | within 0.001 (gate 1e-4) | 0 on all 32 rows |

The first prediction, written before b1 ("N about 1.98 for step 100,001, dN about 0.05"), also landed.

#### HF above the log: every residual so far

A target batch is scored at the checkpoint before the update that trains on it. So HF − log on a target is the gap
between the HF forward and the training forward on the same weights. A neighbour batch is scored at the checkpoint
right after the update that trained on it. `stage1-step100000` holds the weights after update 100,000, and that update
consumed `batch_100000/`. W(100,000) was logged by the same step, before its update.

| batch | role | checkpoint | HF, pad targets excluded | its own log | HF − log | from |
|---|---|---|---|---|---|---|
| `batch_60001/` | target of 60,001 | `stage1-step60000` | 2.1294733 | W(60,001) 2.1240015 | +0.0055 | pilot 0 |
| `batch_61001/` | target of 61,001 | `stage1-step61000` | 2.1252322 | W(61,001) 2.1173441 | +0.0079 | b1 |
| `batch_100001/` | target of 100,001 | `stage1-step100000` | 2.0355981 | W(100,001) 2.0262222 | +0.0094 | b1 |
| `batch_60000/` | neighbour of 60,001 | `stage1-step60000` | 2.0447261 | W(60,000) 2.0835855 | −0.0389 | pilot 0 |
| `batch_61000/` | neighbour of 61,001 | `stage1-step61000` | 2.0350325 | W(61,000) 2.0699108 | −0.0349 | b1 |
| `batch_100000/` | neighbour of 100,001 | `stage1-step100000` | 1.9779166 | W(100,000) 2.0171299 | **−0.0392** | b2 |

W(100,000) = 2.017129898071289 (`batch_100001/wandb_zckrmllo_steps_99000_101002.json`, `_step` 100,000); W(61,000)
from `batch_61001/wandb_zckrmllo_steps_60999_61002.json`; W(60,000) from `GPU_CHECK.md` line 28.

This run's residual is −0.0392: the neighbour lies 0.0392 below its own log. That is the effect of update 100,000,
which trained on this batch, on the loss of this batch. The HF-above-log gap of the same checkpoint is mixed into
it, and was +0.0094 on the target batch. These data do not separate the two. The three neighbour residuals lie
in [−0.0392, −0.0349]. The three target residuals are all positive, +0.0055 to +0.0094.

#### The second data-order check as a whole

| step | S, rows | N, rows | W | dS | dN | branch | from |
|---|---|---|---|---|---|---|---|
| 61,001 | 2.1252322, 512 | 2.0350325, 512 | 2.1173441 | 0.0078880 | 0.0823116 | pass | b1 |
| 100,001 | 2.0355981, 512 | 1.9779166, 512 | 2.0262222 | 0.0093759 | 0.0483056 | pass | S from b1, N from b2 |

Under the frozen rule, neither step is in the fail branch, and both are in the pass branch. The known-number gate
holds: max |diff| 0 on 32 rows, against 1e-4. The verdict is Fable's.

What stays open:
- The cause of the positive HF-above-log gap on the targets (+0.0055, +0.0079, +0.0094) was not measured. The
  largest leaves 0.0030 of room below the 0.0124 band edge.
- The two exclusion readings (whole pad rows dropped, pad targets dropped) differ by 0.0013–0.0017 at the targets.
  Both lie inside the band, so these data do not separate them.
- Step 100,001 has less power than step 61,001. 27.5 % of other steps' logs land within ±0.0124 of its W, against
  16.2 % at 61,001 (room check above).
- The known number covers 32 rows. The plan's J8 card asked for 64.
- No different model has read these artifacts yet.

J8 used at most 1.164 GPU-hours (b1 0.851, b2 at most 0.313), against the registered 1.0 and the deviation ceiling
1.70. Lifecycle and cost: `infra/aws_pg/operations/2026-11-05_next_j8_loss_check2_b2_run.md`.
