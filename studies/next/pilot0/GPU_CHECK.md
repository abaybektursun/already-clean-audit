# pilot0 GPU loss check

## Result (written 2026-11-05T08:45Z, after all three runs)

With pad targets excluded, `batch_60001/` forwarded through `stage1-step60000` gives 2.1294733 against the
log's 2.1240015: +0.0055, 0.22 per-step sd (`batch_60001/loss_check_sdpa.json` →
`pad.mean_ce_excluding_pad_targets`). With every position counted, which is the quantity the recipe registered,
it gives 2.2882038: +0.164, 6.6 sd (same file → `mean_ce`).

**Verdict: undecided.** Pad targets excluded is the reading registered for the verdict (the coordinator's
instruction), decided on 2026-11-05 after the data were seen. That makes it an exploratory deviation from the recipe. Under it, batch_60001 misses
the log by 0.0055: above the frozen tolerance of 0.003, below the frozen ceiling of 0.0124. 51 of the 999 other
steps (5.1 %) land that close to the log value. The neighbour control passes both of its conditions. The frozen
rule applied to the pre-registered all-position quantity gives **not confirmed** (+0.164 > 0.0124). The pad
positions account for that whole miss.

### Three means beside their targets

Each mean is next-token cross-entropy in nats per token over the 512 rows × 8,191 positions of one batch. The
three readings are: every position (`mean_ce`); the rows that hold no pad token (mean of `per_row_mean_ce`
over those rows); and every position whose target is not pad, token-weighted (`pad.mean_ce_excluding_pad_targets`).
Differences are mean − log value, and "sd" is the per-step sd 0.0248.

| file | attention | every position | rows without pad (n) | pad targets excluded | log value | diff, every position | diff, rows without pad | diff, pad targets excluded |
|---|---|---|---|---|---|---|---|---|
| `batch_60001/loss_check_eager.json` | eager | 2.2880968 | 2.1291234 (503) | 2.1294869 | 2.1240015 (step 60,001) | +0.1640953 = +6.62 sd | +0.0051219 = +0.21 sd | +0.0054853 = +0.22 sd |
| `batch_60001/loss_check_sdpa.json` | sdpa | 2.2882038 | 2.1291095 (503) | 2.1294733 | 2.1240015 (step 60,001) | +0.1642023 = +6.62 sd | +0.0051080 = +0.21 sd | +0.0054718 = +0.22 sd |
| `batch_60000/loss_check_sdpa.json` | sdpa | 2.1299317 | 2.0456267 (508) | 2.0447261 | 2.0835855 (step 60,000) | +0.0463462 = +1.87 sd | −0.0379588 = −1.53 sd | −0.0388594 = −1.57 sd |

`batch_60000/` against the step-60,001 value 2.1240015, which is the off-by-one test: every position +0.0059302
= +0.24 sd; rows without pad −0.0783748 = −3.16 sd; pad targets excluded −0.0792754 = −3.20 sd
(`batch_60000/loss_check_sdpa.json`).

### The frozen rule applied

Forward spread F = \|S1 − E\| on the same batch: 0.000107 with every position and 0.0000136 with pad targets
excluded (`batch_60001/loss_check_sdpa.json` minus `batch_60001/loss_check_eager.json`). 3 F is 0.00032 and
0.00004, both below 0.003, so tau = 0.003, the floor.

| condition | every position (registered) | pad targets excluded (exploratory) |
|---|---|---|
| tau ≤ 0.0124 | 0.003, holds | 0.003, holds |
| \|S1 − W1\| ≤ tau | 0.1642, fails; above 0.0124 | 0.0055, fails; at or below 0.0124 |
| \|S0 − W1\| > tau | 0.0059, holds | 0.0793, holds |
| S0 ≤ W0 + tau | 2.1299 > 2.0866, fails | 2.0447, holds |
| outcome | not confirmed | undecided |

The band fraction comes from `$SP/dataorder_in/wandb/zckrmllo_60001_61001.json`: 51 of the 999 steps
60,002–61,000 have a logged loss within ±0.00547 of 2.1240015.

### Spread across rows and the standard error of the batch mean

| file | rows | sd of `per_row_mean_ce` | standard error, sd / √n | in per-step sd |
|---|---|---|---|---|
| `batch_60001/loss_check_sdpa.json` | all 512 | 1.5391 | 0.0680 | 2.74 |
| `batch_60001/loss_check_sdpa.json` | 503 without pad | 0.5721 | 0.0255 | 1.03 |
| `batch_60000/loss_check_sdpa.json` | all 512 | 1.3399 | 0.0592 | 2.39 |
| `batch_60000/loss_check_sdpa.json` | 508 without pad | 0.5628 | 0.0250 | 1.01 |

Without the pad rows, the standard error of a 512-row mean equals the per-step sd of the log, 0.0248. So the
step-to-step spread in the log is about the spread from drawing 512 rows. A wrong batch would miss by about
0.025, while this one misses by 0.0055. The standard error describes other batches, not this one: on a fixed
batch and fixed weights the forward is deterministic. The +0.0055 is therefore an offset between the HF forward
and the training forward on the same tokens, and its cause was not measured.

### Pad tokens (id 100277)

`loss_check.py` counts pad tokens as inputs and as targets and passes no mask, because the README reads the
training code as masking nothing (`README.md`, "GPU loss check").

| quantity | `batch_60001/loss_check_sdpa.json` | `batch_60000/loss_check_sdpa.json` |
|---|---|---|
| positions | 4,193,792 | 4,193,792 |
| pad target positions | 41,863 (0.998 %) | 19,254 (0.459 %) |
| cross-entropy summed over pad targets | 754,829.0 | 396,703.7 |
| mean per pad target | 18.03 nats | 20.60 nats |
| rows holding pad (row: pad tokens in `tokens.npy`, row mean CE) | 29: 3,795, 11.19; 236: 4,185, 12.39; 251: 8,192, 21.28; 272: 6,717, 12.96; 337: 1,801, 3.26; 366: 5,634, 15.41; 391: 5,985, 16.41; 483: 2,821, 3.68; 492: 2,738, 4.04 | 194: 1,901, 6.56; 267: 8,192, 21.28; 370: 972, 2.22; 449: 8,192, 21.28 |

A row of 8,192 pad tokens costs 21.28 nats per token, above ln(100,278) = 11.52, so this model gives pad
almost no probability. Counted, the pad positions put both batches above their logs (+0.164 and +0.046).
Excluded, batch_60001 sits 0.0055 above its log and batch_60000 sits 0.039 below its own. A batch below its own
log is what the README expects of batch_60000 under the assumed weight naming: update 60,000 trained on it
(`README.md`, "Which directory the GPU check uses"). So the logged loss did not count pad targets. This
contradicts the README's reading of `get_labels`. Where the training code drops them was not traced. The two
exclusion readings, whole rows and pad targets, differ by 0.0004 on batch_60001, so these data cannot
separate them. The same pad-row finding was reached independently in `PAD_ANALYSIS.md`, from the copy of
`batch_60001/loss_check_sdpa.json`; its numbers agree with the tables here.

### Vocabulary

The HF model has 100,278 output classes. The training LM head had 100,352 (`loss_check*.json` → `vocab`).
The largest token id in both `tokens.npy` files is 100,277, so no target falls in the 74 extra classes. The
extra logits only add to the training softmax denominator. On the same weights, the training-head loss is
therefore at least the HF loss, by an amount that cannot be measured here: those 74 rows of the LM head are
not in the HF checkpoint. The term has the wrong sign to explain the HF value lying above the log.

### Eager against sdpa (`batch_60001/loss_check_eager.json`, `batch_60001/loss_check_sdpa.json`)

- Mean of the per-row differences, sdpa − eager: +0.000107. Their sd: 0.00128. Largest single row: 0.0185, on
  row 251, the all-pad row.
- Top-1 count differs on 457 of the 512 rows, by −93 tokens in total out of 4,193,792. Top-1 accuracy is
  0.53518 (sdpa) and 0.53520 (eager).

### Banked samples (row 0, field `row0_first32`)

`Ġ` marks a leading space in the tokenizer's byte-level alphabet.

| file | position | last context token | target, p(target) | top-1, p(top-1) | scored |
|---|---|---|---|---|---|
| `batch_60001/loss_check_sdpa.json` | 2 | `Ġhttps` | `://`, 0.9987 | `://`, 0.9987 | right |
| `batch_60001/loss_check_sdpa.json` | 1 | `:` | `Ġhttps`, 0.0030 | `Ġ`, 0.0706 | wrong |
| `batch_60000/loss_check_sdpa.json` | 1 | `ĠThis` | `Ġis`, 0.2222 | `Ġis`, 0.2222 | right |
| `batch_60000/loss_check_sdpa.json` | 0 | `-` | `ĠThis`, 0.00028 | `1`, 0.0454 | wrong |

Row 0 of `batch_60001/` starts with the text `2: https://addons.mozilla.org/...` (`README.md`, instance 0).

### Runs, hardware, time

| file | instance | load s | forward s | s per row | peak GPU memory allocated |
|---|---|---|---|---|---|
| `batch_60001/loss_check_eager.json` | `i-0622a4d0c58663fbd`, g6e.2xlarge, NVIDIA L40S | 9.4 | 2,287.0 | 4.47 | 36.7 GB |
| `batch_60001/loss_check_sdpa.json` | `i-00e0cfc2d4d08716c`, g6e.2xlarge, NVIDIA L40S | 9.7 | 645.2 | 1.26 | 21.19 GB |
| `batch_60000/loss_check_sdpa.json` | `i-00e0cfc2d4d08716c`, g6e.2xlarge, NVIDIA L40S | 2.4 | 624.9 | 1.22 | 21.19 GB |

- Software: torch 2.12.0+cu130, transformers 5.9.0 (fields `torch` and `transformers` in each file).
- Dtype: bfloat16 parameters; logits cast to float32 for the cross-entropy; per-row sums in float64 (field
  `dtype`).
- Checkpoint: `allenai/Olmo-3-1025-7B` at `stage1-step60000`, 14,605,764,269 B, downloaded in-job in 102.4 s
  (attempt 1) and 75.2 s (attempt 2). These come from `out/metrics.json` under
  `s3://pg-lineage-769063704696/results/next_pilot0/2026-11-05_a1/` and `.../2026-11-05_a2/`.
- GPU time:
  - Attempt 1, `i-0622a4d0c58663fbd`: launched 07:27:19Z. Termination was requested at 08:11:28Z, once the
    complete eager file was on S3. 0.736 h.
  - Attempt 2, `i-00e0cfc2d4d08716c`: launched 08:06:43Z. The job exited with `exit=0` and the instance shut
    itself down: STATUS uploaded 08:34:33Z, and the state was first seen as shutting-down at 08:34:56Z.
    At most 0.470 h.
  - Total: at most 1.21 GPU-h (EC2 `describe-instances` launch times and state-transition reasons).

### Deviations from the plan

1. The plan was one instance under 1 GPU-h. The eager pass ran at 4.47 s per row (0.64 GPU-h on its own), while
   the recipe's estimate was 0.1–0.2 GPU-h per pass. The coordinator raised the limit to 1.5 GPU-h. Attempt 1
   was terminated after its eager file was complete on S3. Runs 2 and 3 ran on attempt 2 (`--runs 2,3`). Attempt
   2 was launched 4.7 minutes before the eager run ended, so its sdpa runs began about when eager finished. The
   two instances overlapped from 08:06:43Z to 08:11:28Z.
2. The README's command for `batch_60000/` passes `--target 2.1240015029907227`. This run passed that batch's
   own log value, 2.083585500717163, so the file's `diff` is against 2.0836. The off-by-one comparison against
   2.1240 is in the line under the first table.
3. Pad exclusion as the verdict reading was decided after the data were seen.
4. A partial eager mean (2.2046 over 224 rows) was seen by the coordinator before all three runs finished. The
   rule below had been written before launch.

### Changes to `loss_check.py` before the run

It had never run. These were the changes:
- `torch_dtype=` became `dtype=`, plus an assertion that the parameters are bfloat16. `torch_dtype` is the
  deprecated name in transformers 5.9.0.
- The model is loaded with `device_map="cuda"` instead of a CPU load followed by `.cuda()`.
- `use_cache=False` on every forward; the config sets `use_cache: true`, which built a KV cache of all 8,192
  positions on each row.
- The output file is rewritten atomically every 16 rows; before, it was written once after the loop.
- New assertions: the attention implementation requested is the one set; 100,278 classes and LM-head rows; the
  largest token id below 100,278; id 100277 decodes to `<|pad|>`.
- New records: pad counts, the pad cross-entropy sum and the mean with pad targets excluded; the vocabulary note;
  dtype; load and forward seconds; peak memory; instance type; the full model config; decoded token strings in
  the row-0 samples.

## Rule, written 2026-11-05T07:26Z, before any GPU run

Quantities. Each is the field `mean_ce` of one output file: the mean fp32 next-token cross-entropy over the 512 ×
8,191 positions, pad targets included.

| symbol | file | what it is |
|---|---|---|
| E | `batch_60001/loss_check_eager.json` | batch index 60,000, `stage1-step60000`, eager attention |
| S1 | `batch_60001/loss_check_sdpa.json` | the same batch and weights, sdpa attention |
| S0 | `batch_60000/loss_check_sdpa.json` | batch index 59,999 (the neighbour), sdpa attention |
| W1 | 2.1240015029907227 | W&B `train/CE loss` at step 60,001 (`README.md`) |
| W0 | 2.083585500717163 | W&B `train/CE loss` at step 60,000 (`README.md`) |
| sd | 0.0248 | per-step sd of `train/CE loss`, steps 60,001–61,000 (`README.md`; recomputed 0.024802) |

Tolerance. F = \|S1 − E\| is the spread between two forward passes on one batch. The tolerance is
tau = max(0.003, 3 F).
- Floor 0.003: 26 of the 999 other steps in 60,002–61,000 (2.6 %) have a W&B loss within ±0.003 of W1. A wrong
  batch therefore lands within ±0.003 of W1 with probability about 0.026. Recomputed from
  `$SP/dataorder_in/wandb/zckrmllo_60001_61001.json`; the recipe (`../00b_DATA_ORDER.md`, Step 8) gives 2.7 %
  including step 60,001 itself.
- Ceiling 0.0124 (half the per-step sd): 107 of the 999 other steps (10.7 %) lie within ±0.0124 of W1. Above it
  the check cannot separate this batch from a wrong one.

Verdict on the claim "the reconstructed order is right" (batch_60001 holds the batch logged at step 60,001):
- **confirmed**: all three runs complete, tau ≤ 0.0124, \|S1 − W1\| ≤ tau, \|S0 − W1\| > tau, and S0 ≤ W0 + tau.
  The last condition holds if update 60,000, which trained on batch index 59,999, did not raise that batch's loss.
- **not confirmed**: all three runs complete, tau ≤ 0.0124, and either \|S1 − W1\| > 0.0124, or
  \|S0 − W1\| ≤ tau while \|S1 − W1\| > tau (the neighbour matches step 60,001, so the order is off by one).
- **undecided**: every other case. That covers a run that did not complete, tau > 0.0124,
  tau < \|S1 − W1\| ≤ 0.0124, S0 > W0 + tau, or both batches within tau of W1.

The verdict is read once, after all three files exist, with this rule.

Gates and the input that makes each fail. `loss_check.py` asserts that the sha256 of the token bytes equals
`meta.json` → `token_bytes_sha256`. Before the run, the real bytes passed this check for both batches and a copy
with one token changed failed it (local NumPy check, 2026-11-05). The job module raises when a run exits non-zero
or ends without `"complete": true`. A stub child that exits 3 made it raise (local check, 2026-11-05).
