# M2 offline gate

**Update 2026-11-06, after `score-06`: the run completed on `m2-freeze-1g`** (52 gates pass, every failing input
fails). What this gate predicted for the points no attempt had reached held: G0's failing input u × 1.1 failed at all
11 points (min over layers 0.0943–0.0980); the swap fell under 0.01 at T_6 and T_5; G-EX and G-TOPK at T_6 and T_5 were
bit-identical, with the 4 G-TOPK entries per point counted offline. The verdict is in `M2_SPEC.md` §12.

**Update 2026-11-06, amendment A5 (head's decision 06:50Z, before freeze 1g): go for `score-06`, the last attempt.**
G0's failing input becomes u scaled by 1.1, which gives at least 0.089 on every layer wherever the gate passes; the
activation swap is reported, not gated. This gate now evaluates both sides of every gate the job runs at every point
from the banked files, and of the verdict's gates where the bank holds them. That closes the third gap (a failing input
with no power where it is applied). The dry run printed `M2_DRYRUN_OK` after 130 checks, 0 failed
(`studies/next/m2/offline_gate/dryrun_2026-11-06_A5.log`). The section "Amendment A5" near the end is the record.

**Update 2026-11-06, after `score-05`: M2 ended without a verdict.** `score-05` passed every gate through T_7 and wrote
the degenerate-pole counts this gate predicted, then stopped at T_6 because G0's failing input passed, a third gap in
this gate (section "After `score-05`").

**Update 2026-11-06, amendment A4 (head's decision 05:35Z, before freeze 1f): go for `score-05`, the last relaunch.**
A pole with no token of its own sign at a point gets B = 0 and ratio = 0, ranks last and is not a matched-pole
candidate. This gate now scans all 352 banked stats files and 352 topk files at T_5–T_15 (small members and .npy
headers only) and plants such a pole in the tiny world. The dry run printed `M2_DRYRUN_OK` after 112 checks, 0 failed
(`studies/next/m2/offline_gate/dryrun_2026-11-06_A4.log`). The section "Amendment A4" near the end is the record.

**Update 2026-11-06, after `score-04`: M2 ended without a verdict.** `score-04` passed every gate through T_10 and crashed in the scoring code there on a banked pole with no token of its sign, a case this gate never planted (section "After `score-04`").

**Update 2026-11-06, amendment A3 (head's decision 04:20Z, before freeze 1e): go for `score-04`, the last attempt.**
`score-03` stopped on G-EX at T_13 under A2 (591 read entries, p99 0.05191 against 0.038718) and does not count. A3
makes G-EX at the Blackwell-banked points T_7–T_14 a reported readout, not a gate; G-TOPK, K0, I-ID and G-EX at the
same-device points stay gates. The dry run printed `M2_DRYRUN_OK` after 89 checks, 0 failed
(`studies/next/m2/offline_gate/dryrun_2026-11-06_A3.log`). The section "Amendment A3" near the end is the record.

**Update 2026-11-06, amendment A2 (head's decision 03:45Z, before freeze 1d): go for `score-03`.** `score-02` stopped on
G-EX at T_13 (p99 0.05433 against 0.044273) and does not count. A2 gates the cross-device G-EX on the entries with
|a_bank| ≥ 0.1 only, at 0.038718 (twice the control pair's p99 on that subset). The dry run printed `M2_DRYRUN_OK`
after 86 checks, 0 failed (`studies/next/m2/offline_gate/dryrun_2026-11-06_A2.log`). The section "Amendment A2" near
the end is the record. The risk of another G-EX stop is high and not well estimated (`M2_SPEC.md` §3.8 "A2").

**Update 2026-11-05, the head's revision of A1 (20:50 CST, before freeze 1c): go for `score-02`.** The cross-device
tolerance is now twice the T_14 pair's p99 of d over the entries with |a_bank| < 0.5: 0.044273 (was 0.042318). The
same-device tier stays 0.021159. The dry run printed `M2_DRYRUN_OK` after 83 checks, 0 failed
(`studies/next/m2/offline_gate/dryrun_2026-11-05_A1rev.log`). The section "Revision of A1" near the end is the record.

**Update 2026-11-05, amendment A1 (before freeze 1b): go for `score-02`.** `score-01` stopped on G-EX at T_14 and does
not count. A1 changed the known-number statistic (`M2_SPEC.md` §3.8). The dry run printed `M2_DRYRUN_OK` after 83
checks, 0 failed (`studies/next/m2/offline_gate/dryrun_2026-11-05_A1.log`), including the pass side across devices
that this gate lacked. The section "Amendment A1" at the end is the record. The text up to it is the freeze-1 record.

**Go for the one GPU launch (`score-01`, 1.56–1.72 GPU-h projected on one L40S) once the head has reviewed and three
laptop steps are done.** The weight-free dry run printed `M2_DRYRUN_OK` after 78 checks, 0 failed
(`studies/next/m2/offline_gate/dryrun.log`, 2026-11-05). It ran the job from its entry point on a tiny model whose banks
were written by the banked harvest's own code, showed every registered failing input failing, and produced every
verdict branch. The three laptop steps are:
1. freeze 1 (commit and tag `m2-freeze-1`);
2. the `pg ps` / `pg audit` check;
3. the upload of the 17 input files (`M2_SPEC.md` §9.5 S1).

The pre-check passed: 153 births pass J1's recount, against the 100 required (`studies/next/m2/births.json`).

`SP` below means
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.
`BANK` means `/Users/abaybektursun/projects/interp_research_banks/neurons`.

## What was run

```
cd /Users/abaybektursun/projects/interp_research-m2
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m2 births   --out studies/next/m2/births.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m2 examples --births studies/next/m2/births.json --out studies/next/m2/samples
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m2 project  --births studies/next/m2/births.json > studies/next/m2/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --out $SP/m2/dry
```

- Environment: the scratchpad venv `m1_venv` (Python 3.12, CPU only, torch 2.12.0, transformers 5.9.0, numpy 1.26.4,
  safetensors, huggingface_hub, pyflakes), the one the M1 gate used. pyflakes is clean on the four new modules.
- Wall time of the dry run: 36 s. The log, with the progress bars removed, is `studies/next/m2/offline_gate/dryrun.log`;
  the check values are in `dryrun_summary.json` beside it.
- `$SP/m2/hf` holds three things: `tok/` (the OLMo 3 tokenizer files from the M1 asset reads, `$SP/m1_gate/hf/tok`),
  `config_stage1-step476838.json` (same source), and the 22 files committed at `studies/next/m2/offline_gate/hf/`.
- No weights were downloaded. The tiny models are random tensors made on the laptop.

## Fetched (small files only)

- S3, read-only:
  - the 16 banked grid `manifest.json` files (`results/neurons/w3/grid/{1,1b}/out/<dir>/manifest.json`, about 3 KB
    each), giving the revisions and devices in `m2.POINTS`;
  - `results/neurons/w3/device_check/t14_1b_vs_1c.json` (1,996 B, copy committed);
  - `aws s3 ls` listings: 32 topk and 32 stats files at each of T_5 … T_15, the corpus prefix, and an empty
    `results/next_m2/`;
  - two 4 KB range reads of the zip directories of `1b/out/t10_step100000/{stats,topk}/layer_10.npz`. They list
    `hist`, `hist_edges`, `n_pole_tokens`, `q999`, `max_abs`, `density` and `tok`, `a`, `g`, `u`.
- Hugging Face: `config.json` and `model.safetensors.index.json` of the 11 branches T_5 … T_15 (22 files, committed at
  `studies/next/m2/offline_gate/hf/`).
- Web: four searches and the wdd atlas 39 page (`M2_SPEC.md` §1.5).

## Results by section

Each line of the log is one `CHECK`. All passed.

### REAL: the laptop banks (21 checks)

- Inputs: `arrays.npz` and `births.jsonl` carry their registered sha256. The 16 `exemplar_a.npz` and 16 `contexts.jsonl`
  carry theirs. `arrays.npz` `a[t]` equals point t's `exemplar_a.npz` for 16 of 16 points.
- **The dating rule** (`m2.births_matrix`, a copy of the recount's) reproduces 2,454 of 2,454 banked dates of sample 0.
  **J1's failing input**: setting a birth's exemplars to zero at its birth point moves the date later for 153 of 153.
- **Pre-check**: 153 passing births (p_late ≤ 0.05, R1t), equal to `10_RECOUNT2.md` §8. Halves: freeze 82, test 71,
  both at or above N_MIN = 50. Every birth's half is its neuron parity.
- **Twins**: 149 of 149 have the birth's layer, another neuron, the same size and date k by the rule. Every exemplar
  position equals the carrier's T_15 top-50 position.
- `births.json` rebuilds byte-identical from the banks. Its sha256 `fdca794bc98a78ec…` is registered as
  `m2_score.BIRTHS_SHA`.
- Examples: 10 freeze ids (5 with A_own ≥ 1, 5 with A_own = 0) and 10 test ids.
- G-CFG passes on the 11 real branch configs with no differing key. Its failing input (model type renamed) fails on all
  11. The 11 indexes name exactly three bf16 shards.
- **K0 offline**: the 63 entries of the first 10 carriers at T_15 pass against the bank itself. T_14's banked values at
  the same entries fail with median relative difference 0.3884 and p99 0.9991, against a tolerance of 0.0043 and 0.0222.
- **G-EX offline**: at each of the 11 points the neighbouring point's banked values fail. Medians, as (point, entries,
  median): (15, 2295, 0.269), (14, 477, 0.335), (13, 717, 0.418), (12, 717, 0.372), (11, 643, 0.39), (10, 397, 0.354),
  (9, 243, 0.558), (8, 147, 0.669), (7, 116, 0.493), (6, 5, 0.506), (5, 10, 0.393).

### SCORE: the pure functions (10 checks)

- Percentile on a planted histogram:
  - a value at a bin's lower edge counts only the bins below;
  - a value halfway through a bin (log scale) counts half of it (0.125000);
  - a value above every token gives 1.
- Percentile against the empirical fraction on 200,000 log-normal draws: the largest difference is 0.00029, within one
  bin's share.
- Ranks with ties counted against: (3, 3, 1, 0.5) → (2, 2, 3, 4). Matched poles: K of them, the carrier's neuron
  excluded, nearest first.
- Overlap counts exemplar positions and ignores −1 slots. The ratio uses max(a, 0) for + and max(−a, 0) for −.
- Wilson 0 of 71: upper bound 0.0513. The paired interval excludes 0 for 30 of 71 against 0.

### JOB: the job from its entry point, tiny model, CPU (17 checks)

The tiny world:
- a random Olmo 3 with 4 layers, hidden 64 and MLP width 96, at four planted checkpoints (T_12–T_15);
- a 64-window corpus of 128 tokens per window with the real tokenizer;
- 40 grid poles;
- banks written by `neurons_harvest.run` itself (topk/, stats/, poles/exemplar_a.npz, T_15's positions for the
  earlier points);
- 40 synthetic births with k ∈ {13, 14, 15} and 32 twins;
- a local stand-in hub.

`m2_score.main` ran in a child process with a planted registry.
- It exited 0 with `M2_JOB_OK`, `complete` true, nothing parked.
- All 25 gates passed and all 25 registered failing inputs failed.
- **K0**: 10 poles, exact share 1.000, median 0. Its failing input, T_14's values, has median 0.465. G-EX, G-TOPK and
  I-ID were also bit-identical (exact share 1.000) although the windows ran in batches the harvest never formed.
- G-TOPK compared 1,595, 873, 657 and 343 banked top-50 entries at the four points.
- **I-ID**: 0 carrier flips. Its failing input (T_14's birth-role activations) has median 0.984.
- `metrics.json` carries no rank and no hit. There is one record per planned (unit, role), 184 in all, each in its
  unit's half.
- The construction control at T_15: every carrier is in the top 10 under A, with ranks 1, 2 and 3. A rank above 1 is a
  tie with a pole holding the same positions: the top-10 place is not rank 1 by construction. Under B the carriers are
  in the top 10 at 1.0. The opposite pole stays out.
- The activations are banked per (unit, role). The raw-example files hold the selected ids with top-10 lists and
  decoded contexts.
- A record recomputes exactly from the banked activations plus the grid arrays. That is the path the independent
  reader will take.

### FAIL: each instance gate on a planted input (10 checks)

Each run below is a child process that exited non-zero with the gate named in its last error line.
- G_SAME: transformers registered as 0.0.0.
- INPUTS: a zero births sha256.
- G_CORPUS: a wrong tokens sha256.
- t15_G_REV: a wrong revision.
- t15_G_CFG: width 97 in the registered config.
- K0: T_14's exemplar file in T_15's place (median 0.465).
- t14_G_EX: T_13's file in T_14's place (median 0.789). K0 passed first in this run.
- K0: every exemplar position shifted by +1 (median 0.881).

The stop rule, given a 0.6 s budget, parked T_14, T_13 and T_12 after T_15. The job still ended with `complete` true,
and the parked births have no pre record, so they are not usable.

### VERDICT: freeze and verdict (19 checks)

On planted records of the real half sizes (82 and 71):
- the satisfying case gives `H_A SUPPORTED; H_B SUPPORTED; H_AB LOSES; H_sel_B ABOVE_SELECTION`;
- the bar is the largest of chance and the two floors;
- A never hitting gives H_A REFUTED;
- A drawn like B gives TIES (interval [−0.18, +0.13]); A above B gives BEATS;
- twins drawn like the carriers give AT_SELECTION ([−0.23, +0.07]); twins above them give BELOW_SELECTION;
- half the twins missing gives NOT_RUN;
- matched poles hitting at 0.6 give REFUTED at a bar of 0.707;
- pre-role hits at 0.9 against birth-role hits at 0.2 give "DOES NOT COUNT until inspected".

Refusals and does-not-count branches:
- A construction control at 0.5 makes freeze refuse.
- A construction control failing on the test half gives DOES NOT COUNT, and so does its failing input passing.
- A failed I-ID gives DOES NOT COUNT: gates.
- A failing input that passed makes freeze refuse.
- 30 test births give DOES NOT COUNT: missing.
- A freeze record file changed after the freeze makes the verdict refuse.

The room check reaches SUPPORTED, REFUTED, BEATS, LOSES and TIES at n = 71. Finally, freeze and verdict ran on the tiny
job's own records, with N_MIN lowered to 5 for that world only.

### PROJECT (1 check)

The launch costs 1.72 GPU-h at the lowest banked rates (3,215 tokens per second, 143 MB/s) and 1.56 at the measured
T_15 rates (`studies/next/m2/projection.json`, 4,551 window forwards over 11 points). Both are within 0.9 of the job's
150-minute budget.

## The ten raw examples, zero-GPU part

Written to `studies/next/m2/samples/raw_examples_zero_gpu.md` (and `.json`) from the laptop banks, freeze half only.
- In all ten births the carrier at T_(k−1) is below its own threshold. Its median-to-q999 ratio runs from 0.07 to 0.96,
  and seven of ten sit in the [0.99, 0.999) percentile bracket.
- The five births with A_own = 0 are scored wrong under A for certain: rank 22,016.
- The five with A_own ≥ 1 rank first among their layer's 20 grid poles. Their rank among 22,016 is PENDING, because it
  needs S3 topk.
- The carriers' own top-5 contexts at T_(k−1) do not show the job.
- Each of the three matched grid poles per birth has A = 0 on the exemplars and reads as its own token family.
- In 7 of 9 births with a twin, the twin's ratio at T_(k−1) is higher than the carrier's.

The top-10 lists under A and B need the job. It writes them for the same ten births to `m2/samples/raw_examples_freeze.jsonl`.

## Spec issues (gaps in the brief, the plan and the critic; resolved in `M2_SPEC.md` as stated)

1. **"At the birth point itself the carrier ranks in the top 10 by construction"** (J9 card, brief). It does not. At T_k
   the carrier's median reaches its own q999, which says nothing about the other 22,015 poles. Even at T_15, where E_c
   is the carrier's own top-50, ties with poles holding the same positions push it to rank 2 or 3 (tiny world). M2
   therefore measures the birth-point value as the ceiling and puts the construction gate (≥ 0.90) at T_15.
2. **Three of the four candidate scores do not depend on the job.** Under a uniform pole draw they cannot beat chance
   by more than a few births (`M2_SPEC.md` §2.3). A is the only job-specific candidate.
3. **The reference needs GPU.** "The pole's own current percentile" on the job's contexts needs every pole's activation
   at E_c, which the banks hold only for the 640 grid poles at their own positions. So the measurement is not zero-GPU,
   though A alone is.
4. **Chance at 22,016 poles leaves the existence labels almost no way to fail.** One hit in 71 clears it. The spec says
   so and reads magnitude from the absolute precision and the two paired contrasts.
5. **Counts across documents.** The critic's M2 says 148 jobs and 11,008 poles per layer. The unit here is 153 (R1t over
   all 2,428 clusters; 150 under R1 on the 1,238 late_primary clusters) and 22,016 poles (both signs). The brief's
   chance level per sign, 0.00091, is reported beside 10/22,016 = 0.00045.
6. **The selection of E_c from the carrier's own future top-50** was not named as a confound in the plan or the critic's
   M2. M2 adds the pseudo-birth twins (§3.3) to bound it.

## Rerun after the head's decision (2026-11-05, before freeze 1)

The head chose B as the primary score and A as the reference (`M2_SPEC.md` §2.3). Two runs of the same command as
above, written to the scratchpad and copied here with the progress bars removed:

1. **The swap alone** (`m2.PRIMARY, m2.REFERENCE = "B", "A"`): 78 checks, 2 failed
   (`studies/next/m2/offline_gate/dryrun_2026-11-05_swap_only_fails.log`). The failures were "satisfying case: H_AB
   LOSES" (it read BEATS) and "A above B: H_AB BEATS". The cause: `m2.half_summary` computed the H_AB contrast as
   hit_PRIMARY − hit_REFERENCE, so the swap reversed its sign. The spec defines H_AB as hit_A − hit_B (§1.3, §5 step 7).
2. **After naming the scores in that line** (`paired(hits(pre, ids, "A"), hits(pre, ids, "B"))`): 78 checks, 0 failed,
   `M2_DRYRUN_OK` (`dryrun_2026-11-05_primary_B.log`, `dryrun_2026-11-05_primary_B_summary.json`). Against the first
   log, two lines differ, and only in the order of the score names: the bar dictionary lists B first, and the refused
   freeze names "construction control B" first. Every value is the same.

The job (`m2_score.py`) does not read `PRIMARY`; its identity pair names B explicitly. The record file writes the
top-10 lists and the matched ranks for both A and B, whichever is primary.

## After `score-01` (2026-11-06): a gap in this gate

The launch stopped at `t14_G_EX`: p99 relative difference 0.2563 against the cross-device tolerance 0.0444
(`studies/next/m2/JOURNAL.md`, 02:01:34Z). This gate tested only one side of G-EX at the Blackwell-banked points T_7–T_14:
the failing input (the neighbouring point's bank) fails. It never tested the pass side, a correct L40S forward against
a Blackwell bank on the entries G-EX actually reads. That test was possible offline. The one L40S repeat on S3 (T_14,
`results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz`, 221 KB) against the Blackwell bank, on the 477
entries of the plan at T_14, gives p99 0.2563, the job's value to every digit (`studies/next/m2/diag/score01_t14_gex.py`,
`score01_t14_gex.json`). The tolerance was calibrated on all 32,000 exemplar entries (median |a| 1.95); the entries G-EX
reads at T_14 are pre-birth and birth-point exemplars (median |a| 0.31, 2.5 % below 0.01). Any change to the gate is the
head's decision and needs a new freeze before a relaunch; it should come with this pass-side check, run offline, and
the failing input still failing.

## Amendment A1 (2026-11-05, before freeze 1b)

The head's decision of 2026-11-05 20:15 CST (`M2_SPEC.md` §3.8). Per entry the gates compare
d = |a_new − a_bank| / (0.05 + |a_bank|), and pass when the 99th percentile of d is at most the tolerance: 0.042318
across devices, 0.021159 on the same device type. The tolerance is frozen on the banked T_14 device pair.

What was run (the same environment as above):
```
aws s3 cp s3://pg-lineage-769063704696/results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz \
    $SP/m2b/s3_1c/exemplar_a.npz --profile fuelos --region us-east-1
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz --out $SP/m2b/dry
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/diag/a1_risk.py $SP/m2b/s3_1c/exemplar_a.npz \
    studies/next/m2/diag/a1_risk.json $SP/m2b/topk_t14/layer_02.npz $SP/m2b/topk_t14/layer_14.npz
```
- The repeat file fetched from S3 has sha256 `e9cdf8658615bc7a…`, equal to the copy the first executor used and
  registered as `m2.REPEAT_T14`. The two T_14 topk files (11 MB each, S3 `1b/out/t14_step1000000/topk/`) feed the
  exploratory G-TOPK magnitude check only.
- Dry run: 83 checks, 0 failed, wall time 35 s. Log with the progress bars removed:
  `offline_gate/dryrun_2026-11-05_A1.log`; values: `dryrun_2026-11-05_A1_summary.json`.
- pyflakes is clean on `m2.py`, `m2_score.py`, `m2_dryrun.py`.

The five new checks (REAL A1):

| check | value |
|---|---|
| the repeat has its registered sha256 | `e9cdf8658615bc7a` |
| the tolerance recomputes from the pair | p99 over all 32,000: 0.019737; over the 12,231 with \|a_bank\| < 1: 0.021159; same-device 0.021159, across devices 0.042318 |
| pass side across devices, all 32,000 | median 0.00412, p99 0.01974, max 0.3852: pass |
| pass side across devices, the plan's 477 entries at T_14 (score-01's) | 62 poles; median 0.00427, p99 0.04034, max 0.3852: pass |
| T_13's values in T_14's place | p99 11.157 (477 entries), 4.506 (32,000): fails |

Two existing checks changed their values. Both now pass (new, bank) in the job's order, because d is not symmetric:
- K0 offline, T_14's values at the 63 entries: median 0.3354, p99 3.7936, against 0.021159. Fails, as registered.
- G-EX offline, the neighbour's bank at the 11 points: p99 from 0.988 (T_5) to 14.177 (T_9). Fails at all 11.

In the tiny world (JOB, FAIL) every gate passed bit-identically and every failing input failed under the new statistic.
- K0's failing input: p99 0.085.
- G-EX's: p99 1.30–5.54.
- G-TOPK's: p99 0.95–1.11.
- I-ID's: p99 2.35.
- The planted K0, T_13-in-T_14 and shifted-position runs exited non-zero, naming their gate.

The tiny model's activations are small, so its K0 failing input (p99 0.085) has the least margin, four times the
tolerance. G-TOPK's and I-ID's failing inputs need all-neuron activations, which no bank holds; they are shown failing
only in the tiny world, and the job asserts them on the instance.

Exploratory, not a gate (`diag/a1_risk.json`). The chance that G-EX stops `score-02` at T_7–T_13 is about 0.32 under a
resampling model, which understates T_14. The G-TOPK entries at T_14 in layers 2 and 14 have \|a_bank\| ≥ 0.18, a
range where the pair's p99 is 0.021 or below.

## Revision of A1 (2026-11-05, before freeze 1c)

The head's decision of 2026-11-05 20:50 CST (`M2_SPEC.md` §3.8, "Revision of A1"). Reason: the A1 tolerance came from
the pair's entries with |a_bank| < 1, while G-EX reads pre-birth entries of median |a_bank| 0.31, and on score-01's 477
entries the pair passed by only 0.002. The cross-device tolerance is now twice the pair's p99 of d over |a_bank| < 0.5.

What was run (the same environment and command as A1, third executor; output dir `$SP/m2c/dry`):
```
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz --out $SP/m2c/dry
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/diag/a1_risk.py $SP/m2b/s3_1c/exemplar_a.npz \
    studies/next/m2/diag/a1_risk_rev.json $SP/m2b/topk_t14/layer_02.npz $SP/m2b/topk_t14/layer_14.npz
```
- The repeat file has sha256 `e9cdf8658615bc7a…` (`m2.REPEAT_T14`), checked again.
- Dry run: 83 checks, 0 failed, wall time 34 s (02:52:42Z to 02:53:16Z). Log with the progress bars removed:
  `offline_gate/dryrun_2026-11-05_A1rev.log`; values: `dryrun_2026-11-05_A1rev_summary.json`. Against
  `dryrun_2026-11-05_A1.log` only one line differs, the tolerance check.
- pyflakes is clean on `m2.py`, `m2_score.py`, `m2_dryrun.py`, `diag/a1_risk.py`.

| quantity | value |
|---|---|
| control p99 of d, all 32,000 / \|a_bank\| < 1 (12,231) / < 0.5 (7,424) / < 0.25 (2,726) | 0.019737 / 0.021159 / **0.022137** / 0.034947 |
| tolerance, same device type (unchanged) / across devices (new) | 0.021159 / **0.044273** (was 0.042318) |
| pass side, all 32,000, across devices | p99 0.01974: pass |
| pass side, the plan's 477 entries at T_14 (score-01's) | p99 0.04034 against 0.044273: pass by 0.0039 |
| K0's failing input (T_14's values, same-device tier) | p99 3.7936: fails (179 times) |
| G-EX's failing input (neighbour's bank) at the 11 points | p99 0.988–14.177: fails at all 11 (24 to 320 times) |
| T_13's values in T_14's place | p99 11.157 (477), 4.506 (32,000): fails |
| risk model: draws passing at all of T_7–T_13 | 0.7125 (was 0.679); T_13 0.156 and T_11 0.122 above the tolerance |

Said plainly: the revision moves the cross-device tolerance by 0.00196 (4.6 %). The < 0.5 subset still holds fewer
small entries than G-EX reads (4.9 % below |a_bank| = 0.1, against 14.9 % of the 477), so the room on the pass side
stays small. The same-device tier is unchanged; on one device type the forward was bit-identical in score-01, so its
pass side has the whole tolerance as room (`M2_SPEC.md` §3.8).

## Amendment A2 (2026-11-06, before freeze 1d)

The head's decision of 2026-11-06 03:45Z (`M2_SPEC.md` §3.8, "Amendment A2"). The cross-device G-EX reads only the
entries with |a_bank| ≥ 0.1; those below are reported, not gated. Tolerance: twice the T_14 pair's p99 of d on the
same subset. G-TOPK, K0, I-ID and the same-device G-EX are unchanged.

What was run (third executor; output dir `$SP/m2c/dry_a2`; 03:45:16Z to 03:45:59Z):
```
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz --out $SP/m2c/dry_a2
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/diag/a2_risk.py $SP/m2b/s3_1c/exemplar_a.npz studies/next/m2/diag/a2_risk.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/diag/risk_calibration.py $SP/m2b/s3_1c/exemplar_a.npz \
    studies/next/m2/diag/risk_calibration.json
```
- Dry run: 86 checks, 0 failed. Log with the progress bars removed: `offline_gate/dryrun_2026-11-06_A2.log`; values
  `dryrun_2026-11-06_A2_summary.json`. Against the freeze-1c log: three A1 checks moved to A2 (pass side on all
  entries, pass side on the plan's T_14 entries, T_13 in T_14's place) and three new checks (the A2 tolerance, the
  job's cross-device path in the tiny world, its failing input).
- The tiny world marks every point as made on one device, so before A2 the job's cross-device path never ran offline.
  The new JOB run marks T_14's bank as another device.
- pyflakes is clean on `m2.py`, `m2_score.py`, `m2_dryrun.py` and the three new diagnostics.

| quantity | value |
|---|---|
| control p99 of d, \|a_bank\| ≥ 0.1 (31,635) / < 0.1 (365, reported) | **0.019359** / 0.108214 |
| tolerance, cross-device G-EX (new) / G-TOPK across devices / same device | **0.038718** / 0.044273 / 0.021159 |
| pass side, all 32,000 (gated 31,635) | p99 0.01936: pass |
| pass side, the plan's 477 entries at T_14 (gated 406) | p99 0.02886: pass; 4 gated entries above the tolerance, a fifth would stop it |
| G-EX failing input (neighbour's bank) at T_7–T_14, gated | p99 1.046–14.301: fails (27 to 369 times) |
| T_13's values in T_14's place | p99 7.516 (354 gated), 2.541 (30,661 gated): fails |
| K0's failing input (unchanged) | p99 3.7936: fails |
| tiny world, cross-device path at T_14 | 12 of 157 gated, pass; failing input p99 1.393; T_13 in T_14's place stops the job (p99 0.821) |
| risk, bin model restricted to ≥ 0.1: pass at all of T_7–T_13 | 0.9975, but T_14's known value sits at its 98.3rd percentile |
| risk, plan-pool model: pass at all of T_7–T_13 | 0.085, but it never reaches T_13's observed freeze-1c value |

Said plainly: A2 passes its registered checks, and the pass side at T_14 has one entry of room. The plan's gated
entries at T_14 lie above the tolerance at 9 times the control's rate (4 of 406 against 35 of 31,635), so the
control subset still does not represent the entries the gate reads. Neither risk model is calibrated at T_13.

## Amendment A3 (2026-11-06, before freeze 1e)

The head's decision of 2026-11-06 04:20Z (`M2_SPEC.md` §3.8, "Amendment A3"). At T_7–T_14 G-EX is reported: the job
writes the A2 statistic, the count of read entries above 0.038718 and their list, and continues. At T_5, T_6, T_15
G-EX stays a gate; G-TOPK (unchanged tiers), K0 and I-ID stay gates.

What was run (third executor; output dir `$SP/m2c/dry_a3`; 04:17:51Z to 04:18:38Z): the command of A2 with
`--out $SP/m2c/dry_a3`. 89 checks, 0 failed. Log with the progress bars removed: `offline_gate/dryrun_2026-11-06_A3.log`;
values `dryrun_2026-11-06_A3_summary.json`. pyflakes is clean on `m2.py`, `m2_score.py`, `m2_dryrun.py`.

Against the A2 log, three checks changed and three were added:

| check | result |
|---|---|
| JOB A3: T_14 marked as another device; G-EX there reported, not gated | exit 0; `t14_G_EX` absent from the gates, in the reports: 12 of 157 entries read, p99 0, 0 above; failing input p99 1.393; G-TOPK at T_14 a gate at 0.044273; G-EX at T_13 a gate at 0.021159 |
| JOB A3: the verdict's line on that run | "… at points none (of 1 reported …)" |
| FAIL A3: a planted G-EX excess at a cross-device point (T_13's bank × 1.2, T_13 another device) | the job completes (`M2_JOB_OK`, `complete` true); the report has p99 0.142, 15 of 15 read entries above the tolerance and a list of 15; the line names T_13 |
| FAIL A3: a planted G-TOPK excess (T_14's banked top-50 values × 1.1, T_14 another device) | exit non-zero naming `t14_G_TOPK` (873 entries, p99 0.0822 against 0.044273) |
| VERDICT A3: a reported point over the tolerance planted in the satisfying synthetic case | the headline is unchanged; the line names T_13 |
| removed: FAIL A2 (T_13 in T_14's place on the cross-device path stops the job) | replaced by the two FAIL A3 cases; the same-device version (all points one device) still stops the job at `t14_G_EX` |

Said plainly: the cross-device G-EX can no longer stop the job, so a wrong forward at T_7–T_14 would be caught only by
G-TOPK, G0, G-REV and G-CFG there, and would show in the report (the neighbouring point's bank, a stand-in for a wrong
forward, gives a median d of 0.34–1.83 at T_7–T_14, against 0.0043–0.0047 measured at T_13 and T_14). That is the cost of A3, accepted by the head.

## After `score-04` (2026-11-06): a second gap in this gate

`score-04` passed every gate through T_10, then crashed while scoring T_10: `m2.percentile` asserts that every pole of
the layer has a token of its own sign in the banked histogram, and L24N9826+ has none at T_10 (`n_pole_tokens` 0,
`max_abs` 0). L07N640− at T_8 is the same; no other pole in the 40 layer files scored at T_5–T_10
(`studies/next/m2/diag/score04_empty_poles.json`). `m2.matched` makes the same assumption (`max_abs > 0` over the layer).
The REAL section read the laptop banks of the 640 grid poles only, and the tiny world's random model has no empty
pole, so no check exercised the case. It was checkable offline by the same range reads of the banked stats files. Any
fix (for instance B = 0 for a pole with no token of its sign, and such a pole left out of the matched set) changes the
frozen score code and is the head's decision; it should come with a planted empty pole in the tiny world.

## Amendment A4 (2026-11-06, before freeze 1f)

The head's decision of 2026-11-06 05:35Z (`M2_SPEC.md` §3.8, "Amendment A4"). This closes the gap of the section above.

What was run (fourth executor; `$SP/m2f/`):
```
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/offline_gate/bank_scan.py --bank $BANK \
    --births studies/next/m2/births.json --out studies/next/m2/offline_gate/bank_scan.json      # 05:37:31Z-05:40:04Z
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz \
    --scan studies/next/m2/offline_gate/bank_scan.json --out $SP/m2f/dry_a4                      # 05:41:17Z-05:42:17Z
```
- The scan: 352 stats files and 352 topk files (all 32 layers, T_5–T_15) on S3, 2,112 byte-range or list calls. It
  reads max_abs, hist_edges, n_pole_tokens, q999 and density whole, and only the .npy headers of `hist`, `tok` and
  `a`. Output `offline_gate/bank_scan.json` (sha256 `8f0a41ba72f34295…`), log `offline_gate/bank_scan.log`.
- Its positive control: max_abs and q999 of the 640 grid poles equal the laptop bank at all 11 points (640 of 640,
  max difference 0). Its failing input: against the neighbouring point's laptop values, max_abs matches 0 of 640 and
  q999 0–40 of 640.
- Dry run: 112 checks, 0 failed (89 of freeze 1e, 23 new). Log with the progress bars removed:
  `offline_gate/dryrun_2026-11-06_A4.log`; values `dryrun_2026-11-06_A4_summary.json`. pyflakes is clean on `m2.py`,
  `m2_score.py`, `m2_dryrun.py` and `bank_scan.py`.
- The failing input of the fix: the tiny world with the planted pole, run on the freeze-1e code (`git archive
  m2-freeze-1e infra` in the scratchpad), crashes as score-04 did: `AssertionError: a pole with no token of its sign`,
  `m2.py` line 346 via `m2_score.py` line 334 (`offline_gate/a4_planted_on_freeze_1e.log`).

Poles with no token of their own sign (n_pole_tokens 0, max_abs 0, q999 NaN, density 0 for every one):

| point | all 32 layers | in the layers the point scores (the job's expected `n_degenerate_poles`) |
|---|---|---|
| T_15 | 0 | 0 |
| T_14 | 0 | 0 |
| T_13 | 0 | 0 |
| T_12 | 0 | 0 |
| T_11 | 0 | 0 |
| T_10 | 1: L24N9826+ | 1: L24N9826+ |
| T_9 | 2: L07N640−, L19N726+ | 0 |
| T_8 | 2: L07N640−, L31N3935− | 1: L07N640− |
| T_7 | 2: L08N8160−, L31N3935− | 0 |
| T_6 | 0 | 0 |
| T_5 | 0 | 0 |

The other facts the scoring path assumes, from the same scan: every stats file holds the six members the job loads and
every topk file `tok` and `a`; shapes [11008, 2] (max_abs, n_pole_tokens, q999, density), [11008, 2, 400] (hist),
[11008, 2, 50] (tok, a); `hist_edges` equal to `percentile`'s constants in 352 of 352; q999 finite and positive and
max_abs finite on every other pole; no carrier or twin is degenerate at a point where it is scored; the fewest
matched-pole candidates in a scored layer is 22,013. The full assertion list, with the banked quantity each assumes
and its check, is in `M2_SPEC.md` §3.8 "A4". Two assumptions stay unchecked by the scan, because they need a large
member: `hist.sum(-1)` = `n_pole_tokens` (true by construction in the harvest, asserted by the job at load) and a
finite top-50 `a` where `tok` ≥ 0 (a NaN would fail G-TOPK loudly).

The new checks:

| check | result |
|---|---|
| REAL A4: the scan's files, members, shapes, edges | 352 layer files, all as the job reads |
| REAL A4: positive control and its failing input | 640 of 640 at all 11 points; neighbour 0 (max_abs), 0–40 (q999) |
| REAL A4: the two poles score-04's read found | found again: L24N9826+ at T_10, L07N640− at T_8 |
| REAL A4: every degenerate pole consistent; q999 on the others | 7 of 7 consistent; 0 other poles with q999 not finite and positive |
| REAL A4: no scored unit's own pole degenerate; candidates | none; fewest 22,013 |
| REAL A4: expected counts in the scored layers | T_10 1, T_8 1, 0 elsewhere |
| REAL: the G-EX read subset non-empty at all 11 points | 5 (T_6) to 2,295 (T_15) entries |
| SCORE A4: degenerate pole with v = 1.5 | B 0 (0.618 without the rule), ratio 0, A 0, rank 6 of 6; other poles unchanged |
| SCORE A4: `matched` | the max_abs-0 pole dropped; the rest in the freeze-1e order |
| SCORE A4: three planted inputs | each new assertion stops (histogram and max_abs disagree; carrier with no token; 8 candidates < 20) |
| JOB A4: L00N82− (matched place 1 of carrier L00N40+, B 0.2545, rank 109 of 192) planted at T_13 | the job completes, every gate passes, every failing input fails |
| JOB A4: metrics.json | T_13 lists L00N82−; 0 elsewhere |
| JOB A4: the planted pole | B 0, ratio 0, rank 192 of 192 under both; out of the matched set; the other 99 in order |
| JOB A4: everything else | the same forward; every other pole's A, B, ratio unchanged; the carrier's ranks (B 42, A 192) unchanged |
| JOB A4: the verdict's line | "… T_15 0, T_14 0, T_13 1, T_12 0" |
| FAIL A4: empty histogram, max_abs > 0 | the job stops: "1 poles whose histogram and max_abs disagree" |
| FAIL A4: the carrier planted degenerate at its pre point | the job stops: "the carrier has no token of its sign at the scoring checkpoint" |

Said plainly: the rule touches 2 poles at 2 points of the real run, both competitors. At T_11–T_15 no layer holds such
a pole, so the scores there are computed exactly as under freeze 1e.

## After `score-05` (2026-11-06): a third gap in this gate

`score-05` stopped at `t06_G0`. The gate passed (max ratio 0.00352 against G0_TOL 0.01), but its registered failing
input, the activation swapped (silu → gelu_tanh), passed too: 0.00970. The swap's power falls with the checkpoint. The
job's failing-input max ratio was 0.637 at T_15, 0.066 at T_10, 0.018 at T_7 and 0.0097 at T_6. The banked harvest's own
G0 controls give 0.0113 at T_6 and 0.0062 at T_5, a pass (`studies/next/m2/diag/score05_g0_fail.json`). So T_5 would have
stopped the job too. The harvest also registered a second G0 failing input, u scaled by 1.1, whose power does not depend
on |g|: 0.103–0.105 at every point T_5–T_15. M2's job registered only the swap.

The gap: this gate never read the banked `controls.json` files (3 KB each on S3), which showed it. In the tiny world the
swap's max ratio was 0.150–0.278 at all four points (`offline_gate/dryrun_2026-11-06_A4_tiny_metrics.json`), so no case
had a failing input without power. The A4 assertion list covered banked quantities in the scoring path, not the power
of each gate's failing input at every point. A check of that class, before a launch, is: every registered failing input
of every gate, computed on the banked values where they exist, fails at every point where the gate is applied. Any
change to G0 is the head's decision.

## Amendment A5 (2026-11-06, before freeze 1g)

The head's decision of 2026-11-06 06:50Z (`M2_SPEC.md` §3.8 "A5"). This closes the third gap of the section above: a
failing input with no power where it is applied. The rule for this gate from now on: every gate the job runs, and every
verdict gate the bank holds, is evaluated on both sides at every point where it is applied, from the banked files,
before a freeze. Where the bank does not hold the quantity, the record says what the gate assumes.

What was run (fifth executor; `$SP/m2g/`):
```
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/offline_gate/a5_gate_sides.py --bank $BANK \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz \
    --out studies/next/m2/offline_gate/a5_gate_sides.json                                       # 07:00:04Z-07:00:59Z
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank $BANK --hf $SP/m2/hf \
    --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz \
    --scan studies/next/m2/offline_gate/bank_scan.json --a5 studies/next/m2/offline_gate/a5_gate_sides.json \
    --out $SP/m2g/dry_a5                                                                         # 07:03:07Z-07:04:05Z
```
- `a5_gate_sides.py` read the 11 banked `controls.json` (sha256 equal to those in `diag/score05_g0_fail.json`) and 11
  `manifest.json` files, the HF refs (one API read, response sha256 `511db444…`), the five attempts' `metrics.json`,
  the laptop exemplar arrays, the T_14 L40S repeat, the committed branch configs, and by byte-range reads the topk
  members `tok` and `a` of layer 3 at T_5 and T_6 and of the five layers T_7 scores, and `tok` of the 26 layers that
  hold freeze-half births at T_15. Output `offline_gate/a5_gate_sides.json` (sha256 `717bde0d…`), log
  `a5_gate_sides.log`. G-TOPK is written as aggregates only.
- Dry run: 130 checks, 0 failed (112 of freeze 1f, 18 new). Log with the progress bars removed:
  `offline_gate/dryrun_2026-11-06_A5.log`; values `dryrun_2026-11-06_A5_summary.json`; the tiny job's `metrics.json`
  `dryrun_2026-11-06_A5_tiny_metrics.json`. pyflakes is clean on `m2.py`, `m2_score.py`, `m2_dryrun.py` and
  `a5_gate_sides.py`.
- The failing input of the fix: the no-power swap (the identity) planted in the tiny world on the freeze-1f code
  (`git archive m2-freeze-1f infra`) stops at `t15_G0` with `AssertionError: t15_G0: its failing input passed`, the
  class of score-05's stop (`offline_gate/a5_planted_on_freeze_1f.log`).

G0, both sides at every point (harvest: first batch, windows 0–7; score-05: the plan's first 8 windows):

| point | pass side: harvest / score-05 (max over layers) | u × 1.1: harvest min–max over 32 layers | bound 0.1 ± 1.1 r0 | activation swap (reported): harvest / score-05 |
|---|---|---|---|---|
| T_15 | 0.00511 / 0.00607 | 0.0944–0.1044 | 32 of 32 | 0.676 / 0.637 |
| T_14 | 0.00539 / 0.00503 | 0.0950–0.1042 | 32 of 32 | 0.712 / 0.572 |
| T_13 | 0.00428 / 0.00476 | 0.0956–0.1038 | 32 of 32 | 0.381 / 0.454 |
| T_12 | 0.00617 / 0.00521 | 0.0958–0.1046 | 32 of 32 | 0.299 / 0.368 |
| T_11 | 0.00591 / 0.00465 | 0.0972–0.1047 | 32 of 32 | 0.139 / 0.130 |
| T_10 | 0.00474 / 0.00501 | 0.0950–0.1052 | 32 of 32 | 0.0804 / 0.0662 |
| T_9 | 0.00435 / 0.00425 | 0.0957–0.1037 | 32 of 32 | 0.0519 / 0.0539 |
| T_8 | 0.00362 / 0.00371 | 0.0978–0.1036 | 32 of 32 | 0.0267 / 0.0317 |
| T_7 | 0.00311 / 0.00347 | 0.0978–0.1026 | 32 of 32 | 0.0177 / 0.0177 |
| T_6 | 0.00349 / 0.00352 | 0.0965–0.1035 | 32 of 32 | 0.0113 / **0.0097, passes** |
| T_5 | 0.00334 / not reached | 0.0965–0.1032 | 32 of 32 | **0.0062, passes** / not reached |

Every other gate, both sides (`M2_SPEC.md` §3.8 "A5" has the full table with sources):

| gate | where | pass side | failing side |
|---|---|---|---|
| G-REV | 11 points | registered = banked manifest = HF ref today, 11 of 11 | forty zeros: fails, 11 of 11 |
| G-CFG | 11 points | 11 configs, no differing key | model_type renamed: fails, 11 of 11 |
| K0 | T_15 | bit-identical in score-02 to score-05 | T_14's values: p99 3.7936 (179 times 0.021159) |
| G-EX gated | T_15, T_6, T_5 | T_15 bit-identical (2,295); T_6, T_5 assumed (below) | neighbour's bank: 179, 57, 47 times 0.021159 |
| G-TOPK | 11 points | score-05 at T_15–T_7 (p99 0–0.0158); T_6, T_5 assumed | score-05 p99 1.016–1.403 at T_15–T_7; T_6, T_5 lower bound p99 0.623 and 0.218 against 0.021159 (4 entries each) |
| I-ID | T_13 | score-04, score-05 bit-identical, 0 flips | T_14's birth-role values: p99 3.3811 |
| construction control under A | freeze half, T_15 | 82 of 82 in the top 10 (74 rank 1, 8 tied 2–10, largest 10) | the opposite pole: 0 of 82 |
| usable births | both halves | 82 and 71; twins 79 and 70 | 30 test births (dry run) |

The new checks:

| check | result |
|---|---|
| REAL A5: controls sha256; G0 pass side | 11 of 11 registered; passes at 11 points (harvest) and T_15–T_6 (score-05) |
| REAL A5: u × 1.1 | fails on every layer at all 11 points (min 0.0944) |
| REAL A5: the bound | 352 of 352 layer-points within 0.1 ± 1.1 r0 |
| REAL A5: the swap's power | passes at T_5 (harvest 0.00622) and T_6 (score-05 0.00970); fails elsewhere |
| REAL A5: G-REV, G-CFG; the same-device harvest run | 11 of 11 both sides; T_5, T_6, T_15 from `grid_a`, code `17057bd0`, L40S, the AMI's versions |
| REAL A5: G-EX gated points | failing input 179, 57, 47 times the tolerance; T_15 exact in four attempts |
| REAL A5: G-TOPK | the offline count reproduces score-05's 10 at T_7 and its bound lies below the job's values; T_6 and T_5 have 4 entries each and fail for certain; score-05's T_15–T_7 both sides |
| REAL A5: I-ID; construction control under A; usable births | as in the table above |
| JOB A5: G0 in the tiny world | u × 1.1 min over layers 0.0955–0.0995 at the four points, within the bound; the swap reported at every point (0.150–0.278), not gated |
| FAIL A5: the swap planted as the identity | the job completes; the swap reported under 0.01 (max 0.00468); u × 1.1 still fails |
| FAIL A5: u × 1.0 planted | the job stops at `t15_G0` ("its failing input passed") |

What the bank cannot check, said plainly: the pass side of K0, G-EX and G-TOPK at T_6 and T_5 (no attempt reached the
forward there; the gates assume same-device bit-identity, shown at T_15 in five attempts, with T_5, T_6 and T_15 banked
by one harvest run); the value of u × 1.1 on the job's own windows (bounded); G-TOPK's failing value (a lower bound
only); I-ID (no all-neuron bank; both sides from the jobs at T_13); the construction control under B (needs every
pole's activation at E_c). The construction control under A on the test half is computable but is the verdict's, and
stays blind.

## Not verified here, said plainly

- No GPU number exists yet. The K0 bit-identity on an L40S, the forward rate, the download rate on this lane and the
  stop rule's behaviour at real scale are measured by the launch itself.
- The tiny world checks the code path, the position mapping and every gate. It is not a test of the science: its births
  are synthetic.
- The prior for B (wdd e587) comes from one repository page, read for those numbers only. Stecher et al. (2605.18789)
  was not read in full.
- The D1 parent-test numbers (A = 0.812, a_ctrl 0.924) are quoted from `studies/already-clean/results/RESULTS.md`, whose
  source `metrics.json` is on S3 and was not re-read.
- The windows per point and the cost estimates assume the harvest's banked tokens-per-second rate. That rate includes
  bookkeeping this job does not do.

## Files created

| file | what |
|---|---|
| `studies/next/07_PICKS/M2_SPEC.md` | the pre-registration |
| `studies/next/07_PICKS/M2_OFFLINE_GATE.md` | this record |
| `infra/pg/m2.py` | births, plan, scores, ranks, floors, labels, freeze, verdict, projection; CLI |
| `infra/pg/m2_examples.py` | the example selection and the zero-GPU raw examples |
| `infra/aws_pg/jobs/m2_score.py` | the job |
| `infra/aws_pg/jobs/m2_dryrun.py` | the offline gate |
| `infra/aws_pg/steps/next_m2_score-01.json` | the step list of L1 |
| `studies/next/m2/births.json` | the 153 births, the split, the twins, the example selection, the freeze-half look |
| `studies/next/m2/projection.json` | GPU-hours per point |
| `studies/next/m2/samples/raw_examples_zero_gpu.{md,json}` | the ten raw examples, zero-GPU part |
| `studies/next/m2/offline_gate/{dryrun.log, dryrun_summary.json, device_check_t14_1b_vs_1c.json, hf/}` | the gate's log and inputs |
| `studies/next/m2/offline_gate/dryrun_2026-11-05_A1{.log,_summary.json}`, `studies/next/m2/diag/a1_risk.{py,json,log}` | amendment A1: the dry run after the change; the exploratory risk of a G-EX stop |
| `studies/next/m2/offline_gate/dryrun_2026-11-05_A1rev{.log,_summary.json}`, `studies/next/m2/diag/a1_risk_rev.{json,log}` | the revision of A1: the dry run after it; the risk model rerun at the new tolerance |
| `studies/next/m2/offline_gate/dryrun_2026-11-06_A2{.log,_summary.json}`, `studies/next/m2/diag/{a2_risk,risk_calibration,score02_t13_gex}.{py,json,log}` | amendment A2: the dry run after it; the risk models and their calibration; the score-02 stop diagnostic |
| `studies/next/m2/offline_gate/dryrun_2026-11-06_A3{.log,_summary.json}`, `studies/next/m2/diag/score03_t13_gex.{py,json,log}` | amendment A3: the dry run after it; the score-03 stop diagnostic |
| `studies/next/m2/diag/score04_empty_poles.{py,json,log}` | after score-04: the poles with no token of their sign at T_5–T_10 |
| `studies/next/m2/offline_gate/bank_scan.{py,json,log}`, `studies/next/m2/offline_gate/dryrun_2026-11-06_A4{.log,_summary.json,_tiny_metrics.json}`, `studies/next/m2/offline_gate/a4_planted_on_freeze_1e.log` | amendment A4: the scan of the banked layer files; the dry run after it (with the tiny job's metrics.json); the planted pole on the freeze-1e code |
| `studies/next/m2/diag/score05_g0_fail.{py,json,log}` | after score-05: G0's failing input per point in the five attempts and in the banked harvest's controls |
| `studies/next/m2/offline_gate/a5_gate_sides.{py,json,log}`, `studies/next/m2/offline_gate/dryrun_2026-11-06_A5{.log,_summary.json,_tiny_metrics.json}`, `studies/next/m2/offline_gate/a5_planted_on_freeze_1f.log` | amendment A5: both sides of every gate at every point from the banked files; the dry run after it; the no-power swap on the freeze-1f code |
