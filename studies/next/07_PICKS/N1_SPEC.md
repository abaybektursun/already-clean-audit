# N1: the weight order at birth (J12), with the onsets located first. Pre-registration

**Answer.** One CPU launch (r7i.2xlarge, no GPU), projected at 0.82–1.71 instance-hours, hard kill 134 min
(`studies/next/n1/projection.json`), reads the MLP weights of the 16 grid checkpoints and measures, for every neuron and
every bracket between two checkpoints, the relative change of its input rows (gate and up: the directions it reads) and of
its output column (down projection: the direction it writes). The zero-GPU pre-check located the onsets on the grid first
(`studies/next/n1/onsets/onsets.json`): the dating rule reproduces k for all 442 units, but the rise is spread over several
brackets. In 93 of 153 births the dated bracket holds less of the rise than the two brackets before it together; 20 births
crossed the threshold earlier and fell back; one checkpoint before birth the median ratio of the cluster's activation to
the pole's own threshold is already 0.84. Twins look the same (not concentrated: 89 of 140 distinct-token twins, 85 of 149
M2 twins). So the registered readout is the excess over the rise window, the brackets k−2, k−1 and k, against the neuron's
own earlier brackets, with each bracket normalized within its layer; the birth bracket alone is reported. Locating the
onset inside a bracket needs forwards at intermediate checkpoints and a fresh 99.9th percentile there, about 0.1 GPU-h per
checkpoint, 2.7–3.2 GPU-h for three per bracket (§2.1; not registered). The known number is the down-projection column norm:
the bank holds 775 values at T_5–T_15 and N6's whole T_15 array, bitwise equal across three L40S runs; the neighbouring
checkpoint's value, the registered failing input, differs by at least 0.00155 relative, 12 times the tolerance. The
weight-free dry run printed `N1_DRYRUN_OK` (44 checks, 0 failed): on 16 tiny checkpoints with one neuron's input rows
changed in one bracket, the job's statistics equal an independent recount and the planted neuron reads 13.8 on the input
side and −0.4 on the output side.

Written 2026-11-06 by an Opus worker in worktree `interp_research-bio2`, branch `bio2` (from `clean-neurons` at
`ae8dc637f`). No model was run, no weights were downloaded, nothing was launched. Reads: the laptop banks (J1's recount
arrays), N12's and M2's banked records, N6's banked column norms (branch `n6`, git LFS); S3 listings; two web searches.

---

## 0. Status and provenance

| field | value |
|---|---|
| status | ACCEPTED by the head (2026-11-06 19:25Z) with one change before freeze 1 (§0.6): K-B is reported, not gated. Dry run on the changed code: `N1_DRYRUN_OK`, 44 checks, 0 failed (`studies/next/n1/offline_gate/dryrun.log`; record `N1_OFFLINE_GATE.md`, section "Freeze 1"). The zero-GPU onset pre-check (§2.1) and the ten raw examples' banked part (§10) were read before this draft, on both halves for the onsets (they are dates, not the measured quantity) and on the freeze half for the one weight look (§0.4) |
| freeze 1 | after the head's review: tag `n1-freeze-1` on the commit holding this spec, `infra/pg/n1.py`, `infra/aws_pg/jobs/n1_weights.py`, `infra/aws_pg/jobs/n1_dryrun.py`, `infra/aws_pg/steps/next_n1_weights-01.json`, `studies/next/n1/{units.json, kb_bank.json, inputs/n6_colnorms_t15.npy}`, `studies/next/n1/onsets/onsets.py`, `studies/next/n1/offline_gate/gate_sides.py`, `studies/next/n1/samples/raw_examples.py` |
| freeze 1a | A1 (§14, 2026-11-07): the raw-examples writer reads the example births from the job's registry (`n1.EXAMPLES`), not from `units.json`; weights-01 DOES NOT COUNT (it stopped at its last step with `KeyError: 'examples'`); dry run `N1_DRYRUN_OK`, 46 checks, 0 failed (`studies/next/n1/offline_gate/dryrun_a1.log`); tag `n1-freeze-1a`; the launch is `weights-02` (`infra/aws_pg/steps/next_n1_weights-02.json`). No rule changed |
| freeze 1b | A2 (§14, 2026-11-07): the job budget 120 → 300 min and the hard kill 134 → 320 min (HF downloads at 24–36 MB/s); weights-02 killed after T_2, DOES NOT COUNT; tag `n1-freeze-1b`; the launch is `weights-03` (`infra/aws_pg/steps/next_n1_weights-03.json`). No code or rule changed |
| freeze 2 | `studies/next/n1/thresholds.json` from the freeze half only (`n1 freeze`, sha256 `fad4fe71…`, 2026-11-07T19:34:22Z, weights-03); **Tagged** `n1-freeze-2` (annotated, tag object `88517fc7d`) on commit `6709f754e`, 2026-11-07T19:34:42Z. The verdict ran once at 19:35:10Z (`studies/next/n1/verdict_n1.json`) |
| parents | `studies/next/09_PLAN.md` §3 card J12 ("N1 first locates onsets inside their brackets"); `studies/next/06_CRITIC.md` N1 and Part 6; `studies/next/03_IDEAS.md` N1; `studies/next/05_SCORING.md` N1 row ("ours must use both definitions"); `studies/next/n12/READ_head.md` (the decision table: "the weight change behind the rise is the next object (N1, with the onsets located first)"); `studies/already-clean/SPEC_D1.md` §2.4 (births); `studies/next/12_SHAPING_NOTE.md` §4 items 3–6 |
| method | repo `CLAUDE.md`; the `research-spec` skill; N12's spec format; M2 A5 (both sides of every gate on the banks) |

### 0.1 The brief's requirements and where each sits

| requirement | where |
|---|---|
| which changes first at a birth: input rows or output column; does the onset sit inside the dating bracket | §1.1, §1.3, §2.1 |
| pre-check: onsets in their brackets from the banked exemplar activations; what finer location needs the forward | §2.1; `onsets.json` |
| per birth: relative change of rows and column between T_(k−1) and T_k, earlier brackets as reference, fp32 from the HF weights, on a CPU instance; instance-hours | §2.2, §9.6 (fp64 from the bf16 weights, an exact upcast) |
| twins (the distinct-token twins) as the selection control | §3.3 |
| hypotheses (rows first, column first, together, neither beyond the twins) with refuting observations | §1.3 |
| thresholds frozen on the freeze half; floors from the twins and matched random poles; a ceiling; does-not-count branches | §3.4, §3.5, §5, §6 |
| known number (a banked weight statistic) and the identity pair | §3.1, §3.2 |
| the ten raw examples at zero GPU where the banks allow | §10 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| model, points | `allenai/Olmo-3-1025-7B`, T_0 … T_15 = stage-1 steps 0 … 1,413,814 (`m2.POINTS`); 32 layers, MLP width 11,008, hidden 4,096; gate_proj and up_proj `[11008, 4096]`, down_proj `[4096, 11008]`, bf16, 3 shards, 14.6 GB | `infra/pg/m2.py`; `$SP/m2/hf/index_stage1-*.json`; N13 R1 G-PARTITION (mlp 4,328,521,728 parameters) |
| units | 153 births (82 freeze, 71 test), M2's twins 149 (79, 70), N12's distinct-token twins 140 (73, 67); births at k = 6, 8–15 | `studies/next/n12/units.json`; `studies/next/n1/units.json` |
| matched neurons | M2's 100 activation-matched poles of each birth at T_(k−1) (nearest in log max_abs), as neurons: 98–100 per birth | M2 `ranks_*.jsonl` → `matched`; `units.json` |
| brackets | steps per bracket at the births: k = 6 4,000; 8 15,000; 9 25,000; 10 40,000; 11 80,000; 12 140,000; 13 240,000; 14 440,000; 15 413,814 | `onsets.json` → `bracket_steps_by_k` |
| the banked weight statistic | `value.colnorm` = `down_proj.weight.float().norm(dim=0)` on an L40S, in M2's and N12's records (775 distinct (layer, neuron, point) values at T_5–T_15); N6 L1-02 `colnorms.npy` [32, 11,008] at T_15 | `kb_bank.json`; N6 `n6_edit.colnorms` |
| that statistic's reproducibility | M2 against N12: 458 of 458 bitwise; M2 against N6 at T_15: 184 of 184; N12 against N6: 122 of 122 | `offline_gate/gate_sides.json` |
| optimizer state | not public for stage 1; no Adam preconditioning is possible | `00_ASSETS.md` §6 item 8 |
| CPU instances | the launcher runs them with `--no-weights` (plenum, regrain jobs); the Standard on-demand quota is 16 vCPU (`infra/aws_pg/plenum_agg_launch.sh`); r7i.2xlarge 8 vCPU, 64 GB | the scripts named |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | torch 2.12.0 imports on the CPU instance from the AMI's venv | G-SAME | stop |
| 2 | each checkpoint is the registered revision, with the 96 MLP tensors at the registered shapes and dtype | G-REV, G-SHAPE | stop |
| 3 | the job's column norms equal the banked ones within the fp32 bound | K-B at T_5–T_15 | reported, not gated (§0.6) |
| 4 | reading a checkpoint twice gives bitwise the same statistics | I-ID at T_15 | stop |
| 5 | the rate (download 90–150 s, compute 60–180 s per checkpoint) | `points.<t>.seconds` | the job asserts it is under 0.95 of its budget after each checkpoint (fail loud; no parking: every bracket is needed) |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives N1 |
|---|---|---|---|
| births and their dates | J1's recount: 153 passing births (p_late ≤ 0.05 under R1t); D1's rule | `studies/next/10_RECOUNT2.md`; `m2/births.json` | the units; the dating rule |
| the carrier at birth | N12: the carrier wins by its own rise (close to forced by the dating rule, the reader); its leaders keep their rank; token repetition explains M2's twin gap | `n12/READ_head.md` addendum | the activation side of the birth is the rise; the weight side is this card |
| column norms at birth | M2 and N12 banked the carrier's column norm at T_(k−1) and T_k; freeze-half look (exploratory, before this spec): the log change of the column norm across the birth bracket, real minus distinct-token twin, +0.0021 (sd 0.018, 73 pairs); the change itself follows the bracket (k = 15: 0.28 for real and twins alike; k = 12: 0.04–0.05) | `n12/banks/score-02/n12/ranks_freeze.jsonl`; §10 | a norm change is a lower bound on the column's change; at the norm level the birth does not show |
| weight change, other models | wdd (Pythia, OLMo-1B): the output vector is the moving part in Pythia; the input side selective before entry on OLMo-1B (per `04_WDD.md` §6, REPO-DOC) | `studies/next/06_CRITIC.md` N1 item 4 | both orders have precedent; "ours must use both definitions" (`05_SCORING.md`) |
| a per-block weight change | N13 R1: ‖θ_RL − θ_base‖ / ‖θ_base‖ per block, fp64 on the CPU, bitwise across two runs | `studies/next/n13/s3/s3gate-03/r1_rank/` | the precedent for weight statistics on an instance's CPU; not this card's checkpoints |

### 0.5 Objects, defined once

- **Bracket j** = (T_(j−1), T_j], j = 1 … 15. **Units**: a birth's carrier (real), its distinct-token twin (dt, N12's), M2's
  twin (pb), each a pole of the birth's layer dated k by the same rule; for weights, the pole's **neuron**.
- **in** = the neuron's gate row and up row as one 8,192-vector (the directions it reads); **out** = its down-projection
  column, 4,096 entries (the direction it writes). gate and up are also reported alone.
- **r_M(n, j)** = ‖M_n(T_j) − M_n(T_(j−1))‖ / ‖M_n(T_(j−1))‖, fp64 from the bf16 checkpoints (an exact upcast).
- **z_M(n, j)**: log r_M(n, j) minus its median over the layer's 11,008 neurons, divided by 1.4826 × their median absolute
  deviation. All neurons of a layer share a bracket's step count and learning rate; z removes both (`06_CRITIC.md` 0.5).
- **e_M(n, k)** = mean of z_M over the **rise window** W(k) = {k−2, k−1, k} minus mean of z_M over the **reference
  brackets** {2, …, k−3} of the same neuron: the excess over the neuron's own history, which removes a lasting scale such
  as a small norm. Bracket 1 (initialization to step 1,000) is excluded. **e_k**: the same with the birth bracket alone.
- **cos_M(n, j)**: the cosine between M_n(T_(j−1)) and M_n(T_j) (reported; the J12 card's raw examples ask for it).

### 0.6 The head's decisions before freeze 1 (2026-11-06 19:25Z)

1. **K-B is reported, not gated.** No control pair exists for an L40S fp32 reduction against a CPU fp64 one (the two-path
   rule). The job writes K-B's values and its failing input's to `metrics.json` → `reports` → `tNN_K_B`; a K-B failure no
   longer stops the job. `infra/aws_pg/jobs/n1_weights.py` (`R.report` in place of `R.gate`) and the dry run's two K-B
   checks changed; nothing else in the code.
2. **The onset finding is the basis of the registered readout** and is a stated limitation (§8 row 10): births are
   gradual rises; 93 of 153 births hold less of the rise in the dated bracket than in the two brackets before it; the
   median ratio of the cluster's activation to the pole's threshold one checkpoint before birth is 0.84 (0.838 on each
   half, `onsets.json`). The intermediate-checkpoint onset location (§2.1, 2.7–3.2 GPU-h) is **not registered**.
3. Everything else in the draft is accepted as written.

---

## 1. Question, decision, hypotheses

### 1.1 The question

When a cluster is born on a pole (its median activation on the cluster's contexts rises past the pole's own 99.9th
percentile and stays), which part of the neuron's weights changes beyond what the dating rule's selection gives: the rows
that read (gate and up), the column that writes (down), both, or neither? And does the birth's onset sit inside the bracket
the dating rule names?

### 1.2 The decision this run makes

| outcome (test half) | reading | next |
|---|---|---|
| ROWS_FIRST: in ABOVE_SELECTION, out not | the neuron's reading directions change beyond a selected twin's during the rise; its written direction does not | the job-direction projection of the row change (the critic's fix; §8 item 1) and the readers' change after it (N1's "readers later") |
| COLUMN_FIRST: out ABOVE, in not | the written direction changes beyond selection while the reading directions change as any dated pole's do | the readers of that column (N3's route) |
| TOGETHER: both ABOVE | both sides change beyond selection in the rise window; order unresolved at this resolution | the finer onset location (§2.1) to order them |
| NEITHER (H_0 when both AT) | whole-vector change at a birth is what selection gives | the rise is not a whole-vector weight event; the next instrument is a job-direction projection, not a norm |
| DOES NOT COUNT | no claim | the failed gate and its value |

### 1.3 Hypotheses and the observation that refutes each

On the test half; "paired" is real minus dt by birth on e over the rise window (`n1.half_summary` → `pair_<side>_real_dt`),
percentile bootstrap 95 % interval over births (10,000 resamples, seed 0).

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_rows** (the card's claim) | the input rows change first: beyond selection on the input side, not on the output side | in ABOVE_SELECTION, out AT or BELOW | in AT or BELOW, or out ABOVE |
| **H_col** | the output column changes first | out ABOVE, in AT or BELOW | out AT or BELOW, or in ABOVE |
| **H_together** | both change beyond selection in the same window | both ABOVE | either not ABOVE |
| **H_sel** (the boring one) | what changes at a birth is what dating a pole selects: the dating rule reads the activation, which the input rows and the residual set, so a dated twin shows the same change | both AT_SELECTION; and the input side of real and twin alike above the matched floor (reported) | any ABOVE or BELOW |
| **H_onset** (pre-check, answered at zero GPU) | the onset sits inside the dated bracket | the dated bracket holds most of the rise | done: 93 of 153 births hold less rise in the dated bracket than in the two before it (§2.1); refuted at the grid's resolution |

The order readouts, reported with intervals and no label: the profile of e per bracket k−2, k−1, k, k+1 on each side; the
gate and up rows separately; e_k; real minus pb. They say whether a side's excess appears before the other's within the
window.

### 1.4 The author's prior, logged before any weight statistic exists (2026-11-06)

| quantity | prior | basis |
|---|---|---|
| NEITHER (both AT) | 0.45 | rotational equilibrium under AdamW (Kosson et al., 2305.17212, search summary: per-neuron angular updates converge to a steady state); the gradual onsets; the freeze-half column-norm look (+0.0021) |
| ROWS_FIRST | 0.25 | the dating rule reads the input side, but the twins share that selection; wdd on OLMo-1B (input side selective before entry) |
| TOGETHER | 0.1 | |
| COLUMN_FIRST | 0.1 | wdd on Pythia (the output vector moves) |
| a BELOW label on either side | 0.1 | real births are later than chance for their pole (M2 §3.3), so their neurons may change less |
| K-B passes at all 11 points (reported, §0.6) | 0.95 | the bound is 1,000 times typical fp32 rounding |
| PC passes at every bracket | 0.97 | tiny world 1.0 at every bracket |

### 1.5 Literature (query terms recorded)

Searches of 2026-11-06 (WebSearch): (1) `neuron input weights versus output weights which changes first during feature
learning training dynamics language model checkpoints`; (2) `"rotational equilibrium" AdamW weight decay neurons angular
update per neuron Kosson 2305.17212`. Earlier records: `04_LITERATURE.md` N1; `04_WDD.md` §2, §6; the critic's N1.

- Kosson, Messmer, Jaggi, "Rotational Equilibrium" (ICML 2024, arXiv 2305.17212): weight decay and the update balance on a
  neuron's norm, so its expected angular update per step converges to a steady state, in some setups the same across
  layers and neurons whatever their norms. Search summary and the critic's abstract read; no number cited. It is why r is
  normalized within the layer and against the neuron's own history: a constant per-neuron rotation would read as z ≈ its
  own reference and e ≈ 0.
- wdd-descendants (Pythia, OLMo-1B), per `04_WDD.md`: the output vector moves in Pythia; the input side is selective on
  OLMo-1B eight intervals before entry; a row's change has cosine 0.002 with the negative gradient. REPO-DOC level.
- Not found in two searches (and none in the critic's): a checkpoint study of a language model that orders the input and
  output weight changes of a neuron around a dated feature. Two searches are not a hole in the literature.

---

## 2. The measurement and the instrument

### 2.1 The pre-check: the onsets on the grid (done, zero GPU)

`studies/next/n1/onsets/onsets.py` → `onsets.json` (sha256 `bb7158e5…`), from J1's recount arrays (the activation at each
T_15 exemplar position read at all 16 points, and each pole's q999 per point; sha256 `80e5ffac…`). Per unit,
r_t = median over the cluster's positions of a⁺_t divided by q999_t.

| | real freeze | real test | dt freeze | dt test | pb freeze | pb test |
|---|---|---|---|---|---|---|
| units; k reproduced | 82; 82 | 71; 71 | 73; 73 | 67; 67 | 79; 79 | 70; 70 |
| an earlier crossing that fell back | 12 | 8 | 5 | 7 | 13 | 11 |
| concentrated (the dated bracket's rise exceeds the two brackets before it together) | 34 | 26 | 29 | 22 | 35 | 29 |
| median r at T_(k−1); at T_k | 0.838; 1.99 | 0.838; 1.70 | 0.736; 1.73 | 0.787; 1.59 | 0.870; 2.04 | 0.749; 1.48 |
| median rise in the dated bracket; in the two before (decades) | 0.49; 0.53 | 0.37; 0.47 | 0.43; 0.45 | 0.34; 0.52 | 0.43; 0.45 | 0.34; 0.30 |
| exploratory crossing point inside the bracket (log-linear in log step), median | 0.14 | 0.20 | 0.32 | 0.26 | 0.13 | 0.33 |

What it says. The rule's k is the bracket where a rise that has been under way for two or three brackets crosses the
threshold for good. In 107 of 153 real births at least three of the four brackets k−3 … k raise r by more than 0.05
decades; 49 births sit above 0.9 of the threshold one checkpoint before. Twins rise the same way, as the dating rule
makes them. The crossing point inside the bracket is an interpolation that assumes a smooth monotone rise; no bank can
check it. Consequence for the design: the excess is read over the rise window k−2 … k (registered), and the birth bracket
alone is reported.

**What finer location needs.** The banks stop at the grid. Inside a bracket, a stage-1 checkpoint exists every 1,000 steps.
Locating the crossing needs, at each intermediate checkpoint, the cluster's activation on its positions (a forward of the
birth's exemplar windows, about 6.5 windows per birth at 0.23 s each) and the pole's own q999 there, which needs a corpus
pass (the full 20.56M-token harvest is 1.7 GPU-h per checkpoint; a 1M-token subsample of 512 windows about 2 minutes,
whose q999 must first be checked against the bank's at a grid point, a known number of its own). Per checkpoint about
5–7 minutes with the 14.6 GB download, about 0.1 GPU-h; three interior checkpoints in each of the 9 brackets holding births
are 27 checkpoints, 2.7–3.2 GPU-h; a three-level bisection (7 per bracket) 6.3–7.4 GPU-h. The weight statistics at the
same checkpoints cost only CPU time (about 2.5 minutes each). Not registered here; the head decides after this card.

### 2.2 The instrument, exactly

`infra/aws_pg/jobs/n1_weights.py`, one launch on a CPU instance, checkpoints T_0 … T_15 in order:

1. `m2_score.download` (the registered branch and revision), G-REV; the 96 MLP tensors read with `safetensors.safe_open`
   (any shard), G-SHAPE.
2. Per layer, fp64 (`torch`, the bf16 values upcast exactly): the norm of every gate row, up row and down column
   (`norms [16, 32, 11008, 3]`); for j ≥ 1 the norm of each vector's change since T_(j−1) and the dot product of the two
   vectors (`diff, dot [15, 32, 11008, 3]`). The previous checkpoint's MLP tensors are held in memory (bf16, 8.7 GB); one
   checkpoint on disk at a time.
3. K-B where the bank holds a value (§3.1), reported, not gated (§0.6); TRI reported.
4. PC at j ≥ 2 (§3.4).
5. At T_15, I-ID (§3.2).
6. After the last checkpoint: `stats.npz` (about 390 MB uncompressed); `n1.records` → `n1_<half>.jsonl`, one line per unit
   with r, z and cos at every bracket for each side, e over the window, e_k, e per bracket k−2 … k+1, the matched neurons'
   e, and the ceiling; `samples/raw_examples_<half>.jsonl`. `metrics.json` carries gates, timings and the PC shares, never
   an excess, a z or a label.

### 2.3 What a whole-vector relative change can and cannot say

- **It can say** whether a neuron's reading or writing vectors moved more in the rise window than the layer's typical neuron
  in the same brackets, relative to the same neuron's earlier brackets, and whether a dated twin's moved as much.
- **It cannot say** whether the change points along the job. A row can turn by the layer's typical angle while its
  component along the cluster's input direction grows; the critic's fix (the change projected on the mean residual at the
  exemplars minus activation-matched positions, against random directions) needs the residual at those positions from a
  forward, and the column's job direction needs the readers' gradient. Neither is in the banks; they are the next
  instrument if this one reads NEITHER (§1.2).
- **It cannot use** Adam's second moment: stage 1's optimizer state is not public.
- **Norm-only statistics are not enough**: the freeze-half look shows the column norm's change across the bracket follows
  the bracket for real units and twins alike; the vector change is measured instead, and z is taken within the bracket.

### 2.4 What runs at zero GPU

The onset pre-check (done), the units, the bank's column norms and their control pairs, K-B's failing side, the room of the
labels, the ten examples' banked part. The weight statistics need the 16 checkpoints, 234 GB of downloads: an instance.

---

## 3. Controls and references

Rule 4: fit on both sides (K-B against three banked runs), a positive control that fired (PC on the real weights at every
bracket), the identity pair (I-ID).

### 3.1 The known number K-B

At every checkpoint where the bank holds a column norm (T_5 … T_15: 775 values from M2's and N12's records; at T_15 also
N6's 352,256): |ours − bank| / bank ≤ 1.3e-4 for every compared column. The tolerance is the fp32 rounding bound of the
bank's computation, not a fitted value: the bank sums 4,096 squares in fp32, whose relative error is at most
(n + 1)ε / (1 − (n + 1)ε) = 2.44e-4 on the sum (ε = 2^-24) in any summation order, so 1.22e-4 on the norm, plus the final
rounding. Ours is fp64, exact to about 1e-15. **Two paths** (an L40S's fp32 reduction against a CPU's fp64): the two-path
rule asks for a tolerance from a control pair on those paths or a report. No CPU-fp64 norm of these columns is banked, so
no control pair exists; the bound holds for every order of an fp32 sum, which is what a control pair would bound. The head
chose K-B reported, not gated (§0.6); the job also reports the CPU fp32 norm beside it (`max_rel_fp32_cpu`, `n_exact_fp32`).
The bank's own path is reproducible bitwise (three L40S runs, §0.2). **Failing input**: our T_(j−1) column norms against
the bank's T_j values; offline the banked neighbouring-point change of the same neuron is at least 0.00155 relative over
419 adjacent pairs (median 0.14), 12 times the tolerance (`gate_sides.json` → `K_B`). In the dry run's tiny world K-B passes
at 9 of 9 banked points (largest 1.9e-7; the neighbouring point 0.0017–0.0084) and a bank moved by 0.1 % is reported failing
at 9 of 9 points while the job completes.

### 3.2 The identity pair I-ID

At T_15 the MLP tensors are read from disk a second time: every norm bitwise equal to the first read, and the change of
T_15 against itself exactly 0 for every neuron. Failing input: the change of T_14 against T_15 (non-zero). Dry run: a
second read altered by 1 % in one tensor stops the job (× 1.0001 rounds back to the same bf16 value and does not, the
N13 A30 lesson).

### 3.3 The structured controls: the twins (the boundary of every claim)

- **dt, the distinct-token twin** (N12, M2's registered follow-up): another grid pole of the birth's layer, a group of its
  own T_15 top-50 of the birth's size and the birth's number of distinct tokens, dated k by the same rule. 140 of 153
  births have one (73, 67). Its activation crossed its threshold in the same bracket by construction, so its neuron
  carries the dating rule's selection; it lacks a coherent cluster and the birth's lateness.
- **pb, M2's twin**: the same without the token match (149). Real − pb is reported.
- The labels are real − dt paired by birth; real − twin is the only contrast that removes the selection.

### 3.4 The positive control PC

At every bracket j ≥ 2 and every layer, 32 seeded neurons (`n1.PC_SEED`) get a planted change added to their real change:
a seeded random direction scaled to the layer's 99th percentile of r in that bracket times the neuron's own norm, on the
input rows and on the column separately. Pass: at least 0.9 of the planted neurons land above the layer's 95th percentile of
z, on both sides, at every bracket. Failing input: the same with a planted size 0 (the share must stay below 0.9; tiny world
0.03–0.09). It shows the statistic resolves a change of the layer's 99th-percentile size against the layer at every bracket.
It validates the arithmetic on real weights, not a biological effect; the critic asked for it ("plant a known rank-one change
in one row").

### 3.5 The floors and the ceiling

- **Selection floor**: the dt twin, paired (the labels).
- **Matched floor**: for each real birth its 98–100 activation-matched neurons' e over the same window; a replicate draws
  one per birth and takes the mean; 1,000 replicates (seed 0); the top of the floor is its 97.5th percentile, frozen on the
  freeze half; at the verdict the bar is the larger of the freeze and test values. ABOVE_FLOOR when the real mean's lower
  bound exceeds it (reported beside each label).
- **Ceiling**: per side, the median over real births of the mean over the window of the layer's 99th percentile of z, the
  e a neuron with a 99th-percentile change in every window bracket and an average history would show. A real mean whose
  lower bound exceeds it reads ABOVE_CEILING: DOES NOT COUNT until inspected (a whole-row change beyond the layer's 1 % in
  every bracket for the average birth would point at a tensor or index error).

### 3.6 Read with one direction, edit along the other; monitoring level

The rows are the read directions and the column the written direction (`12_SHAPING_NOTE.md` §4 item 4, the mapping ours):
kept as two quantities, never summed. Nothing is edited. Every quantity is the L1 type (one neuron at a fixed identity); a
NEITHER says nothing about a direction spread over neurons.

---

## 4. Gates

| gate | where | pass | registered failing input; shown failing where | if it fails |
|---|---|---|---|---|
| G-SAME | start | torch 2.12.0 | a version "0" (dry run) | stop |
| INPUTS | start | `units.json` `ada35f38…`, `kb_bank.json` `401c6134…`, `n6_colnorms_t15.npy` `dff570e3…` | a zero sha256 (dry run) | stop |
| G-REV | each checkpoint | the snapshot's commit is the registered revision | a zero revision (dry run: stops at `t02_G_REV`) | stop |
| G-SHAPE | each checkpoint | 96 tensors, `[N, H], [N, H], [H, N]`, bf16 | the down tensor transposed (in job; dry run: a transposed file stops at `t03_G_SHAPE`) | stop |
| K-B | T_5 … T_15 | §3.1 | §3.1 (offline, every banked neuron; dry run) | reported, not gated (§0.6): the values and the failing input's are recorded |
| PC | brackets 2 … 15 | §3.4 | planted size 0 (in job, every bracket) | stop |
| I-ID | T_15 | §3.2 | §3.2 (dry run) | stop |
| TRI | brackets with banked pairs | reported, not a gate: our relative column change ≥ the banked relative norm change (the triangle inequality); the dry run showed its failing input (other neurons' norm changes) has no power | — | — |
| budget | after each checkpoint | under 0.95 of the job's budget | — | stop (no parking) |
| usable | freeze, verdict | ≥ 50 real units and ≥ 50 dt pairs per half | (dry run: 30 dt pairs reads DOES NOT COUNT) | DOES NOT COUNT |
| ceiling | verdict | §3.5 | (dry run: a planted 4.0 reads DOES NOT COUNT) | DOES NOT COUNT until inspected |
| freeze records unchanged | verdict | sha256 | one byte appended (dry run) | the verdict refuses |

---

## 5. Labels and the verdict, as the script runs it

`python -m infra.pg.n1 verdict --bank studies/next/n1/banks/<attempt> --thresholds studies/next/n1/thresholds.json --out studies/next/n1/verdict_n1.json`

1. The freeze records' sha256 must match. 2. Gates (`m2.gates_ok`); else DOES NOT COUNT. 3. ≥ 50 real test units.
4. Per side (in, out): `sel_<side>` = ABOVE / AT / BELOW_SELECTION from the paired real − dt interval (NOT_RUN below 50
pairs); `floor_<side>` = ABOVE_FLOOR or NOT_ABOVE_FLOOR against max(freeze, test) q975; ABOVE_CEILING replaces the selection
label if the real mean's lower bound exceeds the ceiling. 5. A NOT_RUN or ABOVE_CEILING side makes the headline DOES NOT
COUNT. 6. The hypothesis from the two selection labels (`n1.hypothesis`). 7. Reported: real − pb, gate and up alone, e_k,
the profiles. Every value is written whatever the headline. The dry run produced every hypothesis, the floor label, the
ceiling, the missing-twin case and both refusals on synthetic records.

---

## 6. The split, freeze 2 and room

**The split**: M2's (carrier's neuron even → freeze; 82 births, 73 dt pairs; odd → test, 71, 67). Twins follow their birth.
**Read before this spec**: the onsets of all units (dates, both halves; they set the window, not a threshold) and the banked
column norms of the freeze half (§0.4). Nothing of the test half's weights.

**Freeze 2** (`n1.freeze`) refuses unless the gates pass, the freeze half has ≥ 50 real units and ≥ 50 dt units, and the room
check passes. It writes the matched floors (q975 per side), the freeze half's summary labelled DISCOVERY and the sha256s.

**Frozen on control data**: the matched floors (freeze half); K-B's tolerance from the fp32 bound; PC's planted size from each
bracket's own 99th percentile. **A priori, with reasons**: the window k−2 … k and the reference 2 … k−3 (from the pre-check,
before any weight was read); PC's 0.9 and 32 neurons; the paired labels' zero; the ceiling's definition.

**Room**: at 67 pairs every selection label and every hypothesis is reachable on planted values (`n1.room_check`); the interval
half-width is about 0.24 of the per-birth spread of the paired difference, so a difference of a third of that spread is
resolved. The spread itself is unknown until the weights are read.

---

## 7. Unblinding order

1. Freeze 1 after the head's review (`N1_DRYRUN_OK` on the frozen commit; tag `n1-freeze-1`).
2. The one launch, `weights-01`. During the run only `n1/metrics.json` is read.
3. Pull `metrics.json`, `n1_freeze.jsonl`, `samples/raw_examples_freeze.jsonl`. Open the examples and answer in
   `studies/next/n1/JOURNAL.md`: does each unit's column norm at T_(k−1) and T_k equal the banked one; does the profile of r over
   the 15 brackets have the bracket lengths' shape for twins and carriers alike; is the planted PC share near 1 and the size-0
   share near 0.05; one birth with a large and one with a small input-side excess, with their exemplars. Then `n1 freeze`,
   commit, tag `n1-freeze-2`.
4. Pull `n1_test.jsonl`; the verdict once.
5. A fresh Opus reader, its worktree made from a commit whose subject names no label, recomputes r, z, e and the labels from
   `stats.npz` with its own code; the head's read is the different-model read.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | a whole-vector change is mostly the optimizer's rotation | a job-specific change inside it is invisible | z within the bracket, e against the neuron's own history, the twins; NEITHER is read as "not a whole-vector event", not "no change"; the projection is the next instrument |
| 2 | the onset spreads over brackets | the window may start after the rise began | the window spans three brackets; the profile k−2 … k+1 is reported; the finer location is costed (§2.1) |
| 3 | brackets differ 100-fold in steps | r differs by bracket | z is within the bracket; e compares a neuron with itself |
| 4 | the dating rule reads the input side | the input rows look changed by selection | the dt twin carries the same selection (§3.3) |
| 5 | real births are later than chance; twins are not | lateness, not coherence | stated (M2 §3.3); dt matches tokens, not lateness |
| 6 | matched neurons are matched on activation size at T_(k−1), not on norm | the floor differs in norm | e removes a neuron's lasting scale |
| 7 | 153 births on 140 carriers; one neuron's two births share its weights | dependence | stated; halves share no neuron |
| 8 | K-B compares two paths | a false stop | K-B is reported, not gated (§0.6); the bound holds for every fp32 order |
| 9 | one model, one run | no replicate | stated |
| 10 | births are gradual rises, not steps at the grid's resolution: 93 of 153 births hold less of the rise in the dated bracket than in the two before it; one checkpoint before birth the median ratio to the threshold is 0.84 (§2.1) | the birth bracket alone would miss most of the rise; a weight change that precedes the activation's rise by more than two brackets falls in the reference | the registered readout is the window k−2 … k against brackets 2 … k−3; e_k and the profile k−2 … k+1 are reported; the onset inside a bracket is not located (§2.1, not registered) |

---

## 9. Runbook

### 9.1 Checklist before the launch

- `.claude/skills/aws-pg/SKILL.md`; the docstrings of `pipeline.py` and `n1_weights.py`. Profile `fuelos`, region `us-east-1`.
- Liveness: `./infra/aws_pg/pg ps`, or `updated_utc` of `out/weights/n1/metrics.json`. Never poll `job.log`.
- Cost and incidents: `infra/aws_pg/operations/2026-11-<dd>_next_n1.md`; science: `studies/next/n1/JOURNAL.md`.
- The instance is CPU only: it uses the Standard on-demand quota (16 vCPU), not the G quota. Its hourly price is not banked;
  the executor reads it from the AWS price list and records it in the operations note.

### 9.2 Code (NEW or REUSED, checked against the tree at `ae8dc637f`)

| file | status | role |
|---|---|---|
| `infra/pg/n1.py` | NEW | onset rows, units, r, z, e, records, floors, labels, hypotheses, room, freeze, verdict, projection |
| `infra/aws_pg/jobs/n1_weights.py` | NEW | the job |
| `infra/aws_pg/jobs/n1_dryrun.py` | NEW | the offline gate |
| `infra/aws_pg/steps/next_n1_weights-01.json` | NEW | the step list |
| `studies/next/n1/onsets/onsets.py`, `offline_gate/gate_sides.py`, `samples/raw_examples.py` | NEW | pre-check, bank sides, raw examples |
| `infra/pg/m2.py` (`births_matrix`, `POINTS`, `STEPS`, `paired`, `gates_ok`, `read_jsonl`, `parse_pole`, `sha256_file`, `BANK_*`), `infra/aws_pg/jobs/m2_score.py` (`download`, `Run`, `HUB`), `m2_dryrun.py` (`tiny_cfg`, `tiny_model`, `make_remote`, `fake_hub`, `last_error`), `infra/aws_pg/jobs/neurons_harvest.py` (`fetch`) | REUSED, unchanged | dating, loaders, the stand-in hub |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-bio2
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n1 units --out studies/next/n1/units.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n1/onsets/onsets.py --out studies/next/n1/onsets
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n1/offline_gate/gate_sides.py --out studies/next/n1/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n1/samples/raw_examples.py --out studies/next/n1/samples
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n1 project > studies/next/n1/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n1_dryrun --hf $SP/m2/hf \
    --sides studies/next/n1/offline_gate/gate_sides.json --out $SP/bio2/n1_dry
```
It must end with `N1_DRYRUN_OK`; the record is `N1_OFFLINE_GATE.md`.

### 9.4 S3 layout

```
results/next_n1/inputs/{units.json, kb_bank.json, n6_colnorms_t15.npy}
results/next_n1/weights-NN/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_n1/weights-NN/out/weights/n1/{metrics.json, stats.npz, n1_freeze.jsonl, n1_test.jsonl}
results/next_n1/weights-NN/out/weights/n1/samples/{raw_examples_freeze.jsonl, raw_examples_test.jsonl}
```
`results/next_n1/` listed empty on 2026-11-06.

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
```
**S1. Freeze 1 (laptop).** §9.3; the head's review; commit; tag `n1-freeze-1`; `pg ps`, `pg audit`. Upload and read back
(`git lfs pull` the `.npy` first):
```
for f in units.json kb_bank.json inputs/n6_colnorms_t15.npy; do aws s3 cp studies/next/n1/$f \
  $B/results/next_n1/inputs/$(basename $f) --profile fuelos --region us-east-1; done
```
**S2. The launch (`weights-01`; 0.82–1.71 instance-h; job budget 120 min; hard kill 134 min).**
```
aws s3 cp infra/aws_pg/steps/next_n1_weights-01.json $B/results/next_n1/weights-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n1/weights-01/steps.json" \
    --no-weights --type r7i.2xlarge,r6i.2xlarge,m7i.2xlarge --s3-prefix results/next_n1/weights-01 --budget-min 134 --json
./infra/aws_pg/pg watch results/next_n1/weights-01
```
The first gates to read: `t00_G_REV`, `t00_G_SHAPE`, then `t02_PC`, `points.<t>.seconds`; K-B under `reports` → `t05_K_B`.

**S3. Freeze 2**, **S4. Unblind**, **S5. The reader and the head**: §7.

### 9.6 Budget and stop rule

| launch | contents | instance-h center | high | hard kill |
|---|---|---|---|---|
| `weights-01` (CPU, r7i.2xlarge) | 16 checkpoint downloads (234 GB), fp64 statistics of 96 tensors each, K-B, PC, I-ID, records | 0.82 | 1.71 | 134 min |

Rates (`n1.RATES`): download 90–150 s per checkpoint (M2 and N12 measured 45–82 s on g6e, N6 152 s, once 368 s); read and
compute 60–180 s (estimated: 96 tensors, about 4.3 billion values, a few fp64 passes on 8 vCPU); start 300–450 s; records and
upload 180–300 s. GPU-hours: 0. The price of an r7i.2xlarge is not banked (no dollar figure is given). The job asserts after
each checkpoint that it is under 0.95 of its 120-minute budget and stops otherwise (every bracket is needed; nothing parks).

### 9.7 Artifacts

S3 (§9.4); laptop: `studies/next/n1/{units.json, kb_bank.json, inputs/, onsets/, samples/, offline_gate/, projection.json,
banks/, thresholds.json, verdict_n1.json, JOURNAL.md}`; `stats.npz` stays on S3 (the reader pulls it).

---

## 10. The ten raw examples

**Zero-GPU part, written now** (`studies/next/n1/samples/raw_examples_zero_gpu.{md,json}`): the ten freeze-half example births
and their two twins: r at T_(k−3) … T_(k+1), whether the dated bracket is concentrated, the banked column norm at T_(k−1)
and T_k with its rank among the layer's 22,016 poles and its relative change, one exemplar and the first matched random pole's
top context.

What they show:
- The onset: 5 of 10 real births are concentrated; L10N9052+ (NPR "Copyright") climbs 0.000 → 0.070 → 1.456 → 4.652 over
  T_12 … T_15, L20N5782+ sits at 0.87–1.20 for three checkpoints before its birth (an earlier crossing).
- The column norm across the birth bracket changes with the bracket, for carriers and twins alike: k = 15, 0.278–0.283;
  k = 14, 0.25–0.28; k = 12, 0.04–0.05; at k = 10 the carrier L10N9794+ 0.002 against 0.044 for its twins. The layer rank of
  the norm moves in both directions (L07N8340+ 6,934 → 2,240; L16N1206+ 6,670 → 9,630).
- The matched random poles' top contexts are unrelated token families (motivation, NASA, Raspberry Pi in Japanese text).
- PENDING (the job writes them): r and cos of the rows and the column per bracket, z, e.

**GPU part**: none; the job's `samples/raw_examples_<half>.jsonl` carries the same births' full records (real, dt, pb).

---

## 11. The three dimensions

- **Predictive.** H_rows is the card's claim, scored once on the test half against a selection-matched twin under rules frozen
  on the other half. The pre-check already refuted the sharp-onset reading of a birth at the grid's resolution (§2.1).
- **Novel.** Whole-vector weight change at dated births of a 7B model, against selection-matched twins and activation-matched
  neurons, was not found in the searches; wdd's results are on Pythia and OLMo-1B and use other definitions.
- **Practical.** Small. If a side changes beyond selection, a checkpoint diff names the neurons acquiring jobs without a
  forward pass (CPU only); if NEITHER, that shortcut is ruled out at this resolution.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

Filled 2026-11-07 from `studies/next/n1/verdict_n1.json` (labels, test values) and `banks/weights-03/n1/metrics.json` (gates, K-B, PC, I-ID).

| item | rule or prior | observed | verdict |
|---|---|---|---|
| G-REV, G-SHAPE at 16 checkpoints | (0.97) | 16 of 16 and 16 of 16 pass; failing inputs fail (`metrics.json`) | as predicted |
| K-B at 11 points (reported) | ≤ 1.3e-4 (0.95) | 11 of 11 pass; largest max_rel 1.49e-07 (t15, 352,464 columns); failing input 0.0618–0.793 (`metrics.json` → `reports`) | as predicted |
| PC at 14 brackets | ≥ 0.9 (0.97) | 14 of 14 pass; planted share 1.00 at every bracket; size-0 share 0.038–0.068 | as predicted |
| I-ID | bitwise | norms bitwise True, self-change 0 True; failing input fails | as predicted |
| sel_in (test) | ROWS_FIRST 0.25 | BELOW_SELECTION: real − dt -0.432 [-0.817, -0.059], 67 pairs | H_rows refuted; a BELOW label (prior 0.1 for either side) |
| sel_out (test) | COLUMN_FIRST 0.1 | AT_SELECTION: real − dt -0.258 [-0.654, 0.130], 67 pairs | H_col refuted |
| hypothesis | NEITHER 0.45 | `NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR)` | NEITHER, but not H_0: H_sel is refuted by the input side's BELOW |
| floors | reported | in: real -0.297 [-0.550, -0.059] against the bar -0.053 → NOT_ABOVE_FLOOR; out: real -0.039 [-0.276, 0.202] against 0.147 → NOT_ABOVE_FLOOR; ceilings in 2.402, out 2.026 | reported |
| real − pb, gate and up alone, e_k, profiles | reported | real − pb in -0.500 [-0.839, -0.180], out -0.209 [-0.509, 0.086]; real − dt gate -0.387 [-0.721, -0.056], up -0.374 [-0.761, -0.001]; e_k real − dt in -0.488 [-0.955, -0.028], out -0.410 [-0.905, 0.064]; real e_in by bracket k−2, k−1, k, k+1: -0.183, -0.296, -0.412, -0.589; e_out: 0.093, -0.062, -0.148, -0.240 | reported |
| instance-h | 0.82–1.71; kill 134 min | weights-03 at most 3.59 instance-h ($1.90); N1 in all at most 4.68 ($2.48) with weights-01 0.57 and weights-02 0.52; 0 GPU-h; kill raised to 320 min under A2 | over the projection: Hugging Face downloads at 10.8–142 MB/s against 98–294 |

---

## 13. References

`06_CRITIC.md` N1, Part 0 (findings 0.5, 0.6), Part 6; `03_IDEAS.md` N1; `05_SCORING.md`; `n12/READ_head.md`; `M2_SPEC.md` §3.3;
`SPEC_D1.md` §2.4; `12_SHAPING_NOTE.md` §4; Kosson et al. (2305.17212) and wdd at the levels stated in §1.5.

---

## 14. Amendments

### A1 (2026-11-07): the raw-examples writer; weights-01 does not count

**What happened.** `weights-01` (instance `i-00a7070e189540cd3`, r7i.2xlarge, launched 2026-11-06T19:26:08Z; job
19:28:21Z–19:59:16Z, 1,855 s) passed every gate at all 16 checkpoints and wrote `stats.npz`, `n1_freeze.jsonl` and
`n1_test.jsonl`. Its last step, the raw-examples writer, read `uj["examples"][h]` from `units.json`. The registered
`units.json` (sha256 `ada35f38…`) carries only `units` and `n`; the example births live in N12's `units.json`. The job
stopped with `KeyError: 'examples'` (`studies/next/n1/diag/weights-01/job.log` line 227); `STATUS` `exit=1`;
`metrics.json` → `complete` false; `samples/raw_examples_freeze.jsonl` 0 bytes. The dry run did not catch it because its
planted `units.json` carried an `examples` key that the real file does not.

**The fix (code only, no rule).** `n1.EXAMPLES` holds the ten example births per half, copied from
`studies/next/n12/units.json` → `examples` (N12's rule: freeze, 5 with A_own ≥ 1 and 5 with A_own = 0, `default_rng(0)` on
sorted ids; test, 10 by `default_rng(1)` on sorted ids). `n1_weights.registry()` carries them as `examples`; the writer
reads `reg["examples"][h]` and, for each birth, writes the records of the real unit and of its dt and pb twins where they
exist (`L03N7746+:c0` has no twin). Nothing else in the job, in `n1.py` or in any rule changed; `units.json`,
`kb_bank.json` and the N6 array are the same inputs with the same sha256.

**The plant.** The dry run's tiny `units.json` now carries exactly the real file's keys (`n`, `units`); its registry
carries the tiny world's example births. Run against the frozen job code (`n1-freeze-1`), the dry run's JOB stops with
`KeyError: 'examples'`, as weights-01 did (`studies/next/n1/offline_gate/dryrun_a1_plant_on_freeze1_job.log`). On the
fixed code: `N1_DRYRUN_OK`, 46 checks, 0 failed (`dryrun_a1.log`, `dryrun_a1_summary.json`); two checks are new (the
registered example births equal N12's and are real units of their half; the tiny `units.json` has the real file's keys)
and the 44 earlier CHECK lines are byte-equal to freeze 1's.

**weights-01 DOES NOT COUNT.** Its records are complete up to the examples and, if the job is deterministic, equal to
what weights-02 will write; the frozen rule still refuses them: `n1 freeze` and `n1 verdict` read the gates through
`m2.gates_ok`, which requires `metrics.json` → `complete` true, and weights-01's is false. The rule is not bent for a run
that did not reach its end. Its records were not opened.

**The relaunch.** `weights-02` on tag `n1-freeze-1a`, same command, same budget (job 120 min, hard kill 134 min), fresh
prefix `results/next_n1/weights-02/`. §7 and §9.5 read `weights-02` wherever they name `weights-01`.

### A2 (2026-11-07): the job budget; weights-02 killed

**What happened.** weights-02 (`i-0726baabb83baa17a`, r7i.2xlarge, launched 15:23:20Z on `n1-freeze-1a`) passed every gate
it reached, but the checkpoint downloads from the HF hub ran at 24.2–35.9 MB/s (T_0 406.5 s, T_1 504.8 s, T_2 602.8 s at 24.2 MB/s)
against 98.6–294.1 MB/s in weights-01. The read and the statistics take about 27 s per checkpoint, so the download is the
whole cost. The job asserts after each checkpoint that it is under 0.95 of its 120-minute budget (6,840 s) and writes
nothing before the last checkpoint. The rule written in the journal before T_2's time was read (15:44Z): if elapsed +
13 × the mean seconds per checkpoint of T_0–T_2 + 200 s exceeds 6,840 s, kill and relaunch with only the budget changed.
It read 8,725 s. weights-02 was killed at 15:54:35Z (`pg kill --force`; the launcher refuses a job whose counters show work without it); it DOES NOT COUNT (incomplete).

**The change (runbook only).** Job budget 120 → 300 min (`--budget-min 300`; the stop at 0.95 × 300 min = 17,100 s
allows 1,050 s per checkpoint, a download at 14 MB/s); hard kill 134 → 320 min. Steps file
`infra/aws_pg/steps/next_n1_weights-03.json`. No code, no input, no rule of the measurement changed. Instance-hours at
weights-02's rate: about 2.4–2.9 (16 × 530–650 s plus start and end); at most 5.33 at the hard kill, $2.82 at $0.5292 per hour.
The launch is `weights-03` on tag `n1-freeze-1b`.
§7 and §9.5 read `weights-03` wherever they name `weights-01`; §9.6's budget row reads job 300 min, hard kill 320 min.
