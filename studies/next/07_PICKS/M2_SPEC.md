# M2: name the carrier before the birth. Pre-registration, frozen at `m2-freeze-1`, amended (A1, its revision, A2, A3, A4, A5) and re-frozen at `m2-freeze-1g`

**Answer.** **The verdict of 2026-11-06 (`score-06`, the test half's 71 births, rules frozen at `m2-freeze-2`):
`H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES`** (`studies/next/m2/verdict_m2.json`). One
checkpoint before the birth, the carrier is in its layer's top 10 of 22,016 poles under B for 42 of 71 births
(0.592, Wilson 95 % [0.475, 0.698]; bar 0.0122, the matched-pole floor). A random group of another grid pole's own
future top contexts, of the same size and dated to the same point, puts its pole in the top 10 more often: 53 of 70
(0.757); the paired difference is −0.171 [−0.286, −0.057]. Under A the carrier is named for 13 of 71 (0.183
[0.110, 0.288]), and A loses to B by −0.408 [−0.521, −0.296]. Every gate passed and every failing input failed (52
gates). The rest of this paragraph is the record before the verdict.

The pre-check passed: 153 births pass J1's recount, against the 100 the plan requires
(`studies/next/m2/births.json` → `n_births`; the rule is p_late ≤ 0.05 under R1t, 153 against 22.4 expected at chance,
`studies/next/10_recount2/out/05_chance.json` → `summary.R1t.all`). M2 needs one GPU launch, projected at
1.56–1.72 GPU-hours on one L40S (`studies/next/m2/projection.json`), against the plan's 5–8. The primary score is
**B, the pole's own current percentile** of the job's median exemplar activation at T_(k−1), and H_B is the headline
(head's decision, 2026-11-05, before freeze 1, §2.3). B needs one forward pass over the job's harvest windows, and
that pass is the GPU cost. The reference is **A, top-K overlap**: the number of the job's exemplar positions among a
pole's 50 top positions at T_(k−1), read from the banked harvest at zero GPU. The test half is 71 births, the freeze
half 82, split by neuron parity. The weight-free dry run printed `M2_DRYRUN_OK` after 78
checks, 0 failed (`studies/next/m2/offline_gate/dryrun.log`), and again after the head's swap of the primary score
(`offline_gate/dryrun_2026-11-05_primary_B.log`, 78 checks, 0 failed). No gradient is needed: no score uses one.
The first launch, `score-01`, stopped on its own G-EX gate at T_14 and does not count. Amendment A1 (§3.8, head's
decision 2026-11-05 20:15 CST) replaced the known-number statistic with |a_new − a_bank| / (0.05 + |a_bank|) and froze
its tolerance on the banked T_14 device pair: 0.042318 across devices, 0.021159 on the same device. The dry run then
printed `M2_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_2026-11-05_A1.log`). The head's revision of A1
(2026-11-05 20:50 CST, §3.8) sets the cross-device tolerance from the pair's entries with |a_bank| < 0.5: 0.044273. The
same-device tier stays 0.021159. The dry run printed `M2_DRYRUN_OK`, 83 checks, 0 failed
(`offline_gate/dryrun_2026-11-05_A1rev.log`). The second launch, `score-02`, stopped on G-EX at T_13 (p99 0.05433
against 0.044273) and does not count. Amendment A2 (§3.8, head's decision 2026-11-06 03:45Z) gates the cross-device
G-EX on the entries with |a_bank| ≥ 0.1 only, at twice the control pair's p99 on that subset: 0.038718. The dry run
printed `M2_DRYRUN_OK`, 86 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A2.log`). The third launch, `score-03`,
stopped on G-EX at T_13 under A2 (591 gated entries, p99 0.05191 against 0.038718) and does not count. **M2 has no
verdict** yet: all three launches stopped on the known-number gate at a Blackwell-banked point, before any test record
was opened. Amendment A3 (§3.8, head's decision 2026-11-06 04:20Z) makes G-EX at the Blackwell-banked points T_7–T_14 a
reported readout, not a gate; G-TOPK, K0, I-ID and G-EX at the same-device points stay gates. The dry run printed
`M2_DRYRUN_OK`, 89 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A3.log`). `score-04` is the last attempt.
`score-04` passed every gate through T_10 and then crashed in the scoring code at T_10: a banked pole there
(L24N9826+) has no token of its own sign, which `m2.percentile` asserts never happens. It does not count, and by the
head's rule M2 ends without a verdict. Spent: 1.957–2.042 GPU-h over four attempts. The head then decided amendment A4
(§3.8, 2026-11-06 05:35Z) and one last relaunch: a pole with no token of its sign at a point gets B = 0 (and ratio 0),
ranks last and is not a matched-pole candidate. A scan of all 352 banked stats files at T_5–T_15 found 7 such entries
(5 poles), 2 in layers the run scores (L07N640− at T_8, L24N9826+ at T_10). The dry run printed `M2_DRYRUN_OK`, 112
checks, 0 failed (`offline_gate/dryrun_2026-11-06_A4.log`). `score-05` passed every gate through T_7 and wrote the
predicted degenerate-pole counts, then stopped at T_6: G0's registered failing input (the activation swapped) has no
power at early checkpoints, max ratio 0.0097 under the 0.01 bar (0.0062 at T_5 in the banked harvest's own controls).
It does not count, and by the head's rule M2 ends without a verdict. Spent: 2.683–2.798 GPU-h over five attempts.
The head then decided amendment A5 (§3.8, 2026-11-06 06:50Z) and one last attempt: G0's failing input becomes u scaled
by 1.1, which gives at least 0.089 on every layer wherever the gate passes (0.0944–0.1052 in the banked controls at all
11 points); the swap is run and reported, not gated. Both sides of every gate the job runs were evaluated offline at
every point from the banked files (`offline_gate/a5_gate_sides.json`). The dry run printed `M2_DRYRUN_OK`, 130 checks,
0 failed (`offline_gate/dryrun_2026-11-06_A5.log`). `score-06` is the last attempt.
`score-06` completed on `m2-freeze-1g`: 52 gates passed, every failing input failed, nothing was parked. Freeze 2 was
written from the freeze half and tagged `m2-freeze-2`; the verdict ran once (§12). Spent: 0.641–0.669 GPU-h for
`score-06`, 3.324–3.467 GPU-h over six attempts.

Written 2026-11-05 by an Opus worker in worktree `interp_research-m2`, branch `m2` (from `clean-neurons` at
`d604f8f13`). No model was run, no weights were downloaded, nothing was launched. Network reads: the 16 banked grid
`manifest.json` files and the device-check JSON from S3 (small files); two 4 KB range reads of zip directories on S3;
`config.json` and `model.safetensors.index.json` of 11 branches from Hugging Face; four web searches and one page read
(§1.5).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | **Verdict run once (2026-11-06T07:55Z): `H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES`** (`studies/next/m2/verdict_m2.json`, §12). Frozen at freeze 1 (the head reviewed the draft and chose the primary score, §2.3, before the tag), then at 1b, 1c, 1d, 1e, 1f and 1g; freeze 2 at `m2-freeze-2`. Six launches: three stopped on G-EX at a Blackwell-banked point, the fourth crashed in the scoring code at T_10 (an empty pole), the fifth stopped at T_6 because G0's failing input passed, the sixth (`score-06`, freeze 1g) completed. No test record was opened before freeze 2 was tagged. The independent recount and the different-model read (§7 step 5) are not done. |
| freeze 1 | `m2-freeze-1`: this spec, `infra/pg/m2.py`, `infra/pg/m2_examples.py`, `infra/aws_pg/jobs/m2_score.py`, `infra/aws_pg/jobs/m2_dryrun.py`, `infra/aws_pg/steps/next_m2_score-01.json`, `studies/next/m2/births.json` (sha256 `fdca794b…`, registered as `m2_score.BIRTHS_SHA`). The offline gate must print `M2_DRYRUN_OK` first. Tagged by the executor on the head's instruction of 2026-11-05: annotated tag `m2-freeze-1` (tag object `4027440cbea0`) on commit `35ff9d7ce6f8c33853e95cc7a2208fa6ff9aa649`, made 2026-11-06T01:42:35Z. The dry run before it printed `M2_DRYRUN_OK`, 78 checks, 0 failed (`offline_gate/dryrun_2026-11-05_primary_B.log`). |
| attempt `score-01` | 2026-11-06: launched on `m2-freeze-1`; K0 bit-identical; stopped at `t14_G_EX` (p99 0.2563 against 0.0444). **DOES NOT COUNT: gates.** No freeze 2, no verdict. Cause and record: `studies/next/m2/JOURNAL.md`, `studies/next/m2/diag/`. The head's decision on the gate: amendment A1 (§3.8). |
| amendment A1 | 2026-11-05 (head's decision 20:15 CST), before freeze 1b: the statistic of K0, G-EX, G-TOPK and I-ID becomes \|a_new − a_bank\| / (0.05 + \|a_bank\|); the tolerance on its p99 is frozen on the banked T_14 device pair (0.021159 same device, 0.042318 across devices); the pass side across devices enters the offline gate; the headline lists H_B first. §3.8. |
| freeze 1b | `m2-freeze-1b`: the files of freeze 1 with A1 applied (§3.8). Annotated tag `m2-freeze-1b` (tag object `b78c1c5d0652`) on commit `1584962e0bf0a9d736c817fc73b4682a8d7c972b`, made 2026-11-06T02:26:59Z by the second executor on the head's instruction. The dry run before it printed `M2_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_2026-11-05_A1.log`). |
| attempt `score-02` | 2026-11-06 02:27–02:45Z, on `m2-freeze-1b`: **not launched**. Six tries 180 s apart on `--type g6e.2xlarge,g6e.4xlarge` each failed on `InsufficientInstanceCapacity` in every AZ; by the capacity rule the executor stopped and reported to the head. No instance, 0 GPU-h. The prefix holds only `steps.json`. Record: `infra/aws_pg/operations/2026-11-05_next_m2_score.md`. No freeze 2, no verdict. |
| revision of A1 | 2026-11-05 (head's decision 20:50 CST), before freeze 1c: the cross-device tolerance becomes twice the pair's p99 of d over the entries with \|a_bank\| < 0.5 (0.022137), so 0.044273; the same-device tier stays 0.021159. §3.8, "Revision of A1". |
| freeze 1c | `m2-freeze-1c`: the files of freeze 1b with the revision applied. The dry run before it printed `M2_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_2026-11-05_A1rev.log`). Annotated tag `m2-freeze-1c` (tag object `fe8a24c026fc`) on commit `6fa5c46dd9f5656a7997a396d78c80138c6fa295`, made 2026-11-06T02:56:51Z by the third executor on the head's instruction. |
| attempt `score-02` (freeze 1c) | 2026-11-06, on `m2-freeze-1c`, `i-0a72805bc626c095d` (g6e.4xlarge, us-east-1a), launched 03:09:05Z on try 4 of the capacity rule. K0, T_15 G-EX and G-TOPK bit-identical; T_14 G-EX passed (p99 0.040338 against 0.044273); **stopped at `t13_G_EX`** (717 entries, p99 0.05433 against 0.044273, median 0.00431, max 0.222). **DOES NOT COUNT: gates.** About 0.43–0.45 GPU-h. No freeze 2, no verdict. Record: `studies/next/m2/JOURNAL.md`, `diag/score02_t13_gex.json`, the operations note. The head's decision on the gate: amendment A2 (§3.8). |
| amendment A2 | 2026-11-06 03:45Z (head's decision), before freeze 1d: the cross-device G-EX gates the entries with \|a_bank\| ≥ 0.1 only (those below reported, not gated), at twice the control pair's p99 of d on the same subset (0.019359), so 0.038718. G-TOPK (0.044273), K0, I-ID and the same-device G-EX (0.021159) unchanged. §3.8 "A2". |
| freeze 1d | `m2-freeze-1d`: the files of freeze 1c with A2 applied. The dry run before it printed `M2_DRYRUN_OK`, 86 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A2.log`). Annotated tag `m2-freeze-1d` (tag object `a97fc9102ea6`) on commit `b37c0b9dfc1f4cf625ca93d5e91fd8afe19b8eca`, made 2026-11-06T03:48:59Z by the third executor on the head's instruction. |
| attempt `score-03` (freeze 1d) | 2026-11-06, on `m2-freeze-1d`, `i-0c403790a9756a0b7` (g6e.2xlarge, us-east-1d), launched 03:49:45Z on the first try. K0, T_15 G-EX and G-TOPK bit-identical; T_14 G-EX under A2 passed (406 gated, p99 0.028857, the repeat's value, against 0.038718); **stopped at `t13_G_EX`** (591 gated of 717, p99 0.05191 against 0.038718, max 0.0766; below 0.1: 126, p99 0.0637). **DOES NOT COUNT: gates.** No freeze 2, no verdict; no fourth attempt without the head. Record: `studies/next/m2/JOURNAL.md`, `diag/score03_t13_gex.json`, the operations note. |
| amendment A3 | 2026-11-06 04:20Z (head's decision), before freeze 1e: at the points banked on the other device (T_7–T_14) G-EX is reported, not gated (the A2 statistic, the count of read entries above 0.038718 and their list, in `metrics.json` → `reports`); the job continues. G-EX at T_5, T_6, T_15 (same device), G-TOPK at every point, K0 and I-ID stay gates. The verdict carries the line "cross-device G-EX exceeded the A2 tolerance at points …". If any gate stops `score-04`, M2 stops for good. §3.8 "A3". |
| freeze 1e | `m2-freeze-1e`: the files of freeze 1d with A3 applied. The dry run before it printed `M2_DRYRUN_OK`, 89 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A3.log`). Annotated tag `m2-freeze-1e` (tag object `a75762d13b7f`) on commit `ae2a6a43f9c4bbe0e1dc3fe759fa9bed88b7a36f`, made 2026-11-06T04:20:35Z by the third executor on the head's instruction. |
| attempt `score-04` (freeze 1e) | 2026-11-06, on `m2-freeze-1e`, `i-0cfccda25b48ab347` (g6e.2xlarge, us-east-1c), launched 04:27:51Z on try 3. Every gate passed and every failing input failed through T_10 (K0 bit-identical; I-ID bit-identical, 0 flips; G-TOPK at T_15–T_10 p99 0–0.0158). G-EX reported (A3): p99 0.0289 (T_14), 0.0519 (T_13), 0.0325 (T_12), 0.0640 (T_11), 0.0430 (T_10). **Crashed in the scoring code at T_10**: `m2.percentile` asserts every pole has a token of its sign, and L24N9826+ has none in the banked T_10 histogram (also L07N640− at T_8; `diag/score04_empty_poles.json`). **DOES NOT COUNT: incomplete.** No freeze 2, no verdict; no relaunch (the head's rule). 0.903–0.926 GPU-h. |
| amendment A4 | 2026-11-06 05:35Z (head's decision), before freeze 1f: a pole with no token of its own sign at a point (empty histogram, max_abs 0) gets B = 0 and ratio = 0 there, ranks last and is not a matched-pole candidate; A unaffected; the count per point in `metrics.json` and the verdict. The offline gate scans all 352 banked stats and topk files at T_5–T_15 (`offline_gate/bank_scan.json`) and plants such a pole in the tiny world. §3.8 "A4". |
| freeze 1f | `m2-freeze-1f`: the files of freeze 1e with A4 applied, and `studies/next/m2/offline_gate/bank_scan.py`. The dry run before it printed `M2_DRYRUN_OK`, 112 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A4.log`). Annotated tag `m2-freeze-1f` (tag object `0c3d5aa08989`) on commit `8599e326176902fe88fd4e55824535ec2103f6b3`, made 2026-11-06T05:45:28Z by the fourth executor on the head's instruction. |
| attempt `score-05` (freeze 1f) | 2026-11-06, on `m2-freeze-1f`, `i-03bf9bcd82f6f7005` (g6e.4xlarge, us-east-1a), launched 05:57:24Z on try 4 (tries 1–3 `VcpuLimitExceeded`). Every gate passed and every failing input failed through T_7 (K0 and I-ID bit-identical, 0 flips; G-TOPK at T_15–T_7 p99 0–0.0158; T_15 G-EX bit-identical); degenerate poles as the scan predicted (T_10 L24N9826+, T_8 L07N640−, 0 elsewhere). G-EX reported over 0.038718 at T_13, T_11, T_10, T_9, T_7. **Stopped at `t06_G0`**: the gate passed (0.00352), but its failing input, the activation swap, passed too (0.00970 under 0.01), as the banked harvest's own controls show at T_5 (0.0062); `diag/score05_g0_fail.json`. **DOES NOT COUNT: gates.** No freeze 2, no verdict, no relaunch (the head's rule). 0.726–0.756 GPU-h; M2 total 2.683–2.798 GPU-h. |
| amendment A5 | 2026-11-06 06:50Z (head's decision), before freeze 1g: G0's registered failing input becomes u scaled by 1.1 (the banked harvest's second input; at least 0.089 on every layer wherever the gate passes, by the triangle inequality); the activation swap is run and reported (`reports` → `tNN_G0_act_swap`), not gated. Both sides of every gate the job runs, and of the verdict's gates where the bank holds them, evaluated offline at every point (`offline_gate/a5_gate_sides.{py,json,log}`). §3.8 "A5". |
| freeze 1g | `m2-freeze-1g`: the files of freeze 1f with A5 applied, and `studies/next/m2/offline_gate/a5_gate_sides.py`. The dry run before it printed `M2_DRYRUN_OK`, 130 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A5.log`). Annotated tag `m2-freeze-1g` (tag object `52679a40ba14`) on commit `9a8e1a6a1ab6c18134e49623650ea8f225f8a737`, made 2026-11-06T07:07:36Z by the fifth executor on the head's instruction. |
| attempt `score-06` (freeze 1g) | 2026-11-06, on `m2-freeze-1g`, `i-00e24f621cc4be70c` (g6e.2xlarge, us-east-1d), launched 07:12:17Z on try 2 (try 1 `VcpuLimitExceeded`). **Complete** (STATUS `exit=0` at 07:50:44Z): 52 gates passed and every failing input failed; nothing parked. G0's failing input u × 1.1 0.094–0.105 at all 11 points; the swap reported, under 0.01 at T_6 and T_5. T_15–T_7 equal to score-05's to every digit; T_6 and T_5 G-EX and G-TOPK bit-identical. Degenerate poles as the scan predicted. 0.641–0.669 GPU-h. Banks: `studies/next/m2/banks/score-06/`. |
| freeze 2 | `m2-freeze-2`: `studies/next/m2/thresholds.json` (sha256 `dce658c5…`), written by `python -m infra.pg.m2 freeze` from the job's gates and the freeze half only, before the test half's records were opened. Annotated tag `m2-freeze-2` (tag object `53e39ed89090`) on commit `8fb4318071c7abb5f41eb6a7172d170054638081`, made 2026-11-06T07:54:47Z. The freeze-half raw examples were opened and answered first (`studies/next/m2/JOURNAL.md`, 07:53Z). |
| verdict | 2026-11-06T07:54:59Z–07:55:00Z, once: `python -m infra.pg.m2 verdict --banks studies/next/m2/banks/score-06 --thresholds studies/next/m2/thresholds.json --out studies/next/m2/verdict_m2.json` (sha256 `d2800d8a…`); headline `H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES`. §12 is filled from it, from `thresholds.json` and from the job's `metrics.json` (its sha256 frozen in `thresholds.json`). |
| parents | `studies/next/09_PLAN.md` §3 card J9; `studies/next/06_CRITIC.md` M2 (lines 852–881) and line 929; `studies/next/10_RECOUNT2.md`; `studies/next/05_SCORING.md` line 126; `studies/next/03_IDEAS.md` N12; `studies/next/12_SHAPING_NOTE.md` §4 items 3–5 |
| method | repo `CLAUDE.md`; the `research-spec` skill; the format of `07_PICKS/M1_SPEC.md` |
| vocabulary | `studies/next/README.md`. A run whose gate fails is labelled **DOES NOT COUNT**. |

### 0.1 The brief's requirements and where each sits

| requirement (head's brief, 2026-11-05) | where |
|---|---|
| pre-check: at least 100 births pass J1's recount, else stop | §0.4; passed, 153 |
| the question: does a score from the harvest at T_(k−1) rank the carrier in its layer's top 10 among 22,016 poles | §1.1, §2 |
| one primary score from the four candidates, frozen; the rest exploratory | §2.3: B primary, A the reference (head's decision, 2026-11-05, before freeze 1); the other three are job-free and bounded at chance by the pole draw |
| against chance 10/11,008 per sign and against "the pole's own current percentile" | §0.5 (chance), §1.3 (H_A, H_AB), §2.2 (B) |
| thresholds frozen on control data; a split of the births | §6: neuron parity, not checkpoint parity (reason stated) |
| floor from layer- and activation-matched random poles; ceiling from the positive control at the birth point; a does-not-count branch | §3.4, §3.5, §5 |
| known number: banked `exemplar_a.npz` reproduced for 10 poles at one checkpoint before any new number | §3.1, K0 |
| identity pair: the same checkpoint scored twice | §3.2, I-ID |
| ten raw examples first, at zero GPU where possible, with a matched random-pole reading | §10; `studies/next/m2/samples/raw_examples_zero_gpu.md` |
| budget, park order, stop rule; gradients only if needed | §9.6 |
| decision table; the three dimensions scored | §1.2, §11 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| model | `allenai/Olmo-3-1025-7B`, 32 layers, MLP width 11,008, so 22,016 poles per layer | `00_ASSETS.md` §1.1 |
| the 16 grid points | T_0 … T_15 = steps 0, 1k, 2k, 3k, 5k, 8k, 12k, 20k, 35k, 60k, 100k, 180k, 320k, 560k, 1M, 1,413,814; revisions equal the banked manifests and the HF refs read of 2026-11-05 (16 of 16) | `m2.POINTS`; S3 `results/neurons/w3/grid/{1,1b}/out/<dir>/manifest.json`; scratchpad `m1_gate/hf/refs.json` |
| banked device per point | L40S for T_0–T_6 and T_15; RTX PRO 5000 Blackwell for T_7–T_14 | the same manifests, `device_name` |
| harvest settings | windows of 2,048 corpus tokens with BOS prepended, batch 8, bf16, sdpa; window position 0 excluded; 10,044 windows | `infra/pg/neurons_harvest.py` `run`; `m1.SAME` |
| banked arrays per point (S3) | `topk/layer_LL.npz` (`tok`, `a`, `g`, `u`, [11,008, 2, 50]) and `stats/layer_LL.npz` (`hist` [11,008, 2, 400], `hist_edges`, `n_pole_tokens`, `q999`, `max_abs`, `density`, …), 32 layers each; present for all 11 points M2 reads | `aws s3 ls` 2026-11-05: 32 and 32 at each of T_5 … T_15; zip directory of `1b/out/t10_step100000/{stats,topk}/layer_10.npz` read by a 4 KB range request |
| file sizes | about 11 MB per topk file and 15 MB per stats file (T_13 layers 0–2) | `aws s3 ls` |
| laptop banks | the 640 grid poles' `exemplar_a.npz` and `samples/contexts.jsonl` at all 16 points; `next/recount2/{arrays.npz,births.jsonl}` | sha256 registered in `m2.EXEMPLAR_SHA`, `m2.CONTEXTS_SHA`, `m2.BANK_SHA` |
| branch configs and shards | the 11 branches T_5 … T_15: `config.json` equals `m1.REF_CONFIG` (no differing key); the index names three bf16 shards, 14,596,022,272 bytes | `studies/next/m2/offline_gate/hf/` (HF resolve URLs, 2026-11-05); dry run REAL |
| cross-device repeat of T_14 | 32,000 exemplar entries, relative difference \|x−y\|/max(\|x\|,\|y\|): median 0.00431, p99 0.02216 | S3 `results/neurons/w3/device_check/t14_1b_vs_1c.json`, copy at `studies/next/m2/offline_gate/device_check_t14_1b_vs_1c.json` |
| rates | L40S harvest 3,215–3,334 tokens per second; T_15 download 14.61 GB at 234 MB/s | `00_ASSETS.md` §1.1; `M1_SPEC.md` §0.2 |
| prices | g6e.2xlarge $2.24 per hour, g6e.4xlarge $3.00424 per hour (one L40S each) | `09_PLAN.md` lines 13–14 |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | the AMI has torch 2.12.0+cu130 and transformers 5.9.0 | G-SAME at job start | the job stops in seconds |
| 2 | the bf16 forward on an L40S reproduces the banked T_15 exemplar values for the first 10 carriers | K0 | the job stops before any score |
| 3 | the forward of an arbitrary window subset in batches of exactly 8 equals the banked full-pass values (same shape, same kernels) | K0, G-EX and G-TOPK at every point; the CPU dry run gave exact equality (exact share 1.000) | the gate stops the job |
| 4 | the measured forward rate | the job's `measured.tokens_per_s` after T_15 | the stop rule (§9.6) acts on it |

### 0.4 Already-done audit and the pre-check

| cell | prior measurement | path | what it gives M2 |
|---|---|---|---|
| the births | three recounts of the already-clean dates agree: 0 mismatches over 2,454 clusters; 153 of 2,428 clusters (on the 574 poles with m ≥ 2) born later than random same-size groups at one-sided 0.05, against 22.4 expected | `studies/next/10_RECOUNT2.md` §4, §8; `10_recount2/out/05_per_cluster_R1t.csv` (column `p_late`) | **the unit**: the 153 passing births |
| the pre-check | 153 ≥ 100, so M2 proceeds; `M2_PRECHECK.md` is not written | `studies/next/m2/births.json` → `n_births`; dry run REAL ("pre-check", "the count equals …") | go |
| naming the carrier, D1 parent test | a judge shown 10 exemplars of a late-born cluster and two pole records at t_before (the true pole's top-20 and a random same-layer pole's, nearby contexts removed) picks the true one with accuracy A = 0.812 over 1,238 late-born clusters, against 0.924 on controls | `studies/already-clean/results/RESULTS.md` line 18 (numbers from S3 `w5/verdict/metrics.json`, not re-read) | a two-way choice by reading; M2 is a 1-of-22,016 ranking by activation |
| the same question at 410M | wdd e587: the eventual row ranks first among 53,248 rows "by its ratio over the class" two checkpoints before recruitment for 0.52 of words, in the top ten for 0.84; five before 0.72, eight before 0.62; no random-row control reported for e587 | `studies/next/04_WDD.md` line 73; https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/39_migration.md (read 2026-11-05) | the prior for B; the class there is row-free (k-means), here the job comes from the carrier's own T_15 top-50 (§8 threat 1) |
| current usage as the forecast to beat | wdd e588: the row's own current usage forecasts recruitment at AUC 0.964 and 0.968, above the class-side forecast 0.927 | `04_WDD.md` lines 74–75, 384–387 | why B is scored; B is the primary score (head's decision, 2026-11-05, before freeze 1, §2.3) |
| the carriers' own pre-birth values (freeze half only) | the carrier's median a⁺ over E_c divided by its q999 at T_(k−1): quantiles 0, 0.303, 0.581, 0.838, 0.921, 0.964, 0.998 (min, 10 %, 25 %, 50 %, 75 %, 90 %, max; 82 births). A_own ≥ 1 (an exemplar position in the carrier's own T_(k−1) top-50) for 10 of 82 | `births.json` → `freeze_look`, `examples.freeze_counts` | the only birth data read before this spec; no competitor was read, so no rank and no hit is known (§6) |

### 0.5 Objects, defined once

- **Pole** q = (layer l, neuron n, sign s); s = 0 is the a > 0 pole, s = 1 the a < 0 pole. A layer has 22,016 poles.
- **Birth**: a non-echo cluster c of a grid pole p (the **carrier**) whose birth index k (the already-clean rule:
  the first grid point where the median exemplar a⁺ reaches p's own q999 there and at the next point) passes J1:
  p_late ≤ 0.05 under R1t. k is recomputed from the banked arrays for all 153 (dry run REAL).
- **E_c**: the cluster's exemplar positions, the global token indices of p's T_15 top-50 at the cluster's ranks
  (`births.json` → `tok`). 153 births, 140 carriers, 1,234 positions.
- **a⁺_q,t(e)** = max(±a, 0) in the pole's own direction, a = act(g)·u recomputed in fp32 from the bf16 projections.
- **Roles** of a birth: **pre** = T_(k−1), the prediction point; **birth** = T_k; **end** = T_15.
- **Score A (reference; head's decision, 2026-11-05, before freeze 1), top-K overlap**: A_q(t) = |E_c ∩ top50_q(t)|,
  from the banked `topk/layer_LL.npz` → `tok`.
- **Score B (primary; head's decision, 2026-11-05, before freeze 1), the pole's own current percentile**: v_q = median
  over E_c of a⁺_q,t (numpy median; mean of the two middle values, as the dating rule); B_q(t) = the fraction of
  q's own-sign tokens at t whose |a| lies below v_q, from q's banked 400-bin histogram, bins below v's bin counted
  whole and v's bin by the log-linear share below v; B = 0 when v = 0 (`m2.percentile`).
- **Exploratory scores**: ratio = v_q / q999_q(t); max_abs; density; the down-projection column norm ‖W_down[:, n]‖₂.
  The last three do not depend on the job.
- **Rank** of pole x under a score = the number of the layer's 22,016 poles whose score is ≥ x's. Ties count against x.
  A **hit** is rank ≤ 10. **Precision at 10** of a set of births = their share of hits.
- **Chance** = 10 / 22,016 = 0.00045. The brief's per-sign figure, 10 / 11,008 = 0.00091, would apply to a ranking
  within one sign; it is reported, and the bar never falls below the larger of chance and the two floors (§5).
- **Pseudo-birth twin** (§3.3) and **matched poles** (§3.4) are defined where they are used.

---

## 1. Question, decision, hypotheses

### 1.1 The question

For a cluster born at T_k, does a score computed only from the model at T_(k−1) put the eventual carrier among the
10 best of its layer's 22,016 poles?

### 1.2 The decision this run makes

| outcome (test half) | reading | next |
|---|---|---|
| H_B SUPPORTED, H_sel_B AT_SELECTION | The carrier already leads its layer on the job's contexts one point before the birth, at the rate a random same-size group of a pole's own future top contexts, dated to the same point, gives. Naming is what the selection of E_c gives (persistence of tuning), not something specific to jobs. | J12 takes B's top 20 at T_(k−1) as N12's competitor set, with the twins as its control. No claim that jobs are special. |
| H_B SUPPORTED, H_sel_B ABOVE_SELECTION | Coherent late jobs are more nameable than the selection alone gives: a pre-birth signal specific to jobs. | The candidate result. Before any write-up: the same score at T_(k−2) (an exploratory amendment, one more forward per point) and the fresh recount. |
| H_B SUPPORTED, H_sel_B BELOW_SELECTION | Coherent late jobs land in poles that lead less than the selection predicts: other poles held the job's contexts first. | N12 (competition) moves up; its instrument (gradient per unit activation, validated against ablation, `06_CRITIC.md` N2) is specified next. |
| H_B REFUTED or UNDECIDED | Current usage does not name the carrier. | The persistence reading fails at 7B for these births; the gradient score of `06_CRITIC.md` M2 becomes the next spec. |
| H_AB TIES or BEATS | The zero-GPU score names carriers as well as the forward does. | J12 builds its competitor sets from the banks at 0 GPU-h. |
| H_AB LOSES | Naming needs the forward pass. | J12 budgets one forward per scored point. |
| DOES NOT COUNT | No claim. The failed gate and its value are reported. | fix, fresh attempt prefix |

### 1.3 Hypotheses and the observation that refutes each

All on the 71 test births, at the pre role, with the bars frozen as in §5.

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_B** (primary, the headline; head's decision, 2026-11-05, before freeze 1) | Current usage names the carrier. | lower bound of precision at 10 under B above the bar. | precision at 10 under B at or below the bar. |
| **H_sel_B** (the boundary of the claim) | The carrier of a job is more nameable than the carrier of a same-size, same-date random group of a pole's own future top contexts. | the paired difference hit_real − hit_twin under B has its interval above 0. | its interval contains 0 (AT_SELECTION: the selection explains it) or lies below 0. |
| **H_A** (reference, reported with its own label; head's decision, 2026-11-05, before freeze 1) | The carrier is predictable before birth from the banked harvest: top-K overlap puts it in its layer's top 10. | Wilson 95 % lower bound of precision at 10 under A above the bar. | precision at 10 under A at or below the bar (with a bar near chance: no test birth named). |
| **H_AB** (the reference contrast, the secondary practical question; head's decision, 2026-11-05, before freeze 1) | The free score is at least as good as the pole's own current percentile. | the paired difference hit_A − hit_B has a 95 % interval that contains 0 or lies above it. | its interval lies below 0 (LOSES). |
| **H_0** (the boring one) | The score carries no information about the carrier. | precision at the floor of activation-matched random poles. | H_A or H_B SUPPORTED. |

What these labels can and cannot show. Chance is 0.00045, so with 71 births a single hit puts the Wilson lower bound
(0.0025) above it: H_A and H_B test existence, and their room is "zero hits against at least one". The magnitude is read
from the absolute precision with its interval and from the two paired contrasts, whose intervals can fall on either
side of 0 (room shown in §6).

### 1.4 The author's prior, logged before any rank exists (2026-11-05)

| quantity | prior | basis |
|---|---|---|
| precision at 10 under A, test half | 0.04–0.12; H_A SUPPORTED with probability 0.9 | A_own ≥ 1 for 10 of 82 freeze births, an upper bound on A's hits there (`births.json` → `examples.freeze_counts`) |
| precision at 10 under B, test half | 0.3–0.7; H_B SUPPORTED with probability 0.97 | wdd e587 top ten 0.84 two checkpoints before; the carriers' median ratio 0.84 at T_(k−1) on the freeze half; 22,016 competitors |
| H_AB | LOSES with probability 0.85 | A is near 0 for most births |
| H_sel_B | AT_SELECTION 0.45, ABOVE 0.30, BELOW 0.25 | in the ten freeze examples the twin's own ratio at T_(k−1) is higher than the carrier's for 7 of 9 (§10); its rank is not known |
| construction control fires (both scores ≥ 0.90 at T_15) | 0.95 | E_c is the carrier's T_15 top-50 by definition |
| K0 bit-identical on the L40S | 0.7; within tolerance 0.97 | the M1 K0 was bit-identical on the same lane (`M1_SPEC.md` §10) |
| I-ID: zero top-10 flips | 0.9 | same weights, same shapes |

### 1.5 Literature (query terms recorded)

Searches run 2026-11-05, WebSearch: (1) `predict which neuron will learn a feature before it emerges during language
model pretraining checkpoints`; (2) `"feature emergence" forecast "before" training checkpoints neuron recruitment
language model 2026 arXiv`; (3) `"Features have life history" Stecher carriers predicted onset firing Pythia`;
(4) `which MLP neuron will specialize predicted from earlier checkpoint activations "rich get richer" neuron feature
assignment language model training dynamics`. Earlier records: `04_LITERATURE.md` (N12, queries listed there),
`04_RECENT.md` §4, `04_WDD.md`.

- Closest: wdd e587 (above), a repository page read for the e587 numbers only.
- Stecher et al., https://arxiv.org/abs/2605.18789: SAE "carrier" features in Pythia-160M/410M forecast from early
  firing. Read at abstract and search-summary level only; not cited for any number here.
- Crosscoder and SAE tracking across checkpoints (https://arxiv.org/abs/2509.17196 and others found by query 2) tracks
  features after the fact and does not forecast the carrier. Seen in search summaries only.
- Not found: a forecast of which single MLP neuron of a 7B model will carry a dated job, ranked among all neurons of
  its layer, against matched controls. "Not found in four searches" is not a hole in the literature.

---

## 2. The measurement and the instrument

### 2.1 In plain words

- **Quantity.** The carrier's rank among the 22,016 poles of its layer under A and under B at T_(k−1).
- **Items.** The 153 passing births; the headline is on the 71 test-half births.
- **Model.** `allenai/Olmo-3-1025-7B` at the 11 stage-1 branches T_5 … T_15 (commits in `m2.POINTS`).
- **Pairing.** A, B and the exploratory scores are computed on the same births at the same point; the twin is paired
  with its birth.
- **Summary.** Precision at 10, with a Wilson 95 % interval; paired differences of hits with a percentile bootstrap
  95 % interval over births (10,000 resamples, numpy `default_rng(0)`).
- **Also recorded, no rule.** The same ranks at T_k and T_15; ranks under the ratio, max_abs, density and the column
  norm; the twins under A.

### 2.2 The instrument, exactly

`infra/aws_pg/jobs/m2_score.py` runs one launch. For each point t of the plan, in the order T_15, T_14, …, T_5:
1. it reads `topk/` and `stats/` of the needed layers from S3 `results/neurons/w3/grid/<dir>/`;
2. downloads the branch with the banked harvest's own allow patterns (`matrix_models.HF_ALLOW`), checks the commit
   (G-REV) and the config (G-CFG), loads it with `neurons_harvest.load_model` (bf16, sdpa), runs G0;
3. runs the plan's harvest windows: window w is BOS followed by corpus tokens [2,048 w, 2,048 (w+1)), exactly as in the
   banked harvest; batches of exactly 8, the last one padded with its own last window, so every forward has the
   banked shape 8 × 2,049;
4. captures, with `neurons_harvest.MLPTap`, a = act(g)·u in fp32 for all 11,008 neurons of each needed layer at each
   needed position (input index 1 + token mod 2,048);
5. writes one record per (unit, role) through `m2.scores` and `m2.record`: the carrier's rank and value under every
   score, the top-10 lists under A and B, and for the pre role the ranks of the 100 matched poles.

The plan has 4,551 window forwards over the 11 points (`projection.json` → `window_forwards`), 1,732 of them at T_15.

### 2.3 The primary score: A proposed, B chosen by the head

The brief lists four candidates. Three of them (the pole's percentile of max|a| in its layer, its density, its column
norm) do not depend on the job: each puts the same ten poles first for every birth of a layer. The 300 audit neurons
were drawn uniformly at random within layers (`00_ASSETS.md` §1.4), so a carrier is a uniform random pole of its layer
with respect to any job-free score. Its chance of a top-10 place is 10/22,016 whatever the score. About 600 × 10/22,016
= 0.27 of the 600 audit poles are expected in their layer's top 10 under any one such score, so their precision at 10
cannot exceed a few births in 153. They stay exploratory and are computed at no extra cost.

A is the only job-specific candidate. It is computed from the banked harvest alone, which is the question's wording.
The reference B is the quantity that defines a birth (the median exemplar's place in the pole's own distribution),
read one point early. It is the analogue of the wdd "current usage" forecast that beat the class-side forecast. The
critic's version of M2 made B the baseline for a gradient score. No gradient score is run here, because a gradient is
not an update under Adam and the instrument is not validated (`06_CRITIC.md` lines 536–545); it belongs to J12.

The head may prefer B as the headline. The run, the records and every rule are the same either way; only the line
that names the primary changes (`m2.PRIMARY`, `m2.REFERENCE`). That choice is the head's, before freeze 1.

**Head's decision, 2026-11-05, before freeze 1 (19:40 CST).** B is the primary score and H_B the headline. H_sel_B, the
pseudo-birth twins, is the boundary of the claim. A, the top-K overlap from the banked arrays, is the reference; H_A
and H_AB are reported under their own labels as written. The reason: the question is whether the carrier can be named
before birth at all, and whether that is more than the selection of the exemplars gives. The free score's standing
against the forward (H_AB) is the secondary, practical question. Every rule, bar, split, control and the decision
table in §1.2 stay as written. The code change is `m2.PRIMARY, m2.REFERENCE = "B", "A"`. One more line had to name
the scores explicitly: the H_AB contrast was computed as hit_PRIMARY − hit_REFERENCE, so the swap alone turned it into
hit_B − hit_A. The dry run caught that (2 of 78 checks failed, `offline_gate/dryrun_2026-11-05_swap_only_fails.log`).
`m2.half_summary` now computes hit_A − hit_B by name, as §1.3 and §5 step 7 define it. After that change the dry run
printed `M2_DRYRUN_OK`, 78 checks, 0 failed (`offline_gate/dryrun_2026-11-05_primary_B.log`).

### 2.4 Settings identical to the banked harvest

| setting | banked grid | M2 |
|---|---|---|
| weights | the branch at the banked revision | the same revision, asserted (G-REV) |
| corpus | tokens sha256 `2aca80ca…` | the same, asserted (G-CORPUS) |
| windows | 10,044 windows of 2,048, BOS prepended, window position 0 excluded | the same windows, a subset of them |
| batch and shape | 8 × 2,049 | 8 × 2,049, padded |
| dtype, attention | bf16, sdpa | the same (`neurons_harvest.load_model`) |
| a | act(g)·u in fp32 from the hooked bf16 projections | the same formula through the same `MLPTap` |
| versions | torch 2.12.0+cu130, transformers 5.9.0 | asserted (G-SAME) |
| GPU | L40S (T_0–T_6, T_15) and RTX PRO 5000 Blackwell (T_7–T_14) | one L40S for every point; the cross-device tolerance applies at T_7–T_14 (§3.1) |

### 2.5 Does it run at zero GPU on banked arrays?

Only A does. Its inputs, `topk/layer_LL.npz` at T_(k−1), are banked for every pole, but only on S3 (11 MB per file,
up to 32 files per point), so it runs on the instance, not on the laptop. B needs every pole's activation at the
job's positions. The banks hold that only for the 640 grid poles at their own positions (`exemplar_a.npz`) and for
each pole's own top-50. So B, K0, G-EX, G-TOPK and I-ID need the forward pass. Gradients: none.

---

## 3. Controls and references

Rule 4 of "Instruments and controls" asks for three numbers before unblinding. They are fit on both sides (K0 and G-EX
at every point), a positive control that fired (the construction control, §3.5) and the identity-pair value (I-ID).

### 3.1 The known number K0

**Amended by A1 (§3.8) from freeze 1b on.** The statistic is \|a_new − a_bank\| / (0.05 + \|a_bank\|), and a gate
passes when its 99th percentile is at most 0.021159 (bank on the same device type) or 0.044273 (bank on the other;
the revision of A1 in §3.8; 0.042318 at freeze 1b). From freeze 1d (A2, §3.8), G-EX at a point banked on the other
device reads only the entries with \|a_bank\| ≥ 0.1, at 0.038718; G-TOPK keeps 0.044273.
The rest of this subsection is the freeze-1 rule, kept for the record; its relative difference and its tolerances no
longer decide.

At T_15, before anything else is computed, the windows of the first 10 carrier poles (real units sorted by id:
L00N2968−, L00N3387+, L00N5623−, L00N828+, L00N9356−, L01N369+, L01N4337+, L01N4337−, L01N6100+, L01N7380−) run
first. Their 63 activations at their own exemplar positions must equal the banked T_15 `exemplar_a.npz` (an L40S bank,
the same device type as the lane). The rule is median relative difference ≤ 0.0043 and p99 ≤ 0.0222, the device
check's measured values (§0.2), over at least 10 poles. The failing input is T_14's banked values at the same entries.
Offline that gives median 0.3884 and p99 0.9991, about 90 times the tolerance (dry run REAL). If K0 fails, the job
stops: no score is computed.

**G-EX** repeats this at every point for every grid pole the point forwards. At an L40S-banked point (T_5, T_6, T_15)
the tolerance is the same. At a Blackwell-banked point (T_7–T_14) it is twice the device check's values (median
0.0086, p99 0.0444), because the measured cross-device values are the expected difference, not a bound. The failing
input is the neighbouring point's banked values; offline its median is 0.27–0.67 at the 11 points (dry run REAL).
**G-TOPK** checks every captured position that lies in any pole's banked top-50 at that point against the banked
top-50 value. That covers non-grid poles. Its failing input is the neuron axis shifted by one.

### 3.2 The identity pair I-ID

At T_13 (`m2.IDENTITY_T`), the pre-role windows (the 33 births with k = 14) run a second time in reversed order, so
the batches differ. Pass requires two things. The activations of the two passes agree within the same-device tolerance.
No carrier changes its top-10 membership under B. The failing input is the same births' birth-role activations at
T_14, which must fail. A has no identity question: it is a read of the same banked file.

### 3.3 The structured control: pseudo-birth twins (the boundary of the claim)

E_c is chosen from the carrier's own T_15 top-50. A pole's tuning persists, so the carrier has an edge on E_c at every
earlier point that a pole with no relation to E_c lacks. That edge comes from how the job is defined, not from the
job. Each birth therefore gets a twin built by the same operator, with nothing specific to a job. The twin is a random
group of another grid pole's own T_15 top-50 (non-echo ranks), of the birth's size, in the birth's layer, dated to
the birth's k by the same rule. The candidate poles come in a seeded random order (`default_rng([20261105, l, n, s,
c])`), with up to 2,000 draws each, and the first group dated k is taken (`m2.pseudo_birth`). Twins exist for 79 of
82 freeze births and 70 of 71 test births. Every twin is re-dated in the dry run: 149 of 149 have the same layer,
another neuron, the same size and date k.

The twin's pole is ranked among its layer at T_(k−1) on the twin's positions, exactly as the carrier is. H_sel is
the paired difference of hits. A twin matches its birth on layer, size, birth point and the selection operator. It
differs in two ways: the group is not a coherent cluster, and its date is typical for its pole rather than later than
chance.

### 3.4 The floor: layer- and activation-matched random poles

For each birth, the 100 poles of the layer nearest the carrier in log max_abs at T_(k−1) (the carrier's neuron excluded,
`m2.matched`) are ranked on the same E_c under the same score. One replicate draws one matched pole per birth and takes
its precision at 10. Over 1,000 replicates (seed 0) the floor is the 97.5th percentile (`m2.floor_reps`). It answers
whether the score puts any active pole first on these contexts. It is computed for A and B on each half.

### 3.5 The ceiling and the positive control

- **Construction control (end role, T_15).** At T_15, E_c lies in the carrier's top-50 by definition. Under A the
  carrier scores the maximum, |E_c|. Under B its median sits among its own top 50 values out of about 10⁷ tokens. It
  is not rank 1 by construction: a pole holding the same positions ties it, and ties count against it. The tiny world
  of the dry run gave ranks 2 and 3 for that reason (`offline_gate/dryrun.log`, JOB). The gate is precision at 10 ≥ 0.90 under A and under B on each half;
  below that, DOES NOT COUNT. The failing input is the carrier's opposite pole (same neuron, other sign) at the end
  role, which must stay below 0.90.
- **Birth point (birth role, T_k).** The brief calls the birth point's top-10 place "by construction". It is not. At T_k
  the carrier's median reaches its own q999, which says nothing about the other 22,015 poles. So the birth-point
  precision is measured, not assumed. It is the ceiling of the pre-birth prediction: if the pre-role lower bound
  exceeds the birth-role upper bound, the label is ABOVE_CEILING and the headline reads "DOES NOT COUNT until
  inspected".

### 3.6 The matched random-pole reading (12_SHAPING_NOTE.md §4 item 3)

Beside each raw example the three other grid poles of the layer nearest the carrier in log max|a| at T_(k−1) are shown
with their top-3 contexts and their A on E_c (zero GPU, §10). In the job's examples, the first matched pole's top-3
contexts are shown beside each top-10 list. The reader checks that the carrier's contexts are read as specific to the
carrier, not as what any strongly active pole of that layer looks like.

### 3.7 Monitoring level and the read/edit rule (12_SHAPING_NOTE.md §4 items 4–5)

Every M2 score reads one pole at a fixed identity in one layer: the L1 type of the monitorability hierarchy. A job that
is carried before birth by a direction spread over several neurons, or by another layer, is invisible to it. A REFUTED
H_B says "no single pole of this layer leads", not "the job is absent". M2 reads the input side only (a = act(g)·u comes
from the gate and up rows) and makes no edit. The column norm, an exploratory score, is a property of the output side.
Any later edit of a named carrier (J11, J12) edits along the down-projection column and keeps the two sides as
separate quantities.

### 3.8 Amendment A1 (2026-11-05, before freeze 1b): the known-number statistic

The head's decision of 2026-11-05 20:15 CST, after `score-01` stopped and before a second freeze. Written by the second
executor on branch `m2`. Every number below names its file.

**Cause.** `score-01` stopped at `t14_G_EX`: on 477 entries of 62 grid poles, p99 relative difference 0.2563 against the
tolerance 0.0444 (`studies/next/m2/banks/score-01/metrics.json`). The one L40S repeat banked on S3,
`results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz` (sha256 `e9cdf865…`, `m2.REPEAT_T14`), against
the Blackwell bank on the same 477 entries gives the job's median, p99, max and exact share to every digit
(`studies/next/m2/diag/score01_t14_gex.json`; recomputed by the second executor from the file fetched again from S3).
So the job's forward reproduced the banked L40S harvest, and the rule was wrong for these entries. Its tolerance came
from all 32,000 exemplar entries (median \|a\| 1.95). G-EX at T_14 reads the pre-birth and birth-point exemplars
(median \|a\| 0.31); 12 of the 477 lie below \|a\| = 0.01, where the relative difference reaches 1.86. A relative
difference has no meaning near zero.

**The new statistic.** Per entry, d = \|a_new − a_bank\| / (atol + \|a_bank\|), the mixed tolerance of
`numpy.isclose`, with atol = 0.05 in units of a (`m2.ATOL`, `m2.mixed_diff`). The reason for atol (the head's): bf16
noise in act(g)·u is absolute, about the size seen on the device pair, and a relative statistic has no meaning near
zero. On the pair the absolute difference has p99 0.0023 to 0.0132 in the bins of \|a_bank\| below 0.1
(`studies/next/m2/diag/a1_risk.json` → `by_abs_bank_bin`). A gate passes when the 99th percentile of d over its
entries is at most the tolerance. The median and the max are reported, not tested: the head's rule names one threshold.
The statistic is not symmetric: a_new is the forward and a_bank the bank it is compared with, as the job calls it.

**The tolerance, frozen on control data** (`m2.control_p99`: a_new = the L40S repeat, a_bank = the Blackwell bank, T_14):

| set | entries | median d | p99 d |
|---|---|---|---|
| all | 32,000 | 0.004119 | **0.019737** |
| \|a_bank\| < 1 | 12,231 | 0.002978 | **0.021159** |

The larger p99 is 0.021159 (`m2.CTRL_P99` = 0.021158761373165365). Across devices (bank on the Blackwell, T_7–T_14):
twice it, **0.042318**, the head's rule. Same device type (T_5, T_6, T_15; K0 and I-ID): once, **0.021159**. The second
value is the executor's reading: §3.1 had two tiers (the device check's values on the same device, twice them across
devices), and A1 keeps them. On the same device every gate of `score-01` was bit-identical (K0, T_15 G-EX, T_15 G-TOPK,
exact share 1.000), so the stricter tier costs nothing on the pass side.

**The pass side across devices** (new in the offline gate; `offline_gate/dryrun_2026-11-05_A1.log`, REAL A1):
- the L40S repeat against the Blackwell bank, all 32,000 entries: median 0.00412, p99 0.01974, max 0.3852. Pass.
- the same on the plan's 477 entries at T_14, the ones `score-01` read: median 0.00427, p99 0.04034, max 0.3852. Pass,
  0.0020 below the tolerance. score-01 showed the forward reproduces the repeat there, so T_14 passes in score-02.

**The registered failing inputs under the new statistic** (banked arrays, the same log):
- K0: T_14's banked values in place of T_15's at the 63 entries of the first 10 carriers: median 0.3354, p99 3.7936,
  against 0.021159. Fails.
- G-EX: the neighbouring point's bank at each of the 11 points (point, entries, median, p99): (15, 2295, 0.264, 3.787),
  (14, 477, 0.343, 1.046), (13, 717, 0.422, 3.908), (12, 717, 0.390, 2.250), (11, 643, 0.416, 3.347),
  (10, 397, 0.450, 10.295), (9, 243, 1.157, 14.177), (8, 147, 1.840, 7.876), (7, 116, 0.855, 3.896), (6, 5, 1.019,
  1.211), (5, 10, 0.651, 0.988). Fails at all 11.
- T_13's values in T_14's place (the dry run's planted G-EX input), against the L40S repeat as the forward: p99 11.157
  on the 477 entries, 4.506 on all 32,000. Fails.
- G-TOPK (the neuron axis shifted by one) and I-ID (T_14's birth-role activations in place of the second pass) need
  the forward's activations of all 11,008 neurons, which no bank holds; `score-01`'s T_15 vectors stay unopened. In the
  dry run's tiny world they fail under the new statistic (G-TOPK p99 0.95–1.11 at four points, I-ID p99 2.35, against
  0.021159), and the job asserts every failing input: one that passes stops the job.

**Scope.** `m2.values_gate` is the one function behind K0, G-EX, G-TOPK and I-ID, so all four move to the new statistic.
The head named K0, G-EX and G-TOPK; I-ID uses the same statistic, so it moves too, at the same-device tier.

**Risk, measured offline; not a rule** (`studies/next/m2/diag/a1_risk.py`, `a1_risk.json`). No cross-device repeat
exists at T_7–T_13. If each G-EX entry draws its d from the T_14 pair's entries in the same \|a_bank\| bin, the share of
2,000 draws whose p99 exceeds 0.042318 is 0.161 at T_13, 0.151 at T_11, 0.032 at T_8 and at most 0.011 at T_7, T_9,
T_10 and T_12; all seven pass in 0.68 of draws. The model understates T_14 itself (actual p99 0.0403 against a model
median 0.0286). So `score-02` may stop again on G-EX, with a probability of about one in three or more. The frozen rule
decides; a stop is reported and not relaunched without the head.

**Headline order (A1 item 3, cosmetic).** §5 step 10 lists H_B first, then H_sel_B, H_A, H_AB, as the primary is B.
No label and no rule changes.

**The blind protocol.** From `score-01` the executors read only `metrics.json`, `STATUS`, `pipeline_status.json`,
`heartbeat.jsonl`, `dmon.log`, `job.log` (once, after the end) and the G-EX diagnostic. Its rank files
(`ranks_freeze.jsonl`, `ranks_test.jsonl`), its raw-example files and its `vectors/t15/` stay unopened on S3. A1 was
written without them.

**Files changed** (from `m2-freeze-1`): `infra/pg/m2.py` (`ATOL`, `CTRL_P99`, `TOL`, `REPEAT_T14`, `mixed_diff`,
`control_p99`, `values_gate`, the headline order); `infra/aws_pg/jobs/m2_score.py` (the farthest entry by the new
statistic; docstring); `infra/aws_pg/jobs/m2_dryrun.py` (`--repeat`, five A1 checks, the job's (new, bank) order in the
offline K0 and G-EX failing inputs, the new key names). `births.json`, the inputs and the step list are unchanged. The
`score-01` diagnostic (`diag/score01_t14_gex.py`) calls the freeze-1 `values_gate`; it reproduces its record at tag
`m2-freeze-1`.

**Dry run.** `M2_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_2026-11-05_A1.log`): the 78 of freeze 1 and
five new ones. The command (§9.3) takes `--repeat`.

**Launch and banks.** `score-02` is §9.5 S2 with the prefix `results/next_m2/score-02/`, the same 17 inputs (sha256
checked again), the same step list (`infra/aws_pg/steps/next_m2_score-01.json`, which names no attempt),
`--type g6e.2xlarge,g6e.4xlarge`, `--budget-min 165`. The banks are pulled to `studies/next/m2/banks/score-02/m2/`, and
freeze and verdict take `--banks studies/next/m2/banks/score-02`.

**Revision of A1 (the head's decision of 2026-11-05 20:50 CST, before freeze 1c).** Written by the third executor on
branch `m2`. Every number below names its file.

*Reason, in plain words.* A1 set the cross-device tolerance at twice the p99 of d over the control pair's entries with
\|a_bank\| < 1 (0.021159, so 0.042318). The entries G-EX reads before a birth are smaller: their median \|a_bank\| is
0.31. On the 477 entries `score-01` read at T_14, the control pair itself gives p99 0.04034, 0.002 under the tolerance. A
gate whose own control passes by 0.002 has no room. The second executor's risk model put the chance of a stop at T_13 or
T_11 at about one in three. So the control subset has to cover the range of the entries the gate reads.

*The rule.* Across devices (bank on the Blackwell, T_7–T_14): twice the p99 of d over the control pair's entries with
\|a_bank\| < 0.5. The pair is the T_14 Blackwell bank against the L40S repeat, as in A1. Same device type (T_5, T_6,
T_15; K0 and I-ID): unchanged, 0.021159.

| set (`m2.control_p99`, the repeat as new, the Blackwell bank as bank) | entries | median d | p99 d |
|---|---|---|---|
| all | 32,000 | 0.004119 | 0.019737 |
| \|a_bank\| < 1 (the same-device tier, unchanged) | 12,231 | 0.002978 | **0.021159** |
| \|a_bank\| < 0.5 (`m2.CTRL_P99_X` = 0.022136643261325667) | 7,424 | 0.001510 | **0.022137** |
| \|a_bank\| < 0.25 (reported, not used) | 2,726 | 0.002259 | 0.034947 |

Tolerance across devices: 2 × 0.022137 = **0.044273** (`m2.TOL["cross"]`; at freeze 1b 0.042318, so +0.00196 or
4.6 %). Same device type: **0.021159** (`m2.TOL["same"]`).

*What it changes, said plainly: little.* The p99 over \|a_bank\| < 0.5 is close to the one over < 1. Of the subset's
7,424 entries, 365 (4.9 %) lie below \|a_bank\| = 0.1, the bins where the pair's p99 of d is 0.045 to 0.149
(`diag/a1_risk.json` → `by_abs_bank_bin`); of the 477 entries G-EX read at T_14, 14.9 % do
(`diag/a1_risk_rev.json` → `points.14.share_abs_below_0.1`). So the subset still has fewer small entries than the gate
reads. On the 477 entries the pair gives p99 0.04034 against 0.044273: a pass by 0.0039 (0.91 of the tolerance). For
information, twice the p99 over \|a_bank\| < 0.25 would be 0.0699; it is not the rule.

*Same-device tier and I-ID: unchanged, because the reasoning does not apply.* The cross-device tier lacked room because
the device difference itself comes near the tolerance on small entries. On one device type the difference measured so
far is zero. In `score-01` K0 (63 entries), T_15 G-EX (2,295) and T_15 G-TOPK (5,675 entries of all neurons' top-50
lists) were bit-identical to the L40S bank (exact share 1.000, `banks/score-01/metrics.json`), although the job formed
batches the harvest never formed. In the tiny world I-ID was bit-identical (8,832 entries, exact share 1.000; median 0 in
`offline_gate/dryrun_2026-11-05_A1rev.log`, JOB). I-ID compares two passes of one job on one device that differ only in batch order. Its pass side therefore has the
whole tolerance as room, whatever the range of its entries (all 11,008 neurons at the pre-role positions, mostly near
zero). If the L40S were not bit-identical in the job, I-ID would stop it; that would be a finding about the device, and
the rule would not be changed after it.

*The registered failing inputs under the new tolerance* (`offline_gate/dryrun_2026-11-05_A1rev.log`). The statistic is
unchanged, so every value equals A1's; only the threshold moved.
- K0 (same-device tier): T_14's values at the 63 entries, p99 3.7936 against 0.021159 (179 times). Fails.
- G-EX, the neighbouring point's bank: at T_7–T_14, p99 1.046 (T_14) to 14.177 (T_9) against 0.044273 (24 to 320
  times); at T_5, T_6, T_15, p99 0.988, 1.211 and 3.787 against 0.021159 (47 to 179 times). Fails at all 11.
- T_13's values in T_14's place: p99 11.157 on the 477 entries (252 times), 4.506 on all 32,000 (102 times). Fails.
- G-TOPK and I-ID: their failing inputs need all-neuron activations; in the tiny world they fail as before (G-TOPK p99
  0.95–1.11, I-ID 2.35, against the same-device 0.021159), and the job asserts both.

*Risk, rerun; not a rule* (`diag/a1_risk.py`, output `diag/a1_risk_rev.json`, log `diag/a1_risk_rev.log`). The same
model as A1 with the new tolerance. Share of 2,000 draws whose p99 exceeds 0.044273: T_13 0.1555, T_11 0.122, T_8 0.029,
T_12 0.009, T_10 0.002, T_7 and T_9 0.0005. Share of draws passing at all seven points T_7–T_13, counted draw by draw:
**0.7125** (product of the per-point shares 0.711), against 0.679 at 0.042318. The rerun gives every freeze-1b share at
0.042318 exactly (`share_above_tol_freeze_1b`), so the instrument reproduces its own record. The model understates T_14
(actual p99 0.0403, model median 0.0286), so the real chance of a stop is likely above 0.29. The frozen rule decides; a
stop is reported and not relaunched without the head.

*Files changed* (from `m2-freeze-1b`): `infra/pg/m2.py` (`CTRL_P99_X`, `TOL["cross"]`, `control_p99` reports < 1, < 0.5,
< 0.25); `infra/aws_pg/jobs/m2_dryrun.py` (the tolerance check names both tiers and prints the four subsets);
`studies/next/m2/diag/a1_risk.py` (the share at the freeze-1b tolerance beside the new one; the draw-by-draw share).
`infra/aws_pg/jobs/m2_score.py`, `births.json`, the 17 inputs and the step list are unchanged.

*Dry run.* `M2_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_2026-11-05_A1rev.log`, values
`dryrun_2026-11-05_A1rev_summary.json`). Against the freeze-1b log only the tolerance line differs.

*Launch.* `score-02` as above, on `m2-freeze-1c`. The prefix `results/next_m2/score-02/` is reused: it holds only
`steps.json` from the capacity-failed tries of 02:27–02:44Z, and no job ever wrote there (no STATUS, no `metrics.json`),
so no stale artifact can be read (the head's permission, 2026-11-05). Capacity rule for this launch (the head's):
`--type g6e.2xlarge,g6e.4xlarge` every 180 s for 12 tries, then every 600 s until 2 hours after the first try.

**Amendment A2 (the head's decision of 2026-11-06 03:45Z, before freeze 1d).** Written by the third executor on branch
`m2`. Every number below names its file.

*What happened.* `score-02` (freeze 1c) stopped at `t13_G_EX`: 717 entries on 79 grid poles, p99 of d 0.05433 against
0.044273, median 0.00431, max 0.222, exact share 0.310 (`studies/next/m2/banks/score-02/metrics.json`). K0, T_15 G-EX and
T_15 G-TOPK were bit-identical; T_14 G-EX passed at p99 0.040338, the value the banked L40S repeat gives to every digit.

*Reason, in plain words (the head's).* The pre-birth exemplar entries are small by construction: the pole has not
started firing on them. On small activations the difference between two GPU types is noise, whatever statistic is used.
The 99th percentile is then set by the smallest entries, and those say nothing about whether the forward is right. At
T_14 the forward reproduced the banked L40S harvest exactly. T_13 has no repeat; its median (0.0043) and max (0.222)
are no worse than T_14's (0.0043, 0.385). A known-number gate must be computed on entries that carry signal.

*The rule.*
1. Cross-device G-EX (bank on the Blackwell, T_7–T_14): the p99 of d = \|a_new − a_bank\| / (0.05 + \|a_bank\|) over the
   entries with \|a_bank\| ≥ 0.1 only (`m2.EX_ABS_MIN`). The entries below 0.1 are reported (count, median and p99 of d,
   `below_abs_min`) and not gated. The tolerance is twice the control pair's p99 of the same statistic on the same
   subset (`m2.CTRL_P99_EX`, `m2.TOL["ex_cross"]`, `m2.tol_ex`).
2. G-TOPK unchanged (`m2.tol_for`, 0.044273 across devices): its entries are large by construction. K0 and I-ID
   unchanged (same device, 0.021159). G-EX at the same-device points T_5, T_6, T_15 unchanged (all entries, 0.021159).
3. For a failing input, the failing bank takes the bank's place, so the subset is \|a_fail_bank\| ≥ 0.1.

*The tolerance, frozen on control data* (`m2.control_p99`: the T_14 L40S repeat `e9cdf865…` as new, the Blackwell bank
as bank):

| set | entries | median d | p99 d |
|---|---|---|---|
| \|a_bank\| ≥ 0.1 | 31,635 | 0.004127 | **0.019359** (`m2.CTRL_P99_EX` = 0.019359100495742992) |
| \|a_bank\| < 0.1 (reported) | 365 | 0.003081 | 0.108214 |

Tolerance across devices for G-EX: 2 × 0.019359 = **0.038718** (at freeze 1c 0.044273 on all entries).

*The pass side across devices* (`offline_gate/dryrun_2026-11-06_A2.log`, REAL A2):
- the control pair, all 32,000 entries: gated 31,635, median 0.00413, p99 0.01936, max 0.3852. Pass.
- the plan's 477 entries at T_14: gated 406 (71 below 0.1, p99 0.04290 there), median 0.00438, **p99 0.02886** against
  0.038718, max 0.3852. Pass. Said plainly: 4 of the 406 gated entries lie above the tolerance (d 0.0399, 0.0453,
  0.0514, 0.3852, at \|a_bank\| 0.286, 0.616, 0.611, 0.151; `diag/a2_risk.json` → `plan_pool`), and a fifth would make
  the p99 exceed it. The room is one entry.

*The new statistic on the jobs' own entries.* The job banks no per-entry G-EX values: `metrics.json` holds the
aggregates and the farthest entry only. score-02 wrote no T_13 vectors (the gate stopped the point before its records);
score-01 wrote none at T_14. score-02's `vectors/t14/` exist but hold every neuron's activation at the exemplar
positions, from which the B ranks follow; they are blind data and stay unopened. So the A2 value on score-02's T_13
entries and on the two jobs' T_14 entries is not computed from the jobs. At T_14 the job reproduced the repeat to every
printed digit in both attempts (median, p99, max, exact share), so the repeat's value, 0.02886, is the expected one.

*The registered failing inputs under A2* (the same log):
- G-EX, the neighbouring point's bank, p99 over its gated subset: T_14 1.046, T_13 1.935, T_12 1.874, T_11 2.559,
  T_10 10.428, T_9 14.301, T_8 7.943, T_7 3.902 against 0.038718 (27 to 369 times); T_15, T_6, T_5 unchanged (3.787,
  1.211, 0.988 against 0.021159). Fails at all 11.
- T_13's values in T_14's place: p99 7.516 (354 gated of the plan's 477), 2.541 (30,661 gated of 32,000). Fails.
- K0 (unchanged): T_14's values at the 63 entries, p99 3.7936 against 0.021159. Fails.
- The job's cross-device path (new in the tiny world, JOB and FAIL): T_14's bank marked as another device, `t14_G_EX`
  gates 12 of 157 entries, passes (bit-identical) and its failing input fails (p99 1.393); T_13's file in T_14's place
  stops the job at `t14_G_EX` (13 gated, p99 0.821).

*Risk, measured offline; not a rule* (`diag/a2_risk.py`, `a2_risk.json`; `diag/risk_calibration.py`,
`risk_calibration.json`).
- The bin model of A1 restricted to the gated entries gives a pass at all of T_7–T_13 in 0.9975 of draws. It places
  T_14's known value (0.02886) at its 98.3rd percentile, so it understates the tail, as it did in every earlier version.
- The plan's gated entries at T_14 lie above the tolerance at 9 times the control's rate: 4 of 406 (0.99 %) against 35
  of 31,635 (0.11 %); at the control's rate 4 or more occur with probability 0.0012. So the control subset does not
  represent the entries the gate reads, at \|a_bank\| ≥ 0.1 too.
- A second model draws from the plan's own T_14 gated entries: pass at all of T_7–T_13 in 0.085 of draws. It also fails
  its check: drawing T_13's 717 entries from the plan's 477 at T_14 under the freeze-1c statistic never reaches the
  observed 0.0543 (0 of 2,000; the bin model 0.06). T_13's tail is heavier than T_14's.
- Neither model can be trusted at T_13. **Prediction logged before the launch: score-03 stops on G-EX with probability
  at least 0.5, most likely at T_13 or T_12.** The frozen rule decides; a stop is reported and not relaunched.

*The blind protocol.* From `score-01` and `score-02` the executors read only `metrics.json`, `STATUS`,
`pipeline_status.json`, `heartbeat.jsonl`, `dmon.log`, `job.log` (once, after the end) and the diagnostics in
`studies/next/m2/diag/`. Their rank files, raw-example files and vectors stay unopened on S3. A2 was written without
them.

*Files changed* (from `m2-freeze-1c`): `infra/pg/m2.py` (`EX_ABS_MIN`, `CTRL_P99_EX`, `TOL["ex_cross"]`, `gate_mask`,
`tol_ex`, `values_gate` with `n_gated` and `below_abs_min`, `control_p99` reports ≥ 0.1); `infra/aws_pg/jobs/m2_score.py`
(G-EX takes `m2.tol_ex`; the farthest entry is the farthest gated one); `infra/aws_pg/jobs/m2_dryrun.py` (the A2 checks;
the tiny world's cross-device run and its failing input); `studies/next/m2/diag/{a2_risk,risk_calibration,
score02_t13_gex}.py`. `births.json`, the 17 inputs and the step list are unchanged.

*Dry run.* `M2_DRYRUN_OK`, 86 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A2.log`, values
`dryrun_2026-11-06_A2_summary.json`): the 83 of freeze 1c with three A1 checks moved to A2, and three new ones.

*Launch.* `score-03` on `m2-freeze-1d`, the fresh prefix `results/next_m2/score-03/`, the same 17 inputs and step list,
`--type g6e.2xlarge,g6e.4xlarge`, `--budget-min 165`, the head's capacity rule (180 s apart for 12 tries, then 600 s,
until 2 h after the first). The banks are pulled to `studies/next/m2/banks/score-03/m2/`; freeze and verdict take
`--banks studies/next/m2/banks/score-03`. If a gate stops it, no fourth attempt without the head.

**Amendment A3 (the head's decision of 2026-11-06 04:20Z, before freeze 1e).** Written by the third executor on branch
`m2`. Every number below names its file.

*The three stops.*

| attempt | freeze | stopped at | entries read | p99 of the gate | tolerance | median | max |
|---|---|---|---|---|---|---|---|
| `score-01` | 1 | `t14_G_EX` | 477 | 0.2563 (relative difference) | 0.0444 | 0.00524 | 1.953 |
| `score-02` | 1c | `t13_G_EX` | 717 | 0.05433 (d, all entries) | 0.044273 | 0.00431 | 0.222 |
| `score-03` | 1d | `t13_G_EX` | 591 of 717 (\|a_bank\| ≥ 0.1) | 0.05191 (d) | 0.038718 | 0.00474 | 0.0766 |

(`studies/next/m2/banks/score-0{1,2,3}/metrics.json`.) In every attempt K0, T_15 G-EX and T_15 G-TOPK were
bit-identical, and at T_14 the job reproduced the banked L40S repeat to every printed digit (score-02 p99 0.040338 under
freeze 1c, score-03 0.028857 under A2).

*Reason, in plain words (the head's).* On the same statistic and the same kind of entries, T_13's cross-device tail is
twice T_14's (0.0519 against 0.0289 on the entries with \|a_bank\| ≥ 0.1), while its median is the same (0.0047 against
0.0044). So the deviation between the bank harvested on the Blackwell and an L40S forward varies by checkpoint. A
tolerance frozen on the one T_14 pair cannot bound the other Blackwell-banked points. A gate whose threshold cannot be
frozen on control data is not a gate; it is a measurement. The purpose of G-EX, that the forward reproduces the banked
harvest, is served at the same-device points by K0 and T_15 (bit-identical in all three attempts), and at T_14 by the
exact match against the banked L40S repeat. The cross-device residual at T_7–T_13 is a property of the device the bank
was harvested on, not of this job. B's ranks compare poles under one forward, so the residual enters all poles alike.

*The rule.*
1. At the points whose bank was made on the other device (T_7–T_14), G-EX is **reported, not gated**. The job computes
   the A2 statistic (d over the entries with \|a_bank\| ≥ 0.1, against 0.038718), writes p99, max, the count of read
   entries above the tolerance and their list (row, pole, slot, a_bank, a_new, d; `metrics.json` → `reports` →
   `tNN_G_EX`), and continues. The failing input (the neighbouring point's bank) is computed and written beside it.
   Neither stops the job.
2. At the points whose bank was made on the forward's device type (T_5, T_6, T_15; `m2.ex_is_gate`), G-EX stays a gate
   at the same-device tier (0.021159, all entries). The head named T_15; T_5 and T_6 are L40S-banked too, so the rule
   by device keeps them gates (executor's reading).
3. G-TOPK stays a gate at every point, unchanged from freeze 1d: 0.044273 at the Blackwell-banked points, 0.021159 at
   the L40S-banked ones (executor's reading of "the A2 cross-device tier": A2 left G-TOPK's tier unchanged). Its
   entries are the poles' own top-50 values, large by construction; at T_14 it passed at p99 0.0142 in both score-02 and
   score-03. K0 and I-ID unchanged.
4. If G-TOPK or any other gate stops `score-04`, M2 stops for good and is reported DOES NOT COUNT with the cause. No
   fifth attempt.
5. The verdict carries the line "cross-device G-EX exceeded the A2 tolerance at points …" (`m2.cross_device_gex`, §5
   step 9b), filled from the job's reports, with each point's p99, max and count. §12 has a row for it. It is never a
   reason for DOES NOT COUNT. The result document, when written, states it as a limitation.

*What the dry run shows* (`offline_gate/dryrun_2026-11-06_A3.log`, values `dryrun_2026-11-06_A3_summary.json`): 89 checks,
0 failed.
- The informational path: T_13's bank scaled by 1.2 and marked as made on another device. The job completes
  (`M2_JOB_OK`, `complete` true). `t13_G_EX` is absent from the gates and present in the reports, with p99 0.142, 15 of 15
  read entries above the tolerance and a list of 15. The verdict's line reads "… at points T_13".
- The cross-device path with no excess (T_14 marked as another device): `t14_G_EX` reported, 12 of 157 entries read,
  0 above; its failing input p99 1.393; G-TOPK at T_14 a gate at 0.044273; G-EX at T_13 a gate at 0.021159.
- A planted G-TOPK excess still stops the job: T_14's banked top-50 values scaled by 1.1, T_14 cross-device, exit
  non-zero naming `t14_G_TOPK` (873 entries, p99 0.0822 against 0.044273).
- A reported point over the tolerance never changes the headline: the satisfying synthetic case with a planted
  `t13_G_EX` report over the tolerance keeps its headline, and the line names T_13.
- The same-device G-EX failing input (T_13's file in T_14's place, all points one device) still stops the job at
  `t14_G_EX`. K0's failing inputs still stop it. The REAL section is unchanged from A2.

*The blind protocol.* From `score-01`, `score-02` and `score-03` the executors read only `metrics.json`, `STATUS`,
`pipeline_status.json`, `heartbeat.jsonl`, `dmon.log`, `job.log` (once, after the end) and the diagnostics in
`studies/next/m2/diag/`. Their rank files, raw-example files and vectors stay unopened on S3. A3 was written without
them.

*Files changed* (from `m2-freeze-1d`): `infra/pg/m2.py` (`ex_is_gate`, `cross_device_gex`, the verdict's
`cross_device_gex` and its printed line); `infra/aws_pg/jobs/m2_score.py` (`Run.report`; `ex_gate` writes the count and
list above the tolerance; G-EX gated or reported by `m2.ex_is_gate`); `infra/aws_pg/jobs/m2_dryrun.py` (the A3 cases).
`births.json`, the 17 inputs and the step list are unchanged.

*Launch.* `score-04` on `m2-freeze-1e`, the fresh prefix `results/next_m2/score-04/`, the same 17 inputs and step list,
`--type g6e.2xlarge,g6e.4xlarge`, `--budget-min 165`, the head's capacity rule. The banks are pulled to
`studies/next/m2/banks/score-04/m2/`; freeze and verdict take `--banks studies/next/m2/banks/score-04`.

**Amendment A4 (the head's decision of 2026-11-06 05:35Z, before freeze 1f).** Written by the fourth executor on branch
`m2`. Every number below names its file.

*Cause.* `score-04` (freeze 1e) passed every gate through T_10 and then crashed while scoring T_10: `m2.percentile`
asserted that every pole of the layer has a token of its own sign in the banked histogram, and L24N9826+ has none
there (`studies/next/m2/banks/score-04/job.log` lines 167–183; `studies/next/m2/diag/score04_empty_poles.json`).
`m2.matched` made the same assumption (`max_abs > 0` over the layer). The offline gate never planted the case: the
REAL section read the laptop banks of the 640 grid poles only, and the tiny world's random model has no such pole. The
scan below found a third assumption of the same kind: such a pole's banked q999 is NaN (the harvest's
`hist_quantile_edge` writes NaN for an empty histogram), so the exploratory ratio v / q999 would have been NaN and
`m2.ranks` would have stopped on "a NaN score" right after.

*The rule.*
1. A pole with no token of its own sign in the banked harvest at a point (`m2.degenerate`: n_pole_tokens 0, so an
   empty own-sign histogram, and max_abs 0) gets **B = 0** there. It ranks last under B; ties count against the
   carrier, as registered. **A** is unaffected: its banked top-50 slots are empty (`tok` −1), so its overlap is 0.
2. Such a pole is **not a candidate of the matched-pole floor set**: it has no log max_abs, so it cannot be "nearest in
   log max_abs". The other candidates keep the freeze-1e order (`m2.matched`).
3. The ratio score gets **ratio = 0** for such a pole (its q999 is NaN), the same reading as B (item 3 of the head's
   decision: an assumption the scan contradicts, fixed under A4).
4. The job writes the count and the ids of such poles in the layers each point scores (`metrics.json` → `points` →
   `tNN` → `n_degenerate_poles`, `degenerate_poles`). The verdict carries them (`verdict_m2.json` →
   `degenerate_poles`, §5 step 9c). Never a reason for DOES NOT COUNT.
5. Fail loud for anything else: a pole whose histogram is empty while max_abs > 0, or the reverse, stops the job
   (`m2.degenerate`); a carrier with no token of its sign at its pre point stops it; so does a layer with fewer than
   K_MATCH candidates (`m2.matched`).

*The scan of the banks* (`studies/next/m2/offline_gate/bank_scan.py`, output `bank_scan.json` sha256 `8f0a41ba…`, log
`bank_scan.log`; 2026-11-06 05:37–05:40Z). All 32 layers at all 11 points T_5–T_15: 352 stats files and 352 topk files
on S3, read by 2,112 byte-range or list calls. It read only the small members of each stats file (max_abs, hist_edges,
n_pole_tokens, q999, density) and the .npy headers of the large members (stats `hist`; topk `tok`, `a`). Positive
control: its max_abs and q999 for the 640 grid poles equal the laptop bank (`arrays.npz`, registered sha256) at all 11
points, 640 of 640, max difference 0; against the neighbouring point's laptop values max_abs matches 0 of 640 and q999
0 to 40 of 640 (q999 is a bin edge), so a wrong file or a wrong index would show.

| point | poles with no token of their sign, all 32 layers | in the layers the point scores (the job's expected count) |
|---|---|---|
| T_15, T_14, T_13, T_12, T_11 | 0 | 0 |
| T_10 | 1: L24N9826+ | **1**: L24N9826+ (layer 24 is scored; the crash) |
| T_9 | 2: L07N640−, L19N726+ | 0 (layers 7 and 19 are not scored at T_9) |
| T_8 | 2: L07N640−, L31N3935− | **1**: L07N640− (layer 7 is scored) |
| T_7 | 2: L08N8160−, L31N3935− | 0 (layers 8 and 31 are not scored at T_7) |
| T_6, T_5 | 0 | 0 |

Every one of the 7 has n_pole_tokens 0, max_abs 0, q999 NaN and density 0; in every file the empty and the max_abs-0
poles coincide. Every other pole has q999 finite and positive and max_abs finite. No scored unit's own pole (carrier
or twin) is among them at any point it is scored; the fewest candidates in any scored layer is 22,013 (K_MATCH 100). So
the rule touches 2 poles at 2 points of the run, both competitors, and changes nothing at T_11–T_15.

*Every assertion in the scoring path* (`grep -n assert` in `infra/pg/m2.py` and `infra/aws_pg/jobs/m2_score.py`; no
`raise` in either file). The banked quantity each assumes, and whether the scan now checks it against the banks:

| where (line) | assertion | banked quantity it assumes | checked against the banks |
|---|---|---|---|
| `m2.percentile` (freeze 1e, line 346) | every pole has a token of its sign | `stats/hist` sums > 0 | **contradicted** (2 scored poles); replaced by rule 1 |
| `m2.matched` (freeze 1e, line 387) | max_abs > 0 over the layer | `stats/max_abs` | **contradicted**; replaced by rule 2 |
| `m2.ranks` 396 | no NaN score | `stats/q999` finite (ratio), `max_abs`, `density` finite | **contradicted** for q999 on the same poles; rule 3; the scan shows q999 finite and > 0 on every other pole |
| `m2.degenerate` 354 (new) | empty histogram ⇔ max_abs 0 | `n_pole_tokens`, `max_abs` | yes: they coincide in all 352 files |
| `m2.matched` 407 (new) | the carrier has a token of its sign at its pre point | `max_abs` of the carrier | yes: no scored unit's own pole is degenerate at a point where it is scored (`bank_scan.json` → `points` → `units_whose_own_pole_is_degenerate`, empty at all 11) |
| `m2.matched` 410 (new) | ≥ K_MATCH candidates | `max_abs` | yes: at least 22,013 |
| `m2.scores` 383 | a has one column per neuron of the topk file | `topk/tok` shape [11008, 2, 50] | yes: headers of all 352 topk files |
| `m2_score.layer_files` 131 | `hist.sum(-1)` = `n_pole_tokens` | `stats/hist` | not by the scan (needs the large member); true by construction (`neurons_harvest.finalize_layer` writes n_pole_tokens = hist.sum(-1)); the job asserts it at load; it held for every layer loaded at T_15–T_10 in score-04 |
| `m2.values_gate` 481 | something compared | `exemplar_a` (laptop) at each point | yes, in REAL: at all 11 points the read subset is non-empty on the bank's side (5 to 2,295 entries) and on the failing side |
| `m2_score.Run.gate` 80, 81 | gates pass; failing inputs fail | the gates' banks | the gates themselves (§4) |
| `m2_score` 261, 374, 434, 436; `m2.parse_pole` 185 | plan order, I-ID's T_14 birth records, registry, CUDA, pole-id format | none (plan and environment) | not banked; the dry run runs each path |
| `m2.py` 219–293, 529–759 | sha256 of the laptop banks, dating rule, Wilson n > 0, freeze and verdict refusals | laptop banks and records | the REAL and VERDICT sections (unchanged) |

Implicit assumptions without an assert, now checked by the scan: every stats file holds the six members the job loads
(`hist`, `hist_edges`, `q999`, `max_abs`, `density`, `n_pole_tokens`) and every topk file `tok` and `a`; shapes
[11008, 2] and [11008, 2, 400]; `hist_edges` equals the constants `percentile` uses (10^linspace(−6, 4, 401) as
float32), 352 of 352. Not checked: that the banked top-50 `a` is finite where `tok` ≥ 0 (needs the large member); a
NaN there would make G-TOPK's p99 NaN, and the gate fails loud.

*What the dry run shows* (`offline_gate/dryrun_2026-11-06_A4.log`, values `dryrun_2026-11-06_A4_summary.json`): 112
checks, 0 failed (the 89 of freeze 1e and 23 new).
- REAL: the scan's facts above (8 checks) and the non-empty G-EX subset at all 11 points.
- SCORE: a degenerate pole with v = 1.5 gets B = 0 (0.618 without the rule), ratio 0, A 0, rank 6 of 6; the other
  poles' A, B and ratio are unchanged; `matched` drops a max_abs-0 pole and keeps the freeze-1e order for the rest;
  each new assertion stops on its planted input.
- JOB: in the tiny world the matched pole nearest the carrier L00N40+ in log max_abs, L00N82− (B 0.2545, rank 109 of
  192), is planted at T_13 as the harvest writes a pole with no token of its sign. The job completes, every gate passes
  and every failing input fails; `metrics.json` lists L00N82− at T_13 and nothing elsewhere; its B and ratio are 0,
  rank 192 of 192 under both; it leaves the matched set and the other 99 keep their order; the forward is identical
  and every other pole's A, B and ratio are unchanged; the carrier's ranks (B 42, A 192) are unchanged.
- FAIL: an empty histogram with max_abs > 0 stops the job ("1 poles whose histogram and max_abs disagree"); the carrier
  planted the same way at its pre point stops it ("the carrier has no token of its sign").
- The failing input of the whole fix: the same planted world on the freeze-1e code crashes as score-04 did,
  `AssertionError: a pole with no token of its sign` at `m2.py` line 346 via `m2_score.py` line 334
  (`offline_gate/a4_planted_on_freeze_1e.log`).

*The blind protocol.* From `score-01` to `score-04` the executors read only `metrics.json`, `STATUS`,
`pipeline_status.json`, `heartbeat.jsonl`, `dmon.log`, `job.log` (once, after the end), the diagnostics in
`studies/next/m2/diag/` and, for the banked grid harvest, the small members and .npy headers named above. Their rank
files, raw-example files and vectors stay unopened on S3. A4 was written without them.

*Files changed* (from `m2-freeze-1e`): `infra/pg/m2.py` (`degenerate`; `percentile` returns 0 for an empty histogram;
`scores` ratio 0 for a degenerate pole; `matched` excludes max_abs-0 poles; `degenerate_counts` and the verdict's
`degenerate_poles`); `infra/aws_pg/jobs/m2_score.py` (the per-point count and ids); `infra/aws_pg/jobs/m2_dryrun.py`
(`--scan`, the A4 checks, `plant_degenerate`, `a4_job`); new `studies/next/m2/offline_gate/bank_scan.py`. `births.json`,
the 17 inputs and the step list are unchanged.

*Launch.* `score-05` on `m2-freeze-1f`, the fresh prefix `results/next_m2/score-05/`, the same 17 inputs and step list,
`--type g6e.2xlarge,g6e.4xlarge`, `--budget-min 165`, the head's capacity rule (every 180 s for six tries, then every
600 s until two hours after the first; `VcpuLimitExceeded` retried the same way). The banks are pulled to
`studies/next/m2/banks/score-05/m2/`; freeze and verdict take `--banks studies/next/m2/banks/score-05`. On a failure:
`job.log` once, the cause banked, no relaunch.

**Amendment A5 (the head's decision of 2026-11-06 06:50Z, before freeze 1g).** Written by the fifth executor on branch
`m2`. Every number below names its file.

*Cause.* `score-05` (freeze 1f) stopped at `t06_G0`. The gate passed (max ratio 0.00352 against G0_TOL 0.01), and its
registered failing input, the activation swapped (silu → gelu_tanh), passed too: 0.00970
(`studies/next/m2/banks/score-05/metrics.json`). The swap's power depends on |g|: at early checkpoints silu and
gelu_tanh nearly agree on the small gate values. The banked harvest's own controls give 0.0113 at T_6 and 0.0062 at
T_5 (`studies/next/m2/diag/score05_g0_fail.json`). The offline gate never read those `controls.json` files. That is the
third gap: a failing input with no power where it is applied.

*The rule.*
1. G0's registered failing input is **u scaled by 1.1** (`neurons_harvest.WRONG_U_SCALE`, the banked harvest's
   `failing_input_2`), computed on the gate's own first batch at every point. `Run.gate` asserts that it fails, as
   before.
2. The activation swap is still run. It is **reported, not gated**: `metrics.json` → `reports` → `tNN_G0_act_swap`
   (max ratio, layers over 0.01, `under_tol`). It is never a reason for DOES NOT COUNT.
3. Before the freeze, both sides of every gate the job runs are evaluated at every point from the banked files
   (`studies/next/m2/offline_gate/a5_gate_sides.py`, output `a5_gate_sides.json` sha256 `717bde0d…`, log
   `a5_gate_sides.log`), and the dry run checks them (REAL A5). The pass side is what a correct run gives; the failing
   side is the gate's registered failing input.

*Why u × 1.1 has power at every point.* For any input and any layer, act(g)·(1.1 u) − a = 1.1 (act(g)·u − a) + 0.1 a.
By the triangle inequality the failing input's ratio lies within 0.1 ± 1.1 r0, where r0 is the gate's own ratio on the
same layer and windows. The gate passes only if r0 ≤ 0.01 on every layer. So wherever the gate passes, u × 1.1 gives at
least 0.1 − 0.011 = 0.089 on every layer, nine times the bar. This holds for any checkpoint, any windows and any |g|.
The banked controls satisfy the bound on 352 of 352 (point, layer) pairs.

*G0, both sides at every point* (`a5_gate_sides.json` → `G0`; harvest: S3 `results/neurons/w3/grid/<dir>/controls.json`,
first batch, windows 0–7, sha256 as registered in `diag/score05_g0_fail.json`; score-05: the plan's first 8 windows):

| point | pass side: harvest max / score-05 max | u × 1.1 (A5), harvest: min–max over 32 layers (layers over 0.01) | bound 0.1 ± 1.1 r0 holds | u × 1.1 lower bound from score-05's r0 | activation swap (reported): harvest max / score-05 max |
|---|---|---|---|---|---|
| T_15 | 0.00511 / 0.00607 | 0.0944–0.1044 (32) | 32 of 32 | 0.0933 | 0.67612 / 0.63734 |
| T_14 | 0.00539 / 0.00503 | 0.0950–0.1042 (32) | 32 of 32 | 0.0945 | 0.71205 / 0.57245 |
| T_13 | 0.00428 / 0.00476 | 0.0956–0.1038 (32) | 32 of 32 | 0.0948 | 0.38101 / 0.45446 |
| T_12 | 0.00617 / 0.00521 | 0.0958–0.1046 (32) | 32 of 32 | 0.0943 | 0.29941 / 0.36822 |
| T_11 | 0.00591 / 0.00465 | 0.0972–0.1047 (32) | 32 of 32 | 0.0949 | 0.13885 / 0.12976 |
| T_10 | 0.00474 / 0.00501 | 0.0950–0.1052 (32) | 32 of 32 | 0.0945 | 0.08044 / 0.06621 |
| T_9 | 0.00435 / 0.00425 | 0.0957–0.1037 (32) | 32 of 32 | 0.0953 | 0.05192 / 0.05393 |
| T_8 | 0.00362 / 0.00371 | 0.0978–0.1036 (32) | 32 of 32 | 0.0959 | 0.02669 / 0.03174 |
| T_7 | 0.00311 / 0.00347 | 0.0978–0.1026 (32) | 32 of 32 | 0.0962 | 0.01769 / 0.01766 |
| T_6 | 0.00349 / 0.00352 | 0.0965–0.1035 (32) | 32 of 32 | 0.0961 | 0.01126 / **0.00970 (passes)** |
| T_5 | 0.00334 / not reached | 0.0965–0.1032 (32) | 32 of 32 | – | **0.00622 (passes)** / not reached |

Pass side: passes at all 11 points (harvest) and at T_15–T_6 (score-05). Failing side under A5: fails on every layer at
all 11 points. The swap would have stopped the job at T_6 and at T_5; from A5 on it is a readout.

*Every other gate the job runs, both sides* (`a5_gate_sides.json`; the jobs' values from `banks/score-0N/metrics.json`;
score-01 ran the freeze-1 statistic, so its known-number values are not listed):

| gate | where applied | pass side (value, source) | failing side (value, source) | result |
|---|---|---|---|---|
| G_SAME, INPUTS, G_CORPUS | job start | passed in all five attempts (`metrics.json`) | a registered value that a real one never equals (transformers 0.0.0; a zero sha256) | both sides as registered |
| G-REV | each point | registered revision = banked `manifest.json` revision = HF ref today, at 11 of 11 (`rev_cfg`; refs response sha256 `511db444…`, 2026-11-06); the snapshot commit equalled it at T_15–T_6 in score-05 | the commit forty zeros; no registered revision equals it | pass, fail at 11 of 11 |
| G-CFG | each point | the 11 committed branch configs (`offline_gate/hf/`) equal `m1.REF_CONFIG`, no differing key | model_type renamed: fails on 11 of 11 | pass, fail at 11 of 11 |
| G0 | each point | the table above | u × 1.1: the table above | pass, fail at 11 of 11 |
| K0 | T_15 | bit-identical in score-02 to score-05 (p99 0, exact share 1.000; 63 entries) | T_14's banked values at the 63 entries: median 0.3354, p99 3.7936 against 0.021159 (179 times) | pass, fail |
| G-EX, gated (same device, 0.021159, all entries) | T_15, T_6, T_5 | T_15: bit-identical in score-02 to score-05 (2,295 entries). T_6 (5 entries, one pole) and T_5 (10 entries, two poles): not banked, see below | the neighbouring point's bank: T_15 p99 3.787 (179 times), T_6 1.211 (57 times), T_5 0.988 (47 times) | pass at T_15; fail at all three |
| G-EX, reported (A3) | T_14–T_7 | score-05: p99 0.0289, 0.0519, 0.0325, 0.0640, 0.0430, 0.0981, 0.0270, 0.0864; at T_14 the banked L40S repeat gives 0.028857 on the same 406 entries | neighbour's bank, p99 over the read subset: 1.046–14.301 (27 to 369 times 0.038718) | a readout, not a gate |
| G-TOPK | each point | score-05 at T_15–T_7: p99 0 (T_15) and 0.0068–0.0158 (T_14–T_7) against 0.044273 (0.021159 at T_15). T_6, T_5: not banked, see below | neuron axis shifted by one: score-05 p99 1.016–1.403 at T_15–T_7. T_6, T_5: lower bound from the banked top-50 (table below) | pass at T_15–T_7; fail at all 11 |
| I-ID | T_13 | score-04 and score-05: 5,415,936 activations bit-identical, 0 carrier flips | T_14's birth-role activations: p99 3.3811 against 0.021159 | pass, fail |

*G-TOPK at the points no job reached* (`a5_gate_sides.json` → `G_TOPK`; byte-range reads of the banked `topk/layer_LL.npz`
members `tok` and `a`; aggregates only, the quantities the job writes to `metrics.json`). The count is the number of
(pole, slot) entries whose banked top-50 position is a position the plan captures at that point. The failing input
compares the forward's neuron n − 1 at that position with the bank's neuron n. The bank does not hold that value; it
holds a lower bound: the value itself where the position is in neuron n − 1's own top-50, otherwise the distance from
the bank value to the interval that neuron n − 1's top-50 thresholds allow. The p99 of the lower bounds cannot exceed
the p99 of the true values.

| point | layers | positions captured | entries (offline) | entries (score-05) | failing input, lower bound: median, p99 | tolerance | failing input fails |
|---|---|---|---|---|---|---|---|
| T_7 (positive control) | 5 | 116 | 10 | 10 | 0.273, 0.620 (the job measured 1.000, 1.016) | 0.044273 | yes (the job's value) |
| T_6 | 1 (layer 3) | 5 | 4 | not reached | 0.375, 0.623 | 0.021159 | yes, for certain |
| T_5 | 1 (layer 3) | 10 | 4 | not reached | 0.092, 0.218 | 0.021159 | yes, for certain |

*The verdict's gates* (`a5_gate_sides.json` → `construction_A_freeze`, `usable`):

| gate | pass side | failing side | what the bank cannot check |
|---|---|---|---|
| construction control under A at T_15 (≥ 0.90) | freeze half: 82 of 82 in the top 10 (precision 1.000): 74 at rank 1, 8 tied at ranks 2–10, the largest rank 10. The carrier's A equals \|E_c\| for 82 of 82, so the S3 top-50 holds every exemplar position (the read's own positive control) | the opposite pole: 0 of 82 in the top 10 (every rank 22,016; its a⁺ is 0 at every E_c position, so A = 0 and B = 0 by construction) | the test half is computable from the same files, but it is the verdict's and stays blind |
| construction control under B at T_15 | needs every pole's activation at E_c at T_15; the bank holds the carrier's own values and every pole's histogram only | B = 0 for the opposite pole by construction (v = 0) | the pass side under B, on both halves |
| usable births ≥ 50 per half | 82 and 71 if nothing is parked; twins 79 and 70 (H_sel not NOT_RUN) | 30 test births (dry run) | parking: the projection never came near its limit (score-05 at T_6: 2,429 s elapsed + 691 s remaining against 8,100 s) |
| room; freeze records unchanged | pure functions; the dry run | planted in the dry run | none |

*What the bank cannot check, said plainly.*
- The pass side of K0, G-EX and G-TOPK at T_6 and T_5 needs the forward on the L40S. No attempt reached the forward
  there. The gates assume the L40S forward reproduces an L40S bank bit for bit. The evidence: at T_15, K0 (63 entries),
  G-EX (2,295) and G-TOPK (5,675) were bit-identical in all five attempts, and T_5, T_6 and T_15 were banked by one
  harvest run (`/workspace/out/grid_a/`, code `17057bd0`, NVIDIA L40S, torch 2.12.0+cu130, transformers 5.9.0, bf16,
  sdpa, batch 8; the three `manifest.json`). If the forward were not bit-identical there, G-EX at T_6 or T_5 would stop
  the job. That would be a finding about the device, and the rule would not be changed after it.
- The value of u × 1.1 on the job's own windows is not banked. The bound above guarantees at least 0.089 wherever the
  gate passes.
- G-TOPK's failing value is banked only as a lower bound.
- I-ID needs all-neuron activations, which no bank holds. Both its sides come from score-04 and score-05 at T_13, the
  one point where it applies.

*The blind protocol.* From `score-01` to `score-05` the executors read only `metrics.json`, `STATUS`,
`pipeline_status.json`, `heartbeat.jsonl`, `dmon.log`, `job.log` (once each, after the end) and the diagnostics in
`studies/next/m2/diag/`. Of the banked grid harvest A5 read the 11 `controls.json` and 11 `manifest.json` files, and by
byte-range reads the members `tok` and `a` of layer 3 at T_5 and T_6 and of the five layers T_7 scores, and `tok` of
the 26 layers that hold freeze-half births at T_15. G-TOPK is written as aggregates only; no per-pole value of a
test-half unit was computed. The rank files, raw-example files and vectors of every attempt stay unopened on S3. A5 was
written without them.

*Files changed* (from `m2-freeze-1f`): `infra/aws_pg/jobs/m2_score.py` (G0's failing input u × 1.1; the swap reported;
docstring); `infra/aws_pg/jobs/m2_dryrun.py` (`--a5`; REAL A5, 14 checks; JOB A5, 2; FAIL A5, 2; two plants in the
child: the swap made the identity, u × 1.0); new `studies/next/m2/offline_gate/a5_gate_sides.py`. `infra/pg/m2.py`,
`births.json`, the 17 inputs and the step list are unchanged.

*Dry run.* `M2_DRYRUN_OK`, 130 checks, 0 failed (`offline_gate/dryrun_2026-11-06_A5.log`, values
`dryrun_2026-11-06_A5_summary.json`, the tiny job's `metrics.json` `dryrun_2026-11-06_A5_tiny_metrics.json`): the 112 of
freeze 1f and 18 new. In the tiny world u × 1.1 gives 0.0955–0.0995 at the four points (min over layers) and the swap
0.150–0.278. A swap planted with no power (the identity) is reported under the tolerance and the job completes; u × 1.0
planted in place of u × 1.1 stops the job at `t15_G0` ("its failing input passed"). The failing input of the fix: the
no-power swap planted on the freeze-1f code (`git archive m2-freeze-1f infra`) stops the job at `t15_G0` with
`AssertionError: t15_G0: its failing input passed`, score-05's stop (`offline_gate/a5_planted_on_freeze_1f.log`).

*Launch.* `score-06` on `m2-freeze-1g`, the last attempt: the fresh prefix `results/next_m2/score-06/`, the same 17
inputs and step list, `--type g6e.2xlarge,g6e.4xlarge`, `--budget-min 165`, the head's capacity rule (every 180 s for
six tries, then every 600 s until two hours after the first; `VcpuLimitExceeded` retried the same way). The banks are
pulled to `studies/next/m2/banks/score-06/m2/`; freeze and verdict take `--banks studies/next/m2/banks/score-06`. On a
failure: `job.log` once, the cause banked, no relaunch; M2 then ends DOES NOT COUNT with the six-attempt record.

---

## 4. Gates

Every instance gate is asserted in the job. The step exits non-zero, `pipeline.py` aborts the launch, and STATUS shows
`exit≠0`. Each gate's failing-input value is written to `out/score/m2/metrics.json`, and the verdict reads every gate
again (`m2.gates_ok`).

| gate | where | pass | registered failing input (shown failing in the dry run) | if it fails |
|---|---|---|---|---|
| G-SAME | job start | torch, transformers = 2.12.0+cu130, 5.9.0 | transformers registered as 0.0.0 | stop |
| INPUTS | job start | `births.json` and the 16 `exemplar_a_tNN.npz` have their registered sha256 | a zero sha256 | stop |
| G-CORPUS | after the corpus fetch | tokens sha256 `2aca80ca…` | a wrong sha256 | stop |
| G-REV | after each download | snapshot commit = registered revision | a wrong revision | stop |
| G-CFG | before each load | config = `m1.REF_CONFIG`, no differing key | model_type renamed (in-job); width changed (dry run) | stop |
| G0 | first batch of each point | recomputed a equals the down-projection input within 1e-2 | u scaled by 1.1 (in-job, from A5; at least 0.089 wherever the gate passes); the wrong activation function is run and reported, not gated (freeze 1f's input, §3.8 "A5") | stop |
| K0 | T_15, before any score | §3.1 | T_14's banked values (in-job); T_14's file in T_15's place; positions + 1 (dry run) | stop |
| G-EX | each point | §3.1 | the neighbouring point's values (in-job); T_13's file in T_14's place (dry run) | stop |
| G-TOPK | each point | §3.1 | the neuron axis shifted by one (in-job) | stop |
| I-ID | T_13 | §3.2 | T_14's birth-role activations (in-job) | stop |
| construction control | freeze, verdict | precision at 10 at T_15 ≥ 0.90 under A and B | the opposite pole | DOES NOT COUNT |
| usable births | freeze, verdict | ≥ 50 per half with pre and end records | 30 test births (dry run) | DOES NOT COUNT (missing) |
| room | freeze | each label branch reachable at the test size | (the CLI asserts) | no freeze |
| freeze records unchanged | verdict | sha256 of `ranks_freeze.jsonl` = the frozen one | one byte appended (dry run) | the verdict refuses |

---

## 5. Labels and the verdict, as the script runs it

`python -m infra.pg.m2 verdict --banks studies/next/m2/banks --thresholds studies/next/m2/thresholds.json --out studies/next/m2/verdict_m2.json`

1. The frozen freeze records' sha256 must match (`thresholds.json` → `sha256_ranks_freeze`).
2. Gates: every instance gate passed and every failing input failed, and the job reached its end (`complete`).
   Otherwise **DOES NOT COUNT: gates**.
3. Usable test births (pre and end records present) ≥ 50, else **DOES NOT COUNT: missing**. A parked point removes its
   births.
4. The construction control on the test half ≥ 0.90 under A and B, and its failing input below. Otherwise **DOES NOT
   COUNT**.
5. bar(score) = max(chance 0.00045, the freeze floor's 97.5th percentile, the test floor's 97.5th percentile).
6. H_A and H_B: SUPPORTED if the Wilson lower bound > bar; REFUTED if the point value ≤ bar; UNDECIDED otherwise. Then
   ABOVE_CEILING replaces the label if the pre-role lower bound exceeds the birth-role upper bound.
7. H_AB: paired hit_A − hit_B, bootstrap interval; BEATS (lower > 0), LOSES (upper < 0), TIES.
8. H_sel_B and H_sel_A: paired hit_real − hit_twin over births with a twin; ABOVE_SELECTION, AT_SELECTION,
   BELOW_SELECTION. NOT_RUN if fewer than 50 twins.
9. Exploratory: precision at 10 under ratio, max_abs, density and the column norm. No label.
9b. (A3) The line "cross-device G-EX exceeded the A2 tolerance at points …" from the job's `reports`
    (`m2.cross_device_gex`), with each reported point's p99, max and count above the tolerance. It is never a reason for
    DOES NOT COUNT; the result document states it as a limitation.
9c. (A4) The line "poles with no token of their sign in the scored layers …" from the job's `points`
    (`m2.degenerate_counts`), with the count and the ids per point. Never a reason for DOES NOT COUNT.
10. Headline: DOES NOT COUNT (reasons), or "DOES NOT COUNT until inspected (ABOVE_CEILING)", or
    `H_B …; H_sel_B …; H_A …; H_AB …` (the order of A1 item 3, cosmetic; freeze 1 listed H_A first). Every value is
    written to `verdict_m2.json` whatever the headline.

The dry run produced every branch on planted records (`offline_gate/dryrun.log`, section VERDICT). The cases are a
satisfying case; A never hitting (REFUTED); A equal to B (TIES); A above B (BEATS); twins like the carriers
(AT_SELECTION) and above them (BELOW_SELECTION); too few twins (NOT_RUN); a high floor; ABOVE_CEILING; a failing
construction control; its failing input passing; a failed gate; a failing input that passed; 30 test births; and the
freeze records changed after the freeze.

---

## 6. The split, freeze 2 and room

**The split, fixed before any birth was read**: even neuron index → freeze (82 births), odd → test (71). Both poles of
a neuron fall in the same half, so the halves share no pole. By birth point: freeze T_8 2, T_10 3, T_11 5, T_12 18,
T_13 18, T_14 20, T_15 16; test T_6 1, T_8 4, T_9 4, T_10 5, T_11 12, T_12 13, T_13 12, T_14 13, T_15 7
(`births.json` → `by_k`). The brief suggested odd and even checkpoints. That split was not used, for two reasons. All
of one half's scoring points would be absent from the other, so no frozen threshold could be checked on the points it
is applied to. And it would put all 23 T_15-born births, a special case of the rule (present at T_15 only), in one
half.

**What was read before this spec**, freeze half only: the carrier's own ratio at T_(k−1) and A_own (§0.4). No
competitor, so no rank and no hit.

**Freeze 2** (`m2.freeze`) refuses unless:
- the gates pass;
- the freeze half has ≥ 50 usable births;
- the construction control fires under A and B with its failing input below;
- the room check passes.

It writes `thresholds.json` with:
- the freeze floors;
- the sha256 of `ranks_freeze.jsonl` and of the job's `metrics.json`;
- the freeze half's full summary, labelled DISCOVERY: every precision, both contrasts, the twins.

**Room.** On planted outcomes at the test size (71), the frozen labels reach SUPPORTED (all hits) and REFUTED (none),
and the paired labels reach BEATS, LOSES and TIES (`m2.room_check`, dry run). With 71 births a paired difference of
0.15 is about the interval's half-width (the dry run's tie case gave [−0.18, +0.13]). So H_AB and H_sel can resolve
moderate differences in either direction.

---

## 7. Unblinding order

1. Freeze 1: `M2_DRYRUN_OK`; the head reviews; commit; tag `m2-freeze-1`.
2. L1 (`score-01`): the one launch. During the run the executor reads only `m2/metrics.json` (phases, gates,
   timings). No rank file and no vector file is opened.
3. Pull only `m2/metrics.json`, `m2/ranks_freeze.jsonl` and `m2/samples/raw_examples_freeze.jsonl`. Open the freeze
   examples (§10) and answer the sample questions in the M2 journal. Then run `m2 freeze`, commit `thresholds.json` and
   tag `m2-freeze-2`.
4. Pull `m2/ranks_test.jsonl` and `m2/samples/raw_examples_test.jsonl`. Run the verdict once. A rule found wrong
   afterwards gets a dated exploratory deviation; the frozen rule still decides the headline.
5. A fresh Opus agent, blind to this code, recomputes the headline numbers from `m2/vectors/` and the S3 grid arrays,
   on an instance, and writes `studies/next/m2/READ_m2.md`. Fable's read is the different-model read. The result
   document comes after both.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | E_c is selected from the carrier's own T_15 top-50 | raises the carrier's pre-birth ranks through persistence alone | the twins (§3.3) give the selection-only rate; the claim is worded by H_sel |
| 2 | birth dates are mostly set by cluster size and rank (`10_RECOUNT2.md` §8) | a "birth" may be a size effect | only the 153 births later than chance; the twins match size and date |
| 3 | most poles score 0 under A | ties | ties count against the carrier and every matched pole |
| 4 | B is read at histogram resolution (bins of 5.9 %, log-linear inside) | small ordering noise between poles in one bin | the dry run bounds the error by one bin's share; the ratio score (exact numerator) is recorded beside it |
| 5 | activations from an L40S forward, histograms from a Blackwell harvest at T_7–T_14 | B mixes two devices | G-EX at the cross-device tolerance; relative activation differences of 0.4 % move a percentile far less than the gaps between leading poles; the ratio uses the same mix |
| 6 | 153 births on 140 carriers | births of one pole are not independent | stated; the halves share no pole |
| 7 | the clusters come from an LLM clusterer | an incoherent cluster behaves like a random group | biases H_sel toward AT_SELECTION; stated |
| 8 | only one layer, one basis (L1 type) | a job carried by a spread direction is invisible | stated (§3.7); REFUTED means "no single pole leads" |
| 9 | G-EX covers grid poles only | an indexing error on other neurons | G-TOPK covers non-grid poles at their banked top-50 positions |
| 10 | one model, one seed of training | no replicate run | stated as a limit |
| 11 | the wrong weights or windows | wrong activations | G-REV, G-CFG, K0, G-EX, G-TOPK |

---

## 9. Runbook

### 9.1 Checklist before the launch

- Read `.claude/skills/aws-pg/SKILL.md` and the docstrings of `infra/aws_pg/jobs/pipeline.py` and
  `infra/aws_pg/jobs/m2_score.py`. Profile `fuelos`, region `us-east-1`, written explicitly on every call.
- Worktree `../interp_research-m2`, branch `m2`. Commit named files only.
- Liveness: an instance id from `./infra/aws_pg/pg ps`, or the `updated_utc` of `out/score/m2/metrics.json` (rewritten
  after every gate and every point). Never poll `job.log`.
- Cost and incidents go to `infra/aws_pg/operations/2026-11-<dd>_next_m2.md`; the science goes to
  `studies/next/m2/JOURNAL.md`.

### 9.2 Code (NEW or REUSED, each checked against the tree at `d604f8f13`)

| file | status | role |
|---|---|---|
| `infra/pg/m2.py` | NEW | births, plan, scores, ranks, floors, labels, freeze, verdict, projection (numpy) |
| `infra/pg/m2_examples.py` | NEW | the example selection and the zero-GPU raw examples |
| `infra/aws_pg/jobs/m2_score.py` | NEW | the job |
| `infra/aws_pg/jobs/m2_dryrun.py` | NEW | the offline gate |
| `infra/aws_pg/steps/next_m2_score-01.json` | NEW | the step list |
| `infra/pg/neurons_harvest.py` (`load_model`, `MLPTap`, `forward`, `g0_ratios`, `g0_pass`, `act_name`, `WRONG_ACT`, `text_tower`, `decoder`, `context`, `dump`) | REUSED, unchanged | the banked harvest's model path |
| `infra/aws_pg/jobs/neurons_harvest.py` (`fetch`, `fetch_corpus`, `FETCH`) | REUSED, unchanged | S3 inputs |
| `infra/pg/neurons_corpus.py` (`load_tok`), `infra/pg/m1.py` (`config_gate`, `REF_CONFIG`), `infra/pg/diff_harvest.py` (`hf_online_setup`), `infra/aws_pg/jobs/pipeline.py` | REUSED, unchanged | corpus, config gate, HF login, step runner |
| `studies/next/10_recount2/common.py` `births_matrix` | REUSED as a copy in `m2.births_matrix` | the dating rule; reproduces 2,454 of 2,454 banked dates (dry run REAL) |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-m2
aws s3 cp s3://pg-lineage-769063704696/results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz \
    $SP/m2b/s3_1c/exemplar_a.npz --profile fuelos --region us-east-1          # A1: the T_14 L40S repeat, 221 KB
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m2_dryrun --bank /Users/abaybektursun/projects/interp_research_banks/neurons \
    --hf $SP/m2/hf --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz \
    --scan studies/next/m2/offline_gate/bank_scan.json --out $SP/m2b/dry                    # --scan: A4
```
A4: `bank_scan.json` is written by `PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/offline_gate/bank_scan.py --bank
$BANK --births studies/next/m2/births.json --out studies/next/m2/offline_gate/bank_scan.json` (byte-range reads on S3).
A5: `a5_gate_sides.json` is written by `PYTHONPATH=. $SP/m1_venv/bin/python studies/next/m2/offline_gate/a5_gate_sides.py
--bank $BANK --births studies/next/m2/births.json --repeat $SP/m2b/s3_1c/exemplar_a.npz --out
studies/next/m2/offline_gate/a5_gate_sides.json` (S3 reads of controls, manifests and topk members; one HF refs read),
and the dry run takes `--a5 studies/next/m2/offline_gate/a5_gate_sides.json`.
It must end with `M2_DRYRUN_OK`. The record is `M2_OFFLINE_GATE.md`.

### 9.4 S3 layout

One attempt id per launch, never reused: `s3://pg-lineage-769063704696/results/next_m2/<attempt>/`, `<attempt>` =
`score-NN`. The prefix `results/next_m2/` was empty on 2026-11-05.

```
results/next_m2/inputs/{births.json, exemplar_a_t00.npz … exemplar_a_t15.npz}
results/next_m2/score-NN/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_m2/score-NN/out/pipeline_status.json
results/next_m2/score-NN/out/score/m2/{metrics.json, ranks_freeze.jsonl, ranks_test.jsonl}
results/next_m2/score-NN/out/score/m2/samples/{raw_examples_freeze.jsonl, raw_examples_test.jsonl}
results/next_m2/score-NN/out/score/m2/vectors/tNN/<unit>_<role>.npz      (a [m, 11008] fp32, tok [m]; about 270 MB)
```

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
BANK=/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid
```

**S1. Freeze 1 and pre-launch (laptop).**
1. §9.3 to `M2_DRYRUN_OK`. Commit, tag `m2-freeze-1`, and record the tag, SHA and UTC in §0.
2. `./infra/aws_pg/pg ps` and `./infra/aws_pg/pg audit`: no stray instance.
3. `aws s3 ls $B/results/next_m2/ --profile fuelos --region us-east-1` lists nothing.
4. Upload the inputs:
   ```
   aws s3 cp studies/next/m2/births.json $B/results/next_m2/inputs/births.json --profile fuelos --region us-east-1
   for d in 1/out/t00_step0 1/out/t01_step1000 1/out/t02_step2000 1/out/t03_step3000 1/out/t04_step5000 1/out/t05_step8000 \
            1/out/t06_step12000 1b/out/t07_step20000 1b/out/t08_step35000 1b/out/t09_step60000 1b/out/t10_step100000 \
            1b/out/t11_step180000 1b/out/t12_step320000 1b/out/t13_step560000 1b/out/t14_step1000000 1/out/t15_step1413814; do
     t=$(basename $d | cut -c2-3)
     aws s3 cp $BANK/$d/poles/exemplar_a.npz $B/results/next_m2/inputs/exemplar_a_t$t.npz --profile fuelos --region us-east-1
   done
   ```

**S2. L1, the one launch (`score-01`; 1.56–1.72 GPU-h projected; job budget 150 min; launcher budget 165 min).**
```
aws s3 cp infra/aws_pg/steps/next_m2_score-01.json $B/results/next_m2/score-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_m2/score-01/steps.json" \
    --no-weights --type g6e.2xlarge --s3-prefix results/next_m2/score-01 --budget-min 165 --json
./infra/aws_pg/pg watch results/next_m2/score-01
```
On a capacity error, retry on `g6e.4xlarge` (the same L40S). The first gate to read is K0 in `m2/metrics.json`. Then
`measured.tokens_per_s` and `projection`.

**S3. Freeze 2 (laptop, freeze half only).**
```
mkdir -p studies/next/m2/banks/m2/samples
for f in metrics.json ranks_freeze.jsonl samples/raw_examples_freeze.jsonl; do
  aws s3 cp $B/results/next_m2/score-01/out/score/m2/$f studies/next/m2/banks/m2/$f --profile fuelos --region us-east-1; done
PYTHONPATH=. python3 -m infra.pg.m2 freeze --banks studies/next/m2/banks --out studies/next/m2/thresholds.json
```
Before the freeze, open `raw_examples_freeze.jsonl`. For one birth scored right and one scored wrong under each of A
and B, check three things and answer them in `studies/next/m2/JOURNAL.md`: the exemplars decode to the cluster's name;
the top-10 poles' contexts are what their scores claim; the matched pole's contexts read as generic. Then commit
`thresholds.json` and tag `m2-freeze-2`.

**S4. Unblind (laptop).**
```
for f in ranks_test.jsonl samples/raw_examples_test.jsonl; do
  aws s3 cp $B/results/next_m2/score-01/out/score/m2/$f studies/next/m2/banks/m2/$f --profile fuelos --region us-east-1; done
PYTHONPATH=. python3 -m infra.pg.m2 verdict --banks studies/next/m2/banks --thresholds studies/next/m2/thresholds.json \
    --out studies/next/m2/verdict_m2.json
```
Run once, then fill §12.

**S5. The independent recount** (a fresh Opus agent, its own code, on a CPU or GPU instance reading `vectors/` and the
grid arrays; about 0.1 instance-hours) and Fable's read.

### 9.6 Budget, stop rule, park order

| launch | contents | GPU-h at the lowest banked rates (3,215 tokens per second, 143 MB/s) | at the measured T_15 rates (3,334, 234) |
|---|---|---|---|
| L1 `score-01` | boot, corpus, 11 points (S3 reads, download, load, gates, 4,551 window forwards), identity re-run | 1.72 | 1.56 |

Per point (`projection.json`): the weight download (14.61 GB); 90 s for load and gates; 30 s of S3 reads; the windows
at the banked harvest rate; 30 s of writes. T_15 holds 1,732 of the 4,551 windows (about 21–23 min). At $2.24 per hour
the launch is about $3.5–3.9. On g6e.4xlarge ($3.00424) it is $4.7–5.2. The plan's 5–8 GPU-h (12 with gradients) is not
needed: there are 11 loads, not about 30, and no gradients. The harvest rate includes the all-pole top-K bookkeeping,
which this job does not do, so the forward is likely faster than projected; that is not measured.

**Ceiling: 3.0 GPU-h for M2**, which allows one relaunch after an infrastructure failure. The launcher's kill limit is
165 min.

**Note 2026-11-05 21:42 CST (2026-11-06 03:42Z): the ceiling is lifted.** The user removed the per-study GPU-hour
ceilings: "Remove the GPU limits" (the user's words, relayed by the head to the third executor). M2's 3.0 GPU-h ceiling
no longer applies. The per-launch hard kill stays as protection against a runaway job: `--budget-min 165` for the
launcher, 150 min in the job's stop rule. Spent when the note was recorded: 0.690–0.738 GPU-h over `score-01` and
`score-02`.

**Stop rule, in the job.** After T_15 the job measures its fixed cost per point and its seconds per window. Before each
later point it projects the rest. If the elapsed time plus that projection exceeds 0.9 × 150 min, this point and every
later one are PARKED. The job then ends normally with `parked` listed. The dry run planted a 0.6 s budget, which parked
T_14, T_13 and T_12 and left those births without a pre record.

**Park order.** The plan runs T_15 first (K0 and every end role), then T_14 down to T_5. Parking therefore removes the
lowest points first, which hold the fewest births (test half: 1 birth at T_6, 4 at T_8, 4 at T_9). The identity pair
(T_13) comes third and is not parked in any realistic case. If parking leaves fewer than 50 usable test births, the
verdict is DOES NOT COUNT (missing), and a second launch for the parked points is the user's decision.

### 9.7 Artifacts

- On S3 (§9.4): the gates with their failing-input values, the records, the activations, the examples.
- On the laptop: `studies/next/m2/{births.json, projection.json, samples/raw_examples_zero_gpu.{md,json},
  offline_gate/, banks/, thresholds.json, verdict_m2.json, JOURNAL.md, READ_m2.md}`.
- Operations: `infra/aws_pg/operations/2026-11-<dd>_next_m2.md` (instance id, type, launch and STATUS UTC, billed hours
  against the projection).

---

## 10. The ten raw examples

**Zero-GPU part, written now** (`studies/next/m2/samples/raw_examples_zero_gpu.md` and `.json`, by
`python -m infra.pg.m2 examples`). The selection rule is frozen in `m2_examples.select`. It draws ten freeze-half
births: five with A_own ≥ 1 and five with A_own = 0. Each record shows:
- the birth and its exemplar contexts at T_15;
- the carrier at T_(k−1): its median a⁺, q99, q999, q9999, the percentile bracket they give, A_own and its rank under A
  among the layer's 20 grid poles, and its own top-5 contexts;
- the carrier at T_k;
- the twin;
- the matched random-pole reading.

What they show:
- In all ten births the carrier at T_(k−1) is below its own threshold. The ratio runs from 0.07 (L10N9052+ c2, the NPR
  transcript credit) to 0.96 (L24N5066− c1). Seven of ten sit in the [0.99, 0.999) percentile bracket.
- The five A_own = 0 births are scored wrong under A for certain: rank 22,016.
- The five A_own ≥ 1 births are first among the 20 grid poles of their layer under A. Their rank among 22,016 needs S3
  topk and is PENDING.
- At T_(k−1) the carriers' own top-5 contexts do not show the job. For example, L03N7746+ c0 ("C corp") has "Bitstamp",
  "Tapfiliate" and "Invalidate" at T_14, while two of its five C-corp exemplars are already in its top-50.
- Every matched grid pole has A = 0 on the exemplars, and each reads as its own unrelated token family.
- In seven of nine births with a twin, the twin's ratio at T_(k−1) (0.78–0.999) is higher than the carrier's. In three
  the twin's own A_own (2–3) exceeds the carrier's. This is the observation behind the H_sel prior.

**GPU part, written by the job**:
- `m2/samples/raw_examples_freeze.jsonl` holds the same ten births. Each record has the full top-10 lists under A and
  B, each listed pole's top-3 contexts at T_(k−1), the carrier's rank and value, the decoded exemplars, and the first
  matched pole's top-3 contexts.
- `raw_examples_test.jsonl` holds ten test births drawn by `default_rng(1)` without reading their data. It is opened at
  unblinding.
- Beside each score, every record in `ranks_*.jsonl` carries the top-10 lists. The verdict cites one hit and one miss
  per score.

---

## 11. The three dimensions

- **Predictive.** This is a held-out prediction made before the measurement. Each score is computed at T_(k−1) from that
  point's weights and banked arrays only. The test half's carriers are scored once, under rules frozen on the other
  half. The prediction lands if H_A or H_B is SUPPORTED on the test half. Whether it lands for a reason beyond the
  selection is H_sel.
- **Novel.** Moderate. The question has been asked at 410M (wdd e587, top-ten 0.84 two checkpoints before) and for SAE
  features in Pythia (2605.18789, not read in full). Two things are open as far as the searches went: a ranking of the
  eventual carrier among all 22,016 poles of a 7B layer before a dated birth, and the selection control (the twins).
  If H_sel_B is AT_SELECTION, the novel part is the negative: the 410M-style forecast at 7B is explained by how jobs
  are defined.
- **Practical.** Small and stated as such. The measured win would be H_AB TIES or BEATS: competitor sets for J12 from
  the banks at 0 GPU-h instead of one forward per point. No engineer's measure moves.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

Row order: B first, then the boundary H_sel_B, then the reference A (head's decision, 2026-11-05, before freeze 1).

| item | rule or prior | observed | verdict |
|---|---|---|---|
| K0 | §3.1 (prior: bit-identical 0.7) | 63 entries on 10 poles, p99 0, exact share 1.000; failing input (T_14's values) p99 3.7936 against 0.021159 (`metrics.json` → `gates` → `K0`; `verdict_m2.json` → `gates` pass) | pass; bit-identical, as the prior's 0.7 favoured |
| G-EX, G-TOPK at 11 points | §3.1 | G-EX gated at T_15, T_6, T_5: bit-identical (2,295, 5, 10 entries); failing input p99 3.787, 1.211, 0.988 against 0.021159. G-EX at T_14–T_7 reported (A3), row below. G-TOPK at all 11 points: p99 0 at T_15, T_6, T_5 (5,675, 4, 4 entries; exact share 1.000) and 0.0068–0.0158 at T_14–T_7 (934 to 10 entries) against 0.044273; failing input p99 0.998–1.403 (`metrics.json` → `gates`) | pass at every point; every failing input fails |
| I-ID | §3.2 (prior: zero flips 0.9) | 66 units, 5,415,936 activations, p99 0, exact share 1.000, 0 carrier flips under B; failing input p99 3.3811 (`metrics.json` → `gates` → `I_ID`) | pass; zero flips, as the prior favoured |
| construction control, A and B, both halves | ≥ 0.90 (prior 0.95) | test: B 71/71, A 71/71; freeze: B 81/82 = 0.988, A 82/82 (as `a5_gate_sides.json` gave from the bank); opposite pole 0 of 71 and 0 of 82 under both (`verdict_m2.json` → `test` → `end_*`; `thresholds.json` → `discovery`) | fired on both halves; failing input below |
| floors, A and B, both halves | §3.4 | 97.5th percentile of 1,000 matched-pole replicates: freeze 0.0122 under A and B (1 of 82); test 0.0 under both (max 0.0141); bar = max(chance 0.000454, the two floors) = 0.0122 under A and B (`verdict_m2.json` → `bars`) | measured |
| precision at 10 under B, freeze (DISCOVERY) | 0.3–0.7 | 39/82 = 0.476 [0.371, 0.582] (`thresholds.json` → `discovery` → `pre_B`) | inside the prior range |
| precision at 10 under A, freeze (DISCOVERY) | ≤ 0.122 (A_own bound) | 7/82 = 0.085 [0.042, 0.166] (`discovery` → `pre_A`) | inside the bound |
| H_B, test (primary, the headline; head's decision, 2026-11-05, before freeze 1) | prior SUPPORTED 0.97, at 0.3–0.7 | 42/71 = 0.592, Wilson 95 % [0.475, 0.698], bar 0.0122 (`verdict_m2.json` → `labels` → `H_B`) | **SUPPORTED**; the value inside the prior range |
| H_sel_B, test (the boundary of the claim) | prior AT 0.45, ABOVE 0.30, BELOW 0.25 | twins 53/70 = 0.757 [0.645, 0.842] at the pre role under B; hit_real − hit_twin −0.171 [−0.286, −0.057] over 70 pairs (carrier only 3, twin only 15) (`labels` → `H_sel_B`, `test` → `pb_pre_B`). Under A (reported): twins 14/70, difference −0.029 [−0.157, 0.100], AT_SELECTION | **BELOW_SELECTION**, the branch the prior rated least likely |
| H_A, test (reference) | prior SUPPORTED 0.9, at 0.04–0.12 | 13/71 = 0.183, Wilson 95 % [0.110, 0.288], bar 0.0122 (`labels` → `H_A`) | **SUPPORTED**; the value above the prior range |
| H_AB, test | prior LOSES 0.85 | hit_A − hit_B −0.408 [−0.521, −0.296] over 71 births (A only 0, B only 29) (`labels` → `H_AB`) | **LOSES**, as the prior favoured |
| birth-point precision, A and B | measured | test: B 65/71 = 0.915 [0.828, 0.961], A 31/71 = 0.437 [0.327, 0.552]; freeze: B 71/82 = 0.866, A 34/82 = 0.415 (`test` → `birth_*`; `discovery`). The pre-role lower bounds (B 0.475, A 0.110) lie below the birth-role upper bounds (0.961, 0.552) | the ceiling; no ABOVE_CEILING |
| exploratory: ratio, max_abs, density, column norm | job-free ones near chance by the pole draw | test: ratio 41/71 = 0.577 [0.462, 0.685]; max_abs, density, column norm 0/71 (upper bound 0.051) (`verdict_m2.json` → `exploratory`) | the job-free three at 0, as predicted; no label |
| cross-device G-EX at T_7–T_14 (A3, reported; `verdict_m2.json` → `cross_device_gex`) | the A2 statistic against 0.038718; score-02 and score-03 exceeded it at T_13 | over the tolerance at T_13 (p99 0.0519, 11 of 591 read entries above), T_11 (0.0640, 8 of 538), T_10 (0.0430, 4 of 378), T_9 (0.0981, 8 of 240), T_7 (0.0864, 4 of 114); under it at T_14 (0.0289, 4 of 406), T_12 (0.0325, 3 of 650), T_8 (0.0270, 0 of 137); median d 0.0044–0.0055 at every point; the score-05 values to every digit | a limitation, stated; never a reason for DOES NOT COUNT |
| G0 at 11 points (A5): the gate, its failing input u × 1.1, the reported activation swap (`metrics.json` → `gates` → `tNN_G0`, `reports` → `tNN_G0_act_swap`) | gate ≤ 0.01; u × 1.1 at least 0.089; the swap passes at T_6 and T_5 (banked) | gate max 0.00335–0.00607; u × 1.1 min over layers 0.0943–0.0980, max 0.1034–0.1051, fails at all 11; the swap 0.637 (T_15) down to 0.0177 (T_7), under 0.01 at T_6 (0.00970) and T_5 (0.00729) | as predicted at every point |
| poles with no token of their sign in the scored layers (A4; `verdict_m2.json` → `degenerate_poles`) | the bank scan: T_8 1 (L07N640−), T_10 1 (L24N9826+), 0 elsewhere | T_10 1 (L24N9826+), T_8 1 (L07N640−), 0 at the other 9 points | as the scan predicted |
| GPU-h | 1.56–1.72 projected; ceiling 3.0, lifted 2026-11-05 21:42 CST (§9.6) | score-06: 0.641–0.669 GPU-h ($1.44–1.50, g6e.2xlarge), job time 2,214 s; M2 over six attempts 3.324–3.467 GPU-h, $8.53–8.91 (`infra/aws_pg/operations/2026-11-05_next_m2_score.md`) | the completed launch inside the projection; the total above the lifted ceiling |

**Raw examples at unblinding** (`studies/next/m2/banks/score-06/m2/samples/raw_examples_{freeze,test}.jsonl`; the
answers in `studies/next/m2/JOURNAL.md`, 07:53Z for the freeze half before freeze 2, 07:57Z for the test half). Test
half, one hit and one miss per score:
- B hit, A miss: L24N8193− c1 (`is [[ of]] paramount importance`, pre T_7): rank 2 under B (0.9989); A 0, rank 22,016.
- B miss: L02N4649+ c3 (`DEFAULT [[ CHARSET]]=utf8`, pre T_14): B 0.9991 but rank 597; under A two other poles of
  layer 2 hold 4 and 3 of the exemplar positions in their top-50, the carrier 0.
- A hit: L01N369+ c0 (`Boyfriend, [[ Husband]]`, pre T_11): 3 exemplar positions in its top-50, rank 1 under A, 3 under B.

## 13. References

`06_CRITIC.md` M2; `09_PLAN.md` J9; `10_RECOUNT2.md`; `04_WDD.md` and wdd-descendants atlas 39 (e587 numbers only);
`12_SHAPING_NOTE.md` §4; `M1_SPEC.md` (format, K0 practice). No other citation carries a number here.
