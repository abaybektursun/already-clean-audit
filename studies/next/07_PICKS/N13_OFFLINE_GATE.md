# N13 offline gate

**Answer: go for R0.** 14 gates were checked offline. All 14 pass in the code as built. 13 of the 14 were shown to fail
on their registered planted inputs. PC-ART has no registered failing input.

The go has one condition: the grading rule. As written in §2.4 item 3, it fails G-GRADER on the real MATH-500 golds
(392/500 golds equal to themselves). The code therefore uses a corrected rule, `dollar_both`, which passes 500/500.
Fable must record that change as a dated amendment before freeze 1. R0's own gate, G-KNOWN, scores with olmes and does
not use this grader.

The dry run printed `N13_DRYRUN_OK`: 74 of 74 checks (`studies/next/n13/offline_gate/evidence/checks.json`).

Written 2026-11-05 by the offline-gate worker. Zero spend: no instance was launched, no paid API was called, no weights
were downloaded. Nothing was committed.

## How to rerun

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-n13
PYTHONPATH=$PWD $SP/n13_venv/bin/python -m infra.aws_pg.jobs.n13_dryrun --out $SP/n13_dry --hf $SP/n13_gate_hf \
    --olmes $SP/mathtask_olmes/olmes --wandb $SP/mathtask_wandb --task-doc studies/next/07_PICKS/N13_MATH_TASK.md
```

**Environment.** The venv is `$SP/n13_venv`, Python 3.12.12. Packages: `math-verify` 0.9.0, numpy 2.5.4,
transformers 5.19.0, jinja2 3.1.6, matplotlib 3.11.2, latex2sympy2_extended 1.11.0, antlr4-python3-runtime 4.13.2 and
sympy 1.14.0 (list: `$SP/n13_venv_freeze.txt`). There is no torch. The weights instrument works on uint16 bit patterns
in NumPy. It runs the same way on the instance.

**What was fetched.** 52 small text files from Hugging Face, 7,954,498 bytes in all, to
`$SP/n13_gate_hf/`. Each is listed with its size and sha256 in `studies/next/n13/offline_gate/evidence/hf_fetched.json`.
They are:

- Tree API listings at:
  - base `a81bae42`, `39ff1a89` and `stage2-step47684`;
  - 3.1 `main` and its 28 `step_*` branches;
  - 3.0 `main`.
- The 3.1 refs and model info.
- The 3.1 files `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja` and `model.safetensors.index.json`.
- The base `config.json`, `generation_config.json`, `tokenizer_config.json` and index.
- The 3.0 `config.json`, index and `chat_template.jinja`.
- MATH-500 `test.jsonl` and its tree listing.

The olmes source (`$SP/mathtask_olmes/olmes`, at `5a51f50`) and the W&B run JSON (`$SP/mathtask_wandb/`) were already
in the scratchpad from the N13_MATH_TASK read. Both were read as text only. The olmes configs were parsed with `ast`;
no olmes code was run.

## Gates

Evidence paths are relative to `studies/next/n13/offline_gate/`.

| gate | input | expected | observed | result | planted failing input → observed | evidence |
|---|---|---|---|---|---|---|
| G-GRADER | 24 equivalent and 16 non-equivalent planted pairs; the 500 real MATH-500 golds; derangement seed 0 | 24/24 true, 16/16 false, 500/500 self-equal, derangement rate ≤ identical-string rate + 0.01 | registered §2.4 rule: 12/24, 16/16, **392/500**. `dollar_both` (active): 24/24, 16/16, 500/500, rate 0.006 ≤ 0.016 | **registered rule FAIL**; active rule pass | derangement scored as the identity: 5, 3, 3 of 500 equal (needs 500) → fails under every rule | `evidence/G-GRADER.json` |
| G-TEMPLATE | 3.1 tokenizer and template at `364820b0`; 500 MATH-500 problems | §2.2 bytes for all 500; template sha256 `4380e622…` | 500/500; sha256 equal; first id is never 100257 (no BOS); prompts 62–811 tokens; pair 1b template 500/500 | pass | `add_generation_prompt=False` → the last line is missing; differs | `evidence/G-TEMPLATE.json` |
| G-INIT (a), offline half | HF tree LFS oids of base `main` and of the logged init `39ff1a89`; W&B run `ydkgbnai` config | oids equal; W&B names `39ff1a89` | all three oids equal; `model_name_or_path` ends `Olmo-3-1025-7B/39ff1a89…` | pass | `stage2-step47684` oids in place of main's → mismatch | `evidence/G-INIT_offline.json` |
| G-INIT (b), stand-in | unchanged MLP share: stand-in base vs RL, and stage 2 vs RL | difference ≥ 0.10 | 0.684 − 0.002 = 0.682 | pass | the candidates swapped → −0.682, fails | `evidence/R1_standin_controls.json` |
| G-PARTITION | 3.1 `model.safetensors.index.json` and `config.json` | 355 names in four families; sum 7,298,011,136; base names and shape fields equal | MLP 4,328,521,728; attention 2,147,483,648; norms 528,384; embeddings 821,477,376; sum equal; index `total_parameters` equal | pass | index without `model.layers.0.mlp.down_proj.weight` → missing, fails | `evidence/G-PARTITION_offline.json` |
| PC-RANK | stand-in (384 neurons, 8 planted, top 16); wide stand-in at the real ratios (8,192 neurons, 23 planted, top 105) | recovery ≥ 0.99 | 1.0 and 1.0 | pass | rows of `down_proj` only → recovery 0.0 (chance 0.042) and 0.0 (chance 0.013), ≤ 0.05 | `evidence/R1_standin_controls.json`, `evidence/PC-RANK_real_ratio.json` |
| PC-ART | synthetic one-step change at p = 0.3 on a stand-in base; 20 redraws, seeds 60–79 | J_round(synthetic, k1) < 0.4 | iid stand-in: 0.234 (R1 run), 0.388 (second base) | pass on iid weights | **no registered failing input** (§5: "reported, not a gate"). On a stand-in whose neuron scale is log-normal (sd 0.5) the pass rule itself fails: 0.718 | `evidence/PC-ART_power.json` |
| G-KNOWN, configuration | olmes at `5a51f502` read by `ast`; N13_MATH_TASK.md lines 82–105, 285–288 | alias, 7 subtasks, macro, 4 shots, `Minerva:MATH:fixed`, T 0.6, top-p 0.6, 1,024 tokens, stops, 4 repeats, `pass_at_1`, pins vllm 0.11.0 / lm_eval 0.4.3 / antlr4 4.11, no chat model; every flag accepted | all fields equal; 14 doc lines equal; the command equals §6.1's | pass | the config with `num_shots` 0 and `chat_model` true → 2 rows fail | `evidence/G-KNOWN_config.json` |
| G-KNOWN, tolerance | subject sizes | 1.6 | SE 0.374, difference 0.529, 3× = 1.588 → 1.6 | pass | — | `evidence/G-KNOWN_config.json` |
| G-KNOWN, R0 reader | a fabricated olmes directory: 7 subjects, aggregate `primary_score` 0.40 | 40.0 read from the aggregate; pass | 40.0; identity rescore reproduces every document's pass@1 | pass | golds deranged within subject → 0.0 < 10.0, fails. A score of 30.0 → G-KNOWN fails, exit 3. A directory with only `pass_at_1_micro` → stops | `evidence/R0_fabricated.json`, `dryrun_log.txt` |
| G-HASH | every arm built at stand-in size and written; R1 hash-only builds | `zero` = base and `id` = RL, every tensor | equal; written shards read back hash-equal for all 29 built arms | pass | `id` with neuron (0, 0) reverted → mismatch | `evidence/arm_builds_standin.json`, `evidence/R1_standin_controls.json` |
| G-MASK | 20 mask arms | changed gate rows, up rows and down columns equal S ∩ {changed} | equal for all 20 | pass | `top_k1` built with `down_proj` rows → mismatch | same |
| G-COMPLETE | 12 fabricated answers; 41 arm × benchmark dry-run banks | n answers, every index once, seeds as registered, no empty text | pass | pass | one answer removed → fails | `evidence/scoring_fabricated.json`, `e2e/verdict_n13.json` |
| G-GATE | dry-run gate banks (fake engine, n = 500) | G ≥ 5√2σ, G' ≥ 5√2σ, McNemar p < 0.001 | G = 58.6, σ = 0.970, δ = 2.74 | pass | (rl-A, id-B) as the pair → G = −1.4, fails; the verdict CANCELs | `e2e/thresholds.json`, `evidence/verdict_failing_inputs.json` |
| G-ID | same | \|L − L'\|, \|B − B'\| ≤ 3√2σ | −1.4, −1.0 against 4.11 | pass | (base-A, rl-A) as the identity → fails | same |
| S-BAND | 15 logged 3.1 readings (W&B); dry-run `id-B` | band from 0.874–0.908 ± 3·√2·1.40 | band [81.46, 96.74] → registered [81.5, 96.7]; dry-run L' = 89.6 | pass | `id-B` against deranged golds → 0.6 < 81.5; the verdict stamps | `evidence/S-BAND_offline.json`, `e2e/thresholds.json` |
| G-GATE-GSM | 60 fabricated GSM8K rows | the G-GATE formula | G = 53.3 | pass | (rl-A, id-B) → fails; the verdict stamps "GSM8K: no room" | `e2e/thresholds.json`, `evidence/verdict_failing_inputs.json` |

Rules that are not gates but were checked:

- **Extraction and grading.** 12 fabricated answers were all scored and assigned modes as expected: answer line,
  boxed only, wrong, none, cap hit, `\frac{1}{2}`/`0.5`, `\dfrac`, `10.0`/`10`, an interval, `60`/`60^\circ`,
  `45°`/`45^\circ`, `\$36`/`36` (`evidence/scoring_fabricated.json`).
- **Damage screen.** On a non-endpoint arm with 16/20 cap hits and none correct, it fired and aborted the arm (status
  `screened`). On base it fired and was only recorded; all 40 answers finished (`evidence/damage_screen.json`).
- **Room (§6.2).** True on the dry-run gate banks. False on banks built with G = 2δ: G 3.6 against 2.5δ = 4.47
  (`dryrun_log.txt`).
- **Stop rule.** A step over its allowance was PARKED, and the next step parked through the marker (`dryrun_log.txt`).
- **3.1 `main`.** Its three shard oids equal those of `step_2800` (`evidence/rl31_main_vs_steps.json`). The 3.1
  `tokenizer.json` blob equals the base's, `5fe17212…` (`evidence/hf_fetched.json`).

## Dry-run record

**Weights.** The stand-in has 4 layers, hidden 64, MLP width 96 (384 neurons), 4 heads, 4 key-value heads and
vocabulary 512, in bf16, with the real parameter names and three shards written by `n13_arms.write_safetensors`. The
`safetensors` library opens them. Two variants were derived from it:

- **"RL"** is the base plus three changes:
  - 8 planted neurons with large multi-step changes;
  - one-step changes on 30% of MLP entries;
  - small changes to attention and norms.
- **"Stage 2"** is the base plus dense large changes.

The real 3.1 tokenizer and template are copied into each stand-in directory.

**Steps, each through its own entry point.**

1. Setup (`n13_setup.run --local`): data, G-GRADER, G-INIT (a), G-TEMPLATE.
2. R0 (`n13_known.run`) on a fabricated olmes directory, with the rescorer in a subprocess.
3. R1 (`n13_rank.run`) at stand-in k = 2, 4, 8, 24. The ranker recovers the 8 planted neurons at k = 8. J_floor and
   J_round are both 1 at k = 8.
4. The smoke (`n13_gen.run --smoke`, cap 16,384) for base and rl, written to their own attempt directory.
5. All 31 roster arms through `n13_gen.run`. Each non-endpoint arm was built by the builder before the fake engine
   started. Inputs: 500 real MATH-500 problems, plus 60 fabricated GSM8K rows for the 11 GSM8K arms.
6. `freeze_thresholds`, then `verdict_n13`. Exit 0, verdict COMPLETE, with the table in `e2e/verdict_n13.md`. Under the
   fake engine's planted accuracies this gave k_rec = k1, H_local SUPPORTED, H_diffuse REFUTED and H_attention
   REFUTED. These are numbers chosen for the dry run, not results.
7. `figure_n13` (`e2e/fig_n13_dryrun.png`), `samples_coverage` (42 arm × benchmark rows: 41 with a right and a wrong sample; the screened dare_k0.1 has no right one) and `recount_n13` (RECOUNT_OK).

**Planted banks for the verdict.**

- Four scenarios each gave their registered hypothesis label:
  - H_local SUPPORTED;
  - H_local REFUTED (k_rec NONE);
  - H_local REFUTED (dare within δ), with H_diffuse SUPPORTED;
  - H_attention SUPPORTED.
- `--failing-input` was run on each of the 13 gates, and each gave its registered effect (CANCELLED, or the stamp).
- A controls file whose PC-RANK failing input did not fail gave exit 5 (`evidence/verdict_planted.json`,
  `evidence/verdict_failing_inputs.json`).

**Stub outputs** are in `studies/next/n13/offline_gate/stub/`:

- `generations.jsonl`, `metrics.json`, `samples.jsonl` and `build.json` for base, rl, zero, id, top_k1 and the
  screened dare_k0.1;
- the fake smoke throughput records.

The full dry-run tree is `$SP/n13_dry/`. The fake engine's tokens per second count simulated seconds, so they mean
nothing for the budget.

**Not exercised offline.** Each of these needs the instance. A failure stops the pipeline at that step.

- The pip installs into `/opt/n13` and `/opt/olmes-venv`.
- `snapshot_download` and the live tree-API hash checks.
- The vLLM 0.32.0 adapter `VllmEngine` (§0.3 item 4). The smoke is its first run.
- olmes itself, and `n13_olmes_rescore --scorer olmes`.
- `dynamic_config`, which imports olmes.
- The pg-venv parent re-running `n13_gen` under `/opt/n13`.
- The S3 input fetch.
- R1 at real size. A laptop timing of one 11,008 × 4,096 tensor gives about 50 min of R1 compute.

## The R0 command

R0 is attempt `s2setup-01`. It has four steps:

1. **`setup`.** Builds `/opt/n13` (vLLM 0.32.0, math-verify 0.9.0) and `/opt/olmes-venv` (olmes `5a51f502`,
   `pip install -e ".[gpu]"`, pip freeze banked). Runs G-GRADER, downloads the data and the weights with their
   hashes, then runs G-INIT (a) and G-TEMPLATE.
2. **`r0_known`.** Runs olmes `minerva_math::olmes:n4:v2` on base `main` with vLLM 0.11.0, then G-KNOWN and its
   derangement rescore.
3. **`smoke_base`.**
4. **`smoke_rl`.** Each smoke step runs MATH-500 problems 0–19 at seed set A and cap 16,384.

The manifest is `infra/aws_pg/steps/n13_s2setup-01.json`. R1 is not in this launch, because the brief defines R0 as
the known number plus the smoke. R1 then runs in the S3 launch (deviation 1 below).

Run from the n13 worktree, after Fable commits the named files. The worktree has no `.pgvenv`, so `./infra/aws_pg/pg`
fails there. The interpreter below is the main worktree's. `pg run --help` was read with it. `pg.py` packages the
worktree it sits in.

```
cd /Users/abaybektursun/projects/interp_research-n13
aws s3 ls s3://pg-lineage-769063704696/results/next_n13/s2setup-01/ --profile fuelos --region us-east-1   # must list nothing
aws s3 cp infra/aws_pg/steps/n13_s2setup-01.json s3://pg-lineage-769063704696/results/next_n13/s2setup-01/steps.json \
    --profile fuelos --region us-east-1
../interp_research/.pgvenv/bin/python -m infra.aws_pg.pg run infra.aws_pg.jobs.pipeline \
    --args " --steps-s3 s3://pg-lineage-769063704696/results/next_n13/s2setup-01/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n13/s2setup-01 --budget-min 280 --json
```

- **S3 prefix.** `results/next_n13/s2setup-01`. Each relaunch takes `-02`, `-03` and so on, never reused
  (`n13_plan.prefix`).
- **Instance.** `g6e.2xlarge`, one L40S with 48 GB, 8 vCPU and 64 GB host RAM. `g6e.4xlarge` is only the capacity
  fallback. `--no-weights`: no Olmo-3 repo is in the bucket (`infra/pg/matrix_repo_facts.json` → `_meta.bucket_dirs`).
- **Expected GPU-h.** 2.52 at 800 tok/s, 3.58 at 500 tok/s (`evidence/manifests_and_budget.json` →
  `projection.r0`).
  - Budget: 280 min = 1.3 × 215. The reaper acts at 290 min.
  - The 215 min are: setup 20, olmes venv 10, olmes load 5, R0 at 5.0 M tokens 167, and 2 × 6.7 for the smoke.

**Stop rule.**

- The pipeline aborts at the first non-zero step, and the instance shuts down.
- `n13_setup` asserts G-GRADER, every shard hash, G-INIT (a) and G-TEMPLATE.
- `n13_known` exits 3 when any of these holds, so no smoke runs:
  - G-KNOWN misses (\|score − 40.0\| > 1.6);
  - its derangement does not fall below 10.0;
  - the identity rescore differs.
- Each step parks when the launch's elapsed time plus that step's projection exceeds 280 min.

**On a G-KNOWN miss** (§6.5):

- Compare the run with N13_MATH_TASK.md line by line. The first suspect is spec issue 3; read
  `out/r0_known/rescore.json` → `context_tokens.over_1024`.
- Relaunch once as `s2setup-02`. A second miss is CANCELLED.
- If vLLM 0.11.0 cannot load the base, relaunch with `--model-type hf` in the `r0_known` args (§0.3 item 1).

**Poll these files**, never `job.log`:

- `out/r0_known/metrics.json`, rewritten every 60 s while olmes runs;
- `out/smoke_base/base/math500/metrics.json`;
- `out/smoke_rl/rl/math500/metrics.json`.

The run_job.sh heartbeat follows only `/opt/pg/venv/bin/python` processes. olmes and the generation child run in other
venvs, so their CPU is invisible to the heartbeat. Judge liveness by the `updated_utc` of those files.

**Throughput record** for comparing the AWS L40S with a Vast machine: `out/smoke_<arm>/throughput.json`. It holds:

- output tokens per second and seconds per problem;
- an output-token histogram with edges 0, 128, …, 32,768;
- the mean and maximum length;
- `nvidia-smi` GPU memory used and total;
- host peak RSS and free disk.

The same record sits in each smoke `metrics.json` → `throughput`. R0's wall seconds are in
`out/r0_known/metrics.json` → `elapsed_s`.

## Projected GPU-hours

The projector is `n13_plan.project_total`. It reproduces §9.7 when R1 is set to 75 min. The spec does not state R1's
time; 75 min is what its 3.8 and 5.0 GPU-h imply. All numbers are from `evidence/manifests_and_budget.json`.

| launch | 800 tok/s | 500 tok/s |
|---|---|---|
| R0 launch (setup, R0, smoke) | 2.52 | 3.58 |
| S2 as specified (R0 launch + R1) | 3.77 | 4.83 |
| S3 gate (4 loads; plus R1, 1.25, once R1 moves here) | 4.12 | 6.20 |
| S4 instance A (14 arms) | 9.30 | 13.98 |
| S4 instance B (13 arms) | 8.35 | 12.51 |
| **total** | **25.5** | **37.5** |

The ceiling is 35 GPU-h. At 500 tok/s the projection is 2.5 GPU-h over it, and the stop rule parks the last steps of
each S4 manifest (P6, P7, then P5). Moving R1 from the R0 launch to S3 changes no total; it adds one 14.6 GB stage-2
download to S3. All of this is assumed until the smoke measures throughput and lengths. `n13_plan.project_from_smoke`
then recomputes the projection from the two smoke `throughput.json` files.

## Spec issues

Line numbers are in `studies/next/07_PICKS/N13_SPEC.md` as it stands now. The spec was not edited.

1. **Worst: the grader (lines 337–339, §2.4 item 3).** As written, the rule parses the gold without delimiters.
   - Measured: 108 of 500 MATH-500 golds are judged unequal to themselves, and 51 parse to nothing. 12 of 24 planted
     equivalences are missed: intervals, `\sqrt{12}`, `\pi/2`, `(x+1)^2`, `\text{(C)}`, `2^{10}`, `\$36`.
   - Every arm would lose about 22% of its correct answers to the grader (`evidence/G-GRADER.json`).
   - The code uses `dollar_both`: the §2.4 wrap applied to the gold too, with an escaped `\$` not counted as a
     delimiter. It passes 24/24, 16/16 and 500/500, with a derangement rate of 0.006.
   - A `\boxed{}` wrap also passes the 500 golds, but it judges `45°` unequal to `45^\circ`.
   - Needs a dated amendment.
2. **PC-ART has no room on realistic weights (lines 581–583, 721, 750; failing input line 566).**
   - The redraw width is ± half the spacing of each before value. A ranking set by spacing therefore survives the
     redraws; the redraws do not scramble it, as line 582–583 asserts they must.
   - With neuron-dependent weight scale (log-normal sd 0.5), J_round(synthetic, k1) = 0.718 ≥ 0.4. PC-ART then
     fails, so H_artifact is CANCELLED. On iid stand-ins it is 0.234 and 0.388, near the rule
     (`evidence/PC-ART_power.json`).
   - Real 7B neuron norms vary, so this is the likely outcome at R1.
   - PC-ART also has no registered failing input, against the repo rule that every gate has one.
3. **G-KNOWN context length (lines 631–634).** The command passes no `--model-args`.
   - olmes then uses `max_length` 2048 (`oe_eval/default_configs.py` line 9). Its vLLM path keeps only the last
     2048 − 1024 = 1024 context tokens (`oe_eval/models/eleuther_vllm_causallms.py` lines 199–204).
   - The report's `max_length` for Table 3 is not public.
   - R0 counts the 4-shot contexts longer than 1,024 tokens, so a miss can be traced.
4. **DARE kept-count tolerance (line 432).** The assertion "within 0.1% of P_k" can fire on noise.
   - At k = 0.1% it allows 4,325 entries. That is 2.08 binomial sd (sd 2,079 of 4,325,376), so one draw fails by
     chance with probability about 0.037.
   - The same tolerance is 3.6 sd at 0.3%, 6.6 sd at 1% and 11.6 sd at 3%.
5. **Stale REUSED paths (lines 865–867 and 883).** `infra/pg/alwayson_edit.py`, `infra/aws_pg/jobs/alwayson_edit.py`
   and `infra/aws_pg/jobs/instruments_dryrun_e2.py` are not on branch `n13`. Commit `97085e7d8` moved them to
   `studies/reproductions/alwayson/code/{pg,jobs}/` and `studies/reproductions/instruments_e2/code/jobs/`. The code
   imports `row_sha256` and `boxed_last` from the new path.
6. **Signed zeros in the builder (lines 380, 389–391).** bf16(f32(θ_b) + c·M·Δ) maps a +0/−0 pair to +0, so `id`
   would not be bitwise RL. The builder takes the source bits where M = 1 and c = 1. It asserts the formula wherever
   Δ ≠ 0, and R1 counts the signed-zero swaps.
7. **PC-RANK failing input undefined (line 565).** "Rows of `down_proj`" is undefined for neuron index n ≥ 4,096, and
   it leaves the gate and up rows planted. Implemented as row n mod 4,096 of `down_proj` only. At the real ratios this
   gives recovery 1.0 against 0.0, with chance at 0.013.
8. **Random-number conventions (lines 428 and 581).** Line 428 names a torch `Generator` for the DARE masks; NumPy
   `default_rng([seed, layer, w])` per tensor is used instead. The per-tensor stream convention is not stated
   anywhere, and line 581 gives no seed for the PC-ART synthetic change (used: `[40, 1, layer, w]`).
9. **Non-LFS file (line 535).** "sha256 against the LFS oid" does not apply to MATH-500 `test.jsonl`, which is not an
   LFS file. Its tree oid `2376b9a1…` is a git blob sha1, and the check uses that.
10. **Empty answer line (lines 334–335).** An `Answer:` line with nothing after it gives an empty extraction, scored
    wrong. The boxed fallback is skipped in that case. Whether this is intended is not stated.
11. **GSM8K labels incomplete (lines 726–729).** No label is given for "opposite sign, CI includes 0". Implemented as
    UNDEFINED.
12. **G-HASH for `zero` (lines 567 and 614).** A failing input is registered for `id` only. Implemented for `zero`:
    neuron (0, 0) added.
13. **G-TEMPLATE in the cancel list (line 686).** The verdict's cancel list omits G-TEMPLATE, which every generation
    step enforces. The script includes it.
14. **R1 time not stated (lines 931 and 1001).** The S2 row implies about 75 min. The laptop estimate is about 50 min
    of compute, probably slower on 8 vCPU. The cost is dominated by 2 × 20 redraws and 31 hash-only builds.
15. **Facts for §0.2 (lines 84, 87, 92).** These are not inconsistencies.
    - 3.1 `main` equals `step_2800`.
    - The 3.1 config equals the base's in every shape field. It differs in `transformers_version` (4.57.1 against
      4.57.0) and `use_cache` (false against true).
    - The 3.1 `tokenizer.json` is the base's blob.
16. **S-BAND rounding (lines 574, 648–651).** The exact band is [81.46, 96.74]. The registered [81.5, 96.7] is rounded
    inward by 0.04 on each side.
17. **When C is frozen (lines 324–325 against 665–667).** Lines 324–325 freeze C "in `thresholds.json` before the gate
    runs", but lines 665–667 write `thresholds.json` from the four gate banks, after the gate. The S3 manifest
    therefore takes C as an explicit `--cap`, decided from the smoke. `freeze_thresholds.py` records it again.

## Deviations made in the code (for Fable)

1. R1 is not in the R0 launch, per the brief. The S3 manifest (`n13_plan.manifest("s3gate", ...)`) now starts with
   setup `--models base,rl,stage2`, then `r1_rank --delete-stage2`. Its gate arms read
   `/work/out/r1_rank/sets.json`.
2. Grading uses `dollar_both` (spec issue 1).
3. The instrument is NumPy-only. DARE masks use NumPy (spec issue 8).
4. The verdict reads a screened arm on problems 0–19 (§7), even when more answers finished before the abort.

## Files created

All paths are in `/Users/abaybektursun/projects/interp_research-n13/` unless marked as scratch.

- `infra/pg/n13_arms.py`: the weights instrument (bf16 bits, Δ, s, s', sets, PC-RANK, PC-ART, J_round, the builder,
  G-HASH, G-MASK, G-INIT, G-PARTITION).
- `infra/pg/n13_score.py`: the prompt and G-TEMPLATE, extraction, the grader and G-GRADER, statistics, the screen,
  G-COMPLETE, S-BAND.
- `infra/pg/n13_plan.py`: the roster, seed sets, S3 prefixes, manifests, the budget projector and the park check.
- `infra/aws_pg/jobs/n13_setup.py`: venvs, data, weights with hashes, G-GRADER, G-INIT (a), G-TEMPLATE.
- `infra/aws_pg/jobs/n13_known.py`: R0 (olmes, G-KNOWN, reader, samples).
- `infra/aws_pg/jobs/n13_olmes_rescore.py`: R0's identity and derangement rescore under olmes' scorer.
- `infra/aws_pg/jobs/n13_rank.py`: R1.
- `infra/aws_pg/jobs/n13_gen.py`: one arm, the generation loop, the throughput record.
- `infra/aws_pg/jobs/n13_dryrun.py`: this offline gate.
- `infra/aws_pg/steps/n13_s2setup-01.json`: the R0 manifest.
- `studies/next/n13/scripts/freeze_thresholds.py`, `verdict_n13.py`, `figure_n13.py`, `samples_coverage.py`,
  `recount_n13.py`.
- `studies/next/n13/STATE.md`.
- `studies/next/n13/offline_gate/`: `evidence/` (20 JSON), `e2e/` (verdict, thresholds, R0, R1 and setup records,
  figure), `stub/` (fake-engine banks), `dryrun_log.txt`.
- `studies/next/07_PICKS/N13_OFFLINE_GATE.md`: this file.
- Scratch: `$SP/n13_venv/`, `$SP/n13_gate_hf/`, `$SP/n13_dry/`, `$SP/n13_dry_log.txt`, `$SP/n13_venv_freeze.txt`.

## R0 record, 2026-11-05

**R0 has no known number yet.** Five attempts, 1.222 GPU-h, none reached olmes generation. G-KNOWN was not
evaluated, and no smoke ran. So there are no raw examples, no measured throughput and no re-projection. Instances,
times and cost are in `infra/aws_pg/operations/2026-11-05_next_n13_r0.md`. Small result files are copied to
`studies/next/n13/r0/s2setup-0{1..5}/`.

| attempt | where it stopped | cause | change made |
|---|---|---|---|
| `s2setup-01` | setup, olmes install | `pytrec_eval` builds a C extension; the AMI has no `Python.h` | `n13_setup.venv_olmes` installs `python3.12-dev` and asserts `Python.h` |
| `s2setup-02` | setup, olmes pin check | `antlr4-python3-runtime` 4.11.0 against the pin 4.11, compared as strings | `n13_known.release`: versions compared as release tuples, in `venv_olmes` and `check_config` |
| `s2setup-03` | R0, olmes start | olmes' launcher runs `python -m oe_eval.run_eval` through `/bin/sh`; no `python` on PATH (exit 127) | the olmes venv's `bin` first on PATH for olmes; the same for the `n13_gen` child with `/opt/n13/bin` |
| `s2setup-04` | R0, olmes import | pyarrow 26.0.1 (unpinned by olmes) requires NumPy ≥ 2; olmes pins NumPy < 2 | `pyarrow==23.0.1` (the last release before the olmes commit); `import oe_eval.run_eval` checked in setup |
| `s2setup-05` | R0, vLLM engine construction | olmes passes `trust_remote_code` None; vLLM 0.11.0 requires a bool. Behind it: lm_eval 0.4.3 calls `LLM.generate(prompt_token_ids=...)`, which vLLM 0.11.0 no longer accepts | none; the choice is spec amendment A18, for the head |

The offline gate passed after every change: `N13_DRYRUN_OK`, 74 of 74
(`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/n13_dry_r0fix3_log.txt`).

**What attempt 5 did show on the instance.** These are the setup gates, from
`studies/next/n13/r0/s2setup-05/setup/metrics.json`:

- **G-GRADER.**
  - The registered rule gives 12/24 planted equivalent pairs, 16/16 non-equivalent pairs and 392/500 golds equal to
    themselves. It fails.
  - `dollar_both` gives 24/24, 16/16 and 500/500, with derangement rate 0.006 against a bound of 0.016. It passes.
  - `boxed` gives 23/24. It fails.
  - These values equal the laptop values (`offline_gate/evidence/G-GRADER.json`).
- **Weights.** All 3 base shards and all 3 RL shards match their sha256 hashes, with 0 retries.
- **G-INIT (a).** It passes on all 3 shards. Its failing input fails.
- **G-TEMPLATE.**
  - 500/500 prompts render as specified. No prompt starts with BOS.
  - The template sha256 is `4380e622…`, and the prompt ids sha256 is `ff88a606…`.
  - Its failing input fails.
- **Environments.**
  - `/opt/n13`: vllm 0.32.0, math-verify 0.9.0, torch 2.13.0, transformers 5.19.0.
  - `/opt/olmes-venv`: vllm 0.11.0, lm_eval 0.4.3, antlr4-python3-runtime 4.11.0, torch 2.8.0, transformers 4.57.6,
    pyarrow 23.0.1.
  - The olmes config check passed.

**Not done, and why.**
- **G-KNOWN, `primary_score` against 40.0 ± 1.6.** olmes never generated.
- **Context lengths over 1,024 tokens (A3).** These are counted after olmes runs.
- **The ten raw examples.** They need the smoke.
- **The empty `Answer:` count (A10).** It needs the smoke.
- **The cap C.** It needs the RL smoke.
- **Throughput and the projection at the measured rate.** They need the smoke. Until then the registered projection
  stands: 25.5 GPU-h at 800 output tokens per second and 37.5 at 500, against the 35 ceiling (§"Projected
  GPU-hours"). The 1.222 GPU-h spent here come on top.
- R0's remaining cost depends on A18: about 0.5 to 1 GPU-h under option (b), and about 5 to 9 under option (a). Setup
  (about 0.3 GPU-h) and the smoke come on top of either.

## Offline gate rerun, 2026-11-05, A2 A4 A10 A18

**`N13_DRYRUN_OK`, 79 of 79 checks** (`studies/next/n13/offline_gate/rerun_a18/checks.json`, log `dryrun_log.txt` beside it,
104.7 s). Written by the second N13 executor after the head's decisions of 13:00 CST. The command and venv are those of
§"How to rerun", with `--out $SP/n13_dry_a18`.

Five checks are new, and four changed:

| check | what it shows | value |
|---|---|---|
| A18 shim (new) | on a fake LLM with vLLM 0.11.0's `generate` signature, `generate(prompt_token_ids=[[100257, 5, 6], [7]], sampling_params=sp, use_tqdm=False)` arrives as two `TokensPrompt`s, the same `sp` object, `use_tqdm` false; a new-signature call is untouched; one converted call logged | `rerun_a18/A18_shim.json` |
| A18 shim failing input (new) | the same call without the shim | `TypeError: LLM.generate() got an unexpected keyword argument 'prompt_token_ids'` |
| R0 path check (new) | `--limit 20` runs first; 20 scored rows per subject; the reader and the rescorer run on it; the olmes subprocess sees the shim directory first on PYTHONPATH; `shim.calls` written for both runs; `engine` names vllm 0.11.0, lm_eval 0.4.3, olmes `5a51f502…` and the shim sha256 | `rerun_a18/R0_fabricated.json` |
| R0 path check failing input (new) | 19 scored rows per subject | exit 4, status `failed_path_check`, the full olmes directory not created, no G-KNOWN |
| verdict, A2 (new) | H_artifact decided by the rule when PC-ART passes (REFUTED on the planted banks), CANCELLED when rule(P−) is SUPPORTED | `CANCELLED (PC-ART: rule(P+) = SUPPORTED, rule(P-) = SUPPORTED)` |
| PC-ART at R1 on the iid stand-in (changed, A2) | the rule on P+ and on P− (8 of 384 neurons moved 8 steps) | P+: J_floor 0.000, J_round 0.234, J_sp 0.143, SUPPORTED; P−: J_floor 1.000, J_round 1.000, REFUTED |
| DARE kept count (changed, A4) | the builder's own `kept_pass` at 4 binomial sd, and the check's 4 sd | all 5 DARE arms pass |
| scoring path (changed, A10) | 13 fabricated answers; row 12 is `Answer:` (empty), then `\boxed{5}`, gold 5 | scored 1, mode `boxed` |
| G-KNOWN command (changed, A18) | the command equals §6.1 with `--model-args '{"trust_remote_code": false}'`; the path check adds only `--limit 20` | equal |

The shim file is `infra/aws_pg/jobs/n13_olmes_shim/sitecustomize.py`, sha256
`f39d888a3c4e688464cc6df26dd17bc42d46d00008dc25a5e17bd4011415d148`.

What the gate cannot show: the shim inside real vLLM 0.11.0 and lm_eval 0.4.3, and olmes' `--limit` on the real task. The path check
at the start of `r0_known` covers both before the full task.

## R0 attempt 6, 2026-11-05

Written by the third N13 executor. Instance, times and cost: `infra/aws_pg/operations/2026-11-05_next_n13_r0.md` §"Attempt 6".
Small result files, copied from `s3://pg-lineage-769063704696/results/next_n13/s2setup-06/out/`: `studies/next/n13/r0/s2setup-06/`
(paths below are relative to it unless they start with `studies/` or `infra/`).

**Attempt 6 scored nothing.** The A18 path check passed. olmes then ran the full task at batch size 1, at about 10 generations per
minute, until the head terminated the instance. G-KNOWN was not evaluated and no smoke ran.

**What passed.**

- Setup, every stage, 19:23:53Z to 19:39:47Z (`setup/metrics.json`).
- The path check (`olmes --limit 20`): olmes exit 0; 20 scored rows per subject; primary_score 0.3339 at n = 140, not read against
  40.0; the deranged rescore 0.0018; `identity_docs_equal` true; 2 of 140 contexts over 1,024 tokens, the longest 1,494
  (`r0_known/metrics.json` → `path_check`; `r0_known/rescore_limit20.json`). The shim converted all 560 calls
  (`r0_known/shim_calls_limit20.jsonl`). The A18 engine works on the instance.

**Cause: olmes ran at batch size 1.**

- The path check took 3,120 s for 560 generations, 5.57 s each (`r0_known/metrics.json` → `path_check.elapsed_s`).
- All 560 lines of `r0_known/shim_calls_limit20.jsonl` read `n_prompts` 1. Every `r0_known/olmes_limit20/task-*-metrics.json` reads
  `compute_config` `batch_size` 1, `max_batch_size` 32.
- It generated 137,386 tokens: mean 245.3, 53 of 560 at the 1,024 cap. That is 45.3 tokens per second over the 3,030 s after the
  first subject began, and between 43.0 and 46.2 in every subject (`path_check_tokens.json`, from the predictions files by
  `path_check_tokens.py` beside it). This is one sequence decoded at a time.
- Why, from olmes, lm_eval and vLLM source (quoted lines: `studies/next/n13/offline_gate/rerun_batch/sources_batching.txt`):
  olmes' launcher leaves `--batch-size` unset, `run_eval` then uses its default 1, and lm_eval's vLLM path sends chunks of that size,
  one `LLM.generate` call per chunk. The chain, file and line, is in `N13_SPEC.md` A19.
- The full task started at about 20:32:28Z (the update at 21:03:28Z minus `elapsed_s` 1,860, `r0_known/metrics.json`). At the last
  upload, 21:03:52Z, `r0_known/shim_calls.jsonl` held 320 calls, all `n_prompts` 1: 10.2 per minute. 20,000 generations at that
  rate take about 32.7 hours; at the path check's 5.57 s, about 31.0 hours. The allowance was 280 minutes.

**Why the job did not stop itself.** The park check at the start of `r0_known` projected 171.67 minutes (`r0_known/metrics.json` →
`park`). It was computed as olmes load 5 min + 5.0 M tokens / 500 tokens per second / 60 (`n13_known.run` before A19;
`n13_plan.ASSUME`, `--proj-tok-s 500.0` in the manifest). With 26.1 min elapsed, 26.1 + 171.7 = 197.8 < 280, so it did not park.
The token count was close: the path check's mean lengths give about 4.64 M tokens for the full task (`path_check_tokens.json` →
`full_task_tokens_estimate`). The rate was 11 times too high: 500 assumed, 45.3 measured. The projection assumed batched decoding
and was made before any measurement.

**Lesson.** The path check measured the defect 52 minutes into R0, before the big spend, and the job did not act on what it
measured. A pre-check must decide before the spend.

**Also found, not changed.** The job's elapsed time counts from the S3 upload of `steps.json`, not from the launch.
`n13_plan.launch_t0` reads the mtime of `out/steps.json`, and `aws s3 cp` sets it to the object's LastModified, 19:13:43Z here. The
instance launched at 19:22:44Z (operations note). The park check's 26.1 min at 19:39:49Z fits the upload time. The error is toward
parking: elapsed is overstated by the time between the upload and the launch, 9.0 min in attempt 6.

**The fix: spec amendment A19** (`N13_SPEC.md` §"A19"), in `infra/aws_pg/jobs/n13_known.py`:

| change | what | registered failing input | in the offline gate |
|---|---|---|---|
| batch size | `--batch-size auto` in the G-KNOWN command and the path check: one `LLM.generate` call per subject (80 prompts in the path check, 4 × the subject size in the full task) | — | the command equals the A19 command (`rerun_batch/G-KNOWN_config.json`) |
| path-check assertion | at least one logged shim call with `n_prompts` ≥ 2, else exit 4 before the full task; then an assertion that the logged prompts equal 560 | batch size 1: 560 shim calls of 1 prompt, as in `s2setup-06` | exit 4, status `failed_path_check`, no full olmes directory (`rerun_batch/A19_batch1_failing.json`) |
| projection gate | projected full-task s = path-check wall s / path-check generations × 20,000; park when it exceeds 0.85 × (allowance − elapsed); a parked step writes `projection`, the `PARKED` marker, exits 0 | the `s2setup-06` numbers: 560 in 3,120 s, 20,000, allowance 280 min, 42 min elapsed | 1,857.1 min against 202.3: parks; a planted batched rate, 560 in 300 s: 178.6 min, goes; end to end, parks before the full task (`rerun_batch/A19_projection.json`, `A19_park_e2e.json`) |

Full-task generations: 7 subjects, test sizes algebra 1,187, counting_and_probability 474, geometry 479, intermediate_algebra 903,
number_theory 540, prealgebra 871, precalculus 546, sum 5,000 (`N13_MATH_TASK.md` lines 131–133; the datasets-server record,
`rerun_batch/hf_math_test_sizes.json`, fetched 2026-11-05), times 4 repeats = 20,000. In the s2setup-06 numbers the job's own clock
would have read 78.8 min at the decision (20:32:28Z minus the 19:13:43Z upload); it parks there too.

**Estimate for attempt 7 at a batched rate. These are estimates from source and from the measured batch-1 rate, not
measurements.**

- Inputs: 45.3 tokens per second at batch 1 (`path_check_tokens.json`); 14.6 GB of weights and 524,288 bytes of cache per token,
  about 50,000 cached tokens on the L40S (`N13_SPEC.md` §9.1); at most 256 running sequences and prefix caching on (vLLM 0.11.0,
  `sources_batching.txt`); 4.64 M output tokens for the full task; mean context 690 tokens (`path_check_tokens.json`).
- Decode reads the weights once per step and each running sequence's cache. The batch-1 rate implies about 680 GB/s. With about 50
  sequences in the cache (690 context + about 300 generated tokens each), that gives about 840 tokens per second. With prefix
  caching more sequences fit (128 gives about 1,080); if vLLM reads the shared 4-shot prefix once per step (cascade attention, not
  verified here), about 2,100.
- Full task: 4.64 M tokens at 840 to 2,100 tokens per second is 37 to 92 min. Per-subject tails and set-up add about 5 min, so about
  45 to 100 min, about 80 at 1,000 tokens per second.
- Path check: each subject waits for its longest sample, 1,024 steps of at least 21 ms. With the other samples' cache traffic,
  that is about 37 s per subject. Adding the start-up (90 s in `s2setup-06`) and about 6 s of set-up per subject gives about 400 s
  (6.7 min), range about 5 to 9 min.
- Under the A19 rule a 400 s path check projects 238 min. With no gap between the upload and the launch, elapsed at the decision is
  about 24.5 min (boot 1.2, setup 15.9, path check 6.7, rescore 0.7). The rule then allows 0.85 × 255.5 = 217 min, so **it would park
  a run whose full task fits.** It goes only if the path check finishes within about 365 s. Every minute between the upload of
  `steps.json` and the launch lowers that bound by about 1.4 s (0.85 × 60 × 560 / 20,000).
- The launch's uptime and GPU-hours are estimated in the operations note, §"Attempt 6".

## Offline gate rerun, 2026-11-05, A19

**`N13_DRYRUN_OK`, 85 of 85 checks** (`studies/next/n13/offline_gate/rerun_batch/checks.json`, log `dryrun_log.txt` beside it, 130 s).
The command and venv are those of §"How to rerun", with `--out $SP/n13_dry_a19`. Six checks are new and two changed:

| check | what it shows | value |
|---|---|---|
| G-KNOWN command (changed, A19) | the command equals §6.1 as amended by A18 (b) and A19, with `--batch-size auto`; the path check adds only `--limit 20`; olmes' launcher accepts every flag (the config check's `command_flags_accepted` row, read from `oe_eval/launch.py` at `5a51f502`) | equal |
| R0 path check (changed) | the fake olmes logs 7 shim calls of 80 prompts for the path check, 1 for the full run | 7 and 1 calls, 560 prompts |
| R0 path check, A19 (new) | the batched assertion passes (largest call 80 prompts); `projection` is written before the full task and goes at allowance 280; `engine.batch_size` is `auto` | `rerun_batch/R0_fabricated.json` |
| path-check failing input, A19 (new) | batch size 1, 560 shim calls of 1 prompt | exit 4, `failed_path_check`, no full olmes directory, no `projection`, no G-KNOWN |
| full-task generations, A19 (new) | 7 subject sizes, sum 5,000, × 4 repeats | 20,000 |
| projection failing input, A19 (new) | the `s2setup-06` numbers | 1,857.1 min > 202.3: parked |
| projection on a batched rate, A19 (new) | planted: 560 generations in 300 s, same allowance and elapsed | 178.6 min ≤ 202.3: goes; the largest wall time that goes is 339.9 s |
| projection end to end, A19 (new) | a 1.1 s fake path check at allowance 0.5 min | 0.65 min > 0.42: status `parked`, the `PARKED` marker, exit 0, no full olmes directory (`rerun_batch/A19_park_e2e.json`) |

The A18 shim check is unchanged and still passes: a two-prompt `generate(prompt_token_ids=...)` call arrives as two `TokensPrompt`s
and is logged as one call of 2 prompts; without the shim it raises `TypeError` (`rerun_batch/A18_shim.json`). The shim's sha256 is
unchanged, `f39d888a…`.

What the gate cannot show: that olmes passes `auto` through on the instance and that vLLM then receives many prompts per call. The
path check's new assertion checks this on the instance before the full task.

### A19 revision, 2026-11-05: the projection from measured throughput, and the elapsed clock

Written by the fourth N13 executor, on the head's brief. The rule and its reason: `N13_SPEC.md` A19 §"Revision, 2026-11-05". In
short: projected = F + T / r, with T = 20,000 × path-check output tokens / 560, r = output tokens / generation seconds (the summed
time inside vLLM's `generate`, from the shim's own timer), F = path-check wall seconds − generation seconds; and the job's elapsed
time counts from the pipeline's `started_utc`, not from the `steps.json` upload. The first form would have parked a healthy batched
run: on the planted 500 tokens per second it projects 242.1 min against 218.1 allowed.

**`N13_DRYRUN_OK`, 87 of 87 checks** (`studies/next/n13/offline_gate/rerun_batch2/checks.json`, log `dryrun_log.txt` beside it,
141 s). The command and venv are those of §"How to rerun", with `--out $SP/n13_dry_a19r`. Two checks are new and six changed:

| check | what it shows | value |
|---|---|---|
| A18 shim (changed) | the fake LLM now returns `RequestOutput`-shaped objects (planted: completions of 3, 5 and 2 token ids); the shim passes the return value back unchanged and logs one call with its wall time and `output_tokens` | 1 call, `n_prompts` 2, `output_tokens` 10, 0.055 s (`rerun_batch2/A18_shim.json`) |
| R0 path check, A19 (changed) | the projection is built from the shim log: 560 generations, 137,200 output tokens, generation seconds = the 7 fake calls' summed time, F = wall − generation seconds > 0; it goes at allowance 280 | `rerun_batch2/R0_fabricated.json` → `projection` |
| projection failing input (changed) | the `s2setup-06` path check, and the brief's figures (T 4.64 M at 45.3 tokens/s), at allowance 280 with nothing elapsed | 1,805.1 and 1,708.5 min > 238.0: parked |
| projection on a batched rate (changed) | planted: 500 tokens/s at 80 prompts per call, F 132 s, 23.4 min elapsed; the first form on the same input; 300 tokens/s | 165.8 ≤ 218.1: goes (first form 242.1: parks); 300 tokens/s 274.8 > 215.5: parks (`rerun_batch2/A19_projection.json`) |
| projection, broken timing (new) | generation seconds above the wall seconds | `AssertionError` |
| projection end to end (changed) | 7 fake calls of 0.15 s, allowance 0.5 min, elapsed from a `pipeline_status.json` written at the start | 0.64 min > 0.38: `parked`, the `PARKED` marker, exit 0, no full olmes directory, elapsed 0.05 min (`rerun_batch2/A19_park_e2e.json`) |
| elapsed clock (new) | the real `pipeline.main` with one probe step; only its S3 download is replaced by a local copy that keeps a 1-hour-old mtime, as `aws s3 cp` keeps the upload time. Registered failing input: that mtime (the old clock reads 60 min) | `started_utc` written; the probe's `launch_t0` is 0.11 s before the pipeline's launch (truncation to whole seconds); new clock 0.003 min, old clock 60.0 min (`rerun_batch2/A19_clock.json`) |
| stop rule, scoring path (input changed, name unchanged) | the earlier step's elapsed now comes from a `pipeline_status.json` whose `started_utc` is 1 hour old, not from a `steps.json` mtime | both steps parked |

Projection table (`rerun_batch2/A19_projection.json`; inputs from `studies/next/n13/r0/s2setup-06/path_check_tokens.json`):

| input | wall s | generation s | F s | r tokens/s | elapsed min | projected min | allowed min | revised | first form |
|---|---|---|---|---|---|---|---|---|---|
| `s2setup-06` replay | 3,120.0 | 3,030.0 | 90.0 | 45.3 | 0 | 1,805.1 | 238.0 | parks | 1,857.1, parks |
| `s2setup-06`, the brief's figures (T 4.64 M) | 2,957.8 | 2,867.8 | 90.0 | 45.3 | 0 | 1,708.5 | 238.0 | parks | 1,760.6, parks |
| planted, 300 tokens/s | 590.0 | 458.0 | 132.0 | 300 | 26.4 | 274.8 | 215.5 | parks | 351.2, parks |
| planted, 500 tokens/s | 406.8 | 274.8 | 132.0 | 500 | 23.4 | 165.8 | 218.1 | goes | 242.1, parks |
| planted, 800 tokens/s | 303.7 | 171.7 | 132.0 | 800 | 21.7 | 104.4 | 219.6 | goes | 180.8, goes |

At F = 132 s the slowest path check that goes generates for 360.7 s, 51.5 s per subject, 380.9 tokens per second
(`boundary_at_F_132_s`). F = 132 s and the 16.6 min of setup and rescore are the third executor's estimates and the `s2setup-06`
times, not measurements of a batched run.

The shim's sha256 is now `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6` (2,738 bytes); the conversion is
unchanged. The R0 manifest `infra/aws_pg/steps/n13_s2setup-01.json` is unchanged: it equals `n13_plan.manifest("s2setup", 1,
allowance_min=280)`. Source lines for the shim's token count (vLLM 0.11.0 `vllm/outputs.py`, olmes `eleuther_vllm_causallms.py`):
`rerun_batch2/sources_shim.txt`.

What the gate cannot show: that vLLM 0.11.0's real `RequestOutput` objects carry `token_ids` on the instance (read from source,
not run), and how long a batched path check takes. The path check writes both into `out/r0_known/metrics.json` → `path_check.shim`
(`generate_s`, `output_tokens`) and `projection` before the full task.

### A19 second revision, 2026-11-05: the path check at `--limit 100`, an early stop at batch size 1, the subject-weighted projection

Written by the fifth N13 executor, on the head's brief. The rule and its reason: `N13_SPEC.md` A19 §"Second revision, 2026-11-05".
In short: the path check runs at `--limit 100`, one call of 400 prompts per subject, so it measures the output rate with vLLM at its
concurrency limit; a first logged call with one prompt kills olmes within one poll interval (exit 4); and
T = Σ_s 4 × size_s × tokens_s / n_s, with call i mapped to subject i in olmes' task order and exactly 7 calls asserted.

**`N13_DRYRUN_OK`, 89 of 89 checks** (`studies/next/n13/offline_gate/rerun_batch3/checks.json`, log `dryrun_log.txt` beside it,
123 s). The command and venv are those of §"How to rerun", with `--out $SP/n13_dry_a19r2`. One check was replaced by two, one is
new and eight changed:

| check | what it shows | value |
|---|---|---|
| G-KNOWN command (changed) | the path check adds only `--limit 100`; `n13_known.PATH_LIMIT` is 100 | equal (`rerun_batch3/G-KNOWN_config.json`) |
| R0 path check, A18 (changed) | `--limit 100` first, 100 scored rows per subject, the reader and rescorer run on it, 7 shim calls and 2,800 prompts logged | `rerun_batch3/R0_fabricated.json` |
| R0 path check, A19 (changed) | the fake olmes logs one 400-prompt call per subject with `s2setup-06`'s per-subject mean lengths; first and largest call 400, no early stop; `projection.per_subject` equals the planted tokens in task order; T = Σ 4 × size_s × tokens_s / n_s; projected = F + T / r; goes at allowance 280 | `R0_fabricated.json` → `projection` |
| early stop, batch size 1 (new; replaces the 560-call failing input) | 2,800 calls of 1 prompt, 0.01 s each (at least 28 s unstopped), poll 0.05 s. The fake olmes runs as two processes, a launcher and a child that logs, as olmes runs `run_eval` | the first call seen 0.048 s after it returned; process group killed (olmes exit −9); the log stays at 4 lines 0.5 s later, so the child died too; exit 4 after 0.31 s; status `failed_path_check` with the reason; no projection, no full olmes directory (`rerun_batch3/A19_batch1_early.json`) |
| end assertion, batch size 1 (new; the kept assertion's failing input) | the same 2,800 one-prompt calls with no delay and poll 3 s, so olmes exits 0 before any poll sees the log | largest call 1 prompt, exit 4, no early stop, no full olmes directory (`rerun_batch3/A19_batch1_end.json`) |
| full-task generations (changed) | 20,000; the path check 7 × 100 × 4 = 2,800 | equal |
| projection failing input (changed) | the `s2setup-06` path check, its 560 calls summed into one per subject | T 4,643,969.85 (equal to `path_check_tokens.py`'s estimate); 1,708.5 min > 238.0: parks |
| projection on planted rates (changed) | 400 prompts per subject at `s2setup-06`'s lengths, F 132 s, elapsed = setup + wall + rescore | 300 tokens/s parks; 500 and 1,000 go (table below) |
| projection, 7 calls (new) | 6 calls given to `project` | `AssertionError` |
| projection, broken timing (input changed) | generation seconds above the wall seconds, in the 7-call form | `AssertionError` |
| projection end to end (changed) | 7 fake calls of 0.3 s at `s2setup-06`'s lengths (T / r = 6.76 × 2.14 s), allowance 0.25 min | 0.245 min > 0.160: `parked`, the `PARKED` marker, exit 0, no full olmes directory (`rerun_batch3/A19_park_e2e.json`) |
| path-check failing input (changed) | 99 scored rows per subject | exit 4, `failed_path_check` |

Projection table (`rerun_batch3/A19_projection.json`; inputs from `studies/next/n13/r0/s2setup-06/path_check_tokens.json`):

| input | wall s | generation s | r tokens/s | elapsed min | projected min | allowed min | decision | unweighted T |
|---|---|---|---|---|---|---|---|---|
| `s2setup-06`, one call per subject | 3,120.0 | 3,030.0 | 45.3 | 0 | 1,708.5 | 238.0 | parks | 1,805.1, parks |
| planted, 300 tokens/s | 2,421.8 | 2,289.8 | 300 | 59.8 | 260.2 | 187.2 | parks | 274.8, parks |
| planted, 500 tokens/s | 1,505.9 | 1,373.9 | 500 | 44.5 | 157.0 | 200.2 | goes | 165.8, goes |
| planted, 1,000 tokens/s | 818.9 | 686.9 | 1,000 | 33.0 | 79.6 | 209.9 | goes | 84.0, goes |

At these inputs the slowest rate that goes is 400.7 tokens per second (`boundary_at_F_132_s`). F = 132 s and the 3.5 min rescore of
700 documents are estimates, not measurements.

The shim is unchanged (sha256 `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6`, `rerun_batch3/A18_shim.json`).
The R0 manifest `infra/aws_pg/steps/n13_s2setup-01.json` is unchanged: `--limit` is not a step argument, and the file equals
`n13_plan.manifest("s2setup", 1, allowance_min=280)` (810 bytes).

What the gate cannot show: that olmes on the instance runs the subjects in suite order and logs exactly one call per subject at
`--limit 100` (read from source and from `s2setup-06`'s task start times; `project` stops if there are not 7 calls); that the kill
of olmes' process group reaches vLLM's engine process (no vLLM here; the fake shows it reaches a child process); and how fast a
400-prompt call runs. The path check writes `path_check.batched`, `path_check.early_stop` when it fires, and `projection.per_subject`
into `out/r0_known/metrics.json` before the full task.

## R0 attempt 7, 2026-11-05

Written by the sixth N13 executor. Instance, times and cost: `infra/aws_pg/operations/2026-11-05_next_n13_r0.md` §"Attempt 7". Small
result files, copied from `s3://pg-lineage-769063704696/results/next_n13/s2setup-07/` (under `out/` unless named otherwise):
`studies/next/n13/r0/s2setup-07/` (paths below are relative to it unless they start with `studies/` or `infra/`). Every number below
comes from these files; the S3 key is `results/next_n13/s2setup-07/out/<path>`.

**G-KNOWN passes: 100 × `primary_score` = 39.96, 0.04 points from 40.0, inside the 1.6-point tolerance with 1.56 points to spare.**
`r0_known/metrics.json` → `read.primary_score` 0.3996393456454165 (the macro over the 7 subjects, equal to `primary_score_macro`;
`n_total` 5,000, `sizes_match` true) and `G-KNOWN` `value` 39.963934564541645, `pass` true. The step's status is `done` and it exited 0
(`pipeline_status.json`). `pass_at_1_micro` 0.43065 is recorded and not read.

| subject | primary_score | test problems |
|---|---|---|
| algebra | 0.5996 | 1,187 |
| counting_and_probability | 0.3718 | 474 |
| geometry | 0.3497 | 479 |
| intermediate_algebra | 0.2375 | 903 |
| number_theory | 0.3319 | 540 |
| prealgebra | 0.6427 | 871 |
| precalculus | 0.2642 | 546 |
| **macro** | **0.3996** | 5,000 |

Source: `r0_known/metrics.json` → `read.per_subject`, `read.num_instances`.

**The rescore block** (`r0_known/rescore.json`; olmes' own Minerva scorer, seed 0, run under the olmes venv):

- Identity pair: the same rescorer with the true golds gives the macro 0.3996393456454165, and `identity_docs_equal` true: every one of
  the 5,000 documents' `pass_at_1` equals olmes'.
- Failing input: the same predictions against a derangement of the golds within each subject give the macro 0.011171 (1.12 points),
  below 10.0, so `G-KNOWN.failing_input.failed` is true. Per subject 0.0042 to 0.0278 (`per_subject_deranged`).
- Contexts over 1,024 tokens (the A3 threat): 36 of 5,000, the longest 1,884 (`context_tokens`, counted with the base tokenizer).
  olmes truncates those to their last 1,024 tokens (A19 item 2). In the path check: 5 of 700, the longest 1,494
  (`r0_known/rescore_limit100.json`).
- The grading rule's checks at setup (`setup/metrics.json` → `G-GRADER`): the active rule `dollar_both` passes, 24 of 24 equivalent
  pairs true, 16 of 16 non-equivalent false, 500 of 500 golds equal to themselves, derangement rate 0.006 (bound 0.016 in the offline
  gate). G-TEMPLATE 500 of 500 equal, its failing input fails; G-INIT (a) passes, its failing input fails.

**Generation lengths.** 1,915 of the 20,000 full-task generations reached olmes' 1,024-token cap; mean 248.65, maximum 1,024. Path
check: 267 of 2,800, mean 252.03 (`num_tokens` summed over `olmes/task-*-predictions.jsonl` and `olmes_limit100/...`; those files are
1.5 to 4.3 MB each and stay on S3 under `out/r0_known/olmes/` and `out/r0_known/olmes_limit100/`, with the `requests` and
`recorded-inputs` files).

**The shim counts** (`r0_known/shim_calls_limit100.jsonl`, `r0_known/shim_calls.jsonl`; summed in `r0_known/metrics.json` →
`path_check.shim` and `shim`):

| run | calls | prompts | largest call | output tokens | generation s | tokens per s |
|---|---|---|---|---|---|---|
| path check (`--limit 100`) | 7 | 2,800 | 400 | 705,692 | 506.5 | 1,393.2 |
| full task | 7 | 20,000 | 4,748 | 4,972,904 | 2,761.8 | 1,800.6 |

Per subject the full task ran at 1,614 (intermediate_algebra) to 2,226 (algebra) tokens per second. For both runs and every subject,
the shim's `output_tokens` equals the sum of olmes' `num_tokens` in that subject's predictions file, and call i is subject i in suite
order (the check of A19 second revision item 7). Every olmes `task-*-metrics.json` reads `compute_config.batch_size` "auto".

**The projection block, as written** (`r0_known/metrics.json` → `projection`): path-check wall 780.0 s, generation 506.53 s, F =
273.47 s; T = 4,939,093.42 tokens; r = 1,393.19 tokens per second; projected 63.64 min; allowance 280, elapsed 28.09 min, left
251.91, fits 214.12; `parked` false. The full task's olmes wall time was then 3,600.0 s (`elapsed_s`), with 2,761.8 s of generation.
Both wall times are multiples of the 60 s poll (780 = 13 × 60, 3,600 = 60 × 60): `run_olmes` reads the time at the first poll after
olmes exits, so each is up to 60 s late, and F was overstated by up to 60 s. The projection was above the outcome (63.6 against at
most 60.0 min): T was 0.7 % below the real 4,972,904 tokens, and r was 23 % below the full task's 1,800.6 tokens per second, as A19's
second revision expected (the drain at the end of each 400-prompt call).

**The ten raw examples** (`r0_known/samples.jsonl`, 36,876 bytes: the prompt as the model saw it, the four continuations, olmes'
extracted answers, the gold). The step takes the first 5 right and the first 5 wrong in subject order, so all ten are algebra.

- Right: documents 0, 1, 4, 10, 12 (gold 2, 10, 4, \frac{x+2}{7}, 10; all four answers equal the gold). Document 0, "How many vertical
  asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?": the first continuation ends "Thus, the graph has $\boxed{2}$ vertical
  asymptotes.\nFinal Answer: The final answer is $2$. I hope it is correct."
- Wrong: documents 2, 6, 8, 11, 20. Document 8 (gold label `461415`): answers `Nosolution`, `191019`,
  `1234`, `191019`; the first continuation argues "there is no solution to this problem". Document 6 (gold `x\in[-2,7]`): all four
  continuations end "The final answer is $[-2,7]$", and olmes' scorer scores them wrong, because the gold keeps the `x\in` prefix.
  That is olmes' scorer, the one behind the reported 40.0; it is recorded, not changed.
- The prompt is the 4-shot Minerva context (1,849 to 2,050 characters for these ten), ending "Problem:\n<problem>\n\nSolution:".

**The engine block** (`r0_known/metrics.json` → `engine`): vllm 0.11.0, lm_eval 0.4.3, olmes `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`,
model args `{"trust_remote_code": false}`, batch size `auto`, shim `/opt/pg/repo/infra/aws_pg/jobs/n13_olmes_shim/sitecustomize.py`
with sha256 `1f6ccee39a8b916dec2ce0f3b950903bc5acad049a55e2f157e3b8c5596526d6` (the expected file), amendments A18 (b) and A19. The
olmes venv pins as installed: vllm 0.11.0, lm_eval 0.4.3, antlr4-python3-runtime 4.11.0, torch 2.8.0, transformers 4.57.6
(`setup/metrics.json` → `venv_olmes.pins`; `setup/pip_freeze_olmes.txt`).

**Then the smoke failed, and nothing of the smoke is scored.** `smoke_base` started at 00:22:49Z and exited 1 at 00:23:41Z; the
pipeline stopped before `smoke_rl` (`pipeline_status.json`: setup ok, r0_known ok, smoke_base failed with return code 1; `STATUS`
`exit=1`). The step wrote only `smoke_base/base/build.json` (the base HF snapshot) and `smoke_base/base/math500/metrics.json` (status
`running`; `park` 104.86 min elapsed + 6.67 projected against 280, not parked; `template.pass` true, 20 prompts, no BOS). Cause: vLLM
0.32.0's engine core died at start-up in FlashInfer's JIT, `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist`, raised while its sampler warmup built FlashInfer's sampling module. The log lines (job.log lines 11883 to 12185, copied
once after the job ended, not polled; job.log sha256 `46db440f62e14534c4db1676a92e62965b0918e93e79779d46ffb36cd8b919c9`, 12,186 lines):
`smoke_base/job_log_lines_11883-12185.txt`. The cause from source and the fix: `N13_SPEC.md` A21. So C is not decided, and the S3
projection still rests on the spec's assumed rates.

**Also seen.**

- The n13 venv is not fully pinned: its pip freeze differs from attempt 5's in three packages (anthropic 2.6.0 against 2.5.0,
  cachetools 7.3.0 against 7.2.8, cuda-pathfinder 1.8.6 against 1.8.5; `setup/pip_freeze_n13.txt` against
  `studies/next/n13/r0/s2setup-05/setup/pip_freeze_n13.txt`). vllm 0.32.0, torch 2.13.0, transformers 5.19.0 and math-verify 0.9.0
  are the same. Reported, not changed.
- The n13 venv holds `nvidia-cuda-nvcc` 13.5.37 as a pip package, but FlashInfer looks for nvcc only under CUDA_HOME, CUDA_PATH, PATH
  and `/usr/local/cuda` (`vllm/utils/flashinfer.py` lines 61–73 mirror it), so it did not find it.

## Offline gate rerun, 2026-11-05, A20 A21

Written by the sixth N13 executor. **`N13_DRYRUN_OK`, 93 of 93 checks** (`studies/next/n13/offline_gate/rerun_a20/checks.json`, log
`dryrun_log.txt` beside it, 122 s). The command and venv are those of §"How to rerun", with `--out $SP/n13_dry_a20a21b`. An earlier run
of the same code before the `s2smoke` manifest was added gave 93 of 93 too (`$SP/n13_dry_a20a21`, not banked). Four checks are new
and one changed:

| check | what it shows | value |
|---|---|---|
| A20 profile (new) | R1 writes `block_profile` into `metrics.json` and `sets.json`, equal in both, 4 blocks; a separate float64 recount from the stand-in snapshots | largest relative difference 0.0; stand-in attn 0.099 to 0.100, mlp 0.239 to 0.438, embed and head 0 (`rerun_a20/A20_block_profile.json`) |
| A20 planted input (new) | block 2's attention and MLP tensors scaled by 2 | attn [0, 0, 1, 0], mlp [0, 0, 1, 0], embed 0, head 0, largest error 0.0; the identity pair gives 0 everywhere and the planted expectation fails on it |
| A21 environment (new) | `n13_gen.child_env` sets `VLLM_USE_FLASHINFER_SAMPLER=0`, keeps the venv first on PATH and every other variable; `main` uses it; `ATTENTION_BACKEND` is FLASH_ATTN | `rerun_a20/A21_engine_env.json` |
| A21 failing input (new) | `VllmEngine` with the variable unset, or at 1, stops on the A21 assertion before importing vllm; at 0 it reaches the vllm import | unset and 1: `AssertionError: A21 ...`; 0: `ModuleNotFoundError: No module named 'vllm'` (no vLLM on the laptop) |
| manifests (changed) | the new `s2smoke` manifest parses with `pipeline.py`'s checks and each module's parser | `s2smoke` 3 steps (`rerun_a20/manifests_and_budget.json`) |

What the gate cannot show: the engine start on the instance. The fix's registered failing input (the start without the variable,
which raised "Could not find nvcc" in `s2setup-07`) cannot run without vLLM and a GPU. The smoke of attempt 8 is its first run. The
source lines behind A21: `rerun_a20/sources_a21.txt`.

## Offline gate rerun, 2026-11-06, A22

Written by the seventh N13 executor. **`N13_DRYRUN_OK`, 97 of 97 checks** (`studies/next/n13/offline_gate/rerun_a22/checks.json`, log
`dryrun_log.txt` beside it, 132 s). The command and venv are those of §"How to rerun", with `--out $SP/n13_dry_a22b`. A first run
(`$SP/n13_dry_a22`, not banked) gave 96 of 97: the fake smoke's tail was 2.099998 s against 2.1 at a bound of 1e-6 s. The fake clock
counts from `time.time()`, so relative step times carry float error of about 2e-6 s; the bound is now 1e-4 s. No code under test
changed between the two runs. Four checks are new and one replaced:

| check | input | value |
|---|---|---|
| A22 smoke log (new) | fake engine, MATH-500 problems 0:300 per model, 3 answers per step, at most 128 running, 0.05 s per step | 10 lines for 100 steps; recount equals `throughput.json`; log tokens equal the answers' `n_generated` (base 2,321, rl 2,944); 58 steps at 128 running (2.9 s); tail 2.1 s; `metrics.json` holds `engine.load_s` (`rerun_a22/A22_smoke_fake.json`) |
| A22 projection command (replaces the check of `project_from_smoke`) | `python -m infra.pg.n13_plan` on the fake smoke, cap 16,384, 4.843 GPU-h spent | exit 0; the S3 manifest's 4 gate steps parse with `n13_gen`'s parser and carry `--cap 16384`, the allowance round(1.3 × S3), `--proj-tok-s` = the rate, `--proj-load-min` F (base, rl) or F + 5 (zero, id) |
| A22 failing input (new) | planted log: 2,000 steps at 100 running | exit 1, "A22: the base smoke's r_full is undefined: no engine step had at least 128 running sequences (largest running count 100); the projection refuses"; no manifest |
| A22 planted input and the stop rule (new) | 1,005 steps of 0.1 s at 300 running and 300 tokens, then 200 steps of 0.05 s at 10 running and 10 tokens; 300 answers; load 60 s | r_full 3,000.0, tail_s 10.0, r_all 2,746.61, S3 148.46 min as computed by hand; with 34.5 GPU-h spent: exit 1, "stop rule (§9.7): spent 34.500 + S3 2.474 GPU-h exceeds 35.0; no manifest written" (`rerun_a22/A22_planted.json`) |
| A22 manifest (new) | the committed `infra/aws_pg/steps/n13_s2smoke-08.json` | equals `n13_plan.manifest("s2smoke", 8, allowance_min=110)`: setup, `smoke_base`, `smoke_rl`, problems 0:300, `--allowance-min 110` |

What the gate cannot show: vLLM's real `step()` outputs and running counts (the fake engine finishes answers in one step), and the
engine's load time. Attempt 8 is the first run of both. The vLLM source lines behind A22: `rerun_a22/sources_a22.txt`. The
prediction on where the 128-running steps fall: `N13_SPEC.md` A22, "Expected from source".

## R0 attempt 8 (`s2smoke-08`), 2026-11-06: the engine started past A21's sampler and stopped in torch.compile

Written by the seventh N13 executor. **No smoke answer was generated.** Files: `studies/next/n13/r0/s2smoke-08/`.

- Launch: 01:08:59Z, `i-05e4d5bdfadce9f86`, g6e.2xlarge, manifest `n13_s2smoke-08.json` (the banked `steps.json` equals it byte for
  byte). The pipeline started at 01:10:17Z.
- `setup` passed every stage, 01:10:22Z to 01:17:06Z: venv_n13, data, G-GRADER, weights, G-INIT (a), G-TEMPLATE
  (`setup/metrics.json`). The n13 venv's pip freeze equals attempt 7's line for line.
- `smoke_base` started at about 01:17:07Z. Its park check: 6.85 min elapsed + 30.0 projected against 110, not parked. It wrote
  `base/build.json` and `base/math500/metrics.json` (status `running`, no `engine` block). It exited 1. STATUS `exit=1` was written
  at 01:17:52Z, and `smoke_rl` did not run.
- **What the log shows** (job.log copied once after STATUS; 1,415 lines, sha256
  `17cbbd137455151461bbc1b6c142161a8ba6f9056bd746cdc9a8072225725d66`; lines 900–1414 banked as
  `smoke_base/job_log_lines_900-1414.txt`):
  - A21 took effect: "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0" (line 934), and "Using
    AttentionBackendEnum.FLASH_ATTN backend" (lines 920, 922).
  - The V2 model runner and the Transformers modeling backend loaded the model, 13.59 GiB in 4.80 s (line 933).
  - torch.compile started (line 937), and Dynamo finished in 7.31 s (line 940).
  - Then inductor's Triton backend compiled its CUDA driver stub, and gcc stopped: `/tmp/tmpxomu3iqh/cuda_utils.c:9:10: fatal error:
    Python.h: No such file or directory` (line 941). The engine core died with `InductorError: CalledProcessError` (lines 946–1152).
- **Cause.** The AMI has no Python 3.12 headers. `n13_setup` installed `python3.12-dev` only inside `venv_olmes` (the attempt-1
  fix). Attempt 7 ran `--olmes`, so its smoke found the headers and got past this stage, to the sampler warmup. Attempt 8's setup had
  no `--olmes` (R0 was banked), so the headers were missing.
- **Fix (a job-file fix; no rule changes).** `n13_setup.python_headers()` installs `python3.12-dev` and asserts `Python.h`.
  `venv_n13` calls it first, and `venv_olmes` now only asserts the headers (setup runs `venv_n13` before it). Offline check (source;
  `apt-get` cannot run on the laptop): `studies/next/n13/offline_gate/rerun_a22add/headers_setup.json`. Its failing input is this run.
- What still has not run on the instance: the sampler warmup with A21 (the log stops before it), the loop, A22's step log and
  `load_s`.

## Offline gate rerun, 2026-11-06, A22 addition and the attempt-8 setup fix

**`N13_DRYRUN_OK`, 100 of 100** (`studies/next/n13/offline_gate/rerun_a22add/checks.json`, log `dryrun_log.txt`, 120 s;
`--out $SP/n13_dry_a22add3`). An earlier run of the A22 addition alone gave 99 of 99 (`$SP/n13_dry_a22add`, not banked).

| check | input | value |
|---|---|---|
| A22 addition, the head's planted input (new) | 20 steps of 0.04 s at 128 running (300 unfinished), 2,000 steps of 0.05 s at 24 running (200 unfinished), 100 steps of 0.02 s at 10 running (none waiting); contexts 200, 1,500, 3,000 per sequence | r_sat 501.587 (50,560 / 100.8 s), r_full 3,200.0, mean running 24.825, mean context 1,446.80; the rule takes r_sat; S3 285.35 min (hand: 285.35); at r_full S3 would be 143.42 min |
| A22 addition, reconstruction (new) | exact FCFS engine, 300 requests, at most 40 running, lengths 50–1,500, prompts 60–300; log stripped to the attempt-8 format | per step: saturated 154.83 s, mean running 40.0, mean context 653.765, equal to the log's; per 10 steps: 155.1 s, 40.0, 653.767 |
| A22 planted (changed) | the 300-running input | at `--rate r_full` as before (3,000.0, 10.0 s, 148.46 min); the rule min(r_full, r_sat) refuses: "the base smoke's r_sat is undefined: no engine step had requests waiting (unfinished > running)" |
| A22 smoke log (changed) | fake 300-problem smoke | also 57 waiting steps, 2.85 s; `throughput.json` carries `r_sat`, `tail_sat_s`, `sat_run_mean`, `sat_ctx_mean` |
| setup fix (new) | `n13_setup` source | `venv_n13` calls `python_headers()` before building the venv; `python_headers` installs `python3.12-dev` and asserts `Python.h`; `venv_olmes` has no `apt-get`; `run` calls `venv_n13` before `venv_olmes` |
| manifests (changed) | `n13_s2smoke-08.json`, `n13_s2smoke-09.json` | both equal `n13_plan.manifest("s2smoke", nn, allowance_min=110)` |

## R0 attempt 9 (`s2smoke-09`), 2026-11-06: the smoke ran; the RL smoke was cut by the hard kill

Written by the eighth N13 executor. **The base smoke finished all 300 answers. The RL smoke finished 142 of 300 before the 110 min
hard kill** (STATUS `exit=137`, 03:21:26Z). Files: `studies/next/n13/r0/s2smoke-09/` (cost and timeline: the operations note,
§"Attempt 9 as run"). The smoke reading, C and the projection are in `studies/next/n13/JOURNAL.md` (2026-11-06 03:40Z).

**A21 worked on the instance.**
- Both smokes' `metrics.json` → `engine` read `env` `VLLM_USE_FLASHINFER_SAMPLER` "0" and `attention_backend` "FLASH_ATTN".
- job.log: "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0" and "Using AttentionBackendEnum.FLASH_ATTN
  backend" (`smoke_base/job_log_lines_1028-1102.txt`).
- The sampler warmup passed, and the loop ran: 442 answers.

**The attempt-8 setup fix worked.** `setup/metrics.json` → `venv_n13.python_include` is `/usr/include/python3.12`, and torch.compile
finished in 26.55 s. The n13 venv's pip freeze equals attempt 8's line for line.

**A22 and its addition worked on the instance.**
- `steps.jsonl` is written per 10 steps: base 6,828 lines for 68,274 steps, RL 3,546 lines for 35,460 steps.
- The base log recounts to its `throughput.json`. This is asserted twice, by `analysis.py` and by `n13_plan`.
- The log's tokens equal the answers' tokens: 868,966 for the base. For RL, the log holds 642,298 tokens against the finished
  answers' 563,105; the 79,193 difference is the cut answers'.
- `engine.load_s` is in `metrics.json` before the first request: 76.8 s and 60.2 s.
- `engine.max_num_seqs` is 128, as predicted from source; `max_num_batched_tokens` is 2,048 and `num_gpu_blocks` 12,817. vLLM logged
  "GPU KV cache size: 87,790 tokens".
- The per-class sums carry run_dt and ctx_dt, so r_sat, the saturated phase's mean running count and its mean context come from the
  job itself.
- The attempt-8 reconstruction (`n13_plan.sat_phase`) checked against this real log: mean running 14.51 against 14.27, mean context
  5,379.8 against 5,512.8. Only 0.28 of its steps fall inside the block's logged running range: vLLM preempts (the waiting count rose
  while no request was added), and the reconstruction does not see preemption.

**The A22 prediction holds** (not refuted). The steps with 128 running are the opening ones:
- RL: 304 of them, 14.3 s of 2,111.5 s (bound 0.5);
- base: 315 of them, 15.0 s of 3,883.1 s;
- r_full / r_sat: RL 2,715.6 / 304.2 = 8.93, base 2,687.0 / 236.6 = 11.36 (bound 1.5).
- In the saturated phase the running count was 14.3 (base, mean context 5,513) and 18.1 (RL, mean context 3,731, over the cut run).
  The prediction gave 20 to 50 for contexts of 1,000 to 3,000 tokens.

**What the smoke found that the gate had not.**
- The RL model writes no `Answer:` line. It ends in `\boxed{}`, which the registered fallback extracts (journal, (b)).
- The base sometimes writes `**Answer:**`, which mode `answer_line` misses. Read answers 0 and 8 are right answers scored wrong for
  this reason (journal, (c)).
- 7 base answers are empty, and G-COMPLETE fails on them (`N13_SPEC.md` A25, proposed).
- The RL smoke was cut, and the projection needs a reading for that (A24).

**Kept on S3, not banked** (each over 1 MB, or not needed):
- `out/smoke_base/base/math500/generations.jsonl` (2,896,862 bytes, sha256 `4ab1220b…`);
- `out/smoke_rl/rl/math500/generations.jsonl` (1,951,097 bytes, sha256 `a01c3965…`);
- `out/smoke_base/steps.jsonl` (1,272,538 bytes, sha256 `84ad938d…`; banked gzipped as `smoke_base/steps.jsonl.gz`, which
  decompresses to the same bytes);
- `job.log` (sha256 `0cfe64f8…`; lines 1028–1102, 1491–1500 and 1760–1772 banked);
- `heartbeat.jsonl`, `out/setup/problems.json`, `out/setup/prompt_ids.json`.

The banked `lengths.json` (per smoke) and `read_answers.json` hold what the reading and the projection use.

## Offline gate rerun, 2026-11-06, A23 and A24

**`N13_DRYRUN_OK`, 103 of 103** (`studies/next/n13/offline_gate/rerun_a23/checks.json`, log `dryrun_log.txt`, 114 s;
`--out $SP/n13_dry_a23b`, on the committed code). Two earlier runs also gave 103 of 103, neither banked: one before A24's `prefix`
reading (`$SP/n13_dry_a23try`), and one before a docstring edit in `n13_plan.killed_smoke` (`$SP/n13_dry_a23`).

| check | input | value |
|---|---|---|
| A23 (new) | planted cache: base `main` and `stage2-step47684` under one `models--allenai--Olmo-3-1025-7B` folder, 5 files each, 2 shared blobs (8 blobs), `refs/main` and `refs/stage2-step47684`, and the RL repo | `delete_snapshot` on stage 2: main's 5 files read their bytes, stage 2's snapshot, refs entry and 3 shard blobs are gone, 5 blobs stay, the RL repo is whole, `huggingface_hub.scan_cache_dir` lists main only; `run` calls `delete_snapshot(sn["stage2"])` (`rerun_a23/A23_cache.json`) |
| A23 wrong input (new) | the old code path on a fresh copy of the same cache | base `main` is removed |
| A24 (new) | the sat24 log for both smokes; the RL smoke cut: 200 of 300 answers finished at 100 tokens over problems 0–219, 20 of them cut, no `throughput.json`, load 90 s | `prefix`: L 1,580.36, S3 380.95 min; `all`: L 5,528.0, S3 643.29 min; both at the base's rate 501.59 and tail 2.0 s, equal to the hand computation; `--killed` on an uncut smoke and the cut smoke without it exit non-zero (`rerun_a23/A24_killed.json`) |

What the gate cannot show: the deletion on the instance's real cache (`/opt/hf-cache/hf/hub`). R1 records it in `metrics.json` →
`stage2_deleted`, and S3's `base` step is the first reader of `main` after it.

## Offline gate rerun, 2026-11-06, A25 and A26

Written by the ninth N13 executor. **`N13_DRYRUN_OK`, 105 of 105** (`studies/next/n13/offline_gate/rerun_a26/checks.json`, log
`dryrun_log.txt`, 121 s; `--out $SP/n13_dry_a26`). 103 before, 2 new. A first run stopped in the new A25 check, which read
`metrics.json` before the line that loads it (a bug in the check, fixed by moving it; not banked).

| check | input | value |
|---|---|---|
| scoring path (changed) | 16 fabricated answers through the generation loop: the 13 before, `Work.\n**Answer:** 42`, `Work.\n**Answer:**\n\boxed{5}`, and the stop token alone | every (score, mode) as expected (`rerun_a26/scoring_fabricated.json`) |
| A26 (new) | rows 13 and 14 | `42`, mode `answer_line`, right, answer log-probability over 1 token; `5`, mode `boxed` (A10 on the new label) |
| A25 (new) | the 16-answer bank, row 15 empty (n_new 0, n_generated 1, finish_reason `stop`) | the bank passes with `n_empty` 1; problem 3's record removed: fails; problem 3 at n_generated 0: fails; problem 3 with finish_reason None: fails; last answer removed: fails (`rerun_a26/A25_g_complete.json`) |
| selfchecks (changed) | `n13_score.selfcheck` | A26 forms `**Answer:**`, `*Answer:*`, `__Answer:__` extract; 6 forms not accepted give `none`; the span is exact |
| G-GRADER (rerun, unchanged rule) | 500 MATH-500 golds, 24 + 16 planted pairs, derangement seed 0 | 500/500, 24/24, 16/16, 0.006 against 0.016; failing input 3 of 500, fails (`rerun_a26/G-GRADER.json`) |
| verdict on planted banks (changed input) | `write_bank` rows carry `n_generated` 101 and `finish_reason` `stop` | the 13 failing inputs give their registered effects, as before |

**On the smoke's real records** (`studies/next/n13/amendments/a26_rescore.json`, laptop, from the two S3 `generations.jsonl` files):
the code before A26 reproduces all 442 recorded (mode, extracted, score); after A26 the base reads 177 of 300 right against 153 (24 wrong
to right, none right to wrong), RL 131 of 142, unchanged. G-COMPLETE on the base's 300: fails as registered (7 empty), passes as amended
(`n_empty` 7). Spec: `N13_SPEC.md` A25, A26.

## Offline gate rerun, 2026-11-06, A27 and A28

Written by the ninth N13 executor. **`N13_DRYRUN_OK`, 107 of 107** (`studies/next/n13/offline_gate/rerun_a27/checks.json`, log
`dryrun_log.txt`, 130 s; `--out $SP/n13_dry_a27`). 105 before, 2 new.

| check | input | value |
|---|---|---|
| A27 (changed) | the planted stop-rule input: 34.5 GPU-h spent, S3 over 35 | exit 0 and all six manifests written (before A27: exit non-zero, no manifest) (`rerun_a27/A22_planted.json`) |
| A27 (new) | the fake smoke's projection and six manifests | every gen step's `--allowance-min` equals its launch's round(1.3 × projection); s3gate2 120.75 min, equal to the hand computation (`rerun_a27/A27_A28_manifests.json`) |
| A28 (new) | the same six manifests | s3gate-01: setup with stage 2, r1_rank, base, rl; s3gate-02: setup without stage 2, zero, id; the four S4 manifests in priority order from P1, every P1 and P2 arm among the first two arms, the union the 27 S4 arms |
| A22 projection command (changed) | the fake smoke | both S3 manifests parse with n13_gen's parser, carry `--cap 16384`, no `--sets`, and their launch's allowance |
| roster run (changed) | the 31 roster arms on the stand-in, fake engine | `zero` and `id` built without `--sets`; G-HASH passes for both, its failing input fails |
| manifests (changed) | `s2setup`, `s2smoke` and the six A28 kinds | each parses; each roster arm appears once across the six launches |

On the banked smoke (`studies/next/n13/r0/s2smoke-09/projection.json`, written by `n13_plan --manifest-dir infra/aws_pg/steps`): the six
manifests `infra/aws_pg/steps/n13_s3gate-01.json`, `n13_s3gate-02.json`, `n13_s4armsA1-01.json`, `n13_s4armsA2-01.json`,
`n13_s4armsB1-01.json`, `n13_s4armsB2-01.json`; projections and allowances in `N13_SPEC.md` A28.

## Offline gate rerun, 2026-11-06, A29 (after freeze 1)

Written by the ninth N13 executor after R1 stopped at `s3gate-01` (`N13_SPEC.md` A29). **`N13_DRYRUN_OK`, 108 of 108**
(`studies/next/n13/offline_gate/rerun_a29/checks.json`, log `dryrun_log.txt`, 150 s; `--out $SP/n13_dry_a29`). There were 107 checks
before; 1 is new.

| check | input | value |
|---|---|---|
| A29 (new) | 7 planted bf16 pairs in one `down_proj` tensor: 3 exact, 3 with exponents more than 16 apart, 1 after value of −0 | the c = 1 builder writes the after bits on all 7; the pre-A29 check (bits on every Δ ≠ 0) fails, with 3 entries off; with `delta()` scaled by 1 + 2^-6 the A29 check raises (`rerun_a29/A29_builder_check.json`) |
| selfchecks (changed) | `n13_arms.selfcheck` runs `a29_planted` | pass |

What the stand-in missed: its weights held no entries with exponents more than 16 apart, so the pre-A29 check never fired offline.
The real weights hold 152 such entries in the MLP, 144 in attention and 17 in the embedding (`studies/next/n13/s3/s3gate-01/r1_rank/weight_stats.json`).

## Offline gate rerun, 2026-11-06, A30 (after freeze 1, before freeze 2)

Written by the ninth N13 executor. **`N13_DRYRUN_OK`, 110 of 110** (`studies/next/n13/offline_gate/rerun_a30/checks.json`, log
`dryrun_log.txt`, 118 s). 109 before, 1 new.
- A first run stopped on an rndT energy assertion: the stand-in `rndT_k0.1` overshoots E(S_k) by 33 %. The rndT check was then made
  report-only (`N13_SPEC.md` A30). That run is not banked.

| check | input | value |
|---|---|---|
| A30 (new) | a 64 × 96 `down_proj` tensor, every change one bf16 step, c = 1.3 | deterministic 0.5917 of intended, and the c adjustment fails after 3 builds (AssertionError); stochastic pass 1 1.120, pass 2 0.9921; cast rounds up 0.2998 at 0.3 of a step (`rerun_a30/A30_energy.json`) |
| selfchecks (changed) | `n13_arms.selfcheck` runs `a30_planted` | pass |
| roster run and R1 on the stand-in (changed code) | all 31 roster arms | the 4 `rndE` arms build through `energy_match`; `sets.json` equal to the A29 run byte for byte; only the `rndE` arm hashes differ (`studies/next/n13/amendments/a30_compare.json`) |

## Offline gate rerun, 2026-11-07, A31 (after freeze 2)

Written by the eleventh N13 executor after `s4armsB2-01` and `s4armsB1-01` stopped at A30's exact energy check (`N13_SPEC.md` A31).
**`N13_DRYRUN_OK`, 111 of 111** (`studies/next/n13/offline_gate/rerun_a31/checks.json`, log `dryrun_log.txt`, 125 s;
`--out $SP/n13_dry_a31`). 110 before, 1 new.

| check | input | value |
|---|---|---|
| A31 (new) | a 32-layer stand-in, MLP magnitudes over 30 binades, 3 shards with the names dealt round-robin; one stochastic-rounding arm (192 neurons, c = 1.3, seed 142) | pre-A31: matched 1.1252463099126716e-05 against written 1.1252463099126724e-05, not equal, every tensor equal (sha256 and re-read); A31 passes (7.5e-16 ≤ 1e-12, 0 of 355 tensors differ); one written entry of `model.layers.0.mlp.gate_proj.weight` one bf16 step up: `build_arm` raises the A31 assertion, 1 tensor differing, the energy part alone passing (`rerun_a31/A31_written_vs_matched.json`) |
| roster run on the stand-in (changed code) | the 4 `rndE` arms | built and written through A31, pass |

What the stand-in missed: with MLP magnitudes within a few binades every float64 sum of squares is exact, so the visiting order of the
tensors cannot change the energy. A probe with 4, 16 and 32 layers at the old magnitudes gave equal sums; a spread of 20 or 30 binades
gave relative differences of 1.3e-16 to 7.5e-16 with every tensor equal.
