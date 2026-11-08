# 09 — Project plan for the next program, ordered by leverage

**DRAFT by Opus for Fable's review, 2026-11-05**

Written from the files named in each row. No model was run, nothing was launched, no paid call was made.

Terms used here. A **job** is a work item, J0 to J14. The study's word for a group of a pole's top-50
contexts is written **cluster** here, except inside the quoted claim "the main job is the oldest". A
**pole** is one sign of one MLP (multi-layer perceptron) neuron. A **birth point** is the first checkpoint at
which a cluster's median activation reaches the pole's own 99.9th percentile there and at the next
checkpoint (`studies/next/README.md`, Vocabulary). A **lane** is one GPU provider plus one GPU model.
**Parked** means not run and listed as not run in the verdict (`studies/next/07_PICKS/N13_SPEC.md` §9.7).
Prices: AWS g6e.2xlarge (one L40S, 48 GB) $2.24 per hour, g6e.4xlarge $3.00424 per hour
(`studies/next/00_ASSETS.md` lines 222–223). Paths without a prefix are under
`/Users/abaybektursun/projects/interp_research-clean-neurons/`. `BANK/` is
`/Users/abaybektursun/projects/interp_research_banks/`. `S3/` is `s3://pg-lineage-769063704696/`.

## 1. Answer first

1. Order: J1 (recount and retag) and J2 (N13 offline gate), both running at 0 GPU-hours; then J3 (M1 pre-registration, loader fixes, offline gate, 0 GPU-hours); J4 (GPU-lane calibration, 0.3 GPU-hours); J5 (N13 known number, ranking, smoke, 3.8–5.0 GPU-hours); J6 (N13 gate and arms). J7 (M1 harvests) runs beside J6; J8 (second data-order check) runs beside J4.
2. Why: J1 to J4 cost almost nothing and decide whether and where the two large runs spend; J5 opens the first pick with a published number before any new one; J6 is the measurement the program exists for.
3. Through J8: 34.5–47.0 GPU-hours, $77.28–$105.28 at $2.24 per hour (N13 held to its registered 35 GPU-hours); $0 of API calls by the jobs; about $80–$150 of Opus agent tokens (estimate: 10 agents at the head's $8–15 each).
4. The rule that stops waste: no GPU minute before the job's offline gate is green on the laptop, and the first GPU minutes of each study reproduce one known number (N13: Minerva MATH 40.0 ± 1.6 on the base; M1: corpus loss 2.675 nats per token at `stage1-step1413814`).
5. AWS L40S is the default lane; a job moves to Vast only if J4 measures Vast at least 1.5 times cheaper per unit of work and the host passes the reliability check; the Vast ceiling has $6.31 of room today, so any larger Vast run needs the user.

## 2. The job list, ordered by leverage

| job | what it is | question it settles | why it is before the next one | owner | branch, worktree | depends on |
|---|---|---|---|---|---|---|
| J0 | git centralization | — | done (`studies/next/JOURNAL.md`, 09:10 CST entry) | Fable | `clean-neurons` | — |
| J1 | spec-only recount, retag of D1, births picture | does "the main job is the oldest" survive a control that keeps how the main cluster is chosen? | corrects a closed record at 0 GPU-hours and fixes the unit (the passing births) for J9, J11, J12 | Opus (running), Fable commits | recount: `studies/next/10_RECOUNT2.md`, `BANK/neurons/next/recount2/`; retag: new `../interp_research-retag` (branch `retag`) | — |
| J2 | N13 offline gate | does the N13 code run green and fail on every planted failing input? | gates up to 35 GPU-hours | Opus (running) | `../interp_research-n13` (branch `n13`) | — |
| J3 | M1 pre-registration, loader fixes, offline gate | is the M1 measurement written down with a refutation that can fire, and does the loader read the mix weights and not `main`? | gates the second pick at 0 GPU-hours | Opus | `../interp_research-m1` (branch `m1`, exists) | — |
| J4 | GPU-lane calibration | which lane is cheaper per generated token and per forward-pass row, and does Vast finish a job unattended? | 0.3 GPU-hours decide where about 45 GPU-hours run; R0 must run on the final lane | Opus | new `../interp_research-lane` (branch `lane`) | Vast preflight stamp |
| J5 | N13 S2: known number, ranking, smoke | do the base weights give Minerva MATH 40.0 ± 1.6; which checkpoint did RL start from; how long are the answers? | the first spend on the first pick starts with a published number | Opus | `n13` | J2, J4 |
| J6 | N13 S3 gate, S4 arms, S5 close | is the RL-Zero MATH-500 gain carried by the most-changed MLP neurons, beyond DARE (drop and rescale) and energy-matched random neurons? | the main result; S4 spends only after the noise gate passes | Opus | `n13` | J5 |
| J7 | M1 harvests and verdict | do neurons move toward code and math documents in the order of the mixes' code-plus-math share, 0 < 40 < 65 percent? | the second main result; uses the instances N13 leaves free | Opus | `m1` | J3, J4 |
| J8 | second data-order check | does the reconstructed batch reproduce the logged loss at two new steps under a rule frozen in advance? | cheap, independent, closes an asset claim before anyone cites it | Opus (J4's agent) | `lane` | its rule frozen |
| J9 | M2: name the carrier before the birth | does a frozen score rank the eventual carrier pole in its layer's top 10 before the birth? | a prediction made before the measurement; needs J1's births | Opus | new `../interp_research-m2` | J1 |
| J10 | N22 with N25 | does OLMo-3 use an intermediate value as a variable whose next operation depends on its position, and from which checkpoint? | independent of births, but its decisiveness was scored 3 of 5 (`05_SCORING.md` line 32) | Opus | new `../interp_research-n22` | — |
| J11 | N6: edit the young | at equal edit size and matched energy, does an early-born pole cost more corpus loss than a late-born one? | needs J1's adjusted ages | Opus | new `../interp_research-n6` | J1 |
| J12 | N1, N2, N3, N12 on passing births | the weight order, use, route and competition at birth | needs J1, J9, the unite Arm 1 review and J7's identity pair | Opus | new `../interp_research-bio` | J1, J9, J7 |
| J13 | robot grounding gate (deferred) | does MolmoAct2 on LIBERO follow the words at all? | deferred until the N13 and M1 main results are in (decision 4) | Opus | new `../interp_research-robotgate` | J6, J7, user |
| J14 | final synthesis | what we learned and the highest-leverage next step | last | Fable writes; one Opus reader | `clean-neurons` | J6, J7 |

Changes to the candidate list, each with its reason:

- **J3 gains a first step, the M1 pre-registration.** Only an asset check exists (`07_PICKS/M1_ASSETS.md`).
  Its lines 31–42 say the written refutation "has almost no room to fire". A measurement is written down
  before it is taken (`CLAUDE.md`, "Before any compute", item 3).
- **J5 is the spec's S2, 3.8–5.0 GPU-hours, not 1–2.** R0 alone is 2.0–3.0 GPU-hours (`N13_SPEC.md`
  line 1006), and R1 and the smoke share its launch (lines 925–935).
- **J7 uses five harvests, not six.** The identity pair `stage1-step476000` / `stage1-step476838` shares the
  start (`M1_ASSETS.md` line 261, 8.4–9.5 GPU-hours). The main-run pair has no room to fire (lines 31–42).
  The same stage-1 pair is the identity-pair harvest Program A needs (`06_CRITIC.md` Part 6, about 3.5
  GPU-hours), so J12's instrument check costs 0 extra GPU-hours.
- **J4 and J8 share one executor.** Both are short single-instance launches.
- **J6 and J7 each include an independent recount** by a fresh agent with its own code.
- **J4 measures harvest speed only on a GPU model with no banked rate.** The L40S (3,215–3,334 tokens per
  second) and RTX PRO 5000 Blackwell (4,551–4,623) rates are banked (`00_ASSETS.md` lines 22–39).

## 3. Per-job cards

### J0 — git centralization (done)

| item | content |
|---|---|
| state | committed as `ed3e72006` on `clean-neurons` (`JOURNAL.md` 09:10 CST). `git worktree list` shows `../interp_research-n13` (`n13`) and `../interp_research-m1` (`m1`), both at `ed3e72006`. |
| one fact for every executor | the launcher wrapper runs `<worktree>/.pgvenv/bin/python` (`infra/aws_pg/pg` line 5). The `clean-neurons` worktree has no `.pgvenv`, so `pg --help` did not run there. Every worktree that launches needs its own `.pgvenv`. |

### J1 — recount, retag, births picture

| item | content |
|---|---|
| inputs | the rule, `studies/already-clean/SPEC_D1.md` §2.4; `BANK/neurons/w3/grid/*/out/*/poles/exemplar_a.npz`; `BANK/neurons/w5/verdict/samples/q2_clusters.jsonl`; `BANK/neurons/w5/verdict/metrics.json` → `q2.birth_order`; the two prior recounts, `studies/next/06_CRITIC.md` Part 0 and `studies/next/06b_VERIFY_BIRTHS.md` (read only after the recount) |
| pre-checks before money | no money. Failing input: lowering one cluster's median below the threshold at its birth checkpoint must move its birth later. |
| go/no-go, as numbers | retag only if the recount gives 0 mismatches over 2,454 clusters, b within 0.0005 of 0.4756, random-group b within 0.005 of 0.530 with its 2.5th percentile above 0.476, and 145–155 of 1,238 late-born clusters passing against a chance level of 28.3 (`06b_VERIFY_BIRTHS.md` summary table). If any fails: no retag; Fable reads all three recounts and records the disagreement first. |
| retag text | "b 0.476 lies below the random-group reference 0.530 [0.497, 0.563] (300 draws, one-sided p 0.0067); the registered permutation reference (97.5th percentile 0.449) treats the main cluster as arbitrary." |
| cost | 0 GPU-hours, $0 API. One agent, continuing from the recount into the retag. |
| artifacts | `studies/next/10_RECOUNT2.md`; `BANK/neurons/next/recount2/{arrays.npz, births.jsonl}` (exist); the retag in `studies/already-clean/results/RESULTS.md` line 7 and the row at line 88, `REGISTRY.md` line 27; the regenerated `F4_births` figure (`REGISTRY.md` line 12) and `studies/already-clean/walkthrough/births/` (`extract.py` → `data.js`) with the random-group reference drawn |
| review | Fable compares the five numbers across the three implementations (critic, `06b_verify_births/`, recount2) and reads the regenerated picture. Fable's read is the different-model read. Fable commits named files. |
| ten raw examples first | 5 passing and 5 failing late-born clusters: the 16-point median curve, 3 random same-size groups from the same pole, the pole's 99.9th-percentile line, the decoded top contexts |

### J2 — N13 offline gate

| item | content |
|---|---|
| inputs | `07_PICKS/N13_SPEC.md` §9.2 (new and reused code) and §9.3 (the gate, items 1–7) |
| pre-checks before money | this job is the pre-check: stand-in weights with 8 planted neurons, a one-step rounding change and a dense "stage 2"; a fake engine with 12 fabricated problems; the G-GRADER set of 24 equivalent and 16 non-equivalent answer pairs |
| go/no-go | `N13_DRYRUN_OK` printed; each planted verdict branch of §9.3 item 5 produced; every gate's failing input fails; then the tag `n13-freeze-1` |
| cost | 0 GPU-hours. One agent (running). |
| artifacts | `studies/next/07_PICKS/N13_OFFLINE_GATE.md` and the §9.2 modules, in the `n13` worktree; `studies/next/n13/STATE.md` |
| review | Fable reads the dry-run log and the list of failing-input outputs. J5's executor re-runs the dry run from a fresh context before its first launch. If J4 picks Vast for N13, a dated amendment to §9.1 (instance type) and §9.1's S3 layout (Vast boxes write under `results/mdl/*`, `infra/vast/cells/pipeline_job.py` line 11) lands before freeze 1. |
| ten raw examples first | the first 10 of the 12 fabricated problems: rendered prompt, canned output, extraction mode, `math_verify` result, score |

### J3 — M1 pre-registration, loader fixes, offline gate

| item | content |
|---|---|
| inputs | `07_PICKS/M1_ASSETS.md` changes 1–6; `infra/pg/neurons_harvest.py` lines 49, 137–142; `infra/pg/matrix_models.py` line 109; the corpus manifest `S3/results/neurons/w1/q4b/1/out/corpus/corpus/manifest.json` |
| pre-checks: the spec | written with the research-spec skill. Quantity per pole: the share of its top-50 positions that lie in the 4 `stack_edu-*` directories and `finemath-3plus` (`M1_ASSETS.md` change 6); the corpus share of those tokens is about 0.10 (line 240). Shift = end − start. Test: the pre-stated order gen-mc-only < round5 < math-code-reasoning-web (0, 40, 65 percent code plus math, line 123), lead contrast the first against the last. Floor: the identity pair `stage1-step476000` / `stage1-step476838`. A second category definition that adds the 8 `common_crawl` science and software directories is reported beside it. Predictions and a results table with blank cells are written in. A room check runs on synthetic shifts: the rule must return each order. The critic of `README.md` stage 6 reads it before freeze. |
| pre-checks: the loader | the config rewrite `olmo2-retrofit` → `olmo3` asserted over all 355 tensor names (line 17); the download filter `model-*-of-00006.safetensors`, 29.2 GB instead of 43.8 GB (lines 20–26); a load gate that fails when the weights equal `main`, with the three stale bf16 shards (`0490d6668e61…`, `e127ea479fb6…`, `f3ddff10052f…`) as its failing input, checked offline at the level of LFS ids; the harvest also writes a per-pole share array of 704,512 values, so the laptop reads megabytes, not 1.9 GB per harvest |
| go/no-go | offline gate OK on stand-in weights; the load gate fails on its failing input; the ordering rule passes its room check; critic read done; tag `m1-freeze-1` |
| cost | 0 GPU-hours. Two agents: one writer (spec, then code, in sequence), one critic. |
| artifacts | `studies/next/m1/SPEC.md`, `studies/next/m1/OFFLINE_GATE.md`, code on branch `m1` |
| review | Fable reviews the spec and the gate output after the critic's read |
| ten raw examples first | 5 code and 5 math documents taken by the directory rule: first 300 characters and directory, to see that the label means what it says |

### J4 — GPU-lane calibration

| item | content |
|---|---|
| inputs | the N13 pins: vLLM 0.32.0 (`N13_SPEC.md` §2.3) and olmes at `5a51f502…` with vLLM 0.11.0 (§0.3 item 1); base `allenai/Olmo-3-1025-7B` @ `a81bae42…`; `infra/vast/` (`controller.py`, `cells/module_job.py`, `HOSTS.json`, `budget.json`); `infra/aws_pg/config.py` |
| pre-checks before money | the throughput script runs offline on CPU with a tiny model and fails on a planted wrong output length; Vast `preflight.sh` stamp; the Vast room, $6.31, covers 1.3 × the Vast half; manifest `watch` block `max_billed_h` 0.25, `load_backstop_min` 30; AWS `--budget-min` 20 |
| measurement, on each lane | (a) 64 MATH-500 prompts under the N13 template with `min_tokens = max_tokens = 2,048`, then 16 prompts at 8,192: aggregate output tokens per second; (b) Vast only: 4 Minerva samples through the olmes venv, to see that vLLM 0.11.0 runs on that GPU; (c) harvest forward on 200 windows of 2,048 tokens, only for a GPU model with no banked rate; (d) Vast only: download of one 4.97 GB shard from Hugging Face and upload of 1 GB to S3 |
| go/no-go | the lane rule of §4, from two numbers per lane: dollars per 1,000 generated tokens and dollars per 2,048-token row |
| cost | target 0.3 GPU-hours; estimate 0.3–0.4 (AWS: vLLM install about 5 minutes, an estimate; the 14.6 GB download took 75–102 s, `pilot0/GPU_CHECK.md` lines 128–129; generation about 5 minutes; Vast: the image pull is not billed, `infra/vast/README.md` lines 5–7). Stop at 0.5 GPU-hours and default to AWS. Dollars: AWS about $0.45; Vast about $0.30–$0.70 at $2.01–$2.67 per hour (RTX PRO 6000 WS $2.01, `SPEC_D1.md` line 720; H100 NVL quoted $2.67, `infra/aws_pg/operations/2026-09-05_vast_lane_scout.md` line 10) |
| artifacts | `S3/results/next_lane/<attempt>/out/metrics.json`; Vast `S3/results/mdl/next_lane/<rental>/` and `S3/results/shackleton_<rental>/`; `infra/aws_pg/operations/2026-11-<dd>_next_lane.md` and `infra/vast/operations/2026-11-<dd>_next_lane.md`. Not in any research journal (`CLAUDE.md`, Process 6). |
| review | Fable applies §4 to the two `metrics.json` files and the Vast spend record. Infrastructure: no second reader. |
| ten raw examples first | the same 10 prompts decoded greedily on both lanes: full text, token count, finish reason, the prompt token ids |

### J5 — N13 S2: known number, ranking, smoke

| item | content |
|---|---|
| inputs | `N13_SPEC.md` §9.4 S2; tag `n13-freeze-1`; J4's lane |
| pre-checks before money | sha256 of every downloaded shard against the LFS id (§4.1); G-GRADER if not run offline; then on the instance before any arm: G-PARTITION, G-INIT (a) the hash check against `39ff1a89…` and (b) a margin of at least 0.10 over `stage2-step47684`, PC-RANK at least 0.99 with its failing input at most 0.05, PC-ART J_round below 0.4 on the synthetic rounding change (§6.1) |
| go/no-go | G-KNOWN: \|100 × `primary_score` − 40.0\| ≤ 1.6 (line 612), the derangement rescore below 10.0. A miss: one relaunch with the fix; a second miss: cancelled (line 759). Cap C = 8,192 if at most 1 of 20 RL smoke answers exceeds 8,192 tokens, else 16,384 (lines 323–324). The projection for S3 and S4 is redone from the measured tokens per second and answer lengths; if spent plus projection exceeds 35, the stop rule parks the last arms before launch (lines 1016–1021). |
| cost | 3.8–5.0 GPU-hours, $8.51–$11.20 (lines 925, 1001); `--budget-min` 1.3 × the projection |
| artifacts | `S3/results/next_n13/s2setup-01/out/{r0_known, r1_rank, smoke}/`; mirror `studies/next/n13/results/s2setup-01/`; `studies/next/n13/JOURNAL.md`, `STATE.md`; cost in `infra/aws_pg/operations/` |
| review | Fable reads `r0_known/metrics.json` (`primary_score` of the aggregate task, never `pass_at_1_micro`), `r1_rank/init_id.json`, and the journal's answers to (a) is the input what §2.2 says, (b) is the output in the expected format, (c) is the scored quantity what the model produced |
| ten raw examples first | R0: 5 right and 5 wrong Minerva samples (§10.1); smoke: 10 base and 10 RL answers read in full (§9.4) |

### J6 — N13 S3 gate, S4 arms, S5 close

| item | content |
|---|---|
| inputs | J5's banks; `studies/next/n13/thresholds.json` |
| pre-checks before money | G-HASH (`zero` equals the base and `id` equals RL, bit for bit); G-COMPLETE; `thresholds.json` → `room` true (G ≥ 5√2σ = 2.5δ, §6.2); the author's prior (§1.5) copied into the journal before S3; tag `n13-freeze-2` before any arm |
| go/no-go | G-GATE: G ≥ 5√2σ and G' ≥ 5√2σ and exact McNemar p < 0.001 (line 617); G-ID: \|L − L'\| ≤ 3√2σ (line 618); S-BAND 81.5 ≤ L' ≤ 96.7 stamps, does not cancel (line 619). G-GATE fails: pair 2 (prior 0.2 that it passes, line 268); pair 2 fails: cancelled, and S4 is not spent. |
| cost | S3 4.1–6.2, S4 17.6–26.5 GPU-hours (lines 1002–1003); J5 + J6 at most 35 GPU-hours (the registered ceiling); the stop rule parks the lowest-priority arms (§9.7) |
| artifacts | `S3/results/next_n13/{s3gate-01, s4armsA-01, s4armsB-01}/out/<step>/<arm>/<bench>/{generations.jsonl, samples.jsonl, metrics.json}`; `studies/next/n13/{thresholds.json, verdict_n13.json, CLAIMS.md, REGISTRY.md, figures/fig_n13.png, results/SAMPLES_COVERAGE.md, READ_n13.md}` |
| review | the verdict script runs once. A fresh Opus agent recounts with code it writes itself and writes `READ_n13.md`. The spec's S5 has the reader run the executor's `recount_n13.py`; a second implementation is the stronger check (a proposed change). Fable reads the verdict, the samples and the recount. Whether Fable's read is the "different model" read of §9.4 S5 is an open question (Appendix B). |
| ten raw examples first | base-A and rl-A problems 0–9 in full before σ is computed; for every arm, problems 0, 1 and 2 and its first right and first wrong answer before any arm is compared (§10.1) |

### J7 — M1 harvests and verdict

| item | content |
|---|---|
| inputs | tag `m1-freeze-1`; branches `stage1-step476838`, the three `stage2-step47684-mix-*-from-2T-ckpt` ends, `stage1-step476000`; corpus `tok/olmo3/tokens.npy` |
| pre-checks before money | known number first: `stage1-step1413814` corpus loss on the first 120,000 tokens reproduces 2.675 (`00_ASSETS.md` line 52, `controls.json` → `nll_corpus_first_120k.mean_nll`) within a tolerance frozen from two attention backends; estimate 0.1 GPU-hours (the download took 62.3 s, `M1_ASSETS.md` line 249; 120,000 tokens at 3,334 per second). Then, per mix end, the load gate (not equal to `main`); per harvest, `complete` true, `G0_pass` true, `dead_poles` 0 (`00_ASSETS.md` lines 48–49) |
| go/no-go | known number within tolerance, then the harvests; verdict by the frozen ordering rule against the identity floor |
| cost | 8.4–9.5 GPU-hours for 5 harvests, 10.1–11.4 for 6 (`M1_ASSETS.md` lines 258–261); $18.82–$25.54. Proposed ceiling 12 GPU-hours. Order: start, math-code-reasoning-web, gen-mc-only, `stage1-step476000`, round5; a harvest that would cross the ceiling is parked. |
| artifacts | `S3/results/next_m1/<attempt>/out/<branch>/{metrics.json, manifest.json, controls.json, share.npy, samples/}`; `BANK/neurons/next/m1/`; `studies/next/m1/{JOURNAL.md, verdict_m1.json, figures/, READ_m1.md}` |
| review | Fable reads the gates, the ten raw examples and the verdict. A fresh Opus agent recomputes the shares from the raw top-50 positions and the manifest, with its own code, for every pole of one harvest and a seeded 1% of the rest. |
| ten raw examples first | 10 poles, the 5 largest shifts under math-code-reasoning-web and 5 at random: their top-50 contexts at the start and at that mix end, decoded, each with its directory |

### J8 — second data-order check

| item | content |
|---|---|
| inputs | `pilot0/loss_check.py`, `pilot0/reconstruct_batch.py`, `infra/aws_pg/jobs/next_pilot0_loss.py` (committed in `ed3e72006`); the second pair named in `00b_DATA_ORDER.md` line 124 (checkpoint `stage1-step61000`, batch 61,000, logged 2.1173441410064697) and one more step chosen before launch |
| pre-checks before money | the rule is frozen before launch and shown to have room. The first check's pad-excluded reading missed by +0.0055 (`pilot0/GPU_CHECK.md` lines 5–6), above the old floor of 0.003 (line 184), so a rule with that floor has little room to pass. Proposal: pad targets excluded; per step, \|S − W\| ≤ 0.0124, half the per-step sd, a band 10.7% of other steps fall in (line 188); and each step's batch closer to its own log than its neighbour batch. Two steps: false-confirm chance about 0.011 (0.107², estimate, assumes independence). The rule run on the banked first-check files must say confirmed for `batch_60001` and not confirmed with `batch_60000` in its place. Known number: the first 64 rows of `batch_60001` reproduce the banked `per_row_mean_ce` within 1e-4 (64 × 1.26 s = 81 s). |
| go/no-go | the frozen rule's outcome is the verdict |
| cost | registered about 0.3 GPU-hours (`08_SUMMARY.md`). Estimate from the measured 645 s per 512-row batch with sdpa (`pilot0/GPU_CHECK.md` line 122): two batches 0.36 GPU-hours, with a neighbour batch for each 0.72, plus two 14.6 GB downloads: 0.45–0.8 GPU-hours, $1.01–$1.79. Ceiling 1.0 GPU-hours. |
| artifacts | `S3/results/next_pilot0/2026-11-<dd>_b1/out/`; `studies/next/pilot0/GPU_CHECK2.md`; `studies/next/pilot0/batch_<step>/` |
| review | Fable applies the rule to the JSON files; that is the different-model read |
| ten raw examples first | the 10 rows with the largest row loss and 10 random rows: first 200 decoded tokens, pad count, row loss |

### J9 — M2: name the carrier before the birth

| item | content |
|---|---|
| inputs | J1's passing births (`BANK/neurons/next/recount2/births.jsonl`); `06_CRITIC.md` M2, lines 852–881 |
| pre-checks before money | at least 100 births pass J1, else set aside with the count (`06_CRITIC.md` line 929); spec and critic; known number: banked `exemplar_a.npz` activations reproduced for 10 poles at one checkpoint; positive control: at the birth point itself the carrier ranks in the top 10 by construction; layer- and activation-matched random poles |
| go/no-go | precision at 10 against chance 10/11,008, about 0.0009, and against the pole's own current percentile (wdd's best forecast lost to current usage, 0.964 against 0.927, `06_CRITIC.md` M2); thresholds frozen in the spec |
| cost | 5–8 GPU-hours, about 12 with gradients (`06_CRITIC.md` line 880); $11.20–$26.88 |
| artifacts | `studies/next/m2/`; `S3/results/next_m2/<attempt>/` |
| review | Fable; a fresh Opus recount |
| ten raw examples first | 10 births: the top-10 list at the checkpoint before, each pole's percentile, the carrier's rank, the decoded exemplar windows |

### J10 — N22 with N25

| item | content |
|---|---|
| inputs | `03_IDEAS.md` N22, N25; `01_JLENS.md` §5; `infra/pg/vendor/jlens`; `06_CRITIC.md` Part 3 |
| pre-checks before money | an accuracy gate at each checkpoint without chain of thought (about 1,000 single-token answers, chance about 0.1%, Part 3 item 2); the lens-free interchange positive control at the final checkpoint first; slots compared at matched depth; the exact-target 2 × 2 score; N25: a step-0 lens and the projection of the massive-activation direction; a split-half lens for thresholds; known number: the corpus loss at the loaded checkpoint (`controls.json`) |
| go/no-go | go past the final checkpoint only if it answers at least 100 expressions (`03_IDEAS.md` N22) and the interchange control exceeds both floors; else stop: the lens test had no chance (Part 3 item 1.3) |
| cost | N22 5–8 GPU-hours (`06_CRITIC.md` line 761); N25 3–5 (`03_IDEAS.md` N25); $17.92–$29.12 |
| artifacts | `studies/next/n22/`; `S3/results/next_n22/<attempt>/` |
| review | Fable; a fresh Opus recount |
| ten raw examples first | 10 expressions: prompt, answer token and its probability, the lens readout at the intermediate layer, the output after the swap |

### J11 — N6: edit the young

| item | content |
|---|---|
| inputs | J1's ages with the random-group reference; `06_CRITIC.md` N6, lines 642–684 |
| pre-checks before money | pairs in one layer matched on max\|a\|, density and column norm (age against log max\|a\| has Spearman −0.592); edits of equal Frobenius norm; a positive control (each layer's highest-energy pole); a random-direction edit; an efficacy readout; known number: one banked zero-replacement value of `studies/instruments/results/e1/verdict_e1.json` on O7 T_15 |
| go/no-go | matching brings \|ρ(age, log max\|a\|)\| below a value frozen in the spec; the positive control moves corpus loss more than the random-direction edit |
| cost | 4–8 GPU-hours (`05_SCORING.md` line 133); $8.96–$17.92 |
| artifacts | `studies/next/n6/`; `S3/results/next_n6/<attempt>/` |
| review | Fable; a fresh Opus recount |
| ten raw examples first | 10 edits: pole, partner, Frobenius norm, corpus loss change in nats per token, the 5 tokens whose loss changed most |

### J12 — N1, N2, N3, N12 on the passing births

| item | content |
|---|---|
| inputs | J1's births; J9's frozen competitor set; the unite Arm 1 banks; J7's identity-pair harvests |
| pre-checks before money | at least 100 passing births; N12 runs only as J9's prediction, N3 only after the unite Arm 1 review closes, N1 first locates onsets inside their brackets (`06_CRITIC.md` Part 6); known number: banked exemplar activations at the birth point for 10 births |
| go/no-go | at least 100 passing births (`06_CRITIC.md` line 929), else the cards are set aside with the count; each card's rule frozen in its spec |
| cost | about 8 GPU-hours for 100 births (`05_SCORING.md` lines 66–67); $17.92 |
| artifacts | `studies/next/bio/`; `S3/results/next_bio/<attempt>/` |
| review | Fable; a fresh Opus recount |
| ten raw examples first | 10 births: input-row and output-column cosines per interval, upstream contributors with their own ages |

### J13 — robot grounding gate (deferred)

| item | content |
|---|---|
| gate | blank and swapped instructions on four LIBERO suites at 50 episodes each, plus 200 unmodified: 600 episodes, 4.6 GPU-hours at 27.62 s per episode (`05_SCORING.md` line 143; `06_CRITIC.md`, Program C gate) |
| known number | unmodified LIBERO success 194/200 for `allenai/MolmoAct2-Think-LIBERO` (`00_ASSETS.md` line 199, DEPTHOFF) |
| go/no-go | frozen before the run: the robot program is set aside if blank and swapped instructions leave success inside the unmodified run's interval (`05_SCORING.md`, Program C) |
| when | after the N13 and M1 main results, on the user's word |

### J14 — final synthesis

| item | content |
|---|---|
| what | Fable writes the scientific abstract of §10 to `studies/next/20_ABSTRACT.md` (a proposed path) |
| go/no-go | shown to the user only when the reader finds 0 numbers without an artifact and 0 that differ from it |
| review | one fresh Opus reader checks every number against its artifact and lists any number without one |

## 4. GPU lane rule

**The two numbers.** J4 measures both on each lane.

- **G, dollars per 1,000 generated tokens**, for N13-style generation: hourly price ÷ (3.6 × aggregate output
  tokens per second), with vLLM 0.32.0 on Olmo-3-1025-7B under the N13 template.
- **R, dollars per forward-pass row**, for M1-style harvests: hourly price × seconds per 2,048-token harvest
  window ÷ 3,600. J8's 8,192-token row scales from it.

Each becomes an all-in job cost: units × unit cost, plus billed setup per launch (install, download) times the
number of launches.

**What the banked numbers already say (estimates).**

| quantity | AWS L40S | Vast | source |
|---|---|---|---|
| harvest tokens per second | 3,215–3,334 | 4,551–4,623 (RTX PRO 5000 Blackwell) | `00_ASSETS.md` lines 22–39 |
| price per GPU-hour | $2.24 | $0.785 realized over the W3 grid ($22.68 for 28.9 GPU-hours) | `00_ASSETS.md` line 222; `SPEC_D1.md` lines 707–708 |
| R, per 2,048-token row | $0.00038 | $0.00010, about 3.9 times cheaper | arithmetic on the rows above |
| M1, 5–6 harvests | $18.82–$25.54 | about $5–6 | `M1_ASSETS.md` lines 258–261 and the rows above |
| generation tokens per second | 800, assumed, not measured | not measured | `N13_SPEC.md` line 989 |
| G, per 1,000 tokens | $0.00078 ($0.78 per million) at the assumed 800 | — | arithmetic |
| Vast quotes for a larger GPU | — | RTX PRO 6000 WS $2.01; H100 NVL $2.67; H100 SXM $4.54–6.18 per hour | `SPEC_D1.md` line 720; scout line 10 |
| speed needed to be 1.5 times cheaper per token | — | RTX PRO 6000 at $2.01: 1.35 times the L40S rate; H100 NVL at $2.67: 1.79 times; H100 SXM at $4.54: 3.04 times | arithmetic |

The 2026-09-05 scout assumed an H100 runs 2.5–3 times an L40S on lens fits (scout line 14). No generation
rate for this model has been measured on any lane. That is what J4 measures.

**The move rule.** A job moves from AWS L40S to a Vast lane only if all five hold.

1. Its all-in cost on Vast is at most 1/1.5 of its all-in cost on AWS: Vast is at least 1.5 times cheaper.
2. J4's reliability check passed on that GPU model.
3. The Vast ceiling minus spend covers 1.3 times the job's projection. Today the room is $42.00 − $35.69 =
   $6.31 (`infra/vast/operations/2026-11-03_lane_ceiling_closed.md` line 3). The ceiling rises only on the
   user's top-up (`infra/vast/budget.json` → `_why`). M1 on Vast (about $5–6, so $6.5–7.8 at 1.3 times) needs
   a small raise; N13 on Vast needs a larger one.
4. The job's offline gate is green inside the Vast image, with the same pins.
5. Every run inside one frozen comparison stays on one lane and one GPU model. For N13 that means R0, S3 and
   S4 on one GPU model: σ is measured from seed pairs on that hardware, and a fixed sampling seed can give
   different tokens on different kernels. For M1, all harvests on one GPU model, or a cross-device pair added
   to the floor (the T_14 device check found 0 of 640 poles changed presence, `00_ASSETS.md` line 50).

If any condition fails, or the numbers are within 1.5 times of each other, the job runs on AWS L40S, as the
user said. The lane is chosen once per study, before that study's first GPU run.

**The reliability check** (one Vast rental, inside J4):

1. From rent to `running` within 30 minutes (`load_backstop_min` 30; loading is not billed).
2. Download of one 4.97 GB shard at 100 MB/s or more, and an S3 upload of 1 GB.
3. `job_live.log` reaches S3 every 60 s with no gap over 4 minutes (`infra/vast/ARCHITECTURE.md`, signal table).
4. `_rental_complete.json` written, the instance destroyed itself, and billed hours in the spend record within
   10% of uptime.
5. Ten greedy outputs are readable text and use the same prompt token ids as on AWS.

One host failure: the host goes into `infra/vast/HOSTS.json` → `known_bad`, and one other host may be tried
inside J4's 0.5 GPU-hour stop. Two failures: no Vast for this program's main runs.

**Known failure modes.**

| lane | failure | record |
|---|---|---|
| Vast | a slow-network host before the center probe (not billed) | `studies/already-clean/SPEC_D1.md` line 720 |
| Vast | a killed H100 respawn before the center probe (not billed) | same line |
| Vast | a 14.5-hour silent loading wedge, 2 × RTX PRO 6000 WS, Taiwan | `infra/vast/HOSTS.json` → `known_bad` |
| Vast | a 25-minute pull with no progress, 2 × RTX PRO 6000 S, California | same file |
| Vast | 17 MB/s from S3 on one host; Hub reads timed out; one network-probe abort | `infra/aws_pg/operations/2026-09-05_vast_lane_scout.md` lines 13, 22 |
| AWS | no g6e.2xlarge or g6e.4xlarge capacity in any zone, 14:34Z to 15:11Z | `infra/aws_pg/operations/2026-09-05_w6_capacity_and_stall.md` lines 6–10 |
| AWS | heartbeat and output sync stopped after the model loaded; about 0.6 hours spent with no output | same file, "Stall" |
| AWS | a reaper schedule left after self-termination | `infra/aws_pg/operations/2026-11-01_reuse_primary_orphan_reaper.md` |
| AWS | the 32-vCPU quota holds four g6e.2xlarge or two g6e.4xlarge | `infra/aws_pg/config.py` lines 28–33 |

## 5. Concurrency plan

**Dependencies.**

| job | needs before it starts | blocks |
|---|---|---|
| J1 | — | J9, J11, J12 |
| J2 | — | J5 |
| J3 | — (its code part needs its spec frozen) | J7 |
| J4 | Vast preflight stamp; Vast room of at least $1 | the lane of J5, J6, J7 |
| J5 | J2 green and tagged; J4's lane | J6 |
| J6 | J5 passed; tag `n13-freeze-2` | J13, J14 |
| J7 | J3 tagged; J4's lane | J12 (identity pair), J13, J14 |
| J8 | its rule frozen and shown to have room | — |
| J9 | J1 with at least 100 passing births; its spec | J12 |
| J10 | its spec | J14 |
| J11 | J1's ages; its spec | J14 |
| J12 | J1, J9, the unite Arm 1 review, J7's pair | J14 |
| J13 | J6 and J7 main results; the user | — |
| J14 | J6 and J7 closed; J9–J12 closed or set aside | — |

**Phases.** Agent counts include running agents; the cap is four, the normal load three.

| phase | starts when | jobs | GPU instances | Opus agents |
|---|---|---|---|---|
| 0 | now | J1 recount, J2, this plan | 0 | 3 |
| 1 | this plan reviewed | J1 retag, J2, J3 spec, J4 with J8 | AWS 2 (J4, J8), Vast 1 (J4) | 4 at the peak, 3 once J1 closes |
| 2 | J2 tagged; J4 decided | J5; J3 code and offline gate; J8 closing | 1 (J5) | 3 |
| 3 | J5 passed and tag `n13-freeze-2`; J3 tagged | J6 (S3, then S4 on 2 instances); J7 (2 instances) | 4, the quota; N13 has first claim on capacity | 3 (J6, J7, J9 spec writer) |
| 4 | J6 and J7 banks complete | the two independent recounts; J9 critic; J10 spec | 0–1 | 3, at most 4 |
| 5 | J6 and J7 closed | J9, J10, J11, J12 in that order, gated as above | up to 4 | at most 3 |
| 6 | the results of phase 5 | J13 decision (user); J14 | 0 | 1 reader |

**How the head polls.**

- Liveness is an instance ID from `./infra/aws_pg/pg ps` or the Vast spend record, plus the `updated_utc`
  field of the job's `metrics.json` on S3. N13 rewrites that file every 25 answers and at least every 60 s
  (`N13_SPEC.md` line 359).
- Fable polls `metrics.json` on S3 with a Monitor until-loop about every 10 minutes. Never `job.log`.
- A `metrics.json` older than 10 minutes while the instance runs: `pg liveness`. Older than 30 minutes: kill
  and relaunch under a new attempt id, with the old prefix kept.
- GPU-hours used are instance launch to termination, from EC2 `describe-instances` or the Vast spend record,
  written per launch in the operations note.
- An agent's report is not status. Status is an artifact with its timestamp. No job is scored while it runs.

**Delegation rules, for cost.** Agent tokens are estimated at $8–15 per agent (head's estimate), about the
size of the GPU spend, so agents are used where their context pays for itself.

- One agent per study carries spec, code and runs in sequence in its own worktree, handed over by
  `STATE.md`. A fresh agent is used only for an independent read or recount.
- A brief names exact files and line ranges and the one output path. An agent does not re-read the brainstorm.
- An agent does not wait inside its own context for a GPU job. It launches detached (`pg run --detach`) and
  ends its turn. Fable polls and sends it a message when the artifact lands.
- A check of a few tool calls (read one `metrics.json`, apply a frozen rule) is done by Fable, not delegated.
- Each agent's measured token cost is written into §7 after it ends.

## 6. The researcher's checklist

| check | where it bites | the file that shows it was done |
|---|---|---|
| Look at ten raw inputs and outputs before any statistic | every job (the last row of each card) | the study's `JOURNAL.md` line naming the sample file read, for example `studies/next/n13/JOURNAL.md` after S2 |
| Does the model do the task at all before we explain how? | J5 (base format rate, smoke), J10 (accuracy gate), J13 (swapped instructions) | `studies/next/n13/JOURNAL.md`; the N22 gate file; the robot gate verdict |
| Reproduce one published number under the published harness first | J5 Minerva MATH 40.0 ± 1.6; J7 corpus loss 2.675; J8 64 banked rows; J13 194/200 | `S3/results/next_n13/s2setup-01/out/r0_known/metrics.json`; `studies/next/m1/JOURNAL.md`; `pilot0/GPU_CHECK2.md` |
| The strongest baseline, not the easiest | J6 (DARE and tail-energy-matched random, never the base alone, spec §7); J7 (the 40% mix and the identity floor); J9 (the pole's own current percentile); J11 (energy-matched pairs) | `verdict_n13.json` comparisons; `studies/next/m1/SPEC.md`; the M2 and N6 specs |
| Is the effect larger than two runs of the same thing? | J6 (σ from seed pairs, G-ID); J7 (identity pair); J8 (forward spread F) | `thresholds.json`; `verdict_m1.json`; `GPU_CHECK2.md` |
| Does the control have the power to fail? | J6 G-GATE room; J7 ordering room; J8 rule run on banked files | `thresholds.json` → `room`; `studies/next/m1/SPEC.md` room section; `GPU_CHECK2.md` |
| Does the kill condition have room to fire? | J6 (`N13_SPEC.md` §6.4); J9 (chance 0.0009); J10 (interchange control) | the room tables of each spec |
| The boring explanation first | J5 and J6: chat template (G-TEMPLATE), tokenization (prompt ids identical in every arm), extraction regex (G-GRADER), seed (sets A and B), length cap and termination (cap-hit share); J7: dtype (fp32 shards cast to bf16), stale shards (load gate), document overlap; J8: pad tokens, off-by-one (neighbour batch); J1: selection on cluster size and rank | the gate values in `verdict_n13.json`; `verdict_m1.json` load gate; `GPU_CHECK2.md`; `10_RECOUNT2.md` |
| Perturb the measurement slightly | J6 (seed set B, second random draws at 1%); J7 (category with and without the 8 `common_crawl` directories); J8 (whole-row against pad-target exclusion, 0.0004 apart before); J1 (two independent sets of 300 draws) | the verdict files |
| Write the results table with blank cells before running | J6 (`N13_SPEC.md` §11); J7 and J9–J12 specs; J8 rule | the spec at its freeze tag |
| Predict the number before unblinding and record it | J6 (§1.5 priors copied into the journal before S3); J7 (in its spec); J8 (expected offset) | the journals, before the verdict line |
| Two independent implementations of the main number | J1 (critic, `06b`, recount2); J6 and J7 (a fresh agent's own recount) | `10_RECOUNT2.md`; `READ_n13.md`; `READ_m1.md` |
| Plot everything, including the failing inputs | J6 (arms with screen accuracy drawn, failing-input values in the verdict); J1 (births with the random-group reference); J7 (shifts with the floor) | `figures/fig_n13.png`; `walkthrough/births/`; `studies/next/m1/figures/` |
| State units and the object on both sides | J6 (percent of the same 500 problems, same prompt and grader, §2.1); J7 (share of 50 positions, same corpus, mix against mix); J8 (nats per token, pad excluded, same batch) | the spec's measurement section |
| Bank the prompt, prediction and probability beside every score | J6 (`samples.jsonl` fields, §10.1); J7 (decoded contexts with activation values); J8 (row samples) | the samples files beside each `metrics.json` |

## 7. Budget table

GPU dollars at $2.24 per hour. "Est." marks an estimate. The agent column uses the head's estimate of $8–15
per agent of about 100 tool calls; the one measured point (98 tool calls, 184,801 billed tokens) is noted in
Appendix A.

| job | GPU-hours | GPU $ | API $ | Opus agents | agent $ (est.) | running GPU-hours | running GPU $ |
|---|---|---|---|---|---|---|---|
| J0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| J1 | 0 | 0 | 0 | 1 | 8–15 | 0 | 0 |
| J2 | 0 | 0 | 0 | 1 | 8–15 | 0 | 0 |
| J3 | 0 | 0 | 0 | 2 | 16–30 | 0 | 0 |
| J4 | 0.3 (est. 0.3–0.4; stop 0.5) | 0.67 (AWS about 0.45, Vast about 0.30–0.70) | 0 | 1, shared with J8 | 8–15 | 0.3 | 0.67 |
| J5 | 3.8–5.0 | 8.51–11.20 | 0 | 1, the J5–J6 executor | 8–15 | 4.1–5.3 | 9.18–11.87 |
| J6 | 21.7–32.7; J5 + J6 ≤ 35 | 48.61–67.20 | 0 | 1, the recount reader | 8–15 | 25.8–35.3 | 57.79–79.07 |
| J7 | 8.4–11.4 (ceiling 12, proposed) | 18.82–25.54 | 0 | 2 | 16–30 | 34.2–46.7 | 76.61–104.61 |
| J8 | 0.3 registered (est. 0.45–0.8; ceiling 1.0) | 0.67 (est. 1.01–1.79) | 0 | 0 | 0 | 34.5–47.0 | 77.28–105.28 |
| this plan | 0 | 0 | 0 | 1 | 8–15 | | |
| **through J8** | **34.5–47.0** | **77.28–105.28** | **0** | **10** | **80–150** | | |
| J9 | 5–8 (12 with gradients) | 11.20–26.88 | 0 | 2 | 16–30 | 39.5–59.0 | 88.48–132.16 |
| J10 | 8–13 | 17.92–29.12 | 0 | 2 | 16–30 | 47.5–72.0 | 106.40–161.28 |
| J11 | 4–8 | 8.96–17.92 | 0 | 2 | 16–30 | 51.5–80.0 | 115.36–179.20 |
| J12 | about 8; identity pair from J7 | 17.92 | 0 | 2 | 16–30 | 59.5–88.0 | 133.28–197.12 |
| J13 (deferred; not in the totals) | 4.6 | 10.30 | 0 | 1 | 8–15 | — | — |
| J14 | 0 | 0 | 0 | 1 reader | 8–15 | 59.5–88.0 | 133.28–197.12 |
| **program, without J13** | **59.5–88.0** | **133.28–197.12** | **0** | **19** | **152–285** | | |

Ceilings. N13: 35 GPU-hours (registered). M1: 12 GPU-hours (proposed). J4: 0.5. J8: 1.0. J9 to J12: set in
each spec. Vast: $42.00, $6.31 of room. AWS: user to confirm. If every AWS hour fell back to g6e.4xlarge at
$3.00424, the upper bound through J8 would be $141.20.

API. No job calls a paid model; no job uses a judge. If a judge is ever added, five steps come before any
paid line: count the calls from the frozen item list; price 20 calls from measured tokens per call; set the
provider's spend limit at 1.2 times the projection; run the 20 and read every answer; then the rest. The
measured rates to start from: $0.053 per clusterer request and $0.0162 per judge line on Sonnet at list price
(`studies/already-clean/SPEC_D1.md` line 713). Which model may answer is the user's call: the Sonnet exception
in the user's rules covers only the already-clean W4 requests.

## 8. Risk register

| # | way the program wastes money or reports a wrong number | the pre-check that catches it | job |
|---|---|---|---|
| 1 | The N13 harness is off (template, extraction, engine), so every arm's number is wrong | G-KNOWN 40.0 ± 1.6 before any arm; G-TEMPLATE; the G-GRADER planted set; S-BAND | J2, J5 |
| 2 | The RL gain is under the noise, and S4's 17.6–26.5 GPU-hours buy nothing | G-GATE with room before freeze 2; on failure, no S4 | J6 |
| 3 | Throughput or answer length is worse than assumed (800 tokens per second, 2,500-token answers), and N13 crosses 35 GPU-hours | J4's measured rate; J5's measured lengths; the projection redone before S3; the stop rule parks the last arms | J4, J5, J6 |
| 4 | A simpler cause is read as a neuron effect: stopping in time, the `Answer:` format, change energy, or bf16 rounding | cap-hit share, format rate and accuracy among non-capped answers per arm; the DARE, energy-matched and tail-energy-matched arms; PC-ART, J_floor, J_round | J5, J6 |
| 5 | M1 loads the stale `main` shards or downloads 43.8 GB per mix end | the shard filter; the load gate with the stale shards as its failing input | J3, J7 |
| 6 | An M1 shift comes from exposure to the same documents (Stack-Edu, FineMath) and is reported as a data-mix effect on neurons | the exact-text hash check of corpus documents against the training pools before unblinding (`M1_ASSETS.md` confound 3) | J3, J7 |
| 7 | J8's frozen rule has no room (a 0.003 floor below the known +0.0055 offset) and returns "undecided" again | the rule is run on the banked first-check files before launch | J8 |
| 8 | A launch runs without output (Vast loading wedge, slow network, AWS heartbeat stall), or one comparison mixes GPU models | the reliability check; one GPU model per frozen comparison; `max_billed_h` and `--budget-min`; the 10- and 30-minute polling rules | J4, J6, J7 |
| 9 | The birth unit is inflated (all 1,238 late-born clusters instead of those that pass), so J9 and J12 measure cluster size and rank | J1's per-birth p against random same-size groups; the gate of at least 100 passing births | J1, J9, J12 |
| 10 | Agent tokens: agents re-read large files, poll running jobs, or overlap | the delegation rules of §5; the cap of four; the measured cost of each agent recorded | all |

## 9. Communication rules for reports to the user

- **The first sentence is the answer, with the one number that settles it**, its interval and its control.
  Template: "<study>: <measure> on <model> was <value> [<interval>] against <control> <value>; the frozen rule
  says <verdict>." Then stop or add one sentence of reason.
- **Plain technical English.** Short sentences, one idea each. Name the measure: MATH-500 accuracy in percent
  of 500 problems, Minerva MATH `primary_score`, corpus loss in nats per token, LIBERO success rate. Never
  "performance" or "damage". Absolute numbers, not fractions of a gap.
- **Terms that may be used**, each defined in plain words the first time in a report: pole, neuron, cluster,
  birth point, checkpoint, harvest, residual stream, lens, attribution, transplant; also MLP, layer, arm, seed,
  gate, GPU-hour. Expand every acronym the first time.
- **Words that may not be used:** the three words banned by the user's rules and by `studies/next/README.md`
  ("House rules"). They are not written here either. Their replacements: cancelled, or does not count; record;
  chance level, or no result. No analogies, no metaphors, no names coined in this program. In a report, "job"
  means a work item; a group of a pole's top contexts is a cluster.
- **A status report contains**, and nothing is reported without it:
  1. for each running piece, a live instance ID or an artifact timestamp in UTC;
  2. GPU-hours used against the ceiling, per study and in total, with dollars;
  3. the next gate and the number that passes it;
  4. for N13, the run, unrun and parked counts over the roster (`N13_SPEC.md` §9.4);
  5. anything that failed, said plainly in one sentence.
- **Never** status from an agent's say-so, a score of a running job, or a number without its artifact path.
- **Teaching**: one idea, one check question, then wait.

## 10. The end deliverable

`studies/next/20_ABSTRACT.md` (proposed path): an abstract of at most 300 words, then an evidence table with
one row per sentence. Fable writes it; one fresh Opus reader checks every row against its artifact before it
is shown to the user.

| part | content | evidence the sentence needs before it may be written |
|---|---|---|
| Background, 1–2 sentences | the question and the one prior result it builds on | the prior paper read in full for that claim, cited |
| What we did, 2–3 sentences | models and checkpoints with commits; the measurements; the controls | the spec at its freeze tag; the manifests |
| What we found, 3–5 sentences | each finding as an absolute number with its 95% interval and its strongest control's number in the same sentence; the frozen rule's verdict; cancelled gates and record corrections (the D1 retag) included | the artifact path and field; a claim ID with status CONFIRMED in the study's `CLAIMS.md`; the three numbers stamped (fit on both sides, a positive control that fired, the identity-pair value); the independent recount matched; one right and one wrong sample banked; every figure regenerated after the last correction |
| What it means, 1–2 sentences | the reading the result supports and the reading it rules out, nothing beyond the data | the refutation table of the spec, with which row fired |
| Novelty and practical value, 1 sentence each | not done before; a measured win against a named baseline | the literature search with its query phrases recorded and the closest paper read in full; the baseline's number in the engineers' measure |
| Highest-leverage next step, 1–2 sentences | the next study, its GPU-hour cost, its first gate and the number that would stop it | a costed first gate with a registered failing input; the leverage reason stated against the scoring rubric of `05_SCORING.md` |

## Appendix A. Inconsistencies found in the inputs

1. N13 cost. `08_SUMMARY.md` line 10 says about 20–25 GPU-hours and line 47 says 17–24. The amended spec
   projects 25.5–37.7 (`N13_SPEC.md` lines 31 and 1004), and its own target, 20–25 (line 1013), lies below
   its projection.
2. J5. The brief gives 1–2 GPU-hours. The spec's S2 is 3.8–5.0 (lines 925, 1001), and R0 alone is 2.0–3.0
   (line 1006).
3. J8. "About 0.3 GPU-hours" is below the measured rate: one 512-row batch took 645 s with sdpa
   (`pilot0/GPU_CHECK.md` line 122), so two batches alone are 0.36 GPU-hours. The first check's frozen floor,
   0.003 (line 184), is below its measured offset, +0.0055 (lines 5–6).
4. Vast spend. `infra/vast/budget.json` → `_why` says $33.29 spent. The operations note of 2026-11-03 (line 3)
   says $35.69 after the SPEC4 rental ($2.40), with $27.00 of credit. The room is $6.31, not $8.71.
5. M1. Only an asset check exists. `M1_ASSETS.md` lines 31–42 say the written refutation has almost no room;
   `08_SUMMARY.md` line 16 reports the assets as verified, which is true, but there is no pre-registration.
6. Lane. `N13_SPEC.md` lines 830–835 fix N13 to g6e.2xlarge. A Vast run needs a dated amendment and S3 paths
   under `results/mdl/*` (`infra/vast/cells/pipeline_job.py` line 11).
7. Agent cost. The one measured point, 184,801 billed tokens for 98 tool calls, costs at most $3.70 even at
   the $20-per-million output rate. The $8–15 estimate needs its basis (for example, whether cache reads are
   counted).
8. Worktrees. `git worktree list` shows `../interp_research-m1` (branch `m1`, `ed3e72006`). No line of
   `studies/next/JOURNAL.md` records its creation.
9. Launcher. `./infra/aws_pg/pg --help` did not run: the `clean-neurons` worktree has no `.pgvenv`
   (`infra/aws_pg/pg` line 5).

## Appendix B. Open questions for Fable

1. The Vast ceiling. Raise it, by the user's top-up, if J4 favors Vast for M1 (about $5–6) or for N13?
2. The AWS ceiling: user to confirm.
3. With Opus-only workers, is Fable's own read the "model other than the executor" that `N13_SPEC.md` §9.4 S5
   requires, with a fresh Opus agent writing the second implementation?
4. J8's rule: adopt the proposal in J8's card (0.0124 per step plus the neighbour condition), or another rule
   with room?
5. M1: the five-harvest plan and a ceiling of 12 GPU-hours?
6. The user's global rules still say at most two agents at once; `studies/next/JOURNAL.md` (00:20 CST) records
   the user lifting that cap for this study, with four kept as the limit. Is that lift still in force?

## Head's decisions (Fable, 2026-11-05 09:40 CST)

Answers to Appendix B, in order. These are decisions, not proposals.

1. **Vast ceiling.** The lane decision is split by study. N13 stays on AWS L40S (its spec fixes the lane at lines 830–835, and the Vast room of $6.31 cannot cover 1.3 times its projection). M1 is the study where Vast pays: the realized Vast price over the W3 grid was $0.785 per GPU-hour on an RTX PRO 5000 class card against $2.24 for the L40S, about 3.9 times cheaper per harvest row (§4 table). M1's first GPU step (the known number: one banked grid point re-harvested) doubles as J4's reliability check on Vast. The user is asked for a Vast top-up of about $20; until it lands, M1 waits or runs on AWS at $19–26 if the user prefers speed. J4 as a separate generation-throughput job is dropped: the AWS generation rate comes from N13's own smoke (S2), which runs anyway.
2. **AWS ceiling.** The user approved the N13 and M1 runs with their registered ceilings; the dollar total through J8 at $2.24 per hour is $77–105 and is reported to the user as such. No separate AWS dollar ceiling is set beyond the GPU-hour ceilings.
3. **Different-model read.** Fable's read of the artifacts is the "model other than the executor" read. A fresh Opus agent, blind to the executor's code, writes the second implementation of each headline number.
4. **J8 rule, frozen now.** Pad targets excluded. Two steps: `stage1-step61000` (batch 61,000, logged 2.1173441410064697 per `00b_DATA_ORDER.md` line 124) and one more step chosen and written down before launch. Per step: pass if |S − W| ≤ 0.0124 nats (half the per-step standard deviation 0.0248 of the logged loss) and the neighbour batch (the previous step's batch) is farther from the same logged value than the target batch. Fail if either step misses. Does not count if the neighbour batch is also within 0.0124 (the test has no power there) or if fewer than 400 of the 512 rows are scored. Ceiling 1.0 GPU-hours. Verdict read by Fable from the JSON files.
5. **M1.** Five harvests: the shared start, the three mixes, and one identity pair (the same weights harvested twice) as the floor. Ceiling 12 GPU-hours. All harvests on one GPU model. The written refutation must be shown to have room on control data before the freeze (`M1_ASSETS.md` lines 31–42).
6. **Agent cap.** The user lifted the two-agent cap for this program; four at once is the limit, three the normal load (five caused a kernel panic on 2026-10-13).

Also decided: the `08_SUMMARY.md` cost lines for N13 (Appendix A item 1) are corrected to the spec's 25.5–37.7 GPU-hours once the retag agent has finished its one-line edit of that file (one writer per file). Agent cost: the $8–15 estimate counts input and cache reads over about 100 turns, not only the 184,801 billed output tokens; it stays an estimate until a per-turn usage record exists.
