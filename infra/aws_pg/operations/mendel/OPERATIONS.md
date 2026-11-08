# MENDEL — infrastructure operations log

Lifecycle, packaging, durability, spend. Kept OUT of the science journal per CLAUDE.md's
science/infrastructure separation. Readiness is not scientific progress; this file records only
what licenses a scientific test and where the evidence lives.

## Standing configuration

| item | value |
|---|---|
| launcher | `./infra/aws_pg/pg` (worktree `../ntm-frozen-gemma-mendel`, branch `science/mendel`) |
| AWS | profile `fuelos`, region `us-east-1`, explicit on every call |
| GPU type | `g6e.2xlarge` (L40S 48 GB) — house default; never p5/p4de |
| CPU type | `r7i.2xlarge --no-weights` |
| bucket | `s3://pg-lineage-769063704696` |
| poll surface | `out/metrics.json` ONLY — never `job.log` (multi-MB sync spam) |
| prefix rule | fresh S3 prefix per attempt; a re-used prefix serves the previous run's STATUS |
| pre-launch gate | `bash scripts/mendel_check.sh` must exit 0 |

## Control-plane venv

`.pgvenv` created in the worktree (boto3 1.43.74 + pyflakes). Not committed (`.gitignore:47`).
The `pg` wrapper resolves it from the worktree root.

## Ledger

`scripts/mendel_inventory.py` → `studies/lineage/paligemma1-pi0/research/mendel_ledger.json`.
49/49 surfaces verified: code surfaces by IMPORT (a grep hit proves a string is in a file, not
that the symbol resolves), S3 surfaces by list+count. Two thresholds were wrong on first pass and
were corrected against the bucket: `humboldt/banks` = 900 files (126 `core32`, 7 types × 18
layers); `shackleton_census_g2b/out/banks` = 36 files (`q_proj` ONLY × 18 × {core32, anti32}).

## Launch log

| utc | arm | prefix | type | instance | budget | outcome |
|---|---|---|---|---|---|---|
| 2026-08-19T17:10:53Z | G4 | `results/mendel_g4_v1` | g6e.2xlarge | — (never launched) | 150 min | FAILED at launch — `InsufficientInstanceCapacity` in every AZ. No instance, no S3 write, $0. Retried; 4 attempts at ~3 min spacing, capacity returned on the 4th. |
| 2026-08-19T17:22:38Z | G4 | `results/mendel_g4_v1` | g6e.2xlarge | `i-04b05ffd3226e25d2` | 150 min | OK — `STATUS exit=0`, 18.1 min wall (17:22:38Z→17:40:46Z), self-terminated, fleet empty. `verdict=REPRODUCED`, all 3 gates ok. |
| 2026-08-19T17:49:31Z | MN-1 stage 1 (backbone hops) | `results/mendel_mn1_backbone_v1` | r7i.2xlarge | `i-042a641b08a207bb4` | 180 min | OK — `pg run` exit 0, `STATUS exit=0`, 35.0 min wall (17:49:31Z→18:24:31Z), self-terminated, instance `terminated`. `verdict=INHERITED-BEYOND-TRAFFIC`; single gate `mn1_instrument_stable` ok, no void hops. Gated+stamped+shipped sha all `e2c92abe` (HEAD moved to `5d2c219f` under a concurrent session AFTER launch — stamp-per-attempt held). |
| 2026-08-19T17:59:06Z | MN-4 | `results/mendel_mn4_v1` | r7i.2xlarge | `i-008bbd97e138ab314` | 60 min | OK — `STATUS exit=0`, 91 s wall (17:59:10Z→18:00:41Z), self-terminated. `verdict=NO-REACH`, 126 cells, gate `mn4_map_not_mostly_censored` ok. Launch was HELD ~5 min on a dirty tree (see below); gate re-run on the shipped tree. |
| 2026-08-19T18:30:09Z | FT-family | `results/mendel_ftfam_v1` | r7i.2xlarge | `i-0ff8bceb9d328e4b3` | 240 min | OK — `pg run` exit 0, `STATUS exit=0`, 133.5 min wall (18:30:09Z→20:43:36Z), self-terminated, instance `terminated`. All 24/24 fine-tunes processed, 2592 cells. `verdict=AMBIGUOUS`; gates `ftfam_instrument_detects_a_planted_colocation` ok, `mn2_sink_outside_energy_band` ok, `mn2_beats_the_total_write_incumbent` **not ok**. Gated+stamped+shipped sha all `0f15effa`, `dirty=false`. Clean tree at launch; 1 other r7i in flight (8/16 Standard vCPU free). No disk-full, no AssertionError, no HF 401/403. |
| 2026-08-19T18:31:12Z | G4b | `results/mendel_g4b_v1` | g6e.2xlarge | `i-090cd7ee67605f13e` | 90 min | OK — `STATUS exit=0`, 18.4 min wall (18:31:12Z→18:49:38Z), self-terminated, 0 reaper-less. Capacity on the FIRST attempt (`tried: ["g6e.2xlarge"]`, no retry). `verdict=BANKED`, 108 sites, `top_k=512`, 216 `act_gram` files in S3 (108 evecs + 108 evals). Launch was HELD ~5 min on a dirty tree (see below); gate + stamp re-run on the shipped HEAD `0f15effa`, `dirty=false`, gated sha == stamped sha == launched sha. |
| 2026-08-19T18:16:29Z | MN-6 | `results/mendel_mn6_v1` | r7i.2xlarge | `i-0c4f0253a4b4f32ea` | 120 min | OK — `STATUS exit=0`, 27.3 min wall (18:16:29Z→18:43:54Z, elapsed 1640 s), self-terminated. Gate exit 0 and stamp both at `9df8ae05`, `dirty=false`; gated sha == launched sha. `verdict=LORA-RESPECTS-CORE`, 40 adapters used / 1 skipped, 4320 cells, both gates ok. HF census hit 256 repos over 145 owners, no 401/403. |
| 2026-08-19T18:48:04Z | MN-8 | `results/mendel_mn8_v1` | r7i.2xlarge | `i-04ad18a32fd37bce0` | 90 min | OK — `pg run` exit 0, `STATUS exit=0`, 91 s wall (18:48:08Z→18:49:39Z), self-terminated, instance `terminated`, fleet clear of MN-8. Gate exit 0 and stamp both at `b9a8bd30`, `dirty=false`; gated sha == launched sha == metrics `provenance.commit_sha`. `verdict=AMBIGUOUS`, 126 cells, 5/6 statistics present (`topsigma_leverage` absent), `mn8_statistics_share_one_ordering` ok=false, `mn8_l0_dead_entries_are_layer_unique` ok=true. Neither watched AssertionError fired. |
| 2026-08-19T19:05:30Z | MN-3 | `results/mendel_mn3_v1` | g6e.2xlarge | `i-02f8196dc5758a676` | 180 min | OK — `STATUS exit=0`, 57.6 min wall (19:13:03Z→20:10:39Z), self-terminated, 0 reaper-less, acked. Attempt 1 FAILED `InsufficientInstanceCapacity` in every AZ; capacity returned on attempt 2 (~4 min later). Gate exit 0 at `96ae863b`; HEAD moved to `6b1ea164` under a concurrent session during the retry window, and the per-attempt re-stamp shipped `6b1ea164`, `dirty=false` — `git diff 96ae863b 6b1ea164` touched ONLY MN-17 files (new job + tests + journal + artifacts) and `mendel_arms.tsv`, leaving every MN-3 code path byte-identical, so the gate covered the shipped tree. `verdict=NO-TRANSFER`, 8 units, 18480 rows, 96 cells, all 3 `mn3_readout_valid_L*` ok=true, `mn3_core_beats_highway` ok=false. No watched AssertionError, no OOM. `pg run` exit code NOT observed — the laptop-side watcher process was killed by a harness timeout at ~20:11Z, seconds before STATUS landed; `pg watchdog` reports COMPLETE (exit=0). |
| 2026-08-19T18:56:39Z | MN-2 / FT-family, bounded `--n-ft 10` | `results/mendel_ftfam_v2` | r7i.2xlarge | `i-00cc0a4d5458cefc2` | 180 min | OK — `pg run --detach --json` exit 0, capacity on the FIRST attempt (`tried: ["r7i.2xlarge"]`), `STATUS exit=0`, 62.2 min wall (18:56:43Z→19:58:52Z), self-terminated, 0 reaper-less, acked. Gate exit 0 and stamp both at `f41627e4`, `dirty=false`; gated sha == stamped sha == launched sha == metrics `provenance.commit_sha`. `verdict=SINK-COUPLED`, 10/10 fine-tunes, 1080 cells, all 3 gates ok, `partial.json status=COMPLETE n_done=10/10`. ~6.2 min per checkpoint (vs the ~11 min observed on the sibling). No disk-full, no decoder AssertionError, no HF 401/403. Deliberately bounded companion to the 24-FT `results/mendel_ftfam_v1` run (`i-0ff8bceb9d328e4b3`), left untouched. |

## Watching a job longer than the agent's Bash cap (observed 2026-08-19, FT-family)

The agent Bash tool kills any single call — foreground or `run_in_background` — at its timeout
ceiling (600 s max). `pg watch` on a 133 min job therefore CANNOT run to completion in one call:
it gets killed mid-poll and the job silently loses its watcher while still running. This happened
twice on `mendel_ftfam_v1`. The instance is unaffected (the EventBridge reaper is the real
guarantee), but the agent goes blind.

Working pattern for a >10 min job: re-issue `pg watch <prefix>` back-to-back — it re-resolves the
instance from `tag:s3prefix` every time, so re-attaching is free and stateless — and in parallel
keep a long-lived poll that checks BOTH instance state and `STATUS`, breaking on either. Per-unit
progress without touching `job.log` is available by counting distinct unit names in
`out/vectors/` (`colenergy__<unit>__<type>__L<n>.npy`).

## Defects found in shared infrastructure

**Cross-program stamp defect (found 2026-08-19, fixed in `scripts/mendel_stamp.py` only).**
`_git()` returned `stdout.strip()`, and `git status --porcelain` prefixes an unstaged
modification with a leading space (`" M path"`). Stripping the whole blob eats that space on the
FIRST line only, so the `ln[3:]` slice truncates exactly one filename by one character
(`"nfra/pg/..."`). Reproducible. Cosmetic — it does not affect `commit_sha` or the `dirty` flag —
but a provenance record that misspells a file is worse than useless.

`scripts/{darwin,copernicus,humboldt,mercator}_stamp.py` all carry it. NOT fixed here: they belong
to other programs' branches and editing them from this worktree is out of lane. Raise on their
next touch.

## Provenance guard

`mendel_core.provenance()` refuses a stamp whose `program` key is not `"mendel"`. Without it, a
job launched without stamping would read whichever sibling provenance file
`humboldt_core.provenance()` finds first and label every atlas row with HUMBOLDT's sha
(`b73c2a79`, branch `science/humboldt`, 2026-08-12) — well-formed rows pointing at the wrong code.
`infra/pg/mendel_provenance.json` is gitignored: committing it would change the tree the stamp
certifies.

**The worktree is shared and HEAD moves under a launch (observed 2026-08-19, G4).** A concurrent
session committed to `science/mendel` during the ~11 min capacity-retry window, so the gate
(`mendel_check.sh`, exit 0) ran at `5bbfb2cd` while the job shipped `f8bbed21` — the stamp is
re-taken per attempt, so it recorded the launched tree correctly and the run is honestly labelled.
It was gate-covered only by luck: `git diff 5bbfb2cd f8bbed21` touched MN-1 transport, its tests,
and a journal doc, leaving `mendel_g4_preflight.py` and `mendel_core.py` byte-identical. A retry
loop must therefore re-stamp before EVERY attempt (it does) and the launched sha must be diffed
against the gated sha before the result is trusted. Do not assume the pre-launch gate covers the
tree that actually shipped.

**Same race, caught the other way (2026-08-19, MN-4).** At 17:54Z the tree was dirty with exactly
two untracked files — `infra/aws_pg/jobs/mendel_mn4_tstar.py` and `tests/test_mendel_mn4_job.py`,
written 51 s and 85 s *after* the HEAD commit `24d651d8` ("MENDEL MN-4"). The science had been
committed and the transport had not. **Launching then would have shipped them anyway**:
`pg.package_code()` tars the *working tree*'s `infra/` dir, not `git archive`, so untracked files
ride along — the run would have produced correct numbers under a `dirty=true` stamp naming no
commit that contains the code that produced them. The launch was held instead; the sibling session
committed `b4146f5e` at 17:58:40Z (126 tests, both files in), the gate was re-run on the new HEAD,
and the stamp went out `dirty=false` with gated sha == launched sha. Rule: on a dirty tree, hold
and re-gate — never lean on the fact that the tarball would have carried the file.

**Third instance, held and cleared (2026-08-19, G4b).** The gate passed at `5d2c219f` but the tree
carried 4 untracked MN-1 result artifacts under `operations/mendel/mn1/`. Held on the 60 s re-check
loop; over the next two minutes the tree got *dirtier*, not cleaner — a sibling session added an
untracked **code** file (`infra/aws_pg/jobs/mendel_ft_family.py`) plus journal edits, which
vindicates the hold: `package_code()` would have shipped that module under a `dirty=true` stamp.
On the third re-check the sibling committed and the tree went clean; HEAD had moved twice
(`5d2c219f` → `53746551` → `0f15effa`). The gate was re-run on `0f15effa` (20 modules, 154 tests)
and the stamp taken in the same step as the launch, giving `dirty=false` and gated == stamped ==
launched. Two operational notes: (a) the hold window is worth spending even when the dirt looks
inert, because on a shared branch it is a *sample* of an in-progress commit, not a steady state;
(b) the check must be re-run *after* the stamp and immediately before `pg run` — the gate-to-launch
gap is where the race actually lives.

**Fourth instance, a different shape (2026-08-19, FT-family v2).** Here the tree was clean at BOTH
mandated checks — before the stamp and after it — and the launch was still not gate-covered: HEAD
moved *under* the ~40 s gate itself. `mendel_check.sh` was run at `37c99d28`; by the time the stamp
was taken 90 s later a sibling had committed `f41627e4`, so the shipped tree was one commit ahead of
the gated one. The clean-tree check cannot catch this — a fast-forwarded clean tree looks identical
to the one you gated. The fix is cheap and should be standard: after the launch returns, `git diff
--stat <gated_sha> <launched_sha>` and re-run the gate at the launched sha. Here the diff was a
single unrelated job module (`mendel_mn12_kernel.py`) and the re-run gate passed at `f41627e4`, so
the run is gate-covered after the fact. Rule: the clean-tree checks bound *untracked* drift; only a
sha-to-sha diff bounds *committed* drift, and the gate is not evidence about a sha it did not run on.

## Budget arithmetic, done too late

The 24-fine-tune `mendel_ftfam_v1` run was given 240 min. Observed rate is ~11 min per checkpoint
(dominated by the ~5.6 GB HF download), so 24 checkpoints needs ~4.4 h and the run will hit its
reaper before it aggregates. That was arithmetic available BEFORE launch and was not done.

Two consequences, both now fixed:
1. `mendel_ft_family` wrote its per-fine-tune rows only after the loop, so a reaper kill would have
   discarded every completed fine-tune's contrast. It now writes `out/partial.json` after each one
   (the house fleet rule: bind the results container into the streamed report BEFORE the loop).
2. A bounded companion run `mendel_ftfam_v2` (`--n-ft 10`, 180 min) was launched so the arm has a
   guaranteed complete answer regardless of what the 24-FT run does.

Rule for next time: multiply the per-unit rate by the unit count before choosing `--budget-min`,
and never launch a loop job whose results only materialise at the end.

## Quota note

Standard quota is **16 vCPU**, and r7i.2xlarge is 8 vCPU, so **at most two r7i jobs run
concurrently**; a third fails with `VcpuLimitExceeded`. GPU (g6e) draws on the separate G/VT
32-vCPU bucket and does not compete, which is why a g6e job can always be launched alongside two
r7i jobs.

## Defect in a shared surface: `energy_matched_subspace` misses its own tolerance

`infra/pg/conserved_core_probes.py::energy_matched_subspace` takes a `tol` (default 0.03) that
reads like a guarantee. It is not: the routine picks an INTEGER m, draws up to 80 Haar frames in
span(top-m), and **returns the closest attempt whether or not it met tol**, signalling only through
the returned energy value. Because m is an integer the achievable expected energies form a discrete
ladder, and at small targets consecutive rungs are further apart than 3%.

Measured hit rate at tol=0.03, k=32, n=512, 20 seeds per cell:

| profile | target 0.05 | target 0.15 | target 0.30 |
|---|---|---|---|
| lognormal(0,1) | **0%** | 40% | 45% |
| lognormal(0,2) | **0%** | 40% | 35% |
| uniform | 0% | 0% (target outside reachable range) | 0% |

On MENDEL's own substrate this made **39.7% of MN-6's 4320 cells uncontrolled**, the worst at
**350x** its target (q_proj L0's sink captures 4.5e-5 of the activation energy; no 32-dim subspace
of a 2048-dim space can go that low, so the target is structurally unreachable there).

MENDEL's fix is `mendel_core.energy_matched_control`, which returns the match quality and a
`matched` flag beside the subspace, plus `energy_match_report` for the pooled gate. MN-6's verdict
was recomputed on the matched subset and survives (see the journal).

The §13 ledger row for this surface reads "REUSED | verified :237" — it was verified to EXIST, not
to WORK. NOT fixed in `conserved_core_probes` itself: that file belongs to other programs' branches
and changing its behaviour would silently change their banked comparisons. Raise on their next
touch; every program that used it without checking the returned energy has the same exposure.
