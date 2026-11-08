# N13 state

Written 2026-11-07T15:55Z and updated 18:00Z (the handoff below) by the eleventh N13 executor (S4 running). It replaces the
2026-11-06T18:30Z version of the tenth executor (`git show 628f61b2b:studies/next/n13/STATE.md`). The tenth executor was cut off by
an account limit at about 19:28Z on the 6th.
**Note 2026-11-08 (index worker):** N13 is closed; S4 and S5 finished and the independent read is done (`READ_n13.md`, `READ_head.md`), recorded in `studies/next/JOURNAL.md` 2026-11-08 05:31 CST.

## Handoff 2026-11-07

Written 2026-11-07T18:00Z by the eleventh N13 executor before a terminal restart (this session cannot be resumed). Read this section
first; the rest of the file is the record it points into.

**Update by the twelfth N13 executor (started 18:35Z), latest first.** The commands below are in use unchanged; the tools in `$SP`
survived the restart (sha256 equal to the repo copies).
- 10:58Z (2026-11-08). **S5 done except the independent read.** Figures `studies/next/n13/figures/fig_n13.{png,pdf}`; samples
  coverage `studies/next/n13/results/SAMPLES_COVERAGE.md` (42 present, 0 pending); `studies/next/n13/CLAIMS.md` (N13-C1 to C13);
  `studies/next/n13/REGISTRY.md`; §11 of `N13_SPEC.md` filled from the verdict file only (`s5/grid11.py`); operations note with the N13
  total (102.555 GPU-h). The verdict goes to `results/next_n13/s5-01/verdict_n13.json` after this commit. **Next (the head):** the
  independent reader, in a worktree made from tag `n13-s4-complete`, downloads `results/next_n13/s5-01/banks/`, checks it against
  `results/next_n13/s5-01/manifest_sha256.json` (sha256 `56c87b3c06c4…`), runs `recount_n13.py` (must print `RECOUNT_OK`), commits,
  then opens the verdict and writes `READ_n13.md`. Then the result document. No merge by the executor.
- 10:51Z (2026-11-08). **Verdict run once: COMPLETE** (`studies/next/n13/verdict_n13.json`, sha256 `cae55c04ebf8…`). Banks tag
  `n13-s4-complete` (`2d81963d9`). S5 tree uploaded to `results/next_n13/s5-01/banks/` (165 files); manifest
  `results/next_n13/s5-01/manifest_sha256.json`, **sha256 `56c87b3c06c47dff17302f352d0436063add2f5398c57a4abb58f781bed790de`** (copy in
  `studies/next/n13/s5/`). H_local REFUTED (k_rec NONE); H_diffuse SUPPORTED (smallest k 0.1 %); H_attention REFUTED (BOTH); H_artifact
  INDETERMINATE; necessity NOT NECESSARY at all four k. Next: figures, samples coverage, CLAIMS.md, REGISTRY.md, §11; upload the verdict
  to `results/next_n13/s5-01/verdict_n13.json`; the independent reader.
- 10:50Z (2026-11-08). **S4 complete. `s4armsB1-02` ended**: 8 of 8 steps, STATUS `exit=0` at 10:47:04Z; 19.204 GPU-h launch to
  STATUS ($43.02). Banked since 03:16Z, every gate passing and every engine equal: B1-02 `rndT_k0.3`, `rndT_k0.1`, `comp_k0.1`, `topF_k1`.
  No open N13 instance. S4 76.511 GPU-h ($187.81); N13 102.555 GPU-h. Next: the S5 plan below.
- 08:45Z (2026-11-08). **S5 plan with the head's three additions (message received 08:44Z), in this order.** (1) When `topF_k1` is
  banked and B1-02's STATUS and prefix files are in: commit everything (tree clean), tag that commit `n13-s4-complete` (annotated; its
  subject names no label or accuracy); the independent reader's worktree is made from this tag. (2) Assemble the S5 tree
  (`$SP/s5/banks`: `assemble.py` output, plus each arm's `build.json`, `sets.json`, the R0 known `metrics.json`, R1 `controls.json`, the
  `s3gate-03` setup dir and `thresholds.json` under `_inputs/`); write `manifest_sha256.json` (every file's path, size, sha256); upload
  the tree to the fresh prefix `s3://pg-lineage-769063704696/results/next_n13/s5-01/banks/` and the manifest to
  `results/next_n13/s5-01/manifest_sha256.json`; run `verdict_n13.py` once on that same local tree; the manifest's sha256 goes in the
  verdict commit message and here. (3) Figures (`studies/next/n13/figures/`), `studies/next/n13/results/SAMPLES_COVERAGE.md`,
  `CLAIMS.md`, `REGISTRY.md`, §11, operations note, this file; commit; then upload `verdict_n13.json` to
  `results/next_n13/s5-01/verdict_n13.json`; report the tag, the verdict commit, the manifest sha256 and the headline to the head. The
  verdict is at `studies/next/n13/verdict_n13.json` (the spec's paths, not the 18:00Z plan's `s5/`).
- 03:16Z (2026-11-08). **`s4armsB2-02` ended**: 6 of 6 steps, STATUS `exit=0` at 03:11:45Z; 11.208 GPU-h launch to STATUS ($25.11).
  Banked since 19:57Z, every gate passing and every engine equal: B1-02 `attn` (MATH-500, GSM8K), `rndT_k3`; B2-02 `rndE_k0.3`,
  `rndE_k0.1` (A31 relative 1.18e-16 and 1.43e-16, 0 tensors differ), `comp_k0.3`, `rndT_k1_d2`. Open launch: B1-02 only (`rndT_k0.3`
  running, then `rndT_k0.1`, `comp_k0.1`, `topF_k1`; reaper 2026-11-09T03:30:48Z). Poller:
  `python3 -I $SP/s4launch/watch_all.py 400 s4armsB1-02=i-014ec0255a75645f3`. GPU-h at 03:15Z: S4 68.98 ($170.93), N13 95.02.
- 19:57Z. **`s4armsA1-01` ended**: 8 of 8 steps, STATUS `exit=0` at 19:38:55Z, terminated by itself about 19:44Z; 20.647 GPU-h launch
  to STATUS ($46.25). `top_k1_B` banked (seed set B; 355 tensors equal R1 stage 6; G-MASK and G-COMPLETE pass, `n_empty` 11; 76 cap
  hits; engine equal). Prefix files banked. Open launches now: B1-02 (`attn` running, then `rndT_k3`, `rndT_k0.3`, `rndT_k0.1`,
  `comp_k0.1`, `topF_k1`) and B2-02 (`rndE_k0.3` running, then `rndE_k0.1`, `comp_k0.3`, `rndT_k1_d2`). Poller:
  `python3 -I $SP/s4launch/watch_all.py 400 s4armsB1-02=i-014ec0255a75645f3 s4armsB2-02=i-0e4fa2873c2468e5f`.
  GPU-h at 19:57Z: S4 54.43 ($138.35), N13 80.47.
- 19:15Z. B1-02 `rndE_k1` banked (A31 relative 1.016e-15, 0 of 355 tensors differ; G-COMPLETE pass on both benchmarks). B1-02 `attn`
  engine equal.
- 18:35Z. B2-02 `rndE_k0.3` engine equal; A31 written 0.058904975675104894, matched 0.05890497567510489, relative 1.18e-16, 0 of 355
  tensors differ, pass.

**Open launches at 18:00Z** (every one g6e.2xlarge, `ami-0cee78ee4b1c24614`; the progress is from the 17:46Z poll):

| prefix (`results/next_n13/…`) | instance | launched | step now | progress | job hard kill (`timeout -s KILL`, about launch + `--budget-min`) | reaper (EventBridge, terminates the instance) |
|---|---|---|---|---|---|---|
| `s4armsA1-01` | `i-01f83b1bee1fcce84` | 2026-11-06T23:00:07Z | 8 of 8 steps, `top_k1_B` (the last; engine equal to S3's, 17:21Z) | MATH-500 140/500, 10 cap hits | about 2026-11-08T10:46Z (2,146 min) | `at(2026-11-08T10:56:07)` |
| `s4armsB1-02` | `i-014ec0255a75645f3` | 2026-11-07T15:34:48Z | 2 of 8, `rndE_k1` (engine equal) | MATH-500 480/500, 58 cap hits; GSM8K 6/1,319 | about 2026-11-09T03:20Z (2,146 min) | `at(2026-11-09T03:30:48)` |
| `s4armsB2-02` | `i-0e4fa2873c2468e5f` | 2026-11-07T15:59:15Z | 2 of 6, `rndE_k3` (engine equal) | MATH-500 362/500, 42 cap hits | about 2026-11-08T14:57Z (1,378 min) | `at(2026-11-08T15:07:16)` |

Steps still to run after the current ones: B1-02 `attn` (MATH-500 and GSM8K), `rndT_k3`, `rndT_k0.3`, `rndT_k0.1`, `comp_k0.1`, `topF_k1`
(conditional; it runs: J_floor(1 %) 0.642 < 0.9); B2-02 `rndE_k0.3`, `rndE_k0.1`, `comp_k0.3`, `rndT_k1_d2`. No launch loop runs. The
executor never terminates an instance (the head's rule); each self-terminates after STATUS.

**Readings since the 16:05Z journal line** (all blind):
- B2-02 setup (16:00:05Z to 16:08:09Z): G-GRADER passes (500/500, 24/24, 16/16, derangement 0.006, failing input fails); G-TEMPLATE
  passes, prompt ids sha256 `ff88a606d568…` as at S3. Its n13 pip freeze differs from S3's in the same seven packages as B1-02's
  (cachetools 7.3.0 → 8.0.0, cbor2, click, jiter, regex, sse-starlette 3.6.0 → 3.7.0, typer); the five compared libraries are equal.
  cachetools and sse-starlette are new in B1-02 and B2-02; reported to the coordinator at about 16:10Z, recorded in the operations
  note at 18:00Z.
- B2-02 `rndE_k3` A31 (`build.json` → `a31`): written 0.43451684513088845, matched 0.4345168451308886, relative 3.8e-16 (tolerance
  1e-12), 0 of 355 tensors differ, pass. Realized MLP energy 0.8 % under the intended 0.43806 (`energy_flag` false); passes c 1.1478
  → 1.0627, c 1.1134 → 0.9919; `c_built` 1.1134. Only MLP tensors change. Engine equal to S3's.
- A1 `comp_k3` banked (done 17:16:47Z): 355 tensors equal R1 stage 6, G-MASK pass, G-COMPLETE passes (job and recount; `n_empty` 0;
  failing input fails), 27 cap hits, mean 4,498.6 tokens (median 2,733), modes boxed 483 / none 17, screen not fired, 143.9 min
  against the projection 208.0, engine equal.

**Banked in the repo** (`studies/next/n13/s4/<prefix>/<step>/`: files under 1 MB, `arm_record.json`, `engine_check.json`, `on_s3.json`
naming each larger file by size and sha256; full copies in `$SP/s4banks/<prefix>/<step>/`):
- `s4armsA2-01` (done): `dare_k1`, `top_k3`, `top_k0.1`, `dare_k0.3`, `comp_k1`, `rnd0_k1`, `dare_k1_d2`, setup, prefix files.
- `s4armsA1-01`: `top_k1`, `mlp`, `top_k0.3`, `dare_k3`, `dare_k0.1`, `comp_k3`, setup.
- `s4armsB2-01` (stopped, A30): `rndT_k1`, setup, failure lines. `s4armsB1-01` (stopped, A30): setup, failure lines.
- `s4armsB1-02`, `s4armsB2-02`: setup and prefix files only.

**On S3 only** (under `s3://pg-lineage-769063704696/results/next_n13/<prefix>/out/<step>/`): the running steps A1 `top_k1_B`, B1-02
`rndE_k1`, B2-02 `rndE_k3` (their `engine_check.json` is in the repo; nothing else of them is banked); every `generations.jsonl`,
`build.json` and `steps.jsonl` over 1 MB of the banked steps (named in each `on_s3.json`).

**Commands.** The tools hard-code `$SP` (below) and are copies of `studies/next/n13/s4/tools/` (identical by sha256 at this commit).
`/private/tmp` survives a terminal restart, so the new session runs them from `$SP` as they are. Run every command from `cd $SP`.

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
W=/Users/abaybektursun/projects/interp_research-n13
cd $SP
# poller: every 120 s, exits on the first event (step state change, STATUS, instance not running, an engine block first seen,
# the screen firing); state in $SP/s4launch/watch_state.json, one line per poll in $SP/s4launch/w_<prefix>.log. Run in the background.
python3 -I $SP/s4launch/watch_all.py 400 s4armsA1-01=i-01f83b1bee1fcce84 s4armsB1-02=i-014ec0255a75645f3 s4armsB2-02=i-0e4fa2873c2468e5f
# event "engine loaded <step>": compare with S3 (exit 0 equal, 1 different -> report to the head at once, let the instance run)
python3 -I $SP/s4tools/engine_check.py <prefix> <step>
# for an rndE step, read build.a31 then (written_sq, matched_sq, rel_diff, n_tensors_differ, pass) from its build.json on S3
# event "<step>=ok": bank the step, then its record
python3 -I $SP/s4tools/bank_s4.py <prefix> <step>
PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 $SP/n13_venv/bin/python $SP/s4tools/arm_record.py <prefix> <step>
# event "STATUS exit=…": bank the prefix files, drop the two setup inputs from the repo copy (as for every launch), redraw the table
python3 -I $SP/s4tools/bank_s4.py <prefix> --prefix-files
rm -f $W/studies/next/n13/s4/<prefix>/setup/problems.json $W/studies/next/n13/s4/<prefix>/setup/prompt_ids.json
python3 -I $SP/s4tools/table.py
```

After each event: re-run the poller (it exits after one event), commit the new bank files by name, add a journal line. A false event
follows a laptop network loss (`no pipeline_status | ec2=describe-failed` lines in `w_<prefix>.log`, then the same state again); it
happened 17:28Z to 17:44Z. If `$SP` is gone (a reboot), put the tools back at the same paths from `studies/next/n13/s4/tools/` (the
poller under `$SP/s4launch/`, the others under `$SP/s4tools/`, `assemble.py` under `$SP/s5/`); rebuild `$SP/s4banks` with
`bank_s4.py` per banked step; `$SP/s3banks` (the four gate banks: `base`, `rl` from `s3gate-03`, `zero` from `s3gate-02`, `id` from
`s3gate-04`) and `$SP/n13_venv` (Python 3.12, numpy, the n13 scoring packages) have no rebuild script in the repo.

**S5 close-out, as I would run it** (`N13_SPEC.md` §9.4 S5, §9.5: the verdict runs once):
1. Wait until all three launches have STATUS and are terminated. Bank every step and the prefix files. Record each launch's GPU-hours
   (launch to STATUS) in the operations note. List any parked step (`metrics.json` status `parked`) and `topF_k1`'s status.
2. Check the roster: 31 MATH-500 arm runs (4 gate + 27 S4) and 11 GSM8K arm runs, each banked once, every engine check equal (an
   arm whose engine differs does not count), every G-COMPLETE passing with its failing input failing.
3. Assemble one banks tree (copies, no links) from the four S3 gate banks and the 27 S4 steps:
   ```
   python3 -I $SP/s5/assemble.py $SP/s5/banks \
     s4armsA2-01:dare_k1 s4armsA2-01:top_k3 s4armsA2-01:top_k0.1 s4armsA2-01:dare_k0.3 s4armsA2-01:comp_k1 s4armsA2-01:rnd0_k1 s4armsA2-01:dare_k1_d2 \
     s4armsA1-01:top_k1 s4armsA1-01:mlp s4armsA1-01:top_k0.3 s4armsA1-01:dare_k3 s4armsA1-01:dare_k0.1 s4armsA1-01:comp_k3 s4armsA1-01:top_k1_B \
     s4armsB2-01:rndT_k1 \
     s4armsB1-02:rndE_k1 s4armsB1-02:attn s4armsB1-02:rndT_k3 s4armsB1-02:rndT_k0.3 s4armsB1-02:rndT_k0.1 s4armsB1-02:comp_k0.1 s4armsB1-02:topF_k1 \
     s4armsB2-02:rndE_k3 s4armsB2-02:rndE_k0.3 s4armsB2-02:rndE_k0.1 s4armsB2-02:comp_k0.3 s4armsB2-02:rndT_k1_d2
   ```
   It writes `$SP/s5/s5_sources.json` (source, status and `generations.jsonl` sha256 per arm and benchmark) and reads no score. It has
   not been run yet. Copy `s5_sources.json` to `studies/next/n13/s5/`.
4. Unblind once:
   ```
   cd $W && mkdir -p studies/next/n13/s5 && PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 $SP/n13_venv/bin/python -m studies.next.n13.scripts.verdict_n13 \
     --banks $SP/s5/banks --thresholds studies/next/n13/thresholds.json --setup studies/next/n13/s3/s3gate-03/setup \
     --known studies/next/n13/r0/s2setup-07/r0_known/metrics.json --controls studies/next/n13/s3/s3gate-03/r1_rank/controls.json \
     --out studies/next/n13/s5/verdict_n13.json
   ```
   Exit 0 means the verdict was written; 3 or 5 means CANCELLED by a gate. Either way the result stands; it is not re-run.
5. `python -m studies.next.n13.scripts.figure_n13 --verdict studies/next/n13/s5/verdict_n13.json --out-dir studies/next/n13/s5/figures`
   and `python -m studies.next.n13.scripts.samples_coverage --banks $SP/s5/banks --out studies/next/n13/s5/samples_coverage.md`
   (same environment as step 4).
6. Write the claims record and the registry entry. Commit by name.
7. An independent reader, a model other than the executor (repo Provenance rule 3; the head picks it), runs
   `python -m studies.next.n13.scripts.recount_n13 --banks $SP/s5/banks --verdict studies/next/n13/s5/verdict_n13.json` (it must print
   `RECOUNT_OK`) and writes `studies/next/n13/READ_n13.md`. The result document is written after that read.

**GPU-hours and cost so far** (launch to STATUS for ended launches; launch to 18:00Z for open ones; g6e.2xlarge at $2.24/h,
g6e.4xlarge at $3.00424/h):

| launch | type | state | GPU-h | cost | projection, GPU-h | ceiling at hard kill, GPU-h |
|---|---|---|---|---|---|---|
| `s4armsA2-01` | g6e.4xlarge | done, 8 of 8 | 21.482 | $64.54 | 27.52 | 35.93 |
| `s4armsB2-01` | g6e.2xlarge | stopped (A30) | 3.705 | $8.30 | 22.59 | — |
| `s4armsB1-01` | g6e.2xlarge | stopped (A30) | 0.265 | $0.59 | 27.52 | — |
| `s4armsA1-01` | g6e.2xlarge | running | 19.00 | $42.56 | 27.52 | 35.93 |
| `s4armsB1-02` | g6e.2xlarge | running | 2.42 | $5.42 | 27.52 | 35.93 |
| `s4armsB2-02` | g6e.2xlarge | running | 2.01 | $4.51 | 17.67 | 23.13 |
| S4 so far | | | 48.88 | $125.91 | | |

N13 so far: 26.044 GPU-h before S4 plus 48.88 = 74.93 GPU-h. No study ceiling (A27).

**Open for the head:** cachetools and sse-starlette in the B1-02 and B2-02 pip freezes (operations note, 18:00Z). The head's 15:47Z item 3
(report each rndE arm's `a31` record): `rndE_k1` reported at 16:05Z, `rndE_k3` to the coordinator at about 17:18Z; `rndE_k0.3`,
`rndE_k0.1` (B2-02) still to come.


**Roster (31 MATH-500 arm runs, 30 unconditional and 1 conditional; 11 GSM8K arm runs; R0, R1, the smoke): run R0, R1, the smoke,
31 MATH-500 arm runs (4 gate arms, 27 S4, the conditional `topF_k1` among them) and 11 GSM8K arm runs (4 gate, 7 S4); running 0;
0 parked; unrun 0.** (2026-11-08 10:50Z)
- **Spent: 26.044 GPU-h before S4.** S4 so far (launch to STATUS): `s4armsA2-01` 21.482 h (g6e.4xlarge, done, all 8 steps),
  `s4armsB2-01` 3.705 h, `s4armsB1-01` 0.265 h; A1, B1-02 and B2-02 accrue (19.00, 2.42 and 2.01 h at 18:00Z; table in the handoff).
  N13 so far 74.93 GPU-h at 18:00Z.
  No study ceiling (A27).
- **Open instances:**
  - `i-01f83b1bee1fcce84`: `results/next_n13/s4armsA1-01`, g6e.2xlarge, launched 2026-11-06T23:00:07Z, reaper `at(2026-11-08T10:56:07)`.
  - `i-014ec0255a75645f3`: `results/next_n13/s4armsB1-02`, g6e.2xlarge, launched 2026-11-07T15:34:48Z, reaper `at(2026-11-09T03:30:48)`.
  - `i-0e4fa2873c2468e5f`: `results/next_n13/s4armsB2-02`, g6e.2xlarge, launched 2026-11-07T15:59:15Z, reaper `at(2026-11-08T15:07:16)`;
    from the detached worktree `/Users/abaybektursun/projects/interp_research-n13-launch` at `n13-freeze-2a` (`e69f2b2b7`, as B1-02).
- **Freeze 1: `n13-freeze-1`**, tagged 2026-11-06T04:00:49Z on `79e9a28e5` (A1–A28), on the head's authorization.
- **Freeze 1a: `n13-freeze-1a`**, tagged 2026-11-06T05:48:57Z on `b8ae07996` (freeze 1 plus A29, approved by the head).
- **Freeze 2: `n13-freeze-2`**, tagged 2026-11-06T18:09:51Z on `ad8a3da39` (`studies/next/n13/thresholds.json`).
- **Freeze 2a: `n13-freeze-2a`**, tagged 2026-11-07T15:47:29Z on `e69f2b2b7` (freeze 2 plus A31, the head). `N13_DRYRUN_OK` 111 of 111
  (`studies/next/n13/offline_gate/rerun_a31/`).

## S4 (2026-11-06 and 07)

- Four launches (A28). A2 from `9cf5e73c6`; A1, B1-01 and B2-01 from `628f61b2b` (same `infra/` code); B1-02 and B2-02 from
  `e69f2b2b7` (A31). Inputs in every prefix's `inputs/`, read back equal: `sets.json` `49a1065c…`, `thresholds.json` `60bf6d4e…`
  (= `n13-freeze-2`), `controls.json` `f98c30e4…` (`s3gate-03`). Operations: `infra/aws_pg/operations/2026-11-06_next_n13_s4.md`.
- **B1-01 and B2-01 stopped at A30's exact energy check** (B2-01 at `rndE_k3` after `rndT_k1` completed, 02:44:04Z; B1-01 at
  `rndE_k1`, 03:46:52Z). Cause: two float64 sums over the 96 MLP tensors in two orders; the tensors are equal. **A31**: every tensor's
  sha256 equal and the energies within 1e-12 relative (bound 2.1e-14); a written entry one bf16 step up still fails (`N13_SPEC.md` A31).
  Relaunched as `s4armsB1-02` (B1-01's seven arms, allowance 2,146 min) and `s4armsB2-02` (`rndE_k3`, `rndE_k0.3`, `rndE_k0.1`,
  `comp_k0.3`, `rndT_k1_d2`; allowance 1,378). `rndT_k1` is banked from B2-01.
- **First real A31 reading** (B1-02 `rndE_k1`): written 0.16391162291697847, matched 0.1639116229169783, relative 1.016e-15, 0 of 355
  tensors differ, pass; `c_built` 1.1752 (passes 1.0733, 0.9951 of the intended).
- **The head's engine rule** (`N13_SPEC.md` §9.4 S4 head's note): every arm's engine block and the five libraries are compared with
  S3's as it lands (`engine_check.py`); a difference is reported to the head at once, the instance runs on, the executor never
  terminates it, and the arm does not count. Every landed and running step so far is equal. Outside the five libraries the n13 pip
  freezes differ from S3's in click, jiter, typer (all S4 launches), regex (A1, B1-01, B2-01) and cbor2 (B1-01); reported. regex and
  cbor2 are not imported by the scoring path (`n13_score`, `boxed_last`, math-verify and latex2sympy2_extended use the standard `re`).
- **Blind protocol (§9.5):** gates, G-COMPLETE counts, engine blocks, lengths and the samples only; no accuracy is read before S5.
- **Banked, blind** (`studies/next/n13/s4/<prefix>/<step>/`: the files under 1 MB, `arm_record.json`, `engine_check.json`, `on_s3.json`
  naming the larger files by sha256; full copies in `$SP/s4banks/<prefix>/<step>/`). Table: `studies/next/n13/s4/gates_table.txt`.

| launch | step | benches | 355 tensors = R1 stage 6 | G-MASK | DARE kept (sd) | E / E(S_k) | G-COMPLETE (n_empty) | cap hits | mean tokens | wall / projection, min |
|---|---|---|---|---|---|---|---|---|---|---|
| A2 | `dare_k1` | MATH-500, GSM8K | yes | — | −0.27 | — | pass (0, 0) | 52, 60 | 4,666.7, 1,746.3 | 285.0 / 295.4 |
| A2 | `top_k3` | MATH-500 | yes | pass | — | — | pass (9) | 83 | 4,003.9 | 156.9 / 208.0 |
| A2 | `top_k0.1` | MATH-500 | yes | pass | — | — | pass (8) | 75 | 3,545.4 | 140.1 / 208.0 |
| A2 | `dare_k0.3` | MATH-500 | yes | — | −0.59 | — | pass (0) | 49 | 4,569.1 | 155.1 / 208.0 |
| A2 | `comp_k1` | MATH-500, GSM8K | yes | pass | — | — | pass (0, 0) | 24, 25 | 4,523.3, 2,002.4 | 253.6 / 295.4 |
| A2 | `rnd0_k1` | MATH-500 | yes | pass | — | — | pass (8) | 74 | 3,481.8 | 137.8 / 208.0 |
| A2 | `dare_k1_d2` | MATH-500 | yes | — | 0.02 | — | pass (0) | 45 | 4,478.4 | 149.8 / 208.0 |
| A1 | `top_k1` | MATH-500, GSM8K | yes | pass | — | — | pass (9, 72) | 79, 43 | 3,715.3, 857.4 | 217.4 / 295.4 |
| A1 | `mlp` | MATH-500, GSM8K | yes | — | — | — | pass (0, 0) | 42, 73 | 4,468.7, 1,883.3 | 289.6 / 295.4 |
| A1 | `top_k0.3` | MATH-500 | yes | pass | — | — | pass (7) | 67 | 3,347.3 | 128.9 / 208.0 |
| A1 | `dare_k3` | MATH-500 | yes | — | −0.74 | — | pass (0) | 43 | 4,456.1 | 150.1 / 208.0 |
| A1 | `dare_k0.1` | MATH-500 | yes | — | 0.00 | — | pass (0) | 48 | 4,606.7 | 156.9 / 208.0 |
| A1 | `comp_k3` | MATH-500 | yes | pass | — | — | pass (0) | 27 | 4,498.6 | 143.9 / 208.0 |
| B2-01 | `rndT_k1` | MATH-500, GSM8K | yes | pass | — | 1.0001 | pass (9, 73) | 71, 45 | 3,438.0, 861.3 | 206.8 / 295.4 |

Every G-COMPLETE failing input (the bank with its last answer removed) fails; no damage screen fired; every engine block equals S3's.
Setups of A2, A1, B1-01 and B2-01: G-GRADER (500/500, 24/24, 16/16, derangement 0.006; failing input fails) and G-TEMPLATE pass; prompt
ids sha256 `ff88a606d568…` as at S3. The three §9.4 S2 answers on `dare_k1` (A2) and `top_k1` (A1) are in `JOURNAL.md` (2026-11-07
15:45Z). The R1 stage-6 reference is `s3gate-03` `out/r1_rank/arm_hashes.json` (1,190,854 bytes, sha256 `0d71e4923703…`).
- For S5: the R0 G-KNOWN record is `studies/next/n13/r0/s2setup-07/r0_known/metrics.json` (G-KNOWN 39.96, pass; failing input 1.12,
  fails).

## Where things stand

- **C = 16,384** (§2.3 as amended by A17, on RL problems 0–19; `studies/next/n13/r0/s2smoke-09/smoke_analysis.json`).
- **The smoke** (`studies/next/n13/r0/s2smoke-09/`; reading in `JOURNAL.md`, 2026-11-06 03:40Z): base 177 of 300 right under A26
  (153 as recorded on the instance), mean 2,896.6 output tokens, 33 at the cap, 7 empty; RL 131 of the 142 finished right.
- **The rate** is r_sat 236.6 output tokens per second (base smoke; A22, A24).
- **A23, A24** in code (eighth executor). **A25** (G-COMPLETE: n_generated ≥ 1 and a finish_reason; empty text counted as `n_empty`)
  and **A26** (the `Answer:` label inside one markdown emphasis pair) in code at `e897d9b53`; on the smoke, base 153 → 177 of 300,
  RL unchanged (`studies/next/n13/amendments/a26_rescore.json`).
- **A27** (the user, 2026-11-05 21:42 CST: "Remove the GPU limits."): no study ceiling; each launch's allowance is 1.3 × its
  projection, in the manifest and in `--budget-min`; the park order applies only to that allowance.
- **A28** (the head): S3 in two launches, S4 in four. `zero` and `id` read nothing of R1's output.
- `N13_DRYRUN_OK` 107 of 107 (`studies/next/n13/offline_gate/rerun_a27/`).

## Projection per launch (A24 `prefix`; rate 236.6; L_math 5,498.7; L_gsm 940 flagged; F 9.35 min per load; build 5 min per built arm)

Source: `studies/next/n13/r0/s2smoke-09/projection.json` → `launches`. Manifests in `infra/aws_pg/steps/`.

| launch | manifest | arms | projection, min | GPU-h | allowance, min |
|---|---|---|---|---|---|
| S3 | `n13_s3gate-01.json` | setup with stage 2, R1, base, rl (+GSM8K) | 675.8 | 11.26 | 879 |
| S3 | `n13_s3gate-02.json` | setup, zero, id (+GSM8K) | 610.8 | 10.18 | 794 |
| S4 | `n13_s4armsA1-01.json` | top_k1, mlp, top_k0.3, dare_k3, dare_k0.1, comp_k3, top_k1_B | 1,651.0 | 27.52 | 2,146 |
| S4 | `n13_s4armsA2-01.json` | dare_k1, top_k3, top_k0.1, dare_k0.3, comp_k1, rnd0_k1, dare_k1_d2 | 1,651.0 | 27.52 | 2,146 |
| S4 | `n13_s4armsB1-01.json` | rndE_k1, attn, rndT_k3, rndT_k0.3, rndT_k0.1, comp_k0.1, topF_k1 (conditional) | 1,651.0 | 27.52 | 2,146 |
| S4 | `n13_s4armsB2-01.json` | rndT_k1, rndE_k3, rndE_k0.3, rndE_k0.1, comp_k0.3, rndT_k1_d2 | 1,355.6 | 22.59 | 1,762 |
| S4 (A31) | `n13_s4armsB1-02.json` | as B1-01 | 1,651.0 | 27.52 | 2,146 |
| S4 (A31) | `n13_s4armsB2-02.json` | rndE_k3, rndE_k0.3, rndE_k0.1, comp_k0.3, rndT_k1_d2 | 1,060.2 | 17.67 | 1,378 |

S3: 21.44 GPU-h, wall clock about 11.3 h (the longer launch). S4: 105.14 GPU-h, wall clock about 27.5 h. Cumulative 28.50 GPU-h
after S3, 133.65 after S4. S4 through P1 across the four launches: 21.03 GPU-h; through P2: 30.87 (`s4_by_priority_four_launches`).
The S4 manifests read `sets.json`, `controls.json` and `thresholds.json` from `s3://pg-lineage-769063704696/results/next_n13/<kind>-01/inputs/`;
they were not regenerated: S4 launched with the committed manifests (the head, after its read of the S3 gate).

## `s3gate-01` (R1 stopped in stage 6)

- Setup passed: G-GRADER, G-INIT (a) and G-TEMPLATE. R1 stages 1–5 passed:
  - G-PARTITION;
  - G-INIT (b): unchanged MLP share 0.5626 for base `main` against 0.0055 for stage 2, difference 0.557 against the margin 0.10;
  - PC-RANK: recovery 1.0, failing input 0.0;
  - PC-ART: P+ SUPPORTED, P− REFUTED.
- `sets.json` was written. Stage 6 (arm hashes) stopped on `id`: the c = 1 builder check meets inexact Δ entries. R1 counts 152 in
  the MLP, 144 in attention and 17 in the embedding. The arms themselves are bit copies and were never wrong. Proposed fix A29.
- A20 block profile (`r1_rank/weight_stats.json` → `block_profile`): flat. MLP ‖Δθ‖/‖θ_b‖ is 0.00223 to 0.00238 per block (max/min
  1.065); attention is 0.00242 to 0.00296 (max/min 1.22), highest in block 0 and blocks 28–31.
- Banked in `studies/next/n13/s3/s3gate-01/`. `ranking.npz` stays on S3.
- **Block-window arms retired.** The block-window transplant arms of `12_SHAPING_NOTE.md` §4 item 7 were conditional on a
  block peak in A20. The profile is flat (MLP max/min 1.065, attention 1.22), so they do not run.

## S3 close-out (2026-11-06 18:10Z): every gate passes

Inputs are the four gate banks on MATH-500 and GSM8K: `base`-A and `rl`-A from `s3gate-03`, `zero`-B from `s3gate-02`, `id`-B from
`s3gate-04`. `studies/next/n13/thresholds.json` is written by `freeze_thresholds.py --cap 16384 --pair pair1`; bank sha256 are in it.

**Laptop regrade** (`studies/next/n13/s3/regrade.py`, `regrade.json`). All 7,276 answers of the eight banks re-extract and re-grade to
their recorded mode, extracted string and score: 0 mismatches. Every recount equals its `metrics.json`.

| gate | value | bound | failing input | result |
|---|---|---|---|---|
| G-GRADER | 500/500, 24/24, 16/16, derangement 0.006 (each setup) | ≤ 0.016 | derangement as identity: fails | pass |
| G-TEMPLATE | 500/500 and 1,319/1,319 per arm, no BOS | — | `add_generation_prompt=False` differs | pass |
| G-INIT, G-PARTITION, PC-RANK, PC-ART, G-MASK | R1 (`s3gate-03`; `sets.json` = `s3gate-01`'s) | as registered | each fails | pass |
| G-HASH | `zero` = base and `id` = RL, every tensor (at build; R1 too) | — | neuron (0, 0) changed: fails | pass |
| G-COMPLETE (A25) | 8 banks, every index once, seeded, n_generated ≥ 1; `n_empty`: base 7 / 71, rl 0 / 0, zero 11 / 83, id 0 / 0 (MATH-500 / GSM8K) | — | one answer removed: fails | pass |
| G-GATE (MATH-500) | G = L − B = 92.0 − 57.2 = 34.8; G' = L' − B' = 91.2 − 55.4 = 35.8; McNemar (rl-A vs base-A) p = 1.1e-45 | G, G' ≥ 5√2σ = 13.60; p < 0.001 | (rl-A, id-B) as the pair: G 0.8, fails | pass |
| G-ID (MATH-500) | \|L − L'\| = 0.8; \|B − B'\| = 1.8 | ≤ 3√2σ = 8.16 | (base-A, rl-A) as identity: −34.8, fails | pass |
| S-BAND | L' = 91.2 | [81.5, 96.7] | id-B against deranged golds: 0.8, fails | pass |
| room | G = 34.8 | ≥ 2.5δ = 13.60 | — | pass |
| G-GATE-GSM | G = 92.34 − 63.61 = 28.73; G' = 92.04 − 64.22 = 27.82; p = 2.9e-82 | ≥ 5√2σ_gsm = 8.73 | (rl-A, id-B): 0.30, fails | pass |
| G-ID (GSM8K) | \|L − L'\| = 0.30; \|B − B'\| = 0.61 | ≤ 5.24 | (base-A, rl-A): fails | pass |

**Frozen numbers (MATH-500, points):**
- σ_base 1.924 and σ_rl 0.693, so σ = 1.924 and δ = 2√2σ = 5.441;
- B 57.2, L 92.0, G 34.8; B' 55.4, L' 91.2, G' 35.8;
- H = B + G/2 = 74.6; Q = B + G/4 = 65.9;
- C = 16,384.

**GSM8K:** σ 1.234, δ 3.491, B 63.61, L 92.34, G 28.73, H 77.98, Q 70.79. No ceiling or floor stamp: L < 98 and B > 2.

**Same engine on every launch (A28).** Every launch ran vLLM 0.32.0 with max_num_seqs 128, 12,817 GPU blocks, max_num_batched_tokens
2,048, FLASH_ATTN and the FlashInfer sampler off. The n13 pip freezes of `s3gate-01` to `-04` are equal line for line.

**One right and one wrong per arm (MATH-500, from `samples.jsonl`):**
- `base`: problem 0, extracted `(3, π/2)` in mode answer_line (A26), right, answer probability 0.397. Problem 2, boxed `-1` against
  gold `\frac{14}{3}`, wrong, probability 0.995.
- `rl`: problem 0, `(3, \frac{\pi}{2})`, right, 0.450. Problem 9, cut at the 16,384 cap, no answer, wrong.
- `zero`: problem 1, `p - q`, right, 1.000. Problem 0, no answer (mode none), wrong.
- `id`: problem 0, `\left(3, \frac{\pi}{2}\right)`, right, 0.375. Problem 9, boxed `2` against gold `4`, wrong, 1.000.

## `s3gate-03` R1 (done 10:53Z)

- `sets.json` reproduces `s3gate-01`'s bit for bit (sha256 `49a1065c…`); so do `weight_stats.json` and `ranking.npz`
  (`studies/next/n13/s3/s3gate-03/r1_rank/reproduce_s3gate01.json`).
- R1's gates pass, each with its failing input failing: G-PARTITION, G-INIT, PC-RANK 1.0, PC-ART, G-HASH (zero, id), G-MASK (20 arms).
  The DARE kept counts are within tolerance.
- R1 took 145.1 min. The stage-2 deletion left base `main`.
- **Flag → A30** (the head; in code, `N13_DRYRUN_OK` 110 of 110): the `rndE` arms realized 0.70 to 0.82 of their intended energy.
  From freeze 2 on they use stochastic bf16 rounding plus at most three c adjustments to land within 5 %. Only the `rndE` arm
  hashes change. rndT realizes 1.0000 to 1.0011 of E(S_k) (reported).

## `s3gate-04` (`id` banked)

- `id`: MATH-500 500 of 500 answers, GSM8K 1,319 of 1,319. G-COMPLETE passes on both (`n_empty` 0). G-HASH passed at the build (A29
  code) and its failing input fails. The damage screen did not fire. Banked in `studies/next/n13/s3/s3gate-04/`.

## `s3gate-02` (`zero` banked; `id` failed as A29 predicted)

- `zero`: MATH-500 500 of 500 answers and GSM8K 1,319 of 1,319. G-COMPLETE passes on both (A25; `n_empty` 11 and 83). G-HASH
  passed at the build, and its failing input fails. G-TEMPLATE passed. The damage screen did not fire. Status `done`.
- `id` stopped in its build with the pre-A29 AssertionError, as predicted. That is A29's prediction holding.
- `zero`'s MATH-500 mean output was 3,280.6 tokens and GSM8K 571.5, both shorter than the projection's 5,498.7 and 940.
- The n13 pip freeze equals `s3gate-01`'s and attempt 9's.
- Banked in `studies/next/n13/s3/s3gate-02/`. The two `generations.jsonl` files stay on S3; their sha256 are in the operations note.

## Open for the head

1. Carried over: every S3 and S4 arm runs through vLLM's Transformers modeling backend (A21).

## Exact next action

See "Handoff 2026-11-07" at the top: watch, engine-check, bank, then S5.

## Resume steps

See "Handoff 2026-11-07" at the top (commands with their environment).
