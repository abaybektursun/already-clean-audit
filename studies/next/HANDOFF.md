# Handoff, 2026-11-08 — state of the already-clean neuron program for a fresh head session

Written 2026-11-08 by an Opus worker for the head. It replaces `HANDOFF_2026-11-07.md`, which is kept and marked
superseded. Nothing here is a result; every number names its file. Paths are under `studies/next/` unless they start
with `infra/` or `/`. "JOURNAL" is `studies/next/JOURNAL.md`; entries are named by stamp and line number.

## 1. How to resume

1. Read this file, then `00_INDEX.md` (the map: every study's spec, verdict, reads, figure, cost; the method rules).
2. Read `20_ABSTRACT.md` (the final abstract) and JOURNAL from "2026-11-08 05:00 CST — N13 verdict" (line 145) to the
   end (line 155).
3. Check liveness before saying anything about state: from `/Users/abaybektursun/projects/interp_research-clean-neurons`
   (where `.pgvenv` is linked, JOURNAL line 27) run `./infra/aws_pg/pg ps` and `./infra/aws_pg/pg audit` (AWS profile
   `fuelos`, region `us-east-1`). `pg ps` lists only running instances; a launch missing from it has ended.
4. Check git: `git -C /Users/abaybektursun/projects/interp_research-clean-neurons log --oneline -5`. Branch `index`
   (this file, `00_INDEX.md`, the README pointers, three dated stale-status notes) waits for the head's merge into
   `clean-neurons`.
5. Keep the head's journal: one entry per state change, appended to `studies/next/JOURNAL.md` on branch
   `clean-neurons`, committed by name, with the stamp read from the clock (§5).

## 2. What is closed

Every study is closed and merged into `clean-neurons`. Labels are the verdict files' headlines.

| study | label | verdict file | closed in JOURNAL |
|---|---|---|---|
| M1, stage-2 data mixes | `H_lead SUPPORTED; H_order UNDECIDED; H_specific SUPPORTED` | `m1/verdict_m1.json` | 2026-11-05 17:40 CST (line 51) |
| M2, name the carrier before birth | `H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES` | `m2/verdict_m2.json` | 2026-11-06 02:40 CST (line 78) |
| N12, competition at birth | `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT; H_bridge DOES NOT COUNT; H_dt SUPPORTED` | `n12/verdict_n12.json` | 2026-11-06 08:10 CST (line 87) |
| N2V, N12's instrument in fp32 | `G-VAL window PASS; G-FD PASS; G-VAL own PASS; branch A; N12 re-read: ...` | `n2v/verdict_n2v.json` | 2026-11-07 19:50 CST (line 136) |
| N1, weight order at birth | `NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR)` | `n1/verdict_n1.json` | 2026-11-07 14:55 CST (line 129) |
| N13, what RL wrote | `COMPLETE`; H_local REFUTED; H_diffuse SUPPORTED; H_attention REFUTED; H_artifact INDETERMINATE | `n13/verdict_n13.json` | 2026-11-08 05:31 CST (line 146) |
| N6, edit the young | DOES NOT COUNT: `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT`; L2b not launched | `n6/verdict_n6.json` | 2026-11-08 09:25 CST (line 154) |
| J8, second data-order check (`pilot0/`) | both steps in the pass branch | `pilot0/GPU_CHECK2.md` | 2026-11-05 16:05 CST (line 48) |
| J10/N22, a variable in the residual stream | NO-GO at T_15 | `n22/verdict_n22_t15.json` | 2026-11-06 12:05 CST (line 97) |

Also closed: the retag of the already-clean claim "the main job is the oldest" (`10_RETAG.md`; JOURNAL line 29) and the
Goodfire preprint note (`12_SHAPING_NOTE.md`; JOURNAL line 53).

## 3. What is running

Nothing on AWS. This worker's liveness check: `aws ec2 describe-instances --profile fuelos --region us-east-1` with the
states pending, running, shutting-down and stopping returned no instance at 2026-11-08 19:10:34Z. The last journal
entry says "six studies closed, nothing running" (JOURNAL 2026-11-08 09:30 CST, line 155). No executor is assigned to
any study; the last one (N6's) was asked to close the record and stop (JOURNAL line 153).

Left in place on purpose:
- One-shot reaper schedules for instances that already ended; they find nothing (JOURNAL line 48).
- The ECR image `shackleton:n13-v1` from the stopped Vast lane, about $4.2 per month while kept
  (`infra/vast/operations/2026-11-05_n13_lane.md` §6b).
- The orphan branch `gh-pages` and the private walkthrough artifact (JOURNAL line 102).

## 4. The final abstract and its check

- Abstract: `20_ABSTRACT.md`, FINAL 2026-11-08 09:30 CST, commit dd0556111 (JOURNAL line 155).
- Check: `20_ABSTRACT_CHECK.md` (draft 3: 175 numbers checked, 4 mismatched, all corrected), with the scripts
  `abscheck/check_abstract.py` and `abscheck/check_n6.py`. The later passes are in JOURNAL lines 150, 151 and 155 (the
  fourth pass: 31 items, all equal).

## 5. Decisions open for the user

1. **The rank test of the RL delta, named the highest-leverage next step** (`20_ABSTRACT.md` §The highest-leverage next
   step, line 64). Its words: "for each MLP matrix of the RL delta, a rank-r truncation (r = 16, 64, 256, design values)
   transplanted into the base, against a random subspace of equal rank and equal energy, on MATH-500 with GSM8K held out
   on the rank-64 pair, under N13's frozen harness, gates and controls". Hypothesis and refutation: "the gain sits in a
   low-rank component of each matrix's change (the rank-64 arm within δ of the RL model); refuted if it reads at the
   random-subspace control's level." Cost: "Cost at N13's measured rate (2.0–2.6 GPU-h per MATH-500 arm on one L40S,
   3.4–4.8 with GSM8K): six arms, 12–16 GPU-h, 15–20 with GSM8K on the rank-64 pair; the rank-r builds (one SVD per
   matrix) are not costed." Not specified, not launched. The N6 decision ranks it above N6's own next step (JOURNAL line
   153).
2. **The practical card.** As proposed in `HANDOFF_2026-11-07.md` §5: "add the RL model's top-1 % MLP delta to a
   different post-trained OLMo-3 7B model (for example the Instruct checkpoint) and read MATH-500 against that model's
   own score, the full delta and DARE at the same count; four arms, about 14 GPU-h; run ahead of N6 L2a if N13 says
   localized." N13 did not read localized: H_local REFUTED, top-k transplants 57.6–63.8 against the halfway line H 74.6
   (`n13/verdict_n13.json` hypotheses, arms). So the top-1 % transplant proposal is withdrawn (the head's decision, sent
   with this task; no earlier file states it). The journal's last word on the card: "the practical card and the code
   project wait on the user" (JOURNAL 2026-11-07 19:58 CST, line 137). The abstract's practical reading of the rank test: "Practical
   reading if it holds: the RL delta compresses to r × (d + n) numbers per matrix for transplant and merging; if it fails,
   the diffuse account stands at the entry level and DARE's 0.1 % sample is already the cheapest carrier measured here
   (85.2 of 92.0 on MATH-500; the 1 % sample reads 89.8 of 92.3 on GSM8K)" (`20_ABSTRACT.md` lines 74–76).
3. **The code-model project on the RL-Zero-Code checkpoint.** The only statement is the same journal line: "the
   practical card and the code project wait on the user" (line 137). No card or spec for it exists under
   `studies/next/`. Related: `allenai/Olmo-3.1-7B-RL-Zero-Code` is named in `07_PICKS/N13_MATH_TASK.md` line 217;
   `04_PRACTICAL.md` K2 (line 525) proposes merging RL-Zero-Math with RL-Zero-Code by neuron, with RL-Zero-Mix as the
   ceiling.
4. **The old program's result files.** `HANDOFF_2026-11-07.md` §5: "the roughly 1,200 untracked result files in
   clean-neurons (`git status`), whether to commit or ignore them; the user has not decided." JOURNAL 2026-11-05 10:42
   CST (line 31): "Pre-existing untracked results in this worktree (about 1,200 files: instruments/results, mechanics
   screen arrays up to 68 MB, realizednull discovery samples) are untouched and need the user's call."

Registered or proposed next steps that are not scheduled (no decision is pending on them):
- N6's registered step for no power, a larger eval set or more pairs: "not taken now" (`n6/STATE.md` line 7).
- J10: "A later format screen (easier expressions, about 0.3 GPU-h) is possible but not scheduled" (JOURNAL line 97).
- J13, the robot grounding gate: deferred until the N13 and M1 headline results (JOURNAL line 24); its card says "on the
  user's word" (`09_PLAN.md` §3 J13).
- N1's next instrument, a job-direction projection (JOURNAL line 127).
- `12_SHAPING_NOTE.md` §4 item 8 (SFT to DPO harvest, 3.4–3.8 GPU-h): "a new question, not scheduled" (JOURNAL line 53).

## 6. Standing rules

Copied from `HANDOFF_2026-11-07.md` §2 (verbatim where the user was quoted), with the later journal changes marked.

- User, 2026-11-05: "Remove the GPU limits." (JOURNAL line 68; `HANDOFF_2026-11-07.md` §2 spells it "Rremove the gpu
  limits") → no per-study GPU-hour ceiling; each launch's hard kill is 1.3 times its projection (N13 A27 and every later
  spec).
- User, 2026-11-05: AWS L40S only; Vast is reserved for the user's other agent ("I need to keep the vast resources for
  another agent"; JOURNAL line 63, which supersedes the 19:15 CST "just go to vast AI" of line 58). Exception, user
  2026-11-06: a memory-bound step may use a Vast H100 instead of a precision compromise ("in cases like these they can
  use vast.ai"; JOURNAL line 92); executors report such a shortfall as a lane decision.
- User: "go ahead. We want quality." (logged in JOURNAL line 24). Smaller gates are the head's call.
- User: plain human technical language, no made-up jargon; the two words banned on 2026-09-08 (listed in
  `~/.claude/CLAUDE.md`) and "null" as a noun never appear in output (say cancelled, does not count, chance level, no
  result; record or log). Code identifiers, file names and quoted commands are exempt.
- User, 2026-10-13: subagents are Opus only (`model: "opus"`), Fable orchestrates and reviews; at most two agents at once
  on the laptop by the user's CLAUDE.md (the user lifted it to four in this program, JOURNAL line 5; two is safe).
- User, 2026-10-24: never use claude.ai Artifact features. Lifted once, 2026-11-06, for the walkthrough (the user chose
  "CloudFront or Claude artifact", JOURNAL line 102); the artifact is https://claude.ai/artifact/9MZga4B81Ut4QmRhKB9Zba
  (private).
- Never `git add -A` or `git add .`; commit named files. Head's commits end with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Executors' commits end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the `Claude-Session:` line was dropped from 2026-11-07 14:00
  CST ("Executors' commits carry the Co-Authored-By line only from here, as the harness now asks", JOURNAL line 118).
- Laptop is a control plane: no models, weights, containers, archive extraction or log polling locally. Poll
  `metrics.json`, `pipeline_status.json`, STATUS; copy `job.log` once after a job has ended, for the cause.
- Capacity rule: each launch tries `--type g6e.2xlarge` every 180 s for six tries, then every 600 s with no time limit;
  VcpuLimitExceeded and RequestLimitExceeded retried the same way; loops staggered ≥ 60 s; any other error stops that
  launch and reports. The G quota is 32 vCPU (g6e.2xlarge = 8), shared with the user's other agent.
- Never read `research-evaluation/`, `../interp_research-audit`, `infra/operations/credits/`. Never touch `/Volumes`. No
  filesystem-wide searches.
- Blind protocol for N13 S4 (N13_SPEC.md §9.5): "no arm's accuracy is read before the verdict step; executors bank
  files and read only gates, G-COMPLETE counts, engine blocks and raw-example samples." For the other studies
  (`HANDOFF_2026-11-07.md` §6.2): "unblind once in each spec's §7 order; the test half stays unopened until freeze 2 is
  tagged."
- Executors never kill an instance (the J12 executor broke this once on 2026-11-07, recorded as a deviation; JOURNAL line
  107).
- Method rules, verbatim from `HANDOFF_2026-11-07.md` §2: "both sides of every gate evaluated on banked control files
  before launch; a gate comparing two computation paths needs a tolerance frozen from a control pair on those paths or is
  reported not gated; bitwise bars across paths fail (M2 A3, N6 A1/A2, N13 A31, and N2V's t14_G_CAUSAL on 2026-11-07);
  gradient passes use deterministic algorithms and the math attention kernel; read with one direction, edit along the
  other; nonlinear probe beside a failing linear readout; an independent reader's worktree must not expose labels
  through `git log`; report-first on an engine difference." Added after it: "compute the first-order prediction in fp32,
  by forward mode where the backward does not fit, and report the finite-difference check beside the zero ablation"
  (`n2v/READ_head.md` lines 51–53); journal stamps are read from the clock, never extrapolated (JOURNAL line 151); a
  positive control matched on weight size is not matched on realized effect (N6, JOURNAL line 153). The full list with
  incidents is `00_INDEX.md` §(d).
- Status is reported only from a liveness check (an instance ID or an artifact timestamp), never from silence (repo
  `CLAUDE.md`, Process rule 3).

## 7. Known pitfalls

From `HANDOFF_2026-11-07.md` §7 and the journal, still true:
- "A new agent often ends its first turn with an empty summary and "No transcript found" on resume: respawn it with the
  same brief; tell it that its first action is a tool call."
- An agent from before a restart may still be resumable; send work only to the current one (the head's error at JOURNAL
  line 129).
- "EC2 RequestLimitExceeded when loops fire together: stagger by 60 s."
- The launcher writes every launch's code tarball to one fixed path, `/tmp/pg_code.tgz`, so two sessions launching at
  once can corrupt each other's upload (JOURNAL line 36).
- "Every bitwise bar across two computation paths has failed in this program": freeze a tolerance from a control pair
  and keep a hash check for the "differs by one step" failing input.

## 8. Branches and worktrees

The program's work now lives on branch `clean-neurons`, the head is cleaning up the merged branches and worktrees in
parallel, and the record of that cleanup is `studies/next/11_HOUSEKEEPING.md`.
