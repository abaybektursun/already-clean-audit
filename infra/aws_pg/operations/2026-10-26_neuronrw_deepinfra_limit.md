# 2026-10-26 — NEURONRW RE-EXPLAIN: DeepInfra read timeouts, then the user-set spending limit (HTTP 402)

Scope: API infrastructure and cost only (CLAUDE.md Process 6). Counts and science state are in
`studies/neuronrw/JOURNAL.md` (entry 2026-10-26T05:37:24Z).

## What happened
- Client `infra/pg/neurons_llm.py run-many`, settings `infra/pg/settings/deepinfra_neuronrw_judge.json`
  (`openai_compat`, `https://api.deepinfra.com/v1/openai`, key env `NEURONS_API_KEY_1`), concurrency 8, explain dir
  `bank/rw/reexplain` in the `neurons/rw` worktree.
- 05:10:11Z–05:23:01Z: 1,369 answers at a median latency of about 4.2 s.
- 05:23:01Z–05:33:35Z: the 8 requests in flight received no bytes; each hit the 600 s urlopen read timeout
  (`TimeoutError: The read operation timed out`, not a retryable class in the client), and the run raised.
- 05:33:53Z resume: 50 answers, then `HTTP 402 {"message": "inference prohibited, you have reached user-set limit"}`.
- 05:35:38Z one more resume to confirm: 402 again, 0 answers.

## Cost
- Usage files × listed price ($2 / M input, $10 / M output): 22,200,123 input + 490,241 output tokens = $49.30 for the
  1,419 answered requests. The 402 requests are refused before inference.

## State and next action
- No requests lost: answered lines are kept and a rerun skips them. 1,255 requests remain (about $44 at the measured
  mean). The judge steps after them (SPEC §12: about 14,400 Arm 0 and 9,900 Arm 2 calls) need further budget.
- The limit is set at https://deepinfra.com/dash/billing by the account owner. After it is raised:
  `set -a; . ~/.config/interp_research/keys.env; set +a; python -m infra.pg.neurons_llm run-many --explain bank/rw/reexplain
  --batches rx-G4,rxv-G4,rx-O7,rxv-O7 --settings infra/pg/settings/deepinfra_neuronrw_judge.json --concurrency 8`.
- No NEURONRW instance was launched. At 05:38:01Z `pg ps` lists two instances of another study's job (`evilqwen`,
  i-04e451190bb308ec9, i-0055f2fa839b0a153) and `pg audit` reads "2 running, 2 reapers, 0 orphans, 0 reaper-less / clean".

## Addendum 09:42Z — preflight instance lifecycle
- `pre_q4b/1`: i-04eabec1d94728a7e, g6e.4xlarge (g6e.2xlarge had no capacity), RunInstances 09:27:13Z, job exit 1 at
  09:34:53Z, instance-initiated shutdown, `terminated` by 09:42:00Z (about 0.25 instance-hours).
- Its reaper `pg-reap-i-04eabec1d94728a7e` (deadline 10:22:13Z) was left as an orphan by the self-termination, as in
  `2026-10-25_neuronoff_orphan_reapers.md`; removed at 09:42Z with `./infra/aws_pg/pg kill i-04eabec1d94728a7e`.
  `pg audit` → "0 running, 0 reapers, 0 orphans, 0 reaper-less / clean".

## Addendum 10:56Z — preflight attempt 2 lifecycle
- `pre_q4b/2`: i-048d967ab24b75477, g6e.4xlarge (no g6e.2xlarge capacity), RunInstances 10:41:42Z, `STATUS exit=0`,
  `terminated` by 10:56:38Z (about 0.25 instance-hours). Orphan reaper `pg-reap-i-048d967ab24b75477` removed with
  `pg kill` at 10:56Z; `pg audit` clean. Cumulative NEURONRW instance-hours: about 0.50.

## Addendum 13:20Z — preflight 3, Arm 0 cells, capacity
- `pre_q4b/3` i-0371d8098d3386111 (g6e.2xlarge) 11:28:02Z–about 11:42Z; `arm0_q4b/1` i-03e300d15a7bc3154 (g6e.4xlarge)
  11:49:42Z–≤12:41:42Z; `arm0_g4/1` i-03658137aa0e4f225 (g6e.4xlarge) 11:51:44Z–≤13:07Z; `arm0_o7/1` i-0e7a49acab8cb02ab
  (g6e.2xlarge) 12:43:23Z–≤12:56:34Z. Each left an orphan reaper after self-termination; each removed with `pg kill`.
- Capacity: 11:37:42Z and 11:43:27Z no g6e.2xlarge or g6e.4xlarge in any AZ (InsufficientInstanceCapacity); launched at
  11:49:44Z. 12:58:24Z, 13:04:46Z, 13:11:15Z no g6e.2xlarge in any AZ for the O7 P1 re-run (pinned to g6e.2xlarge by
  SPEC P1); retry loop stopped at 13:13Z, nothing created. `pg audit` 13:13Z: 0 running, 0 reapers, clean.

## Addendum 14:12Z — O7 Arm 0 attempt 2
- 13:44:04Z no g6e.2xlarge or g6e.4xlarge capacity; 13:51:06Z `arm0_o7/3` launched on i-0cb264e2a01a552a5 (g6e.4xlarge,
  allowed by A24); STATUS 14:03:49Z; terminated by 14:10:18Z; orphan reaper removed with `pg kill`; `pg audit` clean.

## Addendum 15:20Z — Arm 1 cells
- 14:33Z both launches found no g6e capacity; 14:38Z the Q4B retry failed uploading the code tarball (`S3UploadFailedError
  … IncompleteBody`), nothing created. `arm1_q4b/1` i-0623b96ca094a7ebd (g6e.4xlarge) 14:43:28Z–about 14:52Z;
  `arm1_g4/1` i-0f43b67a0a7dd5744 (g6e.4xlarge) 14:48:26Z–≤15:19Z. Both reapers removed with `pg kill`; `pg audit` clean.

## Addendum 16:22Z — Arm 1 re-run and Q4B Arm 2
- 15:37–15:38Z no g6e capacity for `arm2_g4/1` and `arm1_q4b/2`; the `arm2_g4/1` retry loop was stopped (cell held).
  `arm1_q4b/2` i-07260b818ef1f037a 15:45:27Z–≤16:22Z; `arm2_q4b/1` i-066ce6eccb0cf4430 15:48:20Z–≤16:22Z (both g6e.4xlarge).
  Reapers removed with `pg kill`; `pg audit` clean.

## Addendum 17:50Z — Arm 2 G4 and O7
- `arm2_o7/1` i-0ec4b56bb8a9aface (g6e.2xlarge) 16:47:23Z–≤17:28Z; `arm2_g4/1` i-0f79c4e824803b6e2 (g6e.4xlarge, after one
  no-capacity try) 16:56:39Z–≤17:50Z. Reapers removed with `pg kill`; `pg audit` clean.
- DeepInfra: one A2 guess request returned an empty answer with finish_reason content_filter on three attempts (declined).
  Judge spend $22.50 of the $28 stop; the A2 O7 step was refused by the stop rule before any request was sent.

## Addendum 18:13Z — close
- Bank `bank/rw/` verified against S3 file by file (579 files, 0 mismatches; manifest `results/neurons/rw/BANK_SHA256.txt`).
- `pg audit` 18:13:36Z: 0 running, 0 reapers, 0 orphans, clean. NEURONRW used 13 instances, 5.99 instance-hours.

## Addendum 19:35Z — A28 stage
- DeepInfra limit raised by $20 (A28; stop $48). O7 A2: 1,336 calls, $8.69, 0 declined. Judge total $31.19.
- No GPU instance launched in this stage; `pg audit` 19:35:20Z clean; bank manifest unchanged and verified.
