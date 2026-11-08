# N1 state

Spec: `studies/next/07_PICKS/N1_SPEC.md`. Offline gate: `studies/next/07_PICKS/N1_OFFLINE_GATE.md`. Science:
`studies/next/n1/JOURNAL.md`. Operations: `infra/aws_pg/operations/2026-11-06_next_n1.md`.

## weights-03 ended; freeze 2 (third J12 executor, 2026-11-07)

weights-03 ended 19:31:55Z with STATUS `exit=0`, `complete` true, 49 gates passing (journal). Freeze-half files banked
under `banks/weights-03/n1/`; `read/freeze_read.json`; `thresholds.json` (sha256 `fad4fe71…`); tag `n1-freeze-2`
(tag object `88517fc7d`, commit `6709f754e`, 19:34:42Z). The verdict ran once at 19:35:10Z: `verdict_n1.json`,
headline `NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR)`; §12 filled. Test files
banked. `stats.npz` stays on S3 (sha256 in the journal). Open: spec §7 step 5, the fresh Opus reader in a worktree made
from `n1-freeze-2` (`git worktree add ../interp_research-n1read n1-freeze-2`), recomputing r, z, e and the labels from
`s3://pg-lineage-769063704696/results/next_n1/weights-03/out/weights/n1/stats.npz`; the head spawns it.
**Note 2026-11-08 (index worker):** step 5 is done; the reader's `READ_n1.md` and the head's `READ_head.md` are merged and N1 is closed, recorded in `studies/next/JOURNAL.md` 2026-11-07 14:55 CST.

## Handoff 2026-11-07

Written 2026-11-07T17:58Z before a terminal restart. The session that wrote it cannot be resumed. `SP` below is the old
scratchpad, which stays on disk:
`SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.

### weights-03 (running at 17:55Z)

| field | value |
|---|---|
| instance | `i-02f721a58a0643922`, r7i.2xlarge, us-east-1d, on-demand, $0.5292 per hour; `LaunchTime` 2026-11-07T15:56:51Z |
| prefix | `s3://pg-lineage-769063704696/results/next_n1/weights-03/`; outputs under `out/weights/n1/` |
| code | tag `n1-freeze-1b`, commit `100e56b68` (A2: job budget 300 min, nothing else changed from `n1-freeze-1a`) |
| job start | `started_utc` 2026-11-07T15:58:29Z (pipeline 15:57:37Z) |
| job's own stop | the job asserts after each checkpoint that elapsed < 0.95 × 300 min = 17,100 s, i.e. 2026-11-07T20:43:29Z; past it the job stops with nothing written (every bracket is needed; nothing parks) |
| hard kill | `pg run --budget-min 320`; the launcher's reaper deadline is 2026-11-07T21:26:51Z |
| progress (poll 17:55:11Z) | T_0–T_8 done (9 of 16 checkpoints), elapsed 5,871.0 s at `updated_utc` 17:36:20Z; 27 gates, all pass, no failing input passed; `complete` false; STATUS empty; EC2 running |
| seconds per checkpoint | T_0–T_8: 116.3, 227.3, 280.4, 582.6, 788.7, 807.3, 1,073.1, 911.4, 1,074.9; the download rate fell from 142.2 MB/s (T_0) to 14.1 MB/s (T_6, T_8) |
| projection | at the T_6–T_8 mean (1,019.8 s): 5,871 + 7 × 1,019.8 + 300 s of records ≈ 13,310 s, so the job finishes about 19:40Z. The job's own stop fires if the seven remaining checkpoints average more than (17,100 − 5,871 − 300) / 7 = 1,561 s. T_9 had taken more than 1,130 s at 17:55Z |
| rule | the coordinator's decision, 2026-11-07T16:05Z: a projection that the job cannot finish is reported, and the job's hard kill fires, unless the coordinator authorizes a kill. Never kill weights-03 on your own |
| instance-hours, cost | 1.98 h at 17:55:35Z (`pg ps`), about $1.05 so far. N1 before it: weights-01 0.57 h ($0.30), weights-02 0.52 h ($0.28) |

### The watch

The detached watch `$SP/bio2/n1l3/poll.sh` polls every 120 s and logs to `$SP/bio2/n1l3/poll/poll.log`. If
`ps aux | grep n1l3/poll.sh` shows nothing, restart it with `nohup $SP/bio2/n1l3/poll.sh > /dev/null 2>&1 &`. One poll by
hand (never `job.log`):

```
B=s3://pg-lineage-769063704696/results/next_n1/weights-03
aws s3 cp $B/out/weights/n1/metrics.json - --profile fuelos --region us-east-1 | python3 -c "import json,sys; m=json.load(sys.stdin); g=m['gates']; print(m['phase'], m['elapsed_s'], len(g), [k for k,v in g.items() if not v['pass'] or v.get('failing_input',{}).get('pass')], m['complete'], m['updated_utc'])"
aws s3 cp $B/out/pipeline_status.json - --profile fuelos --region us-east-1
aws s3 cp $B/STATUS - --profile fuelos --region us-east-1
aws ec2 describe-instances --instance-ids i-02f721a58a0643922 --profile fuelos --region us-east-1 --query 'Reservations[].Instances[].State.Name' --output text
```

Files to read while it runs: `out/weights/n1/metrics.json` (`phase`, `elapsed_s`, `points.<t>.seconds`,
`points.<t>.download`, `gates`, `reports.tNN_K_B`, `complete`, `updated_utc`), `out/pipeline_status.json` and `STATUS`.
Nothing else is opened before STATUS.

### Spec §7 steps 3–5, as they will be run

Step 3 runs only if STATUS reads `exit=0`, `metrics.json` has `complete` true, every gate passes and no failing input
passes. Otherwise the attempt does not count and is reported to the coordinator. Commands run from the worktree root:

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
B=s3://pg-lineage-769063704696/results/next_n1/weights-03/out/weights/n1
D=studies/next/n1/banks/weights-03/n1
mkdir -p $D/samples
aws s3 cp $B/metrics.json $D/metrics.json --profile fuelos --region us-east-1
aws s3 cp $B/n1_freeze.jsonl $D/n1_freeze.jsonl --profile fuelos --region us-east-1
aws s3 cp $B/samples/raw_examples_freeze.jsonl $D/samples/raw_examples_freeze.jsonl --profile fuelos --region us-east-1
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n1/read/freeze_read.py --bank studies/next/n1/banks/weights-03 --out studies/next/n1/read/freeze_read.json
```

- Open the ten freeze examples in `raw_examples_freeze.jsonl`. Answer the four questions of §7 step 3 in
  `studies/next/n1/JOURNAL.md`, citing `freeze_read.json`:
  1. Does each example unit's column norm at T_(k−1) and T_k equal the banked one?
  2. Does the profile of r over the 15 brackets have the shape of the bracket lengths, for carriers and twins alike?
  3. Is the planted PC share near 1 and the size-0 share near 0.05?
  4. Give one birth with a large and one with a small input-side excess, with their exemplars.
- Freeze 2, which reads the freeze half only:
  `PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n1 freeze --bank studies/next/n1/banks/weights-03 --out studies/next/n1/thresholds.json`.
  It prints `N1_FREEZE_OK <sha256>` and refuses if the file exists.
- Commit by name: the three bank files, `thresholds.json`, `read/freeze_read.json`, `JOURNAL.md` and this file. The
  commit subject must name no label, because step 5's reader worktree is made from it. Tag
  `git tag -a n1-freeze-2 -m "N1 freeze 2: thresholds from weights-03's freeze half"`. Record the tag object, commit and
  UTC in a following commit.

Step 4 is the verdict, run once:

```
aws s3 cp $B/n1_test.jsonl $D/n1_test.jsonl --profile fuelos --region us-east-1
aws s3 cp $B/samples/raw_examples_test.jsonl $D/samples/raw_examples_test.jsonl --profile fuelos --region us-east-1
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n1 verdict --bank studies/next/n1/banks/weights-03 --thresholds studies/next/n1/thresholds.json --out studies/next/n1/verdict_n1.json
```

It prints the headline and refuses if `verdict_n1.json` exists or if the sha256 of `n1_freeze.jsonl` changed. Bank the
test files, `verdict_n1.json` and the journal entry.

Step 5 is the independent read. A fresh Opus reader (`model: "opus"`; at most two agents at once) works in a worktree
made from the `n1-freeze-2` commit, whose subject names no label. It recomputes r, z, e and the labels from
`s3://pg-lineage-769063704696/results/next_n1/weights-03/out/weights/n1/stats.npz` with its own code. `stats.npz` stays
on S3. The head's read is the different-model read.

### What `freeze_read.py` assumes that is unchecked

`studies/next/n1/read/freeze_read.py` was written before any real record existed and has never been run. It assumes the
following:

- Each line of `raw_examples_freeze.jsonl` has `birth` and the kinds `real`, `dt`, `pb` (`dt` or `pb` may be missing).
  Each kind has `half`, `unit`, `k`, `layer`, `neuron`, and per side (`in`, `out`, `gate`, `up`) the 15-long lists `r`,
  `z`, `cos` (bracket j at index j − 1), plus `e_window`. `in` and `out` also have `e_k` and `e_by_bracket`, and `real`
  has `ceiling`.
- `studies/next/n1/kb_bank.json` has `values` as `[layer, neuron, t, column norm]` rows.
- `metrics.json` has `reports.tNN_K_B` with `pass`, `max_rel`, `n`. The PC gates end in `_PC` and carry `in`, `out`,
  `in_size0` and `out_size0`.
- `studies/next/n1/samples/raw_examples_zero_gpu.json` is a list keyed by `birth`, with per kind `colnorm_pre` and
  `colnorm_birth`, and `exemplar[1]` as the exemplar text.
- `m2.STEPS` has 16 entries, so the bracket step counts are `np.diff(m2.STEPS)`.
- Question 1 is answered through the banked values and the K-B report at T_(k−1) and T_k. The job's own column norm
  per example unit is not read unless the example record carries it.
- The Spearman correlation ranks with `argsort`, so ties are not averaged.

If any key differs, the script stops with a KeyError. Fix the script to the real schema, not the other way round, and
commit the fix before freeze 2.
