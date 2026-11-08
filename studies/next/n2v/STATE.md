# N2V state

Spec: `studies/next/07_PICKS/N2V_SPEC.md`. Offline gate: `studies/next/07_PICKS/N2V_OFFLINE_GATE.md`. Science:
`studies/next/n2v/JOURNAL.md`. Operations (at launch): `infra/aws_pg/operations/2026-11-<dd>_next_n2v.md`.

## Where it stands (2026-11-07)

- Freeze 1: tag `n2v-freeze-1` (annotated, tag object `76840b83d`) on commit `f977f93a3`, 2026-11-07T15:30:31Z; dry run
  `N2V_DRYRUN_OK`, 75 checks, 0 failed (`studies/next/n2v/offline_gate/dryrun_f1.log`).
- Freeze 1a: tag `n2v-freeze-1a` (tag object `baa19e63f`) on `cd628c8a8`, 2026-11-07T16:06:06Z (the coordinator, spec §0.6 item 8): parts core,fd,jvp, so G-VAL own has its prediction; job
  budget 700 min; hard kill 763 min. Dry run `N2V_DRYRUN_OK`, 76 checks (`studies/next/n2v/offline_gate/dryrun_f1a.log`).
- Projection (`studies/next/n2v/projection.json`): 6.53–9.78 GPU-h on one g6e.2xlarge ($14.63–21.91), 9,870 fp32 plain
  rows and 3,948 forward-mode rows; hard kill 763 min (1.3 × 9.78 h); the job's stop rule parks at 0.9 × 700 min.
- Inputs: `results/next_n2v/inputs/{units.json, n12_freeze.jsonl, n12_test.jsonl}` uploaded 2026-11-07T15:30Z and
  read back: 3 of 3 sha256 equal to `n2v_score.UNITS_SHA` and `N12_SHA`. `results/next_n2v/score-01/` holds nothing.
- score-01 (`i-03754787362f997e8`, launched 16:06:49Z on `n2v-freeze-1a`) stopped at `t14_G_CAUSAL`, STATUS `exit=1`
  at 16:44:52Z: 428 of 963 changed rows equal the clean row bitwise before i0. The attempt does not count. Nothing is
  unblinded. The next step is the coordinator's decision (see the handoff below).
- A1 and `n2v-freeze-1b` (2026-11-07, third J12 executor): the cause and the fix; the relaunch is score-02 (below).

## score-02 ended; freeze 2 (2026-11-08)

score-02 (`i-0537218438154119c`) ended 00:44:44Z, STATUS `exit=0`, 42 gates passing, nothing parked (journal). Freeze
files banked under `banks/score-02/n2v/`; `read/freeze_read.json`; A2 (freeze reads the freeze half only);
`thresholds.json` (sha256 `cdd2ed26…`); tag `n2v-freeze-2` (tag object `94de754c6`, commit `0a3742e43`, 00:53:08Z).
The verdict ran once at 00:53:22Z: `verdict_n2v.json`, headline `G-VAL window PASS; G-FD PASS; G-VAL own PASS;
branch A; N12 re-read: H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal REFUTED; H_bridge UNDECIDED; H_dt
SUPPORTED`; §12 filled. `losses/` stays on S3 (93.5 MB). Step 5 done 2026-11-08T01:50Z: `READ_n2v.md` (reader, branch `n2vread`) and `READ_head.md` (head) merged into bio2; the labels
stand; N2V is closed. `losses/` stays on S3 (93.5 MB, sha256s in `READ_n2v.md` §3).

## score-02 on `n2v-freeze-1b` (A1), the launch exactly (third J12 executor, 2026-11-07)

A1 (spec §14): score-01's G-CAUSAL failed because the fp32 loss read each row's logits in place, at a 16-byte alignment
that alternates with the batch slot's parity; `ce_pos` now copies each row first; G-CAUSAL records per-row differences;
report `slot_control` at T_14. Dry run `N2V_DRYRUN_OK`, 83 checks (`offline_gate/dryrun_a1.log`). Manifest
`infra/aws_pg/steps/next_n2v_score-02.json` (parts core,fd,jvp, job budget 700 min); hard kill 763 min; projection
6.53-9.78 GPU-h ($14.63-21.91). `SP` is the scratchpad named in the handoff below.

Before: `./infra/aws_pg/pg ps`, `./infra/aws_pg/pg audit`; `results/next_n2v/score-02/` has no object version
(`aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n2v/score-02 --profile fuelos --region us-east-1`);
`git status --porcelain --untracked-files=no` prints nothing under `infra/`; HEAD's `infra/` equals `n2v-freeze-1b`'s.

```
cd /Users/abaybektursun/projects/interp_research-bio2
aws s3 cp infra/aws_pg/steps/next_n2v_score-02.json s3://pg-lineage-769063704696/results/next_n2v/score-02/steps.json --profile fuelos --region us-east-1
aws s3 cp s3://pg-lineage-769063704696/results/next_n2v/score-02/steps.json - --profile fuelos --region us-east-1 | cmp - infra/aws_pg/steps/next_n2v_score-02.json
nohup $SP/bio2/n2vl2/launch/tries.sh > $SP/bio2/n2vl2/launch/loop.out 2>&1 &
```

`tries.sh` runs, every 180 s for six tries and then every 600 s with no time limit (VcpuLimitExceeded and
RequestLimitExceeded retried; any other error stops the loop with OTHER in `tries.log`):

```
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_n2v/score-02/steps.json" \
    --no-weights --type g6e.2xlarge --s3-prefix results/next_n2v/score-02 --budget-min 763 --json --detach
```

On launch: `nohup $SP/bio2/n2vl2/poll.sh <instance-id> > /dev/null 2>&1 &` (every 120 s; `$SP/bio2/n2vl2/poll/poll.log`,
`events.log`, each new gate and report saved to `gates/`). Read first: `reports.slot_control` (predicted: logits and
copy 0 at every slot; view 0 at slot 2, above 0 at slots 1 and 3; offsets 0, 8, 0, 8), then `t14_G_CAUSAL` with its
per-row record, `points.14.jvp_max_mem_gb`, `measured`.

## The launch, exactly (run from `/Users/abaybektursun/projects/interp_research-bio2` on tag `n2v-freeze-1` or a later
## commit whose `git diff n2v-freeze-1 -- infra/` is empty)

Before: `./infra/aws_pg/pg ps` and `./infra/aws_pg/pg audit` (no stray instance; a free G slot: four g6e.2xlarge fit the
32-vCPU G quota); `results/next_n2v/score-01/` has no object version
(`aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n2v/score-01 --profile fuelos --region us-east-1`);
`git status --porcelain --untracked-files=no` prints nothing under `infra/` (the launcher tars the working tree's
`infra/`).

```
B=s3://pg-lineage-769063704696
cd /Users/abaybektursun/projects/interp_research-bio2
aws s3 cp infra/aws_pg/steps/next_n2v_score-01.json $B/results/next_n2v/score-01/steps.json --profile fuelos --region us-east-1
```

The capacity rule (the coordinator, 2026-11-07: `--type g6e.2xlarge` only; every 180 s for six tries, then every 600 s
with no time limit; VcpuLimitExceeded and RequestLimitExceeded retried; any other launch error stops and is reported;
the script is `$SP/bio2/n2vl/launch/tries.sh`):

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
D=$SP/bio2/n2vl/launch; mkdir -p $D
n=0
while true; do
  n=$((n+1)); start=$(date -u +%s)
  echo "try $n start $(date -u -r $start +%Y-%m-%dT%H:%M:%SZ)" >> $D/tries.log
  ./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_n2v/score-01/steps.json" \
      --no-weights --type g6e.2xlarge --s3-prefix results/next_n2v/score-01 --budget-min 763 --json --detach > $D/try$n.out 2>&1
  rc=$?
  echo "try $n end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc last: $(tail -1 $D/try$n.out | cut -c1-300)" >> $D/tries.log
  if [ $rc -eq 0 ]; then echo "LAUNCHED on try $n"; cat $D/try$n.out; break; fi
  if ! grep -qE "VcpuLimitExceeded|RequestLimitExceeded" $D/try$n.out; then
    echo "OTHER LAUNCH ERROR on try $n"; tail -5 $D/try$n.out; break; fi
  if [ $n -lt 6 ]; then next=$((start + 180)); else next=$((start + 600)); fi
  now=$(date -u +%s); [ $next -gt $now ] && sleep $((next - now))
done
```

Watch every 120 s: `out/score/n2v/metrics.json`, `out/pipeline_status.json`, `STATUS`, the EC2 state; never `job.log`.
The first values to read: `gates.G_SAME`, `gates.INPUTS`, `gates.G_CORPUS`, `gates.K_A` (40 pairs within 1e-6 nats),
`gates.t14_G_FP32`, `gates.t14_G_CAUSAL`, `points.14.jvp_max_mem_gb`, `measured` (seconds per plain row; 1.18 center, 1.40 high), `projection`.

After `STATUS`: spec §7 steps 3–5 (freeze half first; `n2v freeze`; tag `n2v-freeze-2`; then the test half and the
verdict once).

## Handoff 2026-11-07

Written 2026-11-07T17:58Z before a terminal restart. The session that wrote it cannot be resumed. `SP` is the old
scratchpad, which stays on disk:
`SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.

### score-01 (ended: stopped at t14_G_CAUSAL)

| field | value |
|---|---|
| instance | `i-03754787362f997e8`, g6e.2xlarge (1× L40S), us-east-1d, `ami-0cee78ee4b1c24614`, on-demand, $2.24 per hour; `LaunchTime` 2026-11-07T16:06:49Z, first try |
| prefix | `s3://pg-lineage-769063704696/results/next_n2v/score-01/`; outputs under `out/score/n2v/` |
| code | tag `n2v-freeze-1a` (tag object `baa19e63f`) on `cd628c8a8`; manifest `--parts core,fd,jvp --budget-min 700` |
| hard kill | `pg run --budget-min 763`; reaper deadline 2026-11-08T04:59:49Z. It did not fire: the job stopped itself. `pg audit` at 17:55:35Z lists `pg-reap-i-03754787362f997e8` as an orphan reaper (schedule for a terminated instance) |
| end | job start 16:08:39Z (`started_utc`); last `updated_utc` 16:44:28Z, elapsed 2,148.9 s; `STATUS` = `exit=1` (S3 time 16:44:52Z); EC2 `shutting-down` at the 16:46:12Z poll; the instance was no longer listed at 17:53Z |
| progress | T_14 only. Download 935.4 s at 15.6 MB/s (14.6 GB); `fixed_s` 1,001.6; 1,070 plain rows in 1,089.3 s (`plain_s`); forward-mode rows not reached |
| gates passed | `G_SAME` (torch 2.12.0+cu130, transformers 5.9.0, L40S), `INPUTS`, `G_CORPUS`, `t14_G_REV`, `t14_G_CFG`, `K_A` (40 of 40 pairs exact, max abs diff 0.0), `t14_G_FP32` (parameters float32; failing input bfloat16 fails) |
| gate failed | `t14_G_CAUSAL`: `n_rows` 963, `n_bitwise_before_i0` 428. The failing input failed as registered: 2 of the zero rows equal the clean row through i0 |
| not measured | `points.14.jvp_max_mem_gb` (forward mode not reached); `measured` null; `complete` false; no `n2v_freeze.jsonl`, `n2v_test.jsonl` or samples written |
| stop | `m2_score.Run.gate` asserts `pass`, `AssertionError: t14_G_CAUSAL failed`; `[pipeline] ABORT — step score failed rc=1` |
| diagnostics | `studies/next/n2v/diag/score-01/` (metrics.json, pipeline_status.json, heartbeat.jsonl, dmon.log, STATUS) |
| instance-hours, cost | at most 0.66 h (16:06:49Z to the 16:46:12Z poll that saw `shutting-down`), at most $1.47; 0.66 GPU-h |

What the spec says: G-CAUSAL is a stop gate (§3.2, §6; §0 assumption 4: "a row's loss does not depend on its batch
companions (same shape, same kernel)", prior 0.9). The dry run passed it in the tiny world on CPU. On the L40S in fp32,
535 of 963 changed rows (zero, fd and probe) differ from their unit window's clean row somewhere before i0.

What is not known: the job banks only the count. It does not record which rows differ, by how much, or in which batch
they sat. The candidate causes are untested and none is preferred:
- the clean row and its changed rows sat in batches of different shape, so cuBLAS or SDPA chose a different kernel;
- a kernel splits its reduction depending on the companions;
- TF32 or another matmul precision setting is on.

Telling them apart needs per-row max |ℓ_changed(j) − ℓ_clean(j)| over j < i0, plus each row's batch index and batch
size. That is a code change (a freeze 1b) and a fresh prefix (`score-02`). Both are the coordinator's decision. Nothing
is relaunched without it.

### The watch (for the next attempt, `score-NN`)

The old watch `$SP/bio2/n2vl/poll.sh <instance-id>` has the prefix `score-01` written into it. Copy it and change `B`
for a new attempt. It polls every 120 s, logs `poll.log` and `events.log` and saves each new gate to `gates/`. One poll
by hand (never `job.log`):

```
B=s3://pg-lineage-769063704696/results/next_n2v/score-NN
aws s3 cp $B/out/score/n2v/metrics.json - --profile fuelos --region us-east-1 | python3 -c "import json,sys; m=json.load(sys.stdin); g=m['gates']; print(m['phase'], m['elapsed_s'], len(g), [k for k,v in g.items() if not v['pass'] or v.get('failing_input',{}).get('pass')], m.get('parked'), m['complete'], m.get('measured'), m['updated_utc'])"
aws s3 cp $B/out/pipeline_status.json - --profile fuelos --region us-east-1
aws s3 cp $B/STATUS - --profile fuelos --region us-east-1
aws ec2 describe-instances --instance-ids <instance-id> --profile fuelos --region us-east-1 --query 'Reservations[].Instances[].State.Name' --output text
```

Gates to read, in this order:
1. `gates.G_SAME`, `gates.INPUTS`, `gates.G_CORPUS`, `gates.K_A` (40 pairs within 1e-6 nats).
2. `gates.t14_G_FP32` (parameter dtypes float32).
3. `gates.t14_G_CAUSAL` (every changed row bitwise equal to the clean row before i0; the failing input must differ).
4. `points.14.jvp_max_mem_gb` (forward-mode memory at T_14 on the 48 GB L40S).
5. `measured` (seconds per plain row; 1.18 center, 1.40 high) and `projection`.
6. Then each later point's gates and `parked`.

### Unblinding order (spec §7 steps 3–5), as it will be run

Step 3 runs only after STATUS `exit=0`, `complete` true, every gate passing and no failing input passing. Commands run
from the worktree root, with `<attempt>` being the attempt's name:

```
B=s3://pg-lineage-769063704696/results/next_n2v/<attempt>/out/score/n2v
D=studies/next/n2v/banks/<attempt>/n2v
mkdir -p $D/samples
aws s3 cp $B/metrics.json $D/metrics.json --profile fuelos --region us-east-1
aws s3 cp $B/n2v_freeze.jsonl $D/n2v_freeze.jsonl --profile fuelos --region us-east-1
aws s3 cp $B/samples/raw_examples_freeze.jsonl $D/samples/raw_examples_freeze.jsonl --profile fuelos --region us-east-1
```

- Open the freeze examples and answer in `studies/next/n2v/JOURNAL.md`:
  1. The decoded exemplars are N12's.
  2. The rows' losses before i0 equal the clean row's.
  3. The carrier's five largest per-position changes sit at or after the zeroed indices.
  4. The probe is at the fp32 floor.
  5. Give one pair with |ΔL − pred| small and one with it large, with their tokens.
- Freeze 2: `PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n2v freeze --bank studies/next/n2v/banks/<attempt> --out studies/next/n2v/thresholds.json`.
- Commit by name, with a subject that names no label. Tag `git tag -a n2v-freeze-2 -m "N2V freeze 2: thresholds from <attempt>'s freeze half"`.
  Record the tag object, commit and UTC in a following commit.

Step 4: pull `n2v_test.jsonl` and `samples/raw_examples_test.jsonl` into `$D`, then run the verdict once:

```
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n2v verdict --bank studies/next/n2v/banks/<attempt> --thresholds studies/next/n2v/thresholds.json --out studies/next/n2v/verdict_n2v.json
```

The verdict refuses if the sha256 of the freeze records changed. It reads the gates through `m2.gates_ok`.

Step 5: a fresh Opus reader (`model: "opus"`; at most two agents at once) works in a worktree made from the
`n2v-freeze-2` commit. It recomputes every pair's ΔL from `losses/tNN.npz`, the four validations and G-ROOM with its
own code, and on branch A it reruns the re-reading. The head's read is the different-model read. The result document
comes after.
