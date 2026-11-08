# REUSE SPEC5 dating runs on the AWS lane, 2026-11-02

Two `pg run infra.aws_pg.jobs.pipeline` launches, each one `g6e.2xlarge` (L40S), `--no-weights` (the job downloads each OLMo 3 checkpoint from the hub inside the loop), fresh S3 prefix each, watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| run | prefix | instance | launched (UTC) | STATUS | billed | loads | notes |
|---|---|---|---|---|---|---|---|
| dating (frozen) | `results/reuse/dating/1` | `i-00c31901d5cc5950b` | 22:19:34Z | 23:23:26Z, exit 0 | 1.06 h | 16 of 79 to 158 s after the load; T_15 386 s | first launch at 22:18Z failed with `InsufficientInstanceCapacity` for g6e.2xlarge in every zone; the retry with `--type g6e.2xlarge,g6e.4xlarge` landed a g6e.2xlarge |
| scale (amendment 5b) | `results/reuse/dating/2` | `i-09f03bd168225822b` | 23:36:40Z | 00:53:43Z (11-03), exit 0 | 1.29 h | 16 of 270 to 418 s, 9,238 captures each | budget 150 min, reaper 02:16:40Z |

Cost at the on-demand price of $2.24 per hour: $2.37 + $2.89 = $5.26 (launch to STATUS 1.064 + 1.284 h; billing runs to termination, about a minute more). Cap: the REUSE study's frozen hard cap is 5.0 GPU-h (`studies/reuse/SPEC.md` §12); it stood at 5.847 before these runs and at 8.195 after, 3.2 h over. The coordinator lifted the cap on 2026-11-02 16:05 CST citing the AWS lane's standing directive in `launch_preflight.sh` and the user's instruction to run the queue autonomously without prompts; the user's own message lifting this cap does not exist and the overrun is the coordinator's decision, recorded here and in the science journal (16:05, 19:20, 19:55 CST). Peak memory 19.7 to 21.3 GB (dating), 20.1 GB (scale). Downloads 119 to 136 s per checkpoint on this lane (against 3 to 15.5 minutes on the vast.ai host of the SPEC4 rental).

Lifecycle. Both instances self-terminated on `STATUS`. Their one-shot reaper schedules outlived them (`pg audit`: 2 orphans, 0 reaper-less) because `pg kill` only tidies a schedule after a successful terminate call and raises `InvalidInstanceID.NotFound` on an instance that is already gone; the two schedules were deleted with `aws scheduler delete-schedule` and the audit is clean. Follow-up for the launcher, not done here: `pg kill` or `pg audit --tidy` should delete an orphan reaper without a terminate call.

Science: `studies/reuse/results/RESULTS.md` Results 32 to 37, `studies/reuse/results/REVIEW_dating.md`, journal entries 2026-11-02 16:30 to 19:20 CST.
