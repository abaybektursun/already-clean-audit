# NEURONRW — do clean neurons write what they read?

Status: **CLOSED** 2026-10-26 (`CLOSEOUT.md`). Freeze tag `neuronrw-freeze-1` = 967a8db3; amendments A20–A28 after
the freeze. Under the frozen rules there is no headline (H_write MIXED with 1 counted model, H_suff CANCELLED under
P8); the amended readings are exploratory and sit mostly in poles whose label names their own output token; the held-out
test on O7 did not confirm the literal-pole reading (`RESULTS.md`). Of 37 pre-registered predictions, under the frozen
rules 6 missed, 7 landed and 24 could not be scored (20 because a gate cancelled their readout). Final commit: the commit
that adds this sentence (`git log -1 -- studies/neuronrw/README.md`). Independent review: `checks/2026-10-26_result_review.md`
in three passes (first pass, addenda at dbebba9c and c87ea404; every item resolved, N1 and N2 included).

## Files

| file | content |
|---|---|
| `RESULTS.md` | result document: verdicts frozen first, Arm 0, literal split and P-lit, Arm 2, Arm 1, dose scale, samples, amendments, cost |
| `CLOSEOUT.md` | what is established and what is not, bugs not to repeat, next steps |
| `SPEC.md` | pre-registration; freeze record and post-freeze amendments in §13 and §14.2 |
| `SCHEMA.md` | file contract (v6) |
| `AUDIT.md` | prior measurements, the output-token leak, literature read |
| `JOURNAL.md` | predictions, launches, gate readings, decisions, in time order |
| `handoff/2026-10-26_run.md` | final run handoff: roster, gate and readout tables, spend, amendments, review actions |
| `checks/2026-10-26_result_review.md` | independent review: every headline number recomputed, findings |
| `plan/` | items, draws, prompts, judge prompts, budget, dose scales (`make_plan.py`, byte-reproducible) |
| `judge/` | judge requests, responses, keys, call log |
| `results/verdict_arm{0,1,2}_<M>.json` | per-model verdicts with stamps |
| `results/study_arm0.json`, `results/study_arm2.json`, `results/grid.md` | study labels, frozen rule first |
| `results/thresholds_arm{0,1,2}_<M>.json` | thresholds frozen on control data before each unblind (`*_preA25` = before A25) |
| `results/literal_split_<M>.json` | literal / non-literal split in both arms, repetition shares |
| `results/missed_predictions.md` | the 37 predictions of SPEC §5.6 and §11 against their outcomes, frozen and amended |
| `results/repair_<M>.json`, `results/repair_pooled.json` | self-repair (cancelled, underpowered) |
| `results/gates0_<M>.json`, `results/j0_gates_Q4B.json` | Arm 0 analysis gates and the J0 judge gate |
| `results/judge_*.jsonl`, `results/reexplain*_<M>.jsonl` | collected judge answers and re-explained labels |
| `results/*_preA24.json`, `results/*_preA25.json`, `results/*_preA28.*`, `results/*_preC10.*`, `results/study_arm0_preB1.json` | files as written before an amendment or a post-unblinding re-render, kept for comparison |
| `bank/rw/` | full banks (untracked; S3 `s3://<s3-bucket>/results/neurons/rw/`) |
