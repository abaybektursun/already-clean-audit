# ALWAYSON run, 2026-11-02 (AWS lane)

| step | time (UTC) | record |
|---|---|---|
| freeze | 21:08 | tag `alwayson-freeze-1` at `ca772e13e`; dry run green locally (tiny Qwen3.5 text model, both block types, torch path) |
| inputs | 21:08:35 | `s3://pg-lineage-769063704696/results/alwayson/inputs/{items.json, edit_edit.npz}` |
| launch | 21:09:47 | `pg run infra.aws_pg.jobs.alwayson_qwen --args ' --items … --corpus … --edit …' --no-weights --type g6e.2xlarge --budget-min 40 --s3-prefix results/alwayson/1 --json --detach` → `i-0ee19321d1f39ffa6`, reaper at 21:59:46 |
| done | 21:18:34 | `STATUS exit=0`, 527 s from launch by these timestamps (the watch log reports 518 s of elapsed watching); job wall 323 s (windows 30 s, forward 52 s at 19,617 tokens/s, sets and bootstrap 116 s); transformers 5.9.0 on the image already had `qwen3_5` (no install); Gated DeltaNet on the torch reference path; model class `Qwen3_5ForConditionalGeneration` |
| close | 21:25 | instance self-terminated; the orphan reaper schedule `pg-reap-i-0ee19321d1f39ffa6` removed with `pg kill`; 0.14 instance-hours at the on-demand $2.24 per hour, about $0.32 cash (the verdict's §7 row reads the job wall, 0.090 h); API $0 |

Lesson: the pipeline's reaper schedule outlives a clean termination every time (third occurrence); `pg kill` on the id clears it. The job printed its counts in its final line; a future job should print only the gates there, so the laptop's sample check comes before any number.
