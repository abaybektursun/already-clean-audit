# NEURONOFF — does an input-clean neuron pole matter when it is turned off?

Status: **CLOSED** 2026-10-25 (`CLOSEOUT.md`). Freeze tag `neuronoff-freeze-1` = e5a5c33c.

NEURONOFF zeroed single MLP neuron poles in `Qwen/Qwen3-4B-Base` (Q4B) and `google/gemma-3-4b-pt` (G4). It compared the held-out
own-context NLL increase of poles whose explanation passed the parent study's blind test (ONE) with matched poles
whose explanation failed or was declined (UNK). It also ran seven code-construct case studies (Arm B). Every
model gate passed. The per-pole positive control (SPEC §6 G-pc) failed on 65 of 156 Q4B poles and 75 of 202 G4
poles, above the frozen 30% line, so the Arm A labels of both models are cancelled and the study label is MIXED.
Under the cancelled label the paired median differences are −0.000101 nats (Q4B) and −0.0000348 nats (G4). Arm B
gave 0 of 7 SPECIAL and 0 of 7 behaviour changed. One pole's ablation moves own-context NLL by a median of
0.003–0.009 nats. O7 (`allenai/Olmo-3-1025-7B`) was not run; the frozen budget rule dropped it. Exploratory E1 and E2 were chosen
after unblinding and are never the headline. Compute: 2.794 instance-hours, USD 6.44.

## Files

| file | content |
|---|---|
| `CLOSEOUT.md` | status, artifact index, claims record, deviations, follow-up requirements |
| `results/RESULTS.md` | result document: headline, Arm B, absolute effects, E1/E2, missed predictions |
| `SPEC.md` | frozen pre-registration (amendments A1–A13 in §13) |
| `SCHEMA.md` | file contract (v4) |
| `LIT.md` | literature, with queries |
| `AUDIT.md` | prior measurements in the repo |
| `JOURNAL.md` | predictions, launches, gate readings, deviations, in time order |
| `handoff/2026-10-25_run.md` | run handoff: gate tables, thresholds, Arm B readouts, positive-control failures, cost |
| `plan/` | frozen pole selection, held-out contexts, budget (`make_plan.py`) |
| `prompts/` | Arm B prompts, one file per pole |
| `results/thresholds_<M>.json` | thresholds frozen on null pairs before unblind |
| `results/verdict_<M>_v2.json`, `results/armb_counts_<M>_v2.json`, `results/grid_v2.md` | analysis outputs with cancellation stamps |
| `results/verdict_<M>.json`, `results/armb_counts_<M>.json`, `results/grid.md` | analysis outputs as first written by the frozen code |
| `results/<cell>/1/` | committed copies of each cell's metrics, controls, manifest, Arm B files, exploratory outputs |
| `bank/off/<cell>/1/` | full banks (untracked; S3 `s3://<s3-bucket>/results/neurons/off/<cell>/1/`) |
