# already-clean: index of study files and what is live (SPEC_D1 §10)

Written 2026-10-23 at closeout (session S6). "Frozen" files change only by dated amendments; "live" files are appended to.

| file or directory | state | what it is |
|---|---|---|
| `SPEC_D1.md` | frozen at tag `already-clean-freeze-2026-10-20` (dc18799a); dated amendments after it, listed in §0; §11 filled 2026-10-23 and refilled 2026-10-24 from the sixth run; §12 and §14 closeout records 2026-10-23 with 2026-10-24 addenda; §13 status and §15 closures 2026-10-24 | the spec: question, measurement, rules, runbook, grid |
| `JOURNAL.md` | live (append only; UTC headings from `date -u` since 2026-10-21T18:00Z) | the dated record of every decision, number and incident; the PREDICTIONS section is never edited |
| `DATA_CONTRACT.md` | frozen with the code it describes | file formats, paths and invariants of harvest, explain and verdict |
| `CHECKS.md` | record | literature queries and findings (ABSTRACT level); §13 status in `checks/literature_W5.md` when present |
| `DIRECTIONS.md`, `IDEAS.md`, `THEORIES.md` | parents, not edited by this study after 2026-10-19 | the ranking, the ideas and the theories the spec came from |
| `checks/` | checker reports (`CHECKERS.md` brief; `W1-W3.md`; `W4-W5.md`; `closeout_W5.md`; `closeout_W5_recheck.md`; `literature_W5.md`; `acceptance_final.md`; `closeout_slips_2026-10-25.md`; `w5_rerun_2026-10-25.md`) | read-only sanity checks after each wave, answered in the journal |
| `handoff/` | one file per session: `S2-S4.md`, `S5.md`, `S6.md` (S1's record is `verify/VERIFY.md` and SPEC §16); `judge_incident_log.txt` | record state, open jobs, exact next action, resume steps |
| `runbook/` | scripts as run | chunking and import for the manual backend, the per-chunk audit guards, the W5 command, the center-probe copy-back |
| `verify/` | W0 verification record | `VERIFY.md`, `HARVEST.md`, `EXPLAIN.md`, `REVIEW_EXPLAIN.md` |
| `walkthrough/index.html` | draft, uncommitted local edits (not part of the record) | a reader's walkthrough |
| results (explain worktree `bank/w5/`; `s3://<s3-bucket>/results/neurons/w5/`) | final (seventh unblind run, 2026-10-24T04:06Z, metrics sha256 a95bfa57…, equal to the sixth run in every number and label); runs 1–6 kept beside it as `bank/w5_run<n>_<utc>` and under `s3://…/results/neurons/` | `verdict/metrics.json`, `verdict/grid.md`, `verdict/samples/`, `figures/` (svg, pdf/, png/), `RESULTS.md`, `REGISTRY.md` |
| `results/` | copies, 2026-10-24 (seventh run; F4 carries the §6.4 null) | `RESULTS.md`, `REGISTRY.md`, `figures/` (svg, pdf/, png/), `README.md` naming the source and hash |
| explain directories, manual-backend outputs and transcripts (copies, 2026-10-24T04:02Z; re-check F39) | `s3://<s3-bucket>/results/neurons/w4/o7_s_v2/explain/` (43 objects: requests, responses, plans, judgments.jsonl, clusters.jsonl, calls.jsonl, manifest.json, manual_answering_models.jsonl), `…/w4/o7_gate_v2_sonnet_js/` (31), `…/w4/o7_t0_g2/` (8), `…/w2c/{q4b,g4,o7}/explain/` (29 each), `…/manual/<round>/<batch>/` (1,218: index.json and the chunk output files), `…/transcripts/manual/` (977 Claude Code transcripts of the answering agents) | what the sample pointer files and the answering-model attribution point to; the laptop copies are the explain worktree's `bank/` (not under git), the session scratchpad and `<local-agent-transcripts>/` |
| data banks (explain `bank/w1`, `bank/w2c`, `bank/w3`, `bank/w4`; harvest `bank/neurons/`) | sealed inputs, read by the instrument | harvests, explain directories, clusters, judgments, the per-checkpoint probe |
| code | `neurons/explain` d832b80f, `neurons/harvest` bca491a9 | `infra/pg/neurons_*.py`, `infra/aws_pg/jobs/neurons_*.py`, prompts with SHA-256 in each `manifest.json` |
