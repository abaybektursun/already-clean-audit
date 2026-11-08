# D1 verification day (W0), 2026-10-19

Everything in SPEC_D1 N1–N5 was implemented, tested offline with a failing input per gate, and run once
end to end at small scale with Claude Code subagents as the LLM backend. This file is the record. The
harvest side's own notes are `HARVEST.md`; the explain side's are `EXPLAIN.md` and `REVIEW_EXPLAIN.md`.

Branches (not pushed): `neurons/harvest` (worktree `../interp_research-neurons-harvest`) and
`neurons/explain` (worktree `../interp_research-neurons-explain`). Study documents on `drafts/already-clean`.
Banked data (gitignored): `../interp_research-neurons-harvest/bank/neurons/w0/{verify/3,grid/1,probe/1,dev/1}/out/`
and `../interp_research-neurons-explain/bank/w0/{o7,o7dev,o7q2}/`. Everything is also under
`s3://pg-lineage-769063704696/results/neurons/w0/`.

## 1. What ran on the GPU

| run | prefix | model, revision | tokens | wall | GPU-h | purpose |
|---|---|---|---|---|---|---|
| verify/3 | `w0/verify/3` | corpus; Q4, Q4B, G4 (2M/2M/1M); O7 step1000 and final (1M each); C0 (131k) | see HARVEST.md | 70 min | 1.2 (three attempts: 1.6) | N1, N2, N5; G0, G1; samples |
| grid/1 | `w0/grid/1` | O7 at the 16 grid revisions, `--poles` = 4 poles (S_grid) | 524,288 each | 122 min | 2.0 | W3 path: births, parent test, vestigial |
| probe/1 | `w0/probe/1` | O7 final, `--poles` = 40 probe poles, `--probe-texts` 355 texts | 131,072 | 9.5 min | 0.16 | Q1 scoring of the discovery set |
| dev/1 | `w0/dev/1` | O7 final, `--poles` = 4 development poles | 1,000,000 | 13 min | 0.22 | development-set contexts and vocab |
| devprobe/1 | `w0/devprobe/1` | O7 final, `--poles` = 8 probe poles, `--probe-texts` 74 texts | 131,072 | 9 min | 0.15 | Q1 scoring of the development set |

Capacity: six of the day's launches found no g6e.2xlarge or g6e.4xlarge in any availability zone on the
first try and launched on a later try; `--type g6e.2xlarge,g6e.4xlarge` is now the launch form.
Whole day, launch to STATUS, all prefixes including the two failed verify attempts: 245 minutes, about
4.1 GPU-h. `pg ps` is empty and `pg audit` is clean (the reaper schedules of every terminated instance were
removed). Grid steps ran at 3,231–3,365 tokens/s with G0 max ratios 0.0034–0.0060 at every checkpoint;
every grid directory has `n_tokens_scored` 524,288, the same `tokens_sha256`, `poles.json` = S_grid,
exemplar arrays [4, 50] with no NaN, and positions equal to T_15's.

The W0 Q1 loop ran on O7 `stage1-step1413814` rather than Q4B because the Q4B step was run without
`--poles`, so it has no `poles/vocab.json` (the explainer needs the vocabulary table). The Q4B instrument
predictions in JOURNAL.md (most poles RARE/UNK, θ_s > 0.3 at 2M tokens) are therefore not tested today.

## 2. Gates and their failing inputs

| gate | where | value today | pass | failing input that fails it |
|---|---|---|---|---|
| G0 activation identity | every harvest step | max ratio 0.0038–0.0065 (tolerance 0.01) | yes | silu↔gelu_tanh swap: 31–33 of 32–36 layers over tolerance on final checkpoints; only 3/32 at step 1000 (weak there; HARVEST.md follow-up 5) |
| G1 wikitext parity | q4, c0 | Qwen/Qwen3-4B 2.5491 vs banked 2.2444: fail; Qwen3-4B-Instruct-2507 2.244386524302183: pass | split | `g1_pass(2.2444, 2.3444)` false; the spec names the wrong model (HARVEST.md) |
| top-K exactness, band snapshots | tests | — | — | `test_topk_equals_brute_force_and_band_snapshots`; planted bugs (no merge, no band split) caught |
| histogram quantiles | tests | — | — | `test_histogram_quantiles_bracket_exact`; planted off-by-one caught |
| exemplar positions | tests, grid t15 | positions.npz = topk top-50 exactly | yes | planted shift by one caught |
| probe scoring | tests, probe/1 | 355×40, no NaN, argmax inside each text | yes | planted BOS-kept bug caught |
| collect: schema, model, ids | every manual batch | 0 rejections over 40 + 27 + 8 lines | — | `test_malformed_explainer_answer_fails_collect_naming_the_id`, `test_collect_rejects_model_mismatch_and_backend_mismatch`, `test_every_role_schema_is_enforced` |
| B2 leak rules (⟦⟧, digit, "context", a=) | collect | 0 rejections | — | `test_explanation_that_would_leak_to_the_writer_fails_collect`, `test_job_sentence_with_a_digit_fails_collect` |
| B5 identical second reader | collect | 0 | — | `test_second_reader_identical_to_the_explainer_fails` |
| changed prompt | collect | — | — | `test_changed_prompt_is_refused` |
| cluster partition, pairs, echo | collect-clusters | 0 rejections over 8 answers | — | `test_cluster_answers_must_partition_the_top50`, `test_echo_cluster_must_hold_exactly_the_double_negative_contexts` |
| θ_s gate (§6.1) | `verdict theta` | θ_s = 0.2989 (99th pct of 300 shuffled scores; median 0.050, 90th 0.120) | pass (< 0.5) | `test_theta_s_on_known_scores_and_its_gate` |
| θ_s before development explain | `verdict unblind` | refused the discovery directory (θ_s 00:01:26Z after explain start 23:15:05Z), exit 2 | works | the refusal itself, tonight |
| PREDICTIONS gate | `verdict unblind` | heading 2026-10-19T22:00:00Z, committed 22:00:00Z (a57b79f2), every artifact started later | pass | `test_predictions_gate_refuses_and_accepts` (no stamp, no zone, missing readout, uncommitted, committed after an artifact) |
| judge gate a_ctrl ≥ 0.80 and full control plan | `verdict judge-gate` | a_ctrl 1.00 on 4 poles (random-decoy controls 16/16), hard-decoy controls 7/8, f_ctrl 0 | pass | `test_judge_gate_requires_the_full_control_plan` |
| judge gate before first S judge batch | `verdict unblind` | gate 02:01:51Z, first S judge batch 02:02:12Z | pass | code path read; exercised via two explain directories |
| judgments match plans | collect-judge / unblind | 48 + 54 judgments, each id once | pass | `check_judgments_match_plans` (missing, unplanned, repeated) |
| p_u: a_ctrl at chance refused | unblind | — | — | `test_p_u_refuses_a_chance_level_control` |
| grid order and shared corpus | `check_grid` | 16 directories, branches stage1-step0 … stage1-step1413814, one corpus | pass | branch name, tokens_sha256, n_tokens_scored mismatches raise |
| p_u bootstrap on a degenerate draw | `verdict unblind` | refused on the 4-pole grid: a pole-level draw with no late-born cluster | works; see §4 item 6 | the refusal itself, tonight |

Offline suites: harvest 16 tests (`test_neurons_harvest.py`, `test_neurons_corpus.py`), explain 65 tests
(`test_neurons_llm.py` 11, `test_neurons_explain.py` 24, `test_neurons_verdict.py` 30). Both green at
the SHAs below.

## 3. The explain loop with the manual backend

Discovery set: the 20 poles of `verify/3/out/o7_final/samples/sample_draw.json` (10 neurons, both poles).
Requests prepared 23:15 UTC. Answering agents: Opus (`claude-opus-5-5`), one agent for the explainer
batch, a different one for the second reader, one fresh agent per writer and writer_job line (27), one
per clusterer sample (2). Every agent saw only its request line(s) and wrote only its response file;
writer and job lines went to `responses/parts/` and were concatenated by the orchestrator.

| step | lines | agent time | rejections |
|---|---|---|---|
| explainer | 20 | 4.6 min | 0 |
| second reader | 20 | 3.9 min | 0 |
| writer | 20 | ~35 s each | 0 |
| writer_job | 7 (3 UNREL poles) | ~20 s each | 0 |
| clusterer s0, s1 | 4 + 4 | ~2 min each | 0 |

Discovery-set readout at θ_s = 0.2989 (an instrument check; not the reported development number):
12 of 20 explanations pass §6.1; classes 8 ONE, 2 TWO-REL, 2 UNREL, 8 UNK, 0 RARE; the essay rule reads
"win" on 18 and "tie" on 2. Second reader agreed with the explainer's label on 17 of 20 poles.

Clusters (S_grid = L09N2969±, L19N828±): samples agree on 3/3/≥2/4 clusters per pole with matching names;
no double-negative contexts, so no echo cluster.

Development set D (the shuffle partners of S_grid: L09N181-, L09N828+, L19N5544+, L19N6678+), its own
harvest (`dev/1`, 1M tokens, `--poles D`), explain directory `bank/w0/o7dev` started 00:28 UTC, after
θ_s (00:01 UTC). Explainer and second reader (4 lines each, one agent each), 4 writer lines and 2 job lines
(one agent each), probe run `devprobe/1` (74 texts, 8 poles). `unblind --q1 O7=dev,devprobe,o7dev` with the
discovery θ_s ran to completion (exit 0): PREDICTIONS gate passed, θ_s gate passed, `grid.md` and
`metrics.json` written. Readout on D (n = 4, descriptive only): ONE 1, UNK 3; L09N181- (the `new ClassName`
pattern in C#-style code) passes §6.1; strong rates pos 11/20, near 3/20, distant 0/20, shuffled 0/60.
Labels agreed on 3 of 4 poles between the two readers.

Q2 on the 16-point grid (S_grid, 4 poles), two explain directories so the judge gate precedes the S judge
batch. Directory A (`bank/w0/o7`): clusters from two samples (8 answers), judge plan 48 lines (4 late-born
primary clusters × 6, 4 main-cluster controls × 6), one fresh agent per line, two at a time, 0 rejections.
Controls 23/24 correct (random decoys 16/16, hard decoys 7/8); parent tests 24/24 correct (the judge found
the true t_before record every time). `judge-gate`: a_ctrl 1.00, pass. Directory B (`bank/w0/o7q2`, seed
1): clusters 8 answers, judge plan 54 lines (5 late-born primary clusters, 4 controls), 0 rejections.
Controls 24/24; parent tests 26/30 (L09N2969+ 12/12, L19N828+ 6/6, L19N828- 8/12). Late-born clusters sit at
t_before T_10–T_13, which at 524k tokens per checkpoint says the main clusters of these four poles are
present only from step 100,000–560,000 on; n is 4, so none of this is a readout.

## 4. Findings that change the spec or the code

1. θ_s must come from a discovery set whose explain finished before the development set's explain
   started (`unblind` enforces it). The W0 plan had one set; a development set D (the shuffle partners of
   S_grid: L09N181-, L09N828+, L19N5544+, L19N6678+) was added with its own harvest and probe run.
2. The judge gate must be written before the first S judge batch, so the gate set and S are two explain
   directories. As coded, every grid directory's `poles/poles.json` had to equal each directory's
   `positions_poles.json`, which forces the two sets to be identical. Change (explain side, 7bb11a48,
   67 tests): the grid pole list may be a superset of an explain directory's positions list, every grid
   directory must carry the same list, and a pole's exemplar row is its index in the grid's own
   `poles/poles.json`. The real run harvests the grid once with `--poles` = S ∪ S_gate and the W3
   `--positions` file comes from `positions` run over S ∪ S_gate in that order (EXPLAIN.md §10.5).
3. T_15 exemplar record: the harvest now writes `poles/positions.npz` and `poles/exemplar_a.npz` from the
   pole's own top-50 when `--poles` is given without positions, and `--positions-from` prefers that file
   (DATA_CONTRACT §7; harvest c568669c).
4. `check_grid` needs all 16 checkpoints, so the W0 Q2 toy is a 16-point grid at 524,288 tokens.
5. HARVEST.md follow-ups 1–11 stand (G1 model, corpus directory rule, duplicates, window position 0,
   G0 power at early checkpoints, budget 55 GPU-h, hook profiling, zstandard pin, `--weights` validation,
   contract wording, reaper schedules; the schedules were cleaned tonight).
6. `p_u_bootstrap` refused the 4-pole grid: a pole-level bootstrap draw with no late-born cluster has an
   undefined A, and with 3 of 4 poles carrying one, (1/4)^4 of draws are degenerate, which over N_BOOT
   draws is certain. A real S with few late-born primaries would hit it too. Change (explain side, 4e38734e,
   69 tests): degenerate draws are redrawn and counted (`degenerate_draws` in the record and in the p_u row
   of grid.md); the function refuses only when they exceed 1% of N_BOOT. On the W0 grid 52 of 10,000 draws
   were redrawn (EXPLAIN.md §10.6).
7. The local pull of `corpus/` left out `tok/<tokname>/tokenizer/`, which `prepare-judge` needs (the earlier
   steps read contexts from `samples/contexts.jsonl` and never touched the tokenizer). Pulled from S3; the
   pull should sync the whole `corpus/` directory.

## 5. The final unblind and the closing state

`neurons_verdict unblind --q1 O7=dev,devprobe,o7dev --q2-grid <16 dirs> --q2-explain o7q2 --judge-gate
o7/verdict/judge_gate.json --theta o7/verdict/theta.json --seed 0` ran to completion at 02:21 UTC and wrote
`bank/w0/final/verdict/{grid.md, metrics.json, samples/}`. Every gate passed: PREDICTIONS (heading and commit
22:00Z before every artifact), θ_s 0.2989 < 0.5, θ_s before the development explain, judge gate a_ctrl 1.00
before the first S judge batch, 16 grid directories in order on one corpus, judgments matching plans. Every
readout row of the grid filled: p_u, b, v, H_MIG, r(c) at birth, hard-decoy accuracy, Q1 fractions; G2 and
the center curve say PENDING because their optional inputs were not given.

The numbers themselves are from 4 poles and 5 late-born clusters and are not readouts: p_u 0.400
[0.000, 1.000] (A 0.80, a_ctrl 1.00, flagged underpowered, label "mixed"); b 0.25 [0.00, 0.75] against a
null mean 0.58 (p = 1.0, "drift (inside band)"); v 0.143 [0.000, 0.600] (p = 0.97, "not H_VEST"); H_MIG 0 of
4; hard-decoy accuracy 1.00. What they verify is that the instrument computes, labels and bounds each readout
on real harvest data with the manual LLM backend.

Branch heads (not pushed): `neurons/harvest` b0ec4364 (5b499928 … b0ec4364, nine commits since f818d849),
`neurons/explain` 4e38734e (7c103fdf … 4e38734e), `drafts/already-clean` this commit. Agents used tonight:
Opus only; 1 harvest, 1 explain, 1 reviewer, 2 explainer-batch, 2 second-reader-batch, 33 writer and job
lines, 4 clusterer batches, 102 judge lines, never more than two at once. GPU: 4.1 GPU-h on L40S. `pg ps`
empty, `pg audit` clean, laptop 34 GiB free.

Not done today, by design or by block: the Q4B Q1 loop and its journal predictions (no Q4B `poles/`);
G2 (`--t0-explanations`) and the center birth curve (`--center-texts`); any live API call (the manual
backend answered every line; the first live call per backend is still open); the account creation task
(blocked on the Mac's permissions); figures F1–F6; the §15 decisions; pushing the branches.
