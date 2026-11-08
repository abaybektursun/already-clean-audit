# EXPLAIN: the D1 explain and verdict side (N3, N4, LLM backends)

Sections 1-9 describe commits `7c103fdf` and `9be37095`. Section 10 (fixes after review, commit
`c3b82d32`) supersedes them wherever the two differ.

Written 2026-10-19. Branch `neurons/explain` in `../interp_research-neurons-explain`, commits
`7c103fdf` (all files) and `9be37095` (gate value renamed `fail_floor`). Not pushed. Built against
`SPEC_D1.md` and `DATA_CONTRACT.md` as of 2026-10-19 and synthetic fixtures only; the harvest-side
code was not read.

Test command, from the worktree:

    uv run --no-project --with pytest --with transformers --with tokenizers --with anthropic \
        --with numpy pytest infra/pg/tests/test_neurons_explain.py \
        infra/pg/tests/test_neurons_verdict.py infra/pg/tests/test_neurons_llm.py

Result: 45 passed in 1.84 s. `--no-project` was added to the requested command. Without it, `uv run`
syncs the project's `pyproject.toml` dependencies, which would download torch 2.14.1 (32 packages)
and write a `uv.lock` into the worktree (checked with `uv sync --dry-run`). The tests need neither.

## 1. What was built

| file | content |
|---|---|
| `infra/pg/neurons_llm.py` | request/response format, JSON-schema subset validator, `collect`, backends `manual`, `anthropic`, `openai_compat`, CLI `run` |
| `infra/pg/neurons_prompts/{explainer,second_reader,writer,writer_job,clusterer,judge}.md` | prompts; `<!-- system -->` part then `<!-- user -->` template with `{placeholders}` |
| `infra/pg/neurons_prompts/MANUAL_BACKEND.md` | how a Claude Code subagent answers a requests file; role isolation |
| `infra/pg/neurons_explain.py` | N3: contexts, requests and collects for all six roles, shuffle, positions, judge plans |
| `infra/pg/neurons_verdict.py` | N4: banked-file loaders, every frozen rule as a pure function, CLI `theta`, `judge-gate`, `unblind` |
| `infra/pg/tests/neurons_fixture.py` | synthetic corpus/harvest/probe/grid builders (a fourth test file, shared by the three test modules) |
| `infra/pg/tests/test_neurons_{llm,explain,verdict}.py` | 11 + 14 + 20 tests |

The loaders (`parse_pole`, `layer_file`, `pole_stats`, `harvest_poles`, `probe_scores`) live in
`neurons_verdict.py` so that `neurons_explain.py` imports births, presence and main from it with
imports in one direction only.

## 2. CLI subcommands

`python -m infra.pg.neurons_llm`
- `run`: answers one batch through `anthropic` (`--effort` required) or `openai_compat`
  (`--base-url`, `--key-env`, optional `--provider-order`, `--quantizations`). It appends each response
  as it arrives, skips ids already answered (a run stopped by a transport error is resumed by running
  it again), and writes `responses/<batch>.backend.json`.

`python -m infra.pg.neurons_explain`
- `render`: prints one pole's top-n contexts with the marked token.
- `prepare`: explainer and second-reader requests, one batch file per role (`<batch>-explainer`, `<batch>-second_reader`).
- `collect`: validates and logs those batches and appends `explanations.jsonl`.
- `prepare-writer`: writer requests (explanation sentence only) and writer_job requests (one per unrelated job of each UNREL-labelled pole).
- `collect-writer`: writes `texts.jsonl`, `shuffle.json` (seed logged) and `probe_poles.json` (explained and shuffle poles, the probe job's `--poles`).
- `positions`: writes the T_15 top-50 token positions of S to `positions.npz` and `positions_poles.json`, the W3 `--positions` and `--poles` inputs.
- `prepare-clusters`: two clustering samples, one batch file each; checks the harvest against `positions.npz`.
- `collect-clusters`: writes `clusters.jsonl`; every answer must partition the top-50.
- `prepare-judge`: builds the 2AFC parent tests and positive controls (births from `neurons_verdict.grid_cluster_tables`); one batch per judge sample plus `plans/<batch>-judge.jsonl`.
- `collect-judge`: appends `judgments.jsonl` (with `correct` computed from the plan).

`python -m infra.pg.neurons_verdict`
- `theta`: computes θ_s on the discovery set and writes `verdict/theta.json`. Exit 3 when the gate fails.
- `judge-gate`: computes a_ctrl on the development discovery set and writes `verdict/judge_gate.json`.
- `unblind`: runs the PREDICTIONS gate, then writes the Q1 classes per model and the Q2 readouts to `verdict/metrics.json`, `verdict/grid.md` and `verdict/samples/{q1_<model>,q2_parent,q2_births}.jsonl`.

## 3. Spec rule to function

| rule | function |
|---|---|
| §2.2 score | `neurons_verdict.score` |
| §6.1 θ_s and its 0.5 ceiling | `theta_s`, `theta_gate` |
| §6.1 pass rule; the essay's rule beside it | `explanation_passes`; `essay_rule` |
| §2.3 RARE; UNREL support clause; classes | `is_rare`; `job_supported`; `assign_class` |
| §6.2 Wilson, kappa, terciles, θ_s ± 0.1 band, double-negative split | `wilson`, `cohen_kappa`, `q1_metrics`; F2 strong rates in `strong_rates` |
| §2.3 single-token suppressor flag | `single_token_suppressor` |
| §2.4 presence (both readings), birth, late-born, main, r(c), growth | `presence`, `birth`, `late_flags`, `cluster_table` |
| §2.4 formed before step 1000; B3 agreement; B4 sample-1 exclusion; B2 10M births | `cluster_table` statuses; `q2_readouts` |
| §2.4 parent test, decoys, exemplar removal, positive control | `neurons_explain.cmd_prepare_judge`, `record`, `layer_token_strings` |
| §6.3 p_u, bootstrap, rules, flags, validity gate | `p_u_value`, `p_u_bootstrap`, `p_u_rule`, `judged`, `q2_readouts` |
| §6.4 b and permutation null | `birth_order_test` |
| §6.5 v | `vestigial` |
| §6.6 H_MIG | `q2_readouts` (`H_MIG`) |
| §7 Holm | `holm` |
| §2.4 center birth curve (exploratory) | `center_birth_curve` |
| §9.3 PREDICTIONS gate | `predictions_gate` |
| §11 grid, §10 samples | `grid_markdown`, `q1_samples`, `q2_samples` |

## 4. Gates and the input that makes each fail

| gate | failing input | test |
|---|---|---|
| collect: missing, extra, repeated id; non-JSON; schema; model mismatch | the responses file of each kind | `test_collect_lists_every_bad_id_and_logs_nothing`, `test_collect_rejects_model_mismatch_and_backend_mismatch` |
| collect: backend record matches the backend named | sidecar absent or naming another backend | same |
| collect: no id logged twice | second collect of a batch | `test_collect_round_trip_logs_hashes` |
| collect: role rules checked before logging | `check` returning an error | `test_collect_check_callback_runs_before_logging` |
| schema of every role | one invalid answer per role, plus an extra key | `test_every_role_schema_is_enforced` (6 cases) |
| explainer answer out of range | confidence 1.5 | `test_malformed_explainer_answer_fails_collect_naming_the_id` |
| clustering answer must partition the top-50 | rank 24 assigned twice | `test_cluster_answers_must_partition_the_top50` |
| prompts frozen per explain directory | `judge.md` edited after the manifest; `second_reader.md` differing from `explainer.md` | `test_changed_prompt_is_refused` |
| samples/contexts.jsonl agrees with topk/ | one token changed | `test_pole_contexts_from_topk_and_from_samples` |
| anthropic backend cannot honour temperature | temperature 0.0; unknown effort | `test_anthropic_request_construction` |
| refusal or truncation from the API | `stop_reason` "refusal" | `test_call_anthropic_raises_on_refusal` |
| θ_s ceiling (0.5) and floor (θ_s <= 0, gate value `fail_floor`) | scores 0..0.99; all zeros; fewer than 100 scores | `test_theta_s_on_known_scores_and_its_gate` |
| unblind refuses a failed θ_s gate | theta.json with `fail_ceiling` | `test_unblind_q1_end_to_end` |
| PREDICTIONS gate (exit 2, names the file) | no heading; heading after a manifest; date-only heading on the artifact's day; missing manifest | `test_predictions_gate_refuses_and_accepts`, `test_unblind_q1_end_to_end` |
| grid filled once | second unblind into the same directory | `test_unblind_q1_end_to_end` |
| judge validity gate (a_ctrl >= 0.80) | gate "fail": p_u, flags, hard row, H_MIG reported as not valid; birth order still read | `test_unblind_q2_end_to_end` |
| p_u defined only when a_ctrl > 0.5 in every draw | a_ctrl = 0.5 | `test_p_u_refuses_a_chance_level_control` |
| B3 readings must agree | constructed scale-change table: max abs 6 up to T_4 and 20 after; the cluster is born T_0 under the raw ratio and T_5 under the percentile, status `readings_disagree`, excluded | `test_cluster_table_statuses_including_a_scale_drift_exclusion` |
| exemplar activation finite | NaN columns | same |
| judge records exclude exemplars, no numbers | mutation check run once: with the exclusion removed from `record()` the test fails | `test_judge_records_exclude_exemplars_and_show_no_numbers` |

## 5. Synthetic p_u recovery

Design: n = 400 clusters, one per pole, four random-decoy judgments each. A related cluster's
judgments are correct with probability a_ctrl = 0.95 and an unrelated cluster's with 0.5. Each pole
also has four control judgments at 0.95. `p_u_bootstrap` resamples poles, 10,000 draws, bootstrap
seed 0.

Random draw, data seed 0 (the first seed run; no seed was searched):

| truth | A | a_ctrl | point | 95% interval | rule |
|---|---|---|---|---|---|
| 0.05 | 0.9275 | 0.9481 | 0.046 | [0.0057, 0.0855] | H_RG |
| 0.30 | 0.8331 | 0.9481 | 0.257 | [0.2011, 0.3122] | H_SP |

Exact proportions (A and a_ctrl equal their §7 values): at 0.05 the rule is H_RG and at 0.30 it is
H_SP, and both intervals cover the truth (`test_p_u_recovery_at_the_expected_realization`).

Repeat rate over 200 random datasets per truth (2,000 bootstrap draws each, script in the session
scratchpad `neuexp/pu_sim.py`, not committed):

| truth | mean point (sd) | mean interval | coverage | H_RG | H_SP | mixed |
|---|---|---|---|---|---|---|
| 0.05 | 0.050 (0.021) | [0.014, 0.090] | 0.93 | 139 | 0 | 61 |
| 0.30 | 0.299 (0.029) | [0.240, 0.359] | 0.95 | 0 | 185 | 15 |
| 0.05, a_ctrl fixed at 0.95 | | [0.018, 0.084] | | 160 | 0 | 40 |
| 0.30, a_ctrl fixed at 0.95 | | [0.242, 0.356] | | 0 | 187 | 13 |

The mean intervals match §7 (upper ≈ 0.09 at 0.05, lower ≈ 0.24 at 0.30). One realization at a
true 0.05 is labelled mixed 31% of the time (20% when a_ctrl is not resampled), because the point
estimate itself has sd 0.021. §7's sentence "a true 0.05 has upper bound ≈ 0.09" is right for the
expected realization only. The spec should state this rate, or widen n, before freeze.

## 6. Spec ambiguities and how they were resolved

1. **Temperature.** Claude Opus 5.5 rejects sampling parameters, and Sonnet/Haiku 5.5 reject
   non-default values; this was read in the claude-api skill on 2026-10-19 and not measured against
   the API. So every role runs at temperature 1.0, the explainer included, and the second reader is
   an independent sample of the same call. The anthropic backend sends no temperature and refuses a
   request whose temperature is not 1.0. Effort is a required CLI argument with no default. Thinking
   is left at the model default (adaptive). Server-side refusal fallbacks are not enabled: a fallback
   model answering would break the per-call model record, and collect would reject its model string
   anyway.
2. **`related_to_others`** means "related to at least one other listed job". The explainer prompt
   tells the agent to merge related variants into one entry when it labels UNREL, so every listed
   job is then false. The jobs judged unrelated are the jobs with `related_to_others = false`. UNREL
   needs at least two of them and at least two of those supported. A pole with groups A~B and C
   unrelated to both, listed without merging, gets U = {C} and is classed UNK. The contract's
   per-job boolean cannot express groups. A `group` field would be the fix; the contract was kept.
3. **A label that disagrees with its job list** (ONE with two jobs, TWO-REL with an unrelated job or
   four jobs) is classed UNK by `assign_class` ("everything else"). It is not a collect failure.
4. **UNREL does not need §6.1's pass**, only the support clause, as §2.3 is written.
5. **Presence uses the pole-signed magnitude** max(s·a_t, 0), not |a_t|. An exemplar whose activation
   has the other sign at a checkpoint is activity of the other pole and counts 0 here.
6. **Presence floor:** median > 0 is required in both readings. Without it a dead pole at T_0 (q999 =
   max = 0) would be "present".
7. **Birth at the last grid point:** with no next point to stay present at, presence at T_15 alone is
   enough. A cluster never present has no birth (`never_born`).
8. **Late-born** compares with the earliest birth among the pole's born sample-0 clusters. A cluster
   unborn under the raw reading is not late under it, so a percentile-late cluster then
   disagrees and is excluded.
9. **Status order** (each cluster gets the first that applies): never_born, earliest,
   formed_before_step_1000 (late-born with birth T_1), readings_disagree, unmatched_in_sample_1,
   late_primary. Only late_primary enters the parent test. The §4.3 pivot share counts every cluster
   born at T_1.
10. **Two clustering samples:** sample 0 defines the clusters. A sample-0 cluster is "present in the
    other sample" when one sample-1 cluster holds more than half of its exemplars. §5's "both samples
    enter the four-judgment rule" is read as the two judge samples.
11. **main** is the sample-0 cluster with the largest exemplar activation at T_15 (signed magnitude);
    ties go to the lower cluster number.
12. **Exemplar columns.** `positions.npz` holds the T_15 top-50 in rank order, so a column of
    `exemplar_a.npz` is a rank and a cluster's `exemplar_ranks` index it directly.
    `prepare-clusters` and `prepare-judge` check the T_15 harvest against `positions.npz`, and the
    verdict checks every grid `poles.json` against `positions_poles.json`.
13. **Records.** A record is the pole's top-50 at t_before with E_c positions removed, truncated to L
    = min(20, what remains). The decoy record is truncated to the same L, so record length cannot
    identify the true pole. Decoy candidates are same-layer poles other than the true pole's own
    neuron (both its poles excluded) with at least L contexts left after removal. Random decoys are
    two distinct uniform draws from the candidates. The hard decoy is the candidate with the most
    shared top-50 token strings at T_15 (decoded strings), ties broken by pole id.
14. **Positive control:** two samples × two random draws plus two hard judgments, like a test
    cluster. §12 estimated ×2 per control.
15. **Hard-row measurability** (control accuracy against the hard decoy >= 0.80) is read on S's
    controls; the spec does not name the set.
16. **a_ctrl in p_u** comes from S's controls. The validity gate comes from the development discovery
    set (`judge-gate`). In the bootstrap, S's poles are resampled once per draw; clusters move with
    their pole and a_ctrl is recomputed over the same resampled poles.
17. **θ_s inputs** are each pole's positives, near misses and distant texts scored at its shuffle
    pole (§5 shuffle row), not the max sentence or the job texts. The shuffle pole is drawn from both
    signs of the same layer, excluding the pole's own neuron and dead poles. A floor was added: θ_s <= 0
    fails (`fail_floor`), because then a text with zero activation would count as strong.
18. **Holm family:** `b`'s permutation p-value, `v`'s bootstrap p-value against 0.5, and p_u's
    bootstrap p-value only for the rule that fired (P*(p_u >= 0.10) for H_RG, P*(p_u <= 0.20) for
    H_SP). For a mixed result no p_u p-value is quoted.
19. **Judge gate failed:** p_u, flags, hard row and H_MIG are written as not valid. Birth order,
    vestigial and the center curve are still computed. §6.3 names birth order and the center curve;
    §14 also names vestigial, which does not depend on the judge.
20. **Which manifests the PREDICTIONS gate reads:** every probe-scoring job, every explain directory,
    every grid harvest and the center-texts explain directory. The Q1 `stats` harvest is not gated,
    because Q4B's audit stats come from the W1-D discovery harvest, which precedes PREDICTIONS by
    design (§4.4, §9.2), and stats alone yield no class. Other parsing choices: a date-only heading
    counts as 23:59:59 UTC that day, the earliest dated heading is used, and a `started_utc` without
    a timezone is read as UTC.
21. **ids:** `<role>:<pole>:<sample>` for the explainer (0), second reader (1) and clusterer (0, 1);
    `writer:<pole>:0`; `writer_job:<pole>:j<k>`, where k indexes the explainer's job list. Judge ids
    are `judge:<pole>:c<i>:<k>` with k an opaque per-cluster integer. Control, decoy type and draw are
    in `plans/` only, because a manual answering agent sees the id. `job_pos` text ids use k = 5j + i.
    `judgments.jsonl` rows carry `id`, `draw` and `control` (false or true) in addition to the
    contract fields.
22. **Double-negative share** = `sign_counts[3] / sum(sign_counts)`, which equals the contract's `/ K`
    whenever the top-K is full.
23. **Batch files** hold one role, and for the clusterer and judge one sample (`<batch>-<role>`,
    `<batch>-<role>-s<k>`).
24. **Center birth curve:** the max sentence's score uses each checkpoint's own max|a|; first index
    with score >= 0.3 of the T_15 score. "Monotone" is reported as the share of nondecreasing curves.
    It runs only with `--center-texts`; otherwise it is listed PENDING.
25. **B2** runs only with `--b2` and reads `snapshots/band1/stats/` from every grid directory;
    otherwise it is listed PENDING.
26. `prepare` explains every listed pole, RARE ones included; dead poles (max|a| = 0) are not scored.

## 7. Open questions for the harvest side (must match before W1)

Paths are in one dict, `neurons_verdict.HARVEST_FILES`, so a mismatch is a one-line change.
- `poles.json` is read at `out/poles.json`, the literal reading of "copied into `out/`". The rest of
  that bullet lives under `out/poles/`, so `out/poles/poles.json` is the other reading.
- `out/poles/vocab.json`, `out/poles/exemplar_a.npz` and `out/poles/probe/scores.npz` are read under
  `poles/`.
- `samples/contexts.jsonl`: entries are taken in file order and checked token by token against
  `topk/`, so a 1-based `rank` field does not matter but a different order fails loudly.
- `exemplar_a.npz` rows follow `poles.json`, and its columns follow `positions.npz` columns. The W3
  harvest must be run with `--poles explain/positions_poles.json --positions explain/positions.npz`.
- Manifest fields read: `tokname`, `n_layers`, `n_tokens_scored`, `started_utc`, and `branch`
  (`stage1-step<n>`, checked against the §4.3 grid order).
- `scores.npz` `text_ids` is read with `allow_pickle=False`, so it must be a numpy unicode array, not
  an object array.
- The probe job's `--poles` must be `explain/probe_poles.json` (explained and shuffle poles).

## 8. Findings while building

- **Judge leak under the manual backend.** Within one judge sample file, a cluster's true record
  appears in three requests (two random decoys, one hard decoy); each decoy record appears once. An
  agent answering the whole file could pick the repeated record without judging relatedness. The
  runbook requires one fresh agent per judge line, run one at a time. API calls are independent and
  do not have this problem.
- §5's power statement holds for the mean interval only (section 5 above).
- The judge template has no digits, so the test can assert that a judge prompt has no digit at all
  on a digit-free fixture corpus. Real corpus text will contain digits; only activation values are
  kept out.

## 9. Follow-ups (not built)

1. G2 random-init control (clustering at T_0, the share of poles with a confident cluster).
2. Population-weighted Q2 estimates beside the stratified ones (§4.4, §7), and the Q2 split by
   SwiGLU regime.
3. Figures F1–F6 and the figure registry (§10).
4. Read Transluce 2024 and NeuronScope 2601.03671 at source before freeze and fold reusable steps
   into the explainer prompt (§3 N3); then tune the prompts on the discovery set (W2-D).
5. First live API call: one request per backend, opened, before any batch. Neither API backend has
   met a real endpoint. Structured-output acceptance of these schemas, after the minItems, maxItems,
   minimum and maximum keywords are dropped, is unverified.
6. The manual judge's one-agent-per-line rule makes judging cost scale with lines. The API backend is
   the practical route for W4 (≈ 4,400 judge calls).
7. Decide the `poles.json` location with the harvest author (section 7).
8. The essay restricts its legibility count to explanations that describe the positive side
   (essay.md 637); nothing here applies that restriction. Decide before freeze whether Q1 reports it.

## 10. Fixes after review (2026-10-19, commit `c3b82d32`)

Input: `verify/REVIEW_EXPLAIN.md` and the coordinator's calls (SPEC_D1 amended for N1, N2, N4, N8).
Same command as before (`uv run --no-project ...`): **62 passed in 9.15 s** (llm 11, explain 22,
verdict 29). Lint (pyflakes) clean.

### 10.1 What changed, by review item

| item | change | where | test |
|---|---|---|---|
| N1, N2 | kept (spec amended to match) | `signed_magnitude`, `birth` | `test_birth_requires_staying_present` |
| N3 | `b` gets a 95% interval from resampling poles (10,000 draws, seed logged) | `birth_order_test`, `pooled_share` | `test_birth_order_permutation_test_and_interval` |
| N4 | p_u carries no p-value; Holm runs over `b` and `v`, always both | `p_u_bootstrap`, `cmd_unblind` | `test_p_u_regression_pin`, `test_unblind_q2_end_to_end` |
| N5 | a missing or non-finite probe value raises, naming the text id and pole; `score` also checks finiteness | `scorer`, `score` | `test_a_nan_or_missing_probe_value_raises_naming_text_and_pole` |
| N6 | growth undefined when max\|a\| is 0 at T_12 or T_15: the cluster leaves v and the count is reported (`n_excluded_growth_undefined`); a birth point with max\|a\| = 0 raises (see 10.2) | `cluster_table`, `vestigial_counts` | `test_undefined_growth_is_excluded_and_counted` |
| N7 | `judge-gate` requires, for every pole with a non-echo cluster, 4 random + 2 hard planned controls and a judgment for each, and names what is missing; only then a_ctrl | `cmd_judge_gate`, `check_judgments_match_plans` | `test_judge_gate_requires_the_full_control_plan` |
| N8 | clusterer schema and prompt carry `echo`. Collect checks that at most one cluster is an echo and that it holds exactly the double-negative ranks (g < 0, u < 0, from the T_15 topk). Echo clusters get status `echo` and are excluded from m (B4), main, late-born, matching, the parent test and every count; poles whose clusters are all echoes have no main and are listed as `poles_all_echo` | `echo_errors`, `cluster_table`, `q2_readouts` | `test_echo_cluster_must_hold_exactly_the_double_negative_contexts`, `test_echo_clusters_take_no_part_in_main_births_or_counts` |
| N9 | population-weighted p_u, b, v (`--q2-strata KEY`): weight N_k/n_k from that model's Q1 classes over its non-RARE poles; unrepresented strata listed. `f_ctrl` on S and in `judge_gate.json`. G2 from `--t0-explanations` against the 10% line | `stratum_weights`, `q2_readouts`, `g2_readout` | `test_stratum_weights`, `test_p_u_moves_clusters_with_their_pole_and_takes_weights`, `test_g2_against_its_line`, Q2 end-to-end |
| B1 | PREDICTIONS gate: (a) the heading timestamp needs `Z` or ±hh:mm, converted to UTC; (b) the section needs a line starting with each of `p_u`, `b`, `v`, `H_MIG`, `hard-decoy`; (c) the section must equal HEAD's, the heading must be in a commit (`git log --format=%cI -S <heading line>`), and the latest commit time over that and `git blame` of the section lines must precede every `started_utc`, as must the heading time. Only the first `## PREDICTIONS` section counts; there is no fall-through | `predictions_gate`, `committed_time` | `test_predictions_gate_refuses_and_accepts` (no heading, no zone, date only, empty, never committed, edited after commit, committed late, offset, stub first, wrong path, accept) |
| B2 | the explanation and every job sentence may not contain ⟦ ⟧, a digit, "context" (any case) or `a=`/`g=`/`u=`; collect fails naming the pole; the explainer prompt says so | `leak_errors`, `explainer.md` | `test_explanation_that_would_leak_to_the_writer_fails_collect` (4 cases), `test_job_sentence_with_a_digit_fails_collect` |
| B3 | clusterer and judge ids are `<role>:<batch>:<int>`, the integers a seeded random permutation; the mapping is in `plans/<batch>-clusterer.jsonl` and `plans/<batch>-judge.jsonl` | `opaque_ids` | both cluster tests and the judge test |
| B4 | judge batch files are written in random order; every plan row carries `cluster_requests_in_file` (3 here); the runbook keeps one fresh agent per judge line | `cmd_prepare_judge`, `MANUAL_BACKEND.md` | judge test (order differs from plan order) |
| B5 | collect refuses a second-reader answer equal (as a JSON object) to the explainer's for the same pole, and refuses collecting the second reader without the explainer | `same_as_explainer` via `collect_many(cross_check=...)` | `test_second_reader_identical_to_the_explainer_fails` |
| B6 | runbook: one fresh agent per writer line as well | `MANUAL_BACKEND.md` | — |
| B7 | `theta.json` and `judge_gate.json` carry `created_utc` and `inputs` {path, sha256}. `unblind` refuses a θ_s written after any Q1 explain directory started, and a judge gate written after the first S judge batch. Timestamps now carry microseconds | `cmd_theta`, `cmd_judge_gate`, `cmd_unblind`, `neurons_llm.utc_now` | Q1 and Q2 end-to-end (late θ_s, late gate refused) |
| C3 | double-negative share = sign_counts[3] / 50, as the contract states | `pole_stats` | — |
| C4 | `--journal` must end in `studies/already-clean/JOURNAL.md` | `predictions_gate` | gate test |
| C5 | a multi-batch collect validates every batch before logging any; a failed collect leaves `calls.jsonl` unchanged and can be rerun | `neurons_llm.collect_many` | `test_a_failed_multi_batch_collect_logs_nothing_and_can_be_rerun` |
| C6 | `samples/q1_<model>_poles.jsonl` holds every pole with each text's score and strong flag and the shuffled scores; `samples/q2_clusters.jsonl` holds every cluster's birth table with its judgments | `cmd_unblind` | Q1 end-to-end checks all 8 classes |
| C7 | strong rates exclude RARE poles; the rates with RARE are reported beside | `q1_metrics` | Q1 end-to-end |
| C8 | the θ_s − 0.1 band is not computed when θ_s − 0.1 <= 0, and says so | `q1_records`, `q1_metrics` | — |
| C9 | the center curve also scores the five positives (mean per grid point) | `center_birth_curve` | — |
| open 5 (part) | grid harvests must share `tokens_sha256` and `n_tokens_scored`; the probe and stats harvests must be one `revision` | `check_grid`, `scorer` | fixtures carry both fields |
| weak tests | p_u: 200 seeds plus a regression pin, plus a pole-versus-cluster resampling test (200 poles × 3 identical clusters widens the interval by 1.4-2.1× against 600 × 1); permutation null pinned within pole (null mean 0.75 on the mixed case); a score equal to θ_s is strong; vestigial composition; judge exclusion checked for control records too, A/B balance and the logged seed checked | | |

### 10.2 Where this departs from the instructions, and why

1. **p_u recovery test uses 200 seeds, not 50.** Thresholds are as instructed (coverage >= 0.90; H_RG in
   55-85% at 0.05; H_SP >= 85% at 0.30). Measured with the code as committed:

   | truth | seeds | coverage | H_RG | H_SP | mixed |
   |---|---|---|---|---|---|
   | 0.05 | 0-49 | 0.880 | 31 | 0 | 19 |
   | 0.05 | 0-199 | 0.930 | 141 | 0 | 59 |
   | 0.30 | 0-49 | 0.960 | 0 | 45 | 5 |
   | 0.30 | 0-199 | 0.955 | 0 | 186 | 14 |

   At 50 seeds the coverage floor fails: the binomial sd is about 0.036 around a rate near 0.93, so
   the test would fail about one run in five on noise alone. At 200 seeds every assertion holds
   with margin, and the counts equal the reviewer's (141/200, 186/200). The estimator undercovers
   slightly at 0.05 (0.93, not 0.95): the percentile interval is clipped at 0 near the boundary.
   That belongs in §7 beside the 70% power figure.
2. **B1(c) also reads `git blame` of the section lines**, not only `git log -S` of the heading.
   Otherwise a prediction line edited and committed after the artifacts started would pass,
   because the heading's own commit stays early. The commit time used is the latest over both.
   A committer date can be set by hand (`GIT_COMMITTER_DATE`), so this check catches mistakes,
   not deliberate backdating. Requiring the commit to be on the remote before the first artifact
   would close that; not built.
3. **"The earliest valid heading"** is implemented as the first `## PREDICTIONS` heading in
   document order (the journal is append-only). If it fails any check, the gate refuses; a later
   heading is never used.
4. **B2 also checks every job sentence**, because `writer_job` passes each one to the writer. I
   did not add a one-sentence check (not in the coordinator's list; sentence splitting on
   abbreviations is unreliable). Cost of the "context" ban: a legitimate explanation such as "in
   the context of court filings" is rejected and must be reworded.
5. **N6: a birth point with max|a| = 0 raises** instead of excluding the cluster. The median
   exemplar activation is > 0 there, so the pole's max|a| over the same corpus cannot be 0; the stats
   and exemplar files disagree, which is a data error, not a degenerate pole.
6. **N8: the echo flag is checked against g and u.** "Contexts in the double-negative regime" is
   fully determined by the signs, so collect accepts only the exact set. The clusterer therefore
   does a mechanical step; the code could assign it. I kept the agent step as instructed, and the
   check makes it impossible to get wrong silently. A negative pole cannot have double-negative
   entries (g < 0, u < 0 gives a > 0), so negative poles never have an echo cluster.
7. **B7: ordering checks added in `unblind`** (the reviewer's suggestion; the coordinator asked only
   for `created_utc` and hashes). θ_s must be written before every Q1 explain directory starts. The
   judge gate must be written before the first S judge batch is prepared, which is stricter than
   "before any S judgment is read" because the read time is not recorded.
8. **G2's T_0 explanations are not gated by PREDICTIONS.** G2 is a pre-freeze control (§5: failure
   tightens the presence rule before freeze), so its artifact predates PREDICTIONS by design.

### 10.3 Review items not changed

- **C1, C2 (contract text).** `DATA_CONTRACT.md` is not mine to edit. It needs these entries:
  - ids:
    - `writer_job:<pole>:j<k>`;
    - clusterer and judge ids are `<role>:<batch>:<int>`.
  - rows:
    - `judgments.jsonl` rows add `id`, `draw`, and `control` on every row;
    - `clusters.jsonl` clusters add `echo`.
  - new files:
    - `explain/probe_poles.json`, `positions.npz`, `positions_poles.json`;
    - `plans/<batch>-clusterer.jsonl`, `plans/<batch>-judge.jsonl`;
    - `responses/<batch>.backend.json`;
    - `verdict/theta.json`, `verdict/judge_gate.json`;
    - `samples/q1_<model>_poles.jsonl`, `samples/q2_clusters.jsonl`.
  - harvest manifest fields read: `tokname`, `n_layers`, `n_tokens_scored`, `tokens_sha256`, `revision`, `branch`, `started_utc`.
- **Open items 1-4, 6, 7** need a spec or harvest-side decision:
  1. the vocabulary sign for negative poles;
  2. whether the API's `msg.model` equals the requested id;
  3. near-position leakage, since exclusion is exact-position only;
  4. control and test accuracies matched only in kind;
  6. the B4 majority match rule, to be written into §8;
  7. ONE with a label/job-count disagreement classed UNK.

  Item 5 is done in part (10.1).
- **Holm and the CLI exit codes** are as before: θ_s gate exit 3, PREDICTIONS and ordering refusals
  exit 2.

### 10.4 Two decisions implemented (commit `ff8f478e`)

Same command: **65 passed in 9.02 s** (llm 11, explain 24, verdict 30). Lint clean.

1. **Judge records drop exemplar windows** (SPEC §2.4, DATA_CONTRACT §6).
   - `exemplar_windows` gives, for each exemplar e of the cluster, the inclusive token range e − 64 .. e + 16,
     clipped to e's document; `in_windows` tests membership.
   - Both records and the decoy-eligibility count (`avail`) exclude every context whose activating token
     falls in any window.
   - Test `test_judge_records_drop_every_context_in_an_exemplar_window`: a context 10 tokens after an
     exemplar and one 10 tokens before (same document) are removed; one 100 tokens after, and the first
     token of the next document, stay; the windows clip at the document edges.
   - The end-to-end judge test now checks that no record line is any top-50 context, of any pole at
     t_before, that lies in a window.
   - Mutation run once: with windows shrunk to exact positions, the dedicated test fails. The end-to-end
     test still passes in that case, because it computes its expectation with the same function. The
     dedicated test is the one that pins the rule.
   - The fixture corpus grew from 40 to 400 documents. With 25 exemplars × 81-token windows, a
     3,000-token corpus left fewer than 20 contexts per record.
2. **Negative poles read the neuron's vocabulary flipped** (DATA_CONTRACT §6).
   - `neurons_verdict.pole_vocab(vocab, pole)` returns the positive pole's entry as stored. For the
     negative pole it returns the neuron's suppressed items as `promoted` and its promoted items as
     `suppressed`, scores negated, so each list reads in the pole's own direction.
   - Both the explainer prompt input and the last-layer `single_token_suppressor` flag use it. A neuron
     that strongly promotes one token makes its negative pole a suppressor of that token. This closes
     review open item 1 on both reads.
   - Tests:
     - `test_negative_pole_shows_the_neurons_vocabulary_swapped`: the rendered explainer request for
       `L03N7-` lists the neuron's suppressed tokens under "most promoted" with positive scores and its
       promoted tokens under "most suppressed"; `L03N7+` is unchanged.
     - `test_pole_vocab_flips_the_negative_pole_and_the_suppressor_flag_follows`.
   - Mutation run once (no swap): both tests fail.
3. The remaining open items stand as recorded in `DATA_CONTRACT.md` §6: items 2, 4 and 7 are decided
   or open until the first live call.

### 10.5 One grid harvest for S and the judge-gate set (commit `7bb11a48`, on top of `e092cd81`)

`grid_cluster_tables` now requires `positions_poles.json` ⊆ every grid directory's `poles/poles.json` (the
refusal names the directory and the missing poles). Every grid directory must also hold the same list (the
refusal names the directory). A pole's exemplar row is its index in the grid's own list. The W3 harvest's
`--positions` must be built by `positions` over S ∪ S_gate in the same order as its `--poles`. Rows are
per pole from the T_15 top-50, so each explain directory's subset `positions.npz` matches row for row.
Tests: `test_grid_tables_unchanged_when_the_grid_holds_more_poles_in_another_order` and
`test_grid_missing_a_pole_of_the_explain_directory_is_refused`. Indexing by the explain directory's order
fails the first. 67 passed.

### 10.6 p_u bootstrap conditional on a late-born cluster and a control (commit `4e38734e`)

`p_u_bootstrap` redraws any pole-level draw that holds no late-born cluster or no control. The redraw count
is returned as `degenerate_draws` and printed in the p_u row of `grid.md`. More than 1% of n_boot redrawn
refuses, naming the count. The fixed-seed regression pin is unchanged (0 redraws; every pole has a cluster).
The W0 case (one pole of four without a late-born cluster) redrew 30 of 10,000 draws (expected 0.39%) and
gave p_u = 0.429 [0.25, 1.0]. The literal case "exactly one pole of several has late-born clusters" always
refuses under the 1% limit: a draw misses that pole with probability (1 − 1/P)^P, which is 25% at P = 2
and approaches 37% as P grows. The test therefore covers that case as a refusal, beside "no pole has one".
69 passed.

### 10.7 After W0: prompt v2, 256-token contexts, UNK-N/UNK-F, X3, X7, DeepInfra (commit `cee153ed`)

72 passed (llm 13, explain 25, verdict 34). Lint clean.

- **Prompt v2** (`neurons_prompts/explainer.md`, equal to `second_reader.md`). Every v1 item is kept. Added,
  quoting the essay:
  - the shared pattern or distinction, not a list of nouns (essay.md 97–99, "if I couldn't use groups of
    nouns at all");
  - the summary motif on punctuation and the word before the cue (41–43, 187);
  - long-context dependence as a job (191);
  - UNREL only after the shared-pattern search fails and each job can stand alone;
  - UNK as the honest answer (209);
  - encodings are not jobs, and the meaning of the X7 lines.

  `explain/manifest.json` records `prompt_version: "v2"`, the prompt SHA-256s and the context widths. A
  directory made under another version is refused.
- **Widths.**
  - The explainer and clusterer see 256 tokens before and 16 after.
  - The judge's exemplars and records show 64 before.
  - The judge's exclusion window is 256 before and 16 after.
  - `render --before N`.
  - Every context renders from `tokens.npy`; `samples/contexts.jsonl` (64 tokens before) is no longer read.
- **UNK-N / UNK-F.**
  - UNK-N: the explainer's own class is UNK. UNK-F: every other UNK.
  - Each pole also gets `unk_reason`: failed §6.1, UNREL support clause failed, or label and job list disagree.
  - The split appears in `q1_metrics` (every fraction table, beside an unchanged UNK total), in per-pole
    samples and in `grid.md`.
- **X3.** `q1_metrics.X3_blind_max_ge_corpus_max`: per class, and per UNK part, the share of scored poles whose
  `max` text scores ≥ 1.0, with a Wilson interval.
- **X7.** `encoding_notes` checks the marked token and 16 tokens either side:
  - Base64 runs (mixed case, digits, + / or padding; at least 8 characters) and %XX runs that decode to
    readable UTF-8 get a `decoded:` line.
  - Letters outside the Latin script get a `transliterated:` line: romanized for Cyrillic, Greek and Bengali
    (Bengali "প্র" → "pra"); other scripts are named "not transliterated".

  No exception handling is used. Tests: planted Base64, Cyrillic, Bengali and CJK tokens in a toy corpus;
  plain and camelCase words produce no line.
- **Backend.**
  - `neurons_llm run --settings <file>`. `infra/pg/settings/deepinfra.json` holds the base URL
    `https://api.deepinfra.com/v1/openai`, the key env var `NEURONS_API_KEY_1`, every role on
    `deepseek-ai/DeepSeek-V4.1-Flash-1017`, temperature 1.0 and `response_format: json_schema`.
  - Each request's model and temperature must match the settings for its role.
  - The sidecar records the settings and the model ids the responses reported.
    `responses/<batch>.usage.jsonl` logs latency and token counts.
  - The key is read only from the named env var and placed only in the Authorization header. A test
    with a sentinel key finds it in no file and in no output.
- **First live call (SPEC §15 item 1, DATA_CONTRACT §6).** Role writer; explanation "Fires on the comma that
  closes a clause describing relief after a long wait."
  - With `max_tokens` 400 as asked: HTTP 200 and finish_reason `length`. There were 356 prompt tokens, 400
    completion tokens and 0 reasoning tokens, so the writer's 16 texts did not fit. The code raised; nothing
    was recorded.
  - Repeated with `max_tokens` 2000: the response `model` equals the requested id exactly. Latency was
    12.55 s, with 356 prompt and 366 completion tokens. The JSON is valid under the writer schema, and
    `collect` accepted and logged it.
  - DeepInfra accepted `response_format: json_schema` and `temperature`, so the payload is unchanged.
- **Spec text not implemented as written:**
  1. §2.3's banked "share of the pole's top-50 at window position 0" is harvest-side and its field name
     is not in the contract, so the verdict does not read it yet.
  2. §2.2 still names Anthropic and `claude-opus-5-5` as the agents' model; the run now uses DeepInfra
     DeepSeek. Agent 3's "different seed" is an independent second call; no `seed` field is sent.
  3. X7's readout (explainer pass rate on flagged poles) needs the per-pole flag banked in
     `explanations.jsonl`; not built.
  4. DATA_CONTRACT §7 asks that `positions` reproduce T_15's `poles/positions.npz` and that
     `check_positions` fail otherwise. `check_positions` still compares against T_15's `topk/`, which
     should hold the same positions; comparing against `poles/positions.npz` as well is not built.
  5. Prompt v2 is the starting text; §2.2 tunes it on the discovery set at W2-D before freezing.

### 10.8 Concurrent runs, pos0_share, X7 flag, positions check (commit `d281852c`)

78 passed (llm 16, explain 26, verdict 36). Lint clean.

- `neurons_llm run --concurrency N` (default 8) uses a thread pool.
  - HTTP 429 and 5xx are retried with backoff 2, 4, 8, 16 and 32 s. The sixth failure raises, naming the id.
  - `responses/<batch>.usage.jsonl` records `attempts` per id.
  - Only the main thread writes files.
  - On any failure, unstarted requests are cancelled, finished answers are kept, and the run raises. A
    rerun resumes.
  - `run-many --batches a,b,c` runs a wave in one command.
  - Tests use a stub endpoint:
    - Answers arrive out of order and one request gets 429 once. The collected answers, reported model
      and attempt counts equal a sequential run's.
    - A request failing six times gives up naming its id; the four backoffs it slept were checked.
    - A 400 is not retried.
    - `run-many` answers two batches.
- `pos0_share` (DATA_CONTRACT §8) is banked per pole in the Q1 records. `q1_metrics.pos0_share_by_class`
  gives the mean and the 0/25/50/75/100 quantiles. A stats file without the field raises, naming the file.
- X7 flag. `prepare` writes `plans/<batch>-explainer-view.jsonl`: for each pole, whether any context got a
  `decoded:` or `transliterated:` line. `collect` copies it into `explanations.jsonl` as `encoding_flag`.
  `q1_metrics.X7_pass_rate_by_encoding_flag` gives the §6.1 pass rate, with a Wilson interval, for flagged
  and unflagged poles.
- `check_positions`. When T_15 has `poles/positions.npz`, the explain directory's positions must equal it,
  and it must equal T_15's `topk/`. Without that file, `topk/` decides. The grid fixture now writes the file
  for T_15, as the harvest does, and the test exercises both refusals.

### 10.9 Redoing refused answers (commit `1e0c7059`)

81 passed (llm 16, explain 29, verdict 36).

- `neurons_explain reject --out --batch --roles --backend [--max-attempts 3]`.
  - It runs exactly collect's validation through one shared table, `validation()`, which every collect
    command also uses:
    - schema, for every role;
    - leak check and copy cross-check for explainer and second reader;
    - partition and echo checks for the clusterer;
    - schema only for writer, writer_job and judge.
  - Every refused answer line moves to `responses/<batch>.rejected.jsonl` with its reason and utc.
  - What stays untouched: missing answers (not rejections), usage lines, and clean batches
    (byte-identical). Kept lines keep their bytes.
  - An id already rejected max-attempts times makes reject refuse and name it before anything is written.
- The loop collect → reject → run → collect is in `MANUAL_BACKEND.md` and the `neurons_llm` docstring.
- Tried on a copy of `bank/w1/q4b_disc` (`bank/tmp/q4b_disc`): 8 explainer and 6 second-reader lines moved,
  usage unchanged, the other lines byte-identical. A second reject found only missing answers.
  `bank/w1` was not touched.
- **The 14 problems are mostly rules colliding with legitimate text, not model noise.**
  - Six are the word "context" in ordinary English ("in the context of web styling", "measurement context",
    "code contexts", "the context states", "surrounding context", "math contexts"). None refers to the
    contexts the explainer was shown.
  - Two (L35N330+, both readers) are "Base64". The prompt's own X7 paragraph uses that word twice, so the
    prompt teaches a token its rule forbids.
  - Four are names that contain digits: "B2B" (L25N9276+, both readers), "CSS3" (L17N3053-), "FB2"
    (L02N275+).
  - Two are numeric formats that are the job itself: "6-3-1005" and "4.5" for a neuron firing on separators
    inside numeric references (L26N4948+); "50/50" and "70:30" for mixture ratios (L08N5870+). "Spell out a
    number" does not cover formats.
  - Three poles failed in both readers (L24N1090-, L25N9276+, L35N330+), which points to a systematic cause.
    They may reach the 3-attempt cap.
  - Not changed, per instructions. Candidate fixes, if wanted:
    - restrict the digit rule to standalone numbers outside words;
    - restrict the "context" rule to references to the shown contexts ("context #", "the contexts above");
    - remove the word "Base64" from the prompt or exempt it.

### 10.10 Leak check narrowed to the leaks it exists for (commit `52d90981`)

98 passed (llm 16, explain 46, verdict 36).

`LEAK_RES` now refuses only three things:
- the ⟦ ⟧ marker;
- an activation value: `\b[agu]\s*=\s*-?\d`;
- a reference to a shown context: `\bcontexts?\s*#?\s*\d+`, `#\d+`, and "contexts
  above/shown/provided/given/listed/supplied/I saw/I was shown", case-insensitive.

Digits and the word "context" used to describe the job pass. The prompt files are unchanged.

Tests:
- The 15 sentences refused on W2-D d1 (14 answers; one answer had two) are copied into the suite and pass.
- The six pointer examples are refused.

Applied read-only to all 240 original d1 answers in `bank/w1/q4b_disc`, the new check finds 0 problems.

### 10.11 `discovery` report per prompt version (commit `b5554cf1`)

99 passed (llm 16, explain 46, verdict 37).

`neurons_verdict discovery --theta --stats --probe --explain --out` writes two files.
- `verdict/discovery.json`:
  - "set": "discovery";
  - θ_s and the theta file's `created_utc`;
  - the §6.1 pass rate over scored non-RARE poles, with a Wilson interval;
  - the class fractions, including UNK-N and UNK-F;
  - the X7 pass rates, strong rates and X3, exactly as `q1_metrics` computes them;
  - the prompt version and the SHA-256s of explainer, second_reader and writer, from `explain/manifest.json`;
  - per batch: request, response and rejected counts, requested models, backend, and reported models.
- `verdict/discovery_samples.jsonl` holds per pole: class, pass, strong counts per kind, and explanation.

The command applies no timing gates and refuses to overwrite. The explain manifest has no confirmation or S
mark, so no refusal on one exists.

Run on `bank/w1/q4b_disc` (writing only `verdict/discovery*.*`; hashes of every other file under `bank/w1`
unchanged):
- θ_s 0.306.
- Pass rate 27/119 = 0.227 [0.161, 0.310].
- Classes: ONE 23, TWO-REL 3, UNREL 0, UNK 93 (all UNK-F), RARE 1.
- Strong rates: positives 0.304, near misses 0.092, distant 0.012, shuffled 0.010.
- X3: ONE 4/23.
- X7: flagged 9/57 pass against unflagged 18/63. The X7 rates count all 120 scored poles, the RARE one
  included; the headline pass rate excludes it.

X7 defects visible in the real requests (not fixed here):
- All 15 `decoded:` lines are false: random identifiers decode to valid but meaningless UTF-8 (for example
  "E짢Wl"). The readability test accepts any printable Unicode.
- "MODIFIER" and "MICRO" are listed as scripts. They come from Unicode names such as MODIFIER LETTER and
  MICRO SIGN.

Most flags are genuine non-Latin text: 319 transliterated lines, CJK 236.

### 10.12 X7 fixed; Opus and Sonnet explainer settings (commit `84a493a0`)

102 passed (llm 18, explain 47, verdict 37).

- **`decoded:` lines.** A line needs at least four characters, at least 90% of them ASCII letters, digits,
  spaces or `.,;:!?'"()-/`.
- **Script names.** They come from Unicode block ranges for CJK, Cyrillic, Greek, Arabic, Hebrew,
  Devanagari, Bengali, Thai, Hangul and kana only.
- **Effect on the full rendered contexts of the real d1 requests.**
  - 0 decoded lines (the old pre-pass emitted 15, all false).
  - Scripts: CJK 264, Hangul 45, Cyrillic 29, Greek 24, Arabic 18, kana 15, Hebrew 5, Devanagari 3, Thai 1.
  - No "MODIFIER" or "MICRO".
- **d1 answers.** All 240 were produced with the old pre-pass. Their X7 flagged/unflagged comparison is not
  to be read.
- **Settings.** `deepinfra_opus_explainer.json` and `deepinfra_sonnet_explainer.json` put explainer and
  second_reader on `anthropic/claude-opus-5-5` or `anthropic/claude-sonnet-5-5`, and every other role on
  DeepSeek-V4.1-Flash.
- **Model per role.** `prepare --model` names the explainer and second-reader model;
  `prepare-writer --model` names the writer's. A run refuses requests prepared with another model before
  any call.
- **Untested against the live endpoint.** Anthropic's own API rejects `temperature` for Opus 5.5, and
  whether DeepInfra forwards `temperature` and `response_format: json_schema` for these ids is unknown.
  Make the first call one request.

### 10.13 Redoing only the writer: `fork-writer` (commit `bede6f84`)

104 passed (llm 19, explain 48, verdict 37).

A plain copy with the writer files deleted is not enough. The copy's `calls.jsonl` keeps the old writer and
writer_job rows (142 in d2), and `collect-writer` refuses those ids as already logged. The manifest's batch
list still names the DeepSeek writer batches. `verdict/theta.json` and the discovery files belong to the old
texts.

`neurons_explain fork-writer --source <dir> --out <new dir>` copies `explain/` without the writer step:
- it drops the writer batches' files, their call rows, their manifest entries, `texts.jsonl`,
  `shuffle.json`, `probe_poles.json` and the shuffle seed;
- it records `forked_from`;
- it never writes the source and refuses an existing target.

Then `prepare-writer` with the new model, `run-many`, `collect-writer --seed 0` (the same seed redraws the
same shuffle poles; the test checks it), a new probe job, and `theta` and `discovery` into the new directory.

Settings file `deepinfra_opus_explainer_writer.json`: explainer, second_reader, writer and writer_job on
Opus 5.5; clusterer and judge on DeepSeek-V4.1-Flash.

### 10.14 Prompt v3 (commit `9e53d341`)

104 passed. `prepare` records `prompt_version: "v3"` and these SHA-256s (checked against the files):
- explainer and second_reader `e36cf31d162ba668cae09a57d2f60df593342eb09b83609503ce611ee206bd4e`
- writer `ed68dcecb5cd325468c85b513baeb489af0fa18781394157e0d35b8cc604b52b`
- writer_job `3ab19637704e40a3ffd885c84827f3b6bc8c4b76a6586c780f08fe34e223ee43`

v3 contents:
- **The test as a number.** A positive counts at 0.3 of the max |a|; the explanation passes with at
  least three of five positives and at most two near misses.
- **The token form first.** The marked tokens' literal strings, word pieces and shapes are written down
  before any meaning is read.
- **Where the pole lives.** The kind of text the strongest contexts come from is named when the pole
  depends on it.
- **Untestable jobs.** A job a short text cannot carry is said in plain words.
- **Before choosing ONE.** The explainer writes two near misses of its own; confidence is defined as the
  pass probability.
- **Format sentence.** It matches the leak check at `52d90981`.
- **Writer and writer_job.** Positives put the named thing in its exact token form and structure.
  writer_job gets that positive rule only, since it writes no near misses.
- **Writer near misses.** They must not contain the named word, piece, phrase or symbol.

Schemas and code paths are unchanged, apart from `PROMPT_VERSION`.

### 10.15 Provider declines, two-token texts, probe rule (commits `116cd159`, `6531fad1`, `53cc6515`)

109 passed after the last commit.

**Provider declines (`116cd159`, 107 tests).**
- A `content_filter` or `refusal` finish reason becomes an empty answer with the finish reason in the usage
  line. Collect refuses it, reject moves it, and run asks again.
- The rejection that brings an id to `--max-attempts` writes `responses/<batch>.declined.jsonl` (id, role,
  last reason, finish reason, attempts). `run` never asks that id again; validation treats it as neither
  missing nor answered.
- A declined explainer or second reader becomes an explanations row with class UNK and `declined`.
- A declined writer or writer_job leaves the pole without texts. The shuffle is drawn over every
  writer-step pole, and `probe_poles.json` holds the text poles and their partners.
- The verdict counts a declined pole as UNK-N, with reason "<role> declined (<cause>, <n> attempts)",
  never dropping it. A declined second reader leaves the kappa pairs.
- Discovery reports declines per batch and per pole.

**Two-token texts (`6531fad1`, 108 tests).** Every writer and writer_job text must have at least two tokens
under the scored model's tokenizer, because the probe excludes each text's first token. The check is in
the shared validation table, so collect and reject agree.
- `collect-writer` now requires `--corpus`.
- `reject` takes `--harvest` and `--corpus` for writer roles.

**Probe rule (`53cc6515`, 109 tests).** `scorer` refuses a probe directory without `probe_rule` in
`scores.npz` and `probe_excluded_positions == "text position 0"` in its manifest. Both strings are recorded
in `theta.json`, `discovery.json` and the unblind Q1 inputs. A read-only check refuses all three existing
probe outputs, `w2d/q4b_probe/1..3`.

### 10.16 `draw-s`: S and S_gate for the W3 grid (commit `ac4e00dc`)

110 passed.

`neurons_verdict draw-s --stats --audit --extra --seed --out` writes four files under `<out>/draw/`:
- `S.json`: the non-RARE audit poles;
- `S_gate.json`: the non-RARE extra poles;
- `grid_poles.json`: their union, the W3 `--poles`;
- `draw.json`: the rule text, the seed, input paths with SHA-256, counts, and each excluded RARE pole with
  density and max|a|.

RARE is read through `pole_stats` and `is_rare`, as unblind reads it. The draw is blind by construction:
- With 600 audit poles and |S| = 600, the §4.4 class order selects every non-RARE pole, so no explanation
  or class is read.
- More than 600 non-RARE audit poles would need that order, so the command refuses instead of inventing one.
- The seed is recorded only; no random step is taken.

Poles are sorted by (layer, neuron, sign). Population weights stay with unblind's `stratum_weights`.

Real run on O7 T_15 (`neurons/w1/o7_t15/1/out/o7_t15`), audit 600 and extra 40, into `bank/w3/o7/draw/`:
0 RARE excluded, |S| = 600, |S_gate| = 40, `grid_poles` = 640.

### 10.17 No Opus through any API (commit `92823783`)

114 passed. The rule is the user's, given 2026-10-20 ("SPEC_D1 §2.2 model policy"). It is enforced in two
places:
- **Run time.** `load_settings` and `run_batch` refuse any Opus id before a request is sent, on both API
  backends: `anthropic/claude-opus…`, `anthropic.claude-opus…`, or bare `claude-opus…`.
- **Writing a request.** `make_request` and every prepare command refuse a provider-routed
  `anthropic/claude-opus…` id, so no request file or explain directory is written for it.

A bare `claude-opus…` id stays allowed in request files, because manual-backend requests name the model
the Claude Code subagent runs as; `run` refuses it on the API backends. A settings file marked `retired`
is refused.

Settings files:
- `deepinfra_opus_explainer_writer.json` is kept with `"retired": "2026-10-20, no Opus through the API"`.
- `deepinfra_opus_explainer.json` is not marked retired, but `load_settings` refuses it for naming Opus.
- `deepinfra_flash_second_reader.json` puts all six roles on DeepSeek-V4.1-Flash.

### 10.18 Resuming a batch on other models (commit `cb4a7cd0`)

115 passed.

- **`explain remodel --out --batch --roles --model [--unanswered-only]`.** It sets the model of the roles'
  request lines. With the flag, answered lines keep their model; without it, a role with answered lines is
  refused. Opus ids are refused, counts are printed, and the operation is recorded in the manifest
  (`remodels`).
- **`run` / `run-many`.** The Opus refusal and the per-role model check apply only to the requests still to
  be sent; answered lines are skipped. The sidecar records later settings of a resumed batch
  (`later_settings`); a different backend is refused.
- **Per-pole models.**
  - `explanations.jsonl` and `texts.jsonl` rows carry the model their response line reports.
  - The manifest's `models_by_role` counts answering and requested models per role.
  - The verdict takes each pole's models from the response lines.
  - `q1_metrics` and discovery add `by_explainer_model` and `kappa_models` (with `cross_model_pairs`).
- **The run-time model check is per role.** A resume settings file names the explainer, writer and
  writer_job on X and the second reader, clusterer and judge on Flash.
- **Blocker covered by the tools.** A rejected Opus answer becomes a pending line that still names Opus,
  and `run` refuses it. Run `remodel --unanswered-only` again after any reject on the explainer.

### 10.19 Second reader collected on its own (commit `b310674a`)

W2-C collected each cell's explainer alone, so `collect --roles second_reader` was refused ("checked together with
the explainer") and `--roles explainer,second_reader` was refused (explainer ids already in `calls.jsonl`). Now a
second-reader-only collect or reject is compared with the explainer as collected: `collected_explainer` takes the
lines of `responses/<batch>-explainer.jsonl` whose text has the `response_sha256` that `calls.jsonl` logged (a
mismatch raises). It refuses only while some explainer request is neither logged nor declined. Same schema, leak
and identical-answer checks as the joint collect. Second-reader rows and calls are appended, explainer rows are left
as they are, and `models_by_role` is updated. Test: explainer alone, then second reader alone, gives
`explanations.jsonl` and `calls.jsonl` byte-identical to one joint collect. 116 tests.

### 10.20 W4 against the W3 layout (commit `4e765d98`)

Explain directories. No subset command: `positions` makes each one (manifest, own `positions_poles.json`), and from
this commit it records the `--poles` file, its SHA-256 and the harvest. No W4 step reads explanations or texts.
Q2 reads positions, clusters, plans, judgments and the manifest. The center texts and the strata come from
`bank/w2c/o7` through `--center-texts` and `--q1`/`--q2-strata`. Copying `c1` would put one fact in two places.

Changes:
- `harvest_corpus` refuses a corpus whose `manifest.json` SHA-256 is not the harvest's `corpus_manifest_sha256`.
  The W1 corpus manifest is `0c99a8e8…` and all three W1 harvests record it; the check passes on them.
- `judge_gate.json` records `judge` (answering models, judge prompt SHA-256). `unblind` refuses S judgments from
  another judge.
- `layer_file` keeps at most 64 files. Measured: an O7 layer is 34.6 MiB of stats and 21.0 MiB of topk in memory.
  The unbounded cache would have held about 17.3 GiB of grid stats plus up to 10.5 GiB of topk in `prepare-judge`
  on S; the laptop has 24 GiB.
- `prepare-judge` keeps one layer's token strings at a time.

Test: W4 end to end on a grid split over `1/out/t00..t06,t15` and `1b/out/t07..t14`, with S_gate and S as disjoint
subsets of one grid pole list. It covers positions, clusters, judge, `judge-gate`, the S judge batch and `unblind`.
Listing the two prefixes in turn puts T_15 eighth, and `check_grid` refuses it; the grid list needs T_15 last.
117 tests.

Measured request sizes, built on the W1 O7 T_15 harvest as a stand-in, 24 random S poles. Flash prompt tokens are
taken at 3.90 characters per token (d1 explainer requests); the outputs are estimates from the W0 manual answers.

| role | prompt tokens, median (p90) | output tokens |
|---|---|---|
| clusterer | ≈ 15.2k (16.3k) | ≈ 190 |
| judge | ≈ 4.9k (5.3k) | ≈ 90 |

Counts: clusterer 2 per pole (80 and 1,200). Judge 6 × (poles with a main + late-born primary clusters); at the
spec's 1.25 late clusters per pole that is ≈ 540 and ≈ 8,100, with a floor of 240 and 3,600.

Open:
- A declined clusterer sample blocks `prepare-judge` (`clusters_by_pole` needs both samples).
- A declined judge line blocks `judge-gate` and `unblind` (missing judgment).
- The center birth curve needs probe scores at every grid point; W3 scores T_15 only.
- Nothing stops `prepare-judge` on S before the gate; doing it disqualifies that S directory.
- Editing any prompt makes `prepare`/`prepare-writer` refuse on every older directory.
- The PREDICTIONS section's commit time is 2026-10-20T03:13:32Z (heading 02:58:11Z); editing it later would move
  the gate past the W3 manifests.

### 10.21 Declined rule for Q2, `prepare-judge --gate`, center curve not computed (commit `b03cd162`)

Coordinator decisions of 2026-10-21, as implemented:
- **Declined clustering sample.** A clustering sample declined at the attempts cap is a `clusters.jsonl` row with
  `declined`, and that pole leaves Q2. It is reported as `declined_clusterer` (`n`, `poles`: pole → reason) in
  `judge_gate.json` and in the unblind `q2`, beside `n_positions` and `n_poles`.
- **Declined judge line.** A judge line declined at the cap is a `judgments.jsonl` row with `declined`, `answer`
  null and `correct` null. Its cluster leaves p_u (and flags, hard, H_MIG); if it is a control line, its pole leaves
  a_ctrl (and f_ctrl, hard_ctrl). It is reported as `declined_judgments` (`n`, `ids` → reason, `clusters`,
  `control_poles`). `judge_gate.json` counts control lines only, since the gate reads only controls.
- `prepare-judge`, `judge-gate` and `unblind` proceed on the rest. `grid.md` states the counts.
- **`--gate`.** `prepare-judge --gate <dir>` is required. It equals `--out` for the gate directory; any other
  directory refuses before writing unless `<dir>/verdict/judge_gate.json` exists. The record (`gate_dir`, `self`, the
  gate file's SHA-256 and `created_utc`) goes in the manifest under `judge_gates[batch]`. `unblind` keeps its own
  timing check.
- **Center curve.** The center birth curve is `"not computed: W3 scores T_15 only"` in `q2` and `grid.md`;
  `--center-texts` and `center_birth_curve` are removed.

Runbook change: both judge batches take `--gate bank/w4/o7_gate`. 118 tests.

### 10.22 `explain reopen` (commit `a4cd0230`)

The S_gate clustering on Flash declined 24 of 80 samples at the old cap of 3, almost all for "exemplar_ranks do
not cover 0..49 exactly once". The coordinator raised the clusterer's cap to 8.

`explain reopen --out --batch --roles` puts declined ids back to pending:
- It removes the batch's declined ids of those roles from `responses/<batch>.declined.jsonl`.
- It records each one in the manifest under `reopened`: old reason, finish reason, attempts, declined utc, and the
  reopen utc.
- The rejected lines stay, so attempts keep counting and `reject --max-attempts 8` sets the new cap.

A clusterer batch that was already collected is uncollected first: `clusters.jsonl` is removed and the batch's
rows leave `calls.jsonl`; both are recorded under `uncollected`. Both files are derived from the responses and
plans, which stay. Reopen is refused for other collected roles and once a judge batch has been prepared.

The prompt states the partition only by implication. `clusterer.md` says "Every context belongs to exactly one
cluster", and each context carries `#k`. It never says that `exemplar_ranks` must list every rank
0..n_contexts−1 exactly once, nor that a cluster needs at least one rank. No prompt edit was made.

### 10.23 W5 in one step (commit `28430cf5`)

`neurons_figures.py` draws F1–F6 as hand-written SVG (matplotlib is not in `pyproject.toml`) from
`verdict/metrics.json` alone:
- Every bar is drawn to one scale per figure; a test checks that `height / value` is constant.
- Titles are generated from the numbers.
- Footers give the source file, its SHA-256 and the keys read.
- A missing or PENDING input refuses before any file is written.

`unblind` `q2` gains three blocks:
- `births`, for F4.
- `grid_population`, for F3: every pole of every layer at each grid point, reading `max_abs`, `sign_counts` and
  `density` only.
- `gallery`, for F6: one pole of S each for ONE, TWO-REL and UNREL∪UNK, by the `--q2-strata` classes, drawn with
  the seed among poles with at least two born clusters; the exemplars come from the clusterer's sample-0 requests.

`q1` gains `untestable_phrase`: a string match on the wording `explainer.md` asks for, since no field records it.

`verdict results-template --unblind --out` writes RESULTS.md from `metrics.json`:
- §6.2 tables with Wilson intervals, by depth tercile, by SwiGLU regime and by explainer model;
- kappa with its models;
- the §11 grid;
- the judge gate and the declined counts;
- X3, X7 and the untestable count;
- labelled blanks for the prose, the GPU cost and the API cost (`usage.jsonl` holds tokens, not dollars).

124 tests.

Facts read from manifests for the W5 command:
- **PREDICTIONS:** heading 2026-10-20T02:58:11Z, commit 03:13:32Z. It precedes the Q1 explain directories
  (16:50:36Z), the grid (17:05:51Z onward), the probes (2026-10-21T01:24Z onward) and S (2026-10-21T16:56:32Z).
- **θ_s:** created 16:44:57Z, gate pass, before every Q1 directory started.
- **Probe and stats revisions:** equal for Q4B, G4 and O7.
- **Grid:** `check_grid` passes on the 16 real manifests, and `snapshots/band1/stats` exists at all 16, so `--b2`
  can run.
- **O7 Q1 directory:** it holds 640 poles (600 audit and 40 extra, the S_gate source). The O7 Q1 readout is
  therefore over 640, not §6.2's 600, unless a pole filter is added (open).

### 10.24 `--q1 … ,poles=PATH` (commit `9a6562ac`)

Coordinator decision: the O7 Q1 readout covers the 600 audit poles only. With `poles=PATH`:
- The cell's Q1 metrics, samples and strata are computed over the listed poles only.
- `metrics.q1.<key>.poles_list` records the path, SHA-256 and count, and the explain directory's other poles as
  `excluded_by_list` (`n`, ids).
- A listed pole without a Q1 record is refused.
- RESULTS.md states the list and the left-out ids.

Facts checked against the files:
- `bank/w1/o7_split/audit_poles.json` holds 600 poles, S ⊂ that list (0 outside), and S_gate ∩ that list = ∅.
  So `--q2-strata O7` takes its strata from the listed set and covers every S pole.
- No step reads the S_gate poles' Q1 classes: `judge-gate` reads clusters and judgments only, `draw-s` read
  stats only, and unblind uses classes only for S, through the strata and the F6 gallery.

Test: an 8-pole cell with a 6-pole list gives the 6-pole class counts and lists `L01N1-, L01N4-` as excluded.
The W5 fixture now runs O7 with a list. 125 tests.

### 10.25 Clusterer prompt v2 (commit `9623a267`)

The cause: on S_gate both models broke the partition, which the prompt only implied.
- Sonnet 5.5 produced 312 answers for 80 requests; 242 were rejected, 234 of them for coverage.
- Flash at cap 8 reached 37 of 40 poles, with B4 agreement on only 15 of 37.

`clusterer.md` v2 (SHA-256 `557a806d…05d67` → `5d8b55ca…0c1ff`) adds explicit output rules:
- The contexts are numbered 0 to N−1, and N is given above them.
- All `exemplar_ranks` together hold every number 0..N−1 exactly once.
- Every cluster has at least one number.
- The echo cluster follows the same rule.

The example is now a complete partition of N = 6, and the user part gives N and the last rank. What a cluster is
and how to name it are unchanged. `judge.md` and the Q1 prompts are unchanged.

`CLUSTERER_PROMPTS` in `neurons_explain.py` maps each version to its SHA-256. A new manifest records
`clusterer_prompt_version: v2` and refuses a `clusterer.md` that matches no recorded version.

Older directories (`bank/w2c/*`, `bank/w4/o7_gate`, `bank/w4/o7_s`) keep their recorded SHA-256:
- These commands now refuse there, because `ensure_manifest` compares every role's prompt: `prepare`,
  `prepare-writer`, `positions`, `prepare-clusters`, `prepare-judge`.
- These still work: `collect`, `collect-writer`, `collect-clusters`, `collect-judge`, `reject`, `reopen`,
  `remodel`, `fork-writer` (its copy keeps the old SHA-256, so `prepare-writer` refuses in the fork), `run` and
  `run-many` (they send the request lines as stored, with the old prompt), and every verdict command.

W4 therefore restarts in new directories for both S_gate and S.

G2: `--t0-explanations <dir>` reads `<dir>/explain/explanations.jsonl`, rows with `sample == 0` (explainer)
only. No second reader is needed. A declined T_0 explanation counts as confidence 0.0.

### 10.26 `explain fork-clusters` (commit `6d26acff`)

Context: the Flash judge failed the gate on the Sonnet-clustered gate set (a_ctrl 0.673, 39 poles; gate 0.80).
That is the first W4 number read, on the tuning set only. A second judge batch in the same directory is refused
(the control plan doubles), so a fork is needed to gate a stronger judge on the same clusters.

What `fork-clusters --source <dir> --out <new>` copies, byte for byte:
- `positions.npz`, `positions_poles.json`, `clusters.jsonl`;
- `plans/<b>-clusterer.jsonl`;
- every `requests/` and `responses/` file of the clusterer batches;
- the clusterer rows of `calls.jsonl` (all other rows are dropped).

What it leaves behind: judge requests, responses and plans, `judgments.jsonl`, `verdict/`, the judge seeds and
`judge_gates`. The manifest records `forked_from` (source, its manifest SHA-256, utc, dropped batches), and
`models_by_role` is recomputed.

It is refused when the source has no `clusters.jsonl`, and into an existing directory.

Test: fork, then `prepare-judge --gate <fork>` with another model, then `collect-judge` and `judge-gate`. The gate
records the new judge model, and the fork's clusters are byte-equal to the source's. 128 tests.

### 10.27 Presence, birth, r(c) and growth as the verdict computes them (for RESULTS and the device check)

Read from `neurons_verdict.py` at `6d26acff`; `cluster_table`, `signed_magnitude`, `presence`, `birth` and
`vestigial_counts` are the functions. Definitions:
- a⁺_t(e) = max(s·a_t(e), 0), where s = +1 on the positive pole and −1 on the negative pole, and a_t(e) is
  `poles/exemplar_a.npz` `a` at the pole's row (its index in `poles/poles.json`) and exemplar column e (rank in
  the T_15 top-50), at grid point t. An exemplar that fires the other pole counts 0.
- med_t(c) = median over the cluster's exemplars E_c (its sample-0 `exemplar_ranks`) of a⁺_t, in float64. The
  median is per cluster; there is no pole-level presence. A non-finite entry is refused.

Presence at t (both readings require med_t > 0, so a dead pole is never present):
- percentile reading: med_t ≥ q999_t[n, s];
- raw-ratio reading: med_t ≥ 0.3 · max_abs_t[n, s].
Both arrays come from `stats/layer_{l:02d}.npz` of that grid point. There is no smoothing.

Birth (per reading): the first t with presence at t and at t+1. Presence at T_15 alone gives birth T_15. A
cluster never present has no birth.

r(c) = med_b(c) / max_abs_b[n, s], with b the percentile-reading birth. It is undefined for a cluster never born,
and refused when max_abs_b = 0.

Growth over the last half = med_15(c) / max_abs_15[n, s] − med_12(c) / max_abs_12[n, s], with T_12 = step 320k.
It is undefined when max_abs is 0 at T_12 or T_15; such clusters are counted as `n_excluded_growth_undefined`.
"Shrinking" means growth < 0. A non-main cluster is vestigial when it is born at or before main (percentile
reading) and shrinking. v pools vestigial over defined non-main born clusters.
