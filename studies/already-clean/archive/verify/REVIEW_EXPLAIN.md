# Review of the D1 explain and verdict code (branch `neurons/explain`, 7c103fdf + 9be37095)

Reviewer: independent read, 2026-10-19. Code read in place in `../interp_research-neurons-explain`; nothing there was edited (git status clean at 9be37095 after all runs). Scratch scripts and their outputs are in the session scratchpad `review_explain/` (`dump_requests.py`, `dump_judge.py`, `stats_check.py`, `failure_probe.py`, `tz_probe.py`).

## 1. Verdict

**9 findings change a number, 16 do not (7 of those weaken blindness, 9 are cosmetic), 7 are open.** The frozen arithmetic is right. p_u, the pole bootstrap, the within-pole permutation null, Wilson, Holm's formula, the pass rule and the class rules all match SPEC §6. The weak points are guards around that arithmetic. The PREDICTIONS gate can be passed by a backdated or local-time heading. Explanations reach the blind writer unfiltered. Degenerate inputs (NaN, zero max) yield numbers instead of errors. §7's bootstrap interval for `b` is missing.

Tests: `pytest test_neurons_explain.py test_neurons_verdict.py test_neurons_llm.py -q` → **45 passed in 3.03 s**.

## 2. Findings

### 2a. Rule by rule (SPEC §2.3, §2.4, §6, §7)

| rule | where | matches? |
|---|---|---|
| θ_s = 99th pct of shuffled-pole scores | `neurons_verdict.py:141-148` (`theta_s`), `:554-558` (`shuffled_scores`), `neurons_explain.py:294-305` (`draw_shuffle`, same layer, own neuron and dead poles excluded) | matches; adds a floor gate θ ≤ 0 (`:151-158`) |
| pass: ≥3/5 pos, ≤2/5 near, ≤1/5 distant strong (score ≥ θ_s) | `neurons_verdict.py:170-175` | matches |
| essay rule (0.30, win/tie/lose), never deciding | `:178-184`; used only in `q1_metrics` `:677` | matches |
| RARE: density < 50 or max = 0 | `:187-189`, `:635`, `:649-651` | matches |
| ONE / TWO-REL / UNK on failure | `:205-209`, `:214` | matches; ONE also requires exactly one listed job (open item 7) |
| UNREL support clause (≥2 unrelated jobs each with ≥3/5 strong job positives) | `:210-213`, `:192-195`; job requests `neurons_explain.py:283-285` | matches SPEC §2.3 wording ("at least two of them") |
| score = max_a/max_abs (+), min_a/(−max_abs) (−) | `neurons_verdict.py:134-138` | formula matches; no finiteness check (N5) |
| presence, percentile and raw readings | `:256-261` | uses signed magnitude, not \|a\| (N1) |
| birth = first t present and present at t+1 | `:264-271` | matches, plus a T_15 convention (N2) |
| late-born = birth > pole's min birth | `:274-280` | matches |
| T_1 exclusion | `:330-331` | matches |
| B3: readings must agree to enter the count | `:332-333` | matches |
| B4: clusters in one sample only excluded | `:334-335`, match rule `:283-286` | rule is new (open item 6) |
| main = largest max\|a\| at T_15 | `:313` (max over exemplars at the last grid point; tie → lowest c) | matches |
| r(c) | `:323` | matches (signed magnitude) |
| p_u, clip, pole bootstrap, a_ctrl in same draw | `:380-414` | matches (checked in §3) |
| H_RG upper < 0.10, H_SP lower > 0.20, mixed | `:417-424` | matches |
| validity gate a_ctrl ≥ 0.80 | `:594-612`, `:772` | threshold matches; count not checked (N7) |
| flag: ≤1 of 4 random correct | `:758` | matches |
| H_MIG | `:785-786` | matches |
| b and within-pole permutation null (10,000) | `:427-444` | null matches (checked in §3); no bootstrap interval (N3) |
| v, lower bound > 0.5 | `:447-463`, `:790-799` | matches |
| Holm over three p-values | `:466-473` formula correct; family `:945-949` | family varies (N4) |
| Wilson | `:217-225` | matches: 3/20 → [0.05237, 0.36042] by hand and by code |
| growth T_12 → T_15 | `:324-325` | matches; no zero guard (N6) |
| suppressor flag | `:239-243`, `:638` | rule matches; sign of vocab for negative poles open (open item 1) |

Prompts against the essay's protocol (essay.md 305, 373, 389-395, 573-577, 637, 643): the writer prompt asks for five positives, five near misses, a most-distant concept with five texts, and one blind-maximum sentence. The explainer prompt carries the three caveats of SPEC §2.2. `second_reader.md` equals `explainer.md`, and `ensure_manifest` enforces that (`neurons_explain.py:116-118`). The clusterer prompt is the exception (N8).

### 2b. Findings table

| id | file:line | spec clause | what the code does | severity | fix |
|---|---|---|---|---|---|
| N1 | `neurons_verdict.py:248-261`, `:372` | §2.4 presence uses \|a_t\| | Presence and r(c) use max(±a, 0) in the pole's direction. An exemplar with the opposite sign at t counts 0, not \|a\|. | changes a number (births where exemplars flip sign early in training) | Write the signed reading into §2.4 as a dated amendment, or use \|a\|. |
| N2 | `neurons_verdict.py:264-271` | §2.4 birth needs presence at the next grid point | At T_15, presence alone counts as birth. Read literally, a T_15-only cluster is never born. Here it is late-born with t_before = T_14 and enters p_u. | changes a number | State the T_15 convention in §2.4 before freeze, or return None. |
| N3 | `neurons_verdict.py:427-444`, `:800` | §7: p_u, b, v bootstrap over poles, 10,000 resamples | `b` has only the permutation null and p. There is no bootstrap interval. | changes a number (a required interval is absent) | Add a pole bootstrap CI for b. |
| N4 | `neurons_verdict.py:945-949` | §7 Holm over the three p-values | p_u's p enters only when the rule is H_RG or H_SP, and its direction (p_RG or p_SP) is picked from the observed rule. When p_u is mixed or invalid, Holm runs over two, which lowers b's and v's adjusted p. | changes a number | Fix the family at three and pre-register p_u's one-sided p, or state that p_u is never Holm-tested. |
| N5 | `neurons_verdict.py:134-138`, `:643`, `:652-656` | CLAUDE.md fail loud | A NaN max_a/min_a at the explained pole gives score NaN. `n_strong` counts it not strong; the probe showed a pass with NaN in a positive slot. Only shuffled scores are checked (`:146`). | changes a number silently | Raise in `score` when max_a or min_a is not finite. |
| N6 | `neurons_verdict.py:323-325` | fail loud | r(c) and growth divide by max_abs with no zero check. In the probe, max_abs = 0 at T_12 gave growth = −inf, counted as shrinking in v, with only a RuntimeWarning. | changes a number silently (degenerate input) | Raise when max_abs ≤ 0 at the birth point, T_12 or T_15. |
| N7 | `neurons_verdict.py:594-601` | §6.3 validity gate, CLAUDE.md "every gate has a failing input" | `judge-gate` does not check judgments per pole (`judged()` checks 4+2 at `:718`; the gate does not). One judgment per pole passed with a_ctrl = 1.0 in the probe. | changes a number (the gate) | Run the same 4-random/2-hard count check before computing a_ctrl. |
| N8 | `neurons_prompts/clusterer.md:11-12` | §8 double-negative threat; §2.4 clusters | The clusterer must put double-negative echoes in their own cluster. Those clusters then go through births and the parent test as jobs. The explainer is told not to count echoes as jobs. An echo cluster's antecedent is the other pole, so it should fail the 2AFC, raising A's shortfall and p_u toward H_SP. The spec is silent. | changes a number (the decisive one) | Add an `echo` flag to the clusterer schema and pre-register excluding echo clusters from p_u, reporting them as a count. |
| N9 | `neurons_verdict.py:743-807` | §4.4, §7 (population-weighted beside stratified), §5 G2, §2.4 f_ctrl | No population reweighting of p_u, b or v by class stratum. No G2 readout. No control flag rate f_ctrl. | changes a number (required readouts absent) | Add stratum weights from the audit classes; add G2 and f_ctrl, or strike them from the spec. |
| B1 | `neurons_verdict.py:478-496`, `:509-530` | §9.3 PREDICTIONS before the first W2-C and W4 artifact | The date is whatever the heading text says, and the earliest heading wins. Probes that passed: an empty `## PREDICTIONS 2026-10-01` stub plus a later real entry; a heading dated 2019-01-01 written today; `2026-10-19T20:00-10:00` (true 06:00Z on the 20th) against an artifact at 2026-10-20T00:00Z, because offsets and zone-less local times are read as UTC. A later-dated entry is refused only when no earlier heading exists. | changes blindness | Date the entry by the git commit that added the heading (`git log -S`), parse offsets or require `Z`, and refuse if any PREDICTIONS heading postdates an artifact. |
| B2 | `neurons_explain.py:263-271`, `:280-282` | §2.2 the writer sees only a one-sentence explanation, never a context or an activation | The explainer's `explanation` passes to the writer unchanged. A two-sentence explanation quoting `⟦boat⟧`, "context #0" and `a=+9.95` reached the writer's `user` text in the probe. | changes blindness | Add a `check` on explainer collect that rejects ⟦ ⟧, `#n` rank references, `a=` values and more than one sentence. |
| B3 | `neurons_explain.py:488-489` | §2.4 no pole ids or identifiers to the judge | The judge id is `judge:<pole>:c<idx>:<k>`. k is deterministic: k = 4, 5 is always the hard decoy, and c0 is usually the control. The API backends do not send the id (`neurons_llm.py:270-273`, `:304-310`), but a manual-backend agent reads the whole line. | changes blindness (manual backend) | Use random opaque ids and keep the mapping in `plans/`. |
| B4 | `neurons_explain.py:483-497`; `MANUAL_BACKEND.md:30-34` | §2.4 blind 2AFC | Within one sample file the true record appears in three requests and each decoy once. In my run the true record repeated more often than its decoy in 11 of 18 requests per file. Only agent compliance (one agent per line) prevents use. | changes blindness (manual backend) | In manual mode, write one requests file per judge line, or place each cluster's decoys in different files. |
| B5 | `neurons_llm.py:196-257`; `neurons_explain.py:263-271` | §2.2 Agent 3 is an independent sample | No check that the second reader is not a copy. A byte-identical copy of the explainer's responses was collected without error. | changes blindness (inflates kappa) | Refuse, or flag and log, when a second-reader text equals the explainer text for the same pole. |
| B6 | `MANUAL_BACKEND.md:27-29`, `:35-37` | §2.2 the writer sees only its explanation | In manual mode one agent answers a whole writer batch, so it sees every pole's explanation. Distant and near-miss texts can borrow other poles' topics. | changes blindness (manual backend, low) | One agent per writer line, as for the judge. |
| B7 | `neurons_verdict.py:568-577`, `:601-611`, `:914-917` | §9.2 W4 validity gate recorded before any S judgment is read | `theta.json` and `judge_gate.json` carry no creation time, and nothing ties them to discovery inputs. `unblind` accepts any file whose gate is `pass`. | changes blindness | Write `created_utc` and input manifest hashes; in unblind, check the judge-gate time precedes the S explain manifest's first judge batch. |
| C1 | `neurons_explain.py:283`, `:489` | DATA_CONTRACT §3 id = `<role>:<pole or cluster>:<sample>` | writer_job ids end `:j<k>`; judge ids have four fields, the last a per-cluster counter. | cosmetic | Amend the contract or the ids. |
| C2 | `neurons_explain.py:509-512`, `:324-328`, `:344-348`, `:498`; `neurons_verdict.py:574-577`, `:608-611` | DATA_CONTRACT §3-4 file list | judgments rows add `id` and carry `control` on every row. New files: `probe_poles.json`, `positions_poles.json`, `plans/`, `verdict/theta.json`, `verdict/judge_gate.json`. | cosmetic | List them in the contract. |
| C3 | `neurons_verdict.py:115` | DATA_CONTRACT §2 double-negative share = sign_counts[3]/K | Divides by sum(sign_counts), and returns 0.0 for an empty top-K, so dead poles land in the "< 0.8" bucket. | cosmetic | Pick one definition in the contract; raise on empty. |
| C4 | `neurons_verdict.py:970` | DATA_CONTRACT §4 names `studies/already-clean/JOURNAL.md` | `--journal` takes any path. | cosmetic | Default to and check the fixed path. |
| C5 | `neurons_explain.py:263-271`, `:308-320`, `:402-414`, `:501-513`; `neurons_llm.py:247-256` | fail loud with a recovery path | A multi-batch collect logs each batch to `calls.jsonl` before validating the next. Probe: second_reader malformed → explainer ids logged, `explanations.jsonl` not written; after fixing, the rerun refuses ("ids already in calls.jsonl"). Recovery needs hand-editing `calls.jsonl`. | cosmetic (no wrong number; blocks the run) | Validate every batch first, then log and write once. |
| C6 | `neurons_verdict.py:812-823` | §10 Q1 samples per pole with the strong flag | Only the gallery (one pass and one fail per class) is written, with scores but no strong flags. | cosmetic | Write every pole's record to `samples/`. |
| C7 | `neurons_verdict.py:692-703` | §10 F2 strong rates | RARE poles' texts count in the strong and packing rates. The spec is silent. | cosmetic | Exclude RARE, or report both. |
| C8 | `neurons_verdict.py:647-648` | §5 width band θ_s ± 0.1 | θ_s − 0.1 can be ≤ 0, making every non-negative text strong in that band. | cosmetic | Floor the band at a positive value and say so. |
| C9 | `neurons_verdict.py:724-740` | §2.4 center curve: max sentence and the five positives | Only the max sentence is scored. | cosmetic (exploratory) | Score the five positives too. |

Checked and found correct: no try/except swallows anything (the one at `neurons_llm.py:233-237` records the error and `collect` raises). No default model name exists in non-test code (`--model` is required). No temperature is sent to the Anthropic API. `output_config` with `effort` and `format` is the documented Messages API form (claude-api skill, read this session). `minItems`/`maxItems`/`minimum`/`maximum` are stripped for the API and enforced in `collect`. A refusal or `max_tokens` stop raises. A missing, duplicated, extra, model-mismatched or schema-invalid answer fails `collect` and lists every id; my own run tripped the model-mismatch check. The judge `user` text carries no digits and no checkpoint label. Records have equal length. Exemplar positions are removed from both records. A/B order is drawn per call from a seed logged in `explain/manifest.json` (`{"judge": 5}`); 19 true-first and 17 decoy-first in my run. Sample 0 and sample 1 show the same two records with independent order coins.

## 3. Observed p_u recovery (my runs)

Generator copied from `test_neurons_verdict.py:153` (`synthetic`): n = 400 poles with one late-born cluster each, a_ctrl = 0.95, four judgments per cluster and per control, n_boot = 10,000, bootstrap seed = data seed.

| true p_u | seed | A | a_ctrl | point | 95% CI | rule | covers truth |
|---|---|---|---|---|---|---|---|
| 0.05 | 0 | 0.9275 | 0.9481 | 0.0460 | [0.0057, 0.0855] | H_RG | yes |
| 0.05 | 1 | 0.9225 | 0.9563 | 0.0740 | [0.0373, 0.1103] | mixed | yes |
| 0.05 | 2 | 0.9287 | 0.9481 | 0.0432 | [0.0043, 0.0814] | H_RG | yes |
| 0.05 | 3 | 0.9263 | 0.9481 | 0.0488 | [0.0098, 0.0878] | H_RG | yes |
| 0.05 | 4 | 0.9437 | 0.9513 | 0.0166 | [0.0000, 0.0491] | H_RG | no |
| 0.30 | 0 | 0.8331 | 0.9481 | 0.2566 | [0.2011, 0.3122] | H_SP | yes |
| 0.30 | 1 | 0.8131 | 0.9563 | 0.3137 | [0.2594, 0.3675] | H_SP | yes |
| 0.30 | 2 | 0.7981 | 0.9481 | 0.3347 | [0.2761, 0.3936] | H_SP | yes |
| 0.30 | 3 | 0.8163 | 0.9481 | 0.2943 | [0.2361, 0.3522] | H_SP | yes |
| 0.30 | 4 | 0.8394 | 0.9513 | 0.2479 | [0.1931, 0.3032] | mixed | yes |

Over seeds 0-199: truth 0.05 → H_RG 141/200, mixed 59, never H_SP; CI coverage 186/200; mean CI [0.0140, 0.0901]. Truth 0.30 → H_SP 186/200, mixed 14, never H_RG; coverage 191/200; mean CI [0.2396, 0.3584]. This agrees with SPEC §15.7 (139/200 and 185/200; the small gap is bootstrap seeding).

Pole resampling: 200 poles × 3 clusters with identical accuracy within a pole. The code's CI width is 0.1100; a cluster-level bootstrap I wrote gives 0.0617. The ratio is 1.78 (√3 = 1.73), so clusters move with their pole. a_ctrl is drawn from the same index array (`:401-405`).

Permutation null: 50 poles with births [1,1,1,1] plus 50 with [1,5] (main the later one). The code's null mean is 0.7496 against a within-pole expectation of (1 + 0.5)/2 = 0.75; b = 0.5; p = 1.0. The null permutes within each pole.

Wilson: 3/20 by hand gives [0.05237, 0.36042]; the code gives the same. 0/10 → [0, 0.2775]; 10/10 → [0.7225, 1.0].

## 4. Weak tests

- `test_p_u_recovery_on_random_synthetic_judgments` (`test_neurons_verdict.py:160-166`): a single seed (0) that passes. At seed 1 (truth 0.05) and seed 4 (truth 0.30) the asserted rule is `mixed`, so the test pins one lucky realization. One cluster per pole cannot tell pole resampling from cluster resampling.
- `test_p_u_recovery_at_the_expected_realization` (`:169-179`): also one cluster per pole; same blind spot.
- `test_birth_order_permutation_test_on_a_planted_ordering` (`:189-199`): every pole has births [1,4,7]. A permutation pooled across poles gives the same null, so "within each pole" is not pinned.
- `test_judge_records_exclude_exemplars_and_show_no_numbers` (`test_neurons_explain.py:269-308`): exclusion is checked only for the late cluster, not the control records. The last assertion requires the pole id inside the request id, so it asserts B3 instead of catching it. A/B balance and the logged seed are not checked.
- `test_pass_rule_and_the_essay_rule` (`test_neurons_verdict.py:42-53`): no score sits exactly at θ_s, so `>=` against `>` is not pinned for the frozen rule.
- `test_vestigial_rule` (`:202-205`): tests the pooled ratio only. The "born at or before main and shrinking" composition (`:798`) is exercised only by the end-to-end case where v = 0.
- `test_unblind_q1_end_to_end` (`:303-338`): the per-pole class check covers only the six gallery poles (L01N1- and L01N3+ are not checked one by one). `essay_rule` is asserted as win ≥ 5.
- `test_predictions_gate_refuses_and_accepts` (`:216-235`): it asserts that an earlier heading wins over a later one, which pins the weakness in B1. There is no offset or local-time case.
- `test_writer_sees_only_the_sentence_and_texts_carry_a_shuffle` (`test_neurons_explain.py:176-200`): the fixture's explanations never contain ⟦ or numbers, so B2 cannot show.
- Missing: no test for a partial multi-batch collect (C5), a NaN probe value (N5), a zero max at T_12 (N6), or a judge-gate with too few judgments (N7).

## 5. Could not settle

1. Vocabulary sign for negative poles. DATA_CONTRACT §2 does not say whether `vocab.json` is flipped for the negative pole. The explain side shows it as given (`neurons_explain.py:229`, `:239`), and `single_token_suppressor` reads it as given. If the harvest does not flip, negative poles get the opposite output view and the suppressor flag reads the wrong sign. The harvest code is not on this branch.
2. Whether the API's `msg.model` equals the requested id string exactly. If not, every `collect` fails loudly. No API call was made.
3. Exact-position exclusion (spec level). §2.4 removes only exact `(doc_id, position)` matches. A record context one or two tokens from an exemplar shares about 79 of its 81 rendered tokens, so the judge could match by verbatim text. The code follows the spec; the size of the effect on A is empirical.
4. The control matches the test only in kind (spec level). a_ctrl is measured on `main` against T_14. Late-born clusters are tested against earlier t_before, where records are noisier. If related clusters score below a_ctrl there, p_u is biased toward H_SP. The §6.3 model and the simulation above both assume equal accuracy.
5. Equal corpus and provenance (`neurons_verdict.py:342-349`, `:122-129`). Grid points are checked by branch name only. Nothing checks that every grid harvest scored the same 20M tokens (`n_tokens_scored`, `tokens_sha256`), that positions and clusters came from the 20M T_15 reading, or that the probe harvest has the same revision as the stats harvest. Whether a wrong input gets passed depends on job arguments not yet written.
6. B4 match rule (`neurons_verdict.py:283-286`). "More than half of its exemplars in one cluster of the other sample" is a new rule. It should go into SPEC §8 before freeze.
7. ONE with a label/job-count disagreement (`neurons_verdict.py:205`). A label of ONE with two listed jobs becomes UNK. SPEC §2.3 says "Agent 1 says one job" without naming which field decides. The choice can move the ONE and UNK fractions.
