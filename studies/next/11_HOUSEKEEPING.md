# 11. Housekeeping after the retag, 2026-11-05

Written 2026-11-05 by an Opus worker in worktree `interp_research-clean-neurons` (branch `clean-neurons`). CPU only.
No GPU, no paid API, no S3 call, no download. Nothing committed. No worktree removed, no file deleted.

## Answer

- **Figures.** The old command `python3 -m infra.pg.neurons_figures --unblind bank/w5 --out <dir>` writes all six
  figures again (exit 0). Its F4 now comes from `draw()` in `studies/already-clean/runbook/f4_births.py`, with the
  chance level. That F4 equals the committed `studies/already-clean/results/figures/F4_births.svg` byte for byte
  (sha256 `dfc7b713ff5d94eb…`). F1 to F3, F5 and F6 also equal their committed files byte for byte.
- **Tests.** `infra/pg/tests/test_neurons_verdict.py`: 1 failed, 39 passed, 7 errors before; 47 passed after.
  `infra/pg/tests/test_neurons_explain.py` uses the same fixture: 23 failed, 36 passed before; 59 passed after.
- **Fixture cause.** Commit `dce6f9810` (2026-10-24, F41) made `pole_vocab` read `poles/vocab.json` per pole. Since then it
  refuses a "-" entry that is not the sign flip of the same neuron's "+" entry (`infra/pg/neurons_verdict.py:180-183`). The
  fixture (`infra/pg/tests/neurons_fixture.py`, old lines 94-97) still wrote one entry for both signs. It also listed
  "-" poles without their "+" pole. Two unit tests still expected the old layout. The fixture and the two tests are test-data
  errors from before F41. Each fix is a few lines.
- **Worktrees.** Carry 22 files: the `studies/neuronlineage/` study folder (7 files); the depthoff errata, the retime
  spec and the `depth_off` page (6); `walkthrough/VISUALS_PLAN.md` (1); and the `loop_or_answer` and `which_columns`
  pages (8). All 22 are copied into `clean-neurons` at the same paths and their sha256 verified. Drop:
  `neuronslider`'s `studies/already-clean/pages/neuron_across_training/` is an older copy of the committed walkthrough
  module (its `data.js` is identical by sha256), plus two `__pycache__/*.pyc` files.

## 1. The F4 path in `infra/pg/neurons_figures.py`

Change:
- `f4_births.py`: the body of `main()` is split. `draw(m, source, shown)` returns the SVG text and the summary
  (`height`, `b`, `chance_b`, `where`). `main()` keeps the sha256 check of the local metrics copy, calls `draw()` with
  the footer name `bank/w5/verdict/metrics.json`, writes the file and prints the same JSON as before. The asserts
  that tie the chance files to the metrics (births counts, b, pole count) are unchanged and now run on every call.
- `neurons_figures.py`: `f4(metrics, source)` loads `f4_births.py` from its path (`load_f4_births()`; the folder
  name `already-clean` cannot be imported as a package) and returns `draw(metrics, source, str(source))[0]`. The footer
  names the source path as given, as the other five figures do. The module docstring says F4 also reads three
  files under `studies/next` and refuses births or b other than the seventh run's.

Before the change (scratch directory `$T` holds `bank/w5/verdict/metrics.json`, sha256 `a95bfa57bfe65246…`, copied from
`/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json`):

```
cd $T && PYTHONPATH=<worktree> python3 -m infra.pg.neurons_figures --unblind bank/w5 --out fig_before
→ exit 1, RuntimeError: F4 is drawn by studies/already-clean/runbook/f4_births.py since the 2026-11-05 retag […]
  No figure was written.
python3 -I studies/already-clean/runbook/f4_births.py --out $T/f4_script_before
→ sha256 dfc7b713ff5d94eb…, identical to the committed F4_births.svg
```

After the change:

```
cd $T && PYTHONPATH=<worktree> python3 -m infra.pg.neurons_figures --unblind bank/w5 --out fig_after
→ exit 0, six paths printed
```

| figure | bytes | sha256 (first 16) | against the committed file in `results/figures/` |
|---|---|---|---|
| F1_audit_classes.svg | 9,737 | bd0b6c9cd0b528da | equal (cmp) |
| F2_strong_rates.svg | 8,225 | 842b02c97734ded8 | equal |
| F3_grid_population.svg | 19,149 | 2f2ad32d7ccf2123 | equal |
| F4_births.svg | 14,264 | dfc7b713ff5d94eb | equal |
| F5_parent_test.svg | 5,268 | 7e5e87c69a2731ea | equal |
| F6_gallery.svg | 20,675 | 9e723868494d76b2 | equal |

- The script itself after the refactor: `python3 -I studies/already-clean/runbook/f4_births.py --out $T/f4_script_after`
  prints `{"wrote": …, "height": 646, "b": 0.47560975609756095, "chance_b": 0.5300452961672475, "where": "below",
  "sha256": "dfc7b713…"}`, identical to the committed file.
- Byte equality needs the documented form `--unblind bank/w5`, run from the folder that holds `bank/`. With an absolute
  `--unblind` path the footer prints that path, wraps onto one more line, and the SVG is 658 px high instead of 646.
  The footer lines are the only difference. The other five figures print the path in the same way.
- Failing input: the same metrics with one main birth count raised by 1 → `AssertionError: births`, exit 1, no figure
  written. F4 refuses births that are not those the chance level was computed on.

## 2. Tests

Command (from the worktree root): `uv run --offline --no-project --with transformers --with tokenizers --with numpy
--with pytest python -m pytest <file> -q`.

| file | before any change | after every change |
|---|---|---|
| `test_neurons_verdict.py` | 1 failed, 39 passed, 7 errors | 47 passed |
| `test_neurons_explain.py` | 23 failed, 36 passed (22 `KeyError: 'L00N1+'`, 1 sign-flip `ValueError`) | 59 passed |

Steps between, for `test_neurons_verdict.py`: the fixture fix alone gives 3 failed, 44 passed (the unit test of fix 3
and the two figure tests). Adding fix 3 gives 2 failed, 45 passed: the two figure tests stop at the refusing f4. The F4
change and the two updated figure tests give 47 passed. For `test_neurons_explain.py`: the fixture fix alone gives
1 failed, 58 passed, and fix 4 gives 59 passed.

The "before" counts for `test_neurons_explain.py` come from a `git archive HEAD infra` copy in the scratch directory.
That copy also gives 1 failed, 39 passed, 7 errors for the verdict file, the same as the worktree.

Cause and fixes:
1. `neurons_fixture.build_harvest` wrote `vocab.json` with the same entry for "+" and "-". After F41, `pole_vocab`
   checks a "-" entry against its "+" entry (`neurons_verdict.py:183`). Fix: the fixture writes the "-" entry as the
   flip of the "+" entry, as `neurons_harvest.vocab_table` does (sign -1 for the "-" pole, line 450).
2. After fix 1, `pole_vocab` stopped with `KeyError` on "-" poles whose "+" pole is not listed (`POLES` in
   `test_neurons_verdict.py:744` holds `L00N2-` without `L00N2+`). It reads the "+" entry at `neurons_verdict.py:181`. The
   real files hold both poles of every sampled neuron (JOURNAL 2026-10-25 F41: "checked on all 320 O7 neurons"). Fix:
   the fixture writes both poles of every listed neuron. `poles.json` is unchanged.
3. `test_pole_vocab_flips_the_negative_pole_and_the_suppressor_flag_follows` gave the same entry to both signs and
   expected the function to flip it. It now gives the flipped "-" entry and expects the file's own entry back. It also
   checks that an unflipped "-" entry is refused.
4. `test_negative_pole_shows_the_neurons_vocabulary_swapped` (`test_neurons_explain.py`) compared the "-" prompt with
   the "-" entry swapped. It now compares it with the "+" entry swapped (one line, docstring updated).

The two figure tests:
- `test_figures_render_to_scale_from_metrics_alone`: the new F4 exists only for the seventh unblind run, so the test
  draws from that run's `metrics.json`. It reads the local bank copy named by `f4_births.METRICS`, copied into
  `tmp_path/bank/w5/verdict/`, with `monkeypatch.chdir(tmp_path)`. It keeps the old checks: six well-formed SVGs, the
  source in every footer, bars to scale in F1, F2, F4 and F5, the F1 and F5 titles from the metrics, F6's poles from
  `q2.gallery`, and `FileExistsError` on a second build. It adds: F4 holds "chance level: random same-size groups" and
  "so b is below chance", and F4 equals the committed `F4_births.svg` byte for byte.
- `test_figures_refuse_missing_inputs_and_write_nothing`: the `q2.gallery` case now starts from the seventh run's
  metrics, so F1 to F5 draw and F6 refuses on `q2.gallery`. The other two cases still use the synthetic bank. They stop
  at F3, before F4. One case is added: the synthetic bank as it is → `AssertionError: births`, and no folder is
  written.
- Failing input for the first test: `draw()` patched to write "above chance" in place of "below chance" → the test
  fails at the F4 text assertion (scratch script `broken_f4_check.py`).
- Dependence: both tests now read `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json`,
  the file `f4_births.py` already requires. Without that file they stop with `FileNotFoundError`.

## 3. `write_registry.py` and `acceptance_check.py`

Changes:
- `runbook/write_registry.py`: the figure loop takes an inputs cell per row. F4's row now has the content, keys,
  inputs and kind of the committed `results/REGISTRY.md` row 12. Its inputs are metrics.json plus `05_chance.json`,
  `04_b.json` and `s03_histogram_tv.json` with their sha256, hashed from this repo. b's interval is read from
  `04_b.json`. The diff below showed that the b row (committed line 27) also lacked its retag mark. The script now writes
  that mark, with the chance level read from `05_chance.json` → `summary.R1t`. The docstring names the three files.
- `results/REGISTRY.md` row 12: "which now refuses and names that script" → "which since the housekeeping of
  2026-11-05 calls that script's draw()". The script writes the same words.
- `runbook/acceptance_check.py` line 97: F4's check now looks for "chance level (black line on each bar)", and the
  evidence text reads "F4 chance level of random same-size groups (runbook/f4_births.py, 2026-11-05 retag)".
  At HEAD the check looked for the opening words of the old F4 caption.

`write_registry.py` runs (it takes the explain worktree root; no explain worktree is on this laptop, so a scratch
root `E` holds the local bank copies of `bank/w5/verdict/metrics.json` and `bank/w4/o7_s_v2/explain/{judgments,
clusters}.jsonl`):

```
python3 -I studies/already-clean/runbook/write_registry.py $T/E
→ exit 1, FileNotFoundError: …/bank/w4/o7_s_v2/explain/positions.npz
```

`positions.npz` and the G2 input `bank/w4/o7_t0_g2/explain/explanations.jsonl` are not on this laptop. With empty
placeholder files for those two (scratch root `E2`):

```
python3 -I studies/already-clean/runbook/write_registry.py $T/E2
→ REGISTRY.md: 37 lines (exit 0)
diff E2/bank/w5/REGISTRY.md studies/already-clean/results/REGISTRY.md → lines 3 and 26-34 differ
```

With the two placeholder hashes (`e3b0c44298fc1c14`, the sha256 of an empty file) replaced by the committed ones
(`965c88e55ad5f2d9`, `26fae1b3fbec1ad8`), only line 3 differs. Its git fields are empty because `E2` is not a git
checkout. The HEAD script in the same setup differs at lines 3, 12 (F4) and 27 (b).

`acceptance_check.py` runs:
- The full script needs an explain worktree, a harvest worktree and an S3 listing. Run under a wrapper whose `s3ls`
  returns an empty list (no S3 call), on scratch roots holding the local bank copies, the six SVGs from §1, the
  committed pdf/ and png/, `REGISTRY.md` and `RESULTS.md`. It writes 4 PASS rows (Q1 samples for Q4B, G4 and O7, and Q2
  samples). It then stops at line 64: `FileNotFoundError: …/bank/w5/verdict/grid.md`. Further on it would also need
  the harvest manifest `bank/neurons/w3/grid/1/out/t15_step1413814/manifest.json`, which the local bank lacks. It would
  also need `handoff/S6.md`, `checks/` and `INDEX.md` in the study folder, which moved to `archive/` on 2026-11-04.
- The "§10 figures" block alone, copied verbatim from the file and run on the same figure folder (scratch
  `ac_figblock.py`):

| script | F4 file | verdict | F4 evidence |
|---|---|---|---|
| after the change | new F4 (from the old command) | PASS | "F4 chance level of random same-size groups (…) True" |
| after the change | pre-retag F4 (`git show f441f00f2^:…/F4_births.svg`) | SUBSTANCE | "… False" |
| HEAD | new F4 | SUBSTANCE | "F4 §6.4 band False" |

## 4. The rest of the list "Left for their owners" in `10_RETAG.md`

- "re-harvests cluster b" was already reverted in commit `e2e7b03c3`. `results/RESULTS.md:142` reads "re-harvests the W3
  harvest job b". No change.
- §5 line numbers: the two rows for `runbook/results_hand_edits.py` now give 44, 58 (old sentences; marks at 47, 59),
  27 (comment), 31, 38, 90, 91 (Retag section and grid label), and `runbook/f4_births.py:66` (caption).
- Each bullet of the list now ends with a dated "Done" pointer to this file.

## 5. Worktree inventory

All five branches are merged into `clean-neurons` (`git branch --merged clean-neurons`). Each worktree's tracked
files are clean; `git status --porcelain` shows only untracked entries. "In clean-neurons" was checked three ways: by
path, by sha256 against every file of the `clean-neurons` working tree of the same size, and by git blob or LFS oid on
every local branch.

| worktree (branch, HEAD) | untracked content | files, bytes | what it is | in clean-neurons | verdict |
|---|---|---|---|---|---|
| `interp_research-neuronslider` (`neuronslider`, a36a23e5e) | `studies/already-clean/pages/neuron_across_training/` | 4, 2,710,705 | Earlier prototype of the "one neuron across training" page. | Not by path. `data.js` identical by sha256 (`c6c81ef1e7b9ab08`) to `studies/already-clean/walkthrough/neuron_across_training/data.js`, whose README records it. `README.md`, `extract.py` and `index.html` are older versions; `extract.py` reads the old explain and harvest worktrees. | drop (superseded) |
| `interp_research-loopanswer` (`loopanswer`, a36a23e5e) | `studies/instruments/pages/loop_or_answer/` | 4, 3,907,306 | Prototype page: Qwen3.5-0.8B answers to 300 MATH-500 problems under nine E2 arms, with the onset of repetition. | No: not by path, sha256, blob or LFS oid on any local branch. | carry (copied) |
| `interp_research-whichcols` (`whichcols`, a36a23e5e) | `studies/instruments/pages/which_columns/` | 4, 194,863 | Prototype page: which `down_proj` columns of blocks 15 and 19 the E2 arms edit. | No. | carry (copied) |
| `interp_research-walkthrough-slider` (`walkthrough-slider`, 0ae65bc2b) | `studies/already-clean/walkthrough/VISUALS_PLAN.md` | 1, 15,659 | Plan for the walkthrough pictures (4 Nov). Committed files cite it as `../VISUALS_PLAN.md` (`walkthrough/births/README.md:3`, `explanation_test/README.md:3`, both `INSERT.md`, `births/extract.py`). | No. | carry (copied) |
| same | `studies/depthoff/ERRATA_2026-11-04.md`, `SPEC_retime.md` | 2, 61,718 | Four corrections to the DEPTHOFF record; a draft pre-registration that times one `predict_action` call with depth on and off. | No. | carry (copied) |
| same | `studies/depthoff/pages/depth_off/` | 4, 58,737 | Prototype page: 194 of 200 against 195 of 200, call time 2.16 s against 0.98 s. The errata cite it as a source. | No. | carry (copied) |
| same | `…/walkthrough/neuron_across_training/__pycache__/extract.cpython-314.pyc` (git-ignored) | 1, 19,304 | Python bytecode cache. | — | drop |
| `interp_research-neuronlineage` (`research/neuronlineage`, a36a23e5e) | `studies/neuronlineage/` | 7, 285,389 | Study folder: SPEC DRAFT v1.2 (not frozen), already-done audit, literature record, three reviews. 13 jobs, 0 run. | No. `studies/neurons/PROGRAM.md:72` names the branch as already inside `main`; the files were never committed. | carry (copied) |
| same | `infra/pg/__pycache__/read_from_passage.cpython-314.pyc` (git-ignored) | 1, 21,027 | Python bytecode cache. | — | drop |

Copies made (source worktree → the same path under `clean-neurons`; every destination was absent, `cp -p`, sha256 equal
after the copy):

| from | path | bytes | sha256 (first 16) |
|---|---|---|---|
| neuronlineage | `studies/neuronlineage/AUDIT_2026-11-04.md` | 94,225 | e9adafaccc7ddf6e |
| neuronlineage | `studies/neuronlineage/LITERATURE_2026-11-04.md` | 39,241 | 4244ddd09a3e3cb2 |
| neuronlineage | `studies/neuronlineage/README.md` | 651 | 898b96a19efc639c |
| neuronlineage | `studies/neuronlineage/SPEC.md` | 68,159 | f3e33012c2c192d0 |
| neuronlineage | `studies/neuronlineage/review_2026-11-04_critic_v1.md` | 33,813 | 593f846347554ed8 |
| neuronlineage | `studies/neuronlineage/review_2026-11-04_critic_v1_1.md` | 25,229 | 70dfc9c106979e30 |
| neuronlineage | `studies/neuronlineage/review_2026-11-04_verify_v1.md` | 24,071 | 5364ed61942c655a |
| walkthrough-slider | `studies/already-clean/walkthrough/VISUALS_PLAN.md` | 15,659 | 6e1680f6acfcf721 |
| walkthrough-slider | `studies/depthoff/ERRATA_2026-11-04.md` | 12,008 | 43f5cf9046e08def |
| walkthrough-slider | `studies/depthoff/SPEC_retime.md` | 49,710 | 115b56ec35a24c57 |
| walkthrough-slider | `studies/depthoff/pages/depth_off/README.md` | 5,237 | 03d071fbcce6ac5a |
| walkthrough-slider | `studies/depthoff/pages/depth_off/data.js` | 11,973 | eb02f9cb03428417 |
| walkthrough-slider | `studies/depthoff/pages/depth_off/extract.py` | 9,865 | 173f5039daac22d1 |
| walkthrough-slider | `studies/depthoff/pages/depth_off/index.html` | 31,662 | 89009467a4798b0e |
| loopanswer | `studies/instruments/pages/loop_or_answer/README.md` | 6,077 | db7d9cd166a49de6 |
| loopanswer | `studies/instruments/pages/loop_or_answer/data.js` | 3,858,298 | 38d96e0a23e5942a |
| loopanswer | `studies/instruments/pages/loop_or_answer/extract.py` | 11,181 | f95def3dbbbc4282 |
| loopanswer | `studies/instruments/pages/loop_or_answer/index.html` | 31,750 | f96b97ba95cb94dd |
| whichcols | `studies/instruments/pages/which_columns/README.md` | 4,457 | e5b7fef510f98fca |
| whichcols | `studies/instruments/pages/which_columns/data.js` | 151,088 | eb1b7e7b9c2da9e3 |
| whichcols | `studies/instruments/pages/which_columns/extract.py` | 7,218 | 363fe3bbb8139cde |
| whichcols | `studies/instruments/pages/which_columns/index.html` | 32,100 | f75855c7e4ef371a |

Notes on the copies:
- The task named pages as leftovers when they exist elsewhere. The three instrument and depthoff pages exist nowhere
  else in git or in `clean-neurons`, so they are carried.
- `studies/depthoff/` in `clean-neurons` now holds only these six files. The study itself is on branch `depthoff`
  (tag `depthoff-closed`, 93a22b186), which is merged into neither `main` nor `clean-neurons`. The errata say they
  apply "when branch `depthoff` is merged". Whether they are committed here or on `depthoff` is for the committer to decide.
- The copied `extract.py` files keep their old input paths. `which_columns/extract.py` reads
  `studies/instruments/results/e2/…`, which moved on 2026-11-04 to `studies/reproductions/instruments_e2/results/`.
  `loop_or_answer/extract.py` defaults to the main worktree's `studies/instruments/results/e2`, whose `generations/`
  files are untracked there. Neither was run. The copied `data.js` files are the only built copies.
- `studies/neuronlineage/README.md` still names its worktree `../interp_research-neuronlineage`. It is copied verbatim.

## 6. Files changed (none committed)

Edited:
- `infra/pg/neurons_figures.py`
- `infra/pg/tests/neurons_fixture.py`
- `infra/pg/tests/test_neurons_verdict.py`
- `infra/pg/tests/test_neurons_explain.py`
- `studies/already-clean/runbook/f4_births.py`
- `studies/already-clean/runbook/write_registry.py`
- `studies/already-clean/runbook/acceptance_check.py`
- `studies/already-clean/results/REGISTRY.md` (row 12, one phrase)
- `studies/next/10_RETAG.md` (§5, two rows; "Left for their owners", four "Done" pointers)
- `studies/next/11_HOUSEKEEPING.md` (this file, new)

Copied (new, untracked): the 22 files in §5.

Already modified in this worktree before this work, not touched here: `infra/aws_pg/jobs/next_pilot0_loss.py`,
`studies/next/JOURNAL.md`, `studies/next/pilot0/loss_check.py`, `studies/next/pilot0/reconstruct_batch.py`.

## 7. Not done, and why

1. `acceptance_check.py` and `write_registry.py` were not run end to end. The explain worktree's `bank/` (`grid.md`,
   `positions.npz`, the G2 input), the harvest manifests and the S3 listing are not on this laptop or cost money.
   `acceptance_check.py` also reads `handoff/S6.md`, `checks/` and `INDEX.md` at the study root, which moved to
   `archive/` on 2026-11-04. That is a stale-path defect for its owner.
2. The S3 copies (`s3://pg-lineage-769063704696/results/neurons/w5/`) still hold the old F4 (`10_RETAG.md` §6 item 3).
3. `pole_vocab` stops with `KeyError` on a "-" pole whose "+" pole is not in `vocab.json` (`neurons_verdict.py:181`). The
   real files hold both poles, so it does not fire on them. A harvest of single poles would stop there. The error is
   loud, so nothing is hidden; the code was not changed.
4. `10_RETAG.md` §6 item 2 and the Follow-ups item 2 still say f4 refuses. They are dated records of that state and
   were left as written.

## 2026-11-08 cleanup (head's audit)

Done by an Opus worker on the head's explicit lists. Nothing was deleted from disk. Branch refs and worktrees
were removed only where the branch is merged into `clean-neurons`.

### Step 1. Branch record before removal

Every local branch on 2026-11-08, before any deletion. Any deleted branch can be recreated with
`git branch <name> <sha>`. "Merged" is `git branch --merged clean-neurons` at clean-neurons `02879b983`.
"On origin" is whether `git branch -r` lists a branch of the same name.

| Branch | Tip sha | Tip date | Merged into clean-neurons | On origin |
|---|---|---|---|---|
| `abscheck` | `763c85fd3` | 2026-11-08 | yes | no |
| `bg/arms` | `11a77a5ef` | 2026-11-05 | no | no |
| `bg/core` | `1caf360e0` | 2026-11-05 | no | no |
| `bg/harness` | `fe441a2ca` | 2026-11-05 | no | no |
| `bio` | `ff5847e09` | 2026-11-06 | yes | no |
| `bio2` | `81069ba17` | 2026-11-07 | yes | no |
| `clean-neurons` | `02879b983` | 2026-11-08 | yes | no |
| `consolidation/docs` | `4982c663d` | 2026-11-04 | yes | no |
| `consolidation/lib` | `51894f10e` | 2026-11-04 | yes | no |
| `depthanchor` | `a0658e159` | 2026-09-09 | no | no |
| `depthoff` | `93a22b186` | 2026-09-10 | yes | no |
| `depthopen` | `f504aa60b` | 2026-09-10 | yes | no |
| `docs/consolidation` | `97e06357b` | 2026-11-04 | yes | no |
| `docs/walkthrough-2026-11-02` | `f3f19b17f` | 2026-11-04 | yes | no |
| `docs/walkthrough-groups` | `7d62e7b7c` | 2026-11-07 | yes | no |
| `drafts/already-clean` | `41f916b99` | 2026-10-24 | yes | no |
| `fix/instruments-results-errata` | `f613b9bc6` | 2026-11-04 | yes | no |
| `gh-pages` | `2dca1e037` | 2026-11-06 | no | yes |
| `loopanswer` | `a36a23e5e` | 2026-11-04 | yes | no |
| `m1` | `f7fbf955b` | 2026-11-05 | yes | no |
| `m1-read` | `f018f152f` | 2026-11-05 | yes | no |
| `m1x` | `698a14f8b` | 2026-11-05 | yes | no |
| `m2` | `07e93fd2d` | 2026-11-06 | yes | no |
| `m2-read` | `77ed57fdf` | 2026-11-06 | yes | no |
| `main` | `cd031af32` | 2026-11-04 | yes | yes |
| `n12-read` | `5ac085cfd` | 2026-11-06 | yes | no |
| `n13` | `585640e90` | 2026-11-08 | yes | no |
| `n13fig` | `864089b9a` | 2026-11-08 | yes | no |
| `n13read` | `6928784b6` | 2026-11-08 | yes | no |
| `n1read` | `0d98928c9` | 2026-11-07 | yes | no |
| `n22` | `637657988` | 2026-11-06 | yes | no |
| `n2vread` | `1a0a3266a` | 2026-11-07 | yes | no |
| `n6` | `ae6175c46` | 2026-11-08 | yes | no |
| `neurons/explore` | `8beb39866` | 2026-10-23 | yes | no |
| `neurons/show` | `da91edeb9` | 2026-10-26 | yes | no |
| `neuronslider` | `a36a23e5e` | 2026-11-04 | yes | no |
| `postcheck` | `a2bd47041` | 2026-09-09 | no | no |
| `research/bytegraft` | `3ec08340b` | 2026-11-07 | no | yes |
| `research/inst-sanity` | `fd23c8513` | 2026-11-04 | yes | no |
| `research/mechanics` | `a777f3335` | 2026-11-04 | yes | no |
| `research/mechanics-sanity` | `a79d2fc93` | 2026-11-04 | yes | no |
| `research/neuronlineage` | `a36a23e5e` | 2026-11-04 | yes | no |
| `research/reuse` | `8da101a08` | 2026-11-04 | yes | no |
| `research/sharedbirth` | `d69f2b6c2` | 2026-11-02 | yes | no |
| `science/public-check` | `835dc25be` | 2026-11-04 | yes | no |
| `science/realizednull` | `c7258ac2a` | 2026-11-02 | yes | no |
| `vast-n13` | `ae525658c` | 2026-11-05 | no | no |
| `walkthrough-slider` | `0ae65bc2b` | 2026-11-04 | yes | no |
| `whichcols` | `a36a23e5e` | 2026-11-04 | yes | no |

Full shas, for recovery without abbreviation:

```
abscheck 763c85fd3bcb0e0d95f57d1e06b9fb020e1ba6c3
bg/arms 11a77a5ef4aa0cd1b1af3f6ca045ffe9d48e639a
bg/core 1caf360e0ec0ba572eac69c1433a041712548ac0
bg/harness fe441a2ca79ec84c1bc198b84479e4ec5af46398
bio ff5847e0961bb38ad8d5a333b79999b1bf80c957
bio2 81069ba17d7f6cc10f84942337f49b2bf97311bd
clean-neurons 02879b9837479f683a8937b853041fc4279bc3f9
consolidation/docs 4982c663d92bea0188f6d0f6a0ccebad9b805a53
consolidation/lib 51894f10ef1583a5e2b09563b26a940b16f569ac
depthanchor a0658e15990d8f2c2db6deab3a73f3ad8ab659ac
depthoff 93a22b1861d0939e251f805b9df0b245dba4bc8c
depthopen f504aa60b9b0a9fd343cd99d538ab6bfcf7d3ed3
docs/consolidation 97e06357bd24ef8b9c7558bb4544306346d3a618
docs/walkthrough-2026-11-02 f3f19b17f870feb0ed2910140c3e098d29970719
docs/walkthrough-groups 7d62e7b7cc8048992ce081aa4b3abf91fe357bda
drafts/already-clean 41f916b991f40ddfbac9a15ac4f78ff55f8fbcba
fix/instruments-results-errata f613b9bc620bd8e58cba71a7acb2f24a4cbb1b92
gh-pages 2dca1e0372be1766ed52ac343a2d870c967e5d0c
loopanswer a36a23e5e4300875c97342624f4845b443fb4454
m1 f7fbf955b98b3f8031632b1766a89e6802ea9241
m1-read f018f152f2a9a041f21c9963e348bd629a7076ab
m1x 698a14f8b2691c0d9ff738448df728bab1139733
m2 07e93fd2d5cf3f742909562fb4d77d031c5b3f4e
m2-read 77ed57fdf66d505a5a7a741ea4ea1fb38aab91ac
main cd031af326227646102a91680981879c7ea74d3e
n12-read 5ac085cfd432eab64542ad833cbc68f00e4e796f
n13 585640e902351e0b53a0d750bd400a42621f96bb
n13fig 864089b9a6f44cef476b1e6769849be9e5963a8e
n13read 6928784b6036e1b55c554bd139ef5402370fd165
n1read 0d98928c9a20ad6f23780f4d7dbbf2a6c860dfce
n22 637657988d34fcd329c591242501668f20b544d8
n2vread 1a0a3266af95a3a4020fd896c26045083ca1f21e
n6 ae6175c464391471b0faa9709739fc8ccb9fd1d7
neurons/explore 8beb3986652266205ee4eabd9aee20d83b7b0206
neurons/show da91edeb9b38aa683d6d95a3c0ae95fb136235ef
neuronslider a36a23e5e4300875c97342624f4845b443fb4454
postcheck a2bd47041dc9a7e71aeeab05e36166e4659bacc9
research/bytegraft 3ec08340be1b0bd9b3faa2a133a9de4c55b20c26
research/inst-sanity fd23c8513e096a963500cea2b6b8c94632d3f127
research/mechanics a777f3335627ffc1b1f375683d11e4c81bb64d4c
research/mechanics-sanity a79d2fc93b8411117c2757c6c38e53dab7dfe902
research/neuronlineage a36a23e5e4300875c97342624f4845b443fb4454
research/reuse 8da101a082f315364f1519f97d48305c7d46a8d4
research/sharedbirth d69f2b6c217deb7300014dd28f3c7b0281198ec1
science/public-check 835dc25be0fe9ea0b8c1679b3b5fd21fe4d09ed0
science/realizednull c7258ac2aacb7070bf1dddafe2f4643e19b06c6a
vast-n13 ae525658c49dc4ec016774393b2c0839b40ade88
walkthrough-slider 0ae65bc2b88d7e552ca0f42eae35c2f5ea1d99f5
whichcols a36a23e5e4300875c97342624f4845b443fb4454
```

Worktrees before removal (`git worktree list`):

```
/Users/abaybektursun/projects/interp_research                     cd031af32 [main]
/Users/abaybektursun/projects/interp_research-abscheck            763c85fd3 [abscheck]
/Users/abaybektursun/projects/interp_research-bg-arms             11a77a5ef [bg/arms]
/Users/abaybektursun/projects/interp_research-bg-core             1caf360e0 [bg/core]
/Users/abaybektursun/projects/interp_research-bg-harness          fe441a2ca [bg/harness]
/Users/abaybektursun/projects/interp_research-bio                 ff5847e09 [bio]
/Users/abaybektursun/projects/interp_research-bio2                81069ba17 [bio2]
/Users/abaybektursun/projects/interp_research-bytegraft           3ec08340b [research/bytegraft]
/Users/abaybektursun/projects/interp_research-clean-neurons       02879b983 [clean-neurons]
/Users/abaybektursun/projects/interp_research-ghpages             2dca1e037 [gh-pages]
/Users/abaybektursun/projects/interp_research-m1                  f7fbf955b [m1]
/Users/abaybektursun/projects/interp_research-m1read              f018f152f [m1-read]
/Users/abaybektursun/projects/interp_research-m1x                 698a14f8b [m1x]
/Users/abaybektursun/projects/interp_research-m2                  07e93fd2d [m2]
/Users/abaybektursun/projects/interp_research-m2read              77ed57fdf [m2-read]
/Users/abaybektursun/projects/interp_research-n12read             5ac085cfd [n12-read]
/Users/abaybektursun/projects/interp_research-n13                 585640e90 [n13]
/Users/abaybektursun/projects/interp_research-n13-launch          e69f2b2b7 (detached HEAD)
/Users/abaybektursun/projects/interp_research-n13fig              864089b9a [n13fig]
/Users/abaybektursun/projects/interp_research-n13read             6928784b6 [n13read]
/Users/abaybektursun/projects/interp_research-n1read              0d98928c9 [n1read]
/Users/abaybektursun/projects/interp_research-n22                 637657988 [n22]
/Users/abaybektursun/projects/interp_research-n2vread             1a0a3266a [n2vread]
/Users/abaybektursun/projects/interp_research-n6                  ae6175c46 [n6]
/Users/abaybektursun/projects/interp_research-vast-n13            ae525658c [vast-n13]
/Users/abaybektursun/projects/interp_research-walkthrough-groups  7d62e7b7c [docs/walkthrough-groups]
```

### Step 2. The six untracked files in the m1 worktree

The m1 worktree held six untracked files, 62,840,636 bytes in total (60 MB):

| File in `interp_research-m1` | Bytes | sha256 |
|---|---|---|
| `studies/next/m1/banks/k0-01/refs/t00/m1/counts.npz` | 9,457,521 | `076c074c7987825e402b099b9aef37c17799e77e91e475dcf45c33da3b8914ed` |
| `studies/next/m1/banks/k0-01/refs/t12/m1/counts.npz` | 10,718,869 | `5ef992a8aaa9574d9d924bd6e4abf68c8c29c0a2a1ef1b61d8b355d63ee38791` |
| `studies/next/m1/banks/k0-01/refs/t13/m1/counts.npz` | 10,687,516 | `934c10e5803386bb4e91370d57c93875b91d3271a45ff8bc7139bd8b87c7a4a1` |
| `studies/next/m1/banks/k0-01/refs/t14b/m1/counts.npz` | 10,664,009 | `3eb8a29937c079bfb4580a1c544158e66c6668398f68f67365ee0b8c6bbf6189` |
| `studies/next/m1/banks/k0-01/refs/t14c/m1/counts.npz` | 10,663,386 | `e91c2469f4acdaf884c00e4f20a07fbef277b0b53d9bf00631be056283d4a2f7` |
| `studies/next/m1/banks/k0-01/refs/t15/m1/counts.npz` | 10,649,335 | `935c7b710db3de735c3370c354117c500503a10b0c76e50405ba77ee5f5e435a` |

Citation check on `clean-neurons`. No committed document under `studies/next/m1/` cites these six paths.
`k0-01` appears only as `banks/k0-01/k0/m1/metrics.json` (`JOURNAL.md`, `READ_m1.md`) and as an attempt name
(`figures/fig_m1.py`). Every citation of a reference `counts.npz` names `banks/refs/{t00,...,t15}/m1/counts.npz`
(`EXPLORATORY_turnover.md`, `turnover/rule.py`, `turnover/turnover.json`, `turnover/rule.json`, `read/common.py`).

The six files are byte-identical to files already committed in LFS on `clean-neurons`: each sha256 above equals the
sha256 of `studies/next/m1/banks/refs/<same t>/m1/counts.npz`. So no content is at risk.

Decision: not cited, so the "not cited" branch was taken. Nothing was committed. The six files were copied (not
moved) to the scratch folder, sha256 checked equal after the copy:

`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/m1_rescue/studies/next/m1/banks/k0-01/refs/{t00,t12,t13,t14b,t14c,t15}/m1/counts.npz`

The originals are still in `interp_research-m1`. The scratch folder is temporary and may be cleared. The head decides.

### Step 3. Merge of `vast-n13`

`git merge --no-commit --no-ff vast-n13` reported no conflicts (31 files, all under `infra/vast/`). The merge was
committed as `01ddd5ffe`. It brings the six commits `c8578ff7e` to `ae525658c`: the Vast-lane infrastructure for N13
and its spend records. The lane was stopped on the head's order on 2026-11-06; N13 stayed on AWS.

### Step 4. Worktrees removed

Before each removal: `git -C <worktree> status --porcelain` was empty, and the branch was in
`git branch --merged clean-neurons` (after the Step 3 merge, for `vast-n13`). `interp_research-n13-launch` was a
detached HEAD at `e69f2b2b7` with `git rev-list --count clean-neurons..HEAD` = 0. Each was removed with
`git worktree remove <path>` (no `--force`), then `git worktree prune` ran.

Ignored files in these worktrees went with them. They were checked first: only `__pycache__/` folders, `.pgvenv`
symbolic links (their targets, `interp_research/.pgvenv` and `interp_research-clean-neurons/.pgvenv`, are still
there), and `interp_research-vast-n13/infra/vast/.preflight_pass` (10 bytes, the text `bbed876fe`, a marker file).

Removed (18): `interp_research-abscheck`, `-bio`, `-bio2`, `-m1read`, `-m1x`, `-m2`, `-m2read`, `-n12read`, `-n13`,
`-n13-launch`, `-n13fig`, `-n13read`, `-n1read`, `-n22`, `-n2vread`, `-n6`, `-walkthrough-groups`, `-vast-n13`.

Not removed (1): `interp_research-m1`. Its `git status --porcelain` is not empty: it still holds the six untracked
`counts.npz` files of Step 2. Step 2 copied them, it did not move them, and files are never deleted from disk here.
`git worktree remove` without `--force` refuses a worktree with untracked files. To remove it, the head can delete
the six files (byte-identical to the committed `studies/next/m1/banks/refs/*/m1/counts.npz`) and then run
`git worktree remove /Users/abaybektursun/projects/interp_research-m1` and `git branch -d m1`.

A worktree not on the head's list appeared during this work: `interp_research-index` on a new branch `index` at
`02879b983` (created 2026-11-08, after the Step 1 record). It was not touched.

### Step 5. Branch refs deleted

Deleted with `git branch -d` (39). The sha each pointed to is in the Step 1 table.

`abscheck` `bio` `bio2` `m1-read` `m1x` `m2` `m2-read` `n12-read` `n13` `n13fig` `n13read` `n1read` `n22` `n2vread`
`n6` `vast-n13` `consolidation/docs` `consolidation/lib` `depthoff` `depthopen` `docs/consolidation`
`docs/walkthrough-2026-11-02` `docs/walkthrough-groups` `drafts/already-clean` `fix/instruments-results-errata`
`loopanswer` `neurons/explore` `neurons/show` `neuronslider` `research/inst-sanity` `research/mechanics`
`research/mechanics-sanity` `research/neuronlineage` `research/reuse` `research/sharedbirth` `science/public-check`
`science/realizednull` `walkthrough-slider` `whichcols`

Refused (1): `m1`. Git said: `cannot delete branch 'm1' used by worktree at
'/Users/abaybektursun/projects/interp_research-m1'`. It is merged into `clean-neurons`; it stays until the m1
worktree goes.

Kept by rule: `main`, `clean-neurons`, `bg/arms`, `bg/core`, `bg/harness`, `research/bytegraft`, `gh-pages`,
`depthanchor`, `postcheck`, and the new `index`. All tags kept. No remote branch was touched.

### Step 6. Untracked result files of the programs before already-clean

`git status --porcelain --untracked-files=all` showed 9,110 untracked files: 9,109 in five result directories and
`studies/instruments/explainer.html` (12,460 bytes, left untracked). Nothing was deleted. The sizes below count the
untracked files only; each directory also holds tracked files, which stay tracked.

How "on S3" was tested. Recursive listings of the matching S3 prefixes (listing only), then, for every untracked
file, a check that the S3 key `<prefix>/<same relative path>` exists with the same byte size. The S3 layout is
`results/<study>/<stage>/<attempt>/out/<path>` for AWS runs and `results/mdl/<rental>/cells/<cell>/<path>` for the
Vast runs. Contents were not hashed; equal path and equal size is the evidence. Local stage `e6/b/1` maps to S3
`e6b/1`; local `reproductions/instruments_e2/results/<attempt>` maps to S3 `results/instruments/e2/<attempt>`.

S3 prefixes in `s3://pg-lineage-769063704696/`, object counts and sizes on 2026-11-08:

| S3 prefix | Objects | Size |
|---|---|---|
| `results/reuse/` | 22,342 | 853.1 MB |
| `results/mdl/mendeleev_reuse_birth/` (the birth run on Vast) | 5,508 | 3,032.4 MB |
| `results/instruments/` (of which `e2/`: 85 objects, 209.6 MB) | 1,790 | 3,153.0 MB |
| `results/realizednull_pg1_full_a1/` | 445 | 1,527.8 MB |
| `results/mdl/mendeleev_realizednull_q14/` (session 2 on Vast) | 1,001 | 10,080.2 MB |
| `results/mechanics/` | 3,506 | 440.9 MB |

Triage by second-level subdirectory. "Cited by" lists the committed markdown under the study folder that names the
path `results/<subdir>` (paths relative to the study folder; `archive/` documents included).

| Subdirectory (untracked files) | Files | MB | Cited by | On S3, same path and size |
|---|---|---|---|---|
| `studies/reuse/results/ab_primary/1/` | 1,509 | 102.4 | JOURNAL.md; archive/README_before_2026-11-04.md; results/reads/review4/REVIEW_2026-11-01_fourth.md | 1,509 of 1,509 |
| `studies/reuse/results/ab_sniper/1/` | 63 | 12.6 | JOURNAL.md; archive/README_before_2026-11-04.md; results/reads/review4/REVIEW_2026-11-01_fourth.md | 63 of 63 |
| `studies/reuse/results/amend2/1/` | 1,372 | 91.7 | JOURNAL.md; archive/README_before_2026-11-04.md; results/amend2/reads/REVIEW_amend2.md; results/reads/review4/REVIEW_2026-11-01_fourth.md | 1,371 of 1,372 (not: `watch.log`, 907 B) |
| `studies/reuse/results/amend2/judge/` | 1 | 0.0 | JOURNAL.md; archive/POST.md; archive/POST_full.md; results/RESULTS.md | 0 of 1 (`collect.log`, 382 B) |
| `studies/reuse/results/amend2/judge_v2/` | 1 | 0.0 | JOURNAL.md; archive/POST.md; archive/POST_full.md; results/RESULTS.md | 0 of 1 (`collect.log`, 1,065 B) |
| `studies/reuse/results/birth/1/` | 2,709 | 1,505.7 | 16 documents: JOURNAL.md; SPEC11_sharedbirth.md; archive/README_before_2026-11-04.md; GRID5.md under archive/results/dating/ (failing inputs); others | 2,709 of 2,709 (in `results/mdl/mendeleev_reuse_birth/cells/birth/`) |
| `studies/reuse/results/generic/1/` | 300 | 405.9 | JOURNAL.md; SPEC10_sharedinputs.md; SPEC9_sameweights.md; archive/README_before_2026-11-04.md; archive/results/review9/BUILD_NOTES_sameweights.md; results/generic/reads/READ_generic.md | 300 of 300 |
| `studies/reuse/results/phase0/coherence/` | 1 | 0.0 | JOURNAL.md | 0 of 1 (`collect.log`, 1,066 B) |
| `studies/reuse/results/phase0/relatedness/` | 1 | 0.0 | JOURNAL.md | 0 of 1 (`collect.log`, 925 B) |
| `studies/reuse/results/primary/1/` | 492 | 48.9 | JOURNAL.md; archive/README_before_2026-11-04.md; results/RESULTS.md; results/primary/reads/REVIEW_2026-11-01.md; results/reads/review4/REVIEW_2026-11-01_fourth.md | 492 of 492 |
| `studies/reuse/results/sniper/1/` | 39 | 7.0 | JOURNAL.md; SPEC.md; results/sniper/S3.md; results/sniper/SNIPER.md | 39 of 39 |
| `studies/reuse/results/sniper/2/` | 255 | 25.0 | JOURNAL.md; SPEC.md; SPEC6_generic.md; results/RESULTS.md; results/primary/reads/REVIEW_2026-11-01.md; results/sniper/S3.md | 253 of 255 (not: `launch.json` 539 B, `run.json` 424 B) |
| `studies/instruments/results/e1/1/` | 208 | 20.3 | SPEC_instruments.md; results/e1/reads/READ_e1.md | 208 of 208 |
| `studies/instruments/results/e1b/1/` | 174 | 30.6 | none at this path (6 documents name `results/e1b/`) | 174 of 174 |
| `studies/instruments/results/e3/1/` | 353 | 198.1 | JOURNAL.md; results/e3/reads/READ_e3.md; results/e4/reads/READ_e4.md | 353 of 353 |
| `studies/instruments/results/e3/failing_input_{ac,ceiling,known,samples,sanity}/` | 10 | 0.0 | none at these paths | 0 of 10 (each folder: `GRID_e3.md`, `verdict_e3.json`; local verdict outputs, 77,665 bytes in all) |
| `studies/instruments/results/e4/1/` | 278 | 5.5 | archive/BUILD_NOTES_e4.md; results/e4/reads/READ_e4.md | 278 of 278 |
| `studies/instruments/results/e5/1/` | 153 | 26.6 | none at this path (5 documents name `results/e5/`) | 153 of 153 |
| `studies/instruments/results/e6/b/` | 122 | 559.7 | SPEC_instruments.md; results/e6/reads/READ_e6b.md | 122 of 122 (S3 `e6b/1/`) |
| `studies/instruments/results/e6c/1/` | 154 | 15.1 | archive/BUILD_NOTES_e6.md | 154 of 154 |
| `studies/realizednull/results/pg1_full_a1/samples/` | 205 | 117.4 | none at this path (7 documents name `results/pg1_full_a1/`) | 205 of 205 |
| `studies/realizednull/results/pg1_full_a1/sealed/` | 202 | 115.9 | HANDOFF.md | 202 of 202 |
| `studies/realizednull/results/q14_full_a1/samples/` | 201 | 116.3 | HANDOFF.md | 201 of 201 (`mendeleev_realizednull_q14/cells/q14_full/`) |
| `studies/realizednull/results/q14_full_a1/sealed/` | 200 | 115.3 | HANDOFF.md; JOURNAL.md | 200 of 200 (same) |
| `studies/realizednull/results/q14_smoke_a1/samples/` | 30 | 17.4 | HANDOFF.md (and CLAIMS.md, JOURNAL.md, README.md, SPEC.md name `q14_smoke_a1`) | 0 of 30. S3 `cells/q14_smoke/` holds the second smoke attempt; `attempts/1/` holds only logs |
| `studies/realizednull/results/q14_smoke_a2/samples/` | 38 | 22.0 | HANDOFF.md | 38 of 38 (`cells/q14_smoke/`) |
| `studies/reproductions/instruments_e2/results/1/` | 16 | 72.3 | none at this path | 16 of 16 (S3 `results/instruments/e2/1/`) |
| `studies/reproductions/instruments_e2/results/1_m2/` | 9 | 62.3 | none at this path | 9 of 9 |
| `studies/reproductions/instruments_e2/results/1_m3/` | 9 | 61.6 | none at this path | 9 of 9 |
| `studies/mechanics/results/screen/1/` | 4 | 105.6 | 10 documents: JOURNAL.md; README.md; SPEC_mechanics.md; results/RESULTS.md; results/screen/1/reads/READ_mechanics.md; others | 4 of 4 |

Summary and recommendation per directory:

| Directory | Untracked files | Untracked size | On S3 | Not on S3 | Recommendation |
|---|---|---|---|---|---|
| `studies/reuse/results/` | 6,743 | 2,199.2 MB | 6,736 | 7 small logs (5.3 KB) | Safe to delete the 6,736 files S3 holds. Keep the 7 listed above, or commit them. |
| `studies/instruments/results/` | 1,452 | 855.8 MB | 1,442 | 10 failing-input outputs of E3 | Safe to delete the 1,442 files S3 holds. Keep `e3/failing_input_*/`. |
| `studies/realizednull/results/` | 876 | 504.3 MB | 846 | 30 (`q14_smoke_a1/samples/`, 17.4 MB) | Safe to delete the 846 files S3 holds. Keep `q14_smoke_a1/samples/`: it is the only copy. |
| `studies/reproductions/instruments_e2/` | 34 | 196.2 MB | 34 | 0 | Safe to delete; S3 holds all 34. |
| `studies/mechanics/results/` | 4 | 105.6 MB | 4 | 0 | Safe to delete; S3 holds all 4. |

Citations do not change the recommendation: a cited file that S3 holds can be pulled back by its path.

`.gitignore` now ignores the five directories (comment line names this section). This hides only untracked files.
The tracked files in them (18,734 in `studies/reuse/results/`, 341 in `studies/instruments/results/`, 207 in
`studies/realizednull/results/`, 81 in `studies/reproductions/instruments_e2/`, 3,539 in `studies/mechanics/results/`)
stay tracked. A new file in these directories now needs `git add -f`.

After the `.gitignore` change, `git status --porcelain --untracked-files=all | grep -c '^??'` prints 1:
`studies/instruments/explainer.html`, left untracked as asked.

## 2026-11-08 final cleanup

Done by an Opus worker on the head's list of actions. `git fetch origin --prune` ran first; it printed nothing.
The only remote is `origin` (`git@github.com:abaybektursun/borrowed-geometry.git`).

### Step 1. Record before any action

`git worktree list`:

```
/Users/abaybektursun/projects/interp_research                cd031af32 [main]
/Users/abaybektursun/projects/interp_research-bg-arms        11a77a5ef [bg/arms]
/Users/abaybektursun/projects/interp_research-bg-core        1caf360e0 [bg/core]
/Users/abaybektursun/projects/interp_research-bg-harness     fe441a2ca [bg/harness]
/Users/abaybektursun/projects/interp_research-bytegraft      bd24152a1 [research/bytegraft]
/Users/abaybektursun/projects/interp_research-clean-neurons  2aca2fd75 [clean-neurons]
/Users/abaybektursun/projects/interp_research-ghpages        2dca1e037 [gh-pages]
```

Every local branch. "Merged" is `git merge-base --is-ancestor <name> clean-neurons` at clean-neurons
`2aca2fd75`. "origin" is `refs/remotes/origin/<name>` from `git for-each-ref refs/heads refs/remotes/origin`
after the fetch.

| Branch | Local sha | Merged into clean-neurons | `origin/<name>` |
|---|---|---|---|
| `bg/arms` | `11a77a5ef` | no | none |
| `bg/core` | `1caf360e0` | no | none |
| `bg/harness` | `fe441a2ca` | no | none |
| `clean-neurons` | `2aca2fd75` | yes | `2aca2fd75` |
| `depthanchor` | `a0658e159` | no | none |
| `gh-pages` | `2dca1e037` | no | `2dca1e037` |
| `main` | `cd031af32` | yes | `cd031af32` |
| `postcheck` | `a2bd47041` | no | none |
| `research/bytegraft` | `bd24152a1` | no | `bd24152a1` |

`origin/HEAD` points at `cd031af32` (`origin/main`).

### Step 2. The gh-pages worktree: removed

`git -C /Users/abaybektursun/projects/interp_research-ghpages status --porcelain` printed nothing (also nothing
with `--ignored`). `gh-pages` and `origin/gh-pages` were both `2dca1e037`. `git worktree remove` exited 0; the
directory is gone. The local branch `gh-pages` stays at `2dca1e037` as the deploy branch of the walkthrough
site. The repo has one remote, `origin`; no second remote exists in this clone.

### Step 3. `depthanchor` and `postcheck`: pushed to origin, kept as local branches

Neither branch existed on origin (`git ls-remote origin` returned nothing for both). Both were pushed with
`git push origin <name>`. After a second fetch, `origin/<name>` equals the local sha for both. They are not
merged into clean-neurons, so `git branch -d` would refuse; they stay as local branches for the head to decide.
`depthanchor` is not an ancestor of `postcheck`; each is 2 commits past its merge-base with clean-neurons.

| Branch | Sha | On origin before | On origin now | `git diff --stat clean-neurons...<name>` |
|---|---|---|---|---|
| `depthanchor` | `a0658e159` | no | yes, `a0658e159` | 4 files changed, 82 insertions(+), 4 deletions(-) |
| `postcheck` | `a2bd47041` | no | yes, `a2bd47041` | 3 files changed, 67 insertions(+), 9 deletions(-) |

`git log --oneline -3 depthanchor`:

```
a0658e159 depthanchor: correct the file count — 379 data files, not the hand-typed 378
f65e35431 depthanchor: correct the initial-state probe — the episodes DO start from the shipped init states
9255751ce repo index: DEPTHANCHOR closed as H_epi
```

`git log --oneline -3 postcheck`:

```
a2bd47041 postcheck: P4 re-measured over all 379 files — 373 off by one, nothing absent
02c224332 postcheck: retag P4 — the scan used a hard-coded 378-file count and never opened file-378
a9ecaa89c postcheck: repo index entry — the follow-up study, closed
```

What each holds: `depthanchor` holds two corrections to the closed DEPTHANCHOR study (initial-state probe,
379-file count). `postcheck` holds the closed follow-up study's repo-index entry and its P4 retag and re-measure
over all 379 files.

### Step 4. Fast-forward of `main`: refused by git, nothing forced

`git merge-base --is-ancestor main clean-neurons` returned true (clean-neurons is 642 commits ahead of `main`).
`git -C /Users/abaybektursun/projects/interp_research merge --ff-only clean-neurons` refused:

```
Updating cd031af32..2aca2fd75
error: The following untracked working tree files would be overwritten by merge:
	studies/depthanchor/SPEC.md
	studies/depthopen/SPEC.md
	studies/postcheck/SPEC.md
Please move or remove them before you merge.
Aborting
```

The three colliding paths are untracked files in the main worktree that clean-neurons tracks. Nothing was
moved, removed or forced. `main` and `origin/main` are both still `cd031af32`; nothing was pushed. The main
worktree has 0 tracked changes and 78 untracked entries (`git status --porcelain | grep -c '^??'`), the same
before and after the attempt.

### Step 5. `studies/instruments/explainer.html`: committed

12,460 bytes, dated 2026-11-04. It is a standalone HTML page titled "INSTRUMENTS, explained with the math"
that loads KaTeX 0.16.11 from cdnjs. `grep -c -E 'AKIA|PRIVATE KEY|hf_[A-Za-z0-9]{20}|ghp_'` printed 0. The
earlier section left it untracked; on the head's decision for this cleanup it was committed by name as
`961c8f44c` ("instruments: explainer.html (untracked file committed in the final cleanup)").

### Step 6. Python caches in the clean-neurons worktree: 9 directories removed

`find /Users/abaybektursun/projects/interp_research-clean-neurons -type d -name __pycache__ -prune` listed 9
directories, 2,832 KB in all. All 9 were git-ignored, held no tracked file and held only `.pyc` files. Each was
removed with `rm -r`; a second `find` lists 0. Nothing else was deleted.

```
infra/__pycache__
infra/aws_pg/__pycache__
infra/pg/__pycache__
infra/neurons/__pycache__
infra/aws_pg/jobs/__pycache__
infra/pg/tests/__pycache__
studies/already-clean/runbook/__pycache__
studies/mechanics/scripts/next0_control_forward/__pycache__
studies/reproductions/alwayson/code/pg/__pycache__
```

### State after this cleanup

`git worktree list`:

```
/Users/abaybektursun/projects/interp_research                cd031af32 [main]
/Users/abaybektursun/projects/interp_research-bg-arms        11a77a5ef [bg/arms]
/Users/abaybektursun/projects/interp_research-bg-core        1caf360e0 [bg/core]
/Users/abaybektursun/projects/interp_research-bg-harness     fe441a2ca [bg/harness]
/Users/abaybektursun/projects/interp_research-bytegraft      bd24152a1 [research/bytegraft]
/Users/abaybektursun/projects/interp_research-clean-neurons  (this commit) [clean-neurons]
```

The four worktrees and branches of the other agent were not entered or changed. Read-only state
(`git --no-optional-locks -C <worktree> status --porcelain`, no index lock taken):

| Worktree | Branch | Sha | Against origin | Working tree |
|---|---|---|---|---|
| `interp_research-bytegraft` | `research/bytegraft` | `bd24152a1` | ahead 0, behind 0 of `origin/research/bytegraft` | 11 porcelain lines (not clean) |
| `interp_research-bg-arms` | `bg/arms` | `11a77a5ef` | no `origin/bg/arms`; 0 commits outside origin refs | clean |
| `interp_research-bg-core` | `bg/core` | `1caf360e0` | no `origin/bg/core`; 0 commits outside origin refs | clean |
| `interp_research-bg-harness` | `bg/harness` | `fe441a2ca` | no `origin/bg/harness`; 0 commits outside origin refs | clean |

Left for the head: the three untracked `SPEC.md` files that block the fast-forward of `main`; the local
branches `depthanchor` and `postcheck` (now safe on origin).

## 2026-11-08 temporary files removed

Done by an Opus worker on the head's list. No tracked file was deleted. No git command that writes was run.

### Step 1. Session scratchpad: 445 entries deleted, 22,228,840 KiB (22.8 GB)

Path: `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.
`du -sk` before: 22,228,844 KiB; after: 4 KiB. Every one of the 445 top-level entries was removed with `rm -rf`.
The largest, in KiB: `bio2` 4,468,620; `n12` 2,262,624; `n22` 1,512,636; `n12dl` 1,063,896; `m2c` 1,031,092;
`m2` 958,392; `pub` 844,940; `m1_gate` 696,968; `m1_venv` 633,360; `hk` 560,100; `n13read` 402,264; `s5` 401,940;
`n13_venv` 241,928. Also `head/`, `m1_rescue/` (the six copies of Step 2 above, byte-identical to committed LFS
files), `n2vread/` and about 430 smaller files and folders.

Checks before deletion:

- `head/21_REFLECTION.md` and `head/abs6.txt`: `studies/next/21_REFLECTION.md` and `studies/next/20_ABSTRACT.md`
  exist and are tracked. The committed reflection is a later edit of the draft (one line differs). The abstract
  contains the text of `abs6.txt`.
- `pub/already-clean-audit`: `git status --porcelain` empty; local HEAD `2e43474ae1bbfa353e222993cb706283a47bee7b`
  equals `gh api repos/abaybektursun/already-clean-audit/commits/main`.

Not touched: `pycache_list.txt` (840 bytes, 13:36), which another worker wrote after the deletion. An orphaned
`tail -F` process (PID 64754, started 2026-11-07 07:31, parent launchd) was following
`scratchpad/bio2/n1l2/poll/poll.log`; it was not killed.

### Step 2. Python environments: none deleted

No entry of `/Users/abaybektursun/projects/` has `venv` in its name. The only `.pgvenv` in a remaining worktree is
the symbolic link `interp_research-clean-neurons/.pgvenv -> /Users/abaybektursun/projects/interp_research/.pgvenv`
(the main worktree, not touched). `interp_research-ghpages` no longer exists.

### Step 3. Local copies of old-program results: 9,062 files deleted, 3,843,683,051 bytes (3.84 GB)

S3 calls: six `list-objects-v2` listings of bucket `pg-lineage-769063704696`, none failed. The object counts equal
those of the Step 6 table above (22,342; 5,508; 1,790; 445; 1,001; 3,506). Local paths were mapped to keys by the
Step 6 rules. `q14_smoke_a1` and `e3/failing_input_*` were not mapped.

A file counted as verified only if the S3 object at its key has the same size and its ETag equals the local MD5
(single-part ETag) or equals the multipart ETag recomputed locally with 8 MiB parts. All 64 multipart objects
matched that way, so no file was deleted on size alone. Negative control: the 38 `q14_smoke_a1` sample files
against the same-named objects of the second smoke attempt (`cells/q14_smoke/`): 0 of 38 equal in size, 0 equal
in MD5. Three MD5 values were checked against `md5 -q`: equal.

Tracked files were excluded with a fresh `git ls-files` of the five directories (22,902 paths). None of the deleted
files was tracked, and all 22,902 tracked paths still exist. 136 directories became empty and were removed; none
was empty before. The 12 tracked symbolic links in `studies/reproductions/instruments_e2/results/bank/` still
resolve. `du -sk` of the five directories: 4,556,244 KiB before, 784,764 KiB after.

| Directory | Ignored files before | Deleted (MD5 / multipart ETag) | Bytes deleted | Kept |
|---|---|---|---|---|
| `studies/reuse/results/` | 6,743 | 6,736 (6,723 / 13) | 2,199,173,100 | 7 |
| `studies/instruments/results/` | 1,452 | 1,442 (1,402 / 40) | 855,749,572 | 10 |
| `studies/realizednull/results/` | 876 | 846 (846 / 0) | 486,895,509 | 30 |
| `studies/reproductions/instruments_e2/` | 34 | 34 (25 / 9) | 196,218,589 | 0 |
| `studies/mechanics/results/` | 4 | 4 (2 / 2) | 105,646,281 | 0 |
| total | 9,109 | 9,062 | 3,843,683,051 | 47 |

Kept (47 files, still git-ignored on disk):

- `studies/reuse/results/` (7, 5,308 bytes): no S3 object at the mapped key. `amend2/1/watch.log`,
  `amend2/judge/collect.log`, `amend2/judge_v2/collect.log`, `phase0/coherence/collect.log`,
  `phase0/relatedness/collect.log`, `sniper/2/launch.json`, `sniper/2/run.json`.
- `studies/instruments/results/` (10, 77,665 bytes): local verdict outputs with no S3 copy.
  `e3/failing_input_{ac,ceiling,known,samples,sanity}/{GRID_e3.md,verdict_e3.json}`.
- `studies/realizednull/results/` (30, 17,408,285 bytes): `q14_smoke_a1/samples/smoke/*.json`, the only copies. S3
  `cells/q14_smoke/` holds the second attempt.
