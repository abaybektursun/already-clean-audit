# The judge's calls

A picture for the walkthrough (spec P7, "the judge's calls", in `../VISUALS_PLAN.md`). Every call the judge
made in the parent test on OLMo-3 7B is one dot: 4,952 calls on the 1,238 late-born groups and 2,400 calls on
the 600 control groups, four calls per group. The dots stand in two columns, one per kind of group, with the
calls the judge got right at the bottom and the wrong ones above them, so the share right reads as the height of
the lower block; the two shares, 81.2% and 92.4%, are printed above. When the picture first comes into view the
dots fall into place and the two shares count up as the right calls land. A button replaces every outcome with a
fair coin flip, drawn in the browser, so the reader sees what 50% right looks like next to 81 and 92, and turns
back. One line under the columns derives the share of unrelated late-born groups from the two shares, 26.5% with
a 95% range of 21 to 32%. A menu opens one call: the neuron, the group's contexts the judge saw, the two records in
the order the judge saw them, which record was the neuron's own, the judge's pick and whether it was right. The
first call in the menu is from the lithium neuron L03N6372+.

It has two hosts: the walkthrough page (`../index.html`, once the section in `INSERT.md` is pasted in) and
`index.html` here. Both load the same three files.

The spec's sentence counted 1,838 calls. The records hold four calls per group (two random neurons, each shown
to the judge twice), so the picture shows 7,352 calls on 1,838 groups. The study's A and a_ctrl are means over
groups of the share of each group's four calls that were right; with four calls in every group they equal the
pooled shares 4,019 / 4,952 and 2,218 / 2,400.

## Files

- `judge.css`: the styles. Every rule is scoped under `#judgecalls` and every class is prefixed `jd-`, so the
  host page's own classes (`.record`, `.row`, `.num`, ...) neither leak in nor out. It reads the host's `--bg`,
  `--ink`, `--muted`, `--rule`, `--accent`, `--font-text` and `--font-mono`, light and dark. The two outcome
  colors are the pair `../one_job/` uses (light `#036f4f` and `#c9690c`, dark `#48a581` and `#cb7229`); the
  dataviz validator passes them against the host surfaces, `#f4f6f4` light (CVD ΔE 9.4, normal-vision ΔE 25.0,
  both above 3:1) and `#101514` dark (CVD ΔE 10.1, normal-vision ΔE 20.1). Position carries the outcome as well:
  right calls are always the lower block of a column.
- `judge.js`: builds the picture inside `<div id="judgecalls">` and does nothing if that element is absent. No
  framework, no library; the dots are drawn on one canvas. It checks on load that its per-column counts equal
  the ones `extract.py` wrote, and throws if not. The fall runs once, when the columns are first 35% in view;
  under prefers-reduced-motion, or after any press inside the picture, the final state shows at once. Pointing
  at a dot shows its neuron, group, birth point, the other neuron and the outcome; pressing a dot whose call is
  in the menu opens it. Pressing a context in the opened call shows its whole text.
- `data.js`: sets `window.JUDGE_DATA`. Written by `extract.py`. About 0.76 MB, so it is stored as a plain file,
  not through Git LFS.
- `extract.py`: reads the saved files, writes `data.js`, prints its checks and rewrites the checks block below.
  Run `python3 extract.py` from this folder (it needs numpy). No model, no download, no network.
- `index.html`: the standalone page, with the walkthrough's tokens, fonts (Atkinson Hyperlegible, IBM Plex
  Mono) and reading column (47.5rem). The data is loaded with a script tag, because Chrome refuses `fetch()` of
  local JSON from a page opened from disk.
- `INSERT.md`: the section to paste into `../index.html`, where it goes, and the prose it replaces.

## Inputs

Local copies under `/Users/abaybektursun/projects/interp_research_banks/neurons/`, each from
`s3://pg-lineage-769063704696/results/neurons/` at the same path:

- `w4/o7_s_v2/explain/judgments.jsonl` (2,537,271 bytes, 11,028 lines): one line per call, `id`, `cluster`
  (`<neuron>:c<k>`), `control`, `decoy_type` (`random` or `hard`), `decoy_pole`, `draw`, `t_before`, `sample`,
  `shown_order`, `answer`, `correct` (`DATA_CONTRACT.md` §3 and §6). Its sha256 must equal
  `q2.inputs.judgments.sha256` in the verdict. The picture uses the 7,352 calls against a random neuron, the
  study's primary test (SPEC_D1 §2.4, §6.3); the 3,676 hard-decoy calls only feed one check.
- `w4/o7_s_v2/explain/manifest.json`: the judge model (`anthropic/claude-sonnet-5-5`, shown as Claude Sonnet
  5.5) and the judge prompt's sha256 (`11b5fd81…`, the prompt `infra/pg/neurons_prompts/judge.md`).
- `w5/verdict/samples/q2_clusters.jsonl` (196,269,741 bytes, 2,465 lines): one line per group, written by
  `unblind` in `infra/pg/neurons_verdict.py`: `cluster`, `pole`, `name`, `echo`, `ranks`, `present_pct`,
  `birth_pct`, `late_pct`, `late_raw`, `matched`, `is_main`, `status`, `acc_random`, `flagged` and `judgments`,
  the group's calls with `judge_saw`, the user message the judge received. The script splits it into the group's
  contexts and records A and B; each context carries exactly one token marked ⟦ ⟧.
- `w5/verdict/samples/q2_parent.jsonl` (63,747 bytes, 4 lines): the verdict's saved sample calls with their
  text, compared with the per-group record.
- `w5/verdict/metrics.json` (136,032 bytes): `q2.p_u`, `q2.flags`, `q2.f_ctrl`, `q2.status_counts`,
  `q2.births`, `q2.hard`, `q2.echo_clusters`, `q2.n_clusters`, `q2.n_poles` and the 16 saved points in
  `q2.inputs.grid`.

How the call was set up, from `cmd_prepare_judge` in `infra/pg/neurons_explain.py`: the group's contexts are its
ten strongest at the end of training (fewer if the group is smaller), 64 tokens before the marked token; each
record is a neuron's strongest contexts at the saved point before the group's birth (T_14 for a control), up to
20, without any context that lies in the window of one of the group's own contexts; the other neuron is drawn
from the same layer, and its record is cut to the same length. Two such neurons are drawn per group, and each
pair of records is shown twice, the order of A and B drawn per call.

## What `data.js` holds

- `groups`: 1,838 rows, the 1,238 late-born groups in the test (sorted by neuron and group number), then the 600
  control groups (each neuron's strongest group, sorted by neuron). Each row is `[neuron, group name, birth
  point 0–15, control 0/1, contexts in the group, outcomes, other neurons]`, where `outcomes` is four characters,
  `1` for right, in the order (sample, draw) = (0, 0), (0, 1), (1, 0), (1, 1), and `other neurons` lists the
  random neuron of each call in the same order.
- `examples`: 30 calls with the text the judge saw: the 10 in `NAMED` in `extract.py` (the lithium neuron's
  first), each with the label the menu shows and where the write-up names it, then 5 drawn at random from each
  of four pools (late-born right, late-born wrong, control right, control wrong). Each holds the group's
  contexts and records A and B as `[before, marked token, after]`, the shown order (`td` = the neuron's own record
  is A), the answer and the saved point of the records.
- The totals the page prints: A, a_ctrl, the derived share and its interval, the right counts, the overlap shares
  for the caption, the per-group histogram for the table, the 16 steps.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w4/o7_s_v2/explain/judgments.jsonl` and `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/samples/q2_clusters.jsonl`.

- The judge's record has sha256 `800fb2f774254ad5fde9cb0437800595d751e9fdbe700bfdddbe03bf034bd78f`, the value at `q2.inputs.judgments.sha256` in `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json`.
- 11,028 calls: 7,352 against a random other neuron of the same layer (the primary test, shown in the picture) and 3,676 against a hard decoy (not shown). The judge model in the manifest is `anthropic/claude-sonnet-5-5`, prompt sha256 `11b5fd81…`.
- Every call's answer and shown order give back its stored outcome. Every call in the judge's record appears with the same fields in the per-group record, which also holds the text the judge saw.
- Each judged group has exactly four random-decoy calls: two random neurons, each shown twice (`sample` 0 and 1, `draw` 0 and 1). Each context the judge saw carries exactly one marked token; the two records of a call have the same length; a group shows min(10, its size) contexts.

| check | computed | expected |
|---|---:|---:|
| lines in q2_clusters.jsonl (groups, with the echo groups) | 2,465 | 2,465 |
| echo groups (left out of every count) | 11 | 11 (q2.echo_clusters) |
| groups | 2,454 | 2,454 (q2.n_clusters) |
| strongest groups at the end (one per neuron; the control groups) | 600 | 600 (q2.n_poles) |
| groups first in the top 0.1% at the neuron's earliest point | 919 | 919 (q2.status_counts) |
| late-born groups, percentile reading | 1,535 | |
| of these, not late-born under the raw-ratio reading | 244 | 244 (q2.status_counts) |
| of these, no match in the second sorting | 53 | 53 (q2.status_counts) |
| late-born groups in the test | 1,238 | 1,238 (q2.p_u.n_clusters) |
| calls on late-born groups, right | 4,019 of 4,952 | |
| A, mean share right per late-born group | 0.811591 | 0.811591 (q2.p_u.A) |
| calls on control groups, right | 2,218 of 2,400 | 2,218 of 2,400 (E_skeptic.md §1) |
| a_ctrl, mean share right per control group | 0.924167 | 0.924167 (q2.p_u.a_ctrl) |
| p_u = (a_ctrl − A) / (a_ctrl − 0.5) | 0.265404 | 0.265404 (q2.p_u.point) |
| its 95% interval, 10,000 draws over neurons, seed 0 | [0.210894, 0.320826] | [0.210894, 0.320826] (q2.p_u.ci95) |
| late-born groups flagged (at most 1 of 4 right) | 125 | 125 (q2.flags.n_flagged) |
| control groups with at most 1 of 4 right | 19 | 19 (q2.f_ctrl × 600) |
| control calls sharing a marked token with the neuron's own record | 1,200 of 2,400 = 0.500 | 0.500 (E_skeptic.md §3.1) |
| late-born calls sharing a marked token with the neuron's own record | 924 of 4,952 = 0.187 | 0.187 (E_skeptic.md §3.1) |
| the same against the other neuron's record (control, late-born) | 0.108, 0.075 | well below both |
| A from the hard-decoy calls instead (a different input; must differ) | 0.6971 | 0.6971 (q2.hard.accuracy) |
| a fair coin, seed 0: share right, late-born and control | 0.494, 0.494 | 0.5, 0.5 |
| a fair coin: late-born groups with at most 1 of 4 right | 409 | 387 (5/16 of 1,238) |

Groups by the number of their four calls the judge got right:

| right of 4 | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| late-born groups | 70 | 55 | 159 | 170 | 784 |
| control groups | 8 | 11 | 39 | 39 | 503 |

Groups by the training point where they were born (first in the neuron's top 0.1% and still there at the next point):

| point | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | T15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step | 0 | 1,000 | 2,000 | 3,000 | 5,000 | 8,000 | 12,000 | 20,000 | 35,000 | 60,000 | 100,000 | 180,000 | 320,000 | 560,000 | 1,000,000 | 1,413,814 |
| all groups | 0 | 1 | 3 | 14 | 30 | 49 | 62 | 97 | 160 | 176 | 238 | 346 | 467 | 489 | 286 | 36 |
| q2.births, main + non_main | 0 | 1 | 3 | 14 | 30 | 49 | 62 | 97 | 160 | 176 | 238 | 346 | 467 | 489 | 286 | 36 |
| late-born groups in the test | 0 | 0 | 0 | 0 | 2 | 10 | 14 | 27 | 67 | 76 | 114 | 164 | 247 | 271 | 212 | 34 |

Every expected number above matches, and the hard-decoy calls give a different A.

The 4 calls the verdict saved with their text in `q2_parent.jsonl` (judge:w4j:5512, judge:w4j:4922, judge:w4j:8541, judge:w4j:2414) carry the same text in the per-group record: 4 of 4.

The picker holds the 10 named calls in extract.py and 20 drawn by `random.Random(0).sample(pool, 5)` from each of four pools (late-born right, late-born wrong, control right, control wrong; call ids sorted as strings, named calls left out): judge:w4j:8602 (L14N6165-:c3), judge:w4j:3812 (L01N3050-:c1), judge:w4j:7732 (L09N10681+:c2), judge:w4j:9082 (L09N8078+:c5), judge:w4j:4197 (L28N3+:c1), judge:w4j:10269 (L22N7990+:c0), judge:w4j:2725 (L10N4224+:c3), judge:w4j:5403 (L04N6542-:c3), judge:w4j:5196 (L26N5340-:c3), judge:w4j:4167 (L03N9245-:c0), judge:w4j:5600 (L14N3173-:c0), judge:w4j:8756 (L31N4660-:c0), judge:w4j:6580 (L20N8604+:c0), judge:w4j:4130 (L24N5066+:c0), judge:w4j:9290 (L01N369-:c0), judge:w4j:2120 (L06N7759-:c0), judge:w4j:4029 (L18N3476+:c2), judge:w4j:1241 (L02N4436-:c0), judge:w4j:8635 (L18N6315-:c0), judge:w4j:3461 (L05N534+:c0).

Wrote `data.js`: 761,927 bytes, 1,838 groups (1,238 late-born, 600 control) with their 7,352 calls, and 30 calls with their text; sha256 `bf022b16342f2eb66bf3b03b3c0c0b8288acc8902dbd8252aec62b31b7416e58`.

The first call in the picker, judge:w4j:3893 on L03N6372+:c0, judged against the records at step 320,000: the neuron's own record marks `lithium` 19, `rika` 1; the judge answered B, which is wrong. Its first context:

```
…lly, is provided in a format readable by libfsmtest 222Licensed according to MIT license https://opensource.org/licenses[/MIT]. Source code available under https://bi
```
<!-- checks:end -->
