# When a neuron's groups are born

A picture for the walkthrough (spec P6 in `../VISUALS_PLAN.md`). One OLMo-3 7B neuron at a time, the lithium
neuron L03N6372+ first. Each row is one of the neuron's groups: a language model sorted the neuron's 50 strongest
contexts at the end of training into groups. Each row has 16 squares, one per saved training point, on the same
evenly spaced axis as "One neuron across training" (`../neuron_across_training/`): the same rail, ticks, thumb,
step labels (0, 1k, ... 1.41M) and percent of training. A square is filled from the group's birth on wherever the
group is present, drawn as an open square where it is present but not at the next point (which does not count as a
birth), and as a hairline box where it is not present. The rows are sorted by birth point. Since 2026-11-05 each row
also has a tick at its chance level: the median birth point of random groups of the same size drawn from the same
neuron's contexts and dated by the same rule (`studies/next/10_RECOUNT2.md` §7, the reference called R1t there).

The count is the number of the neuron's groups born by the chosen point ("9 of 15 groups born by step 320,000").
The scrubber moves the point; the squares after it stay blank. Under the axis, a strip shows how many of all 2,454
groups of the 600 neurons were born at each point (`q2.births`), with a line on each bar at the chance level (the mean
count of 300 random groupings of the same sizes, `studies/next/06b_VERIFY_BIRTHS.md` §3), and a line under it gives the
share born by the chosen point. A menu picks among 7 neurons named in the write-up and 33 drawn at random; a "Random neuron" button
picks one of the 33. When the chart first comes into view the point runs once from step 0 to the last step and the
rows fill in birth order.

It has two hosts: the walkthrough page (`../index.html`, once the section in `INSERT.md` is pasted in) and
`index.html` here. Both load the same three files.

## Definitions (SPEC_D1.md §2.4 and `infra/pg/neurons_verdict.py` `presence`, `birth`, `late_flags`)

- **Present** at a point: the median, over the group's contexts, of the neuron's activation in its own direction
  (`max(a, 0)` for a + neuron, `max(-a, 0)` for a − neuron), measured at that point on the same text, is above
  zero and at or above the neuron's own 99.9th percentile at that point.
- **Born**: the first point at which the group is present and still present at the next point. At the last point
  (step 1,413,814) presence alone is a birth.
- **Late-born**: born after the neuron's first-born group. The 1,238 late-born groups in the parent test are those
  on which the raw-ratio reading agrees, that the second sorting matched, and that were not born at step 1,000.
- **Chance level of a group's birth** (added 2026-11-05): the group's contexts are replaced by a random set of the same
  size from the same neuron's non-echo top-50 contexts, 2,000 times, and each random set is dated by the rule above;
  for the strongest group the random set keeps the neuron's top context, which is what makes a group the strongest.
  The tick is at the median of those 2,000 birth points. A group is **born later than chance** if at most 5% of its
  random sets are born at the same point or later, and **earlier than chance** if at most 5% are born at the same
  point or earlier (`05_per_cluster_R1t.csv`: `ref_median`, `p_late`, `p_early`). A neuron with one group has no
  chance level, because its group holds all its contexts.
- **Chance level of the strip**: the mean, over 300 random groupings of every neuron's contexts into groups of the
  real sizes, of the number born at each point (`s03_histogram_tv.json` → `random_mean_counts_A`). Random groups
  reproduce most of the strip's shape; the real groups have more births at the last two points (06b §3).

The picture shows presence, birth and the chance level of each birth. It does not show which group is strongest at
the end, nor whether a group is related to an earlier one (the judge picture, `../judge/`, covers that). The study's
birth-order claim, that the strongest group is the oldest, was retagged on 2026-11-05: the strongest group is among
the first born in 0.476 of the 574 neurons with two or more groups, below the 0.530 [0.495, 0.568] that random groups
give (`../../results/RESULTS.md`, Retag; `studies/next/10_RETAG.md`).

## Files

- `births.css`: the styles. Every rule is scoped under `#nbirths` and every class is prefixed `nb-`. It reads the
  host's `--bg`, `--ink`, `--muted`, `--rule`, `--bar`, `--accent`, `--font-text` and `--font-mono`, light and
  dark. Colors: born-and-present squares are `--accent`; present-before-birth squares are an `--accent` ring, so
  the two differ in shape as well as fill; absent squares are a `--rule` hairline. The dataviz validator gives the
  accent against the strip's grey `--bar` a CVD ΔE of 29.7 (light) and 29.8 (dark); its lightness and chroma
  checks fail by design, because `--bar` is a grey de-emphasis, not a categorical slot, and its contrast warning
  (1.74:1 light, 2.44:1 dark) is met by the readout line and the table. The chance level is drawn in `--ink`: on a
  row as a tick above and below the square (the square itself stays visible), on a bar as a short line wider than
  the bar.
- `births.js`: builds the picture inside `<div id="nbirths">` and does nothing if that element is absent. No
  framework, no library. It plays once when the chart is first 35% in view (not under prefers-reduced-motion,
  where the last point shows); a press inside the picture or a key press stops it. Arrow keys, Home and End move
  the point while focus is inside the picture, or anywhere on the page when `#nbirths` has `data-keys="page"`
  (as in `index.html`). Pointing at a square shows the group, the point, its state, its birth step, its size and
  its strongest context at the end, and its chance level; pointing at a strip bar shows that point's births over all
  600 neurons and their chance level. The table under "Show this neuron's groups as a table" holds the same values.
- `data.js`: sets `window.BIRTHS_DATA`. Written by `extract.py`. About 35 KB, a plain file (not LFS).
- `extract.py`: reads the saved files, writes `data.js`, prints its checks and rewrites the checks block below.
  Run `python3 extract.py` from this folder (needs numpy). No model, no download, no network.
- `index.html`: the standalone page, with the walkthrough's tokens, fonts (Atkinson Hyperlegible, IBM Plex Mono)
  and reading column (47.5rem). The data is loaded with a script tag, because Chrome refuses `fetch()` of local
  JSON from a page opened from disk.
- `INSERT.md`: the section to paste into `../index.html`, where it goes, and the prose it replaces.

## Inputs

Local copies under `/Users/abaybektursun/projects/interp_research_banks/neurons/`, each from
`s3://pg-lineage-769063704696/results/neurons/` at the same path:

- `w5/verdict/samples/q2_clusters.jsonl` (196,269,741 bytes, 2,465 lines, one per group, read without its
  `judgments`): `cluster`, `pole`, `name`, `echo`, `ranks` (columns of the end-of-training top 50), `med` [16],
  `present_pct` [16], `present_raw` [16], `birth_pct`, `matched`, `is_main`, `status`. The 11 echo groups are left
  out, as in every count of the study.
- `w5/verdict/metrics.json` (136,032 bytes): `q2.births` (births per point, main and non-main groups),
  `q2.n_clusters`, `q2.n_poles`, `q2.echo_clusters`, `q2.p_u.n_clusters`, `q2.status_counts`, `q2.birth_order`
  and the 16 points in `q2.inputs.grid`.
- `w3/grid/{1,1b}/out/<point>/poles/exemplar_a.npz` (16 files): `a` [640, 50], each of the 640 neurons' (600
  studied, 40 for tuning) activations at its end-of-training top-50 positions, measured at that point. Its rows
  follow the lines of `samples/contexts.jsonl` beside it; at the last point the script checks all 640 × 50 values
  against the `a` of the top-50 entries there.
- `w3/grid/{1,1b}/out/<point>/samples/contexts.jsonl` (16 files): `q999`, each neuron's 99.9th percentile at that
  point, and at the last point (`t15_step1413814`) the text of the top-50 contexts (`rank`, `before`, `token`,
  `after`), from which each group's strongest context is taken. Each point's `metrics.json` is read for
  `complete` and `n_tokens_scored` (20,560,068 at every point).

The chance level, added 2026-11-05 (paths relative to `studies/next/`; written by CPU-only recomputations from the
same banked arrays, `10_RECOUNT2.md` and `06b_VERIFY_BIRTHS.md`):

- `10_recount2/out/05_per_cluster_R1t.csv` (2,428 rows, one per group of the 574 neurons with two or more groups):
  `pole`, `c`, `size`, `real` (the group's birth point, checked against the record), `p_late`, `p_early`,
  `ref_median`.
- `10_recount2/out/05_chance.json`: `reps` (2,000) and `summary.R1t` (`n_clusters`, `b_real`, `b_ref_mean`,
  `b_ref_ci95_over_replicates`, `all.late_beat`, `all.late_expected`, `all.early_beat`, `all.early_expected`).
- `10_recount2/out/04_b.json`: `b`, checked against the b computed here.
- `06b_verify_births/out/s03_histogram_tv.json`: `n_clusters`, `n_draws` (300), `real_counts` (checked against
  `q2.births`) and `random_mean_counts_A`.

## What `data.js` holds

`steps` (the 16 points: index, step, percent of training = step / 1,413,814), `n_tokens_scored`, `population`
(the 16 birth counts of `q2.births`, main plus non-main, 2,454 groups, 600 neurons) and `neurons`: the 7 in `NAMED`
in `extract.py` (the lithium neuron first, then the got and system neurons of walkthrough line 659 and the four
neurons the judge picture names), then 33 drawn by `random.Random(0).sample` from the other 593 neurons sorted as
strings. Each neuron holds its groups, sorted by birth, then by size, then by group number, as
`[name, contexts in the group, birth point 0–15, presence as 16 characters "0"/"1", strongest context at the end
as [last 64 characters before, marked token, first 28 characters after], chance level (the random groups' median
birth point 0–15, a half point when the median falls between two), "later" / "earlier" / "chance"]`; the last two
hold no value on a neuron with one group. `population.chance` holds the strip's 16 chance counts and
`population.chance_draws` their number of draws; `chance` holds the recount's totals for the caption (`reps`,
`alpha`, `n_groups`, `later`, `later_expected`, `earlier`, `earlier_expected`).

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/samples/q2_clusters.jsonl`, `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json` and the 16 points under `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid`.

| check | computed | expected |
|---|---:|---:|
| lines in q2_clusters.jsonl | 2,465 | 2,465 |
| echo groups (left out) | 11 | 11 (q2.echo_clusters) |
| groups | 2,454 | 2,454 (q2.n_clusters) |
| neurons | 600 | 600 (q2.n_poles) |
| presence cells recomputed from exemplar_a.npz and each point's q999 that equal the record | 39,264 of 39,264 | 39,264 |
| largest relative difference, recomputed median against the record's `med` | 0.0e+00 | below 1e-6 |
| the same with each point given the next point's q999 (must fail) | 33,620 of 39,264 | fewer than 39,264 |
| births recomputed from presence that equal the record's birth point | 2,454 of 2,454 | 2,454 |
| groups never born | 0 | 0 (q2.births) |
| late-born groups (born after the neuron's first-born group) | 1,535 | |
| late-born groups in the test (both readings agree, matched in the second sorting, not born at step 1,000) | 1,238 | 1,238 (q2.p_u.n_clusters); status late_primary 1,238 |
| groups present at an earlier point that was not followed by presence (before birth) | 155 | |
| groups absent at some point after their birth | 44 | 44 (E_skeptic.md §3.2) |
| b, neurons with 2+ groups whose strongest group is among the first born (ties count as first) | 273 of 574 = 0.4756 | 0.4756 of 574 (q2.birth_order.b); 0.4756 (recount `04_b.json`) |
| b at the chance level: random groups of the same sizes from the same neuron, the strongest group's holding the top context | 0.530 [0.495, 0.568] over 2,000 replicates | `05_chance.json` summary.R1t; b is below it. The registered permutation band (97.5th percentile 0.449) keeps the real groups' ties and is not a chance level for size and rank (retagged 2026-11-05) |
| groups on neurons with 2+ groups, and of them in the recount's per-group file | 2,428, 2,428 | 2,428 (summary.R1t.n_clusters) |
| of them, the recount's size and birth point equal the record's | 2,428 and 2,428 | 2,428 |
| the same join with each group given the next group's row of its neuron (must fail) | 655 of 2,428 | fewer than 2,428 |
| born later than chance (p_late ≤ 0.05) | 153 | 153 (summary.R1t.all; 22.4 expected at chance) |
| born earlier than chance (p_early ≤ 0.05) | 160 | 160 (summary.R1t.all; 29.6 expected at chance) |
| births per point in the verifier's file that equal q2.births | 16 of 16 | 16 |

Groups born at each training point, all 600 neurons:

| point | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | T15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step | 0 | 1,000 | 2,000 | 3,000 | 5,000 | 8,000 | 12,000 | 20,000 | 35,000 | 60,000 | 100,000 | 180,000 | 320,000 | 560,000 | 1,000,000 | 1,413,814 |
| recomputed | 0 | 1 | 3 | 14 | 30 | 49 | 62 | 97 | 160 | 176 | 238 | 346 | 467 | 489 | 286 | 36 |
| q2.births, main + non_main | 0 | 1 | 3 | 14 | 30 | 49 | 62 | 97 | 160 | 176 | 238 | 346 | 467 | 489 | 286 | 36 |
| chance level: random groups of the same sizes, mean of 300 draws (06b `s03_histogram_tv.json`) | 0.0 | 0.1 | 4.2 | 7.9 | 24.4 | 48.0 | 55.0 | 108.3 | 158.7 | 189.7 | 263.8 | 389.9 | 457.5 | 492.7 | 241.8 | 12.0 |

The lithium neuron, L03N6372+: 15 groups (present: 1, by point T0 to T15).

| group | contexts | born at | step | present | chance level (median of random groups of its size) | against chance |
|---|---:|---:|---:|---|---:|---|
| UC Santa Cruz | 4 | T5 | 8,000 | `0000011110001111` | T12 | earlier |
| misc proper-noun completions (Australian, Mozambique, Neue, Microgaming, salutations) | 5 | T10 | 100,000 | `0000000000111111` | T12 | earlier |
| Palo Alto | 5 | T12 | 320,000 | `0000000001001111` | T12 | chance |
| baking parchment | 3 | T12 | 320,000 | `0000000000001111` | T12 | chance |
| xampp / ampps | 2 | T12 | 320,000 | `0000000000001111` | T12 | chance |
| KubeCon | 2 | T12 | 320,000 | `0000000000001111` | T12 | chance |
| pexels URL | 2 | T12 | 320,000 | `0000000000001111` | T12 | chance |
| code/tech identifier fragments (git-svn, #![, Euler, apt-key adv, letsencrypt, eslint, jsonplaceholder) | 7 | T13 | 560,000 | `0000000000000111` | T12 | chance |
| MIT license / opensource URL tokens | 6 | T13 | 560,000 | `0000000000000111` | T13 | chance |
| HuggingFace | 3 | T13 | 560,000 | `0000000000000111` | T12 | chance |
| Solana | 2 | T13 | 560,000 | `0000000000000111` | T12 | chance |
| David Beckham surname | 3 | T14 | 1,000,000 | `0000000000000011` | T12 | chance |
| Industrial Hemp | 2 | T14 | 1,000,000 | `0000000000000011` | T12 | chance |
| Canyon (Laurel/Diablo) | 2 | T14 | 1,000,000 | `0000000000000011` | T12 | chance |
| Thomson Reuters | 2 | T15 | 1,413,814 | `0000000000000001` | T12 | later |

The got neuron, L03N7155+: groups born at steps 5,000, 20,000, 35,000, 100,000; expected 5,000, 20,000, 35,000, 100,000 (A_training_dynamics.md §5).

The picker holds the 7 named neurons in `extract.py` and 33 drawn by `random.Random(0).sample(pool, 33)` from the other 593 neurons (ids sorted as strings): L21N3516-, L23N10071-, L02N5291+, L14N4543+, L28N1930-, L26N929-, L22N4270-, L16N5767-, L26N5599+, L19N7374-, L12N1003+, L27N955+, L07N7004-, L15N10390-, L07N8340+, L05N534-, L13N2999-, L29N38-, L08N10948-, L16N864+, L05N8662-, L04N3542+, L18N2010-, L26N3583-, L30N9731-, L05N8714-, L19N2718-, L24N1234+, L17N4727+, L11N445+, L30N6280-, L24N3927-, L28N5955-.

Every expected number above matches, and both failing inputs fail (the next point's line, the next group's row).

Wrote `data.js`: 35,258 bytes, 40 neurons, 156 groups; sha256 `fade144e57b73fb3157d7a68754fa5e453183b9e2d78d499fb2ea8c93c006985`.
<!-- checks:end -->

## Verified on 2026-11-04

The standalone page from `file://` in Chrome through Playwright, at 1280 and 390 px wide, light and dark: no
console errors or warnings; `document.documentElement.scrollWidth` equals `window.innerWidth` (1280 and 390)
before and after the interactions; the autoplay runs from 0 to 15 of 15 groups for the lithium neuron and stops at
step 1,413,814; a click on the axis at the tenth point gives step 60,000 and "1 of 15 groups born", with "1 of them
is not present at this point" (UC Santa Cruz, absent from step 60,000 to 180,000); the right arrow then gives step
100,000 and 2 of 15; the menu's got neuron shows its four groups filled from points T4, T7, T8 and T10; the random
button shows a drawn neuron; the square and strip tooltips and the table open; under prefers-reduced-motion the
last point shows at once. Not verified: touch dragging on a real phone, and the picture inside `../index.html`
(not pasted yet).

## Verified on 2026-11-05 (the chance level)

Before the change, `python3 -I extract.py --out <scratch>` reproduced the committed `data.js` byte for byte (sha256
`68ed0ad8…`). After it, every old field of `data.js` is unchanged; each group gains its chance level and its call
against chance, and `population` and `chance` gain the counts above. The standalone page and `../index.html` were
loaded from `file://` in headless Chrome (no Playwright here), light and dark, with reduced motion so the last point
shows: the lithium neuron draws 15 rows with 15 chance ticks and 16 strip lines; its readout says 1 group later than
chance, 2 earlier and 12 at chance (UC Santa Cruz and the misc proper-noun group earlier, Thomson Reuters later); the
table carries the two new columns; the caption carries 153, 160, 22.4, 29.6 and 2,115. `scrollWidth` equals
`innerWidth` at 500 and 1280 px, before and after the change; headless Chrome does not go below 500 px, so 390 px
was not checked. The one console line on `../index.html` ("ResizeObserver loop completed with undelivered
notifications", INFO) appears on the unchanged page too. Tooltips, by synthetic mouse pointer events in a scratch
copy: UC Santa Cruz at step 8,000 reads "Chance level: step 320,000, … born earlier than chance", Thomson Reuters
"… born later than chance", the strip bar at step 1,000,000 "Chance level: 241.8 random groups …, the mean of 300
draws", and the one-group neuron L21N3516- "No chance level" with no tick drawn. Not verified: touch dragging, the
autoplay.
