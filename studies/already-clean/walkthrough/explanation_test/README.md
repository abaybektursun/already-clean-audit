# One explanation and its test texts

A one-picture module, spec P3 of `../VISUALS_PLAN.md`. It shows one side of one OLMo-3 7B neuron: the
explanation a language model wrote after reading the neuron's 30 strongest contexts, and the 16 texts another
model wrote from that sentence alone (5 meant to fire the neuron, 5 near misses, 5 distant texts, and one meant
to fire it as hard as possible). Each text has a bar for its score and the word where the neuron peaked is
marked. A vertical line, the strong line, runs across all the bars. Each group of five counts its strong texts
in five squares against its limit (at least 3, at most 2, at most 1), and the head says passes or fails.

The viewer drags the line (on its tag, on the axis, or along any bar) and watches the result change; picks
another neuron from the menu; or presses "Score the same texts at a random neuron", which redraws the bars with
the same 15 texts scored at another neuron of the same layer drawn at random, so the bars fall to near zero and
the explanation fails. On first view the bars grow from zero, the line drops in from its tag, and then the
result appears (skipped under prefers-reduced-motion; any press or key inside the picture finishes it).

## Status on 2026-11-04

Built from local copies on the laptop. Not yet inserted in the walkthrough page; `INSERT.md` holds the markup,
the place (line 356, in place of the section "One pass and one fail, from the real data") and the prose it
replaces. `data.js` is about 114 KB, so it is not stored through Git LFS.

## Files

- `explanation_test.css`: the styles. Every rule is scoped under `#xtest` and every class is prefixed `xt-`.
  It reads the host's `--bg`, `--ink`, `--muted`, `--rule`, `--grid`, `--bar`, `--accent`, `--font-text` and
  `--font-mono`, light and dark.
- `explanation_test.js`: builds the picture inside `<div id="xtest">` and does nothing if that element is
  absent. No framework, no library. On load it recomputes every neuron's tally and pass or fail at the study's
  line and throws if any differs from what `extract.py` stored beside it.
- `index.html`: the standalone page, with the walkthrough's fonts (Atkinson Hyperlegible, IBM Plex Mono),
  tokens and reading column (47.5rem). The data is loaded with a script tag, because Chrome refuses `fetch()`
  of local JSON from a page opened from disk.
- `data.js`: sets `window.EXPLANATION_TEST_DATA`. Written by `extract.py`.
- `extract.py`: reads the saved files, writes `data.js`, prints the checks and rewrites the checks block
  below. Run `python3 extract.py` from this folder. No model, no download, no network. It loads the rule
  from `../one_job/extract.py` (`n_strong`, `explanation_passes`, `assign_class`, `class_at`, `compact`) rather
  than porting it a second time.
- `INSERT.md`: the section to paste into `../index.html`, where it goes, and the lines it replaces.

## Inputs

Local copies, with their S3 source in bucket `pg-lineage-769063704696`:

- `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json` (136,032 bytes), from
  `results/neurons/w5/verdict/metrics.json`. Keys read: `theta` (the strong line and its gate) and `q1.O7`
  (`fractions`, `band_theta_minus_0.1`, `band_theta_plus_0.1`, `strong_rates`, `by_explainer_model`,
  `X7_pass_rate_by_encoding_flag`, `inputs.scores.sha256`).
- `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/samples/q1_O7_poles.jsonl`
  (24,420,010 bytes), from `results/neurons/w5/verdict/samples/`. 600 lines, one per side of 300 neurons,
  written by `unblind` in `infra/pg/neurons_verdict.py`. Fields read: `pole`, `explanation`, `explainer_model`,
  `texts` (`pos`, `near`, `distant`, `max`, each text with `text_id`, `text`, `score`, `strong`), `shuffle_pole`
  and `shuffled` (the same 15 texts scored at that neuron), `passes`, and for the class check `label`, `jobs`,
  `rare`, `declined`, `class`, `class_theta_minus`, `class_theta_plus`. One side, L27N7623−, has no texts
  (its writer declined three times).
- `/Users/abaybektursun/projects/interp_research/.pg_pulls/results/neurons/w2c/o7_probe/1/probe/poles/`:
  `probe/scores.npz` (94,951,894 bytes; sha256 `1ed18219…`, the file `metrics.json` names at
  `q1.O7.inputs.scores`), `probe/tokens.json` (2,816,849 bytes) and `poles.json` (the column order), from
  `results/neurons/w2c/o7_probe/1/probe/poles/` (the probe step of `steps.json`, OLMo-3 7B at
  stage1-step1413814). Arrays read: `text_ids`, `max_a`, `min_a`, `argmax_tok`, `argmin_tok`, `n_tokens`.

## How pass or fail is derived

SPEC_D1 §2.2 and §6.1, as `infra/pg/neurons_verdict.py` implements them and `../one_job/extract.py` ports them.

1. A text's score is its peak activation at the neuron side over text positions 1 and up, divided by that side's
   largest activation in the 50.5M-token corpus (`max_a / max|a|` for a positive side, `min_a / −max|a|` for a
   negative one). The score is stored; the page does not recompute it.
2. A text is **strong** if its score is at least the strong line (`>=`).
3. The explanation **passes** if at least 3 of the 5 texts written to fire the neuron are strong, at most 2 of
   the 5 near misses and at most 1 of the 5 distant texts. The text written to fire it as hard as possible is
   not part of the rule.

A pass is not the class "one job": that also needs the explanation's own label and job list to agree
(`assign_class`). `extract.py` checks both against the stored values.

## The strong line

θ_s = 0.31525558829307565, the 99th percentile of 1,785 scores of texts at a random neuron on the Qwen3-4B
discovery set (`metrics.json` `theta`; SPEC §6.1). The registered stops are θ_s − 0.1, θ_s and θ_s + 0.1, shown
as 0.2153, 0.3153 and 0.4153. The line moves from 0 to 1.2 in steps of 0.005; within half a step of a stop it
takes the stop's exact value. The axis also runs from 0 to 1.2, so 1 (as strong as the neuron's strongest token
in the corpus) sits inside it; a score above 1.2 (2 of the 656 texts on the page) is drawn to the end with a
break, and its number is printed.

## The marked word

The word marked in each text is the token where the neuron side peaked: `argmax_tok` for a positive side,
`argmin_tok` for a negative one, at the side's column of `scores.npz`; at the random neuron, at that neuron's
column. `tokens.json` holds each text's tokens; `extract.py` lays them along the text and asserts that they
cover it exactly. Where the tokenizer split one character into byte pieces (shown as U+FFFD), the run of pieces
gets the span of the characters it covers. A token's leading space is left outside the mark on the page.

## The random comparison

The button uses the stored `shuffled` scores: the same 15 texts scored at `shuffle_pole`, another neuron side of
the same layer drawn at random (SPEC §5). The record has no score for the text written to fire it as hard as
possible at that neuron, so that row reads "not scored" there. At the study's line, 23 of the 8,985 such scores
are strong and no explanation passes.

## The examples

The walkthrough's worked examples (lines 356–411) are two poles of the Qwen3-4B discovery set, L09N8987−
(passes) and L02N3919− (fails on its near misses). Their verdict files are on S3 only and are not among the
local files, so the picture uses OLMo sides instead:

- L03N7155+, the got neuron, named on the page (lines 288, 622); it passes with 5, 0 and 0 strong, the tally
  of L09N8987−.
- L25N5542+, the side that fails on its near misses with the tally nearest L02N3919−'s 5, 5 and 0: it has
  5, 3 and 0 (rule in `extract.py`: near misses above 2, smallest distance to 5 and 5, then pole id). At 0.4153
  it passes.
- L03N6372+, the lithium neuron, named on the page (lines 293, 622); it passes with 5, 0 and 0.
- 19 sides that pass and 19 that fail at the study's line, drawn with `random.Random(0).sample` from the other
  596 sides with texts, sorted by pole id. One of them, L03N7155−, is the negative side of the got neuron.

## Colours

A strong text's bar and the filled squares use the host's accent (#0a6b5d light, #4cc3ae dark); a text below
the line keeps the host's --bar grey (#b3bfbb, #4a5754). With the dataviz validator the pair separates by
ΔE 32.8 (light) and 31.4 (dark) for normal vision and 29.7 and 29.8 under colour-vision deficiency. The grey
is a de-emphasis colour and is below 3:1 against the surface (1.74 and 2.44), so every score is printed beside
its bar, the squares carry the count in words beside them, and the table view lists every score.

## Not shown

The explanation's class (one job, two related jobs, ...) is not on the page; the picture is about the test.
The texts written from each job of an "unrelated jobs" explanation (`job_pos:k`, 34 sides) are not shown.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict` and `/Users/abaybektursun/projects/interp_research/.pg_pulls/results/neurons/w2c/o7_probe/1/probe/poles`.

scores.npz sha256 1ed18219c54bae4802a50510a80bf5d7df916f4b1cb820ff775d82fdf6f3ca8b equals metrics.json q1.O7.inputs.scores.sha256; 10,404 texts x 1,263 neuron sides
stored scores against scores.npz: one multiple per neuron side for all 1188 sides (the same check on the neighbouring column fails for 889)
pass or fail recomputed at the strong line 0.3153: agrees with the stored value for 599 of 599 neuron sides with texts (1 without texts: L27N7623-); fed 0.4153 instead it disagrees for 94
class recomputed at 0.2153, 0.3153 and 0.4153: equals the stored class_theta_minus, class and class_theta_plus for all 600

| strong line | passes (this run) | passes in metrics.json | one job (this run) | one job in metrics.json | expected |
|---:|---:|---:|---:|---:|---:|
| 0.2153 | 434 of 599 | not stored | 370 | 370 | 370 |
| 0.3153 | 365 of 599 | 365 (by_explainer_model) and 365 (X7) | 311 | 311 | 311 |
| 0.4153 | 275 of 599 | not stored | 235 | 235 | 235 |

| texts strong at the study's line | this run | metrics.json q1.O7.strong_rates | expected |
|---|---:|---:|---:|
| texts written to fire it | 1,794 of 2,995 | 1,794 of 2,995 | 1,794 of 2,995 |
| near misses | 112 of 2,995 | 112 of 2,995 | 112 of 2,995 |
| distant texts | 2 of 2,995 | 2 of 2,995 | 2 of 2,995 |
| the same 15 texts at the random neuron | 23 of 8,985 | 23 of 8,985 | 23 of 8,985 |

Scored at the random neuron instead, 0 of 599 explanations pass at 0.3153 (at their own neuron: 365).

data.js: 113,604 bytes, 41 neuron sides: L03N7155+ (the got neuron), L25N5542+ (the fail stand-in), L03N6372+ (the lithium neuron), 19 passes and 19 fails drawn with random.Random(0) from the other 596 sides with texts. Scores at 6 decimals give the same pass or fail as full precision at all 241 positions of the strong line (0.0 to 1.2 in steps of 0.005), at the neuron and at the random neuron.

Two of the page's neuron sides, with the token where each text peaked marked ⟦ ⟧:

L03N7155+, explanation by Opus 5.5, passes at 0.3153 with 5, 0 and 0 strong; random neuron L03N9307+:

> Fires on the word piece right after a capital or lowercase "G" that spells colloquial "got" words: "otta" in "Gotta" (especially sentence-initial "Gotta say/love/admit/catch 'em all"), "ot" in "Got it!"/"GOT", and "tin" in "Gettin'".

- pos, score 0.7639 (strong); at the random neuron 0.0495: G⟦otta⟧ say, that pizza was the best thing I've eaten all year.
- pos, score 0.7639 (strong); at the random neuron 0.0080: G⟦otta⟧ catch 'em all! Pikachu is next on my list.
- near, score 0.0188; at the random neuron 0.0452: I⟦ have⟧ to say, that pizza was the best thing I've eaten all year.
- near, score 0.0094; at the random neuron 0.0061: You must catch⟦ them⟧ all! Pikachu is next on my list.
- distant, score 0.0299; at the random neuron 0.0199: Basaltic magma rises through fractures in the⟦ crust⟧ and collects in a shallow chamber beneath the volcano.
- max, score 0.7639 (strong): G⟦otta⟧ say, gotta love it, gotta catch 'em all — got it, I'm gettin' started right now!

L25N5542+, explanation by Sonnet 5.5, fails at 0.3153 with 5, 3 and 0 strong; random neuron L25N10671+:

> Fires on function words or punctuation (the, its, 's, ', or, a, comma, colon) that sit right before an expected capitalized acronym or proper-name title, such as a government agency, program or exam/certification (DHS' AIS, the IELTS, FBI's Virtual Case File, the GMAT, SCJP, A Levels, the U.S. State Department's STEP) in US government, cybersecurity, or exam/certification prose.

- pos, score 0.4824 (strong); at the random neuron 0.0786: Applicants who want to study in Canada must first pass the IELTS, and many also⟦ take⟧ the GMAT before applying to business school.
- pos, score 0.7903 (strong); at the random neuron 0.0888: The agency's rollout of DHS⟦'⟧ AIS program lets private firms share indicators of compromise with federal analysts in near real time.
- near, score 0.3582 (strong); at the random neuron 0.0712: Applicants who want to study in Canada must first pass an English proficiency test, and many also⟦ take⟧ an entrance exam before applying to business school.
- near, score 0.1702; at the random neuron 0.0898: The agency's rollout of an⟦ automated⟧ sharing program lets private firms send threat indicators to federal analysts in near real time.
- distant, score 0.0297; at the random neuron 0.0818: Feed your starter the night before, then mix it with flour⟦ and⟧ warm water until a shaggy dough forms.
- max, score 0.4777 (strong): Before applying, candidates must pass the GMAT and the IELTS, while security staff follow the DHS' AIS guidance and the FBI⟦'s⟧ Virtual Case File lessons.

Every expected number above matches.
<!-- checks:end -->
