# What a top context is

A picture for the walkthrough (spec P1 in `../VISUALS_PLAN.md`). Pick a neuron of OLMo-3 7B at the end of
training. A long thin line stands for the corpus it was scored on, in reading order; its 50 strongest
contexts stand on the line as ticks, each at its position in the corpus, each as tall as its activation. A
dashed line marks the neuron's 99.9th percentile over all scored tokens. Clicking a tick opens that context
below, with the token it fired on marked and its activation shown, in the same row format as "One neuron
across training". When the picture first comes into view the line draws across once, the ticks rise as it
passes them, and the strongest one opens.

It has two hosts: the walkthrough page (`../index.html`, once the section in `INSERT.md` is pasted in) and
`index.html` here. Both load the same three files.

## Files

- `top_contexts.css`: the styles, every rule scoped under `#topctx`. It reads the host's tokens (`--bg`,
  `--ink`, `--muted`, `--rule`, `--accent`, `--font-text`, `--font-mono`, light and dark). The picker,
  buttons, context row and tooltip copy `../neuron_across_training/slider.css`.
- `top_contexts.js`: builds the picture inside `<div id="topctx">` and does nothing if that element is
  absent. No framework. The intro runs once, when the plot is first 60% in view, and lasts 2.6 s; under
  prefers-reduced-motion, or after any press or focus inside the picture, it is skipped to its end. A click
  in the plot opens the tick whose top is nearest the pointer; with focus on the plot, the arrow keys, Home
  and End step through the 50 by rank, as do the buttons "Stronger" and "Weaker". The table under "Show the
  50 as a table" lists every context, and its rank buttons open them too, so ticks that overlap (contexts
  from one document sit within a pixel of each other) can all be reached. "Random neuron" picks one of the
  18 drawn neurons.
- `data.js`: sets `window.TOP_CONTEXTS_DATA`. Written by `extract.py`. About 0.25 MB, so a plain file, not
  Git LFS.
- `extract.py`: reads the saved files, writes `data.js`, prints its checks and rewrites the checks block
  below. Run `python3 extract.py` from this folder. No model, no download, no network.
- `index.html`: the standalone page, with the walkthrough's tokens and fonts (Atkinson Hyperlegible, IBM
  Plex Mono). The data is loaded by a script tag because Chrome refuses `fetch()` of local JSON from a page
  opened from disk.
- `INSERT.md`: the section to paste into the walkthrough page, and where.

## Inputs

Default paths are the local copy on the laptop (`/Users/abaybektursun/projects/interp_research_banks/neurons`).
The S3 source of each is in bucket `pg-lineage-769063704696`, under `results/neurons/` at the same path.

- `w3/grid/1/out/t15_step1413814/samples/contexts.jsonl` (16,502,054 bytes): 640 lines, one per neuron
  side, each with `max_abs`, `q99`, `q999`, `q9999`, `density` and `topk`, 50 entries of `rank`, `tok`,
  `doc`, `pos`, `a`, `g`, `u`, `before`, `token`, `after` (`DATA_CONTRACT.md` §2). `before` holds the 64
  tokens before the marked token and `after` the 16 after, decoded and cut at the document's edges.
- `w3/grid/1/out/t15_step1413814/metrics.json`: `complete`, `n_tokens_scored`.
- `w3/grid/1/out/t15_step1413814/poles/exemplar_a.npz`: `a`, `g`, `u` [640, 50]. Read only as a check.
- `w3/o7/draw/S.json`: the 600 study neurons, for the random draw (rebuilt on 2026-11-04; see the note
  beside it and `../neuron_across_training/README.md`).

## Positions, and the axis

Positions are in the record. `tok` is the index of the marked token in the corpus token stream, `doc` its
document, `pos` its place in the document. So the ticks are drawn by position, not by rank.

The axis is the stream, 20,570,112 positions, not the 20,560,068 scored tokens. The harvest cut the stream
into 10,044 windows of 2,048 tokens and did not score the first token of each window, which has no text
before it (`infra/pg/neurons_harvest.py`: `n_scored = n_windows * (window - 1)`). `tok` counts every
position, so 6 of the 32,000 saved positions are 20,560,068 or more, and the plan's assert "every tok <
20,560,068" fails as written. `extract.py` asserts instead that every `tok` is below 20,570,112 and that
none sits at a window's first position; both hold.

## The comparison with ordinary text

The brief asked, if the record allows it, for the same neuron's activation at 50 random positions beside
its 50 strongest. It does not allow it: `exemplar_a.npz` at the final step holds the activations at the
neuron's own top-50 positions only (they equal the 50 in `contexts.jsonl` for all 640 sides), and no file
on the laptop holds activations at random positions. The picture draws the neuron's 99.9th percentile
(`q999`) instead. It is the upper edge of the histogram bin that holds that quantile of |a| over the side's
tokens (bins 5.9% wide, `DATA_CONTRACT.md` §2), so at most 1 in 1,000 scored tokens goes past it; the label
says exactly that. The caption also gives `density`, the number of scored tokens at or above 1% of the
neuron's strongest activation (same histogram, so "about"), to say that the 50 are the neuron's peak, not
all it does (`VISUALS_PLAN.md` P1, "Avoid").

## What `data.js` holds

20 neurons: the lithium neuron (L03N6372+), the "got" neuron (L03N7155+), then 18 drawn by
`random.Random(0).sample` from S.json minus those two, sorted as strings. The brief named lithium first
and a seed-0 draw; the got neuron was added because `VISUALS_PLAN.md` P1 names it as the default and its
claim (0.663 and 0.050) is about it. Each neuron keeps `max_abs`, `q999` (4 significant digits), `density`,
and its 50 contexts in rank order as `[tok, doc, a, before, token, after]`, with `before` cut to its last
180 characters and `after` to its first 36 (about 40 and 8 tokens; the decoded text carries no token
boundaries). Negative sides are drawn upward by size, with signed labels.

## Viewer text

No "pole", "harvest", "exemplar" or "W3". The caption names the corpus and says that the explanations
elsewhere on the page came from a separate, larger run of about 50 million tokens. That figure is the
walkthrough's own (line 321, "About 50 million tokens"); `writeup/stories/A_training_dynamics.md` line 208
gives 50.5M and `F_validation.md` line 33 gives 50,515,866 for OLMo from the run's manifest, which is not on
the laptop and was not read here.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid/1/out/t15_step1413814`.

- `n_tokens_scored` 20,560,068 (expected 20,560,068): match; `complete` true.
- The stream: 10,044 windows of 2,048 tokens = 20,570,112 corpus positions; the first position of each window is not scored, which leaves 10,044 x 2,047 = 20,560,068.
- contexts.jsonl: 640 lines, one per neuron side. Entry fields: a, after, before, doc, g, pos, rank, tok, token, u.
- Positions exist: yes. `tok` is the position in the corpus token stream, `doc` the document, `pos` the position inside the document (DATA_CONTRACT §0, §2).
- Sides with 50 valid contexts (no `tok` -1, no NaN): 640 of 640.
- Every side's 50 are in rank order 0 to 49, sorted in the side's sign, all of that sign, and rank 0 equals the line's `max_abs`.
- Largest `tok` 20,569,527, smallest 1,401. Every `tok` < 20,570,112 (the stream): yes. None at a window's first position (`tok` % 2048 == 0): yes.
- The plan's assert "every tok < 20,560,068" fails for 6 of 32,000 entries: `tok` indexes the 20,570,112-position stream, not the count of scored tokens. The picture's axis is the stream.
- `doc` never decreases as `tok` increases, over all 32,000 entries: yes.
- 31,878 distinct positions; a position that is in several sides' top 50 decodes to the same text in each.
- exemplar_a.npz: `a` (640, 50); row i equals line i's 50 activations for 640 of 640. It holds only the top-50 positions, so the record has no activations at random positions; the picture shows the neuron's 99.9th-percentile line instead.
- Random neurons: `random.Random(0).sample(pool, 18)`, pool = S.json (600) minus L03N6372+ and L03N7155+, sorted (598): L21N2147+, L23N10037+, L02N4649-, L13N9773-, L27N9651-, L26N718-, L22N3469+, L16N3013+, L26N4202+, L19N6189+, L31N7529-, L11N7616-, L27N8796+, L07N6924+, L15N10355+, L07N6924-, L05N4149-, L13N2847+.
- Wrote data.js: 253,851 bytes, 20 neurons x 50 contexts.
- The lithium neuron (L03N6372+): strongest activation 0.246 (expected 0.246): match; 99.9th percentile 0.050; 50th strongest 0.157; 40 documents.
- The got neuron (L03N7155+): strongest activation 0.663 (expected 0.663): match; 99.9th percentile 0.050; 50th strongest 0.492; 44 documents.
- The got neuron's 99.9th percentile 0.050 (expected 0.050): match. It is the upper edge of a histogram bin (resolution 5.9%, DATA_CONTRACT §2), so at least 99.9% of the side's tokens lie below it.
- ' lithium' among the lithium neuron's final 50: 0.
- The got neuron's most common marked tokens: 'otta' 18, 'Got' 12, 'ot' 7.

Lithium neuron, rank 1, activation 0.246, position 15,856,638, document 11,116:

```
…lly, is provided in a format readable by libfsmtest 222Licensed according to MIT license https://opensource.org/licenses[/MIT]. Source code available under https://bitbucket.org/JanPeleska/lib
```
<!-- checks:end -->

## Verified

With Playwright and the installed Chrome, from `file://`, at 1280 and 390 px, light and dark: no console
errors or warnings, `document.documentElement.scrollWidth` equal to `window.innerWidth`, the intro ends
with 20,560,068 and rank 1 open (activation 0.246, `/MIT`), a tick click opens that tick (spike 10 gives
rank 10, 0.197, ` Beckham`), the arrow key and the "Weaker" button step the rank, the picker opens the got
neuron (0.663, line at 0.050), "Random neuron" changes the neuron, and a table rank button opens its
context. Under reduced motion the page starts at the end state.

In a scratch copy of the walkthrough page with the three pastes of `INSERT.md` applied (not committed), at
1280 px light and 390 px dark: no errors, no horizontal scroll, the intro runs, and the slider "One neuron
across training" still builds beside it.

The dataviz palette validator on the two tick colors (the accent and the gray of the other ticks): CVD
separation ΔE 13.3 in light and 15.9 in dark, normal-vision ΔE 17.2 and 18.8, contrast at least 3:1 against
the surface in both. It flags the gray's low chroma, which is intended, and the dark accent's lightness,
which is the host's own token. The open tick also differs by width and by its dot.
