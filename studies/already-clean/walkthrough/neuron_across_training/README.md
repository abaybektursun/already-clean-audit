# One neuron across training

A one-page viewer. Pick a neuron of OLMo-3 7B and drag through 16 saved training points. The page shows
the neuron's 20 strongest contexts at that point, with the token it fired on marked, and two summary rows
for all 16 points: how many of the top 20 fire on one token, and the strongest activation (log scale).

It has two hosts: the section "One neuron across training" of the walkthrough page (`../index.html`) and
`index.html` here. Both load `../site.js` and the same three files.

## Status on 2026-11-04

`data.js` is built from the copy of the banks on the laptop (`/Users/abaybektursun/projects/interp_research_banks/neurons`), moved there from the external drive.
The checks below are from that run. Re-run from this folder on 2026-11-04, it wrote a `data.js` identical to
the one built earlier in `studies/already-clean/pages/neuron_across_training` of the `neuronslider` worktree (sha256
`c6c81ef1e7b9ab0812cefe9cee69d515b8b18ae60d55dd6ae68fba58dff03a06`, 2,659,905 bytes, 41 poles).

2026-11-08: `extract.py` now also writes `meta` (each of the 35 inputs with its sha256, the public repository's
commit and the page's data link) and prints each input's `path sha256`. Re-run that day, `data.js` is the earlier
file with the `meta` block added at its end and every other byte unchanged (sha256 `c7d820461e0e26220ca0f89266fd6b73b6bffa4f5e079cd0263b6720a825f740`,
2,666,255 bytes); the checks block below came out unchanged.

## Files

- `slider.css`: the styles, every rule scoped under `#nat`. It reads the host's color and font tokens
  (`--bg`, `--ink`, `--muted`, `--rule`, `--accent`, `--font-text`, `--font-mono`, light and dark).
- `slider.js`: registers its mount with `../site.js` under the id `nat`; site.js builds the viewer inside
  `<div id="nat">` when that element is on the page, and shows a message and throws when data.js is missing. It
  plays through the 16 points once when the chart first comes into view (not under prefers-reduced-motion);
  a click inside the viewer or any key press stops it. Arrow keys act while focus is inside the viewer, or
  anywhere on the page when `#nat` has `data-keys="page"` (as in `index.html`).
- `index.html`: the standalone page. It links `../site.css` (the tokens and the picture frame) and loads
  `../site.js` and the three files; site.js builds the caption's data link from `meta` in data.js. No
  framework, no JS library; the fonts are Atkinson Hyperlegible and IBM Plex Mono from Google Fonts, loaded
  without blocking the page, with system fonts as the fallback. The data is loaded with a script tag, because
  Chrome refuses `fetch()` of local JSON from a page opened from disk.
- `data.js`: sets `window.NEURON_DATA`, with `meta.sources` (each input's path and sha256), `meta.commit` and
  `meta.link` (this README in the public repository, since the inputs are on S3). Written by `extract.py`.
  Stored through Git LFS.
- `extract.py`: reads the saved JSON, writes `data.js`, prints its checks and rewrites the checks block
  below. Run `python3 extract.py` from this folder. No model, no download, no network.

## Inputs

Default paths are the local copy on the laptop. The S3 source of each is in bucket
`pg-lineage-769063704696`.

- Contexts: `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid/{1,1b}/out/tNN_stepN/samples/contexts.jsonl`,
  from `s3://pg-lineage-769063704696/results/neurons/w3/grid/{1,1b}/out/<step>/samples/contexts.jsonl`.
  16 directories (t00 to t06 and t15 in `1/`, t07 to t14 in `1b/`; the script takes each tNN once and
  fails if one is missing or repeated). 640 lines each, one per pole, field `pole`, and `topk` with 50
  entries (`tok`, `a`, `before`, `token`, `after`, ...).
- Completeness: each step's `metrics.json` next to `samples/`, from
  `s3://pg-lineage-769063704696/results/neurons/w3/grid/{1,1b}/out/<step>/metrics.json`; the script
  asserts `complete` is true and that `n_tokens_scored` is the same at every step.
- The late group: `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json`, from
  `s3://pg-lineage-769063704696/results/neurons/w5/verdict/metrics.json`, key `q2.H_MIG.poles` (poles whose
  main pattern at the final point appeared late and that a judge model could not link to their earlier
  top contexts). The key holds 17 poles, lithium among them; lithium is removed, leaving 16.
- The study set: `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/o7/draw/S.json` (600 poles) and
  `S_gate.json` (40 gate poles); the script asserts they are disjoint and together equal the 640 harvested
  poles. These two files are rebuilt, not original. The provenance note beside them
  (`/Users/abaybektursun/projects/interp_research_banks/neurons/w3/o7/draw/README.md`) reads:

  > S.json (600) and S_gate.json (40) rebuilt 2026-11-04 from the 640 pole ids in w3/grid/1/out/t15_step1413814/samples/contexts.jsonl minus the 40 ids at q1.O7.poles_list.excluded_by_list.poles in w5/verdict/metrics.json (S3 results/neurons/w5/verdict/metrics.json). The original draw files were not found on S3.

- The 24 random neurons: `random.Random(0).sample(pool, 24)`, where `pool` is `S.json` minus the lithium
  pole and the late group, sorted as strings. Removing them keeps the button on ordinary neurons; it
  departs from a draw over all 600.

## What `data.js` holds

For each of the 41 poles (lithium, the 16 others of the late group, the 24) and each of the 16 steps: the top 20 contexts by
activation in the pole's sign (padding slots with `tok` -1 or `a` NaN dropped), each as `before` (its last
180 characters, about 40 tokens at 4.5 characters per token; the decoded text carries no token
boundaries), `token`, `after` (its first 36 characters, about 8 tokens) and `a` (4 significant digits);
the largest |a| over the 50; and the count of each distinct `token` string among the top 20. A context
that appears at several steps is stored once, keyed by its corpus position `tok`; the script fails if the
same position decodes to different text at two steps. Per step it also stores the step number and the
percent of training, step / 1,413,814.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid`.

| point | step | ' lithium' in top 20 | ' lithium' in top 50 | strongest activation | expected |
|---|---:|---:|---:|---:|---|
| t00 | 0 | 0 of 20 | 0 of 50 | 113 |  |
| t01 | 1,000 | 0 of 20 | 0 of 50 | 83.2 |  |
| t02 | 2,000 | 0 of 20 | 0 of 50 | 124 |  |
| t03 | 3,000 | 0 of 20 | 0 of 50 | 146 | max about 146 |
| t04 | 5,000 | 0 of 20 | 0 of 50 | 127 |  |
| t05 | 8,000 | 0 of 20 | 0 of 50 | 144 |  |
| t06 | 12,000 | 1 of 20 | 1 of 50 | 98.1 |  |
| t07 | 20,000 | 0 of 20 | 1 of 50 | 45.9 |  |
| t08 | 35,000 | 11 of 20 | 25 of 50 | 18 |  |
| t09 | 60,000 | 20 of 20 | 47 of 50 | 5.46 | 20 of 20, max about 5.46 |
| t10 | 100,000 | 19 of 20 | 40 of 50 | 2.41 | 19 of 20 |
| t11 | 180,000 | 19 of 20 | 44 of 50 | 2.31 | 19 of 20 |
| t12 | 320,000 | 19 of 20 | 40 of 50 | 1.47 | 19 of 20 |
| t13 | 560,000 | 1 of 20 | 6 of 50 | 1.18 |  |
| t14 | 1,000,000 | 3 of 20 | 7 of 50 | 0.547 | 3 of 20 |
| t15 | 1,413,814 | 0 of 20 | 0 of 50 | 0.246 | 0 of 50, max about 0.246 |

Every expected number above matches.

Top context at step 60,000 (t09), rank 1, activation 5.46, token ` lithium`:

```
…ently⏎⏎Smart mode is the recommended and easiest option for most situations. It switches automatically between using the[ lithium]-ion battery and the petrol-powered range extender.⏎⏎In this mode the taxi will
```

Top context at the final step (t15), rank 1, activation 0.246, token `/MIT`:

```
…lly, is provided in a format readable by libfsmtest 222Licensed according to MIT license https://opensource.org/licenses[/MIT]. Source code available under https://bitbucket.org/JanPeleska/lib
```

<!-- checks:end -->
