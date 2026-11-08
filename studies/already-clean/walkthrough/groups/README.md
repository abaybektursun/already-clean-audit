# One neuron's 50 contexts, sorted into groups

A picture for the walkthrough (spec P5 in `../VISUALS_PLAN.md`). Pick a neuron of OLMo-3 7B. Its 50 strongest
contexts at the end of training appear as one list of tiles, strongest first, each tile showing the token
the neuron fired on. When the picture first comes into view the tiles move into the groups that a language
model sorted them into, one row per group with the group's name from the record, and the number of groups
appears. A switch shows the second sorting of the same neuron by the same model, and the tiles move into
its groups. A small chart shows how many groups each of the 600 study neurons got under the sorting shown,
with this neuron's count marked. Below, the selected group's contexts are listed as the slider "One neuron
across training" lists them: text before, the marked token, text after, activation.

It has two hosts: the walkthrough page (`../index.html`, once the section in `INSERT.md` is pasted in) and
`index.html` here. Both load the same three files.

## Files

- `groups.css`: the styles, every rule scoped under `#ngroups`. It reads the host's tokens (`--bg`, `--ink`,
  `--muted`, `--rule`, `--accent`, `--font-text`, `--font-mono`, light and dark).
- `groups.js`: builds the picture inside `<div id="ngroups">` and does nothing if that element is absent. No
  framework. The move from one list into groups runs once, when the board is first 35% in view; under
  prefers-reduced-motion, or after any press inside the picture, the groups show at once. Clicking a
  group's row, or pressing a tile, selects that group for the list below. Tiles carry a hover tooltip with
  the context; the list below holds the same text without hovering.
- `data.js`: sets `window.GROUPS_DATA`. Written by `extract.py`. It is about 0.5 MB, so it is stored as a
  plain file, not through Git LFS.
- `extract.py`: reads the saved files, writes `data.js`, prints its checks and rewrites the checks block
  below. Run `python3 extract.py` from this folder. No model, no download, no network.
- `index.html`: the standalone page, with the walkthrough's tokens and fonts (Atkinson Hyperlegible, IBM Plex
  Mono). The data is loaded by a script tag because Chrome refuses `fetch()` of local JSON from a page
  opened from disk.
- `INSERT.md`: the section to paste into the walkthrough page, and where.

## Inputs

Default paths are the local copy on the laptop (`/Users/abaybektursun/projects/interp_research_banks/neurons`).
The S3 source of each is in bucket `pg-lineage-769063704696`, under `results/neurons/` at the same path.

- The groups: `w4/o7_s_v2/explain/clusters.jsonl` (944,095 bytes, 1,200 lines). One line per neuron and
  sorting: `pole`, `sample` (0 is the first sorting, 1 the second), `clusters` (each `c`, `name`, `echo`,
  `exemplar_ranks`) and `pairs`. `exemplar_ranks` are ranks 0 to 49 in the end-of-training top 50
  (`DATA_CONTRACT.md` §3). The sorting model was Claude Sonnet 5.5 (`manifest.json` beside the file:
  batches `w4c-clusterer-s0` and `w4c-clusterer-s1`, 600 requests each), with the prompt
  `infra/pg/neurons_prompts/clusterer.md` (sha256 `5d8b55ca…`, as in the manifest). The prompt asks for
  one group per job, lets related offshoots of one job form their own group, and asks for one group, marked
  `echo`, holding exactly the contexts where both input terms of the neuron are negative. The study drops
  echo groups from every count (11 in each sorting), and so does this picture.
- The contexts: `w3/grid/1/out/t15_step1413814/samples/contexts.jsonl` (640 lines, one per neuron side,
  `topk` with 50 entries `rank`, `tok`, `a`, `before`, `token`, `after`, ...), and the `metrics.json` beside
  `samples/` (`complete`, `n_tokens_scored`).
- The verdict: `w5/verdict/metrics.json`, for `q2.inputs.clusters.sha256`, `q2.n_clusters`,
  `q2.echo_clusters`, `q2.B4_m_within_1` and `q2.gallery`.
- The study set: `w3/o7/draw/S.json` (600 neurons; rebuilt on 2026-11-04, see the note beside it and
  `../neuron_across_training/README.md`).

## What `data.js` holds

40 neurons. The 17 examples in `extract.py` (`NAMED`), each with the label the picker shows and where the
write-up names it; all are in the write-up except L26N929-, which was chosen here as a neuron with one
group under both sortings (the checks below count how many neurons have that). Then 23 drawn by `random.Random(0).sample` from S.json
minus the examples and the 11 neurons that have an echo group, sorted as strings. Leaving the echo neurons
out keeps every tile in a counted group; it departs from a draw over all 600.

Per neuron: the 50 contexts in rank order, each as `before` (its last 180 characters, about 40 tokens),
`token`, `after` (its first 36 characters, about 8 tokens) and the activation `a` (4 significant digits);
and for each sorting the groups without the echo group, each as its name and its ranks. The page orders the
groups by size, then by strongest context. For all 600 it holds only the distribution of group counts per
sorting, the totals, and how many neurons got the same count, or counts within one, under both sortings.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w4/o7_s_v2/explain/clusters.jsonl` and `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid/1/out/t15_step1413814/samples/contexts.jsonl`.

- The record's sha256 is `afa666e2b0bf2650cd1270a5a54b6a3510846229e5713ea6235259828a65f843`, the value at `q2.inputs.clusters.sha256` in `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json`.
- 1200 lines: 600 neurons (the 600 of S.json), each sorted twice. Every sorting puts each of the 50 contexts in exactly one group.
- The contexts file is complete (`complete` true, 20,560,068 tokens scored); every study neuron has 50 contexts, ranks 0 to 49 in activation order. A group's `exemplar_ranks` are these ranks.

| check | computed | expected |
|---|---:|---:|
| lithium neuron L03N6372+, groups under the first sorting | 15 | 15 |
| lithium neuron L03N6372+, groups under the second sorting | 3 | 3 |
| L06N572+, groups under the first and second sorting | 6 and 1 | 6 and 1 |
| L12N8337+, groups under the first and second sorting | 9 and 2 | 9 and 2 |
| L02N4649-, groups under the first and second sorting | 11 and 5 | 11 and 5 |
| L06N7759+, groups under the first and second sorting | 3 and 2 | 3 and (not stated) |
| groups over the 600, first sorting (echo groups dropped) | 2,454 | 2,454 (q2.n_clusters) |
| groups over the 600, second sorting | 2,025 | |
| echo groups, first and second sorting | 11 and 11 | 11 (q2.echo_clusters) |
| neurons whose two counts are equal | 182 of 600 | 182 (stories B_clutter.md Story 6) |
| neurons whose two counts are within one | 407 of 600 = 0.6783 | 0.6783 (q2.B4_m_within_1) |
| neurons with one group under both sortings | 9 (7 without an echo group) | |
| mean groups per neuron, first and second sorting | 4.090 and 3.375 | 4.09 and 3.375 (stories B_clutter.md Story 6) |
| q2.gallery exemplars found at one rank inside their group (3 neurons) | 26 of 26 | 26 of 26 |
| the same with every rank moved by one (a wrong join; must fail) | 3 of 26 | well below 26 |

How many groups the 600 study neurons get (echo groups dropped):

| groups | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first sorting | 26 | 71 | 149 | 150 | 96 | 53 | 29 | 12 | 7 | 3 | 2 | 1 | 0 | 0 | 1 |
| second sorting | 40 | 120 | 181 | 153 | 69 | 19 | 14 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

Every expected number above matches, and the wrong join fails.

The picker holds the 17 examples in extract.py and `random.Random(0).sample(pool, 23)`, where `pool` is S.json minus the examples and the 11 neurons with an echo group, sorted as strings (572 neurons): L22N6104+, L24N3498+, L02N58-, L15N10355+, L29N38+, L27N9651+, L23N1430+, L17N8536+, L27N7794-, L21N10623+, L12N9888-, L28N8194-, L08N6643+, L16N5767+, L08N6643-, L05N924-, L14N671-, L30N6507+, L09N10255+, L18N2010-, L06N2537+, L04N6542-, L19N1554+.

Wrote `data.js`: 504,195 bytes, 40 neurons, sha256 `b5be396ff6a4b5d23586dd34c82e2873480b0f9b8d17e43709c706d451e3317f`.

The lithium neuron's strongest context (rank 0, activation 0.246, token `/MIT`), in the group "MIT license / opensource URL tokens" (6 contexts) under the first sorting and "mid-identifier tokens in license URLs, tool and package names, code" (19 contexts) under the second:

```
…lly, is provided in a format readable by libfsmtest 222Licensed according to MIT license https://opensource.org/licenses[/MIT]. Source code available under https:
```
<!-- checks:end -->
