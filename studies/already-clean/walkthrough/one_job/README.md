# How many neurons have one job

A one-picture module, spec P4 of `../VISUALS_PLAN.md`. Three columns of 600 dots, one column per model
(Qwen3-4B base, Gemma-3 4B, OLMo-3 7B). Each dot is one side, positive or negative, of one of 300 neurons
drawn at random from the model, coloured by its class: one job, two related jobs, unrelated jobs, or
cannot tell. The dots fill each column from the bottom, one job first, so the one-job count reads against a
line drawn at 540 of 600, the essay's estimate of about 90%. The viewer drags the strong line and the
classes are recomputed from the stored test scores. Pointing at a dot, or pressing "Random neuron", shows
that neuron's explanation, its class, and the tally of strong texts that decided it.

On first view the dots stand in a fixed shuffle, then sort into their classes once while the one-job
counts count up (skipped under prefers-reduced-motion; any press or focus inside the picture sorts it at
once).

## Status on 2026-11-04

Built from the copy of the banks on the laptop. Not yet inserted in the walkthrough page; `INSERT.md`
holds the markup and the place. `data.js` is 1,003,605 bytes and is stored through Git LFS.

## Files

- `one_job.css`: the styles. Every rule is scoped under `#onejob` and every class is prefixed `oj-`, so
  the host page's own classes (`.ref`, `.track`, `.row`, ...) neither leak in nor out. It reads the host's
  `--bg`, `--ink`, `--muted`, `--rule`, `--accent`, `--font-text` and `--font-mono`, light and dark.
- `one_job.js`: builds the picture inside `<div id="onejob">` and does nothing if that element is absent.
  No framework, no library. It checks on load that its one-job counts at the study's line equal the ones
  `extract.py` computed, and throws if not.
- `index.html`: the standalone page, with the walkthrough's fonts (Atkinson Hyperlegible, IBM Plex Mono),
  tokens and reading column (47.5rem). The data is loaded with a script tag, because Chrome refuses
  `fetch()` of local JSON from a page opened from disk.
- `data.js`: sets `window.ONE_JOB_DATA`. Written by `extract.py`.
- `extract.py`: reads the saved JSON, writes `data.js`, prints the checks and rewrites the checks block
  below. Run `python3 extract.py` from this folder. No model, no download, no network.
- `INSERT.md`: the section to paste into `../index.html`, where it goes, and the prose it replaces.

## Inputs

Local copy, with its S3 source in bucket `pg-lineage-769063704696`:

- `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/metrics.json` (136,032 bytes),
  from `s3://pg-lineage-769063704696/results/neurons/w5/verdict/metrics.json`. Keys read: `theta`
  (θ_s and its gate), `q1.<model>.fractions`, `band_theta_minus_0.1`, `band_theta_plus_0.1` (the class
  counts at the three stops), `q1.<model>.strong_rates.shuffled` and `q1.O7.poles_list`.
- `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict/samples/q1_{Q4B,G4,O7}_poles.jsonl`
  (23,450,997, 23,685,393 and 24,420,010 bytes), from
  `s3://pg-lineage-769063704696/results/neurons/w5/verdict/samples/`. 600 lines each, one per pole, written
  by `unblind` in `infra/pg/neurons_verdict.py` (`samples[f"q1_{key}_poles"]`). Fields read: `pole`,
  `label` (the explainer's class), `jobs` (`related_to_others` of each), `explanation`, `explainer_model`,
  `rare`, `declined`, `texts` (`pos`, `near`, `distant` and `job_pos:k`, five texts each, with `score`
  and `strong`), `shuffled`, and the stored classes `class`, `class_theta_minus`, `class_theta_plus`. The
  explainer's inputs (`explainer_saw`) are not used. The lines hold U+2028, U+2029 and U+0085 inside strings, so the script
  splits on `\n` only.

## How the class is derived

The rule is SPEC_D1 §2.3 (classes) and §6.1 (strong response and legibility), as implemented in
`infra/pg/neurons_verdict.py` by `n_strong`, `explanation_passes`, `job_supported`, `assign_class` and the
per-pole loop of `q1_records`. `extract.py` ports those functions; `one_job.js` ports them again.

1. A text's score is its peak activation at the pole divided by the pole's largest activation in the corpus
   (`score`, SPEC §2.2; stored). A text is **strong** if its score is at least the strong line (`>=`).
2. An explanation **passes** if at least 3 of its 5 positives are strong, at most 2 of its 5 near misses
   and at most 1 of its 5 distant texts.
3. A pole whose `rare` is true (fewer than 50 corpus tokens with |a| at least 1% of its maximum, or a
   maximum of 0) is **RARE**, whatever its texts.
4. A pole with no texts (its explainer or writer declined after three attempts, or it never fires on its
   side) is RARE if rare, else **UNK**.
5. Otherwise: **ONE** if the explainer said ONE, listed exactly one job, and the explanation passes;
   **TWO-REL** if it said TWO-REL, listed 2 or 3 jobs all marked related to the others, and the explanation
   passes; **UNREL** if it said UNREL, marked at least 2 jobs unrelated, and at least 2 of those are
   supported (at least 3 of the 5 positives written from that job's sentence alone are strong); else
   **UNK**.

The picture's four groups are ONE ("one job"), TWO-REL ("two related jobs"), UNREL ("unrelated jobs"), and
UNK together with RARE ("cannot tell"). Only Qwen3-4B base has RARE poles (8), so its "cannot tell" is
192 + 8 = 200 at the study's line.

## The strong line

θ_s = 0.31525558829307565, the 99th percentile of 1,785 shuffled-pole scores on the Qwen3-4B discovery
set (`metrics.json` `theta`; SPEC §6.1). The three registered stops are θ_s − 0.1, θ_s and θ_s + 0.1
(`THETA_BAND`, the width control of SPEC §5), shown as 0.2153, 0.3153 and 0.4153.

The drag is continuous. The poles files hold the score of every test text, so the class at any line is
recomputed, not looked up. The slider runs from 0.05 to 0.80 in steps of 0.005; a position within half a
step of a stop takes the stop's exact value, so 0.215, 0.315 and 0.415 are replaced by the three stops.
Scores are stored at 6 decimals; `extract.py` fails if that rounding changes any class at any of the 151
positions (at 4 and at 5 decimals it did, at 6 it does not). In Chrome, the page's four counts per model
equalled the script's at all 151 positions.

## Colours

One job `#036f4f`, two related `#4ea683`, unrelated `#c9690c`, cannot tell `#bcc5c2` on the light surface
`#f4f6f4`; `#48a581`, `#047755`, `#cb7229` and `#3a4542` on the dark surface `#101514`. The three hues pass
the dataviz validator on all pairs in both modes (lightness band, chroma floor, CVD separation 9.4 light and
8.7 dark, normal-vision floor 18.1 and 15.1); two related is below 3:1 on the light surface, so the legend,
the panel and the table carry the class in words. The grey is a de-emphasis colour, not a hue; against
two related it measures 17.9 (light) and 15.5 (dark) normal-vision. The host's accent `#0a6b5d` fails the
chroma floor (0.084), so one job uses a nearby teal.

## Not shown

VISUALS_PLAN P4 names a second random comparison, the classes recomputed from the same texts scored at a
random neuron (the stored `shuffled` scores). The picture's random control is the "Random neuron" button
instead. `extract.py` prints that comparison: at the study's line it gives one job to 2, 1 and 0 poles.
The shuffled scores cover the 15 explanation texts only, not the job texts, so the unrelated class cannot
be recomputed from them.

## Checks

<!-- checks:start -->
Computed by `python3 extract.py` from `/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict`.

Strong line θ_s = 0.31525558829307565 (metrics.json `theta.theta_s`); the stops are θ_s - 0.1, θ_s and θ_s + 0.1.

| model | strong line | one job | expected | two related | unrelated | cannot tell (UNK) | too rare (RARE) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B base | 0.2153 | 383 | 383 | 33 | 10 | 166 | 8 |
| Qwen3-4B base | 0.3153 | 358 | 358 | 34 | 8 | 192 | 8 |
| Qwen3-4B base | 0.4153 | 313 | 313 | 25 | 6 | 248 | 8 |
| Gemma-3 4B | 0.2153 | 373 | 373 | 43 | 7 | 177 | 0 |
| Gemma-3 4B | 0.3153 | 338 | 338 | 36 | 7 | 219 | 0 |
| Gemma-3 4B | 0.4153 | 294 | 294 | 29 | 5 | 272 | 0 |
| OLMo-3 7B | 0.2153 | 370 | 370 | 40 | 10 | 180 | 0 |
| OLMo-3 7B | 0.3153 | 311 | 311 | 36 | 4 | 249 | 0 |
| OLMo-3 7B | 0.4153 | 235 | 235 | 27 | 4 | 334 | 0 |

At every stop the recomputed class equals the stored per-pole class for all 600 poles of each model, and the counts equal metrics.json `q1.<model>.band_theta_minus_0.1`, `fractions` and `band_theta_plus_0.1`. The page folds RARE into "cannot tell". The essay's estimate drawn on the page: 540 of 600.

| model | explainer said one job (expected) | shuffled texts strong at θ_s (expected) | one job from shuffled scores at θ_s | most one-job poles on the slider | classes at θ_s + 0.1 that differ from the stored class at θ_s |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | 525 (525) | 48 of 8,865 (48) | 2 | 388 at 0.2 | 70 |
| Gemma-3 4B | 520 (520) | 26 of 8,940 (26) | 1 | 394 at 0.14 | 63 |
| OLMo-3 7B | 521 (521) | 23 of 8,985 (23) | 0 | 399 at 0.14 | 89 |

The last column is the class check fed the wrong stop; it fails there, so it can fail. Scores are stored in `data.js` at 6 decimals; at all 151 slider positions (0.05 to 0.80 in steps of 0.005, with 0.215, 0.315 and 0.415 replaced by the exact stops) they give the same class as the full-precision scores for every pole.

Every expected number above matches.

One pole scored one job and one scored cannot tell, at θ_s, with two of each kind of text:

OLMo-3 7B, L00N181+, explainer Opus 5.5 said ONE; class at θ_s ONE:

> Fires on the token right after a tech/media/web noun such as website, media, application, project, email, file, news, book, home, editor, api, Java or photo, mainly when that next token is an unusual word or the start of a compound (e.g. "website necessitates", "project inception", "Java getter", "Book Trailers", "photoresponsive", "Website Extras").

- pos, score 0.5759 (strong): The new website necessitates a complete overhaul of our hosting plan before the launch.
- pos, score 0.4297 (strong): During project inception, the team drafted a charter and sketched the initial architecture.
- near, score 0.4094 (strong): The new plan necessitates a complete overhaul of our budget before the launch.
- near, score 0.0120: During the early weeks, the team drafted a charter and sketched the initial garden layout.
- distant, score 0.0289: The falconer lifted the glove and the peregrine launched into the cold morning air.
- distant, score 0.0235: Training a young hawk requires patience, hunger management, and a steady, quiet voice.
- scores, pos [0.576, 0.43, 0.577, 0.428, 0.711], near [0.409, 0.012, 0.048, 0.044, 0.044], distant [0.029, 0.023, 0.101, 0.012, 0.065]

OLMo-3 7B, L00N2968+, explainer Opus 5.5 said ONE; class at θ_s UNK:

> Fires on a verb token such as Send, Receive, Populate, Write or unsubscribe when it begins a CamelCase method, variable or class name in code, mostly C# or JavaScript networking and UI code (e.g. SendData, SendBytes, ReceiveBuffer, PopulateIndividual, Populate(), WriteRatio, unsubscribeRequest), rather than the same verb used as an ordinary word in prose.

- pos, score 0.1464: public async Task SendData(byte[] payload) { await socket.SendAsync(payload, SocketFlags.None); }
- pos, score 0.6139 (strong): private byte[] ReceiveBuffer = new byte[4096]; int n = stream.Read(ReceiveBuffer, 0, ReceiveBuffer.Length);
- near, score 0.0182: Please send the data to the server tomorrow morning, and I will receive it before lunch.
- near, score 0.0348: public async Task Transmit(byte[] payload) { await socket.EmitAsync(payload, SocketFlags.None); }
- distant, score 0.0386: The cheesemaker stirred the curds in the copper vat as the morning fog lifted over the alpine pasture.
- distant, score 0.0599: Wheels of cheese were aged for eight months in a cool stone cellar and turned by hand every week.
- scores, pos [0.146, 0.614, 0.311, 0.045, 0.351], near [0.018, 0.035, 0.037, 0.042, 0.026], distant [0.039, 0.06, 0.031, 0.033, 0.022]

<!-- checks:end -->
