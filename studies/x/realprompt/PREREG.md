# REALPROMPT — pre-registration

2026-10-26. Frozen before any launch; the commit that adds this file and `infra/pg/realprompt_plan.py` is the
freeze. A side experiment to NEURONRW (closed at `29bc4c43`), approved by the user. NEURONRW code paths are not
modified; the job imports `infra/pg/neuron_rw_core.py`.

## 1. Question

NEURONRW Arm 2 pushed each pole from prompts with no domain ("Here is some text:\n"), so a code neuron had to drag
prose into code. Here each pole gets a prompt in its own domain and nothing more. One prediction is registered (§7);
everything else is descriptive (samples, §9).

Refuting observation: for a code neuron, at d* (§7) the 95% bootstrap interval of push presence minus control
presence lies below 0.15.

## 2. Already-done audit

- NEURONRW Arm 2 (`studies/neuronrw/plan/arm2_<M>.json`; banks `results/neurons/rw/arm2_<m>/1/out/samples.jsonl`):
  120 poles per model, five neutral prompts, doses Q4B 2, 4, 32 and G4, O7 2, 4, 16, 10 samples, temperature 0.7,
  128 new tokens. Same push, same seeds rule, same weights as here.
- NEURONRW Arm 1 (`studies/neuronrw/RESULTS.md` §5; `results/verdict_arm1_{Q4B,G4}.json`): the six code neurons on
  prompts that already contain their construct. All labels CANCELLED under P8; curves exploratory. React useEffect
  presence (200 samples per point): 0 → 0.81, +4 → 0.66, +8 → 0.52, +16 → 0.02; matched control at +16 0.37. Q4B
  paired S_t − S_c at dose 64: FOREIGN KEY 0.096 [0.020, 0.193], Angular 0.397 [0.228, 0.577], HTML title 0.390
  [0.193, 0.604]. The Arm 1 prompts carried the construct; here the code prompt names no language.
- Curated samples: `scratchpad/curate/BEST.md` (top 25 poles from the Arm 2 banks, Q4B and G4).
  `scratchpad/curate2/BEST_O7_ARM1.md` did not exist when this run started, so no Olmo picks are added.
- Literature: not searched for this side run. The registered prediction is an in-repo check of the push on code
  neurons; the rest is a sample listing.

## 3. Poles and domain assignment

Each pole is assigned one domain from its re-explained label (`studies/neuronrw/results/reexplain_<M>.jsonl`).
Source: page = the 12 poles on the user's page; arm1 = NEURONRW Arm 1 target (TeX was dropped, so six, React among
them); best = `curate/BEST.md` top 25.

| model | pole | label (short) | source | domain |
|---|---|---|---|---|
| Q4B | L32N6216+ | verb of making a writing plan (outline, thesis) | page | essay |
| Q4B | L32N9698- | before a serious illness is named | page | news |
| Q4B | L23N1238- | before "vice versa" / "the other way around" | page | essay |
| Q4B | L28N5262+ | after "failed to" / "unable to" in reports | page | news |
| Q4B | L11N5238+ | SQL FOREIGN KEY | arm1 | code |
| Q4B | L27N2924- | Angular | arm1 | code |
| Q4B | L08N3398- | HTML title attribute | arm1 | code |
| Q4B | L12N7367- | flat surface, UV unwrap, unrolled | best | essay |
| Q4B | L09N1795+ | statistics terms (crosstab, exploratory) | best | essay |
| Q4B | L07N4960- | after a named official (met with, attended) | best | news |
| Q4B | L26N4718- | head noun in web boilerplate and form text | best | email |
| Q4B | L31N3418+ | before "property" in real-estate text | best | email |
| Q4B | L34N2004- | "In a surprising development" | best | news |
| Q4B | L10N8041- | CJK administrative unit suffix | best | essay |
| Q4B | L24N6912- | before a foreign title in English prose | best | essay |
| Q4B | L24N3091+ | opens an infinitive clause (how to, able to) | best | essay |
| Q4B | L29N9473- | motion verb before "to" | best | story |
| Q4B | L31N5699+ | before "hold/held" a view | best | essay |
| G4 | L11N1220+ | forward-looking-statement boilerplate | page | news |
| G4 | L15N9089+ | "business" in manufacturer boilerplate | page | email |
| G4 | L28N5924+ | before "India" | page | news |
| G4 | L07N10202- | React useEffect | page, arm1 | code |
| G4 | L07N10202+ | "class" as a cohort (Class of 2018) | page | news |
| G4 | L06N599+ | Python import ... as | arm1 | code |
| G4 | L08N7823+ | SQL REFERENCES | arm1 | code |
| G4 | L18N3736- | "in my opinion" hedge | best | essay |
| G4 | L15N2237- | British royals, tell-all book | best | news |
| G4 | L04N7359+ | "amateur" | best | news |
| G4 | L03N10205+ | "let it go" | best | story |
| G4 | L16N2951- | packaging and safety-data-sheet text | best | email |
| G4 | L24N1919+ | before "plate" (license plate) | best | news |
| G4 | L27N10083- | before "write/wrote" | best | story |
| G4 | L21N6835- | participle before "before/prior to" a date | best | essay |
| G4 | L04N8650+ | clothing word (jersey, shirt) | best | email |
| G4 | L04N4329+ | "bikini" | best | story |
| G4 | L26N9472- | before a group noun ("us regular folk") | best | essay |
| G4 | L14N6393+ | gun violence (shooting, gun) | best | news |
| G4 | L01N4035- | British planning text (amenities, A34) | best | news |
| G4 | L15N2237+ | stock analyst ratings | best | news |
| O7 | L26N6019+ | before broadcast / spectator / game words | page | news |
| O7 | L07N7004- | medical-education and journal terms | page | essay |
| O7 | L03N7540+ | inside a recognized proper name | page | news |

42 poles (Q4B 18, G4 21, O7 3). Random controls for the six code neurons, same layer and sign, drawn by
`realprompt_plan.draw_control` (numpy `default_rng([20261025, layer, neuron])`, redrawn while equal to the target;
MLP widths Q4B 9728, G4 10240 from the banked `config.json`, asserted by the job):

| code neuron | control |
|---|---|
| Q4B L11N5238+ SQL FOREIGN KEY | L11N3497+ |
| Q4B L27N2924- Angular | L27N7222- |
| Q4B L08N3398- HTML title | L08N3836- |
| G4 L07N10202- React useEffect | L07N2158- |
| G4 L06N599+ Python import ... as | L06N3187+ |
| G4 L08N7823+ SQL REFERENCES | L08N6357+ |

## 4. Prompts

Two per domain: [0] the user's domain prompt, [1] a second in-domain opener written here. BOS prepended when the
tokenizer defines one (G4, O7), none for Q4B (`neuron_rw_core.prompt_ids`). Seed prompt index = 2 × domain position
+ j, the same in every model.

| domain | [0] | [1] |
|---|---|---|
| code | `` Here is the code:\n``` `` | `` Here is the program:\n``` `` |
| email | `Here is the email:\n` | `Here is the email:\n\nSubject:` |
| news | `Here is the news report:\n` | `Here is the news report:\n\nHeadline:` |
| essay | `Here is the essay:\n` | `Here is the essay:\n\nTitle:` |
| story | `Here is the story:\n` | `Here is the story:\n\nChapter 1\n\n` |

## 5. Push, doses, sampling

- Push: `neuron_rw_core.ResidualSteer` at the pole's layer output, direction `pole_direction` = sign · W_down[:, i] /
  ‖W_down[:, i]‖ (fp32), alpha_abs = dose × that layer's residual RMS from the Arm 0 `rms.json`
  (`results/neurons/rw/arm0_q4b/1`, `arm0_g4/1`, `arm0_o7/3`), added at every position from P − 1 on.
- Doses: Q4B {0, 4, 8, 16, 32}; G4 and O7 {0, 2, 4, 8, 16}.
- 10 samples per pole × prompt × positive dose. Dose 0 is unsteered and does not depend on the pole, so it is
  generated once per prompt (10 samples) and shared by every pole of that prompt.
- Sampling as Arm 2: temperature 0.7, top_k 0, top_p 1.0, repetition penalty 1.0, 128 new tokens, stop at EOS.
- Rows per prompt: 10 unsteered, then each direction of the domain (poles, then controls) × positive doses × 10, in
  plan order, chunked into generate calls of at most 160 rows; `torch.manual_seed(20261025 + 1000 × prompt index +
  c)` before call c of a prompt. Rows: Q4B 1,780, G4 2,020, O7 280.
- Every sample row stores the prompt, the decoded completion, its first 60 and first 64 tokens decoded, the token
  ids, the dose and alpha_abs, the seed, and the realized cosine and norm ratio at decode step 1.

## 6. Code rule and construct regexes

- Code rule (frozen): a completion is code when it contains a closing fence (a newline followed by ```; the
  opening fence is the prompt's) or at least 8 characters from `{ } [ ] ( ) < > ; =` in the decoded text of its
  first 64 tokens. Written examples (dry run): a Python import block that closes its fence is code; a CREATE TABLE
  line is code; a prose sentence with one parenthesis and one semicolon is not.
- Construct presence: the NEURONOFF SPEC §5.4 regexes, verbatim (`realprompt_plan.CODE`); a match on
  `prompt[-200:] + completion` that ends inside the completion (`neuron_rw_analysis.construct_count`). NEURONOFF
  checked each pattern on 20 written pairs before freeze (20 of 20 matched per pole).

## 7. Registered prediction

For each code neuron n: d* = the smallest positive dose (Q4B 4, 8, 16, 32; G4 2, 4, 8, 16) at which at least 80% of
n's 20 push samples are code by the rule. The brief's "gentlest dose" is read literally. At d*:

gap = presence rate under the push of n − presence rate under n's control, over the 20 (prompt, sample) units,
paired by unit. Interval: 95% percentile bootstrap, 10,000 resamples of the 20 units, numpy `default_rng(20261025)`.

Prediction: gap ≥ 0.15 for React useEffect, and the same bar for SQL FOREIGN KEY, Angular, HTML title, Python import
... as, and SQL REFERENCES.

| verdict | rule |
|---|---|
| REFUTED | interval upper end < 0.15 |
| SUPPORTED | gap ≥ 0.15 and interval lower end > 0 |
| NOT DECIDED | neither |
| NOT SCORABLE | a model gate failed (§8); or no positive dose keeps 80% code; or control presence at d* > 0.85 (no room for the bar) |

Room to fire: when both presences are 0 the interval is [0, 0] and REFUTED fires; a planted 12-of-20 push against
0-of-20 control reads SUPPORTED; a control planted at 18 of 20 reads NOT SCORABLE (dry run, §10). With 20 units a
true gap near 0.15 will usually read NOT DECIDED; that is the power of this run, stated before it.

Prior observation, not a change to the bar: on construct prompts Arm 1 saw React presence fall under the positive
push (0.81 → 0.66 at +4).

## 8. Instrument checks (model gates)

| check | pass | registered failing input | if it fails |
|---|---|---|---|
| identity | logits with the hook at dose 0 equal the unhooked logits exactly (max abs diff 0) on the model's first prompt and first pole | dose 1 must differ (max abs diff > 0) | job aborts (`aborted_gate`) |
| realized change | per (direction, dose) cell, median over its rows of the realized cosine at decode step 1 ≥ 0.95 and median norm ratio in [0.9, 1.1] (G-gen's bounds), every cell | the hook keyed per forward (never steers a decode step), top dose, 4 rows: step-1 cosine must be < 0.95 | failing input that does not fire: job aborts; a failing cell: every verdict of the model NOT SCORABLE |
| code-rule controls | dose-0 code-prompt samples ≥ 0.80 code; dose-0 prose-prompt samples ≤ 0.20 code | — (two-sided bounds on the run's own unsteered samples) | every verdict of the model NOT SCORABLE |
| two-instance token identity | not run: one instance per model | | |

## 9. Descriptive outputs (not tests)

- Presence table per code neuron and dose: push, control and dose 0, with intervals and code shares.
- For every pole and positive dose: the two best samples, first 60 tokens verbatim, beside the unsteered sample of
  the same prompt and sample index. Best = fluent first (distinct-bigram ratio ≥ 0.5 over the whole completion,
  `neuron_rw_analysis.distinct_bigram_ratio`), then most matches of the pole's theme regex, then the higher ratio,
  then prompt and sample index. Samples below 0.5 are flagged as loops.
- Theme regexes (`realprompt_plan.POLES`) were written from each label before any sample of this run was read; code
  poles use their construct regex. Their counts per dose describe the cells only.

## 10. Runbook

1. Dry run, green before launch: `PYTHONPATH=. python -m infra.aws_pg.jobs.realprompt_dryrun --out <dir>` (real
   plan counts, controls re-drawn, code rule examples; the job on tiny random Qwen3, Gemma3 and Olmo3 models with the
   abort path; the scorer on the job's output and on planted banks).
2. Launch, at most two instances at once, budget 30 minutes each, fresh prefixes:
   ```
   ./infra/aws_pg/pg run infra.aws_pg.jobs.realprompt --args " --model-key Q4B --repo Qwen/Qwen3-4B-Base --tokname qwen3 --rms $B/results/neurons/rw/arm0_q4b/1/out/arm0/rms.json" \
     --s3-prefix results/neurons/x/realprompt/1/q4b --type g6e.2xlarge,g6e.4xlarge --budget-min 30 --weights models--Qwen--Qwen3-4B-Base --detach --json
   ./infra/aws_pg/pg run infra.aws_pg.jobs.realprompt --args " --model-key G4 --repo google/gemma-3-4b-pt --tokname gemma3 --rms $B/results/neurons/rw/arm0_g4/1/out/arm0/rms.json" \
     --s3-prefix results/neurons/x/realprompt/1/g4 --type g6e.2xlarge,g6e.4xlarge --budget-min 30 --weights models--google--gemma-3-4b-pt --detach --json
   ./infra/aws_pg/pg run infra.aws_pg.jobs.realprompt --args " --model-key O7 --repo allenai/Olmo-3-1025-7B --revision stage1-step1413814 --tokname olmo3 --rms $B/results/neurons/rw/arm0_o7/3/out/arm0/rms.json" \
     --s3-prefix results/neurons/x/realprompt/1/o7 --type g6e.2xlarge,g6e.4xlarge --budget-min 30 --no-weights --detach --json
   ```
   Watch with `pg watch`; poll `out/metrics.json` only. A failed attempt gets the next number `<n>`.
3. After `STATUS exit=0`: `aws s3 sync $B/results/neurons/x/realprompt/<n>/<m>/out/ bank/x/realprompt/<M>/`.
4. Score: `python -m infra.pg.realprompt_score --bank bank/x/realprompt --costs studies/x/realprompt/results/costs.json
   --json studies/x/realprompt/results/score.json --report studies/x/realprompt/REPORT.md`.
5. Reaper cleanup and `pg audit` clean.
