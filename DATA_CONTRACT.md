# D1 data contract — files exchanged between the harvest (GPU) and explain/verdict (laptop) sides

Status: draft 2026-10-19, written before any code. Amends `SPEC_D1.md` §2.1 where marked AMEND; the spec is
not frozen, so these amendments are edits, dated here. Every path below is relative to a job's `out/`
(`s3://<s3-bucket>/results/neurons/<wave>/<cell>/<attempt>/out/`). Arrays are `.npz`
(numpy, no pickle). Text records are `.jsonl`. No file is ever read from `job.log`.

## 0. Identifiers

- `model_key` ∈ {`Q4B`, `Q4`, `G4`, `O7`} (SPEC §4.1). `revision` = the resolved commit sha (from the local
  HF-cache snapshot directory name, or the Hub for O7), never a branch name alone; the branch name
  (`stage1-step1000`) is recorded beside it.
- Pole id string: `L{layer:02d}N{neuron}{sign}` with sign `+` (a > 0 pole) or `-` (a < 0 pole),
  e.g. `L12N4051+`. Pole axis in arrays: index 0 = positive pole, 1 = negative pole.
- Global token index `tok`: an int64 index into `tokens.npy` of the corpus for that model's tokenizer.
  `doc_id` and `pos_in_doc` are derived from `doc_offsets.npy` (`doc_id = searchsorted(doc_offsets, tok,
  'right') - 1`). `-1` marks an empty slot.

## 1. Corpus job (`infra/aws_pg/jobs/neurons_corpus.py`, science in `infra/pg/neurons_corpus.py`)

Writes `corpus/`:

- `manifest.json`: `dataset_repo`, `dataset_commit` (HF sha), `n_dirs_total`, `k`, `dirs` (50), `shards`
  (path, bytes_consumed, sha256_of_consumed_bytes, n_docs_selected per band), `tokens_per_shard`
  (default 1,000,000; verification runs use a smaller value and say so), `bands` (see below),
  `tokenizers` {name: {repo, revision, n_tokens, sha256_of_tokens_npy, eos_id, bos_id}},
  `zstandard_version`, `created_utc`.
- `docs.jsonl.gz`: one line per selected document in corpus order: `{"i": int, "shard": str, "band": 1|2|3,
  "text": str}` (gzip, stdlib; the laptop has no zstandard).
- `tok/<tokname>/tokens.npy` int32, documents concatenated in corpus order with the tokenizer's EOS after
  each document; `tok/<tokname>/doc_offsets.npy` int64 of length n_docs + 1 (start of each doc, final end);
  `tok/<tokname>/tokenizer/` from `save_pretrained`. `tokname` ∈ {`olmo3`, `qwen3`, `gemma3`}.
  A model's harvest uses the `tokname` of its own tokenizer (Q4B and Q4 share `qwen3`).

**Corpus order and bands (AMEND §4.2 so that 10M ⊂ 20M ⊂ 50M are prefixes of one token stream).**
Documents are selected with the OLMo-3 tokenizer. For each shard in order, documents are read in file
order and assigned to bands by the shard's running OLMo token count: band 1 = documents that fill the
first 20% of `tokens_per_shard` (200k of 1M), band 2 = the next 20% (to 400k), band 3 = the rest (to 1M).
A document that straddles a boundary belongs to the band where it starts; the band's budget is exceeded by
at most one document. Corpus order = all band-1 documents (shard order, then file order), then all band-2,
then all band-3. So the 10M corpus is band 1, the 20M corpus is bands 1–2, the 50M corpus is bands 1–3,
and each is a prefix of the stream under every tokenizer. `manifest.bands` records the document index
where each band starts and the token index where it starts under each tokenizer.

## 2. Harvest job (`infra/aws_pg/jobs/neurons_harvest.py`, science in `infra/pg/neurons_harvest.py`)

One checkpoint per invocation. Arguments: `--model-key --repo --revision --corpus <local or s3 path to
corpus/> --tokname --max-tokens N --snapshot-bands 1,2 --poles poles.json --positions positions.npz
--probe-texts texts.jsonl --sample-neurons 300 --seed 0 --wikitext-parity 2.2444 --batch 8 --window 2048
--out`. Model loading through `AutoModelForCausalLM` (for `G4`, the text tower of
`Gemma3ForConditionalGeneration`; the loaded MLP module's `act_fn` class name is logged). bf16, SDPA,
`inference_mode`. BOS prepended to every window when the tokenizer defines one; BOS positions are excluded
from all statistics.

Hooks: forward hooks on each layer's `mlp.gate_proj` and `mlp.up_proj` capture `g` and `u`;
`a = act_fn(g) * u` is recomputed; a forward pre-hook on `mlp.down_proj` captures the model's own input
to `down_proj`. **G0** compares them: per layer `max|recomputed − actual| / max|actual|` over the first 8
windows; pass iff ≤ 1e-2 at every layer. G0 runs at the start of every harvest job and a failure aborts
the job (fail-loud, no harvest written).

Writes:

- `manifest.json`: `model_key`, `repo`, `revision`, `branch`, `transformers`, `torch`, `act_fn`,
  `n_layers`, `width`, `hidden`, `corpus_manifest_sha256`, `tokname`, `tokens_sha256`, `n_tokens_scored`
  (BOS excluded), `n_windows`, `batch`, `window`, `tokens_per_second` (forward + hooks, wall clock),
  `started_utc`, `finished_utc`, `args`, and `topk_K`, `hist_edges` spec.
- `controls.json`: `G0` {per-layer `ratio`, `pass`}, `G1` {`wikitext_mean_nll`, `banked`, `tolerance`,
  `pass`} present only when `--wikitext-parity` is given (Q4), `nll_corpus_first_120k` (mean next-token
  NLL on the first 120k scored tokens, every model). (AMEND: replaces `pass_through.json` + `nll.json`.)
- `metrics.json`: rewritten every 50 batches and at the end: `n_tokens_scored`, `tokens_per_second`,
  `batches_done`, `complete: bool`, `G0_pass`, `G1_pass_or_null`, `dead_poles` (max_abs == 0), per-layer
  `mean_max_abs`. A reader checks `complete` before using anything else.
- `topk/layer_{l:02d}.npz`: `tok` int64 [N, 2, K], `a` f32, `g` f32, `u` f32, K = 50 (`-1` / NaN when a
  pole has fewer than K nonzero tokens). Positive pole sorted by `a` descending; negative pole by `a`
  ascending. Exact (running top-K merged per batch on the GPU), not sampled.
- `stats/layer_{l:02d}.npz`: `max_abs` f32 [N, 2]; `hist` int32 [N, 2, B] counts of log10|a| over fixed
  edges `hist_edges` f32 [B + 1] from 1e-6 to 1e4, B = 400 (bin ratio 10^(10/400) ≈ 1.059; `|a| < 1e-6`
  counts in bin 0, `|a| ≥ 1e4` in bin B−1); `n_pole_tokens` int64 [N, 2] (tokens with a > 0 / a < 0);
  `q99`, `q999`, `q9999` f32 [N, 2] = upper edge of the bin holding that quantile of |a| over the pole's
  tokens (resolution 5.9%, stated in every figure that uses them); `density` int64 [N, 2] = count of
  tokens with |a| ≥ 0.01·max_abs (from the histogram, same resolution); `top_same_token_frac` f32 [N, 2]
  (fraction of the top-K that is the most common token id among them); `sign_counts` int16 [N, 2, 4]
  over the top-K for (g ≥ 0, u ≥ 0), (g ≥ 0, u < 0), (g < 0, u ≥ 0), (g < 0, u < 0); the double-negative
  share of a pole is `sign_counts[..., 3] / K`.
- `snapshots/band{b}/topk/…` and `snapshots/band{b}/stats/…`: the same two directories frozen at the end
  of band b when `--snapshot-bands` names it (so one 50M run yields the 10M and 20M readings). (AMEND:
  replaces the separate 10M/20M harvests for the final checkpoint.)
- `poles/` only when `--poles` is given (`poles.json` = list of pole id strings, copied into `out/`):
  - `exemplar_a.npz` when `--positions` is given: `positions.npz` holds `tok` int64 [P, E] (E positions
    per pole, −1 padded); output `a`, `g`, `u` f32 [P, E] at exactly those token positions at this
    checkpoint. (AMEND: replaces `sparse/`; exemplar sets are fixed positions, so presence at any
    checkpoint is read here without a second forward.)
  - `top1000.npz`: `tok` int64 [P, 1000], `a` f32 — exact running top-1000 by |a| in the pole's sign.
  - `vocab.json`: per pole the top-10 promoted and top-10 suppressed vocabulary items by
    `W_U @ (gamma_final ⊙ W_down[:, i])` (final norm gain folded in; stated), as `{pole: {"promoted":
    [[token_str, score]…], "suppressed": […]}}`. Same for every pole in `--poles`.
  - `probe/scores.npz` when `--probe-texts` is given (`texts.jsonl`: `{"text_id": str, "text": str}`):
    `text_ids` str array [T], `max_a` f32 [T, P], `min_a` f32 [T, P], `argmax_tok` int32 [T, P],
    `argmin_tok` int32 [T, P] (positions within the text, BOS excluded), `n_tokens` int32 [T];
    `probe/tokens.json` `{text_id: [token strings]}`. Each text is one forward of its own, BOS prepended.
- `samples/contexts.jsonl`: for every pole in `--poles`, or for `--sample-neurons` neurons drawn uniformly
  with `--seed` stratified over layers (both poles), one line: `{"pole": str, "max_abs": f, "q99": f,
  "q999": f, "q9999": f, "density": int, "sign_counts": [4], "topk": [{"rank": int, "tok": int, "doc": int,
  "pos": int, "a": f, "g": f, "u": f, "before": str, "token": str, "after": str}, …]}` with `before` = the
  64 tokens before, `after` = the 16 after, decoded with the model's own tokenizer, document-bounded.
  `samples/first_batch.json`: 10 contexts for 3 poles from the first batch with raw values (SPEC §9.1).
  `samples/sample_draw.json`: the seed and the drawn neuron list.

## 3. Explain side (`infra/pg/neurons_explain.py`, `infra/pg/neurons_llm.py`), laptop, no GPU

Reads `topk/`, `stats/`, `poles/vocab.json`, `samples/contexts.jsonl` and `corpus/tok/<tokname>/` fetched
from S3 into a local `bank/` directory (banked arrays, allowed on the laptop). Renders contexts from
`tokens.npy` with the saved tokenizer when a pole is not in `samples/`.

LLM calls go through `neurons_llm.py` with one request/response format and three backends:
`anthropic` (Messages API), `openai_compat` (OpenRouter and the first-party open-weights hosts; provider
and quantization pinned in the request), and `manual` (requests are written to a file and answered by a
Claude Code subagent; this is how every verification run today is done). The pipeline never knows which
backend answered beyond the logged `model` string.

- `explain/requests/<batch>.jsonl`: `{"id": str, "role": "explainer"|"second_reader"|"writer"|
  "writer_job"|"clusterer"|"judge", "model": str, "temperature": f, "max_tokens": int, "system": str,
  "user": str, "schema": {json schema of the expected JSON answer}}`. `id` = `<role>:<pole or cluster
  id>:<sample index>`.
- `explain/responses/<batch>.jsonl`: `{"id": str, "model": str, "text": str, "utc": str}`; `text` must be
  one JSON object valid under `schema`; invalid or missing answers fail the collect step loudly, listing
  the ids.
- `explain/calls.jsonl`: one line per request/response pair: `id, role, model, backend, temperature,
  prompt_sha256, response_sha256, utc`.
- Prompts are files under `infra/pg/neurons_prompts/<role>.md` with `{placeholders}`; their SHA-256 goes
  into `explain/manifest.json`.
- `explain/explanations.jsonl` (Agent 1 and 3): `{"pole", "sample", "explanation", "jobs": [{"name",
  "related_to_others": bool, "reasoning"}], "class": "ONE"|"TWO-REL"|"UNREL"|"UNK", "confidence": f}`.
- `explain/texts.jsonl` (Agent 2; the GPU probe input): `{"text_id": "<pole>:<kind>:<k>", "pole": str,
  "kind": "pos"|"near"|"distant"|"max"|"job_pos", "job": int|null, "text": str}`.
- `explain/shuffle.json`: for every pole the same-layer shuffle pole drawn with a logged seed (its texts
  are also scored at that pole; `texts.jsonl` carries both poles in `--poles`).
- Q2: `explain/clusters.jsonl` `{"pole", "sample", "clusters": [{"c": int, "name": str, "exemplar_ranks":
  [ints into the T_15 top-50]}], "pairs": [[c1, c2, "related"|"unrelated"]]}`;
  `explain/judgments.jsonl` `{"cluster": "<pole>:c<idx>", "decoy_type": "random"|"hard", "decoy_pole",
  "t_before", "sample", "shown_order": ["true","decoy"] shuffled, "answer": "A"|"B", "correct": bool}`;
  positive controls carry `"control": true`.

- 2026-10-21: a response row written by a Claude Code answering agent into a batch an API backend also answered
  carries a fifth key `backend: "manual"`; such a batch is collected with `--backend mixed`, and each `calls.jsonl`
  row's `backend` is the one that answered that id (explain ae1a5e4e).

## 4. Verdict (`infra/pg/neurons_verdict.py`), laptop

Reads only banked files from §2 and §3 plus `probe/scores.npz`. Refuses to run (non-zero exit, message
naming the file) unless `studies/already-clean/JOURNAL.md` contains a `## PREDICTIONS` entry whose date
precedes the `started_utc` of every confirmation artifact it is given. Writes `verdict/metrics.json`,
`verdict/grid.md` (SPEC §11 filled), `verdict/samples/` (one passing and one failing pole per class and
model). Scores: `score(text, pole) = max_a / max_abs` (positive pole) or `min_a / (−max_abs)` (negative
pole), from `probe/scores.npz` and `stats/`. θ_s, classes, `p_u`, `b`, `v`, H_MIG exactly as SPEC §6–7.

## 5. Scale of the verification runs (today)

`tokens_per_shard = 60,000` (3M-token corpus; bands 12k / 24k / 60k per shard), `--max-tokens 2,000,000`
per checkpoint for Q4, Q4B, G4; 1,000,000 for the two O7 revisions; `--sample-neurons 10`; ten poles
explained through the manual backend; one scoring job. Every gate is exercised with an input that makes it
fail (G0 with a deliberately wrong `act_fn`; G1 with a wrong banked value; the PREDICTIONS refusal; a
malformed LLM answer) and the failure is recorded in `studies/already-clean/VERIFY.md`.

## 6. Additions after the explain-side build and review (2026-10-19, matching branch `neurons/explain` c3b82d32)

- Request ids: explainer / second_reader / writer `<role>:<pole>:<sample>`; `writer_job:<pole>:j<k>`; clusterer and
  judge ids are opaque `<role>:<batch>:<int>` from a per-batch random permutation; the mapping and the per-cluster
  request count live in `explain/plans/<batch>-clusterer.jsonl` and `explain/plans/<batch>-judge.jsonl`, never in
  the request.
- `explain/responses/<batch>.backend.json`: backend name, model string, utc, for the whole batch.
- `explain/clusters.jsonl` clusters carry `echo: bool` (the one cluster holding exactly the contexts with g < 0 and
  u < 0; excluded from m, main, births, the parent test and every count).
- `explain/judgments.jsonl` rows carry `id`, `draw`, `control` on every row.
- `explain/probe_poles.json` (the pole list the scoring job takes as `--poles`), `explain/positions.npz` and
  `explain/positions_poles.json` (the T_15 top-50 positions of S, the input W3 takes as `--positions`).
- `verdict/theta.json` and `verdict/judge_gate.json` with `created_utc` and the SHA-256 of their inputs;
  `verdict/samples/q1_<model>_poles.jsonl`, `verdict/samples/q2_clusters.jsonl`.
- The explain side reads these harvest `manifest.json` fields and nothing else from it: `tokname`, `n_layers`,
  `n_tokens_scored`, `tokens_sha256`, `revision`, `branch`, `started_utc`.
- `poles/vocab.json` is computed once per neuron in the positive write direction `W_U @ (gamma ⊙ W_down[:, i])`; **(2026-10-24 clarification, F41:** the file holds one entry per pole; the "-" entry is the "+" entry with promoted and suppressed swapped and signs negated; readers must not flip it again.)
  for the negative pole the explain side swaps promoted and suppressed (the negative pole writes `−W_down[:, i]`).
- Judge records exclude every context whose activating token lies in the same document within 64 tokens before or
  16 after any exemplar of the cluster (the rendered window), not only exact exemplar positions.
- B4 operationalized: a cluster from clustering sample 1 enters the primary count only if some non-echo cluster
  in sample 2 holds a strict majority of its exemplars; the exclusion rate is reported.
- Open until the first live API call: whether the response's `model` field equals the requested id (logged and
  asserted at that point).

## 7. After the W0 harvest run (2026-10-19, branches `neurons/harvest` 042fac19 and `neurons/explain` e092cd81)

- `poles/poles.json` is where the harvest writes the pole list and where the explain side reads it
  (`HARVEST_FILES["poles"]`, changed from `poles.json` at `out/`).
- The T_15 exemplar record. `grid_cluster_tables` reads `poles/exemplar_a.npz` from every grid directory,
  T_15 included, and its columns are the T_15 top-50 ranks. T_15 cannot take `--positions` (its top-50 is
  known only after its own harvest), so the harvest job, when `--poles` is given and no `--positions` or
  `--positions-from`, writes `poles/positions.npz` (`tok` int64 [P, 50]: the pole's own top-50 token
  positions, rank order, −1 padded) and `poles/exemplar_a.npz` (`a`, `g`, `u` there, which equal the
  top-50 values). `--positions-from <dir>` takes `<dir>/poles/positions.npz` when it exists, else
  `<dir>/topk/` top-50; it no longer takes `top1000.npz`, so every grid directory has E = 50 and the same
  columns. `neurons_explain positions` must reproduce T_15's `poles/positions.npz` from `topk/`, and
  `check_positions` fails otherwise.
- `check_grid` requires exactly the 16 grid directories in order (`manifest.branch` = `stage1-step<n>`,
  equal `tokens_sha256` and `n_tokens_scored`). A two-point grid cannot exercise the W3 path, so the W0
  Q2 verification is a 16-point grid at 524,288 tokens per checkpoint on the W0 3M corpus.
- One run cannot serve both pole lists: the grid directories' `poles/poles.json` must equal the Q2 set S
  (`positions_poles.json`), while the Q1 scoring directory's must equal `probe_poles.json` (S ∪ shuffle
  poles). The Q1 scoring is therefore its own step (`--poles probe_poles.json --probe-texts texts.jsonl`)
  on the final checkpoint, and `neurons_verdict theta --stats <T_15 grid dir> --probe <that step>`.
- W0 Q1 explain loop runs on O7 at `stage1-step1413814` (the only W0 directory with `poles/vocab.json`;
  the Q4B step was run without `--poles`), with the 20 poles of its `samples/sample_draw.json`.

## 8. W1 additions (2026-10-20)

- `stats/layer_{l:02d}.npz` gains `pos0_share` float32 [N, 2]: the pre-exclusion share of each pole's top-50
  that sat at window position 0 (SPEC §4.2 W0 amendment); window position 0 is excluded from `topk/` and
  `stats/` otherwise. The verdict banks `pos0_share` per pole in the Q1 records.
- With `--sample-neurons` and no `--poles`, the harvest writes `poles/` for the sampled poles (SPEC §15 item 10).
- Corpus manifest gains `n_duplicates_skipped` per directory (dedupe by text SHA-256 before selection).
- The explain side renders every context from `tok/<tokname>/tokens.npy` (256 before, 16 after for the
  explainer and clusterer; 64 before for the judge's records); `samples/contexts.jsonl` is a harvest-side
  check only. The corpus pull must include `tok/<tokname>/tokenizer/`.
- Backend: `openai_compat` against DeepInfra (`https://api.deepinfra.com/v1/openai`), key from the environment
  variable named in `infra/pg/settings/deepinfra.json`, every role on `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
  temperature 1.0, `response_format` json_schema; the sidecar records the reported model id and
  `responses/<batch>.usage.jsonl` the latency and token counts. First live call 2026-10-20: reported model id
  equals the requested id; `max_tokens` 400 truncates a writer answer, the roles' limits (§2.2) stand.

## 9. W2-D additions (2026-10-20, branch `neurons/explain` 1e0c7059 → 84a493a0)

- `explain reject --out --batch --roles --backend [--max-attempts 3]`: applies the same validation as that
  role's collect (schema; `leak_errors` for explainer and second reader; the clusterer's partition and echo
  checks; truncated or invalid JSON for every role) and moves each refused response line from
  `responses/<batch>.jsonl` to `responses/<batch>.rejected.jsonl` with `reason` and `utc`; usage lines stay.
  `llm run` then re-asks only the ids without a response. Loop: collect → reject → run → collect, at most
  `--max-attempts` rejections per id, then it raises and names the id.
- `leak_errors` (52d90981) refuses only: the ⟦ ⟧ marker; an activation value (`\b[agu]\s*=\s*-?\d`); a
  reference to a shown context (`contexts? #?\d+`, `#\d+`, `contexts? (above|shown|provided|given|listed|
  supplied|I saw|I was shown)`), case-insensitive. The W0 bare-digit and bare-"context" patterns are gone.
- `verdict discovery --theta --stats --probe --explain --out` (b5554cf1) writes `<out>/verdict/discovery.json`
  (`set: "discovery"`, θ_s and its created_utc, §6.1 pass rate with Wilson 95%, class fractions from
  `q1_metrics`, strong rates, X3, X7, per-role prompt SHA-256 from the requests, model id and backend from the
  sidecars, request/response/rejected counts per batch) and `discovery_samples.jsonl` (per-pole records).
  It applies no blind-protocol timing gate and refuses to overwrite. It is the W2-D prompt-version report.
- X7 pre-pass (84a493a0): a `decoded:` line needs at least four characters with at least 90% ASCII letters,
  digits, spaces or common punctuation; script names come from Unicode block ranges for CJK, Cyrillic, Greek,
  Arabic, Hebrew, Devanagari, Bengali, Thai, Hangul and Japanese kana only. The 240 `d1` answers were produced
  with the old pre-pass (15 false `decoded:` lines); their X7 split is not read.
- Settings files: `infra/pg/settings/deepinfra_opus_explainer.json` and `deepinfra_sonnet_explainer.json`
  (explainer and second reader on `anthropic/claude-opus-5-5` or `anthropic/claude-sonnet-5-5`, every other
  role on `deepseek-ai/DeepSeek-V4.1-Flash-1017`; same base URL, key variable, temperature 1.0, json_schema).
  A run refuses a batch whose requests name a model the settings do not. `prepare --model` names the explainer
  and second reader; `prepare-writer --model` names the writer. DeepInfra's usage object carries
  `estimated_cost` (USD) per call, kept in `responses/<batch>.usage.jsonl`.
- Scoring job for a discovery round: `results/neurons/w2d/q4b_probe/<n>/` with `texts.jsonl` and
  `probe_poles.json` uploaded next to `steps.json`; the step is `neurons_w2d_q4b_probe.json` (harvest f0eb3b9c),
  Q4B on the W1 corpus at 131,072 tokens with `--poles` and `--probe-texts`; `out/probe/poles/probe/scores.npz`
  is what `verdict theta` and `verdict discovery` read.
- Probe rule (harvest 62d865d3, W2-D amendment of SPEC §2.2): `max_a`, `min_a`, `argmax_tok`, `argmin_tok` in
  `poles/probe/scores.npz` are taken over text positions ≥ 1 for every model; `argmax_tok`/`argmin_tok` index the
  text's own tokens (always ≥ 1); `n_tokens` is the full count; a text under two tokens fails the job with its
  `text_id` in the message. `scores.npz` carries `probe_rule` (string array equal to `neurons_harvest.PROBE_RULE`);
  the probe job's manifest carries `probe_excluded_positions: "text position 0"` and `probe_rule` (null for jobs
  without probe texts). Probe outputs before this commit (`w2d/q4b_probe/1..3`) lack the field and are refused by
  the verdict from the matching explain commit on. `pg run` ships the harvest worktree's working tree (tar of
  `infra/`, `configs/`, `scripts/`); `pg_git_sha` is HEAD plus `-dirty` when tracked files differ.
- W2-C probe manifests (harvest d5b8177e): `neurons_w2c_{q4b,g4,o7}_probe.json`, prefixes
  `results/neurons/w2c/<cell>_probe/1/`; weights `models--Qwen--Qwen3-4B-Base`, `models--google--gemma-3-4b-pt`,
  O7 downloaded in-job (`--no-weights`; 14.6 GB in about 2 minutes); budget 90 min each (≈ 25 min expected).
- Declined answers (explain 116cd159): a `content_filter` or `refusal` finish reason is recorded as an empty
  response with the finish reason in the usage line; collect refuses it, reject moves it, run re-asks; at
  `--max-attempts` reject appends the id, role and last reason to `responses/<batch>.declined.jsonl` and run never
  asks it again. A declined explainer or writer makes the pole UNK-N with the role and reason; a declined writer
  leaves the pole without texts and out of `probe_poles.json`; the shuffle is drawn over every pole that reached
  the writer step; a declined second reader is left out of the kappa pairs. Collect prints declined ids;
  `discovery.json` counts `declined` per batch; the unblind samples carry the reason.
- Texts under two tokens (explain 6531fad1): `collect-writer` needs `--corpus` and tokenizes every text with the
  harvest's tokenizer (`tok/<tokname>/tokenizer/`); an answer with a text under two tokens is refused with the
  `text_id`; `reject` needs `--harvest` and `--corpus` when the roles include the writer.
- Verdict refusal (explain 53cc6515): `theta`, `discovery` and `unblind` refuse a probe directory unless
  `scores.npz` carries `probe_rule` and the manifest says `probe_excluded_positions == "text position 0"`; both
  strings are recorded in `theta.json`, `discovery.json` and the unblind Q1 inputs.
- Vast lane (harvest e4278745, 95c6c6db): `infra/vast/cells/pipeline_job.py` runs `infra.aws_pg.jobs.pipeline` on the
  cell's `steps_s3` with `--out` the cell directory and `PG_GIT_SHA` from the cell (`pg_git_sha`, the commit whose
  `infra/` ships; the controller archives HEAD). Manifest `neurons_w2d_q4b_probe_r5r4.json`: image `shackleton:mdl-v1`,
  weights from `s3://…/weights/hf/` (`weights_mode: s3`), env `HF_HUB_CACHE=/workspace/hf/hub`. The box's IAM user reads
  and writes `results/mdl/*` only, so inputs are mirrored (server-side copies) to `results/mdl/neurons/<same path>` and
  the step arguments name the mirror; cell outputs land at `results/mdl/mendeleev_<rental>/cells/<cell>/` and are
  copied back to the canonical `results/neurons/<wave>/<cell>/<attempt>/out/` with the rental's `job.log`,
  `timings.tsv`, `weights_source.json`, `manifest.json` and `_rental_complete.json` under `…/<attempt>/vast/`. A vast
  run has no `STATUS` file; completion is the cell's `_done.json` (sha256 and byte count per artifact) and the rental's
  `_rental_complete.json`. `run_cells` disables TF32 (no effect on the bf16 forward). The probe manifests of rounds 4
  and 5 carry `pg_git_sha: e4278745` and `device_name` NVIDIA L40S.
- Verdict directories for the two like-for-like rounds: `bank/w1/q4b_disc_v3/verdict/` (round 4, prompt v3; the frozen
  `theta.json`) and `bank/w1/q4b_disc_opus_w_r5/verdict/` (round 5, the `q4b_disc_opus_w` explain directory rescored);
  `verdict theta --out <root>` writes `<root>/verdict/theta.json`. Correction to §9 above: the usage lines record
  `prompt_tokens`, `completion_tokens`, `latency_s` and `attempts` per call; no provider cost field is kept, and costs
  in the journal are list prices times those counts.
- W2-C vast manifests (harvest 489c925d): `neurons_w2c_{q4b,g4,o7}_probe.json`, one `pipeline_job` cell `probe` each,
  `steps_s3` at `results/mdl/neurons/w2c/<cell>_probe/1/steps.json` (the AWS step with every input path moved under the
  mirror), `pg_git_sha` 17057bd0; Q4B and G4 weights from the S3 cache (`weights_mode: s3`), O7 in-job hub download
  (`weights_mode: download` with no `weights_hf_repo`, env `HF_ONLINE=1`, `HF_HOME=/workspace/hf`,
  `HF_HUB_CACHE=/workspace/hf/hub`; `hf_online_setup` uses the shipped `<api-key-env-var>` and does not call SSM; the checkpoint is
  deleted after the harvest as on AWS). Disk 60 / 60 / 80 GB; bounds 1.5 billed hours, 30 min load backstop.

## 10. W2-C, W3 and W4 additions (2026-10-20/21, branch `neurons/explain` ac4e00dc → b03cd162)

- `verdict draw-s` (ac4e00dc): S and S_gate from `stats/` only (RARE excluded; the §4.4 class order is vacuous at
  600 of 600); outputs `draw/{S.json,S_gate.json,grid_poles.json,draw.json}` with input sha256s and counts; refuses to
  overwrite and refuses more than 600 non-RARE audit poles. Real draw: `bank/w3/o7/draw/` (600, 40, 640; no RARE pole).
- Opus refusal (92823783): a settings file, request or `prepare*` `--model` naming any Opus form on the `openai_compat` or
  `anthropic` backend is refused before anything is sent or written; the manual backend is exempt; a settings file with
  `"retired"` is refused on load (`deepinfra_opus_explainer_writer.json` is retired). Settings on record:
  `deepinfra_flash_second_reader.json` (six roles on Flash), `deepinfra_sonnet55_q1roles.json` (calibration round 6),
  `deepinfra_w2c_mixed.json` (explainer, writer, job writer Sonnet 5.5; second reader, clusterer, judge Flash; 02660245).
- Mixed-model batches (cb4a7cd0): `explain remodel --roles … --model … [--unanswered-only]` rewrites request models and
  records the operation in the manifest; a run checks models and the Opus rule only on requests it will send; every
  `explanations.jsonl` and `texts.jsonl` row carries the model of its response line; the manifest's `models_by_role`
  counts models per role; `discovery` and `unblind` add `by_explainer_model` and label the kappa with both models.
- Second reader on its own (b310674a): `collect --roles second_reader` after an explainer-only collect pairs each answer
  with the collected explainer answer (sha256-matched through `calls.jsonl`), runs the same checks, appends to
  `explanations.jsonl` and `calls.jsonl`; refuses while an explainer id is neither logged nor declined.
- W2-C state: `bank/w2c/{q4b,g4,o7}` batch c1; explainer Opus 463/455/426 + Sonnet 137/145/214, declined 1/3/0; second
  reader Flash 600/600/640 (G4 one declined); texts 9,714 / 9,681 / 10,404 for 1,197 / 1,174 / 1,263 probe poles.
  Scoring: `results/neurons/w2c/{q4b,o7}_probe/1/` on AWS (`pg_git_sha` cd285387-dirty, the dirt being the vast spend
  file), `g4_probe/1` on vast (manifest `neurons_w2c_g4_probe.json`, copied back on completion).
- W3 grid (harvest bb6c2a68): `results/neurons/w3/grid/1/` job a (`t15_step1413814`, `t00_step0` … `t06_step12000`),
  `grid/1b/` job b (`t07_step20000` … `t14_step1000000`, `--positions` = job a's T_15 `poles/positions.npz` by S3 URI);
  20,570,743 tokens per checkpoint (band-2 end of the W1 corpus), `--snapshot-bands 1`, poles = `grid_poles.json`; both
  on vast, outputs copied back from the rentals' `cells/grid_{a,b}/` to the canonical prefixes.
- W4 tooling (4e765d98, b03cd162): `check_grid` takes the 16 step dirs with T_15 last; the layer cache holds at most 64
  files; a corpus that does not match the harvest is refused; `positions` records the pole list and its sha256;
  `prepare-judge --gate <dir>` is required and refuses an S judge batch before `<dir>/verdict/judge_gate.json` exists;
  the gate records the judge identity (models, prompt sha256) and `unblind` checks it against S; declined clusterer
  samples (`declined_clusterer`) and judgments (`declined_judgments`) are written with reasons and excluded from Q2,
  `p_u`, `a_ctrl` and the flags; the center birth curve is recorded as "not computed: W3 scores T_15 only"
  (`--center-texts` removed). Directories: `bank/w4/o7_gate` (S_gate, seeds 4101/4102), `bank/w4/o7_s` (S, 4103/4104).
- W3 outputs (2026-10-21): `results/neurons/w3/grid/1/out/{t15_step1413814,t00_step0,…,t06_step12000}/` and
  `grid/1b/out/{t07_step20000,…,t14_step1000000}/`, each a full harvest directory (`manifest.json`, `metrics.json`,
  `controls.json`, `poles/`, `stats/`, `topk/`, `snapshots/band_1`, `samples/`); `device_name` NVIDIA L40S for job a and
  NVIDIA RTX PRO 5000 Blackwell for job b; `pg_git_sha` 17057bd0 for both. The local mirror is
  `bank/neurons/w3/grid/{1,1b}/out/`. W4's `check_grid` takes the 16 step directories with T_15 last.
- W3 device check (2026-10-21): `results/neurons/w3/grid/1c/out/t14_step1000000/` is T_14 harvested again on an L40S
  (vast manifest `neurons_w3_grid_c_t14.json`, rental record under `grid/1c/vast/`), same arguments as job b; it is not a
  grid step for W4 (`check_grid` still takes `1/` and `1b/`). The comparison `neurons_device_check` (harvest db47a5d3)
  wrote `results/neurons/w3/device_check/t14_1b_vs_1c.json` (local `bank/neurons/w3/device_check/`): 0 of 640 poles flip
  presence under either reading. The per-cluster reading (`--clusters`) is run once `bank/w4/o7_s_v2/explain/clusters.jsonl`
  exists.
- Round 13 and the frozen-θ_s rescoring (2026-10-21): `bank/w1/q4b_disc_v3_sw` (explain: `fork-writer` of `q4b_disc_v3`,
  writer `anthropic/claude-sonnet-5-5`, prepared from the explain tree at 9a6562ac because the fork records the v1
  clusterer sha; probe `results/neurons/w2d/q4b_probe/13/`); `verdict discovery --theta` may take another round's
  `theta.json`, which it records; the `_frozen` and `_own` roots hold the same explain directory read at two thresholds.
