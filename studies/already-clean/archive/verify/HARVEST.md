# D1 harvest side: W0 verification run (N1, N2, N5)

The pipeline completed at the third attempt: 7 of 7 steps ran, STATUS `exit=0`. G0 passed on every model,
with a max ratio of 0.0065 against the 0.01 tolerance. G1 reproduced the banked B3 value bit for bit
(2.244386524302183) on the model it was banked for, Qwen3-4B-Instruct-2507. It failed on Qwen/Qwen3-4B
(2.5491, gap +0.305), because the spec names the wrong model for G1.

Written 2026-10-19 by the harvest agent. Branch `neurons/harvest` in `../interp_research-neurons-harvest`.
Commits: 5b499928 (code, tests, manifest), 5b629882 (gz shards and next-file rule), 042fac19 (hook lifetime).
Banked artifacts: `../interp_research-neurons-harvest/bank/neurons/w0/verify/3/out/` (the q4 band
snapshots were not pulled), and `s3://pg-lineage-769063704696/results/neurons/w0/verify/3/out/`.

## What ran

| attempt | prefix | instance | type | launched (UTC) | STATUS (UTC) | exit | cause |
|---|---|---|---|---|---|---|---|
| 1 | `results/neurons/w0/verify/1` | i-0df29a9d9e2b02da9 | g6e.2xlarge | 21:14:20 | 21:25:10 | 1 | corpus: `data/olmocr_science_pdfs-entertainment` holds only `.jsonl.gz`; code accepted `.jsonl.zst` only |
| 2 | `results/neurons/w0/verify/2` | i-07686203ab1e64e21 | g6e.4xlarge (no 2xlarge capacity) | 21:32:51 | 21:42:39 | 1 | q4: G1's plain forward hit MLP hooks left registered with no consumer (`TypeError: 'NoneType' object is not callable`) |
| 3 | `results/neurons/w0/verify/3` | i-0255f7c40c2b098b0 | g6e.2xlarge | 21:46:57 | 22:57:14 | 0 | complete |

Attempt 3 steps (`out/pipeline_status.json`: all `ok`), from each step's `manifest.json`:

| step | model, revision | started | finished | wall s |
|---|---|---|---|---|
| corpus | 50 Dolma 3 dirs, 60,000 OLMo tokens per dir | 21:52 | 21:56:26 | about 270 |
| q4 | Qwen/Qwen3-4B `1cfa9a72` | 21:57:20 | 22:12:57 | 938 |
| q4b | Qwen/Qwen3-4B-Base `906bfd4b` | 22:13:05 | 22:28:10 | 906 |
| g4 | google/gemma-3-4b-pt `cc012e0a` | 22:28:20 | 22:35:05 | 405 |
| o7_s1000 | allenai/Olmo-3-1025-7B `stage1-step1000` = `d82e70ec` | 22:37:11 | 22:44:56 | 465 |
| o7_final | allenai/Olmo-3-1025-7B `stage1-step1413814` = `373bad25` | 22:47:03 | 22:54:17 | 434 |
| c0 | Qwen/Qwen3-4B-Instruct-2507 `cdbee75f` (added; see G1) | 22:54:25 | 22:57:08 | 163 |

Steps manifest: `infra/aws_pg/steps/neurons_w0_verify.json`. Probe texts: `infra/aws_pg/steps/neurons_probe_smoke.jsonl`.
Every step ran under transformers 5.9.0, torch 2.12.0+cu130, on an NVIDIA L40S, at git `042fac195a94`.

Launch differences from the brief, each a routine call:
- `--weights` takes HF-cache directory names (`models--Qwen--Qwen3-4B,...`). Repo ids (`Qwen/Qwen3-4B`)
  make `run_job.sh` include `hub/Qwen/Qwen3-4B/*`, which matches no file. The job would then have
  downloaded the weights from the Hub without saying so.
- The weights also included `models--Qwen--Qwen3-4B-Instruct-2507`, for step `c0`.
- `--type g6e.2xlarge,g6e.4xlarge`, because attempt 2 found no 2xlarge capacity in any AZ.
- `--budget-min` was 120 for attempts 1 and 2 and 150 for attempt 3, against the brief's 90.
  Throughput was unmeasured and there was one extra step. The pipeline used 70 minutes, so 90 would have sufficed.

## Gates and whether each could fail

| gate | value | pass | registered failing input and its result |
|---|---|---|---|
| G0 q4 | max ratio 0.00500 (layer 4) | yes | silu replaced by gelu_tanh: 33/36 layers over 0.01, median 0.038 |
| G0 q4b | 0.00545 (layer 3) | yes | 33/36 over, median 0.038 |
| G0 c0 | 0.00650 (layer 5) | yes | 33/36 over, median 0.039 |
| G0 g4 | 0.00435 (layer 20) | yes | gelu_tanh replaced by silu: 31/34 over, median 0.039 |
| G0 o7_s1000 | 0.00382 (layer 28) | yes | 3/32 over, median 0.0057, max 0.017 |
| G0 o7_final | 0.00511 (layer 21) | yes | 32/32 over, median 0.108 |
| G1 q4 (Qwen/Qwen3-4B) | wikitext mean NLL 2.54915, banked 2.2444, gap +0.30475 | no | the banked number belongs to another model; see below |
| G1 c0 (Qwen3-4B-Instruct-2507) | 2.244386524302183, gap -0.0000135 against 2.2444 | yes | offline: `g1_pass(2.2444, 2.2444 + 0.1)` is false |

G0 is `max|act(g) * u - a_model| / max|a_model|` per layer, over the first batch of 8 windows. Here g and u
are the bf16 hook outputs, the recomputation is fp32, and a_model is the input to `down_proj`. Every model
would have failed G0 under the swapped activation, since one layer over tolerance fails the gate. At
stage1-step1000 only 3 of 32 layers moved over tolerance. G0's failing input has little power on early
checkpoints, where |g| is small and silu and gelu_tanh nearly agree.

G1 model defect. SPEC §4.1 and §5 call `Qwen/Qwen3-4B` (sha `1cfa9a72`) "STAGEDIFF C0". That is wrong.
`studies/stagediff/results/w1/bat-C0/1/out/manifest.json` gives repo `Qwen/Qwen3-4B-Instruct-2507`, sha
`cdbee75f`. Qwen/Qwen3-4B is STAGEDIFF's P0, and P0 has no banked B3 number. The B3 number also came from
`text_eval.shell_text_perplexity` through `StageForward(bos_id=None)` (`_stagediff_shared.run_b3`), not
from `clean_text_perplexity`. The banked record itself says "NOT comparable to clean_text_perplexity
values", and `clean_text_perplexity` asserts a BOS, which no Qwen3 tokenizer defines. So G1 here calls
`shell_text_perplexity(StageForward(model, tok, bos_id=None), window=2048, max_tokens=120000)`, which is
B3's exact call. Step `c0` was added so G1 runs on the model the number was banked for. It reproduced the
banked value with the same 119,941 tokens and 59 dropped leading targets. The q4 result stands as
measured: Qwen/Qwen3-4B reads 2.549 on the same protocol.

Offline gate: `uv run --with pytest --with transformers --with zstandard pytest infra/pg/tests/test_neurons_*.py`
passes 14 of 14. It also passes 14 of 14 when pinned to the AMI versions (transformers 5.9.0, torch 2.12.0,
zstandard 0.25.0, numpy 2.4.6). Five deliberately planted bugs were each caught: no merge of the running
top-K, no split at a band boundary, a histogram bin off by one, exemplar positions shifted by one, and a
probe that kept the BOS position. The G1-on-tapped-model test failed before the attempt-2 fix.

## Throughput and the §12 re-derivation

Harvest-loop rate (forward plus hooks, 8 windows of 2048 per batch): Q4 3,990, Q4B 4,023, C0 3,926,
G4 3,778, O7 step1000 3,291, O7 final 3,393 tokens/s. The GPU SM was at 100 % during the loops (`out/dmon.log`).
§12 assumed 15,900 tokens/s for the 4B models and 8,700 for 7B. Measured is 4.0 times and 2.6 times lower.
Writing the arrays (`np.savez_compressed`, one thread, GPU idle) takes 116-156 s for the final arrays and
about 131 s per band snapshot.
O7 Hub download: 14,605,764,269 bytes in 118.5 s (123.2 MB/s) at step1000, and 117.2 s (124.6 MB/s) at the
final revision. Both are above the 50 MB/s pivot. Both snapshots were deleted after their harvest.

Re-derived from these rates, with writes and the download included:

| wave | runs | GPU-h each | GPU-h | §12 had |
|---|---|---|---|---|
| W1-D Q4B 50M | 1 | 3.6 | 3.6 | 1.1 |
| W1-C G4 50M; O7 T_15 50M | 2 | 3.8; 4.2 | 8.0 | 3.1 |
| W3 O7 16 checkpoints x 20M | 16 | 1.8 | 28.5 | 14.4 |
| W0 (this run, three attempts), W2-D, W2-C | | | 2.4 | 2.2 |
| total before contingency | | | 42.5 | 20.8 |
| with 30 % contingency | | | 55 | 27 |

One W3 job of 8 checkpoints is now about 14 GPU-h. The split between forward cost and hook cost was not
measured (follow-up 7).

## Corpus (N2)

- Dataset commit `2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b`. It has 165 `data/*` directories. The saved
  100,000-entry listing in the spec session's scratchpad was truncated at 140 and missed `stack_edu` and `rpj`.
- Directory rule as implemented: indices `floor(i * 165 / 50)`, so `k` is recorded as 3.3. The spec's
  `k = floor(n_dirs / 50)` gives 3, which leaves 55 directories, not 50. The picks are 38 common_crawl,
  7 olmocr_science_pdfs, 4 stack_edu and 1 finemath.
- The mix holds both `.jsonl.zst` (about 47k files) and `.jsonl.gz` (about 52k). A directory's shard
  files are both kinds, in lexicographic order. The next file is opened only when the previous one ends
  before the budget, so this matches the spec's first-shard rule whenever the first file is enough. At
  60,000 tokens per directory no directory needed a second file. The first `.gz` shards of the picked PDF
  directories are 0.11-0.28 MB, so the 1,000,000-token production rule will need the next-file step.
- 2,364 documents. Bands under OLMo-3 tokens end at 765,567, 1,371,588 and 3,186,563. The nominal ends are
  600,000, 1,200,000 and 3,000,000; each directory overshoots by at most one document, but single documents
  are long. Per-directory selection ranges from 60,074 to 116,919 tokens.
- Tokens: olmo3 3,186,563 (BOS = EOS = 100257); qwen3 3,265,623 (no BOS, EOS 151643 from the Base
  tokenizer); gemma3 3,326,273 (BOS 2, EOS 1). The vocabulary files of Qwen3-4B, -Base and
  -Instruct-2507 are byte-identical, and the harvest asserts vocabulary equality per model.
- The `zstandard` install was a no-op: version 0.25.0 was already present (`zstandard_preinstalled: true`
  in the manifest; the AMI's `bake/pip-freeze.txt` also lists it).
- Duplicate documents: 2,364 selected documents hold only 1,358 distinct texts. Repeat copies are 34.3 % of
  the characters, with at most 7 copies of one text, and every duplicate group sits inside one directory.
  This fits within-shard upsampling in the Dolma 3 mix. In the top-K, the median share of filled slots
  that are distinct (document text, position) pairs is 0.80 (Q4B), 0.82 (Q4, O7 both), 0.86 (G4, C0).

## Numbers per model (from `stats/` and `topk/`)

| model | poles | dead (max_abs = 0) | double-negative share >= 0.8 | top-K slots at window position 0 | poles with >= 25/50 slots there |
|---|---|---|---|---|---|
| Q4 | 700,416 | 5,577 | 24,250 (3.46 %) | 0.22 % | 1,406 |
| Q4B | 700,416 | 5,253 | 24,602 (3.51 %) | 0.30 % | 1,632 |
| C0 (131k tokens only) | 700,416 | 5,437 | 46,398 (6.62 %) | 0.17 % | 573 |
| G4 | 696,320 | 8 | 3,745 (0.54 %) | 1.51 % | 9,256 |
| O7 step1000 | 704,512 | 0 | 16 (0.002 %) | 0.01 % | 3 |
| O7 final | 704,512 | 2 | 3,182 (0.45 %) | 0.07 % | 81 |

A uniform spread would put 1/2048 = 0.05 % of slots at window position 0. The Qwen dead poles sit in layers
1-3. No neuron is zero on every token. In Q4B layer 1, 3,623 of 9,728 neurons never took one of the two
signs in 2M tokens. Corpus NLL over the first 120k scored tokens: Q4 2.764, Q4B 2.372, C0 2.604, G4 2.348,
O7 step1000 4.217, O7 final 2.151.

## Samples opened

Q4B (`q4b/samples/contexts.jsonl`, the first three sampled poles, ten contexts each; the token is in [[ ]]):
- `L00N8274+`, max_abs 0.221. All 10 contexts are double-negative (g about -0.5 to -0.9, u about -0.8 to
  -0.99), and sign_counts is [5,0,0,45]. Tokens: `.readFileSync` (one JavaScript document three times,
  ranks 0-2), ` dumpsters` twice, `-of`, `MethodInfo`, `.getElementsBy` twice, ` reimb`. No common meaning;
  low magnitude.
- `L00N8274-`, max_abs 0.933, g about +1.2, u about -0.9. Every context marks ` career` ("racing career",
  "radio career", "dream career"). One job, coherent.
- `L03N6196+`, max_abs 2.43. Name-ending subwords: `ly` in "Wil[[ly]]", `nie` in "Don[[nie]] Wahlberg"
  (7 of 10, at 4 distinct document positions, 3 of them as repeated copies), "Lilly", "Billy". Coherent: the end of a given name.
- Also opened: `L03N6196-` (` Hardy`, `y` in "hardy", `andy` in "Tandy") and `L07N4972-` (the second item
  of an "X and Y" pair: "signs and[[ symptoms]]", "Arts and[[ Humanities]]").

O7 final (`o7_final/samples/contexts.jsonl`, the first three of the 20 poles drawn at step1000):
- `L00N9363+`, max_abs 0.149, all (g >= 0, u >= 0). Mostly a proper noun after "and" or in a list:
  " Cuba", " Egypt", "Lux" in Luxor, " Brussels", " Venice", "Nass" in Nassim. Also "424" in a URL and
  "glor". Reads as one loose job.
- `L00N9363-`, max_abs 0.151, sign_counts [0,15,35,0]. " McA" in "Lim and McAleer" (two documents
  that are copies, 4 slots), " coward" twice, " Islam" twice, " plaint". Mixed; no single reading.
- `L03N7011+`, max_abs 0.139. Mid-word fragments: "Bore[[alis]]", "Car[[ab]]ott" (5 of 10 slots, five
  copies of one document), "K[[ory]]o", "casc"ading, " cannon", " failure". Mixed, dominated by one
  repeated document.

`samples/first_batch.json` exists for every model. Q4B after its first 16,384 tokens: three poles with 10
contexts each.

Band snapshot (q4b `snapshots/band1`, boundary 785,774 under qwen3): 0 of 34,680,198 top-K slots index at or
past the boundary, and the largest index is 785,773. All 14,649,313 full-run top-K entries that lie before
the boundary also appear in the band-1 top-K.

Exemplar record. The 20 O7 poles are evaluated at their step1000 top-50 positions with the final
checkpoint, as |a_final| / max_abs_final. Per-pole medians in pole order L00N9363+ ... L28N8952-:
0.0238, 0.0151, 0.0028, 0.0034, 0.0036, 0.0116, 0.0093, 0.0055, 0.0008, 0.0182, 0.0050, 0.0080, 0.0035,
0.0166, 0.0143, 0.0507, 0.0159, 0.0220, 0.0032, 0.0196. Median over poles: 0.0104. References, at the
final checkpoint for the same poles: the median of the pole's own top-50 is 0.39-0.76 of max_abs. The
upper edge of the median bin over all of the pole's tokens is 0.0026-0.055 of max_abs. The share of the
step1000 positions whose final a has the pole's sign is 0.18-0.68.

Probe smoke: 5 texts scored in 0.8 s (15 to 32 tokens each). `vocab.json`: the two poles of each neuron
mirror each other, because the sign is folded in. The rule is stated in `manifest.vocab_rule`.

## Pitfalls met

1. `pg run --weights` with repo ids syncs nothing and gives no warning.
2. Dolma 3 directories mix `.jsonl.gz` and `.jsonl.zst`. Some names carry digits after `.zst`
   (`shard_x.jsonl.zst338588528`); the suffix filter excludes them.
3. MLP hooks that stay registered break any other forward of the same model. They now exist only inside
   `MLPTap.run`.
4. Qwen3 has no BOS, so the first corpus token of each Qwen window has no context. Its activations are
   harvested as ordinary tokens, and probe texts have the same first-token condition. Gemma's first token
   after BOS holds 1.5 % of all top-K slots.
5. After the run, `pg audit` lists four orphan reaper schedules for terminated instances: one per attempt
   (`pg-reap-i-0df29a9d9e2b02da9`, `-i-07686203ab1e64e21`, `-i-0255f7c40c2b098b0`) and
   `pg-reap-i-0c39e8ec895674dae`, which predates this run. `pg ps` is empty. They were not removed.

## Follow-ups (listed, not fixed)

1. SPEC §4.1 and §5 G1: name Qwen3-4B-Instruct-2507 (`cdbee75f`) and `shell_text_perplexity`, or bank a
   P0 number for Qwen/Qwen3-4B. SPEC §15.6 is still open: `matrix_repo_facts.json` has no Qwen/Qwen3-4B row.
2. SPEC §4.2: replace `k = floor(n_dirs / 50)` with the evenly spaced rule, name both shard formats, and
   adopt or reject the next-file rule. The 1M-token production corpus needs it for the `.gz` PDF directories.
3. Duplicate documents (34 % of characters). Deciding whether to deduplicate by text hash before
   selection belongs to the spec owner. Duplicates take about 15-20 % of top-K slots and inflate
   density and the quantiles.
4. Window position 0: 9,256 G4 poles and 1,632 Q4B poles have half or more of their top-50 there.
   Excluding window position 0, or flagging those poles for the explainer, is a spec decision.
5. G0's failing input (activation swap) moves only 3/32 layers over tolerance at stage1-step1000. It will
   likely not fail at T_0. W3 needs a stronger registered failing input for early checkpoints.
6. §12 budget: about 55 GPU-h with contingency, not 27 (table above).
7. Profile the hook cost: run the loop without hooks to get the bare forward rate, then time the
   histogram `bincount` and the top-K separately. Writes are single-threaded `savez_compressed`.
8. `bake_env.sh`: zstandard 0.25.0 is on the AMI, pulled in by another package and not pinned. Pin it
   at the next bake.
9. `pg run` could check `--weights` names against `matrix_repo_facts.json:_meta.bucket_dirs`.
10. DATA_CONTRACT §1: `docs.jsonl.gz` `shard` now holds the directory, because a document's source file is
    recorded per directory in `manifest.shards[].files`. The contract wording says shard path.
11. The four orphan reaper schedules.
