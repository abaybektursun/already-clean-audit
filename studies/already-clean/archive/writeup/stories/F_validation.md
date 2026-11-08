# F validation: recomputed numbers, artifact tests, quotes and tiers

Validator: Claude Opus 5.5 (`claude-opus-5-5`), read-only. Work span: 2026-10-25T03:32Z to 2026-10-25T03:58Z (UTC).
Report under test: `writeup/stories/F_cross_model.md` (F). Every number below was computed this session from a file I opened; the tag in brackets names it.
Intermediate scripts and caches live only in the session scratchpad,
`/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad/v/` (one script per test, named in each section). Nothing else was written.

## Sources and conventions I checked

- **[H]** `interp_research-neurons-harvest/bank/neurons/w1/{q4b,g4,o7_t15}/1/out/<cell>/stats/layer_LL.npz`, all 102 layers.
- **[T]** the matching `topk/layer_LL.npz`. Its `tok` array holds global corpus positions, not token ids (`infra/pg/neurons_harvest.py` lines 360 to 366 index `tokens[tok]`). I decoded `tokens.npy[tok]` with each model's own `tokenizer.json` (**[K]**: `.../corpus/corpus/tok/{qwen3,gemma3,olmo3}/`). Decoded tokens equal the ⟦ ⟧ marks in [P] for 450 of 450 checked contexts per model, and my recomputed same-token share equals [H] `top_same_token_frac` exactly in every layer.
- **[HM]** `manifest.json` per cell. **[P]** `interp_research-neurons-explain/bank/w5/verdict/samples/q1_{Q4B,G4,O7}_poles.jsonl`. **[M]** `bank/w5/verdict/metrics.json`. **[W2C]** `bank/w2c/<cell>/explain/explanations.jsonl`. **[S]** `SPEC_D1.md`. **[HC]** `infra/pg/neurons_harvest.py`. **[VC]** `infra/pg/neurons_verdict.py`. **[TF]** `transformers` 5.20.2 modeling files in the harvest worktree's `.venv` (the harvest itself ran 5.9.0 per [HM]).
- Sign index 0 is the + pole: [HC] docstring line 5 ("Pole axis 0 is a > 0, axis 1 is a < 0"). I checked all 1,800 sampled poles, not one: the `max |a|` and `density` in every [P] header equal [H] at that index.
- `sign_counts` order is (g≥0,u≥0), (g≥0,u<0), (g<0,u≥0), (g<0,u<0) ([HC] lines 369 to 372). Double negative is index 3 and applies to the + pole.
- Density is the count in the histogram bin holding 0.01·max and every bin above ([HM] `density_rule`; [HC] `finalize_layer`, line 387 on). Recomputed from `hist`, it equals the file for all 2.1 million poles.
- RARE: fewer than 50 tokens with |a| ≥ 0.01·max, or max = 0 ([S] line 250).
- A pole's `max_abs` is exactly 0 only when no token outside window position 0 had that sign ([HC] lines 252 to 254; tokens with a = 0 are dropped, line 290). Window position 0 is excluded from top-k and stats ([S] lines 386 to 389; [HC] docstring lines 9 to 13; [HM] `excluded_positions`).
- Model identities ([HM] `repo`, `branch`): Q4B is `Qwen/Qwen3-4B-Base` (F says "Qwen3-4B"); O7 is `allenai/Olmo-3-1025-7B` at `stage1-step1413814`, the end of stage 1, not the released model (F says "OLMo 3 7B"); G4 is `google/gemma-3-4b-pt`.

## Summary

- Every population number in F reproduces exactly. Every sample number built on the `class` field reproduces exactly. Regex and classifier counts reproduce in direction; their size moves with the regex, sometimes a lot (name-start counts, passage overlap, Jaccard ratio).
- Small errors: one positive range, one ranking sentence, one "within 1%" claim, one "exactly 0.44", one peak-height range, the double-negative means (dead poles counted as 0), and an O7 tercile split copied from the spec that disagrees with the sample file.
- Bigger problems: (1) story 3's "deep neurons fire on function words" is a base-rate effect in Q4B and G4; (2) story 9's depth band and "the writer cannot fake broken text" do not survive a regex scan or the pole's own scores; (3) "only Qwen splits one prefix two ways" fails a population test; (4) "damaged" and "dead" are the wrong words for Qwen's early layers.
- Upgrades from the population: story 8 (deep name-start neurons) and the non-English U shape are confirmed without any explainer.

## 1. Recomputation table

"Match" means equal at F's printed precision. My values come from the named source with my own code.

| # | Claim in F | F value | My value | Verdict |
|---|---|---|---|---|
| 0.1 | Tokens scored | 52,442,093 / 53,512,674 / 50,515,866 | same ([HM] `n_tokens_scored`) | match |
| 0.2 | Neurons | 1.05 million | 350,208 + 348,160 + 352,256 = 1,050,624 ([H]) | match |
| 0.3 | Sign check Q4B L05N2328 | 6.866 / 4.425 | 6.866 / 4.425 ([H]); all 1,800 headers agree | match |
| 0.4 | Model names | Qwen3-4B; OLMo 3 7B | Qwen3-4B-Base; OLMo-3 stage-1 final checkpoint ([HM]) | naming caveat |
| 1.1 | ONE by within-layer max quartile, Q4B | 0.479 (163), 0.636 (140), 0.630 (146), 0.692 (143) | identical ([P] `class` joined to [H]) | match |
| 1.2 | Same, G4 | 0.425 (153), 0.541 (146), 0.607 (150), 0.682 (151) | identical | match |
| 1.3 | Same, O7 | 0.359 (145), 0.500 (144), 0.553 (159), 0.651 (152) | identical | match |
| 1.4 | ONE by density quartile | Q4B 0.659, 0.665, 0.611, 0.494; G4 0.678, 0.628, 0.514, 0.439; O7 0.694, 0.506, 0.493, 0.389 | identical | match |
| 1.5 | Corr. of loudness and density percentiles | -0.811, -0.851, -0.819 | Pearson -0.811, -0.851, -0.819 | match |
| 1.6 | Explainer split, below vs at/above layer median | Q4B O 0.565/0.706, S 0.500/0.521; G4 O 0.572/0.665, S 0.221/0.574; O7 O 0.446/0.638, S 0.391/0.506 | identical (n: Q4B S 66/71, G4 S 77/68, O7 S 87/87) | match |
| 1.7 | ONE by q999/max quartile | Q4B 0.696, 0.642, 0.534, 0.547; G4 0.680, 0.540, 0.487, 0.547; O7 0.680, 0.547, 0.440, 0.407 | identical using quartiles of the sample; within-layer quartiles give Q4B 0.649, 0.656, 0.553, 0.558; G4 0.614, 0.617, 0.511, 0.515; O7 0.635, 0.523, 0.523, 0.393 | match (definition noted) |
| 1.8 | "Explainer said UNK" poles | 10 / 9 / 13; max pct 0.15 / 0.27 / 0.19; density pct 0.74 / 0.81 / 0.67 | 10 / 9 / 13; 0.150 / 0.273 / 0.187; 0.736 / 0.807 / 0.674 | match |
| 1.9 | ONE poles' mean percentiles | 0.53 / 0.55 / 0.56 and 0.48 / 0.46 / 0.44 | 0.525 / 0.547 / 0.560 and 0.480 / 0.457 / 0.444 | match |
| 2.1 | Density peak layer and height | Q4B L17 (0.472) 22.82%; G4 L14 (0.412) 20.47%; O7 L14 (0.438) 20.96% | identical ([H]) | match |
| 2.2 | "Peak heights about 21% to 23%" | 21% to 23% | 20.47% to 22.82% | minor mismatch (G4 is 20.5%) |
| 2.3 | Top five layers | Q4B 14, 16 to 19; G4 10, 12, 13, 14, 16; O7 11 to 15 | identical | match |
| 2.4 | Density by decile (30 values) | as printed | identical as pooled medians over layer/L deciles | match |
| 2.5 | Share top_same_token_frac ≥ 0.5 by decile (30 values) | as printed | identical | match |
| 2.6 | Lowest single-token layer | Q4B L15 0.040; O7 L10 0.029; G4 L29 0.051, local low L14 0.080 | identical | match |
| 2.7 | Opus-only ONE by quintile | Q4B 0.742 (124), 0.746 (122), 0.464 (112), 0.546 (97); G4 0.730 (122), 0.605 (124), 0.521 (121), 0.625 (88); O7 0.667 (132), 0.474 (116), 0.477 (128), 0.580 (50) | identical; no Opus pole in the top quintile | match |
| 2.8 | Strong share of writer positives by quintile | Q4B 0.83, 0.81, 0.49, 0.64, 0.65; G4 0.86, 0.61, 0.56, 0.70, 0.46; O7 0.77, 0.55, 0.54, 0.67, 0.45 | identical | match |
| 2.9 | Median q999/max | 0.136, 0.131, 0.170 | 0.136, 0.131, 0.170 | match |
| 2.10 | Share of neurons whose + pole holds 40% to 60% of tokens | 0.775, 0.805, 0.924 | 0.775, 0.805, 0.924 | match |
| 3.1 | Marks per bottom/top fifth | 3,804 / 3,720 etc. | identical; Q4B's 3,804 includes 54 marks from RARE poles with fewer than 30 contexts | match |
| 3.2 | Word pieces, bottom to top fifth | Q4B 40.7% to 10.4% (low 6.6% at 0.6 to 0.8); G4 19.2% to 6.4% (low 5.2%); O7 26.7% to 8.7% | my classifier: Q4B 41.5% to 10.6% (low 8.4%); G4 19.8% to 6.6% (low 5.6%); O7 27.2% to 9.8% | match in direction, small shifts |
| 3.3 | Function words, bottom to top fifth | Q4B 5.6% to 20.3% (peak 26.2%); G4 9.6% to 21.9%; O7 0.9% to 36.8% | Q4B 5.7% to 20.2% (peak 28.0%); G4 13.3% to 22.1%; O7 0.9% to 37.3% | match except G4 bottom (my 110-word list counts "i", "not", "or") |
| 3.4 | Top-fifth function words | Q4B the 73, in 70, a 56, to 52; G4 and 79, the 75, of 60, a 58; O7 the 110, to 101, a 84, of 77 | identical | match |
| 3.5 | Layer 0 share with same-token ≥ 0.5 | 0.548, 0.555, 0.115 | 0.548, 0.555, 0.115 | match |
| 4.1 | Opus next-word language, ONE vs other | Q4B 0.515 (134) vs 0.682; G4 0.500 (162) vs 0.686; O7 0.491 (114) vs 0.567 | my regex: 0.511 (139) vs 0.687; 0.506 (164) vs 0.684; 0.516 (126) vs 0.560 | match in direction |
| 4.2 | Opus layout language | Q4B 0.344 (32) vs 0.655; G4 0.417 (24) vs 0.631; O7 0.269 (26) vs 0.565 | my regex: 0.431 (51) vs 0.658; 0.490 (51) vs 0.636; 0.386 (44) vs 0.565. Narrow regex: 0.400 (35), 0.423 (26), 0.333 (27) | direction matches; size regex-dependent |
| 4.3 | ONE by modal firing-token type | Q4B piece 0.76 (95), word 0.61 (263), function 0.56 (89), punct 0.48 (62), space 0.50 (30); G4 0.69, 0.60, 0.54, 0.44, 0.29; O7 0.69, 0.53, 0.45, 0.33, 0.33 | Q4B 0.72 (116), 0.61 (244), 0.54 (96), 0.48 (75), 0.67 (18); G4 0.69 (90), 0.58 (261), 0.57 (107), 0.45 (84), 0.25 (20); O7 0.66 (86), 0.52 (330), 0.47 (113), 0.38 (42), 0.25 (8) | order holds except Q4B whitespace (n under 30, definition-dependent) |
| 5.1 | Lean share (+ pole under 40% or over 60% of tokens) | Q4B L18 0.07, L32 to L35 0.25, 0.30, 0.46, 0.56; G4 L17 0.15, L30 to L33 0.14, 0.27, 0.26, 0.46; O7 L16 0.05, L28 to L31 0.13, 0.18, 0.25, 0.34 | identical ([H] `n_pole_tokens`, [HM] tokens) | match |
| 5.2 | "Highest anywhere ... Q4B L2 (0.82), then L35 (0.56)" | L2 then L35 | L2 0.816, L1 0.749, L3 0.617, then L35 0.564 | mismatch (ranking sentence) |
| 5.3 | Final layer 5th and 95th percentile | Q4B 0.081, 0.927; G4 0.100, 0.899; O7 0.101, 0.891 | identical | match |
| 5.4 | Median stronger/weaker max ratio | middle 1.716, 1.767, 1.981; last four as printed | identical | match |
| 5.5 | "+ and - medians agree within 1% in every decile" | within 1% | O7 decile 3: 1.029 vs 1.042 (-1.2%); per layer up to 2.9% (Q4B L10) | minor mismatch |
| 6.1 | Jaccard, same neuron vs same layer | Q4B 0.0424 vs 0.0126; G4 0.0393 vs 0.0143; O7 0.0416 vs 0.0130 (ratio 2.7 to 3.4) | my stop list: 0.0592 vs 0.0239; 0.0555 vs 0.0257; 0.0593 vs 0.0262 (ratio 2.2 to 2.5) | direction matches; "three times" is "two to three times" |
| 6.2 | 72-neuron reading | S/R/F/U counts | sums are consistent (S 14, R 21, F 11, U 26; 33 + 39 = 72) | not verifiable (judgment) |
| 6.3 | Neurons with a shared passage in both top-30 lists | 20/296, 27/300, 11/300 | 40-character text window: 58/296, 53/300, 32/300. Harvest positions within 80 tokens: 20, 19, 8; within 272 tokens: 35, 34, 16 ([T], [K]) | definition-dependent |
| 6.4 | Class concordance | Q4B 0.659 vs 0.538; G4 0.631 vs 0.485; O7 0.528 vs 0.482 | identical | match |
| 7.1 | Twin pole ids, depths, classes (26 poles) | as printed | all 26 exist in [P] with the stated depth and class; G4 L07N10202- positives 0.76, 0.92, 0.85, 0.69, 0.75 and blind max 1.29 | match |
| 7.2 | TF-IDF depth test | top 100: 0.274 vs all 0.332; Opus 0.222 vs 0.249 | my TF-IDF (no stop words): 0.275 vs 0.332; Opus 0.220 vs 0.249 | match |
| 7.3 | "respectively twice at exactly 0.44 depth" | exactly 0.44 | 16/36 = 0.444 and 14/32 = 0.438 | minor mismatch |
| 8.1 | Name-start token counts (11 poles) | e.g. " Han" 23 and "Han" 6 | all counts identical in [P] `explainer_saw` | match |
| 8.2 | Q4B L27N6887- positives | 0.66 to 0.84 | 0.66, 0.66, 0.84, 0.61, 0.69 | mismatch (min is 0.61) |
| 8.3 | Opus share of the 11 | 7 of 11 | 7 of 11 | match |
| 8.4 | Name-start regex count, total / depth ≥ 0.6 | 23/24/29; 18/17/12; O7 14 in bottom fifth | my regex 40/43/49; 26/34/16; O7 28. Narrow 30/37/41. Names only 9/25/18, deep 5/19/5 | regex-dependent |
| 9.1 | Broken-text poles: depths, classes, ranges, scores | 12 poles, 11 UNK, 1 ONE; 0.07; 0.04; 0.01 | identical | match |
| D.1 | Q4B poles with max exactly 0 / below RARE | 3,521 / 6,490; L1 to L4 3,515 / 6,472; L1 2,467 / 3,786 of 19,456 | identical | match |
| D.2 | G4 and O7 | 3 and 56; 0 and 2 | identical | match |
| D.3 | Mean double-negative share of + poles, Q4B L1 to L3 | 0.481, 0.808, 0.477 | 0.552, 0.836, 0.487 over poles that have a top-50; F's values appear only if the 1,255 / 321 / 193 dead + poles count as 0 | method mismatch |
| D.4 | Max layer mean double-negative share | G4 0.015; O7 0.051 | 0.015 (L33); 0.051 (L0) | match |
| D.5 | pos0_share ≥ 0.5 | G4 L9 to L11 1,125, 3,215, 3,334; O7 max 10; Q4B L2 1,276, L35 146 | identical | match |
| D.6 | Layer-0 median max; O7 rise; Q4B rise; G4 path | 1.292, 10.504, 0.299; O7 L2 0.249 to L31 30.575; Q4B L17 4.872 to L35 64.944; G4 values | identical; O7 rises at every step from L2; Q4B at every step from L17 | match |
| D.7 | Code share, bottom to top fifth | O7 0.30 to 0.05; Q4B 0.20 to 0.25; G4 0.23 to 0.20 | 0.31 to 0.05; 0.21 to 0.25; 0.23 to 0.20 | match |
| D.8 | Median depth of Python, JavaScript, C# poles | Q4B 0.39, 0.60, 0.33; G4 0.29, 0.44, 0.41; O7 0.19, 0.14, 0.16 | Q4B 0.40, 0.60, 0.33 (n 16, 18, 12); G4 0.29, 0.41, 0.32 (25, 21, 17); O7 0.19, 0.14, 0.16 (13, 14, 19) | G4 differs slightly |
| D.9 | Language/script share, top fifth; overall; "Chinese" | 0.22, 0.15, 0.04; 0.14, 0.11, 0.06; 31, 9, 9 | 0.21, 0.14, 0.04; 0.13, 0.10, 0.05; 31, 9, 9 | match |
| D.10 | Digit marks | Q4B 1,011; G4 1,399; O7 197 (93, 43, 61) | 1,011; 1,398; 196 (92, 43, 61) | match within one |
| D.11 | Shallow neurons sharing ≥ 3 preceding fragments | 5/66, 0/62, 1/66 | 5/66 (lowercase), 0/62, 1/66 | match (but see artifact test) |
| X.1 | Word-piece language within 0.6 to 0.8 depth, Opus vs Sonnet | Opus 0.06, 0.19, 0.02; Sonnet 0.36, 0.31, 0.25 | Opus 0.03, 0.17, 0.02; Sonnet 0.27 (n 11), 0.25, 0.22 | match in direction |
| X.2 | Deep-tercile Opus/Sonnet | 65/135, 54/140, 14/186 (cited from [S]) | [P]: 65/135, 54/140, 14/174. [S] line 210 counts O7 over 640 poles ([W2C] has 640, 40 more than the sample) | O7 figure mismatches the sample file |
| X.3 | Every description Opus or Sonnet | [M] `kappa_models` | [M]: Q4B 463/137, G4 455/144 (+1 declined), O7 426/174 | match |

## 2. Artifact tests

### 2.1 Story 1: loud poles are legible

Scripts: `s1_base.py`, `s1_art.py`, `s1_instr.py`, `s1_mech.py`. Data: [P] `class`, `explainer_model`, `texts`, `shuffled`; [H] `max_abs`, `q999`, `q9999`, `density`, `hist`. Intervals are 2.5th to 97.5th percentiles of 2,000 pole resamples.

Loud minus quiet quartile (Q4 minus Q1 of ONE share):

| Loudness measure | Q4B | G4 | O7 |
|---|---|---|---|
| max_abs, RARE excluded | +0.21 [+0.10, +0.32] | +0.26 [+0.15, +0.36] | +0.29 [+0.18, +0.40] |
| max_abs, RARE included as not ONE | +0.24 [+0.14, +0.34] | same (no RARE) | same (no RARE) |
| q9999 | +0.16 [+0.04, +0.27] | +0.28 [+0.17, +0.39] | +0.31 [+0.20, +0.42] |
| q999 | +0.07 [-0.04, +0.19] | +0.12 [+0.01, +0.24] | +0.23 [+0.12, +0.34] |

Inside depth and explainer. Loud half minus quiet half, weighted over depth quintile by explainer strata: Q4B +0.11 [+0.03, +0.19], G4 +0.15 [+0.07, +0.22], O7 +0.18 [+0.10, +0.25]. Opus only: +0.14 [+0.05, +0.23], +0.09 [+0.01, +0.19], +0.19 [+0.10, +0.28]. Sonnet only: +0.02 [-0.15, +0.18], +0.35 [+0.19, +0.51], +0.12 [-0.05, +0.26]. Per depth quintile the loud half wins in 11 of 15 cells. Q4B's top quintile is exactly level (0.500 vs 0.500); G4 at 0.2 to 0.6 (-0.02, -0.03) and O7's top quintile (-0.02) are slightly negative. The top quintile is all Sonnet in every model, so the effect is weakest among deep Sonnet poles outside G4.

Density in disguise? In a logistic model with depth quintile and explainer controls, loudness alone gives +1.14, +1.38, +1.59 (all intervals above 0) and density alone gives -1.13, -1.41, -1.63. With both in, neither interval excludes 0. They cannot be separated, and the reason is mechanical: density counts tokens above 1% of the max, so a louder max raises the bar. With a density threshold that does not use the max, the correlation with loudness falls from -0.81/-0.85/-0.82 to -0.41/-0.47/-0.32 (1% of q9999), -0.34/-0.45/-0.41 (10% of q999) and -0.02/-0.08/+0.21 (10% of q99). The density effect on ONE shrinks with it: sparse minus dense quartile is +0.17/+0.24/+0.31 under F's rule and +0.03/+0.10/-0.12 under the q99 rule. So "density" is the max seen again, not a second finding. F half-says this ("one finding seen twice"); the reason is the rule.

What "legible" measures here. Of the failing poles, 92% (Q4B), 94% (G4) and 97% (O7) fail the positive clause (fewer than 3 of 5 writer texts reach θ_s = 0.3153 of the max). Near-miss and distant clauses almost never decide it (at most 7 failures per quartile). Unrelated texts scored at a pole (the `shuffled` control) are strong 0% to 1.6% of the time in every quartile. So the effect is: the writer can drive loud poles to a third of their max more often. Mean positive score rises from 0.43 to 0.56 (Q4B), 0.40 to 0.51 (G4), 0.37 to 0.47 (O7) across quartiles.

Size. The AUC of the within-layer loudness percentile for ONE is 0.593, 0.607, 0.619. It is a real but weak predictor.

Verdict: survives depth, explainer, RARE and the q9999 loudness measure. F's sentence "it holds inside Opus-explained and inside Sonnet-explained poles of each model" is not supported for Q4B Sonnet (+0.02) and is not significant for O7 Sonnet. "Predicts which neurons an LLM explainer can describe" should read "is associated with, AUC about 0.6".

### 2.2 Story 2: shared depth landmark

Scripts: `s2_base.py`, `s2_art.py`, `s2_one.py`. Data: [H] `density`, `hist`, `q99`, `q999`, `q9999`, `top_same_token_frac`, [P].

Peak of the per-layer density curve (layer and depth by layer/L; in brackets the layers within 5% of the peak):

| Density rule and aggregate | Q4B | G4 | O7 |
|---|---|---|---|
| File rule (1% of max), median, both signs | L17, 0.47 (L14 to L19) | L14, 0.41 (L10 to L14) | L14, 0.44 (L10 to L15) |
| Same, + pole only / - pole only | L17 / L17 | L14 / L14 | L11 / L14 (same plateau) |
| Same, mean | L1 (the near-silent layer dominates) | L14 | L11 (same plateau) |
| 1% of q9999, median | L16, 0.44 | L14, 0.41 | L15, 0.47 (L10 to L18) |
| 10% of q999, median | L10, 0.28 (L10 to L16) | L14, 0.41 | L13, 0.41 |
| 10% of q99, median | L10, 0.28 | L23, 0.68 | L21, 0.66 |
| Depth as layer/(L-1) | 0.49 | 0.42 | 0.45 |

Heights under the file rule are 22.8%, 20.5%, 21.0%; under 1% of q9999 they are 38.3%, 37.0%, 35.7%. The location is stable across sign, median versus layer/(L-1), and every rule tied to the tail (max, q9999). It moves under a rule tied to the body of the distribution (q99). So the landmark is "the layers where the peak sits least far above the bulk", not "the layers where most tokens are active". Density divided by the pole's own token count peaks in the same plateau (Q4B L17, G4 L14, O7 L11), so it is not a sign-balance effect. Bins are 5.9% wide, which blurs every layer alike and cannot move the peak. G4 is not unimodal: its median density rises again to 16.9% at L29 (0.85 depth).

Single-token concentration low point: stable across thresholds 0.3, 0.5, 0.7 and both signs. Q4B L15 to L18 (0.42 to 0.50); O7 L10 (0.31) in every variant; G4's global low is L29 to L30 (0.85 to 0.88) with a local low at L11 to L14.

ONE-share recovery at 0.6 to 0.8 depth, Opus only: Q4B +0.08 [-0.05, +0.22], G4 +0.10 [-0.04, +0.25], O7 +0.10 [-0.06, +0.27]. Pooled mean of the three: +0.10 [+0.01, +0.18]. Inside each model the recovery is within sampling noise. The Opus poles in that band are its lowest layers only (Q4B L22 to L28, G4 L21 to L25, O7 L20 to L22), because Opus explained the first poles in pole-id order ([S] line 200).

Verdict: the population landmark is real and shared under tail-based rules, with a definition caveat. The per-model recovery claim is not supported; only the pooled one is, barely.

### 2.3 Story 3: from word pieces to function words

Scripts: `classify.py`, `s3_sample.py`, `s3_pop.py`, `baserate.py`. My classifier: whitespace-only, digit, punctuation, then letter tokens with a leading space are function words (my 110-word list) or whole words; letter tokens without a leading space are word pieces when the preceding character is a letter, word starts otherwise.

On the sampled contexts my counts agree with F (table rows 3.2 to 3.4). Regex sensitivity: counting a piece after a digit, or a function word at a line start, moves no share by more than 2.2 points.

Population, all poles of all layers, top-50 decoded with each model's tokenizer ([T], [K]; about 105 million top-50 slots):

| Share of top-50 tokens | Q4B bottom / middle / top fifth | G4 | O7 |
|---|---|---|---|
| Word piece | 39.4% / 10.4% / 12.1% | 23.8% / 5.3% / 8.4% | 26.0% / 10.6% / 8.0% |
| Function word | 9.0% / 18.7% / 19.3% | 12.1% / 21.5% / 21.5% | 1.6% / 16.7% / 37.9% |
| Corpus base rate: word piece | 7.1% | 4.4% | 7.3% |
| Corpus base rate: function word | 27.0% | 26.3% | 28.1% |
| Function-word enrichment, top fifth | 0.71x | 0.82x | 1.35x |
| Word-piece enrichment, bottom to top | 5.6x to 1.7x | 5.4x to 1.9x | 3.6x to 1.1x |

Per layer (top-50, word piece / function word): Q4B L2 57/6 to L23 9/26, then L34 15/12; G4 L2 41/9 to L33 9/28; O7 L2 44/1 to L27 7/41.

Verdict. The sampled result matches the population, and the word-piece decline is real in all three models. The function-word reading is not. In Q4B and G4 the top-fifth function-word share stays below what a random token from the corpus would give (0.71x, 0.82x). Deep firing tokens move toward the corpus mix: what changes is selectivity, not a preference for "the", "a", "of". Only OLMo's top fifth is enriched for function words. "Deep neurons fire on the slot before the word they are about" cannot be read off these counts.

Layer 0 lookup: merging case and leading-space variants of a token, the share of layer-0 poles with half their top 50 on one token is 0.601, 0.603, 0.130. OLMo's layer 0 is not a lookup under either definition.

### 2.4 Story 4: form and prediction jobs

Script: `s4.py`, `s4b.py`. Data: [P] `explanation`, `class`, `explainer_model`.

Gap in ONE share (regex hit minus other Opus poles), raw and stratified by depth quintile:

| Family | Q4B | G4 | O7 |
|---|---|---|---|
| Next word, raw | -0.18 [-0.27, -0.07] | -0.18 [-0.27, -0.09] | -0.04 [-0.15, +0.06] |
| Next word, depth-stratified | -0.11 [-0.21, -0.00] | -0.15 [-0.25, -0.05] | -0.02 [-0.13, +0.09] |
| Layout, depth-stratified | -0.18 [-0.33, -0.03] | -0.14 [-0.29, +0.01] | -0.16 [-0.31, -0.01] |

Next-word descriptions sit deeper (mean depth 0.49 vs 0.33 in Q4B), so part of the raw gap is depth. Verdict: next-word holds in Q4B and G4, not O7; layout holds in Q4B and O7, borderline in G4. "In all three models" is not supported for either family alone.

### 2.5 Story 5: last-layer lean

Scripts: `s5.py`, `s5push.py`. Data: [H] `n_pole_tokens`, `max_abs`, `hist`, [HM].

Shape by layer (lean share): Q4B is 0.02 to 0.08 from L6 to L21, steps to 0.18 at L22 (0.61 depth), holds 0.25 to 0.33 to L33, then 0.46 and 0.56. G4 is high at L0 to L6 (0.23 to 0.40), falls to 0.02 at L14, rises to 0.28 at L23, dips to 0.05 at L29 (0.85), then 0.14, 0.27, 0.26, 0.46. O7 rises slowly from 0.05 (L16) to 0.08 to 0.10 (L18 to L26), then 0.11, 0.13, 0.18, 0.25, 0.34. The final layer is the highest in all three (outside Q4B L1 to L3). The steep part is the last two to four layers in all three; the lead-up differs (Q4B a plateau from 0.61, G4 a dip at 0.85, O7 a gradual ramp).

Tokens with a = 0 average at most 0.14% per neuron in any layer, so the lean is not zeros counted as one sign.

Is there a standing offset? From the histograms (geometric bin centres), the median |E[a]| / E[|a|] of lean neurons is 0.35 to 0.57 in every checked layer, against 0.07 to 0.11 for other neurons, and the mean's sign follows the majority sign in 94% to 100% of lean neurons. This supports F's "standing push" reading as a description of the activation. It says nothing about the effect on the output.

Sign symmetry: decile medians of + and - max agree within 1.2% (O7 decile 3 is -1.2%); per layer within 2.9%. The check is a statistical expectation from the sign-flip symmetry of (W_up row, W_down column), not an exact law.

Verdict: the lean and the pole-ratio convergence reproduce and are a last-layers property in all three, sitting on model-specific lead-ups.

### 2.6 Differences: "Qwen's damaged early layers"

Script: `dmg.py`. Data: [H] `max_abs`, `density`, `n_pole_tokens`, `sign_counts`, `pos0_share`.

| Q4B layer | + poles max 0 | - poles max 0 | neurons with both signs 0 | RARE poles | neurons gate-closed on all 100 top tokens | median max of layer |
|---|---|---|---|---|---|---|
| L1 | 1,255 | 1,212 | 0 | 3,786 | 4,788 of 9,728 | 0.876 |
| L2 | 321 | 332 | 0 | 1,758 | 6,449 | 0.041 |
| L3 | 193 | 177 | 0 | 895 | 3,114 | 2.414 |
| L4 | 15 | 10 | 0 | 33 | 246 | 8.270 |
| G4, worst layer | | | 0 | 36 (L11) | at most 36 per layer | |
| O7, worst layer | | | 0 | 1 | at most 22 per layer | |

- No neuron in any model has both signs at max 0. Every one of the 3,521 neurons with a zero pole has its other pole on 100% of the 52.4 million tokens, a median max |a| of 0.0052 (0.6% of its layer's median max), and a top-50 that is 100% gate-negative (g < 0). These neurons are one-signed and near-silent on every token. They are not dead: they write a small nonzero value everywhere.
- 551 of the 3,521 "exactly 0" poles have `pos0_share` = 1.0: they do take that sign, only at window position 0, which the harvest excludes. The other 2,970 have no token of that sign even there.
- Double-negative share (index 3 of `sign_counts`, + pole): 0.552, 0.836, 0.487 in L1 to L3 when computed over poles that have a top-50. F's 0.481, 0.808, 0.477 count the undefined dead poles as 0.
- Word choice. "Dead" is wrong (no neuron is silent on both signs, and the one-signed ones are never exactly zero). "Constant" is too strong (live-pole q99/max median 0.19). "Damaged" is a judgment the data cannot support. The accurate phrase is "near-silent, one-signed neurons whose gate never opens on this corpus".
- What the data can and cannot say. It establishes that on 52 million tokens of OLMo's stage-1 mix, outside window position 0, these gates stay closed. It cannot say whether they open on Qwen's own training text (more Chinese and code), at window position 0 in other contexts, or on instruction-format text, and it cannot say whether they matter to the output. Q4B is the Base checkpoint.

### 2.7 Gemma first-token sink

Data: [H] `pos0_share`, [HM], [S]. Counts reproduce (row D.5). The exclusion rule is confirmed in [S] lines 386 to 389, [HC] lines 9 to 13 and [HM] ("BOS and window position 0"). G4 and O7 prepend a BOS ([HM] `bos_id` 2 and 100257), so their position 0 is the first token after BOS; Q4B has no BOS (`bos_id` null), so its position 0 is the model's real first input. Totals: G4 9,652 poles ≥ 0.5 (L9 to L11 hold 7,674); Q4B 3,371, almost all in L0 to L6. "Three middle layers" should read "three layers at 26% to 32% depth".

### 2.8 Other differences

- **OLMo layer 0 and activation growth.** [TF] `models/olmo3/modeling_olmo3.py` lines 262 to 265 apply the MLP to the raw residual stream (norm after the MLP); Qwen3 (lines 311 to 313) and Gemma 3 (lines 412 to 415) normalise before it. This accounts for OLMo's monotonic growth of max |a| that F left unchecked. Cross-model comparisons of raw max are not meaningful; within-layer percentiles are.
- **Non-English at both ends (population check, `nonlatin.py`).** Share of top-50 tokens containing a non-Latin letter, by depth fifth: Q4B 4.7%, 0.6%, 0.8%, 1.9%, 2.5% (corpus 0.24%); G4 2.1%, 0.5%, 0.6%, 1.0%, 1.5% (0.22%); O7 1.0%, 0.6%, 0.3%, 0.3%, 0.3% (0.26%). The U shape in Qwen and Gemma and the bottom-only pattern in OLMo hold without any explainer.
- **Digits (`digits.py`).** OLMo's tokenizer packs 2.0 digits per digit token, so the corpus has half as many digit tokens (4.0% vs 7.7% and 7.5%). Beyond that, digit tokens are less enriched in O7's top-50 (0.46x vs 0.71x and 0.93x), and poles whose most common top-30 type is a digit are 1.2% in O7 against 4.7% and 6.0%. F's inference that digit neurons are rarer in OLMo holds, but F's own evidence (mark counts) could not show it.
- **Only Qwen splits one prefix two ways (`prefix_pop.py`).** Over every neuron below 0.2 depth, using the preceding token on word-piece marks in each pole's top 30, the share of neurons whose two poles share at least 3 preceding-token occurrences is 3.0% (Q4B, 2,230 of 74,309), 3.7% (G4) and 4.8% (O7). The sample's 5/0/1 is small-sample noise. The claim fails.
- **Name-start neurons (story 8, `s8pop.py`).** Share of poles with half or more of their top-50 on a short capitalized word start followed by a lowercase continuation: under 0.3% in the middle of every model; Q4B rises from L23 to 5.3% at L35; G4 from L20 to 4.7% at L33; O7 from L17 to a peak of 2.4% at L20, falling to 0.1% to 0.7% at the top. G4 and O7 also have a layer-0 bump (2.6%, 2.4%). The family is real in all three without an explainer; in O7 it sits at 0.53 to 0.72 depth, not near the output. It is a small part of story 2's single-token rise: in Q4B at L23 to L28 these poles are 1.1% to 2.3% of all poles while poles with same-token share ≥ 0.5 are 13% to 16%. O7 L25N8888- (" fl" in "fling", " Sc" in "Scrum") is a word-start pole, not a name pole.
- **Story 9 band and the writer (`[P]` scores, regex scan).** My regex (spun, incoherent, word salad, garbled, gibberish, scrambled, spliced, machine- or auto-generated, keyword-stuffed, SEO spam, jumbled) finds 13 / 24 / 13 poles at depths from 0.00 to 0.97; 2 Q4B, 8 G4 and 5 O7 hits sit above 0.75 (Sonnet-explained; several about spun SEO text, others about OCR-garbled text). The 20% to 40% band is a property of F's hand selection. On the 12 listed poles (11 with texts), at least one writer text reached θ_s on five: G4 L10N9543- passes (3 of 5 strong), G4 L12N4911+ has positives 0.42 and 0.37, G4 L13N5135+ has a blind maximum of 0.77 (its one-sentence max text splices collocations exactly as described), Q4B L13N2349- has positives 0.66 and 0.73 and a blind max of 0.66, O7 L10N5687+ has a positive at 0.66. "A fluent model's attempts at nonsense barely move them" holds for 6 of the 11 testable poles.
- **Story 6 overlap and the vocabulary line (`s6vocab.py`).** In all 900 sampled neurons the explainer saw identical output-vocabulary lines for both poles (the sign-blind line reported in C). Removing every word that appears in those lines leaves the same-neuron overlap ratio at 2.3, 2.0, 2.1. The shared input does not explain the relatedness. Same-document co-occurrence is common: 25%, 25%, 24% of all neurons have a + and a - top-30 token in the same document ([T], [K] `doc_offsets.npy`), so "rarely in the same passage" holds only for windows of about 100 tokens or less.
- **Story 7 depth test (`s7.py`).** A within-model depth permutation keeping the top 100 pairs gives p = 0.005 with stop words removed but p = 0.10 with all words kept; Opus-only p = 0.06 and 0.015. "Slightly closer in depth" is fragile; F's conclusion that a matching job does not imply a matching depth stands.

## 3. Quotes and pole ids

Every pole id in F (57 signed pole ids on 58 neurons) exists in [P] under the model F attributes it to; I checked depth and class for every one that F annotates. Quotes checked against the named pole's `explanation` or `texts` in [P]:

| # | Pole | Quote in F | Status |
|---|---|---|---|
| 1 | Q4B L20N7249- | "Diffuse, context-heavy pole with no shared token" | verbatim |
| 2 | G4 L31N7089- | "The pole fires, weakly almost everywhere" | verbatim but truncated: the sentence goes on to name a specific job (next word a form of "monitor") |
| 3 | O7 L16N2846+ | "A diffuse, dense pole with no shared token or topic" | verbatim |
| 4 | Q4B L16N9243- | "Fires on the first word or symbol of a new structural segment, especially right after a markdown heading marker" | verbatim |
| 5 | O7 L17N5377+ | "actual results may differ materially" | altered: source reads "actual results may/could differ materially" |
| 6 | O7 L17N5377- | "whether promised things are actually carried out" | paraphrase presented as quotation: source reads "whether previously stated, promised, planned or claimed things are actually carried out" |
| 7 | G4 L24N3271- | "Predicts that a comma comes next" | verbatim |
| 8 | G4 L05N4009+ | "Fires on the floating-point type keyword in code" | verbatim |
| 9 | G4 L05N4009- | "Fires on the digit token right after a decimal point" | verbatim |
| 10 | O7 L30N1510+ | "Fires on a verb or preposition that is about to be followed by an article" | verbatim |
| 11 | O7 L30N1510- | "Fires on the word right after a determiner like "a", "an", "one" or "the" ... when a singular countable noun phrase is still unfinished" | verbatim with a marked elision |
| 12 | Q4B L16N764- | "most strongly on "respectively" after two parallel items" | verbatim |
| 13 | O7 L14N5001+ | "right before the word "respectively" would come" | verbatim |
| 14 | O7 L14N5001- | "Fires on the first item of a paired list where a matching second item and then "respectively" is about to follow" | verbatim |
| 15 | Q4B L27N6887+ | "Fires on the capitalized token "Han" (with or without a leading space) when it begins a proper name" | verbatim |
| 16 | G4 L22N1709+ | "Fires on the token "Wool" when it begins a proper name rather than the word wool" | verbatim |
| 17 | O7 L23N809- | "Fires strongly on the word-piece "Gul" (or "ul" after G) when it begins a rare proper name or place" | verbatim |
| 18 | O7 L13N2659- | "fires most strongly on sentence-ending periods in incoherent, spun or machine-generated essay-mill text, where each sentence jumps to an unrelated topic" | verbatim (mid-sentence) |
| 19 | O7 L13N2659- (writer) | "The weather of the quarterly report was blue with ancient hamburgers. Dolphins have always negotiated the tax code of Belgium." scored 0.07 | verbatim, score 0.072 |
| 20 | G4 L10N9543+ (writer) | "Best weight loss pills diet fast burn fat cheap buy online garcinia..." at 0.04 | verbatim, score 0.043 |
| 21 | G4 L13N5135+ | "Fires on the second word of a stock two-word collocation that has been wrongly spliced into a sentence" | verbatim |
| 22 | G4 L13N5135+ (writer) | "check your credit history Card" 0.01 | verbatim, score 0.010; F omits that the same pole's blind max text scored 0.77 |
| 23 | Q4B L13N2349- | "Fires on a word that abruptly breaks the flow of the text" | verbatim |
| 24 | O7 L01N10284- | "useEffect" and "Exactly" | supported (marks " useEffect" 18, " Exactly" 6) |
| 25 | O7 L00N1929- | the "™" of "вЂ™" | supported (explanation and 26 of 30 marks) |
| 26 | Q4B L11N5238+ | " KEY" after FOREIGN | supported (30 of 30 marks) |
| 27 | Q4B L07N4960+ | Harris and "useState" | supported |
| 28 | Q4B L03N6845 | "con"/"cont" | supported (+ pole preceding fragments cont 12, con 9; - pole Cont 8, cont 5) |
| 29 | Q4B L05N8095 | "E" for Eck- and E-ston-ia | supported |
| 30 | Story 8 token counts, 11 poles | " Han" 23, " Wool" 25, " Gul" 23, " T" 25 and others | all identical |

Paraphrase or alteration presented as quotation: rows 5 and 6. Selective truncation that changes the reading: row 2. Selective omission of a contrary score: row 22.

## 4. Tiers

| Finding | Tier | Caveat or reason |
|---|---|---|
| S1 Loud poles are legible | SAFE WITH CAVEAT | Real across depth, explainer, and the max and q9999 loudness measures but weak (AUC about 0.6), absent in Q4B's Sonnet poles, and "density" is the same signal by construction. |
| S2a Shared density landmark at 41% to 47% depth | SAFE WITH CAVEAT | Stable under max- and q9999-based rules, sign and aggregation, but it moves under a q99-based rule, so call it the layers where the peak sits least far above the bulk; G4 has a second rise at 0.85. |
| S2b ONE share dips in the middle and recovers at 0.6 to 0.8 inside Opus | SAFE WITH CAVEAT | The dip is clear; the recovery is within noise in every model alone and only marginal pooled (+0.10, 0.01 to 0.18). |
| S3a Word pieces fall with depth in all three | SAFE | Population-confirmed: enrichment over the corpus falls from 5.6x/5.4x/3.6x to 1.7x/1.9x/1.1x. |
| S3b Deep neurons fire on function words (detector to predictor) | DO NOT PUBLISH | In Q4B and G4 deep function-word share stays below the corpus base rate; the shift is loss of selectivity, and only OLMo shows enrichment. |
| S3c OLMo layer 0 is not a token lookup | SAFE | Holds after merging case and space variants (0.60, 0.60, 0.13). |
| S4 Form and prediction jobs resist explanation in every model | SAFE WITH CAVEAT | Each family holds in two of three models after depth stratification; it is description-based. |
| S5 Last-layer lean with converging pole maxima | SAFE WITH CAVEAT | Reproduces exactly and lean neurons carry a signed mean, but the lead-up is model-specific and the output role is untested. |
| S6 Two poles are related, rarely co-fire, weak class concordance | SAFE WITH CAVEAT | Overlap is two to three times baseline (stop-list-dependent), "rarely co-fire" holds only for windows near 100 tokens (25% share a document), concordance is not significant in O7, and the 72-neuron reading is unverifiable. |
| S7 Concept twins across families | SAFE WITH CAVEAT | Every cited pole checks out, but they are anecdotes from 300 neurons and the depth test is fragile. |
| S8 Deep name-start neurons | SAFE WITH CAVEAT | Population-confirmed in all three, but in O7 it peaks at 0.53 to 0.72 depth and fades at the top, and the regex counts move by a factor of two. |
| S9 Broken-text poles in one band; the writer cannot fake them | DO NOT PUBLISH | The band is F's selection (regex hits span 0.0 to 0.97) and a writer text reached θ_s on five of the eleven testable poles. |
| D1 Qwen early layers | SAFE WITH CAVEAT | Counts are exact; say "near-silent, one-signed neurons whose gate never opens on this corpus", not "damaged" or "dead", and note the 551 position-0 poles and the corpus limit. |
| D2 Gemma first-token sink | SAFE WITH CAVEAT | Counts exact; the layers are at 26% to 32% depth, not the middle, and position 0 means different things with and without BOS. |
| D3 OLMo activation growth | SAFE | Monotonic from L2 and explained by OLMo 3 applying the MLP to the un-normalised residual stream. |
| D4 OLMo handles code low | SAFE WITH CAVEAT | Description-based; my regex and a narrow one agree (a broad one that also catches "function word" does not), and per-language medians rest on 12 to 25 poles. |
| D5 Non-English at both ends in Qwen and Gemma | SAFE | Population-confirmed with non-Latin-script tokens. |
| D6 Digits are a tokenizer story | SAFE WITH CAVEAT | Half of it is token frequency; the population still shows fewer digit poles in O7. |
| D7 Only Qwen splits one prefix two ways | DO NOT PUBLISH | Population share is 3.0% Q4B, 3.7% G4, 4.8% O7. |

Counts: SAFE 4, SAFE WITH CAVEAT 12, DO NOT PUBLISH 3.

Three findings I would defend in front of a frontier-lab audience:
1. **S5**, the last-layer lean: instrument-free, exact on all 1.05 million neurons, present in three families, and the histograms show a signed offset in lean neurons.
2. **S8**, deep capitalized word-start neurons: confirmed on every neuron without any explainer, rising in the upper layers of all three models.
3. **S1**, loudness and legibility: survives depth, explainer and RARE controls with pole-resampled intervals, stated with its size (AUC about 0.6) and its meaning (the writer reaches a third of the max more often).

Most likely to be an artifact: **S3b**, "deep neurons fire on function words". The counts are right, but the corpus base rate explains them in two of the three models.

## Closing

F is careful with numbers: every population figure and every class-based sample figure reproduces, and its pole ids and most of its quotes are exact. Its weak points are interpretive. Three claims read more into the data than it carries (the function-word shift, the broken-text band, the Qwen-only prefix split), and several hooks state "in all three models" where the per-model intervals say two. The fixes are cheap: report base-rate enrichment for firing-token types, give intervals per model, rename Qwen's early-layer neurons, correct the two altered quotes, name the checkpoints exactly (Qwen3-4B-Base; OLMo-3 stage-1 final), and lead with the findings the population confirms on its own: the last-layer lean, the deep name-start neurons, the non-English U shape and the loudness effect.
