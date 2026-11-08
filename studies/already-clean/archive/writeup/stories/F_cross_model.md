# F. What three model families share: a cross-model reading of 1,800 poles and 1.05 million neurons

Angle: commonalities. What holds in Qwen3-4B (Q4B, 36 layers), G4 (34 layers) and OLMo 3 7B (O7, 32 layers) despite different families, sizes, tokenizers and training data. This report builds on the earlier three-model report (C) and avoids its stories: near-silent Qwen layer-1 poles, the sign-blind vocabulary line, copy jobs, carriage returns, forward-looking statements, the Russian spam neuron, gambling, AI-style prose, sourdough and lava, zero suppressors, next-word language rising with depth.

## Sources and conventions

- **[M]** `/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w5/verdict/metrics.json`.
- **[P:Q4B] [P:G4] [P:O7]** `/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w5/verdict/samples/q1_<cell>_poles.jsonl`, 600 rows each. Fields: `pole`, `layer`, `class`, `explanation`, `explainer_saw`, `texts`, `blind_max_score`, `explainer_model`.
- **[H:q4b] [H:g4] [H:o7]** population statistics for every neuron, `/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1/{q4b/1/out/q4b, g4/1/out/g4, o7_t15/1/out/o7_t15}/stats/layer_LL.npz`. Fields `max_abs`, `q999`, `density`, `n_pole_tokens`, `top_same_token_frac`, `sign_counts`, `pos0_share`, each shaped (neurons, 2). Index 0 is the + pole, index 1 the - pole. I checked this against Q4B L05N2328: [H:q4b] gives max 6.866 and 4.425, the same as the + and - headers in [P:Q4B].
- **[HM]** the `manifest.json` beside each stats directory: tokens scored 52,442,093 (Q4B), 53,512,674 (G4), 50,515,866 (O7). The G4 manifest names the checkpoint `google/gemma-3-4b-pt`.
- **[S]** `/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/SPEC_D1.md`. **[C]** `writeup/stories/C_three_models.md` in the same study.

Definitions I use. Relative depth is layer divided by the layer count (36, 34, 32). "Within-layer percentile" of a pole is the share of all poles in that layer (both signs of every neuron, from [H]) with a smaller value. Density share is `density` divided by tokens scored; `density` counts tokens with |a| at least 1% of the pole's own max ([S] §2.1). "Single-token concentration" is `top_same_token_frac`: the share of a pole's top 50 that is its most common token. Firing-token types come from my classifier over the ⟦ ⟧ marks in `explainer_saw` (30 contexts per pole). "Function word" means a leading-space word in a 90-word stop list I wrote. "Word piece" means an alphabetic mark with no leading space right after a letter. Counts from regexes or my classifier are labelled "my count". All quintile tables use relative depth 0 to 0.2, 0.2 to 0.4, and so on.

---

## The ranked commonalities

### 1. Loud, sharp poles are the legible ones, in all three models

**Hook:** Rank a layer's poles by how hard they ever fire. The louder half gets "one job" far more often, in every model and under both explainers.

**What the data shows.** For each sampled pole I looked up where its peak |a| sits among all poles of its own layer. In all three models, the share of ONE rises from the quietest quarter to the loudest. The mirror image holds for density: poles that sit above 1% of their own max on many tokens are the least legible. The two measures are nearly one variable (correlation -0.81 to -0.85). Poles with a sharp peak, where the max sits far above the 99.9th percentile, also do better. The effect is not an artifact of the score. Every text is scored against the pole's own max ([S] §2.2), so a quiet pole has a lower absolute bar to clear, not a higher one. It is also not the explainer split: it holds inside Opus-explained and inside Sonnet-explained poles of each model. The poles the explainer gave up on are the extreme case. In each model they sit near the bottom of their layer in loudness and near the top in density, and the explainer's own words say so: "Diffuse, context-heavy pole with no shared token" (Q4B L20N7249-), "The pole fires, weakly almost everywhere" (G4 L31N7089-), "A diffuse, dense pole with no shared token or topic" (O7 L16N2846+).

**Evidence** (my count, joining [P:*] `class` and `pole` to [H:*] `max_abs`, `density`, `q999`; RARE excluded):
- ONE share by within-layer max |a| quartile, quietest to loudest: Q4B 0.479 (n=163), 0.636 (140), 0.630 (146), 0.692 (143); G4 0.425 (153), 0.541 (146), 0.607 (150), 0.682 (151); O7 0.359 (145), 0.500 (144), 0.553 (159), 0.651 (152).
- ONE share by within-layer density quartile, sparsest to densest: Q4B 0.659, 0.665, 0.611, 0.494; G4 0.678, 0.628, 0.514, 0.439; O7 0.694, 0.506, 0.493, 0.389. Correlation of the two percentiles: -0.811, -0.851, -0.819.
- Split by explainer (`explainer_model`), ONE share for poles below vs at or above the layer median of max |a|: Q4B Opus 0.565 vs 0.706, Sonnet 0.500 vs 0.521; G4 Opus 0.572 vs 0.665, Sonnet 0.221 vs 0.574; O7 Opus 0.446 vs 0.638, Sonnet 0.391 vs 0.506.
- ONE share by quartile of q999 / max (sharpest peak first): Q4B 0.696, 0.642, 0.534, 0.547; G4 0.680, 0.540, 0.487, 0.547; O7 0.680, 0.547, 0.440, 0.407.
- Poles with `unk_reason` "explainer said UNK" (10 / 9 / 13): mean within-layer max percentile 0.15 / 0.27 / 0.19 and density percentile 0.74 / 0.81 / 0.67. ONE poles: 0.53 / 0.55 / 0.56 and 0.48 / 0.46 / 0.44.

**Why an AI researcher cares:** a free, model-agnostic statistic (peak loudness within the layer) predicts which neurons an LLM explainer can describe, so "how clean is a network" depends on how far down the loudness ranking you go.

---

### 2. A shared depth map: the densest, least token-bound layers sit at 40% to 47% depth, and that is where explanations fail

**Hook:** Read every neuron in three unrelated models and the same landmark appears: a dense, token-free middle near 0.4 relative depth.

**What the data shows.** Across all 1.05 million neurons, the median pole's density share rises from the bottom, peaks between 41% and 47% of depth, and falls toward the top. The peak heights are close: about 21% to 23% of tokens. In the same region the share of poles whose top 50 is dominated by one token reaches a low. It is clean in Q4B and O7. G4 has a local low there and a second low near 85% depth. Above the middle, single-token concentration climbs again to about 50% to 70% depth and then eases off. The sampled ONE share follows the same shape: high at the bottom, lowest in the middle, a partial recovery at 60% to 80% depth. That recovery holds inside Opus-explained poles, so it is not the Sonnet handover. The writer's strong-positive rate has the same dip and recovery. The population data uses no explainer at all, so the middle being hard is at least partly a property of the networks.

**Evidence:**
- Median density share by layer ([H:*] `density` / [HM] tokens; my count). Peak layer: Q4B L17 (0.472 depth) 22.82%; G4 L14 (0.412) 20.47%; O7 L14 (0.438) 20.96%. Top five layers: Q4B 14, 16, 17, 18, 19; G4 10, 12, 13, 14, 16; O7 11 to 15. By relative-depth decile, Q4B 2.95, 8.19, 13.67, 19.66, 22.19, 21.15, 16.52, 10.31, 6.15, 4.53 (%); G4 3.57, 5.53, 16.22, 18.45, 17.84, 15.49, 10.34, 9.71, 15.61, 10.79; O7 12.03, 13.51, 17.62, 20.77, 20.80, 18.24, 16.93, 15.80, 14.75, 14.50.
- Share of poles with `top_same_token_frac` at least 0.5, by decile: Q4B 0.358, 0.268, 0.144, 0.079, 0.042, 0.076, 0.144, 0.133, 0.110, 0.108; G4 0.390, 0.264, 0.155, 0.120, 0.088, 0.132, 0.177, 0.111, 0.057, 0.089; O7 0.136, 0.087, 0.050, 0.032, 0.064, 0.101, 0.100, 0.078, 0.053, 0.069. Lowest single layer: Q4B L15 (0.417 depth) 0.040; O7 L10 (0.312) 0.029; G4 L29 (0.853) 0.051, with a local low of 0.080 at L14 (0.412).
- ONE share by quintile, Opus-explained poles only ([P:*] `class`, `explainer_model`): Q4B 0.742 (n=124), 0.746 (122), 0.464 (112), 0.546 (97); G4 0.730 (122), 0.605 (124), 0.521 (121), 0.625 (88); O7 0.667 (132), 0.474 (116), 0.477 (128), 0.580 (50). The top quintile has no Opus poles in any model.
- Strong share of writer positives by quintile ([P:*] `texts.pos.strong`): Q4B 0.83, 0.81, 0.49, 0.64, 0.65; G4 0.86, 0.61, 0.56, 0.70, 0.46; O7 0.77, 0.55, 0.54, 0.67, 0.45.
- Shape constants shared by all three ([H:*], all poles): median q999 / max is 0.136, 0.131 and 0.170, so a typical pole's peak is 6 to 8 times its 1-in-1,000 level. Neurons are never "off": the share of neurons whose + pole holds 40% to 60% of all tokens (`n_pole_tokens`) is 0.775, 0.805 and 0.924.

**Why an AI researcher cares:** three independently trained families put their densest, least lexical computation at the same relative depth, and an LLM explainer struggles exactly there; a depth-aware sampling plan would change any headline "clean share".

---

### 3. The firing token changes from word pieces to "the", "a", "of" and "to", measured on the contexts, not the descriptions

**Hook:** Count what token each pole actually fires on. Shallow neurons fire inside words; deep neurons fire on function words, in all three models.

**What the data shows.** [C] showed that deep descriptions talk about the next word, and that this partly reflects which model wrote them. The 30 marked contexts per pole are data, not prose, so they test the same shift without the explainer. In the bottom fifth of each model, word pieces are a large share of firing tokens and function words are rare. By the top fifth this has flipped. The deep function words are the same short list in every model: the, a, of, to, and, in. This is the summary motif of [S] §2.2 seen from the data side: deep neurons fire on the slot before the word they are about. At layer 0 the population data adds a twist. Q4B and G4 layer-0 neurons are token lookups (over half of poles have half their top 50 on one token). OLMo's layer 0 is not.

**Evidence** (my classifier over `explainer_saw` marks, [P:*]):
- Word pieces, bottom fifth to top fifth: Q4B 1,550 of 3,804 marks (40.7%) to 387 of 3,720 (10.4%), with a low of 215 of 3,240 (6.6%) at 0.6 to 0.8; G4 715 of 3,720 (19.2%) to 199 of 3,120 (6.4%), low 194 of 3,720 (5.2%) at 0.4 to 0.6; O7 1,059 of 3,960 (26.7%) to 298 of 3,420 (8.7%).
- Function words, bottom fifth to top fifth: Q4B 213 of 3,804 (5.6%) to 757 of 3,720 (20.3%), peak 848 of 3,240 (26.2%) at 0.6 to 0.8; G4 358 of 3,720 (9.6%) to 684 of 3,120 (21.9%); O7 36 of 3,960 (0.9%) to 1,257 of 3,420 (36.8%).
- Most common function words in the top fifth: Q4B "the" 73, "in" 70, "a" 56, "to" 52; G4 "and" 79, "the" 75, "of" 60, "a" 58; O7 "the" 110, "to" 101, "a" 84, "of" 77.
- Layer 0, share of poles with `top_same_token_frac` at least 0.5 ([H:*]): Q4B 0.548, G4 0.555, O7 0.115.

**Why an AI researcher cares:** the detector-to-predictor transition is visible in raw firing positions across three families, so it is a property of the models, not only of how an LLM likes to phrase deep neurons.

---

### 4. Jobs about form, position and prediction resist explanation in every model

**Hook:** In all three models, an Opus description that names "what comes next" or a line break passes far less often than one that names a thing.

**What the data shows.** I grouped Opus-written descriptions by what they claim. Next-word descriptions and layout descriptions (newlines, indentation, blank lines, headings, bullets) are ONE less often in all three models. The same order shows in the data side: poles whose most common firing token is a word piece are the most legible, then whole words, then function words, then punctuation and whitespace. A job carried by form or position needs the writer to rebuild the exact surrounding structure from one sentence. A job carried by a word only needs the word. One explainer-UNK pole shows the type: Q4B L16N9243- "Fires on the first word or symbol of a new structural segment, especially right after a markdown heading marker".

**Evidence** (my count, [P:*] `explanation`, `class`, `explainer_model`):
- Opus only, next-word language (my regex: predict, right before, just before, expected next, next word, about to, and similar): ONE 0.515 (n=134) vs 0.682 for other Opus poles in Q4B; 0.500 (n=162) vs 0.686 in G4; 0.491 (n=114) vs 0.567 in O7.
- Opus only, layout language: ONE 0.344 (n=32) vs 0.655; 0.417 (n=24) vs 0.631; 0.269 (n=26) vs 0.565.
- ONE share by the pole's most common firing-token type (my classifier; word piece includes word starts without a space): Q4B word piece 0.76 (n=95), whole word 0.61 (263), function word 0.56 (89), punctuation 0.48 (62), newline or space 0.50 (30); G4 0.69 (71), 0.60 (278), 0.54 (103), 0.44 (88), 0.29 (17); O7 0.69 (68), 0.53 (351), 0.45 (107), 0.33 (30), 0.33 (18).

**Why an AI researcher cares:** "one sentence, five texts" measures how nameable a job is as much as how many jobs a neuron has, and the unnameable share grows exactly where models do their prediction work.

---

### 5. In the last layers, a neuron's two signs grow equally strong, and the neuron leans to one side

**Hook:** At the top of all three models, up to half the neurons spend most tokens on one sign, while their two poles peak at nearly the same height.

**What the data shows.** In the middle of each model, a neuron's activation is positive on close to half of all tokens; the job lives in the tail. In the last few layers of all three, this changes. The share of neurons that are positive on under 40% or over 60% of tokens climbs steeply and is highest (outside Q4B's damaged layers 1 to 3) in the final layer. In the final layer, 5% of neurons are positive on about 8% to 10% of tokens and another 5% on about 89% to 93%. At the same time the two poles' peaks converge: the stronger pole is about 1.7 to 2.0 times the weaker in the middle and about 1.3 to 1.5 times at the top. A neuron that sits on one side most of the time works partly like a standing push on the output. That reading is mine, not a study measurement. It fits [C]'s finding that both poles of deep neurons are often described as predicting the same word.

**Evidence** (my count, [H:*] `n_pole_tokens`, `max_abs`, [HM] tokens):
- Share of neurons whose + pole holds under 40% or over 60% of tokens, middle layer then last four layers: Q4B L18 0.07; L32 to L35 0.25, 0.30, 0.46, 0.56. G4 L17 0.15; L30 to L33 0.14, 0.27, 0.26, 0.46. O7 L16 0.05; L28 to L31 0.13, 0.18, 0.25, 0.34. Highest anywhere: G4 L33 (0.46, against 0.40 at L0); O7 L31 (0.34); Q4B L2 (0.82, a layer with 653 dead poles), then L35 (0.56).
- Final layer, 5th and 95th percentile of the + pole's token share: Q4B 0.081 and 0.927; G4 0.100 and 0.899; O7 0.101 and 0.891.
- Median ratio of the stronger to the weaker pole's max, live neurons: middle layer Q4B 1.716, G4 1.767, O7 1.981; last four layers Q4B 1.517, 1.407, 1.420, 1.460; G4 1.320, 1.358, 1.318, 1.421; O7 1.391, 1.398, 1.402, 1.376.
- Sanity check from symmetry: medians of the + and - pole max agree within 1% in every decile (for example Q4B top decile 57.212 vs 57.150), as they must, since flipping the sign of a neuron's up-projection row and down-projection column leaves a gated MLP unchanged (my reasoning).

**Why an AI researcher cares:** "a pole is a feature" fits middle layers, but last-layer neurons look more like signed dials around an offset, so the unit of explanation may need to change near the output in every family.

---

### 6. The two poles of one neuron are cousins: related about half the time, rarely in the same passage

**Hook:** The positive and negative side of a neuron describe related things about three times more often than two neurons from the same layer, yet almost never fire on the same text.

**What the data shows.** The essay treats the negative side as an echo or a second job. Across all three models, the descriptions of a neuron's two poles share content words about three times more often than random pairs from the same layer. Reading 72 random neurons (24 per model, my judgment), about half have two poles about related things. Some pairs are elegant: a floating-point keyword and the first digit after a decimal point (G4 L05N4009); cell proliferation and tree branching (O7 L04N7402); "actual results may differ materially" and "whether promised things are actually carried out" (O7 L17N5377); a list word expected to be followed by "or" against "Predicts that a comma comes next" (G4 L24N3271). Shallow pairs are more often spelling siblings, like the "ition" of Pract-ition-ers and the "ag" of Pr-ag-matic (Q4B L04N6541). Deeper pairs more often do the same job on both sides. Still, the two poles rarely meet: only 4% to 9% of neurons have even one passage in both top-30 lists. Whether one pole is ONE says little about the other.

**Evidence:**
- Mean Jaccard overlap of content words, same-neuron pole pairs vs same-layer pairs from other neurons (my count over `explanation`): Q4B 0.0424 vs 0.0126; G4 0.0393 vs 0.0143; O7 0.0416 vs 0.0130.
- My reading of 24 random neurons per model (seed 7), categories same job or same slot (S), related subject (R), same token shape only (F), unrelated (U): Q4B 5, 3, 5, 11; G4 4, 11, 3, 6; O7 5, 7, 3, 9. Shallow neurons (depth under 0.33, n=33): S 2, R 10, F 7, U 14. Deeper (n=39): S 12, R 11, F 4, U 12.
- Neurons whose + and - top-30 share at least one passage (40-character window match over `explainer_saw`, my count): Q4B 20 of 296, G4 27 of 300, O7 11 of 300.
- Class concordance: P(- pole ONE given + pole ONE) vs given + pole not ONE: Q4B 0.659 vs 0.538; G4 0.631 vs 0.485; O7 0.528 vs 0.482.
- Quotes ([P:G4], [P:O7]): G4 L05N4009+ "Fires on the floating-point type keyword in code"; G4 L05N4009- "Fires on the digit token right after a decimal point"; O7 L30N1510+ "Fires on a verb or preposition that is about to be followed by an article"; O7 L30N1510- "Fires on the word right after a determiner like "a", "an", "one" or "the" ... when a singular countable noun phrase is still unfinished".

**Why an AI researcher cares:** a neuron's sign is neither noise nor a free second slot; in all three families it usually encodes a nearby concept, which matters for anyone counting features per neuron or pruning one sign.

---

### 7. Concept twins across families: React hooks, "respectively", foreign keys, divergence, mojibake

**Hook:** In random samples of only 300 neurons per model, the same narrow jobs turn up in two or three unrelated models.

**What the data shows.** I compared every description with every description from the other two models and read the closest pairs. Several narrow jobs recur. React's `useEffect` or `useState` has a pole in all three models, always in the bottom quarter, and in two of them it shares a neuron with an unrelated job (the surname Harris in Q4B; the word "Exactly" in O7). The word "respectively" has a pole in Q4B and two in O7, both at relative depth 0.44: Qwen fires on "respectively" itself, while one OLMo neuron fires on the first item of the pair with one sign and the last item before "respectively" with the other. SQL foreign keys, divergence, mojibake, random-number generators, "apart from / except for", page-range dashes and code indentation each appear in two of the three models. Yet a matching job does not imply a matching depth. The 100 most similar cross-model pairs sit only slightly closer in relative depth than random cross-model pairs. When a twin sits at two depths, the shallow one can read the word while the deep one predicts what follows: "keep" itself at 0.25 in Q4B against the object after "keep" at 0.72 in O7; a layer-0 detector for " macroeconomic" in G4 against "macro" in the sense of large scale at 0.69 in O7. Those are two examples, not a count.

**Evidence** ([P:*] `pole`, `layer`, `class`, `explanation`, `texts`):
- React: Q4B L07N4960+ (0.19, UNREL: Harris and "useState"), Q4B L08N6028- (0.22, UNREL, JSX `key`), G4 L07N10202- (0.21, ONE, positives 0.76, 0.92, 0.85, 0.69, 0.75, `blind_max_score` 1.29), O7 L01N10284- (0.03, UNK: "useEffect" and "Exactly").
- "respectively": Q4B L16N764- (0.44, ONE) "most strongly on "respectively" after two parallel items"; O7 L14N5001+ (0.44, ONE) "right before the word "respectively" would come"; O7 L14N5001- (0.44, UNK) "Fires on the first item of a paired list where a matching second item and then "respectively" is about to follow".
- SQL foreign keys: Q4B L11N5238+ (0.31, ONE, " KEY" after FOREIGN), G4 L08N7823+ (0.24, ONE, REFERENCES after FOREIGN KEY). Divergence: Q4B L06N7457- (0.17, ONE), O7 L04N3542- (0.12, ONE). Mojibake: Q4B L04N5781+ (0.11, ONE), O7 L00N1929- (0.00, ONE, the "™" of "вЂ™"). Random-number generators: Q4B L06N8698+ (0.17, ONE), O7 L03N9245+ (0.09, ONE), O7 L06N4452- (0.19, UNREL with "rhyme"). Exception markers: Q4B L06N2242+ (0.17), G4 L12N3344- (0.35). Page-range dash: Q4B L22N7061+ (0.61), G4 L06N2449+ (0.18). Code indentation: Q4B L18N621- (0.50), G4 L19N4173+ (0.56). Keep: Q4B L09N9062+ (0.25), O7 L23N6535+ (0.72). Macro: G4 L00N6518- (0.00), O7 L22N4270+ (0.69).
- Depth test (my TF-IDF cosine over `explanation`, all cross-model pairs): mean |difference in relative depth| 0.274 for the top 100 pairs (cosine at least 0.209), 0.332 for all cross-model pairs; Opus-only top 100: 0.222 vs 0.249.

**Why an AI researcher cares:** some features look universal across families at the level of a single neuron, which is the precondition for transferring a neuron atlas between models; their depth is not.

---

### 8. Deep "name-start" neurons: the first piece of a rare name

**Hook:** Late in all three models sit neurons that fire on "Han", "Wool", "Gul" or "Vi" when that piece opens a rare name, and both signs of one neuron can take different letters.

**What the data shows.** Between about 60% and 95% of depth, each model has poles keyed to the opening piece of a split proper name: Han|ford and Han|over, Ra|quel, Wool|ley and Wool|worth, Gul|benkian, Dum|ortier, Vi|orel, Som|erville. Their top 30 are dominated by one token, and the writer drives them hard from a one-line description. In three neurons both signs do it for different openings: Q4B L27N6887 is "Han" on one side and "Ra" on the other; G4 L23N5356 is "sø"/"sel" against "Som"; G4 L31N9849 is "De"/"Da" against "K"/"C" first names. 7 of the 11 in the evidence list were explained by Opus, so they are not only Sonnet's word-piece habit (Sonnet does use "word piece" much more in deep layers; see the last section). These poles are part of why single-token concentration rises again above the middle in story 2. They look like the machinery a model needs to continue a rare name it has started.

**Evidence** ([P:*] `explainer_saw` token counts in the 30 contexts, `texts.pos` scores, `explainer_model`):
- Q4B L27N6887+ (0.75, Opus, ONE): " Han" 23 and "Han" 6 of 30; positives 0.91, 0.07, 0.89, 0.92, 0.88. Description: "Fires on the capitalized token "Han" (with or without a leading space) when it begins a proper name". L27N6887- (Opus, ONE): " Ra" 20 and "Ra" 9; positives 0.66 to 0.84.
- G4 L22N1709+ (0.65, Opus, ONE): " Wool" 25 and "Wool" 5; positives 0.73 to 0.90; "Fires on the token "Wool" when it begins a proper name rather than the word wool". G4 L21N9810- (0.62) " Dum" 15, " Dumb" 6; G4 L25N5115- (0.74) " Vi" 11, "Vi" 6; G4 L23N5356- (0.68) "Som" 12, " Som" 7; G4 L24N6751+ (0.71) " Sher" 10, "Sher" 9; G4 L31N9849- (0.91, Sonnet) " K" 12, " C" 6, positives 1.05, 1.06, 0.84, 0.94, 0.66. All ONE.
- O7 L23N809- (0.72, Sonnet, ONE): " Gul" 23 of 30; positives 0.95, 1.04, 1.00, 0.95, 0.88; "Fires strongly on the word-piece "Gul" (or "ul" after G) when it begins a rare proper name or place". O7 L25N8888- (0.78, ONE) " Sc" 10, " fl" 9. Q4B L33N5981- (0.92, ONE) " T" 25.
- My regex for name-start or word-start language finds 23 / 24 / 29 poles; at relative depth 0.6 or more: 18 / 17 / 12 (O7 also has 14 in its bottom fifth).

**Why an AI researcher cares:** spelling is not only an early-layer job; all three families keep letter-level name machinery near the output, a concrete target for studying how models copy and complete rare names.

---

### 9. Every model has poles described as watching for broken text, and the writer cannot fake broken text

**Hook:** Mostly between 20% and 40% of depth, all three models have poles that the explainer reads as firing where text stops making sense, and a fluent model's attempts at nonsense barely move them.

**What the data shows.** This is a different family from [C]'s polished AI-style prose. These descriptions are about incoherence: sentence ends in spun spam where each sentence jumps topic, a collocation wrongly spliced into a sentence, a fragment cut into the middle of a line, readable text resuming after garbled PDF math. They cluster in the same band in all three models. Almost all are UNK. The failures are informative. Sonnet's word salad for O7 L13N2659- reads "The weather of the quarterly report was blue with ancient hamburgers. Dolphins have always negotiated the tax code of Belgium." It scored 0.07. For G4 L10N9543+ the writer produced keyword lists ("Best weight loss pills diet fast burn fat cheap buy online garcinia...") at 0.04, while the real contexts are grammatical-looking scrambles ending in "?!". So either the descriptions are wrong, or genuine machine-spun text has a signature that a fluent writer does not reproduce. The study cannot tell which.

**Evidence** ([P:*] `explanation`, `class`, `texts`; the selection is my reading of regex hits for spun, incoherent, spliced, garbled):
- G4: L10N9543+ (0.29, UNK, positives 0.02 to 0.09), L10N9543- (0.29, ONE, the one pass, with juxtaposed-sentence texts scoring 0.21 to 0.45), L11N1220- (0.32, UNK, writer refused by the provider's content filter, [P:G4] `unk_reason`), L12N4911+ (0.35, UNK), L13N5135+ (0.38, UNK) "Fires on the second word of a stock two-word collocation that has been wrongly spliced into a sentence", writer text "check your credit history Card" 0.01.
- O7: L07N8340- (0.22, UNK), L10N5687+ (0.31, UNK, spintax "|"), L12N8337+ (0.38, UNK), L13N2659- (0.41, UNK) "fires most strongly on sentence-ending periods in incoherent, spun or machine-generated essay-mill text, where each sentence jumps to an unrelated topic", positives 0.07 to 0.12, `blind_max_score` 0.04.
- Q4B: L04N5781- (0.11, UNK, a stray "l" in spun spam), L12N1451- (0.33, UNK, readable text resuming after garbled math, positives 0.02 to 0.07), L13N2349- (0.36, UNK) "Fires on a word that abruptly breaks the flow of the text".
- Totals for these 12 poles: 11 UNK, 1 ONE.

**Why an AI researcher cares:** models of three families seem to track text quality in the same depth band, and LLM-written probes are a poor instrument for exactly the text that LLM-generated spam contains.

---

## Striking differences

- **Qwen's damaged early layers, at population scale.** [C] found the near-silent layer-1 poles in the sample. [H:q4b] shows the scale: 3,521 poles with max |a| exactly 0 and 6,490 below the RARE rule, 3,515 and 6,472 of them in layers 1 to 4 (L1 alone: 2,467 and 3,786 of 19,456). G4 has 3 and 56; O7 has 0 and 2. Q4B layers 1 to 3 are also dominated by the double-negative regime: mean double-negative share of + poles 0.481, 0.808, 0.477 ([H:q4b] `sign_counts`), against at most 0.015 in G4 and 0.051 in O7 at any layer.
- **Gemma's first-token sink lives in three middle layers.** Poles with half or more of their pre-exclusion top 50 at window position 0 ([H:g4] `pos0_share`): 1,125, 3,215 and 3,334 of 20,480 in G4 layers 9, 10, 11. O7 never exceeds 10 in a layer. Q4B peaks at 1,276 in L2 and has 146 in L35.
- **OLMo's layer 0 is not a token lookup, and its activations grow every layer.** Layer-0 median max |a|: Q4B 1.292, G4 10.504, O7 0.299. O7's median max rises in every layer from L2 (0.249) to L31 (30.575), about 123-fold. Q4B rises from L17 (4.872) to L35 (64.944). G4 goes up and down (27.459 at L3, 1.347 at L14, 9.426 at L24, 1.976 at L29, 14.648 at L33) ([H:*] `max_abs`). I did not check the norm placement that would explain this.
- **OLMo handles code low in the stack.** Share of descriptions naming code (my regex), bottom fifth to top fifth: O7 0.30 to 0.05; Q4B 0.20 to 0.25; G4 0.23 to 0.20. Median relative depth of Python, JavaScript and C# poles: O7 0.19, 0.14, 0.16; Q4B 0.39, 0.60, 0.33; G4 0.29, 0.44, 0.41.
- **Non-English sits at both ends in Qwen and Gemma, only at the bottom in OLMo.** Share of descriptions naming a language or script, top fifth: Q4B 0.22, G4 0.15, O7 0.04; overall 0.14, 0.11, 0.06. "Chinese" appears in 31 Q4B descriptions, 9 in G4, 9 in O7 (my regex). O7 reads its own training mix here; the others do not.
- **Digits are a tokenizer story.** Every digit mark in Q4B (1,011) and G4 (1,399) is a single digit. O7 has only 197 digit marks, of lengths 1, 2 and 3 (93, 43, 61), so digit neurons are rarer there (my count, `explainer_saw`).
- **Only Qwen's shallow poles split one prefix two ways.** Neurons below 0.2 depth whose two poles share at least 3 identical preceding word fragments: Q4B 5 of 66 (for example L03N6845 "con"/"cont" 20 times, L05N8095 "E" for Eck- and E-ston-ia), G4 0 of 62, O7 1 of 66 (my count).

## Numbers that surprised me

- **The same peak density in three families.** 22.82%, 20.47% and 20.96% of tokens, at 47%, 41% and 44% of depth ([H:*]).
- **Final-layer neurons lean hard.** In Q4B's last layer, 1 in 20 neurons is positive on fewer than 8.1% of 52 million tokens and another 1 in 20 on more than 92.7%.
- **"I don't know" is quiet.** The poles the explainer refused to describe sit at a mean within-layer loudness percentile of 0.15 in Q4B.
- **Sonnet's loudness gap in Gemma.** For Sonnet-explained G4 poles, ONE is 0.221 below the layer median of max |a| and 0.574 above it.
- **"respectively" twice at exactly 0.44 depth**, in two families (Q4B L16N764-, O7 L14N5001).
- **Dolphins and the tax code of Belgium.** A fluent nonsense sentence scored 0.07 on a pole described as firing in incoherent text (O7 L13N2659-).
- **Two signs, one passage, rarely.** Only 11 of 300 OLMo neurons have any passage in both poles' top 30.
- **OLMo's layer 0 almost never fixates on one token.** 0.115 of poles, against 0.548 and 0.555 in Q4B and G4.
- **3,334 Gemma layer-11 poles** had half or more of their strongest contexts on the first token of a window before the study excluded it.

## What cuts against the story

- **One explainer family for all three.** Every description is Opus 5.5 or Sonnet 5.5 ([M] `kappa_models`), and every test text is Sonnet 5.5 ([S] §2.2, model policy amendment). Shared description patterns (stories 4, 6, 7, 8, 9) may be the explainer's vocabulary rather than the networks'. One example I found: the rise of "word piece" language in deep layers is mostly Sonnet. Within Opus at 0.6 to 0.8 depth it is 0.06 (Q4B), 0.19 (G4), 0.02 (O7); Sonnet in the same band writes it 0.36, 0.31, 0.25 (my regex). Stories 2, 3 and 5 rest on population statistics and raw firing tokens, which no explainer touched.
- **Depth and explainer are confounded.** Opus explained nearly all of the bottom two thirds and Sonnet most of the top third ([S] §2.2: deep-tercile Opus/Sonnet 65/135, 54/140, 14/186). The top quintile has no Opus poles. I report Opus-only figures where I could.
- **One corpus, OLMo's.** All three read OLMo 3's own stage-1 mix ([S] §4.2). Q4B and G4 read text they were not trained on. Differences such as OLMo's scarcity of non-English and top-of-stack code neurons may reflect this.
- **The G4 checkpoint.** The harvest manifest names `google/gemma-3-4b-pt`. Claims about "Gemma" are about that checkpoint.
- **600 poles per model.** That is 300 neurons out of about 350,000 per model. The twins, name-start and broken-text families are anecdotes with pole ids, not rates.
- **My classifiers.** Firing-token types, the stop list, every regex family, the 40-character passage match and the TF-IDF twin search are mine. They miss paraphrases and catch some false hits. The 72-neuron two-pole reading is my judgment.
- **Histogram resolution.** Density and q999 come from a log-spaced histogram with 5.9% bins ([HM] `density_rule`; header lines in `explainer_saw`).
- **Loudness and density are nearly one variable.** Their within-layer percentiles correlate at -0.81 to -0.85, so story 1 is one finding seen twice, not two.
- **Some population facts are forced.** The equal medians of + and - pole maxima, and a median neuron positive on half the tokens, follow from sign symmetry; I use them as checks, not findings.
- **Reading, not mechanism.** The last-layer "standing push" in story 5 and the "complete the rare name" reading in story 8 are my interpretations. Neither was tested by intervention.
