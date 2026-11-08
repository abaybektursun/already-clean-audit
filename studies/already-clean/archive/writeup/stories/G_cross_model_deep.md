# G. Deeper across three models: what every neuron shows without an explainer

This report extends F (`writeup/stories/F_cross_model.md`). F read 1,800 sampled poles and some population statistics. Here I read the population files for every neuron of Qwen3-4B-Base (Q4B, 36 layers), `google/gemma-3-4b-pt` (G4, 34 layers) and OLMo 3 7B at `stage1-step1413814` (O7, 32 layers), about 1.05 million neurons, and I decode every pole's top 50 with the model's own tokenizer. Most findings use no explainer output at all. Two use the explainer's labels or descriptions and are marked so. I avoid repeating F; where a finding tests one of F's stories at population scale, I say which.

## Sources and conventions

- **[H:q4b] [H:g4] [H:o7]** stats files: `/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1/{q4b/1/out/q4b, g4/1/out/g4, o7_t15/1/out/o7_t15}/stats/layer_LL.npz`. Arrays used: `max_abs`, `q99`, `q999`, `density`, `n_pole_tokens`, `top_same_token_frac`, `pos0_share`, `sign_counts`.
- **[T:q4b] [T:g4] [T:o7]** top-50 files in the same directories, `topk/layer_LL.npz`: `tok`, `a`, `g`, `u`, shape (neurons, 2, 50).
- **[HM]** `manifest.json` beside them: tokens scored 52,442,093 (Q4B), 53,512,674 (G4), 50,515,866 (O7); `model_class` Qwen3ForCausalLM, Gemma3ForConditionalGeneration, Olmo3ForCausalLM.
- **[K]** corpus tokens and tokenizers: `/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1/q4b/1/out/corpus/corpus/tok/{qwen3,gemma3,olmo3}/{tokens.npy, doc_offsets.npy, tokenizer/tokenizer.json}`, and the corpus `manifest.json` there (token-array SHA-256 per tokenizer; the Q4B harvest manifest's `tokens_sha256` matches the qwen3 entry).
- **[HC]** harvest code: `/Users/abaybektursun/projects/interp_research-neurons-harvest/infra/pg/neurons_harvest.py` (the job file named in the brief, `infra/aws_pg/jobs/neurons_harvest.py`, only calls it).
- **[P:Q4B] [P:G4] [P:O7]** `/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w5/verdict/samples/q1_<cell>_poles.jsonl`. **[S]** `studies/already-clean/SPEC_D1.md`. **[F]** the F report.

Facts about the files that matter for every number below:

- **`tok` is a corpus position, not a token id.** [HC] `TopK.update` stores "tok: [T] global token index", and `same_token_frac` maps it through `tokens`. In [T:q4b] layer 5, `tok` runs from 15 to 52,467,533. I get the token id as `tokens.npy[tok]`, its left neighbour as `tokens.npy[tok-1]` and its right neighbour as `tokens.npy[tok+1]`. Window position is `tok % 2048` ([HM] `window` 2048; `n_tokens_windowed` 52,467,712 = 25,619 x 2,048 for Q4B). Document position uses `doc_offsets.npy`; I checked that the token before every document start is the EOS id in all three token arrays (34,789 of 34,789).
- **Signs.** Axis 0 is the + pole (a > 0), axis 1 the - pole ([HC] module docstring). `sign_counts` is ordered (g>=0, u>=0), (g>=0, u<0), (g<0, u>=0), (g<0, u<0) ([HC] `sign_counts`). Measured: + poles only ever hold the first and last category and - poles only the middle two, in every layer of every model. So the "double-negative" category (g<0, u<0) exists only on + poles. Its mirror on the - pole is (g<0, u>=0). I use the sign-symmetric **gate-negative share**: the share of a pole's top 50 where the gate is negative (categories 3 + 4).
- **RARE** is density < 50 or max |a| = 0 ([S] §2.3). **Dead** is max |a| = 0.
- **Relative depth** is layer / L in tables and quintiles, as in F. For curve comparisons I interpolate each per-layer curve onto 41 points of layer / (L - 1).
- **Token types** (my classifier, applied to the decoded token string and its neighbours): *whitespace* (only spaces, tabs or newlines); *punctuation* (every non-space character is Unicode P or S); *digit* (only Unicode Nd); a *letter token* is letters or marks, optionally after one apostrophe; among letter tokens, *function word* is a whole word (leading space, or left neighbour not ending in a letter, and the right neighbour does not continue the word) whose lowercase form is one of the 20 listed in the brief (the, a, an, of, to, and, in, on, for, with, that, is, it, as, at, by, from, or, be, this); *word start* is any other letter token with a leading space or a non-letter left neighbour; *word piece* is a letter token with no leading space right after a letter. *Other* is the rest, mostly mixed tokens like ".blogspot" or "_path".
- **Nulls for curves.** Mean Spearman correlation over the three model pairs on the common grid, against 1,000 draws that shuffle layers within each model and 1,000 draws that rotate each model's curve by a random number of layers (circular shift, which keeps smoothness). Ties get average ranks.
- Scripts ran inline under `uv run --no-project --with numpy --with tokenizers --with scipy python`; none were saved (read-only rules). Every number is my count from the files named.

---

## Ranked findings

### 1. Qwen's "dead" early neurons are gate-closed units, and thousands of them share the same strongest tokens

**Hook:** Qwen's near-silent layers 1 to 3 are not dead. Half their neurons keep the SiLU gate shut at every one of their strongest tokens, and their top-50 lists are copies of each other.

**What the data shows.** Call a neuron gate-closed when the gate g is negative at all 50 top tokens of both its poles. Q4B has 14,710 such neurons, almost all in layers 1 to 4. G4 has 144 and O7 has 41 in total. In Q4B layer 1, 4,788 of 9,728 neurons are gate-closed, and in layer 2, 6,449. A gate-closed neuron passes signal only through SiLU's small negative lobe, so its activation is tiny and its sign is set by the up projection u. The 3,521 dead poles in Q4B are this state seen from one side: no Q4B neuron is dead on both signs, and for all 3,521 the other pole is live on every one of the 52,442,093 scored tokens and sits in the gate-negative regime at all 50 of its top tokens. In layer 1 those live poles peak at a median |a| of 0.0047. So "damaged" ([F] striking differences) is the wrong word for most of these. They are always-on, constant-sign, very low-amplitude units.

Their strongest tokens are shared. Across all pole pairs from different neurons, Q4B layer 1 has 2,020,462 pairs that share at least 25 of their 50 top positions (the same corpus token, not just the same string), involving 4,630 neurons. One single position sits in the top 50 of 3,883 poles. In layers 1, 2 and 3, the positions held by at least 100 poles take 27.6%, 39.5% and 17.5% of all top-50 slots. The hubs are rare tokens: ".blogspot" in "1.bp.blogspot", the final "c" of "RePEc", " Bid" on freelance job pages. Outside Q4B layers 1 to 3 and G4 layers 10 and 11 (finding 4), no layer of any model has more than 4,544 such pairs.

This explains a pattern in the explainer's output. Of the 50 sampled Q4B poles in layers 1 to 3, 34 sit on gate-closed neurons; 21 of those 34 were labelled ONE, 8 RARE, 4 UNK, 1 TWO-REL. Three sampled layer-3 neurons (L03N8511, L03N9234, L03N8169) share 32 to 44 of 50 top positions between poles, and their descriptions all name the "c" of "RePEc" or "Bid" on Freelancer pages. Two layer-1 poles (L01N5391-, L01N8338+) share 44 of 50 positions; both descriptions list ".blogspot", ".Xna" and "oreferrer", which are the hub tokens. Their TF-IDF cosine is 0.753, the highest of any pair among the 1,788 non-RARE descriptions under my stop lists (finding 8).

**Evidence** ([T:q4b] `g`, `tok`; [H:q4b] `max_abs`, `sign_counts`, `n_pole_tokens`; [P:Q4B] `explanation`, `class`):
- Gate-closed neurons per layer, Q4B L0 to L4: 7, 4,788, 6,449, 3,114, 246; table T3 has every layer. Totals: Q4B 14,710, G4 144, O7 41.
- Dead poles, Q4B: L1 1,255 (+) and 1,212 (-); L2 321 and 332; L3 193 and 177; L4 15 and 10. Neurons dead on both signs: 0 in all three models. G4 has 3 dead poles, O7 none.
- All 3,521 Q4B neurons with a dead pole (layers 1, 2, 3, 4, 9, 10, 12): live-pole token share minimum 1.0 and no token with a = 0; live-pole gate-negative share minimum 1.0. Q4B L1, dead + pole (1,255) and dead - pole (1,212): live-pole max |a| median 0.0047; the largest g among their live top-50 entries is -1.180 and -3.094.
- Mean gate-negative share of live poles: Q4B L1 0.5534, L2 0.8378, L3 0.4870, L4 0.1295. The highest anywhere in G4 is 0.0148 (L33) and in O7 0.0506 (L0).
- Pole pairs from different neurons sharing at least 25 of 50 top positions: Q4B L1 2,020,462 (4,630 neurons), L2 1,277,841 (5,053), L3 1,865,326 (2,601). Largest number of poles holding one position: 3,883, 3,402, 2,010.
- Sampled duplicates: L03N8511+ "Fires on the boilerplate tokens of Freelancer.com-style job listing pages, chiefly the word "Bid"..."; L03N9234- "...the currency code in 'N jobs found, pricing in USD/EUR'..."; L03N8169+ "Fires on the final lowercase "c" of the token sequence "RePEc"...". Shared top positions: 8511+ and 9234- 32; 8511- and 9234+ 41; 8511- and 8169+ 44; 9234+ and 8169+ 41.

Instrument-free (the link to sampled descriptions is description-based).

**Why an AI researcher cares:** a dead-neuron count or a "near-silent" label hides a specific state of gated MLPs, and that state makes thousands of neurons look like the same rare-token detector, which inflates any count of "clean" single-job neurons in those layers.

---

### 2. On all 1.05 million neurons, firing tokens move from word pieces to function words, and the three depth curves move together

**Hook:** F saw the word-piece-to-function-word shift in 600 sampled poles per model. Decoding every pole's top 50 shows the same shift in every model, with depth curves that match across families far better than chance.

**What the data shows.** In the bottom fifth of each model, word pieces are 23.9% to 39.8% of top-50 tokens and function words 0.7% to 7.2%. In the top fifth, word pieces are 8.5% to 12.6% and function words 13.6% to 25.1%. The per-layer word-piece curve peaks at layer 2 or 3 in all three (0.59 Q4B, 0.41 G4, 0.44 O7). Function words peak late (Q4B L31, G4 L33, O7 L27). The shift is not monotone at the very top: in Q4B's last three layers function words fall back to 0.080 to 0.126 and word pieces rise to 0.137 to 0.149; O7's function share falls from 0.270 (L27) to 0.214 (L31). Word starts are the largest class in every quintile except Q4B's bottom fifth (0.358 to 0.602 per quintile), and their curve is not shared across models.

Per pole, the top 50 is close to one type at layer 0 (median share of the most common type 0.90, 0.98, 0.88) and mixed in the middle (lowest medians 0.54, 0.52, 0.58). The share of poles with at least 90% of their top 50 in one type falls from 0.524, 0.680, 0.491 at layer 0 to 0.035 (Q4B L16), 0.049 (G4 L29) and 0.053 (O7 L28).

Cross-model similarity of the per-layer curves: word pieces 0.725, function words 0.727, punctuation 0.651, whitespace 0.603. All four beat the shuffled-layer null (95th percentile 0.176 to 0.196) and the stricter circular-shift null (p 0.045, 0.003, 0.037, 0.039). Digits reach 0.551 but miss the circular null (p 0.075). Word starts reach 0.074 and beat neither.

**Evidence** ([T:*] `tok` decoded through [K]; quintiles by layer / L; table T1 has every layer):
- Word pieces by quintile: Q4B 0.3976, 0.1818, 0.1081, 0.1001, 0.1262; G4 0.2390, 0.0846, 0.0535, 0.0714, 0.0850; O7 0.2615, 0.2006, 0.1101, 0.0957, 0.0900.
- Function words by quintile: Q4B 0.0545, 0.0933, 0.1160, 0.1545, 0.1363; G4 0.0718, 0.1037, 0.1390, 0.1277, 0.1554; O7 0.0068, 0.0375, 0.0906, 0.1767, 0.2505.
- Punctuation by quintile: Q4B 0.0931, 0.1074, 0.1586, 0.1588, 0.1320; G4 0.1068, 0.1652, 0.1935, 0.1570, 0.1516; O7 0.0458, 0.0729, 0.1275, 0.1043, 0.0825.
- Curve correlations by pair (Q4B-G4, Q4B-O7, G4-O7): word pieces 0.845, 0.788, 0.541; function words 0.586, 0.820, 0.774.
- F's sample, for comparison ([F] story 3, different classifier and a 90-word list): word pieces 40.7% to 10.4% (Q4B), 19.2% to 6.4% (G4), 26.7% to 8.7% (O7).

Instrument-free.

**Why an AI researcher cares:** the detokenize-then-predict picture of depth holds on every neuron of three unrelated families, measured from raw firing positions, so it cannot be an artifact of how an LLM phrases explanations.

---

### 3. Which depth curves the three families share, and which they do not

**Hook:** Put 19 per-layer statistics on a common depth axis and ask whether three unrelated models trace the same shape. Nine do, beyond a strict null. Loudness, sharpness, dead share and the first-token share do not.

**What the data shows.** Shared beyond both nulls: median density, single-token concentration, top-50 token entropy, gate-negative share, two-pole token overlap (finding 5) and the four token-type curves of finding 2. The entropy of a pole's top-50 token ids peaks in the middle (Q4B 4.859 bits at L18, O7 5.204 at L12) and is lowest at layer 0 in Q4B and G4 (1.872 and 1.733 bits). The gate-negative share has the same U shape in all three: it falls to 0.0001 or less somewhere between 60% and 90% depth and rises again in the last two to four layers (Q4B L35 0.0063, G4 L33 0.0148, O7 L31 0.0040). The magnitudes outside Q4B's layers 1 to 4 are small.

Not shared, or shared only against the weak null: median max |a| (0.214), median q999 / max (0.207), median q99 / max (0.240), position-0 share (0.384), the share of neurons leaning to one sign (0.232), distinct documents per top 50 (0.359), digits (0.551). These beat shuffled layers but not circular shifts. RARE share (0.160) beats neither. Dead share cannot be tested because O7 has no dead pole.

**Evidence** ([H:*], [T:*]; mean of three pairwise Spearman correlations; shuffle 95th percentile; circular p-value):
- Median density share 0.787 (pairs 0.675, 0.899, 0.788), shuffle 95th 0.174, circular p 0.004.
- Share of poles with `top_same_token_frac` >= 0.5: 0.640, 0.179, p 0.008.
- Median top-50 token entropy: 0.758 (0.739, 0.748, 0.786), 0.176, p 0.001.
- Mean gate-negative share: 0.730 (0.587, 0.885, 0.718), 0.182, p 0.019.
- Mean two-pole multiset Jaccard: 0.681, 0.175, p 0.016.
- Median max |a|: 0.214 (0.045, 0.806, -0.210); circular p 0.220. Q4B and O7 both rise toward the top; G4 does not ([F] differences).
- Position-0 share >= 0.5: 0.384 (0.434, 0.057, 0.661), shuffle p 0.001, circular p 0.166.
- Lean share: 0.232 (0.467, 0.264, -0.035), circular p 0.176. RARE share: 0.160, shuffle p 0.076.
- Changing the grid to layer / L, or dropping layer 0, moves the density correlation to 0.786 and 0.790.

Instrument-free.

**Why an AI researcher cares:** "where in depth does X happen" transfers across families for density, lexical concentration and token type, but not for activation scale or sink effects, so depth landmarks found in one model should be checked statistic by statistic before being assumed in another.

---

### 4. Gemma's first-token sink was moved, not removed: 25 windows fill 5% of a middle layer's top-50 slots

**Hook:** The harvest excluded window position 0 because Gemma piled its top-k there. In G4 layers 10 and 11 the pile moved one token to the right.

**What the data shows.** In G4 layers 10 and 11, 3.74% and 5.41% of all top-50 slots sit at window position 1, which is 76 and 110 times a uniform spread (1 / 2,047). Almost all of it is 25 windows out of 26,142 whose position 0 is "\r" and position 1 is "\n": a carriage-return line feed split by the window boundary. Those 25 positions hold 35,253 top-50 slots in layer 10 and 53,148 in layer 11 (3.44% and 5.19% of all slots), and 1,391 and 2,109 poles have at least 25 of their top 50 at window position 1. These two layers also have the largest within-layer overlaps in G4: 921,763 and 2,186,110 pole pairs from different neurons share at least 25 top positions. They are the same middle layers where F found the pre-exclusion first-token pile (L9 to L11). The effect is gone by layer 12 (0.10% of slots, 13 poles).

The other two models show weaker start effects of different kinds. Q4B puts 1.39% of layer-0 slots and 0.35% of layer-35 slots at window position 1. O7 over-weights the first token of a document, right after the EOS token, which in OLMo is also its BOS id ([K] corpus manifest: `eos_id` and `bos_id` 100257). The share of O7 top-50 slots at document position 0 or 1 is 3.72% at layer 0 and 1.0% to 1.7% in layers 2 to 19, against a base rate of 0.14%. Those document-start tokens are file and page openers ("﻿using" for C# files with a byte-order mark, "'use", "Filed", "Posts", "Category", "<!["), so they may be real jobs rather than a sink.

**Evidence** ([T:*] `tok` modulo 2,048 and against `doc_offsets.npy`; [K] `tokens.npy`):
- Share of valid top-50 slots at window position 1, G4 L9 to L12: 0.0027, 0.0374, 0.0541, 0.0010 (2,761, 38,275, 55,385, 987 slots of 1,024,000, 1,024,000, 1,024,000, 1,023,950).
- Most common (position 0, position 1) token pairs among those slots: ("\r", "\n") 1,355 (L9), 35,253 (L10), 53,148 (L11), 195 (L12). Windows starting "\r", "\n": 25.
- Poles with at least 25 of 50 at window position 1: G4 L9 58, L10 1,391, L11 2,109, L12 13.
- Example contexts: '}\r\n            }\r\n\r' then "\n" at window position 1, document position 572.
- O7 share at document position 0 or 1: L0 0.0372, L5 0.0172, L12 0.0105, L24 0.0039, L31 0.0019. Q4B L0 0.0050, G4 L2 0.0034. Base rate 2 x 34,789 / 50,515,866 = 0.0014.

Instrument-free.

**Why an AI researcher cares:** excluding the first position does not remove a sink; any top-k neuron atlas built on windowed text needs to check the first few positions and boundary artifacts, or a few dozen contexts will define thousands of "features".

---

### 5. A neuron's two signs fire on the same tokens more than chance, and on the same kind of token

**Hook:** F found that a neuron's two poles are described as related. Their raw top-50 token ids agree too, at every depth of every model, and the agreement is about type, not complementarity.

**What the data shows.** For each neuron I compare the multiset of token ids in its + top 50 with its - top 50, and with the - top 50 of a random other neuron of the same layer. Same-neuron overlap beats the random pairing in every one of the 102 layers. The ratio is largest at the bottom (3.5, 7.0, 19.9 in the bottom fifth) and about 1.7 to 2.8 at the top. Absolute overlap is small: the mean Jaccard is at most 0.0744 in any layer (G4 L29). The share of neurons whose two poles share no token id at all falls from 0.88 to 0.99 at layer 0 to between 0.08 and 0.16 at its lowest layer (Q4B 0.158 at L18, G4 0.081 at L29, O7 0.123 at L27).

The brief asked whether one pole tends to fire on function words when the other fires on content words. It does not. When the + pole's most common type is a function word, the - pole's is a function word too far more often than chance: odds ratios 2.0 to 28.8 against 0.7 to 1.6 under random pairing. When the + pole is mostly content words, the - pole is mostly function words slightly less often than the base rate. The two signs of one neuron sit on the same kind of slot.

Dead-on-one-sign neurons by layer: Q4B L1 2,467, L2 653, L3 370, L4 25, then 2, 3 and 1 in L9, L10 and L12; G4 one each in L5, L12 and L17; O7 none. No neuron is dead on both signs.

**Evidence** ([T:*] `tok` decoded; one random permutation per layer for the null; table T3 has every layer):
- Mean multiset Jaccard, same neuron vs random pairing, by quintile: Q4B 0.0210 vs 0.0060, 0.0250 vs 0.0077, 0.0494 vs 0.0197, 0.0451 vs 0.0176, 0.0477 vs 0.0170; G4 0.0170 vs 0.0024, 0.0420 vs 0.0173, 0.0528 vs 0.0258, 0.0422 vs 0.0211, 0.0565 vs 0.0304; O7 0.0058 vs 0.0003, 0.0151 vs 0.0031, 0.0321 vs 0.0107, 0.0369 vs 0.0189, 0.0565 vs 0.0326. Smallest per-layer ratio: Q4B 1.06 (L2, a hub layer), G4 1.71 (L33), O7 1.65 (L27).
- Share of neurons with no shared token id, same neuron vs random, by quintile: Q4B 0.759 vs 0.915 to 0.300 vs 0.521; G4 0.827 vs 0.931 to 0.177 vs 0.294; O7 0.909 vs 0.980 to 0.158 vs 0.277.
- Odds ratio that both poles are function-word dominated, observed vs random pairing, by quintile: Q4B 8.99/1.28, 7.39/1.19, 6.55/1.31, 2.74/1.06, 2.77/1.20; G4 5.61/1.28, 6.05/1.09, 3.87/1.05, 2.69/1.05, 2.63/1.07; O7 28.82/1.57, 24.09/0.69, 10.24/1.15, 2.48/1.30, 1.96/1.07.
- P(- pole function-dominated | + pole content-dominated) vs the base rate, top fifth: Q4B 0.0917 vs 0.1042; G4 0.0804 vs 0.0904; O7 0.1568 vs 0.1722.

Instrument-free.

**Why an AI researcher cares:** the negative side of a gated neuron is tied to its positive side in what tokens it reads, in all three families, so treating signs as independent features (or pruning one) discards a measurable relation.

---

### 6. Final-layer leaning neurons: Gemma's rare side is the loud one, OLMo's rare side is the quiet one

**Hook:** F showed that last-layer neurons lean to one sign. Which side then carries the big activations differs by family.

**What the data shows.** I confirm F's lean numbers exactly (share of neurons whose + pole holds under 40% or over 60% of tokens: Q4B L32 to L35 0.2524, 0.3014, 0.4593, 0.5640; G4 L30 to L33 0.1398, 0.2689, 0.2640, 0.4607; O7 L28 to L31 0.1282, 0.1830, 0.2510, 0.3429). The rise is not unique to the top. G4 has 0.3973 at layer 0 and 0.22 to 0.28 at layers 22 to 25. Q4B layers 1 to 3 lean hardest of all (0.75, 0.82, 0.62) because of the gate-closed neurons of finding 1.

For strongly leaning neurons (+ share under 0.2 or over 0.8) I compare the mean |a| of the top 50 on the rare side with the common side. Equal counts make this fair. In G4's last four layers the rare side is 2.9 to 6.5 times stronger, and it is the stronger side in 70% to 84% of neurons. In O7's last four layers it is 0.60 to 0.73 times as strong, and the stronger side in only 21% to 41%. Q4B is mixed (1.94 at L32, 0.87 at L34, 1.11 at L35). Balanced neurons in the same layers sit near 1.0 in Q4B and G4 and at 0.91 to 0.95 in O7. So a leaning G4 neuron looks like a sparse detector riding on an offset, while a leaning O7 neuron looks like a dial that mostly sits on one side and makes small excursions.

Leaning neurons' top-50 tokens differ from balanced neurons' in the same final layer in one consistent way: more punctuation and whitespace. Function words are lower only in G4.

**Evidence** ([H:*] `n_pole_tokens`, [HM] tokens scored, [T:*] `a`, `tok` decoded):
- Ratio of rare-side to common-side top-50 mean |a|, strongly leaning neurons (n), and share where the rare side is stronger: G4 L30 6.459 (544, 0.836), L31 6.527 (1,071, 0.810), L32 4.275 (918, 0.816), L33 2.878 (1,773, 0.698); O7 L28 0.732 (531, 0.409), L29 0.595 (902, 0.314), L30 0.629 (1,469, 0.265), L31 0.642 (1,981, 0.212); Q4B L32 1.938 (567, 0.792), L33 1.398 (838, 0.638), L34 0.865 (1,851, 0.438), L35 1.107 (2,161, 0.532). The ratio of the 50th-largest values agrees (for example G4 L31 7.120, O7 L31 0.621).
- Balanced neurons (+ share 0.45 to 0.55), same ratio: Q4B L32 to L35 0.980, 0.989, 0.979, 1.002; G4 0.987, 1.010, 1.014, 1.006; O7 0.912, 0.924, 0.927, 0.949.
- Final-layer token types, rare-side pole / common-side pole / balanced neurons' poles. Punctuation: Q4B 0.160 / 0.188 / 0.148; G4 0.161 / 0.139 / 0.114; O7 0.123 / 0.139 / 0.082. Whitespace: Q4B 0.022 / 0.025 / 0.016; G4 0.038 / 0.033 / 0.014; O7 0.033 / 0.036 / 0.016. Function words: Q4B 0.093 / 0.103 / 0.095; G4 0.183 / 0.154 / 0.229; O7 0.211 / 0.231 / 0.210.
- Final layer 5th and 95th percentile of the + pole's token share: Q4B 0.081 and 0.9265; G4 0.1003 and 0.8987; O7 0.1008 and 0.8915.

Instrument-free.

**Why an AI researcher cares:** "the last layer is full of one-sided neurons" is shared, but the functional form is not; a method that treats the rare side as the feature would be right for Gemma and backwards for OLMo.

---

### 7. Loudness predicts legibility inside every depth band and both explainers, and depth carries its own weight

**Hook:** F's strongest commonality (loud poles get ONE) survives the controls F could not run together: depth fifth, explainer, and a joint fit.

**What the data shows.** Pooling the three models, the loudest within-layer quartile beats the quietest in every depth fifth (0.82 vs 0.54, 0.64 vs 0.61, 0.62 vs 0.33, 0.81 vs 0.33, 0.47 vs 0.31). The gap is smallest at 20% to 40% depth, where the two quartiles' intervals overlap almost entirely. Model by model the cells are small (19 to 40 poles) and not every one is monotone: Q4B at 20% to 40% depth is flat, and O7's top fifth reverses (0.37 to 0.28). Inside each explainer the gradient holds: Opus 0.47 to 0.71, Sonnet 0.30 to 0.55, pooled. Opus poles split at the layer median of loudness gain 0.04 to 0.25 in each of the four depth fifths it covers.

In a logistic fit of ONE on loudness percentile, density percentile, an explainer flag and depth-fifth dummies, pooled over models with model dummies, both loudness and density keep intervals that exclude zero. Per model, loudness never does once density is in, and density does only in O7 (-0.970 [-1.914, -0.055]), because the two correlate at -0.81 to -0.85. The middle depth fifth is lower than the bottom fifth in all three per-model fits, with intervals excluding zero. The explainer flag is near zero once depth is in (pooled -0.03), but Sonnet explained only the top two fifths, so explainer and depth are not separable. This fit is descriptive, not causal.

**Evidence** ([P:*] `class`, `explainer_model`, `pole`; [H:*] `max_abs`, `density`; within-layer percentile over both signs; RARE excluded, n = 592, 600, 600; 95% intervals from 1,000 bootstrap draws over neurons):
- Pooled ONE share by loudness quartile, depth fifth 0: 0.54 [0.44, 0.64] n98, 0.71 [0.63, 0.81] n91, 0.76 [0.66, 0.85] n88, 0.82 [0.74, 0.89] n103. Fifth 1: 0.61 [0.51, 0.71], 0.57, 0.61, 0.64 [0.55, 0.74]. Fifth 2: 0.33 [0.23, 0.42], 0.48, 0.51, 0.62 [0.52, 0.72]. Fifth 3: 0.33 [0.23, 0.44], 0.57, 0.65, 0.81 [0.72, 0.89]. Fifth 4: 0.31 [0.22, 0.40], 0.43, 0.46, 0.47 [0.36, 0.57].
- By explainer, pooled: Opus 0.47 [0.41, 0.52] n333, 0.59, 0.62, 0.71 [0.66, 0.76] n340; Sonnet 0.30 [0.23, 0.38] n128, 0.44, 0.52, 0.55 [0.45, 0.63] n106. Per model, Opus quietest to loudest: Q4B 0.51 to 0.73, G4 0.50 to 0.69, O7 0.38 to 0.72; Sonnet: Q4B 0.40 to 0.52, G4 0.21 to 0.66, O7 0.30 to 0.48.
- Opus, below vs above the layer median of loudness, depth fifths 0 to 3: 0.63 vs 0.79; 0.59 vs 0.63; 0.41 vs 0.56; 0.46 vs 0.71. Sonnet fifths 3 and 4: 0.41 vs 0.75; 0.36 vs 0.46.
- Pooled logistic coefficients (log-odds per unit percentile, i.e. quietest to loudest): loudness 0.711 [0.088, 1.327]; density -0.785 [-1.450, -0.207]; Sonnet -0.032 [-0.476, 0.435]; depth fifths 1 to 4 vs 0: -0.484, -0.997 [-1.347, -0.666], -0.563, -1.271; G4 -0.218, O7 -0.410 [-0.673, -0.156].
- Per model, loudness alone: 1.138 (Q4B), 1.384 (G4), 1.587 (O7). With density added: loudness 0.664 [-0.410, 1.770], 0.684 [-0.510, 1.892], 0.808 [-0.196, 1.833]. Middle-fifth dummy: -1.232 [-1.866, -0.621], -0.959 [-1.577, -0.390], -0.829 [-1.420, -0.301].

Description-based (the outcome is the study's ONE label).

**Why an AI researcher cares:** a free activation statistic predicts which neurons an LLM can describe, within depth and within explainer, so the headline clean share depends on the loudness mix of the sample as well as on depth.

---

### 8. Cross-model description twins are real but a third as common as within-model duplicates, and their depth match disappears once the explainer is held fixed

**Hook:** Matching 1,788 descriptions across models finds twins far beyond chance. But a pole is about three times more likely to have a twin in its own model, and twins do not sit at matching depths beyond what the Opus/Sonnet split predicts.

**What the data shows.** With TF-IDF over the explanations (English stop words and generic description words removed), 27 cross-model pairs reach cosine 0.3. Shuffling words across descriptions within each model, keeping lengths, gives no pair at 0.3 in any of 50 draws, and the 100th-best cross-model cosine is 0.250 against at most 0.116. So twins are not vocabulary chance. Within-model pairs (different neurons) reach 0.3 at a rate that would predict 76 cross-model pairs. The within-model excess is concentrated in Q4B (23 pairs) and G4 (13). O7 has 2, a rate below its cross-model rate. Removing Q4B layers 1 to 3 (finding 1's hub layers) still leaves 28 within-model pairs, a rate predicting 59.

F reported that the top 100 cross-model pairs sit closer in depth than random pairs. I find the same (mean |relative depth difference| 0.284 against 0.329 under a permutation null, p 0.026). But Opus explained almost all shallow poles and Sonnet the deep ones, and 75% of those top pairs share an explainer. Permuting depth only among poles with the same explainer gives 0.287 (p 0.43). Within-model twins are different: their top 100 sit at 0.207 against 0.285 under the same stratified null (p 0.001), and 39% are within two layers. Within a model, near-duplicate jobs cluster in depth; across models, the evidence for depth matching is the explainer split.

**Evidence** ([P:*] `explanation`, `class`, `explainer_model`, `layer`; 1,065,627 cross-model pairs, 531,063 within-model pairs):
- Pairs at cosine >= 0.3: cross 27; within 38 (rate 7.16e-05, predicting 76.3); word-shuffle null mean 0.0. At >= 0.4: cross 1, within 16 (predicting 32.1). With only English stop words: cross 27 and 2, within 40 and 13.
- Within-model pairs >= 0.3 by model: Q4B 23 of 174,054, G4 13 of 177,609, O7 2 of 179,400. Cross-model rate 2.53e-05.
- Both poles passing (ONE or TWO-REL): cross 9 of 412,410 pairs, within 22 of 206,031.
- Per pole, share whose best match in one other model beats its best match in its own model (598 or 600 candidates each): 0.380 to 0.473.
- Depth, top 300 cross pairs: 0.272 vs stratified null 0.294 (p 0.038), so a weak residual alignment may exist in the longer tail.
- Twins not in F, with pole ids, relative depth, class and explainer:
  - Q4B L14N4399- (0.39, ONE, Opus) "In source code, fires on the indentation whitespace (or closing brace) that ends the first short method of a class, typically a constructor, __init__ or Start() body" and G4 L19N5005+ (0.56, ONE, Opus) "In object-oriented code, fires on the token that begins a class's constructor or calls the parent constructor". Cosine 0.390. One ends a constructor, the other begins one.
  - Q4B L07N1025+ (0.19, ONE, Opus) "Fires on the word right after a sentence-initial purpose "To + verb"" and O7 L17N10828- (0.53, UNK, Opus) "Fires on the last word, or the comma, that closes a sentence-initial infinitive purpose clause". Cosine 0.384.
  - Q4B L21N1823+ (0.58, ONE, Opus) "Fires on the verbs "attract" and "draw" ... in the sense of pulling in people's attention" and O7 L22N3469+ (0.69, ONE, Opus) "Fires on the verb (or the word right before it) in an idiom about drawing someone's notice, predicting that "attention", "eye" or "heads" comes next". Cosine 0.296.
  - G4 L06N3238- (0.18, ONE, Opus) "Fires on the period (or opening quote/parenthesis) that closes the author-name portion of a bibliography" and O7 L07N5613+ (0.22, ONE, Opus) "In an academic author-year citation, fires on the separator right after the first author's surname". Cosine 0.282.
  - Q4B L22N7232- (0.61, ONE, Opus) "the comma (or closing quote-comma) that ends a full reported clause right before a source attribution such as "the newspaper said,"" and O7 L16N864+ (0.50, ONE, Opus) "Fires on speech-attribution points in narrative". Cosine 0.276.
  - G4 L17N9317+ (0.50, ONE, Opus) "words inside verbatim quoted or titled old-language material" and O7 L04N10274- (0.12, ONE, Opus) "the first token of a new clause or verse line inside Latin or archaic old-language text". Cosine 0.318.
  - Q4B L02N4108- (0.06, ONE, Opus) "the byte-fragment token inside a polytonic Greek letter" and G4 L29N8196+ (0.85, UNK, Sonnet) "a mid-word fragment inside a long word in Greek (polytonic Ancient Greek) or Ukrainian Cyrillic". Cosine 0.379, a depth gap of 0.80.
  - Q4B L35N6252+ (0.97, UNK, Sonnet) "a word-initial fragment 'Pr' (also 'Tr', 'Cr', 'Thr', 'fr', 'r', 'cl') that begins a longer word" and G4 L29N4505- (0.85, UNK, Sonnet) "a short word-start fragment (Im, Lim, St, Th, Cl, st, cl, tr, th) that begins a longer word". Cosine 0.276. Same family as F's story 8.
  - The single best pair is F's indentation pair (Q4B L18N621-, G4 L19N4173+, cosine 0.410).

Description-based.

**Why an AI researcher cares:** neuron-level universality across families is real but weaker than within-family redundancy, and a depth match between twins can be manufactured by which explainer wrote which layers.

---

### 9. OLMo's layer 0 reads classes of tokens, not single tokens

**Hook:** In Qwen and Gemma, a layer-0 pole's top 50 is about 6 or 7 distinct tokens. In OLMo it is 23.

**What the data shows.** F noted that O7's layer 0 is not a token lookup by `top_same_token_frac`. Normalizing case and spaces does not change that: 0.601 and 0.603 of Q4B and G4 layer-0 poles have half their top 50 on one normalized string, against 0.130 in O7. O7 layer-0 poles still have one token type (median type concentration 0.88, mostly word starts), but many different words. Examples: the top tokens of O7 L0N3656+ include " Mund", " tantal", " mentally", " freely", " fiercely", " meticulously" and " progressively", mostly adverbs; G4 L0N5961+ " Peace" 29, " PEACE" 9, " Peer" 6, " peer" 4. O7's layer-0 activations are also small (median max |a| 0.2985, against 1.292 and 10.504; [F]).

**Evidence** ([T:*] `tok` decoded; normalized string = decoded token stripped and lowercased):
- Layer 0: share of poles with top-id fraction >= 0.5: 0.548, 0.555, 0.115; with top normalized-string fraction >= 0.5: 0.601, 0.603, 0.130; median distinct token ids per top 50: 7, 6, 23.
- Layer 1: 0.287, 0.325, 0.133; distinct ids 10, 12, 22. Middle layer (Q4B L18, G4 L17, O7 L16): distinct ids 35, 26, 33.

Instrument-free.

**Why an AI researcher cares:** "layer-0 neurons are token detectors" is a family property, not a law; OLMo's first MLP already responds to sets of words, which changes what an early-layer atlas should expect.

---

### 10. Gemma's neurons overlap the most and OLMo's the least, by positions and by descriptions

**Hook:** Count pole pairs from different neurons whose strongest tokens are the same corpus positions. Outside the hub layers, Gemma has about five times as many as the other two, and OLMo's deep layers have almost none.

**What the data shows.** The median layer has 2,738 pole pairs sharing at least 10 of 50 top positions in G4, against 538.5 in Q4B and 569 in O7. From about 60% depth up (Q4B L21, G4 L20, O7 L19 onward) the medians are 1,670 (G4), 537 (Q4B) and 179 (O7). From O7 L23 to L30 no layer has more than 5 pairs sharing 25 positions. G4 also has hub positions held by 100 or more poles in layers 5, 7, 9, 10, 11, 13, 14, 18 to 20, 22, 32 and 33, Q4B only in layers 1 to 4, and O7 in none. The description side agrees: within-model description duplicates at cosine 0.3 are 23 (Q4B), 13 (G4) and 2 (O7) (finding 8). Two unrelated measures put OLMo at the low end of redundancy.

**Evidence** ([T:*] `tok`; sparse pole-by-position products per layer; table T3 has the count at 25 shared positions per layer):
- Pairs sharing >= 25 of 50 positions, median over layers: Q4B 23, G4 311.5, O7 50.
- Largest number of poles holding one position, middle layers: G4 L19 150, L20 135; Q4B L17 31; O7 L13 33.

Instrument-free.

**Why an AI researcher cares:** neuron redundancy differs by family, so "how many distinct features does a model have" cannot be read off neuron counts in the same way for each.

---

### 11. Denser poles have more varied top tokens in all 102 layers, and dense poles are not function-word poles

**Hook:** One relation holds in every layer of every model. Another that might be expected does not hold at all.

**What the data shows.** Within each layer, a pole's log density and the entropy of its top-50 token ids have a positive rank correlation in all 102 layers (0.084 to 0.423). The brief asked whether the dense middle layers fire on function words. Within a layer, function-word share and density are unrelated in the middle (rank correlations -0.071 to 0.063 between 39% and 47% depth) and change sign across depth. In the middle layers the densest tenth of poles has about the same function-word share as the sparsest tenth. Distinct documents in a pole's top 50 are 30 to 45 at the median in every layer; poles drawing 50 contexts from 5 or fewer documents are at most 2.06% of a layer everywhere except Q4B L3 (8.15%, a hub layer).

**Evidence** ([H:*] `density`, [T:*] `tok`, `doc_offsets.npy`; live non-RARE poles with 50 valid slots):
- Rank correlation of entropy with log density, smallest and largest layer: Q4B 0.088 (L18) and 0.423 (L1); G4 0.084 (L3) and 0.360 (L22); O7 0.115 (L12) and 0.386 (L17).
- Function-word share vs log density, range over layers: Q4B -0.094 to 0.360; G4 -0.125 to 0.181; O7 -0.103 to 0.163.
- Function share, densest tenth vs sparsest tenth: Q4B L17 0.0847 vs 0.0978; G4 L14 0.0854 vs 0.0911; O7 L14 0.0620 vs 0.0629.

Instrument-free.

**Why an AI researcher cares:** density is a proxy for lexical diffuseness in every family, not for grammatical role, so "dense" poles should not be assumed to be the function-word predictors of the upper layers.

---

## Per-layer tables

Relative depth is layer / L. All values are my computation from the files named in each table.

### T1. Firing-token type shares of all valid top-50 slots (both poles), and per-pole type concentration

Columns: share of slots that are word piece, word start, function word, punctuation, whitespace, digit, other; median over poles of the most common type's share; share of poles with at least 90% of their top 50 in one type. Source: [T:*] `tok` decoded through [K].

**Q4B**

| L | rel | piece | start | func | punct | ws | digit | other | med conc | conc>=0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.2904 | 0.4539 | 0.0463 | 0.1233 | 0.0077 | 0.0050 | 0.0735 | 0.90 | 0.5238 |
| 1 | 0.028 | 0.4983 | 0.2061 | 0.0153 | 0.1084 | 0.0059 | 0.0033 | 0.1627 | 0.84 | 0.4280 |
| 2 | 0.056 | 0.5762 | 0.2224 | 0.0437 | 0.0778 | 0.0135 | 0.0118 | 0.0547 | 0.84 | 0.3914 |
| 3 | 0.083 | 0.5899 | 0.2625 | 0.0223 | 0.0715 | 0.0114 | 0.0119 | 0.0305 | 0.87 | 0.4702 |
| 4 | 0.111 | 0.3848 | 0.3894 | 0.0308 | 0.1155 | 0.0186 | 0.0266 | 0.0344 | 0.78 | 0.3441 |
| 5 | 0.139 | 0.3313 | 0.4582 | 0.0583 | 0.0842 | 0.0132 | 0.0240 | 0.0309 | 0.80 | 0.3556 |
| 6 | 0.167 | 0.2808 | 0.4279 | 0.1066 | 0.0788 | 0.0161 | 0.0536 | 0.0361 | 0.76 | 0.3108 |
| 7 | 0.194 | 0.2693 | 0.4017 | 0.1035 | 0.0865 | 0.0208 | 0.0826 | 0.0356 | 0.72 | 0.2373 |
| 8 | 0.222 | 0.1928 | 0.4702 | 0.1291 | 0.1010 | 0.0287 | 0.0463 | 0.0320 | 0.68 | 0.1841 |
| 9 | 0.250 | 0.2143 | 0.4817 | 0.0924 | 0.0931 | 0.0292 | 0.0510 | 0.0383 | 0.64 | 0.1088 |
| 10 | 0.278 | 0.1995 | 0.4904 | 0.0791 | 0.1011 | 0.0338 | 0.0530 | 0.0430 | 0.62 | 0.0941 |
| 11 | 0.306 | 0.1871 | 0.4985 | 0.0827 | 0.1005 | 0.0317 | 0.0550 | 0.0444 | 0.62 | 0.0810 |
| 12 | 0.333 | 0.1769 | 0.4996 | 0.0858 | 0.1033 | 0.0321 | 0.0583 | 0.0439 | 0.62 | 0.0838 |
| 13 | 0.361 | 0.1652 | 0.4894 | 0.0864 | 0.1137 | 0.0364 | 0.0658 | 0.0431 | 0.62 | 0.0952 |
| 14 | 0.389 | 0.1365 | 0.4678 | 0.0978 | 0.1387 | 0.0449 | 0.0727 | 0.0416 | 0.58 | 0.0651 |
| 15 | 0.417 | 0.1314 | 0.4792 | 0.0977 | 0.1330 | 0.0395 | 0.0786 | 0.0407 | 0.58 | 0.0459 |
| 16 | 0.444 | 0.1195 | 0.4586 | 0.1068 | 0.1498 | 0.0487 | 0.0763 | 0.0402 | 0.54 | 0.0352 |
| 17 | 0.472 | 0.1144 | 0.4479 | 0.1032 | 0.1669 | 0.0480 | 0.0794 | 0.0402 | 0.54 | 0.0436 |
| 18 | 0.500 | 0.1035 | 0.4640 | 0.1144 | 0.1554 | 0.0471 | 0.0772 | 0.0384 | 0.56 | 0.0491 |
| 19 | 0.528 | 0.0973 | 0.4535 | 0.1251 | 0.1693 | 0.0464 | 0.0740 | 0.0344 | 0.58 | 0.0809 |
| 20 | 0.556 | 0.0995 | 0.4447 | 0.1275 | 0.1691 | 0.0463 | 0.0796 | 0.0332 | 0.58 | 0.1007 |
| 21 | 0.583 | 0.0915 | 0.4544 | 0.1377 | 0.1667 | 0.0469 | 0.0722 | 0.0307 | 0.62 | 0.1353 |
| 22 | 0.611 | 0.0930 | 0.4541 | 0.1438 | 0.1668 | 0.0451 | 0.0700 | 0.0272 | 0.62 | 0.1656 |
| 23 | 0.639 | 0.0962 | 0.4365 | 0.1636 | 0.1633 | 0.0492 | 0.0668 | 0.0243 | 0.62 | 0.1497 |
| 24 | 0.667 | 0.1064 | 0.4208 | 0.1564 | 0.1654 | 0.0577 | 0.0708 | 0.0227 | 0.58 | 0.1350 |
| 25 | 0.694 | 0.1008 | 0.4392 | 0.1509 | 0.1575 | 0.0569 | 0.0711 | 0.0235 | 0.60 | 0.1466 |
| 26 | 0.722 | 0.1027 | 0.4427 | 0.1506 | 0.1548 | 0.0604 | 0.0637 | 0.0252 | 0.58 | 0.1273 |
| 27 | 0.750 | 0.1032 | 0.4437 | 0.1525 | 0.1543 | 0.0615 | 0.0575 | 0.0273 | 0.56 | 0.1269 |
| 28 | 0.778 | 0.0985 | 0.4504 | 0.1637 | 0.1496 | 0.0543 | 0.0555 | 0.0280 | 0.56 | 0.1297 |
| 29 | 0.806 | 0.1080 | 0.4466 | 0.1500 | 0.1429 | 0.0592 | 0.0643 | 0.0291 | 0.56 | 0.1162 |
| 30 | 0.833 | 0.1064 | 0.4457 | 0.1661 | 0.1387 | 0.0548 | 0.0587 | 0.0297 | 0.54 | 0.1041 |
| 31 | 0.861 | 0.1128 | 0.4606 | 0.1720 | 0.1349 | 0.0422 | 0.0470 | 0.0306 | 0.56 | 0.0841 |
| 32 | 0.889 | 0.1253 | 0.4687 | 0.1635 | 0.1334 | 0.0307 | 0.0437 | 0.0346 | 0.56 | 0.0798 |
| 33 | 0.917 | 0.1447 | 0.5029 | 0.1258 | 0.1085 | 0.0228 | 0.0448 | 0.0506 | 0.60 | 0.0836 |
| 34 | 0.944 | 0.1493 | 0.5107 | 0.0803 | 0.1024 | 0.0250 | 0.0457 | 0.0866 | 0.62 | 0.1051 |
| 35 | 0.972 | 0.1367 | 0.4653 | 0.0967 | 0.1629 | 0.0202 | 0.0394 | 0.0787 | 0.58 | 0.0891 |

**G4**

| L | rel | piece | start | func | punct | ws | digit | other | med conc | conc>=0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.2488 | 0.6095 | 0.0202 | 0.0874 | 0.0137 | 0.0130 | 0.0075 | 0.98 | 0.6797 |
| 1 | 0.029 | 0.3328 | 0.5063 | 0.0284 | 0.0928 | 0.0130 | 0.0203 | 0.0065 | 0.96 | 0.6233 |
| 2 | 0.059 | 0.4139 | 0.3803 | 0.0540 | 0.0916 | 0.0161 | 0.0378 | 0.0062 | 0.90 | 0.5100 |
| 3 | 0.088 | 0.2759 | 0.4276 | 0.0972 | 0.1218 | 0.0257 | 0.0484 | 0.0035 | 0.82 | 0.4047 |
| 4 | 0.118 | 0.1589 | 0.5691 | 0.0992 | 0.1010 | 0.0279 | 0.0396 | 0.0042 | 0.80 | 0.3368 |
| 5 | 0.147 | 0.1299 | 0.5878 | 0.1017 | 0.1076 | 0.0325 | 0.0366 | 0.0038 | 0.82 | 0.3709 |
| 6 | 0.176 | 0.1130 | 0.5341 | 0.1022 | 0.1455 | 0.0460 | 0.0553 | 0.0039 | 0.74 | 0.2339 |
| 7 | 0.206 | 0.0998 | 0.5793 | 0.0858 | 0.1298 | 0.0425 | 0.0599 | 0.0028 | 0.74 | 0.2574 |
| 8 | 0.235 | 0.0956 | 0.5057 | 0.1138 | 0.1736 | 0.0493 | 0.0592 | 0.0028 | 0.66 | 0.1422 |
| 9 | 0.265 | 0.0835 | 0.5232 | 0.1141 | 0.1569 | 0.0493 | 0.0707 | 0.0022 | 0.68 | 0.1513 |
| 10 | 0.294 | 0.0727 | 0.4464 | 0.0985 | 0.1907 | 0.1134 | 0.0741 | 0.0041 | 0.58 | 0.0885 |
| 11 | 0.324 | 0.0801 | 0.4909 | 0.0918 | 0.1561 | 0.1082 | 0.0703 | 0.0026 | 0.60 | 0.0864 |
| 12 | 0.353 | 0.0724 | 0.4899 | 0.1200 | 0.1849 | 0.0522 | 0.0776 | 0.0028 | 0.62 | 0.1041 |
| 13 | 0.382 | 0.0880 | 0.5024 | 0.1016 | 0.1646 | 0.0577 | 0.0821 | 0.0037 | 0.64 | 0.1115 |
| 14 | 0.412 | 0.0751 | 0.5111 | 0.1073 | 0.1710 | 0.0496 | 0.0830 | 0.0029 | 0.62 | 0.0937 |
| 15 | 0.441 | 0.0513 | 0.4211 | 0.1292 | 0.2213 | 0.0751 | 0.0989 | 0.0031 | 0.54 | 0.0793 |
| 16 | 0.471 | 0.0496 | 0.4165 | 0.1389 | 0.2315 | 0.0688 | 0.0924 | 0.0024 | 0.54 | 0.0788 |
| 17 | 0.500 | 0.0456 | 0.4363 | 0.1409 | 0.2174 | 0.0791 | 0.0788 | 0.0020 | 0.58 | 0.1204 |
| 18 | 0.529 | 0.0533 | 0.4912 | 0.1440 | 0.1699 | 0.0424 | 0.0976 | 0.0015 | 0.64 | 0.1416 |
| 19 | 0.559 | 0.0473 | 0.4823 | 0.1607 | 0.1747 | 0.0435 | 0.0898 | 0.0017 | 0.64 | 0.1485 |
| 20 | 0.588 | 0.0523 | 0.5084 | 0.1519 | 0.1684 | 0.0409 | 0.0766 | 0.0013 | 0.68 | 0.1957 |
| 21 | 0.618 | 0.0596 | 0.4978 | 0.1581 | 0.1559 | 0.0425 | 0.0845 | 0.0015 | 0.68 | 0.1875 |
| 22 | 0.647 | 0.0601 | 0.5000 | 0.1411 | 0.1549 | 0.0500 | 0.0919 | 0.0018 | 0.68 | 0.2167 |
| 23 | 0.676 | 0.0779 | 0.4783 | 0.1235 | 0.1604 | 0.0649 | 0.0936 | 0.0014 | 0.64 | 0.1768 |
| 24 | 0.706 | 0.0766 | 0.4941 | 0.1109 | 0.1478 | 0.0548 | 0.1147 | 0.0011 | 0.62 | 0.1562 |
| 25 | 0.735 | 0.0749 | 0.5168 | 0.1098 | 0.1511 | 0.0480 | 0.0982 | 0.0013 | 0.62 | 0.1481 |
| 26 | 0.765 | 0.0766 | 0.5102 | 0.1197 | 0.1602 | 0.0440 | 0.0878 | 0.0015 | 0.60 | 0.1123 |
| 27 | 0.794 | 0.0739 | 0.4946 | 0.1304 | 0.1689 | 0.0445 | 0.0864 | 0.0012 | 0.56 | 0.0948 |
| 28 | 0.824 | 0.0807 | 0.4981 | 0.1348 | 0.1641 | 0.0398 | 0.0813 | 0.0012 | 0.56 | 0.0798 |
| 29 | 0.853 | 0.0810 | 0.4551 | 0.1330 | 0.1822 | 0.0492 | 0.0983 | 0.0013 | 0.52 | 0.0489 |
| 30 | 0.882 | 0.0766 | 0.4976 | 0.1583 | 0.1601 | 0.0401 | 0.0660 | 0.0012 | 0.56 | 0.0581 |
| 31 | 0.912 | 0.0922 | 0.5197 | 0.1578 | 0.1486 | 0.0352 | 0.0450 | 0.0015 | 0.60 | 0.0804 |
| 32 | 0.941 | 0.0906 | 0.5730 | 0.1478 | 0.1244 | 0.0234 | 0.0393 | 0.0015 | 0.66 | 0.1354 |
| 33 | 0.971 | 0.0886 | 0.5224 | 0.2009 | 0.1305 | 0.0240 | 0.0321 | 0.0016 | 0.60 | 0.0816 |

**O7**

| L | rel | piece | start | func | punct | ws | digit | other | med conc | conc>=0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.1092 | 0.7615 | 0.0006 | 0.0378 | 0.0021 | 0.0192 | 0.0695 | 0.88 | 0.4909 |
| 1 | 0.031 | 0.2296 | 0.6750 | 0.0008 | 0.0310 | 0.0010 | 0.0081 | 0.0544 | 0.90 | 0.5426 |
| 2 | 0.062 | 0.4392 | 0.4209 | 0.0021 | 0.0361 | 0.0017 | 0.0076 | 0.0925 | 0.76 | 0.3133 |
| 3 | 0.094 | 0.2840 | 0.5675 | 0.0029 | 0.0447 | 0.0036 | 0.0082 | 0.0892 | 0.74 | 0.2408 |
| 4 | 0.125 | 0.2471 | 0.5961 | 0.0079 | 0.0512 | 0.0064 | 0.0126 | 0.0786 | 0.72 | 0.1773 |
| 5 | 0.156 | 0.2596 | 0.5594 | 0.0132 | 0.0652 | 0.0118 | 0.0162 | 0.0747 | 0.68 | 0.1349 |
| 6 | 0.188 | 0.2621 | 0.5800 | 0.0199 | 0.0549 | 0.0112 | 0.0124 | 0.0596 | 0.66 | 0.1140 |
| 7 | 0.219 | 0.2448 | 0.5795 | 0.0307 | 0.0608 | 0.0133 | 0.0193 | 0.0515 | 0.68 | 0.1276 |
| 8 | 0.250 | 0.2333 | 0.5986 | 0.0324 | 0.0518 | 0.0124 | 0.0206 | 0.0509 | 0.68 | 0.1055 |
| 9 | 0.281 | 0.2145 | 0.6087 | 0.0332 | 0.0571 | 0.0149 | 0.0234 | 0.0482 | 0.70 | 0.1108 |
| 10 | 0.312 | 0.1944 | 0.6031 | 0.0376 | 0.0747 | 0.0193 | 0.0268 | 0.0442 | 0.68 | 0.0846 |
| 11 | 0.344 | 0.1689 | 0.6053 | 0.0432 | 0.0912 | 0.0211 | 0.0279 | 0.0425 | 0.68 | 0.0811 |
| 12 | 0.375 | 0.1478 | 0.6160 | 0.0479 | 0.1020 | 0.0213 | 0.0271 | 0.0379 | 0.70 | 0.0917 |
| 13 | 0.406 | 0.1386 | 0.6096 | 0.0573 | 0.1088 | 0.0225 | 0.0274 | 0.0358 | 0.70 | 0.1158 |
| 14 | 0.438 | 0.1236 | 0.6007 | 0.0677 | 0.1252 | 0.0232 | 0.0274 | 0.0323 | 0.70 | 0.1119 |
| 15 | 0.469 | 0.1101 | 0.5872 | 0.0855 | 0.1426 | 0.0237 | 0.0247 | 0.0262 | 0.70 | 0.1304 |
| 16 | 0.500 | 0.1026 | 0.5917 | 0.0960 | 0.1418 | 0.0251 | 0.0205 | 0.0223 | 0.70 | 0.1561 |
| 17 | 0.531 | 0.0990 | 0.6125 | 0.0987 | 0.1278 | 0.0238 | 0.0191 | 0.0191 | 0.72 | 0.1730 |
| 18 | 0.562 | 0.0930 | 0.6042 | 0.1163 | 0.1248 | 0.0267 | 0.0194 | 0.0155 | 0.68 | 0.1423 |
| 19 | 0.594 | 0.1038 | 0.5969 | 0.1127 | 0.1218 | 0.0274 | 0.0207 | 0.0167 | 0.68 | 0.1498 |
| 20 | 0.625 | 0.1061 | 0.5993 | 0.1176 | 0.1116 | 0.0274 | 0.0216 | 0.0163 | 0.68 | 0.1479 |
| 21 | 0.656 | 0.1036 | 0.5821 | 0.1323 | 0.1141 | 0.0298 | 0.0223 | 0.0158 | 0.66 | 0.1476 |
| 22 | 0.688 | 0.1030 | 0.5729 | 0.1518 | 0.1107 | 0.0264 | 0.0190 | 0.0163 | 0.64 | 0.1404 |
| 23 | 0.719 | 0.0962 | 0.5642 | 0.1785 | 0.1045 | 0.0228 | 0.0177 | 0.0160 | 0.62 | 0.1276 |
| 24 | 0.750 | 0.0878 | 0.5505 | 0.2160 | 0.0978 | 0.0184 | 0.0150 | 0.0145 | 0.60 | 0.1032 |
| 25 | 0.781 | 0.0777 | 0.5326 | 0.2638 | 0.0869 | 0.0132 | 0.0129 | 0.0130 | 0.58 | 0.0828 |
| 26 | 0.812 | 0.0777 | 0.5384 | 0.2698 | 0.0773 | 0.0106 | 0.0127 | 0.0135 | 0.58 | 0.0695 |
| 27 | 0.844 | 0.0816 | 0.5309 | 0.2700 | 0.0801 | 0.0108 | 0.0125 | 0.0141 | 0.58 | 0.0565 |
| 28 | 0.875 | 0.0834 | 0.5360 | 0.2670 | 0.0743 | 0.0097 | 0.0144 | 0.0151 | 0.58 | 0.0525 |
| 29 | 0.906 | 0.0903 | 0.5310 | 0.2525 | 0.0800 | 0.0116 | 0.0164 | 0.0181 | 0.58 | 0.0565 |
| 30 | 0.938 | 0.0987 | 0.5291 | 0.2299 | 0.0844 | 0.0162 | 0.0183 | 0.0234 | 0.58 | 0.0580 |
| 31 | 0.969 | 0.1080 | 0.5090 | 0.2138 | 0.0988 | 0.0225 | 0.0189 | 0.0291 | 0.60 | 0.0728 |

### T2. Population statistics per layer

Columns: median density share (`density` / tokens scored, all poles); median q999 / max (live poles); share of poles with `top_same_token_frac` >= 0.5; dead share; RARE share; share with `pos0_share` >= 0.5; mean gate-negative share (live poles); median max |a| (all poles); share of neurons with + token share outside 0.4 to 0.6; neurons dead on exactly one sign. Source: [H:*], [HM].

**Q4B**

| L | rel | dens share | q999/max | same>=0.5 | dead | RARE | pos0>=0.5 | gate-neg | med max abs | lean | dead one sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.1239 | 0.1976 | 0.5477 | 0.0000 | 0.0000 | 0.0167 | 0.0657 | 1.292 | 0.0976 | 0 |
| 1 | 0.028 | 0.0049 | 0.1351 | 0.3288 | 0.1268 | 0.1946 | 0.0201 | 0.5534 | 0.8759 | 0.7491 | 2467 |
| 2 | 0.056 | 0.0291 | 0.1493 | 0.2612 | 0.0336 | 0.0904 | 0.0669 | 0.8378 | 0.04096 | 0.8161 | 653 |
| 3 | 0.083 | 0.0122 | 0.0701 | 0.3501 | 0.0190 | 0.0460 | 0.0370 | 0.4870 | 2.414 | 0.6171 | 370 |
| 4 | 0.111 | 0.0191 | 0.0642 | 0.2533 | 0.0013 | 0.0017 | 0.0108 | 0.1295 | 8.27 | 0.3782 | 25 |
| 5 | 0.139 | 0.1199 | 0.1305 | 0.2665 | 0.0000 | 0.0000 | 0.0043 | 0.0176 | 6.294 | 0.0960 | 0 |
| 6 | 0.167 | 0.1052 | 0.1198 | 0.2939 | 0.0000 | 0.0001 | 0.0132 | 0.0062 | 7.86 | 0.0336 | 0 |
| 7 | 0.194 | 0.0929 | 0.1067 | 0.2575 | 0.0000 | 0.0000 | 0.0002 | 0.0042 | 8.899 | 0.0288 | 0 |
| 8 | 0.222 | 0.1517 | 0.1282 | 0.2170 | 0.0000 | 0.0000 | 0.0000 | 0.0060 | 8.023 | 0.0192 | 0 |
| 9 | 0.250 | 0.1259 | 0.0957 | 0.1174 | 0.0001 | 0.0004 | 0.0001 | 0.0147 | 11.06 | 0.0474 | 2 |
| 10 | 0.278 | 0.1310 | 0.0942 | 0.0985 | 0.0002 | 0.0004 | 0.0001 | 0.0157 | 10.29 | 0.0713 | 3 |
| 11 | 0.306 | 0.1553 | 0.1106 | 0.0803 | 0.0000 | 0.0000 | 0.0000 | 0.0089 | 8.299 | 0.0637 | 0 |
| 12 | 0.333 | 0.1907 | 0.1333 | 0.0847 | 0.0001 | 0.0001 | 0.0001 | 0.0133 | 6.303 | 0.0645 | 1 |
| 13 | 0.361 | 0.2107 | 0.1565 | 0.0812 | 0.0000 | 0.0000 | 0.0003 | 0.0157 | 5.613 | 0.0671 | 0 |
| 14 | 0.389 | 0.2176 | 0.1490 | 0.0686 | 0.0000 | 0.0000 | 0.0000 | 0.0176 | 5.395 | 0.0663 | 0 |
| 15 | 0.417 | 0.2162 | 0.1531 | 0.0398 | 0.0000 | 0.0000 | 0.0002 | 0.0148 | 5.075 | 0.0510 | 0 |
| 16 | 0.444 | 0.2218 | 0.1450 | 0.0408 | 0.0000 | 0.0000 | 0.0000 | 0.0121 | 5.093 | 0.0698 | 0 |
| 17 | 0.472 | 0.2282 | 0.1634 | 0.0450 | 0.0000 | 0.0000 | 0.0000 | 0.0066 | 4.872 | 0.0824 | 0 |
| 18 | 0.500 | 0.2226 | 0.1643 | 0.0421 | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 5.176 | 0.0736 | 0 |
| 19 | 0.528 | 0.2202 | 0.1699 | 0.0674 | 0.0000 | 0.0000 | 0.0000 | 0.0027 | 5.859 | 0.0825 | 0 |
| 20 | 0.556 | 0.2058 | 0.1645 | 0.0813 | 0.0000 | 0.0000 | 0.0000 | 0.0016 | 6.795 | 0.0762 | 0 |
| 21 | 0.583 | 0.1956 | 0.1584 | 0.1138 | 0.0000 | 0.0000 | 0.0000 | 0.0008 | 8.05 | 0.0820 | 0 |
| 22 | 0.611 | 0.1925 | 0.1649 | 0.1474 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 9.632 | 0.1836 | 0 |
| 23 | 0.639 | 0.1757 | 0.1539 | 0.1553 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 12.44 | 0.2535 | 0 |
| 24 | 0.667 | 0.1525 | 0.1406 | 0.1348 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 15.5 | 0.2872 | 0 |
| 25 | 0.694 | 0.1401 | 0.1397 | 0.1399 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 17.39 | 0.2549 | 0 |
| 26 | 0.722 | 0.1157 | 0.1218 | 0.1309 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 21.17 | 0.2874 | 0 |
| 27 | 0.750 | 0.1020 | 0.1217 | 0.1328 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 24.33 | 0.2916 | 0 |
| 28 | 0.778 | 0.0926 | 0.1252 | 0.1350 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 28.08 | 0.3285 | 0 |
| 29 | 0.806 | 0.0583 | 0.1116 | 0.1277 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 37.05 | 0.3294 | 0 |
| 30 | 0.833 | 0.0604 | 0.1235 | 0.1217 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 39.92 | 0.3125 | 0 |
| 31 | 0.861 | 0.0616 | 0.1315 | 0.0979 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 41.93 | 0.2674 | 0 |
| 32 | 0.889 | 0.0641 | 0.1459 | 0.0920 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 44.23 | 0.2524 | 0 |
| 33 | 0.917 | 0.0623 | 0.1581 | 0.0859 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 47.98 | 0.3014 | 0 |
| 34 | 0.944 | 0.0508 | 0.1604 | 0.1172 | 0.0000 | 0.0000 | 0.0001 | 0.0020 | 59.03 | 0.4593 | 0 |
| 35 | 0.972 | 0.0198 | 0.1152 | 0.1206 | 0.0000 | 0.0000 | 0.0075 | 0.0063 | 64.94 | 0.5640 | 0 |

**G4**

| L | rel | dens share | q999/max | same>=0.5 | dead | RARE | pos0>=0.5 | gate-neg | med max abs | lean | dead one sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.0925 | 0.2153 | 0.5549 | 0.0000 | 0.0000 | 0.0074 | 0.0054 | 10.5 | 0.3973 | 0 |
| 1 | 0.029 | 0.0386 | 0.1066 | 0.3255 | 0.0000 | 0.0000 | 0.0016 | 0.0023 | 12.24 | 0.2735 | 0 |
| 2 | 0.059 | 0.0274 | 0.0825 | 0.3314 | 0.0000 | 0.0000 | 0.0018 | 0.0016 | 14.7 | 0.2582 | 0 |
| 3 | 0.088 | 0.0097 | 0.0677 | 0.3484 | 0.0000 | 0.0000 | 0.0014 | 0.0022 | 27.46 | 0.3840 | 0 |
| 4 | 0.118 | 0.0656 | 0.1211 | 0.2692 | 0.0000 | 0.0000 | 0.0007 | 0.0085 | 11.15 | 0.2328 | 0 |
| 5 | 0.147 | 0.0509 | 0.1020 | 0.3093 | 0.0000 | 0.0000 | 0.0011 | 0.0113 | 12.5 | 0.2744 | 1 |
| 6 | 0.176 | 0.0500 | 0.1051 | 0.2134 | 0.0000 | 0.0000 | 0.0021 | 0.0102 | 12.33 | 0.3156 | 0 |
| 7 | 0.206 | 0.1240 | 0.1471 | 0.1794 | 0.0000 | 0.0000 | 0.0018 | 0.0085 | 6.962 | 0.2108 | 0 |
| 8 | 0.235 | 0.1500 | 0.1557 | 0.1486 | 0.0000 | 0.0000 | 0.0033 | 0.0095 | 5.46 | 0.1968 | 0 |
| 9 | 0.265 | 0.1718 | 0.1568 | 0.1332 | 0.0000 | 0.0000 | 0.0549 | 0.0102 | 3.293 | 0.0870 | 0 |
| 10 | 0.294 | 0.1965 | 0.1623 | 0.1589 | 0.0000 | 0.0000 | 0.1570 | 0.0128 | 2.634 | 0.1337 | 0 |
| 11 | 0.324 | 0.1544 | 0.1451 | 0.1694 | 0.0000 | 0.0018 | 0.1628 | 0.0138 | 4.273 | 0.1861 | 0 |
| 12 | 0.353 | 0.2026 | 0.1537 | 0.0932 | 0.0000 | 0.0000 | 0.0023 | 0.0070 | 1.842 | 0.0571 | 1 |
| 13 | 0.382 | 0.1877 | 0.1420 | 0.0988 | 0.0000 | 0.0000 | 0.0039 | 0.0047 | 1.959 | 0.0388 | 0 |
| 14 | 0.412 | 0.2047 | 0.1464 | 0.0803 | 0.0000 | 0.0000 | 0.0134 | 0.0051 | 1.347 | 0.0218 | 0 |
| 15 | 0.441 | 0.1531 | 0.1386 | 0.0925 | 0.0000 | 0.0000 | 0.0082 | 0.0027 | 4.095 | 0.1523 | 0 |
| 16 | 0.471 | 0.1721 | 0.1418 | 0.0912 | 0.0000 | 0.0000 | 0.0242 | 0.0041 | 3.018 | 0.0986 | 0 |
| 17 | 0.500 | 0.1668 | 0.1416 | 0.1170 | 0.0000 | 0.0000 | 0.0113 | 0.0032 | 3.24 | 0.1493 | 1 |
| 18 | 0.529 | 0.1533 | 0.1316 | 0.1167 | 0.0000 | 0.0000 | 0.0010 | 0.0006 | 3.815 | 0.1303 | 0 |
| 19 | 0.559 | 0.1501 | 0.1313 | 0.1376 | 0.0000 | 0.0000 | 0.0007 | 0.0005 | 3.82 | 0.1098 | 0 |
| 20 | 0.588 | 0.1506 | 0.1273 | 0.1571 | 0.0000 | 0.0000 | 0.0007 | 0.0010 | 3.903 | 0.1400 | 0 |
| 21 | 0.618 | 0.1242 | 0.1171 | 0.1647 | 0.0000 | 0.0000 | 0.0009 | 0.0003 | 5.535 | 0.1896 | 0 |
| 22 | 0.647 | 0.1109 | 0.1109 | 0.1929 | 0.0000 | 0.0000 | 0.0006 | 0.0005 | 6.409 | 0.2241 | 0 |
| 23 | 0.676 | 0.0774 | 0.1007 | 0.1737 | 0.0000 | 0.0000 | 0.0009 | 0.0002 | 9.155 | 0.2780 | 0 |
| 24 | 0.706 | 0.0785 | 0.1074 | 0.1372 | 0.0000 | 0.0000 | 0.0006 | 0.0001 | 9.426 | 0.2411 | 0 |
| 25 | 0.735 | 0.0854 | 0.1079 | 0.1299 | 0.0000 | 0.0000 | 0.0011 | 0.0004 | 8.999 | 0.2309 | 0 |
| 26 | 0.765 | 0.1085 | 0.1167 | 0.0972 | 0.0000 | 0.0000 | 0.0005 | 0.0003 | 7.19 | 0.1825 | 0 |
| 27 | 0.794 | 0.1144 | 0.1194 | 0.0808 | 0.0000 | 0.0000 | 0.0001 | 0.0003 | 6.739 | 0.1647 | 0 |
| 28 | 0.824 | 0.1492 | 0.1365 | 0.0679 | 0.0000 | 0.0000 | 0.0001 | 0.0001 | 3.873 | 0.0923 | 0 |
| 29 | 0.853 | 0.1690 | 0.1402 | 0.0509 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 1.976 | 0.0470 | 0 |
| 30 | 0.882 | 0.1485 | 0.1438 | 0.0514 | 0.0000 | 0.0000 | 0.0005 | 0.0017 | 4.43 | 0.1398 | 0 |
| 31 | 0.912 | 0.1274 | 0.1419 | 0.0692 | 0.0000 | 0.0000 | 0.0006 | 0.0081 | 7.268 | 0.2689 | 0 |
| 32 | 0.941 | 0.1373 | 0.1480 | 0.1052 | 0.0000 | 0.0000 | 0.0005 | 0.0073 | 6.353 | 0.2640 | 0 |
| 33 | 0.971 | 0.0407 | 0.1090 | 0.0923 | 0.0000 | 0.0005 | 0.0030 | 0.0148 | 14.65 | 0.4607 | 0 |

**O7**

| L | rel | dens share | q999/max | same>=0.5 | dead | RARE | pos0>=0.5 | gate-neg | med max abs | lean | dead one sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.1350 | 0.2033 | 0.1146 | 0.0000 | 0.0000 | 0.0001 | 0.0506 | 0.2985 | 0.0200 | 0 |
| 1 | 0.031 | 0.1103 | 0.1692 | 0.1334 | 0.0000 | 0.0000 | 0.0000 | 0.0298 | 0.3308 | 0.0168 | 0 |
| 2 | 0.062 | 0.1182 | 0.1703 | 0.1516 | 0.0000 | 0.0000 | 0.0000 | 0.0210 | 0.2488 | 0.0195 | 0 |
| 3 | 0.094 | 0.1173 | 0.1535 | 0.1447 | 0.0000 | 0.0000 | 0.0000 | 0.0217 | 0.3123 | 0.0205 | 0 |
| 4 | 0.125 | 0.1193 | 0.1456 | 0.1028 | 0.0000 | 0.0000 | 0.0002 | 0.0180 | 0.4124 | 0.0247 | 0 |
| 5 | 0.156 | 0.1382 | 0.1491 | 0.0946 | 0.0000 | 0.0000 | 0.0004 | 0.0195 | 0.4513 | 0.0312 | 0 |
| 6 | 0.188 | 0.1565 | 0.1500 | 0.0640 | 0.0000 | 0.0000 | 0.0001 | 0.0162 | 0.5736 | 0.0432 | 0 |
| 7 | 0.219 | 0.1587 | 0.1469 | 0.0656 | 0.0000 | 0.0000 | 0.0003 | 0.0173 | 0.7181 | 0.0353 | 0 |
| 8 | 0.250 | 0.1839 | 0.1633 | 0.0437 | 0.0000 | 0.0000 | 0.0002 | 0.0129 | 0.72 | 0.0253 | 0 |
| 9 | 0.281 | 0.1859 | 0.1613 | 0.0405 | 0.0000 | 0.0000 | 0.0005 | 0.0123 | 0.839 | 0.0127 | 0 |
| 10 | 0.312 | 0.2051 | 0.1664 | 0.0285 | 0.0000 | 0.0000 | 0.0004 | 0.0108 | 0.9688 | 0.0451 | 0 |
| 11 | 0.344 | 0.2094 | 0.1690 | 0.0334 | 0.0000 | 0.0000 | 0.0003 | 0.0092 | 1.022 | 0.0378 | 0 |
| 12 | 0.375 | 0.2082 | 0.1697 | 0.0355 | 0.0000 | 0.0000 | 0.0001 | 0.0063 | 1.106 | 0.0298 | 0 |
| 13 | 0.406 | 0.2069 | 0.1639 | 0.0567 | 0.0000 | 0.0000 | 0.0004 | 0.0054 | 1.339 | 0.0279 | 0 |
| 14 | 0.438 | 0.2096 | 0.1685 | 0.0562 | 0.0000 | 0.0000 | 0.0001 | 0.0034 | 1.544 | 0.0372 | 0 |
| 15 | 0.469 | 0.2073 | 0.1631 | 0.0803 | 0.0000 | 0.0000 | 0.0002 | 0.0023 | 1.906 | 0.0502 | 0 |
| 16 | 0.500 | 0.1965 | 0.1573 | 0.1022 | 0.0000 | 0.0000 | 0.0004 | 0.0015 | 2.263 | 0.0492 | 0 |
| 17 | 0.531 | 0.1852 | 0.1555 | 0.1074 | 0.0000 | 0.0000 | 0.0004 | 0.0014 | 3.035 | 0.0669 | 0 |
| 18 | 0.562 | 0.1798 | 0.1557 | 0.0948 | 0.0000 | 0.0000 | 0.0003 | 0.0007 | 4.122 | 0.1024 | 0 |
| 19 | 0.594 | 0.1707 | 0.1508 | 0.0984 | 0.0000 | 0.0000 | 0.0001 | 0.0002 | 5.09 | 0.0882 | 0 |
| 20 | 0.625 | 0.1715 | 0.1526 | 0.1003 | 0.0000 | 0.0000 | 0.0001 | 0.0001 | 6.051 | 0.0991 | 0 |
| 21 | 0.656 | 0.1695 | 0.1511 | 0.1064 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 6.953 | 0.0971 | 0 |
| 22 | 0.688 | 0.1668 | 0.1533 | 0.0942 | 0.0000 | 0.0000 | 0.0001 | 0.0001 | 8.189 | 0.0960 | 0 |
| 23 | 0.719 | 0.1624 | 0.1606 | 0.0880 | 0.0000 | 0.0000 | 0.0001 | 0.0001 | 9.581 | 0.0862 | 0 |
| 24 | 0.750 | 0.1581 | 0.1681 | 0.0794 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 11.04 | 0.0789 | 0 |
| 25 | 0.781 | 0.1542 | 0.1799 | 0.0679 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 12.9 | 0.0862 | 0 |
| 26 | 0.812 | 0.1509 | 0.1904 | 0.0589 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 14.66 | 0.0837 | 0 |
| 27 | 0.844 | 0.1454 | 0.1948 | 0.0516 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 16.83 | 0.1062 | 0 |
| 28 | 0.875 | 0.1465 | 0.2024 | 0.0493 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 18.57 | 0.1282 | 0 |
| 29 | 0.906 | 0.1437 | 0.2063 | 0.0573 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 21.33 | 0.1830 | 0 |
| 30 | 0.938 | 0.1441 | 0.2157 | 0.0638 | 0.0000 | 0.0000 | 0.0000 | 0.0013 | 25.53 | 0.2510 | 0 |
| 31 | 0.969 | 0.1479 | 0.2236 | 0.0858 | 0.0000 | 0.0000 | 0.0000 | 0.0040 | 30.57 | 0.3429 | 0 |

### T3. Two-pole overlap, entropy, gate-closed neurons, start positions, shared positions

Columns: mean multiset Jaccard of the two poles' top-50 token ids (neurons with both poles live); share of those neurons sharing no token id; the same two under a random pairing within the layer (one permutation, numpy seed 0); median top-50 token entropy in bits; rank correlation of entropy with log density; gate-closed neurons; share of slots at window position 1; share of slots at document position 0 or 1; pole pairs from different neurons sharing at least 25 of 50 top positions. Source: [T:*], [H:*], [K].

**Q4B**

| L | rel | J | no shared id | J null | no shared null | entropy | rho(H, dens) | gate-closed | wpos 1 | doc pos 0-1 | pairs >=25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.0315 | 0.8827 | 0.0006 | 0.9903 | 1.872 | 0.293 | 7 | 0.0139 | 0.0050 | 744 |
| 1 | 0.028 | 0.0274 | 0.7032 | 0.0160 | 0.8949 | 2.949 | 0.423 | 4788 | 0.0072 | 0.0029 | 2020462 |
| 2 | 0.056 | 0.0169 | 0.7631 | 0.0159 | 0.7900 | 3.201 | 0.214 | 6449 | 0.0069 | 0.0030 | 1277841 |
| 3 | 0.083 | 0.0279 | 0.6277 | 0.0099 | 0.9293 | 2.744 | 0.104 | 3114 | 0.0102 | 0.0027 | 1865326 |
| 4 | 0.111 | 0.0230 | 0.7090 | 0.0012 | 0.9540 | 3.075 | 0.246 | 246 | 0.0034 | 0.0029 | 300 |
| 5 | 0.139 | 0.0114 | 0.8303 | 0.0014 | 0.9454 | 3.090 | 0.280 | 24 | 0.0021 | 0.0019 | 181 |
| 6 | 0.167 | 0.0130 | 0.8295 | 0.0024 | 0.9251 | 3.028 | 0.335 | 4 | 0.0072 | 0.0028 | 2573 |
| 7 | 0.194 | 0.0187 | 0.7076 | 0.0039 | 0.8800 | 3.222 | 0.381 | 1 | 0.0031 | 0.0023 | 60 |
| 8 | 0.222 | 0.0226 | 0.6100 | 0.0055 | 0.8010 | 3.388 | 0.402 | 7 | 0.0031 | 0.0015 | 12 |
| 9 | 0.250 | 0.0189 | 0.5961 | 0.0052 | 0.7696 | 3.924 | 0.223 | 23 | 0.0016 | 0.0010 | 15 |
| 10 | 0.278 | 0.0194 | 0.5694 | 0.0063 | 0.7246 | 4.033 | 0.227 | 15 | 0.0011 | 0.0010 | 23 |
| 11 | 0.306 | 0.0206 | 0.5093 | 0.0071 | 0.6801 | 4.131 | 0.253 | 5 | 0.0007 | 0.0009 | 21 |
| 12 | 0.333 | 0.0237 | 0.4615 | 0.0078 | 0.6453 | 4.236 | 0.261 | 6 | 0.0010 | 0.0008 | 21 |
| 13 | 0.361 | 0.0308 | 0.3979 | 0.0087 | 0.6284 | 4.414 | 0.222 | 7 | 0.0016 | 0.0010 | 47 |
| 14 | 0.389 | 0.0389 | 0.2765 | 0.0134 | 0.4973 | 4.557 | 0.141 | 4 | 0.0013 | 0.0011 | 49 |
| 15 | 0.417 | 0.0397 | 0.2233 | 0.0168 | 0.4016 | 4.764 | 0.132 | 7 | 0.0010 | 0.0010 | 21 |
| 16 | 0.444 | 0.0444 | 0.1793 | 0.0208 | 0.3200 | 4.781 | 0.107 | 2 | 0.0007 | 0.0011 | 33 |
| 17 | 0.472 | 0.0475 | 0.1687 | 0.0223 | 0.3092 | 4.823 | 0.118 | 1 | 0.0004 | 0.0009 | 39 |
| 18 | 0.500 | 0.0495 | 0.1582 | 0.0225 | 0.3193 | 4.859 | 0.088 | 0 | 0.0004 | 0.0006 | 23 |
| 19 | 0.528 | 0.0575 | 0.1884 | 0.0205 | 0.3835 | 4.726 | 0.124 | 0 | 0.0007 | 0.0007 | 12 |
| 20 | 0.556 | 0.0554 | 0.2162 | 0.0190 | 0.4344 | 4.623 | 0.159 | 0 | 0.0009 | 0.0006 | 10 |
| 21 | 0.583 | 0.0522 | 0.2838 | 0.0162 | 0.5019 | 4.453 | 0.198 | 0 | 0.0012 | 0.0013 | 35 |
| 22 | 0.611 | 0.0496 | 0.3518 | 0.0161 | 0.5420 | 4.198 | 0.286 | 0 | 0.0011 | 0.0025 | 22 |
| 23 | 0.639 | 0.0492 | 0.3425 | 0.0166 | 0.5343 | 4.059 | 0.341 | 0 | 0.0008 | 0.0032 | 20 |
| 24 | 0.667 | 0.0481 | 0.3223 | 0.0192 | 0.4983 | 4.073 | 0.340 | 0 | 0.0006 | 0.0030 | 14 |
| 25 | 0.694 | 0.0459 | 0.3435 | 0.0171 | 0.5320 | 4.001 | 0.361 | 0 | 0.0005 | 0.0030 | 13 |
| 26 | 0.722 | 0.0430 | 0.3300 | 0.0181 | 0.5033 | 4.054 | 0.338 | 0 | 0.0004 | 0.0024 | 18 |
| 27 | 0.750 | 0.0414 | 0.3273 | 0.0183 | 0.5005 | 4.039 | 0.288 | 0 | 0.0003 | 0.0022 | 12 |
| 28 | 0.778 | 0.0386 | 0.3376 | 0.0182 | 0.4977 | 4.013 | 0.250 | 0 | 0.0003 | 0.0018 | 15 |
| 29 | 0.806 | 0.0375 | 0.3313 | 0.0208 | 0.4701 | 4.020 | 0.171 | 0 | 0.0002 | 0.0011 | 20 |
| 30 | 0.833 | 0.0451 | 0.2915 | 0.0219 | 0.4390 | 4.083 | 0.150 | 0 | 0.0002 | 0.0006 | 28 |
| 31 | 0.861 | 0.0515 | 0.2610 | 0.0235 | 0.4164 | 4.195 | 0.191 | 0 | 0.0001 | 0.0006 | 29 |
| 32 | 0.889 | 0.0550 | 0.2415 | 0.0208 | 0.4543 | 4.259 | 0.153 | 0 | 0.0002 | 0.0005 | 20 |
| 33 | 0.917 | 0.0523 | 0.2639 | 0.0133 | 0.5672 | 4.309 | 0.160 | 0 | 0.0001 | 0.0005 | 19 |
| 34 | 0.944 | 0.0530 | 0.3543 | 0.0078 | 0.7096 | 4.092 | 0.237 | 0 | 0.0004 | 0.0006 | 1984 |
| 35 | 0.972 | 0.0395 | 0.3540 | 0.0109 | 0.5925 | 4.286 | 0.251 | 0 | 0.0035 | 0.0010 | 1216 |

**G4**

| L | rel | J | no shared id | J null | no shared null | entropy | rho(H, dens) | gate-closed | wpos 1 | doc pos 0-1 | pairs >=25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.0022 | 0.9903 | 0.0004 | 0.9936 | 1.733 | 0.093 | 5 | 0.0087 | 0.0032 | 808 |
| 1 | 0.029 | 0.0061 | 0.9615 | 0.0007 | 0.9789 | 2.844 | 0.156 | 0 | 0.0019 | 0.0029 | 275 |
| 2 | 0.059 | 0.0139 | 0.9136 | 0.0012 | 0.9729 | 2.809 | 0.179 | 1 | 0.0019 | 0.0034 | 543 |
| 3 | 0.088 | 0.0208 | 0.8133 | 0.0029 | 0.9332 | 2.727 | 0.084 | 4 | 0.0027 | 0.0032 | 240 |
| 4 | 0.118 | 0.0202 | 0.7531 | 0.0026 | 0.9214 | 3.145 | 0.176 | 10 | 0.0029 | 0.0027 | 213 |
| 5 | 0.147 | 0.0249 | 0.7502 | 0.0027 | 0.9253 | 2.970 | 0.204 | 36 | 0.0027 | 0.0024 | 4544 |
| 6 | 0.176 | 0.0306 | 0.6055 | 0.0065 | 0.7922 | 3.458 | 0.095 | 20 | 0.0013 | 0.0013 | 1726 |
| 7 | 0.206 | 0.0257 | 0.6074 | 0.0060 | 0.7946 | 3.657 | 0.136 | 29 | 0.0023 | 0.0017 | 903 |
| 8 | 0.235 | 0.0343 | 0.4373 | 0.0120 | 0.6196 | 3.859 | 0.169 | 19 | 0.0022 | 0.0015 | 415 |
| 9 | 0.265 | 0.0413 | 0.3642 | 0.0138 | 0.5589 | 4.028 | 0.178 | 17 | 0.0027 | 0.0011 | 1845 |
| 10 | 0.294 | 0.0556 | 0.2345 | 0.0274 | 0.3692 | 4.197 | 0.154 | 0 | 0.0374 | 0.0010 | 921763 |
| 11 | 0.324 | 0.0409 | 0.2759 | 0.0220 | 0.4142 | 4.298 | 0.286 | 1 | 0.0541 | 0.0008 | 2186110 |
| 12 | 0.353 | 0.0514 | 0.2207 | 0.0220 | 0.3765 | 4.451 | 0.179 | 1 | 0.0010 | 0.0009 | 421 |
| 13 | 0.382 | 0.0448 | 0.2739 | 0.0178 | 0.4369 | 4.423 | 0.168 | 0 | 0.0011 | 0.0012 | 319 |
| 14 | 0.412 | 0.0445 | 0.2436 | 0.0204 | 0.3855 | 4.458 | 0.118 | 1 | 0.0025 | 0.0011 | 762 |
| 15 | 0.441 | 0.0653 | 0.1568 | 0.0361 | 0.2500 | 4.366 | 0.161 | 0 | 0.0024 | 0.0007 | 587 |
| 16 | 0.471 | 0.0637 | 0.1489 | 0.0361 | 0.2410 | 4.379 | 0.099 | 0 | 0.0041 | 0.0008 | 1801 |
| 17 | 0.500 | 0.0629 | 0.2078 | 0.0302 | 0.3249 | 4.233 | 0.172 | 0 | 0.0023 | 0.0007 | 401 |
| 18 | 0.529 | 0.0458 | 0.2745 | 0.0214 | 0.4180 | 4.146 | 0.258 | 0 | 0.0007 | 0.0010 | 125 |
| 19 | 0.559 | 0.0462 | 0.2837 | 0.0207 | 0.4331 | 4.036 | 0.258 | 0 | 0.0006 | 0.0011 | 92 |
| 20 | 0.588 | 0.0414 | 0.3736 | 0.0160 | 0.5358 | 3.866 | 0.288 | 0 | 0.0009 | 0.0015 | 142 |
| 21 | 0.618 | 0.0432 | 0.3786 | 0.0162 | 0.5344 | 3.828 | 0.319 | 0 | 0.0007 | 0.0016 | 148 |
| 22 | 0.647 | 0.0414 | 0.3985 | 0.0164 | 0.5551 | 3.700 | 0.360 | 0 | 0.0008 | 0.0024 | 291 |
| 23 | 0.676 | 0.0402 | 0.3953 | 0.0187 | 0.5284 | 3.716 | 0.236 | 0 | 0.0010 | 0.0023 | 307 |
| 24 | 0.706 | 0.0390 | 0.3562 | 0.0206 | 0.4826 | 3.880 | 0.209 | 0 | 0.0007 | 0.0017 | 154 |
| 25 | 0.735 | 0.0361 | 0.3532 | 0.0205 | 0.4602 | 3.938 | 0.194 | 0 | 0.0008 | 0.0018 | 140 |
| 26 | 0.765 | 0.0426 | 0.2726 | 0.0246 | 0.3717 | 4.166 | 0.169 | 0 | 0.0005 | 0.0013 | 145 |
| 27 | 0.794 | 0.0530 | 0.2142 | 0.0303 | 0.3056 | 4.211 | 0.207 | 0 | 0.0003 | 0.0012 | 224 |
| 28 | 0.824 | 0.0562 | 0.1748 | 0.0307 | 0.2858 | 4.348 | 0.256 | 0 | 0.0004 | 0.0010 | 166 |
| 29 | 0.853 | 0.0744 | 0.0812 | 0.0426 | 0.1757 | 4.470 | 0.249 | 0 | 0.0003 | 0.0006 | 734 |
| 30 | 0.882 | 0.0638 | 0.1193 | 0.0358 | 0.2131 | 4.526 | 0.225 | 0 | 0.0003 | 0.0005 | 316 |
| 31 | 0.912 | 0.0534 | 0.1825 | 0.0253 | 0.3343 | 4.498 | 0.249 | 0 | 0.0006 | 0.0009 | 38 |
| 32 | 0.941 | 0.0404 | 0.2865 | 0.0182 | 0.4382 | 4.447 | 0.225 | 0 | 0.0011 | 0.0018 | 51 |
| 33 | 0.971 | 0.0509 | 0.2181 | 0.0298 | 0.3197 | 4.564 | 0.119 | 0 | 0.0007 | 0.0008 | 30 |

**O7**

| L | rel | J | no shared id | J null | no shared null | entropy | rho(H, dens) | gate-closed | wpos 1 | doc pos 0-1 | pairs >=25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.0098 | 0.8834 | 0.0003 | 0.9809 | 4.008 | 0.161 | 22 | 0.0004 | 0.0372 | 307 |
| 1 | 0.031 | 0.0109 | 0.8882 | 0.0002 | 0.9827 | 3.914 | 0.246 | 6 | 0.0002 | 0.0079 | 543 |
| 2 | 0.062 | 0.0042 | 0.9364 | 0.0002 | 0.9843 | 3.706 | 0.193 | 2 | 0.0004 | 0.0145 | 995 |
| 3 | 0.094 | 0.0025 | 0.9552 | 0.0002 | 0.9883 | 3.735 | 0.336 | 2 | 0.0005 | 0.0171 | 860 |
| 4 | 0.125 | 0.0028 | 0.9411 | 0.0002 | 0.9862 | 4.053 | 0.317 | 3 | 0.0007 | 0.0151 | 310 |
| 5 | 0.156 | 0.0044 | 0.9077 | 0.0003 | 0.9795 | 4.228 | 0.301 | 1 | 0.0011 | 0.0172 | 196 |
| 6 | 0.188 | 0.0057 | 0.8484 | 0.0006 | 0.9578 | 4.494 | 0.326 | 0 | 0.0012 | 0.0147 | 112 |
| 7 | 0.219 | 0.0087 | 0.7722 | 0.0009 | 0.9378 | 4.559 | 0.336 | 2 | 0.0013 | 0.0136 | 99 |
| 8 | 0.250 | 0.0097 | 0.7058 | 0.0014 | 0.9035 | 4.803 | 0.320 | 0 | 0.0015 | 0.0139 | 91 |
| 9 | 0.281 | 0.0114 | 0.6451 | 0.0019 | 0.8706 | 4.957 | 0.256 | 1 | 0.0013 | 0.0128 | 32 |
| 10 | 0.312 | 0.0157 | 0.5386 | 0.0035 | 0.7888 | 5.061 | 0.255 | 0 | 0.0009 | 0.0105 | 35 |
| 11 | 0.344 | 0.0207 | 0.4578 | 0.0048 | 0.7219 | 5.117 | 0.221 | 0 | 0.0009 | 0.0120 | 102 |
| 12 | 0.375 | 0.0243 | 0.4071 | 0.0065 | 0.6681 | 5.204 | 0.115 | 0 | 0.0008 | 0.0105 | 92 |
| 13 | 0.406 | 0.0265 | 0.4092 | 0.0067 | 0.6733 | 5.174 | 0.169 | 0 | 0.0008 | 0.0129 | 95 |
| 14 | 0.438 | 0.0305 | 0.3658 | 0.0086 | 0.6099 | 5.069 | 0.216 | 0 | 0.0006 | 0.0102 | 75 |
| 15 | 0.469 | 0.0370 | 0.3431 | 0.0115 | 0.5576 | 4.876 | 0.298 | 0 | 0.0009 | 0.0121 | 87 |
| 16 | 0.500 | 0.0374 | 0.3623 | 0.0119 | 0.5704 | 4.709 | 0.327 | 0 | 0.0014 | 0.0160 | 113 |
| 17 | 0.531 | 0.0339 | 0.3920 | 0.0106 | 0.6069 | 4.581 | 0.386 | 0 | 0.0016 | 0.0143 | 60 |
| 18 | 0.562 | 0.0320 | 0.3502 | 0.0134 | 0.5139 | 4.589 | 0.302 | 0 | 0.0009 | 0.0112 | 33 |
| 19 | 0.594 | 0.0276 | 0.3884 | 0.0122 | 0.5551 | 4.469 | 0.311 | 0 | 0.0009 | 0.0119 | 40 |
| 20 | 0.625 | 0.0256 | 0.4029 | 0.0115 | 0.5633 | 4.425 | 0.275 | 0 | 0.0008 | 0.0108 | 31 |
| 21 | 0.656 | 0.0278 | 0.3973 | 0.0125 | 0.5491 | 4.301 | 0.228 | 0 | 0.0006 | 0.0104 | 24 |
| 22 | 0.688 | 0.0306 | 0.3685 | 0.0138 | 0.5276 | 4.278 | 0.197 | 0 | 0.0004 | 0.0076 | 14 |
| 23 | 0.719 | 0.0354 | 0.3260 | 0.0175 | 0.4628 | 4.288 | 0.188 | 0 | 0.0003 | 0.0060 | 5 |
| 24 | 0.750 | 0.0439 | 0.2487 | 0.0239 | 0.3844 | 4.326 | 0.162 | 0 | 0.0002 | 0.0039 | 2 |
| 25 | 0.781 | 0.0583 | 0.1673 | 0.0343 | 0.2700 | 4.428 | 0.137 | 0 | 0.0001 | 0.0023 | 4 |
| 26 | 0.812 | 0.0605 | 0.1385 | 0.0357 | 0.2449 | 4.501 | 0.143 | 0 | 0.0001 | 0.0013 | 4 |
| 27 | 0.844 | 0.0619 | 0.1225 | 0.0375 | 0.2253 | 4.566 | 0.159 | 0 | 0.0001 | 0.0008 | 3 |
| 28 | 0.875 | 0.0607 | 0.1299 | 0.0362 | 0.2428 | 4.599 | 0.146 | 0 | 0.0001 | 0.0008 | 3 |
| 29 | 0.906 | 0.0563 | 0.1540 | 0.0338 | 0.2549 | 4.631 | 0.152 | 0 | 0.0000 | 0.0006 | 1 |
| 30 | 0.938 | 0.0510 | 0.1809 | 0.0281 | 0.3137 | 4.646 | 0.143 | 1 | 0.0001 | 0.0010 | 3 |
| 31 | 0.969 | 0.0485 | 0.2237 | 0.0241 | 0.3786 | 4.599 | 0.144 | 1 | 0.0001 | 0.0019 | 14 |

---

## What cuts against these

- **The top 50 is the extreme tail.** It is 50 tokens out of about 50 million. Type shares, overlaps and entropies describe where a pole is strongest, not how it usually behaves.
- **My classifier.** The function list is the brief's 20 words, not F's 90, so my function shares are lower than F's. "Other" depends on the tokenizer: Qwen's and OLMo's pre-tokenizers attach one leading non-letter to a word (".blogspot", "_path"), Gemma's does not, so Q4B's "other" share (up to 0.163 at L1) has no Gemma counterpart. Digits are single tokens in Q4B and G4 and up to three digits in O7. CJK characters count as letters, so a run of Chinese characters reads as word pieces; tokens containing CJK are under 2% of slots in every layer except Q4B L0 to L4 (up to 7.66%) and G4 L0 (2.85%).
- **Hub positions distort several statistics.** In Q4B L1 to L3 and G4 L10 to L11 a few positions fill a large share of all top-50 lists. Single-token concentration, type shares, two-pole overlap and entropy in those layers partly measure the hubs. F's single-token and position-0 numbers for those layers carry the same caveat.
- **Gate-closed is defined on the top 50.** I show the gate is negative at each neuron's 50 strongest tokens on both signs. I do not show it never opens. For neurons with a dead pole, the live pole holding every token is a stronger fact.
- **Curve similarity.** I tested 19 curves. With 1,000 draws the smallest possible p is about 0.001, and a Bonferroni bar for 19 tests at 0.05 is 0.0026; only entropy (0.001) clears it under the circular null. Density (0.004) and function words (0.003) are close. The grid, the interpolation and the choice of Spearman are mine.
- **Explainer and depth are confounded** for findings 7 and 8. Sonnet explained only the top two fifths. The logistic fit cannot separate explainer from depth. The twin analysis uses one explainer family for all three models, so shared phrasing can create twins.
- **Lean and loudness ratios are shape statistics.** The rare-side vs common-side comparison uses equal counts (top 50), but the two sides have very different token counts. I propose no mechanism for the G4 vs O7 difference. O7's balanced neurons also show a ratio below 1 (0.91 to 0.95 in the last layers, 0.761 at L16), so part of O7's pattern is not specific to leaning neurons.
- **One corpus, OLMo's own.** O7's document-start effect, its scarce within-model duplicates and its layer-0 behaviour could reflect familiarity with these exact documents and boundaries. Q4B and G4 read text they were not trained on.
- **The window-position-1 finding is about 25 windows.** It is large in slot share but rests on one boundary artifact; it says nothing about how G4 behaves on unbroken text.
- **Random pairing nulls** in finding 5 use one permutation per layer. With 9,728 to 11,008 neurons per layer this is stable, but it is one draw.
- **No interventions.** Every reading here ("detector on an offset", "dial") is an interpretation of activation statistics.

## Literature context (web search, kept separate from the data)

Nothing in this section changes a data claim above. I read abstracts or project pages; where I relied on a search summary I say so.

- **Massive activations and the first delimiter.** Sun, Chen, Kolter and Liu report a few activations up to 100,000 times larger than the rest, nearly constant across inputs, acting as bias terms and concentrating attention on their tokens ([arXiv 2402.17762](https://arxiv.org/abs/2402.17762)). Their project page says that in LLaMA2-7B they occur on "the starting token, and the first period "." or newline "\n" token" ([project page](https://eric-mingjie.github.io/massive-activations/index.html)). G4's layer 10 and 11 hub is a "\n" at window position 1, right after BOS and a "\r". That fits the pattern, but I did not measure residual-stream activations, so it is a match in description only.
- **Attention sinks.** Xiao and coauthors show that initial tokens absorb attention and that keeping them restores windowed attention ([arXiv 2309.17453](https://arxiv.org/abs/2309.17453)). A 2025 paper asks why LLMs attend to the first token and, per a search summary I did not verify in the paper, argues it limits over-mixing between tokens ([arXiv 2504.02732](https://arxiv.org/abs/2504.02732)). The study's position-0 exclusion ([S] §4.2) is a response to this phenomenon; finding 4 says the exclusion was not enough for G4.
- **Dead and token-detector neurons.** Voita, Ferrando and Nalmpantis find that in OPT models (ReLU) some layers of the 66B model have over 70% of neurons that never fire, and that many live early neurons detect specific tokens or n-grams ([arXiv 2309.04827](https://arxiv.org/abs/2309.04827)). Q4B's gate-closed neurons are a gated-MLP analogue that still leaks a small signal through the SiLU negative lobe; I found no paper that measures gate-closed neurons in Qwen.
- **Super weights.** Yu and coauthors report single down-projection weights, usually in early layers, that create a large "super activation" in Llama, Mistral and OLMo ([arXiv 2411.07191](https://arxiv.org/abs/2411.07191); details from a search summary). I did not look at weights.
- **Rising activation scale.** Heimersheim and Turner report that residual-stream norms grow roughly exponentially over the forward pass in many models ([LessWrong](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward)). OLMo 2 moved RMSNorm from sublayer inputs to sublayer outputs and added QK-norm ([arXiv 2501.00656](https://arxiv.org/abs/2501.00656)); a search summary says OLMo 3 keeps this layout. If the MLP input is the unnormalized residual stream, MLP activations would grow with the residual norm, which is one candidate reading of O7's monotone 123-fold rise in median max |a| ([F]). I did not check the O7 config, so this stays a hypothesis.
- **Stages of inference.** Lad, Gurnee and Tegmark propose depth stages (detokenization, feature engineering, prediction ensembling, residual sharpening) from layer-deletion experiments ([arXiv 2406.19384](https://arxiv.org/abs/2406.19384)). Finding 2's word-piece-to-function-word shift and the last-layer reversals are consistent with that picture.
- **Universal neurons.** Gurnee and coauthors find about 1% to 5% of GPT-2 neurons universal across random seeds ([arXiv 2401.12181](https://arxiv.org/abs/2401.12181)). Finding 8 is a much weaker, description-based, cross-family version.
- **Final-layer neurons that act through the norm.** Stolfo and coauthors describe entropy neurons with large weight norms that act through the final LayerNorm ([arXiv 2406.16254](https://arxiv.org/abs/2406.16254)). Whether the leaning final-layer neurons of finding 6 include such units is untested.

## The three findings I would stand behind

First, Qwen's near-silent early layers are gate-closed: 14,710 Q4B neurons keep the gate negative at all their strongest tokens, every neuron with a dead pole is live on every token on the other sign, and thousands share the same top positions, which is also why the explainer wrote near-identical descriptions for unrelated sampled neurons there (finding 1). Second, the move from word pieces to function words holds on every neuron of all three families, and its per-layer curves, with density, single-token concentration and top-50 entropy, match across models beyond a circular-shift null, while activation scale, sharpness and dead share do not (findings 2 and 3). Third, the study's position-0 exclusion left a sink one token to the right in Gemma: 25 windows that split a carriage return from its line feed fill 5.19% of all top-50 slots in G4 layer 11 and dominate 2,109 poles (finding 4). Each rests only on activation files and the corpus, not on any explainer.
