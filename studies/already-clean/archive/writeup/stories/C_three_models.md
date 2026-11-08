# C. What the explainer could and could not explain, across three models

Angle: the descriptions themselves. What an LLM explainer wrote about 1,800 neuron poles in Qwen3-4B (Q4B), Gemma 4 (G4) and OLMo 3 7B (O7), which descriptions held up when a blind writer turned them into new texts, which did not, and why.

## Sources and conventions

Paths are relative to `/Users/abaybektursun/projects/interp_research-neurons-explain` unless absolute.

- **[M]** `bank/w5/verdict/metrics.json` (study metrics, field named each time).
- **[P:Q4B] [P:G4] [P:O7]** `bank/w5/verdict/samples/q1_<cell>_poles.jsonl`, 600 rows each. Fields named each time (`class`, `explanation`, `explainer_saw`, `texts`, `shuffled`, `blind_max_score`, `label_2`, `explainer_model`, `pos0_share`, `double_negative_share`).
- **[E:q4b] [E:g4] [E:o7]** `bank/w2c/<cell>/explain/explanations.jsonl` (sample 0 is the explainer, sample 1 the second reader, with `confidence` and `jobs`).
- **[T0]** `bank/w4/o7_t0_g2/explain/explanations.jsonl` (the untrained-checkpoint control that feeds [M] `G2`).
- **[MAN]** `bank/w2c/<cell>/explain/manifest.json`, field `models_by_role`.
- **[R]** `bank/w5/RESULTS.md`. **[S]** `/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/SPEC_D1.md`.

"My count" means I computed it from the named fields with a short read-only Python pass. Counts that depend on a text pattern I chose are marked "my regex". Scores are the probe score of a text divided by the pole's own corpus max |a|; a text is strong at θ_s = 0.3153 ([M] `theta.theta_s`). Context snippets are copied from `explainer_saw`, where ⟦ ⟧ marks the firing token and ↵ is a newline (I write " / " for ↵).

Who did what ([MAN] `models_by_role`, [M] `kappa_models`): the explainer was Opus 5.5 for 463 / 455 / 426 poles and Sonnet 5.5 for the rest; the writer was Sonnet 5.5 for every pole; the second reader was DeepSeek V4.1 Flash for every pole.

---

## The ranked stories

### 1. Qwen3-4B has near-silent neurons that pass as one-job detectors

**Hook:** Five of the eight random layer-1 neurons sampled from Qwen3-4B behave like one near-silent neuron, and the test calls each of them clean.

**What the data shows.** The audit drew 8 neurons from Q4B layer 1. Five of them (N5391, N7440, N7930, N8338, N9087) have one working pole whose strongest activation in 52.4 million tokens is between 0.002657 and 0.005143. The poles of the other three layer-1 neurons in the sample peak at 0.8966 to 17.22. The five working poles sit above 1% of their own max on 47.5 to 52.4 million of the 52,442,093 scored tokens: they are on almost everywhere, at a whisper. Their top 30 contexts are nearly the same list. Pairs share up to 27 of 30 marked contexts, all pieces like ".blogspot" and "oreferrer". The other pole of each is dead or nearly so. Two never fired once (max |a| 0). Two fired above 1% of max on 12 and 18 tokens, mostly " thaimassage" in Swedish spam. Because every text is scored against the pole's own maximum, a whisper can clear θ_s. All five working poles are ONE. The shuffled-pole control is what notices: for L01N5391- all five positives also fired a random same-layer pole strongly, and so did three of its near misses. Layer 3 holds a smaller family on the economics archive RePEc and on Freelancer.com listings, with the signs swapped between two neurons. Gemma and OLMo show none of this.

**Evidence.**
- [P:Q4B] `explainer_saw` stat lines ("max |a|", "density"): L01N5391- max 0.003488, density 48,247,315; L01N7440- max 0.005143, density 52,442,092 of 52,442,093; L01N7930- 0.003877, 52,440,003; L01N8338+ 0.003449, 49,489,291; L01N9087- 0.002657, 47,506,361. Other layer-1 neurons: L01N2695- 17.22, L01N3832- 15.24, L01N326- 0.9611.
- Other poles: L01N7440+ and L01N7930+ max 0, density 0; L01N5391+ max 0.000409 (18 tokens); L01N8338- max 0.0002993 (12 tokens). All RARE.
- Shared marked contexts (my count, same-layer pairs from different neurons): L01N5391- with L01N8338+ 27 of 30; with L01N9087- 26; L01N8338+ with L01N9087- 25. Typical context: `Byou%2B2.jpg / https://2.bp⟦.blogspot⟧.com/-1c2X` and `gur.com/aTpo0.png" rel="n⟦oreferrer⟧"><img src`.
- Quote, Q4B L01N5391- (ONE, 5 of 5 positives strong): "Fires on the second or later word-piece inside a rare multi-token name, domain or code identifier, especially in URLs and HTML/code: ".blogspot" in "2.bp.blogspot.com", "oreferrer" in rel="noreferrer"..."
- Quote, Q4B L01N7440+ (RARE): "This pole never fires anywhere in the corpus (zero tokens with positive activation, max of 0), so no input pattern can be identified, and no short text is expected to drive it."
- [P:Q4B] `shuffled` for L01N5391- (shuffle pole L01N6545-): own scores 0.94, 0.33, 0.53, 0.43, 0.79; shuffled-pole scores 0.96, 0.42, 0.71, 0.38, 0.33, all strong. L01N7930- on L01N9000-: 5 of 5 strong.
- Layer 3 ([P:Q4B]): L03N8511+ Freelancer.com boilerplate, L03N8511- the "c" of RePEc; L03N9234+ RePEc, L03N9234- Freelancer.com; L03N8169+ RePEc. Their maxima are 0.01985 to 0.4918, against 1.33 to 28.88 for the other layer-3 neurons sampled. Shared contexts: L03N8169+ with L03N8511- 22; L03N8511+ with L03N9234- 20.
- Three models (my count): poles below 1% of the model's median max |a|: Q4B 23 (layers 1 to 3), G4 0, O7 0. Cross-neuron same-layer pairs sharing 5 or more of their top-30 contexts: Q4B 22, G4 1, O7 1. RARE: 8 / 0 / 0 ([M] `fractions.RARE.k`). Shuffled strong including RARE: 55 of 8,925 / 26 of 8,940 / 23 of 8,985 ([M] `strong_rates_including_rare.shuffled`), and 20 of Q4B's 55 are in layer 1 (my count).

**Why an AI researcher cares:** a score normalized to each pole's own maximum cannot tell a crisp detector from a near-dead neuron, so a "one job" count needs a magnitude floor; and Qwen3-4B's early layers differ in kind from Gemma's and OLMo's.

---

### 2. The Sudoku neuron

**Hook:** One Qwen3-4B neuron fires one way on the digit a Sudoku row forces, and the other way on a digit that breaks the rule.

**What the data shows.** Q4B L13N2197 has two ONE poles. The explainer got the negative pole exactly: the last digit in a no-repeat row, and the last letter of an Enigma rotor alphabet, where each symbol must appear once. The positive pole was described as "unpredictable" keyboard mash. Its 30 contexts say something sharper. Its top context is the same published Sudoku solution the negative pole loves, but the marked cell is the second 4 in "3 8 2 1 4 4 5 9 7", a row that breaks the rule. In code it marks the second 1 in `{ 3, 0, 0, 4, 8, 1, 1, 7, 9}`. In fake phone numbers it marks a repeated digit. In my count, 27 of 28 single-symbol marks on the positive pole repeat a symbol from the previous 12 characters. On the negative pole, 0 of 30 do. One sign means "this symbol must come because it has not appeared". The other means "this symbol already appeared". The description that passed for the positive pole is looser than what the neuron seems to do.

**Evidence.**
- [P:Q4B] L13N2197- (ONE, positives 5 of 5 strong, near 0, `blind_max_score` 0.83, `label_2` TWO-REL): "Fires on single digits late in a row of a Sudoku-style grid ..., and on the last letters of a permutation alphabet such as an Enigma rotor wiring string, where the remaining symbol is forced because each appears exactly once." Contexts: `9 4 8 3 7 6 2 1 5 / 7 3 9 1 5 4 6 2 ⟦8⟧`; `UWYGADFPVZBECKMTHXSLRINQO⟦J⟧`.
- [P:Q4B] L13N2197+ (ONE, positives 3 of 5, `blind_max_score` 0.57): "Fires on a single digit (or letter) inside a long run of made-up, arbitrary characters where the next symbol is unpredictable: a digit in the middle of a fake phone number or ID typed by mashing number keys (like 7896542 or 33789987), a cell of a sudoku grid or number matrix..." Contexts: `Sudoku solution from March 3, 2021 / ... / 3 8 2 1 4 ⟦4⟧ 5 9 7`; `new int[] { 3, 0, 0, 4, 8, 1, ⟦1⟧, 7, 9} };`; `'7896542⟦2⟧'`; `C I Q⟦ Q⟧`.
- Repeat check (my count over `explainer_saw` marks): positive pole 27 of 28 single-symbol marks repeat a symbol within the preceding 12 characters; negative pole 0 of 30.

**Why an AI researcher cares:** the two signs of one neuron can encode a clean logical opposition (forced versus repeated), and a passing description is not always the sharpest one.

---

### 3. One phrase, two signs: "in the process"

**Hook:** Deep in Qwen3-4B, one neuron's two poles fire one token apart inside "in the process", and the explainer says both predict "process".

**What the data shows.** L31N2783+ fires on "the" in "in the process". L31N2783- fires on "in", one token earlier. The same sentence appears in both top-30 lists, marked one token apart. Both are ONE. Both explainers wrote that the pole predicts the next word "process". Both saw the same output-vocabulary list, headed by " process" (+3.14). That list is the neuron's weights times the unembedding, not signed by pole. I checked: the list text is identical for both poles of all 300 neurons in each of the three models. A negative activation multiplies those weights by a negative number, so by direct-path arithmetic the negative pole pushes "process" down at "in", where "the" must come first. That is my reasoning, not a study measurement, and the explainer did not say it. A second deep neuron shows the cost. L31N6653+ fires just before "reasonable" in legal text and passes. L31N6653- got nearly the same description, the writer put "reasonable" in its texts, and they scored 0.00, 0.04 and 0.29. Its blind max was 0.01.

**Evidence.**
- [P:Q4B] L31N2783+ (ONE): "Fires on the word " the" when it comes right after "in" in the phrases "in the process of …" or "in the process" ..., where the model is predicting that the next word will be "process"." Context: `chipping a tooth in⟦ the⟧ process! Howev`.
- [P:Q4B] L31N2783- (ONE): "Fires on the preposition "in" (also "during", "throughout", "part") right before "the process", mostly in the idiom "X, in the process" for a side effect that happens while doing something, so it predicts the next word "process"." Context: `ing out with his teeth, chipping a tooth⟦ in⟧ the process! H`.
- Both `explainer_saw`: `Output vocabulary, most promoted: " process" (+3.14), "过程" (+3.04), "process" (+2.84) ...`. Identical vocabulary text for both poles: 300 of 300 neurons in Q4B, G4 and O7 (my count).
- L31N6653+ (ONE, `blind_max_score` 0.59), writer positives ending right before the word: "...caused by events beyond its" 0.69; "The Licensee agrees to take all necessary and" 0.83; "...beyond a" 0.77. L31N6653- (UNK, `blind_max_score` 0.01): "Fires on the token just before a likely 'reasonable/reasonably' continuation..."; its positives contain "reasonable" and score 0.00, 0.04, 0.29.
- Scale (my regex for next-word language): in the deepest third, both poles of a neuron are described in next-word terms for 40 of 100 Q4B neurons, 45 of 97 G4, 54 of 94 O7. Same quoted target word on both poles, examples: Q4B L31N1192 "require", L31N3418 "property", L32N8029 "poor"; G4 L27N1674 "city", L27N10083 "write"; O7 L29N10799 "particular"/"specific".

**Why an AI researcher cares:** the output-centric view handed to the explainer is sign-blind, so for negative poles it invites a "predicts X" story that may run backwards.

---

### 4. Copy neurons cannot be caught with one sentence

**Hook:** When the explainer saw a neuron copying something from earlier in the same document, the writer's short texts almost never fired it.

**What the data shows.** 44 descriptions across the three models say the pole completes or repeats something seen earlier in the same text. 36 are UNK. They cluster in the middle third of each model, which is also where the Opus-explained ONE share is lowest. Their writer texts fire weakly: median blind max 0.128, against 0.626 for every other pole. The explainer was told to write a fixed phrase when a job cannot be carried by a short text. It never used that phrase. It did warn in its own words 43 times, and 28 of those poles are UNK. The one striking success is a neuron that notices a typo of a name spelled correctly earlier. The writer built tiny two-sentence setups for it, and one scored 1.17.

**Evidence.**
- My regex over `explanation` ("earlier in the same text/document", "repeated verbatim", "being repeated", "copy or induction" and close variants): Q4B 20, G4 16, O7 8. Classes: 36 UNK, 8 ONE. Middle tercile: 31 of 44.
- Blind max (my count from `blind_max_score`): median 0.128 for these 44, 0.626 for the other 1,746 poles with texts. Strong positives: 49 of 220 (22%) against 5,710 of 8,730 (65%).
- [M] `untestable_phrase`: 0 of 600 in each model contain "cannot be reproduced in a short text". My regex for short-text or long-document warnings: 43 descriptions, 28 UNK.
- Opus-explained ONE share by tercile (my count from `explainer_model`, `tercile`, `class`): Q4B 0.738 / 0.515 / 0.585; G4 0.695 / 0.513 / 0.704; O7 0.620 / 0.466 / 0.643 (deep O7 Opus n = 14).
- Quote, Q4B L19N1020- (UNK, `blind_max_score` 0.10): "Fires on a token in the middle of a phrase, URL, title or code expression that is being repeated verbatim from earlier in the same text, so the token can be predicted by copying..."
- Quote, Q4B L16N1066- (ONE, positives 4 of 5): "Fires on the first word-piece of a misspelled or garbled rendering of a name or word that was already spelled correctly earlier in the same text (e.g. "Criag Venter" after "Craig Venter", "Esptein" after "Epstein", "Hirshsprung" after "Hirschsprung", "infleuncing", "Issac")..." Writer text scoring 1.17: "Hirschsprung disease affects the large intestine of newborns. Doctors diagnose Hirshsprung disease with a rectal biopsy in most cases."

**Why an AI researcher cares:** a test built on short standalone texts cannot carry in-context jobs, so part of UNK is a limit of the instrument, not evidence of messy neurons.

---

### 5. The invisible carriage return

**Hook:** Three Gemma neurons fire hardest on Windows line endings, and the character was invisible to the explainer's prompt.

**What the data shows.** In G4 L10N1890-, L10N4243- and L11N9800-, 25 of the 30 marked contexts are a newline that follows a carriage return. The prompt shows newlines as ↵ but leaves the carriage return as a raw control character. No description mentions it. No writer text contains one. L10N1890- became "a 'this newline is inside program code' detector", and the writer's tidy code samples scored 0.00 to 0.07. The explainer did notice something odd: its top five contexts all sit at exactly -6.717, and it wrote "saturating at the same maximum in almost any code file". There is one more twist. Before the study dropped the first token of each 2,048-token window, all 50 of each pole's strongest contexts sat on that context-free first token.

**Evidence.**
- [P:G4] `explainer_saw`: 25 of 30 marks preceded by "\r" in each of the three poles (my count). Raw form of a mark: `'\r⟦↵⟧'`. Context: `isFraud']==0)] / 2313206 /len(notfrauds) # 36% rows / / \r⟦ / ⟧#checking occurences`. First five activations of L10N1890-: -6.717 each.
- Quote, G4 L10N1890- (UNK-F): "Fires on the newline token that ends a line inside source code (C#, Python, JavaScript, SQL), especially a line that closes a statement or block (after `;`, `}`, `)`, or a comment line), saturating at the same maximum in almost any code file; it is a 'this newline is inside program code' detector, not a detector for newlines in prose."
- `texts`: 0 of 15 positives across the three poles contain "\r" (my count). L10N1890- positives 0.07, 0.00, 0.01; `blind_max_score` 0.051. L11N9800- `blind_max_score` 0.0026. L10N4243- was described as DOI digits that "also fires on line breaks between statements in source code"; UNK.
- `pos0_share` = 1.0 for all three ([S] §4.2 defines it as the share of the pre-exclusion top 50 at window position 0). L10N1890- and L10N4243- share 25 of 30 contexts (my count).

**Why an AI researcher cares:** how the evidence is rendered is part of the instrument; an invisible byte produced a confident, wrong description and a failed test.

---

### 6. The second reader never sees two jobs

**Hook:** In 1,799 independent second reads, DeepSeek V4.1 Flash never once labelled a pole as having unrelated jobs.

**What the data shows.** The explainer said UNREL 41 times. 19 of those survived the rule that each job's own texts must fire the pole. The second reader said UNREL zero times. It folds odd pairs into one story. For an OLMo pole whose two jobs were "`new Random()` in C#" and "the word rhyme", each backed by its own texts, the second reader wrote a single job about "rhymed suffix ... code syntax, or wordplay". Raw label agreement is high, 81% to 84%, because both readers mostly say ONE. Kappa, which corrects for that, is 0.07 to 0.18, and it is below zero for Gemma poles explained by Sonnet. The explainer said "I don't know" on 32 poles that had activity. The second reader agreed once. Yet 8 of the explainer's 32 hedged best guesses passed the test.

**Evidence.**
- [P:*] `label_2` counts: Q4B ONE 549, TWO-REL 46, UNK 5; G4 ONE 529, TWO-REL 69, UNK 1, missing 1; O7 ONE 561, TWO-REL 38, UNK 1. Explainer `label` UNREL: 13 / 12 / 16. [M] `fractions.UNREL.k`: 8 / 7 / 4.
- Raw agreement (my count, `label` against `label_2`): 505 of 600, 488 of 599, 498 of 600. [M] `kappa_label`: 0.180, 0.167, 0.070. [M] `by_explainer_model` G4 Sonnet `kappa_label` -0.0146.
- [E:o7] L06N4452- sample 0 jobs: "In C# source code, the type name Random immediately after the keyword new when creating a random-number generator, e.g. "static Random rand = new Random();"." and "The word rhyme or the "ym" piece inside rhymed, rhyming or rhymes, in prose describing poems, stanzas, rhyme schemes or poetic devices." [P:O7] `job_support` both true. Sample 1 (ONE, confidence 0.35): "This neuron fires most strongly on the second half of a word ending in an -ing or rhymed suffix, particularly in the context of describing rhyme schemes, code syntax, or wordplay..."
- Other supported UNREL pairs ([E] sample 0 `jobs`): Q4B L07N4960+ the surname Harris and "The token 'useState' inside a React import statement"; G4 L05N7786- the "ine" of "Sistine" and the "7" of "COP27"; G4 L04N4329- "blouse" and the inch mark in 12".
- Explainer said UNK ([P:*] `unk_reason` "explainer said UNK"): 10 / 9 / 13. Second reader UNK on those: 1 (G4 L23N5378+). `passes` true: 4 / 1 / 3. Example, Q4B L02N5112+ (5 of 5 strong): "Fires (weakly and diffusely) on the second piece of a split word or phrase. The most reproducible cases are 'just' in 'not just' (as in 'not just one but two', 'ate not just the rats') and 'ball' in 'snowball' used as a verb ('costs can snowball into major issues')."

**Why an AI researcher cares:** the share of "unrelated" jobs depends on which model reads the neuron; one reader can manufacture unity that the other reader's texts refute.

---

### 7. Neurons for machine-written prose

**Hook:** All three models carry poles the explainer described as tuned to AI-written web copy, and the AI writer mostly failed to trigger them.

**What the data shows.** Eight descriptions name AI-style, AI-written, AI-generated or ChatGPT-style prose. Two pass. The failures are the funny part. The writer is Sonnet 5.5. Asked for "ChatGPT-style" prose full of "crucial, pivotal, foster", it produced a perfect parody: "In today's ever-evolving digital landscape, it is crucial to foster a comprehensive and holistic approach that plays a pivotal role..." That sentence scored 0.27, under the 0.3153 line. The corpus contexts are quieter than the caricature: "The Davis Center plays a crucial role at Williams College by offering support...". Several explainers wrote that these poles build up over long passages, which short texts cannot supply. The two that pass are narrower: terse, article-dropping explainer prose in Qwen, and the paragraph break between AI-written FAQ answers in OLMo.

**Evidence.**
- My regex over `explanation`: Q4B 5 (L17N3004+, L17N3004-, L19N6516+, L24N2529+, L24N2529-), G4 1 (L22N4049+), O7 2 (L06N8439-, L20N10809-). ONE: Q4B L24N2529+ and O7 L20N10809-.
- Quote, Q4B L24N2529- (UNK, positives 2 of 5): "Fires on ordinary connective tokens (to, and, is, In, commas, periods, paragraph breaks after headings) inside polished, formulaic, ChatGPT-style expository or promotional English prose, the kind packed with words like crucial, pivotal, foster, enhance, comprehensive, testament and plays a crucial role, rather than on any particular word..." Context: `The Davis Center plays a crucial role at Williams College by offering support and resources tailored⟦ to⟧ the needs`. Writer blind max text, score 0.27: "In today's ever-evolving digital landscape, it is crucial to foster a comprehensive and holistic approach that plays a pivotal role in enhancing engagement, and this is a true testament to the transformative power of innovation."
- Q4B L19N6516+ (UNK, 0 of 5, `blind_max_score` 0.13), writer: "...harness the power of innovation and embark on a transformative journey, a true testament to what's possible when you unlock your full potential."
- Quote, Q4B L24N2529+ (ONE, 4 of 5): "In terse, AI-generated-sounding explainer or SEO web prose (compressed sentences that drop articles, such as "Sponsorship demonstrates employer support and confirms job necessity")..."
- Quote, O7 L20N10809- (ONE, 5 of 5): "In SEO-style Q&A web articles made of a question heading followed by an answer paragraph (especially AI-generated FAQ text), this fires at the end of an answer paragraph (the period plus blank line) before the next question heading..."

**Why an AI researcher cares:** web-trained models hold internal detectors for model-written text, and a caricature written by another model does not reliably reach them.

---

### 8. Random neurons still get job titles

**Hook:** On untrained OLMo, the explainer called 18 of 40 random neurons one-job, though never with confidence above 0.40.

**What the data shows.** The study ran the explainer on 40 poles of OLMo's random-init checkpoint as a control. It passed on its own terms: no pole got confidence above 0.5. But the labels alone look like a trained network: 18 ONE, 20 UNK, 2 UNREL. The descriptions give the game away. 28 of 40 talk about firing deep inside long documents, as if position were the only signal. Some name single documents: one random neuron was said to fire on "the Zambian DJ H-Mac/Daev song story". The same 40 poles at the final checkpoint, explained by the same model, got mean confidence 0.407 instead of 0.171, and only 4 of 40 descriptions mention long-document position.

**Evidence.**
- [T0]: 40 sample-0 answers, all `model` Sonnet 5.5. `class`: 18 ONE, 20 UNK, 2 UNREL. `confidence` from 0.08 to 0.40, mean 0.171 (my count). [M] `G2`: n_poles 40, k_confidence_gt_0.5 0, line 0.1.
- [E:o7] same 40 poles, sample 0, all Sonnet 5.5: mean confidence 0.407; 30 ONE, 6 TWO-REL, 3 UNK, 1 UNREL (my count).
- My regex for long-document position: 28 of 40 at T_0, 4 of 40 at T_15.
- Quote, T_0 L25N2814+ (ONE, 0.08): "Fires on ordinary tokens (word pieces, pronouns, quote marks, punctuation) inside a few specific, idiosyncratic web articles, namely the Zambian DJ H-Mac/Daev song story, the Sony TV/Hulu/Pandora rant and the Olympus Has Fallen re..." The same pole at T_15 (TWO-REL, 0.3) is about hair-loss prose.
- T_0 L28N1238+ (ONE, 0.4) is C# doc-comment syntax; at T_15 (ONE, 0.8): "Fires on the word just before an expected "long" in duration phrases about how much time something lasts or took ("didn't last for long", "for so long", "took so long", "for too long")..."

**Why an AI researcher cares:** a label count alone would call half of a random network clean; the explainer's confidence and the writer test carry the signal, not the label.

---

### 9. Down the stack, neurons become "the word before a word", and the essay's suppressors never appear

**Hook:** Deep descriptions turn from what a token is into what comes next, and not one single-token suppressor exists in any model's last layer.

**What the data shows.** Shallow and middle descriptions name a token or a topic: "Zinnia", "RePEc", grief. Deep descriptions name the next word: "the token right before 'require' is expected". This shift rises with depth even for one explainer model. Opus uses next-word language for 8% to 11% of shallow poles and 29% to 44% of the shallow end of the deep third. Sonnet, which wrote most deep descriptions, uses it for 74% to 77%. So the deep picture mixes depth with explainer style. Last layers are hard: Gemma's layer 33 has 1 ONE pole of 16. The essay expects some last-layer neurons to be single-token suppressors. The study's rule asks for the strongest output weight to be negative and at least twice the second. None qualified in any model. In the 26 last-layer neurons sampled, the top entry is at most 1.27 times the second, often because the top two are one word in two spellings.

**Evidence.**
- My regex for next-word language over `explanation`, by `tercile` and `explainer_model`: Opus Q4B 17/202, 38/196, 22/65; G4 22/210, 58/191, 24/54; O7 21/208, 43/204, 4/14. Sonnet deep third: Q4B 101/135, G4 104/140, O7 134/174.
- [M] `by_depth_tercile` ONE: Q4B 0.735 / 0.515 / 0.535; G4 0.689 / 0.505 / 0.485; O7 0.620 / 0.466 / 0.463.
- Last layer ONE (my count from `layer`, `class`): Q4B layer 35, 6 of 16; G4 layer 33, 1 of 16; O7 layer 31, 4 of 20.
- [M] `single_token_suppressors`: 0, 0, 0. Rule: [S] §2.3. Ratios from the output-vocabulary lines in `explainer_saw` (my count): maximum 1.27, Q4B L35N5401 (" k" 0.468 against "{k" 0.368). Same-word pairs: Q4B L35N1588 " die" 1.35 and " Die" 1.27; G4 L33N4341 " index" 1.21 and "index" 1.19; O7 L31N3262 " loved" -0.251 and " Loved" -0.243.
- Quote, Q4B L31N9536- (ONE): "Fires on the token "a" (or "-a" in "once-in-a-lifetime") when it is the article inside a fixed idiom that predicts "lifetime" next, such as "once-in-a-lifetime" or "last a lifetime"..."
- Quote, O7 L31N10276- (UNK): "Fires when the next word is likely to begin with the letter G, mainly the next word in a G-alliteration chain or a G-initial acronym expansion..."

**Why an AI researcher cares:** a 2x single-token test cannot fire when every word has several spellings in the vocabulary, so the "say or just think" motif needs a word-level test; and depth effects here cannot be separated from which model wrote the description.

---

### 10. Same corpus, different obsessions

**Hook:** All three models read the same documents, and each grew its own fixations.

**What the data shows.** Gemma is obsessed with securities disclaimers: "forward-looking statements" appears in the top contexts of 41 Gemma poles and 8 in each other model. Gemma also likes DOI prefixes and the context-free first token of a window. Qwen has the near-silent families of story 1 and most of the double-negative poles. OLMo is the only one whose sampled descriptions mention COVID. Some jobs recur. Machine-translated Russian spam ("I think, that you are mistaken. Write to me in PM, we will communicate.") has a neuron in Qwen and in Gemma, both in layer 26. Gambling text has poles in all three. One Gemma layer-1 neuron is Nigerian on both sides: the start of Yoruba and Igbo surnames, and the word "such" in Nigerian news.

**Evidence.**
- "forward-looking" in `explainer_saw`: G4 41 poles, Q4B 8, O7 8. In `explanation`: G4 16, Q4B 0, O7 1 (my count). Quote, G4 L14N8610- (ONE): "Fires inside boilerplate legal disclaimers, chiefly the forward-looking-statements / safe-harbor paragraph of corporate press releases..."
- My regex over `explanation`: "DOI" G4 14, Q4B 3, O7 1; "COVID" O7 4, others 0; casino or gambling terms Q4B 14, G4 19, O7 22.
- `pos0_share` of 0.5 or more: G4 9 poles, Q4B 2, O7 0 (my count). [M] `double_negative_ge_0.8.n`: 16 / 2 / 1.
- "Write to me" in `explainer_saw`: Q4B L20N7603-, L26N5319+; G4 L21N7662+, L26N2364-, L28N576-; O7 none. Quote, Q4B L26N5319+ (ONE, 5 of 5): "Fires inside English text that is recognizably Russian in origin, most strongly in the stock machine-translated Russian spam comments ("I think, that you are mistaken. Write to me in PM, we will communicate.", "Many thanks for the help in this question, now I will not commit such error.", "In it something is.")..." Quote, G4 L26N2364- (ONE): "Fires on the word right before an expected "private"/PM, mainly "in" in the spam-comment phrase "Write to me in PM"..." Context: `I think, that you are mistaken. Write to me⟦ in⟧ PM, we will communicate.`
- G4 L01N2838+ (ONE): first piece of Yoruba or Igbo surnames, context `Mr Bamidele⟦ O⟧dugbesan`. G4 L01N2838- (ONE, 5 of 5): "Fires on the word "such" (as in "such a problem", "such incidents", "no such deal") when the surrounding text is Nigerian news or opinion prose..."
- [M] `fractions.ONE.p`: Q4B 0.597, G4 0.563, O7 0.518.

**Why an AI researcher cares:** neuron vocabularies differ by model family on identical text, so a "clean share" partly measures which boilerplate a model chose to memorize.

---

### 11. Sourdough and lava

**Hook:** Asked for "the most distant concept it can invent", the writer reached for bread or a volcano in 71% of poles.

**What the data shows.** For each pole the writer wrote five texts on the most distant concept it could think of, as a control. Across 1,790 poles, the distant set mentions sourdough, dough, loaves, flour or a boule in 757, and lava, magma or an eruption in 524. That is 1,279 poles with one or the other. Bread alone fills 2,572 of the 8,950 distant texts. It almost never matters, because only 7 bread texts fired strongly. But one Gemma neuron fired on all five of its sourdough texts. Its description was about the word "Satellite" and the start of "Datasheet".

**Evidence.**
- My regex over `texts.distant`: bread words in 271 / 260 / 226 poles, volcano words in 149 / 191 / 184, either in 419 / 450 / 410; bread texts 906 / 911 / 755 and volcano texts 539 / 719 / 680 of 2,975 / 2,980 / 2,995 distant texts. Strong bread texts: 2 / 4 / 1.
- [M] `strong_rates.distant`: 2 of 2,955, 11 of 2,980, 2 of 2,995.
- [P:G4] L00N770- (UNK): all five distant texts strong, 0.33 to 0.57, including "She scored the top of the boule with a razor blade just before sliding it into the oven." (0.39) and "If your starter smells like vinegar, feed it more often and keep it a bit warmer." (0.44). Its description: "A token-identity neuron in the first layer whose strongest hits are a few specific tokens: the capitalized word "Satellite" ..., the piece "Datas" that begins "Datasheet"..."
- Writer identity: [MAN] `models_by_role.answered.writer`, Sonnet 5.5 for 599 / 596 / 639 answers.

**Why an AI researcher cares:** the distant control samples a narrow, repeated slice of text, so it tests "not bread or lava" more than "not anything else".

---

### 12. Small human moments, and sentences that beat the corpus

**Hook:** Some of the cleanest neurons are about tiny social scenes, and about 1 in 9 or 10 one-job poles was driven harder by one invented sentence than by anything in 52 million tokens.

**What the data shows.** Below are passing poles whose descriptions read like short stories. Several neurons carry one theme on both signs: grief on one side and obituary formulas on the other; "the dish we know today" and "what is now Bangladesh"; code's `else` with a conversational "No" against "If so". For about one in nine or ten ONE poles, the writer's single maximal sentence scored at or above the pole's corpus maximum.

**Evidence.**
- O7 L09N8024- (ONE, 5 of 5, `blind_max_score` 1.339): "Fires inside idioms of one party paying for another, on the payment word or the thing paid for: "I'm buying", "let me buy you a drink", "my treat", "pick up the tab", "foot the bill", "buy a round"..." Context: `Hey, hey, hey. / I'm⟦ buying⟧.` Writer: "Don't even reach for your wallet, tonight is my treat, I'm buying the first round and I'll pick up the tab for dinner too."
- G4 L21N9849+ (ONE, `blind_max_score` 0.76): "Fires inside a phrase about taking off, hanging up or setting down wearable items or belongings on arriving indoors ("hung her keys on the hook", "take their shoes off at the door", "Can I take your hat and coat", "kicked off my boots")..." Context: `“Did she give you any trouble?” Evelyn asked as she⟦ hung⟧ her keys on the hook.`
- Q4B L16N5096+ (ONE, 5 of 5): "Fires inside nostalgic, grieving remembrance of a specific lost or absent loved one, on the words that recall that person's endearing everyday habits and personality..." Context: `I know I won't feel him hug me anymore, or⟦ say⟧, 'I love you mommy,'`. Its other pole, L16N5096- (ONE, 5 of 5), fires on obituary formulas: `passed away on Wednesday, October 11, 2023,⟦ at⟧ Lancaster General Hospital`.
- Q4B L15N418- (ONE): "In a tech-support forum or Q&A thread, fires on the words and punctuation of the original asker's follow-up announcing they solved their own problem ("I found the solution.", "The mistake causing the problem was…", "Never mind, I figured it out", "I answered my own question")..."
- G4 L10N7669- (ONE, 5 of 5): "Fires on the name of a rival or alternative technology product or service mentioned right after the dominant one it competes with, most strongly on "Bing" in phrases like "Google or Bing"..., (Galileo, BeiDou or BDS after GPS and GLONASS; Mastodon after Twitter and Facebook; Mistral after GPT, Claude and Gemini; Preact after React)". Context: `Claude 3 Haiku/ Sonnet 3.5/ Opus / Gemini 1.5 pro/ Flash / Mist⟦ral⟧ Large`. Writer text scoring 0.64: "We benchmarked GPT-4, Claude, Gemini and Mistral on the same set of reasoning questions."
- O7 L15N1600+ (ONE): "Predicts an upcoming "today"/"now": fires on the words inside a relative clause describing something's present form after a history of change, as in "evolved into the iconic dish we know today"..."; O7 L15N1600- (ONE): "Fires at the moment a narrative about the past makes an anachronistic link to the present: on the word or name right after phrases like "what is now", "present", "known today"..."
- Q4B L12N2605+ (ONE): code `else` and a negative answer, contexts `print("stack Full") / ⟦ else⟧:` and `Is it the final word in musical fidelity? No⟦,⟧ but it's really not bad.` Q4B L12N2605- (UNK): `If⟦ so⟧, then aren't you going t` and `“Unfortunately,⟦ yes⟧,” Jim said.`
- G4 L16N4652- (ONE): "Fires inside exasperated rhetorical complaints that something should be easy or obvious for someone to do but they failed to do it, peaking on the word 'hard'/'difficult' in 'Is it really that hard to…' / 'How hard can it be…'..." Writer text scoring 0.91: "One star. The waiter forgot our drinks twice, and I'm sorry, but how hard can it be to write down two orders and bring them out?"
- G4 L01N6867- (ONE, 5 of 5): "The word "few" used bare, without the article "a", in non-native (often Indian) English, as in "after few minutes", "here are few things" or "I have few questions"..."
- Q4B L12N7367- (ONE, `blind_max_score` 1.837, the highest in the three samples): "Fires on the word naming a flattening or unfolding of a 3D or stacked thing into a flat 2D layout, such as UV unwrapping or UV maps of a 3D model, nets of solids, an unrolled section, flattening a tensor..." Writer: "To texture the model, I did a UV unwrap and exported the flattened UV map, which looked just like the net of a box laid out flat."
- [M] `X3_blind_max_ge_corpus_max.ONE`: 41 of 358, 34 of 338, 35 of 311.

**Why an AI researcher cares:** these are the essay's best case, abstract and human jobs that a one-line description captures well enough to write new, stronger examples from scratch.

---

## Numbers that surprised me

- **The essay's own rule lands near the essay's own number.** Under the essay's comparison (strong at 0.30, strong positives outnumber strong near misses), poles "win" 526 of 595, 517 of 596 and 518 of 599 times ([M] `essay_rule`), about 87% to 88%. Under the study's rule, 412, 384 and 365 pass ([P:*] `passes`). Among essay wins, 39, 55 and 54 had only one strong positive out of five (my count).
- **A whisper beat the corpus 2.8 times over.** Q4B L02N1208- is RARE: max |a| 0.001223, only 20 tokens above 1% of that ([P:Q4B] `explainer_saw`). The explainer still found "use strict" at the top of a JavaScript file, and two writer positives scored 2.83 and 2.81.
- **On almost every token.** Q4B L01N7440- sits above 1% of its own max on 52,442,092 of 52,442,093 scored tokens.
- **Shuffled controls fired less than their own threshold implies.** θ_s is the 99th percentile of shuffled-pole scores on the discovery set ([M] `theta`), so about 1% would be expected there. On the audit sets the shuffled strong rate is 0.54%, 0.29% and 0.26% ([M] `strong_rates.shuffled`).
- **OLMo's passes sit close to the line.** Raising θ_s by 0.1 drops O7's ONE share from 0.518 to 0.392; Q4B goes from 0.597 to 0.522 and G4 from 0.563 to 0.490 ([M] `band_theta_plus_0.1`).
- **Confidence barely separates pass from fail.** Explainer confidence averages 0.610 / 0.620 / 0.569 on ONE poles and 0.506 / 0.458 / 0.458 on UNK-F poles (my count, [E] sample 0).
- **Agreement looks high and is near chance.** 84%, 81% and 83% raw agreement on the label, with kappa 0.180, 0.167 and 0.070.
- **Gemma's last layer is almost unexplained.** 1 ONE pole of 16 in layer 33.
- **Odd text is harder in Qwen only.** Poles flagged for code, markup or non-English contexts pass 0.601 of the time against 0.735 unflagged in Q4B, but 0.615 against 0.608 in O7 ([M] `X7_pass_rate_by_encoding_flag`).
- **"I don't know" was sometimes wrong.** 8 of 32 hedged best guesses passed.
- **Every declined pole was paperwork.** Four explainer answers were refused after three tries for quoting the contexts too literally (Q4B L29N458+, G4 L05N9115-, L06N2449-, L18N804+); one writer request was blocked by the provider's content filter on a description of essay-mill spam (G4 L11N1220-; the record does not say why); one writer returned six positives instead of five, three times (O7 L27N7623-) ([M] `declined`).

## What cuts against the story

- **Language models all the way down.** Every description is a hypothesis written by Opus 5.5 or Sonnet 5.5; every test text was written by Sonnet 5.5; the second reader is DeepSeek V4.1 Flash. Only the probe scores are measurements. The writer's habits shape the controls (story 11), and its caricatures shape which descriptions pass (story 7).
- **The thresholds are the study's.** θ_s = 0.3153 was set on 1,785 shuffled scores from the Q4B discovery set. Moving it by 0.1 in either direction moves ONE from 0.638 to 0.522 (Q4B), 0.622 to 0.490 (G4) and 0.617 to 0.392 (O7) ([M] `band_theta_minus_0.1`, `band_theta_plus_0.1`).
- **The essay's looser rule gives the essay's answer.** 86.5% to 88.4% wins ([M] `essay_rule`). Which rule is right is a choice, not a finding.
- **Depth and explainer are confounded.** Opus explained layers 0 to 28 in Q4B, 0 to 25 in G4 and 0 to 22 in O7; Sonnet explained nearly all deeper layers (my count from `explainer_model`; [R] footnote gives Opus/Sonnet per tercile: Q4B 65/135, G4 54/140, O7 14/186 in the deep third). ONE by explainer: Opus 0.622 / 0.620 / 0.547, Sonnet 0.511 / 0.386 / 0.448 ([M] `by_explainer_model`). [R] reports that per-pole classes depend on the explainer for about 28% of poles (33 of 117 on discovery). Story 9's depth shift is real within Opus, but its size in the deep third is mostly Sonnet's style.
- **Scoring against each pole's own max flatters near-dead poles.** Story 1 shows ONE poles that peak at 0.003. The headline Q4B ONE share includes them, though they are 5 of 358.
- **My pattern counts are mine.** The copy, next-word, AI-style, long-document, forward-looking, bread and volcano counts come from regexes I wrote. They miss paraphrases and catch a few false hits. The study's own string check is narrower ([M] `untestable_phrase`).
- **Where a text fired is not in the sample files.** `texts` carry scores, not the token where each text peaked. When I say where a pole fires, that comes from the explainer's 30 contexts, not the probe.
- **The Sudoku reading is a heuristic.** "Repeats a symbol in the previous 12 characters" is my test on 30 contexts per pole. It was not probed.
- **Two claims are unverified.** I did not verify that the explainer model could not perceive the raw carriage return (story 5). The negative-pole sign argument in story 3 is direct-path arithmetic, not a measurement.
- **Samples and data.** 600 poles per model; single poles are anecdotes. O7's audit set leaves out 40 poles of its explain directory ([R]). The corpus is OLMo 3's own stage-1 mix; for Q4B and G4 it is not their training data ([S] §4.2). The untrained-checkpoint comparison is 40 poles with one explainer sample each.
