# Clutter: the strangest true stories in the "Already Clean" study

Angle: neurons collecting extra jobs, unrelated jobs, odd job mixes, where and when the mess appears, migrations, and the hard-decoy near miss.

**Where the numbers come from.** Paths are relative to `/Users/abaybektursun/projects/interp_research-neurons-explain` unless they say otherwise.
- "Study figure" means I read the number from `bank/w5/verdict/metrics.json`, `bank/w5/verdict/grid.md` or `bank/w5/RESULTS.md`.
- "My count" means I computed it from the sample files: `bank/w5/verdict/samples/q2_clusters.jsonl` (2,465 rows, 600 O7 poles), `bank/w5/verdict/samples/q1_{Q4B,G4,O7}_poles.jsonl`, and `bank/w4/o7_s_v2/explain/clusters.jsonl` (two clustering samples per pole). My unrelated-share numbers use the study's formula `p_u = (a_ctrl − A) / (a_ctrl − 0.5)` (SPEC_D1 §6.3). My intervals come from 2,000 pole-resampling draws with seed 0, resampling the controls in the same draw.
- "Judge right k/4" is the share of the four random-decoy judgments that picked the true pole (`acc_random` × 4). "Flagged" means at most one of the four was right.
- "Record at T_x" lists the most common marked tokens in the pole's own activation record as the judge saw it (`judgments[].judge_saw`). That record shows up to 20 contexts at the checkpoint before the job's birth, with contexts near the job's exemplars removed.
- Checkpoint steps: T_4 5k, T_5 8k, T_8 35k, T_9 60k, T_10 100k, T_11 180k, T_12 320k, T_13 560k, T_14 1M, T_15 1,413,814.

---

## 1. Ranked stories

### Story 1. The lithium neuron that became a famous-name finisher (O7, L03N6372+)

**Hook:** The most cluttered neuron in the study spent its middle age firing on "lithium". It ended training with 15 jobs. Two other readers say it has one job, or three.

**What the data shows.** L03N6372+ has 15 jobs from its 50 top contexts, the most of any of the 600 poles (my count; the runner-up has 12). It is one of the study's 17 migration poles (study figure, `q2.H_MIG.poles`). Twelve of its jobs were born late and tested. The judge got 9 of them wrong often enough to flag them. That is the highest number of flagged jobs on any pole (my count; the next highest has 4). The records show what the neuron was doing before. At T_9, T_11 and T_12, its own top contexts were "lithium" in 19 or 20 of 20 contexts, from mining and battery news. At T_13 the record changes to "/licenses" 8, "OpenSSL" 5, "Euler" 3, "License" 2. At T_14 it shows "Cassandra" 3, "lithium" 3, "Ore" 2, "ore" 2. At the end of training, lithium is not among its jobs at all.

**Full job list at T_15** (name as the clusterer wrote it; contexts; birth; judge right vs random decoy):
- c0, main: "MIT license / opensource URL tokens" (6; T_13; 0/4, flagged)
- c1: "baking parchment" (3; T_12; the two birth readings disagree, so it was not tested)
- c2: "xampp / ampps" (2; T_12; 0/4, flagged)
- c3: "misc proper-noun completions (Australian, Mozambique, Neue, Microgaming, salutations)" (5; T_10; 1/4, flagged)
- c4: "David Beckham surname" (3; T_14; 0/4, flagged)
- c5: "UC Santa Cruz" (4; T_5; earliest job)
- c6: "Industrial Hemp" (2; T_14; 4/4)
- c7: "Palo Alto" (5; T_12; 0/4, flagged)
- c8: "code/tech identifier fragments (git-svn, #![, Euler, apt-key adv, letsencrypt, eslint, jsonplaceholder)" (7; T_13; 0/4, flagged)
- c9: "Solana" (2; T_13; 1/4, flagged)
- c10: "Thomson Reuters" (2; T_15; 2/4)
- c11: "KubeCon" (2; T_12; readings disagree, not tested)
- c12: "HuggingFace" (3; T_13; 0/4, flagged). Exemplar: "apps like ChatGPT, Hugging⟦Face⟧'s HuggingChat".
- c13: "Canyon (Laurel/Diablo)" (2; T_14; 4/4)
- c14: "pexels URL" (2; T_12; 0/4, flagged)

**The other two readers disagree.** The Q1 explainer (Opus 5.5) labelled the same pole ONE, and the explanation passed the §6.1 test with 5 of 5 strong positives: "Fires on the word or word-piece that completes a well-known multi-part name or fixed compound term, where the earlier part makes it nearly certain: the second word of a famous name or place (David Beckham, Palo Alto, Santa Cruz, South Australian), a stock collocation (baking parchment, Industrial Hemp), or the tail of a familiar tech or license string" (`q1_O7_poles.jsonl`). The second independent clustering (`clusters.jsonl`, sample 1) found 3 jobs: "mid-identifier tokens in license URLs, tool and package names, code" (19 contexts), "parchment after baking" (3), and "second token of multi-token proper names and terms in prose (Santa Cruz, Palo Alto, Beckham, Reuters, Hemp)" (28).

**Why an AI researcher cares:** One neuron holds the whole debate. It has 15 topics, 1 mechanism and a full change of job mid-training, so whether it counts as superposition depends on whether "job" means topic or computation.

---

### Story 2. The mess arrives in the last stretch of training

**Hook:** Jobs born before step 60k almost never look unrelated. Jobs born after step 560k look unrelated about 40% of the time.

**What the data shows.** I split the 1,238 late-born jobs by birth checkpoint and applied the study's estimator to each window (my count, `q2_clusters.jsonl`, with the global a_ctrl 0.924):

| born at | steps | late jobs | judge accuracy A | p_u (mine) | 95% interval | flagged |
|---|---|---|---|---|---|---|
| T_4 to T_9 | 5k to 60k | 196 | 0.930 | 0 | [0, 0.059] | 6 |
| T_10 to T_12 | 100k to 320k | 525 | 0.821 | 0.242 | [0.169, 0.317] | 48 |
| T_13 to T_15 | 560k to 1.41M | 517 | 0.757 | 0.395 | [0.321, 0.472] | 71 |

None of the 120 jobs born at or before T_8 (35k steps) was flagged. Single checkpoints show the same climb: p_u is 0.371 for jobs born at T_13 and 0.444 for jobs born at T_14. Size does not explain it. Among large jobs (11 or more contexts), p_u is 0.024 for jobs born by T_9, 0.110 for T_10 to T_12, and 0.256 for T_13 to T_15 (my count).

Most jobs are born late. The study's birth counts (`q2.births`) put 1,246 of 1,854 side jobs and 342 of 600 main jobs at T_11 to T_14. Only 1 of 2,454 jobs is born at the first trained checkpoint, step 1,000 (`q2.share_born_T_1` = 0.0004). New jobs are born weak, with a median exemplar at 0.29 of the pole's maximum (`q2.r_at_birth`). Once they arrive, they stay: 1,762 of 1,854 side jobs (95.0%) gained relative strength from T_12 to T_15, including all 108 flagged side jobs (my count, `growth_last_half`). The study's vestigial share is 0.035 [0.026, 0.044] (`q2.vestigial`).

**Why an AI researcher cares:** If unrelated jobs pile up in the final stretch of pretraining, a model looks cleanest early in training. A claim that a final model is clean has to account for jobs added late.

---

### Story 3. The neuron that said "Exactly" and then learned React (O7, L01N10284-)

**Hook:** A layer-1 neuron started out on sincere adverbs like "Truly" and "Exactly". By the end of training its strongest job was the `useEffect` hook in React import lines.

**What the data shows.** This is a migration pole (`q2.H_MIG`). Its earliest job, "sentence-initial adverb (Truly, Exactly, Honestly, Again)", was born at T_4 (step 5k) and holds 24 of its 50 contexts. Its main job at the end, "useEffect in React import / hook", was born at T_12 (step 320k) and holds 12. The judge was shown ten exemplars such as `import React, { useState, ⟦useEffect⟧ } from "react"`. Next to them were two records: the pole's own record from T_11, which read "Truly" 9, "uly" 7, "Seriously" 2, "Exactly" 2, and a decoy record full of "loan shark" and "transaction". The judge picked the loan sharks. It got 1 of 4 right.

The Q1 explainer called it two unrelated jobs: "the token "useEffect" inside a JavaScript React import line ... and also on a one-word emphatic agreement or sincerity adverb opening a reply sentence, chiefly "Exactly"". The near-miss texts show how sharp the React job is. `import React, { useState, useEffect } from "react";` scored 0.784, while the same line without useEffect, `import React, { useState, useCallback } from "react";`, scored 0.201. On the prose side, the same reply with "Yes." in place of "Exactly." scored 0.023. Its class still came out UNK, because the "Exactly" job alone got only 2 of 5 strong texts ("UNREL support clause failed"). "Honestly, I didn't expect the movie to be that good" scored 0.089.

React keeps turning up as the odd partner job. The other sign of the same neuron, L01N10284+, has a flagged job "JavaScript import/JSX identifiers (ReactDOM, setName, Footer)" (judge right 0/4). On Qwen3-4B, the confirmed two-job pole L07N4960+ pairs "the surname Harris or Haris" with "the 'useState' hook inside a React import line" (`q1_Q4B_poles.jsonl`, UNREL, both jobs supported). React, JSX or a React hook appears in 8 job names on 8 different O7 poles (my count).

**Why an AI researcher cares:** It is a clean, testable case of one neuron that changed jobs. The before and after records are both quoted, and the near-miss texts fix the boundary at a single token.

---

### Story 4. From "?LinkId=" to "Haha" (O7, L06N7759+)

**Hook:** At step 35k this neuron fired on the question marks in URLs. By step 60k it was the laughing neuron.

**What the data shows.** Full job list at T_15:
- c0, main: "laughter interjections (haha, lol, hehe, HA) in informal comments and blogs" (38 of 50 contexts; born T_9; judge right 0/4, flagged)
- c1: "informal 'Nah' dismissal interjection" (4; born T_4; earliest)
- c2: "Magic card definitions and card-maker code (+X/+X, power, flashback, attack tokens)" (8; born T_14; 2/4)

Its own record at T_8 (step 35k) reads "/?" 11, "?" 6, "|" 2. The contexts are lines like `go.microsoft.com/fwlink⟦/?⟧LinkId=317598` and `wordpress.org⟦/?⟧v=4.6.6`. One exemplar of the job that replaced them: "Oh, ping-pong is an Olympic sport. H⟦aha⟧. Haha." The judge picked the random decoy all four times. It is a migration pole. The Q1 explainer read it as one job: "a casual laughter or dismissal interjection, mostly "Haha"/"haha" ... and also "lol", "heh" or "Nah"" (ONE, 4 of 5 strong).

**Why an AI researcher cares:** In one checkpoint interval, the neuron's strongest job went from URL syntax to a feature you would put on a slide. Late in training it also picked up Magic: The Gathering card text.

---

### Story 5. Odd couples that pass every test

**Hook:** These neurons do C# random-number generators and poetry rhyme schemes. Or door latches and an Entity Framework exception. Or obituaries and C# braces.

**What the data shows.** The UNREL class needs two jobs the explainer calls unrelated, and each job has to drive the neuron on its own (at least 3 of 5 strong texts written from that job's sentence alone, SPEC §2.3). Only 8 Q4B, 7 G4 and 4 O7 poles qualify (study figures, `q1.<cell>.fractions`, at most 0.013). They are rare, and they are strange. Job names below are quoted from the `jobs` fields:
- O7 L06N4452-: C# "the type name Random immediately after the keyword new" and "The word rhyme or the "ym" piece inside rhymed, rhyming or rhymes". The clusterer's full list for this pole: "C# 'new Random' (Random class instantiation)" (19 contexts) | main "rhyme / rh-ym- tokens" (4) | "butter in baking ingredient lists" (10) | "trump suit in card game rules" (3; judge right 4/4) | "C# using namespace imports (System.Runtime.InteropServices, SQLite)" (4; 1/4, flagged) | "Homestar Runner name fragment" (3; 4/4) | "mid-word fragments in misc. text (Olympians, richest, quirer, conversion, peak, BDS, Ukrainian)" (7; 0/4, flagged).
- O7 L16N10459+: "lock, latch, hinge, zipper, clip, clutch, gun slide" and "The 'Update' piece inside DbUpdateConcurrencyException". Clusterer's full list: main "mechanical part or action word (fold, latch, hinge, zipper, clasp)" (15) | "web boilerplate and page-title word fragments" (9; 4/4) | "C# DbUpdateConcurrencyException token" (7; 0/4, flagged; exemplar `catch (Db⟦Update⟧ConcurrencyException)`) | "sentence-ending period in spam essay text" (19; 2/4).
- O7 L08N10397+: shape words ("oval, triangular, semi-circular, an 'H' shape, squiggly line") and "The activity noun in light-exercise advice such as 'go for a walk'".
- O7 L09N10681+: "git head", "cvs head", "rawhide" in bug reports and "The word "back" in the web-copy invitation "check back"". It is also a migration pole.
- G4 L22N9321-: memorial tributes ("will live on", "will be missed") together with "In C# source files, the line break right after the opening brace of a namespace declaration" and "The dot in URLs such as about.com or inc.com".
- G4 L05N7786-: "'ine' ... completing 'Sistine'", "the '7' in 'COP27'" and "the chapter number '7' or '17' right after 'Matthew'".
- G4 L04N4329-: "blouse" and "the inch mark in 12"".
- G4 L10N795+: "'A Study in Y'", "'annualized'", "'parallel', 'Parallelism'" and "'static' or 'ESD'". That is four unrelated jobs.
- G4 L00N1794-: "concur", "Accent", Arabic transliteration letters such as "Ḳ in "Ḳurʾān"", and word pieces after a capital Y.
- Q4B L00N4968-: C# LINQ ".ElementAt" and " Kardashian".
- Q4B L25N5080-: "pi = 3.14" in beginner code and inner words of Indian government scheme names ("Pradhan Mantri Gram Sadak Yojana").
- Q4B L08N6028-: the React `key` prop and homework instructions ("check all that apply").

By my reading of the job names, 14 of the 19 UNREL poles pair a code or software job with a prose job: all 8 on Q4B, 3 of 7 on G4 (L10N795+, L22N9321-, L25N644+), and 3 of 4 on O7 (L06N4452-, L09N10681+, L16N10459+). Q2 agrees on the O7 four. Their 13 late-born jobs give A = 0.577 and p_u = 0.795. For poles the explainer called ONE, the figures are A = 0.825 and p_u = 0.242 (my count, using each class's own control accuracy).

**Why an AI researcher cares:** The genuinely unrelated pairs are rare, but they are verified one job at a time, and code is the usual partner. That fits the idea that code and prose rarely share contexts, so a neuron can hold one of each at little cost.

---

### Story 6. How many jobs does a neuron have? Ask twice, get two answers

**Hook:** On the same 600 O7 poles, the explainer gives one job to 532. The clusterer gives one job to 26.

**What the data shows.** The Q1 explainer wrote `n_jobs` = 1 for 532 of 600 O7 poles, and the DeepSeek second reader labelled 561 ONE (my count, `q1_O7_poles.jsonl`). The Q2 clusterer, looking at each pole's top 50 contexts, made a single job for 26 poles. Its mean is 4.09 jobs, and the maximum is 15 (my count, `q2_clusters.jsonl`). The clusterer also marked 3,051 of the 4,829 job pairs it named (63.2%) as unrelated (my count, `clusters.jsonl` sample 0). The job count does not track the explainer's verdict. Poles the explainer called ONE average 4.09 clusters, the same as UNK poles at 4.00, and only 11 of 311 ONE poles got a single cluster.

The clusterer does not agree with itself either. Two independent samples give the same count on 182 of 600 poles (30.3%, my count) and counts within one of each other on 67.8% (study figure, `q2.B4_m_within_1`). Mean counts are 4.09 and 3.375. The largest splits (sample 0 vs sample 1):
- L06N572+, 6 vs 1. Sample 0 has "Juneteenth: 'enth' completing the holiday name", "'Leap second' thread", "Batik: 'ik' completing batik in interior-design text", "GitOps: 'Ops' completing GitOps" and two more. Sample 1 has one job, "later token of a multi-token rare term already seen in the document (batik, Juneteenth, Leap second, GitOps)". The single job describes a repeated-term (induction-like) completion, and the six are its topics.
- L12N8337+, 9 vs 2. Sample 0 splits one punctuation job into "sentence-ending periods in online lottery/gambling SEO text", "periods in spun biography/timeline essay text", "periods in date night/marriage advice spun text", "periods in batik interior design articles", "periods in power cord buying tips", "antivirus/Avast software text", "period before paragraph break in toilet flushing FAQ", "periods in health screening (Life Line/HealthFair) text" and "closing parenthesis in market-report titles". Sample 1: "sentence-final period in spammy SEO text" (47 contexts) plus the parenthesis job.
- L03N6372+, 15 vs 3 (Story 1). L04N5845+, 8 vs 1 ("mid-word fragment completion", 44 contexts).
- L02N4649-, 11 jobs in sample 0 and 5 in sample 1. Q1 explainer: ONE, "the piece after "Rou" in "Roulette" ... and in the name "Rouhani", and also on pieces of foreign or political proper names such as Mao, Likud and Manziel". Sample 0's list: main "Rou|hani (Rouhani name suffix)" | "Rou|lette (roulette, casino text)" | "Mao (Mao Zedong / Maoist)" (flagged) | "Grund|y (surname fragment)" (flagged) | "Lik (Likud, Likavec)" | "misc word-piece fragments in names/words" | "treasury (state/US Treasury)" (flagged) | "Abe (Japanese PM)" | "forfeiture" | earliest "Mou|lin (Moulin Rouge, Moulin-à-Vent)" | "mant (mantenimiento, Spanish/SQL)" (flagged).

**Why an AI researcher cares:** Counts of "jobs per neuron" are as much a property of the reader as of the neuron. The explainer lumps by mechanism and the clusterer splits by topic. Neither one is checked against the model's computation.

---

### Story 7. Seventeen neurons that changed careers

**Hook:** For 17 of 600 neurons (2.8%), the main job at the end of training was born late, and the judge could not link it to what the neuron had been doing before.

**What the data shows.** H_MIG counts poles whose main job is late-born and flagged (study figure, `q2.H_MIG`, 17 poles, share 0.028). The table shows each one's own record just before the main job was born. Eleven of the 17 main jobs got 0 of 4 (my count).

| pole | earliest job (birth) | own record just before the main was born | main job at the end (birth) | judge right |
|---|---|---|---|---|
| L03N6372+ | "UC Santa Cruz" (T_5) | T_12: "lithium" 19 of 20 | "MIT license / opensource URL tokens" (T_13) | 0/4 |
| L06N7759+ | "informal 'Nah' dismissal interjection" (T_4) | T_8: "/?" 11, "?" 6 | "laughter interjections (haha, lol, hehe, HA)" (T_9) | 0/4 |
| L01N10284- | "sentence-initial adverb (Truly, Exactly, Honestly, Again)" (T_4) | T_11: "Truly" 9, "uly" 7 | "useEffect in React import / hook" (T_12) | 1/4 |
| L14N5001- | "respectively" (T_11) | T_11: "respectively" 17 | "spun-text essay spam, verbs and pronouns" (T_12) | 0/4 |
| L01N9435+ | "specified/designated after a" (T_4) | T_11: "prescribed" 9, "specified" 4 | "Nou in Art Nouveau" (T_12) | 1/4 |
| L01N7380- | "storm names and disaster place names", "space and Mars names", "-shima in Hiroshima" (T_11) | T_11: "comet" 10, "Comet" 5, "Apollo" 3 | "first fragment of split words and miscellaneous names (dri-est, fou-led, mul-ches, basil-isk, Rudd-iman, etc.)" (T_12) | 0/4 |
| L09N10681+ | "incub- word start (incubation)" (T_5) | T_10: "sit" 4, "utilizing" 2, "minutes" 2 | "git/cvs head and rawhide (development branch) in bug reports" (T_11) | 0/4 |
| L23N809- | "gulp / gulping verb" (T_5) | T_8: "." 5, "excellence" 3 | "Gul- at start of proper names (Gulbenkian, Gulian, Gulkišar, Gulati, Guldan)" (T_9) | 1/4 |
| L21N7401- | "OEIS entry text", "Diophantine polynomial integer roots question" (T_11) | T_11: "." 15 | "Mathematica Ball documentation page" (T_12, 3 contexts) | 0/4 |
| L05N2634+ | "verb after subject noun in machine-spun spam text" (T_8, 47 contexts) | T_8: blank lines and broken bytes | "Begins after Article in Iraqi dinar blog" (T_9, 3 contexts) | 1/4 |
| L14N4978- | "Glory Casino brand name completion" (T_13) | T_13: "uggy" 2, then singletons | "numeric/ID tokens repeating an earlier reference" (T_14) | 0/4 |
| L18N9686+ | "first piece of a split name or word (Shepherd, Maver-ick, vigil-ance)" (T_12) | T_13: "Listen" 4, "Gard" 3, "vigil" 2 | "participle or word before a following complement (should be cleaned, dealt, fired, monitored)" (T_14) | 0/4 |
| L01N8975- | "Non-English/multilingual fragments", "Truncated 'Cert' word" (T_12) | T_13: "Boundary" 5, "pel" 5 | "URL/domain fragments in links (.games, /modules, .access, www.paint, etc.)" (T_14) | 1/4 |
| L12N10223+ | "recipe ingredient and food words" and two others (T_10) | T_10: "1" 4, "Votes" 2 | "mid-word fragments and proper-name continuations" (T_11) | 0/4 |
| L14N4543+ | "function words and punctuation ... after a noun phrase" (T_14) | T_14: "for" 3, "about" 3 | "mid-word/identifier subtoken completion (camelCase, names, compounds)" (T_15) | 0/4 |
| L17N5407+ | "head noun completing a compound noun in a list" (T_9, 43 contexts) | T_10: singletons ("Theory", "goat", "Trail") | "non-English and odd tokens" (T_11) | 1/4 |
| L18N6315- | "footnote numbers in citation lists" and one other (T_13) | T_13: singletons | "clause-introducing words (way that, ones you see, since there)" (T_14) | 0/4 |

**Why an AI researcher cares:** The essay's InceptionV1 comparison allows drift, and these are drift cases you can open and read. A neuron that went from "lithium" to MIT licenses, or from "respectively" to essay spam, is a warning for anyone who labels a neuron from one checkpoint.

---

### Story 8. Early layers hold the most jobs

**Hook:** Neurons in the first quarter of the network have a third more jobs, and nearly three times the flag rate, of those in the last quarter.

**What the data shows** (my count, `q2_clusters.jsonl`, grouped by the layer in the pole id):

| layers | poles | mean jobs | late jobs per pole | flagged share of late jobs | p_u (mine) | 95% interval |
|---|---|---|---|---|---|---|
| 0 to 7 | 152 | 4.92 | 2.81 | 0.143 | 0.307 | [0.167, 0.435] |
| 8 to 15 | 152 | 3.82 | 1.67 | 0.102 | 0.255 | [0.148, 0.364] |
| 16 to 23 | 146 | 3.94 | 1.92 | 0.085 | 0.279 | [0.194, 0.363] |
| 24 to 31 | 150 | 3.67 | 1.84 | 0.051 | 0.172 | [0.097, 0.248] |

Layer 2 has the highest mean, 5.61 jobs per pole. Of the 420 jobs whose names say "fragment", "mid-word", "word-piece" or "subword", 213 are in layers 0 to 7 (my count). Q1 says the opposite. On O7, the shallow tercile (layers 0 to 10) has the most ONE poles (0.620) and the fewest UNK (0.288). The deep tercile (layers 22 to 31) has ONE 0.463 and UNK 0.479 (study figures, `q1.O7.by_depth_tercile`). Q4B (ONE 0.735 in tercile 0) and G4 (0.689) follow the same pattern. A likely reading: early neurons are token detectors. Their top contexts cover many topics but share one token rule, so the clusterer splits them and the explainer does not. The intervals above overlap, so the depth trend in p_u is suggestive, not established.

**Why an AI researcher cares:** "Where is the mess?" has opposite answers depending on whether a job means a token rule or a topic. That matters most in the early layers, where most dictionary-learning work starts.

---

### Story 9. The G neuron: when "unrelated" is a false alarm

**Hook:** A last-layer neuron predicts that the next word starts with G. The judge called one of its G jobs unrelated to its other G jobs.

**What the data shows.** O7 L31N10276- has three jobs, and all three are G-prediction: main "Animalia alliteration: next word starts with G" (exemplar: "Great green gor⟦illas⟧ growing grapes in a gorgeous glass greenhouse"); "Acronym/abbreviation expansions where the next word starts with G (Group, Guide, Grade, Goal, etc.)"; and "Other contexts predicting a G-initial next token (names, words, lists)". The third one was flagged (judge right 1/4). Its exemplars include "J. G. (⟦John⟧ Gadsby)", "⟦Thanks⟧ for the interview George!" and "I find myself⟦ second⟧ guessing". The pole's own record at T_8 was full of G: index lines "Galligan, Richard ... Ganey, Rodney ... Ganong", "g as in garden", "gnæt (gnat)". The judge still picked the decoy in three of four tries. In the try quoted here, the decoy record's most common token was "still". The pole has a sibling. O7 L31N3362- predicts a next token starting with capital O (its Q1 explanation: "Fires at the last position before the model expects a next token starting with a capital O"). Its "General web text where the next token starts with O (OEM, OLED, Oxley, CSGO, SEO, Oxford ...)" job was judged 0/4.

Two more false alarms come from what the judge was shown. L12N1003-'s jobs "does/did/do after It/He/They" and "is after It" are both 0/4, next to a main job "modal after It/They (can, will, would, could, should)". The record the judge saw for this pole at T_9 was "chemical" 12, "chemicals" 3. The record shows only the top 20 contexts, so the modal job, already present by the birth rule, was not in view. L05N924+ is the "together with" neuron. The Q1 explanation includes "Latin "cum"", and it passed 5 of 5. The clusterer split "Latin 'cum' (with)" into its own job, which the judge scored 0/4. The pole's T_11 record showed "Rabbit" 7 (from "Jojo Rabbit" in Oscar lists) and "supported" 3 (from "Your web browser is no longer supported"), with nothing about "with".

**Why an AI researcher cares:** The parent test asks whether the pole's strongest earlier contexts look like the new job. A sibling of a weaker job, or a job defined by the next token, can fail that test while being the same computation. Some of the 0.265 is this kind of false alarm.

---

### Story 10. Two judgments short: the hard-decoy near miss

**Hook:** The study's hardest control needed 960 correct answers out of 1,200. It got 958.

**What the data shows.** On controls (the main job against the pole's own T_14 record), the hard decoy is the same-layer pole whose top 50 shares the most token strings. The judge was right on 958 of 1,200 judgments, 0.798 (my count; study figure `q2.hard.ctrl_accuracy` 0.7983), just under the 0.80 bar. So the hard-decoy row is "not measurable" (`q2.hard.measurable` false). At the gate it was 0.692 (`RESULTS.md`). On late jobs the hard-decoy accuracy was 0.697. On flagged jobs it was 0.356, or 89 of 250 judgments (my count), well below a coin flip. For 109 late jobs the judge was 4/4 against a random decoy and 0/2 against the hard one. For 82 of 600 main jobs, it picked the hard decoy both times (my count).

Two near misses read like neighbour stories. O7 L06N10970+'s main job is "computing, programming, coding terms". At T_14 its own record showed "evac" 3, "Designer" 2, "entrepreneurial". Its same-layer neighbour L06N1501+ showed "coding" 10, "programming" 3, "code" 3. The judge picked the neighbour twice. O7 L03N10451+ is a "switch off" neuron (main "off after switch/turn", 28 contexts, born at T_2, one of the earliest main jobs in the study). At T_14 its own record was mostly "Dora⟦emon⟧" and "⟦Terminator⟧", with only one "turned ⟦OFF⟧". The neighbour L03N10379+ fired on " on" in "Follow us on Twitter". The judge picked the neighbour twice. The pole's other jobs are "Terminator", "shutoff/termination word pieces and Lair" and "emon in Doraemon". The judge called all three related, 4/4 each.

**Why an AI researcher cares:** The secondary test that would tell "new job" apart from "a job its neighbour already had" fell short by two answers. Nearby neurons in a layer can look more like a job than the neuron that holds it.

---

### Story 11. The "main job" is often a side gig

**Hook:** For one neuron, 47 of its 50 top contexts are spam verbs. Its "main job" is three contexts from an Iraqi dinar blog.

**What the data shows.** The study names as "main" the job with the largest activation at T_15 (`peak_end`, `infra/pg/neurons_verdict.py` lines 386 and 395). In 279 of the 574 poles with two or more jobs, the main job is not the biggest by context count (my count). In 56 of them it beats the runner-up by less than 2%. Seven poles have a main of 3 or fewer contexts beside a job of 30 or more. Examples: O7 L05N2634+, main "Begins after Article in Iraqi dinar blog" (3) beside "verb after subject noun in machine-spun spam text" (47). L13N6183+, main "Der kleine Vampir spam: 'could much apologize/analysed'" (2) beside "verb after 'could not / ca not' in spam error messages" (47). L06N2185+, main ""Research" in journal citation / bibliography abbreviations" (2) beside ""connection/connect/connected" in emotional-bond sense" (32). L05N2634+ is also a migration pole.

**Why an AI researcher cares:** Birth order and migration both depend on which job counts as main. A top-1 activation rule sometimes crowns a three-context outlier.

---

## 2. Numbers that surprised me

- **532 vs 26.** O7 poles given exactly one job: 532 of 600 by the explainer, 26 of 600 by the clusterer (my counts).
- **30.3%.** How often two runs of the same clusterer on the same 50 contexts agree on the exact job count (182 of 600, my count). Within one: 67.8% (study figure).
- **0 of 120.** Late jobs born by step 35k that were flagged (my count).
- **0.395 vs 0.** p_u for jobs born after step 560k vs before step 60k (my count).
- **95.0%.** Side jobs that gained relative strength over the last half of training (1,762 of 1,854, my count). See the caveat below.
- **958 of 1,200.** Hard-decoy controls answered right; 960 was the bar (my count).
- **70.** Late jobs where the judge picked the random decoy all four times (my count, 70 of 1,238).
- **0.007 vs 0.265.** O7's UNREL share in Q1 (4 of 600, study figure) vs the unrelated share of late-born jobs in Q2 (study figure). These answer different questions, but the gap is the whole debate in two numbers.
- **0.242.** p_u among late jobs on poles the explainer called ONE, against 0.277 on UNK poles (my count). "Clean" poles collect unrelated jobs almost as often as unclear ones.
- **155 of 600.** O7 poles that have a job literally named "miscellaneous" or "misc", 18 of them as the main job (my count). 420 job names say "fragment", "mid-word", "word-piece" or "subword" (my count).
- **Spam everywhere.** 35 of 597 G4 explanations mention spam, spun text or word salad (my regex count; 24 of the 35 are UNK). The G4 writer was blocked three times by the provider's content filter for L11N1220-. That pole is a neuron for "sentence-ending periods inside incoherent, machine-generated spam prose (spun essay-writing / thesis-help SEO filler ...)" (`q1.G4.declined`, `q1_G4_poles.jsonl`).
- **Two neurons for one Swedish spam word.** On Q4B, two layer-1 poles, L01N5391+ and L01N8338-, are explained as firing mainly on the token " thaimassage" in "gay sex games thaimassage malmö tantra" spam. Both are RARE, so neither is classified.
- **Letter neurons in the last layer.** O7 L31N10276- (next word starts with G) and L31N3362- (next token starts with capital O) both fail §6.1 as UNK, and both have a late job flagged unrelated.
- **15.** The most jobs on one pole (L03N6372+). A second clustering of the same pole found 3.
- **Every new job rejected.** 19 of the 476 poles with a late-born job had all of their late jobs flagged. Three had two or more (my count). O7 L01N6100+ is a "Chief" neuron: main job ""Chief" as executive title (C-suite role) after a person's name" (38 contexts, born T_3). Its three later jobs, "URL path segments after domain (/sites, /details)", "code/library identifiers (React.Children, StyleSheet)" and "word fragments: fal(con) wings, bers(ama), Surge", all scored 0/4.
- **Why poles end up UNK.** On O7, 220 of 249 UNK poles are UNK because the explanation failed the §6.1 test. Only 13 are UNK because the explainer itself said UNK (`q1.O7.unk_reasons`). Single-token suppressors: 0 in all three models (`q1.<cell>.single_token_suppressors`).

---

## 3. What cuts against the story

- **The judge is a language model.** Claude Sonnet 5.5 was right on 0.924 of controls on S (`q2.p_u.a_ctrl`) and 0.910 on 39 gate poles (`RESULTS.md`). Story 9 shows clear misses on same-mechanism jobs (the G and O letter neurons) and on jobs whose sibling was outside the top-20 record (L12N1003-, L05N924+). A two-way choice measures "closer of the two records", not "related". A job unrelated to both records still scores 0.5 on average, and the estimator assumes exactly that.
- **Jobs are a model's partition.** Two independent clusterings agree on the job count within one only 67.8% of the time (`q2.B4_m_within_1`) and exactly 30.3% (my count). 116 of 2,454 jobs (0.047, `q2.B4_unmatched_share`; my count matches) have no job in the second sample holding most of their exemplars. The two birth readings agree on the early or late split for 81.0% (`q2.B3_late_split_agreement`). The clusterer splits by topic (L12N8337+'s nine "periods in X" jobs), which inflates both job counts and late births. Small jobs look less related: p_u is 0.456 for 1 to 2 context jobs against 0.140 for jobs of 11 or more (my count). The judge has less to go on for small jobs.
- **Growth is partly selection.** Exemplars are the T_15 top 50, so they are chosen to be strong at the end. The 95% growth figure is expected from that alone.
- **One training run.** All of Q2 is OLMo 3 7B, one seed, 16 checkpoints. T_7 to T_14 ran on a different GPU. Dropping jobs born at T_7 or T_15 gives p_u 0.275 [0.220, 0.332] (`q2.sensitivity_F2_excl_T7_T15`). The label holds, but it is one run.
- **The essay's version of "clean" holds in part.** The main job is the oldest more often than chance: b = 0.476 [0.436, 0.517] against a null 97.5th percentile of 0.449 (`q2.birth_order`). Vestigial jobs are rare (v = 0.035). Only 17 poles migrated. Confirmed unrelated pairs in Q1 are at most 0.013 per model.
- **The essay's own looser rule gives much higher shares.** Under the essay's rule (strong = 0.30; an explanation wins if strong positives outnumber strong near misses), `q1.<cell>.essay_rule` is Q4B win 526, tie 67, lose 2; G4 win 517, tie 78, lose 1; O7 win 518, tie 80, lose 1. That is 0.884, 0.867 and 0.865 wins among rated poles (my division), close to the essay's 90%. Under the study's rule, ONE is 0.597, 0.563 and 0.518. Before any test, the explainers themselves wrote one job for 532 (Q4B), 527 (G4) and 532 (O7) of 600 poles. ONE is also sensitive to the threshold: on O7 it is 0.617 at θ_s − 0.1 and 0.392 at θ_s + 0.1 (`q1.O7.band_theta_minus_0.1`, `band_theta_plus_0.1`).
- **The explainer changes with depth.** Opus wrote most explanations in terciles 0 and 1, and Sonnet wrote most in tercile 2 (O7: 208/12, 204/16, 14/186, `RESULTS.md`). Q1 label agreement with the second reader is weak: kappa 0.07 (O7), 0.17 (G4), 0.18 (Q4B).
- **Per-window and per-layer p_u values are mine.** They use the study's estimator but are not pre-registered readouts. Their intervals come from 2,000 draws, not the study's 10,000, and the layer intervals overlap.
