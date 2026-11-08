# Training dynamics: the strongest true stories in the "Already Clean" study

Angle: when jobs form, in what order, what sticks, what dies, what migrates. Model: OLMo 3 7B ("O7"), 600 poles (set S), 16 stage-1 checkpoints T_0 (step 0) to T_15 (step 1,413,814).

## Sources and how to read the numbers

Paths are relative to `/Users/abaybektursun/projects/interp_research-neurons-explain/` unless written in full.

- **M** = `bank/w5/verdict/metrics.json` (the study's own numbers).
- **R** = `bank/w5/RESULTS.md`; **F4** = `bank/w5/figures/F4_births.svg`.
- **C** = `bank/w5/verdict/samples/q2_clusters.jsonl` (2,465 rows; 2,454 jobs after the 11 echo rows are dropped).
- **P** = `bank/w5/verdict/samples/q1_O7_poles.jsonl` (class, explanation, the 30 contexts the explainer saw).
- **H** = per-checkpoint pole statistics, `/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w3/grid/{1,1b}/out/tNN_stepN/stats/layer_LL.npz` (fields `max_abs`, `q999`; these are the grid inputs listed in M `q2.inputs.grid`).
- **Code** = `infra/pg/neurons_verdict.py` lines 324-430 (presence, birth, main, r, growth) and 575-646 (birth order, vestigial).
- **SPEC** = `/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/SPEC_D1.md` §2.4, §4.3, §6.

"(mine)" marks a number I computed from those files with read-only Python; it is exploratory and not a pre-registered readout. Everything else is copied from M or R.

Three definitions matter for every story below (Code, SPEC §2.4):

- A job is a cluster of the pole's top-50 contexts **at T_15**. So every job is present at T_15 by construction (2,454 of 2,454, C `present_pct[15]`).
- "Born" = the first checkpoint where the median exemplar reaches the pole's own 99.9th percentile **and is still there at the next checkpoint** (Code `birth()`). It is not "stays there forever": 44 jobs (1.8%) drop out again after birth (mine, C).
- The "record" the parent-test judge sees is the pole's top-20 contexts at the checkpoint before the job's birth (C `judgments[].judge_saw`). These records are the only window the study has onto what a pole did earlier in training. Several stories below come from reading them.

---

## Part 1. Ranked stories

### 1. Early jobs are family; late jobs are strangers

**Hook:** a job born in the first 4% of training almost always grew out of what the neuron already did; a job born after 40% of training is unrelated about 4 times in 10.

**What the data shows.** The study's primary readout pools all late-born jobs into one unrelated share, p_u = 0.265 [0.211, 0.321] (M `q2.p_u`). Split by when the job was born, that one number turns out to be a ramp. I applied the study's own estimator, p_u = (a_ctrl - A)/(a_ctrl - 0.5) with a_ctrl = 0.924 (SPEC §6.3, M), to the 1,238 late-born primary jobs grouped by birth checkpoint (mine, C `acc_random`, `birth_pct`; intervals are a pole bootstrap inside each bin, 4,000 draws, a_ctrl held fixed):

| born at | jobs | judge accuracy A | implied unrelated share |
|---|---|---|---|
| T_2 to T_9 (by step 60k, 4.2% of training) | 196 | 0.930 [0.902, 0.955] | -0.013 [-0.073, 0.051], so about 0 |
| T_10 to T_11 (100k to 180k) | 278 | 0.833 [0.797, 0.867] | 0.216 [0.136, 0.301] |
| T_12 (320k) | 247 | 0.809 [0.765, 0.851] | 0.272 [0.173, 0.376] |
| T_13 (560k) | 271 | 0.767 [0.718, 0.811] | 0.371 [0.268, 0.487] |
| T_14 to T_15 (1M to 1.41M) | 246 | 0.746 [0.701, 0.787] | 0.420 [0.322, 0.525] |

The flag rate (at most 1 of 4 judgments correct) tells the same story: 0 of 53 jobs born by step 20k are flagged, then 4.3%, 9.1%, 11.3%, 13.3% and 14.6% for jobs born at T_8-T_10, T_11, T_12, T_13 and T_14, with 11.8% for the 34 born at T_15 (mine, C `flagged`).

It is not a depth artifact. Inside each layer tercile, with that tercile's own control accuracy, the rise repeats (mine, C, P `tercile`; bins: born by step 100k / 180k-320k / 560k or later): layers 0-10 (a_ctrl 0.882) -0.03, 0.36, 0.34; layers 11-21 (0.934) 0.00, 0.27, 0.44; layers 22-31 (0.960) 0.11, 0.22, 0.35. Two terciles rise at every step; the shallow one rises and then holds.

**Evidence:** C fields `status == "late_primary"`, `acc_random`, `birth_pct`, `flagged`; a_ctrl from M `q2.p_u.a_ctrl` = 0.9241666.

**Why an AI researcher cares:** the essay's "every job grows from an earlier one" may be true of the first few percent of training and false of the rest. One pooled p_u hides a phase change from growing families to packing in strangers, which is exactly what H_SP predicted ("the share grows with training as capacity tightens", SPEC §1).

### 2. The lithium neuron that became a famous-name neuron

**Hook:** for a quarter of training, L03N6372+ fired hardest on one word, "lithium". At the end, lithium is not among its 15 jobs.

**What the data shows.** The judge records give this pole's top-20 contexts at several checkpoints (mine, C `judge_saw`, true record, marked tokens counted):

- step 60k (T_9): " lithium" in 20 of 20 contexts.
- step 180k (T_11): " lithium" 19 of 20. Step 320k (T_12): 19 of 20.
- step 560k (T_13): "/licenses" 8, " OpenSSL" 5, " Euler" 3, " License" 2.
- step 1M (T_14): " ore" 4, " Cassandra" 3, " lithium" 3, "eslint" 1.

Example lithium-era contexts (T_9 record): "Altura Mining is developing the Pilgangoora⟦ lithium⟧ project", "Battery maker 24M just received $22 million in funding for its SemiSolid⟦ lithium⟧-ion battery".

At T_15 the clusterer finds 15 jobs, the most of any of the 600 poles (C). They are completions of famous names and fixed phrases: "UC Santa Cruz" (born T_5), misc proper-noun completions (T_10), "baking parchment", "xampp / ampps", "Palo Alto", "KubeCon", "pexels URL" (all T_12), the main job "MIT license / opensource URL tokens" (T_13), code identifiers (git-svn, #![, Euler), "Solana", "HuggingFace" (T_13), "David Beckham surname", "Industrial Hemp", "Canyon (Laurel/Diablo)" (T_14), "Thomson Reuters" (T_15). Typical end contexts: "opensource.org/licenses⟦/MIT⟧", "David⟦ Beckham⟧", "Palo⟦ Alto⟧". None of the 15 job names mentions lithium, and the word "lithium" does not appear anywhere in the contexts the explainer saw for this pole (P `explainer_saw`, 30 marked contexts). The explainer called the pole ONE job: "Fires on the word or word-piece that completes a well-known multi-part name or fixed compound term" (P `explanation`).

The judge rates 9 of the pole's 12 primary late-born jobs as unrelated to the earlier record (C `flagged`). Its main job is late-born and flagged (acc_random 0.0), so it is one of the 17 H_MIG poles (M `q2.H_MIG.poles`).

**Why an AI researcher cares:** this is a single neuron whose dominant behavior was replaced mid-training, visible only because the parent test happened to save its earlier top contexts. A one-job explanation of the final model says nothing about the job it used to have.

### 3. Most of the structure is in place by 40% of training, and the birth rate per step collapses

**Hook:** jobs keep appearing until the end, but per training step the rate of new jobs falls about 190-fold from its peak.

**What the data shows.** Birth counts per checkpoint (M `q2.births`):

- main jobs (600): 0, 1, 1, 5, 8, 19, 17, 34, 42, 50, 74, 89, 97, 109, 47, 7.
- extra jobs (1,854): 0, 0, 2, 9, 22, 30, 45, 63, 118, 126, 164, 257, 370, 380, 239, 29.

Cumulative (mine, from those counts): 56.7% of main jobs are born by step 180k (T_11, 12.7% of training), 72.8% by step 320k, 91.0% by step 560k (39.6%). Extra jobs: 45.1%, 65.0% and 85.5% at the same points. Main jobs are born at median T_11 and extra jobs at median T_12 (F4 caption).

Jobs first seen after step 560k (born T_14 or T_15) are 322 of 2,454 = 13.1% of all jobs (mine): 54 of 600 main (9.0%) and 268 of 1,854 extra (14.5%). The brief's "14%" matches the extra-job share, not all jobs.

Per 1,000 steps, across all 600 poles (mine; births in a bin divided by the bin's width): 14.0 (T_3), 15.0 (T_4), 16.3 (T_5, steps 5k to 8k, the peak), 15.5 (T_6), 12.1, 10.7, 7.0, 5.95, 4.3, 3.3, 2.0 (T_13), 0.65 (T_14), 0.087 (T_15, steps 1M to 1.41M). Peak over last = about 188.

Almost nothing is born in the first 1,000 steps: 1 job of 2,454 (M `q2.share_born_T_1` = 0.0004). It is a main job, L05N534- "product noun head (optimizer) in TF docs". The next earliest main is L03N10451+ "off after switch/turn" (T_2, step 2k). The pre-registered resolution pivot (more than half born at T_1) did not fire (M `q2.pivot_more_than_half_born_T_1` false).

**Why an AI researcher cares:** the count histogram peaks late only because the grid's late bins are wide. Per step, job formation is a burst in the first 1% of training followed by a long, thin tail, the shape you would expect from a learning-rate-limited process.

### 4. The Trello neuron: one brand became a category, then the brand left

**Hook:** at steps 35k and 60k, all 20 of L04N6289+'s top contexts were "T|rello". At the end, Trello is gone and the pole completes app names in general.

**What the data shows.** Records (mine, C `judge_saw`): T_8 (step 35k) "rello" 20 of 20; T_9 (60k) "rello" 20 of 20. Example: "The Best Apps For Project Management ... 1. T⟦rello⟧". At T_14 (1M) the top-20 is " Music" 7, "utilus" 4, "lab" 3, "XD" 2.

Its six jobs at T_15 (C): main "second token of Google product names (Drive, Photos, Classroom, Suite, Docs)" born T_8; "completing app/brand names: Apple Music, Spotify" T_8; "completing messaging/chat/social app names (Viber, Discord, Messenger, Notion, T...)" T_8; "Nautilus/gedit and OneDrive software name completions" T_8; "GitLab/Bitbucket second token" T_9; "LXD suffix after L" T_10. The two late-born jobs were judged against Trello-only records and passed 4 of 4 against random decoys (C `acc_random` 1.0 for both). The explainer's one-job description: "Fires on the name of a software app or online service, especially the second word or piece of a branded product name" (P). No "rello" anywhere in the contexts the explainer saw (P).

**Why an AI researcher cares:** this is recursive generalization in the essay's sense with a twist the essay does not discuss. The seed job generalized into a category and then fell out of the neuron's top contexts, so a study of the final model could never find the seed.

### 5. Family trees that pass every test

**Hook:** some neurons grow a lineage you can read token by token, and the judge confirms every link against both decoys.

**What the data shows.**

- **L03N7155+, the "got" neuron.** Jobs: "whole-token Got starting a clause" (born T_4, step 5k), "otta/onna of gotta and gonna" (T_7, 20k), "split ot/OT piece of got" (T_8, 35k), main "tin in gettin' (dropped-g getting)" (T_10, 100k). All three late jobs score 4/4 against random decoys and 2/2 against hard decoys (C `acc_random`, `acc_hard`). The top-20 marches with them (mine, C records): T_6 "Got" 18/20; T_7 "onna" 15 and " Gott" 5; T_9 "onna" 13; T_14 "otta" 14. End contexts: "G⟦otta⟧ admit, it's a lot lighter", "Get⟦tin⟧' tipsy". The main job is the youngest of the four.
- **L04N3936-, the "system" neuron.** "'System' after Management/Quality" (T_3, step 3k), then the main "method/process/approach/technique (system-synonym)" and "system/systems word or fragment (sys-tem, sistem, SISTEMAS, テム)" (both T_10), then "management-system acronyms (QMS, ISMS, ITSM, ITIL, ISO)" (T_11). All three late jobs 4/4 random and 2/2 hard (C). The word spread to its synonyms, its translations and its acronyms.
- **L01N4337-, the "beginnings" neuron.** Main "Birth token" (T_3), "Origin/originate (source, origination, 来源)" (T_4), "Begin/start/commence" (T_10, 4/4 random), "Produ-" (T_12, 2/4). Records: "Origin" 20/20 at T_9; at T_14 the top-20 holds " Beginning", " originates", "来源", " Childhood", " Starting" (mine, C).

Across the study, 45 poles have three or more primary late-born jobs that all score 4/4 (mine, C).

**Why an AI researcher cares:** these are clean, checkpoint-resolved examples of the essay's mechanism in a 2025 open LM, with births dated to within one grid interval. They are good candidates for causal follow-up.

### 6. The main job is the oldest only by a hair, and in half the neurons it overtook an older job

**Hook:** the study's "main job is oldest" result counts ties. Counting only clear wins, the main job is the uniquely oldest in 22% of neurons against 17.5% by chance.

**What the data shows.** b = 0.476 [0.436, 0.517] over 574 poles, null mean 0.414, null 97.5th percentile 0.449, p = 0.0005 (M `q2.birth_order`). The code counts a main job tied for earliest as earliest (Code line 578). Only 365 of the 574 poles have a single earliest job; in 209 the earliest checkpoint is shared by 2 to 7 jobs (mine, C). With ties removed, the main job is alone at the front in 126 of 574 poles (22.0%) against an expected 17.5% under the same within-pole permutation null (mine).

In 301 of 574 poles (52.4%, the study's "drift share") the main job was born strictly after another job (mine, matches M `drift_share`). In 10 poles the gap is 8 or 9 checkpoints (mine). Examples (C):

- L18N3476+: "intensifier 'very' after determiner" born T_4 (step 5k); main "web UI boilerplate list items (Main, More, Create...)" born T_13 (560k).
- L30N6280+ (layer 30): "non-science web text: 1Win betting spam and music biography" born T_2 (step 2k; only 2 extra jobs in the study are born that early); main "Sr2IrO4 Mott insulator seminar abstract (physics text)" born T_10, with a higher end peak (18.86 against 12.67, C `peak_end`).

**Why an AI researcher cares:** the "stays on its field" picture is a weak tendency, not a rule. A neuron's first job and its strongest final job differ more often than not.

### 7. Jobs that died are invisible to the main readout, but the records caught several

**Hook:** a neuron that read URL query strings at step 35k ends training laughing ("Haha").

**What the data shows.** Because jobs are defined at T_15, a job that existed and then vanished cannot appear in C as a job. It can only show up in the earlier top-20 records the judge saw. Cases where a record is dominated by something absent from the pole's final jobs (mine, C records; P `explainer_saw` checked for the word):

- **L06N7759+.** T_8 (35k): "/?" 11 and "?" 6 of 20, e.g. "go.microsoft.com/fwlink⟦/?⟧LinkId=317598", "wordpress.org⟦/?⟧v=2.8.4". T_13 (560k): "haha" 8, "aha" 6. T_14: "aha" 8, " Nah" 2. Final jobs: "informal 'Nah' dismissal" (T_4), main "laughter interjections (haha, lol, hehe, HA)" (T_9), "Magic card definitions (+X/+X ...)" (T_14). No URL job; "/?" absent from the explainer's contexts. End context: "Oh, ping-pong is an Olympic sport. H⟦aha⟧." This pole is in H_MIG.
- **L12N1003-.** T_9 (60k): " chemical" 13 and " chemicals" 3 of 20, e.g. "basic⟦ chemical⟧ capacity will expand". T_14: " does" 9, " is" 3. Final jobs: modal after It/They, They, does/did after It/He/They, "is" after It. Both late jobs score 0/4.
- **L03N10451+.** T_5 (8k): " off" 10 of 20. T_8 (35k): " Knight" 10 of 20, every one in "Dark⟦ Knight⟧". T_14: "emon" 11, " Terminator" 3. Final jobs: "off after switch/turn" (T_2), "shutoff/termination word pieces" (T_6), "Terminator" (T_8), "emon in Doraemon" (T_9). No "Knight" anywhere in the explainer's contexts.
- **L23N809-.** T_7 and T_8 (20k, 35k): marketing-copy nouns, "Beyond academic⟦ excellence⟧", "Engaging Learning⟦ Experience⟧", "Fast-Track Skill⟦ Acquisition⟧". T_14: " Gul" 10, " Bel" 5. No such job at the end.

At scale the turnover is large in shallow layers. In layers 0-10, a pole's top-20 at step 100k shares a median of 0% of its marked tokens with its top-20 at step 1M, and 55% of those poles share none (38 poles); between steps 560k and 1M the median share is 15% and 14% share none (78 poles) (mine, C records). Token overlap is a crude measure for deep layers, where marked tokens are often function words, so I do not quote it for them.

**Why an AI researcher cares:** "what dies" is the half of the developmental story this design cannot count, and these records show it is not empty. A study that defines jobs from the start of training, or from every checkpoint, would see deaths and replacements that the T_15-anchored count calls zero.

### 8. Two clocks: a neuron's theme arrives early, its exact top examples arrive late

**Hook:** by step 3,000 the median neuron already answers its own end-of-training description at 30% strength. Its actual top contexts need until step 180,000 to get there.

**What the data shows.** The study's center birth curve: the blind-maximum probe sentence reaches 0.3 of its T_15 score at median T_3 (step 3k, 0.21% of training) over 598 poles; ONE-class poles T_4, TWO-REL T_6, UNK T_1 (M `q2.center_birth_curve`, R). I applied the same rule to the jobs' own exemplars, dividing each job's median exemplar activation by the pole's max|a| at each checkpoint (mine, C `med`, H `max_abs`). The first checkpoint where a job reaches 0.3 of its own T_15 value is median T_11 (step 180k) for all 2,454 jobs and also T_11 for main jobs. The median job's exemplars sit at 0.012 of the pole's max at step 12k, 0.046 at 35k, 0.114 at 100k, 0.190 at 180k, 0.291 at 320k, 0.404 at 560k and 0.681 at the end (mine).

**Why an AI researcher cares:** "when does a feature form" has two answers 60 times apart in steps, depending on whether you ask about the neuron's direction or its sharpest examples. Claims about early feature formation need to say which one they mean.

### 9. The judge missed an obvious sibling, and the migration count includes it

**Hook:** L23N809- fires on " Gul" in "gulp", "gulch" and "Gulbenkian". The judge accepted gulch as a descendant and rejected Gulbenkian, so this pole is counted as a "migration".

**What the data shows.** Jobs (C): "gulp / gulping verb" (born T_5), "gulch / gulley terrain word" (T_8, 4/4 random, 2/2 hard), main "Gul- at start of proper names (Gulbenkian, Gulian, Gulati...)" (T_9, 1/4 random, 0/2 hard, flagged), "Bel- word-start fragment" (T_10, 4/4), misc weak tokens (T_12, 4/4). The gulch and names jobs mark the same token: "San Gabriel⟦ Gul⟧ch" and "Calouste⟦ Gul⟧benkian". The explainer called the pole ONE: "Fires strongly on the word-piece 'Gul'" (P). The flagged main makes it one of the 17 H_MIG poles (M).

Why the miss: the record the judge saw at T_8 was the marketing-noun job from story 7 ("excellence" 4, "." 5 of 20). The gulp job had been present since T_5 by the percentile rule but was not in the top-20. So "unrelated to the earlier record" here means "the parent was not the pole's dominant job at the time". The gulch job, judged against a similar gul-free T_7 record, still scored 4/4 (mine, C records), which also says the judge's verdicts are noisy.

A second H_MIG pole, L06N7759+, shows the opposite kind of disagreement: its main laughter job scores 0/4 against random decoys and 2/2 against the hard decoy (C).

**Why an AI researcher cares:** the 17-pole migration count (2.8% of S) mixes real replacements (stories 2 and 7) with instrument misses. The parent test asks "did this grow from what the neuron mostly did?", not "did this grow from anything the neuron did?".

### 10. Late bloomers, and polarity that settles late

**Hook:** one last-layer neuron's future jobs pushed it the wrong way, or not at all, until step 180k to 320k (12.7% to 22.6% of training). Then it switched on.

**What the data shows.** L31N3362- (layer 31, UNK class; explainer: "Fires at the last position before the model expects a next token starting with a capital O", P). Its four jobs (C `med`, median exemplar activation in the pole's direction, clipped at 0): every one is exactly 0 at every checkpoint from T_1 to T_11 (three of them through T_12), then turns on, e.g. "General web text where the next token starts with O (OEM, OLED, Oxley...)": 0.0 through T_12, then 10.3, 31.3, 32.7. All four are born at T_13 or T_14. A median of exactly 0 means at least half the job's own end-of-training contexts drove the neuron the other way or not at all (Code `signed_magnitude`).

This is the extreme of a population pattern (mine, C `med`). The share of jobs whose median exemplar is 0 at a checkpoint is 44.5% at init (random signs give about half), 25.1% at step 12k, 16.1% at 60k, 11.0% at 100k, 6.9% at 180k, 3.0% at 320k and 0.5% at 560k. 1,758 of 2,454 jobs (71.6%) have at least one such checkpoint after init. Deep layers settle first: at step 12k the share is 34.7% in layers 0-10, 24.2% in 11-21 and 12.2% in 22-31 (P `tercile`). Main jobs are born later in shallow layers too: median T_12 in layers 0-10, T_11 in 11-21, T_10 in 22-31 (mine).

Whole neurons can start late: 14 poles have no job at all until step 1M (70.7% of training), 78 none until step 560k (mine, C earliest birth per pole). 7 main jobs are born at T_15, absent at step 1M (M `q2.births.main`). Example: L14N4543+, whose main "mid-word/identifier subtoken completion (camelCase, names, compounds)" is born at T_15 and judged unrelated 0/4 (C; it is in H_MIG).

**Why an AI researcher cares:** "shallow layers learn simple token features first" is the default intuition. Here the shallow neurons are the slow ones, both in settling which way their contexts push them and in when their main job appears.

### 11. Extra jobs are not leftovers, but the design could not have seen leftovers

**Hook:** 58% of extra jobs are as old as the main job or older. Only 5% of extra jobs shrank in the last 77% of training.

**What the data shows.** Vestigial share v = 0.035 [0.026, 0.044] over 1,854 extra jobs (M `q2.vestigial`); the H_VEST rule needed a lower bound above 0.5. The two halves of the definition fail very differently (mine, C): 1,071 of 1,854 extra jobs (57.8%) are born at or before their pole's main job, so the timing half holds for most. Only 92 of 1,854 (5.0%) shrank from T_12 to T_15 (C `growth_last_half` < 0), so the shrinking half almost never holds. Main jobs shrank in 20 of 600. Only 2 of 2,454 jobs (0.08%) ever reached twice their final normalized strength at an earlier checkpoint (mine, C `med` and H `max_abs`). The largest shrinker is L20N10809-:c1, "There is no one/universal answer to this question" in repeated promotional text: 0.82 of the pole's max at T_12, 0.53 at T_15 (mine).

**Why an AI researcher cares:** the 2021 question "are mixtures leftovers from early training?" gets a "no" here, but see Part 3: a job picked from the final top-50 is selected to be strong at the end, so "shrinking" is close to ruled out by construction.

---

## Part 2. Numbers that surprised me

1. **Activations shrink by two orders of magnitude.** Per pole, max|a| at step 3k is a median 392 times its final value; 180 times at init, 22 times at 60k, 7.3 times at 100k, 2.4 times at 560k, 1.6 times at 1M (mine, H `max_abs`, 600 poles). Layer 31's median max|a| over all its neurons is about 3,990 at steps 5k-8k and 28 at the end (mine, H). This is why every presence rule in the study is per-checkpoint relative. I did not check which tokens carry those early maxima.
2. **Neurons get sparser.** The median pole's 99.9th percentile falls from 0.377 of its max at init to 0.178 at steps 560k-1M (mine, H `q999`/`max_abs`).
3. **"One job" neurons have four jobs.** Of the 311 poles the explainer called ONE, only 3.5% get a single cluster at T_15; the median is 4 (mine, C and P). Even the study's ONE gallery pole, L25N5542- (CT tomography, main born T_8), has 4 extra jobs born at T_12-T_13 (M `q2.gallery`). "4.1 jobs per pole" (2,454/600, C) is the clusterer's granularity, not 4 unrelated functions.
4. **Nothing is born strong.** Median strength at birth r = 0.295 for all jobs and 0.291 for late-born ones (M `q2.r_at_birth`); not one of 2,454 jobs is born at 0.9 of the pole's max or more, and the 95th percentile is 0.555 (mine, C).
5. **Unrelated jobs are born exactly as weak as related ones.** Median r at birth: 0.290 for flagged (judged unrelated) late jobs, 0.291 for late jobs judged 4/4 related (mine, C). H_SP's side prediction that packed-in jobs "appear at full strength" (SPEC §1) gets no support. Flagged jobs do grow faster in the last stretch (median growth 0.537 against 0.382).
6. **The main job is uniquely oldest in 22.0% of poles**, against 17.5% by chance (story 6). The headline 47.6% counts ties.
7. **A pole's response to its own explanation is not monotone.** Only 1% of per-pole center curves are non-decreasing (M `q2.center_birth_curve.blind_max.share_nondecreasing` = 0.010), and 111 of 598 poles already reach the 0.3 line at random init (R).
8. **Jobs born at the very end look strong because of how jobs are picked.** Median r at birth is between 0.257 and 0.352 for jobs born T_4 to T_13, then 0.411 at T_14 and 0.673 at T_15 (mine, C).

---

## Part 3. What I could not verify, or what cuts against these stories

- **Survivorship is built in.** Jobs are clusters of the T_15 top-50. Every job is present at T_15, "never born" is 0 (M `q2.status_counts`), growth into T_15 is favored by selection, and a job that died has no row. So v (story 11), the low shrink rate and the "no job ever peaked early" numbers are partly forced by the design. Deaths (stories 2 and 7) are visible only through the judge records, which exist only for poles and checkpoints the parent test happened to need. I have no count of deaths, only examples.
- **"Born" is a threshold crossing, and the threshold moves.** The 99.9th percentile is 0.38 of max at init and 0.18 at the end (Part 2 item 2). r at birth (0.29) sits just above that threshold, so "born weak" mostly restates the detection rule. Birth also needs only two consecutive present checkpoints, not permanent presence (44 jobs drop out again). F4's caption says "stays there", which is looser than the code.
- **The grid is coarse late.** T_13 spans 240k steps and T_14 440k. A job "born at T_14" formed somewhere between 39.6% and 70.7% of training. The count peak at T_13 is a bin-width effect, which is why I quote rates per step.
- **The two birth readings disagree on 19% of jobs.** The percentile and raw-ratio readings agree on the late/early split for 81.0% of jobs (M `q2.B3_late_split_agreement`); 244 jobs are "readings_disagree" and leave the parent test. The 10M/20M corpus check is reassuring (1.3% change, M `q2.B2_split_changed_10m`).
- **The clusterer is noisy.** Two independent clusterings agree on the job count within one for only 67.8% of poles (M `q2.B4_m_within_1`). The 15-job lithium pole and the "4 jobs per ONE pole" figure both depend on this one clusterer sample.
- **Story 1 is exploratory and has a confound I cannot remove.** The judge's ceiling a_ctrl was measured only on main jobs against T_14 records. If the judge finds related pairs harder at late checkpoints (busier top-20s, minor parents hidden behind a dominant job, as in story 9), the ramp would partly be a ceiling ramp, not a relatedness ramp. Literal token overlap between exemplars and the true record also falls with birth time (26% of judgments for jobs born by step 100k, 14% for jobs born at 560k or later, mine), so easier surface cues may help early jobs. The hard-decoy control is below its validity bar (0.798 against 0.80, M `q2.hard`), so the "judge wins by domain" explanation is not ruled out for any of the parent-test numbers.
- **The parent test only sees the dominant job.** The record is the top-20 at one checkpoint. A late job whose parent was a minor job of the pole will look unrelated (story 9). So p_u, the ramp and H_MIG all overcount "unrelated" by an unknown amount.
- **Story 8 compares two different statistics.** The center curve uses the maximum over a probe sentence; my job curve uses the median over exemplar tokens. A maximum over many tokens crosses a 0.3 line more easily, so part of the 60-fold gap is instrument, not development.
- **Q1 classes are shaky inputs to any by-class split.** On O7 the cross-model label agreement is kappa 0.070 (R), and per-pole classes depend on the explainer model for about 28% of poles (R footnotes). No story above rests on ONE versus UNK.
- **The judge itself.** All 11,034 judge lines on S ran through Claude Code Sonnet subagents on a manual backend, with an audit history of re-answered chunks (R footnotes). The study's leave-out checks left p_u unchanged in label, but I did not re-audit the judgments I quote per pole.
- **One run, one stage, one corpus.** Everything here is one training run of one model, stage 1 only (T_15 is 100% of stage 1, SPEC §4.3), not the released model after later stages. The grid corpus is 20,560,068 tokens (M `q2.grid_population`); the explainer's contexts come from a different, larger scoring run (50.5M tokens, P `explainer_saw` header). Checkpoints T_7 to T_14 were harvested on a different GPU; the study's sensitivity check moves neither label (R).
- **Token-overlap turnover (story 7) is crude.** Marked-token overlap ignores meaning and the records drop contexts near each job's own exemplars, so it can overstate churn. I use it only for layers 0-10, and the named examples were read by eye.
- **One headline in the brief did not reproduce exactly.** All jobs born in the last 60% of training are 13.1%, not 14% (14.5% is the extra-job share).
