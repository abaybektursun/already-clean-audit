# N13: the head's read of the verdict and of the independent read

2026-11-08 05:40 CST (Fable). Read: `verdict_n13.json` (sha256 cae55c04…), `thresholds.json` (60bf6d4e…), `CLAIMS.md`, `REGISTRY.md`,
`results/SAMPLES_COVERAGE.md`, the executor's `JOURNAL.md` and `STATE.md`, `READ_n13.md` and the reader's outputs under `read_n13/`
(`recount_read.json`, `verdict_read.json`, `checks_read.json`, `compare_verdict.json`, `samples_read.json`, `gates_read.json`).

**The frozen verdict stands and is quotable: the RL math gain of OLMo-3.1-7B-RL-Zero-Math over its base (MATH-500 92.0 [89.3, 94.1]
against 57.2 [52.8, 61.5]) is not carried by the most-changed MLP neurons at any size. Transplanting the top 0.1, 0.3, 1 or 3 % of
neurons by change into the base reads 57.6, 61.4, 61.8, 63.8, every one below the halfway line 74.6; random neurons matched on change
energy read the same (58.0 to 63.8); H_local REFUTED. A random sample of as many MLP entries, scaled to estimate the whole change
(DARE), reads 85.2 to 87.8 at every size; H_diffuse SUPPORTED. All MLP alone reads 86.4 and attention with norms alone 83.0, so neither
family is necessary; H_attention REFUTED (BOTH). Removing the top set from the RL model costs nothing (90.4 to 91.6).** The reader
recomputed every one of the 30,009 banked answers' scores by re-grading and re-extracting (0 disagreements), the 31 accuracies, the
14 paired tests, the labels and the hypotheses with its own code before opening the verdict, and 465 of 465 verdict fields agree;
all 13 gates pass with their failing inputs failing; the manifest's 165 files verified by sha256.

## What holds, and how firmly

1. No hypothesis verdict moves under any reading the reader tried: cap-hit answers scored wrong, cap-hit problems dropped (top_3 %
   then reads PART, k_rec stays NONE), a lenient re-score that credits the gold in formats the rule does not read, 100 other bootstrap
   seeds, BH at q 0.10, a Wilson-based interval in place of the bootstrap (`checks_read.json` b, c, g). The decisive comparisons rest
   on 141 to 164 discordant problems (top against DARE) and never on fewer than 56.
2. The controls are what they claim. rndE's realized energy is within 0.8 % of the top set's at every k; rndT is never short
   (765, 1,861, 5,221, 13,910 neurons); DARE's kept counts are within 0.74 binomial standard deviations; the second draws and the
   seed-B top arm are within δ of their first readings (+1.8, −0.4, +2.0); σ_mid is 0.98 σ (checks d, e).
3. The held-out benchmark agrees on the decisive line: GSM8K top − dare −23.7 [−26.5, −21.1], and top_1 % 66.0 against base 63.6
   (check i). The two REPLICATES labels for top − rndE (+2.1 [+0.2, +3.9]) and top − rndT (+1.9 [+0.1, +3.6]) are fragile: under a
   Wilson-based interval both read CONSISTENT and top − rndT flips under 18 of 100 bootstrap seeds. Quote those two as consistent,
   not as replicated.
4. The format of the answers does not make the gap. RL and every arm that recovers the gain box their answers; base and every arm
   that does not write an "Answer:" line in 39 to 46 % of answers. Within boxed answers the recovering arms read 92.6 to 94.4 and the
   others 65.5 to 82.4; the lenient re-score moves no hypothesis (check c). The rule does cost base-like arms 17 to 31 answers each,
   recorded as a limitation of the registered extraction rule, not a retag.
5. The mlp − attn difference (+3.4, BH p 0.107) lies entirely in cap-hit problems (0.0 on the 423 uncapped); the two halves are
   not separated.

## What the result does and does not say

- "Diffuse" is a statement about neuron subsets, not about entries at matched energy. DARE keeps each entry with probability q and
  scales it by 1/q, so its realized change energy is 33 to 1,000 times the whole change's and 845 to 463,347 times the top set's
  (check d). DARE reads the same as the whole MLP change (dare − mlp between −1.2 and +0.4, p ≥ 0.50; check j1). So the verdict says:
  the gain sits in the direction of the whole MLP change, which an unbiased sparse estimate of it carries and which no set of 0.1 to
  3 % of whole neurons carries, at its own energy or at the top set's. Whether the network tolerates the 1/q noise because the effect
  is low-rank, because it is spread evenly, or for another reason is not measured here; that is the next instrument (below).
- The top set is not special, and the claim that it carries a small real part cannot be made under the frozen rule. top − rndE and
  top − rndT are ≥ 0 at k ≥ 0.3 % (+1.6 to +3.4, none significant after BH), top_1 % reads 84.9 against rndE_1 % 78.7 among extracted
  answers, and on GSM8K top_1 % sits 2.4 points above base where rndE and rndT sit within 0.6 (checks i, j2). A few points at most,
  far under δ 5.44 and under the 17 points to the halfway line. Exploratory.
- The top set's identity is only moderately stable under the rounding redraws (J_floor 0.642, J_round 0.534; H_artifact
  INDETERMINATE), which is consistent with a ranking that selects nothing special.
- The RL change is a package. Every arm that recovers accuracy also adopts RL's boxing and its longer outputs (mean new tokens 4,333
  to 4,671 against 3,133 to 4,004; cap-hit share 3 to 12 % against 12 to 17 %); the arms at base level keep base's habits. The
  format and the length travel with the accuracy under DARE, under all-MLP and under attention-with-norms alike (checks b, c).
- Practical dimension (spec §11), measured: a random 0.1 % of the MLP entries of the RL delta, scaled by 1,000, carries 85.2 of
  the 34.8-point gain's endpoint 92.0 on MATH-500 (the 1 % sample reads 89.8 against 92.3 on GSM8K; corrected 2026-11-08 after the abstract check, which found the GSM8K reading attributed to the 0.1 % arm, which has none); the whole-neuron top sets, the natural
  object of "localized" accounts, carry none of it. For transplanting or merging an RL delta, the cheap object is a sparse
  unbiased sample of the delta, not its most-changed neurons.

## Reading

Of the program's five hypotheses the localized one is gone and the diffuse one holds, in the strongest form the design could
show: the most-changed 3 % of neurons, holding 3.9 % of the change energy, move MATH-500 by 6.6 points over base; a random
thousandth of the entries, rescaled, moves it by 28. The next instrument is a rank test rather than a sparsity test: a rank-r
approximation of the delta per matrix against random subspaces of equal rank and energy, which separates "low-rank direction"
from "evenly spread", and tells whether DARE's tolerance is the tolerance of a low-rank signal to isotropic noise. Hypothesis to
carry, with its refutation: the gain is carried by a low-rank component of each matrix's change (recovered at rank r ≪ min(d, n)
within δ of RL); refuted if the rank-r arm at r = 64 reads at the random-subspace control's level.

## Status of the claims

| id | claim | status |
|---|---|---|
| N13-1 | H_local REFUTED (k_rec NONE): top_k 57.6, 61.4, 61.8, 63.8 (k 0.1, 0.3, 1, 3 %), all NONE against H 74.6; rndE and rndT within δ at every k (smallest BH p 0.107) | stands under the frozen rule |
| N13-2 | H_diffuse SUPPORTED: dare_k 85.2, 86.8, 85.8, 86.2 REC; top − dare −27.6, −25.4, −24.0, −22.4, BH p ≤ 2.6e-21; dare equals the whole MLP change (within 1.2 points) at 33 to 1,000 times its energy | stands; "diffuse" read as in §2 above |
| N13-3 | H_attention REFUTED (BOTH): mlp 86.4 REC, attn 83.0 REC; the +3.4 lies in cap-hit problems | stands |
| N13-4 | Necessity: comp_k 90.4, 90.8, 91.6, 90.6 within δ of rl; NOT NECESSARY at every k | stands |
| N13-5 | GSM8K at 1 %: top − dare −23.7 [−26.5, −21.1] REPLICATES; top − rndE +2.1 [+0.2, +3.9] and top − rndT +1.9 [+0.1, +3.6] labeled REPLICATES by the frozen rule, fragile under seed and interval method | first stands; the other two quoted as consistent |
| N13-6 | The three numbers and the gates: G-KNOWN 39.96; G 34.8 against 13.60; identity pairs 0.8 and 1.8; S-BAND 91.2; PC-RANK 1.0; 13 gates pass, 13 failing inputs fail; controls matched as registered | stands |
| N13-7 | H_artifact INDETERMINATE (J_floor 0.642, J_round 0.534; PC-ART valid) | stands, reported |
| N13-8 | The registered extraction rule costs base-like arms 17 to 31 answers each (formats it does not read); no hypothesis changes under a lenient re-score | limitation recorded |
| N13-9 | The figure lacks the m_k annotation on the rndT points that spec §10.3 asks for, and three labels overlap at k 3 %; the plotted values are right | to regenerate (no verdict change) |

Not verified by anyone: the tensor hashes and masks against the weights themselves (the reader checked `build.json` for
consistency; R1's numbers are read from `controls.json` and `sets.json`); G-KNOWN from the R0 record; the generated token ids
(not banked; the text is the instance's decode); the golds against the Hugging Face files; the mechanism of DARE's tolerance.
