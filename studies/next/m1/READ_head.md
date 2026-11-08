# M1, the head's read (Fable, the different-model read, 2026-11-05 23:35Z)

**Answer.** The verdict stands as written: `H_lead SUPPORTED; H_order UNDECIDED; H_specific SUPPORTED`. My own code
(`read_head/recount.py`, numpy only, reading the five `counts.npz` files and nothing from `infra/pg/m1.py`) gives every band
mean to the verdict's precision (largest difference 4e-17), every contrast mean, all twelve band labels, the identity pair
(exactly 0 in every band; the two files share sha256 `8c332d7d…`), and the specificity test with my own random sets (99th
percentiles within 3e-4 of the verdict's). Output: `read_head/recount_output.txt`. The Opus reader's blind re-implementation
(`READ_m1.md`) had already matched all 131 deterministic fields. The result is closed.

## What the numbers say, in plain words

The measure, per neuron pole: the fraction of its 50 strongest-activation positions on the fixed 20.6M-token corpus that fall
in the five code and math directories (C#, JavaScript, Python, SQL code; FineMath). Band means over the 176,128 poles of
each block of 8 layers. The corpus has 11.6 % of its tokens in those directories.

| checkpoint | layers 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| start (stage 1, 2T tokens; harvested twice, identical) | 0.1238 | 0.1109 | 0.0761 | 0.0547 |
| G, 100B tokens with 0 % code+math | 0.1125 | 0.0938 | 0.0588 | 0.0386 |
| R, 40 % code+math | 0.1202 | 0.1007 | 0.0702 | 0.0536 |
| M, 65 % code+math | 0.1234 | 0.1038 | 0.0761 | 0.0583 |
| drift bar (stage-1 neighbours) | 0.0082 | 0.0066 | 0.0023 | 0.0035 |

- The supported contrast M − G is +0.0109, +0.0100, +0.0174, +0.0196 of 50 slots: half a slot to one slot per pole. Its
  lower bounds exceed the drift bar by factors 1.3, 1.4, 7.5 and 5.4; it is 5 to 10 % of the ceiling κ.
- The contrast is mostly G moving away from code and math, not M moving toward it. G is 0.011 to 0.017 below the start in
  every band. M is within the bar of the start in three bands, 0.007 below it in layers 8–15, and above it only in layers
  24–31 by +0.0035, which equals that band's bar. R is below the start in every band.
- The plain statement: 100B tokens of training with no code or math move the neurons' strongest activations off code and
  math documents by 0.6–0.9 of one slot (corrected 2026-11-08 after the abstract check: the earlier text said "about one slot in fifty"; the shift is −0.0113 to −0.0173 of 50 slots); a mix with 65 % code and math moves them by at most 0.36 of a slot (corrected the same day: "holds them where stage 1 left them" was too strong for layers 8–15, −0.0070 against the bar 0.0066); a 40 % mix
  holds them 66 to 76 % of the way (λ = Δ_RG / Δ_MG = 0.70, 0.68, 0.66, 0.76; 0.62 if linear in the share).
- "Code and math" is mostly code. By directory, C# carries +0.0048 to +0.0056 of the contrast in every band, JavaScript
  +0.0033 to +0.0053, Python +0.0010 to +0.0053 (large only in layers 16–31), SQL +0.0006 to +0.0023. FineMath is
  negative in layers 0–15 (−0.0011, −0.0011) and positive in layers 16–31 (+0.0012, +0.0021).
- H_order. R lies between G and M in every band, and both gaps are positive with intervals that exclude zero (M − R lower
  bounds 0.0027, 0.0028, 0.0057, 0.0044). The frozen rule asks each gap to exceed the drift bar, and in layers 0–15 the
  M − R gap (0.0032) is below it (0.0082, 0.0066). UNDECIDED therefore means: the ordering holds at the resolution of the
  bootstrap but not at the resolution of ordinary stage-1 drift. It does not mean R falls outside the G–M range. The spec's
  §1.2 row for this outcome says "Round5 does not sit between them"; that sentence is wrong for these data and must not be
  used in any write-up.
- Controls. The positive control fires: M's loss on 32 code+math windows is 0.148 nats below G's, and 0.240 nats below it
  relative to 32 web windows, so the two mixes differ on these documents as their names say. The floor: the identity pair
  (the start harvested twice on two instances) differs by exactly 0 in every band, and the only run-to-run spread with any
  size is the banked cross-device pair, 2.8e-5. G-ID's failing input (T_15 for start_b) fails at 0.0185.
- One example read in `raw_examples_m1.md` (pole L04N3542−, the largest shift under M, 0.22 → 0.64): its strongest positions
  are the tokens `erg`, `gent`, `-div`, `.prod`, `.Cross` in "divergence", "convergent", "cross product", `numpy.prod`.
  The feature is a token-level one that occurs in web, PDF and code documents alike; under M, code documents supply more
  of its strongest instances. So the measure answers "which documents hold a feature's strongest instances", which a mix
  can change without changing what the feature responds to. The write-up should say so.

## The three dimensions

- Predictive: pre-registered, thresholds frozen on stage-1 data before any mix was read, unblinded once. The primary
  prediction landed (prior 0.65) at the low end of the prior's range (+0.01 to +0.06). The ordering prediction (prior 0.4)
  did not resolve. λ landed above the linear expectation.
- Novel: as far as the spec's literature section found (§11), the shift of neuron top-activation document preference under
  documented stage-2 mixes at 7B had not been measured. Not re-checked by me.
- Practical: nothing measured against a baseline. A descriptive result.

## Code–spec differences and threats

I agree with the eight differences in `READ_m1.md` §3; none bears on this verdict. Two must be fixed before the instrument
is reused: H_order must be evaluated whatever the lead's label (spec §1.3), and the verdict's G-SAME must compare against
the registry (`m1.SAME`), not the harvest's own copy. Threats left open: one FineMath document of 398,967 tokens holds
half that directory's tokens (the reader bounds its share of the contrast at 0.0003); one GPU model; one corpus; the
spec's "about 10 %" code+math token share is 11.64 %.

## Decision

Closed as the verdict states. Under spec §1.2 the next step for this outcome is to report, not to run the per-pole
analysis. The program's abstract carries the plain statement above, with the caveats on direction (G moves, M holds) and
on content (mostly code; a token-level feature's strongest instances).

## Qualification after the two exploratory checks (head, 2026-11-05 18:20 CST)

Two zero-GPU checks on the banked data (`EXPLORATORY_turnover.md`, `EXPLORATORY_reading_control.md`, branch m1x) change
two sentences above.

- "Mostly G moving away" is a loss at fixed pole identity, not a move between poles. Of the poles in the top tenth of
  code+math share at the start (threshold frozen on stage-1 references), 0.29 to 0.40 fall below it under G by band,
  while only 0.085 to 0.165 of the others rise above it: the lowest arrival fraction of the nine pairs that span
  training, and 0.07 to 0.20 below M's in every band (paired intervals exclude zero). Poles below the threshold lose
  code and math under G too. The check cannot see a move out of the neuron basis or into ranks below 50. The
  frozen label ("loss at fixed identity" in layers 8–31) is not specific on its own: a planted long stage-1 span
  (start to T_15) earns the same label, so the arrival fractions, not the label, carry the reading.
- The clause "which a mix can change without changing what the feature responds to" is too strong. For L04N3542−
  the two token families persist, but their order changes under M: divergence word-pieces fall from 37 of the top 50
  at the start to 27 under G and 6 under M, while vector-product operations in code rise from 11 to 22 to 40, and at
  fixed positions the response to `erg` in "φ-divergences" relative to `.cross(` goes from 1.22 to 1.08 to 0.44. The
  mix changes which of the feature's token families dominates its strongest instances, and with it which documents
  hold them. The token list picks out the pole (none of five matched random layer-4 poles shares it); the phrase
  "a token-level feature spanning document types" does not (three of the five fit it). Write-ups use the token list
  and the three counts, not the phrase.
