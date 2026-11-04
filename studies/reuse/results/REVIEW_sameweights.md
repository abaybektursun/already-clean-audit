# REUSE SPEC9 (the same-weights null for the shared-driver overlap) review, 2026-11-03 (Opus built the job and the verdict; Fable 5.1 coordinated and reviews; revised after an Opus read, 21 items, all applied)

Read-only on the bank (`sameweights/1/t15/`). Own recount `../scripts/review/fable_sameweights_recount.py` (`sameweights/recount_fable.txt`, `recount_fable_sameweights.json`; seed 1 against the verdict's seed 0): s_O, s_swap, d, the share with s_O > s_swap, the restricted read, the known number against SPEC6's sets and the room from the ceiling splits, from the npz and the items alone. Samples opened before any number (`samples/cases.json`: three cases' prompts for J, O* and a swap with their D′ sets; the first case's s_O 0.18 against swaps 0.02, 0.00, 0.02; its J token ids equal the items' and the first swap's equal its source case's J contexts; the pole's mean activation 0.199 on J against −0.002, −0.026 and 0.000 on the swaps).

**Verdict.** Every gate passes (known number against SPEC6 exactly 1.00; room 0.32; swap validity 0 against 243). W2 0.127 [0.093, 0.153] and W2f 0.133 [0.093, 0.167], both above 0.03: H_weights refuted on both sets. Four failing inputs each exit 1. Six of six §7 rows landed.

## 1. Verified, recount (seed 1) against the verdict (seed 0)

| quantity | verdict | recount |
|---|---|---|
| s_O primary; fresh | 0.140 [0.110, 0.180]; 0.160 [0.120, 0.180] | 0.140 [0.120, 0.180]; 0.160 [0.110, 0.180] |
| s_swap primary; fresh | 0.013 [0.013, 0.020]; 0.013 [0.007, 0.013] | same |
| d primary (W2); fresh (W2f) | 0.127 [0.093, 0.153]; 0.133 [0.093, 0.167] | 0.127 [0.100, 0.157]; 0.133 [0.093, 0.160] |
| share s_O > s_swap, primary; fresh | 0.866; 0.934 | 0.87; 0.93 |
| known number J; O* | 1.000; 1.000 | 1.000; 1.000 |
| halves overlap; room | 0.38; 0.32 | 0.380; 0.320 |
| restricted read, n primary; fresh | 2; 1 | 2; 1 |

## 2. Findings

**F1. The frozen reading reproduces exactly and every gate is a real test, but the registered null is weak in the way §8 named.** The known number against SPEC6's independent T_15 capture is exact (300 of 300 sets), B_t is rebuilt member for member, the seed-9 swap draw re-implements for all 243 cases, no swap shares a neuron or a marked token with its case (34 of 729 share a document), every failing input fires, the halves' ceiling (0.38) is reproduced here, and the 0.060 is SPEC4's banked number, asserted. The rule had room in arithmetic only: the pole was off on 726 of 729 swapped sets, the §8 guard caught 3 swaps, and the restricted read (2 and 1 cases) cannot be read. With the pole silent the attribution ranks the upstream neurons in a different coefficient regime (D′(swap) leans to the up loadings), so the null changed the ranking as well as the contexts, and s_swap does not move with the pole's activation (0.015 to 0.020 across bins; Spearman 0.07; 0.041 on the 16 sets at or above 0.05). My first version said s_swap was low "for that reason as much as any"; the bank does not show that.

**F2. The sharper null was already in the bank, and in SPEC6's bank before the launch, and it does not separate.** For the 108 live cases whose pole has other clusters among the cases (46 neurons), the pole is active on them at 0.71 to 1.32 of its J activation; D′(J) overlaps their drivers by 0.125 [0.090, 0.180] against 0.140 [0.100, 0.160] with D′(O*): d_third 0.000 [−0.020, 0.030] (fresh −0.040 [−0.080, 0.020]); SPEC6's T_15 sets alone give 0.005 [0.000, 0.050] on 64 primary cases, an offline reading that cost no GPU-hour and that the journal's 02:00 CST entry did not see. NEXT item 9's own refuting observation (the matched sets' overlap within 0.03 of s_O by interval) is met by this reading. The shared drivers are a property of the pole's clusters in general; the headline's "older task's inputs" is restated as the neuron's tasks sharing inputs (RESULTS Decision; `POST_full.md`).

**F3. S1 at T_15.** On the pole's own top contexts the two clusters' drivers overlap by 0.14 and 0.16 at the final checkpoint, above SPEC4's birth-checkpoint 0.12 (no rule attaches to the difference), with the halves at 0.38 and the pole's other clusters at 0.125.

**F4. Spec defects, for the next author.** (a) The swapped sets should have been matched on the pole's activation; §8 named the threat and the guard caught 3 swaps. (b) The same-pole sets were available offline; the spec should have read them first. (c) §5's room gate wrote "halves of J" where the splits halve O*; fixed before freeze. (d) §8's shared-document premise was wrong for these items. (e) `GRID9.md` prints the restricted read's 2-case and 1-case bootstraps as readings; they are not.

**F5. Errata of this review's first version (journal 03:05 CST).** "the two-neuron null is reproduced from the same bank" (asserted from SPEC4); "the rule had room both ways"; "s_swap low for that reason as much as any"; "one more capture-only load" for the same-pole null (offline); "the restricted read had no cases" (2 and 1); "the overlap is specific to the two clusters" without the silent-contexts qualifier; the journal's "611 s" (624 s by the run record) and "381 files" (379).

## 3. Disposition

H_weights refuted on both sets under the frozen rule: the shared-driver overlap is specific to the two clusters against other neurons' contexts on which the pole is silent. Against the pole's own active clusters it is not specific (exploratory, offline): a neuron's tasks share about an eighth of their drivers pairwise. The ablation version of that comparison (J's dependence on the pole's other clusters' drivers against its dependence on O*'s) is the next pre-registration (`NEXT.md` item 9).
