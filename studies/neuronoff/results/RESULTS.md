# NEURONOFF — results

The frozen rules give no Arm A answer: the SPEC §6 positive control (G-pc) failed on 65 of 156 Q4B poles (41.7%)
and 75 of 202 G4 poles (37.1%), above the 30% line, so the Arm A labels of both models are cancelled
(`results/verdict_Q4B_v2.json → pc_cancel_share 0.4167`, `results/verdict_G4_v2.json → pc_cancel_share 0.3713`).

All paths below are relative to `studies/neuronoff/`. Banks: `bank/off/<cell>/1/out/` (untracked; S3
`s3://<s3-bucket>/results/neurons/off/<cell>/1/`). Freeze tag `neuronoff-freeze-1` = e5a5c33c. A
different model reproduced every number in the run brief from the raw banks (result review, 2026-10-25; file
`NEURONOFF_REVIEW.md` in the session scratchpad, not in git).

## 1. Headline (frozen)

| model | Arm A poles measured | G-pc failures | share | valid pairs | valid null pairs | own-context headline |
|---|---|---|---|---|---|---|
| Q4B | 156 of 212 | 65 | 0.4167 | 26 of 96 | 4 of 33 | CANCELLED: instrument not shown to fire; underpowered after gates |
| G4 | 202 of 217 | 75 | 0.3713 | 40 of 97 | 9 of 38 | CANCELLED: instrument not shown to fire; underpowered after gates |

Source: `results/verdict_<M>_v2.json → pc_cancel_share, n_valid, n_null_valid, headline_own`. The collateral,
energy-normalized and every secondary row are cancelled by the same rule. The study label is MIXED because no
model counts (`results/grid_v2.md`). O7 was not run (§8).

Every model gate passed in both models, and every registered failing input failed
(`results/{q4b,g4}/1/controls.json`; handoff §1). G-known reproduced the banked mean NLL within 2.2e-5 nats (Q4B)
and 5.0e-5 nats (G4). The cancellation is the positive-control rule, not the hook.

The numbers under the cancelled label, from the valid pairs (median D, 95% bootstrap interval, nats;
`results/verdict_<M>_v2.json → stats.own`): Q4B −0.000101 [−0.004477, 0.010809], n 26, δ 0.008075; G4 −0.0000348
[−0.008433, 0.005257], n 40, δ 0.013030 (`results/thresholds_<M>.json → arm_a.delta_own`).

## 2. Arm B (seven code-construct poles; R1 = summed log-probability of the written continuation, Δ in nats)

| model | pole (label) | Δ_pole | Δ_ctrl1 | Δ_layer | m_B* | D mean [95%] | frozen verdict | label in `_v2` (after review) | construct present, clean / pole / ctrl1 (of 400) | clean = ablated, token-identical (of 400) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4B | L11N5238+ (SQL FOREIGN KEY) | −0.00362 | −0.00325 | 0.538 | 0.0325 | −0.000365 [−0.0509, 0.0533] | NOT-SPECIAL | NOT-SPECIAL, interval crosses margin | 389 / 388 / 388 | 256 |
| Q4B | L27N2924- (Angular) | 0.0409 | −0.00399 | 1.008 | 0.0472 | 0.0449 [−0.0514, 0.1462] | NOT-SPECIAL | NOT-SPECIAL, interval crosses margin | 358 / 352 / 356 | 89 |
| Q4B | L08N3398- (HTML title attribute) | 0.2158 | 0.0522 | 1.342 | 0.0700 | 0.1636 [−0.0363, 0.4128] | UNRESOLVED | UNRESOLVED, interval crosses margin | 304 / 303 / 303 | 132 |
| Q4B | L04N5165+ (TeX) | −0.01284 | −0.02584 | 0.806 | 0.0639 | 0.0130 [−0.0275, 0.0522] | NOT-SPECIAL | instrument not shown to fire | mean matches 9.84 / 9.905 / 9.8675 | 180 |
| G4 | L07N10202- (React useEffect) | 0.00984 | 0.00575 | 0.816 | 0.0380 | 0.00408 [−0.0159, 0.0244] | NOT-SPECIAL | NOT-SPECIAL, interval below margin | 321 / 321 / 320 | 145 |
| G4 | L06N599+ (Python import … as) | 0.000945 | 0.000259 | 0.123 | 0.0200 | 0.000686 [−0.00175, 0.00433] | NOT-SPECIAL | NOT-SPECIAL, interval below margin | 391 / 391 / 391 | 153 |
| G4 | L08N7823+ (SQL REFERENCES) | −0.000965 | −0.00406 | 0.389 | 0.0384 | 0.00309 [−0.0247, 0.0324] | NOT-SPECIAL | NOT-SPECIAL, interval below margin | 394 / 394 / 394 | 287 |

Sources: `results/verdict_<M>_v2.json → arm_b.<pole>.readouts.R1, m_B_star, pc_mean_drop, verdict` (numbers equal
the frozen `results/verdict_<M>.json`, whose `verdict` field holds the frozen-verdict column);
`results/armb_counts_<M>_v2.json → poles.<pole>.table`; token-identical counts from
`results/<cell>/1/arm_b/<stem>_samples.jsonl` (same prompt, seed and sample index, clean against ablate_pole).

Labels after review (`results/verdict_<M>_v2.json → arm_b.<pole>.verdict, interval_vs_margin,
pole_activation_max_frac`; code-only deviation D1, same numbers as the frozen verdicts):
- "interval crosses margin": the frozen rule decides NOT-SPECIAL on the mean alone (mean D ≤ m_B*). The upper end of
  the interval is above m_B* for L11N5238+ (0.0533 > 0.0325) and L27N2924- (0.1462 > 0.0472); L08N3398- reads
  "UNRESOLVED, interval crosses margin" in `_v2` (0.4128 > 0.0700). For the three G4 poles the whole interval is
  below m_B* ("interval below margin").
- TeX L04N5165+: at every scored position of all 20 prompts the pole's clean activation stays at or below 0.0525 of
  its banked maximum (max 0.412 against 7.857; `results/q4b/1/arm_b/L04N5165p_teacher.jsonl → pole_a_inputs`,
  `plan/plan_Q4B.json → poles.L04N5165+.banked_max_abs`; `pole_activation_max_frac 0.0525`). R3 had no usable prompt
  (0 of 19). The ablation was not tested where the pole fires. `_v2` prints "instrument not shown to fire (pole
  activation at most 0.1 of its banked maximum at every scored position)"; the 0.1 line is D1's, set after review.
- L08N3398-'s positive mean D comes from prompts 10 and 11, where Δ_pole is 1.323 and 1.722 nats.

Behaviour changed (§5.4): 0 of 7 (`results/armb_counts_<M>_v2.json → behaviour_changed`).

React useEffect (G4 L07N10202-), prompt 12 ("Infinite loop when fetching data with hooks …"), seed 20273025,
sample 9 (`results/g4/1/arm_b/L07N10202m_samples.jsonl`). Clean:

````
useEffect(() => {
    async function fetchUser() {
        const response = await axios.get(`https://api.example.com/user/${props.username}`);
        setUser(response.data);
    }
    fetchUser();
}, [props.username]);
```
````

ablate_pole:

````
useEffect(() => {
    async function fetchUser() {
        const response = await axios.get(`https://api.example.com/user/${id}`);
        setUser(response.data);
    }
    fetchUser();
}, [id]);
```
````

The two differ only at `props.username` against `id`. 145 of the 400 clean/ablated pairs for this pole are
token-identical.

## 3. Absolute effect sizes (own-context ΔNLL of one pole's ablation, nats)

Median [interquartile range] of own_dnll per pole, per side, recomputed from `bank/off/<cell>/1/out/poles/*.json →
own.contexts[].dnll`:

| model | pair set | n pairs | ONE side | UNK side |
|---|---|---|---|---|
| Q4B | frozen valid pairs | 26 | 0.0093 [0.0010, 0.0273] | 0.0052 [−0.0010, 0.0302] |
| G4 | frozen valid pairs | 40 | 0.0079 [−0.0034, 0.0234] | 0.0072 [−0.0024, 0.0204] |
| Q4B | E1 valid pairs | 70 | 0.0032 [−0.0020, 0.0155] | 0.0026 [−0.0028, 0.0108] |
| G4 | E1 valid pairs | 89 | 0.0043 [−0.0027, 0.0167] | 0.0048 [−0.0025, 0.0114] |

Per held-out context, over every measured Arm A pole: the median ΔNLL is 0.000037 nats (Q4B, 15,600 contexts) and
0.000018 nats (G4, 20,200 contexts). 37.9% (Q4B) and 39.8% (G4) of contexts move by less than 0.01 nats in absolute
value. The largest single-context ΔNLL is 5.685 (Q4B) and 7.247 (G4). The reviewer's rounding of the same
quantities over its pair sets: median ≤ 0.00005, 34–41% below 0.01, maximum 7.25.

## 4. Exploratory analyses (chosen after unblinding, never the headline)

### E1 — pooled positive control (chosen after unblinding, never the headline)

Rule (`infra/pg/neuron_off_exploratory.py` docstring): the per-pole G-pc gate is replaced by one per-model pooled
control, the mean whole-layer ΔNLL over every banked positive-control position (floor 0.05 nats; failing input =
identity readout; ceiling = the known-number layer-0 shift). Reading (director): the null pairs are re-gated
the same way (`regated_nulls`); the frozen-nulls block is kept for transparency.

| model | pooled control mean [95%] (positions) | valid pairs | valid null pairs | δ (re-gated) | M [95%] | label (re-gated) | frozen-nulls label |
|---|---|---|---|---|---|---|---|
| Q4B | 0.2293 [0.1816, 0.2819] (1,560) | 70 | 24 | 0.00747 | 0.00175 [−0.00202, 0.00553] | H_morcos | CANCELLED: underpowered after gates (4 null pairs) |
| G4 | 0.2364 [0.1973, 0.2773] (2,020) | 89 | 34 | 0.01109 | 0.000234 [−0.00348, 0.00335] | H_morcos | CANCELLED: underpowered after gates (9 null pairs) |

Source: `results/<cell>/1/exploratory_E1_v2.json → pooled_positive_control, regated_nulls.result, result`.

Caveats (reviewer):
- δ is larger than the signed median own-context ΔNLL of either side on the E1 pairs (δ 0.00747 and 0.01109
  against side medians 0.0026–0.0048, table §3). Against the median |ΔNLL| (E2: 0.0059–0.0086), δ is larger in G4
  (0.01109 against 0.0086 and 0.0073) and about equal in Q4B (0.00747 against 0.0079 and 0.0059). The upper ends of
  the E1 intervals (0.00553, 0.00335) are about one typical pole effect, so an interval inside ±δ still allows a
  ONE–UNK difference of that size.
- The pooled control passes on its tail: the top 5% of positions carry 73% (Q4B) and 66% (G4) of the sum. Its
  failing input is exactly 0 by construction.
- With the frozen null pairs, E1 is cancelled by G-n.

### E2 — relative effect and absolute effect per side (chosen after unblinding, never the headline)

Rule (`results/<cell>/1/exploratory_E2.json → rule`): over the valid E1 pairs, the median of the relative difference
D_k / mean(|own_one_k|, |own_ctrl_k|) with a 95% percentile bootstrap interval over pairs, and per side the median,
interquartile range and median |·| of own_dnll.

| model | valid E1 pairs | relative difference, median [95%] | ONE own_dnll median [IQR] | UNK own_dnll median [IQR] | median \|ΔNLL\| ONE / UNK |
|---|---|---|---|---|---|
| Q4B | 70 | +0.255 [−0.489, +0.676] | 0.0032 [−0.0020, 0.0155] | 0.0026 [−0.0028, 0.0108] | 0.0079 / 0.0059 |
| G4 | 89 | +0.083 [−0.490, +0.654] | 0.0043 [−0.0027, 0.0167] | 0.0048 [−0.0025, 0.0114] | 0.0086 / 0.0073 |

Source: `results/q4b/1/exploratory_E2.json`, `results/g4/1/exploratory_E2.json → relative_D, own_dnll_one,
own_dnll_ctrl` (no pair had a zero denominator).

Meaning, in one sentence: for a pair whose two effects are both positive, the interval runs from a ONE pole
mattering about 0.6 times as much as its control to about 2 times as much (relative difference r maps to a ratio
(1 + r/2)/(1 − r/2)), so at the relative scale the study cannot distinguish the hypotheses.

## 5. Missed predictions (SPEC §5.6, §11; predicted 2026-10-25 before any GPU run, `JOURNAL.md` PREDICTIONS)

| prediction | predicted | observed | artifact |
|---|---|---|---|
| Arm B SPECIAL count | ≥ 3 of 7 (p 0.6) | 0 of 7 | `results/verdict_<M>_v2.json → arm_b.*.verdict` |
| G-pc pass rate | ≥ 90% of poles per model (p 0.7) | 58.3% (Q4B), 62.9% (G4) | `results/verdict_<M>_v2.json → pc_cancel_share` |
| own-context M | > 0 | −0.000101 (Q4B), −0.0000348 (G4) | `results/verdict_<M>_v2.json → M` |

Landed: behaviour changed 0 of 7 (predicted 0 of 7, p 0.8); absolute own_dnll medians below 0.05 nats on both
sides (table §3).

## 6. Three dimensions

- Predictive: the decisive pre-registered predictions missed (§5).
- Novel: small single-neuron ablation effects (Gurnee et al. 2023) and explained neurons no more causal than
  random ones (Huang et al. 2023) are published (`LIT.md`); this study's only new contrast, a matched
  passed-versus-failed explanation with single-pole ablation, rests on exploratory E1 alone.
- Practical: none.

## 7. Instrument lesson for the next spec

The per-pole G-pc floor of 0.05 nats sits above the per-context median whole-layer effect: 0.0139 nats (Q4B) and
0.0207 nats (G4) over all 10-context positive-control readouts (`bank/off/<cell>/1/out/poles/*.json →
positive_control.contexts[].dnll`). So the positive control failed, not the hook: G-ident is 0 on every pole and
G-floor reads 0 on every layer. A positive control must be checked for room to pass, as well as for room to fail.

## 8. Budget actually spent

2.794 instance-hours, USD 6.44 (EC2 on-demand list price; `handoff/2026-10-25_run.md` §6; `JOURNAL.md`). The
frozen L4 rule dropped O7 and cut each remaining model to the first 70 pairs. Q4B measured 70 of 96 pairs; G4 measured
90 of 97, because pair 70's null partner sat at Arm A index 202. The rule used R = 2,291 tokens/s, a throughput
dominated by Arm B sampling. Arm A alone ran at 13,355 tokens/s in the preflight (`results/pre_q4b/1/manifest.json
→ stage_s`; `JOURNAL.md` L4 entry).
