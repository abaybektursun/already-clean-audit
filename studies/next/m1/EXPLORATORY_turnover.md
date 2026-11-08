Dated exploratory analysis, 2026-11-05, on the closed and unblinded M1 result (`verdict_m1.json`, `READ_head.md`). No rule here decides a headline.

# M1 pole turnover: loss at fixed pole identity, or movement between poles?

**Answer.** Under G (0 % code+math), code and math leave poles without arriving at other poles: G's arrival fraction is
0.165, 0.165, 0.085, 0.090 by band, the lowest of the nine pairs that span training (G, R, M and six stage-1 pairs), and
0.072 to 0.198 below M's in every band (paired 95 % intervals exclude zero). The frozen rule reads band 0 "within drift" and bands 1–3 "loss at fixed identity". Its
"loss at fixed identity" label also fires on a pure stage-1 pair (start → t15, planted input P1), so the label alone does
not separate a mix effect from a longer span of stage-1 training; the comparison against M and R at the same start does.
All numbers below are from `studies/next/m1/turnover/turnover.json` unless another file is named.

## The question

M1's band means (C_X(b), the mean of cm_share over the 176,128 poles of a band) fall by 0.011 to 0.017 from the start
to G. A band mean cannot tell two cases apart. (a) Loss at fixed identity: the poles that held code and math lose it, and
no other pole gains it. (b) Movement between poles: code and math leave some poles and arrive at others of the same band,
and the mean falls by the difference. This is the relocation threat of `studies/next/12_SHAPING_NOTE.md` §3, tested here
as its §4 item 1.

## Objects

- cm_share(p) = slots of pole p's top-50 positions in the five code+math directories / filled slots, exactly as in
  `read_head/recount.py`. Unit: a fraction of 50 slots. Every one of the ten count files passes
  `counts.sum(-1) == filled` (asserted in `turnover/rule.py`, `share_from_counts`).
- The code reproduces two known numbers before any new one: C_G − C_start = −0.0113, −0.0170, −0.0173, −0.0161 (the
  journal's `exploratory.shift_from_start` in `verdict_m1.json`) and C_G − C_M = −0.0109, −0.0100, −0.0174, −0.0196
  (the verdict's Δ_MG with the sign reversed).
- A pole is "high" at X if cm_share ≥ τ_b at X. Leavers (X → Y) = poles high at X and not high at Y; arrivals = poles
  not high at X and high at Y; both divided by the number of poles high at X.
- Interval: 95 % bootstrap over the 88,064 neurons of a band, the two poles of a neuron resampled together, 2,000
  resamples, numpy `default_rng(0)` created once per call, so every pair uses the same resamples and differences between
  pairs are paired.

## The frozen rule (`turnover/rule.json`, written by `turnover/rule.py`, committed in e170384ca before the mixes were read)

- τ_b = the 90th percentile of cm_share at t15 over the band's poles: **0.44, 0.30, 0.20, 0.12**. Because shares are
  multiples of 1/50, 0.1049, 0.1046, 0.1042, 0.1133 of t15's poles sit at or above τ.
- Floor = the largest point value over the stage-1 neighbour pairs t12 → t13, t13 → t14b, t14b → t15 and the identity
  pair t14b → t14c, per band. **Leavers: 0.303, 0.322, 0.252, 0.278. Arrivals: 0.353, 0.380, 0.271, 0.285.** The
  three neighbour pairs span 240,000, 440,000 and 413,814 stage-1 steps (1.0T, 1.85T, 1.74T tokens, `M1_SPEC.md` §3.2;
  each mix is 100B tokens); t14b → t14c is one checkpoint harvested on two
  devices (0.006 to 0.015 on both sides).
- Ceiling: the value if pole identity were scrambled within the band. For G it is 0.92, 0.91, 0.92, 0.92 for leavers
  and 0.80, 0.72, 0.65, 0.60 for arrivals (`mixes` → `shuffle_leavers`, `shuffle_arrivals`).
- Side label: ABOVE if the interval's lower end exceeds the floor, WITHIN if its upper end is at or below the floor,
  otherwise UNDECIDED. Reading: leavers ABOVE and arrivals WITHIN → "loss at fixed identity"; both ABOVE → "movement
  between poles"; both WITHIN → "within drift"; leavers WITHIN and arrivals ABOVE → "arrivals only"; any side UNDECIDED
  → "undecided".
- The readings do not count if the identity pair start_a → start_b has any leaver or arrival. It has none in any band
  (the two count files are byte-identical), so they count.
- The brief names t14b → t15 as the third neighbour pair; `M1_SPEC.md` §3.2 used t14c → t15. The two give the same
  values to 0.001 (`rule.json` → `per_pair_point_values`).

## Leavers and arrivals per band

The stage-1 pairs set the floor, so their intervals often straddle it and read "undecided"; they are shown for scale,
not as tests.

**Band 0 (layers 0-7).** τ = 0.44; floor: leavers 0.303, arrivals 0.353.

| pair (X = start_a unless named) | leavers [95 %] | arrivals [95 %] | poles ≥ τ at X → at Y | reading |
|---|---|---|---|---|
| G (0 % code+math) | 0.289 [0.282, 0.296] | 0.165 [0.158, 0.172] | 15,599 → 13,663 | within drift |
| R (40 %) | 0.252 [0.246, 0.260] | 0.212 [0.204, 0.220] | 15,599 → 14,966 | within drift |
| M (65 %) | 0.265 [0.258, 0.272] | 0.245 [0.236, 0.254] | 15,599 → 15,288 | within drift |
| identity: start_a -> start_b | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 15,599 → 15,599 | within drift |
| identity: t14b -> t14c (two devices) | 0.006 [0.005, 0.007] | 0.006 [0.005, 0.008] | 17,339 → 17,343 | within drift |
| stage 1: t12 -> t13 | 0.303 [0.295, 0.311] | 0.348 [0.337, 0.359] | 15,315 → 16,008 | undecided |
| stage 1: t13 -> t14b | 0.269 [0.262, 0.276] | 0.353 [0.342, 0.364] | 16,008 → 17,339 | undecided |
| stage 1: t14b -> t15 | 0.231 [0.224, 0.237] | 0.297 [0.287, 0.306] | 17,339 → 18,482 | within drift |
| planted P1: G := t15 | 0.299 [0.292, 0.306] | 0.484 [0.470, 0.497] | 15,599 → 18,482 | undecided |
| planted P5: G := t13 | 0.232 [0.226, 0.239] | 0.258 [0.249, 0.267] | 15,599 → 16,008 | within drift |
| planted P2: G-sized loss, no drift | 0.114 [0.109, 0.119] | 0.000 [0.000, 0.000] | 15,599 → 13,822 | within drift |
| planted P3: half of the high set zeroed | 0.500 [0.493, 0.508] | 0.000 [0.000, 0.000] | 15,599 → 7,800 | loss at fixed identity |
| planted P4: half of the high set swapped | 0.500 [0.492, 0.508] | 0.500 [0.486, 0.515] | 15,599 → 15,599 | movement between poles |

**Band 1 (layers 8-15).** τ = 0.3; floor: leavers 0.322, arrivals 0.380.

| pair (X = start_a unless named) | leavers [95 %] | arrivals [95 %] | poles ≥ τ at X → at Y | reading |
|---|---|---|---|---|
| G (0 % code+math) | 0.357 [0.350, 0.364] | 0.165 [0.159, 0.171] | 19,632 → 15,858 | loss at fixed identity |
| R (40 %) | 0.319 [0.312, 0.326] | 0.198 [0.191, 0.205] | 19,632 → 17,260 | undecided |
| M (65 %) | 0.324 [0.317, 0.330] | 0.237 [0.229, 0.244] | 19,632 → 17,928 | undecided |
| identity: start_a -> start_b | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 19,632 → 19,632 | within drift |
| identity: t14b -> t14c (two devices) | 0.015 [0.013, 0.017] | 0.014 [0.012, 0.015] | 19,408 → 19,385 | within drift |
| stage 1: t12 -> t13 | 0.307 [0.300, 0.314] | 0.380 [0.370, 0.390] | 19,282 → 20,678 | undecided |
| stage 1: t13 -> t14b | 0.322 [0.316, 0.329] | 0.261 [0.253, 0.269] | 20,678 → 19,408 | undecided |
| stage 1: t14b -> t15 | 0.256 [0.250, 0.263] | 0.206 [0.199, 0.213] | 19,408 → 18,423 | within drift |
| planted P1: G := t15 | 0.366 [0.359, 0.374] | 0.305 [0.296, 0.314] | 19,632 → 18,423 | loss at fixed identity |
| planted P5: G := t13 | 0.242 [0.236, 0.249] | 0.296 [0.287, 0.304] | 19,632 → 20,678 | within drift |
| planted P2: G-sized loss, no drift | 0.186 [0.180, 0.191] | 0.000 [0.000, 0.000] | 19,632 → 15,983 | within drift |
| planted P3: half of the high set zeroed | 0.500 [0.493, 0.507] | 0.000 [0.000, 0.000] | 19,632 → 9,816 | loss at fixed identity |
| planted P4: half of the high set swapped | 0.500 [0.493, 0.507] | 0.500 [0.487, 0.513] | 19,632 → 19,632 | movement between poles |

**Band 2 (layers 16-23).** τ = 0.2; floor: leavers 0.252, arrivals 0.271.

| pair (X = start_a unless named) | leavers [95 %] | arrivals [95 %] | poles ≥ τ at X → at Y | reading |
|---|---|---|---|---|
| G (0 % code+math) | 0.357 [0.350, 0.364] | 0.085 [0.081, 0.090] | 19,191 → 13,978 | loss at fixed identity |
| R (40 %) | 0.255 [0.249, 0.261] | 0.155 [0.149, 0.162] | 19,191 → 17,277 | undecided |
| M (65 %) | 0.230 [0.224, 0.236] | 0.230 [0.222, 0.238] | 19,191 → 19,185 | within drift |
| identity: start_a -> start_b | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 19,191 → 19,191 | within drift |
| identity: t14b -> t14c (two devices) | 0.012 [0.011, 0.014] | 0.012 [0.010, 0.013] | 18,720 → 18,712 | within drift |
| stage 1: t12 -> t13 | 0.252 [0.246, 0.258] | 0.271 [0.263, 0.280] | 19,075 → 19,449 | undecided |
| stage 1: t13 -> t14b | 0.240 [0.234, 0.246] | 0.203 [0.196, 0.209] | 19,449 → 18,720 | within drift |
| stage 1: t14b -> t15 | 0.187 [0.181, 0.193] | 0.168 [0.162, 0.174] | 18,720 → 18,356 | within drift |
| planted P1: G := t15 | 0.282 [0.275, 0.288] | 0.238 [0.231, 0.246] | 19,191 → 18,356 | loss at fixed identity |
| planted P5: G := t13 | 0.190 [0.184, 0.195] | 0.203 [0.197, 0.210] | 19,191 → 19,449 | within drift |
| planted P2: G-sized loss, no drift | 0.206 [0.200, 0.212] | 0.000 [0.000, 0.000] | 19,191 → 15,240 | within drift |
| planted P3: half of the high set zeroed | 0.500 [0.493, 0.507] | 0.000 [0.000, 0.000] | 19,191 → 9,596 | loss at fixed identity |
| planted P4: half of the high set swapped | 0.500 [0.493, 0.507] | 0.500 [0.488, 0.514] | 19,191 → 19,191 | movement between poles |

**Band 3 (layers 24-31).** τ = 0.12; floor: leavers 0.278, arrivals 0.285.

| pair (X = start_a unless named) | leavers [95 %] | arrivals [95 %] | poles ≥ τ at X → at Y | reading |
|---|---|---|---|---|
| G (0 % code+math) | 0.402 [0.395, 0.408] | 0.090 [0.086, 0.094] | 21,689 → 14,921 | loss at fixed identity |
| R (40 %) | 0.248 [0.242, 0.254] | 0.207 [0.200, 0.214] | 21,689 → 20,794 | within drift |
| M (65 %) | 0.230 [0.224, 0.236] | 0.288 [0.280, 0.295] | 21,689 → 22,935 | undecided |
| identity: start_a -> start_b | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 21,689 → 21,689 | within drift |
| identity: t14b -> t14c (two devices) | 0.012 [0.011, 0.014] | 0.013 [0.011, 0.014] | 20,809 → 20,824 | within drift |
| stage 1: t12 -> t13 | 0.264 [0.258, 0.270] | 0.285 [0.277, 0.293] | 21,827 → 22,282 | undecided |
| stage 1: t13 -> t14b | 0.278 [0.272, 0.284] | 0.212 [0.206, 0.219] | 22,282 → 20,809 | undecided |
| stage 1: t14b -> t15 | 0.222 [0.217, 0.228] | 0.182 [0.175, 0.188] | 20,809 → 19,960 | within drift |
| planted P1: G := t15 | 0.326 [0.319, 0.332] | 0.246 [0.239, 0.253] | 21,689 → 19,960 | loss at fixed identity |
| planted P5: G := t13 | 0.207 [0.202, 0.213] | 0.234 [0.227, 0.241] | 21,689 → 22,282 | within drift |
| planted P2: G-sized loss, no drift | 0.223 [0.217, 0.228] | 0.000 [0.000, 0.000] | 21,689 → 16,854 | within drift |
| planted P3: half of the high set zeroed | 0.500 [0.494, 0.507] | 0.000 [0.000, 0.000] | 21,689 → 10,845 | loss at fixed identity |
| planted P4: half of the high set swapped | 0.500 [0.493, 0.507] | 0.500 [0.488, 0.513] | 21,689 → 21,689 | movement between poles |


G against the controls at the same start and the same stage-2 length, paired (`paired_differences`):

| band | G − M leavers [95 %] | G − M arrivals [95 %] | G − R leavers [95 %] | G − R arrivals [95 %] |
|---|---|---|---|---|
| 0 | +0.024 [+0.016, +0.032] | -0.080 [-0.089, -0.071] | +0.036 [+0.029, +0.043] | -0.047 [-0.055, -0.039] |
| 1 | +0.034 [+0.027, +0.041] | -0.072 [-0.079, -0.064] | +0.038 [+0.032, +0.045] | -0.033 [-0.040, -0.026] |
| 2 | +0.127 [+0.120, +0.134] | -0.144 [-0.152, -0.137] | +0.102 [+0.095, +0.109] | -0.070 [-0.076, -0.065] |
| 3 | +0.172 [+0.165, +0.178] | -0.198 [-0.205, -0.190] | +0.153 [+0.148, +0.160] | -0.117 [-0.124, -0.111] |

## Planted inputs (`planted`)

Each replaces G's counts and runs through the same code against start_a. The expectations were written into
`turnover.py` before its first run.

| input | what it is | expected | result by band | as expected |
|---|---|---|---|---|
| P1: G := t15 | stage-1 training from step 476,838 to 1,413,814 (936,976 steps, about twice the longest floor pair) | "movement between poles" in every band | undecided; loss at fixed identity ×3 | **no** |
| P5: G := t13 | stage-1, 83,162 steps, shorter than every floor pair | "within drift" | within drift ×4 | yes |
| P2: G-sized loss, no drift | start_a with each code+math slot removed with probability r_b = 0.091, 0.154, 0.228, 0.294, so the band mean falls by G's amount (realized −0.0112, −0.0170, −0.0174, −0.0160 against G's −0.0113, −0.0170, −0.0173, −0.0161) | arrivals 0; leavers: the power check, no expectation | within drift ×4 (leavers 0.114, 0.186, 0.206, 0.223) | — |
| P3: half of the high set zeroed | a random half of the start's high poles lose every code+math slot | "loss at fixed identity" | loss at fixed identity ×4 | yes |
| P4: half of the high set swapped | a random half of the high poles swap counts with as many random low poles of the band; band means unchanged (Δ = 0.0000) | "movement between poles" | movement between poles ×4 | yes |

What the planted inputs show:

- The rule returns each of its three main readings when the input calls for it (P3, P4, P5), and the identity pair
  returns exactly 0.
- P1 landed elsewhere than expected. From the start to t15 the band means change by +0.0185, −0.0062, −0.0018, −0.0034
  (`planted` → `P1_G_is_t15` → `delta`). Where the mean falls (bands 1–3), stage-1 training over a span about twice the
  floor pairs' gives leavers above the floor and arrivals within it. So the "loss at fixed identity" label is set partly
  by the length of the span, and the floor pairs (at most 440,000 steps) do not bound a longer one.
- P2 is the power check. A loss of exactly G's size at fixed identity, with no other change, gives leavers 0.114 to
  0.223, all within the floor. The leaver side of the rule cannot detect a G-sized loss on its own. G's leavers exceed
  the floor because stage-2 training also turns poles over: M alone has leavers 0.230 to 0.324.
- A first version of P2 rounded each pole's thinned count to the nearest integer and realized only 71–93 % of G's
  band-mean change; it was replaced by the binomial thinning above before this document was written.

## Decomposition of the band-mean change

Δ = C_Y − C_X = (part from poles high at X) + (part from the rest), each part summed over its poles and divided by the
band's 176,128 poles. The two right columns are the mean change of a pole in each part (`high_mean`, `rest_mean`).


| band | pair | Δ (band mean) | part from poles ≥ τ at X [95 %] | part from the rest [95 %] | mean change of a pole ≥ τ | mean change of a pole < τ |
|---|---|---|---|---|---|---|
| 0 | G | -0.0113 | -0.0097 [-0.0100, -0.0094] | -0.0016 [-0.0019, -0.0012] | -0.1098 | -0.0017 |
| 0 | R | -0.0036 | -0.0071 [-0.0074, -0.0068] | +0.0035 [+0.0031, +0.0038] | -0.0798 | +0.0038 |
| 0 | M | -0.0004 | -0.0076 [-0.0079, -0.0073] | +0.0072 [+0.0068, +0.0076] | -0.0859 | +0.0079 |
| 0 | t12 → t13 | +0.0041 | -0.0097 [-0.0101, -0.0094] | +0.0138 [+0.0133, +0.0143] | -0.1121 | +0.0152 |
| 0 | t13 → t14b | +0.0082 | -0.0078 [-0.0081, -0.0074] | +0.0160 [+0.0155, +0.0165] | -0.0854 | +0.0176 |
| 0 | t14b → t15 | +0.0077 | -0.0063 [-0.0066, -0.0060] | +0.0140 [+0.0135, +0.0144] | -0.0641 | +0.0155 |
| 1 | G | -0.0170 | -0.0121 [-0.0124, -0.0117] | -0.0050 [-0.0052, -0.0047] | -0.1082 | -0.0056 |
| 1 | R | -0.0102 | -0.0096 [-0.0099, -0.0093] | -0.0006 [-0.0009, -0.0003] | -0.0861 | -0.0007 |
| 1 | M | -0.0070 | -0.0099 [-0.0103, -0.0096] | +0.0029 [+0.0026, +0.0032] | -0.0891 | +0.0033 |
| 1 | t12 → t13 | +0.0055 | -0.0076 [-0.0080, -0.0073] | +0.0132 [+0.0128, +0.0136] | -0.0697 | +0.0148 |
| 1 | t13 → t14b | -0.0066 | -0.0100 [-0.0103, -0.0096] | +0.0034 [+0.0031, +0.0038] | -0.0849 | +0.0039 |
| 1 | t14b → t15 | -0.0043 | -0.0061 [-0.0064, -0.0059] | +0.0018 [+0.0016, +0.0021] | -0.0556 | +0.0021 |
| 2 | G | -0.0173 | -0.0130 [-0.0133, -0.0127] | -0.0043 [-0.0045, -0.0042] | -0.1195 | -0.0048 |
| 2 | R | -0.0059 | -0.0062 [-0.0064, -0.0059] | +0.0003 [+0.0001, +0.0005] | -0.0567 | +0.0003 |
| 2 | M | +0.0001 | -0.0043 [-0.0045, -0.0040] | +0.0043 [+0.0041, +0.0045] | -0.0390 | +0.0049 |
| 2 | t12 → t13 | +0.0023 | -0.0038 [-0.0040, -0.0035] | +0.0060 [+0.0058, +0.0063] | -0.0347 | +0.0067 |
| 2 | t13 → t14b | -0.0018 | -0.0049 [-0.0051, -0.0047] | +0.0031 [+0.0029, +0.0033] | -0.0443 | +0.0034 |
| 2 | t14b → t15 | -0.0013 | -0.0033 [-0.0035, -0.0031] | +0.0019 [+0.0018, +0.0021] | -0.0308 | +0.0022 |
| 3 | G | -0.0161 | -0.0139 [-0.0143, -0.0136] | -0.0022 [-0.0023, -0.0021] | -0.1131 | -0.0025 |
| 3 | R | -0.0011 | -0.0037 [-0.0039, -0.0034] | +0.0025 [+0.0024, +0.0027] | -0.0298 | +0.0029 |
| 3 | M | +0.0035 | -0.0021 [-0.0023, -0.0018] | +0.0056 [+0.0054, +0.0058] | -0.0168 | +0.0064 |
| 3 | t12 → t13 | +0.0019 | -0.0035 [-0.0038, -0.0033] | +0.0054 [+0.0052, +0.0056] | -0.0286 | +0.0062 |
| 3 | t13 → t14b | -0.0035 | -0.0064 [-0.0066, -0.0061] | +0.0028 [+0.0027, +0.0030] | -0.0503 | +0.0032 |
| 3 | t14b → t15 | -0.0015 | -0.0033 [-0.0035, -0.0031] | +0.0018 [+0.0017, +0.0020] | -0.0280 | +0.0021 |

- Measured from the start, the high set carries 0.86, 0.71, 0.75, 0.86 of G's band-mean change
  (`high_part_fraction_of_delta`).
- That share includes regression toward the mean. In every pair, M and the stage-1 pairs included, the poles selected
  high at X fall (mean change −0.017 to −0.112 per pole) and the rest rise. Under G alone among the mixes and stage-1
  pairs, the rest falls in every band (−0.0017, −0.0056, −0.0048, −0.0025 per pole); under R it falls only in band 1
  (−0.0007). The rest also rises under start → t15 (planted P1), including bands 1–3 where that pair's band mean
  falls: +0.0056, +0.0049, +0.0043 (`planted` → `P1_G_is_t15` → `rest_mean`). This is what separates G from P1, which
  the rule's label does not.
- Against M at the same start and length (paired), the rest carries 0.81, 0.79, 0.50, 0.40 of the G − M difference
  (part from the rest −0.0088, −0.0079, −0.0086, −0.0078; part from the high set −0.0021, −0.0021, −0.0088, −0.0119;
  `paired_differences` → `G-M`). In layers 0–15 the mix effect sits mostly in the 89–91 % of poles below τ at the start; in layers
  16–31 it is split about evenly between the two parts.

## Plain reading

The reading is not "movement between poles". Movement would need arrivals to rise with leavers, and under G arrivals
fall instead: they are the lowest of the nine pairs that span training in every band and below M's by 0.072 to 0.198. Code and math
also do not move to poles below τ within the band: those poles lose code+math share under G and gain it under M and
under every stage-1 pair, start → t15 included. By the frozen rule band 0 is "within drift" (leavers 0.289 against a floor of 0.303) and bands
1–3 are "loss at fixed identity". Planted input P1 shows that this label also fires on stage-1 training over a long span
when the band mean falls, so it is not specific to the 0 % mix. The specific evidence is the paired comparison with M
and R: from the same start and over the same 100B tokens, G has more leavers (+0.024 to +0.172 against M) and fewer
arrivals (−0.072 to −0.198) in every band. In the neuron basis, at the resolution of top-50 document shares, G's drop is
a loss at fixed pole identity spread across the band, not a move of code and math to other poles of the band. About 60
to 71 % of the start's high poles stay high under G (1 − leavers), against a scrambled-identity level of 8 to 9 %.

## What this cannot see

- A move out of the neuron basis: code and math carried by directions that mix many neurons, where no single pole's
  top-50 changes. That needs activations, not top-50 counts.
- A move into ranks below 50: a pole whose code and math positions now sit at ranks 51 to 500 shows nothing here.
- What a pole responds to. The share says which documents hold a pole's 50 strongest positions, not the activation
  values or the token-level pattern.
- Stage-2 drift. The floor comes from stage-1 pairs of 240,000 to 440,000 steps; there is no second stage-2 run from the
  same start with the same mix, so the turnover that stage-2 training gives by chance is not measured. M and R are the
  nearest controls.
- Other bands: every band's mean falls under G, so no band gains what another loses, but a move between layers inside
  one band would only show as arrivals, which are low.

## Files

- `studies/next/m1/turnover/rule.py`, `rule.json`: the rule frozen on t12, t13, t14b, t14c, t15 (commit e170384ca).
- `studies/next/m1/turnover/turnover.py`, `turnover.json`: the mixes, the floor pairs with intervals, the paired
  differences, the decomposition and the planted inputs. Runtime about 20 s on the laptop, numpy only, 0 GPU-hours.
- Inputs: `studies/next/m1/banks/{start_a,start_b,G,R,M}/m1/counts.npz`, `studies/next/m1/banks/refs/{t12,t13,t14b,t14c,t15}/m1/counts.npz`.
