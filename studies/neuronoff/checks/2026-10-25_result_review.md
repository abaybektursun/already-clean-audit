# NEURONOFF result review, 2026-10-25

Every brief number reproduces from the raw banks; the frozen cancellation holds. Unstated in the summaries: one pole's ablation moves own-context NLL by a median 0.002–0.009 nats, and E1's δ exceeds that whole effect.

As of `neurons/off` b9c5d4f6 (E1 merged 35507ae4); SPEC unchanged since e5a5c33c.

## 1. Recomputed from raw banks (all equal the claim)

- G-pc failures 65/156 (Q4B), 75/202 (G4); 63% and 73% of failures negative. P5 shares 0.4167, 0.3713 (> 0.30); over planned poles 0.307, 0.346.
- δ: 0.008075 from 4 null pairs; 0.013030 from 9.
- M: Q4B −0.000101 [−0.004477, 0.010809] (26 pairs); G4 −0.0000348 [−0.008433, 0.005257] (40 pairs).
- E1, G4: pooled mean 0.236 (2,020 positions); M +0.00023 [−0.00348, 0.00335] (89 pairs); δ 0.01109; R-b hw 0.00365 < δ.
- E1, Q4B: pooled mean 0.229 (1,560); M +0.00175 [−0.00202, 0.00553] (70); δ 0.00747; R-b hw 0.00326 < δ.
- Both E1 intervals cover 0 inside ±δ: H_morcos follows from A1.
- useEffect: Δ_pole 0.00984, Δ_ctrl1 0.00575, Δ_layer 0.816, D 0.00408 [−0.0159, 0.0244], m_B* 0.0380, construct counts 321/321/320.
- Arm B: 6 NOT-SPECIAL, 1 UNRESOLVED, 0/7 behaviour changed. All handoff §4 cells match.
- E1 changed no frozen output (merge: new module, two new E1 files, one test-fixture line).

## 2. Gates: each pass value, then the registered failing input (Q4B ; G4)

- G-known: 2.2e-5 ; 5.0e-5. Failing input 1.852 ; 1.712.
- G-det: 0. Failing input 21.04 ; 12.05.
- G-ident: 0 on all 358 poles and all 140 Arm B prompts. Failing input 10.06 ; 0.906.
- G-floor: every layer reads 0. Failing input minimum 0.25 ; 0.297.
- G-fit: 0 ; 1 fail (L16N8452+). Failing input failed on every pole.
- G-pc: failing input = identity, 0 on every pole.
- G-bal: −0.055 ; +0.083. Failing input −0.525 ; +0.553.
- G-n: fails at 26/4 ; 40/9. Its failing input also fails.

## 3. Absolute own-context ΔNLL (nats; median [IQR])

- Q4B frozen pairs: ONE 0.0093 [0.0011, 0.0273]; UNK 0.0052 [−0.0010, 0.0302].
- G4 frozen pairs: ONE 0.0079 [−0.0035, 0.0234]; UNK 0.0072 [−0.0024, 0.0204].
- Q4B E1 pairs: ONE 0.0033 [−0.0020, 0.0155]; UNK 0.0026 [−0.0028, 0.0108].
- G4 E1 pairs: ONE 0.0043 [−0.0027, 0.0167]; UNK 0.0048 [−0.0025, 0.0114].

Per context: median ≤ 0.00005; 34–41% move < 0.01; maximum 7.25.

## 4. Samples

G4 L07N10202-, prompt 12, seed 20273025, k 9. Clean:
```
useEffect(() => {\n    async function fetchUser() {\n        const response = await axios.get(`https://api.example.com/user/${props.username}`);\n        setUser(response.data);\n    }\n    fetchUser();\n}, [props.username]);\n```\n
```
ablate_pole:
```
useEffect(() => {\n    async function fetchUser() {\n        const response = await axios.get(`https://api.example.com/user/${id}`);\n        setUser(response.data);\n    }\n    fetchUser();\n}, [id]);\n```\n
```
They differ only at `props.username`/`id`. 145/400 pairs are token-identical.

Q4B L08N3398-, prompt 4, seed 20265025, k 16. Clean and ablated are both `' frameborder="0" allowfullscreen></iframe>\n</div>'`. The text is identical but the token ids are not (12 tokens against 13).

## 5. Literature (four searches)

Queries: explanation score vs ablation; single-neuron ablation self-repair; monosemantic vs polysemantic ablation; per-neuron zero-ablation loss. Already published (per LIT.md; not re-read): small single-neuron effects (Gurnee et al. 2023, 0.2% per-sequence loss at 6.9B) and explained neurons no more causal than random (Huang et al. 2023, GPT-2 XL neuron sets). No paper found runs the matched passed-vs-failed single-pole contrast; that rests only on exploratory E1.

## 6. Findings

1. **MUST-FIX-PROSE.** E1's H_morcos uses δ (0.0111 and 0.0075), which is larger than the median absolute effect of either pole (0.0026–0.0048). The interval's upper ends are roughly one typical pole effect. Both the E1 rule and its `regated_nulls` reading were chosen after unblinding. With the frozen nulls, E1 is CANCELLED by G-n.
2. **MUST-FIX-PROSE.** Cancelled labels are printed without a stamp:
   - verdict secondary rows (G4 "H_morcos", Q4B "INCONCLUSIVE");
   - E1's frozen-nulls secondary rows ("H_morcos");
   - grid.md `MIXED | [] | | ['G4','Q4B']`, which reads as two counted models when zero counted.
   The cause is `subset_row` (neuron_off_analysis.py:432), which ignores `model_reasons`. The handoff's correction covers only the verdicts.
3. **MUST-FIX-PROSE.** NOT-SPECIAL is decided on the point estimate alone. Two intervals reach above m_B*: L11N5238+ (0.053 > 0.033) and L27N2924- (0.146 > 0.047). For the other four NOT-SPECIAL poles the whole interval is below m_B*.
4. **MUST-FIX-PROSE.** TeX L04N5165+ stays at or below 0.052 of its banked maximum at every scored position on all 20 prompts. Its NOT-SPECIAL does not test the pole.
5. **MUST-FIX-PROSE.** Missed predictions are not reported (§5.6):
   - Arm B SPECIAL: predicted ≥ 3/7, observed 0/7.
   - G-pc pass rate: predicted ≥ 90%, observed 58% and 63%.
   - M: predicted > 0, observed negative in both models.
6. **NOTE (spec amendment).** G-pc's 0.05 floor exceeds the per-context median of mid-layer whole-MLP ablation (0.014, 0.021); Q4B fails 54% at layers 12–23, 19% at 24+. The label describes the positive control, not the hook. G-pc cancels more UNK than ONE poles (Q4B 47% vs 36%).
7. **NOTE.** With 4 or 9 null values, R-b's half-width is a gap between adjacent null values, identical from n = 26 to 100; the frozen "room" stamps are uninformative.
8. **NOTE.** E1's pooled control passes because of its tail: the top 5% of positions carry 66–73% of the sum. Its failing input is exactly 0, so it cannot fail to fail. The quantity is the ΔNLL of the pole's own layer, not "layer-zero" as the brief says.
9. **NOTE.** The L08N3398- D comes from prompts 10 and 11, where Δ_pole is 1.32 and 1.72 nats.
10. **NOTE.** Handoff line 149 prints 9.90 where the value is 9.905. Line 106 says P5 cancels; it is §6 G-pc that cancels.

## 7. Own errors; next check

I suspected the identical hw95 values were a code bug; they are bootstrap discreteness. My E1 pooled intervals differ in the fourth decimal (position order).

Next check (banked JSON, no GPU): E1's equivalence relative to the pole effect, median D_k / mean(|own_one|, |own_ctrl|) with interval.
