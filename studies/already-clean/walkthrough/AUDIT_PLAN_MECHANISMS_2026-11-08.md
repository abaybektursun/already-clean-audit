# Audit of the public plan: mechanisms, numbers, data files, vocabulary, pages

Written 2026-11-08 by an Opus auditor in worktree `interp_research-audit-mech` (branch `audit-mech`, from
`clean-neurons` at `5cd349f61`). Object: `studies/already-clean/walkthrough/PLAN_PUBLIC_2026-11-08.md` (called PLAN
below; line numbers are PLAN's). This file reports; it does not rewrite the plan. Every claim names a file and a line or
a JSON key. Recounts made here are marked "recount (this audit)".

**Answer.** The plan's skeleton holds, but five of its twelve mechanism pictures (M4, M5, M6, M7, M11) describe a different
operation from the one that ran, M1 and M2 leave out OLMo-3's norm after each sublayer, and the three biggest errors
change what a reader learns. (1) The 16-bit failure was in the switch-off
measurement, not the gradient prediction. The 0.960 pairs the unchanged 16-bit prediction with a 32-bit switch-off
(`studies/next/n2v/verdict_n2v.json` → `test.window.validation`: `"prediction": "pred_bf16"`, `"measurement": "dL"`). M6
copies N6's per-token loss, not the switch-off of chapters 6 and 8. (2) M5's birth rule is wrong, and chapters 5 to 8 rest
on a filter that no picture shows: 153 passing births out of 2,428 tested groups. Chapter 5's "22,016 neurons" are 22,016
sides of 11,008 neurons. (3) Six chapter claims say more than their files: chapters 1, 4, 6 and 9, F8 and chapter 10.
Counts are in §7.

---

## 1. Mechanisms missing

Each row is an operation that a chapter's "what we did" needs and that no M-picture shows exactly. "Partial" means an
M-picture touches it but leaves out the part that decides the number.

| # | operation | chapter | defined in | the exact definition the picture should show |
|---|---|---|---|---|
| 1.1 | **A neuron's positive and negative sides** | 1, 4, 5, 7, 8, 10 | `07_PICKS/M2_SPEC.md` §0.5 lines 150, 156; `M1_SPEC.md` §0.5 lines 99–103; `N6_SPEC.md` §0.5 line 103 | a = silu(g)·u is signed. Side s = 0 is the tokens with a > 0, side s = 1 the tokens with a < 0. Each side's activation in its own direction is a⁺ = max(±a, 0). Each side has its own top 50 (largest a, or largest −a). A layer has 22,016 sides (11,008 × 2), the model 704,512. The two sides share one set of weights, so N1 and N13 measure neurons, not sides. N6 restricts its edit to the side's own tokens. Partial: M3 says "separately the fifty most negative" and nothing more. |
| 1.2 | **The fixed corpus** behind top contexts and births | 1, 4–8 | `studies/already-clean/SPEC_D1.md` §4.2 lines 365–395; `M1_SPEC.md` §0.2 (banked grid settings), §2.2 table | 50 directories of OLMo-3's own stage-1 mix (`allenai/dolma3_mix-6T-1025-7B`), picked at indices ⌊i·165/50⌋. From each, the documents that fill the first 400,000 tokens, deduplicated by text SHA-256 and joined with EOS. Then 10,044 windows of 2,048 tokens, each with BOS prepended. Window position 0 is excluded, so 10,044 × 2,047 = 20,560,068 scored tokens. |
| 1.3 | **The windows for the switch-off loss** | 6, 8 | `N2V_SPEC.md` §0.5 lines 100–102; `infra/aws_pg/jobs/n12_score.py` `ablate` lines 175–204 | A "unit window" is one of the 10,044 corpus windows above (BOS + 2,048 tokens) that holds at least one of the group's exemplar positions at input index 1 to 2,047. There are 987 unit windows over the 153 units (`N2V_SPEC.md` line 66). No other text is used. |
| 1.4 | **The windows for the edit loss** | 10 | `N6_SPEC.md` §0.5 lines 125–131 | 1,024 windows of 2,048 tokens from corpus band 3, outside the 20M corpus. Window index = 10,045 + ⌊i·14,633/1,024⌋, i = 0…1,023. That gives 2,097,152 targets. The loss is fp32, in nats per token (sha256 of the index list in line 127). |
| 1.5 | **The passing births** (the 153) | 5, 6, 7, 8 | `studies/next/m2/births.json` → `rule`; `studies/next/10_RECOUNT2.md` lines 144–150, 186–200 | For each group of a side with two or more groups, 2,000 random groups of the same size are drawn from the same side's 50 (the side's top-ranked context keeps the main group's label: R1t). Each is dated by the same rule. p_late = the share of random groups born at the same snapshot or later. A birth "passes" when p_late ≤ 0.05. That holds for 153 of 2,428 groups, against 22.4 at chance. Every carrier in chapters 5–8 is one of these 153 (82 in the first half, 71 held out). |
| 1.6 | **The split into the half that set the rules and the held-out half** | 5–8, 10 | `m2/births.json` → `split` ("neuron index even → freeze, odd → test"); `N6_SPEC.md` §2.3 "The split" | By neuron index: even goes to the rule-setting half, odd to the held-out half. In N6 the parity is that of the early member's neuron. The rules are written from the first half before the held-out half is opened. M12 does not show this. |
| 1.7 | **Score B and the carrier's rank in its layer** | 5 | `M2_SPEC.md` §0.5 lines 160–168; `infra/pg/m2.py` `percentile` lines 358–370 | v = the median of a⁺ over the group's 50-context positions at the snapshot before the birth. B = the share of the side's own-sign tokens at that snapshot whose \|a\| is below v, read from the side's 400-bin histogram with log-linear interpolation. Rank = the number of the layer's 22,016 sides with B ≥ the carrier's (ties count against it). A hit is rank ≤ 10. Chance is 10/22,016 = 0.00045. The matched-side floor is 0.0122 (§3.4 lines 374–379). |
| 1.8 | **The comparison group in chapter 5** (the plan's "stranger") | 5 | `M2_SPEC.md` §3.3 lines 358–372 | A random group of another grid side's own end-of-training 50 (echo contexts excluded). It has the birth's size and layer and is dated to the same snapshot k by the same rule. Candidate sides come in seeded order (`default_rng([20261105, l, n, s, c])`), up to 2,000 draws each, and the first group dated k is kept. 70 of 71 held-out births have one. |
| 1.9 | **The comparison neuron in chapter 7** ("dated alike") | 7 | `N1_SPEC.md` §3.3 lines 280–286; `N12_SPEC.md` §0.5 lines 112–119 | The same as 1.8, plus exactly the birth's number of distinct tokens d, where d = the distinct decoded token strings at the group's positions. 67 of 71 held-out births have one. It is matched on layer, group size, birth snapshot, the selection rule and d. |
| 1.10 | **The vocabulary-breadth measure** (F10's slider) | 5 | `N12_SPEC.md` §0.5 lines 118–119; `m2/read/r5b_token_share.json` | d = the number of distinct decoded tokens among the group's positions (from the grid's `samples/contexts.jsonl`). The split into repeated-token and varied-token births is post hoc (abstract line 36). |
| 1.11 | **The own-rise decomposition** ("the carrier wins by its own rise") | 8 | `N12_SPEC.md` §0.5 lines 126–129; `n12/READ_n12.md` lines 18–22 | r_c = the carrier's rank with its own value at the birth snapshot, among the others' values one snapshot earlier. r_o = its rank with its own earlier value, among the others' values at the birth snapshot. F = log10 r_o − log10 r_c. The dating rule forces B to rise in 442 of 442 units, twins included (READ_n12 line 19). The picture must show that twins rise the same way. |
| 1.12 | **The gradient signal that answers chapter 8** | 6, 8 | `N12_SPEC.md` §0.5 lines 120–124, 132–146 | G_n(i) = ∂L/∂h_n(i), where L is the window's summed next-token loss. Pull: s_q(i) = −sign_q · G_n(i) · q999_q. P = the mean over the group's positions. P′ = the same over 16 loss-matched positions per exemplar (same window, nearest loss, none within 8 tokens of a group position). S = P − P′. u = the share of the 20 competitors (the top 20 by B one snapshot earlier) at or above the carrier. M7 shows only the switch-off prediction, which is the check, not this signal. |
| 1.13 | **Rank agreement and its shuffle** (the 0.10 and the 0.96) | 6, 8 | `N2V_SPEC.md` §2.1 "Summary", §3.3–3.4 lines 324–366; `verdict_n2v.json` → `test.window` | Spearman ρ between prediction and measurement over the half's pairs (284 held out, 328 in the first half). The interval is a percentile bootstrap over units (1,000 resamples, seed 0). The shuffle reference is the 97.5th percentile of ρ under 1,000 shuffles of the prediction (0.118). Pass: ρ ≥ 0.7 and the lower bound above the shuffle value. A ceiling of 0.999 triggers an inspection. The room check adds each unit's measured rounding floor to a perfect instrument and needs median ρ ≥ 0.9. |
| 1.14 | **The weight-change statistic** of chapter 7 | 7 | `N1_SPEC.md` §0.5 lines 87–97; `infra/pg/m2.py` `paired` lines 546–553 | "in" = the gate row and the up row as one 8,192-vector; "out" = the 4,096-entry down column. r = ‖M(T_j) − M(T_(j−1))‖ / ‖M(T_(j−1))‖, in fp64. z = (log r − the layer's median) / (1.4826 × MAD) within each layer and bracket. e = the mean z over the three brackets k−2, k−1, k minus the mean z over brackets 2…k−3. The headline is the **mean** of real − comparison over 67 pairs, with a percentile bootstrap. |
| 1.15 | **The data-mix measurement** of chapter 4 | 4 | `M1_SPEC.md` §0.5 lines 99–113, §3.1–3.2 lines 229–251 | cm_share = the side's filled top-50 slots in the 5 code and math directories (`stack_edu-{CSharp,JavaScript,Python,SQL}`, `finemath-3plus`) / its filled slots. The summary is the mean over the 176,128 sides of a layer band (0–7, 8–15, 16–23, 24–31), paired by side. The drift bar comes from neighbouring stage-1 snapshots. The identity pair is the start snapshot harvested twice (0 of 704,512 sides differ). |
| 1.16 | **The snapshots outside the sixteen** | 4, 9 | `M1_SPEC.md` §0.2 lines 51–55; `M1_ASSETS.md` line 67; `studies/next/JOURNAL.md` line 18; `N13_SPEC.md` lines 282–284 | The three mixes start at `stage1-step476838` (2.0T tokens), between T_12 and T_13 and not one of the 16. They end at stage-2 step 47,684 (2,097,152 tokens per step = 100B). Chapter 9's base is `main` (= `stage3-step11921`, after stages 2 and 3), not T_15. The RL model is `Olmo-3.1-7B-RL-Zero-Math` at `364820b0`. |
| 1.17 | **How "most-changed" is ranked** | 9 | `N13_SPEC.md` §0.5 lines 189–192 | s_n = sqrt(‖Δg_n‖² + ‖Δu_n‖² + ‖Δd_n‖²) in float64, over the neuron's gate row, up row and down column. Ranked over all 352,256 MLP neurons of all 32 layers together; ties by (layer, index). Partial: M8 names "a selection" only. |
| 1.18 | **"Attention with norms"** (the plan's "attention only") | 9 | `N13_SPEC.md` §3.2 line 413; `infra/pg/n13_arms.py` lines 29–30, 501–502, 552 | From the RL model: q, k, v, o projections, and every norm weight: q_norm, k_norm, post_attention_layernorm, **post_feedforward_layernorm** and the final norm. Embeddings, LM head and all MLP matrices stay at the base. The post_feedforward_layernorm weights scale every MLP's output, so this condition is not "attention only". |
| 1.19 | **The complement** ("the model minus its most-changed neurons") | 9 | `N13_SPEC.md` §3.2 line 408 | Every tensor from RL (attention, norms, embeddings, LM head, MLP) except the rows and columns of the top-k neurons, which stay at the base. |
| 1.20 | **The 500-problem score** | 9 | `N13_SPEC.md` §2.1–2.4 lines 278–346; `studies/next/n13/thresholds.json` → `C` | `HuggingFaceH4/MATH-500` `test.jsonl` at commit `6e4ed1a2…`, in file order. The RL repo's chat template (prompt bytes at lines 304–306). vLLM 0.32.0 in bf16, one sample per problem, temperature 0.6, top-p 0.95, seed 1,000,000 + index, cap C = 16,384 tokens. The answer is the text after the last `Answer:` line, else the last `\boxed{}`, else wrong. It is graded by `math_verify.verify(parse(gold), parse(extracted))`, math-verify 0.9.0. GSM8K uses 1,319 rows of `openai/gsm8k` main at `740312ad…`. |
| 1.21 | **The noise of one reading** | 9 | `N13_SPEC.md` §0.5 lines 200–204; `n13/thresholds.json` | σ = 100·sqrt(Σ_i (x_iA − x_iB)²/2)/500 from two seeds of the same weights, maximum of base and RL: 1.924 points. δ = 2√2σ = 5.44 points. The gain G = 34.8 points was required to exceed 13.60. |
| 1.22 | **The judge's control and the unrelated share** | 1 | `SPEC_D1.md` lines 295–308, 484–495; `studies/already-clean/results/RESULTS.md` lines 18, 89 | Control: the same two-choice task with the side's main group and the record at T_14. The gate is ≥ 0.80 on the 40 tuning sides (0.910 on 39). On S it is a_ctrl = 0.924. p_u = (a_ctrl − A)/(a_ctrl − 0.5) = 0.265 [0.211, 0.321] with A = 0.812. Partial: M4 says "the control the judge had to pass" but not the formula that turns accuracy into the share. |
| 1.23 | **The explanation test and the one-job class** | 1 | `SPEC_D1.md` §2.2, §2.3, §6.1 lines 468–476 | Score of a written text = the max over positions ≥ 1 of a (min for the negative side) / the side's corpus max\|a\| from the 50M-token harvest. Strong = score ≥ 0.3153. Pass = ≥ 3 of 5 positives strong, ≤ 2 of 5 near misses strong, ≤ 1 of 5 distant texts strong. Covered by the existing `explanation_test` and `one_job` pages, not by an M-picture. |
| 1.24 | **The edit pairs and "older"** | 10 | `N6_SPEC.md` §0.5 lines 105–108, §2.3 | Age = the birth index k of the side's main group. Pairs are two dated sides of one layer, different neurons, births ≥ 3 snapshots apart. They are matched on ln max\|a\| (caliper 0.10) and ln density (0.50), then on ln‖w‖ and ln E⁺. p_col = the share of pairs whose early member's column edit costs more: 16 of 21 = 0.762 (`n6/thresholds.json` → `discovery`). |
| 1.25 | **The power check** ("the noise exceeded the effect") | 10 | `N6_SPEC.md` §3.4; `n6/thresholds.json` → `G_POWER` | τ_s = 1.5096e-6 nats per token must be ≤ 0.5 × median \|D_col\| = 6.363e-7. The ratio was 1.186 against a bar of 0.5. |
| 1.26 | **The rounding probe** | 6, 8, 10 | `N2V_SPEC.md` §3.3 lines 324–337; `N6_SPEC.md` §0.5 "probe" | The same change is added and then subtracted in the same forward: zero in exact arithmetic. Its loss change is the measured rounding floor (median 1.47e-4 nats per pair window sum, `verdict_n2v.json` → `test.probe_median_abs`). |
| 1.27 | **Chapter 2's operations** | 2 | gh-pages `index.html` text lines 540, 552, 593; `studies/reuse/SPEC4_birth.md` §0; `studies/reuse/SPEC6_generic.md`; `N6_SPEC.md` §3.1 K1 | Writing a value (average, median, zero or a random draw) into a group's 50 input neurons at every position and reading the target neuron's fall, (a_before − a_after)/a_before. Also: the fake-neuron control; the always-on cut (the top 1 % of earlier-layer neurons by mean activation, 3,412 neurons); and the normalization-step test (holding the RMS divisor fixed). The plan condenses this text but no picture or definition carries these operations. |

---

## 2. Exactness check of M1 to M12

Compared against the code that ran and the specs. "Severity: changes a result" means a reader would take away a wrong
fact.

| # | M | the plan says (PLAN line) | the correct form | file and line | severity |
|---|---|---|---|---|---|
| 2.1 | M1 | "in each, attention then an MLP; a vector of 4,096 numbers per token that every layer reads and adds to" (61) | OLMo-3 has no norm on a sublayer's input. It has an RMSNorm on each sublayer's **output** before the add: h ← h + RMSNorm_postattn(Attn(h)); h ← h + RMSNorm_postffn(MLP(h)). A final norm comes before the LM head. | `infra/pg/n13_arms.py` lines 30, 50–56 (the 355 tensors: q_norm, k_norm, post_attention_layernorm, post_feedforward_layernorm, model.norm; no input norm); transformers `Olmo3DecoderLayer` (local cache 5.22.1, lines 258–265; the runs used 5.9.0, `M1_SPEC.md` §2.2); gh-pages `index.html` text line 552 ("Each layer divides its output by the output's own size") | changes a result (N6's R is read after this norm; N13's "attention" condition carries its weights) |
| 2.2 | M1 | attention not specified (61) | 32 heads and 32 key-value heads (no grouped query). Query and key RMSNorm. Layers alternate three sliding-window layers (window 4,096) and one full layer. On 2,049-token inputs every layer sees the whole window. Vocabulary 100,278, untied embeddings, 7,298,011,136 parameters. | `infra/pg/m1.py` REF_CONFIG lines 94–101; `M1_ASSETS.md` line 19; `M1_SPEC.md` §0.2 line 53 | detail |
| 2.3 | M1 | "the sixteen steps and their token counts" (61) | Correct for T_0…T_15 (tokens = step × 4,194,304; `SPEC_D1.md` §4.3 lines 399–416). The timeline must also mark the off-grid points the chapters use (item 1.16). | as cited | changes a result (chapter 4's start point) |
| 2.4 | M1 | source "the model config in the repo" (61) | The config is `infra/pg/m1.py` REF_CONFIG lines 94–101 (the as-released mix config says `olmo2-retrofit`; `M1_SPEC.md` §2.3). | as cited | source |
| 2.5 | M2 | "one row of each input matrix and one column of the output matrix, with the true sizes (4,096 by 11,008)" (62) | gate_proj and up_proj weights are **11,008 × 4,096** (row n is neuron n). down_proj is **4,096 × 11,008** (column n). That is 12,288 numbers per neuron. | `N13_SPEC.md` §0.5 lines 177–178; `infra/pg/n13_arms.py` lines 54–56 | detail |
| 2.6 | M2 | "the exact SwiGLU expression" (62) | a_n = silu(g_n)·u_n, with g_n = W_gate[n,:]·h, u_n = W_up[n,:]·h and silu(x) = x·σ(x). h is the raw residual vector (no input norm). The harvest computes a in fp32 from the bf16 g and u. | `N6_SPEC.md` line 103; `M2_SPEC.md` line 156; `infra/pg/neurons_harvest.py` lines 161–186, 484 | detail |
| 2.7 | M2 / §1 | "units that switch on for some inputs and stay off for others" (12–13) | a is signed and continuous. A neuron has a positive side and a negative side (item 1.1). The double-negative regime (g < 0, u < 0) is the other side's echo and is excluded from groups. | `M2_SPEC.md` line 150; `SPEC_D1.md` §2.1 lines 146–149, §2.4 lines 274–279 | changes a result (sides are the unit of chapters 5, 8, 10) |
| 2.8 | M2 | "it adds activation times its output column to the shared vector" (62) | The 11,008 writes a_n·W_down[:, n] are summed. The sum is divided by its RMS over the 4,096 coordinates and multiplied by the post_feedforward_layernorm weight, and only then added. One neuron's write is not added on its own. | as 2.1 | changes a result |
| 2.9 | M3 | "run the model over the fixed corpus" (63) | The corpus must be stated (item 1.2), including the exclusion of window position 0. | `SPEC_D1.md` §4.2 | detail |
| 2.10 | M3 | "keep the fifty largest … (and separately the fifty most negative)" (63) | Per side: the 50 largest a for the positive side, the 50 largest −a for the negative side, as raw values over 20,560,068 tokens. | `SPEC_D1.md` line 146; `M1_SPEC.md` lines 100–103 | detail |
| 2.11 | M3 | "with their surrounding text" (63) | The stored record has 64 tokens before and 16 after. The explainer and the group-maker saw 256 before and 16 after. The judge's records use 64 before. | `SPEC_D1.md` lines 161–166 | detail |
| 2.12 | M3 / M5 | — | The positions are fixed at T_15 and read at every earlier snapshot. T_0–T_14 were harvested with T_15's positions. So "the fifty" on the births slider are the end-of-training positions, not each snapshot's own top 50. | `SPEC_D1.md` lines 168–173 | changes a result (a reader would think groups enter a moving top 50) |
| 2.13 | M4 | "the fifty contexts sorted into groups by shared content" (64); chapter 1 "had a judge name each group" (86) | The groups are made and named by a separate model (Claude Sonnet 5.5; 883 answers through Claude Code subagents, 317 through the API). The judge names no group. Echo contexts form one flagged group that is excluded. A second independent sorting gives m′ (agreement \|m − m′\| ≤ 1). | `SPEC_D1.md` §2.4 lines 274–279; `RESULTS.md` lines 137, 145 | changes a result |
| 2.14 | M4 | "what the judge was asked, word for word" (64) | It is a two-choice parent test. The judge sees 10 contexts of a late-born group and two unlabeled records. Each record is the top-20 contexts at the snapshot before the birth: one from the group's own side, one from a random side of the same layer. Contexts inside the group's rendered windows are removed from both. No numbers are shown. There are 4 judgments per group. Judge: Sonnet 5.5, prompt sha `11b5fd81…`. | `SPEC_D1.md` lines 295–304; `RESULTS.md` lines 18, 145 | detail |
| 2.15 | M4 | "the control the judge had to pass" (64) | See item 1.22. Also: the hard-decoy row could not be measured (control accuracy 0.798 < 0.80). | `RESULTS.md` line 18 | detail |
| 2.16 | M5 | "the earliest snapshot at which its contexts are among the neuron's fifty" (65) | **Wrong rule.** A group is present at t when the median over its end-of-training positions of a⁺_t is ≥ the side's own 99.9th percentile of a⁺ at t. It is born at the first t where it is present at t and at t+1 (a group first present at T_15 is born at T_15). The existing births page states it correctly (gh-pages `births/index.html` line 57). | `SPEC_D1.md` lines 280–287; `studies/next/README.md` lines 48–50; `M2_SPEC.md` lines 151–153 | changes a result |
| 2.17 | M5 | — | For a group to enter the primary count, the raw-ratio reading (median ≥ 0.3·max\|a_t\|) must agree on early versus late. | `SPEC_D1.md` lines 285–287, 500–501 | detail |
| 2.18 | M5 | — | Chapters 5–8 use only the 153 passing births (item 1.5). | `m2/births.json` → `rule`, `n_births` | changes a result |
| 2.19 | M5 | — | Births are gradual. In 93 of 153 births the dated bracket holds less of the rise than the two brackets before it. One snapshot earlier the median ratio to the threshold is already 0.84. | `N1_SPEC.md` §0.6 item 2 line 107; §2.1 table | changes the reading of "the moment a group appears" (chapter 1's open question, chapter 7) |
| 2.20 | M6 | "set the activation to zero at every token" (66) | Only at the group's own positions inside each unit window: the neuron's down-projection input h is set to 0 at those indices (1 to 2,047), and everywhere else it is untouched. | `N12_SPEC.md` §0.5 line 147; `n12_score.py` lines 175–199; `N2V_SPEC.md` lines 100–106 | changes a result |
| 2.21 | M6 | "run the model over W windows of 2,048 tokens" (66) | Over the unit windows only (item 1.3), each BOS + 2,048 tokens = 2,049 inputs. | as cited | detail |
| 2.22 | M6 | "the difference in loss, in nats per token, averaged" (66) | ΔL = Σ_j [ℓ_zero(j) − ℓ_clean(j)] over j = 0…2,047 and **summed** over the unit's windows, in nats, in fp64 over fp32 per-position differences. It is not a per-token mean. The per-token mean over 1,024 windows is N6's edit loss (`N6_SPEC.md` line 130), a different study. | `N2V_SPEC.md` lines 108–110; `infra/pg/n2v.py` `pair_values` lines 109–129 | changes a result (units of the 0.022-nat medians) |
| 2.23 | M6 | "Show one window token by token: the loss before, after" (66) | The 32-bit rows exist (S3, §4). The 16-bit per-token rows were never saved: N12 banked only each pair's window sum (`abl[i][2]`; `n2v_test.jsonl` → `meas_bf16`). A 16-bit against 32-bit token-by-token view cannot be drawn from the banks. The forward is `model.float()`, matmul precision "highest", TF32 off. | `N2V_SPEC.md` §2.1 lines 226–227, §2.2 step 7; `n12/READ_n12.md` line 72 | data limit |
| 2.24 | M7 | "activation times the derivative of the loss with respect to the activation, summed over tokens" (67) | ΔL̂ = Σ_i **−**h_n(i)·G_n(i). G_n(i) = ∂L/∂h_n(i), with L the window's **summed** next-token cross-entropy. The sum runs over the zeroed indices i only. The minus sign comes from the change δh = −h. | `N12_SPEC.md` lines 132–134, 147–148; `N2V_SPEC.md` line 111, §2.3 | changes the sign of every plotted prediction if copied |
| 2.25 | M7 | "the same number computed in 16-bit and in 32-bit on one window, side by side" (67) | The 16-bit/32-bit contrast that decided is in the **measurement** (M6): 16-bit median \|ΔL\| 0.65 nats against \|ΔL̂\| 0.022 (`n12/READ_n12.md` lines 23–27, §6). The prediction was fine in 16-bit: 16-bit against 32-bit prediction ρ = 0.995, median relative difference 2.6 % (`verdict_n2v.json` → `test.bf16_vs_jvp_prediction`). The 0.960 headline keeps the 16-bit prediction (`test.window.validation.prediction = "pred_bf16"`, `measurement = "dL"`). Per pair, not per window (`n_windows` varies). | as cited; `studies/next/JOURNAL.md` line 87 (2026-11-06 08:10 CST: "G-VAL measured the resolution of a bf16 window-summed ablation, not the gradient's validity") | changes a result |
| 2.26 | M7 | — | The 32-bit prediction came from forward-mode differentiation along the tangent −h, because a 32-bit backward needs 67.2 GB against the L40S's 48.3 GB. It equals reverse mode to 3e-6 relative on a tiny model. | `N2V_SPEC.md` §2.3 lines 256–282, §2.2 step 5 | detail |
| 2.27 | M8 | "a selection picks entries (whole neurons, or random single entries)" (68) | The ranking norm must be shown (item 1.17). | `N13_SPEC.md` lines 189–192 | detail |
| 2.28 | M8 | "the original plus the selected change" (68) | θ_T = bf16( float32(θ_b) + c ⊙ M ⊙ (float32(θ_a) − float32(θ_b)) ) per tensor. There is a final bf16 cast. The builder asserts bitwise θ_a on M = 1 and θ_b on M = 0 when c = 1. | `N13_SPEC.md` §3.1 lines 381–391 | detail |
| 2.29 | M8 | — | Which non-MLP tensors come from which model differs by condition: base for top, random, sample and all-MLP; RL for the complement; RL attention and norms for "attention with norms". | `N13_SPEC.md` §3.2 lines 407–414 | changes a result |
| 2.30 | M8 | "the original model" (chapter 9, 94) | Base = `allenai/Olmo-3-1025-7B` `main` (`a81bae42`), the finished base model after stages 2 and 3, not T_15. RL = `allenai/Olmo-3.1-7B-RL-Zero-Math` (`364820b0`). | `N13_SPEC.md` lines 282–284; `studies/next/JOURNAL.md` line 18 | detail |
| 2.31 | M9 | "keep each entry with probability q" (69) | The pool is MLP entries only (a stated deviation from published DARE). q_k = k_n/352,256. 1/q = 1,000.73, 333.26, 99.99, 33.33. The mask is a Bernoulli draw per entry (torch Generator, seed 30 + j). | `N13_SPEC.md` line 411, lines 428–432 | changes a result (chapter 9 and F8 say "all changed numbers") |
| 2.32 | M9 | "show why the expected value equals the full change" (69) | The expectation holds for c·M·Δ before the final bf16 cast. The cast rounds each kept entry. The realized kept count is asserted within 0.1 % of plan. | `N13_SPEC.md` lines 381–383, 433–436 | detail |
| 2.33 | M9 | "the sum of squares that grows by 1/q" (69) | Correct in expectation: E[Σ(MΔ/q)²] = E/q, 33 to 1,000 times the change's energy. | `20_ABSTRACT.md` line 67 | equal |
| 2.34 | M10 | "the sum of squared changes over a selection" (70) | E(S) = Σ_{n∈S} s_n² = Σ_{n∈S} (‖Δg_n‖² + ‖Δu_n‖² + ‖Δd_n‖²), over whole neurons, never over single entries. | `N13_SPEC.md` lines 189–190 | detail |
| 2.35 | M10 | "two selections with equal sums" (70) | There are two constructions. rndE: k_n random tail neurons with their change times one scalar c_k = sqrt(E(S_k)/E(R)). rndT: tail neurons at their own change in random order, added until E ≥ E(S_k); the count m_k is reported. The realized energy was checked within 5 % (A30) and to 1e-12 (A31). Chapter 9's "random neurons of equal energy" must say which. | `N13_SPEC.md` lines 409–410, 421–427, A30 line 2463, A31 line 2530; `n13/s3/s3gate-01/r1_rank/weight_stats.json` → `rndE_c`, `rndT_m` | changes a result |
| 2.36 | M11 | "add c times a unit vector to a neuron's output column" (71) | This is not a weight edit applied everywhere. At each eval position x ≥ 1 where s·a_n(x) > 0 (the side's own tokens), the down projection's output gains a_n(x)·δ. It is a rank-one change of column n applied on that side's tokens only. | `N6_SPEC.md` lines 110–113, 224–225 | changes a result |
| 2.37 | M11 | "its own direction" (71) | δ = **−**(c/‖w‖)·w. It removes a fraction κ = c/‖w‖ of the side's write (κ = 1 for the smaller-norm member of the pair). Random: δ = c·r, with r a seeded unit vector. | `N6_SPEC.md` lines 114–116 | changes a result (the sign) |
| 2.38 | M11 | "c" (71) | c = min(‖w_early‖, ‖w_late‖), the same for all four edits of a pair. For a control neuron, col runs at κ = 1 and rand at the same c. | `N6_SPEC.md` lines 121–122, §3.3 | detail |
| 2.39 | M11 | "the change this makes to the shared vector, measured as the spec defines it" (71) | R = ‖h_(l+1),edit − h_(l+1),base‖_F over all eval positions and 4,096 coordinates, read **after** layer l: after the post-feedforward RMSNorm and the residual add. | `N6_SPEC.md` lines 226–229 | changes a result (2.1 applies) |
| 2.40 | M11 | "the loss difference over the eval windows" (71) | ΔL = (1/(2,048·1,024)) Σ [ℓ_edit − ℓ_base], in nats per token, fp32, over the band-3 windows (item 1.4). | `N6_SPEC.md` lines 125–131 | detail |
| 2.41 | M11 | chapter 10 "a control pair in eight layers" (95) | The control is one neuron side per layer: the side with the largest E⁺ = Σ_x (s·a_n(x))₊² among the layer's 22,016, in layers 0, 4, …, 28. Its column edit at κ = 1 is compared with a random edit at the same c. Pass needed col > rand + τ_eps in ≥ 6 of 8. | `N6_SPEC.md` line 135, §3.3 | detail |
| 2.42 | M12 | "a known number reproduced …; a positive control …; a control that must land at chance; predictions written before the data; a second model re-deriving every number" (72) | Missing: (a) every check has a registered **failing input** that must fail; (b) the **identity pair**, the same weights run twice, which must agree bitwise or to a stated tolerance; (c) the split rule (item 1.6); (d) a study whose check fails is set aside before its held-out half is read (N6 `n_test_read` 0, `n6/JOURNAL.md` lines 264–271). | `studies/next/README.md` house rules; `CLAUDE.md` "Instruments and controls" 2, 4 | detail |
| 2.43 | M12 | "a second model re-deriving every number" (72) | Correct in kind. The readers were fresh Opus agents in their own worktrees; the head is Fable. The abstract's reader made four passes (`20_ABSTRACT.md` line 3). | as cited | equal |

---

## 3. Numbers in the plan

Each number in §1, §3 and §4 against `studies/next/20_ABSTRACT.md` (evidence table lines 28–62) and the files it names.

| # | PLAN line | the plan's number or claim | the file's value | status | file |
|---|---|---|---|---|---|
| 3.1 | 14, 86 | sixteen snapshots | T_0…T_15, steps 0 to 1,413,814 (5.93T tokens) | equal | `SPEC_D1.md` lines 399–416 |
| 3.2 | 15 | seven-billion-parameter | 7,298,011,136 | rounded | `M1_SPEC.md` line 53 |
| 3.3 | 16 | "Most end up with one recognizable job" | OLMo-3: 0.518 [0.478, 0.558] one job under passing explanations; UNK 0.415 | mismatch: the interval includes 0.5 | `RESULTS.md` lines 47, 104 |
| 3.4 | 86 | "The claim holds for most neurons" | the essay claims about 90 %; OLMo-3 gives 51.8 % | mismatch | `SPEC_D1.md` §6.2 line 482; `RESULTS.md` line 47 |
| 3.5 | 86 | "later groups mostly grow from earlier ones" | p_u = 0.265 [0.211, 0.321]; lower bound > 0.20, so the frozen rule reads packing (H_SP); birth order retagged, not supported | mismatch with the registered reading | `RESULTS.md` lines 7–18; `SPEC_D1.md` line 495; gh-pages `index.html` text line 475 |
| 3.6 | 87 | 24 October to 4 November | the site's chronology heading | equal | gh-pages `index.html` text line 494 |
| 3.7 | 88 | 5 November, one night | journal 2026-11-05 00:10–02:33 CST | equal | `studies/next/JOURNAL.md` lines 3–19 |
| 3.8 | 88 | order: mix, carrier, gradient, weights, later skill, edit cost | N13 (the later skill) was the first pick; M1 second; the birth studies (J12) after | mismatch | `studies/next/09_PLAN.md` lines 20–21; `JOURNAL.md` line 15 ("N13 first") |
| 3.9 | 88 | "Chosen by expected information per GPU-hour" | leverage = wow × novelty × decisiveness × edge × inverted cost | mismatch | `studies/next/05_SCORING.md` lines 8–19 |
| 3.10 | 89 | 5 November | M1 journal 16:14Z to 23:21Z on 5 November; verdict 22:58Z | equal | `studies/next/m1/JOURNAL.md` line 21 |
| 3.11 | 89, 103 | 100 billion tokens | 47,684 steps × 2,097,152 = 100.0B | equal | `M1_ASSETS.md` line 67 |
| 3.12 | 89 | 0 %, 40 %, 65 % code and math | gen-mc-only 0 %, round5 40 %, math-code-reasoning-web 65 % | equal | `M1_ASSETS.md` line 41 |
| 3.13 | 89, 103 | "less than one slot of fifty" | G − start, band means: −0.0113, −0.0170, −0.0173, −0.0161 of 50 slots = 0.56–0.87 slots (exploratory) | rounded; the quantity is code-and-math slots, not "favorites move" | `verdict_m1.json` → `exploratory.shift_from_start.G`; abstract line 28 |
| 3.14 | 89 | "in the direction and roughly the amount the mix predicts" | λ = 0.70, 0.68, 0.66, 0.76 against the prior 0.62; H_order UNDECIDED (bands 0–1 below the bar) | mismatch in claim | `M1_SPEC.md` lines 159, 632; `verdict_m1.json` → `headline` |
| 3.15 | 89, 103 | F12 "how many of the fifty changed" | M1 measured the change in code-and-math slots per side, not position turnover | mismatch in quantity | `M1_SPEC.md` §0.5 lines 106–108, §2.1 line 170 |
| 3.16 | 90 | 5 to 6 November | M2 journal 2026-11-05 19:40 CST to 2026-11-06T07:57Z | equal | `studies/next/m2/JOURNAL.md` |
| 3.17 | 90 | "Scored all 22,016 neurons" | 22,016 sides of one layer = 11,008 neurons × 2 | mismatch | `M2_SPEC.md` lines 150, 166 |
| 3.18 | 90 | "five attempts failed their own checks before the sixth ran clean" | score-01, -02, -03 stopped on the known-number check; score-04 crashed in scoring code (an assertion on a side with no tokens of its sign); score-05 stopped because a check's failing input had no power. Two stop rules ("M2 ends without a verdict") were overridden by amendments A4 and A5. | rounded; the crash is not a check | `M2_SPEC.md` lines 26–58 |
| 3.19 | 90, 104 | 59 % | 42 of 71 = 0.592 [0.475, 0.698] | rounded | `verdict_m2.json` → `labels.H_B` |
| 3.20 | 90, 104 | 76 % | 53 of 70 = 0.757; carrier − twin −0.171 [−0.286, −0.057] | rounded | `verdict_m2.json` → `test.pb_pre_B`, `labels.H_sel_B` |
| 3.21 | 90 | "how many distinct words" | distinct decoded **tokens** | mismatch in unit | `N12_SPEC.md` lines 118–119 |
| 3.22 | 90 | "The score measures vocabulary breadth, not the job" | twins matched on distinct-token count close the gap (real 42/71, dt 41/67; real − dt −0.030 [−0.149, +0.090]); the carrier is still named at 1,300× chance. No file states the plan's sentence. | stronger than the files | abstract lines 13–14, 39; `n12/verdict_n12.json` → `labels.H_sel_dt` |
| 3.23 | 91 | 6 November | N12 journal 09:50Z to 13:29Z | equal | `studies/next/n12/JOURNAL.md` |
| 3.24 | 91, 105 | agreement 0.10 | ρ 0.0996 on the first half (328 pairs); 0.0425 on the held-out half (284) | rounded; the half is unstated. F9's toggle would set a first-half value beside a held-out 0.96. | `READ_n12.md` line 23; abstract line 40 |
| 3.25 | 91, 105 | "the prediction is noise" | the 16-bit switch-off measurement was noise; the prediction was within about 1 % | mismatch | `READ_n12.md` lines 146–203; `JOURNAL.md` line 87; `verdict_n2v.json` → `test.bf16_vs_jvp_prediction` |
| 3.26 | 91 | "Its answers were set aside unread" | the registered labels read DOES NOT COUNT, but the reader printed the exploratory labels on 6 November (H_signal REFUTED, H_bridge UNDECIDED) | mismatch | `READ_n12.md` lines 126, 323–326 |
| 3.27 | 91 | "rebuilt it in full precision, in a forward mode" | the switch-off was re-measured in plain fp32 forwards. Forward mode gave the fp32 prediction (needed for the own-token readout). The decisive comparison used the 16-bit prediction. | mismatch in what was rebuilt | `N2V_SPEC.md` lines 347–350, §2.3 |
| 3.28 | 92 | 6 to 7 November | N1 verdict 2026-11-07T19:35Z | equal | `studies/next/n1/JOURNAL.md` line 73 |
| 3.29 | 92 | 67 births | 67 pairs (real − dt) | equal | `verdict_n1.json` → `test.pair_in_real_dt.n` |
| 3.30 | 92 | "the median with its band" | the registered summary is the **mean** −0.432 [−0.817, −0.059]; 23 above, 44 below | mismatch | `verdict_n1.json` → `test.pair_in_real_dt`; `infra/pg/m2.py` lines 546–553 |
| 3.31 | 92, 106 | "At the moment a job appears" | e is read over the three brackets k−2, k−1, k against the neuron's own earlier brackets | rounded scope | `N1_SPEC.md` line 94 |
| 3.32 | 92 | margin two births | two births; the first half alone −0.217 [−0.638, 0.204]; natural units −1.6 % [−3.2 %, +0.03 %] | equal; the natural-unit interval nearly reaches zero and belongs on F11 | abstract lines 46, 49; `n1/read_n1/scale.json` |
| 3.33 | 93 | 7 November | verdict 2026-11-08T00:53Z (UTC) = 7 November 18:53 CST | rounded (time zone) | `studies/next/n2v/JOURNAL.md` line 107 |
| 3.34 | 93 | 284 switch-offs | 284 held-out pairs, 71 units | equal | `verdict_n2v.json` → `test.n_pairs` |
| 3.35 | 93, 105 | 0.96 | 0.95996 [0.915, 0.990]; shuffle 0.118 | rounded | `verdict_n2v.json` → `test.window.validation` |
| 3.36 | 93 | "The carrier wins by its own rise" | true for real units, and for twins equally (F_dt 0.942; real − dt +0.067 [−0.210, +0.333]). The dating rule forces it. | omits the control | `READ_n12.md` lines 18–22; `JOURNAL.md` line 87 |
| 3.37 | 94 | 5 to 8 November | spec 5 November; S5 read closed 2026-11-08T11:50Z | equal | `studies/next/n13/JOURNAL.md` line 344 |
| 3.38 | 94 | 0.1 to 3 % | k grid 0.1, 0.3, 1, 3 % of 352,256 neurons | equal | `N13_SPEC.md` line 193 |
| 3.39 | 94, 107 | "a random 0.1 % of all changed numbers scaled up a thousandfold" | each **MLP** entry kept with q = 352/352,256, scaled by 1,000.73 | rounded factor; mismatch in pool | `N13_SPEC.md` lines 411, 428–429 |
| 3.40 | 94 | 500 competition problems | 500 rows of MATH-500 | equal | `N13_SPEC.md` line 280 |
| 3.41 | 94 | "each checked against a known number first" | the published number was reproduced once (Minerva MATH 39.96 against 40.0 ± 1.6, `r0/s2setup-07`); the four reference conditions ran once (S3); each S4 condition checked engine settings and library versions against the S3 run (`engine_check.json`) | mismatch | abstract line 52; `n13/s4/s4armsA1-01/dare_k0.1/engine_check.json` |
| 3.42 | 94 | 57.6 to 63.8 against 57.2 | top 57.6, 61.4, 61.8, 63.8; base 57.2 | equal | `verdict_n13.json` → `arms.top_k*`, `arms.base` |
| 3.43 | 94 | "carry almost nothing" | top 3 % recovers 6.6 of 34.8 points (19 %); equal-energy random neurons read 58.8–63.8 | rounded wording | same; `thresholds.json` → `G` |
| 3.44 | 94 | 85.2 of 92.0 | dare 0.1 % 85.2; rl 92.0 | equal | `verdict_n13.json` → `arms.dare_k0.1`, `arms.rl` |
| 3.45 | 94 | "attention only" | attention plus every norm weight (item 1.18): 83.0 | mismatch | `N13_SPEC.md` line 413; abstract line 21 |
| 3.46 | 107 | F8: top 1 % against a random 0.1 % | not matched in count: top 1 % is 43,290,624 numbers, the 0.1 % sample about 4,325,376. The registered matched pair is top 0.1 % (57.6) against sample 0.1 % (85.2), the same count (`N13_SPEC.md` §3.5 "dare_k: the H_diffuse test at the same parameter count"). | mismatch in design | `N13_SPEC.md` lines 193–195, 454 |
| 3.47 | 95 | 6 to 8 November | N6 journal 2026-11-06T15:10Z to 2026-11-08T15:15Z | equal | `studies/next/n6/JOURNAL.md` lines 6, 264 |
| 3.48 | 95 | 21 pairs | 21 | equal | `n6/thresholds.json` → `discovery.n` |
| 3.49 | 95 | eight layers | layers 0, 4, …, 28 | equal | `N6_SPEC.md` §3.3 |
| 3.50 | 95 | 1.3 to 2.5 times | recount (this audit) from `edits.jsonl`: rand/col R = 2.51, 1.83, 1.81, 1.91, 2.06, 1.75, 1.33, 1.62 in layers 0…28; col > rand in 2 of 8 | equal; scope: the 8 control sides only. On the 21 pairs the two edits moved the residual alike (median R 60.8 against 60.7, recount). F13's sentence must carry that. | `n6/banks/L2a-01/edits.jsonl`; abstract line 62 |
| 3.51 | 95 | "The noise also exceeded the effect" | τ_s 1.51e-6 > median \|D_col\| 1.27e-6 | equal | `n6/thresholds.json` → `G_POWER` |
| 3.52 | 96 | 8 November; four times; five results, one set aside | abstract FINAL 2026-11-08 09:30 CST; fourth pass; results (1)–(5), (6) does not count | equal | `20_ABSTRACT.md` lines 3, 9–22 |

---

## 4. Data files

Every path named in PLAN §4 exists in this branch. The table says whether each holds the per-item data its picture
needs. "LFS" marks a git LFS file: data scripts need a checkout with LFS objects, and a raw GitHub link to an LFS file
returns the pointer, not the data.

| # | picture | per-item data needed | named file: what it holds | where the per-item data is |
|---|---|---|---|---|
| 4.1 | F12 | per side, per mix: code-and-math slots (start, G, R, M) | `m1/verdict_m1.json`: band summaries only. `m1/banks/<harvest>/m1/counts.npz` (LFS, 10.7 MB each): counts[l, n, s, directory] for all 704,512 sides, per harvest | per-side shares: `m1/banks/{start_a,G,R,M}/m1/counts.npz`. If the picture keeps "how many of the fifty changed" (position turnover), positions exist in the repo only for the 640 grid sides (`m1/banks/*/samples/contexts.jsonl`, LFS). For all sides they are on S3 only: `s3://pg-lineage-769063704696/results/next_m1/<attempt>/out/<step>/topk/` (`M1_SPEC.md` lines 467–479). `m1/turnover/turnover.json` holds class summaries, not per-side values. |
| 4.2 | F10 | per birth: the carrier's rank, the twin's rank, d, the contexts | `m2/verdict_m2.json`: summaries. `m2/births.json`: per birth (153: layer, k, size, positions, twin). `m2/banks/score-06/m2/ranks_test.jsonl`: per birth and role, rank and value under every score plus top-10 lists (71 real × 3 roles, 70 twins × 2) | present for 59 % and 76 %. **Missing from the plan** for the vocabulary slider: `n12/units.json` (d per unit), `n12/banks/score-02/n12/ranks_test.jsonl` (distinct-token twin ranks), `m2/read/r5b_token_share.json`. Context text: 10 births in `m2/banks/score-06/m2/samples/raw_examples_test.jsonl`. For all births, the grid's end-of-training contexts are in the laptop bank (`infra/pg/m1.py` line 116, `t15_contexts.jsonl`, outside the repo) or on S3 at `results/next_m1/inputs/t15_contexts.jsonl` (`M1_SPEC.md` line 472). |
| 4.3 | F9 | per pair: 16-bit prediction, 16-bit measurement, 32-bit measurement, 32-bit prediction, the neuron and its text | `studies/next/n12` is a directory. `n12/banks/score-02/n12/n12_{freeze,test}.jsonl` (LFS) hold N12's records. `n2v/verdict_n2v.json`: summaries. `n2v/read_n2v/r1_recount.json`: 612 pairs recounted | both states of F9 come from one pair of small files: `n2v/banks/score-02/n2v/n2v_freeze.jsonl` (328) and `n2v_test.jsonl` (284), not LFS. Each row has `pred_bf16`, `meas_bf16`, `dL`, `pred_jvp`, `dL_own`, `pred_jvp_own`, `probe`, `sum_abs_h`, `n_later`. M6's per-token rows (32-bit only) are on S3 only: `results/next_n2v/score-02/` → `losses/tNN.npz`, 93.5 MB (`n2v/STATE.md` lines 28–29; sha256s in `READ_n2v.md` lines 55–63). The 16-bit per-token rows exist nowhere. Hover text: 10 units in `n2v/banks/score-02/n2v/samples/raw_examples_test.jsonl`; carriers as in 4.2. The three competitors per unit are not grid sides; their contexts need S3 `topk/` plus the corpus `tokens.npy`. |
| 4.4 | F11 | per birth: e of the carrier and of its comparison | `n1/verdict_n1.json`: summaries. `n1/read_n1/fragility.json`: drop-one summaries, not per birth | `n1/banks/weights-03/n1/n1_test.jsonl` (LFS, 208 rows: 71 real, 67 dt, 70 pb) holds e_window, e_k, r, z per side. Natural units: `n1/read_n1/scale.json`. The full statistics array is on S3 only: `results/next_n1/weights-03/out/weights/n1/stats.npz` (`n1/JOURNAL.md` line 91). |
| 4.5 | F8 | per condition accuracy; per-problem answers; three example problems | `n13/verdict_n13.json`: accuracy for all 33 conditions | Per-problem answers for all 500 × 33 are on S3 only: `s3://pg-lineage-769063704696/results/next_n13/s5-01/banks/` (165 files; manifest sha256 `56c87b3c06c4…`; `n13/STATE.md` lines 23–24, 36). **The three example problems need no fetch**: problems 0, 1 and 2 are banked for every condition in `n13/s3/s3gate-0{2,3,4}/<arm>/math500/samples.jsonl` and `n13/s4/<launch>/<arm>/<arm>/math500/samples.jsonl` (fields `text`, `extracted`, `gold`, `score`). M8–M10 also need `n13/s3/s3gate-01/r1_rank/sets.json` (neuron sets, E_top, E_tail, dare_q) and `weight_stats.json` (rndE_c, rndT_m), which the plan does not name. |
| 4.6 | F13 | per edit: loss change and residual change | `n6/banks/L2a-01/edits.jsonl` (LFS, 113 rows: 42 pair col, 42 pair rand, 8 control col, 8 control rand, 8 control col_half, 4 probes, 1 zero; each with c, κ, ‖δ‖, dL, dL_win [1,024], R, E⁺, kl_eff). `metrics.json`: gates | present. |
| 4.7 | M2, M8 | real weights (a row, a column; one matrix's change as a heat map) | none | not banked. These need the checkpoint weights (and for M8 both checkpoints). That is a download, which the laptop rule forbids. It needs an instance, or the picture uses per-neuron values from `sets.json`. |

---

## 5. Vocabulary

### 5.1 Hits in the plan's own text (§1 lines 10–27, §3a lines 51–77, §3 lines 79–96, §4 lines 98–111)

Word-boundary grep. Rows from the §2 left column first, then the extra words the head listed, then other internal or
invented words.

| # | word | PLAN line(s) | context | note |
|---|---|---|---|---|
| 5.1 | arm, arms | 68, 69, 70 | "N13 arms", "the DARE arm definition", "the energy-matched arms" | source column; builders will copy it |
| 5.2 | DARE | 69 | source column | as above |
| 5.3 | label | 77 | "every mark is a datum or a label" | UI sense. The scanner of line 49 will fail on every axis label. |
| 5.4 | slot | 89, 103 | "one slot of fifty", "less than one slot" | §2 bans "top-50 slot" |
| 5.5 | S3 | 107 | data column (the bucket) | §2 bans "S3/S4" as stage codes; the scanner cannot tell the two apart |
| 5.6 | verdict | 103–107 | file names `verdict_*.json` | conflict: §4 asks for "data" links naming files, and the scanner fails on "verdict" (also "banks", "read_n2v", "thresholds") |
| 5.7 | set aside | 87, 91, 96 | §2's own public phrase | allowed |
| 5.8 | readout, pole, gate, freeze, card, lane, executor, head, ablation, transplant, residual stream, MATH-500, GSM8K, Spearman, Wilson, bootstrap, LFS, sha256 | none in these lines | — | sha256 appears at line 116 (§5); if a data.js hash is shown on a page, it needs a public word |
| 5.9 | stranger | 90, 104 | "a stranger's future contexts" | an invented word, against PLAN line 27. Use "a random group of another neuron's contexts, of the same size and dated to the same snapshot". |
| 5.10 | dated alike | 92 | "a comparison neuron dated alike" | internal shorthand; state the matching variables (item 1.9) |
| 5.11 | known number | 72, 88, 94 | | internal term; "a number published or measured before, reproduced first" |
| 5.12 | favorites | 89, 103, 104 | "fifty favorites" | informal; §2 gives "the fifty inputs a neuron responds to most" |
| 5.13 | agreement | 91, 93 | "agreement 0.10/0.96" | §2 prescribes "rank agreement" |
| 5.14 | half precision / full precision; 16-bit / 32-bit | 91, 93, 105; 67 | | two vocabularies for one thing. bf16 is not IEEE half precision. Use "16-bit (bfloat16)" and "32-bit". |
| 5.15 | the instrument | 91, 93 | | internal; "the method", "the prediction" |
| 5.16 | chooser | 94 | | UI jargon; "a menu" |
| 5.17 | the always-on neurons; the training-point test; the normalization step; a known edit | 87 | | each needs a definition or a plain name (item 1.27; 2.1) |
| 5.18 | control pair | 95 | | inaccurate (item 2.41) |
| 5.19 | the evidence table; the rank test | 96 | | internal document and study names |
| 5.20 | W | 66 | "W windows" | a symbol without a value |
| 5.21 | carrier, birth, job, group, top contexts | 63–65, 86–106 | | kept by §2, line 47; each needs its one definition, and birth's definition must be the right one (2.16) |
| 5.22 | MLP, SwiGLU, nats, first-order, RL | 61, 62, 66, 67, 68, 94 | | real terms; define once |

### 5.2 Internal words in the specs, READ files and the current site that the §2 table does not list

| # | words | where they occur | suggested public form |
|---|---|---|---|
| 5.23 | twin, pseudo-birth twin, pb, dt, distinct-token twin, unit (real/pb/dt) | `M2_SPEC.md` §3.3; `N12_SPEC.md` §0.5; `N1_SPEC.md` §3.3 | "comparison group" with its matching variables |
| 5.24 | competitor, competitor set, leaders, pull, loss-matched, u | `N12_SPEC.md` §0.5 | "the 20 sides that scored highest one snapshot earlier"; describe the gradient quantity |
| 5.25 | exemplar, exemplar positions, E_c, a⁺, q999 | `M2_SPEC.md` §0.5; `SPEC_D1.md` §2.4 | "the group's contexts"; "the side's own top 0.1 % of activations" |
| 5.26 | score A, score B, hit, precision at 10, matched poles, floor, ceiling, room, bar | `M2_SPEC.md` §0.5, §3.4–3.5 | "in the top ten"; "chance level" |
| 5.27 | probe, identity pair, failing input, K0, K-A, K-B, G-EX, G-VAL, G-PC, G-POWER, G-ROOM, G-CAUSAL, G-TOPK | every spec §3–§4 | "a rounding check"; "the same weights run twice"; "an input that must fail" |
| 5.28 | SUPPORTED, REFUTED, UNDECIDED, BELOW_SELECTION, AT_SELECTION, ABOVE_CEILING, OWN_RISE, LOSES, NONE, NOT NECESSARY | verdict files; abstract lines 15–22, 45–59 | plain sentences |
| 5.29 | study and wave codes M1, M2, N12, N2V, N1, N13, N6, N22, J1–J14, D1, E1, W0–W5; model codes O7, Q4B, G4 | everywhere; data paths contain them | none in prose; paths may show them |
| 5.30 | T_0…T_15, T_k, k, bracket, rise window, roles pre/birth/end, R1t, p_late | `M2_SPEC.md` §0.5; `N1_SPEC.md` §0.5; `10_RECOUNT2.md` | "the snapshot before the birth"; step numbers |
| 5.31 | e, z, r; cm_share, band, drift bar τ, λ | `N1_SPEC.md`; `M1_SPEC.md` | name the quantity in words |
| 5.32 | rndE, rndT, rnd0, comp, topF, top_k, mlp, attn, dare; seed set A/B; σ, δ, H, Q, G; cap hit | `N13_SPEC.md` §0.5, §3.2 | describe each condition |
| 5.33 | echo, grid pole, S, S_gate, audit sample, discovery set, strong line, θ_s, ONE, TWO-REL, UNREL, UNK, RARE | `SPEC_D1.md`; current site | plain names |
| 5.34 | reuse score, replacement reuse, fake neurons, one-pass estimate, halves, sorting model, readout, label | current front page (fake neuron 22×, always-on 15×, sorting model 14×, readout 3×, label 4×) | define or replace; the scanner will catch "readout" and "label" in the current text |
| 5.35 | bf16, fp32, TF32, sdpa, BOS, EOS, instrument | every spec | explain once or drop |

Scanner conflicts to settle before the build: "label" (axis labels), "verdict" / "S3" / "null" inside file names and
JSON shown by data links. Either exclude code spans and paths, or rename files in the public copy.

---

## 6. UI and UX

Seven picture pages on `gh-pages` (`git ls-tree -r --name-only gh-pages`; commit `2dca1e037`).

| page | own data.js | viewport meta | phone media query | shared stylesheet | caption | data link | notes |
|---|---|---|---|---|---|---|---|
| births | yes (35 KB), line 61 | yes, line 5 | `births.css` lines 17, 177 (max-width 640px) | no: own `births.css` plus an inline token block (variant A) | yes, JS `nb-caption` (`births.js` lines 74, 513–518) | no link; prose "Source:" line 59 | says "600 OLMo-3 7B neurons"; they are 600 sides of 300 neurons (`SPEC_D1.md` §4.4) |
| explanation_test | yes (114 KB) | yes | `explanation_test.css` lines 18, 182 | no (variant A) | yes, `xt-caption` | no link; "Source:" line | correct "600 sides of 300 neurons" |
| groups | yes (504 KB) | yes | inline line 44; `groups.css` line 153 | no (variant B, no `--grid`/`--bar`) | yes, `ngroups-caption` | none | no "Source:" line on the standalone page |
| judge | yes (762 KB) | yes | `judge.css` line 127 | no (variant A) | yes, `jd-caption` | no link; "Source:" line | — |
| neuron_across_training | yes (2.66 MB) | yes | inline line 43; `slider.css` lines 20, 194, 198 | no (variant C) | yes, `nat-caption` | none | the hero's data.js is half the site's data |
| one_job | yes (1.00 MB) | yes | `one_job.css` lines 24, 147 | no (variant A) | yes, `oj-caption` | no link; "Source:" line | says "600 neurons tested in a model"; they are 600 sides |
| top_contexts | yes (254 KB) | yes | inline line 44; `top_contexts.css` lines 16, 163 | no (variant B) | yes, `tc-caption` | none | no "Source:" line on the standalone page |

Cross-page findings:

- 6.1 **No shared stylesheet.** Each page has its own CSS plus a copied inline `:root` block in three variants (A:
  births, explanation_test, judge, one_job; B: groups, top_contexts; C: neuron_across_training; compared by md5 of the
  `<style>` blocks). PLAN line 74 asks for one stylesheet.
- 6.2 **No data link anywhere.** No page has a link to its data file. The front page and 4 of 7 standalone pages name
  the source in prose only. PLAN line 77 asks for a "data" link.
- 6.3 **Not fully offline.** All pages load fonts from fonts.googleapis.com (line 9 of each). Data is local, and text
  falls back to system fonts, so the pages work but change typeface without a network.
- 6.4 **Phone width.** Every page has a viewport meta and a 640px media query. The front page's figure 3 says "On a
  narrow screen this figure scrolls sideways" (gh-pages `index.html` line 419), which breaks the no-horizontal-scroll
  rule.
- 6.5 **Front-page weight.** The front page loads all seven data.js files (`index.html` lines 707–720): 5.33 MB
  before any picture is used.
- 6.6 **Missing-data message.** If data.js fails, each page tells the public reader to "Run python3 extract.py in the
  folder" (for example `births.js` line 12). extract.py is not published on gh-pages (commit `2dca1e037` message).

---

## 7. Count

| section | items |
|---|---|
| 1. Mechanisms missing | 27 (1.1–1.27; 4 of them partial: 1.1, 1.17, 1.22, and 1.23 covered by an existing page) |
| 2. Exactness of M1–M12 | 43 rows (2.1–2.43): 19 change a result, 20 details, 2 equal, 1 data limit, 1 source |
| 3. Numbers | 52 rows (3.1–3.52): 23 equal, 10 rounded, 17 mismatched (3.3, 3.4, 3.5, 3.8, 3.9, 3.14, 3.15, 3.17, 3.21, 3.25, 3.26, 3.27, 3.30, 3.39, 3.41, 3.45, 3.46), 2 claims stronger than the files (3.22, 3.36) |
| 4. Data files | 7 rows: 3 present as named (F11 via `n1/banks`, F13, F10 for 59 %/76 %); 4 need files the plan does not name (F10 slider, F9 exact rows, F8 sets, F12 if turnover is kept); 3 need S3 (M6 per-token rows, F8 per-problem grid, F12 turnover); 2 not banked anywhere (16-bit per-token rows; weights for M2 and M8) |
| 5. Vocabulary | 22 rows on the plan's text (5.1–5.22): 6 words from §2's left column with hits (5.1–5.6), 1 allowed phrase (5.7), 1 row of listed words with no hit (5.8), 14 other internal or invented words (5.9–5.22); and 13 proposed additions to the §2 table (5.23–5.35) |
| 6. UI and UX | 7 pages in the table and 6 cross-page findings |

The three findings that matter most:

1. **Which side of the 16-bit failure was broken (2.22–2.26, 3.25–3.27).** M7, chapter 6 and F9 say the prediction
   was noise in 16-bit. The files say the 16-bit switch-off measurement was the noise. The 0.960 pairs N12's
   **16-bit** prediction with a **32-bit** switch-off (`verdict_n2v.json` → `test.window.validation`; journal
   2026-11-06 08:10 CST, `studies/next/JOURNAL.md` line 87). M6's definition (zero at every token, nats per token
   averaged) is N6's edit loss. The switch-off zeroes the neuron only at its group's positions and sums the loss
   change over whole windows, in nats.
2. **The birth rule and the 153 (2.16–2.19, 1.5, 3.17).** M5's rule is not the one that ran. The rule is the median
   activation at the end-of-training positions reaching the side's own 99.9th percentile, at t and at t+1. Every
   carrier in chapters 5–8 is one of 153 passing births (p_late ≤ 0.05 against random same-size groups), and no picture
   shows that filter. Chapter 5's "22,016 neurons" are 22,016 sides of 11,008 neurons.
3. **Claims stronger than their files.** Six places: chapter 1 ("most neurons", "later groups mostly grow from earlier
   ones" against 0.518 [0.478, 0.558] and the frozen rule's packing reading); chapter 4 (F12 measures code-and-math
   share, not how many of the fifty changed; H_order UNDECIDED); chapter 6 ("set aside unread"); chapter 9 ("attention
   only" carries every norm weight, including the one on each MLP's output); F8 (top 1 % against a 0.1 % sample is not
   the registered same-count pair, which is 57.6 against 85.2); chapter 10 (the 1.3–2.5× holds for the eight control
   sides; the 21 pairs moved the residual alike, 60.8 against 60.7). M1 and M2 also leave out OLMo-3's norm after each
   sublayer, which sits between N6's edit and its measured residual change.
