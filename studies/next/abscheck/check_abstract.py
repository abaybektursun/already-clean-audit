"""Recompute every number of studies/next/20_ABSTRACT.md from its artifacts (numpy-free; json only).

Run from the worktree root:  python -I studies/next/abscheck/check_abstract.py
Prints one line per check: status, the text as written, the value from the artifact.
A planted wrong value is checked first; it must print MISMATCH, or the comparator is broken.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
N = ROOT / "studies/next"


def J(rel):
    return json.loads((N / rel).read_text())


rows = []


def chk(text, value, nd, art):
    """EQUAL when the artifact value rounded to nd decimals equals the number written."""
    got = round(value, nd)
    status = "EQUAL" if abs(got - text) < 0.5 * 10 ** (-nd) else "MISMATCH"
    rows.append((status, text, value, art))
    return status


def note(status, text, value, art):
    rows.append((status, text, value, art))


# --- the comparator must fire on a planted wrong value
assert chk(0.961, 0.9599593469315053, 3, "planted") == "MISMATCH", "comparator cannot fail"
rows.pop()

# --- M1
m1 = J("m1/verdict_m1.json")
sh = m1["exploratory"]["shift_from_start"]
for b, t in enumerate([-0.0113, -0.0170, -0.0173, -0.0161]):
    chk(t, sh["G"][b][0], 4, "m1 verdict exploratory.shift_from_start.G")
for b, t in enumerate([-0.0004, -0.0070, 0.0001, 0.0035]):
    chk(t, sh["M"][b][0], 4, "m1 verdict exploratory.shift_from_start.M")
for b, t in enumerate([0.0109, 0.0100, 0.0174, 0.0196]):
    chk(t, m1["lead"][b]["mean"], 4, "m1 verdict lead.mean")
for b, t in enumerate([0.0082, 0.0066, 0.0023, 0.0035]):
    chk(t, m1["bar"][b], 4, "m1 verdict bar")
C = m1["exploratory"]["C"]
lam = [(C["R"][b] - C["G"][b]) / (C["M"][b] - C["G"][b]) for b in range(4)]
for b, t in enumerate([0.70, 0.68, 0.66, 0.76]):
    chk(t, lam[b], 2, "lambda from m1 verdict exploratory.C")
note("DERIVED", "66-76 %", f"{100*min(lam):.2f}-{100*max(lam):.2f} %", "lambda")
g = [abs(sh["G"][b][0]) for b in range(4)]
note("DERIVED", "about one top-50 slot in fifty (0.02)",
     f"G-start {min(g):.4f}-{max(g):.4f}; slots of 50: {50*min(g):.2f}-{50*max(g):.2f}; "
     f"one slot in {1/max(g):.0f}-{1/min(g):.0f}; band mean {sum(g)/4:.4f} = one in {4/sum(g):.0f}",
     "m1 verdict exploratory.shift_from_start.G")
chk(1.0, 50 * max(g), 1, "abstract 'about one slot in fifty': the largest band's G-start shift in slots of 50")
chk(0.0, max(abs(x) for x in m1["G_ID"]["I"]), 6, "m1 verdict G_ID.I")

# --- M2
m2 = J("m2/verdict_m2.json")
hb = m2["labels"]["H_B"]
note("EQUAL" if (hb["hits"], hb["n"]) == (42, 71) else "MISMATCH", "42 of 71", (hb["hits"], hb["n"]), "m2 labels.H_B")
chk(0.592, hb["p"], 3, "m2 labels.H_B.p")
chk(0.475, hb["lo"], 3, "m2 labels.H_B.lo")
chk(0.698, hb["hi"], 3, "m2 labels.H_B.hi")
chk(0.0122, hb["bar"], 4, "m2 labels.H_B.bar")
chk(59, 100 * hb["p"], 0, "abstract 59 % = 42/71")
chk(0.05, 100 * 10 / 22016, 2, "abstract chance 0.05 % = 10/22,016")
pb = m2["test"]["pb_pre_B"]
note("EQUAL" if pb[3] == 53 and m2["test"]["n_pb"] == 70 else "MISMATCH", "53 of 70", (pb[3], m2["test"]["n_pb"]), "m2 test.pb_pre_B")
chk(0.757, pb[0], 3, "m2 test.pb_pre_B")
chk(76, 100 * pb[0], 0, "abstract 76 %")
s = m2["labels"]["H_sel_B"]
for t, v in zip([-0.171, -0.286, -0.057], [s["diff"], s["lo"], s["hi"]]):
    chk(t, v, 3, "m2 labels.H_sel_B")
r5 = J("m2/read/r5b_token_share.json")["both"]
for t, v in zip([-0.014, -0.113, 0.085], [r5["real_share_ge_0.67"]["diff"], *r5["real_share_ge_0.67"]["diff_ci"]]):
    chk(t, v, 3, "m2 read/r5b both.real_share_ge_0.67 (varied)")
for t, v in zip([-0.397, -0.513, -0.282], [r5["real_share_lt_0.67"]["diff"], *r5["real_share_lt_0.67"]["diff_ci"]]):
    chk(t, v, 3, "m2 read/r5b both.real_share_lt_0.67 (repeated)")

# --- N12
n12 = J("n12/verdict_n12.json")
cf = n12["labels"]["H_cf"]
for t, v in zip([0.951, 0.738, 1.180], [cf["mean"], cf["lo"], cf["hi"]]):
    chk(t, v, 3, "n12 labels.H_cf")
chk(0.77, n12["test"]["F_pb"]["mean"], 2, "n12 test.F_pb")
chk(0.94, n12["test"]["F_dt"]["mean"], 2, "n12 test.F_dt")
chk(0.939, n12["test"]["drop_real"]["mean"], 3, "n12 test.drop_real")
chk(1.164, n12["test"]["drop_pb"]["mean"], 3, "n12 test.drop_pb")
a = n12["labels"]["H_away_sel_pb"]
for t, v in zip([-0.226, -0.408, -0.056], [a["diff"], a["lo"], a["hi"]]):
    chk(t, v, 3, "n12 labels.H_away_sel_pb")
hit = n12["test"]["hits"]
note("EQUAL" if round(hit["dt"]["pre"][0] * 67) == 41 else "MISMATCH", "dt 41 of 67", hit["dt"]["pre"][0] * 67, "n12 test.hits.dt.pre")
d = n12["labels"]["H_sel_dt"]
for t, v in zip([-0.030, -0.149, 0.090], [d["diff"], d["lo"], d["hi"]]):
    chk(t, v, 3, "n12 labels.H_sel_dt")
note("EQUAL" if "freeze rho 0.100, test rho 0.043" in n12["labels"]["H_signal"]["reason"] else "MISMATCH",
     "G-VAL bf16 0.100 freeze, 0.043 test", n12["labels"]["H_signal"]["reason"], "n12 labels.H_signal.reason")

# --- N2V
n2v = J("n2v/verdict_n2v.json")
w = n2v["test"]["window"]["validation"]
for t, v in zip([0.960, 0.915, 0.990, 0.118], [w["rho"], w["lo"], w["hi"], w["shuffle_q975"]]):
    chk(t, v, 3, "n2v test.window.validation")
note("EQUAL" if (w["n_pairs"], w["n_units"]) == (284, 71) else "MISMATCH", "284 pairs, 71 units", (w["n_pairs"], w["n_units"]), "n2v")
note("NOTE", "the instrument's prediction", w["prediction"] + " against " + w["measurement"], "n2v test.window.validation")
fz = J("n2v/thresholds.json")["freeze"]["window"]["validation"]
for t, v in zip([0.971, 0.939, 0.993], [fz["rho"], fz["lo"], fz["hi"]]):
    chk(t, v, 3, "n2v thresholds freeze.window.validation")
for key, ts in [("fd", [0.987, 0.971, 0.995]), ("own", [0.986, 0.969, 0.996]), ("window_jvp", [0.967, 0.923, 0.994])]:
    v = n2v["test"][key]["validation"]
    for t, x in zip(ts, [v["rho"], v["lo"], v["hi"]]):
        chk(t, x, 3, f"n2v test.{key}.validation")
r6 = J("n2v/read_n2v/r6_checks.json")
aw = r6["a_within_unit"]["test"]["window"]
tm = aw["top_abs_pole_match"]
note("EQUAL" if round(tm["mean"] * 71) == 65 else "MISMATCH", "65 of 71 (top pole)", tm["mean"] * 71, "r6 a.test.window.top_abs_pole_match")
for t, v in zip([0.915, 0.845, 0.972], [tm["mean"], tm["lo"], tm["hi"]]):
    chk(t, v, 3, "r6 a.test.window.top_abs_pole_match")
note("DERIVED", "units whose four poles are ordered as zero ablation orders them",
     f"signed Kendall tau = 1 in {aw['kendall_signed']['share_tau_1']*71:.0f} of 71; "
     f"absolute in {aw['kendall_abs']['share_tau_1']*71:.0f} of 71", "r6 a.test.window.kendall_*.share_tau_1")
chk(65, aw["kendall_signed"]["share_tau_1"] * 71, 0, "abstract 'orders a unit's poles as zero ablation does in 65 of 71'")
chk(0.286, aw["within_unit_perm"]["rho_pairs_median"], 3, "r6 a.test.window.within_unit_perm")
lin = r6["c_linearity"]
dec = lin["freeze"]["dL_over_pred_bf16_by_abs_pred_decile"] + lin["test"]["dL_over_pred_bf16_by_abs_pred_decile"]
med = [x["median"] for x in dec]
note("DERIVED", "dL/pred 0.94-1.06 by decile", f"{min(med):.3f}-{max(med):.3f} (both halves)", "r6 c")
chk(0.988, lin["test"]["slope_fd_over_0.1"], 3, "r6 c.test.slope_fd_over_0.1")
chk(1.008, lin["freeze"]["slope_fd_over_0.1"], 3, "r6 c.freeze.slope_fd_over_0.1")
so = [lin[h]["second_order_share_median_abs_b_over_abs_dL"] for h in ("test", "freeze")]
note("DERIVED", "second-order share 0.18-0.20", f"{so[0]:.4f} test, {so[1]:.4f} freeze", "r6 c")
rr = n2v["n12_reread"]["labels"]
for t, v in zip([0.527, 0.451, 0.600, 0.4225], [rr["H_signal"]["mean"], rr["H_signal"]["lo"], rr["H_signal"]["hi"], rr["H_signal"]["bar"]]):
    chk(t, v, 4 if t == 0.4225 else 3, "n2v n12_reread.H_signal")
for k, ts in [("H_sel_signal_pb", [-0.039, -0.134, 0.057]), ("H_sel_signal_dt", [-0.020, -0.124, 0.084])]:
    for t, v in zip(ts, [rr[k]["diff"], rr[k]["lo"], rr[k]["hi"]]):
        chk(t, v, 3, f"n2v n12_reread.{k}")
for t, v in zip([-0.013, -0.072, 0.046], [rr["H_bridge"]["mean"], rr["H_bridge"]["lo"], rr["H_bridge"]["hi"]]):
    chk(t, v, 3, "n2v n12_reread.H_bridge")

# --- N1
n1 = J("n1/verdict_n1.json")
p = n1["test"]["pair_in_real_dt"]
for t, v in zip([-0.432, -0.817, -0.059], [p["diff"], p["lo"], p["hi"]]):
    chk(t, v, 3, "n1 test.pair_in_real_dt")
for t, v in zip([-0.43, -0.82, -0.06], [p["diff"], p["lo"], p["hi"]]):
    chk(t, v, 2, "n1 test.pair_in_real_dt (abstract)")
note("EQUAL" if (p["n"], p["x_only"], p["y_only"]) == (67, 23, 44) else "MISMATCH", "67 pairs, 23 above, 44 below",
     (p["n"], p["x_only"], p["y_only"]), "n1 test.pair_in_real_dt")
fr = J("n1/read_n1/fragility.json")["test"]
note("DERIVED", "margin two births", f"drop-one flips {fr['drop_one_in']['n_births_whose_removal_flips_BELOW_to_AT_pct']}; "
     f"drop two most negative: upper {fr['drop_extremes_in']['drop_2_most_negative']['pct'][1]:+.3f}", "n1 read_n1/fragility.json")
disc = J("n1/thresholds.json")["discovery"]["pair_in_real_dt"]
for t, v in zip([-0.217, -0.638, 0.204], [disc["diff"], disc["lo"], disc["hi"]]):
    chk(t, v, 3, "n1 thresholds.json discovery.pair_in_real_dt (freeze half)")
pool = J("n1/read_n1/verdict_recount.json")["confounds"]["c_halves"]["in"]["pooled_both_halves"]
for t, v in zip([-0.320, -0.604, -0.035], [pool["mean"], *pool["pct_seed0"]]):
    chk(t, v, 3, "n1 read_n1/verdict_recount.json confounds.c_halves.in.pooled_both_halves")
o = n1["test"]["pair_out_real_dt"]
for t, v in zip([-0.258, -0.654, 0.130], [o["diff"], o["lo"], o["hi"]]):
    chk(t, v, 3, "n1 test.pair_out_real_dt")
L = n1["labels"]
for t, v in zip([-0.297, -0.550, -0.059, -0.053], [L["floor_in"]["mean"], L["floor_in"]["lo"], L["floor_in"]["hi"], L["floor_in"]["bar"]]):
    chk(t, v, 3, "n1 labels.floor_in")
for t, v in zip([-0.039, -0.276, 0.202, 0.147], [L["floor_out"]["mean"], L["floor_out"]["lo"], L["floor_out"]["hi"], L["floor_out"]["bar"]]):
    chk(t, v, 3, "n1 labels.floor_out")
sc = J("n1/read_n1/scale.json")["test_pair_mean_log_r_window_in"]["as_relative"]
for t, v in zip([-1.6, -3.2, 0.03], sc):
    chk(t, 100 * v, 1 if t != 0.03 else 2, "n1 read_n1/scale.json test_pair_mean_log_r_window_in.as_relative")

# --- N13
v13 = J("n13/verdict_n13.json")
th = J("n13/thresholds.json")
A = v13["arms"]
for t, k in [(57.2, "B"), (92.0, "L"), (55.4, "B_prime"), (91.2, "L_prime"), (34.8, "G"), (74.6, "H")]:
    chk(t, th[k], 1, f"n13 thresholds.{k}")
chk(13.60, th["room"]["bound_2.5delta"], 2, "n13 thresholds.room.bound_2.5delta")
chk(1.924, th["sigma"], 3, "n13 thresholds.sigma")
chk(39.96, v13["gates"]["G-KNOWN"]["value"], 2, "n13 verdict gates.G-KNOWN")
for arm, t in [("top_k0.1", 57.6), ("top_k0.3", 61.4), ("top_k1", 61.8), ("top_k3", 63.8),
               ("rndE_k0.1", 58.8), ("rndE_k0.3", 59.8), ("rndE_k1", 59.2), ("rndE_k3", 63.8),
               ("rndT_k0.1", 59.4), ("rndT_k0.3", 58.0), ("rndT_k1", 60.0), ("rndT_k3", 62.8),
               ("dare_k0.1", 85.2), ("dare_k0.3", 86.8), ("dare_k1", 85.8), ("dare_k3", 86.2),
               ("mlp", 86.4), ("attn", 83.0), ("comp_k0.1", 90.4), ("comp_k0.3", 90.8), ("comp_k1", 91.6), ("comp_k3", 90.6)]:
    chk(t, A[arm]["acc"], 1, f"n13 arms.{arm}.acc ({A[arm]['label']})")
cm = v13["comparisons"]
ctrl = [cm[f"top_k{k}|{c}_k{k}"]["diff"] for k in ("0.1", "0.3", "1", "3") for c in ("rndE", "rndT")]
note("DERIVED", "top - control -1.2 to +3.4", f"{min(ctrl):+.1f} to {max(ctrl):+.1f}", "n13 comparisons top_k*|rndE_k*, rndT_k*")
chk(-1.2, min(ctrl), 1, "table row N13 energy-matched: low end of top - control")
chk(3.4, max(ctrl), 1, "table row N13 energy-matched: high end of top - control")
bh_ctrl = min(cm[f"top_k{k}|{c}_k{k}"]["bh_p"] for k in ("0.1", "0.3", "1", "3") for c in ("rndE", "rndT"))
chk(0.107, bh_ctrl, 3, "n13 smallest BH p, top vs rndE/rndT")
dd = [cm[f"top_k{k}|dare_k{k}"]["diff"] for k in ("0.1", "0.3", "1", "3")]
note("DERIVED", "top - dare -27.6 to -22.4", f"{min(dd):+.1f} to {max(dd):+.1f}", "n13 comparisons top_k*|dare_k*")
note("DERIVED", "BH p <= 2.6e-21", f"max {max(cm[f'top_k{k}|dare_k{k}']['bh_p'] for k in ('0.1','0.3','1','3')):.3e}", "n13 comparisons")
chk(3.4, cm["mlp|attn"]["diff"], 1, "n13 comparisons mlp|attn")
chk(0.107, cm["mlp|attn"]["bh_p"], 3, "n13 comparisons mlp|attn bh_p")
ck = J("n13/read_n13/checks_read.json")
dm = [x["dare_minus_mlp"] for x in ck["j_other"]["dare_vs_mlp"].values()]
note("DERIVED", "dare - mlp within 1.2", f"{min(dm):+.1f} to {max(dm):+.1f}; dare_k1_d2 - mlp {A['dare_k1_d2']['acc']-A['mlp']['acc']:+.1f}",
     "n13 checks_read j_other.dare_vs_mlp")
ratio = [ck["d_matching"][k]["dare"]["over_full"] for k in ("k0.1", "k0.3", "k1", "k3")]
note("DERIVED", "DARE 33-1,000 times the change's energy", ", ".join(f"{r:.1f}" for r in ratio), "n13 checks_read d_matching.*.dare.over_full")
gs = v13["gsm8k"]
for key, ts in [("top_k1|dare_k1", [-23.7, -26.5, -21.1]), ("top_k1|rndE_k1", [2.1, 0.2, 3.9]), ("top_k1|rndT_k1", [1.9, 0.1, 3.6])]:
    for t, x in zip(ts, [gs[key]["diff_gsm8k"], *gs[key]["ci"]]):
        chk(t, x, 1, f"n13 gsm8k.{key}")
gi = ck["i_gsm8k"]
chk(89.8, gi["dare_k1"]["acc"], 1, "n13 checks_read i_gsm8k.dare_k1 (q = 1 %, not 0.1 %)")
chk(92.3, gi["rl"]["acc"], 1, "n13 checks_read i_gsm8k.rl")
note("NOTE", "GSM8K arms run", ", ".join(sorted(gi)), "n13 checks_read i_gsm8k (no dare_k0.1)")
note("EQUAL" if "dare_k0.1" in gi else "MISMATCH", "next step: DARE's 0.1 % sample reads 89.8 on GSM8K",
     "dare_k0.1 has a GSM8K reading" if "dare_k0.1" in gi else f"no GSM8K reading for dare_k0.1; 89.8 is dare_k1 ({gi['dare_k1']['acc']:.3f})",
     "n13 checks_read i_gsm8k")
hy = v13["hypotheses"]
for k in ("H_local", "H_diffuse", "H_attention", "H_artifact"):
    note("LABEL", k, hy[k], "n13 hypotheses")
note("LABEL", "necessity", sorted(set(hy["necessity"].values())), "n13 hypotheses.necessity")
chk(0.642, hy["H_artifact_values"]["J_floor_k1"], 3, "n13 H_artifact_values.J_floor_k1")
chk(0.534, hy["H_artifact_values"]["J_round_k1"], 3, "n13 H_artifact_values.J_round_k1")

# --- N13 per-arm rate, from the banked S4 arm records and the operations note's launch-to-STATUS GPU-h
walls = {}
for ar in sorted((N / "n13/s4").glob("*/*/arm_record.json")):
    d = json.loads(ar.read_text())
    b = d["bench"]
    walls[(d["prefix"], d["step"])] = (next(iter(b.values()))["wall_min"], "gsm8k" in b)
math_only = sorted(w for w, gsm in walls.values() if not gsm)
with_gsm = sorted(w for w, gsm in walls.values() if gsm)
note("DERIVED", "S4 step wall, MATH-500 only (h)",
     f"n {len(math_only)}, {math_only[0]/60:.2f}-{math_only[-1]/60:.2f}, median {math_only[len(math_only)//2]/60:.2f}", "n13 s4 */*/arm_record.json")
note("DERIVED", "S4 step wall, MATH-500 + GSM8K (h)",
     f"n {len(with_gsm)}, {with_gsm[0]/60:.2f}-{with_gsm[-1]/60:.2f}", "n13 s4 */*/arm_record.json")
launch_h = {"s4armsA2-01": 21.482, "s4armsA1-01": 20.647, "s4armsB1-02": 19.204, "s4armsB2-02": 11.208,
            "s4armsB2-01": 3.705, "s4armsB1-01": 0.265}  # infra/aws_pg/operations/2026-11-06_next_n13_s4.md, final table
chk(76.511, sum(launch_h.values()), 3, "operations note S4 total")
for pre, h in launch_h.items():
    n_arms = sum(1 for (p_, _s) in walls if p_ == pre)
    if n_arms:
        sw = sum(w for (p_, _s), (w, _g) in walls.items() if p_ == pre) / 60
        note("DERIVED", f"{pre} launch-to-STATUS per arm (h)",
             f"{h/n_arms:.2f} over {n_arms} arms ({sum(1 for (p_, _s), (_w, g_) in walls.items() if p_ == pre and g_)} with GSM8K); "
             f"overhead beyond step walls {h - sw:.2f} h", "operations note + arm records")
note("DERIVED", "S4 GPU-h per arm, all launches", f"{sum(launch_h.values())/len(walls):.2f} over {len(walls)} arms", "operations note + arm records")

# --- draft 4 (clean-neurons 95df5fa58): the lines above check draft 3's numbers; these check the new ones
note("DRAFT4", "---", "---", "draft 4 checks start")
gs50 = [50 * x for x in g]
ms50 = [50 * abs(sh["M"][b][0]) for b in range(4)]
note("DERIVED", "draft 4: G moved 0.6-0.9 of a slot (per pole)", ", ".join(f"{x:.4f}" for x in gs50), "m1 verdict shift_from_start.G x 50")
chk(0.6, min(gs50), 1, "draft 4: low end of G shift in slots of 50")
chk(0.9, max(gs50), 1, "draft 4: high end of G shift in slots of 50")
note("DERIVED", "draft 4: M moved at most 0.35 of a slot", ", ".join(f"{x:.4f}" for x in ms50), "m1 verdict shift_from_start.M x 50")
chk(0.35, max(ms50), 2, "draft 4: largest |M shift| in slots of 50")
note("EQUAL" if w["n_pairs"] == 284 and w["measurement"] == "dL" else "MISMATCH", "draft 4: ranks 284 fp32 zero ablations",
     f"n_pairs {w['n_pairs']}, measurement {w['measurement']} (fp32), prediction {w['prediction']}", "n2v test.window.validation")
fr_lab = "AT" if disc["lo"] < 0 < disc["hi"] else "not AT"
note("EQUAL" if fr_lab == "AT" else "MISMATCH", "draft 4: the freeze half alone at selection level",
     f"{disc['diff']:.3f} [{disc['lo']:.3f}, {disc['hi']:.3f}] -> {fr_lab}", "n1 thresholds.json discovery.pair_in_real_dt")
comp = [A[f"comp_k{k}"]["acc"] for k in ("0.1", "0.3", "1", "3")]
chk(90.4, min(comp), 1, "draft 4: without its top sets, low end")
chk(91.6, max(comp), 1, "draft 4: without its top sets, high end")
subsets = {a: A[a]["label"] for a in A if a.startswith(("top_", "rndE_", "rndT_", "rnd0_", "topF_"))}
note("DERIVED", "draft 4: 'No neuron subset carries the gain'",
     f"0.1-3 % subsets: {sorted(set(subsets.values()))}; complements comp_k: {[A[f'comp_k{k}']['label'] for k in ('0.1','0.3','1','3')]} "
     f"({min(comp)}-{max(comp)}); mlp {A['mlp']['label']}", "n13 arms")
note("EQUAL" if round(aw["kendall_signed"]["share_tau_1"] * 71) == 60 else "MISMATCH", "draft 4: full signed order 60 of 71",
     aw["kendall_signed"]["share_tau_1"] * 71, "r6 a.test.window.kendall_signed.share_tau_1")
chk(-1.8, min(ctrl), 1, "draft 4: top - control low end")
mo, gm = [m / 60 for m in math_only], [m / 60 for m in with_gsm]
chk(2.0, mo[0], 1, "draft 4: MATH-500 arm, low end (h)")
chk(2.6, mo[-1], 1, "draft 4: MATH-500 arm, high end (h)")
chk(3.4, gm[0], 1, "draft 4: arm with GSM8K, low end (h)")
chk(4.8, gm[-1], 1, "draft 4: arm with GSM8K, high end (h)")
note("DERIVED", "draft 4: six arms at the stated 2.0-2.6", f"{6*2.0:.1f}-{6*2.6:.1f}; at the banked walls {6*mo[0]:.1f}-{6*mo[-1]:.1f}", "6 x rate")
chk(13, 6 * 2.0, 0, "draft 4: six arms, low end of '13-16 GPU-h' at the stated rate")
chk(16, 6 * 2.6, 0, "draft 4: six arms, high end of '13-16 GPU-h' at the stated rate")
note("DERIVED", "draft 4: GSM8K on the rank-64 pair",
     f"stated rates {4*2.0+2*3.4:.1f}-{4*2.6+2*4.8:.1f}; banked walls {4*mo[0]+2*gm[0]:.1f}-{4*mo[-1]+2*gm[-1]:.1f}", "4 x MATH-500 arm + 2 x GSM8K arm")
chk(89.8, gi["dare_k1"]["acc"], 1, "draft 4: the 1 % sample on GSM8K")

# --- draft 5 (clean-neurons 2497fbb71)
note("DRAFT5", "---", "---", "draft 5 checks start")
note("EQUAL" if max(ms50) <= 0.36 else "MISMATCH", "draft 5: M moved at most 0.36 of a slot", f"max {max(ms50):.4f}", "m1 verdict shift_from_start.M x 50")
chk(12, 6 * 2.0, 0, "draft 5: six arms, low end of '12-16 GPU-h'")
chk(16, 6 * 2.6, 0, "draft 5: six arms, high end of '12-16 GPU-h'")
chk(15, 4 * 2.0 + 2 * 3.4, 0, "draft 5: low end of '15-20 with GSM8K on the rank-64 pair'")
chk(20, 4 * 2.6 + 2 * 4.8, 0, "draft 5: high end of '15-20 with GSM8K on the rank-64 pair'")
jv = r6["a_within_unit"]["test"]["window_jvp"]
chk(0.915, jv["top_abs_pole_match"]["mean"], 3, "draft 5 n2v/READ_head: fp32 forward-mode pass (window_jvp) names the same largest-effect pole")
chk(0.845, jv["kendall_signed"]["share_tau_1"], 3, "draft 5 n2v/READ_head: fp32 forward-mode pass (window_jvp) full signed order")
note("NOTE", "0.845 is the bf16 prediction's readout", f"window share_tau_1 {aw['kendall_signed']['share_tau_1']:.4f}", "r6 a.test.window")

# --- the abstract text: word count and banned words
text = (N / "20_ABSTRACT.md").read_text()
abstract = text.split("## Abstract", 1)[1].split("## Evidence table", 1)[0]
note("COUNT", "abstract words (len(text.split()))", len(abstract.split()), "20_ABSTRACT.md")
for word in (w[::-1] for w in ("diov", "regdel", "llun")):  # stored reversed so this file holds none of them
    hits = [i + 1 for i, line in enumerate(text.splitlines()) if re.search(word, line, re.I)]
    note("BANNED", f"banned word {len(word)} letters", hits, "20_ABSTRACT.md lines")
# the scan must fire on a planted line
assert re.search("diov"[::-1], "a planted " + "diov"[::-1], re.I)

for r in rows:
    print("\t".join(str(x) for x in r))
bad = [r for r in rows if r[0] == "MISMATCH"]
print(f"\n{len(rows)} lines; {len(bad)} MISMATCH", file=sys.stderr)
