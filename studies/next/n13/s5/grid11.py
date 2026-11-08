"""§11 tables from verdict_n13.json only. usage: python3 -I grid11.py <verdict_n13.json>"""
import json, math, sys
V = json.load(open(sys.argv[1]))
g, T, A, H, C = V["gates"], V["thresholds"], V["arms"], V["hypotheses"], V["comparisons"]
tn = V["three_numbers"]
f1 = lambda x: f"{x:.1f}"
def arm(k):
    r = A[k]; w = r["wilson"]
    return f"| {k} | {f1(r['acc'])} | [{f1(w[0])}, {f1(w[1])}] | {r['label'] or '-'} | {r['cap_hit_share']:.2f} | {r['answer_line_share']:.2f} | {r['mean_tokens']:.0f} |"
hv = H["H_artifact_values"]
gs = V["gsm8k"]
gsm = "; ".join(f"{k.replace('|', ' − ')}: {v['diff_gsm8k']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}] {v['label']}" for k, v in gs.items())
rows = [
 ("G-INIT", "`main` is the init (pair 1: hash check against `39ff1a89…` and the share comparison)",
  f"pass; share difference {g['G-INIT']['value']:.3f}; failing input failed", "PASS"),
 ("G-KNOWN", "base Minerva MATH (`minerva_math::olmes:n4:v2`, `primary_score`) within 1.6 points of 40.0",
  f"{g['G-KNOWN']['value']:.2f} (failing input {g['G-KNOWN']['failing_value']:.2f} failed)", "PASS"),
 ("S-BAND (pair 1)", "L' within [81.5, 96.7] (prior: L 80 to 92)", f"L' {f1(g['S-BAND']['value'])}; L {f1(T['L'])}", "PASS"),
 ("G-GATE", "G ≥ 5√2σ (prior: G 15 to 35 points)", f"G {f1(T['G'])} ≥ {tn['positive_control']['bound_5sqrt2sigma']:.2f} (σ {T['sigma']:.3f})", "PASS"),
 ("G-ID", "\\|L − L'\\| ≤ 3√2σ", f"{tn['fit_both_sides']['L_minus_Lprime']:.1f} ≤ {tn['fit_both_sides']['bound_3sqrt2sigma']:.2f}; B − B' {tn['identity_pair']['B_minus_Bprime']:.1f}", "PASS"),
 ("unchanged share of MLP entries", "prior 0.4 to 0.9", "not in `verdict_n13.json` (R1 `weight_stats.json`); not filled here", "—"),
 ("H_artifact", "J_floor, J_round at 1%", f"J_floor {hv['J_floor_k1']:.3f}, J_round {hv['J_round_k1']:.3f}; PC-ART rule(P+) {hv['rule_P_plus']}, rule(P−) {hv['rule_P_minus']}", H["H_artifact"]),
 ("k_rec", "", "NONE: no top_k labeled REC or ABOVE", "NONE"),
 ("H_local", "§1.3", "k_rec NONE", H["H_local"]),
 ("H_diffuse", "§1.3", f"dare_k REC at every k; top − dare {', '.join(f1(C[f'top_{k}|dare_{k}']['diff']) for k in ('k0.1','k0.3','k1','k3'))} (k 0.1, 0.3, 1, 3)", H["H_diffuse"]),
 ("H_attention", "§1.3", f"mlp {f1(A['mlp']['acc'])} {A['mlp']['label']}, attn {f1(A['attn']['acc'])} {A['attn']['label']}", H["H_attention"]),
 ("necessity at k_rec", "", "k_rec NONE; at all four k: " + ", ".join(f"{k} {v}" for k, v in H["necessity"].items()), "NOT NECESSARY (all k)"),
 ("GSM8K at 1%", "", f"G-GATE-GSM {f1(g['G-GATE-GSM']['value'])}, pass; {gsm}", "REPLICATES (3 of 3)"),
]
out = ["| item | prediction or rule | observed | verdict |", "|---|---|---|---|"]
out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
out += ["", "| arm | acc (%) | Wilson 95% | label | cap-hit share | `answer_line` share | mean tokens |", "|---|---|---|---|---|---|---|"]
order = ["base", "rl", "zero", "id"] + [f"{t}_{k}" for t in ("top", "comp", "rndE", "rndT", "dare") for k in ("k0.1", "k0.3", "k1", "k3")] + \
        ["mlp", "attn", "rnd0_k1", "top_k1_B", "rndT_k1_d2", "dare_k1_d2", "topF_k1"]
assert set(order) == set(A), set(A) ^ set(order)
out += [arm(k) for k in order]
print("\n".join(out))
