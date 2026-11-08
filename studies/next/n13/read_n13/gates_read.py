"""N13 independent read, step 1: every gate of N13_SPEC.md 6.1 with its value, pass rule and the registered failing input's value,
from the banks (S3 prefix results/next_n13/s5-01/banks/), the R0 and R1 records and the S4 arm records in the worktree; G-HASH and
G-MASK also by the reader's own comparison of the per-tensor sha256 and mask audits in build.json; A31 on the four rndE arms; the
engine blocks against S3's; the three numbers of 5.

    python -I gates_read.py <banks dir> <worktree> <recount_read.json> <out.json>

Imports only the planted grader pairs (data) from infra.pg.n13_score. Fail-loud: no try/except.
"""
import json
import math
import pathlib
import re
import sys
from fractions import Fraction

import numpy as np

BANKS, WT, RC, OUT = (pathlib.Path(x) for x in sys.argv[1:5])
sys.path.insert(0, str(WT))
from infra.pg.n13_score import PLANTED_EQ, PLANTED_NEQ  # noqa: E402  (data only)
from math_verify import parse, verify  # noqa: E402

I = BANKS / "_inputs"
jl = lambda p: json.loads(pathlib.Path(p).read_text())
T, C, K = jl(I / "thresholds.json"), jl(I / "controls.json"), jl(I / "r0_known/metrics.json")
GR, TP, SETS = jl(I / "setup/grader.json"), jl(I / "setup/template.json"), jl(I / "sets.json")
RCJ = jl(RC)["arms"]
R2 = math.sqrt(2)
G = {}


def wrap(s):
    return s if (re.search(r"(?<!\\)\$", s) or "\\boxed" in s) else "$" + s + "$"


def grade(g, x):
    return 0 if x is None else int(bool(verify(parse(wrap(g)), parse(wrap(x)))))


def recs(arm, bench):
    p = BANKS / arm / arm / bench / "generations.jsonl"
    r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return sorted(r, key=lambda q: q["problem_index"])


def derangement(n, seed=0):
    # the construction stated in the spec's code docstring: a uniform random cycle order p, each p[i] mapped to p[i+1]
    p = np.random.default_rng(seed).permutation(n)
    d = np.empty(n, dtype=np.int64)
    d[p] = np.roll(p, -1)
    assert (d != np.arange(n)).all()
    return d


def norm(s):
    return re.sub(r"\s+", "", s.replace("\\left", "").replace("\\right", ""))


def vec(arm, bench="math500"):
    return np.array([int(c) for c in RCJ[f"{arm}/{bench}"]["scores_banked"]])


def mcnemar_p(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), 2 * Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)), 2 ** n)))


def sig(xa, xb):
    return 100 * math.sqrt(float(((xa - xb) ** 2).sum()) / 2) / len(xa)


# ---- G-GRADER (recomputed on the banked MATH-500 golds)
base_m = recs("base", "math500")
golds = [r["gold"] for r in base_m]
eq = sum(grade(g, x) for g, x in PLANTED_EQ)
neq_false = sum(1 - grade(g, x) for g, x in PLANTED_NEQ)
self_eq = sum(grade(g, g) for g in golds)
d = derangement(500, 0)
der = [grade(golds[i], golds[d[i]]) for i in range(500)]
ident = float(np.mean([norm(golds[i]) == norm(golds[d[i]]) for i in range(500)]))
rate = sum(der) / 500
G["G-GRADER"] = {"rule": "24/24 planted equivalent true, 16/16 non-equivalent false, 500/500 golds self-equal, derangement rate <= "
                         "identical-normalised rate + 0.01",
                 "mine": {"planted_eq": eq, "planted_neq_false": neq_false, "self_equal": self_eq, "derangement_rate": rate,
                          "bound": ident + 0.01, "pass": eq == 24 and neq_false == 16 and self_eq == 500 and rate <= ident + 0.01},
                 "failing_input": {"what": "the derangement scored as the identity (needs 500/500)", "value": sum(der), "failed": sum(der) != 500},
                 "banked": {k: GR["rules"][GR["active_rule"]][k] for k in ("planted_eq_true", "planted_neq_false", "self_equal",
                                                                          "derangement_rate", "derangement_bound", "pass")},
                 "banked_failing_value": GR["rules"][GR["active_rule"]]["failing_input"]["value"], "active_rule": GR["active_rule"]}

# ---- G-TEMPLATE
PRE = ("Solve the following problem step by step. The last line of your response should be the answer to the problem in form Answer: "
       "$Answer (without quotes) where $Answer is the answer to the problem.\n\n")
SUF = "\n\nRemember to put your answer on its own line after \"Answer:\""
shape_ok, arm_tpl = {}, {}
for gen in sorted(BANKS.rglob("generations.jsonl")):
    arm, bench = gen.parent.parent.name, gen.parent.name
    rr = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
    shape_ok[f"{arm}/{bench}"] = all(r["prompt"].startswith(PRE) and r["prompt"].endswith(SUF) and len(r["prompt"]) > len(PRE) + len(SUF)
                                     for r in rr)
    m = jl(gen.parent / "metrics.json")
    arm_tpl[f"{arm}/{bench}"] = {"pass": m["template"]["pass"], "n": m["template"]["n"], "bos": m["template"]["first_id_is_bos"]}
G["G-TEMPLATE"] = {"rule": "rendered prompt = the 2.2 string with the problem substituted, every problem; ids identical in every arm",
                   "banked_setup": {"pass": TP["pass"], "n": TP["n"], "ids_sha256": TP["ids_sha256"], "first_id_is_bos": TP["first_id_is_bos"],
                                    "chat_template_sha256": TP["chat_template_sha256"]},
                   "per_arm_metrics_all_pass": all(v["pass"] for v in arm_tpl.values()),
                   "per_arm_n": sorted({v["n"] for v in arm_tpl.values()}),
                   "mine_prompt_has_2.2_prefix_and_suffix_all": all(shape_ok.values()),
                   "mine_prompt_text_and_ids_identical_across_arms": jl(RC)["consistency"]["prompts_consistent_across_arms"],
                   "failing_input": {"what": "render with add_generation_prompt=False", "failed": TP["failing_input"]["failed"],
                                     "tail": TP["failing_input"]["tail"][-40:]}}

# ---- R1 gates (controls.json) and R0
gi = C["G-INIT"]
G["G-INIT"] = {"rule": "(a) base main shards = LFS oids at 39ff1a89; (b) unchanged-MLP share(main) - share(stage2) >= 0.10",
               "a": gi["a"]["pass"], "b_share_main": gi["b"]["share_main"], "b_share_other": gi["b"]["share_other"],
               "b_difference": gi["b"]["difference"], "b_check": gi["b"]["difference"] >= 0.10,
               "failing_input": {"a_failed": gi["a"]["failing_input"]["failed"], "b_failed": gi["b"]["failing_input"]["failed"],
                                 "b_swapped_value": gi["b"]["share_other"] - gi["b"]["share_main"]}, "pass": gi["pass"]}
gp = C["G-PARTITION"]
G["G-PARTITION"] = {"rule": "355 names cover every tensor; families sum to 7,298,011,136", "per_family": gp["per_family"],
                    "sum_mine": sum(gp["per_family"].values()), "sum_ok": sum(gp["per_family"].values()) == 7298011136,
                    "matches_spec_0.2": gp["per_family"] == {"mlp": 4328521728, "attn": 2147483648, "norm": 528384, "embed": 821477376},
                    "n_names": gp["n_names"], "pass": gp["pass"], "failing_input": gp["failing_input"]}
pr = C["PC-RANK"]
G["PC-RANK"] = {"rule": "recovery >= 0.99; failing input <= 0.05", "value": pr["recovery"], "pass_mine": pr["recovery"] >= 0.99,
                "failing_value": pr["failing_input"]["value"], "failing_failed_mine": pr["failing_input"]["value"] <= 0.05,
                "chance": pr["chance"], "pass": pr["pass"]}
gk = K["G-KNOWN"]
G["G-KNOWN"] = {"rule": "|100 x primary_score - 40.0| <= 1.6; failing input (deranged golds) < 10.0", "value": gk["value"],
                "pass_mine": abs(gk["value"] - 40.0) <= 1.6, "failing_value": gk["failing_input"]["value"],
                "failing_failed_mine": gk["failing_input"]["value"] < 10.0, "identity_rescore_pass": gk["identity"]["pass"], "pass": gk["pass"]}
pa = C["PC-ART"]
G["PC-ART"] = {"rule": "H_artifact rule SUPPORTED on P+ and REFUTED on P- (A2)", "P_plus_k1": {"J_floor": pa["P_plus"]["J_floor"]["k1"],
               "J_round": pa["P_plus"]["J_round"]["k1"], "rule": pa["rule_P_plus"]}, "P_minus_k1": {"J_floor": pa["P_minus"]["J_floor"]["k1"],
               "J_round": pa["P_minus"]["J_round"]["k1"], "rule": pa["rule_P_minus"]}, "pass": pa["pass"]}

# ---- G-HASH: controls, gate builds, arm records, and the reader's own family comparison against zero (= base) and id (= RL)
zero_h, id_h = jl(BANKS / "zero/zero/build.json")["sha256"], jl(BANKS / "id/id/build.json")["sha256"]
assert set(zero_h) == set(id_h) and len(zero_h) == 355
fam = lambda n: "mlp" if ".mlp." in n else ("norm" if "norm" in n else ("attn" if "self_attn" in n else "embed"))
assert {fam(n) for n in zero_h} == {"mlp", "norm", "attn", "embed"}
n_same_endpoints = sum(zero_h[n] == id_h[n] for n in zero_h)
arm_recs = {}
for p in sorted((WT / "studies/next/n13/s4").glob("*/*/arm_record.json")):
    r = jl(p)
    assert r["step"] not in arm_recs, r["step"]
    arm_recs[r["step"]] = {"prefix": r["prefix"], "rec": r}
fam_check = {}
for bj in sorted(BANKS.glob("*/*/build.json")):
    arm = bj.parent.name
    if arm in ("base", "rl", "zero", "id"):
        continue
    b = jl(bj)
    h = b["sha256"]
    rows = {}
    for f in ("mlp", "attn", "norm", "embed"):
        ns = [n for n in h if fam(n) == f]
        rows[f] = {"n": len(ns), "eq_base": sum(h[n] == zero_h[n] for n in ns), "eq_rl": sum(h[n] == id_h[n] for n in ns),
                   "eq_rl_not_base": sum(h[n] == id_h[n] != zero_h[n] for n in ns)}
    nonmlp = b["spec"].get("non_mlp")
    if arm == "attn":
        exp_ok = rows["mlp"]["eq_base"] == 96 and rows["embed"]["eq_base"] == rows["embed"]["n"] and \
            all(rows[f]["eq_rl"] == rows[f]["n"] for f in ("attn", "norm"))
    elif arm == "mlp":
        exp_ok = rows["mlp"]["eq_rl"] == 96 and all(rows[f]["eq_base"] == rows[f]["n"] for f in ("attn", "norm", "embed"))
    elif nonmlp == "before":
        exp_ok = all(rows[f]["eq_base"] == rows[f]["n"] for f in ("attn", "norm", "embed"))
    else:
        exp_ok = all(rows[f]["eq_rl"] == rows[f]["n"] for f in ("attn", "norm", "embed"))
    ar = arm_recs[arm]["rec"]["build"]["hashes_vs_R1_stage6"]
    fam_check[arm] = {"non_mlp": nonmlp, "families": rows, "only_the_changed_family_differs": bool(exp_ok),
                      "arm_record_vs_R1_stage6": f"{ar['n_equal']}/{ar['n_tensors']}", "R1_note": arm_recs[arm]["rec"]["build"].get("R1_stage6_note")}
G["G-HASH"] = {"rule": "zero = base and id = RL, every tensor sha256", "controls": {"zero": C["G-HASH"]["zero"]["pass"], "id": C["G-HASH"]["id"]["pass"],
               "failing_input_failed": C["G-HASH"]["failing_input"]["failed"]},
               "zero_build": jl(BANKS / "zero/zero/build.json")["G-HASH"], "id_build": jl(BANKS / "id/id/build.json")["G-HASH"],
               "tensors_equal_between_zero_and_id": n_same_endpoints, "per_arm": fam_check,
               "all_S4_only_their_family": all(v["only_the_changed_family_differs"] for v in fam_check.values())}

# ---- G-MASK: banked flag and the reader's comparison of realized rows/columns with S n {nonzero change}
setmap = {}
for k in ("k0.1", "k0.3", "k1", "k3"):
    setmap[f"top_{k}"] = set(SETS["top"][k])
    setmap[f"comp_{k}"] = set(range(352256)) - set(SETS["top"][k])
    setmap[f"rndE_{k}"] = set(SETS["rndE"][k]["neurons"])
    setmap[f"rndT_{k}"] = set(SETS["rndT"][k]["neurons"])
setmap.update({"top_k1_B": set(SETS["top"]["k1"]), "rnd0_k1": set(SETS["rnd0"]["neurons"]), "rndT_k1_d2": set(SETS["rndT_d2"]["neurons"]),
               "topF_k1": set(SETS["topF"])})
gm = {}
for arm, S in sorted(setmap.items()):
    b = jl(BANKS / arm / arm / "build.json")
    m = b["mask"]
    rows_eq = all(m[key] == m["expected"][key] for key in ("gate_rows", "up_rows", "down_cols"))
    realized = {x for key in ("gate_rows", "up_rows", "down_cols") for v in m[key].values() for x in v}
    gm[arm] = {"banked_pass": m.get("pass"), "realized_equals_expected": rows_eq, "realized_subset_of_S": realized <= S,
               "n_set": len(S), "n_neurons_in_set_banked": b["n_neurons_in_set"], "n_realized_changed": len(realized),
               "arm_record_G-MASK": arm_recs[arm]["rec"]["build"]["G-MASK"],
               "set_size_matches_spec": (352256 - len(S) if arm.startswith("comp_") else len(S)) == b["n_neurons_in_set"]}
G["G-MASK"] = {"rule": "realized changed rows and columns = S n {nonzero change}", "controls": C["G-MASK"], "per_arm": gm,
               "all_pass": all(v["banked_pass"] and v["realized_equals_expected"] and v["realized_subset_of_S"] and v["arm_record_G-MASK"]
                               and v["set_size_matches_spec"] for v in gm.values())}

# ---- G-COMPLETE (A25) from the recount; failing input = one record removed
gc = {}
for key, v in RCJ.items():
    c = v["complete"]
    n = 500 if v["bench"] == "math500" else 1319
    ok = c["n"] == n and c["covers"] and c["seeded"] and c["n_generated_ge1"] and c["finish_reason"]
    gc[key] = {"pass": ok, "n_empty": c["n_empty"], "failing_input_failed": (c["n"] - 1) != n}
G["G-COMPLETE"] = {"rule": "every index once, seeds as registered, n_generated >= 1, a finish_reason (A25)", "per_arm": gc,
                   "all_pass": all(v["pass"] for v in gc.values()), "n_arm_bench": len(gc),
                   "arm_records_failing_input_failed": all(b.get("G-COMPLETE_failing_input_failed") for a in arm_recs.values()
                                                           for b in a["rec"]["bench"].values())}

# ---- G-GATE, G-ID, S-BAND, G-GATE-GSM recomputed from the banked scores
def gate_block(bench):
    xb, xr, xz, xi = vec("base", bench), vec("rl", bench), vec("zero", bench), vec("id", bench)
    n = len(xb)
    s = max(sig(xb, xz), sig(xr, xi))
    a = lambda x: 100 * x.sum() / n
    Gv, Gp = a(xr) - a(xb), a(xi) - a(xz)
    p = mcnemar_p(xr, xb)
    out = {"sigma_base": sig(xb, xz), "sigma_rl": sig(xr, xi), "sigma": s, "B": a(xb), "L": a(xr), "B_prime": a(xz), "L_prime": a(xi),
           "G": Gv, "G_prime": Gp, "bound_5sqrt2sigma": 5 * R2 * s, "mcnemar_p": p,
           "G-GATE_pass": bool(Gv >= 5 * R2 * s and Gp >= 5 * R2 * s and p < 0.001)}
    fG, fGp, fp = a(xr) - a(xi), a(xi) - a(xr), mcnemar_p(xr, xi)
    out["G-GATE_failing_input"] = {"what": "(rl-A, id-B) as the pair", "G": fG, "p": fp,
                                   "failed": not (fG >= 5 * R2 * s and fGp >= 5 * R2 * s and fp < 0.001)}
    out["G-ID"] = {"L_minus_Lp": a(xr) - a(xi), "B_minus_Bp": a(xb) - a(xz), "bound_3sqrt2sigma": 3 * R2 * s,
                   "pass": bool(abs(a(xr) - a(xi)) <= 3 * R2 * s and abs(a(xb) - a(xz)) <= 3 * R2 * s)}
    out["G-ID"]["failing_input"] = {"what": "(base-A, rl-A) as an identity pair", "value": a(xb) - a(xr),
                                    "failed": not abs(a(xb) - a(xr)) <= 3 * R2 * s}
    return out


G["G-GATE_G-ID"] = gate_block("math500")
idr = recs("id", "math500")
dd = derangement(500, 0)
der_acc = 100 * sum(grade(idr[dd[i]]["gold"], idr[i]["extracted"]) for i in range(500)) / 500
Lp = G["G-GATE_G-ID"]["L_prime"]
G["S-BAND"] = {"rule": "81.5 <= L' <= 96.7", "value": Lp, "pass": bool(81.5 <= Lp <= 96.7),
               "failing_input": {"what": "id-B scored against deranged golds (seed 0)", "value": der_acc, "failed": der_acc < 81.5}}
G["G-GATE-GSM"] = gate_block("gsm8k")
G["thresholds_json_agrees"] = {
    "G": abs(G["G-GATE_G-ID"]["G"] - T["G-GATE"]["G"]) < 1e-9, "mcnemar_p": abs(G["G-GATE_G-ID"]["mcnemar_p"] - T["G-GATE"]["mcnemar_p"]) < 1e-55,
    "S-BAND_failing": abs(der_acc - T["S-BAND"]["failing_input"]["value"]) < 1e-9,
    "gsm_G": abs(G["G-GATE-GSM"]["G"] - T["gsm8k"]["G-GATE-GSM"]["G"]) < 1e-9}

# ---- A31 on the four rndE arms; energy within 5 % (A30)
a31 = {}
for k in ("k0.1", "k0.3", "k1", "k3"):
    b = jl(BANKS / f"rndE_{k}" / f"rndE_{k}" / "build.json")
    x = b["a31"]
    rel = abs(x["written_sq"] - x["matched_sq"]) / x["matched_sq"]
    a31[f"rndE_{k}"] = {"written_sq": x["written_sq"], "matched_sq": x["matched_sq"], "rel_diff_mine": rel, "rel_diff_banked": x["rel_diff"],
                        "tol": x["tol"], "n_tensors_differ": x["n_tensors_differ"], "pass_mine": rel <= 1e-12 and x["n_tensors_differ"] == 0,
                        "written_equals_families_mlp": x["written_sq"] == b["families"]["mlp"]["realized_sq"],
                        "realized_over_intended": b["families"]["mlp"]["realized_sq"] / b["intended_mlp_energy"],
                        "intended_equals_E_top": abs(b["intended_mlp_energy"] - SETS["E_top"][k]) / SETS["E_top"][k], "c_built": b["c_built"],
                        "passes": b["energy_passes"]}
G["A31"] = a31

# ---- engine blocks against S3 (base, s3gate-03)
def eng(m):
    e = dict(m["engine"])
    ea = {k: v for k, v in e.pop("engine_args").items() if k != "model"}
    e.pop("load_s", None)
    e.pop("amendment", None)
    return e, ea


ref_e, ref_ea = eng(jl(BANKS / "base/base/math500/metrics.json"))
engs = {}
for mp in sorted(BANKS.rglob("metrics.json")):
    if mp.parent.name not in ("math500", "gsm8k"):
        continue
    e, ea = eng(jl(mp))
    arm = mp.parent.parent.name
    ec = arm_recs.get(arm, {}).get("rec")
    engs[f"{arm}/{mp.parent.name}"] = {"fields_equal": e == ref_e, "engine_args_equal_but_model": ea == ref_ea,
                                       "diff_fields": sorted(k for k in set(e) | set(ref_e) if e.get(k) != ref_e.get(k)),
                                       "diff_args": sorted(k for k in set(ea) | set(ref_ea) if ea.get(k) != ref_ea.get(k))}
eng_checks = {}
for p in sorted((WT / "studies/next/n13/s4").glob("*/*/engine_check.json")):
    r = jl(p)
    eng_checks[f"{r['prefix']}/{r['step']}"] = {"equal": r["equal"], "libs_equal": r["libs"]["s3"] == r["libs"]["arm"], "libs": r["libs"]["arm"]}
G["engine"] = {"reference": "base/math500 (s3gate-03)", "ref_fields": ref_e, "per_arm": engs,
               "all_equal": all(v["fields_equal"] and v["engine_args_equal_but_model"] for v in engs.values()),
               "worktree_engine_checks": eng_checks}

# ---- the three numbers (5)
gg = G["G-GATE_G-ID"]
G["three_numbers"] = {
    "fit_both_sides": {"G-KNOWN": gk["value"], "target": 40.0, "tol": 1.6, "L_minus_Lprime": gg["L"] - gg["L_prime"],
                       "bound_3sqrt2sigma": gg["G-ID"]["bound_3sqrt2sigma"], "id_hash_equal_rl": C["G-HASH"]["id"]["pass"], "S-BAND_Lprime": Lp},
    "positive_control": {"G": gg["G"], "bound_5sqrt2sigma": gg["bound_5sqrt2sigma"], "PC-RANK": pr["recovery"]},
    "identity_pair": {"L_minus_Lprime": gg["L"] - gg["L_prime"], "B_minus_Bprime": gg["B"] - gg["B_prime"]}}

OUT.write_text(json.dumps(G, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
summary = {
    "G-GRADER": G["G-GRADER"]["mine"]["pass"], "G-GRADER_fi_failed": G["G-GRADER"]["failing_input"]["failed"],
    "G-TEMPLATE": G["G-TEMPLATE"]["per_arm_metrics_all_pass"] and G["G-TEMPLATE"]["mine_prompt_has_2.2_prefix_and_suffix_all"],
    "G-INIT": G["G-INIT"]["pass"], "G-PARTITION": G["G-PARTITION"]["sum_ok"], "PC-RANK": G["PC-RANK"]["pass_mine"],
    "G-KNOWN": G["G-KNOWN"]["pass_mine"], "G-HASH_family": G["G-HASH"]["all_S4_only_their_family"], "G-MASK": G["G-MASK"]["all_pass"],
    "G-COMPLETE": G["G-COMPLETE"]["all_pass"], "G-GATE": gg["G-GATE_pass"], "G-GATE_fi_failed": gg["G-GATE_failing_input"]["failed"],
    "G-ID": gg["G-ID"]["pass"], "G-ID_fi_failed": gg["G-ID"]["failing_input"]["failed"], "S-BAND": G["S-BAND"]["pass"],
    "S-BAND_fi": G["S-BAND"]["failing_input"]["value"], "G-GATE-GSM": G["G-GATE-GSM"]["G-GATE_pass"], "A31": all(v["pass_mine"] for v in a31.values()),
    "engine_all_equal": G["engine"]["all_equal"], "thresholds_agree": G["thresholds_json_agrees"]}
print(json.dumps(summary, indent=0, default=str))
