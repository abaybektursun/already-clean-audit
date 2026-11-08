"""N13 independent read, step 5: every number and label of the registered verdict_n13.json against the reader's recount.

    python -I compare_verdict.py <verdict_n13.json> <read_n13 dir> <out.json>

Reads verdict_read.json, gates_read.json, checks_read.json and recount_read.json written before the verdict was downloaded.
Floats agree when |a - b| <= 1e-9 * max(1, |a|, |b|). Fail-loud: no try/except.
"""
import json
import math
import pathlib
import sys

V = json.loads(pathlib.Path(sys.argv[1]).read_text())
RD = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])
M = json.loads((RD / "verdict_read.json").read_text())
G = json.loads((RD / "gates_read.json").read_text())
CK = json.loads((RD / "checks_read.json").read_text())
RC = json.loads((RD / "recount_read.json").read_text())["arms"]
rows, bad = [], []


def eq(a, b):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def chk(what, theirs, mine):
    ok = eq(theirs, mine)
    rows.append({"what": what, "verdict": theirs, "reader": mine, "agree": ok})
    if not ok:
        bad.append(what)


for arm, r in V["arms"].items():
    m = M["arms"][arm]
    chk(f"arms.{arm}.acc", r["acc"], m["acc"])
    chk(f"arms.{arm}.n", r["n"], m["n"])
    chk(f"arms.{arm}.label", r["label"], m["label"])
    chk(f"arms.{arm}.wilson", r["wilson"], RC[f"{arm}/math500"]["wilson_banked"])
    cp = CK["b_caps"]["per_arm"][f"{arm}/math500"]
    chk(f"arms.{arm}.cap_hit_share", r["cap_hit_share"], cp["cap_share"])
    chk(f"arms.{arm}.mean_tokens", r["mean_tokens"], cp["mean_new_tokens"])
    chk(f"arms.{arm}.answer_line_share", r["answer_line_share"], CK["c_format"]["per_arm"][arm]["modes"]["answer_line"] / 500)
    chk(f"arms.{arm}.status", r["status"], RC[f"{arm}/math500"]["status"])
    chk(f"arms.{arm}.screened", r["screened"], False)
assert set(V["arms"]) == set(M["arms"]), (set(V["arms"]) ^ set(M["arms"]))
for k, c in V["comparisons"].items():
    if k.startswith("_"):
        continue
    m = M["comparisons"][k]
    for f, g in (("diff", "diff"), ("mcnemar_p", "mcnemar_p"), ("bh_p", "bh_p"), ("bootstrap", "ci95")):
        chk(f"comparisons.{k}.{f}", c[f], m[g])
    chk(f"comparisons.{k}.n_paired", c["n_paired"], 500)
chk("comparisons._family_size_available", V["comparisons"]["_family_size_available"], 14)
chk("comparisons._indeterminate_parked", V["comparisons"]["_indeterminate_parked"], [])
H = V["hypotheses"]
for k in ("H_local", "H_diffuse", "H_attention", "H_artifact", "necessity"):
    chk(f"hypotheses.{k}", H[k], M["hypotheses"][k])
chk("hypotheses.k_rec", H["k_rec"] or "NONE", M["hypotheses"]["k_rec"])
for k, v in H["H_artifact_values"].items():
    chk(f"hypotheses.H_artifact_values.{k}", v, M["hypotheses"]["H_artifact_values"][k])
for k, v in V["gsm8k"].items():
    m = M["gsm8k"][k]
    chk(f"gsm8k.{k}.diff_gsm8k", v["diff_gsm8k"], m["diff_gsm8k"])
    chk(f"gsm8k.{k}.ci", v["ci"], m["ci95_gsm8k"])
    chk(f"gsm8k.{k}.label", v["label"], m["label"])
for k, v in V["thresholds"].items():
    if k == "C":
        chk("thresholds.C", v, 16384)
    else:
        chk(f"thresholds.{k}", v, M["frozen_recomputed"][k])
tn, mt = V["three_numbers"], G["three_numbers"]
chk("three.G-KNOWN", tn["fit_both_sides"]["G-KNOWN"], mt["fit_both_sides"]["G-KNOWN"])
chk("three.L_minus_Lprime", tn["fit_both_sides"]["L_minus_Lprime"], mt["fit_both_sides"]["L_minus_Lprime"])
chk("three.bound_3sqrt2sigma", tn["fit_both_sides"]["bound_3sqrt2sigma"], mt["fit_both_sides"]["bound_3sqrt2sigma"])
chk("three.id_hash_equal_rl", tn["fit_both_sides"]["id_hash_equal_rl"], mt["fit_both_sides"]["id_hash_equal_rl"])
chk("three.S-BAND", tn["fit_both_sides"]["S-BAND"], mt["fit_both_sides"]["S-BAND_Lprime"])
chk("three.G", tn["positive_control"]["G"], mt["positive_control"]["G"])
chk("three.bound_5sqrt2sigma", tn["positive_control"]["bound_5sqrt2sigma"], mt["positive_control"]["bound_5sqrt2sigma"])
chk("three.PC-RANK", tn["positive_control"]["PC-RANK_recovery"], mt["positive_control"]["PC-RANK"])
chk("three.identity.L_minus_Lprime", tn["identity_pair"]["L_minus_Lprime"], mt["identity_pair"]["L_minus_Lprime"])
chk("three.identity.B_minus_Bprime", tn["identity_pair"]["B_minus_Bprime"], mt["identity_pair"]["B_minus_Bprime"])
chk("three.missing", tn["missing"], [])
g = V["gates"]
gg = G["G-GATE_G-ID"]
chk("gates.G-GRADER", [g["G-GRADER"]["pass"], g["G-GRADER"]["failed"], g["G-GRADER"]["value"]],
    [G["G-GRADER"]["mine"]["pass"], G["G-GRADER"]["failing_input"]["failed"], G["G-GRADER"]["mine"]["self_equal"]])
chk("gates.G-TEMPLATE", [g["G-TEMPLATE"]["pass"], g["G-TEMPLATE"]["failed"]],
    [G["G-TEMPLATE"]["per_arm_metrics_all_pass"] and G["G-TEMPLATE"]["mine_prompt_has_2.2_prefix_and_suffix_all"], G["G-TEMPLATE"]["failing_input"]["failed"]])
chk("gates.G-INIT", [g["G-INIT"]["pass"], g["G-INIT"]["failed"], g["G-INIT"]["value"]], [G["G-INIT"]["pass"], True, G["G-INIT"]["b_difference"]])
chk("gates.PC-RANK", [g["PC-RANK"]["pass"], g["PC-RANK"]["value"], g["PC-RANK"]["failing_value"]],
    [G["PC-RANK"]["pass_mine"], G["PC-RANK"]["value"], G["PC-RANK"]["failing_value"]])
chk("gates.G-KNOWN", [g["G-KNOWN"]["pass"], g["G-KNOWN"]["value"], g["G-KNOWN"]["failing_value"]],
    [G["G-KNOWN"]["pass_mine"], G["G-KNOWN"]["value"], G["G-KNOWN"]["failing_value"]])
chk("gates.G-HASH", [g["G-HASH"]["pass"], g["G-HASH"]["failed"]], [G["G-HASH"]["controls"]["zero"] and G["G-HASH"]["controls"]["id"], G["G-HASH"]["controls"]["failing_input_failed"]])
chk("gates.G-MASK", g["G-MASK"]["pass"], G["G-MASK"]["all_pass"])
chk("gates.G-GATE", [g["G-GATE"]["pass"], g["G-GATE"]["value"], g["G-GATE"]["failing_value"]], [gg["G-GATE_pass"], gg["G"], gg["G-GATE_failing_input"]["G"]])
chk("gates.G-ID", [g["G-ID"]["pass"]] + g["G-ID"]["value"], [gg["G-ID"]["pass"], gg["G-ID"]["L_minus_Lp"], gg["G-ID"]["B_minus_Bp"]])
chk("gates.S-BAND", [g["S-BAND"]["pass"], g["S-BAND"]["value"], g["S-BAND"]["failing_value"]],
    [G["S-BAND"]["pass"], G["S-BAND"]["value"], G["S-BAND"]["failing_input"]["value"]])
chk("gates.G-GATE-GSM", [g["G-GATE-GSM"]["pass"], g["G-GATE-GSM"]["value"]], [G["G-GATE-GSM"]["G-GATE_pass"], G["G-GATE-GSM"]["G"]])
chk("gates.G-COMPLETE.pass", g["G-COMPLETE"]["pass"], G["G-COMPLETE"]["all_pass"])
for k, v in g["G-COMPLETE"]["n_empty"].items():
    chk(f"gates.G-COMPLETE.n_empty.{k}", v, G["G-COMPLETE"]["per_arm"][k]["n_empty"])
chk("gates.G-COMPLETE.n_arm_bench", len(g["G-COMPLETE"]["value"]), G["G-COMPLETE"]["n_arm_bench"])
for k in g:
    chk(f"gates.{k}.failing_input_failed", g[k]["failed"], True)
chk("verdict", V["verdict"], "COMPLETE")
chk("stamps", V["stamps"], [])
chk("parked", V["parked"], [])
res = {"n_compared": len(rows), "n_agree": sum(r["agree"] for r in rows), "disagreements": bad, "rows": rows}
OUT.write_text(json.dumps(res, indent=1))
print(f"compared {len(rows)}, agree {res['n_agree']}, disagree {len(bad)}: {bad}")
