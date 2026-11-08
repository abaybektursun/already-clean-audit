"""N13 independent read, step 4: checks (a) to (j) of the reader's brief, each by a computation on the banks.

    python -I checks_read.py <banks dir> <recount_read.json> <out.json>

(a) concordance under shared seeds; (b) cap hits, lengths, and the verdict with cap-hit answers scored wrong or dropped; (c) format
classes, accuracy within each, and wrong answers that hold the gold in a format the rule does not read (with the verdict under a
lenient re-score, exploratory); (d) realized MLP change energy of the controls against the top set and the full change; (e) second
draws and the seed-B top arm against delta; (f) base-right/arm-wrong counts; (g) leverage: bootstrap seed, BH at q 0.10, Newcombe-Wilson
in place of the bootstrap; (h) H_local at every k against each control; (i) GSM8K per-arm accuracies; (j) other findings.
The verdict tree is verdict_read.run (this directory). Fail-loud: no try/except.
"""
import json
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import verdict_read as V  # noqa: E402
from math_verify import parse, verify  # noqa: E402

BANKS, RC, OUT = (pathlib.Path(x) for x in sys.argv[1:4])
jl = lambda p: json.loads(pathlib.Path(p).read_text())
T, C, SETS = jl(BANKS / "_inputs/thresholds.json"), jl(BANKS / "_inputs/controls.json"), jl(BANKS / "_inputs/sets.json")
R = jl(RC)["arms"]
KS = V.KS
CAP = 16384
Z = 1.959964


def load(arm, bench):
    p = BANKS / arm / arm / bench / "generations.jsonl"
    return sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()), key=lambda r: r["problem_index"])


MATH = {v["arm"]: load(v["arm"], "math500") for v in R.values() if v["bench"] == "math500"}
GSM = {v["arm"]: load(v["arm"], "gsm8k") for v in R.values() if v["bench"] == "gsm8k"}
sc = lambda rr: np.array([int(r["score"]) for r in rr], dtype=np.int64)
VM, VG = {a: sc(rr) for a, rr in MATH.items()}, {a: sc(rr) for a, rr in GSM.items()}
for a in VM:
    assert "".join(map(str, VM[a])) == R[f"{a}/math500"]["scores_banked"]
FAM = [(f"top_{k}", f"{c}_{k}") for k in KS for c in ("dare", "rndE", "rndT")] + [("mlp", "attn"), ("top_k1", "comp_k1")]
D = T["delta"]
OUTJ = {}


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def hyp_summary(o):
    h = o["hypotheses"]
    return {"k_rec": h["k_rec"], "H_local": h["H_local"], "H_diffuse": h["H_diffuse"], "H_attention": h["H_attention"],
            "H_artifact": h["H_artifact"], "necessity": h["necessity"], "gsm8k": {k: v["label"] for k, v in o["gsm8k"].items() if k != "acc"},
            "labels": {a: v["label"] for a, v in o["arms"].items() if v["label"]}}


REF = V.run(VM, VG, C, T)
REFS = hyp_summary(REF)

# ---- (a) concordance
def pair_stats(x, y, rr_x, rr_y):
    vx, vy = VM[x] if rr_x is None else rr_x, VM[y] if rr_y is None else rr_y
    return vx, vy


conc = {}
pairs = FAM + [("base", "zero"), ("rl", "id"), ("top_k1", "top_k1_B"), ("base", "top_k0.1"), ("base", "rndE_k1"), ("dare_k1", "mlp"),
               ("dare_k1", "dare_k1_d2"), ("rndT_k1", "rndT_k1_d2"), ("rl", "comp_k1"), ("base", "top_k1")]
bug_identical_text_diff_score = 0
for x, y in pairs:
    tx, ty = [r["text"] for r in MATH[x]], [r["text"] for r in MATH[y]]
    same = [a == b for a, b in zip(tx, ty)]
    same_nonempty = [a == b and a != "" for a, b in zip(tx, ty)]
    vx, vy = VM[x], VM[y]
    b, c, p = V.mcnemar(vx, vy)
    bug_identical_text_diff_score += sum(1 for s, a1, a2 in zip(same, vx, vy) if s and a1 != a2)
    conc[f"{x}|{y}"] = {"discordant": b + c, "b": b, "c": c, "mcnemar_p": p, "share_identical_text": float(np.mean(same)),
                        "n_identical_text": int(sum(same)), "n_identical_nonempty": int(sum(same_nonempty)), "p_rests_on_lt20_discordant": b + c < 20,
                        "primary": (x, y) in FAM}
OUTJ["a_concordance"] = {"pairs": conc, "identical_text_but_different_score": bug_identical_text_diff_score}

# ---- (b) cap hits and lengths; the verdict with cap-hit answers scored wrong, or dropped
caps = {}
for bench, D_ in (("math500", MATH), ("gsm8k", GSM)):
    for a, rr in D_.items():
        cap = np.array([r["cap_hit"] for r in rr])
        x = sc(rr)
        caps[f"{a}/{bench}"] = {"cap_hits": int(cap.sum()), "cap_share": float(cap.mean()), "mean_new_tokens": float(np.mean([r["n_new"] for r in rr])),
                                "median_new_tokens": float(np.median([r["n_new"] for r in rr])),
                                "acc_noncapped": float(100 * x[~cap].mean()), "n_noncapped": int((~cap).sum()),
                                "right_among_capped": int(x[cap].sum()), "acc_all": float(100 * x.mean())}
capwrong_m = {a: np.where(np.array([r["cap_hit"] for r in MATH[a]]), 0, VM[a]) for a in VM}
capwrong_g = {a: np.where(np.array([r["cap_hit"] for r in GSM[a]]), 0, VG[a]) for a in VG}
o_cw = V.run(capwrong_m, capwrong_g, C, None)
um = np.array([any(MATH[a][i]["cap_hit"] for a in MATH) for i in range(500)])
ug = np.array([any(GSM[a][i]["cap_hit"] for a in GSM) for i in range(1319)])
o_cd = V.run({a: v[~um] for a, v in VM.items()}, {a: v[~ug] for a, v in VG.items()}, C, None)
pair_drop = {}
for x, y in FAM:
    keep = ~(np.array([r["cap_hit"] for r in MATH[x]]) | np.array([r["cap_hit"] for r in MATH[y]]))
    vx, vy = VM[x][keep], VM[y][keep]
    pair_drop[f"{x}|{y}"] = {"n": int(keep.sum()), "diff": float(100 * (vx.mean() - vy.mean())), "mcnemar_p": V.mcnemar(vx, vy)[2]}
OUTJ["b_caps"] = {"per_arm": caps, "capwrong": {"summary": hyp_summary(o_cw), "thresholds": o_cw["thresholds_used"],
                                                "acc": {a: v["acc"] for a, v in o_cw["arms"].items()}},
                  "capdrop_union": {"n_math": int((~um).sum()), "n_gsm": int((~ug).sum()), "summary": hyp_summary(o_cd),
                                    "thresholds": o_cd["thresholds_used"], "acc": {a: v["acc"] for a, v in o_cd["arms"].items()}},
                  "pairwise_drop": pair_drop,
                  "changes_vs_registered": {"capwrong": {k: (REFS[k], hyp_summary(o_cw)[k]) for k in REFS if REFS[k] != hyp_summary(o_cw)[k]},
                                            "capdrop_union": {k: (REFS[k], hyp_summary(o_cd)[k]) for k in REFS if REFS[k] != hyp_summary(o_cd)[k]}}}

# ---- (c) format
LABELS = ("**Answer:**", "*Answer:*", "__Answer:__", "Answer:")


def has_label_line(t):
    for line in t.split("\n"):
        s = line.lstrip(" ")
        for lab in LABELS:
            if s.startswith(lab) and s[len(lab):].strip():
                return True
    return False


def boxes(t):
    out, start, key = [], 0, "\\boxed{"
    while True:
        i = t.find(key, start)
        if i < 0:
            return out
        depth, j = 1, i + len(key)
        while j < len(t) and depth:
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        if depth == 0:
            out.append(t[i + len(key):j - 1])
        start = i + 1


ANY_LABEL = re.compile(r"(?i)answer\s*(?:\*\*|__|\*)?\s*(?::|is\b)\s*(?:\*\*|__|\*)?\s*(.*\S)")


def label_anywhere(t):
    hit = None
    for line in t.split("\n"):
        ms = list(ANY_LABEL.finditer(line))
        if ms:
            g = ms[-1].group(1).strip().strip("*").strip()
            g = g[:-1] if g.endswith(".") else g
            if g:
                hit = g
    return hit


def wrap(s):
    return s if (re.search(r"(?<!\\)\$", s) or "\\boxed" in s) else "$" + s + "$"


GC = {}


def ok(gold, x):
    k = (gold, x)
    if k not in GC:
        GC[k] = int(bool(verify(parse(wrap(gold)), parse(wrap(x)))))
    return GC[k]


def ok_free(gold, x):
    k = ("free", gold, x)
    if k not in GC:
        GC[k] = int(bool(verify(parse(wrap(gold)), parse(x))))
    return GC[k]


fmt, lenient_m, lenient_ll_m, examples = {}, {}, {}, {}
for a, rr in MATH.items():
    x = sc(rr)
    modes = np.array([r["mode"] for r in rr])
    hl = np.array([has_label_line(r["text"]) for r in rr])
    hb = np.array([len(boxes(r["text"])) > 0 for r in rr])
    cap = np.array([r["cap_hit"] for r in rr])
    cls = {"label_line_only": hl & ~hb, "box_only": ~hl & hb, "both": hl & hb, "neither": ~hl & ~hb}
    acc_mode = {m: (float(100 * x[modes == m].mean()) if (modes == m).any() else None) for m in ("answer_line", "boxed")}
    gains = {"label_anywhere": [], "any_box": [], "last_line": []}
    len_ = np.zeros(500, dtype=np.int64)
    len_ll = np.zeros(500, dtype=np.int64)
    for i, r in enumerate(rr):
        if r["score"] == 1:
            len_[i] = len_ll[i] = 1
            continue
        t, g = r["text"], r["gold"]
        la = label_anywhere(t)
        hit_la = la is not None and ok(g, la)
        hit_bx = any(ok(g, b) for b in boxes(t))
        lines = [l.strip().strip("*").strip() for l in t.split("\n") if l.strip()]
        hit_ll = bool(lines) and ok_free(g, lines[-1])
        if hit_la:
            gains["label_anywhere"].append(i)
        if hit_bx:
            gains["any_box"].append(i)
        if hit_ll:
            gains["last_line"].append(i)
        len_[i] = int(hit_la or hit_bx)
        len_ll[i] = int(hit_la or hit_bx or hit_ll)
    lenient_m[a], lenient_ll_m[a] = len_, len_ll
    fmt[a] = {"modes": {m: int((modes == m).sum()) for m in ("answer_line", "boxed", "none")},
              "text_classes": {k: int(v.sum()) for k, v in cls.items()},
              "acc_by_mode": acc_mode, "acc_among_extracted": float(100 * x[modes != "none"].mean()),
              "n_extracted": int((modes != "none").sum()), "none_noncapped": int(((modes == "none") & ~cap).sum()),
              "acc_by_text_class": {k: (float(100 * x[v].mean()) if v.any() else None) for k, v in cls.items()},
              "wrong_with_gold_elsewhere": {k: len(v) for k, v in gains.items()},
              "wrong_with_gold_elsewhere_idx": {k: v[:40] for k, v in gains.items()},
              "wrong_with_gold_elsewhere_noncapped": {k: sum(1 for i in v if not rr[i]["cap_hit"]) for k, v in gains.items()},
              "lenient_gain_noncapped": int(sum(1 for i in range(500) if len_[i] and not x[i] and not rr[i]["cap_hit"])),
              "lenient_gain_total": int(len_.sum() - x.sum()),
              "acc_lenient": float(100 * len_.mean()), "acc_lenient_plus_last_line": float(100 * len_ll.mean())}
lenient_g = {}
for a, rr in GSM.items():
    v = sc(rr).copy()
    for i, r in enumerate(rr):
        if v[i] == 0:
            la = label_anywhere(r["text"])
            if (la is not None and ok(r["gold"], la)) or any(ok(r["gold"], b) for b in boxes(r["text"])):
                v[i] = 1
    lenient_g[a] = v
o_len = V.run(lenient_m, lenient_g, C, None)
o_len_ll = V.run(lenient_ll_m, lenient_g, C, None)
OUTJ["c_format"] = {"per_arm": fmt, "lenient": {"summary": hyp_summary(o_len), "thresholds": o_len["thresholds_used"],
                                                "acc": {a: v["acc"] for a, v in o_len["arms"].items()},
                                                "comparisons": {k: {kk: v[kk] for kk in ("diff", "mcnemar_p", "bh_p")} for k, v in o_len["comparisons"].items()},
                                                "gsm_acc": {a: float(100 * v.mean()) for a, v in lenient_g.items()}},
                    "lenient_plus_last_line": {"summary": hyp_summary(o_len_ll), "thresholds": o_len_ll["thresholds_used"],
                                               "acc": {a: v["acc"] for a, v in o_len_ll["arms"].items()}},
                    "changes_vs_registered": {"lenient": {k: (REFS[k], hyp_summary(o_len)[k]) for k in REFS if REFS[k] != hyp_summary(o_len)[k]},
                                              "lenient_plus_last_line": {k: (REFS[k], hyp_summary(o_len_ll)[k]) for k in REFS
                                                                         if REFS[k] != hyp_summary(o_len_ll)[k]}}}

# ---- (d) controls' matching
E_total = jl(BANKS / "id/id/build.json")["families"]["mlp"]["realized_sq"]
match = {"E_total_mlp": E_total, "E_total_sets": SETS["E_total"]}
for k in KS:
    top = jl(BANKS / f"top_{k}" / f"top_{k}" / "build.json")["families"]["mlp"]
    row = {"top": {"realized": top["realized_sq"], "E_top_sets": SETS["E_top"][k], "changed": top["changed"], "in_mask": top["in_mask"],
                   "share_of_full": top["realized_sq"] / E_total}}
    for c in ("rndE", "rndT", "dare"):
        b = jl(BANKS / f"{c}_{k}" / f"{c}_{k}" / "build.json")
        f = b["families"]["mlp"]
        r = {"realized": f["realized_sq"], "over_top": f["realized_sq"] / top["realized_sq"], "over_full": f["realized_sq"] / E_total,
             "changed": f["changed"], "in_mask": f["in_mask"], "n_neurons": b["n_neurons_in_set"]}
        if c == "rndE":
            r.update({"within_5pct_of_top": abs(f["realized_sq"] / top["realized_sq"] - 1) <= 0.05, "c_built": b["c_built"],
                      "c0": SETS["rndE"][k]["c"]})
        if c == "rndT":
            r.update({"m_k": SETS["rndT"][k]["m"], "short": SETS["rndT"][k]["short"]})
        if c == "dare":
            r.update({"kept": b["kept"], "kept_expected": b["kept_expected"], "kept_sd_units": b["kept_sd_units"], "kept_pass": b["kept_pass"],
                      "q": b["spec"]["q"], "c": b["spec"]["c"], "realized_over_intended": f["realized_sq"] / b["intended_mlp_energy"],
                      "expected_over_full_is_1_over_q": 1 / b["spec"]["q"]})
        row[c] = r
    match[k] = row
for extra in ("rnd0_k1", "rndT_k1_d2", "dare_k1_d2", "topF_k1", "mlp"):
    b = jl(BANKS / extra / extra / "build.json")
    match[extra] = {"realized": b["families"]["mlp"]["realized_sq"], "over_top_k1": b["families"]["mlp"]["realized_sq"] / match["k1"]["top"]["realized"],
                    "over_full": b["families"]["mlp"]["realized_sq"] / E_total, "n_neurons": b["n_neurons_in_set"], "kept": b.get("kept")}
OUTJ["d_matching"] = match

# ---- (e) spread
e = REF["exploratory"]
OUTJ["e_spread"] = {**e, "each_within_delta": {k: abs(e[k]) <= D for k in ("top_k1_minus_top_k1_B", "rndT_k1_d2_minus_rndT_k1", "dare_k1_d2_minus_dare_k1")},
                    "sigma_mid_le_1.5_sigma": e["sigma_mid_over_sigma"] <= 1.5}

# ---- (f) base's own answers
fb = {}
for a in sorted(VM):
    if a == "base":
        continue
    fb[a] = {"base_right_arm_wrong": int(((VM["base"] == 1) & (VM[a] == 0)).sum()), "base_wrong_arm_right": int(((VM["base"] == 0) & (VM[a] == 1)).sum())}
OUTJ["f_base_vs_arm"] = fb

# ---- (g) leverage
seeds = {}
for s in range(1, 101):
    o = V.run(VM, VG, C, T, boot_seed=s)
    hs = hyp_summary(o)
    seeds[s] = {k: hs[k] for k in ("H_local", "H_diffuse", "H_attention", "necessity", "gsm8k")}
changed = {s: {k: v for k, v in hs.items() if v != {k2: REFS[k2] for k2 in hs}[k]} for s, hs in seeds.items()}
gsm_ci = {}
for s in range(0, 101):
    o = V.run(VM, VG, C, T, boot_seed=s) if s else REF
    for kk, v in o["gsm8k"].items():
        if kk != "acc":
            gsm_ci.setdefault(kk, []).append(v["ci95_gsm8k"][0])
o_q10 = V.run(VM, VG, C, T, q=0.10)
o_w = V.run(VM, VG, C, T, ci="wilson_diff")
OUTJ["g_leverage"] = {"boot_seeds_1_100_changes": {s: v for s, v in changed.items() if v},
                      "n_seeds_with_any_change": sum(1 for v in changed.values() if v),
                      "gsm8k_ci_lower_over_seeds_0_100": {kk: {"min": min(v), "max": max(v), "n_le_0": sum(1 for x in v if x <= 0)} for kk, v in gsm_ci.items()},
                      "bh_q010": {"summary": hyp_summary(o_q10), "changes": {k: (REFS[k], hyp_summary(o_q10)[k]) for k in REFS if REFS[k] != hyp_summary(o_q10)[k]},
                                  "rejected_at_010": [k for k, v in o_q10["comparisons"].items() if v["bh_p"] < 0.10]},
                      "wilson_newcombe_ci": {"summary": hyp_summary(o_w), "changes": {k: (REFS[k], hyp_summary(o_w)[k]) for k in REFS if REFS[k] != hyp_summary(o_w)[k]},
                                             "gsm8k": {k: v for k, v in o_w["gsm8k"].items() if k != "acc"}}}

# ---- (h) H_local at every k against each control
hk = {}
arms = REF["arms"]
for k in KS:
    t = f"top_{k}"
    row = {"top_acc": arms[t]["acc"], "top_label": arms[t]["label"], "controls": {}}
    for c in ("dare", "rndE", "rndT"):
        y = f"{c}_{k}"
        cm = REF["comparisons"][f"{t}|{y}"]
        row["controls"][y] = {"acc": arms[y]["acc"], "label": arms[y]["label"], "top_minus_control": cm["diff"], "within_delta": arms[y]["acc"] >= arms[t]["acc"] - D,
                              "top_beats": cm["diff"] > D and cm["bh_p"] < 0.05, "control_beats_top_exploratory": -cm["diff"] > D and cm["bh_p"] < 0.05,
                              "bh_p": cm["bh_p"], "ci95": cm["ci95"]}
    best = REF["top_vs_best_control"][k]
    row["best_control"] = best
    row["if_this_k_were_k_rec"] = ("REFUTED (a control within delta)" if any(v["within_delta"] for v in row["controls"].values()) else
                                   "SUPPORTED" if all(v["top_beats"] for v in row["controls"].values()) else "INDETERMINATE")
    hk[k] = row
OUTJ["h_local_every_k"] = hk

# ---- (i) GSM8K per arm
OUTJ["i_gsm8k"] = {a: {"k": int(v.sum()), "acc": float(100 * v.mean()), "wilson": wilson(int(v.sum()), len(v))} for a, v in sorted(VG.items())}

# ---- (j) other: damage screens, empty answers, dare against mlp, seed-B identity of text
scr = {}
for mp in sorted(BANKS.rglob("metrics.json")):
    if mp.parent.name == "math500":
        m = jl(mp)
        scr[mp.parent.parent.name] = m["screen"]
OUTJ["j_other"] = {"damage_screen_math500": scr,
                   "dare_vs_mlp": {k: {"dare_minus_mlp": float(100 * (VM[f"dare_{k}"].mean() - VM["mlp"].mean())),
                                       "mcnemar_p": V.mcnemar(VM[f"dare_{k}"], VM["mlp"])[2]} for k in KS},
                   "registered_reference": REFS}

OUT.write_text(json.dumps(OUTJ, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("checks written")
print("capwrong changes:", OUTJ["b_caps"]["changes_vs_registered"]["capwrong"])
print("capdrop changes:", OUTJ["b_caps"]["changes_vs_registered"]["capdrop_union"], OUTJ["b_caps"]["capdrop_union"]["n_math"], OUTJ["b_caps"]["capdrop_union"]["n_gsm"])
print("lenient changes:", OUTJ["c_format"]["changes_vs_registered"])
print("seeds with change:", OUTJ["g_leverage"]["n_seeds_with_any_change"], json.dumps(OUTJ["g_leverage"]["gsm8k_ci_lower_over_seeds_0_100"]))
print("q010 changes:", OUTJ["g_leverage"]["bh_q010"]["changes"], OUTJ["g_leverage"]["bh_q010"]["rejected_at_010"])
print("wilson changes:", OUTJ["g_leverage"]["wilson_newcombe_ci"]["changes"])
print("identical text but different score:", bug_identical_text_diff_score)
