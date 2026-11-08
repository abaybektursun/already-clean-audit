"""M2 ten raw examples, the zero-GPU part (studies/next/07_PICKS/M2_SPEC.md §10). Laptop banks only, freeze half only.

Selection (frozen in `select`, run inside `m2 births`). Freeze half: the carrier-side overlap A_own = the number of
E_c positions in the carrier's own top-50 at T_(k-1) (the grid point's samples/contexts.jsonl). Five births with
A_own >= 1 and five with A_own = 0, each drawn by numpy default_rng(0) from the sorted ids. Under A a birth with
A_own = 0 has rank 22,016 for certain (every pole scores >= 0 and ties count against the carrier): a birth scored
wrong. A birth with A_own >= 1 is scored right only if fewer than ten other poles of the layer reach its score; that
needs every pole's top-50 (S3 topk/), so it stays PENDING here. Test half: ten ids drawn by default_rng(1) from the
sorted test ids, without reading any of their data; their records are written by the job and opened at unblinding.

What each record holds, from the laptop banks: the birth; its exemplar contexts at T_15 (64 tokens before, 16 after,
decoded by the harvest); the carrier at T_(k-1): the median a+ over E_c, its q99 / q999 / q9999 / max_abs, the
percentile bracket those quantiles give (the full histogram is on S3 only), A_own, the carrier's rank under A among
the 20 grid poles of its layer, its own top-5 contexts; the carrier at T_k; the pseudo-birth twin; and the matched
random-pole reading (12_SHAPING_NOTE.md §4 item 3): the three other grid poles of the layer nearest the carrier in
log max|a| at T_(k-1), each with its A on E_c and its top-3 contexts. The top-10 lists among the 22,016 poles under A
and B are PENDING: they need topk/ of every pole and the forward at T_(k-1); the job writes them to
out/m2/samples/raw_examples_freeze.jsonl.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from infra.pg import m2


def contexts(bank, t: int) -> dict:
    p = pathlib.Path(bank) / "w3" / "grid" / m2.POINTS[t][1] / "samples" / "contexts.jsonl"
    assert m2.sha256_file(p) == m2.CONTEXTS_SHA[t], f"{p}: sha256 differs from the registered value"
    return {r["pole"]: r for r in (json.loads(x) for x in open(p))}


def own_overlap(ctx_t: dict, pole: str, tok: list) -> int:
    return len({e["tok"] for e in ctx_t[pole]["topk"]} & set(tok))


def select(bank, births: list) -> dict:
    fz = sorted((b for b in births if b["half"] == "freeze"), key=lambda b: b["id"])
    cache = {}
    a_own = {}
    for b in fz:
        t = b["k"] - 1
        if t not in cache:
            cache[t] = contexts(bank, t)
        a_own[b["id"]] = own_overlap(cache[t], b["pole"], b["tok"])
    pos = [b["id"] for b in fz if a_own[b["id"]] >= 1]
    zero = [b["id"] for b in fz if a_own[b["id"]] == 0]
    rng = np.random.default_rng(m2.SEED)
    pick_f = sorted(rng.choice(pos, 5, replace=False).tolist()) + sorted(rng.choice(zero, 5, replace=False).tolist())
    te = sorted(b["id"] for b in births if b["half"] == "test")
    pick_t = sorted(np.random.default_rng(m2.SEED + 1).choice(te, 10, replace=False).tolist())
    return {"rule": "freeze: 5 with A_own >= 1 and 5 with A_own = 0, default_rng(0) on sorted ids; "
                    "test: 10 by default_rng(1) on sorted ids, no data read",
            "freeze": pick_f, "test": pick_t, "freeze_a_own": {i: a_own[i] for i in pick_f},
            "freeze_counts": {"a_own_ge1": len(pos), "a_own_0": len(zero)}}


def bracket(v: float, c: dict) -> str:
    if v <= 0:
        return "0 (median a+ is 0)"
    if v < c["q99"]:
        return "below 0.99"
    if v < c["q999"]:
        return "[0.99, 0.999)"
    if v < c["q9999"]:
        return "[0.999, 0.9999)"
    return "at or above 0.9999"


def ctx_rows(c: dict, n: int) -> list:
    return [{"rank": e["rank"], "tok": e["tok"], "a": e["a"], "text": e["before"][-160:] + " [[" + e["token"] + "]] "
             + e["after"]} for e in c["topk"][:n]]


def build(bank, births_json: dict) -> list:
    B = m2.load_bank(bank)
    births = {b["id"]: b for b in births_json["births"]}
    ids = births_json["examples"]["freeze"]
    cache = {}

    def ctx(t):
        if t not in cache:
            cache[t] = contexts(bank, t)
        return cache[t]

    out = []
    for i in ids:
        b = births[i]
        assert b["half"] == "freeze", f"{i} is not in the freeze half"
        t, k = b["k"] - 1, b["k"]
        row = B["row"][b["pole"]]
        ap_pre = B["a_plus"][t, row, b["ranks"]]
        ap_k = B["a_plus"][k, row, b["ranks"]]
        c15, cpre, ck = ctx(15)[b["pole"]], ctx(t)[b["pole"]], ctx(k)[b["pole"]]
        med_pre, med_k = float(np.median(ap_pre)), float(np.median(ap_k))
        layer_poles = [p for p in ctx(t) if m2.parse_pole(p)[0] == b["layer"]]
        a_grid = {p: own_overlap(ctx(t), p, b["tok"]) for p in layer_poles}
        a_own = a_grid[b["pole"]]
        others = [p for p in layer_poles if m2.parse_pole(p)[1] != b["neuron"]]
        lm = np.log(cpre["max_abs"])
        near = sorted(others, key=lambda p: (abs(np.log(ctx(t)[p]["max_abs"]) - lm), p))[:3]
        pb = b["pb"]
        pb_rec = None
        if pb is not None:
            prow = B["row"][pb["pole"]]
            pmed = float(np.median(B["a_plus"][t, prow, pb["ranks"]]))
            pb_rec = {"pole": pb["pole"], "ranks": pb["ranks"], "median_a_plus_pre": pmed,
                      "ratio_pre": pmed / float(B["q999"][t, prow]), "a_own_pre": own_overlap(ctx(t), pb["pole"], pb["tok"])}
        out.append({
            "id": i, "name": b["name"], "size": b["size"], "k": k, "step_k": b["step_k"], "step_pre": b["step_pre"],
            "p_late": b["p_late"], "is_main": b["is_main"],
            "exemplars_T15": [{"rank": r, "tok": c15["topk"][r]["tok"], "doc": c15["topk"][r]["doc"],
                               "pos": c15["topk"][r]["pos"], "a_T15": c15["topk"][r]["a"],
                               "a_plus_pre": float(ap_pre[j]), "a_plus_birth": float(ap_k[j]),
                               "text": c15["topk"][r]["before"][-160:] + " [[" + c15["topk"][r]["token"] + "]] "
                               + c15["topk"][r]["after"]} for j, r in enumerate(b["ranks"])],
            "carrier_pre": {"t": t, "median_a_plus": med_pre, "q99": cpre["q99"], "q999": cpre["q999"],
                            "q9999": cpre["q9999"], "max_abs": cpre["max_abs"], "ratio": med_pre / cpre["q999"],
                            "percentile_bracket": bracket(med_pre, cpre), "a_own": a_own,
                            "rank_A_among_20_grid_poles": int(sum(v >= a_own for v in a_grid.values())),
                            "top5": ctx_rows(cpre, 5)},
            "carrier_birth": {"t": k, "median_a_plus": med_k, "q999": ck["q999"], "ratio": med_k / ck["q999"],
                              "present": bool(med_k > 0 and med_k >= ck["q999"])},
            "score_A_rank_22016": (m2.N_POLES_LAYER if a_own == 0 else
                                   "PENDING: needs every pole's top-50 at T_(k-1), S3 "
                                   f"results/neurons/w3/grid/{m2.POINTS[t][1]}/topk/layer_{b['layer']:02d}.npz"),
            "scored": "wrong under A (rank 22,016 for certain)" if a_own == 0 else "PENDING",
            "top10_A_B": "PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl",
            "pseudo_birth": pb_rec,
            "matched_reading": [{"pole": p, "max_abs_pre": ctx(t)[p]["max_abs"], "a_on_E": a_grid[p],
                                 "top3": ctx_rows(ctx(t)[p], 3)} for p in near]})
    return out


def write(bank, births_json: dict, out: pathlib.Path):
    out.mkdir(parents=True, exist_ok=True)
    recs = build(bank, births_json)
    (out / "raw_examples_zero_gpu.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    lines = ["# M2 ten raw examples, zero-GPU part (freeze half)", "",
             "Written by `python -m infra.pg.m2 examples` from the laptop banks; the rule is in "
             "`infra/pg/m2_examples.py`. Units: a+ in raw MLP activation units; q999 is the carrier's banked "
             "99.9th percentile at that point (a histogram bin edge). The text marks the activating token as [[ ]].",
             ""]
    for r in recs:
        c, cb = r["carrier_pre"], r["carrier_birth"]
        lines += [f"## {r['id']} — {r['name']}", "",
                  f"Size {r['size']}, born T_{r['k']} (step {r['step_k']:,}); scored at T_{c['t']} (step "
                  f"{r['step_pre']:,}); p_late {r['p_late']:.4f}; main {r['is_main']}.", "",
                  f"- Carrier at T_{c['t']}: median a+ over the exemplars {c['median_a_plus']:.4g} against q999 "
                  f"{c['q999']:.4g} (ratio {c['ratio']:.3f}); percentile bracket {c['percentile_bracket']}; "
                  f"max|a| {c['max_abs']:.4g}.",
                  f"- Carrier at T_{cb['t']}: median {cb['median_a_plus']:.4g}, q999 {cb['q999']:.4g}, ratio "
                  f"{cb['ratio']:.3f}, present {cb['present']}.",
                  f"- A_own (exemplar positions in the carrier's own T_{c['t']} top-50): {c['a_own']}; rank under A "
                  f"among the layer's 20 grid poles: {c['rank_A_among_20_grid_poles']}; rank among 22,016: "
                  f"{r['score_A_rank_22016']}. Scored: {r['scored']}.",
                  f"- Top-10 lists under A and B: {r['top10_A_B']}.",
                  "- Pseudo-birth twin: " + ("none found" if r["pseudo_birth"] is None else
                                              f"{r['pseudo_birth']['pole']} ranks {r['pseudo_birth']['ranks']}, "
                                              f"ratio at T_{c['t']} {r['pseudo_birth']['ratio_pre']:.3f}, "
                                              f"A_own {r['pseudo_birth']['a_own_pre']}"), "",
                  "Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):", ""]
        for e in r["exemplars_T15"][:8]:
            lines.append(f"- rank {e['rank']} (a+ {e['a_plus_pre']:.3g} → {e['a_plus_birth']:.3g}): "
                         f"`{e['text']!r}`")
        if len(r["exemplars_T15"]) > 8:
            lines.append(f"- … {len(r['exemplars_T15']) - 8} more in the JSON file")
        lines += ["", f"Carrier's own top-5 at T_{c['t']}:", ""]
        lines += [f"- rank {e['rank']} (a {e['a']:.3g}): `{e['text']!r}`" for e in c["top5"]]
        lines += ["", f"Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at "
                      f"T_{c['t']}), read beside the carrier:", ""]
        for m in r["matched_reading"]:
            lines.append(f"- {m['pole']} (max|a| {m['max_abs_pre']:.4g}, A on the exemplars {m['a_on_E']}):")
            lines += [f"  - rank {e['rank']} (a {e['a']:.3g}): `{e['text']!r}`" for e in m["top3"]]
        lines.append("")
    (out / "raw_examples_zero_gpu.md").write_text("\n".join(lines))
    print("M2_EXAMPLES_OK", len(recs), str(out))
