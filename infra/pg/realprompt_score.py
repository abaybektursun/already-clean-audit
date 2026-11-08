"""REALPROMPT scorer (studies/neuronrw/side_realprompt/PREREG.md §7-§9): presence table, prediction verdicts, instrument checks
and the shown samples, from the banked job outputs bank/x/realprompt/<M>/{metrics.json, plan.json, samples.jsonl}.

    python -m infra.pg.realprompt_score --bank bank/x/realprompt --costs studies/neuronrw/side_realprompt/results/costs.json \
        --json studies/neuronrw/side_realprompt/results/score.json --report studies/neuronrw/side_realprompt/REPORT.md

Laptop analysis of banked JSON only. Presence = NEURONOFF §5.4 (a regex match on prompt[-200:] + completion that
ends inside the completion). Intervals: 95% percentile bootstrap, N_BOOT resamples, numpy default_rng(SEED), over
the 20 (prompt, sample) units of a cell; the push - control gap resamples the units paired.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re

import numpy as np

from infra.pg import realprompt_plan as P
from infra.pg.neuron_rw_analysis import REPEAT_CUT, construct_count, distinct_bigram_ratio

assert P.REPEAT_CUT == REPEAT_CUT, "the plan's loop cut differs from neuron_rw_analysis.REPEAT_CUT"


def load(bank: pathlib.Path) -> dict:
    out = {}
    for m in P.MODELS:
        d = bank / m
        if not d.exists():
            continue
        met = json.loads((d / "metrics.json").read_text())
        assert met["status"] == "complete", f"{m}: job status {met['status']}"
        rows = [json.loads(x) for x in (d / "samples.jsonl").read_text().splitlines()]
        assert len(rows) == met["progress"]["rows_written"], f"{m}: {len(rows)} rows, metrics say {met['progress']}"
        out[m] = {"metrics": met, "plan": json.loads((d / "plan.json").read_text()), "rows": rows}
    assert out, f"no model bank under {bank}"
    return out


def present(regex: str, r: dict) -> int:
    return int(construct_count(regex, r["prompt"], r["completion"]) > 0)


# Exploratory deviation D1 (2026-10-26, written after reading Q4B's unsteered code samples; the frozen rule decides
# the headline): the frozen code rule called 5 of 20 unsteered Q4B code-prompt samples prose although all five are
# Python (import blocks, docstrings: few of its punctuation characters, no closing fence within 128 tokens). D1 also
# calls a completion code when at least D1_MIN_LINES lines of its first 64 tokens start like a line of code.
D1_LINE = re.compile(r"^\s*(import |from \S+ import|def |class |return\b|#|//|/\*|SELECT\b|CREATE\b|INSERT\b|ALTER\b|"
                     r"function\b|const |let |var |public |private |<[A-Za-z!/])", re.M)
D1_MIN_LINES = 2
RULES = ("frozen", "d1")


def code_of(r: dict, rule: str = "frozen") -> bool:
    assert rule in RULES, rule
    frozen = P.is_code(r["completion"], r["head64"])
    return frozen if rule == "frozen" else frozen or len(D1_LINE.findall(r["head64"])) >= D1_MIN_LINES


def units(rows: list) -> dict:
    """{(prompt_index, k): row}; asserts one row per unit."""
    u = {(r["prompt_index"], r["k"]): r for r in rows}
    assert len(u) == len(rows), "two rows share a (prompt, sample) unit"
    return u


def boot_rate(x: list) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(P.SEED)
    b = x[rng.integers(0, len(x), (P.N_BOOT, len(x)))].mean(1)
    return {"n": len(x), "rate": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}


def boot_gap(a: list, b: list) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "unpaired gap"
    rng = np.random.default_rng(P.SEED)
    idx = rng.integers(0, len(a), (P.N_BOOT, len(a)))
    g = a[idx].mean(1) - b[idx].mean(1)
    return {"n": len(a), "gap": float(a.mean() - b.mean()), "lo": float(np.percentile(g, 2.5)),
            "hi": float(np.percentile(g, 97.5))}


def realized_check(rows: list) -> dict:
    cells = {}
    for r in rows:
        if r["dose"] > 0:
            cells.setdefault((r["pole"], r["dose"]), []).append(r)
    out, worst = [], None
    for (p, d), rs in sorted(cells.items()):
        c = [r["cos_step1"] for r in rs if r["cos_step1"] is not None]
        q = [r["ratio_step1"] for r in rs if r["ratio_step1"] is not None]
        assert c and len(c) == len(q), f"{p} dose {d}: no step-1 readout"
        cell = {"pole": p, "dose": d, "n": len(c), "cos_median": float(np.median(c)), "cos_min": float(min(c)),
                "ratio_median": float(np.median(q))}
        cell["pass"] = cell["cos_median"] >= P.COS_MIN and P.RATIO_BAND[0] <= cell["ratio_median"] <= P.RATIO_BAND[1]
        out.append(cell)
        if worst is None or cell["cos_median"] < worst["cos_median"]:
            worst = cell
    return {"n_cells": len(out), "n_pass": sum(c["pass"] for c in out), "pass": all(c["pass"] for c in out),
            "worst_cell": worst, "min_row_cos": min(c["cos_min"] for c in out),
            "failing_cells": [c for c in out if not c["pass"]]}


def code_controls(rows: list, rule: str) -> dict:
    z = [r for r in rows if r["dose"] == 0]
    pos = [code_of(r, rule) for r in z if r["domain"] == "code"]
    neg = [code_of(r, rule) for r in z if r["domain"] != "code"]
    out = {"positive": {"n": len(pos), "code_share": float(np.mean(pos)) if pos else None},
           "negative": {"n": len(neg), "code_share": float(np.mean(neg)) if neg else None}}
    out["pass"] = None if not pos else (out["positive"]["code_share"] >= P.CODE_POS_MIN and bool(neg) and
                                        out["negative"]["code_share"] <= P.CODE_NEG_MAX)
    return out


def verdict(cells: dict, gate_reason: str | None) -> dict:
    """cells: {dose: {"push": [0/1 by unit], "ctrl": [...], "code": [bool by push unit]}} for positive doses.
    Order: a failed model gate, no dose at CODE_SHARE_MIN, no room, then the bar."""
    if gate_reason:
        return {"verdict": "NOT SCORABLE", "reason": gate_reason, "d_star": None}
    d_star = next((d for d in sorted(cells) if np.mean(cells[d]["code"]) >= P.CODE_SHARE_MIN), None)
    if d_star is None:
        return {"verdict": "NOT SCORABLE", "reason": f"no positive dose keeps {P.CODE_SHARE_MIN:.0%} of push samples code",
                "d_star": None}
    c = cells[d_star]
    g = boot_gap(c["push"], c["ctrl"])
    out = {"d_star": d_star, "code_share": float(np.mean(c["code"])), "push_rate": float(np.mean(c["push"])),
           "ctrl_rate": float(np.mean(c["ctrl"])), **g}
    if out["ctrl_rate"] > P.ROOM_MAX:
        return {**out, "verdict": "NOT SCORABLE",
                "reason": f"control presence {out['ctrl_rate']:.2f} > {P.ROOM_MAX} (no room)"}
    if g["hi"] < P.GAP_BAR:
        return {**out, "verdict": "REFUTED", "reason": f"interval upper end {g['hi']:.3f} < {P.GAP_BAR}"}
    if g["gap"] >= P.GAP_BAR and g["lo"] > 0:
        return {**out, "verdict": "SUPPORTED", "reason": f"gap {g['gap']:.3f} >= {P.GAP_BAR}, lower end {g['lo']:.3f} > 0"}
    return {**out, "verdict": "NOT DECIDED", "reason": f"interval [{g['lo']:.3f}, {g['hi']:.3f}] neither below {P.GAP_BAR} "
                                                      f"nor above 0 with gap >= {P.GAP_BAR}"}


def score_model(m: str, b: dict, rule: str = "frozen") -> dict:
    rows, met = b["rows"], b["metrics"]
    g = met["gates"]
    real = realized_check(rows)
    code = code_controls(rows, rule)
    gates = {"identity": g["identity"], "realized_failing_input": g["realized_failing_input"], "realized": real,
             "code_rule": code}
    reasons = [n for n, ok in [("identity", g["identity"]["pass"] and g["identity"]["failing_input"]["fired"]),
                               ("realized failing input", g["realized_failing_input"]["fired"]),
                               ("realized change", real["pass"]), ("code rule controls", code["pass"] is not False)]
               if not ok]
    gate_reason = "model gate failed: " + ", ".join(reasons) if reasons else None
    by = {}
    for r in rows:
        by.setdefault((r["pole"], r["dose"]), []).append(r)
    zero = {}
    for r in rows:
        if r["dose"] == 0:
            zero.setdefault(r["domain"], []).append(r)
    preds = []
    for pole, spec in P.CODE.items():
        if spec["model"] != m:
            continue
        rx, ctrl = spec["regex"], spec["ctrl"]
        table, cells = [], {}
        z = sorted(zero["code"], key=lambda r: (r["prompt_index"], r["k"]))
        table.append({"dose": 0, "push": boot_rate([present(rx, r) for r in z]), "ctrl": None,
                      "code_share_push": float(np.mean([code_of(r, rule) for r in z]))})
        for d in P.DOSES[m][1:]:
            up, uc = units(by[(pole, d)]), units(by[(ctrl, d)])
            assert set(up) == set(uc) and len(up) == 2 * P.N_SAMPLES, f"{pole} dose {d}: units differ"
            keys = sorted(up)
            push = [present(rx, up[k]) for k in keys]
            cc = [present(rx, uc[k]) for k in keys]
            cells[d] = {"push": push, "ctrl": cc, "code": [code_of(up[k], rule) for k in keys]}
            table.append({"dose": d, "push": boot_rate(push), "ctrl": boot_rate(cc), "gap": boot_gap(push, cc),
                          "code_share_push": float(np.mean(cells[d]["code"])),
                          "code_share_ctrl": float(np.mean([code_of(uc[k], rule) for k in keys]))})
        preds.append({"pole": pole, "label": spec["label"], "ctrl": ctrl, "regex": rx, "table": table,
                      **verdict(cells, gate_reason)})
    return {"gates": gates, "gate_failures": reasons, "predictions": preds}


def theme_hits(theme: str, r: dict) -> int:
    return construct_count(theme, r["prompt"], r["completion"])


def shown(m: str, b: dict) -> list:
    """Per pole and positive dose: the two best samples (fluent first, then theme matches, then distinct-bigram
    ratio, then prompt and k), the unsteered sample of the first one's prompt and k, and the theme counts."""
    rows = b["rows"]
    zero = {(r["prompt_index"], r["k"]): r for r in rows if r["dose"] == 0}
    out = []
    for d in P.directions(m):
        if d["role"] != "pole":
            continue
        z = [r for r in rows if r["dose"] == 0 and r["domain"] == d["domain"]]
        ent = {"pole": d["pole"], "domain": d["domain"], "source": d["source"], "theme": d["theme"],
               "theme_dose0": sum(theme_hits(d["theme"], r) > 0 for r in z), "n_dose0": len(z), "doses": []}
        for dose in P.DOSES[m][1:]:
            rs = [r for r in rows if r["pole"] == d["pole"] and r["dose"] == dose]
            assert len(rs) == 2 * P.N_SAMPLES, f"{d['pole']} dose {dose}: {len(rs)} rows"
            info = []
            for r in rs:
                dbr = distinct_bigram_ratio(r["completion"])
                info.append({"r": r, "dbr": dbr, "fluent": dbr >= P.REPEAT_CUT, "hits": theme_hits(d["theme"], r)})
            info.sort(key=lambda x: (-x["fluent"], -x["hits"], -x["dbr"], x["r"]["prompt_index"], x["r"]["k"]))
            best = info[:2]
            u = zero[(best[0]["r"]["prompt_index"], best[0]["r"]["k"])]
            ent["doses"].append({"dose": dose, "n": len(rs), "n_fluent": sum(x["fluent"] for x in info),
                                 "theme_steered": sum(x["hits"] > 0 for x in info),
                                 "best": [{"prompt": x["r"]["prompt"], "k": x["r"]["k"], "dbr": x["dbr"],
                                           "loop": not x["fluent"], "hits": x["hits"], "text": x["r"]["head60"]}
                                          for x in best],
                                 "unsteered": {"prompt": u["prompt"], "k": u["k"],
                                               "dbr": distinct_bigram_ratio(u["completion"]), "text": u["head60"]}})
        out.append(ent)
    return out


def score(banks: dict) -> dict:
    return {m: {**score_model(m, b), "d1": score_model(m, b, "d1"), "shown": shown(m, b), "status": b["metrics"]["status"],
                "revision": b["metrics"].get("revision"), "rows": len(b["rows"]),
                "gen_s": b["metrics"]["throughput"]["gen_s"], "load_s": b["metrics"].get("load_s")}
            for m, b in banks.items()}


# ------------------------------------------------------------------------------------------------ report
def f2(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def rate(c) -> str:
    return "—" if c is None else f"{c['rate']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"


def pre(t: str) -> str:
    return "<pre>" + html.escape(t) + "</pre>"


def qprompt(p: str) -> str:
    return "`" + p.replace("\n", "\\n").replace("`", "ˋ") + "`"


def render(s: dict, costs: dict | None) -> str:
    L = ["# REALPROMPT — report", "",
         "Pre-registration: `studies/neuronrw/side_realprompt/PREREG.md`. Plan: `infra/pg/realprompt_plan.py`. Job: "
         "`infra/aws_pg/jobs/realprompt.py`. Scorer: `infra/pg/realprompt_score.py`. Banks: `bank/x/realprompt/<M>/`. "
         "Every number below is computed from the banks by the scorer; `results/score.json` holds them all.", ""]
    preds = [(m, p) for m in s for p in s[m]["predictions"]]
    cnt = {m: {} for m in s}
    for m, p in preds:
        cnt[m][p["verdict"]] = cnt[m].get(p["verdict"], 0) + 1
    d1c = {}
    for m in s:
        for p in s[m]["d1"]["predictions"]:
            d1c[p["verdict"]] = d1c.get(p["verdict"], 0) + 1
    if preds:
        top = max(((m, p, t) for m, p in preds for t in p["table"][1:]), key=lambda x: x[2]["push"]["rate"])
        L += ["**Result.** Frozen verdicts: " + "; ".join(f"{m} " + ", ".join(f"{n} {v}" for v, n in sorted(c.items()))
                                                      for m, c in cnt.items() if c) +
              ". Under the exploratory code rule D1 (§1b): " + ", ".join(f"{n} {v}" for v, n in sorted(d1c.items())) +
              f". The highest push presence of any code neuron at any dose is {top[1]['label']} ({top[0]}) at dose "
              f"{top[2]['dose']}: {rate(top[2]['push'])} against its control's {rate(top[2]['ctrl'])}.", ""]
    L += ["## 1. Prediction verdicts", "",
          f"Registered: at d* (the smallest positive dose where at least {P.CODE_SHARE_MIN:.0%} of the push samples are code), "
          f"construct presence under the push minus under the same-layer random control is at least {P.GAP_BAR}. "
          f"REFUTED when the 95% interval's upper end is below {P.GAP_BAR}; SUPPORTED when the gap is at least {P.GAP_BAR} "
          "and the lower end is above 0; otherwise NOT DECIDED; NOT SCORABLE when a gate failed, no dose keeps code, or "
          f"the control's presence at d* is above {P.ROOM_MAX}.", "",
          "| model | code neuron | control | d* | code share at d* | push | control | gap [95% CI] | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, p in preds:
        gap = "—" if p.get("gap") is None else f"{p['gap']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"
        L.append(f"| {m} | {p['label']} {p['pole']} | {p['ctrl']} | {p['d_star'] if p['d_star'] is not None else '—'} | "
                 f"{f2(p.get('code_share'))} | {f2(p.get('push_rate'))} | {f2(p.get('ctrl_rate'))} | {gap} | {p['verdict']} |")
    L += [""] + [f"- {p['label']} ({m} {p['pole']}): {p['verdict']} — {p['reason']}." for m, p in preds] + [""]
    d1 = {(m, p["pole"]): p for m in s for p in s[m]["d1"]["predictions"]}
    L += ["### 1b. Exploratory deviation D1 (not the headline)", "",
          "Dated 2026-10-26, written after reading Q4B's unsteered code samples: the frozen code rule called 5 of 20 of "
          "them prose although all five are Python (import blocks and docstrings carry few of its punctuation characters "
          f"and no closing fence within 128 tokens). D1 also calls a completion code when at least {D1_MIN_LINES} lines of "
          "its first 64 tokens start like a line of code (`" + D1_LINE.pattern + "`). Everything "
          "else as registered. Code-rule controls under D1: " +
          "; ".join(f"{m} code prompts {f2(s[m]['d1']['gates']['code_rule']['positive']['code_share'])}, prose prompts "
                    f"{f2(s[m]['d1']['gates']['code_rule']['negative']['code_share'])}" for m in s) + ".", "",
          "| model | code neuron | d* | code share at d* | push | control | gap [95% CI] | verdict under D1 |",
          "|---|---|---|---|---|---|---|---|"]
    for m, p in preds:
        q = d1[(m, p["pole"])]
        gap = "—" if q.get("gap") is None else f"{q['gap']:+.2f} [{q['lo']:+.2f}, {q['hi']:+.2f}]"
        L.append(f"| {m} | {q['label']} {q['pole']} | {q['d_star'] if q['d_star'] is not None else '—'} | "
                 f"{f2(q.get('code_share'))} | {f2(q.get('push_rate'))} | {f2(q.get('ctrl_rate'))} | {gap} | {q['verdict']} |")
    L.append("")
    L += ["## 2. Presence table per code neuron and dose", "",
          "Presence rate over 20 samples (2 prompts × 10) with its 95% bootstrap interval; dose 0 is the unsteered "
          "code-prompt samples (shared by every code neuron of the model); gap = push − control, paired by (prompt, "
          "sample). Code share = share of samples that are code by the frozen code rule.", ""]
    for m, p in preds:
        L += [f"### {m} {p['pole']} — {p['label']} (control {p['ctrl']}; regex `{p['regex']}`)", "",
              "| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |",
              "|---|---|---|---|---|---|---|---|"]
        for t, u in zip(p["table"], d1[(m, p["pole"])]["table"]):
            gap = "—" if "gap" not in t else f"{t['gap']['gap']:+.2f} [{t['gap']['lo']:+.2f}, {t['gap']['hi']:+.2f}]"
            L.append(f"| {t['dose']} | {rate(t['push'])} | {rate(t['ctrl'])} | {gap} | {f2(t['code_share_push'])} | "
                     f"{f2(t.get('code_share_ctrl'))} | {f2(u['code_share_push'])} | {f2(u.get('code_share_ctrl'))} |")
        L.append("")
    L += ["## 3. Instrument checks", "",
          "| model | identity: max abs logit diff at dose 0 (must be 0) | failing input (dose 1) | realized cos failing "
          "input (per-forward hook; must be < 0.95) | step-1 realized cos: cells passing | worst cell median cos | "
          "lowest row cos | code rule: dose-0 code prompts (≥ 0.80) | dose-0 prose prompts (≤ 0.20) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in s:
        g = s[m]["gates"]
        i, rf, re_, cr = g["identity"], g["realized_failing_input"], g["realized"], g["code_rule"]
        L.append(f"| {m} | {i['max_abs_diff']:.3g} | {i['failing_input']['max_abs_diff']:.3g} | "
                 f"max {max(rf['cos_step1']):.3f} | {re_['n_pass']}/{re_['n_cells']} | "
                 f"{re_['worst_cell']['cos_median']:.4f} ({re_['worst_cell']['pole']}, dose {re_['worst_cell']['dose']}) | "
                 f"{re_['min_row_cos']:.4f} | {f2(cr['positive']['code_share'])} (n {cr['positive']['n']}) | "
                 f"{f2(cr['negative']['code_share'])} (n {cr['negative']['n']}) |")
    L += ["", "Two-instance token identity: not run (one instance per model). Gate failures per model: " +
          "; ".join(f"{m}: {', '.join(s[m]['gate_failures']) or 'none'}" for m in s) + ".", ""]
    if costs:
        L += ["## 4. Instance-hours and cost", "", "| model | instance | type | hours | USD |", "|---|---|---|---|---|"]
        for r in costs["instances"]:
            L.append(f"| {r['model']} | {r['instance_id']} | {r['type']} | {r['hours']:.3f} | {r['usd']:.2f} |")
        L += [f"| total | | | {costs['total_hours']:.3f} | {costs['total_usd']:.2f} |", "", costs["source"], ""]
    L += ["## 5. Sample-check notes (read after the run)", "",
          "- The frozen code rule under-calls Python: import blocks and docstrings carry few of its punctuation "
          "characters (5 of 20 unsteered Q4B code samples, all Python, read as prose; D1 in §1b). It also calls prose "
          "code when the prose contains a fence line (one Q4B news sample opens with two fence lines, then a news "
          "paragraph).",
          "- The distinct-bigram loop flag splits on word characters, so repetition inside a word passes it: G4 "
          "L26N9472- at dose 8 writes \"folk typessss typesss\" with ratio 0.90.", ""]
    L += ["## 6. Samples", "",
          f"Per pole and positive dose: the two best samples (fluent first — distinct-bigram ratio ≥ {P.REPEAT_CUT} — then "
          "most theme-regex matches, then highest ratio) and the unsteered sample of the same prompt and sample index. "
          f"First {P.SHOW_TOKENS} tokens, verbatim. **[loop]** marks a ratio below {P.REPEAT_CUT}. The theme regex was written "
          "from the label before this run; its counts describe the cells and are not a test.", ""]
    for m in s:
        for e in s[m]["shown"]:
            L += [f"### {m} {e['pole']} — {e['domain']} ({e['source']})", "",
                  f"Theme regex `{e['theme']}`: dose 0 {e['theme_dose0']}/{e['n_dose0']}; " +
                  ", ".join(f"dose {x['dose']} {x['theme_steered']}/{x['n']} (fluent {x['n_fluent']})" for x in e["doses"]), ""]
            for x in e["doses"]:
                u = x["unsteered"]
                L.append(f"**Dose {x['dose']}.** Unsteered, {qprompt(u['prompt'])} k {u['k']}"
                         f"{' **[loop]**' if u['dbr'] < P.REPEAT_CUT else ''}:")
                L.append(pre(u["text"]))
                for j, b in enumerate(x["best"]):
                    L.append(f"Steered {j + 1}, {qprompt(b['prompt'])} k {b['k']}, theme matches {b['hits']}, ratio "
                             f"{b['dbr']:.2f}{' **[loop]**' if b['loop'] else ''}:")
                    L.append(pre(b["text"]))
                L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--costs", default=None)
    a = ap.parse_args()
    s = score(load(pathlib.Path(a.bank)))
    pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.json).write_text(json.dumps(s, indent=1) + "\n")
    costs = json.loads(pathlib.Path(a.costs).read_text()) if a.costs else None
    pathlib.Path(a.report).write_text(render(s, costs))
    for m in s:
        for p in s[m]["predictions"]:
            print(m, p["pole"], p["label"], p["verdict"], p["reason"])


if __name__ == "__main__":
    main()
