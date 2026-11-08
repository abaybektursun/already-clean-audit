"""NEURONSHOW scorer and judge-batch builder (studies/neuronshow/SPEC.md §5-§8), laptop only, on banked job outputs.

    python -m infra.pg.neuronshow_score rules  --bank bank/show --results studies/neuronshow/results
    python -m infra.pg.neuronshow_score judge  --bank bank/show --judge studies/neuronshow/judge
    python -m infra.pg.neuronshow_score verdict --bank bank/show --judge studies/neuronshow/judge --results studies/neuronshow/results
    python -m infra.pg.neuronshow_score sha    --bank bank/show

bank/show/<M>/ holds one job's DIR (samples.jsonl, metrics.json, gate.json, dose.json, cells.json, controls.json,
projections.jsonl, STATUS).
- rules: recomputes the gate and the dose rule from the banked samples, asserts they equal the job's, runs the gate's
  registered failing inputs on the real samples (prompt copied in front of each completion; each completion's first 8
  words repeated 10 times: both must drop every prompt), and writes results/gate_<M>.json, dose_<M>.json and
  m3m4_<M>.json (per cell: M3 promoted share, M3 label share, M4 non-repetitive share and mean ratio, 95% bootstrap
  intervals over the 20 samples; identity and realized-cosine flags from cells.json).
- judge: the blind batches (judge/batches/batch_NNN.jsonl, at most 40 items, items of all conditions and models in one
  seeded random order) and the key (judge/key.jsonl): per pole and kept prompt, samples k = 0..9 of the target and
  control cells at the chosen dose and of the prompt's unsteered gate samples; the dose-story pole's target cells at the
  other doses; 2 positive-control items per pole (text = its true label: judge wiring); a 10% seeded retest.
- verdict: reads judge/answers/batch_NNN.jsonl ({item_id, choice, answer}), runs freeze (room checks from unsteered and
  control answers only, results/thresholds.json) and then unblinds the target answers (results/verdict.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from infra.pg import neuronshow_plan as P

QUESTION = ("Which description fits the neuron whose push produced this text? Answer A, B or C, then yes/no: is this "
            "text an answer to the prompt shown?")
LETTERS = "ABC"
SEED = 20261025
N_BOOT = 10_000
PURPOSE = {"letters": 21, "order": 22, "retest": 23}
RETEST_SHARE = 0.10
GAP_MIN, M2_MIN, ROOM_MAX = 0.30, 0.5, 0.70       # H-register: target - control M1 >= 0.30 with target M2 >= 0.5
PC_MIN, PC_SHUF_MAX = 0.90, 0.50                  # judge gate: positive control rate; shuffled-key failing input


def jsonl(p) -> list:
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def write_jsonl(p, rows):
    pathlib.Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def ci(x) -> list:
    x = np.asarray(x, dtype=np.float64)
    m = x[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def share(v) -> dict:
    return {"k": int(sum(v)), "n": len(v), "rate": float(np.mean(v)), "ci95": ci(v)}


def models_in(bank: pathlib.Path) -> list:
    return [m for m in ("Q4B", "G4", "O7") if (bank / m / "metrics.json").exists()]


def load_bank(bank: pathlib.Path, M: str) -> dict:
    d = bank / M
    met = json.loads((d / "metrics.json").read_text())
    assert met["status"] == "complete", f"{M}: job status {met['status']}"
    return {"metrics": met, "rows": jsonl(d / "samples.jsonl"), "gate": json.loads((d / "gate.json").read_text()),
            "dose": json.loads((d / "dose.json").read_text()), "cells": json.loads((d / "cells.json").read_text()),
            "controls": json.loads((d / "controls.json").read_text()), "projections": jsonl(d / "projections.jsonl")}


# ------------------------------------------------------------------------------------------ rules: gate, dose, M3/M4
def gate_of(rows: list, plan: dict, edit=None) -> dict:
    out = {}
    for i, pr in enumerate(plan["prompts"]):
        rs = [r for r in rows if r["stage"] == "gate" and r["prompt_index"] == i]
        if edit:
            rs = [{**r, "completion": edit(r["completion"], pr["text"])} for r in rs]
        out[pr["id"]] = P.gate(rs, pr["text"], plan["rules"])
    return out


def rules_model(b: dict, M: str, plan: dict) -> tuple:
    rules, rows = plan["rules"], b["rows"]
    gate = gate_of(rows, plan)
    assert {k: v for k, v in gate.items()} == b["gate"]["per_prompt"], f"{M}: gate recomputed differs from the job's"
    kept = [k for k, v in gate.items() if v["kept"]]
    assert kept == b["gate"]["kept"]
    fail_copy = gate_of(rows, plan, lambda c, p: p + c)
    fail_rep = gate_of(rows, plan, lambda c, p: " ".join((c.split()[:8] or ["x"]) * 10))
    g = {"model_key": M, "rule": {k: rules[k] for k in ("n_samples", "repeat_cut", "gate_min_nonrep",
                                                         "gate_min_answer_form", "answer_min_chars", "copy_head_chars")},
         "per_prompt": gate, "kept": kept,
         "failing_inputs": {"prompt_copied_in_front": {"kept": [k for k, v in fail_copy.items() if v["kept"]],
                                                       "fired": not any(v["kept"] for v in fail_copy.values())},
                            "first_8_words_x10": {"kept": [k for k, v in fail_rep.items() if v["kept"]],
                                                  "fired": not any(v["kept"] for v in fail_rep.values())}}}
    poles = [p["pole"] for p in P.model_poles(plan, M)]
    grid = plan["models"][M]["doses"]
    dr = [r for r in rows if r["stage"] == "dose"]
    counts = {d: {p: sum(P.nonrep(r["completion"], rules) for r in dr if r["pole"] == p and r["dose"] == d) for p in poles}
              for d in grid}
    dose = P.choose_dose(counts, grid, rules)
    assert dose == b["dose"], f"{M}: dose recomputed differs from the job's"
    return g, {"model_key": M, **dose}


def cell_flags(cells: list) -> dict:
    return {(c["stage"], c["condition"], c["pole"], c["dose"], c["prompt_id"]): c for c in cells}


def m3m4_model(b: dict, M: str, plan: dict) -> dict:
    rules, rows = plan["rules"], b["rows"]
    poles = P.model_poles(plan, M)
    proj = {r["pole"]: r for r in b["projections"]}
    flags = cell_flags(b["cells"])
    d0 = b["dose"]["chosen"]
    out = {"model_key": M, "dose": d0, "dose_rule_met": b["dose"]["met"], "kept": b["gate"]["kept"], "cells": []}
    for p in poles:
        top = [t["id"] for t in (p["top20"] if p["top20"] is not None else proj[p["pole"]]["top_full"])]
        for pid in b["gate"]["kept"]:
            groups = {"unsteered": [r for r in rows if r["stage"] == "gate" and r["prompt_id"] == pid]}
            for d in sorted({r["dose"] for r in rows if r["stage"] == "main" and r["target"] == p["pole"]}):
                for cond in ("target", "control"):
                    rs = [r for r in rows if r["stage"] == "main" and r["target"] == p["pole"] and r["condition"] == cond
                          and r["dose"] == d and r["prompt_id"] == pid]
                    if rs:
                        groups[f"{cond}@{d:g}"] = rs
            for name, rs in groups.items():
                assert len(rs) == rules["n_samples"], f"{M} {p['pole']} {pid} {name}: {len(rs)} rows"
                cond = name.split("@")[0]
                c = {"pole": p["pole"], "group": p["group"], "prompt_id": pid, "condition": cond,
                     "dose": 0 if cond == "unsteered" else rs[0]["dose"], "steered_pole": rs[0]["pole"],
                     "m3_promoted": share([P.m3_promoted(r["completion_token_ids"], top) for r in rs]),
                     "m3_label": share([P.m3_label(r["completion"], p["label_regex"]) for r in rs]),
                     "m4_nonrep": share([P.nonrep(r["completion"], rules) for r in rs]),
                     "m4_mean_ratio": float(np.mean([P.distinct_bigram_ratio(r["completion"]) for r in rs]))}
                if cond != "unsteered":
                    f = flags[("main", cond, rs[0]["pole"], rs[0]["dose"], pid)]
                    c.update({"identity_pass": f["identity_pass"], "cos_step1_median": f["cos_step1_median"],
                              "cos_pass": f["cos_pass"], "counts": f["identity_pass"] and f["cos_pass"]})
                out["cells"].append(c)
    return out


def cmd_rules(bank: pathlib.Path, results: pathlib.Path, plan: dict | None = None):
    plan = plan or P.load()
    results.mkdir(parents=True, exist_ok=True)
    for M in models_in(bank):
        b = load_bank(bank, M)
        g, d = rules_model(b, M, plan)
        (results / f"gate_{M}.json").write_text(json.dumps(g, indent=1) + "\n")
        (results / f"dose_{M}.json").write_text(json.dumps(d, indent=1) + "\n")
        (results / f"m3m4_{M}.json").write_text(json.dumps(m3m4_model(b, M, plan), indent=1, ensure_ascii=False) + "\n")
        print(M, "kept", g["kept"], "dose", d["chosen"], "met", d["met"])


# ------------------------------------------------------------------------------------------ judge batches
def judge_items(banks: dict, plan: dict) -> list:
    """Unordered items: (key fields, text, prompt, true label, decoy labels)."""
    items = []
    n = plan["rules"]["judge_k"]
    for M, b in banks.items():
        rows, d0 = b["rows"], b["dose"]["chosen"]
        for p in P.model_poles(plan, M):
            def add(cond, rs, dose):
                for r in sorted(rs, key=lambda r: r["k"])[:n]:
                    items.append({"model_key": M, "pole": p["pole"], "condition": cond, "steered_pole": r["pole"],
                                  "prompt_id": r["prompt_id"], "dose": dose, "k": r["k"], "kind": "cell",
                                  "text": r["completion"], "prompt": r["prompt"]})
            for pid in b["gate"]["kept"]:
                add("unsteered", [r for r in rows if r["stage"] == "gate" and r["prompt_id"] == pid], 0)
                for cond in ("target", "control"):
                    for d in sorted({r["dose"] for r in rows if r["stage"] == "main" and r["target"] == p["pole"]
                                     and r["condition"] == cond}):
                        add(cond, [r for r in rows if r["stage"] == "main" and r["target"] == p["pole"]
                                   and r["condition"] == cond and r["dose"] == d and r["prompt_id"] == pid], d)
            for j in range(2):
                items.append({"model_key": M, "pole": p["pole"], "condition": "positive_control", "steered_pole": None,
                              "prompt_id": "P1", "dose": None, "k": j, "kind": "positive_control", "text": p["label"],
                              "prompt": plan["prompts"][0]["text"]})
    return items


def cmd_judge(bank: pathlib.Path, judge: pathlib.Path):
    banks = {M: load_bank(bank, M) for M in models_in(bank)}
    assert banks, f"no complete bank under {bank}"
    print(json.dumps(build_judge(banks, P.load(), judge)))


def build_judge(banks: dict, plan: dict, judge: pathlib.Path) -> dict:
    items = judge_items(banks, plan)
    cells = [i for i, it in enumerate(items) if it["kind"] == "cell"]
    g = np.random.default_rng([plan["seed"], PURPOSE["retest"]])
    for i in sorted(g.choice(cells, int(round(RETEST_SHARE * len(cells))), replace=False).tolist()):
        items.append({**items[i], "kind": "retest", "retest_of_index": i})
    labels = {p["pole"]: p for p in plan["poles"]}
    g = np.random.default_rng([plan["seed"], PURPOSE["letters"]])
    truth = np.array((list(LETTERS) * (len(items) // 3 + 1))[:len(items)])[g.permutation(len(items))]
    order = np.random.default_rng([plan["seed"], PURPOSE["order"]]).permutation(len(items))
    pub, key = [], []
    for rank, i in enumerate(order.tolist()):
        it, t = items[i], LETTERS.index(truth[i])
        lp = labels[it["pole"]]
        decoys = [d for d in lp["decoys"]]
        if g.random() < 0.5:
            decoys = decoys[::-1]
        cand = decoys[:t] + [{"pole": it["pole"], "label": lp["label"]}] + decoys[t:]
        iid = f"q{rank:05d}"
        pub.append({"item_id": iid, "text": it["text"], "candidates": [c["label"] for c in cand], "question": QUESTION,
                    "prompt": it["prompt"]})
        key.append({"item_id": iid, **{k: v for k, v in it.items() if k not in ("text", "prompt")},
                    "truth": truth[i], "candidate_poles": [c["pole"] for c in cand], "item_index": i})
    idx = {k["item_index"]: k["item_id"] for k in key}
    for k in key:
        if k["kind"] == "retest":
            k["retest_of"] = idx[k.pop("retest_of_index")]
    (judge / "batches").mkdir(parents=True, exist_ok=True)
    for old in (judge / "batches").glob("batch_*.jsonl"):
        old.unlink()
    m = plan["rules"]["judge_batch_max"]
    nb = 0
    for s in range(0, len(pub), m):
        nb += 1
        write_jsonl(judge / "batches" / f"batch_{nb:03d}.jsonl", pub[s:s + m])
    write_jsonl(judge / "key.jsonl", key)
    summary = {"n_items": len(pub), "n_batches": nb, "by_kind": {}, "by_condition": {},
               "truth_letters": {L: int((truth == L).sum()) for L in LETTERS}, "models": list(banks)}
    for k in key:
        summary["by_kind"][k["kind"]] = summary["by_kind"].get(k["kind"], 0) + 1
        summary["by_condition"][k["condition"]] = summary["by_condition"].get(k["condition"], 0) + 1
    (judge / "manifest.json").write_text(json.dumps({**summary, "batches_sha256": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((judge / "batches").glob("batch_*.jsonl"))},
        "key_sha256": hashlib.sha256((judge / "key.jsonl").read_bytes()).hexdigest()}, indent=1) + "\n")
    return summary


# ------------------------------------------------------------------------------------------ verdict (after the judge)
def answers(judge: pathlib.Path) -> dict:
    out = {}
    for f in sorted((judge / "answers").glob("batch_*.jsonl")):
        for a in jsonl(f):
            assert a["item_id"] not in out, f"duplicate answer {a['item_id']}"
            assert a["choice"] in LETTERS and a["answer"] in ("yes", "no"), a
            out[a["item_id"]] = a
    return out


def cell_rates(key: list, ans: dict) -> dict:
    """(model, pole, prompt, condition, dose) -> {m1: [0/1], m2: [0/1]} over cell items with an answer."""
    out = {}
    for k in key:
        if k["kind"] != "cell" or k["item_id"] not in ans:
            continue
        c = out.setdefault((k["model_key"], k["pole"], k["prompt_id"], k["condition"], k["dose"]), {"m1": [], "m2": []})
        c["m1"].append(int(ans[k["item_id"]]["choice"] == k["truth"]))
        c["m2"].append(int(ans[k["item_id"]]["answer"] == "yes"))
    return out


def judge_gate(key: list, ans: dict, seed: int) -> dict:
    pc = [k for k in key if k["kind"] == "positive_control" and k["item_id"] in ans]
    hit = [int(ans[k["item_id"]]["choice"] == k["truth"]) for k in pc]
    shuf = np.random.default_rng([seed, 24]).permutation([k["truth"] for k in pc])
    hit_s = [int(ans[k["item_id"]]["choice"] == t) for k, t in zip(pc, shuf)]
    rt = [k for k in key if k["kind"] == "retest" and k["item_id"] in ans and k["retest_of"] in ans]
    by = {k["item_id"]: k for k in key}
    agree = [int((ans[k["item_id"]]["choice"] == k["truth"]) == (ans[k["retest_of"]]["choice"] == by[k["retest_of"]]["truth"]))
             for k in rt]
    out = {"positive_control": {"n": len(hit), "rate": float(np.mean(hit)) if hit else None, "min": PC_MIN},
           "shuffled_key_failing_input": {"rate": float(np.mean(hit_s)) if hit_s else None, "max": PC_SHUF_MAX},
           "retest_m1_agreement": {"n": len(agree), "rate": float(np.mean(agree)) if agree else None, "reported": True}}
    out["pass"] = bool(hit) and bool(np.mean(hit) >= PC_MIN and np.mean(hit_s) <= PC_SHUF_MAX)
    return out


def freeze(key: list, ans: dict, plan: dict, m3m4: dict) -> dict:
    """Room checks from unsteered and control answers only (target answers are not read here)."""
    rates = cell_rates([k for k in key if k["condition"] in ("unsteered", "control")], ans)
    out = {}
    for p in [p for p in plan["poles"] if p["group"] == "register"]:
        M = p["model_key"]
        if M not in m3m4:
            continue
        d0 = m3m4[M]["dose"]
        for pid in m3m4[M]["kept"]:
            u = rates.get((M, p["pole"], pid, "unsteered", 0))
            c = rates.get((M, p["pole"], pid, "control", d0))
            ccell = next(x for x in m3m4[M]["cells"] if x["pole"] == p["pole"] and x["prompt_id"] == pid
                         and x["condition"] == "control" and x["dose"] == d0)
            tcell = next(x for x in m3m4[M]["cells"] if x["pole"] == p["pole"] and x["prompt_id"] == pid
                         and x["condition"] == "target" and x["dose"] == d0)
            row = {"unsteered_m1": None if not u else float(np.mean(u["m1"])),
                   "control_m1": None if not c else float(np.mean(c["m1"])),
                   "instrument_ok": ccell["counts"] and tcell["counts"]}
            row["scorable"] = (row["instrument_ok"] and row["unsteered_m1"] is not None and row["control_m1"] is not None
                               and row["unsteered_m1"] <= ROOM_MAX and row["control_m1"] <= ROOM_MAX)
            out[f"{M} {p['pole']} {pid}"] = row
    return out


def verdict(key: list, ans: dict, plan: dict, m3m4: dict, thr: dict, jg: dict) -> dict:
    rates = cell_rates(key, ans)
    neurons, hits_any = {}, 0
    for p in [p for p in plan["poles"] if p["group"] == "register"]:
        M = p["model_key"]
        if M not in m3m4:
            neurons[f"{M} {p['pole']}"] = {"model_key": M, "status": "CANCELLED", "why": "no bank"}
            continue
        d0 = m3m4[M]["dose"]
        cells = []
        for pid in m3m4[M]["kept"]:
            t, c = rates.get((M, p["pole"], pid, "target", d0)), rates.get((M, p["pole"], pid, "control", d0))
            th = thr[f"{M} {p['pole']} {pid}"]
            if not th["scorable"] or not t or not c:
                cells.append({"prompt_id": pid, "scorable": False})
                continue
            m1t, m1c, m2t = float(np.mean(t["m1"])), float(np.mean(c["m1"])), float(np.mean(t["m2"]))
            cells.append({"prompt_id": pid, "scorable": True, "m1_target": m1t, "m1_control": m1c, "m2_target": m2t,
                          "m1_target_ci95": ci(t["m1"]), "m1_control_ci95": ci(c["m1"]),
                          "hit": m1t - m1c >= GAP_MIN and m2t >= M2_MIN, "control_above_target": m1c > m1t})
        sc = [c for c in cells if c["scorable"]]
        r = {"model_key": M, "cells": cells, "n_kept": len(cells), "n_scorable": len(sc),
             "n_hits": sum(c["hit"] for c in sc)}
        if not m3m4[M]["dose_rule_met"]:
            r.update(status="CANCELLED", why="dose rule not met for the model")
        elif not sc:
            r.update(status="CANCELLED", why="no scorable prompt")
        elif sum(c["control_above_target"] for c in sc) > len(sc) / 2:
            r.update(status="CANCELLED", why="control M1 above target on more than half the prompts (generic push)")
        else:
            r["status"] = "REACHES" if r["n_hits"] >= len(cells) / 2 else "DOES NOT REACH"
        hits_any += int(r.get("status") in ("REACHES", "DOES NOT REACH") and r["n_hits"] >= 1)
        neurons[f"{M} {p['pole']}"] = r
    reach = sum(r["status"] == "REACHES" for r in neurons.values())
    if not jg["pass"]:
        v = "CANCELLED (judge gate)"
    elif reach >= 3:
        v = "SUPPORTED"
    elif hits_any < 2:
        v = "REFUTED"
    else:
        v = "NOT DECIDED"
    return {"H_register": v, "n_reach": reach, "n_with_any_hit": hits_any, "judge_gate": jg, "neurons": neurons}


def cmd_verdict(bank: pathlib.Path, judge: pathlib.Path, results: pathlib.Path, plan: dict | None = None) -> dict:
    plan = plan or P.load()
    key, ans = jsonl(judge / "key.jsonl"), answers(judge)
    m3m4 = {M: json.loads((results / f"m3m4_{M}.json").read_text()) for M in models_in(bank)}
    jg = judge_gate(key, ans, plan["seed"])
    thr = freeze(key, ans, plan, m3m4)
    (results / "thresholds.json").write_text(json.dumps({"judge_gate": jg, "room": thr}, indent=1) + "\n")
    v = verdict(key, ans, plan, m3m4, thr, jg)
    (results / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    print(v["H_register"], v["n_reach"], v["n_with_any_hit"])
    return v


def cmd_sha(bank: pathlib.Path):
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(bank)}"
             for f in sorted(bank.rglob("*")) if f.is_file() and f.name != "SHA256.txt"]
    (bank / "SHA256.txt").write_text("\n".join(lines) + "\n")
    print(len(lines), "files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rules", "judge", "verdict", "sha"])
    ap.add_argument("--bank", default="bank/show")
    ap.add_argument("--results", default="studies/neuronshow/results")
    ap.add_argument("--judge", default="studies/neuronshow/judge")
    a = ap.parse_args()
    bank, results, judge = pathlib.Path(a.bank), pathlib.Path(a.results), pathlib.Path(a.judge)
    {"rules": lambda: cmd_rules(bank, results), "judge": lambda: cmd_judge(bank, judge),
     "verdict": lambda: cmd_verdict(bank, judge, results), "sha": lambda: cmd_sha(bank)}[a.cmd]()


if __name__ == "__main__":
    main()
