"""Weight-free dry run of the NEURONSHOW job and scorer (studies/neuronshow/SPEC.md §10).

    python -m infra.aws_pg.jobs.neuronshow_dryrun --out <dir>

1. The real plan without a model: 15 poles with labels, decoys and regexes; row counts per model; the frozen rules
   on written inputs (gate: 20 good answers kept, 20 prompt copies dropped, 20 loops dropped, 20 code fences
   dropped; dose rule picks the largest qualifying dose and reports met = False when none qualifies; the control
   draw stays within 10% and skips study neurons).
2. The job on a tiny random model of each architecture (nothing downloaded) with a tiny plan of the same shape
   (two register poles in layers 0-1, the six prompts, 4 samples per cell, 12 new tokens, 8 rows per call), Q4B's
   architecture with the known number read through neuron_rw.Job.g_known on a fake corpus (neuron_rw_dryrun);
   then three abort paths: a known number banked 1 nat off, a G-proj banked list from another column, and a grid
   dose 0 (its identity failing input cannot fire). Each must end with status aborted_gate and STATUS-free metrics.
3. The scorer on the job's own outputs: gate and dose recomputed equal to the job's, the gate's failing inputs fire,
   m3m4 cells complete, judge batches (<= 40 items, conditions mixed, key complete, truth letters balanced), and the
   verdict on planted answers: target right and control wrong -> SUPPORTED; every answer wrong -> REFUTED; positive
   controls wrong -> CANCELLED (judge gate); control right as often as target -> no neuron reaches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import shutil

import numpy as np
import torch

from infra.aws_pg.jobs import neuronshow as J
from infra.aws_pg.jobs.neuron_off_dryrun import BOS, tiny_model, tiny_tokenizer
from infra.pg import neuronshow_plan as P
from infra.pg import neuronshow_score as S

ARCH = {"Q4B": "qwen3", "G4": "gemma3", "O7": "olmo3"}


def check_real_plan() -> dict:
    plan = P.load()
    r = plan["rules"]
    assert len(plan["poles"]) == 15 and len(plan["prompts"]) == 6
    assert sum(p["group"] == "register" for p in plan["poles"]) == 6
    for p in plan["poles"]:
        assert len(p["decoys"]) == 2 and all(d["label"] for d in p["decoys"]) and p["label"]
        assert p["pole"] not in [d["pole"] for d in p["decoys"]]
        assert p["top20"] is None or len(p["top20"]) == 20
    rows = {}
    for M, spec in plan["models"].items():
        ps = P.model_poles(plan, M)
        story = 1 if spec["dose_story"] else 0
        rows[M] = {"gate": 6 * r["n_samples"], "dose": len(ps) * len(spec["doses"]) * r["n_samples"],
                   "main_max": 6 * (2 * len(ps) + story * (len(spec["doses"]) - 1)) * r["n_samples"]}
    # gate rule on written inputs
    pr = plan["prompts"][0]["text"]
    good = [{"completion": f" Tell her that losing a job happens to many people, number {k}, and offer to help her "
                           f"update her resume this week."} for k in range(20)]
    copy_ = [{"completion": " " + pr + " Tell her it will be fine."} for _ in range(20)]
    loop = [{"completion": " sorry" * 40} for _ in range(20)]
    fence = [{"completion": " ```python\nprint('tell her it is fine and that she will find a job')\n```"} for _ in range(20)]
    assert P.gate(good, pr, r)["kept"]
    assert not P.gate(copy_, pr, r)["kept"] and not P.gate(loop, pr, r)["kept"] and not P.gate(fence, pr, r)["kept"]
    mix = good[:14] + loop[:6]
    assert not P.gate(mix, pr, r)["kept"] and P.gate(good[:15] + loop[:5], pr, r)["kept"]
    # dose rule
    d = P.choose_dose({4: {"a": 20, "b": 19}, 8: {"a": 15, "b": 16}, 16: {"a": 14, "b": 20}, 32: {"a": 3, "b": 2}},
                      [4, 8, 16, 32], r)
    assert d["chosen"] == 8 and d["met"], d
    d = P.choose_dose({4: {"a": 14}, 8: {"a": 2}}, [4, 8], r)
    assert d["chosen"] == 4 and not d["met"], d
    # control draw
    norms = np.linspace(0.5, 1.5, 200).astype(np.float32)
    c = P.draw_control(norms, 3, 100, -1, {100, 101, 99}, 0, plan)
    assert c["neuron"] not in (99, 100, 101) and abs(c["ratio"] - 1) <= 0.10 and c["control"].endswith("-")
    assert c == P.draw_control(norms, 3, 100, -1, {100, 101, 99}, 0, plan)
    return rows


def tiny_plan(model, tok, M: str, doses=(2, 4)) -> dict:
    from infra.pg import neuron_rw_core as R
    plan = copy.deepcopy(P.load())
    ids = [torch.tensor([R.prompt_ids(tok, "import numpy as np\nThe cat sat on the mat. " * 4)])]
    rms = [x["rms"] for x in R.residual_rms(model, ids, 1)["layers"]]
    plan["models"] = {M: {**plan["models"][M], "n_layers": len(rms), "rms": rms, "doses": list(doses),
                          "dose_story": "L01N9+"}}
    allv = torch.arange(R.unembedding(model).shape[0])
    poles = []
    for l, n, s in ((0, 7, -1), (1, 9, 1)):
        pid = P.pole_id(l, n, s)
        pr = R.vocab_projection(model, tok, [(pid, l, n, s)], allv, k=20)[pid]
        poles.append({"model_key": M, "pole": pid, "layer": l, "neuron": n, "sign": s, "handoff_number": 0,
                      "group": "register", "label": f"tiny label {pid} the cat", "label_regex": r"\bcat\b",
                      "top20": None if l == 0 else [{"id": t["id"], "str": t["str"]} for t in pr["top_full"]],
                      "top_full10_banked": [t["id"] for t in pr["top_full"][:10]], "w_down_norm_banked": pr["w_down_norm"],
                      "decoys": [{"pole": f"L00N{90 + j}+", "label": f"decoy {j} import numpy"} for j in range(2)]})
    plan["poles"] = poles
    plan["sampling"] = {**plan["sampling"], "max_new_tokens": 12, "max_rows": 8}
    plan["rules"] = {**plan["rules"], "n_samples": 4, "gate_min_nonrep": 0, "gate_min_answer_form": 0, "dose_min_nonrep": 0,
                     "judge_k": 2}
    return plan


def known_tiny(model, tok, out: pathlib.Path, offset: float = 0.0):
    from infra.aws_pg.jobs.neuron_off_dryrun import fake_corpus
    from infra.aws_pg.jobs.neuron_rw_dryrun import fake_plans
    tokens = fake_corpus(tok)
    plans = fake_plans(model, tok, tokens, "qwen3")
    k0 = copy.deepcopy(plans["arm0"]["known"])
    k0["own_clean_nll_mean_banked"] += offset
    return J.known_from(model, tok, plans["arm2"], "dry", k0, tokens, out)


def run_tiny(M: str, out: pathlib.Path, plan_edit=None, known_offset: float | None = None) -> tuple:
    model, tok = tiny_model(ARCH[M]), tiny_tokenizer(BOS[ARCH[M]])
    plan = tiny_plan(model, tok, M)
    if plan_edit:
        plan_edit(plan)
    known = None if known_offset is None else known_tiny(model, tok, out, known_offset)
    meta = {"model_key": M, "dry_run": True, "plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest()}
    return J.Show(model, tok, M, plan, out, meta, known), plan


def check_tiny(root: pathlib.Path) -> dict:
    rep, plans = {}, {}
    bank = root / "bank"
    for M in ARCH:
        out = bank / M
        show, plan = run_tiny(M, out, known_offset=0.0 if M == "Q4B" else None)
        m = show.run()
        disk = json.loads((out / "metrics.json").read_text())
        assert m["status"] == disk["status"] == "complete"
        rows = S.jsonl(out / "samples.jsonl")
        n, ps = plan["rules"]["n_samples"], P.model_poles(plan, M)
        grid = plan["models"][M]["doses"]
        assert len(rows) == 6 * n + len(ps) * len(grid) * n + 6 * (2 * len(ps) + len(grid) - 1) * n, len(rows)
        assert all(isinstance(r["completion"], str) for r in rows)
        st = [r for r in rows if r["pole"] is not None]
        assert all(r["cos_step1"] > 0.99 for r in st), min(r["cos_step1"] for r in st)
        assert all(r["cos_step1"] is None for r in rows if r["pole"] is None)
        cells = json.loads((out / "cells.json").read_text())
        assert all(c["identity_pass"] and c["identity_failing_input_fired"] and c["cos_pass"] for c in cells)
        seeds = {(r["prompt_index"], r["call"]): r["seed"] for r in rows}
        assert all(s == 20261025 + 1000 * i + c for (i, c), s in seeds.items())
        assert len({r["seed"] for r in rows if r["prompt_index"] == 0}) == max(c for i, c in seeds if i == 0) + 1
        ctl = json.loads((out / "controls.json").read_text())
        assert all(abs(d["ratio"] - 1) <= 0.10 for d in ctl.values())
        g = disk["gates"]
        assert g["g_proj"]["pass"] and not g["g_proj"]["failing_input_pass"] and g["realized_failing_input"]["fired"]
        if M == "Q4B":
            kn = g["known"]
            assert kn["pass"] and not kn["failing_input"]["pass"], kn
            rep["known"] = {"clean": kn["own_clean_nll_mean"]["ours"], "dnll": kn["own_dnll_mean"]["ours"],
                            "failing_shift": kn["failing_input"]["shift"]}
        (out / "STATUS").write_text("exit=0\n")
        plans[M] = plan
        rep[M] = {"rows": len(rows), "cells": len(cells), "dose": disk["dose"]["chosen"], "kept": disk["gate"]["kept"],
                  "min_cos_step1": min(r["cos_step1"] for r in st)}
    return rep, plans


def check_aborts(root: pathlib.Path) -> dict:
    out = {}

    def expect_abort(name, M, **kw):
        d = root / "abort" / name
        show, _ = run_tiny(M, d, **kw)
        try:
            show.run()
            raise RuntimeError(f"{name}: the job ran through a gate that should stop it")
        except AssertionError as e:
            assert "gate failed" in str(e), e
        assert json.loads((d / "metrics.json").read_text())["status"] == "aborted_gate"
        out[name] = "aborted_gate"

    expect_abort("known_off_by_1_nat", "Q4B", known_offset=1.0)

    def wrong_list(plan):
        plan["poles"][1]["top_full10_banked"] = plan["poles"][0]["top_full10_banked"]
    expect_abort("g_proj_wrong_column", "G4", plan_edit=wrong_list)

    def dose0(plan):
        plan["models"][next(iter(plan["models"]))]["doses"] = [0, 2]
    expect_abort("identity_dose0", "O7", plan_edit=dose0)
    return out


def merged_plan(plans: dict) -> dict:
    p = copy.deepcopy(next(iter(plans.values())))
    p["models"] = {M: pl["models"][M] for M, pl in plans.items()}
    p["poles"] = [q for pl in plans.values() for q in pl["poles"]]
    return p


def check_scorer(root: pathlib.Path, plans: dict) -> dict:
    plan = merged_plan(plans)
    bank, res, jd = root / "bank", root / "results", root / "judge"
    S.cmd_rules(bank, res, plan)
    for M in plans:
        mm = json.loads((res / f"m3m4_{M}.json").read_text())
        assert mm["cells"] and all(c["m3_promoted"]["n"] == plan["rules"]["n_samples"] for c in mm["cells"])
    # the gate's failing inputs under the real thresholds, on the tiny model's real samples planted with good text
    real = P.load()
    b = S.load_bank(bank, "Q4B")
    rows = [{**r, "completion": f" Tell her it is not her fault, and that you are proud of her, reason {r['k']} {r['prompt_index']}."}
            if r["stage"] == "gate" else r for r in b["rows"]]
    rows = [r for r in rows if r["stage"] == "gate"]
    rows = [{**r, "k": k} for i in range(6) for k, r in enumerate([x for x in rows if x["prompt_index"] == i] * 5)]
    gp = S.gate_of(rows, real)
    assert all(v["kept"] for v in gp.values()), gp
    assert not any(v["kept"] for v in S.gate_of(rows, real, lambda c, p: p + c).values())
    assert not any(v["kept"] for v in S.gate_of(rows, real, lambda c, p: " ".join((c.split()[:8] or ["x"]) * 10)).values())
    banks = {M: S.load_bank(bank, M) for M in plans}
    summ = S.build_judge(banks, plan, jd)
    pub = [x for f in sorted((jd / "batches").glob("batch_*.jsonl")) for x in S.jsonl(f)]
    key = S.jsonl(jd / "key.jsonl")
    assert len(pub) == len(key) == summ["n_items"] and all(len(S.jsonl(f)) <= 40 for f in (jd / "batches").glob("*.jsonl"))
    assert set(pub[0]) == {"item_id", "text", "candidates", "question", "prompt"}
    kb = {k["item_id"]: k for k in key}
    for x in pub:
        k = kb[x["item_id"]]
        lab = next(p["label"] for p in plan["poles"] if p["pole"] == k["pole"])
        assert x["candidates"][S.LETTERS.index(k["truth"])] == lab and len(set(x["candidates"])) == 3
    first = [kb[x["item_id"]]["condition"] for x in pub[:40]]
    assert len(set(first)) >= 3, first
    assert max(summ["truth_letters"].values()) - min(summ["truth_letters"].values()) <= 1

    def plant(fn):
        (jd / "answers").mkdir(exist_ok=True)
        for f in (jd / "answers").glob("*.jsonl"):
            f.unlink()
        S.write_jsonl(jd / "answers" / "batch_001.jsonl", [{"item_id": k["item_id"], **fn(k)} for k in key])
        return S.cmd_verdict(bank, jd, res, plan)["H_register"]

    wrong = lambda k: S.LETTERS[(S.LETTERS.index(k["truth"]) + 1) % 3]  # noqa: E731
    right_t = lambda k: {"choice": k["truth"] if k["condition"] in ("target", "positive_control") else wrong(k), "answer": "yes"}  # noqa: E731
    out = {"supported": plant(right_t),
           "refuted": plant(lambda k: {"choice": wrong(k), "answer": "yes"} if k["kind"] != "positive_control"
                            else {"choice": k["truth"], "answer": "yes"}),
           "judge_gate": plant(lambda k: {"choice": wrong(k), "answer": "yes"}),
           "generic": plant(lambda k: {"choice": k["truth"] if k["condition"] != "unsteered" else wrong(k), "answer": "yes"})}
    assert out["supported"] == "SUPPORTED" and out["refuted"] == "REFUTED", out
    assert out["judge_gate"] == "CANCELLED (judge gate)" and out["generic"] != "SUPPORTED", out
    S.cmd_sha(bank)
    return {"judge": summ, "verdicts": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rep = {"real_plan_rows": check_real_plan()}
    rep["tiny"], plans = check_tiny(root)
    rep["aborts"] = check_aborts(root)
    rep["scorer"] = check_scorer(root, plans)
    print("NEURONSHOW_DRYRUN_OK", json.dumps(rep))


if __name__ == "__main__":
    main()
