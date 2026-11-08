"""Weight-free dry run of the REALPROMPT job and scorer (studies/neuronrw/side_realprompt/PREREG.md §10).

    python -m infra.aws_pg.jobs.realprompt_dryrun --out <dir>

1. The real plan without a model: row and call counts, controls re-drawn, regexes compile, the code rule on written
   examples.
2. The job on a tiny random model of each architecture (neuron_off_dryrun.tiny_model; nothing downloaded) with a
   small plan of the same shape (poles in layers 0-1, a code pole with its drawn control, 3 samples, 12 new tokens,
   8 rows per call so prompts span several calls); then the abort path: a failing input that cannot fire
   (FAIL_ALPHA 0) must stop the job with status aborted_gate; the scorer reads the job's own output files.
3. The scorer end to end on a planted G4 bank in the real plan's shape: React useEffect planted in 12 of 20 push
   samples at dose 2 and in none of the control's must read SUPPORTED; the other two code neurons (0 vs 0) REFUTED;
   a control planted at 18 of 20 NOT SCORABLE (no room); push samples that are not code NOT SCORABLE; a failed
   realized-change cell NOT SCORABLE through the model gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib

import torch

from infra.aws_pg.jobs import realprompt as J
from infra.aws_pg.jobs.neuron_off_dryrun import BOS, tiny_model, tiny_tokenizer
from infra.pg import realprompt_plan as P
from infra.pg import realprompt_score as S

ARCH = {"Q4B": "qwen3", "G4": "gemma3", "O7": "olmo3"}
TINY_POLES = {"Q4B": [("L01N5+", "code", "arm1", None), ("L00N7-", "essay", "best", r"(?i)\bthe\b"),
                      ("L01N9+", "news", "page", r"(?i)\ba\b")],
              "G4": [("L01N11-", "code", "arm1", None), ("L00N3+", "email", "best", r"(?i)\bimport")],
              "O7": [("L01N4+", "news", "page", r"(?i)\bselect"), ("L00N2-", "story", "best", r"(?i)\bcat")]}


def check_real_plan():
    n = {}
    for m in P.MODELS:
        calls = J.build_calls(m)
        assert all(len(r) <= P.MAX_ROWS for *_, r in calls)
        n[m] = sum(len(r) for *_, r in calls)
        doms = {d["domain"] for d in P.directions(m)}
        assert {dom for _, dom, *_ in calls} == doms
        for i, dom, pr, c, rows in calls:
            assert pr in P.PROMPTS[dom] and i == P.prompt_index(dom, P.PROMPTS[dom].index(pr))
    assert n == {"Q4B": 1780, "G4": 2020, "O7": 280}, n
    for pole, c in P.CODE.items():
        assert P.draw_control(c["model"], pole) == c["ctrl"], pole
    py = "python\nimport numpy as np\nimport pandas as pd\n```\nThat is all."
    sql = "sql\nCREATE TABLE t (id INT, PRIMARY KEY (id), FOREIGN KEY (u) REFERENCES u(id));"
    prose = "Dear John, thank you for your note (sent on Monday). I will reply soon; best wishes."
    assert P.is_code(py, py) and P.is_code(sql, sql) and not P.is_code(prose, prose)
    return n


def tiny_plan(m: str):
    P.POLES = {m: TINY_POLES[m]}
    P.D_MLP = {"Q4B": 128, "G4": 128}
    P.CODE = {}
    for pole, dom, *_ in TINY_POLES[m]:
        if dom == "code":
            P.CODE[pole] = {"model": m, "label": "planted", "regex": r"\bimport\b", "ctrl": None}
            P.CODE[pole]["ctrl"] = P.draw_control(m, pole)
    P.N_SAMPLES = 3


def tiny_run(m: str, out: pathlib.Path) -> dict:
    from infra.pg import neuron_rw_core as R
    model, tok = tiny_model(ARCH[m]), tiny_tokenizer(BOS[ARCH[m]])
    ids = torch.tensor([R.prompt_ids(tok, "import numpy as np\nThe cat sat on the mat. " * 4)])
    rms = [x["rms"] for x in R.residual_rms(model, [ids], 1)["layers"]]
    met = J.run(model, tok, m, rms, out, {"model_key": m, "dry_run": True},
                max_rows=8, sampling={**P.SAMPLING, "max_new_tokens": 12})
    disk = json.loads((out / "metrics.json").read_text())
    assert met["status"] == disk["status"] == "complete" and disk["progress"]["calls_done"] == disk["progress"]["calls_planned"]
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    n_dirs = len(P.directions(m))
    doms = {d["domain"] for d in P.directions(m)}
    assert len(rows) == disk["progress"]["rows_written"] == \
        2 * len(doms) * P.N_SAMPLES + 2 * n_dirs * (len(P.DOSES[m]) - 1) * P.N_SAMPLES
    assert all(isinstance(r["completion"], str) and "head60" in r for r in rows)
    st = [r for r in rows if r["dose"] > 0]
    assert all(r["cos_step1"] is not None and r["cos_step1"] > 0.99 for r in st), min(r["cos_step1"] or 0 for r in st)
    assert all(r["realized_cos"] is None for r in rows if r["dose"] == 0)
    g = disk["gates"]
    assert g["identity"]["pass"] and g["identity"]["max_abs_diff"] == 0.0 and g["identity"]["failing_input"]["fired"]
    assert g["realized_failing_input"]["fired"] and max(g["realized_failing_input"]["cos_step1"]) < P.COS_MIN
    # abort path: a failing input that cannot fire
    J.FAIL_ALPHA = 0.0
    try:
        J.run(model, tok, m, rms, out.parent / f"{m}_abort", {"model_key": m}, max_rows=8,
              sampling={**P.SAMPLING, "max_new_tokens": 12})
        raise RuntimeError("the job ran with a failing input that cannot fire")
    except AssertionError as e:
        assert "gate failed" in str(e), e
    finally:
        J.FAIL_ALPHA = 1.0
    assert json.loads((out.parent / f"{m}_abort" / "metrics.json").read_text())["status"] == "aborted_gate"
    sc = S.score(S.load(out.parent))                     # the scorer on the job's own output files
    (out.parent / "REPORT.md").write_text(S.render(sc, None))
    return {"rows": len(rows), "identity": g["identity"]["max_abs_diff"],
            "verdicts": {p["pole"]: p["verdict"] for p in sc[m]["predictions"]},
            "realized_failing_cos": max(g["realized_failing_input"]["cos_step1"]),
            "min_step1_cos": min(r["cos_step1"] for r in st)}


CODE_TXT = "jsx\nconst x = f(a); let y = [1, 2]; if (x) { y = g(x); }\n```\n"
PROSE_TXT = "It was a quiet morning and the town slept while the river ran on past the old mill."


def planted_bank(out: pathlib.Path, edit=None) -> pathlib.Path:
    """A G4 bank in the real plan's shape with planted completions; edit(row) may change rows (a planted failure)."""
    m = "G4"
    d = out / m
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, dom, pr, c, spec in J.build_calls(m):
        for dd, dose, k in spec:
            txt = CODE_TXT if dom == "code" else PROSE_TXT + f" {dose} {k} " + ("word " * (k + 1))
            if dd is not None and dd["pole"] == "L07N10202-" and dose == 2 and k < 6:
                txt = "jsx\nuseEffect(() => { load(); }, []);\n```\n"
            r = {"model_key": m, "pole": None if dd is None else dd["pole"], "role": "unsteered" if dd is None else dd["role"],
                 "target": None if dd is None else dd["target"], "domain": dom, "prompt_index": i, "prompt": pr,
                 "dose": dose, "alpha_abs": float(dose), "k": k, "seed": P.SEED + 1000 * i + c, "call": c,
                 "completion": txt, "head60": txt, "head64": txt, "completion_token_ids": [], "n_new_tokens": 0,
                 "stopped_eos": False, "realized_cos": None if dd is None else 0.999,
                 "cos_step1": None if dd is None else 0.999, "ratio_step1": None if dd is None else 1.0}
            rows.append(edit(r) if edit else r)
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gates = {"identity": {"pass": True, "max_abs_diff": 0.0, "failing_input": {"max_abs_diff": 0.5, "fired": True}},
             "realized_failing_input": {"cos_step1": [0.0] * 4, "fired": True}}
    (d / "metrics.json").write_text(json.dumps({"status": "complete", "gates": gates, "revision": "planted",
                                                "progress": {"rows_written": len(rows)},
                                                "throughput": {"gen_s": 1.0}, "load_s": 1.0}))
    (d / "plan.json").write_text("{}")
    return out


def verdicts(bank: pathlib.Path) -> dict:
    s = S.score(S.load(bank))
    S.render(s, None)
    return {p["pole"]: p["verdict"] for p in s["G4"]["predictions"]}, s


def check_scorer(out: pathlib.Path) -> dict:
    v, s = verdicts(planted_bank(out / "plant_base"))
    assert v == {"L07N10202-": "SUPPORTED", "L06N599+": "REFUTED", "L08N7823+": "REFUTED"}, v
    react = next(p for p in s["G4"]["predictions"] if p["pole"] == "L07N10202-")
    assert react["d_star"] == 2 and abs(react["gap"] - 0.6) < 1e-9, react

    def room(r):
        if r["pole"] == "L07N2158-" and r["dose"] == 2 and r["k"] < 9:
            r = {**r, "completion": "jsx\nuseEffect(() => {}, []);\n```\n"}
        return r

    def prose_push(r):
        if r["pole"] == "L06N599+":
            r = {**r, "completion": PROSE_TXT, "head64": PROSE_TXT}
        return r

    def bad_cos(r):
        if r["pole"] == "L15N9089+" and r["dose"] == 16:
            r = {**r, "cos_step1": 0.5}
        return r
    v_room, _ = verdicts(planted_bank(out / "plant_room", room))
    v_code, _ = verdicts(planted_bank(out / "plant_code", prose_push))
    v_cos, _ = verdicts(planted_bank(out / "plant_cos", bad_cos))
    assert v_room["L07N10202-"] == "NOT SCORABLE", v_room
    assert v_code["L06N599+"] == "NOT SCORABLE" and v_code["L07N10202-"] == "SUPPORTED", v_code
    assert set(v_cos.values()) == {"NOT SCORABLE"}, v_cos
    return {"base": v, "room": v_room["L07N10202-"], "not_code": v_code["L06N599+"], "bad_cos": sorted(set(v_cos.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"real_plan_rows": check_real_plan(), "scorer": check_scorer(out)}
    saved = copy.deepcopy((P.POLES, P.CODE, P.D_MLP, P.N_SAMPLES))
    for m in ARCH:
        tiny_plan(m)
        rep[m] = tiny_run(m, out / f"tiny_{m}" / m)
    P.POLES, P.CODE, P.D_MLP, P.N_SAMPLES = saved
    print("REALPROMPT_DRYRUN_OK", json.dumps(rep))


if __name__ == "__main__":
    main()
