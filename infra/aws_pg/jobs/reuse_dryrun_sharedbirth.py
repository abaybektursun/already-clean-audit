"""Weight-free dry run of SPEC11 SHAREDBIRTH (studies/reuse/SPEC11_sharedbirth.md §9). No model and no download: tiny tensors only.

    python -m infra.aws_pg.jobs.reuse_dryrun_sharedbirth --out <dir>

Part 1, the real inputs (read only): SPEC10's banked case list (results/sharedinputs/1/t15/cases.json) equals jobs.reuse_sharedinputs.case_list on
studies/reuse/inputs/birth_items.json with SPEC4's live flags at birth; jobs.reuse_sharedbirth.birth_list on it must give 47 primary and 36 fresh
cases, 101 K reads, births 3, 4, 4, 10, 8, 24, 30 over T_7..T_13, 34 K and 24 cases dropped, and must equal the verdict's re-implementation
(verdict_sharedbirth.expected); the K-validity gate passes and its failing input gives a problem in each of the 30 cases that had a K dropped; at each
of the seven loads the job's validity and check_refs pass on SPEC4's local birth bank (results/birth/1/t<nn>: metrics.json and falls/). The K that the
brief's wording keeps and §2 drops is printed. Each load's captures, ablated forwards and tokens are counted with reuse_ablate.n_forwarded and timed at
SPEC10's measured rate and fixed costs (results/sharedinputs/1/t15/metrics.json).
Part 2, a synthetic seven-load bank written by the job's own code: jobs.reuse_sharedbirth.Job.run at T_7..T_13 (validity, check_refs, case, SPEC10's
case/read_arm/sample, known_spec4, the exclusion and identity-arm records) and jobs.reuse_ablate.Job.evaluate (the parent's banking, unchanged) on tiny
torch tensors, with the model replaced at two points only, as SPEC10's dry run does (its Plan and DryJob methods are reused): Job.side returns planted
captures and reuse_ablate.readout returns a_P = a_clean (1 - planted fall). Items: 14 primary cases (10 listed) and 11 fresh (8 listed) on poles of
layers 2 to 6 with births 7 to 13: K born later (dropped and banked), K with no own case (only an O*, kept), K born at the same checkpoint (kept), a
case left with no K on each side, a pole whose other case is cancelled at birth, a load with fresh cases only (T_7: no known number to read), one K
cluster with own fall 0.15 (the §8 restriction drops it) and one fresh case cancelled at its load. Planted: own' 0.35 to 0.55; primary I_O 0.30 to 0.32
and I_K 0.08 to 0.11 (Y1 reads H_shared-at-birth refuted); fresh I_O 0.20 to 0.22 and I_K 0.27 to 0.30 (Y1f reads H_old-at-birth refuted); random' 0.02,
0.30 for the cancelled case. SPEC4's birth-load falls for the listed primary cases differ from the planted ones by 0.004; SPEC4's per-case I by 0.01.
Part 3, the verdict (studies/reuse/scripts/verdict_sharedbirth.py, with --spec on the real SPEC11): exit 4 on the bank without one K's Kown npz (nothing
written); exit 0 with the planted answers (Y1, Y1f, I_O, I_K, K's own fall, the share and the restricted reading equal to the plan's values over the live
cases; the known number 0.004; the exclusion's failing input 3 members per ranked set; K validity 0 with its failing input in 7 cases; the births by
load and the dropped K as planted); exit 1 under each failing input with its target failed, the other gates passing and both rules "not read".
Prints REUSE_DRYRUN_SHAREDBIRTH_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_dryrun_sharedinputs as DX
from infra.aws_pg.jobs import reuse_sharedbirth as JSB
from infra.aws_pg.jobs import reuse_sharedinputs as JX
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
VERDICT = REPO / "studies/reuse/scripts/verdict_sharedbirth.py"
SPEC10_METRICS = REPO / "studies/reuse/results/sharedinputs/1/t15/metrics.json"
SPEC10_CASES = REPO / "studies/reuse/results/sharedinputs/1/t15/cases.json"
BIRTH_RUN = REPO / "studies/reuse/results/birth/1"
N_UP, W, M = DX.N_UP, DX.W, JX.M
LOADS = [int(t.split("_")[1]) for t in JSB.LOADS]
REAL = {"n": {"primary": 47, "fresh": 36}, "K": 101, "births": {7: 3, 8: 4, 9: 4, 10: 10, 11: 8, 12: 24, 13: 30}, "K_dropped": 34, "cases_dropped": 24, "with_dropped": 30}
MODES = {"known_next_case": ["known_spec4"], "capture_wrong_layer": [f"capture_identity t{t:02d}" for t in LOADS],
         "positive_random": ["positive_control primary", "positive_control fresh"], "cancel_swapped": ["cancel_share primary", "cancel_share fresh"],
         "k_born_later": ["k_validity"]}
PRIMARY = [("L03N10+", 1, 0, True, 8), ("L03N10+", 2, 0, True, 10), ("L03N10+", 3, 4, True, 12), ("L04N20-", 1, 0, True, 9), ("L04N20-", 2, 0, True, 11),
           ("L05N30+", 1, 2, True, 13), ("L05N30+", 5, 3, True, 13), ("L02N40+", 1, 0, True, 9), ("L02N40+", 2, 5, True, 9), ("L06N50+", 1, 0, True, 12),
           ("L06N50+", 2, 1, True, 12), ("L04N60+", 1, 0, True, 11), ("L04N60+", 2, 0, False, 11), ("L03N10-", 2, 1, True, 10)]   # (pole, c, o_star, live at birth, birth)
FRESH = [("L03N110+", 1, 0, True, 7), ("L03N110+", 2, 0, True, 7), ("L03N110+", 3, 0, True, 13), ("L05N120-", 1, 0, True, 8), ("L05N120-", 2, 3, True, 12),
         ("L04N130+", 2, 0, True, 9), ("L04N130+", 4, 1, False, 10), ("L06N140+", 1, 2, True, 10), ("L06N140+", 3, 2, True, 11), ("L02N150+", 1, 0, True, 10),
         ("L02N150+", 2, 0, True, 10)]
EXPECTED = {"L03N10+:c1": ([4], [2, 3]), "L03N10+:c2": ([1, 4], [3]), "L03N10+:c3": ([0, 1, 2], []), "L04N20-:c1": ([], [2]), "L04N20-:c2": ([1], []),
            "L05N30+:c1": ([3, 5], []), "L05N30+:c5": ([1, 2], []), "L02N40+:c1": ([2, 5], []), "L02N40+:c2": ([0, 1], []), "L06N50+:c1": ([2], []),
            "L06N50+:c2": ([0], []), "L03N110+:c1": ([2], [3]), "L03N110+:c2": ([1], [3]), "L03N110+:c3": ([1, 2], []), "L05N120-:c1": ([3], [2]),
            "L05N120-:c2": ([0, 1], []), "L06N140+:c1": ([], [3]), "L06N140+:c3": ([1], []), "L02N150+:c1": ([2], []), "L02N150+:c2": ([1], [])}
# (kept K, K born later) per SPEC10 case, worked by hand from §2's words: e.g. L03N10+:c1 born at 8 keeps c4 (no own case: only the O* of c3) and drops
# c2 (born 10) and c3 (born 12); L06N50+:c1 keeps c2, its child born at the same checkpoint; L06N50+:c2 keeps c0, its O*'s O*.
LOW_K = ("L02N40+", 5)            # an O*-only cluster with own fall 0.15: the §8 restriction drops it
CANCEL = ("L06N140+", 3)          # random' 0.30: cancelled at its load


def load_verdict_module():
    spec = importlib.util.spec_from_file_location("verdict_sharedbirth", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Plan(DX.Plan):
    """SPEC10's dry-run plan (planted D' per cluster, the scores that rank them, the readout) with this run's own falls, I_O, I_K and cancelled case."""

    def own(self, p: str, rc: int) -> float:
        return 0.15 if (p, rc) == LOW_K else super().own(p, rc)

    def ratio(self, p: str, rc: int, sc: int) -> float:
        if rc == sc:
            return 1.0
        if self.ostar.get((p, rc)) == sc:
            return 0.9 if (p, rc) == CANCEL else (0.30 if self.prim[p] else 0.20) + 0.01 * (rc % 3)
        return (0.08 if self.prim[p] else 0.27) + 0.01 * (sc % 4)

    def random_fall(self, p: str, rc: int) -> float:
        return 0.30 if (p, rc) == CANCEL else 0.02


class DryJob(JSB.Job):
    """jobs.reuse_sharedbirth.Job with no model: setup() as the job; SPEC10's dry-run planted captures and synthetic known-number, capture-identity and
    background records."""
    known = DX.DryJob.known
    identity = DX.DryJob.identity
    background = DX.DryJob.background
    side = DX.DryJob.side
    prompts = DX.DryJob.prompts

    def __init__(self, out, a, items: dict, live: dict, refs: dict, plan=None, abar=None):
        self.plan, self.abar_t, self.model, self.tokens, self.bos = plan, abar, None, None, 1
        self.setup(out, a, items, "cpu", W, N_UP + 1, "torch.float32", live, refs)


def args_ns(t: int, isha: str, bsha: str, csha: str):
    return types.SimpleNamespace(stage="sharedbirth", t=f"T_{t}", revision=JR.REVISION[f"T_{t}"], items_sha256=isha, birth_verdict_sha256=bsha,
                                 cases_t15_sha256=csha, harvest_window=UR.HARVEST_WINDOW, batch=32, full_batch=8, scan_tokens=20_570_743)


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ part 1: the real inputs
def real_inputs(out: pathlib.Path) -> dict:
    ip, bp = REPO / "studies/reuse/inputs/birth_items.json", REPO / "studies/reuse/results/birth/verdict_birth.json"
    items, bv = json.loads(ip.read_text()), json.loads(bp.read_text())
    live = JX.live_flags(bv)
    s10 = json.loads(SPEC10_CASES.read_text())
    assert s10["cases"] == JX.case_list(items, live), "SPEC10's banked case list differs from case_list on the real items"
    allc = JSB.birth_list(s10["cases"], items)
    listed = [c for c in allc if c["K"]]
    n = {s: sum(c["set"] == s for c in listed) for s in ("primary", "fresh")}
    births = {t: sum(c["t_b"] == t for c in listed) for t in sorted({c["t_b"] for c in listed})}
    got = {"n": n, "K": sum(len(c["K"]) for c in listed), "births": births, "K_dropped": sum(len(c["K_dropped"]) for c in allc),
           "cases_dropped": sum(not c["K"] for c in allc), "with_dropped": sum(bool(c["K_dropped"]) for c in allc)}
    assert got == REAL, (got, REAL)
    VSB = load_verdict_module()
    assert [(c["job"], c["t_b"], [k["c"] for k in c["K"]], [k["c"] for k in c["K_dropped"]]) for c in allc] == \
        [(e["job"], e["t_b"], e["K"], e["K_dropped"]) for e in VSB.expected(items, live)], "the job's list differs from the verdict's"
    kv = JSB.kb_validity(allc, items, live)
    assert kv["value"] == 0 and kv["failing_input"]["value"] == REAL["with_dropped"] and kv["failing_input_failed"], kv
    brief = [(c["job"], c["t_b"], k["c"], k["birth"]) for c in allc for k in c["K_dropped"] if any(s == "O" for _, s in k["carriers"])]
    assert len(brief) == 1 and not next(c for c in allc if c["job"] == brief[0][0])["K"] and brief[0][0] in {x["job"] for x in items["primary"]}, brief
    by = {x["job"]: x for x in items["primary"] + items["fresh"]}
    bos, hw = 1, UR.HARVEST_WINDOW
    nf = lambda t: RA.n_forwarded(np.asarray(t, np.int64), RA.LENGTH, bos, hw)
    kt = items["known"]["tok"]
    idc = next(x for x in items["primary"] if x["layer"] >= 1 and len(JX.capped(x["J_tok"])) + len(JX.capped(x["O_tok"])) >= 20)
    fixed_tok0 = sum(RA.n_forwarded(np.asarray(x)[:JR.KNOWN_CTX], None, bos, hw) for x in kt) + \
        2 * nf(np.concatenate([JX.capped(idc["J_tok"]), JX.capped(idc["O_tok"])])[:20])
    m10 = json.loads(SPEC10_METRICS.read_text())
    rate = m10["units_tokens_per_second"]
    fixed_s = m10["download_and_corpus_s"] + m10["load_s"] + sum(m10["timing_s"][k] for k in ("known", "identity", "background"))
    loads, refs = {}, {}
    for t in LOADS:
        d = out / "real_check" / f"t{t:02d}"
        job = DryJob(d, args_ns(t, sha(ip), sha(bp), sha(SPEC10_CASES)), items, live, {"cases_t15": SPEC10_CASES, "birth_bank": BIRTH_RUN / f"t{t:02d}"})
        job.validity()
        job.check_refs()
        refs[t] = job.m["refs_checked"]["n_cases"]
        e = {"cases": len(job.cases), "primary": job.m["n_cases"]["primary"], "fresh": job.m["n_cases"]["fresh"], "K_reads": sum(len(c["K"]) for c in job.cases),
             "captures": 0, "capture_tokens": 0, "forwards": 0, "forward_tokens": 0}
        for c in job.cases:
            j = by[c["job"]]
            Jt, Ot, Ks = JX.capped(j["J_tok"]), JX.capped(j["O_tok"]), [k["tok"] for k in c["K"]]
            e["captures"] += len(Jt) + len(Ot) + sum(map(len, Ks))
            e["capture_tokens"] += nf(Jt) + nf(Ot) + sum(map(nf, Ks))
            nja = len(JX.j_arms([k["c"] for k in c["K"]]))
            e["forwards"] += 2 * nja * len(Jt) + sum(4 * len(k) for k in Ks)            # each arm: a clean readout and the ablated one
            e["forward_tokens"] += 2 * nja * nf(Jt) + sum(4 * nf(k) for k in Ks)
        e["fixed_tokens"] = fixed_tok0 + nf(UR.background_positions(t, 20_570_743 // hw * hw, hw)["e1"])
        e["total_tokens"] = e["capture_tokens"] + e["forward_tokens"] + e["fixed_tokens"]
        e["units_s"] = (e["capture_tokens"] + e["forward_tokens"]) / rate
        loads[t] = e
    shutil.rmtree(out / "real_check")
    tot = {k: sum(e[k] for e in loads.values()) for k in ("cases", "K_reads", "captures", "capture_tokens", "forwards", "forward_tokens", "fixed_tokens", "total_tokens", "units_s")}
    tot["job_wall_h"] = (tot["units_s"] + len(LOADS) * fixed_s) / 3600
    return {**got, "loads": loads, "total": tot, "refs": refs, "brief": brief, "rate": rate, "fixed_s": fixed_s}


# ------------------------------------------------------------------------------------------ part 2: the synthetic bank
def build(out: pathlib.Path) -> dict:
    rng = np.random.default_rng(0)
    abar = torch.tensor(rng.normal(0, 1, (N_UP, W)), dtype=torch.float32)
    mask = RB.high_background(abar, N_UP)["mask"].numpy()
    tok = iter(range(10, 10 ** 7, 200))
    ctx = {}

    def cases(spec, label):
        out_ = []
        for pole, c, o, _, b in spec:
            for cl in (c, o):
                if (pole, cl) not in ctx:
                    ctx[(pole, cl)] = [next(tok) for _ in range(6 + (len(ctx) * 5) % 7)]
            out_.append({"job": f"{pole}:c{c}", "pole": pole, "c": c, "o_star": o, "layer": int(pole[1:3]), "neuron": int(pole[4:-1]), "sign": pole[-1], "tercile": 0,
                         "birth": b, "t_before": b - 1, "label": label, "J_tok": ctx[(pole, c)], "O_tok": ctx[(pole, o)], "ceiling": None})
        return out_
    items = {"primary": cases(PRIMARY, "related"), "fresh": cases(FRESH, "unrelated"), "planted": [],
             "known": {"poles": [], "tok": [], "a_bank": {}, "a_wrong": {}, "wrong_rule": "synthetic"}, "sources": {"grid": []}}
    ip = out / "birth_items.json"
    DX.dump(ip, items)
    live = {f"{p}:c{c}": lv for p, c, _, lv, _ in PRIMARY + FRESH}
    plan = Plan(items, mask, rng)
    RA.readout = plan.readout          # the model's readout, replaced in this process only
    by = {x["job"]: x for x in items["primary"] + items["fresh"]}
    bvp = out / "verdict_birth.json"
    I4 = lambda x: plan.ratio(x["pole"], x["c"], x["o_star"]) + 0.01
    DX.dump(bvp, {"cases": {k: [{"job": x["job"], "cancelled": not live[x["job"]], "I": I4(x)} for x in items[s]] for k, s in (("primary_birth", "primary"), ("fresh_birth", "fresh"))},
                  "rules": {"R1c": {"I": {"value": 0.207}}}})
    cp = out / "cases_t15.json"
    s10 = JX.case_list(items, live)
    assert {c["job"]: [k["c"] for k in c["K"]] for c in s10} == {j: sorted(k + dr) for j, (k, dr) in EXPECTED.items()}, [(c["job"], [k["c"] for k in c["K"]]) for c in s10]
    DX.dump(cp, {"rule": JX.case_list.__doc__, "live": "synthetic", "birth_verdict_sha256": sha(bvp), "n": {s: sum(c["set"] == s for c in s10) for s in ("primary", "fresh")},
                 "cases": s10})
    # SPEC4's birth-load bank per load: every primary case born there, own' and cross' moved by +-0.004 from the plan
    brun = out / "birth_run"
    listedP = [j for j, (k, _) in EXPECTED.items() if k and j in {x["job"] for x in items["primary"]}]
    arms6 = [f"{x}_direct_m{M}" if x != "identity" else x for x in ("identity", "own", "cross", "random", "foreign", "rj")]
    for t in LOADS:
        bornP = [x for x in items["primary"] if x["birth"] == t]
        DX.dump(brun / f"t{t:02d}" / "metrics.json", {"stage": "birth", "arms": "birth_load", "t": f"T_{t}", "status": "done", "items_sha256": sha(ip),
                                                      "jobs": {x["job"]: {} for x in bornP}})
        (brun / f"t{t:02d}" / "falls").mkdir(parents=True, exist_ok=True)
        for x in bornP:
            p, c, o = x["pole"], x["c"], x["o_star"]
            dlt = 0.004 * (1 if listedP.index(x["job"]) % 2 == 0 else -1) if x["job"] in listedP else 0.0
            base = plan.base[(p, c)]
            fr = {"own": plan.own(p, c) + dlt, "cross": plan.own(p, c) * plan.ratio(p, c, o) - dlt}
            aft = np.stack([(base * np.float32(1 - fr.get(a.split("_")[0], 0.0))).astype(np.float32) for a in arms6])
            np.savez_compressed(brun / f"t{t:02d}" / "falls" / f"{JX.key_of(x)}.npz", tok=np.asarray(JX.capped(x["J_tok"]), np.int64), arm=np.array(arms6),
                                a_before=base, a_after=aft)
    run = out / "run"
    for t in LOADS:
        job = DryJob(run / f"t{t:02d}", args_ns(t, sha(ip), sha(bvp), sha(cp)), items, live, {"cases_t15": cp, "birth_bank": brun / f"t{t:02d}"}, plan, abar)
        want = [j for j, (k, _) in EXPECTED.items() if k and by[j]["birth"] == t]
        assert [(c["job"], [k["c"] for k in c["K"]], [k["c"] for k in c["K_dropped"]]) for c in job.cases] == [(j, *EXPECTED[j]) for j in want], \
            (t, [(c["job"], [k["c"] for k in c["K"]]) for c in job.cases])
        job.run()
        job.m.update({"download_and_corpus_s": 60.0, "load_s": 30.0, "device_name": "none (dry run)"})
        job.m["timing_s"]["wall_after_load"] = 300.0
        job.write()
    # the plan's answers, worked from the plan and EXPECTED, not from the bank
    rows = []
    for jb, (Ks, _) in EXPECTED.items():
        if not Ks:
            continue
        x = by[jb]
        p, c, o = x["pole"], x["c"], x["o_star"]
        IO, IK = plan.ratio(p, c, o), float(np.mean([plan.ratio(p, c, k) for k in Ks]))
        Kr = [k for k in Ks if plan.own(p, k) >= 0.25]
        rows.append({"job": jb, "set": "primary" if jb in {y["job"] for y in items["primary"]} else "fresh", "t_b": x["birth"], "I_O": IO, "I_K": IK, "d": IO - IK,
                     "d_strong": IO - float(np.mean([plan.ratio(p, c, k) for k in Kr])), "Kown": float(np.mean([plan.own(p, k) for k in Ks])), "live": (p, c) != CANCEL})
    n_ranked = sum(2 + 3 * len(Ks) for Ks, _ in EXPECTED.values() if Ks)
    return {"items": ip, "bv": bvp, "cases_t15": cp, "birth_run": brun, "run": run, "rows": rows, "n_ranked": n_ranked, "by": by, "listedP": listedP,
            "n_with_dropped": sum(bool(dr) for _, dr in EXPECTED.values()), "n_K_dropped": sum(len(dr) for _, dr in EXPECTED.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    e = real_inputs(out)
    print(f"real inputs: {e['n']['primary']} primary and {e['n']['fresh']} fresh cases with a K existing at the birth, {e['K']} K reads; births "
          f"{', '.join(str(e['births'][t]) for t in LOADS)} over T_7..T_13; {e['K_dropped']} K born later dropped, {e['cases_dropped']} SPEC10 cases left with no K; "
          f"equal to the verdict's list; K validity 0, its failing input in {e['with_dropped']} cases; SPEC4's birth falls checked per load "
          f"{', '.join(f'T_{t} {n}' for t, n in e['refs'].items())}")
    for job, tb, k, b in e["brief"]:
        print(f"brief's wording keeps, section 2 drops: {job} (t_b T_{tb}) K c{k}, born T_{b} and also an O* of a carrier; with it the count is "
              f"{e['n']['primary'] + 1} primary, {e['K'] + 1} K reads, {e['births'][tb] + 1} births at T_{tb}")
    for t, x in e["loads"].items():
        print(f"load T_{t}: {x['primary']} primary + {x['fresh']} fresh cases, {x['K_reads']} K reads; captures {x['captures']} ({x['capture_tokens']} tokens), "
              f"ablated-arm forwards {x['forwards']} ({x['forward_tokens']} tokens), fixed blocks {x['fixed_tokens']} tokens; units about {x['units_s']:.0f} s")
    tt = e["total"]
    print(f"seven loads: {tt['cases']} cases, {tt['K_reads']} K reads; captures {tt['captures']}, forwards {tt['forwards']}; total {tt['total_tokens']} tokens; units about "
          f"{tt['units_s']:.0f} s at SPEC10's {e['rate']:.0f} tokens/s, plus {e['fixed_s']:.0f} s fixed per load (download, load, known number, identity, background); "
          f"job wall about {tt['job_wall_h']:.2f} h")

    b = build(out)
    vout = out / "verdict"
    cmd = [sys.executable, str(VERDICT), "--items", str(b["items"]), "--birth-verdict", str(b["bv"]), "--cases-t15", str(b["cases_t15"]),
           "--birth-run", str(b["birth_run"]), "--out", str(vout)]
    exits = {}
    nl = sum(1 for _, (k, _) in EXPECTED.items() if k)
    print(f"synthetic bank: {nl} listed cases over seven loads written by jobs.reuse_sharedbirth.Job.run with planted captures and readouts; {b['n_ranked']} ranked sets")

    # an incomplete bank stops the verdict
    inc = out / "run_incomplete"
    shutil.copytree(b["run"], inc)
    gone = f"{JX.key_of(b['by']['L05N120-:c2'])}__Kown_c0.npz"
    (inc / "t12" / "falls" / gone).unlink()
    r = subprocess.run(cmd + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "SHAREDBIRTH_BANK_INCOMPLETE" in r.stderr and gone in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits["incomplete"] = r.returncode
    print("verdict on a bank without t12/falls/" + gone, ":", r.stderr.splitlines()[1].strip(), "exit", r.returncode)
    shutil.rmtree(inc)

    # green, with the planted answers
    r = subprocess.run(cmd + ["--run", str(b["run"])], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_SHAREDBIRTH_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_sharedbirth.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    live = {s: [x for x in b["rows"] if x["set"] == s and x["live"]] for s in ("primary", "fresh")}
    assert (v["count"]["primary"], v["count"]["fresh"], v["count"]["primary_live"], v["count"]["fresh_live"]) == (10, 8, 10, 7), v["count"]
    for name, s in (("Y1", "primary"), ("Y1f", "fresh")):
        want = float(np.median([x["d"] for x in live[s]]))
        assert abs(v["rules"][name]["value"] - want) < 1e-5, (name, v["rules"][name]["value"], want)
        for k, f in (("I_O", "I_O"), ("I_K", "I_K"), ("K_own_fall", "Kown"), ("strong_K_reading", "d_strong")):
            assert abs(v["reported"][s][k]["value"] - float(np.median([x[f] for x in live[s]]))) < 1e-5, (s, k, v["reported"][s][k]["value"])
        assert v["reported"][s]["share_I_O_gt_I_K"]["value"] == float(np.mean([x["I_O"] > x["I_K"] for x in live[s]])), (s, v["reported"][s]["share_I_O_gt_I_K"])
        assert v["reported"][s]["n_cancelled"] == (1 if s == "fresh" else 0), (s, v["reported"][s]["n_cancelled"])
        assert abs(v["reported"][s]["spec4"]["abs_diff_median"] - 0.01) < 1e-5, (s, v["reported"][s]["spec4"])
    vp = v["reported"]["primary"]
    assert vp["n_K_under_0.25"] == 1 and vp["strong_K_reading"]["n_K"] == vp["n_K_reads"] - 1 and v["reported"]["fresh"]["n_K_under_0.25"] == 0, \
        (vp["n_K_under_0.25"], vp["strong_K_reading"]["n_K"], vp["n_K_reads"])
    assert v["rules"]["Y1"]["verdict"].startswith("H_shared-at-birth refuted") and v["rules"]["Y1f"]["verdict"].startswith("H_old-at-birth refuted"), \
        (v["rules"]["Y1"]["verdict"], v["rules"]["Y1f"]["verdict"])
    kn = v["gates"]["known_spec4"]
    assert abs(kn["value"] - 0.004) < 1e-5 and kn["n_cases"] == len(b["listedP"]) and kn["per_load"]["t07"] is None, kn
    assert v["gates"]["exclusion"]["failing_input"]["value"] == 3 * b["n_ranked"] and v["gates"]["exclusion"]["value"]["n_in_B"] == 0, v["gates"]["exclusion"]
    kg = v["gates"]["k_validity"]
    assert kg["value"] == 0 and kg["failing_input"]["value"] == b["n_with_dropped"] == 7 and kg["n_K_dropped"] == b["n_K_dropped"], kg
    bl = {t: (x["primary"], x["fresh"]) for t, x in v["reported"]["births_by_load"].items()}
    assert bl == {"T_7": (0, 2), "T_8": (1, 1), "T_9": (2, 0), "T_10": (1, 2), "T_11": (1, 1), "T_12": (3, 1), "T_13": (2, 1)}, bl
    assert sorted(x["job"] for x in v["reported"]["dropped"]["cases"]) == ["L04N20-:c1", "L06N140+:c1"], v["reported"]["dropped"]["cases"]
    assert v["reported"]["a_before_max_abs_diff_same_contexts"] == 0.0, v["reported"]["a_before_max_abs_diff_same_contexts"]
    assert (vout / "GRID11.md").read_text().startswith("Cases 10 primary"), "GRID11.md does not open with the count line"
    samp = json.loads((vout / "samples" / "sharedbirth_samples.json").read_text())
    assert len(samp["job_samples"]) == 17 and len(samp["extremes"]) == 4 and all(x["contexts"]["own"] and x["contexts"]["crossO"] for x in samp["extremes"].values()), \
        (len(samp["job_samples"]), list(samp["extremes"]))
    print(f"planted answers: Y1 {v['rules']['Y1']['value']:.4f} and Y1f {v['rules']['Y1f']['value']:.4f} equal the plan's medians over the live cases (the fresh case "
          f"cancelled at its load left out); I_O, I_K, K's own fall, the share and the reading on K with own fall >= 0.25 "
          f"({v['reported']['primary']['strong_K_reading']['value']:.4f} primary) equal; known number {kn['value']:.4f} over {kn['n_cases']} primary cases (T_7 has none); "
          f"exclusion failing input {v['gates']['exclusion']['failing_input']['value']} members over {b['n_ranked']} ranked sets; K validity failing input "
          f"{kg['failing_input']['value']} cases")

    # each failing input
    fails = {}
    for mode, target in MODES.items():
        r = subprocess.run(cmd + ["--run", str(b["run"]), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_sharedbirth.json").read_text())
        assert all(not f["gates"][t]["ok"] for t in target) and all(x["verdict"] == "not read" for x in f["rules"].values()), (mode, {t: f["gates"][t] for t in target})
        assert all(g["ok"] for k, g in f["gates"].items() if k not in target), (mode, {k: g["ok"] for k, g in f["gates"].items()})
        fails[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_sharedbirth exit codes:", exits)
    (out / "metrics.json").write_text(json.dumps({"exits": exits, "failing_inputs": fails, "Y1": v["rules"]["Y1"]["value"], "Y1f": v["rules"]["Y1f"]["value"],
                                                  "real_inputs": {k: x for k, x in e.items() if k != "loads"}, "real_loads": {f"T_{t}": x for t, x in e["loads"].items()}},
                                                 indent=1, default=str))
    print("PASS sharedbirth: the job's code wrote a seven-load bank with the model replaced by planted captures and readouts; the verdict stops on an "
          "incomplete bank, is green with the planted answers, and fails on each of its five failing inputs")
    print("REUSE_DRYRUN_SHAREDBIRTH_OK")


if __name__ == "__main__":
    main()
