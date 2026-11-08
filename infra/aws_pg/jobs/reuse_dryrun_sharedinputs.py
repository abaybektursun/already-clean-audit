"""Weight-free dry run of SPEC10 SHAREDINPUTS (studies/reuse/SPEC10_sharedinputs.md §9). No model and no download: tiny tensors only.

    python -m infra.aws_pg.jobs.reuse_dryrun_sharedinputs --out <dir>

Part 1, the real inputs (read only): jobs.reuse_sharedinputs.case_list on studies/reuse/inputs/birth_items.json with SPEC4's live flags at birth
(results/birth/verdict_birth.json) must give 64 primary cases on 28 neurons and 43 fresh on 17 with 1 or 2 K each, and must equal the verdict's own
re-implementation (verdict_sharedinputs.expected_cases); the K-validity gate passes on them and its failing input fails for every case; the job's
check_refs passes on SPEC6's local T_15 falls (results/generic/1/generic/t15/falls). The load's captures, ablated forwards and tokens are counted
with reuse_ablate.n_forwarded and timed at SPEC6's measured units rate (printed).
Part 2, a synthetic bank written by the job's own code: jobs.reuse_sharedinputs.Job.run (check_refs, validity, case, read_arm, sample, known_spec6,
the exclusion and identity-arm records) and jobs.reuse_ablate.Job.evaluate (the parent's banking, unchanged) on tiny torch tensors, with the model
replaced at two points only: Job.side returns planted captures (scores whose top 50 outside B_t is a planted D' per cluster, three B_t members above
them), and reuse_ablate.readout (patched in this process) returns a_P = a_clean (1 - planted fall of the ablated set on the read cluster). The known
number, completeness and capture identity are synthetic passing records (they need the model); ā is random over 6 upstream layers of width 1,000
(B_t 60 members). Items: 13 primary cases (9 listed: three poles with 2 or 3 cases, one pole whose second case is cancelled at birth so it gives no
K, one pole alone, one pole of the same neuron with the other sign) and 9 fresh (7 listed). Planted falls: own' 0.35 to 0.55 per cluster; primary
I_O 0.30 to 0.32 and I_K 0.08 to 0.11 per K (d about 0.2: X1 reads H_shared refuted); fresh I_O 0.20 to 0.22 and I_K 0.27 to 0.30 (d about -0.08:
X1f reads H_old refuted); random' 0.02, except one fresh case with random' 0.30 (cancelled at this load, I_O 0.9: it must leave the rule).
SPEC6's falls for the listed primary cases differ from the planted ones by 0.004.
Part 3, the verdict (studies/reuse/scripts/verdict_sharedinputs.py, run with --spec on the real SPEC10): exit 4 on the bank without one fresh K's
Kown npz (nothing written); exit 0 with the planted answers (X1, X1f, I_O, I_K, K's own fall and the share equal to the plan's values over the live
cases; the known number 0.004; the exclusion's failing input 3 members per ranked set; K validity 0 with its failing input in every case); exit 1
under each failing input with its target failed, the other gates passing and both rules "not read". Prints REUSE_DRYRUN_SHAREDINPUTS_OK.
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
from infra.aws_pg.jobs import reuse_sharedinputs as JX
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
VERDICT = REPO / "studies/reuse/scripts/verdict_sharedinputs.py"
SPEC6_FALLS = REPO / "studies/reuse/results/generic/1/generic/t15/falls"
SPEC6_UNITS_TOK_PER_S = 16633.876408809203     # results/generic/1/generic/t15/metrics.json -> units_tokens_per_second (L40S)
SPEC6_FIXED_S = 135.9 + 87.9 + 16.1 + 7.0      # the same metrics.json: download_and_corpus_s, load_s, timing_s.known, timing_s.background
N_UP, W, M = 6, 1000, JX.M
MODES = {"known_next_case": ["known_spec6"], "capture_wrong_layer": ["capture_identity t15"], "positive_random": ["positive_control primary", "positive_control fresh"],
         "cancel_swapped": ["cancel_share primary", "cancel_share fresh"], "k_is_j": ["k_validity"]}
PRIMARY = [("L03N10+", 1, 0, True), ("L03N10+", 2, 0, True), ("L03N10+", 3, 4, True), ("L04N20-", 1, 0, True), ("L04N20-", 2, 0, True), ("L05N30+", 1, 2, True),
           ("L05N30+", 5, 3, True), ("L02N40+", 1, 0, True), ("L06N50+", 1, 0, True), ("L06N50+", 2, 1, True), ("L04N60+", 1, 0, True), ("L04N60+", 2, 0, False),
           ("L03N10-", 2, 1, True)]                    # (pole, c, o_star, live at birth)
FRESH = [("L03N110+", 1, 0, True), ("L03N110+", 2, 0, True), ("L03N110+", 3, 0, True), ("L05N120-", 1, 0, True), ("L05N120-", 2, 3, True), ("L04N130+", 2, 0, True),
         ("L04N130+", 4, 1, False), ("L06N140+", 1, 2, True), ("L06N140+", 3, 2, True)]
EXPECTED_K = {"L03N10+:c1": [2, 3, 4], "L03N10+:c2": [1, 3, 4], "L03N10+:c3": [0, 1, 2], "L04N20-:c1": [2], "L04N20-:c2": [1], "L05N30+:c1": [3, 5],
              "L05N30+:c5": [1, 2], "L06N50+:c1": [2], "L06N50+:c2": [0], "L03N110+:c1": [2, 3], "L03N110+:c2": [1, 3], "L03N110+:c3": [1, 2],
              "L05N120-:c1": [2, 3], "L05N120-:c2": [0, 1], "L06N140+:c1": [3], "L06N140+:c3": [1]}     # worked by hand from §3's words
CANCEL_T15 = ("L06N140+", 3)


def dump(p: pathlib.Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def load_verdict_module():
    spec = importlib.util.spec_from_file_location("verdict_sharedinputs", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Plan:
    """Planted D' per cluster, the scores that make SPEC4's ranking return them, and the planted fall of every (read cluster, ablated set)."""

    def __init__(self, items: dict, mask: np.ndarray, rng):
        self.mask, self.rng = mask, rng
        self.ctx = JX.cluster_contexts(items)
        self.cluster_of = {(p, tuple(t)): cl for (p, cl), t in self.ctx.items()}
        assert len(self.cluster_of) == len(self.ctx), "two clusters share a context list"
        pset = {x["pole"] for x in items["primary"]}
        self.prim = {p: p in pset for p, _ in self.ctx}
        self.ostar = {(x["pole"], x["c"]): x["o_star"] for x in items["primary"] + items["fresh"]}
        self.D, self.set_cluster, self.base = {}, {}, {}
        for (p, cl), t in sorted(self.ctx.items()):
            L = int(p[1:3])
            U = [(l, k) for l in range(L) for k in range(W) if not mask[l, k]]
            D = [U[int(i)] for i in rng.choice(len(U), M, replace=False)]
            self.D[(p, cl)] = D
            self.set_cluster[(p, frozenset(D))] = cl
            self.base[(p, cl)] = rng.uniform(0.5, 1.5, len(t)).astype(np.float32)

    def own(self, p: str, rc: int) -> float:
        return 0.35 + 0.04 * ((rc + int(p[1:3])) % 6)

    def ratio(self, p: str, rc: int, sc: int) -> float:
        if rc == sc:
            return 1.0
        if self.ostar.get((p, rc)) == sc:
            return 0.9 if (p, rc) == CANCEL_T15 else (0.30 if self.prim[p] else 0.20) + 0.01 * (rc % 3)
        return (0.08 if self.prim[p] else 0.27) + 0.01 * (sc % 4)

    def random_fall(self, p: str, rc: int) -> float:
        return 0.30 if (p, rc) == CANCEL_T15 else 0.02

    def score(self, L: int, D: list) -> torch.Tensor:
        s = torch.tensor(self.rng.uniform(0, 1, L * W), dtype=torch.float32)
        for r, (l, k) in enumerate(D):
            s[l * W + k] = 100.0 - r
        bt = [(l, int(k)) for l in range(L) for k in np.flatnonzero(self.mask[l])][:3]
        assert len(bt) == 3, (L, "fewer than 3 B_t members below the pole")
        for r, (l, k) in enumerate(bt):
            s[l * W + k] = 200.0 - r
        return s

    def capture(self, unit: dict, toks):
        p, L = unit["unit"], unit["layer"]
        rc = self.cluster_of[(p, tuple(int(t) for t in toks))]
        n = len(toks)
        A = (self.score(L, self.D[(p, rc)]) / n).repeat(n, 1)
        return {"a": {l: torch.tensor(self.rng.uniform(-1, 1, (n, W)), dtype=torch.float32) for l in range(L)}}, A

    def readout(self, model, toks, unit: dict, ablation, **kw) -> dict:
        """reuse_ablate.readout's return for the planted model: a = a_clean (1 - fall)."""
        p = unit["unit"]
        rc = self.cluster_of[(p, tuple(int(t) for t in toks))]
        frac = 0.0
        if ablation:
            S = frozenset((int(l), int(k)) for l, (idx, _) in ablation.items() for k in idx.tolist())
            if S:
                sc = self.set_cluster.get((p, S))
                frac = self.own(p, rc) * self.ratio(p, rc, sc) if sc is not None else self.random_fall(p, rc)
        a = (self.base[(p, rc)] * np.float32(1.0 - frac)).astype(np.float32)
        return {"a": a, "g": a[:, None].copy(), "u": np.ones((len(a), 1), np.float32)}


class DryJob(JX.Job):
    """jobs.reuse_sharedinputs.Job with no model: setup() as the job, planted captures, synthetic known-number and capture-identity records."""

    def __init__(self, out, a, items: dict, live: dict, spec6, plan=None, abar=None):
        self.plan, self.abar_t, self.model, self.tokens, self.bos = plan, abar, None, None, 1
        self.setup(out, a, items, "cpu", W, N_UP + 1, "torch.float32", live, spec6)

    def known(self):
        self.m["known_number"] = {"ratio": 0.001, "wrong_ratio": 0.4, "tol": 0.01, "pass": True, "failing_input_failed": True, "per_pole": {}, "n_ctx": 16,
                                  "wrong_rule": "synthetic", "failing_input": "a_wrong at T_15: synthetic"}
        self.m["completeness"] = {"median_err_over_pole_max": 0.002, "wrong_layer": 0.3, "tol": 0.01, "pass": True, "failing_input_failed": True}
        self.write()

    def identity(self):
        c = self.items["primary"][0]
        self.m["capture_identity"] = {"pole": c["pole"], "job": c["job"], "n_contexts": 20, "max_abs_diff": 0.0, "tol": 1e-5, "pass": True,
                                      "failing_input": f"P's neuron read at layer {c['layer'] - 1} (the hook on L-1)", "failing_input_max_abs_diff": 1.7,
                                      "failing_input_failed": True, "a_P_abs_max": 3.0}
        self.write()

    def background(self):
        self.abar = self.abar_t
        hb = RB.high_background(self.abar, N_UP)
        self.mask = hb["mask"]
        RB.write_high_background(self.out / "high_background.json", hb, self.a.t)
        self.m["high_background"] = {k: hb[k] for k in ("cut", "count", "n_entries", "n_up", "frac", "per_layer")}
        self.write()

    def side(self, toks, unit):
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)
        return self.plan.capture(unit, toks)

    def prompts(self, toks):
        return [{"prompt": f"synthetic context {int(t)}", "marked_token": " x", "n_tokens": 8} for t in toks]


def args_ns(isha: str, bsha: str):
    return types.SimpleNamespace(stage="sharedinputs", t="T_15", revision=JR.REVISION["T_15"], items_sha256=isha, birth_verdict_sha256=bsha,
                                 harvest_window=UR.HARVEST_WINDOW, batch=32, full_batch=8, scan_tokens=20_570_743)


# ------------------------------------------------------------------------------------------ part 1: the real inputs
def real_inputs(out: pathlib.Path) -> dict:
    ip, bp = REPO / "studies/reuse/inputs/birth_items.json", REPO / "studies/reuse/results/birth/verdict_birth.json"
    items, bv = json.loads(ip.read_text()), json.loads(bp.read_text())
    live = JX.live_flags(bv)
    cl = JX.case_list(items, live)
    n = {s: sum(c["set"] == s for c in cl) for s in ("primary", "fresh")}
    nn = {s: len({c["neuron"] for c in cl if c["set"] == s}) for s in ("primary", "fresh")}
    nk = sorted({len(c["K"]) for c in cl})
    assert n == {"primary": 64, "fresh": 43} and nn == {"primary": 28, "fresh": 17} and nk == [1, 2], (n, nn, nk)
    VS = load_verdict_module()
    assert [(c["job"], [k["c"] for k in c["K"]]) for c in cl] == VS.expected_cases(items, live), "the job's case list differs from the verdict's"
    job = DryJob(out / "real_check", args_ns(hashlib.sha256(ip.read_bytes()).hexdigest(), hashlib.sha256(bp.read_bytes()).hexdigest()), items, live, SPEC6_FALLS)
    job.check_refs()
    job.validity()
    kv = job.m["k_validity"]
    assert kv["value"] == 0 and kv["failing_input"]["value"] == len(cl), kv
    by = {x["job"]: x for x in items["primary"] + items["fresh"]}
    bos, hw = 1, UR.HARVEST_WINDOW
    nf = lambda t: RA.n_forwarded(np.asarray(t, np.int64), RA.LENGTH, bos, hw)
    e = {"captures": 0, "capture_tokens": 0, "forwards": 0, "forward_tokens": 0}
    for c in cl:
        j = by[c["job"]]
        Jt, Ot, Ks = JX.capped(j["J_tok"]), JX.capped(j["O_tok"]), [k["tok"] for k in c["K"]]
        e["captures"] += len(Jt) + len(Ot) + sum(map(len, Ks))
        e["capture_tokens"] += nf(Jt) + nf(Ot) + sum(map(nf, Ks))
        nja = len(JX.j_arms([k["c"] for k in c["K"]]))
        e["forwards"] += 2 * nja * len(Jt) + sum(4 * len(k) for k in Ks)            # each arm: a clean readout and the ablated one
        e["forward_tokens"] += 2 * nja * nf(Jt) + sum(4 * nf(k) for k in Ks)
    kt = items["known"]["tok"]
    fixed = sum(RA.n_forwarded(np.asarray(x)[:JR.KNOWN_CTX], None, bos, hw) for x in kt)
    idc = next(x for x in items["primary"] if x["layer"] >= 1 and len(JX.capped(x["J_tok"])) + len(JX.capped(x["O_tok"])) >= 20)
    fixed += 2 * nf(np.concatenate([JX.capped(idc["J_tok"]), JX.capped(idc["O_tok"])])[:20])
    fixed += nf(UR.background_positions(15, 20_570_743 // hw * hw, hw)["e1"])
    e["fixed_tokens"] = fixed
    e["total_tokens"] = e["capture_tokens"] + e["forward_tokens"] + fixed
    e["units_s"] = (e["capture_tokens"] + e["forward_tokens"]) / SPEC6_UNITS_TOK_PER_S
    e["job_wall_h"] = (e["units_s"] + SPEC6_FIXED_S) / 3600
    e["K_reads"] = sum(len(c["K"]) for c in cl)
    e["n"], e["neurons"] = n, nn
    e["spec6_checked"] = job.m["refs_checked"]["n_cases"]
    shutil.rmtree(out / "real_check")
    return e


# ------------------------------------------------------------------------------------------ part 2: the synthetic bank
def build(out: pathlib.Path) -> dict:
    rng = np.random.default_rng(0)
    abar = torch.tensor(rng.normal(0, 1, (N_UP, W)), dtype=torch.float32)
    mask = RB.high_background(abar, N_UP)["mask"].numpy()
    tok = iter(range(10, 10 ** 7, 200))
    ctx = {}

    def cases(spec, label):
        out_ = []
        for pole, c, o, _ in spec:
            for cl in (c, o):
                if (pole, cl) not in ctx:
                    ctx[(pole, cl)] = [next(tok) for _ in range(6 + (len(ctx) * 5) % 7)]
            out_.append({"job": f"{pole}:c{c}", "pole": pole, "c": c, "o_star": o, "layer": int(pole[1:3]), "neuron": int(pole[4:-1]), "sign": pole[-1], "tercile": 0,
                         "birth": 10, "t_before": 9, "label": label, "J_tok": ctx[(pole, c)], "O_tok": ctx[(pole, o)], "ceiling": None})
        return out_
    items = {"primary": cases(PRIMARY, "related"), "fresh": cases(FRESH, "unrelated"), "planted": [],
             "known": {"poles": [], "tok": [], "a_bank": {}, "a_wrong": {}, "wrong_rule": "synthetic"}, "sources": {"grid": ["t15_step1413814"]}}
    ip = out / "birth_items.json"
    dump(ip, items)
    bvp = out / "verdict_birth.json"
    dump(bvp, {"cases": {"primary_birth": [{"job": f"{p}:c{c}", "cancelled": not lv} for p, c, _, lv in PRIMARY],
                         "fresh_birth": [{"job": f"{p}:c{c}", "cancelled": not lv} for p, c, _, lv in FRESH]}})
    live = {f"{p}:c{c}": lv for p, c, _, lv in PRIMARY + FRESH}
    plan = Plan(items, mask, rng)
    RA.readout = plan.readout          # the model's readout, replaced in this process only
    # SPEC6's T_15 falls for the listed primary cases: the planted own' and cross_O moved by +-0.004
    s6 = out / "spec6_falls"
    s6.mkdir(parents=True)
    by = {x["job"]: x for x in items["primary"] + items["fresh"]}
    plist = [j for j in EXPECTED_K if j in {x["job"] for x in items["primary"]}]
    for i, job in enumerate(plist):
        x = by[job]
        p, c, o = x["pole"], x["c"], x["o_star"]
        dlt = 0.004 * (1 if i % 2 == 0 else -1)
        base = plan.base[(p, c)]
        for arm, f in (("own", plan.own(p, c) + dlt), ("cross", plan.own(p, c) * plan.ratio(p, c, o) - dlt)):
            np.savez_compressed(s6 / f"{JX.key_of(x)}__{arm}__g1__free.npz", unit=JX.key_of(x), pole=p, arm=arm, exclusion="g1", scale="free",
                                tok=np.asarray(x["J_tok"], np.int64), a_before=base, a_after=(base * np.float32(1 - f)).astype(np.float32))
    d = out / "run" / "t15"
    job = DryJob(d, args_ns(hashlib.sha256(ip.read_bytes()).hexdigest(), hashlib.sha256(bvp.read_bytes()).hexdigest()), items, live, s6, plan, abar)
    assert [(c["job"], [k["c"] for k in c["K"]]) for c in job.cases] == list(EXPECTED_K.items()), [(c["job"], [k["c"] for k in c["K"]]) for c in job.cases]
    job.run()
    job.m.update({"download_and_corpus_s": 60.0, "load_s": 30.0, "device_name": "none (dry run)"})
    job.m["timing_s"]["wall_after_load"] = 600.0
    job.write()
    # the plan's answers, worked from the plan and EXPECTED_K, not from the bank
    rows = []
    for jb, Ks in EXPECTED_K.items():
        x = by[jb]
        p, c, o = x["pole"], x["c"], x["o_star"]
        IO, IK = plan.ratio(p, c, o), float(np.mean([plan.ratio(p, c, k) for k in Ks]))
        rows.append({"job": jb, "set": "primary" if jb in {y["job"] for y in items["primary"]} else "fresh", "I_O": IO, "I_K": IK, "d": IO - IK,
                     "Kown": float(np.mean([plan.own(p, k) for k in Ks])), "live": (p, c) != CANCEL_T15})
    n_ranked = sum(2 + 3 * len(Ks) for Ks in EXPECTED_K.values())
    return {"items": ip, "bv": bvp, "spec6": s6, "run": out / "run", "rows": rows, "n_ranked": n_ranked, "n_cases": len(EXPECTED_K), "plist": plist, "by": by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    e = real_inputs(out)
    print(f"real inputs: {e['n']['primary']} primary cases on {e['neurons']['primary']} neurons and {e['n']['fresh']} fresh on {e['neurons']['fresh']}, 1 to 2 K each "
          f"({e['K_reads']} K reads); equal to the verdict's list; K validity 0, its failing input in every case; SPEC6's falls checked for {e['spec6_checked']} cases")
    print(f"load estimate: captures {e['captures']} ({e['capture_tokens']} tokens), ablated-arm forwards {e['forwards']} ({e['forward_tokens']} tokens, clean + ablated "
          f"per arm), fixed blocks {e['fixed_tokens']} tokens; total {e['total_tokens']} tokens; units about {e['units_s']:.0f} s at SPEC6's "
          f"{SPEC6_UNITS_TOK_PER_S:.0f} tokens/s; job wall about {e['job_wall_h']:.2f} h")

    b = build(out)
    vout = out / "verdict"
    cmd = [sys.executable, str(VERDICT), "--items", str(b["items"]), "--birth-verdict", str(b["bv"]), "--spec6-falls", str(b["spec6"]), "--out", str(vout)]
    exits = {}
    print(f"synthetic bank: {b['n_cases']} listed cases written by jobs.reuse_sharedinputs.Job.run with planted captures and readouts; {b['n_ranked']} ranked sets")

    # an incomplete bank stops the verdict
    inc = out / "run_incomplete"
    shutil.copytree(b["run"], inc)
    gone = f"{JX.key_of(b['by']['L05N120-:c2'])}__Kown_c1.npz"
    (inc / "t15" / "falls" / gone).unlink()
    r = subprocess.run(cmd + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "SHAREDINPUTS_BANK_INCOMPLETE" in r.stderr and gone in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits["incomplete"] = r.returncode
    print("verdict on a bank without", gone, ":", r.stderr.splitlines()[1].strip(), "exit", r.returncode)
    shutil.rmtree(inc)

    # green, with the planted answers
    r = subprocess.run(cmd + ["--run", str(b["run"])], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_SHAREDINPUTS_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_sharedinputs.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    live = {s: [x for x in b["rows"] if x["set"] == s and x["live"]] for s in ("primary", "fresh")}
    assert (v["count"]["primary"], v["count"]["fresh"], v["count"]["primary_live"], v["count"]["fresh_live"]) == (9, 7, 9, 6), v["count"]
    for name, s in (("X1", "primary"), ("X1f", "fresh")):
        want = float(np.median([x["d"] for x in live[s]]))
        assert abs(v["rules"][name]["value"] - want) < 1e-5, (name, v["rules"][name]["value"], want)
        for k, f in (("I_O", "I_O"), ("I_K", "I_K"), ("K_own_fall", "Kown")):
            assert abs(v["reported"][s][k]["value"] - float(np.median([x[f] for x in live[s]]))) < 1e-5, (s, k, v["reported"][s][k]["value"])
        assert v["reported"][s]["share_I_O_gt_I_K"]["value"] == float(np.mean([x["I_O"] > x["I_K"] for x in live[s]])), (s, v["reported"][s]["share_I_O_gt_I_K"])
        assert v["reported"][s]["n_cancelled"] == (1 if s == "fresh" else 0), (s, v["reported"][s]["n_cancelled"])
    assert v["rules"]["X1"]["verdict"].startswith("H_shared refuted") and v["rules"]["X1f"]["verdict"].startswith("H_old refuted"), \
        (v["rules"]["X1"]["verdict"], v["rules"]["X1f"]["verdict"])
    assert abs(v["gates"]["known_spec6"]["value"] - 0.004) < 1e-5 and v["gates"]["known_spec6"]["n_cases"] == len(b["plist"]), v["gates"]["known_spec6"]
    assert v["gates"]["exclusion"]["failing_input"]["value"] == 3 * b["n_ranked"] and v["gates"]["exclusion"]["value"]["n_in_B"] == 0, v["gates"]["exclusion"]
    assert v["gates"]["k_validity"]["value"] == 0 and v["gates"]["k_validity"]["failing_input"]["value"] == b["n_cases"], v["gates"]["k_validity"]
    assert v["reported"]["a_before_max_abs_diff_same_contexts"] == 0.0, v["reported"]["a_before_max_abs_diff_same_contexts"]
    assert (vout / "GRID10.md").read_text().startswith("Cases 9 primary"), "GRID10.md does not open with the count line"
    samp = json.loads((vout / "samples" / "sharedinputs_samples.json").read_text())
    assert len(samp["job_samples"]) == 3 and len(samp["extremes"]) == 4 and all(x["contexts"]["own"] and x["contexts"]["crossO"] for x in samp["extremes"].values()), \
        (len(samp["job_samples"]), list(samp["extremes"]))
    print(f"planted answers: X1 {v['rules']['X1']['value']:.4f} and X1f {v['rules']['X1f']['value']:.4f} equal the plan's medians over the live cases (the fresh case "
          f"cancelled at this load left out); I_O, I_K, K's own fall and the share equal; known number {v['gates']['known_spec6']['value']:.4f}; exclusion failing "
          f"input {v['gates']['exclusion']['failing_input']['value']} members over {b['n_ranked']} ranked sets")

    # each failing input
    fails = {}
    for mode, target in MODES.items():
        r = subprocess.run(cmd + ["--run", str(b["run"]), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_sharedinputs.json").read_text())
        assert all(not f["gates"][t]["ok"] for t in target) and all(x["verdict"] == "not read" for x in f["rules"].values()), (mode, {t: f["gates"][t] for t in target})
        assert all(g["ok"] for k, g in f["gates"].items() if k not in target), (mode, {k: g["ok"] for k, g in f["gates"].items()})
        fails[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_sharedinputs exit codes:", exits)
    (out / "metrics.json").write_text(json.dumps({"exits": exits, "failing_inputs": fails, "X1": v["rules"]["X1"]["value"], "X1f": v["rules"]["X1f"]["value"],
                                                  "real_inputs": e}, indent=1))
    print("PASS sharedinputs: the job's code wrote the bank with the model replaced by planted captures and readouts; the verdict stops on an incomplete "
          "bank, is green with the planted answers, and fails on each of its five failing inputs")
    print("REUSE_DRYRUN_SHAREDINPUTS_OK")


if __name__ == "__main__":
    main()
