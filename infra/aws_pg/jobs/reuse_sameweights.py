"""SPEC9 SAMEWEIGHTS job, one load at T_15, captures only (studies/reuse/SPEC9_sameweights.md §2-§5, §9). A subclass of jobs.reuse_birth.Job:
the known number and completeness (jobs.reuse_ablate.Job.known), ā and B_t (JB.Job.background: high_background.json), the capture and the direct
attribution (JB.Job.side: unite_route.capture + reuse_ablate.direct_A), the ranking with B_t excluded (JB.Job.rank: reuse_birth.top_m_excluded) and
the top-200 bank (reuse_birth.top_bank) are SPEC4's code, imported unchanged; a_P from g and u is jobs.reuse_dating.Job.gu.

    python -m infra.aws_pg.jobs.reuse_sameweights --stage sameweights --t T_15 --revision stage1-step1413814 --items <s3 birth_items.json> \
        --corpus <s3> --spec6-bank <s3 SPEC6 T_15 step dir> --out <dir>

Before any forward: SPEC6's T_15 bank (metrics.json, prebirth/<case>__J.npz and __O.npz) is fetched and every primary case's two files are asserted
present, at T_15, of the case's pole and on the case's capped J and O* contexts; the swaps are drawn and the swap-validity gate is read and asserted.
Swaps (SPEC9 §2): for each case in item order (the 150 primary, then the 93 fresh), three distinct other cases on another neuron (the pole without
its sign), drawn by one numpy default_rng(9) with rng.choice(len(pool), 3, replace=False) from the pool of all 243 cases on another neuron, in item
order (swaps.json). A swapped set is the source case's J contexts (capped at 12 by rank) read on P.
Then the parent's known number and completeness (asserted), the capture identity (the first primary case with L >= 1 and 20 capped J + O* contexts:
its first 20 captured twice in identical calls, a_P equal to 1e-5; failing input P's neuron read at layer L-1 from the second call; asserted after it
is written), ā over layers 0..30 on UNITE's e1 positions and B_t (abar.npz, background.json, high_background.json).
Units, every set capped at 12 contexts by rank, 256-token window, m 50:
  cases (243): captures of J, O*, swap_1, swap_2, swap_3 at the marked token on P; per set the direct attribution summed over its contexts, D' = the
      top 50 with B_t excluded, the top-200 bank (no exclusion) and a_P per context; banked at once in sets/<pole>__c<c>.npz.
  primary ceiling splits (129 distinct, SPEC4's): captures of side 1 and side 2 on P; D' and the top-200 bank per side; ceiling/<pole>__c<c>.npz.
npz fields (sets/ and ceiling/): job or split, pole, layer, t, sets (the set names), per set X: tok_X [n] int64, D_layer_X [50] int16, D_neuron_X [50]
int32, D_in_B_X [50] bool, top_idx_X / top_A_X / top_abar_X [200]; sets/ adds aP_X [n] float64 (sign silu(g) u of P), aP_mean_X, swap_jobs [3],
swap_poles [3]; ceiling/ adds c.
metrics.json (rewritten after every unit): per case s_O = |D'(J) ∩ D'(O*)| / 50, s_swap_k, s_swap (their mean), d = s_O - s_swap, the O* overlaps with
the swaps, P's mean a_P per set, the swap sources, and for primary cases the overlap with SPEC6's D'(J) and D'(O*); per split s; gates and records:
known_number, completeness (the parent's), capture_identity, swap_validity (failing input: per case a swap drawn from P's own neuron), known_spec6
(median overlap with SPEC6's sets, J and O*, line 0.90; failing input the next case's SPEC6 sets; recorded, the verdict decides), exclusion (no D'
member in B_t; failing input the unexcluded ranking reuse_ablate.top_m of every set against B_t), high_background, memory, timing, counts.
samples/cases.json: the first three primary cases: for J, O* and swap_1 every context's prompt, marked token and a_P, and the set's D'.
Statistics over the bank (the rules W2 and W2f, the room, the reported items) are the verdict's (studies/reuse/scripts/verdict_sameweights.py).
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_birth as JB
from infra.aws_pg.jobs import reuse_dating as JD
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

M = RB.M
CAP = JB.CAP
N_SWAP = 3
SWAP_SEED = 9
N_ID = 20
ID_TOL = JD.ID_TOL
KNOWN_LINE = 0.90
N_SAMPLE_CASES = 3
SAMPLE_SETS = ("J", "O", "swap_1")
SETS = ("J", "O") + tuple(f"swap_{k}" for k in range(1, N_SWAP + 1))


# ------------------------------------------------------------------------------------------ pure helpers (the dry run calls these too)
def neuron_of(pole: str) -> str:
    return pole[:-1]


def key_of(j: dict) -> str:
    return f"{j['pole']}__c{j['c']}"


def capped(toks) -> np.ndarray:
    return np.asarray(toks, np.int64)[:CAP]


def overlap(A: list, B: list) -> float:
    return len(set(map(tuple, A)) & set(map(tuple, B))) / M


def draw_swaps(cases: list, seed: int = SWAP_SEED, k: int = N_SWAP) -> dict:
    """{job: [source job x k]}: per case in item order, k distinct cases on another neuron, one default_rng(seed) over the whole pass."""
    rng = np.random.default_rng(seed)
    out = {}
    for j in cases:
        pool = [c["job"] for c in cases if neuron_of(c["pole"]) != neuron_of(j["pole"])]
        out[j["job"]] = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
    return out


def same_neuron_swaps(cases: list) -> dict:
    """The swap-validity gate's failing input: per case a swap drawn from P's own neuron (the first other case on it in item order, else the case)."""
    out = {}
    for j in cases:
        same = [c["job"] for c in cases if neuron_of(c["pole"]) == neuron_of(j["pole"]) and c["job"] != j["job"]]
        out[j["job"]] = (same or [j["job"]])[0]
    return out


def swap_problems(j: dict, srcs: list, by_job: dict) -> list:
    out = [] if len(set(srcs)) == len(srcs) else [f"{j['job']}: two swaps name one case"]
    for s in srcs:
        if s == j["job"]:
            out.append(f"{j['job']}: swap {s} is the case itself")
        if neuron_of(by_job[s]["pole"]) == neuron_of(j["pole"]):
            out.append(f"{j['job']}: swap {s} is on P's neuron {neuron_of(j['pole'])}")
    return out


def swap_validity(cases: list, swaps: dict) -> dict:
    """The gate record: every case's swaps valid; the failing input (a swap from P's own neuron in place of swap_1) invalid for every case."""
    by_job = {c["job"]: c for c in cases}
    bad = {j["job"]: swap_problems(j, swaps[j["job"]], by_job) for j in cases}
    sn = same_neuron_swaps(cases)
    fi = {j["job"]: swap_problems(j, [sn[j["job"]]] + swaps[j["job"]][1:], by_job) for j in cases}
    n_bad, n_fi = sum(bool(v) for v in bad.values()), sum(bool(v) for v in fi.values())
    return {"value": n_bad, "pass": n_bad == 0, "n_cases": len(cases), "problems": [p for v in bad.values() for p in v][:20],
            "failing_input": {"input": "per case swap_1 replaced by a case on P's own neuron (the first other case on it in item order, else the case itself)",
                              "value": n_fi, "sources": sn}, "failing_input_failed": n_fi == len(cases),
            "rule": "every swap's source is another case on another neuron (the pole without its sign), three distinct; its contexts are its own capped J contexts",
            "seed": SWAP_SEED, "n_swap": N_SWAP}


def set_record(j: dict, fresh: bool, sets: dict, srcs: list, by_job: dict) -> dict:
    """The per-case record of the five sets' D' (sets[X]["D"]) and a_P (sets[X]["aP"])."""
    D = {x: s["D"] for x, s in sets.items()}
    sk = [overlap(D["J"], D[f"swap_{k}"]) for k in range(1, N_SWAP + 1)]
    so = [overlap(D["O"], D[f"swap_{k}"]) for k in range(1, N_SWAP + 1)]
    s_O = overlap(D["J"], D["O"])
    return {"set": "fresh" if fresh else "primary", "pole": j["pole"], "layer": j["layer"], "tercile": j["tercile"], "birth": j["birth"],
            "n": {x: int(len(s["tok"])) for x, s in sets.items()}, "aP_mean": {x: float(np.mean(s["aP"])) for x, s in sets.items()},
            "swaps": [{"job": s, "pole": by_job[s]["pole"], "layer": by_job[s]["layer"]} for s in srcs],
            "s_O": s_O, "s_swap_k": sk, "s_swap": float(np.mean(sk)), "d": s_O - float(np.mean(sk)), "s_O_swap_k": so, "path": f"sets/{key_of(j)}.npz"}


def known_spec6(P: list, D: dict, spec6: dict) -> dict:
    """Median over primary cases of |D'(X) ∩ D'_SPEC6(X)| / 50 for X in J, O*; failing input the next case's SPEC6 sets (item order, cyclic)."""
    jobs = [j["job"] for j in P if j["job"] in D]
    n = len(jobs)
    val = {x: float(np.median([overlap(D[a][x], spec6[a][x]) for a in jobs])) for x in ("J", "O")}
    fi = {x: float(np.median([overlap(D[jobs[i]][x], spec6[jobs[(i + 1) % n]][x]) for i in range(n)])) for x in ("J", "O")}
    return {"value": val, "pass": min(val.values()) >= KNOWN_LINE, "line": KNOWN_LINE, "n_cases": n,
            "failing_input": {"input": "the next case's SPEC6 sets (item order, cyclic)", "value": fi}, "failing_input_failed": max(fi.values()) < KNOWN_LINE,
            "rule": "median over primary cases of |D'(X) ∩ D'_SPEC6(X)| / 50, X = J and O*, each at least 0.90 (recorded here; the verdict decides)"}


def bank(path: pathlib.Path, meta: dict, sets: dict, mask) -> None:
    """One npz: the meta fields as they are; per set X tok_X, D_layer_X, D_neuron_X, D_in_B_X, top_idx_X, top_A_X, top_abar_X, and aP_X, aP_mean_X
    when the set carries a_P."""
    arr = {}
    for x, s in sets.items():
        D = s["D"]
        assert len(D) == M, (path.name, x, len(D))
        arr.update({f"tok_{x}": np.asarray(s["tok"], np.int64), f"D_layer_{x}": np.array([l for l, _ in D], np.int16),
                    f"D_neuron_{x}": np.array([k for _, k in D], np.int32), f"D_in_B_{x}": RB.in_B(D, mask),
                    f"top_idx_{x}": np.asarray(s["top"]["idx"], np.int64), f"top_A_{x}": np.asarray(s["top"]["A"], np.float32),
                    f"top_abar_{x}": np.asarray(s["top"]["abar"], np.float32)})
        if "aP" in s:
            arr[f"aP_{x}"] = np.asarray(s["aP"], np.float64)
            arr[f"aP_mean_{x}"] = float(np.mean(s["aP"]))
    np.savez_compressed(path, sets=np.array(list(sets)), **meta, **arr)


def excl_add(excl: dict, D: list, D0: list, mask):
    """Exclusion counts: D' members in B_t (must be 0) and, the failing input, members of the unexcluded top 50 in B_t."""
    n, n0 = int(RB.in_B(D, mask).sum()), int(RB.in_B(D0, mask).sum())
    excl["n_sets"] += 1
    excl["n_members"] += len(D)
    excl["n_in_B"] += n
    excl["unexcluded_n_in_B"] += n0
    excl["unexcluded_n_sets_with_B"] += int(n0 > 0)


class Job(JB.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str, refs: dict):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "sets", "ceiling", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert a.stage == "sameweights" and a.t == "T_15", (a.stage, a.t)
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P and the coefficients assume silu"
        self.refs = refs
        self.cases = items["primary"] + items["fresh"]
        self.by_job = {c["job"]: c for c in self.cases}
        assert len(self.by_job) == len(self.cases) and len({key_of(c) for c in self.cases}) == len(self.cases), "a job id or a case key repeats"
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "m": M, "cap": CAP, "sets": list(SETS),
                  "ranking": "direct attribution summed over the set's contexts (reuse_ablate.direct_A), top 50 with B_t excluded (reuse_birth.top_m_excluded)",
                  "a_P": "sign * silu(g) * u of the pole at the marked token, g and u fp32 from unite_route.capture, a_P float64 from them",
                  "timing_s": {}, "tokens_forwarded": 0, "forwards": 0, "captures": 0, "refs": {k: str(v) for k, v in refs.items()}, "cases": {}, "ceiling": {}}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0, "unexcluded_n_in_B": 0, "unexcluded_n_sets_with_B": 0}
        self.write()

    # -------------------------------------------------------------------------------------- before any forward
    def check_refs(self):
        """SPEC6's T_15 bank holds both prebirth files of every primary case, at T_15, of its pole, on its capped J and O* contexts."""
        r = self.refs
        s6 = json.loads(pathlib.Path(r["spec6_metrics"]).read_text())
        assert s6["t"] == "T_15" and s6["status"] == "done" and s6["items_sha256"] == self.a.items_sha256, \
            ("SPEC6's bank is not the T_15 load of these items", s6["t"], s6["status"], s6["items_sha256"])
        self.spec6 = {}
        for j in self.items["primary"]:
            sides = {}
            for x, toks in (("J", capped(j["J_tok"])), ("O", capped(j["O_tok"]))):
                p = pathlib.Path(r["spec6_prebirth"]) / f"{key_of(j)}__{x}.npz"
                assert p.exists(), (j["job"], f"SPEC6's bank lacks {p.name}")
                z = np.load(p)
                assert str(z["t"]) == "T_15" and str(z["pole"]) == j["pole"] and str(z["side"]) == x, (p.name, str(z["t"]), str(z["pole"]), str(z["side"]))
                assert np.array_equal(z["tok"], toks), (p.name, "SPEC6's contexts differ from the items'")
                assert len(z["D_layer"]) == len(z["D_neuron"]) == M, (p.name, len(z["D_layer"]))
                sides[x] = list(zip(z["D_layer"].astype(int).tolist(), z["D_neuron"].astype(int).tolist()))
            self.spec6[j["job"]] = sides
        self.m["refs_checked"] = {"n_cases": len(self.spec6), "rule": "every primary case's SPEC6 prebirth __J and __O npz present, T_15, its pole, its capped contexts"}
        self.write()

    def draw(self):
        self.swaps = draw_swaps(self.cases)
        NH.dump(self.out / "swaps.json", {"rule": f"numpy default_rng({SWAP_SEED}); per case in item order (primary, then fresh) rng.choice(len(pool), "
                                                  f"{N_SWAP}, replace=False) over the cases on another neuron in item order", "swaps": self.swaps})
        self.m["swap_validity"] = swap_validity(self.cases, self.swaps)
        self.write()
        sv = self.m["swap_validity"]
        assert sv["pass"], f"swap validity failed: {sv['problems']}"
        assert sv["failing_input_failed"], f"swap validity's failing input passed for some case: {sv['failing_input']['value']} of {sv['n_cases']}"

    # -------------------------------------------------------------------------------------- the capture identity
    def identity(self):
        """Two identical captures of the first eligible primary case's first 20 contexts; the failing input reads P's neuron at layer L-1."""
        t0 = time.time()
        c = next(j for j in self.items["primary"] if j["layer"] >= 1 and len(capped(j["J_tok"])) + len(capped(j["O_tok"])) >= N_ID)
        unit = UR.pole_unit(c["pole"])
        L = unit["layer"]
        toks = np.concatenate([capped(c["J_tok"]), capped(c["O_tok"])])[:N_ID]
        _, _, a1 = JD.Job.gu(UR.capture(self.model, toks, L, {L}, **self.kw(RA.LENGTH)), unit)
        cap2 = UR.capture(self.model, toks, L, {L - 1, L}, **self.kw(RA.LENGTH))
        _, _, a2 = JD.Job.gu(cap2, unit)
        _, _, aw = JD.Job.gu(cap2, unit, L - 1)
        self.m["captures"] += 2 * N_ID
        self.m["tokens_forwarded"] += 2 * RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
        d, dw = float(np.abs(a2 - a1).max()), float(np.abs(aw - a1).max())
        self.m["capture_identity"] = {"pole": c["pole"], "job": c["job"], "n_contexts": N_ID, "max_abs_diff": d, "tol": ID_TOL, "pass": d <= ID_TOL,
                                      "failing_input": f"P's neuron read at layer {L - 1} (the hook on L-1)", "failing_input_max_abs_diff": dw,
                                      "failing_input_failed": dw > ID_TOL, "a_P_abs_max": float(np.abs(a1).max())}
        self.m["timing_s"]["identity"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["capture_identity"]["pass"], f"capture identity failed: {d}"
        assert self.m["capture_identity"]["failing_input_failed"], f"capture identity's failing input passed: {dw}"

    # -------------------------------------------------------------------------------------- one set, one case, one split
    def read(self, toks, unit, with_aP: bool) -> dict:
        cap, A = self.side(toks, unit)
        sc = A.sum(0)
        D = self.rank(sc)
        excl_add(self.excl, D, RA.top_m(sc, M, self.W), self.mask)
        out = {"tok": np.asarray(toks, np.int64), "D": D, "top": RB.top_bank(sc, self.abar, self.W)}
        if with_aP:
            out["aP"] = JD.Job.gu(cap, unit)[2]
        return out

    def case(self, j: dict, fresh: bool, sample: bool):
        t0 = time.time()
        unit = UR.pole_unit(j["pole"])
        L = unit["layer"]
        assert L == j["layer"], (j["job"], L, j["layer"])
        srcs = self.swaps[j["job"]]
        toks = {"J": capped(j["J_tok"]), "O": capped(j["O_tok"]), **{f"swap_{k}": capped(self.by_job[s]["J_tok"]) for k, s in enumerate(srcs, 1)}}
        sets = {x: self.read(t, unit, True) for x, t in toks.items()}
        key = key_of(j)
        bank(self.out / "sets" / f"{key}.npz", {"job": j["job"], "pole": j["pole"], "layer": L, "t": self.a.t, "swap_jobs": np.array(srcs),
                                                "swap_poles": np.array([self.by_job[s]["pole"] for s in srcs])}, sets, self.mask)
        rec = set_record(j, fresh, sets, srcs, self.by_job)
        if not fresh:
            rec["known"] = {x: overlap(sets[x]["D"], self.spec6[j["job"]][x]) for x in ("J", "O")}
            self.D_primary[j["job"]] = {x: sets[x]["D"] for x in ("J", "O")}
        if sample:
            p = self.out / "samples" / "cases.json"
            cur = json.loads(p.read_text()) if p.exists() else {}
            cur[j["job"]] = {"pole": j["pole"], "layer": L, "s_O": rec["s_O"], "s_swap_k": rec["s_swap_k"],
                             "sets": {x: {"source_job": j["job"] if x in ("J", "O") else srcs[int(x[-1]) - 1], "D": [[int(l), int(k)] for l, k in sets[x]["D"]],
                                          "aP_mean": float(np.mean(sets[x]["aP"])),
                                          "contexts": [{"tok": int(t), **pr, "a_P": float(a)} for t, pr, a in zip(sets[x]["tok"], self.prompts(sets[x]["tok"]), sets[x]["aP"])]}
                                      for x in SAMPLE_SETS}}
            NH.dump(p, cur)
        rec["seconds"] = round(time.time() - t0, 1)
        self.m["cases"][j["job"]] = rec

    def ceiling(self, key: str, pole: str, c: dict):
        t0 = time.time()
        unit = UR.pole_unit(pole)
        s1, s2 = (capped(x) for x in c["sides_tok"])
        sets = {"side1": self.read(s1, unit, False), "side2": self.read(s2, unit, False)}
        bank(self.out / "ceiling" / f"{key}.npz", {"split": key, "pole": pole, "layer": unit["layer"], "c": c["c"], "t": self.a.t}, sets, self.mask)
        self.m["ceiling"][key] = {"pole": pole, "c": c["c"], "n_sides": [int(len(s1)), int(len(s2))], "s": overlap(sets["side1"]["D"], sets["side2"]["D"]),
                                  "path": f"ceiling/{key}.npz", "seconds": round(time.time() - t0, 1)}

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.check_refs()
        self.draw()
        self.known()
        self.m["captures"] += sum(min(JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.identity()
        self.background()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P, F = self.items["primary"], self.items["fresh"]
        splits = self.splits_of(P)
        self.m["n_to_read"] = {"primary": len(P), "fresh": len(F), "splits": len(splits)}
        self.m["n_done"] = {k: 0 for k in self.m["n_to_read"]}
        self.m["samples"] = {"cases": [j["job"] for j in P[:N_SAMPLE_CASES]], "sets": list(SAMPLE_SETS), "file": "samples/cases.json"}
        self.D_primary = {}
        for fresh, cases in ((False, P), (True, F)):
            for i, j in enumerate(cases):
                self.case(j, fresh, sample=not fresh and i < N_SAMPLE_CASES)
                self.m["n_done"]["fresh" if fresh else "primary"] += 1
                self.write()
            if not fresh:
                self.m["known_spec6"] = known_spec6(P, self.D_primary, self.spec6)
                self.write()
        for key, (pole, c) in splits.items():
            self.ceiling(key, pole, c)
            self.m["n_done"]["splits"] += 1
            self.write()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.m["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0, "failing_input_failed": self.excl["unexcluded_n_in_B"] > 0,
                               "rule": "no member of any D' is in B_t; failing input: the unexcluded top 50 (reuse_ablate.top_m) of every set against B_t"}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["sameweights"])
    ap.add_argument("--t", required=True, choices=["T_15"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--spec6-bank", required=True, help="SPEC6's T_15 step directory (local or s3): metrics.json and prebirth/")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def fetch_refs(a, fetch) -> dict:
    s6 = a.spec6_bank.rstrip("/")
    return {"spec6_metrics": fetch(f"{s6}/metrics.json", "reuse_sameweights_spec6_t15/metrics.json"),
            "spec6_prebirth": fetch(f"{s6}/prebirth/", "reuse_sameweights_spec6_t15/prebirth", recursive=True), "spec6_bank": a.spec6_bank}


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    refs = fetch_refs(a, fetch)
    items_path = fetch(a.items, "reuse_sameweights_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
    assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda", refs)
    job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha,
                  "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
    job.m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("REUSE_SAMEWEIGHTS_OK", json.dumps({"t": a.t}), flush=True)


if __name__ == "__main__":
    main()
