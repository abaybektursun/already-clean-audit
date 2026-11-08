"""Weight-free dry run of SPEC9 SAMEWEIGHTS (studies/reuse/SPEC9_sameweights.md §9). No model and no download: tiny tensors only.

    python -m infra.aws_pg.jobs.reuse_dryrun_sameweights --out <dir>

A synthetic bank in the layout jobs.reuse_sameweights writes (<run>/t15/: metrics.json, high_background.json, swaps.json, samples/cases.json,
sets/<case>.npz, ceiling/<split>.npz), built with the job's own helpers (draw_swaps, swap_validity, set_record, known_spec6, bank, excl_add) and
SPEC4's code (reuse_birth.high_background, write_high_background, top_m_excluded, top_bank, reuse_ablate.top_m) on tiny torch tensors: 6 upstream
layers of width 1,000 (B_t 60 members), 12 primary cases on 10 neurons (two cases on one pole, two poles of one neuron; 11 distinct ceiling splits) and
10 fresh cases on 9 neurons; beside it SPEC6's T_15 prebirth files, SPEC4's birth verdict (one primary and one fresh case cancelled) and doc offsets.
Planted answers: each set's score puts the planted D' at ranks 1..50 outside B_t and three B_t members above them, so top_m_excluded returns the
planted D' and the unexcluded top 50 holds 3 B_t members. Primary: |D'(J) ∩ D'(O*)| 5, 6, 7 (by case index mod 3), swaps 0, 1, 2 shared members
with D'(J) (s_swap 0.02), so d is 0.08 to 0.12 and W2 reads H_weights refuted. Fresh: |D'(J) ∩ D'(O*)| 0 or 1, s_swap 1/150, so W2f reads
H_specific refuted. The swaps share nothing with D'(O*) outside D'(J) ∩ D'(O*) (s_O_swap 0). swap_3 of every even-index case carries P's mean a_P
under a tenth of J's (the §8 restricted read drops it). SPEC6's D'(J) differs from the job's by one member (0.98), D'(O*) not at all (1.00). Ceiling
halves share 18 to 22 members (room 0.40 - 0.06). Two primary cases have J and O* in one document.
Checks: verdict_sameweights.py stops on a bank without one case's sets npz (exit 4, nothing written); is green (exit 0) with the planted answers
(W2 and W2f medians equal to an independent recomputation from the plan over the live cases, the restricted read, the known number 0.98 and 1.00,
the room, the shared-document count, the exclusion's failing input 3 members per set); exits 1 under each failing input with its target failed and
both rules "not read". Prints REUSE_DRYRUN_SAMEWEIGHTS_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_sameweights as JS
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB

REPO = pathlib.Path(__file__).resolve().parents[3]
VERDICT = REPO / "studies/reuse/scripts/verdict_sameweights.py"
N_UP, W, M = 6, 1000, JS.M
MODES = {"known_next_case": "known_spec6", "capture_wrong_layer": "capture_identity t15", "swap_same_neuron": "swap_validity", "room_shuffled": "room"}
PRIMARY = [("L02N10+", 1, 0), ("L02N10+", 2, 0), ("L03N20-", 1, 5), ("L03N20+", 2, 3), ("L04N30+", 1, 2), ("L04N31+", 1, 0), ("L05N40-", 3, 1),
           ("L05N41+", 1, 4), ("L06N50+", 1, 2), ("L06N51-", 2, 0), ("L03N60+", 4, 1), ("L04N70-", 1, 3)]          # (pole, c, ceiling c)
FRESH = [("L02N110+", 1), ("L02N110+", 2), ("L03N120-", 1), ("L04N130+", 2), ("L05N140-", 1), ("L06N150+", 1), ("L03N160+", 3), ("L04N170-", 1),
         ("L05N180+", 2), ("L06N190+", 1)]
SHARED_DOC = (0, 2)                    # primary cases whose O*'s first context is in J's first context's document
CANCELLED = ("L04N70-:c1", "L06N190+:c1")


def dump(p: pathlib.Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def gate_records(pole: str, job: str) -> dict:
    """The parent's known number and completeness (unite_stats.gate_known / gate_complete) and the job's capture identity record, passing."""
    return {"known_number": {"ratio": 0.001, "wrong_ratio": 0.4, "tol": 0.01, "pass": True, "failing_input_failed": True, "per_pole": {}, "n_ctx": 16,
                             "wrong_rule": "the previous grid checkpoint's exemplar_a", "failing_input": "a_wrong at T_15: the previous grid checkpoint's exemplar_a"},
            "completeness": {"median_err_over_pole_max": 0.002, "wrong_layer": 0.3, "tol": 0.01, "pass": True, "failing_input_failed": True},
            "capture_identity": {"pole": pole, "job": job, "n_contexts": 20, "max_abs_diff": 0.0, "tol": JS.ID_TOL, "pass": True,
                                 "failing_input": "P's neuron read at layer 1 (the hook on L-1)", "failing_input_max_abs_diff": 1.7, "failing_input_failed": True,
                                 "a_P_abs_max": 3.0}}


class Plan:
    """Planted D' sets and the score vectors that make SPEC4's ranking return them."""

    def __init__(self, mask: np.ndarray, rng):
        self.mask, self.rng = mask, rng

    def universe(self, L: int) -> list:
        return [(l, k) for l in range(L) for k in range(W) if not self.mask[l, k]]

    def pick(self, pool: list, n: int, avoid: set) -> list:
        cand = [p for p in pool if p not in avoid]
        return [cand[int(i)] for i in self.rng.choice(len(cand), n, replace=False)]

    def score(self, L: int, D: list) -> torch.Tensor:
        """Noise in [0, 1); the planted D' at 100 - rank; three B_t members of layers < L at 200 and above."""
        s = torch.tensor(self.rng.uniform(0, 1, L * W), dtype=torch.float32)
        for r, (l, k) in enumerate(D):
            s[l * W + k] = 100.0 - r
        bt = [(l, k) for l in range(L) for k in np.flatnonzero(self.mask[l])][:3]
        assert len(bt) == 3, (L, "fewer than 3 B_t members below the pole")
        for r, (l, k) in enumerate(bt):
            s[l * W + int(k)] = 200.0 - r
        return s


def build(out: pathlib.Path) -> dict:
    rng = np.random.default_rng(0)
    abar_t = torch.tensor(rng.normal(0, 1, (N_UP + 1, W)), dtype=torch.float32)
    hb = RB.high_background(abar_t, N_UP)
    mask_t = hb["mask"]
    mask = mask_t.numpy()
    tok = iter(range(10, 10 ** 7, 200))                                     # one context per 100-token document
    cases = []
    for i, (pole, c, cc) in enumerate(PRIMARY):
        L = int(pole[1:3])
        nJ = (12, 9, 10, 12, 8)[i % 5]
        J, O = [next(tok) for _ in range(nJ)], [next(tok) for _ in range(12)]
        if i in SHARED_DOC:
            O[0] = J[0] + 1
        cases.append({"job": f"{pole}:c{c}", "pole": pole, "c": c, "layer": L, "neuron": int(pole[4:-1]), "sign": pole[-1], "tercile": 0, "birth": 8 + i % 5,
                      "t_before": 7 + i % 5, "label": "related", "J_tok": J, "O_tok": O, "ceiling": {"c": cc, "sides_tok": None}})
    split_sides = {}
    for j in cases:
        key = f"{j['pole']}__c{j['ceiling']['c']}"
        split_sides.setdefault(key, [[next(tok) for _ in range(6)], [next(tok) for _ in range(5 + len(split_sides) % 2)]])
        j["ceiling"]["sides_tok"] = split_sides[key]
    fresh = []
    for i, (pole, c) in enumerate(FRESH):
        L = int(pole[1:3])
        fresh.append({"job": f"{pole}:c{c}", "pole": pole, "c": c, "layer": L, "neuron": int(pole[4:-1]), "sign": pole[-1], "tercile": 0, "birth": 9 + i % 4,
                      "t_before": 8 + i % 4, "label": "unrelated", "J_tok": [next(tok) for _ in range((12, 7, 11)[i % 3])], "O_tok": [next(tok) for _ in range(12)],
                      "ceiling": None})
    items = {"primary": cases, "fresh": fresh, "known": {"poles": [], "tok": [], "a_bank": {}, "a_wrong": {}, "wrong_rule": "synthetic"},
             "sources": {"grid": ["t15_step1413814"]}, "planted": []}
    ip = out / "birth_items.json"
    dump(ip, items)
    isha = hashlib.sha256(ip.read_bytes()).hexdigest()
    allc = cases + fresh
    by = {c["job"]: c for c in allc}

    d = out / "run" / "t15"
    for sub in ("sets", "ceiling", "samples"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    RB.write_high_background(d / "high_background.json", hb, "T_15")
    swaps = JS.draw_swaps(allc)
    dump(d / "swaps.json", {"rule": "numpy default_rng(9), as the job", "swaps": swaps})
    plan = Plan(mask, rng)
    excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0, "unexcluded_n_in_B": 0, "unexcluded_n_sets_with_B": 0}
    m = {"stage": "sameweights", "t": "T_15", "revision": "stage1-step1413814", "items_sha256": isha, "status": "done", "width": W, "n_layers": N_UP + 1,
         "m": M, "cap": JS.CAP, "sets": list(JS.SETS), "pg_git_sha": "dry", "device_name": "none", "download_and_corpus_s": 60.0, "load_s": 30.0,
         "timing_s": {"wall_after_load": 600.0}, "captures": 0, "tokens_forwarded": 0, "memory_max_allocated_gb": 0.0, "cases": {}, "ceiling": {},
         "high_background": {k: hb[k] for k in ("cut", "count", "n_entries", "n_up", "frac", "per_layer")},
         "swap_validity": JS.swap_validity(allc, swaps), **gate_records(cases[0]["pole"], cases[0]["job"])}

    def read(L: int, toks, D: list, aP=None) -> dict:
        sc = plan.score(L, D)
        Dx = RB.top_m_excluded(sc, M, W, mask_t)
        assert Dx == [tuple(p) for p in D], "SPEC4's ranking did not return the planted D'"
        JS.excl_add(excl, Dx, RA.top_m(sc, M, W), mask_t)
        m["captures"] += len(toks)
        out_ = {"tok": np.asarray(toks, np.int64), "D": Dx, "top": RB.top_bank(sc, abar_t, W)}
        if aP is not None:
            out_["aP"] = aP
        return out_

    plant, spec6, D_primary = {}, {}, {}
    for fresh_set, cs in ((False, cases), (True, fresh)):
        for i, j in enumerate(cs):
            L = j["layer"]
            U = plan.universe(L)
            DJ = plan.pick(U, M, set())
            kO = (i % 2) if fresh_set else (5, 6, 7)[i % 3]
            kS = [i % 2, 0, (i + 1) % 2] if fresh_set else [(i + k) % 3 for k in range(3)]
            DO = DJ[:kO] + plan.pick(U, M - kO, set(DJ))
            Dsw = []
            for k in range(3):
                own = [p for p in DJ if p not in set(DO)][k * 3:k * 3 + kS[k]]
                Dsw.append(own + plan.pick(U, M - kS[k], set(DJ) | set(DO)))
            low = i % 2 == 0
            aPs = {"J": rng.uniform(0.8, 1.2, len(j["J_tok"][:JS.CAP])), "O": rng.uniform(0.5, 1.0, 12)}
            srcs = swaps[j["job"]]
            for k, s in enumerate(srcs, 1):
                n = len(by[s]["J_tok"][:JS.CAP])
                aPs[f"swap_{k}"] = rng.uniform(0.0, 0.05, n) if (k == 3 and low) else rng.uniform(0.5, 1.0, n)
            toks = {"J": JS.capped(j["J_tok"]), "O": JS.capped(j["O_tok"]), **{f"swap_{k}": JS.capped(by[s]["J_tok"]) for k, s in enumerate(srcs, 1)}}
            Ds = {"J": DJ, "O": DO, **{f"swap_{k}": Dsw[k - 1] for k in range(1, 4)}}
            sets = {x: read(L, toks[x], Ds[x], aPs[x]) for x in JS.SETS}
            JS.bank(d / "sets" / f"{JS.key_of(j)}.npz", {"job": j["job"], "pole": j["pole"], "layer": L, "t": "T_15", "swap_jobs": np.array(srcs),
                                                        "swap_poles": np.array([by[s]["pole"] for s in srcs])}, sets, mask_t)
            rec = JS.set_record(j, fresh_set, sets, srcs, by)
            if not fresh_set:
                s6J = DJ[:M - 1] + plan.pick(U, 1, set(DJ))
                spec6[j["job"]] = {"J": s6J, "O": DO}
                for x, D6, tk in (("J", s6J, toks["J"]), ("O", DO, toks["O"])):
                    p6 = out / "spec6" / "prebirth" / f"{JS.key_of(j)}__{x}.npz"
                    p6.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p6, t="T_15", read="generic_t15", side=x, pole=j["pole"], tok=tk, D_layer=np.array([l for l, _ in D6], np.int16),
                                        D_neuron=np.array([k for _, k in D6], np.int32))
                rec["known"] = {x: JS.overlap(sets[x]["D"], spec6[j["job"]][x]) for x in ("J", "O")}
                D_primary[j["job"]] = {x: sets[x]["D"] for x in ("J", "O")}
            rec["seconds"] = 0.1
            m["cases"][j["job"]] = rec
            keep = [kS[k] / M for k in range(3) if np.mean(aPs[f"swap_{k + 1}"]) >= 0.1 * np.mean(aPs["J"])]
            plant[j["job"]] = {"set": "fresh" if fresh_set else "primary", "neuron": j["pole"][:-1], "d": kO / M - np.mean(kS) / M,
                               "d_restricted": kO / M - np.mean(keep) if keep else None, "n_kept": len(keep), "s_O": kO / M, "s_swap": np.mean(kS) / M}
            if not fresh_set and i < JS.N_SAMPLE_CASES:
                p = d / "samples" / "cases.json"
                cur = json.loads(p.read_text()) if p.exists() else {}
                cur[j["job"]] = {"pole": j["pole"], "layer": L, "s_O": rec["s_O"], "s_swap_k": rec["s_swap_k"],
                                 "sets": {x: {"source_job": j["job"] if x in ("J", "O") else srcs[int(x[-1]) - 1], "D": [[int(l), int(k)] for l, k in sets[x]["D"]],
                                              "aP_mean": float(np.mean(aPs[x])), "contexts": [{"tok": int(t), "prompt": f"synthetic context {t}", "marked_token": " x",
                                                                                                 "n_tokens": 8, "a_P": float(a)} for t, a in zip(toks[x], aPs[x])]}
                                          for x in JS.SAMPLE_SETS}}
                dump(p, cur)
    m["known_spec6"] = JS.known_spec6(cases, D_primary, spec6)
    room = []
    for n, (key, sides) in enumerate(sorted(split_sides.items())):
        pole = key.split("__")[0]
        L = int(pole[1:3])
        U = plan.universe(L)
        kC = 18 + n % 5
        D1 = plan.pick(U, M, set())
        D2 = D1[:kC] + plan.pick(U, M - kC, set(D1))
        sets = {"side1": read(L, sides[0], D1), "side2": read(L, sides[1], D2)}
        JS.bank(d / "ceiling" / f"{key}.npz", {"split": key, "pole": pole, "layer": L, "c": int(key.split("__c")[1]), "t": "T_15"}, sets, mask_t)
        m["ceiling"][key] = {"pole": pole, "c": int(key.split("__c")[1]), "n_sides": [len(sides[0]), len(sides[1])], "s": JS.overlap(D1, D2),
                             "path": f"ceiling/{key}.npz", "seconds": 0.1}
        room.append(kC / M)
    m["exclusion"] = {**excl, "pass": excl["n_in_B"] == 0, "failing_input_failed": excl["unexcluded_n_in_B"] > 0}
    m["n_to_read"] = m["n_done"] = {"primary": len(cases), "fresh": len(fresh), "splits": len(split_sides)}
    dump(d / "metrics.json", m)
    offsets = np.arange(0, 10 ** 7 + 100, 100, dtype=np.int64)
    np.save(out / "doc_offsets.npy", offsets)
    dump(out / "verdict_birth.json", {"cases": {"primary_birth": [{"job": c["job"], "cancelled": c["job"] in CANCELLED} for c in cases],
                                                "fresh_birth": [{"job": c["job"], "cancelled": c["job"] in CANCELLED} for c in fresh]},
                                      "rules": {"S1": {"s": {"value": 0.12}, "null_s": {"p95": 0.06}}, "S3": {"s": {"value": 0.12}, "null_s": {"p95": 0.06}}}})
    return {"items": ip, "run": out / "run", "spec6": out / "spec6", "bv": out / "verdict_birth.json", "docs": out / "doc_offsets.npy", "plant": plant,
            "room": room, "n_sets": excl["n_sets"], "B_count": hb["count"], "cases": cases, "fresh": fresh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    b = build(out)
    vout = out / "verdict"
    cmd = [sys.executable, str(VERDICT), "--items", str(b["items"]), "--birth-verdict", str(b["bv"]), "--spec6-bank", str(b["spec6"]),
           "--doc-offsets", str(b["docs"]), "--out", str(vout)]
    exits = {}
    print(f"synthetic bank: {len(b['cases'])} primary and {len(b['fresh'])} fresh cases, {len(b['room'])} ceiling splits, {b['n_sets']} sets ranked by "
          f"top_m_excluded, B_t {b['B_count']} members")

    # an incomplete bank stops the verdict
    inc = out / "run_incomplete"
    shutil.copytree(b["run"], inc)
    gone = f"{JS.key_of(b['fresh'][3])}.npz"
    (inc / "t15" / "sets" / gone).unlink()
    r = subprocess.run(cmd + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "SAMEWEIGHTS_BANK_INCOMPLETE" in r.stderr and b["fresh"][3]["job"] in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits["incomplete"] = r.returncode
    print("verdict on a bank without", gone, ":", r.stderr.splitlines()[1].strip(), "exit", r.returncode)
    shutil.rmtree(inc)

    # green, with the planted answers
    r = subprocess.run(cmd + ["--run", str(b["run"])], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_SAMEWEIGHTS_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_sameweights.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    live = {s: [x for j, x in b["plant"].items() if x["set"] == s and j not in CANCELLED] for s in ("primary", "fresh")}
    assert v["count"]["primary_live"] == len(live["primary"]) == 11 and v["count"]["fresh_live"] == len(live["fresh"]) == 9, v["count"]
    for name, s in (("W2", "primary"), ("W2f", "fresh")):
        want = float(np.median([x["d"] for x in live[s]]))
        assert abs(v["rules"][name]["value"] - want) < 1e-12, (name, v["rules"][name]["value"], want)
        rr = [x["d_restricted"] for x in live[s] if x["d_restricted"] is not None]
        assert abs(v["reported"][s]["restricted"]["value"] - float(np.median(rr))) < 1e-12 and v["reported"][s]["restricted"]["n_cases"] == len(rr), (s, v["reported"][s]["restricted"])
        assert abs(v["reported"][s]["s_O_T15"]["value"] - float(np.median([x["s_O"] for x in live[s]]))) < 1e-12, s
        assert abs(v["reported"][s]["s_swap"]["value"] - float(np.median([x["s_swap"] for x in live[s]]))) < 1e-12, s
        assert v["reported"][s]["s_O_swap"]["value"] == 0.0, (s, v["reported"][s]["s_O_swap"])
    assert v["rules"]["W2"]["verdict"].startswith("H_weights refuted") and v["rules"]["W2f"]["verdict"].startswith("H_specific refuted"), \
        (v["rules"]["W2"]["verdict"], v["rules"]["W2f"]["verdict"])
    assert v["gates"]["known_spec6"]["value"] == {"J": 0.98, "O": 1.0}, v["gates"]["known_spec6"]["value"]
    assert abs(v["gates"]["room"]["value"] - (float(np.median(b["room"])) - 0.06)) < 1e-12, (v["gates"]["room"]["value"], b["room"])
    assert v["gates"]["exclusion"]["failing_input"]["value"] == 3 * b["n_sets"] and v["gates"]["exclusion"]["value"]["n_in_B"] == 0, v["gates"]["exclusion"]
    assert v["gates"]["swap_validity"]["value"] == 0 and v["gates"]["swap_validity"]["failing_input"]["value"] == len(b["plant"]), v["gates"]["swap_validity"]
    assert v["reported"]["primary"]["shared_document"]["n_sharing"] == len(SHARED_DOC) and v["reported"]["fresh"]["shared_document"]["n_sharing"] == 0, \
        (v["reported"]["primary"]["shared_document"], v["reported"]["fresh"]["shared_document"])
    assert (vout / "GRID9.md").read_text().startswith("Cases 12 primary"), "GRID9.md does not open with the count line"
    samp = json.loads((vout / "samples" / "sameweights_samples.json").read_text())
    assert len(samp["job_samples"]) == 3 and len(samp["extremes"]) == 4, (len(samp["job_samples"]), list(samp["extremes"]))
    print(f"planted answers: W2 {v['rules']['W2']['value']:.4f} and W2f {v['rules']['W2f']['value']:.4f} equal the plan's medians over the live cases; "
          f"restricted read, s_O, s_swap equal; known number J 0.98, O* 1.00; room {v['gates']['room']['value']:.3f}; exclusion failing input "
          f"{v['gates']['exclusion']['failing_input']['value']} members; {len(SHARED_DOC)} shared documents")

    # each failing input
    fails = {}
    for mode, target in MODES.items():
        r = subprocess.run(cmd + ["--run", str(b["run"]), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_sameweights.json").read_text())
        assert not f["gates"][target]["ok"] and all(x["verdict"] == "not read" for x in f["rules"].values()), (mode, f["gates"][target])
        assert all(g["ok"] for k, g in f["gates"].items() if k != target), (mode, {k: g["ok"] for k, g in f["gates"].items()})
        fails[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_sameweights exit codes:", exits)
    (out / "metrics.json").write_text(json.dumps({"exits": exits, "failing_inputs": fails, "W2": v["rules"]["W2"]["value"], "W2f": v["rules"]["W2f"]["value"]}, indent=1))
    print("PASS sameweights: a synthetic bank in the job's layout built with the job's helpers; the verdict stops on an incomplete bank, is green with the "
          "planted answers, and fails on each of its four failing inputs")
    print("REUSE_DRYRUN_SAMEWEIGHTS_OK")


if __name__ == "__main__":
    main()
