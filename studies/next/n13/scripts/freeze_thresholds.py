"""N13 freeze 2 (studies/next/07_PICKS/N13_SPEC.md §6.1 G-GATE, G-ID, S-BAND, G-GATE-GSM; §6.2 thresholds and room). Laptop only.

    python -m studies.next.n13.scripts.freeze_thresholds --banks <dir> --cap C --pair pair1 --out studies/next/n13/thresholds.json

Reads the four gate banks (base-A, rl-A, zero-B, id-B; MATH-500 and GSM8K) found under --banks as <arm>/<bench>/generations.jsonl.
Writes C, sigma_base, sigma_rl, sigma, delta, B, L, G, B', L', G', H, Q, the GSM8K counterparts, the sha256 of every gate bank, room,
and the gates G-GATE, G-ID, S-BAND (pair 1), G-GATE-GSM, each with its failing input evaluated.
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

import numpy as np

from infra.pg import n13_score as S

GATE_ARMS = ("base", "rl", "zero", "id")
R2 = math.sqrt(2)


def find(banks, arm: str, bench: str) -> pathlib.Path:
    hits = [p for p in pathlib.Path(banks).rglob("generations.jsonl") if p.parent.name == bench and p.parent.parent.name == arm]
    assert len(hits) == 1, (arm, bench, hits)
    return hits[0]


def load(banks, arm: str, bench: str) -> tuple:
    p = find(banks, arm, bench)
    recs = sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()), key=lambda r: r["problem_index"])
    m = json.loads((p.parent / "metrics.json").read_text())
    return recs, m, hashlib.sha256(p.read_bytes()).hexdigest()


def vec(recs: list, n=None) -> np.ndarray:
    x = np.array([r["score"] for r in recs], dtype=float)
    assert n is None or len(x) == n, (len(x), n)
    return x


def gate_numbers(xa_base, xa_rl, xb_zero, xb_id) -> dict:
    sb, sr = S.sigma(xa_base, xb_zero), S.sigma(xa_rl, xb_id)
    sg = max(sb, sr)
    B, L, Bp, Lp = S.acc(xa_base), S.acc(xa_rl), S.acc(xb_zero), S.acc(xb_id)
    G = L - B
    return {"sigma_base": sb, "sigma_rl": sr, "sigma": sg, "delta": 2 * R2 * sg, "B": B, "L": L, "G": G, "B_prime": Bp, "L_prime": Lp,
            "G_prime": Lp - Bp, "H": B + G / 2, "Q": B + G / 4, "binomial_sd_at_half": 100 * math.sqrt(0.25 / len(xa_base))}


def g_gate(t: dict, x_after, x_before, sg: float) -> dict:
    G = S.acc(x_after) - S.acc(x_before)
    p = S.mcnemar(x_after, x_before)["p"]
    return {"G": G, "G_prime": t["G_prime"], "bound": 5 * R2 * sg, "mcnemar_p": p,
            "pass": bool(G >= 5 * R2 * sg and t["G_prime"] >= 5 * R2 * sg and p < 0.001)}


def g_id(L, Lp, B, Bp, sg) -> dict:
    return {"L_minus_Lp": L - Lp, "B_minus_Bp": B - Bp, "bound": 3 * R2 * sg, "pass": bool(abs(L - Lp) <= 3 * R2 * sg and abs(B - Bp) <= 3 * R2 * sg)}


def deranged_acc(recs: list, seed: int = 0) -> float:
    d = S.derangement(len(recs), seed)
    return S.acc([S.grade(recs[d[i]]["gold"], recs[i]["extracted"])["score"] for i in range(len(recs))])


def bench_block(banks, bench: str, n=None) -> dict:
    R = {a: load(banks, a, bench) for a in GATE_ARMS}
    x = {a: vec(R[a][0], n) for a in GATE_ARMS}
    assert all(len(x[a]) == len(x["base"]) for a in GATE_ARMS)
    t = gate_numbers(x["base"], x["rl"], x["zero"], x["id"])
    sg = t["sigma"]
    gg = g_gate(t, x["rl"], x["base"], sg)
    fi = g_gate({**t, "G_prime": S.acc(x["id"]) - S.acc(x["rl"])}, x["rl"], x["id"], sg)
    gg["failing_input"] = {"what": "(rl-A, id-B) treated as the stage pair", "value": fi["G"], "failed": not fi["pass"]}
    gi = g_id(t["L"], t["L_prime"], t["B"], t["B_prime"], sg)
    gif = g_id(t["B"], t["L"], t["B"], t["B_prime"], sg)
    gi["failing_input"] = {"what": "(base-A, rl-A) treated as an identity pair", "value": t["B"] - t["L"], "failed": not gif["pass"]}
    t["room"] = {"G": t["G"], "bound_2.5delta": 2.5 * t["delta"], "pass": bool(t["G"] >= 2.5 * t["delta"])}
    t["G-GATE"], t["G-ID"] = gg, gi
    t["bank_sha256"] = {a: R[a][2] for a in GATE_ARMS}
    t["n"] = len(x["base"])
    t["_id_recs"] = R["id"][0]
    return t


def freeze(banks, cap: int, pair: str = "pair1", n_math=None, n_gsm=None, with_gsm: bool = True) -> dict:
    mt = bench_block(banks, "math500", n_math)
    idr = mt.pop("_id_recs")
    out = {"C": cap, "pair": pair, **{k: v for k, v in mt.items()}}
    if pair == "pair1":
        sb = S.s_band(mt["L_prime"])
        dv = deranged_acc(idr)
        sb["failing_input"] = {"what": "the id-B generations scored against a derangement of the golds (seed 0)", "value": dv, "failed": dv < 81.5}
        out["S-BAND"] = sb
    if with_gsm:
        g = bench_block(banks, "gsm8k", n_gsm)
        g.pop("_id_recs")
        g["G-GATE-GSM"] = g.pop("G-GATE")
        out["gsm8k"] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--pair", default="pair1")
    ap.add_argument("--n-math", type=int, default=500)
    ap.add_argument("--n-gsm", type=int, default=1319)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = freeze(a.banks, a.cap, a.pair, a.n_math, a.n_gsm)
    pathlib.Path(a.out).write_text(json.dumps(t, indent=1))
    print(json.dumps({k: t[k] for k in ("C", "sigma", "delta", "B", "L", "G", "H", "Q", "room")}, indent=1))


if __name__ == "__main__":
    main()
