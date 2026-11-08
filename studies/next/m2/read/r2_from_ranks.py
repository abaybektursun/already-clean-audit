"""Reader of M2, phase 1, step 2: every headline quantity from the job's per-record outputs.

Own code from the spec's definitions (M2_SPEC.md 0.5, 1.3, 3.3-3.5, 5, 6). Inputs: banks/score-06/m2/ranks_*.jsonl
(sha256 equal to the S3 copies under results/next_m2/score-06/out/score/m2/), births.json.
Output: read/r2_from_ranks.json.
"""
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"
RANKS = M2 / "banks/score-06/m2"
Z = 1.959963984540054
CHANCE = 10 / 22016
SCORES = ["A", "B", "ratio", "maxabs", "density", "colnorm"]


def wilson(h, n):
    p = h / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    w = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [float(c - w), float(c + w)]


def boot(x, seed=0, B=10000):
    x = np.asarray(x, float)
    n = len(x)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    m = x[idx].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def floor(matched_ranks, seed=0, R=1000, q=97.5):
    M = np.asarray(matched_ranks)  # births x 100
    n, k = M.shape
    draw = np.random.default_rng(seed).integers(0, k, size=(R, n))
    hits = (M[np.arange(n)[None, :], draw] <= 10).mean(1)
    return float(np.percentile(hits, q)), hits


def load(half):
    R = [json.loads(l) for l in (RANKS / f"ranks_{half}.jsonl").open()]
    out = {}
    for r in R:
        out[(r["birth"], r["kind"], r["role"])] = r
    return R, out


def main():
    d = json.loads((M2 / "births.json").read_text())
    births = {b["id"]: b for b in d["births"]}
    res = {"sha256": {h: hashlib.sha256((RANKS / f"ranks_{h}.jsonl").read_bytes()).hexdigest()
                      for h in ("freeze", "test")}}
    for half in ("freeze", "test"):
        R, rec = load(half)
        ids = [b for b in births if births[b]["half"] == half]
        cons = []  # consistency problems
        for r in R:
            b = births[r["birth"]]
            if r["half"] != half or r["layer"] != b["layer"]:
                cons.append([r["unit"], "half/layer"])
            tt = {"pre": b["k"] - 1, "birth": b["k"], "end": 15}[r["role"]]
            if r["t"] != tt:
                cons.append([r["unit"], "t", r["t"], tt])
            want = b["pole"] if r["kind"] == "real" else b["pb"]["pole"]
            if r["carrier"] != want:
                cons.append([r["unit"], "carrier"])
            for s in ("A", "B"):
                names = [x[0] for x in r["top10"][s]]
                vals = [x[1] for x in r["top10"][s]]
                if vals != sorted(vals, reverse=True):
                    cons.append([r["unit"], s, "top10 not sorted"])
                rk = r["rank"][s]
                v = r["value"][s]
                # rank = number of poles with score >= v: every listed pole with value >= v counts
                n_ge_listed = sum(1 for x in vals if x >= v)
                if rk <= 10 and r["carrier"] not in names:
                    cons.append([r["unit"], s, "hit but not in top10"])
                if rk < n_ge_listed:
                    cons.append([r["unit"], s, "rank below listed count"])
                if r["carrier"] in names and rk <= 10 and vals[names.index(r["carrier"])] != v:
                    cons.append([r["unit"], s, "value differs in list"])
        H = {"n_records": len(R), "consistency_problems": cons}
        pre = [rec[(i, "real", "pre")] for i in ids]
        H["n_pre"] = len(pre)
        for s in SCORES:
            h = int(sum(r["rank"][s] <= 10 for r in pre))
            H[f"pre_{s}"] = {"hits": h, "n": len(pre), "p": h / len(pre), "wilson": wilson(h, len(pre))}
        # floor
        for s in ("A", "B"):
            M = [r["matched_rank"][s] for r in pre]
            assert all(len(m) == 100 for m in M)
            f, hits = floor(M)
            H[f"floor_{s}"] = {"p97.5": f, "mean": float(hits.mean()), "max": float(hits.max()),
                               "matched_hit_share": float((np.asarray(M) <= 10).mean()),
                               "matched_carrier_neuron_present": int(sum(
                                   any(int(p[4:-1]) == births[r["birth"]]["neuron"] for p in r["matched"]) for r in pre)),
                               "matched_len_100": all(len(r["matched"]) == 100 for r in pre)}
        # birth role and end role
        for role in ("birth", "end"):
            rr = [rec[(i, "real", role)] for i in ids]
            for s in ("A", "B"):
                h = int(sum(r["rank"][s] <= 10 for r in rr))
                H[f"{role}_{s}"] = {"hits": h, "n": len(rr), "p": h / len(rr), "wilson": wilson(h, len(rr))}
            if role == "end":
                for s in ("A", "B"):
                    h = int(sum(r["rank_opposite"][s] <= 10 for r in rr))
                    H[f"end_opposite_{s}"] = {"hits": h, "n": len(rr), "p": h / len(rr),
                                              "max_rank_carrier": int(max(r["rank"][s] for r in rr)),
                                              "min_rank_opposite": int(min(r["rank_opposite"][s] for r in rr))}
        # twins
        tid = [i for i in ids if (i, "pb", "pre") in rec]
        H["n_twins_pre"] = len(tid)
        H["n_twins_end"] = sum((i, "pb", "end") in rec for i in ids)
        for s in ("A", "B"):
            ht = [int(rec[(i, "pb", "pre")]["rank"][s] <= 10) for i in tid]
            hr = [int(rec[(i, "real", "pre")]["rank"][s] <= 10) for i in tid]
            diff = np.array(hr) - np.array(ht)
            m, ci = boot(diff)
            lab = "ABOVE_SELECTION" if ci[0] > 0 else ("BELOW_SELECTION" if ci[1] < 0 else "AT_SELECTION")
            H[f"twin_{s}"] = {"twin_hits": int(sum(ht)), "real_hits_on_twinned": int(sum(hr)), "n": len(tid),
                              "twin_p": sum(ht) / len(tid), "twin_wilson": wilson(sum(ht), len(tid)),
                              "diff_mean": m, "diff_ci": ci, "label": lab,
                              "pairs_real1_twin0": int(sum((np.array(hr) == 1) & (np.array(ht) == 0))),
                              "pairs_real0_twin1": int(sum((np.array(hr) == 0) & (np.array(ht) == 1)))}
        # H_AB
        hA = np.array([int(r["rank"]["A"] <= 10) for r in pre])
        hB = np.array([int(r["rank"]["B"] <= 10) for r in pre])
        m, ci = boot(hA - hB)
        H["H_AB"] = {"diff_mean": m, "diff_ci": ci,
                     "label": "BEATS" if ci[0] > 0 else ("LOSES" if ci[1] < 0 else "TIES"),
                     "A1B0": int(sum((hA == 1) & (hB == 0))), "A0B1": int(sum((hA == 0) & (hB == 1)))}
        H["pre_ranks"] = {i: {"A": rec[(i, "real", "pre")]["rank"]["A"], "B": rec[(i, "real", "pre")]["rank"]["B"],
                              "B_value": rec[(i, "real", "pre")]["value"]["B"],
                              "twin_A": rec[(i, "pb", "pre")]["rank"]["A"] if (i, "pb", "pre") in rec else None,
                              "twin_B": rec[(i, "pb", "pre")]["rank"]["B"] if (i, "pb", "pre") in rec else None,
                              "twin_B_value": rec[(i, "pb", "pre")]["value"]["B"] if (i, "pb", "pre") in rec else None}
                          for i in ids}
        res[half] = H
    # bars and labels (test half decides)
    lab = {}
    for s in ("A", "B"):
        bar = max(CHANCE, res["freeze"][f"floor_{s}"]["p97.5"], res["test"][f"floor_{s}"]["p97.5"])
        t = res["test"][f"pre_{s}"]
        if t["wilson"][0] > bar:
            L = "SUPPORTED"
        elif t["p"] <= bar:
            L = "REFUTED"
        else:
            L = "UNDECIDED"
        ceil_up = res["test"][f"birth_{s}"]["wilson"][1]
        if t["wilson"][0] > ceil_up:
            L = "ABOVE_CEILING"
        lab[f"H_{s}"] = {"label": L, "bar": bar, "lower": t["wilson"][0], "birth_upper": ceil_up}
    lab["H_sel_B"] = res["test"]["twin_B"]["label"]
    lab["H_sel_A"] = res["test"]["twin_A"]["label"]
    lab["H_AB"] = res["test"]["H_AB"]["label"]
    cc = {h: all(res[h][f"end_{s}"]["p"] >= 0.90 for s in ("A", "B")) and
          all(res[h][f"end_opposite_{s}"]["p"] < 0.90 for s in ("A", "B")) for h in ("freeze", "test")}
    lab["construction_control_fires"] = cc
    lab["chance"] = CHANCE
    res["labels"] = lab
    (M2 / "read/r2_from_ranks.json").write_text(json.dumps(res, indent=1))
    show = {h: {k: v for k, v in res[h].items() if k != "pre_ranks"} for h in ("freeze", "test")}
    print(json.dumps(show, indent=1))
    print(json.dumps(lab, indent=1))


if __name__ == "__main__":
    main()
