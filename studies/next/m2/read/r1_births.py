"""Reader of M2, phase 1, step 1: births, split, dating rule, twins' properties.

Own code from the spec's definitions (M2_SPEC.md 0.5, 3.3, 6). Inputs: the laptop banks named in births.json
-> inputs, and births.json itself. Output: read/r1_births.json.
"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"
BANK = Path("/Users/abaybektursun/projects/interp_research_banks")
ARR = BANK / "neurons/next/recount2/arrays.npz"
CLU = BANK / "neurons/w4/o7_s_v2/explain/clusters.jsonl"
RBI = BANK / "neurons/next/recount2/births.jsonl"
PCL = ROOT / "studies/next/10_recount2/out/05_per_cluster_R1t.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def aplus(a, sign):
    return np.maximum(a, 0.0) if sign == 0 else np.maximum(-a, 0.0)


def date(A, Q, row, ranks, sign):
    """first t with median a+ >= q999 at t and t+1; T_15 counts when present there (no next point)."""
    med = np.array([np.median(aplus(A[t, row, ranks], sign)) for t in range(16)])
    pres = med >= Q[:, row]
    for t in range(15):
        if pres[t] and pres[t + 1]:
            return t, med
    if pres[15]:
        return 15, med
    return None, med


def main():
    d = json.loads((M2 / "births.json").read_text())
    ins = d["inputs"]
    shas = {"arrays": sha(ARR), "clusters": sha(CLU), "births": sha(RBI), "per_cluster": sha(PCL)}
    reg = {k: ins[k]["sha256"] for k in shas}
    z = np.load(ARR, allow_pickle=False)
    A, Q, P, tok15 = z["a"], z["q999"], [str(p) for p in z["poles"]], z["tok15"]
    row_of = {p: i for i, p in enumerate(P)}
    sign_of = lambda p: 0 if p.endswith("+") else 1

    # sample-0 non-echo clusters
    clus = [json.loads(l) for l in CLU.open()]
    s0 = {c["pole"]: c for c in clus if c["sample"] == 0}
    nonecho = []
    for pole, c in s0.items():
        for k in c["clusters"]:
            if not k["echo"]:
                nonecho.append((pole, k["c"], sorted(k["exemplar_ranks"])))
    rb = {(b["pole"], b["c"]): b for b in (json.loads(l) for l in RBI.open()) if b["sample"] == 0}

    agree, disagree, none_ct = 0, [], 0
    mydate = {}
    for pole, c, ranks in nonecho:
        k, med = date(A, Q, row_of[pole], ranks, sign_of(pole))
        mydate[(pole, c)] = k
        bk = rb[(pole, c)]["birth_pct"]
        if k is None:
            none_ct += 1
        if k == bk:
            agree += 1
        else:
            disagree.append([pole, c, k, bk])

    # J1: p_late <= 0.05
    rows = list(csv.DictReader(PCL.open()))
    passing = [(r["pole"], int(r["c"])) for r in rows if float(r["p_late"]) <= 0.05]
    pass_set = set(passing)

    B = d["births"]
    ids_json = {(b["pole"], b["c"]) for b in B}
    checks = {}
    checks["n_births"] = len(B)
    checks["passing_p_late_le_0.05"] = len(passing)
    checks["births_json_eq_passing"] = ids_json == pass_set
    # k, size, tok, half
    bad_k, bad_tok, bad_half, bad_size = [], [], [], []
    for b in B:
        p = b["pole"]
        r = row_of[p]
        if r != b["row"]:
            bad_tok.append([b["id"], "row"])
        k = mydate[(p, b["c"])]
        if k != b["k"]:
            bad_k.append([b["id"], k, b["k"]])
        ex = s0[p]["clusters"]
        ranks = sorted([x for x in ex if x["c"] == b["c"]][0]["exemplar_ranks"])
        if ranks != sorted(b["ranks"]) or len(ranks) != b["size"]:
            bad_size.append(b["id"])
        if [int(x) for x in tok15[r, b["ranks"]]] != b["tok"]:
            bad_tok.append([b["id"], "tok"])
        h = "freeze" if b["neuron"] % 2 == 0 else "test"
        if h != b["half"]:
            bad_half.append(b["id"])
    checks["k_mismatch"] = bad_k
    checks["tok_or_row_mismatch"] = bad_tok
    checks["ranks_or_size_mismatch"] = bad_size
    checks["half_mismatch"] = bad_half
    halves = Counter(b["half"] for b in B)
    by_k = {h: dict(sorted(Counter(b["k"] for b in B if b["half"] == h).items())) for h in ("freeze", "test")}
    poles_by_half = {h: {b["pole"] for b in B if b["half"] == h} for h in ("freeze", "test")}
    neur_by_half = {h: {(b["layer"], b["neuron"]) for b in B if b["half"] == h} for h in ("freeze", "test")}

    # twins
    tw = {"freeze": 0, "test": 0}
    tw_bad = []
    draws = []
    for b in B:
        pb = b.get("pb")
        if not pb:
            continue
        tw[b["half"]] += 1
        p = pb["pole"]
        r = row_of[p]
        errs = []
        if r != pb["row"]:
            errs.append("row")
        if pb["layer"] != b["layer"]:
            errs.append("layer")
        if pb["neuron"] == b["neuron"]:
            errs.append("same neuron")
        if len(pb["ranks"]) != b["size"]:
            errs.append("size")
        if [int(x) for x in tok15[r, pb["ranks"]]] != pb["tok"]:
            errs.append("tok")
        k, _ = date(A, Q, r, pb["ranks"], pb["sign"])
        if k != b["k"]:
            errs.append(f"date {k} != {b['k']}")
        ne = set()
        if p in s0:
            for x in s0[p]["clusters"]:
                if not x["echo"]:
                    ne |= set(x["exemplar_ranks"])
        else:
            errs.append("twin pole has no sample-0 clustering")
        if not set(pb["ranks"]) <= ne:
            errs.append("rank in an echo cluster")
        if errs:
            tw_bad.append([b["id"], errs])
        draws.append(pb.get("draws_used"))
    no_twin = [b["id"] for b in B if not b.get("pb")]

    out = {
        "inputs_sha256_match_registered": {k: shas[k] == reg[k] for k in shas},
        "dating": {"n_sample0_nonecho_clusters": len(nonecho), "agree_with_recount2_birth_pct": agree,
                   "disagree": disagree[:20], "n_disagree": len(disagree), "n_undated": none_ct},
        "n_clusters_tested_csv": len(rows),
        "checks": checks,
        "halves": dict(halves), "by_k": by_k,
        "n_carriers": len({b["pole"] for b in B}), "n_positions": sum(b["size"] for b in B),
        "carriers_by_half": {h: len(v) for h, v in poles_by_half.items()},
        "halves_share_a_neuron": bool(neur_by_half["freeze"] & neur_by_half["test"]),
        "twins": tw, "twins_bad": tw_bad, "no_twin": no_twin,
        "twin_draws_used_quantiles": [float(x) for x in np.quantile([x for x in draws if x is not None],
                                                                     [0, .5, .9, 1])] if draws else None,
    }
    (M2 / "read/r1_births.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
