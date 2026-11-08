"""Reader of M2, phase 3: the twins' seeded draws, re-run with the procedure read from m2.pseudo_birth (candidates:
sample-0 clustered poles of the layer with another neuron, sorted by id, permuted by default_rng([20261105, l, n, s,
c]); per candidate 2,000 groups drawn at once, sorted; the first dated k). Phase 1 (r1) checked the twins' properties
only; its own guess at the draw order reproduced 33 of 153. Own dating code (r1_births.date). Output: r9_twin_draws.json.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r1_births import ARR, CLU, date

M2 = Path(__file__).resolve().parents[1]


def main():
    z = np.load(ARR, allow_pickle=False)
    A, Q, P = z["a"], z["q999"], [str(p) for p in z["poles"]]
    row = {p: i for i, p in enumerate(P)}
    s0 = {c["pole"]: c["clusters"] for c in (json.loads(l) for l in CLU.open()) if c["sample"] == 0}
    ne = {p: sorted(r for x in cl if not x["echo"] for r in x["exemplar_ranks"]) for p, cl in s0.items()}
    d = json.loads((M2 / "births.json").read_text())
    ok, bad = 0, []
    for b in d["births"]:
        rng = np.random.default_rng([20261105, b["layer"], b["neuron"], b["sign"], b["c"]])
        cands = sorted(q for q in s0 if int(q[1:3]) == b["layer"] and int(q[4:-1]) != b["neuron"])
        got = None
        for q in rng.permutation(np.array(cands, dtype=object)):
            r = np.array(ne[q])
            if r.size < b["size"]:
                continue
            g = np.sort(np.stack([rng.choice(r, b["size"], replace=False) for _ in range(2000)]), axis=1)
            sg = 0 if q.endswith("+") else 1
            hit = [j for j in range(2000) if date(A, Q, row[q], g[j], sg)[0] == b["k"]][:1]
            if hit:
                got = [q, g[hit[0]].tolist(), hit[0] + 1]
                break
        bank = [b["pb"]["pole"], b["pb"]["ranks"], b["pb"]["draws_used"]] if b.get("pb") else None
        if got == bank:
            ok += 1
        else:
            bad.append([b["id"], got, bank])
    out = {"n": len(d["births"]), "reproduced": ok, "differ": bad}
    (M2 / "read/r9_twin_draws.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
