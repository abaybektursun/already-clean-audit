"""Step 2a. Random regroupings. For each pole, the non-echo top-50 positions (all T_15 ranks outside the echo
cluster; the clusters partition ranks 0..49 exactly) are shuffled and cut into groups of the real non-echo cluster
sizes, in the real clusters' order (slot s has the size of real cluster s). Every group gets the rule of
s01 (percentile and raw births, main = largest T_15 a+, late flags, B4 match against sample-1 clusters).
Two independent sets of 300 draws: A (seed 20261105) is the reference, B (seed 20261106) is for noise floors.
Output: out/random_groups.npz."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
z = np.load(OUT / "inputs.npz", allow_pickle=False)
poles, A, Q999, MAX = list(z["poles"]), z["a"], z["q999"], z["max_abs"]
row = {p: i for i, p in enumerate(poles)}
clusters = json.load(open(OUT / "clusters.json"))
D, T = 300, 16


def births(present):
    """present [16, D] -> first t with present[t] and (t == 15 or present[t+1]); -1 if never."""
    ok = present & np.vstack([present[1:], np.ones((1, present.shape[1]), bool)])
    return np.where(ok.any(0), ok.argmax(0), -1)


def run(seed, identity=False):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("bp", "br", "main", "late_pct", "late_raw", "matched")}
    meta = []
    for pole in sorted(clusters):
        i = row[pole]
        s = 1.0 if pole.endswith("+") else -1.0
        mag = np.maximum(s * A[:, i, :], 0.0)
        jobs = [c for c in clusters[pole]["0"] if not c["echo"]]
        assert [c["c"] for c in jobs] == sorted(c["c"] for c in jobs)
        others = [set(c["exemplar_ranks"]) for c in clusters[pole]["1"] if not c["echo"]]
        pos = np.array(sorted(r for c in jobs for r in c["exemplar_ranks"]))
        sizes = [len(c["exemplar_ranks"]) for c in jobs]
        if identity:    # identity check: the real clusters themselves, in slot order
            perm = np.tile(np.array([r for c in jobs for r in c["exemplar_ranks"]]), (D, 1))
        else:
            perm = rng.permuted(np.tile(pos, (D, 1)), axis=1)                    # [D, P]
        bp, br, peak, matched = [], [], [], []
        o = 0
        for k in sizes:
            idx = perm[:, o:o + k]                                             # [D, k]
            o += k
            sub = mag[:, idx]                                                  # [16, D, k]
            med = np.median(sub, axis=2)                                       # [16, D]
            bp.append(births((med > 0) & (med >= Q999[:, i, None])))
            br.append(births((med > 0) & (med >= 0.3 * MAX[:, i, None])))
            peak.append(sub[15].max(1))
            matched.append([any(2 * len(set(g) & oo) > k for oo in others) for g in idx.tolist()])
        bp, br, peak, matched = (np.array(x).T for x in (bp, br, peak, matched))   # [D, m]
        assert (bp >= 0).all()          # every group is present at T_15 (top-50 positions)
        main = np.zeros_like(bp, bool)
        main[np.arange(D), peak.argmax(1)] = True                             # ties -> lowest slot = lowest c
        late_pct = bp > bp.min(1, keepdims=True)
        brm = np.where(br >= 0, br, 99)
        late_raw = (br >= 0) & (br > brm.min(1, keepdims=True))
        for k_, v in (("bp", bp), ("br", br), ("main", main), ("late_pct", late_pct), ("late_raw", late_raw),
                      ("matched", matched)):
            cols[k_].append(v)
        meta += [(f"{pole}:c{c['c']}", pole, len(c["exemplar_ranks"])) for c in jobs]
    return {k: np.concatenate(v, axis=1) for k, v in cols.items()}, meta


# Identity check: with the real grouping in place of a shuffle, the generator must give the banked values.
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
cols, meta = run(0, identity=True)
key = {"bp": "birth_pct", "main": "is_main", "late_pct": "late_pct", "late_raw": "late_raw", "matched": "matched"}
idm = {k: int(sum(cols[k][0, j] != banked[m[0]][key[k]] for j, m in enumerate(meta))) for k in key}
idm["br"] = int(sum((None if cols["br"][0, j] < 0 else cols["br"][0, j]) != banked[m[0]]["birth_raw"] for j, m in enumerate(meta)))
print("identity check, mismatches vs banked:", idm)
(OUT / "s02_identity.txt").write_text(f"identity grouping through the random-group code, mismatches vs q2_clusters.jsonl over {len(meta)} non-echo clusters: {idm}\n")

res = {}
for name, seed in (("A", 20261105), ("B", 20261106)):
    cols, meta = run(seed)
    res.update({f"{name}_{k}": v for k, v in cols.items()})
np.savez(OUT / "random_groups.npz", cluster=np.array([m[0] for m in meta]), pole=np.array([m[1] for m in meta]),
         size=np.array([m[2] for m in meta]), **res)
print({k: v.shape for k, v in res.items()})
