"""Step 4, extra variant (not in the review). Random regrouping in which the main's size is also held fixed: the
pole's top non-echo position (largest T_15 a+) is put in the real main's slot and the other positions are shuffled
into the remaining places. In s02 the top position falls in a slot chosen in proportion to slot size, so there the
random main's size differs from the real main's. 300 draws, seed 20261107.
Second variant (size only): all positions shuffled, and "main" is the real main's slot whatever lands in it, so it
holds the main's size fixed without giving it the top position. 300 draws, seed 20261108.
Outputs: out/s05b_b_main_slot_fixed.txt, out/s05b_b_main_slot_fixed.json."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
z = np.load(OUT / "inputs.npz", allow_pickle=False)
poles, A, Q999 = list(z["poles"]), z["a"], z["q999"]
row = {p: i for i, p in enumerate(poles)}
clusters = json.load(open(OUT / "clusters.json"))
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
D = 300
rng = np.random.default_rng(20261107)


def births(present):
    ok = present & np.vstack([present[1:], np.ones((1, present.shape[1]), bool)])
    return np.where(ok.any(0), ok.argmax(0), -1)


rng2 = np.random.default_rng(20261108)
hits, hits_size = [], []
n_top_check = 0
for pole in sorted(clusters):
    jobs = [c for c in clusters[pole]["0"] if not c["echo"]]
    if len(jobs) < 2:
        continue
    i = row[pole]
    mag = np.maximum((1.0 if pole.endswith("+") else -1.0) * A[:, i, :], 0.0)
    pos = np.array(sorted(r for c in jobs for r in c["exemplar_ranks"]))
    top = pos[np.argmax(mag[15, pos])]
    mi = [banked[f"{pole}:c{c['c']}"]["is_main"] for c in jobs].index(True)
    n_top_check += top in jobs[mi]["exemplar_ranks"]
    rest = pos[pos != top]
    perm = rng.permuted(np.tile(rest, (D, 1)), axis=1)
    sizes = [len(c["exemplar_ranks"]) for c in jobs]
    bp, o = [], 0
    for s, k in enumerate(sizes):
        if s == mi:
            idx = np.hstack([np.full((D, 1), top), perm[:, o:o + k - 1]]); o += k - 1
        else:
            idx = perm[:, o:o + k]; o += k
        med = np.median(mag[:, idx], axis=2)
        bp.append(births((med > 0) & (med >= Q999[:, i, None])))
    bp = np.array(bp).T                                  # [D, m]
    hits.append(bp[:, mi] == bp.min(1))
    perm2 = rng2.permuted(np.tile(pos, (D, 1)), axis=1)
    bp2, o = [], 0
    for k in sizes:
        med = np.median(mag[:, perm2[:, o:o + k]], axis=2); o += k
        bp2.append(births((med > 0) & (med >= Q999[:, i, None])))
    bp2 = np.array(bp2).T
    hits_size.append(bp2[:, mi] == bp2.min(1))
hits = np.array(hits, float)                            # [poles, D]
b = hits.mean(0)
b2 = np.array(hits_size, float).mean(0)
res = {"size_only": {"b_mean": b2.mean(), "p2_5": float(np.percentile(b2, 2.5)), "p97_5": float(np.percentile(b2, 97.5)),
                     "draws_le_real_0.4756": int((b2 <= 0.47560975609756095).sum())},
       "n_poles": hits.shape[0], "top_position_in_real_main": n_top_check, "b_mean": b.mean(),
       "p2_5": float(np.percentile(b, 2.5)), "p97_5": float(np.percentile(b, 97.5)),
       "draws_le_real_0.4756": int((b <= 0.47560975609756095).sum())}
json.dump(res, open(OUT / "s05b_b_main_slot_fixed.json", "w"), indent=1, default=float)
txt = (f"main slot fixed (top position in the real main's slot, sizes fixed): b mean {b.mean():.4f}, 2.5th {res['p2_5']:.4f}, "
       f"97.5th {res['p97_5']:.4f}, draws <= real 0.4756: {res['draws_le_real_0.4756']}/300; poles {hits.shape[0]}; "
       f"real main holds the pole's top non-echo position in {n_top_check} of {hits.shape[0]} poles")
txt += (f"\nsize only (real main's slot, random content, top position not forced): b mean {b2.mean():.4f}, "
        f"2.5th {res['size_only']['p2_5']:.4f}, 97.5th {res['size_only']['p97_5']:.4f}, draws <= real 0.4756: {res['size_only']['draws_le_real_0.4756']}/300")
(OUT / "s05b_b_main_slot_fixed.txt").write_text(txt + "\n")
print(txt)
