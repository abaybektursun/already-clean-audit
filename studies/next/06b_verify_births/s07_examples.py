"""Two banked examples (first passing and first failing late_primary job in cluster-id order): name, size, real
birth, the random same-size births of its slot (set A), p, judge flag. Output: out/s07_examples.txt."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
R = np.load(OUT / "random_groups.npz", allow_pickle=False)
cl = list(R["cluster"])
pc = json.load(open(OUT / "s04_per_cluster_p.json"))["per_cluster"]
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json"))}
clusters = json.load(open(OUT / "clusters.json"))
prim = sorted(c for c, v in pc.items() if v["status"] == "late_primary")
L = []
for which, c in (("passing", next(c for c in prim if pc[c]["p"] < 0.05)), ("failing", next(c for c in prim if pc[c]["p"] >= 0.05))):
    pole, ci = c.rsplit(":c", 1)
    info = next(x for x in clusters[pole]["0"] if x["c"] == int(ci))
    j = cl.index(c)
    rb = np.bincount(R["A_bp"][:, j], minlength=16)
    b = banked[c]
    L.append(f"{which}: {c} '{info['name']}', size {len(info['exemplar_ranks'])}, real birth T_{b['birth_pct']}, "
             f"p {pc[c]['p']:.4f} ({pc[c]['n_random_ge']} of 300 random groups born at or after it); "
             f"random-group births by point T_0..T_15: {rb.tolist()}; judge acc_random {b['acc_random']}, flagged {b['flagged']}; "
             f"real med (a+) T_0..T_15: {[round(x, 3) for x in b['med']]}")
(OUT / "s07_examples.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
