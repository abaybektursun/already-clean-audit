"""Step 4. b (share of poles with >= 2 born non-echo clusters whose main has the earliest birth, ties count as
earliest; neurons_verdict.birth_order_test) under random regrouping, beside the registered reference
(births permuted within the pole, 10,000 permutations, seed 0), recomputed with the repo function.
Inputs: out/q2_banked.json, out/random_groups.npz, out/clusters.json, out/inputs.npz.
Outputs: out/s05_b_random.json, out/s05_b_random.txt."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/abaybektursun/projects/interp_research")
import infra.pg.neurons_verdict as V

OUT = Path(__file__).resolve().parent / "out"
R = np.load(OUT / "random_groups.npz", allow_pickle=False)
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
cl, pole = list(R["cluster"]), np.array(R["pole"])
upoles = sorted(set(pole))
slots = {p: np.where(pole == p)[0] for p in upoles}


def b_of(bp, main):
    hits = []
    for p in upoles:
        j = slots[p]
        if len(j) < 2:                      # all groups are born (s02 asserts it), so born count = m
            continue
        hits.append(bp[j][main[j]][0] == bp[j].min())
    return float(np.mean(hits)), len(hits)


real_bp = np.array([banked[c]["birth_pct"] for c in cl])
real_main = np.array([banked[c]["is_main"] for c in cl])
b_real, n_real = b_of(real_bp, real_main)
bA = np.array([b_of(R["A_bp"][d], R["A_main"][d])[0] for d in range(R["A_bp"].shape[0])])
bB = np.array([b_of(R["B_bp"][d], R["B_main"][d])[0] for d in range(R["B_bp"].shape[0])])
order = []
for p in upoles:
    j = slots[p]
    if len(j) >= 2:
        order.append((real_bp[j].tolist(), int(np.where(real_main[j])[0][0])))
reg = V.birth_order_test(order, V.N_PERM, 0)
perm_mean_exact = float(np.mean([np.mean(np.array(b) == min(b)) for b, _ in order]))
# main's size against the other clusters of its pole (what the regrouping holds fixed and the permutation does not)
size = np.array(R["size"])
main_larger = np.mean([size[slots[p]][real_main[slots[p]]][0] >= np.median(size[slots[p]]) for p in upoles if len(slots[p]) >= 2])
res = {"b_real": b_real, "n_poles": n_real,
       "b_random_A": {"mean": bA.mean(), "p2_5": float(np.percentile(bA, 2.5)), "p97_5": float(np.percentile(bA, 97.5)),
                      "min": bA.min(), "max": bA.max(), "share_le_real": float((bA <= b_real).mean())},
       "b_random_B": {"mean": bB.mean(), "p2_5": float(np.percentile(bB, 2.5)), "p97_5": float(np.percentile(bB, 97.5)),
                      "share_le_real": float((bB <= b_real).mean())},
       "p_one_sided_real_below_random": float((1 + (np.r_[bA, bB] <= b_real).sum()) / (1 + len(bA) + len(bB))),
       "registered_permutation_reference_recomputed": {k: reg[k] for k in ("b", "ci95", "n_poles", "null_mean", "null_97_5", "p", "supports_main_oldest")},
       "permutation_mean_exact": perm_mean_exact,
       "share_poles_main_size_ge_pole_median": float(main_larger)}
json.dump(res, open(OUT / "s05_b_random.json", "w"), indent=1, default=float)
L = [f"b real {b_real:.4f} on {n_real} poles (bootstrap 95% {reg['ci95']})",
     f"b random regrouping, set A (300 draws): mean {bA.mean():.4f}, 2.5th {np.percentile(bA,2.5):.4f}, 97.5th {np.percentile(bA,97.5):.4f}, min {bA.min():.4f}, max {bA.max():.4f}; draws <= real: {(bA<=b_real).sum()}/300",
     f"b random regrouping, set B (300 draws): mean {bB.mean():.4f}, 2.5th {np.percentile(bB,2.5):.4f}, 97.5th {np.percentile(bB,97.5):.4f}; draws <= real: {(bB<=b_real).sum()}/300",
     f"one-sided p (real at or below random, 600 draws): {res['p_one_sided_real_below_random']:.4f}",
     f"registered reference, births permuted within pole (repo birth_order_test, 10,000, seed 0): mean {reg['null_mean']:.4f}, 97.5th {reg['null_97_5']:.4f}, p {reg['p']:.5f}; exact mean {perm_mean_exact:.4f}",
     f"share of poles whose main is at least the pole's median cluster size: {main_larger:.3f}"]
(OUT / "s05_b_random.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
