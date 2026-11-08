"""Room check for the amended PC-ART (N13_SPEC.md "Amendments before freeze 1, 2026-11-05", row A2). Weight-free: random stand-in bf16
weights. Run from the n13 worktree root: PYTHONPATH=$PWD <n13 venv python> studies/next/n13/amendments/pcart_room.py <out.json>
(the venv of N13_OFFLINE_GATE.md "How to rerun"). Output of 2026-11-05: pcart_room.json beside this file.
P+ = the §5 synthetic one-step change (p 0.3); P- = P+ plus 1% of neurons moved 8 bf16 steps on every entry (random sign).
The H_artifact rule must give SUPPORTED on P+ and REFUTED on P-. Also J_sp: Jaccard of S_k by s with S_k by the spacing-only score."""
import json, sys
import numpy as np
from infra.pg import n13_arms as A

out_path = sys.argv[1]
L, H, W = 4, 1024, 1024          # 4,096 neurons, 3,072 entries per neuron
cfg = {"num_hidden_layers": L, "hidden_size": H, "intermediate_size": W}
NN = L * W
ks = [int(round(f * NN)) for f in (0.001, 0.003, 0.01, 0.03)]

class D:
    def __init__(self, t): self.t = t
    def names(self): return sorted(self.t)
    def get(self, n): return self.t[n]

def base(seed, hetero):
    rng = np.random.default_rng(seed)
    t = {}
    for l in range(L):
        for w, m in enumerate(A.MLP):
            shape = (W, H) if w < 2 else (H, W)
            x = 0.02 * rng.standard_normal(shape)
            if hetero:
                sc = np.exp(0.5 * rng.standard_normal(W))
                x = x * (sc[:, None] if w < 2 else sc[None, :])
            t[f"model.layers.{l}.{m}.weight"] = A.bf16(x)
    return D(t)

def rule(jf, jr):
    if jf >= 0.7 and jr >= 0.7: return "REFUTED"
    if jf < 0.4 or jr < 0.4: return "SUPPORTED"
    return "INDETERMINATE"

def measure(b, a, sp2):
    s2, s2f, _ = A.weight_stats(b, a, cfg)
    o, of, osp = A.order(s2), A.order(s2f), A.order(sp2)
    jf = {t: A.jaccard(o[:k], of[:k]) for t, k in zip(A.K_TAGS, ks)}
    jr = A.j_round(b, a, cfg, s2, ks)["J_round"]
    jsp = {t: A.jaccard(o[:k], osp[:k]) for t, k in zip(A.K_TAGS, ks)}
    t1 = A.K_TAGS[A.PRIMARY]
    return {"J_floor": jf, "J_round": jr, "J_sp": jsp, "rule_k1": rule(jf[t1], jr[t1])}

res = {"shape": {"layers": L, "hidden": H, "mlp_width": W, "neurons": NN, "entries_per_neuron": 3 * H}, "ks": ks}
for label, het in (("iid", False), ("neuron_scale_lognormal_0.5", True)):
    b = base(21, het)
    sp2 = np.zeros((L, W))
    for l in range(L):
        sp2[l] = A.neuron_sq(*(A.spacing(x) for x in A.layer_triplet(b, l)))
    syn = A.synthetic_rounding(b, cfg)
    plus = A.Overlay(b, syn)
    k1 = ks[A.PRIMARY]
    planted = np.sort(np.random.default_rng(41).choice(NN, k1, replace=False))
    minus_t = {n: v.copy() for n, v in syn.items()}
    rng = np.random.default_rng(42)
    for f in planted:
        l, n = divmod(int(f), W)
        for w, m in enumerate(A.MLP):
            key = f"model.layers.{l}.{m}.weight"
            bt = b.get(key)
            if w < 2:
                sg = np.where(rng.random(H) < 0.5, -1, 1)
                minus_t[key][n, :] = A.from_ordered(A.ordered(bt[n, :]) + 8 * sg)
            else:
                sg = np.where(rng.random(H) < 0.5, -1, 1)
                minus_t[key][:, n] = A.from_ordered(A.ordered(bt[:, n]) + 8 * sg)
    minus = A.Overlay(b, minus_t)
    res[label] = {"P_plus": measure(b, plus, sp2), "P_minus": measure(b, minus, sp2), "n_planted": int(k1)}
    print(label, json.dumps({k: res[label][k]["rule_k1"] for k in ("P_plus", "P_minus")}), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
