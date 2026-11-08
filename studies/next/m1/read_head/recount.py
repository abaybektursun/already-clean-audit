"""Head's recount of the M1 verdict from the five counts.npz files. Own code; only numpy and the bank files."""
import json, sys, numpy as np, hashlib, pathlib
B = pathlib.Path("studies/next/m1/banks"); keys = ["start_a", "start_b", "G", "R", "M"]
z = {k: np.load(B / k / "m1" / "counts.npz") for k in keys}
dirs = z["start_a"]["dirs"]; mask = z["start_a"]["cm_mask"]
cm_names = [str(d) for d, m in zip(dirs, mask) if m]
print("code+math directories (mask):", cm_names)
assert len(cm_names) == 5 and all(("stack_edu-" in d) or d.endswith("finemath-3plus") for d in cm_names)
for k in keys:
    assert (z[k]["dirs"] == dirs).all() and (z[k]["cm_mask"] == mask).all()
sha = {k: hashlib.sha256((B / k / "m1" / "counts.npz").read_bytes()).hexdigest()[:8] for k in keys}
print("sha256 prefixes:", sha)
print("start_a == start_b counts and filled:", np.array_equal(z["start_a"]["counts"], z["start_b"]["counts"]), np.array_equal(z["start_a"]["filled"], z["start_b"]["filled"]))
def share(k, m=mask):
    c = z[k]["counts"]; f = z[k]["filled"].astype(np.float64)
    assert (f > 0).all() and (c.sum(-1) == z[k]["filled"]).all(), "counts do not sum to filled"
    return c[..., m].sum(-1) / f                       # [32, 11008, 2]
S = {k: share(k) for k in keys}
band = (np.arange(32) * 4) // 32
v = json.load(open("studies/next/m1/verdict_m1.json")); th = json.load(open("studies/next/m1/thresholds.json"))
bar = v["bar"]; kappa = th["kappa"]
print("\nC per checkpoint by band (mine | verdict):")
for k in keys:
    mine = [float(S[k][band == b].mean()) for b in range(4)]
    print(f"  {k:8s}", [round(x, 6) for x in mine], "| max|diff| vs verdict", f"{max(abs(a - c) for a, c in zip(mine, v['exploratory']['C'][k])):.1e}")
rng = np.random.default_rng(7)
def contrast(x, y, nboot=2000):
    d = (S[x] - S[y]).mean(-1)   # per neuron [32, 11008]
    out = []
    for b in range(4):
        vv = d[band == b].ravel(); n = vv.size
        idx = rng.integers(0, n, (nboot, n))
        means = vv[idx].mean(1)
        out.append((float(vv.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))))
    return out
res = {}
for name, (x, y) in {"MG": ("M", "G"), "RG": ("R", "G"), "MR": ("M", "R"), "G-start": ("G", "start_a"), "R-start": ("R", "start_a"), "M-start": ("M", "start_a")}.items():
    res[name] = contrast(x, y)
    print(f"\n{name}:")
    for b, (m, lo, hi) in enumerate(res[name]):
        print(f"  band {b}: mean {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  bar {bar[b]:.5f}  kappa {kappa[b]:.3f}")
def label_lead(m, lo, hi, b):
    return "ABOVE_CEILING" if m > kappa[b] else "SUPPORTED" if lo > bar[b] else "REFUTED" if hi < bar[b] else "UNDECIDED"
def label_between(g1, g2, b):
    return "SUPPORTED" if g1[1] > bar[b] and g2[1] > bar[b] else "REFUTED" if g1[2] < -bar[b] or g2[2] < -bar[b] else "UNDECIDED"
lead = [label_lead(*res["MG"][b], b) for b in range(4)]
between = [label_between(res["RG"][b], res["MR"][b], b) for b in range(4)]
def combine(L):
    return "ABOVE_CEILING" if "ABOVE_CEILING" in L else "SUPPORTED" if all(x == "SUPPORTED" for x in L) else "REFUTED" if "REFUTED" in L else "UNDECIDED"
print("\nlead labels", lead, "->", combine(lead), "| verdict", v["H_lead"])
print("between labels", between, "->", combine(between), "| verdict", v["H_order"])
print("lambda = RG/MG by band:", [round(res["RG"][b][0] / res["MG"][b][0], 2) for b in range(4)])
# G-ID: I(b) = C(start_b) - C(start_a); line = min tau / 4
I = [float((S["start_b"] - S["start_a"])[band == b].mean()) for b in range(4)]
print("G-ID I by band", I, "line", min(th["tau"]) / 4, "pass", max(abs(x) for x in I) <= min(th["tau"]) / 4)
# specificity: 1,000 random 5-sets of the other 45 directories, my own seed
others = np.where(~mask)[0]; rs = np.random.default_rng(11)
absd = np.zeros((1000, 4))
for i in range(1000):
    m5 = np.zeros(50, bool); m5[rs.choice(others, 5, replace=False)] = True
    d = (share("M", m5) - share("G", m5))
    absd[i] = [abs(float(d[band == b].mean())) for b in range(4)]
print("\nspecificity: code+math Delta_MG | p99 |random| (mine, verdict) | median (mine, verdict)")
for b in range(4):
    print(f"  band {b}: {res['MG'][b][0]:+.5f} | {np.quantile(absd[:, b], 0.99):.5f}, {v['specificity'][b]['random_abs_q99']:.5f} | {np.median(absd[:, b]):.5f}, {v['specificity'][b]['random_abs_median']:.5f}")
# per-directory contribution to Delta_MG (which of the five carries it)
print("\nper-directory Delta_MG contribution by band (slots share):")
for d in np.where(mask)[0]:
    m1 = np.zeros(50, bool); m1[d] = True
    dd = share("M", m1) - share("G", m1)
    print(f"  {str(dirs[d]):28s}", [f"{float(dd[band == b].mean()):+.5f}" for b in range(4)])
