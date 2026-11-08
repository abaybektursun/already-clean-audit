"""Dated exploratory analysis (2026-11-05) on the closed, unblinded M1 result; no rule here decides a headline.
Step 2 of the pole-turnover analysis: freeze tau and the floor on the stage-1 references only.
This file reads only the five stage-1 reference count files; it never opens start_a, start_b, G, R or M.
Run from the repo root: python3 studies/next/m1/turnover/rule.py  ->  studies/next/m1/turnover/rule.json
"""
import json, pathlib, numpy as np

B = pathlib.Path("studies/next/m1/banks")
OUT = pathlib.Path("studies/next/m1/turnover")
CM = ["data/finemath-3plus", "data/stack_edu-CSharp", "data/stack_edu-JavaScript", "data/stack_edu-Python", "data/stack_edu-SQL"]
band = (np.arange(32) * 4) // 32          # layers 0-7, 8-15, 16-23, 24-31


def path(key):
    return B / key / "m1" / "counts.npz" if key in ("start_a", "start_b", "G", "R", "M") else B / "refs" / key / "m1" / "counts.npz"


def share_from_counts(c, f, mask):
    """cm_share per pole as in read_head/recount.py: slots in the code+math directories / filled slots. [32, 11008, 2]"""
    assert (f > 0).all() and (c.sum(-1) == f).all(), "counts do not sum to filled"
    return c[..., mask].sum(-1) / f.astype(np.float64)


def load(key):
    z = np.load(path(key))
    mask = z["cm_mask"]
    assert [str(d) for d, m in zip(z["dirs"], mask) if m] == CM, key
    return z["counts"], z["filled"], mask


def share(key):
    return share_from_counts(*load(key))


def per_neuron(SX, SY, tau, b):
    """Per neuron of band b (both poles summed): H = poles at or above tau at X, L = leavers, A = arrivals,
    dH, dR = sums of the per-pole difference Y - X over poles at or above tau at X, and over the rest."""
    x, y = SX[band == b], SY[band == b]   # [8, 11008, 2]
    hx, hy = x >= tau, y >= tau
    d = y - x
    return {"H": hx.sum(-1).ravel(), "Hy": hy.sum(-1).ravel(), "L": (hx & ~hy).sum(-1).ravel(), "A": (~hx & hy).sum(-1).ravel(),
            "dH": np.where(hx, d, 0.0).sum(-1).ravel(), "dR": np.where(~hx, d, 0.0).sum(-1).ravel()}


def point(SX, SY, tau, b):
    q = per_neuron(SX, SY, tau, b)
    nH, nHy, N = int(q["H"].sum()), int(q["Hy"].sum()), 2 * q["H"].size
    return {"leavers": float(q["L"].sum() / nH), "arrivals": float(q["A"].sum() / nH), "n_high_X": nH, "n_high_Y": nHy, "n_poles": N,
            # ceiling: the expected value if pole identity were scrambled within the band between X and Y
            "shuffle_leavers": 1 - nHy / N, "shuffle_arrivals": (N - nH) * (nHy / N) / nH}


if __name__ == "__main__":
    REFS = ["t12", "t13", "t14b", "t14c", "t15"]
    S = {k: share(k) for k in REFS}
    t15 = S["t15"]
    tau = [float(np.quantile(t15[band == b], 0.9)) for b in range(4)]
    frac_ge = [float((t15[band == b] >= tau[b]).mean()) for b in range(4)]
    pairs = [("t12", "t13"), ("t13", "t14b"), ("t14b", "t15"), ("t14b", "t14c")]
    info_pairs = [("t14c", "t15")]   # the spec's third drift pair (M1_SPEC 3.2); reported, not in the floor
    per_pair = {f"{x}->{y}": [point(S[x], S[y], tau[b], b) for b in range(4)] for x, y in pairs + info_pairs}
    floor_L = [max(per_pair[f"{x}->{y}"][b]["leavers"] for x, y in pairs) for b in range(4)]
    floor_A = [max(per_pair[f"{x}->{y}"][b]["arrivals"] for x, y in pairs) for b in range(4)]
    rule = {
        "label": "Dated exploratory analysis, 2026-11-05, on the closed and unblinded M1 result. No rule here decides a headline. Frozen on stage-1 references before start_a, start_b, G, R, M were read for this analysis.",
        "share": "cm_share(p) = slots of pole p's top-50 in the five code+math directories / filled slots (read_head/recount.py)",
        "bands": "layers 0-7, 8-15, 16-23, 24-31; 176,128 poles each",
        "tau": tau,
        "tau_definition": "per band, np.quantile(cm_share at t15 over the band's 176,128 poles, 0.9), numpy default linear method",
        "fraction_of_t15_poles_at_or_above_tau": frac_ge,
        "leavers": "poles with share >= tau at X and < tau at Y, divided by the number of poles with share >= tau at X",
        "arrivals": "poles with share < tau at X and >= tau at Y, divided by the number of poles with share >= tau at X",
        "floor_pairs": [f"{x}->{y}" for x, y in pairs],
        "floor_leavers": floor_L,
        "floor_arrivals": floor_A,
        "floor_definition": "per band, the largest point value over the four floor pairs (three stage-1 neighbour pairs and the identity pair t14b->t14c)",
        "ceiling": "per pair and band, the value if pole identity were scrambled within the band: leavers 1 - n_high_Y/N, arrivals (N - n_high_X) * (n_high_Y/N) / n_high_X",
        "interval": "95 % bootstrap over neurons of the band, the two poles of a neuron resampled together, 2,000 resamples, numpy default_rng(0) created once per pair and used for bands 0..3 in order, percentile 2.5 and 97.5",
        "side_label": "ABOVE if the interval's lower end > floor; WITHIN if its upper end <= floor; UNDECIDED otherwise (applied to leavers and to arrivals separately, per band)",
        "reading": {"leavers ABOVE, arrivals WITHIN": "loss at fixed identity",
                    "leavers ABOVE, arrivals ABOVE": "movement between poles",
                    "leavers WITHIN, arrivals WITHIN": "within drift",
                    "leavers WITHIN, arrivals ABOVE": "arrivals only",
                    "any side UNDECIDED": "undecided"},
        "does_not_count_if": "any counts file fails counts.sum(-1) == filled (assert, fails loud), or the identity pair start_a->start_b has a nonzero leaver or arrival fraction in any band",
        "decomposition": "per band, Delta = mean over the band's poles of (share at Y - share at X) = high_part + rest_part, where high_part sums the differences of poles at or above tau at X and rest_part the others, both divided by the band's pole count",
        "per_pair_point_values": per_pair,
        "note_on_pairs": "The brief names (t14b, t15) as the third neighbour pair; M1_SPEC 3.2 used (t14c, t15) for its drift tau. The floor follows the brief; t14c->t15 is reported beside it, not in the floor.",
        "source_files": [str(path(k)) for k in REFS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rule, open(OUT / "rule.json", "w"), indent=1)
    print("tau", [round(t, 4) for t in tau], "frac >= tau at t15", [round(f, 4) for f in frac_ge])
    print("floor leavers", [round(v, 4) for v in floor_L], "floor arrivals", [round(v, 4) for v in floor_A])
    for k, v in per_pair.items():
        print(f"  {k:11s} L", [round(r["leavers"], 4) for r in v], "A", [round(r["arrivals"], 4) for r in v], "nH", [r["n_high_X"] for r in v])
