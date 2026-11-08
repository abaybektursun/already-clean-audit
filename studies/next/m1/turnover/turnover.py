"""Dated exploratory analysis (2026-11-05) on the closed, unblinded M1 result; no rule here decides a headline.
Steps 3 and 4 of the pole-turnover analysis. tau and the floor are read from rule.json (frozen and committed first,
e170384ca). Run from the repo root: python3 studies/next/m1/turnover/turnover.py  ->  studies/next/m1/turnover/turnover.json
"""
import json, numpy as np
from rule import load, share_from_counts, per_neuron, point, band, OUT, path

RULE = json.load(open(OUT / "rule.json"))
TAU, FL, FA = RULE["tau"], RULE["floor_leavers"], RULE["floor_arrivals"]
STATS = ("leavers", "arrivals", "delta", "high_part", "rest_part")
KEYS = ["start_a", "start_b", "G", "R", "M", "t12", "t13", "t14b", "t14c", "t15"]


def boot(SX, SYs, nboot=2000, chunk=200):
    """Resampled statistics per Y against one X, per band. default_rng(0) is created per call, so every call
    (and every Y within it) uses the same neuron resamples; the two poles of a neuron are resampled together.
    A resample is held as the count of times each neuron is drawn (W), so a resampled sum is W @ per-neuron values."""
    rng = np.random.default_rng(0)
    out = {k: [] for k in SYs}
    for b in range(4):
        qs = {k: per_neuron(SX, SY, TAU[b], b) for k, SY in SYs.items()}
        n = len(next(iter(qs.values()))["H"]); N = 2 * n
        acc = {k: [] for k in qs}
        for _ in range(nboot // chunk):
            idx = rng.integers(0, n, (chunk, n))
            W = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)
            for k, q in qs.items():
                acc[k].append(W @ np.stack([q["H"], q["L"], q["A"], q["dH"], q["dR"]], 1).astype(np.float64))
        for k in qs:
            H, L, A, dH, dR = np.concatenate(acc[k]).T
            out[k].append({"leavers": L / H, "arrivals": A / H, "high_part": dH / N, "rest_part": dR / N, "delta": (dH + dR) / N})
    return out


def side(lo, hi, floor):
    return "ABOVE" if lo > floor else "WITHIN" if hi <= floor else "UNDECIDED"


def reading(sl, sa):
    if "UNDECIDED" in (sl, sa):
        return "undecided"
    return {("ABOVE", "WITHIN"): "loss at fixed identity", ("ABOVE", "ABOVE"): "movement between poles",
            ("WITHIN", "WITHIN"): "within drift", ("WITHIN", "ABOVE"): "arrivals only"}[(sl, sa)]


def ci(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def run(SX, SYs):
    samples = boot(SX, SYs)
    res = {}
    for k, SY in SYs.items():
        rows = []
        for b in range(4):
            p = point(SX, SY, TAU[b], b)
            q = per_neuron(SX, SY, TAU[b], b); N = 2 * len(q["H"])
            p["high_part"], p["rest_part"] = float(q["dH"].sum() / N), float(q["dR"].sum() / N)
            p["delta"] = p["high_part"] + p["rest_part"]
            p["high_mean"] = float(q["dH"].sum() / p["n_high_X"])            # mean Y - X over poles at or above tau at X
            p["rest_mean"] = float(q["dR"].sum() / (N - p["n_high_X"]))      # mean Y - X over the other poles
            p["high_part_fraction_of_delta"] = p["high_part"] / p["delta"] if p["delta"] != 0 else "not defined: delta is 0"
            for s in STATS:
                p[s + "_ci95"] = ci(samples[k][b][s])
            p["side_leavers"] = side(*p["leavers_ci95"], FL[b])
            p["side_arrivals"] = side(*p["arrivals_ci95"], FA[b])
            p["reading"] = reading(p["side_leavers"], p["side_arrivals"])
            p["floor_leavers"], p["floor_arrivals"] = FL[b], FA[b]
            rows.append(p)
        res[k] = rows
    return res, samples


def counts_of(key):
    c, f, m = load(key)
    return c.copy(), f.copy(), m


if __name__ == "__main__":
    S = {}
    for k in KEYS:                                  # share_from_counts asserts counts.sum(-1) == filled for every file
        S[k] = share_from_counts(*load(k))
    out = {"label": "Dated exploratory analysis, 2026-11-05, on the closed and unblinded M1 result. No rule here decides a headline.",
           "rule_file": "studies/next/m1/turnover/rule.json (committed e170384ca before this script read start_a, start_b, G, R, M)",
           "tau": TAU, "floor_leavers": FL, "floor_arrivals": FA,
           "source_files": [str(path(k)) for k in KEYS], "assert_counts_sum_equals_filled": "passed for all ten files"}

    # step 3: the mixes and the identity pair, all from start_a, same neuron resamples
    mix, ms = run(S["start_a"], {k: S[k] for k in ("start_b", "G", "R", "M")})
    out["mixes"] = {f"start_a->{k}": v for k, v in mix.items()}
    out["identity_pair_zero"] = all(r["leavers"] == 0 and r["arrivals"] == 0 for r in mix["start_b"])
    out["result_counts"] = out["identity_pair_zero"]        # rule.json: the readings do not count if the identity pair is not zero
    # the floor pairs with intervals and their decomposition (the regression-to-the-mean comparison)
    out["floor_pairs"] = {}
    for x, y in [("t12", "t13"), ("t13", "t14b"), ("t14b", "t15"), ("t14b", "t14c"), ("t14c", "t15")]:
        r, _ = run(S[x], {y: S[y]})
        out["floor_pairs"][f"{x}->{y}"] = r[y]

    # step 4: planted inputs, each replacing G's counts and run through the same code against start_a
    c0, f0, mask = counts_of("start_a")
    noncm0 = int(np.where(~mask)[0][0])
    pole_band = np.repeat(band, 11008 * 2)          # band of each flat pole index, layer-major order
    hi_flat = np.concatenate([(S["start_a"][l] >= TAU[band[l]]).ravel() for l in range(32)])
    rng = np.random.default_rng(0)

    # P2: a G-sized loss at fixed identity with no drift: in band b each code+math slot of every pole is removed with
    # probability r_b = -Delta_G(b) / C_start(b) (binomial thinning); removed slots go to the first non-code+math directory.
    # (A first version rounded c * (1 - r_b) to the nearest integer; it realized only 71-93 % of G's band-mean change.)
    r_b = [-mix["G"][b]["delta"] / float(S["start_a"][band == b].mean()) for b in range(4)]
    c2 = c0.copy().astype(np.int64)
    for l in range(32):
        cm = c2[l][..., mask]
        keep = rng.binomial(cm, 1 - r_b[band[l]])
        c2[l][..., noncm0] += (cm - keep).sum(-1)
        c2[l][..., mask] = keep
    c2 = c2.astype(np.uint8)

    # P3: a strong loss at fixed identity: a random half of the poles at or above tau at the start lose every code+math slot
    C3 = c0.reshape(-1, 50).copy()
    sel3 = np.concatenate([rng.choice(np.where(hi_flat & (pole_band == b))[0], int((hi_flat & (pole_band == b)).sum()) // 2, replace=False) for b in range(4)])
    C3[sel3, noncm0] += C3[sel3][:, mask].sum(1).astype(np.uint8)
    C3[np.ix_(sel3, np.where(mask)[0])] = 0

    # P4: movement between poles: a random half of the high poles swap their 50-directory counts with as many random
    # poles below tau of the same band; the band mean does not change
    C4 = c0.reshape(-1, 50).copy()
    for b in range(4):
        h = np.where(hi_flat & (pole_band == b))[0]; lo = np.where(~hi_flat & (pole_band == b))[0]
        sh = rng.choice(h, len(h) // 2, replace=False); sl = rng.choice(lo, len(h) // 2, replace=False)
        C4[sh], C4[sl] = c0.reshape(-1, 50)[sl], c0.reshape(-1, 50)[sh]

    planted_S = {"P1_G_is_t15": S["t15"], "P5_G_is_t13": S["t13"],
                 "P2_G_sized_loss_no_drift": share_from_counts(c2, f0, mask),
                 "P3_half_of_high_set_zeroed": share_from_counts(C3.reshape(c0.shape), f0, mask),
                 "P4_half_of_high_set_swapped": share_from_counts(C4.reshape(c0.shape), f0, mask)}
    expected = {"P1_G_is_t15": "start_a (step 476,838) -> t15 (step 1,413,814) spans 936,976 stage-1 steps, about twice the longest floor pair (440,000). Expected before running: leavers and arrivals both ABOVE the floor in every band, read as 'movement between poles'. This is stage-1 drift, so if it fires it shows the floor is set by the span of the pairs.",
                "P5_G_is_t13": "start_a -> t13 (step 560,000) spans 83,162 steps, shorter than every floor pair. Expected: 'within drift' in every band.",
                "P2_G_sized_loss_no_drift": "start_a with each code+math slot removed with probability r_b so the band mean falls by G's measured amount, no drift. Expected: arrivals 0 (WITHIN). Leavers: the power check; no expectation was fixed. If leavers stay WITHIN, a G-sized loss at fixed identity cannot be told from stage-1 drift by this rule.",
                "P3_half_of_high_set_zeroed": "Expected: leavers about 0.5 (ABOVE), arrivals 0 (WITHIN): 'loss at fixed identity' in every band.",
                "P4_half_of_high_set_swapped": "Expected: leavers about 0.5 and arrivals about 0.5 (both ABOVE): 'movement between poles' in every band; band-mean change 0."}
    expected_reading = {"P1_G_is_t15": ["movement between poles"] * 4, "P5_G_is_t13": ["within drift"] * 4, "P2_G_sized_loss_no_drift": "no expectation fixed (power check)",
                        "P3_half_of_high_set_zeroed": ["loss at fixed identity"] * 4, "P4_half_of_high_set_swapped": ["movement between poles"] * 4}
    pl, ps = run(S["start_a"], planted_S)
    # paired differences (every run() call uses the same neuron resamples): G - M, G - R, and G - t15 (planted P1)
    out["paired_differences"] = {}
    for name, (r1, s1), (r2, s2) in [("G-M", (mix["G"], ms["G"]), (mix["M"], ms["M"])), ("G-R", (mix["G"], ms["G"]), (mix["R"], ms["R"])),
                                     ("G-t15", (mix["G"], ms["G"]), (pl["P1_G_is_t15"], ps["P1_G_is_t15"]))]:
        out["paired_differences"][name] = [{s: {"mean": r1[b][s] - r2[b][s], "ci95": ci(s1[b][s] - s2[b][s])} for s in STATS} for b in range(4)]
        for d in out["paired_differences"][name]:
            d["rest_part_fraction_of_delta"] = d["rest_part"]["mean"] / d["delta"]["mean"]
    out["planted"] = {k: {"expected": expected[k], "expected_reading": expected_reading[k], "readings": [r["reading"] for r in pl[k]],
                          "expectation_met": "not applicable" if isinstance(expected_reading[k], str) else [r["reading"] for r in pl[k]] == expected_reading[k],
                          "result": pl[k]} for k in planted_S}
    out["planted"]["P2_G_sized_loss_no_drift"]["r_b"] = r_b
    out["planted"]["P2_G_sized_loss_no_drift"]["realized_delta_vs_G_delta"] = [[pl["P2_G_sized_loss_no_drift"][b]["delta"], mix["G"][b]["delta"]] for b in range(4)]

    json.dump(out, open(OUT / "turnover.json", "w"), indent=1)

    def show(name, rows):
        print(f"{name:30s}", " | ".join(f"L {r['leavers']:.3f} [{r['leavers_ci95'][0]:.3f},{r['leavers_ci95'][1]:.3f}] A {r['arrivals']:.3f} [{r['arrivals_ci95'][0]:.3f},{r['arrivals_ci95'][1]:.3f}] {r['reading']}" for r in rows))
    print("tau", TAU, "floor L", [round(v, 3) for v in FL], "floor A", [round(v, 3) for v in FA])
    for k, v in out["mixes"].items(): show(k, v)
    for k, v in out["floor_pairs"].items(): show(k, v)
    for k, v in out["planted"].items(): show(k, v["result"])
    print("\ndecomposition (delta = high_part + rest_part):")
    for k, v in list(out["mixes"].items()) + list(out["floor_pairs"].items()):
        print(f"{k:16s}", " | ".join(f"{r['delta']:+.4f} = {r['high_part']:+.4f} + {r['rest_part']:+.4f}" for r in v))
    for name, v in out["paired_differences"].items():
        print(name, " | ".join(" ".join(f"{s[0:4]} {d[s]['mean']:+.4f} [{d[s]['ci95'][0]:+.4f},{d[s]['ci95'][1]:+.4f}]" for s in STATS) for d in v))
    print("high_mean / rest_mean:")
    for k, v in list(out["mixes"].items()) + list(out["floor_pairs"].items()):
        print(f"{k:16s}", " | ".join(f"{r['high_mean']:+.4f} / {r['rest_mean']:+.4f}  nH {r['n_high_X']}->{r['n_high_Y']}" for r in v))
    print("identity pair zero:", out["identity_pair_zero"], "| P2 realized vs G delta:", [[round(a, 4), round(c, 4)] for a, c in out["planted"]["P2_G_sized_loss_no_drift"]["realized_delta_vs_G_delta"]])
