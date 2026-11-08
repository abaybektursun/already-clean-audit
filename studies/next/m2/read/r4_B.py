"""Reader of M2, phase 1, step 4: score B recomputed for a sample of (layer, point) pairs from the job's vectors and
the banked 400-bin histograms.

B_q(t) (M2_SPEC.md 0.5): v_q = numpy median over E_c of a+_q (a+ = max(a, 0) for the + pole, max(-a, 0) for the -
pole); B_q = share of q's own-sign tokens with |a| below v_q from hist: bins below v's bin whole, v's bin by the
log-linear share below v; B = 0 when v = 0 or the pole has no own-sign token (A4). Rank = number of the layer's
22,016 poles with B >= the unit's. Inputs: S3 results/next_m2/score-06/out/score/m2/vectors/tNN/<unit>_<role>.npz;
S3 results/neurons/w3/grid/<dir>/out/tNN_stepS/stats/layer_LL.npz (hist, hist_edges, n_pole_tokens, max_abs, q999);
the laptop bank arrays.npz (the carrier's own banked a at E_c); births.json; ranks_*.jsonl (compared only).
Output: read/r4_B.json.
"""
import io, json, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s3range
from r3_A import key, pid, parse, rank, matched

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"
ARR = Path("/Users/abaybektursun/projects/interp_research_banks/neurons/next/recount2/arrays.npz")
VEC = "results/next_m2/score-06/out/score/m2/vectors"
PAIRS = [(1, 11), (2, 14), (3, 11), (3, 12), (3, 14), (5, 12), (6, 14), (7, 13), (10, 9), (10, 12), (10, 13),
         (12, 12), (16, 13), (19, 12), (20, 11), (22, 12), (24, 7), (24, 11), (27, 11), (28, 9),  # raw examples
         (3, 5), (22, 8), (24, 10),  # T_5, T_8, T_10 (holds the degenerate L24N9826+)
         (3, 15), (10, 15)]  # end role (construction control under B)


def percentile(hist, edges, v):
    """hist [P, 400] int, edges [401], v [P] -> B [P]."""
    h = hist.astype(np.float64)
    N = h.sum(1)
    cum = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, 1)], 1)  # cum[:, j] = sum of bins < j
    e = edges.astype(np.float64)
    b = np.searchsorted(e, v, side="right") - 1
    out = np.zeros_like(v)
    inside = (v > 0) & (b >= 0) & (b < 400) & (N > 0)
    bi = b[inside]
    lo, hi = np.log(e[bi]), np.log(e[bi + 1])
    frac = (np.log(v[inside]) - lo) / (hi - lo)
    rows = np.nonzero(inside)[0]
    out[inside] = (cum[rows, bi] + h[rows, bi] * frac) / N[inside]
    top = (v > 0) & (b >= 400) & (N > 0)
    out[top] = 1.0
    return out


def vec(t, unit, role):
    k = f"{VEC}/t{t:02d}/{unit.replace(':', '_')}_{role}.npz"
    url = s3range.presign(k)
    with urllib.request.urlopen(url) as r:
        z = np.load(io.BytesIO(r.read()), allow_pickle=False)
        return {n: z[n] for n in z.files}


def main():
    d = json.loads((M2 / "births.json").read_text())
    B = d["births"]
    z = np.load(ARR, allow_pickle=False)
    bankA = z["a"]
    rec = {}
    for h in ("freeze", "test"):
        for l in (M2 / f"banks/score-06/m2/ranks_{h}.jsonl").open():
            r = json.loads(l)
            rec[(r["birth"], r["kind"], r["role"])] = r
    need = defaultdict(list)
    for b in B:
        k = b["k"]
        for role, t in (("pre", k - 1), ("birth", k), ("end", 15)):
            if (b["layer"], t) in PAIRS:
                need[(b["layer"], t)].append((b, "real", role))
                if b.get("pb") and role != "birth":
                    need[(b["layer"], t)].append((b, "pb", role))
    out, t0, ctl = {}, time.time(), {}
    for i, (l, t) in enumerate(PAIRS):
        st = s3range.members(key(t, l, "stats"), ["hist", "hist_edges", "n_pole_tokens", "max_abs", "q999"])
        hist = st["hist"].reshape(22016, 400)
        assert np.array_equal(st["hist"].sum(-1), st["n_pole_tokens"])
        edges = st["hist_edges"]
        degen = [pid(j // 2, j % 2, l) for j in np.nonzero(st["n_pole_tokens"].reshape(-1) == 0)[0]]
        for b, kind, role in need[(l, t)]:
            src = b if kind == "real" else b["pb"]
            unit = b["id"] if kind == "real" else b["id"] + ":pb"
            V = vec(t, unit, role)
            a = V["a"].astype(np.float64)
            assert a.shape == (len(src["tok"]), 11008)
            _, n, s = parse(src["pole"])
            vpos = np.median(np.maximum(a, 0), 0)
            vneg = np.median(np.maximum(-a, 0), 0)
            v = np.stack([vpos, vneg], 1).reshape(-1)  # flat index 2n+s
            Ball = percentile(hist, edges, v)
            q = st["q999"].reshape(-1).astype(np.float64)
            fin = np.isfinite(q) & (q > 0)
            ratio = np.where(fin, v / np.where(fin, q, 1), 0.0)
            j = 2 * n + s
            u = {"t": t, "role": role, "kind": kind, "B": float(Ball[j]), "rank_B": rank(Ball, Ball[j]),
                 "ratio": float(ratio[j]), "rank_ratio": rank(ratio, ratio[j]),
                 "tok_equal": [int(x) for x in V["tok"]] == src["tok"]}
            o = np.argsort(-Ball, kind="stable")[:10]
            u["top10_B"] = [[pid(x // 2, x % 2, l), float(Ball[x])] for x in o]
            # 10th best B among the other poles: the bar the unit must reach
            oth = Ball.copy(); oth[j] = -1
            u["B_10th_other"] = float(np.sort(oth)[-10])
            u["n_other_B_ge_unit"] = int((oth >= Ball[j]).sum())
            # the carrier's own activations against the laptop bank (its own exemplar positions)
            ab = bankA[t, src["row"], src["ranks"]]
            an = a[:, n]
            u["own_a_vs_bank_maxabs"] = float(np.max(np.abs(an - ab)))
            u["own_a_vs_bank_p99_d"] = float(np.percentile(np.abs(an - ab) / (0.05 + np.abs(ab)), 99))
            r = rec[(b["id"], kind, role)]
            u["job_rank_B"] = r["rank"]["B"]
            u["job_B"] = r["value"]["B"]
            u["job_rank_ratio"] = r["rank"]["ratio"]
            u["job_ratio"] = r["value"]["ratio"]
            u["top10_B_names_equal_job"] = [x[0] for x in u["top10_B"]] == [x[0] for x in r["top10"]["B"]]
            u["top10_B_maxdiff_job"] = float(max(abs(x[1] - y[1]) for x, y in zip(u["top10_B"], r["top10"]["B"])))
            if role == "end":
                u["rank_opposite_B"] = rank(Ball, Ball[2 * n + 1 - s])
                u["job_rank_opposite_B"] = r["rank_opposite"]["B"]
            if role == "pre" and kind == "real":
                M = matched(st["max_abs"], n, s)
                names = [pid(x // 2, x % 2, l) for x in M]
                u["matched_equal_job_list"] = names == r["matched"]
                jr = [rank(Ball, Ball[2 * parse(p)[1] + parse(p)[2]]) for p in r["matched"]]
                u["matched_rank_B_equal_job"] = jr == r["matched_rank"]["B"]
                u["matched_rank_B"] = jr
            if not ctl and role == "pre" and kind == "real":
                # failing input: the vectors' neuron axis shifted by one; the carrier's B and rank must move
                Bs = percentile(hist, edges, np.stack([np.median(np.maximum(np.roll(a, 1, 1), 0), 0),
                                                       np.median(np.maximum(-np.roll(a, 1, 1), 0), 0)], 1).reshape(-1))
                ctl = {"unit": unit, "B_true": float(Ball[j]), "rank_true": rank(Ball, Ball[j]),
                       "B_shifted": float(Bs[j]), "rank_shifted": rank(Bs, Bs[j])}
            out[f"{unit}|{role}"] = u
        print(f"{i + 1}/{len(PAIRS)} L{l:02d} T{t} units {len(need[(l, t)])} degenerate {degen} "
              f"{time.time() - t0:.0f}s", flush=True)
        out[f"_degenerate|L{l:02d}|T{t}"] = degen
    U = {k: v for k, v in out.items() if not k.startswith("_")}
    summ = {"pairs": len(PAIRS), "units": len(U),
            "units_by_role_kind": {f"{r}|{k}": sum(1 for u in U.values() if u["role"] == r and u["kind"] == k)
                                   for r in ("pre", "birth", "end") for k in ("real", "pb")},
            "births_pre_real": sorted({k.split("|")[0] for k, u in U.items() if u["role"] == "pre" and u["kind"] == "real"}),
            "rank_B_equal_job": sum(u["rank_B"] == u["job_rank_B"] for u in U.values()),
            "max_abs_B_diff": max(abs(u["B"] - u["job_B"]) for u in U.values()),
            "rank_ratio_equal_job": sum(u["rank_ratio"] == u["job_rank_ratio"] for u in U.values()),
            "max_rel_ratio_diff": max(abs(u["ratio"] - u["job_ratio"]) / max(1e-12, abs(u["job_ratio"])) for u in U.values()),
            "top10_names_equal_job": sum(u["top10_B_names_equal_job"] for u in U.values()),
            "top10_max_value_diff": max(u["top10_B_maxdiff_job"] for u in U.values()),
            "tok_equal": sum(u["tok_equal"] for u in U.values()),
            "rank_B_differs": [[k, u["rank_B"], u["job_rank_B"], u["B"], u["job_B"]] for k, u in U.items()
                               if u["rank_B"] != u["job_rank_B"]],
            "matched_rank_B_equal_job": sum(u.get("matched_rank_B_equal_job", False) for u in U.values()),
            "matched_list_equal_job": sum(u.get("matched_equal_job_list", False) for u in U.values()),
            "n_pre_real": sum(1 for u in U.values() if u["role"] == "pre" and u["kind"] == "real"),
            "own_a_vs_bank_maxabs_by_t": {}, "failing_input_shift": ctl,
            "opposite_equal_job": sum(u.get("rank_opposite_B") == u.get("job_rank_opposite_B") for u in U.values() if u["role"] == "end")}
    for u in U.values():
        tt = str(u["t"])
        summ["own_a_vs_bank_maxabs_by_t"][tt] = max(summ["own_a_vs_bank_maxabs_by_t"].get(tt, 0), u["own_a_vs_bank_maxabs"])
    (M2 / "read/r4_B.json").write_text(json.dumps({"summary": summ, "units": out}))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
