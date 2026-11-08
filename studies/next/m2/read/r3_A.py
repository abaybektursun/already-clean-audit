"""Reader of M2, phase 1, step 3: score A for every unit and role from the banked top-50 arrays on S3,
the matched-pole set from the banked max_abs, and the competition counts at the pre role.

A_q(t) = |E_c intersect top50_q(t)| (M2_SPEC.md 0.5). Rank = number of the layer's 22,016 poles with score >= the
unit's. Matched set (3.4): the 100 poles nearest the carrier in log max_abs at T_(k-1), the carrier's neuron excluded,
poles with max_abs 0 excluded (A4). Inputs: S3 results/neurons/w3/grid/<dir>/out/tNN_stepS/{topk,stats}/layer_LL.npz
(members tok, max_abs only, by range reads), births.json, banks/score-06/m2/ranks_*.jsonl (compared, never used to
compute). Output: read/r3_A.json.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s3range

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"
STEPS = [0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 60000, 100000, 180000, 320000, 560000, 1000000, 1413814]


def key(t, l, kind):
    d = "1b" if 7 <= t <= 14 else "1"
    return f"results/neurons/w3/grid/{d}/out/t{t:02d}_step{STEPS[t]}/{kind}/layer_{l:02d}.npz"


def pid(n, s, l):
    return f"L{l:02d}N{n}{'+' if s == 0 else '-'}"


def parse(p):
    return int(p[1:3]), int(p[4:-1]), 0 if p.endswith("+") else 1


def rank(S, v):
    return int((S >= v).sum())


def matched(maxabs, n_c, s_c, K=100):
    flat = maxabs.reshape(-1).astype(np.float64)
    ok = flat > 0
    ok[2 * n_c] = ok[2 * n_c + 1] = False
    idx = np.nonzero(ok)[0]
    dist = np.abs(np.log(flat[idx]) - np.log(float(maxabs[n_c, s_c])))
    o = np.argsort(dist, kind="stable")[:K]
    return idx[o]


def main():
    d = json.loads((M2 / "births.json").read_text())
    B = d["births"]
    rec = {}
    for h in ("freeze", "test"):
        for l in (M2 / f"banks/score-06/m2/ranks_{h}.jsonl").open():
            r = json.loads(l)
            rec[(r["birth"], r["kind"], r["role"])] = r
    # units per (layer, t)
    need = defaultdict(list)
    for b in B:
        k = b["k"]
        for role, t in (("pre", k - 1), ("birth", k), ("end", 15)):
            need[(b["layer"], t)].append((b, "real", role))
            if b.get("pb") and role != "birth":
                need[(b["layer"], t)].append((b, "pb", role))
    pre_pairs = {(b["layer"], b["k"] - 1) for b in B}
    out, t0 = {}, time.time()
    control = {}
    for i, (l, t) in enumerate(sorted(need)):
        tok = s3range.members(key(t, l, "topk"), ["tok"])["tok"]
        assert tok.shape == (11008, 2, 50)
        mx = s3range.members(key(t, l, "stats"), ["max_abs"])["max_abs"] if (l, t) in pre_pairs else None
        for b, kind, role in need[(l, t)]:
            src = b if kind == "real" else b["pb"]
            E = np.array(src["tok"], dtype=np.int64)
            _, n, s = parse(src["pole"])
            Aall = np.isin(tok, E).sum(-1).astype(np.int64)  # (11008, 2)
            a_c = int(Aall[n, s])
            r_c = rank(Aall, a_c)
            order = np.argsort(-Aall.reshape(-1), kind="stable")
            top10_vals = [int(Aall.reshape(-1)[j]) for j in order[:10]]
            u = {"t": t, "A": a_c, "rank_A": r_c, "top10_A_values": top10_vals}
            if role == "end":
                u["rank_opposite_A"] = rank(Aall, int(Aall[n, 1 - s]))
                u["A_eq_size"] = a_c == len(E)
            other = Aall.copy()
            other[n, s] = 0
            u["n_other_poles_A_ge1"] = int((other >= 1).sum())
            u["n_other_poles_A_ge2"] = int((other >= 2).sum())
            u["best_other_A"] = int(other.max())
            held = np.isin(E, tok.reshape(-1)[tok.reshape(-1) >= 0])  # position in any pole's top-50
            own = np.isin(E, tok[n, s])
            held_other = np.array([bool(((tok == e).any(-1).sum() - int((tok[n, s] == e).any())) > 0) for e in E])
            u["share_pos_in_own_top50"] = float(own.mean())
            u["share_pos_in_other_top50"] = float(held_other.mean())
            u["n_other_pole_slots_on_pos"] = int(np.isin(tok, E).sum() - int(np.isin(tok[n, s], E).sum()))
            r = rec.get((b["id"], kind, role))
            if r is not None:
                u["job_rank_A"] = r["rank"]["A"]
                u["job_value_A"] = r["value"]["A"]
                u["job_top10_A_values"] = [x[1] for x in r["top10"]["A"]]
                if role == "end":
                    u["job_rank_opposite_A"] = r["rank_opposite"]["A"]
            if role == "pre" and kind == "real":
                M = matched(mx, n, s)
                names = [pid(j // 2, j % 2, l) for j in M]
                u["matched_equal_job_list"] = names == r["matched"]
                u["matched_equal_job_set"] = set(names) == set(r["matched"])
                u["matched_overlap"] = len(set(names) & set(r["matched"]))
                mr = [rank(Aall, int(Aall.reshape(-1)[j])) for j in M]
                u["matched_rank_A"] = mr
                # job's matched ranks under A, recomputed on the job's own list
                jr = [rank(Aall, int(Aall[parse(p)[1], parse(p)[2]])) for p in r["matched"]]
                u["matched_rank_A_job_list_equal"] = jr == r["matched_rank"]["A"]
            out[f"{b['id']}|{kind}|{role}"] = u
        if t == 15 and "shift" not in control:
            # failing input: neuron axis shifted by one; the construction control must break
            b, kind, role = [x for x in need[(l, t)] if x[1] == "real" and x[2] == "end"][0]
            E = np.array(b["tok"], dtype=np.int64)
            _, n, s = parse(b["pole"])
            sh = np.roll(tok, 1, axis=0)
            control["shift"] = {"unit": b["id"], "A_carrier_true": int(np.isin(tok[n, s], E).sum()),
                                "A_carrier_shifted": int(np.isin(sh[n, s], E).sum()), "size": len(E)}
        print(f"{i + 1}/{len(need)} L{l:02d} T{t} units {len(need[(l, t)])} {time.time() - t0:.0f}s", flush=True)
    # summary
    summ = {"units": len(out)}
    keys = list(out)
    summ["rank_A_equal_job"] = sum(out[k].get("job_rank_A") == out[k]["rank_A"] for k in keys)
    summ["value_A_equal_job"] = sum(out[k].get("job_value_A") == out[k]["A"] for k in keys)
    summ["top10_values_equal_job"] = sum(out[k].get("job_top10_A_values") == out[k]["top10_A_values"] for k in keys)
    summ["rank_A_differs"] = [[k, out[k]["rank_A"], out[k].get("job_rank_A")] for k in keys
                              if out[k].get("job_rank_A") != out[k]["rank_A"]][:20]
    ends = [k for k in keys if k.endswith("|real|end")]
    summ["end_A_eq_size"] = sum(out[k]["A_eq_size"] for k in ends)
    summ["n_end_real"] = len(ends)
    summ["rank_opposite_equal_job"] = sum(out[k]["rank_opposite_A"] == out[k].get("job_rank_opposite_A")
                                          for k in keys if k.endswith("|end"))
    pres = [k for k in keys if k.endswith("|real|pre")]
    summ["matched_list_equal"] = sum(out[k]["matched_equal_job_list"] for k in pres)
    summ["matched_set_equal"] = sum(out[k]["matched_equal_job_set"] for k in pres)
    summ["matched_min_overlap"] = min(out[k]["matched_overlap"] for k in pres)
    summ["matched_rank_A_job_list_equal"] = sum(out[k]["matched_rank_A_job_list_equal"] for k in pres)
    summ["n_pre_real"] = len(pres)
    summ["failing_input_shift"] = control.get("shift")
    res = {"summary": summ, "units": out}
    (M2 / "read/r3_A.json").write_text(json.dumps(res))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
