"""N12, the ten raw examples at zero GPU (N12_SPEC.md section 10), from banked files only.

The ten births are M2's freeze-half example births (studies/next/m2/births.json -> examples.freeze; selected by M2's
frozen rule, before N12). For each:
  - the competitor set at T_(k-1): B's top 20 on E_c, the carrier's neuron excluded (n12.competitors), each with its
    own B and layer rank at T_(k-1) and at T_k; their median drop; the carrier's ranks r0, r1 and the counterfactual
    r_c, r_o, F;
  - the decoded own top-3 contexts at T_(k-1) of the top three competitors and the first matched random pole, from
    M2's banked raw examples (score-06 samples/raw_examples_freeze.jsonl, top10 lists with top3 texts);
  - the exemplars decoded (T_15 contexts.jsonl), the distinct-token count, M2's twin and the N12 distinct-token twin.
Inputs: M2's job vectors on S3 (results/next_m2/score-06/out/score/m2/vectors/tNN/<unit>_<role>.npz, all 11,008
neurons at E_c) and the banked histograms (results/neurons/w3/grid/<dir>/stats/layer_LL.npz, members hist,
hist_edges, n_pole_tokens, q999, max_abs by byte-range reads; nothing else of the file is fetched).
Checks written beside the examples (read by the offline gate):
  - G-M2TOP10: the top-10 under B recomputed from the vectors and the histograms equals M2's banked top10 (names and
    values) at the pre and the birth role; failing input: the vectors' neuron axis rolled by one;
  - the decomposition's identity on real arrays (T_k replaced by T_(k-1)): r_c = r_o = r0, F = 0.
Freeze half only: no test-half unit is read.

    PYTHONPATH=. python studies/next/n12/samples/raw_examples.py --out studies/next/n12/samples --cache <scratch>
"""
import argparse
import io
import json
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies/next/m2/read"))
import s3range  # noqa: E402

from infra.pg import m2, n12  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
VEC = "results/next_m2/score-06/out/score/m2/vectors"
M2B = ROOT / "studies/next/m2/banks/score-06/m2"


def stats_key(t, layer):
    return f"results/neurons/w3/grid/{m2.POINTS[t][1]}/stats/layer_{layer:02d}.npz"


def vec(t, name):
    url = s3range.presign(f"{VEC}/t{t:02d}/{name}.npz")
    with urllib.request.urlopen(url) as r:
        z = np.load(io.BytesIO(r.read()), allow_pickle=False)
        return {k: z[k] for k in z.files}


def b_array(a, st):
    """B [N, 2] from a [m, N] and a stats dict (m2.percentile; 0 for an empty histogram, A4)."""
    return m2.percentile(m2.median_a_plus(a), st["hist"], st["hist_edges"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True, help="a scratch directory for the fetched histogram members")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    uj = json.loads((ROOT / "studies/next/n12/units.json").read_text())
    U = {u["unit"]: u for u in uj["units"]}
    ids = uj["examples"]["freeze"]
    m2rec = {(r["unit"], r["role"]): r for r in m2.read_jsonl(M2B / "ranks_freeze.jsonl")}
    m2ex = {r["unit"]: r for r in m2.read_jsonl(M2B / "samples/raw_examples_freeze.jsonl")}
    ctx = {}
    for line in open(BANK / "w3/grid" / m2.POINTS[15][1] / "samples/contexts.jsonl"):
        r = json.loads(line)
        ctx[r["pole"]] = {c["rank"]: c for c in r["topk"]}
    stats = {}
    rows, checks = [], {"top10_pre": [], "top10_birth": [], "top10_fail": [], "identity": [], "n_comp": []}
    for bid in ids:
        u = U[bid]
        assert u["half"] == "freeze"
        n, s = u["carrier"]
        lyr, k = u["layer"], u["k"]
        B = {}
        for role, t in (("pre", k - 1), ("birth", k)):
            if (lyr, t) not in stats:
                cf_ = pathlib.Path(a.cache) / f"stats_t{t:02d}_L{lyr:02d}.npz"
                if not cf_.exists():          # a local copy of the five members, so a rerun reads S3 once
                    np.savez(cf_, **s3range.members(stats_key(t, lyr), ["hist", "hist_edges", "n_pole_tokens",
                                                                         "max_abs", "q999"]))
                st = dict(np.load(cf_))
                assert (st["hist"].sum(-1) == st["n_pole_tokens"]).all()
                stats[(lyr, t)] = st
            v = vec(t, f"{bid.replace(':', '_')}_{role}")
            assert v["tok"].tolist() == u["tok"], bid
            B[role] = b_array(v["a"], stats[(lyr, t)])
            top = m2.top_list(B[role], lyr)
            want = m2rec[(bid, role)]["top10"]["B"]
            checks[f"top10_{role}"].append(bool([p for p, _ in top] == [p for p, _ in want] and
                                               max(abs(x - y) for (_, x), (_, y) in zip(top, want)) < 1e-12))
            if role == "pre":
                rolled = m2.top_list(b_array(np.roll(v["a"], 1, axis=1), stats[(lyr, t)]), lyr)
                checks["top10_fail"].append(bool([p for p, _ in rolled] == [p for p, _ in want]))
        comp = n12.competitors(B["pre"], n)
        checks["n_comp"].append(len(comp))
        cf = n12.cf_ranks(B["pre"], B["birth"], n, s)
        idn = n12.cf_ranks(B["pre"], B["pre"], n, s)
        checks["identity"].append(bool(idn["r_c"] == idn["r_o"] == idn["r0"] and idn["F"] == 0.0))
        assert cf["r0"] == m2rec[(bid, "pre")]["rank"]["B"] and cf["r1"] == m2rec[(bid, "birth")]["rank"]["B"], bid
        ex = m2ex[bid]
        texts = {e["pole"]: e["top3"] for e in ex["top10"]["B"]}
        comp_rows = []
        for j, (cn, cs) in enumerate(comp):
            pid = m2.pole_id(lyr, cn, cs)
            comp_rows.append({"j": j + 1, "pole": pid, "B_pre": float(B["pre"][cn, cs]),
                              "rank_pre": n12.layer_rank(B["pre"], cn, cs), "B_birth": float(B["birth"][cn, cs]),
                              "rank_birth": n12.layer_rank(B["birth"], cn, cs),
                              "top3": texts.get(pid) if j < 3 else None})
        pb, dt = U.get(bid + ":pb"), U.get(bid + ":dt")

        def exem(pole, ranks):
            return [ctx[pole][r]["before"][-120:] + " [[" + ctx[pole][r]["token"] + "]] " + ctx[pole][r]["after"]
                    for r in ranks]
        rows.append({
            "birth": bid, "name": [x for x in json.loads((ROOT / "studies/next/m2/births.json").read_text())["births"]
                                   if x["id"] == bid][0]["name"],
            "layer": lyr, "k": k, "t_pre": k - 1, "size": u["size"], "d": u["d"],
            "carrier": {"pole": u["pole"], "B_pre": float(B["pre"][n, s]), "B_birth": float(B["birth"][n, s]), **cf,
                        "bar_pre": float(np.sort(np.delete(B["pre"].ravel(), 2 * n + s))[-m2.TOP])},
            "exemplars": exem(u["pole"], u["ranks"]),
            "competitors": comp_rows,
            "drop": float(np.median([np.log10(r["rank_birth"]) - np.log10(r["rank_pre"]) for r in comp_rows])),
            "median_dB": float(np.median([r["B_birth"] - r["B_pre"] for r in comp_rows])),
            "n_comp_up": int(sum(r["rank_birth"] < r["rank_pre"] for r in comp_rows)),
            "n_comp_down": int(sum(r["rank_birth"] > r["rank_pre"] for r in comp_rows)),
            "n_leaders_pre": int(sum(r["B_pre"] > B["pre"][n, s] for r in comp_rows)),
            "n_leaders_birth": int(sum(r["B_birth"] > B["birth"][n, s] for r in comp_rows)),
            "matched_first": ex["matched_first"],
            "pb": None if pb is None else {"pole": pb["pole"], "d": pb["d"],
                                           "rank_pre_B": m2rec[(bid + ":pb", "pre")]["rank"]["B"]},
            "dt": None if dt is None else {"pole": dt["pole"], "d": dt["d"], "exemplars": exem(dt["pole"], dt["ranks"])},
        })
    res = {"what": __doc__.split("\n")[0], "births": ids, "checks": checks,
           "checks_pass": {"top10_pre": all(checks["top10_pre"]), "top10_birth": all(checks["top10_birth"]),
                           "top10_fail_fails": not any(checks["top10_fail"]), "identity": all(checks["identity"]),
                           "n_comp_20": all(c == n12.K_COMP for c in checks["n_comp"])},
           "stats_reads": sorted([list(x) for x in stats]), "examples": rows}
    (out / "raw_examples_zero_gpu.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    md = ["# N12 raw examples, zero GPU (freeze half; M2's ten freeze example births)", "",
          "Written by `studies/next/n12/samples/raw_examples.py` from M2's banked vectors and the banked histograms. "
          "B is the pole's own percentile of its median a+ over E_c; rank is the layer rank among 22,016 poles "
          "(ties against the pole). r_c: the carrier's rank with only the carrier moved to T_k; r_o: with only the "
          "others moved. F = log10 r_o - log10 r_c. The gradient and ablation columns are PENDING (the job writes them "
          "to `samples/raw_examples_freeze.jsonl`).", "",
          f"Checks: {json.dumps(res['checks_pass'])}", ""]
    for r in rows:
        c = r["carrier"]
        md += [f"## {r['birth']}: {r['name']}", "",
               f"Layer {r['layer']}, born T_{r['k']}, pre T_{r['t_pre']}; size {r['size']}, distinct tokens {r['d']}.", "",
               f"Carrier {c['pole']}: B {c['B_pre']:.6f} -> {c['B_birth']:.6f}; rank r0 {c['r0']} -> r1 {c['r1']}; "
               f"r_c {c['r_c']}, r_o {c['r_o']}, F {c['F']:+.3f}; 10th-best other B at T_(k-1) {c['bar_pre']:.6f}. "
               f"Competitors above the carrier: {r['n_leaders_pre']} of 20 at T_(k-1), {r['n_leaders_birth']} at T_k; "
               f"competitors that lose rank {r['n_comp_down']}, gain rank {r['n_comp_up']}; their median drop "
               f"{r['drop']:.3f} decades of layer rank, median change of B {r['median_dB']:+.5f}.", "",
               "Exemplars (T_15 contexts):", ""]
        md += [f"- `{x[-160:]}`".replace("\n", " ") for x in r["exemplars"][:4]]
        md += ["", "| j | competitor | B at T_(k-1) | rank | B at T_k | rank |", "|---|---|---|---|---|---|"]
        md += [f"| {x['j']} | {x['pole']} | {x['B_pre']:.6f} | {x['rank_pre']} | {x['B_birth']:.6f} | {x['rank_birth']} |"
               for x in r["competitors"]]
        md += ["", "Own top-3 contexts at T_(k-1) of the top three competitors (M2's banked raw examples):", ""]
        for x in r["competitors"][:3]:
            for t3 in (x["top3"] or [])[:3]:
                md.append(f"- {x['pole']} (a {t3['a']:+.3f}): `{t3['text'][-150:]}`".replace("\n", " "))
        mf = r["matched_first"]
        md += ["", f"Matched random pole {mf['pole']} (rank under B {mf['rank']['B']}):", ""]
        md += [f"- `{t3['text'][-150:]}`".replace("\n", " ") for t3 in mf["top3"][:3]]
        if r["pb"]:
            md += ["", f"M2 twin {r['pb']['pole']}: distinct tokens {r['pb']['d']}, rank under B at T_(k-1) "
                       f"{r['pb']['rank_pre_B']}."]
        if r["dt"]:
            md += [f"Distinct-token twin {r['dt']['pole']}: distinct tokens {r['dt']['d']}; first exemplar "
                   f"`{r['dt']['exemplars'][0][-120:]}`".replace("\n", " ")]
        else:
            md += ["Distinct-token twin: none (no candidate group dated k with the same count)."]
        md.append("")
    (out / "raw_examples_zero_gpu.md").write_text("\n".join(md))
    print(json.dumps(res["checks_pass"]), "stats reads", len(stats))


if __name__ == "__main__":
    main()
