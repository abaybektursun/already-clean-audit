"""N1 spec section 7 step 3: the freeze-half read before freeze 2 (laptop; banked JSON only; no test-half record opened).

    PYTHONPATH=. python studies/next/n1/read/freeze_read.py --bank studies/next/n1/banks/weights-03 \
        --out studies/next/n1/read/freeze_read.json

Answers, from n1/metrics.json, n1/n1_freeze.jsonl, n1/samples/raw_examples_freeze.jsonl, studies/next/n1/kb_bank.json and
studies/next/n1/samples/raw_examples_zero_gpu.json:
  1. does each example unit's column norm at T_(k-1) and T_k equal the banked one: the banked values of those units and
     the K-B report (every banked value compared) at those points;
  2. does r over the 15 brackets follow the bracket lengths for carriers and twins alike: Spearman of r against the
     bracket's step count, per unit, median by kind;
  3. PC: the planted share and the size-0 share over brackets 2-15;
  4. the example birth with the largest and the one with the smallest input-side excess, with an exemplar.
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import m2

ROOT = pathlib.Path(__file__).resolve().parents[4]
BSTEPS = np.diff(m2.STEPS)                   # bracket j = 1..15 at index j - 1


def ranks(a):
    """Ranks with ties averaged (brackets 1-3 are 1,000 steps each)."""
    a = np.asarray(a, np.float64)
    o = np.argsort(a, kind="stable")
    r = np.empty(a.size)
    r[o] = np.arange(a.size)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bank = pathlib.Path(a.bank) / "n1"
    jm = json.loads((bank / "metrics.json").read_text())
    exs = m2.read_jsonl(bank / "samples" / "raw_examples_freeze.jsonl")
    fz = m2.read_jsonl(bank / "n1_freeze.jsonl")
    assert all(r["half"] == "freeze" for r in fz)
    kb = {(x[0], x[1], x[2]): x[3] for x in json.loads((ROOT / "studies/next/n1/kb_bank.json").read_text())["values"]}
    zg = {x["birth"]: x for x in json.loads((ROOT / "studies/next/n1/samples/raw_examples_zero_gpu.json").read_text())}
    rep = jm.get("reports", {})
    out = {"colnorm": [], "profile": [], "examples": []}
    for e in exs:
        b = e["birth"]
        for kind in ("real", "dt", "pb"):
            if kind not in e:
                continue
            r = e[kind]
            assert r["half"] == "freeze"
            k, l, n = r["k"], r["layer"], r["neuron"]
            row = {"unit": r["unit"], "k": k}
            for t, nm in ((k - 1, "pre"), (k, "birth")):
                row[f"bank_{nm}"] = kb.get((l, n, t))
                kr = rep.get(f"t{t:02d}_K_B")
                row[f"KB_t{t:02d}"] = None if kr is None else {"pass": kr["pass"], "max_rel": kr["max_rel"], "n": kr["n"]}
            if zg[b].get(kind):
                row["zero_gpu_colnorm_pre_birth"] = [zg[b][kind]["colnorm_pre"], zg[b][kind]["colnorm_birth"]]
            out["colnorm"].append(row)
            out["profile"].append({"unit": r["unit"], "kind": kind, "k": k,
                                   "rho_in": spearman(r["in"]["r"], BSTEPS), "rho_out": spearman(r["out"]["r"], BSTEPS)})
        rr = e["real"]
        k = rr["k"]
        out["examples"].append({
            "birth": b, "k": k, "e_window": {s: rr[s]["e_window"] for s in ("in", "out", "gate", "up")},
            "e_k": {s: rr[s]["e_k"] for s in ("in", "out")}, "e_by_bracket": {s: rr[s]["e_by_bracket"] for s in ("in", "out")},
            "z_window": {s: [rr[s]["z"][j - 1] for j in (k - 2, k - 1, k)] for s in ("in", "out")},
            "r_window": {s: [rr[s]["r"][j - 1] for j in (k - 2, k - 1, k)] for s in ("in", "out")},
            "cos_k": {s: rr[s]["cos"][k - 1] for s in ("in", "out")},
            "twins_e_window": {kind: {s: e[kind][s]["e_window"] for s in ("in", "out")} for kind in ("dt", "pb") if kind in e},
            "ceiling": rr["ceiling"], "exemplar": zg[b]["exemplar"][1]})
    pcs = {g: v for g, v in jm["gates"].items() if g.endswith("_PC")}
    out["pc"] = {"n_brackets": len(pcs), "planted_min": min(min(v["in"], v["out"]) for v in pcs.values()),
                 "size0_min": min(min(v["in_size0"], v["out_size0"]) for v in pcs.values()),
                 "size0_max": max(max(v["in_size0"], v["out_size0"]) for v in pcs.values())}
    ok = [c for c in out["colnorm"] for t in ("pre", "birth") if c[f"bank_{t}"] is not None]
    zq = [c for c in out["colnorm"] if "zero_gpu_colnorm_pre_birth" in c]
    out["colnorm_summary"] = {"example_units": len(out["colnorm"]), "unit_points": 2 * len(out["colnorm"]),
                              "unit_points_with_banked_values": len(ok),
                              "units_zero_gpu_file_equal_to_bank": sum(c["zero_gpu_colnorm_pre_birth"] ==
                                                                       [c["bank_pre"], c["bank_birth"]] for c in zq),
                              "units_in_zero_gpu_file": len(zq),
                              "KB_max_rel_at_these_points": max(c[x]["max_rel"] for c in out["colnorm"] for x in c
                                                                if x.startswith("KB_t") and c[x] is not None),
                              "KB_points_all_pass": all(c[x]["pass"] for c in out["colnorm"] for x in c
                                                        if x.startswith("KB_t") and c[x] is not None)}
    for kind in ("real", "dt", "pb"):
        pr = [p for p in out["profile"] if p["kind"] == kind]
        out[f"profile_median_rho_{kind}"] = {"n": len(pr), "in": float(np.median([p["rho_in"] for p in pr])),
                                             "out": float(np.median([p["rho_out"] for p in pr]))}
    ex = sorted(out["examples"], key=lambda x: x["e_window"]["in"])
    out["largest_in"] = {"birth": ex[-1]["birth"], "e_in": ex[-1]["e_window"]["in"], "exemplar": ex[-1]["exemplar"]}
    out["smallest_in"] = {"birth": ex[0]["birth"], "e_in": ex[0]["e_window"]["in"], "exemplar": ex[0]["exemplar"]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("colnorm", "profile", "examples")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
