"""Date every non-echo cluster of every pole and clustering sample with the registered rule.

Outputs: <BIG>/births.jsonl (one line per cluster, both samples, with per-t medians and presence),
out/births.pointer.json, out/births_sample0.csv (pole, c, size, min_rank, is_main, birth_pct, birth_raw,
births under the lower and upper median), out/02_counts.json.
"""
import json, os, hashlib
import numpy as np
from collections import Counter
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = C.load_arrays()
rows = C.load_clusters()
lines, csv = [], ["pole,sample,c,size,min_rank,is_main,birth_pct,birth_raw,birth_pct_lower,birth_pct_upper"]
cnt = Counter()
mdist = {0: Counter(), 1: Counter()}
alt_reading = 0
for r in rows:
    i = A["row"][r["pole"]]
    s = r["sample"]
    cl = [c for c in r["clusters"] if not c["echo"]]
    cnt[f"sample{s}_echo"] += len(r["clusters"]) - len(cl)
    cnt[f"sample{s}_poles"] += 1
    mdist[s][len(cl)] += 1
    if not cl:
        cnt[f"sample{s}_poles_all_echo"] += 1
        continue
    peak = [float(np.abs(A["a"][15, i, c["exemplar_ranks"]]).max()) for c in cl]
    main_c = cl[int(np.argmax(peak))]["c"]
    assert sorted(peak)[-1] > (sorted(peak)[-2] if len(peak) > 1 else -1), "tie for main"
    for c, pk in zip(cl, peak):
        R = list(c["exemplar_ranks"])
        med = C.medians(A["a_plus"][:, i, :], R)
        pp = C.present_pct(med, A["q999"][:, i])
        prw = C.present_raw(med, A["max_abs"][:, i])
        b_pct = C.birth_from_present(pp)
        b_raw = C.birth_from_present(prw)
        b_lo = C.date_group(A, i, R, "lower")
        b_up = C.date_group(A, i, R, "upper")
        # the stricter reading of "first present at T_15": present somewhere before 15 but never two in a row
        if b_pct == 15 and pp[:15].any():
            alt_reading += (s == 0)
        cnt[f"sample{s}_clusters"] += 1
        if b_pct is None:
            cnt[f"sample{s}_never_born_pct"] += 1
        lines.append({"pole": r["pole"], "sample": s, "c": c["c"], "size": len(R), "ranks": R,
                      "min_rank": min(R), "peak_end": pk, "is_main": c["c"] == main_c,
                      "med": [float(x) for x in med], "q999": [float(x) for x in A["q999"][:, i]],
                      "present_pct": [bool(x) for x in pp], "present_raw": [bool(x) for x in prw],
                      "birth_pct": b_pct, "birth_raw": b_raw, "birth_pct_lower": b_lo, "birth_pct_upper": b_up})
        csv.append(",".join(str(x) for x in [r["pole"], s, c["c"], len(R), min(R), int(c["c"] == main_c),
                                               b_pct, b_raw, b_lo, b_up]))

with open(C.BIG + "/births.jsonl", "w") as f:
    for l in lines:
        f.write(json.dumps(l) + "\n")
open(OUT + "/births_all.csv", "w").write("\n".join(csv) + "\n")
s0 = [l for l in lines if l["sample"] == 0]
counts = dict(cnt)
counts["m_dist_sample0"] = dict(sorted(mdist[0].items()))
counts["m_dist_sample1"] = dict(sorted(mdist[1].items()))
counts["sample0_poles_m_ge_2"] = sum(v for k, v in mdist[0].items() if k >= 2)
counts["sample0_birth_pct_hist"] = dict(sorted(Counter(str(l["birth_pct"]) for l in s0).items()))
counts["sample0_main_birth_pct_hist"] = dict(sorted(Counter(str(l["birth_pct"]) for l in s0 if l["is_main"]).items()))
counts["sample0_main_min_rank_hist"] = dict(sorted(Counter(l["min_rank"] for l in s0 if l["is_main"]).items()))
counts["sample0_birth15_with_earlier_isolated_presence"] = alt_reading
counts["sample0_lower_median_changes"] = sum(l["birth_pct_lower"] != l["birth_pct"] for l in s0)
counts["sample0_upper_median_changes"] = sum(l["birth_pct_upper"] != l["birth_pct"] for l in s0)
json.dump(counts, open(OUT + "/02_counts.json", "w"), indent=1)
json.dump({"path": C.BIG + "/births.jsonl", "bytes": os.path.getsize(C.BIG + "/births.jsonl"),
           "sha256": hashlib.sha256(open(C.BIG + "/births.jsonl", "rb").read()).hexdigest(),
           "written_by": "studies/next/10_recount2/02_births.py"},
          open(OUT + "/births.pointer.json", "w"), indent=1)
print(json.dumps(counts, indent=1))
