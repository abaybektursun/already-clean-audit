"""Reader of M2, phase 1, step 5b: the paired real - twin difference under B, split by the real birth's share of
distinct current tokens at its exemplar positions (a mechanical coherence proxy; 1/size = one token everywhere).

Input: read/r5_twins.json (rows). Output: read/r5b_token_share.json.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r2_from_ranks import boot

M2 = Path(__file__).resolve().parents[1]


def cell(R):
    if not R:
        return {"n": 0}
    hr = np.array([x["real"]["hit_B"] for x in R], float)
    ht = np.array([x["pb"]["hit_B"] for x in R], float)
    m, ci = boot(hr - ht)
    return {"n": len(R), "real_hits": int(hr.sum()), "twin_hits": int(ht.sum()), "diff": m, "diff_ci": ci,
            "real_B10_median": float(np.median([x["real"]["B10_listed"] for x in R])),
            "twin_B10_median": float(np.median([x["pb"]["B10_listed"] for x in R])),
            "real_own_B_median": float(np.median([x["real"]["B"] for x in R])),
            "twin_own_B_median": float(np.median([x["pb"]["B"] for x in R])),
            "real_other_slots_per_pos_median": float(np.median([x["real"]["n_other_pole_slots_per_pos"] for x in R]))}


def main():
    rows = json.loads((M2 / "read/r5_twins.json").read_text())["rows"]
    out = {}
    for h in ("test", "freeze", "both"):
        R = rows if h == "both" else [x for x in rows if x["half"] == h]
        lo = [x for x in R if x["real"]["distinct_token_share"] < 0.67]
        hi = [x for x in R if x["real"]["distinct_token_share"] >= 0.67]
        out[h] = {"all": cell(R), "real_share_lt_0.67": cell(lo), "real_share_ge_0.67": cell(hi)}
    # within real births: hit rate against the share, all 149 with a twin
    out["examples_low_share"] = [[x["birth"], x["real"]["distinct_token_share"], x["real"]["rank_B"]]
                                 for x in rows if x["real"]["distinct_token_share"] < 0.34][:12]
    (M2 / "read/r5b_token_share.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
