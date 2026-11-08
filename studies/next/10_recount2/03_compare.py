"""Compare the recount with the study's banked dates (w5/verdict/samples/q2_clusters.jsonl), then run the
same comparison on planted failing inputs to show that the comparison can fail.

Outputs: out/03_compare.json (counts, mismatch table, failing-input results).
"""
import json, os
import numpy as np
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = C.load_arrays()
mine = [json.loads(l) for l in open(C.BIG + "/births.jsonl")]
mine0 = {f"{l['pole']}:c{l['c']}": l for l in mine if l["sample"] == 0}
banked = {}
for l in open(C.BANKED):
    r = json.loads(l)
    banked[r["cluster"]] = {k: r[k] for k in ["pole", "c", "echo", "ranks", "med", "present_pct", "present_raw",
                                              "birth_pct", "birth_raw", "is_main", "status", "peak_end"]}
bk = {k: v for k, v in banked.items() if not v["echo"]}
res = {"n_banked_rows": len(banked), "n_banked_echo": len(banked) - len(bk), "n_banked_non_echo": len(bk),
       "n_mine_sample0": len(mine0), "keys_only_in_mine": sorted(set(mine0) - set(bk)),
       "keys_only_in_banked": sorted(set(bk) - set(mine0))}
keys = sorted(set(mine0) & set(bk))
res["n_common"] = len(keys)
res["ranks_equal"] = sum(mine0[k]["ranks"] == bk[k]["ranks"] for k in keys)
medd = np.array([np.abs(np.array(mine0[k]["med"]) - np.array(bk[k]["med"])).max() for k in keys])
res["med_max_abs_diff"] = float(medd.max())
res["present_pct_equal"] = sum(mine0[k]["present_pct"] == bk[k]["present_pct"] for k in keys)
res["present_raw_equal"] = sum(mine0[k]["present_raw"] == bk[k]["present_raw"] for k in keys)
res["birth_pct_equal"] = sum(mine0[k]["birth_pct"] == bk[k]["birth_pct"] for k in keys)
res["birth_raw_equal"] = sum(mine0[k]["birth_raw"] == bk[k]["birth_raw"] for k in keys)
res["is_main_equal"] = sum(mine0[k]["is_main"] == bk[k]["is_main"] for k in keys)
res["birth_pct_lower_median_equal"] = sum(mine0[k]["birth_pct_lower"] == bk[k]["birth_pct"] for k in keys)
res["birth_pct_upper_median_equal"] = sum(mine0[k]["birth_pct_upper"] == bk[k]["birth_pct"] for k in keys)
res["mismatches"] = [{"cluster": k, "field": f, "mine": mine0[k][f], "banked": bk[k][f]}
                     for k in keys for f in ["birth_pct", "birth_raw", "is_main"] if mine0[k][f] != bk[k][f]]
res["n_banked_birth_raw_none"] = sum(bk[k]["birth_raw"] is None for k in keys)


def match_rate(births):
    """births: dict cluster key -> date. Exact matches with the banked birth_pct."""
    return sum(births[k] == bk[k]["birth_pct"] for k in keys)


def redate(a_plus, q999, ranks_of):
    out = {}
    for k in keys:
        i = A["row"][mine0[k]["pole"]]
        med = np.median(a_plus[:, i, ranks_of(k)], axis=1)
        out[k] = C.birth_from_present((med > 0) & (med >= q999[:, i]))
    return out


rng = np.random.default_rng(20261105)
fail = {}
# sanity: the same routine on the true input reproduces the recount
fail["true_input"] = match_rate(redate(A["a_plus"], A["q999"], lambda k: mine0[k]["ranks"]))
# (a) checkpoints in a shuffled order (activations and thresholds moved together)
perm = rng.permutation(16)
fail["a_checkpoints_shuffled"] = {"perm": perm.tolist(),
                                  "match": match_rate(redate(A["a_plus"][perm], A["q999"][perm],
                                                             lambda k: mine0[k]["ranks"]))}
# (b) pole rows shifted by two (a + pole read as the next neuron's + pole)
fail["b_pole_rows_shifted_by_2"] = {"match": match_rate(redate(np.roll(A["a_plus"], 2, axis=1),
                                                               np.roll(A["q999"], 2, axis=1),
                                                               lambda k: mine0[k]["ranks"]))}
# (c) exemplar ranks replaced by a random same-size draw from the pole's non-echo ranks
nonecho = {}
for k in keys:
    nonecho.setdefault(mine0[k]["pole"], []).extend(mine0[k]["ranks"])
draws = {k: sorted(rng.choice(nonecho[mine0[k]["pole"]], size=len(mine0[k]["ranks"]), replace=False).tolist())
         for k in keys}
fail["c_random_ranks_same_size"] = {"match": match_rate(redate(A["a_plus"], A["q999"], lambda k: draws[k]))}
# (d) every checkpoint's threshold replaced by the T_15 threshold
fail["d_threshold_from_T15"] = {"match": match_rate(redate(A["a_plus"], np.repeat(A["q999"][15:16], 16, axis=0),
                                                           lambda k: mine0[k]["ranks"]))}
# (e) median replaced by the maximum over the exemplar set
def redate_max():
    out = {}
    for k in keys:
        i = A["row"][mine0[k]["pole"]]
        med = A["a_plus"][:, i, mine0[k]["ranks"]].max(axis=1)
        out[k] = C.birth_from_present((med > 0) & (med >= A["q999"][:, i]))
    return out
fail["e_max_instead_of_median"] = {"match": match_rate(redate_max())}
# (f) one-point rule: birth = first t present (no next-point requirement)
def redate_onepoint():
    out = {}
    for k in keys:
        i = A["row"][mine0[k]["pole"]]
        med = np.median(A["a_plus"][:, i, mine0[k]["ranks"]], axis=1)
        p = (med > 0) & (med >= A["q999"][:, i])
        out[k] = int(np.argmax(p)) if p.any() else None
    return out
fail["f_one_point_rule"] = {"match": match_rate(redate_onepoint())}
res["failing_inputs"] = fail
json.dump(res, open(OUT + "/03_compare.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))
print("n mismatches:", len(res["mismatches"]))
for m in res["mismatches"][:40]:
    print(m)
