"""Shared loading and the dating rule as read from SPEC_D1.md lines 280-292 (see 10_RECOUNT2.md section 2)."""
import json
import numpy as np

BANK = "/Users/abaybektursun/projects/interp_research_banks/neurons"
BIG = BANK + "/next/recount2"
CLUSTERS = BANK + "/w4/o7_s_v2/explain/clusters.jsonl"
BANKED = BANK + "/w5/verdict/samples/q2_clusters.jsonl"
NT = 16


def load_arrays():
    z = np.load(BIG + "/arrays.npz")
    poles = [str(p) for p in z["poles"]]
    sgn = np.array([1.0 if p.endswith("+") else -1.0 for p in poles])
    # a_plus[t, i, k] = max(s_i * a_t, 0): activation in the pole's own direction (SPEC line 281)
    a_plus = np.maximum(z["a"] * sgn[None, :, None], 0.0)
    return {"poles": poles, "row": {p: i for i, p in enumerate(poles)}, "sgn": sgn, "a": z["a"],
            "a_plus": a_plus, "q999": z["q999"], "max_abs": z["max_abs"], "tok15": z["tok15"]}


def load_clusters():
    return [json.loads(l) for l in open(CLUSTERS)]


def medians(a_plus_i, ranks, how="mid"):
    """Median over the exemplar set at every t. a_plus_i: [16, 50] for one pole. Returns [16]."""
    v = a_plus_i[:, ranks]
    if how == "mid":
        return np.median(v, axis=1)
    s = np.sort(v, axis=1)
    n = len(ranks)
    if how == "lower":      # the smaller middle value when n is even
        return s[:, (n - 1) // 2]
    if how == "upper":      # the larger middle value when n is even
        return s[:, n // 2]
    raise ValueError(how)


def birth_from_present(present):
    """First t with present at t and t+1; T_15 alone gives T_15; None if never (SPEC lines 284-286)."""
    for t in range(NT - 1):
        if present[t] and present[t + 1]:
            return t
    if present[NT - 1]:
        return NT - 1
    return None


def present_pct(med, q999_i):
    return (med > 0) & (med >= q999_i)


def present_raw(med, max_abs_i):
    return (med > 0) & (med >= 0.3 * max_abs_i)


def date_group(A, i, ranks, how="mid"):
    med = medians(A["a_plus"][:, i, :], ranks, how)
    return birth_from_present(present_pct(med, A["q999"][:, i]))


def births_matrix(a_plus_i, q999_i, groups):
    """Vectorized percentile-reading birth for many groups of one pole.
    groups: int array [G, n] of ranks (same size n). Returns int array [G] with 99 for never born."""
    v = a_plus_i[:, groups]                     # [16, G, n]
    med = np.median(v, axis=2)                  # [16, G]
    pres = (med > 0) & (med >= q999_i[:, None])
    two = pres[:-1] & pres[1:]                  # [15, G]
    out = np.full(groups.shape[0], 99, dtype=np.int64)
    has15 = pres[-1]
    out[has15] = 15
    any_two = two.any(axis=0)
    first_two = np.argmax(two, axis=0)
    out[any_two] = first_two[any_two]
    return out
