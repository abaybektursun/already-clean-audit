"""N6 (J11), edit the young: the pure logic of studies/next/07_PICKS/N6_SPEC.md. numpy only; no model, no torch.

    pairs       the dated poles, the frozen caliper rule, greedy matching, balance, the split, the eval windows -> pairs.json
    projection  GPU-hours of L1 and L2 from FLOP counts at three fp32 rates -> projection.json
    freeze      thresholds from the freeze half of the job's records -> thresholds.json            (freeze 2)
    verdict     the test half, read once with thresholds.json -> verdict_n6.json
    (amendment A5: L2 runs as two launches, L2a on the freeze half and L2b on the test half; freeze reads L2a's bank, the verdict
    reads L2a's and L2b's banks as one, read_banks)

    PYTHONPATH=. python3 -m infra.pg.n6 pairs --out studies/next/n6/pairs.json
    PYTHONPATH=. python3 -m infra.pg.n6 projection --pairs studies/next/n6/pairs.json --out studies/next/n6/projection.json
    PYTHONPATH=. python3 -m infra.pg.n6 freeze --bank <L2a's out/n6 dir> --pairs <pairs.json> --out <thresholds.json>
    PYTHONPATH=. python3 -m infra.pg.n6 verdict --bank <L2a's dir> [--bank <L2b's dir>] --pairs <pairs.json> --thresholds <t.json> --out <v.json>

Objects (spec section 0.5). A pole is (layer l, neuron n, sign s). Its age is the birth index of its main cluster under the D1 percentile
reading (J1's recount, births_all.csv, sample 0). Its adjusted age is the age minus the R1t reference mean date of the main cluster
(05_per_cluster_R1t.csv; 0 for the 26 poles with one cluster). Matching quantities from the T_15 bank: ln max|a| and ln density.
Fail loud: asserts carry their reason; no try/except.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib

import numpy as np

# ------------------------------------------------------------------------------------------ inputs (sha256 registered)
BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
RECOUNT = pathlib.Path("studies/next/10_recount2/out")
FILES = {
    "births": RECOUNT / "births_all.csv",
    "ref": RECOUNT / "05_per_cluster_R1t.csv",
    "ctx15": BANK / "w3/grid/1/out/t15_step1413814/samples/contexts.jsonl",
    "arrays": BANK / "next/recount2/arrays.npz",
    "e1_samples": pathlib.Path("studies/instruments/results/e1/1/t15/samples/e1.jsonl"),
    "e1_items": pathlib.Path("studies/instruments/inputs/e1_items.json"),
}
SHA = {
    "births": "439b66118c4de13c5a4f2211a2e3de905268d3bcce0c35c4bd605e3dbe6f28b1",
    "ref": "1083132bedf0f4e1a675205c0e8314b4f016a88361718cd85b6f3f1517bc1532",
    "ctx15": "a4d46cd837f7a7532c300cac7372df1908e37f25b5acf57b6ea1ea37493e3269",
    "arrays": "80e5ffac44ea44b3d7794f3e57bd51aa67ba5b6b8508985d60d1b12d6c6ddba4",
    "e1_samples": "3967eda52f61e7c2d1ec61f76aac8f5a9f5d66aa3b36cb810e90831290bda608",
    "e1_items": "21b132a08479ca6e7c529cbefc056e6e14394191b4330612c950c0300947f83d",
}
E1_PICK = {"pair": "PL05_00", "side": "J", "value": "zero", "construction": "u"}   # the own arm is the known number, cross its failing input
N_LAYERS, WIDTH, HIDDEN, VOCAB = 32, 11008, 4096, 100278
REPO, REVISION, SNAPSHOT = "allenai/Olmo-3-1025-7B", "stage1-step1413814", "373bad25002f1624757a73235c5ca844c6375c25"

# ------------------------------------------------------------------------------------------ the pairing rule (spec section 2.3)
DMIN_AGE = 3                                         # grid points between the two main-cluster births
GRID_MX = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)       # caliper on |d ln max|a||
GRID_DEN = (0.1, 0.2, 0.3, 0.4, 0.5)                 # caliper on |d ln density|
RHO_MAX = 0.10                                       # |rho_pair(age, ln max|a|)| and |rho_pair(age, ln density)|
SMD_MAX = 0.10                                       # |standardized mean difference| of ln max|a| and ln density

# ------------------------------------------------------------------------------------------ the eval set (spec section 2.2)
HW = 2048
BAND3_TOK = (20_570_743, 50_540_791)                 # corpus manifest bands["3"]["tok_start"/"tok_end"]["olmo3"]
W_EVAL = 96                                          # stage 1 and L1 (freeze 1); G-INST's windows at L2
W_EVAL_L2 = 1024                                     # amendment A4: L2's eval set, the same rule (eval_windows) at 1,024 windows
KNOWN_NLL_TOKENS = 120_000

# ------------------------------------------------------------------------------------------ rules (spec section 6)
PC_LAYER_STEP = 4                                    # amendment A4: the positive control in every fourth layer, 0, 4, ..., 28
PC_MIN = 6                                           # A4: of 8, P(X >= 6 | n = 8, p = 0.5) = 37/256 = 0.145 (24 of 32 at freeze 1)
FLAT_M = 0.20                                        # FLAT band [0.30, 0.70]
R_MIN = 0.50                                         # FLAT needs half the test pairs resolved at 2 tau_s
TIE_MULT = 10.0                                      # tau_eps = 10 x F_eps
EPS_ABS = 1e-6                                       # nats per token
EPS_REL = 0.02                                       # A4: F_eps <= q10 / 50 of |dL_col| over the freeze half's pair edits (q10 / 100 at freeze 1)
POWER_MAX = 0.5                                      # A4: tau_s <= 0.5 x median |D_col| over the freeze pairs, or UNDECIDED_NO_POWER
PARK_MAX = 0.20
N_BOOT = 10_000
N_BOOT_WIN = 1_000
SEED = 0


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pole_id(layer: int, neuron: int, sign: int) -> str:
    return f"L{layer:02d}N{neuron}{'+' if sign > 0 else '-'}"


def parse_pole(p: str) -> tuple[int, int, int]:
    assert p[0] == "L" and p[-1] in "+-", p
    return int(p[1:3]), int(p[4:-1]), 1 if p[-1] == "+" else -1


# ------------------------------------------------------------------------------------------ statistics
def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def stratified_spearman(x, y, strata) -> float:
    """Rank x and y within each stratum, centre within the stratum, correlate the pooled values (the critic's within-layer Spearman,
    06_CRITIC.md N6; with each pair as its stratum it is 1 - 2 f, f the share of pairs whose earlier member has the larger y)."""
    x, y, strata = np.asarray(x, float), np.asarray(y, float), np.asarray(strata)
    rx, ry = np.zeros(len(x)), np.zeros(len(x))
    for s in np.unique(strata):
        m = strata == s
        rx[m] = ranks(x[m]) - ranks(x[m]).mean()
        ry[m] = ranks(y[m]) - ranks(y[m]).mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    assert d > 0, "no variation within strata"
    return float((rx * ry).sum() / d)


def within_sd(x, strata) -> float:
    x, strata = np.asarray(x, float), np.asarray(strata)
    return float(math.sqrt(np.mean([x[strata == s].var() for s in np.unique(strata)])))


def wilson(k: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval of k successes in n (k may hold half counts for ties)."""
    assert n > 0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def win_share(d: np.ndarray, tau: float) -> tuple[float, int]:
    """Share of pairs with d > tau, ties (|d| <= tau) counted one half; returns (count, n)."""
    d = np.asarray(d, float)
    return float((d > tau).sum() + 0.5 * (np.abs(d) <= tau).sum()), len(d)


def boot_ci(f, n: int, seed: int = SEED, b: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.array([f(rng.integers(0, n, n)) for _ in range(b)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------------------------------ the dated poles and the pairs
def check_inputs() -> dict:
    out = {}
    for k, p in FILES.items():
        assert p.exists(), f"missing input {p}"
        got = sha256_file(p)
        assert got == SHA[k], f"{p}: sha256 {got} is not the registered {SHA[k]}"
        out[k] = {"path": str(p), "sha256": got}
    return out


def load_dated() -> list:
    """The 600 dated poles of sample 0 with age, adjusted age, the T_15 bank statistics and the 50 exemplars."""
    main = {}
    for r in csv.DictReader(open(FILES["births"])):
        if r["sample"] == "0" and r["is_main"] == "1":
            assert r["pole"] not in main, f"two mains for {r['pole']}"
            main[r["pole"]] = int(r["birth_pct"])
    ref = {r["pole"]: float(r["ref_mean"]) for r in csv.DictReader(open(FILES["ref"])) if r["is_main"] == "True"}
    ctx = {}
    for line in open(FILES["ctx15"]):
        r = json.loads(line)
        ctx[r["pole"]] = r
    z = np.load(FILES["arrays"])
    row = {str(p): i for i, p in enumerate(z["poles"])}
    out = []
    for p in sorted(main, key=lambda q: (parse_pole(q)[0], parse_pole(q)[1], -parse_pole(q)[2])):
        lyr, n, s = parse_pole(p)
        c = ctx[p]
        assert c["max_abs"] > 0 and c["density"] > 0, f"{p} has no activity at T_15"
        i = row[p]
        tok = z["tok15"][i].astype(np.int64)
        a15 = z["a"][15, i].astype(np.float64)
        assert np.array_equal(tok, np.array([k["tok"] for k in c["topk"]])), f"{p}: arrays.npz positions differ from contexts.jsonl"
        assert np.array_equal(a15, np.array([k["a"] for k in c["topk"]])), f"{p}: arrays.npz T_15 a differs from contexts.jsonl"
        out.append({"pole": p, "layer": lyr, "neuron": n, "sign": s, "age": main[p], "ref_mean": ref.get(p),
                    "adj": main[p] - ref[p] if p in ref else 0.0, "max_abs": float(c["max_abs"]), "density": int(c["density"]),
                    "q999": float(c["q999"]), "row": i, "tok15": tok.tolist(), "a15": a15.tolist(),
                    "a14": z["a"][14, i].astype(np.float64).tolist()})
    assert len(out) == 600, len(out)
    return out


def covariates(poles: list, inst: dict | None = None) -> dict:
    """ln max|a| and ln density from the T_15 bank; with `inst` (L1's arrays) also ln ||w|| and ln E+ on the eval windows."""
    out = {"ln_max_abs": np.log([p["max_abs"] for p in poles]), "ln_density": np.log([p["density"] for p in poles])}
    if inst is not None:
        out["ln_colnorm"] = np.log([inst["colnorms"][p["layer"], p["neuron"]] for p in poles])
        out["ln_E_plus"] = np.log([inst["energies"][p["layer"], p["neuron"], 0 if p["sign"] > 0 else 1] for p in poles])
    return out


def match(poles: list, cal: dict, cov: dict, dmin: int = DMIN_AGE) -> list:
    """Greedy matching within a layer: candidate edges between poles of different neurons with |d age| >= dmin and |d x| <= cal[x]
    for every caliper; edges sorted by d = sum_x |d x| / cal[x], ties by (row, row); each pole in at most one pair. Returns
    [(early index, late index)] with early the smaller age."""
    keys = sorted(cal)
    cand = []
    for i, p in enumerate(poles):
        for j in range(i + 1, len(poles)):
            q = poles[j]
            if p["layer"] != q["layer"] or p["neuron"] == q["neuron"] or abs(p["age"] - q["age"]) < dmin:
                continue
            dx = [abs(cov[k][i] - cov[k][j]) for k in keys]
            if all(d <= cal[k] for d, k in zip(dx, keys)):
                cand.append((sum(d / cal[k] for d, k in zip(dx, keys)), i, j))
    cand.sort()
    used, out = set(), []
    for _, i, j in cand:
        if i in used or j in used:
            continue
        used |= {i, j}
        out.append((i, j) if poles[i]["age"] < poles[j]["age"] else (j, i))
    return out


def balance(poles: list, pairs: list, cov: dict) -> dict:
    """Balance of the matched set. rho_pair: the critic's statistic with each pair as its stratum; smd: mean over pairs of
    (early - late) / the within-layer sd over all 600 dated poles; pooled: the critic's within-layer statistic on the matched poles."""
    lay = np.array([p["layer"] for p in poles])
    age = np.array([p["age"] for p in poles], float)
    E, La = np.array([e for e, _ in pairs]), np.array([l_ for _, l_ in pairs])
    idx = np.r_[E, La]
    strata = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
    out = {"n_pairs": len(pairs)}
    for k, v in cov.items():
        out[k] = {"rho_pair": stratified_spearman(age[idx], v[idx], strata),
                  "smd": float((v[E] - v[La]).mean() / within_sd(v, lay)),
                  "pooled_within_layer": stratified_spearman(age[idx], v[idx], lay[idx]),
                  "mean_abs_diff": float(np.abs(v[E] - v[La]).mean())}
    out["unmatched_all_600"] = {k: stratified_spearman(age, v, lay) for k, v in cov.items()}
    return out


def balanced(b: dict, rho_keys=("ln_max_abs", "ln_density"), smd_keys=None) -> bool:
    """|rho_pair| <= RHO_MAX on the bank quantities (the card's gate); |SMD| <= SMD_MAX on every matched quantity."""
    smd_keys = smd_keys or [k for k in ("ln_max_abs", "ln_density", "ln_colnorm", "ln_E_plus") if k in b]
    return all(abs(b[k]["rho_pair"]) <= RHO_MAX for k in rho_keys) and all(abs(b[k]["smd"]) <= SMD_MAX for k in smd_keys)


def caliper_scan(poles: list, cov: dict, grid: dict, fixed: dict | None = None) -> dict:
    """The frozen design rule: over the grid, the point with the most pairs whose matched set is balanced; ties to the smaller
    calipers (in key order). `fixed` holds calipers set at an earlier stage."""
    import itertools
    keys = sorted(grid)
    rows = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cal = {**(fixed or {}), **dict(zip(keys, vals))}
        pr = match(poles, cal, cov)
        b = balance(poles, pr, cov) if len(pr) >= 10 else None
        rows.append({"cal": cal, "n_pairs": len(pr), "balanced": bool(b is not None and balanced(b)),
                     **({} if b is None else {f"{k}_{m}": b[k][m] for k in cov for m in ("rho_pair", "smd")})})
    ok = [r for r in rows if r["balanced"]]
    if not ok:
        return {"grid": rows, "chosen": None}
    best = sorted(ok, key=lambda r: (-r["n_pairs"], [r["cal"][k] for k in keys]))[0]
    return {"grid": rows, "chosen": best["cal"]}


def unmatched_pairs(poles: list) -> list:
    """The registered failing input of the balance gate: same layer, different neurons, |d age| >= 3, no caliper, greedy by the
    largest age gap (ties by row)."""
    cand = []
    for i, p in enumerate(poles):
        for j in range(i + 1, len(poles)):
            q = poles[j]
            if p["layer"] == q["layer"] and p["neuron"] != q["neuron"] and abs(p["age"] - q["age"]) >= DMIN_AGE:
                cand.append((-abs(p["age"] - q["age"]), i, j))
    cand.sort()
    used, out = set(), []
    for _, i, j in cand:
        if i in used or j in used:
            continue
        used |= {i, j}
        out.append((i, j) if poles[i]["age"] < poles[j]["age"] else (j, i))
    return out


def eval_windows(w_eval: int = W_EVAL) -> list:
    """w_eval harvest-aligned windows lying wholly in band 3, spread evenly over it (band 3 is ordered by directory), ascending."""
    w0 = -(-BAND3_TOK[0] // HW)
    w1 = BAND3_TOK[1] // HW                           # exclusive: windows w0 .. w1 - 1 are full
    n = w1 - w0
    out = [w0 + (i * n) // w_eval for i in range(w_eval)]
    assert len(set(out)) == w_eval and out[0] * HW >= BAND3_TOK[0] and (out[-1] + 1) * HW <= BAND3_TOK[1]
    return out


GRID_BANK = {"ln_max_abs": GRID_MX, "ln_density": GRID_DEN}
GRID_INST = {"ln_colnorm": (0.05, 0.10, 0.15, 0.20, 0.30), "ln_E_plus": (0.25, 0.50, 0.75, 1.00)}
MIN_PAIRS_L2 = 40


SPLIT_L1 = "freeze when the early member's neuron index is even"
SPLIT_L2 = ("amendment A3: test when the early member's neuron index is even, freeze when odd; a pair holding a pole that L1 edited "
            "is freeze")


def half_of(pe: dict, pl: dict, l1_edited) -> str:
    """The split (spec section 6.1). Stage 1 (l1_edited None): freeze when the early member's neuron index is even. Stage 2,
    amendment A3: the halves swapped (test when even), and a pair holding a pole that L1 edited stays in the freeze half."""
    if l1_edited is None:
        return "freeze" if pe["neuron"] % 2 == 0 else "test"
    return "freeze" if pe["neuron"] % 2 == 1 or {pe["pole"], pl["pole"]} & set(l1_edited) else "test"


def pairs_doc(poles, pr, cov, scan, stage, extra=None, l1_edited=None) -> dict:
    bal = balance(poles, pr, cov)
    assert balanced(bal), bal
    un = unmatched_pairs(poles)
    pairs = []
    for k, (e, l_) in enumerate(pr):
        pe, pl = poles[e], poles[l_]
        pairs.append({"pair": f"P{k:03d}", "layer": pe["layer"], "half": half_of(pe, pl, l1_edited),
                      "early": pe["pole"], "late": pl["pole"], "age_early": pe["age"], "age_late": pl["age"],
                      "adj_early": pe["adj"], "adj_late": pl["adj"], "concordant_adj": bool(pe["adj"] <= pl["adj"] - 1.0)})
    lay = np.array([p["layer"] for p in poles])
    adj = np.array([p["adj"] for p in poles])
    age = np.array([p["age"] for p in poles], float)
    m2 = np.array([p["ref_mean"] is not None for p in poles])
    adjusted = {"rule": "adjusted age = age - the R1t reference mean date of the main cluster (J1); 0 for the 26 one-cluster poles",
                "within_layer_spearman_adj_vs_ln_max_abs_600": stratified_spearman(adj, cov["ln_max_abs"], lay),
                "within_layer_spearman_adj_vs_ln_max_abs_574": stratified_spearman(adj[m2], cov["ln_max_abs"][m2], lay[m2]),
                "within_layer_spearman_adj_vs_ln_density_574": stratified_spearman(adj[m2], cov["ln_density"][m2], lay[m2]),
                "within_layer_spearman_adj_vs_age_574": stratified_spearman(adj[m2], age[m2], lay[m2]),
                "n_pairs_concordant": sum(bool(poles[e]["adj"] <= poles[l_]["adj"] - 1.0) for e, l_ in pr)}
    graph = match_graph_poles(poles, scan["chosen"], cov)
    keep = {p["pole"]: {k: p[k] for k in ("layer", "neuron", "sign", "age", "adj", "ref_mean", "max_abs", "density", "q999",
                                           "tok15", "a15")} for i, p in enumerate(poles) if i in graph}
    return {"what": f"N6 pairs, stage {stage}: studies/next/07_PICKS/N6_SPEC.md section 2.3", "stage": stage, "inputs": check_inputs(),
            "rule": {"dmin_age": DMIN_AGE, "grid_bank": GRID_BANK, "grid_inst": GRID_INST, "rho_max": RHO_MAX, "smd_max": SMD_MAX,
                     "split": SPLIT_L1 if l1_edited is None else SPLIT_L2},
            "caliper_scan": scan, "adjusted_age": adjusted, "balance": {**bal, "adj_difference": float(np.mean([p["adj_early"] - p["adj_late"] for p in pairs]))},
            "failing_input_unmatched": {"n_pairs": len(un), "balance": balance(poles, un, cov),
                                        "balanced": balanced(balance(poles, un, cov))},
            "n_pairs": len(pairs), "n_freeze": sum(p["half"] == "freeze" for p in pairs),
            "n_test": sum(p["half"] == "test" for p in pairs), "pairs": pairs, "poles": keep,
            **eval_set(stage), **(extra or {})}


def eval_set(stage: str) -> dict:
    """The eval windows of a stage. L1: W_EVAL. L2 (amendment A4): W_EVAL_L2 windows by the same rule, and G-INST's windows, the
    W_EVAL windows L1 read (the values the final pairs were matched on)."""
    if stage == "L1":
        return {"eval_windows": eval_windows(W_EVAL), "w_eval": W_EVAL}
    return {"eval_windows": eval_windows(W_EVAL_L2), "w_eval": W_EVAL_L2, "inst_windows": eval_windows(W_EVAL)}


def pc_layers(n_layers: int = N_LAYERS) -> list:
    """The positive-control layers (amendment A4): every PC_LAYER_STEP-th layer from 0."""
    return list(range(0, n_layers, PC_LAYER_STEP))


def match_graph_poles(poles, cal, cov) -> set:
    """Indices of the poles with at least one candidate edge under `cal` (stage 2 re-matches over these, so their exemplars ride along)."""
    out = set()
    keys = sorted(cal)
    for i, p in enumerate(poles):
        for j in range(i + 1, len(poles)):
            q = poles[j]
            if p["layer"] == q["layer"] and p["neuron"] != q["neuron"] and abs(p["age"] - q["age"]) >= DMIN_AGE and \
                    all(abs(cov[k][i] - cov[k][j]) <= cal[k] for k in keys):
                out |= {i, j}
    return out


def build_pairs() -> dict:
    """Stage 1 (now, banks only): the L1 pairs."""
    poles = load_dated()
    cov = covariates(poles)
    scan = caliper_scan(poles, cov, GRID_BANK)
    assert scan["chosen"] is not None, "no bank grid point gives a balanced matched set"
    return pairs_doc(poles, match(poles, scan["chosen"], cov), cov, scan, "L1")


def final_pairs(stage1: dict, colnorms: np.ndarray, energies: np.ndarray, l1_edited=()) -> dict:
    """Stage 2 (after L1, before L2): the bank calipers stay as chosen; the instance calipers on ln ||w|| and ln E+ (L1's arrays) are
    chosen over GRID_INST by the same rule, and the whole set is matched again. Fewer than MIN_PAIRS_L2 pairs: L2 does not launch.
    l1_edited: the poles L1 edited (its edits.jsonl); a pair holding one is kept in the freeze half (amendment A3)."""
    poles = load_dated()
    inst = {"colnorms": colnorms, "energies": energies}
    cov = covariates(poles, inst)
    scan = caliper_scan(poles, cov, GRID_INST, fixed=stage1["caliper_scan"]["chosen"])
    assert scan["chosen"] is not None, "no instance grid point gives a balanced matched set: L2 does not launch (spec section 9.5)"
    pr = match(poles, scan["chosen"], cov)
    assert len(pr) >= MIN_PAIRS_L2, f"{len(pr)} pairs, fewer than {MIN_PAIRS_L2}: L2 does not launch (spec section 9.5)"
    doc = pairs_doc(poles, pr, cov, scan, "L2", l1_edited=sorted(set(l1_edited)))
    doc["l1_edited"] = sorted(set(l1_edited))
    doc["instance"] = {q: {"colnorm": float(colnorms[v["layer"], v["neuron"]]),
                           "E_plus": float(energies[v["layer"], v["neuron"], 0 if v["sign"] > 0 else 1])} for q, v in doc["poles"].items()}
    return doc


def e1_known() -> dict:
    """The known number K1: one banked zero-replacement value of E1 at O7 T_15 (verdict_e1.json's run, samples/e1.jsonl): the planted
    unit PL05_00, side J, zero written into the own set's 50 upstream neurons at every window position (BOS left), job_fall the mean
    over the 12 contexts of (a_before - a_after) / a_before. The cross arm's banked value is the registered failing input."""
    check_inputs()
    rows = [json.loads(x) for x in open(FILES["e1_samples"])]
    pick = {r["arm"]: r for r in rows if all(r[k] == v for k, v in E1_PICK.items())}
    assert sorted(pick) == ["cross", "own"], sorted(pick)
    items = json.loads(FILES["e1_items"].read_text())
    unit = [u for u in items["planted"] if u["unit"] == E1_PICK["pair"]]
    assert len(unit) == 1
    u = unit[0]
    own, cross = pick["own"], pick["cross"]
    toks = [c["tok"] for c in own["contexts"]]
    assert toks == u["J_tok"], "the sample's contexts are not the item's J side"
    assert all(m[3] == [0.0] for m in own["members"]), "the zero arm wrote a non-zero value"
    return {"what": "E1 known number for N6 (K1)", "source": {k: {"path": str(FILES[k]), "sha256": SHA[k]} for k in ("e1_samples", "e1_items")},
            "unit": {"unit": u["unit"], "layer": u["layer"], "a": u["a"], "b": u["b"], "lam": u["lam"]},
            "members": [[int(m[0]), int(m[1])] for m in own["members"]], "toks": toks, "length": 256, "batch": 32,
            "a_before": [c["a_before"] for c in own["contexts"]], "a_after": [c["a_after"][0] for c in own["contexts"]],
            "job_fall": own["job_fall"][0], "failing_input_cross_job_fall": cross["job_fall"][0],
            "prompts_head": [c["prompt"][:120] for c in own["contexts"][:2]], "marked_tokens": [c["marked_token"] for c in own["contexts"]]}


# ------------------------------------------------------------------------------------------ the edit plan (the job reads this)
def plan(pairs: dict, n_layers: int = N_LAYERS, half: str | None = None) -> list:
    """Every edit of L2 in the order the job runs it: by layer; within a layer the positive control's three edits (amendment A4:
    in the layers of pc_layers only), then the rounding probes, then the pair edits (freeze pairs first). The pair edit size c is set
    on the instance (min of the two column norms); the positive control's pole is chosen on the instance (largest E+ in its layer).
    Amendment A5, the launch's half: None, every edit; "freeze" (L2a), the positive control, its probes and every edit on a freeze
    pair (the pair probes and the zero edit sit on freeze poles); "test" (L2b), the test pairs' col and rand edits and the same two
    pair probes and zero edit, which L2b's own G-EPS and G-ZN read."""
    assert half in (None, "freeze", "test"), half
    out = []
    probe_poles = probes(pairs)
    for lyr in range(n_layers):
        if lyr in pc_layers(n_layers):
            out += [{"edit": f"PC{lyr:02d}_{k}", "layer": lyr, "role": "pc", "kind": k} for k in ("col", "rand", "col_half")]
        if lyr in (0, n_layers - 1):
            out += [{"edit": f"PC{lyr:02d}_probe", "layer": lyr, "role": "probe_pc", "kind": "probe"}]
        for pr in sorted([p for p in pairs["pairs"] if p["layer"] == lyr], key=lambda p: (p["half"] != "freeze", p["pair"])):
            for m in ("early", "late"):
                for k in ("col", "rand"):
                    out.append({"edit": f"{pr['pair']}_{m}_{k}", "layer": lyr, "role": "pair", "pair": pr["pair"],
                                "member": m, "pole": pr[m], "kind": k, "half": pr["half"]})
                if pr[m] in probe_poles:
                    out.append({"edit": f"{pr['pair']}_{m}_probe", "layer": lyr, "role": "probe_pair", "pair": pr["pair"],
                                "member": m, "pole": pr[m], "kind": "probe", "half": pr["half"]})
                if pr[m] == probe_poles[0]:
                    out.append({"edit": f"{pr['pair']}_{m}_zero", "layer": lyr, "role": "zero", "pair": pr["pair"],
                                "member": m, "pole": pr[m], "kind": "zero", "half": pr["half"]})
    if half == "freeze":
        out = [e for e in out if e["role"] in ("pc", "probe_pc") or e.get("half") == "freeze"]
    elif half == "test":
        out = [e for e in out if e["role"] in ("probe_pair", "zero") or e.get("half") == "test"]
    return out


def probes(pairs: dict) -> list:
    """Two pair poles for the rounding probe and the zero-norm edit: the late members of the first freeze pair in the lowest and in
    the highest layer that hold a freeze pair."""
    fz = sorted([p for p in pairs["pairs"] if p["half"] == "freeze"], key=lambda p: (p["layer"], p["pair"]))
    assert len(fz) >= 2
    lo, hi = fz[0], [p for p in fz if p["layer"] == fz[-1]["layer"]][0]
    assert lo["late"] != hi["late"]
    return [lo["late"], hi["late"]]


def raw_example_pairs(pairs: dict) -> list:
    """The ten raw examples: the early and late column edits of five freeze pairs, the first freeze pair of the five layers
    nearest 0, 8, 16, 24, 31 that hold a freeze pair."""
    fz = [p for p in pairs["pairs"] if p["half"] == "freeze"]
    layers = sorted({p["layer"] for p in fz})
    assert len(layers) >= 5, "fewer than five layers hold a freeze pair"
    top = max(layers)
    pick = []
    for target in [round(i * top / 4) for i in range(5)]:
        lyr = min((x for x in layers if x not in [q["layer"] for q in pick]), key=lambda x: (abs(x - target), x))
        pick.append(sorted([p for p in fz if p["layer"] == lyr], key=lambda p: p["pair"])[0])
    return [p["pair"] for p in pick]


# ------------------------------------------------------------------------------------------ projection (GPU-hours)
FLOP_LAYER = 2 * (4 * HIDDEN * HIDDEN + 3 * HIDDEN * WIDTH)      # per token per layer, the matmuls
FLOP_ATTN = 4 * 1024 * HIDDEN                                    # per token per layer, the attention scores at a mean context of 1,024
FLOP_HEAD = 2 * HIDDEN * VOCAB
RATES = {"low": 60e12, "center": 40e12, "high": 25e12}           # fp32 FLOP/s sustained on one L40S (no TF32); L1 measures it
FIXED_S = {"low": 420, "center": 600, "high": 900}               # download 14.6 GB, load, cast, known numbers, records


def flops_from(layer: int, tokens: int) -> float:
    return tokens * ((N_LAYERS - layer) * (FLOP_LAYER + FLOP_ATTN) + FLOP_HEAD)


def projection(pairs: dict) -> dict:
    """GPU-hours of L1 and L2. fp32 terms at RATES; bf16 terms (K2, G-POLE, the bf16 base pass and its two rounding probes) at four
    times the fp32 rate; the fixed cost covers download, load, cast and records."""
    tok = W_EVAL * (HW + 1)
    pl = plan(pairs)
    lay = {p["pair"]: p["layer"] for p in pairs["pairs"]}
    g_pole = sum(10 * 1024 * (q["layer"] + 1) * (FLOP_LAYER + FLOP_ATTN) for q in pairs["poles"].values())
    bf16 = (flops_from(0, 59 * (HW + 1)) + 3 * flops_from(0, tok) + g_pole) / 4
    k3 = 2 * flops_from(0, 59 * (HW + 1))                       # K3 and its off-by-one failing input
    base = 3 * flops_from(0, tok)                                # two base passes and the progression
    edits = sum(flops_from(e["layer"], tok) for e in pl)
    raw = sum(2 * flops_from(lay[p], tok) for p in raw_example_pairs(pairs)) + 3 * flops_from(N_LAYERS // 2, tok)
    out = {"tokens_per_edit": tok, "n_edits_L1": 13, "n_edits_L2": len(pl),
           "flop": {"base": base, "edits_L2": edits, "edits_L1": raw, "k3": k3, "bf16_terms_fp32_equivalent": bf16}}
    for k, r in RATES.items():
        l1 = (FIXED_S[k] + (base + raw + k3 + bf16) / r) / 3600
        l2 = (FIXED_S[k] + (base + edits + k3 + bf16) / r) / 3600
        out[k] = {"L1_gpu_h": round(l1, 3), "L2_gpu_h": round(l2, 3), "total_gpu_h": round(l1 + l2, 3),
                  "usd_at_2.24": round(2.24 * (l1 + l2), 2)}
    out["hard_kill_min"] = {"L1": math.ceil(1.3 * out["high"]["L1_gpu_h"] * 60), "L2": math.ceil(1.3 * out["high"]["L2_gpu_h"] * 60)}
    return out


# ------------------------------------------------------------------------------------------ reading the records
def index(records: list) -> dict:
    out = {}
    for r in records:
        assert r["edit"] not in out, f"edit {r['edit']} twice"
        out[r["edit"]] = r
    return out


def pair_table(pairs: dict, rec: dict, half: str) -> dict:
    """Per pair of the half: D_col, D_rand, the Q ratio difference, the energy-rule pick, the window-level D for the bootstrap."""
    rows, parked = [], []
    for p in pairs["pairs"]:
        if p["half"] != half:
            continue
        keys = [f"{p['pair']}_{m}_{k}" for m in ("early", "late") for k in ("col", "rand")]
        if not all(k in rec for k in keys):
            parked.append(p["pair"])
            continue
        ec, er, lc, lr = (rec[k] for k in keys)
        rows.append({"pair": p["pair"], "concordant_adj": p["concordant_adj"],
                     "D_col": ec["dL"] - lc["dL"], "D_rand": er["dL"] - lr["dL"],
                     "D_col_win": (np.array(ec["dL_win"]) - np.array(lc["dL_win"])).tolist(),
                     "Q_early": ec["dL"] / ec["kl_eff"] if ec["kl_eff"] > 0 else None,
                     "Q_late": lc["dL"] / lc["kl_eff"] if lc["kl_eff"] > 0 else None,
                     "dL_early": ec["dL"], "dL_late": lc["dL"], "E_early": ec["E_plus"], "E_late": lc["E_plus"],
                     "kl_col_gt_rand": [ec["kl_eff"] > er["kl_eff"], lc["kl_eff"] > lr["kl_eff"]]})
    n_all = sum(p["half"] == half for p in pairs["pairs"])
    return {"rows": rows, "parked": parked, "n_all": n_all}


def boot_sd_win(dwin: np.ndarray, seed: int) -> float:
    """sd over N_BOOT_WIN window resamples of the mean of the per-window D (the eval-set sampling noise of one pair's D)."""
    rng = np.random.default_rng(seed)
    w = len(dwin)
    return float(np.std([dwin[rng.integers(0, w, w)].mean() for _ in range(N_BOOT_WIN)]))


def gate_floor(gates: dict) -> dict:
    """F_eps from the fp32 rounding probes (the symmetric add-and-subtract edits) and the tie margin."""
    f = max(abs(v) for v in gates["G_EPS"]["fp32"].values())
    return {"F_eps": f, "tau_eps": max(TIE_MULT * f, 1e-12)}


def freeze(pairs: dict, records: list, gates: dict) -> dict:
    rec = index(records)
    fl = gate_floor(gates)
    t = pair_table(pairs, rec, "freeze")
    rows = t["rows"]
    assert rows, "no freeze pair was read"
    sds = [boot_sd_win(np.array(r["D_col_win"]), SEED + i) for i, r in enumerate(rows)]
    tau_s = float(np.median(sds))
    dl = np.array([abs(rec[f"{r['pair']}_{m}_col"]["dL"]) for r in rows for m in ("early", "late")])
    q10 = float(np.percentile(dl, 10))
    D = np.array([r["D_col"] for r in rows])
    Dr = np.array([r["D_rand"] for r in rows])
    k, n = win_share(D, fl["tau_eps"])
    kr, _ = win_share(Dr, fl["tau_eps"])
    med_abs = float(np.median(np.abs(D)))
    n_test = sum(p["half"] == "test" for p in pairs["pairs"])
    lo5, hi5 = wilson(0.5 * n_test, n_test)
    need = next((w for w in range(n_test + 1) if wilson(w, n_test)[0] > 0.5), None)
    return {"what": "N6 freeze 2: thresholds from the freeze half only (spec section 6.2)", **fl, "tau_s": tau_s,
            "tau_s_rule": "median over freeze pairs of the window-bootstrap sd of D_col (1,000 resamples of the eval windows)",
            "q10_abs_dL_col_freeze": q10, "G_RES": {"value": fl["F_eps"], "bound": min(EPS_ABS, EPS_REL * q10),
                                                   "pass": bool(fl["F_eps"] <= min(EPS_ABS, EPS_REL * q10))},
            "G_POWER": {"value": tau_s, "median_abs_D_col": med_abs, "bound": POWER_MAX * med_abs,
                        "ratio": tau_s / med_abs if med_abs > 0 else math.inf, "pass": bool(tau_s <= POWER_MAX * med_abs),
                        "rule": "amendment A4: tau_s <= 0.5 x median |D_col| over the freeze pairs; if not, the verdict reads "
                                "UNDECIDED_NO_POWER before the test half is read"},
            "discovery": {"n": n, "p_col": k / n, "p_col_ci": wilson(k, n), "p_rand": kr / n, "p_rand_ci": wilson(kr, n),
                          "R": float(np.mean(np.abs(D) > 2 * tau_s)), "median_D_col": float(np.median(D)),
                          "parked": t["parked"]},
            "room": {"n_test": n_test, "wilson_at_half": [lo5, hi5], "flat_band": [0.5 - FLAT_M, 0.5 + FLAT_M],
                     "flat_reachable": bool(lo5 >= 0.5 - FLAT_M and hi5 <= 0.5 + FLAT_M),
                     "wins_needed_for_lower_bound_above_half": need}}


def label(k: float, n: int, ceiling_hi: float, extra_ok: bool | None, resolved: float) -> dict:
    lo, hi = wilson(k, n)
    if lo > ceiling_hi:
        lab = "ABOVE_CEILING"
    elif lo > 0.5:
        lab = "SUPPORTED" if extra_ok in (None, True) else "SUPPORTED_UNSPECIFIC"
    elif hi < 0.5:
        lab = "REVERSED"
    elif lo >= 0.5 - FLAT_M and hi <= 0.5 + FLAT_M:
        lab = "FLAT" if resolved >= R_MIN else "UNDECIDED_NO_POWER"
    else:
        lab = "UNDECIDED"
    return {"label": lab, "p": k / n, "ci": [lo, hi], "n": n, "resolved": resolved}


def instrument_gates(gates: dict) -> list:
    """Names of failed gates among those the verdict reads (each the job's record of pass and of its failing input)."""
    failed = []
    for name, g in gates.items():
        if not isinstance(g, dict) or "pass" not in g:
            continue
        if not g["pass"]:
            failed.append(name)
        if "failing_input" in g and g["failing_input"].get("pass") is True:
            failed.append(name + ":failing_input_passed")
    return failed


def no_power(failed: list, th: dict) -> dict:
    """Amendment A4: the power gate failed at freeze 2, so the test half is not read and no test label is issued. A failed
    instrument gate still makes every label DOES NOT COUNT, with UNDECIDED_NO_POWER kept as the realized label."""
    lab = "UNDECIDED_NO_POWER"
    out = {"what": "N6 verdict: G-POWER failed at freeze 2 (amendment A4); the test half was not read", "failed_gates": failed,
           "G_POWER": th["G_POWER"], "n_test_read": 0}
    for h in ("H_age", "H_adj", "H_eff"):
        out[h] = {"label": "DOES NOT COUNT", "realized": lab} if failed else {"label": lab}
    out["headline"] = "; ".join(f"{h} {out[h]['label']}" for h in ("H_age", "H_adj", "H_eff"))
    return out


def verdict(pairs: dict, records: list, gates: dict, th: dict) -> dict:
    failed = instrument_gates(gates) + ([] if th["G_RES"]["pass"] else ["G_RES"])
    pc = gates.get("G_PC")
    if pc is None:                                        # parked before the positive controls were all read
        failed.append("G_PC_missing")
    if not th["G_POWER"]["pass"]:
        return no_power(failed, th)
    rec = index(records)
    t = pair_table(pairs, rec, "test")
    rows = t["rows"]
    if len(t["parked"]) > PARK_MAX * t["n_all"]:
        failed.append("G_PARK")
    tau, tau_s = th["tau_eps"], th["tau_s"]
    ceil_hi = 1.0 if pc is None else wilson(pc["n_col_gt_rand"], pc["n_layers"])[1]
    D = np.array([r["D_col"] for r in rows])
    Dr = np.array([r["D_rand"] for r in rows])
    n = len(rows)
    k, _ = win_share(D, tau)
    kr, _ = win_share(Dr, tau)
    wc = np.where(D > tau, 1.0, np.where(np.abs(D) <= tau, 0.5, 0.0))
    wr = np.where(Dr > tau, 1.0, np.where(np.abs(Dr) <= tau, 0.5, 0.0))
    dp_ci = boot_ci(lambda i: wc[i].mean() - wr[i].mean(), n)
    R = float(np.mean(np.abs(D) > 2 * tau_s))
    age = label(k, n, ceil_hi, dp_ci[0] > 0, R)
    conc = [r for r in rows if r["concordant_adj"]]
    Dc = np.array([r["D_col"] for r in conc])
    kc, nc = win_share(Dc, tau) if conc else (0.0, 0)
    adj = label(kc, nc, ceil_hi, None, float(np.mean(np.abs(Dc) > 2 * tau_s))) if nc else {"label": "NOT_RUN", "n": 0}
    qr = [r for r in rows if r["Q_early"] is not None and r["Q_late"] is not None]
    Dq = np.array([r["Q_early"] - r["Q_late"] for r in qr])
    kq, nq = win_share(Dq, 0.0) if qr else (0.0, 0)
    eff = label(kq, nq, ceil_hi, None, 1.0) if nq else {"label": "NOT_RUN", "n": 0}
    pick_age = np.array([r["dL_late"] for r in rows])
    pick_rand = np.array([(r["dL_early"] + r["dL_late"]) / 2 for r in rows])
    pick_E = np.array([r["dL_early"] if r["E_early"] < r["E_late"] else r["dL_late"] for r in rows])
    gain_vs_rand = pick_rand - pick_age
    gain_vs_E = pick_E - pick_age
    out = {"what": "N6 verdict: the test half, read once (spec section 6)", "failed_gates": failed,
           "thresholds": {"tau_eps": tau, "tau_s": tau_s, "ceiling_hi": ceil_hi},
           "H_age": age, "p_rand": {"p": kr / n, "ci": list(wilson(kr, n))}, "delta_p_col_minus_rand": {"value": float(wc.mean() - wr.mean()),
                                                                                                    "ci": list(dp_ci)},
           "H_adj": adj, "H_eff": eff,
           "efficacy_share_col_gt_rand": float(np.mean([x for r in rows for x in r["kl_col_gt_rand"]])),
           "median_D_col": float(np.median(D)), "median_D_col_ci": list(boot_ci(lambda i: np.median(D[i]), n)),
           "practical": {"gain_vs_random_pick": float(gain_vs_rand.mean()), "ci": list(boot_ci(lambda i: gain_vs_rand[i].mean(), n)),
                         "gain_vs_energy_pick": float(gain_vs_E.mean()), "ci_E": list(boot_ci(lambda i: gain_vs_E[i].mean(), n))},
           "parked": t["parked"], "n_test_read": n}
    if failed:
        for h in ("H_age", "H_adj", "H_eff"):
            out[h] = {**out[h], "label": "DOES NOT COUNT", "realized": out[h].get("label")}
    out["headline"] = "; ".join(f"{h} {out[h]['label']}" for h in ("H_age", "H_adj", "H_eff"))
    return out


# ------------------------------------------------------------------------------------------ CLI
def read_jsonl(p) -> list:
    return [json.loads(x) for x in open(p) if x.strip()]


SAME_KEYS = ("eval_windows_sha256", "eval_tokens_sha256", "pairs_sha256", "w_eval")    # A5: what makes L2a and L2b one comparison


def read_banks(dirs: list, want: tuple) -> tuple[list, dict]:
    """Amendment A5: L2 ran as two launches, L2a (half freeze) and L2b (half test). `dirs` are their out/n6 directories (metrics.json,
    edits.jsonl); their halves must be `want` (freeze: ("freeze",); the verdict: ("freeze", "test"), or ("freeze",) when G-POWER failed
    and L2b was never launched). Every bank is stage L2 and reads the same eval windows, eval tokens and pairs content (sha256). The
    records: every record of L2a (the positive control, the probes, the zero edit, the freeze pairs) and the test-pair records of L2b
    (L2b's own probe and zero records are read by its G-EPS and G-ZN, in its metrics.json). The gates: L2a's under their names, L2b's
    under 'L2b:<name>', so a failed gate or a failing input that passed in either launch is read by the verdict."""
    banks = [(json.loads((pathlib.Path(d) / "metrics.json").read_text()), read_jsonl(pathlib.Path(d) / "edits.jsonl")) for d in dirs]
    halves = sorted(m["half"] for m, _ in banks)
    assert halves == sorted(want), f"banks of halves {halves}, this step reads {sorted(want)} (amendment A5)"
    m0 = banks[0][0]
    for m, _ in banks:
        assert m["stage"] == "L2", f"a bank of stage {m['stage']}"
        for k in SAME_KEYS:
            assert m[k] == m0[k], f"A5: the {m['half']} launch's {k} {m[k]} is not the {m0['half']} launch's {m0[k]}: the two launches " \
                                  "are not one comparison"
    records, gates = [], {}
    for m, recs in banks:
        if m["half"] == "freeze":
            assert not [r for r in recs if r["role"] == "pair" and r["half"] != "freeze"], "L2a holds a test-pair record"
            records += recs
            gates.update(m["gates"])
        else:
            assert all(r["role"] in ("pair", "probe_pair", "zero") for r in recs), "L2b holds a positive-control record"
            assert not [r for r in recs if r["role"] == "pair" and r["half"] != "test"], "L2b holds a freeze-pair record"
            records += [r for r in recs if r["role"] == "pair"]
            gates.update({f"L2b:{k}": v for k, v in m["gates"].items()})
    return records, gates


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "final", "e1", "projection", "freeze", "verdict"])
    ap.add_argument("--l1", help="final: L1's out/n6 directory (colnorms.npy, energies.npy, edits.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs")
    ap.add_argument("--bank", action="append", help="freeze, verdict: an L2 launch's out/n6 directory (A5: L2a; the verdict also L2b)")
    ap.add_argument("--thresholds")
    a = ap.parse_args(argv)
    if a.cmd == "pairs":
        res = build_pairs()
    elif a.cmd == "final":
        d = pathlib.Path(a.l1)
        edited = {json.loads(x)["pole"] for x in open(d / "edits.jsonl") if x.strip()}
        res = final_pairs(json.loads(pathlib.Path(a.pairs).read_text()), np.load(d / "colnorms.npy"), np.load(d / "energies.npy"),
                          edited)
    elif a.cmd == "e1":
        res = e1_known()
    elif a.cmd == "projection":
        res = projection(json.loads(pathlib.Path(a.pairs).read_text()))
    else:
        pairs = json.loads(pathlib.Path(a.pairs).read_text())
        if a.cmd == "freeze":
            res = freeze(pairs, *read_banks(a.bank, ("freeze",)))
        else:
            th = json.loads(pathlib.Path(a.thresholds).read_text())
            res = verdict(pairs, *read_banks(a.bank, ("freeze", "test") if th["G_POWER"]["pass"] else ("freeze",)), th)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(a.cmd, "written", a.out, json.dumps({k: res[k] for k in res if k in ("n_pairs", "n_freeze", "n_test", "headline")}))


if __name__ == "__main__":
    main()
