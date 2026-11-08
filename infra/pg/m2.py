"""M2: name the carrier before the birth (studies/next/07_PICKS/M2_SPEC.md).

Numpy only (the job imports torch itself). Users:
  - the laptop CLI below: `births` (the unit table from the banked recount, before any GPU), `examples` (the zero-GPU
    part of the ten raw examples, freeze half only), `project` (GPU-hours from the window plan), `freeze`
    (thresholds from the freeze half and the controls, before the test half is read) and `verdict` (once, on the
    test half, after freeze 2);
  - the job `infra/aws_pg/jobs/m2_score.py` on the instance: the plan, the scores, the ranks, the gates;
  - the dry run `infra/aws_pg/jobs/m2_dryrun.py`.

    python -m infra.pg.m2 births   --bank <BANK/neurons> --out studies/next/m2/births.json
    python -m infra.pg.m2 examples --bank <BANK/neurons> --births studies/next/m2/births.json --out studies/next/m2/samples
    python -m infra.pg.m2 project  --births studies/next/m2/births.json
    python -m infra.pg.m2 freeze   --banks studies/next/m2/banks --out studies/next/m2/thresholds.json
    python -m infra.pg.m2 verdict  --banks studies/next/m2/banks --thresholds studies/next/m2/thresholds.json \
                                   --out studies/next/m2/verdict_m2.json

Objects (SPEC §0.5). A birth is a non-echo cluster c of a grid pole p = (layer l, neuron n, sign s) whose birth index
k under the already-clean rule is later than chance in J1's recount: p_late <= 0.05 under R1t
(`studies/next/10_recount2/out/05_per_cluster_R1t.csv`). E_c is the cluster's exemplar set: the global token indices
of p's T_15 top-50 at the cluster's ranks. At checkpoint t, for every pole q = (n', s') of layer l (22,016 poles):
  A  (reference) top-K overlap: the number of E_c positions among q's 50 top positions at t (banked topk/ `tok`).
  B  (primary)   the pole's own current percentile: the fraction of q's own-sign tokens at t whose |a| lies below
                 v = median over E_c of a+_q (a+ = max(sign * a, 0)), read from q's banked 400-bin histogram
                 (stats/ `hist`), log-linear inside the bin holding v; 0 when v = 0.
  exploratory    ratio v / q999; max_abs; density; the down-projection column norm (job-free, the last three).
Primary B, reference A: the head's decision, 2026-11-05, before freeze 1 (SPEC §2.3).
The rank of pole x under a score is the number of the layer's 22,016 poles whose score is >= x's score: ties count
against x. A hit is rank <= 10.

Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np

REPO = "allenai/Olmo-3-1025-7B"
N_LAYERS, N_NEURONS, K_TOP = 32, 11008, 50
N_POLES_LAYER = 2 * N_NEURONS
TOP = 10
CHANCE = TOP / N_POLES_LAYER                    # 0.000454; per sign (11,008 poles) it would be 0.000908
WINDOW, N_WINDOWS, BOS = 2048, 10044, 100257    # the banked grid harvest (T_15 manifest)
N_WINDOWED = WINDOW * N_WINDOWS
HIST_LO, HIST_HI, HIST_B = -6.0, 4.0, 400       # neurons_harvest constants
P_PASS = 0.05                                    # J1's per-cluster rule (10_RECOUNT2.md §8)
K_MATCH = 100                                    # activation-matched poles per birth
N_FLOOR_REPS = 1000
N_BOOT = 10000
SEED = 0
SEED_PB = 20261105
PB_DRAWS = 2000
N_MIN = 50                                       # usable births per half, else DOES NOT COUNT
CTRL_MIN = 0.90                                  # construction control at T_15
# Known-number statistic, amendment A1 (head's decision 2026-11-05 20:15 CST, before freeze 1b; M2_SPEC.md §3.8).
# Per entry |new - bank| / (ATOL + |bank|), the mixed tolerance of numpy.isclose, with ATOL = 0.05 in units of a: bf16
# noise in act(g) * u is absolute, about the size seen on the device pair, and a relative difference has no meaning near
# zero (score-01 stopped on it at T_14). A gate passes when the 99th percentile over the compared entries is within
# tolerance. The tolerance is frozen on control data: the banked device pair at T_14 (Blackwell bank against the L40S
# repeat REPEAT_T14), the 99th percentile of the statistic over all 32,000 entries (0.019737) and over the 12,231 with
# |a_bank| < 1 (0.021159). CTRL_P99 is the larger. Other device type than the bank: twice it (the head's rule). Same
# device type: once (the two tiers of §3.1 kept).
# Revision of A1 (head's decision 2026-11-05 20:50 CST, before freeze 1c): the entries G-EX reads before the birth have
# median |a_bank| 0.31, and the plan's 477 entries at T_14 gave p99 0.04034 on the pair, 0.002 under 2 * CTRL_P99. So the
# control subset is matched to that range: across devices the tolerance is twice the pair's p99 over the entries with
# |a_bank| < 0.5 (CTRL_P99_X). The same-device tier stays CTRL_P99: on one device type the forward was bit-identical.
ATOL = 0.05
CTRL_P99 = 0.021158761373165365     # m2.control_p99(bank T_14, REPEAT_T14)["below_1"]["p99"]
CTRL_P99_X = 0.022136643261325667   # m2.control_p99(bank T_14, REPEAT_T14)["below_0.5"]["p99"]
# Amendment A2 (head's decision 2026-11-06 03:45Z, before freeze 1d; M2_SPEC.md §3.8 "A2"): the cross-device G-EX gate
# reads only the entries with |a_bank| >= EX_ABS_MIN; those below are reported, not gated. Pre-birth exemplar entries
# are small by construction, and on small activations the device difference is noise that carries no signal about the
# forward. The tolerance is twice the control pair's p99 of the same statistic on the same subset (CTRL_P99_EX).
# G-TOPK keeps "cross" (its entries are large by construction); K0, I-ID and the same-device G-EX keep "same".
EX_ABS_MIN = 0.1
CTRL_P99_EX = 0.019359100495742992  # m2.control_p99(bank T_14, REPEAT_T14)["ge_0.1"]["p99"]
TOL = {"same": {"p99": CTRL_P99}, "cross": {"p99": 2 * CTRL_P99_X},
       "ex_cross": {"p99": 2 * CTRL_P99_EX, "abs_min": EX_ABS_MIN}}
# Amendment A3 (head's decision 2026-11-06 04:20Z, before freeze 1e; M2_SPEC.md §3.8 "A3"): at a point whose bank was
# made on another device (T_7-T_14) G-EX is reported, not gated: the job writes the A2 statistic, the count of gated
# entries above TOL["ex_cross"] and their list, and continues. At the same-device points (T_5, T_6, T_15) G-EX stays a
# gate. The deviation between the Blackwell bank and an L40S forward varies by checkpoint (T_13's tail is twice T_14's
# with the same median), so no tolerance frozen on the one T_14 pair bounds the other Blackwell points.
# Amendment A4 (head's decision 2026-11-06 05:35Z, before freeze 1f; M2_SPEC.md §3.8 "A4"): a pole with no token of its
# own sign in the banked harvest at a point (n_pole_tokens 0, an empty histogram, and max_abs 0; `degenerate`) gets
# B = 0 and ratio = 0 there, so it ranks last (ties count against the carrier, as registered), and it is not a
# candidate of the matched-pole floor set. A is unaffected (its banked top-50 slots are empty). score-04 crashed on
# such a pole (L24N9826+ at T_10). Anything else unexpected still stops the job: a pole whose histogram and max_abs
# disagree, a carrier with no token of its sign at its pre point, fewer than K_MATCH candidates.
REPEAT_T14 = {"s3": "results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz",
              "sha256": "e9cdf8658615bc7a6e2d03d0b5c3e67e0237a40d048c83ac5e569e603ac7e771"}
SCORES = ["A", "B", "ratio", "maxabs", "density", "colnorm"]
PRIMARY, REFERENCE = "B", "A"          # head's decision, 2026-11-05, before freeze 1 (M2_SPEC.md §2.3)

L40S, BW = "NVIDIA L40S", "NVIDIA RTX PRO 5000 Blackwell"
# index, banked directory under the grid root, HF branch, revision (= the banked manifest's `revision` and the HF refs
# read of 2026-11-05, scratchpad m1_gate/hf/refs.json), GPU type of the bank (manifest `device_name`).
POINTS = [
    (0, "1/out/t00_step0", "stage1-step0", "f129a5d178fd784d802a521d6f388c6df6988fd6", L40S),
    (1, "1/out/t01_step1000", "stage1-step1000", "d82e70ec28daa79d25405cd128e127c52bb10516", L40S),
    (2, "1/out/t02_step2000", "stage1-step2000", "6a238f66840f4db2deff7049b12fbc608267e14f", L40S),
    (3, "1/out/t03_step3000", "stage1-step3000", "9de48909b402a200848e5cba488dbfd04b6227a3", L40S),
    (4, "1/out/t04_step5000", "stage1-step5000", "825e84ed82eb78b1b395d0aa0164378f64544f92", L40S),
    (5, "1/out/t05_step8000", "stage1-step8000", "ebfe7b62b31bfadff464a8926a48a256c9431db7", L40S),
    (6, "1/out/t06_step12000", "stage1-step12000", "a1cbaea430374e12c8e7b463d33ecbf8da657dea", L40S),
    (7, "1b/out/t07_step20000", "stage1-step20000", "04eeee4fa74f9f3d2cb11c31e155545dad32365e", BW),
    (8, "1b/out/t08_step35000", "stage1-step35000", "d1dbb0e2d4d3fbdc9c185a34c4e99f13ef6ae2a6", BW),
    (9, "1b/out/t09_step60000", "stage1-step60000", "08b5da69ff99e0858f18d9870ac87f43d7602cf0", BW),
    (10, "1b/out/t10_step100000", "stage1-step100000", "1d074dbc15d91544abe47d5508ac1670ef8c7129", BW),
    (11, "1b/out/t11_step180000", "stage1-step180000", "c515906cd0f5e30e9e2f56f5f28870c375d8bef0", BW),
    (12, "1b/out/t12_step320000", "stage1-step320000", "af83fc98693475b503de0a86407f0260a7281786", BW),
    (13, "1b/out/t13_step560000", "stage1-step560000", "db8c654b5cbe06311f92e86d68cd4f7d31642a5a", BW),
    (14, "1b/out/t14_step1000000", "stage1-step1000000", "cb286dcdaaf3f92b3c419dd77439078fdebffe53", BW),
    (15, "1/out/t15_step1413814", "stage1-step1413814", "373bad25002f1624757a73235c5ca844c6375c25", L40S),
]
STEPS = [int(p[2].split("step")[1]) for p in POINTS]
IDENTITY_T = 13          # the identity pair: T_13's pre-role windows scored a second time, in reversed order

# Laptop banks (sha256 read 2026-11-05; studies/next/10_recount2/out/*.pointer.json for the first two).
BANK_FILES = {
    "arrays": "next/recount2/arrays.npz",
    "births": "next/recount2/births.jsonl",
    "clusters": "w4/o7_s_v2/explain/clusters.jsonl",
}
BANK_SHA = {
    "arrays": "80e5ffac44ea44b3d7794f3e57bd51aa67ba5b6b8508985d60d1b12d6c6ddba4",
    "births": "f21230ec07a6912effb73807641e852b945ea1ef0b16f73b4f138cfe5955bb10",
}
# poles/exemplar_a.npz and samples/contexts.jsonl of the 16 grid points on the laptop (BANK/w3/grid/<dir>/), as
# recorded by the recount (studies/next/10_recount2/out/arrays.pointer.json). The job checks the uploaded copies.
EXEMPLAR_SHA = [
    "30c424c37f2e16b00648fa9e710dd991c682990eab9859b71be05ba9d015de74",  # T_0
    "1b25209ebad148c4df0eefbc579affd7e6133591fe889193d78e90732a43cfe2",  # T_1
    "9fb744bd4b06161056eff98b0b871efab587a65d4d50b03515121cb6cbdc2211",  # T_2
    "7dea0b22fde45a486208a96fa6220474c84fc4d4c7df162593b59636200a1b40",  # T_3
    "e75de996b2dcb85bc81aec805ed57c57d9c16081e86b344d5e75d256f481cd59",  # T_4
    "267fcec9cf28dc754156297bd1788a7c7704e7e53b6f836dac197d016bccdf3d",  # T_5
    "ec9743010cce40da2a94bc165629d165e3084a58aaae6da2dea402035b047bb2",  # T_6
    "ab9d78e05307d32c0e7315ab1f848986f9b90203e82de5af2c8e3282e45e70fe",  # T_7
    "de72e1de5977f37ecb7e439e51397b5a6781fde8709fd38e0bc1ab35be86f7a3",  # T_8
    "8b1bad79aa8e635eb2ab54f4c5210193b00f11044784854e4e507e14db2096f2",  # T_9
    "9e79b4209c37e84c362602aecb50a8eef3016a3d73224dddfbe70ff601293989",  # T_10
    "27569b4b8210973491a9b59cd075eca7da8ed77049f112cc28d0a45142174b3a",  # T_11
    "6e015a60d7e6719d7a3e6825a36c710bdfe11838a0a5cd7ae9896116b1b7d6bc",  # T_12
    "7976ca719702c6cdb1dc7db5ea95c296e43893be76438c3f9c2173c257e35d8d",  # T_13
    "9fabd78be6d81b5c4cecbc52e9ca883ec9a0ea3db5f2a8067c86e8e25c37e09e",  # T_14
    "84bd59fdaf6b45727778b86ae82602b6815d4b8cda591a1237f022aa09ace300",  # T_15
]
CONTEXTS_SHA = [
    "37749b6b33bfa2a6599f1db69575a94b519c6e6a696d0e0eb9b11c4947a93ddc",  # T_0
    "650bf599c70ea8b6abcb1acc0fcb32dad4ff8850308a9a67134d7eb003ddf700",  # T_1
    "0467369a31c17b072e0c24756cd0404f97aa04d23cd4d299e0640df025f71693",  # T_2
    "abb702c715275eca838c7289cffce583c1952d5de70188dfc2e50e02611fba34",  # T_3
    "277f222642b5575bf1061630fdd0e9d2402ad2257edd574f0999befd7c5113f8",  # T_4
    "6388809179817850bb67713ad170f837ed70bd723a57744c121527eb0487f99e",  # T_5
    "d816a227a7dd127c364489a838bd49f074affade276bf05e5504b8cbf8fc1c3d",  # T_6
    "c115b531d2637d4f3f6f853d00094d248ceada879193be24ce162cc12f015b24",  # T_7
    "e9fb602d42417d89c45ffdaed1c802f8445f92378c36a6fac5e15a5a175c35b3",  # T_8
    "15b0dad7350df871dd1d5ee3471a58b4781b2db24796d1dd02a498a109fa21cd",  # T_9
    "c700f2e882bccdae862e20f4c605cf1b2b8b5a4a2e78749f810d3dc7d85b12b2",  # T_10
    "cb579d3707983a75133f56040bad8c828efa01ffefaa454f2c951e78d924d5a8",  # T_11
    "ee7de41e40f3151a59470e2615a6c8edb98f3adce5df7c1dca7d5c25434824d5",  # T_12
    "4338fb3238130ac860145406dfada723bc953bf8292f1b2b48165dd782069a65",  # T_13
    "692e85854bebb5446522b6c3cb16138bee0d57c581de5d1ecd48595f645ddc9a",  # T_14
    "a4d46cd837f7a7532c300cac7372df1908e37f25b5acf57b6ea1ea37493e3269",  # T_15
]
PER_CLUSTER = pathlib.Path(__file__).resolve().parents[2] / "studies/next/10_recount2/out/05_per_cluster_R1t.csv"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pole(p: str) -> tuple[int, int, int]:
    """'L12N4051+' -> (12, 4051, 0); sign axis 0 = + pole, 1 = - pole (DATA_CONTRACT §0)."""
    assert p[0] == "L" and p[3] == "N" and p[-1] in "+-", p
    return int(p[1:3]), int(p[4:-1]), "+-".index(p[-1])


def pole_id(layer: int, n: int, s: int) -> str:
    return f"L{layer:02d}N{n}{'+-'[s]}"


def half_of(neuron: int) -> str:
    """The split, fixed before any birth was read: even neuron index -> freeze, odd -> test (both poles together)."""
    return "freeze" if neuron % 2 == 0 else "test"


# ------------------------------------------------------------------------------- the dating rule (10_recount2/common.py)
def births_matrix(a_plus_i: np.ndarray, q999_i: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Birth index of many same-size exemplar groups of one pole: a_plus_i [16, 50], q999_i [16], groups [G, m] of
    T_15 ranks. Present at t: median > 0 and median >= q999 (numpy median, mean of the two middle values); birth = the
    first t <= 14 present at t and t + 1, else 15 if present at 15, else 99 (never). SPEC_D1 lines 280-286."""
    v = a_plus_i[:, groups]
    med = np.median(v, axis=2)
    pres = (med > 0) & (med >= q999_i[:, None])
    two = pres[:-1] & pres[1:]
    out = np.full(groups.shape[0], 99, dtype=np.int64)
    out[pres[-1]] = 15
    any_two = two.any(axis=0)
    out[any_two] = np.argmax(two, axis=0)[any_two]
    return out


def load_bank(bank) -> dict:
    bank = pathlib.Path(bank)
    paths = {k: bank / v for k, v in BANK_FILES.items()}
    sha = {k: sha256_file(p) for k, p in paths.items()}
    for k, want in BANK_SHA.items():
        assert sha[k] == want, f"bank {k}: sha256 {sha[k]} != registered {want}"
    z = np.load(paths["arrays"])
    poles = [str(p) for p in z["poles"]]
    sgn = np.array([1.0 if p.endswith("+") else -1.0 for p in poles])
    clusters = {}
    for line in open(paths["clusters"]):
        r = json.loads(line)
        if r["sample"] == 0:
            clusters[r["pole"]] = r["clusters"]
    births = [json.loads(line) for line in open(paths["births"])]
    per = {(r["pole"], int(r["c"])): r for r in csv.DictReader(open(PER_CLUSTER))}
    return {"poles": poles, "row": {p: i for i, p in enumerate(poles)}, "a": z["a"],
            "a_plus": np.maximum(z["a"] * sgn[None, :, None], 0.0), "q999": z["q999"], "max_abs": z["max_abs"],
            "tok15": z["tok15"], "clusters": clusters, "births": births, "per_cluster": per,
            "inputs": {**{k: {"path": str(p), "sha256": sha[k]} for k, p in paths.items()},
                       "per_cluster": {"path": "studies/next/10_recount2/out/05_per_cluster_R1t.csv",
                                       "sha256": sha256_file(PER_CLUSTER)}}}


def nonecho_ranks(clusters: list) -> list:
    return sorted(r for c in clusters if not c["echo"] for r in c["exemplar_ranks"])


def pseudo_birth(b: dict, B: dict, rng) -> dict | None:
    """The structured control (SPEC §3.3): a random group of another grid pole's own T_15 top-50 (non-echo ranks), of
    the birth's size, in the birth's layer, born at the birth's index k by the same rule. Candidates: the grid poles of
    sample 0 in that layer whose neuron differs, in a seeded random order; up to PB_DRAWS draws each; the first group
    dated k is taken. None if no candidate yields one."""
    cands = sorted(q for q in B["clusters"] if parse_pole(q)[0] == b["layer"] and parse_pole(q)[1] != b["neuron"])
    for q in rng.permutation(np.array(cands, dtype=object)):
        ranks = np.array(nonecho_ranks(B["clusters"][q]))
        if ranks.size < b["size"]:
            continue
        g = np.sort(np.stack([rng.choice(ranks, b["size"], replace=False) for _ in range(PB_DRAWS)]), axis=1)
        i = B["row"][q]
        dates = births_matrix(B["a_plus"][:, i, :], B["q999"][:, i], g)
        hit = np.flatnonzero(dates == b["k"])
        if hit.size:
            ranks_q = g[hit[0]].tolist()
            lq, nq, sq = parse_pole(q)
            return {"pole": q, "layer": lq, "neuron": nq, "sign": sq, "row": i, "ranks": ranks_q,
                    "tok": B["tok15"][i, ranks_q].tolist(), "draws_used": int(hit[0] + 1)}
    return None


def build_births(bank) -> dict:
    """The unit table: every passing birth with its exemplar positions, split half and pseudo-birth twin."""
    B = load_bank(bank)
    out, n_all, n_tested = [], 0, 0
    for r in B["births"]:
        if r["sample"] != 0:
            continue
        n_all += 1
        key = (r["pole"], int(r["c"]))
        if key not in B["per_cluster"]:          # the 26 clusters of m = 1 poles are not tested (10_RECOUNT2.md §8)
            continue
        n_tested += 1
        pc = B["per_cluster"][key]
        assert int(pc["real"]) == r["birth_pct"], f"{key}: per-cluster date {pc['real']} != births.jsonl {r['birth_pct']}"
        if float(pc["p_late"]) > P_PASS:
            continue
        layer, n, s = parse_pole(r["pole"])
        i = B["row"][r["pole"]]
        k = int(births_matrix(B["a_plus"][:, i, :], B["q999"][:, i], np.array([r["ranks"]]))[0])
        assert k == r["birth_pct"], f"{key}: the dating rule gives {k}, the bank {r['birth_pct']}"
        names = {c["c"]: c["name"] for c in B["clusters"][r["pole"]]}
        b = {"id": f"{r['pole']}:c{r['c']}", "pole": r["pole"], "layer": layer, "neuron": n, "sign": s,
             "c": int(r["c"]), "name": names[int(r["c"])], "size": int(r["size"]), "ranks": list(r["ranks"]),
             "tok": B["tok15"][i, r["ranks"]].tolist(), "row": i, "k": k, "step_k": STEPS[k],
             "step_pre": STEPS[k - 1], "half": half_of(n), "p_late": float(pc["p_late"]),
             "is_main": bool(r["is_main"])}
        b["pb"] = pseudo_birth(b, B, np.random.default_rng([SEED_PB, layer, n, s, int(r["c"])]))
        out.append(b)
    for b in out:
        assert all(0 < t < N_WINDOWED and t % WINDOW != 0 for t in b["tok"]), f"{b['id']}: a position off the windows"
    halves = {h: sum(b["half"] == h for b in out) for h in ("freeze", "test")}
    by_k = {h: {str(k): sum(b["half"] == h and b["k"] == k for b in out) for k in sorted({b["k"] for b in out})}
            for h in ("freeze", "test")}
    # The freeze-half look made before the spec was written: the carrier's own median-to-q999 ratio at T_(k-1).
    # No competitor is read here, so no rank and no hit is known from it.
    fr = np.array([np.median(B["a_plus"][b["k"] - 1, b["row"], b["ranks"]]) / B["q999"][b["k"] - 1, b["row"]]
                   for b in out if b["half"] == "freeze"])
    look = {"what": "freeze half only: the carrier's own median a+ over E_c divided by its q999 at T_(k-1)",
            "n": int(fr.size), "quantiles_0_10_25_50_75_90_100": np.round(np.quantile(
                fr, [0, .1, .25, .5, .75, .9, 1]), 4).tolist()}
    from infra.pg import m2_examples
    return {"rule": "p_late <= 0.05 under R1t over the clusters of the m >= 2 poles, sample 0",
            "split": "neuron index even -> freeze, odd -> test", "n_clusters_sample0": n_all,
            "n_clusters_tested": n_tested, "n_births": len(out), "halves": halves,
            "n_with_pb": {h: sum(b["half"] == h and b["pb"] is not None for b in out) for h in ("freeze", "test")},
            "by_k": by_k, "n_poles": len({b["pole"] for b in out}), "n_positions": sum(b["size"] for b in out),
            "freeze_look": look, "examples": m2_examples.select(bank, out), "inputs": B["inputs"], "births": out}


# ------------------------------------------------------------------------------------------------- the window plan
def units(births: list) -> list:
    """The scored units: each birth (roles pre = T_(k-1), birth = T_k, end = T_15) and its pseudo-birth twin (pre, end)."""
    out = []
    for b in births:
        base = {"half": b["half"], "layer": b["layer"], "birth": b["id"], "k": b["k"]}
        out.append({**base, "unit": b["id"], "kind": "real", "carrier": [b["neuron"], b["sign"]], "row": b["row"],
                    "pole": b["pole"], "tok": b["tok"], "ranks": b["ranks"],
                    "roles": {"pre": b["k"] - 1, "birth": b["k"], "end": 15}})
        if b["pb"] is not None:
            p = b["pb"]
            out.append({**base, "unit": b["id"] + ":pb", "kind": "pb", "carrier": [p["neuron"], p["sign"]],
                        "row": p["row"], "pole": p["pole"], "tok": p["tok"], "ranks": p["ranks"],
                        "roles": {"pre": b["k"] - 1, "end": 15}})
    return out


def plan(births: list) -> dict:
    """Per checkpoint: the harvest windows to run and, per window, the (layer, index in the BOS-prepended input, token)
    entries to capture. Window w covers corpus tokens [w * WINDOW, (w + 1) * WINDOW); input index = 1 + tok % WINDOW."""
    us = units(births)
    pts = {}
    for u in us:
        for role, t in u["roles"].items():
            p = pts.setdefault(t, {"need": {}, "roles": []})
            p["roles"].append([u["unit"], role])
            for e in u["tok"]:
                p["need"].setdefault(e // WINDOW, set()).add((u["layer"], 1 + e % WINDOW, e))
    for t, p in pts.items():
        p["windows"] = sorted(p["need"])
        p["need"] = {w: sorted(v) for w, v in p["need"].items()}
    return {"units": us, "points": dict(sorted(pts.items(), reverse=True))}


# ------------------------------------------------------------------------------------------------------- scores
def degenerate(stats: dict) -> np.ndarray:
    """A4: [N, 2] True for a pole with no token of its own sign in the banked harvest at this point. The harvest counts
    every a != 0 in both the histogram and the top-K, so an empty histogram (n_pole_tokens 0) and max_abs 0 must
    coincide; a pole with one and not the other stops the job."""
    empty = np.asarray(stats["n_pole_tokens"]) == 0
    dead = ~(np.asarray(stats["max_abs"]) > 0)
    assert (empty == dead).all(), f"{int((empty != dead).sum())} poles whose histogram and max_abs disagree"
    return empty


def percentile(v: np.ndarray, hist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of each pole's own-sign tokens with |a| below v: v [N, 2], hist [N, 2, 400] (counts of |a| per pole and
    sign, a == 0 dropped), edges [401]. Bins below v's bin count whole; v's own bin counts the log-linear share below v.
    0 when v = 0, and (A4) 0 for a pole with an empty histogram (no token of its sign at this point)."""
    n = hist.sum(-1).astype(np.float64)
    lv = np.log10(np.where(v > 0, v, 1.0))
    b = np.clip(np.floor((lv - HIST_LO) * HIST_B / (HIST_HI - HIST_LO)), 0, HIST_B - 1).astype(np.int64)
    cum = np.concatenate([np.zeros(hist.shape[:-1] + (1,)), np.cumsum(hist, -1, dtype=np.float64)], -1)
    below = np.take_along_axis(cum, b[..., None], -1)[..., 0]
    inbin = np.take_along_axis(hist, b[..., None], -1)[..., 0].astype(np.float64)
    le = np.log10(edges.astype(np.float64))
    frac = np.clip((lv - le[b]) / (le[b + 1] - le[b]), 0.0, 1.0)
    return np.where((v > 0) & (n > 0), (below + frac * inbin) / np.maximum(n, 1.0), 0.0)


def median_a_plus(a: np.ndarray) -> np.ndarray:
    """a [m, N] signed activations at the m exemplar positions -> v [N, 2]: median of a+ for the + and - poles."""
    a = a.astype(np.float64)
    return np.stack([np.median(np.maximum(a, 0.0), 0), np.median(np.maximum(-a, 0.0), 0)], -1)


def scores(a: np.ndarray, topk_tok: np.ndarray, stats: dict, colnorm: np.ndarray, tok: list) -> dict:
    """Every score [N, 2] for one unit at one checkpoint. a [m, N]; topk_tok [N, 2, 50]; stats: hist, hist_edges, q999,
    max_abs, density, n_pole_tokens of the layer at that checkpoint; colnorm [N]; tok = the unit's m exemplar positions.
    A4: a degenerate pole (no token of its sign; its banked q999 is NaN) gets B = 0 and ratio = 0."""
    assert a.shape == (len(tok), topk_tok.shape[0]), (a.shape, len(tok))
    deg = degenerate(stats)
    v = median_a_plus(a)
    return {"A": np.isin(topk_tok, np.asarray(tok, np.int64)).sum(-1).astype(np.float64),
            "B": percentile(v, stats["hist"], stats["hist_edges"]),
            "ratio": np.divide(v, stats["q999"].astype(np.float64), out=np.zeros_like(v), where=~deg),
            "maxabs": stats["max_abs"].astype(np.float64),
            "density": stats["density"].astype(np.float64),
            "colnorm": np.repeat(colnorm.astype(np.float64)[:, None], 2, 1)}


def ranks(score: np.ndarray) -> np.ndarray:
    """rank[x] = number of poles with score >= score[x] (ties count against x); [N, 2] -> [N, 2]."""
    assert not np.isnan(score).any(), "a NaN score"
    s = score.ravel()
    srt = np.sort(s)
    return (s.size - np.searchsorted(srt, s, "left")).reshape(score.shape)


def matched(max_abs: np.ndarray, n: int, s: int, k: int = K_MATCH) -> list:
    """The k poles of the layer nearest the carrier in log max_abs at the scoring checkpoint, the carrier's neuron
    (both poles) excluded; stable order. A4: a pole with max_abs 0 (no token of its sign) is not a candidate; the
    carrier must have a token of its sign, and the layer at least k candidates."""
    ma = max_abs.astype(np.float64)
    assert ma[n, s] > 0, "the carrier has no token of its sign at the scoring checkpoint"
    ok = ma > 0
    ok[n, :] = False
    assert ok.sum() >= k, f"{int(ok.sum())} candidate poles < {k}"
    d = np.full(ma.shape, np.inf)
    d[ok] = np.abs(np.log(ma[ok]) - np.log(ma[n, s]))
    idx = np.argsort(d.ravel(), kind="stable")[:k]
    return [[int(i // 2), int(i % 2)] for i in idx]


def top_list(score: np.ndarray, layer: int, k: int = TOP) -> list:
    idx = np.argsort(-score.ravel(), kind="stable")[:k]
    return [[pole_id(layer, int(i // 2), int(i % 2)), float(score.ravel()[i])] for i in idx]


def record(u: dict, role: str, t: int, sc: dict, max_abs=None) -> dict:
    """One line of ranks_<half>.jsonl: the carrier's rank and value under every score; for the pre role the ranks of
    the K_MATCH activation-matched poles; for the end role the rank of the carrier's opposite pole (the failing input
    of the construction control); the top-10 lists under A and B."""
    n, s = u["carrier"]
    rk = {k: ranks(v) for k, v in sc.items()}
    rec = {"unit": u["unit"], "kind": u["kind"], "half": u["half"], "birth": u["birth"], "role": role, "t": t,
           "layer": u["layer"], "carrier": u["pole"], "rank": {k: int(r[n, s]) for k, r in rk.items()},
           "value": {k: float(v[n, s]) for k, v in sc.items()},
           "top10": {k: top_list(sc[k], u["layer"]) for k in (PRIMARY, REFERENCE)}}
    if role == "pre" and u["kind"] == "real":
        m = matched(max_abs, n, s)
        rec["matched"] = [pole_id(u["layer"], a, b) for a, b in m]
        rec["matched_rank"] = {k: [int(rk[k][a, b]) for a, b in m] for k in (PRIMARY, REFERENCE, "ratio")}
    if role == "end":
        rec["rank_opposite"] = {k: int(rk[k][n, 1 - s]) for k in (PRIMARY, REFERENCE)}
    return rec


# ------------------------------------------------------------------------------------------- known number, identity
def rel_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    den = np.maximum(np.abs(x), np.abs(y))
    return np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1.0), 0.0)


def mixed_diff(new: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """The gate statistic of amendment A1: |new - bank| / (ATOL + |bank|)."""
    new, bank = np.asarray(new, np.float64), np.asarray(bank, np.float64)
    return np.abs(new - bank) / (ATOL + np.abs(bank))


def control_p99(bank: np.ndarray, repeat: np.ndarray) -> dict:
    """A1's control: the statistic of the L40S repeat (new) against the Blackwell bank at T_14, its 99th percentile over
    all entries and over the entries with |a_bank| below 1, 0.5 and 0.25. The same-device tolerance is the larger of
    "all" and "below_1"; the cross-device one is twice "below_0.5" (the revision of A1); "below_0.25" is reported."""
    d = mixed_diff(repeat.ravel(), bank.ravel())
    out = {"all": {"n": int(d.size), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99))}}
    for c in (1, 0.5, 0.25):
        m = np.abs(bank.ravel()) < c
        out[f"below_{c}"] = {"n": int(m.sum()), "median": float(np.median(d[m])), "p99": float(np.quantile(d[m], 0.99))}
    m = np.abs(bank.ravel()) >= EX_ABS_MIN          # A2: the cross-device G-EX tolerance is twice this p99
    out[f"ge_{EX_ABS_MIN}"] = {"n": int(m.sum()), "median": float(np.median(d[m])), "p99": float(np.quantile(d[m], 0.99))}
    return out


def gate_mask(bank, tol: dict) -> np.ndarray:
    """The entries a gate reads: all, or (A2, tol["abs_min"]) those with |bank| >= abs_min."""
    bank = np.asarray(bank, np.float64)
    return np.abs(bank) >= tol["abs_min"] if "abs_min" in tol else np.ones(bank.shape, bool)


def values_gate(new, bank, tol: dict, min_poles: int = 1, n_poles: int | None = None) -> dict:
    """K0, G-EX, G-TOPK and I-ID: the 99th percentile of the A1 statistic within tolerance (median and max reported).
    With tol["abs_min"] (A2, the cross-device G-EX) it is taken over the entries with |bank| >= abs_min; the entries
    below are reported (count, median, p99) and not gated. n and exact_share count every compared entry."""
    r_all = mixed_diff(new, bank)
    keep = gate_mask(bank, tol)
    r = r_all[keep]
    assert r.size > 0, "nothing compared"
    med, p99 = float(np.median(r)), float(np.quantile(r, 0.99))
    ok = p99 <= tol["p99"] and (n_poles is None or n_poles >= min_poles)
    out = {"pass": bool(ok), "n": int(r_all.size), "n_gated": int(r.size), "n_poles": n_poles,
           "statistic": f"|new - bank| / ({ATOL} + |bank|)", "median": med, "p99": p99, "max": float(r.max()),
           "exact_share": float((np.asarray(new) == np.asarray(bank)).mean()), "tol": tol}
    if "abs_min" in tol:
        lo = r_all[~keep]
        out["below_abs_min"] = {"n": int(lo.size), "median": float(np.median(lo)) if lo.size else None,
                                "p99": float(np.quantile(lo, 0.99)) if lo.size else None}
    return out


def tol_for(t: int, device: str) -> dict:
    """K0 and G-TOPK (and the freeze-1c G-EX): the same-device or the cross-device tier."""
    return TOL["same"] if POINTS[t][4] == device else TOL["cross"]


def tol_ex(t: int, device: str) -> dict:
    """G-EX from A2 on: the same-device tier, or across devices the A2 statistic over |a_bank| >= EX_ABS_MIN."""
    return TOL["same"] if POINTS[t][4] == device else TOL["ex_cross"]


def ex_is_gate(t: int, device: str) -> bool:
    """A3: G-EX is a gate only where the bank was made on the forward's device type; elsewhere it is reported."""
    return POINTS[t][4] == device


def cross_device_gex(job_metrics: dict) -> dict:
    """A3: the verdict's line on the reported cross-device G-EX (the job's metrics.json -> reports)."""
    rep = {k: v for k, v in job_metrics.get("reports", {}).items() if k.endswith("_G_EX")}
    over = sorted((int(k[1:3]) for k, v in rep.items() if not v["pass"]), reverse=True)
    pts = {k: {x: v[x] for x in ("n", "n_gated", "median", "p99", "max", "n_above_tol")} for k, v in sorted(rep.items())}
    return {"line": "cross-device G-EX exceeded the A2 tolerance at points " +
                    (", ".join(f"T_{t}" for t in over) if over else "none") +
                    f" (of {len(rep)} reported; tolerance {TOL['ex_cross']['p99']:.6f} on |a_bank| >= {EX_ABS_MIN})",
            "points_over": over, "points": pts}


def degenerate_counts(job_metrics: dict) -> dict:
    """A4: the poles with no token of their sign in the layers each point scored (the job's metrics.json -> points)."""
    pts = {t: {"n": p["n_degenerate_poles"], "poles": p["degenerate_poles"]} for t, p in job_metrics["points"].items()}
    return {"line": "poles with no token of their sign in the scored layers (B = 0, ratio = 0, not matched): " +
                    ", ".join(f"T_{t} {v['n']}" for t, v in pts.items()), "points": pts}


# ----------------------------------------------------------------------------------------------------- statistics
def wilson(h: int, n: int, z: float = 1.959964) -> tuple:
    assert n > 0
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0.0, c - w)), float(min(1.0, c + w))


def floor_reps(matched_hits: np.ndarray, reps: int = N_FLOOR_REPS, seed: int = SEED) -> dict:
    """matched_hits [units, K] bool: replicate r draws one matched pole per unit; its precision at 10."""
    rng = np.random.default_rng(seed)
    u, k = matched_hits.shape
    v = np.array([matched_hits[np.arange(u), rng.integers(0, k, u)].mean() for _ in range(reps)])
    return {"mean": float(v.mean()), "q975": float(np.quantile(v, 0.975)), "max": float(v.max()),
            "all_pairs_mean": float(matched_hits.mean()), "reps": reps, "seed": seed}


def paired(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of x - y over paired units (hits as 0/1) with a percentile bootstrap 95 % interval over units."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(d.size), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "x_only": int((d > 0).sum()), "y_only": int((d < 0).sum())}


def label_exist(h: int, n: int, bar: float) -> dict:
    """SUPPORTED: the Wilson lower bound exceeds the bar. REFUTED: the point value is at or below the bar.
    UNDECIDED otherwise."""
    p, lo, hi = wilson(h, n)
    lab = "SUPPORTED" if lo > bar else ("REFUTED" if p <= bar else "UNDECIDED")
    return {"label": lab, "hits": int(h), "n": int(n), "p": p, "lo": lo, "hi": hi, "bar": float(bar)}


def label_pair(pr: dict, names=("BEATS", "TIES", "LOSES")) -> str:
    return names[0] if pr["lo"] > 0 else (names[2] if pr["hi"] < 0 else names[1])


# --------------------------------------------------------------------------------------------- freeze and verdict
def read_jsonl(p) -> list:
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def by(recs: list, kind: str, role: str) -> dict:
    return {r["birth"]: r for r in recs if r["kind"] == kind and r["role"] == role}


def hits(recs: dict, ids: list, score: str) -> np.ndarray:
    return np.array([recs[i]["rank"][score] <= TOP for i in ids])


def usable(recs: list) -> list:
    """Births with both a pre record and an end record (a parked checkpoint leaves its births out)."""
    pre, end = by(recs, "real", "pre"), by(recs, "real", "end")
    return sorted(set(pre) & set(end))


def half_summary(recs: list, half: str) -> dict:
    """Every number one half gives, under the frozen definitions. No label."""
    assert all(r["half"] == half for r in recs), "records of another half"
    ids = usable(recs)
    pre, birth, end = by(recs, "real", "pre"), by(recs, "real", "birth"), by(recs, "real", "end")
    pbp = by(recs, "pb", "pre")
    out = {"half": half, "n_usable": len(ids), "n_records": len(recs)}
    for sc in SCORES:
        h = hits(pre, ids, sc)
        out[f"pre_{sc}"] = wilson(int(h.sum()), len(ids)) + (int(h.sum()),)
    for sc in (PRIMARY, REFERENCE):
        ib = [i for i in ids if i in birth]
        out[f"birth_{sc}"] = wilson(int(hits(birth, ib, sc).sum()), len(ib)) + (int(hits(birth, ib, sc).sum()),)
        he = hits(end, ids, sc)
        out[f"end_{sc}"] = wilson(int(he.sum()), len(ids)) + (int(he.sum()),)
        ho = np.array([end[i]["rank_opposite"][sc] <= TOP for i in ids])
        out[f"end_opposite_{sc}"] = wilson(int(ho.sum()), len(ids)) + (int(ho.sum()),)
        mh = np.array([[r <= TOP for r in pre[i]["matched_rank"][sc]] for i in ids])
        out[f"floor_{sc}"] = floor_reps(mh)
    # H_AB is hit_A - hit_B as written (SPEC §1.3, §5 step 7), whichever score is primary: the names are explicit
    # (head's decision, 2026-11-05, before freeze 1; with PRIMARY = B the constants would flip its sign).
    out["pair_A_B"] = paired(hits(pre, ids, "A"), hits(pre, ids, "B"))
    tw = [i for i in ids if i in pbp]
    out["n_pb"] = len(tw)
    for sc in (PRIMARY, REFERENCE):
        out[f"pb_pre_{sc}"] = wilson(int(hits(pbp, tw, sc).sum()), max(len(tw), 1)) + (int(hits(pbp, tw, sc).sum()),)
        out[f"pair_real_pb_{sc}"] = paired(hits(pre, tw, sc), hits(pbp, tw, sc)) if tw else None
    return out


def gates_ok(job_metrics: dict) -> dict:
    """Every instance gate passed and every registered failing input failed (from the job's m2/metrics.json)."""
    bad = []
    for name, g in job_metrics["gates"].items():
        if not g["pass"]:
            bad.append(f"{name} failed")
        if "failing_input" in g and g["failing_input"]["pass"]:
            bad.append(f"{name}: its failing input passed")
    return {"pass": not bad and job_metrics.get("complete") is True, "problems": bad,
            "complete": job_metrics.get("complete")}


def room_check() -> dict:
    """The frozen labels reach each branch on planted outcomes of the test half's size (SPEC §6)."""
    n = 71
    r = {"exist_all": label_exist(n, n, 0.01)["label"], "exist_none": label_exist(0, n, 0.01)["label"],
         "pair_x": label_pair(paired(np.ones(n), np.zeros(n))), "pair_y": label_pair(paired(np.zeros(n), np.ones(n))),
         "pair_equal": label_pair(paired(np.ones(n), np.ones(n)))}
    ok = r == {"exist_all": "SUPPORTED", "exist_none": "REFUTED", "pair_x": "BEATS", "pair_y": "LOSES",
               "pair_equal": "TIES"}
    return {"pass": bool(ok), "labels": r}


def freeze(banks) -> dict:
    """Freeze 2: from the job's gates and the freeze half only. Writes nothing about the test half."""
    banks = pathlib.Path(banks)
    jm = json.loads((banks / "m2" / "metrics.json").read_text())
    g = gates_ok(jm)
    assert g["pass"], f"freeze refused: gates {g}"
    fz = read_jsonl(banks / "m2" / "ranks_freeze.jsonl")
    s = half_summary(fz, "freeze")
    assert s["n_usable"] >= N_MIN, f"freeze half has {s['n_usable']} usable births, needs {N_MIN}"
    for sc in (PRIMARY, REFERENCE):
        assert s[f"end_{sc}"][0] >= CTRL_MIN, f"construction control {sc} {s[f'end_{sc}']} below {CTRL_MIN}"
        assert s[f"end_opposite_{sc}"][0] < CTRL_MIN, f"construction control {sc}: its failing input passed"
    rc = room_check()
    assert rc["pass"], f"room check failed: {rc}"
    return {"created_from": str(banks / "m2" / "ranks_freeze.jsonl"),
            "sha256_ranks_freeze": sha256_file(banks / "m2" / "ranks_freeze.jsonl"),
            "sha256_job_metrics": sha256_file(banks / "m2" / "metrics.json"),
            "chance": CHANCE, "floor_freeze": {sc: s[f"floor_{sc}"] for sc in (PRIMARY, REFERENCE)},
            "ctrl_min": CTRL_MIN, "n_min": N_MIN, "room": rc, "gates": g, "discovery": s}


def verdict(banks, th: dict) -> dict:
    """Once, on the test half, with the frozen thresholds. Every value is written whatever the headline."""
    banks = pathlib.Path(banks)
    assert sha256_file(banks / "m2" / "ranks_freeze.jsonl") == th["sha256_ranks_freeze"], "freeze records changed"
    jm = json.loads((banks / "m2" / "metrics.json").read_text())
    g = gates_ok(jm)
    te = read_jsonl(banks / "m2" / "ranks_test.jsonl")
    s = half_summary(te, "test")
    v = {"gates": g, "test": s, "thresholds_sha256": th.get("_sha256"), "labels": {}}
    reasons = [] if g["pass"] else [f"gates: {g['problems'] or 'incomplete'}"]
    if s["n_usable"] < N_MIN:
        reasons.append(f"missing: {s['n_usable']} usable test births < {N_MIN}")
    for sc in (PRIMARY, REFERENCE):
        if s[f"end_{sc}"][0] < CTRL_MIN:
            reasons.append(f"construction control {sc} {s[f'end_{sc}'][0]:.3f} < {CTRL_MIN}")
        if s[f"end_opposite_{sc}"][0] >= CTRL_MIN:
            reasons.append(f"construction control {sc}: failing input passed")
    bars = {sc: max(CHANCE, th["floor_freeze"][sc]["q975"], s[f"floor_{sc}"]["q975"]) for sc in (PRIMARY, REFERENCE)}
    v["bars"] = bars
    for sc in (PRIMARY, REFERENCE):
        lab = label_exist(s[f"pre_{sc}"][3], s["n_usable"], bars[sc])
        b = s[f"birth_{sc}"]
        if lab["lo"] > b[2]:
            lab["label"] = "ABOVE_CEILING"
        v["labels"][f"H_{sc}"] = lab
    v["labels"]["H_AB"] = {"label": label_pair(s["pair_A_B"]), **s["pair_A_B"]}
    for sc in (PRIMARY, REFERENCE):
        pr = s[f"pair_real_pb_{sc}"]
        v["labels"][f"H_sel_{sc}"] = ({"label": "NOT_RUN"} if pr is None or pr["n"] < N_MIN else
                                      {"label": label_pair(pr, ("ABOVE_SELECTION", "AT_SELECTION", "BELOW_SELECTION")),
                                       **pr})
    v["exploratory"] = {sc: s[f"pre_{sc}"] for sc in SCORES if sc not in (PRIMARY, REFERENCE)}
    v["cross_device_gex"] = cross_device_gex(jm)             # A3: reported, never a reason
    v["degenerate_poles"] = degenerate_counts(jm)            # A4: reported, never a reason
    if reasons:
        v["headline"] = "DOES NOT COUNT: " + "; ".join(reasons)
    elif any(v["labels"][f"H_{sc}"]["label"] == "ABOVE_CEILING" for sc in (PRIMARY, REFERENCE)):
        v["headline"] = "DOES NOT COUNT until inspected (ABOVE_CEILING)"
    else:
        L = v["labels"]
        v["headline"] = (f"H_B {L['H_B']['label']}; H_sel_B {L['H_sel_B']['label']}; H_A {L['H_A']['label']}; "
                         f"H_AB {L['H_AB']['label']}")        # A1 item 3: the primary first (cosmetic)
    return v


# ------------------------------------------------------------------------------------------------- GPU projection
def project(births: list, rate=(3215.0, 3334.0), mb_s=(143.0, 234.0)) -> dict:
    """GPU-hours of one launch from the window plan. Per checkpoint: weight download 14.61 GB, load and gates 90 s,
    S3 reads of the needed layers' topk/ and stats/ 30 s, the windows (2,049 positions each, batches of 8, the last
    batch padded) at the harvest's banked tokens per second, record writes 30 s. Plus boot 164 s, the identity
    re-run, corpus fetch 60 s. Two cases: (lowest banked L40S rate, lowest download rate) and (measured T_15 values)."""
    pl = plan(births)
    out = {"n_points": len(pl["points"]), "points": {}}
    tot = [164.0 + 60.0, 164.0 + 60.0]
    for t, p in pl["points"].items():
        nb = -(-len(p["windows"]) // 8)
        tokens = nb * 8 * (WINDOW + 1)
        row = {"windows": len(p["windows"]), "batches": nb, "entries": sum(len(v) for v in p["need"].values())}
        for j in (0, 1):
            sec = 14610.0 / mb_s[j] + 90 + 30 + tokens / rate[j] + 30
            if t == IDENTITY_T:
                pre_w = {e // WINDOW for u in pl["units"] for e in u["tok"] if u["roles"].get("pre") == t}
                sec += -(-len(pre_w) // 8) * 8 * (WINDOW + 1) / rate[j]
            tot[j] += sec
            row[["s_low", "s_measured"][j]] = round(sec)
        out["points"][t] = row
    out["gpu_h_low_rate"] = round(tot[0] / 3600, 2)
    out["gpu_h_measured_rate"] = round(tot[1] / 3600, 2)
    out["window_forwards"] = sum(r["windows"] for r in out["points"].values())
    return out


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["births", "examples", "project", "freeze", "verdict"])
    ap.add_argument("--bank", default="/Users/abaybektursun/projects/interp_research_banks/neurons")
    ap.add_argument("--births", default=None)
    ap.add_argument("--banks", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", required=False)
    a = ap.parse_args(argv)
    if a.cmd == "births":
        r = build_births(a.bank)
        pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
        print("M2_BIRTHS_OK", json.dumps({k: r[k] for k in ("n_births", "halves", "n_with_pb")}))
    elif a.cmd == "examples":
        from infra.pg import m2_examples
        m2_examples.write(a.bank, json.loads(pathlib.Path(a.births).read_text()), pathlib.Path(a.out))
    elif a.cmd == "project":
        r = project(json.loads(pathlib.Path(a.births).read_text())["births"])
        print(json.dumps(r, indent=1))
    elif a.cmd == "freeze":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; a freeze is written once"
        th = freeze(a.banks)
        pathlib.Path(a.out).write_text(json.dumps(th, indent=1))
        print("M2_FREEZE_OK", sha256_file(a.out))
    elif a.cmd == "verdict":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; the verdict runs once"
        th = json.loads(pathlib.Path(a.thresholds).read_text())
        th["_sha256"] = sha256_file(a.thresholds)
        v = verdict(a.banks, th)
        pathlib.Path(a.out).write_text(json.dumps(v, indent=1))
        print(v["headline"])
        print(v["cross_device_gex"]["line"])
        print(v["degenerate_poles"]["line"])


if __name__ == "__main__":
    main()
