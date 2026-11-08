"""REALIZEDNULL pure core: every frozen rule, construction and aggregation as a pure function.

Spec: studies/realizednull/SPEC.md at tag realizednull-freeze-v1.0 (commit 0f0384c8). The "§x.y" in each
docstring points there. The job (infra/aws_pg/jobs/realizednull_pg1.py) and the analysis
(scripts/realizednull_analyze.py) orchestrate; the rules live here and are tested on tiny tensors in
infra/pg/test_realizednull_core.py. No S3, no model, and no filesystem access other than provenance() and
family_params("q14"), which reads GB-3's verdict JSON (the session-2 reference numbers, §3.8, §6.1 G14-PRE).

Fail loud: every public function raises on a bad shape or a non-finite input, and nothing here catches an
error. One place reads non-finite numbers on purpose: gates(), because §6.1 registers a NaN or infinite
positive-control cost (G-POS, G-MEANPOS) as a failing input whose result is a label, not a crash.
"""
import json
import math
import pathlib
import re
import time

import numpy as np
from scipy import stats as sps

from infra.aws_pg.jobs.gutenberg_gb1 import K, N_IN, N_LAYERS, PPL_NONE_BAND, TOPE_MIN_RATIO
from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G

# ---------------------------------------------------------------------------
# Frozen numbers (§6.1, §6.2), each with the artifact key it comes from.
# ---------------------------------------------------------------------------
# research/gutenberg_gb2c_o1twin.json → cost_ratio_energymatched_over_core[0], → gain_ratio_energymatched_over_core
GB2C_COST_RATIO = 2.9023283157273467
GB2C_GAIN_RATIO = 2.2317194866907966
DELTA0 = math.sqrt(GB2C_COST_RATIO / GB2C_GAIN_RATIO)          # §6.2: 1.14039
ALPHA = 0.05                                                    # §6.2, per step
POWER_MIN = 0.8                                                 # §6.4
S_STAR = 0.22                                                   # §6.5 top-up trigger
THETA_H2 = 1.30                                                 # §6.2: the GB-2c residual, as registered
EPS_M_FLOOR = 0.002                                             # §6.2
REL_PRE = 2e-3                                                  # §6.1 G-PRE0, G-PRE1
ID_MAX = 1e-3                                                   # §6.1 G-ID1, G-ID3
ID2_MAX = 0.02                                                  # §6.1 G-ID2
SCALE_RANGE = (5.0, 500.0)                                      # §6.1 G-SCALE
COST_CEILING = 20.0                                             # §6.1 G-POS, G-MEANPOS
MEANPOS_FLOOR = 0.1                                             # §6.1 G-MEANPOS
BRACKET_MAX_SHARE = 0.10                                        # §6.1 G-BRACKET
EMATCH_TOL = 0.10                                               # §6.1 G-EMATCH
MATCH_SUM_TOL, MATCH_HEAVY_TOL = 0.01, 0.055                    # §6.1 G-MATCH
GEN_TOL = 0.05                                                  # §6.1 G-GEN
PIVOT_MAX_SHARE = EXHAUST_MAX_SHARE = 0.25                      # §6.1 G-PIVOT, G-EXHAUST
DRAWS_MIN_SHARE = 0.8                                           # §6.1 G-DRAWS: 32 of 40, 48 of 60
DOSE_RANGE = (1.41, 4.0)                                        # §6.1 G-DOSE
PSEUDO_MAX = 0.10                                               # §6.1 G-PSEUDO
BOOT_MAX_SHARE = 0.01                                           # §6.1 G-BOOT
CROSSJOB_MAX = 0.005                                            # §6.1 G-CROSSJOB
HEAVY_FRAC = 0.01                                               # §2.2, §6.1 G-MATCH heavy layers
BOOT_SEED, N_BOOT = 20261101, 10000                             # §2.1, §7
BOOT_BLOCK = 500                                                # replicates drawn per block (fixes the stream)
REDRAW_STRIDE = 10_000_000                                      # §2.2 step 6
GRID_SEED_BASE = 190_000                                        # §2.2, §3.2: reach_grid seed 190_000 + L

# research/gutenberg_gb1_verdict.json → sites.q.ppl (GB-1, windows 0–97); "none" = gates.ppl_none
GB1_PPL_Q = {
    "none": 22.02970644601963,
    "core32": 25.651377364420576,
    "anti32": 1301.924405786236,
    "bottomsigma32": 22.021591340103345,
    "topsigma32": 1325.8643332245006,
    "topenergy32": 10859.74604175714,
    "energymatched32_s0": 34.684349555832725,
    "energymatched32_s1": 30.59601856648921,
    "energymatched32_s2": 37.9119918265937,
    "specmatched32_s0": 34.3704542304772,
    "specmatched32_s1": 28.823903724709215,
    "specmatched32_s2": 32.82475075634106,
    "haar32_s0": 22.21881215040791,
    "haar32_s1": 22.07144858628743,
    "haar32_s2": 22.235379406931226,
}


def _gb1_dnll(arm):
    return math.log(GB1_PPL_Q[arm] / GB1_PPL_Q["none"])


# §6.2: 2.842 = geometric mean of GB-1's per-draw EM/band ratios, from sites.q.ppl
THETA_GB1 = math.exp(np.mean([math.log(_gb1_dnll(f"energymatched32_s{s}") / _gb1_dnll("core32"))
                              for s in range(3)]))

# §6.1 G-PRE2–4: this run's arm name → GB-1's arm name
PRE_ARMS = {
    "band": "core32", "em_s0": "energymatched32_s0", "em_s1": "energymatched32_s1",
    "em_s2": "energymatched32_s2", "spec_s0": "specmatched32_s0", "spec_s1": "specmatched32_s1",
    "spec_s2": "specmatched32_s2", "topenergy32": "topenergy32", "topsigma32": "topsigma32",
    "anti32": "anti32", "haar_s0": "haar32_s0", "haar_s1": "haar32_s1", "haar_s2": "haar32_s2",
    "bottomsigma32": "bottomsigma32",
}
PRE4_GATING = ("em_s1", "em_s2", "spec_s0", "spec_s1", "spec_s2", "topenergy32", "topsigma32", "anti32")
PRE4_RECORDED = ("haar_s0", "haar_s1", "haar_s2", "bottomsigma32")

MODES = ("causal", "bidir", "prefixlm")
# confirmatory statistic → the family of draws it is read on, and its does-not-count label (§6.1, §6.4)
STAT_FAMILY = {"E": "em", "M": "em_mean", "R": "rt", "T": "emt"}
STAT_DNC = {"E": "E-DOES-NOT-COUNT", "M": "M-DOES-NOT-COUNT", "R": "RT-DOES-NOT-COUNT",
            "T": "T-DOES-NOT-COUNT"}
CONSTRUCTED = ("rt", "rtall", "rthalf", "rt2x", "emt")          # RT-family and EM_T (§3.2 A7–A10, A15)
REACH_LABEL = {"rt": "RT-DOES-NOT-COUNT", "emt": "T-DOES-NOT-COUNT", "rthalf": "excluded from G-DOSE",
               "rt2x": "excluded from G-DOSE", "rtall": "marked"}
# exploratory multi-draw families and the operation whose references bracket them (§6.1 G-BRACKET)
EXPLORATORY_OP = {"spec": "zero", "rtall": "zero", "rthalf": "zero", "rt2x": "zero", "em_zt": "zero",
                  "rt_zt": "zero", "rt_mean": "mean", "spec_mean": "mean"}
ENV_KEYS = ("transformers", "torch", "numpy", "cuda", "gpu_name", "instance_type")

# ---------------------------------------------------------------------------
# Family parameter sets (§3.8): which reference numbers and shapes a run is read with. §6's rules are the same
# for both. pg1: session 1 (PaliGemma-1, the job N3). q14: session 2 (the Qwen2.5-14B census core O3, the cell N7).
# ---------------------------------------------------------------------------
PG1_BASE = "google/paligemma-3b-pt-224"          # N3's env.json → base, before the "@" (§0.3)
GB3_VERDICT = (pathlib.Path(__file__).resolve().parents[2]
               / "studies/lineage/paligemma1-pi0/research/gutenberg_gb3_verdict.json")
# G14-PRE's reference numbers in GB-3's verdict JSON (§6.1 "Session 2 preflight"), by key path
GB3_KEYS = {"ppl_rel": "reproduction_gate.tol", "checked": "reproduction_gate.checked",
            "dnll_core32": "dnll.core32[0]", "E_over_C": "E_over_C[0]", "em_arms": "fresh_draw_arms.ematch",
            "n_windows": "n_windows"}
G14_DNLL_TOL = (0.002, 0.02)                     # §6.1 G14-PRE: band ΔNLL within max(0.002, 2%) of dnll.core32[0]
G14_EC_TOL = 0.05                                # §6.1 G14-PRE: E/C within 5% of E_over_C[0]
D_Q14, LAYERS_Q14 = 5120, 48                     # §3.8: Qwen2.5-14B q_proj input width and layer count
# The families a run must contain, each with the env.json → args key that sets its draw count: N3's names
# (draws_*) and N7's (realizednull_q14.py cell_args); session 2 has no RT_all (§3.8).
PG1_ARGS = {"em": "draws_em", "em_mean": "draws_em", "rt": "draws_rt", "emt": "draws_emt", "spec": "draws_spec",
            "rtall": "draws_rtall", "rthalf": "draws_dose", "rt2x": "draws_dose", "haar": "draws_haar",
            "em_zt": "draws_zt", "rt_zt": "draws_zt", "rt_mean": "draws_mean_rt", "spec_mean": "draws_mean_spec",
            "haar_mean": "draws_mean_haar"}
Q14_ARGS = {"em": "em", "em_mean": "em", "rt": "rt", "emt": "emt", "spec": "spec", "rthalf": "dose", "rt2x": "dose",
            "haar": "haar", "em_zt": "zt", "rt_zt": "zt", "rt_mean": "mean_rt", "spec_mean": "mean_spec",
            "haar_mean": "mean_haar"}
RUN_GATES = ("G-ENV", "G-PRE1", "G-PRE2", "G-PRE3", "G-PRE4", "G-ID1", "G-ID3", "G-ID2", "G-SCALE", "G-POS")
PG1 = {"name": "pg1", "attn": "eager", "d_model": N_IN, "n_layers": N_LAYERS, "ematch_floor": K / N_IN,
       "args": PG1_ARGS, "constructed": CONSTRUCTED, "scale_range": SCALE_RANGE, "run_gates": RUN_GATES,
       "preflight": {"ppl": GB1_PPL_Q, "ppl_none_band": PPL_NONE_BAND, "rel": REL_PRE,
                     "source": "research/gutenberg_gb1_verdict.json → sites.q.ppl, gates.ppl_none; "
                               "gutenberg_gb1.py:36 → PPL_NONE_BAND"}}


def key_path(doc, path):
    """The value at a key path such as "dnll.core32[0]": keys joined by '.', list items as [i]."""
    x = doc
    for key, idx in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
        x = x[key] if key else x[int(idx)]
    return x


def family_params(name):
    """A family's parameter set by name. q14's G14-PRE references are read from GB-3's verdict JSON by the key
    paths in GB3_KEYS, here and not at import: the vast box ships infra/, configs/ and scripts/ only, and the
    cell imports this module."""
    if name == "pg1":
        return PG1
    if name != "q14":
        raise ValueError(f"no parameter set for family {name!r}")
    v = json.loads(GB3_VERDICT.read_text())
    pre = {k: key_path(v, p) for k, p in GB3_KEYS.items()}
    pre.update(dnll_tol=G14_DNLL_TOL, ec_tol=G14_EC_TOL,
               source={k: f"research/gutenberg_gb3_verdict.json → {p}" for k, p in GB3_KEYS.items()})
    return {"name": "q14", "attn": "sdpa", "d_model": D_Q14, "n_layers": LAYERS_Q14, "ematch_floor": K / D_Q14,
            "args": Q14_ARGS, "constructed": tuple(f for f in CONSTRUCTED if f != "rtall"),
            "scale_range": SCALE_RANGE,
            "run_gates": ("G-ENV", "G14-PRE", "G-ID1", "G-ID3", "G-ID2", "G-SCALE", "G-POS"), "preflight": pre}


def family_of(env):
    """A run's parameter set, from its env.json: the cell N7 writes family "q14" with d_model and n_layers (checked
    here); the job N3 writes no family and base "google/paligemma-3b-pt-224@<sha>"."""
    if "family" in env:
        fam = family_params(env["family"])
        if (env["d_model"], env["n_layers"]) != (fam["d_model"], fam["n_layers"]):
            raise ValueError(f"env.json: d_model {env['d_model']}, n_layers {env['n_layers']}; family "
                             f"{fam['name']} is {fam['d_model']}, {fam['n_layers']}")
        return fam
    if env["base"].split("@")[0] != PG1_BASE:
        raise ValueError(f"env.json has no family and base {env['base']!r} is not {PG1_BASE}")
    return PG1


PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "realizednull_provenance.json"


# ---------------------------------------------------------------------------
# Input checks (fail loud).
# ---------------------------------------------------------------------------
def _arr(x, ndim, what):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise ValueError(f"{what}: expected a {ndim}-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what}: non-finite entries")
    return a


def _square(x, n, what):
    a = _arr(x, 2, what)
    if a.shape != (n, n):
        raise ValueError(f"{what}: expected shape ({n}, {n}), got {a.shape}")
    return a


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what}: expected a number, got {type(x).__name__}")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{what}: non-finite value {v}")
    return v


def _int(x, what, lo):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) < lo:
        raise ValueError(f"{what}: expected an integer >= {lo}, got {x!r}")
    return int(x)


def _t(alpha, D):
    return float(sps.t.ppf(1.0 - alpha / 2.0, D - 1))


# ---------------------------------------------------------------------------
# §2.2 — eigen-coordinates and the realized quantity
# ---------------------------------------------------------------------------
def eigencoords(sigma_all, M, sigma_S):
    """§2.2: the eigenbasis of the all-position Gram and the two matrices N_L is read from.

    sigma_all: Σ_all = Σ_{t ∈ all positions} h_t h_tᵀ on C. M: W_qᵀ W_q (the identity for EM_T). sigma_S: the
    Gram over the position set S of the matched quantity (T for RT and EM_T, all positions for RT_all).
    Returns V, mu (eigenvectors and eigenvalues of Σ_all, descending, eigenvalues clamped ≥ 0, as
    gutenberg_gb1.py:238-240), Mp = Vᵀ M V, Sp = Vᵀ Σ_S V (fp64, symmetrized), and
    denom = tr(M Σ_S) = Σ_{t∈S} ‖W_q h_t‖², the realized-share denominator (§0.3)."""
    sigma_all = _arr(sigma_all, 2, "sigma_all")
    n = sigma_all.shape[0]
    sigma_all = _square(sigma_all, n, "sigma_all")
    M = _square(M, n, "M")
    sigma_S = _square(sigma_S, n, "sigma_S")
    mu, V = np.linalg.eigh(sigma_all)
    mu = np.clip(mu[::-1], 0.0, None)
    V = np.ascontiguousarray(V[:, ::-1])
    Mp = V.T @ M @ V
    Sp = V.T @ sigma_S @ V
    return {"V": V, "mu": mu, "Mp": (Mp + Mp.T) / 2.0, "Sp": (Sp + Sp.T) / 2.0,
            "denom": float((M * sigma_S).sum())}


def realized_quantity(Ue, Mp, Sp):
    """§2.2: N_L(U) = tr((Ueᵀ M′ Ue)(Ueᵀ S′ Ue)) for Ue = Vᵀ U, equal to Σ_{t∈S} ‖W_q U Uᵀ h_t‖²; with M′ the
    identity it is the text input energy E_L(U) (§0.3)."""
    Ue = _arr(Ue, 2, "Ue")
    n = Ue.shape[0]
    Mp = _square(Mp, n, "Mp")
    Sp = _square(Sp, n, "Sp")
    A = Ue.T @ Mp @ Ue
    B = Ue.T @ Sp @ Ue
    return float(np.trace(A @ B))


def _block(n, m, family):
    """Index range of the top-m or bottom-m eigenvectors (eigen-coordinates are descending)."""
    if family == "top":
        return slice(0, m)
    if family == "bottom":
        return slice(n - m, n)
    raise ValueError(f"family must be 'top' or 'bottom', got {family!r}")


def _block_quantity(R, Mp, Sp, sl):
    """realized_quantity of the frame that is R on the rows `sl` and zero elsewhere, in O(m²k)."""
    Mb = Mp[sl, sl]
    Sb = Sp[sl, sl]
    A = R.T @ (Mb @ R)
    B = R.T @ (Sb @ R)
    return float((A * B.T).sum())


def _embed(R, n, sl):
    Ue = np.zeros((n, R.shape[1]), dtype=np.float64)
    Ue[sl] = R
    return Ue


def _grid_m(k, n, n_grid):
    m = np.unique(np.round(np.geomspace(k, n, n_grid)).astype(np.int64))
    if m.size != n_grid:
        raise ValueError(f"{n_grid} log-spaced integers in [{k}, {n}] are not distinct (got {m.size})")
    return m


def reach_grid(Mp, Sp, k=32, n_grid=64, n_probe=16, *, seed):
    """§2.2 grid scan, once per layer: for each of n_grid log-spaced integers m in [k, n] and each family
    ("top": Haar k-frames on the top-m eigenvectors; "bottom": on the bottom-m), the median of N_L over
    n_probe probe frames. Stream: np.random.default_rng(seed), m ascending, "top" before "bottom", n_probe
    frames each (G.haar_frame). The job passes seed = 190_000 + L (§3.2). Nothing assumes the medians are
    monotone in m."""
    Mp = _arr(Mp, 2, "Mp")
    n = Mp.shape[0]
    Mp = _square(Mp, n, "Mp")
    Sp = _square(Sp, n, "Sp")
    k = _int(k, "k", 1)
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")
    n_probe = _int(n_probe, "n_probe", 1)
    m = _grid_m(k, n, _int(n_grid, "n_grid", 3))
    rng = np.random.default_rng(_int(seed, "seed", 0))
    med = {"top": np.empty(m.size), "bottom": np.empty(m.size)}
    for i, mi in enumerate(m):
        for fam in ("top", "bottom"):
            sl = _block(n, int(mi), fam)
            med[fam][i] = np.median([_block_quantity(G.haar_frame(int(mi), k, rng), Mp, Sp, sl)
                                     for _ in range(n_probe)])
    return {"m": m, "median_top": med["top"], "median_bottom": med["bottom"]}


def _bracket_choice(grid, target):
    """§2.2 step 1: the (family, grid index) whose probe median is closest to the target, among grid points
    whose two neighbours' medians bracket the target. None if no grid point brackets it."""
    best = None
    for fam in ("top", "bottom"):
        med = grid[f"median_{fam}"]
        for i in range(1, med.size - 1):
            if min(med[i - 1], med[i + 1]) <= target <= max(med[i - 1], med[i + 1]):
                d = abs(med[i] - target)
                if best is None or d < best[0]:
                    best = (d, fam, i)
    return None if best is None else (best[1], best[2])


def _closest_choice(grid, target):
    """A light layer with no bracketing grid point: the grid point (either family) whose median is closest
    to the target. Light layers are free (§2.2 step 2), so this only fixes where their one frame is drawn."""
    best = None
    for fam in ("top", "bottom"):
        med = grid[f"median_{fam}"]
        for i in range(med.size):
            d = abs(med[i] - target)
            if best is None or d < best[0]:
                best = (d, fam, i)
    return best[1], best[2]


def _replay(state, n, k, m, family):
    """Regenerate a frame drawn earlier in an attempt from the generator state saved before the draw."""
    g = np.random.Generator(np.random.PCG64(0))
    g.bit_generator.state = state
    return _embed(G.haar_frame(m, k, g), n, _block(n, m, family))


def realized_matched_subspace(Mp, Sp, targets, seed, k=32, tol_sum=0.005, tol_layer=0.05,
                              heavy_frac=HEAVY_FRAC, n_grid=64, n_probe=16, max_resample=20000,
                              max_closing=50, grids=None):
    """§2.2 "Construction of one draw": one attempt at a realized-matched frame per layer. Never raises on
    a failed match; the result says what happened.

    Mp, Sp: lists of per-layer (n, n) fp64 matrices from eigencoords (Mp = identity for EM_T). targets:
    per-layer N*_L (0.5× or 2× for the dose families). grids: per-layer reach_grid results; None computes
    them with seed 190_000 + L (§3.2). Passing them precomputed gives the same result and saves the scan
    per attempt. Per-layer streams are np.random.default_rng([seed, L]) (§2.2 step 6).

    Step 1: per layer, the bracketing grid point closest to the target (_bracket_choice); a heavy layer
    (N*_L ≥ heavy_frac · N*) with no bracketing point, or N* outside [Σ_L min median, Σ_L max median],
    returns reachable = False (G-REACH). A light layer with no bracketing point takes the closest median.
    Step 2: heavy layers draw at m*_L until |N_L/N*_L − 1| ≤ tol_layer, max_resample draws, then at the grid
    neighbours −1, +1, −2, +2 (max_resample draws each); still outside → matched = False. Light layers: one draw.
    Step 3: up to max_closing rounds while |ΣN/N* − 1| > tol_sum. The heaviest layer's residual target is
    r = N* − Σ_{L≠hl} N_L, which is §2.2's N*_L − (Σ N − N*) with the sums over the other layers. If
    |r/N*_hl − 1| ≤ tol_layer the heaviest layer is redrawn toward r; otherwise the heavy layer whose
    deviation N_L − N*_L has the sign of the summed error and the largest size is redrawn toward its own
    target. A round draws up to max_resample frames and keeps the first within 0.002·N* of its goal that
    also keeps the layer within tol_layer; a round with no candidate layer ends the closing.
    Step 4 (φ pivot): if the sum is still off, the drawn frame of the heaviest layer closest above r and the
    one closest below r give U(φ) = orth(cos φ U0 + sin φ U1); φ ∈ [0, π/2] is bisected, at most 60 halvings,
    until the sum is within tol_sum (pivot = True). No frame above or no frame below r, or 60 halvings
    without closing, gives matched = False.
    Acceptance: N recomputed with realized_quantity on the final frames; matched iff the sum is within
    tol_sum and every heavy layer within tol_layer. Ue frames are checked orthonormal to 1e-10.

    Returns the keys of §2.2 step 5 (Ue, N, N_sum, family, m, draws, pivot, matched, reachable,
    unreachable_layers, redraw, seconds) and fail_reason (None when matched). Ue, N and N_sum are None
    when matched is False."""
    t0 = time.perf_counter()
    if len(Mp) != len(Sp):
        raise ValueError(f"Mp and Sp have {len(Mp)} and {len(Sp)} layers")
    n_layers = len(Mp)
    targets = _arr(targets, 1, "targets")
    if targets.shape != (n_layers,) or np.any(targets <= 0):
        raise ValueError(f"targets: need {n_layers} positive values, got {targets}")
    n = np.asarray(Mp[0]).shape[0]
    Mp = [_square(x, n, f"Mp[{L}]") for L, x in enumerate(Mp)]
    Sp = [_square(x, n, f"Sp[{L}]") for L, x in enumerate(Sp)]
    k = _int(k, "k", 1)
    seed = _int(seed, "seed", 0)
    max_resample = _int(max_resample, "max_resample", 1)
    max_closing = _int(max_closing, "max_closing", 0)
    if grids is None:
        grids = [reach_grid(Mp[L], Sp[L], k=k, n_grid=n_grid, n_probe=n_probe, seed=GRID_SEED_BASE + L)
                 for L in range(n_layers)]
    if len(grids) != n_layers:
        raise ValueError(f"grids: {len(grids)} layers, expected {n_layers}")
    n_grid = len(grids[0]["m"])
    for L, g in enumerate(grids):
        for key in ("m", "median_top", "median_bottom"):
            if np.asarray(g[key]).shape != (n_grid,):
                raise ValueError(f"grids[{L}][{key}] has shape {np.asarray(g[key]).shape}")
        _arr(g["median_top"], 1, f"grids[{L}].median_top")
        _arr(g["median_bottom"], 1, f"grids[{L}].median_bottom")

    N_star = float(targets.sum())
    heavy = [L for L in range(n_layers) if targets[L] >= heavy_frac * N_star]
    hl = int(np.argmax(targets))
    fam, idx, unreachable = [None] * n_layers, [None] * n_layers, []
    for L in range(n_layers):
        choice = _bracket_choice(grids[L], targets[L])
        if choice is None and L in heavy:
            unreachable.append(L)
            continue
        fam[L], idx[L] = choice if choice is not None else _closest_choice(grids[L], targets[L])
    lo_sum = sum(min(g["median_top"].min(), g["median_bottom"].min()) for g in grids)
    hi_sum = sum(max(g["median_top"].max(), g["median_bottom"].max()) for g in grids)
    if not lo_sum <= N_star <= hi_sum:
        unreachable.append("sum")

    draws = np.zeros(n_layers, dtype=np.int64)

    def m_at(L, i):
        return int(grids[L]["m"][i])

    def result(matched, reason, Ue=None, N=None, pivot=False):
        return {"Ue": Ue, "N": N, "N_sum": None if N is None else float(N.sum()), "family": list(fam),
                "m": [None if idx[L] is None else m_at(L, idx[L]) for L in range(n_layers)],
                "draws": draws.copy(), "pivot": pivot, "matched": matched,
                "reachable": not unreachable, "unreachable_layers": list(unreachable), "redraw": 0,
                "seconds": time.perf_counter() - t0, "fail_reason": reason}

    if unreachable:
        return result(False, f"unreachable: {unreachable}")

    rngs = [np.random.default_rng([seed, L]) for L in range(n_layers)]
    hl_log = []                                  # (N, generator state before the draw, m, family)

    def draw(L, i):
        mi = m_at(L, i)
        sl = _block(n, mi, fam[L])
        state = rngs[L].bit_generator.state if L == hl else None
        R = G.haar_frame(mi, k, rngs[L])
        q = _block_quantity(R, Mp[L], Sp[L], sl)
        draws[L] += 1
        if L == hl:
            hl_log.append((q, state, mi, fam[L]))
        return _embed(R, n, sl), q

    def within(L, q):
        return abs(q / targets[L] - 1.0) <= tol_layer

    # step 2
    Ue = [None] * n_layers
    N = np.zeros(n_layers)
    for L in range(n_layers):
        if L not in heavy:
            Ue[L], N[L] = draw(L, idx[L])
            continue
        tried = [idx[L]] + [idx[L] + d for d in (-1, 1, -2, 2) if 0 <= idx[L] + d < n_grid]
        for j in tried:
            for _ in range(max_resample):
                U_, q = draw(L, j)
                if within(L, q):
                    Ue[L], N[L], idx[L] = U_, q, j
                    break
            if Ue[L] is not None:
                break
        if Ue[L] is None:
            return result(False, f"heavy layer {L} outside tol_layer after the grid neighbours")

    def sum_ok(N):
        return abs(N.sum() / N_star - 1.0) <= tol_sum

    # step 3
    tol_abs = 0.002 * N_star
    for _ in range(max_closing):
        if sum_ok(N):
            break
        err = N.sum() - N_star
        r = N[hl] - err
        if abs(r / targets[hl] - 1.0) <= tol_layer:
            L, goal = hl, r
        else:
            cand = [L for L in heavy if (N[L] - targets[L]) * err > 0]
            if not cand:
                break
            L = max(cand, key=lambda j: abs(N[j] - targets[j]))
            goal = targets[L]
        for _ in range(max_resample):
            U_, q = draw(L, idx[L])
            if abs(q - goal) <= tol_abs and within(L, q):
                Ue[L], N[L] = U_, q
                break

    # step 4
    pivot = False
    if not sum_ok(N):
        others = N.sum() - N[hl]
        r = N_star - others
        above = [e for e in hl_log if e[0] > r]
        below = [e for e in hl_log if e[0] < r]
        if not above or not below:
            return result(False, "phi pivot cannot bracket the residual target")
        a = min(above, key=lambda e: e[0])
        b = max(below, key=lambda e: e[0])
        U0 = _replay(a[1], n, k, a[2], a[3])
        U1 = _replay(b[1], n, k, b[2], b[3])
        lo, hi = 0.0, math.pi / 2.0
        for _ in range(60):
            phi = (lo + hi) / 2.0
            Uphi = np.linalg.qr(math.cos(phi) * U0 + math.sin(phi) * U1)[0]
            q = realized_quantity(Uphi, Mp[hl], Sp[hl])
            if abs((others + q) / N_star - 1.0) <= tol_sum:
                Ue[hl], N[hl], pivot = Uphi, q, True
                break
            if q > r:
                lo = phi
            else:
                hi = phi
        if not pivot:
            return result(False, "phi pivot did not close the sum in 60 halvings")

    for L in range(n_layers):
        C.orthonormal_or_die(Ue[L], f"realized-matched frame L{L}", tol=1e-10)
    N = np.array([realized_quantity(Ue[L], Mp[L], Sp[L]) for L in range(n_layers)])
    if not sum_ok(N):
        return result(False, "summed quantity outside tol_sum", pivot=pivot)
    if not all(within(L, N[L]) for L in heavy):
        return result(False, "a heavy layer outside tol_layer at acceptance", pivot=pivot)
    return result(True, None, Ue=Ue, N=N, pivot=pivot)


def construct_draw(Mp, Sp, targets, base_seed, max_redraws=3, **kw):
    """§2.2 step 6: one draw index. Calls realized_matched_subspace with seed base_seed + 10_000_000·r for
    r = 0 … max_redraws until an attempt is matched. Returns that attempt's result with redraw = r and
    exhausted = not matched; seconds is the total over the attempts. No raise and no loop beyond r = 3."""
    base_seed = _int(base_seed, "base_seed", 0)
    max_redraws = _int(max_redraws, "max_redraws", 0)
    total = 0.0
    for r in range(max_redraws + 1):
        res = realized_matched_subspace(Mp, Sp, targets, base_seed + REDRAW_STRIDE * r, **kw)
        total += res["seconds"]
        if res["matched"]:
            break
    res["redraw"] = r
    res["exhausted"] = not res["matched"]
    res["seconds"] = total
    return res


# ---------------------------------------------------------------------------
# §2.3, §3.6 — means, centered Grams, per-arm statistics
# ---------------------------------------------------------------------------
def class_means(sums, counts):
    """§2.3: μ_c = (Σ_{t∈c} h_t) / n_c per position class c (each prefix position, and the text block)."""
    if set(sums) != set(counts) or not sums:
        raise ValueError(f"class keys differ or are empty: {sorted(sums)} vs {sorted(counts)}")
    out, n = {}, None
    for c in sums:
        s = _arr(sums[c], 1, f"sums[{c}]")
        n = s.size if n is None else n
        if s.size != n:
            raise ValueError(f"sums[{c}] has length {s.size}, expected {n}")
        out[c] = s / _int(counts[c], f"counts[{c}]", 1)
    return out


def centered_gram(sigma, class_sums, class_counts):
    """§3.6: Σ_c Σ_{t∈c} (h_t − μ_c)(h_t − μ_c)ᵀ = Σ − Σ_c s_c s_cᵀ / n_c, where Σ, s_c and n_c are summed
    over the same positions, so μ_c = s_c / n_c is the class mean of exactly those positions."""
    sigma = _arr(sigma, 2, "sigma")
    n = sigma.shape[0]
    sigma = _square(sigma, n, "sigma")
    mu = class_means(class_sums, class_counts)
    out = sigma.copy()
    for c, m in mu.items():
        if m.size != n:
            raise ValueError(f"class {c}: mean has length {m.size}, expected {n}")
        out -= class_counts[c] * np.outer(m, m)
    return out


def arm_stats(U_by_layer, grams, W_by_layer, means):
    """§3.6 columns for one arm on one window split, per layer.

    U_by_layer: per-layer (n, k) orthonormal bases in model coordinates (k ≥ 1). W_by_layer: per-layer W_q.
    grams: per-layer lists under "all" (Σ over all positions), "T" (Σ over the scored text positions T),
    "V" (descending eigenvectors of the construction Gram, from eigencoords), "denom_all" = tr(M Σ_all) and
    "denom_T" = tr(M Σ_T) (eigencoords' denom). means: None (no centered columns, e.g. on D) or per-layer
    lists under "mu" ((n_classes, n) class means on C, class_means), "counts" ((n_classes,) positions per
    class in the "all" Gram), "text" (row of the text-block class), "T_sum" (Σ_{t∈T} h_t) and "T_count" (|T|).

    Columns: e_all, e_T (uncentered input energy shares), E_T = Σ_T ‖Uᵀh‖², N_T and rho_T, N_all and
    rho_all (realized perturbation and share), w_top1 = ‖Uᵀv1‖², w_top5 = ‖UᵀV[:, :5]‖²_F,
    leverage_max = max_i ‖Uᵀe_i‖², gain = ‖W_q U‖²_F / k (G.reader_gain); with means: e_all_centered (each
    position minus its class mean μ_c) and N_T_centered = Σ_{t∈T} ‖W_q U Uᵀ (h_t − μ_text)‖², the realized
    perturbation of mean-ablation at T. Sums over layers: N_T_sum, N_all_sum, E_T_sum."""
    n_layers = len(U_by_layer)
    if len(W_by_layer) != n_layers:
        raise ValueError("U_by_layer and W_by_layer differ in length")
    keys = ["e_all", "e_T", "E_T", "N_T", "rho_T", "N_all", "rho_all", "w_top1", "w_top5",
            "leverage_max", "gain"]
    if means is not None:
        keys += ["e_all_centered", "N_T_centered"]
    out = {key: [] for key in keys}
    for L in range(n_layers):
        U = _arr(U_by_layer[L], 2, f"U[{L}]")
        n, k = U.shape
        if k < 1:
            raise ValueError(f"U[{L}] has no columns")
        W = _arr(W_by_layer[L], 2, f"W[{L}]")
        if W.shape[1] != n:
            raise ValueError(f"W[{L}] has shape {W.shape}, U has {n} rows")
        S_all = _square(grams["all"][L], n, f"grams.all[{L}]")
        S_T = _square(grams["T"][L], n, f"grams.T[{L}]")
        V = _square(grams["V"][L], n, f"grams.V[{L}]")
        WU = W @ U
        A = WU.T @ WU                                 # Uᵀ M U
        P_all = U.T @ S_all @ U
        P_T = U.T @ S_T @ U
        N_T = float(np.trace(A @ P_T))
        N_all = float(np.trace(A @ P_all))
        out["e_all"].append(float(np.trace(P_all) / np.trace(S_all)))
        out["e_T"].append(float(np.trace(P_T) / np.trace(S_T)))
        out["E_T"].append(float(np.trace(P_T)))
        out["N_T"].append(N_T)
        out["rho_T"].append(N_T / _num(grams["denom_T"][L], f"denom_T[{L}]"))
        out["N_all"].append(N_all)
        out["rho_all"].append(N_all / _num(grams["denom_all"][L], f"denom_all[{L}]"))
        out["w_top1"].append(float(((U.T @ V[:, 0]) ** 2).sum()))
        out["w_top5"].append(float(((U.T @ V[:, :5]) ** 2).sum()))
        out["leverage_max"].append(float((U ** 2).sum(axis=1).max()))
        out["gain"].append(G.reader_gain(W, U))
        if means is not None:
            mu = _arr(means["mu"][L], 2, f"means.mu[{L}]")
            cnt = _arr(means["counts"][L], 1, f"means.counts[{L}]")
            if mu.shape[1] != n or cnt.shape != (mu.shape[0],):
                raise ValueError(f"means at L{L}: mu {mu.shape}, counts {cnt.shape}")
            Pm = mu @ U                               # (n_classes, k)
            num = np.trace(P_all) - float((cnt[:, None] * Pm ** 2).sum())
            den = np.trace(S_all) - float((cnt * (mu ** 2).sum(axis=1)).sum())
            out["e_all_centered"].append(float(num / den))
            a = U.T @ _arr(means["T_sum"][L], 1, f"means.T_sum[{L}]")
            b = Pm[_int(means["text"][L], "means.text", 0)]
            nT = _int(means["T_count"][L], "means.T_count", 1)
            P_Tc = P_T - np.outer(a, b) - np.outer(b, a) + nT * np.outer(b, b)
            out["N_T_centered"].append(float(np.trace(A @ P_Tc)))
    for key in ("N_T", "N_all", "E_T"):
        out[f"{key}_sum"] = float(np.sum(out[key]))
    return out


# ---------------------------------------------------------------------------
# §2.4 — attention-mask identification
# ---------------------------------------------------------------------------
def mask_probe_verdict(d_before, d_after, r_prior, r_self):
    """§2.4: text label CAUSAL iff d_before ≤ 1e-3, BIDIRECTIONAL iff d_before ≥ 0.05, positive control
    d_after ≥ 0.05; image label IMAGE-BIDIR iff r_prior ≥ 1e-3, IMAGE-CAUSAL iff r_prior ≤ 1e-6, positive
    control r_self ≥ 1e-3; otherwise MASK-DOES-NOT-COUNT. Mode: causal = (CAUSAL, IMAGE-CAUSAL); prefixlm =
    (CAUSAL, IMAGE-BIDIR); bidir whenever the text label is BIDIRECTIONAL (§2.4 conditions it on nothing
    else); any MASK-DOES-NOT-COUNT probe gives MASK-DOES-NOT-COUNT."""
    vals = {"d_before": d_before, "d_after": d_after, "r_prior": r_prior, "r_self": r_self}
    for name, v in vals.items():
        if _num(v, name) < 0:
            raise ValueError(f"{name} is a magnitude, got {v}")
    if d_after < 0.05 or r_self < 1e-3:
        return "MASK-DOES-NOT-COUNT"
    text = "CAUSAL" if d_before <= 1e-3 else "BIDIRECTIONAL" if d_before >= 0.05 else None
    image = "IMAGE-BIDIR" if r_prior >= 1e-3 else "IMAGE-CAUSAL" if r_prior <= 1e-6 else None
    if text is None or image is None:
        return "MASK-DOES-NOT-COUNT"
    if text == "BIDIRECTIONAL":
        return "bidir"
    return "causal" if image == "IMAGE-CAUSAL" else "prefixlm"


# ---------------------------------------------------------------------------
# §2.1, §6.4, §7 — ratios and intervals
# ---------------------------------------------------------------------------
def _Y_pair(Y, Y_band):
    Y = _arr(Y, 2, "Y")
    Y_band = _arr(Y_band, 1, "Y_band")
    if Y.shape[1] != Y_band.shape[0]:
        raise ValueError(f"Y has {Y.shape[1]} windows, Y_band has {Y_band.shape[0]}")
    return Y, Y_band


def per_draw_ratios(Y, Y_band):
    """§2.1: r_d = c(twin_d) / c(band), c = mean over the split's windows of Y(arm, w). A ratio of costs is
    defined only for positive costs; a non-positive cost raises (G-BOOT and G-MDENOM decide before this)."""
    Y, Y_band = _Y_pair(Y, Y_band)
    c_band = float(Y_band.mean())
    c = Y.mean(axis=1)
    if c_band <= 0 or np.any(c <= 0):
        raise ValueError(f"non-positive cost: band {c_band}, draws min {c.min()}")
    return c / c_band


def decision_interval(Y, Y_band, alpha=ALPHA, n_boot=N_BOOT, seed=BOOT_SEED):
    """§2.1 decision interval: θ = exp(mean_d ln r_d); I = hull of (i) the log-t interval
    exp(mean ln r ± t_{1−α/2, D−1} · sd(ln r)/√D) and (ii) a percentile two-way bootstrap (draws and windows
    resampled independently; one window resample shared by the band and every draw), both at level 1 − α.
    Bootstrap stream: default_rng(seed), blocks of 500 replicates, window indices then draw indices per block.
    A replicate with a non-positive mean cost in the band or any draw is counted in n_degenerate and left out
    of the percentiles (G-BOOT reads n_degenerate / n_boot). When the full-sample costs are not all positive,
    theta, lo, hi and logt are None."""
    Y, Y_band = _Y_pair(Y, Y_band)
    D, W = Y.shape
    if D < 2:
        raise ValueError(f"need at least 2 draws, got {D}")
    n_boot = _int(n_boot, "n_boot", 1)
    rng = np.random.default_rng(_int(seed, "seed", 0))
    logs, n_deg = [], 0
    for start in range(0, n_boot, BOOT_BLOCK):
        nb = min(BOOT_BLOCK, n_boot - start)
        w = rng.integers(0, W, (nb, W))
        d = rng.integers(0, D, (nb, D))
        cb = Y_band[w].mean(axis=1)                               # (nb,)
        cd = np.take_along_axis(Y[:, w].mean(axis=2).T, d, axis=1)  # (nb, D)
        good = (cb > 0) & np.all(cd > 0, axis=1)
        n_deg += int((~good).sum())
        logs.append(np.mean(np.log(cd[good]), axis=1) - np.log(cb[good]))
    logs = np.concatenate(logs)
    boot = None
    if logs.size:
        q = np.percentile(logs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        boot = (float(np.exp(q[0])), float(np.exp(q[1])))
    out = {"theta": None, "lo": None, "hi": None, "logt": None, "boot": boot, "n_degenerate": n_deg,
           "n_boot": n_boot, "D": D}
    c_band, c = float(Y_band.mean()), Y.mean(axis=1)
    if c_band <= 0 or np.any(c <= 0) or boot is None:
        return out
    lr = np.log(per_draw_ratios(Y, Y_band))
    m, s = float(lr.mean()), float(lr.std(ddof=1))
    h = _t(alpha, D) * s / math.sqrt(D)
    logt = (math.exp(m - h), math.exp(m + h))
    out.update(theta=math.exp(m), logt=logt, lo=min(logt[0], boot[0]), hi=max(logt[1], boot[1]))
    return out


def difference_interval(c_draws, c_band, alpha=ALPHA):
    """§6.4 step M under G-MDENOM: Δ_d = c(em_mean_d) − c(band_mean); returns (mean, lo, hi) of the
    t-interval over draws, t = t_{1−α/2, D−1}."""
    c = _arr(c_draws, 1, "c_draws")
    if c.size < 2:
        raise ValueError(f"need at least 2 draws, got {c.size}")
    d = c - _num(c_band, "c_band")
    m, h = float(d.mean()), _t(alpha, d.size) * float(d.std(ddof=1)) / math.sqrt(d.size)
    return m, m - h, m + h


def prediction_interval(x_draws, alpha=ALPHA):
    """§6.4 step R: PI = [m − t·s·√(1 + 1/D), m + t·s·√(1 + 1/D)] over the log costs x_d of the draws,
    t = t_{1−α/2, D−1}."""
    x = _arr(x_draws, 1, "x_draws")
    if x.size < 2:
        raise ValueError(f"need at least 2 draws, got {x.size}")
    D = x.size
    h = _t(alpha, D) * float(x.std(ddof=1)) * math.sqrt(1.0 + 1.0 / D)
    return float(x.mean()) - h, float(x.mean()) + h


# ---------------------------------------------------------------------------
# §6.2 — power, pseudo-band calibration, thresholds
# ---------------------------------------------------------------------------
def power_figures(s, D, delta, alpha=ALPHA):
    """§6.2 power figures with s treated as known and the log-ratio estimate ~ N(ln θ, s²/D); the interval is
    ln θ̂ ± t·s/√D, t = t_{1−α/2, D−1} (§3.2's table is computed this way).
      power_E = min(P(hi < δ | θ = 1), P(lo > δ | θ = 2.842)); power_M uses the same formula on its family.
      p_control_rule_at_1 = P(I ⊂ [1/δ, δ] | θ = 1); p_lo_at_1.30 = P(lo > δ | θ = 1.30).
      p_beyond_at_1.30 = P(lo > δ and the band below the PI | θ = 1.30); the band is below the PI iff
      ln θ̂ > t·s·√(1 + 1/D).
      h2_reachable_at_1.30 = ln 1.30 > t·s·√(1 + 1/D).
      power_T = min(P(lo > δ | θ = 2.842), P(I ⊂ [1/δ, δ] | θ = 1)).
    Also t and smallest_theta_pi_clause = exp(t·s·√(1 + 1/D)) (§3.2's last column)."""
    s = _num(s, "s")
    D = _int(D, "D", 2)
    delta = _num(delta, "delta")
    if s <= 0 or delta <= 1:
        raise ValueError(f"need s > 0 and delta > 1, got s={s}, delta={delta}")
    t = _t(alpha, D)
    se = s / math.sqrt(D)
    h = t * se
    ld = math.log(delta)
    Phi = sps.norm.cdf

    def p_lo_above(theta):
        return float(Phi((math.log(theta) - ld - h) / se))

    def p_inside(theta):
        return float(max(0.0, Phi((ld - h - math.log(theta)) / se) - Phi((h - ld - math.log(theta)) / se)))

    pi_half = t * s * math.sqrt(1.0 + 1.0 / D)
    return {
        "t": t,
        "power_E": min(float(Phi((ld - h) / se)), p_lo_above(THETA_GB1)),
        "p_control_rule_at_1": p_inside(1.0),
        "p_lo_at_1.30": p_lo_above(THETA_H2),
        "p_beyond_at_1.30": float(Phi((math.log(THETA_H2) - max(ld + h, pi_half)) / se)),
        "h2_reachable_at_1.30": bool(math.log(THETA_H2) > pi_half),
        "power_T": min(p_lo_above(THETA_GB1), p_inside(1.0)),
        "smallest_theta_pi_clause": math.exp(pi_half),
    }


def pseudo_band_calibration(Y_draws, delta, alpha=ALPHA):
    """§6.1 G-PSEUDO: f_H2 = share of the draws that, each treated as the band against the other draws,
    return BEYOND-* under §6.4's step-R rule: the decision interval entirely outside [1/δ, δ] and the
    pseudo-band's log cost outside the others' prediction interval on the same side. A pseudo-band whose
    interval is not formed (non-positive costs) does not return BEYOND-*."""
    Y = _arr(Y_draws, 2, "Y_draws")
    delta = _num(delta, "delta")
    D = Y.shape[0]
    if D < 3:
        raise ValueError(f"need at least 3 draws, got {D}")
    hits = []
    for j in range(D):
        band, others = Y[j], np.delete(Y, j, axis=0)
        di = decision_interval(others, band, alpha)
        if di["theta"] is None:
            continue
        x_b = math.log(band.mean())
        lo_pi, hi_pi = prediction_interval(np.log(others.mean(axis=1)), alpha)
        if (di["lo"] > delta and x_b < lo_pi) or (di["hi"] < 1 / delta and x_b > hi_pi):
            hits.append(j)
    return {"f_H2": len(hits) / D, "beyond_draws": hits, "D": D}


def thresholds_from_controls(discovery):
    """§6.2 and §6.5, on the discovery windows D (control data only), written to thresholds.json.

    Reads discovery["families"][stat] = {"s": sd(ln r_d) on D or None, "D": valid draws after any top-up}
    for stat in E, M, R, T; discovery["c_D"] (ident_mean, band_dup, band); discovery["f_H2"].
    δ = δ0 = √(2.9023…/2.2317…); ε_M = max(0.002, 2|c_D(ident_mean)|, 2|c_D(band_dup) − c_D(band)|).
    Power figures from power_figures per family (power_M is power_E's formula on the EM-mean family); a
    family with s None gets None. Top-up (§6.5): fires iff sd(ln r_d) on D exceeds 0.22 for any family;
    run_families names what R4 draws ("em" runs under both zero and mean ablation when E or M triggers)."""
    fam = discovery["families"]
    c = discovery["c_D"]
    eps_M = max(EPS_M_FLOOR, 2 * abs(_num(c["ident_mean"], "c_D.ident_mean")),
                2 * abs(_num(c["band_dup"], "c_D.band_dup") - _num(c["band"], "c_D.band")))
    pf = {}
    for stat in "EMRT":
        s = fam[stat]["s"]
        pf[stat] = None if s is None else power_figures(s, fam[stat]["D"], DELTA0)
    exceeding = [st for st in "EMRT" if fam[st]["s"] is not None and fam[st]["s"] > S_STAR]
    run = sorted({"em" if st in "EM" else "rt" if st == "R" else "emt" for st in exceeding})

    def get(stat, key):
        return None if pf[stat] is None else pf[stat][key]

    return {
        "delta": DELTA0, "alpha": ALPHA, "s_star": S_STAR, "eps_M": eps_M,
        "theta_gb1": THETA_GB1, "theta_h2": THETA_H2,
        "power_E": get("E", "power_E"), "power_M": get("M", "power_E"),
        "p_control_rule_at_1": get("R", "p_control_rule_at_1"), "p_lo_at_1.30": get("R", "p_lo_at_1.30"),
        "p_beyond_at_1.30": get("R", "p_beyond_at_1.30"),
        "h2_reachable_at_1.30": get("R", "h2_reachable_at_1.30"),
        "power_T": get("T", "power_T"),
        "f_H2": _num(discovery["f_H2"], "f_H2"),
        "families": {st: {"s": fam[st]["s"], "D": fam[st]["D"]} for st in "EMRT"},
        "power_by_family": pf,
        "topup": {"fires": bool(exceeding), "stats_exceeding": exceeding, "run_families": run},
    }


# ---------------------------------------------------------------------------
# §6.1 — gates
# ---------------------------------------------------------------------------
def _gate(value, ok, label, gating=True, **extra):
    return {"value": value, "pass": bool(ok), "label": None if ok else label, "gating": gating, **extra}


def _cost(x):
    """A positive-control cost read as given: NaN and inf are gate results for G-POS and G-MEANPOS."""
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError(f"expected a number, got {type(x).__name__}")
    return float(x)


def _match_ok(v):
    N = _arr(v["N"], 1, "verify_C.N")
    t = _arr(v["targets"], 1, "verify_C.targets")
    if N.shape != t.shape:
        raise ValueError(f"verify_C: N {N.shape} vs targets {t.shape}")
    heavy = t >= HEAVY_FRAC * t.sum()
    return bool(abs(N.sum() / t.sum() - 1) <= MATCH_SUM_TOL
                and np.all(np.abs(N[heavy] / t[heavy] - 1) <= MATCH_HEAVY_TOL))


def _gen_ok(g):
    return abs(_num(g["N_sum"], "gen_D.N_sum") / _num(g["target_sum"], "gen_D.target_sum") - 1) <= GEN_TOL


def _pre_gb1(jm, ppl, pre, is_r3):
    """§6.1 G-PRE0 (smoke only) and G-PRE1–4 (session 1) against GB-1's numbers in the pg1 parameter set."""
    out = {}
    # G-PRE0 (smoke only)
    if "pre0" in jm:
        pairs = (("none", "none"), ("band", "core32"), ("em_s0", "energymatched32_s0"))
        rel = {}
        for arm, gb1_arm in pairs:
            a = _arr(jm["pre0"]["run"][arm], 1, f"pre0.run.{arm}")
            b = _arr(jm["pre0"]["gb1"][gb1_arm], 1, f"pre0.gb1.{gb1_arm}")
            if a.shape != b.shape:
                raise ValueError(f"pre0 {arm}: {a.shape} vs {b.shape}")
            rel[arm] = float(np.max(np.abs(a / b - 1)))
        out["G-PRE0"] = _gate(rel, all(v <= pre["rel"] for v in rel.values()), "pivot P1-smoke")

    # G-PRE1
    gb1, band = pre["ppl"], pre["ppl_none_band"]
    ppl_none = _num(ppl["none"], "ppl_R.none")
    ref = jm["pre1_reference"]
    if is_r3 != (ref is not None):
        raise ValueError("pre1_reference is R2's none_causal ppl for R3, and None for every other run")
    if ref is None:
        ok = band[0] <= ppl_none <= band[1] and abs(ppl_none / gb1["none"] - 1) <= pre["rel"]
    else:
        ok = abs(ppl_none / _num(ref, "pre1_reference") - 1) <= pre["rel"]
    out["G-PRE1"] = _gate({"ppl_none": ppl_none, "reference": gb1["none"] if ref is None else ref},
                          ok, "pivot P1")

    # G-PRE2–4: |ΔNLL_run − ΔNLL_GB1| ≤ max(0.002, 0.02·ΔNLL_GB1), ΔNLL = ln(ppl(arm)/ppl(none))
    def one(arm):
        d_run = math.log(_num(ppl[arm], f"ppl_R.{arm}") / ppl_none)
        d_gb1 = math.log(gb1[PRE_ARMS[arm]] / gb1["none"])
        tol = max(0.002, 0.02 * d_gb1)
        return {"dnll": d_run, "dnll_gb1": d_gb1, "tol": tol, "pass": abs(d_run - d_gb1) <= tol}

    for gid, arms in (("G-PRE2", ("band",)), ("G-PRE3", ("em_s0",)), ("G-PRE4", PRE4_GATING)):
        rows = {arm: one(arm) for arm in arms}
        if gid == "G-PRE4":
            rows.update({arm: {**one(arm), "recorded_only": True} for arm in PRE4_RECORDED})
        bad = [arm for arm in arms if not rows[arm]["pass"]]
        out[gid] = _gate(rows, not bad, f"RUN-DOES-NOT-COUNT ({gid}: {', '.join(bad)})", gating=not is_r3)
    return out


def _g14pre(rec, ppl, pre):
    """§6.1 G14-PRE (session 2, the vast path), read again from the cell's g14pre.json with GB-3's numbers from
    the q14 parameter set: every reproduction arm's ppl on windows 0–97 within rel reproduction_gate.tol of GB-3's
    row (the row the cell read from v1abl_rows.json and recorded as gb3_ppl), the band's ΔNLL within
    max(0.002, 2%) of dnll.core32[0], and E/C over the fresh_draw_arms.ematch slots within 5% of E_over_C[0]. The
    run's numbers are ppl_R's: with ntok equal in every window, mean per-window ΔNLL = ln(ppl(arm)/ppl(none)).
    The record must name GB-3's checked arms and hold the same run numbers, references and outcome, else this
    raises. §6.6 registers no response to a miss; it is read as a run gate, like G-PRE2–4."""
    w = f"_w0_{pre['n_windows'] - 1}"
    rows_in = rec["rows"]
    if sorted(r["gb3_arm"] for r in rows_in.values()) != sorted(pre["checked"]):
        raise ValueError(f"g14pre.json checks {sorted(r['gb3_arm'] for r in rows_in.values())}, GB-3 checked "
                         f"{sorted(pre['checked'])}")
    none = _num(ppl["none"], "ppl_R.none")
    rows, recorded = {}, []
    for arm, r in rows_in.items():
        p = _num(ppl[arm], f"ppl_R.{arm}")
        recorded.append((f"rows.{arm}.ppl{w}", r[f"ppl{w}"], p))
        rel = abs(p / _num(r["gb3_ppl"], f"g14pre.rows.{arm}.gb3_ppl") - 1)
        rows[arm] = {"ppl": p, "gb3_arm": r["gb3_arm"], "gb3_ppl": r["gb3_ppl"], "rel": rel,
                     "pass": rel <= pre["ppl_rel"]}
    d_band = math.log(_num(ppl["band"], "ppl_R.band") / none)
    tol = max(pre["dnll_tol"][0], pre["dnll_tol"][1] * pre["dnll_core32"])
    band = {"dnll": d_band, "gb3": pre["dnll_core32"], "tol": tol, "pass": abs(d_band - pre["dnll_core32"]) <= tol}
    em = [f"em_s{int(a.rsplit('_s', 1)[1])}" for a in pre["em_arms"]]
    ec = float(np.mean([math.log(_num(ppl[a], f"ppl_R.{a}") / none) for a in em])) / d_band
    ec_rel = abs(ec / pre["E_over_C"] - 1)
    e_c = {"E_over_C": ec, "em_arms": em, "gb3": pre["E_over_C"], "rel": ec_rel, "pass": ec_rel <= pre["ec_tol"]}
    bad = ([a for a, r in rows.items() if not r["pass"]]
           + [k for k, x in (("band ΔNLL", band), ("E/C", e_c)) if not x["pass"]])
    recorded += [(f"band_dnll.dnll{w}", rec["band_dnll"][f"dnll{w}"], d_band),
                 ("band_dnll.gb3", rec["band_dnll"]["gb3"], pre["dnll_core32"]),
                 (f"E_over_C.E_over_C{w}", rec["E_over_C"][f"E_over_C{w}"], ec),
                 ("E_over_C.gb3", rec["E_over_C"]["gb3"], pre["E_over_C"]),
                 ("ppl_rel_tol", rec["ppl_rel_tol"], pre["ppl_rel"])]
    differ = [k for k, a, b in recorded if not math.isclose(_num(a, k), b, rel_tol=1e-9)]
    if differ or bool(rec["pass"]) != (not bad):
        raise ValueError(f"g14pre.json differs from G14-PRE read again: {differ or 'pass'}")
    return _gate({"rows": rows, "band_dnll": band, "E_over_C": e_c, "ppl_rel_tol": pre["ppl_rel"],
                  "source": pre["source"]}, not bad, f"RUN-DOES-NOT-COUNT (G14-PRE: {', '.join(bad)})")


def gates(job_metrics, discovery, params=PG1):
    """§6.1: every gate as {value, pass, label, gating}. label is None when the gate passes; gating False
    marks a record the verdict does not stop on (G-MASK, per-draw exclusions, exploratory families, and
    G-PRE2–4 in R3). The verdict (§6.4) reads gates by id; family gates carry the family or statistic after
    a colon (G-REACH:rt, G-DRAWS:R, G-BRACKET:E, …).

    params: the run's family parameter set (family_of(env.json)); PG1 is session 1's. It sets G-ENV's attention,
    the preflight (G-PRE0–4 against GB-1, or G14-PRE against GB-3 from job_metrics → g14pre), G-EMATCH's k/d
    floor, the constructed families and G-SCALE's range; every rule is §6.1's for both (§3.8).

    job_metrics (job side, never F): env; run ("R1"…"R5"); mask (the --mask flag); mask_probe {native:
    probe numbers, forced_masks: "on"|"off", forced: {mode: probe numbers}}; pre0 (smoke only) {run: {none,
    band, em_s0: nll[0..5]}, gb1: {none, core32, energymatched32_s0: nll[0..5]}}; ppl_R {arm: ppl on 0–97};
    pre1_reference (R3: R2's none_causal ppl; otherwise None); ematch (list per EM slot: None for a banked
    GB-1 basis, else per-layer {e_target, e_realized}); construction {family: [construct_draw record]};
    verify_C {family: [{N, targets} or None]}; gen_D {family: [{N_sum, target_sum} or None]};
    readout_validity {stat: {fit, positive_control, identity}}; g14pre (q14 only: the cell's g14pre.json).
    discovery (analysis side, on the phase's windows): c_D {band, band_dup, ident_mean, topenergy32,
    topsigma32_mean}; c_D_draws {family: [c_D per draw index, None where the arm was not run]} with haar,
    haar_mean, em, em_mean, rt, emt, rthalf, rt2x and any exploratory family; Y_band_mean_D (per-window Y of
    band_mean on D); id_max {ident_zero, ident_mean} and id_windows ("0-97" or "98-195"); boot {stat:
    {n_degenerate, n_boot}} (M absent when G-MDENOM fires); f_H2; noleak {leaks: [...]}; planned_draws
    {stat: 40 or 60}; c_D_band_R4 (R4 only)."""
    out = {}
    jm, dc = job_metrics, discovery
    is_r3 = jm["run"] == "R3"

    # G-ENV
    env = jm["env"]
    missing = [k for k in ENV_KEYS if not env.get(k)]
    attn = env.get("_attn_implementation")
    out["G-ENV"] = _gate({"missing": missing, "_attn_implementation": attn},
                         not missing and attn == params["attn"], "RUN-DOES-NOT-COUNT")

    # G-MASK (§2.4)
    mp = jm["mask_probe"]
    native = mask_probe_verdict(**mp["native"])
    if mp["forced_masks"] not in ("on", "off"):
        raise ValueError(f"forced_masks must be 'on' or 'off', got {mp['forced_masks']!r}")
    forced_on = mp["forced_masks"] == "on"
    if forced_on and set(mp["forced"]) != set(MODES):
        raise ValueError(f"forced masks on but probes for {sorted(mp['forced'])}")
    selftest = {mode: mask_probe_verdict(**mp["forced"][mode]) == mode for mode in MODES} if forced_on else {}
    if jm["mask"] not in ("native",) + MODES:
        raise ValueError(f"mask flag {jm['mask']!r}")
    decision_mode = native if jm["mask"] == "native" else jm["mask"]
    r3_required = decision_mode != "causal"
    mask_unknown = r3_required and not (forced_on and selftest["causal"])
    out["G-MASK"] = _gate({"native": native, "decision_mode": decision_mode, "forced_masks": mp["forced_masks"],
                           "selftest": selftest}, native in MODES, "MASK-DOES-NOT-COUNT", gating=False,
                          r3_required=r3_required, mask_unknown=mask_unknown)

    ppl = jm["ppl_R"]
    if params["name"] == "q14":
        out["G14-PRE"] = _g14pre(jm["g14pre"], ppl, params["preflight"])
    else:
        out.update(_pre_gb1(jm, ppl, params["preflight"], is_r3))

    # G-ID1 / G-ID3, G-ID2
    if dc["id_windows"] not in ("0-97", "98-195"):
        raise ValueError(f"id_windows {dc['id_windows']!r}")
    for gid, arm in (("G-ID1", "ident_zero"), ("G-ID3", "ident_mean")):
        v = _num(dc["id_max"][arm], f"id_max.{arm}")
        out[gid] = _gate({"max_abs_Y": v, "windows": dc["id_windows"]}, v <= ID_MAX, "RUN-DOES-NOT-COUNT")
    cD = dc["c_D"]
    band, dup = _num(cD["band"], "c_D.band"), _num(cD["band_dup"], "c_D.band_dup")
    s_id = abs(math.log(dup / band)) if band > 0 and dup > 0 else None
    out["G-ID2"] = _gate(s_id, s_id is not None and s_id <= ID2_MAX, "RUN-DOES-NOT-COUNT")

    # G-SCALE, G-POS
    draws = dc["c_D_draws"]
    haar = [_num(x, "c_D_draws.haar") for x in draws["haar"]]
    haar_max = max(haar)
    scale = band / haar_max if haar_max > 0 else None
    lo, hi = params["scale_range"]
    out["G-SCALE"] = _gate(scale, scale is not None and lo <= scale <= hi, "RUN-DOES-NOT-COUNT")
    te = _cost(cD["topenergy32"])
    out["G-POS"] = _gate({"c_topenergy32": te, "haar_max": haar_max},
                         math.isfinite(te) and te >= TOPE_MIN_RATIO * haar_max and te <= COST_CEILING,
                         "RUN-DOES-NOT-COUNT")

    # valid draws: EM by G-EMATCH; RT-family and EM_T by construction, G-MATCH and G-GEN
    valid, rejected = {}, []
    for s, rows in enumerate(jm["ematch"]):
        if rows is None:
            continue
        bad = [L for L, r in enumerate(rows)
               if _num(r["e_target"], "e_target") >= params["ematch_floor"]
               and abs(_num(r["e_realized"], "e_realized") - r["e_target"]) > EMATCH_TOL * r["e_target"]]
        if bad:
            rejected.append({"slot": s, "layers": bad})
    out["G-EMATCH"] = _gate(rejected, not rejected, "draw does not count", gating=False)
    em_valid = [True] * len(jm["ematch"])
    for r in rejected:
        em_valid[r["slot"]] = False
    valid["em"] = valid["em_mean"] = em_valid

    for fam in params["constructed"]:
        recs, ver, gen = jm["construction"][fam], jm["verify_C"][fam], jm["gen_D"][fam]
        if not len(recs) == len(ver) == len(gen) or not recs:
            raise ValueError(f"{fam}: {len(recs)} records, {len(ver)} verifications, {len(gen)} D checks")
        unreach = [i for i, r in enumerate(recs) if not r["reachable"]]
        layers = recs[unreach[0]]["unreachable_layers"] if unreach else []
        gating = fam in ("rt", "emt")
        out[f"G-REACH:{fam}"] = _gate({"unreachable_draws": unreach, "unreachable_layers": layers},
                                      not unreach, REACH_LABEL[fam], gating=gating)
        piv = sum(bool(r["pivot"]) for r in recs) / len(recs)
        out[f"G-PIVOT:{fam}"] = _gate(piv, piv <= PIVOT_MAX_SHARE, REACH_LABEL[fam], gating=gating)
        exh = sum(bool(r["exhausted"]) for r in recs) / len(recs)
        out[f"G-EXHAUST:{fam}"] = _gate(exh, exh <= EXHAUST_MAX_SHARE, REACH_LABEL[fam], gating=gating)
        v, mfail, gfail = [], [], []
        for i, r in enumerate(recs):
            if not r["matched"]:
                v.append(False)
                continue
            mok, gok = _match_ok(ver[i]), _gen_ok(gen[i])
            if not mok:
                mfail.append(i)
            if not gok:
                gfail.append(i)
            v.append(mok and gok)
        out[f"G-MATCH:{fam}"] = _gate(mfail, not mfail, "draw does not count", gating=False)
        out[f"G-GEN:{fam}"] = _gate(gfail, not gfail, "draw does not count", gating=False)
        valid[fam] = v

    # G-DRAWS
    for stat, fam in STAT_FAMILY.items():
        n_valid = int(sum(valid[fam]))
        need = DRAWS_MIN_SHARE * _int(dc["planned_draws"][stat], f"planned_draws.{stat}", 1)
        out[f"G-DRAWS:{stat}"] = _gate({"valid": n_valid, "planned": dc["planned_draws"][stat]},
                                       n_valid >= need, STAT_DNC[stat])

    def valid_costs(fam):
        c, v = draws[fam], valid[fam]
        if len(c) != len(v):
            raise ValueError(f"{fam}: {len(c)} costs for {len(v)} draw indices")
        return [(i, _num(x, f"c_D_draws.{fam}[{i}]")) for i, (x, ok) in enumerate(zip(c, v)) if ok]

    # G-MDENOM (decided before G-BRACKET for M, which then applies only the ceiling)
    hmean_max = max(_num(x, "c_D_draws.haar_mean") for x in draws["haar_mean"])
    y = _arr(dc["Y_band_mean_D"], 1, "Y_band_mean_D")
    m_bm, lo_bm, hi_bm = G.boot_mean_ci(y, n_boot=N_BOOT, seed=BOOT_SEED)
    fires = (lo_bm <= 0 <= hi_bm) or hi_bm < hmean_max
    out["G-MDENOM"] = _gate({"c_band_mean": m_bm, "ci": [lo_bm, hi_bm], "haar_mean_max": hmean_max},
                            True, None, gating=False, fires=bool(fires))

    # G-BRACKET
    ts_mean = _cost(cD["topsigma32_mean"])
    refs = {"zero": (haar_max, te), "mean": (-math.inf if fires else hmean_max, ts_mean)}

    def bracket(pairs, op):
        lo, hi = refs[op]
        flags = [i for i, c in pairs if not lo < c < hi]
        return flags, (len(flags) / len(pairs) if pairs else 0.0)

    for stat, fam in STAT_FAMILY.items():
        flags, share = bracket(valid_costs(fam), "mean" if stat == "M" else "zero")
        out[f"G-BRACKET:{stat}"] = _gate({"share": share, "flagged": flags,
                                          "references": list(refs["mean" if stat == "M" else "zero"])},
                                         share <= BRACKET_MAX_SHARE, STAT_DNC[stat])
    for fam, op in EXPLORATORY_OP.items():
        if fam in draws:
            pairs = valid_costs(fam) if fam in valid else [
                (i, _num(x, f"c_D_draws.{fam}[{i}]")) for i, x in enumerate(draws[fam]) if x is not None]
            flags, share = bracket(pairs, op)
            out[f"G-BRACKET:{fam}"] = _gate({"share": share, "flagged": flags}, share <= BRACKET_MAX_SHARE,
                                            "marked", gating=False)

    # G-DOSE
    def usable(fam):
        return (out[f"G-REACH:{fam}"]["pass"] and out[f"G-PIVOT:{fam}"]["pass"]
                and out[f"G-EXHAUST:{fam}"]["pass"] and any(valid[fam]))

    def gmean(fam):
        c = np.array([x for _, x in valid_costs(fam)])
        if np.any(c <= 0):
            raise ValueError(f"{fam}: non-positive cost in a dose geometric mean")
        return float(np.exp(np.mean(np.log(c))))

    steps, max2x = {}, None
    if any(valid["rt"]):
        if usable("rthalf"):
            steps["rt/rthalf"] = gmean("rt") / gmean("rthalf")
        if usable("rt2x"):
            steps["rt2x/rt"] = gmean("rt2x") / gmean("rt")
            max2x = max(x for _, x in valid_costs("rt2x"))
    ok = all(DOSE_RANGE[0] <= v <= DOSE_RANGE[1] for v in steps.values()) and (max2x is None or max2x < te)
    out["G-DOSE"] = _gate({"steps": steps, "max_rt2x": max2x, "c_topenergy32": te}, ok, "RT-DOES-NOT-COUNT",
                          n_steps=len(steps))

    # G-MEANPOS
    floor = max(MEANPOS_FLOOR, TOPE_MIN_RATIO * hmean_max)
    out["G-MEANPOS"] = _gate({"c_topsigma32_mean": ts_mean, "floor": floor},
                             math.isfinite(ts_mean) and floor <= ts_mean <= COST_CEILING, "M-DOES-NOT-COUNT")

    # G-PSEUDO
    f = _num(dc["f_H2"], "f_H2")
    out["G-PSEUDO"] = _gate(f, f <= PSEUDO_MAX, "INCONCLUSIVE-R (RULE-MISCALIBRATED)", gating=False)

    # G-BOOT, G-READOUT
    for stat in STAT_FAMILY:
        if stat == "M" and fires:
            out["G-BOOT:M"] = _gate(None, True, None, note="theta_M not formed (G-MDENOM); Delta_M is a t-interval")
        else:
            b = dc["boot"][stat]
            share = _int(b["n_degenerate"], "n_degenerate", 0) / _int(b["n_boot"], "n_boot", 1)
            out[f"G-BOOT:{stat}"] = _gate(share, share <= BOOT_MAX_SHARE, STAT_DNC[stat])
        rv = jm["readout_validity"][stat]
        miss = [k for k in ("fit", "positive_control", "identity") if rv.get(k) is None]
        out[f"G-READOUT:{stat}"] = _gate({"missing": miss}, not miss, STAT_DNC[stat])

    # G-NOLEAK, G-CROSSJOB
    leaks = list(dc["noleak"]["leaks"])
    out["G-NOLEAK"] = _gate(leaks, not leaks, "SEAL-BROKEN", gating=False)
    if "c_D_band_R4" in dc:
        x = abs(math.log(_num(dc["c_D_band_R4"], "c_D_band_R4") / band))
        out["G-CROSSJOB"] = _gate(x, x <= CROSSJOB_MAX, "R4 draws do not count", gating=False)
    return out


# ---------------------------------------------------------------------------
# §6.4 — the verdict
# ---------------------------------------------------------------------------
E_GATES = ("G-DRAWS:E", "G-BRACKET:E", "G-BOOT:E", "G-READOUT:E")
M_GATES = ("G-MEANPOS", "G-DRAWS:M", "G-BRACKET:M", "G-BOOT:M", "G-READOUT:M")
R_GATES = ("G-REACH:rt", "G-PIVOT:rt", "G-EXHAUST:rt", "G-DRAWS:R", "G-DOSE", "G-BRACKET:R", "G-BOOT:R",
           "G-READOUT:R")
T_GATES = ("G-REACH:emt", "G-PIVOT:emt", "G-EXHAUST:emt", "G-DRAWS:T", "G-BRACKET:T", "G-BOOT:T",
           "G-READOUT:T")


def _failing(gates_, ids):
    return [g for g in ids if not gates_[g]["pass"] and gates_[g]["gating"]]


def _interval(stats, key):
    d = stats[key]
    return _num(d["lo"], f"{key}.lo"), _num(d["hi"], f"{key}.hi")


def verdict(stats, gates, thresholds, params=PG1):
    """§6.4, as a script runs it, on F after the gates; fixed sequence E → M → R → T. params: the run's family
    parameter set, whose run_gates are the run gates read first (session 2: G14-PRE in place of G-PRE1–4).

    stats: E, R, T (and M when θ_M is formed) as {theta, lo, hi} from decision_interval; Delta_M as
    {mean, lo, hi} from difference_interval when G-MDENOM fires; x_band = ln c_F(band); x_rt = ln c_F(rt_d)
    over the valid RT draws. gates: from gates(). thresholds: from thresholds_from_controls().
    Returns {label, notes, numbers}. Every does-not-count label names its failing gates. MASK-UNKNOWN is
    added after the run gates when G-MASK says so; SEAL-BROKEN is added to every return when G-NOLEAK
    recorded a leak. numbers carries every statistic passed in, the prediction interval, δ, ε_M, the power
    figures read and the step at which the tree stopped (None when it ran to the end)."""
    delta = _num(thresholds["delta"], "delta")
    notes = []
    seal = not gates["G-NOLEAK"]["pass"]
    pi = prediction_interval(stats["x_rt"], thresholds["alpha"]) if len(stats["x_rt"]) >= 2 else None
    x_b = _num(stats["x_band"], "x_band")
    numbers = {"delta": delta, "eps_M": thresholds["eps_M"], "PI": None if pi is None else list(pi),
               "x_band": x_b, "stats": stats,
               "power": {k: thresholds[k] for k in ("power_E", "power_M", "p_control_rule_at_1",
                                                    "p_beyond_at_1.30", "h2_reachable_at_1.30", "power_T")}}

    def done(label, step):
        return {"label": label, "notes": notes + (["SEAL-BROKEN"] if seal else []),
                "numbers": {**numbers, "stopped_at": step}}

    fail = _failing(gates, params["run_gates"])
    if fail:
        return done(f"RUN-DOES-NOT-COUNT ({', '.join(fail)})", "RUN")
    if gates["G-MASK"]["mask_unknown"]:
        notes.append("MASK-UNKNOWN")

    # step E
    fail = _failing(gates, E_GATES)
    if fail:
        return done(f"E-DOES-NOT-COUNT ({', '.join(fail)})", "E")
    lo, hi = _interval(stats, "E")
    if hi < delta:
        return done("H0-WINDOWS", "E")
    if not lo > delta:
        return done("UNDERPOWERED-E" if thresholds["power_E"] < POWER_MIN else "INCONCLUSIVE-E", "E")

    # step M: H0-MEAN stops; every other outcome is a note
    fail = _failing(gates, M_GATES)
    if fail:
        notes.append(f"M-DOES-NOT-COUNT ({', '.join(fail)})")
    elif gates["G-MDENOM"]["fires"]:
        lo, hi = _interval(stats, "Delta_M")
        eps = _num(thresholds["eps_M"], "eps_M")
        if hi <= eps:
            return done("H0-MEAN", "M")
        if not lo > eps:
            notes.append("INCONCLUSIVE-M")
    else:
        lo, hi = _interval(stats, "M")
        if hi < delta:
            return done("H0-MEAN", "M")
        if not lo > delta:
            notes.append("UNDERPOWERED-M" if thresholds["power_M"] < POWER_MIN else "INCONCLUSIVE-M")

    # step R
    fail = _failing(gates, R_GATES)
    if fail:
        return done(f"RT-DOES-NOT-COUNT ({', '.join(fail)})", "R")
    if gates["G-DOSE"]["n_steps"] == 0:
        notes.append("DOSE-DOES-NOT-COUNT")
    lo, hi = _interval(stats, "R")
    if 1 / delta <= lo and hi <= delta:
        label = "CONTROL-RULE"
        if thresholds["p_control_rule_at_1"] < POWER_MIN:
            notes.append("LOW-POWER")
    elif lo > delta or hi < 1 / delta:
        if not gates["G-PSEUDO"]["pass"]:
            return done("INCONCLUSIVE-R (RULE-MISCALIBRATED)", "R")
        side = "CHEAPER" if lo > delta else "COSTLIER"
        if (side == "CHEAPER" and x_b < pi[0]) or (side == "COSTLIER" and x_b > pi[1]):
            label = f"BEYOND-{side}"
            if thresholds["p_beyond_at_1.30"] < POWER_MIN:
                notes.append("LOW-POWER")
        elif not thresholds["h2_reachable_at_1.30"] or thresholds["p_beyond_at_1.30"] < POWER_MIN:
            return done("UNDERPOWERED-R (H2-UNREACHABLE)", "R")
        else:
            return done("BEYOND-WITHIN-SPREAD", "R")
    else:
        if thresholds["p_control_rule_at_1"] < POWER_MIN:
            return done("UNDERPOWERED-R (H1-POWER)", "R")
        return done("INCONCLUSIVE-R", "R")

    # step T: only after CONTROL-RULE; always a note
    if label == "CONTROL-RULE":
        fail = _failing(gates, T_GATES)
        if fail:
            notes.append(f"T-DOES-NOT-COUNT ({', '.join(fail)})")
        else:
            lo, hi = _interval(stats, "T")
            if lo > delta:
                notes.append("T-QUANTITY")
            elif hi < 1 / delta:
                notes.append("T-CHEAPER")
            elif 1 / delta <= lo and hi <= delta:
                notes.append("T-POSITIONS-OR-SCHEME")
            else:
                notes.append("T-UNDERPOWERED" if thresholds["power_T"] < POWER_MIN else "T-INCONCLUSIVE")
    return done(label, None)


# ---------------------------------------------------------------------------
# §10.1 — samples
# ---------------------------------------------------------------------------
def sample_record(window, logp_none, logp_arm, top_none, top_arm):
    """§10.1: one window of an arm's sample file.

    window: {index, text_seen (the processor input decoded, the 256 <image> tokens shown as a count), n_text,
    target_ids, target_strs} with ntok = n_text − 1 targets. logp_none / logp_arm: log p(target) at each
    scored position under the before-model and the after-model. top_none / top_arm: (top-1 ids, top-1
    strings, top-1 probabilities) at each scored position. Returns the window's NLL sums, Y, every scored
    position (target, both log-probabilities, both top-1 tokens with probabilities, whether each top-1 is the
    target), and right_example / wrong_example: the first position whose after-model top-1 is / is not the
    target (None when the window has no such position)."""
    n_text = _int(window["n_text"], "n_text", 2)
    ntok = n_text - 1
    ids = [int(x) for x in window["target_ids"]]
    strs = list(window["target_strs"])
    lp_n = _arr(logp_none, 1, "logp_none")
    lp_a = _arr(logp_arm, 1, "logp_arm")
    if not len(ids) == len(strs) == lp_n.size == lp_a.size == ntok:
        raise ValueError(f"ntok {ntok}: ids {len(ids)}, strs {len(strs)}, logp {lp_n.size}/{lp_a.size}")
    if np.any(lp_n > 0) or np.any(lp_a > 0):
        raise ValueError("log-probabilities must be <= 0")
    tops = {}
    for name, (t_ids, t_strs, t_p) in (("none", top_none), ("arm", top_arm)):
        p = _arr(t_p, 1, f"top_{name} probabilities")
        if not len(t_ids) == len(t_strs) == p.size == ntok or np.any(p < 0) or np.any(p > 1):
            raise ValueError(f"top_{name}: lengths {len(t_ids)}, {len(t_strs)}, {p.size} or p outside [0, 1]")
        tops[name] = ([int(x) for x in t_ids], list(t_strs), p)
    positions = []
    for i in range(ntok):
        row = {"pos": i, "target_id": ids[i], "target_str": strs[i],
               "logp_none": float(lp_n[i]), "logp_arm": float(lp_a[i])}
        for name in ("none", "arm"):
            t_ids, t_strs, p = tops[name]
            row[f"top1_{name}"] = t_strs[i]
            row[f"p_top1_{name}"] = float(p[i])
            row[f"top1_{name}_is_target"] = t_ids[i] == ids[i]
        positions.append(row)
    nll_none, nll_arm = float(-lp_n.sum()), float(-lp_a.sum())
    right = [r for r in positions if r["top1_arm_is_target"]]
    wrong = [r for r in positions if not r["top1_arm_is_target"]]
    return {"index": _int(window["index"], "index", 0), "text_seen": str(window["text_seen"]),
            "n_text": n_text, "ntok": ntok, "nll_sum_none": nll_none, "nll_sum_arm": nll_arm,
            "Y": (nll_arm - nll_none) / ntok,
            "n_right": {"none": sum(r["top1_none_is_target"] for r in positions), "arm": len(right)},
            "right_example": right[0] if right else None, "wrong_example": wrong[0] if wrong else None,
            "positions": positions}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def provenance():
    """Run provenance stamped by scripts/realizednull_stamp.py into infra/pg/realizednull_provenance.json.
    Refuses a stamp whose program is not "realizednull": another program's committed stamp must never be
    read as this program's code identity."""
    d = json.loads(PROVENANCE_FILE.read_text())
    if d.get("program") != "realizednull":
        raise ValueError(f"{PROVENANCE_FILE} is stamped for program {d.get('program')!r}, not 'realizednull'")
    return d
