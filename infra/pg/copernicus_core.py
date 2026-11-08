"""COPERNICUS core — program identity, frozen thresholds, the ΔNLL statistics.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-17_copernicus_program_spec.md

This is a RECORDING + GATING library (§0.1). It records, it applies the frozen §6 decision
rules, and it refuses malformed input. It never interprets.

Layering. `humboldt_core` (H) is the program-neutral measurement substrate and is imported,
never forked. MERCATOR's `mercator_core` is NOT imported: its Atlas hard-binds MERCATOR's
register/wave vocabulary and its provenance() asserts `program == "mercator"`, so riding it
would stamp COPERNICUS rows with another program's identity — the defect DARWIN logged as D1
and MERCATOR re-logged in its own provenance note. COPERNICUS gets its own stamp here.

WHY ITS OWN SCALE. Every A1 contrast in this program is additive on ΔNLL = log ppl_ablated −
log ppl_clean, never a ratio of Δppl (§6.1). ΔNLL differences telescope — ΔNLL(a) − ΔNLL(b) =
log ppl_a − log ppl_b — so one paired-bootstrap routine serves both "damage vs clean" and
"bank vs bank", and the clean arm cancels out of every contrast instead of carrying its noise
into it.

FAIL LOUD (CLAUDE.md): no try/except, no synthetic fallback, no default that hides a missing
input. Heavy imports live inside functions so this py_compiles and unit-tests with numpy alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

import numpy as np

from infra.pg import humboldt_core as H

# ---------------------------------------------------------------------------
# Frozen program constants (§0). Changing any post-freeze is an ADDENDUM.
# ---------------------------------------------------------------------------
PROGRAM = "copernicus"
CLAIMED_BY = "copernicus-executor"
BUCKET = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
COPERNICUS_S3 = f"s3://{BUCKET}/{PROGRAM}"

REGISTER = "ATLAS"                                   # §0: all rows tagged COPERNICUS, register ATLAS
WAVES = ("WF", "W0", "W1", "W2", "RES")

# §0 verdict vocabulary — house standard. These are RECORDED OUTCOMES; none is a failure.
REFUTER_FIRED = "REFUTER-FIRED"
REFUTER_DID_NOT_FIRE = "REFUTER-DID-NOT-FIRE"
VOID_INSTRUMENT = "VOID-INSTRUMENT"
COVERAGE_IMPOSSIBLE = "COVERAGE-IMPOSSIBLE"
AMBIGUOUS = "AMBIGUOUS"
OUTCOMES = (REFUTER_FIRED, REFUTER_DID_NOT_FIRE, VOID_INSTRUMENT, COVERAGE_IMPOSSIBLE, AMBIGUOUS)

# §6.1.5 CO-A1 branch literals.
A1_ARTIFACT = "ARTIFACT"
A1_SINK_SPECIFIC = "SINK-SPECIFIC"
A1_TRAFFIC = "TRAFFIC"
A1_EXTREMES = "EXTREMES"
A1_COMPOUND_DEFLATIONARY = "compound deflationary"
A1_AMBIGUOUS_MULTI = "AMBIGUOUS-MULTI"

# §6.2.4 CO-A2 branch literals.
A2_VOID_FOR_CONFOUND = "VOID-FOR-CONFOUND"
A2_COVERAGE_IMPOSSIBLE = COVERAGE_IMPOSSIBLE
A2_SPLIT = "SPLIT"
A2_MECHANISM_CLEAN = "MECHANISM-CLEAN"
A2_DEGRADATION_DRIVEN = "DEGRADATION-DRIVEN"
A2_AMBIGUOUS = AMBIGUOUS

# §6.1.3 — the two margin FRACTIONS are frozen HERE; the SCALES they multiply come from the
# discovery half at run time and land in thresholds.json.
M_SUP_FRAC = 0.10
M_EQ_FRAC = 0.25
# §2.1 bank-overlap ceiling: above this mean cos² the equal-k contrast is NON-ADJUDICATING.
BANK_OVERLAP_CSTAR = 0.25
# §6.1.4 saturation REPORTING flag (never an exclusion).
CAP_SAT_MULT = 10.0
# §6.1.6 instrument dynamic-range gate: topsigma must clear the random band by this many w_rand.
TOPSIGMA_DYNAMIC_RANGE = 10.0
# §6.1.6 sane-ppl bound: clean ppl must stay under this multiple of the anchor's clean value.
SANE_PPL_MULT = 2.0

# ---------------------------------------------------------------------------
# Provenance. COPERNICUS needs its OWN stamp for the reason mercator_core states: several
# `*_provenance.json` files are committed in the tree, so an inherited provenance() would put
# another program's frozen sha on every COPERNICUS row.
# ---------------------------------------------------------------------------
PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "copernicus_provenance.json"


def provenance() -> dict:
    """Run provenance for every atlas row. `pg run` ships only `infra/`, so there is no .git on
    the instance; scripts/copernicus_launch.sh stamps this file in the same step as the launch,
    so it cannot go stale (DARWIN defect D41)."""
    if PROVENANCE_FILE.exists():
        d = json.loads(PROVENANCE_FILE.read_text())
        assert d.get("program") == PROGRAM, (
            f"{PROVENANCE_FILE.name} carries program={d.get('program')!r}, not {PROGRAM!r} — a "
            f"foreign stamp would put another program's sha on COPERNICUS rows")
        return d
    root = pathlib.Path(__file__).resolve().parents[2]
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       check=True, capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                       check=True, capture_output=True, text=True)
    return {"program": PROGRAM, "commit_sha": r.stdout.strip(), "dirty": bool(d.stdout.strip()),
            "source": "git (control plane)"}


def commit_sha() -> str:
    return provenance()["commit_sha"]


class Atlas(H.Atlas):
    """COPERNICUS atlas writer.

    Constructor-bound: `wave`, `queue_item`, `expected_ref` — an arm runs in exactly one wave,
    maps to exactly one queue item (§13's coverage key), and has exactly one pre-logged
    expectation (§9). Making any per-call would make a mixed-arm file representable.

    The three ride in `flags` rather than becoming new columns: H's schema assert is exact, and
    a forked column set would make COPERNICUS parquets unconcatenable with the rest of the
    lineage for no gain.
    """

    def __init__(self, out_dir, arm: str, run_id: str, *, wave: str, queue_item: str,
                 expected_ref: str, tag: str = H.DISCOVERY):
        assert wave in WAVES, f"unknown wave {wave!r} (expected one of {WAVES})"
        assert queue_item and str(queue_item).strip(), (
            f"{arm}: queue_item is the §13 coverage-map key and may not be empty")
        assert expected_ref and expected_ref.startswith(f"{arm}::"), (
            f"expected_ref {expected_ref!r} does not belong to arm {arm} — §9 requires the arm's "
            f"own pre-logged expectation id")
        super().__init__(out_dir, arm=arm, run_id=run_id, tag=tag)
        self.sha = commit_sha()                      # re-stamp: H set it from HUMBOLDT's file
        self.wave = wave
        self.queue_item = str(queue_item)
        self.expected_ref = expected_ref

    def add(self, statistic: str, *, flags=None, **kw):
        rid = {"program": PROGRAM, "wave": self.wave, "queue_item": self.queue_item,
               "expected_ref": self.expected_ref}
        return super().add(statistic, flags={**rid, **(flags or {})}, **kw)


Gates = H.Gates


# ---------------------------------------------------------------------------
# §9.1b — mechanical threshold enforcement. Every frozen band/margin is written once, hashed,
# and every downstream scoring call asserts the hash. Promissory blinding is not blinding.
# ---------------------------------------------------------------------------
def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def thresholds_hash(payload: dict) -> str:
    return hashlib.sha256(_canon(payload)).hexdigest()


def write_thresholds(path, payload: dict) -> str:
    """Freeze the thresholds and return their content hash. Refuses to overwrite: a second write
    would silently move a band that a prior cell was already scored against."""
    p = pathlib.Path(path)
    assert not p.exists(), (
        f"{p} already exists — thresholds are frozen once, before the confirmation half (§9.1b); "
        f"rewriting them post hoc is the rule swap §9.7 forbids")
    p.parent.mkdir(parents=True, exist_ok=True)
    h = thresholds_hash(payload)
    p.write_text(json.dumps({"sha256": h, "thresholds": payload}, indent=2, sort_keys=True))
    return h


def load_thresholds(path, expect_hash: str) -> dict:
    """Read frozen thresholds, asserting the caller's expected hash (M-D99 pattern: the consumer
    states what it expects, the artifact proves it)."""
    p = pathlib.Path(path)
    assert p.exists(), f"no thresholds at {p} — the C-half may not be scored before §9.1b runs"
    d = json.loads(p.read_text())
    got = thresholds_hash(d["thresholds"])
    assert got == d["sha256"], f"{p}: self-hash mismatch (file edited by hand?) {got} != {d['sha256']}"
    assert got == expect_hash, (
        f"{p}: threshold hash {got} != expected {expect_hash} — the bands moved between freezing "
        f"and scoring")
    return d["thresholds"]


class Expectations:
    """`copernicus_expectations.jsonl` — one row per gate, written BEFORE that gate's contrast
    cell is computed (§0, the SHACKLETON pattern). Append-only; a second row for the same gate
    id fails loud rather than overwriting a prediction that has already been spent."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] \
            if self.path.exists() else []

    def log(self, arm: str, gate: str, prediction: str, *, confidence: str, basis: str,
            family: str) -> str:
        gid = f"{arm}::{gate}"
        assert not any(r["id"] == gid for r in self.rows), (
            f"{gid} already has a logged expectation — a prediction may be logged once, before "
            f"the cell is computed")
        row = {"id": gid, "arm": arm, "gate": gate, "prediction": prediction,
               "confidence": confidence, "basis": basis, "family": family,
               "utc": utc_now(), "commit_sha": commit_sha()}
        with open(self.path, "a", buffering=1) as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows.append(row)
        return gid

    def require(self, arm: str, gate: str = None) -> str:
        """The id of the arm's pre-logged expectation. An arm that cannot show one must not run."""
        want = f"{arm}::{gate}" if gate else None
        hit = [r for r in self.rows if (r["id"] == want if want else r["arm"] == arm)]
        assert hit, f"no pre-logged expectation for {want or arm} — §9 forbids running blind-less"
        return hit[0]["id"]


def utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# §6.1 — the ΔNLL scale and its paired bootstrap.
#
# A per-window row is [nll_sum, n_tok]; log ppl = Σnll / Σtok over the resampled windows. Since
# every arm scores the SAME windows in the SAME order, the token column must match exactly and
# the resample index is SHARED — that is what makes the contrast paired.
# ---------------------------------------------------------------------------
def log_ppl(pw, idx=None) -> float:
    a = np.asarray(pw, dtype=np.float64)
    s = a if idx is None else a[idx]
    return float(s[:, 0].sum() / s[:, 1].sum())


def ppl(pw, idx=None) -> float:
    return float(np.exp(log_ppl(pw, idx)))


def paired_log_ratio_ci(pw_num, pw_den, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Paired bootstrap of log ppl(num) − log ppl(den) over a SHARED window resample.

    Serves BOTH A1 quantities, because ΔNLL differences telescope:
      ΔNLL(arm)            = paired_log_ratio_ci(arm,  clean)
      ΔNLL(a) − ΔNLL(b)    = paired_log_ratio_ci(a,    b)      (clean cancels exactly)

    Returns the point estimate, the 95% interval, its one-sided bounds, and the bootstrap tail
    probability that the quantity is ≤ 0 (the p-value BH consumes for the gates that produce one).
    """
    t = np.asarray(pw_num, dtype=np.float64)
    u = np.asarray(pw_den, dtype=np.float64)
    assert t.shape == u.shape, (t.shape, u.shape)
    assert t.ndim == 2 and t.shape[1] == 2, f"per-window rows must be [nll_sum, n_tok], got {t.shape}"
    assert np.array_equal(t[:, 1], u[:, 1]), (
        "paired arms have mismatched per-window token counts — they did not score the same "
        "windows in the same order, so the pairing is fiction")
    W = t.shape[0]
    assert W >= 2, f"paired bootstrap needs >=2 windows, got {W}"
    point = log_ppl(t) - log_ppl(u)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, W, W)
        boots[b] = log_ppl(t, idx) - log_ppl(u, idx)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "lo_one_sided": float(np.percentile(boots, 5.0)),
            "hi_one_sided": float(np.percentile(boots, 95.0)),
            "p_le0": float(np.mean(boots <= 0.0)), "n_windows": int(W),
            "half_width": float((hi - lo) / 2.0), "n_boot": int(n_boot), "seed": int(seed),
            "construction": "paired bootstrap of log ppl(num)-log ppl(den) on a shared window resample"}


def band(values, lo_q: float = 2.5, hi_q: float = 97.5) -> dict:
    """A null band from its own draws: the quantile interval AND its width w (§6.1.2). The width
    is the floor under every superiority margin, so it ships on the row with the band."""
    v = np.asarray(values, dtype=np.float64)
    assert v.size >= 3, f"a band needs >=3 draws, got {v.size} (a single draw is noise)"
    lo, hi = float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))
    return {"lo": lo, "hi": hi, "width": hi - lo, "n_draws": int(v.size),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "draws": [float(x) for x in v],
            "construction": f"{lo_q}-{hi_q} quantile interval over {v.size} draws"}


def margins(w_rand: float, dnll_core_D: float) -> dict:
    """§6.1.3 — two scales, because the null floor alone cannot express the claims (the anchor's
    w_rand sits ~230× below the effect scale at k=8, which would make EXTREMES unfireable)."""
    assert dnll_core_D > 0, f"core ΔNLL on D must be positive to set a margin scale, got {dnll_core_D}"
    assert w_rand >= 0, w_rand
    return {"m_sup": float(max(w_rand, M_SUP_FRAC * dnll_core_D)),
            "m_eq": float(max(2.0 * w_rand, M_EQ_FRAC * dnll_core_D)),
            "w_rand": float(w_rand), "dnll_core_D": float(dnll_core_D),
            "m_sup_frac": M_SUP_FRAC, "m_eq_frac": M_EQ_FRAC,
            "construction": "m_sup=max(w_rand, 0.10*dNLL_D(core)); m_eq=max(2*w_rand, 0.25*dNLL_D(core))"}


def ci_separated(ci: dict, margin: float) -> bool:
    """§6.1.3 — 'CI-separated' means the paired contrast's one-sided 95% LOWER bound clears the
    margin. Stated once here so no gate can quietly use a different definition."""
    return bool(ci["lo_one_sided"] > margin)


def tost_within(ci: dict, m_eq: float) -> bool:
    """§6.1.5 branch 4 — equivalence: the two-sided interval lies wholly inside ±m_eq."""
    return bool(ci["lo"] > -m_eq and ci["hi"] < m_eq)


# ---------------------------------------------------------------------------
# v0.8 amendment §A1 — measurement validity replaces the ppl-scale saturation cap, and a twin
# BATTERY replaces the single permutation twin.
# ---------------------------------------------------------------------------
MEAS_PRECISION_NATS = 0.10          # §A1.1 (i) — frozen readout precision
MEAS_SANITY_NATS = float(np.log(1000.0))   # §A1.1 (ii) ≈ 6.908 — frozen readout-sanity bound
TWIN_ALPHA = 0.05                   # §A1.2 — two-sided rank-sum level


def measurement_valid(dnll: float, ci_half_width: float) -> dict:
    """§A1.1 — a cell is MEASUREMENT-VALID iff it is precisely measured and inside the readout's
    trusted range.

    This replaces v0.7's `cap_sat = 10× clean`, which voided CO-A1 because `anti` and `topenergy`
    exceed a ppl-scale cap BY CONSTRUCTION. A devastated model is information, not an invalid
    readout — but only while the estimator is still trustworthy and the interval is still tight,
    which is exactly what these two clauses check.
    """
    prec_ok = bool(ci_half_width <= MEAS_PRECISION_NATS)
    sane_ok = bool(dnll < MEAS_SANITY_NATS)
    return {"valid": bool(prec_ok and sane_ok), "precision_ok": prec_ok, "sanity_ok": sane_ok,
            "ci_half_width": float(ci_half_width), "dnll": float(dnll),
            "precision_bound": MEAS_PRECISION_NATS, "sanity_bound": MEAS_SANITY_NATS,
            "construction": ("§A1.1: half-width <= 0.10 nats AND dNLL < ln(1000); the old cap_sat "
                             "flag is context only and gates nothing")}


def twin_battery_gate(twin_draws, ref_draws, alpha: float = TWIN_ALPHA) -> dict:
    """§A1.2 — DISTRIBUTION against DISTRIBUTION, two-sided rank-sum.

    v0.7 compared ONE twin draw against a 5-draw band, which on a heavy-tailed statistic fails on a
    draw: a random 128-coordinate set that happens to catch a massive-activation coordinate jumps.
    A single hot draw must not be able to void the instrument, while a genuinely displaced twin
    POPULATION still must. The gate FIRES (i.e. the twin is displaced, which is bad) when the two
    populations separate at alpha.
    """
    from scipy import stats
    t = np.asarray(twin_draws, dtype=np.float64)
    r = np.asarray(ref_draws, dtype=np.float64)
    assert t.size >= 3 and r.size >= 3, (
        f"a distribution-vs-distribution gate needs >=3 draws a side, got {t.size} vs {r.size}")
    res = stats.mannwhitneyu(t, r, alternative="two-sided")
    displaced = bool(res.pvalue < alpha)
    return {"twin_draws": [float(x) for x in t], "ref_draws": [float(x) for x in r],
            "u_statistic": float(res.statistic), "p_two_sided": float(res.pvalue),
            "alpha": alpha, "twin_displaced": displaced, "passes": not displaced,
            "twin_median": float(np.median(t)), "ref_median": float(np.median(r)),
            "construction": (f"two-sided Mann-Whitney rank-sum, {t.size} twin draws vs {r.size} "
                             f"reference draws, alpha={alpha}; passes when the populations do NOT "
                             f"separate")}


def a1_verdict(c: dict) -> dict:
    """§6.1.5 — evaluate ALL branches, then apply the frozen precedence table. Pure function of
    measured quantities so every branch can be exercised against planted inputs before the run.

    Required keys: instrument_ok, unflagged_k, dec_k, dnll (bank→k→value), bands (k→band),
    m_sup, m_eq, non_adjudicating, anti_non_adjudicating, sd_dnll_core, sd_band, sd_m_sup,
    te_sd_outside, and the CI dicts ci_core_rand / ci_core_te / ci_core_anti / ci_anti_core /
    ci_eqE / ci_sd_rand / ci_sd_vs.
    """
    if not c["instrument_ok"]:
        return {"verdict": VOID_INSTRUMENT, "fired": [],
                "reason": "an §6.1.6 instrument gate did not pass"}
    if c["dec_k"] is None:
        return {"verdict": VOID_INSTRUMENT, "fired": [],
                "reason": "every k saturation-flagged on an adjudicated bank"}
    k, m_sup, m_eq = c["dec_k"], c["m_sup"], c["m_eq"]

    # 1 ARTIFACT — every adjudicated bank inside the matched-random band at every unflagged k.
    artifact = all(c["bands"][kk]["lo"] <= c["dnll"][b][kk] <= c["bands"][kk]["hi"]
                   for b in ("core", "anti", "topenergy") for kk in c["unflagged_k"])

    # 2 SINK-SPECIFIC. The NON-ADJUDICATING fallback replaces ONLY the topenergy leg; the anti leg
    # stands unless the (core, anti) pair itself trips c*.
    if c["non_adjudicating"]:
        te_leg = (ci_separated(c["ci_sd_rand"], c["sd_m_sup"])
                  and ci_separated(c["ci_sd_vs"], 0.0))
    else:
        te_leg = ci_separated(c["ci_core_te"], m_sup) and c["ci_eqE"]["lo_one_sided"] > 0.0
    anti_leg = True if c["anti_non_adjudicating"] else ci_separated(c["ci_core_anti"], m_eq)
    sink_specific = bool(te_leg and anti_leg
                         and ci_separated(c["ci_core_rand"], m_sup)
                         and ci_separated(c["ci_sd_rand"], c["sd_m_sup"]))

    # 3 TRAFFIC — symmetric, and an UNDERPOWERED cell may not land here: the proportional margins
    # loosen exactly where damage saturates, so the half-width is capped absolutely.
    sd_inside = bool(c["sd_band"]["lo"] <= c["sd_dnll_core"] <= c["sd_band"]["hi"])
    if c["non_adjudicating"]:
        traffic = bool(sd_inside and c["te_sd_outside"])
    else:
        traffic = bool(c["ci_eqE"]["hi"] < m_sup and c["ci_eqE"]["half_width"] <= m_sup
                       and sd_inside)

    # 4 EXTREMES — TOST inside ±m_eq with both banks out of the random band.
    extremes = bool(tost_within(c["ci_anti_core"], m_eq)
                    and c["dnll"]["core"][k] > c["bands"][k]["hi"]
                    and c["dnll"]["anti"][k] > c["bands"][k]["hi"])

    fired = [n for n, f in ((A1_ARTIFACT, artifact), (A1_SINK_SPECIFIC, sink_specific),
                            (A1_TRAFFIC, traffic), (A1_EXTREMES, extremes)) if f]
    # precedence table, top row first
    if artifact:
        verdict = A1_ARTIFACT                       # outranks TRAFFIC, whose clauses need no band exit
    elif sink_specific and set(fired) <= {A1_SINK_SPECIFIC, A1_EXTREMES}:
        verdict = A1_SINK_SPECIFIC
    elif traffic and extremes:
        verdict = A1_COMPOUND_DEFLATIONARY
    elif len(fired) == 1:
        verdict = fired[0]
    elif len(fired) > 1:
        verdict = A1_AMBIGUOUS_MULTI
    else:
        verdict = AMBIGUOUS
    return {"verdict": verdict, "fired": fired,
            "legs": {"te_leg": bool(te_leg), "anti_leg": bool(anti_leg),
                     "sd_inside": sd_inside, "artifact": bool(artifact)}}


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values (§7, q=0.10 within a hypothesis-tree family),
    returned in input order."""
    p = np.asarray(pvals, dtype=np.float64)
    assert p.size > 0, "BH over an empty family"
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out


# ---------------------------------------------------------------------------
# §2.2 / §6.2 — CO-A2: the release signature, the ladder regression, the seed permutation.
# ---------------------------------------------------------------------------
def release_signature(delivery: dict, core_bank: str = "core32") -> float:
    """S = log delivery(core) − median over the OTHER banks of log delivery.

    The bedrock finding is "core rises while the others fall", so S — not raw core delivery — is
    what a degradation account has to explain. A ladder arm whose whole battery rises together is
    a global-motion artifact and lands in the VOID-FOR-CONFOUND branch instead.
    """
    assert core_bank in delivery, f"{core_bank} missing from {sorted(delivery)}"
    others = [v for b, v in delivery.items() if b != core_bank]
    assert len(others) >= 3, f"S needs a battery, got {len(others)} other banks"
    assert all(v > 0 for v in delivery.values()), f"non-positive delivery: {delivery}"
    return float(np.log(delivery[core_bank]) - np.median(np.log(others)))


def whole_battery_rose(delivery: dict, reference: dict, tol: float = 1.0) -> bool:
    """§6.2.4 branch 1 — every bank's delivery above the reference arm's means the arm moved
    globally, so its release signature is uninterpretable (that arm is inadmissible)."""
    return all(delivery[b] > tol * reference[b] for b in reference if b in delivery)


def linear_prediction_interval(x, y, x0: float, alpha: float = 0.05) -> dict:
    """OLS of y on x with the (1−alpha) PREDICTION interval at x0 — the §2.2 summary measure.

    A prediction interval, not a confidence interval: the question is whether the hadamard
    replicas' OWN release signature is a plausible draw from the ladder's relationship, which is
    about where a NEW point would fall, not about where the line is.
    """
    from scipy import stats
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.size >= 4, (
        f"a prediction interval needs >=4 ladder points, got {x.size}")
    n = x.size
    sxx = float(((x - x.mean()) ** 2).sum())
    assert sxx > 0, "ladder has no spread in ppl — the regression is undefined"
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    s = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    se = s * float(np.sqrt(1.0 + 1.0 / n + (x0 - x.mean()) ** 2 / sxx))
    t = float(stats.t.ppf(1.0 - alpha / 2.0, n - 2))
    fit = intercept + slope * x0
    return {"slope": slope, "intercept": intercept, "fit_at_x0": float(fit),
            "pi_lo": float(fit - t * se), "pi_hi": float(fit + t * se),
            "half_width": float(t * se), "resid_sd": s, "n": int(n), "x0": float(x0),
            "construction": f"OLS + {int((1 - alpha) * 100)}% prediction interval at x0"}


def exact_permutation_p(a, b) -> dict:
    """Exact two-group permutation test on the difference of means. At n=3 vs 3 the smallest
    attainable one-sided p is 1/C(6,3) = 0.05 — that floor IS the pre-registered test (§2.2), and
    a bootstrap over three seeds would be a fiction."""
    from itertools import combinations
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pool = np.concatenate([a, b])
    n = a.size
    obs = float(a.mean() - b.mean())
    ge = 0
    total = 0
    for idx in combinations(range(pool.size), n):
        g1 = pool[list(idx)]
        g2 = np.delete(pool, list(idx))
        total += 1
        if float(g1.mean() - g2.mean()) >= obs - 1e-12:
            ge += 1
    return {"observed_diff": obs, "p_one_sided": ge / total, "n_perm": total,
            "complete_separation": bool(ge == 1),
            "floor": 1.0 / total,
            "construction": f"exact permutation, {n} vs {b.size}, one-sided"}


def a2_verdict(c: dict) -> dict:
    """§6.2.4 — evaluate all branches, then the frozen precedence.

    Keys: anchor_ok, ladders {name: {admissible, brackets, pi_ok, above_net, above_path,
    inside_net, inside_path}}, void_for_confound (list of arm names), permutation
    {complete_separation}.

    COVERAGE-IMPOSSIBLE OUTRANKS the two substantive branches: if no ladder survives, the
    confirmatory positive must not be allowed to rest on the permutation test alone.
    """
    if not c["anchor_ok"]:
        return {"verdict": VOID_INSTRUMENT, "reason": "§6.2.1 anchor recompute did not reproduce"}
    surviving = {n: L for n, L in c["ladders"].items()
                 if L["admissible"] and L["brackets"] and L["pi_ok"]}
    if not surviving:
        return {"verdict": COVERAGE_IMPOSSIBLE, "surviving": [],
                "reason": ("no ladder bracketed hadamard's ppl with an adequate prediction "
                           "interval; the confound stays OPEN and named, and the bedrock cannot "
                           "enter the seal"),
                "void_for_confound": c.get("void_for_confound", [])}
    # SPLIT is checked BEFORE the clean/degradation branches.
    split = any(L["above_net"] != L["above_path"] for L in surviving.values())
    if split:
        return {"verdict": A2_SPLIT, "surviving": sorted(surviving),
                "reason": "net and path delivery disagree on the above/inside call"}
    if (all(L["above_net"] and L["above_path"] for L in surviving.values())
            and c["permutation"]["complete_separation"]):
        return {"verdict": A2_MECHANISM_CLEAN, "surviving": sorted(surviving)}
    if any(L["inside_net"] and L["inside_path"] for L in surviving.values()):
        return {"verdict": A2_DEGRADATION_DRIVEN, "surviving": sorted(surviving)}
    return {"verdict": A2_AMBIGUOUS, "surviving": sorted(surviving)}


# ---------------------------------------------------------------------------
# §2.1 — bank construction. Every derivation runs on the fp32 checkpoint path; the callers own
# that, these are pure array routines.
# ---------------------------------------------------------------------------
def nested_random_subspaces(n: int, k_max: int, seed: int):
    """ONE Haar-orthonormal (n × k_max) draw per seed; `[:, :k]` is the matched-random null at k.

    NESTED by construction, because the structured banks are: core-8 ⊂ core-32 ⊂ core-128 are the
    bottom eigenvectors of one Gram. Drawing an independent QR per k — which is what the anchor's
    `_random_subspace` does, since numpy fills row-major and changing k changes every entry —
    would compare a nested treatment against a non-nested null and read the difference as dose.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k_max)))
    return np.ascontiguousarray(Q[:, :k_max].astype(np.float64))


def topenergy_bank(gram: np.ndarray, k: int, gains: np.ndarray = None):
    """One-hot top-k coordinates by activation energy (the `mercator_e2` construction).

    `gram` is the activation Gram at the ablation site, accumulated on the READOUT corpus D-half
    ONLY — an activation-derived bank built on the full corpus would leak the confirmation half.
    `gains`, when given, is γ = 1 + norm.weight: the post-LN site is γ-contaminated (item 74), so
    the caller banks a γ-corrected twin beside the raw bank and no interpretation of CO-A1 rides
    on which one it is.
    """
    e = np.asarray(np.diag(gram), dtype=np.float64).copy()
    assert e.ndim == 1 and e.size >= k, (e.shape, k)
    if gains is not None:
        g = np.asarray(gains, dtype=np.float64)
        assert g.shape == e.shape, (g.shape, e.shape)
        e = e / np.maximum(g ** 2, 1e-30)      # undo the γ² inflation of the post-LN diagonal
    idx = np.argsort(-e)[:k]
    Q = np.zeros((e.size, k), dtype=np.float64)
    Q[idx, np.arange(k)] = 1.0
    return Q, np.sort(idx)


def set_difference_bank(A: np.ndarray, B: np.ndarray, tol: float = 1e-8):
    """A ⊖ B: project B out of A, re-orthonormalise, drop directions that B absorbed (§2.1).

    The realized dimension is RETURNED, not assumed: the whole point of the cell is that the two
    banks overlap, so its width is data. Callers truncate the pair to a common dim and take the
    matched-DIMENSION null band at that width (§6.1.2).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    assert A.shape[0] == B.shape[0], (A.shape, B.shape)
    R = A - B @ (B.T @ A)
    Q, S, _ = np.linalg.svd(R, full_matrices=False)
    keep = int((S > tol * max(1.0, float(S[0]))).sum())
    assert keep > 0, "the set difference is empty — the comparator absorbed the bank entirely"
    return np.ascontiguousarray(Q[:, :keep]), keep, [float(x) for x in S]


def mean_cos2(U: np.ndarray, V: np.ndarray) -> float:
    """Mean cos² of the principal angles between two subspaces = ‖UᵀV‖²_F / min(dim). The
    bank-overlap statistic the §2.1 c* ceiling is read against."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    assert U.shape[0] == V.shape[0], (U.shape, V.shape)
    return float(np.sum((U.T @ V) ** 2) / min(U.shape[1], V.shape[1]))


def coordinate_permutation_twin(Q: np.ndarray, seed: int):
    """The identity-preserving twin: permute the COORDINATE rows of the bank. Width, leverage and
    spectrum are preserved exactly; only which coordinates the bank names changes. If the twin
    does what the bank does, the effect was width, not address."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Q.shape[0])
    return np.ascontiguousarray(np.asarray(Q, dtype=np.float64)[perm, :]), perm


def orthonormal_or_die(Q: np.ndarray, what: str, tol: float = 1e-8) -> float:
    resid = float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())
    assert resid < tol, f"{what}: not orthonormal, max|QᵀQ − I| = {resid:.3e}"
    return resid


def bank_hash(Q: np.ndarray) -> str:
    """Content hash of a bank, banked BEFORE any ablation so a post-hoc regeneration cannot pass
    as the thing that was scored."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(Q, dtype=np.float64)).tobytes()).hexdigest()
