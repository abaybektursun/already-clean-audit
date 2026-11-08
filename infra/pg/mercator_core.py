"""MERCATOR core — the two-register atlas, the coverage contract, the dual-basis guard.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-14_mercator_program_spec.md

This is a RECORDING + GATING library (§0.1). It records, it computes the frozen §6 decision
formula, and it refuses malformed rows. It never interprets: no function here returns a sentence,
and the only verdicts it emits are the §6 literals plus the named non-verdict outcomes.

Layering. `humboldt_core` (H) is the measurement substrate; `darwin_core` (D) is the parent
program's recording library. Both are imported, never forked. MERCATOR adds exactly what the spec
asks for on top of DARWIN's schema (§3):

    ... DARWIN columns ... + queue_item        (the §5 coverage-map key)

and two standing properties DARWIN did not have:

  * the DUAL-BASIS rule (§3, new): any statistic whose claim is about COORDINATES is reported in
    both the coordinate basis and one random reference basis. Enforced by `check_dual_basis`.
  * the READOUT-VALIDITY gate (§0.2.9): a fine-tune must beat its base on its own task, else the
    cell is VOID-readout. DARWIN applied this by hand per arm; here it is a function.

FAIL LOUD (CLAUDE.md): no try/except, no synthetic fallback, no default that hides a missing
input. Heavy imports live inside functions so this py_compiles and unit-tests on the control plane
with numpy alone.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H

# ---------------------------------------------------------------------------
# Frozen program constants (§0.2, §3). Changing any post-freeze is an ADDENDUM.
# ---------------------------------------------------------------------------
PROGRAM = "mercator"
CLAIMED_BY = "mercator-executor"                     # §0.2.4 — one arm, one owner
BUCKET = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
MERCATOR_S3 = f"s3://{BUCKET}/{PROGRAM}"
BANKS_S3 = f"{MERCATOR_S3}/banks"                    # output-side + foreign + Muon-base banks

# §0 preamble: TWO registers. STAGE2 does not exist in MERCATOR (§6: "No G0-style program gate").
REGISTERS = ("ATLAS", "GRAD")
WAVES = ("K", "W0", "W1", "W2", "WP")

# §6 verdict literals — all three are RECORDED OUTCOMES, none is a failure (§0.1).
GRADUATED = D.GRADUATED
NOT_GRADUATED = D.NOT_GRADUATED
VOID_UNDERPOWERED = D.VOID_UNDERPOWERED
VERDICTS = D.VERDICTS

# Named non-verdict outcomes. These are dispositions of a CELL, not graduation verdicts, and they
# are literals so that a coverage table can distinguish "did not run" from "ran and could not
# read out" from "cannot be run at all with the artifacts that exist".
VOID_READOUT = "VOID-readout"                    # §0.2.9 readout-validity gate fired
VOID_INSTRUMENT = "VOID-INSTRUMENT"              # the instrument, not the hypothesis, failed
COVERAGE_IMPOSSIBLE = "COVERAGE-IMPOSSIBLE"      # §4 pivot: the artifact does not exist
OUTCOMES = VERDICTS + (VOID_READOUT, VOID_INSTRUMENT, COVERAGE_IMPOSSIBLE)

# §3 dose discipline, inherited unchanged.
MIN_DOSES = D.MIN_DOSES

# ---------------------------------------------------------------------------
# §0.2.7 compute policy. g6e L40S ONLY (H100 CloudTrail-verified inaccessible: 0/105 attempts);
# p5 is dropped from every --type list. CPU arms on r7i.2xlarge --no-weights.
# ---------------------------------------------------------------------------
GPU_TYPES = D.GPU_TYPES
GPU_TYPE_LADDER = D.GPU_TYPE_LADDER
CPU_TYPE = D.CPU_TYPE
check_instance_types = D.check_instance_types

# §0.2.7 names M-A2/A3/A5, M-11, M-B2, M-E4 as the optimizer-mechanics arms on which 8-bit or
# otherwise approximate Adam is PROHIBITED (the second moment is the object under study).
#
# PRE-FREEZE AMENDMENT A1 (journaled 2026-08-14, before freeze, before any data): three further
# arms have the optimizer itself as the object under study and are therefore added to the set —
# widening a prohibition, never narrowing one:
#   M-Q  — the causal Adam-basis test (rotate, then fine-tune; the basis of the second moment IS
#          the independent variable);
#   M-3  — the Muon natural experiment (the mechanism claim is "no per-coordinate second moment");
#   M-10 — switch (b) replaces per-coordinate v_i with one subspace-level v_blk.
# An approximate optimizer on any of these would confound the treatment with its approximation.
OPTIMIZER_MECHANICS_ARMS = frozenset({"M-A2", "M-A3", "M-A5", "M-11", "M-B2", "M-E4",
                                      "M-Q", "M-3", "M-10"})
APPROXIMATE_OPTIMIZERS = D.APPROXIMATE_OPTIMIZERS


def check_optimizer(arm: str, optimizer_name: str, *, offload_exact_fp32: bool = None) -> dict:
    """Refuse an approximate optimizer on an optimizer-mechanics arm (§0.2.7 + amendment A1).

    `offload_exact_fp32` must be stated explicitly (True/False) for those arms, so the artifact
    records that the exact-math configuration was a decision rather than a default.
    """
    name = str(optimizer_name).lower().replace("-", "").replace(" ", "")
    approx = any(k.replace("_", "") in name.replace("_", "") for k in APPROXIMATE_OPTIMIZERS)
    if arm in OPTIMIZER_MECHANICS_ARMS:
        assert not approx, (
            f"{arm}: optimizer {optimizer_name!r} is approximate, and §0.2.7 PROHIBITS 8-bit or "
            f"approximate Adam on optimizer-mechanics arms {sorted(OPTIMIZER_MECHANICS_ARMS)} — "
            f"the second-moment representation is the object under study")
        assert offload_exact_fp32 is not None, (
            f"{arm}: state offload_exact_fp32 explicitly — §0.2.7's memory rule for 3B-class full "
            f"FT on 48GB is CPU-offloaded optimizer states with EXACT fp32 Adam math")
    return {"arm": arm, "optimizer": optimizer_name, "is_approximate": approx,
            "optimizer_mechanics_arm": arm in OPTIMIZER_MECHANICS_ARMS,
            "offload_exact_fp32": offload_exact_fp32}


# §3 standing control battery — inherited from DARWIN unchanged (topsigma32 included).
TWIN_IDS = D.TWIN_IDS
BANK_IDS = D.BANK_IDS
bank_ids_for = D.bank_ids_for
check_controls = D.check_controls
check_dose_grid = D.check_dose_grid

# ---------------------------------------------------------------------------
# Provenance. MERCATOR needs its OWN stamp: `infra/pg/darwin_provenance.json` and
# `infra/pg/humboldt_provenance.json` are BOTH committed in the tree, so on the instance an
# inherited provenance() would stamp every MERCATOR row with another program's frozen sha —
# exactly the failure DARWIN logged as defect D1.
# ---------------------------------------------------------------------------
PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "mercator_provenance.json"


def provenance() -> dict:
    """Run provenance for every atlas row. `pg run` ships only `infra/`, so there is no .git on the
    instance; scripts/mercator_launch.sh stamps this file immediately before each launch, in the
    same step as the launch, so it cannot go stale (DARWIN defect D41)."""
    if PROVENANCE_FILE.exists():
        d = json.loads(PROVENANCE_FILE.read_text())
        assert d.get("program") == PROGRAM, (
            f"{PROVENANCE_FILE.name} carries program={d.get('program')!r}, not {PROGRAM!r} — a "
            f"foreign stamp would put another program's sha on MERCATOR rows")
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


# §0.2.3 / §3 — the expectation protocol is DARWIN's, unchanged and unforked. Its console line
# reads "[darwin][expect] …" even in a MERCATOR job; that string is cosmetic and the ROW it writes
# carries the arm id, so it is left alone rather than forked for a log prefix.
Expectations = D.Expectations

# The expectation store lives under studies/, and `pg run` tars only infra/ — so an arm running on
# an instance cannot see it. Publishing it to S3 at freeze is what makes §3's "pre-logged BEFORE
# the arm runs" enforceable ON THE INSTANCE, from the durable artifact, rather than on the honour
# of whoever assembled the job's arguments.
EXPECTATIONS_S3 = f"{MERCATOR_S3}/expectations.jsonl"


def publish_expectations(path):
    """Upload the frozen expectation store. Called at freeze and after each per-wave pre-log."""
    p = pathlib.Path(path)
    assert p.exists() and p.stat().st_size > 0, f"no expectations to publish at {p}"
    subprocess.run(["aws", "s3", "cp", str(p), EXPECTATIONS_S3, "--region", "us-east-1"],
                   check=True)
    return EXPECTATIONS_S3


def fetch_expectations(dest):
    """Pull the frozen expectation store and return it. Fails loud if it is absent: an arm that
    cannot verify its own pre-logged expectation must not run."""
    d = pathlib.Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", EXPECTATIONS_S3, str(d), "--region", "us-east-1"],
                   check=True)
    e = Expectations(d)
    assert e.rows, f"{EXPECTATIONS_S3} is empty — no arm has a pre-logged expectation"
    return e

# ---------------------------------------------------------------------------
# §3 — the intervention atlas. DARWIN's schema + `queue_item`.
# ---------------------------------------------------------------------------
MERCATOR_EXTRA = ("queue_item",)
ATLAS_COLUMNS = D.ATLAS_COLUMNS + MERCATOR_EXTRA

# §3 dual-basis rule. A statistic whose claim is about coordinates is written twice: once in the
# model's coordinate basis, once in ONE random reference basis. `basis` rides in `flags` rather
# than becoming a 27th column because it qualifies a measurement, and every other measurement
# qualifier in this schema already lives there.
BASIS_COORD = "coord"
BASIS_RANDOM_REF = "random_ref"
BASES = (BASIS_COORD, BASIS_RANDOM_REF)


class Atlas(H.Atlas):
    """Append-only two-register atlas writer.

    Constructor-bound: `register`, `wave`, `expected_ref`, `queue_item` — an arm belongs to exactly
    one register, runs in exactly one wave, has exactly one live expectation, and maps to exactly
    one queue item (§5). Making any of them per-call would make a mixed-arm file representable.

    Per-call: `dose`, `incumbent_id`, `basis` — a dose curve is many rows of one arm, a Register-B
    arm records treatment and incumbent side by side, and the dual-basis rule is a property of the
    individual measurement.

    Assembly is local rather than delegated to D.Atlas.add for the reason DARWIN gave for not
    delegating to H.Atlas.add: the parent asserts its own column set, so delegation would force a
    read-modify-write of the last line of the artifact on every row.
    """

    def __init__(self, out_dir, arm: str, run_id: str, *, register: str, wave: str,
                 expected_ref: str, queue_item: str, tag: str = H.DISCOVERY):
        assert register in REGISTERS, f"unknown register {register!r} (expected one of {REGISTERS})"
        assert wave in WAVES, f"unknown wave {wave!r} (expected one of {WAVES})"
        assert expected_ref and expected_ref.startswith(f"{arm}::"), (
            f"expected_ref {expected_ref!r} does not belong to arm {arm} — §3 requires the arm's "
            f"own pre-logged expectation id")
        assert queue_item and str(queue_item).strip(), (
            f"{arm}: queue_item is the §5 coverage-map key and may not be empty — an arm with no "
            f"queue item cannot be checked against the no-stone-unturned contract")
        super().__init__(out_dir, arm=arm, run_id=run_id, tag=tag)
        # Re-stamp: the parent set self.sha from HUMBOLDT's committed provenance file.
        self.sha = commit_sha()
        self.register = register
        self.wave = wave
        self.expected_ref = expected_ref
        self.queue_item = str(queue_item)
        # Per-STATISTIC provenance, accumulated as rows are written. All three exist because the
        # corresponding contract is a property of a contrast, not of a job (critic findings F7,
        # F10) — checking them on the arm-level union is what let DARWIN's gaps through.
        self.bases_seen = {}          # statistic -> set(basis)     -> check_dual_basis
        self.stat_banks = {}          # statistic -> set(bank_id)   -> check_controls_per_statistic
        self.stat_corpora = {}        # statistic -> set(corpus_id) -> check_readout_corpus

    def add(self, statistic: str, *, value=None, vector_ref=None, layer=None, type=None,
            bank_id=None, model_id=None, model_sha=None, n=None, ci_lo=None, ci_hi=None,
            seed=None, corpus_id=None, item_id=None, token_class=None, flags=None, tag=None,
            incumbent_id=None, dose=None, basis=None):
        """One atlas row, assembled and written in one append.

        `basis` (§3 dual-basis rule) must be one of BASES when given. It is recorded in `flags` and
        tracked on the writer so `check_dual_basis` can enforce the pairing at the end of the arm.
        """
        assert (value is None) != (vector_ref is None), \
            f"{statistic}: exactly one of value / vector_ref (got {value!r} / {vector_ref!r})"
        if value is not None:
            value = float(value)
            assert value == value, f"{statistic}: NaN value (record it as a flag, not a silent NaN)"
        flags = dict(H._jsonable(flags or {}))
        if basis is not None:
            assert basis in BASES, f"{statistic}: unknown basis {basis!r} (expected one of {BASES})"
            flags["basis"] = basis
            self.bases_seen.setdefault(statistic, set()).add(basis)
        self.stat_banks.setdefault(statistic, set()).add(bank_id)
        self.stat_corpora.setdefault(statistic, set()).add(corpus_id)
        row = {"arm": self.arm, "model_id": model_id, "model_sha": model_sha,
               "layer": None if layer is None else int(layer), "type": type, "bank_id": bank_id,
               "statistic": statistic, "value": value, "vector_ref": vector_ref,
               "n": None if n is None else int(n),
               "ci_lo": None if ci_lo is None else float(ci_lo),
               "ci_hi": None if ci_hi is None else float(ci_hi),
               "seed": None if seed is None else int(seed), "corpus_id": corpus_id,
               "item_id": item_id, "token_class": token_class,
               "flags": json.dumps(flags, sort_keys=True),
               "run_id": self.run_id, "commit_sha": self.sha, "tag": tag or self.tag,
               "register": self.register, "wave": self.wave, "incumbent_id": incumbent_id,
               "dose": None if dose is None else float(dose),
               "expected_ref": self.expected_ref, "claimed_by": CLAIMED_BY,
               "queue_item": self.queue_item}
        assert set(row) == set(ATLAS_COLUMNS), \
            f"schema drift: {sorted(set(row) ^ set(ATLAS_COLUMNS))}"
        self._fh.write(json.dumps(row) + "\n")
        self.n_rows += 1

    def add_ci(self, statistic: str, values, **kw):
        """Sampled statistic: mean + bootstrap CI + n. The CI regime rides in the flags."""
        seed = kw.pop("seed", 1337)
        s = H.mean_ci(values, seed=seed)
        flags = {**(kw.pop("flags", None) or {}), "ci_method": s["ci_method"],
                 "n_boot": s["n_boot"]}
        self.add(statistic, value=s["mean"], n=s["n"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                 seed=seed, flags=flags, **kw)
        return s

    def check_dual_basis(self, statistics):
        """§3: every named coordinate-claim statistic carries BOTH bases in this table.

        Called at the end of an arm with the list of statistics whose claim is about coordinates.
        A statistic that was never written at all is as much a defect as one written in one basis:
        silent absence is how a dual-basis rule degrades into a single-basis result.
        """
        return check_dual_basis(self.bases_seen, statistics, arm=self.arm)

    def check_controls_per_statistic(self, statistics):
        """§3 battery, per bank-treatment statistic (critic finding F7)."""
        return check_controls_per_statistic(self.stat_banks, statistics, arm=self.arm)

    def check_readout_corpus(self, statistics):
        """Every declared task/ppl readout names its corpus (critic finding F10)."""
        return check_readout_corpus(self.stat_corpora, statistics, arm=self.arm)

    def to_parquet(self):
        import pandas as pd
        self.close()
        df = pd.read_json(self.path, lines=True)
        # AN EMPTY ATLAS IS A LEGITIMATE OUTCOME, NOT DRIFT (M-D92). An arm that reaches a
        # recorded outcome — VOID-INSTRUMENT, COVERAGE-IMPOSSIBLE — before adding a row leaves an
        # empty file, and pandas gives an empty frame no columns at all, so the drift assert fires
        # with `[]`. The column-order check is about a schema that CHANGED; an empty file has no
        # schema to change. Guarded in ALL THREE copies of this method: fixing only darwin_core's
        # left mercator_core's to abort the same pipeline one run later.
        if df.empty:
            return {"n_rows": 0, "path": str(self.path), "parquet": None,
                    "note": "atlas empty: the arm recorded an outcome before adding any row"}
        assert list(df.columns) == list(ATLAS_COLUMNS), \
            f"parquet column order drift: {list(df.columns)}"
        for c in ("model_id", "model_sha", "type", "bank_id", "vector_ref", "corpus_id",
                  "item_id", "token_class", "flags", "run_id", "commit_sha", "tag", "arm",
                  "statistic", "register", "wave", "incumbent_id", "expected_ref", "claimed_by",
                  "queue_item"):
            df[c] = df[c].astype("object").where(df[c].notna(), None)
        p = self.out / "atlas.parquet"
        df.to_parquet(p, index=False)
        return {"path": str(p), "n_rows": int(len(df)),
                "n_vectors": len(list(self.vec.glob("*.npy")))}


def check_dual_basis(bases_seen: dict, statistics, *, arm: str):
    """§3 grid-artifact guard, as a free function so an aggregator can apply it to a loaded table.

    `bases_seen` maps statistic -> iterable of basis tags. `statistics` is the arm's declared list
    of coordinate-claim statistics.
    """
    missing = {}
    for s in statistics:
        have = set(bases_seen.get(s, ()))
        need = set(BASES) - have
        if need:
            missing[s] = sorted(need)
    assert not missing, (
        f"{arm}: dual-basis rule violated for {missing}. §3 requires every coordinate claim in "
        f"BOTH the coordinate basis and one random reference basis — a coordinate result measured "
        f"only in coordinates cannot be distinguished from a measurement artifact")
    return sorted(statistics)


def random_reference_basis(n: int, seed: int):
    """The ONE random reference basis a dual-basis measurement is repeated in (§3).

    A Haar-random orthogonal Q on R^n. Measuring in this basis means measuring the SAME operator
    after conjugation by Q: a quantity that is genuinely about the model's coordinate grid moves,
    and one that is basis-free does not. Reuses HUMBOLDT's Haar generator so the reference basis
    and the haar32 twin banks come from one tested implementation.
    """
    import numpy as np
    Q = H.haar_subspace(n, n, seed)
    assert Q.shape == (n, n), f"reference basis shape {Q.shape}, expected ({n}, {n})"
    assert H.orthonormal_check(Q) < 1e-8, "reference basis is not orthonormal"
    return np.asarray(Q, dtype=np.float64)


# ---------------------------------------------------------------------------
# §0.2.9 — the readout-validity gate, now a standing property of every task readout.
# ---------------------------------------------------------------------------
def readout_validity(*, ft_score: float, base_score: float, higher_is_better: bool,
                     model_id: str, task: str, margin: float = 0.0) -> dict:
    """A fine-tune must beat its BASE on its OWN task, else the cell is VOID-readout (§0.2.9).

    This is not a quality judgement about the fine-tune. It is an instrument check: if the readout
    cannot see the thing the fine-tune was trained to do, then any retention/damage number measured
    through that readout is measuring the instrument, not the model. DARWIN applied this per arm by
    hand; here it is one function so that no arm can quietly skip it.

    `margin` (>=0, in readout units) is the amount by which the FT must beat the base. Default 0
    means strict improvement is required, and a tie reads as VOID-readout.
    """
    assert margin >= 0.0, f"margin must be non-negative, got {margin}"
    delta = float(ft_score) - float(base_score)
    signed = delta if higher_is_better else -delta
    ok = bool(signed > margin)
    return {"gate": "readout_validity", "task": task, "model_id": model_id, "passed": ok,
            "outcome": None if ok else VOID_READOUT,
            "ft_score": float(ft_score), "base_score": float(base_score),
            "signed_improvement": signed, "margin": float(margin),
            "higher_is_better": bool(higher_is_better)}


# ---------------------------------------------------------------------------
# §6 — the graduation calculus. The FORMULA is DARWIN's, reused unforked. What is NOT reused is
# DARWIN's way of CALLING it, and that is the Wave-K fold's single most consequential change.
#
# Critic finding F1 (BLOCKING): DARWIN's only production call site passed `mde=0.0` and
# `cost_ratio=1.0` as literals. `ci_halfwidth` is asserted non-negative, so `hw < 0.0` is never
# true, so `underpowered` was always False, so the VOID-underpowered branch was STRUCTURALLY
# UNREACHABLE for all 28 graduation rows — and `cost_ratio <= 0.1` was never true either, so the
# "ties at <= 1/10 cost" pass-path was dead as well. `power_floor()` existed, was unit-tested, and
# was never called from production code.
#
# That is the one-sided-threshold failure CLAUDE.md and §3 forbid, sitting in the exact artifact
# MERCATOR reads verdicts from. Re-exporting D.effect_criterion would have let MERCATOR repeat it
# verbatim. So MERCATOR's `effect_criterion` does not accept `mde` or `cost_ratio` at all: the MDE
# is DERIVED from the arm's own control sample via `power_floor`, and the cost ratio is DERIVED
# from two measured costs. A caller cannot fabricate either one.
#
# MERCATOR has no G0-style program gate, so `gate_g0` is deliberately NOT re-exported.
# ---------------------------------------------------------------------------
ci_halfwidth = D.ci_halfwidth
power_floor = D.power_floor
surprise = D.surprise
SurpriseLog = D.SurpriseLog


def effect_criterion(*, treatment_mean: float, treatment_ci_lo: float, treatment_ci_hi: float,
                     incumbent_mean: float, higher_is_better: bool, control_values,
                     n_designed: int, treatment_cost: float, incumbent_cost: float,
                     cost_unit: str) -> dict:
    """§6(1) EFFECT, with both of F1's fabricated inputs replaced by derived ones.

    `control_values` is the arm's OWN control column at the same cell — the sample whose bootstrap
    spread defines what the arm could have detected. `n_designed` is the n the arm was designed
    for. Together they give the MDE, and the VOID-underpowered branch becomes reachable.

    `treatment_cost` / `incumbent_cost` are MEASURED, in a stated unit (L40S-h, bytes, tokens —
    whatever the arm's §6 cost is), so the "<= 1/10 cost" path is reachable too.

    Returns DARWIN's row, plus the power floor and the cost inputs, so the artifact carries every
    number the formula used.
    """
    pf = power_floor(control_values, n_designed=n_designed)
    assert pf["mde"] > 0, (
        "the power floor came out at 0 — a control sample with no spread cannot define a minimum "
        "detectable effect, and passing 0 would make the VOID-underpowered branch unreachable "
        "(critic finding F1)")
    assert cost_unit and cost_unit.strip(), "state the cost unit; §6 records cost in-row"
    assert treatment_cost > 0 and incumbent_cost > 0, (
        f"costs must be measured and positive (got {treatment_cost} / {incumbent_cost}); a "
        f"hardcoded ratio of 1.0 is what killed DARWIN's tie-at-a-tenth path")
    d = D.effect_criterion(treatment_mean=treatment_mean, treatment_ci_lo=treatment_ci_lo,
                           treatment_ci_hi=treatment_ci_hi, incumbent_mean=incumbent_mean,
                           higher_is_better=higher_is_better,
                           cost_ratio=treatment_cost / incumbent_cost, mde=pf["mde"])
    d["power_floor"] = pf
    d["cost"] = {"treatment": float(treatment_cost), "incumbent": float(incumbent_cost),
                 "unit": cost_unit}
    return d


def graduate(arm: str, *, effect: dict, envelope_measured: bool, envelope_evidence: str,
             interaction_terms=(), incumbent_id: str = None) -> dict:
    """§6 verdict. DARWIN's calculus, with one added precondition from critic finding F14.

    F14: DARWIN applied ONE global W2-F interaction term to every GRAD arm's graduation cell,
    including arms whose relationship to W2-F's actual factors was never established. §6 says the
    COMPOSITION leg applies "where it joins a stack" and needs "the RELEVANT interaction cell". So
    every interaction term must name the arms it was measured for, and a term that does not name
    this arm is refused rather than silently applied.
    """
    assert "power_floor" in effect, (
        f"{arm}: effect dict has no 'power_floor' key, so it did not come from "
        f"mercator_core.effect_criterion — only that function derives the MDE from a control "
        f"sample, and a hand-built dict can carry mde=0 (critic finding F1)")
    for t in interaction_terms:
        assert "applies_to" in t, (
            f"{arm}: interaction term {t.get('name')!r} does not say which arms it was measured "
            f"for. §6 needs the RELEVANT interaction cell; a global term applied to every arm is "
            f"critic finding F14")
        assert arm in t["applies_to"], (
            f"{arm}: interaction term {t['name']!r} was measured for {sorted(t['applies_to'])} and "
            f"does not cover this arm — record COMPOSITION as unmeasured instead of borrowing it")
    return D.graduate(arm, effect=effect, envelope_measured=envelope_measured,
                      envelope_evidence=envelope_evidence,
                      interaction_terms=[{k: v for k, v in t.items() if k != "applies_to"}
                                         for t in interaction_terms],
                      incumbent_id=incumbent_id)


def check_controls_per_statistic(stat_banks: dict, statistics, *, arm: str):
    """§3's battery, checked PER STATISTIC — critic finding F7.

    DARWIN's `check_controls` is called once per arm on the UNION of bank_ids across all of that
    arm's statistics. W1-K's real post-fine-tune refuter statistic carried 4 of 8 banks (missing
    topsigma32, bottomsigma32, haar32_s1, haar32_s2) and the check still passed, because OTHER
    statistics in the same arm carried the full battery and masked the gap.

    A control battery is a property of a CONTRAST, not of a job. `stat_banks` maps statistic ->
    iterable of bank_id; `statistics` names the bank-treatment statistics that must carry the
    battery.
    """
    bad = {}
    for s in statistics:
        have = set(stat_banks.get(s, ()))
        need = {"core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2"} - have
        if need:
            bad[s] = sorted(need)
    assert not bad, (
        f"{arm}: per-statistic control battery incomplete: {bad}. A full battery on a DIFFERENT "
        f"statistic of the same arm does not control this one (critic finding F7)")
    return sorted(statistics)


def check_readout_corpus(stat_corpora: dict, statistics, *, arm: str):
    """Every task/perplexity readout names the corpus it was measured on — critic finding F10.

    DARWIN carries four unreconciled `base_ppl` anchors for the SAME base model, and `corpus_id` is
    null on all 76,564 atlas rows, so which eval window produced which anchor cannot be recovered
    from the artifact at all. That is not a small bookkeeping gap: it makes cross-arm perplexity
    comparison unfalsifiable. Here the corpus is required on every declared readout statistic.
    """
    missing = sorted(s for s in statistics
                     if not {c for c in stat_corpora.get(s, ()) if c})
    assert not missing, (
        f"{arm}: readout statistics {missing} carry no corpus_id. Four unreconciled base_ppl "
        f"anchors with a null corpus column is critic finding F10 — a readout without its corpus "
        f"cannot be compared to another arm's")
    return sorted(statistics)


# ---------------------------------------------------------------------------
# Bank library + artifacts.
# ---------------------------------------------------------------------------
def fetch_all_banks(dest, *, require_topsigma: bool = True, require_mercator: bool = False):
    """HUMBOLDT's library + DARWIN's derived banks + MERCATOR's own, in ONE directory.

    Three prefixes rather than one because derived banks must stay attributable to the program that
    derived them: writing MERCATOR's output-side banks back into `humboldt/banks/` would make the
    HUMBOLDT library no longer the object HUMBOLDT published.
    """
    d = D.fetch_all_banks(dest, require_topsigma=require_topsigma)
    subprocess.run(["aws", "s3", "sync", BANKS_S3, str(d), "--region", "us-east-1"], check=True)
    n_row = len(list(d.glob("*__row__core32.npy")))
    if require_mercator:
        assert n_row > 0, (
            f"no MERCATOR output-side banks at {BANKS_S3} — M-B1 must publish them before any arm "
            f"that treats the row space")
    print(f"[mercator] bank library: {len(list(d.glob('*.npy')))} objects "
          f"({n_row} MERCATOR-derived output-side)", flush=True)
    return d


def publish_banks(out_dir):
    """Sync out/banks/ to the MERCATOR bank prefix."""
    subprocess.run(["aws", "s3", "sync", str(pathlib.Path(out_dir) / "banks"), BANKS_S3,
                    "--region", "us-east-1"], check=True)


def fetch_prior_arm(dest, s3_prefix: str):
    """Pull a completed arm's artifacts so a re-analysis arm can join against them (§0.22:
    never a number from prose). Fails loud if the prefix has no atlas."""
    return D.fetch_prior_arm(dest, s3_prefix)


load_atlas = D.load_atlas
vector_of = D.vector_of
write_json = D.write_json
utc_now = D.utc_now


def s3_sync(out_dir, s3_prefix: str):
    """§0.2.5 incremental per-arm sync."""
    subprocess.run(["aws", "s3", "sync", str(out_dir), f"s3://{BUCKET}/{s3_prefix}",
                    "--region", "us-east-1"], check=True)


def load_recensus_banks(prefixes, dest):
    """Index the re-census eigenvector banks held under one or more S3 prefixes.

    Returns `(index, origin, complete_bases)`: filename -> local path, filename -> the prefix it
    came from, and the set of bases whose census actually FINISHED.

    THREE THINGS THIS EXISTS TO PREVENT, all of them observed:

    1. A STALE PRODUCER PREFIX. `mercator_launch.sh` refuses a prefix that already holds a STATUS,
       so a dependency that has to be retried lands under a new name each time (recensus, recensus2,
       … recensus6). A consumer that spells the prefix as a constant is wrong at the first retry and
       fails silently, because `aws s3 sync` of an empty prefix succeeds with zero files and the arm
       then records a truthful-looking COVERAGE-IMPOSSIBLE naming a dependency that was in fact
       being produced two prefixes over. That is exactly how M-7, M-9b and M-C4 were decided (M-D99).

    2. A COLLISION BETWEEN TWO PRODUCERS. Each prefix syncs into its own subdirectory and a repeated
       filename is asserted rather than resolved by precedence. The bank files are NOT
       interchangeable across runs: the census roster is a live Hub query, so two re-census runs of
       one base can sum over different derivative sets and produce different eigenvectors under
       identical names. Letting the later sync win would silently mix two Grams.

    3. A REAPED CENSUS THAT LOOKS COMPLETE. The EventBridge reaper terminates the instance outright,
       and `run_job.sh` streams `out/` every 30 s, so a census killed part-way through its base list
       leaves banks on S3 with no STATUS and no way to tell from the `.npy` files alone which bases
       finished. `recensus_partial.json` is written only AFTER a base returns, so it — not the
       presence of the files — is the completeness signal. A consumer must read it, or it will
       report a result for a base whose Gram was never finished.

    Fail-loud: a prefix whose sibling `recensus_partial.json` is unreadable raises. A prefix with no
    completed base is a broken dependency, and a crash naming the missing file is more use than a
    silent zero that reads as a scientific null.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    index, origin, complete = {}, {}, {}
    for pfx in [p.strip() for p in (prefixes.split(",") if isinstance(prefixes, str)
                                    else prefixes) if p.strip()]:
        sub = dest / pfx.replace("/", "_")
        sub.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{pfx}", str(sub),
                        "--region", "us-east-1"], check=True)
        for p in sorted(sub.glob("*.npy")):
            assert p.name not in index, (
                f"bank {p.name} is present in BOTH {origin[p.name]} and {pfx}; two re-census runs "
                f"banked the same cell and they may sum over different rosters — pass only one")
            index[p.name], origin[p.name] = p, pfx
        # the sibling of the banks directory, i.e. `<run>/out/recensus_partial.json`
        sib = pfx.rsplit("/banks", 1)[0].rstrip("/") + "/recensus_partial.json"
        raw = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{sib}", "-",
                              "--region", "us-east-1"], check=True, capture_output=True, text=True)
        for base, rec in json.loads(raw.stdout).items():
            # `n_cells_banked` is written by `census_one` on the success path only; a base that
            # returned COVERAGE-IMPOSSIBLE carries an `outcome` and no cell count.
            if isinstance(rec, dict) and rec.get("n_cells_banked"):
                complete[base] = pfx
        print(f"[banks] {pfx}: {len(index)} files indexed, complete bases so far "
              f"{sorted(complete)}", flush=True)
    return index, origin, complete


def load_decoder_matrices(repo: str, sha: str, types, layers) -> dict:
    """{(type, layer): tensor} for a PaliGemma checkpoint's LANGUAGE-MODEL decoder.

    NOT `darwin_census.load_decoder_matrices`. That one resolves a key by suffix match on
    `layers.N.self_attn.T.weight`, and PaliGemma's SigLIP vision tower carries the identical tail —
    so it finds two hits and fails loud on every PG1 checkpoint. Correct behaviour from a resolver
    that was written for single-tower checkpoints, and the wrong function to call here; it is
    defect M-D6 and then M-D37, met by four separate jobs.

    `mercator_rowgram._load_decoder_retrying` is already scoped to the language model, carries the
    bounded counted transport retry a 24-checkpoint stream needs, and is what banked M-B1's cells —
    so routing every MERCATOR job through it also means two arms reading "the same cell" read the
    same tensor.
    """
    from infra.pg import mercator_rowgram as _RG
    from infra.pg.humboldt_weights import matrix_key as _mk
    full, _retries = _RG._load_decoder_retrying(repo, sha)
    out = {}
    for t in types:
        for L in layers:
            key = _mk(t, int(L))
            assert key in full, (
                f"{repo}: {key!r} absent from the decoder load; present sample "
                f"{sorted(k for k in full if f'layers.{L}.' in k)[:4]}")
            out[(t, int(L))] = full[key]
    return out


def battery_supervised_target(task, row) -> str:
    """The string a fine-tune WOULD be trained on for one battery item, per task, by name.

    The battery's tasks carry a scorer, not a target: `vqav2` scores against
    `[a["answer"] for a in row["answers"]]` and has no single gold, `coco_captions` is a corpus
    task with five references, and the rest keep their answer under their own key. Three MERCATOR
    arms (M-G, M-C3, M-C9) need a supervised target, and the first version guessed
    `row.get("answer")` — which is empty for every VQAv2 row, so M-G built zero training batches
    and asserted after paying for two full task evaluations (defect M-D36).

    Keyed by task NAME and fails loud on an unknown one: a silent fallback is exactly what produced
    the defect.
    """
    name = task.name
    if task.is_corpus:
        assert task.refs_fn is not None, f"{name}: corpus task without refs_fn"
        refs = task.refs_fn(row)
        assert refs, f"{name}: corpus row with no references"
        return str(refs[0])
    if name == "vqav2":
        # The modal human answer: what a supervised fine-tune of this task is trained toward, and
        # the answer the soft-accuracy scorer rewards most.
        import collections
        answers = [str(a["answer"]) for a in row["answers"]]
        assert answers, "vqav2 row with no answers"
        return collections.Counter(answers).most_common(1)[0][0]
    if name == "textvqa":
        answers = [str(a) for a in row.get("answers", []) if str(a).strip()]
        assert answers, "textvqa row with no answers"
        import collections
        return collections.Counter(answers).most_common(1)[0][0]
    if name == "refcoco":
        raise AssertionError(
            "refcoco's target is a <locDDDD> token sequence and is not a supervised string this "
            "helper can build; an arm that wants it must say so explicitly")
    raise AssertionError(f"no supervised target rule for battery task {name!r}")


def ensure_pycocoevalcap():
    """`projector_ablation._cider_boot` imports pycocoevalcap for the corpus (CIDEr-D) tasks, and
    the AWS AMI does not carry it — M-G died on the import after already scoring VQAv2.

    Same shape as `diff_harvest.ensure_wandb`: idempotent, fail-loud, called at the top of any job
    that scores a corpus task. Only the CIDEr scorer is used, which is pure Python; the package's
    METEOR/SPICE scorers need a JVM and are never touched.
    """
    import importlib.util
    import subprocess
    import sys as _sys
    if importlib.util.find_spec("pycocoevalcap") is None:
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap"],
                       check=True)
    import pycocoevalcap.cider.cider as _cider
    assert _cider is not None
    return True


def retry_transport(fn, *, what: str, attempts: int = 4):
    """Call `fn()`, retrying a TRANSPORT failure a bounded number of times. Returns (value, n_retry).

    A SYSTEM boundary, not a scientific one — the same carve-out `mercator_rowgram` documents and
    the only kind this program guards. An arm that streams a dozen multi-gigabyte checkpoints will
    eventually meet `peer closed connection without sending complete message body` mid-file, and
    M-D2 lost a run to one after it had already trained its dictionary to FVU 0.4372 (M-D67).

    Only transport errors are caught. A missing repo, a gated repo, a shape mismatch or anything
    scientific still propagates on the first occurrence, and the retry count is returned so a flaky
    run is visible in the artifact rather than indistinguishable from a clean one.
    """
    import time
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    transient = (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                 httpx.ReadTimeout, httpx.WriteError, HfHubHTTPError, ConnectionError)
    last = None
    for i in range(attempts):
        try:
            return fn(), i
        except transient as e:            # noqa: PERF203 - a bounded transport retry, counted
            last = e
            print(f"[retry] transport failure on {what} (attempt {i + 1}/{attempts}): "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{what}: {attempts} transport attempts failed; last was "
                       f"{type(last).__name__}: {last}")


def ensure_lerobot():
    """The π0/π0.5 policies live in `lerobot`, which the AWS AMI does not carry — M-A5 died on the
    import after its supervised reference had already trained for 600 steps.

    INSTALLED BEFORE ANY MODEL IS LOADED, and that ordering is the whole reason this is a function
    rather than an import. `lerobot[libero,pi]` pulls the maintainer's PINNED transformers, so
    installing it partway through a job swaps the library under an already-constructed PaliGemma —
    the arm must call this at the top of main(), before the first model touches the GPU, so both
    halves of the run see one transformers.

    The pip-only form: no apt, no EGL, no LIBERO config seed. Those matter only for RENDERING, and
    an arm that trains without rolling out does not need them (LACUNA's `--skip-egl` path is the
    same install and is the tested precedent).
    """
    import importlib.util
    import subprocess
    import sys as _sys
    if importlib.util.find_spec("lerobot") is None:
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                        "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"],
                       check=True)
    import lerobot
    assert lerobot is not None
    return True


def s3_sync_incremental(out_dir):
    """Push `out/` to this run's own prefix mid-flight, using the prefix the runner exported.

    The EventBridge reaper TERMINATES the instance at the deadline — it does not signal it — so a
    job whose only upload is the EXIT trap loses everything it produced if it overruns. Any loop
    whose iterations cost minutes should call this at the end of each iteration; the cost is one
    `aws s3 sync` of a directory that is almost entirely unchanged.

    Returns False when the runner did not export a prefix (a local dry-run), so a caller can tell
    "nothing to sync to" from "synced".
    """
    import os
    import pathlib as _pl
    prefix = os.environ.get("PG_S3_PREFIX")
    if not prefix:
        return False
    # MIRROR THE PATH THE TEARDOWN USES. `run_job.sh` uploads `$OUT` to `<prefix>/out/`, and inside
    # a pipeline each step's OUT is `out/<step-name>/`. Syncing the step's directory straight to
    # `<prefix>` flattens every step's files onto one another at the prefix root — two steps and
    # the second overwrites the first's atlas. Observed on the Wave-2 pipeline, where M-10's
    # partial atlas landed at `results/mercator_w2/atlas.jsonl` (defect M-D35).
    d = _pl.Path(out_dir).resolve()
    dest = f"{prefix}/out/{d.name}" if d.parent.name == "out" else f"{prefix}/out"
    s3_sync(d, dest)
    return True
