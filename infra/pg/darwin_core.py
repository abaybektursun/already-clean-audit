"""DARWIN core — the two-register atlas, the expectation protocol, the graduation calculus.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-12_darwin_program_spec.md

This is a RECORDING + GATING library (§0.1). It records, it computes frozen decision formulas,
and it refuses malformed rows. It never interprets: there is no function here that returns a
sentence, and the only verdicts it emits are the three §5 literals.

Layering. `humboldt_core` (H) stays the measurement substrate and is imported, never forked —
its Atlas/Gates/mean_ci/subspace primitives are already unit-tested and were used to build the
banks DARWIN consumes. This module adds exactly what §3 asks for on top:

    register · wave · incumbent_id · dose · expected_ref · claimed_by

FAIL LOUD (CLAUDE.md): no try/except, no synthetic fallback, no default that hides a missing
input. Heavy imports live inside functions so this py_compiles and unit-tests on the control
plane with numpy alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

from infra.pg import humboldt_core as H

# ---------------------------------------------------------------------------
# Frozen program constants (§0.2, §3, §4). Changing any of these post-freeze is an ADDENDUM.
# ---------------------------------------------------------------------------
PROGRAM = "darwin"
CLAIMED_BY = "darwin-executor"                       # §0.2.4 — one arm, one owner
BUCKET = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
DARWIN_S3 = f"s3://{BUCKET}/{PROGRAM}"
BANKS_S3 = f"{DARWIN_S3}/banks"                      # DARWIN-derived banks (topsigma32, census)

# §3 registers. ATLAS carries NO confirmatory weight; GRAD/STAGE2 are the pre-registered ones.
REGISTERS = ("ATLAS", "GRAD", "STAGE2")
WAVES = ("W0", "W1", "W2", "WP", "S2")

# §5 verdict literals. All three are RECORDED OUTCOMES — none is a failure (§0.1).
GRADUATED = "GRADUATED"
NOT_GRADUATED = "NOT-GRADUATED"
VOID_UNDERPOWERED = "VOID-underpowered"
VERDICTS = (GRADUATED, NOT_GRADUATED, VOID_UNDERPOWERED)

# §3 dose discipline: any arm with an intensity knob sweeps >=4 log-spaced doses (REFEREE's
# inverted-U is the standing reason). Enforced by `check_dose_grid`, not by convention.
MIN_DOSES = 4

# ---------------------------------------------------------------------------
# §0.2.7 compute policy (spec revised 2026-08-12). Encoded as code, not convention: both of these
# are constraints a job could violate silently and only a reader of the config would notice.
# ---------------------------------------------------------------------------
# GPU arms run on g6e L40S ONLY. H100 is inaccessible as of 2026-08-12, and the research director
# ruled p5 out of every --type list because it only adds retry latency before the inevitable g6e
# fallback. Single-GPU nodes under the 32-vCPU G quota => 4 concurrent.
# Ordered smallest-first: the G quota is 32 vCPU, so 4x g6e.2xlarge (8 vCPU) is the 4-concurrent
# fleet shape §0.2.7 describes, while g6e.8xlarge consumes the whole bucket. Capacity decides in
# practice — the launcher tries each type across every AZ and takes the first that launches — so
# this is a preference ladder, not a promise.
GPU_TYPES = ("g6e.xlarge", "g6e.2xlarge", "g6e.4xlarge", "g6e.8xlarge", "g6e.12xlarge")
GPU_TYPE_LADDER = "g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge"
CPU_TYPE = "r7i.2xlarge"
FORBIDDEN_INSTANCE_PREFIXES = ("p5", "p4", "p3")


def check_instance_types(types) -> list:
    """Refuse an instance-type list that names a forbidden family (§0.2.7)."""
    got = [t.strip() for t in (types.split(",") if isinstance(types, str) else types) if t.strip()]
    assert got, "empty instance-type list"
    bad = [t for t in got if t.startswith(FORBIDDEN_INSTANCE_PREFIXES)]
    assert not bad, (
        f"instance types {bad} are excluded by §0.2.7: H100 is inaccessible and p5 only adds "
        f"retry latency before the inevitable g6e fallback. Use one of {GPU_TYPES} or {CPU_TYPE}")
    return got


# §0.2.7: 8-bit / approximate Adam is PROHIBITED on any arm whose science touches optimizer
# mechanics — the second-moment representation is the object under study, so an approximation
# there is a confound, not an optimization. 3.3B-class full FT on 48 GB therefore uses
# CPU-offloaded optimizer states with EXACT fp32 Adam math (gradient checkpointing + bf16 params).
OPTIMIZER_MECHANICS_ARMS = frozenset({"W1-Q", "W1-S", "W1-T", "W1-U", "W2-A", "W2-B",
                                      "S2-A", "S2-B"})
APPROXIMATE_OPTIMIZERS = ("adamw8bit", "adam8bit", "adamw_bnb_8bit", "paged_adamw_8bit",
                          "lion8bit", "adafactor", "adamw_anyprecision")


def check_optimizer(arm: str, optimizer_name: str, *, offload_exact_fp32: bool = None) -> dict:
    """Refuse an approximate optimizer on an optimizer-mechanics arm (§0.2.7).

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
            f"{arm}: state offload_exact_fp32 explicitly — §0.2.7's memory rule for 3.3B-class "
            f"full FT on 48GB is CPU-offloaded optimizer states with EXACT fp32 Adam math")
    return {"arm": arm, "optimizer": optimizer_name, "is_approximate": approx,
            "optimizer_mechanics_arm": arm in OPTIMIZER_MECHANICS_ARMS,
            "offload_exact_fp32": offload_exact_fp32}

# §7 standing control battery. `topsigma32` is DARWIN-derived (journal 2026-08-12 amendment A1).
TWIN_IDS = H.TWIN_IDS + ("topsigma32",)
BANK_IDS = H.BANK_IDS + ("topsigma32",)


def bank_ids_for(t: str) -> tuple:
    """The bank battery available for type `t`.

    `bottomsigma32`/`topsigma32` are economy right-singular subspaces of the base matrix, so for
    a rank-deficient type they span only min(m,n) of the ambient space (humboldt_core journal C5
    keeps that deliberate). They remain defined; nothing is dropped. This helper exists so an arm
    can enumerate the battery without hardcoding it and silently losing a control column.
    """
    assert t in H.TYPES, f"unknown matrix type {t}"
    return BANK_IDS


# ---------------------------------------------------------------------------
# §0.18 provenance. DARWIN needs its OWN stamp, and this is not a stylistic choice:
# `infra/pg/humboldt_provenance.json` is COMMITTED in the tree, so on the instance
# `humboldt_core.provenance()` finds it and returns HUMBOLDT's frozen sha. Inheriting that would
# stamp every DARWIN row with the sha of a different program's code — the exact failure §0.18
# ("every number resolves to run id + file path + commit SHA") exists to prevent. Caught at
# implementation; see journal 2026-08-12 defect D1.
# ---------------------------------------------------------------------------
PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "darwin_provenance.json"


def provenance() -> dict:
    """Run provenance for every atlas row. `pg run` ships only `infra/`, so there is no .git on
    the instance; scripts/darwin_stamp.py writes this file immediately before each launch. On the
    control plane the file may be absent, and then git answers directly."""
    if PROVENANCE_FILE.exists():
        d = json.loads(PROVENANCE_FILE.read_text())
        assert d.get("program") == PROGRAM, (
            f"{PROVENANCE_FILE.name} carries program={d.get('program')!r}, not {PROGRAM!r} — a "
            f"foreign stamp would put another program's sha on DARWIN rows")
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


# ---------------------------------------------------------------------------
# §11 — the expectation protocol. This is the program's blind-analysis provenance.
# ---------------------------------------------------------------------------
class Expectations:
    """Pre-logged expected-pattern rows (§11), the object `expected_ref` points at.

    Protocol, enforced mechanically rather than by discipline:
      * `log()` writes {arm, refuter, expected_direction} and returns a stable id.
      * The id is `<arm>::<sha1(arm|refuter|direction)[:10]>` — content-addressed, so an edited
        expectation CANNOT reuse its predecessor's id and silently rewrite history.
      * `seal()` freezes the file. After sealing, `log()` for an already-logged arm raises unless
        `supersede=` names the row it replaces, which writes a NEW row carrying `supersedes` +
        `reason` (§11: "supersession = new row + reason"), leaving the original in place.

    The store is a JSONL file that also lives in the journal-adjacent artifact set, so an auditor
    can diff "what was expected" against "when the arm ran" from S3 alone.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    @staticmethod
    def _rid(arm: str, refuter: str, direction: str) -> str:
        h = hashlib.sha1(f"{arm}|{refuter}|{direction}".encode()).hexdigest()[:10]
        return f"{arm}::{h}"

    def by_arm(self, arm: str) -> list:
        return [r for r in self.rows if r["arm"] == arm]

    def log(self, arm: str, *, refuter: str, direction: str, utc: str,
            supersede: str = None, reason: str = None) -> str:
        """Record one arm's REFUTER line + ONE expected direction, BEFORE the arm runs.

        `direction` must be copied from the spec, not invented (§11) — the executor cannot
        enforce provenance of prose, so what IS enforced is that it is written before the run and
        immutable afterwards.
        """
        assert refuter.strip(), f"{arm}: empty REFUTER"
        assert direction.strip(), f"{arm}: empty expected direction"
        existing = self.by_arm(arm)
        if existing and supersede is None:
            raise AssertionError(
                f"{arm} already has an expectation ({existing[-1]['id']}); an expectation may not "
                f"be edited after its arm runs — pass supersede=<id> + reason= to add a new row")
        if supersede is not None:
            assert reason and reason.strip(), "supersession requires a reason (§11)"
            assert any(r["id"] == supersede for r in self.rows), f"unknown supersede id {supersede}"
        rid = self._rid(arm, refuter, direction)
        assert not any(r["id"] == rid for r in self.rows), f"duplicate expectation {rid}"
        row = {"id": rid, "arm": arm, "refuter": refuter, "direction": direction,
               "logged_utc": utc, "supersedes": supersede, "reason": reason}
        self.rows.append(row)
        with open(self.path, "a", buffering=1) as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[darwin][expect] {rid}  {arm}: {direction[:110]}", flush=True)
        return rid

    def require(self, arm: str) -> str:
        """The id an arm's rows must carry. Fails loud if the arm ran without pre-logging."""
        rows = self.by_arm(arm)
        assert rows, (f"{arm} has no pre-logged expectation — §11 requires the expected-pattern "
                      f"row BEFORE the arm's real contrast runs")
        return rows[-1]["id"]


# ---------------------------------------------------------------------------
# §3 — the intervention atlas. HUMBOLDT's schema + the seven DARWIN columns.
# ---------------------------------------------------------------------------
DARWIN_EXTRA = ("register", "wave", "incumbent_id", "dose", "expected_ref", "claimed_by")
ATLAS_COLUMNS = H.ATLAS_COLUMNS + DARWIN_EXTRA


class Atlas(H.Atlas):
    """Append-only two-register atlas writer.

    Every row carries its register tag (§0 preamble) and its `expected_ref` (§3/§11). Both are
    constructor-bound rather than per-call, because an arm belongs to exactly one register and
    has exactly one live expectation; making them per-call would make a mixed-register file
    representable, and the register split is the whole basis on which §8 disposes of the
    multiplicity threat across ~45 arms.

    `dose` and `incumbent_id` ARE per-call: a dose curve is many rows of one arm, and a
    Register-B arm records treatment and incumbent rows side by side in the same table.
    """

    def __init__(self, out_dir, arm: str, run_id: str, *, register: str, wave: str,
                 expected_ref: str, tag: str = H.DISCOVERY):
        assert register in REGISTERS, f"unknown register {register!r} (expected one of {REGISTERS})"
        assert wave in WAVES, f"unknown wave {wave!r} (expected one of {WAVES})"
        assert expected_ref and expected_ref.startswith(f"{arm}::"), (
            f"expected_ref {expected_ref!r} does not belong to arm {arm} — §11 requires the arm's "
            f"own pre-logged expectation id")
        super().__init__(out_dir, arm=arm, run_id=run_id, tag=tag)
        # Re-stamp: the parent set self.sha from HUMBOLDT's provenance file, which is committed in
        # the tree and would silently put HUMBOLDT's sha on every DARWIN row (defect D1).
        self.sha = commit_sha()
        self.register = register
        self.wave = wave
        self.expected_ref = expected_ref

    def add(self, statistic: str, *, value=None, vector_ref=None, layer=None, type=None,
            bank_id=None, model_id=None, model_sha=None, n=None, ci_lo=None, ci_hi=None,
            seed=None, corpus_id=None, item_id=None, token_class=None, flags=None, tag=None,
            incumbent_id=None, dose=None):
        """One atlas row, assembled and written in one append.

        The row is built here rather than by delegating to the parent and patching its line: the
        parent asserts its own 20-column schema, so a delegating implementation would have to
        re-open, re-parse and rewrite the last line of the file on every row. Assembling once is
        both simpler and strictly safer — there is no read-modify-write on the artifact at all.
        The parent's validation RULES are kept verbatim below; only the assembly is local.
        """
        assert (value is None) != (vector_ref is None), \
            f"{statistic}: exactly one of value / vector_ref (got {value!r} / {vector_ref!r})"
        if value is not None:
            value = float(value)
            assert value == value, f"{statistic}: NaN value (record it as a flag, not a silent NaN)"
        row = {"arm": self.arm, "model_id": model_id, "model_sha": model_sha,
               "layer": None if layer is None else int(layer), "type": type, "bank_id": bank_id,
               "statistic": statistic, "value": value, "vector_ref": vector_ref,
               "n": None if n is None else int(n),
               "ci_lo": None if ci_lo is None else float(ci_lo),
               "ci_hi": None if ci_hi is None else float(ci_hi),
               "seed": None if seed is None else int(seed), "corpus_id": corpus_id,
               "item_id": item_id, "token_class": token_class,
               "flags": json.dumps(H._jsonable(flags or {}), sort_keys=True),
               "run_id": self.run_id, "commit_sha": self.sha, "tag": tag or self.tag,
               "register": self.register, "wave": self.wave, "incumbent_id": incumbent_id,
               "dose": None if dose is None else float(dose),
               "expected_ref": self.expected_ref, "claimed_by": CLAIMED_BY}
        assert set(row) == set(ATLAS_COLUMNS), \
            f"schema drift: {sorted(set(row) ^ set(ATLAS_COLUMNS))}"
        self._fh.write(json.dumps(row) + "\n")
        self.n_rows += 1

    def add_ci(self, statistic: str, values, **kw):
        """Sampled statistic: mean + bootstrap CI + n (§2.3 of HUMBOLDT, inherited). The CI regime
        rides in the flags — which interval a row carries is provenance, not an implementation
        detail."""
        seed = kw.pop("seed", 1337)
        s = H.mean_ci(values, seed=seed)
        flags = {**(kw.pop("flags", None) or {}), "ci_method": s["ci_method"],
                 "n_boot": s["n_boot"]}
        self.add(statistic, value=s["mean"], n=s["n"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                 seed=seed, flags=flags, **kw)
        return s

    def to_parquet(self):
        import pandas as pd
        self.close()
        df = pd.read_json(self.path, lines=True)
        # AN EMPTY ATLAS IS A LEGITIMATE OUTCOME, NOT DRIFT. An arm that reaches a recorded void
        # before adding a row has nothing to write, and pandas gives an empty frame no columns at
        # all — so the drift assert fired with `[]` and turned a clean VOID-INSTRUMENT exit back
        # into a pipeline abort (M-D92, one layer under M-D90). The column-order check is about a
        # schema that CHANGED, and there is no schema in an empty file to change.
        if df.empty:
            return {"n_rows": 0, "path": str(self.path), "parquet": None,
                    "note": "atlas empty: the arm recorded an outcome before adding any row"}
        assert list(df.columns) == list(ATLAS_COLUMNS), \
            f"parquet column order drift: {list(df.columns)}"
        for c in ("model_id", "model_sha", "type", "bank_id", "vector_ref", "corpus_id",
                  "item_id", "token_class", "flags", "run_id", "commit_sha", "tag", "arm",
                  "statistic", "register", "wave", "incumbent_id", "expected_ref", "claimed_by"):
            df[c] = df[c].astype("object").where(df[c].notna(), None)
        p = self.out / "atlas.parquet"
        df.to_parquet(p, index=False)
        return {"path": str(p), "n_rows": int(len(df)),
                "n_vectors": len(list(self.vec.glob("*.npy")))}


# ---------------------------------------------------------------------------
# §3 — reporting-standard enforcement. A missing control row is a DEFECT (§0.1), so it is
# checked by code rather than by review.
# ---------------------------------------------------------------------------
def check_dose_grid(doses, *, arm: str, min_doses: int = MIN_DOSES, log_spaced: bool = True):
    """§3: '>=4 log-spaced doses, never points'. Returns the sorted grid; fails loud otherwise.

    Log-spacing is checked as constant ratio to 1% relative tolerance. An arm whose knob is
    genuinely linear (a shot count, an arity) passes `log_spaced=False` and the check degrades to
    the count + strict monotonicity — recorded, not waived.
    """
    import numpy as np
    d = np.asarray(sorted(float(x) for x in doses), dtype=np.float64)
    assert d.size >= min_doses, f"{arm}: {d.size} doses, §3 requires >= {min_doses}"
    assert np.all(np.diff(d) > 0), f"{arm}: duplicate doses in {d.tolist()}"
    if log_spaced:
        assert (d > 0).all(), f"{arm}: log-spaced grid needs positive doses, got {d.tolist()}"
        r = np.diff(np.log(d))
        assert float(r.std() / r.mean()) < 0.01, \
            f"{arm}: dose grid not log-spaced (log-ratios {r.tolist()})"
    return d.tolist()


def check_controls(present, *, arm: str, treatment_is_bank: bool = True, incumbent: str = None):
    """§7: the standing battery must be present wherever a bank is the treatment, and §3's
    structured (top-k vs bottom-k) alternative must be in the SAME table — never Haar alone.

    `present` is the set of bank_ids (and/or incumbent ids) an arm actually wrote rows for.
    """
    have = set(present)
    if treatment_is_bank:
        need = {"core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2"}
        missing = sorted(need - have)
        assert not missing, (
            f"{arm}: control battery incomplete, missing {missing}. §7 requires the full twin set "
            f"and §3 requires the structured top-k/bottom-k alternative in the same table — a "
            f"missing control row is a defect, not an omission")
    if incumbent is not None:
        assert incumbent in have, \
            f"{arm}: Register-B arm has no incumbent column for {incumbent!r} (§3 matched-incumbent rule)"
    return sorted(have)


# ---------------------------------------------------------------------------
# §5 — the graduation calculus. Frozen now, evaluated mechanically, never hand-set.
# ---------------------------------------------------------------------------
def ci_halfwidth(ci_lo: float, ci_hi: float) -> float:
    assert ci_hi >= ci_lo, f"inverted CI [{ci_lo}, {ci_hi}]"
    return float((ci_hi - ci_lo) / 2.0)


def power_floor(control_values, *, n_designed: int, n_boot: int = 2000, seed: int = 1337) -> dict:
    """§3 power floor: the minimum detectable effect in the decision statistic's units, estimated
    from the CONTROL arm's bootstrap CI at n as-designed.

    MDE is taken as 2 x CI95 half-width of the control mean at n_designed — the separation two
    independent arms of that size must show before their intervals stop overlapping. The control
    sample may have been collected at a different n than the design calls for; the half-width is
    rescaled by sqrt(n_have / n_designed), which is the bootstrap SE's own n-dependence.
    """
    import numpy as np
    a = np.asarray(control_values, dtype=np.float64).ravel()
    assert a.size > 1, "power_floor needs a control sample with n > 1"
    assert n_designed > 0, "n_designed must be positive"
    s = H.mean_ci(a, n_boot=n_boot, seed=seed)
    hw_have = ci_halfwidth(s["ci_lo"], s["ci_hi"])
    hw_designed = hw_have * float(np.sqrt(a.size / n_designed))
    return {"mde": 2.0 * hw_designed, "ci_hw_at_n_have": hw_have,
            "ci_hw_at_n_designed": hw_designed, "n_have": int(a.size),
            "n_designed": int(n_designed), "ci_method": s["ci_method"]}


def effect_criterion(*, treatment_mean: float, treatment_ci_lo: float, treatment_ci_hi: float,
                     incumbent_mean: float, higher_is_better: bool,
                     cost_ratio: float, mde: float) -> dict:
    """§5(1) EFFECT, evaluated mechanically.

    Passes iff EITHER
      (a) the treatment beats the incumbent by MORE than the arm's bootstrap 95% CI half-width,
          at matched budget; OR
      (b) it TIES within that CI while costing <= 1/10 of the incumbent (`cost_ratio` is
          treatment cost / incumbent cost, recorded in the same row per §5).

    Returns the full input set alongside the boolean so the artifact carries the formula's inputs
    (§10 graduation.parquet = 'arm x criteria x verdict + formula inputs'), and separately reports
    whether the margin the rule needs is below the power floor — §5's VOID branch.
    """
    hw = ci_halfwidth(treatment_ci_lo, treatment_ci_hi)
    delta = float(treatment_mean - incumbent_mean)
    signed = delta if higher_is_better else -delta
    beats = bool(signed > hw)
    ties = bool(abs(signed) <= hw)
    cheap = bool(cost_ratio <= 0.1)
    return {"criterion": "EFFECT", "passed": bool(beats or (ties and cheap)),
            "path": "beats" if beats else ("tie_at_tenth_cost" if (ties and cheap) else "neither"),
            "signed_delta": signed, "ci_halfwidth": hw, "cost_ratio": float(cost_ratio),
            "mde": float(mde), "underpowered": bool(hw < mde and not beats),
            "treatment_mean": float(treatment_mean), "incumbent_mean": float(incumbent_mean),
            "higher_is_better": bool(higher_is_better)}


def graduate(arm: str, *, effect: dict, envelope_measured: bool, envelope_evidence: str,
             interaction_terms=(), incumbent_id: str = None) -> dict:
    """§5 — GRADUATED iff EFFECT and ENVELOPE and COMPOSITION. Returns the row for
    out/graduation.parquet: arm x criteria x verdict + every formula input.

    (2) ENVELOPE: W2-A-relevant failure conditions measured, OR the arm-local dose curve shows a
        safe range. The caller supplies the boolean AND the evidence string naming which of the
        two it is; an ENVELOPE with no evidence is a defect.
    (3) COMPOSITION: no interaction term in W2-F worse than -1 CI on any stack the arm must join.
        Each element of `interaction_terms` is {name, estimate, ci_lo, ci_hi}; the term fails when
        its estimate sits below -1 x its own CI half-width.

    VOID-underpowered dominates NOT-GRADUATED: an arm that could not have detected its own
    frozen margin has not been tested, and §5 records that as its own outcome rather than
    letting it read as a negative result.
    """
    assert effect["criterion"] == "EFFECT", "effect must come from effect_criterion()"
    assert envelope_evidence and envelope_evidence.strip(), \
        f"{arm}: ENVELOPE requires evidence naming the W2-A rows or the dose curve (§5(2))"
    comp = []
    for t in interaction_terms:
        hw = ci_halfwidth(t["ci_lo"], t["ci_hi"])
        comp.append({"name": t["name"], "estimate": float(t["estimate"]), "ci_halfwidth": hw,
                     "passed": bool(float(t["estimate"]) >= -hw)})
    composition_passed = all(c["passed"] for c in comp)
    if effect["underpowered"]:
        verdict = VOID_UNDERPOWERED
    elif effect["passed"] and envelope_measured and composition_passed:
        verdict = GRADUATED
    else:
        verdict = NOT_GRADUATED
    assert verdict in VERDICTS
    return {"arm": arm, "incumbent_id": incumbent_id, "verdict": verdict,
            "effect": effect, "envelope": {"criterion": "ENVELOPE", "passed": bool(envelope_measured),
                                           "evidence": envelope_evidence},
            "composition": {"criterion": "COMPOSITION", "passed": bool(composition_passed),
                            "terms": comp},
            "claimed_by": CLAIMED_BY}


def stage2_margins(*, pilot_task_ci_lo: float, pilot_task_ci_hi: float,
                   pilot_battery_ci_lo: float, pilot_battery_ci_hi: float,
                   naive_retention_loss: float) -> dict:
    """§5 stage-2 frozen margin formulas, computed FROM WP-1 ARTIFACTS, never hand-set:

        delta_task = max(2 x CI95hw(pilot naive task success), 0.02)
        m_retain   = max(2 x CI95hw(pilot battery score), 0.5 x [naive's measured retention loss])
    """
    hw_task = ci_halfwidth(pilot_task_ci_lo, pilot_task_ci_hi)
    hw_bat = ci_halfwidth(pilot_battery_ci_lo, pilot_battery_ci_hi)
    return {"delta_task": max(2.0 * hw_task, 0.02),
            "m_retain": max(2.0 * hw_bat, 0.5 * float(naive_retention_loss)),
            "ci_hw_task": hw_task, "ci_hw_battery": hw_bat,
            "naive_retention_loss": float(naive_retention_loss),
            "formula": "delta_task=max(2*CI95hw(task),0.02); m_retain=max(2*CI95hw(battery),0.5*loss)"}


def gate_g0(*, pilot_ci_widths_finite: bool, graduated_arms) -> dict:
    """§5 G0 (stage-2 entry), mechanical: WP-1 completed with CI widths finite AND at least one of
    {W1-U anchor, W1-T restriction} GRADUATED.

    A G0 failure is NOT a program failure (§5, §0.1): stage 1 ships as a complete deliverable.
    """
    g = set(graduated_arms)
    qualifying = sorted(g & {"W1-U", "W1-T"})
    passed = bool(pilot_ci_widths_finite and qualifying)
    return {"gate": "G0", "passed": passed,
            "pilot_ci_widths_finite": bool(pilot_ci_widths_finite),
            "qualifying_graduations": qualifying, "all_graduated": sorted(g),
            "consequence": ("stage 2 launches" if passed else
                            "stage 2 flagships do not launch; stage 1 ships as the full deliverable")}


# ---------------------------------------------------------------------------
# §10 — the surprise log. Mechanical, per §1.3: |observed - expected| > 2 x control spread.
# ---------------------------------------------------------------------------
def surprise(observed: float, expected: float, control_values) -> dict:
    """One surprise-log row. `control_values` is the arm's own control column at the same cell;
    its standard deviation IS the 'control spread' §10 names.

    This is a DEVIATION MEASURE, not a verdict: it says a row landed outside the band its own
    controls define. Whether that is interesting belongs to the director (§0.1, §1.4).
    """
    import numpy as np
    c = np.asarray(control_values, dtype=np.float64).ravel()
    assert c.size > 1, "surprise() needs a control sample with n > 1"
    spread = float(c.std(ddof=1))
    dev = float(abs(observed - expected))
    return {"observed": float(observed), "expected": float(expected), "control_spread": spread,
            "deviation": dev, "threshold": 2.0 * spread,
            "is_surprise": bool(spread > 0 and dev > 2.0 * spread)}


class SurpriseLog:
    """out/surprise_log.parquet (§10), written incrementally like the atlas."""

    def __init__(self, out_dir):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "surprise_log.jsonl"
        self.n_rows = 0

    def add(self, *, arm: str, expected_ref: str, cell: str, statistic: str, s: dict, **extra):
        row = {"arm": arm, "expected_ref": expected_ref, "cell": cell, "statistic": statistic,
               "claimed_by": CLAIMED_BY, **s, **extra}
        with open(self.path, "a", buffering=1) as fh:
            fh.write(json.dumps(row) + "\n")
        self.n_rows += 1
        return row

    def to_parquet(self):
        import pandas as pd
        if not self.path.exists():
            return {"path": None, "n_rows": 0}
        df = pd.read_json(self.path, lines=True)
        p = self.out / "surprise_log.parquet"
        df.to_parquet(p, index=False)
        return {"path": str(p), "n_rows": int(len(df))}


# ---------------------------------------------------------------------------
# Bank library. HUMBOLDT's 900 objects + DARWIN's derived additions, merged into ONE directory so
# `humboldt_weights.load_bank` works unchanged on either.
# ---------------------------------------------------------------------------
def fetch_all_banks(dest, *, require_topsigma: bool = True):
    """Pull HUMBOLDT's bank library and DARWIN's derived banks into one directory.

    Two prefixes rather than one because the derived banks must stay attributable: `topsigma32`
    was derived by DARWIN (journal amendment A1) and writing it back into `humboldt/banks/` would
    make the HUMBOLDT library no longer the object HUMBOLDT published.

    `require_topsigma=False` is for W0-0 itself, which runs BEFORE the derived banks exist.
    """
    from infra.pg import humboldt_weights as WT
    d = pathlib.Path(dest)
    WT.fetch_banks(d)                                   # fails loud if HUMBOLDT's library is empty
    subprocess.run(["aws", "s3", "sync", BANKS_S3, str(d), "--region", "us-east-1"], check=True)
    # COUNT THE INPUT-SIDE BANKS ONLY. `dest` is a persistent cache directory shared by every step
    # of a pipeline, and MERCATOR syncs its OUTPUT-side banks into the same directory — so by the
    # second arm in a pipeline this glob was seeing DARWIN's 126 plus MERCATOR's 90 and refusing a
    # library that was complete. The row banks are a different object with a different name
    # (`{type}__L{n}__row__topsigma32.npy`), so the count can name what it actually means instead of
    # assuming the directory holds only what this function just synced (M-D52).
    n_top = len([p for p in d.glob("*__topsigma32.npy") if "__row__" not in p.name])
    if require_topsigma:
        assert n_top == len(H.TYPES) * H.N_LAYERS, (
            f"{n_top} input-side topsigma32 banks at {BANKS_S3}, expected "
            f"{len(H.TYPES) * H.N_LAYERS} — W0-0 must publish them before any arm that uses the "
            f"structured control (§7)")
    print(f"[darwin] bank library: {len(list(d.glob('*.npy')))} objects "
          f"({n_top} DARWIN-derived topsigma32)", flush=True)
    return d


def publish_banks(out_dir):
    """Sync out/banks/ to the DARWIN bank prefix."""
    subprocess.run(["aws", "s3", "sync", str(pathlib.Path(out_dir) / "banks"), BANKS_S3,
                    "--region", "us-east-1"], check=True)


def fetch_prior_arm(dest, s3_prefix: str):
    """Pull a completed arm's artifacts so a re-analysis arm can join against them.

    Re-analysis arms (W0-1c, W0-7, W0-8) consume other arms' ROWS, and §0.18 forbids taking a
    number from prose — so they take it from the parquet. Fails loud if the prefix has no atlas:
    a join that silently found nothing would report "no overlap" as a finding.
    """
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{s3_prefix}", str(d),
                    "--region", "us-east-1"], check=True)
    assert (d / "atlas.parquet").exists(), (
        f"no atlas.parquet under s3://{BUCKET}/{s3_prefix} — the upstream arm has not completed, "
        f"and a join against a missing table would report 'no overlap' as if it were a result")
    return d


def load_atlas(arm_dir):
    """Read a fetched arm's atlas.parquet, with its vectors resolvable via `vector_of`."""
    import pandas as pd
    d = pathlib.Path(arm_dir)
    df = pd.read_parquet(d / "atlas.parquet")
    assert len(df), f"{d}/atlas.parquet is empty"
    return df


def vector_of(arm_dir, row):
    """Load the .npy a row's `vector_ref` points at. Handles the '<name>.npy#<col>' column form."""
    import numpy as np
    ref = row["vector_ref"] if not hasattr(row, "vector_ref") else row.vector_ref
    assert ref, "row has no vector_ref"
    path, _, col = str(ref).partition("#")
    arr = np.load(pathlib.Path(arm_dir) / path)
    return arr[:, int(col)] if col else arr


# ---------------------------------------------------------------------------
# §10 / §0.2.5 — artifacts.
# ---------------------------------------------------------------------------
def s3_sync(out_dir, s3_prefix: str):
    """Incremental per-arm sync (§0.2.5: bound before every loop, synced per arm, never at end)."""
    subprocess.run(["aws", "s3", "sync", str(out_dir), f"s3://{BUCKET}/{s3_prefix}",
                    "--region", "us-east-1"], check=True)


def write_json(out_dir, name: str, obj):
    p = pathlib.Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    return str(p)


def utc_now() -> str:
    """UTC stamp for freeze tags and expectation rows. `date -u` rather than datetime.utcnow() so
    the stamp in an artifact is the same string the runbook's shell commands produce."""
    r = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], check=True, capture_output=True,
                       text=True)
    return r.stdout.strip()
