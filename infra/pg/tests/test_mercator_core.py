"""Weight-free unit tests for the MERCATOR recording + gating library (§8 step 3).

Run: PYTHONPATH=. python3 infra/pg/tests/test_mercator_core.py

Each test states the property the code must have and the observation that would falsify it. The
gating functions get adversarial cases, not just happy paths: a gate that cannot fail is the one
defect class §0.1 names explicitly.
"""
import json
import pathlib
import tempfile

import numpy as np

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M


def _atlas(td, **kw):
    """An atlas over a fresh temp dir. The expectation is logged once per (dir, arm) — re-logging
    is refused by the protocol itself, which is the behaviour test_mercator_core relies on."""
    arm = kw.get("arm", "M-B1")
    e = M.Expectations(pathlib.Path(td) / "exp.jsonl")
    rid = (e.by_arm(arm)[-1]["id"] if e.by_arm(arm)
           else e.log(arm, refuter="r", direction="d", utc="2026-08-14T00:00:00Z"))
    defaults = dict(arm="M-B1", run_id="t", register="ATLAS", wave="W0", expected_ref=rid,
                    queue_item="B1")
    defaults.update(kw)
    return M.Atlas(td, **defaults)


# ---------------------------------------------------------------------------
# §3 schema: DARWIN's columns + queue_item, and nothing else.
# ---------------------------------------------------------------------------
def test_schema_is_darwin_plus_queue_item():
    assert set(M.ATLAS_COLUMNS) - set(D.ATLAS_COLUMNS) == {"queue_item"}
    assert set(D.ATLAS_COLUMNS) - set(M.ATLAS_COLUMNS) == set()


def test_every_row_carries_the_queue_item():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.add("x", value=1.0)
        a.close()
        rows = [json.loads(l) for l in (pathlib.Path(td) / "atlas.jsonl").read_text().splitlines()]
        assert rows[0]["queue_item"] == "B1"
        assert rows[0]["claimed_by"] == M.CLAIMED_BY
        assert set(rows[0]) == set(M.ATLAS_COLUMNS)


def test_empty_queue_item_is_refused():
    with tempfile.TemporaryDirectory() as td:
        try:
            _atlas(td, queue_item="")
            raise SystemExit("an arm with no queue item must fail loud (§5)")
        except AssertionError as ex:
            assert "queue_item" in str(ex)


def test_stage2_register_and_darwin_only_wave_are_refused():
    with tempfile.TemporaryDirectory() as td:
        for bad in ({"register": "STAGE2"}, {"wave": "S2"}):
            try:
                _atlas(td, **bad)
                raise SystemExit(f"{bad} must be refused — MERCATOR has no stage 2 (§6)")
            except AssertionError:
                pass
        _atlas(td, wave="K").close()          # K is MERCATOR's own wave and must be accepted


def test_foreign_expected_ref_is_refused():
    with tempfile.TemporaryDirectory() as td:
        e = M.Expectations(pathlib.Path(td) / "exp.jsonl")
        rid = e.log("M-8", refuter="r", direction="d", utc="2026-08-14T00:00:00Z")
        try:
            M.Atlas(td, arm="M-B1", run_id="t", register="ATLAS", wave="W0",
                    expected_ref=rid, queue_item="B1")
            raise SystemExit("an arm may not borrow another arm's expectation id")
        except AssertionError as ex:
            assert "does not belong to arm" in str(ex)


# ---------------------------------------------------------------------------
# §3 dual-basis rule — the new grid-artifact guard.
# ---------------------------------------------------------------------------
def test_dual_basis_requires_both_bases():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.add("spikiness", value=1.0, basis=M.BASIS_COORD)
        try:
            a.check_dual_basis(["spikiness"])
            raise SystemExit("a coordinate claim in one basis only must fail loud (§3)")
        except AssertionError as ex:
            assert "random_ref" in str(ex)
        a.add("spikiness", value=0.9, basis=M.BASIS_RANDOM_REF)
        assert a.check_dual_basis(["spikiness"]) == ["spikiness"]


def test_dual_basis_treats_a_never_written_statistic_as_a_violation():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.add("spikiness", value=1.0, basis=M.BASIS_COORD)
        a.add("spikiness", value=0.9, basis=M.BASIS_RANDOM_REF)
        try:
            a.check_dual_basis(["spikiness", "leverage_max"])
            raise SystemExit("silent absence must read as a violation, not as a pass")
        except AssertionError as ex:
            assert "leverage_max" in str(ex)


def test_unknown_basis_tag_is_refused():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        try:
            a.add("x", value=1.0, basis="whatever")
            raise SystemExit("an unknown basis tag must fail loud")
        except AssertionError as ex:
            assert "unknown basis" in str(ex)


def test_random_reference_basis_is_orthogonal_and_seed_stable():
    Q1 = M.random_reference_basis(16, 7)
    Q2 = M.random_reference_basis(16, 7)
    Q3 = M.random_reference_basis(16, 8)
    assert Q1.shape == (16, 16)
    assert np.allclose(Q1, Q2) and not np.allclose(Q1, Q3)
    assert H.orthonormal_check(Q1) < 1e-8


# ---------------------------------------------------------------------------
# §0.2.9 readout-validity gate.
# ---------------------------------------------------------------------------
def test_readout_validity_fires_when_the_ft_does_not_beat_its_base():
    lo = M.readout_validity(ft_score=0.61, base_score=0.60, higher_is_better=True,
                            model_id="ft-vqav2", task="vqa")
    assert lo["passed"] and lo["outcome"] is None
    bad = M.readout_validity(ft_score=0.59, base_score=0.60, higher_is_better=True,
                             model_id="ft-vqav2", task="vqa")
    assert not bad["passed"] and bad["outcome"] == M.VOID_READOUT
    # a tie is NOT a pass at the default margin — an instrument that cannot see the training signal
    tie = M.readout_validity(ft_score=0.60, base_score=0.60, higher_is_better=True,
                             model_id="ft-vqav2", task="vqa")
    assert not tie["passed"]
    # lower-is-better readouts (perplexity) invert correctly
    ppl = M.readout_validity(ft_score=18.0, base_score=22.43, higher_is_better=False,
                             model_id="ft", task="ppl")
    assert ppl["passed"] and ppl["signed_improvement"] > 0


# ---------------------------------------------------------------------------
# §0.2.7 optimizer policy, including pre-freeze amendment A1.
# ---------------------------------------------------------------------------
def test_approximate_adam_is_refused_on_every_optimizer_mechanics_arm():
    for arm in sorted(M.OPTIMIZER_MECHANICS_ARMS):
        try:
            M.check_optimizer(arm, "adamw8bit")
            raise SystemExit(f"{arm} must refuse 8-bit Adam (§0.2.7)")
        except AssertionError as ex:
            assert "PROHIBITS" in str(ex)
        try:
            M.check_optimizer(arm, "adamw_torch")
            raise SystemExit(f"{arm} must require an explicit offload_exact_fp32")
        except AssertionError as ex:
            assert "offload_exact_fp32" in str(ex)
        ok = M.check_optimizer(arm, "adamw_torch", offload_exact_fp32=True)
        assert not ok["is_approximate"] and ok["optimizer_mechanics_arm"]


def test_amendment_a1_widened_the_set_and_never_narrowed_it():
    spec_named = {"M-A2", "M-A3", "M-A5", "M-11", "M-B2", "M-E4"}
    assert spec_named <= M.OPTIMIZER_MECHANICS_ARMS, "amendment A1 may only WIDEN the prohibition"
    assert {"M-Q", "M-3", "M-10"} <= M.OPTIMIZER_MECHANICS_ARMS
    # an arm outside the set is unaffected
    r = M.check_optimizer("M-C3", "adamw8bit")
    assert r["is_approximate"] and not r["optimizer_mechanics_arm"]


def test_forbidden_instance_families_are_refused():
    try:
        M.check_instance_types("p5.4xlarge,g6e.2xlarge")
        raise SystemExit("p5 must be refused (§0.2.7)")
    except AssertionError as ex:
        assert "p5.4xlarge" in str(ex)
    assert M.check_instance_types(M.GPU_TYPE_LADDER)[0] == "g6e.2xlarge"


# ---------------------------------------------------------------------------
# §6 graduation calculus is DARWIN's, reused unforked — and G0 is deliberately absent.
# ---------------------------------------------------------------------------
def _effect(**kw):
    """An effect row with every derived input supplied. Defaults are a tight control sample, so
    the MDE is small and a real difference reads as a real difference."""
    d = dict(treatment_mean=1.0, treatment_ci_lo=0.9, treatment_ci_hi=1.1, incumbent_mean=0.5,
             higher_is_better=True, control_values=np.linspace(0.499, 0.501, 40),
             n_designed=40, treatment_cost=1.0, incumbent_cost=1.0, cost_unit="L40S-h")
    d.update(kw)
    return M.effect_criterion(**d)


def test_graduation_formula_is_the_reused_one_and_g0_is_not_exported():
    g = M.graduate("M-C1", effect=_effect(), envelope_measured=True,
                   envelope_evidence="dose curve")
    assert g["verdict"] == M.GRADUATED
    assert not hasattr(M, "gate_g0"), "MERCATOR has no G0-style program gate (§6)"


def test_underpowered_dominates_not_graduated():
    # a wide control sample => a large MDE => an arm that could not have detected its own margin
    eff = _effect(treatment_mean=0.51, treatment_ci_lo=0.4, treatment_ci_hi=0.62,
                  control_values=np.linspace(0.0, 1.0, 40))
    g = M.graduate("M-C2", effect=eff, envelope_measured=False, envelope_evidence="none measured")
    assert g["verdict"] == M.VOID_UNDERPOWERED


# ---------------------------------------------------------------------------
# Wave-K fold: the critic's BLOCKING/MAJOR findings, made structurally impossible.
# ---------------------------------------------------------------------------
def test_f1_the_void_branch_is_reachable_and_mde_cannot_be_fabricated():
    """Critic F1: DARWIN's only call site passed mde=0.0, so `underpowered` was always False and
    the VOID-underpowered branch was unreachable for all 28 graduation rows."""
    # MERCATOR's signature has no `mde` and no `cost_ratio` to fabricate.
    import inspect
    sig = set(inspect.signature(M.effect_criterion).parameters)
    assert "mde" not in sig and "cost_ratio" not in sig
    assert {"control_values", "n_designed", "treatment_cost", "incumbent_cost"} <= sig
    # a degenerate control sample cannot produce mde=0 silently
    try:
        _effect(control_values=np.full(40, 0.5))
        raise SystemExit("a zero-spread control sample must fail loud, not yield mde=0")
    except AssertionError as ex:
        assert "power floor" in str(ex)
    # and the VOID branch actually fires on a real underpowered arm
    eff = _effect(treatment_mean=0.51, treatment_ci_lo=0.4, treatment_ci_hi=0.62,
                  control_values=np.linspace(0.0, 1.0, 40))
    assert eff["underpowered"] and eff["mde"] > 0


def test_f1_the_tie_at_a_tenth_cost_path_is_reachable():
    """The second half of F1: cost_ratio hardcoded to 1.0 killed §6's cheap-tie pass-path."""
    eff = _effect(treatment_mean=0.5, treatment_ci_lo=0.45, treatment_ci_hi=0.55,
                  incumbent_mean=0.5, treatment_cost=1.0, incumbent_cost=100.0)
    assert eff["path"] == "tie_at_tenth_cost" and eff["passed"]
    assert eff["cost"]["unit"] == "L40S-h"
    try:
        _effect(treatment_cost=0.0)
        raise SystemExit("an unmeasured cost must fail loud")
    except AssertionError as ex:
        assert "measured and positive" in str(ex)


def test_f1_a_hand_built_effect_dict_cannot_reach_graduate():
    eff = D.effect_criterion(treatment_mean=1.0, treatment_ci_lo=0.9, treatment_ci_hi=1.1,
                             incumbent_mean=0.5, higher_is_better=True, cost_ratio=1.0, mde=0.0)
    try:
        M.graduate("M-C1", effect=eff, envelope_measured=True, envelope_evidence="e")
        raise SystemExit("an effect dict with a fabricated mde must not reach a verdict")
    except AssertionError as ex:
        assert "power_floor" in str(ex)


def test_f14_a_global_interaction_term_cannot_be_borrowed_by_every_arm():
    """Critic F14: one W2-F term was applied to every GRAD arm's graduation cell."""
    term = {"name": "AxB", "estimate": -0.1, "ci_lo": -0.5, "ci_hi": 0.3,
            "applies_to": ["M-7"]}
    ok = M.graduate("M-7", effect=_effect(), envelope_measured=True, envelope_evidence="e",
                    interaction_terms=[term])
    assert ok["composition"]["passed"]
    try:
        M.graduate("M-C1", effect=_effect(), envelope_measured=True, envelope_evidence="e",
                   interaction_terms=[term])
        raise SystemExit("an interaction term measured for another arm must be refused")
    except AssertionError as ex:
        assert "does not cover this arm" in str(ex)
    try:
        M.graduate("M-7", effect=_effect(), envelope_measured=True, envelope_evidence="e",
                   interaction_terms=[{k: v for k, v in term.items() if k != "applies_to"}])
        raise SystemExit("an interaction term with no scope must be refused")
    except AssertionError as ex:
        assert "does not say which arms" in str(ex)


def test_f7_the_battery_is_checked_per_statistic_not_on_the_arm_union():
    """Critic F7: a full battery on a DIFFERENT statistic of the same arm masked a 4/8 gap."""
    full = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
            "haar32_s0", "haar32_s1", "haar32_s2")
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        for b in full:                                    # statistic A: complete
            a.add("energy_fraction", value=0.1, bank_id=b)
        for b in ("core32", "anti32", "complement32", "haar32_s0"):   # statistic B: W1-K's gap
            a.add("watermark_recovery_after_training", value=0.1, bank_id=b)
        # the arm-level union is complete, and DARWIN's check therefore passes...
        assert M.check_controls(set().union(*a.stat_banks.values()) - {None}, arm="M-B1")
        # ...while the per-statistic check catches it
        try:
            a.check_controls_per_statistic(["energy_fraction",
                                            "watermark_recovery_after_training"])
            raise SystemExit("the per-statistic battery gap must fail loud")
        except AssertionError as ex:
            assert "topsigma32" in str(ex) and "watermark_recovery_after_training" in str(ex)
        assert a.check_controls_per_statistic(["energy_fraction"])


def test_f10_a_readout_without_its_corpus_is_refused():
    """Critic F10: four unreconciled base_ppl anchors, corpus_id null on all 76,564 rows."""
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.add("base_ppl", value=22.43, corpus_id="wikitext103-val-61320tok")
        a.add("task_score", value=0.61)
        assert a.check_readout_corpus(["base_ppl"])
        try:
            a.check_readout_corpus(["base_ppl", "task_score"])
            raise SystemExit("a readout with a null corpus_id must fail loud")
        except AssertionError as ex:
            assert "task_score" in str(ex)


def test_outcome_literals_are_distinct_and_complete():
    assert len(set(M.OUTCOMES)) == len(M.OUTCOMES)
    assert set(M.VERDICTS) < set(M.OUTCOMES)
    assert {"VOID-readout", "VOID-INSTRUMENT", "COVERAGE-IMPOSSIBLE"} <= set(M.OUTCOMES)


# ---------------------------------------------------------------------------
# Provenance: MERCATOR must never inherit another program's stamp.
# ---------------------------------------------------------------------------
def test_provenance_refuses_a_foreign_stamp():
    p = M.PROVENANCE_FILE
    saved = p.read_text() if p.exists() else None
    try:
        p.write_text(json.dumps({"program": "darwin", "commit_sha": "deadbeef"}))
        try:
            M.provenance()
            raise SystemExit("a foreign program stamp must fail loud")
        except AssertionError as ex:
            assert "foreign stamp" in str(ex)
    finally:
        if saved is None:
            p.unlink()
        else:
            p.write_text(saved)


def test_controls_battery_is_inherited_intact():
    full = {"core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
            "haar32_s0", "haar32_s1", "haar32_s2"}
    assert M.check_controls(full, arm="M-B1") == sorted(full)
    try:
        M.check_controls(full - {"topsigma32"}, arm="M-B1")
        raise SystemExit("a missing control column is a defect, not an omission (§3)")
    except AssertionError as ex:
        assert "topsigma32" in str(ex)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
