"""Weight-free unit tests for the DARWIN recording + gating library (§9 step 2).

Run: PYTHONPATH=. python3 -m pytest infra/pg/tests/test_darwin_core.py -q
     PYTHONPATH=. python3 infra/pg/tests/test_darwin_core.py     (no pytest needed)

Each test states the property the code must have and the observation that would falsify it.
The gating functions get adversarial cases, not just happy paths: a gate that cannot fail is the
one defect class §0.1 names explicitly.
"""
import json
import pathlib
import tempfile

import numpy as np

from infra.pg import darwin_core as D


# ---------------------------------------------------------------------------
# §11 expectation protocol.
# ---------------------------------------------------------------------------
def test_expectation_id_is_content_addressed_and_immutable():
    with tempfile.TemporaryDirectory() as td:
        e = D.Expectations(pathlib.Path(td) / "exp.jsonl")
        rid = e.log("W0-1a", refuter="twin overlap bands indistinguishable",
                    direction="core leverage overlaps outlier dims above twin band",
                    utc="2026-08-12T00:00:00Z")
        assert rid.startswith("W0-1a::") and len(rid.split("::")[1]) == 10
        # Same content ⇒ same id (content-addressed, so an edit cannot masquerade as the original).
        assert D.Expectations._rid("W0-1a", "twin overlap bands indistinguishable",
                                   "core leverage overlaps outlier dims above twin band") == rid
        # A second, different expectation for a logged arm is refused without supersede=.
        try:
            e.log("W0-1a", refuter="other", direction="other", utc="2026-08-12T00:01:00Z")
            raise SystemExit("editing an expectation must fail loud")
        except AssertionError as ex:
            assert "supersede" in str(ex)


def test_expectation_supersession_keeps_the_original_row():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "exp.jsonl"
        e = D.Expectations(p)
        first = e.log("W1-A", refuter="in-core burn <= its own twin burns",
                      direction="in-core burn exceeds twin burns", utc="2026-08-12T00:00:00Z")
        second = e.log("W1-A", refuter="in-core burn <= its own twin burns",
                       direction="revised direction", utc="2026-08-12T01:00:00Z",
                       supersede=first, reason="dose grid rescaled pre-freeze")
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        assert len(rows) == 2 and rows[0]["id"] == first          # original survives verbatim
        assert rows[1]["supersedes"] == first and rows[1]["reason"]
        assert e.require("W1-A") == second                        # arms carry the live one
        # Supersession without a reason is a defect.
        try:
            e.log("W1-A", refuter="x", direction="y", utc="z", supersede=second)
            raise SystemExit("supersession without a reason must fail loud")
        except AssertionError as ex:
            assert "reason" in str(ex)


def test_require_refuses_an_arm_that_never_pre_logged():
    with tempfile.TemporaryDirectory() as td:
        e = D.Expectations(pathlib.Path(td) / "exp.jsonl")
        try:
            e.require("W1-Z")
            raise SystemExit("an arm with no expectation must fail loud")
        except AssertionError as ex:
            assert "no pre-logged expectation" in str(ex)


def test_expectations_reload_from_disk():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "exp.jsonl"
        rid = D.Expectations(p).log("W0-2", refuter="r", direction="d", utc="u")
        assert D.Expectations(p).require("W0-2") == rid            # survives process restart


# ---------------------------------------------------------------------------
# §3 atlas schema.
# ---------------------------------------------------------------------------
def _atlas(td, **kw):
    kw.setdefault("register", "ATLAS")
    kw.setdefault("wave", "W0")
    kw.setdefault("expected_ref", "W0-1a::abcdef0123")
    return D.Atlas(td, arm="W0-1a", run_id="t", **kw)


def test_atlas_row_carries_all_seven_darwin_columns():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.add("stat", value=1.0, bank_id="core32", layer=3, type="q_proj")
        a.add_ci("sampled", [1.0, 2.0, 3.0, 4.0], bank_id="haar32_s0", dose=0.5,
                 incumbent_id="lora_r32")
        a.close()
        rows = [json.loads(l) for l in (pathlib.Path(td) / "atlas.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        for r in rows:
            assert set(r) == set(D.ATLAS_COLUMNS)
            assert r["register"] == "ATLAS" and r["wave"] == "W0"
            assert r["claimed_by"] == D.CLAIMED_BY
            assert r["expected_ref"] == "W0-1a::abcdef0123"
        assert rows[0]["dose"] is None and rows[0]["incumbent_id"] is None
        assert rows[1]["dose"] == 0.5 and rows[1]["incumbent_id"] == "lora_r32"
        assert rows[1]["n"] == 4 and rows[1]["ci_lo"] <= rows[1]["value"] <= rows[1]["ci_hi"]


def test_atlas_rejects_bad_register_wave_and_foreign_expected_ref():
    with tempfile.TemporaryDirectory() as td:
        for kw, needle in (({"register": "NOPE"}, "unknown register"),
                           ({"wave": "W9"}, "unknown wave"),
                           ({"expected_ref": "W9-Q::0000000000"}, "does not belong to arm")):
            try:
                _atlas(td, **kw)
                raise SystemExit(f"must fail loud on {kw}")
            except AssertionError as ex:
                assert needle in str(ex)


def test_atlas_value_xor_vector_and_nan_refusal():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        for kw in ({}, {"value": 1.0, "vector_ref": "v/x.npy"}):
            try:
                a.add("s", **kw)
                raise SystemExit("value/vector_ref must be exclusive-or")
            except AssertionError as ex:
                assert "exactly one of" in str(ex)
        try:
            a.add("s", value=float("nan"))
            raise SystemExit("NaN must fail loud")
        except AssertionError as ex:
            assert "NaN" in str(ex)
        a.close()


def test_atlas_parquet_roundtrip_preserves_column_order():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        ref = a.vector("W0-1a__lev__q_proj__L0", np.arange(5.0))
        a.add("leverage", vector_ref=ref, layer=0, type="q_proj", bank_id="core32")
        a.add("scalar", value=0.25, layer=0, type="q_proj", bank_id="topsigma32", dose=1.0)
        info = a.to_parquet()
        assert info["n_rows"] == 2 and info["n_vectors"] == 1
        import pandas as pd
        df = pd.read_parquet(info["path"])
        assert list(df.columns) == list(D.ATLAS_COLUMNS)
        assert df["dose"].tolist() == [None, 1.0] or np.isnan(df["dose"].iloc[0])
        assert set(df["claimed_by"]) == {D.CLAIMED_BY}


def test_vector_ref_collision_fails_loud():
    with tempfile.TemporaryDirectory() as td:
        a = _atlas(td)
        a.vector("dup", np.zeros(3))
        try:
            a.vector("dup", np.ones(3))
            raise SystemExit("a silently overwritten vector is unrecoverable data loss")
        except AssertionError as ex:
            assert "collision" in str(ex)
        a.close()


# ---------------------------------------------------------------------------
# §3 reporting-standard enforcement.
# ---------------------------------------------------------------------------
def test_dose_grid_requires_four_log_spaced_points():
    assert len(D.check_dose_grid([0.1, 0.3, 0.9, 2.7], arm="W1-A")) == 4      # ratio 3, exact
    for bad, needle in (([0.1, 1.0, 10.0], "requires >= 4"),
                        ([0.1, 0.2, 0.9, 2.7], "not log-spaced"),
                        ([1.0, 1.0, 2.0, 4.0], "duplicate"),
                        ([0.0, 1.0, 2.0, 3.0], "positive")):
        try:
            D.check_dose_grid(bad, arm="W1-A")
            raise SystemExit(f"{bad} must fail loud")
        except AssertionError as ex:
            assert needle in str(ex)
    # A genuinely linear knob (shot count) degrades to count + monotonicity, recorded not waived.
    assert D.check_dose_grid([0, 1, 4, 16], arm="W1-B", log_spaced=False) == [0.0, 1.0, 4.0, 16.0]


def test_controls_require_the_structured_alternative_not_haar_alone():
    full = {"core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
            "haar32_s0", "haar32_s1", "haar32_s2"}
    assert D.check_controls(full, arm="W1-K")
    try:                                    # Haar-only is the exact failure §3 legislates against
        D.check_controls(full - {"topsigma32"}, arm="W1-K")
        raise SystemExit("missing structured control must fail loud")
    except AssertionError as ex:
        assert "topsigma32" in str(ex)
    try:
        D.check_controls(full, arm="W1-T", incumbent="lora_r32")
        raise SystemExit("missing incumbent column must fail loud")
    except AssertionError as ex:
        assert "incumbent column" in str(ex)
    assert D.check_controls(full | {"lora_r32"}, arm="W1-T", incumbent="lora_r32")


# ---------------------------------------------------------------------------
# §5 graduation calculus. Both branches of every rule (feedback_one_sided_thresholds).
# ---------------------------------------------------------------------------
def test_effect_beats_ties_and_neither():
    common = dict(treatment_ci_lo=0.58, treatment_ci_hi=0.62, higher_is_better=True,
                  cost_ratio=1.0, mde=0.01)                       # CI half-width = 0.02
    beats = D.effect_criterion(treatment_mean=0.60, incumbent_mean=0.55, **common)
    assert beats["passed"] and beats["path"] == "beats"
    near = D.effect_criterion(treatment_mean=0.60, incumbent_mean=0.59, **common)
    assert not near["passed"] and near["path"] == "neither"       # tie at full cost is not a pass
    cheap = D.effect_criterion(treatment_mean=0.60, incumbent_mean=0.59,
                               **{**common, "cost_ratio": 0.05})
    assert cheap["passed"] and cheap["path"] == "tie_at_tenth_cost"
    worse = D.effect_criterion(treatment_mean=0.50, incumbent_mean=0.55, **common)
    assert not worse["passed"] and worse["signed_delta"] < 0


def test_effect_respects_lower_is_better():
    """A ppl-style readout must not be silently scored as if bigger were better."""
    kw = dict(treatment_ci_lo=9.9, treatment_ci_hi=10.1, cost_ratio=1.0, mde=0.01)
    good = D.effect_criterion(treatment_mean=10.0, incumbent_mean=12.0,
                              higher_is_better=False, **kw)
    bad = D.effect_criterion(treatment_mean=10.0, incumbent_mean=12.0,
                             higher_is_better=True, **kw)
    assert good["passed"] and not bad["passed"]


def test_underpowered_dominates_not_graduated():
    """An arm that could not detect its own margin is VOID, never a negative result (§5)."""
    eff = D.effect_criterion(treatment_mean=0.60, incumbent_mean=0.595, treatment_ci_lo=0.55,
                             treatment_ci_hi=0.65, higher_is_better=True, cost_ratio=1.0,
                             mde=0.20)                            # hw 0.05 < mde 0.20
    assert eff["underpowered"] and not eff["passed"]
    g = D.graduate("W1-T", effect=eff, envelope_measured=True, envelope_evidence="dose curve")
    assert g["verdict"] == D.VOID_UNDERPOWERED
    # A genuine beat is NOT voided merely because the floor is high.
    eff2 = D.effect_criterion(treatment_mean=0.90, incumbent_mean=0.50, treatment_ci_lo=0.85,
                              treatment_ci_hi=0.95, higher_is_better=True, cost_ratio=1.0,
                              mde=0.20)
    assert not eff2["underpowered"] and eff2["passed"]


def test_graduate_requires_all_three_criteria():
    eff = D.effect_criterion(treatment_mean=0.70, incumbent_mean=0.60, treatment_ci_lo=0.68,
                             treatment_ci_hi=0.72, higher_is_better=True, cost_ratio=1.0,
                             mde=0.01)
    ok = D.graduate("W1-U", effect=eff, envelope_measured=True,
                    envelope_evidence="W2-A rows L1-L4", interaction_terms=[
                        {"name": "anchor x restriction", "estimate": -0.01,
                         "ci_lo": -0.05, "ci_hi": 0.05}])         # -0.01 >= -0.05 ⇒ passes
    assert ok["verdict"] == D.GRADUATED
    no_env = D.graduate("W1-U", effect=eff, envelope_measured=False, envelope_evidence="none run")
    assert no_env["verdict"] == D.NOT_GRADUATED
    bad_comp = D.graduate("W1-U", effect=eff, envelope_measured=True, envelope_evidence="e",
                          interaction_terms=[{"name": "x", "estimate": -0.30,
                                              "ci_lo": -0.05, "ci_hi": 0.05}])
    assert bad_comp["verdict"] == D.NOT_GRADUATED
    assert not bad_comp["composition"]["terms"][0]["passed"]
    try:
        D.graduate("W1-U", effect=eff, envelope_measured=True, envelope_evidence="  ")
        raise SystemExit("ENVELOPE without evidence must fail loud")
    except AssertionError as ex:
        assert "evidence" in str(ex)


def test_stage2_margins_use_the_floor_when_the_pilot_is_precise():
    """delta_task's 0.02 floor and m_retain's half-the-loss floor must both actually bind."""
    # Exact equality is the wrong assertion on a float built from subtraction: (0.51-0.49)/2*2
    # is 0.020000000000000018, not 0.02. Compare with a tolerance far tighter than any margin
    # the rule could act on.
    m = D.stage2_margins(pilot_task_ci_lo=0.40, pilot_task_ci_hi=0.42,
                         pilot_battery_ci_lo=0.10, pilot_battery_ci_hi=0.11,
                         naive_retention_loss=4.0)
    assert abs(m["delta_task"] - 0.02) < 1e-9                     # 2*0.01 = 0.02, floor ties
    assert abs(m["m_retain"] - 2.0) < 1e-9                        # 0.5*4.0 beats 2*0.005
    # The floor must actually BIND, not merely tie: a pilot 10x more precise still yields 0.02.
    tight = D.stage2_margins(pilot_task_ci_lo=0.400, pilot_task_ci_hi=0.402,
                             pilot_battery_ci_lo=0.10, pilot_battery_ci_hi=0.101,
                             naive_retention_loss=0.0)
    assert abs(tight["delta_task"] - 0.02) < 1e-9                 # floor, not 2*0.001
    assert abs(tight["m_retain"] - 0.001) < 1e-9                  # no loss ⇒ CI term wins
    wide = D.stage2_margins(pilot_task_ci_lo=0.30, pilot_task_ci_hi=0.70,
                            pilot_battery_ci_lo=0.0, pilot_battery_ci_hi=8.0,
                            naive_retention_loss=1.0)
    assert abs(wide["delta_task"] - 0.40) < 1e-9                  # a vague pilot ⇒ wide margins
    assert abs(wide["m_retain"] - 8.0) < 1e-9


def test_gate_g0_both_branches():
    assert D.gate_g0(pilot_ci_widths_finite=True, graduated_arms=["W1-U"])["passed"]
    assert D.gate_g0(pilot_ci_widths_finite=True, graduated_arms=["W1-T", "W1-K"])["passed"]
    # A graduation that is not one of the two named arms does not open G0.
    assert not D.gate_g0(pilot_ci_widths_finite=True, graduated_arms=["W1-K"])["passed"]
    assert not D.gate_g0(pilot_ci_widths_finite=False, graduated_arms=["W1-U"])["passed"]
    assert "stage 1 ships" in D.gate_g0(pilot_ci_widths_finite=False, graduated_arms=[])["consequence"]


def test_power_floor_scales_with_designed_n():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 400)
    at400 = D.power_floor(x, n_designed=400)
    at1600 = D.power_floor(x, n_designed=1600)
    assert at400["mde"] > at1600["mde"]                            # 4x n ⇒ ~half the MDE
    assert abs(at400["mde"] / at1600["mde"] - 2.0) < 0.05


# ---------------------------------------------------------------------------
# §10 surprise log.
# ---------------------------------------------------------------------------
def test_surprise_is_relative_to_the_arms_own_control_spread():
    ctrl = [0.10, 0.11, 0.09, 0.10, 0.12]                          # sd ~ 0.011
    assert D.surprise(observed=0.50, expected=0.10, control_values=ctrl)["is_surprise"]
    assert not D.surprise(observed=0.11, expected=0.10, control_values=ctrl)["is_surprise"]
    # A degenerate (zero-spread) control cannot manufacture a surprise.
    assert not D.surprise(observed=5.0, expected=0.0, control_values=[1.0, 1.0, 1.0])["is_surprise"]


def test_surprise_log_writes_parquet():
    with tempfile.TemporaryDirectory() as td:
        s = D.SurpriseLog(td)
        s.add(arm="W0-1a", expected_ref="W0-1a::0123456789", cell="q_proj/L0",
              statistic="jaccard", s=D.surprise(0.9, 0.1, [0.1, 0.12, 0.09]))
        info = s.to_parquet()
        assert info["n_rows"] == 1
        import pandas as pd
        df = pd.read_parquet(info["path"])
        assert df["is_surprise"].iloc[0] and df["claimed_by"].iloc[0] == D.CLAIMED_BY


def test_bank_battery_includes_topsigma_for_every_type():
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert "topsigma32" in D.bank_ids_for(t)
        assert "core32" in D.bank_ids_for(t) and "anti32" in D.bank_ids_for(t)


# ---------------------------------------------------------------------------
# §0.2.7 compute policy (spec revised 2026-08-12).
# ---------------------------------------------------------------------------
def test_forbidden_instance_families_are_refused():
    assert D.check_instance_types("g6e.2xlarge") == ["g6e.2xlarge"]
    assert D.check_instance_types(["g6e.8xlarge", "g6e.4xlarge"]) == ["g6e.8xlarge", "g6e.4xlarge"]
    for bad in ("p5.4xlarge", "g6e.2xlarge,p5.4xlarge", "p4de.24xlarge"):
        try:
            D.check_instance_types(bad)
            raise SystemExit(f"{bad} must be refused")
        except AssertionError as ex:
            assert "excluded by" in str(ex)


def test_approximate_adam_is_prohibited_on_optimizer_mechanics_arms():
    """The second-moment representation is the object under study on these arms."""
    for arm in ("W1-Q", "W1-S", "W1-T", "W1-U", "W2-A", "W2-B", "S2-A", "S2-B"):
        assert arm in D.OPTIMIZER_MECHANICS_ARMS
        try:
            D.check_optimizer(arm, "adamw8bit", offload_exact_fp32=True)
            raise SystemExit(f"{arm} must refuse 8-bit Adam")
        except AssertionError as ex:
            assert "PROHIBITS" in str(ex)
        # Exact Adam is allowed, but the offload configuration must be stated, not defaulted.
        try:
            D.check_optimizer(arm, "adamw_torch")
            raise SystemExit(f"{arm} must require an explicit offload_exact_fp32")
        except AssertionError as ex:
            assert "offload_exact_fp32" in str(ex)
        ok = D.check_optimizer(arm, "adamw_torch", offload_exact_fp32=True)
        assert not ok["is_approximate"] and ok["optimizer_mechanics_arm"]
    # An arm whose science does NOT touch optimizer mechanics may use an approximate optimizer.
    r = D.check_optimizer("W1-D", "adamw8bit")
    assert r["is_approximate"] and not r["optimizer_mechanics_arm"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
