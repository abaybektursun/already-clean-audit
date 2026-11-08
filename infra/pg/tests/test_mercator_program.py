"""Weight-free unit tests for the MERCATOR program registry (§5 coverage, §10 budget, §8 order).

Run: PYTHONPATH=. python3 infra/pg/tests/test_mercator_program.py

These are the tests that make three named defect classes impossible rather than discouraged:
silent coverage shrinkage, an unbudgeted wave, and an out-of-order launch.
"""
import pathlib
import tempfile

from infra.pg import mercator_program as P


# ---------------------------------------------------------------------------
# §5 — the no-stone-unturned contract.
# ---------------------------------------------------------------------------
def test_every_queue_item_has_a_disposition():
    assert P.uncovered() == [], f"uncovered queue items: {P.uncovered()}"
    rows = P.coverage_table()
    assert len(rows) == len(P.QUEUE_ITEMS) == 53
    assert {r["queue_item"] for r in rows} == set(P.QUEUE_ITEMS)


def test_the_five_standing_items_are_all_present_including_w0_5():
    standing = {"S-W1-G", "S-W1-Q", "S-W1-AB", "S-W2-A", "S-W0-5"}
    assert standing <= set(P.QUEUE_ITEMS)
    for s in standing:
        assert P.arms_for(s), f"standing item {s} has no arm — this is the §5 hole"
    # S-W0-5 (the VLA census) is the one the spec left without an arm; amendment A3 gave it M-V.
    assert P.arms_for("S-W0-5") == ["M-V"]


def test_every_arm_maps_to_a_queue_item():
    for arm, meta in P.ARMS.items():
        assert meta["queue_item"] == "gate" or meta["queue_item"] in P.QUEUE_ITEMS, \
            f"{arm} has queue_item {meta['queue_item']!r}, which is not in the queue"


def test_item_nine_is_covered_by_two_arms_and_that_is_recorded():
    assert P.arms_for("9") == ["M-9a", "M-9b"]
    row = next(r for r in P.coverage_table() if r["queue_item"] == "9")
    assert row["disposition"] == P.ARM and len(row["arms"]) == 2


def test_non_arm_dispositions_are_named_not_silent():
    d = {r["queue_item"]: r["disposition"] for r in P.coverage_table()}
    assert d["A1"] == P.RIDES_INSIDE and d["C5"] == P.RIDES_INSIDE and d["E1"] == P.RIDES_INSIDE
    assert d["E5"] == P.PARKED
    assert d["D6"] == P.CLOSEOUT
    assert d["D7"] == P.DIRECTOR_DECISION and d["E6"] == P.DIRECTOR_DECISION


def test_removing_a_disposition_breaks_the_table():
    saved = P.NON_ARM_DISPOSITION.pop("E5")
    try:
        P.coverage_table()
        raise SystemExit("an undisposed queue item must fail loud — that IS the silent shrinkage")
    except AssertionError as ex:
        assert "E5" in str(ex)
    finally:
        P.NON_ARM_DISPOSITION["E5"] = saved
    P.coverage_table()


def test_every_arm_has_a_refuter_or_says_why_not():
    for arm, m in P.ARMS.items():
        r = m["refuter"].strip()
        assert r, f"{arm} has no REFUTER line"
        if r.startswith("n/a"):
            assert "(" in r, f"{arm}: an n/a REFUTER must name what it delivers instead"


# ---------------------------------------------------------------------------
# §10 — the budget ledger.
# ---------------------------------------------------------------------------
def test_wave_cap_audit_reports_the_oversubscription_rather_than_hiding_it():
    a = {r["wave"]: r for r in P.wave_cap_audit()}
    assert a["W1"]["sum_arm_caps_l40s_h"] == 118.0 and a["W1"]["wave_cap_l40s_h"] == 110.0
    assert a["W1"]["oversubscribed_by_l40s_h"] == 8.0
    assert a["W2"]["oversubscribed_by_l40s_h"] == 13.0
    assert a["W0"]["oversubscribed_by_l40s_h"] == 0.0
    assert a["WP"]["oversubscribed_by_l40s_h"] == 0.0


def test_total_cap_matches_the_spec():
    assert P.TOTAL_CAP_L40S_H == 329.0            # 4 + 110 + 130 + 60 + 25 ("~330" in §10)
    assert P.CURIOSITY_RESERVE_L40S_H == 25.0


def test_cap_group_is_shared_not_per_arm():
    for a in ("M-B4", "M-B5", "M-B6"):
        assert P.arm_cap(a) == 16.0
    # and the wave audit charges the group ONCE
    w2 = next(r for r in P.wave_cap_audit() if r["wave"] == "W2")
    assert w2["sum_arm_caps_l40s_h"] == 40 + 45 + 20 + 12 + 16 + 10


def test_budget_refuses_a_launch_that_does_not_fit():
    with tempfile.TemporaryDirectory() as td:
        b = P.WaveBudget(pathlib.Path(td) / "budget.json")
        b.check("M-12", 8.0)
        b.spend("M-12", 7.5, instance_type="g6e.2xlarge", run_id="r1")
        assert b.spent("W1") == 7.5 and b.remaining("W1") == 102.5
        b.spend("M-G", 100.0, instance_type="g6e.2xlarge", run_id="r2")
        try:
            b.check("M-Q", 12.0)
            raise SystemExit("a launch past the wave cap must fail loud (§10)")
        except AssertionError as ex:
            assert "degradation order" in str(ex)


def test_budget_refuses_an_arm_over_its_own_cap():
    with tempfile.TemporaryDirectory() as td:
        b = P.WaveBudget(pathlib.Path(td) / "budget.json")
        b.spend("M-A4", 3.0, instance_type="g6e.2xlarge", run_id="r1")   # cap is 3.0
        try:
            b.check("M-A4", 1.0)
            raise SystemExit("an arm past its own §4 cap must fail loud")
        except AssertionError as ex:
            assert "exceeds its §4 cap" in str(ex)


def test_curiosity_hours_do_not_consume_the_wave_budget():
    with tempfile.TemporaryDirectory() as td:
        b = P.WaveBudget(pathlib.Path(td) / "budget.json")
        b.spend("M-12", 5.0, instance_type="g6e.2xlarge", run_id="r1", curiosity=True)
        assert b.remaining("W1") == 110.0        # §10: the reserve is a separate pot
        assert b.spent("W1") == 5.0
        assert b.curiosity_remaining() == 20.0


def test_curiosity_reserve_cannot_rescue_a_grad_arm_or_overrun():
    with tempfile.TemporaryDirectory() as td:
        b = P.WaveBudget(pathlib.Path(td) / "budget.json")
        try:
            b.spend("M-C1", 1.0, instance_type="g6e.2xlarge", run_id="r1", curiosity=True)
            raise SystemExit("§10 forbids spending the reserve on a Register-B arm")
        except AssertionError as ex:
            assert "never to rescue a GRAD arm" in str(ex)
        b.spend("M-12", 24.0, instance_type="g6e.2xlarge", run_id="r2", curiosity=True)
        try:
            b.spend("M-12", 2.0, instance_type="g6e.2xlarge", run_id="r3", curiosity=True)
            raise SystemExit("the 25 h reserve must bind")
        except AssertionError as ex:
            assert "reserve of 25.0" in str(ex)


def test_budget_survives_a_session_boundary():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "budget.json"
        P.WaveBudget(p).spend("M-12", 4.0, instance_type="g6e.2xlarge", run_id="r1")
        assert P.WaveBudget(p).spent("W1") == 4.0


# ---------------------------------------------------------------------------
# §8 — execution order.
# ---------------------------------------------------------------------------
def test_wave_one_runs_m12_first():
    try:
        P.check_deps("M-G", completed=set())
        raise SystemExit("§8 runs M-12 first in W1")
    except AssertionError as ex:
        assert "M-12" in str(ex)
    P.check_deps("M-G", completed={"M-12"})
    P.check_deps("M-12", completed=set())         # the first arm itself is not blocked


def test_named_dependency_edges_are_enforced():
    for arm, need in (("M-11", "M-Q"), ("M-C9", "M-G"), ("M-7", "M-E3"), ("M-6", "M-Q")):
        try:
            P.check_deps(arm, completed={"M-12"})
            raise SystemExit(f"{arm} must wait for {need}")
        except AssertionError as ex:
            assert need in str(ex)


def test_render_coverage_md_contains_every_item():
    md = P.render_coverage_md()
    for item in P.QUEUE_ITEMS:
        assert f"| {item} |" in md, f"{item} missing from the rendered coverage table"
    assert "Wave budget audit" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
