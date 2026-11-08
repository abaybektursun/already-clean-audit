"""Unit tests for the pg launcher state machines.

Covers _wait_for_drainers (the function whose initial PR-52 version broke pg map by raising
on pending/running) + watch() exit codes 0/2/3/4/5. Uses botocore.stub.Stubber against the
module-level ec2/s3 clients; monkey-patches time.sleep + _now so tests are instant and
deterministic.

Run:
    cd /Users/abaybektursun/projects/interp_research-aws
    python -m pytest infra/aws_pg/tests/test_pg.py -v
"""
from __future__ import annotations

import datetime as dt

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from infra.aws_pg import pg


# ----- fixtures -----------------------------------------------------------------------------

@pytest.fixture
def stubs(monkeypatch):
    """Stubber-backed ec2/s3/scheduler clients on the pg module. time.sleep is a no-op so
    tests are instant. _now is monkeypatched to a fixed UTC moment so deadline math is
    deterministic."""
    ec2_stub = Stubber(pg.ec2)
    s3_stub = Stubber(pg.s3)
    sched_stub = Stubber(pg.sched)
    ec2_stub.activate()
    s3_stub.activate()
    sched_stub.activate()
    monkeypatch.setattr(pg.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(pg.time, "monotonic", _monotonic_counter())
    monkeypatch.setattr(pg, "_now", lambda: dt.datetime(2026, 5, 30, 12, 0, 0, tzinfo=dt.timezone.utc))
    try:
        yield ec2_stub, s3_stub, sched_stub
        ec2_stub.assert_no_pending_responses()
        s3_stub.assert_no_pending_responses()
        sched_stub.assert_no_pending_responses()
    finally:
        ec2_stub.deactivate()
        s3_stub.deactivate()
        sched_stub.deactivate()


def _monotonic_counter():
    """Each call advances by 1.0s so _wait_for_drainers's `time.monotonic() - start` makes
    forward progress against the timeout bound without real sleeping."""
    state = {"t": 0.0}

    def _m():
        state["t"] += 1.0
        return state["t"]
    return _m


def _ec2_running_response(instance_ids_states):
    """Build a describe_instances response: instance_ids_states = [(iid, state), ...]."""
    return {
        "Reservations": [{
            "Instances": [{
                "InstanceId": iid, "InstanceType": "g6e.8xlarge",
                "State": {"Name": state},
                "LaunchTime": dt.datetime(2026, 5, 30, 11, 0, 0, tzinfo=dt.timezone.utc),
                "Tags": [{"Key": "app", "Value": "pg-lineage"}],
            } for iid, state in instance_ids_states]
        }]
    }


def _ec2_empty_response():
    return {"Reservations": []}


# ----- _launch capacity fallback (try next type/AZ on any transient skip; never wait) --------

class TestLaunchFallback:
    """`pg run --type a,b` tries each type×AZ; first that launches wins; capacity/quota/unsupported
    attempts are skipped instantly (never waited on); all exhausted → fail loud."""

    def _runs_kwargs(self, monkeypatch):
        # stub out the side-effecting helpers _launch calls
        monkeypatch.setattr(pg, "package_code", lambda: "code/deadbeef.tgz")
        monkeypatch.setattr(pg, "render_user_data", lambda *a, **k: "#!/bin/bash\n")

    def test_first_type_first_az_wins(self, stubs, monkeypatch):
        ec2, _, _ = stubs
        self._runs_kwargs(monkeypatch)
        ec2.add_response("run_instances", {"Instances": [{"InstanceId": "i-win"}]})
        iid, t = pg._launch("m", "", "results/x", 600, ["g6e.8xlarge", "p5.4xlarge"], True)
        assert iid == "i-win" and t == "g6e.8xlarge"

    def test_falls_to_second_type_when_first_dry(self, stubs, monkeypatch):
        ec2, _, _ = stubs
        self._runs_kwargs(monkeypatch)
        # g6e.8xlarge: InsufficientInstanceCapacity in all 6 AZs + EC2-chosen (7 attempts)
        for _ in range(len(pg.C.SUBNETS) + 1):
            ec2.add_client_error("run_instances", service_error_code="InsufficientInstanceCapacity")
        # p5.4xlarge: first AZ wins
        ec2.add_response("run_instances", {"Instances": [{"InstanceId": "i-p5"}]})
        iid, t = pg._launch("m", "", "results/x", 600, ["g6e.8xlarge", "p5.4xlarge"], True)
        assert iid == "i-p5" and t == "p5.4xlarge"

    def test_quota_full_falls_through_no_wait(self, stubs, monkeypatch):
        """VcpuLimitExceeded (G bucket full / draining) must NOT wait — fall to the P-bucket type."""
        ec2, _, _ = stubs
        self._runs_kwargs(monkeypatch)
        for _ in range(len(pg.C.SUBNETS) + 1):
            ec2.add_client_error("run_instances", service_error_code="VcpuLimitExceeded")
        ec2.add_response("run_instances", {"Instances": [{"InstanceId": "i-p5"}]})
        iid, t = pg._launch("m", "", "results/x", 600, ["g6e.8xlarge", "p5.4xlarge"], True)
        assert iid == "i-p5" and t == "p5.4xlarge"

    def test_all_exhausted_raises(self, stubs, monkeypatch):
        ec2, _, _ = stubs
        self._runs_kwargs(monkeypatch)
        for _ in range(2 * (len(pg.C.SUBNETS) + 1)):  # both types, all AZs
            ec2.add_client_error("run_instances", service_error_code="InsufficientInstanceCapacity")
        with pytest.raises(RuntimeError, match="none of .* launchable"):
            pg._launch("m", "", "results/x", 600, ["g6e.8xlarge", "p5.4xlarge"], True)

    def test_non_transient_error_fails_loud(self, stubs, monkeypatch):
        """A real error (e.g. bad AMI) must NOT be swallowed as a skip."""
        ec2, _, _ = stubs
        self._runs_kwargs(monkeypatch)
        ec2.add_client_error("run_instances", service_error_code="InvalidAMIID.NotFound")
        with pytest.raises(ClientError):
            pg._launch("m", "", "results/x", 600, ["g6e.8xlarge"], True)


# ----- watch state machine -------------------------------------------------------------------

class TestWatch:
    """Each test drives watch() through exactly one path to its terminal exit code."""

    def test_status_present_exits_0(self, stubs):
        _, s3, _ = stubs
        # _read_status: GetObject succeeds → return body → exit 0
        s3.add_response("get_object",
                        {"Body": _fake_body(b"exit=0\n")},
                        expected_params={"Bucket": "pg-lineage-769063704696",
                                         "Key": "results/foo/STATUS"})
        assert pg.watch("results/foo", "i-xxx", deadline_utc=None, poll_s=0, heartbeat_min=99) == 0

    def test_terminated_without_status_exits_2(self, stubs):
        ec2, s3, _ = stubs
        # 1) STATUS absent (404)
        s3.add_client_error("get_object", service_error_code="NoSuchKey")
        # 2) describe_instances → terminated
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-xxx", "terminated")]))
        # 3) tail_log: head_object 404 → "no job.log" marker
        s3.add_client_error("head_object", service_error_code="404")
        assert pg.watch("results/foo", "i-xxx", deadline_utc=None, poll_s=0) == pg.WATCH_EXIT_TERMINATED

    def test_stopped_exits_3(self, stubs):
        ec2, s3, _ = stubs
        s3.add_client_error("get_object", service_error_code="NoSuchKey")
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-xxx", "stopping")]))
        assert pg.watch("results/foo", "i-xxx", deadline_utc=None, poll_s=0) == pg.WATCH_EXIT_STOPPED

    def test_deadline_passed_exits_4(self, stubs, monkeypatch):
        ec2, s3, _ = stubs
        # _now() is fixed at 2026-05-30 12:00:00. Pass a deadline in the past so the grace check fires.
        past = "2026-05-30T11:30:00Z"
        # Past-at-start sanity check (deadline already passed at watch() start) → function downgrades
        # to deadline=None and continues. To exercise the deadline-firing path INSIDE the loop, we
        # need deadline ≥ _now() at parse time but < _now() + grace later. _now is constant in this
        # fixture, so simpler: rig _now to advance past the deadline+grace mid-loop.
        moments = iter([
            dt.datetime(2026, 5, 30, 12, 0, 0, tzinfo=dt.timezone.utc),  # parse-time
            dt.datetime(2026, 5, 30, 13, 0, 0, tzinfo=dt.timezone.utc),  # mid-loop, past deadline+grace
        ])
        monkeypatch.setattr(pg, "_now", lambda: next(moments))
        s3.add_client_error("get_object", service_error_code="NoSuchKey")
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-xxx", "running")]))
        future = "2026-05-30T12:01:00Z"
        assert pg.watch("results/foo", "i-xxx", deadline_utc=future, poll_s=0) == pg.WATCH_EXIT_DEADLINE

    def test_invalid_instance_id_exits_5(self, stubs):
        _, s3, _ = stubs
        ec2_stub, _, _ = stubs
        s3.add_client_error("get_object", service_error_code="NoSuchKey")
        # Real boto3 behavior for a bogus iid: raises InvalidInstanceID.NotFound.
        ec2_stub.add_client_error("describe_instances", service_error_code="InvalidInstanceID.NotFound")
        assert pg.watch("results/foo", "i-bogus", deadline_utc=None,
                        poll_s=0) == pg.WATCH_EXIT_NOT_FOUND

    def test_past_deadline_at_parse_is_warned_not_errored(self, stubs, capsys):
        """If --deadline-utc is in the past at watch start, downgrade to deadline=None and
        warn — don't spuriously exit 4 on iteration 0."""
        _, s3, _ = stubs
        past = "2026-05-30T11:00:00Z"  # before our fixture's fixed _now
        # Loop will then poll once — STATUS present → exit 0.
        s3.add_response("get_object", {"Body": _fake_body(b"exit=0\n")})
        rc = pg.watch("results/foo", "i-xxx", deadline_utc=past, poll_s=0)
        assert rc == 0
        err = capsys.readouterr().err
        assert "already in the past" in err


# ----- helpers -------------------------------------------------------------------------------

class _FakeBody:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


def _fake_body(b):
    return _FakeBody(b)


# ----- resumable watch by prefix --------------------------------------------------------------

class TestResolveByPrefix:
    def test_resolves_single_match(self, stubs):
        ec2, _, _ = stubs
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-aaa", "running")]),
                         expected_params={"Filters": [
                             {"Name": "tag:app", "Values": ["pg-lineage"]},
                             {"Name": "tag:s3prefix", "Values": ["results/foo"]},
                             {"Name": "instance-state-name", "Values": ["pending", "running"]},
                         ]})
        assert pg._resolve_instance_by_prefix("results/foo") == "i-aaa"

    def test_no_match_raises(self, stubs):
        ec2, _, _ = stubs
        ec2.add_response("describe_instances", _ec2_empty_response())
        with pytest.raises(RuntimeError, match="no running pg-lineage instance"):
            pg._resolve_instance_by_prefix("results/missing")

    def test_ambiguous_raises(self, stubs):
        ec2, _, _ = stubs
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-aaa", "running"), ("i-bbb", "running")]))
        with pytest.raises(RuntimeError, match="ambiguous"):
            pg._resolve_instance_by_prefix("results/foo")


# ----- pg audit --------------------------------------------------------------------------------

class TestAudit:
    def test_clean(self, stubs, capsys):
        ec2, _, sch = stubs
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-aaa", "running")]))
        sch.add_response("list_schedules",
                         {"Schedules": [{"Name": "pg-reap-i-aaa"}]},
                         expected_params={"NamePrefix": "pg-reap-"})
        pg.audit()
        out = capsys.readouterr().out
        assert "1 running, 1 reapers, 0 orphans, 0 reaper-less" in out
        assert "clean." in out

    def test_orphan_reaper(self, stubs, capsys):
        ec2, _, sch = stubs
        ec2.add_response("describe_instances", _ec2_empty_response())
        sch.add_response("list_schedules",
                         {"Schedules": [{"Name": "pg-reap-i-zombie"}]})
        pg.audit()
        out = capsys.readouterr().out
        assert "orphan reaper" in out
        assert "i-zombie" in out

    def test_reaper_less_instance(self, stubs, capsys):
        """The DANGEROUS case — instance with no reaper armed."""
        ec2, _, sch = stubs
        ec2.add_response("describe_instances",
                         _ec2_running_response([("i-danger", "running")]))
        sch.add_response("list_schedules", {"Schedules": []})
        pg.audit()
        out = capsys.readouterr().out
        assert "REAPER-LESS" in out
        assert "i-danger" in out
