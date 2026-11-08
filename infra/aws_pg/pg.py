#!/usr/bin/env python3
"""pg — stateless launcher for the PaliGemma lineage study on AWS.

One job = one ephemeral p5.4xlarge (1x H100-80GB) born from the env AMI + the weights
snapshot, which runs the job, streams results to S3, and self-terminates. Correctness of
the "no instance outlives its job" guarantee is an AWS-side one-shot EventBridge schedule
(armed per instance, self-deleting), NOT the in-OS shutdown — so a wedged box still dies.

State lives entirely in EC2 tags + S3; this launcher holds none and can be closed the
instant `run` returns. fuelos / us-east-1 are hardcoded (see config.py).

`_arm_deadline` is idempotent: if `pg-reap-<instance>` already exists (ConflictException) the
existing schedule's expression is returned and the instance is left alone; only OTHER arming
failures terminate the just-launched instance (so it can never orphan without a reaper).

Subcommands: run | map | ps | kill | logs | collect
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import boto3
from botocore.exceptions import ClientError

from infra.aws_pg import config as C

_SESSION = boto3.Session(profile_name=C.PROFILE, region_name=C.REGION)
ec2 = _SESSION.client("ec2")
s3 = _SESSION.client("s3")
sched = _SESSION.client("scheduler")

REPO = pathlib.Path(__file__).resolve().parents[2]  # worktree root (.../infra/aws_pg/pg.py -> root)
RUN_JOB_SH = "infra/aws_pg/run_job.sh"

# Reaper grace: reaper deadline = job budget + this. watch() applies the same grace to its
# deadline+ check (single source of truth — the two were drifting apart in PR #52).
_DEADLINE_GRACE_S = 600

# An attempt (type, AZ) is SKIPPED — never waited on — when the type has no capacity in that AZ
# (InsufficientInstanceCapacity), the AZ doesn't offer the type (Unsupported), or our vCPU bucket
# is full (VcpuLimitExceeded, e.g. a prior same-bucket instance still draining). The launcher
# falls through to the next AZ, then the next instance TYPE in the preference list — which can be
# a DIFFERENT quota bucket (g6e=G, p5=P), so a busy/dry G bucket never blocks a P-bucket launch.
# "Don't wait for an unavailable machine; launch on an available one." Anything else fails loud.
_LAUNCH_SKIP = {"InsufficientInstanceCapacity", "Unsupported", "VcpuLimitExceeded"}


def _now():
    return dt.datetime.now(dt.timezone.utc)


# ---- code packaging (handles uncommitted working-tree code) -------------------------------
def package_code():
    """tar the working-tree infra/ (+ configs/, scripts/, preflight references), content-hash it, upload to s3://BUCKET/code/<hash>.tgz."""
    # 2026-11-05: one fixed path let two sessions launching at once overwrite each other's tarball between
    # the hash and the upload (an IncompleteBody upload during the N13 R0 relaunches). One directory per launch.
    tgz = pathlib.Path(tempfile.mkdtemp(prefix="pg_code_")) / "pg_code.tgz"
    # MENDELEEV (2026-09-03): jobs read configs/matrix_recipes.json, scripts/, and the banked preflight
    # references; a tarball of infra/ alone made the first smoke fail at the preflight stage.
    # noether_sweep reads broca_metrics.json at start-up (BROCA_ARTIFACT) and asserts its own baselines
    # reproduce it; without the file in the tarball the job dies on FileNotFoundError before the GPU is
    # touched. Added 2026-09-03 when the NOETHER W1 queue entry was made launchable.
    extra = ["configs", "scripts", "studies/lineage/cross-lineage/results/carrel_metrics.json",
             "studies/lineage/cross-lineage/results/carrel_tail_metrics.json",
             "studies/lineage/cross-lineage/results/broca_metrics.json",
             # 2026-11-04: the replications' job and library modules live under studies/reproductions/<claim>/code/
             "studies/reproductions/evilqwen/code", "studies/reproductions/alwayson/code",
             "studies/reproductions/instruments_e2/code"]
    subprocess.run(["tar", "czf", str(tgz), "-C", str(REPO), "infra"] + [e for e in extra if (REPO / e).exists()], check=True)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()[:12]
    key = f"code/{digest}.tgz"
    s3.upload_file(str(tgz), C.BUCKET, key)
    return key


# ---- user-data -----------------------------------------------------------------------------
def render_user_data(code_key, job_module, job_args, s3_prefix, budget_sec, weights_s3,
                     weights_include=""):
    # set -euo pipefail: a failed `aws s3 cp` of the code tarball used to silently no-op, letting
    # `tar -xzf` extract garbage and run_job.sh try to import a non-existent module — the instance
    # reached `terminated` with zero forensic surface. Fail-fast here means the user-data script
    # exits non-zero before run_job.sh starts, so `watch` returns 2 with a clear "no job.log
    # streamed" tail and the EventBridge reaper still backstops teardown.
    git_sha = _git_sha()
    # The ~91 GB image-dataset restore is opt-in per launch (run_job.sh explains why it is not the
    # default): `PG_DATASETS_VISION=1 pg run …` passes the launching shell's value to the instance.
    ds_vision = os.environ.get("PG_DATASETS_VISION", "0")
    return f"""#!/usr/bin/env bash
set -euxo pipefail
mkdir -p /opt/pg/repo
aws s3 cp s3://{C.BUCKET}/{code_key} /tmp/pg_code.tgz --region {C.REGION}
tar -xzf /tmp/pg_code.tgz -C /opt/pg/repo
export PG_WEIGHTS_INCLUDE="{weights_include}"
export PG_GIT_SHA="{git_sha}"
export PG_DATASETS_VISION="{ds_vision}"
bash /opt/pg/repo/{RUN_JOB_SH} "{job_module}" "{job_args}" "{s3_prefix}" "{budget_sec}" "{weights_s3}"
"""


# ---- launch --------------------------------------------------------------------------------
def _launch(job_module, job_args, s3_prefix, budget_sec, instance_types, attach_weights,
            weights_include=""):
    """Try each instance TYPE in preference order, each across all AZs (then EC2-chosen-AZ),
    taking the FIRST that launches. Capacity/quota/unsupported attempts are skipped instantly —
    never waited on. Returns (instance_id, launched_type). Fails loud only when every type is
    exhausted."""
    code_key = package_code()
    weights_s3 = C.WEIGHTS_S3_URI if attach_weights else ""
    ud = render_user_data(code_key, job_module, job_args, s3_prefix, budget_sec, weights_s3,
                          weights_include)
    short = job_module.split(".")[-1]
    base = dict(
        ImageId=C.AMI_ID,
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": C.INSTANCE_PROFILE},
        SecurityGroupIds=[C.SECURITY_GROUP],
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=base64.b64encode(ud.encode()).decode(),
        BlockDeviceMappings=[{
            "DeviceName": C.ROOT_DEVICE,
            "Ebs": {"VolumeSize": C.ROOT_GB, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "app", "Value": "pg-lineage"},
            {"Key": "job", "Value": short},
            {"Key": "s3prefix", "Value": s3_prefix},
            {"Key": "Name", "Value": f"pg-{short}"},
        ]}],
    )
    last = None
    for it in instance_types:
        # each AZ in order, then None = EC2-chosen-AZ (AWS's documented recommendation for scarce
        # types when every pinned AZ is dry).
        for subnet in C.SUBNETS + [None]:
            kw = dict(base, InstanceType=it)
            if subnet is not None:
                kw["SubnetId"] = subnet
            try:
                r = ec2.run_instances(**kw)
                return r["Instances"][0]["InstanceId"], it
            except ClientError as e:
                if e.response["Error"]["Code"] in _LAUNCH_SKIP:
                    last = (it, e.response["Error"]["Code"])
                    continue
                raise
    raise RuntimeError(
        f"none of {list(instance_types)} launchable in any AZ (no capacity / quota / unsupported); "
        f"last={last}")


def _git_sha():
    """HEAD sha of the launching worktree (+ '-dirty' if the tree differs), for manifest.json (§11.9)."""
    import subprocess as _sp
    h = _sp.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    d = _sp.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if d else "")


def _arm_deadline(instance_id, budget_sec):
    """Arm the OS-independent kill: a one-shot, self-deleting EventBridge schedule that
    terminates this instance at the deadline. Returns (expr, deadline_dt) so callers don't
    recompute the deadline (which drifted in PR #52). If arming fails we terminate the
    just-launched instance (so it can never orphan) and re-raise — cleanup, not error-hiding."""
    deadline = _now() + dt.timedelta(seconds=budget_sec + _DEADLINE_GRACE_S)
    expr = "at(" + deadline.strftime("%Y-%m-%dT%H:%M:%S") + ")"
    name = f"pg-reap-{instance_id}"
    try:
        sched.create_schedule(
            Name=name,
            ScheduleExpression=expr,
            ScheduleExpressionTimezone="UTC",
            FlexibleTimeWindow={"Mode": "OFF"},
            ActionAfterCompletion="DELETE",
            Target={
                "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                "RoleArn": C.SCHEDULER_ROLE_ARN,
                "Input": json.dumps({"InstanceIds": [instance_id]}),
            },
        )
    except sched.exceptions.ConflictException:
        # ALREADY ARMED (a second caller, a re-run): the instance is protected — the existing
        # deadline stands and is returned. This branch used to fall into the terminate-and-raise
        # below and killed a healthy image builder mid-push (2026-09-03, i-0526300f3840cc8a5).
        existing = sched.get_schedule(Name=name)
        expr = existing["ScheduleExpression"]
        deadline = dt.datetime.strptime(expr[3:-1], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=dt.timezone.utc)
        return expr, deadline
    except Exception:
        ec2.terminate_instances(InstanceIds=[instance_id])
        raise
    return expr, deadline


def run(job_module, job_args="", budget_min=240, instance_type=None, attach_weights=True,
        s3_prefix=None, as_json=False, detach=False, weights=None):
    """Launch one ephemeral GPU job. By default WATCHES the launched instance until STATUS
    appears or any anomaly fires (terminated-without-STATUS, stopped, deadline+grace). Pass
    `detach=True` to return immediately after launch (the run_job.sh trap + EventBridge
    reaper still guarantee teardown). Returns dict for detached mode; returns watch exit
    code (int 0-5) for watched mode. `weights` = comma list of HF-cache model dirs
    (models--org--name) to sync selectively from S3 (MENDELEEV contract §9); None = full sync;
    attach_weights=False = no sync at all."""
    # instance_type may be a comma-list ("g6e.8xlarge,p5.4xlarge") of fallbacks in preference
    # order, or None → the config default list.
    if instance_type:
        types = [t.strip() for t in instance_type.split(",") if t.strip()]
    else:
        types = list(C.INSTANCE_TYPES)
    short = job_module.split(".")[-1]
    s3_prefix = s3_prefix or f"results/{short}"
    budget_sec = int(budget_min * 60)
    iid, launched_type = _launch(job_module, job_args, s3_prefix, budget_sec, types, attach_weights,
                                 weights or "")
    expr, deadline = _arm_deadline(iid, budget_sec)
    _register(s3_prefix, iid, deadline)        # durable registry → reconciler can never miss it
    out = {"instance_id": iid, "type": launched_type, "tried": types, "s3_prefix": s3_prefix,
           "reap": expr, "deadline_utc": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "bucket": C.BUCKET}
    if as_json:
        print(json.dumps(out))
    else:
        print(f"launched {iid}  type={launched_type}  s3={s3_prefix}  reap={expr}")
    if detach:
        return out
    return watch(s3_prefix, iid, out["deadline_utc"])


# ---- watch (fail-fast observability) -------------------------------------------------------
# Watch exit codes (caller-checkable, documented in argparse epilog):
WATCH_EXIT_STATUS = 0       # STATUS marker present — read its body for the job's exit code
WATCH_EXIT_TERMINATED = 2   # instance terminated WITHOUT STATUS (died before run_job.sh trap)
WATCH_EXIT_STOPPED = 3      # instance stopped/stopping (shutdown-behavior should have been terminate)
WATCH_EXIT_DEADLINE = 4     # reaper deadline + grace passed without STATUS
WATCH_EXIT_NOT_FOUND = 5    # describe_instances cannot find the instance (typo, wrong region)
# (Exit 1 reserved for uncaught Python exceptions.)


def _read_status(s3_prefix):
    """Single-call GetObject for the STATUS marker. Returns body string if present, None on 404.
    Re-raises all other ClientErrors (AccessDenied, NoSuchBucket, transient throttling) per the
    fail-loud policy. Codes whitelisted: 404 (HEAD-style), NoSuchKey (GET-style), NotFound (older
    SDK translation) — same key, different reporting conventions."""
    try:
        return s3.get_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/STATUS")["Body"].read().decode().strip()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def _tail_log(s3_prefix, n_bytes=4096):
    """Last n_bytes of s3://BUCKET/{prefix}/job.log, or a 'not streamed' marker. Used to
    disambiguate exit-2 causes (user-data crash vs OOM vs network partition)."""
    try:
        head = s3.head_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/job.log")
        size = int(head["ContentLength"])
        start = max(0, size - n_bytes)
        rng = f"bytes={start}-{size-1}"
        body = s3.get_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/job.log", Range=rng)["Body"].read()
        return body.decode(errors="replace")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return "(no job.log streamed — died before run_job.sh's streamer loop fired)"
        raise


def _resolve_instance_by_prefix(s3_prefix):
    """Find the (single) running pg-lineage instance tagged with this s3prefix. Lets
    `pg watch <prefix>` resume without knowing the instance id — the laptop-is-incidental
    promise. Returns instance id, or raises if 0 or >1 matches."""
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:app", "Values": ["pg-lineage"]},
        {"Name": "tag:s3prefix", "Values": [s3_prefix]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    iids = [i["InstanceId"] for res in r["Reservations"] for i in res["Instances"]]
    if not iids:
        raise RuntimeError(
            f"no running pg-lineage instance tagged s3prefix={s3_prefix!r}. "
            f"It may have already terminated — check `aws s3 ls s3://{C.BUCKET}/{s3_prefix}/STATUS`.")
    if len(iids) > 1:
        raise RuntimeError(f"ambiguous: {len(iids)} pg-lineage instances tagged s3prefix={s3_prefix!r}: {iids}")
    return iids[0]


def watch(s3_prefix, instance_id=None, deadline_utc=None, poll_s=30, heartbeat_min=5):
    """Poll BOTH s3://BUCKET/<prefix>/STATUS and the instance state. Exits as soon as either
    a terminal STATUS appears OR any anomaly fires. Anomalies include: instance terminated
    without STATUS (probable user-data crash); unexpected stopped/stopping; reaper deadline +
    grace passed; describe_instances cannot find the named instance (typo / wrong region).
    Heartbeat line every heartbeat_min minutes (to stderr) so a tail of the output shows life
    without polluting stdout when paired with --json.

    Exit codes: see WATCH_EXIT_* module constants + argparse epilog.
    """
    if instance_id is None:
        instance_id = _resolve_instance_by_prefix(s3_prefix)
        print(f"[watch] resolved instance from tag:s3prefix={s3_prefix} → {instance_id}",
              file=sys.stderr)
    deadline = None
    if deadline_utc:
        deadline = dt.datetime.strptime(deadline_utc.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
        if _now() > deadline:
            print(f"[watch] ERROR: --deadline-utc {deadline_utc} is already in the past; "
                  f"watching without a deadline check.", file=sys.stderr)
            deadline = None
    last_heartbeat = 0.0
    last_state = None
    started = time.monotonic()
    while True:
        # 1) STATUS marker?
        body = _read_status(s3_prefix)
        if body is not None:
            print(f"[watch] STATUS appeared: {body} (elapsed {int(time.monotonic()-started)}s)",
                  file=sys.stderr)
            return WATCH_EXIT_STATUS
        # 2) instance state? (boto3 raises InvalidInstanceID.NotFound for a bogus id)
        try:
            r = ec2.describe_instances(InstanceIds=[instance_id])
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
                print(f"[watch] ERROR: instance {instance_id} not found (wrong region/typo)",
                      file=sys.stderr)
                return WATCH_EXIT_NOT_FOUND
            raise
        instances = [i for res in r["Reservations"] for i in res["Instances"]]
        if not instances:
            print(f"[watch] ERROR: instance {instance_id} not found (empty reservation)",
                  file=sys.stderr)
            return WATCH_EXIT_NOT_FOUND
        state = instances[0]["State"]["Name"]
        if state == "terminated":
            tail = _tail_log(s3_prefix)
            print(f"[watch] ERROR: instance {instance_id} TERMINATED without writing STATUS — "
                  f"died before run_job.sh's finish-trap fired. Last 4KB of job.log:\n{tail}",
                  file=sys.stderr)
            return WATCH_EXIT_TERMINATED
        if state in ("stopped", "stopping"):
            print(f"[watch] ERROR: instance {instance_id} state={state} — unexpected; "
                  f"shutdown-behavior should be terminate.", file=sys.stderr)
            return WATCH_EXIT_STOPPED
        # 3) deadline?
        if deadline and _now() > deadline + dt.timedelta(seconds=_DEADLINE_GRACE_S // 2):
            print(f"[watch] ERROR: reaper deadline {deadline.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                  f"+ {_DEADLINE_GRACE_S//2}s grace passed; instance state={state}; STATUS absent. "
                  f"The reaper should have fired — investigate.", file=sys.stderr)
            return WATCH_EXIT_DEADLINE
        # 4) heartbeat (stderr so --json users can pipe stdout cleanly)
        now_m = time.monotonic()
        if now_m - last_heartbeat >= heartbeat_min * 60 or state != last_state:
            up_h = (_now() - instances[0]["LaunchTime"]).total_seconds() / 3600
            print(f"[watch] {dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%SZ')} "
                  f"state={state} uptime={up_h:.2f}h ({instance_id}, s3={s3_prefix})",
                  file=sys.stderr)
            last_heartbeat = now_m
            last_state = state
        time.sleep(poll_s)


# ---- observe / control ---------------------------------------------------------------------
def _running():
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:app", "Values": ["pg-lineage"]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    return [i for res in r["Reservations"] for i in res["Instances"]]


def ps():
    insts = _running()
    if not insts:
        print("no pg-lineage instances running  ->  $0 GPU spend")
        return
    now = _now()
    print(f"{'INSTANCE':22} {'TYPE':14} {'JOB':16} {'STATE':10} {'UPTIME':>9}  COST")
    for i in insts:
        tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
        up_h = (now - i["LaunchTime"]).total_seconds() / 3600
        cost = f"~${up_h * C.ONDEMAND_USD_PER_HR:,.2f}" if C.ONDEMAND_USD_PER_HR else "(set price)"
        print(f"{i['InstanceId']:22} {i['InstanceType']:14} {tags.get('job',''):16} "
              f"{i['State']['Name']:10} {up_h:8.2f}h  {cost}")


# A job is "progressing" if any of these counters moved between the two most recent heartbeats.
# They are cumulative, so a strict increase is unambiguous — there is no threshold to tune and no
# way for a stuck process to fake one.
_PROGRESS_COUNTERS = ("cpu_seconds", "net_rx_bytes", "read_bytes", "write_bytes")


def liveness(s3_prefix):
    """Classify a job from its heartbeat: COMPUTING / DOWNLOADING / IO / NO-PROGRESS / UNKNOWN.

    This is the sensor that `job.log` never was. The log only moves when the science code prints,
    which it does not do inside a shard download or a dense eigendecomposition, so a silent log
    means nothing at all. These counters come from /proc and move whether or not the job speaks.
    """
    try:
        body = s3.get_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/heartbeat.jsonl")["Body"].read()
    except s3.exceptions.NoSuchKey:
        # pre-heartbeat launches, and jobs that have not written one yet
        return {"state": "UNKNOWN", "why": "no heartbeat.jsonl (job predates the heartbeat, or "
                                           "has not completed its first 30s cycle)"}
    rows = [json.loads(ln) for ln in body.decode().splitlines() if ln.strip()]
    return classify_heartbeats(rows)


def classify_heartbeats(rows):
    """Pure classifier over heartbeat samples — separated from S3 so it can be tested.

    Ordered most-specific first: CPU movement means the process is running code, which is the
    strongest evidence and subsumes the rest. Network movement with no CPU is the download case
    that fooled the recensus7 diagnosis. Disk-only movement is a slow write or a cache eviction.
    """
    if len(rows) < 2:
        return {"state": "UNKNOWN",
                "why": f"only {len(rows)} heartbeat sample(s); need 2 to see a delta",
                "latest": rows[-1] if rows else None}
    a, b = rows[-2], rows[-1]
    # "I COULD NOT FIND THE PROCESS" IS NOT "THE PROCESS IS STUCK". Without this, a heartbeat that
    # failed to resolve the interpreter reports every per-process counter as 0 and the job
    # classifies NO-PROGRESS — a measured-nothing/nothing-to-measure conflation, the same one that
    # produced M-D71, M-D99 and M-D100, except here it would make `pg kill`'s guard fail OPEN on the
    # exact job it exists to protect.
    if str(b.get("pid", "none")) in ("none", "", "0"):
        return {"state": "UNKNOWN", "why": "heartbeat could not resolve the job's pid; per-process "
                                           "counters are unmeasured, not zero",
                "latest": b, "n_samples": len(rows)}
    delta = {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in _PROGRESS_COUNTERS}
    if delta["cpu_seconds"] > 0:
        state = "COMPUTING"
    elif delta["net_rx_bytes"] > 0:
        state = "DOWNLOADING"
    elif delta["read_bytes"] > 0 or delta["write_bytes"] > 0:
        state = "IO"
    else:
        state = "NO-PROGRESS"
    return {"state": state, "delta": delta, "from": a["utc"], "to": b["utc"],
            "pid": b.get("pid"), "rss_kb": b.get("rss_kb"), "n_samples": len(rows),
            "log_bytes": b.get("log_bytes")}


def kill(ids=None, job=None, all_=False, force=False):
    insts = _running()
    if all_:
        targets = [i["InstanceId"] for i in insts]
    elif job:
        targets = [i["InstanceId"] for i in insts
                   if {t["Key"]: t["Value"] for t in i.get("Tags", [])}.get("job") == job]
    else:
        targets = ids or []
    if not targets:
        print("nothing to kill")
        return
    # THE GUARD. Twice a healthy job has been diagnosed as stalled from a frozen log — once it was
    # killed and five hours of work were lost (M-D97). The rule "never kill on a monitoring
    # inference" was written down and then violated by the same reasoning a second time, which is
    # the signal that a rule is the wrong mechanism. Now the tool refuses: an instance whose
    # heartbeat shows CPU, network or disk movement is DEMONSTRABLY working, and terminating it
    # takes an explicit --force. The reaper and the budget remain the intended way jobs end.
    if not force:
        by_id = {i["InstanceId"]: i for i in insts}
        blocked = []
        for iid in targets:
            tags = {t["Key"]: t["Value"] for t in by_id.get(iid, {}).get("Tags", [])}
            pfx = tags.get("s3prefix")
            if not pfx:
                continue
            live = liveness(pfx)
            if live["state"] in ("COMPUTING", "DOWNLOADING", "IO"):
                blocked.append((iid, pfx, live))
        if blocked:
            for iid, pfx, live in blocked:
                print(f"REFUSED {iid} ({pfx}): {live['state']} — "
                      f"{live['delta']} between {live['from']} and {live['to']}", file=sys.stderr)
            print("A frozen job.log is not evidence of a stall; these counters are evidence of "
                  "work. Re-run with --force if you mean it.", file=sys.stderr)
            targets = [i for i in targets if i not in {b[0] for b in blocked}]
            if not targets:
                return
    ec2.terminate_instances(InstanceIds=targets)
    for iid in targets:
        sched.delete_schedule(Name=f"pg-reap-{iid}")  # tidy the now-moot reaper
    print("terminated:", " ".join(targets))


def logs(s3_prefix):
    obj = s3.get_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/job.log")
    sys.stdout.write(obj["Body"].read().decode(errors="replace"))


def audit():
    """Diff running instances against reaper schedules. Surfaces (a) orphan reaper
    schedules whose instance has already terminated (harmless but messy), and (b) running
    pg-lineage instances with NO armed reaper (the dangerous case — a wedged box that
    nothing will kill). Prints a one-line summary then either 'clean' or the discrepancies."""
    running = {i["InstanceId"]: i for i in _running()}
    scheds = {}
    paginator = sched.get_paginator("list_schedules")
    for page in paginator.paginate(NamePrefix="pg-reap-"):
        for s in page["Schedules"]:
            scheds[s["Name"][len("pg-reap-"):]] = s["Name"]
    orphan_reapers = sorted(set(scheds) - set(running))
    reaperless = sorted(set(running) - set(scheds))
    print(f"audit: {len(running)} running, {len(scheds)} reapers, "
          f"{len(orphan_reapers)} orphans, {len(reaperless)} reaper-less")
    if not orphan_reapers and not reaperless:
        print("  clean.")
        return
    for iid in reaperless:
        print(f"  REAPER-LESS instance (DANGER — manually `pg kill {iid}`): {iid}")
    for iid in orphan_reapers:
        print(f"  orphan reaper (schedule for terminated instance): pg-reap-{iid}")


def collect(prefix, expect=None):
    r = s3.list_objects_v2(Bucket=C.BUCKET, Prefix=prefix)
    done = [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith("/STATUS")]
    print(f"{len(done)} finished under {prefix}" + (f" (expected {expect})" if expect else ""))
    for k in done:
        body = s3.get_object(Bucket=C.BUCKET, Key=k)["Body"].read().decode().strip()
        print(f"  {k}  {body}")


# ---- cli -----------------------------------------------------------------------------------
# ===========================================================================
# Durable active-jobs registry + fleet-wide reconciler — the root-cause fix for the
# silent-idle / missed-completion failure mode. `pg run` registers every launch; a
# scheduled reconciler routine (cron, server-side) lists the registry every N min,
# classifies each job, and ACTS on terminal ones — so no completion/error is ever missed,
# independent of the laptop, this session, or background-notification reliability. A
# terminal job STAYS registered until `pg ack`ed, so a skipped tick can't lose it.
# ===========================================================================

REG_PREFIX = "registry/active/"
BOOT_GRACE_MIN = 12        # no-instance + no-STATUS within this of launch = LAUNCHING, not STALLED


def _reg_key(s3_prefix):
    return REG_PREFIX + s3_prefix.replace("/", "__") + ".json"


def _register(s3_prefix, instance_id, deadline):
    """Record a launched job so a later reconcile finds it AFTER its instance is gone."""
    body = json.dumps({"s3_prefix": s3_prefix, "instance_id": instance_id,
                       "launched_utc": _now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "deadline_utc": deadline.strftime("%Y-%m-%dT%H:%M:%SZ")})
    s3.put_object(Bucket=C.BUCKET, Key=_reg_key(s3_prefix), Body=body.encode())


def _list_registry():
    out = []
    r = s3.list_objects_v2(Bucket=C.BUCKET, Prefix=REG_PREFIX)
    for o in r.get("Contents", []):
        try:
            out.append(json.loads(s3.get_object(Bucket=C.BUCKET, Key=o["Key"])["Body"].read()))
        except ClientError:
            pass
    return out


def _pipeline_progress(s3_prefix):
    try:
        body = s3.get_object(Bucket=C.BUCKET, Key=f"{s3_prefix}/out/pipeline_status.json")["Body"].read()
        return json.loads(body)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        return None


def classify(s3_prefix, launched_utc=None, instance_id=None):
    """State of ONE job. Returns (state, detail) with state in
    COMPLETE / FAILED / RUNNING / LAUNCHING / STALLED / EXTERNAL. `launched_utc` (from the registry)
    kills the launch-window false-positive: no-instance + no-STATUS during boot reads LAUNCHING, not
    STALLED. STALLED = instance gone, no terminal STATUS, past boot grace (the real silent death).
    EXTERNAL (N16): a `vast:<id>` handle — the job runs on Vast under infra/vast/watchdog.py, which
    owns its liveness; there is no EC2 instance or STATUS marker to read, so it is never STALLED."""
    if str(instance_id or "").startswith("vast:"):
        return "EXTERNAL", f"external (vast watchdog) {instance_id}"
    status = _read_status(s3_prefix)
    prog = _pipeline_progress(s3_prefix)
    tag = (f"{prog.get('completed')}/{prog.get('n_steps')} steps" if prog else "")
    if (status and status.strip() == "exit=0") or (prog and prog.get("state") == "complete"):
        return "COMPLETE", f"exit=0 {tag}".strip()
    if status and status.strip() != "exit=0":
        return "FAILED", f"STATUS {status.strip()} {tag}".strip()
    try:
        _resolve_instance_by_prefix(s3_prefix)
        return "RUNNING", tag or "instance up"
    except RuntimeError:
        pass
    if launched_utc:
        age = (_now() - dt.datetime.strptime(launched_utc.rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
               .replace(tzinfo=dt.timezone.utc)).total_seconds() / 60
        if age < BOOT_GRACE_MIN:
            return "LAUNCHING", f"{age:.0f}m old, booting {tag}".strip()
    return "STALLED", f"instance gone, no STATUS {tag}".strip()


def watchdog(s3_prefix):
    """Single-job state check (uses the registry's launched_utc if present for the boot grace).
    exit 0=COMPLETE, 1=FAILED/STALLED (alarm), 2=RUNNING/LAUNCHING."""
    launched = handle = None
    try:
        reg = json.loads(s3.get_object(Bucket=C.BUCKET, Key=_reg_key(s3_prefix))["Body"].read())
        launched, handle = reg.get("launched_utc"), reg.get("instance_id")
    except ClientError:
        pass
    st, detail = classify(s3_prefix, launched, handle)
    print(f"[watchdog] {s3_prefix}: {st} ({detail})")
    return {"COMPLETE": 0, "FAILED": 1, "STALLED": 1, "RUNNING": 2, "LAUNCHING": 2, "EXTERNAL": 2}[st]


def reconcile():
    """Fleet-wide reconciler — the awareness primitive a scheduled routine runs every N min.
    Classifies every job in the active registry. A terminal job (COMPLETE/FAILED/STALLED) stays
    registered until `pg ack`ed, so it re-surfaces every reconcile — it CANNOT be silently missed.
    Exit code = the action the caller must take:
      1  some job FAILED or STALLED          → surface the error (loud), investigate
      2  some job COMPLETE (and none failed)  → pull results, advance pipeline, then `pg ack`
      0  all jobs RUNNING/LAUNCHING, or registry empty → nothing to do (legitimately idle)."""
    entries = _list_registry()
    if not entries:
        print("[reconcile] active registry empty — no AWS jobs in flight")
        return 0
    buckets = {}
    for e in sorted(entries, key=lambda x: x.get("s3_prefix", "")):
        st, detail = classify(e["s3_prefix"], e.get("launched_utc"), e.get("instance_id"))
        buckets.setdefault(st, []).append(e["s3_prefix"])
        print(f"[reconcile] {st:9s} {e['s3_prefix']}  ({detail})")
    if buckets.get("FAILED") or buckets.get("STALLED"):
        print(f"[reconcile] ALARM: failed={buckets.get('FAILED', [])} stalled={buckets.get('STALLED', [])}")
        return 1
    if buckets.get("COMPLETE"):
        print(f"[reconcile] ACTIONABLE: complete (pull+ack)={buckets['COMPLETE']}")
        return 2
    return 0


def ack(s3_prefix):
    """Remove a job from the active registry once its terminal state has been HANDLED (results
    pulled / error surfaced). Until ack'd it re-surfaces in every reconcile (so it can't be lost)."""
    s3.delete_object(Bucket=C.BUCKET, Key=_reg_key(s3_prefix))
    print(f"[ack] {s3_prefix} removed from active registry")


def register_cli(s3_prefix, instance_id="(manual)", deadline_min=240):
    """Manually register a job already launched outside the auto-register path. A `vast:<id>`
    handle registers a Vast rental so `pg reconcile` lists it as EXTERNAL (N16), never STALLED."""
    _register(s3_prefix, instance_id, _now() + dt.timedelta(minutes=deadline_min))
    print(f"[register] {s3_prefix} added to active registry (handle {instance_id})")


def main():
    p = argparse.ArgumentParser(prog="pg")
    sub = p.add_subparsers(dest="cmd", required=True)

    _WATCH_EPILOG = (
        "watch exit codes:\n"
        f"  {WATCH_EXIT_STATUS}  STATUS marker present (read it for the job's exit code)\n"
        f"  {WATCH_EXIT_TERMINATED}  instance terminated WITHOUT STATUS (died before run_job.sh trap)\n"
        f"  {WATCH_EXIT_STOPPED}  instance stopped/stopping (shutdown-behavior should be terminate)\n"
        f"  {WATCH_EXIT_DEADLINE}  reaper deadline + grace passed without STATUS\n"
        f"  {WATCH_EXIT_NOT_FOUND}  instance not found (typo, wrong region)\n"
        "  1  uncaught Python exception")

    r = sub.add_parser("run",
                       help="launch one job; watches by default (use --detach to skip the watch)",
                       epilog=_WATCH_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("job_module")
    r.add_argument("--args", default="")
    r.add_argument("--budget-min", type=int, default=240)
    r.add_argument("--type", default=None,
                   help="instance type, or comma-list of fallbacks in preference order "
                        "(default: config INSTANCE_TYPES). First with capacity wins; no waiting.")
    r.add_argument("--no-weights", action="store_true")
    r.add_argument("--weights", default=None,
                   help="comma list of HF-cache model dirs (models--google--gemma-2-2b,...) to sync "
                        "selectively from S3; default = full weights sync; absent models download "
                        "from the hub in-job")
    r.add_argument("--s3-prefix", default=None)
    r.add_argument("--json", action="store_true", help="emit single-line JSON for parsing")
    r.add_argument("--detach", action="store_true",
                   help="opt OUT of watching; return after launch (instance still self-terminates)")

    w = sub.add_parser("watch", help="poll BOTH S3 STATUS and instance state; exit fast on anomaly",
                       epilog=_WATCH_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    w.add_argument("s3_prefix")
    w.add_argument("--instance", default=None,
                   help="instance id; omit to resolve from tag:s3prefix=<prefix> (laptop-resumable)")
    w.add_argument("--deadline-utc", default=None, help="YYYY-MM-DDTHH:MM:SSZ; exit 4 on deadline+grace")
    w.add_argument("--poll-s", type=int, default=30)
    w.add_argument("--heartbeat-min", type=int, default=5)

    sub.add_parser("audit", help="diff running instances vs reaper schedules (orphans + reaper-less)")

    m = sub.add_parser("map")
    m.add_argument("job_module")
    m.add_argument("--over", required=True, help="JSON list of arg-strings, one instance each")
    m.add_argument("--budget-min", type=int, default=240)
    m.add_argument("--type", default=None, help="instance type or comma-list of fallbacks")
    m.add_argument("--no-weights", action="store_true")
    m.add_argument("--weights", default=None, help="comma list of HF-cache model dirs to sync selectively")

    sub.add_parser("ps")

    k = sub.add_parser("kill")
    k.add_argument("ids", nargs="*")
    k.add_argument("--job")
    k.add_argument("--all", action="store_true")
    k.add_argument("--force", action="store_true",
                   help="terminate even if the heartbeat shows the job is working. Without this, an instance with advancing CPU/network/disk counters is REFUSED (M-D97).")

    lv = sub.add_parser("liveness",
                        help="is this job WORKING? classifies from /proc counters in heartbeat.jsonl (COMPUTING/DOWNLOADING/IO/NO-PROGRESS) — job.log going quiet proves nothing")
    lv.add_argument("s3_prefix")

    lg = sub.add_parser("logs")
    lg.add_argument("s3_prefix")

    co = sub.add_parser("collect")
    co.add_argument("prefix")
    co.add_argument("--expect", type=int)

    wd = sub.add_parser("watchdog",
                        help="single-job state: exit 0=complete, 1=failed/STALLED (alarm), 2=running")
    wd.add_argument("s3_prefix")

    sub.add_parser("reconcile",
                   help="fleet-wide reconciler over the active registry (the scheduled-routine "
                        "primitive): exit 1=any failed/stalled, 2=any complete (pull+ack), 0=all in flight")

    ak = sub.add_parser("ack", help="remove a HANDLED terminal job from the active registry")
    ak.add_argument("s3_prefix")

    rg = sub.add_parser("register", help="manually register a job into the active registry")
    rg.add_argument("s3_prefix")
    rg.add_argument("--handle", default="(manual)",
                    help="instance handle; vast:<instance_id> marks an EXTERNAL (vast-watchdog) job")
    rg.add_argument("--deadline-min", type=int, default=240)

    a = p.parse_args()
    if a.cmd == "run":
        result = run(a.job_module, a.args, a.budget_min, a.type, not a.no_weights,
                     a.s3_prefix, as_json=a.json, detach=a.detach, weights=a.weights)
        if not a.detach:
            sys.exit(result)
    elif a.cmd == "map":
        # Map fanout: launch N concurrent jobs, then watch the LAST one (safest single-process
        # behavior — earlier instances run independently with their own reapers). Use --detach
        # at the map level if you want strict fire-and-forget across all N.
        argsets = json.loads(a.over)
        short = a.job_module.split(".")[-1]
        outs = [run(a.job_module, args, a.budget_min, a.type, not a.no_weights,
                    s3_prefix=f"results/{short}/{i}", detach=True, weights=a.weights)
                for i, args in enumerate(argsets)]
        if not getattr(a, "detach", False) and outs:
            last = outs[-1]
            print(f"[map] watching last of {len(outs)} fanout instances: {last['instance_id']}",
                  file=sys.stderr)
            sys.exit(watch(last["s3_prefix"], last["instance_id"], last["deadline_utc"]))
    elif a.cmd == "watch":
        sys.exit(watch(a.s3_prefix, a.instance, a.deadline_utc, a.poll_s, a.heartbeat_min))
    elif a.cmd == "audit":
        audit()
    elif a.cmd == "ps":
        ps()
    elif a.cmd == "kill":
        kill(a.ids, a.job, a.all, a.force)
    elif a.cmd == "liveness":
        print(json.dumps(liveness(a.s3_prefix), indent=1))
    elif a.cmd == "logs":
        logs(a.s3_prefix)
    elif a.cmd == "collect":
        collect(a.prefix, a.expect)
    elif a.cmd == "watchdog":
        sys.exit(watchdog(a.s3_prefix))
    elif a.cmd == "reconcile":
        sys.exit(reconcile())
    elif a.cmd == "ack":
        ack(a.s3_prefix)
    elif a.cmd == "register":
        register_cli(a.s3_prefix, a.handle, a.deadline_min)


if __name__ == "__main__":
    main()
