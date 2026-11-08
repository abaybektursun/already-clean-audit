"""Server-side backstop reaper — EventBridge fires this every 5 minutes.

The laptop-independent money guarantee: during `loading` NOTHING runs on the
instance (no lifecycle trap, no streamer), so a wedged host with the laptop
asleep bled storage-dollars unbounded (the 14 h / $0.81 lesson, D14). This
Lambda enforces ONLY policy.backstop_decide(): author-declared per-manifest
bounds with generous margins, marker cleanup, and orphan reaping. The laptop
watchdog remains the precise layer; a genuine job inside its own declared
contract is never killed here (see test_policy.py backstop cases).

Deploy: bash deploy_reaper.sh  (zips this file + policy.py)
"""
import json
import time
import urllib.request

import boto3

from policy import backstop_decide

BUCKET = "pg-lineage-769063704696"
# every program that rides this harness (program.json "program" + "s3_results"):
# label prefix -> S3 key base the rental dir hangs under (results/<label>/ for the
# early programs; MENDELEEV banks under results/mdl/ inside its scoped IAM prefix)
BASES = {"shackleton_": "results/", "amundsen_": "results/", "mendeleev_": "results/mdl/", "gutenberg_": "results/"}
API = "https://console.vast.ai/api/v0"


def vast_req(method, path, key):
    req = urllib.request.Request(
        API + path, method=method, data=b"{}" if method == "DELETE" else None,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def s3_find(s3, key):
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=key, MaxKeys=1)
    for o in r.get("Contents", []):
        if o["Key"] == key:
            return o
    return None


def s3_json(s3, key):
    if s3_find(s3, key) is None:
        return None
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def s3_age(s3, key):
    o = s3_find(s3, key)
    return None if o is None else time.time() - o["LastModified"].timestamp()


def handler(event, context):
    ssm = boto3.client("ssm", region_name="us-east-1")
    key = ssm.get_parameter(Name="/shackleton/vast-api-key",
                            WithDecryption=True)["Parameter"]["Value"]
    s3 = boto3.client("s3", region_name="us-east-1")
    data = vast_req("GET", "/instances/", key)
    insts = [i for i in data.get("instances", [])
             if str(i.get("label", "")).startswith(tuple(BASES))]
    now, reaped = time.time(), []
    for i in insts:
        label = i["label"]
        pfx = next(p for p in BASES if label.startswith(p))
        rental = label[len(pfx):]
        base = f"{BASES[pfx]}{label}/"
        man = s3_json(s3, base + "manifest.json")
        dec = backstop_decide(
            phase=str(i.get("actual_status")),
            age_min=(now - float(i.get("start_date", now))) / 60.0,
            declared=(man or {}).get("watch"),
            has_complete=s3_find(s3, base + "_rental_complete.json") is not None,
            has_joblog=s3_find(s3, base + "job.log") is not None,
            has_manifest=man is not None,
            liveness_age_s=s3_age(s3, base + "job_live.log"))
        if dec["action"] == "kill":
            vast_req("DELETE", f"/instances/{i['id']}/", key)
            reaped.append({"id": i["id"], "rental": rental,
                           "reason": dec["reason"]})
            print(f"REAPED {i['id']} {rental}: {dec['reason']}")
    print(f"checked {len(insts)} harness instances, reaped {len(reaped)}")
    return {"checked": len(insts), "reaped": reaped}
