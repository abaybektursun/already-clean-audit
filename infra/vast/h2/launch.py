#!/usr/bin/env python3
"""Launch the HELMHOLTZ-2 full-FT AdamW ladder on one ephemeral Vast 80GB box.

Careful-money guards: explicit offer id (chosen from a fresh search), box-side TTL
dead-man (curl DELETE with the per-instance key) + the laptop watcher as second layer,
early-exit on gate pass, no automatic paid retry, no AWS creds on the box (presigned URLs).
Run via .pgvenv python (boto3 for PUT presigns). Usage: launch.py OFFER_ID TTL_SECONDS
"""
import json, pathlib, subprocess, sys, time


def pick_offer_auto():
    """User policy (2026-09-02): (1) most efficient FAST hardware — bigger GPU wins when its
    speedup beats its premium; (2) no network/other bottlenecks; (3) cheapest satisfying 1+2.
    Mechanically: sm80/90 only (stock-torch kernels), reliability>0.98, cpu_ram>=64G,
    listed inet_down>=800 (floor only — the runtime 50MB net probe is the real gate),
    rank by DLPerf per dollar, tie-break cheaper."""
    q = subprocess.run(["vastai", "search", "offers",
                        "gpu_ram>=79 num_gpus=1 reliability>0.98 cpu_ram>=64 inet_down>=800",
                        "-o", "dph", "--raw"], check=True, capture_output=True, text=True)
    offs = [o for o in json.loads(q.stdout)
            if any(k in o["gpu_name"] for k in ("A100", "H100", "H200"))
            and o.get("dlperf") and o["dph_total"] < 4.0]
    assert offs, "no eligible sm80/90 offer"
    offs.sort(key=lambda o: (-o["dlperf"] / o["dph_total"], o["dph_total"]))
    for o in offs[:3]:
        print(f"  candidate {o['id']} {o['gpu_name']:<12} ${o['dph_total']:.3f}/h "
              f"dlperf/$={o['dlperf']/o['dph_total']:.1f} rel={o['reliability2']:.3f} "
              f"inet↓{o['inet_down']:.0f}", flush=True)
    return offs[0]["id"]


OFFER = pick_offer_auto() if sys.argv[1] == "auto" else int(sys.argv[1])
TTL = int(sys.argv[2])
BUCKET, REGION, PROFILE = "pg-lineage-769063704696", "us-east-1", "fuelos"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
WT = pathlib.Path(__file__).resolve().parents[3]

def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True, **kw)

# presigns
import boto3
s3 = boto3.Session(profile_name=PROFILE, region_name=REGION).client("s3")
bind_url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": "results/h2_seed/out/metrics.json"}, ExpiresIn=TTL)
put_tgz = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": "results/h2_fullft_vast/result.tgz", "ContentType": "application/octet-stream"}, ExpiresIn=TTL)
put_done = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": "results/h2_fullft_vast/DONE.json", "ContentType": "application/json"}, ExpiresIn=TTL)
hf_token = run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token", "--with-decryption",
                "--query", "Parameter.Value", "--output", "text", "--profile", PROFILE, "--region", REGION]).stdout.strip()

# create (never print stderr on failure: may carry credentials — harness lesson)
c = subprocess.run(["vastai", "create", "instance", str(OFFER), "--image", IMAGE, "--disk", "80",
                    "--ssh", "--direct", "--raw"], capture_output=True, text=True, timeout=90)
assert c.returncode == 0, "vast create failed (stderr withheld)"
payload = json.loads(c.stdout)
assert payload.get("success") is True and isinstance(payload.get("new_contract"), int), payload.get("success")
iid = payload["new_contract"]; ikey = payload.get("instance_api_key", "")
print(f"LAUNCHED vast instance {iid}", flush=True)

# wait ssh
host = port = None
for _ in range(120):
    time.sleep(10)
    q = subprocess.run(["vastai", "show", "instance", str(iid), "--raw"], capture_output=True, text=True)
    if q.returncode == 0:
        row = json.loads(q.stdout)
        if row.get("actual_status") == "running" and row.get("ssh_host"):
            host, port = row["ssh_host"], int(row["ssh_port"])
            break
assert host, f"instance {iid} never reached running+ssh"
ssh = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", f"root@{host}"]
for _ in range(60):
    if subprocess.run([*ssh, "true"], capture_output=True).returncode == 0:
        break
    time.sleep(10)
else:
    raise SystemExit(f"ssh to {iid} never came up — destroy manually: vastai destroy instance {iid}")

# box-side TTL dead-man (second layer to the laptop watcher)
if ikey:
    dead = (f"nohup bash -c 'sleep {TTL}; until curl -fsS -X DELETE -H \"Authorization: Bearer {ikey}\" "
            f"https://console.vast.ai/api/v0/instances/{iid}/; do sleep 10; done' >/tmp/ttl.log 2>&1 </dev/null &")
    run([*ssh, dead])
# code + env + start
tar = "/tmp/h2repo.tgz"
run(["bash", "-c", f"git -C {WT} archive HEAD | gzip > {tar}"])
run([*ssh, "mkdir -p /workspace/repo"])
run(["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no", tar, f"root@{host}:/workspace/repo.tgz"])
run([*ssh, "tar xzf /workspace/repo.tgz -C /workspace/repo"])
env = f"export HF_TOKEN='{hf_token}'\nexport BIND_URL='{bind_url}'\nexport PUT_TGZ_URL='{put_tgz}'\nexport PUT_DONE_URL='{put_done}'\n"
run([*ssh, f"cat > /workspace/h2env << 'EV'\n{env}EV"])
run([*ssh, "nohup bash /workspace/repo/infra/vast/h2/driver.sh > /workspace/driver.log 2>&1 </dev/null &"])
print(f"DRIVER STARTED on {iid} ({host}:{port}); TTL dead-man {TTL}s armed: {bool(ikey)}", flush=True)
print(json.dumps({"instance": iid, "host": host, "port": port}))
