#!/usr/bin/env python3
"""Local controller: plan / rent / respawn / ledger / kill.

plan    <manifest>                    search offers matching the manifest
rent    <manifest> <offer_id>         create the instance; --race N hedges a
                                      wedge-prone big image with N-1 siblings
                                      (watchdog destroys the losers, cents)
respawn <rental> [--exclude-machine]  re-create from the BANKED manifest+code
                                      on the next-best host (watchdog calls
                                      this on infra-classified kills)
ledger                                pull realized $ from vast invoices
kill    <instance_id>                 destroy now

Detection/destroy lives ONLY in watchdog.py (+ the Lambda reaper backstop);
this file has no watch loop. All vast calls go through the vastai CLI. AWS
side uses --profile fuelos; the keys SHIPPED TO THE BOX come from the profile
named by program.json "box_profile" (MENDELEEV: the scoped IAM user
mendeleev-box, put/get under results/mdl/* only — D23).
"""
import argparse
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the program namespace (S3 base, instance label, spend ledger, box creds) is read
# from the tree, not the environment: the branch decides it and cannot forget it
CFG = json.load(open(os.path.join(HERE, "program.json")))
PROGRAM = CFG["program"]
LEDGER = os.path.join(HERE, "..", "..", CFG["ledger"])
S3_BASE = CFG["s3_results"]          # rentals live at {S3_BASE}/{PROGRAM}_{rental}/
BOX_PROFILE = CFG["box_profile"]     # local aws profile whose keys ship to the box
AWS = ["aws", "--profile", "fuelos", "--region", "us-east-1"]
# why each term exists: see README.md "Offer filter". No geolocation filter:
# image pull is unbilled, so geography costs wall clock, not dollars —
# and dropping it halves $/hr (measured 2026-08-16: A6000 $0.47 vs US $1.12)
BASE_QUERY = ("verified=true rentable=true reliability>0.98 "
              "inet_down>=200 inet_down_cost<={c} inet_up_cost<={c} "
              "storage_cost<=0.30")
# A manifest may raise the bandwidth cap ($/GB) when its one GPU model has no free-bandwidth host: N13 is pinned to the L40S
# (R0 ran on one), and on 2026-11-05 every L40S host charged $0.00065 to $0.0091 per GB. The pull bytes are then billed, so
# its estimated_download_gb is the image size and plan ranks by the estimate.
INET_COST_FREE = 0.0001


def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    # destroy prompts [y/N] interactively and replies with prose, not JSON (D9)
    confirm = args[0] == "destroy"
    r = subprocess.run(cmd, capture_output=True, text=True,
                       input="y\n" if confirm else None)
    # Create arguments contain credentials. Never include them in an error.
    detail = re.sub(r"[A-Za-z0-9_+/=.\-]{24,}", "[redacted]", (r.stderr or r.stdout)[:800])
    assert r.returncode == 0, f"vastai {' '.join(args[:2])} failed: {detail}"
    if confirm:
        return r.stdout
    if raw and not r.stdout.lstrip().startswith(("[", "{")):
        raise RuntimeError(f"vastai {' '.join(args[:2])} returned non-JSON: {detail}")
    return json.loads(r.stdout) if raw else r.stdout


def load_ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []


def save_ledger(rows):
    json.dump(rows, open(LEDGER, "w"), indent=2)


def wedged_machines():
    """Hosts that wedged (loading) or went dark (running, zero S3 bytes) on
    us before. vast's reliability score predicted neither — our own rental
    history is the better prior. With 30+ offers per plan, excluding a host
    on one bad rental costs nothing."""
    return {r.get("machine_id") for r in load_ledger()
            if r.get("host_verdict") in ("wedged", "died", "slow_net")
            and r.get("machine_id")}


def search_offers(man, bid=False, exclude_machines=()):
    q = f"{BASE_QUERY.format(c=man.get('inet_cost_max', INET_COST_FREE))} {man['gpu_query']} disk_space>={man['disk']}"
    if man.get("weights_mode") in ("download", "s3"):
        # weights ride BILLED time on both paths — only fast pipes make the trade win. "s3" is held
        # to the SAME bar as "download": our cache is fast, but the host's pipe is what we pay for,
        # and a 7.4 MB/s host is as fatal pulling from S3 as it was pulling from the hub.
        q += " inet_down>=800"
    offers = vast("search", "offers", q, "--storage", str(man["disk"]),
                  *(["--type", "bid"] if bid else []), "-o", "dph")
    bad = wedged_machines() | set(exclude_machines)
    offers = [o for o in offers if o.get("machine_id") not in bad]
    expected = re.sub(r"[\s_]", "", man.get("expected_gpu", "")).lower()
    if expected:
        offers = [o for o in offers if expected in re.sub(r"[\s_]", "", o.get("gpu_name", "")).lower()]
    # efficiency rule: overhead-bound rentals rank by $/hr; compute-bound by
    # $ per unit work ($/dlperf-hr) — the metric the ledger audits
    if man.get("bound", "overhead") == "compute":
        offers.sort(key=lambda o: o["dph_total"] / max(o.get("dlperf", 1), 1))
    elif "estimated_billed_h" in man and "estimated_download_gb" in man:
        offers.sort(key=lambda o: estimated_cost(o, man))
    return offers


def estimated_cost(offer, manifest):
    """Advertised GPU+disk time and model download; excludes unmeasured overhead."""
    return float(offer["dph_total"]) * float(manifest["estimated_billed_h"]) + \
        float(offer.get("inet_down_cost", 0)) * float(manifest["estimated_download_gb"])


def cmd_plan(man, bid):
    offers = search_offers(man, bid)
    for o in offers[:8]:
        # cpu_ram is shown because it is a real failure mode, not a nicety: a
        # host that allots ~12 GB per GPU slot starves any instrument that holds
        # full fp32 logits in numpy (AMUNDSEN's AM-2 lost ~30 min to exactly
        # that, GPU idle, one core pegged)
        print(f"offer {o['id']:>10}  {o['num_gpus']}x {o['gpu_name']:<18} "
              f"${o['dph_total']:.3f}/hr  work=${1000*o['dph_total']/max(o.get('dlperf',1),1):.2f}m  min_bid=${o.get('min_bid', 0):.3f}  "
              f"down={o['inet_down']:.0f}Mb/s  disk=${o['storage_cost']:.2f}/GB/mo  "
              f"ram={o.get('cpu_ram', 0)/1024:.0f}GB/{o.get('cpu_cores_effective', 0):.0f}c  "
              f"rel={o.get('reliability2', o.get('reliability', 0)):.3f}  "
              f"m{o.get('machine_id')}  {o['geolocation']}")
        if "estimated_billed_h" in man and "estimated_download_gb" in man:
            print(f"  advertised session estimate ${estimated_cost(o, man):.3f} "
                  f"({man['estimated_billed_h']} h + {man['estimated_download_gb']} GB download)")
    print(f"\n{len(offers)} offers ({len(wedged_machines())} wedged hosts excluded)")


def format_env(pairs):
    """Quote values for Vast's env parser, which otherwise truncates at spaces.

    Its parser does not support escaped/nested quotes. Reject those explicitly
    instead of silently changing a credential or passing extra environment flags.
    """
    out = []
    for key, value in pairs.items():
        if value is None or value == "":
            continue
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key), "invalid environment key"
        value = str(value)
        assert not any(c in value for c in "\"'\r\n\0"), f"unsupported quoting in environment value for {key}"
        out.append(f'-e {key}="{value}"')
    return " ".join(out)


def creds_env(man):
    get = lambda k: subprocess.run(["aws", "configure", "get", k, "--profile",
                                    BOX_PROFILE], capture_output=True, text=True
                                   ).stdout.strip()
    assert get("aws_access_key_id"), f"no local aws profile '{BOX_PROFILE}' (program.json box_profile)"
    vast_key = open(os.path.expanduser("~/.config/vastai/vast_api_key")).read().strip()
    pairs = {
        "RENTAL": man["rental"], "S3_BASE": S3_BASE, "VAST_API_KEY": vast_key,
        "PROGRAM": PROGRAM,
        "AWS_ACCESS_KEY_ID": get("aws_access_key_id"),
        "AWS_SECRET_ACCESS_KEY": get("aws_secret_access_key"),
        "AWS_DEFAULT_REGION": "us-east-1",
        "EXPECTED_GPU": man.get("expected_gpu", ""),
        "WEIGHTS_MODE": man.get("weights_mode", "baked"),
        # weights_mode "s3": pull these HF-cache dirs from our own S3 weight cache instead of the
        # hub (lifecycle.sh). WEIGHTS_HUB_MODELS is informational — the models the cache does NOT
        # hold, which fall through to the hub in-run — and is banked in weights_source.json.
        "WEIGHTS_S3_BASE": man.get("weights_s3_base", ""),
        "WEIGHTS_S3_MODELS": man.get("weights_s3_models", ""),
        "WEIGHTS_HUB_MODELS": man.get("weights_hub_models", ""),
        "WEIGHTS_HF_REPO": man.get("weights_hf_repo", ""),
        "WEIGHTS_DIR": man.get("weights_dir", ""),
        "WEIGHTS_IGNORE": man.get("weights_ignore", ""),
    }
    # the ONLY live HF token is SSM /pg-lineage/hf-token (memory: reference_infrastructure) — the
    # same parameter the AWS lane's hf_online_setup() reads via the instance role. The box has no
    # SSM access (mendeleev-box is S3-only), so the controller ships the value at rent time.
    tok = subprocess.run([*AWS, "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                          "--with-decryption", "--query", "Parameter.Value", "--output", "text",
                          "--region", "us-east-1"], capture_output=True, text=True, check=True
                         ).stdout.strip()
    assert tok.startswith("hf_"), "SSM /pg-lineage/hf-token returned no token"
    pairs["HF_TOKEN"] = tok
    pairs.update(man.get("env", {}))
    return format_env(pairs)


def ship(man):
    """Bank manifest + code on S3 BEFORE any instance exists — the Lambda
    reaper treats a labeled instance without a banked manifest as foreign."""
    root = os.path.join(HERE, "..", "..")
    sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    stamp_path = os.path.join(HERE, ".preflight_pass")
    stamp = open(stamp_path).read().strip() if os.path.exists(stamp_path) else ""
    # What SHIPS is `git archive HEAD infra`, so what preflight has to have
    # validated is HEAD's infra/ TREE — not the whole-repo SHA. Comparing tree
    # hashes keeps the gate exactly as strict on code (a changed byte under
    # infra/ changes the tree hash) while letting a journal commit through.
    def tree(ref):
        r = subprocess.run(["git", "-C", root, "rev-parse", f"{ref}:infra"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    assert stamp and tree(stamp) and tree(stamp) == tree("HEAD"), (
        f"BLOCKED: preflight stamp '{stamp}' does not cover HEAD '{sha}' infra/ tree "
        f"({tree(stamp)} vs {tree('HEAD')}) — run: bash launch_preflight.sh")
    # the manifest AND the code that run are the ones banked at rent time
    # (D11 + the 14h lesson: never couple a 1 KB fix to a 145 GB image)
    subprocess.run([*AWS, "s3", "cp", man["_path"],
                    f"{S3_BASE}/{PROGRAM}_{man['rental']}/manifest.json"],
                   check=True, capture_output=True)
    # MENDELEEV 2026-09-03: the job modules read configs/matrix_recipes.json (the frozen recipes, asserted
    # at parse) and scripts/ — shipping infra/ alone made them import- or assert-fail on the box. The
    # preflight tree gate below still covers infra/; the extra paths are data and are hashed in the
    # manifest each job writes.
    _extra = [p for p in ("configs", "scripts") if os.path.isdir(os.path.join(root, p))]
    # Jobs can require frozen data outside infra/. Ship explicitly named, tracked paths.
    for path in man.get("code_paths", []):
        assert isinstance(path, str) and not os.path.isabs(path) and ".." not in path.split("/"), path
        subprocess.run(["git", "-C", root, "cat-file", "-e", f"HEAD:{path}"], check=True)
        if path not in _extra:
            _extra.append(path)
    subprocess.run(["git", "-C", root, "archive", "--format=tar.gz",
                    "-o", f"/tmp/rentcode_{sha}.tar.gz", "HEAD", "infra"] + _extra,
                   check=True)
    subprocess.run([*AWS, "s3", "cp", f"/tmp/rentcode_{sha}.tar.gz",
                    f"{S3_BASE}/{PROGRAM}_{man['rental']}/code.tar.gz"],
                   check=True, capture_output=True)


def create_instance(man, offer_id, bid=None, auto_respawn=False):
    args = ["create", "instance", str(offer_id),
            "--image", man["image"], "--disk", str(man["disk"]),
            "--label", f"{PROGRAM}_{man['rental']}",
            # onstart does the controller-side things no image bakes in:
            # fetch the CURRENT code + rental manifest from S3 (D11 — rent-time
            # state, not image state), start the 60 s live streamer (job.log +
            # out/ artifacts — a crash loses <=60 s of rows), hand off to the
            # baked lifecycle
            "--onstart-cmd",
            f"aws s3 cp {S3_BASE}/{PROGRAM}_{man['rental']}/code.tar.gz /tmp/c.tgz && "
            "tar xzf /tmp/c.tgz -C /app && "
            f"aws s3 cp {S3_BASE}/{PROGRAM}_{man['rental']}/manifest.json "
            f"/app/infra/vast/manifests/{man['rental']}.json && "
            "( while true; do aws s3 cp /workspace/job.log "
            f"{S3_BASE}/{PROGRAM}_{man['rental']}/job_live.log >/dev/null 2>&1; "
            "aws s3 sync /workspace/out "
            f"{S3_BASE}/{PROGRAM}_{man['rental']}/live_out >/dev/null 2>&1; "
            "sleep 60; done & ); bash /app/infra/vast/lifecycle.sh",
            "--env", creds_env(man), "--ssh", "--cancel-unavail"]
    if ".ecr." in man["image"]:
        # ECR tokens live 12h; mint at create time — the host pulls immediately,
        # and interruptible restarts reuse the already-pulled image (no re-auth)
        registry = man["image"].split("/")[0]
        token = subprocess.run([*AWS, "ecr", "get-login-password",
                                "--region", "us-east-1"],
                               capture_output=True, text=True, check=True
                               ).stdout.strip()
        args += ["--login", f"-u AWS -p {token} {registry}"]
    if bid:
        args += ["--bid_price", str(bid)]
    res = vast(*args)
    rows = load_ledger()
    rows.append({"rental": man["rental"], "session": man.get("session"),
                 "instance_id": res.get("new_contract"),
                 "offer_id": offer_id, "machine_id": None,
                 "image": man["image"], "interruptible_bid": bid,
                 "auto_respawn": auto_respawn, "created_utc":
                 time.strftime("%FT%TZ", time.gmtime()), "status": "open",
                 "realized_usd": None})
    save_ledger(rows)
    print(f"[rent] instance {res.get('new_contract')} "
          f"({'respawn' if auto_respawn else 'primary'}) recorded in ledger")
    return res


def cmd_rent(man, offer_id, bid, race):
    # no blind global rules — so bounds are MANDATORY and author-declared:
    # the watchdog and the Lambda reaper enforce exactly what you declare here
    w = man.get("watch") or {}
    assert "max_billed_h" in w and "load_backstop_min" in w, (
        "BLOCKED: manifest must declare watch.max_billed_h and "
        "watch.load_backstop_min — per-job bounds, never global ones")
    ship(man)
    create_instance(man, offer_id, bid)
    if race > 1:
        offers = search_offers(man, bid=bid is not None)
        extra = [o for o in offers if str(o["id"]) != str(offer_id)][:race - 1]
        assert extra, "race requested but no alternate offers available"
        for o in extra:
            create_instance(man, o["id"], bid)
            print(f"[race] sibling on machine m{o.get('machine_id')} — "
                  f"watchdog destroys whichever loses (storage cents)")


def cmd_respawn(rental, exclude_machine):
    r = subprocess.run([*AWS, "s3", "cp",
                        f"{S3_BASE}/{PROGRAM}_{rental}/manifest.json", "-"],
                       capture_output=True, text=True, check=True)
    man = json.loads(r.stdout)
    man["_path"] = None   # banked manifest/code are reused as-is — no re-ship
    prior = [row.get("machine_id") for row in load_ledger()
             if row.get("rental") == rental and row.get("machine_id")]
    offers = search_offers(man, exclude_machines=set(prior) | {exclude_machine})
    assert offers, f"respawn {rental}: no offers left after exclusions"
    bid = next((row.get("interruptible_bid") for row in reversed(load_ledger())
                if row.get("rental") == rental), None)
    create_instance(man, offers[0]["id"], bid, auto_respawn=True)


def cmd_ledger():
    inv = vast("show", "invoices")
    rows = load_ledger()
    items = inv if isinstance(inv, list) else inv.get("invoices", inv.get("charges", []))
    for row in rows:
        iid = row.get("instance_id")
        mine = [it for it in items if it.get("instance_id") == iid]
        if not mine:
            continue
        cats = {}
        for it in mine:
            kind = "gpu" if "GPU" in it.get("description", "") else \
                   "storage" if "storage" in it.get("description", "") else "other"
            cats.setdefault(kind, {"usd": 0.0, "hours": 0.0})
            cats[kind]["usd"] += float(it.get("amount", 0))
            cats[kind]["hours"] += float(it.get("quantity", 0))
        row["billing"] = {k: {kk: round(vv, 4) for kk, vv in v.items()}
                          for k, v in cats.items()}
        row["realized_usd"] = round(sum(v["usd"] for v in cats.values()), 4)
        gpu_h = cats.get("gpu", {}).get("hours", 0)
        sto_h = cats.get("storage", {}).get("hours", 0)
        # cross-check of the unbilled-loading claim: lifetime minus GPU-billed
        row["unbilled_loading_hours"] = round(sto_h - gpu_h, 4)
    save_ledger(rows)
    total = sum(r["realized_usd"] or 0 for r in rows)
    print(f"[ledger] {len(rows)} rentals, realized total ${total:.2f}. "
          f"Raw invoices saved alongside ledger.")
    json.dump(items, open(LEDGER.replace(".json", "_invoices_raw.json"), "w"),
              indent=2)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan");  sp.add_argument("manifest"); sp.add_argument("--bid", action="store_true")
    sr = sub.add_parser("rent");  sr.add_argument("manifest"); sr.add_argument("offer_id")
    sr.add_argument("--bid", type=float); sr.add_argument("--race", type=int, default=1)
    sw = sub.add_parser("respawn"); sw.add_argument("rental")
    sw.add_argument("--exclude-machine", type=int, default=0)
    sub.add_parser("ledger")
    sk = sub.add_parser("kill");  sk.add_argument("instance_id")
    a = p.parse_args()

    if a.cmd in ("plan", "rent"):
        man = json.load(open(a.manifest))
        man["_path"] = a.manifest
    if a.cmd == "plan":
        cmd_plan(man, a.bid)
    elif a.cmd == "rent":
        cmd_rent(man, a.offer_id, a.bid, a.race)
    elif a.cmd == "respawn":
        cmd_respawn(a.rental, a.exclude_machine)
    elif a.cmd == "ledger":
        cmd_ledger()
    elif a.cmd == "kill":
        print(vast("destroy", "instance", a.instance_id))


if __name__ == "__main__":
    main()
