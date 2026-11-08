#!/usr/bin/env bash
# POSTCHECK seed (EC2 user-data, root, Amazon Linux 2023 arm64, m7g.medium).
#
# WHAT AND WHY. POSTCHECK re-verifies five claims from FRESH downloads at PINNED revisions, so this
# seeder's job is not only to fetch bytes but to record exactly which bytes:
#   1. allenai/MolmoAct2-Think — the un-gated base, 21.8 GB, which cell N1 needs on a GPU. This is the
#      one heavy object; everything else here is text.
#   2. Both checkpoints' config.json / norm_stats.json / remote code, as plain trees, at the revisions
#      resolved in stage 0 — the files §1 and P2 are read from, so the control plane never reads them
#      from memory or from DEPTHANCHOR's copy.
#   3. EVERY released allenai MolmoAct2 model's config.json (a few kB each), so P1's sentence about
#      "all released Think fine-tunes" is a measurement over the collection rather than over the two
#      checkpoints that happen to be in hand.
#   4. The published recipe from the allenai/molmoact2 GitHub repo — the depth fine-tuning README block
#      and the training launcher whose flag defaults P2 quotes.
#
# Shape and its reasons are the DEPTHANCHOR seeder's (see seed_depthanchor_cpu.sh): download-then-upload
# needs no GPU and must not eat the 32-vCPU G quota; the Standard on-demand quota is 1 vCPU so
# m7g.medium is the only shape that fits; swap because the OOM killer once took a multi-shard fetch on
# this shape and said only "Killed"; sync after every stage; self-terminating via the EXIT trap.
exec > /var/log/pg-seed-pc.log 2>&1
B=pg-lineage-769063704696
PFX=postcheck
up() { aws s3 cp /var/log/pg-seed-pc.log s3://$B/$PFX/seed/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

( while true; do aws s3 cp /var/log/pg-seed-pc.log s3://$B/$PFX/seed/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
free -h

dnf install -y python3.11 python3.11-pip git
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
/opt/pg/venv/bin/pip install huggingface_hub

mkdir -p /opt/pg/hf /opt/pg/code /opt/pg/collection
export HF_HOME=/opt/pg/hf HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

BASE=allenai/MolmoAct2-Think
TUNED=allenai/MolmoAct2-Think-LIBERO

# ---- stage 0: resolve and PIN the revisions ---------------------------------------------------
# Written before anything is downloaded, so every later artifact can name the commit it came from.
# `sha` is the repo's current commit; the downloads below pass it explicitly rather than tracking main.
/opt/pg/venv/bin/python - "$BASE" "$TUNED" <<'PY' > /opt/pg/code/revisions.json
import json, sys, datetime
from huggingface_hub import HfApi
api = HfApi()
out = {"resolved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "repos": {}}
for repo in sys.argv[1:]:
    info = api.model_info(repo, files_metadata=False)
    out["repos"][repo] = {
        "sha": info.sha,
        "last_modified": str(info.lastModified),
        "n_siblings": len(info.siblings or []),
        "files": sorted(s.rfilename for s in (info.siblings or [])),
    }
print(json.dumps(out, indent=1))
PY
cat /opt/pg/code/revisions.json | head -20
aws s3 cp /opt/pg/code/revisions.json s3://$B/$PFX/code/v1/revisions.json --region us-east-1
echo "STAGE0_OK revisions pinned"

# ---- stage 1: both checkpoints' text, at those revisions --------------------------------------
/opt/pg/venv/bin/python - <<'PY'
import json, pathlib
from huggingface_hub import snapshot_download
rev = json.loads(pathlib.Path("/opt/pg/code/revisions.json").read_text())["repos"]
for repo, local in (("allenai/MolmoAct2-Think", "think_base"),
                    ("allenai/MolmoAct2-Think-LIBERO", "think_libero")):
    p = snapshot_download(repo, revision=rev[repo]["sha"],
                          local_dir=f"/opt/pg/code/{local}",
                          allow_patterns=["*.py", "*.json", "*.jinja", "*.md", "*.txt"])
    print("TEXT_AT", repo, rev[repo]["sha"][:12], p)
PY
ls -la /opt/pg/code/think_base /opt/pg/code/think_libero
aws s3 cp /opt/pg/code/ s3://$B/$PFX/code/v1/ --recursive --exclude '.cache/*' --exclude '*/.cache/*' --region us-east-1
echo "STAGE1_OK both checkpoints' text seeded"

# ---- stage 2: every released allenai MolmoAct2 model's config ---------------------------------
# P1 wants to say something about "all released Think fine-tunes". That is a claim about the
# COLLECTION, so the collection is enumerated and every config.json read — a few kB each. A repo whose
# config cannot be fetched is recorded as such rather than skipped.
/opt/pg/venv/bin/python - <<'PY' > /opt/pg/collection/molmoact2_configs.json
import json, sys, datetime
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
models = sorted({m.id for m in api.list_models(author="allenai", search="MolmoAct")},)
out = {"listed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "n_models": len(models), "models": {}}
for mid in models:
    entry = {}
    try:
        info = api.model_info(mid)
        entry["sha"] = info.sha
        entry["last_modified"] = str(info.lastModified)
        entry["files"] = sorted(s.rfilename for s in (info.siblings or []))
    except Exception as exc:
        entry["info_error"] = f"{type(exc).__name__}: {exc}"
    try:
        p = hf_hub_download(mid, "config.json", revision=entry.get("sha"))
        cfg = json.loads(open(p).read())
        entry["config"] = {k: cfg.get(k) for k in sorted(cfg) if not isinstance(cfg.get(k), (dict, list))}
        entry["gate"] = {k: cfg.get(k) for k in cfg if "depth" in k.lower() or "gate" in k.lower()}
        entry["architectures"] = cfg.get("architectures")
    except Exception as exc:
        entry["config_error"] = f"{type(exc).__name__}: {exc}"
    try:
        p = hf_hub_download(mid, "norm_stats.json", revision=entry.get("sha"))
        entry["norm_tags"] = sorted(json.loads(open(p).read()).keys())
    except Exception as exc:
        entry["norm_stats_error"] = f"{type(exc).__name__}: {exc}"
    out["models"][mid] = entry
    # progress goes to STDERR: stdout is redirected into the JSON file, and a progress line in the
    # middle of it makes the artifact unparseable (measured — it broke `results/postcheck/files/1`)
    print("COLLECTION", mid, entry.get("gate"), file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
PY
tail -5 /opt/pg/collection/molmoact2_configs.json
aws s3 cp /opt/pg/collection/molmoact2_configs.json s3://$B/$PFX/code/v1/molmoact2_configs.json --region us-east-1
echo "STAGE2_OK collection configs seeded"

# ---- stage 3: the published recipe, from the repository itself --------------------------------
# P2 quotes the depth fine-tuning command and the launcher's flag defaults. Clone at depth 1 for the
# files, then a FULL log grep for the flag so "no commit sets it" is a measurement and not an absence
# of evidence. Both the clone's HEAD and the grep counts are recorded.
mkdir -p /opt/pg/code/molmoact2_repo
git clone --quiet https://github.com/allenai/molmoact2.git /opt/pg/src/molmoact2 || echo "CLONE_FAILED"
if [ -d /opt/pg/src/molmoact2 ]; then
  cd /opt/pg/src/molmoact2
  git log -1 --format='%H %ad %s' > /opt/pg/code/molmoact2_repo/HEAD.txt
  # every file that mentions the flag, and every commit that touched it
  grep -rn 'action_expert_depth_gate' . --include='*' -l > /opt/pg/code/molmoact2_repo/files_mentioning_gate.txt || true
  grep -rn 'action_expert_depth_gate' . > /opt/pg/code/molmoact2_repo/grep_gate.txt || true
  git log --all --oneline -S 'action_expert_depth_gate' > /opt/pg/code/molmoact2_repo/commits_touching_gate.txt || true
  git log --all --oneline -S 'depth_code_input_noise_rate' > /opt/pg/code/molmoact2_repo/commits_touching_noise.txt || true
  for F in experiments/README.md README.md experiments/launch_scripts/train_lerobot.py \
           launch_scripts/train_lerobot.py scripts/train_lerobot.py ; do
    if [ -f "$F" ]; then
      mkdir -p "/opt/pg/code/molmoact2_repo/$(dirname "$F")"
      cp "$F" "/opt/pg/code/molmoact2_repo/$F"
      echo "RECIPE_OK $F"
    else
      echo "RECIPE_MISS $F"
    fi
  done
  find . -name 'train*.py' -o -name '*.md' | head -40 > /opt/pg/code/molmoact2_repo/tree_sample.txt
  cd /
fi
ls -laR /opt/pg/code/molmoact2_repo | head -40
aws s3 cp /opt/pg/code/molmoact2_repo/ s3://$B/$PFX/code/v1/molmoact2_repo/ --recursive --region us-east-1
echo "STAGE3_OK recipe seeded"

# ---- stage 4: the base checkpoint, the only heavy object --------------------------------------
# max_workers=1: 4 GB of RAM cannot hold concurrent multi-GB shard buffers, and the network is the
# bottleneck anyway. Pinned to the stage-0 revision.
/opt/pg/venv/bin/python - <<'PY'
import json, pathlib
from huggingface_hub import snapshot_download
rev = json.loads(pathlib.Path("/opt/pg/code/revisions.json").read_text())["repos"]
repo = "allenai/MolmoAct2-Think"
print("MODEL_AT", snapshot_download(repo, revision=rev[repo]["sha"], max_workers=1))
PY
du -sh /opt/pg/hf; df -h /
aws s3 sync /opt/pg/hf/ s3://$B/weights/hf/ --region us-east-1
echo "STAGE4_OK base checkpoint seeded: $BASE"

# ---- manifest + DONE ---------------------------------------------------------------------------
/opt/pg/venv/bin/python - <<'PY' > /tmp/manifest.json
import hashlib, json, pathlib, datetime
out = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "trees": {}}
for label, root in (("weights_hf", "/opt/pg/hf"), ("code", "/opt/pg/code")):
    files = {}
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            files[str(p.relative_to(root))] = p.stat().st_size
    out["trees"][label] = {"n_files": len(files), "bytes": sum(files.values()), "files": files}
sha = {}
for p in sorted(pathlib.Path("/opt/pg/code").rglob("*.json")):
    if p.is_file():
        sha[str(p.relative_to("/opt/pg/code"))] = hashlib.sha256(p.read_bytes()).hexdigest()
out["code_sha256"] = sha
print(json.dumps(out, indent=1))
PY
aws s3 cp /tmp/manifest.json s3://$B/$PFX/seed/manifest.json --region us-east-1
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/$PFX/seed/DONE --region us-east-1
echo "SEED_DONE"
