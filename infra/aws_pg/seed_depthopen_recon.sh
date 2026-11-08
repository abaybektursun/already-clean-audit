#!/usr/bin/env bash
# DEPTHOPEN recon seed (EC2 user-data, root, Amazon Linux 2023 arm64, m7g.medium).
#
# WHAT AND WHY. DEPTHOPEN trains five arms with the makers' own launcher, on their own data format, and
# evaluates on a third-party benchmark. Every one of those three has a shape this control plane has not
# read yet, and guessing any of them wastes GPU hours. So this job downloads nothing heavy: it reads the
# three sources and banks the text that the study's design has to be written from.
#
#   1. allenai/molmoact2 at the PINNED commit POSTCHECK recorded (66b87e64) — the full source tree as a
#      tarball for the training instance to fetch, plus a curated set of small files the control plane
#      reads directly: the mixture registry, the depth-annotation generator, the tag metadata, the
#      training plan, the packaging metadata, and a grep index so the next question can be answered
#      without another instance.
#   2. sylvestf/LIBERO-plus — the perturbation benchmark R5 is measured on. Its tree, its README, its
#      extra requirements, and above all `task_classification.json`, which is the file the 200-task
#      camera and layout samples are drawn from. A tarball too.
#   3. allenai/MolmoAct2-LIBERO-Dataset `meta/` only, plus the full repo file listing with sizes. The
#      subset in SPEC §4 has to be built from the real v3 layout (how data files, video files and
#      episodes line up), not from an assumed one.
#
# Shape and its reasons are the POSTCHECK seeder's: download-then-upload needs no GPU and must not eat
# the 32-vCPU G quota; the Standard on-demand quota is 1 vCPU so m7g.medium is the only shape that fits;
# swap because the OOM killer once took a multi-shard fetch on this shape and said only "Killed"; sync
# after every stage; self-terminating via the EXIT trap.
exec > /var/log/pg-seed-do.log 2>&1
B=pg-lineage-769063704696
PFX=depthopen
up() { aws s3 cp /var/log/pg-seed-do.log s3://$B/$PFX/seed/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

( while true; do aws s3 cp /var/log/pg-seed-do.log s3://$B/$PFX/seed/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
free -h

dnf install -y python3.11 python3.11-pip git tar gzip
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
/opt/pg/venv/bin/pip install huggingface_hub pyarrow

mkdir -p /opt/pg/hf /opt/pg/src /opt/pg/recon
export HF_HOME=/opt/pg/hf HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

MO2_SHA=66b87e64efd99dfd103241418113955cf64dfa9c   # POSTCHECK's pinned commit, seeded 2026-09-09
DATASET=allenai/MolmoAct2-LIBERO-Dataset

# ---- stage 1: the makers' training repo, at the pinned commit ---------------------------------
mkdir -p /opt/pg/recon/molmoact2
git clone --quiet https://github.com/allenai/molmoact2.git /opt/pg/src/molmoact2
cd /opt/pg/src/molmoact2
git checkout --quiet "$MO2_SHA"
git log -1 --format='%H %ad %s' > /opt/pg/recon/molmoact2/HEAD.txt
git rev-parse HEAD > /opt/pg/recon/molmoact2/sha.txt

# the whole tree, for the training instance to fetch in one object
tar --exclude=.git -czf /opt/pg/recon/molmoact2_at_${MO2_SHA:0:12}.tar.gz -C /opt/pg/src molmoact2
ls -la /opt/pg/recon/*.tar.gz

# a full file listing with sizes, so a later question does not need another instance
find . -path ./.git -prune -o -type f -printf '%10s %p\n' > /opt/pg/recon/molmoact2/files.txt
wc -l /opt/pg/recon/molmoact2/files.txt

# grep index: where the study's load-bearing names live
for TERM in TAG_METADATA_BY_TAG MOLMOACT2_LEROBOT_MIXTURES build_single_lerobot_mixture \
            depth_code_input_noise_rate action_expert_depth_gate vae_ckpt vae-final \
            depth_codes buffer_codes depth_updated_mask LEROBOT_DEPTH_DATA_ROOT \
            DepthAnything Depth-Anything libero_object num_trials_per_task ; do
  grep -rn "$TERM" . --include='*.py' --include='*.md' --include='*.toml' --include='*.yaml' \
       > "/opt/pg/recon/molmoact2/grep_${TERM}.txt" 2>/dev/null || echo "NO_HITS $TERM"
done
wc -l /opt/pg/recon/molmoact2/grep_*.txt

# the curated small files the design is written from
for F in experiments/launch_scripts/data_mixtures.py \
         experiments/launch_scripts/lerobot_utils/train_plan.py \
         experiments/scripts/generate_depth_annotation.py \
         experiments/pyproject.toml \
         experiments/setup.py \
         experiments/README.md \
         experiments/launch_scripts/train_lerobot.py ; do
  if [ -f "$F" ]; then
    mkdir -p "/opt/pg/recon/molmoact2/$(dirname "$F")"
    cp "$F" "/opt/pg/recon/molmoact2/$F"; echo "GOT $F"
  else
    echo "MISS $F"
  fi
done

# every file whose name mentions depth, lerobot eval, or libero — copied if small
find . -path ./.git -prune -o -type f \( -name '*depth*' -o -name '*libero*' -o -name '*eval*' \) \
     -size -400k -printf '%p\n' > /opt/pg/recon/molmoact2/candidate_files.txt
while read -r F; do
  mkdir -p "/opt/pg/recon/molmoact2/tree/$(dirname "$F")"
  cp "$F" "/opt/pg/recon/molmoact2/tree/$F" || true
done < /opt/pg/recon/molmoact2/candidate_files.txt
du -sh /opt/pg/recon/molmoact2

cd /
aws s3 cp /opt/pg/recon/molmoact2/ s3://$B/$PFX/recon/v1/molmoact2/ --recursive --region us-east-1
aws s3 cp /opt/pg/recon/molmoact2_at_${MO2_SHA:0:12}.tar.gz \
          s3://$B/$PFX/src/molmoact2_at_${MO2_SHA:0:12}.tar.gz --region us-east-1
echo "STAGE1_OK molmoact2 at $MO2_SHA"

# ---- stage 2: LIBERO-plus, the benchmark R5 is measured on ------------------------------------
mkdir -p /opt/pg/recon/liberoplus
git clone --quiet https://github.com/sylvestf/LIBERO-plus.git /opt/pg/src/LIBERO-plus
cd /opt/pg/src/LIBERO-plus
git log -1 --format='%H %ad %s' > /opt/pg/recon/liberoplus/HEAD.txt
git rev-parse HEAD > /opt/pg/recon/liberoplus/sha.txt
find . -path ./.git -prune -o -type f -printf '%10s %p\n' > /opt/pg/recon/liberoplus/files.txt
wc -l /opt/pg/recon/liberoplus/files.txt
tar --exclude=.git -czf /opt/pg/recon/liberoplus_src.tar.gz -C /opt/pg/src LIBERO-plus
for F in README.md extra_requirements.txt requirements.txt setup.py pyproject.toml \
         libero/libero/benchmark/task_classification.json \
         libero/libero/benchmark/__init__.py \
         libero/libero/__init__.py ; do
  if [ -f "$F" ]; then
    mkdir -p "/opt/pg/recon/liberoplus/$(dirname "$F")"
    cp "$F" "/opt/pg/recon/liberoplus/$F"; echo "GOT $F"
  else
    echo "MISS $F"
  fi
done
# how many bddl task files it ships, and where
find . -name '*.bddl' | wc -l > /opt/pg/recon/liberoplus/n_bddl.txt
find . -name '*.bddl' | head -20 > /opt/pg/recon/liberoplus/bddl_sample.txt
find . -name '*.bddl' -printf '%h\n' | sort | uniq -c | sort -rn | head -30 \
     > /opt/pg/recon/liberoplus/bddl_dirs.txt
grep -rn "num_trials_per_task" . --include='*.py' > /opt/pg/recon/liberoplus/grep_trials.txt || true
cd /
aws s3 cp /opt/pg/recon/liberoplus/ s3://$B/$PFX/recon/v1/liberoplus/ --recursive --region us-east-1
aws s3 cp /opt/pg/recon/liberoplus_src.tar.gz s3://$B/$PFX/src/liberoplus_src.tar.gz --region us-east-1
echo "STAGE2_OK LIBERO-plus"

# ---- stage 3: the dataset's real layout, meta only -------------------------------------------
/opt/pg/venv/bin/python - "$DATASET" <<'PY' > /opt/pg/recon/dataset_facts.json
import json, sys, collections
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
repo = sys.argv[1]
info = api.dataset_info(repo, files_metadata=True)
sha = info.sha
rows = [{"path": s.rfilename, "size": s.size} for s in info.siblings]
by_dir = collections.Counter()
for r in rows:
    by_dir["/".join(r["path"].split("/")[:-1])] += 1
size_by_top = collections.Counter()
for r in rows:
    size_by_top[r["path"].split("/")[0]] += (r["size"] or 0)
local = snapshot_download(repo, repo_type="dataset", revision=sha,
                          allow_patterns=["meta/*"], max_workers=2)
out = {"repo": repo, "sha": sha, "n_files": len(rows),
       "dirs": dict(sorted(by_dir.items())),
       "bytes_by_top_dir": dict(size_by_top),
       "meta_local": local,
       "video_files": sorted(r["path"] for r in rows if r["path"].endswith(".mp4"))[:12],
       "n_video_files": sum(1 for r in rows if r["path"].endswith(".mp4")),
       "n_parquet_data": sum(1 for r in rows if r["path"].startswith("data/")),
       }
print(json.dumps(out, indent=1))
PY
cat /opt/pg/recon/dataset_facts.json
LOCAL=$(/opt/pg/venv/bin/python -c "import json;print(json.load(open('/opt/pg/recon/dataset_facts.json'))['meta_local'])")
mkdir -p /opt/pg/recon/dataset_meta
cp -r "$LOCAL/meta" /opt/pg/recon/dataset_meta/
find /opt/pg/recon/dataset_meta -type f -printf '%10s %p\n'
# parquet meta as JSON so the control plane never needs pyarrow for it
/opt/pg/venv/bin/python - <<'PY'
import json, pathlib, pyarrow.parquet as pq
root = pathlib.Path("/opt/pg/recon/dataset_meta/meta")
out = {}
for p in sorted(root.rglob("*.parquet")):
    t = pq.read_table(p)
    key = str(p.relative_to(root))
    out[key] = {"n_rows": t.num_rows, "columns": t.column_names,
                "head": t.slice(0, 3).to_pylist()}
pathlib.Path("/opt/pg/recon/dataset_meta/parquet_summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v["columns"] for k, v in out.items()}, indent=1))
PY
aws s3 cp /opt/pg/recon/dataset_facts.json s3://$B/$PFX/recon/v1/dataset_facts.json --region us-east-1
aws s3 cp /opt/pg/recon/dataset_meta/ s3://$B/$PFX/recon/v1/dataset_meta/ --recursive --region us-east-1
echo "STAGE3_OK dataset meta"

# ---- stage 4: the benchmark's own asset release ----------------------------------------------
/opt/pg/venv/bin/python - <<'PY' > /opt/pg/recon/liberoplus_hf.json
import json
from huggingface_hub import HfApi
api = HfApi()
out = {}
for repo, kind in [("Sylvest/LIBERO-plus", "dataset"),
                   ("Sylvest/libero_plus_lerobot", "dataset"),
                   ("Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata", "model")]:
    try:
        info = api.repo_info(repo, repo_type=kind, files_metadata=True)
        out[repo] = {"kind": kind, "sha": info.sha,
                     "files": sorted(({"path": s.rfilename, "size": s.size}
                                      for s in info.siblings),
                                     key=lambda r: -(r["size"] or 0))[:40]}
    except Exception as e:                      # recorded, not hidden: a missing repo is a fact
        out[repo] = {"kind": kind, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(out, indent=1))
PY
cat /opt/pg/recon/liberoplus_hf.json
aws s3 cp /opt/pg/recon/liberoplus_hf.json s3://$B/$PFX/recon/v1/liberoplus_hf.json --region us-east-1
echo "STAGE4_OK benchmark assets listed"

df -h /
touch /opt/pg/recon/DONE
aws s3 cp /opt/pg/recon/DONE s3://$B/$PFX/seed/DONE --region us-east-1
echo "SEED_DONE"
