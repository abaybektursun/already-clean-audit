#!/usr/bin/env bash
# CPU EVAL-SUBSET seed (EC2 user-data, runs as root on Amazon Linux 2023 arm64).
#
# WHY THIS EXISTS (2026-09-03). The dataset seed of the same day cached the SPLITS the O column reads
# and measured what that costs: 181.8 GB of images (VQAv2 155.4, TextVQA 21.0, RefCOCO 5.4) against
# 434 MB of text. 181.8 GB does not fit a 200 GB job volume beside a row's weights, so the image half
# was gated off and unused, and every O job kept pulling those splits from the hub at ~2.8 MB/s —
# two jobs each spent over an hour before their first GPU stage. The cause is upstream, in
# `matrix_items._select_rows`: `load_dataset(repo, split=...).select(idx)` MATERIALISES the split to
# read 200 rows (214,354 image-bearing rows for VQAv2).
#
# WHAT THIS SEEDS: the 200 rows the frozen index lists actually score, images included, one
# `datasets.save_to_disk` directory per task, to s3://$B/datasets/eval_subsets/. Tens of MB per task
# instead of tens of GB. `run_job.sh` restores them into /opt/hf-cache/eval_subsets and
# `matrix_items.vision_rows` reads them; with the subset absent, everything falls back to the split.
#
# CORRECTNESS IS THE POINT, NOT SPEED. This is a FROZEN pre-registered item set: scoring different
# rows quickly would be far worse than scoring the right rows slowly. So the hashes come from the
# CANONICAL path (`_select_rows` over the materialised split, on this box, from the same split the
# subset is cut from), every written row is re-hashed against them before the subset is published,
# and the restore probe below re-verifies from a FRESH S3 restore with no split on disk at all.
# `build_vision_subset` raises on any mismatch; nothing here repairs, and nothing is try/except'd.
#
# n_split IS NOT TAKEN FROM THE NETWORK. `_hf_split_rows` reads hub metadata; here the split sizes
# are pinned from the dataset seed's own BUILDER lines (i-0d8f0f3593f47ee97, 2026-09-03) and
# `build_vision_subset` asserts the materialised split really holds that many rows — a stronger check
# than the metadata, and it lets the whole build run OFFLINE.
#
# WHY CPU / WHY m7g.medium: identical to seed_weights_cpu.sh and seed_datasets_cpu.sh — restore,
# select, upload; no GPU; and it must not touch the 32-vCPU G quota the science fills. The account's
# Standard on-demand quota is 1 vCPU (verified 2026-09-03), so m7g.medium is the only shape that fits.
#
# Self-terminates via the EXIT trap + InstanceInitiatedShutdownBehavior=terminate; an EventBridge
# one-shot reaper is the backstop.
exec > /var/log/pg-seed-subsets.log 2>&1
B=pg-lineage-769063704696
S3SUB=s3://$B/datasets/eval_subsets
up() { aws s3 cp /var/log/pg-seed-subsets.log $S3SUB/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

( while true; do aws s3 cp /var/log/pg-seed-subsets.log $S3SUB/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /
dnf install -y python3.11 python3.11-pip tar
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
# PINNED TO THE AMI (bake_env.sh) and to the dataset seed: the arrow cache key is a function of the
# library's builder-config fingerprint, so a version mismatch would miss the cache this restores.
# Pillow is NOT a datasets dependency, and without it `ds[k]` raises on the Image column — the first
# run of this script built and published three correct subsets and then died in its own restore probe
# on exactly that (2026-09-03, i-005eb0e5d8a508495). The GPU AMI has it via torchvision; this box did not.
/opt/pg/venv/bin/pip install 'datasets==4.8.5' 'huggingface_hub==1.16.1' 'numpy<3' pillow

# the science code, exactly as the launcher tarred it (infra/ only; matrix_items needs numpy alone)
mkdir -p /opt/pg/repo
aws s3 cp s3://$B/seed/eval_subsets_code.tgz /tmp/code.tgz --region us-east-1
tar -xzf /tmp/code.tgz -C /opt/pg/repo
test -f /opt/pg/repo/infra/pg/matrix_items.py

# IDEMPOTENT: with the subsets already published, skip the 181.8 GB restore and the build and go
# straight to the restore probe. That is the only way to re-run the probe after it fails, and after
# the eval-split half of the seed is deleted the build cannot run again at all.
BUILD=1
aws s3 ls $S3SUB/vision_eval_index.json --region us-east-1 && BUILD=0 || true
if [ "$BUILD" = "1" ]; then
# ---- restore the seeded IMAGE splits (this is the last use they will ever get) ---------------
mkdir -p /opt/pg/hfds
time aws s3 sync s3://$B/datasets/hf/ /opt/pg/hfds/ --exclude 'xet/*' --region us-east-1
du -sh /opt/pg/hfds; df -h /

# ---- build + verify, OFFLINE ------------------------------------------------------------------
export HF_HOME=/opt/pg/hfds HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=/opt/pg/repo
mkdir -p /opt/pg/subsets
/opt/pg/venv/bin/python - <<'PY'
import json, os
from infra.pg import matrix_items as MI

# split sizes from the dataset seed's BUILDER lines (i-0d8f0f3593f47ee97, 2026-09-03); asserted
# against the materialised split inside build_vision_subset, so a wrong pin cannot pass silently.
PINNED = {("lmms-lab-encoder/RefCOCO", "testA"): 1975,
          ("lmms-lab-encoder/textvqa", "validation"): 5000,
          ("lmms-lab-encoder/VQAv2", "validation"): 214354}
frozen = {}
for task in MI.VISION:
    out = f"/opt/pg/subsets/{task}"
    rec = MI.build_vision_subset(task, out, split_rows=lambda r, s: PINNED[(r, s)])
    n = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"BUILT {task}: {rec['n']} rows, n_split {rec['n_split']}, {n/1e6:.1f} MB, "
          f"hash-match {len(rec['hashes'])}/{rec['n']}", flush=True)
    frozen[task] = rec
with open("/opt/pg/subsets/vision_eval_index.json", "w") as f:
    json.dump(frozen, f, indent=1)
print("BUILD_OK", flush=True)
PY
du -sh /opt/pg/subsets/*

# ---- publish ----------------------------------------------------------------------------------
aws s3 sync /opt/pg/subsets/ $S3SUB/ --region us-east-1
aws s3 ls $S3SUB/ --recursive --summarize | tail -3

else
  echo 'subsets already published at '$S3SUB' — skipping the restore and the build, probing only'
fi

# ---- restore probe: a FRESH S3 copy must load and re-verify with NO split on disk --------------
# The dataset seed's own probe never ran (182 GB cannot be restored beside the 175 GB it was built
# from on one 300 GB volume), so its blob-less layout was never proved. This one can run, because the
# thing being proved is 100 MB. Free the splits first: that is also what a job's volume looks like.
du -sh /opt/pg/hfds || true; rm -rf /opt/pg/hfds /opt/pg/subsets; df -h /
mkdir -p /opt/hf-cache/eval_subsets /opt/pg/emptyhf
time aws s3 sync $S3SUB/ /opt/hf-cache/eval_subsets/ --exclude 'seed.log' --region us-east-1
du -sh /opt/hf-cache/eval_subsets
# the index the job will read is the one that was just published — this is the file to commit
cp /opt/hf-cache/eval_subsets/vision_eval_index.json /opt/pg/repo/infra/pg/vision_eval_index.json
HF_HOME=/opt/pg/emptyhf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=/opt/pg/repo \
  PG_EVAL_SUBSETS=/opt/hf-cache/eval_subsets /opt/pg/venv/bin/python - <<'PY'
import json, time
from infra.pg import matrix_items as MI

frozen = MI.frozen_vision_index()
t0 = time.time()
vis = MI.vision_index_lists()                      # the exact call the O column makes
assert all(v["hash_path"] == "subset-cached" for v in vis.values()), \
    {t: v["hash_path"] for t, v in vis.items()}
for task, v in vis.items():
    assert v["hashes"] == frozen[task]["hashes"] and v["indices"] == frozen[task]["indices"], task
    rows = MI.vision_rows(task)                    # verifies again, and decodes the images
    sizes = [r["image"].size for r in rows]
    assert len(rows) == 200 and all(w > 0 and h > 0 for w, h in sizes), task
    print(f"PROBE {task}: 200 rows, hash-match {len(v['hashes'])}/200, "
          f"images {min(sizes)}..{max(sizes)}, index0 {v['indices'][0]}, hash0 {v['hashes'][0]}", flush=True)
print(f"PROBE elapsed {time.time() - t0:.1f}s")
print("SUBSET_RESTORE_OK", flush=True)
PY

date -u > /tmp/DONE
aws s3 cp /tmp/DONE $S3SUB/DONE --region us-east-1
up
