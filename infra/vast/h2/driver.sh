#!/usr/bin/env bash
# HELMHOLTZ-2 full-FT AdamW ladder on a Vast 80GB box. Fail-loud; no AWS creds here.
# Env (written by the launcher into /workspace/h2env): HF_TOKEN, BIND_URL, PUT_TGZ_URL, PUT_DONE_URL
set -euxo pipefail
source /workspace/h2env
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repo
python -c "import torch; assert torch.cuda.get_device_properties(0).total_memory > 75e9, 'need 80GB GPU'"
# NET PROBE first (the network lottery): 50MB ranged pull from HF; abort cheap if host is throttled.
SPEED=$(curl -sL -H "Authorization: Bearer $HF_TOKEN" --range 0-52428800 -o /dev/null -w '%{speed_download}' --max-time 120 "https://huggingface.co/google/gemma-3-4b-pt/resolve/main/model-00001-of-00002.safetensors" || echo 0)
SPEED=${SPEED%.*}
echo "NET_PROBE bytes/s=$SPEED"
if [ "$SPEED" -lt 10485760 ]; then
  echo '{"rc": 9, "reason": "slow-net", "speed": '"$SPEED"'}' > /tmp/done.json
  curl -fsS -X PUT -H "Content-Type: application/json" --upload-file /tmp/done.json "$PUT_DONE_URL"
  echo DRIVER_ABORT_SLOW_NET; exit 0
fi
pip install -q "transformers==5.9.0" "datasets==4.8.5" "accelerate>=1.10,<2" "safetensors>=0.5,<1" hf_transfer > /tmp/pip.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/out
rc=0
for dose in ${DOSES:-5000000 20000000 60000000}; do
  name="fullft_know_${dose}${TAG:-}"
  python -m infra.aws_pg.jobs.helmholtz2_arm --arm know --full-ft --dose-tokens "$dose" --seed 0 ${EXTRA_ARGS:-} \
    --bind-s3 "$BIND_URL" --out "/workspace/out/$name" 2>&1 | tee "/workspace/out/$name.log" || { rc=$?; break; }
  # early exit on gate pass: qualifying dose found, higher rungs unnecessary (saves money)
  python - "$name" << 'PYEOF'
import json, sys, pathlib
m = json.loads((pathlib.Path("/workspace/out")/sys.argv[1]/"metrics.json").read_text())
raise SystemExit(0 if not m["margins"]["gate_pass"] else 7)
PYEOF
  gate=$?; [ "$gate" = "7" ] && break
done
tar czf /workspace/h2_fullft_vast.tgz -C /workspace out
curl -fsS -X PUT -H "Content-Type: application/octet-stream" --upload-file /workspace/h2_fullft_vast.tgz "$PUT_TGZ_URL"
echo "{\"rc\": $rc, \"done_utc\": \"$(date -u +%FT%TZ)\"}" > /tmp/done.json
curl -fsS -X PUT -H "Content-Type: application/json" --upload-file /tmp/done.json "$PUT_DONE_URL"
echo DRIVER_DONE rc=$rc
