#!/usr/bin/env bash
# GPU smoke of the V1 instrument on the Ubuntu box's RTX 4090 Laptop (16 GB) —
# validates the CUDA path (bf16 hooks, fp64 GPU accumulators, dtype asserts)
# on a real small Qwen before any vast rental. Run via run_on_ubuntu.sh.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1

python3 -m venv /tmp/gpuvenv 2>/dev/null || true
/tmp/gpuvenv/bin/pip install -q -U torch transformers accelerate numpy datasets huggingface_hub
/tmp/gpuvenv/bin/python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA on box'"

mkdir -p build/models
/tmp/gpuvenv/bin/hf download Qwen/Qwen2.5-0.5B \
  --local-dir build/models/Qwen_Qwen2.5-0.5B \
  --exclude "*.bin" --exclude "*.pth" --exclude "*.gguf"
/tmp/gpuvenv/bin/python3 prepare_corpus.py build/models/Qwen_Qwen2.5-0.5B

OUT=$(mktemp -d)
PYTHONPATH=../.. /tmp/gpuvenv/bin/python3 - "$OUT" <<'EOF'
import json, sys
from infra.vast.cells import v1_actstats
out = sys.argv[1]
cell = {"id": "gpu_smoke_v1", "model_dir": "build/models/Qwen_Qwen2.5-0.5B", "batch": 8}
v1_actstats.run(cell, out)
flags = json.load(open(f"{out}/actstats_flags.json"))
assert flags["dtype"] == "torch.bfloat16", flags["dtype"]
rows = flags["per_site"]
assert all(r["tokens"] == flags["n_windows"] * flags["window"] for r in rows)
print(f"layers={flags['n_layers']} sites={len(rows)} "
      f"corpus={flags['corpus_sha256'][:12]}")
print("pos0_share by layer (q_proj_in):",
      [round(r["pos0_share"], 3) for r in rows if r["site"] == "q_proj_in"][:8], "...")
print("GPU-SMOKE-V1 PASS: CUDA bf16 path, hooks, fp64 accumulators, asserts")
EOF
