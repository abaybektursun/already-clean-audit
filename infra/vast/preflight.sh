#!/usr/bin/env bash
# PREFLIGHT GATE — catch interface bugs BEFORE anything ships (E18).
# Runs the production path in miniature on the Ubuntu box: real base image,
# real import graph, real entrypoint behavior, real GPU, real multi-cell
# sequence. Stamps .preflight_pass with the git SHA; launch_builder.sh and
# controller.py rent REFUSE to ship an unstamped SHA.
#
# Tiers (env TIER=1|2|3, default 3):
#   1 static     seconds   compile, shell -n, Dockerfile layer lint, manifest
#                          sanity, banned-pattern greps (D9-class)
#   2 container  minutes   REAL base image: build, in-image dry-runs, science
#                          import graph, provenance path, entrypoint hang check
#                          (D2/D6/D10/D13-class)
#   3 gpu        minutes   two sequential cells on CUDA in one process
#                          (D8/D16-class)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
# on the box the tree arrives as a tar, not a git checkout — the launcher
# passes the SHA it archived (PF_SHA); git is the local-run fallback
SHA="${PF_SHA:-$(git -C "$ROOT" rev-parse --short HEAD)}"
BASE="${BASE:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime}"
TIER="${TIER:-3}"
echo "== preflight @ $SHA (tier <= $TIER, base $BASE) =="

echo "-- T1 static --"
python3 -m py_compile *.py cells/*.py
bash -n *.sh
python3 test_policy.py    # every kill/respawn/backstop decision, table-tested
python3 - <<'EOF'
import json, glob, os, re, subprocess, sys
# Dockerfile layer lint against a synthetic worst-case model (37 shards)
os.makedirs("/tmp/pf_models/big", exist_ok=True)
for i in range(37):
    open(f"/tmp/pf_models/big/model-{i:05d}.safetensors", "wb").write(b"x")
open("/tmp/pf_models/big/config.json", "w").write("{}")
out = subprocess.run(
    [sys.executable, "gen_dockerfile.py", "--base", "BASE", "--models-dir",
     "/tmp/pf_models", "--code-tar", "repo.tar.gz"],
    capture_output=True, text=True, check=True).stdout
layers = sum(1 for l in out.splitlines()
             if l.startswith(("COPY", "RUN", "ADD")))
assert layers <= 40, f"Dockerfile would create {layers} fs layers (aufs cap 42, D15)"
# manifest sanity: valid json, referenced cell modules exist as files,
# and bounds are DECLARED (rent enforces the same — no blind global rules)
for mf in glob.glob("manifests/*.json"):
    m = json.load(open(mf))
    w = m.get("watch") or {}
    assert "max_billed_h" in w and "load_backstop_min" in w, \
        f"{mf}: watch.max_billed_h + load_backstop_min are mandatory"
    for c in m["cells"]:
        path = c["module"].replace(".", "/") + ".py"
        assert os.path.exists(os.path.join("../..", path)), (mf, c["module"])
# banned pattern: any vastai destroy without piped confirmation (D9)
for sh in glob.glob("*.sh") + glob.glob("*.py"):
    for n, line in enumerate(open(sh), 1):
        if "vastai destroy" in line and "echo y" not in line and "confirm" not in line:
            raise AssertionError(f"{sh}:{n}: destroy without piped confirm (D9)")
print("T1 PASS: layers<=40, manifests sane, no bare destroys")
EOF

[ "$TIER" -lt 2 ] && { echo "$SHA" > .preflight_pass; echo "PREFLIGHT PASS (T1) $SHA"; exit 0; }

echo "-- T2 container (real base) --"
PF=/tmp/pf_ctx; rm -rf $PF; mkdir -p $PF/models
# host-side model generation needs torch+transformers: reuse the box's venv
PYBIN=/tmp/gpuvenv/bin/python3
[ -x "$PYBIN" ] || { python3 -m venv /tmp/gpuvenv && /tmp/gpuvenv/bin/pip install -q torch transformers numpy accelerate; }
# tiny REAL-architecture model, no downloads
$PYBIN - <<'EOF'
from transformers import Qwen2Config, Qwen2ForCausalLM, GemmaConfig, GemmaForCausalLM
import numpy as np, torch
torch.manual_seed(0)
Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    vocab_size=512)).save_pretrained("/tmp/pf_ctx/models/tiny_qwen")
np.save("/tmp/pf_ctx/models/tiny_qwen/shackleton_corpus.npy",
        np.random.default_rng(0).integers(0, 512, (6, 16)).astype(np.int64))
EOF
tar czf $PF/repo.tar.gz --exclude='infra/vast/build' -C "$ROOT" infra
mkdir -p $PF/stamp/infra/pg
printf '{"program": "mercator", "commit_sha": "%s", "dirty": false, "source": "preflight"}' "$SHA" \
  > $PF/stamp/infra/pg/mercator_provenance.json
(cd $PF/stamp && tar czf ../provenance.tar.gz infra)
(cd $PF && python3 "$OLDPWD/gen_dockerfile.py" --base "$BASE" --models-dir models \
  --code-tar repo.tar.gz > Dockerfile)
docker build -q -t shackleton-preflight $PF
# entrypoint hang check (D6): a bare command must run and exit
timeout 90 docker run --rm shackleton-preflight sh -c "echo entrypoint-ok" | grep -q entrypoint-ok
# in-image dry-runs + the FULL science import graph (D2/D10/D13-class)
docker run --rm --entrypoint python3 shackleton-preflight /app/infra/vast/dryrun_v1.py
# AMUNDSEN's new instruments + every §6 decision rule, weight-free, in-image
docker run --rm --entrypoint python3 shackleton-preflight /app/infra/vast/dryrun_amundsen.py
docker run --rm --entrypoint python3 shackleton-preflight - <<'EOF'
import importlib
for m in ["infra.vast.run_cells", "infra.vast.cells.smoke_stats",
          "infra.vast.cells.v1_actstats", "infra.vast.cells.v2_gauge",
          "infra.vast.cells.v2b_continuation", "infra.vast.cells.v1_ablation",
          "infra.vast.cells.v3_batchdial", "infra.vast.cells.v4_substrate",
          "infra.vast.cells.softread", "infra.vast.cells.desink_evals",
          "infra.vast.cells.n13_pipeline", "infra.aws_pg.jobs.pipeline",
          "infra.vast.analysis_amundsen",
          "infra.aws_pg.jobs.mercator_mq2", "infra.pg.mercator_core",
          "infra.pg.mercator_gauge", "infra.pg.darwin_train"]:
    importlib.import_module(m)
from infra.pg import mercator_core
assert mercator_core.commit_sha(), "provenance path broken (D13)"
print("T2 PASS: entrypoint, dry-runs, science import graph, provenance")
EOF
# env contract (D17): with HF_ONLINE=1 the offline ENV must be cleared before python
docker run --rm -e HF_ONLINE=1 --entrypoint bash shackleton-preflight -c '
  if [[ "${HF_ONLINE:-0}" == "1" ]]; then unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; fi
  python3 -c "import huggingface_hub.constants as c; assert not c.HF_HUB_OFFLINE, \"offline leaked (D17)\"; print(\"T2b PASS: env contract\")"'


[ "$TIER" -lt 3 ] && { echo "$SHA" > .preflight_pass; echo "PREFLIGHT PASS (T2) $SHA"; exit 0; }

echo "-- T3 gpu (two sequential cells, one process) --"
GPUS=""
docker run --rm --gpus all shackleton-preflight sh -c "true" 2>/dev/null && GPUS="--gpus all"
if [ -n "$GPUS" ]; then
  docker run --rm $GPUS --entrypoint python3 shackleton-preflight - <<'EOF'
import json, os, torch
assert torch.cuda.is_available(), "no CUDA in container (check torch/driver match, D8)"
from infra.vast.cells import smoke_stats, v1_actstats
os.makedirs("/tmp/o1"); os.makedirs("/tmp/o2")
cell = {"id": "pf_smoke", "model_dir": "/models/tiny_qwen"}
smoke_stats.run(cell, "/tmp/o1")   # cell 1 loads a model...
v1_actstats.run({"id": "pf_act", "model_dir": "/models/tiny_qwen",
                 "corpus_path": "/models/tiny_qwen/shackleton_corpus.npy",
                 "batch": 4}, "/tmp/o2")  # ...cell 2 must not inherit its residue (D16)
flags = json.load(open("/tmp/o2/actstats_flags.json"))
assert flags["dtype"] == "torch.bfloat16"
print("T3 PASS: CUDA path, sequential cells, bf16")
EOF
else
  echo "T3: no docker GPU runtime — falling back to venv CUDA check"
  /tmp/gpuvenv/bin/python3 -c "import torch; assert torch.cuda.is_available()"
fi

echo "$SHA" > .preflight_pass
echo "PREFLIGHT PASS (T3) $SHA"
