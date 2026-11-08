#!/usr/bin/env bash
# N13 image gate: runs INSIDE the built image (build_image.sh GATE=n13, before the push). Inputs at /gate_in:
# expected_freeze.txt (studies/next/n13/r0/s2setup-07/setup/pip_freeze_n13.txt, the venv that ran on AWS) and the
# offline gate's small files (n13_gate_hf/, mathtask_olmes/olmes/, mathtask_wandb/). N13_MODELS = the baked repo@commit list.
# CODE = the code tree the offline gate runs (default /app, the image's code layer; launch_preflight.sh mounts the current commit).
# Prints N13_IMAGE_GATE_OK only when every check holds. Fail-loud: set -e, no fallback.
set -euo pipefail
V=/opt/n13/bin/python
CODE="${CODE:-/app}"

echo "-- 1. the n13 venv holds exactly the AWS freeze"
$V -m pip freeze > /tmp/freeze_image.txt
diff <(sort -f /gate_in/expected_freeze.txt) <(sort -f /tmp/freeze_image.txt)
echo "GATE freeze: $(wc -l < /tmp/freeze_image.txt) lines, equal to the AWS freeze"
$V -m pip check

echo "-- 2. what Triton and Inductor compile with (s2smoke-08 smoke_base died on a missing Python.h)"
gcc --version | head -1
inc=$($V -c "import sysconfig; print(sysconfig.get_paths()['include'])")
test -f "$inc/Python.h"
echo "GATE Python.h at $inc"

echo "-- 3. the engine stack imports; A21's variable is set for every process"
test "${VLLM_USE_FLASHINFER_SAMPLER:-}" = "0"
$V -c "import vllm, math_verify, torch, transformers; print('GATE vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
python3 -c "import torch, numpy, huggingface_hub, pyarrow; print('GATE parent python', torch.__version__, numpy.__version__, huggingface_hub.__version__, pyarrow.__version__)"

echo "-- 3b. GPU code compiled into the engine stack: Ada (L40S, sm_89; sm_80/86 code runs there) and Hopper (H100, sm_90/90a)"
$V -c "import torch; f = torch._C._cuda_getArchFlags(); print('GATE torch arch', f); assert 'sm_90' in f and any(a in f for a in ('sm_80', 'sm_86', 'sm_89')), f"
CUOBJ=$($V -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin', 'cuobjdump'))")
test -x "$CUOBJ"
$V - "$CUOBJ" <<'PY'
import glob, os, re, subprocess, sys
import vllm
root = os.path.dirname(vllm.__file__)
# vLLM's compiled extensions (names differ by release: _C*.so, _moe_C*.so, vllm_flash_attn/_vllm_fa2_C / _vllm_fa3_C ...)
libs = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.so"), recursive=True)) if "_C" in os.path.basename(p)]
assert libs, f"no compiled extension under {root}"
seen = {}
for p in libs:
    r = subprocess.run([sys.argv[1], "--list-elf", p], capture_output=True, text=True)
    arch = sorted(set(re.findall(r"sm_\d+a?", r.stdout)))
    seen[os.path.relpath(p, os.path.dirname(root))] = arch
    print("GATE", os.path.relpath(p, os.path.dirname(root)), "rc", r.returncode, arch, r.stderr.strip()[:120])
allarch = {a for v in seen.values() for a in v}
assert any(a.startswith("sm_90") for a in allarch) and any(a in allarch for a in ("sm_80", "sm_86", "sm_89")), seen
fa = {k: v for k, v in seen.items() if "fa" in os.path.basename(k)}
print("GATE flash-attention:", fa)
PY
echo "-- 4. the baked revisions (n13_setup.MODELS) sit where snapshot_download(repo, revision=commit) looks"
for m in ${N13_MODELS:?}; do
  repo=${m%@*}; rev=${m#*@}
  d="$HF_HUB_CACHE/models--${repo//\//--}/snapshots/$rev"
  n=$(ls "$d"/*.safetensors | wc -l)
  test "$n" -eq 3
  echo "GATE $repo@$rev: $n shards, $(du -shL "$d" | cut -f1)"
done

echo "-- 5. the N13 offline gate under the venv's interpreter (matplotlib, for figure_n13 only, in an overlay dir)"
$V -m pip install -q --no-deps --target /tmp/gate_extra matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 \
  fonttools==4.68.0 kiwisolver==1.5.1 pyparsing==3.3.3 python-dateutil==4.0.0
cd "$CODE"
# the dry run exits 1 on any failed check; its record (evidence/checks.json) is judged below, so its exit code is not
set +e
PYTHONPATH="$CODE:/tmp/gate_extra" $V -m infra.aws_pg.jobs.n13_dryrun --out /tmp/n13_dry --hf /gate_in/n13_gate_hf \
  --olmes /gate_in/mathtask_olmes/olmes --wandb /gate_in/mathtask_wandb \
  --task-doc studies/next/07_PICKS/N13_MATH_TASK.md > /tmp/n13_dry.log 2>&1
set -e
grep -E " FAIL|^Traceback|Error" /tmp/n13_dry.log | head -20 || true
tail -25 /tmp/n13_dry.log
# The laptop gate has no vLLM. One of its checks needs that: A21's failing input expects "No module named 'vllm'" once the
# variable is 0. Here vLLM is installed, so that probe goes past the import and stops later. That check, and only it, is
# judged by the in-image form: unset and 1 stop on the A21 assertion, 0 passes it.
$V - <<'PY'
import json
c = json.load(open("/tmp/n13_dry/evidence/checks.json"))
bad = [x for x in c["checks"] if not x["ok"]]
print(f"GATE offline gate: {c['ok']}/{c['n']} checks ok")
for x in bad:
    print("GATE failed check:", x["check"][:160], "|", json.dumps(x["detail"], default=str)[:600])
other = [x for x in bad if not x["check"].startswith("A21 failing input")]
assert not other, f"offline gate failed on {len(other)} checks besides the vLLM-absence probe"
if bad:
    runs = json.load(open("/tmp/n13_dry/evidence/A21_engine_env.json"))["probe"]
    assert all(runs[k]["returncode"] != 0 and "AssertionError: A21" in runs[k]["last_line"] for k in ("unset", "set_to_1")), runs
    assert "AssertionError: A21" not in runs["set"]["last_line"], runs
    print("GATE A21 in the image: unset and 1 stop on the A21 assertion; 0 passes it:", runs["set"]["last_line"][:300])
print("GATE offline gate judged:", "N13_DRYRUN_OK" if not bad else f"{c['ok']}/{c['n']}, the one vLLM-absence check judged in the in-image form")
PY
echo "N13_IMAGE_GATE_OK"
