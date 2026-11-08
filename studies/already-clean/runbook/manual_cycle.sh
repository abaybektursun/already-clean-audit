#!/bin/bash
# Import the answering agents' chunk outputs into a mixed batch and validate them (MANUAL_BACKEND.md, 2026-10-21 note):
# import every chunk_*.out.jsonl of the given chunk dirs (renamed .imported.jsonl afterwards), then `reject --backend mixed`
# moves refused answers aside (their ids are pending again; cap CAP attempts), then prints what is still pending per batch.
# usage: manual_cycle.sh <explain root (holds explain/)> <batch prefix, e.g. w4c> <roles, e.g. clusterer> <chunk dir>...
set -euo pipefail
OUT=$1; B=$2; ROLES=$3; shift 3
R=$(cd "$(dirname "$0")" && pwd)
cd /Users/abaybektursun/projects/interp_research-neurons-explain
for d in "$@"; do
  batch=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['batch'])" "$d/index.json")
  for f in "$d"/chunk_*.out.jsonl; do
    [ -e "$f" ] || continue
    python3 -I "$R/manual_import.py" "$OUT/explain" "$batch" "$f" && mv "$f" "${f%.out.jsonl}.imported.jsonl"
  done
done
"$R/ne" explain reject --out "$OUT" --batch "$B" --roles "$ROLES" --backend mixed --max-attempts "${CAP:-8}" 2> >(grep -v 'PyTorch was not found' >&2)
python3 -I - "$OUT/explain" "$B" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); prefix = sys.argv[2]
for req in sorted(d.glob(f"requests/{prefix}-*.jsonl")):
    b = req.stem; done = set()
    for suf in (".jsonl", ".declined.jsonl"):
        p = d / "responses" / (b + suf)
        if p.exists(): done |= {json.loads(l)["id"] for l in open(p)}
    rows = [json.loads(l) for l in open(req)]
    rej = d / "responses" / (b + ".rejected.jsonl"); nrej = sum(1 for _ in open(rej)) if rej.exists() else 0
    print(f"{b}: {len(rows)} requests, {len(rows) - len([r for r in rows if r['id'] not in done])} answered or declined, "
          f"{len([r for r in rows if r['id'] not in done])} pending, {nrej} rejected lines so far")
PY
