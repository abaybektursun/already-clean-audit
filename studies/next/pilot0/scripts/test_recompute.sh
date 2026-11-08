#!/usr/bin/env bash
# Dry checks of recompute.py on banked files (no model, no new run). Run from the repo root:
#   PY=<python with numpy> studies/next/pilot0/scripts/test_recompute.sh > studies/next/pilot0/scripts/test_output.txt 2>&1
# Known-number inputs: the first 32 rows of the banked pilot-0 eager and sdpa files of batch_60001/, written to a
# temporary directory (rows_done 32 and the per-row lists cut to 32; every other field unchanged).
set -u
P=studies/next/pilot0
S=$P/scripts
T=$(mktemp -d)
"$PY" -I - "$P" "$T" <<'EOF'
import json, sys
p, t = sys.argv[1:3]
for att in ("eager", "sdpa"):
    r = json.load(open(f"{p}/batch_60001/loss_check_{att}.json"))
    r["rows_done"] = 32
    for k in ("per_row_mean_ce", "per_row_top1"):
        r[k] = r[k][:32]
    r["pad"]["per_row_pad_target_positions"] = r["pad"]["per_row_pad_target_positions"][:32]
    json.dump(r, open(f"{t}/{att}32.json", "w"))
EOF
echo "== test 1: b1 320-row neighbour, eager 32 rows as the new known-number run (expect: N 1.9455010, does not count by rows; gate fails)"
"$PY" -I $S/recompute.py $P/batch_100000/loss_check_sdpa_b1_320rows.json $P/batch_100000/tokens.npy \
    $P/batch_100001/loss_check_sdpa.json $T/eager32.json $P/batch_60001/loss_check_sdpa.json
echo "== test 2: identity, banked sdpa 32 rows against itself; target file as its own neighbour (expect: gate holds, diff 0; dS = dN both within 0.0124 -> does not count)"
"$PY" -I $S/recompute.py $P/batch_100001/loss_check_sdpa.json $P/batch_100001/tokens.npy \
    $P/batch_100001/loss_check_sdpa.json $T/sdpa32.json $P/batch_60001/loss_check_sdpa.json
echo "== test 3: wrong tokens.npy for the neighbour, batch_100001/ tokens with the batch_100000/ file (expect: assert on non-pad counts)"
"$PY" -I $S/recompute.py $P/batch_100000/loss_check_sdpa_b1_320rows.json $P/batch_100001/tokens.npy \
    $P/batch_100001/loss_check_sdpa.json $T/sdpa32.json $P/batch_60001/loss_check_sdpa.json 2>&1 | tail -1
rm -r "$T"
