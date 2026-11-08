#!/bin/bash
# TESSERA-FIX closeout — runs once C1's sibling band is complete.
#
# Every step applies a rule frozen BEFORE its data existed:
#   * tessera_c1_verdict implements spec §6 verbatim and was dry-run against all three of its
#     outcomes (tight band -> SUPPORTED, wide band -> NOT SUPPORTED, degenerate shuffle -> BAND VOID)
#     before C1 produced a single number. It VOIDs a cell that has no band at its OWN site rather
#     than borrowing the pooled band.
#   * the pre-committed consequences (spec §1.2) are not re-decidable here:
#       pi0 LOST inside the band at EITHER site      -> Result 6 downgrades to within-reference-noise
#       install emergent @L14 inside the band        -> Result 5 retracts entirely
#
# Ordering is the spec's: verdict -> docs -> MAP -> registries. Registries last, because a
# downgrade would invalidate rows written early (and the ordering was already violated once,
# recorded in Result 10).
set -euo pipefail
cd "$(dirname "$0")/../../.."
A="--profile fuelos --region us-east-1"
B=s3://pg-lineage-769063704696/results
ART=studies/lineage/paligemma1-pi0/research/artifacts
# The merged Arm X artifact is read from the REPO, not /tmp -- a closeout that depends on
# scratch files is a closeout that stops working the moment the machine is rebooted.
XM=$ART/tessera_x_merged_metrics.json

test -f "$XM" || { echo "missing $XM (the merged Arm X artifact)"; exit 1; }

echo "=== 1. artifacts ==="
aws s3 cp $B/tessera_c1_sibling/out/metrics.json /tmp/tc1.json $A --only-show-errors
cp /tmp/tc1.json $ART/tessera_c1_sibling_metrics.json
python3 -c "
import json; d=json.load(open('/tmp/tc1.json'))
s=d['pairs']['sibling']['sites']
done=[L for L,v in s.items() if 'seeds' in v]
print('sibling sites with seeds:', sorted(done, key=int))
for L in sorted(s, key=int):
    v=s[L]
    if 'seeds' not in v: print(f'  L{L}: INCOMPLETE'); continue
    print(f\"  L{L} emergent={[round(q['emergent_fraction'],5) for q in v['seeds']]}\"
          f\" LOST={[round(q['lost_fraction'],5) for q in v['seeds']]}\"
          f\" shuffle_em={[round(z['emergent_fraction'],4) for z in v.get('shuffle_null',[])]}\")
"

echo
echo "=== 2. the frozen re-verdict (spec §6) ==="
python3 -m infra.aws_pg.jobs.tessera_c1_verdict --out /tmp/tc1v --x "$XM" --c1 /tmp/tc1.json
cp /tmp/tc1v/metrics.json $ART/tessera_c1_verdict_metrics.json
aws s3 cp /tmp/tc1v/metrics.json $B/tessera_c1_verdict/out/metrics.json $A --only-show-errors || true

echo
echo "=== 3. structural check on the sibling band (is its zero MEASURED or structural?) ==="
python3 -m infra.aws_pg.jobs.tessera_c0 --out /tmp/tc0b --x /tmp/tc1.json 2>/dev/null | tail -6 || \
  echo "(c0 expects the merged Arm X schema; the sibling reach was checked inline in the journal)"

echo
echo "=== 4. gates ==="
python3 -m infra.aws_pg.jobs.tessera_fix_check --root . --out $ART/fix_check || true
python3 -m infra.aws_pg.jobs.tessera_consistency --root . --out $ART/consistency || true

echo
echo "=== 5. REMAINING BY HAND (deliberately not scripted) ==="
cat <<'NOTE'
  a. fold the C1 verdict into Result 5 / Result 6 / grid rows 3-5 of the parent doc, and FIX-7 of
     the fix doc -- wording depends on which branch landed, so it is written, not templated;
  b. regenerate figures (F4 gains the sibling band);
  c. MAP final pass: the C1 reference band + re-verdicted Arm-X cells (spec §4.3);
  d. RESEARCH_INDEX + README ledger TESSERA rows (spec §3 row 14) -- ONLY after (a)-(c);
  e. re-run both gates; finding 3 and finding 14 should then close.
NOTE
