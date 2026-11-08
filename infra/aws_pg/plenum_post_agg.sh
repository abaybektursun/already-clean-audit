#!/usr/bin/env bash
# Post-aggregation closeout: render the MAP, regenerate the figures, run the completeness gate.
#
# Split out from `plenum_closeout.sh` because that script runs its OWN aggregation, and the
# aggregation is now a ~2-hour CPU job (corpus CIDEr at 200 bootstrap resamples over every unit with
# caption rows). Re-running it just to redraw a figure is the wrong shape.
#
# The figures are regenerated here rather than trusted, because they go stale silently and have
# already done so once: `figures_manifest.json` recorded F1 drawn at pending=737 — the FIRST
# aggregation, before ~700 cells were measured — and F2, the LAW figure, skipped for "no behavioural
# cells with a label". The program's headline result had no figure at all while the manifest sat there
# looking complete. That is the post-critic artifact desync failure mode
# (`memory/feedback_post_critic_artifact_desync.md`), so figure regeneration is part of closeout, not
# an optional follow-up.
#
#   ./infra/aws_pg/plenum_post_agg.sh <agg-prefix>       # e.g. plenum_agg_ah
set -eu
cd "$(dirname "$0")/../.."
AGG="${1:?usage: plenum_post_agg.sh <agg-prefix>}"
B=s3://pg-lineage-769063704696/results
W=/tmp/plenum_post/$AGG
mkdir -p "$W"

st=$(aws s3 cp "$B/$AGG/STATUS" - --profile fuelos 2>/dev/null || echo "MISSING")
echo "[post] $AGG STATUS=$st"
[ "$st" = "exit=0" ] || { echo "[post] refusing to render from a non-clean aggregation" >&2; exit 1; }

for f in metrics.json cells.json plenum_prediction_score.json; do
  aws s3 cp "$B/$AGG/out/$f" "$W/$f" --profile fuelos >/dev/null 2>&1 || echo "[post] (no $f)"
done
aws s3 cp "$B/plenum_freeze2/out/plenum_predictions.json" "$W/predictions.json" \
  --profile fuelos >/dev/null 2>&1 || true
aws s3 cp "$B/plenum_patch/out/metrics.json" "$W/patch.json" --profile fuelos >/dev/null 2>&1 || true

echo "[post] rendering MAP"
python3 scripts/plenum_map_render.py --records "$W/cells.json" --out studies/lineage/MAP.md

echo "[post] figures"
FIGS=studies/lineage/paligemma1-pi0/research/plenum/figs
python3 -m infra.aws_pg.jobs.plenum_figures --out "$FIGS" \
  --agg "$W/metrics.json" --cells "$W/cells.json" \
  $([ -f "$W/patch.json" ] && echo "--patch $W/patch.json") \
  $([ -f "$W/predictions.json" ] && echo "--predictions $W/predictions.json")

echo "[post] figures manifest:"
cat "$FIGS/figures_manifest.json"
# A manifest that records a SKIP for the law figure is not a rendered result set. F2 is the headline
# (the n=26 fit); if it skipped, say so loudly rather than committing a directory that looks complete.
python3 - "$FIGS/figures_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
skipped = {k: v["skipped"] for k, v in m.items() if isinstance(v, dict) and "skipped" in v}
if skipped:
    print(f"[post] WARNING: figures skipped: {skipped}")
    print("[post] F2 is the law figure — a skip there means the headline result has no figure.")
else:
    print("[post] all figures rendered")
PY

echo "[post] completeness gate (non-zero exit = the run is NOT done)"
python3 scripts/plenum_map_render.py --records "$W/cells.json" --require-complete || true
