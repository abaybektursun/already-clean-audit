"""Which sample-0 clusters are late-born primary under the current clusters.jsonl, against the judged set (W4-W5 F15).

usage (from the explain worktree, PYTHONPATH=.): python primary_set_diff.py <explain dir> <grid dirs comma-separated, T_15 last> <out json>
Computes the cluster tables with neurons_verdict.grid_cluster_tables (the unblind's own code), lists the late_primary
clusters, and compares them with the clusters that hold judge lines in plans/*-judge.jsonl: `entering` (late-born primary now,
never judged) and `leaving` (judged, no longer late-born primary). Writes the lists and the status counts to <out json>.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import infra.pg.neurons_verdict as v


def main(explain, grid_csv, out):
    explain = Path(explain)
    tables = v.grid_cluster_tables(grid_csv.split(","), explain)
    rows = [r for t in tables.values() for r in t if not r["echo"]]
    primary = sorted(r["cluster"] for r in rows if r["status"] == "late_primary")
    judged = sorted({p["cluster"] for p in v.judge_plans(explain) if not p["control"]})
    entering = sorted(set(primary) - set(judged))
    leaving = sorted(set(judged) - set(primary))
    res = {"n_late_primary": len(primary), "n_judged": len(judged), "entering": entering, "leaving": leaving,
           "status_counts": dict(Counter(r["status"] for r in rows)),
           "t_before": {r["cluster"]: r["t_before"] for r in rows if r["cluster"] in set(entering)}}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_late_primary", "n_judged", "status_counts")}),
          "entering", len(entering), "leaving", len(leaving))


if __name__ == "__main__":
    main(*sys.argv[1:])
