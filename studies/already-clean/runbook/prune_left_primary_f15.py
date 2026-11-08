"""Take the judge lines of clusters that left the late-born primary set out of judgments.jsonl and plans/ (W4-W5 F15).

usage (from the explain worktree, PYTHONPATH=.): python prune_left_primary_f15.py <explain root holding explain/> <judge batch, e.g. w4j> <primary_set_diff json>

After sample-1 clusters were re-answered, a sample-0 cluster judged in the first batch can stop being late-born primary
(status unmatched_in_sample_1). neurons_verdict requires every planned judge line to have a judgment row and the judged
clusters to equal the late-born primary clusters, so such a cluster's plan rows and judgment rows are moved aside together:
judgments.jsonl -> judgments.left_primary_F15_<utc>.jsonl and plans/<batch>-judge.jsonl -> plans/<batch>-judge.left_primary_F15_<utc>.jsonl
(a name the plans glob *-judge.jsonl does not match). The previous files are kept as *.before_prune_<utc>.jsonl; the requests,
responses and calls.jsonl rows stay where they are. The manifest records the clusters and counts under f15_left_primary.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import infra.pg.neurons_llm as llm


def main(root, batch, diff_json):
    explain = Path(root) / "explain"
    leaving = set(json.load(open(diff_json))["leaving"])
    if not leaving:
        print("no cluster left the late-born primary set; nothing to prune")
        return
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jpath, ppath = explain / "judgments.jsonl", explain / "plans" / f"{batch}-judge.jsonl"
    judgments, plans = llm.read_jsonl(jpath), llm.read_jsonl(ppath)
    j_out = [r for r in judgments if not r["control"] and r["cluster"] in leaving]
    p_out = [r for r in plans if not r["control"] and r["cluster"] in leaving]
    assert {r["cluster"] for r in p_out} == leaving, sorted(leaving - {r["cluster"] for r in p_out})
    assert {r["id"] for r in j_out} == {r["id"] for r in p_out}, "judgment rows and plan rows of the leaving clusters differ"
    kept = {}
    for path, rows, out, tag in ((jpath, judgments, j_out, "judgments"), (ppath, plans, p_out, "plans")):
        before = path.with_name(path.name.replace(".jsonl", f".before_prune_{utc}.jsonl"))
        aside = path.with_name(path.name.replace(".jsonl", f".left_primary_F15_{utc}.jsonl"))
        assert not before.exists() and not aside.exists()
        gone = {r["id"] for r in out}
        tmp = path.with_name(path.name + ".tmp")
        llm.write_jsonl_new(tmp, [r for r in rows if r["id"] not in gone])
        llm.write_jsonl_new(aside, out)
        os.replace(path, before)
        os.replace(tmp, path)
        kept[tag] = {"before": before.name, "aside": aside.name, "rows_moved": len(out)}
    m = json.load(open(explain / "manifest.json"))
    m.setdefault("f15_left_primary", {})[utc] = {"clusters": sorted(leaving), **kept}
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({"clusters": len(leaving), **kept}))


if __name__ == "__main__":
    main(*sys.argv[1:])
