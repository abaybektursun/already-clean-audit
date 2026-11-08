"""Prepare a supplementary judge batch for a given set of late-born primary clusters (W4-W5 F15, 2026-10-24).

usage (from the explain worktree, PYTHONPATH=.):
  python prepare_judge_subset.py <explain root holding explain/> <grid csv, T_15 last> <corpus dir> <batch, e.g. w4j2> <model> <seed> <gate dir> <clusters file (one cluster id per line)>

The same construction as neurons_explain.cmd_prepare_judge (§2.4 parent test: the cluster's record at t_before against two
random same-layer decoys and the hard decoy, two judge samples, shown order drawn per call, exemplar windows removed from both
records), restricted to the listed clusters and without controls (every pole's control was judged in the first batch).
Writes requests/<batch>-judge-s{0,1}.jsonl and plans/<batch>-judge.jsonl, records the seed and the gate in manifest.json.
Refuses a cluster that is not late-born primary under the current clusters.jsonl or that already has judge lines.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import infra.pg.neurons_explain as ne
import infra.pg.neurons_llm as llm
import infra.pg.neurons_verdict as v


def main(root, grid_csv, corpus, batch, model, seed, gate_dir, clusters_file):
    llm.refuse_opus(model, f"{batch}: --model", llm.OPUS_ROUTED)
    out = Path(root)
    explain = out / "explain"
    wanted = {l.strip() for l in open(clusters_file) if l.strip()}
    grid = grid_csv.split(",")
    end = grid[-1]
    gate = ne.judge_gate_record(Path(gate_dir), out)
    ne.ensure_manifest(explain)
    pos = ne.check_positions(explain, end)
    tables = ne.grid_cluster_tables(grid, explain)
    by_cluster = {r["cluster"]: (pole, r) for pole, t in tables.items() for r in t}
    judged = {p["cluster"] for p in v.judge_plans(explain) if not p["control"]}
    for c in wanted:
        if c not in by_cluster or by_cluster[c][1]["status"] != "late_primary":
            raise ValueError(f"{c}: not a late-born primary cluster under the current clusters.jsonl")
        if c in judged:
            raise ValueError(f"{c}: already has judge lines")
    rng = np.random.default_rng(int(seed))
    ne.note_in_manifest(explain, seeds={f"judge:{batch}": int(seed)})
    strings = {}
    specs = []
    offsets = ne.harvest_corpus(end, corpus)[1]
    for pole in sorted({by_cluster[c][0] for c in wanted}):
        layer, neuron, _ = ne.parse_pole(pole)
        if layer not in strings:
            strings = {layer: ne.layer_token_strings(end, corpus, layer)}
        mine = strings[layer][pole]
        by_overlap = sorted((p for p in strings[layer] if ne.parse_pole(p)[1] != neuron),
                            key=lambda p: (-len(mine & strings[layer][p]), p))
        top = ne.pole_contexts(end, corpus, pole, ne.N_TOPK, ne.JUDGE_BEFORE)
        for r in tables[pole]:                                # table order, as cmd_prepare_judge walks it
            if r["cluster"] not in wanted:
                continue
            c = r["cluster"]
            tb = r["t_before"]
            windows = ne.exemplar_windows(offsets, [int(pos[pole][k]) for k in r["ranks"]])
            exemplars = [ne.marked(top[k]) for k in sorted(r["ranks"])[:ne.N_JUDGE_EXEMPLARS]]
            true_rec = ne.record(grid[tb], corpus, pole, windows, ne.N_JUDGE_RECORD)
            if not true_rec:
                raise ValueError(f"{c}: no context left in its record at {ne.GRID_LABELS[tb]}")
            tk = ne.layer_file(str(grid[tb]), "topk", layer)["tok"][:, :, :ne.N_TOPK]
            avail = ((tk >= 0) & ~ne.in_windows(tk, windows)).sum(-1)
            eligible = [p for p in by_overlap if avail[ne.parse_pole(p)[1], ne.parse_pole(p)[2]] >= len(true_rec)]
            draws = sorted(eligible)
            picks = rng.choice(len(draws), size=2, replace=False)
            for dtype, dpole, draw in (("random", draws[picks[0]], 0), ("random", draws[picks[1]], 1), ("hard", eligible[0], 0)):
                recs = {"true": true_rec, "decoy": ne.record(grid[tb], corpus, dpole, windows, len(true_rec))}
                for sample in (0, 1):
                    order = ["true", "decoy"] if rng.random() < 0.5 else ["decoy", "true"]
                    values = {"exemplars": ne.bullet(exemplars), "record_a": ne.bullet(recs[order[0]]), "record_b": ne.bullet(recs[order[1]])}
                    specs.append((values, {"cluster": c, "control": False, "decoy_type": dtype, "decoy_pole": dpole, "draw": draw,
                                           "t_before": ne.GRID_LABELS[tb], "sample": sample, "shown_order": order}))
    ids = ne.opaque_ids("judge", batch, len(specs), rng)
    per_file = Counter((plan["cluster"], plan["sample"]) for _, plan in specs)
    plans, reqs = [], {0: [], 1: []}
    for rid, (values, plan) in zip(ids, specs):
        reqs[plan["sample"]].append(ne.build_request("judge", rid, model, values))
        plans.append({"id": rid, **plan, "cluster_requests_in_file": per_file[(plan["cluster"], plan["sample"])]})
    for sample in (0, 1):
        ne.write_batch(explain, f"{batch}-judge-s{sample}", [reqs[sample][i] for i in rng.permutation(len(reqs[sample]))])
    llm.write_jsonl_new(explain / "plans" / f"{batch}-judge.jsonl", plans)
    m = ne.read_json(explain / "manifest.json")
    m.setdefault("judge_gates", {})[batch] = gate
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({"batch": batch, "clusters": len(wanted), "requests": len(specs), "seed": int(seed)}))


if __name__ == "__main__":
    main(*sys.argv[1:])
