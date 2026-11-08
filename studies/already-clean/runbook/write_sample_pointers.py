"""Per-pole and per-cluster pointers from the W5 samples to the agent requests they came from (closeout checker F23, F29).

usage: python3 -I write_sample_pointers.py <explain worktree root>
SPEC_D1 §10 wants, per pole, the contexts as the explainer saw them and, per cluster, the records as the judge saw them. The
instrument's sample files carry those texts for the gallery poles and four parent tests only; for every pole and cluster the
texts are the request lines the agents answered, banked under bank/w2c/<cell>/explain/requests/ and
bank/w4/o7_s_v2/explain/requests/. This writes, beside the samples, one derived index per file:
  q1_<cell>_poles.requests_index.jsonl : per pole, every explainer and second-reader request id with the sha256 of its line
  q2_clusters.requests_index.jsonl     : per cluster, its pole's clusterer request ids and its judge request ids, each with sha256
The first line of each index says it is derived and names its sources. Nothing in the instrument's files changes.
"""
import datetime
import hashlib
import json
import pathlib
import sys

CELLS = {"Q4B": "q4b", "G4": "g4", "O7": "o7"}


def lines_by_id(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out[json.loads(line)["id"]] = hashlib.sha256(line.rstrip("\n").encode()).hexdigest()
    return out


def main(root):
    E = pathlib.Path(root); S = E / "bank/w5/verdict/samples"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = {}
    for cell, d in CELLS.items():
        req_dir = E / "bank/w2c" / d / "explain/requests"
        files = sorted(p for p in req_dir.glob("*.jsonl") if "explainer" in p.name or "second_reader" in p.name)
        idx = {p.name: lines_by_id(p) for p in files}
        rows = [json.loads(l) for l in open(S / f"q1_{cell}_poles.jsonl") if l.strip()]
        out = [{"derived": True, "written_utc": now, "cell": cell, "sources": [str(p.relative_to(E)) for p in files],
                "note": "pole -> the agent request lines holding the contexts as the explainer and second reader saw them; sha256 of each request line (runbook/write_sample_pointers.py)"}]
        for r in rows:
            reqs = [{"file": name, "id": i, "sha256": h} for name, m in idx.items() for i, h in m.items() if i.split(":")[1] == r["pole"]]
            assert reqs, (cell, r["pole"])
            out.append({"pole": r["pole"], "requests": sorted(reqs, key=lambda x: (x["file"], x["id"]))})
        p = S / f"q1_{cell}_poles.requests_index.jsonl"
        p.write_text("".join(json.dumps(o) + "\n" for o in out)); written[p.name] = len(out) - 1
    X = E / "bank/w4/o7_s_v2/explain"
    judge = {}
    for p in sorted((X / "requests").glob("w4j*-judge-s*.jsonl")):
        for i, h in lines_by_id(p).items():
            judge[i] = (p.name, h)
    clus = {}
    for p in sorted((X / "requests").glob("w4c-clusterer-s*.jsonl")):
        for i, h in lines_by_id(p).items():
            clus[i] = (p.name, h)
    plan = {r["id"]: r for r in (json.loads(l) for l in open(X / "plans/w4c-clusterer.jsonl") if l.strip())}
    by_pole = {}
    for i, r in plan.items():
        by_pole.setdefault(r["pole"], []).append(i)
    rows = [json.loads(l) for l in open(S / "q2_clusters.jsonl") if l.strip()]
    out = [{"derived": True, "written_utc": now, "sources": ["bank/w4/o7_s_v2/explain/requests/w4c-clusterer-s{0,1}.jsonl", "bank/w4/o7_s_v2/explain/requests/w4j-judge-s{0,1}.jsonl", "bank/w4/o7_s_v2/explain/requests/w4j2-judge-s{0,1}.jsonl", "bank/w4/o7_s_v2/explain/plans/w4c-clusterer.jsonl"],
            "note": "cluster -> its pole's clusterer request lines (the contexts the clusterer saw) and its judge request lines (the exemplars and the two records as the judge saw them); sha256 of each request line (runbook/write_sample_pointers.py)"}]
    missing = 0
    for r in rows:
        jr = []
        for j in r.get("judgments") or []:
            if j["id"] in judge:
                jr.append({"id": j["id"], "file": judge[j["id"]][0], "sha256": judge[j["id"]][1]})
            else:
                missing += 1
        cr = [{"id": i, "file": clus[i][0], "sha256": clus[i][1], "sample": plan[i]["sample"]} for i in sorted(by_pole.get(r["pole"], [])) if i in clus]
        out.append({"cluster": r["cluster"], "pole": r["pole"], "clusterer_requests": cr, "judge_requests": jr})
    p = S / "q2_clusters.requests_index.jsonl"
    p.write_text("".join(json.dumps(o) + "\n" for o in out)); written[p.name] = len(out) - 1
    print(json.dumps({"written": written, "judge_ids_not_in_requests": missing}))


if __name__ == "__main__":
    main(sys.argv[1])
