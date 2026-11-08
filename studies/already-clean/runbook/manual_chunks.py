"""Chunk a batch's pending requests for Claude Code answering agents (manual backend, MANUAL_BACKEND.md 2026-10-21 note).

usage: python3 manual_chunks.py <explain dir> <batch> <chunk size> <out dir> [<plan jsonl> <group key>]

Pending = requests with no response line and no decline. Chunks keep the requests file's order (a random
permutation) and never hold two requests that share the plan's <group key> (judge: `cluster`, so no agent
sees a cluster's true record twice; clusterer: `pole`). Writes <out dir>/chunk_NNN/r<k>/{meta.json, system.txt, user.txt, schema.json}
per request (files the Read tool can load) and <out dir>/index.json; the agent writes <out dir>/chunk_NNN.out.jsonl. Refuses a non-empty out dir. Reads no responses text, no plan field but the key.
"""
import json
import pathlib
import sys


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if pathlib.Path(p).exists() else []


def main(explain, batch, size, out, plan=None, key=None):
    explain, out = pathlib.Path(explain), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    done = {r["id"] for r in read_jsonl(explain / "responses" / f"{batch}.jsonl")}
    done |= {r["id"] for r in read_jsonl(explain / "responses" / f"{batch}.declined.jsonl")}
    pending = [r for r in read_jsonl(explain / "requests" / f"{batch}.jsonl") if r["id"] not in done]
    group = {r["id"]: r[key] for r in read_jsonl(plan)} if plan else {}
    chunks, cur, cur_groups = [], [], set()
    for r in pending:
        g = group.get(r["id"], r["id"])
        if len(cur) >= size or g in cur_groups:
            chunks.append(cur); cur, cur_groups = [], set()
        cur.append(r); cur_groups.add(g)
    if cur:
        chunks.append(cur)
    index = {}
    for i, rows in enumerate(chunks):
        name = f"chunk_{i:03d}"
        d = out / name
        d.mkdir()
        for k, r in enumerate(rows):   # one directory per request, in files the Read tool can load
            rd = d / f"r{k}"
            rd.mkdir()
            (rd / "meta.json").write_text(json.dumps({key: r[key] for key in ("id", "role", "model", "temperature", "max_tokens")}, indent=1))
            (rd / "system.txt").write_text(r["system"])
            (rd / "user.txt").write_text(r["user"])
            (rd / "schema.json").write_text(json.dumps(r["schema"], indent=1))
        index[name] = [r["id"] for r in rows]
    (out / "index.json").write_text(json.dumps({"explain": str(explain), "batch": batch, "size": size, "group_key": key,
                                                "n_pending": len(pending), "chunks": index}, indent=1))
    print(json.dumps({"batch": batch, "pending": len(pending), "chunks": len(chunks), "out": str(out)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), a[3], *(a[4:6] if len(a) > 4 else ()))
