"""Which model answered each manual-backend call (closeout checker F31).

usage: python3 -I manual_answering_models.py <explain dir (…/explain)> <subagents dir> <manual chunk root> <out jsonl>
calls.jsonl records the requested model id for manual-backend rows (manual_import.py requires it to equal the request's).
The answering model is in the Claude Code transcript of the agent that wrote the chunk file: every assistant message carries
its model id. For every manual call row (id, utc) this finds the imported chunk file holding that id with that utc, the
transcript that produced the file (its Write content equals the file, else its tool results carry the rows' utc, else it is
the one transcript briefed on the chunk whose shell commands name the output file; the F22 audit's attribution), and the
model ids of that transcript's assistant messages. Writes one row per manual call: id, role, utc, chunk, transcript,
answering_models {model: messages}; a first line with the summary. Reads transcripts for their model field and tool-call
metadata only.
"""
import collections
import datetime
import glob
import json
import os
import pathlib
import re
import sys


def scan(p):
    brief, ws, res, cmds, models = None, [], [], [], collections.Counter()
    with open(p) as f:
        for line in f:
            try: o = json.loads(line)
            except Exception: continue
            m = o.get("message", o); c = m.get("content") if isinstance(m, dict) else None
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("model"): models[m["model"]] += 1
            if brief is None and isinstance(m, dict) and m.get("role") == "user":
                text = c if isinstance(c, str) else "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
                mm = re.search(r"Chunk directory: (\S+)", text or ""); brief = mm.group(1) if mm else ""
            if not isinstance(c, list): continue
            for x in c:
                if not isinstance(x, dict): continue
                if x.get("type") == "tool_use":
                    i = x.get("input") or {}
                    if x.get("name") == "Write" and "/manual/" in str(i.get("file_path", "")): ws.append((i["file_path"], i.get("content", "")))
                    if x.get("name") == "Bash": cmds.append(str(i.get("command", "")))
                elif x.get("type") == "tool_result" and not x.get("is_error"):
                    r = x.get("content"); res.append(r if isinstance(r, str) else "".join(y.get("text", "") for y in r if isinstance(y, dict)))
    return brief, ws, "\n".join(res), "\n".join(cmds), dict(models)


def main(explain, sub, root, out):
    explain, root = pathlib.Path(explain), pathlib.Path(root)
    by_write, by_brief, info = {}, {}, {}
    for p in sorted(glob.glob(os.path.join(sub, "agent-*.jsonl"))):
        brief, ws, res, cmds, models = scan(p)
        if not brief and not ws: continue
        info[p] = (res, cmds, models, ws)
        for path, content in ws: by_write.setdefault(path, []).append((p, content))
        if brief: by_brief.setdefault(brief, []).append(p)
    # (id, utc) -> (chunk file, producer transcript)
    where = {}
    for imp in sorted(root.glob("*/*/chunk_*.imported.jsonl")):
        chunk = imp.parent / imp.name.replace(".imported.jsonl", ""); outpath = str(chunk) + ".out.jsonl"
        content = imp.read_text(); rows = [json.loads(l) for l in content.splitlines() if l.strip()]
        prod = [p for p, c in by_write.get(outpath, []) if c.strip() == content.strip()]
        if not prod:
            utcs = {r["utc"] for r in rows}
            prod = [p for p in by_brief.get(str(chunk), []) if all(u in info[p][0] for u in utcs)]
        how = "write" if prod else None
        if not prod:
            named = [p for p in by_brief.get(str(chunk), []) if os.path.basename(outpath) in info[p][1]]
            if len(named) == 1: prod, how = named, "brief+command"
        if not prod:                                   # the one transcript briefed on the chunk that wrote the output path at all
            sole = [p for p in by_brief.get(str(chunk), []) if any(path == outpath for path, _ in info[p][3])]
            if len(sole) == 1: prod, how = sole, "sole brief with a Write to the path (content differs from the imported file)"
        for r in rows:
            where[(r["id"], r["utc"])] = (str(imp.relative_to(root)), prod[-1] if prod else None, how)
    calls = [json.loads(l) for l in open(explain / "calls.jsonl") if l.strip()]
    rows, unattributed, tally = [], 0, collections.Counter()
    for c in calls:
        if c.get("backend") != "manual": continue
        chunk, prod, how = where.get((c["id"], c["utc"]), (None, None, None))
        allm = info[prod][2] if prod else {}
        models = {k: v for k, v in allm.items() if not k.startswith("<")}      # "<synthetic>" marks harness-inserted messages, not a model
        synthetic = sum(v for k, v in allm.items() if k.startswith("<"))
        if not prod: unattributed += 1
        for mdl in models: tally[(c.get("role") or c["id"].split(":")[0], mdl)] += 1
        rows.append({"id": c["id"], "role": c.get("role") or c["id"].split(":")[0], "utc": c["utc"], "requested_model": c.get("model"),
                     "chunk_file": chunk, "transcript": os.path.basename(prod) if prod else None, "attributed_by": how,
                     "answering_models": models, "harness_synthetic_messages": synthetic})
    head = {"derived": True, "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manual_calls": len(rows), "unattributed": unattributed, "attributed_by": dict(collections.Counter(r["attributed_by"] for r in rows if r["attributed_by"])),
            "answering_models_by_role": {f"{role}:{mdl}": n for (role, mdl), n in sorted(tally.items())},
            "note": "answering model per manual-backend call, read from the producing Claude Code transcript's assistant messages (runbook/manual_answering_models.py); calls.jsonl keeps the requested id"}
    pathlib.Path(out).write_text(json.dumps(head) + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({k: head[k] for k in ("manual_calls", "unattributed", "answering_models_by_role")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
