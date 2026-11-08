"""usage: chk_calls.py <agent task output jsonl> <chunk dir>
Audits an answering agent's tool calls by name and target only (never tool results):
every r<k>/user.txt in the chunk must have been opened with the Read tool, in full (no offset or limit); Bash is allowed
only for a whole command that is `ls [args]` without shell metacharacters or `date -u +%Y-%m-%dT%H:%M:%SZ`; the only
Write target is <chunk dir>.out.jsonl; that file must have been Read after it was last written; no tool other than Read,
Write and Bash; every transcript line must parse. Hardened 2026-10-23 after the W4-W5 checker's F16. A Read outside the
chunk that failed (its tool_result is_error with the fixed text 'File does not exist', so no content reached the agent) is
listed under failed_reads_outside_chunk and is not a violation (2026-10-24, F15 round r6); tool results are inspected for
that error flag and prefix only, never read for content.
Prints a JSON summary and exits 1 on any violation."""
import json, pathlib, re, sys
out, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
want_out = str(d.with_name(d.name + ".out.jsonl"))
n_req = len([p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"r\d+", p.name)])
calls, unparseable, not_found = [], 0, set()
for line in open(out):
    if not line.strip():
        continue
    try:
        ev = json.loads(line)
    except Exception:
        unparseable += 1
        continue
    msg = ev.get("message", ev)
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, list):
        continue
    for c in content:
        if isinstance(c, dict) and c.get("type") == "tool_use":
            calls.append((c.get("name"), c.get("input", {}) or {}, c.get("id")))
        if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
            r = c.get("content"); txt = r if isinstance(r, str) else "".join(x.get("text", "") for x in r if isinstance(x, dict))
            if txt.startswith("File does not exist"):
                not_found.add(c.get("tool_use_id"))
BASH_OK = re.compile(r"^\s*(ls(\s+[^;&|<>$`()\n]*)?|date\s+-u\s+\+%Y-%m-%dT%H:%M:%SZ)\s*$")
reads = [i.get("file_path", "") for n, i, u in calls if n == "Read" and u not in not_found]
failed_foreign = [i.get("file_path", "") for n, i, u in calls if n == "Read" and u in not_found]
user_read = {k for k in range(n_req) if str(d / f"r{k}" / "user.txt") in reads}
partial_reads = [i.get("file_path", "") for n, i, u in calls if n == "Read" and str(i.get("file_path", "")).endswith("/user.txt")
                 and (i.get("offset") is not None or i.get("limit") is not None)]
bash_bad = [i.get("command", "")[:120] for n, i, u in calls if n == "Bash" and not BASH_OK.match(i.get("command", "") or "")]
other_tools = sorted({n for n, _, _ in calls if n not in ("Read", "Write", "Bash")})
writes_bad = [i.get("file_path", "") for n, i, u in calls if n in ("Write", "Edit", "NotebookEdit") and i.get("file_path") != want_out]
wrote_at = [k for k, (n, i, u) in enumerate(calls) if n == "Write" and i.get("file_path") == want_out]
reread = any(k > wrote_at[-1] and n == "Read" and i.get("file_path") == want_out for k, (n, i, u) in enumerate(calls)) if wrote_at else False
foreign_reads = [p for p in reads if not (p.startswith(str(d) + "/") or p == want_out)]
res = {"chunk": d.name, "requests": n_req, "user_txt_read_via_Read": len(user_read), "partial_user_txt_reads": partial_reads,
       "bash_not_ls_or_date": bash_bad, "other_tools": other_tools, "writes_elsewhere": writes_bad,
       "reads_outside_chunk": foreign_reads, "failed_reads_outside_chunk": failed_foreign, "reread_output_after_write": reread, "unparseable_lines": unparseable}
ok = (len(user_read) == n_req and not partial_reads and not bash_bad and not other_tools and not writes_bad
      and not foreign_reads and reread and unparseable == 0)
print(json.dumps(res)); sys.exit(0 if ok else 1)
