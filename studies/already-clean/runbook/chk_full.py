"""usage: chk_full.py <agent task output jsonl> <chunk dir>
Did every r<k>/user.txt reach the agent whole? The Read tool caps a result's size and returns the first lines only, with
no error flag and no marker in the result (seen 2026-10-24: 42 of 52 numbered lines for a 73 KB file). For every request every
line of 20+ characters of its user.txt must appear verbatim in the Read results of that file (the all-rounds audit's rule; a cut
result loses lines).
Earlier rules overcounted (splitlines() separators; a lone CR changes the tool's numbering) or looked only at the final line's tail (the committed gate until 2026-10-24T04:4xZ); the every-line rule replaced them. Reads tool results for the input's own lines only. Prints a JSON summary; exits 1 if any user.txt never arrived whole."""
import json, pathlib, re, sys
out, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
reqs = sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"r\d+", p.name))
def lines(path):            # every line of 20+ characters (CR removed): each must appear verbatim in a Read result of the file
    return [ln for ln in path.read_text(encoding="utf-8").replace("\r", "").split("\n") if len(ln) >= 20]
want = {str(p / "user.txt"): lines(p / "user.txt") for p in reqs}
uses, seen = {}, {k: "" for k in want}
for line in open(out):
    if not line.strip():
        continue
    ev = json.loads(line); msg = ev.get("message", ev); content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, list):
        continue
    for c in content:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "tool_use" and c.get("name") == "Read" and c.get("input", {}).get("file_path") in want:
            uses[c["id"]] = c["input"]["file_path"]
        if c.get("type") == "tool_result" and c.get("tool_use_id") in uses and not c.get("is_error"):
            r = c.get("content"); txt = r if isinstance(r, str) else "".join(x.get("text", "") for x in r if isinstance(x, dict))
            p = uses[c["tool_use_id"]]
            seen[p] += "\n" + txt.replace("\r", "")
short = [pathlib.Path(p).parent.name for p in want if any(ln not in seen[p] for ln in want[p])]
print(json.dumps({"chunk": d.name, "requests": len(reqs), "user_txt_whole": len(want) - len(short), "short_reads": short}))
sys.exit(0 if not short else 1)
