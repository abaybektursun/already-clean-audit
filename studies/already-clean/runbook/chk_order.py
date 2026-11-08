"""usage: chk_order.py <chunk dir>: exit 1 unless <chunk dir>.out.jsonl has one line per request directory with ids in r0, r1, ... order (ids only; no answer text is read)."""
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); out = d.with_name(d.name + ".out.jsonl")
want = [json.load(open(d / f"r{k}" / "meta.json"))["id"] for k in range(len([p for p in d.iterdir() if p.is_dir()]))]
got = [json.loads(l)["id"] for l in open(out)]
ok = got == want
print(json.dumps({"chunk": d.name, "requests": len(want), "lines": len(got), "in_order": ok}))
sys.exit(0 if ok else 1)
