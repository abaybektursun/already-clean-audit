"""N13 independent read, step 0: verify every downloaded bank file against manifest_sha256.json.

    python -I verify_manifest.py <manifest_sha256.json> <banks dir> <out.json>

Fails loud on any missing, extra or mismatched file.
"""
import hashlib
import json
import pathlib
import sys

man_p, banks, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
man = json.loads(man_p.read_text())
listed = {f["path"]: f for f in man["files"]}
on_disk = {str(p.relative_to(banks)) for p in banks.rglob("*") if p.is_file()}
assert man["n_files"] == len(listed) == 165, (man["n_files"], len(listed))
assert on_disk == set(listed), (sorted(on_disk - set(listed)), sorted(set(listed) - on_disk))
rows, total = [], 0
for path, f in sorted(listed.items()):
    b = (banks / path).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    assert len(b) == f["bytes"] and h == f["sha256"], (path, len(b), f["bytes"], h, f["sha256"])
    total += len(b)
    rows.append({"path": path, "bytes": len(b), "sha256": h})
assert total == 411082754, total
res = {"manifest_sha256": hashlib.sha256(man_p.read_bytes()).hexdigest(), "n_files": len(rows), "total_bytes": total,
       "all_match": True, "files": rows}
out.write_text(json.dumps(res, indent=1))
print("MANIFEST_OK", res["manifest_sha256"], len(rows), total)
