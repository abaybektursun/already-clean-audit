"""S5 (the head's addition 2, 2026-11-08): complete the assembled banks tree with the files the independent reader needs, then write
the manifest. Adds <arm>/<arm>/build.json for every built arm (S4 steps from $SP/s4banks, gate arms from $SP/s3banks), and _inputs/
with thresholds.json, sets.json, controls.json, r0_known/metrics.json and the s3gate-03 setup dir, each checked against its registered
sha256 where one is registered. Writes <tree>/../manifest_sha256.json: every file under the tree, path relative to the tree, bytes,
sha256. Reads no score.
usage: python3 -I s5_extras.py <tree> <prefix>:<step> ..."""
import hashlib, json, pathlib, shutil, sys

SP = pathlib.Path(__file__).resolve().parents[1]
W = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13")
N = W / "studies/next/n13"
tree = pathlib.Path(sys.argv[1])
assert tree.is_dir(), tree
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

for arm in ("base", "rl", "zero", "id"):
    b = SP / "s3banks" / arm / arm / "build.json"
    if b.exists():
        shutil.copyfile(b, tree / arm / arm / "build.json")
for ps in sys.argv[2:]:
    p, s = ps.split(":")
    b = SP / "s4banks" / p / s / s / "build.json"
    assert b.exists(), b
    assert (tree / s / s).is_dir(), s
    shutil.copyfile(b, tree / s / s / "build.json")

inp = tree / "_inputs"
assert not inp.exists(), inp
for src, dst, want in ((N / "thresholds.json", "thresholds.json", "60bf6d4eea9c"),
                       (N / "s3/s3gate-01/r1_rank/sets.json", "sets.json", "49a1065c5705"),
                       (N / "s3/s3gate-03/r1_rank/controls.json", "controls.json", "f98c30e4c3d2"),
                       (N / "r0/s2setup-07/r0_known/metrics.json", "r0_known/metrics.json", None)):
    d = inp / dst
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, d)
    assert want is None or sha(d).startswith(want), (dst, sha(d))
shutil.copytree(N / "s3/s3gate-03/setup", inp / "setup")
for bad in inp.rglob("*"):
    assert bad.name not in ("math500", "gsm8k", "generations.jsonl", "samples.jsonl"), bad   # keeps the verdict's rglob on the arm dirs

files = sorted(p for p in tree.rglob("*") if p.is_file())
man = {"tree": "results/next_n13/s5-01/banks/", "n_files": len(files),
       "files": [{"path": str(p.relative_to(tree)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
out = tree.parent / "manifest_sha256.json"
out.write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps({"n_files": len(files), "bytes": sum(f["bytes"] for f in man["files"]), "manifest": str(out), "manifest_sha256": sha(out)}))
