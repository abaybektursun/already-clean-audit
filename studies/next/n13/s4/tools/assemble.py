"""S5: one banks tree with exactly one <arm>/<arm>/<bench>/{generations.jsonl, metrics.json, samples.jsonl} per arm and benchmark:
S3's four gate arms from $SP/s3banks (base, rl from s3gate-03; zero from s3gate-02; id from s3gate-04) and the S4 arms from
$SP/s4banks/<prefix>/<step>/. Copies (no links: the verdict's rglob does not follow linked directories). Writes <out>/../s5_sources.json:
per arm and benchmark the source, the metrics status and the sha256 of generations.jsonl. Reads no score.
usage: python3 -I assemble.py <out dir> <prefix>:<step> ..."""
import hashlib, json, pathlib, shutil, sys
SP = pathlib.Path(__file__).resolve().parents[1]
out = pathlib.Path(sys.argv[1])
assert not out.exists(), f"{out} exists"
src = {a: SP / "s3banks" / a / a for a in ("base", "rl", "zero", "id")}
for ps in sys.argv[2:]:
    p, s = ps.split(":")
    d = SP / "s4banks" / p / s / s
    assert d.is_dir(), d
    assert s not in src, f"{s} twice"
    src[s] = d
rec = {}
for arm, d in src.items():
    for bdir in sorted(x for x in d.iterdir() if x.is_dir()):
        g, m = bdir / "generations.jsonl", bdir / "metrics.json"
        assert m.exists(), m
        dst = out / arm / arm / bdir.name
        dst.mkdir(parents=True)
        for f in ("generations.jsonl", "metrics.json", "samples.jsonl"):
            if (bdir / f).exists():
                shutil.copyfile(bdir / f, dst / f)
        st = json.loads(m.read_text())["status"]
        rec[f"{arm}/{bdir.name}"] = {"source": str(bdir), "status": st,
                                     "generations_sha256": hashlib.sha256(g.read_bytes()).hexdigest() if g.exists() else None}
hits = {}
for g in out.rglob("metrics.json"):
    k = f"{g.parent.parent.name}/{g.parent.name}"
    hits[k] = hits.get(k, 0) + 1
assert all(v == 1 for v in hits.values()), hits
(out.parent / "s5_sources.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps({k: v["status"] for k, v in rec.items()}, indent=1))
