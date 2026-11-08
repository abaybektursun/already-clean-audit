"""Bank one landed S4 step (or the prefix-level files) from S3.
Every file of results/next_n13/<prefix>/out/<step>/ is copied to the scratch tree <scratch>/s4banks/<prefix>/<step>/ (the verdict reads
that tree: generations.jsonl beside metrics.json). Files under 1 MB are also copied into studies/next/n13/s4/<prefix>/<step>/; larger
ones stay on S3 and are named with size and sha256 in studies/next/n13/s4/<prefix>/<step>/on_s3.json.
With --prefix-files: STATUS, steps.json, out/pipeline_status.json, out/steps.json, out/dmon.log, out/setup/* (under 1 MB) into
studies/next/n13/s4/<prefix>/.
usage: python3 -I bank_s4.py <prefix> (<step> | --prefix-files)
Fail-loud: no try/except."""
import hashlib, json, pathlib, subprocess, sys

B = "pg-lineage-769063704696"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]
REPO = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad")
MB = 1_000_000
prefix, what = sys.argv[1], sys.argv[2]


def ls(key):
    r = subprocess.run(["aws", "s3", "ls", f"s3://{B}/results/next_n13/{prefix}/{key}", "--recursive", *AWS], check=True, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 3)
        out.append((int(parts[2]), parts[3]))
    return out


def cp(key, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"s3://{B}/{key}", str(dst), "--only-show-errors", *AWS], check=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


base = f"results/next_n13/{prefix}/"
if what == "--prefix-files":
    keys = [(s, k) for s, k in ls("") if k[len(base):] in ("STATUS", "steps.json", "out/pipeline_status.json", "out/steps.json", "out/dmon.log")
            or k[len(base):].startswith("out/setup/")]
    dst_root = REPO / "studies/next/n13/s4" / prefix
    big = {}
    for s, k in keys:
        rel = k[len(base):].removeprefix("out/")
        if s >= MB:
            tmp = SCR / "s4banks" / prefix / "_prefix" / rel
            cp(k, tmp)
            big[rel] = {"bytes": s, "sha256": sha(tmp), "s3": f"s3://{B}/{k}"}
            continue
        cp(k, dst_root / rel)
    if big:
        (dst_root / "on_s3.json").write_text(json.dumps(big, indent=1) + "\n")
    print(json.dumps({"banked": len(keys) - len(big), "on_s3": big}, indent=1))
    sys.exit(0)

step = what
keys = ls(f"out/{step}/")
assert keys, f"nothing under out/{step}/"
dst_root = REPO / "studies/next/n13/s4" / prefix / step
big, small = {}, []
for s, k in keys:
    rel = k[len(base) + len(f"out/{step}/"):]
    full = SCR / "s4banks" / prefix / step / rel
    cp(k, full)
    assert full.stat().st_size == s, (k, s, full.stat().st_size)
    if s >= MB:
        big[rel] = {"bytes": s, "sha256": sha(full), "s3": f"s3://{B}/{k}"}
    else:
        (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst_root / rel).write_bytes(full.read_bytes())
        small.append(rel)
(dst_root).mkdir(parents=True, exist_ok=True)
(dst_root / "on_s3.json").write_text(json.dumps(big, indent=1) + "\n")
print(json.dumps({"step": step, "banked": small, "on_s3": big}, indent=1))
