"""The head's S4 engine rule (message received 2026-11-06 18:35Z; N13_SPEC.md §9.4 S4 head's note): every arm's engine block and its
launch's library versions are compared with S3's as it lands; any difference is reported to the head at once and the instance
keeps running (the head decides whether to terminate it, message received 18:37Z); an arm with a differing engine does not count. Reference: S3 (s3gate-03 base, MATH-500; the four S3 arms agree on every field below, and every
engine_args key but `model` agrees across them). Compared:
  - from the arm's metrics.json -> engine: vllm, max_num_seqs, max_num_batched_tokens, num_gpu_blocks, attention_backend, env (the
    sampler environment), and every engine_args key except `model` (the weights path, which differs by arm by design);
  - from the launch's setup pip_freeze_n13.txt: vllm, torch, triton, transformers, tokenizers.
Reads the live S3 object, so it runs as soon as the engine block is written (before the first request). No accuracy is read.
Writes studies/next/n13/s4/<prefix>/<step>/engine_check.json; exit 0 equal, 1 different.
usage: python3 -I engine_check.py <prefix> <step>"""
import json, pathlib, subprocess, sys

REPO = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13")
B = "s3://pg-lineage-769063704696/results/next_n13"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]
LIBS = ("vllm", "torch", "triton", "transformers", "tokenizers")
FIELDS = ("vllm", "max_num_seqs", "max_num_batched_tokens", "num_gpu_blocks", "attention_backend", "env")
prefix, step = sys.argv[1], sys.argv[2]


def cat(key):
    return subprocess.run(["aws", "s3", "cp", f"{B}/{key}", "-", *AWS], check=True, capture_output=True, text=True).stdout


def libs(text):
    d = dict(l.split("==", 1) for l in text.splitlines() if "==" in l)
    return {k: d.get(k) for k in LIBS}


ref_m = json.loads((REPO / "studies/next/n13/s3/s3gate-03/base/math500/metrics.json").read_text())["engine"]
ref_l = libs((REPO / "studies/next/n13/s3/s3gate-03/setup/pip_freeze_n13.txt").read_text())
out = {"prefix": prefix, "step": step, "reference": "studies/next/n13/s3/s3gate-03 (base MATH-500 engine block; setup pip freeze)", "bench": {}}
diffs = []
lib_now = libs(cat(f"{prefix}/out/setup/pip_freeze_n13.txt"))
out["libs"] = {"s3": ref_l, "arm": lib_now}
diffs += [f"lib {k}: {ref_l[k]} -> {lib_now[k]}" for k in LIBS if ref_l[k] != lib_now[k]]
for bn in ("math500", "gsm8k"):
    r = subprocess.run(["aws", "s3", "cp", f"{B}/{prefix}/out/{step}/{step}/{bn}/metrics.json", "-", *AWS], capture_output=True, text=True)
    if r.returncode != 0:
        continue
    e = json.loads(r.stdout).get("engine")
    assert e is not None, f"{bn}: no engine block yet"
    d = [f"{bn} {k}: {ref_m.get(k)} -> {e.get(k)}" for k in FIELDS if ref_m.get(k) != e.get(k)]
    ea_ref, ea = ref_m["engine_args"], e["engine_args"]
    d += [f"{bn} engine_args.{k}: {str(ea_ref.get(k))[:60]} -> {str(ea.get(k))[:60]}" for k in sorted(set(ea_ref) | set(ea)) if k != "model" and ea_ref.get(k) != ea.get(k)]
    out["bench"][bn] = {"fields": {k: e.get(k) for k in FIELDS}, "engine_args_model": ea.get("model"), "load_s": e.get("load_s"), "differences": d}
    diffs += d
assert out["bench"], "no metrics.json for this step"
out["differences"], out["equal"] = diffs, not diffs
dst = REPO / "studies/next/n13/s4" / prefix / step / "engine_check.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"step": step, "equal": out["equal"], "differences": diffs, "fields": {b: v["fields"] for b, v in out["bench"].items()}}, indent=1))
sys.exit(0 if out["equal"] else 1)
