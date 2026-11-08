#!/usr/bin/env python3
"""V1 roster gate (§2d): count key-aligned FULL fine-tunes per Qwen2.5 rung.

Key-aligned = the candidate's model.safetensors.index.json weight-key set
equals the base's. Only index files are fetched (~100 KB each) — no weights,
no GPU, fully off-meter. Verdict per rung: PASS (>=12) or COVERAGE-IMPOSSIBLE
(the rung's (a)/(d) cells are dropped; (b)/(c) run on the base).

Usage: python3 roster_gate.py 14B [--need 12] [--max-check 40]
"""
import argparse
import json
import re
import urllib.request

BAD = re.compile(r"mlx|gguf|4bit|8bit|awq|gptq|bnb|quant|exl2|w8a8|fp8|int[48]", re.I)


def api(url):
    headers = {"User-Agent": "shackleton-roster-gate"}
    import os
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    req = urllib.request.Request(url, headers=headers)
    return json.load(urllib.request.urlopen(req))


def index_keys(repo):
    # /resolve/ follows LFS (some repos store the index as an LFS object,
    # where /raw/ returns the pointer text instead of JSON)
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    return frozenset(api(url)["weight_map"].keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("size", choices=["14B", "32B", "72B"])
    p.add_argument("--need", type=int, default=12)
    p.add_argument("--max-check", type=int, default=40)
    a = p.parse_args()

    base = f"Qwen/Qwen2.5-{a.size}"
    base_keys = index_keys(base)
    models = api(f"https://huggingface.co/api/models?filter=base_model:finetune:"
                 f"{base}&limit=200&sort=downloads&full=true")
    # candidate = quant/format-clean AND actually ships a sharded-safetensors
    # index (checked from the siblings listing, so a fetch below never 404s)
    cands = [m["id"] for m in models
             if not BAD.search(m["id"] + " " + " ".join(m.get("tags", [])))
             and any(s["rfilename"] == "model.safetensors.index.json"
                     for s in m.get("siblings", []))]
    print(f"[roster] {base}: {len(models)} tagged, {len(cands)} after filters")

    aligned, checked = [], 0
    for repo in cands:
        if checked >= a.max_check or len(aligned) >= a.need:
            break
        checked += 1
        keys = index_keys(repo)
        if keys == base_keys:
            aligned.append(repo)
            print(f"  ALIGNED {repo}")
        else:
            print(f"  key-mismatch {repo} ({len(keys)} vs {len(base_keys)} keys)")

    verdict = "PASS" if len(aligned) >= a.need else "COVERAGE-IMPOSSIBLE"
    out = {"rung": base, "need": a.need, "checked": checked,
           "aligned_count": len(aligned), "aligned": aligned, "verdict": verdict}
    fn = f"roster_gate_{a.size}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"[roster] VERDICT {base}: {verdict} ({len(aligned)}/{a.need}) -> {fn}")


if __name__ == "__main__":
    main()
