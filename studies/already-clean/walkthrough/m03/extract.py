"""m03: the corpus and a side's fifty contexts. The corpus rule and its banked counts; one side's fifty strongest
contexts at the end of training; and positions fixed at the end of training read at every earlier snapshot.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import re
import subprocess

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


SPEC = "studies/already-clean/SPEC_D1.md"
MANIFEST = "studies/next/m1/banks/start_a/manifest.json"
COUNTS = "studies/next/m1/banks/start_a/m1/counts.npz"
EXAMPLES = "studies/next/10_recount2/out/06_examples.json"
FIRST_EDITION = "studies/already-clean/walkthrough/top_contexts/data.js"
ASSETS = "studies/next/00_ASSETS.md"
SIDE = "L03N6372+"           # the first edition's named side ("the lithium neuron")


def main():
    spec = src(SPEC).read_text()
    n_dirs = int(re.search(r"\((\d+) at commit `2ca900fb", spec)[1])
    per_dir = int(re.search(r"those that fill the first ([\d,]+) form the 20M corpus", spec)[1].replace(",", ""))
    before_after = re.search(r"A context is the (\d+) tokens before and (\d+) after the activating token", spec)
    man = json.loads(src(MANIFEST).read_text())
    z = np.load(src(COUNTS))
    dirs, groups = [str(d) for d in z["dirs"]], [str(g) for g in z["groups"]]
    assert len(dirs) == 50 and int(z["cm_mask"].sum()) == 5
    picked = [i * n_dirs // 50 for i in range(50)]
    scored = man["n_windows"] * (man["window"] - 1)
    assert scored == man["n_tokens_scored"], (scored, man["n_tokens_scored"])
    assert man["n_windows"] * man["window"] == man["n_tokens_windowed"]
    corpus = {
        "dataset": "allenai/dolma3_mix-6T-1025-7B", "directories_listed": n_dirs, "directories_used": len(dirs),
        "directory_index_rule": "floor(i * 165 / 50), i = 0..49, over the sorted list", "directory_indices": picked,
        "directories": [{"name": d, "kind": k, "code_or_math": bool(c)} for d, k, c in zip(dirs, groups, z["cm_mask"])],
        "tokens_per_directory": per_dir, "windows": man["n_windows"], "window_tokens": man["window"],
        "tokens_in_windows": man["n_tokens_windowed"], "scored_tokens": man["n_tokens_scored"],
        "excluded_positions": man["excluded_positions"], "start_token_id": man["bos_id"],
        "corpus_manifest_sha256": man["corpus_manifest_sha256"], "fifty": man["topk_K"],
        "tokens_before_context": int(before_after[1]), "tokens_after_context": int(before_after[2]),
    }
    print(f"PLAN 10,044 windows -> corpus.windows = {corpus['windows']}; PLAN 20,560,068 -> corpus.scored_tokens = "
          f"{corpus['scored_tokens']}; PLAN 50 directories, 400,000 tokens, 165 -> corpus.directories_used, "
          f"tokens_per_directory, directories_listed = {len(dirs)}, {per_dir}, {n_dirs}")

    # the sixteen snapshot steps, for labelling
    steps = [int(re.search(r"`stage1-step(\d+)`", s)[1]) for s in src(ASSETS).read_text().splitlines()
             if re.match(r"\| T_\d+ \| `stage1-step", s)]
    assert len(steps) == 16

    # one side's fifty at the end of training: only the first edition's data.js holds a full fifty in this repository
    t = src(FIRST_EDITION).read_text()
    fe = json.loads(t[t.index("=") + 1:].rstrip().rstrip(";"))
    assert fe["step"] == steps[15] and fe["k"] == 50
    side = next(n for n in fe["neurons"] if n["pole"] == SIDE)
    assert len(side["ctx"]) == 50
    fifty = {"id": SIDE, "side": "positive", "layer": 3, "unit": 6372, "step": fe["step"],
             "max_abs_activation": side["max"], "percentile_99_9": side["q999"], "density": side["density"],
             "contexts": [{"rank": i, "corpus_token": c[0], "window": c[0] // fe["window"],
                           "position_in_window": c[0] % fe["window"], "document": c[1], "a": c[2],
                           "before": c[3], "token": c[4], "after": c[5]} for i, c in enumerate(side["ctx"])]}

    # positions fixed at the end of training, read at every snapshot (third recount's raw examples)
    ex = json.loads(src(EXAMPLES).read_text())
    read_back = []
    for key, v in ex.items():
        assert len(v["a_plus_by_t"]) == 16 == len(v["median_by_t"]) == len(v["q999_by_t"])
        read_back.append({
            "example": key, "id": v["pole"], "side": "positive" if v["pole"].endswith("+") else "negative",
            "group": v["c"], "group_name": v["name"], "group_size": v["size"], "ranks_in_fifty": v["ranks"],
            "positions_at_end": [{"rank": p["rank"], "document": p["doc"], "position_in_document": p["pos"],
                                  "corpus_token": p["tok"], "token": p["token"], "before_tail": p["before_tail"]}
                                 for p in v["positions_T15"]],
            "activation_own_direction_by_snapshot": v["a_plus_by_t"],
            "median_by_snapshot": v["median_by_t"], "threshold_by_snapshot": v["q999_by_t"]})
    data = {"corpus": corpus, "snapshot_steps": steps, "fifty_at_end": fifty, "positions_read_back": read_back,
            "not_banked": "Each side's own fifty with activations at the end of training (the harvest's samples/contexts.jsonl "
                          "at step 1,413,814) is not in the repository; it is in the laptop bank and on S3 "
                          "(results/next_m1/inputs/t15_contexts.jsonl). Activations of all fifty positions at earlier "
                          "snapshots are banked here only for the groups in positions_read_back."}
    notes = {
        "corpus": "The 20.6-million-token corpus: the rule from SPEC_D1.md §4.2, the counts and hashes from the start-point "
                  "harvest manifest (identical corpus for every harvest), the 50 directory names and kinds from counts.npz "
                  "(code_or_math marks the five code-and-math directories).",
        "directory_indices": "The indices into the sorted list of 165 directories, computed here from the stated rule; the "
                             "list of 165 names is not banked, so names cannot be matched to indices here.",
        "fifty_at_end": "Copied from the first edition's top_contexts/data.js (built from the laptop bank's end-of-training "
                        "harvest): a and max_abs_activation are rounded there to 4 significant digits, before is cut to its last "
                        "180 characters and after to its first 36; window = corpus_token // 2,048 and position_in_window = "
                        "corpus_token % 2,048, computed here.",
        "positions_read_back": "Three groups from studies/next/10_recount2/out/06_examples.json: their positions are fixed at the "
                               "end of training (positions_at_end, before_tail verbatim) and read at each of the sixteen snapshots; "
                               "activation_own_direction_by_snapshot[t][j] is max(sign x a, 0) of position j at snapshot t, in raw "
                               "activation units; threshold_by_snapshot is the side's own 99.9th percentile at t (internal q999); percentile_99_9 in fifty_at_end is the same quantity at the end of training.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
