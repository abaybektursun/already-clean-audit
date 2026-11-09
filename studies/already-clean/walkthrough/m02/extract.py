"""m02: a unit and its two sides. True matrix shapes of one MLP unit, and the banked activations a = silu(g) * u
of one unit's two sides at their fifty strongest contexts, with g and u, recomputed here from g and u.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import importlib.util
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


CONTEXTS = "studies/next/m1/banks/start_a/samples/contexts.jsonl"
MANIFEST = "studies/next/m1/banks/start_a/manifest.json"
UNIT = "L10N9794"            # the unit of the birth shown in m05 and m06 (10_recount2/out/06_examples.json)


def main():
    src(CONTEXTS)  # first, so the data link opens the contexts file (head, 2026-11-08)
    spec = importlib.util.spec_from_file_location("n13_arms", src("infra/pg/n13_arms.py"))
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    man = json.loads(src(MANIFEST).read_text())
    cfg = {"num_hidden_layers": man["n_layers"], "hidden_size": man["hidden"], "intermediate_size": man["width"],
           "vocab_size": 100278, "num_attention_heads": 32, "num_key_value_heads": 32}
    shapes = arms.expected_shapes(cfg)
    gate, up, down = (list(shapes[f"model.layers.0.mlp.{k}_proj.weight"]) for k in ("gate", "up", "down"))
    per_unit = gate[1] + up[1] + down[0]
    print(f"PLAN 11,008 x 4,096 -> matrix_shapes.gate = {gate}; down = {down}; PLAN 12,288 -> numbers_per_unit = {per_unit}")

    rows = [json.loads(s) for s in src(CONTEXTS).read_text().splitlines() if s.strip()]
    # recompute a = silu(g) * u in 32-bit from the banked 16-bit g and u, for every banked entry (640 sides x 50)
    g = np.array([[e["g"] for e in r["topk"]] for r in rows], np.float32)
    u = np.array([[e["u"] for e in r["topk"]] for r in rows], np.float32)
    a = np.array([[e["a"] for e in r["topk"]] for r in rows], np.float32)
    assert np.all((g.view(np.uint32) & 0xFFFF) == 0) and np.all((u.view(np.uint32) & 0xFFFF) == 0), "g or u not bfloat16"
    a_re = (g / (np.float32(1) + np.exp(-g))).astype(np.float32) * u
    rel = np.abs(a_re - a) / np.abs(a)
    sign_ok = all((np.sign(a[i]) == (1 if r["pole"].endswith("+") else -1)).all() for i, r in enumerate(rows))
    assert sign_ok, "an entry has the other side's sign"
    print(f"CHECK a = silu(g)*u recomputed for {a.size} entries: max relative difference {rel.max():.2e}")
    quad = np.array([r["sign_counts"] for r in rows]).sum(0)

    sides = []
    for s in ("+", "-"):
        r = next(r for r in rows if r["pole"] == UNIT + s)
        sides.append({
            "side": "positive" if s == "+" else "negative", "id": r["pole"], "layer": int(UNIT[1:3]),
            "unit": int(UNIT.split("N")[1]), "max_abs_activation": r["max_abs"], "percentile_99": r["q99"], "percentile_99_9": r["q999"],
            "percentile_99_99": r["q9999"], "density": r["density"],
            "sign_pattern_counts": dict(zip(["g>=0,u>=0", "g>=0,u<0", "g<0,u>=0", "g<0,u<0"], r["sign_counts"])),
            "contexts": [{"rank": e["rank"], "corpus_token": e["tok"], "document": e["doc"], "position_in_document": e["pos"],
                          "before": e["before"], "token": e["token"], "after": e["after"],
                          "g": e["g"], "u": e["u"], "a": e["a"]} for e in r["topk"]]})
    data = {
        "model": man["repo"], "snapshot": {"branch": man["branch"], "commit": man["revision"], "step": int(man["branch"].split("-step")[1])},
        "matrix_shapes": {"gate": gate, "up": up, "down": down}, "numbers_per_unit": per_unit,
        "units_per_layer": man["width"], "sides_per_layer": 2 * man["width"],
        "sides_total": 2 * man["width"] * man["n_layers"], "hidden_size": man["hidden"],
        "unit_shown": sides,
        "recompute_check": {"entries": int(a.size), "sides": len(rows),
                            "max_relative_difference": float(rel.max()), "median_relative_difference": float(np.median(rel))},
        "sign_pattern_counts_all_banked_sides": dict(zip(["g>=0,u>=0", "g>=0,u<0", "g<0,u>=0", "g<0,u<0"],
                                                         [int(x) for x in quad])),
        "not_banked": "Activations at every token of a whole context are not banked in the repository; each banked "
                      "context carries a, g and u at its marked token only. Weight values are not banked.",
    }
    print(f"PLAN 22,016 -> sides_per_layer = {data['sides_per_layer']}; PLAN 704,512 -> sides_total = {data['sides_total']}")
    notes = {
        "snapshot": "The banked contexts with g and u come from the harvest of the stage-1 snapshot at step 476,838 (the start "
                    "of the three data mixes, between released snapshots 12 and 13), the only harvest in the repository that "
                    "banks a side's full fifty with g and u; it is not the end of training.",
        "matrix_shapes": "Rows x columns of the layer's gate, up and down matrices (infra/pg/n13_arms.py expected_shapes); "
                         "unit n owns row n of gate and up and column n of down.",
        "unit_shown": "Both sides of unit 9,794 in layer 10 (internal id L10N9794+/-), each with its fifty strongest contexts at "
                      "this snapshot in rank order; before/token/after are the banked decoded strings verbatim (64 tokens before, "
                      "16 after, as harvested); g and u are the 16-bit gate and up outputs; a is the banked 32-bit activation.",
        "percentile_99_9": "The side's own 99.9th percentile of |a| over the corpus at this snapshot (internal q999; percentile_99 and percentile_99_99 likewise, internal q99 and q9999); density is the "
                "count of tokens with |a| at or above 0.01 x max |a| (bin resolution), from the harvest manifest's density_rule.",
        "sign_pattern_counts": "Of the side's fifty, how many have each sign pattern of (g, u) (banked sign_counts, ordered "
                               "g>=0&u>=0, g>=0&u<0, g<0&u>=0, g<0&u<0); g<0 and u<0 is the region excluded from groups.",
        "recompute_check": "a recomputed here as float32(g / (1 + exp(-g))) * u from the banked g and u for all 32,000 banked "
                           "entries of the 640 banked sides, against the banked a.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
