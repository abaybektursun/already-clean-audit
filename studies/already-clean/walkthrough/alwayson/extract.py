"""alwayson: the always-on units of OLMo-3 7B at the end of stage 1. The 1 % of the units of layers 0-30 with the largest
|mean activation| (3,412 of 341,248) and their mean activations against the rest, per layer.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.ALWAYSON_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

# The per-unit means of all 341,248 units are banked (abar.npz) with the cut that the method applied in the same run.
BANK = "studies/reuse/results/generic/1/generic/t15"
# The first run with the cut (2 November, SPEC4_birth.md), computed on its own instance; banked without abar.npz.
FIRST = "studies/reuse/results/birth/1/t15/high_background.json"
FRAC = 0.01
BIN = 0.1                                    # histogram bin width in log10 |mean activation|
LO, HI = -8.0, 1.3                           # histogram edges in log10 |mean activation|


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
    data["meta"] = {"commit": commit, "link": "studies/reuse/results/RESULTS.md",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def main():
    z = np.load(src(f"{BANK}/abar.npz"))
    a = z["abar"]                                        # [31 layers, 11,008 units], float32
    hb = json.loads(src(f"{BANK}/high_background.json").read_text())
    bg = json.loads(src(f"{BANK}/background.json").read_text())
    first = json.loads(src(FIRST).read_text())
    results_md = src("studies/reuse/results/RESULTS.md")  # the data link opens this file; it states the count
    assert "3,412" in results_md.read_text(), "RESULTS.md no longer states the count 3,412"
    L, W = a.shape
    n = L * W
    assert L == hb["n_up"] == bg["n_layers"] == 31 and n == hb["n_entries"] == 341248 and hb["frac"] == FRAC
    assert bg["t_index"] == 15 and hb["t"] == "T_15" and list(z["e1"]) == bg["e1"] and len(bg["e1"]) == 200

    # the rule: the round(1 % x 341,248) units with the largest |mean activation| (high_background.json rule)
    k = round(FRAC * n)
    flat = np.abs(a).ravel()
    order = np.argsort(-flat, kind="stable")[:k]
    is_member = np.zeros(n, bool)
    is_member[order] = True
    banked = {(l, u): v for l, u, v in hb["members"]}
    assert set(banked) == {(int(i // W), int(i % W)) for i in order}, "top 1 % of abar.npz differs from high_background.json"
    assert all(float(a[l, u]) == v for (l, u), v in banked.items())
    members = sorted(([int(i // W), int(i % W), float(a.ravel()[i])] for i in order), key=lambda r: -abs(r[2]))
    cut = float(flat[order[-1]])
    member_layer = np.array([m[0] for m in members])
    per_layer = np.bincount(member_layer, minlength=L).tolist()

    absa = np.abs(a)
    med = np.median(absa, axis=1)                         # float32, as banked
    edges = np.round(np.arange(LO, HI + BIN / 2, BIN), 10)
    logs = np.log10(absa.astype(np.float64))
    assert logs.min() >= LO and logs.max() < HI
    mem2 = is_member.reshape(L, W)
    hist_members = [np.histogram(logs[l][mem2[l]], bins=edges)[0].tolist() for l in range(L)]
    hist_rest = [np.histogram(logs[l][~mem2[l]], bins=edges)[0].tolist() for l in range(L)]

    overlap = len(set(banked) & {(l, u) for l, u, _ in first["members"]})
    data = {
        "model": "OLMo-3 7B, end of stage 1 (snapshot 15, step 1,413,814)",
        "units_considered": n, "layers_considered": [0, L - 1], "units_per_layer": W,
        "positions_averaged": len(bg["e1"]),
        "always_on_count": k,
        "check_count": compare("always-on count", k, hb["count"], f"{BANK}/high_background.json count", 0),
        "cut": cut,
        "check_cut": compare("cut = smallest member |mean activation|", cut, hb["cut"], f"{BANK}/high_background.json cut", 0),
        "members": members,
        "members_with_negative_mean": int(sum(m[2] < 0 for m in members)),
        "per_layer": {
            "always_on": per_layer,
            "check_always_on": compare("members per layer (sum of absolute differences)",
                                       sum(abs(x - y) for x, y in zip(per_layer, hb["per_layer"])), 0,
                                       f"{BANK}/high_background.json per_layer", 0),
            "median_abs_mean_activation": [float(x) for x in med],
            "check_median": compare("per-layer median |mean activation| (largest absolute difference)",
                                    float(np.max(np.abs(med.astype(np.float64) - np.array(bg["abar_abs_median_by_layer"])))),
                                    0, f"{BANK}/background.json abar_abs_median_by_layer", 0),
            "max_abs_mean_activation": [float(x) for x in absa.max(1)],
            "check_max": compare("per-layer max |mean activation| (largest absolute difference)",
                                 float(np.max(np.abs(absa.max(1).astype(np.float64) - np.array(bg["abar_abs_max_by_layer"])))),
                                 0, f"{BANK}/background.json abar_abs_max_by_layer", 0),
            "histogram_log10_edges": edges.tolist(),
            "histogram_always_on": hist_members,
            "histogram_rest": hist_rest,
        },
        "first_run_with_the_cut": {"date": "2026-11-02", "count": first["count"], "cut": first["cut"],
                                   "per_layer": first["per_layer"], "members_shared_with_this_bank": overlap},
    }
    assert sum(sum(h) for h in hist_members) == k and sum(sum(h) for h in hist_rest) == n - k
    in_27_30 = sum(per_layer[27:31])
    print(f"NOTE members in layers 27-30: {in_27_30}; layer 30: {per_layer[30]}; layers 0-6: {sum(per_layer[:7])}; "
          f"negative mean: {data['members_with_negative_mean']}; shared with the 2 November run: {overlap} of {k}")
    for num, key in [("3,412", "always_on_count"), ("1 %", "always_on_count / units_considered"),
                     ("1,575 in layer 30", "per_layer.always_on[30]"), ("3,021 in layers 27-30", "per_layer.always_on[27:31]"),
                     ("cut 0.241", "cut"), ("1,654 negative", "members_with_negative_mean")]:
        print(f"PLAN {num} -> {key}")

    notes = {
        "sentence": "Makes true: 'The 1 % of units with the largest mean activation (3,412) are excluded from the "
                    "input-replacement method' (from 2 November; SPEC4_birth.md §2: removed from every candidate input set "
                    "before ranking).",
        "always_on_count": "round(0.01 x 341,248) = 3,412: the units of layers 0-30 (31 x 11,008; layer 31 has no later layer to "
                           "feed) with the largest |mean activation| (banked as B_t / high_background).",
        "mean_activation": "The signed mean of the unit's activation a = silu(g) * u over the 200 background positions of the "
                           "end-of-stage-1 corpus at snapshot 15 (banked 'abar', positions 'e1'); the ranking uses its "
                           "absolute value, so members can have a negative mean.",
        "members": "[layer, unit, mean activation] for all 3,412, sorted by |mean| descending; recomputed as the top 3,412 of "
                   "abar.npz and asserted equal, member for member and value for value, to high_background.json.",
        "cut": "The smallest |mean activation| among the members.",
        "per_layer": "Per layer 0-30: members, the median and the maximum |mean activation| over the layer's 11,008 units "
                     "(recomputed from abar.npz, float32 median as banked), and histograms of log10 |mean activation| for "
                     "members and for the rest (bin edges histogram_log10_edges, 0.1 wide).",
        "bank": f"{BANK} is one of five banks that hold the identical abar.npz (same LFS object 7de30118...: birthobject, "
                "dating, generic, sameweights, sharedinputs); the instruments e1 run's high_background.json has the same 3,412 "
                "members and cut. The plan's 'studies/alwayson/' moved to "
                "studies/reproductions/alwayson/ and is the Qwen3.5-0.8B edit check (its abar.npz is 24 x 3,584, another "
                "model), so it is not used here.",
        "first_run_with_the_cut": "The 2 November run (studies/reuse/results/birth, SPEC4_birth.md) computed its own cut on "
                                  "another instance: 3,412 members, cut 0.2407; its abar.npz is not in the repository; "
                                  "members_shared_with_this_bank counts the units both lists hold.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
