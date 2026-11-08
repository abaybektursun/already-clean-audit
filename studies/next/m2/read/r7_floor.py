"""Reader of M2, phase 1, step 7: the floor from my own matched-pole ranks under A (read/r3_A.json, all 153 births)
and from the job's matched ranks under B (equal to mine on the 45 sampled births, read/r4_B.json); the bar.
Also the floor's sensitivity to the draw (two other ways to draw one matched pole per birth).
Output: read/r7_floor.json.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r2_from_ranks import floor, CHANCE

M2 = Path(__file__).resolve().parents[1]


def main():
    d = json.loads((M2 / "births.json").read_text())
    half = {b["id"]: b["half"] for b in d["births"]}
    A3 = json.loads((M2 / "read/r3_A.json").read_text())["units"]
    rec = {}
    for h in ("freeze", "test"):
        for l in (M2 / f"banks/score-06/m2/ranks_{h}.jsonl").open():
            r = json.loads(l)
            if r["kind"] == "real" and r["role"] == "pre":
                rec[r["birth"]] = r
    out = {}
    for h in ("freeze", "test"):
        ids = sorted(i for i in half if half[i] == h)
        MA = np.array([A3[f"{i}|real|pre"]["matched_rank_A"] for i in ids])
        MB = np.array([rec[i]["matched_rank"]["B"] for i in ids])
        o = {}
        for s, M in (("A", MA), ("B", MB)):
            f, hits = floor(M)
            rng = np.random.default_rng(0)
            alt1 = np.percentile([(M[np.arange(len(ids)), rng.choice(100, len(ids))] <= 10).mean()
                                  for _ in range(1000)], 97.5)
            alt2 = np.percentile([(M[np.arange(len(ids)), np.random.default_rng(r).integers(0, 100, len(ids))] <= 10).mean()
                                  for r in range(1000)], 97.5)
            o[s] = {"p97.5": f, "alt_choice": float(alt1), "alt_seed_per_rep": float(alt2),
                    "n_births_with_any_matched_hit": int(((M <= 10).any(1)).sum()),
                    "matched_hits_total": int((M <= 10).sum()), "n": len(ids)}
        out[h] = o
    out["bar"] = {s: max(CHANCE, out["freeze"][s]["p97.5"], out["test"][s]["p97.5"]) for s in ("A", "B")}
    (M2 / "read/r7_floor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
