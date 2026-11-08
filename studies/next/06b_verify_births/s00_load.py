"""Load the banked inputs once and save the fields the later scripts need.
Inputs (read-only): BANK/w3/grid/{1,1b}/out/tNN_stepN/{poles/exemplar_a.npz, samples/contexts.jsonl},
BANK/w4/o7_s_v2/explain/clusters.jsonl, BANK/w5/verdict/samples/q2_clusters.jsonl.
Outputs: out/inputs.npz, out/clusters.json, out/q2_banked.json, out/s00_load.txt."""
import json
from pathlib import Path
import numpy as np

BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
OUT = Path(__file__).resolve().parent / "out"
STEPS = (0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 60000, 100000, 180000,
         320000, 560000, 1000000, 1413814)

dirs = sorted(BANK.glob("w3/grid/*/out/t*_step*"), key=lambda p: int(p.name[1:3]))
assert [int(p.name.split("_step")[1]) for p in dirs] == list(STEPS), [p.name for p in dirs]
log = [f"grid dirs: {[str(p.relative_to(BANK)) for p in dirs]}"]

a, q999, max_abs, order, top_a_t15 = [], [], [], None, None
for t, d in enumerate(dirs):
    with np.load(d / "poles/exemplar_a.npz", allow_pickle=False) as z:
        a.append(z["a"].astype(np.float64))
    rows = [json.loads(l) for l in open(d / "samples/contexts.jsonl", encoding="utf-8")]
    poles = [r["pole"] for r in rows]
    if order is None:
        order = poles
    assert poles == order, f"{d}: pole order differs from {dirs[0]}"
    q999.append([r["q999"] for r in rows])
    max_abs.append([r["max_abs"] for r in rows])
    if t == 15:
        top_a_t15 = np.array([[x["a"] for x in r["topk"]] for r in rows], dtype=np.float64)
a = np.stack(a)                      # [16, 640, 50]
q999 = np.array(q999)                # [16, 640]
max_abs = np.array(max_abs)          # [16, 640]
assert a.shape[1] == len(order)
# Row order check: exemplar row i, column r at T_15 must be contexts line i, topk rank r at T_15.
d15 = np.abs(a[15] - top_a_t15).max()
log.append(f"poles in grid: {len(order)}; exemplar_a shape {a.shape}")
log.append(f"T_15 check, max |exemplar_a[15] - contexts.topk.a| = {d15:.3e}")
assert d15 < 1e-4
np.savez(OUT / "inputs.npz", poles=np.array(order), a=a, q999=q999, max_abs=max_abs)

clusters = {}
for l in open(BANK / "w4/o7_s_v2/explain/clusters.jsonl", encoding="utf-8"):
    r = json.loads(l)
    assert "declined" not in r
    clusters.setdefault(r["pole"], {})[r["sample"]] = r["clusters"]
json.dump(clusters, open(OUT / "clusters.json", "w"))
log.append(f"clusters.jsonl: {len(clusters)} poles, samples {sorted({s for v in clusters.values() for s in v})}")

KEEP = ("cluster", "pole", "c", "echo", "ranks", "med", "present_pct", "present_raw", "birth_pct", "birth_raw",
        "matched", "peak_end", "is_main", "late_pct", "late_raw", "status", "t_before",
        "acc_random", "acc_hard", "flagged")
banked = []
for l in open(BANK / "w5/verdict/samples/q2_clusters.jsonl", encoding="utf-8"):
    r = json.loads(l)
    banked.append({k: r[k] for k in KEEP if k in r})
json.dump(banked, open(OUT / "q2_banked.json", "w"))
log.append(f"q2_clusters.jsonl: {len(banked)} rows, {sum(not r['echo'] for r in banked)} non-echo, "
           f"{sum('flagged' in r for r in banked)} with a judge flag")
(OUT / "s00_load.txt").write_text("\n".join(log) + "\n")
print("\n".join(log))
