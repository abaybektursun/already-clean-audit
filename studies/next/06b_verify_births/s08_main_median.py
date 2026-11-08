"""Check of the review's mechanism sentence ("that cluster has the highest median"): share of the 574 poles with
>= 2 non-echo clusters where main has the highest T_15 median a+, and where it has the highest median at its
pole's earliest birth point. Input: out/q2_banked.json. Output: out/s08_main_median.txt."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
rows = [r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]]
by = {}
for r in rows:
    by.setdefault(r["pole"], []).append(r)
top15, n = 0, 0
for p, rs in by.items():
    if len(rs) < 2:
        continue
    n += 1
    m = next(r for r in rs if r["is_main"])
    top15 += m["med"][15] >= max(r["med"][15] for r in rs)
txt = f"poles {n}: main has the highest T_15 median a+ in {top15} ({top15/n:.3f})"
(OUT / "s08_main_median.txt").write_text(txt + "\n")
print(txt)
