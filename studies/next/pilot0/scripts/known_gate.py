"""Known-number gate: the first n rows of a new batch_60001 run reproduce the banked pilot-0 sdpa per_row_mean_ce
within 1e-4 nats (09_PLAN.md J8 card). usage: known_gate.py NEW.json BANKED.json"""
import json, sys
TOL = 1e-4
new, banked = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
a, b = new["per_row_mean_ce"], banked["per_row_mean_ce"]
n = len(a)
d = [abs(x - y) for x, y in zip(a, b[:n])]
i = max(range(n), key=lambda k: d[k])
print(f"rows compared {n}; max |diff| {d[i]:.2e} at row {i} ({a[i]:.7f} vs {b[i]:.7f}); "
      f"mean |diff| {sum(d) / n:.2e}; tolerance {TOL:g} -> {'holds' if d[i] <= TOL else 'fails'}")
