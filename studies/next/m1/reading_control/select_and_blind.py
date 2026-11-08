"""Dated exploratory analysis (2026-11-05) on the closed M1 result. Run from the repo root:
python3 studies/next/m1/reading_control/select_and_blind.py studies/next/m1/banks/start_a/samples/contexts.jsonl studies/next/m1/reading_control
Reading control: pick five matched layer-4 grid poles, shuffle with the target, write blind dumps. Prints no map."""
import json, sys, numpy as np
src, out_dir = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
TARGET = "L04N3542-"
t = [r for r in rows if r["pole"] == TARGET][0]
lay4 = [r for r in rows if r["pole"].startswith("L04N") and r["pole"] != TARGET]
print("target", TARGET, "max_abs", t["max_abs"], "window", t["max_abs"] / 2, t["max_abs"] * 2)
cands = [r for r in lay4 if t["max_abs"] / 2 <= r["max_abs"] <= t["max_abs"] * 2]
print("layer-4 grid poles other than the target:", len(lay4), "| within a factor of 2:", len(cands))
for r in lay4:
    print(f"  {r['pole']:12s} max_abs {r['max_abs']:.4f} {'qualifies' if r in cands else 'outside'}")
pick = np.random.default_rng(0).choice(len(cands), 5, replace=False)
chosen = [cands[i] for i in pick]
print("default_rng(0).choice(%d, 5, replace=False) ->" % len(cands), pick.tolist(), [r["pole"] for r in chosen])
six = chosen + [t]
order = np.random.default_rng(1).permutation(6)
labels = "ABCDEF"
mapping = {labels[j]: six[i]["pole"] for j, i in enumerate(order)}
json.dump({"permutation_default_rng1": order.tolist(), "six_in_order_before_shuffle": [r["pole"] for r in six], "map": mapping},
          open(f"{out_dir}/map.json", "w"), indent=1)
def clip(s, n, left):
    s = s.replace("\n", "⏎")
    return ("…" + s[-n:] if len(s) > n else s) if left else (s[:n] + "…" if len(s) > n else s)
with open(f"{out_dir}/blind_top20.txt", "w") as f:
    for j, i in enumerate(order):
        f.write(f"\n===== {labels[j]} =====\n")
        for k in six[i]["topk"][:20]:
            f.write(f"{k['rank']:2d} | {clip(k['before'], 90, True)}[[{k['token']}]]{clip(k['after'], 50, False)}\n")
print("blind dump written; map written to a separate file (not printed)")
