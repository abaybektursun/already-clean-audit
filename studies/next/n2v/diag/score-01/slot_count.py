# PYTHONPATH=. python studies/next/n2v/diag/score-01/slot_count.py > studies/next/n2v/diag/score-01/slot_count.json
"""Weight-free: rebuild score-01's T_14 plain-row order exactly as n2v_score.run does, and count the changed rows whose
batch slot has the same parity as their clean row's slot (the alignment diagnosis predicts these, and only these, are
bitwise equal before i0)."""
import json, pathlib, sys
from infra.pg import m2, n2v
R = pathlib.Path(__file__).resolve().parents[5]
units = json.loads((R / "studies/next/n12/units.json").read_text())["units"]
n12r = [r for h in ("freeze", "test") for r in m2.read_jsonl(R / f"studies/next/n12/banks/score-02/n12/n12_{h}.jsonl")]
pl = n2v.plan(units, n12r)
order = list(pl)
t = order[0]
items = pl[t]
kinds = [("clean", None)] + [("zero", q) for q in range(n2v.N_POLES)] + [("fd", q) for q in range(n2v.N_POLES)] + [("probe", 0)]
keyz = [(ii, w, kind, q) for ii, it in enumerate(items) for w, _ in it["uw"] for kind, q in kinds]
B = 4
pos = {k: i for i, k in enumerate(keyz)}
same = other = 0
by_kind = {}
for k, i in pos.items():
    if k[2] == "clean":
        continue
    c = pos[(k[0], k[1], "clean", None)]
    s = (i % B) % 2 == (c % B) % 2
    same += s; other += (not s)
    by_kind.setdefault(f"{k[2]}{k[3]}", set()).add(s)
print(json.dumps({"t": t, "units": len(items), "unit_windows": sum(len(it["uw"]) for it in items), "plain_rows": len(keyz),
                  "changed_rows": same + other, "same_slot_parity": same, "other_slot_parity": other,
                  "kind_same_parity": {k: sorted(v) for k, v in by_kind.items()}}, indent=1))
