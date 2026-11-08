"""J8 rerun b2: recompute from the JSON files.

usage: recompute.py NEIGHBOUR.json NEIGHBOUR_TOKENS.npy TARGET.json KNOWN_NEW.json KNOWN_BANKED.json
N: pad-excluded mean CE of the neighbour batch (sum of per-row CE sums over pad-excluded targets / sum of
non-pad target counts), checked against the file's own field and against non-pad counts recounted from tokens.npy.
S: the target file's field. W: W&B zckrmllo train/CE loss at step 100,001. Branch: the frozen rule code
(rule.py in this directory, branch()). Known number: per-row |new - banked| of per_row_mean_ce over the rows the new
run scored (the registered gate), and beside it the pad-excluded per-row mean on the rows with no pad target (the
banked pilot-0 file has no per-row pad-excluded field; on a row with no pad target the two means are the same number).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PAD = 100277
W = 2.0262222290039062
RULE = Path(__file__).resolve().parent / "rule.py"
spec = importlib.util.spec_from_file_location("rule", RULE)
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)

nb_path, tok_path, tg_path, kn_path, kb_path = sys.argv[1:6]
nb, tg = json.load(open(nb_path)), json.load(open(tg_path))

# neighbour: recount N from per-row fields and from tokens.npy
rows = nb["rows_done"]
sums = nb["per_row_ce_sum_excluding_pad_targets"]
cnt = nb["per_row_nonpad_target_positions"]
assert len(sums) == len(cnt) == rows, (len(sums), len(cnt), rows)
tok = np.load(tok_path)
assert tok.dtype == np.uint32 and tok.shape == (512, 8192), (tok.dtype, tok.shape)
nonpad_tok = (tok[:rows, 1:] != PAD).sum(axis=1)
assert [int(x) for x in nonpad_tok] == cnt, "non-pad target counts differ from tokens.npy"
assert int((tok[:rows, 1:] == PAD).sum()) == nb["pad"]["pad_target_positions"], "pad target count differs"
N_re = float(np.sum(np.array(sums, dtype=np.float64))) / int(np.sum(cnt))
N = nb["pad"]["mean_ce_excluding_pad_targets"]
S = tg["pad"]["mean_ce_excluding_pad_targets"]
print(f"neighbour {nb_path}")
print(f"  rows_done {rows}  complete {nb['complete']}  revision {nb['revision']}  attn {nb['attn_implementation']}")
print(f"  N (file field) {N:.7f}  N (recount from per-row sums) {N_re:.7f}  |diff| {abs(N - N_re):.1e}")
print(f"  non-pad targets {int(np.sum(cnt)):,}  pad targets {nb['pad']['pad_target_positions']:,} (both equal tokens.npy)")
print(f"  every position mean_ce {nb['mean_ce']:.7f}  top-1 {nb['top1_accuracy']:.5f}")
print(f"target {tg_path}: S {S:.7f} rows_done {tg['rows_done']} complete {tg['complete']} revision {tg['revision']}")
b, dS, dN = rule.branch(S, N, W, tg["rows_done"], rows)
print(f"W {W!r}  S - W {S - W:+.7f}  N - W {N - W:+.7f}  |S - W| {dS:.7f}  |N - W| {dN:.7f}")
print(f"step 100,001 under the frozen rule: {b}")

# known number
kn, kb = json.load(open(kn_path)), json.load(open(kb_path))
a, c = kn["per_row_mean_ce"], kb["per_row_mean_ce"]
n = len(a)
assert n == kn["rows_done"], (n, kn["rows_done"])
d = [abs(x - y) for x, y in zip(a, c[:n])]
i = max(range(n), key=lambda k: d[k])
padn, padb = kn["pad"]["per_row_pad_target_positions"], kb["pad"]["per_row_pad_target_positions"][:n]
print(f"known number {kn_path} against {kb_path}")
print(f"  rows {n}  revision {kn['revision']}  attn {kn['attn_implementation']}  complete {kn['complete']}")
print(f"  per-row pad target counts equal on the {n} rows: {padn == padb}")
print(f"  max |diff| per_row_mean_ce {d[i]:.2e} at row {i} ({a[i]:.7f} vs {c[i]:.7f}); mean |diff| {sum(d) / n:.2e}")
print(f"  mean of per_row_mean_ce over rows 0-{n - 1}: new {sum(a) / n:.7f}  banked {sum(c[:n]) / n:.7f}")
print(f"  registered gate (max |diff| <= 1e-4): {'holds' if d[i] <= 1e-4 else 'fails'};"
      f"  prediction (<= 0.001 per row): {'holds' if d[i] <= 1e-3 else 'fails'}")
top_n, top_b = kn["per_row_top1"], kb["per_row_top1"][:n]
print(f"  max |diff| per_row_top1 (top-1 hits per row) {max(abs(x - y) for x, y in zip(top_n, top_b))}")
nopad = [k for k in range(n) if padn[k] == 0 and padb[k] == 0]
if "per_row_mean_ce_excluding_pad_targets" in kn:
    ex = kn["per_row_mean_ce_excluding_pad_targets"]
    assert all(ex[k] == a[k] for k in nopad), "pad-excluded and every-position row means differ on a row with no pad"
    dx = [abs(ex[k] - c[k]) for k in nopad]
    j = max(range(len(nopad)), key=lambda k: dx[k])
    print(f"  pad-excluded per-row, {len(nopad)} rows with no pad target: max |diff| {dx[j]:.2e} at row {nopad[j]}"
          f"; rows with pad targets {[k for k in range(n) if k not in nopad]} have no banked pad-excluded value")
else:
    print("  pad-excluded per-row: the new file has no per_row_mean_ce_excluding_pad_targets field")

# stack on each side
for name, r in (("neighbour", nb), ("target", tg), ("known new", kn), ("known banked", kb)):
    print(f"stack {name}: gpu {r['gpu']}  instance {r['instance_type']}  torch {r['torch']}"
          f"  transformers {r['transformers']}  dtype {r['dtype']['parameters']}  attn {r['attn_implementation']}")
