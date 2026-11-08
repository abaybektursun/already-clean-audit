"""Recount the two N6 evidence-table rows of 20_ABSTRACT.md draft 6 from N6's files (json only).

Run from the worktree root:  python -I studies/next/abscheck/check_n6.py
Fields used from banks/L2a-01/edits.jsonl: role, kind, layer, half, dL (loss change of the edit), R (realized residual change).
"""
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
N6 = ROOT / "studies/next/n6"

verdict = json.loads((N6 / "verdict_n6.json").read_text())
th = json.loads((N6 / "thresholds.json").read_text())
gpc = json.loads((N6 / "banks/L2a-01/metrics.json").read_text())["gates"]["G_PC"]
edits = [json.loads(line) for line in (N6 / "banks/L2a-01/edits.jsonl").read_text().splitlines() if line.strip()]

print("headline", verdict["headline"])
print("failed_gates", verdict["failed_gates"], "n_test_read", verdict["n_test_read"])
gp = th["G_POWER"]
print("G_POWER value %.4e median_abs_D_col %.4e bound %.4e ratio %.4f pass %s" % (gp["value"], gp["median_abs_D_col"], gp["bound"], gp["ratio"], gp["pass"]))
assert gp == verdict["G_POWER"], "verdict and thresholds disagree on G_POWER"
d = th["discovery"]
print("discovery n %d p_col %.4f [%.4f, %.4f] p_col*21 %.2f p_rand %.4f R %.4f" % (d["n"], d["p_col"], *d["p_col_ci"], d["p_col"] * d["n"], d["p_rand"], d["R"]))
print("G_PC", {k: gpc[k] for k in ("pass", "n_col_gt_rand", "n_layers", "need", "tau", "failing_input_swapped", "layers")}, "failing_input.pass", gpc["failing_input"]["pass"])

# recount 1: PC layers where col dL exceeds rand dL by more than tau_eps
tau = th["tau_eps"]
assert tau == gpc["tau"]
pc = {(e["layer"], e["kind"]): e for e in edits if e["role"] == "pc"}
layers = sorted({layer for layer, _ in pc})
assert layers == gpc["layers"], layers
gt = [layer for layer in layers if pc[(layer, "col")]["dL"] - pc[(layer, "rand")]["dL"] > tau]
swapped = [layer for layer in layers if pc[(layer, "rand")]["dL"] - pc[(layer, "col")]["dL"] > tau]
print("recount col dL - rand dL > tau_eps: %d of %d, layers %s; swapped labels %d of %d" % (len(gt), len(layers), gt, len(swapped), len(layers)))
for layer in layers:
    c, r = pc[(layer, "col")], pc[(layer, "rand")]
    print("  layer %2d  dL col %+.4e  rand %+.4e  R col %9.3f  rand %9.3f  R rand/col %.3f" % (layer, c["dL"], r["dL"], c["R"], r["R"], r["R"] / c["R"]))
ratio = [pc[(layer, "rand")]["R"] / pc[(layer, "col")]["R"] for layer in layers]
print("R rand/col over the PC layers: %.3f-%.3f, all > 1: %s" % (min(ratio), max(ratio), all(x > 1 for x in ratio)))

# recount 2: medians of R over the freeze-pair col and rand edits
for kind in ("col", "rand"):
    vals = [e["R"] for e in edits if e["role"] == "pair" and e["kind"] == kind and e["half"] == "freeze"]
    print("freeze pair %s: n %d, median R %.3f" % (kind, len(vals), statistics.median(vals)))
print("halves present in pair edits", sorted({e["half"] for e in edits if e["role"] == "pair"}))

# GPU-hours as written in the operations note
ops = (ROOT / "infra/aws_pg/operations/2026-11-07_next_n6.md").read_text()
for pat in (r"18\.9[0-9]", r"19\.[56][0-9]"):
    print(pat, sorted(set(re.findall(pat, ops))))
