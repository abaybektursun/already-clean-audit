"""m12: an edit along a direction, on the 8 control sides, with the pair rule and the power check.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M12_DATA = {...}. Prints `path sha256` for every input.

Per control layer (0, 4, ..., 28) the side with the largest own-sign energy and its three edits (own direction at full
fraction, random direction at the same size, own direction at half fraction) from edits.jsonl: loss change,
shared-vector change, edit size. Also the rounding checks and the zero edit, the pair rule's edit size checked on the
21 pairs, and the banked power check. f13 holds every edit and recomputes the pair summaries and the power check.

    python3 extract.py
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

N6 = "studies/next/n6"
EDITS = f"{N6}/banks/L2a-01/edits.jsonl"
METRICS = f"{N6}/banks/L2a-01/metrics.json"
PAIRS = f"{N6}/pairs_L2.json"
THRESH = f"{N6}/thresholds.json"
VERDICT = f"{N6}/verdict_n6.json"
CONTROL_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
WINDOW = 2048
KIND = {"col": "own_direction", "rand": "random_direction", "col_half": "own_direction_half_fraction",
        "probe": "rounding_check", "zero": "zero_edit"}


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


def edit(r):
    return {"kind": KIND[r["kind"]], "edit_size_c": r["c"], "column_norm": r["norm_w"], "fraction_removed": r["kappa"],
            "edit_vector_norm": r["delta_norm"], "loss_change_per_token": r["dL"], "shared_vector_change": r["R"]}


def main():
    rows = [json.loads(l) for l in src(EDITS).read_text().splitlines() if l.strip()]
    E = {r["edit"]: r for r in rows}
    m = json.loads(src(METRICS).read_text())
    gates = m["gates"]
    P = json.loads(src(PAIRS).read_text())
    th = json.loads(src(THRESH).read_text())
    vd = json.loads(src(VERDICT).read_text())
    tau = th["tau_eps"]

    layers = []
    for L in CONTROL_LAYERS:
        c, r, h = E[f"PC{L:02d}_col"], E[f"PC{L:02d}_rand"], E[f"PC{L:02d}_col_half"]
        assert c["pole"] == r["pole"] == h["pole"] and c["c"] == r["c"] == c["norm_w"] and c["kappa"] == 1.0
        layers.append({"layer": L, "side": c["pole"], "own_sign_energy": c["E_plus"],
                       "edits": [edit(c), edit(r), edit(h)],
                       "shared_vector_ratio_random_over_own": r["R"] / c["R"],
                       "own_cost_more_than_random": bool(c["dL"] > r["dL"] + tau)})
    g = gates["G_PC"]
    assert g["layers"] == CONTROL_LAYERS
    own_more = sum(x["own_cost_more_than_random"] for x in layers)

    # the pair rule: c = min(||w_early||, ||w_late||), the same for the four edits; the smaller column removed fully
    fz = [p for p in P["pairs"] if p["half"] == "freeze"]
    n_rule = 0
    for p in fz:
        four = [E[f"{p['pair']}_{mm}_{k}"] for mm in ("early", "late") for k in ("col", "rand")]
        cs = {x["c"] for x in four}
        lo = min(four[0]["norm_w"], four[2]["norm_w"])
        n_rule += int(cs == {lo} and min(four[0]["kappa"], four[2]["kappa"]) <= 1.0
                      and max(four[0]["kappa"], four[2]["kappa"]) == 1.0)
    others = [{"edit": r["edit"], "kind": KIND[r["kind"]], "layer": r["layer"], "side": r["pole"],
               "loss_change_per_token": r["dL"], "shared_vector_change": r["R"]}
              for r in rows if r["kind"] in ("probe", "zero")]
    print(f"CHECK pairs whose four edits share c = the smaller column norm: {n_rule} of {len(fz)}")
    gp, dsc = th["G_POWER"], th["discovery"]

    data = {
        "control_layers": layers,
        "control_own_cost_more_count": compare("control layers where the own direction cost more", own_more,
                                               g["n_col_gt_rand"], f"{METRICS} -> gates.G_PC.n_col_gt_rand", 0),
        "control_needed": g["need"], "control_of_layers": g["n_layers"], "tie_margin": tau,
        "rounding_checks_and_zero_edit": others,
        "pairs": {"rule_setting": compare("pairs on the rule-setting half", len(fz), P["n_freeze"],
                                          f"{PAIRS} -> n_freeze", 0),
                  "held_out": P["n_test"], "held_out_read": vd["n_test_read"],
                  "edit_size_rule_holds": n_rule, "edit_size_rule_of_pairs": len(fz),
                  "early_own_cost_more_banked": dsc["p_col"] * dsc["n"], "banked_at": f"{THRESH} -> discovery"},
        "power_check": {"window_spread_median": gp["value"], "half_median_abs_pair_difference": gp["bound"],
                        "median_abs_pair_difference": gp["median_abs_D_col"], "ratio": gp["ratio"],
                        "ratio_needed_at_most": 0.5, "passed": gp["pass"], "banked_at": f"{THRESH} -> G_POWER"},
        "evaluation": {"windows": m["w_eval"], "tokens_per_window": WINDOW, "tokens": m["w_eval"] * WINDOW},
    }
    print(f"PLAN 2.51 ... 1.62 -> control_layers[].shared_vector_ratio_random_over_own "
          f"({[round(x['shared_vector_ratio_random_over_own'], 2) for x in layers]})")
    print(f"PLAN 2 of 8 (need 6) -> control_own_cost_more_count ({own_more}, need {g['need']})")
    print(f"PLAN 21 -> pairs.rule_setting ({len(fz)})")
    print(f"PLAN 16 of 21 -> pairs.early_own_cost_more_banked ({dsc['p_col'] * dsc['n']})")
    print(f"PLAN 1.51e-6 -> power_check.window_spread_median ({gp['value']})")
    print(f"PLAN 6.36e-7 -> power_check.half_median_abs_pair_difference ({gp['bound']})")
    print(f"PLAN 1.186 -> power_check.ratio ({gp['ratio']})")
    print(f"PLAN 1,024 x 2,048 -> evaluation ({m['w_eval']} x {WINDOW})")
    notes = {
        "control_layers": "One side per layer 0, 4, ..., 28: the layer's side with the largest own-sign energy "
                          "(bank role pc, edits PCnn_col, PCnn_rand, PCnn_col_half).",
        "edits[].kind": "Bank kind col -> own_direction (delta = -(c/||w||) w, here c = ||w||, so the side's write is "
                        "removed fully), rand -> random_direction (delta = c r, r a seeded unit vector, the same c), "
                        "col_half -> own_direction_half_fraction (delta = -0.5 w).",
        "edit_size_c": "c, the size of the edit; for a control side c = ||w||.",
        "column_norm": "norm_w: ||w||, the side's down-matrix column norm.",
        "fraction_removed": "kappa = c / ||w|| as banked (0.5 for the half-fraction edit).",
        "edit_vector_norm": "delta_norm: ||delta|| as realized.",
        "loss_change_per_token": "dL: mean over 1,024 windows x 2,048 tokens of (loss edited - loss base), nats per "
                                 "token, 32-bit, on windows outside the 20.6-million-token corpus.",
        "shared_vector_change": "R: ||h_edit - h_base|| over all evaluation positions and 4,096 coordinates, read "
                                "after the edited layer (after the norm and the add).",
        "own_sign_energy": "E_plus: the sum over evaluation positions of the side's own-sign activation squared.",
        "own_cost_more_than_random": "Own-direction loss change above the random one by more than the tie margin "
                                     "(tau_eps = 10 x the largest |rounding check|); the check needed this in at least "
                                     "6 of 8 layers.",
        "rounding_checks_and_zero_edit": "Bank kinds probe (the random delta added and subtracted in one forward, zero "
                                         "in exact arithmetic) and zero (delta = 0).",
        "pairs": "Two dated sides of one layer, different units, births at least three snapshots apart, matched on ln "
                 "max |a| and ln density, then on ln ||w|| and ln E_plus (calipers in N6_SPEC.md §2.3, not carried). "
                 "edit_size_rule_holds counts pairs whose four edits share c = min(||w_early||, ||w_late||) with the "
                 "smaller column removed fully. early_own_cost_more_banked is the exploratory 16 of 21, banked.",
        "power_check": "Banked (thresholds.json G_POWER; f13 recomputes it): window_spread_median = tau_s, the median "
                       "over the 21 pairs of the window-bootstrap spread of the own-direction pair difference; it "
                       "needed to be at most half the median |pair difference|. The plan calls tau_s 'the rounding "
                       "floor'; the bank's rule names it the window-bootstrap spread.",
        "held_out_read": "verdict_n6.json n_test_read: the held-out half was never read.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
