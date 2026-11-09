"""f13: own-direction edits against random-direction edits, on the 8 control sides and the 21 pairs.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.F13_DATA = {...}. Prints `path sha256` for every input.

All 113 edits of N6's first half (edits.jsonl): 42 own-direction and 42 random-direction edits on the 21 pairs of the
half that set the rules, 8 own-direction, 8 random-direction and 8 half-fraction own-direction edits on the 8 control
sides, 4 rounding checks and 1 zero edit. Per edit the loss change, the shared-vector change, the edit size and the
side's energy; the 1,024 per-window loss changes are read (for the power check's spread) but not carried. Recomputed:
the random/own shared-vector ratio per control layer, the count of control layers where the own direction cost more,
the medians on the pairs, the 16 of 21, and the power check.

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

N6 = "studies/next/n6"
EDITS = f"{N6}/banks/L2a-01/edits.jsonl"
METRICS = f"{N6}/banks/L2a-01/metrics.json"
PAIRS = f"{N6}/pairs_L2.json"
THRESH = f"{N6}/thresholds.json"
VERDICT = f"{N6}/verdict_n6.json"
AUDIT = "studies/already-clean/walkthrough/AUDIT_PLAN_MECHANISMS_2026-11-08.md"
CONTROL_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
# AUDIT_PLAN_MECHANISMS_2026-11-08.md §3.50, a recount from edits.jsonl, typed from that row (2 decimals)
AUDIT_RATIO = [2.51, 1.83, 1.81, 1.91, 2.06, 1.75, 1.33, 1.62]
AUDIT_MEDIAN_R = {"own_direction": 60.8, "random_direction": 60.7}
KIND = {"col": "own_direction", "rand": "random_direction", "col_half": "own_direction_half_fraction",
        "probe": "rounding_check", "zero": "zero_edit"}
ROLE = {"pair": "pair", "pc": "control", "probe_pc": "rounding_check_on_control_side",
        "probe_pair": "rounding_check_on_pair_side", "zero": "zero_edit"}
HALF = {"freeze": "rule_setting", "test": "held_out"}
N_BOOT_WIN = 1000                            # n6.N_BOOT_WIN: window resamples for one pair's spread
SEED = 0                                     # n6.SEED


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


def main():
    rows = [json.loads(l) for l in src(EDITS).read_text().splitlines() if l.strip()]
    E = {r["edit"]: r for r in rows}
    assert len(E) == len(rows) == 113
    gates = json.loads(src(METRICS).read_text())["gates"]
    P = json.loads(src(PAIRS).read_text())
    th = json.loads(src(THRESH).read_text())
    vd = json.loads(src(VERDICT).read_text())
    audit = src(AUDIT).read_text()
    assert "2.51, 1.83, 1.81, 1.91, 2.06, 1.75, 1.33, 1.62" in audit and "60.8 against 60.7" in audit
    births = {}
    for p in P["pairs"]:
        births[p["early"]], births[p["late"]] = p["age_early"], p["age_late"]

    edits = []
    for r in rows:
        assert abs(r["kappa"] - r["c"] / r["norm_w"]) < 1e-12 or r["kind"] == "col_half", r["edit"]
        edits.append({"edit": r["edit"], "kind": KIND[r["kind"]], "role": ROLE[r["role"]], "layer": r["layer"],
                      "side": r["pole"], "pair": r.get("pair"), "member": r.get("member"),
                      "half": HALF[r["half"]] if "half" in r else None, "side_birth_snapshot": births.get(r["pole"]),
                      "edit_size_c": r["c"], "column_norm": r["norm_w"], "fraction_removed": r["kappa"],
                      "edit_vector_norm": r["delta_norm"], "loss_change_per_token": r["dL"],
                      "shared_vector_change": r["R"], "own_sign_energy": r["E_plus"],
                      "kl_at_own_50_positions": r["kl_eff"]})
    from_pairs = [e for e in edits if e["role"] == "pair"]
    assert all(e["half"] == "rule_setting" for e in from_pairs) and len(from_pairs) == 84

    # the 8 control sides
    tau = th["tau_eps"]
    ctrl = []
    for L, want in zip(CONTROL_LAYERS, AUDIT_RATIO):
        c, r, h = E[f"PC{L:02d}_col"], E[f"PC{L:02d}_rand"], E[f"PC{L:02d}_col_half"]
        ctrl.append({"layer": L, "side": c["pole"],
                     "shared_vector_ratio_random_over_own": compare(
                         f"layer {L} random/own shared-vector change", round(r["R"] / c["R"], 2), want,
                         f"{AUDIT} §3.50 (a recount typed from the document)", 0.0),
                     "shared_vector_ratio_random_over_own_unrounded": r["R"] / c["R"],
                     "own_cost_more_than_random": bool(c["dL"] > r["dL"] + tau),
                     "random_cost_more_than_own": bool(r["dL"] > c["dL"] + tau),
                     "full_cost_more_than_half": bool(c["dL"] > h["dL"] + tau)})
    g = gates["G_PC"]
    assert g["layers"] == CONTROL_LAYERS and g["tau"] == tau
    control = {"layers": ctrl,
               "own_cost_more_count": compare("control layers where the own direction cost more",
                                              sum(x["own_cost_more_than_random"] for x in ctrl), g["n_col_gt_rand"],
                                              f"{METRICS} -> gates.G_PC.n_col_gt_rand", 0),
               "random_cost_more_count": compare("control layers where the random direction cost more",
                                                 sum(x["random_cost_more_than_own"] for x in ctrl),
                                                 g["failing_input_swapped"],
                                                 f"{METRICS} -> gates.G_PC.failing_input_swapped", 0),
               "needed": g["need"], "of_layers": g["n_layers"],
               "full_cost_more_than_half_count": compare("control layers where the full edit cost more than half",
                                                         sum(x["full_cost_more_than_half"] for x in ctrl),
                                                         gates["G_DOSE"]["n_full_gt_half"],
                                                         f"{METRICS} -> gates.G_DOSE.n_full_gt_half", 0),
               "ratio_range": [min(x["shared_vector_ratio_random_over_own_unrounded"] for x in ctrl),
                               max(x["shared_vector_ratio_random_over_own_unrounded"] for x in ctrl)]}

    # the 21 pairs of the half that set the rules
    fz = [p for p in P["pairs"] if p["half"] == "freeze"]
    D, Dr, sds = [], [], []
    for i, p in enumerate(fz):
        ec, er, lc, lr = (E[f"{p['pair']}_{m}_{k}"] for m in ("early", "late") for k in ("col", "rand"))
        assert ec["pole"] == p["early"] and lc["pole"] == p["late"] and ec["c"] == lc["c"] == min(ec["norm_w"], lc["norm_w"])
        D.append(ec["dL"] - lc["dL"])
        Dr.append(er["dL"] - lr["dL"])
        dw = np.array(ec["dL_win"]) - np.array(lc["dL_win"])
        assert dw.size == 1024
        rng = np.random.default_rng(SEED + i)
        sds.append(float(np.std([dw[rng.integers(0, dw.size, dw.size)].mean() for _ in range(N_BOOT_WIN)])))
    D, Dr = np.array(D), np.array(Dr)
    win = float((D > tau).sum() + 0.5 * (np.abs(D) <= tau).sum())
    med = {k: float(np.median([e["shared_vector_change"] for e in from_pairs if e["kind"] == k]))
           for k in ("own_direction", "random_direction")}
    tau_s, med_abs = float(np.median(sds)), float(np.median(np.abs(D)))
    dsc, gp = th["discovery"], th["G_POWER"]
    pairs = {"pairs": compare("pairs on the rule-setting half", len(fz), dsc["n"], f"{THRESH} -> discovery.n", 0),
             "early_own_cost_more": compare("pairs where the early side's own-direction edit cost more", win,
                                            dsc["p_col"] * dsc["n"], f"{THRESH} -> discovery.p_col x n", 1e-9),
             "median_shared_vector_change_own": compare("median shared-vector change, own direction, pairs",
                                                        med["own_direction"], AUDIT_MEDIAN_R["own_direction"],
                                                        f"{AUDIT} §3.50 (a recount typed from the document)", 0.05),
             "median_shared_vector_change_random": compare("median shared-vector change, random direction, pairs",
                                                           med["random_direction"], AUDIT_MEDIAN_R["random_direction"],
                                                           f"{AUDIT} §3.50 (a recount typed from the document)", 0.05),
             "pair_difference_own": [{"pair": p["pair"], "layer": p["layer"], "early_side": p["early"],
                                      "late_side": p["late"], "early_birth_snapshot": p["age_early"],
                                      "late_birth_snapshot": p["age_late"], "difference_own": float(d),
                                      "difference_random": float(dr), "window_spread": s}
                                     for p, d, dr, s in zip(fz, D, Dr, sds)]}
    power = {"window_spread_median": compare("power check: median window-bootstrap spread of the pair difference",
                                             tau_s, gp["value"], f"{THRESH} -> G_POWER.value", 1e-18),
             "median_abs_pair_difference": compare("power check: median |pair difference|", med_abs,
                                                   gp["median_abs_D_col"], f"{THRESH} -> G_POWER.median_abs_D_col",
                                                   1e-18),
             "half_median_abs_pair_difference": 0.5 * med_abs,
             "ratio": compare("power check: spread / median |pair difference|", tau_s / med_abs, gp["ratio"],
                              f"{THRESH} -> G_POWER.ratio", 1e-12),
             "ratio_needed_at_most": 0.5, "passed": gp["pass"]}
    probes = [r for r in rows if r["kind"] == "probe"]
    f_eps = max(abs(r["dL"]) for r in probes)
    rounding = {"largest_abs_rounding_check": compare("largest |rounding check| loss change", f_eps, th["F_eps"],
                                                      f"{THRESH} -> F_eps", 0.0),
                "tie_margin": th["tau_eps"]}

    data = {"edits": edits, "control": control, "pairs_rule_setting": pairs, "power_check": power,
            "rounding": rounding, "held_out_pairs_read": vd["n_test_read"], "held_out_pairs": P["n_test"],
            "eval_windows": json.loads(src(METRICS).read_text())["w_eval"]}
    print(f"PLAN 2.51 1.83 1.81 1.91 2.06 1.75 1.33 1.62 -> control.layers[].shared_vector_ratio_random_over_own "
          f"({[x['shared_vector_ratio_random_over_own']['recomputed'] for x in ctrl]})")
    print(f"PLAN 1.3 to 2.5 -> control.ratio_range ({control['ratio_range']})")
    print(f"PLAN 2 of 8 -> control.own_cost_more_count ({control['own_cost_more_count']['recomputed']})")
    print(f"PLAN 6 of 8 (random cost more) -> control.random_cost_more_count "
          f"({control['random_cost_more_count']['recomputed']})")
    print(f"PLAN 6 needed -> control.needed ({g['need']})")
    print(f"PLAN 21 -> pairs_rule_setting.pairs ({len(fz)})")
    print(f"PLAN 16 of 21 -> pairs_rule_setting.early_own_cost_more ({win})")
    print(f"PLAN 60.8 -> pairs_rule_setting.median_shared_vector_change_own ({med['own_direction']})")
    print(f"PLAN 60.7 -> pairs_rule_setting.median_shared_vector_change_random ({med['random_direction']})")
    print(f"PLAN 1.5096e-6 -> power_check.window_spread_median ({tau_s})")
    print(f"PLAN 6.363e-7 -> power_check.half_median_abs_pair_difference ({0.5 * med_abs})")
    print(f"PLAN 1.186 -> power_check.ratio ({tau_s / med_abs})")
    print(f"PLAN 0 -> held_out_pairs_read ({vd['n_test_read']})")
    print(f"PLAN 1,024 -> eval_windows ({data['eval_windows']})")
    notes = {
        "edits": "All 113 rows of edits.jsonl in file order; the per-window loss changes (dL_win, 1,024 values per edit) "
                 "and the five largest token changes (top5) are omitted.",
        "edit": "The bank's edit id, kept as a key: PCnn_<kind> is the control side of layer nn, Pnnn_<member>_<kind> a "
                "pair edit.",
        "kind": "Bank kind col -> own_direction (delta = -(c/||w||) w), rand -> random_direction (delta = c r, r a seeded "
                "unit vector), col_half -> own_direction_half_fraction (delta = -0.5 w, control sides only), probe -> "
                "rounding_check (the random delta added and subtracted in the same forward), zero -> zero_edit "
                "(delta = 0).",
        "role": "Bank role pair -> pair, pc -> control (the layer's side with the largest own-sign energy), probe_pc / "
                "probe_pair -> rounding check on a control or a pair side, zero -> zero_edit.",
        "half": "Bank half freeze -> rule_setting; every pair edit is on the half that set the rules; the held-out half's "
                "24 pairs were never run or read.",
        "member": "early or late: the pair's side born earlier or later (births at least three snapshots apart).",
        "side_birth_snapshot": "The birth snapshot of the side's main group (pairs_L2.json age_early / age_late).",
        "edit_size_c": "c = min(||w_early||, ||w_late||) for the four edits of a pair (asserted); for a control side "
                       "c = ||w||.",
        "column_norm": "norm_w: ||w||, the side's down-matrix column norm.",
        "fraction_removed": "kappa = c / ||w|| (asserted for every edit except the half-fraction edit, whose delta is "
                            "-0.5 w).",
        "edit_vector_norm": "delta_norm: ||delta|| as realized.",
        "loss_change_per_token": "dL: mean over 1,024 windows x 2,048 tokens of (loss edited - loss base), nats per "
                                 "token, 32-bit, windows outside the 20.6-million-token corpus.",
        "shared_vector_change": "R: ||h_edit - h_base|| over all evaluation positions and 4,096 coordinates, read after "
                                "the edited layer (after the norm and the add).",
        "own_sign_energy": "E_plus: the sum over evaluation positions of the side's own-sign activation squared.",
        "kl_at_own_50_positions": "kl_eff: mean over the side's 50 strongest evaluation positions of KL(base || edited) "
                                  "of the next-token distribution, nats.",
        "control": "Per control layer: the random/own ratio of shared-vector change (rounded to 2 decimals to compare "
                   "with the audit's typed recount, and unrounded), and whether one edit's loss change exceeds the "
                   "other's by more than the tie margin tau_eps (n6 G_PC rule); the check needed the own direction to "
                   "cost more in at least 6 of 8.",
        "ratio_range": "Smallest and largest unrounded random/own ratio over the 8 control sides.",
        "pairs_rule_setting": "early_own_cost_more = pairs whose early side's own-direction edit cost more than the "
                              "late side's by more than the tie margin, ties counted one half (n6.win_share); an "
                              "exploratory reading, not a claim. The medians are over the 42 own-direction and the 42 "
                              "random-direction pair edits. difference_own = D_col (early minus late loss change), "
                              "difference_random = D_rand.",
        "window_spread": "The window-bootstrap standard deviation of one pair's own-direction difference (1,000 "
                         "resamples of the 1,024 windows, seed 0 + the pair's index on the half), recomputed from the "
                         "banked dL_win.",
        "power_check": "Banked as tau_s (thresholds.json G_POWER): the median over the 21 pairs of window_spread; "
                       "needed at most 0.5 x the median |difference_own|. The plan calls tau_s 'the rounding floor'; "
                       "the bank's rule names it the window-bootstrap spread of the pair difference.",
        "rounding": "largest_abs_rounding_check = F_eps, the largest |loss change| of the four rounding checks; "
                    "tie_margin = tau_eps = 10 x F_eps.",
        "held_out_pairs_read": "verdict_n6.json n_test_read: the held-out half was never read.",
        "audit values": "The audit's per-layer ratios and pair medians are compared as typed from "
                        "AUDIT_PLAN_MECHANISMS_2026-11-08.md §3.50; the script asserts those strings are in the file.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
