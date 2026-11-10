"""m08: switching a side off at its group's positions, one held-out birth in 32-bit, token by token.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M08_DATA = {...}. Prints `path sha256` for every input.

The per-token losses come from the head-fetched copy of the N2V job's losses/t05.npz (sources/SOURCES.md): one birth,
side L03N9245+ (group c4), read at snapshot 5, over its five unit windows. Key `<birth>|<window>|<kind>|<q>`; q is the
side's place in N12's switched-off list (infra/pg/n2v.py docstring, N2V_SPEC.md §0.5): 0 the carrier, 1 to 3 the three
other sides of the layer that scored highest one snapshot before the birth; the clean row has q None, the rounding check
q 0. This script recomputes each side's window-summed loss change from the rows and compares it with n2v_test.jsonl.

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

NPZ = "studies/already-clean/walkthrough/sources/n2v_losses_t05.npz"
BIRTHS = "studies/next/m2/births.json"
PAIRS = "studies/next/n2v/banks/score-02/n2v/n2v_test.jsonl"
VERDICT = "studies/next/n2v/verdict_n2v.json"
METRICS = "studies/next/n2v/banks/score-02/n2v/metrics.json"
PROJECTION = "studies/next/n2v/projection.json"
BIRTH = "L03N9245+:c4"
WINDOW = 2048                                # m2.WINDOW: corpus tokens per window; input 0 is the start-of-text token
N_SIDES = 4                                  # n2v.N_POLES: the carrier and the three other sides
ROLE = ["the cluster's own pole", "another pole of the layer, ranked 1st at the checkpoint before the onset",
        "another pole of the layer, ranked 2nd at the checkpoint before the onset",
        "another pole of the layer, ranked 3rd at the checkpoint before the onset"]


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
    data["meta"] = {"commit": commit, "link": "studies/next/n2v/banks/score-02/n2v/n2v_test.jsonl",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def f32_list(a):
    """Each float32 as the shortest decimal that reads back to the same float32 (asserted): an exact encoding."""
    out = [float(str(v)) for v in a]
    assert np.array_equal(np.array(out, np.float32), a), "a float32 did not read back exactly"
    return out


def unit_windows(tok):
    """n2v.unit_windows: {window: input indices} of the positions at input index 1 to 2,047."""
    byw = {}
    for e in tok:
        i = 1 + e % WINDOW
        if i <= WINDOW - 1:
            byw.setdefault(e // WINDOW, []).append(i)
    return {w: sorted(byw[w]) for w in sorted(byw)}


def main():
    births = json.loads(src(BIRTHS).read_text())["births"]
    b = next(x for x in births if x["id"] == BIRTH)
    uw = unit_windows(b["tok"])
    z = np.load(src(NPZ))
    rows = [json.loads(l) for l in src(PAIRS).read_text().splitlines() if l.strip()]
    pr = sorted([r for r in rows if r["unit"] == BIRTH], key=lambda r: r["rank"])
    assert [r["rank"] for r in pr] == list(range(N_SIDES)), pr
    assert pr[0]["pole"] == b["pole"] and pr[0]["t"] == b["k"] - 1 and pr[0]["n_windows"] == len(uw)
    want = {f"{BIRTH}|{w}|{k}|{q}" for w in uw for k, q in [("clean", None), ("probe", 0)] +
            [(k, q) for k in ("zero", "fd", "jvp") for q in range(N_SIDES)]}
    assert set(z.files) == want, sorted(set(z.files) ^ want)

    # one row per window: clean and carrier-switched-off losses, their difference, the window sum
    windows, d_by_q, j_by_q, own_by_q, probe = [], {q: 0.0 for q in range(N_SIDES)}, {q: 0.0 for q in range(N_SIDES)}, \
        {q: 0.0 for q in range(N_SIDES)}, 0.0
    for w, idx in uw.items():
        clean = z[f"{BIRTH}|{w}|clean|None"]
        assert clean.dtype == np.float32 and clean.shape == (WINDOW,)
        for q in range(N_SIDES):
            d = z[f"{BIRTH}|{w}|zero|{q}"].astype(np.float64) - clean.astype(np.float64)
            d_by_q[q] += float(d.sum())
            own_by_q[q] += float(d[idx].sum())
            j_by_q[q] += float(z[f"{BIRTH}|{w}|jvp|{q}"].astype(np.float64).sum())
        probe += float((z[f"{BIRTH}|{w}|probe|0"].astype(np.float64) - clean.astype(np.float64)).sum())
        off = z[f"{BIRTH}|{w}|zero|0"]
        i0 = idx[0]
        assert np.array_equal(off[:i0], clean[:i0]), f"window {w}: a loss before the first position changed"
        diff = off.astype(np.float64) - clean.astype(np.float64)
        windows.append({
            "window": int(w), "group_positions": idx, "first_changed_loss_index": i0,
            "first_loss_that_differs": int(np.flatnonzero(diff)[0]),
            "loss_clean": f32_list(clean),
            "loss_switched_off_from_first_position": f32_list(off[i0:]),
            "difference_from_first_position": [float(x) for x in diff[i0:]],
            "window_sum": float(diff.sum()),
            "tokens_changed": int((diff != 0).sum()),
            "window_sum_32bit_prediction": float(z[f"{BIRTH}|{w}|jvp|0"].astype(np.float64).sum()),
            "window_sum_rounding_check": float((z[f"{BIRTH}|{w}|probe|0"].astype(np.float64)
                                                - clean.astype(np.float64)).sum())})

    sides = []
    for q, r in enumerate(pr):
        sides.append({
            "side": r["pole"], "role": ROLE[q], "place_in_list": q,
            "loss_change_32bit": compare(f"{r['pole']} summed loss change 32-bit", d_by_q[q], r["dL"],
                                         f"{PAIRS} unit {BIRTH} rank {q} dL", 1e-12),
            "loss_change_own_positions_32bit": compare(f"{r['pole']} own-position loss change", own_by_q[q], r["dL_own"],
                                                       f"{PAIRS} unit {BIRTH} rank {q} dL_own", 1e-12),
            "prediction_32bit": compare(f"{r['pole']} 32-bit prediction", j_by_q[q], r["pred_jvp"],
                                        f"{PAIRS} unit {BIRTH} rank {q} pred_jvp", 1e-12),
            "prediction_16bit": r["pred_bf16"], "measurement_16bit": r["meas_bf16"]})
    rounding = compare("rounding check summed over the five windows", probe, pr[0]["probe"],
                       f"{PAIRS} unit {BIRTH} probe", 1e-12)

    # the study-wide numbers the caption uses
    v = json.loads(src(VERDICT).read_text())["test"]
    val = v["window"]["validation"]
    assert val["prediction"] == "pred_bf16" and val["measurement"] == "dL"
    probe_med = float(np.median([abs(r["probe"]) for r in rows if r["rank"] == 0]))
    n_uw = sum(len(unit_windows(x["tok"])) for x in births)
    pts = json.loads(src(METRICS).read_text())["points"]
    proj = json.loads(src(PROJECTION).read_text())
    assert proj["unit_windows"] == sum(p["unit_windows"] for p in pts.values())
    step = {int(t): int(p["download"]["branch"].split("step")[1]) for t, p in pts.items()}

    data = {
        "birth": {"id": BIRTH, "side": b["pole"], "layer": b["layer"], "unit": b["neuron"],
                  "side_sign": "positive" if b["pole"].endswith("+") else "negative", "group_name": b["name"],
                  "group_size": b["size"], "group_positions_in_corpus": b["tok"], "birth_snapshot": b["k"],
                  "birth_step": b["step_k"], "measured_at_snapshot": pr[0]["t"], "measured_at_step": step[pr[0]["t"]],
                  "half": {"test": "held_out", "freeze": "rule_setting"}[b["half"]], "unit_windows": len(uw)},
        "windows": windows,
        "loss_change_summed_over_windows": sides[0]["loss_change_32bit"],
        "sides_switched_off": sides,
        "rounding_check_summed_over_windows": rounding,
        "rank_agreement_held_out": {"prediction": "16-bit gradient prediction", "measurement": "32-bit switch-off",
                                    "n_pairs": val["n_pairs"], "n_births": val["n_units"], "value": val["rho"],
                                    "band_low": val["lo"], "band_high": val["hi"],
                                    "shuffle_reference": val["shuffle_q975"],
                                    "banked_at": f"{VERDICT} -> test.window.validation"},
        "rounding_check_median_held_out": compare("median |rounding check| over held-out births", probe_med,
                                                  v["probe_median_abs"], f"{VERDICT} -> test.probe_median_abs", 1e-15),
        "unit_windows_all_births": {"births": len(births),
                                    "windows": compare("unit windows over the 153 births", n_uw,
                                                       sum(p["unit_windows"] for p in pts.values()),
                                                       f"{METRICS} -> sum of points.*.unit_windows", 0)},
    }
    print(f"PLAN 0.118 -> rank_agreement_held_out.shuffle_reference ({val['shuffle_q975']})")
    print(f"PLAN 0.960 -> rank_agreement_held_out.value ({val['rho']})")
    print(f"PLAN 1.47e-4 -> rounding_check_median_held_out.recomputed ({probe_med})")
    print(f"PLAN 987 -> unit_windows_all_births.windows.recomputed ({n_uw})")
    print(f"PLAN 153 -> unit_windows_all_births.births ({len(births)})")
    print(f"PLAN dL {pr[0]['dL']} -> loss_change_summed_over_windows.banked")
    print(f"PLAN 2,048 -> windows[*].loss_clean (length {len(windows[0]['loss_clean'])})")
    notes = {
        "source of the per-token rows": f"{NPZ}, the N2V job's losses/t05.npz copied once from S3 by the head "
                                        "(sources/SOURCES.md, same sha256 as READ_n2v.md §3).",
        "npz key": "`<birth>|<window>|<kind>|<q>`; kind clean, zero (= switched off), fd (h scaled by 0.9), jvp "
                   "(forward-mode derivative of each token's loss along -h, the 32-bit prediction), probe (= rounding "
                   "check); q is the side's place in N12's list: 0 the carrier, 1 to 3 the three other sides of the "
                   "layer with the highest ranking score one snapshot before the birth (infra/pg/n2v.py docstring; "
                   "N2V_SPEC.md §0.5); the clean row has q None and the rounding check q 0. Decoding verified: every "
                   "side's recomputed sums equal its banked dL, dL_own and pred_jvp exactly.",
        "birth": "From studies/next/m2/births.json -> births (id L03N9245+:c4): born at snapshot 6 (step 12,000); the "
                 "switch-off is read at snapshot 5 (step 8,000), the snapshot before the birth (n2v_test.jsonl field "
                 "t = 5). sources/SOURCES.md says 'born at snapshot 5'; the record says snapshot 6.",
        "group_positions_in_corpus": "births.json `tok`: corpus token indices of the group's positions; window = "
                                     "index // 2048, input index = 1 + index mod 2048 (n2v.unit_windows).",
        "windows[].group_positions": "Input indices (1 to 2,047) in that window where the carrier's input to the down "
                                     "matrix is set to 0; input 0 is the start-of-text token.",
        "loss index j": "loss_clean[j] is the next-token loss of input j predicting input j+1, j = 0 ... 2,047, "
                        "32-bit; a change at input i can reach only losses j >= i.",
        "loss_clean": "All 2,048 per-token losses of the clean forward, float32, each written as the shortest "
                      "decimal that reads back to the same float32 (asserted), so no value is altered.",
        "loss_switched_off_from_first_position": "The carrier-switched-off losses from j = first_changed_loss_index to "
                                                 "2,047; before that index they are bitwise equal to loss_clean "
                                                 "(asserted), so they are not repeated. Kept at full resolution for all "
                                                 "five windows instead of selecting windows.",
        "difference_from_first_position": "Switched off minus clean per token, float64 difference of the float32 "
                                          "losses (as n2v.pair_values computes it), from first_changed_loss_index; "
                                          "earlier differences are exactly 0.",
        "window_sum": "Sum of the 2,048 differences of the window, nats (a sum, not a mean).",
        "loss_change_summed_over_windows": "Delta L of the carrier: the five window sums added (recomputed) against "
                                           "the banked dL of n2v_test.jsonl.",
        "sides_switched_off": "All four switched-off sides of this birth (n2v_test.jsonl rows of unit L03N9245+:c4). "
                              "dL -> loss_change_32bit, dL_own -> loss_change_own_positions_32bit (the same sum over "
                              "the group's positions only), pred_jvp -> prediction_32bit, pred_bf16 -> "
                              "prediction_16bit (N12's banked sum of -h G), meas_bf16 -> measurement_16bit (N12's "
                              "banked 16-bit window-summed switch-off).",
        "window_sum_32bit_prediction": "Sum over the window of the jvp row (the 32-bit forward-mode prediction) for "
                                       "the carrier.",
        "rounding check": "probe in the bank: the carrier's output change added and subtracted in the same forward, "
                          "zero in exact arithmetic; its summed loss change is the measurement's floor.",
        "rank_agreement_held_out": "verdict_n2v.json test.window.validation: Spearman rank agreement of pred_bf16 "
                                   "with dL over 284 held-out pairs; band = percentile bootstrap over births; "
                                   "shuffle_reference = 97.5th percentile over 1,000 shuffles (shuffle_q975).",
        "rounding_check_median_held_out": "Median over the 71 held-out births of |probe| (rank-0 rows), as "
                                          "n2v.verdict computes probe_median_abs.",
        "unit_windows_all_births": "Distinct windows holding a group position at input index 1 to 2,047, summed over "
                                   "the 153 births of births.json, against the job's per-snapshot counts.",
        "gap": "The window's token text is not in the repository (the corpus tokens are on S3); only the group's name "
               "from births.json is carried.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
