"""m11: copying changed weights, and the conditions. Which units each condition copies, how much change energy each
carries, and how the score was taken.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M11_DATA = {...}. Prints `path sha256` for every input.

The per-unit change sizes s_n of all 352,256 units (ranking.npz, 11,273,184 bytes) stay on S3; the head fetched that file
once and reduced it to studies/already-clean/walkthrough/sources/n13_ranking_quantiles.json (a 399-point sorted curve, the
four most-changed sets' energies, 32 per-layer rows; provenance in sources/SOURCES.md). This script reads that derived
file for the sorted curve and checks its total energy against sets.json's to the last digit. Added by the head 2026-11-08
after the data worker's report named the gap.

    python3 extract.py
"""
import collections
import datetime
import glob
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

N13 = "studies/next/n13"
R1 = f"{N13}/s3/s3gate-01/r1_rank"
W = 11008                                    # units per layer (n13_arms.neuron_count: layers x intermediate_size)
ENTRIES_PER_UNIT = 3 * 4096                  # gate row + up row + down column (N13_SPEC.md §0.5)
TAG_PCT = {"k0.1": 0.1, "k0.3": 0.3, "k1": 1.0, "k3": 3.0}
TAG_NAME = {"k0.1": "0_1pct", "k0.3": "0_3pct", "k1": "1pct", "k3": "3pct"}


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


def build(arm):
    found = sorted(glob.glob(str(ROOT / f"{N13}/s4/*/{arm}/{arm}/build.json")))
    if not found:
        return None
    assert len(found) == 1, found
    b = json.loads(src(str(Path(found[0]).relative_to(ROOT))).read_text())
    assert b["arm"] == arm
    return b


def main():
    s = json.loads(src(f"{R1}/sets.json").read_text())
    ws = json.loads(src(f"{R1}/weight_stats.json").read_text())
    th = json.loads(src(f"{N13}/thresholds.json").read_text())
    met = json.loads(src(f"{N13}/s3/s3gate-03/base/math500/metrics.json").read_text())
    rep = json.loads(src(f"{N13}/s3/s3gate-03/r1_rank/reproduce_s3gate01.json").read_text())

    n = s["n_neurons"]
    L = ws["block_profile"]["n_blocks"]
    assert n == L * W == 352256 and ws["n_neurons"] == n and s["ks"] == ws["ks"]
    E = s["E_total"]

    # ---- the sizes and the four set families, per size
    sizes = []
    for j, (tag, k) in enumerate(zip(s["k_tags"], s["ks"])):
        p = TAG_PCT[tag]
        assert k == round(p / 100 * n)
        q = s["dare_q"][tag]
        rE, rT = s["rndE"][tag], s["rndT"][tag]
        assert s["top"][tag] == sorted(s["top"][tag]) and len(s["top"][tag]) == k and len(rE["neurons"]) == k
        assert rT["m"] == len(rT["neurons"]) == ws["rndT_m"][tag] and not rT["short"] and rE["c"] == ws["rndE_c"][tag]
        assert set(s["top"][tag]).isdisjoint(rE["neurons"]) and set(s["top"][tag]).isdisjoint(rT["neurons"])
        b_top, b_E, b_T, b_D = build(f"top_{tag}"), build(f"rndE_{tag}"), build(f"rndT_{tag}"), build(f"dare_{tag}")
        rec = {
            "percent_of_units": p, "units": k, "entries": k * ENTRIES_PER_UNIT,
            "most_changed": {"energy": s["E_top"][tag], "share_of_whole_change": s["E_top"][tag] / E,
                             "rest_energy": s["E_tail"][tag],
                             "check_rest": compare(f"E_tail {tag} = E_total - E_top", E - s["E_top"][tag], s["E_tail"][tag],
                                                   "sets.json E_tail", 1e-12),
                             "realized_mlp_energy": b_top["families"]["mlp"]["realized_sq"] if b_top else None},
            "random_equal_energy_scaled": {
                "units": k, "energy_before_scaling": rE["E_R"], "scale_planned": rE["c"],
                "check_scale": compare(f"rndE c {tag} = sqrt(E_top / E_R)", math.sqrt(rE["E_S"] / rE["E_R"]), rE["c"],
                                       "sets.json rndE.c", 1e-12),
                "scale_built": b_E["spec"]["c"] if b_E else None,
                "realized_mlp_energy": b_E["families"]["mlp"]["realized_sq"] if b_E else None,
                "realized_over_most_changed": b_E["families"]["mlp"]["realized_sq"] / s["E_top"][tag] if b_E else None},
            "random_equal_energy_own_change": {
                "units": rT["m"], "energy": rT["E"], "energy_over_most_changed": rT["E"] / rT["E_S"],
                "realized_mlp_energy": b_T["families"]["mlp"]["realized_sq"] if b_T else None},
            "scaled_sample": {
                "keep_probability": q, "scale": 1 / q,
                "check_scale": compare(f"1/q {tag} = 352,256 / k", n / k, 1 / q, "sets.json dare_q", 1e-9),
                "entries_kept": b_D["kept"], "entries_expected": b_D["kept_expected"],
                "realized_mlp_energy": b_D["families"]["mlp"]["realized_sq"],
                "realized_over_whole_change": b_D["families"]["mlp"]["realized_sq"] / E,
                "realized_over_most_changed": b_D["families"]["mlp"]["realized_sq"] / s["E_top"][tag]},
        }
        assert b_D["spec"]["q"] == q and b_D["spec"]["c"] == 1 / q
        sizes.append(rec)

    # ---- per layer: units of each set, and the layer's change energy
    def per_layer(units):
        c = collections.Counter(u // W for u in units)
        return [c.get(l, 0) for l in range(L)]

    sets_named = {f"most_changed_{TAG_NAME[t]}": s["top"][t] for t in s["k_tags"]}
    sets_named |= {f"random_equal_energy_scaled_{TAG_NAME[t]}": s["rndE"][t]["neurons"] for t in s["k_tags"]}
    sets_named |= {f"random_equal_energy_own_change_{TAG_NAME[t]}": s["rndT"][t]["neurons"] for t in s["k_tags"]}
    sets_named |= {"random_units_1pct_unscaled": s["rnd0"]["neurons"],
                   "random_equal_energy_own_change_1pct_second_draw": s["rndT_d2"]["neurons"],
                   "most_changed_1pct_above_rounding": s["topF"]}
    counts = {k: per_layer(v) for k, v in sets_named.items()}
    hist_check = compare("units of the most-changed 1 % per layer (sum of absolute differences)",
                         sum(abs(a - b) for a, b in zip(counts["most_changed_1pct"], ws["layer_hist_k1"])), 0,
                         "weight_stats.json layer_hist_k1", 0)
    bp = ws["block_profile"]
    layer_energy = bp["mlp_sq"]["delta"]
    energy_check = compare("whole MLP change energy = sum over layers", sum(layer_energy), E,
                           "sets.json E_total (and weight_stats.json families.mlp.energy)", 1e-9)
    assert abs(ws["families"]["mlp"]["energy"] - E) < 1e-9

    # ---- one layer: the layer holding the most of the most-changed 1 %
    pick = max(range(L), key=lambda l: (counts["most_changed_1pct"][l], -l))
    one_layer = {"layer": pick, "units_in_layer": W,
                 "rule": "the layer holding the most units of the most-changed 1 %",
                 "unit_indices": {k: sorted(u % W for u in v if u // W == pick) for k, v in sets_named.items()}}

    # ---- cumulative energy at the banked ranks (all 32 layers ranked together)
    cumulative = [{"rank": 0, "energy": 0.0}] + [{"rank": k, "energy": s["E_top"][t]} for t, k in zip(s["k_tags"], s["ks"])] \
        + [{"rank": n, "energy": E}]

    C = th["C"]
    se = met["sampling_effective"]
    data = {
        "units_total": n, "layers": L, "units_per_layer": W, "entries_per_unit": ENTRIES_PER_UNIT,
        "whole_mlp_change_energy": E,
        "sizes": sizes,
        "cumulative_energy_at_banked_ranks": cumulative,
        "per_layer": {"mlp_change_energy": layer_energy, "mlp_relative_change": bp["mlp"],
                      "attention_relative_change": bp["attn"], "units_per_set": counts,
                      "check_counts": hist_check, "check_energy": energy_check},
        "one_layer": one_layer,
        "change_by_tensor_family": {f: {"entries": r["entries"], "unchanged": r["unchanged"], "one_16bit_step": r["one_step"],
                                        "two_or_more_steps": r["above_floor"], "energy": r["energy"],
                                        "share_of_energy": ws["energy_share"][f]}
                                    for f, r in ws["families"].items()},
        "score_protocol": {"problems": th["n"], "samples_per_problem": se["n"], "temperature": se["temperature"],
                           "top_p": se["top_p"], "max_new_tokens": se["max_tokens"], "length_cap": C,
                           "first_problem_seed": se["seed"]},
        "noise_of_one_reading": {"sigma_points": th["sigma"], "difference_must_exceed_points": th["delta"],
                                 "gain_points": th["G"], "gain_required_above": th["G-GATE"]["bound"]},
        "sorted_curve": None, "top_sets_from_curve": None, "per_layer_from_curve": None,
    }
    q = json.loads(src("studies/already-clean/walkthrough/sources/n13_ranking_quantiles.json").read_text())
    assert q["meta"]["sha256"] == rep["kept_on_S3"]["out/r1_rank/ranking.npz"][1], "derived file is not from the banked ranking.npz"
    assert q["meta"]["bytes"] == rep["kept_on_S3"]["out/r1_rank/ranking.npz"][0]
    assert q["whole_mlp_change_energy"] == data["whole_mlp_change_energy"], (q["whole_mlp_change_energy"], data["whole_mlp_change_energy"])
    data["sorted_curve"] = q["sorted_curve"]
    data["top_sets_from_curve"] = q["top_sets"]
    data["per_layer_from_curve"] = q["per_layer"]
    assert se["max_tokens"] == C == 16384 and se["temperature"] == 0.6 and se["top_p"] == 0.95 and th["n"] == 500

    for num, key in [("352,256", "units_total"), ("1,000.73", "sizes[0].scaled_sample.scale"),
                     ("333.26", "sizes[1].scaled_sample.scale"), ("99.99", "sizes[2].scaled_sample.scale"),
                     ("33.33", "sizes[3].scaled_sample.scale"),
                     ("33 to 1,000 times", "sizes[*].scaled_sample.realized_over_whole_change"),
                     ("500 problems", "score_protocol.problems"), ("temperature 0.6", "score_protocol.temperature"),
                     ("top-p 0.95", "score_protocol.top_p"), ("16,384 tokens", "score_protocol.max_new_tokens"),
                     ("1.924", "noise_of_one_reading.sigma_points"), ("5.44", "noise_of_one_reading.difference_must_exceed_points"),
                     ("34.8", "noise_of_one_reading.gain_points"), ("13.60", "noise_of_one_reading.gain_required_above")]:
        print(f"PLAN {num} -> {key}")
    for r, want in zip(sizes, (1000.73, 333.26, 99.99, 33.33)):
        assert round(r["scaled_sample"]["scale"], 2) == want, (r["scaled_sample"]["scale"], want)
    print("NOTE scaled-sample realized energy / whole change:",
          [round(r["scaled_sample"]["realized_over_whole_change"], 1) for r in sizes])
    print(f"NOTE one layer: {pick} ({counts['most_changed_1pct'][pick]} of the most-changed 1 %)")

    notes = {
        "internal_names": "Banked names -> public fields: top -> most_changed; rndE -> random_equal_energy_scaled; rndT -> "
                          "random_equal_energy_own_change; dare -> scaled_sample; rnd0 -> random_units_1pct_unscaled; rndT_d2 -> "
                          "random_equal_energy_own_change_1pct_second_draw; topF -> most_changed_1pct_above_rounding; E_top -> "
                          "most_changed.energy; E_tail -> rest_energy; dare_q -> keep_probability; rndE c -> scale_planned; "
                          "rndT m -> random_equal_energy_own_change.units; k0.1/k0.3/k1/k3 -> 0_1pct/0_3pct/1pct/3pct (0.1, 0.3, 1, 3 % of units).",
        "sorted_curve": "399 points [rank, s_n] of all 352,256 units sorted by s_n descending (ranks 1-100 every rank, then "
                        "geometric), from sources/n13_ranking_quantiles.json, which the head derived once from the S3 ranking.npz "
                        "(its sha256 is checked against reproduce_s3gate01.json). top_sets_from_curve: the four most-changed sets' "
                        "energy, energy share and smallest s_n; per_layer_from_curve: median, 99th percentile and largest s_n and "
                        "the energy per layer. Values between the 399 ranks are not in the file; draw the points, not a fit.",
        "whole_mlp_change_energy": "Sum over all 352,256 units of s_n^2, s_n = sqrt(|change of gate row|^2 + |change of up row|^2 "
                                   "+ |change of down column|^2), change = mathematics model minus base in 32-bit, sums in "
                                   "64-bit (sets.json E_total).",
        "sizes": "Per size (0.1, 0.3, 1, 3 % of the 352,256 units ranked together): the four constructions. Energies are in the "
                 "same units as whole_mlp_change_energy; realized_mlp_energy is the built model's squared MLP change from its "
                 "build.json (null where no build.json is banked).",
        "random_equal_energy_scaled": "Random units outside the most-changed set, as many as it holds, with one scalar so their "
                                      "energy equals the most-changed set's; scale_planned from sets.json, scale_built after "
                                      "the stochastic-rounding fix (build.json spec.c).",
        "random_equal_energy_own_change": "Random units outside the most-changed set at their own change, added in random order "
                                          "until their energy reaches the most-changed set's; units = how many that took.",
        "scaled_sample": "Each MLP entry's change kept with probability q = units / 352,256 and multiplied by 1/q; the pool is the "
                         "MLP only; realized_over_whole_change is the realized squared change over the whole MLP change "
                         "energy (expected value 1/q).",
        "cumulative_energy_at_banked_ranks": "Energy held by the top r units of the global ranking, at the only ranks banked "
                                             "(0, the four sizes, all units). Points between them are not banked.",
        "per_layer": "Per layer 0-31: the MLP change energy and the relative change |change| / |base| of MLP and attention "
                     "(weight_stats.json block_profile), and how many units of each set fall in the layer (from sets.json).",
        "one_layer": "Unit indices within one layer for every set, for drawing the masks along that layer's 11,008 units in index "
                     "order (the per-unit sizes that would sort them are the missing file).",
        "change_by_tensor_family": "Per tensor family: entries, entries unchanged, moved by one 16-bit step, moved by two or "
                                   "more, the change energy and its share (weight_stats.json families, energy_share).",
        "score_protocol": "From the base model's banked metrics.json sampling_effective and thresholds.json: 500 problems, one "
                          "sample each, temperature 0.6, top-p 0.95, at most 16,384 new tokens, seed 1,000,000 + problem "
                          "index.",
        "noise_of_one_reading": "thresholds.json sigma, delta, G and G-GATE.bound.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
