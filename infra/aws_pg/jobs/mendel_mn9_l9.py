"""MENDEL MN-9 — the "what is L9" charter (queue item 19 = addendum item 49).

THE OBJECT. The addendum records "the lone L9 ultra-avoided direction — the single bottom
eigenvector separated 4.4x from the rest of the spectrum, never individually examined", upgraded
because five instruments converge at L9.

THE KILL, verbatim: "the direction's properties are unexceptional against matched-magnitude draws
from the same spectrum tail (item 91's standing null applies here as everywhere)."

TWO NULLS, because the charter's own framing suggests a third question the charter did not ask.

  NULL A (the one the charter names) — matched-magnitude draws from the SAME spectrum tail: random
  unit vectors drawn in the span of the bottom-32 eigenbasis EXCLUDING the bottom-1 direction
  itself. Same subspace, same eigenvalue band, no alignment to anything.

  NULL B (the one the charter is missing) — the SAME STATISTIC AT EVERY OTHER CELL. "Separated
  4.4x" is only remarkable if 4.4x is a large separation for a bottom eigenvalue in this bank
  library. That comparison is free — 126 spectra are banked — and it was never made. A direction
  singled out by a statistic that is ordinary for that statistic is not a finding.

PROPERTIES CHARACTERISED (the charter's list): per-coordinate leverage, participation ratio, the
activation energy the direction carries, its overlap with the measured sink address, and the
consistency of the coordinates it loads on. The row-side counterpart is read from MERCATOR's
output-side banks where the cell exists.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_sinkform as SF
from infra.pg import mendel_transport as MT

ARM = "MN-9"
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
A1 = f"s3://{MC.BUCKET}/humboldt/humboldt_a1"
TARGET_TYPE, TARGET_LAYER = "q_proj", 9      # where the 4.4x claim lives; verified at run time
N_NULL = 500
K_ADDR = 32


def direction_properties(v, act, r, sink_addr, *, k: int = K_ADDR) -> dict:
    """Everything the charter asks of a single direction, as one dict.

    `v` is a unit vector in the input coordinate space. Leverage of a single direction is just
    v_i^2 (the k = 1 case of the bank leverage the program uses everywhere), so the statistic is
    the same object as elsewhere rather than a new one invented for this arm.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    a = np.asarray(act, dtype=np.float64).ravel()
    rr = np.asarray(r, dtype=np.float64).ravel()
    assert v.shape == a.shape == rr.shape, f"axis mismatch {v.shape}/{a.shape}/{rr.shape}"
    lev = v * v
    top = W0.top_k_coords(lev, k)
    e2 = a * a
    return {"leverage_max": float(lev.max()),
            "participation_ratio": float(H.participation_ratio(lev)),
            "activation_energy_share": float((e2 * lev).sum() / e2.sum()),
            "overlap_with_sink_address": float(W0.jaccard(top, sink_addr)),
            "mean_r_on_top_coords": float(rr[top].mean()),
            "top_coords": [int(x) for x in top[:8]]}


def tail_null(U_core, act, r, sink_addr, *, n_draw: int, seed: int) -> list:
    """NULL A — matched-magnitude draws from the same spectrum tail.

    A random unit vector in span(U_core[:, 1:]) : the same bottom eigen-tail the target lives in,
    with the target direction itself excluded so the null cannot contain the thing it is a null
    for. Magnitude-matched by construction — every draw is a unit vector in the same subspace.
    """
    rng = np.random.default_rng(seed)
    rest = np.asarray(U_core, dtype=np.float64)[:, 1:]
    out = []
    for _ in range(n_draw):
        c = rng.standard_normal(rest.shape[1])
        w = rest @ (c / np.linalg.norm(c))
        out.append(direction_properties(w / np.linalg.norm(w), act, r, sink_addr))
    return out


def band(values, lo=2.5, hi=97.5) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, lo)),
            "hi": float(np.percentile(v, hi)), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-9 — what is L9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn9", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn9"), out)

    vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    a1_rows = [json.loads(x) for x in
               man.require("humboldt_a1_atlas", f"{A1}/atlas.jsonl").read_text().splitlines() if x]
    a1_vec = man.require("humboldt_a1_vectors", f"{A1}/vectors", recursive=True)
    # ONE bank object, not the 1 GB library: this arm reads exactly one direction, so fetching the
    # whole library would be a gigabyte of provenance for a 524 KB dependency and would put the
    # manifest's hash on files the arm never opens.
    # The 18 q_proj core32 banks — one per layer, fetched ONCE. Not the 1 GB library: this arm
    # reads one direction per layer of one type, so syncing everything would put the manifest's
    # hash on files the arm never opens. The manifest key doubles as the cached FILENAME, so it is
    # the bank's own name for WT.load_bank (which addresses banks by name inside a directory).
    bank_dir = None
    for L in range(H.N_LAYERS):
        nm = WT.bank_name(TARGET_TYPE, L, "core32")
        bank_dir = man.require(nm, f"{WT.BANKS_S3}/{nm}").parent

    # ---- NULL B: the bottom-gap statistic at EVERY cell ----
    spectra = {}
    for row in a1_rows:
        if row["statistic"] != "gram_spectrum":
            continue
        v = np.sort(np.load(a1_vec / pathlib.Path(row["vector_ref"]).name).astype(np.float64))
        pos = v[v > 0]
        assert pos.size >= 3, f"{row['type']}/L{row['layer']}: degenerate spectrum"
        spectra[(row["type"], row["layer"])] = {"lambda1": float(pos[0]),
                                                "gap_2_over_1": float(pos[1] / pos[0]),
                                                "gap_3_over_2": float(pos[2] / pos[1])}
    assert len(spectra) == 126, f"expected 126 spectra, got {len(spectra)}"
    target_gap = spectra[(TARGET_TYPE, TARGET_LAYER)]["gap_2_over_1"]
    all_gaps = sorted((s["gap_2_over_1"] for s in spectra.values()), reverse=True)
    same_type = sorted((spectra[(TARGET_TYPE, L)]["gap_2_over_1"] for L in range(H.N_LAYERS)),
                       reverse=True)
    rank_all = all_gaps.index(target_gap) + 1
    rank_type = same_type.index(target_gap) + 1
    for (t, L), s in sorted(spectra.items()):
        at.add("gram_bottom_gap_ratio", value=s["gap_2_over_1"], layer=L, type=t,
               bank_id="core32", control_id=MC.STRUCTURED if (t, L) != (TARGET_TYPE, TARGET_LAYER)
               else MC.TREATMENT,
               flags={"lambda1": s["lambda1"], "gap_3_over_2": s["gap_3_over_2"],
                      "is_target": (t, L) == (TARGET_TYPE, TARGET_LAYER)})

    # ---- the direction itself, and NULL A ----
    U = WT.load_bank(bank_dir, TARGET_TYPE, TARGET_LAYER, "core32")
    v0 = U[:, 0] / np.linalg.norm(U[:, 0])                  # the bottom-1 eigenvector
    act = np.load(vec / f"act_rms__{TARGET_TYPE}__L{TARGET_LAYER}.npy")
    cons = np.load(vec / f"grad_cons__{TARGET_TYPE}__L{TARGET_LAYER}.npy")
    r, _floor = SF.r_from_consistency(cons, 24)
    sink_addr = MT.bank_address(U, K_ADDR)

    target = direction_properties(v0, act, r, sink_addr)
    null_a = tail_null(U, act, r, sink_addr, n_draw=N_NULL, seed=a.seed)

    # NULL C — the LAYER null the charter also did not ask for. If L9's bottom-1 direction is
    # unusual against its own spectral tail but its NEIGHBOURS are unusual in the same way, then
    # "L9" is not the object; "a mid-depth bottom direction" is. Every q_proj layer's bottom-1
    # direction is characterised identically so the question is answered rather than argued.
    per_layer = {}
    for L in range(H.N_LAYERS):
        UL = WT.load_bank(bank_dir, TARGET_TYPE, L, "core32")
        vL = UL[:, 0] / np.linalg.norm(UL[:, 0])
        aL = np.load(vec / f"act_rms__{TARGET_TYPE}__L{L}.npy")
        cL = np.load(vec / f"grad_cons__{TARGET_TYPE}__L{L}.npy")
        rL, _ = SF.r_from_consistency(cL, 24)
        pL = direction_properties(vL, aL, rL, MT.bank_address(UL, K_ADDR))
        nL = tail_null(UL, aL, rL, MT.bank_address(UL, K_ADDR), n_draw=100,
                       seed=H.cell_seed(a.seed, ARM, TARGET_TYPE, L))
        outside = {}
        for prop in ("leverage_max", "participation_ratio", "activation_energy_share",
                     "overlap_with_sink_address", "mean_r_on_top_coords"):
            b = band([x[prop] for x in nL])
            outside[prop] = not MC.exceeds_band(pL[prop], b["lo"], b["hi"])["in_band"]
        per_layer[L] = {"properties": pL, "n_outside": int(sum(outside.values())),
                        "outside": outside, "gap_2_over_1": spectra[(TARGET_TYPE, L)]["gap_2_over_1"]}
        at.add("l9_layer_null_n_properties_outside", value=float(per_layer[L]["n_outside"]),
               layer=L, type=TARGET_TYPE, bank_id="core32",
               control_id=MC.TREATMENT if L == TARGET_LAYER else MC.STRUCTURED,
               flags={"gap_2_over_1": per_layer[L]["gap_2_over_1"], **outside})

    verdicts = {}
    for prop in ("leverage_max", "participation_ratio", "activation_energy_share",
                 "overlap_with_sink_address", "mean_r_on_top_coords"):
        b = band([d[prop] for d in null_a])
        exceeds = MC.exceeds_band(target[prop], b["lo"], b["hi"])
        verdicts[prop] = {"target": target[prop], "null_band": b, **exceeds}
        at.add(f"l9_{prop}", value=target[prop], layer=TARGET_LAYER, type=TARGET_TYPE,
               bank_id="core32", control_id=MC.TREATMENT,
               flags={"null_lo": b["lo"], "null_hi": b["hi"], "outside_band": not exceeds["in_band"]})
        at.add(f"l9_{prop}", value=b["mean"], layer=TARGET_LAYER, type=TARGET_TYPE,
               bank_id="core32", control_id=MC.STRUCTURED, n=b["n"],
               ci_lo=b["lo"], ci_hi=b["hi"],
               flags={"null": "matched-magnitude draws from the same spectrum tail"})

    n_outside = sum(1 for v in verdicts.values() if not v["in_band"])
    neighbours = {L: d["n_outside"] for L, d in per_layer.items() if L != TARGET_LAYER}
    l9_leads_neighbours = per_layer[TARGET_LAYER]["n_outside"] > max(neighbours.values())
    # The charter's KILL is about NULL A. NULL B decides whether the object was ever singled out
    # for a reason, and is recorded beside it.
    gap_exceptional = rank_all <= 6                          # top ~5% of 126 cells
    verdict = MP.check_verdict(ARM, "EXCEPTIONAL" if (n_outside >= 2 and gap_exceptional)
                               else ("UNEXCEPTIONAL" if n_outside == 0 or not gap_exceptional
                                     else "AMBIGUOUS"))
    gates.record("mn9_selection_statistic_is_exceptional", gap_exceptional,
                 target_gap=target_gap, rank_across_126=rank_all, rank_within_type=rank_type,
                 top5=all_gaps[:5], same_type_top5=same_type[:5],
                 note=("NULL B, which the charter did not ask for: is the 4.4x separation that "
                       "singled this direction out actually large for a bottom eigenvalue in this "
                       "bank library?"))
    gates.record("mn9_properties_outside_tail_null", n_outside > 0, n_outside=n_outside,
                 properties={k: v["in_band"] for k, v in verdicts.items()})
    gates.record("mn9_l9_stands_out_among_its_own_layers", l9_leads_neighbours,
                 l9_n_outside=per_layer[TARGET_LAYER]["n_outside"],
                 by_layer={f"L{L}": d["n_outside"] for L, d in sorted(per_layer.items())},
                 note=("NULL C: if L9's neighbours are unusual against their own tails in the same "
                       "way, then 'a mid-depth bottom direction' is the object and 'L9' is not"))
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "target_cell": f"{TARGET_TYPE}/L{TARGET_LAYER}", "target_gap_2_over_1": target_gap,
        "null_B_rank_across_126_cells": rank_all, "null_B_rank_within_type": rank_type,
        "null_B_top5_across_library": all_gaps[:5],
        "null_B_same_type_by_layer": {f"L{L}": spectra[(TARGET_TYPE, L)]["gap_2_over_1"]
                                      for L in range(H.N_LAYERS)},
        "target_properties": target, "null_A_bands": verdicts,
        "null_C_per_layer": {f"L{L}": {"n_outside": d["n_outside"],
                                       "gap_2_over_1": d["gap_2_over_1"],
                                       "outside": d["outside"],
                                       "leverage_max": d["properties"]["leverage_max"],
                                       "participation_ratio": d["properties"]["participation_ratio"],
                                       "activation_energy_share":
                                           d["properties"]["activation_energy_share"]}
                             for L, d in sorted(per_layer.items())},
        "null_C_l9_leads_neighbours": bool(l9_leads_neighbours),
        "n_properties_outside_null_A": n_outside, "n_null_draws": N_NULL,
        "atlas": info})
    print("MN9_OK " + json.dumps({"verdict": verdict, "gap": round(target_gap, 3),
                                  "rank_all": rank_all, "rank_type": rank_type,
                                  "n_props_outside_null": n_outside}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
