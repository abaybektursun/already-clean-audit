"""MENDEL MN-13 — arrangement co-location per hop (queue item 26b, representation half).

WHY THIS ARM RUNS AT ALL. The roster gates MN-13 on MN-10, and MN-10 returned NO-VARIANT-PASSES.
That kills ONE of MN-13's three legs, not the arm:

  * arrangement-via-repaired-Bures  BLOCKED. MN-10's kill rule fired — no cost variant passed its
    own split-half stability gate — so the repaired-Bures instrument does not exist. Running this
    leg would mean scoring with an instrument measured unable to hold still.
  * arrangement-via-TESSERA         RUNS. This job. The k1 banks hold, per layer and per faculty,
    the eigenbasis of the task-curvature operator Q in RESIDUAL COORDINATES — the same axis
    core32 lives on for residual-input types. Coordinate-level, same 18-layer PG1 lineage.
  * the LATLOC L16/26 locus         BLOCKED, and not by MN-10. LATLOC is a causal-patching MDE on
    the 26-LAYER gemma-2 lineage reported at LAYER grain (`delta_rf_profile`, one number per
    layer). It has no coordinate-level locus to co-locate, and its layer index is not MENDEL's:
    importing it would cross the generation firewall on numbers. Recorded, not attempted.

THE OBJECT. A "hop" here is a faculty pair. TESSERA banks Q per faculty (caption / ocr / vqa), so
the ARRANGEMENT CHANGE across a hop is the difference of the two stiff projectors,

    dP = S_A S_A^T - S_B S_B^T,     S = Q_evecs[:, :k]  (top-k by curvature eigenvalue)

whose per-coordinate energy says WHERE the arrangement moved. That vector is then asked MN-2's
question exactly: does the change land in the sink more than in an energy-matched subspace, or
than in the random band?

Projectors rather than raw eigenvectors because eigenvectors are only defined up to sign and up to
rotation inside a degenerate eigenvalue block; the projector onto the top-k span is invariant to
both, so a hop's dP measures a real change of subspace and not a relabelling of one.

k IS FIXED ACROSS FACULTIES AND LAYERS, and is not the banked `k_stiff`. Using each faculty's own
k_stiff would make dP's rank differ by faculty, so a hop between a rank-40 and a rank-12 stiff
subspace would register a large change from the rank gap alone. k=32 is the pre-registered primary
because it is core32's width; k in {8, 128} ride along as a ladder and are reported, never
substituted for the primary.

FROZEN RULE (mirrors MN-2's, per the roster's `kill`): SINK-COUPLED iff the paired sink-minus-
energy-matched contrast clears zero in either direction AND the sink sits outside the band on
cells spanning >= 2 types and >= 2 layers; ENERGY-ONLY iff the sink fraction never exceeds the
energy-matched band; else AMBIGUOUS.

CPU only, banked artifacts. FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_transport as MT
from infra.pg.endow_writefrac import write_fraction

ARM = "MN-13"
BANKS_S3 = {
    "caption": "s3://pg-lineage-769063704696/results/tessera_k1_banks_caption2/out/"
               "tessera_k1_banks_caption.pt",
    "ocr": "s3://pg-lineage-769063704696/results/tessera_k1_banks_ocr3/out/"
           "tessera_k1_banks_ocr.pt",
    "vqa": "s3://pg-lineage-769063704696/results/tessera_k1_banks_vqa/out/"
           "tessera_k1_banks_vqa.pt",
}
K_PRIMARY = 32
K_LADDER = (8, 32, 128)
N_HAAR = 64

# The types whose INPUT IS THE RESIDUAL STREAM — named explicitly, NOT taken from
# `H.TYPES_RESID`. That constant is the set of 2048-input types, and it includes `o_proj`, whose
# input is the concatenated attention-head output. On this architecture that space is ALSO
# 2048-dimensional, so every shape assertion passes while the comparison is between two different
# spaces that happen to share a dimension — the failure would be silent and the number meaningless.
#
# q/k/v_proj read the post-input-layernorm residual and gate/up_proj the post-post-attention one;
# layernorm is elementwise, so coordinates survive it and these five are on TESSERA's axis.
# down_proj (16384-dim MLP hidden) is excluded by dimension and would have failed loudly anyway.
TYPES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
assert "o_proj" not in TYPES and set(TYPES) < set(H.TYPES_RESID)


def stiff_projector(evecs: torch.Tensor, evals: torch.Tensor, k: int) -> np.ndarray:
    """P = S S^T for S the top-k eigenvectors of Q, ordered by DESCENDING eigenvalue.

    The banked order is not assumed — it is imposed here, because `core32` is banked ASCENDING
    (it is the bottom of a Gram) and reading one convention with the other's assumption is exactly
    the class of error that produced this program's retracted results.
    """
    ev = evals.double().numpy()
    order = np.argsort(-ev)[:k]
    S = evecs.double().numpy()[:, order]
    return S @ S.T


def cell_controls(U_sink, evecs, evals, k_bank: int, seed: int) -> dict:
    """The energy-matched subspace and the haar band for ONE cell, on the REAL activation basis.

    Built once per (type, layer) and shared across hops, because a control is a property of the
    CELL, not of the hop being scored. Rebuilding it per hop would also make the three hops'
    controls differ by seed, so a hop-to-hop difference could come from the controls rather than
    from the arrangement — the comparison this arm exists to make.

    G4b's banked activation Gram eigenbasis, not the diagonal per-coordinate proxy MN-2 and MN-6
    shipped with. G4b's own metrics name MN-13 as the consumer of exactly this. The difference is
    not cosmetic: a diagonal control can only select whole coordinates, so it cannot match a sink
    that lies obliquely to the coordinate frame, and core32 is an eigenbasis of a Gram with no
    reason to be axis-aligned.

    The (evecs, evals) pair must arrive ALIGNED — same width — which is what
    `MC.load_activation_eigenbank` guarantees. G4b banks the vectors truncated to its top 512 and
    the values in full, and doing that alignment at each call site is what killed MN-13 v1 and
    MN-7 v2 with the same IndexError.
    """
    V = np.asarray(evecs, dtype=np.float64)
    e = np.asarray(evals, dtype=np.float64)
    assert V.ndim == 2 and e.ndim == 1 and V.shape[1] == e.size, (
        f"eigenbank not aligned: {V.shape[1]} vectors vs {e.size} values — load it with "
        f"MC.load_activation_eigenbank, which cuts the full spectrum to the banked width")
    e_sink = MC.sink_energy_share(V, e, U_sink)
    U_energy, match = MC.energy_matched_control_gram(V, e, k_bank, e_sink, seed)
    rng = np.random.default_rng(seed + 7)
    haar = [np.linalg.qr(rng.standard_normal((U_sink.shape[0], k_bank)))[0]
            for _ in range(N_HAAR)]
    return {"U_energy": U_energy, "haar": haar, "e_sink": e_sink, "match": match}


def hop_cell(dP: np.ndarray, U_sink, ctrl: dict) -> dict:
    """MN-2's contrast on an arrangement change, with the full control trinity."""
    cell = MT.colocation_cell(dP, U_sink=U_sink, U_energy=ctrl["U_energy"],
                              U_haar_list=ctrl["haar"])
    band = MC.exceeds_band(cell["sink"], min(cell["energy"], cell["haar_lo"]),
                           max(cell["energy"], cell["haar_hi"]))
    m = ctrl["match"]
    return {**cell, **band, "e_sink_target": ctrl["e_sink"],
            "e_matched_actual": m["e_achieved"], "energy_match_rel_error": m["rel_error"],
            "energy_matched_ok": m["matched"]}


def aggregate(cells, matches, trace_shares, *, hops, layers, queue_item: str, seed: int) -> dict:
    """Everything after the grid — the shuffle, the frozen rule, the robustness pair, the gate
    payloads and the metrics body — as a PURE function of the computed cells.

    EXTRACTED SO THE DRY-RUN GATE CAN EXECUTE IT. Three consecutive MN-13 launches died in this
    region while a 244-test suite passed green: a two-group permutation test handed three hop
    labels, then a report key spelled `frac_matched` against a producer that returns
    `fraction_matched`. Neither is reachable by testing pure helpers, and each cost a full
    instance and the entire 810-cell grid to discover — the most expensive possible place to find
    a one-word mistake. Nothing here touches S3, the atlas writer or the filesystem, so a test can
    run the whole path on synthetic cells in milliseconds.
    """
    prim = [c for c in cells if c["k_stiff"] == K_PRIMARY]
    assert prim, f"no cells at the primary k_stiff={K_PRIMARY}"
    assert len(prim) < len(cells), (
        f"every one of {len(cells)} cells is at the primary rung — the ladder collapsed, which is "
        f"what a key collision on the rung label looks like")

    # THREE hops, so the k-group form. `permutation_pvalue` is structurally two-group (its
    # statistic is a difference of means and it enumerates with comb(n, n_a)); handed three labels
    # it asserts, which it should.
    perm = MC.permutation_pvalue_multi([c["sink"] for c in prim], [c["hop"] for c in prim],
                                       seed=seed)
    paired = MC.paired_delta_ci([c["sink"] for c in prim], [c["energy"] for c in prim], seed=seed)

    # THE FROZEN RULE IS ONE-SIDED, because MN-2's is and the roster says MN-13 mirrors it:
    # SINK-COUPLED requires the arrangement change to land in the sink MORE than in the
    # energy-matched control. A change landing LESS returns ENERGY-ONLY under the rule as written,
    # however large the gap. An earlier version scored it two-sided; the spec does not license
    # that and no freeze amendment adds it. See MN-2 for the same correction.
    above = [c for c in prim if c["above_band"]]
    below = [c for c in prim if c["below_band"]]
    n_types = len({c["type"] for c in above})
    n_layers = len({c["layer"] for c in above})
    if paired["ci_lo"] > 0 and n_types >= 2 and n_layers >= 2:
        verdict = "SINK-COUPLED"
    elif not above:
        verdict = "ENERGY-ONLY"
    else:
        verdict = "AMBIGUOUS"
    verdict = MP.check_verdict(ARM, verdict)

    match_report = MC.energy_match_report(matches)
    ok = [c for c in prim if c["energy_matched_ok"]]
    paired_matched = (MC.paired_delta_ci([c["sink"] for c in ok], [c["energy"] for c in ok],
                                         seed=seed) if len(ok) >= 8 else None)
    paired_haar = MC.paired_delta_ci([c["sink"] for c in prim], [c["haar_mean"] for c in prim],
                                     seed=seed)
    by_k = {k: MC.paired_delta_ci([c["sink"] for c in cells if c["k_stiff"] == k],
                                  [c["energy"] for c in cells if c["k_stiff"] == k], seed=seed)
            for k in sorted({c["k_stiff"] for c in cells})}
    assert len(by_k) == len(K_LADDER), (
        f"banked {len(by_k)} rungs of the {len(K_LADDER)}-rung ladder: {sorted(by_k)}")

    gate_rows = {
        "mn13_sink_ABOVE_energy_band": (
            bool(paired["ci_lo"] > 0),
            {"n_above_types": n_types, "n_above_layers": n_layers, "n_cells_above": len(above),
             "n_cells_below": len(below), **paired}),
        "mn13_energy_controls_actually_matched": (
            match_report["fraction_matched"] >= 0.9, dict(match_report)),
        "mn13_survives_on_matched_cells_only": (
            bool(paired_matched is not None and
                 (paired_matched["ci_hi"] < 0 or paired_matched["ci_lo"] > 0)),
            {"n_cells": len(ok), **(paired_matched or {})}),
        "mn13_survives_against_the_haar_band": (
            bool(paired_haar["ci_hi"] < 0 or paired_haar["ci_lo"] > 0), dict(paired_haar)),
    }
    metrics = {
        "arm": ARM, "verdict": verdict, "queue_item": queue_item,
        "hops": ["->".join(h) for h in hops], "layers": list(layers), "types": list(TYPES),
        "k_primary": K_PRIMARY, "k_ladder": list(K_LADDER), "n_cells": len(cells),
        "sink_vs_energy": paired, "k_ladder_contrasts": by_k, "hop_shuffle": perm,
        "energy_match_report": match_report,
        "banked_span_trace_share": {
            "median": float(np.median(trace_shares)), "min": float(np.min(trace_shares)),
            "max": float(np.max(trace_shares)),
            "what": ("fraction of the activation trace inside G4b's banked top-512 span. e_sink "
                     "and the control's achieved energy are both measured inside it, so the match "
                     "is internally consistent; this number says how much of the activation "
                     "energy that span holds")},
        "energy_matched_basis": (
            "G4b's banked activation Gram eigenbasis (top 512 per site), NOT the diagonal "
            "per-coordinate proxy MN-2 and MN-6 shipped with. G4b's metrics name MN-13 as the "
            "consumer of exactly this. e_sink and the control's achieved energy are both measured "
            "inside the same truncated span, so the match is internally consistent"),
        "sink_vs_energy_matched_cells_only": paired_matched,
        "sink_vs_haar_band": paired_haar,
        "observed_direction": {
            "n_cells_above_band": len(above), "n_cells_below_band": len(below),
            "above_types": n_types, "above_layers": n_layers,
            "sink_mean": paired["treatment_mean"],
            "energy_matched_mean": paired["control_mean"],
            "note": ("the frozen rule is one-sided; this block describes what was measured, "
                     "whichever way it points, and is not a second verdict")},
        "legs_not_attempted": {
            "repaired_bures_arrangement": (
                "BLOCKED by MN-10 NO-VARIANT-PASSES — no cost variant passed its own split-half "
                "stability gate, so the repaired-Bures instrument does not exist"),
            "latloc_L16_of_26_locus": (
                "BLOCKED and NOT by MN-10 — LATLOC reports at LAYER grain on the 26-layer gemma-2 "
                "lineage; there is no coordinate-level locus to co-locate and its layer index is "
                "not MENDEL's, so importing it would cross the generation firewall on numbers")},
    }
    headline = {"verdict": verdict, "delta": round(paired["delta_mean"], 5),
                "ci": [round(paired["ci_lo"], 5), round(paired["ci_hi"], 5)],
                "n_above": len(above), "n_below": len(below),
                "fraction_matched": round(match_report["fraction_matched"], 3)}
    return {"verdict": verdict, "prim": prim, "shuffle": perm, "gates": gate_rows,
            "metrics": metrics, "headline": headline}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-13 — arrangement co-location per hop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn13", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn13"), out)

    # ---- instrument validity BEFORE the treatment (§9): plant a co-location and detect it ------
    dW_p, U_p = MT.planted_colocation(n=H.IN_DIM["q_proj"], k=K_PRIMARY, out_dim=64,
                                      seed=a.seed, strength=0.81)
    void = MT.instrument_is_void(float(write_fraction(dW_p, U_p)), 0.9)
    gates.record("mn13_instrument_detects_a_planted_colocation", not void["void"], **void)
    assert not void["void"], f"instrument is VOID, refusing to score a treatment: {void}"

    banks = {}
    for fac, uri in sorted(BANKS_S3.items()):
        p = man.require(f"tessera_k1_{fac}", uri)
        banks[fac] = torch.load(p, map_location="cpu", weights_only=False)
        print(f"[mn13] {fac}: {len(banks[fac])} layers banked", flush=True)

    core = WT.fetch_banks(H.cache_dir(out, "banks"))
    vec = man.require("g4b_vectors", f"{MC.RESULTS_S3}/mendel_g4b_v1/out/vectors",
                      recursive=True)
    layers = sorted(set.intersection(*(set(b) for b in banks.values())))
    hops = sorted(itertools.combinations(sorted(banks), 2))
    print(f"[mn13] {len(hops)} hops x {len(layers)} layers x {len(TYPES)} types "
          f"x {len(K_LADDER)} k", flush=True)

    cells, matches = [], []
    for k in K_LADDER:
        proj = {(f, L): stiff_projector(banks[f][L]["Q_evecs"], banks[f][L]["Q_evals"], k)
                for f in banks for L in layers}
        # controls first, once per cell, shared by every hop scored on that cell
        ctrl, trace_shares = {}, []
        for L in layers:
            for t in TYPES:
                U = WT.load_bank(core, t, L, "core32")
                assert U.shape[0] == proj[(sorted(banks)[0], L)].shape[0], (
                    f"axis mismatch at {t} L{L}: core32 is {U.shape[0]}-dim but TESSERA's Q is "
                    f"{proj[(sorted(banks)[0], L)].shape[0]}-dim — different spaces")
                ev, ea, share = MC.load_activation_eigenbank(vec, t, L)
                c = cell_controls(U, ev, ea, U.shape[1], H.cell_seed(a.seed, ARM, t, L, k))
                c["trace_share"] = share
                ctrl[(t, L)] = {**c, "U": U}
                trace_shares.append(c["trace_share"])
                matches.append(c["match"])
        for (fa, fb) in hops:
            hop = f"{fa}->{fb}"
            for L in layers:
                dP = proj[(fa, L)] - proj[(fb, L)]
                for t in TYPES:
                    c = ctrl[(t, L)]
                    # THE LADDER RUNG IS `k_stiff`, NOT `k`. MT.colocation_cell returns its own
                    # "k" — the width of the sink bank, always 32 — and a spread that lands after
                    # a literal "k" silently overwrites it. That is exactly what happened: every
                    # row was stamped k=32, the primary filter then selected all 810 cells instead
                    # of 270, and the headline contrast silently pooled all three rungs. The job
                    # exited 0 with every gate recorded. Distinct names, and the spread first.
                    row = {**hop_cell(dP, c["U"], c),
                           "hop": hop, "type": t, "layer": L, "k_stiff": k}
                    cells.append(row)
                    if k == K_PRIMARY:
                        for name, val, cid in (("sink", row["sink"], MC.TREATMENT),
                                               ("energy_matched", row["energy"], MC.STRUCTURED),
                                               ("haar", row["haar_mean"], MC.HAAR)):
                            at.add("arrangement_write_fraction", value=val, layer=L, type=t,
                                   bank_id=name, hop_id=hop, control_id=cid,
                                   flags={"k": k, "e_sink_target": row["e_sink_target"],
                                          "energy_matched_ok": row["energy_matched_ok"]})
            print(f"[mn13] k={k} {hop}: done {len(layers)} layers", flush=True)

    agg = aggregate(cells, matches, trace_shares, hops=hops, layers=layers,
                    queue_item=meta["item"], seed=a.seed)
    at.add_ci("arrangement_write_fraction", [c["sink"] for c in agg["prim"]], bank_id="sink",
              control_id=MC.SHUFFLE, seed=a.seed,
              flags={"p_hop_label_permutation": agg["shuffle"]["p_value"],
                     "method": agg["shuffle"]["method"]})
    for name, (ok, payload) in agg["gates"].items():
        gates.record(name, ok, **payload)
    at.check_control_trinity(["arrangement_write_fraction"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {**agg["metrics"], "cells": cells, "atlas": info})
    print("MN13_OK " + json.dumps(agg["headline"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
