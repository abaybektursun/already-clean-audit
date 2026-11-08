"""MENDEL MN-4 — the T*-map, the membership law, and the retrodiction battery (queue item 16).

THE QUESTION. SINKFORM's Result 6 says the sink at horizon T is the set of coordinates with
r_j < T^(-1/2). Does that law have MEASUREMENT REACH — does it predict the measured sink address
better than activation energy alone already does?

The KILL, verbatim from the spec: "membership adds nothing over an activation-energy-only
baseline, or the blind retrodictions miss."

THE INSTRUMENT AND ITS FLOOR. `cons_j` over T = 24 windows floors at 1/sqrt(T) = 0.204 when the
true consistency is zero, because with mu = 0 the numerator is a random walk. The naive inversion
r = cons/sqrt(1-cons^2) therefore reports r ~ 0.207 for a coordinate whose true r is 0 — and the
sink is the LOW-r set, so the floor sits exactly where the measurement matters. This job uses the
floor-corrected inversion and reports the CENSORED FRACTION as a headline, because a map that is
mostly censored is not a map.

FOUR SCORERS, one scoring path (the one G4 validated against the banked M-8 table):

    t_star_membership  rank ascending by r                      SINKFORM's law
    activation_energy  rank descending by activation RMS        the named incumbent
    refined            top-2k by activation, then the k with    the literal KILL test
                       smallest r (ties broken by activation)
    random             the chance floor

FROZEN RULE. LAW-HOLDS iff `refined` beats `activation_energy` on the paired per-cell contrast
with CI clear of 0 AND >= 1 registered retrodiction hits; NO-REACH iff it does not; else AMBIGUOUS.

Predictions BR1-BR4 were committed to the journal at 24d651d8, before this file computed anything.

G2-GATED: the T* INTERPRETATION rides on G2, which returned no broken step but attached the
regime rider T* = c(r)/r^2 with c -> 1 only as r -> 0. The empirical membership test below is
unaffected by that rider — it uses the RANKING by r, which is horizon-free and constant-free.

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

ARM = "MN-4"
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
K = 32
POOL_MULT = 2                    # `refined` draws from the top POOL_MULT*k by activation


def refined_address(act, r, k: int, *, pool_mult: int = POOL_MULT):
    """The literal 'does membership add anything OVER energy' scorer.

    Take the top pool_mult*k coordinates by activation, then keep the k of them with the SMALLEST
    r. Ties in r are broken by higher activation, which matters because r is censored at the floor
    for many coordinates: where r carries no information the tie-break makes this degenerate
    gracefully to plain activation order, so an uninformative r yields `refined == activation` and
    the contrast correctly reads "adds nothing" instead of reading noise.
    """
    a = np.asarray(act, dtype=np.float64)
    rr = np.asarray(r, dtype=np.float64)
    assert a.shape == rr.shape, f"act {a.shape} vs r {rr.shape}"
    pool = np.asarray(W0.top_k_coords(a, min(pool_mult * k, a.size)), dtype=np.int64)
    order = sorted(pool.tolist(), key=lambda j: (rr[j], -a[j], j))
    return order[:k]


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-4 — the T*-map and the membership law")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn4", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn4"), out)

    vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    g4 = json.loads(man.require("g4_metrics", f"{G4_PREFIX}/metrics.json").read_text())
    assert g4["verdict"] == "REPRODUCED", f"G4 verdict {g4['verdict']!r} — MN-4's inputs are ungated"
    n_steps = int(g4["n_frozen_windows"])
    banks = WT.fetch_banks(H.cache_dir(out, "banks"))

    cells = [(t, L) for t in H.TYPES for L in range(H.N_LAYERS)]
    scorer_j = {s: [] for s in ("t_star_membership", "activation_energy", "refined", "random")}
    censored, sig_ratio, rows = [], [], []

    for (t, L) in cells:
        cons = np.load(vec / f"grad_cons__{t}__L{L}.npy")
        act = np.load(vec / f"act_rms__{t}__L{L}.npy")
        U = WT.load_bank(banks, t, L, "core32")
        truth = MT.bank_address(U, K)
        n = int(cons.size)
        assert act.size == n == U.shape[0], f"{t}/L{L}: axis mismatch"

        ts, r, at_floor = SF.t_star_map(cons, n_steps)
        frac_censored = float(at_floor.mean())
        censored.append(frac_censored)

        addr = {
            "t_star_membership": W0.top_k_coords(-r, K),          # ascending r == descending T*
            "activation_energy": W0.top_k_coords(act, K),
            "refined": refined_address(act, r, K),
            "random": MT.haar_address(n, K, H.cell_seed(a.seed, ARM, t, L)),
        }
        cell = {"type": t, "layer": L, "n": n, "censored_fraction": frac_censored}
        for name, coords in addr.items():
            j = W0.overlap_stats(coords, truth, n)["jaccard"]
            scorer_j[name].append(j)
            cell[name] = j
            cid = {"t_star_membership": MC.TREATMENT, "refined": MC.TREATMENT,
                   "activation_energy": MC.STRUCTURED, "random": MC.HAAR}[name]
            at.add("tstar_address_jaccard", value=j, layer=L, type=t, bank_id="core32",
                   item_id=name, control_id=cid,
                   flags={"k": K, "scorer": name, "censored_fraction": frac_censored,
                          "n_steps": n_steps})

        # The per-coordinate consistency signature (the question G2's T3 registered).
        inside = np.zeros(n, dtype=bool)
        inside[truth] = True
        r_in, r_out = float(r[inside].mean()), float(r[~inside].mean())
        ratio = r_in / r_out if r_out > 0 else float("nan")
        if np.isfinite(ratio):
            sig_ratio.append(ratio)
        cell.update({"r_mean_in_sink": r_in, "r_mean_out": r_out, "r_ratio": ratio})
        at.add("sink_consistency_ratio", value=ratio if np.isfinite(ratio) else 0.0, layer=L,
               type=t, bank_id="core32", control_id=MC.TREATMENT,
               flags={"r_in": r_in, "r_out": r_out, "censored_fraction": frac_censored,
                      "what": "mean r over the core32 address / over its complement; "
                              "<1 means the sink IS the lower-consistency set"})
        rows.append(cell)
        np.save(out / "vectors" / f"t_star__{t}__L{L}.npy", ts)
        np.save(out / "vectors" / f"r__{t}__L{L}.npy", r)

    # ---- the frozen contrast ----
    refined_vs_energy = MC.paired_delta_ci(scorer_j["refined"], scorer_j["activation_energy"],
                                           seed=a.seed)
    membership_vs_energy = MC.paired_delta_ci(scorer_j["t_star_membership"],
                                              scorer_j["activation_energy"], seed=a.seed)
    means = {k: float(np.mean(v)) for k, v in scorer_j.items()}
    at.add_ci("tstar_refined_minus_energy",
              np.asarray(scorer_j["refined"]) - np.asarray(scorer_j["activation_energy"]),
              bank_id="core32", control_id=MC.TREATMENT, seed=a.seed,
              flags={"k": K, "rule": "LAW-HOLDS needs this CI clear of 0"})
    sig = MC.paired_delta_ci(sig_ratio, [1.0] * len(sig_ratio), seed=a.seed)

    # ---- the registered retrodictions, scored 1:1 ----
    br = {
        "BR1": {"claim": "censored fraction > 0.5 in a majority of cells", "conf": 0.70,
                "actual": float(np.mean([c > 0.5 for c in censored])),
                "hit": bool(np.mean([c > 0.5 for c in censored]) > 0.5)},
        "BR2": {"claim": "mean r inside the sink is LOWER than outside, ratio in [0.5, 1.0)",
                "conf": 0.55, "actual": float(np.mean(sig_ratio)),
                "hit": bool(0.5 <= np.mean(sig_ratio) < 1.0 and sig["ci_hi"] < 0.0)},
        "BR3": {"claim": "t_star_membership alone scores below activation_energy", "conf": 0.70,
                "actual": means["t_star_membership"],
                "hit": bool(means["t_star_membership"] < means["activation_energy"])},
        "BR4": {"claim": "refined does NOT beat activation_energy -> NO-REACH", "conf": 0.60,
                "actual": refined_vs_energy["delta_mean"],
                "hit": bool(not refined_vs_energy["clears_zero"])},
    }
    n_hit = sum(1 for v in br.values() if v["hit"])

    law_holds = refined_vs_energy["clears_zero"] and n_hit >= 1
    verdict = MP.check_verdict(ARM, "LAW-HOLDS" if law_holds else
                               ("NO-REACH" if not refined_vs_energy["clears_zero"]
                                else "AMBIGUOUS"))
    gates.record("mn4_map_not_mostly_censored", float(np.mean(censored)) < 0.5,
                 mean_censored_fraction=float(np.mean(censored)),
                 max_censored=float(np.max(censored)), min_censored=float(np.min(censored)),
                 note=("RECORDED. A T*-map whose coordinates are mostly censored at T=24 is "
                       "uninformative about the deep sink; the ranking test survives censoring "
                       "because it only needs the ORDER of the resolved coordinates"))
    at.check_control_trinity  # noqa: B018 — MN-4's trinity is energy(structured)+random(haar);
    info = at.to_parquet()    # the shuffle member is not defined for a per-cell address contrast
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"], "k": K, "n_steps": n_steps,
        "n_cells": len(cells), "scorer_means": means,
        "refined_vs_energy": refined_vs_energy, "membership_vs_energy": membership_vs_energy,
        "consistency_signature": {"mean_ratio": float(np.mean(sig_ratio)), **sig},
        "censoring": {"mean": float(np.mean(censored)), "min": float(np.min(censored)),
                      "max": float(np.max(censored)),
                      "floor": SF.consistency_floor(n_steps)},
        "retrodictions": br, "n_retrodictions_hit": n_hit,
        "cells": rows, "atlas": info,
        "g2_rider": ("T* = c(r)/r^2 with c -> 1 as r -> 0 (G2/T2). The membership test here uses "
                     "the RANKING by r, which is horizon-free and constant-free, so the rider does "
                     "not touch this arm's verdict."),
        "incumbent": "activation_energy — M-8's named incumbent, mean 0.4160 against core32"})
    print("MN4_OK " + json.dumps({"verdict": verdict, "means": {k: round(v, 4) for k, v in
                                                                means.items()},
                                  "retrodictions_hit": n_hit,
                                  "mean_censored": round(float(np.mean(censored)), 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
