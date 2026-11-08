"""MENDEL — the PG1 fine-tune family, streamed ONCE for three arms.

The 24 official `paligemma-3b-ft-*-224` checkpoints are ~5.6 GB each. Downloading them three times
to answer three questions would be ~400 GB of avoidable transfer, so this job streams them once and
banks everything the family is asked for:

  MN-1 (family hop)  per-fine-tune column write energy + the per-fine-tune Gram, so the
                     leave-one-out parent bank (freeze A2) can be built without a second pass.
  MN-2 (faculty)     per-fine-tune write fraction into the SINK bank, against an energy-matched
                     control and a haar band.
  MN-5 (tier 1)      the per-fine-tune continuous features MN-5's rank readout wants, now that its
                     accuracy readout is known to have no headroom.

MN-2's QUESTION, in the sharpest form the banked data supports. The spec asks whether faculty
change is co-located with the sink. PLENUM gives, per fine-tune, which faculties survived it; this
job gives, per fine-tune, how much of its write landed in the sink. So:

    (a) THE FROZEN RULE — is the sink write fraction outside the energy-matched band? Reported per
        cell and aggregated by type and layer, exactly as §2's MN-2 rule reads.
    (b) THE FUNCTIONAL COUPLING — does the SINK-DIRECTED share of the write predict faculty
        survival BETTER than the total write does? MN-5 already measured the incumbent:
        Spearman(||dW||, survival rate) = -0.473 (p = 0.084, n = 24). A sink statistic that does
        not beat that is not a co-location finding, it is the same finding in new coordinates.

(b) is why the incumbent is named in the metrics: without it, any negative correlation here would
read as "the sink predicts loss" when it may only be re-measuring total motion.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

from infra.aws_pg.jobs.mendel_mn1_transport import _delta, _ft_decoder, _lineage_decoder
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_critic as CR
from infra.pg import mendel_program as MP
from infra.pg import mendel_transport as MT
from infra.pg.endow_writefrac import write_fraction

ARM = "MN-2"
TYPES = H.TYPES_RESID
K = H.BANK_K
N_HAAR = 8
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
PLENUM = f"{MC.RESULTS_S3}/plenum_agg_final2/out"
LABELLED_FACULTIES = ("caption", "ocr", "reading", "vqa")
# MN-5's incumbent, recomputed there from artifacts, named here so (b) has a bar to clear.
MN5_INCUMBENT_RHO = -0.4734


def energy_matched(act, k: int, e_target: float, seed: int):
    """A k-subspace matched on removed activation energy, in the coordinate basis.

    Via `mendel_core.energy_matched_control`, which wraps the reused
    `conserved_core_probes.energy_matched_subspace` and returns HOW WELL it matched. The wrapper
    exists because the underlying routine takes a `tol` that reads like a guarantee and is not: it
    returns its closest attempt silently, and on this substrate it missed by up to 350x, which
    biases toward finding sink-avoidance. Match quality is carried into the metrics and gated.

    The Gram handed to it is the DIAGONAL per-coordinate activation energy from G4 — stated, not
    glossed. G4b banks the real eigenbasis; when that artifact is present this control should be
    rebuilt on it, and the metrics record which basis was used so the two runs are never confused.
    """
    return MC.energy_matched_control(act, k, e_target, seed, tol=0.03)


def plenum_survival(metrics: dict) -> dict:
    """{ft-unit: {faculty: retained_bool}} over the labelled faculties."""
    out = collections.defaultdict(dict)
    for c in metrics["cells"]:
        u = str(c.get("unit", ""))
        if not u.startswith("ft-") or c.get("faculty") not in LABELLED_FACULTIES:
            continue
        if c.get("label") is None:
            continue
        out[u][c["faculty"]] = (c["label"] == "retained")
    return dict(out)


def main():
    ap = argparse.ArgumentParser(description="MENDEL — the PG1 fine-tune family, one pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ft", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-grams", action="store_true",
                    help="bank the per-fine-tune Gram for MN-1's leave-one-out parent bank")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_ftfam", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_ftfam"), out)

    vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    plenum = json.loads(man.require("plenum_metrics", f"{PLENUM}/metrics.json").read_text())
    banks = WT.fetch_banks(H.cache_dir(out, "banks"))
    survival = plenum_survival(plenum)
    assert survival, "no labelled ft-* cells in the PLENUM matrix"

    # ---- the §5 positive control, before any real number ----
    dW_p, U_p = MT.planted_colocation(n=2048, k=K, out_dim=64, seed=a.seed, strength=0.81)
    void = MT.instrument_is_void(float(write_fraction(dW_p, U_p)), 0.9)
    gates.record("ftfam_instrument_detects_a_planted_colocation", not void["void"], **void)

    from huggingface_hub import HfApi
    repos = WT.ft_cloud(HfApi())[: a.n_ft]
    pg1_sd, pg1_sha = _lineage_decoder("PG1")
    cells = [(t, L) for t in TYPES for L in range(H.N_LAYERS)]

    # Cache the per-cell controls once: they depend on the CELL, not on the fine-tune.
    ctrl, matches = {}, []
    for (t, L) in cells:
        U = WT.load_bank(banks, t, L, "core32")
        act = np.load(vec / f"act_rms__{t}__L{L}.npy")
        e2 = act ** 2
        e_sink = float((e2[:, None] * U ** 2).sum() / e2.sum())
        seed = H.cell_seed(a.seed, ARM, t, L)
        Ue, match = energy_matched(act, K, e_sink, seed)
        matches.append(match)
        rng = np.random.default_rng(seed + 7)
        haar = [np.linalg.qr(rng.standard_normal((H.IN_DIM[t], K)))[0] for _ in range(N_HAAR)]
        ctrl[(t, L)] = {"U": U, "Ue": Ue, "haar": haar, "e_sink": e_sink,
                        "e_matched": match["e_achieved"], "match": match}

    # INCREMENTAL SAVE (house rule): the results containers are bound into a streamed report
    # BEFORE the fine-tune loop and rewritten after every fine-tune. This job is
    # download-dominated (~11 min per checkpoint), so a reaper deadline or a crash at fine-tune 20
    # would otherwise discard twenty completed fine-tunes' worth of contrast rows. `partial.json`
    # is a complete, readable result at every point in the loop.
    per_ft, rows = {}, []
    partial = {"arm": ARM, "status": "RUNNING", "per_fine_tune": per_ft, "cells": rows,
               "n_target": len(repos)}

    def _flush():
        partial["n_done"] = len(per_ft)
        (out / "partial.json").write_text(json.dumps(partial, indent=2, sort_keys=True))

    _flush()
    gram_dir = H.cache_dir(out, "ft_grams")
    for i, repo in enumerate(repos):
        sd, sha = _ft_decoder(repo)
        unit = "ft-" + repo.split("paligemma-3b-ft-")[1].rsplit("-224", 1)[0]
        tot_fro2, sink_w, energy_w, haar_w = 0.0, [], [], []
        for (t, L) in cells:
            dW = _delta(sd, pg1_sd, t, L)
            c = ctrl[(t, L)]
            cell = MT.colocation_cell(dW, U_sink=c["U"], U_energy=c["Ue"], U_haar_list=c["haar"])
            band = MC.exceeds_band(cell["sink"], min(cell["energy"], cell["haar_lo"]),
                                   max(cell["energy"], cell["haar_hi"]))
            tot_fro2 += float((dW ** 2).sum())
            sink_w.append(cell["sink"])
            energy_w.append(cell["energy"])
            haar_w.append(cell["haar_mean"])
            rows.append({"unit": unit, "type": t, "layer": L, "sink": cell["sink"],
                         "energy": cell["energy"], "haar_mean": cell["haar_mean"],
                         "above_band": band["above_band"], "below_band": band["below_band"],
                         "energy_matched_ok": c["match"]["matched"],
                         "energy_match_rel_error": c["match"]["rel_error"]})
            np.save(out / "vectors" / f"colenergy__{unit}__{t}__L{L}.npy",
                    MT.column_write_energy(dW).astype(np.float32))
            if a.save_grams:
                g = dW.T @ dW / float((dW ** 2).sum())
                np.save(gram_dir / f"{i}__{t}__L{L}.npy", g.astype(np.float32))
            for name, val, cid in (("sink", cell["sink"], MC.TREATMENT),
                                   ("energy_matched", cell["energy"], MC.STRUCTURED),
                                   ("haar", cell["haar_mean"], MC.HAAR)):
                at.add("faculty_write_fraction", value=val, layer=L, type=t, bank_id=name,
                       item_id=unit, control_id=cid, model_id=repo, model_sha=sha,
                       flags={"k": K, "e_sink": c["e_sink"], "e_matched": c["e_matched"],
                              "energy_match_rel_error": c["match"]["rel_error"],
                              "energy_matched_ok": c["match"]["matched"]})
        per_ft[unit] = {"repo": repo, "dW_norm": float(np.sqrt(tot_fro2)),
                        "sink_mean": float(np.mean(sink_w)),
                        "energy_mean": float(np.mean(energy_w)),
                        "haar_mean": float(np.mean(haar_w)),
                        "sink_minus_energy": float(np.mean(sink_w) - np.mean(energy_w))}
        del sd
        _flush()
        print(f"[ftfam] {i + 1}/{len(repos)} {unit}: sink {per_ft[unit]['sink_mean']:.4f} "
              f"energy {per_ft[unit]['energy_mean']:.4f}", flush=True)

    partial["status"] = "COMPLETE"
    _flush()

    # ---- (a) the frozen rule ----
    below = collections.Counter((r["type"], r["layer"]) for r in rows if r["below_band"])
    above = collections.Counter((r["type"], r["layer"]) for r in rows if r["above_band"])
    # The frozen rule is scored on ALL cells and is NOT swapped (§6/e). Beside it, the robustness
    # pair a silently-unmatched control demands: the same contrast on cells where the match held,
    # and the contrast against the haar band, which has no energy-matching dependency at all.
    match_report = MC.energy_match_report(matches)
    ok = [r for r in rows if r["energy_matched_ok"]]
    paired_matched = (MC.paired_delta_ci([r["sink"] for r in ok], [r["energy"] for r in ok],
                                         seed=a.seed) if len(ok) >= 8 else None)
    paired_haar = MC.paired_delta_ci([r["sink"] for r in rows], [r["haar_mean"] for r in rows],
                                     seed=a.seed)
    paired = MC.paired_delta_ci([r["sink"] for r in rows], [r["energy"] for r in rows],
                                seed=a.seed)
    at.add_ci("faculty_write_fraction", [r["sink"] - r["energy"] for r in rows], bank_id="sink",
              control_id=MC.SHUFFLE, seed=a.seed,
              flags={"what": "sink minus energy-matched, pooled over fine-tunes and cells"})

    # ---- (b) the functional coupling, against MN-5's incumbent ----
    units = sorted(u for u in per_ft if u in survival)
    surv = [float(np.mean(list(survival[u].values()))) for u in units]
    rho_sink = CR.spearman([per_ft[u]["sink_mean"] for u in units], surv)
    rho_delta = CR.spearman([per_ft[u]["sink_minus_energy"] for u in units], surv)
    rho_norm = CR.spearman([per_ft[u]["dW_norm"] for u in units], surv)
    beats_incumbent = abs(rho_sink) > abs(rho_norm)
    for name, rho in (("sink_fraction", rho_sink), ("sink_minus_energy", rho_delta),
                      ("dW_norm_incumbent", rho_norm)):
        at.add("faculty_survival_spearman", value=rho, n=len(units), item_id=name,
               control_id=MC.STRUCTURED if name.endswith("incumbent") else MC.TREATMENT,
               flags={"n_units": len(units), "mn5_banked_incumbent": MN5_INCUMBENT_RHO})

    n_below_types = len({t for t, _ in below})
    n_below_layers = len({L for _, L in below})
    n_above_types = len({t for t, _ in above})
    n_above_layers = len({L for _, L in above})

    # THE FROZEN RULE, AS WRITTEN, AND IT IS ONE-SIDED. The spec's MN-2 row reads: "SINK-COUPLED
    # iff sink fraction EXCEEDS the energy-matched band on >=2 faculties at >=2 hops; ENERGY-ONLY
    # iff never; else AMBIGUOUS." An earlier version of this file scored it TWO-sided — it called
    # SINK-COUPLED on the sink sitting BELOW the band — which is a rule the spec does not license
    # and no freeze amendment (A1-A12) adds. That is corrected here rather than argued with.
    #
    # It matters, because the data runs the other way: on the first completed run every one of the
    # 1080 cells sat BELOW the band and none above, so the rule as written returns ENERGY-ONLY.
    #
    # And "ENERGY-ONLY" MISDESCRIBES that data — the sink is written about a third as much as an
    # equally-energetic subspace, which is a large sink-specific effect, the opposite of "energy
    # explains the placement". The pre-registration failed to anticipate the SIGN, and a token that
    # misdescribes the observation is a defect in the pre-registration, not a licence to relabel
    # after the fact. The verdict below is the rule's; `observed_direction` beside it is the
    # description, and both are banked (§6/b).
    coupled = n_above_types >= 2 and n_above_layers >= 2 and paired["ci_lo"] > 0 and beats_incumbent
    verdict = MP.check_verdict(ARM, "SINK-COUPLED" if coupled else
                               ("ENERGY-ONLY" if not above else "AMBIGUOUS"))
    observed_direction = {
        "n_cells_above_band": len(above), "n_cells_below_band": len(below),
        "above_types": n_above_types, "above_layers": n_above_layers,
        "below_types": n_below_types, "below_layers": n_below_layers,
        "sink_mean": paired["treatment_mean"], "energy_matched_mean": paired["control_mean"],
        "reading": ("the sink is written LESS than an equally-energetic subspace" if len(below)
                    > len(above) else "the sink is written MORE than an equally-energetic subspace"),
        "note": ("the frozen rule is one-sided and anticipated the opposite sign; this block is "
                 "the description of what was measured, not a second verdict")}
    gates.record("mn2_energy_controls_actually_matched",
                 match_report["fraction_matched"] >= 0.9, **match_report)
    gates.record("mn2_survives_on_matched_cells_only",
                 bool(paired_matched is not None and
                      (paired_matched["ci_hi"] < 0 or paired_matched["ci_lo"] > 0)),
                 n_cells=len(ok), **(paired_matched or {}))
    gates.record("mn2_survives_against_the_haar_band",
                 bool(paired_haar["ci_hi"] < 0 or paired_haar["ci_lo"] > 0), **paired_haar)
    gates.record("mn2_sink_outside_energy_band", bool(paired["ci_hi"] < 0 or paired["ci_lo"] > 0),
                 **paired)
    gates.record("mn2_beats_the_total_write_incumbent", beats_incumbent,
                 rho_sink=rho_sink, rho_dW_norm=rho_norm, mn5_banked=MN5_INCUMBENT_RHO,
                 note=("a sink statistic that does not beat total ||dW|| is the same finding in "
                       "new coordinates, not a co-location finding"))
    at.check_control_trinity(["faculty_write_fraction"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "observed_direction": observed_direction,
        "frozen_rule_is_one_sided": (
            "SINK-COUPLED requires the sink to EXCEED the energy-matched band. A sink BELOW the "
            "band returns ENERGY-ONLY under the rule as written, however large the gap"),
        "queue_item": meta["item"],
        "n_fine_tunes": len(per_ft), "n_cells": len(rows), "types": list(TYPES),
        "energy_match_report": match_report,
        "sink_vs_energy_matched_cells_only": paired_matched,
        "sink_vs_haar_band": paired_haar,
        "energy_matched_basis": "coordinate (diagonal activation energy from G4); G4b banks the "
                                "true eigenbasis and this control should be rebuilt on it",
        "frozen_rule": {"paired_sink_minus_energy": paired,
                        "n_cells_below_band": sum(below.values()),
                        "n_cells_above_band": sum(above.values()),
                        "below_types": n_below_types, "below_layers": n_below_layers},
        "functional_coupling": {"rho_sink_fraction": rho_sink,
                                "rho_sink_minus_energy": rho_delta,
                                "rho_dW_norm_incumbent": rho_norm,
                                "beats_incumbent": bool(beats_incumbent),
                                "mn5_banked_incumbent": MN5_INCUMBENT_RHO,
                                "n_units": len(units)},
        "per_fine_tune": per_ft, "cells": rows, "atlas": info,
        "banked_for_other_arms": {"MN-1": "vectors/colenergy__* (per-fine-tune column write energy)"
                                         + (" and ft_grams (leave-one-out)" if a.save_grams else ""),
                                  "MN-5": "per_fine_tune.dW_norm and sink_mean as tier-1 features"}})
    print("FTFAM_OK " + json.dumps({"verdict": verdict, "n_ft": len(per_ft),
                                    "sink_minus_energy": round(paired["delta_mean"], 5),
                                    "rho_sink": round(rho_sink, 4),
                                    "rho_norm": round(rho_norm, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
