"""REGRAIN Arm D (CPU, banks only) — draw agreement, adjudicated under the A1 rule.

The question the label actually names: **do independent re-estimations of a read select the same
ATOMS more than their shared SUBSPACE alone predicts?**

v1 compared a Jaccard (referenced to a permutation null of 0.0) against a canonicity (already
referenced to its chance floor k/d and to the ceiling 1) and read the difference as a
dissociation. On a common scale there was none. The A1 rule fixes the comparison:

  surrogate null : partners V' of a draw V built at the SAME canonicity (regrain_core.
                   surrogate_at_canonicity), scored by the same top-n aligned-atom Jaccard.
  H_basis-ambiguity   iff J_cilo > surrogate_p95        (atoms agree MORE than the span implies)
  H_subspace-explains iff permutation_p95 < J_cilo ≤ surrogate_p95
  H_genuinely-family  iff J_cilo ≤ permutation_p95

Canonicity is measured at the SAME rank the atom readout uses (k=K_READ); the ENDOW-F k_can=16
value is reported for continuity and never enters the rule (it is rank-dependent: the same S_vl
draws give 0.357 at k=128 and 0.639 at k=16).

READS. Adjudication runs on **S_vl** with 5 disjoint-item draws (A5b). **S_action is DESCRIPTIVE
at n=2**: a third draw needs the π0.5 ACTION loss, which the reused VL-loss Gram cannot supply.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_d_draws --type r7i.4xlarge --no-weights \
      --s3-prefix results/regrain_d --budget-min 90 \
      --args "--harvest-s3 s3://…/results/regrain_harvest/out"
"""
import argparse
import json
import pathlib

N_SURROGATE = 200
N_PERM = 4000
N_BOOT = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--dicts-l14-s3", default=None, help="defaults to the GRAIN L14 banks")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import orthonormalize
    from infra.pg.endow_atlas import canonicity

    rep = {"object": "regrain_d_v1", "top_n": RS.TOP8, "k_readout": RS.K_READ,
           "n_surrogate": N_SURROGATE, "n_perm": N_PERM}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banks = RS.load_read_banks("/tmp/regrain")

    def adjudicate(Vs, Wd, tag, n_boot=N_BOOT):
        """The frozen A1 pipeline for one read's draw family."""
        U = RC.unit_decoder(Wd)
        sets = [RC.top_aligned_atoms(U, V, RS.TOP8, already_unit=True) for V in Vs]
        jac_mean, jac_pairs = GB.pairwise_jaccard(sets)
        c_align = canonicity(Vs, IP.D)                       # AT THE READOUT'S RANK
        c_16 = canonicity([V[:, : RS.K_CAN] for V in Vs], IP.D)
        perm = GB.jaccard_null(Wd.shape[1], RS.TOP8, N_PERM, seed=3)
        perm_p95 = float(torch.quantile(perm, 0.95))
        surro, cs = RC.canonicity_matched_jaccard_null(Wd, Vs[0], c_align, IP.D, RS.TOP8,
                                                       N_SURROGATE, seed=11)
        surro_p95 = float(torch.quantile(surro, 0.95))
        entry = {"n_draws": len(Vs), "sets": sets, "jaccard_mean": jac_mean,
                 "jaccard_pairs": jac_pairs, "canonicity_at_readout_rank": c_align,
                 "canonicity_kcan16_reported_only": c_16,
                 "permutation_p95": perm_p95, "permutation_mean": float(perm.mean()),
                 "surrogate_p95": surro_p95, "surrogate_mean": float(surro.mean()),
                 "surrogate_realized_c_first4": [float(x) for x in cs[:4]],
                 "shared_atoms": sorted(set.intersection(*[set(s) for s in sets]))}
        if len(Vs) >= 3:
            # pair bootstrap on the mean pairwise Jaccard → the CI-lower the rule consumes
            rng = np.random.default_rng(1337)
            pv = np.asarray(jac_pairs, float)
            boots = np.asarray([rng.choice(pv, size=pv.size, replace=True).mean()
                                for _ in range(n_boot)])
            cilo = float(np.percentile(boots, 2.5))
            entry.update({"jaccard_cilo": cilo, "cilo_rule": "pair-bootstrap p2.5",
                          "adjudicated": True,
                          "verdict": RC.adjudicate_draws(jaccard_cilo=cilo,
                                                         permutation_p95=perm_p95,
                                                         surrogate_p95=surro_p95)})
        else:
            entry.update({"jaccard_cilo": float(min(jac_pairs)), "cilo_rule": "min-pair (n=2)",
                          "adjudicated": False, "verdict": None,
                          "note": "n=2 draws → DESCRIPTIVE only (§12 pivot; a 3rd S_action draw "
                                  "needs the pi0.5 ACTION loss, A5b). The surrogate band is "
                                  "reported so the comparison is still on a common scale."})
        rep[tag] = entry
        _w()
        return entry

    # ---------------- S_vl (ADJUDICATED, 5 disjoint draws) ----------------
    draws = torch.load(RS.fetch(f"{a.harvest_s3}/svl_draws5.pt", "/tmp/regrain/svl_draws5.pt"),
                       map_location="cpu", weights_only=False)
    Vs = [orthonormalize(torch.as_tensor(draws[k]["V"], dtype=torch.float64))
          for k in sorted(draws)]
    assert len(Vs) >= 3, f"S_vl draws {len(Vs)} < 3 — cannot adjudicate"
    assert Vs[0].shape[1] == RS.K_READ
    src14 = a.dicts_l14_s3 or RS.GRAIN_DICTS_L14
    n14 = RS.dict_bank_name("secondary", 1, 0)
    ae14, _c14, _q14, _s14 = GB.load_dict_bank(RS.fetch(f"{src14}/{n14}", f"/tmp/regrain/{n14}"))
    rep["svl_dictionary"] = {"bank": n14, "src": src14, "m": 1,
                             "note": "atom ALIGNMENT is a reconstruction-basis use, not a splice "
                                     "use — the L14 splice pivot (GRAIN R4) does not bind it. "
                                     "Held-out FVU at L14 for this dict is 0.070 (GRAIN R1)."}
    svl = adjudicate(Vs, RC.decoder_matrix(ae14), "svl")

    # ---------------- S_action (DESCRIPTIVE, n=2 banked draws) ----------------
    import subprocess

    from infra.aws_pg.jobs._grain_shared import FULL_BANK
    from infra.aws_pg.jobs._inherit_shared import load_bank, load_redraw_bases
    from infra.pg.inherit_core import slice_k
    Lp = banks["saction"]["L"]
    bases, _h, _t = load_bank("/tmp/regrain")
    draws_sa = [slice_k(bases[Lp], RS.K_READ)]                       # draw 1: ctrliface_discovery
    if subprocess.run(["aws", "s3", "ls", FULL_BANK, "--region", "us-east-1"],
                      capture_output=True).returncode == 0:
        draws_sa.append(load_redraw_bases(FULL_BANK, "/tmp/regrain")[Lp])   # draw 2: pi05 redraw
    n0 = RS.dict_bank_name("primary", 1, 0)
    ae0, _c0, _q0, _s0 = GB.load_dict_bank(RS.fetch(f"{RS.GRAIN_DICTS_L0}/{n0}",
                                                    f"/tmp/regrain/{n0}"))
    if len(draws_sa) >= 2:
        adjudicate([orthonormalize(torch.as_tensor(V, dtype=torch.float64)) for V in draws_sa],
                   RC.decoder_matrix(ae0), "saction")
    else:
        rep["saction"] = {"n_draws": len(draws_sa), "adjudicated": False,
                          "note": "fewer than 2 banked S_action draws — UNRUN"}

    rep["headline"] = {"adjudicated_read": "svl", "verdict": svl["verdict"],
                       "jaccard_cilo": svl["jaccard_cilo"],
                       "permutation_p95": svl["permutation_p95"],
                       "surrogate_p95": svl["surrogate_p95"]}
    print("REGRAIN_D_OK", json.dumps(rep["headline"], default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
