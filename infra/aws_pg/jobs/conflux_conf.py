"""CONFLUX confirmation job (CPU, r7i) — unblind once.

Loads the frozen r* + 12/12 split from conflux_disc, re-estimates U_bb on the SEALED confirmation
half, computes the single headline o(U_bb^conf, U_sus) at r* and the frozen verdict, then the
corroboration arms D/C/N/B + the sink-quotient control + seed-stability. Writes
results/conflux_conf/out/metrics.json.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.conflux_conf \
      --type r7i.4xlarge,r7i.2xlarge --no-weights --s3-prefix results/conflux_conf --budget-min 300
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc-prefix", default="results/conflux_disc")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-layers", type=int, default=18)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wd = pathlib.Path("/tmp/conflux"); wd.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _conflux_shared as CS
    from infra.pg import conflux_core as CX
    import numpy as np

    rep = {"object": "conflux_conf_v1"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- load the frozen discovery decision (r*, split) ----
    disc = json.loads(CS.fetch(f"{CS.S3_BUCKET}/{a.disc_prefix}/out/metrics.json",
                               wd / "conflux_disc.json").read_text())
    assert disc.get("green"), "conflux_disc not green"
    rstar = int(disc["rstar"])
    disc_slugs = list(disc["split"]["discovery"])
    conf_slugs = list(disc["split"]["confirmation"])
    rep["rstar"] = rstar
    rep["split"] = disc["split"]
    _w()

    # ---- substrate ----
    qb = CS.load_q_bank(str(wd))
    evecs = np.asarray(qb["evecs"], dtype=np.float64)
    k_stiff = int(qb["k_stiff"])
    d = evecs.shape[0]
    slug2i, i2slug, anat = CS.roster_index(wd)
    partners = CS.load_partners(wd)
    carrier = CS.load_carrier(wd)
    conf_ridx = [slug2i[s] for s in conf_slugs]
    disc_ridx = [slug2i[s] for s in disc_slugs]
    dW_conf = CS.load_dW_list(wd, conf_ridx, L=0)
    dW_disc = CS.load_dW_list(wd, disc_ridx, L=0)
    print(f"[conflux-conf] r*={rstar}; loaded {len(dW_conf)} conf + {len(dW_disc)} disc ΔW", flush=True)
    _w()

    # ---- HEADLINE: single sealed test at r* (+ seed-stability) ----
    head = CX.rung_overlap(dW_conf, evecs, k_stiff, rstar, partners, n_null=a.n_null,
                           n_boot=a.n_boot, seed=CX.SEED, boot=True)
    rep["headline"] = head
    rep["verdict"] = CX.adjudicate_conflux(head)
    head2 = CX.rung_overlap(dW_conf, evecs, k_stiff, rstar, partners, n_null=a.n_null,
                            n_boot=a.n_boot, seed=2024, boot=True)
    rep["headline_seed2"] = {"o_real": head2["o_real"], "verdict": CX.adjudicate_conflux(head2)["verdict"],
                             "excess_within_ci": head2.get("excess_within_ci")}
    rep["seed_stable"] = bool(rep["verdict"]["verdict"] == rep["headline_seed2"]["verdict"])
    print(f"[conflux-conf] HEADLINE r*={rstar} o_real={head['o_real']:.4f} verdict={rep['verdict']['verdict']}"
          f" seed_stable={rep['seed_stable']}", flush=True)
    _w()

    # ---- positive control: cross-half U_bb stability o(U_bb^disc, U_bb^conf) at r* ----
    Ubb_conf, _ = CX.build_ubb(dW_conf, rstar)
    Ubb_disc, _ = CX.build_ubb(dW_disc, rstar)
    rep["cross_half_ubb_stability"] = CX.overlap(Ubb_disc, Ubb_conf)
    print(f"[conflux-conf] cross-half U_bb stability @r*={rstar}: {rep['cross_half_ubb_stability']:.4f}",
          flush=True); _w()

    # ---- full confirmation ladder (point-only; descriptive/robustness for the sweep figure) ----
    ks = CX.ladder_ks(k_stiff)
    rep["conf_ladder"] = {"ks": list(ks),
                          "rungs": {int(r): (head if int(r) == rstar else
                                    CX.rung_overlap(dW_conf, evecs, k_stiff, r, partners,
                                                    n_null=a.n_null, boot=False))
                                    for r in ks}}
    for r in ks:
        rg = rep["conf_ladder"]["rungs"][int(r)]
        print(f"[conflux-conf] conf r={r:4d} o_real={rg['o_real']:.4f} within_null={rg['within_stiff_null']['o_ubb_median']:.4f} void={rg['void']}",
              flush=True); _w()

    # ---- Arm D: per-layer backbone vs the (L0) sensitive subspace ----
    dW_by_layer = {L: CS.load_dW_list(wd, conf_ridx, L=L) for L in range(int(a.n_layers))}
    rep["arm_d_per_layer"] = CX.per_layer_overlap(dW_by_layer, evecs, k_stiff, rstar, n_null=a.n_null)
    print(f"[conflux-conf] Arm D per-layer done ({a.n_layers} layers)", flush=True); _w()

    # ---- Arm C: three-way backbone ∩ sensitive ∩ carrier ----
    U_sus_r = CX.usus_basis(evecs, rstar)
    rep["arm_c_three_way"] = CX.three_way(Ubb_conf, U_sus_r, carrier)
    _w()

    # ---- Arm N: residual U_bb ⊖ U_sus characterisation ----
    rep["arm_n_residual"] = CX.residual_characterisation(Ubb_conf, U_sus_r)
    _w()

    # ---- Arm B: off-U_bb vs off-U_sus energy fraction (held-out discovery FTs) vs banked gains ----
    gains = CS.load_gains(wd)
    dW_by_ft = {s: dW_disc[i] for i, s in enumerate(disc_slugs)}     # held-out from U_bb^conf
    rep["arm_b_predictive"] = CX.predictive_arm(dW_by_ft, evecs, k_stiff, rstar, Ubb_conf,
                                                {s: gains.get(s) for s in disc_slugs})
    _w()

    # ---- sink-quotient control (T4) at r* ----
    rep["sink_quotient"] = CX.sink_quotient_overlap(dW_conf, evecs, k_stiff, rstar, n_null=a.n_null)
    print(f"[conflux-conf] sink-quotient excess_within={rep['sink_quotient']['excess_within_quotient']:+.4f}",
          flush=True)

    rep["green"] = True
    _w()
    print("CONFLUX_CONF_OK", json.dumps({"verdict": rep["verdict"]["verdict"], "rstar": rstar,
                                         "o_real": head["o_real"]}))


if __name__ == "__main__":
    main()
