"""CONFLUX discovery job (CPU, r7i) — the blind half.

Strict order (spec §9; journal M0.4): dry-run logic gate → preflight (haar_p95 bit-exact +
core_uwr fp16 cross-check) → positive/null controls → discovery-half overlap ladder → freeze c_sat
and pick r*. Writes results/conflux_disc/out/metrics.json. Touches ONLY the 12 discovery FTs; the
confirmation half is sealed for conflux_conf.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.conflux_disc \
      --type r7i.4xlarge,r7i.2xlarge --no-weights --s3-prefix results/conflux_disc --budget-min 240
"""
import argparse
import json
import os
import pathlib


def _preflight(wd, evecs, k_stiff, slug2i, partners, rep):
    """Prove the overlap wiring on banked numbers before any new contrast.
    HARD gate: reproduce handle_t haar_p95 bit-exact (Q-evecs + fixed seeds only) to ≤1e-6.
    Cross-check: reproduce core_uwr_overlap_med from the fp16 scratch ΔW to fp16 tolerance."""
    import numpy as np
    import torch
    from infra.pg import comwrite_core as CoreC
    from infra.pg import comwrite_surgery as SURG
    from infra.aws_pg.jobs import _conflux_shared as CS
    from infra.aws_pg.jobs._handle_shared import load_carry_fts

    ht = json.loads(CS.fetch(f"{CS.HANDLE_T}/metrics.json", wd / "handle_t.json").read_text())
    census = ht["census"]
    kstar = int(ht["kstar"])
    r_cen = int(ht["r_census"])
    assert int(ht["k_stiff"]) == int(k_stiff), f"k_stiff drift {ht['k_stiff']} vs {k_stiff}"

    # ---- HARD gate: haar_p95 bit-exact (handle_t.py:218-225 recipe, verbatim) ----
    Ust = torch.as_tensor(evecs, dtype=torch.float64)[:, :int(k_stiff)]
    haar_ov = []
    for n in range(200):
        Rs = [Ust @ torch.linalg.qr(torch.randn(int(k_stiff), r_cen,
              generator=torch.Generator().manual_seed(1000 + n * 2 + t), dtype=torch.float64))[0]
              for t in range(2)]
        haar_ov.append(CoreC.subspace_overlap(Rs[0].numpy(), Rs[1].numpy()))
    haar_p95 = float(np.quantile(haar_ov, 0.95))
    d_haar = abs(haar_p95 - float(census["haar_p95"]))
    assert d_haar <= 1e-6, f"PREFLIGHT haar_p95 mismatch {haar_p95} vs {census['haar_p95']} (Δ={d_haar:.2e})"

    # ---- cross-check: core_uwr_overlap_med from fp16 scratch ΔW (census recipe) ----
    fts = load_carry_fts(str(wd))
    assert ht.get("setprop_module_def") in (False, None), "HANDLE Arm-T was set-property; census recipe differs"
    Ucore = torch.as_tensor(evecs, dtype=torch.float64)[:, :kstar]
    U_wr = torch.as_tensor(partners["U_wr"], dtype=torch.float64)[:, :r_cen]
    core_dirs = []
    for slug in fts["discovery"]:
        dW = torch.as_tensor(CS.load_dW_oproj(wd, slug2i[slug], L=0))
        core_dW = SURG.project_delta(dW, Ucore, "o_proj").double()
        Usvd, _S, _V = torch.linalg.svd(core_dW, full_matrices=False)
        core_dirs.append(Usvd[:, :r_cen].contiguous())
    uwr_ov = float(np.median([CoreC.subspace_overlap(cd.numpy(), U_wr.numpy()) for cd in core_dirs]))
    d_uwr = abs(uwr_ov - float(census["core_uwr_overlap_med"]))
    assert d_uwr <= 2e-2, f"PREFLIGHT core_uwr drift {uwr_ov} vs {census['core_uwr_overlap_med']} (Δ={d_uwr:.2e})"

    rep["preflight"] = {"haar_p95_repro": haar_p95, "haar_p95_banked": float(census["haar_p95"]),
                        "haar_p95_delta": d_haar, "core_uwr_repro": uwr_ov,
                        "core_uwr_banked": float(census["core_uwr_overlap_med"]),
                        "core_uwr_delta_fp16": d_uwr, "kstar": kstar, "r_census": r_cen, "green": True}
    print(f"[conflux-disc] PREFLIGHT green — haar_p95 Δ={d_haar:.2e} (≤1e-6), core_uwr Δ={d_uwr:.2e} (fp16)",
          flush=True)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wd = pathlib.Path("/tmp/conflux"); wd.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _conflux_shared as CS
    from infra.pg import conflux_core as CX
    from infra.pg.test_conflux_core import dry_run_gate
    import numpy as np

    rep = {"object": "conflux_disc_v1"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- stage 0: weight-free dry-run logic gate (spec §9.2) ----
    rep["dry_run"] = dry_run_gate()
    print(f"[conflux-disc] DRY-RUN green — {rep['dry_run']['n_checks']} checks", flush=True); _w()

    # ---- load banked substrate ----
    qb = CS.load_q_bank(str(wd))
    evecs = np.asarray(qb["evecs"], dtype=np.float64)
    k_stiff = int(qb["k_stiff"])
    d = evecs.shape[0]
    slug2i, i2slug, anat = CS.roster_index(wd)
    partners = CS.load_partners(wd)
    rep["substrate"] = {"d": d, "k_stiff": k_stiff, "metric_key": qb["metric_key"],
                        "n_roster": len(slug2i)}
    _w()

    # ---- stage 1: preflight ----
    _preflight(wd, evecs, k_stiff, slug2i, partners, rep); _w()

    # ---- frozen 12/12 split; load the 12 DISCOVERY ΔW only ----
    disc_slugs, conf_slugs = CS.conflux_split(sorted(slug2i.keys()))
    rep["split"] = {"discovery": disc_slugs, "confirmation": conf_slugs}
    disc_ridx = [slug2i[s] for s in disc_slugs]
    dW_disc = CS.load_dW_list(wd, disc_ridx, L=0)
    print(f"[conflux-disc] loaded {len(dW_disc)} discovery ΔW", flush=True); _w()

    # ---- stage 2: controls ----
    # wiring: o(U_sus, U_sus) = 1 exactly; within-disc split-half U_bb stability (a real shared object?).
    ks = CX.ladder_ks(k_stiff)
    half_a = [dW_disc[i] for i in range(0, len(dW_disc), 2)]
    half_b = [dW_disc[i] for i in range(1, len(dW_disc), 2)]
    stability = {}
    for r in (20, 50):
        Ua, _ = CX.build_ubb(half_a, r)
        Ub, _ = CX.build_ubb(half_b, r)
        stability[int(r)] = CX.overlap(Ua, Ub)
    U_sus20 = CX.usus_basis(evecs, 20)
    rep["controls"] = {"wiring_sus_self": CX.overlap(U_sus20, U_sus20),
                       "ubb_splithalf_stability": stability}
    # banked Q eigvecs are fp32 → self-overlap = 1 ± ~1e-4 (fp32 orthonormality); a real wiring
    # break would be O(0.1). Tolerance separates the two.
    assert abs(rep["controls"]["wiring_sus_self"] - 1.0) < 1e-3, \
        f"wiring control failed: o(U_sus,U_sus)={rep['controls']['wiring_sus_self']}"
    print(f"[conflux-disc] controls: split-half U_bb stability {stability}", flush=True); _w()

    # ---- stage 3: discovery overlap ladder ----
    lad = CX.overlap_ladder(dW_disc, evecs, k_stiff, partners, ks=ks, n_null=a.n_null,
                            n_boot=a.n_boot, boot=True)
    rep["ladder"] = lad
    for r in lad["ks"]:
        rg = lad["rungs"][int(r)]
        ci = rg.get("excess_within_ci", {}).get("ci95")
        print(f"[conflux-disc] r={r:4d}  o_real={rg['o_real']:.4f}  within_null={rg['within_stiff_null']['o_ubb_median']:.4f}"
              f"  excess_within_lo={(ci[0] if ci else float('nan')):+.4f}  void={rg['void']}", flush=True)
        _w()

    # ---- stage 4: r* selection ----
    rstar = CX.select_rstar(lad)
    rep["rstar"] = rstar
    rep["rstar_rung"] = lad["rungs"][int(rstar)]
    print(f"[conflux-disc] r* = {rstar}", flush=True)

    rep["green"] = True
    _w()
    print("CONFLUX_DISC_OK", json.dumps({"rstar": rstar}))


if __name__ == "__main__":
    main()
