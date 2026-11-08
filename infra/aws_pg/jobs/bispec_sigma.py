"""BISPEC Gate-0a (CPU, weight-free) — the site covariance Σ, its eigenbasis, and the FROZEN bands.

This job produces the y-axis of every surface figure in the program. It is therefore the single
most load-bearing artifact BISPEC writes, and the spec did not define it (amendment A2).

**A2, frozen here.** Σ is the CENTERED covariance of the site's residual rows with BOS EXCLUDED
from the estimate:

  * *Centered*, because the house's own null — `ctrliface_core.apply_mean_ablation`, the em-family
    used by CTRLIFACE / ENDOW / GRAIN / REGRAIN — removes `QQᵀ(x − h̄)`, i.e. the centered
    component. Bands cut from an UNCENTERED second moment would not be the object those nulls act
    on.
  * *BOS excluded*, because REGRAIN R16 measured BOS at ~10× the activation norm of content
    tokens (2615 vs 228/271 at L14). Included, the sink token would define the top band, and the
    top-band row of F-S1 would silently be a statement about the attention sink rather than about
    the data's principal axes.

Both choices are FROZEN, not tuned. The uncentered/all-position second moment is computed here as
the frozen **sensitivity row**, so the result doc can bound how much the y-axis definition drives
the surface — rather than leaving that question unaskable.

Also banked here, weight-free:
  * band purity of every §3 injector, measured on REAL activations (not planted tensors)
  * the level-set dispersion frontier at the real Σ — the V2/V3 instrument's own calibration,
    measured BEFORE any behavioural cell is read, so `H_thin-ball` cannot be an artifact of a
    sampler that collapses dispersion by construction.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_sigma --type r7i.4xlarge --no-weights \
      --s3-prefix results/bispec_sigma --budget-min 120 \
      --args "--harvest-s3 s3://…/results/regrain_pipeA/out/harvest"
"""
import argparse
import json
import pathlib

REAL_K = 25                 # REGRAIN's B* rank — the rank V2 must interrogate
LEVELSET_E = (0.03, 0.05, 0.10, 0.25)      # V3's fat cells + R1b's operating point
N_LEVELSET_DRAWS = 8
N_PURITY_ROWS = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--sites", default="L0,L14")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs._grain_shared import POS_BOS
    from infra.pg import bispec_core as BC

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    rep = {"object": "bispec_sigma_v1", "harvest_s3": a.harvest_s3,
           "amendment": "A2: centered covariance, BOS excluded; uncentered/all-pos = sensitivity row",
           "band_edges": list(BC.BAND_EDGES), "sites": {}}

    def _w():
        (out / "sigma.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banked = {}
    for key in a.sites.split(","):
        shard_names = meta["shards"][key]
        d = None
        # Two streaming accumulators in ONE pass: the frozen primary (centered, BOS-excluded) and
        # the sensitivity row (uncentered, all positions). fp64 throughout — the R1b fp32 roundoff
        # lesson applies to any Gram accumulated over 10^6 rows.
        acc = {"primary": {"S": None, "s": None, "n": 0}, "sens": {"S": None, "s": None, "n": 0}}
        n_bos = 0
        for name in shard_names:
            blob = torch.load(RS.fetch(f"{a.harvest_s3}/{name}", f"/tmp/bispec/shards/{name}"),
                              map_location="cpu", weights_only=False)
            X = blob["X"].double()
            lab = blob["labels"]
            if d is None:
                d = X.shape[1]
                for v in acc.values():
                    v["S"] = torch.zeros(d, d, dtype=torch.float64)
                    v["s"] = torch.zeros(d, dtype=torch.float64)
            keep = lab != POS_BOS
            n_bos += int((~keep).sum())
            for tag, Z in (("primary", X[keep]), ("sens", X)):
                acc[tag]["S"] += Z.transpose(0, 1) @ Z
                acc[tag]["s"] += Z.sum(0)
                acc[tag]["n"] += int(Z.shape[0])
            del blob, X, lab
            print(f"[sigma] {key} {name}: rows={acc['sens']['n']} bos={n_bos}", flush=True)

        site = {"n_rows_total": acc["sens"]["n"], "n_rows_bos": n_bos, "d": d}

        # primary: centered covariance, BOS-excluded.  Σ = E[zzᵀ] − h̄h̄ᵀ
        n = acc["primary"]["n"]
        hbar = acc["primary"]["s"] / n
        Sigma = acc["primary"]["S"] / n - torch.outer(hbar, hbar)
        Sigma = 0.5 * (Sigma + Sigma.transpose(0, 1))
        bands = BC.eig_bands(Sigma)
        site["primary"] = {"n_rows": n, "convention": "centered, BOS-excluded",
                           "band_slices": bands["band_slices"], "band_energy": bands["band_energy"],
                           "band_ranks": [hi - lo for lo, hi in bands["band_slices"]],
                           "top_eigs": [float(x) for x in bands["spectrum"][:8]],
                           "trace": float(bands["spectrum"].sum())}

        # sensitivity row: uncentered second moment, all positions (BOS included)
        ns = acc["sens"]["n"]
        M2 = acc["sens"]["S"] / ns
        M2 = 0.5 * (M2 + M2.transpose(0, 1))
        bands_s = BC.eig_bands(M2)
        hbar_all = acc["sens"]["s"] / ns
        # the diagnostic that motivated A2: how much of the uncentered top eigendirection IS the mean?
        v1 = bands_s["basis"][:, 0]
        mean_align = float(abs(v1 @ (hbar_all / hbar_all.norm())))
        site["sensitivity"] = {"n_rows": ns, "convention": "uncentered second moment, all positions",
                               "band_ranks": [hi - lo for lo, hi in bands_s["band_slices"]],
                               "band_energy": bands_s["band_energy"],
                               "top1_alignment_with_mean_direction": mean_align}
        # and how much the two bases actually disagree, at the rank of the top band
        k0 = bands["band_slices"][0][1]
        th = BC.principal_angles(bands["basis"][:, :k0], bands_s["basis"][:, :k0])
        site["sensitivity"]["top_band_principal_angles_deg"] = [float(x) * 180 / 3.14159265 for x in th]

        # ---- band purity of the real injectors, on REAL rows (pre-mortem (iii)) ----
        rows = torch.load(RS.fetch(f"{a.harvest_s3}/{shard_names[0]}",
                                   f"/tmp/bispec/shards/{shard_names[0]}"),
                          map_location="cpu", weights_only=False)["X"][:N_PURITY_ROWS].double()
        purity = []
        for b in range(len(bands["band_slices"])):
            U = BC.band_projector(bands, b)
            gen = torch.Generator().manual_seed(1000 + b)
            G = torch.randn(64, U.shape[1], generator=gen, dtype=torch.float64)
            delta = G @ U.transpose(0, 1)
            pur = BC.band_purity(delta, U)
            # deletion purity: the removed component must live inside the band too
            comp = ((rows - hbar) @ U) @ U.transpose(0, 1)
            del_pur = BC.band_purity(comp, U)
            purity.append({"band": b, "rank": U.shape[1], "add_purity": pur,
                           "delete_purity": del_pur,
                           "leak_ok": (1 - pur) < BC.LEAK_TOL and (1 - del_pur) < BC.LEAK_TOL})
        site["band_purity"] = purity
        assert all(p["leak_ok"] for p in purity), f"{key}: injector band leak ≥ 1% — cells VOID"

        # ---- the level-set dispersion frontier at the REAL Σ (V2/V3 calibration) ----
        # A8: the ladder is built from THIS site's reachable range. R1b's "rank 25, E = 25%" is an
        # L0 statement; at a flatter site E_max(25) can sit below 0.25 and the regime simply does
        # not exist there. Refusing it is correct; silently clamping it would not be.
        eig = torch.linalg.eigh(Sigma)          # (evals asc, evecs) — computed ONCE, reused per draw
        E_min_r, E_max_r, E_max_analytic = BC.reachable_energy_range(Sigma, REAL_K, eig=eig)
        # The analytic ceiling is attained only by the exact top-k eigenbasis — a single point with
        # zero dispersion. The samplable ladder must stop short of it, so the top rung is placed at
        # 95% of the NUMERICALLY REACHABLE max, and both numbers are banked.
        ladder = sorted({E for E in LEVELSET_E if E <= 0.95 * E_max_r} | {round(0.95 * E_max_r, 5)})
        deg = 180.0 / 3.141592653589793

        haar_ref = BC.haar_frames(d, REAL_K, N_LEVELSET_DRAWS, seed=77)
        ref_disp = BC.grassmann_diameter(haar_ref)
        ref_edisp = BC.energy_dispersion(haar_ref, Sigma)
        haar_E = [BC.removed_energy_fraction(Q, Sigma) for Q in haar_ref]

        frontier = []
        for E in ladder:
            frames, realized = [], []
            for s in range(N_LEVELSET_DRAWS):
                Q, Er = BC.sample_level_set(Sigma, REAL_K, E, seed=200 + s, eig=eig)
                frames.append(Q)
                realized.append(Er)
            disp = BC.grassmann_diameter(frames)
            edisp = BC.energy_dispersion(frames, Sigma)
            frontier.append({
                "target_E": E, "realized_E": realized,
                # the spec's statistic (B9) — reported, and expected to saturate
                "chordal_diameter": disp["diameter"], "chordal_mean": disp["mean_chordal"],
                "chordal_ratio_to_unconstrained": disp["diameter"] / ref_disp["diameter"],
                "mean_min_angle_deg": disp["mean_min_angle"] * deg,
                "mean_max_angle_deg": disp["mean_max_angle"] * deg,
                # the A7 statistic — dispersion of the REMOVED-ENERGY operator
                "energy_mean_distance": edisp["mean_distance"],
                "energy_mean_alignment": edisp["mean_alignment"],
                "energy_ratio_to_unconstrained": (edisp["mean_distance"] / ref_edisp["mean_distance"]
                                                  if ref_edisp["mean_distance"] > 0 else float("nan")),
            })
            f = frontier[-1]
            print(f"[sigma] {key} level-set E={E}: chordal_ratio={f['chordal_ratio_to_unconstrained']:.4f} "
                  f"min_angle={f['mean_min_angle_deg']:.1f}deg "
                  f"energy_align={f['energy_mean_alignment']:.4f} "
                  f"energy_ratio={f['energy_ratio_to_unconstrained']:.4f}", flush=True)
        site["levelset_frontier"] = {
            "rank": REAL_K, "n_draws": N_LEVELSET_DRAWS,
            "E_max_analytic": E_max_analytic, "E_max_reachable": E_max_r, "E_min_reachable": E_min_r,
            "ladder": ladder,
            "unconstrained_chordal_diameter": ref_disp["diameter"],
            "unconstrained_chordal_mean": ref_disp["mean_chordal"],
            "unconstrained_min_angle_deg": ref_disp["mean_min_angle"] * deg,
            "unconstrained_max_angle_deg": ref_disp["mean_max_angle"] * deg,
            "unconstrained_energy_mean_distance": ref_edisp["mean_distance"],
            "unconstrained_energy_mean_alignment": ref_edisp["mean_alignment"],
            "haar_removed_energy_mean": sum(haar_E) / len(haar_E),
            "frontier": frontier}

        rep["sites"][key] = site
        _w()
        banked[key] = {"Sigma": Sigma, "hbar": hbar, "basis": bands["basis"],
                       "spectrum": bands["spectrum"], "band_slices": bands["band_slices"],
                       "basis_uncentered": bands_s["basis"], "spectrum_uncentered": bands_s["spectrum"],
                       "hbar_all": hbar_all}
        del acc, M2

    torch.save(banked, out / "sigma_bank.pt")
    print("BISPEC_SIGMA_OK", json.dumps(
        {k: {"band_ranks": v["primary"]["band_ranks"],
             "top1_share_of_trace": v["primary"]["top_eigs"][0] / v["primary"]["trace"],
             "bos_rows": v["n_rows_bos"],
             "uncentered_top1_mean_alignment": v["sensitivity"]["top1_alignment_with_mean_direction"],
             "E_max_analytic": v["levelset_frontier"]["E_max_analytic"],
             "E_max_reachable": v["levelset_frontier"]["E_max_reachable"],
             "ladder": v["levelset_frontier"]["ladder"]}
         for k, v in rep["sites"].items()}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
