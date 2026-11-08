"""BISPEC V2/V3 — is the dispersion of an energy-matched subspace null a function of E/E_max(k)?

The protected curiosity slice (spec §15), and it turned out to serve the MDE.

Gate-0a measured the level-set dispersion at rank 25 on both sites and found the SAME nominal
regime — "rank 25, E = 25 %" — to be FAT at L0 (energy-ratio 0.886) and THIN at L14 (0.035). The
two sites' spectra differ enormously: λ₁ carries 70.2 % of the centered trace at L0 but 10.3 % at
L14, so E_max(25) is 0.836 vs 0.307. Rescaled by E/E_max the two frontiers nearly coincide.

That suggests a law with no name yet:

    the dispersion of a rank-k, energy-E subspace ensemble depends (to leading order) only on
    E / E_max(k) — the fraction of the MAXIMUM energy that k directions can remove —
    and not separately on E, on k, or on the site.

If it holds, it is computable from the covariance spectrum alone, with **zero forwards**. It tells
any study, before it spends a GPU-hour, whether its energy-matched subspace null has room to
falsify anything — and it bounds every em-null this program has issued (CTRLIFACE, ENDOW, GRAIN,
REGRAIN).

This job tests it where it could fail: a rank ladder × an E/E_max ladder × two sites. If the curves
do NOT collapse, the observation was a rank-25 coincidence and we say so — the collapse is the
hypothesis, and separation is a reachable, informative observation.

DISCOVERY / EXPLORATORY. Weight-free. Adjudicates nothing behavioural.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_levelset_law --type r7i.4xlarge --no-weights \
      --s3-prefix results/bispec_levelset_law --budget-min 180 \
      --args "--sigma-s3 s3://…/results/bispec_sigma/out"
"""
import argparse
import json
import pathlib

RANKS = (8, 25, 64, 128, 256)
FRACS = (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)   # of E_max(k), the site-and-rank-free coordinate
N_DRAWS = 6
N_STEPS = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--sites", default="L0,L14")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import bispec_core as BC

    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)
    rep = {"object": "bispec_levelset_law_v1", "ranks": list(RANKS), "fracs": list(FRACS),
           "n_draws": N_DRAWS, "n_steps": N_STEPS,
           "hypothesis": "dispersion depends only on E/E_max(k); collapse across (site, rank)",
           "kill": "curves separate by rank or site at matched E/E_max => rank-25 coincidence",
           "sites": {}}

    def _w():
        (out / "levelset_law.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    deg = 180.0 / 3.141592653589793
    for site in a.sites.split(","):
        Sigma = bank[site]["Sigma"]
        d = Sigma.shape[0]
        eig = torch.linalg.eigh(0.5 * (Sigma + Sigma.transpose(0, 1)))
        srec = {"d": d, "top1_share": float(eig[0].flip(0)[0] / eig[0].clamp_min(0).sum()), "ranks": {}}
        for k in RANKS:
            E_min_r, E_max_r, E_max_an = BC.reachable_energy_range(Sigma, k, eig=eig, n_steps=N_STEPS)
            haar = BC.haar_frames(d, k, N_DRAWS, seed=77)
            h_ch = BC.grassmann_diameter(haar)
            h_en = BC.energy_dispersion(haar, Sigma)
            E_haar = sum(BC.removed_energy_fraction(Q, Sigma) for Q in haar) / N_DRAWS
            rows = []
            for fr in FRACS:
                E = fr * E_max_r
                if not (E_min_r < E < E_max_r):
                    print(f"[law] {site} k={k} frac={fr}: E={E:.5f} outside reachable "
                          f"({E_min_r:.5f},{E_max_r:.5f}) — SKIPPED (recorded, not imputed)", flush=True)
                    rows.append({"frac_of_Emax": fr, "target_E": E, "status": "UNREACHABLE"})
                    _w()
                    continue
                frames, realized = [], []
                for s in range(N_DRAWS):
                    Q, Er = BC.sample_level_set(Sigma, k, E, seed=200 + s, eig=eig,
                                                n_steps=N_STEPS, tol=1e-3)
                    frames.append(Q)
                    realized.append(Er)
                ch = BC.grassmann_diameter(frames)
                en = BC.energy_dispersion(frames, Sigma)
                rows.append({
                    "frac_of_Emax": fr, "target_E": E, "realized_E_mean": sum(realized) / len(realized),
                    "status": "OK",
                    "energy_mean_alignment": en["mean_alignment"],
                    "energy_ratio": en["mean_distance"] / h_en["mean_distance"],
                    "chordal_ratio": ch["diameter"] / h_ch["diameter"],
                    "mean_min_angle_deg": ch["mean_min_angle"] * deg,
                })
                r = rows[-1]
                print(f"[law] {site} k={k:3d} E/Emax={fr:.2f} E={E:.4f}: "
                      f"en_ratio={r['energy_ratio']:.4f} chord_ratio={r['chordal_ratio']:.4f} "
                      f"min_ang={r['mean_min_angle_deg']:.1f}deg", flush=True)
                _w()
            srec["ranks"][str(k)] = {
                "E_max_analytic": E_max_an, "E_max_reachable": E_max_r, "E_min_reachable": E_min_r,
                "haar_removed_energy": E_haar,
                "haar_energy_alignment": h_en["mean_alignment"],
                "haar_chordal_diameter": h_ch["diameter"],
                "rows": rows}
            _w()
        rep["sites"][site] = srec
        _w()

    # ---- the collapse test: at each frac, how much does energy_ratio spread across (site, rank)? ----
    collapse = []
    for fr in FRACS:
        vals = []
        for site, srec in rep["sites"].items():
            for k, kr in srec["ranks"].items():
                for row in kr["rows"]:
                    if row["frac_of_Emax"] == fr and row.get("status") == "OK":
                        vals.append((site, int(k), row["energy_ratio"]))
        if len(vals) >= 2:
            v = [x[2] for x in vals]
            collapse.append({"frac_of_Emax": fr, "n": len(v), "min": min(v), "max": max(v),
                             "spread": max(v) - min(v), "mean": sum(v) / len(v),
                             "by_cell": [{"site": s, "rank": k, "energy_ratio": e} for s, k, e in vals]})
    rep["collapse_test"] = collapse
    # A collapse means the spread at each frac is small compared to the range the curve traverses.
    traversed = max(c["mean"] for c in collapse) - min(c["mean"] for c in collapse)
    worst = max(c["spread"] for c in collapse)
    rep["collapse_summary"] = {
        "curve_range_across_fracs": traversed, "worst_spread_within_a_frac": worst,
        "ratio": worst / traversed if traversed > 0 else float("nan"),
        "reading": "ratio << 1 => the curves collapse onto E/E_max; ratio ~ 1 => they do not, and "
                   "the rank-25 observation was a coincidence"}
    _w()

    print("BISPEC_LEVELSET_LAW_OK", json.dumps(rep["collapse_summary"], default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
