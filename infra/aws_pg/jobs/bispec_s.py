"""BISPEC Arm S (GPU) — the bi-spectral susceptibility surface Ŝ_r(band, mode, sign).

For each readout r, site L, and sign s ∈ {add, delete}: inject a perturbation whose power is
confined to one FEATURE band (a cumulative-energy quantile band of the site's Σ) and one SEQUENCE
mode, at two frozen amplitudes, ≥3 draws. The result is a small matrix per (readout × site × sign)
whose entries are behavioural damage with CIs.

**Measurement, not adjudication.** No gate fires here (spec §5.2). What the surface settles is
pictorial and quantitative: whether the VQA and ppl surfaces are dark in the same cells.

**A5 — the ω coordinate is NOT the same object on the two signs, and is never compared across them.**
  ω_add  the sequence envelope of the injected noise: dc (one draw, all positions) /
         sin_k (a low-frequency envelope) / white (an independent draw per position).
  ω_del  the per-position REMOVAL FRACTION c_t applied to the band projector. A projection is a
         per-token linear map; it has no sequence mode of its own. Giving it one means modulating
         how much is removed where — a different object, matched on mean-square so the total removed
         energy is comparable across modes, with the REALIZED removed energy measured per cell.
`support_overlap` is therefore computed within a sign only. The spec's cross-sign overlap
coefficient is not defined and is not reported.

**Named rows.** The program's named subspaces become rows of the surface, directly comparable to
their energy-band neighbours: `U_unt` (transported exactly to the residual by `orth(D_γ U)`, the A1
identity), `U_wr` (COMWRITE's consensus write), and REGRAIN's `B*` span. Each is compared to a
level-set twin drawn at ITS OWN measured removed energy — and each twin ensemble ships with its
energy-dispersion, because Gate-0a established that a matched-energy null does not automatically
have room to differ.

**What Gate-0a means for reading this surface at L0.** A single eigendirection carries 70.22 % of
the centered residual trace, so band 0 has rank 1 and band 1 has rank 10. The feature axis at L0 is
not a smooth spectrum; it is one direction, then nine, then hundreds. That is reported on the
figure's y-axis rather than hidden by it.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_s --type g6e.xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/bispec_s --budget-min 900 --args "--site L0 ..."
"""
import argparse
import json
import os
import pathlib

MODES_ADD = (("dc", None), ("sin", 1.0), ("sin", 4.0), ("white", None))
# "full" is the canonical band deletion (c=1), reported as its own row. dc/sin/white share a
# frozen mean-square (BC.DEL_MS_TARGET) and form the omega_del contrast (A5).
MODES_DEL = (("full", None), ("dc", None), ("sin", 1.0), ("white", None))
AMPS = (0.2, 0.4)            # × activation rms — the finite-radius disclosure (two points, never a Hessian)
N_DRAWS = 3
N_ITEMS = 160


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--named-rows-s3", default="", help="optional: banked U_unt/U_wr/B* bases (.pt)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs.bispec_gate0 import bispec_split
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    RS.install_pg()
    L = int(a.site[1:])
    rep = {"object": "bispec_s_v1", "site": a.site, "amps_over_rms": list(AMPS), "n_draws": N_DRAWS,
           "half": "discovery", "label": "DISCOVERY — measurement arm; no gate fires here",
           "amendment_A5": "omega_add (noise envelope) and omega_del (per-position removal fraction) "
                           "are different coordinates; support_overlap is computed WITHIN a sign only",
           "cells": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)[a.site]
    Sigma, hbar = bank["Sigma"], bank["hbar"]
    bands = BC.eig_bands(Sigma)
    eig = torch.linalg.eigh(Sigma)
    rep["bands"] = {"slices": bands["band_slices"], "energy": bands["band_energy"],
                    "ranks": [hi - lo for lo, hi in bands["band_slices"]],
                    "top1_share": float(bands["spectrum"][0] / bands["spectrum"].sum())}
    _w()

    sha, _ = RS.pinned_pg1_sha("/tmp/bispec")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/bispec/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = bispec_split(hashes, name="bispec_vqa")
    items = [rows[i] for i in sp["disc_idx"]][: a.n_items]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl, "dead_floor_vqa": floor,
                        "n_items": len(items)}
    _w()

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{a.harvest_s3}/{shard0}", f"/tmp/bispec/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
    xbar, rms, d = X.mean(0), float(X.pow(2).mean().sqrt()), X.shape[1]
    rep["site_rms"] = rms
    hb32 = hbar.float().cuda()

    def cell(row, mode, freq, sign, amp, draw, hook_fn, r2_rows=None, **extra):
        with torch.inference_mode():
            r2 = RC.r2_of(X, r2_rows(), xbar) if r2_rows is not None else None
        r = IP.eval_pg1(model, proc, vqa, items, "cuda", ctx=BC.hook_at(layers, L, hook_fn()))
        with BC.hook_at(layers, L, hook_fn()):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rec = {"row": row, "mode": mode if freq is None else f"{mode}{freq:g}", "sign": sign,
               "amp_over_rms": amp, "draw": draw, "realized_r2": r2,
               "vqa": r["score"], "dvqa": r["score"] - intact["score"],
               "effect": RS.effect_from(r["per_item"], intact["per_item"], floor),
               "effect_se": RS.effect_se(r["per_item"], intact["per_item"], floor),
               "ppl": p, "dppl": p - base_ppl, **extra}
        rep["cells"].append(rec)
        _w()
        print(f"[s] {sign} {row} {rec['mode']} a={amp}: dVQA={rec['dvqa']:+.4f} "
              f"dppl={rec['dppl']:+.3f} R2={'' if r2 is None else round(r2,4)}", flush=True)
        return rec

    # ---------------- the quantile-band rows ---------------------------------
    band_U = {f"band{b}": BC.band_projector(bands, b) for b in range(len(bands["band_slices"]))}

    # ---------------- the named rows ------------------------------------------
    named = {}
    if a.named_rows_s3:
        nb = torch.load(RS.fetch(a.named_rows_s3, "/tmp/bispec/named_rows.pt"),
                        map_location="cpu", weights_only=False)
        for nm, Q in nb.items():
            Q = torch.as_tensor(Q, dtype=torch.float64)
            err = float((Q.transpose(0, 1) @ Q - torch.eye(Q.shape[1], dtype=torch.float64)).abs().max())
            assert err < 1e-6, f"named row {nm} not orthonormal ({err})"
            named[nm] = Q
    rep["named_rows"] = {nm: {"rank": int(Q.shape[1]),
                              "removed_energy": BC.removed_energy_fraction(Q, Sigma)}
                         for nm, Q in named.items()}
    _w()

    # each named row gets a level-set twin at ITS OWN measured removed energy, with dispersion
    twins = {}
    for nm, Q in named.items():
        E = BC.removed_energy_fraction(Q, Sigma)
        k = Q.shape[1]
        E_min, E_max, E_an = BC.reachable_energy_range(Sigma, k, eig=eig)
        if not (E_min < E < E_max):
            rep["named_rows"][nm]["twin"] = {"status": "UNREACHABLE", "E": E,
                                             "E_reachable": [E_min, E_max]}
            continue
        fr = [BC.sample_level_set(Sigma, k, E, seed=1100 + i, eig=eig, tol=1e-3)[0] for i in range(3)]
        disp = BC.energy_dispersion(fr, Sigma)
        twins[nm] = fr
        rep["named_rows"][nm]["twin"] = {
            "status": "OK", "E_over_Emax": E / E_max,
            "energy_mean_alignment_among_twins": disp["mean_alignment"],
            "energy_alignment_to_row": [BC.energy_alignment(Q, T, Sigma) for T in fr],
            "reading": "alignment ≈ 1 ⇒ the twin had no room to differ from the row; the row's "
                       "'beyond-energy' status is untestable at this (k, E)"}
    _w()

    # ---------------- ADD side -----------------------------------------------
    for name, U in list(band_U.items()) + [(nm, Q) for nm, Q in named.items()]:
        Ug = U.float().cuda()
        for mode, freq in MODES_ADD:
            for amp in AMPS:
                sigma = amp * rms
                for s in range(N_DRAWS):
                    def rr(sg=sigma, Uu=Ug, m=mode, f=freq, ss=s):
                        # mirror the hook exactly: unit-norm coefficients, scaled by sigma*sqrt(d)
                        gen = torch.Generator().manual_seed(3 + 1000 * ss)
                        G = torch.randn(X.shape[0], Uu.shape[1], generator=gen)
                        G = (G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)).cuda()
                        return X + sg * (d ** 0.5) * (G @ Uu.transpose(0, 1))
                    cell(name, mode, freq, "add", amp, s,
                         lambda sg=sigma, Uu=Ug, m=mode, f=freq, ss=s:
                             BC.make_hook_band_noise(Uu, sg, 3 + 1000 * ss, m, freq=f, d_total=d),
                         rr if mode == "white" else None,
                         band_rank=int(U.shape[1]))

    # ---------------- DELETE side --------------------------------------------
    for name, U in list(band_U.items()) + [(nm, Q) for nm, Q in named.items()]:
        Ug = U.float().cuda()
        for mode, freq in MODES_DEL:
            cell(name, mode, freq, "delete", 1.0, 0,
                 lambda Uu=Ug, m=mode, f=freq: BC.make_hook_band_deletion(Uu, hb32, m, freq=f, seed=17),
                 (lambda Uu=Ug: X - ((X - hb32) @ Uu) @ Uu.transpose(0, 1)) if mode == "full" else None,
                 band_rank=int(U.shape[1]),
                 removed_energy=BC.removed_energy_fraction(U, Sigma),
                 envelope_ms=(1.0 if mode == "full" else BC.DEL_MS_TARGET),
                 in_omega_del_contrast=(mode != "full"))
        # the level-set twin of a named row, deleted the same way (the beyond-energy comparison)
        for i, T in enumerate(twins.get(name, [])):
            Tg = T.float().cuda()
            cell(f"{name}_twin{i}", "full", None, "delete", 1.0, i,
                 lambda Tt=Tg: BC.make_hook_band_deletion(Tt, hb32, "full"),
                 None, band_rank=int(T.shape[1]),
                 removed_energy=BC.removed_energy_fraction(T, Sigma), is_twin=True)

    # ---------------- the surface + its summaries -----------------------------
    def surface(readout, sign):
        rows = [r for r in band_U] + [nm for nm in named]
        modes = sorted({c["mode"] for c in rep["cells"] if c["sign"] == sign and not c.get("is_twin")})
        M, ann = [], []
        for r_ in rows:
            line, aline = [], []
            for m in modes:
                v = [c[readout] for c in rep["cells"]
                     if c["row"] == r_ and c["mode"] == m and c["sign"] == sign
                     and (c["amp_over_rms"] == max(AMPS) or sign == "delete")]
                line.append(float(np.mean(v)) if v else float("nan"))
                aline.append(float(np.std(v)) if len(v) > 1 else 0.0)
            M.append(line)
            ann.append(aline)
        return {"rows": rows, "modes": modes, "matrix": M, "sd": ann}

    rep["surfaces"] = {}
    for sign in ("add", "delete"):
        s_vqa = surface("dvqa", sign)
        s_ppl = surface("dppl", sign)
        rep["surfaces"][sign] = {"vqa": s_vqa, "ppl": s_ppl}
        A = np.abs(np.asarray(s_vqa["matrix"], dtype=float))
        B = np.abs(np.asarray(s_ppl["matrix"], dtype=float))
        ok = np.isfinite(A).all() and np.isfinite(B).all()
        if ok:
            rep["surfaces"][sign]["support_overlap"] = BC.support_overlap(A, B, q=0.75)
            rep["surfaces"][sign]["overlap_reading"] = (
                "overlap < 0.3 ⇒ the two readouts are dark in different regions of the "
                "(band × mode) plane — H_disjoint-support, within this sign (A5)")
        else:
            rep["surfaces"][sign]["support_overlap"] = None
    _w()

    print("BISPEC_S_OK", json.dumps({"n_cells": len(rep["cells"]),
                                     "overlap_add": rep["surfaces"]["add"].get("support_overlap"),
                                     "overlap_delete": rep["surfaces"]["delete"].get("support_overlap")},
                                    default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
