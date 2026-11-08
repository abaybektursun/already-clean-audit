"""SPECTER Arm C (CPU, zero forwards) — the naming arms, all read from the operator/library
banks + parent artifacts:

  C-1 (the MDE): u_dc vs the regenerated banked coherent draws. ppl leg = the 6 UNIQUE vectors
      (3 benign-disc + 3 mid-disc; A4), vqa leg = all 9 (ATLAS R5 disclosure). Guard first:
      regressor span < 10× ⇒ VOID-no-range ⇒ the top-8 spike-subspace energy regressor
      adjudicates instead. Exact permutation p; BH q=0.10 over the 2-leg family, p emitted.
      Pre-unblind prediction row written BEFORE the regression is computed.
  C-2: the S3 band×mode reversal at quadratic order (R-C2), from banked traces + probe draws.
  C-3: the coherence dial — R(site) vs the two behaviourally measured ratios, both recomputed
      from their parent cells on log(1+Δppl) means (A4); R(L8) journaled as the registered
      prediction when both sites land.
  C-4: the census — every stable spike (+ u_dc) vs the frozen library, matched-rank Haar p95
      nulls (4000 draws), identification ≥ 3×p95.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_c --type r7i.2xlarge --no-weights \
      --s3-prefix results/specter_c --budget-min 90
"""
import argparse
import json
import math
import os
import pathlib

N_HAAR_NULL = 4000
BAND_OF_SB = [0] + [1] * 3 + [2] * 4 + [3] * 4 + [4] * 4      # SUBBAND_COUNTS (1,3,4,4,4)


def _regen_coherent_draw(ss, d=2048):
    """The bispec_k coherent draw, regenerated exactly: make_hook_gaussian_group(σ, 11+1000·ss,
    g=None) draws cache[0] = σ·randn(d, gen((11+1000·ss)·1_000_003 + 0), fp32). Direction only —
    σ is applied from the banked cell's own realized level."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed((11 + 1000 * ss) * 1_000_003 + 0)
    return torch.randn(d, generator=gen, dtype=torch.float32)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-l8l14", action="store_true",
                    help="merge the results/specter_o_l8l14 operator pass (second C run)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("c")
    rep = {"object": "specter_c_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load_banks"):
        o_metrics = json.loads(SS.fetch(f"{SS.SPECTER_O}/metrics.json",
                                        "/tmp/specter/o_metrics.json").read_text())
        assert o_metrics.get("green"), "Arm O not green — Arm C refuses to run"
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt",
                                   "/tmp/specter/operator_bank.pt"),
                          map_location="cpu", weights_only=False)
        lib = torch.load(SS.fetch(f"{SS.SPECTER_O}/library_bank.pt",
                                  "/tmp/specter/library_bank.pt"),
                         map_location="cpu", weights_only=False)
        if a.with_l8l14:
            o2 = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_o_l8l14/out/metrics.json",
                                     "/tmp/specter/o2_metrics.json").read_text())
            assert o2.get("green"), "L8/L14 operator pass not green"
            bank2 = torch.load(SS.fetch(f"{SS.S3_BUCKET}/results/specter_o_l8l14/out/operator_bank.pt",
                                        "/tmp/specter/operator_bank2.pt"),
                               map_location="cpu", weights_only=False)
            bank_extra = bank2
            o_metrics_dial_extra = o2.get("dial", {})
        else:
            bank_extra, o_metrics_dial_extra = {}, {}
        bk = json.loads(SS.fetch(SS.BISPEC_K, "/tmp/specter/bispec_k.json").read_text())
        bks = json.loads(SS.fetch(SS.BISPEC_K_SEALED, "/tmp/specter/bispec_ks.json").read_text())
        at = json.loads(SS.fetch(SS.ATLAS_T, "/tmp/specter/atlas_t.json").read_text())
        bs = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/bispec_s/out/metrics.json",
                                 "/tmp/specter/bispec_s.json").read_text())
        bank.update(bank_extra)
        o_metrics.setdefault("dial", {}).update(o_metrics_dial_extra)
    _w()

    # =======================================================================
    # C-1 — the MDE. Prediction row FIRST (spec §9.3), then the regression.
    # =======================================================================
    with tm.stage("c1"):
        pred = {"row": "P4", "prediction": "tau >= 0.6, carrier named",
                "credence": "med (favourite #1 — hostile treatment noted)",
                "u_dc_stable": o_metrics.get("u_dc_stable")}
        (out / "pre_unblind_c1.json").write_text(json.dumps(pred, indent=2))
        u_dc = lib["u_dc"].double()
        u_dc = u_dc / u_dc.norm()

        def _coherent_cells(metrics):
            # the 6 unique discovery vectors live under TWO families: (P1, benign) and the
            # g-sweep's fully-coherent rung (P1g, g=None → arm 'coherent', mid) — verified
            # against the parent artifact's cell taxonomy (E5)
            return [c for c in metrics["cells"]
                    if c.get("family") in ("P1", "P1g") and c.get("arm") == "coherent"]

        disc_cells = _coherent_cells(bk)
        sealed_cells = _coherent_cells(bks)
        assert len(disc_cells) == 6, f"expected the 6 unique discovery coherent draws, got {len(disc_cells)}"
        assert len(sealed_cells) == 3, f"expected 3 sealed coherent draws, got {len(sealed_cells)}"
        rows = []
        for src, cells in (("disc", disc_cells), ("sealed", sealed_cells)):
            for c in cells:
                ss = int(c["draw"])
                g = _regen_coherent_draw(ss).double()
                sig = c.get("sigma_over_rms")
                assert sig is not None, f"cell missing sigma_over_rms: {list(c)}"
                delta = float(sig) * g                    # rms units cancel in rank statistics
                rows.append({"src": src, "draw": ss, "level": c.get("level"),
                             "sigma_over_rms": float(sig),
                             "reg_udc": float((delta @ u_dc) ** 2),
                             "log1p_dppl": math.log1p(max(float(c["dppl"]), -0.99)),
                             "effect": c.get("effect"), "dppl": float(c["dppl"])})
        rep["c1_rows"] = rows
        # the alternative regressor: top-8 spike-subspace energy of H_dc(L0, ppl)
        sp = bank["L0/dc/ppl"]["spikes"][:8]
        if sp:
            P8 = bank["L0/dc/ppl"]["evecs_A1"][:, [s["idx"] for s in sp]].double()
            for r, c in zip(rows, disc_cells + sealed_cells):
                ss = int(c["draw"])
                delta = float(c["sigma_over_rms"]) * _regen_coherent_draw(ss).double()
                r["reg_p8"] = float(((delta @ P8) ** 2).sum())
        # guard: span of the primary regressor across the UNIQUE draw set
        uniq = [r for r in rows if r["src"] == "disc"]
        regs = np.array([r["reg_udc"] for r in uniq])
        span = float(regs.max() / max(regs.min(), 1e-300))
        rep["c1_span_ratio"] = span
        reg_key = "reg_p8" if (span < 10.0 and sp) else "reg_udc"
        # legs (A4): ppl = the 6 unique (disc) vectors; vqa = all 9
        ppl_rows = [r for r in rows if r["src"] == "disc"]
        vqa_rows = [r for r in rows if r["effect"] is not None]
        tau_p, p_p = AF.perm_p_tau(np.array([r["log1p_dppl"] for r in ppl_rows]),
                                   np.array([r[reg_key] for r in ppl_rows]))
        tau_v, p_v = AF.perm_p_tau(np.array([r["effect"] for r in vqa_rows]),
                                   np.array([r[reg_key] for r in vqa_rows]))
        passed = SS.emit_family(rep, "c1", [p_p, p_v], ["ppl", "vqa"])
        rep["c1"] = dict(SC.adjudicate_rc1(
            tau_ppl=float(tau_p), p_ppl=float(p_p), tau_vqa=float(tau_v), p_vqa=float(p_v),
            bh_pass_ppl=bool(passed[0]), bh_pass_vqa=bool(passed[1]),
            regressor_span_ratio=span if reg_key == "reg_udc" else
            float(max(r["reg_p8"] for r in uniq) / max(min(r["reg_p8"] for r in uniq), 1e-300)),
            u_dc_stable=bool(o_metrics.get("u_dc_stable"))),
            regressor=reg_key, n_ppl=len(ppl_rows), n_vqa=len(vqa_rows),
            tau_ppl=float(tau_p), tau_vqa=float(tau_v))
        print(f"[c1] {rep['c1']['verdict']} tau_ppl={tau_p:.3f} (p={p_p:.4f}) "
              f"tau_vqa={tau_v:.3f} (p={p_v:.4f}) reg={reg_key}", flush=True)
    _w()

    # =======================================================================
    # C-2 — the reversal at quadratic order.
    # =======================================================================
    with tm.stage("c2"):
        key_w, key_d = "L0/white/ppl", "L0/dc/ppl"
        sbs_w, sbs_d = bank[key_w], bank[key_d]

        def band_stats(bk_entry, band):
            idx = [i for i, b in enumerate(BAND_OF_SB) if b == band]
            tr = sum(bk_entry["traces_sb"][i] for i in idx)
            kb = sum(round(bk_entry["traces_sb"][i] / max(bk_entry["permode_sb"][i], 1e-300))
                     for i in idx)
            draws = [q for i in idx for q in bk_entry["probe_qs_sb"][i]]
            return tr, kb, draws
        tr_w1, k1, draws_w1 = band_stats(sbs_w, 1)
        tr_d1, _, draws_d1 = band_stats(sbs_d, 1)
        tr_w4, k4, draws_w4 = band_stats(sbs_w, 4)
        tr_d4, _, draws_d4 = band_stats(sbs_d, 4)
        b1_ratio = (tr_w1 / max(k1, 1)) / max(float(np.median(draws_d1)), 1e-300)
        b4_ratio = (tr_d4 / max(k4, 1)) / max(tr_w4 / max(k4, 1), 1e-300)
        rep["c2"] = dict(SC.adjudicate_rc2(band1_white_ratio=b1_ratio,
                                           band4_reversed_ratio=b4_ratio),
                         tr_white_b1_per_dir=tr_w1 / max(k1, 1),
                         median_dc_b1=float(np.median(draws_d1)),
                         tr_dc_b4_per_dir=tr_d4 / max(k4, 1),
                         tr_white_b4_per_dir=tr_w4 / max(k4, 1),
                         banked_s3_source="results/bispec_s/out/metrics.json",
                         banked_s3_green=bool(bs.get("green")))
        print(f"[c2] {rep['c2']['verdict']} b1={b1_ratio:.2f} b4={b4_ratio:.2f}", flush=True)
    _w()

    # =======================================================================
    # C-3 — the coherence dial (comparison points recomputed; never pass/fail cuts).
    # =======================================================================
    with tm.stage("c3"):
        (out / "pre_unblind_c3.json").write_text(json.dumps(
            {"row": "P6", "prediction": "both sites within ±0.4", "credence": "low-med"},
            indent=2))
        R_l0 = o_metrics["dial"]["L0/ppl"]["R"]
        p3s = [c for c in bks["cells"] if c.get("family") == "P3"]
        coh = [math.log1p(max(float(c["dppl"]), -0.99)) for c in p3s if c["arm"] == "coherent"]
        ind = [math.log1p(max(float(c["dppl"]), -0.99)) for c in p3s if c["arm"] == "independent"]
        assert coh and ind, "bispec_k_sealed P3 cells missing"
        ratio_l0 = float(np.mean(coh) / max(np.mean(ind), 1e-12))
        t4 = [c for c in at["cells"] if c.get("family") == "T4_L14"]
        sh = [c["log1p_dppl"] for c in t4 if c["arm"] == "shared"]
        pt = [c["log1p_dppl"] for c in t4 if c["arm"] == "pertoken"]
        assert sh and pt, "atlas_t T4_L14 cells missing"
        ratio_l14 = float(np.mean(sh) / max(np.mean(pt), 1e-12))
        R_l14 = o_metrics["dial"].get("L14/ppl", {}).get("R")
        if R_l14 is None:
            rep["c3"] = {"verdict": "PENDING-L14", "R_l0": R_l0, "ratio_l0": ratio_l0,
                         "ratio_l14": ratio_l14,
                         "note": "L14 operator pass not yet run; C-3 re-adjudicated then"}
        else:
            rep["c3"] = dict(SC.adjudicate_rc3(R_l0=R_l0, ratio_l0=ratio_l0,
                                               R_l14=R_l14, ratio_l14=ratio_l14),
                             R_l0=R_l0, ratio_l0=ratio_l0, R_l14=R_l14, ratio_l14=ratio_l14)
            if rep["c3"]["verdict"] == "H_dial-quadratic":
                R_l8 = o_metrics["dial"].get("L8/ppl", {}).get("R")
                rep["c3"]["registered_prediction_R_L8"] = R_l8
                (out / "pre_unblind_c3_l8.json").write_text(json.dumps(
                    {"registered_prediction_R_L8": R_l8,
                     "battery": "L8 coherent-vs-iid contrast, pre-approved"}, indent=2))
        print(f"[c3] {rep['c3']['verdict']}", flush=True)
    _w()

    # =======================================================================
    # C-4 — the census, all stable spikes (+ u_dc) × the library, Haar nulls.
    # =======================================================================
    with tm.stage("c4"):
        d = 2048
        gen = torch.Generator(device="cpu").manual_seed(909)
        haar = torch.randn(N_HAAR_NULL, d, generator=gen, dtype=torch.float64)
        haar = haar / haar.norm(dim=1, keepdim=True)
        vec_objs, sub_objs = {}, {}
        for name, ten in lib.items():
            if not isinstance(ten, torch.Tensor) or name == "u_dc":
                # u_dc is THIS program's own measurement — a probe, never a library object
                # (the first run self-identified every spike with it; E5)
                continue
            t = ten.double()
            if t.dim() == 1 or (t.dim() == 2 and t.shape[1] == 1):
                v = t.reshape(-1)
                vec_objs[name] = v / v.norm()
            elif t.dim() == 2 and t.shape[0] == d:
                sub_objs[name] = t
        meta = lib["band_projectors_meta"]
        basis = lib["sigma_basis_L0"].double()
        for b, (lo, hi) in enumerate(meta["band_slices"]):
            sub_objs[f"band{b}_projector"] = basis[:, lo:hi]
        # spikes to census: every stable spike at every (site, mode, loss), plus u_dc
        probes = {"u_dc": lib["u_dc"].double() / lib["u_dc"].double().norm()}
        for key, entry in bank.items():
            for s in entry.get("spikes", [])[:8]:
                u = entry["evecs_A1"][:, s["idx"]].double()
                probes[f"spike/{key}/{s['idx']}"] = u / u.norm()
        census, nulls = {}, {}
        for oname, v in vec_objs.items():
            nulls[oname] = float((haar @ v).abs().quantile(0.95))
        for oname, Q in sub_objs.items():
            Qo, _ = torch.linalg.qr(Q)
            sub_objs[oname] = Qo
            nulls[oname] = float((haar @ Qo).norm(dim=1).quantile(0.95))
        for pname, u in probes.items():
            row = {}
            for oname, v in vec_objs.items():
                row[oname] = abs(float(u @ v))
            for oname, Q in sub_objs.items():
                row[oname] = float((Q.transpose(0, 1) @ u).norm())
            census[pname] = row
        rep["c4_census"] = census
        rep["c4_haar_p95"] = nulls
        # operator-INTERNAL alignment (descriptive, not identification): every spike vs u_dc
        u0 = lib["u_dc"].double()
        u0 = u0 / u0.norm()
        rep["c4_udc_alignment"] = {p: abs(float(u @ u0)) for p, u in probes.items()}
        adjs = {}
        for pname, row in census.items():
            adjs[pname] = SC.adjudicate_rc4(row, nulls)
        rep["c4"] = {p: a for p, a in adjs.items()}
        n_ident = sum(1 for a in adjs.values() if a["verdict"] == "H_identified")
        rep["c4_summary"] = {"n_probes": len(probes), "n_with_identifications": n_ident}
        print(f"[c4] probes={len(probes)} identified={n_ident}", flush=True)
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_C_OK", json.dumps({"c1": rep["c1"]["verdict"], "c2": rep["c2"]["verdict"],
                                      "c3": rep["c3"]["verdict"],
                                      "c4_idents": rep["c4_summary"]["n_with_identifications"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
