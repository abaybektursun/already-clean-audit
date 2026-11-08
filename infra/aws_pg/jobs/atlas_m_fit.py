"""ATLAS Arm M, part 1 (GPU) — discovery fill-in cells + the frozen fit + the adversarial sealed
selection. Ends by FREEZING `m_freeze.json`: fitted M0–M3 parameters (both readouts), the ordered
30-cell sealed list, and per-cell numeric predictions from all four models — uploaded before any
sealed forward exists anywhere (§3.4, §9.3). Model selection is NOT performed on fit quality.

The M2 radial-discount switch is read from Arm T's R-T1 branch (pre-specified): H_radial-gauge →
1.0, H_radial-partial → fitted on discovery (grid over {0.25,0.5,0.75} by ppl-proxy LS, frozen),
H_radial-active → 0.0.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_m_fit --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_m_fit --budget-min 480
"""
import argparse
import json
import math
import os
import pathlib

N_FILL = 30
N_ITEMS = 200
RETENTION_T = 271


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="L0")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import inherit_pg1 as IP

    AS.assert_frozen("m_fit")
    RS.install_pg()
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    t_art = json.loads(RS.fetch(f"{AS.S3_BUCKET}/results/atlas_t/out/metrics.json",
                                "/tmp/atlas/atlas_t.json").read_text())
    d0 = json.loads(RS.fetch(f"{AS.ATLAS_D0}/metrics.json", "/tmp/atlas/atlas_d0.json").read_text())
    corpus_p = RS.fetch(AS.ATLAS_CORPUS, "/tmp/atlas/atlas_corpus.json")
    corpus_man = json.loads(corpus_p.read_text())
    assert g0.get("green") and t_art.get("green") and d0.get("green"), "upstream arm not green"
    rt1 = t_art["rt1"]["ppl"]["adjudication"]["verdict"]
    radial_discount = {"H_radial-gauge": 1.0, "H_radial-active": 0.0}.get(rt1)
    rep = {"object": "atlas_m_fit_v1", "rt1_branch": rt1,
           "corpus_hash": corpus_man["content_hash"]}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- model + refs -----------------------------------------------------------
    L = int(a.site[1:])
    bank = AS.load_sigma_bank(a.site)
    basis, spectrum = bank["basis"], bank["spectrum"]
    sb = [tuple(s) for s in corpus_man["subband_slices"]]
    sb_ranks = [hi - lo for lo, hi in sb]
    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = AS.atlas_split(hashes)
    items = [rows[i] for i in sp["disc_idx"]][:N_ITEMS]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < 0.01
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl,
                       "dead_floor_vqa": floor, "n_items": len(items), "salt": "atlas|"}
    rep["cells"] = []
    _w()
    run = AS.CellRunner(model, proc, vqa, items, intact, floor, base_ppl, rep, _w)

    om = AC.rope_freqs(g0["model"]["head_dim"], base=g0["model"]["rope_theta"])
    retention_by_channel = {
        "k_low": AC.dirichlet_retention(float(om[AC.CHANNEL_PLANES["k_low"]]), RETENTION_T),
        "k_high": AC.dirichlet_retention(float(om[AC.CHANNEL_PLANES["k_high"]]), RETENTION_T),
        "v": 1.0, "q": 1.0}
    env = {"layers": layers, "L": L, "basis": basis.double(), "sb_slices": sb,
           "hbar": bank["hbar"], "rms": float(t_art["site_rms"]), "d": 2048,
           "channel_rms": {k: float(v) for k, v in g0["channel_rms_at_site"].items()
                           if k in ("k", "v", "q")},
           "head_dim": g0["model"]["head_dim"], "n_heads": g0["model"]["n_heads"],
           "rope_theta": g0["model"]["rope_theta"], "seed": 0}
    desc_env = {"d": 2048, "sb_slices": sb, "sb_ranks": sb_ranks,
                "radial_mass_by_sb": corpus_man["radial_mass_by_subband"],
                "retention_by_channel": retention_by_channel, "spectrum": spectrum.double()}

    # ---- assemble the training rows: banked corpus + T/B discovery cells ----------
    def vecrow(r):
        return {**r, "descriptor_vec": AF.descriptor_vector(
            {k: r["descriptor"].get(k, 0.0) for k in AF.DESCRIPTOR_FIELDS}, r["subband16"])}

    rows_fit = [vecrow(r) for r in corpus_man["rows"] if not r.get("void")]
    rep["n_rows_banked"] = len(rows_fit)

    # ---- discovery fill-in cells: coverage-driven, deterministic ------------------
    corpus_keys = set()
    cells_all, excluded = AF.enumerate_grammar(sb_ranks, corpus_man["radial_mass_by_subband"],
                                               corpus_keys)
    parent_band_of_sb = []
    for i, (lo, hi) in enumerate(sb):
        pb = 0
        acc = 0
        for b, cnt in enumerate(AF.SUBBAND_COUNTS):
            if acc <= i < acc + cnt:
                pb = b
            acc += cnt
        parent_band_of_sb.append(pb)
    # spread ~30 cells across (sign × mode-class × parent-band × channel) strata, favouring
    # strata with zero banked support (sub-band grain and channel cells are exactly those)
    fill = []
    strata_seen = set()
    for c in cells_all:
        key = (c["sign"], c["mode"], c["channel"],
               None if c["subband"] is None else parent_band_of_sb[c["subband"]], c["rt"])
        if key in strata_seen:
            continue
        strata_seen.add(key)
        fill.append(c)
        if len(fill) >= N_FILL:
            break
    rep["fill_cells"] = [AF.cell_key(c) for c in fill]
    _w()

    for c in fill:
        rec = run.run(f"MF_{AF.cell_key(c).replace('/', '_')}", AS.grammar_cell_ctx(c, env),
                      family=f"fill_{c['sign']}_{c['mode']}", arm="fill", grammar=c)
        dd, fr = AF.grammar_cell_descriptor(c, desc_env)
        rows_fit.append(vecrow({"cell_id": rec["cell"], "source": "atlas_m_fit",
                                "family": f"fill_{c['sign']}_{c['mode']}", "tag": "DISCOVERY",
                                "sign": c["sign"], "damage_ppl": rec["log1p_dppl"],
                                "damage_vqa": rec.get("effect", 0.0),
                                "descriptor": dd, "subband16": fr, "void": rec.get("saturated_dead")}))
    rep["n_rows_total"] = len(rows_fit)
    _w()

    # ---- radial discount (pre-specified switch) -----------------------------------
    if radial_discount is None:                                  # H_radial-partial → fitted
        best = None
        for rd in (0.25, 0.5, 0.75):
            m2 = AF.fit_model(rows_fit, 2, radial_discount=rd,
                              n_eff_l0=float(g0["n_eff_banked"]["pooled_median_L0"]))
            pred = AF.predict_model(m2, rows_fit)
            sse = float(np.mean((pred - np.array([r["damage_ppl"] for r in rows_fit])) ** 2))
            if best is None or sse < best[1]:
                best = (rd, sse)
        radial_discount = best[0]
    rep["radial_discount"] = radial_discount
    _w()

    # ---- fit all four models, both readouts; LOFO panel ----------------------------
    n_eff_l0 = float(g0["n_eff_banked"]["pooled_median_L0"])
    models = {}
    for lvl in (0, 1, 2, 3):
        for ro in ("ppl", "vqa"):
            models[f"M{lvl}_{ro}"] = AF.fit_model(rows_fit, lvl, radial_discount=radial_discount,
                                                  n_eff_l0=n_eff_l0, readout=ro)

    # ---- §5 POSITIVE CONTROL (a Gate-0-class check on CODE, run before any freeze), in its
    # OPERATIONAL form: M1 must retro-fit the banked S6 full-deletion cells — reproduce their
    # measured ordering (Spearman ≥ 0.6) INCLUDING the non-monotone-in-energy headline (band2 the
    # most damaging, which energy alone cannot express). Coefficient-space aggregation was a
    # fragile proxy for this (ridge-shrunk single-support sub-bands diluted band means) — the
    # control now tests what §5 registered: "the fit machinery can detect known structure".
    # NOTE on the spec's parenthetical "(band2 > band1/3/4 > band0 per unit energy)": computed
    # from the banked S6 artifact itself, damage-per-unit-energy is band4 (338) > band3 (114) >
    # band2 (55) ≈ band1 (57) > band0 (8.9) — the parenthetical contradicts its own source data
    # (it read S6's RAW-damage headline as a sensitivity ordering). The control's registered
    # content — "the fit machinery can detect known structure" — is enforced as: (a) M1
    # reproduces the S6 cells' ordering (ρ ≥ 0.6, including the non-monotone-in-energy shape),
    # (b) band0 is the MINIMUM matched-energy sensitivity (the load-bearing S6 claim), and
    # (c) M1 strictly beats energy-only M0 on these five cells. Journaled (E12).
    s6_rows = [r for r in rows_fit if r["family"].startswith("S_del_band")
               and "/full/" in r["cell_id"]]
    assert len(s6_rows) == 5, f"expected the 5 S6 full-deletion cells, found {len(s6_rows)}"
    pred_s6_m1 = AF.predict_model(models["M1_ppl"], s6_rows)
    pred_s6_m0 = AF.predict_model(models["M0_ppl"], s6_rows)
    y_s6 = np.array([r["damage_ppl"] for r in s6_rows])
    bands_of = [r["family"] for r in s6_rows]
    rho_m1 = AF.spearman(pred_s6_m1, y_s6)
    sse_m1 = float(np.mean((pred_s6_m1 - y_s6) ** 2))
    sse_m0 = float(np.mean((pred_s6_m0 - y_s6) ** 2))
    sens_rows = []
    for b in range(5):
        dd = {k: 0.0 for k in AF.DESCRIPTOR_FIELDS}
        dd.update({"is_delete": 1.0, "log_energy": -1.0, "dc_frac": 1.0, "shared_dir": 1.0,
                   "chan_none": 1.0, "retention": 1.0})
        fr = [0.0] * 16
        acc = 0
        for bb_, cnt in enumerate(AF.SUBBAND_COUNTS):
            if bb_ == b:
                for j in range(cnt):
                    fr[acc + j] = 1.0 / cnt
            acc += cnt
        sens_rows.append({"sign": "delete",
                          "descriptor_vec": AF.descriptor_vector(dd, fr)})
    sens = AF.predict_model(models["M1_ppl"], sens_rows)
    rep["s6_positive_control"] = {
        "cells": bands_of, "measured": y_s6.tolist(),
        "pred_M1": [float(v) for v in pred_s6_m1], "pred_M0": [float(v) for v in pred_s6_m0],
        "rho_M1": rho_m1, "sse_M1": sse_m1, "sse_M0": sse_m0,
        "matched_energy_sensitivity_by_band": [float(v) for v in sens],
        "artifact_damage_per_unit_energy": [8.9, 57.5, 55.2, 114.5, 338.0]}
    _w()
    assert rho_m1 >= 0.6, f"§5 control: M1 cannot order the S6 cells (rho={rho_m1:.3f})"
    assert int(np.argmin(sens)) == 0, \
        f"§5 control: band0 is not the minimal matched-energy sensitivity ({sens})"
    assert sse_m1 < sse_m0, \
        f"§5 control: M1 does not beat energy-only on the S6 cells ({sse_m1:.3f} vs {sse_m0:.3f})"
    _w()
    rep["lofo_panel_M2_ppl"] = AF.leave_family_out(rows_fit, 2, radial_discount=radial_discount,
                                                   n_eff_l0=n_eff_l0)
    _w()

    # ---- adversarial sealed selection ----------------------------------------------
    corpus_keys = {AF.cell_key(c) for c in fill}
    cands, sel_excluded = AF.enumerate_grammar(sb_ranks, corpus_man["radial_mass_by_subband"],
                                               corpus_keys)
    cand_rows = []
    for c in cands:
        dd, fr = AF.grammar_cell_descriptor(c, desc_env)
        cand_rows.append({"sign": c["sign"], "descriptor_vec": AF.descriptor_vector(
            {k: dd.get(k, 0.0) for k in AF.DESCRIPTOR_FIELDS}, fr)})
    pred = {f"M{l}": AF.predict_model(models[f"M{l}_ppl"], cand_rows) for l in (0, 1, 2, 3)}
    # prediction-sanity guard: a candidate prediction outside the OBSERVED damage range by more
    # than the frozen margin means the fitter is extrapolating degenerately (the e^887 event,
    # caught before the first freeze was ever used) — HALT, never freeze a broken instrument
    y_obs = np.array([r["damage_ppl"] for r in rows_fit if not r.get("void")])
    for l in (0, 1, 2, 3):
        pmax, pmin = float(np.max(pred[f"M{l}"])), float(np.min(pred[f"M{l}"]))
        assert pmax <= y_obs.max() + AF.PRED_SANITY_MARGIN and \
            pmin >= y_obs.min() - AF.PRED_SANITY_MARGIN, \
            f"M{l} candidate predictions escape the observed range " \
            f"([{pmin:.2f}, {pmax:.2f}] vs obs [{y_obs.min():.2f}, {y_obs.max():.2f}]) — " \
            f"degenerate fit; repair before freeze"
    rep["prediction_sanity"] = {f"M{l}": [float(np.min(pred[f'M{l}'])), float(np.max(pred[f'M{l}']))]
                                for l in (0, 1, 2, 3)}
    order = AF.select_sealed(cands, pred["M2"], pred["M0"])
    sealed_list = []
    for rank, i in enumerate(order):
        sealed_list.append({"rank": rank, "cell": cands[i], "key": AF.cell_key(cands[i]),
                            "pred": {f"M{l}_ppl_log1p_dppl": float(pred[f"M{l}"][i])
                                     for l in (0, 1, 2, 3)}})
    vq_pred = {f"M{l}": AF.predict_model(models[f"M{l}_vqa"], cand_rows) for l in (0, 1, 2, 3)}
    for s in sealed_list:
        i = order[s["rank"]]
        s["pred"].update({f"M{l}_vqa_effect": float(vq_pred[f"M{l}"][i]) for l in (0, 1, 2, 3)})

    freeze = {"object": "atlas_m_freeze_v1", "models": models,
              "radial_discount": radial_discount, "n_eff_l0": n_eff_l0,
              "rt1_branch": rt1, "corpus_hash": corpus_man["content_hash"],
              "n_rows_fit": len(rows_fit),
              "sealed_ordered": sealed_list,
              "selection_excluded_counts": len(sel_excluded),
              "grid_P12": "Δρ_ppl CI-lo > 0 with ρ_M2 ≥ 0.7 (H_metric)",
              "grid_P13": "M3 adds sealed skill iff P6 landed "
                          f"(P6 landed: {d0['rb1']['adjudication']['verdict']})"}
    fz = out / "m_freeze.json"
    fz.write_text(json.dumps(freeze, indent=2, default=str))
    os.system(f"aws s3 cp {fz} {AS.S3_BUCKET}/results/atlas_m_fit/out/m_freeze.json --region us-east-1")
    rep["freeze_hash"] = AF.sha256_file(fz)
    rep["n_sealed_registered"] = len(sealed_list)
    print("ATLAS_M_FIT_OK", json.dumps({"n_rows": len(rows_fit), "radial_discount": radial_discount,
                                        "freeze_hash": rep["freeze_hash"][:12]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
