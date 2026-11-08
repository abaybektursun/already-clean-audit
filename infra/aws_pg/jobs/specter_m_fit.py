"""SPECTER Arm M fit (CPU — zero forwards; the sealed battery is a separate GPU job run ONCE)
— assemble the link-calibration corpus, compute the frozen closed-form q(cell) functionals,
fit the two 1-D isotone links (M0 energy-only, M4 operator), enumerate the sealed grammar,
select the 30 adversarial cells, and freeze everything with per-cell numeric predictions
journaled BEFORE any sealed forward.

M4's contract (spec §3.1): zero fitted directional parameters. q(cell) per production, frozen
here (journal E5; A3 — banked cells use the EXPECTED functional over the construction's draw
distribution, SPECTER cells the same at selection time):
  add, channel=none, sub-band b, mode ω, rt r:
      q = E_tok · Σ_b sbfrac_b · permode_sb[ω][b];  rt=radial → the measured radial census
      constant replaces the band mean; rt=tangential → the band mean (radial mass excluded)
  add, channel ch, mode ω:   q = E_tok · channel_census[ch/ω]           (measured constants)
  delete, sub-band b, mode:  q = E_removed · permode_sb[ω_del][b], ω_del: shared→dc,
      per-token→white
Links fit per sign (add/delete — the parent atlas_fit convention, disclosed); PRED-SANITY
guard: candidate predictions may not leave the observed range by > 3 log1p units (halt).

Corpus (spec §4.3, disclosed): atlas_corpus.json rows + ATLAS's 20 sealed cells (their `atlas|`
contract closed at ATLAS's unblind; training data here) + SPECTER Arm S cells.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_m_fit --type r7i.2xlarge --no-weights \
      --s3-prefix results/specter_m_fit --budget-min 60
"""
import argparse
import json
import math
import os
import pathlib

BAND_OF_SB = [0] + [1] * 3 + [2] * 4 + [3] * 4 + [4] * 4
PRED_SANITY_MARGIN = 3.0


def q_m4_for(desc, sb16, permode, radial_census, chan_census):
    """The frozen closed-form functional. desc = DESCRIPTOR_FIELDS dict; sb16 = 16 fracs."""
    E = 10.0 ** float(desc["log_energy"])
    chans = [c for c in ("k_low", "k_high", "v", "q") if float(desc.get(f"chan_{c}", 0)) > 0]
    if chans:
        ch = chans[0]
        mode = "dc" if float(desc.get("dc_frac", 0)) >= 0.5 else "white"
        return E * chan_census[f"{ch}/{mode}"]
    if float(desc.get("is_delete", 0)) > 0:
        mode = "dc" if float(desc.get("shared_dir", 0)) >= 0.5 else "white"
        per_dir = sum(f * permode[mode][b] for b, f in enumerate(sb16))
        return E * max(per_dir, 1e-30)
    if float(desc.get("dc_frac", 0)) >= 0.5:
        mode = "dc"
    elif float(desc.get("shared_dir", 0)) >= 0.5:
        mode = "sin4"
    else:
        mode = "white"
    rf = float(desc.get("radial_frac", 0))
    per_dir = sum(f * permode[mode][b] for b, f in enumerate(sb16))
    if rf >= 0.999:                     # pure radial production: the measured census constant
        return E * max(radial_census, 1e-30)
    return E * max(per_dir, 1e-30)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("m_fit")
    rep = {"object": "specter_m_fit_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        o_metrics = json.loads(SS.fetch(f"{SS.SPECTER_O}/metrics.json",
                                        "/tmp/specter/o_metrics.json").read_text())
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt",
                                   "/tmp/specter/operator_bank.pt"),
                          map_location="cpu", weights_only=False)
        corpus = json.loads(SS.fetch(SS.ATLAS_CORPUS, "/tmp/specter/atlas_corpus.json").read_text())
        m_sealed = json.loads(SS.fetch(SS.ATLAS_M_SEALED, "/tmp/specter/atlas_ms.json").read_text())
        s_metrics = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_s/out/metrics.json",
                                        "/tmp/specter/s_metrics.json").read_text())
        assert o_metrics.get("green") and s_metrics.get("green")
    permode = {m: bank[f"L0/{m}/ppl"]["permode_sb"] for m in ("dc", "sin4", "white")}
    radial_census = o_metrics["ro2"]["L0/ppl/radial"]["q_radial"]
    chan_census = o_metrics["channel_census"]["L0/ppl"]
    sb = [tuple(x) for x in corpus["subband_slices"]]
    rmass = corpus["radial_mass_by_subband"]
    sig = SS.load_sigma_bank("L0", workdir="/tmp/specter")
    spectrum_l0 = torch.as_tensor(sig["spectrum"], dtype=torch.float64)   # REAL spectrum: the
    # delete-cell descriptor's removed-energy term depends on it (a flat fake would distort it)
    _w()

    # ---- corpus rows (each with q_M0, q_M4, damages) --------------------------------
    with tm.stage("corpus"):
        rows = []
        for r in corpus["rows"]:
            if r.get("void"):
                continue
            rows.append({"cell_id": r["cell_id"], "source": "atlas_corpus",
                         "sign": r["sign"], "desc": r["descriptor"], "sb16": r["subband16"],
                         "damage_ppl": r["damage_ppl"], "damage_vqa": r["damage_vqa"]})
        for e in m_sealed["scored"]:
            c = e["cell"]
            desc, sb16 = AF.grammar_cell_descriptor(c, {
                "d": 2048, "sb_slices": sb, "radial_mass_by_sb": rmass,
                "retention_by_channel": {}, "spectrum": spectrum_l0})
            rows.append({"cell_id": f"atlas_sealed/{e['key']}", "source": "atlas_sealed_disclosed",
                         "sign": c["sign"], "desc": desc, "sb16": sb16,
                         "damage_ppl": e["measured_log1p_dppl"],
                         "damage_vqa": e["measured_vqa_effect"]})
        n_guard_dirty = 0
        for c in s_metrics["cells"]:
            if c["family"] != "spike" or "log1p_dppl" not in c:
                continue
            if c.get("saturated_ppl") or c.get("saturated_dead"):
                # §4.3 inclusion rule: GUARD-CLEAN only — an obliterated cell's mean-NLL
                # readout is not on the behavioural damage scale (E8); counted, never silent
                n_guard_dirty += 1
                continue
            al = abs(c.get("alpha", 0.1))
            desc = {k: 0.0 for k in AF.DESCRIPTOR_FIELDS}
            desc.update({"log_energy": math.log10(al ** 2), "dc_frac": 1.0, "shared_dir": 1.0,
                         "mean_shift": al, "radial_frac": 0.0, "chan_none": 1.0,
                         "retention": 1.0})
            # spike cells are realized directions: q_M4 for them is the REALIZED spectral
            # completion (they were constructed from banked eigvecs, i.e. known directions)
            rows.append({"cell_id": c["cell"], "source": "specter_s", "sign": "add",
                         "desc": desc, "sb16": None, "spike_idx": c.get("spike"),
                         "arm": c["arm"], "damage_ppl": c["log1p_dppl"],
                         "damage_vqa": c.get("effect")})
        rep["corpus_counts"] = {}
        for r in rows:
            rep["corpus_counts"][r["source"]] = rep["corpus_counts"].get(r["source"], 0) + 1
        rep["corpus_excluded"] = {"specter_s_guard_dirty_saturated": n_guard_dirty}
    _w()

    # ---- q values + links -------------------------------------------------------------
    with tm.stage("fit"):
        lib = torch.load(SS.fetch(f"{SS.SPECTER_O}/library_bank.pt",
                                  "/tmp/specter/library_bank.pt"),
                         map_location="cpu", weights_only=False)
        basis = lib["sigma_basis_L0"].double()
        e0 = bank["L0/dc/ppl"]
        sp = e0["spikes"][:8]
        U = e0["evecs_A1"][:, [s["idx"] for s in sp]].double() if sp else None
        lam = [s["eval_a1"] for s in sp]
        bulk = SC.band_bulk_from_traces(e0["traces_sb"], sb, U, lam, basis)
        for r in rows:
            E = 10.0 ** float(r["desc"]["log_energy"])
            r["q_m0"] = E
            if r["source"] == "specter_s" and r.get("spike_idx") is not None \
                    and r.get("arm") == "spike":
                promoted = s_metrics["promoted"][r["spike_idx"]]
                u = bank[promoted["key"]]["evecs_A1"][:, promoted["idx"]].double()
                r["q_m4"] = E * SC.spectral_completion_quadform(u, U, lam, bulk, basis, sb)
            elif r["source"] == "specter_s":
                # null/rotated cells: expected functional of a generic direction
                r["q_m4"] = E * (e0["trace_total"] / 2048.0)
            else:
                r["q_m4"] = q_m4_for(r["desc"], r["sb16"] or [0.0] * 16, permode,
                                     radial_census, chan_census)
        links = {}
        for sign in ("add", "delete"):
            sub = [r for r in rows if r["sign"] == sign]
            if not sub:
                continue
            links[sign] = {
                "m0_ppl": SC.fit_link([r["q_m0"] for r in sub],
                                      [r["damage_ppl"] for r in sub]),
                "m4_ppl": SC.fit_link([r["q_m4"] for r in sub],
                                      [r["damage_ppl"] for r in sub]),
                "m0_vqa": SC.fit_link([r["q_m0"] for r in sub if r["damage_vqa"] is not None],
                                      [r["damage_vqa"] for r in sub
                                       if r["damage_vqa"] is not None]),
                "m4_vqa": SC.fit_link([r["q_m4"] for r in sub if r["damage_vqa"] is not None],
                                      [r["damage_vqa"] for r in sub
                                       if r["damage_vqa"] is not None])}
        rep["links_n"] = {s: {k: v["n"] for k, v in d.items()} for s, d in links.items()}
    _w()

    # ---- grammar enumeration + adversarial selection ------------------------------------
    with tm.stage("select"):
        sb_ranks = [hi - lo for lo, hi in sb]
        corpus_keys = {e["key"] for e in m_sealed.get("scored", []) if "key" in e}
        corpus_keys |= {v["key"] for v in m_sealed.get("voids", []) if "key" in v}
        cands, excluded = AF.enumerate_grammar(sb_ranks, rmass, corpus_keys)
        rep["grammar"] = {"n_candidates": len(cands), "n_excluded": len(excluded),
                          "excluded_reasons": {}}
        for e in excluded:
            k = e["reason"]
            rep["grammar"]["excluded_reasons"][k] = rep["grammar"]["excluded_reasons"].get(k, 0) + 1
        env = {"d": 2048, "sb_slices": sb, "radial_mass_by_sb": rmass,
               "retention_by_channel": {}, "spectrum": spectrum_l0}
        preds = []
        for c in cands:
            desc, sb16 = AF.grammar_cell_descriptor(c, env)
            qm0 = 10.0 ** float(desc["log_energy"])
            qm4 = q_m4_for(desc, sb16, permode, radial_census, chan_census)
            sign = c["sign"]
            lk = links[sign]
            preds.append({"cell": c, "key": AF.cell_key(c), "q_m0": qm0, "q_m4": qm4,
                          "pred": {"M0_ppl": float(SC.predict_link(lk["m0_ppl"], [qm0])[0]),
                                   "M4_ppl": float(SC.predict_link(lk["m4_ppl"], [qm4])[0]),
                                   "M0_vqa": float(SC.predict_link(lk["m0_vqa"], [qm0])[0]),
                                   "M4_vqa": float(SC.predict_link(lk["m4_vqa"], [qm4])[0])}})
        obs = [r["damage_ppl"] for r in rows]
        lo_s, hi_s = min(obs) - PRED_SANITY_MARGIN, max(obs) + PRED_SANITY_MARGIN
        bad = [p for p in preds
               for v in (p["pred"]["M0_ppl"], p["pred"]["M4_ppl"]) if not lo_s <= v <= hi_s]
        assert not bad, f"PRED-SANITY violated on {len(bad)} candidates — fitter degenerate, halt"
        chosen = AF.select_sealed(cands, [p["pred"]["M4_ppl"] for p in preds],
                                  [p["pred"]["M0_ppl"] for p in preds])
        ordered = [dict(rank=i, **preds[j]) for i, j in enumerate(chosen)]
        fz = {"object": "specter_m_freeze_v1", "links": links,
              "sealed_ordered": ordered, "corpus_hash": corpus["content_hash"],
              "subband_slices": sb, "radial_mass_by_subband": rmass,
              "q_functionals": "specter_m_fit.q_m4_for (frozen in code)",
              "operator_bank_sha256": o_metrics["bank_sha256"]}
        (out / "m_freeze.json").write_text(json.dumps(fz, indent=2, default=str))
        rep["sealed_ordered_keys"] = [o["key"] for o in ordered]
        rep["n_selected"] = len(ordered)
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_M_FIT_OK", json.dumps({"n_cands": rep["grammar"]["n_candidates"],
                                          "n_selected": rep["n_selected"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
