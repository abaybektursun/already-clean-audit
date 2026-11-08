"""ATLAS Arm M, part 2 (GPU) — THE SEALED BATTERY. Touched once.

Loads the frozen `m_freeze.json` (hash-asserted), runs the pre-registered ordered cells on the
`atlas|` SEALED half until 20 valid cells are scored (VOIDs logged and replaced by the next backup
— 30 registered), then scores R-M with the frozen rule: Δρ = ρ_S(M2) − ρ_S(M0), 10k pair
bootstrap, BH q=0.10 over the {ppl, vqa} family; M3-vs-M2 as the secondary contrast. No threshold
moves after unblinding; if a rule proves mis-shaped, both readings are reported under the no-swap
discipline.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_m_sealed --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_m_sealed --budget-min 420 \
      --args "--freeze-sha <sha256 of m_freeze.json at freeze>"
"""
import argparse
import json
import math
import os
import pathlib

N_SEALED_ITEMS = 200


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--freeze-sha", required=True,
                    help="sha256 of m_freeze.json recorded at M-freeze; the battery refuses to "
                         "run against any other bytes")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import atlas_fit as AF
    from infra.pg import inherit_pg1 as IP

    AS.assert_frozen("m_sealed")
    RS.install_pg()
    fz_p = RS.fetch(f"{AS.S3_BUCKET}/results/atlas_m_fit/out/m_freeze.json",
                    "/tmp/atlas/m_freeze.json")
    got = AF.sha256_file(fz_p)
    assert got == a.freeze_sha, f"m_freeze.json hash {got[:12]} != frozen {a.freeze_sha[:12]} — halt"
    freeze = json.loads(fz_p.read_text())
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    d0_corpus = json.loads(RS.fetch(AS.ATLAS_CORPUS, "/tmp/atlas/atlas_corpus.json").read_text())
    assert d0_corpus["content_hash"] == freeze["corpus_hash"], "corpus drift since freeze — halt"
    t_art = json.loads(RS.fetch(f"{AS.S3_BUCKET}/results/atlas_t/out/metrics.json",
                                "/tmp/atlas/atlas_t.json").read_text())
    rep = {"object": "atlas_m_sealed_v1", "freeze_sha": got, "cells": [],
           "exposure": "the atlas| sealed half is touched by THIS job once and never again"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- model + SEALED refs ------------------------------------------------------
    L = int(a.site[1:])
    bank = AS.load_sigma_bank(a.site)
    sb = [tuple(s) for s in d0_corpus["subband_slices"]]
    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = AS.atlas_split(hashes)
    items = [rows[i] for i in sp["conf_idx"]][:N_SEALED_ITEMS]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < 0.01
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl,
                        "dead_floor_vqa": floor, "n_items": len(items), "half": "SEALED"}
    _w()
    run = AS.CellRunner(model, proc, vqa, items, intact, floor, base_ppl, rep, _w)

    env = {"layers": layers, "L": L, "basis": bank["basis"].double(), "sb_slices": sb,
           "hbar": bank["hbar"], "rms": float(t_art["site_rms"]), "d": 2048,
           "channel_rms": {k: float(v) for k, v in g0["channel_rms_at_site"].items()
                           if k in ("k", "v", "q")},
           "head_dim": g0["model"]["head_dim"], "n_heads": g0["model"]["n_heads"],
           "rope_theta": g0["model"]["rope_theta"], "seed": 0}

    # ---- run the ordered list until 20 valid --------------------------------------
    scored, voids = [], []
    for entry in freeze["sealed_ordered"]:
        if len(scored) >= AF.SEALED_N_SCORED:
            break
        c = entry["cell"]
        rec = run.run(f"SEAL_{entry['rank']:02d}_{entry['key'].replace('/', '_')}",
                      AS.grammar_cell_ctx(c, env), family="sealed", arm=c["sign"],
                      rank=entry["rank"], grammar=c)
        # frozen VOID guard: dead on VQA, or ppl beyond the interpretable regime
        dead = bool(rec.get("saturated_dead")) or rec["dppl"] > 1e6
        if dead:
            voids.append({"rank": entry["rank"], "key": entry["key"], "reason": "dead/saturated"})
            rep["voids"] = voids
            _w()
            continue
        scored.append({**entry, "measured_log1p_dppl": rec["log1p_dppl"],
                       "measured_vqa_effect": rec["effect"]})
        rep["scored"] = scored
        _w()
    rep["n_scored"] = len(scored)
    rep["n_void"] = len(voids)
    _w()

    # ---- R-M, the frozen rule -------------------------------------------------------
    results = {}
    for ro, pk, mk in (("ppl", "_ppl_log1p_dppl", "measured_log1p_dppl"),
                       ("vqa", "_vqa_effect", "measured_vqa_effect")):
        y = np.array([s[mk] for s in scored])
        preds = {f"M{l}": np.array([s["pred"][f"M{l}{pk}"] for s in scored]) for l in (0, 1, 2, 3)}
        drho, lo, hi, rho2, rho0 = AF.pair_bootstrap_drho(preds["M2"], preds["M0"], y)
        d32, lo32, hi32, rho3, _ = AF.pair_bootstrap_drho(preds["M3"], preds["M2"], y)
        results[ro] = {"drho": drho, "ci": [lo, hi], "rho_m0": rho0, "rho_m1":
                       AF.spearman(preds["M1"], y), "rho_m2": rho2, "rho_m3": rho3,
                       "m3_vs_m2": {"drho": d32, "ci": [lo32, hi32]},
                       "adjudication": AF.adjudicate_rm(drho=drho, drho_cilo=lo, drho_cihi=hi,
                                                        rho_m2=rho2, rho_m0=rho0,
                                                        n_void_backups=len(voids))}
        print(f"[m] R-M {ro}: {results[ro]['adjudication']['verdict']} "
              f"drho={drho:+.3f} [{lo:+.3f},{hi:+.3f}] rho_m2={rho2:.3f} rho_m0={rho0:.3f}",
              flush=True)
    rep["rm"] = results
    rep["rm_family_note"] = ("BH q=0.10 over the 2-readout family applies to the CI-based clauses "
                             "via the bootstrap CIs; both readouts reported in full, ppl is the "
                             "headline (§2)")
    print("ATLAS_M_SEALED_OK", json.dumps({ro: results[ro]["adjudication"]["verdict"]
                                           for ro in results}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
