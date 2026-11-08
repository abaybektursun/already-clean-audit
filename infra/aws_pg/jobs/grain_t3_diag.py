"""GRAIN T3-v2 diagnostic + labeled re-adjudication (GPU) — journal E6.

Measures the cell the v1 gate never took: the em-ppl band (blank-image wikitext Δppl under
the SAME three energy-matched random spans used for the em-VQA band, seeds 500–502 at each
config's r_e). Re-adjudicates every banked config under the T3-v2 rule FROZEN in journal E6
BEFORE this job ran:

    vqa leg (unchanged): ΔVQA_splice ≥ em_vqa_mean − max(em_vqa_halfwidth, 0.01)
    ppl leg (v2):        Δppl_splice ≤ em_ppl_mean + max(em_ppl_halfwidth, 0.5)

Writes out/gates.json = the v1 artifact with per-config em_ppl_band, gate_pass (v2),
gate_pass_v1 preserved, m_star/pivot re-derived — the downstream --gates-s3 for G, and
--mstar-from for the secondary run. If the em-ppl band is tight, v2 ≡ v1 and INSTRUMENT-NULL
stands.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_t3_diag --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_t3diag --budget-min 150 --args "\
      --dicts-s3 s3://…/grain_t2/out/train_primary --harvest-s3 s3://…/grain_t/out/harvest \
      --gate0-s3 s3://…/grain_gate0/out"
"""
import argparse
import json
import os
import pathlib

M_GRID = (1, 4, 8, 16)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dicts-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import energy_matched_random

    GS.install_pg()
    rep = json.loads(GS.fetch(f"{a.dicts_s3}/gates.json", "/tmp/grain/gates_v1.json").read_text())
    assert rep["object"] == "grain_train_v2", "unexpected gates artifact"
    L = int(rep["L"])
    rep["object"] = "grain_train_v2+t3v2"
    rep["t3_rule"] = "v2 (journal E6): ppl leg em-calibrated; v1 verdict preserved per config"

    def _w():
        (out / "gates.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    meta = json.loads(GS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/grain/harvest_meta.json").read_text())
    stats = torch.load(GS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/grain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    sha, _ = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)["ppl"]
    assert abs(base_ppl - rep["gate_slice"]["base_ppl"]) < 1e-6, \
        "base ppl not reproduced — instrument drift"
    rep["diag_base_ppl"] = base_ppl
    _w()

    em_ppl_cache = {}

    def em_ppl_band(cell):
        band = cell["em_band"]
        r_key = int(round(band["r_e"] / 8.0) * 8) or 1
        if r_key not in em_ppl_cache:
            # reconstruct the SAME spans: energy_matched_random is deterministic in
            # (E_target, trace, seed); E_target/trace are recomputed per-config in v1's
            # em_band but the SPAN only depends on r_e + seed — reproduce via the banked
            # E_target-equivalent rank by drawing at the same seed and asserting rank.
            deltas = []
            for s in range(3):
                Qem, r_e = energy_matched_random(
                    IP.D, float(band["r_e"]), float(IP.D), 500 + s, "cpu", torch.float64)
                assert int(r_e) == int(band["r_e"]), "em span rank reconstruction drift"
                with IP.ablate_at(layers, L, Qem, hbar):
                    pem = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)["ppl"]
                deltas.append(pem - base_ppl)
            em_ppl_cache[r_key] = deltas
        d = em_ppl_cache[r_key]
        return {"deltas": d, "mean": float(np.mean(d)),
                "halfwidth": float((max(d) - min(d)) / 2)}

    for cid, cell in rep["configs"].items():
        if cell.get("state") != "ok":
            continue
        pband = em_ppl_band(cell)
        vqa_ok = bool(cell["gate_parts"]["vqa_ok"])
        ppl_ok_v2 = cell["delta_ppl"] <= pband["mean"] + max(pband["halfwidth"], 0.5)
        cell["em_ppl_band"] = pband
        cell["gate_pass_v1"] = bool(cell["gate_pass"])
        cell["gate_pass"] = bool(vqa_ok and ppl_ok_v2)
        cell["gate_parts"] = {"vqa_ok": vqa_ok, "ppl_ok_v2": bool(ppl_ok_v2)}
        print(f"[t3v2] {cid}: dppl={cell['delta_ppl']:.3f} em_ppl={pband['mean']:.3f}"
              f"±{pband['halfwidth']:.3f} → ppl_ok_v2={ppl_ok_v2} pass={cell['gate_pass']}",
              flush=True)
        _w()

    passing = {m: rep["configs"][f"m{m}_s{0}"] for m in M_GRID
               if rep["configs"].get(f"m{m}_s0", {}).get("gate_pass")}
    m_gt1 = {m: c for m, c in passing.items() if m > 1}
    if m_gt1:
        rep["m_star"] = int(min(m_gt1, key=lambda m: m_gt1[m]["quality"]["fvu"]))
        rep["pivot"] = "none"
    elif 1 in passing:
        rep["m_star"] = None
        rep["pivot"] = "grain1_only"
    else:
        rep["m_star"] = None
        rep["pivot"] = "instrument_null"
    rep["green"] = True
    _w()
    print("T3DIAG_OK", json.dumps({"pivot": rep["pivot"], "m_star": rep["m_star"],
                                   "gates_v2": {k: v.get("gate_pass")
                                                for k, v in rep["configs"].items()}},
                                  default=str))


if __name__ == "__main__":
    main()
