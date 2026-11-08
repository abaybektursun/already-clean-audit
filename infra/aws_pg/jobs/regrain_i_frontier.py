"""REGRAIN I3 (GPU) — the distortion frontier: how much corruption does the PG1 forward tolerate?

The BSF paper's App. G.1 protocol, adapted from a FROZEN PROBE to the LIVE FORWARD. The paper
degrades activations with `x_q = Δ·round(x/Δ)` and reads a probe trained once on clean features;
it never splices a reconstruction back into the network. That is exactly the gap REGRAIN's splice
gate fills — so the frontier must be measured on the forward itself, before any featurizer is
judged against it.

Four corruption families at the site, each swept to give a ΔVQA / Δppl curve against the realized
reconstruction fidelity R²(x, x') = 1 − ‖x'−x‖²/‖x−x̄‖² (the paper's definition, verbatim):

  quant   x' = Δ·round(x/Δ)                          Δ in units of the activation RMS  [App. G.1]
  gauss   x' = x + σ·ε                               isotropic, direction-free
  em_abl  x' = x − QQᵀ(x − h̄), Q random rank-r       low-rank REMOVAL (the house em null)
  rot_err x' = x + α·Q(x̂(x) − x)                     the m=1 dictionary's error, direction
                                                     destroyed at α=1 — the featurizer-anchored leg

Three distinct geometries at matched R² (per-coordinate rounding / isotropic noise / low-rank
removal / rotated reconstruction error) — so "how much distortion" and "what shape of distortion"
are separated. Every I2 featurizer is later placed on these curves at its own R².

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_i_frontier --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_i_frontier --budget-min 240 \
      --args "--harvest-s3 s3://…/results/regrain_harvest/out"
"""
import argparse
import json
import os
import pathlib

QUANT_STEPS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)      # × activation RMS
GAUSS_SIGMAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)        # × activation RMS
EM_RANKS = (8, 16, 32, 64, 128, 256, 512)
ROT_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--site", default="primary", choices=("primary", "secondary"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-r2-rows", type=int, default=8192)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import apply_mean_ablation, random_basis

    RS.install_pg()
    banks = RS.load_read_banks("/tmp/regrain")
    L = banks["saction"]["L"] if a.site == "primary" else banks["svl"]["L"]
    key = f"L{L}"
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    X, _lab, _st = RS.load_shards_stream(a.harvest_s3, meta, key, workdir="/tmp/regrain/shards")
    stats = torch.load(RS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    Xr = X[: a.n_r2_rows].float()                       # the R² measurement sample (fp32)
    rms = float(Xr.pow(2).mean().sqrt())
    rep = {"object": "regrain_i_frontier_v1", "site": a.site, "L": L, "rms": rms,
           "n_r2_rows": int(Xr.shape[0]), "n_ppl_seq": RS.N_PPL_SEQ, "rot_seed": RS.ROT_SEED,
           "r2_definition": "1 - ||x'-x||^2 / ||x-xbar||^2 (BSF App. G.1)", "points": []}

    def _w():
        (out / "frontier.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    items = [rows[i] for i in split["disc_idx"]][: a.n_items]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl, "n_items": len(items)}
    _w()

    Q = RC.haar_rotation(IP.D, RS.ROT_SEED, device="cuda", dtype=torch.float32)
    name = RS.dict_bank_name("primary", 1, 0)
    ae_m1, _c, _q, _m = GB.load_dict_bank(RS.fetch(f"{RS.GRAIN_DICTS_L0}/{name}",
                                                   f"/tmp/regrain/{name}"))
    ae_m1 = ae_m1.to("cuda").eval()
    Xr_gpu = Xr.to("cuda")
    xbar = Xr_gpu.mean(0)
    with torch.inference_mode():
        rec_m1 = ae_m1.decode(ae_m1.encode(Xr_gpu, use_threshold=True))

    def measure(family, level, hook_factory, corrupt_rows):
        """R² on the harvest sample + ΔVQA/Δppl on the live forward, under the SAME transform."""
        with torch.inference_mode():
            r2 = RC.r2_of(Xr_gpu, corrupt_rows(), xbar)     # fp64 promotion happens on device
        r = IP.eval_pg1(model, proc, vqa, items, "cuda",
                        ctx=RC.hook_at(layers, L, hook_factory()))
        with RC.hook_at(layers, L, hook_factory()):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        pt = {"family": family, "level": level, "r2": r2, "vqa": r["score"],
              "dvqa": r["score"] - intact["score"], "ppl": p, "dppl": p - base_ppl}
        rep["points"].append(pt)
        _w()
        print(f"[frontier] {family} {level}: R2={r2:.4f} dVQA={pt['dvqa']:+.4f} "
              f"dppl={pt['dppl']:+.3f}", flush=True)
        return pt

    for qs in QUANT_STEPS:
        step = qs * rms
        measure("quant", qs, lambda st=step: RC.make_pre_hook_quantize(st),
                lambda st=step: RC.quantize_step(Xr_gpu, st))

    for sg in GAUSS_SIGMAS:
        sigma = sg * rms
        def corrupt(s=sigma):
            g = torch.Generator(device="cuda").manual_seed(1234)
            return Xr_gpu + s * torch.randn(Xr_gpu.shape, generator=g, device="cuda")
        measure("gauss", sg, lambda s=sigma: RC.make_pre_hook_gaussian(s, 1234), corrupt)

    hb64 = hbar.double().to("cuda")
    for r_ in EM_RANKS:
        Qem = random_basis(IP.D, r_, 777, "cpu", torch.float64)
        Qem_gpu = Qem.to("cuda")
        def corrupt(Qe=Qem_gpu):
            return apply_mean_ablation(Xr_gpu.double(), Qe, hb64).float()
        measure("em_abl", r_, lambda Qe=Qem: IP.make_pg1_pre_hook_ablate(Qe, hbar), corrupt)

    for al in ROT_ALPHAS:
        def corrupt(al_=al):
            return RC.rotate_error(Xr_gpu, rec_m1, Q, alpha=al_)
        measure("rot_err", al, lambda al_=al: RC.make_pre_hook_rotated_splice(ae_m1, Q, alpha=al_),
                corrupt)

    # the true m=1 splice, as the frontier's anchor point (the un-rotated rot_err family member)
    with torch.inference_mode():
        r2_spl = RC.r2_of(Xr_gpu, rec_m1, xbar)
    r = IP.eval_pg1(model, proc, vqa, items, "cuda", ctx=GB.splice_at(layers, L, ae_m1))
    with GB.splice_at(layers, L, ae_m1):
        p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    rep["m1_splice_anchor"] = {"r2": r2_spl, "dvqa": r["score"] - intact["score"],
                               "dppl": p - base_ppl}
    _w()

    # frozen readouts: the R² at which each readout first moves beyond a 95%-of-clean bar
    def first_break(family, field, frac):
        pts = sorted([p for p in rep["points"] if p["family"] == family],
                     key=lambda p: -p["r2"])
        for p in pts:
            if field == "vqa" and p["vqa"] < frac * intact["score"]:
                return p["r2"]
            if field == "ppl" and p["ppl"] > base_ppl / frac:
                return p["r2"]
        return None

    rep["tolerance"] = {f: {"vqa_95": first_break(f, "vqa", 0.95),
                            "vqa_90": first_break(f, "vqa", 0.90),
                            "ppl_95": first_break(f, "ppl", 0.95)}
                        for f in ("quant", "gauss", "em_abl", "rot_err")}
    print("I_FRONTIER_OK", json.dumps({"tolerance": rep["tolerance"],
                                       "m1_anchor": rep["m1_splice_anchor"],
                                       "n_points": len(rep["points"])}, default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
