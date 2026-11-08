"""REGRAIN Arm F (GPU) — steering the handle (causal register, discovery half).

If Arm C confirmed a handle, its atoms span a subspace of the L0 stream. The paper's
two-resolution reading (§3.5) says a unit carries a concept's PRESENCE in its code norm and the
concept's INTERNAL GEOMETRY in the principal coordinates of its contributions. Arm F asks whether
those coordinates are a *control surface*: does moving along them move behaviour monotonically?

  coordinates  contribution cloud of the B*-union over discovery activations → PCA → v₁, v₂ ⊂
               span(B*); σ_k = std of the cloud's projection on v_k
  traversal    x' = x + α·σ_k·v_k on the residual entering layers[L], all positions, α ∈ ±3
  readouts     VQA soft-accuracy; answer-distribution shift (total variation vs α=0)
  controls     (1) matched RANDOM unit directions (3 seeds), same ασ_k scale
               (2) the SAME traversal rotated by the program's Haar rotation, v_k → Q v_k —
                   equal norm, no relation to the handle's span

Frozen readout: monotone dose-response = a strictly ordered |α| → score-drop profile AND a
Spearman |ρ| ≥ 0.8 between α and the answer-distribution shift, with both controls flat
(|ρ| < 0.5). Labelled EXPLORATORY if Arm C did not confirm the handle.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_f_steer --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_f --budget-min 180 --args "\
      --cdisc-s3 s3://…/regrain_c_disc/out --csealed-s3 s3://…/regrain_c_sealed/out \
      --harvest-s3 s3://…/regrain_harvest/out"
"""
import argparse
import contextlib
import json
import os
import pathlib

ALPHAS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--csealed-s3", default=None)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-pca-rows", type=int, default=40000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.inherit_core import _split_hidden

    RS.install_pg()
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    L = int(cd["L"])
    confirmed = None
    if a.csealed_s3:
        cs = json.loads(RS.fetch(f"{a.csealed_s3}/metrics.json",
                                 "/tmp/regrain/c_sealed.json").read_text())
        confirmed = cs.get("verdict")
    label = "CONFIRMATORY-ON-CONFIRMED-HANDLE" if confirmed in ("H_atom-handle",
                                                                "H_atom-handle-nonselective") \
        else "EXPLORATORY (Arm C did not confirm a handle)"
    rep = {"object": "regrain_f_v1", "L": L, "bstar_len": len(bstar), "median_seed": med_s,
           "arm_c_verdict": confirmed, "label": label, "alphas": list(ALPHAS)}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    X, _labs, _st = RS.load_shards_stream(a.harvest_s3, meta, f"L{L}",
                                          workdir="/tmp/regrain/shards")
    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/{nm}"))
    ae = ae.to("cuda").eval()
    Wd = RC.decoder_matrix(ae)
    Db = Wd[:, bstar].float().to("cuda")                     # (d, |B*|) unit columns

    # ---- contribution-PCA coordinates of the B*-union ----
    with torch.inference_mode():
        xs = X[: a.n_pca_rows].to("cuda").float()
        f = ae.encode(xs, use_threshold=True)
        contrib = f[:, bstar] @ Db.T                          # (n, d)
        fired = contrib.abs().sum(1) > 0
        C = contrib[fired].double()
        mu = C.mean(0)
        U, S, Vh = torch.linalg.svd(C - mu, full_matrices=False)
        dirs = Vh[:2]                                         # (2, d) top-2 PCs
        sigmas = (S[:2] / max(1, C.shape[0] - 1) ** 0.5)      # std along each PC
    rep["pca"] = {"n_fired_rows": int(fired.sum()), "sigma": [float(x) for x in sigmas],
                  "spectrum_top8": [float(x) ** 2 / max(1, C.shape[0]) for x in S[:8]],
                  "rank_triplet": RC.rank_triplet((S ** 2 / max(1, C.shape[0])).cpu())}
    _w()
    del xs, f, contrib, C, U, S, Vh
    torch.cuda.empty_cache()

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
    Qrot = RC.haar_rotation(IP.D, RS.ROT_SEED, device="cuda", dtype=torch.float32)

    def add_hook(vec, scale):
        v = (vec / vec.norm()).float().to("cuda") * float(scale)

        def hook(_module, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            return rebuild((hs.float() + v).to(hs.dtype))
        return hook

    @contextlib.contextmanager
    def steer(vec, scale):
        h = layers[L].register_forward_pre_hook(add_hook(vec, scale), with_kwargs=True)
        yield
        h.remove()

    def answers(ctx=None):
        r = IP.eval_pg1(model, proc, vqa, items, "cuda", ctx=ctx)
        return r["score"], np.asarray(r["per_item"], float)

    def tv(p, q):
        """total variation between two answer-correctness profiles (proxy: mean |Δ| per item)."""
        return float(np.abs(np.asarray(p) - np.asarray(q)).mean())

    base_score, base_pi = answers()
    rep["reference"] = {"intact_vqa": base_score, "n_items": len(items)}
    _w()

    def sweep(tag, vec, sigma):
        pts = []
        for al in ALPHAS:
            if al == 0.0:
                s, pi = base_score, base_pi
            else:
                s, pi = answers(steer(vec, al * float(sigma)))
            pts.append({"alpha": al, "vqa": s, "dvqa": s - base_score,
                        "answer_shift_tv": tv(pi, base_pi)})
            rep.setdefault("sweeps", {})[tag] = pts
            _w()
        return pts

    def spearman(xs_, ys_):
        rx = np.argsort(np.argsort(np.asarray(xs_, float)))
        ry = np.argsort(np.argsort(np.asarray(ys_, float)))
        return float(np.corrcoef(rx, ry)[0, 1])

    dirs_cpu = dirs.cpu()
    real = {f"pc{k+1}": sweep(f"pc{k+1}", dirs_cpu[k], sigmas[k]) for k in range(2)}
    rot = sweep("pc1_rotated", (Qrot.double().cpu() @ dirs_cpu[0]), sigmas[0])
    rand = {}
    for s in range(3):
        g = torch.Generator().manual_seed(900 + s)
        v = torch.randn(IP.D, generator=g, dtype=torch.float64)
        rand[f"rand{s}"] = sweep(f"rand{s}", v, sigmas[0])

    def profile(pts):
        al = [p["alpha"] for p in pts]
        return {"spearman_alpha_vs_shift": spearman(al, [p["answer_shift_tv"] for p in pts]),
                "spearman_absalpha_vs_shift": spearman([abs(x) for x in al],
                                                       [p["answer_shift_tv"] for p in pts]),
                "max_dvqa": min(p["dvqa"] for p in pts),
                "max_shift": max(p["answer_shift_tv"] for p in pts)}

    rep["profiles"] = {"pc1": profile(real["pc1"]), "pc2": profile(real["pc2"]),
                       "pc1_rotated": profile(rot),
                       **{k: profile(v) for k, v in rand.items()}}
    pc1 = rep["profiles"]["pc1"]["spearman_absalpha_vs_shift"]
    ctrl = max(abs(rep["profiles"][k]["spearman_absalpha_vs_shift"])
               for k in ("pc1_rotated", "rand0", "rand1", "rand2"))
    rep["dose_response"] = {
        "rule": "monotone iff |rho(|alpha|, shift)| >= 0.8 on pc1 AND all controls |rho| < 0.5",
        "pc1_rho": pc1, "max_control_rho": ctrl,
        "monotone": bool(abs(pc1) >= 0.8 and ctrl < 0.5), "label": label}
    print("REGRAIN_F_OK", json.dumps({"label": label, "pc1_rho": pc1, "max_ctrl_rho": ctrl,
                                      "monotone": rep["dose_response"]["monotone"],
                                      "max_dvqa_pc1": rep["profiles"]["pc1"]["max_dvqa"]},
                                     default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
