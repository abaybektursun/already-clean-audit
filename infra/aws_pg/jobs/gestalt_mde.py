"""GESTALT MDE — Arm S at one site: the K-vs-fidelity curve, real vs moment-matched twin.
Spec §1.2/§2 Arm S, §3; journal FR-1/FR-4/FR-8 + the P3 realization note.

Order (each step gated, incremental metrics.json from the first write):
  0. Pull the banked harvest (gestalt_harvest) for --site; load acts + moments + item manifest.
  1. **P1-at-scale** (d=2048): planted circle / two-cluster / curved-manifold checks at the real
     dimensionality with the frozen rank — the instrument must still detect plants here.
  2. Twin generation from the banked moments (fp64 sym-sqrt; spec sha'd) + **P2-at-scale**
     (twin-of-twin fit consistency at K=8).
  3. Fits: {real, twin} x K_GRID x EM_SEEDS on the frozen fit subsample; every atlas banked
     (npz + sha); VOID cells recorded per the FR-4 divergence rule.
  4. Shell + ppl anchor; fidelity items re-loaded and hash-asserted against the harvest bank.
  5. **P3 hook gate**: the IDENTITY atlas (r=d — the exact K->infinity limit) swap must
     reproduce the unswapped battery score (|delta| <= 0.005); a failure is a HOOK bug -> halt.
     The K=64 real fit's swap score is banked as the family's achievable CEILING (the fidelity
     gate calibration of spec §8), reported, never a halt.
  6. The sweep: unswapped baseline + every atlas swapped in -> per-item VQAv2 soft-acc banked.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_mde \
      --args "--harvest-prefix results/gestalt_harvest_pg1 --site 9 --rank 4" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/gestalt_mde_s9 --budget-min 420

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

N_FIDELITY = 800


def _pull(bucket, prefix, dest):
    """`prefix` = FULL path under the bucket to a harvest OUT dir (no /out appended)."""
    subprocess.run(["aws", "s3", "sync", f"s3://{bucket}/{prefix}", str(dest),
                    "--region", "us-east-1", "--only-show-errors"], check=True)


def p1_at_scale(GS, r, d, device, rep):
    """P1 at the real dimensionality. Plant noise is scaled by sqrt(32/d) so the TOTAL
    off-structure noise energy matches the d=32 regime the offline gate proved detectable —
    at fixed per-dim noise the plant drowns (noise radius 0.15*sqrt(2048)≈6.8 vs ring radius 6;
    defect D10, confirmed from the mde_debug artifact: all three detections failed with
    two-cluster LL-gain 0.041 vs twin 0.089)."""
    import torch
    n = 60_000
    ns = (32.0 / d) ** 0.5
    res = {}
    X = GS.make_circle(n, d, seed=1, noise=0.15 * ns).to(device)
    m, C = GS.stream_moments(X)
    L = GS.cov_sym_sqrt(C)
    a = GS.fit_mppca(X, K=8, r=r, seed=0)
    tw = GS.twin_sample(m, L, n, seed=100, device=device)
    at = GS.fit_mppca(tw, K=8, r=r, seed=0)
    assert not a["void"] and not at["void"]
    res["circle"] = {"real_cv": GS.circle_stat(a, X), "twin_cv": GS.circle_stat(at, X)}
    res["circle"]["detected"] = bool(res["circle"]["real_cv"] < 0.5 * res["circle"]["twin_cv"])
    X2, lab = GS.make_two_cluster(n, d, seed=2, noise=1.0 * ns)
    X2 = X2.to(device)
    m2, C2 = GS.stream_moments(X2)
    L2 = GS.cov_sym_sqrt(C2)
    g_r = (GS.ll_per_point(GS.fit_mppca(X2, K=2, r=r, seed=0), X2)
           - GS.ll_per_point(GS.fit_mppca(X2, K=1, r=r, seed=0), X2))
    tw2 = GS.twin_sample(m2, L2, n, seed=200, device=device)
    g_t = (GS.ll_per_point(GS.fit_mppca(tw2, K=2, r=r, seed=0), tw2)
           - GS.ll_per_point(GS.fit_mppca(tw2, K=1, r=r, seed=0), tw2))
    res["two_cluster"] = {"ll_gain_real": g_r, "ll_gain_twin": g_t,
                          "detected": bool(g_r > 2.0 * g_t)}
    # D19 (supersedes D18; per-plant noise policy): the D10 energy scaling applies where noise
    # energy would BURY the structure (circle var 36 and cluster var 16 vs unscaled noise 46 —
    # buried; scaled). The manifold's structure variance (~56 incl. z-var 32) EXCEEDS the
    # unscaled noise energy, so it keeps sigma=0.15/curv=0.35 — which also keeps the local
    # sagitta/sigma ratio in the rank-2-fittable regime. D18's curv*ns flattening shrank the
    # twin's z-range to ~1.06, making the 0.5x bar unreachable for ANY fit (read off p1gpu3:
    # real 1.19 vs twin 1.12). The d=32 gate is unaffected (all factors are 1 there).
    X3 = GS.make_curved_manifold(n, d, seed=3).to(device)
    m3, C3 = GS.stream_moments(X3)
    L3 = GS.cov_sym_sqrt(C3)
    dr = GS.manifold_dist_stat(GS.fit_mppca(X3, K=16, r=r, seed=0), 0.35, 3, d)
    tw3 = GS.twin_sample(m3, L3, n, seed=300, device=device)
    dt = GS.manifold_dist_stat(GS.fit_mppca(tw3, K=16, r=r, seed=0), 0.35, 3, d)
    res["curved_manifold"] = {"real_dist": dr, "twin_dist": dt, "detected": bool(dr < 0.5 * dt)}
    rep["P1_at_scale"] = res
    ok = all(res[k]["detected"] for k in res)
    assert ok, f"P1-at-scale FAILED: {json.dumps(res)}"
    print(f"[MDE] P1-at-scale PASS: {json.dumps(res)}", flush=True)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-prefix", default="",
                    help="S3 results prefix of a gestalt_harvest run (pulled to /tmp)")
    ap.add_argument("--harvest-local", default="",
                    help="local harvest out-dir (pipeline mode: same instance, no S3 pull)")
    ap.add_argument("--site", type=int, default=9)
    ap.add_argument("--rank", type=int, required=True,
                    help="the offline gate's r_frozen (provenance: gestalt_offline metrics.json)")
    ap.add_argument("--k-grid", default="", help="smoke override, e.g. '1,8' (default = frozen)")
    ap.add_argument("--seeds", default="", help="smoke override, e.g. '0'")
    ap.add_argument("--n-fidelity", type=int, default=N_FIDELITY)
    ap.add_argument("--n-fit", type=int, default=0, help="smoke override of GS.N_FIT")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs._inherit_shared import install_pg
    install_pg()               # pycocoevalcap (battery deps) is not in the AMI
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg import config as C
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.aws_pg.jobs.lacuna_ladder import PPL_ANCHOR, PPL_REL_TOL
    from infra.pg import gestalt_splat as GS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS

    assert torch.cuda.is_available(), "gestalt_mde needs CUDA"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    k_grid = tuple(int(x) for x in a.k_grid.split(",")) if a.k_grid else GS.K_GRID
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else GS.EM_SEEDS
    n_fit = a.n_fit or GS.N_FIT
    assert bool(a.harvest_prefix) != bool(a.harvest_local), \
        "exactly one of --harvest-prefix / --harvest-local"
    rep = {"object": "GESTALT MDE (Arm S)", "site": a.site, "rank": a.rank,
           "k_grid": list(k_grid), "seeds": list(seeds), "n_fit": n_fit,
           "harvest_prefix": a.harvest_prefix or None,
           "harvest_local": a.harvest_local or None,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells                     # bound BEFORE the loops (fleet lesson)
    banks = {}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        (out / "eval_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- 0. the harvest bank (local in pipeline mode, else pulled from S3) -------------------
    if a.harvest_local:
        hdir = pathlib.Path(a.harvest_local)
        assert (hdir / "item_hashes.json").exists(), f"no harvest bank at {hdir}"
    else:
        hdir = pathlib.Path("/tmp/gestalt_bank")
        _pull(C.BUCKET, a.harvest_prefix, hdir)
    site_dir = hdir / f"site{a.site}"
    shards = sorted(site_dir.glob("acts_*.npy"))
    assert shards, f"no activation shards under {site_dir}"
    X_real = torch.cat([torch.from_numpy(np.load(p)).to(torch.float32) for p in shards], 0)
    mz = np.load(site_dir / "moments.npz")
    mean = torch.from_numpy(mz["mean"])
    cov = torch.from_numpy(mz["cov"])
    n_real, d = X_real.shape
    man = json.loads((hdir / "item_hashes.json").read_text())
    hman = json.loads((hdir / "metrics.json").read_text())
    rep["bank"] = {"n_tokens": int(n_real), "d": int(d),
                   "harvest_unit": hman["unit"],
                   "harvest_gpu": hman["gpu"],
                   "mean_sha": hashlib.sha256(mz["mean"].tobytes()).hexdigest()[:16],
                   "cov_sha": hashlib.sha256(mz["cov"].tobytes()).hexdigest()[:16]}
    assert rep["bank"]["mean_sha"] == hman[f"site{a.site}"]["mean_sha"], "moments sha mismatch vs harvest"
    print(f"[MDE] bank: {n_real} tokens d={d}", flush=True)
    # X_real stays on CPU: only the frozen fit subsample ever goes to GPU (peak ~0.8GB instead
    # of ~21GB — makes 24GB pools viable when the 48/80GB pools are capacity-dry)
    _w()

    # ---- 1. P1 at scale ---------------------------------------------------------------------
    p1_at_scale(GS, a.rank, d, device, rep)
    _w()

    # ---- 2. twin + P2 at scale (D20: MOMENT reproduction, fit-free) -------------------------
    # The LL-based twin-of-twin consistency leg is a d=32 offline-gate responsibility: at
    # d=2048 the per-point LL's sigma2-jitter noise floor (~2 nats/pt, the D14 mechanism)
    # exceeds any signal the check could carry (observed gap 2.246 = exactly that scale).
    # Twin-generator health at scale = the generated sample reproduces the TARGET moments,
    # measured exactly as the offline P2 does (RMS-normalized mean err + relative cov err).
    Lc = GS.cov_sym_sqrt(cov)
    tspec = GS.twin_spec(mean, cov, n_real, seed=0)
    rep["twin_spec"] = tspec
    tw_chk = GS.twin_sample(mean, Lc, 200_000, seed=44, device=device)
    m2, C2 = GS.stream_moments(tw_chk)
    scale = float(torch.sqrt(torch.trace(cov)))
    mean_err = float((m2.cpu() - mean).norm()) / scale
    cov_err = float((C2.cpu() - cov).norm() / cov.norm().clamp(min=1e-9))
    rep["P2_at_scale"] = {"mean_rel_err": mean_err, "cov_rel_err": cov_err,
                          "passes": bool(mean_err < 0.05 and cov_err < 0.05)}
    assert rep["P2_at_scale"]["passes"], f"P2-at-scale twin moments off: {rep['P2_at_scale']}"
    del tw_chk
    torch.cuda.empty_cache()
    print(f"[MDE] P2-at-scale PASS mean_err={mean_err:.4f} cov_err={cov_err:.4f}", flush=True)
    _w()

    # ---- 3. fits ----------------------------------------------------------------------------
    fits_dir = out / "atlases"
    fits_dir.mkdir(exist_ok=True)
    atlases = {}
    X_fit_real = GS.fit_subsample(X_real, seed=0, n_fit=n_fit).to(device)
    # the twin fit sample is drawn at n_fit directly: subsampling an iid Gaussian draw of
    # n_real IS an iid draw of n_fit (distributionally identical; the seed labels the draw)
    X_fit_twin = GS.twin_sample(mean, Lc, min(n_fit, n_real), seed=0, device=device)
    for src_tag, X_fit in (("real", X_fit_real), ("twin", X_fit_twin)):
        for K in k_grid:
            for s in seeds:
                t1 = time.time()
                atlas = GS.fit_mppca(X_fit, K=K, r=a.rank, seed=s)
                key = f"{src_tag}|K{K}|s{s}"
                if atlas["void"]:
                    cells.append({"kind": "fit", "key": key, "void": True,
                                  "void_reason": atlas["void_reason"]})
                    _w()
                    print(f"[MDE] fit {key} VOID: {atlas['void_reason']}", flush=True)
                    continue
                cpu = GS.atlas_to_cpu(atlas)
                sha = GS.atlas_sha(cpu)
                np.savez(fits_dir / f"{src_tag}_K{K}_s{s}.npz",
                         **{k: cpu[k].numpy() for k in ("log_pi", "mu", "U", "lam", "sigma2")})
                atlases[key] = atlas
                cells.append({"kind": "fit", "key": key, "void": False, "sha": sha,
                              "n_iters": atlas["n_iters"], "ll_final": atlas["ll_trace"][-1],
                              "collapsed": atlas["collapsed"],
                              "fit_s": round(time.time() - t1, 1)})
                _w()
                print(f"[MDE] fit {key} sha={sha} iters={atlas['n_iters']} "
                      f"({time.time() - t1:.0f}s)", flush=True)
    del X_fit_real, X_fit_twin, X_real
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 4. shell + fidelity items ----------------------------------------------------------
    model, proc, layers, sha_pin, _ppl_pin = SS.load_model("/tmp/gestalt_mde")
    ppl = IP.ppl_gate(model, proc, device)
    val = ppl["ppl"] if isinstance(ppl, dict) else float(ppl)
    rel = abs(float(val) - PPL_ANCHOR) / PPL_ANCHOR
    rep["ppl_gate"] = {"ppl": float(val), "rel": rel, "passes": rel < PPL_REL_TOL}
    assert rel < PPL_REL_TOL, f"MDE shell ppl anchor FAIL {val}"
    tasks = LS.mm_tasks()
    vqa = tasks["vqav2"]
    n_load = 200 if a.n_fidelity < N_FIDELITY else 4000     # smoke matches the harvest's cap
    rows, hashes = IP.load_items(vqa, n_load)
    disc, _conf = LC.parity_split(hashes)
    d_rows = [r for r, m_ in zip(rows, disc) if m_]
    d_hashes = [h for h, m_ in zip(hashes, disc) if m_]
    fid_rows = d_rows[: a.n_fidelity]
    fid_hashes = d_hashes[: a.n_fidelity]
    want = man["fidelity_hashes"][: a.n_fidelity]
    assert fid_hashes == want, "fidelity item hashes DRIFTED vs the harvest bank — not the frozen list"
    rep["fidelity"] = {"n": len(fid_rows),
                       "sha": hashlib.sha256("".join(fid_hashes).encode()).hexdigest()[:16]}
    _w()

    # ---- 5+6. the sweep ---------------------------------------------------------------------
    site_layer = layers[a.site]

    def run_eval(tag, ctx):
        t1 = time.time()
        r = IP.eval_pg1(model, proc, vqa, fid_rows, device, ctx=ctx, batch_size=16)
        banks[tag] = {"per_item": r["per_item"]}
        cells.append({"kind": "eval", "key": tag, "score": r["score"], "ci95": r["ci95"],
                      "n": r["n"], "eval_s": round(time.time() - t1, 1)})
        _w()
        print(f"[MDE] eval {tag:>22s} score={r['score']:.4f} ({time.time() - t1:.0f}s)",
              flush=True)
        return r

    base = run_eval("unswapped", None)

    # P3: identity atlas (r=d) — the exact K->inf limit; halts on failure (hook bug)
    eye = torch.eye(d, device=device)
    ident = {"log_pi": torch.zeros(1, device=device),
             "mu": mean.to(torch.float32).to(device).unsqueeze(0),
             "U": eye.unsqueeze(0), "lam": torch.ones(1, d, device=device),
             "sigma2": torch.tensor([1.0], device=device), "void": False}
    p3 = run_eval("P3_identity", GS.SplatSwapHook(site_layer, ident))
    p3_delta = p3["score"] - base["score"]
    rep["P3_gate"] = {"delta": p3_delta, "passes": bool(abs(p3_delta) <= 0.005)}
    assert abs(p3_delta) <= 0.005, f"P3 HOOK GATE FAILED: identity swap moved score by {p3_delta}"
    print(f"[MDE] P3 identity-swap PASS delta={p3_delta:+.4f}", flush=True)
    _w()

    for key, atlas in atlases.items():
        for k in ("log_pi", "mu", "U", "lam", "sigma2"):
            atlas[k] = atlas[k].to(device)
        run_eval(key, GS.SplatSwapHook(site_layer, atlas))

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("GESTALT_MDE_OK", json.dumps({c["key"]: c.get("score") for c in cells
                                        if c["kind"] == "eval"}), flush=True)


if __name__ == "__main__":
    main()
    # D7: datasets/cuda extension threads abort the interpreter during finalization
    # (PyGILState_Release fatal, rc=-6) AFTER all artifacts are banked; every write is
    # explicit and already streamed, so skip finalization outright.
    import os as _os
    _os._exit(0)
