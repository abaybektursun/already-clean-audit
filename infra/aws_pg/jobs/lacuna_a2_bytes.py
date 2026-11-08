"""LACUNA Arm 2 (a/b/c/e) — the cached-weights census. CPU only; runs FIRST (spec §9 step 3).

Sub-arms (spec §2.2):
  2a  vision tower π0 / π0.5 vs PG1 — **byte test in stored dtype** (max|Δ|), then per-matrix
      ‖Δ‖/‖W‖, stable rank, top-k principal-subspace overlap vs the analytic matched-rotation null.
      CONFIRMATORY: H_vision-frozen (max|Δ| = 0) → terminal for the sub-arm.
  2b  PG1 vision vs the SigLIP-so400m release — the SAME profile, but **assumption-gated
      exploratory** (release == PG1's init is not byte-verifiable) — labelled in every record.
  2c  gemma-2b → PG1 LM ΔW: stable rank / depth × matrix-type profile vs a Frobenius-matched
      random-ΔW band. CONFIRMATORY: H_fullrank-A.
  2e  raw-ΔW census π0−PG1, π0.5−PG1, π0.5−π0 (srank, depth, RAW out-of-span) — exploratory,
      every out-of-span number tagged RAW (`reference_fperp_two_objects`: never comparable to an
      install-gradient f_perp).
  2f  (LACUNA addition) embedding-table census: the policies' tied head vs PG1's embedding rows.
      This is what decides whether the ladder's own-embed and PG1-embed rows CAN differ; it costs
      one matrix subtraction and it was unmeasured.

Everything reads RAW safetensors (never a model-loaded, dtype-cast copy) so "stored dtype" means
what it says.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_a2_bytes --type r7i.4xlarge --no-weights \
      --s3-prefix results/lacuna_a2_bytes --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
SIGLIP_REPO = "google/siglip-so400m-patch14-224"
POLICY_REPOS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}
PALI_PREFIX = "paligemma_with_expert.paligemma."
TOPK = 32                 # top-k subspace for the overlap statistic (fixed here, BEFORE any real Δ)
NULL_DRAWS = 64
SRANK_DRAWS = 32


def _raw_sd(repo, token_ok=True):
    """Every tensor of a repo's safetensors, in STORED dtype (no model instantiation)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(x for x in list_repo_files(repo) if x.endswith(".safetensors")):
        sd.update(load_file(hf_hub_download(repo, f)))
    assert sd, f"{repo}: no safetensors tensors"
    return sd


def canon(k):
    return ".".join(p for p in k.split(".") if p != "model")


def _top_left(M, k):
    """Top-k LEFT singular vectors of M (out, in) via the SMALLER Gram — identical to
    `svd(M)[0][:, :k]` up to sign, but eigendecomposes a 2048² matrix instead of factorising a
    16384×2048 one. The LM's MLP matrices make the difference between minutes and an hour."""
    import numpy as np
    A = np.asarray(M, dtype=np.float64)
    out_d, in_d = A.shape
    if in_d <= out_d:
        w, V = np.linalg.eigh(A.T @ A)
        idx = np.argsort(w)[::-1][:k]
        s = np.sqrt(np.clip(w[idx], 1e-300, None))
        return (A @ V[:, idx]) / s[None, :], float(np.sqrt(max(w.max(), 0.0)))
    w, U = np.linalg.eigh(A @ A.T)
    idx = np.argsort(w)[::-1][:k]
    return U[:, idx], float(np.sqrt(max(w.max(), 0.0)))


def _sigma1(M):
    """Largest singular value via the smaller Gram (all `stable_rank` needs, since
    srank = ‖M‖²_F / σ₁² — no full spectrum is required)."""
    import numpy as np
    A = np.asarray(M, dtype=np.float64)
    G = (A.T @ A) if A.shape[1] <= A.shape[0] else (A @ A.T)
    return float(np.sqrt(max(np.linalg.eigvalsh(G).max(), 0.0)))


def _profile_pair(a, b, topk=TOPK, null_draws=NULL_DRAWS, seed=0):
    """Per-matrix comparison of two same-shape tensors in fp64: byte test, relative Frobenius,
    stable ranks, and top-k left/right subspace overlap vs the matched-rotation null."""
    import numpy as np
    from infra.pg import lacuna_core as LC
    A = a.detach().to("cpu").double().numpy()
    B = b.detach().to("cpu").double().numpy()
    assert A.shape == B.shape, f"shape mismatch {A.shape} vs {B.shape}"
    d = B - A
    out = {"shape": list(A.shape), "max_abs_delta": float(np.abs(d).max()),
           "rel_frob": float(np.linalg.norm(d) / max(np.linalg.norm(A), 1e-300))}
    if A.ndim != 2 or min(A.shape) < 4:
        return out
    k = int(min(topk, min(A.shape) - 1))
    Ua, s1a = _top_left(A, k)
    Ub, s1b = _top_left(B, k)
    Va, _ = _top_left(A.T, k)
    Vb, _ = _top_left(B.T, k)
    out["stable_rank_a"] = float((A ** 2).sum() / s1a ** 2)
    out["stable_rank_b"] = float((B ** 2).sum() / s1b ** 2)
    out["k"] = k
    out["left"] = LC.overlap_vs_rotation_null(Ua, Ub, n_draw=null_draws, seed=seed)
    out["right"] = LC.overlap_vs_rotation_null(Va, Vb, n_draw=null_draws, seed=seed + 1)
    if np.linalg.norm(d) > 0:
        # RAW out-of-span: energy of ΔW outside the top-k LEFT singular span of the BASE matrix.
        P = Ua
        out["raw_out_of_span_topk"] = float(
            (np.linalg.norm(d - P @ (P.T @ d)) ** 2) / (np.linalg.norm(d) ** 2))
    return out


def _delta_census(base, upd, tag, topk=TOPK, seed=0):
    """srank / depth / RAW-out-of-span census of ΔW = upd − base over shared 2-D keys."""
    import numpy as np
    from infra.pg import lacuna_core as LC
    rows = []
    shared = sorted(k for k in base if k in upd and base[k].ndim == 2)
    for k in shared:
        A = base[k].detach().double().numpy()
        B = upd[k].detach().double().numpy()
        d = B - A
        nd = float(np.linalg.norm(d))
        if nd == 0:
            rows.append({"key": k, "shape": list(A.shape), "rel_frob": 0.0, "srank_delta": None,
                         "raw_out_of_span": None})
            continue
        s1_d = _sigma1(d)
        kk = int(min(topk, min(A.shape) - 1))
        P, s1_a = _top_left(A, kk)
        s1_b = _sigma1(B)
        rows.append({"key": k, "shape": list(A.shape),
                     "rel_frob": nd / max(float(np.linalg.norm(A)), 1e-300),
                     "srank_delta": float((nd ** 2) / (s1_d ** 2)),
                     # INTERPRETIVE CONTROL (added 2026-07-28, see the census note below): the
                     # stable rank of the BASE and UPDATED matrices themselves, on the same band.
                     "srank_base": float((A ** 2).sum() / max(s1_a ** 2, 1e-300)),
                     "srank_upd": float((B ** 2).sum() / max(s1_b ** 2, 1e-300)),
                     "srank_null": LC.srank_random_band(A.shape, n_draws=SRANK_DRAWS, seed=seed),
                     "raw_out_of_span": float((np.linalg.norm(d - P @ (P.T @ d)) ** 2) / nd ** 2),
                     "k_span": kk})
    live = [r for r in rows if r["srank_delta"] is not None]
    med = float(np.median([r["srank_delta"] for r in live])) if live else None
    band_lo = float(np.median([r["srank_null"]["p2_5"] for r in live])) if live else None
    band_hi = float(np.median([r["srank_null"]["p97_5"] for r in live])) if live else None
    med_base = float(np.median([r["srank_base"] for r in live])) if live else None
    med_upd = float(np.median([r["srank_upd"] for r in live])) if live else None
    # Is "below the Gaussian band" a property of the UPDATE, or of trained matrices in general?
    # A Gaussian 2048² has stable rank ≈512 while any trained matrix is spectrally concentrated,
    # so the frozen band is cleared by essentially any real ΔW. The ratio below is the
    # interpretive control: srank(ΔW) / srank(W_base) ≈ 1 means the update is no more concentrated
    # than the weights it edits, and "low-rank" is then a statement about trained matrices, not
    # about the install. Reported BESIDE the frozen verdict, which stands as computed.
    ratio = (None if not med_base else float(med / med_base))
    return {"tag": tag, "n_matrices": len(rows), "n_nonzero": len(live),
            "median_srank_delta": med, "median_null_p2_5": band_lo, "median_null_p97_5": band_hi,
            "median_srank_base": med_base, "median_srank_updated": med_upd,
            "srank_delta_over_base": ratio,
            "n_base_below_band": int(sum(1 for r in live if r["srank_base"] < r["srank_null"]["p2_5"])),
            "n_delta_below_band": int(sum(1 for r in live if r["srank_delta"] < r["srank_null"]["p2_5"])),
            "fullrank_like": (None if med is None else bool(band_lo <= med <= band_hi)),
            "below_band": (None if med is None else bool(med < band_lo)),
            "median_rel_frob": float(np.median([r["rel_frob"] for r in rows])) if rows else None,
            "median_raw_out_of_span": (float(np.median([r["raw_out_of_span"] for r in live]))
                                       if live else None),
            "out_of_span_object": "RAW-ΔW (NEVER comparable to an install-gradient f_perp)",
            "per_matrix": rows}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subarms", default="2a,2b,2f,2c,2e")   # cheap+decisive before expensive
    ap.add_argument("--topk", type=int, default=TOPK)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = {s.strip() for s in a.subarms.split(",") if s.strip()}

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    t0 = time.time()
    rep = {"object": "lacuna_a2_bytes_v1", "subarms": sorted(want), "topk": a.topk,
           "torch": torch.__version__, "gpu_seconds": {"total": 0.0, "note": "CPU job"}}
    res = {}
    rep["results"] = res

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    print("[A2] loading PG1 raw safetensors ...", flush=True)
    pg1 = {canon(k): v for k, v in _raw_sd(PG1_REPO).items()}
    rep["pg1_keys"] = len(pg1)
    _w()

    pol_vision, pol_lm, pol_head = {}, {}, {}
    if want & {"2a", "2e", "2f"}:
        for nm, repo in POLICY_REPOS.items():
            print(f"[A2] loading {nm} raw safetensors ...", flush=True)
            sd = _raw_sd(repo)
            pali = {canon(k[len(PALI_PREFIX):]): v for k, v in sd.items() if k.startswith(PALI_PREFIX)}
            pol_vision[nm] = {k: v for k, v in pali.items() if k.startswith("vision_tower.")}
            pol_lm[nm] = {k[len("language_model."):]: v for k, v in pali.items()
                          if k.startswith("language_model.")}
            pol_head[nm] = pali["lm_head.weight"]
            pol_vision[nm]["__projector__"] = pali["multi_modal_projector.linear.weight"]
            del sd, pali
            print(f"[A2] {nm}: vision={len(pol_vision[nm])-1} lm={len(pol_lm[nm])} "
                  f"head={list(pol_head[nm].shape)}", flush=True)

    # ---- 2a — the byte test + vision profile (the curiosity-first free shot) -------------
    if "2a" in want:
        res["2a"] = {}
        pg1_vis = {k: v for k, v in pg1.items() if k.startswith("vision_tower.")}
        pg1_vis["__projector__"] = pg1["multi_modal_projector.linear.weight"]
        for nm in POLICY_REPOS:
            shared = sorted(set(pg1_vis) & set(pol_vision[nm]))
            assert len(shared) == len(pg1_vis), \
                f"{nm}: vision key mismatch — {len(shared)} shared of {len(pg1_vis)}"
            maxes = {k: float((pol_vision[nm][k].double() - pg1_vis[k].double()).abs().max())
                     for k in shared}
            gmax = max(maxes.values())
            block = {"n_keys": len(shared), "global_max_abs_delta": gmax,
                     "byte_frozen": bool(gmax == 0.0),
                     "dtypes": {"pg1": str(pg1_vis[shared[0]].dtype),
                                "policy": str(pol_vision[nm][shared[0]].dtype)},
                     "top_changed": sorted(maxes.items(), key=lambda kv: -kv[1])[:8]}
            print(f"[A2] 2a {nm} vs PG1 vision: max|Δ|={gmax:.6g} frozen={block['byte_frozen']}",
                  flush=True)
            if not block["byte_frozen"]:
                prof = {k: _profile_pair(pg1_vis[k], pol_vision[nm][k], topk=a.topk)
                        for k in shared if pg1_vis[k].ndim == 2 and min(pg1_vis[k].shape) >= 8}
                ovl = [p["left"]["overlap"] for p in prof.values()]
                nul = [p["left"]["null_p95"] for p in prof.values()]
                block["profile"] = prof
                block["median_left_overlap"] = float(np.median(ovl))
                block["median_null_p95"] = float(np.median(nul))
                block["preserved"] = bool(np.median(ovl) > np.median(nul))
                block["median_rel_frob"] = float(np.median([p["rel_frob"] for p in prof.values()]))
                print(f"[A2] 2a {nm} overlap med={block['median_left_overlap']:.4f} "
                      f"null_p95 med={block['median_null_p95']:.4f} "
                      f"preserved={block['preserved']}", flush=True)
            res["2a"][nm] = block
            _w()

    # ---- 2b — PG1 vision vs the SigLIP release (ASSUMPTION-GATED EXPLORATORY) ------------
    if "2b" in want:
        print("[A2] loading SigLIP release ...", flush=True)
        sig = {canon(k): v for k, v in _raw_sd(SIGLIP_REPO).items()}
        pairs = {}
        for k, v in pg1.items():
            if not k.startswith("vision_tower."):
                continue
            s = k[len("vision_tower."):]
            if s in sig and tuple(sig[s].shape) == tuple(v.shape):
                pairs[k] = s
        prof = {k: _profile_pair(sig[s], pg1[k], topk=a.topk)
                for k, s in pairs.items() if pg1[k].ndim == 2 and min(pg1[k].shape) >= 8}
        gmax = max((float((pg1[k].double() - sig[s].double()).abs().max())
                    for k, s in pairs.items()), default=None)
        res["2b"] = {"label": "ASSUMPTION-GATED EXPLORATORY — the release is assumed to be PG1's "
                              "vision init; that identity is NOT byte-verifiable",
                     "n_matched_keys": len(pairs), "n_pg1_vision_keys":
                         sum(1 for k in pg1 if k.startswith("vision_tower.")),
                     "global_max_abs_delta": gmax,
                     "median_rel_frob": float(np.median([p["rel_frob"] for p in prof.values()])),
                     "median_left_overlap": float(np.median([p["left"]["overlap"] for p in prof.values()])),
                     "median_null_p95": float(np.median([p["left"]["null_p95"] for p in prof.values()])),
                     "per_matrix": prof}
        res["2b"]["preserved"] = bool(res["2b"]["median_left_overlap"] > res["2b"]["median_null_p95"])
        print(f"[A2] 2b PG1-vs-SigLIP: relF med={res['2b']['median_rel_frob']:.4f} "
              f"overlap med={res['2b']['median_left_overlap']:.4f}", flush=True)
        del sig
        _w()

    # ---- 2c — the gemma-2b → PG1 LM ΔW (CONFIRMATORY: H_fullrank-A) ----------------------
    if "2c" in want:
        print("[A2] loading gemma-2b raw safetensors ...", flush=True)
        g1 = {canon(k): v for k, v in _raw_sd(GEMMA1_REPO).items()}
        base_lm = {k: v for k, v in g1.items() if k.startswith("layers.") or k == "norm.weight"}
        pg1_lm = {k[len("language_model."):]: v for k, v in pg1.items()
                  if k.startswith("language_model.") and "embed_tokens" not in k}
        shared = sorted(set(base_lm) & set(pg1_lm))
        assert len(shared) >= 126, f"only {len(shared)} shared LM keys gemma-2b↔PG1"
        res["2c"] = _delta_census(base_lm, pg1_lm, "gemma2b→PG1 LM", topk=a.topk)
        res["2c"]["embed_compare"] = {
            "gemma_rows": list(g1["embed_tokens.weight"].shape),
            "pg1_rows": list(pg1["language_model.embed_tokens.weight"].shape),
            "max_abs_delta_shared_rows": float(
                (pg1["language_model.embed_tokens.weight"][:g1["embed_tokens.weight"].shape[0]].double()
                 - g1["embed_tokens.weight"].double()).abs().max())}
        print(f"[A2] 2c median srank(ΔW)={res['2c']['median_srank_delta']} "
              f"band=[{res['2c']['median_null_p2_5']},{res['2c']['median_null_p97_5']}] "
              f"fullrank_like={res['2c']['fullrank_like']}", flush=True)
        del g1
        _w()

    # ---- 2e — the VLA-hop raw-ΔW census (EXPLORATORY, RAW) -------------------------------
    if "2e" in want:
        pg1_lm = {k[len("language_model."):]: v for k, v in pg1.items()
                  if k.startswith("language_model.") and "embed_tokens" not in k}
        res["2e"] = {}
        for tag, base, upd in (("pi0−PG1", pg1_lm, pol_lm["pi0"]),
                               ("pi05−PG1", pg1_lm, pol_lm["pi05"]),
                               ("pi05−pi0", pol_lm["pi0"], pol_lm["pi05"])):
            res["2e"][tag] = _delta_census(base, upd, tag, topk=a.topk)
            res["2e"][tag]["label"] = "EXPLORATORY census"
            print(f"[A2] 2e {tag}: relF med={res['2e'][tag]['median_rel_frob']:.4f} "
                  f"srank med={res['2e'][tag]['median_srank_delta']} "
                  f"raw_oos med={res['2e'][tag]['median_raw_out_of_span']}", flush=True)
            _w()

    # ---- 2f — the embedding-table census (LACUNA addition) -------------------------------
    # Runs BEFORE 2c/2e: it is one matrix subtraction and it decides whether the ladder's
    # own-embed and PG1-embed rows can differ at all, while 2e is the expensive exploratory tail.
    if "2f" in want:
        emb = pg1["language_model.embed_tokens.weight"].double()
        res["2f"] = {"pg1_embed_rows": list(emb.shape)}
        for nm in POLICY_REPOS:
            h = pol_head[nm].double()
            n = h.shape[0]
            d = h - emb[:n]
            res["2f"][nm] = {"own_head_rows": list(h.shape),
                             "max_abs_delta_vs_pg1": float(d.abs().max()),
                             "rel_frob_vs_pg1": float(d.norm() / emb[:n].norm()),
                             "byte_identical": bool(float(d.abs().max()) == 0.0),
                             "frac_rows_changed": float((d.abs().sum(dim=1) > 0).double().mean())}
            print(f"[A2] 2f {nm} own-head vs PG1 embed: max|Δ|="
                  f"{res['2f'][nm]['max_abs_delta_vs_pg1']:.6g} "
                  f"relF={res['2f'][nm]['rel_frob_vs_pg1']:.4f} "
                  f"rows_changed={res['2f'][nm]['frac_rows_changed']:.4f}", flush=True)
        d05 = (pol_head["pi05"].double() - pol_head["pi0"].double())
        res["2f"]["pi05_vs_pi0"] = {"max_abs_delta": float(d05.abs().max()),
                                    "rel_frob": float(d05.norm() / pol_head["pi0"].double().norm())}
        _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_A2_BYTES_OK", json.dumps(
        {k: (v if not isinstance(v, dict) else
             {kk: vv for kk, vv in v.items() if kk not in ("profile", "per_matrix")})
         for k, v in res.items()}, default=str)[:2500], flush=True)


if __name__ == "__main__":
    main()
