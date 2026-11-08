"""GAUGE-RMT (P3) — bulk/outlier decomposition of the 24-FT PG1 ΔW spectrum (CPU, --no-weights).

Streams the 24 fine-tunes exactly as invariant_core.py / gauge_cs.py do (enumerate -224 live, pin
SHAs, load base once, difference per decoder matrix in fp64, fp16 scratch, delete cache). Per
(matrix-type, layer) MATRIX and per FT it runs the RMT decomposition (gauge_rmt.analyze_matrix): an
MP bulk fit with aspect-ratio-correct edge + an entry-permutation null that calibrates the
edge-fluctuation buffer and the E_out null band. It accumulates per-FT E_out excess over the
permutation null on the DISCOVERY 12, pools the permutation E_out samples into δ_RMT, adjudicates
discovery-clear per matrix-type row, and measures the outlier-subspace reproducibility vs the Haar
floor. Sealed-12 confirmation + BH over ≤3 promoted rows run only if a row clears (autonomous;
frozen §6.2 rules).

  # smoke (2 ft, q_proj only):
  pg run infra.aws_pg.jobs.gauge_rmt --type r7i.4xlarge --no-weights \
      --args "--ft-tasks vqav2,gqa --types q_proj --n-perm 3" --budget-min 60
  # full:
  pg run infra.aws_pg.jobs.gauge_rmt --type r7i.8xlarge --no-weights \
      --args "--types q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --n-perm 3" --budget-min 600
"""
import argparse
import json
import pathlib

from infra.pg import gauge_common as GC
from infra.pg import gauge_cs as CS
from infra.pg import gauge_rmt as RMT
from infra.pg import relaw as RL          # bh_fdr for the sealed-confirmation BH step (§6.2)
# REUSED streaming spine (invariant_core.py): base/ft decoder load, matrix matching, cache delete.
from infra.aws_pg.jobs.invariant_core import (PG1_REPO, PROJ, _load_decoder, _decoder_matrices,
                                              _ftype, _delete_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty ⇒ enumerate all -224 (24-FT)")
    ap.add_argument("--types", default=",".join(PROJ))
    ap.add_argument("--n-perm", type=int, default=3, help="entry-permutation draws per matrix (≥3)")
    ap.add_argument("--n-haar", type=int, default=1000, help="Haar overlap-floor draws")
    ap.add_argument("--q-lo", type=float, default=0.25, help="central-quantile lo for the σ² fit")
    ap.add_argument("--q-hi", type=float, default=0.75, help="central-quantile hi for the σ² fit")
    ap.add_argument("--buf-pct", type=float, default=50.0, help="edge-buffer pct of perm overshoot")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    assert a.n_perm >= 3, "spec §6.2 requires ≥3 permutations"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in PROJ for t in want_types), f"unknown type in {want_types}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    api = HfApi()

    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    # deterministic 12/12 split (Gate-0-frozen; depends only on repo-ids).
    discovery, confirmation = GC.sha1_split(repos) if len(repos) >= GC.N_DISCOVERY else (repos, [])
    disc_idx = {r: i for i, r in enumerate(repos)}
    print(f"[gauge-rmt] base={PG1_REPO}@{base_sha[:8]} n_ft={len(repos)} "
          f"discovery={len(discovery)} sealed={len(confirmation)} types={want_types}", flush=True)

    rep = {"object": "gauge_rmt_v1", "generation": "gemma1_18L", "base_repo": PG1_REPO,
           "base_sha": base_sha, "cloud_repos": repos, "cloud_shas": shas,
           "discovery": discovery, "confirmation": confirmation,
           "n_perm": a.n_perm, "n_haar": a.n_haar, "q_lo": a.q_lo, "q_hi": a.q_hi,
           "buf_pct": a.buf_pct, "seed": a.seed,
           "estimand": "family-median E_out excess over the entry-permutation null per matrix-type; "
                       "outlier-subspace overlap vs the Haar floor; MP bulk = aspect-ratio-correct scaffolding"}

    w_base = _load_decoder(PG1_REPO, base_sha)

    # ---- PASS 1: difference every FT, store per-(type,layer,ft) fp16 ΔW scratch. ----
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    for ridx, repo_id in enumerate(repos):
        print(f"[gauge-rmt] diff {repo_id}@{shas[repo_id][:8]} ({ridx+1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo_id, shas[repo_id])
        matched = _decoder_matrices(w_base, w_ft)
        if ridx == 0:
            GC.assert_gemma1(list(matched.keys()))          # firewall (18-layer)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            if t not in want_types:
                continue
            layer = GC.layer_index(key)
            dW = (wf.to(torch.float64) - wb.to(torch.float64)).cpu().numpy()
            np.save(scratch / f"{t}__L{layer}__{ridx}.npy", dW.astype(np.float16))
        del w_ft, matched
        _delete_cache(repo_id)

    def _load(t, layer, ridx):
        f = scratch / f"{t}__L{layer}__{ridx}.npy"
        return np.load(f).astype(np.float64) if f.exists() else None

    # ---- PASS 2: per (type,layer) MATRIX — analyze each FT; accumulate discovery E_out excess. ----
    per_cell = []                       # one record per (type, layer)
    perm_pool = []                      # pooled discovery permutation E_out samples → δ_RMT
    for t in want_types:
        layers = sorted({int(f.stem.split("__L")[1].split("__")[0])
                         for f in scratch.glob(f"{t}__L*__0.npy")})
        for layer in layers:
            excesses, n_out_disc, rows = [], [], {}
            for r in discovery:
                A = _load(t, layer, disc_idx[r])
                if A is None:
                    continue
                seed_m = (a.seed * 1_000_003 + hash(f"{t}|{layer}|{r}") % 1_000_000) % (2 ** 31)
                res = RMT.analyze_matrix(A, n_perm=a.n_perm, seed=seed_m,
                                         q_lo=a.q_lo, q_hi=a.q_hi, buf_pct=a.buf_pct)
                excesses.append(CS.excess(res["e_out"], res["e_out_perm_median"]))
                n_out_disc.append(res["n_out"])
                perm_pool.extend(res["e_out_perm"])
                rows[r] = res
            if not excesses:
                continue
            eci = CS.median_excess_ci(excesses)
            mp_reject = [rows[r]["mp_reject"] for r in rows]
            per_cell.append({
                "type": t, "layer": layer, "n_disc": len(excesses),
                "e_out_excess": eci, "median_n_out": float(np.median(n_out_disc)),
                "mp_reject_any": bool(any(mp_reject)),
                "gamma": rows[next(iter(rows))]["gamma"],
                "per_ft": {r: {k: rows[r][k] for k in
                               ("n_out", "e_out", "e_out_perm_median", "lambda_plus",
                                "buffer", "sigma2", "ks", "ks_bound", "mp_reject")}
                           for r in rows}})
            print(f"[gauge-rmt] {t} L{layer} E_out_excess CI-lo={eci['ci_lo']:+.4f} "
                  f"median_n_out={np.median(n_out_disc):.1f} mp_reject_any={any(mp_reject)}", flush=True)
            (out / "gauge_rmt.json").write_text(json.dumps({**rep, "per_cell": per_cell}, indent=2))

    # ---- δ_RMT = pooled discovery permutation E_out band half-width (pre-real, frozen). ----
    delta_rmt = RMT.band_halfwidth(perm_pool) if perm_pool else 0.0
    # fit-quality gate: MP reject fraction over ALL discovery cells (per FT×matrix flags pooled).
    all_reject = [f["mp_reject"] for c in per_cell for f in c["per_ft"].values()]
    frac_reject, abort_descriptive = RMT.mp_fit_gate(all_reject) if all_reject else (0.0, False)

    # ---- discovery-clear per matrix-type ROW (median E_out excess CI-lo > δ_RMT, aggregated). ----
    for c in per_cell:
        c["delta_rmt"] = delta_rmt
        c["discovery_clear"] = RMT.discovery_clear_rmt(c["e_out_excess"]["ci_lo"], delta_rmt)

    # promote the ≤3 discovery-cleared matrix-type ROWS with the largest E_out-excess CI-lo (§6.2).
    by_type = {}
    for c in per_cell:
        if c["discovery_clear"]:
            by_type.setdefault(c["type"], []).append(c)
    row_scores = [(t, max(cs, key=lambda c: c["e_out_excess"]["ci_lo"])["e_out_excess"]["ci_lo"])
                  for t, cs in by_type.items()]
    row_scores.sort(key=lambda x: -x[1])
    promoted_types = [t for t, _ in row_scores[:3]]

    # ---- reproducibility: outlier-subspace overlap vs the Haar floor, on DISCOVERY (per type). ----
    repro = []
    for t in promoted_types:
        cells = [c for c in per_cell if c["type"] == t and c["discovery_clear"]]
        cell = max(cells, key=lambda c: c["e_out_excess"]["ci_lo"])
        layer = cell["layer"]
        k = max(1, int(round(cell["median_n_out"])))
        dws = [A for A in (_load(t, layer, disc_idx[r]) for r in discovery) if A is not None]
        m, n = dws[0].shape
        entry = {"type": t, "layer": layer, "k": k}
        for side, d in (("left", m), ("right", n)):
            real = RMT.mean_pairwise_overlap(dws, k, side=side)
            floor = RMT.haar_overlap_floor(k, d, n_draws=a.n_haar, seed=a.seed)
            beats = RMT.overlap_beats_floor(real["overlap"], floor)
            entry[side] = {"overlap": real["overlap"], "n_pairs": real["n_pairs"],
                           "haar_p95": floor["pcts"]["p95.0"], "haar_mean": floor["mean"],
                           "beats_floor": beats}
        entry["beats_floor"] = bool(entry["left"]["beats_floor"] or entry["right"]["beats_floor"])
        repro.append(entry)
        print(f"[gauge-rmt] repro {t} L{layer} k={k} "
              f"L_ov={entry['left']['overlap']:.4f}(p95={entry['left']['haar_p95']:.4f}) "
              f"R_ov={entry['right']['overlap']:.4f} beats={entry['beats_floor']}", flush=True)

    # ---- SEALED confirmation of promoted matrix-type rows (§6.2): recompute E_out excess on the
    #      SEALED 12 FTs at each promoted type's representative (largest-discovery-CI-lo) cell;
    #      DETECTION = sealed CI-lo > δ_RMT ∧ BH(q=.10) over the ≤3 rows ∧ outlier-subspace overlap
    #      beats the Haar floor. Reuses the in-run scratch (no re-download). Suppressed when the MP
    #      fit-quality gate has fired (abort_to_descriptive ⇒ edge/outlier machinery not interpreted).
    sealed_rows, sealed_p, sealed_meta = [], [], []
    repro_by_type = {e["type"]: e for e in repro}
    if not abort_descriptive and confirmation:          # sealed needs a non-empty held-out half
        for t in promoted_types:
            cells_t = [c for c in per_cell if c["type"] == t
                       and RMT.discovery_clear_rmt(c["e_out_excess"]["ci_lo"], delta_rmt)]
            cell = max(cells_t, key=lambda c: c["e_out_excess"]["ci_lo"])
            layer = cell["layer"]
            exc = []
            for r in confirmation:
                A = _load(t, layer, disc_idx[r])
                if A is None:
                    continue
                seed_m = (a.seed * 1_000_003 + hash(f"{t}|{layer}|{r}") % 1_000_000) % (2 ** 31)
                res = RMT.analyze_matrix(A, n_perm=a.n_perm, seed=seed_m,
                                         q_lo=a.q_lo, q_hi=a.q_hi, buf_pct=a.buf_pct)
                exc.append(CS.excess(res["e_out"], res["e_out_perm_median"]))
            sci = CS.median_excess_ci(exc)
            sealed_p.append(CS.bootstrap_p_le(exc, delta_rmt))
            sealed_meta.append((t, layer, sci, bool(repro_by_type.get(t, {}).get("beats_floor", False))))
        reject = RL.bh_fdr(sealed_p, q=0.10).tolist() if sealed_p else []
        for i, (t, layer, sci, beats) in enumerate(sealed_meta):
            det = RMT.detection_rmt(sci["ci_lo"], delta_rmt, bool(reject[i]), beats)
            sealed_rows.append({"type": t, "layer": layer, "sealed_e_out_excess": sci,
                                "delta_rmt": delta_rmt, "bh_p": sealed_p[i], "bh_ok": bool(reject[i]),
                                "overlap_beats_floor": beats, "detection": det})
            print(f"[gauge-rmt][SEALED] {t} L{layer}: sealed CI-lo={sci['ci_lo']:+.4f} "
                  f"δ={delta_rmt:.5f} p={sealed_p[i]:.3f} bh_ok={bool(reject[i])} "
                  f"beats_floor={beats} → DETECTION={det}", flush=True)
    n_detect = sum(1 for r in sealed_rows if r["detection"])
    rep_verdict = ("H_non-MP (fit-quality gate fired; descriptive only)" if abort_descriptive else
                   ("H_bulk+outliers (≥1 confirmed reproducible outlier family)" if n_detect >= 1 else
                    "H_all-bulk (0 confirmed outlier detections over q/k/v/o at n=24/12-12)"))
    rep.update({"per_cell": per_cell, "delta_rmt": delta_rmt,
                "sealed_confirmation": sealed_rows, "n_detections": n_detect, "verdict": rep_verdict,
                "mp_fit_frac_reject": frac_reject, "abort_to_descriptive": abort_descriptive,
                "promoted_types": promoted_types, "reproducibility": repro,
                "note": ("δ_RMT = pooled discovery permutation E_out band half-width (pre-real). "
                         "Sealed-half confirmation + BH over ≤3 promoted matrix-type rows run per "
                         "§6.2 only for promoted rows; abort_to_descriptive=True ⇒ H_non-MP "
                         "characterization, edge/outlier machinery NOT interpreted; overlap failing "
                         "the Haar floor ⇒ 'outliers exist but FT-idiosyncratic', not a detection.")})
    (out / "gauge_rmt.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78)
    print(f"GAUGE_RMT_OK n_ft={len(repos)} n_cells={len(per_cell)} delta_rmt={delta_rmt:.5f} "
          f"promoted={promoted_types} detections={n_detect} abort_descriptive={abort_descriptive} "
          f"verdict={rep_verdict}")


if __name__ == "__main__":
    main()
