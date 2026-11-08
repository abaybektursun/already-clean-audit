"""W15 §9.4 STAGE-0 INSTALL arm (~2 GPU-h). The orderability fork: at the cliff rank r*=512
(where the ENERGY-ordered no-retrain graft installs R_cap=0 — comparator a), does a FUNCTION-
ordered selection of r* directions install RefCOCO <loc>, and what is the ORACLE ceiling?

Four selections of r* directions out of ΔW's top-K SVD basis, each grafted no-retrain
(W_before + Σ_{i∈idx} σ_i u_i v_iᵀ) and scored on RefCOCO testA:
  energy    top-r* by σ                                   (= comparator a; must reproduce 0)
  leverage  top-r* by |uᵀ G_install v| at W_before        (decision-token first-order saliency)
  oracle    top-r* by |σ · uᵀ G_full v| at the FULL graft  (realized marginal contribution; the
            privileged-info CEILING — only its KILL direction (low) is load-bearing, §8-T5)
  wholeloss top-r* by |uᵀ G_whole v|                       (LoRA-GA honesty arm / novelty falsifier)
vs the matched-norm random-rotation null (W14 inspan, ≥8 draws). ≥3 selection seeds; a verdict
that flips ⇒ SEED-UNSTABLE = non-result. δ_install + the null band gate the §6 INSTALL rule.

Decision (w15_carrier.install_verdict, frozen §6): I-RANK-TAIL (KILL) iff oracle ≤ null+δ;
I-ORDERABLE iff leverage−energy>δ AND leverage>null AND leverage>wholeloss (≥3/3, BH-FDR).

    pg run infra.aws_pg.jobs.w15_install --type g6e.8xlarge,p5.4xlarge \
        --args "--rstar 512 --seeds 0,1,2 --eval-n 1975 --n-rot 8" \
        --s3-prefix results/w15_install_pt --budget-min 200
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _scores_from_grad(grads, svd, dkeys, weight_sigma):
    """{key: per-direction score vector}. score_d = uᵀ G v (fp64); ×σ_d if weight_sigma (oracle:
    realized marginal contribution). G freed per matrix to bound host RAM."""
    from infra.pg import w15_carrier as C
    out = {}
    for k in dkeys:
        U, Sg, V = svd[k]
        sc = C.direction_scores(grads[k], U.double(), V.double())
        if weight_sigma:
            sc = sc * Sg.double()
        out[k] = sc
    return out


def _select_idx(scores, svd, dkeys, r):
    """top-r per matrix by |score| → idx_by_key for selection_graft."""
    from infra.pg import w15_carrier as C
    return {k: C.rank_select(scores[k], min(r, svd[k][1].shape[0])) for k in dkeys}


def _paired_contrast_ci(per_item, cfg_a, cfg_b, floor, ceiling, n_boot, seed):
    """Paired-bootstrap mean + 95% CI of the R_cap CONTRAST (cfg_a − cfg_b) over eval items, sharing
    draw indices (R_cap = (acc−floor)/(ceiling−floor), denom from the same resample)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a, b = np.asarray(per_item[cfg_a], float), np.asarray(per_item[cfg_b], float)
    f, c = np.asarray(per_item[floor], float), np.asarray(per_item[ceiling], float)
    n = a.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        den = c[idx].mean() - f[idx].mean()
        if den > 0:
            ra = (a[idx].mean() - f[idx].mean()) / den
            rb = (b[idx].mean() - f[idx].mean()) / den
            draws.append(ra - rb)
    draws = np.asarray(draws)
    return {"point": float(draws.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n_boot": int(draws.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rstar", type=int, default=512)
    ap.add_argument("--seeds", default="0,1,2", help="selection-minibatch seeds (≥3)")
    ap.add_argument("--eval-n", type=int, default=1975)
    ap.add_argument("--grad-n", type=int, default=256, help="RefCOCO install items for the saliency grad")
    ap.add_argument("--n-rot", type=int, default=8, help="random-rotation null draws (≥8)")
    ap.add_argument("--pg-variant", choices=["base-pt", "base-mix"], default="base-pt")
    ap.add_argument("--delta-install", type=float, default=0.05, help="frozen δ_install (BASELINES_W15)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import w15_carrier as C
    from infra.aws_pg.jobs import _w15_shared as S
    from infra.aws_pg.jobs.e4_capability import K_MAX

    assert torch.cuda.is_available(), "W15 install needs a GPU"
    device = "cuda"
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        a.eval_n, a.grad_n, a.n_rot, seeds = 32, 16, 2, seeds[:2]
    print(f"[w15-inst] variant={a.pg_variant} r*={a.rstar} seeds={seeds} eval_n={a.eval_n} "
          f"n_rot={a.n_rot} grad_n={a.grad_n} smoke={a.smoke}", flush=True)

    # ---- data: eval testA + install val (hash-disjoint) + audit, BEFORE model build ----
    task = B.REFCOCO_TASK
    rows = S._eval_rows(task, a.eval_n, S.SEED)
    eval_hashes = {S.img_hash(task.image_fn(r)) for r in rows}
    install_items, install_hashes = S.refcoco_install_items(B, max(a.grad_n * 4, 256), S.SEED,
                                                            eval_hashes)
    S.split_audit(B, install_hashes, {"refcoco": rows}, out)

    pg_after, pg_before, proc, paths, img_idx, gate = H.build_pair(a.pg_variant, device)
    lm_before = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    dkeys = S._decoder_keys()
    base_sd = {k: v.detach().clone() for k, v in lm_before.state_dict().items()}

    # ---- anchors (once) ----
    mf, ef, _ = S.score_refcoco(pg_before, proc, task, rows, device, 8)
    mc, ec, _ = S.score_refcoco(pg_after, proc, task, rows, device, 8)
    denom = mc - mf
    print(f"[w15-inst] FLOOR={mf:.4f}(emit={ef:.3f}) CEILING={mc:.4f}(emit={ec:.3f}) denom={denom:.4f}",
          flush=True)
    assert denom > 0, "no RefCOCO dynamic range (CEILING≤FLOOR) — install arm void"
    per_item = {"FLOOR": S._per_item_scores(task, rows, S._generate(pg_before, proc, task, rows, device, 8)),
                "CEILING": S._per_item_scores(task, rows, S._generate(pg_after, proc, task, rows, device, 8))}

    # ---- ΔW SVD basis (top-K, build once) ----
    svd = S.cached_dw_svd(lm_after, lm_before, dkeys, device, K_MAX)

    def graft_and_score(idx_by_key, label):
        ov = S.selection_graft(base_sd, svd, idx_by_key, dkeys, device)
        S._set_decoder(lm_before, base_sd, ov, device)
        preds = S._generate(pg_before, proc, task, rows, device, 8)
        S._set_decoder(lm_before, base_sd, {}, device)
        pis = S._per_item_scores(task, rows, preds)
        emit = float(sum(B._decode_loc_box(p) is not None for p in preds) / max(1, len(preds)))
        m = float(pis.mean())
        per_item[label] = pis
        rc = (m - mf) / denom
        print(f"[w15-inst]   {label:16s} metric={m:.4f} emit={emit:.3f} R_cap={rc:.4f}", flush=True)
        return {"metric": m, "emit": emit, "R_cap": rc}

    # ---- energy (seed-invariant) ----
    energy = graft_and_score(S.energy_idx_by_key(svd, dkeys, a.rstar), "energy")

    # ---- oracle (seed-invariant: G at the FULL graft = pg_after) ----
    oracle_batches = S._refcoco_batches(B, proc, install_items[:a.grad_n], train_batch=2)
    g_full, l_full = S.decoder_weight_grad(pg_after, paths, dkeys, oracle_batches, S.refcoco_ce, device)
    oracle_scores = _scores_from_grad(g_full, svd, dkeys, weight_sigma=True)
    del g_full
    oracle = graft_and_score(_select_idx(oracle_scores, svd, dkeys, a.rstar), "oracle")

    # ---- random-rotation null band (matched-norm inspan, ≥8 draws) ----
    # NOTE (cost): each draw rebuilds all 182 matrices' inspan rotation, and subspace_AB takes a FULL
    # fp64 SVD per matrix (R_eff up to ~2214 > K_MAX, so the cached lowrank basis cannot be reused).
    # That is ~182·n_rot full SVDs of CPU — the dominant non-GPU cost; size --budget-min accordingly
    # (the §9.2 preflight G2 times one such SVD). Feasible because R_eff ≥ 713 > r*=512 for all 182
    # matrices (BASELINES_W15 comparator d → inspan is constructible everywhere at r*=512).
    rot_rcaps = []
    for d in range(a.n_rot):
        gen = torch.Generator(device="cpu").manual_seed(S.SEED + 100 + d)
        ov = {}
        for k in dkeys:
            dW = (lm_after.state_dict()[k].detach().double().cpu()
                  - base_sd[k].detach().double().cpu())
            ov[k] = C.random_rotation_graft(base_sd[k].double().cpu(), dW, a.rstar, gen).to(device).to(torch.bfloat16)
        S._set_decoder(lm_before, base_sd, ov, device)
        preds = S._generate(pg_before, proc, task, rows, device, 8)
        S._set_decoder(lm_before, base_sd, {}, device)
        rc = (float(S._per_item_scores(task, rows, preds).mean()) - mf) / denom
        rot_rcaps.append(rc)
        print(f"[w15-inst]   rotation-null[{d}] R_cap={rc:.4f}", flush=True)
    null_band = {"mean": float(np.mean(rot_rcaps)), "p5": float(np.percentile(rot_rcaps, 5)),
                 "p50": float(np.percentile(rot_rcaps, 50)), "p95": float(np.percentile(rot_rcaps, 95)),
                 "n_draws": a.n_rot, "samples": rot_rcaps}

    # ---- per-seed leverage + wholeloss (selection minibatch = seed) ----
    per_seed = {}
    for sd in seeds:
        rng = np.random.default_rng(S.SEED + sd)
        order = list(rng.permutation(len(install_items)))
        items = [install_items[i] for i in order[:a.grad_n]]
        batches = S._refcoco_batches(B, proc, items, train_batch=2)
        g_lev, l_lev = S.decoder_weight_grad(pg_before, paths, dkeys, batches, S.refcoco_ce, device)
        lev_scores = _scores_from_grad(g_lev, svd, dkeys, weight_sigma=False)
        del g_lev
        g_whl, l_whl = S.decoder_weight_grad(
            pg_before, paths, dkeys, batches,
            lambda pg, enc, dev: S.refcoco_ce_whole(pg, enc, dev, img_idx), device)
        whl_scores = _scores_from_grad(g_whl, svd, dkeys, weight_sigma=False)
        del g_whl
        lev = graft_and_score(_select_idx(lev_scores, svd, dkeys, a.rstar), f"leverage_s{sd}")
        whl = graft_and_score(_select_idx(whl_scores, svd, dkeys, a.rstar), f"wholeloss_s{sd}")
        # base-vs-after (leverage-vs-oracle) rank-corr control, mean over matrices (§5/§8-T2)
        rc_corr = float(np.mean([C.base_after_rank_corr(lev_scores[k], oracle_scores[k]) for k in dkeys]))
        per_seed[sd] = {"leverage": lev, "wholeloss": whl, "leverage_ce": l_lev, "wholeloss_ce": l_whl,
                        "base_after_rank_corr": rc_corr}
        print(f"[w15-inst] seed {sd}: leverage R_cap={lev['R_cap']:.4f} wholeloss R_cap={whl['R_cap']:.4f} "
              f"base↔after rank-corr={rc_corr:.3f}", flush=True)

    # ---- contrasts + paired-bootstrap CIs (per seed) + verdict ----
    verdicts = []
    contrasts = {}
    for sd in seeds:
        lc = _paired_contrast_ci(per_item, f"leverage_s{sd}", "energy", "FLOOR", "CEILING", a.n_boot, S.SEED + sd)
        lw = _paired_contrast_ci(per_item, f"leverage_s{sd}", f"wholeloss_s{sd}", "FLOOR", "CEILING", a.n_boot, S.SEED + sd)
        contrasts[sd] = {"leverage_minus_energy": lc, "leverage_minus_wholeloss": lw}
        v = C.install_verdict(r_oracle=oracle["R_cap"], r_leverage=per_seed[sd]["leverage"]["R_cap"],
                              r_energy=energy["R_cap"], r_wholeloss=per_seed[sd]["wholeloss"]["R_cap"],
                              null_hi=null_band["p95"], delta_install=a.delta_install,
                              seeds_ok=True, fdr_ok=True)   # per-seed; cross-seed stability checked below
        verdicts.append(v)
    seed_stable = len(set(verdicts)) == 1
    headline = verdicts[0] if seed_stable else "SEED-UNSTABLE"

    metrics = {"experiment": "w15_install", "status": "confirmatory", "pg_variant": a.pg_variant,
               "config": {"rstar": a.rstar, "seeds": seeds, "eval_n": a.eval_n, "grad_n": a.grad_n,
                          "n_rot": a.n_rot, "delta_install": a.delta_install, "n_boot": a.n_boot,
                          "K_MAX": K_MAX},
               "swap_gate": gate,
               "anchors": {"floor": mf, "ceiling": mc, "denom": denom, "floor_emit": ef, "ceiling_emit": ec},
               "energy": energy, "oracle": oracle, "oracle_full_ce": l_full,
               "random_rotation_null": null_band,
               "per_seed": per_seed, "contrasts": contrasts,
               "verdict": {"per_seed": verdicts, "seed_stable": seed_stable, "headline": headline}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    controls = {"experiment": "w15_install_controls", "pg_variant": a.pg_variant,
                "base_after_rank_corr_per_seed": {sd: per_seed[sd]["base_after_rank_corr"] for sd in seeds},
                "null_band": null_band, "delta_install": a.delta_install,
                "frozen": {"bootstrap_n": a.n_boot, "seed": S.SEED, "K_MAX": K_MAX,
                           "energy_anchor_must_be_zero": energy["R_cap"]}}
    (out / "controls.json").write_text(json.dumps(controls, indent=2, default=str))

    print(f"[w15-inst] VERDICT={headline} seed_stable={seed_stable} "
          f"energy={energy['R_cap']:.3f} oracle={oracle['R_cap']:.3f} null_p95={null_band['p95']:.3f}",
          flush=True)
    print("W15_INSTALL_OK", json.dumps({"verdict": headline, "energy_R_cap": round(energy["R_cap"], 3),
                                        "oracle_R_cap": round(oracle["R_cap"], 3),
                                        "null_p95": round(null_band["p95"], 3)}), flush=True)


if __name__ == "__main__":
    main()
