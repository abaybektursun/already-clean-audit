"""M-C1 — the fine-tune linter (queue C1). Wave P, Register B.

    THE DELIVERABLE. Given a base and a fine-tune's weights — no training, no gradients, no eval
    set — predict how much base capability the fine-tune lost. If the program's weight-space
    picture is worth anything practically, that prediction should beat knowing nothing.

    REFUTER (pre-logged): forecasts no closer than the no-model baseline at the pre-set factor.

THE FEATURES ARE THE PROGRAM'S OWN OBJECTS, one number per bank plus two scale terms:

    sink_leak       ||dW U_core||^2 / ||dW||^2         the share of the edit that lands in the sink
    highway         ||dW U_anti||^2 / ||dW||^2         and in the consensus TOP
    topsigma/bottomsigma/complement/haar shares        the rest of the battery, including the
                                                       matched-random level a share is read against
    log_motion      log ||dW||_F                       how big the edit is at all
    eff_rank        participation ratio of dW's singular values, normalised by min(m, n)

THE BASELINE IS THE HONEST ONE. "No model" means predicting the TRAINING mean of the target for
every held-out fine-tune — the best a forecaster with no features can do. Beating zero would be
trivial and beating a fitted-on-the-same-data model would be circular, so the comparison is
leave-one-out: every fine-tune is predicted by a model that never saw it, and the baseline is
computed the same way.

THE FACTOR IS FROZEN IN THIS SOURCE, before the run: the forecast's mean absolute error must be at
most 1/1.25 of the baseline's. A factor chosen after seeing the errors would make the refuter
unfalsifiable.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-C1"
BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
         "haar32_s0", "haar32_s1", "haar32_s2")
# FROZEN BEFORE THE RUN: the forecast must cut the no-model baseline's MAE by at least this much.
REQUIRED_FACTOR = 1.25
PPL_CORPUS = "wikitext2:window512"


def features(dW_by_cell, banks_by_cell) -> dict:
    """One feature vector for a whole fine-tune: bank shares averaged over cells, plus scale."""
    shares = {b: [] for b in BANKS}
    motions, eff_ranks = [], []
    for key, dW in dW_by_cell.items():
        n2 = float((dW ** 2).sum())
        if n2 == 0:
            continue
        motions.append(float(np.sqrt(n2)))
        for b in BANKS:
            U = banks_by_cell[key][b]
            shares[b].append(float(((dW @ U) ** 2).sum()) / n2)
        sv = np.linalg.svd(dW, compute_uv=False)
        s2 = sv ** 2
        eff_ranks.append(float((s2.sum() ** 2) / (s2 ** 2).sum() / min(dW.shape)))
    assert motions, "every cell of this fine-tune is bit-identical to the base"
    f = {f"share_{b}": float(np.mean(shares[b])) for b in BANKS}
    f["log_motion"] = float(np.log(np.sum(motions)))
    f["eff_rank"] = float(np.mean(eff_ranks))
    f["n_cells_moved"] = float(len(motions))
    return f


def loo_ridge(X, y, alpha: float = 1.0):
    """Leave-one-out ridge predictions, and the no-model (training-mean) predictions beside them.

    Both are computed the SAME way — refit per held-out point — so the comparison is not between a
    cross-validated model and an in-sample baseline.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    assert n > d + 1, f"{n} fine-tunes against {d} features; leave-one-out ridge would interpolate"
    pred, base = np.zeros(n), np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        Xt, yt = X[m], y[m]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd[sd == 0] = 1.0
        Z = (Xt - mu) / sd
        A = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        R = alpha * np.eye(A.shape[1])
        R[-1, -1] = 0.0                                   # never penalise the intercept
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        z = np.concatenate([(X[i] - mu) / sd, [1.0]])
        pred[i] = float(z @ w)
        base[i] = float(yt.mean())
    return pred, base


def main():
    ap = argparse.ArgumentParser(description="M-C1 — the fine-tune linter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--n-models", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_c1", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.pg import darwin_census as DC
    from infra.aws_pg.jobs.invariant_core_func import _load_full_pg1, _ppl_blankimg
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    base = _load_full_pg1(base_sha, "cuda").eval()
    base_ppl = float(_ppl_blankimg(base, processor, "cuda", n_seq=a.n_seq)["ppl"])
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[M-C1] base ppl {base_ppl:.5f}", flush=True)

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    keys = [(t, L) for t in types for L in layers]
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in BANKS} for k in keys}
    base_mats = M.load_decoder_matrices(WT.PG1_REPO, base_sha, types, layers)

    # The roster: the PG1 fine-tune family, enumerated from the Hub by the same rule the census
    # uses, so "the models the linter was fitted on" is a reproducible set rather than a list.
    from transformers import AutoConfig
    geom = DC.base_geometry(AutoConfig.from_pretrained(WT.PG1_REPO, revision=base_sha))
    accepted, log = DC.select_derivatives(api, WT.PG1_REPO, geom, n_accept=a.n_models, out=out)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    # HEADER PREFLIGHT. This arm LOADS every derivative to score its perplexity, so a checkpoint
    # that cannot be instantiated into the base architecture takes it down mid-roster — it died 17
    # of 24 in on a repo whose vision tower is 768-wide against PG1's 1152. That is the defect
    # M-D54 added the loadability check for; the check went into validate_candidate and was wired
    # into a DIFFERENT arm, leaving the one that exposed it still unguarded (M-D66).
    accepted, prevalid_log = DC.preflight_roster(WT.PG1_REPO, base_sha, accepted, shas,
                                                 types, layers, tag="M-C1")
    M.write_json(out, "preflight.json", prevalid_log)
    print(f"[M-C1] roster: {len(accepted)} of {log['n_candidates']} candidates accepted",
          flush=True)
    M.write_json(out, "roster.json", {"accepted": [r["id"] for r in accepted],
                                      "shas": shas, "selection": log})
    # THE FORECASTER NEEDS MORE FINE-TUNES THAN FEATURES, and leave-one-out ridge on an
    # under-determined design interpolates rather than generalises. Checking here means a thin
    # roster is a recorded coverage limit, not a forecast that looks excellent because it memorised.
    n_features = len(BANKS) + 3
    assert len(accepted) > n_features + 1, (
        f"{len(accepted)} fine-tunes against {n_features} features; the linter cannot be "
        f"cross-validated on this roster")

    rows = []
    for i, rec in enumerate(accepted):
        repo = rec["id"]
        sha = shas[repo]
        mats = M.load_decoder_matrices(repo, sha, types, layers)
        dW = {k: (mats[k].to(torch.float64) - base_mats[k].to(torch.float64)).numpy()
              for k in keys}
        f = features(dW, banks)
        del mats, dW
        gc.collect()
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=sha, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager").eval()
        ppl = float(_ppl_blankimg(m, processor, "cuda", n_seq=a.n_seq)["ppl"])
        del m
        gc.collect()
        torch.cuda.empty_cache()
        _delete_cache(repo)
        target = float(np.log(ppl / base_ppl))
        rows.append({"repo": repo, "sha": sha, "ppl": ppl, "log_damage": target, **f})
        at.add("linter_measured_damage", value=target, model_id=repo, model_sha=sha,
               corpus_id=PPL_CORPUS,
               flags={**f, "ppl": ppl, "base_ppl": base_ppl,
                      "target": "log(ppl_ft / ppl_base) on the blank-image wikitext readout"})
        print(f"[M-C1] {i+1}/{len(accepted)} {repo.split('/')[-1]:28s} ppl={ppl:8.3f} "
              f"log_damage={target:+.4f} sink_leak={f['share_core32']:.4f} "
              f"highway={f['share_anti32']:.4f}", flush=True)
        M.s3_sync_incremental(out)

    feat_names = [f"share_{b}" for b in BANKS] + ["log_motion", "eff_rank", "n_cells_moved"]
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=np.float64)
    y = np.array([r["log_damage"] for r in rows], dtype=np.float64)
    pred, base_pred = loo_ridge(X, y, alpha=a.alpha)
    mae = float(np.abs(pred - y).mean())
    mae_base = float(np.abs(base_pred - y).mean())
    factor = mae_base / mae if mae > 0 else float("inf")
    beats = bool(factor >= REQUIRED_FACTOR)
    at.add("linter_forecast_mae", value=mae, n=len(rows), incumbent_id="no_model_training_mean",
           flags={"mae_no_model": mae_base, "factor": factor,
                  "required_factor": REQUIRED_FACTOR, "features": feat_names,
                  "alpha": a.alpha, "cv": "leave-one-out, baseline refit identically"})
    # The fitted coefficients, on standardized features, for the whole roster — the "leak law" in
    # the only form this arm can support: a linear read-out with its sign and size stated.
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.concatenate([Z, np.ones((len(rows), 1))], axis=1)
    R = a.alpha * np.eye(A.shape[1])
    R[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    coefs = {n: float(c) for n, c in zip(feat_names + ["intercept"], w)}
    for n, c in coefs.items():
        at.add("linter_coefficient", value=c, bank_id=(n[6:] if n.startswith("share_") else None),
               flags={"feature": n, "standardized": True, "alpha": a.alpha, "n": len(rows)})
    gates.record("c1_forecast_beats_no_model_baseline", beats, mae=mae, mae_no_model=mae_base,
                 factor=factor, required_factor=REQUIRED_FACTOR, n=len(rows),
                 note=("the factor was frozen in the source before the run; both arms are "
                       "leave-one-out so the baseline is not an in-sample comparison"))
    # THE SINK'S OWN CONTRIBUTION, isolated: refit without the core32 share and see what the
    # forecast loses. A linter that predicts just as well without the program's central object is
    # a useful linter and a null result for the program, and both halves get recorded.
    j = feat_names.index("share_core32")
    keep = [i for i in range(len(feat_names)) if i != j]
    pred_ns, _ = loo_ridge(X[:, keep], y, alpha=a.alpha)
    mae_ns = float(np.abs(pred_ns - y).mean())
    at.add("linter_forecast_mae_without_sink", value=mae_ns, n=len(rows),
           flags={"mae_with_sink": mae, "delta": mae_ns - mae,
                  "features": [feat_names[i] for i in keep]})
    gates.record("c1_sink_feature_carries_forecast_value", bool(mae_ns > mae),
                 mae_with_sink=mae, mae_without_sink=mae_ns,
                 note=("dropping the core32 share and refitting: if the forecast does not get "
                       "worse, the linter works without the program's central object"))
    # THE CONTROL COLUMN IS THE PER-FOLD BASELINE ERRORS, not the single number they average to.
    # `surprise` derives its band from the spread of the control sample, and a one-element list has
    # no spread — the arm crashed there with both gates already recorded and all 23 fine-tunes
    # measured (M-D76). The no-model arm's own leave-one-out errors are the matched control: same
    # folds, same targets, one prediction rule apart.
    base_errs = np.abs(base_pred - y)
    sur.add(arm=ARM, expected_ref=expected_ref, cell="loo", statistic="linter_forecast_mae",
            s=M.surprise(mae, mae_base, base_errs))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "base_ppl": base_ppl, "types": list(types), "layers": layers,
                  "n_finetunes": len(rows), "features": feat_names, "alpha": a.alpha,
                  "required_factor": REQUIRED_FACTOR, "rows": rows,
                  "mae": mae, "mae_no_model": mae_base, "factor": factor,
                  "beats_baseline": beats, "coefficients": coefs,
                  "mae_without_sink_feature": mae_ns,
                  "predictions": {r["repo"]: {"actual": float(y[i]), "forecast": float(pred[i]),
                                              "no_model": float(base_pred[i])}
                                  for i, r in enumerate(rows)}})
    print("C1_OK " + json.dumps({"rows": info["n_rows"], "mae": mae, "mae_no_model": mae_base,
                                 "factor": factor, "beats": beats}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
