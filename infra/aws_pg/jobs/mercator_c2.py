"""M-C2 — the merge pre-check (queue C2). Wave P, Register B.

    THE DELIVERABLE. Two fine-tunes of the same base are about to be TIES-merged. From their
    weights alone, before merging, predict how much the merge will cost. If the program's
    highway/sink decomposition is worth anything here, a "highway collision" profile — how much
    the two edits overlap in the subspaces that actually carry function — should predict merge
    damage better than the obvious scale baseline.

    REFUTER (pre-logged): prediction no better than the norm-product baseline.

THE BASELINE IS THE ONE A PRACTITIONER WOULD ACTUALLY USE. `||dW_a|| * ||dW_b||` — big edits
collide, small ones do not. It is a real baseline rather than a straw one, and beating it is the
whole claim; the collision features have to earn their keep against scale.

THE COLLISION FEATURES, per pair, aggregated over cells:

    collide_<bank>   <P_U dW_a, P_U dW_b> / (||dW_a|| ||dW_b||)   signed overlap INSIDE a bank
    cos_full         the same over the whole matrix — the scale-free baseline feature
    sign_conflict    the fraction of coordinates where the two edits disagree in sign, which is
                     the quantity TIES itself resolves, so it belongs in the comparison

THE TARGET IS MEASURED, NOT MODELLED. Each pair is actually TIES-merged and evaluated: damage is
log(ppl_merge / ppl_best_parent), so a merge that is no worse than its better parent scores <= 0.
`ppl_best_parent` rather than the base, because the practitioner's alternative to merging is
shipping one of the two models, not shipping the base.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-C2"
BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
         "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TIES_DENSITY = 0.2          # the standard TIES trim: keep the top 20% of |dW| per model


def ties_merge(deltas, density: float):
    """TIES: trim each delta to its top-`density` magnitude entries, elect a sign per coordinate
    by summed magnitude, then average only the entries agreeing with the elected sign.

    Written out rather than imported so the merge this arm predicts is the merge it evaluates, and
    so the sign election — the step the collision features are about — is readable.
    """
    import torch
    trimmed = []
    for d in deltas:
        flat = d.reshape(-1)
        k = max(1, int(round(density * flat.numel())))
        idx = torch.topk(flat.abs(), k, sorted=False).indices
        t = torch.zeros_like(flat)
        t[idx] = flat[idx]
        trimmed.append(t)
    stack = torch.stack(trimmed)
    elected = torch.sign(torch.sign(stack).mul(stack.abs()).sum(0))
    agree = (torch.sign(stack) == elected.unsqueeze(0)) & (stack != 0)
    num = (stack * agree).sum(0)
    den = agree.sum(0).clamp(min=1)
    return (num / den).reshape(deltas[0].shape)


def pair_features(dA, dB, banks) -> dict:
    """One feature vector per pair, averaged over cells."""
    f = {f"collide_{b}": [] for b in BANKS}
    cos_full, sign_conflict, log_norm_product = [], [], []
    for key in dA:
        a, b_ = dA[key], dB[key]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b_))
        if na == 0 or nb == 0:
            continue
        cos_full.append(float((a * b_).sum()) / (na * nb))
        sign_conflict.append(float((np.sign(a) * np.sign(b_) < 0).mean()))
        log_norm_product.append(np.log(na) + np.log(nb))
        for bk in BANKS:
            U = banks[key][bk]
            f[f"collide_{bk}"].append(float(((a @ U) * (b_ @ U)).sum()) / (na * nb))
    assert cos_full, "both edits are bit-identical to the base in every cell"
    out = {k: float(np.mean(v)) for k, v in f.items()}
    out["cos_full"] = float(np.mean(cos_full))
    out["sign_conflict"] = float(np.mean(sign_conflict))
    out["log_norm_product"] = float(np.mean(log_norm_product))
    return out


def main():
    ap = argparse.ArgumentParser(description="M-C2 — the merge pre-check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-models", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--density", type=float, default=TIES_DENSITY)
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
    at = M.Atlas(out, arm=ARM, run_id="m_c2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, PaliGemmaProcessor
    from infra.pg import darwin_census as DC
    from infra.aws_pg.jobs.invariant_core_func import _load_full_pg1, _ppl_blankimg
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.mercator_c1 import loo_ridge

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, "cuda")
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    keys = [(t, L) for t in types for L in layers]
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers
    from infra.pg import darwin_train as T
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    print(f"[M-C2] base ppl {base_ppl:.5f} over {len(mods)} cells", flush=True)

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in BANKS} for k in keys}
    geom = DC.base_geometry(AutoConfig.from_pretrained(WT.PG1_REPO, revision=base_sha))
    accepted, log = DC.select_derivatives(api, WT.PG1_REPO, geom, n_accept=a.n_models, out=out)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    repos = [r["id"] for r in accepted]
    print(f"[M-C2] roster: {len(repos)} of {log['n_candidates']} candidates", flush=True)

    # The deltas, kept on CPU: n_models x len(keys) matrices of 2048x2048 fp32 is a few GB.
    deltas, parent_ppl = {}, {}
    for repo in repos:
        mats = M.load_decoder_matrices(repo, shas[repo], types, layers)
        deltas[repo] = {k: (mats[k].to(torch.float32) - pristine[k].cpu()) for k in keys}
        for k, m in mods.items():
            m.weight.data.copy_(mats[k].to(m.weight.dtype).to("cuda"))
        parent_ppl[repo] = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        del mats
        gc.collect()
        _delete_cache(repo)
        print(f"[M-C2] parent {repo.split('/')[-1]:28s} ppl={parent_ppl[repo]:8.3f}", flush=True)
        M.s3_sync_incremental(out)

    rng = np.random.default_rng(a.seed)
    all_pairs = list(itertools.combinations(range(len(repos)), 2))
    assert len(all_pairs) >= a.n_pairs, f"{len(all_pairs)} pairs available, wanted {a.n_pairs}"
    pick = rng.choice(len(all_pairs), size=a.n_pairs, replace=False)
    rows = []
    for pi in sorted(pick):
        i, j = all_pairs[pi]
        ra, rb = repos[i], repos[j]
        f = pair_features({k: deltas[ra][k].numpy().astype(np.float64) for k in keys},
                          {k: deltas[rb][k].numpy().astype(np.float64) for k in keys}, banks)
        with torch.no_grad():
            for k, m in mods.items():
                merged = ties_merge([deltas[ra][k], deltas[rb][k]], a.density)
                m.weight.data.copy_((pristine[k].cpu() + merged).to(m.weight.dtype).to("cuda"))
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        best_parent = min(parent_ppl[ra], parent_ppl[rb])
        target = float(np.log(ppl / best_parent))
        rows.append({"a": ra, "b": rb, "merged_ppl": ppl, "best_parent_ppl": best_parent,
                     "log_merge_damage": target, **f})
        at.add("merge_measured_damage", value=target, corpus_id=PPL_CORPUS,
               model_id=f"{ra}+{rb}",
               flags={**f, "merged_ppl": ppl, "best_parent_ppl": best_parent,
                      "base_ppl": base_ppl, "density": a.density,
                      "target": "log(ppl_merge / ppl_best_parent); <=0 means no worse than the "
                                "better parent"})
        print(f"[M-C2] {ra.split('/')[-1][:18]:18s} + {rb.split('/')[-1][:18]:18s} "
              f"merged={ppl:8.3f} best_parent={best_parent:8.3f} "
              f"damage={target:+.4f} collide_core={f['collide_core32']:+.5f}", flush=True)
        M.s3_sync_incremental(out)

    feat_names = [f"collide_{b}" for b in BANKS] + ["cos_full", "sign_conflict"]
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=np.float64)
    y = np.array([r["log_merge_damage"] for r in rows], dtype=np.float64)
    pred, _ = loo_ridge(X, y, alpha=a.alpha)
    # THE NORM-PRODUCT BASELINE, fitted leave-one-out exactly like the treatment so the comparison
    # is not a cross-validated model against an in-sample one.
    Xn = np.array([[r["log_norm_product"]] for r in rows], dtype=np.float64)
    pred_n, pred_mean = loo_ridge(Xn, y, alpha=a.alpha)
    mae = float(np.abs(pred - y).mean())
    mae_norm = float(np.abs(pred_n - y).mean())
    mae_mean = float(np.abs(pred_mean - y).mean())
    beats = bool(mae < mae_norm)
    at.add("merge_forecast_mae", value=mae, n=len(rows), incumbent_id="norm_product",
           flags={"mae_norm_product": mae_norm, "mae_no_model": mae_mean,
                  "features": feat_names, "alpha": a.alpha, "n_pairs": len(rows),
                  "cv": "leave-one-out; both arms refit identically"})
    # BEATING THE PRACTITIONER'S BASELINE MEANS NOTHING IF BOTH LOSE TO THE NULL. The first run
    # had this gate PASS on 0.005041 against 0.005049 — a 0.16% margin on 20 pairs — while
    # `c2_any_model_beats_knowing_nothing` FAILED at 0.005041 against the 0.004725 you get by
    # predicting the mean. A model that loses to knowing nothing has not earned a comparison with
    # a rival model, and a reader seeing PASS two rows below FAIL should not have to join them up
    # (M-D83, the same shape as M-A5's motion_matched and M-G's denominator gates).
    beats_null = bool(mae < mae_mean)
    gates.record("c2_collision_profile_beats_norm_product", beats,
                 mae=mae, mae_norm_product=mae_norm, mae_no_model=mae_mean, n_pairs=len(rows),
                 leave_one_out_is_optimistic=True, beats_no_model_baseline=beats_null,
                 margin_over_norm_product=float(mae_norm - mae),
                 outcome=None if beats_null else M.VOID_INSTRUMENT,
                 note=("the practitioner's baseline is scale: big edits collide. The collision "
                       "features have to beat that, not beat zero. NOTE: pairs share parents, so "
                       "leave-one-out over PAIRS is optimistic for both arms — it is a fair "
                       "comparison between them and an upper bound on either one's accuracy"))
    gates.record("c2_any_model_beats_knowing_nothing", bool(min(mae, mae_norm) < mae_mean),
                 mae=mae, mae_norm_product=mae_norm, mae_no_model=mae_mean,
                 note=("if neither beats the training mean, the pairs' damages are not "
                       "predictable from weights at this sample size and both readings are that"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell="loo", statistic="merge_forecast_mae",
            s=M.surprise(mae, mae_norm, [mae_norm, mae_mean]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "base_ppl": base_ppl, "types": list(types), "layers": layers,
                  "roster": repos, "parent_ppl": parent_ppl, "density": a.density,
                  "n_pairs": len(rows), "features": feat_names, "rows": rows,
                  "mae": mae, "mae_norm_product": mae_norm, "mae_no_model": mae_mean,
                  "beats_norm_product": beats,
                  "merge": "TIES, written out in this job so the predicted merge IS the "
                           "evaluated merge",
                  "cv_caveat": ("pairs share parents, so leave-one-out over pairs is optimistic "
                                "for both the collision model and the norm-product baseline; the "
                                "comparison between them is fair, the absolute MAEs are an upper "
                                "bound on accuracy")})
    print("C2_OK " + json.dumps({"rows": info["n_rows"], "mae": mae,
                                 "mae_norm_product": mae_norm, "beats": beats}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
