"""rmask cell runner — is the Gemma-2 L16 MMLU-MCQ deletion a READOUT MASK of preserved computation, or a
genuine information LOSS? (prereg `studies/lineage/paligemma2/research/2026-07-01_readout_mask_prereg.md`.)

Reads the DELETED residual ALONE with a minimal linear probe (removability of the difference is answered —
LATABLATE/FLENS). Identification triangle: intact-L16 CEILING (+ fail-fast preflight), early-layer FLOOR
(L_e=4), shuffled-label NULL — so NO outcome is instrument-ambiguous. Pure gap/bootstrap/null logic lives in
`infra.pg.rmask` (weight-free, --dry-run-tested); the harvest + the sklearn probe are here.

Two entrypoints:
  --dry-run   weight-free: pure-logic dry-run (infra.pg.rmask.dry_run) AND a tiny sklearn-probe dry-run
              (real LogisticRegression on separable/shuffled blobs) on CPU. Dispatch on pg `--no-weights`.
  (default)   the GPU MDE: eager loader + 26-firewall + deleted/intact MCQ floor + ONE multi-layer
              last-token harvest per model (disc ∪ conf) + fit-disc/test-conf probes + the controls.

    python -m infra.aws_pg.jobs.rmask_cell --dry-run
    python -m infra.aws_pg.jobs.rmask_cell --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.latloc_cell import assert_generation_layers, _capture, build_confirmation_items


# ===========================================================================
# Multi-layer last-token residual harvest (ONE forward per model; prereg §3 step 2). Left-pad so position
# −1 is the readout token for every item; both models see IDENTICAL (ids, am, pv) per batch → rows paired.
# ===========================================================================

def harvest_layers(model, layers, items, proc, gray_img, device, layer_set, batch_size=4):
    import numpy as np
    import torch

    tok = proc.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    rows = {L: [] for L in layer_set}
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        inp = proc(text=[it["prompt"] for it in batch], images=[gray_img] * len(batch),
                   return_tensors="pt", padding="longest").to(device)
        ids, am = inp["input_ids"], inp["attention_mask"]
        pv = inp["pixel_values"].to(torch.bfloat16)
        cap = _capture(model, layers, list(layer_set), ids, am, pv)   # {L: (b,T,d)}
        for L in layer_set:
            rows[L].append(cap[L][:, -1, :].float().cpu().numpy())
    tok.padding_side = orig
    out = {L: np.concatenate(rows[L], 0).astype(np.float32) for L in layer_set}
    n = out[layer_set[0]].shape[0]
    assert n == len(items), f"harvest row mismatch {n} vs {len(items)}"
    return out


# ===========================================================================
# The sklearn probe: StandardScaler (fit on discovery) + LogisticRegression with C by 5-fold CV on
# DISCOVERY ONLY (prereg §6). Returns the fitted pipeline + the CV accuracy (the preflight/instrument read).
# ===========================================================================

def fit_probe(Xtr, ytr, c_grid, seed=0, fixed_c=None):
    """StandardScaler (fit on Xtr) + LogisticRegression. C by 5-fold CV on discovery — UNLESS fixed_c is
    given, in which case CV is skipped and C is held at fixed_c (the data-efficiency curve holds C at the
    headline-selected C so ONLY train size varies, and this dodges StratifiedKFold degeneracy at tiny
    train sizes). Returns (scaler, clf, cv, C); cv is NaN when fixed_c is used."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    if fixed_c is not None:
        best_c, best_cv = fixed_c, float("nan")
    else:
        best_c, best_cv = None, -1.0
        for C in c_grid:
            clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=C, random_state=seed)  # default penalty=l2
            cv = float(np.mean(cross_val_score(clf, Xs, ytr, cv=5)))
            if cv > best_cv:
                best_cv, best_c = cv, C
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=best_c,
                             random_state=seed).fit(Xs, ytr)  # default penalty=l2
    return scaler, clf, best_cv, best_c


def score_probe(scaler, clf, Xte, yte):
    import numpy as np
    pred = clf.predict(scaler.transform(Xte))
    correct = (pred == np.asarray(yte)).astype(int)
    return float(correct.mean()), correct.tolist()


# ===========================================================================
# Weight-free sklearn dry-run (prereg §13) — a REAL LogisticRegression on separable / shuffled blobs. CPU.
# ===========================================================================

def dry_run_sklearn() -> dict:
    import numpy as np
    import infra.pg.rmask as RM

    rng = np.random.default_rng(0)
    d, n, nc = 32, 800, 4
    centers = rng.standard_normal((nc, d))
    y = rng.integers(0, nc, n)
    X = 3.0 * centers[y] + rng.standard_normal((n, d))
    disc = np.arange(n) % 2 == 0
    sc, clf, cv, C = fit_probe(X[disc], y[disc], RM.C_GRID)
    acc, _ = score_probe(sc, clf, X[~disc], y[~disc])
    # shuffled labels → chance on held-out (memorization control)
    ys = RM.shuffled_labels(y, 301)
    scs, clfs, _, _ = fit_probe(X[disc], ys[disc], RM.C_GRID)
    acc_s, _ = score_probe(scs, clfs, X[~disc], ys[~disc])
    assert acc > 0.80, f"sklearn probe should decode separable blobs (got {acc:.3f})"
    assert abs(acc_s - RM.CHANCE) < 0.08, f"shuffled sklearn null should be ~chance (got {acc_s:.3f})"
    return {"separable_acc": round(acc, 3), "shuffled_null": round(acc_s, 3), "cv": round(cv, 3), "C": C}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="weight-free §13 dry-run (pure logic + sklearn), CPU")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()

    if args.dry_run:
        import infra.pg.rmask as RM
        pure = RM.dry_run()
        skl = dry_run_sklearn()
        print(json.dumps({"pure_logic": pure, "sklearn_probe": skl}, indent=2, default=str))
        print("[rmask] WEIGHT-FREE DRY-RUN PASSED (pure logic + real sklearn probe).")
        return

    assert args.out, "--out required for the GPU run"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as Wp
    from infra.pg import rmask as RM
    from infra.pg.gemma_scope import resolve_gemma2_layers

    assert torch.cuda.is_available(), "rmask GPU run needs CUDA."
    device = "cuda"
    n_per = args.n_per_subject or None
    layer_set = RM.LAYER_SET

    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, attn_impl="eager")
    n_dec = assert_generation_layers(pg_after.model.language_model, 26)
    a_layers = resolve_gemma2_layers(pg_after.model.language_model)     # deleted
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)    # intact
    print(f"[rmask] eager loader OK; firewall {n_dec}==26; gate={gate['decoder_max_abs_delta_vs_PG']:.3g}")

    # ---- deleted/intact MCQ floor on the sealed half (the readout anchors), recomputed on the SAME items ----
    mcq_del = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=Wp.is_confirmation)
    mcq_int = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, item_filter=Wp.is_confirmation)
    del_mcq, int_mcq = mcq_del["acc"], mcq_int["acc"]
    print(f"[rmask] confirmation MCQ: deleted={del_mcq:.3f} intact={int_mcq:.3f} (chance {RM.CHANCE})")

    disc_items, gray_img, letter_ids = build_confirmation_items(proc, Wp.is_grid_subset, n_per_subject=n_per)
    conf_items, _gimg, _lids = build_confirmation_items(proc, Wp.is_confirmation, n_per_subject=n_per)
    y_disc = np.array([it["answer"] for it in disc_items])
    y_conf = np.array([it["answer"] for it in conf_items])
    print(f"[rmask] items: discovery={len(disc_items)} confirmation={len(conf_items)}; harvesting {len(layer_set)} layers ×2 models ×2 splits")

    # ---- ONE multi-layer harvest per (model, split) ----
    Hd_disc = harvest_layers(pg_after, a_layers, disc_items, proc, gray_img, device, layer_set)
    Hd_conf = harvest_layers(pg_after, a_layers, conf_items, proc, gray_img, device, layer_set)
    Hi_disc = harvest_layers(pg_before, b_layers, disc_items, proc, gray_img, device, layer_set)
    Hi_conf = harvest_layers(pg_before, b_layers, conf_items, proc, gray_img, device, layer_set)
    print(f"[rmask] harvested; d_model={Hd_disc[layer_set[0]].shape[1]}")

    # ---- fit-disc / test-conf probe per (model, layer); the CV acc is the instrument read ----
    def run_probe(Xd, yd, Xc, yc):
        sc, clf, cv, C = fit_probe(Xd, yd, RM.C_GRID)
        acc, correct = score_probe(sc, clf, Xc, yc)
        return {"cv": cv, "C": C, "acc": acc, "correct": correct, "scaler": sc, "clf": clf}

    profile = {"deleted": {}, "intact": {}}
    for L in layer_set:
        profile["deleted"][L] = run_probe(Hd_disc[L], y_disc, Hd_conf[L], y_conf)
        profile["intact"][L] = run_probe(Hi_disc[L], y_disc, Hi_conf[L], y_conf)
        print(f"[rmask] L={L:2d}  deleted-probe acc={profile['deleted'][L]['acc']:.3f} (cv {profile['deleted'][L]['cv']:.3f})"
              f"  intact-probe acc={profile['intact'][L]['acc']:.3f} (cv {profile['intact'][L]['cv']:.3f})")

    # ---- fail-fast preflight (prereg §2/§9): the intact probe must decode a KNOWN-present signal (CV on
    #      discovery) at the headline layer; else promote L per the fixed pivot order. ----
    L_head = None
    for L in RM.L_PIVOT_ORDER:
        if profile["intact"][L]["cv"] >= RM.CHANCE + RM.DELTA_DECODE:
            L_head = L
            break
    preflight_pass = L_head is not None
    if not preflight_pass:
        L_head = RM.L_HEADLINE   # report the instrument limitation at L16
    print(f"[rmask] PREFLIGHT: intact-probe clears chance+δ at L_head={L_head} (pass={preflight_pass}); "
          f"intact CV @16={profile['intact'][RM.L_HEADLINE]['cv']:.3f}")

    # ---- the headline contrast at L_head + the identification triangle ----
    d16 = profile["deleted"][L_head]
    i16 = profile["intact"][L_head]
    dfl = profile["deleted"][RM.L_FLOOR]
    # shuffled-label null: fit deleted-L_head on SHUFFLED discovery labels, test on conf (5 seeds)
    shuf_accs = []
    for sd in RM.SHUFFLE_NULL_SEEDS:
        ys = RM.shuffled_labels(y_disc, sd)
        r = run_probe(Hd_disc[L_head], ys, Hd_conf[L_head], y_conf)   # test vs TRUE conf labels → chance if no memorization
        shuf_accs.append(r["acc"])
    shuf_mean = float(np.mean(shuf_accs))

    # masking gap = deleted-L16 probe − deleted-MCQ, paired per item (mmlu_blankimg per_item is same
    # (sorted-subject, row, is_confirmation) order as build_confirmation_items → item-aligned).
    mcq_pi = [int(x) for x in mcq_del["per_item"]]
    assert len(mcq_pi) == len(d16["correct"]), f"MCQ per_item {len(mcq_pi)} != probe items {len(d16['correct'])}"
    masking_gap, mask_lo, mask_hi = RM.bootstrap_gap(d16["correct"], mcq_pi)
    comp_gap, comp_lo, comp_hi = RM.bootstrap_gap(d16["correct"], dfl["correct"])
    ceil_gap, ceil_lo, ceil_hi = RM.bootstrap_gap(i16["correct"], d16["correct"])
    d16_pt, d16_lo, d16_hi = RM.bootstrap_acc(d16["correct"])

    # ---- data-efficiency at L_head (deleted), preserved-vs-repriced. C HELD FIXED at the headline-selected
    #      C (only train size varies; dodges tiny-n StratifiedKFold degeneracy — prereg §6/§7). ----
    def data_eff(Xd, yd, Xc, yc, fixed_c):
        def fit_acc(idx):
            sc, clf, _, _ = fit_probe(Xd[idx], yd[idx], RM.C_GRID, fixed_c=fixed_c)
            return score_probe(sc, clf, Xc, yc)[0]
        acc_by = {}
        for ntr in RM.DATA_EFF_SIZES:
            if ntr > len(yd):
                continue
            accs = [fit_acc(np.random.default_rng(seed).permutation(len(yd))[:ntr]) for seed in RM.PROBE_SEEDS]
            acc_by[ntr] = float(np.mean(accs))
        acc_by["all"] = fit_acc(np.arange(len(yd)))
        return acc_by
    de_del = data_eff(Hd_disc[L_head], y_disc, Hd_conf[L_head], y_conf, d16["C"])
    sat_del = RM.saturation_index(de_del, small_key=100)

    # ---- readout-alignment (rough cross-layer descriptor, prereg §5 threat 5): probe class direction in
    #      residual space (coef/scale) vs each model's choice-letter unembedding row. ----
    def align(model, prof):
        U = model.get_output_embeddings().weight.detach().float().cpu().numpy()[np.array(letter_ids)]
        W = prof["clf"].coef_ / prof["scaler"].scale_    # (n_classes, d) in raw residual space
        return RM.readout_alignment(W, U)
    align_del = align(pg_after, d16)
    align_int = align(pg_before, i16)

    # ---- verdict (prereg §6) ----
    shuffled_ok = shuf_mean <= RM.CHANCE + RM.SHUFFLE_NULL_TOL
    beats_mcq = mask_lo > RM.DELTA_DECODE
    beats_floor = comp_lo > RM.DELTA_DECODE
    near_ceiling = d16_lo > (i16["acc"] - RM.GAMMA_CEILING)          # deleted CI-lo within γ of ceiling
    partial_below = d16_hi < (i16["acc"] - RM.GAMMA_CEILING)         # deleted CI-hi clearly below ceiling−γ (spec §6 H_partial)
    below_ceiling = ceil_lo > RM.DELTA_DECODE
    at_floor = comp_hi < RM.DELTA_DECODE
    if not preflight_pass:
        verdict = "UNINTERPRETABLE_preflight_failed"
    elif not shuffled_ok:
        verdict = "H_artifact"
    elif beats_mcq and beats_floor and near_ceiling:
        verdict = "H_mask"
    elif at_floor and below_ceiling:
        verdict = "H_loss"
    elif beats_floor and partial_below:
        verdict = "H_partial"
    else:
        verdict = "ambiguous"

    row = {
        "study": "rmask", "firewall_n_layers": n_dec, "attn_impl": "eager",
        "d_model": int(Hd_disc[layer_set[0]].shape[1]), "layer_set": list(layer_set),
        "L_headline": L_head, "L_floor": RM.L_FLOOR, "preflight_pass": preflight_pass,
        "chance": RM.CHANCE, "delta_decode": RM.DELTA_DECODE, "gamma_ceiling": RM.GAMMA_CEILING,
        "n_disc": len(disc_items), "n_conf": len(conf_items),
        "mcq": {"deleted": del_mcq, "intact": int_mcq},
        "profile_acc": {m: {int(L): profile[m][L]["acc"] for L in layer_set} for m in ("deleted", "intact")},
        "profile_cv": {m: {int(L): profile[m][L]["cv"] for L in layer_set} for m in ("deleted", "intact")},
        "headline": {
            "deleted_L16_acc": d16_pt, "deleted_L16_ci": [d16_lo, d16_hi],
            "intact_L16_acc": i16["acc"], "deleted_floor_acc": dfl["acc"], "shuffled_null_mean": shuf_mean,
            "shuffled_null_seeds": [float(x) for x in shuf_accs],
            "masking_gap": masking_gap, "masking_gap_ci": [mask_lo, mask_hi],
            "computation_gap": comp_gap, "computation_gap_ci": [comp_lo, comp_hi],
            "ceiling_gap": ceil_gap, "ceiling_gap_ci": [ceil_lo, ceil_hi],
            "beats_mcq_by_delta": beats_mcq, "beats_floor_by_delta": beats_floor,
            "near_ceiling": near_ceiling, "partial_below_ceiling": partial_below,
            "at_floor": at_floor, "below_ceiling": below_ceiling, "shuffled_ok": shuffled_ok},
        "data_efficiency_deleted": de_del, "saturation_index_deleted": sat_del,
        "readout_alignment": {"deleted": align_del, "intact": align_int},
        "raw": {"deleted_L16_correct": d16["correct"], "deleted_floor_correct": dfl["correct"],
                "intact_L16_correct": i16["correct"], "conf_item_hash": [it["hash"] for it in conf_items],
                "gold": [int(x) for x in y_conf]},
        "verdict": verdict, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / "rmask.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[rmask] VERDICT={verdict}  deleted-L{L_head} acc={d16_pt:.3f} CI[{d16_lo:.3f},{d16_hi:.3f}] "
          f"| masking_gap={masking_gap:+.3f} comp_gap={comp_gap:+.3f} ceil_gap={ceil_gap:+.3f} "
          f"| shuffled={shuf_mean:.3f} sat={sat_del} | banked → {out/'rmask.json'}")


if __name__ == "__main__":
    main()
