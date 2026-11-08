"""LATMECH preflight — the §9.3 fail-fast FIRST-BILLABLE GPU step + the content-carrier build.

Order (gate the contrast behind it):
  1. Reproduce a KNOWN number — deleted (pg_after) MCQ at ~chance + cloze PRESERVED, ceiling
     (base-LM) MCQ lifted (the REPRICE/W17 anchor; recompute, gate on positive lift not an
     absolute), proving the harness is wired.
  2. Build the cloze-content carrier ONCE (latmech_carrier) and SAVE carrier.pt.
  3. The energy-controlled VALIDITY GATE: content-ablation DROPS cloze by ≥ thresh while the
     energy-matched-random ablation drops it by < thresh. HALT (no try/except) on failure ⇒
     INSTRUMENT-NULL (the carrier ID is invalid, do not interpret the MDE).
  4. Lobotomy dynamic-range: reinit one mid layer drops cloze (the cloze axis CAN move).
  5. Trainer smoke + zero-delta (n=0 attach changes nothing).

    python -m infra.aws_pg.jobs.latmech_preflight --out <dir> --carrier-k 128 --svd-q 512 \
        --grad-subjects 8 --grad-per-subject 8 --eval-per-subject 4 --cloze-drop-thresh 0.10 \
        --seed 1337
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--carrier-k-sweep", default="8,16,32,64,128,256",
                    help="comma list of carrier-k to sweep on discovery; the §6 gate picks the "
                         "smallest k that separates content from energy-matched-random")
    ap.add_argument("--svd-q", type=int, default=512)
    ap.add_argument("--grad-subjects", type=int, default=8)
    ap.add_argument("--grad-per-subject", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=4)
    ap.add_argument("--eval-per-subject", type=int, default=4)
    ap.add_argument("--cloze-drop-thresh", type=float, default=0.10,
                    help="discovery-calibrated validity-gate threshold (frozen before the sealed half)")
    ap.add_argument("--lobotomy-layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from transformers import AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import w14_localize as W14
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15
    from infra.aws_pg.jobs.e4_compress_curve import _set_decoder

    assert torch.cuda.is_available(), "latmech_preflight needs GPU."
    device = "cuda"
    pred = W.is_grid_subset           # preflight uses the discovery bucket (sealed half untouched)
    n_eval = args.eval_per_subject
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    dkeys = W14.decoder_keys()
    rep = {"args": vars(args), "dkeys_n": len(dkeys)}

    # ---- 1. known-number reproduction (deleted vs ceiling) ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    mcq_deleted = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_eval, item_filter=pred)
    clz_deleted = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=n_eval,
                                item_filter=pred)
    mcq_ceiling = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_eval, item_filter=pred)
    clz_ceiling = TE.mmlu_cloze(pg_before, tok, device, processor=proc, n_per_subject=n_eval,
                                item_filter=pred)
    rep["anchor"] = {
        "mcq_deleted": round(100 * sum(mcq_deleted["per_item"]) / len(mcq_deleted["per_item"]), 2),
        "cloze_deleted": clz_deleted["acc_pct"],
        "mcq_ceiling": round(100 * sum(mcq_ceiling["per_item"]) / len(mcq_ceiling["per_item"]), 2),
        "cloze_ceiling": clz_ceiling["acc_pct"], "n_items": clz_deleted["n"],
    }
    # REPRICE anchor: content preserved at the deleted (head-only) decoder, ceiling lifts MCQ.
    assert rep["anchor"]["mcq_ceiling"] > rep["anchor"]["mcq_deleted"], \
        f"ceiling MCQ must beat deleted MCQ (anchor broken): {rep['anchor']}"
    assert rep["anchor"]["cloze_deleted"] > 25.0, \
        f"deleted cloze must exceed chance (content preserved per REPRICE): {rep['anchor']}"
    del pg_before
    torch.cuda.empty_cache()

    # ---- 2. build the carrier factors ONCE (gradient + SVD + content scores) ----
    full, gloss = S15.latmech_carrier_factors(
        pg_after, proc, paths, dkeys, svd_q=args.svd_q,
        grad_subjects=args.grad_subjects, grad_per_subject=args.grad_per_subject,
        batch=args.grad_batch, item_filter=pred, seed=args.seed, device=device)
    rep["carrier"] = {"grad_loss": gloss, "svd_q": args.svd_q}

    # ---- 3. carrier-k SWEEP → energy-controlled CONTRAST gate (properly powered; §3.1/§8-T2 intent) ----
    #
    # The §6 validity criterion is "content-ablation removes cloze content BEYOND the energy-matched
    # random null". Operationalized as the paired per-item CONTRAST (emrand_cloze − content_cloze) with
    # a bootstrap CI — NOT a single absolute threshold, which conflates the carrier's SPECIFICITY
    # (content beats emrand) with its MAGNITUDE (how much cloze it removes). Banks per-k drops + the
    # contrast CI; picks the smallest k whose contrast CI excludes 0 (content removes more cloze than
    # emrand beyond noise). The absolute content drop is reported too (the carrier MAGNITUDE = the MDE's
    # dynamic range). Does NOT hard-halt — the validity call is made from the banked CIs (a discovery
    # calibration record), and the cell smokes run regardless. INSTRUMENT-NULL iff no k has CI>0.
    import numpy as np
    from infra.pg import latmech as Lm
    from infra.pg import reprice_stats as RS
    lm = pg_after.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone() for k, v in lm.state_dict().items()}
    cloze0 = clz_deleted["acc_pct"]
    del_h = clz_deleted["per_item_hash"]
    k_grid = [int(x) for x in args.carrier_k_sweep.split(",")]
    sweep, factors_by_k = [], {}
    for kk in k_grid:
        fk = S15.select_carrier_factors(full, dkeys, kk, args.seed)
        factors_by_k[kk] = fk
        S15.apply_carrier(lm, base_sd, dkeys, fk, "carrier", device)
        cc = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=n_eval, item_filter=pred)
        _set_decoder(lm, base_sd, {}, device)                 # restore the deleted decoder
        S15.apply_carrier(lm, base_sd, dkeys, fk, "emrand", device)
        ce = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=n_eval, item_filter=pred)
        _set_decoder(lm, base_sd, {}, device)
        Lm.position_align_tripwire(del_h, cc["per_item_hash"])
        Lm.position_align_tripwire(del_h, ce["per_item_hash"])
        c_pi = np.array(cc["per_item"], float); e_pi = np.array(ce["per_item"], float)
        # contrast = mean(emrand cloze) − mean(content cloze); CI>0 ⇒ content removes MORE cloze (specific).
        ctr = RS.paired_cluster_bootstrap([e_pi, c_pi], lambda a: a[0].mean() - a[1].mean(),
                                          n=4000, seed=args.seed)
        sweep.append({"k": kk, "cloze_content": cc["acc_pct"], "cloze_emrand": ce["acc_pct"],
                      "drop_content": (cloze0 - cc["acc_pct"]) / 100.0,
                      "drop_emrand": (cloze0 - ce["acc_pct"]) / 100.0,
                      "contrast_point": ctr["point"], "contrast_lo": ctr["lo"], "contrast_hi": ctr["hi"],
                      "n_items": len(c_pi), "removed_E": sum(fk[k]["removed_E"] for k in dkeys),
                      "emrand_E": sum(fk[k]["emrand_E"] for k in dkeys)})
        print(f"[latmech-preflight] k={kk}: cloze c {cc['acc_pct']:.2f} e {ce['acc_pct']:.2f} "
              f"contrast {ctr['point']:.3f} [{ctr['lo']:.3f},{ctr['hi']:.3f}] "
              f"drop_c {sweep[-1]['drop_content']:.3f}", flush=True)

    valid = [s for s in sweep if s["contrast_lo"] > 0]               # specificity beyond noise
    chosen_k = min((s["k"] for s in valid), default=None)
    best_k = max(sweep, key=lambda s: s["contrast_point"])["k"]      # fallback carrier.pt to save
    save_k = chosen_k if chosen_k is not None else best_k
    torch.save(factors_by_k[save_k], out / "carrier.pt")
    rep["validity_sweep"] = {"cloze_deleted": cloze0, "n_items": clz_deleted["n"], "sweep": sweep,
                             "gate": "contrast_ci_excludes_0", "chosen_k": chosen_k,
                             "best_contrast_k": best_k, "saved_k": save_k,
                             "abs_thresh_legacy": args.cloze_drop_thresh}
    rep["carrier"].update({"chosen_k": chosen_k, "saved_k": save_k})
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[latmech-preflight] GATE: chosen_k={chosen_k} (contrast CI>0) saved_k={save_k}", flush=True)

    # ---- 4. lobotomy dynamic range (cloze CAN drop) ----
    W.reinit_lm_layer_(lm, args.lobotomy_layer, seed=args.seed)
    cloze_lobo = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=n_eval,
                               item_filter=pred)["acc_pct"]
    _set_decoder(lm, base_sd, {}, device)
    rep["dynamic_range"] = {"cloze_lobotomy": cloze_lobo,
                            "drop": (cloze0 - cloze_lobo) / 100.0,
                            "has_range": bool(cloze_lobo < cloze0)}
    assert rep["dynamic_range"]["has_range"], \
        f"no cloze dynamic range (lobotomy did not drop cloze): {rep['dynamic_range']}"

    # ---- 5. trainer smoke + zero-delta ----
    from infra.pg import relearn as Rl
    arep = Rl.attach_adapters(lm, rank=64, seed=args.seed, alpha=64.0)
    pg_after.to(device)
    mcq_zero = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_eval, item_filter=pred)
    zero_delta = round(100 * sum(mcq_zero["per_item"]) / len(mcq_zero["per_item"]), 2)
    rep["trainer_smoke"] = {"adapters": arep["n_modules"], "trainable": arep["trainable_params"],
                            "mcq_zero_step": zero_delta, "mcq_deleted": rep["anchor"]["mcq_deleted"]}
    assert abs(zero_delta - rep["anchor"]["mcq_deleted"]) < 1.0, \
        f"zero-step adapters changed MCQ (B≠0 init?): {zero_delta} vs {rep['anchor']['mcq_deleted']}"

    rep["torch"] = torch.__version__
    rep["gpu"] = torch.cuda.get_device_name(0)
    rep["PREFLIGHT_OK"] = True
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[LATMECH-PREFLIGHT] OK — anchor {rep['anchor']} chosen_k={rep['validity_sweep']['chosen_k']}")


if __name__ == "__main__":
    main()
