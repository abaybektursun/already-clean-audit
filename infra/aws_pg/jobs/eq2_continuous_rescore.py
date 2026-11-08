"""E-Q2 — continuous-metric re-score of the no-retrain edit family (GraftLM trained-exploit prereg
research/2026-05-30_graftlm_trained_exploit_prereg.md §9 step 1, the cheap gate).

PURPOSE: rule out the discontinuous-metric confound. The capability readout scored RefCOCO with
acc@0.5 (a 0/1 IoU threshold) and found a 0% no-retrain floor. Does a CONTINUOUS metric reveal smooth
sub-threshold recovery (H_metric — the cliff is partly a threshold artifact), or is it ALSO ~0 because
the model emits no <loc> boxes to partially credit (H_policy — genuine generation-policy
non-installation)? Strong prior from the readout (0/12 <loc> at rank<=512): H_policy.
This is NOT "invoking Schaeffer" — Schaeffer's mirage needs a latent rising signal; 0/12 emission is
genuine absence (his excluded case). E-Q2 rules the confound OUT.

The frozen sweep stored only 12 gens/config (e4_capability_sweep.json) -> we REGENERATE. Generation is
greedy + seed-1337 + the same edit family -> BIT-IDENTICAL to the frozen sweep, a faithful re-analysis.
This job also PERSISTS the full per-example table (the sweep did not) so no future regeneration is
needed.

Re-analysis of the frozen no-retrain edit (no new TRAINED model) -> cannot HARK the E-Q1 trained
contrast; safe to run pre-freeze.

Metrics per RefCOCO example, per config:
  - emitted        : did the greedy gen parse >=4 <loc> tokens? (the policy-flip / format indicator)
  - iou_cont       : continuous IoU (no 0.5 threshold); 0 if not emitted
  - acc05          : the readout's thresholded acc (for the side-by-side cliff plot)
  - coord_l1       : mean |pred-gt| over the 4 normalized box coords (precision | emitted)
  - digit_acc      : per-digit exact-match of the 4 <locDDDD> 4-digit strings, 16 digits (precision | emitted)

Decomposition reported: R_cap(continuous-IoU) vs R_cap(acc@0.5) vs emission-fraction across the rank
ladder. H_metric if continuous-IoU R_cap rises smoothly while acc05 stays 0; H_policy if continuous-IoU
R_cap ~ 0 wherever emission ~ 0.

    pg run infra.aws_pg.jobs.eq2_continuous_rescore --args="--n=1975" \
        --type g6e.8xlarge,p4d.24xlarge --s3-prefix results/eq2_continuous_rescore --budget-min 180
    # smoke (wiring + a couple configs):
    pg run infra.aws_pg.jobs.eq2_continuous_rescore --args="--smoke" --type g6e.8xlarge \
        --s3-prefix results/eq2_smoke --budget-min 45
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.e4_capability import _build_edit_family, _eval_rows, _generate, SEED, N_BOOT
from infra.aws_pg.jobs.e4_compress_curve import _set_decoder

# the cliff-decomposition subset of the edit family (FLOOR + rank ladder + CEILING); E_scale/N3 nulls
# are already characterized in the sweep and not needed for the metric-vs-policy decomposition.
EQ2_CONFIGS = ("FLOOR", "rank8", "rank32", "rank128", "rank512", "rank1024", "CEILING")
N_FULL = 1975          # RefCOCO testA full split (matches the frozen sweep manifest)


def _loc_digits(text):
    """The 4-digit strings of the first 4 <locDDDD> tokens, or None if <4."""
    from infra.pg.battery_tasks import _LOC_RE
    locs = _LOC_RE.findall(text)
    return locs[:4] if len(locs) >= 4 else None


def _continuous_scores(rows, preds):
    """Per-example continuous metrics for one config's RefCOCO generations."""
    import numpy as np
    from infra.pg import battery_tasks as B
    emitted, iou_cont, acc05, coord_l1, digit_acc = [], [], [], [], []
    for r, p in zip(rows, preds):
        box = B._decode_loc_box(p)
        gt = B._refcoco_gt_box(r)
        emit = box is not None
        emitted.append(1.0 if emit else 0.0)
        iou = B._iou(box, gt) if emit else 0.0
        iou_cont.append(iou)
        acc05.append(1.0 if (emit and iou >= 0.5) else 0.0)
        if emit:
            coord_l1.append(float(np.mean([abs(box[i] - gt[i]) for i in range(4)])))
            pd, gd = _loc_digits(p), B._refcoco_loc_str(gt)
            gd = B._LOC_RE.findall(gd)
            # per-digit exact match over the 16 digits of the 4 loc tokens
            digit_acc.append(float(np.mean([pd[i][j] == gd[i][j] for i in range(4) for j in range(4)])))
        else:
            coord_l1.append(float("nan"))
            digit_acc.append(float("nan"))
    return {"emitted": np.array(emitted), "iou_cont": np.array(iou_cont), "acc05": np.array(acc05),
            "coord_l1": np.array(coord_l1), "digit_acc": np.array(digit_acc)}


def _rcap_paired(scores_by_cfg, key, n_boot):
    """R_cap(cfg) = (m(cfg)-m(FLOOR))/(m(CEILING)-m(FLOOR)) for a per-item metric `key`, with a paired
    bootstrap over example indices (same draw across cfg/FLOOR/CEILING)."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    cfgs = list(scores_by_cfg.keys())
    n = scores_by_cfg[cfgs[0]][key].size
    m = {c: float(np.nanmean(scores_by_cfg[c][key])) for c in cfgs}
    denom = m["CEILING"] - m["FLOOR"]
    idx_draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for c in cfgs:
        rcap = (m[c] - m["FLOOR"]) / denom if denom > 0 else float("nan")
        rdraws = []
        for ids in idx_draws:
            mf = float(np.nanmean(scores_by_cfg["FLOOR"][key][ids]))
            mk = float(np.nanmean(scores_by_cfg["CEILING"][key][ids]))
            if mk > mf:
                rdraws.append((float(np.nanmean(scores_by_cfg[c][key][ids])) - mf) / (mk - mf))
        out[c] = {"metric": m[c],
                  "R_cap": rcap,
                  "R_cap_ci": ({"lo": float(np.percentile(rdraws, 2.5)),
                                "hi": float(np.percentile(rdraws, 97.5))} if rdraws else None)}
    return {"denom_ceiling_minus_floor": denom, "configs": out}


def _frac_ci(arr, n_boot):
    """Bootstrap mean + CI for a 0/1 array (emission fraction)."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    a = np.asarray(arr, dtype=float)
    draws = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return {"mean": float(a.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_FULL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--smoke", action="store_true", help="tiny N + {FLOOR,rank8,CEILING} — wiring")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "E-Q2 regeneration needs a GPU (greedy generation)"
    print(f"[eq2] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} smoke={a.smoke}", flush=True)

    task = B.REFCOCO_TASK
    n = 16 if a.smoke else a.n
    configs = ("FLOOR", "rank8", "CEILING") if a.smoke else EQ2_CONFIGS
    rows = _eval_rows(task, n, SEED)
    print(f"[eq2] task={task.name} split={task.eval_split} n={len(rows)} configs={configs}", flush=True)

    pg_after, pg_before, proc, paths, _img, gate = H.build_pair("base-pt", device)
    lm_edit, base_sd, after_sd, dkeys, builders, edited_params, ldesc = _build_edit_family(
        pg_after, pg_before, paths, device)
    del pg_after
    torch.cuda.empty_cache()
    by_name = {name: (edited, override) for name, edited, override in builders}

    scores_by_cfg, samples = {}, {}
    for cname in configs:
        edited, override = by_name[cname]
        _set_decoder(lm_edit, override, edited, device)
        preds = _generate(pg_before, proc, task, rows, device, a.batch)
        scores_by_cfg[cname] = _continuous_scores(rows, preds)
        s = scores_by_cfg[cname]
        samples[cname] = preds[:12]
        print(f"[eq2]   {cname:9s} emit={s['emitted'].mean():.3f} iou_cont={s['iou_cont'].mean():.4f} "
              f"acc05={s['acc05'].mean():.4f}  e.g. {preds[0][:64]!r}", flush=True)

    # decomposition: R_cap on continuous-IoU vs acc@0.5; emission fraction; precision among emitted
    rcap_iou = _rcap_paired(scores_by_cfg, "iou_cont", a.n_boot)
    rcap_acc = _rcap_paired(scores_by_cfg, "acc05", a.n_boot)
    emission = {c: _frac_ci(scores_by_cfg[c]["emitted"], a.n_boot) for c in configs}
    precision = {}
    for c in configs:
        em = scores_by_cfg[c]["emitted"].astype(bool)
        precision[c] = {"n_emitted": int(em.sum()),
                        "coord_l1_mean": (float(np.nanmean(scores_by_cfg[c]["coord_l1"])) if em.any() else None),
                        "digit_acc_mean": (float(np.nanmean(scores_by_cfg[c]["digit_acc"])) if em.any() else None)}

    results = {"experiment": "eq2_continuous_rescore", "status": "exploratory-descriptive",
               "purpose": "rule out the discontinuous-metric confound (H_metric vs H_policy)",
               "seed": SEED, "task": task.name, "split": task.eval_split, "n": len(rows),
               "configs": list(configs), "n_boot": a.n_boot,
               "provenance": {"axis": "base-pt", "edit_family": "no-retrain SVD of base->pt LM-decoder "
                              "delta (bit-identical to e4_capability sweep)", "swap_gate": gate,
                              "compare_sweep": "s3://pg-lineage-769063704696/results/e4_capreadout_sweep",
                              "L_desc_params": ldesc},
               "R_cap_iou_continuous": rcap_iou, "R_cap_acc05": rcap_acc,
               "emission_fraction": emission, "precision_among_emitted": precision,
               "samples": samples}
    (out / "eq2_continuous_rescore.json").write_text(json.dumps(results, indent=2))

    # persist the FULL per-example table (the sweep did not) -> no future regeneration needed
    import numpy as _np
    table = {c: {k: scores_by_cfg[c][k].tolist() for k in scores_by_cfg[c]} for c in configs}
    (out / "eq2_per_example.json").write_text(json.dumps({"task": task.name, "n": len(rows),
                                                           "configs": list(configs), "scores": table}))

    verdict = {c: {"emission": round(emission[c]["mean"], 3),
                   "R_cap_iou_cont": (round(rcap_iou["configs"][c]["R_cap"], 3)
                                      if rcap_iou["configs"][c]["R_cap"] == rcap_iou["configs"][c]["R_cap"] else None),
                   "R_cap_acc05": (round(rcap_acc["configs"][c]["R_cap"], 3)
                                   if rcap_acc["configs"][c]["R_cap"] == rcap_acc["configs"][c]["R_cap"] else None)}
               for c in configs}
    print("EQ2_OK", json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
