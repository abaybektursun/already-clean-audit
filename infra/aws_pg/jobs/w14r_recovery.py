"""W14-R — deletion-vs-suppression RECOVERY GATE orchestration (AWS pg, inference-only).

Runs the frozen 18-transform set (infra.pg.recovery_transforms) on the adapted PaliGemma-2 LM's
MMLU, INFERENCE-ONLY, weights untouched, and reads max_t R_recover against floor, the base
positive control, and the W11 weight-swap restore. Primary forward = TEXT-ONLY (the validated W11
instrument); SECONDARY = blank-image (gray 224²). See the execution journal
`studies/lineage/paligemma2/research/2026-06-02_w14r_execution_journal.md` (D1/D2/D3, C1–C7).

    pg run infra.aws_pg.jobs.w14r_recovery --type g6e.xlarge --args \
        "--stage preflight --forward both --n-per-subject 20"

Stages (`--stage`, runbook §9 strict order, fail-loud):
  manifest   build all prompts on the real MMLU+MMMLU, write data_manifest.json (no model).
  preflight  (a) IDENTITY reproduces the legA anchors (after/before/floor) within tol;
             (b) base positive control: every transform on pg_before ≈ full, disqualify any that
                 lowers it past the margin; δ_rec = the base recovery spread; (c) swap-restore
                 (=pg_before) reproduces; (d) τ_rec=0.5 > δ_rec. Writes preflight.json + controls.json.
  discovery  per-transform R on pg_after, discovery sha1-half. No thresholds touched.
  unblind    confirmation half + held-out: max-statistic bootstrap + the §6 verdict. metrics.json.

REUSED W11 instruments (verified in-repo): infra.pg.recovery_eval (load_mmlu + score_prompts[_blankimg],
factored from text_eval.mmlu_accuracy / w11_legA_diagnostic.mmlu_blankimg);
infra.pg.diff_harvest.build_before_after / build_random_after (AFTER / BEFORE=swap-restore / FLOOR).
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

from infra.pg import recovery_transforms as RT
from infra.pg import recovery_eval as RE

# §4 frozen held-out subjects (10, spanning the 4 MMLU categories).
HELD_OUT = ["college_chemistry", "astronomy", "machine_learning", "college_mathematics",
            "philosophy", "high_school_world_history", "professional_law", "econometrics",
            "us_foreign_policy", "nutrition"]

# legA text-only anchors (recompute pointers; reproduced at preflight, not trusted as prose).
LEGA = {"after": 0.25424, "before": 0.51674, "floor": 0.24676}          # source: legA_unblind/controls
LEGA_BLANKIMG = {"after": 0.2386, "before": 0.4883, "floor": 0.2708}    # source: legA_diagnostic_v3


def _sha1_bucket(h, halves=2):
    """Deterministic discovery/confirmation split by item hash (parent sha1-bucket discipline)."""
    return int(h, 16) % halves


def _exemplars_for(t, item, dev_exemplars, test_by_subject, n_shots):
    """C7: translation transforms get same-subject TRANSLATED test items as shots (in-language,
    Kotha-faithful, excl. the scored item by its unique gidx); all others get the English dev shots
    (W11-identical, so IDENTITY reproduces the anchor)."""
    if t[2] in RT.CONJ_LANG:
        pool = [x for x in test_by_subject[item["subject"]] if x["gidx"] != item["gidx"]]
        assert len(pool) >= n_shots, (f"{t[0]}/{item['subject']}: only {len(pool)} same-subject "
                                      f"translated shots (need {n_shots}); raise --n-per-subject")
        return pool[:n_shots]
    return dev_exemplars.get(item["subject"], [])[:n_shots]


def score_transform(model, proc, text_tok, device, items, t, dev_exemplars, test_by_subject, forward,
                    n_shots=5, batch_size=4):
    """Build the per-item prompts for transform `t` and score them on `model` under `forward`
    ('text'|'blank'). Returns a per-item 0/1 LIST aligned by POSITION to `items` (collision-proof —
    the item hash is not unique; gidx/position is the pairing key).

    text → `text_tok` = the base gemma-2-2b tokenizer (prepends the BOS the decoder needs; the
    PaliGemma processor tokenizer does NOT add BOS → few-shot collapses to ~0-shot, journal C11).
    blank → the full `proc` (inserts image tokens + BOS), W11 `mmlu_blankimg`-faithful."""
    surfaces = RT.answer_candidates(t)
    prompts, gold = [], []
    for it in items:
        ex = _exemplars_for(t, it, dev_exemplars, test_by_subject, n_shots)
        prompts.append(RT.apply_transform(it, t, ex))
        gold.append(it["answer"])
    if forward == "text":
        return RE.score_prompts(model, text_tok, device, prompts, surfaces, gold, batch_size)
    if forward == "blank":
        return RE.score_prompts_blankimg(model, proc, device, prompts, surfaces, gold, batch_size)
    raise ValueError(f"unknown forward {forward!r}")


def _acc(per_list):
    return sum(per_list) / len(per_list) if per_list else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("manifest", "preflight", "discovery", "unblind"), required=True)
    ap.add_argument("--forward", choices=("text", "blank", "both"), default="text")
    ap.add_argument("--n-score", type=int, default=1500,
                    help="SIZE-WEIGHTED global subsample of scored items (seeded); -1=full. NOT per-subject "
                         "balanced — balancing shrinks the base-floor gap (journal: before=0.305 artifact).")
    ap.add_argument("--n-score-held", type=int, default=600, help="size-weighted subsample of held-out items")
    ap.add_argument("--frac", type=float, default=0.5, help="τ_rec = frac × R_swap-restore (frozen 0.5)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-shots", type=int, default=5)
    ap.add_argument("--anchor-tol", type=float, default=0.03, help="abs tol for the IDENTITY anchor gate")
    ap.add_argument("--base-drop-margin", type=float, default=0.10,
                    help="disqualify a transform if base acc drops > this below identity")
    ap.add_argument("--controls-s3", default="", help="S3 URI of the frozen preflight controls.json (unblind)")
    ap.add_argument("--freeze-sha", default="")
    args = ap.parse_args()
    forwards = ["text", "blank"] if args.forward == "both" else [args.forward]

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    transforms = RT.transform_set()

    import torch
    from infra.pg import diff_harvest as H
    H.hf_online_setup()                                    # HF_TOKEN + online (gated model + datasets)

    print(f"[W14R] stage={args.stage} forward={args.forward} n_score={args.n_score} "
          f"transforms={len(transforms)} freeze_sha={args.freeze_sha}")
    import numpy as np
    dev_items, held_items, dev_exemplars = RE.load_mmlu(HELD_OUT, n_per_subject=None,
                                                        langs=tuple(RT.CONJ_LANG.values()), n_shots=args.n_shots)
    test_by_subject = defaultdict(list)                  # FULL pool for translation few-shot (pre-subsample)
    for it in dev_items + held_items:
        test_by_subject[it["subject"]].append(it)

    _rng = np.random.default_rng(args.seed)
    def _sub(items, n):                                  # size-weighted seeded subsample of SCORED items
        if n < 0 or n >= len(items):
            return items
        return [items[i] for i in sorted(_rng.choice(len(items), n, replace=False))]
    print(f"[W14R] loaded dev={len(dev_items)} held={len(held_items)} subjects={len(test_by_subject)}")

    if args.stage == "manifest":
        manifest = {"transform_ids": [t[0] for t in transforms], "n_dev": len(dev_items),
                    "n_held_out": len(held_items), "held_out_subjects": HELD_OUT,
                    "langs": list(RT.CONJ_LANG.values()), "n_shots": args.n_shots,
                    "item_hashes": sorted(it["hash"] for it in dev_items + held_items)}
        (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[W14R-MAN] {len(transforms)} transforms × {len(dev_items)+len(held_items)} items; manifest written.")
        return

    device = "cuda"
    print(f"[W14R] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    pg_after, pg_before, proc, paths, _img_tok, gate = H.build_before_after(device)
    pg_random = H.build_random_after(pg_before, paths, device, seed=args.seed)
    print(f"[W14R] swap_gate={json.dumps(gate, default=str)[:200]}")
    from transformers import AutoTokenizer
    from infra.pg.entities import ENTITIES
    gemma_tok = AutoTokenizer.from_pretrained(ENTITIES["L0"].repo_id, revision=ENTITIES["L0"].revision)
    print(f"[W14R] text tokenizer = {ENTITIES['L0'].repo_id} (BOS={gemma_tok.bos_token_id})")
    models = {"after": pg_after, "before": pg_before, "floor": pg_random}
    ident = next(t for t in transforms if t[0] == "identity")

    if args.stage == "preflight":
        pre_items = _sub(dev_items, args.n_score)            # size-weighted (reproduces the legA anchors)
        print(f"[W14R-PRE] scoring {len(pre_items)} size-weighted items")
        result = {"forward_anchors": {}, "delta_rec": {}, "tau_rec": {}, "disqualified": {},
                  "n_scored": len(pre_items), "swap_gate": gate, "anchor_tol": args.anchor_tol,
                  "PREFLIGHT_OK": True}
        for fwd in forwards:
            ref = LEGA if fwd == "text" else LEGA_BLANKIMG
            # (a)+(c) IDENTITY reproduces after/before/floor
            acc_id = {st: _acc(score_transform(m, proc, gemma_tok, device, pre_items, ident, dev_exemplars,
                                               test_by_subject, fwd, args.n_shots)) for st, m in models.items()}
            print(f"[W14R-PRE][{fwd}] IDENTITY after={acc_id['after']:.4f} before={acc_id['before']:.4f} "
                  f"floor={acc_id['floor']:.4f}  (ref {ref})")
            anchor_ok = all(abs(acc_id[k] - ref[k]) <= args.anchor_tol for k in ref)
            # (b) base positive control + δ_rec: every transform on pg_before
            base_acc = {t[0]: _acc(score_transform(pg_before, proc, gemma_tok, device, pre_items, t, dev_exemplars,
                                                   test_by_subject, fwd, args.n_shots)) for t in transforms}
            base_id = acc_id["before"]; floor = acc_id["floor"]
            disq = [tid for tid, a in base_acc.items() if a < base_id - args.base_drop_margin]
            kept = {tid: a for tid, a in base_acc.items() if tid not in disq}
            delta = RT.delta_rec(kept, base_id, floor)
            # (d) swap restore (=before) → R_swap=1.0 → τ_rec
            swap_r = RT.r_recover(base_id, base_id, floor)
            tau = RT.tau_rec(swap_r, args.frac)
            tau_ok = tau > delta
            print(f"[W14R-PRE][{fwd}] anchor_ok={anchor_ok} δ_rec={delta:.3f} τ_rec={tau:.3f} "
                  f"tau>δ={tau_ok} disq={disq}")
            result["forward_anchors"][fwd] = {"identity_acc": acc_id, "ref": ref, "anchor_ok": anchor_ok,
                                              "base_acc_by_transform": base_acc, "swap_restore_r": swap_r}
            result["delta_rec"][fwd] = delta; result["tau_rec"][fwd] = tau
            result["disqualified"][fwd] = disq
            result["PREFLIGHT_OK"] = result["PREFLIGHT_OK"] and anchor_ok and tau_ok
        (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
        (out / "controls.json").write_text(json.dumps(
            {"delta_rec": result["delta_rec"], "tau_rec": result["tau_rec"],
             "disqualified": result["disqualified"], "frac": args.frac, "anchors": LEGA,
             "anchors_blankimg": LEGA_BLANKIMG, "freeze_sha": args.freeze_sha}, indent=2))
        print(f"[W14R-PRE] PREFLIGHT_OK={result['PREFLIGHT_OK']}")
        if not result["PREFLIGHT_OK"]:
            raise RuntimeError("PREFLIGHT HALT — see preflight.json (anchor or τ>δ gate failed).")
        return

    # discovery / unblind share the adapted scoring spine: the paired max-statistic bootstrap of
    # max_t R_recover over `items`, denominator = pg_before identity per-item, floor = the legA anchor.
    def _bootstrap(items, fwd):
        floor = LEGA["floor"] if fwd == "text" else LEGA_BLANKIMG["floor"]
        base_correct = score_transform(pg_before, proc, gemma_tok, device, items, ident, dev_exemplars,
                                       test_by_subject, fwd, args.n_shots)          # list, items-aligned
        cbt = {t[0]: score_transform(pg_after, proc, gemma_tok, device, items, t, dev_exemplars,
                                     test_by_subject, fwd, args.n_shots) for t in transforms}
        return RT.max_stat_bootstrap(cbt, base_correct, floor, n_boot=args.n_boot, seed=args.seed)

    if args.stage == "discovery":
        half = _sub([it for it in dev_items if _sha1_bucket(it["hash"]) == 0], args.n_score)  # discovery half
        result = {"stage": "discovery", "n_items": len(half),
                  "per_transform": {fwd: _bootstrap(half, fwd) for fwd in forwards}}
        for fwd in forwards:
            b = result["per_transform"][fwd]
            print(f"[W14R-DISC][{fwd}] max_t={b['max_transform']} max_R={b['max_r']:.3f} "
                  f"CI=[{b['ci_lo']:.3f},{b['ci_hi']:.3f}]")
        (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
        return

    # unblind (ONCE): confirmation sha1-half + the §4 held-out set; read the FROZEN preflight controls
    # (δ_rec, τ_rec) — never recomputed here; apply the §6 verdict on the max-statistic CI bounds.
    assert args.controls_s3, "unblind needs --controls-s3 (the frozen preflight controls.json)"
    import subprocess
    ctl_local = out / "controls_frozen.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctl_local), "--region", "us-east-1"], check=True)
    controls = json.loads(ctl_local.read_text())
    conf = _sub([it for it in dev_items if _sha1_bucket(it["hash"]) == 1], args.n_score)
    held_score = _sub(held_items, args.n_score_held)
    print(f"[W14R-UNBLIND] confirmation={len(conf)} held_out={len(held_score)} (reading frozen controls)")
    result = {"stage": "unblind", "controls_s3": args.controls_s3, "freeze_sha": args.freeze_sha,
              "verdict": {}, "confirmation": {}, "held_out": {}, "grid": {}}
    for fwd in forwards:
        delta = float(controls["delta_rec"][fwd]); tau = float(controls["tau_rec"][fwd])
        cboot = _bootstrap(conf, fwd)
        hboot = _bootstrap(held_score, fwd)
        v = RT.verdict(cboot["ci_lo"], cboot["ci_hi"], hboot["ci_lo"],
                       swap_restore_r=1.0, base_spread=delta, frac=args.frac)
        result["confirmation"][fwd] = cboot
        result["held_out"][fwd] = hboot
        result["verdict"][fwd] = v
        result["grid"][fwd] = {"max_t_R_recover_ci": [cboot["ci_lo"], cboot["max_r"], cboot["ci_hi"]],
                               "held_out_R_ci": [hboot["ci_lo"], hboot["max_r"], hboot["ci_hi"]],
                               "max_transform": cboot["max_transform"], "tau_rec": tau, "delta_rec": delta}
        print(f"[W14R-UNBLIND][{fwd}] VERDICT={v}  max_R={cboot['max_r']:.3f} "
              f"CI=[{cboot['ci_lo']:.3f},{cboot['ci_hi']:.3f}] heldout_lo={hboot['ci_lo']:.3f} "
              f"τ={tau:.3f} δ={delta:.3f}")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W14R-UNBLIND] DONE — verdict(text)={result['verdict'].get('text')}")


if __name__ == "__main__":
    main()
