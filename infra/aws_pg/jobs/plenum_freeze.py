"""PLENUM freeze (spec §9 step 3) — the ONE job that turns the spec into a sealed instrument.

Stages, in order, each failing loud:
  ledger      every REUSED symbol in §13 is importable HERE, not "verified by ls" — a moved symbol
              fails at freeze rather than six hours into a GPU run
  roster      the 24 FTs re-derived from HF + shas pinned; both MDE units present
  recipes     CLAIM-LEVEL cite-verification of every recipe descriptor against the model card
              (`feedback_cite_verify_gate`: verify the MAPPING, not just that the repo exists). A
              card that does not substantiate the claim marks the unit `recipe_unverified` — it is
              still MEASURED, it just drops out of the law fit (§8 threat row).
  similarity  the frozen task-similarity predictor, computed PRE-MEASUREMENT from gemma-2b hidden
              states over frozen task descriptions, and sealed. Computing it later would let the
              measured grid influence the predictor it is supposed to be tested against.
  predict     per-cell predictions of ALL FOUR hypotheses, sealed before the first grid job (§7).
              The predictions are mechanical functions of the frozen descriptors, so "what H_recipe
              predicts" cannot drift after unblinding.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_freeze --type r7i.4xlarge --no-weights \
      --args "--stages ledger,roster,recipes,similarity,predict" \
      --s3-prefix results/plenum_freeze --budget-min 120
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib

# §13 reuse ledger, mechanized: (module, symbol). A missing symbol fails the freeze.
LEDGER = [
    ("infra.pg.lacuna_core", "label_row"), ("infra.pg.lacuna_core", "theta_null"),
    ("infra.pg.lacuna_core", "boot_S_paired"), ("infra.pg.lacuna_core", "bh_reject"),
    ("infra.pg.lacuna_core", "power_gate"), ("infra.pg.lacuna_core", "p_above_band"),
    ("infra.pg.lacuna_shell", "build_row_policy"), ("infra.pg.lacuna_shell", "install_canon"),
    ("infra.pg.lacuna_shell", "restore_pg1"), ("infra.pg.lacuna_shell", "mm_tasks"),
    ("infra.pg.lacuna_shell", "build_row_matched_dw"), ("infra.pg.lacuna_shell", "resolve_keys"),
    ("infra.pg.lacuna_units", "build_unit"), ("infra.pg.lacuna_units", "compartments"),
    ("infra.pg.inherit_pg1", "eval_pg1"), ("infra.pg.inherit_pg1", "load_items"),
    ("infra.pg.inherit_pg1", "ppl_gate"), ("infra.pg.inherit_pg1", "load_pg1"),
    ("infra.pg.battery_tasks", "bootstrap_S_ci"), ("infra.pg.battery_tasks", "_refcoco_score"),
    ("infra.pg.battery_tasks", "_textvqa_score"), ("infra.pg.battery_tasks", "_coco_caps"),
    ("infra.pg.savb", "loo_r2"), ("infra.pg.savb", "delta_loo_perm"),
    ("infra.pg.grain_bsf", "BlockTopKSAE"), ("infra.pg.grain_bsf", "train_block_sae"),
    ("infra.pg.grain_bsf", "block_quality"),
    ("infra.aws_pg.jobs.lacuna_mmcloze", "score_item"),
    ("infra.aws_pg.jobs.lacuna_a2_bytes", "_delta_census"),
    ("infra.pg.plenum_core", "choices_for_task"), ("infra.pg.plenum_core", "delta_loo_grouped"),
    ("infra.pg.plenum_core", "onset_statistic"), ("infra.pg.plenum_core", "sae_gates"),
    ("infra.pg.plenum_map", "all_cells"), ("infra.pg.plenum_units", "install_unit"),
]

# Frozen task descriptions — the ONLY text the similarity predictor sees. Written from the task
# definitions, not from any result, and sealed before measurement.
TASK_DESCRIPTIONS = {
    "vqa": "answering open-ended natural language questions about the content of a photograph",
    "ocr": "reading text that appears inside an image and answering questions about that text",
    "grounding": "localizing the image region a referring expression points to, as a bounding box",
    "caption": "producing a fluent natural language sentence describing the whole image",
    "text_recall": "answering multiple-choice knowledge questions from memory without an image",
    "reading": "answering a question whose answer is stated verbatim in an accompanying passage",
    "control": "predicting continuous robot action chunks from camera images and joint state",
}
FT_TASK_DESCRIPTIONS = {
    "ai2d": "answering multiple-choice questions about science diagrams",
    "aokvqa-da": "answering knowledge-intensive visual questions with a direct answer",
    "aokvqa-mc": "answering knowledge-intensive visual questions in multiple-choice form",
    "coco35l": "producing image captions in thirty-five languages",
    "cococap": "producing English image captions for photographs",
    "docvqa": "answering questions about scanned document images by reading their text",
    "gqa": "answering compositional questions about object relationships in a scene",
    "infovqa": "answering questions about infographics by reading their text and charts",
    "nlvr2": "judging whether a sentence is true of a pair of photographs",
    "ocrvqa": "answering questions about book covers by reading the printed text",
    "okvqa": "answering visual questions that require outside world knowledge",
    "refcoco-seg": "segmenting the image region a referring expression points to",
    "rsvqa-hr": "answering questions about high-resolution remote sensing imagery",
    "rsvqa-lr": "answering questions about low-resolution remote sensing imagery",
    "scicap": "captioning scientific figures using the text in the figure",
    "science-qa": "answering multiple-choice science questions with diagrams",
    "screen2words": "summarizing a mobile user interface screenshot in words",
    "stvqa": "answering questions about scene text photographed in the wild",
    "tallyqa": "counting how many instances of an object appear in an image",
    "textcaps": "captioning images in a way that reads and uses the text in them",
    "textvqa": "answering questions that require reading scene text in a photograph",
    "vizwizvqa": "answering visual questions asked by blind users about their own photos",
    "vqav2": "answering open-ended questions about the content of a photograph",
    "widgetcap": "captioning an individual user interface widget",
}
# Claim-level card verification: the token that must appear in the card for the descriptor's
# dataset claim to be substantiated. Verifying only that the repo EXISTS is what the cite-verify
# gate was written to stop.
CARD_TOKENS = {
    "ai2d": ["ai2d"], "aokvqa-da": ["a-okvqa", "aokvqa"], "aokvqa-mc": ["a-okvqa", "aokvqa"],
    "coco35l": ["coco-35l", "coco35l"], "cococap": ["coco captions", "cococap"],
    "docvqa": ["docvqa"], "gqa": ["gqa"], "infovqa": ["infovqa", "infographicvqa"],
    "nlvr2": ["nlvr2"], "ocrvqa": ["ocr-vqa", "ocrvqa"], "okvqa": ["okvqa"],
    "refcoco-seg": ["refcoco"], "rsvqa-hr": ["rsvqa"], "rsvqa-lr": ["rsvqa"],
    "scicap": ["scicap"], "science-qa": ["scienceqa", "science-qa", "science qa"],
    "screen2words": ["screen2words"], "stvqa": ["st-vqa", "stvqa"], "tallyqa": ["tallyqa"],
    "textcaps": ["textcaps"], "textvqa": ["textvqa"], "vizwizvqa": ["vizwiz"],
    "vqav2": ["vqav2", "vqa v2"], "widgetcap": ["widget"],
}

# PRE-FREEZE AMENDMENT (found by inspecting the sealed predictions before any grid job, 2026-07-28).
# H_similarity was written with an ABSOLUTE threshold (`retained iff similarity ≥ 0.5`). Mean-pooled
# gemma-2b cosines are anisotropic — every value landed in [0.895, 1.000] — so that rule labelled
# 168/168 cells `retained` and could not fail. The ORDERING is informative (each fine-tune peaks on
# its own faculty); the absolute scale is not.
#
# The rule is now a WITHIN-UNIT RANK: H_similarity predicts `retained` for the top-k faculties by
# similarity, where k is the number of law-faculties in that unit's recipe — the same k H_recipe
# predicts. The two hypotheses therefore predict the SAME NUMBER of retained cells per unit and
# differ only in WHICH, which is the contest that was meant: graded similarity vs binary membership,
# at matched sparsity, with no base-rate advantage to either.
#
# The REGRESSION predictor is unaffected: the law fit standardizes similarity per LOO fold, so the
# narrow range costs it nothing and ΔLOO(similarity) remains a real test.
SIM_RULE = "within-unit rank: top-k by similarity, k = |recipe ∩ law faculties|"
# H_fragility's content: a UNIT-INDEPENDENT faculty ordering. The ordering is the HYPOTHESIS's
# definition (what "intrinsic fragility" would mean on this battery), sealed pre-measurement. It is
# NOT a threshold imported from LACUNA for adjudicating PLENUM — no PLENUM verdict is decided by
# comparing to a LACUNA number (`feedback_no_post_unblind_rule_swap` firewall).
FRAGILITY_ORDER = ("vqa", "ocr", "text_recall", "reading", "caption", "grounding")


def _mean_hidden(texts, device="cpu"):
    """Frozen sentence embeddings: mean last-hidden-state of BASE gemma-2b over each description.

    gemma-2b is already the program's pre-install reference (LACUNA R1), so this adds no new model
    dependency. Deterministic and computed once, pre-measurement."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = "google/gemma-2b"
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32,
                                             device_map=device,
                                             attn_implementation="eager").eval()
    out = {}
    for k, t in texts.items():
        enc = tok(t, return_tensors="pt").to(device)
        with torch.inference_mode():
            h = m(**enc, output_hidden_states=True).hidden_states[-1][0]
        v = h.mean(dim=0).double()
        out[k] = (v / v.norm()).tolist()
    del m
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="ledger,roster,recipes,similarity,predict")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.pg import plenum_map as PM

    rep = {"object": "plenum_freeze_v1", "stages": stages}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- ledger -----------------------------------------------------------------------
    if "ledger" in stages:
        rep["ledger"] = {}
        for mod, sym in LEDGER:
            m = importlib.import_module(mod)
            assert hasattr(m, sym), f"§13 ledger BROKEN: {mod}.{sym} missing"
            rep["ledger"][f"{mod}.{sym}"] = "ok"
        print(f"[FRZ] ledger ok ({len(LEDGER)} symbols)", flush=True)
        _w()

    # ---- roster -----------------------------------------------------------------------
    if "roster" in stages:
        from huggingface_hub import HfApi
        api = HfApi()
        want = sorted(PM.FT_REPO.format(slug=s) for s in PM.FT_SLUGS)
        live = sorted(m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                      if m.id.endswith("-224") and "-jax" not in m.id
                      and m.id.startswith("google/paligemma-3b-ft-"))
        assert live == want, f"roster drift: only-live={set(live)-set(want)} " \
                             f"only-frozen={set(want)-set(live)}"
        rep["roster"] = {"repos": want, "shas": {r: api.model_info(r).sha for r in want},
                         "base_repo": "google/paligemma-3b-pt-224",
                         "base_sha": api.model_info("google/paligemma-3b-pt-224").sha}
        print(f"[FRZ] roster ok ({len(want)} FTs pinned)", flush=True)
        _w()

    # ---- recipes: CLAIM-LEVEL cite verification ---------------------------------------
    if "recipes" in stages:
        from huggingface_hub import hf_hub_download
        rec = {}
        for slug in PM.FT_SLUGS:
            repo = PM.FT_REPO.format(slug=slug)
            card = pathlib.Path(hf_hub_download(repo, "README.md", repo_type="model")).read_text()
            low = card.lower()
            toks = CARD_TOKENS[slug]
            hits = [t for t in toks if t in low]
            evidence = None
            for line in card.splitlines():
                if any(t in line.lower() for t in toks):
                    evidence = line.strip()[:200]
                    break
            rec[f"ft-{slug}"] = {
                "repo": repo, "faculties": list(PM.FT_FACULTIES[slug]),
                "card_tokens": toks, "matched": hits,
                "verified": bool(hits), "evidence": evidence,
                "note": ("card substantiates the dataset claim" if hits else
                         "CARD DOES NOT SUBSTANTIATE THE CLAIM — unit drops from the law fit "
                         "(§8) but is still measured"),
            }
            print(f"[FRZ] recipe {slug:>14s} verified={bool(hits)} ev={str(evidence)[:70]}",
                  flush=True)
        rep["recipes"] = rec
        rep["n_recipe_unverified"] = sum(1 for v in rec.values() if not v["verified"])
        _w()

    # ---- similarity: frozen, pre-measurement -------------------------------------------
    if "similarity" in stages:
        import numpy as np
        emb = _mean_hidden({**{f"faculty:{k}": v for k, v in TASK_DESCRIPTIONS.items()},
                            **{f"ft:{k}": v for k, v in FT_TASK_DESCRIPTIONS.items()}})
        def cos(a, b):
            return float(np.dot(emb[a], emb[b]))
        sim = {}
        for slug in PM.FT_SLUGS:
            sim[f"ft-{slug}"] = {f: cos(f"ft:{slug}", f"faculty:{f}") for f in PM.LAW_FACULTIES}
        # the policies' "recipe task" is their own faculty set — use the max over their faculties
        for uid, facs in (("pi0", ("control",)), ("pi05", ("control", "vqa", "caption")),
                          ("u3", ("control",)), ("u3c", ("control", "vqa"))):
            sim[uid] = {f: max(cos(f"faculty:{g}", f"faculty:{f}") for g in facs)
                        for f in PM.LAW_FACULTIES}
        rep["similarity"] = sim
        rep["similarity_source"] = ("mean last-hidden-state of google/gemma-2b over the frozen "
                                    "TASK_DESCRIPTIONS in this file; computed pre-measurement")
        print("[FRZ] similarity sealed", flush=True)
        _w()

    # ---- predictions: all four hypotheses, per cell, sealed -----------------------------
    if "predict" in stages:
        assert "similarity" in rep, "predictions need the sealed similarity"
        import numpy as np
        ub = PM.unit_by_id()
        fit_units = [u["id"] for u in PM.units() if u["role"] in ("fit", "heldout")]
        # ‖ΔW‖ is not known until the census runs, so H_magnitude's per-unit split is expressed as
        # a RULE over the census output rather than a baked value — sealing a number we do not have
        # would be theatre. The rule is frozen here; the aggregator applies it.
        preds = {}
        for uid in fit_units:
            u = ub[uid]
            law_recipe = [f for f in PM.LAW_FACULTIES if f in u["recipe_faculties"]]
            k = len(law_recipe)
            sims = rep["similarity"].get(uid, {})
            # top-k faculties by frozen similarity — the matched-sparsity competitor to membership
            top_k = set(sorted(PM.LAW_FACULTIES, key=lambda f: -sims.get(f, -1.0))[:k]) if k else set()
            for f in PM.LAW_FACULTIES:
                cell = f"{uid}|{f}"
                mem = f in u["recipe_faculties"]
                preds[cell] = {
                    "H_recipe": "retained" if mem else "consumed",
                    "H_fragility": ("retained" if f in FRAGILITY_ORDER[:2] else
                                    "partial" if f in FRAGILITY_ORDER[2:4] else "consumed"),
                    "H_similarity": ("retained" if f in top_k else "consumed"),
                    "H_magnitude": "RULE:retained iff unit ‖ΔW‖_LM below the roster median "
                                   "(faculty-independent, applied by the aggregator)",
                    "PROGRAM": "retained" if mem else "consumed",   # H_recipe is the primary
                    "_k": k, "_similarity": sims.get(f),
                }
        rep["predictions"] = preds
        rep["prediction_rules"] = {
            "H_recipe": "retained iff the faculty is in the unit's cite-verified recipe",
            "H_magnitude": "faculty-independent; retained iff ‖ΔW‖_LM below the roster median",
            "H_fragility": f"unit-independent ordering {list(FRAGILITY_ORDER)}; top-2 retained, "
                           f"middle-2 partial, bottom-2 consumed",
            "H_similarity": SIM_RULE,
            "note": "sealed BEFORE the first grid job (§7); scored in plenum_prediction_score.json",
        }
        (out / "plenum_predictions.json").write_text(json.dumps(
            {"predictions": preds, "rules": rep["prediction_rules"],
             "similarity": rep["similarity"]}, indent=2))
        print(f"[FRZ] sealed {len(preds)} per-cell predictions", flush=True)
        _w()

    print("PLENUM_FREEZE_OK", json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v)
                                          for k, v in rep.items() if k != "stages"},
                                         default=str)[:400], flush=True)


if __name__ == "__main__":
    main()
