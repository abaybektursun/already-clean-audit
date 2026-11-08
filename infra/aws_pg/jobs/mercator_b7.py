"""M-B7 — the language-variance anomaly (queue item B7). Wave 1.

HUMBOLDT's F9a measured sink ENERGY shifting ~10% across languages against ~4.5% for random
subspaces, and nobody explained it. A pure addressing story does not predict it: if the sink is a
fixed set of coordinates the model reserves, the fraction of activation variance landing there
should not care what language the tokens are in.

The arm measures the per-language profile properly and then asks whether the spread survives its
covariates:

  * ENERGY per language per layer, in core32 AND in the full twin battery, so "languages differ" is
    read against how much the TWINS differ across the same languages rather than against zero;
  * a VARIANCE DECOMPOSITION against two covariates that could produce the same pattern without any
    language-specific structure — script (whether the language shares the Latin script) and
    tokenization density (tokens per character, which is how much of the model's budget a language
    costs). If the between-language variance in core32 is no larger than the twins' AND is
    explained by these, the anomaly is an artifact of what the tokenizer does to the text.

REFUTER (pre-logged): per-language variance within the twin band, or explained by script or
tokenization covariates.

The corpora are HUMBOLDT's own resolved multilingual set, so the languages are the ones F9a
measured rather than a new selection.

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

ARM = "M-B7"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
# HUMBOLDT B1's resolved parallel corpus and its languages (facebook/xnli premise field).
XNLI_REPO = "facebook/xnli"
LANGS = ("en", "zh", "ar", "sw", "ru")
LATIN_SCRIPT = {"en", "sw"}


def _sentences(lang: str, n: int):
    from datasets import load_dataset
    ds = load_dataset(XNLI_REPO, lang, split="validation", streaming=True)
    out = []
    for row in ds:
        t = (row.get("premise") or "").strip()
        if len(t) >= 24:
            out.append(t)
        if len(out) >= n:
            break
    assert len(out) == n, f"{lang}: only {len(out)} usable sentences, wanted {n}"
    return out


def _energy_by_bank(gram, banks: dict) -> dict:
    """trace(U^T G U) / trace(G) per bank — the fraction of activation variance inside U."""
    G = np.asarray(gram, dtype=np.float64)
    tot = float(np.trace(G))
    assert tot > 0, "degenerate activation Gram"
    return {b: float(np.trace(U.T @ G @ U) / tot) for b, U in banks.items()}


def variance_decomposition(per_lang: dict, covariates: dict) -> dict:
    """Between-language variance of a per-language statistic, and what the covariates explain.

    Two covariates, each a real alternative explanation rather than a nuisance to regress away:
    `latin` (script family) and `tokens_per_char` (how densely the tokenizer encodes the language).
    Reported as the R^2 of a two-covariate least-squares fit, so "explained" is a number rather
    than an adjective — with only five languages this is deliberately a DESCRIPTIVE fit and its
    n is reported beside it.
    """
    langs = sorted(per_lang)
    y = np.array([per_lang[l] for l in langs], dtype=np.float64)
    X = np.stack([np.array([float(covariates[l]["latin"]) for l in langs]),
                  np.array([float(covariates[l]["tokens_per_char"]) for l in langs]),
                  np.ones(len(langs))], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"langs": langs, "values": y.tolist(), "mean": float(y.mean()),
            "sd": float(y.std(ddof=1)), "cv": float(y.std(ddof=1) / y.mean()) if y.mean() else None,
            "r2_covariates": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_langs": len(langs), "coef_latin": float(coef[0]),
            "coef_tokens_per_char": float(coef[1]),
            "note": "descriptive fit at n=5 languages; reported with its n, not as inference"}


def main():
    ap = argparse.ArgumentParser(description="M-B7 per-language sink energy + its covariates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sent", type=int, default=48)
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_b7", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_InputCapture, _blank_image,
                                                       _decoder_layers, _load_full_pg1)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    dec = _decoder_layers(model)
    blank = _blank_image(128)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    battery = {L: {b: WT.load_bank(bank_dir, "q_proj", L, b) for b in FULL_BATTERY}
               for L in layers}

    per_lang_energy, covariates = {}, {}
    for lang in LANGS:
        sents = _sentences(lang, a.n_sent)
        n_chars = sum(len(s) for s in sents)
        n_tok = sum(len(processor.tokenizer(s, add_special_tokens=False).input_ids) for s in sents)
        covariates[lang] = {"latin": lang in LATIN_SCRIPT,
                            "tokens_per_char": n_tok / max(n_chars, 1),
                            "n_sentences": len(sents), "n_chars": n_chars, "n_tokens": n_tok}
        with _InputCapture(dec) as cap:
            with torch.inference_mode():
                for s in sents:
                    inp = processor(text=[s], images=[blank], return_tensors="pt",
                                    padding=False).to("cuda")
                    model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                          pixel_values=inp["pixel_values"].to(torch.bfloat16))
        by_layer = {}
        for L in layers:
            g = cap.gram[L]
            assert g is not None, f"{lang}: layer {L} captured nothing"
            e = _energy_by_bank(g, battery[L])
            by_layer[L] = e
            for b, v in e.items():
                at.add("lang_activation_energy", value=v, layer=L, type="q_proj", bank_id=b,
                       corpus_id=f"xnli:{lang}", model_id=WT.PG1_REPO, model_sha=base_sha,
                       n=int(cap.n_tok[L]),
                       flags={"lang": lang, **{k: v2 for k, v2 in covariates[lang].items()}})
        per_lang_energy[lang] = by_layer
        print(f"[M-B7] {lang}: core32 mean over layers = "
              f"{np.mean([by_layer[L]['core32'] for L in layers]):.6f}", flush=True)

    at.check_controls_per_statistic(["lang_activation_energy"])

    # Between-language spread per bank, so core32's spread is read against the twins' spread.
    spread = {}
    for b in FULL_BATTERY:
        pl = {l: float(np.mean([per_lang_energy[l][L][b] for L in layers])) for l in LANGS}
        spread[b] = variance_decomposition(pl, covariates)
    twin_cvs = [spread[b]["cv"] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    core_cv = spread["core32"]["cv"]
    band = [float(min(twin_cvs)), float(max(twin_cvs))]
    at.add("lang_energy_cv", value=core_cv, bank_id="core32", type="q_proj",
           flags={"twin_band": band, "twin_cvs": twin_cvs,
                  "r2_covariates": spread["core32"]["r2_covariates"],
                  "n_langs": len(LANGS)})
    s = M.surprise(core_cv, float(np.mean(twin_cvs)), twin_cvs)
    sur.add(arm=ARM, expected_ref=expected_ref, cell="between-language CV",
            statistic="lang_energy_cv/core32", s=s)
    gates.record("b7_core_cv_outside_twin_band",
                 not (band[0] <= core_cv <= band[1]), core_cv=core_cv, twin_band=band)
    gates.record("b7_covariates_explain_the_spread",
                 spread["core32"]["r2_covariates"] > 0.9,
                 r2=spread["core32"]["r2_covariates"], n_langs=len(LANGS),
                 note="a descriptive fit at n=5; reported with its n")

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "langs": list(LANGS), "layers": layers, "n_sentences": a.n_sent,
                  "covariates": covariates, "spread_by_bank": spread,
                  "core_cv": core_cv, "twin_cv_band": band, "surprise": s})
    print("B7_OK " + json.dumps({"rows": info["n_rows"], "core_cv": core_cv,
                                 "twin_band": band}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
