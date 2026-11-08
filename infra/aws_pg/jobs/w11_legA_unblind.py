"""W11 §9 Leg-A UNBLIND (step 5: compute the treatment ONCE, apply frozen rules).

Evaluate AFTER = the released PG2-LM (PaliGemma2-3b-pt-224 language model run TEXT-ONLY,
sdpa to match the BEFORE/FLOOR controls), then:

  S_text (capability) = (acc_AFTER − acc_FLOOR) / (acc_BEFORE − acc_FLOOR)

with a PAIRED nonparametric bootstrap over MMLU items (AFTER per-item from this run;
BEFORE/FLOOR per-item from the FROZEN controls.json — same items, same order). Plus the
secondary perplexity ratio AFTER/BEFORE (window bootstrap). Frozen §6A rule:

  S_text ≥ 1 − SESOI (=0.90)  → H_preserve   (kept native language)
  S_text < 1 − SESOI          → H_forget      (degraded; report magnitude + which metric)

Both metrics reported; disagreement (fluent-but-forgot, or vice-versa) IS the finding.
Single read; no try/except.

    python -m infra.aws_pg.jobs.w11_legA_unblind --out <dir> \
        --controls-s3 s3://pg-lineage-769063704696/results/w11_legA_controls/out/legA_controls.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def _boot_S(after_pi, before_pi, floor_pi, n_boot=2000, seed=1337):
    import numpy as np
    a = np.asarray(after_pi, float); b = np.asarray(before_pi, float); f = np.asarray(floor_pi, float)
    n = a.size
    assert b.size == n and f.size == n, f"per-item arrays misaligned: {a.size}/{b.size}/{f.size}"
    rng = np.random.default_rng(seed)
    Ss = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accA, accB, accF = a[idx].mean(), b[idx].mean(), f[idx].mean()
        den = accB - accF
        if den > 0:
            Ss.append((accA - accF) / den)
    Ss = np.asarray(Ss)
    return float(Ss.mean()), float(np.percentile(Ss, 2.5)), float(np.percentile(Ss, 97.5)), int(Ss.size)


def _boot_ppl_ratio(after_win, before_win, n_boot=2000, seed=1337):
    import numpy as np
    aw = np.asarray(after_win, float); bw = np.asarray(before_win, float)   # [:, (nll, ntok)]
    n = aw.shape[0]
    assert bw.shape[0] == n, f"ppl window arrays misaligned: {aw.shape}/{bw.shape}"
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ppl_a = np.exp(aw[idx, 0].sum() / aw[idx, 1].sum())
        ppl_b = np.exp(bw[idx, 0].sum() / bw[idx, 1].sum())
        ratios.append(ppl_a / ppl_b)
    ratios = np.asarray(ratios)
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True, help="frozen legA_controls.json on S3")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="MUST match controls (0=full)")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration

    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "Leg-A unblind needs GPU."
    device = "cuda"

    # --- read FROZEN controls (instance-role aws cli; provenance, never transcribed) ---
    ctrl_local = out / "legA_controls.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local), "--region", "us-east-1"],
                   check=True)
    ctrl = json.loads(ctrl_local.read_text())
    assert ctrl["CONTROLS_OK"] is True, "Leg-A controls did NOT pass (CONTROLS_OK=False) — fix harness, do not unblind."
    assert int(ctrl["mmlu_n_per_subject"]) == args.mmlu_n_per_subject, (
        f"n_per_subject mismatch: controls={ctrl['mmlu_n_per_subject']} unblind={args.mmlu_n_per_subject}")
    sesoi = float(ctrl["sesoi"])
    item_hashes = ctrl["item_hashes"]
    before_pi = ctrl["mmlu"]["BEFORE"]["per_item"]
    floor_pi = ctrl["mmlu"]["FLOOR"]["per_item"]
    acc_before = float(ctrl["mmlu"]["BEFORE"]["acc"])
    acc_floor = float(ctrl["mmlu"]["FLOOR"]["acc"])
    # §9-4 control-driven correction: the perplexity ratio compares AFTER vs the STANDALONE BASE
    # (both self-consistent LMs on the same token stream). BEFORE (decoder-reset) is a base-decoder +
    # PG-drifted-embed/head CHIMERA — robust for the MMLU letter-logit contrast (its denominator) but
    # PATHOLOGICAL for generative ppl (controls showed BEFORE ppl ≫ base). So ppl uses base, MMLU uses
    # BEFORE. The primary S_text (MMLU) is unaffected.
    base_win = ctrl["perplexity"]["base"]["per_window"]
    ppl_base = float(ctrl["perplexity"]["base"]["ppl"])
    ppl_before_chimera = float(ctrl["perplexity"]["BEFORE"]["ppl"])   # recorded, NOT the denominator
    print(f"[W11-A-UNB] frozen BEFORE acc={acc_before:.4f} FLOOR acc={acc_floor:.4f} "
          f"base ppl={ppl_base:.2f} SESOI={sesoi}")

    # --- AFTER = released PG2-LM, text-only, sdpa (matches BEFORE/FLOOR) ---
    PG = ENTITIES["PG"]
    pg_after = PaliGemmaForConditionalGeneration.from_pretrained(
        PG.repo_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    gemma_tok = AutoTokenizer.from_pretrained(ENTITIES["L0"].repo_id, revision=ENTITIES["L0"].revision)
    n_per = args.mmlu_n_per_subject or None

    after_mmlu = TE.mmlu_accuracy(pg_after, gemma_tok, device, n_per_subject=n_per)
    after_ppl = TE.clean_text_perplexity(pg_after, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    acc_after = float(after_mmlu["acc"])
    after_pi = after_mmlu["per_item"]
    print(f"[W11-A-UNB] AFTER MMLU={after_mmlu['acc_pct']:.2f} ppl={after_ppl['ppl']:.2f}")
    assert len(after_pi) == len(before_pi) == len(floor_pi), (
        f"MMLU item-count mismatch vs frozen controls: after={len(after_pi)} "
        f"before={len(before_pi)} floor={len(floor_pi)} — n_per_subject must match controls.")
    assert after_mmlu["per_item_hash"] == item_hashes, (
        "MMLU item ORDER differs from frozen controls (question-hash mismatch) — the paired "
        "bootstrap would misalign. Aborting.")

    # --- S_text (capability, primary) + paired bootstrap ---
    denom = acc_before - acc_floor
    S_text = (acc_after - acc_floor) / denom
    S_mean, S_lo, S_hi, S_nb = _boot_S(after_pi, before_pi, floor_pi)

    # --- perplexity ratio (secondary fluency): AFTER vs standalone BASE ---
    ppl_ratio = after_ppl["ppl"] / ppl_base
    pr_mean, pr_lo, pr_hi = _boot_ppl_ratio(after_ppl["per_window"], base_win)

    # --- frozen §6A verdict (capability is the decision variable; point estimate governs) ---
    threshold = 1.0 - sesoi
    verdict = "H_preserve" if S_text >= threshold else "H_forget"
    # CI-awareness: flag when the point clears the bar but the lower CI bound does not (tight call).
    marginal = bool(S_text >= threshold and S_lo < threshold)
    # secondary: ppl ratio ≈ 1 within SESOI on the POINT estimate (not the lower bound — a high
    # ratio with a wide CI must NOT read as preserved; critic fix).
    ppl_preserve = bool(ppl_ratio <= 1.0 + sesoi)
    print(f"[W11-A-UNB] S_text={S_text:.4f} CI[{S_lo:.4f},{S_hi:.4f}] thr={threshold} → {verdict}"
          f"{' (MARGINAL: CI straddles)' if marginal else ''}")
    print(f"[W11-A-UNB] ppl_ratio={ppl_ratio:.4f} CI[{pr_lo:.4f},{pr_hi:.4f}] ppl_preserve={ppl_preserve}")

    result = {
        "torch": torch.__version__,
        "controls_s3": args.controls_s3,
        "sesoi": sesoi, "preserve_threshold": threshold,
        "mmlu": {"acc_after_pct": after_mmlu["acc_pct"], "acc_before": acc_before,
                 "acc_floor": acc_floor, "frozen_denominator": denom, "n": after_mmlu["n"]},
        "S_text_capability": {"point": S_text, "boot_mean": S_mean, "ci95": [S_lo, S_hi],
                              "n_boot_valid": S_nb, "marginal_ci_straddles": marginal},
        "perplexity": {"ppl_after": after_ppl["ppl"], "ppl_base": ppl_base,
                       "ppl_before_chimera": ppl_before_chimera,
                       "denominator": "standalone base gemma-2-2b (BEFORE is a decoder/embed chimera)",
                       "ratio_point": ppl_ratio, "ratio_boot_mean": pr_mean, "ratio_ci95": [pr_lo, pr_hi],
                       "ppl_preserve": ppl_preserve},
        "VERDICT": verdict,
        "prediction_grid": {
            "S_text_capability_vs_base": f"{S_text:.3f} (preserve if ≥{threshold})",
            "perplexity_ratio": f"{ppl_ratio:.3f} (preserve if ≈1)",
        },
        "after_per_item": after_pi,   # banked for figures / reanalysis
    }
    (out / "legA_unblind.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-A-UNB] VERDICT={verdict}  S_text={S_text:.4f}")


if __name__ == "__main__":
    main()
