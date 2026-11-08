"""flens faithfulness DIAGNOSTIC (not part of the frozen instrument; no rule change made here).

The discovery preflight read EV = −1.17 on the intact L16 LAST-TOKEN residual while the behavioral
round-trip MCQ was PRESERVED (0.497 ≈ intact 0.476). This script decides WHY: an ill-conditioned EV
metric on a homogeneous last-token distribution (tiny mean-centered variance baseline + Gemma-2
massive-activation dims) vs a real SAE-application bug. It compares, with the EXACT flens SAE code:

  (a) LAST-TOKEN L16 residual (the study's patch site — reproduces the −1.17), vs
  (b) ALL real token positions (the broad distribution C5 validated EV on),

reporting EV, per-item cosine(x, recon), relative L2 error, and the fraction of squared error carried
by the top-5 error dims + the massive-activation dims. If (b) is reasonable and (a) is ill-conditioned
with error concentrated on massive dims and high cosine, the SAE application is correct and the
last-token EV hard-gate is mis-specified (→ behavioral round-trip is the right gate). Reuses pin_sae.

    python -m infra.aws_pg.jobs.flens_faith_diag --out <dir> --n-per-subject 3
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--n-per-subject", type=int, default=3)
    args = ap.parse_args()
    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg.gemma_scope import resolve_gemma2_layers, explained_variance
    from infra.aws_pg.jobs.latloc_cell import build_confirmation_items, _capture, assert_generation_layers
    from infra.aws_pg.jobs.flens_cell import pin_sae
    from infra.pg import w16_probes as Wp

    assert torch.cuda.is_available(), "diag needs CUDA"
    device = "cuda"
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, attn_impl="eager")
    assert_generation_layers(pg_before.model.language_model, 26)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)
    sae, choice = pin_sae(device)
    L = 16
    print(f"[diag] SAE {choice.rel_path}; d_sae={sae.d_sae} d_model={sae.d_model}")

    items, gray_img, _ = build_confirmation_items(proc, Wp.is_grid_subset, n_per_subject=args.n_per_subject)
    tok = proc.tokenizer
    tok.padding_side = "left"
    last_rows, all_rows = [], []
    for s in range(0, len(items), 4):
        batch = items[s:s + 4]
        inp = proc(text=[it["prompt"] for it in batch], images=[gray_img] * len(batch),
                   return_tensors="pt", padding="longest").to(device)
        ids, am = inp["input_ids"], inp["attention_mask"]
        pv = inp["pixel_values"].to(torch.bfloat16)
        res = _capture(pg_before, b_layers, [L], ids, am, pv)[L]      # (b, T, d)
        last_rows.append(res[:, -1, :].float().cpu().numpy())
        for r in range(res.shape[0]):
            keep = am[r].bool()
            all_rows.append(res[r][keep].float().cpu().numpy())
    last = np.concatenate(last_rows, 0)
    allp = np.concatenate(all_rows, 0)

    # (c) TEXT-ONLY gemma-2-2b L16 (the SAE's TRAINING distribution; no image, run the LM directly) —
    # isolates "PG2 multimodal forward is OOD for the text-trained SAE" vs "L16 SAE / code is the issue".
    lm = pg_before.model.language_model
    text_rows = []
    for s in range(0, len(items), 4):
        batch = items[s:s + 4]
        enc = tok([it["prompt"] for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        res = _capture(lm, b_layers, [L], enc["input_ids"], enc["attention_mask"])[L]   # pv=None → text-only
        for r in range(res.shape[0]):
            keep = enc["attention_mask"][r].bool()
            text_rows.append(res[r][keep].float().cpu().numpy())
    text = np.concatenate(text_rows, 0)

    def ev_cos(X):
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.inference_mode():
            recon = sae.decode(sae.encode(xt))
        ev = explained_variance(xt, recon)
        cos = torch.nn.functional.cosine_similarity(xt, recon, dim=-1).mean().item()
        relerr = ((xt - recon).norm(dim=-1) / xt.norm(dim=-1).clamp_min(1e-6)).mean().item()
        sqerr = ((xt - recon) ** 2).mean(0).cpu().numpy()
        top = np.argsort(sqerr)[::-1][:5]
        absx = xt.abs().mean(0).cpu().numpy()
        mdims = np.argsort(absx)[::-1][:5]
        var_baseline = ((xt - xt.mean(0, keepdim=True)) ** 2).sum(-1).mean().item()
        return {"ev": float(ev), "cosine": float(cos), "rel_l2_err": float(relerr), "n": int(X.shape[0]),
                "frac_sqerr_top5dims": float(sqerr[top].sum() / sqerr.sum()),
                "top_err_dims": [int(x) for x in top], "massive_dims": [int(x) for x in mdims],
                "max_mean_abs_act": float(absx.max()), "var_baseline_per_item": var_baseline,
                "mean_l0": float((sae.encode(xt) != 0).float().sum(-1).mean().item())}

    res_last = ev_cos(last)
    res_all = ev_cos(allp)
    res_text = ev_cos(text)
    for name, r in (("LAST-TOK(PG2-mm)", res_last), ("ALL-POS(PG2-mm)", res_all), ("ALL-POS(TEXT-only)", res_text)):
        print(f"[diag] {name:18s} (n={r['n']:6d}): EV={r['ev']:+.3f} cos={r['cosine']:.3f} "
              f"relerr={r['rel_l2_err']:.3f} frac_top5err={r['frac_sqerr_top5dims']:.3f} "
              f"var_base={r['var_baseline_per_item']:.0f} L0={r['mean_l0']:.0f} maxact={r['max_mean_abs_act']:.0f}")
    # decisive read: if TEXT-only EV is good (~0.7) but PG2-multimodal is negative, the SAE+code are
    # correct and the multimodal forward is OOD (a study caveat); the behavioral round-trip is the gate.
    if res_text["ev"] > 0.5:
        verdict = f"MULTIMODAL-OOD (text-only EV={res_text['ev']:.2f} OK → SAE+code correct; behavioral gate is right)"
    elif res_all["cosine"] > 0.8 and res_last["cosine"] > 0.8:
        verdict = "LOSSY-BUT-DIRECTIONAL (cosine high, mean-dominated residual → EV ill-conditioned; behavioral gate)"
    else:
        verdict = "INVESTIGATE — text-only EV also poor; SAE may be a weak lens at L16"
    print(f"[diag] VERDICT: {verdict}")
    (out / "faith_diag.json").write_text(json.dumps(
        {"last_token_pg2": res_last, "all_pos_pg2": res_all, "all_pos_text_only": res_text,
         "sae": choice.rel_path, "verdict": verdict}, indent=1))
    print("[diag] banked → faith_diag.json")


if __name__ == "__main__":
    main()
