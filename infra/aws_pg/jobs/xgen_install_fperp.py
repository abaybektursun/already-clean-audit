"""XGEN install-carrier f_perp on PG1 (RESEARCH_INDEX §5.2) — the W15 INSTALL-GRADIENT object,
per-lineage. Tests whether the W15 install-carrier out-of-span (f_perp = 0.81 on PG2 / Gemma-2)
replicates cross-generation on PG1 / Gemma-1.

FIREWALL (memory reference_fperp_two_objects): this is the INSTALL-GRADIENT f_perp (∂RefCOCO-CE/∂W
projected against W_before's span — the W15 object), NOT the raw-ΔW `decompose` f_perp (the W12
object, which XGEN V2 already replicated at-null). Never numerically transport across generations;
compare the VERDICT (out-of-span vs at-null) only.

Recipe (mirrors w15_retention): G_install = decoder_weight_grad(pg_before, refcoco_ce); per rectangular
decoder matrix, SVD(W_before) → U,V,R_eff; f_perp = grad_out_of_span_frac(G, U, V, R_eff); pool over
matrices vs a dim-counting null (random-gradient out-of-span fraction at matched shape/R_eff).

  python -m infra.aws_pg.jobs.xgen_install_fperp --out <dir> [--grad-n 256] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def build_before_after_pg1(device):
    """PG1 analog of diff_harvest.build_before_after: PaliGemma-1 (after) + a second PaliGemma-1 whose
    Gemma-1 decoder is reset to base gemma-2b (before), KEEPING vision/projector/embed. Fail-loud
    gates: decoder differs, vision/projector/embed PG1-identical, finite RefCOCO forward."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (AutoModelForCausalLM, PaliGemmaForConditionalGeneration,
                              PaliGemmaProcessor)
    from infra.pg.contrast_a1 import _lm_swap_state_dict
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    repo = ENTITIES["PG1"].repo_id              # google/paligemma-3b-pt-224 (Gemma-1, 18L)
    base = ENTITIES["L0_g1"]                     # google/gemma-2b (Gemma-1)
    proc = PaliGemmaProcessor.from_pretrained(repo)
    pg_after = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager").eval()
    paths = resolve_submodule_paths(pg_after)
    img_idx = getattr(pg_after.config, "image_token_index", None)
    assert img_idx is not None, "PG1 config has no image_token_index"

    pg_before = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager").eval()
    lm_before = pg_before.get_submodule(paths["language_model"])
    pg_lm_orig = {k: v.detach().clone().cpu() for k, v in lm_before.state_dict().items()}
    vt_orig = {k: v.detach().clone().cpu()
               for k, v in pg_before.get_submodule(paths["vision_tower"]).state_dict().items()}
    proj_orig = {k: v.detach().clone().cpu()
                 for k, v in pg_before.get_submodule(paths["multi_modal_projector"]).state_dict().items()}

    g1 = AutoModelForCausalLM.from_pretrained(
        base.repo_id, revision=base.revision, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager")
    new_lm_sd, linfo = _lm_swap_state_dict(lm_before.state_dict(), g1.model.state_dict())
    missing, unexpected = lm_before.load_state_dict(
        {k: v.to(device) for k, v in new_lm_sd.items()}, strict=True)
    assert not missing and not unexpected, f"PG1 BEFORE LM load not clean: {missing} {unexpected}"
    del g1
    if device == "cuda":
        torch.cuda.empty_cache()

    lm_now = lm_before.state_dict()
    dec_keys = [k for k in pg_lm_orig if "embed_tokens" not in k and "lm_head" not in k]
    dec_delta = max((lm_now[k].detach().cpu().float() - pg_lm_orig[k].float()).abs().max().item()
                    for k in dec_keys)
    assert dec_delta > 0.0, f"PG1 BEFORE decoder bitwise-equal to PG1 (swap did nothing) Δ={dec_delta}"
    for k in [k for k in pg_lm_orig if "embed_tokens" in k or "lm_head" in k]:
        assert torch.equal(lm_now[k].detach().cpu(), pg_lm_orig[k]), f"PG1 BEFORE embed/head changed: {k}"

    def _identical(now, orig):
        return set(now) == set(orig) and all(torch.equal(now[k].detach().cpu(), orig[k]) for k in orig)
    assert _identical(pg_before.get_submodule(paths["vision_tower"]).state_dict(), vt_orig), \
        "LEAK: PG1 BEFORE vision_tower changed"
    assert _identical(pg_before.get_submodule(paths["multi_modal_projector"]).state_dict(), proj_orig), \
        "LEAK: PG1 BEFORE projector changed"

    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text="detect cat\n", images=img, return_tensors="pt")
    moved = {k: (v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device))
             for k, v in inp.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        oa = pg_after(**moved); ob = pg_before(**moved)
    assert torch.isfinite(oa.logits).all() and torch.isfinite(ob.logits).all(), "non-finite PG1 forward"
    gate = {"decoder_max_abs_delta_vs_PG1": dec_delta, "lm_swap_info": linfo, "image_token_index": img_idx}
    return pg_after, pg_before, proc, paths, img_idx, gate


def _decoder_matrix_keys(lm):
    """2-D decoder weight matrices (q/k/v/o/gate/up/down per layer), excluding embed/lm_head/norms."""
    return sorted(k for k, p in lm.named_parameters()
                  if p.ndim == 2 and "layers." in k
                  and "embed_tokens" not in k and "lm_head" not in k)


def _dimcount_null_fperp(out_d, in_d, R_eff):
    """Dim-counting null = E[f_perp] of an ISOTROPIC random gradient: UᵀGV is rotation-invariant, so
    the top-R×R block holds R²/(out·in) of the energy in expectation → E[f_perp] = 1 − R²/(out·in).
    ANALYTIC (the empirical QR/draw version did a 16384² QR per MLP matrix → timeout)."""
    R = int(min(R_eff, out_d, in_d))
    return 1.0 - (R * R) / float(out_d * in_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["pg1", "pg2"], default="pg1",
                    help="pg1=Gemma-1 (paligemma-3b-pt-224); pg2=Gemma-2 (paligemma2-3b-pt-224, W15 re-measure with null)")
    ap.add_argument("--grad-n", type=int, default=256, help="RefCOCO install items for G_install")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import w15_carrier as C
    from infra.aws_pg.jobs import _w15_shared as S

    assert torch.cuda.is_available(), "needs GPU (gradients)"
    device = "cuda"
    grad_n = 16 if a.smoke else a.grad_n
    n_null = 50 if a.smoke else a.n_null
    print(f"[XGEN-FPERP] PG1 install-gradient f_perp | grad_n={grad_n} smoke={a.smoke}", flush=True)

    task = B.REFCOCO_TASK
    eval_ref = S._eval_rows(task, 32 if a.smoke else 300, S.SEED)
    eval_hashes = {S.img_hash(task.image_fn(r)) for r in eval_ref}
    install_items, _ = S.refcoco_install_items(B, max(grad_n * 4, 64), S.SEED, eval_hashes)

    if a.variant == "pg1":
        pg_after, pg_before, proc, paths, img_idx, gate = build_before_after_pg1(device)
        dec_delta = gate["decoder_max_abs_delta_vs_PG1"]
    else:                                            # pg2: reuse the W15 build (gemma-2-2b → PG2), add the null
        from infra.pg.diff_harvest import build_before_after
        pg_after, pg_before, proc, paths, img_idx, gate = build_before_after(device)
        dec_delta = gate.get("decoder_max_abs_delta_vs_PG", float("nan"))
    print(f"[XGEN-FPERP] {a.variant} before/after built: decoder Δ={dec_delta:.4g}", flush=True)
    lm_before = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    dkeys = _decoder_matrix_keys(lm_before)
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_before.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    del pg_after, lm_after
    if device == "cuda":
        torch.cuda.empty_cache()

    rc_batches = S._refcoco_batches(B, proc, install_items[:grad_n], train_batch=2)
    Gi, li = S.decoder_weight_grad(pg_before, paths, dkeys, rc_batches, S.refcoco_ce, device)
    print(f"[XGEN-FPERP] install gradient cached: refcoco_ce={li:.4f} on {len(dkeys)} matrices", flush=True)
    del pg_before
    if device == "cuda":
        torch.cuda.empty_cache()

    per_matrix = []
    for k in dkeys:
        Wb = base_sd[k].detach().double().cpu()
        dW = after_sd[k].detach().double().cpu() - Wb
        U, s, V = C.svd_fp64(Wb)
        R_eff = C.effective_rank_energy(s)
        fp = C.grad_out_of_span_frac(Gi[k].double().cpu(), U, V, R_eff)
        out_d, in_d = Wb.shape
        rectangular = out_d != in_d
        null_mean = _dimcount_null_fperp(out_d, in_d, R_eff) if rectangular else None
        per_matrix.append({"key": k, "shape": [int(out_d), int(in_d)], "rectangular": bool(rectangular),
                           "R_eff": int(R_eff), "f_perp": fp, "null_mean": null_mean})

    rect = [m for m in per_matrix if m["rectangular"]]
    fperp_rect = float(np.mean([m["f_perp"] for m in rect])) if rect else float("nan")
    null_rect = float(np.mean([m["null_mean"] for m in rect])) if rect else float("nan")
    # verdict vs null (firewall: compare to PG1's OWN dim-counting null; W15 0.81 is a Gemma-2 LABEL only)
    enriched = fperp_rect > null_rect  # install gradient MORE out-of-span than a random gradient?
    verdict = ("OUT-OF-SPAN-ENRICHED (install carrier lives out-of-span beyond dim-counting — W15-like)"
               if enriched and (fperp_rect - null_rect) > 0.02 else
               "AT-NULL (install gradient out-of-span = dimension-counting; not an enriched carrier)")
    result = {"experiment": "xgen_install_fperp", "variant": a.variant,
              "lineage": "PG1/Gemma-1" if a.variant == "pg1" else "PG2/Gemma-2",
              "object": "install-gradient f_perp (W15)",
              "firewall": "NOT raw-ΔW decompose (W12); W15 PG2 0.81 is a label-only cross-gen reference",
              "grad_n": grad_n, "n_matrices": len(dkeys), "n_rectangular": len(rect),
              "refcoco_install_ce": li, "swap_gate": gate,
              "f_perp_install_rect_mean": fperp_rect, "dimcount_null_rect_mean": null_rect,
              "enrichment_over_null": fperp_rect - null_rect, "verdict": verdict,
              "w15_pg2_reference_label_only": 0.81, "per_matrix": per_matrix}
    (out / "xgen_install_fperp.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[XGEN-FPERP] f_perp(install) rect-mean={fperp_rect:.4f}  dimcount-null={null_rect:.4f}  "
          f"enrichment={fperp_rect - null_rect:+.4f} → {verdict}", flush=True)
    print(f"[XGEN-FPERP] (W15 PG2 reference label-only: 0.81; firewall: verdict-compare, no transport)")


if __name__ == "__main__":
    main()
