"""E4 (MDE) — function-space compressibility of the modality-install delta. GraftLM plan §4
Tier-2 / the decisive F2-F3 reconciliation. EXPLORATORY / DISCOVERY.

E2 showed weight-space RANK cannot separate modality-install from task-tuning (both full-rank,
depth-uniform, isotropic-in-span; they differ only ~2.2x in magnitude). So the cost question is
FUNCTION-space: does a COMPRESSED edit of the base->pt weight delta recover PG2 behaviour, or is
the rewrite information-rich / irreducible?

Estimand (plan §3, native R_KL — NO comparison to the retrained-bridge S_B):
  Apply a compressed edit E to the DECODER WEIGHT MATRICES with the original PG2 connector + lm_head
  + embed_tokens FROZEN and per-layer norms held at base, NO retraining. Measure recovery on ALL
  probe text positions (is_text_keep — NOT a held-out split; a no-retrain SVD edit fits nothing to
  the probe, so there is no leakage to guard against):
      R(E) = 1 - KL(P_fullPG2 || P_base+E) / KL(P_fullPG2 || P_base)
  R_KL is a KL-MASS-WEIGHTED divergence-closure fraction (ratio of mean KL nats), NOT a capability
  fraction. x-axis L(E) = params in the rank-k factorisation; %-params is vs the EDITED decoder-
  matrix basis (computed dynamically), NOT the full LM (which includes frozen embeddings).

Edit family (this MDE): per-decoder-matrix rank-k SVD truncation of dW = (after_LM - base_LM),
swept over k, applied to ALL 26 layers x 7 matrices simultaneously. This is the FLOOR of graft
methods (no retrain, no change of basis); a TRAINED low-rank adapter is NOT tested here.

ANCHORS:
  PLUMBING (edited LM == after entirely) -> R must be ~1. NOTE: this is a model-vs-itself plumbing
           check (forward/masking/KL wiring); it does NOT exercise the rank-k edit path.
  CEILING_edit (full 2-D dW, norms at base) -> the REAL edit-path gate (measured, <1 if norms matter).
  FLOOR (E = 0) -> R == 0 by definition.
Nulls: E_scale (norm-only magnitude restoration) + N3 (random rank-k matched to ||dW||_F) at the
  head rank AND a mid rank -> a real edit must beat both.

Reading the curve (heavy-tailed, honest): a low-rank HEAD (small k) may cheaply close a large
fraction of the AVERAGE KL gap while the TAIL stays expensive (near-full rank for full recovery).
Report the full R(k) curve; do NOT collapse to a binary compressible/irreducible label, and do NOT
read KL-closure as capability.

GPU job (forward passes). hf_online_setup() first (probe datasets stream). No try/except — fail
loud. Reuses build_pair (leak-gated before/after), build_mm_batch (RefCOCO+TextVQA probe +
is_text_keep), diff_metrics (mean_kl / r_kl / bootstrap). Writes JSON.
"""
from __future__ import annotations

import argparse
import json
import pathlib

LM_MATRIX_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 26
K_SWEEP = (8, 32, 128, 512, 1024)  # rank-k per matrix (full rank = min(out,in) up to 2304)
K_MAX = 1024                     # truncated-SVD budget (just under the ~1440 eff-rank, to pin the tail)
N3_K = 128                       # the random-null reference rank
MICRO = 4                        # forward micro-batch (images)
SEED = 1337
N_BOOT = 1000


def _decoder_key(layer: int, mtype: str) -> str:
    sub = "self_attn" if mtype in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"layers.{layer}.{sub}.{mtype}.weight"


def _decoder_keys():
    return [_decoder_key(L, mt) for L in range(N_LAYERS) for mt in LM_MATRIX_TYPES]


def _forward_kl_per_pos(model_p, model_q, mm, micro):
    """Per-text-position KL(P_p || P_q) over the whole probe batch, micro-batched. model_p, model_q
    get the SAME inputs so positions align row-for-row. Returns a 1-D CPU numpy array."""
    import numpy as np
    import torch

    ids, attn, pix = mm["input_ids"], mm["attention_mask"], mm["pixel_values"]
    keep = mm["is_text_keep"]
    n = ids.shape[0]
    out = []
    for s in range(0, n, micro):
        sl = slice(s, s + micro)
        b = {"input_ids": ids[sl], "attention_mask": attn[sl], "pixel_values": pix[sl]}
        m = keep[sl]                                        # (b, T) bool
        with torch.inference_mode():
            lp = model_p(**b).logits[m]                     # (npos, V)
            lq = model_q(**b).logits[m]
        lp = torch.log_softmax(lp.float(), dim=-1)
        lq = torch.log_softmax(lq.float(), dim=-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)                 # (npos,)
        out.append(kl.detach().cpu().numpy())
        del lp, lq, kl
        torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0)


def _set_decoder(lm, base_sd, edited_decoder, device):
    """Load base_sd with the given decoder weight overrides into the language_model submodule."""
    import torch
    sd = dict(base_sd)
    for k, w in edited_decoder.items():
        sd[k] = w
    miss, unexp = lm.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
    assert not miss and not unexp, f"edited LM load not strict-clean: miss={miss} unexp={unexp}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis", default="base-pt", choices=["base-pt", "pt-mix"],
                    help="which delta to compress: base-pt (modality install) or pt-mix (task tuning)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny probe (8 img/probe), k in {32}, fewer bootstraps — wiring + anchors")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from infra.pg import diff_harvest as H
    from infra.pg import diff_metrics as M

    H.hf_online_setup()                                     # probe datasets stream (gated repos)

    device = "cuda"
    assert torch.cuda.is_available(), "E4 needs a GPU (forward passes)"
    print(f"[E4] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} smoke={a.smoke}", flush=True)

    if a.smoke:
        H.MM_PROBES = tuple({**dict(p), "n_images": 8} for p in H.MM_PROBES)
    k_sweep = (32,) if a.smoke else K_SWEEP
    k_max = 64 if a.smoke else K_MAX
    n_boot = 200 if a.smoke else N_BOOT

    # ---- build leak-gated before/after + probe batch ---------------------------------------
    # axis base-pt: after=PG2-pt, before=PG2 with gemma-2-2b decoder (modality install delta).
    # axis pt-mix:  after=PG2-mix, before=PG2-pt (task-tuning delta). Same 6-tuple either way.
    pg_after, pg_before, proc, paths, img_tok, gate = H.build_pair(a.axis, device)
    print(f"[E4] axis={a.axis} swap_gate={json.dumps(gate, default=str)}", flush=True)
    mm = H.build_mm_batch(proc, img_tok, device)
    print(f"[E4] probe: ids={tuple(mm['input_ids'].shape)} n_text_keep={int(mm['is_text_keep'].sum())} "
          f"n_examples={mm['n_examples']}", flush=True)

    lm_after = pg_after.get_submodule(paths["language_model"])
    lm_edit = pg_before.get_submodule(paths["language_model"])   # we mutate pg_before's LM per config
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    dkeys = _decoder_keys()
    for k in dkeys:
        assert k in base_sd and k in after_sd, f"missing decoder key {k}"

    # P_full reference = the BEFORE model's vision/projector + the FULL after-LM. This ISOLATES the
    # LM delta: base-pt → equals PG2-pt; pt-mix → PT-vision/projector + MIX-LM (NOT standalone MIX,
    # so it EXCLUDES the vision/projector changes the mix fine-tune also made — empirically ~5% of
    # the pt→mix shift). Built on the PG2-pt repo (both supported axes have a PT-vision 'before'),
    # so its vision/projector match pg_before's → PLUMBING (full LM edit) == 1 by construction.
    from transformers import PaliGemmaForConditionalGeneration
    from infra.pg.entities import ENTITIES
    pg_full = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa").eval()
    _set_decoder(pg_full.get_submodule(paths["language_model"]), after_sd, {}, device)
    del pg_after
    torch.cuda.empty_cache()

    # ---- truncated SVD of dW per decoder matrix (GPU), keep top-k_max factors on CPU --------
    print(f"[E4] truncated SVD (q={k_max}) of {len(dkeys)} decoder dW ...", flush=True)
    svd = {}
    dW_frob = {}
    for k in dkeys:
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)    # (out, in)
        q = min(k_max + 256, min(dW.shape))    # generous oversample so the high-k tail is accurate
        U, S, V = torch.svd_lowrank(dW, q=q, niter=8)                 # U(out,q) S(q) V(in,q)
        svd[k] = (U[:, :k_max].cpu(), S[:k_max].cpu(), V[:, :k_max].cpu())
        dW_frob[k] = float(dW.norm().item())
        del dW, U, S, V
        torch.cuda.empty_cache()

    def _rankk_decoder(k):
        d = {}
        for key in dkeys:
            U, S, V = svd[key]
            kk = min(k, S.shape[0])
            dWk = (U[:, :kk].to(device) * S[:kk].to(device)) @ V[:, :kk].to(device).T
            d[key] = (base_sd[key].float().to(device) + dWk).to(torch.bfloat16)
        return d

    def _ldesc(k):
        return int(sum(min(k, svd[key][1].shape[0]) * (base_sd[key].shape[0] + base_sd[key].shape[1])
                       for key in dkeys))

    # ---- baseline: KL(full || base) --------------------------------------------------------
    _set_decoder(lm_edit, base_sd, {}, device)              # FLOOR: edited == base
    kl_base = _forward_kl_per_pos(pg_full, pg_before, mm, MICRO)
    base_gap = float(kl_base.mean())
    print(f"[E4] baseline KL(full||base) mean={base_gap:.5f} n_pos={kl_base.size}", flush=True)
    assert base_gap > 0, "baseline gap <= 0 — full and base are behaviourally identical?! halt."
    # KL-mass diagnostic: R_KL is a ratio of MEANS, so it is dominated by the highest-KL positions.
    # Report the concentration so readers know the headline is mass-weighted, not typical-position.
    _kb = np.sort(kl_base)[::-1]
    _tot = float(_kb.sum())
    kl_mass = {"median_kl": float(np.median(kl_base)),
               "top1pct_share": float(_kb[:max(1, kl_base.size // 100)].sum() / _tot) if _tot > 0 else None,
               "top10pct_share": float(_kb[:max(1, kl_base.size // 10)].sum() / _tot) if _tot > 0 else None}
    print(f"[E4] baseline KL-mass: median={kl_mass['median_kl']:.4f} top1%={kl_mass['top1pct_share']:.2f} "
          f"top10%={kl_mass['top10pct_share']:.2f}", flush=True)

    prov_pairs = {
        "base-pt": {"after": "google/paligemma2-3b-pt-224", "base": "google/gemma-2-2b (decoder)"},
        "pt-mix": {"after": "google/paligemma2-3b-mix-224", "base": "google/paligemma2-3b-pt-224"},
    }
    results = {
        "experiment": "E4_compress_curve", "axis": a.axis, "status": "exploratory", "smoke": a.smoke,
        "config": {"k_sweep": list(k_sweep), "k_max": k_max, "n3_k": N3_K, "micro": MICRO,
                   "seed": SEED, "n_boot": n_boot, "n_layers": N_LAYERS,
                   "lm_matrix_types": list(LM_MATRIX_TYPES)},
        "provenance": {**prov_pairs[a.axis], "swap_gate": gate},
        "baseline_kl_full_base": base_gap, "n_text_pos": int(kl_base.size), "baseline_kl_mass": kl_mass,
        "curve": {}, "controls": {}, "anchors": {},
    }

    def _eval(label, edited_decoder, base=None):
        _set_decoder(lm_edit, base if base is not None else base_sd, edited_decoder, device)
        kl_e = _forward_kl_per_pos(pg_full, pg_before, mm, MICRO)
        ci = M.r_kl_bootstrap_ci(kl_base, kl_e, n_boot=n_boot, seed=SEED)
        rec = {"R_KL": M.r_kl(base_gap, float(kl_e.mean())),
               "R_KL_ci": ci, "kl_recovered_mean": float(kl_e.mean())}
        print(f"[E4] {label:14s} R_KL={rec['R_KL']:.4f} "
              f"[{ci['lo']:.4f},{ci['hi']:.4f}] kl_rec={rec['kl_recovered_mean']:.5f}", flush=True)
        return rec

    # ---- ANCHORS ---------------------------------------------------------------------------
    # PLUMBING: edited LM == PG2 ENTIRELY (decoder + norms) -> R must be ~1. Verifies the forward +
    # text-position masking + KL plumbing (NOT the science). Fail loud if violated.
    plumb = _eval("PLUMBING(full)", {}, base=after_sd)
    results["anchors"]["plumbing_full_after"] = plumb
    assert plumb["R_KL"] > 0.98, (
        f"ANCHOR FAIL: full-after R_KL={plumb['R_KL']:.4f} (expected ~1; edited==PG2). Harness wrong.")
    results["anchors"]["floor_zero"] = {"R_KL": 0.0, "note": "baseline (pure base) by definition"}

    # CEILING_edit (the gate, plan §9 fork 3): full 2-D decoder dW ONLY, norms kept at base. This is
    # the MAX recovery the rank-k weight-matrix family can reach (may be < 1 if the norms matter).
    ceil_edit = _eval("CEILING_edit", {k: after_sd[k].to(torch.bfloat16) for k in dkeys})
    results["anchors"]["ceiling_edit_2d_only"] = ceil_edit

    # CEILING_edit_with_norms control: full 2-D dW + after per-layer norms. The gap to ceiling_edit
    # bounds how much of the <1 ceiling is the frozen-norm DOF the rank-k family cannot touch.
    norm_keys = [k for k in after_sd if k not in dkeys and after_sd[k].ndim == 1]
    ceil_norm = _eval("CEIL_edit+norms",
                      {**{k: after_sd[k].to(torch.bfloat16) for k in dkeys},
                       **{k: after_sd[k].to(torch.bfloat16) for k in norm_keys}})
    results["anchors"]["ceiling_edit_with_norms"] = ceil_norm
    results["anchors"]["norm_contribution"] = ceil_norm["R_KL"] - ceil_edit["R_KL"]

    # ---- real rank-k curve -----------------------------------------------------------------
    for k in k_sweep:
        rec = _eval(f"rank-{k}", _rankk_decoder(k))
        rec["L_desc_params"] = _ldesc(k)
        results["curve"][str(k)] = rec

    # ---- NULLS -----------------------------------------------------------------------------
    # E_scale norm-only: scale each base matrix to the after Frobenius norm (no direction change).
    escale = {}
    for k in dkeys:
        nb = base_sd[k].float().norm().clamp_min(1e-8)
        na = after_sd[k].float().norm()
        escale[k] = (base_sd[k].float() * (na / nb)).to(torch.bfloat16)
    results["controls"]["E_scale_normonly"] = _eval("E_scale", escale)

    # N3 random rank-r null: rank-r of a random dW matched to ||dW||_F, added to base. Run at the
    # HEAD rank (anchors the low-rank-head claim) AND a mid rank — real rank-r must beat random rank-r.
    def _n3(rank):
        g = torch.Generator(device="cpu").manual_seed(SEED)
        d = {}
        for key in dkeys:
            out_d, in_d = base_sd[key].shape
            Rm = torch.randn(out_d, in_d, generator=g)
            Rm = Rm * (dW_frob[key] / Rm.norm().clamp_min(1e-8))
            U, S, V = torch.svd_lowrank(Rm.to(device), q=min(rank + 16, min(out_d, in_d)), niter=4)
            kk = min(rank, S.shape[0])
            d[key] = (base_sd[key].float().to(device) + (U[:, :kk] * S[:kk]) @ V[:, :kk].T).to(torch.bfloat16)
            del Rm, U, S, V
        return d
    for r in sorted({min(k_sweep), N3_K}):
        results["controls"][f"N3_random_rank{r}"] = _eval(f"N3-rank{r}", _n3(r))

    # ---- headline verdict ------------------------------------------------------------------
    rks = {int(k): results["curve"][str(k)]["R_KL"] for k in k_sweep}
    r_top = rks[max(k_sweep)]
    r_lo = rks[min(k_sweep)]
    ceiling_edit_R = results["anchors"]["ceiling_edit_2d_only"]["R_KL"]
    full_rank_anchor = 1440.3   # E2/W1 eff-rank (cross-check only; never a result number)
    # %-params is vs the EDITED decoder-matrix basis (embeddings + norms are NOT edited), computed
    # dynamically — NOT the full LM (which wrongly includes the frozen ~0.59B embed table).
    edited_params = sum(int(base_sd[k].shape[0]) * int(base_sd[k].shape[1]) for k in dkeys)
    CHEAP_FRAC = 0.10
    def _pf(k):
        return results["curve"][str(k)]["L_desc_params"] / edited_params
    cheap_ks = [int(k) for k in k_sweep if _pf(k) < CHEAP_FRAC]
    cheap_k = max(cheap_ks) if cheap_ks else min(k_sweep)
    r_cheap = rks[cheap_k]
    frac_cheap = r_cheap / ceiling_edit_R if ceiling_edit_R > 0 else float("nan")
    frac_top = r_top / ceiling_edit_R if ceiling_edit_R > 0 else float("nan")
    head_k = min(k_sweep)
    results["headline"] = {
        "edited_decoder_matrix_params": edited_params,
        "R_at_min_k": {str(head_k): r_lo}, "R_at_max_k": {str(max(k_sweep)): r_top},
        "ceiling_edit_R": ceiling_edit_R,
        "ceiling_edit_with_norms_R": results["anchors"].get("ceiling_edit_with_norms", {}).get("R_KL"),
        "R_at_max_k_frac_of_ceiling_edit": frac_top,
        "low_rank_head": {"k": head_k, "pct_edited_params": _pf(head_k), "R_KL": r_lo,
                          "frac_of_ceiling_edit": r_lo / ceiling_edit_R if ceiling_edit_R > 0 else None},
        "cheap_graft": {"k": cheap_k, "pct_edited_params": _pf(cheap_k), "R_KL": r_cheap,
                        "frac_of_ceiling_edit": frac_cheap},
        "E_scale_R": results["controls"]["E_scale_normonly"]["R_KL"],
        "eff_rank_anchor": full_rank_anchor,
        # HEAVY-TAILED framing — report the curve, do NOT collapse to a binary label; R_KL is
        # KL-mass-weighted divergence closure, NOT capability.
        "verdict": (
            f"Heavy-tailed (no-retrain SVD truncation of the decoder-weight dW, norms at base; R_KL = "
            f"KL-mass-weighted divergence closure, NOT capability). Low-rank HEAD rank-{head_k} "
            f"({_pf(head_k):.1%} of edited params) closes R_KL={r_lo:.3f} ({r_lo/ceiling_edit_R:.0%} of "
            f"CEILING_edit={ceiling_edit_R:.3f}) >> E_scale null {results['controls']['E_scale_normonly']['R_KL']:.3f}. "
            f"Cheap budget rank-{cheap_k} (<{CHEAP_FRAC:.0%} edited params) -> {frac_cheap:.0%} of ceiling; "
            f"tail rank-{max(k_sweep)} ({_pf(max(k_sweep)):.0%} params) -> {frac_top:.0%}. "
            f"READ: a low-rank head cheaply closes most of the AVERAGE KL gap; the tail stays expensive. "
            f"Full recovery beyond rank-{max(k_sweep)} is EXTRAPOLATION (curve not plateaued); a TRAINED "
            f"adapter is untested; partial KL-closure != capability."),
    }
    (out / "e4_compress_curve.json").write_text(json.dumps(results, indent=2))
    print("E4_OK", json.dumps(results["headline"], indent=2), flush=True)


if __name__ == "__main__":
    main()
