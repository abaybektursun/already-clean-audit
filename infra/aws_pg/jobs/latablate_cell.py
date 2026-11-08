"""latablate cell runner — is the Gemma-2 mid-stack MMLU-MCQ deletion a LOW-RANK removable
direction? (prereg `studies/lineage/paligemma2/research/2026-06-29_latloc_ablate_prereg.md`, the MDE).

LATLOC localized the blank-image MMLU-MCQ deletion to L16/26 and showed the corruption there is
generic + saturating, while a localized WEIGHT-restore is null. This job estimates the disabling
subspace at L16 from the (deleted − intact) last-token residual differences, then ABLATES (projects
out) the top-k directions from the DELETED forward at L16 via a fixed rank-k projector (NEW
AblateHook) — input-independent, no intact model, no retraining — and sweeps RF(k).

This is the NEW glue the prereg §13 calls for. The projection ALGEBRA (h − (hV)Vᵀ) is shared with
invariant_core_func._InputProject, but that hook acts on the q/k-proj INPUT; AblateHook acts on the
decoder layer's OUTPUT residual (output[0]) — the latloc deletion locus where the intact-patch
restored. Pure analysis math lives in `infra.pg.latablate` (weight-free, --dry-run-tested).

Two entrypoints:
  --dry-run   weight-free: pure-logic dry-run (infra.pg.latablate.dry_run) AND the torch AblateHook
              dry-run (real hook on a tiny stack) on CPU — prereg §13. Dispatch on pg `--no-weights`.
  (default)   the GPU MDE: eager loader + 26-layer firewall + structural preflight + L16 residual
              harvest (both models) + SVD subspace + the k-sweep ablation (real + random + shuffle).

    python -m infra.aws_pg.jobs.latablate_cell --dry-run
    python -m infra.aws_pg.jobs.latablate_cell --out <dir> --eval-split discovery
    python -m infra.aws_pg.jobs.latablate_cell --out <dir> --eval-split confirmation --headline-k 8
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.latloc_cell import _assert_raises, _capture, assert_generation_layers


# ===========================================================================
# AblateHook (NEW; prereg §13) — a fixed rank-k projector on ONE decoder layer's OUTPUT residual.
# Mirrors lm_patching.PatchHook's output[0] read/replace + shape-check + tuple rebuild, but
# SUBTRACTS the projection onto a frozen orthonormal V instead of overwriting masked positions,
# and acts on the residual-STREAM output (the latloc deletion locus), NOT the q/k-proj input that
# invariant_core_func._InputProject hooks. Input-independent → the deployable surgical fix.
# ===========================================================================

class AblateHook:
    """Forward hook on `layer_module` that ablates span(V) from the layer's OUTPUT residual at ALL
    positions: h ← h − (h V) Vᵀ. V is a torch (d, k) tensor with ORTHONORMAL columns (the disabling
    subspace), on the model device/dtype. k=0 (V is (d, 0)) is a no-op (the floor). A residual-dim
    mismatch RAISES (no silent broadcast, no try/except). Never zeros the residual (it removes only
    the k-dim span). Context-managed so the handle is always removed."""

    def __init__(self, layer_module, V):
        self.layer = layer_module
        self.V = V
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output      # (b, T, d) residual
        V = self.V.to(hs.device)
        if V.shape[0] != hs.shape[-1]:
            raise RuntimeError(
                f"AblateHook: V rows {V.shape[0]} != residual dim {hs.shape[-1]} "
                f"— V must be (d, k) for this layer's hidden size.")
        if V.shape[1] == 0:
            return output                                            # k=0: identity (the floor)
        # fp32 projection (B2): Gemma-2 residuals carry massive-activation dims (~1e3–1e4); a bf16
        # `h − (hV)Vᵀ` incurs DIRECTION-DEPENDENT catastrophic cancellation (worse when V aligns with
        # a massive dim), which would bias the exact real−random headline. Project in fp32, cast back.
        Vf = V.float()
        new = (hs.float() - (hs.float() @ Vf) @ Vf.transpose(-2, -1)).to(hs.dtype)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


def _ablate_and_run(block, x, V):
    """Run `block(x)` with an AblateHook(block, V) live — used to assert shape-mismatch RAISES."""
    with AblateHook(block, V):
        return block(x)


# ===========================================================================
# Weight-free torch-hook dry-run (prereg §13) — the REAL AblateHook on a tiny stack. CPU only;
# complements the pure-logic dry-run in infra.pg.latablate.
# ===========================================================================

def dry_run_torch() -> dict:
    import numpy as np
    import torch

    from infra.pg.latablate import estimate_disabling_subspace, project_out

    torch.manual_seed(0)
    d, T, b, r = 16, 5, 3, 2

    class _TinyBlock(torch.nn.Module):
        """A decoder-layer stand-in: output[0] = x + a fixed shift (the planted 'deletion')."""
        def __init__(self, shift):
            super().__init__()
            self.register_buffer("shift", shift)

        def forward(self, x):
            return (x + self.shift,)

    Q = torch.linalg.qr(torch.randn(d, d))[0]
    U = Q[:, :r]                                          # planted disabling subspace (d, r)
    shift = (5.0 * torch.randn(b, T, r)) @ U.T            # (b,T,d), in span(U)
    x = torch.randn(b, T, d)
    intact_out = x.clone()                               # intact residual at this layer (no shift)
    deleted = _TinyBlock(shift)
    deleted_out = deleted(x)[0]                          # x + shift (the deleted residual)

    # estimate V from the (deleted − intact) residual differences (the real estimator path).
    D = (deleted_out - intact_out).reshape(b * T, d).numpy()
    V = torch.tensor(estimate_disabling_subspace(D, r), dtype=torch.float32)

    # ---- check A: the torch hook computes EXACTLY the numpy core (h − (hV)Vᵀ) ----
    with AblateHook(deleted, V):
        out = deleted(x)[0]
    np_expected = project_out(deleted_out.reshape(b * T, d).numpy(), V.numpy()).reshape(b, T, d)
    assert np.allclose(out.numpy(), np_expected, atol=1e-5), "AblateHook != numpy project_out core"

    # ---- check B: Δ removed — projecting span(V) out of deleted == out of intact (NOT == intact;
    #      the projector also removes intact's in-span component, exactly the pure-core check 1). Cast
    #      to the hook's dtype: project_out returns numpy float64, and torch.allclose RAISES on a
    #      float32-vs-float64 compare (numpy.allclose silently upcasts; the torch gate caught this). ----
    intact_proj = torch.tensor(
        project_out(intact_out.reshape(b * T, d).numpy(), V.numpy()).reshape(b, T, d),
        dtype=out.dtype)
    assert torch.allclose(out, intact_proj, atol=1e-4), "Δ not removed (proj-deleted != proj-intact)"

    # ---- check C: idempotent (P² = P), never zeros the residual ----
    once = project_out(deleted_out.reshape(b * T, d).numpy(), V.numpy())
    twice = project_out(once, V.numpy())
    assert np.allclose(once, twice, atol=1e-6), "projector not idempotent"
    assert float(out.abs().sum()) > 0.0, "AblateHook zeroed the residual"

    # ---- check D: residual-dim mismatch RAISES (no silent broadcast) ----
    badV = torch.randn(d + 1, r)
    _assert_raises(lambda: _ablate_and_run(deleted, x, badV), RuntimeError)

    # ---- check E: k=0 (V is (d,0)) is a no-op (the floor) ----
    with AblateHook(deleted, torch.zeros(d, 0)):
        out0 = deleted(x)[0]
    assert torch.allclose(out0, deleted_out, atol=1e-6), "k=0 AblateHook not a no-op"

    return {"hook_matches_numpy_core": "pass", "delta_removed": "pass",
            "idempotent_never_zero": "pass", "shape_raises": "pass", "k0_noop": "pass"}


# ===========================================================================
# L16 last-token residual harvest (both models, row-aligned) — the subspace-estimation input.
# Left-pad so position -1 is the real last token for every item (the readout position); both
# models see IDENTICAL (ids, am, pv) per batch, so deleted/intact rows are paired by item. No
# length-bucketing is needed (unlike latloc's donor swap) — the projector is input-independent.
# ===========================================================================

def harvest_pair(pg_after, pg_before, items, proc, gray_img, device, layer, batch_size=4):
    import numpy as np
    import torch

    from infra.pg.gemma_scope import resolve_gemma2_layers

    a_layers = resolve_gemma2_layers(pg_after.model.language_model)    # deleted (subject)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)   # intact (donor/ceiling)
    tok = proc.tokenizer
    orig_side = tok.padding_side
    tok.padding_side = "left"

    del_rows, int_rows = [], []
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        prompts = [it["prompt"] for it in batch]
        inp = proc(text=prompts, images=[gray_img] * len(batch),
                   return_tensors="pt", padding="longest").to(device)
        ids, am = inp["input_ids"], inp["attention_mask"]
        pv = inp["pixel_values"].to(torch.bfloat16)
        di = _capture(pg_after, a_layers, [layer], ids, am, pv)[layer][:, -1, :]    # (b, d) last token
        ii = _capture(pg_before, b_layers, [layer], ids, am, pv)[layer][:, -1, :]
        del_rows.append(di.float().cpu().numpy())
        int_rows.append(ii.float().cpu().numpy())

    tok.padding_side = orig_side
    deleted_last = np.concatenate(del_rows, 0)
    intact_last = np.concatenate(int_rows, 0)
    assert deleted_last.shape == intact_last.shape, "harvest row mismatch"
    return deleted_last, intact_last


# ===========================================================================
# Eval the DELETED model under a fixed projector at the locus → mmlu_blankimg per-item + acc.
# ===========================================================================

def eval_under_projector(model, a_layers, V_np, proc, device, split_pred, layer, n_per=None):
    """mmlu_blankimg on the deleted model with AblateHook(layer, V_np) live during every forward.
    V_np is numpy (d, k); k=0 (empty) → plain deleted eval (the floor). Returns the mmlu_blankimg
    dict (acc + per_item + per_item_hash, paired by item to the floor/intact runs on split_pred)."""
    import torch

    from infra.pg import text_eval as TE

    if V_np.shape[1] == 0:
        return TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, item_filter=split_pred)
    V = torch.tensor(V_np, dtype=torch.float32, device=device)     # fp32 projector buffer (B2)
    with AblateHook(a_layers[layer], V):
        return TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, item_filter=split_pred)


def _paired(a, b):
    """Assert two mmlu_blankimg results are per-item aligned (same hashes, same order) and return
    their per_item 0/1 arrays — the bootstrap requires item-pairing."""
    import numpy as np
    assert a["per_item_hash"] == b["per_item_hash"], "per-item hashes misaligned (reordering?)"
    return np.array(a["per_item"], float), np.array(b["per_item"], float)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="weight-free §13 dry-run (pure logic + real AblateHook), CPU, no model")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--headline-k", type=int, default=-1,
                    help="FROZEN k* from discovery; required (>=0) for --eval-split confirmation.")
    ap.add_argument("--frozen-v-prefix", default="",
                    help="S3 prefix of the DISCOVERY run; confirmation LOADS its V_full.npy (the frozen "
                         "subspace) instead of re-deriving it (B1 freeze integrity). Required for confirmation.")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()

    if args.dry_run:
        from infra.pg.latablate import dry_run
        pure = dry_run()
        torchres = dry_run_torch()
        print(json.dumps({"pure_logic": pure, "torch_hooks": torchres}, indent=2, default=str))
        print("[latablate] WEIGHT-FREE DRY-RUN PASSED (pure logic + real AblateHook).")
        return

    assert args.out, "--out required for the GPU run"
    if args.eval_split == "confirmation":
        assert args.headline_k >= 0, "confirmation needs the FROZEN --headline-k from discovery"
        assert args.frozen_v_prefix, "confirmation needs --frozen-v-prefix (the discovery V_full.npy; B1)"
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
    from infra.pg import latablate as LA
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.aws_pg.jobs.latloc_cell import build_confirmation_items

    assert torch.cuda.is_available(), "latablate GPU run needs CUDA."
    device = "cuda"
    n_per = args.n_per_subject or None
    L = LA.L_LOCUS

    # EAGER loader (latloc Catch-2): sdpa mishandles PaliGemma's prefix-LM mask → wrong read-out.
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, attn_impl="eager")
    n_dec = assert_generation_layers(pg_after.model.language_model, LA.GEMMA2_N_LAYERS)
    a_layers = resolve_gemma2_layers(pg_after.model.language_model)
    print(f"[latablate] eager loader OK; firewall {n_dec}==26; gate={gate['decoder_max_abs_delta_vs_PG']:.3g}")

    eval_pred = Wp.is_grid_subset if args.eval_split == "discovery" else Wp.is_confirmation

    # ---- preflight structural gates (prereg §9 step 1) — recomputed on the SAME items, never quoted ----
    floor_res = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=eval_pred)
    intact_res = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, item_filter=eval_pred)
    floor_pi, intact_pi = _paired(floor_res, intact_res)
    del_acc, int_acc = floor_res["acc"], intact_res["acc"]
    gap = int_acc - del_acc
    assert del_acc < 0.34, f"PREFLIGHT: deleted MCQ {del_acc:.3f} not near chance (~0.25) — wrong bucket?"
    assert gap >= 0.15, f"PREFLIGHT: intact−deleted gap {gap:.3f} < 0.15 — no dynamic range"
    print(f"[latablate] PREFLIGHT MCQ ({args.eval_split}): intact={int_acc:.3f} deleted={del_acc:.3f} gap={gap:.3f}")

    import hashlib
    import subprocess
    from infra.aws_pg import config as C
    K_max = max(LA.K_GRID)

    def rf_pi(ablate_pi):
        return LA.rf_from_items(ablate_pi, floor_pi, intact_pi)

    # ---- the disabling subspace V is estimated ONCE on DISCOVERY (is_grid_subset), banked to S3, and
    #      LOADED unchanged in confirmation (B1 freeze integrity — the two pg runs may land on different
    #      GPUs → different bf16 residuals → a different V; re-deriving would break the freeze). ----
    deleted_last = intact_last = spectrum = mean_top_align = None
    if args.eval_split == "discovery":
        disc_items, gray_img, _letter_ids = build_confirmation_items(proc, Wp.is_grid_subset, n_per_subject=n_per)
        deleted_last, intact_last = harvest_pair(pg_after, pg_before, disc_items, proc, gray_img, device, L)
        D = deleted_last - intact_last
        d_dim = D.shape[1]
        V_full = LA.estimate_disabling_subspace(D, K_max).astype(np.float32)   # (d, K_max), FROZEN
        assert LA.is_orthonormal(V_full), "estimated subspace not orthonormal"
        np.save(out / "V_full.npy", V_full)                                     # confirmation loads THIS
        spectrum = LA.singular_spectrum(D, k_top=K_max + 8)
        mean_dir = LA.mean_difference_direction(D)
        mean_top_align = float(abs(mean_dir[:, 0] @ V_full[:, 0])) if K_max >= 1 else None
        v_sha = hashlib.sha256(V_full.tobytes()).hexdigest()[:16]
        print(f"[latablate] harvested {D.shape[0]} discovery residuals (d={d_dim}); banked V_full.npy "
              f"sha={v_sha}; top-1 energy={spectrum['cum_energy_fraction'][0]:.3f} mean·top1={mean_top_align:.3f}")
    else:
        vpath = out / "V_full_frozen.npy"   # instance role (no --profile), mirroring the figures' _pull
        subprocess.run(["aws", "s3", "cp", f"s3://{C.BUCKET}/{args.frozen_v_prefix}/out/V_full.npy",
                        str(vpath), "--region", C.REGION], check=True)
        V_full = np.load(vpath).astype(np.float32)
        d_dim = V_full.shape[0]
        assert V_full.shape == (d_dim, K_max), f"frozen V shape {V_full.shape} != (d,{K_max})"
        assert LA.is_orthonormal(V_full), "frozen V not orthonormal"
        v_sha = hashlib.sha256(V_full.tobytes()).hexdigest()[:16]
        print(f"[latablate] LOADED frozen V from {args.frozen_v_prefix} sha={v_sha} shape={V_full.shape}")

    row = {
        "study": "latablate", "eval_split": args.eval_split, "locus_layer": L,
        "firewall_n_layers": n_dec, "attn_impl": "eager", "d_model": d_dim,
        "preflight": {"mcq_intact": int_acc, "mcq_deleted": del_acc, "gap": gap},
        "n_eval_items": int(len(floor_pi)), "v_sha": v_sha, "frozen_v_prefix": args.frozen_v_prefix or None,
        "k_grid": list(LA.K_GRID), "tau_restore": LA.TAU_RESTORE, "k_small": LA.K_SMALL,
        "delta_rand": LA.DELTA_RAND, "cloze_retain_delta": LA.CLOZE_RETAIN_DELTA,
        "singular_spectrum": spectrum, "mean_top1_alignment": mean_top_align,
        "raw": {"floor_per_item": floor_res["per_item"], "intact_per_item": intact_res["per_item"],
                "eval_item_hash": floor_res["per_item_hash"]},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }

    if args.eval_split == "discovery":
        # ---- DISCOVERY: full k-sweep, real + random(5) + shuffle(5) + energy-null → RF(k); nominate k* ----
        sweep = {}
        for k in LA.K_GRID:
            real_res = eval_under_projector(pg_after, a_layers, V_full[:, :k], proc, device, eval_pred, L, n_per)
            real_pi, _ = _paired(real_res, floor_res)
            rf_real = rf_pi(real_pi)
            _, rf_real_lo, rf_real_hi = LA.bootstrap_rf(real_pi, floor_pi, intact_pi)   # §10 F1 needs per-k CI
            rand_rfs = [rf_pi(_paired(eval_under_projector(
                pg_after, a_layers, LA.random_subspace(d_dim, k, sd), proc, device, eval_pred, L, n_per),
                floor_res)[0]) for sd in LA.RANDOM_NULL_SEEDS]
            shuf_rfs = [rf_pi(_paired(eval_under_projector(
                pg_after, a_layers, LA.shuffle_subspace(deleted_last, intact_last, k, sd),
                proc, device, eval_pred, L, n_per), floor_res)[0]) for sd in LA.SHUFFLE_NULL_SEEDS]
            # energy-matched null (audit S3): top-k of the DELETED residual itself (deterministic).
            energy_res = eval_under_projector(pg_after, a_layers, LA.energy_subspace(deleted_last, k),
                                              proc, device, eval_pred, L, n_per)
            rf_energy = rf_pi(_paired(energy_res, floor_res)[0])
            sweep[k] = {"rf_real": rf_real, "rf_real_ci_lo": rf_real_lo, "rf_real_ci_hi": rf_real_hi,
                        "acc_real": real_res["acc"],
                        "rf_random_mean": float(np.mean(rand_rfs)), "rf_random_sd": float(np.std(rand_rfs)),
                        "rf_random_seeds": [float(x) for x in rand_rfs],
                        "rf_shuffle_mean": float(np.mean(shuf_rfs)), "rf_shuffle_sd": float(np.std(shuf_rfs)),
                        "rf_shuffle_seeds": [float(x) for x in shuf_rfs],
                        "rf_energy": rf_energy,
                        "beats_random": LA.real_beats_random(rf_real, float(np.mean(rand_rfs)))}
            print(f"[latablate] k={k:2d} RF_real={rf_real:+.3f} RF_rand={np.mean(rand_rfs):+.3f} "
                  f"RF_shuf={np.mean(shuf_rfs):+.3f} RF_energy={rf_energy:+.3f} beats_rand={sweep[k]['beats_random']}")
        rf_by_k = {k: sweep[k]["rf_real"] for k in LA.K_GRID}
        # k* = minimal k that BOTH reaches τ AND beats random by δ (the low-rank-removable criterion).
        kstar = None
        for k in sorted(rf_by_k):
            if k <= LA.K_SMALL and rf_by_k[k] >= LA.TAU_RESTORE and sweep[k]["beats_random"]:
                kstar = int(k)
                break
        rf_monotone = LA.is_monotone_nondecreasing(rf_by_k, kstar)   # §11 dose-response descriptor (S1)
        row["discovery"] = {"sweep": {str(k): sweep[k] for k in LA.K_GRID},
                            "rf_by_k": {str(k): rf_by_k[k] for k in LA.K_GRID},
                            "minimal_k_star": kstar,
                            "minimal_k_star_tau_only": LA.minimal_k(rf_by_k),
                            "rf_monotone_to_kstar": rf_monotone}
        print(f"[latablate] DISCOVERY k*={kstar} (τ+random); τ-only k*={LA.minimal_k(rf_by_k)}; monotone={rf_monotone}")
    else:
        # ---- CONFIRMATION (once): frozen V + frozen k*; RF(k*) CI + equal-k random null + cloze sanity ----
        kstar = args.headline_k
        assert kstar in LA.K_GRID, f"--headline-k {kstar} not in the frozen grid {LA.K_GRID}"
        V_kstar = V_full[:, :kstar]
        real_res = eval_under_projector(pg_after, a_layers, V_kstar, proc, device, eval_pred, L, n_per)
        real_pi, _ = _paired(real_res, floor_res)
        rf_pt, rf_lo, rf_hi = LA.bootstrap_rf(real_pi, floor_pi, intact_pi)
        rand_rfs = [rf_pi(_paired(eval_under_projector(
            pg_after, a_layers, LA.random_subspace(d_dim, kstar, sd), proc, device, eval_pred, L, n_per),
            floor_res)[0]) for sd in LA.RANDOM_NULL_SEEDS]
        rand_mean = float(np.mean(rand_rfs))
        # cloze-under-projector content sanity (the fix must restore READOUT without destroying content)
        V_t = torch.tensor(V_kstar, dtype=torch.float32, device=device)
        with AblateHook(a_layers[L], V_t):
            cloze_abl = TE.mmlu_cloze(pg_after, None, device, processor=proc,
                                      n_per_subject=n_per, item_filter=eval_pred)
        cloze_floor = TE.mmlu_cloze(pg_after, None, device, processor=proc,
                                    n_per_subject=n_per, item_filter=eval_pred)
        cloze_retained = bool(cloze_abl["acc"] >= cloze_floor["acc"] - LA.CLOZE_RETAIN_DELTA)   # frozen (S2)
        verdict_lowrank = bool(rf_lo > LA.TAU_RESTORE and (rf_pt - rand_mean) > LA.DELTA_RAND
                               and kstar <= LA.K_SMALL)
        row["confirmation"] = {
            "k_star": int(kstar), "rf_real": rf_pt, "rf_ci_lo": rf_lo, "rf_ci_hi": rf_hi,
            "rf_random_mean": rand_mean, "rf_random_sd": float(np.std(rand_rfs)),
            "rf_random_seeds": [float(x) for x in rand_rfs],
            "margin_over_random": float(rf_pt - rand_mean),
            "cloze_acc_floor": cloze_floor["acc"], "cloze_acc_ablated": cloze_abl["acc"],
            "cloze_retained": cloze_retained,
            "exceeds_tau": bool(rf_lo > LA.TAU_RESTORE),
            "beats_random_by_delta": bool((rf_pt - rand_mean) > LA.DELTA_RAND),
            "verdict_H_lowrank": verdict_lowrank}
        print(f"[latablate] CONFIRMATION k*={kstar} RF={rf_pt:.3f} CI[{rf_lo:.3f},{rf_hi:.3f}] "
              f"rand={rand_mean:.3f} margin={rf_pt - rand_mean:+.3f} cloze_ret={cloze_retained} "
              f"H_lowrank={verdict_lowrank}")

    (out / "latablate.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[latablate] banked → {out / 'latablate.json'}")


if __name__ == "__main__":
    main()
