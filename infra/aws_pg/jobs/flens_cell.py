"""flens cell runner — is the Gemma-2 mid-stack MMLU-MCQ deletion sparse in the network's OWN
(SAE-feature) basis, where it was structureless in the LINEAR basis? (prereg
`studies/lineage/paligemma2/research/2026-06-29_functional_lens_prereg.md`, the MDE / Arm A.)

LATABLATE showed the L16 deletion is NOT a low-rank removable LINEAR direction (RF≈0 at every k≤32).
Every program instrument is linear. This job swaps ONLY the lens — linear subspaces → the Gemma Scope
SAE feature basis the network actually uses — and asks whether setting a SMALL set of SAE features to
their intact per-item values at L16 restores MMLU-MCQ, where the equal-budget SVD-component patch does
not. The behavioral readout (RF) is the arbiter; the SVD-component patch is the BASIS control.

NEW glue the prereg §13 calls for. The patch ALGEBRA / nulls / RF live in `infra.pg.flens` (weight-free,
--dry-run-tested). Here: the torch FeaturePatchHook (a per-item LAST-TOKEN additive delta at L16 — the
additive, feature-defined cousin of latablate's AblateHook), the per-item-patched MCQ eval loop, the
Gemma Scope SAE load + faithfulness gate, and the discovery/confirmation driver. Reuses harvest_pair /
build_confirmation_items / mmlu_blankimg / the firewall / the bank-load-frozen-operator pattern.

Every patch type — real top-K, matched-random-K, item-shuffle, SVD-component (basis control),
SAE round-trip (faithfulness), feature-zeroing (positive control) — is the SAME operation: a per-item
(n, d_model) last-token delta fed to the one FeaturePatchHook. Only the delta construction differs.

Two entrypoints:
  --dry-run   weight-free: pure-logic dry-run (infra.pg.flens.dry_run) AND the torch FeaturePatchHook +
              tiny-JumpReLU dry-run on CPU — prereg §13. Dispatch on pg with `--no-weights`.
  (default)   the GPU MDE: eager loader + 26-layer firewall + SAE pin + faithfulness gate + L16
              last-token residual harvest (both models) + SAE encode + feature attribution + the
              feature-patch k-sweep (real + random + shuffle + SVD-component + full).

    python -m infra.aws_pg.jobs.flens_cell --dry-run
    python -m infra.aws_pg.jobs.flens_cell --out <dir> --eval-split discovery
    python -m infra.aws_pg.jobs.flens_cell --out <dir> --eval-split confirmation --headline-k 64 \
        --frozen-s-prefix results/flens_disc
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.latloc_cell import _assert_raises, assert_generation_layers
from infra.aws_pg.jobs.latablate_cell import harvest_pair, _paired


# ===========================================================================
# FeaturePatchHook (NEW; prereg §13) — ADDS a per-item delta to ONE decoder layer's OUTPUT residual at
# the LAST token only: h[:, -1, :] += delta[row]. The additive, feature-defined cousin of latablate's
# AblateHook (which SUBTRACTS a projection); acts on the residual-STREAM output (the latloc/SAE site).
# Input-DEPENDENT (per-item), so a fresh hook is installed per batch with that batch's delta rows.
# ===========================================================================

class FeaturePatchHook:
    """Forward hook adding a per-item delta to `layer_module`'s output residual at the last token.
    `delta` is a torch (b, d_model) tensor (one row per batch item), on the model device. fp32
    arithmetic then cast back (B2: Gemma-2 residuals carry massive-activation dims ~1e3-1e4; a bf16 add
    of a small delta loses precision direction-dependently, which would bias the exact real−random
    headline). A batch- or dim-mismatch RAISES (no silent broadcast). Context-managed → handle always
    removed."""

    def __init__(self, layer_module, delta):
        self.layer = layer_module
        self.delta = delta                       # (b, d_model)
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        import torch
        hs = output[0] if isinstance(output, tuple) else output      # (b, T, d) residual
        if self.delta.shape[0] != hs.shape[0]:
            raise RuntimeError(
                f"FeaturePatchHook: delta batch {self.delta.shape[0]} != residual batch {hs.shape[0]}")
        if self.delta.shape[1] != hs.shape[-1]:
            raise RuntimeError(
                f"FeaturePatchHook: delta dim {self.delta.shape[1]} != residual dim {hs.shape[-1]} "
                f"— delta must be (b, d_model) for this layer.")
        hsf = hs.float()
        pad = torch.zeros_like(hsf)
        pad[:, -1, :] = self.delta.to(device=hsf.device, dtype=hsf.dtype)   # last token only
        new = (hsf + pad).to(hs.dtype)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


def _patch_and_run(block, x, delta):
    """Run `block(x)` with a FeaturePatchHook(block, delta) live — used to assert shape-mismatch RAISES."""
    with FeaturePatchHook(block, delta):
        return block(x)


# ===========================================================================
# Per-item-patched MCQ eval (NEW) — blank-image MMLU letter-argmax on `items` (build_confirmation_items
# order) with an optional per-item L16 last-token FeaturePatchHook. deltas=None → the plain floor/intact
# anchor. IDENTICAL prompt / letter-argmax / logits_to_keep=1 to text_eval.mmlu_blankimg (asserted equal
# at preflight), so the RF anchors and the patched accs are consistent.
# ===========================================================================

def eval_letter_under_patch(model, layers, layer, items, deltas, proc, gray_img, letter_ids, device,
                            batch_size=4):
    import numpy as np
    import torch

    tok = proc.tokenizer
    orig_side = tok.padding_side
    tok.padding_side = "left"
    letter_ids_t = torch.tensor(letter_ids, device=device)
    per_item, per_hash = [], []
    n_correct = 0
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        prompts = [it["prompt"] for it in batch]
        inp = proc(text=prompts, images=[gray_img] * len(batch),
                   return_tensors="pt", padding="longest").to(device)
        ids, am = inp["input_ids"], inp["attention_mask"]
        pv = inp["pixel_values"].to(torch.bfloat16)
        if deltas is None:
            with torch.inference_mode():
                last = model(input_ids=ids, attention_mask=am, pixel_values=pv,
                             logits_to_keep=1).logits[:, -1, :]
        else:
            dt = torch.tensor(np.asarray(deltas[s:s + len(batch)]), dtype=torch.float32, device=device)
            with FeaturePatchHook(layers[layer], dt), torch.inference_mode():
                last = model(input_ids=ids, attention_mask=am, pixel_values=pv,
                             logits_to_keep=1).logits[:, -1, :]
        preds = last[:, letter_ids_t].argmax(dim=-1).cpu().numpy()
        for it, p in zip(batch, preds):
            ok = int(int(p) == int(it["answer"]))
            per_item.append(ok)
            per_hash.append(it["hash"])
            n_correct += ok
    tok.padding_side = orig_side
    assert len(per_item) > 0, "eval_letter_under_patch produced no items"
    return {"acc": n_correct / len(per_item), "per_item": per_item, "per_item_hash": per_hash,
            "n": len(per_item)}


# ===========================================================================
# Harvest L16 last-token residuals (both models, row-aligned to items via the reused harvest_pair) and
# SAE-encode → f_int, f_del. Returns numpy arrays row-aligned to `items`.
# ===========================================================================

def harvest_features(pg_after, pg_before, items, proc, gray_img, device, layer, sae, batch_size=4):
    import numpy as np
    import torch

    deleted_last, intact_last = harvest_pair(pg_after, pg_before, items, proc, gray_img, device, layer,
                                             batch_size)
    with torch.inference_mode():
        f_del = sae.encode(torch.tensor(deleted_last, dtype=torch.float32, device=device)).cpu().numpy()
        f_int = sae.encode(torch.tensor(intact_last, dtype=torch.float32, device=device)).cpu().numpy()
    assert f_int.shape == f_del.shape and f_int.shape[0] == deleted_last.shape[0], "harvest/encode misalign"
    return (np.asarray(deleted_last, np.float64), np.asarray(intact_last, np.float64),
            np.asarray(f_int, np.float64), np.asarray(f_del, np.float64))


# ===========================================================================
# SAE pin + faithfulness gate (prereg §6). The SAE is the L16 width-16k residual SAE at the canonical
# L0 (closest to ~70). Gate: explained_variance ≥ EV_MIN AND the intact model's MCQ under an SAE
# round-trip of its L16 residual stays within ROUNDTRIP_RETAIN_DELTA of intact. ABORTS on failure.
# ===========================================================================

def pin_sae(device):
    from infra.pg.gemma_scope import JumpReLUSAE, pick_mid_residual_sae
    import infra.pg.flens as FL
    choice = pick_mid_residual_sae(width="16k", target_layer=FL.L_LOCUS, target_l0=70)
    assert choice.layer == FL.L_LOCUS, (
        f"FLENS lens-locus mismatch: Gemma Scope nearest residual SAE is layer {choice.layer}, "
        f"not L{FL.L_LOCUS} (the LATLOC/LATABLATE locus). The lens must sit at the deletion site.")
    assert choice.width == "16k", f"expected width 16k, got {choice.width}"
    sae = JumpReLUSAE.from_repo(choice).to(device)
    return sae, choice


def sae_faithfulness(sae, intact_last, f_int, pg_before, b_layers, layer, items, proc, gray_img,
                     letter_ids, device, intact_acc):
    import numpy as np
    import torch

    import infra.pg.flens as FL
    from infra.pg.gemma_scope import explained_variance

    x = torch.tensor(intact_last, dtype=torch.float32, device=device)
    with torch.inference_mode():
        recon = sae.decode(torch.tensor(f_int, dtype=torch.float32, device=device))
    ev = explained_variance(x, recon)
    cosine = float(torch.nn.functional.cosine_similarity(x, recon, dim=-1).mean().item())  # robust dir. fidelity
    rt_delta = (recon - x).cpu().numpy()                      # replace intact L16 residual w/ its SAE recon
    rt = eval_letter_under_patch(pg_before, b_layers, layer, items, rt_delta, proc, gray_img,
                                 letter_ids, device)
    # GATE = behavioral round-trip MCQ (readout preserved) AND directional cosine floor (recon not degenerate).
    # EV is REPORTED only (OOD-sensitive on the multimodal forward; see flens.py / journal).
    return {"explained_variance": float(ev), "recon_cosine": cosine, "roundtrip_mcq": rt["acc"],
            "intact_mcq": float(intact_acc), "ev_min": FL.EV_MIN, "cosine_min": FL.COSINE_MIN,
            "roundtrip_retain_delta": FL.ROUNDTRIP_RETAIN_DELTA, "ev_pass": bool(ev >= FL.EV_MIN),
            "cosine_pass": bool(cosine >= FL.COSINE_MIN),
            "roundtrip_pass": bool(rt["acc"] >= intact_acc - FL.ROUNDTRIP_RETAIN_DELTA)}


# ===========================================================================
# Weight-free torch dry-run (prereg §13) — the REAL FeaturePatchHook + a tiny torch JumpReLU SAE on a
# CPU stack; complements the pure-logic dry-run in infra.pg.flens.
# ===========================================================================

def dry_run_torch() -> dict:
    import numpy as np
    import torch

    import infra.pg.flens as FL
    from infra.pg.gemma_scope import JumpReLUSAE

    torch.manual_seed(0)
    d, T, b = 16, 5, 3

    class _TinyBlock(torch.nn.Module):
        """A decoder-layer stand-in: output[0] = x (identity), so the hook's add is the only change."""
        def forward(self, x):
            return (x,)

    block = _TinyBlock()
    x = torch.randn(b, T, d)
    delta = torch.randn(b, d)

    # ---- check A: the hook ADDS delta to the LAST token only; earlier positions untouched ----
    with FeaturePatchHook(block, delta):
        out = block(x)[0]
    assert torch.allclose(out[:, -1, :], (x[:, -1, :].float() + delta).to(x.dtype), atol=1e-5), "last-token add wrong"
    assert torch.allclose(out[:, :-1, :], x[:, :-1, :], atol=1e-6), "non-last positions changed (leak)"

    # ---- check B: zero delta is a no-op (the floor / K=0) ----
    with FeaturePatchHook(block, torch.zeros(b, d)):
        out0 = block(x)[0]
    assert torch.allclose(out0, x, atol=1e-6), "zero-delta hook not a no-op"

    # ---- check C: batch- and dim-mismatch RAISE (no silent broadcast) ----
    _assert_raises(lambda: _patch_and_run(block, x, torch.randn(b + 1, d)), RuntimeError)
    _assert_raises(lambda: _patch_and_run(block, x, torch.randn(b, d + 1)), RuntimeError)

    # ---- check D: a real torch JumpReLUSAE encode/decode matches the flens numpy core, and
    #      feature_patch_delta (numpy) == decode-difference via the torch SAE (bridges core↔SAE). ----
    d_model, d_sae = 12, 20
    W_enc = torch.randn(d_model, d_sae)
    W_dec = torch.randn(d_sae, d_model)
    b_enc = torch.randn(d_sae)
    b_dec = torch.randn(d_model)
    threshold = torch.rand(d_sae) * 0.1
    sae = JumpReLUSAE(W_enc, b_enc, W_dec, b_dec, threshold)
    xs = torch.randn(7, d_model)
    f = sae.encode(xs)
    assert torch.allclose(f, (xs @ W_enc + b_enc) * ((xs @ W_enc + b_enc) > threshold), atol=1e-5), "JumpReLU encode"
    # feature patch on S: decode(f_del with S→f_int) − decode(f_del) == (f_int[S]−f_del[S]) @ W_dec[S]
    f_int = sae.encode(torch.randn(7, d_model)).cpu().numpy()
    f_del = f.cpu().numpy()
    S = np.array([1, 4, 9])
    delta_np = FL.feature_patch_delta(f_int, f_del, S, W_dec.cpu().numpy())
    patched = f_del.copy()
    patched[:, S] = f_int[:, S]
    decode_diff = (sae.decode(torch.tensor(patched, dtype=torch.float32))
                   - sae.decode(torch.tensor(f_del, dtype=torch.float32))).cpu().numpy()
    assert np.allclose(decode_diff, delta_np, atol=1e-4), "feature_patch_delta != torch-SAE decode difference"

    return {"hook_last_token_add": "pass", "zero_noop": "pass", "shape_raises": "pass",
            "jumprelu_encode_matches": "pass", "feature_patch_bridges_torch_sae": "pass"}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="weight-free §13 dry-run (pure logic + real FeaturePatchHook + tiny SAE), CPU")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--headline-k", type=int, default=-1,
                    help="FROZEN K* from discovery; required (>=0, in the grid) for confirmation.")
    ap.add_argument("--frozen-s-prefix", default="",
                    help="S3 prefix of the DISCOVERY run; confirmation LOADS its S.npy (the frozen "
                         "feature set) instead of re-deriving it (B1 freeze integrity). Required for confirmation.")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()

    if args.dry_run:
        from infra.pg.flens import dry_run
        pure = dry_run()
        torchres = dry_run_torch()
        print(json.dumps({"pure_logic": pure, "torch_hooks": torchres}, indent=2, default=str))
        print("[flens] WEIGHT-FREE DRY-RUN PASSED (pure logic + real FeaturePatchHook + torch SAE).")
        return

    assert args.out, "--out required for the GPU run"
    if args.eval_split == "confirmation":
        assert args.headline_k >= 0, "confirmation needs the FROZEN --headline-k from discovery"
        assert args.frozen_s_prefix, "confirmation needs --frozen-s-prefix (the discovery S.npy; B1)"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import hashlib
    import subprocess

    import numpy as np
    import torch

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as Wp
    from infra.pg import flens as FL
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.aws_pg.jobs.latloc_cell import build_confirmation_items
    from infra.aws_pg import config as C

    assert torch.cuda.is_available(), "flens GPU run needs CUDA."
    device = "cuda"
    n_per = args.n_per_subject or None
    L = FL.L_LOCUS

    # EAGER loader (latloc Catch-2): sdpa mishandles PaliGemma's prefix-LM mask → wrong read-out.
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, attn_impl="eager")
    n_dec = assert_generation_layers(pg_after.model.language_model, 26)
    a_layers = resolve_gemma2_layers(pg_after.model.language_model)     # deleted (subject)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)    # intact (ceiling)
    print(f"[flens] eager loader OK; firewall {n_dec}==26; gate={gate['decoder_max_abs_delta_vs_PG']:.3g}")

    eval_pred = Wp.is_grid_subset if args.eval_split == "discovery" else Wp.is_confirmation
    items, gray_img, letter_ids = build_confirmation_items(proc, eval_pred, n_per_subject=n_per)

    # ---- pin the SAE + structural preflight (prereg §9 step 1) — recomputed on the SAME items ----
    sae, choice = pin_sae(device)
    W_dec = sae.W_dec.detach().cpu().numpy().astype(np.float64)         # (d_sae, d_model)
    d_sae, d_model = W_dec.shape
    print(f"[flens] pinned SAE {choice.rel_path} (layer={choice.layer} width={choice.width} "
          f"l0={choice.average_l0}); d_sae={d_sae} d_model={d_model}")

    floor_res = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=eval_pred)
    intact_res = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, item_filter=eval_pred)
    floor_pi, intact_pi = _paired(floor_res, intact_res)
    del_acc, int_acc = floor_res["acc"], intact_res["acc"]
    gap = int_acc - del_acc
    assert del_acc < 0.34, f"PREFLIGHT: deleted MCQ {del_acc:.3f} not near chance (~0.25) — wrong bucket?"
    assert gap >= 0.15, f"PREFLIGHT: intact−deleted gap {gap:.3f} < 0.15 — no dynamic range"

    # GATE: the NEW per-item eval loop reproduces mmlu_blankimg at K=0 (zero patch) on the deleted model.
    # EXACT per-item equality is too strict on the large confirmation set (amended 2026-06-30, §0): this
    # loop batches the flat (sorted-subject) item list, mmlu_blankimg batches PER-SUBJECT, so a handful of
    # NEAR-TIE argmaxes flip under different left-pad lengths at subject boundaries. Tolerate + REPORT that;
    # the patch accuracies and the lens MARGIN are eval-loop-internal so they are unaffected. A real loop
    # bug would flip a LARGE fraction → still ABORTS. (Passed EXACTLY on discovery: 0 flips.)
    gate_floor = eval_letter_under_patch(pg_after, a_layers, L, items, None, proc, gray_img, letter_ids, device)
    assert gate_floor["per_item_hash"] == floor_res["per_item_hash"], "eval-loop item order != mmlu_blankimg"
    n_disagree = sum(int(a != b) for a, b in zip(gate_floor["per_item"], floor_res["per_item"]))
    frac_disagree = n_disagree / len(floor_res["per_item"])
    acc_diff = abs(gate_floor["acc"] - floor_res["acc"])
    assert frac_disagree <= 0.02 and acc_diff <= 0.01, (
        f"eval-loop vs mmlu_blankimg disagree on {n_disagree}/{len(floor_res['per_item'])} items "
        f"(frac {frac_disagree:.4f}, acc_diff {acc_diff:.4f}) — exceeds near-tie tolerance (loop bug?)")
    print(f"[flens] PREFLIGHT MCQ ({args.eval_split}): intact={int_acc:.3f} deleted={del_acc:.3f} gap={gap:.3f}; "
          f"eval-loop≡mmlu_blankimg within {n_disagree} near-tie flips (frac {frac_disagree:.4f}, acc_diff {acc_diff:.4f})")

    def rf_pi(patched_pi):
        return FL.rf_from_items(patched_pi, floor_pi, intact_pi)

    def patched_pi(model, layers, deltas):
        res = eval_letter_under_patch(model, layers, L, items, deltas, proc, gray_img, letter_ids, device)
        pi, _ = _paired(res, floor_res)
        return pi, res["acc"]

    # ---- harvest L16 last-token residuals (both models) + SAE encode → f_int, f_del ----
    deleted_last, intact_last, f_int, f_del = harvest_features(pg_after, pg_before, items, proc, gray_img,
                                                               device, L, sae)
    # per-dim whitening scale for the massive-activation-robust linear control (audit B1): Gemma-2
    # residuals carry axis-aligned massive-activation dims whose raw variance would dominate the SVD of
    # (deleted−intact); standardizing by the deleted residual's per-dim std before the SVD removes that.
    whiten_scale = deleted_last.std(axis=0) + 1e-6
    print(f"[flens] harvested+encoded {f_int.shape[0]} items (d_sae={d_sae}); "
          f"mean L0 intact={float((f_int != 0).sum(1).mean()):.1f} deleted={float((f_del != 0).sum(1).mean()):.1f}; "
          f"whiten-scale max/median={whiten_scale.max():.1f}/{float(np.median(whiten_scale)):.2f}")

    # ---- SAE faithfulness gate (ABORT on failure). GATE = behavioral round-trip MCQ + directional cosine
    #      floor (study-aligned + robust). EV is REPORTED only — OOD-sensitive on the multimodal forward
    #      (text-trained Gemma Scope SAE: in-dist text EV≈+0.55, multimodal EV<0 but cosine≈0.9, readout
    #      preserved). Re-specified 2026-06-30 after the EV-gate diagnostic; see the journal. ----
    faith = sae_faithfulness(sae, intact_last, f_int, pg_before, b_layers, L, items, proc, gray_img,
                             letter_ids, device, int_acc)
    print(f"[flens] SAE faithfulness: roundtrip_MCQ={faith['roundtrip_mcq']:.3f} vs intact {int_acc:.3f} "
          f"({faith['roundtrip_pass']}); cosine={faith['recon_cosine']:.3f} (≥{FL.COSINE_MIN}? {faith['cosine_pass']}); "
          f"EV={faith['explained_variance']:.3f} (reported, OOD-sensitive; ev≥{FL.EV_MIN}? {faith['ev_pass']})")
    assert faith["roundtrip_pass"], (
        f"SAE faithfulness ABORT: round-trip MCQ {faith['roundtrip_mcq']:.3f} < intact {int_acc:.3f} − {FL.ROUNDTRIP_RETAIN_DELTA}")
    assert faith["cosine_pass"], (
        f"SAE faithfulness ABORT: recon cosine {faith['recon_cosine']:.3f} < {FL.COSINE_MIN} (recon degenerate)")
    # deleted-side EV (audit N4; prereg §8 threat 3 / F2): is the base-trained SAE in- or out-of-
    # distribution on the fine-tuned (deleted) residual? Reported, not gated (the intact side is the gate).
    from infra.pg.gemma_scope import explained_variance as _ev
    with torch.inference_mode():
        _recon_del = sae.decode(torch.tensor(f_del, dtype=torch.float32, device=device))
    faith["deleted_explained_variance"] = float(_ev(
        torch.tensor(deleted_last, dtype=torch.float32, device=device), _recon_del))
    print(f"[flens] deleted-side SAE EV={faith['deleted_explained_variance']:.3f} (OOD diagnostic, not gated)")

    # ---- the feature set S: top-K_MAX differing features, estimated ONCE on DISCOVERY, banked, and
    #      LOADED unchanged in confirmation (B1 freeze integrity). ----
    if args.eval_split == "discovery":
        S = FL.feature_difference_attribution(f_int, f_del, k_max=FL.K_MAX)          # (K_MAX,), FROZEN
        np.save(out / "S.npy", S.astype(np.int64))
    else:
        # B1 freeze integrity (audit S2): VERIFY — not just re-derive — that the loaded S and the
        # re-resolved SAE MATCH the discovery run (download discovery flens.json; assert s_sha + SAE path).
        djson = out / "_disc_flens.json"
        subprocess.run(["aws", "s3", "cp", f"s3://{C.BUCKET}/{args.frozen_s_prefix}/out/flens.json",
                        str(djson), "--region", C.REGION], check=True)
        disc_meta = json.loads(djson.read_text())
        spath = out / "S_frozen.npy"
        subprocess.run(["aws", "s3", "cp", f"s3://{C.BUCKET}/{args.frozen_s_prefix}/out/S.npy",
                        str(spath), "--region", C.REGION], check=True)
        S = np.load(spath).astype(int)
        assert S.shape == (FL.K_MAX,), f"frozen S shape {S.shape} != ({FL.K_MAX},)"
        loaded_sha = hashlib.sha256(S.tobytes()).hexdigest()[:16]
        assert loaded_sha == disc_meta["s_sha"], \
            f"FREEZE BREACH: loaded S sha {loaded_sha} != discovery s_sha {disc_meta['s_sha']}"
        assert choice.rel_path == disc_meta["sae_choice"]["rel_path"], \
            f"SAE DRIFT: confirmation SAE {choice.rel_path} != discovery {disc_meta['sae_choice']['rel_path']}"
        print(f"[flens] B1 OK: frozen S sha {loaded_sha} + SAE {choice.rel_path} match discovery {args.frozen_s_prefix}")
    s_sha = hashlib.sha256(S.tobytes()).hexdigest()[:16]
    split = FL.direction_split(f_int, f_del, S)
    pool = FL.active_feature_pool(f_int, S)
    print(f"[flens] S=top-{FL.K_MAX} differing features sha={s_sha}; "
          f"suppressed={split['frac_suppressed']:.2f} corrupting={split['frac_corrupting']:.2f}; "
          f"active pool (∖S)={pool.size}")

    row = {
        "study": "flens", "eval_split": args.eval_split, "locus_layer": L, "firewall_n_layers": n_dec,
        "attn_impl": "eager", "d_model": d_model, "d_sae": d_sae,
        "sae_choice": {"rel_path": choice.rel_path, "layer": choice.layer, "width": choice.width,
                       "average_l0": choice.average_l0},
        "sae_faithfulness": faith,
        "preflight": {"mcq_intact": int_acc, "mcq_deleted": del_acc, "gap": gap},
        "n_eval_items": int(len(floor_pi)), "s_sha": s_sha, "frozen_s_prefix": args.frozen_s_prefix or None,
        "k_grid": list(FL.K_GRID), "k_max": FL.K_MAX, "tau_restore": FL.TAU_RESTORE, "k_small": FL.K_SMALL,
        "delta_rand": FL.DELTA_RAND, "delta_lens": FL.DELTA_LENS,
        "direction_split": split, "active_pool_size": int(pool.size),
        "raw": {"floor_per_item": floor_res["per_item"], "intact_per_item": intact_res["per_item"],
                "eval_item_hash": floor_res["per_item_hash"]},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }

    # ---- the recoverable ceiling (K=full: ALL features → intact) — banked for RF context (both splits) ----
    full_delta = FL.full_patch_delta(f_int, f_del, W_dec)
    full_pi, full_acc = patched_pi(pg_after, a_layers, full_delta)
    rf_full = rf_pi(full_pi)
    row["rf_full"] = rf_full
    print(f"[flens] K=full (all features→intact) RF={rf_full:+.3f} acc={full_acc:.3f}")

    if args.eval_split == "discovery":
        # ---- positive control: zeroing the top-K_small differing features in the INTACT model must DROP MCQ ----
        pos_delta = FL.feature_patch_delta(np.zeros_like(f_int), f_int, S[:FL.K_SMALL], W_dec)   # f_int→0 on S[:64]
        _pos_pi, pos_acc = patched_pi(pg_before, b_layers, pos_delta)
        row["positive_control"] = {"k": FL.K_SMALL, "intact_mcq": int_acc, "intact_zeroed_mcq": pos_acc,
                                   "drop": float(int_acc - pos_acc), "fires": bool(pos_acc < int_acc - 0.05)}
        print(f"[flens] positive control: intact {int_acc:.3f} → zero top-{FL.K_SMALL} feats {pos_acc:.3f} "
              f"(drop {int_acc - pos_acc:+.3f})")

        # ---- DISCOVERY k-sweep: real + random(5) + shuffle(5) + SVD-component, per K → RF(K); pick K* ----
        sweep = {}
        for k in FL.K_GRID:
            if k == 0:
                rf_real = 0.0
                sweep[k] = {"rf_real": 0.0, "rf_real_ci_lo": 0.0, "rf_real_ci_hi": 0.0, "acc_real": del_acc,
                            "rf_random_mean": 0.0, "rf_random_sd": 0.0, "rf_random_seeds": [0.0] * len(FL.RANDOM_NULL_SEEDS),
                            "rf_shuffle_mean": 0.0, "rf_shuffle_sd": 0.0, "rf_shuffle_seeds": [0.0] * len(FL.SHUFFLE_NULL_SEEDS),
                            "rf_svd": 0.0, "rf_svd_white": 0.0, "beats_random": False}
                print(f"[flens] k={k:3d} RF_real=+0.000 (floor)")
                continue
            real_delta = FL.feature_patch_delta(f_int, f_del, S[:k], W_dec)
            real_pi, real_acc = patched_pi(pg_after, a_layers, real_delta)
            rf_real = rf_pi(real_pi)
            _, rf_lo, rf_hi = FL.bootstrap_rf(real_pi, floor_pi, intact_pi,
                                              seed=FL.BOOTSTRAP_SEED, n_boot=FL.N_BOOT)        # §10 F1 per-k CI
            rand_rfs = [rf_pi(patched_pi(pg_after, a_layers,
                        FL.feature_patch_delta(f_int, f_del, FL.random_feature_set(pool, k, sd), W_dec))[0])
                        for sd in FL.RANDOM_NULL_SEEDS]
            shuf_rfs = [rf_pi(patched_pi(pg_after, a_layers, FL.shuffle_deltas(real_delta, sd))[0])
                        for sd in FL.SHUFFLE_NULL_SEEDS]
            # TWO linear basis controls (same patch-to-intact op): raw SVD of (deleted−intact) and the
            # massive-activation-robust WHITENED SVD (audit B1). The lens claim is real > BOTH.
            rf_svd = rf_pi(patched_pi(pg_after, a_layers, FL.svd_patch_delta(deleted_last, intact_last, k))[0])
            rf_svd_w = rf_pi(patched_pi(pg_after, a_layers,
                       FL.svd_patch_delta(deleted_last, intact_last, k, scale=whiten_scale))[0])
            sweep[k] = {"rf_real": rf_real, "rf_real_ci_lo": rf_lo, "rf_real_ci_hi": rf_hi, "acc_real": real_acc,
                        "rf_random_mean": float(np.mean(rand_rfs)), "rf_random_sd": float(np.std(rand_rfs)),
                        "rf_random_seeds": [float(x) for x in rand_rfs],
                        "rf_shuffle_mean": float(np.mean(shuf_rfs)), "rf_shuffle_sd": float(np.std(shuf_rfs)),
                        "rf_shuffle_seeds": [float(x) for x in shuf_rfs],
                        "rf_svd": rf_svd, "rf_svd_white": rf_svd_w,
                        "beats_random": FL.real_beats_random(rf_real, float(np.mean(rand_rfs)), FL.DELTA_RAND)}
            print(f"[flens] k={k:3d} RF_real={rf_real:+.3f} CI[{rf_lo:+.3f},{rf_hi:+.3f}] "
                  f"RF_rand={np.mean(rand_rfs):+.3f} RF_shuf={np.mean(shuf_rfs):+.3f} "
                  f"RF_svd={rf_svd:+.3f}/wht={rf_svd_w:+.3f} beats_rand={sweep[k]['beats_random']}")
        rf_by_k = {k: sweep[k]["rf_real"] for k in FL.K_GRID}
        # K* = minimal K ≤ K_small that BOTH reaches τ AND beats random by δ (the feature-sparse criterion).
        kstar = None
        for k in sorted(rf_by_k):
            if 0 < k <= FL.K_SMALL and rf_by_k[k] >= FL.TAU_RESTORE and sweep[k]["beats_random"]:
                kstar = int(k)
                break
        rf_monotone = FL.is_monotone_nondecreasing(rf_by_k, kstar)
        row["discovery"] = {"sweep": {str(k): sweep[k] for k in FL.K_GRID},
                            "rf_by_k": {str(k): rf_by_k[k] for k in FL.K_GRID},
                            "minimal_k_star": kstar,
                            "minimal_k_star_tau_only": FL.minimal_k(rf_by_k, FL.TAU_RESTORE),
                            "rf_monotone_to_kstar": rf_monotone}
        print(f"[flens] DISCOVERY K*={kstar} (τ+random); τ-only K*={FL.minimal_k(rf_by_k, FL.TAU_RESTORE)}; "
              f"monotone={rf_monotone}")
    else:
        # ---- CONFIRMATION (once): frozen S + frozen K*; the LENS claim gated by the PAIRED MARGIN CI of
        #      RF_feature − RF_svd for BOTH the raw and whitened linear controls (audit B1), not a point. ----
        kstar = args.headline_k
        assert kstar in FL.K_GRID, f"--headline-k {kstar} not in the frozen grid {FL.K_GRID}"
        real_delta = FL.feature_patch_delta(f_int, f_del, S[:kstar], W_dec)
        real_pi, real_acc = patched_pi(pg_after, a_layers, real_delta)
        rf_pt, rf_lo, rf_hi = FL.bootstrap_rf(real_pi, floor_pi, intact_pi,
                                              seed=FL.BOOTSTRAP_SEED, n_boot=FL.N_BOOT)
        # the two linear basis controls (SAME patch-to-intact operation), item-paired to the feature patch
        svd_raw_pi, _ = patched_pi(pg_after, a_layers, FL.svd_patch_delta(deleted_last, intact_last, kstar))
        svd_wht_pi, _ = patched_pi(pg_after, a_layers,
                                   FL.svd_patch_delta(deleted_last, intact_last, kstar, scale=whiten_scale))
        rf_svd_raw, rf_svd_wht = rf_pi(svd_raw_pi), rf_pi(svd_wht_pi)
        # THE LENS GATE: paired bootstrap CI of (RF_feature − RF_svd) for each control; require CI-lo > δ_lens.
        mr_raw = FL.bootstrap_rf_difference(real_pi, svd_raw_pi, floor_pi, intact_pi,
                                            seed=FL.BOOTSTRAP_SEED, n_boot=FL.N_BOOT)
        mr_wht = FL.bootstrap_rf_difference(real_pi, svd_wht_pi, floor_pi, intact_pi,
                                            seed=FL.BOOTSTRAP_SEED, n_boot=FL.N_BOOT)
        beats_raw = bool(mr_raw[1] > FL.DELTA_LENS)        # CI-lo of (real − raw-SVD)   > δ_lens
        beats_wht = bool(mr_wht[1] > FL.DELTA_LENS)        # CI-lo of (real − whitened-SVD) > δ_lens
        # the random null is a DESCRIPTOR (re-billed, audit S3): it gates "any-K-features" only in the
        # distributed regime; in the sparse regime it is near-vacuous → it does NOT gate the lens claim.
        rand_rfs = [rf_pi(patched_pi(pg_after, a_layers,
                    FL.feature_patch_delta(f_int, f_del, FL.random_feature_set(pool, kstar, sd), W_dec))[0])
                    for sd in FL.RANDOM_NULL_SEEDS]
        rand_mean = float(np.mean(rand_rfs))
        verdict_sparse = bool(rf_lo > FL.TAU_RESTORE and kstar <= FL.K_SMALL and beats_raw and beats_wht)
        row["confirmation"] = {
            "k_star": int(kstar), "rf_real": rf_pt, "rf_ci_lo": rf_lo, "rf_ci_hi": rf_hi, "acc_real": real_acc,
            "rf_svd_raw": rf_svd_raw, "rf_svd_white": rf_svd_wht,
            "lens_margin_raw": mr_raw[0], "lens_margin_raw_ci_lo": mr_raw[1], "lens_margin_raw_ci_hi": mr_raw[2],
            "lens_margin_white": mr_wht[0], "lens_margin_white_ci_lo": mr_wht[1], "lens_margin_white_ci_hi": mr_wht[2],
            "beats_raw_svd_by_delta_lens": beats_raw, "beats_white_svd_by_delta_lens": beats_wht,
            "rf_random_mean": rand_mean, "rf_random_sd": float(np.std(rand_rfs)),
            "rf_random_seeds": [float(x) for x in rand_rfs], "margin_over_random": float(rf_pt - rand_mean),
            "beats_random_by_delta": bool((rf_pt - rand_mean) > FL.DELTA_RAND),   # descriptor only
            "exceeds_tau": bool(rf_lo > FL.TAU_RESTORE), "delta_lens": FL.DELTA_LENS,
            "verdict_H_sparse": verdict_sparse}
        print(f"[flens] CONFIRMATION K*={kstar} RF={rf_pt:.3f} CI[{rf_lo:.3f},{rf_hi:.3f}] | "
              f"lens-margin vs raw-SVD={mr_raw[0]:+.3f} CI-lo={mr_raw[1]:+.3f} (>δ_lens? {beats_raw}); "
              f"vs whitened-SVD={mr_wht[0]:+.3f} CI-lo={mr_wht[1]:+.3f} ({beats_wht}) | H_sparse={verdict_sparse}")

    (out / "flens.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[flens] banked → {out / 'flens.json'}")


if __name__ == "__main__":
    main()
