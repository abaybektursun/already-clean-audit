"""latloc cell runner — causal depth-resolved localization of the Gemma-2 latent head-only
MMLU deletion via cross-MODEL residual transplant (prereg
`studies/lineage/paligemma2/research/2026-06-28_latent_locus_prereg.md`, Arm 1 / the MDE).

This is the NEW glue the prereg §13 calls for: the within-model logprob-recovery driver
(`modal_pg_w5.py:585-637`) only shares the raw hooks; the cross-MODEL intact→deleted transplant
computing accuracy-RF, the depth-matched wrong-donor ΔRF, and the length-bucketed alignment are
all new. Pure analysis math lives in `infra.pg.latloc` (weight-free, unit-tested by --dry-run).

Two entrypoints:
  --dry-run   weight-free: runs the pure-logic dry-run (infra.pg.latloc.dry_run) AND the torch-hook
              dry-run (real ResidualCache/PatchHook on a tiny stack) on CPU — prereg §13, §9 step 2.
              Dispatch on pg with `--no-weights` (CPU r-family). NO model, NO GPU.
  (default)   the GPU MDE: eager loader + 26-layer firewall + structural preflight gates + the
              length-bucketed correct/wrong-donor patch sweep; banks RAW per-layer per-item arrays
              (ΔRF/step/greedy computed downstream from the bank, blind-protocol §9).

    python -m infra.aws_pg.jobs.latloc_cell --dry-run
    python -m infra.aws_pg.jobs.latloc_cell --out <dir> --eval-split confirmation
"""
from __future__ import annotations

import argparse
import json
import pathlib


# ===========================================================================
# Generation firewall (Catch-1). relaw.n_decoder_layers does NOT assert 26/18 (it only
# checks the index set is non-empty — verified against the tree 2026-06-28). The numeric
# assertion the prereg §0 firewall relies on is enforced HERE, at the live model.
# ===========================================================================

def assert_generation_layers(language_model, expected: int) -> int:
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.latloc import GEMMA1_N_LAYERS, GEMMA2_N_LAYERS
    assert expected in (GEMMA1_N_LAYERS, GEMMA2_N_LAYERS), f"unexpected generation size {expected}"
    n = len(resolve_gemma2_layers(language_model))
    assert n == expected, (
        f"GENERATION FIREWALL: expected {expected} decoder layers for this generation, found {n}. "
        f"A latloc Gemma-2 run must see 26; importing an 18-layer Gemma-1 model here is a firewall breach.")
    return n


# ===========================================================================
# Weight-free torch-hook dry-run (prereg §13 checks 1 & 6) — the REAL hooks on a tiny stack.
# CPU only; complements the pure-logic dry-run in infra.pg.latloc.
# ===========================================================================

def _assert_raises(fn, exc):
    """Expected-raise assertion (test scaffolding, NOT error-hiding): the call MUST raise `exc`."""
    raised = False
    try:
        fn()
    except exc:
        raised = True
    assert raised, f"expected {exc.__name__} but nothing raised"


def dry_run_torch() -> dict:
    import torch
    from infra.pg.lm_patching import PatchHook, ResidualCache

    class _TinyBlock(torch.nn.Module):
        """A decoder-layer stand-in: returns a TUPLE whose [0] is the post-block residual,
        so ResidualCache/PatchHook (which read/replace output[0]) exercise their real paths."""
        def __init__(self, Wmat):
            super().__init__()
            self.register_buffer("W", Wmat)

        def forward(self, x):
            return (x @ self.W.T,)

    def _run(blocks, x):
        h = x
        for blk in blocks:
            h = blk(h)[0]
        return h

    torch.manual_seed(0)
    N, d, T = 6, 8, 4
    A = [torch.linalg.qr(torch.randn(d, d))[0] for _ in range(N)]

    def _small_rot():
        g = torch.randn(d, d)
        return torch.linalg.qr(torch.eye(d) + 0.4 * (g - g.T) / 2.0)[0]

    # deleted differs from intact at EVERY layer (small rotation each). So the mid-layer residual
    # genuinely differs between intact and deleted (the patch is non-trivial vs deleted) AND the
    # corrupted downstream layers recompute (the patched output differs from intact) → check 1.
    B = [A[i] @ _small_rot() for i in range(N)]
    intact = torch.nn.ModuleList([_TinyBlock(A[i]) for i in range(N)])
    deleted = torch.nn.ModuleList([_TinyBlock(B[i]) for i in range(N)])
    x = torch.randn(1, T, d)

    with ResidualCache(intact) as cache:
        _run(intact, x)
    intact_res = {L: cache.cache[L].clone() for L in cache.cache}
    out_intact = _run(intact, x)
    out_deleted = _run(deleted, x)

    # ---- check 1: middle-layer patch is non-trivial ----
    mid = 2  # upstream of corrupt_layer = N-1
    pm_all = torch.ones(1, T, dtype=torch.bool)
    with PatchHook(deleted[mid], intact_res[mid], pm_all, mode="resample"):
        out_patched = _run(deleted, x)
    assert not torch.allclose(out_patched, out_intact, atol=1e-4), \
        "patched == pure-intact: deleted downstream did not recompute (check 1 fail)"
    assert not torch.allclose(out_patched, out_deleted, atol=1e-4), \
        "patched == pure-deleted: the patch had no effect (check 1 fail)"

    # ---- check 6a: torch.where patches EXACTLY the masked positions, never zeros ----
    pm = torch.zeros(1, T, dtype=torch.bool)
    pm[0, [0, 2]] = True
    src = torch.randn(1, T, d)
    with PatchHook(deleted[mid], src, pm, mode="resample"), ResidualCache(deleted) as c_patched:
        _run(deleted, x)                              # ResidualCache registered after → sees patched output
    patched_mid = c_patched.cache[mid]
    with ResidualCache(deleted) as c_plain:
        _run(deleted, x)
    plain_mid = c_plain.cache[mid]
    assert torch.allclose(patched_mid[0, [0, 2]], src[0, [0, 2]].to(patched_mid.dtype), atol=1e-5), \
        "masked positions != src (torch.where wrong)"
    assert torch.allclose(patched_mid[0, [1, 3]], plain_mid[0, [1, 3]], atol=1e-5), \
        "UNMASKED positions changed (torch.where leaked)"
    assert float(patched_mid.abs().sum()) > 0.0, "never-zero invariant violated"

    # ---- check 6b: shape mismatch RAISES (no silent broadcast) ----
    bad = torch.randn(1, T, d + 1)
    _assert_raises(lambda: _patch_and_run(deleted, x, mid, bad, pm_all, _run), RuntimeError)
    return {"check1_middle_nontrivial": "pass", "check6_mask_exact_and_shape_raises": "pass"}


def _patch_and_run(blocks, x, L, src, pm, run_fn):
    from infra.pg.lm_patching import PatchHook
    with PatchHook(blocks[L], src, pm, mode="resample"):
        return run_fn(blocks, x)


# ===========================================================================
# Length-bucketed cross-MODEL patch sweep (the load-bearing NEW glue; Catch-5).
# Wrong-donors are drawn WITHIN a same-token-length bucket so the patched residual is shape-
# AND position-aligned (image↔image, text↔text); only the item CONTENT differs.
# ===========================================================================

def _bucketize(proc, items, gray_img):
    """Tokenize each item ONCE (single-item processor call → exact length, no pad) and group item
    indices by EXACT total token length (≥2 per bucket → a valid within-bucket donor; Catch-5). The
    bucketing strategy is isolated HERE so it can be swapped without touching the sweep/corrupt
    callers. Returns (enc_per, buckets, lengths)."""
    from infra.pg.latloc import length_buckets
    enc_per, lengths = [], []
    for it in items:
        e = proc(text=it["prompt"], images=gray_img, return_tensors="pt")
        enc_per.append(e)
        lengths.append(int(e["input_ids"].shape[1]))
    return enc_per, length_buckets(lengths, min_bucket=2), lengths


def _bucket_tensors(enc_per, idxs, device):
    import torch
    ids = torch.cat([enc_per[i]["input_ids"] for i in idxs], 0).to(device)
    am = torch.cat([enc_per[i]["attention_mask"] for i in idxs], 0).to(device)
    pv = torch.cat([enc_per[i]["pixel_values"] for i in idxs], 0).to(torch.bfloat16).to(device)
    assert (am == 1).all(), "same-length bucket must have NO padding (else positions misalign)"
    return ids, am, pv


def _letter_acc(model, ids, am, pv, letter_ids_t, ans):
    import torch
    with torch.inference_mode():
        last = model(input_ids=ids, attention_mask=am, pixel_values=pv,
                     logits_to_keep=1).logits[:, -1, :]
    return (last[:, letter_ids_t].argmax(dim=-1).cpu().numpy() == ans).astype(float)


MAX_CHUNK = 16   # cap items processed at once so the per-layer residual capture can't OOM a big bucket


def _chunks(idxs, max_chunk=MAX_CHUNK):
    """Split a same-length bucket (≥2 items) into chunks each of size in [2, max_chunk] — never a
    singleton (which would have no valid within-chunk donor). Caps the residual-capture memory."""
    n = len(idxs)
    if n <= max_chunk:
        return [idxs]
    nch = (n + max_chunk - 1) // max_chunk
    base, rem = divmod(n, nch)
    out, i = [], 0
    for c in range(nch):
        sz = base + (1 if c < rem else 0)
        out.append(idxs[i:i + sz]); i += sz
    return out


def _capture(model, layers, layer_idxs, ids, am, pv=None):
    """Forward once, capturing ONLY the residual at `layer_idxs` (output[0] per layer) — keyed by the
    ORIGINAL layer index. Capturing only the layers we patch (vs all 26) is what keeps a big chunk in
    memory. Returns {L: residual}."""
    import torch
    cache, handles = {}, []

    def _mk(L):
        def hook(m, a, o):
            cache[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    for L in layer_idxs:
        handles.append(layers[L].register_forward_hook(_mk(L)))
    with torch.inference_mode():
        if pv is not None:
            model(input_ids=ids, attention_mask=am, pixel_values=pv)
        else:
            model(input_ids=ids, attention_mask=am)
    for h in handles:
        h.remove()
    return cache


def bucketed_patch_sweep(pg_before, pg_after, proc, items, gray_img, letter_ids, device,
                         donor_seeds, layers=None):
    """intact→deleted RESTORATION sweep (prereg Arm 1). For each EXACT-length bucket: capture the
    intact (donor) residuals once; for each layer in `layers` patch correct-donor (this item's own
    intact residual) and wrong-donor for EACH seed (a within-bucket item shuffle), reading the
    choice-letter argmax at the last token. `layers=None` ⇒ all 26 (discovery profile); a subset
    (e.g. [L_onset]) ⇒ confirmation tests ONLY the frozen layer (blind, and ~25× cheaper). Returns
    per-item, paired by item:
      floor (n,), intact (n,), correct (nL,n), wrong (nSeeds,nL,n)  — NaN at unswept layers / dropped items.
    """
    import numpy as np
    import torch

    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.latloc import GEMMA2_N_LAYERS, donor_permutation
    from infra.pg.lm_patching import PatchHook, ResidualCache

    a_layers = resolve_gemma2_layers(pg_after.model.language_model)    # deleted (subject)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)   # intact (ceiling/donor)
    n_layers = len(a_layers)
    assert n_layers == GEMMA2_N_LAYERS, f"firewall: {n_layers} != {GEMMA2_N_LAYERS}"

    enc_per, buckets, lengths = _bucketize(proc, items, gray_img)
    n_used = sum(len(idx) for idx in buckets.values())
    print(f"[latloc] {len(items)} items → {len(buckets)} length-buckets covering {n_used} items "
          f"({len(items) - n_used} singletons dropped); retention={n_used/max(1,len(items)):.2%}")

    layer_list = list(range(n_layers)) if layers is None else list(layers)
    letter_ids_t = torch.tensor(letter_ids, device=device)
    nS = len(donor_seeds)
    floor = np.full(len(items), np.nan)
    intact = np.full(len(items), np.nan)
    correct = np.full((n_layers, len(items)), np.nan)
    wrong = np.full((nS, n_layers, len(items)), np.nan)
    answers = np.array([int(it["answer"]) for it in items])

    for _Llen, idxs in sorted(buckets.items()):
        for chunk in _chunks(list(idxs)):                  # cap memory; donor stays within-chunk
            ids, am, pv = _bucket_tensors(enc_per, chunk, device)
            ans = answers[chunk]
            floor[chunk] = _letter_acc(pg_after, ids, am, pv, letter_ids_t, ans)
            intact[chunk] = _letter_acc(pg_before, ids, am, pv, letter_ids_t, ans)
            intact_res = _capture(pg_before, b_layers, layer_list, ids, am, pv)   # ONLY swept layers
            donors = [donor_permutation(len(chunk), s) for s in donor_seeds]
            pm = torch.ones_like(ids, dtype=torch.bool)    # MCQ: no answer suffix → full prompt patchable (Catch-4)
            for L in layer_list:
                with PatchHook(a_layers[L], intact_res[L], pm, mode="resample"):
                    correct[L, chunk] = _letter_acc(pg_after, ids, am, pv, letter_ids_t, ans)
                for si, dperm in enumerate(donors):
                    with PatchHook(a_layers[L], intact_res[L][dperm], pm, mode="resample"):
                        wrong[si, L, chunk] = _letter_acc(pg_after, ids, am, pv, letter_ids_t, ans)

    return {"floor": floor, "intact": intact, "correct": correct, "wrong": wrong,
            "lengths": lengths, "n_buckets": len(buckets), "n_used": n_used, "layers": layer_list}


def corrupt_patch_at(pg_before, pg_after, proc, items, gray_img, letter_ids, device,
                     layer, donor_seeds):
    """Symmetric corrupt patch (prereg Arm 1 step 5): patch the DELETED residual into the INTACT
    forward at ONE `layer`; measure how much it DEGRADES intact MCQ — correct-donor (this item's own
    deleted residual) vs wrong-donor (a same-length item's). Item-specific causal directionality iff
    the correct donor degrades intact MORE than the wrong donor at that depth. Returns per-item
    intact_clean, corrupt_correct, corrupt_wrong (wrong averaged over seeds)."""
    import numpy as np
    import torch

    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.latloc import donor_permutation
    from infra.pg.lm_patching import PatchHook, ResidualCache

    a_layers = resolve_gemma2_layers(pg_after.model.language_model)    # deleted (donor of corruption)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)   # intact (patched-into)
    enc_per, buckets, _ = _bucketize(proc, items, gray_img)
    letter_ids_t = torch.tensor(letter_ids, device=device)
    answers = np.array([int(it["answer"]) for it in items])
    clean = np.full(len(items), np.nan)
    cc = np.full(len(items), np.nan)
    cw = np.full(len(items), np.nan)

    for _Llen, idxs in sorted(buckets.items()):
        for chunk in _chunks(list(idxs)):
            ids, am, pv = _bucket_tensors(enc_per, chunk, device)
            ans = answers[chunk]
            clean[chunk] = _letter_acc(pg_before, ids, am, pv, letter_ids_t, ans)
            del_res = _capture(pg_after, a_layers, [layer], ids, am, pv)[layer]   # only `layer` from deleted
            pm = torch.ones_like(ids, dtype=torch.bool)
            with PatchHook(b_layers[layer], del_res, pm, mode="resample"):        # this item's deleted residual → intact
                cc[chunk] = _letter_acc(pg_before, ids, am, pv, letter_ids_t, ans)
            wacc = []
            for s in donor_seeds:
                d = donor_permutation(len(chunk), s)
                with PatchHook(b_layers[layer], del_res[d], pm, mode="resample"):  # a DIFFERENT item's deleted residual
                    wacc.append(_letter_acc(pg_before, ids, am, pv, letter_ids_t, ans))
            cw[chunk] = np.mean(wacc, axis=0)

    return {"intact_clean": clean, "corrupt_correct": cc, "corrupt_wrong": cw}


# ===========================================================================
# Confirmation-item construction — reuse text_eval's EXACT prompt build (so accuracy matches
# the eval) but expose per-item prompt/answer/hash for the bucketed sweep.
# ===========================================================================

def build_confirmation_items(proc, split_pred, n_per_subject=None):
    """Reproduce mmlu_blankimg's 5-shot prompt for each item in the split (prereg §4). Returns
    (items, gray_img, letter_ids). split_pred(hash) selects the sealed half."""
    import hashlib
    import numpy as np
    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image

    from infra.pg import text_eval as TE

    tok = proc.tokenizer
    letter_ids = []
    for L in TE.CHOICES:
        ids = tok.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"' {L}' → {len(ids)} tokens"
        letter_ids.append(ids[0])
    assert len(set(letter_ids)) == 4, f"letter ids collide: {letter_ids}"

    ds = load_dataset(TE.MMLU_REPO, "all")
    shots_by_subj, test_by_subj = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        shots_by_subj[r["subject"]].append(r)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)

    items = []
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:5]
        assert len(shots) == 5, f"{subj}: {len(shots)} dev shots"
        prefix = TE._subject_header(subj) + "".join(
            TE._q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        for r in rows:
            h = hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12]
            if not split_pred(h):
                continue
            items.append({"prompt": prefix + TE._q_block(r["question"], r["choices"]),
                          "answer": int(r["answer"]), "hash": h})
    gray_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    assert items, "no confirmation items selected (over-restrictive split?)"
    return items, gray_img, letter_ids


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="weight-free §13 dry-run (pure logic + real hooks), CPU, no model")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], default="confirmation")
    ap.add_argument("--headline-layer", type=int, default=-1,
                    help="FROZEN onset from discovery; when >=0 (confirmation) the job banks the "
                         "ΔRF(L) bootstrap CI at this layer AND runs the symmetric corrupt patch at "
                         "it. When -1 (discovery) the job nominates the onset from the profile.")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()

    if args.dry_run:
        from infra.pg.latloc import dry_run
        pure = dry_run()
        torchres = dry_run_torch()
        print(json.dumps({"pure_logic": pure, "torch_hooks": torchres}, indent=2, default=str))
        print("[latloc] WEIGHT-FREE DRY-RUN PASSED (pure logic + real hooks).")
        return

    assert args.out, "--out required for the GPU run"
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
    from infra.pg import latloc as LL

    assert torch.cuda.is_available(), "latloc GPU run needs CUDA."
    device = "cuda"
    n_per = args.n_per_subject or None

    # EAGER loader (Catch-2): sdpa mishandles PaliGemma's prefix-LM mask → wrong read-out logits.
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, attn_impl="eager")
    n_dec = assert_generation_layers(pg_after.model.language_model, LL.GEMMA2_N_LAYERS)
    print(f"[latloc] eager loader OK; firewall {n_dec}==26; gate={gate['decoder_max_abs_delta_vs_PG']:.3g}")

    # established lineage split (REPRICE/latmech comparability) — see journal reconciliation note.
    split_pred = Wp.is_confirmation if args.eval_split == "confirmation" else Wp.is_grid_subset

    # ---- preflight structural gates (prereg §9 step 1) — recomputed, never quoted ----
    mcq_del = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=split_pred)
    mcq_int = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, item_filter=split_pred)
    del_acc, int_acc = mcq_del["acc"], mcq_int["acc"]
    gap = int_acc - del_acc
    assert del_acc < 0.34, f"PREFLIGHT: deleted MCQ {del_acc:.3f} not near chance (~0.25) — wrong bucket?"
    assert gap >= 0.15, f"PREFLIGHT: intact−deleted gap {gap:.3f} < 0.15 — no dynamic range"
    print(f"[latloc] PREFLIGHT MCQ: intact={int_acc:.3f} deleted={del_acc:.3f} gap={gap:.3f}")

    # ---- the MDE: length-bucketed correct + 5-seed wrong-donor patch sweep ----
    donor_seeds = list(LL.WRONG_DONOR_SEEDS)
    items, gray_img, letter_ids = build_confirmation_items(proc, split_pred, n_per_subject=n_per)
    # discovery sweeps ALL 26 layers (nominate the onset); confirmation sweeps ONLY the frozen onset
    # layer (blind: test once at L_d) — and ~25× cheaper.
    sweep_layers = None if args.headline_layer < 0 else [args.headline_layer]
    sweep = bucketed_patch_sweep(pg_before, pg_after, proc, items, gray_img, letter_ids, device,
                                 donor_seeds=donor_seeds, layers=sweep_layers)

    used = np.isfinite(sweep["floor"])
    floor_i = sweep["floor"][used]
    intact_i = sweep["intact"][used]
    correct = sweep["correct"][:, used]                 # (nL, nUsed)
    wrong_seeds = sweep["wrong"][:, :, used]            # (nSeeds, nL, nUsed)
    wrong_mean = wrong_seeds.mean(0)                    # (nL, nUsed) — the 5-seed null
    nL = correct.shape[0]

    lengths = np.array(sweep["lengths"]); drop = ~used
    bias = {"used_len_mean": float(lengths[used].mean()) if used.any() else None,
            "dropped_len_mean": float(lengths[drop].mean()) if drop.any() else None,
            "retention": float(used.mean())}

    discovery = None
    if args.headline_layer < 0:    # DISCOVERY — full ΔRF profile + onset + FDR (the nomination)
        points, ci_lo, ci_hi = LL.bootstrap_delta_rf_profile(correct, wrong_mean, floor_i, intact_i)
        seed_sd = np.array([np.std([LL.delta_rf_point(correct[L], wrong_seeds[s, L], floor_i, intact_i)
                                    for s in range(len(donor_seeds))]) for L in range(nL)])
        drf_profile = [float(x) for x in points]
        step_val, step_arg = LL.step_statistic(drf_profile)
        stable_onset = LL.stable_onset_depth(drf_profile)
        l_star = int(np.argmax(np.array(drf_profile) >= LL.DELTA)) if (np.array(drf_profile) >= LL.DELTA).any() else None
        pvals = [LL.one_sided_p_delta_rf(correct[L], wrong_mean[L], floor_i, intact_i, n_boot=2000) for L in range(nL)]
        discovery = {"delta_rf_profile": drf_profile,
                     "delta_rf_ci_lo": [float(x) for x in ci_lo], "delta_rf_ci_hi": [float(x) for x in ci_hi],
                     "delta_rf_seed_sd": [float(x) for x in seed_sd],
                     "stable_onset_depth": stable_onset, "step_statistic": float(step_val),
                     "step_argmax": step_arg, "L_star_earliest_cross_delta": l_star,
                     "bh_fdr_pass": LL.bh_fdr(pvals).tolist()}
        L_h = stable_onset
    else:
        L_h = args.headline_layer

    # headline ΔRF(L_h) CI: nominated onset (discovery) or frozen onset (confirmation = the verdict).
    headline = None
    corrupt = None
    if L_h is not None and 0 <= L_h < nL and bool(np.isfinite(correct[L_h]).any()):
        h_pt, h_lo, h_hi = LL.bootstrap_delta_rf(correct[L_h], wrong_mean[L_h], floor_i, intact_i)
        headline = {"layer": int(L_h), "is_frozen": args.headline_layer >= 0,
                    "delta_rf": h_pt, "ci_lo": h_lo, "ci_hi": h_hi,
                    "exceeds_delta": bool(h_lo > LL.DELTA),
                    "novel_per_gate": bool(L_h < nL - 2)}   # final 1-2 layers = trivial-confirmed (§6)
        if args.headline_layer >= 0:   # CONFIRMATION — symmetric corrupt patch at the frozen layer
            cp = corrupt_patch_at(pg_before, pg_after, proc, items, gray_img, letter_ids, device,
                                  L_h, donor_seeds)
            cu = np.isfinite(cp["intact_clean"])
            clean_a, cc_a, cw_a = cp["intact_clean"][cu], cp["corrupt_correct"][cu], cp["corrupt_wrong"][cu]
            gap_i = cw_a - cc_a   # item-specific: correct donor degrades intact MORE → cc < cw → gap>0
            rng = np.random.default_rng(LL.BOOTSTRAP_SEED)
            boot = np.array([gap_i[rng.integers(0, len(gap_i), len(gap_i))].mean() for _ in range(2000)])
            g_lo, g_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            corrupt = {"layer": int(L_h), "intact_clean": float(clean_a.mean()),
                       "corrupt_correct_acc": float(cc_a.mean()), "corrupt_wrong_acc": float(cw_a.mean()),
                       "degrade_correct": float((clean_a - cc_a).mean()),
                       "degrade_wrong": float((clean_a - cw_a).mean()),
                       "item_specific_gap": float(gap_i.mean()), "gap_ci_lo": g_lo, "gap_ci_hi": g_hi,
                       "item_specific": bool(g_lo > 0.0)}

    row = {
        "study": "latloc", "arm": "1_MDE", "eval_split": args.eval_split,
        "firewall_n_layers": n_dec, "attn_impl": "eager", "donor_seeds": donor_seeds,
        "preflight": {"mcq_intact": int_acc, "mcq_deleted": del_acc, "gap": gap},
        "n_items_total": len(items), "n_items_used": int(used.sum()),
        "n_buckets": sweep["n_buckets"], "length_bias": bias, "swept_layers": sweep["layers"],
        "delta": LL.DELTA, "k_step": LL.K_STEP,
        "discovery": discovery, "headline": headline, "corrupt_patch": corrupt,
        "raw": {"floor": sweep["floor"].tolist(), "intact": sweep["intact"].tolist(),
                "item_hash": [it["hash"] for it in items], "lengths": sweep["lengths"]},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / "latloc_arm1.json").write_text(json.dumps(row, indent=1, default=str))
    hl = (f"ΔRF(L{headline['layer']})={headline['delta_rf']:.3f} "
          f"CI[{headline['ci_lo']:.3f},{headline['ci_hi']:.3f}] exceeds_δ={headline['exceeds_delta']}"
          if headline else "no headline layer")
    phase = (f"DISCOVERY onset=L{discovery['stable_onset_depth']} step={discovery['step_statistic']:.2f}(descr)"
             if discovery else f"CONFIRMATION frozen-L{args.headline_layer}")
    cp_s = (f" | corrupt item_specific={corrupt['item_specific']} gap={corrupt['item_specific_gap']:.3f}"
            if corrupt else "")
    print(f"[latloc] Arm1 banked; retention={bias['retention']:.2%} {phase} | {hl}{cp_s}")


if __name__ == "__main__":
    main()
