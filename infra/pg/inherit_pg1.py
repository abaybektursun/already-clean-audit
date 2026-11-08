"""INHERIT PG1-side glue — the ONE place the study touches the host model
(google/paligemma-3b-pt-224), so Gate-0, controls, discovery, confirmation and the S_vl job share
a single verified code path. Heavy imports (torch/transformers/datasets) live inside functions so
the module py_compiles on the control plane; the pure logic it feeds is in inherit_core /
ctrliface_core (weight-free-tested).

Pinned facts (invcore-FUNC Gate-0 lineage, re-asserted at INHERIT Gate-0):
  * model     PaliGemmaForConditionalGeneration pt-224, bf16, **eager attention MANDATORY**
              (sdpa mishandles the prefix-LM mask → wrong ppl; reference_paligemma_sdpa_wrong_forward)
  * decoder   18 Gemma-1 layers, d=2048 — located via invariant_core_func._decoder_layers
  * hook      forward_pre_hook on layers[L] = the residual ENTERING layer L (the bank's contract);
              kwarg-robust (inherit_core adapter); ALL positions (prefill + decode; parity across arms)
  * eval      VQAv2 (per-item soft-acc) + COCO captions (corpus CIDEr) via projector_ablation's
              streaming/dedup loaders; floor = patch-scrambled images (measured)
  * ppl gate  invariant_core_func._ppl_blankimg reproduced against the invcore_func.json artifact

No try/except hiding — a wrong assumption RAISES (CLAUDE.md).
"""
from __future__ import annotations

import contextlib

from infra.pg.inherit_core import (
    finalize_capture,
    make_pg1_pre_hook_ablate,
    make_pg1_pre_hook_capture,
    make_pg1_pre_hook_replace,
)

PG1_REPO = "google/paligemma-3b-pt-224"
N_LAYERS = 18
D = 2048


def load_pg1(device="cuda", sha=None):
    """PG1 on GPU, bf16, EAGER (mandatory). sha=None resolves the current revision; Gate-0 records
    the resolved sha in the artifact so every later job pins it."""
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, revision=sha, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(PG1_REPO, revision=sha)
    return model, proc


def decoder_layers(model):
    """The 18 Gemma-1 decoder layers (invcore-FUNC addressing, re-asserted)."""
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers
    layers = _decoder_layers(model)
    assert len(layers) == N_LAYERS
    return layers


# ---------------------------------------------------------------------------
# Interventions on the residual ENTERING layers[L] (the bank's contract).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def ablate_at(layers, L, Q, hbar):
    """Mean-ablate span(Q)→h̄ on the residual entering layers[L] for the duration."""
    h = layers[L].register_forward_pre_hook(make_pg1_pre_hook_ablate(Q, hbar), with_kwargs=True)
    yield
    h.remove()


@contextlib.contextmanager
def replace_at(layers, L, hbar):
    """Full-read positive control: the residual entering layers[L] ← h̄ for the duration."""
    h = layers[L].register_forward_pre_hook(make_pg1_pre_hook_replace(hbar), with_kwargs=True)
    yield
    h.remove()


def harvest(layers, forward_fn, batches, L_list, d=D, x_cap=0):
    """One no-grad pass over `batches`: per requested layer the PG1-NATIVE h̄_L, tr(C_L) and up to
    x_cap raw residual rows (for the energy-matched null's E_target). Returns
    {L: (hbar, trace_C, X, n)}."""
    import torch
    accs = {L: {} for L in L_list}
    handles = [layers[L].register_forward_pre_hook(
        make_pg1_pre_hook_capture(accs[L], d, x_cap=x_cap), with_kwargs=True) for L in L_list]
    with torch.no_grad():
        for b in batches:
            forward_fn(b)
    for h in handles:
        h.remove()
    return {L: finalize_capture(accs[L]) for L in L_list}


def accumulate_gram_pre(model, layers, loss_fn, batches, L_list, d=D):
    """S_vl estimator (Arm G-reverse): activation-space Gram G_L = Σ ∇_{h_L}ℓ ∇ᵀ of the VL loss
    w.r.t. the residual ENTERING layers[L], all requested layers in ONE backward per batch.
    Mirrors ctrliface_run.accumulate_gram_multi but at the layer INPUT (the bank's contract).
    Returns ({L: G d×d fp64 cpu}, n_positions)."""
    import torch
    from infra.pg.ctrliface_core import gram_update
    G = {L: torch.zeros(d, d, dtype=torch.float64) for L in L_list}
    n_pos = 0

    def _mk(store):
        def hook(_m, args, kwargs):
            if len(args) >= 1:
                hs, rest = args[0], lambda h: (h,) + tuple(args[1:])
            else:
                hs = kwargs["hidden_states"]
                rest = None
            if not hs.requires_grad:
                hs.requires_grad_(True)
            hs.retain_grad()
            store["h"] = hs
            return (rest(hs), kwargs) if rest is not None else (args, {**kwargs, "hidden_states": hs})
        return hook

    for batch in batches:
        stores = {L: {} for L in L_list}
        handles = [layers[L].register_forward_pre_hook(_mk(stores[L]), with_kwargs=True)
                   for L in L_list]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_fn(batch)
            assert loss.requires_grad, "loss does not require grad — forward ran under no_grad?"
            loss.backward()
        for h in handles:
            h.remove()
        bp = None
        for L in L_list:
            g = stores[L]["h"].grad
            assert g is not None, f"∂loss/∂h is None at layer {L}"
            g = g.reshape(-1, d).double()
            assert torch.isfinite(g).all(), f"non-finite grad at layer {L}"
            gram_update(G[L], g)
            bp = g.shape[0] if bp is None else bp
        n_pos += bp
    assert n_pos > 0, "no positions accumulated"
    return G, n_pos


# ---------------------------------------------------------------------------
# Eval tasks (VQAv2 per-item + COCO corpus), floor images, item hashes.
# ---------------------------------------------------------------------------
def pg1_tasks():
    """The two §4 PG1 eval tasks, reused from the W11 task builders (datasets/prompts/scorers are
    PaliGemma-generic; only the processor/model differ — the generation firewall shares CODE, never
    numbers)."""
    from infra.pg.projector_ablation import build_tasks
    tasks = {t.name: t for t in build_tasks()}
    return tasks["vqav2"], tasks["coco_captions"]


def load_items(task, n_total):
    """Streaming + content-hash-dedup rows (projector_ablation.load_eval_items) plus the per-item
    content hashes (the alignment key + the sha1 split input)."""
    from infra.pg.projector_ablation import _content_hash, load_eval_items
    rows = load_eval_items(task, n_total)
    hashes = [_content_hash(task, r) for r in rows]
    return rows, hashes


def scramble_image(im, seed, grid=4):
    """The measured-floor construction: shuffle the image as a grid×grid patch permutation
    (content destroyed, low-level statistics preserved). Deterministic in (item, seed)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB"))
    H, W = a.shape[0] // grid * grid, a.shape[1] // grid * grid
    a = a[:H, :W]
    tiles = [a[i:i + H // grid, j:j + W // grid]
             for i in range(0, H, H // grid) for j in range(0, W, W // grid)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tiles))
    assert (perm != np.arange(len(tiles))).any()
    rows = [np.concatenate([tiles[perm[r * grid + c]] for c in range(grid)], axis=1)
            for r in range(grid)]
    return Image.fromarray(np.concatenate(rows, axis=0))


def eval_pg1(model, proc, task, rows, device, ctx=None, batch_size=16, scramble_seed=None,
             cider_boot=300):
    """Greedy-decode + score `rows` under an optional intervention context manager `ctx`
    (ablate_at/replace_at/nullcontext). Per-item tasks → {score, ci95, per_item}; corpus (COCO) →
    CIDEr-D + resampled CI. scramble_seed≠None → the floor arm (patch-scrambled images).
    Same decode/scoring spine as projector_ablation.eval_task; the intervention is a layer hook
    context instead of a projector hook (the NEW-glue re-point)."""
    import numpy as np
    import torch
    ctx = ctx if ctx is not None else contextlib.nullcontext()
    per_item, cand, ref, preds = [], {}, {}, []
    with ctx:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = [task.image_fn(r).convert("RGB") for r in batch]
            if scramble_seed is not None:
                images = [scramble_image(im, scramble_seed + start + j)
                          for j, im in enumerate(images)]
            prompts = [task.prompt_fn(r) for r in batch]
            inp = proc(text=prompts, images=images, return_tensors="pt",
                       padding="longest").to(device)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            in_len = inp["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=task.max_new_tokens, do_sample=False)
            dec = proc.batch_decode(gen[:, in_len:], skip_special_tokens=not task.keep_special)
            for r, pred in zip(batch, dec):
                pred = pred.strip()
                preds.append(pred)
                if task.is_corpus:
                    i = len(cand)
                    cand[i] = [pred]
                    ref[i] = task.refs_fn(r)
                else:
                    per_item.append(float(task.score_fn(pred, r)))
    if task.is_corpus:
        from infra.pg.projector_ablation import _cider_boot
        score, lo, hi = _cider_boot(cand, ref, n_boot=cider_boot)
        # `cand`/`ref` are returned as well (LACUNA amendment, ADDITIVE — no existing caller reads
        # them): a corpus metric has no per-item score, so an item-PAIRED bootstrap across rows can
        # only be built from the banked candidate/reference strings.
        return {"task": task.name, "score": score, "ci95": [lo, hi], "n": len(cand),
                "per_item": None, "corpus": True, "cand": cand, "ref": ref}
    from infra.pg.diff_metrics import bootstrap_ci
    arr = np.asarray(per_item, dtype=float)
    ci = bootstrap_ci(arr, stat="mean")
    # `preds` is returned as well (LACUNA amendment, ADDITIVE — no existing caller reads it): the
    # S1 answer-shuffle control rescores the SAME generations against permuted ground truth, which
    # needs the raw decodes and costs no extra forward.
    return {"task": task.name, "score": float(arr.mean()), "ci95": [ci["lo"], ci["hi"]],
            "n": len(per_item), "per_item": per_item, "corpus": False, "preds": preds}


def ppl_gate(model, proc, device, n_seq=120, window=512):
    """The invcore-FUNC blank-image wikitext ppl instrument, verbatim (eager forward; exact
    token-block text location). The caller compares against the invcore_func.json artifact value."""
    from infra.aws_pg.jobs.invariant_core_func import _ppl_blankimg
    return _ppl_blankimg(model, proc, device, n_seq=n_seq, window=window)
