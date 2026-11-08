"""W11 Leg-B — PG2 connector SVD-ablation instrument (pure, importable; no Modal/AWS deps).

Behavioral probe of the question W10's Λ geometry left open: does the learned connector
W_proj INSTALL vision→language readability (info concentrated in a few bridge directions) or
merely re-address pre-existing SigLIP structure (diffuse re-basis)?

Mechanism (mean-ablation, never zero — parent §5.1 / W5 lesson):
  * SVD W_proj = U S Vᵀ; the OUTPUT-space directions are the columns of U (in LM-hidden 2304-d).
  * Harvest h̄ = mean projector output over discovery image tokens.
  * Mean-ablate a direction set spanned by orthonormal Q (out_dim×k):
        out' = out − Q Qᵀ (out − h̄)        (replace the span(Q) component with h̄'s)
    structured  → Q = U[:, :k]   (top-k singular directions)
    random      → Q = QR(randn)  (k random orthonormal directions; the FLOOR)
    all-ablate  → Q = U (full r) (projector → constant mean image embed → DEAD; positive control)
  * N4 shuffle → roll the projector OUTPUT across the batch (image features → wrong sequence).

Capability (NOT divergence — GraftLM KL≠capability lesson): per-task acc / acc@0.5 / ANLS /
CIDEr-D. RefCOCO/TextVQA/VQAv2 are per-item (bootstrappable); COCO is corpus CIDEr (resampled).

No try/except hiding; shape/lib errors RAISE. Heavy imports inside functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

VISION_TOKENS_PER_IMAGE = 256  # PaliGemma2-224: 16×16 patches

# frozen concentration-curve grid (k = #directions ablated). k=1152 = full rank = all-ablate
# (projector → constant mean image embed → DEAD), the positive control endpoint of the curve.
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 512, 1152)


@dataclass
class BTask:
    name: str
    dataset: str
    eval_split: str
    prompt_fn: Callable          # (row) -> str
    score_fn: Callable           # (pred_str, row) -> float   (per-item tasks)
    image_fn: Callable           # (row) -> PIL.Image
    max_new_tokens: int
    is_corpus: bool = False
    keep_special: bool = False   # RefCOCO keeps <locDDDD> tokens
    refs_fn: Callable | None = None   # (row) -> list[str]   (corpus tasks: references)


def build_tasks() -> list:
    """The 4 §2B tasks, reusing the battery scorers/prompts + the transfer_a1 VQAv2 recipe."""
    from infra.pg import battery_tasks as BT
    from infra.pg import transfer_a1 as TA

    coco = BT.COCO_TASK
    refcoco = BT.REFCOCO_TASK
    textvqa = BT.TEXTVQA_TASK
    return [
        BTask("refcoco", refcoco.dataset, refcoco.eval_split, refcoco.prompt_fn,
              refcoco.score_fn, refcoco.image_fn, refcoco.max_new_tokens, keep_special=True),
        BTask("textvqa", textvqa.dataset, textvqa.eval_split, textvqa.prompt_fn,
              textvqa.score_fn, textvqa.image_fn, textvqa.max_new_tokens),
        BTask("vqav2", TA.VQA_DATASET, TA.VQA_EVAL_SPLIT,
              lambda r: f"{TA.VQA_PREFIX}{r['question']}",
              lambda pred, r: TA.vqa_soft_accuracy(pred, [a["answer"] for a in r["answers"]]),
              lambda r: r["image"], 8),
        BTask("coco_captions", coco.dataset, coco.eval_split, coco.prompt_fn,
              lambda pred, r: 0.0, coco.image_fn, coco.max_new_tokens,
              is_corpus=True, refs_fn=BT._coco_caps),
    ]


# ===========================================================================
# Projector SVD + mean-output harvest + direction bases.
# ===========================================================================
def get_projector(model, paths):
    """Return (proj_module, W_proj float32 cpu [out,in], bias-or-None). The PaliGemma2
    multi_modal_projector is a single Linear (vision_dim → lm_hidden); resolve robustly."""
    import torch.nn as nn
    proj_module = model.get_submodule(paths["multi_modal_projector"])
    linear = proj_module if isinstance(proj_module, nn.Linear) else None
    if linear is None:                       # find the (single) Linear child
        linears = [m for m in proj_module.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 1, f"expected 1 Linear in projector, found {len(linears)}: {linears}"
        linear = linears[0]
    W = linear.weight.detach().cpu().float()              # (out, in)
    b = linear.bias.detach().cpu().float() if linear.bias is not None else None
    return proj_module, W, b


def svd_U(W):
    """Left singular vectors (output-space directions), columns ordered by singular value desc."""
    import torch
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)   # U:(out,r) S:(r) Vt:(r,in)
    return U.contiguous(), S.contiguous()


def random_basis(out_dim, k, seed, device, dtype):
    """k random orthonormal directions in the out_dim output space (QR of a seeded Gaussian)."""
    import torch
    assert k <= out_dim, f"random_basis k={k} exceeds out_dim={out_dim}"
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(out_dim, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k].to(device).to(dtype)


def harvest_hbar(model, proc, images, proj_module, device, batch_size=16):
    """Mean projector OUTPUT over all image tokens across `images` (the mean-ablation target h̄)."""
    import torch
    acc = {"sum": None, "n": 0}

    def hook(_m, _i, out):
        flat = out.reshape(-1, out.shape[-1]).float()
        s = flat.sum(0).cpu()
        acc["sum"] = s if acc["sum"] is None else acc["sum"] + s
        acc["n"] += flat.shape[0]

    h = proj_module.register_forward_hook(hook)
    blank = "caption en\n"
    for start in range(0, len(images), batch_size):
        ims = [im.convert("RGB") for im in images[start:start + batch_size]]
        inp = proc(text=[blank] * len(ims), images=ims, return_tensors="pt",
                   padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            model(**inp)
    h.remove()
    assert acc["n"] > 0, "harvest_hbar saw no image tokens"
    return (acc["sum"] / acc["n"]).to(device)            # (out_dim,) float32


# ===========================================================================
# Ablation / shuffle forward hooks (fire once per generate, on the prefill).
# ===========================================================================
def make_ablation_hook(Q, hbar):
    """out' = out − Q Qᵀ (out − h̄). Mean-ablates the span(Q) component to h̄'s component."""
    def hook(_m, _i, out):
        dt = out.dtype
        x = out.float()
        Qf = Q.float()
        coeff = (x - hbar.float()) @ Qf                  # (..., k)
        return (x - coeff @ Qf.T).to(dt)
    return hook


def make_n4_shuffle_hook():
    """N4: roll the projector output across the batch (image i → sequence i+1) → image↔text
    pairing broken → structured-ablation signal must collapse to the random floor."""
    def hook(_m, _i, out):
        import torch
        if out.shape[0] < 2:
            return out
        return torch.roll(out, shifts=1, dims=0)
    return hook


# ===========================================================================
# Data: load eval items, content-hash dedup, deterministic disjoint split.
# ===========================================================================
def _content_hash(task, row):
    import hashlib
    im = task.image_fn(row).convert("RGB").resize((32, 32))
    return hashlib.md5(task.prompt_fn(row).encode() + im.tobytes()).hexdigest()


def load_eval_items(task, n_total, seed=0):
    """STREAMING load of the task eval split (lazy → reads only the first ~shard, disk-safe: the
    full lmms-lab/VQAv2 validation is 68 parquet shards / tens of GB and overflows a 200 GB root).
    Natural shard order → deterministic across the controls and unblind jobs (no buffer-shuffle
    nondeterminism). Dedup by content hash, take up to n_total."""
    from datasets import load_dataset
    ds = load_dataset(task.dataset, split=task.eval_split, streaming=True)
    rows, seen = [], set()
    for r in ds:
        h = _content_hash(task, r)
        if h in seen:
            continue
        seen.add(h)
        rows.append(r)
        if len(rows) >= n_total:
            break
    assert len(rows) == n_total, f"{task.name}: only {len(rows)} unique items (<{n_total})"
    return rows


def split_items(n, seed=1337):
    """Deterministic disjoint discovery/confirmation halves via enforce_split (held=confirmation)."""
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    rng = np.random.default_rng(seed)
    held = np.zeros(n, dtype=bool)
    held[rng.permutation(n)[: n // 2]] = True            # ~half sealed as confirmation
    return enforce_split(held, name="legB")


# ===========================================================================
# Eval-only runner (lifted from battery_tasks 857-930; NO training).
# ===========================================================================
def eval_task(model, proc, task, rows, device, hook_fn=None, proj_module=None,
              batch_size=16, cider_boot=300):
    """Greedy-decode + score `rows` under an optional projector hook. Per-item tasks return
    per_item scores (+ bootstrap CI); corpus (COCO) returns CIDEr-D point + resampled CI."""
    import numpy as np
    import torch

    handle = proj_module.register_forward_hook(hook_fn) if (hook_fn and proj_module) else None
    per_item = []
    cand, ref = {}, {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = [task.image_fn(r).convert("RGB") for r in batch]
        prompts = [task.prompt_fn(r) for r in batch]
        inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        in_len = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=task.max_new_tokens, do_sample=False)
        dec = proc.batch_decode(gen[:, in_len:], skip_special_tokens=not task.keep_special)
        for r, pred in zip(batch, dec):
            pred = pred.strip()
            if task.is_corpus:
                i = len(cand)
                cand[i] = [pred]
                ref[i] = task.refs_fn(r)
            else:
                per_item.append(float(task.score_fn(pred, r)))
    if handle:
        handle.remove()

    if task.is_corpus:
        score, lo, hi = _cider_boot(cand, ref, n_boot=cider_boot)
        return {"task": task.name, "score": score, "ci95": [lo, hi], "n": len(cand),
                "per_item": None, "corpus": True}
    from infra.pg.diff_metrics import bootstrap_ci
    arr = np.asarray(per_item, dtype=float)
    ci = bootstrap_ci(arr, stat="mean")
    return {"task": task.name, "score": float(arr.mean()), "ci95": [ci["lo"], ci["hi"]],
            "n": len(per_item), "per_item": per_item, "corpus": False}


def _cider_boot(cand, ref, n_boot=300, seed=1337):
    """CIDEr-D (pure-python pycocoevalcap.Cider) with whitespace tokenization — a RELATIVE drop
    metric (ablated vs baseline), not the PTB-tokenized published absolute. Bootstrap over images."""
    import numpy as np
    from pycocoevalcap.cider.cider import Cider

    def tok(d):
        return {k: [" ".join(s.lower().strip().split()) for s in v] for k, v in d.items()}

    gts, res = tok(ref), tok(cand)
    keys = list(cand.keys())
    point, _ = Cider().compute_score(gts, res)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(keys, size=len(keys), replace=True)
        g = {i: gts[k] for i, k in enumerate(samp)}
        r = {i: res[k] for i, k in enumerate(samp)}
        s, _ = Cider().compute_score(g, r)
        boots.append(s)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
