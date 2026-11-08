"""Invariant-core FUNCTIONAL test — is the shared-untouched q/k residual band LIVE or INERT?

Follow-up to invariant_core.py / invariant_core_triv.py (same branch family). Those CONFIRMED, for
the 24 same-init PaliGemma-1 fine-tunes of google/paligemma-3b-pt-224, a per-layer, q/k-shared,
mildly-low-sigma residual INPUT subspace that ALL fine-tunes leave essentially untouched (bottom-k
eigvecs of the consensus Gram G = sum_i dW_i^T dW_i / ||dW_i||^2_F, far below a matched-random
rotation null; sigma-ratio 0.65..0.91, NOT dead-weight). THE OPEN QUESTION: is that band FUNCTIONALLY
LIVE (carries real activation AND removing it costs capability MORE than matched-random / bottom-sigma)
or INERT (gradient-starved in data space -> no gradient -> untouched by construction -> behaviorally
silent diagnostic null)?

Reuses invariant_core.py's load+difference path VERBATIM (imported) to rebuild U_unt per layer.
Discovery-grade, exploratory (no pre-reg). FAIL LOUD (no try/except, no synthetic fallback).

THE BAND (U_unt). Per layer L, k in {8,32,128}: U_unt(L,k) = the k smallest eigvecs of the q_proj
consensus Gram G_q(L) (the PRIMARY confirmed object: q_proj 18/18 layers BEYOND_NULL; q/k-shared at
114x random so the q-defined band IS the shared band). 2048-dim residual-input space.

CAPABILITY READOUT — dynamic-range note (VERIFIED from artifacts, not assumed). The base
google/paligemma-3b-pt-224 is a captioner/VQA checkpoint; its TEXT-ONLY / blank-image MMLU is at
CHANCE (native blank-image MMLU 25.49 in studies/.../xgen/controls.json -> NO MMLU dynamic range on
the native base). So MMLU accuracy alone is a degenerate instrument here and CANNOT carry the
contrast. The graded readout with real dynamic range on the native base is BLANK-IMAGE wikitext
PERPLEXITY (a continuous LM-capability score, well-defined at any accuracy level). We report BOTH:
blank-image MMLU (shown to sit at chance, as a sanity floor) AND blank-image perplexity (the decisive
graded contrast). The forward is the model's IN-DISTRIBUTION path (a uniform-gray 224 image present),
matching how the repo evaluates native pt-224 text competence (text_eval.mmlu_blankimg).

PIPELINE (one self-terminating job; the cheap gate runs first and CAN end the verdict):
  PRE  preflight: load full pt-224; reproduce native blank-image MMLU ~= 25.49 cached chain number
       (prove the capability harness is wired); assert U_unt orthonormal + projector idempotent.
  ACT  activation read (cheap, decisive-if-INERT): forward pt-224 (blank-image) over a few hundred
       MMLU prompts, capture the q_proj/k_proj INPUT (post-input_layernorm residual) per layer,
       accumulate its Gram, compute the fraction of activation ENERGY inside U_unt(L,k) vs a
       matched-random k-subspace. IF U_unt holds <= random energy => gradient-starved => INERT and
       we STOP with that verdict (no ablation).
  ABL  ablation contrast (only if LIVE-candidate): project the q/k INPUT onto the COMPLEMENT of the
       ablated subspace at EVERY layer (x' = x - Q Q^T x, removing the band from what attention
       reads), measure capability under (a) U_unt, (b) matched-random k, (c) base-bottom-sigma k,
       (d) none. Sweep k. THE VERDICT IS THE CONTRAST: U_unt-ablation hurts ~= random/bottom-sigma
       => INERT; >> random AND bottom-sigma => LIVE / load-bearing.

  # smoke (one ft-cloud member would not define a consensus; smoke uses few prompts + 1 layer-set):
  pg run infra.aws_pg.jobs.invariant_core_func --args="--smoke" --no-weights \
      --type g6e.8xlarge,p5.4xlarge --s3-prefix results/invcore_func_smoke --budget-min 90
  # full:
  pg run infra.aws_pg.jobs.invariant_core_func --args="--k-grid 8,32,128 --n-act 400 --n-mmlu-per 2" \
      --no-weights --type g6e.8xlarge,p5.4xlarge --s3-prefix results/invcore_func --budget-min 300
"""
import argparse
import json
import pathlib

# Reuse the load + difference path VERBATIM (imported, not re-implemented).
from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO,
    N_LAYERS,
    _load_decoder,
    _decoder_matrices,
    _ftype,
    _delete_cache,
)

# energy-fraction ACT gate: U_unt is "gradient-live" only if it holds MEANINGFULLY more activation
# energy than a matched-random k-subspace. random k-subspace of R^n holds ~ k/n in expectation; a
# LIVE band must clear it by a real margin (we report ratio energy(U_unt)/energy(random), and a
# proceed flag at ratio >= ACT_LIVE_RATIO). This is the cheap INERT-or-not decision.
ACT_LIVE_RATIO = 2.0    # U_unt must hold >= 2x the matched-random energy to be worth the ablation.

# §13.2 — the random ablation null is a BAND over >=3 independent draws (a single draw is noise);
# the verdict requires seed-stability across the band, and the positive-control validity gate is
# defined against the band's UPPER edge.
N_RAND_DRAWS = 3


def _consensus_gram_qproj(repos, shas, w_base, want_layers):
    """Per-layer q_proj consensus Gram G_q(L) = sum_i dW_i^T dW_i / ||dW_i||^2_F over the 24 ft,
    fp64, n x n (n=2048). Loads each ft ONCE, accumulates, deletes its HF snapshot (24 x ~12GB fp32
    overflows root). Returns {L: G (2048x2048 np.float64)}. VERBATIM the triv accumulation."""
    import numpy as np
    import torch
    G = {L: np.zeros((2048, 2048), dtype=np.float64) for L in want_layers}
    n_terms = {L: 0 for L in want_layers}
    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[func][gram] diff {repo_id}@{sha[:8]} ({ridx+1}/{len(repos)}) ...", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base, w_ft)
        for L in want_layers:
            key = f"layers.{L}.self_attn.q_proj.weight"
            wb, wf = matched[key]
            assert _ftype(key) == "q_proj"
            A = (wf.to(torch.float64) - wb.to(torch.float64))            # dW, m x n
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"zero dW {key} in {repo_id}"
            G[L] += (A.transpose(0, 1) @ A).cpu().numpy().astype(np.float64) / fro2
            n_terms[L] += 1
        del w_ft, matched
        _delete_cache(repo_id)
    for L, nt in n_terms.items():
        assert nt == len(repos), f"L{L} got {nt} terms, expected {len(repos)}"
    return G


def _bottom_eigvecs(G, k):
    """fp64 symmetric eig (ascending). n x k matrix of the k SMALLEST eigvecs (U_unt), orthonormal."""
    import numpy as np
    w, V = np.linalg.eigh(G)
    kk = int(min(k, w.size))
    return V[:, :kk].astype(np.float64), w[:kk].astype(np.float64)


def _base_bottom_sigma(W0, k):
    """The k smallest-sigma RIGHT-singular vectors of the base weight W0 (m x n): the base's
    'barely-used' input directions. fp64. Returns n x k orthonormal."""
    import numpy as np
    import torch
    _, s, Vh = torch.linalg.svd(W0.detach().to(torch.float64), full_matrices=False)  # Vh: r x n
    V = Vh.transpose(0, 1).cpu().numpy().astype(np.float64)                            # n x r, desc sigma
    r = V.shape[1]
    assert k <= r, f"k={k} exceeds base rank r={r}"
    return V[:, r - k:]                                                                # smallest-sigma k


def _base_top_sigma(W0, k):
    """POSITIVE CONTROL (§5, §13.1). The k LARGEST-sigma RIGHT-singular vectors of the base weight
    W0 (m x n): the base's MOST-USED input directions. Ablating these must produce a large ppl-rise
    -> it proves the ablation readout has dynamic range at this k. fp64. Returns n x k orthonormal."""
    import numpy as np
    import torch
    _, s, Vh = torch.linalg.svd(W0.detach().to(torch.float64), full_matrices=False)  # Vh: r x n, desc sigma
    V = Vh.transpose(0, 1).cpu().numpy().astype(np.float64)                            # n x r, desc sigma
    r = V.shape[1]
    assert k <= r, f"k={k} exceeds base rank r={r}"
    return V[:, :k]                                                                    # largest-sigma k


def _random_subspace(n, k, seed):
    """k random orthonormal directions in R^n (QR of a seeded Gaussian). fp64 numpy."""
    import numpy as np
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q[:, :k].astype(np.float64)


# ===========================================================================
# Model + decoder-layer addressing.
# ===========================================================================
def _load_full_pg1(sha, device):
    """Full PaliGemmaForConditionalGeneration pt-224 on GPU (bf16, eager). The genuine base the
    discovery is about — we ablate its LM decoder q/k inputs in its in-distribution forward."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, revision=sha, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    return m


def _decoder_layers(model):
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.entities import resolve_submodule_paths
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    layers = resolve_gemma2_layers(lm)
    assert len(layers) == N_LAYERS, f"expected {N_LAYERS} decoder layers, got {len(layers)}"
    return layers


# ===========================================================================
# Forward hooks: q/k INPUT capture (ACT) and q/k INPUT projection (ABL).
# A forward_pre_hook on q_proj/k_proj sees the SHARED post-input_layernorm residual (the band's
# space). ACT records it; ABL replaces it with its projection onto the complement of the band.
# ===========================================================================
class _InputCapture:
    """Accumulate the Gram (X^T X, fp64) of the q_proj INPUT over a forward, per layer. Energy in a
    subspace U = trace(U^T (X^T X) U) / trace(X^T X) = fraction of total activation variance in U."""
    def __init__(self, layers):
        self.layers = layers
        self.gram = {L: None for L in range(len(layers))}
        self.n_tok = {L: 0 for L in range(len(layers))}
        self._handles = []

    def _mk(self, L):
        def pre_hook(_m, args):
            x = args[0]                                   # (b, T, n) q_proj input = post-LN residual
            flat = x.reshape(-1, x.shape[-1]).double()    # (bT, n) fp64
            g = (flat.transpose(0, 1) @ flat).cpu().numpy()
            self.gram[L] = g if self.gram[L] is None else self.gram[L] + g
            self.n_tok[L] += flat.shape[0]
            # record only, do not modify (return None == no-op)
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            self._handles.append(
                layer.self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class _InputProject:
    """forward_pre_hook on q_proj AND k_proj of every layer: replace the shared input x by its
    projection onto the COMPLEMENT of Q(L) -> x' = x - Q (Q^T x). Removes span(Q) from what attention
    reads. Q[L] is an (n x k) orthonormal basis (the ablated subspace) as a torch tensor on device."""
    def __init__(self, layers, Q_by_layer, device, dtype):
        import torch
        self.layers = layers
        self.Q = {L: torch.as_tensor(Q_by_layer[L], device=device, dtype=dtype)
                  for L in Q_by_layer}                    # (n x k)
        self._handles = []

    def _mk(self, L):
        Q = self.Q[L]
        def pre_hook(_m, args):
            x = args[0]                                   # (b, T, n)
            coeff = x @ Q                                 # (b, T, k)
            x2 = x - coeff @ Q.transpose(0, 1)            # remove span(Q)
            return (x2,) + tuple(args[1:])                # modified positional args
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            if L not in self.Q:
                continue
            self._handles.append(
                layer.self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
            self._handles.append(
                layer.self_attn.k_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ===========================================================================
# Capability: blank-image MMLU (sanity floor) + blank-image wikitext perplexity (graded contrast).
# In-distribution forward: a uniform-gray 224 image present (text_eval.mmlu_blankimg convention).
# ===========================================================================
def _blank_image(gray=128):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))


def _mmlu_blankimg(model, processor, device, n_per_subject, batch_size=4):
    from infra.pg import text_eval as TE
    return TE.mmlu_blankimg(model, processor, device, n_shots=5,
                            n_per_subject=n_per_subject, batch_size=batch_size)


def _processor_text_ids(tok, text):
    """The ids the PaliGemma processor will ACTUALLY produce for `text`, ready to anchor on.

    The processor wraps text as [image .. <bos> .. text .. "\n"], and that trailing newline is not
    always a token of its own: for some texts it MERGES into the final token, so `tok(text)` and
    the processor's text block differ in exactly one id while both lengths still look plausible.
    `_locate_text_block` then fails with a message that sounds like the block is missing when it is
    present and off by one token — which is how M-E2 (M-D45) and M-D2 (M-D51) both died, and why
    several other callers have run for months without noticing: whether the merge happens depends
    on the last character of the window.

    Tokenising `text + "\n"` is correct EITHER WAY: it yields the merged token when the tokenizer
    merges and a separate newline token when it does not, which is exactly what the processor
    produced in each case.

    Existing callers that already banked results are deliberately NOT migrated — the number of
    tokens they score would change by one, and that is a change to a measurement that is already
    on the record, not a bug fix.
    """
    return tok(text + "\n", return_tensors="pt", add_special_tokens=False).input_ids[0]


def _locate_text_block(in_ids, win_ids):
    """Index of the RIGHTMOST contiguous occurrence of win_ids inside in_ids (1-D LongTensors).
    PaliGemma's processor wraps text as [image tokens .. <bos> .. text .. trailing "\\n"] — it
    APPENDS a trailing newline, so the text block is NOT the exact tail. We anchor on the exact
    token block (same tokenizer on the same string => exact match), rightmost to skip any accidental
    match inside the prefix. FAIL LOUD if absent (no silent misalignment)."""
    import torch
    n = int(win_ids.shape[0])
    hay = in_ids.cpu()
    for i in range(hay.shape[0] - n, -1, -1):
        if torch.equal(hay[i:i + n], win_ids):
            return i
    raise AssertionError(f"text tokens (n={n}) not found in processor output (len {hay.shape[0]})")


def _ppl_blankimg(model, processor, device, n_seq, window=512, gray=128):
    """Blank-image wikitext-2 per-token CE perplexity (graded LM capability with dynamic range on
    the native base). Each non-overlapping window scored with a blank gray image present (the model's
    in-distribution multimodal forward); per-window [nll_sum, n_tok] banked for a token-weighted
    bootstrap. Reads logits only over the TEXT tokens (the image tokens are not targets)."""
    import numpy as np
    import torch
    from datasets import load_dataset

    import math
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    # split the corpus into ~window-token text windows on a clean char boundary (no decode round-trip:
    # we tokenize each window STRING directly via the processor, and self-align targets to the
    # processor's own text tokens — the trailing run after the 256 image tokens + BOS prefix).
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blank = _blank_image(gray)
    # PaliGemma wraps text as [image tokens .. <bos> .. text .. trailing "\n"]; the text is NOT the
    # exact tail (a newline is appended). We locate the text region as the exact token block inside
    # the processor output (same tokenizer on the same string), then score next-token over it.
    nll_sum = 0.0
    n_tok = 0
    per_window = []
    n_done = 0
    for start in range(0, ids.size(0), window):
        if n_seq is not None and n_done >= n_seq:
            break
        chunk_ids = ids[start:start + window]
        if chunk_ids.size(0) < 8:
            break
        window_text = tok.decode(chunk_ids.tolist())
        # direct (image-free) tokenization of the window string == the exact text-token sequence the
        # processor will place after the image+BOS prefix (same tokenizer, add_special_tokens=False).
        win_ids = tok(window_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        n_text = int(win_ids.shape[0])
        if n_text < 8:
            continue
        inp = processor(text=[window_text], images=[blank], return_tensors="pt",
                        padding=False).to(device)
        in_ids = inp["input_ids"][0]
        assert in_ids.shape[0] > n_text, "text span >= prompt — image prefix missing"
        pos = _locate_text_block(in_ids, win_ids)         # exact text block (handles trailing "\n")
        with torch.inference_mode():
            logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                           pixel_values=inp["pixel_values"].to(torch.bfloat16)).logits[0]
        # shifted LM loss over the text region: logits at position pos+j predict in_ids[pos+j+1].
        # targets win_ids[1:] sit at input positions pos+1 .. pos+n_text-1, predicted by logits at
        # positions pos .. pos+n_text-2.
        text_logits = logits[pos:pos + n_text - 1, :].float()          # (n_text-1, V)
        targets = win_ids[1:].to(device)                               # (n_text-1,)
        assert text_logits.shape[0] == targets.shape[0], (text_logits.shape, targets.shape)
        loss = torch.nn.functional.cross_entropy(text_logits, targets, reduction="sum")
        per_window.append([float(loss.item()), int(targets.numel())])
        nll_sum += float(loss.item())
        n_tok += int(targets.numel())
        n_done += 1
    assert n_tok > 0, "perplexity saw no tokens"
    return {"ppl": math.exp(nll_sum / n_tok), "mean_nll": nll_sum / n_tok,
            "n_tokens": n_tok, "n_windows": len(per_window), "window": window,
            "per_window": per_window}


def _avg_per_window(per_windows):
    """Average nll_sum across draws per window for the random BAND (§13.2). Every draw scored the
    SAME windows in the SAME order, so per-window token counts are identical (asserted, FAIL LOUD).
    Input: list of (W,2) arrays [nll_sum, n_tok]. Returns one (W,2) array [mean_nll_sum, n_tok]."""
    import numpy as np
    arrs = [np.asarray(p, dtype=np.float64) for p in per_windows]
    n_tok = arrs[0][:, 1]
    for a in arrs:
        assert np.array_equal(a[:, 1], n_tok), "random draws scored different windows (n_tok mismatch)"
    nll = np.mean([a[:, 0] for a in arrs], axis=0)
    return np.stack([nll, n_tok], axis=1)


def _paired_contrast_ci(pw_treat, pw_null, n_boot=1000, seed=1337):
    """PAIRED bootstrap of the perplexity contrast ppl(treat) - ppl(null) over a SHARED resample of
    windows (§7, §13.3). The SAME resampled window indices are applied to BOTH arms each draw; since
    both arms add the same baseline ppl(none), it cancels and this IS Δppl(treat) - Δppl(null).
    pw_* = (W,2) array [nll_sum, n_tok]; both arms scored the SAME windows in the SAME order, so the
    token-count columns must match exactly (asserted). Returns the contrast point estimate, its 95%
    CI, and the one-sided bootstrap p that the contrast <= 0."""
    import math
    import numpy as np
    t = np.asarray(pw_treat, dtype=np.float64)
    u = np.asarray(pw_null, dtype=np.float64)
    assert t.shape == u.shape, (t.shape, u.shape)
    assert np.array_equal(t[:, 1], u[:, 1]), "paired arms have mismatched per-window token counts"
    W = t.shape[0]

    def _ppl(a, idx):
        s = a[idx]
        return math.exp(s[:, 0].sum() / s[:, 1].sum())

    full = np.arange(W)
    point = _ppl(t, full) - _ppl(u, full)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, W, W)
        boots[b] = _ppl(t, idx) - _ppl(u, idx)
    return {"contrast": float(point), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "p_le0": float(np.mean(boots <= 0.0)), "n_windows": int(W)}


def _bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values across the k-family (§7), returned in input order."""
    import numpy as np
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # enforce monotonicity from the largest
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty => all -224 (the N5c cloud)")
    ap.add_argument("--layers", default="", help="comma layer idxs; empty => all 18")
    ap.add_argument("--k-grid", default="8,32,128")
    ap.add_argument("--n-act", type=int, default=400, help="#MMLU prompts for the activation read")
    ap.add_argument("--n-mmlu-per", type=int, default=2,
                    help="MMLU items per subject for the capability read (None-like full if <=0)")
    ap.add_argument("--n-ppl-seq", type=int, default=120, help="#wikitext windows for ppl")
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cached-mmlu", type=float, default=25.49,
                    help="native PG1 blank-image MMLU from xgen/controls.json (preflight anchor)")
    ap.add_argument("--cached-tol", type=float, default=3.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in a.k_grid.split(",")]
    if a.smoke:
        k_grid = [8, 32]
        a.n_act = 64
        a.n_ppl_seq = 8
        want_layers = [0, 9, 17]
        # smoke validates GPU-forward + hook + capability WIRING, not the science: a tiny ft set
        # gives a valid-shaped (scientifically meaningless) U_unt cheaply, so we don't pay 24×12GB
        # downloads just to check plumbing. The FULL run uses the whole -224 cloud.
        if not a.ft_tasks.strip():
            a.ft_tasks = "vqav2,gqa,ai2d"
    else:
        want_layers = ([int(x) for x in a.layers.split(",") if x.strip()]
                       if a.layers.strip() else list(range(N_LAYERS)))
    n_mmlu_per = None if a.n_mmlu_per <= 0 else a.n_mmlu_per

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required for the pt-224 forward"
    device = "cuda"
    api = HfApi()

    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    print(f"[func] base={PG1_REPO}@{base_sha[:8]} cloud={len(repos)} ft layers={want_layers} "
          f"k={k_grid} n_act={a.n_act} smoke={a.smoke} gpu={torch.cuda.get_device_name(0)}", flush=True)

    rep = {"object": "invariant_core_functional_v1", "base_repo": PG1_REPO, "base_sha": base_sha,
           "cloud_repos": repos, "cloud_shas": shas, "n_ft": len(repos),
           "layers": want_layers, "k_grid": k_grid, "seed": a.seed, "smoke": a.smoke,
           "band": "U_unt(L,k) = k smallest eigvecs of q_proj consensus Gram (q/k-shared band)",
           "capability_note": ("native pt-224 blank-image MMLU is at chance (~25.49, xgen/controls); "
                               "MMLU has NO dynamic range on the native base. Graded contrast = "
                               "blank-image wikitext perplexity. Both reported."),
           "act_live_ratio_threshold": ACT_LIVE_RATIO}

    # ---- (0) U_unt per layer: q_proj consensus Gram across the 24 ft, then bottom-k eigvecs.
    w_base = _load_decoder(PG1_REPO, base_sha)
    base_W0 = {L: w_base[f"layers.{L}.self_attn.q_proj.weight"] for L in want_layers}
    for L in want_layers:
        assert base_W0[L].shape[1] == 2048, f"q_proj input dim {base_W0[L].shape[1]} != 2048 (L{L})"
    G_q = _consensus_gram_qproj(repos, shas, w_base, want_layers)
    del w_base
    U_unt = {}      # U_unt[(L,k)] = (2048 x k) np.float64
    bandinfo = []
    n = 2048
    for L in want_layers:
        for k in k_grid:
            Uk, eigk = _bottom_eigvecs(G_q[L], k)
            assert abs((Uk.T @ Uk) - np.eye(k)).max() < 1e-8, f"U_unt not orthonormal L{L} k{k}"
            U_unt[(L, k)] = Uk
            bandinfo.append({"layer": L, "k": k, "eig_min": float(eigk.min()),
                             "eig_max": float(eigk.max())})
    rep["band_info"] = bandinfo
    (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
    print(f"[func] built U_unt for {len(want_layers)} layers x {len(k_grid)} k", flush=True)

    # ---- load the full base model + processor (the genuine pt-224 the discovery concerns).
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, device)
    layers = _decoder_layers(model)

    # ---- PRE: preflight. (i) projector idempotent on a random batch; (ii) native blank-image MMLU
    # reproduces the cached chain number => the capability harness is wired before any contrast.
    Lp, kp = want_layers[0], k_grid[0]
    Qt = torch.as_tensor(U_unt[(Lp, kp)], device=device, dtype=torch.float64)
    xr = torch.randn(3, 5, n, device=device, dtype=torch.float64)
    x1 = xr - (xr @ Qt) @ Qt.transpose(0, 1)
    x2 = x1 - (x1 @ Qt) @ Qt.transpose(0, 1)
    proj_resid = float((x2 - x1).abs().max())
    inside_removed = float((x1 @ Qt).abs().max())             # span(Q) component of x1 must be ~0
    assert proj_resid < 1e-9 and inside_removed < 1e-9, \
        f"projector not idempotent/clean: resid={proj_resid} inside={inside_removed}"
    print(f"[func][PRE] projector idempotent resid={proj_resid:.2e} inside_removed={inside_removed:.2e}",
          flush=True)

    pre_mmlu = _mmlu_blankimg(model, processor, device, n_per_subject=(2 if a.smoke else n_mmlu_per))
    d_cached = abs(pre_mmlu["acc_pct"] - a.cached_mmlu)
    preflight_ok = bool(a.smoke or d_cached <= a.cached_tol)
    print(f"[func][PRE] native blank-image MMLU={pre_mmlu['acc_pct']:.2f} (cached {a.cached_mmlu}) "
          f"|d|={d_cached:.2f} tol={a.cached_tol} -> ok={preflight_ok}", flush=True)
    rep["preflight"] = {"projector_resid": proj_resid, "projector_inside_removed": inside_removed,
                        "native_blankimg_mmlu_pct": pre_mmlu["acc_pct"], "cached_mmlu": a.cached_mmlu,
                        "delta": d_cached, "tol": a.cached_tol, "ok": preflight_ok,
                        "mmlu_n": pre_mmlu["n"]}
    (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
    assert preflight_ok, f"PREFLIGHT FAIL: native MMLU {pre_mmlu['acc_pct']} vs cached {a.cached_mmlu}"

    # ---- ACT: activation energy of U_unt vs matched-random, per layer per k. Forward pt-224
    # (blank-image) over n_act MMLU prompts capturing the q_proj input Gram.
    from datasets import load_dataset
    from infra.pg import text_eval as TE
    ds = load_dataset(TE.MMLU_REPO, "all")
    test_rows = list(ds["test"])
    rng = np.random.default_rng(a.seed)
    sel = rng.permutation(len(test_rows))[:a.n_act]
    blank = _blank_image()
    cap = _InputCapture(layers)
    with cap:
        for s0 in range(0, len(sel), 4):
            batch = [test_rows[int(i)] for i in sel[s0:s0 + 4]]
            prompts = [TE._q_block(r["question"], r["choices"]) for r in batch]
            inp = processor(text=prompts, images=[blank] * len(batch),
                            return_tensors="pt", padding="longest").to(device)
            with torch.inference_mode():
                model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(torch.bfloat16))
    act = []
    rand_seed_base = a.seed * 7919
    for L in want_layers:
        gram = cap.gram[L]
        assert gram is not None and cap.n_tok[L] > 0, f"no activation captured at L{L}"
        tot = float(np.trace(gram))
        rec = {"layer": L, "n_tok": int(cap.n_tok[L]), "total_energy": tot, "k": {}}
        for k in k_grid:
            Uk = U_unt[(L, k)]
            e_unt = float(np.trace(Uk.T @ gram @ Uk)) / tot
            # matched-random: average over a few random k-subspaces (cheap, fp64).
            rr = []
            for d in range(5):
                Qr = _random_subspace(n, k, rand_seed_base + 1000 * L + 100 * k + d)
                rr.append(float(np.trace(Qr.T @ gram @ Qr)) / tot)
            e_rand = float(np.mean(rr))
            rec["k"][str(k)] = {"energy_unt": e_unt, "energy_rand": e_rand,
                                "ratio_unt_over_rand": e_unt / e_rand if e_rand > 0 else float("inf"),
                                "energy_rand_kover_n": float(k / n)}
        act.append(rec)
        k0 = str(k_grid[0])
        print(f"[func][ACT] L{L}: k={k_grid[0]} E_unt={rec['k'][k0]['energy_unt']:.4f} "
              f"E_rand={rec['k'][k0]['energy_rand']:.4f} "
              f"ratio={rec['k'][k0]['ratio_unt_over_rand']:.2f}", flush=True)
    rep["activation_energy"] = act
    # aggregate the ACT gate: mean ratio per k across layers + proceed flag.
    act_agg = {}
    for k in k_grid:
        ks = str(k)
        ratios = [r["k"][ks]["ratio_unt_over_rand"] for r in act]
        e_unt = [r["k"][ks]["energy_unt"] for r in act]
        e_rand = [r["k"][ks]["energy_rand"] for r in act]
        act_agg[ks] = {"mean_ratio": float(np.mean(ratios)), "min_ratio": float(np.min(ratios)),
                       "max_ratio": float(np.max(ratios)), "mean_energy_unt": float(np.mean(e_unt)),
                       "mean_energy_rand": float(np.mean(e_rand))}
    rep["activation_energy_aggregate"] = act_agg
    proceed = any(act_agg[str(k)]["mean_ratio"] >= ACT_LIVE_RATIO for k in k_grid)
    rep["act_proceed_to_ablation"] = bool(proceed)
    (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78, flush=True)
    for k in k_grid:
        ks = str(k)
        print(f"[func][ACT-AGG] k={k}: mean ratio U_unt/random = {act_agg[ks]['mean_ratio']:.2f} "
              f"(E_unt {act_agg[ks]['mean_energy_unt']:.4f} vs E_rand {act_agg[ks]['mean_energy_rand']:.4f})",
              flush=True)

    if not proceed:
        rep["verdict"] = (f"INERT (diagnostic null): the untouched q/k band holds <= "
                          f"{ACT_LIVE_RATIO}x matched-random activation energy at every k "
                          f"(mean ratio "
                          f"{ {str(k): round(act_agg[str(k)]['mean_ratio'], 2) for k in k_grid} }) "
                          f"=> gradient-starved => untouched by construction => behaviorally silent. "
                          f"No capability ablation run (the contrast would be undefined).")
        (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
        print(f"\nINVARIANT_CORE_FUNC_OK verdict=INERT (ACT gate; ratios "
              f"{ {str(k): round(act_agg[str(k)]['mean_ratio'], 2) for k in k_grid} })", flush=True)
        return

    # ---- ABL: arms = none(baseline) + U_unt + random-BAND(N_RAND_DRAWS draws) + base-bottom-sigma
    #       + top-sigma(POSITIVE CONTROL). The decisive readout is blank-image wikitext perplexity;
    #       the verdict is a PAIRED bootstrap CI on the contrast Δppl(U_unt) - Δppl(null) over shared
    #       resampled windows. MMLU is baseline-only (sanity floor; at chance, cannot carry the
    #       contrast). Every arm scores the SAME windows in the SAME order -> per-window pairing.
    print(f"[func] ACT gate PROCEED -> ablation contrast (random band = {N_RAND_DRAWS} draws)", flush=True)

    dtype = next(model.parameters()).dtype

    def _arm_ppl(Qby):
        """Blank-image wikitext ppl while projecting the q/k input onto the complement of Qby."""
        with _InputProject(layers, Qby, device, dtype):
            return _ppl_blankimg(model, processor, device, n_seq=a.n_ppl_seq, window=a.ppl_window)

    base_mmlu = _mmlu_blankimg(model, processor, device, n_per_subject=(2 if a.smoke else n_mmlu_per))
    base_pp = _ppl_blankimg(model, processor, device, n_seq=a.n_ppl_seq, window=a.ppl_window)
    base_ppl = base_pp["ppl"]
    rep["capability_baseline"] = {"mmlu_pct": base_mmlu["acc_pct"], "mmlu_n": base_mmlu["n"],
                                  "ppl": base_ppl, "ppl_n_tokens": base_pp["n_tokens"]}
    print(f"[func][ABL] BASELINE mmlu={base_mmlu['acc_pct']:.2f} ppl={base_ppl:.3f}", flush=True)

    abl = []
    for k in k_grid:
        Q_unt = {L: U_unt[(L, k)] for L in want_layers}
        Q_bots, Q_tops = {}, {}
        for L in want_layers:
            W0 = layers[L].self_attn.q_proj.weight.detach().float().cpu()
            Q_bots[L] = _base_bottom_sigma(W0, k)        # smallest-sigma null
            Q_tops[L] = _base_top_sigma(W0, k)           # largest-sigma POSITIVE CONTROL

        pp_unt = _arm_ppl(Q_unt)
        pp_bots = _arm_ppl(Q_bots)
        pp_tops = _arm_ppl(Q_tops)
        # random null as a BAND over N_RAND_DRAWS independent draws (same windows each draw).
        rand_pws, rand_dppls = [], []
        for d in range(N_RAND_DRAWS):
            Q_rand = {L: _random_subspace(n, k, a.seed * 104729 + 1000 * d + L) for L in want_layers}
            pp_r = _arm_ppl(Q_rand)
            rand_pws.append(pp_r["per_window"])
            rand_dppls.append(pp_r["ppl"] - base_ppl)
        rand_pw_avg = _avg_per_window(rand_pws)
        rand_band = {"dppls": rand_dppls, "dppl_mean": float(np.mean(rand_dppls)),
                     "dppl_min": float(np.min(rand_dppls)), "dppl_max": float(np.max(rand_dppls)),
                     "dppl_std": float(np.std(rand_dppls)), "n_draws": N_RAND_DRAWS}

        d_unt = pp_unt["ppl"] - base_ppl
        d_bots = pp_bots["ppl"] - base_ppl
        d_tops = pp_tops["ppl"] - base_ppl

        # PAIRED contrast CIs on shared resampled windows: U_unt vs the random-band mean and vs bottom-sigma.
        ci_rand = _paired_contrast_ci(pp_unt["per_window"], rand_pw_avg, seed=a.seed)
        ci_bots = _paired_contrast_ci(pp_unt["per_window"], pp_bots["per_window"], seed=a.seed)

        # POSITIVE-CONTROL validity gate: top-sigma Δppl must clear the random band's UPPER edge,
        # else the ablation readout has no dynamic range at this k and the contrast is VOID.
        valid = bool(d_tops > rand_band["dppl_max"])

        contrast = {"dppl_unt": d_unt, "dppl_bots": d_bots, "dppl_top_sigma_poscontrol": d_tops,
                    "rand_band": rand_band, "poscontrol_valid": valid,
                    "unt_over_rand": (d_unt / rand_band["dppl_mean"]) if rand_band["dppl_mean"] > 0 else float("inf"),
                    "unt_over_bots": (d_unt / d_bots) if d_bots > 0 else float("inf"),
                    "ci_unt_minus_rand": ci_rand, "ci_unt_minus_bots": ci_bots}
        abl.append({"k": k, "ppl_unt": pp_unt["ppl"], "ppl_bots": pp_bots["ppl"],
                    "ppl_top_sigma": pp_tops["ppl"], "contrast": contrast})
        rep["ablation"] = abl
        (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
        print(f"[func][ABL] k={k} dppl U_unt={d_unt:+.3f} rand[{rand_band['dppl_min']:+.3f}.."
              f"{rand_band['dppl_max']:+.3f}] bottom-sigma={d_bots:+.3f} TOPσ(posctrl)={d_tops:+.3f} "
              f"valid={valid} | CI(unt-rand)=[{ci_rand['lo']:+.3f},{ci_rand['hi']:+.3f}] "
              f"CI(unt-bots)=[{ci_bots['lo']:+.3f},{ci_bots['hi']:+.3f}]", flush=True)

    # ---- BH-FDR across the k-family on the one-sided contrast p (both nulls); reported. The frozen
    # verdict rule is the largest-VALID-k paired-CI rule below.
    ks_all = [r["k"] for r in abl]
    p_rand = [r["contrast"]["ci_unt_minus_rand"]["p_le0"] for r in abl]
    p_bots = [r["contrast"]["ci_unt_minus_bots"]["p_le0"] for r in abl]
    rep["bh_fdr"] = {"k": ks_all,
                     "p_unt_minus_rand": p_rand, "bh_unt_minus_rand": _bh_adjust(p_rand).tolist(),
                     "p_unt_minus_bots": p_bots, "bh_unt_minus_bots": _bh_adjust(p_bots).tolist()}

    # ---- VERDICT (§6). Among k whose POSITIVE CONTROL is valid (readout has dynamic range at that k),
    # take the LARGEST. LIVE iff the paired lower-CI of BOTH contrasts > 0 AND both Δppl ratios >= 1.5.
    # Else INERT (with a valid positive control). No valid k -> VOID (instrument floor: uninterpretable).
    valid_ks = [r["k"] for r in abl if r["contrast"]["poscontrol_valid"]]
    if not valid_ks:
        rep["verdict_basis"] = {"valid_ks": [], "reason": "no k cleared the positive-control gate"}
        rep["verdict"] = ("VOID (instrument floor): at no k did the top-sigma positive control raise "
                          "ppl above the random-null band -> the ablation readout has no dynamic range "
                          "at these k; neither LIVE nor INERT is interpretable.")
        final = "VOID"
    else:
        kk = max(valid_ks)
        c = next(r["contrast"] for r in abl if r["k"] == kk)
        ci_r, ci_b = c["ci_unt_minus_rand"], c["ci_unt_minus_bots"]
        live = (ci_r["lo"] > 0) and (ci_b["lo"] > 0) \
            and (c["unt_over_rand"] >= 1.5) and (c["unt_over_bots"] >= 1.5)
        rep["verdict_basis"] = {"k": kk, "valid_ks": valid_ks, **c}
        final = "LIVE" if live else "INERT"
        if live:
            rep["verdict"] = (f"LIVE (functional preserved invariant): at the largest valid k={kk}, "
                              f"removing the untouched q/k band costs MORE ppl than BOTH matched-random "
                              f"and base-bottom-sigma with paired lower-CI>0 for both. "
                              f"dppl(U_unt)={c['dppl_unt']:+.3f} vs rand-mean={c['rand_band']['dppl_mean']:+.3f} "
                              f"vs bottom-sigma={c['dppl_bots']:+.3f} "
                              f"(ratios {c['unt_over_rand']:.2f}x / {c['unt_over_bots']:.2f}x; "
                              f"CI(unt-rand)=[{ci_r['lo']:+.3f},{ci_r['hi']:+.3f}], "
                              f"CI(unt-bots)=[{ci_b['lo']:+.3f},{ci_b['hi']:+.3f}]).")
        else:
            rep["verdict"] = (f"INERT (diagnostic null): at the largest valid k={kk}, removing the "
                              f"untouched q/k band costs ppl ~= matched-random / base-bottom-sigma "
                              f"(no load-bearing excess). dppl(U_unt)={c['dppl_unt']:+.3f} vs "
                              f"rand-mean={c['rand_band']['dppl_mean']:+.3f} vs "
                              f"bottom-sigma={c['dppl_bots']:+.3f}; "
                              f"CI(unt-rand)=[{ci_r['lo']:+.3f},{ci_r['hi']:+.3f}] "
                              f"(lower CI not >0 or ratio<1.5).")
    (out / "invariant_core_func.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78, flush=True)
    print(rep["verdict"], flush=True)
    print(f"INVARIANT_CORE_FUNC_OK verdict={final} "
          f"n_ft={len(repos)} layers={len(want_layers)} k={k_grid}", flush=True)


if __name__ == "__main__":
    main()
