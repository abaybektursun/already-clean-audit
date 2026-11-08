"""GAUGE-DIST (P6) — the edit-distance profile on gemma-2b (primary) + PG1 (secondary, exploratory).

Measures ε*(S,T) = the minimal RELATIVE-Frobenius-norm weight edit, confined to support S and of edit
type T, that pushes the function readout past δ_fun (§9.2). Support classes S ∈ {single_matrix,
single_layer, band3, global}; edit types T ∈ {random direction, adversarial (frozen-budget gradient
ascent on ΔL within S)}. The premium statistic π(S)=ε*_adv(S)/ε*_adv(global) is compared against the
count-matched RANDOM-edit ratio band; the search-artifact gate is a budget-doubling control on the
smallest support. All search logic is the pure, toy-quadratic-tested harness in infra/pg/gauge_dist.py
— this job only supplies the REAL forwards (function_readout / grad_readout) and mirrors the pure
direction search on GPU (a global edit is model-sized, so directions live as torch tensors, never fp64
numpy). Gate real GPU behind the offline gate first.

Firewall (CLAUDE.md / GC): Gemma-1 (google/gemma-2b, 18L) ONLY — `gemma-2b` is the one-char-shorter
Gemma-1 id, NOT Gemma-2. The 18-layer count is asserted on the loaded decoder before any statistic.

Readout provenance (§9.2, Gate-0):
  * gemma-2b PRIMARY, registered: mean next-token NLL shift ΔL on a FROZEN held-out wikitext-2 stream.
  * PG1 SECONDARY, exploratory: the registered PG1 readout is VQA discovery-half soft-accuracy; this
    job wires the SAME next-token-NLL readout through PG1's in-distribution (blank-image) multimodal
    forward as the exploratory NLL PROXY. Soft-accuracy is DISCLOSED as deferred to a successor (it is
    not differentiable, so it cannot drive the adversarial gradient ascent directly). Reported clearly.

Run (offline gate — real data + tokenizer, NO model forward; run before any GPU minute):
    python -m infra.aws_pg.jobs.gauge_dist --gate --out <dir>
Run (real; p5.4xlarge H100 first, g6e L40S fallback):
    pg run infra.aws_pg.jobs.gauge_dist --type p5.4xlarge \
        --args "--model gemma2b --n-windows 16 --window 256" --budget-min 240
    pg run infra.aws_pg.jobs.gauge_dist --type p5.4xlarge \
        --args "--model pg1 --n-windows 16 --window 256" --budget-min 240   # exploratory row
"""
import argparse
import json
import math
import os
import pathlib

# The global-support edit is full-D (~1.98B params); several full-D vectors (base copy, grads,
# flat_grad, fp32 direction+edit) coexist on the GPU. expandable_segments cuts allocator
# fragmentation; the real headroom comes from running on an 80 GB H100 (p5) not a 44 GB L40S.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from infra.pg import gauge_common as GC
from infra.pg import gauge_dist as GD

# Frozen localized sites (§9.2): the middle of the 18-layer Gemma-1 decoder — a neutral, non-edge site.
MID_LAYER = 9
BAND3 = (8, 9, 10)
SEEDS = (0, 1, 2)                 # the 3 seeds (§9.2)
ADV_STEPS = 8                     # the frozen adversarial-ascent budget (budget-doubling → 16)
PROBE_NORM = 1e-2                 # the fixed probe radius (relative norm) the gradient is sampled at


# ---------------------------------------------------------------------------
# frozen held-out text stream.
# ---------------------------------------------------------------------------
def _windows(tok, n_windows, window):
    """The FROZEN held-out probe stream: the first `n_windows` non-trivial windows of the wikitext-2
    test split, tokenized with the model's own tokenizer (add_special_tokens=False so each window is a
    continuous LM chunk). Returns a list of (window_text, win_ids) — text for the PG1 processor forward,
    ids for the gemma-2b text forward. Mirrors conserved_core._wikitext_windows."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    wins = []
    start = 0
    while start < ids.size(0) and len(wins) < n_windows:
        chunk = ids[start:start + window]
        start += window
        if chunk.size(0) < 8:
            break
        wtext = tok.decode(chunk.tolist())
        wids = tok(wtext, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if int(wids.shape[0]) < 8:
            continue
        wins.append((wtext, wids))
    assert wins, "no wikitext windows built"
    return wins


# ---------------------------------------------------------------------------
# the mean next-token NLL forwards (the ΔL readout, both models).
# ---------------------------------------------------------------------------
def _nll_gemma2b(model, device, windows, grad):
    """Mean next-token NLL over the frozen stream via the gemma-2b TEXT-ONLY forward. Returns a torch
    scalar = nll_sum / n_tok. When `grad`, the backward is done PER WINDOW and gradients ACCUMULATE in
    the model's .grad (the caller reads them via editor.flat_grad); the returned value is detached.
    Per-window backward bounds the autograd graph to ONE 256-token window — the full 4096-token graph
    over the 1.98B-param decoder OOMs a 44 GB GPU. Shifted LM loss, fp32 logits."""
    import torch
    n_tok = sum(int(w[1].numel()) - 1 for w in windows)
    assert n_tok > 0, "NLL saw no tokens"
    if not grad:
        total = 0.0
        with torch.no_grad():
            for _wtext, wids in windows:
                inp = wids.unsqueeze(0).to(device)
                logits = model(input_ids=inp).logits[0].float()
                tgt = inp[0, 1:]
                total += float(torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="sum"))
        return torch.tensor(total / n_tok)
    total = 0.0
    with torch.enable_grad():
        for _wtext, wids in windows:
            inp = wids.unsqueeze(0).to(device)
            logits = model(input_ids=inp).logits[0].float()
            tgt = inp[0, 1:]
            loss = torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="sum") / n_tok
            loss.backward()                                          # accumulates into model .grad
            total += float(loss.detach())
    return torch.tensor(total)                                       # detached; grads already set


def _nll_pg1(model, processor, device, windows, grad, blank):
    """Mean next-token NLL over the frozen stream via PG1's IN-DISTRIBUTION multimodal forward (blank
    image + text; the reference PaliGemma text-only-logits gotcha). Exploratory NLL proxy for the PG1
    row. Returns a torch scalar (autograd if `grad`). Text block located by exact token-match to handle
    the processor's trailing newline."""
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    ctx = torch.enable_grad() if grad else torch.no_grad()
    total = None
    n_tok = 0
    with ctx:
        for wtext, wids in windows:
            inp = processor(text=[wtext], images=[blank], return_tensors="pt", padding=False).to(device)
            in_ids = inp["input_ids"][0]
            n_text = int(wids.shape[0])
            assert in_ids.shape[0] > n_text, "text span >= prompt — image prefix missing"
            pos = _locate_text_block(in_ids, wids)
            logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                           pixel_values=inp["pixel_values"].to(torch.bfloat16)).logits[0]
            text_logits = logits[pos:pos + n_text - 1, :].float()
            tgt = wids[1:].to(device)
            loss = torch.nn.functional.cross_entropy(text_logits, tgt, reduction="sum")
            total = loss if total is None else total + loss
            n_tok += int(tgt.numel())
    assert n_tok > 0, "NLL saw no tokens"
    return total / n_tok


# ---------------------------------------------------------------------------
# the in-scope decoder weight editor (apply / restore / gradient).
# ---------------------------------------------------------------------------
class _Editor:
    """Holds the in-scope decoder weight Parameters (18 layers × 7 GC.PROJ), their base copies, and the
    flat coordinate layout the support masks index. `apply` writes W = W_base + edit_segment per key;
    `restore` writes W = W_base; `flat_grad` gathers dNLL/dW over the whole flat vector after a backward
    (= dΔL/dedit, since the base NLL is constant). Directions/edits are full-D torch tensors on device
    (a global edit is model-sized — never fp64 numpy)."""

    def __init__(self, inscope):
        import torch
        self.entries = []            # (name, param, base, offset, numel, shape)
        off = 0
        for name, p in inscope:
            base = p.detach().clone()
            self.entries.append((name, p, base, off, p.numel(), tuple(p.shape)))
            off += p.numel()
        self.D = off
        self.device = self.entries[0][2].device
        self.by_name = {name: (o, o + k) for name, _p, _b, o, k, _s in self.entries}
        self.by_lt = {(GC.layer_index(name), GC.matrix_type(name)): name
                      for name, _p, _b, _o, _k, _s in self.entries}

    def ranges(self, names):
        return [self.by_name[n] for n in names]

    def base_norm(self, names):
        """‖W_base restricted to these keys‖_F (the relative→absolute scale for this support)."""
        f2 = 0.0
        for name, _p, b, _o, _k, _s in self.entries:
            if name in names:
                f2 += float((b.float() ** 2).sum())
        return math.sqrt(f2)

    def apply(self, edit_flat):
        # edit_flat lives in HOST RAM (full-D ~1.98B fp32 = 8 GB — kept off the GPU); each matrix's
        # slice is moved to the GPU transiently so peak GPU memory is model+grads+one slice, not
        # several coexisting 8 GB full-D vectors (which OOM'd both a 44 GB L40S and forced 80 GB).
        import torch
        with torch.no_grad():
            for _name, p, b, o, k, s in self.entries:
                p.data = b + edit_flat[o:o + k].view(s).to(device=p.device, dtype=p.dtype)

    def restore(self):
        import torch
        with torch.no_grad():
            for _name, p, b, _o, _k, _s in self.entries:
                p.data = b

    def zero_grad(self):
        for _name, p, _b, _o, _k, _s in self.entries:
            p.grad = None

    def flat_grad(self):
        import torch
        gs = []
        for _name, p, _b, _o, _k, _s in self.entries:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            gs.append(g.reshape(-1).float().cpu())     # to HOST — the full-D grad never sits on GPU
        return torch.cat(gs)


# ---------------------------------------------------------------------------
# GPU direction search (mirrors the pure gauge_dist reference; torch, model-scale).
# ---------------------------------------------------------------------------
def _random_dir(D, ranges, seed, device):
    """Unit-ℓ2 random direction confined to the support ranges (torch twin of
    gauge_dist.random_support_direction)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    d = torch.zeros(D, dtype=torch.float32)             # HOST fp32 full-D direction (kept off GPU)
    for lo, hi in ranges:
        d[lo:hi] = torch.randn(hi - lo, generator=g)
    d = d / d.norm()
    return d                                            # stays on host; apply() moves slices to GPU


def _adv_dir(grad_readout, D, ranges, seed, n_steps, probe_norm, base_norm, device):
    """Frozen-budget gradient-ascent direction (torch twin of
    gauge_dist.adversarial_support_direction): power-iteration on the support-restricted curvature at a
    fixed probe radius, re-confined to the support each step."""
    import torch
    mask = torch.zeros(D, dtype=torch.bool)             # HOST — the full-D vectors stay off the GPU
    for lo, hi in ranges:
        mask[lo:hi] = True
    d = _random_dir(D, ranges, seed, device)
    radius = float(probe_norm) * float(base_norm)
    for _ in range(int(n_steps)):
        g = grad_readout(radius * d)                    # returns a HOST fp32 full-D grad
        g = torch.where(mask, g, torch.zeros_like(g))
        nrm = g.norm()
        if float(nrm) == 0.0:
            break
        d = g / nrm
    return d


# ---------------------------------------------------------------------------
# offline gate — real data + tokenizer, NO model forward.
# ---------------------------------------------------------------------------
def _offline_gate(out, n_windows, window):
    """§ offline gate: prove the wiring on REAL data + tokenizer + the pure harness, minus the model
    forward, before any GPU minute (memory: feedback_offline_gate_before_gpu). Builds the frozen stream,
    then runs the ENTIRE pure search on a toy quadratic with a KNOWN ε* (mirrors the test's toy) so the
    bisection/premium/detection wiring is exercised end-to-end. Fail-loud."""
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    from transformers import AutoTokenizer
    from infra.pg.entities import ENTITIES
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)
    wins = _windows(tok, n_windows, window)
    assert len(wins) >= 4, f"only {len(wins)} windows — stream too short"

    # toy-quadratic self-test of the pure harness (a planted locality premium; no model).
    d_dim, base_norm, delta = 40, 1.0, 0.5
    rng = np.random.default_rng(0)
    v = np.ones(d_dim) / np.sqrt(d_dim)
    H = 20.0 * np.outer(v, v) + 0.5 * np.eye(d_dim)
    grid = GD.frozen_norm_grid(1e-3, 1e2, 64)

    def readout(edit):
        x = np.asarray(edit, dtype=np.float64)
        return 0.5 * float(x @ H @ x)

    def grad(edit):
        return H @ np.asarray(edit, dtype=np.float64)

    s_mask = GD.mask_from_ranges(d_dim, [(0, 5)])
    g_mask = np.ones(d_dim, dtype=bool)

    def adv_eps(mask, seed):
        dd = GD.adversarial_support_direction(grad, mask, seed, ADV_STEPS, PROBE_NORM, base_norm)
        return GD.eps_star_for_direction(readout, dd, base_norm, grid, delta)["eps_star"]

    eps_g = adv_eps(g_mask, 0)
    eps_s = adv_eps(s_mask, 0)
    pi = GD.premium_ratio(eps_s, eps_g)
    assert eps_g > 0 and eps_s > 0 and math.isfinite(pi), "toy ε*/π not finite"
    assert pi > 1.0, "toy locality premium did not reproduce — harness wiring broken"

    rep = {"object": "gauge_dist_offline_gate", "n_windows": len(wins),
           "window": window, "tokenizer": L0g1.repo_id,
           "toy_eps_global": eps_g, "toy_eps_single": eps_s, "toy_pi": pi,
           "supports": list(GD.SUPPORT_CLASSES), "edit_types": list(GD.EDIT_TYPES),
           "note": "real stream + tokenizer OK; pure ε*/premium harness reproduces a planted premium"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "offline.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"GAUGE_DIST_OFFLINE_OK windows={len(wins)} toy_pi={pi:.3f} eps_global={eps_g:.4g}")


# ---------------------------------------------------------------------------
# real run.
# ---------------------------------------------------------------------------
def _load(model_name, device):
    """Load the model + the in-scope decoder parameters + the NLL forward closure. gemma-2b: text-only
    AutoModelForCausalLM. pg1: PaliGemma multimodal (exploratory NLL proxy). Returns
    (model, inscope, nll_fn, disclosure)."""
    import torch
    from infra.pg.entities import ENTITIES

    def _inscope(named):
        out = []
        for name, p in named:
            if "layers." not in name:
                continue
            if any(t in name for t in GC.PROJ) and name.endswith(".weight"):
                out.append((name, p))
        assert out, "no in-scope decoder projection weights found"
        return out

    if model_name == "gemma2b":
        from transformers import AutoModelForCausalLM
        L0g1 = ENTITIES["L0_g1"]
        model = AutoModelForCausalLM.from_pretrained(
            L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="eager").eval()
        inscope = _inscope(model.named_parameters())

        def nll(windows, grad):
            return _nll_gemma2b(model, device, windows, grad)
        return model, inscope, nll, "gemma-2b registered next-token-NLL readout"

    if model_name == "pg1":
        from infra.aws_pg.jobs.invariant_core import PG1_REPO
        from infra.aws_pg.jobs.invariant_core_func import _load_full_pg1, _blank_image
        from huggingface_hub import HfApi
        from transformers import AutoProcessor
        sha = HfApi().model_info(PG1_REPO).sha
        model = _load_full_pg1(sha, device)
        processor = AutoProcessor.from_pretrained(PG1_REPO, revision=sha)
        blank = _blank_image()
        # in-scope = the LM-tower decoder projections only (firewall still 18L).
        inscope = _inscope((n, p) for n, p in model.named_parameters() if "language_model" in n)

        def nll(windows, grad):
            return _nll_pg1(model, processor, device, windows, grad, blank)
        return (model, inscope, nll,
                "PG1 EXPLORATORY NLL proxy (registered readout = VQA soft-accuracy, deferred)")

    raise ValueError(f"unknown model {model_name!r}")


def _run(model_name, out, n_windows, window, n_delta_seeds):
    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "GAUGE-DIST needs a GPU"
    device = "cuda"
    import numpy as np

    from transformers import AutoTokenizer
    from infra.pg.entities import ENTITIES
    tok = AutoTokenizer.from_pretrained(ENTITIES["L0_g1"].repo_id, revision=ENTITIES["L0_g1"].revision)
    windows = _windows(tok, n_windows, window)
    print(f"[gauge-dist] {len(windows)} frozen windows (window={window})", flush=True)

    model, inscope, nll_fn, disclosure = _load(model_name, device)
    GC.assert_gemma1([name for name, _p in inscope])          # 18-layer firewall
    editor = _Editor(inscope)
    print(f"[gauge-dist] model={model_name} D={editor.D} inscope_mats={len(inscope)} — {disclosure}",
          flush=True)

    # base NLL + δ_fun from the readout's own seed-noise band (window resampling, NO edit; pre-real).
    base_nll = float(nll_fn(windows, grad=False).item())
    rng = np.random.default_rng(1337)
    idx = np.arange(len(windows))
    noise = []
    for _k in range(max(3, n_delta_seeds)):
        sub = [windows[i] for i in rng.choice(idx, size=len(windows), replace=True)]
        noise.append(float(nll_fn(sub, grad=False).item()) - base_nll)
    delta_fun = GD.calibrate_delta_fun(noise, k=3.0)
    grid = GD.frozen_norm_grid(1e-4, 1e0, 64)
    print(f"[gauge-dist] base_nll={base_nll:.4f} δ_fun={delta_fun:.4g} (seed-noise band, k=3)", flush=True)

    def function_readout(edit_flat):
        editor.apply(edit_flat)
        dl = float(nll_fn(windows, grad=False).item()) - base_nll
        editor.restore()
        return dl

    def grad_readout(edit_flat):
        editor.apply(edit_flat)
        editor.zero_grad()
        nll_fn(windows, grad=True)                            # per-window backward accumulates .grad
        g = editor.flat_grad()
        editor.restore()
        return g                                              # dΔL/dedit (full-D torch)

    # frozen supports as key-sets over the in-scope layout.
    supports = {
        "single_matrix": [editor.by_lt[(MID_LAYER, "q_proj")]],
        "single_layer": [editor.by_lt[(MID_LAYER, t)] for t in GC.PROJ],
        "band3": [editor.by_lt[(L, t)] for L in BAND3 for t in GC.PROJ],
        "global": [name for name, _p in inscope],
    }
    n_params = {s: sum((hi - lo) for lo, hi in editor.ranges(names)) for s, names in supports.items()}
    base_norms = {s: editor.base_norm(set(names)) for s, names in supports.items()}

    # ---- ε*(S,T,seed) over the frozen grid ----------------------------------
    eps = {s: {"random": [], "adversarial": []} for s in GD.SUPPORT_CLASSES}
    detail = []
    for s in GD.SUPPORT_CLASSES:
        rng_s = editor.ranges(supports[s])
        bn = base_norms[s]
        for seed in SEEDS:
            d_rand = _random_dir(editor.D, rng_s, seed, device)
            r_rand = GD.eps_star_for_direction(function_readout, d_rand, bn, grid, delta_fun)
            d_adv = _adv_dir(grad_readout, editor.D, rng_s, seed, ADV_STEPS, PROBE_NORM, bn, device)
            r_adv = GD.eps_star_for_direction(function_readout, d_adv, bn, grid, delta_fun)
            eps[s]["random"].append(r_rand["eps_star"])
            eps[s]["adversarial"].append(r_adv["eps_star"])
            detail.append({"support": s, "seed": seed, "base_norm": bn,
                           "random": r_rand, "adversarial": r_adv})
            print(f"[gauge-dist] {s:13s} seed{seed} ε*_rand={r_rand['eps_star']:.4g} "
                  f"ε*_adv={r_adv['eps_star']:.4g} "
                  f"(sat_r={r_rand['saturated']} sat_a={r_adv['saturated']})", flush=True)

    # ---- premium π(S), count-matched null band, analytic overlay ------------
    eps_adv_g = np.asarray(eps["global"]["adversarial"], dtype=float)
    pi_by_support, null_hi_by_support, consist_by_support, profile = {}, {}, {}, {}
    for s in GD.LOCALIZED_SUPPORTS:
        pi_seeds = np.asarray(eps[s]["adversarial"], dtype=float) / eps_adv_g
        band = GD.random_ratio_band(eps[s]["random"], eps["global"]["random"])
        pi_by_support[s] = float(np.median(pi_seeds))
        null_hi_by_support[s] = band["hi"]
        consist_by_support[s] = GD.seed_consistent(pi_seeds.tolist(), band["hi"])
        profile[s] = {"pi_seeds": pi_seeds.tolist(), "pi_median": pi_by_support[s],
                      "random_band": band, "seed_consistent": consist_by_support[s],
                      "analytic_overlay": GD.analytic_count_overlay(n_params[s], n_params["global"]),
                      "n_params": n_params[s]}

    # ---- budget-doubling control on the smallest support (single_matrix) -----
    s0 = "single_matrix"
    rng0 = editor.ranges(supports[s0]); bn0 = base_norms[s0]
    pi_single = pi_by_support[s0]
    pi_double_seeds = []
    for seed in SEEDS:
        d2 = _adv_dir(grad_readout, editor.D, rng0, seed, 2 * ADV_STEPS, PROBE_NORM, bn0, device)
        e2 = GD.eps_star_for_direction(function_readout, d2, bn0, grid, delta_fun)["eps_star"]
        pi_double_seeds.append(e2 / eps_adv_g[seed])
    pi_double = float(np.median(pi_double_seeds))
    seed_band = float(np.max(np.asarray(eps[s0]["adversarial"]) / eps_adv_g)
                      - np.min(np.asarray(eps[s0]["adversarial"]) / eps_adv_g))
    budget_ok = GD.budget_doubling_ok(pi_single, pi_double, seed_band)

    det = GD.detection_premium(pi_by_support, null_hi_by_support, consist_by_support, budget_ok)

    rep = {"object": "gauge_dist_v1", "generation": "gemma1_18L", "model": model_name,
           "disclosure": disclosure, "base_nll": base_nll, "delta_fun": delta_fun,
           "seed_noise_shifts": noise, "n_windows": len(windows), "window": window,
           "adv_steps": ADV_STEPS, "probe_norm": PROBE_NORM, "seeds": list(SEEDS),
           "mid_layer": MID_LAYER, "band3": list(BAND3), "n_params": n_params,
           "grid": {"lo": 1e-4, "hi": 1e0, "n": 64}, "eps_star": eps, "profile": profile,
           "budget_doubling": {"pi_single": pi_single, "pi_double": pi_double,
                               "pi_double_seeds": pi_double_seeds, "seed_band": seed_band,
                               "budget_ok": budget_ok},
           "detection": det, "detail": detail,
           "estimand": "ε*(S,T) minimal relative-Frobenius function-changing edit; premium "
                       "π(S)=ε*_adv(S)/ε*_adv(global) vs the count-matched random-edit ratio band"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gauge_dist.json").write_text(json.dumps(rep, indent=2, default=str))
    print("=" * 78)
    print(f"GAUGE_DIST_OK model={model_name} detected={det['detected']} "
          f"premium_supports={det['premium_supports']} budget_ok={budget_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="store_true", help="offline gate (real data, NO model forward)")
    ap.add_argument("--model", default="gemma2b", choices=("gemma2b", "pg1"))
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-delta-seeds", type=int, default=5, help="readout seed-noise draws for δ_fun")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.gate:
        _offline_gate(out, a.n_windows, a.window)
    else:
        _run(a.model, out, a.n_windows, a.window, a.n_delta_seeds)


if __name__ == "__main__":
    main()
