"""TESSERA-FIX C6 part (b) — is the untouched core LOAD-BEARING in `gemma-2b` itself?

Spec `tessera-fix-freeze-2026-08-03b` §2 C6(b): *"FUNC-style ablation on `gemma-2b` itself — ablate
the core basis from the base LM's q/k inputs (text-only forward, wikitext ppl, matched-random +
bottom-σ controls, the invcore-FUNC instrument verbatim)."*

WHY IT ONLY RUNS NOW. C6(c) gates this, and it PASSED: PG1 and gemma-2b read q/k input subspaces
overlapping at a median of **0.9109** against a Haar p97.5 of **0.0325**, 36/36 matrices. So "the same
core subspace" is meaningful across the install hop, and ablating a PG1-derived basis from gemma-2b
is not a category error.

WHAT IT DECIDES. `H_core-inherited` needs BOTH legs: the install write avoids the core (part a) AND
the core is load-bearing in the base LM (this). If the core is inert in `gemma-2b`, "inherited" would
mean the children inherited a **dead** subspace — untouched because nothing reads it — which is the
`INERT` reading `invariant_core_func` was built to separate from `LIVE`.

THE INSTRUMENT, and the one place it necessarily differs from the PG1 version. `invariant_core_func`
runs blank-image multimodal forwards because PG1 is a VLM and its text-only MMLU sits at chance.
`gemma-2b` is a plain causal LM, so the forward is **text-only wikitext** — simpler, and the readout
(windowed perplexity) is the same graded quantity that instrument settled on for exactly this reason.
Recorded as **AF-8**: the ablation, the controls, the sweep and the verdict rule are the FUNC
instrument's; the forward is the one the model actually has.

  ACT (cheap, decisive-if-INERT). Capture the q/k INPUT (post-`input_layernorm` residual) over
  wikitext, accumulate its Gram, and compare the activation ENERGY inside U_unt(L,k) against a
  matched-random k-subspace. If U_unt holds ≤ random energy, the band is gradient-starved — untouched
  because nothing drives it — and the verdict is INERT without spending the ablation.

  ABL (only if LIVE-candidate). Ablate at EVERY layer, x' = x − QQᵀx, and measure wikitext ppl under
  (a) U_unt, (b) matched-random k, (c) base bottom-σ k, (d) none. **The verdict is the CONTRAST**:
  U_unt-ablation ≈ random/bottom-σ ⇒ INERT; U_unt-ablation costs materially more ⇒ LIVE.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_c6b --type g6e.2xlarge \\
      --args "--basis s3://.../tessera_c6b/out/core_basis.pt" \\
      --s3-prefix results/tessera_c6_func --budget-min 240

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

GEMMA1_REPO = "google/gemma-2b"
K_GRID = (8, 32, 128)
N_ACT_SEQ = 64
N_PPL_SEQ = 120
PPL_WINDOW = 512
# FUNC's OWN threshold, read from the banked artifact rather than invented:
# invcore_func/out/invariant_core_func.json -> act_live_ratio_threshold = 2.0.
ACT_LIVE_MIN = 2.0
# The banked PG1 anchors the positive control must reproduce in KIND (same instrument, different
# forward -- see AF-8): ACT mean ratio 43.97 @k=8, and LIVE by the ablation contrast
# (dppl_unt +49.138 vs random +2.063 = 23.82x at k=128).
PG1_ANCHOR = {"act_mean_ratio_k8": 43.97, "dppl_unt_k128": 49.138, "unt_over_rand_k128": 23.82,
              "verdict": "LIVE", "source": "results/invcore_func/out/invariant_core_func.json"}
PG1_REPO = "google/paligemma-3b-pt-224"


def _read_basis(uri):
    import torch
    dst = "/tmp/c6_core_basis.pt"
    if str(uri).startswith("s3://"):
        subprocess.run(["aws", "s3", "cp", str(uri), dst, "--only-show-errors"], check=True)
    else:
        dst = uri
    return torch.load(dst, map_location="cpu", weights_only=False)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", required=True, help="core_basis.pt from tessera_c6")
    ap.add_argument("--n-act", type=int, default=N_ACT_SEQ)
    ap.add_argument("--n-ppl", type=int, default=N_PPL_SEQ)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    bank = _read_basis(a.basis)
    basis = bank["basis"]
    tok = AutoTokenizer.from_pretrained(GEMMA1_REPO)

    def load_lm(which):
        """A text-only causal LM. `gemma-2b` is the base; `pg1` is PG1's DECODER in a gemma-2b
        shell — PG1's LM *is* a gemma-2b, so this is the same architecture with the child's
        weights, and it lets the identical text-shell pipeline run on both."""
        m = AutoModelForCausalLM.from_pretrained(GEMMA1_REPO, torch_dtype=torch.float32,
                                                 device_map="cuda",
                                                 attn_implementation="eager").eval()
        if which == "pg1":
            from infra.aws_pg.jobs.tessera_s_core import _hf_lm_sd
            sd = _hf_lm_sd(PG1_REPO)
            n = 0
            for nm, prm in m.model.named_parameters():
                key = f"layers.{nm}" if not nm.startswith("layers.") else nm
                if key in sd and tuple(sd[key].shape) == tuple(prm.shape):
                    with torch.no_grad():
                        prm.copy_(sd[key].to(prm.dtype))
                    n += 1
            assert n >= 100, f"PG1 decoder swap wrote only {n} tensors"
            print(f"[C6b] PG1 decoder installed into the gemma-2b shell ({n} tensors)", flush=True)
        return m

    model = load_lm("gemma2b")
    layers = model.model.layers
    n_layers = len(layers)
    rep = {"object": "TESSERA-FIX C6(b) — is the untouched core load-bearing in gemma-2b?",
           "spec": "tessera-fix-freeze-2026-08-03b §2 C6(b)",
           "amendment": "AF-8 — the FUNC instrument's ablation, controls, sweep and verdict rule are "
                        "reused verbatim; the FORWARD is text-only wikitext because gemma-2b is a "
                        "plain causal LM, where the PG1 version needs blank-image multimodal "
                        "forwards. The readout (windowed ppl) is the same graded quantity.",
           "gate": "runs only if C6c PASSES",
           "k_grid": list(K_GRID), "n_layers": n_layers,
           "act_live_min": ACT_LIVE_MIN, "act_live_min_source": "FUNC's banked "
           "act_live_ratio_threshold, not a value invented here",
           "pg1_anchor": PG1_ANCHOR,
           "positive_control": "the SAME text-shell pipeline is run on PG1's decoder, where the "
                               "band is known LIVE. If this instrument cannot detect it there, the "
                               "gemma-2b reading is VOID -- an instrument that finds nothing must "
                               "first be shown capable of finding something (§2 C6b).",
           "spec_tag": "core-provenance-freeze-2026-08-03"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # q_proj is the PRIMARY confirmed object (invariant_core: 18/18 layers BEYOND_NULL, and the
    # band is q/k-shared at 114x random), so the q-defined band IS the shared band.
    U = {}
    for L in range(n_layers):
        key = f"layers.{L}.self_attn.q_proj.weight"
        assert key in basis, f"core basis has no {key}"
        U[L] = basis[key].to(torch.float32)
    d_model = U[0].shape[0]

    # Reuse the repo's own constants. Hardcoding the legacy canonical name "wikitext" fails on the
    # current huggingface_hub ("Repository id must be 'namespace/name'"), and text_eval has carried
    # the namespaced id all along -- the same class as citing an artifact path that does not exist.
    from infra.pg.text_eval import WIKITEXT_CONFIG, WIKITEXT_REPO
    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    txt = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(txt, return_tensors="pt").input_ids[0]
    wins = [ids[i * PPL_WINDOW:(i + 1) * PPL_WINDOW] for i in range(max(a.n_act, a.n_ppl))]
    wins = [w for w in wins if w.numel() == PPL_WINDOW]
    assert len(wins) >= a.n_ppl, f"only {len(wins)} full windows"

    # ---- ACT: how much activation energy sits inside U_unt vs a matched-random subspace? --------
    grams = {L: torch.zeros(d_model, d_model, dtype=torch.float64) for L in range(n_layers)}
    hooks = []

    def mk(L):
        def h(mod, inp, o):
            x = inp[0].detach().reshape(-1, d_model).double().cpu()
            grams[L] += x.T @ x
        return h
    for L in range(n_layers):
        hooks.append(layers[L].self_attn.q_proj.register_forward_hook(mk(L)))
    with torch.inference_mode():
        for w in wins[:a.n_act]:
            model(input_ids=w.unsqueeze(0).to("cuda"))
    for h in hooks:
        h.remove()

    rng = np.random.default_rng(a.seed)
    act = {}
    for kk in K_GRID:
        ratios = []
        for L in range(n_layers):
            G = grams[L].numpy()
            tot = float(np.trace(G))
            Uk = U[L][:, :kk].numpy().astype(np.float64)
            e_core = float(np.trace(Uk.T @ G @ Uk)) / tot
            rnd = []
            for _ in range(20):
                Q, _r = np.linalg.qr(rng.standard_normal((d_model, kk)))
                rnd.append(float(np.trace(Q.T @ G @ Q)) / tot)
            ratios.append(e_core / float(np.mean(rnd)))
        act[str(kk)] = {"per_layer_ratio": ratios, "median_ratio": float(np.median(ratios)),
                        "min": float(min(ratios)), "max": float(max(ratios))}
        print(f"[C6b ACT] k={kk} energy-in-core / random = median {act[str(kk)]['median_ratio']:.4f}",
              flush=True)
    rep["ACT"] = act
    live_candidate = bool(act["8"]["median_ratio"] > ACT_LIVE_MIN)
    rep["ACT_verdict"] = ("LIVE-candidate — the band holds more activation energy than a matched-"
                          "random subspace, so the ablation contrast is worth running"
                          if live_candidate else
                          "INERT — the band holds no more activation energy than matched-random, "
                          "i.e. it is gradient-starved: untouched because nothing drives it. FUNC's "
                          "own rule stops here without spending the ablation.")
    _w()
    print(f"[C6b] ACT verdict: {rep['ACT_verdict'][:70]}", flush=True)

    # ---- ABL: only if LIVE-candidate ------------------------------------------------------------
    if not live_candidate:
        rep["H_core_load_bearing"] = {"verdict": "INERT", "leg": "ACT gate",
                                      "reading": "an INHERITED-but-INERT core would mean PG1's "
                                                 "children inherited a subspace nothing reads — "
                                                 "untouched by construction, not by protection"}
        rep["elapsed_s"] = time.time() - t0
        _w()
        print("TESSERA_C6B_OK", flush=True)
        return

    def ppl(proj=None):
        tot, ntok = 0.0, 0
        hs = []
        if proj is not None:
            # Ablate on the projection INPUT at every layer -- x' = x - QQ'x -- so the band is
            # removed from what attention reads, which is the FUNC instrument's intervention.
            def mkpre(L):
                def h(mod, args):
                    x = args[0]
                    Q = proj[L].to(x.device, x.dtype)
                    return (x - (x @ Q) @ Q.T,)
                return h
            for L in range(n_layers):
                hs.append(layers[L].self_attn.q_proj.register_forward_pre_hook(mkpre(L)))
                hs.append(layers[L].self_attn.k_proj.register_forward_pre_hook(mkpre(L)))
        with torch.inference_mode():
            for w in wins[:a.n_ppl]:
                x = w.unsqueeze(0).to("cuda")
                o = model(input_ids=x, labels=x)
                tot += float(o.loss) * (x.numel() - 1)
                ntok += x.numel() - 1
        for h in hs:
            h.remove()
        return float(np.exp(tot / ntok))

    base_ppl = ppl(None)
    print(f"[C6b ABL] unablated ppl {base_ppl:.4f}", flush=True)
    abl = {}
    for kk in K_GRID:
        arms = {}
        arms["core"] = ppl({L: U[L][:, :kk] for L in range(n_layers)})
        rQ = {}
        for L in range(n_layers):
            Q, _r = np.linalg.qr(rng.standard_normal((d_model, kk)))
            rQ[L] = torch.from_numpy(Q).float()
        arms["matched_random"] = ppl(rQ)
        bQ = {}
        for L in range(n_layers):
            W = layers[L].self_attn.q_proj.weight.detach().float().cpu().numpy()
            _u, _s, vt = np.linalg.svd(W, full_matrices=False)
            bQ[L] = torch.from_numpy(np.ascontiguousarray(vt[-kk:].T)).float()
        arms["bottom_sigma"] = ppl(bQ)
        abl[str(kk)] = {**arms, "unablated": base_ppl,
                        "core_cost": arms["core"] / base_ppl,
                        "random_cost": arms["matched_random"] / base_ppl,
                        "bottom_sigma_cost": arms["bottom_sigma"] / base_ppl,
                        "core_over_random": (arms["core"] / base_ppl) / (arms["matched_random"] / base_ppl)}
        print(f"[C6b ABL] k={kk} core {arms['core']:.3f} random {arms['matched_random']:.3f} "
              f"bottomσ {arms['bottom_sigma']:.3f} (base {base_ppl:.3f})", flush=True)
        rep["ABL"] = abl
        _w()

    r8 = abl["8"]
    live = bool(r8["core_over_random"] > ACT_LIVE_MIN)
    rep["H_core_load_bearing"] = {
        "verdict": "LIVE" if live else "INERT",
        "leg": "ablation contrast",
        "core_over_random_k8": r8["core_over_random"],
        "rule": "the CONTRAST is the verdict: core-ablation ≈ random/bottom-σ ⇒ INERT; materially "
                "worse ⇒ LIVE (FUNC's rule verbatim)"}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"\nH_core load-bearing in gemma-2b: {rep['H_core_load_bearing']['verdict']}")
    print("TESSERA_C6B_OK", flush=True)


if __name__ == "__main__":
    main()
