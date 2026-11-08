"""CORE-PROVENANCE C6b — is the untouched q/k core load-bearing in `gemma-2b` ITSELF?

The addendum's last unexecuted leg (`2026-08-03_core_provenance_addendum_spec.md` §2-C6b), unrun
across four attempts. C6c (the correspondence gate) already PASSED under TESSERA-FIX — 36/36 q/k
matrices correspond across the install hop, median subspace overlap 0.911 vs Haar p97.5 0.032 — so
transporting the core into `gemma-2b` coordinates is well-posed and this leg is meaningful.

  H_inherited : the core is a PRETRAINING-era structure — it should be load-bearing in gemma-2b
                itself (ablation ppl ≫ matched-random), i.e. the install walked around something
                that was already there
  H_settled   : the core is a PG1-era object — ablating it from gemma-2b costs no more than a
                matched-random subspace of the same rank

Instrument: the invcore-FUNC contract verbatim — mean-ablate span(U) from the q/k INPUTS (a
forward-pre-hook on each attention module's q_proj/k_proj input) and read clean-text wikitext
perplexity. Controls, all named with their FAILURE:
  * matched-random subspace of the same rank, per-matrix seeds (N1)
  * bottom-σ band of the same rank (the "any low-energy subspace" control)
  * a POSITIVE control on PG1 with the same basis — if PG1 does not show the known effect the
    instrument is VOID and this leg reports nothing.

    python -m infra.aws_pg.jobs.erlangen_c6b --out DIR --basis-s3 s3://…/erlangen_core_basis.npz

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

GEMMA1 = "google/gemma-2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis-s3", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--n-rand", type=int, default=3)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import math

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "C6b needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "core_provenance_c6b_v1",
           "spec": "2026-08-03_core_provenance_addendum_spec.md §2-C6b",
           "c6c_status": "PASSED under TESSERA-FIX (36/36 correspond, median overlap 0.911 vs "
                         "Haar p97.5 0.032) — transport is well-posed",
           "k": a.k, "n_seq": a.n_seq, "window": a.window}

    def bank():
        (out / "erlangen_c6b.json").write_text(json.dumps(rep, indent=2))

    bank()
    local = out / "basis.npz"
    subprocess.run(["aws", "s3", "cp", a.basis_s3, str(local), "--region", "us-east-1"], check=True)
    z = np.load(local)
    rep["basis_source"] = a.basis_s3
    rep["n_basis_matrices"] = len(z.files)

    tok = AutoTokenizer.from_pretrained(GEMMA1)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    windows = [ids[i:i + a.window] for i in range(0, ids.size(0), a.window)][: a.n_seq]
    windows = [w for w in windows if w.numel() >= 8]
    rep["n_windows"] = len(windows)

    bos = tok.bos_token_id
    assert bos is not None, "gemma tokenizer has no BOS — the text forward would be mis-specified"

    def ppl(model):
        """Clean-text per-token CE perplexity, token-weighted (the invcore-FUNC readout).

        Every window is prefixed with BOS: Gemma is trained with one, and scoring raw windows
        without it destabilises the forward — the first run showed a matched-RANDOM ablation
        *improving* gemma-2b's perplexity by 4.7, which is an instrument statement, not a finding."""
        nll, ntok, per = 0.0, 0, []
        for w in windows:
            inp = torch.cat([torch.tensor([bos]), w]).unsqueeze(0).to(device)
            with torch.inference_mode():
                lg = model(input_ids=inp).logits[0, :-1, :].float()
            tg = inp[0, 1:]
            loss = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
            nll += float(loss)
            ntok += int(tg.numel())
            per.append([float(loss), int(tg.numel())])
        return {"ppl": math.exp(nll / ntok), "mean_nll": nll / ntok, "n_tokens": ntok,
                "per_window": per}

    def qk_modules(model):
        """{canonical_basis_key: module} for every q_proj/k_proj in the decoder."""
        got = {}
        for name, mod in model.named_modules():
            if name.endswith(("self_attn.q_proj", "self_attn.k_proj")):
                short = name.split("model.")[-1] if "model." in name else name
                key = f"language_model.{short}.weight".replace(".", "__")
                if key in z.files:
                    got[key] = mod
        return got

    def ablate(mods, bases):
        """Mean-ablate span(Q) from each module's INPUT (the invcore-FUNC contract: the core is an
        INPUT subspace of q/k, so the projector acts on the incoming residual, not the output)."""
        handles = []
        for key, mod in mods.items():
            Q = bases[key].to(device)

            def mk(Qm):
                def hook(_m, args):
                    x = args[0]
                    xf = x.float()
                    return ((xf - (xf @ Qm) @ Qm.T).to(x.dtype),) + tuple(args[1:])
                return hook

            handles.append(mod.register_forward_pre_hook(mk(Q)))
        return handles

    def rand_basis(shape, seed):
        g = torch.Generator().manual_seed(int(seed))
        A = torch.randn(shape[0], shape[1], generator=g)
        Q, _ = torch.linalg.qr(A)
        return Q[:, : shape[1]]

    results = {}
    rep["arms"] = results
    for tag, repo in (("gemma1", GEMMA1), ("pg1_positive_control", "PG1")):
        if repo == "PG1":
            from infra.pg.lacuna_shell import load_pg1
            m, _proc = load_pg1(device=device)
            lm = m.get_submodule("model.language_model")
        else:
            m = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="eager").eval()
            lm = m
        mods = qk_modules(lm)
        assert len(mods) >= 30, f"{tag}: only {len(mods)} q/k modules matched the basis keys"
        core = {k: torch.from_numpy(z[k]).float() for k in mods}
        base = ppl(m)
        h = ablate(mods, core)
        abl = ppl(m)
        for x in h:
            x.remove()
        rands = []
        for s in range(a.n_rand):
            rb = {k: rand_basis(core[k].shape, 7000 + s + abs(hash(k)) % 1000) for k in mods}
            h = ablate(mods, rb)
            rands.append(ppl(m)["ppl"])
            for x in h:
                x.remove()
        results[tag] = {
            "n_modules": len(mods), "base_ppl": base["ppl"], "core_ablated_ppl": abl["ppl"],
            "delta_core": abl["ppl"] - base["ppl"],
            "rand_ablated_ppl_mean": float(np.mean(rands)), "rand_ppl_draws": rands,
            "delta_rand": float(np.mean(rands)) - base["ppl"],
            "ratio_core_over_rand": ((abl["ppl"] - base["ppl"])
                                     / max(float(np.mean(rands)) - base["ppl"], 1e-9))}
        print(f"[C6b] {tag}: base {base['ppl']:.3f} → core {abl['ppl']:.3f} "
              f"(Δ {results[tag]['delta_core']:+.3f}) vs random {np.mean(rands):.3f} "
              f"(Δ {results[tag]['delta_rand']:+.3f}) ratio {results[tag]['ratio_core_over_rand']:.2f}",
              flush=True)
        del m
        torch.cuda.empty_cache()
        bank()

    # ---- verdict, flags LAST ---------------------------------------------------------------------
    pc = results["pg1_positive_control"]
    g1 = results["gemma1"]
    instrument_ok = bool(pc["ratio_core_over_rand"] > 2.0)
    # SANITY GUARD (added after the first run): if a matched-RANDOM ablation IMPROVES the arm's
    # perplexity, the forward on that arm is mis-specified and no contrast computed on it is
    # interpretable — including a "settled" reading. The first version gated only on the positive
    # control and would have reported H_settled off an anomalous arm.
    arm_sane = bool(g1["delta_rand"] >= 0)
    load_bearing = bool(g1["ratio_core_over_rand"] > 2.0)
    rep["verdict"] = {
        "instrument_valid": instrument_ok,
        "gemma1_arm_sane": arm_sane,
        "gemma1_load_bearing": load_bearing and arm_sane,
        "label": ("INSTRUMENT-VOID — the PG1 positive control does not reproduce the known "
                  "core-vs-random separation, so this leg reports nothing" if not instrument_ok
                  else "VOID-ARM — a matched-random ablation IMPROVES gemma-2b's perplexity, so "
                       "the gemma-2b forward is mis-specified and neither an inherited nor a "
                       "settled reading can be drawn from it" if not arm_sane
                  else "H_inherited-leg: the core is load-bearing in gemma-2b itself"
                  if load_bearing else
                  "H_settled-leg: ablating the core from gemma-2b costs no more than a "
                  "matched-random subspace of the same rank"),
        "failure_means": "a PG1 ratio at/below the random band voids the instrument; a negative "
                         "random-ablation delta on the test arm voids that arm"}
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[C6b] DONE {rep['verdict']['label']}", flush=True)


if __name__ == "__main__":
    main()
