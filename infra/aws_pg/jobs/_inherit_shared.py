"""INHERIT shared job glue — the one place the Arm-G jobs (gate0/controls/disc/confirm) bind the
pure instrument (inherit_core / ctrliface_core) to artifacts and arms, so every stage shares a
single verified path (no drift). Heavy imports inside functions.

Frozen selection provenance: layers/k*/k-grid are RESOLVED HERE from the CTRLIFACE artifacts
(confirm metrics.json depth profile + controls.json k) — never typed in. The V-bank is LOADED
(never re-estimated); h̄/E_target/trace are PG1-NATIVE, measured on discovery items only, with
E_target and trace computed over the SAME captured rows (the em-rank ratio is otherwise biased
by sample size).
"""
from __future__ import annotations

import json
import pathlib
import subprocess

S3_BUCKET = "s3://pg-lineage-769063704696"
CTRLIFACE_BANK = f"{S3_BUCKET}/results/ctrliface_discovery/out/bases.pt"
CTRLIFACE_CONFIRM = f"{S3_BUCKET}/results/ctrliface_confirm/out/metrics.json"
CTRLIFACE_CONTROLS = f"{S3_BUCKET}/results/ctrliface_controls/out/controls.json"
INVCORE_FUNC_JSON = f"{S3_BUCKET}/results/invcore_func/out/invariant_core_func.json"
INVCORE_SCRATCH = f"{S3_BUCKET}/results/invcore_full2/out/scratch"
N_FT = 24                     # invariant_core fine-tune cloud size (asserted on the file listing)

ARM_KINDS = ("identity", "saction", "saction_redraw", "random_k", "random_energy", "shuffle",
             "topsig_native", "topsig_transported", "positive", "floor")


def fetch(uri, dst):
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def load_bank(workdir="/tmp/inherit"):
    """Download + validate the frozen CTRLIFACE V-bank. Returns (bases, pi05_hbars, pi05_traces)
    — the π0.5 hbar/trace are returned ONLY for provenance logging; PG1 arms never consume them."""
    import torch
    from infra.pg.inherit_core import validate_bank
    p = fetch(CTRLIFACE_BANK, f"{workdir}/bases.pt")
    bundle = torch.load(p, map_location="cpu", weights_only=False)
    return validate_bank(bundle)


def load_redraw_bases(full_bank_s3, workdir="/tmp/inherit"):
    """The re-bank's full-Gram top-128 per layer — an INDEPENDENT same-recipe S_action draw, the
    reproducibility robustness leg (`saction_redraw`). Returns {L: Q (d,128) fp64}."""
    import torch
    p = fetch(full_bank_s3, f"{workdir}/bases_full.pt")
    full = torch.load(p, map_location="cpu", weights_only=False)["full"]
    return {int(L): torch.as_tensor(full[L]["vecs"], dtype=torch.float64)[:, :128] for L in full}


def resolve_selection(workdir="/tmp/inherit"):
    """The frozen §3 selection, mechanically from the CTRLIFACE artifacts: primary/secondary =
    two strongest-bite depth-profile layers, anchors = two flattest; k* and the k-grid verbatim."""
    from infra.pg.inherit_core import promote_layers
    m = json.loads(fetch(CTRLIFACE_CONFIRM, f"{workdir}/ctrliface_confirm.json").read_text())
    c = json.loads(fetch(CTRLIFACE_CONTROLS, f"{workdir}/ctrliface_controls.json").read_text())
    prof = {L: v["saction"]["rf"] for L, v in m["arm_a"]["rf_by_L"].items()}
    sel = promote_layers(prof)
    sel["k_star"] = int(c["k"])
    sel["k_grid"] = sorted(int(k) for k in m["arm_a"]["rf_by_k"])
    sel["depth_profile"] = prof
    assert sel["k_star"] in sel["k_grid"]
    return sel


def em_rank_inputs(X, hbar, Vk):
    """(E_target, trace) for the energy-matched null, BOTH over the SAME rows of X (n,d)."""
    import torch
    from infra.pg.ctrliface_core import removed_energy
    E = removed_energy(X, Vk, hbar)
    tr = float(((torch.as_tensor(X, dtype=torch.float64)
                 - torch.as_tensor(hbar, dtype=torch.float64)) ** 2).sum())
    assert 0 < E < tr, f"E_target {E} vs trace {tr}"
    return E, tr


def topsig_basis(X, hbar, k):
    """Top-k eigvecs of the activation covariance Σ(x−h̄)(x−h̄)ᵀ over X — the top-σ subspace
    (native positive when X is PG1's; transported positive when X is π0.5's, banked)."""
    import torch
    from infra.pg.ctrliface_core import topk_eigvecs
    Xc = torch.as_tensor(X, dtype=torch.float64) - torch.as_tensor(hbar, dtype=torch.float64)
    Q, _ = topk_eigvecs(Xc.transpose(0, 1) @ Xc, k)
    return Q


def build_arm_Q(kind, L, k, bases, seed, X=None, hbar=None, topsig_bank=None, redraw_bases=None,
                n_layers=18):
    """The ablation basis for one PG1 arm. Returns Q (d,k′ fp64) or None (identity/positive/floor
    are dedicated constructions). h̄ is ALWAYS the PG1-native target, supplied by the caller.
    `redraw_bases` = an independent same-recipe re-estimate's top-k (the reproducibility robustness
    leg); `topsig_bank` = the π0.5 activation-covariance top-σ (transported positive)."""
    import torch
    from infra.pg.ctrliface_core import energy_matched_random
    from infra.pg.inherit_core import slice_k
    from infra.pg.projector_ablation import random_basis
    assert kind in ARM_KINDS, f"unknown arm kind {kind}"
    d = bases[0].shape[0]
    if kind in ("identity", "positive", "floor"):
        return None
    if kind == "saction":
        return slice_k(bases[L], k)
    if kind == "saction_redraw":
        assert redraw_bases is not None, "saction_redraw needs the re-bank full-Gram top-k"
        return slice_k(redraw_bases[L], k)
    if kind == "random_k":
        return random_basis(d, k, seed, "cpu", torch.float64)
    if kind == "random_energy":
        E, tr = em_rank_inputs(X, hbar, slice_k(bases[L], k))
        Q, _r = energy_matched_random(d, E, tr, seed, "cpu", torch.float64)
        return Q
    if kind == "shuffle":
        return slice_k(bases[(L + n_layers // 2) % n_layers], k)
    if kind == "topsig_native":
        return topsig_basis(X, hbar, k)
    assert topsig_bank is not None, "topsig_transported needs the π0.5 covariance bank"
    return slice_k(topsig_bank[L], k)


def eval_arm(model, proc, layers, task, rows, kind, L, k, hbar, device="cuda",
             Q=None, seed=0, batch_size=16):
    """Score one (arm, task) cell under the frozen intervention contract. h̄ is PG1-native."""
    import contextlib
    from infra.pg import inherit_pg1 as IP
    if kind == "identity":
        ctx = contextlib.nullcontext()
    elif kind == "floor":
        return IP.eval_pg1(model, proc, task, rows, device, ctx=None,
                           batch_size=batch_size, scramble_seed=seed)
    elif kind == "positive":
        ctx = IP.replace_at(layers, L, hbar)
    else:
        assert Q is not None
        ctx = IP.ablate_at(layers, L, Q, hbar)
    return IP.eval_pg1(model, proc, task, rows, device, ctx=ctx, batch_size=batch_size)


def vqa_forward_batches(proc, task, rows, device, batch_size=16):
    """Prefill-only forward batches over eval items (the h̄/X harvest input — no generation)."""
    import torch
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = [task.image_fn(r).convert("RGB") for r in batch]
        prompts = [task.prompt_fn(r) for r in batch]
        inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        out.append(inp)
    return out


def derive_uunt(layer_list, k, workdir="/tmp/inherit/uunt"):
    """U_unt(L,k) from the banked invcore_full2 scratch ΔWs (fp16 q_proj, 24 FTs/layer):
    consensus Gram → bottom-k. Returns {L: (U (2048,k), eigvals)}. The fp16-scratch precision
    (vs the FUNC job's fp32-repo path) is a disclosed instrument difference, validated at Gate-0
    (orthonormality + near-zero bottom-band eigenvalues + FUNC-artifact cross-checks)."""
    import numpy as np
    from infra.pg.inherit_core import bottom_eigvecs, consensus_gram
    out = {}
    for L in layer_list:
        deltas = []
        for i in range(N_FT):
            p = fetch(f"{INVCORE_SCRATCH}/q_proj__L{L}__{i}.npy", f"{workdir}/q_proj__L{L}__{i}.npy")
            deltas.append(np.load(p).astype(np.float64))
        G = consensus_gram(deltas)
        out[L] = bottom_eigvecs(G, k)
    return out


def install_pg(repo_root="/opt/repo"):
    """CPU/GPU box deps for the PG1 jobs (transformers stack; NO lerobot needed on the PG1 side)."""
    import subprocess as sp
    import sys
    sp.run([sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap"], check=True)
