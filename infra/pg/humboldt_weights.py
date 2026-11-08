"""HUMBOLDT — weight-side plumbing: fp64 consensus-Gram streaming, bank I/O, S3 bank library.

Split out of humboldt_core.py so that module stays pure-numpy and unit-testable on the control
plane. Everything here touches torch / transformers / S3.

Precision (journal C3): the Gram is accumulated in fp64 straight from the fp32 checkpoints. The
cached invariant_core scratch is fp16 and is used ONLY by the G2a bit-determinism gate.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

from infra.aws_pg.jobs.invariant_core import (       # REUSED verbatim (§13 ledger row 1)
    PG1_REPO, _load_decoder, _decoder_matrices, _ftype, _delete_cache,
)
from infra.pg.humboldt_core import IN_DIM

BUCKET = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
BANKS_S3 = f"s3://{BUCKET}/humboldt/banks"


def ft_cloud(api):
    """The 24 official PG1 -224 fine-tunes, enumerated by the SAME mechanical rule as
    invariant_core.py (identical selection ⇒ the consensus object is the same object)."""
    repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                    if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                        and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    assert len(repos) >= 2, f"ft cloud degenerate: {repos}"
    return repos


def matrix_key(t: str, layer: int) -> str:
    sub = "self_attn" if t.startswith(("q", "k", "v", "o")) else "mlp"
    return f"layers.{int(layer)}.{sub}.{t}.weight"


def stream_consensus_grams(repos, shas, base_sha, types, layers, progress_cb=None):
    """G[(type, layer)] = Σ_i ΔW_iᵀΔW_i / ‖ΔW_i‖²_F over the fine-tune cloud, fp64, STREAMING.

    One pass over the cloud: load ft → difference against base in fp64 → accumulate → delete the HF
    snapshot. No fp16 scratch anywhere (journal C3).

    Returns (G, W0, fro2_by):
      G       {(type, layer): n×n fp64}                 the consensus Gram
      W0      {(type, layer): m×n fp32 torch}           the base matrices (bottomσ twin + G1 + A2).
                                                        VIEWS into the loaded base state dict, not
                                                        clones — cloning would double an 8 GB hold
                                                        for no benefit.
      fro2_by {(type, layer): {repo: ‖ΔW‖²_F}}          per-fine-tune write energy (free to record)

    Full ΔW singular spectra are deliberately NOT computed here: A1 does not ask for them (§3/A1
    records the GRAM spectrum), and 3024 economy SVDs of 16384×2048 matrices would add ~1.5 CPU-h
    for a statistic no arm consumes. External-model ΔW spectra ARE recorded, in A4, as specified.

    Memory: Σ n² × 8 bytes over the requested cells (6 residual types × 18 × 2048² = 3.6 GB;
    down_proj × 18 × 16384² = 38.7 GB → down_proj needs r7i.4xlarge, §0.2.6's stated escape)."""
    import gc
    import numpy as np
    import torch

    types = tuple(types)
    layers = [int(L) for L in layers]
    keys = [(t, L) for t in types for L in layers]

    w_base_full = _load_decoder(PG1_REPO, base_sha)
    W0 = {}
    for t, L in keys:
        W = w_base_full[matrix_key(t, L)]
        assert W.ndim == 2 and W.shape[1] == IN_DIM[t], \
            f"{matrix_key(t, L)} shape {tuple(W.shape)} — expected input dim {IN_DIM[t]}"
        W0[(t, L)] = W

    G = {k: np.zeros((IN_DIM[k[0]], IN_DIM[k[0]]), dtype=np.float64) for k in keys}
    n_terms = {k: 0 for k in keys}
    fro2_by = {k: {} for k in keys}

    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[hum][gram] {repo_id}@{sha[:8]} ({ridx + 1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base_full, w_ft)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            L = int(key.split("layers.")[1].split(".")[0])
            if (t, L) not in G:
                continue
            A = (wf.to(torch.float64) - wb.to(torch.float64)).numpy()
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"zero ΔW {key} in {repo_id} — a frozen ft slipped the scale gate"
            G[(t, L)] += (A.T @ A) / fro2
            fro2_by[(t, L)][repo_id] = fro2
            n_terms[(t, L)] += 1
            del A
        del w_ft, matched
        # 24 checkpoints × ~12 GB pass through this loop while up to 42 GB of Grams stay resident.
        # Refcount alone frees them, but a delayed collection here costs the whole run, so collect
        # explicitly rather than hope.
        gc.collect()
        _delete_cache(repo_id)
        if progress_cb is not None:
            progress_cb(ridx + 1, len(repos))

    for k, nt in n_terms.items():
        assert nt == len(repos), f"{k} accumulated {nt} terms, expected {len(repos)}"
    for k, g in G.items():
        tr = float(np.trace(g))
        assert abs(tr - len(repos)) < 1e-6 * len(repos), \
            f"{k}: trace(G)={tr} != N={len(repos)} (unit-Frobenius normalisation broken)"
    return G, W0, fro2_by


def eig_ascending(G):
    """fp64 symmetric eigendecomposition, ASCENDING. Returns (Λ, V). Tiny negative eigenvalues are
    fp roundoff on a PSD matrix and are clipped to 0 — the same treatment invariant_core applies."""
    import numpy as np
    w, V = np.linalg.eigh(np.asarray(G, dtype=np.float64))
    return np.clip(w, 0.0, None), V


# ---------------------------------------------------------------------------
# Bank library on S3 (out/banks/ → s3://…/humboldt/banks/). Downstream arms load, never rebuild.
# ---------------------------------------------------------------------------
def bank_name(t: str, layer: int, bank_id: str) -> str:
    return f"{t}__L{int(layer)}__{bank_id}.npy"


def save_banks(out_dir, t: str, layer: int, banks: dict):
    import numpy as np
    d = pathlib.Path(out_dir) / "banks"
    d.mkdir(parents=True, exist_ok=True)
    for bid, U in banks.items():
        np.save(d / bank_name(t, layer, bid), np.asarray(U, dtype=np.float64))


def publish_banks(out_dir):
    subprocess.run(["aws", "s3", "sync", str(pathlib.Path(out_dir) / "banks"), BANKS_S3,
                    "--region", "us-east-1"], check=True)


def fetch_banks(dest):
    """Pull the whole bank library from S3 into `dest`. Fails loud if it is empty."""
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", BANKS_S3, str(d), "--region", "us-east-1"], check=True)
    n = len(list(d.glob("*.npy")))
    assert n > 0, f"bank library empty at {BANKS_S3} — A1 must run first"
    print(f"[hum] fetched {n} banks from {BANKS_S3}", flush=True)
    return d


def load_bank(bank_dir, t: str, layer: int, bank_id: str):
    import numpy as np
    p = pathlib.Path(bank_dir) / bank_name(t, layer, bank_id)
    assert p.exists(), f"missing bank {p}"
    U = np.load(p)
    assert U.shape == (IN_DIM[t], 32), f"{p.name} shape {U.shape}"
    return U


def load_battery(bank_dir, t: str, layer: int) -> dict:
    from infra.pg.humboldt_core import BANK_IDS
    return {bid: load_bank(bank_dir, t, layer, bid) for bid in BANK_IDS}
