"""M-B1 — the OUTPUT-SIDE census instrument (queue item B1).

Every measurement in HUMBOLDT/DARWIN is input-space: the consensus Gram is ΔWᵀΔW, an n×n object
on the matrix's INPUT coordinates, and every bank is a subspace of the residual stream a matrix
READS. The row-space consensus ΔWΔWᵀ — an m×m object on the coordinates a matrix WRITES — has
never been computed. This module is that half.

The two objects are genuinely different and must never be conflated:

    G_in  = Σ_i ΔW_iᵀ ΔW_i / ‖ΔW_i‖²_F     (n×n)   which INPUT directions the cloud avoids
    G_out = Σ_i ΔW_i ΔW_iᵀ / ‖ΔW_i‖²_F     (m×m)   which OUTPUT directions the cloud avoids

Their eigenvalue SPECTRA are identical up to |m−n| zeros (both are the squared singular values of
the same stack), so a "the spectrum looks the same" observation is a tautology, not a finding. What
is NOT determined by the input side is WHERE the small directions sit in the output coordinates,
how they distribute leverage over rows, and whether they line up across layers — which is what this
instrument measures.

Normalisation, precision, and the streaming/delete-cache pattern are HUMBOLDT's verbatim: unit
Frobenius per fine-tune, fp64 accumulation straight from fp32 checkpoints, one checkpoint resident
at a time. Only the side of the product changes.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import pathlib

from infra.aws_pg.jobs.invariant_core import (       # REUSED verbatim (§11 reuse ledger)
    PG1_REPO, _load_decoder, _decoder_matrices, _ftype, _delete_cache,
)
from infra.pg import humboldt_core as H
from infra.pg.humboldt_weights import matrix_key

SIDE = "row"                     # the flag every M-B1 atlas row carries
BANK_K = H.BANK_K


def out_dim(W0_by_key: dict) -> dict:
    """m per (type, layer), read from the BASE matrices rather than hardcoded.

    Hardcoding would encode PG1's MQA shape (k_proj/v_proj are 256×2048) into a module that also
    has to run on foreign bases in M-9b, and a wrong m does not fail loudly — it fails as a
    silently-truncated Gram.
    """
    return {k: int(W.shape[0]) for k, W in W0_by_key.items()}


DOWNLOAD_ATTEMPTS = 4


def _load_decoder_retrying(repo_id: str, sha: str, attempts: int = DOWNLOAD_ATTEMPTS):
    """Load one checkpoint, retrying a TRANSPORT failure a bounded number of times.

    This is the one guarded boundary in this module and it is a SYSTEM boundary, not a scientific
    one: a 24-checkpoint stream moves hundreds of gigabytes from the Hub, and a mid-file
    `RemoteProtocolError` ("peer closed connection without sending complete message body") killed a
    2-hour run after five checkpoints. Everything else here still fails loud.

    The retry is BOUNDED and COUNTED. Only transport errors are caught — a missing repo, a gated
    repo or a shape mismatch is a real failure and still propagates on the first occurrence — and
    the number of retries is returned so a flaky run is visible in the manifest rather than
    indistinguishable from a clean one.
    """
    import time
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    transient = (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                 httpx.ReadTimeout, httpx.WriteError, HfHubHTTPError, ConnectionError, OSError)
    last = None
    for i in range(attempts):
        try:
            return _load_decoder(repo_id, sha), i
        except transient as e:            # noqa: PERF203 - a bounded transport retry, counted
            last = e
            print(f"[rowgram] transport failure on {repo_id} (attempt {i + 1}/{attempts}): "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            _delete_cache(repo_id)        # a partial snapshot must not be reused
            time.sleep(5 * (i + 1))
    raise RuntimeError(
        f"{repo_id}: {attempts} transport failures loading the checkpoint; last was "
        f"{type(last).__name__}: {last}") from last


def stream_row_grams(repos, shas, base_sha, types, layers, progress_cb=None):
    """G_out[(type, layer)] = Σ_i ΔW_i ΔW_iᵀ / ‖ΔW_i‖²_F over the fine-tune cloud, fp64, STREAMING.

    Returns (G, W0, fro2_by) with the same shapes/keys contract as
    `humboldt_weights.stream_consensus_grams`, except that G is m×m.

    Memory: Σ m² × 8 bytes over the requested cells. For PG1 that is 1.8 GB for
    {q,k,v,o,down}_proj × 18 layers, but 38.7 GB for gate_proj × 18 alone (m = 16384) — so the
    caller partitions the type list across jobs. `assert_fits` makes that budget explicit instead
    of discovering it as an OOM four checkpoints in.
    """
    import gc
    import numpy as np
    import torch

    types = tuple(types)
    layers = [int(L) for L in layers]
    keys = [(t, L) for t in types for L in layers]

    w_base_full, base_retries = _load_decoder_retrying(PG1_REPO, base_sha)
    n_retries = {"base": base_retries}
    W0 = {}
    for t, L in keys:
        W = w_base_full[matrix_key(t, L)]
        assert W.ndim == 2, f"{matrix_key(t, L)} is not a matrix: {tuple(W.shape)}"
        W0[(t, L)] = W

    M = out_dim(W0)
    G = {k: np.zeros((M[k], M[k]), dtype=np.float64) for k in keys}
    n_terms = {k: 0 for k in keys}
    fro2_by = {k: {} for k in keys}

    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[mercator][rowgram] {repo_id}@{sha[:8]} ({ridx + 1}/{len(repos)})", flush=True)
        w_ft, retries = _load_decoder_retrying(repo_id, sha)
        n_retries[repo_id] = retries
        matched = _decoder_matrices(w_base_full, w_ft)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            L = int(key.split("layers.")[1].split(".")[0])
            if (t, L) not in G:
                continue
            A = (wf.to(torch.float64) - wb.to(torch.float64)).numpy()
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"zero ΔW {key} in {repo_id} — a frozen ft slipped the scale gate"
            G[(t, L)] += (A @ A.T) / fro2
            fro2_by[(t, L)][repo_id] = fro2
            n_terms[(t, L)] += 1
            del A
        del w_ft, matched
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
    total_retries = sum(n_retries.values())
    if total_retries:
        print(f"[rowgram] completed with {total_retries} transport retries across "
              f"{sum(1 for v in n_retries.values() if v)} checkpoints", flush=True)
    return G, W0, fro2_by, n_retries


def assert_fits(types, layers, out_dims: dict, *, budget_gb: float):
    """Refuse a cell partition whose Grams cannot be held. `out_dims` maps type -> m."""
    need = len(layers) * sum(out_dims[t] ** 2 * 8 for t in types) / 1e9
    assert need <= budget_gb, (
        f"row Grams for types={list(types)} x {len(layers)} layers need {need:.1f} GB, budget is "
        f"{budget_gb} GB — partition the type list across jobs (gate_proj/up_proj are m=16384 and "
        f"take 38.7 GB per type at 18 layers)")
    return {"types": list(types), "n_layers": len(layers), "gram_gb": round(need, 2),
            "budget_gb": float(budget_gb)}


def build_row_banks_from_extremes(V_bottom, V_top, W0, k: int = BANK_K,
                                  seed_base: int = 1337) -> dict:
    """The row-side battery from the extreme eigenvectors alone (see `eig_extremes`).

    Same battery, same ids, same construction as `build_row_banks`; it simply does not require the
    interior of the spectrum, which the banks never used.
    """
    import numpy as np
    from infra.pg import darwin_core as D
    Vb = np.ascontiguousarray(np.asarray(V_bottom, dtype=np.float64))
    Vt = np.ascontiguousarray(np.asarray(V_top, dtype=np.float64))
    m = Vb.shape[0]
    A = np.asarray(W0, dtype=np.float64)
    assert A.shape[0] == m, f"W0 rows {A.shape[0]} != Gram dim {m}"
    assert Vb.shape[1] == k and Vt.shape[1] == k, "extreme blocks must have k columns"
    bots, tops, _s = H.sigma_subspaces(A.T, k)
    banks = {"core32": Vb, "anti32": Vt, "bottomsigma32": bots, "topsigma32": tops,
             "complement32": H.complement_subspace(Vb, seed_base + 900)}
    for s in H.HAAR_SEEDS:
        banks[f"haar32_s{s}"] = H.haar_subspace(m, k, seed_base + s)
    for bid, U in banks.items():
        assert U.shape == (m, k), f"{bid} shape {U.shape} != {(m, k)}"
        H.orthonormal_check(U)
    assert set(banks) == set(D.BANK_IDS), f"battery mismatch: {sorted(banks)}"
    return banks


def build_row_banks(V_gram, W0, k: int = BANK_K, seed_base: int = 1337) -> dict:
    """The full §3 battery in the ROW space of one (type, layer) matrix.

    Bank ids are the SAME literals as the input-side battery (core32, anti32, bottomsigma32,
    topsigma32, complement32, haar32_s0..2) because it IS the same battery, measured on the other
    side; the side is recorded in each atlas row's flags and in the bank filename. Keeping the ids
    identical means `check_controls`, the figure style map, and every downstream aggregator work
    unchanged — renaming them would fork eight tested call sites to say the same thing.

    Row-side σ twins are the LEFT singular vectors of W0, i.e. the right singular vectors of W0ᵀ.
    """
    import numpy as np
    m = V_gram.shape[0]
    A = np.asarray(W0, dtype=np.float64)
    assert A.shape[0] == m, f"W0 rows {A.shape[0]} != Gram dim {m}"
    core = np.ascontiguousarray(V_gram[:, :k].astype(np.float64))
    anti = np.ascontiguousarray(V_gram[:, -k:].astype(np.float64))
    bots, tops, _s = H.sigma_subspaces(A.T, k)          # A.T is n×m -> right-singular = left of A
    banks = {"core32": core, "anti32": anti, "bottomsigma32": bots, "topsigma32": tops,
             "complement32": H.complement_subspace(core, seed_base + 900)}
    for s in H.HAAR_SEEDS:
        banks[f"haar32_s{s}"] = H.haar_subspace(m, k, seed_base + s)
    for bid, U in banks.items():
        assert U.shape == (m, k), f"{bid} shape {U.shape} != {(m, k)}"
        H.orthonormal_check(U)
    from infra.pg import darwin_core as D
    assert set(banks) == set(D.BANK_IDS), f"battery mismatch: {sorted(banks)} vs {sorted(D.BANK_IDS)}"
    return banks


def row_bank_name(t: str, layer: int, bank_id: str) -> str:
    """Row-side banks live beside the input-side ones with a `row__` infix, so one directory holds
    both and neither can be loaded in place of the other by accident."""
    return f"{t}__L{int(layer)}__row__{bank_id}.npy"


def save_row_banks(out_dir, t: str, layer: int, banks: dict):
    import numpy as np
    d = pathlib.Path(out_dir) / "banks"
    d.mkdir(parents=True, exist_ok=True)
    for bid, U in banks.items():
        np.save(d / row_bank_name(t, layer, bid), np.asarray(U, dtype=np.float64))


def load_row_bank(bank_dir, t: str, layer: int, bank_id: str, *, m: int = None):
    import numpy as np
    p = pathlib.Path(bank_dir) / row_bank_name(t, layer, bank_id)
    assert p.exists(), f"missing row bank {p}"
    U = np.load(p)
    assert U.ndim == 2 and U.shape[1] == BANK_K, f"{p.name} shape {U.shape}"
    if m is not None:
        assert U.shape[0] == m, f"{p.name} ambient {U.shape[0]} != expected {m}"
    return U


# ---------------------------------------------------------------------------
# The REFUTER's decision statistic, and the smoke that proves the instrument can see a real one.
# ---------------------------------------------------------------------------
def row_avoidance(G, k: int = BANK_K) -> dict:
    """The output-side analogue of the avoidance statistic: how much of the cloud's write energy
    lands in the bottom-k row directions, against the flat expectation k/m.

    trace(G) is N by construction (unit-Frobenius per fine-tune), so the bottom-k eigenvalue sum
    divided by trace is the ENERGY FRACTION in those directions, and k/m is what a rotationally
    invariant cloud would put there. `ratio` < 1 is avoidance. Returned, never judged.
    """
    lam, _V = _eig_ascending(G)
    m = lam.size
    tot = float(lam.sum())
    assert tot > 0, "degenerate row Gram (zero trace)"
    frac = float(lam[:k].sum() / tot)
    flat = float(k) / float(m)
    return {"bottom_k_energy_fraction": frac, "flat_expectation": flat,
            "ratio": frac / flat, "m": int(m), "k": int(k), "trace": tot}


def _eig_ascending(G):
    import numpy as np
    w, V = np.linalg.eigh(np.asarray(G, dtype=np.float64))
    return np.clip(w, 0.0, None), V


# THE DENSE PATH IS THE ONLY ONE THAT WORKS HERE, and two measured attempts say so.
#
# The row Grams for gate_proj/up_proj are 16384 x 16384 and a full `numpy.linalg.eigh` takes on the
# order of ten minutes each on this CPU box — eighteen layers per type is hours. Two attempts to
# avoid that both failed, and both are recorded because the second looked obviously right:
#
#   1. LAPACK's index-SUBSET driver (`scipy.linalg.eigh(subset_by_index=...)`). Measured at ~9
#      minutes per cell, barely better. The cubic term is the Householder TRIDIAGONALISATION, which
#      a subset request does not avoid; only the eigenvector back-transform shrinks.
#   2. LANCZOS on the shifted operator cI - A, to get the small end from a large-end solve. This is
#      WORSE, not better, and the reason is structural: a PSD consensus Gram has most of its
#      eigenvalues near zero, so the shift maps them to a dense CLUSTER at the top of the shifted
#      spectrum — precisely the case where Lanczos converges slowest. Measured: one cell had not
#      finished after roughly four hours.
#
# So the bottom-32 of a dense 16384-wide PSD matrix costs O(n^3), and the budget is sized to it
# rather than the algorithm being sized to the budget. The threshold below therefore admits every
# width this program uses; `eig_extremes` keeps its interface so the callers and the
# dense-equivalence test are unchanged.
FULL_EIGH_MAX_DIM = 65536

# Retained for the record and for any caller that has a genuinely well-separated small end.
KRYLOV_TOL = 1e-9


def eig_extremes(G, k: int, force_krylov: bool = False):
    """The k smallest and k largest eigenpairs, without computing the whole spectrum.

    A full eigendecomposition is O(n^3) and the banks only ever use 2k of the n eigenvectors, so
    for the wide types LAPACK's index-subset driver does the same job in a fraction of the time.
    Returns (lam_bottom, V_bottom, lam_top, V_top, trace, full_spectrum_or_None).

    `trace` is taken from the DIAGONAL, which is exact and O(n) — the avoidance statistic needs the
    total energy, and taking it from a spectrum that was never computed would be the only reason to
    compute one. The full spectrum is returned when it was affordable and None otherwise, so a
    downstream consumer can see that it is genuinely absent rather than silently receiving a
    truncated one.
    """
    import numpy as np
    A = np.asarray(G, dtype=np.float64)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], f"row Gram must be square, got {A.shape}"
    assert 0 < k <= n // 2, f"k={k} out of range for n={n}"
    tr = float(np.trace(A))
    if n <= FULL_EIGH_MAX_DIM and not force_krylov:
        lam, V = _eig_ascending(A)
        return lam[:k], V[:, :k], lam[-k:], V[:, -k:], tr, lam
    # KRYLOV, NOT A DENSE SUBSET DRIVER. LAPACK's index-subset routines still perform the full
    # Householder tridiagonalisation — the O(n^3) part — and only shorten the back-transform, so
    # subsetting a 16384-wide Gram bought almost nothing (measured: ~9 minutes per cell, against
    # the same order for the full decomposition). A Lanczos solver touches the matrix only through
    # matrix-vector products, so 2k extremes cost O(n^2 * iterations) instead.
    #
    # The SMALL end is obtained from the LARGE end of (cI - A) with c an upper bound on the
    # spectrum: converging on the algebraically largest eigenvalues is what Lanczos does well, and
    # the small end of a PSD matrix is where it does worst. trace(A) bounds the spectrum for a PSD
    # matrix, so it is the shift, and the eigenvalues map back as lambda = c - mu exactly.
    from scipy.sparse.linalg import LinearOperator, eigsh
    # Lanczos on a 16384-wide Gram is MEMORY-BANDWIDTH bound: each matvec streams the whole 2.1 GB
    # matrix, so the cost is (number of matvecs) x (matrix size), and the two knobs that matter are
    # the Krylov width and the tolerance. A wide basis costs a full pass per restart vector, and
    # machine-precision convergence buys accuracy the banks cannot use — an eigenvector good to
    # 1e-9 and one good to 1e-15 give the same subspace to every statistic in this program. Both
    # are recorded rather than tuned silently, and the equivalence test still compares against the
    # dense decomposition.
    ncv = min(n, 2 * k + 8)
    mu_t, Vt = eigsh(A, k=k, which="LA", ncv=ncv, tol=KRYLOV_TOL)
    C = tr
    # A LinearOperator rather than materialising C*I - A: that matrix is another n^2 fp64 array
    # (2.1 GB at n=16384) and Lanczos only ever needs its action on a vector.
    shifted = LinearOperator((n, n), dtype=np.float64, matvec=lambda x: C * x - A @ x,
                             rmatvec=lambda x: C * x - A @ x)
    mu_b, Vb = eigsh(shifted, k=k, which="LA", ncv=ncv, tol=KRYLOV_TOL)
    lb = C - mu_b
    order_b, order_t = np.argsort(lb), np.argsort(mu_t)
    return (np.clip(lb[order_b], 0.0, None), np.ascontiguousarray(Vb[:, order_b]),
            np.clip(mu_t[order_t], 0.0, None), np.ascontiguousarray(Vt[:, order_t]), tr, None)


def row_avoidance_from_extremes(lam_bottom, trace: float, m: int, k: int) -> dict:
    """The avoidance statistic from the bottom-k eigenvalues and the trace alone.

    Identical to `row_avoidance` when the full spectrum is available; this form exists so the wide
    types can report the same number without an eigendecomposition they cannot afford.
    """
    import numpy as np
    lb = np.asarray(lam_bottom, dtype=np.float64)
    assert lb.size == k, f"expected {k} bottom eigenvalues, got {lb.size}"
    assert trace > 0, "degenerate row Gram (zero trace)"
    frac = float(lb.sum() / trace)
    flat = float(k) / float(m)
    return {"bottom_k_energy_fraction": frac, "flat_expectation": flat,
            "ratio": frac / flat, "m": int(m), "k": int(k), "trace": float(trace)}


def planted_rowspace_null(m: int = 64, n: int = 96, n_ft: int = 12, k: int = 8,
                          suppression: float = 0.02, seed: int = 1337) -> dict:
    """SMOKE (§4 M-B1: 'Smoke: planted row-space null').

    Builds a synthetic cloud whose row space has a KNOWN suppressed k-block, runs the real
    estimator on it, and reports how much of the planted block the recovered bottom-k subspace
    captures. Two arms, because a smoke that only shows the planted case passes trivially:

      * `planted`  — deltas are Q diag(d) R_i with d small on a fixed k-block: the estimator must
                     recover that block (capture -> 1).
      * `null`     — deltas are isotropic Gaussian: the estimator must NOT recover any block
                     (capture -> the chance level k/m).

    Returns both captures plus the chance level. The caller gates on them; this function judges
    nothing.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((m, m)))[0]
    planted_idx = np.arange(k)
    d = np.ones(m)
    d[planted_idx] = suppression

    G_planted = np.zeros((m, m))
    G_null = np.zeros((m, m))
    for _ in range(n_ft):
        R = rng.standard_normal((m, n))
        A = Q @ (d[:, None] * R)
        G_planted += (A @ A.T) / float((A ** 2).sum())
        B = rng.standard_normal((m, n))
        G_null += (B @ B.T) / float((B ** 2).sum())

    U_true = Q[:, planted_idx]
    out = {}
    for name, G in (("planted", G_planted), ("null", G_null)):
        _lam, V = _eig_ascending(G)
        U_hat = V[:, :k]
        # captured fraction of the planted subspace inside the recovered one, in [0, 1]:
        # ‖U_hatᵀ U_true‖²_F / k, which is k for a perfect match and k²/m at chance.
        out[f"{name}_capture"] = float(np.sum((U_hat.T @ U_true) ** 2) / k)
    out.update({"chance_capture": float(k) / float(m), "m": m, "n": n, "k": k,
                "n_ft": n_ft, "suppression": suppression, "seed": seed})
    return out
