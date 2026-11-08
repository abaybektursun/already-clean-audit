"""BISPEC core — the bi-spectral susceptibility surface: pure injectors + pure math.

The object under measurement is the behavioural susceptibility functional

    B_r(δ) = (readout r on the clean forward) − (readout r on the forward with x ← x + δ at site L)

REGRAIN R10 established that B_r is NOT a function of ‖δ‖ or of the realized reconstruction R².
BISPEC's hypothesis is that it is (to leading order) a function of two coordinates:

  * the FEATURE coordinate  — which Σ-eigendirections carry δ's power   (band b)
  * the SEQUENCE coordinate — whether δ is coherent or independent      (mode ω)
                              across token positions

Everything here is weight-free and unit-testable. The hooks are shape-aware: unlike the REGRAIN
family (which flattens (B,T,D) → (B·T,D) and therefore cannot express a sequence mode at all),
every injector below sees the token axis and tracks ABSOLUTE POSITION across the prefill/decode
boundary, so a "coherent" perturbation stays coherent through greedy generation.

--------------------------------------------------------------------------------------------
The cancellation law, derived here (spec §0.5 A3; the cited synthesis parent does not exist)
--------------------------------------------------------------------------------------------
Write the perturbation at site L as δ_t at token t, with matched per-token second moment
E‖δ_t‖² = D·σ² for every mode (so per-token fidelity, hence per-token R², is matched by
construction — this is the whole point of the contrast).

A downstream attention head at query position q reads a convex combination of value vectors,
v_out = Σ_t w_qt · W_V (x_t + δ_t) with Σ_t w_qt = 1, w_qt ≥ 0. The perturbation it sees is

    Δv_out = W_V ( Σ_t w_qt δ_t ).

  * COHERENT (δ_t ≡ δ, one draw broadcast to all positions):
        Σ_t w_qt δ_t = δ                            ⇒  E‖Δv_out‖² ∝ D·σ²          (no attenuation)

  * INDEPENDENT (δ_t iid, mean zero):
        E‖Σ_t w_qt δ_t‖² = (Σ_t w_qt²)·D·σ² = D·σ² / n_eff,   n_eff := 1/Σ_t w_qt²

    i.e. the classical inverse-participation-ratio effective sample size of the attention row.

So the ROUTED component of the perturbation has its VARIANCE attenuated by 1/n_eff, equivalently
its AMPLITUDE attenuated by 1/√n_eff. Both statements appear in the spec (§1.1 says "1/n_eff",
F-K2/B2 expect a log–log slope of −1/2); they are the same law read on the two different scales.
`fit_cancellation_law` below reports the AMPLITUDE exponent p, so the ideal-pool value is p = 1/2.

Interpolating with coherence group size g (positions partitioned into blocks of g sharing a draw)
gives, for an attention row spread over ≫ g positions, Σ over blocks of (block weight)² · D σ²
≈ D σ² · g / n_eff — the routed amplitude falls like g^(−1/2) as g → 1.

--------------------------------------------------------------------------------------------
Why a bare power-law fit is misspecified (spec §0.5 A4)
--------------------------------------------------------------------------------------------
The injection is a forward_pre_hook on layers[L], so the residual stream carries δ_t forward on
the SKIP CONNECTION directly into the readout at position t. That direct path is g-INDEPENDENT
(per-token energy is matched across every mode by construction). Hence

    damage(g) = a + b · g^(−p)                     a = direct-path floor,  b·g^(−p) = routed part

Fitting a bare power law to a curve with a > 0 biases the estimated slope toward 0 and would be
mis-read as "flatter than −1/2 ⇒ structured routing". `fit_cancellation_law` fits both terms.

NOTE the corollary that makes P1 (g=1 vs g=all) the clean cell: the direct term is IDENTICAL in
distribution for shared and iid draws, so the entire shared-vs-iid gap is routed. P1 is
identified without knowing `a` at all.
"""
import contextlib
import math

import torch

from infra.pg.inherit_core import _split_hidden

# ---------------------------------------------------------------------------
# Frozen constants (spec §2 / §0.5).
# ---------------------------------------------------------------------------
BAND_EDGES = (0.50, 0.80, 0.95, 0.99, 1.00)   # cumulative-energy quantiles → 5 feature bands
LEAK_TOL = 0.01                                # band purity assert: out-of-band energy < 1%
BOS_EXCLUDED_FROM_SIGMA = True                 # A2: BOS carries ~10× content-token norm (REGRAIN R16)


# ===========================================================================
# Σ — the site's activation covariance, its eigenbasis, and the frozen bands.
# ===========================================================================
def accumulate_sigma(X, centered=True, weights=None):
    """Σ at a site from banked activation rows X (n,d).

    A2 (frozen): `centered=True` is the primary convention — it matches the house mean-ablation
    null `x − QQᵀ(x − h̄)` (ctrliface_core.apply_mean_ablation), which removes the CENTERED
    component. `centered=False` (the uncentered second moment) is computed as the frozen
    sensitivity row: there the top eigendirection ≈ the mean direction, so the (top band × DC)
    cell of the surface would be confounded with Arm A's mean-shift mechanism.

    Returns (Sigma (d,d) fp64, hbar (d,) fp64, n_rows).
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    n = X.shape[0]
    assert n > X.shape[1], f"need n > d for a non-degenerate Σ: n={n} d={X.shape[1]}"
    hbar = X.mean(0)
    Z = X - hbar if centered else X
    if weights is not None:
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1, 1)
        assert w.shape[0] == n
        Sigma = (Z * w).transpose(0, 1) @ Z / float(w.sum())
    else:
        Sigma = Z.transpose(0, 1) @ Z / float(n)
    return Sigma, hbar, n


def eig_bands(Sigma, edges=BAND_EDGES):
    """Eigendecompose Σ (fp64, descending) and cut the spectrum at cumulative-energy quantiles.

    Bands are defined by ENERGY quantiles, not by index — the pre-mortem (vi) requirement: the
    tail of the spectrum is the noisiest part of the estimate, and an index cut would put its
    noise directly on the band boundary.

    Returns dict(basis (d,d) columns=eigvecs desc, spectrum (d,), cum (d,), band_slices list of
    (lo, hi) index pairs, band_energy list of realized energy fractions).
    """
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    Sigma = 0.5 * (Sigma + Sigma.transpose(0, 1))          # symmetrize away fp asymmetry
    evals, evecs = torch.linalg.eigh(Sigma)                # ascending
    idx = torch.argsort(evals, descending=True)
    evals, evecs = evals[idx], evecs[:, idx]
    assert float(evals[-1]) > -1e-8 * float(evals[0]), "Σ not PSD — check the accumulator"
    evals = evals.clamp_min(0.0)
    cum = torch.cumsum(evals, 0) / evals.sum()
    slices, band_energy, lo = [], [], 0
    for e in edges:
        hi = int(torch.searchsorted(cum, torch.tensor(float(e), dtype=torch.float64)).item()) + 1
        hi = min(max(hi, lo + 1), evals.shape[0])
        slices.append((lo, hi))
        band_energy.append(float(cum[hi - 1] - (cum[lo - 1] if lo > 0 else 0.0)))
        lo = hi
        if lo >= evals.shape[0]:
            break
    return {"basis": evecs, "spectrum": evals, "cum": cum,
            "band_slices": slices, "band_energy": band_energy, "edges": list(edges)}


def band_projector(bands, b):
    """Orthonormal basis (d,k) of feature band b."""
    lo, hi = bands["band_slices"][b]
    return bands["basis"][:, lo:hi].contiguous()


def band_purity(delta, U):
    """Realized fraction of δ's energy inside span(U). The §3 injector assert (leak < 1%).

    Reported per cell as the REALIZED coordinate — never the nominal one (spec §10).
    """
    delta = torch.as_tensor(delta, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64)
    tot = float((delta ** 2).sum())
    assert tot > 0, "degenerate δ"
    inb = float(((delta @ U) ** 2).sum())
    return inb / tot


# ===========================================================================
# Sequence-mode envelopes (the ω axis, ADD side — spec §0.5 A5).
# ===========================================================================
def envelope(mode, positions, freq=None):
    """c_t for absolute positions `positions` (long tensor). Unit mean-square by construction, so
    every mode carries the SAME per-token expected energy — the matched-fidelity requirement.

      dc      c_t = 1                       (fully coherent along the sequence)
      sin_k   c_t = √2 · cos(2π k t / T0)   (a low-frequency envelope; T0 = 512, the ppl window)
      white   c_t ~ N(0,1) iid              (fully incoherent)

    NOTE (A5): this ω is the envelope of an ADDED perturbation. The DELETE side has no such
    object — a projection is a per-token linear map. `deletion_envelope` below defines the
    delete-side ω separately (per-position removal FRACTION) and the two are never compared
    across signs by a single overlap coefficient.
    """
    t = torch.as_tensor(positions, dtype=torch.float64)
    if mode == "dc":
        return torch.ones_like(t)
    if mode.startswith("sin"):
        assert freq is not None, "sinusoidal envelope needs freq"
        return math.sqrt(2.0) * torch.cos(2.0 * math.pi * float(freq) * t / 512.0)
    if mode == "white":
        raise ValueError("white envelope is drawn per-call inside the hook (needs a generator)")
    raise ValueError(f"unknown sequence mode {mode!r}")


DEL_MS_TARGET = 0.25       # frozen mean-square of the delete-side envelope in the ω_del contrast


def deletion_envelope(mode, positions, freq=None, seed=0, ms_target=DEL_MS_TARGET):
    """The DELETE-side ω: a per-position removal FRACTION c_t ∈ [0, 1].

    A5 asked for the modulated delete modes to be matched on mean-square to `dc`. That is
    **unsatisfiable** as originally written: a removal fraction is bounded above by 1, so no
    modulated envelope can reach the mean-square of `c ≡ 1` (full removal). The frozen resolution:

      "full"   c_t = 1              — the canonical band deletion. NOT part of the ω_del contrast;
                                      reported as its own row, at its own removed energy.
      "dc"     c_t = √ms_target     — a CONSTANT partial removal at the frozen mean-square.
      "sin"    a low-frequency envelope, rescaled to `ms_target`, clamped to [0,1].
      "white"  an iid per-position fraction (|N(0,1)| keyed to the ABSOLUTE position, so it is
               stable across the prefill/decode boundary), rescaled to `ms_target`, clamped.

    So `dc`, `sin` and `white` share a mean-square by construction and differ only in how the
    removal is distributed across positions — which is exactly the ω_del coordinate. The REALIZED
    removed energy is measured per cell and reported; it is never assumed to be matched, because
    clamping can shave the tail of the white envelope.
    """
    n = len(positions)
    if mode == "full":
        return torch.ones(n, dtype=torch.float64)
    if mode == "dc":
        return torch.full((n,), math.sqrt(ms_target), dtype=torch.float64)
    if mode.startswith("sin"):
        assert freq is not None, "sinusoidal delete envelope needs freq"
        t = torch.as_tensor(positions, dtype=torch.float64)
        c = 0.5 * (1.0 + torch.cos(2.0 * math.pi * float(freq) * t / 512.0))
    elif mode == "white":
        c = torch.stack([
            torch.randn(1, generator=torch.Generator(device="cpu").manual_seed(
                int(seed) * 1_000_003 + int(p)), dtype=torch.float64)[0].abs()
            for p in torch.as_tensor(positions).tolist()])
    else:
        raise ValueError(f"unknown delete-side mode {mode!r}")
    ms = float((c ** 2).mean())
    assert ms > 0, "degenerate delete envelope"
    c = c * math.sqrt(ms_target / ms)
    return c.clamp(0.0, 1.0)


# ===========================================================================
# Position-tracking hook base — the prefill/decode boundary.
# ===========================================================================
class _PositionTracker:
    """Absolute token positions across a generate() call.

    A decoder-layer pre-hook fires once with the full prefill (T>1) and then once per decode step
    (T=1) with the KV cache carrying the past. `positions(T)` returns the absolute indices for the
    current call and advances the counter, so an envelope or a per-token draw stays keyed to the
    SAME token throughout generation. Without this a "coherent" perturbation silently decoheres at
    the first decode step.
    """

    def __init__(self):
        self.n_seen = 0

    def positions(self, T):
        if T > 1:                       # prefill — a new sequence starts here
            self.n_seen = T
            return torch.arange(0, T, dtype=torch.long)
        p = torch.arange(self.n_seen, self.n_seen + 1, dtype=torch.long)
        self.n_seen += 1
        return p

    def reset(self):
        self.n_seen = 0


# ===========================================================================
# Arm K — the coherence trio.
# ===========================================================================
def make_hook_gaussian_group(sigma, seed, g, d):
    """P1 / P1g: Gaussian noise with COHERENCE GROUP SIZE g.

      g = 1     → iid per token                       (fully independent; the REGRAIN cell)
      g = None  → one draw broadcast to every token   (fully coherent; "shared")
      g = k     → positions partitioned into blocks of k, one draw shared inside each block

    Per-token second moment is D·σ² for EVERY g — matched per-token fidelity by construction, so
    the realized per-token R² is matched and the entire damage difference is routed (see module
    docstring). The draw is keyed to the ABSOLUTE block index, so it is stable across the
    prefill/decode boundary and identical for the same block in every batch.
    """
    trk = _PositionTracker()
    cache = {}

    def _draws_for_blocks(uniq, device):
        """One randn per DISTINCT block id, cached across forwards so a coherent draw stays the
        same vector for the whole run (and across the prefill/decode boundary)."""
        out = []
        for u in uniq:
            if u not in cache:
                gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(u))
                cache[u] = float(sigma) * torch.randn(d, generator=gen, dtype=torch.float32)
            out.append(cache[u])
        return torch.stack(out).to(device)

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        if float(sigma) == 0.0:
            return None                                   # exact no-op: the identity anchor
        pos = trk.positions(T)
        block = torch.zeros_like(pos) if g is None else (pos // int(g))
        uniq, inv = torch.unique(block, return_inverse=True)
        draws = _draws_for_blocks(uniq.tolist(), hs.device)      # (n_uniq, D)
        delta = draws[inv.to(hs.device)]                          # (T, D) — one gather, no loop
        return rebuild((hs.float() + delta.unsqueeze(0)).to(hs.dtype))

    hook.tracker = trk
    return hook


def quant_error(x, step):
    """e(x) = Δ·round(x/Δ) − x — the per-coordinate sawtooth error of App. G.1 quantization."""
    assert float(step) > 0
    return float(step) * torch.round(x / float(step)) - x


def make_hook_quant_permuted(step, seed, permute):
    """P2: quantization error, decohered.

      permute=False → x' = x + e(x)                    (the original corruption, identically)
      permute=True  → x' = x + Π e(x)                  (Π permutes the error rows across POSITIONS)

    Π preserves the error's MARGINAL distribution over tokens exactly (it is a relabelling of
    rows) and destroys its alignment with the token that produced it. So the pair isolates
    sequence-coherence at *identical* marginals — the P2 matched twin.

    Only permutes within a prefill (T>1); a decode step (T=1) has nothing to permute against and
    receives its own un-permuted error. Reported as the realized permuted fraction.
    """
    trk = _PositionTracker()

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        trk.positions(T)
        x = hs.float()
        e = quant_error(x, step)
        if permute and T > 1:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            perm = torch.randperm(T, generator=gen).to(hs.device)
            e = e[:, perm, :]
        return rebuild((x + e).to(hs.dtype))

    hook.tracker = trk
    return hook


def pertoken_subspace_pool(d, k, n_pool, seed, device="cuda", dtype=torch.float32):
    """A pool of `n_pool` independent orthonormal rank-k frames, built ONCE with a batched QR.

    Why a pool. The faithful construction — a fresh QR per token per forward — costs ~60 000 QRs of
    (2048×128) on the 120-window ppl pass and never finishes. The pool is precomputed on device in
    one batched call and indexed by absolute position, so position p uses frame `p mod n_pool`.

    DISCLOSURE (frozen): the per-token subspaces are independent across any window shorter than
    `n_pool`, and repeat with period `n_pool` beyond it. Since the contrast P3 draws is between
    *adjacent-token independence* and *whole-sequence sharing*, a period of 512 is
    indistinguishable from true per-token independence for the sequence lengths this program uses
    (VQA prompts ≈ 271 tokens; ppl windows ≤ 512). `n_pool` is reported per cell.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(n_pool, d, k, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A.to(device))
    return Q.to(dtype)


def make_hook_subspace_removal(U_shared, hbar, per_token, seed, d, k, pool=None):
    """P3: rank-k subspace removal, decohered.

      per_token=False → one shared random subspace, removed at every position  (the classical
                        em-null: x − QQᵀ(x − h̄))
      per_token=True  → an INDEPENDENT random rank-k subspace per token, drawn from `pool`

    Matched rank by construction; the realized per-token removed energy is measured and reported
    (it is matched in expectation, not exactly — a random k-subspace captures k/d of the trace in
    expectation regardless of which one it is).

    If the per-token nulls are far weaker than the shared one at matched rank+energy, then every
    em-band this program has ever drawn is inflated by coherence, and the correction factor is
    measurable (spec §1.2, Arm K row 3).
    """
    trk = _PositionTracker()
    if per_token:
        assert pool is not None, "per-token removal needs a precomputed subspace pool"

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        x = hs.float()
        hb = hbar.to(x.device, x.dtype)
        z = x - hb
        if not per_token:
            Q = U_shared.to(x.device, x.dtype)
            out = x - (z @ Q) @ Q.transpose(0, 1)
        else:
            P = pool.to(x.device, x.dtype)                       # (n_pool, D, k)
            Qt = P[(pos % P.shape[0]).to(x.device)]              # (T, D, k)
            coeff = torch.einsum("btd,tdk->btk", z, Qt)
            out = x - torch.einsum("btk,tdk->btd", coeff, Qt)
        return rebuild(out.to(hs.dtype))

    hook.tracker = trk
    return hook


# ===========================================================================
# Arm S — the bi-spectral cells.
# ===========================================================================
def make_hook_band_noise(U_band, sigma, seed, mode, freq=None, d_total=None):
    """ADD side: δ_t = c_t · σ√d · U_band ĝ_t, power confined to feature band span(U_band) and the
    sequence mode `mode` ∈ {dc, sin, white}.

      dc     → one coefficient draw g, broadcast (coherent)
      sin    → one draw g, modulated by the unit-mean-square envelope c_t
      white  → an independent draw g_t per token (incoherent)

    **Per-token energy is matched ACROSS BANDS.** The coefficient vector is normalized to the unit
    sphere before injection, so ‖δ_t‖ = σ·√d regardless of the band's rank. Without this the
    injected energy would be σ²·k, and at L0 the bands have ranks [1, 10, 575, 880, 582] — the
    surface would have measured band RANK (a 582× energy spread) rather than where the energy sits,
    which is the entire point of the instrument. Caught by a live cell reporting realized
    R² = 0.9999 for a rank-1 band at an amplitude that should have been substantial.

    `d_total` is the ambient dimension (defaults to `U_band.shape[0]`), so σ is in the same units as
    the Gaussian family's σ·rms convention: per-token energy = d·σ²·rms².

    Band purity is EXACT by construction (δ_t ∈ span(U_band)); the dry-run asserts it numerically
    anyway, because that is the assert whose failure would void the cell.
    """
    trk = _PositionTracker()
    k = U_band.shape[1]
    d_amb = int(d_total if d_total is not None else U_band.shape[0])
    scale = float(sigma) * math.sqrt(d_amb)
    cache = {}

    def _coef(idx, device):
        """A unit-norm coefficient vector in the band — so ‖δ_t‖ does not depend on the band rank."""
        if idx not in cache:
            gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            g = torch.randn(k, generator=gen, dtype=torch.float32)
            cache[idx] = g / g.norm().clamp_min(1e-12)
        return cache[idx].to(device)

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        U = U_band.to(hs.device, torch.float32)
        if mode == "white":
            G = torch.stack([_coef(int(p), hs.device) for p in pos.tolist()])
            c = torch.ones(T, device=hs.device, dtype=torch.float32)
        else:
            G = _coef(0, hs.device).unsqueeze(0).expand(T, k)
            c = envelope(mode, pos, freq=freq).to(hs.device, torch.float32)
        delta = scale * (G @ U.transpose(0, 1)) * c.unsqueeze(1)
        return rebuild((hs.float() + delta.unsqueeze(0)).to(hs.dtype))

    hook.tracker = trk
    return hook


def make_hook_band_deletion(U_band, hbar, mode, freq=None, seed=0):
    """DELETE side: x'_t = x_t − c_t · U U ᵀ (x_t − h̄), with c_t the delete-side ω (A5).

    `mode="full"` is the canonical band deletion (c ≡ 1). `dc` / `sin` / `white` share a frozen
    mean-square and differ only in how the removal is distributed across positions — the ω_del
    coordinate. The realized removed energy is measured per cell, never assumed.
    """
    trk = _PositionTracker()
    cache = {}

    def _env(pos, device, dtype):
        key = (int(pos[0]), int(pos[-1]), len(pos))
        if key not in cache:
            cache[key] = deletion_envelope(mode, pos, freq=freq, seed=seed)
        return cache[key].to(device, dtype)

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        x = hs.float()
        U = U_band.to(x.device, x.dtype)
        hb = hbar.to(x.device, x.dtype)
        c = _env(pos, x.device, x.dtype)
        comp = ((x - hb) @ U) @ U.transpose(0, 1)
        return rebuild((x - c.view(1, T, 1) * comp).to(hs.dtype))

    hook.tracker = trk
    return hook


# ===========================================================================
# Arm A — the asymmetric direction and its mean-shift null.
# ===========================================================================
def make_hook_traverse(v, alpha, orthogonalize_to=None):
    """x' = x + α·v, optionally with v ⊥ h̄ (the A2(ii) orthogonalized traversal).

    Returns (hook, v_used) so the realized direction is banked with the cell.
    """
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    if orthogonalize_to is not None:
        h = torch.as_tensor(orthogonalize_to, dtype=torch.float64).reshape(-1)
        v = v - (v @ h) / (h @ h) * h
    v = v / v.norm()

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        vv = v.to(hs.device, torch.float32)
        return rebuild((hs.float() + float(alpha) * vv.view(1, 1, -1)).to(hs.dtype))

    return hook, v


def make_hook_gain_match(c_fn):
    """A2(i) the mean-shift null: a PURE RESCALE x → c·x, with c chosen per token to reproduce the
    rms norm change that the true traversal induced. If this reproduces the collapse, R18's
    "direction-specific" effect is the RMSNorm cross-term 2α⟨x,v⟩ converted into uniform content
    attenuation — a mandatory control for every steering claim, and one this program did not have.

    `c_fn(norms)` maps per-token clean norms (B,T) → per-token gains (B,T).
    """
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        x = hs.float()
        nrm = x.norm(dim=-1)
        c = c_fn(nrm).to(x.dtype)
        return rebuild((x * c.unsqueeze(-1)).to(hs.dtype))
    return hook


def cross_term_telemetry(x, v):
    """Per-token ⟨x_t, v⟩ and the realized relative norm change under ±α — the A2(iii) log.

    A data-correlated v has E⟨x,v⟩ ≠ 0, so +α and −α inflate/deflate ‖x+αv‖ asymmetrically and
    RMSNorm converts that into a gain asymmetry. This is the quantity that makes H_mean-shift
    testable rather than rhetorical.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    v = v / v.norm()
    ip = x @ v
    return {"inner": ip, "mean": float(ip.mean()), "std": float(ip.std()),
            "frac_positive": float((ip > 0).double().mean())}


def gains_for_alpha(x, v, alpha):
    """The gain c_t(α) = ‖x_t + αv‖ / ‖x_t‖ that a pure rescale must reproduce."""
    x = torch.as_tensor(x, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    v = v / v.norm()
    return (x + float(alpha) * v).norm(dim=-1) / x.norm(dim=-1)


# ===========================================================================
# Arm V — the level set, its dispersion, and the sampler that measures it.
# ===========================================================================
def removed_energy_fraction(Q, Sigma):
    """E(Q) = tr(Qᵀ Σ Q) / tr(Σ) — the fraction of the (centered) residual trace that projecting
    out span(Q) removes."""
    Q = torch.as_tensor(Q, dtype=torch.float64)          # (d, k)
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)  # (d, d)
    return float(torch.einsum("ai,ab,bi->", Q, Sigma, Q) / torch.diagonal(Sigma).sum())


def sample_level_set(Sigma, k, target_E, seed, tol=1e-4, t_lo=-30.0, t_hi=30.0, max_iter=100,
                     eig=None, n_steps=48):
    """Draw a rank-k subspace with removed-energy fraction EXACTLY target_E (to `tol`).

    Rejection sampling on the energy constraint is hopeless here — REGRAIN R1b measured 0 of 4000
    draws inside ±5% of a 25%-energy target at rank 25. Instead: start from a Haar draw Q₀ and
    TILT it by a matrix POWER of Σ,

        Q(t) = orthonormalize( Σ^t Q₀ ),     t ∈ ℝ

    E(Q(t)) is monotone increasing in t: t = 0 returns the Haar draw, t → +∞ drives span(Q) onto
    the top-k eigendirections, t → −∞ onto the bottom-k. A bisection on t therefore lands every
    draw on the level set while preserving as much of the Haar randomness as the constraint
    allows.

    NOTE on the tilt that does NOT work: `Q(t) = orth((I + tΣ)Q₀)` is the obvious first guess and
    it is wrong — span((I + tΣ)Q₀) → span(ΣQ₀) as |t| → ∞ for EITHER sign, so E(t) is non-monotone
    and a bisection on it silently returns an off-constraint subspace. The matrix power is monotone
    because Σ^t inverts the spectrum for t < 0 rather than merely rescaling it.

    That residual randomness is exactly what V2 measures: if the level set is a thin shell, the
    tilted draws collapse together regardless of their independent starts, and R1b's
    "direction-free at matched energy" was measured inside a keyhole.

    NOTE on conditioning — why the tilt is applied as ORTHOGONAL ITERATION rather than one power.
    Forming `Σ^t Q₀` in one shot and orthonormalizing collapses numerically: at the real L0 spectrum
    λ₂/λ₁ = 0.055, so at t = 30 every subdominant weight is ~1e-38 and the QR of a numerically
    rank-1 matrix returns e₁ plus 24 columns of round-off. `E(30)` then reports 0.7099 — exactly
    λ₁'s share of the trace — instead of approaching E_max(25) = 0.836. Splitting the same total
    tilt into `n_steps` small steps with a QR between each (subspace iteration) keeps all k
    directions at unit scale and converges properly. `Q_of` stays continuous in t because `n_steps`
    is fixed, so the bisection remains valid.
    """
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    d = Sigma.shape[0]
    if eig is None:
        evals, evecs = torch.linalg.eigh(0.5 * (Sigma + Sigma.transpose(0, 1)))
    else:
        evals, evecs = eig
    ev = evals.clamp_min(0.0)
    lam = (ev / ev.max()).clamp_min(1e-300).log()                # log λ, normalized so max = 0
    tr = float(ev.sum())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    Q0, _ = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64))
    A0 = evecs.transpose(0, 1) @ Q0                              # Q₀'s coefficients in the eigenbasis

    def _A_of(t):
        s = float(t) / n_steps
        A = A0
        for _ in range(n_steps):
            u = s * lam
            u = u - u.max()                                      # QR is scale-invariant → safe exp
            A, _ = torch.linalg.qr(u.exp().unsqueeze(1) * A)
        return A

    def _E_of_A(A):
        # E = Σ_i λ_i ‖A_i·‖² / tr(Σ) — the removed energy depends only on the eigen-row norms
        return float((ev * (A ** 2).sum(1)).sum() / tr)

    def _Q_of_A(A):
        return evecs @ A

    lo, hi = float(t_lo), float(t_hi)
    A_lo, A_hi = _A_of(lo), _A_of(hi)
    E_lo, E_hi = _E_of_A(A_lo), _E_of_A(A_hi)
    assert E_lo <= target_E <= E_hi, (
        f"target_E={target_E} outside the NUMERICALLY REACHABLE tilt range at k={k}: "
        f"E({lo})={E_lo:.6f} .. E({hi})={E_hi:.6f} "
        f"(analytic E_max = {float(ev.flip(0)[:k].sum() / tr):.6f}; the level set degenerates to a "
        f"point as E → E_max, so near-maximal rungs are not samplable)")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        A = _A_of(mid)
        E = _E_of_A(A)
        if abs(E - target_E) < tol:
            return _Q_of_A(A), E
        if E < target_E:
            lo = mid
        else:
            hi = mid
    A = _A_of(0.5 * (lo + hi))
    return _Q_of_A(A), _E_of_A(A)


def reachable_energy_range(Sigma, k, eig=None, t_lo=-30.0, t_hi=30.0, n_steps=48):
    """(E_min, E_max_reachable, E_max_analytic) for the level-set sampler at rank k.

    A8: every site's ladder is built from THIS. The analytic ceiling `E_max` is attained only by
    the exact top-k eigenbasis — a single point with zero dispersion — so the samplable ladder must
    stop short of it. Returning the reachable range explicitly is what lets the ladder be chosen
    from data rather than from the spec's L0-specific "E = 25 %".
    """
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    evals, _ = eig if eig is not None else torch.linalg.eigh(0.5 * (Sigma + Sigma.transpose(0, 1)))
    ev = evals.clamp_min(0.0)
    E_analytic = float(ev.flip(0)[:k].sum() / ev.sum())
    lo_Q, E_lo = sample_level_set_raw(Sigma, k, t_lo, seed=0, eig=eig, n_steps=n_steps)
    hi_Q, E_hi = sample_level_set_raw(Sigma, k, t_hi, seed=0, eig=eig, n_steps=n_steps)
    return E_lo, E_hi, E_analytic


def sample_level_set_raw(Sigma, k, t, seed, eig=None, n_steps=48):
    """The tilt at a FIXED t (no bisection) — used to probe the reachable range."""
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    d = Sigma.shape[0]
    evals, evecs = eig if eig is not None else torch.linalg.eigh(0.5 * (Sigma + Sigma.transpose(0, 1)))
    ev = evals.clamp_min(0.0)
    lam = (ev / ev.max()).clamp_min(1e-300).log()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    A, _ = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64))
    A = evecs.transpose(0, 1) @ A
    s = float(t) / n_steps
    for _ in range(n_steps):
        u = s * lam
        u = u - u.max()
        A, _ = torch.linalg.qr(u.exp().unsqueeze(1) * A)
    return evecs @ A, float((ev * (A ** 2).sum(1)).sum() / float(ev.sum()))


def principal_angles(Qa, Qb):
    """The k principal angles (radians, ASCENDING) between two orthonormal k-frames.

    `svdvals` returns cosines in DESCENDING order and `arccos` is decreasing, so `arccos(s)` is
    already ascending in angle. (An earlier version flipped `s` first, which silently reversed the
    order — so `th[-1]`, documented as the largest angle, was returning the smallest. Caught when
    Gate-0a reported an implausible "largest principal angle" of 77.9° between Haar draws, where
    the largest must sit at ≈90° and the SMALLEST at ≈77°.)

    th[0]  — the smallest angle: how nearly the two frames share a direction.
    th[-1] — the largest angle: how nearly one frame has a direction orthogonal to the other.
    """
    Qa = torch.as_tensor(Qa, dtype=torch.float64)
    Qb = torch.as_tensor(Qb, dtype=torch.float64)
    s = torch.linalg.svdvals(Qa.transpose(0, 1) @ Qb).clamp(-1.0, 1.0)
    return torch.arccos(s)


def grassmann_diameter(frames):
    """Mandatory dispersion telemetry (spec §9, applied program-wide from day one).

    Returns the pairwise chordal-distance distribution and the diameter (max), plus the mean
    smallest and largest principal angles. A null ensemble that reports a diameter near zero
    cannot have tested direction-dependence, whatever its energy matching says.

    CAUTION (A7): in high ambient dimension the chordal diameter ‖sinΘ‖_F over k angles saturates
    at √k regardless of how hard an energy constraint binds, because it counts DIRECTIONS the
    draws are free in and does not weight them by whether the data lives there. Read it beside
    `energy_dispersion`, never alone. The mean SMALLEST angle is the chordal-side statistic that
    does move under the constraint.
    """
    dists, min_angles, max_angles = [], [], []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            th = principal_angles(frames[i], frames[j])
            dists.append(float(torch.sin(th).norm()))
            min_angles.append(float(th[0]))
            max_angles.append(float(th[-1]))
    if not dists:
        return {"chordal": [], "diameter": 0.0, "mean_chordal": 0.0, "n_pairs": 0}
    dd = torch.tensor(dists, dtype=torch.float64)
    return {"chordal": dists, "diameter": float(dd.max()), "mean_chordal": float(dd.mean()),
            "mean_min_angle": float(torch.tensor(min_angles).mean()),
            "mean_max_angle": float(torch.tensor(max_angles).mean()),
            "n_pairs": len(dists)}


def max_removable_energy(Sigma, k, eig=None):
    """E_max(k) = the energy fraction removed by the TOP-k eigendirections — the ceiling of the
    level-set ladder at rank k.

    A8: the spec's V2 regime ("rank 25, E = 25 %") is an L0 statement. It is inherited from
    REGRAIN R1b, whose B* lives at the embedding site. At L14 the spectrum is flatter and
    E_max(25) < 0.25, so the regime does not exist there and the sampler correctly refuses it.
    Every site's ladder is therefore built from its OWN reachable range.
    """
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    evals = eig[0] if eig is not None else torch.linalg.eigvalsh(0.5 * (Sigma + Sigma.transpose(0, 1)))
    ev = evals.flip(0).clamp_min(0.0)                 # descending
    return float(ev[:k].sum() / ev.sum())


def haar_frames(d, k, n, seed):
    """The UNCONSTRAINED rank-k reference ensemble — the denominator of the thin-ball fraction."""
    out = []
    for i in range(n):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + i)
        Q, _ = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64))
        out.append(Q)
    return out


# ---------------------------------------------------------------------------
# A7 — the energy-weighted level-set metric.
#
# Gate-0a measured the CHORDAL Grassmann diameter of the energy-constrained rank-25 ensemble at
# the real L0 Σ and got 0.996× the unconstrained diameter — at E = 25 %, where the spec predicted
# a thin ball (B9: ratio < 0.3). That is not a fat level set. It is a statistic that cannot report
# thinness, and it is the FIFTH such statistic in this lineage.
#
# Why it saturates: ‖sinΘ‖_F over 25 principal angles in d = 2048. To reach E = 25 % the tilt only
# needs to pull a small component of each frame onto the top few eigendirections; the remaining
# ~24 directions stay near-orthogonal between draws, so ‖sinΘ‖_F ≈ √24 ≈ 4.9 against a maximum of
# √25 = 5.0 no matter how strongly the constraint binds. The chordal metric counts DIMENSIONS the
# draws are free in; it does not weight them by whether anything lives there.
#
# R1b's claim was never about dimensions. It was: *at matched removed energy, WHICH directions are
# removed does not matter.* The thin-ball explanation of it is: matched-energy subspaces are forced
# to remove THE SAME ENERGY, so they had no room to differ in what they removed. The statistic that
# adjudicates that is the alignment of the REMOVED-ENERGY OPERATORS, not the angle between frames.
# ---------------------------------------------------------------------------
def removed_energy_operator(Q, Sigma):
    """R_Q = P_Q Σ P_Q, the PSD operator whose trace is the energy that ablating span(Q) removes.

    Returned in the compact (Q, M) form with M = QᵀΣQ (k×k); never materialized as d×d.
    """
    Q = torch.as_tensor(Q, dtype=torch.float64)
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
    return Q, Q.transpose(0, 1) @ Sigma @ Q


def energy_alignment(Qa, Qb, Sigma):
    """cos∠(R_a, R_b) = ⟨R_a, R_b⟩_F / (‖R_a‖_F ‖R_b‖_F) ∈ [0, 1] — Cauchy–Schwarz, so a genuine
    cosine between the two removed-energy operators.

    1 ⇒ the two subspaces remove the same energy in the same directions (no room to differ).
    ≈0 ⇒ they remove energy from disjoint parts of the spectrum.

    Computed entirely in k×k: with C = QaᵀQb,
        ⟨R_a, R_b⟩_F = tr(M_a C M_b Cᵀ),   ‖R_a‖_F = ‖M_a‖_F.
    """
    Qa, Ma = removed_energy_operator(Qa, Sigma)
    Qb, Mb = removed_energy_operator(Qb, Sigma)
    C = Qa.transpose(0, 1) @ Qb
    num = float(torch.trace(Ma @ C @ Mb @ C.transpose(0, 1)))
    den = float(torch.linalg.matrix_norm(Ma) * torch.linalg.matrix_norm(Mb))
    assert den > 0, "degenerate removed-energy operator"
    return num / den


def energy_dispersion(frames, Sigma):
    """The A7 dispersion telemetry: the pairwise energy-DISTANCE distribution D = 1 − alignment.

    Reported alongside `grassmann_diameter` for every subspace-null ensemble in every arm. Where
    the two disagree, the disagreement is the finding: the chordal metric measures how many
    directions the null was free in, the energy metric measures whether that freedom touched
    anything the model reads.
    """
    d_list, a_list = [], []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            al = energy_alignment(frames[i], frames[j], Sigma)
            a_list.append(al)
            d_list.append(1.0 - al)
    if not d_list:
        return {"alignments": [], "distances": [], "mean_distance": 0.0, "diameter": 0.0, "n_pairs": 0}
    dd = torch.tensor(d_list, dtype=torch.float64)
    return {"alignments": a_list, "distances": d_list,
            "mean_distance": float(dd.mean()), "diameter": float(dd.max()),
            "mean_alignment": float(torch.tensor(a_list, dtype=torch.float64).mean()),
            "n_pairs": len(d_list)}


# ===========================================================================
# n_eff — measured from attention, never assumed (pre-mortem (ii)).
# ===========================================================================
def n_eff_from_attention(attn):
    """Inverse participation ratio of each attention row: n_eff(q) = 1 / Σ_t w_qt².

    `attn` is (heads, q, kv) or (B, heads, q, kv). Returns per-row n_eff, flattened. The BOS sink
    concentrates attention mass, so n_eff is measured and the cancellation law is FIT against the
    measured values — the spec's own pre-mortem (ii).
    """
    a = torch.as_tensor(attn, dtype=torch.float64)
    if a.dim() == 4:
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
    s = (a ** 2).sum(-1).clamp_min(1e-30)
    return (1.0 / s).reshape(-1)


# ===========================================================================
# The cancellation-law fit (A4): damage(g) = a + b·g^(−p).
# ===========================================================================
def fit_cancellation_law(g_values, damage, n_boot=2000, seed=1337):
    """Fit the two-term law and bootstrap CIs over the per-g draw replicates.

    `damage` is a list (one entry per g) of lists of per-draw damages. Returns a, b, p with CIs
    plus the bare-power-law slope for comparison — the number the spec's B2 would have reported,
    kept visible so the misspecification is legible rather than silently corrected.

    The ideal-pool prediction is p = 1/2 (AMPLITUDE scale; see the module docstring). `a` is the
    direct-path floor: the damage a perturbation does by riding the skip connection into the
    readout at its own position, which no amount of decoherence can average away.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    g = np.asarray(g_values, dtype=float)
    means = np.array([np.mean(d) for d in damage], dtype=float)

    def model(gg, a, b, p):
        return a + b * gg ** (-p)

    def _fit(y):
        p0 = [float(y.min()), float(max(y.max() - y.min(), 1e-9)), 0.5]
        popt, _ = curve_fit(model, g, y, p0=p0, maxfev=200000,
                            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 4.0]))
        return popt

    a_hat, b_hat, p_hat = _fit(means)
    # bare power law, for the record: log(damage) ~ const − p_bare·log(g)
    ok = means > 0
    p_bare = float(-np.polyfit(np.log(g[ok]), np.log(means[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        y = np.array([rng.choice(np.asarray(d, dtype=float), size=len(d), replace=True).mean()
                      for d in damage])
        boots.append(_fit(y))
    B = np.asarray(boots)
    q = lambda col: [float(np.percentile(B[:, col], 2.5)), float(np.percentile(B[:, col], 97.5))]
    return {"a": float(a_hat), "a_ci95": q(0), "b": float(b_hat), "b_ci95": q(1),
            "p": float(p_hat), "p_ci95": q(2), "p_bare_powerlaw": p_bare,
            "ideal_pool_p": 0.5, "g": list(map(float, g)), "damage_mean": list(map(float, means))}


# ===========================================================================
# Arm W — the write–read–data triangle.
# ===========================================================================
def band_profile(M, bands):
    """Energy of the columns of M (d,·) in each Σ-band: the fraction of ‖M‖_F² inside each band.

    Applied to a fine-tune's ΔW (input side: rows live in the activation space of the matrix's
    input) this is *where fine-tuning writes, on the data's own axes*. Applied to U_unt / U_wr it
    gives those named subspaces a spectral address on the same axis.
    """
    M = torch.as_tensor(M, dtype=torch.float64)
    tot = float((M ** 2).sum())
    assert tot > 0, "degenerate M"
    out = []
    for b in range(len(bands["band_slices"])):
        U = band_projector(bands, b)
        out.append(float(((U.transpose(0, 1) @ M) ** 2).sum()) / tot)
    return out


def random_write_reference(bands):
    """The reference profile of a HAAR-RANDOM ΔW: energy splits by DIMENSION, not by variance.

        profile_random(b) = rank(b) / d

    This is the null Arm W was missing, and without it `H_offset` is uninterpretable. At L0 the band
    ranks are [1, 10, 575, 880, 582], so a random write already places ≈99.5 % of its Frobenius energy
    in bands 2–4 — exactly where the measured consensus write sits. "The write avoids the data's top
    eigendirection" is only a finding if a random write does not avoid it *by the same margin*.

    Two distinct questions, two distinct references:
      vs `lazy_training_reference` (band ENERGY)     — is the write DATA-ALIGNED?   (NTK / lazy)
      vs `random_write_reference`  (band DIMENSION)  — is the write STRUCTURED at all?
    """
    d = int(bands["basis"].shape[0])
    return [(hi - lo) / d for lo, hi in bands["band_slices"]]


def adjudicate_write_structure(profiles, random_ref, tol=0.05):
    """Is the consensus write distinguishable from a Haar-random ΔW in the input eigenbasis?

    H_write-structured  iff some band's profile departs the random reference by more than `tol`.
    H_write-isotropic   iff every band matches the random reference within `tol` — in which case the
                        write carries no information about the data's eigenbasis at all, and the
                        `lazy_training_reference` comparison says only that 2037 of 2048 dimensions
                        are not the top one.

    FAILS (reports H_write-isotropic) when the measured profile lies on the dimension-counting curve.
    That is a reachable observation and it would void the H_offset reading entirely.
    """
    dev = {k: [p[b] - random_ref[b] for b in range(len(random_ref))] for k, p in profiles.items()}
    worst = {k: max(abs(x) for x in v) for k, v in dev.items()}
    structured = {k: bool(v > tol) for k, v in worst.items()}
    return {"verdict": "H_write-structured" if any(structured.values()) else "H_write-isotropic",
            "per_matrix_worst_dev_vs_random": worst, "structured": structured,
            "random_reference": random_ref, "tol": tol}


def lazy_training_reference(bands):
    """The parameter-free NTK/lazy-training null: write density ∝ data covariance, i.e. the write's
    band profile equals the band's ENERGY fraction. H_ntk-aligned iff Ω's profile tracks this
    curve; H_offset iff it is displaced toward mid/low bands.

    This is the program's first direct test of the lazy-training prediction on third-party
    fine-tunes — and it is a hypothesis-to-kill, not a favourite.
    """
    return list(bands["band_energy"])


def form_alignment(A, B):
    """Normalized alignment of two PSD quadratic forms: tr(AB) / (‖A‖_F ‖B‖_F) ∈ [0,1].

    The three forms of the triangle: Σ (what the data varies), H (what behaviour reads — the
    measured surface lifted to a form), Ω (what fine-tuning writes). Drawn honestly small in
    F-W2, as the hypothesis-to-kill it is.
    """
    A = torch.as_tensor(A, dtype=torch.float64)
    B = torch.as_tensor(B, dtype=torch.float64)
    na, nb = float(torch.linalg.matrix_norm(A)), float(torch.linalg.matrix_norm(B))
    assert na > 0 and nb > 0
    return float((A * B).sum() / (na * nb))


# ===========================================================================
# Surface summaries.
# ===========================================================================
def support_overlap(S1, S2, q=0.75):
    """Frozen overlap coefficient between two readouts' surfaces (spec §5.2).

    Binarize each surface at its own q-quantile of damage ("dark cells"), then
    overlap = |A ∩ B| / min(|A|, |B|)  ∈ [0,1].  H_disjoint-support ⇔ overlap < 0.3 (B5).

    A5: computed WITHIN a sign (add-vs-add, delete-vs-delete). The ω coordinate does not mean the
    same thing on the two sides, so a cross-sign overlap is not defined.
    """
    import numpy as np
    A = np.asarray(S1, dtype=float)
    B = np.asarray(S2, dtype=float)
    assert A.shape == B.shape, "surfaces must share the (band × mode) grid"
    a = A >= np.quantile(A, q)
    b = B >= np.quantile(B, q)
    inter = int((a & b).sum())
    denom = min(int(a.sum()), int(b.sum()))
    assert denom > 0, "degenerate dark-cell set"
    return inter / denom


@contextlib.contextmanager
def hook_at(layers, L, hook):
    """Residual ENTERING layers[L], all positions, eager. The house site convention.

    Resets the hook's position tracker on entry so a cell's first prefill starts at position 0.
    """
    if hasattr(hook, "tracker"):
        hook.tracker.reset()
    h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    yield
    h.remove()


# ===========================================================================
# Adjudicators — each with the observation that makes it report FAILURE.
#
# This program has now watched three of its own frozen statistics turn out to be incapable of
# firing: REGRAIN R1's `em_reproduces` (saturated — a dead model and a faithful control were the
# same event), R12's em-calibrated ppl band (grew with the featurizer's own error), and R18's
# monotone rule (satisfied by every control up to ρ = 1.000). Every rule below therefore carries
# an explicit SATURATION GUARD that returns VOID, and each docstring names the observation that
# makes the rule report failure. A rule that cannot report failure is not evidence.
# ===========================================================================
def saturation_guard(*, damage_treat, damage_null, floor, intact, name):
    """VOID iff the comparison carries no information because BOTH arms are pinned.

    Fires FAILURE when: both arms are at the dead-model floor (nothing left to separate), or both
    are indistinguishable from intact (nothing happened), or the readout scale is degenerate.
    A VOID cell is documented and dropped — never imputed, never read as a null.
    """
    span = float(intact) - float(floor)
    if not span > 0:
        return {"void": True, "reason": f"{name}: degenerate readout span (intact={intact}, floor={floor})"}
    rt = (float(damage_treat) - floor) / span
    rn = (float(damage_null) - floor) / span
    if rt < 0.02 and rn < 0.02:
        return {"void": True, "reason": f"{name}: both arms at the dead-model floor — the "
                                        f"comparison cannot distinguish 'reproduced' from 'obliterated'"}
    if rt > 0.98 and rn > 0.98:
        return {"void": True, "reason": f"{name}: both arms indistinguishable from intact"}
    return {"void": False, "treat_frac": rt, "null_frac": rn}


SEQUENCE_AXIS_CELLS = ("P1", "P3")     # A9: P2 is NOT a sequence-coherence cell — see below


def adjudicate_coherence(cells, delta_K, sequence_cells=SEQUENCE_AXIS_CELLS):
    """Arm K (§4.2). `cells` maps cell-name → dict(gap_cilo, levels_cleared, void).

    H_coherence(cell) iff the coherent arm's damage exceeds its matched independent twin's band
    with CI-lo > δ_K at BOTH frozen levels. H_spectrum-suffices iff NO sequence cell clears.
    Discordance (some clear, some do not) is its own landing: H_conditional.

    **A9 — only P1 and P3 adjudicate the SEQUENCE axis.** The spec's trio put P2
    (quantization error vs the same error permuted across positions) on the coherence axis. It does
    not sit there. P1 and P3 vary the correlation of the perturbation *across* token positions at a
    fixed relationship to each token. P2 varies the correlation of the perturbation *with the token
    it is applied to*, holding the across-position structure fixed — every token still receives
    exactly one error row. Those are different coordinates, and Gate-0b showed they behave
    differently: permuting the quantization error made it 4.6× MORE VQA-damaging (−0.458 vs −0.099)
    and ~50× worse on ppl (72.9 vs 21.7) at identical error marginals and identical R². The spec's
    §1.2 table named only "permuted loses damage" and "permuted ≈ original"; the observed branch —
    permuted is far worse — was not among them. Scored on the coherence rule as frozen, this real,
    large, sign-reversed effect would have been recorded as "no coherence effect."

    So P2 is adjudicated by `adjudicate_alignment` on its own axis, and the coherence verdict rests
    on P1 and P3.

    FAILS (reports H_spectrum-suffices) when the coherent and independent arms are statistically
    indistinguishable — i.e. when shared-draw noise costs no more than iid noise at matched
    per-token fidelity. That is a real, reachable observation and it kills the second axis.
    """
    live = {k: v for k, v in cells.items()
            if k in sequence_cells and not v.get("void")}
    assert live, "every sequence-axis cell VOID — the trio carries no information; halt and document"
    cleared = [k for k, v in live.items() if v["gap_cilo"] > delta_K and v["levels_cleared"] >= 2]
    if not cleared:
        return {"verdict": "H_spectrum-suffices", "cleared": [], "live": sorted(live),
                "axis": "sequence", "excluded_from_axis": sorted(set(cells) - set(sequence_cells))}
    if len(cleared) == len(live):
        return {"verdict": "H_coherence", "cleared": cleared, "live": sorted(live),
                "axis": "sequence", "excluded_from_axis": sorted(set(cells) - set(sequence_cells))}
    return {"verdict": "H_conditional", "cleared": cleared, "live": sorted(live),
            "axis": "sequence", "excluded_from_axis": sorted(set(cells) - set(sequence_cells))}


def adjudicate_alignment(*, gap, gap_cilo, gap_cihi, delta_K, void=False):
    """A9 — the perturbation–token ALIGNMENT axis, adjudicated TWO-SIDED.

    The estimand: damage(perturbation that is a function of its own token) −
                  damage(the same perturbation dealt to the wrong tokens).
    Matched by construction on the per-token error marginals and hence on R².

    The rule is two-sided because the spec's one-sided rule could not express the observed result.
    A large negative gap (aligned corruption is BENIGN relative to its shuffled twin) is as much a
    finding as a positive one, and it means something specific: a perturbation that is the token's
    own rounding residual leaves the forward near a structured point, while the identical bytes
    dealt to other tokens do not.

    FAILS (reports H_alignment-inert) when the two arms agree within δ_K — i.e. when it does not
    matter whether a perturbation is a function of the token it lands on. That is reachable and it
    would mean the marginal distribution of the corruption is the whole story.
    """
    if void:
        return {"verdict": "VOID", "reason": "both arms pinned; the contrast carries no information"}
    if gap_cilo > delta_K:
        return {"verdict": "H_alignment-matters", "sign": "aligned_more_damaging", "gap": gap}
    if gap_cihi < -delta_K:
        return {"verdict": "H_alignment-matters", "sign": "aligned_LESS_damaging", "gap": gap,
                "reading": "the corruption is benign precisely because it is a function of its own "
                           "token; the same bytes dealt to other tokens are far more destructive"}
    return {"verdict": "H_alignment-inert", "gap": gap,
            "reading": "the corruption's marginal distribution is the whole story"}


def adjudicate_asymmetry(*, gain_frac, gain_frac_threshold, orth_asym, rand_asym_hi,
                         kink_score, cubic_score, proxy_rho, proxy_rho_cilo,
                         gain_frac_ceiling=1.25, grid_saturated=False, controls_saturated=False):
    """Arm A (§6), precedence frozen: the mean-shift null is tested FIRST because it is the
    cheapest and the strongest competitor — and because it is the author's own favoured mechanism
    (§14 B7), which is precisely why it must be given the first chance to win.

    H_mean-shift  iff the gain-matched rescale reproduces ≥ 0.8 of the +3σ collapse AND the
                  orthogonalized traversal's asymmetry falls inside the random-direction band.
    else routed by curve shape: kinked → H_boundary; smooth odd-order growth → H_cubic.
    survives all three controls → H_direction-primitive.

    FAILS (reports H_mean-shift, killing lead 1) when a pure per-token rescale — carrying NO
    direction information at all — reproduces the collapse. That single observation would
    demote R18's "direction-specific control surface" to a normalization artifact.

    **The over-shoot guard (added after the first Arm A launch).** `gain_frac >= 0.8` is one-sided
    and unbounded above. A gain cell that OBLITERATES the model "reproduces the collapse" trivially,
    and so does every control — which is exactly what happened: at α = 3σ₁ with σ₁ = √λ₁ = 88.8 the
    perturbation had norm 266 ≈ 200 % of the residual's typical norm, every direction (including all
    three matched-norm random controls) drove VQA to exactly 0.0000, `orth_asym` and `rand_asym_hi`
    were both floored to 0.0, and `H_mean-shift` fired on a dead grid.

    A matched null must REPRODUCE, not exceed. `gain_frac` is now required to land inside
    [threshold, ceiling], and a saturated grid or saturated controls return VOID. This is the third
    one-sided threshold in this program that could not fail; all three lacked an over-shoot guard.
    """
    if grid_saturated:
        return {"verdict": "VOID", "reason": "the dose grid is saturated: cells sit at the dead-model "
                                             "floor, so no direction can be distinguished from any "
                                             "other. Re-scale the amplitude and re-run."}
    if controls_saturated:
        return {"verdict": "VOID", "reason": "the matched-norm random controls are themselves at the "
                                             "floor: the control band carries no information "
                                             "(orth_asym <= rand_asym_hi is vacuous)."}
    if gain_frac > gain_frac_ceiling:
        return {"verdict": "VOID", "reason": f"the gain-matched null OVER-SHOOTS (gain_frac="
                                             f"{gain_frac:.3f} > {gain_frac_ceiling}): it does more "
                                             f"damage than the effect it is meant to reproduce, so it "
                                             f"is not a matched null. Same failure family as "
                                             f"REGRAIN R1's saturated em_reproduces clause."}
    if gain_frac >= gain_frac_threshold and orth_asym <= rand_asym_hi:
        return {"verdict": "H_mean-shift", "why": "gain-matched rescale reproduces the collapse and "
                                                  "the orthogonalized traversal's asymmetry is inside "
                                                  "the random-direction band"}
    if kink_score > cubic_score and proxy_rho_cilo > 0:
        return {"verdict": "H_boundary", "why": "kinked dose curve and damage tracks the dictionary-FVU "
                                                "manifold proxy"}
    if cubic_score >= kink_score:
        return {"verdict": "H_cubic", "why": "smooth odd-order growth; controls quiet"}
    return {"verdict": "H_direction-primitive", "why": "survives the mean-shift null, the shape "
                                                       "routing and the manifold proxy"}


def adjudicate_levelset(*, constrained_diameter, unconstrained_diameter, thin_fraction=0.30):
    """Arm V2. H_thin-ball iff the energy-constrained ensemble's Grassmann diameter is below
    `thin_fraction` of the unconstrained rank-k diameter.

    FAILS (reports H_isotropy) when constrained draws are as mutually distant as Haar draws — in
    which case R1b's 'direction-free at matched energy' was measured on a genuinely fat level set
    and the energy law stands unqualified. Both landings are calibrations the program keeps.
    """
    assert unconstrained_diameter > 0, "degenerate reference ensemble"
    ratio = constrained_diameter / unconstrained_diameter
    return {"verdict": "H_thin-ball" if ratio < thin_fraction else "H_isotropy",
            "ratio": ratio, "thin_fraction": thin_fraction}


def adjudicate_write(profiles, reference, top_band=0, mid_bands=(1, 2), tol=0.05):
    """Arm W. `profiles` maps (layer, matrix_type) → band profile; `reference` is the
    parameter-free lazy-training curve (band energy fractions).

    H_ntk-aligned iff every profile tracks the reference within `tol` in the top band.
    H_offset     iff profiles are displaced from the top band toward mid bands, consistently.
    H_stratified iff the sign of the top-band displacement differs across matrix types.

    FAILS (reports H_ntk-aligned) when fine-tuning's write density really does follow the data
    covariance — the lazy-training prediction landing on real third-party fine-tunes. That would
    be theory earning a predictive win, and it is a reachable observation, not a straw man.
    """
    devs = {k: p[top_band] - reference[top_band] for k, p in profiles.items()}
    if all(abs(dv) <= tol for dv in devs.values()):
        return {"verdict": "H_ntk-aligned", "top_band_dev": devs}
    signs = {1 if dv > tol else (-1 if dv < -tol else 0) for dv in devs.values()}
    if len(signs - {0}) > 1:
        return {"verdict": "H_stratified", "top_band_dev": devs}
    mid = {k: sum(p[b] for b in mid_bands) - sum(reference[b] for b in mid_bands)
           for k, p in profiles.items()}
    return {"verdict": "H_offset", "top_band_dev": devs, "mid_band_excess": mid}
