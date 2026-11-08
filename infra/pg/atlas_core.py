"""ATLAS core — the exact symmetries of PG1's forward, and the injectors/estimators that test them.

The theory block (spec §3.1) registers three EXACT statements about the architecture and five
hypotheses-to-kill. This module carries the weight-free half of all of them:

  T1  RMSNorm(x) = (1+γ) ⊙ x·√d̄/‖x‖ annihilates the per-token radial direction x̂_t: its Jacobian
      is (1/rms)·D_{1+γ}·(I − x̂x̂ᵀ). A radial perturbation reaches the network only through the raw
      skip path. → radial/tangential injectors + the analytic Jacobian (also Arm X's transport).
  T2  A coherent shift on every POST-RoPE key changes each logit row by a j-independent constant
      ⇒ no attention pattern moves, exactly. RoPE grades the PRE-RoPE key gauge by frequency plane:
      a coherent pre-RoPE shift in plane f decoheres at rate ω_f; the attention-visible retention
      is the Dirichlet-kernel factor |Σ_j w_j e^(i j ω_f)|. → RoPE machinery: exact rotation,
      counter-rotation (post-RoPE injection implemented as pre-RoPE injection of the
      counter-rotated vector — no monkeypatch, exact by construction), plane vectors, retention.
  T3  Mean-shift-free value/transport-path perturbations pool: coherent/iid amplitude ratio √n_eff.
  T4  For D(v) = vᵀMv over Haar unit v, CV²(D) = 2/PR(M) ⇒ d_eff := 2/CV² estimates the
      participation ratio of whatever susceptibility form a multi-draw family probed. Zero forwards.
  T5b The one-sided boundary's substrate is a dormant GeGLU gate population. → census + clamp hooks.

Everything here is weight-free and dry-run-testable. Live-forward certification of the RoPE
convention happens at Gate-0d (drivers refuse to run T-2/T-3 cells before it passes).

House rules inherited from bispec_core: hooks are shape-aware and track ABSOLUTE POSITION across
the prefill/decode boundary; every adjudicator names the observation that makes it report FAILURE
and carries a ceiling or VOID (no floor without a ceiling); no try/except hiding.
"""
import contextlib
import math

import torch

from infra.pg.bispec_core import _PositionTracker
from infra.pg.inherit_core import _split_hidden

# ---------------------------------------------------------------------------
# Frozen constants (spec §3, §6).
# ---------------------------------------------------------------------------
T1_EPS = (0.1, 0.2, 0.4)          # ×rms; 0.4 is a disclosed high-amplitude row, not adjudicated
T2_N_PLANES = 8                    # K-path rotary planes probed (of head_dim/2 = 128)
GATE_CLAMP_LAYERS = (0, 1, 2, 3)   # Arm B census/clamp layers
GATE_DORMANT_MAXFRAC = 0.0         # dormant := pre-activation strictly negative on every census row
GATE_FLIP_MINFRAC = 0.01           # flipped := dormant AND >1% of rows positive under the traversal


# ===========================================================================
# T1 — RMSNorm geometry: forward, Jacobian, radial/tangential injectors.
# ===========================================================================
def gemma_rmsnorm(x, gamma, eps=1e-6):
    """The Gemma-1 RMSNorm forward, exactly as `GemmaRMSNorm` computes it (fp32 internal):
    out = x / sqrt(mean(x²) + eps) · (1 + γ). Kept here so the T1 Jacobian identity and Arm X's
    transport are tested against the SAME function Gate-0 certifies against the live module."""
    x = torch.as_tensor(x, dtype=torch.float64)
    g = torch.as_tensor(gamma, dtype=torch.float64)
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / rms * (1.0 + g)


def rmsnorm_jacobian(x, gamma, eps=1e-6):
    """Analytic per-token Jacobian of `gemma_rmsnorm`: J = (1/rms)·D_{1+γ}·(I − x xᵀ/(d·rms²)).

    With eps → 0 the deflation direction is exactly x̂ (unit), so J x̂ ≈ 0: the radial direction is
    annihilated. The dry-run checks this against `torch.autograd.functional.jacobian` on small
    tensors (T1's registered identity) — the observation that would kill T1's exactness claim is a
    non-vanishing J x̂ beyond the eps-floor.
    """
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    g = torch.as_tensor(gamma, dtype=torch.float64).reshape(-1)
    d = x.shape[0]
    ms = float(x.pow(2).mean()) + eps
    rms = math.sqrt(ms)
    outer = torch.outer(x, x) / (d * ms)
    return ((1.0 + g).unsqueeze(1) * (torch.eye(d, dtype=torch.float64) - outer)) / rms


def make_hook_radial(eps_abs, sign=+1.0):
    """T-1 radial arm: δ_t = sign · eps_abs · x̂_t (per-token norm EXACTLY eps_abs).

    Deterministic in the token (no seed, no tracker needed — a decode step's own x defines its δ).
    The radial field carries a DC component of norm ≈ 0.82·eps_abs (mean(x̂) alignment, AM-6);
    its coherence-matched twin is `make_hook_tangential(shared=True)`, never the iid one.
    """
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        x = hs.float()
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return rebuild((x + float(sign) * float(eps_abs) * x / n).to(hs.dtype))
    return hook


def make_hook_tangential(eps_abs, seed, shared=True, orthogonalize=True, d=2048):
    """T-1 tangential arm: δ_t = eps_abs · t̂_t, t̂_t = normalize(P_{⊥x_t} u_t).

      shared=True   one draw u per seed, orthogonalized per token — the COHERENCE-MATCHED twin of
                    the radial field (AM-6: both are coherent fields with comparable DC components)
      shared=False  an independent draw per absolute position — the iid support row
      orthogonalize=False  skips the projection — the §5 matched-energy Haar null (in d = 2048 a
                    Haar direction is already ≈90° from x_t; the realized angle is telemetry)

    Per-token norm EXACTLY eps_abs in every variant. Draws keyed to absolute position (stable
    across the prefill/decode boundary).
    """
    trk = _PositionTracker()
    cache = {}

    def _draw(idx):
        if idx not in cache:
            gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            u = torch.randn(d, generator=gen, dtype=torch.float32)
            cache[idx] = u / u.norm()
        return cache[idx]

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        x = hs.float()
        if shared:
            U = _draw(0).to(x.device).view(1, 1, D).expand(B, T, D)
        else:
            U = torch.stack([_draw(int(p)) for p in pos.tolist()]).to(x.device).unsqueeze(0)
        if orthogonalize:
            xhat = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            t = U - (U * xhat).sum(-1, keepdim=True) * xhat
        else:
            t = U
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return rebuild((x + float(eps_abs) * t).to(hs.dtype))

    hook.tracker = trk
    return hook


def radial_fraction(delta_rows, x_rows):
    """Realized per-token radial energy fraction ⟨(x̂ᵀδ)²⟩/⟨‖δ‖²⟩ — the telemetry every T-1 cell
    banks beside itself (and the descriptor's radial coordinate, measured on banked rows)."""
    D = torch.as_tensor(delta_rows, dtype=torch.float64)
    X = torch.as_tensor(x_rows, dtype=torch.float64)
    xhat = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    num = ((D * xhat).sum(-1) ** 2).mean()
    den = (D ** 2).sum(-1).mean()
    assert float(den) > 0
    return float(num / den)


# ===========================================================================
# T2 — RoPE machinery. HF Gemma rotate_half convention: plane i pairs dims (i, i + dim/2),
# angle θ_{p,i} = p · ω_i, ω_i = base^(−2i/dim). Certified live at Gate-0d.
# ===========================================================================
def rope_freqs(head_dim, base=10000.0):
    """ω_i for i ∈ [0, head_dim/2): the per-plane angular rates, from the config alone."""
    half = head_dim // 2
    return base ** (-2.0 * torch.arange(half, dtype=torch.float64) / head_dim)


def rope_rotate(x, positions, base=10000.0, sign=+1.0):
    """Apply (sign=+1) or invert (sign=−1) HF Gemma RoPE to (T, head_dim) rows at `positions`.

    Pair (a, b) = (x_i, x_{i+half}) → (a·cosθ − b·sinθ, b·cosθ + a·sinθ) at sign=+1 — bit-matching
    `apply_rotary_pos_emb`'s `x·cos + rotate_half(x)·sin`. sign=−1 is the exact inverse; the
    dry-run asserts rope_rotate(rope_rotate(x, +1), −1) == x to fp64 round-off AND cross-checks a
    complex-exponential implementation. The live Gate-0d check compares a manual attention
    recomputation built on THIS function against the model's own layer output (< 1e-4 fp32);
    T-2/T-3 drivers refuse to inject before that gate is green.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    T, D = x.shape
    half = D // 2
    om = rope_freqs(D, base=base).to(x.device)
    th = torch.as_tensor(positions, dtype=torch.float64).to(x.device).unsqueeze(1) * om.unsqueeze(0)
    c, s = torch.cos(th), torch.sin(th) * float(sign)
    a, b = x[:, :half], x[:, half:]
    return torch.cat([a * c - b * s, b * c + a * s], dim=1)


def plane_vector(head_dim, plane, phase=0.0):
    """Unit vector confined to rotary plane `plane`: cos(φ)·e_plane + sin(φ)·e_{plane+half}."""
    v = torch.zeros(head_dim, dtype=torch.float64)
    half = head_dim // 2
    assert 0 <= plane < half
    v[plane] = math.cos(phase)
    v[plane + half] = math.sin(phase)
    return v


def dirichlet_retention(omega, T):
    """The zero-parameter derived retention |Σ_{j<T} e^(i j ω)|/T = |sin(Tω/2)/(T·sin(ω/2))| —
    coherence surviving RoPE for a uniform attention row over a T-window. Monotone-decreasing in ω
    over the first lobe; the F-T2 overlay curve."""
    if abs(omega) < 1e-12:
        return 1.0
    return abs(math.sin(T * omega / 2.0) / (T * math.sin(omega / 2.0)))


def measured_retention(omega, attn_rows, positions):
    """The measured-attention retention: mean over rows q of |Σ_j w_qj e^(i j ω)| — the same
    functional with the REAL attention weights. Reported beside the Dirichlet curve; where they
    disagree, the deviation localizes how attention's read differs from the uniform-window kernel."""
    W = torch.as_tensor(attn_rows, dtype=torch.float64)          # (n_rows, n_kv)
    p = torch.as_tensor(positions, dtype=torch.float64)
    ph = p * float(omega)
    re = (W * torch.cos(ph)).sum(-1)
    im = (W * torch.sin(ph)).sum(-1)
    return float(torch.sqrt(re ** 2 + im ** 2).mean())


def make_proj_out_hook_add(delta_fn, expect_dim=None):
    """Forward (post) hook on a projection Linear inside self_attn: out ← out + δ(positions).

    `delta_fn(pos_long, device) → (T, dim) fp32`. This is the ONE injection primitive for the
    whole channel family:
      * pre-RoPE  K plane cell (T-2):  δ constant across positions, confined to a plane
      * post-RoPE K/Q coherent (T-3):  δ_pre(p) = rope_rotate(δ_post, p, sign=−1) — post-RoPE
        injection AS pre-RoPE injection of the counter-rotated vector, exact, no monkeypatch
      * V coherent (T-3):              δ constant (V carries no RoPE)
    Positions tracked absolutely, so KV-cache entries written during prefill already carry their δ
    and each decode step injects at its own absolute position.
    """
    trk = _PositionTracker()

    def hook(_module, _args, out):
        B, T, D = out.shape
        if expect_dim is not None:
            assert D == expect_dim, f"projection width {D} != expected {expect_dim}"
        pos = trk.positions(T)
        d = delta_fn(pos, out.device)
        assert d.shape == (T, D), f"delta shape {tuple(d.shape)} != {(T, D)}"
        return (out.float() + d.unsqueeze(0)).to(out.dtype)

    hook.tracker = trk
    return hook


def delta_const(vec):
    """δ constant across positions (pre-RoPE plane cells, V-coherent cells)."""
    v = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)

    def fn(pos, device):
        return v.to(device).unsqueeze(0).expand(len(pos), v.shape[0]).contiguous()
    return fn


def delta_postrope_coherent(vec, base=10000.0):
    """δ_pre(p) = R_{−θ_p}·δ — after the model applies RoPE the injected shift is EXACTLY the
    position-independent post-RoPE vector δ. For K this is the T2 gauge direction (the theorem
    check); for Q it is the active channel. Norm is preserved per position (rotations)."""
    v = torch.as_tensor(vec, dtype=torch.float64).reshape(1, -1)

    def fn(pos, device):
        out = rope_rotate(v.expand(len(pos), v.shape[1]), pos, base=base, sign=-1.0)
        return out.float().to(device)
    return fn


def delta_multihead(delta_fn_one, n_heads, head_dim):
    """Replicate a single-head (T, head_dim) delta across n_heads heads of a fused projection
    output (T, n_heads·head_dim), scaled 1/√n_heads so the TOTAL per-token energy equals the
    single-head cell's (the Q-channel matched-energy convention; disclosed in the descriptor)."""
    s = 1.0 / math.sqrt(n_heads)

    def fn(pos, device):
        d1 = delta_fn_one(pos, device) * s
        return d1.repeat(1, n_heads)
    return fn


@contextlib.contextmanager
def proj_hook_at(layer, attr, hook):
    """Forward hook on layers[L].self_attn.<attr> for the duration; resets the position tracker."""
    if hasattr(hook, "tracker"):
        hook.tracker.reset()
    h = getattr(layer.self_attn, attr).register_forward_hook(hook)
    yield
    h.remove()


# ===========================================================================
# Arm B — dormant GeGLU gates: census + clamp.
# ===========================================================================
class GateCensus:
    """Sign census of gate_proj PRE-ACTIVATIONS at the requested layers.

    Accumulates, per unit: n rows seen, count(pre > 0), max(pre). `dormant(L)` = units whose
    pre-activation was ≤ 0 on EVERY census row (GATE_DORMANT_MAXFRAC = 0 — strictly-negative
    on-distribution, the T5b population). The census is the first map of PG1's dormant-gate
    reservoir and ships regardless of the clamp verdict.
    """

    def __init__(self, layers, census_layers=GATE_CLAMP_LAYERS):
        self.layers = layers
        self.Ls = list(census_layers)
        self.n = {L: 0 for L in self.Ls}
        self.pos = {L: None for L in self.Ls}
        self.mx = {L: None for L in self.Ls}
        self._h = []

    def _mk(self, L):
        def hook(_m, _a, out):
            o = out.reshape(-1, out.shape[-1]).float()
            p = (o > 0).sum(0).cpu()
            m = o.max(0).values.cpu()
            self.pos[L] = p if self.pos[L] is None else self.pos[L] + p
            self.mx[L] = m if self.mx[L] is None else torch.maximum(self.mx[L], m)
            self.n[L] += o.shape[0]
        return hook

    def __enter__(self):
        self._h = [self.layers[L].mlp.gate_proj.register_forward_hook(self._mk(L)) for L in self.Ls]
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()

    def frac_positive(self, L):
        assert self.n[L] > 0, f"census saw no rows at L{L}"
        return (self.pos[L].double() / self.n[L])

    def dormant(self, L):
        return self.frac_positive(L) <= GATE_DORMANT_MAXFRAC


def flipped_population(census_clean, census_trav):
    """{L: bool-mask} of units dormant on the clean forward AND positive on ≥ GATE_FLIP_MINFRAC of
    rows under the traversal. FAILS (empty masks everywhere) when +αv₁ flips nothing — a reachable
    observation that would kill T5b's substrate claim on its own."""
    out = {}
    for L in census_clean.Ls:
        dorm = census_clean.dormant(L)
        flip = census_trav.frac_positive(L) >= GATE_FLIP_MINFRAC
        out[L] = dorm & flip
    return out


def make_gate_clamp_hook(unit_mask):
    """The clamp: CAP the selected units' gate pre-activations at 0 (prevent switching on).

    Chosen over 'set to −∞' because gelu(x) < 0 for mildly negative x, so zeroing dormant units
    outright would CHANGE the clean forward and the identity-at-α=0 integrity check could never
    pass. Capping at 0 is an exact no-op whenever the clean pre-activation is ≤ 0 — which is the
    definition of the population being clamped — so identity-at-α=0 holds by construction and any
    violation is an instrument bug, fail loud. Under the traversal the capped unit cannot fire.
    """
    mask = torch.as_tensor(unit_mask, dtype=torch.bool)

    def hook(_m, _a, out):
        m = mask.to(out.device)
        clamped = out.float()
        clamped[..., m] = clamped[..., m].clamp_max(0.0)
        return clamped.to(out.dtype)
    return hook


@contextlib.contextmanager
def gate_clamps_at(layers, masks):
    """Install clamp hooks per layer for the duration. `masks` = {L: bool-mask}."""
    hs = [layers[L].mlp.gate_proj.register_forward_hook(make_gate_clamp_hook(m))
          for L, m in masks.items() if bool(m.any())]
    yield
    for h in hs:
        h.remove()


# ===========================================================================
# T4 — the dispersion dimension d_eff = 2/CV², and its calibration.
# ===========================================================================
def d_eff_from_damages(vals):
    """d_eff = 2/CV² over a multi-draw damage family. For D(v)=vᵀMv with Haar unit v,
    CV²(D) = 2·PR⁻¹·(1 + O(1/d)) with PR = (tr M)²/tr(M²), so d_eff estimates PR(M) — the
    dimensionality of the susceptibility form the family probed, at zero forwards.

    FAILS (returns inf/degenerate) when the draws are identical — a family with no dispersion has
    no measurable dimension, and the calibration (R-D) turns n=3 readings into intervals, not
    points. Returns dict with the raw ingredients so nothing is hidden behind the ratio.
    """
    import numpy as np
    v = np.asarray(vals, dtype=float)
    assert v.size >= 2, "d_eff needs at least 2 draws"
    m = float(v.mean())
    assert m != 0.0, "degenerate family: zero mean damage"
    cv2 = float(v.var(ddof=1)) / m ** 2
    return {"n": int(v.size), "mean": m, "cv2": cv2,
            "d_eff": (float("inf") if cv2 == 0 else 2.0 / cv2)}


def synthetic_form_damages(pr, d, n_draws, seed):
    """Block-spectrum synthetic form with EXACT participation ratio `pr` (pr unit eigenvalues,
    rest 0): D(v) = Σ_{i<pr} v_i² over Haar unit v — the R-D calibration ground truth."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    g = torch.randn(n_draws, d, generator=gen, dtype=torch.float64)
    v = g / g.norm(dim=1, keepdim=True)
    return (v[:, :int(pr)] ** 2).sum(1).tolist()


def calibrate_d_eff(prs=(1, 4, 32, 256), n_draws=(3, 8), d=2048, n_rep=400, seed=7):
    """The R-D calibration table: recovered d_eff quantiles per (PR, n). The n=3 bias curve is the
    point — quantifying HOW wide n=3 is. Returns {(pr, n): {q05, q25, median, q75, q95, factor2_ok}}."""
    import numpy as np
    out = {}
    for pr in prs:
        for n in n_draws:
            vals = []
            for r in range(n_rep):
                dm = synthetic_form_damages(pr, d, n, seed * 1_000_003 + pr * 977 + n * 31 + r)
                vals.append(d_eff_from_damages(dm)["d_eff"])
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            q = lambda p: float(np.percentile(v, p))
            out[(pr, n)] = {"q05": q(5), "q25": q(25), "median": q(50), "q75": q(75), "q95": q(95),
                            "n_rep": int(v.size),
                            "factor2_ok": bool(q(50) >= pr / 2 and q(50) <= 2 * pr)}
    return out


# ===========================================================================
# Adjudicators (Arm T + Arm B rules). Each names its failure observation; each has a ceiling
# or VOID. Status flags are computed LAST by the drivers (REGRAIN E17).
# ===========================================================================
def adjudicate_rt1(*, ratio_by_eps, tang_by_eps, seed_noise, saturated_by_eps,
                   floor_ratio=0.25, ceil_ratio=0.75, range_mult=3.0):
    """R-T1 (the MDE). ratio_rt per adjudicated ε ∈ {0.1, 0.2}.

    FAILS H_radial-gauge (the author's favourite) when ratio ≥ 0.75 (`H_radial-active`, the
    ceiling): the gauge picture bounded at its cheapest point. VOIDs when the tangential arm has
    no range (< 3× seed-noise) or either arm is saturated — both named, both reachable.
    """
    eps_adj = [e for e in ratio_by_eps if abs(e - 0.4) > 1e-9]
    assert eps_adj, "no adjudicated amplitudes"
    if any(saturated_by_eps[e] for e in eps_adj):
        return {"verdict": "VOID-saturated", "response": "drop to eps=0.05 once (pre-specified)"}
    if all(tang_by_eps[e] < range_mult * seed_noise for e in eps_adj):
        return {"verdict": "VOID-no-range",
                "response": "one pre-specified re-run at eps=0.4, else the tolerance curve IS the result"}
    rs = [ratio_by_eps[e] for e in eps_adj]
    if all(r <= floor_ratio for r in rs):
        return {"verdict": "H_radial-gauge", "ratios": rs}
    if all(r >= ceil_ratio for r in rs):
        return {"verdict": "H_radial-active", "ratios": rs,
                "reading": "the gauge is bounded at its cheapest point; M2 runs WITHOUT the radial "
                           "discount (pre-specified) and the theory block's error is the finding"}
    return {"verdict": "H_radial-partial", "ratios": rs,
            "reading": "the skip path carries a measurable share; the radial discount becomes a "
                       "FITTED parameter in M2 (pre-specified fallback)"}


def adjudicate_rt2(*, rho_freq, gap, gap_cilo, gap_cihi, rho_shuffle, any_void):
    """R-T2. FAILS H_ladder when the ladder is flat (|ρ| ≤ 0.3, gap CI straddles 0) — T2's gauge
    stays a theorem but is not load-bearing at this energy. `H_inverted` is the ceiling branch."""
    if any_void:
        return {"verdict": "VOID", "response": "re-run the dead plane at half energy (pre-specified, once)"}
    if abs(rho_shuffle) > 0.3:
        return {"verdict": "VOID-shuffle", "reading": "the frequency-shuffled control shows a ladder "
                                                      "of its own — the ladder statistic is confounded "
                                                      "and cannot adjudicate"}
    if rho_freq >= 0.8 and gap_cilo > 0:
        return {"verdict": "H_ladder", "rho": rho_freq}
    if rho_freq <= -0.8:
        return {"verdict": "H_inverted", "rho": rho_freq,
                "reading": "low-frequency planes carry the load — a slow-rotating addressing "
                           "subspace; successor named"}
    if abs(rho_freq) <= 0.3 and gap_cilo <= 0 <= gap_cihi:
        return {"verdict": "H_flat", "rho": rho_freq,
                "reading": "K-path damage is not addressing-decoherence-limited at this energy; "
                           "the damage carrier sits in V/Q (sharpens T-3's reading)"}
    return {"verdict": "H_graded", "rho": rho_freq,
            "reading": "partial ordering; the deviation profile from the retention curve localizes "
                       "where the read deviates from the kernel prediction"}


def adjudicate_rt3(*, k_damage, v_damage, q_damage, seed_noise, haar_damage):
    """R-T3 — descriptive triple + the THEOREM CHECK. K-coherent post-RoPE damage > 3× seed-noise
    means the hook is not where the theorem applies ⇒ INSTRUMENT BUG, fail loud (a signal designed
    to be able to fail). The 6 orderings each map to a named statement; the ordering is the finding."""
    theorem_ok = k_damage <= 3.0 * seed_noise
    order = sorted([("K", k_damage), ("V", v_damage), ("Q", q_damage)], key=lambda t: -t[1])
    readings = {
        ("Q", "V", "K"): "the registered T2 prediction: addressing queries carry stream damage, "
                         "values ride at gain 1, the key gauge is exact",
        ("V", "Q", "K"): "transport outweighs addressing at this energy; the key gauge holds",
        ("Q", "K", "V"): "values are the protected channel, keys leak — check retention accounting",
        ("K", "Q", "V"): "the key gauge FAILED the live check — instrument bug branch",
        ("K", "V", "Q"): "the key gauge FAILED the live check — instrument bug branch",
        ("V", "K", "Q"): "queries are inert at this energy — the addressing circuit reads through "
                         "a robust code; named structure",
    }
    key = tuple(n for n, _ in order)
    return {"theorem_check_pass": bool(theorem_ok), "ordering": key,
            "damages": {"K": k_damage, "V": v_damage, "Q": q_damage,
                        "haar_channel_null": haar_damage, "seed_noise": seed_noise},
            "reading": readings[key]}


def adjudicate_rt4_legA(*, ratio_ci, pred_law, pred_const, degenerate_gap=0.3):
    """R-T4 leg A. VOID-degenerate iff |√n_eff(L14) − √n_eff(L0)| < 0.3 (no discrimination — itself
    a finding about attention-geometry stability; leg B then carries the family alone). Four
    overlap patterns, each with a lesson."""
    lo, hi = ratio_ci
    if abs(pred_law - pred_const) < degenerate_gap:
        return {"verdict": "VOID-degenerate",
                "reading": "the two predictions coincide at this site pair — attention geometry is "
                           "depth-stable; leg B carries the family alone"}
    in_law = lo <= pred_law <= hi
    in_const = lo <= pred_const <= hi
    v = {(True, False): ("H_law", "the ratio moves with the site's own attention pooling — T3 is a law"),
         (False, True): ("H_constant", "the factor is a site constant, not a pooling quantity — M2's "
                                       "sequence factor becomes a fitted constant (pre-specified) and "
                                       "'what sets the constant?' is the named new question"),
         (True, True): ("H_both", "the CI is too wide to separate the predictions — a power statement, "
                                  "reported as such"),
         (False, False): ("H_neither", "the measured ratio excludes BOTH predictions — the transport "
                                       "factor is set by something neither model names")}[(in_law, in_const)]
    return {"verdict": v[0], "reading": v[1], "ci": [lo, hi],
            "pred_law": pred_law, "pred_const": pred_const}


# ===========================================================================
# Grammar-cell injectors (Arm M): sub-band-confined noise with sequence mode + radial/tangential
# projection, sub-band subspace deletion (shared / per-token), and channel cells. One constructor
# family used by BOTH the discovery fill-ins and the sealed battery — the sealed driver contains
# no cell-construction code of its own.
# ===========================================================================
def make_hook_subband_noise(U_sb, eps_abs, seed, mode, rt="none", freq=4.0):
    """δ_t = eps_abs · c_t · P_rt(U_sb ĝ_t), renormalized per token so ‖δ_t‖ = eps_abs·|c_t| exactly.

    mode: dc (one unit-norm coefficient draw, broadcast) · sin4 (same draw, ±√2cos envelope) ·
    white (fresh draw per absolute position). rt: none · radial (project onto x̂_t) · tangential
    (project off x̂_t); after projection δ is rescaled back to the matched per-token energy, so the
    rt axis moves WHERE the energy points, never how much of it there is. Realized radial fraction
    is telemetry, banked by the driver.
    """
    from infra.pg.bispec_core import envelope
    trk = _PositionTracker()
    k = U_sb.shape[1]
    cache = {}

    def _coef(idx):
        if idx not in cache:
            gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            g = torch.randn(k, generator=gen, dtype=torch.float32)
            cache[idx] = g / g.norm().clamp_min(1e-12)
        return cache[idx]

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        x = hs.float()
        U = U_sb.to(x.device, torch.float32)
        if mode == "white":
            G = torch.stack([_coef(int(p)) for p in pos.tolist()]).to(x.device)
            c = torch.ones(T, device=x.device)
        elif mode == "dc":
            G = _coef(0).to(x.device).unsqueeze(0).expand(T, k)
            c = torch.ones(T, device=x.device)
        elif mode == "sin4":
            G = _coef(0).to(x.device).unsqueeze(0).expand(T, k)
            c = envelope("sin4", pos, freq=freq).to(x.device, torch.float32)
        else:
            raise ValueError(f"unknown grammar add mode {mode!r}")
        delta = (G @ U.transpose(0, 1)).unsqueeze(0).expand(B, T, D).contiguous()
        if rt != "none":
            xhat = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rad = (delta * xhat).sum(-1, keepdim=True) * xhat
            delta = rad if rt == "radial" else (delta - rad)
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = float(eps_abs) * delta * c.view(1, T, 1)
        return rebuild((x + delta).to(hs.dtype))

    hook.tracker = trk
    return hook


def make_hook_subband_deletion(U_sb, hbar, frac, seed, per_token, pool_frames=None):
    """Delete a rank-k′ (k′ = max(1, r//2)) random sub-subspace of span(U_sb), removal fraction
    c = √frac at every position. per_token=True draws an independent sub-subspace per absolute
    position from `pool_frames` (small QR pool inside the sub-band's coefficient space)."""
    trk = _PositionTracker()
    r = U_sb.shape[1]
    kp = max(1, r // 2)
    c = math.sqrt(float(frac))
    if not per_token:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        A = torch.randn(r, kp, generator=gen, dtype=torch.float64)
        Qc, _ = torch.linalg.qr(A)
        Q_shared = (U_sb.double() @ Qc)

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        pos = trk.positions(T)
        x = hs.float()
        hb = hbar.to(x.device, x.dtype)
        z = x - hb
        if not per_token:
            Q = Q_shared.to(x.device, x.dtype)
            out = x - c * (z @ Q) @ Q.transpose(0, 1)
        else:
            P = pool_frames.to(x.device, x.dtype)                # (n_pool, D, kp)
            Qt = P[(pos % P.shape[0]).to(x.device)]
            coeff = torch.einsum("btd,tdk->btk", z, Qt)
            out = x - c * torch.einsum("btk,tdk->btd", coeff, Qt)
        return rebuild(out.to(hs.dtype))

    hook.tracker = trk
    return hook


def subband_pool(U_sb, n_pool, seed):
    """Per-token sub-subspace pool INSIDE span(U_sb): (n_pool, d, k′) frames."""
    r = U_sb.shape[1]
    kp = max(1, r // 2)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(n_pool, r, kp, generator=gen, dtype=torch.float64)
    Qc, _ = torch.linalg.qr(A)
    return torch.einsum("dr,nrk->ndk", U_sb.double(), Qc).float()


CHANNEL_PLANES = {"k_low": 96, "k_high": 2}      # ω ≈ 1e-3 vs ≈ 0.87 rad/token (θ=1e4, dim 256)


def channel_delta_fn(channel, mode, eps_abs, seed, head_dim, n_heads, rope_theta):
    """The channel-cell delta factory for `make_proj_out_hook_add`. Returns (attr, delta_fn).

    k_low/k_high: pre-RoPE plane-confined; V: plain coherent/iid; Q: post-RoPE coherent
    (counter-rotated) split across heads at matched total energy. mode dc = one draw broadcast;
    white = fresh draw per absolute position (phase/direction respectively)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if channel in ("k_low", "k_high"):
        plane = CHANNEL_PLANES[channel]

        def one(idx):
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            ph = float(torch.rand(1, generator=g) * 2 * math.pi)
            return eps_abs * plane_vector(head_dim, plane, phase=ph)

        if mode == "dc":
            return "k_proj", delta_const(one(0))
        v_cache = {}

        def fn(pos, device):
            outv = []
            for p in pos.tolist():
                if p not in v_cache:
                    v_cache[p] = one(p)
                outv.append(v_cache[p])
            return torch.stack(outv).float().to(device)
        return "k_proj", fn
    if channel == "v":
        def onev(idx):
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            u = torch.randn(head_dim, generator=g, dtype=torch.float64)
            return eps_abs * u / u.norm()
        if mode == "dc":
            return "v_proj", delta_const(onev(0))
        cache = {}

        def fnv(pos, device):
            outv = []
            for p in pos.tolist():
                if p not in cache:
                    cache[p] = onev(p)
                outv.append(cache[p])
            return torch.stack(outv).float().to(device)
        return "v_proj", fnv
    if channel == "q":
        u = torch.randn(head_dim, generator=gen, dtype=torch.float64)
        u = eps_abs * u / u.norm()
        if mode == "dc":
            return "q_proj", delta_multihead(delta_postrope_coherent(u, base=rope_theta),
                                             n_heads, head_dim)
        cache = {}

        def one_q(idx):
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
            uu = torch.randn(head_dim, generator=g, dtype=torch.float64)
            return eps_abs * uu / uu.norm()

        def fnq(pos, device):
            rows = []
            for p in pos.tolist():
                if p not in cache:
                    cache[p] = one_q(p)
                rows.append(cache[p])
            base_rows = torch.stack(rows)
            rot = rope_rotate(base_rows, torch.as_tensor(pos), base=rope_theta, sign=-1.0)
            return (rot.float() / math.sqrt(n_heads)).repeat(1, n_heads).to(device)
        return "q_proj", fnq
    raise ValueError(f"unknown channel {channel!r}")


# ===========================================================================
# T8 — Arm X: intervention transport by conjugation through the RMSNorm Jacobian.
#
# V1r proved the SUBSPACE transports exactly (orth(D_γU)) while the ABLATION does not (identity
# gap 0.355): deleting span(D_γU) from x lands the post-norm point in the right complement at the
# wrong place. The conjugated carry targets the native post-norm point directly:
#
#     want   φ(x + Δ) ≈ (I − UUᵀ) φ(x)          (the native operator's output)
#     Δ*     = argmin_{Δ ⊥ x̂} ‖J(x)Δ − w‖,  w = −UUᵀφ(x)
#
# With J = c·D·M (c = 1/rms, D = diag(1+γ), M = I − x̂x̂ᵀ) the constrained least-squares has a
# CLOSED FORM per token:  Δ = (1/c)·[D⁻¹w − μ·D⁻²x̂],  μ = (x̂ᵀD⁻¹w)/(x̂ᵀD⁻²x̂).
# An exact solution cannot exist in general — (I−P)φ(x) leaves the RMSNorm image manifold — so the
# residual after the (optionally Newton-refined) step is REPORTED per token; how much of the 0.355
# gap first-order conjugation closes is exactly R-X's estimand.
# ===========================================================================
def conjugated_removal_delta(x, U, gamma, eps=1e-6, newton_steps=1):
    """Per-token conjugated-transport correction Δ (rows match x). Returns (delta, telemetry)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    U = torch.as_tensor(U, dtype=torch.float64)
    g = torch.as_tensor(gamma, dtype=torch.float64).reshape(1, -1)
    D = 1.0 + g
    d = x.shape[1]

    def phi(xx):
        rms = (xx.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        return xx / rms * D

    def step(xx, target):
        z = phi(xx)
        w = target - z
        rms = (xx.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        c = 1.0 / rms
        xhat = xx / xx.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Dinv_w = w / D
        mu = (xhat * Dinv_w).sum(-1, keepdim=True) / (xhat.pow(2) / D.pow(2)).sum(-1, keepdim=True)
        return (Dinv_w - mu * xhat / D.pow(2)) / c

    z0 = phi(x)
    target = z0 - (z0 @ U) @ U.transpose(0, 1)
    delta = step(x, target)
    for _ in range(int(newton_steps)):
        delta = delta + step(x + delta, target)
    zf = phi(x + delta)
    resid = (zf - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-12)
    # the reader-visible criterion: how much energy is left in span(U) after the carry
    in_span = ((zf @ U) ** 2).sum(-1) / zf.pow(2).sum(-1).clamp_min(1e-12)
    return delta, {"rel_resid_mean": float(resid.mean()), "rel_resid_p95":
                   float(resid.quantile(0.95)), "in_span_after_mean": float(in_span.mean())}


def make_hook_conjugated_removal(U, gamma, newton_steps=1):
    """Residual-entering hook applying the conjugated carry per token, every position."""
    U64 = torch.as_tensor(U, dtype=torch.float64)
    g64 = torch.as_tensor(gamma, dtype=torch.float64)

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        B, T, D = hs.shape
        x = hs.reshape(-1, D).double()
        delta, _ = conjugated_removal_delta(x, U64.to(x.device), g64.to(x.device),
                                            newton_steps=newton_steps)
        return rebuild((x + delta).reshape(B, T, D).to(hs.dtype))
    return hook


# ===========================================================================
# Arm W — per-mode error-coefficient correlation ρ_i (T6's visible-variance factor).
#
#   ∇_W = Σ_t g_t y_tᵀ  (the raw accumulated gradient of the frozen base under a proxy loss)
#   ρ_i = ‖∇_W u_i‖² / (Σ_t‖g_t‖² · Σ_t (y_tᵀu_i)²)  ∈ [0, 1]   (Cauchy–Schwarz)
#
# so mode-i write intensity λ_i ρ_i ∝ ‖∇_W u_i‖² — the T6 prediction IS the gradient's own mode
# profile; ρ is still computed explicitly because the shuffled-ρ control permutes it against λ.
# ===========================================================================
class GradNormCapture:
    """Backward hooks accumulating Σ_t ‖g_t‖² (g = ∂loss/∂ module-output, per position) for the
    requested (layer, attr) modules; the driver reads module.weight.grad for ∇_W itself."""

    def __init__(self, modules):
        self.mods = modules                     # {key: nn.Linear}
        self.sum_g2 = {k: 0.0 for k in modules}
        self.n_pos = {k: 0 for k in modules}
        self._h = []

    def _mk(self, key):
        def hook(_m, _gin, gout):
            g = gout[0]
            assert g is not None, f"no grad at {key}"
            gg = g.reshape(-1, g.shape[-1]).double()
            self.sum_g2[key] += float((gg ** 2).sum())
            self.n_pos[key] += gg.shape[0]
        return hook

    def __enter__(self):
        self._h = [m.register_full_backward_hook(self._mk(k)) for k, m in self.mods.items()]
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()


def rho_per_mode(nabla_w, basis, coef_sq_per_mode, sum_g2):
    """ρ_i per input-space mode. `coef_sq_per_mode` = Σ_t (y_tᵀu_i)² = n·(uᵢᵀ M uᵢ) from the SAME
    input-Gram capture (the bispec_w convention). All fp64."""
    G = torch.as_tensor(nabla_w, dtype=torch.float64)
    B = torch.as_tensor(basis, dtype=torch.float64)
    proj = ((G @ B) ** 2).sum(0)                                     # ‖∇_W u_i‖² per mode
    cs = torch.as_tensor(coef_sq_per_mode, dtype=torch.float64).clamp_min(1e-300)
    rho = proj / (float(sum_g2) * cs)
    # Cauchy–Schwarz holds EXACTLY only with same-token accounting (∇_W, Σ‖g‖², Σc² from the
    # same pass); fp32-forward/fp64-Gram roundoff allows ~1e-3 excess on near-null tail modes
    assert float(rho.max()) <= 1.0 + 1e-2, f"ρ escaped [0,1]: {float(rho.max())} — accounting bug"
    return rho.clamp(0.0, 1.0)


def band_aggregate(per_mode_vals, sb_or_band_slices):
    """Sum a per-mode intensity into band bins and normalize to a profile (sums to 1)."""
    v = torch.as_tensor(per_mode_vals, dtype=torch.float64)
    parts = torch.tensor([float(v[lo:hi].sum()) for lo, hi in sb_or_band_slices], dtype=torch.float64)
    tot = float(parts.sum())
    assert tot > 0, "degenerate intensity profile"
    return (parts / tot).tolist()


def adjudicate_rb2(*, recovered_frac, identity_dev, seed_noise, random_clamp_recovered,
                   id_mult=2.0, rand_ceiling=0.25):
    """R-B2. Guards FIRST (the clamp integrity pair), then the branch. FAILS H_dormant-gate when
    recovered_frac ≤ 0.25 (`H_elsewhere`) — and the census still ships as the reservoir map."""
    if identity_dev > id_mult * seed_noise:
        return {"verdict": "VOID-instrument",
                "reading": "the clamp has its own footprint at alpha=0 — scoped; battery re-run "
                           "with soft clamp once (pre-specified)"}
    if random_clamp_recovered > rand_ceiling:
        return {"verdict": "VOID-clamp-artifact",
                "reading": "ANY clamping helps — a finding about MLP fragility, reported; arm ends"}
    if recovered_frac >= 0.5:
        return {"verdict": "H_dormant-gate", "recovered_frac": recovered_frac,
                "reading": "mechanism nailed causally; the saturation-shaped proxy replaces A3's "
                           "monotone one program-wide"}
    if recovered_frac <= 0.25:
        return {"verdict": "H_elsewhere", "recovered_frac": recovered_frac,
                "reading": "the boundary is not layer-local MLP gating — attention/later layers are "
                           "the successor's territory; the gate census still ships"}
    return {"verdict": "H_partial", "recovered_frac": recovered_frac,
            "reading": "the fraction is the finding; the boundary is distributed across mechanisms"}
