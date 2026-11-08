"""REGRAIN core — the corrected featurizer class + the pure logic GRAIN lacked
(studies/lineage/paligemma1-pi0/research/2026-07-09_regrain_spec.md §3, §5, §7).

Everything here is unit-testable weight-free (infra/pg/tests/test_regrain_core.py); the PG1
forward lives in the jobs. `grain_bsf` is REUSED unchanged for m=1 (the house BatchTopKSAE
equivalence contract) — this module only ADDS:

  * SIGNED codes (BSF App. D: "no more ReLU involved") — the §5 attribution factorial's moved
    variable. `grain_bsf.BlockTopKSAE(signed=True)` is the vanilla arm; `GrassmannBSF` is the
    paper's own workhorse (tied encoder, per-block orthonormal charts, QR retraction, scalar γ).
  * MDL at a frozen distortion (paper Eq. 5-6): support bits + reverse-water-filled code bits +
    amortized dictionary bits. Descriptive/selection only — never a gate.
  * The ROTATED-ERROR null: splice x + Q(x̂ − x) with Q a seeded Haar rotation. Identical error
    NORM, destroyed error DIRECTION. Separates direction-carried from energy-carried damage.
  * The DISTORTION FRONTIER corruptions (App. G.1 adapted to the live forward): uniform
    quantization / Gaussian / energy-matched-ablation / rotated-error, each reported at its
    realized R².
  * The AMENDED nulls + adjudicators (spec §0.5):
      A2  usage-matched random-atom unions (matched on the nuisance selection used)
      A1  canonicity-matched surrogate subspaces → the Jaccard null Arm D actually needs
      A3  the selectivity clause (off-target ppl) inside the atom-handle table

Design rules (CLAUDE.md): NO try/except hiding — a wrong assumption RAISES with a trace.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

from infra.pg.ctrliface_core import orthonormalize, principal_angle_overlap
from infra.pg.endow_atlas import normalized_canonicity
from infra.pg.grain_bsf import _geometric_median, _lr_schedule, jaccard
from infra.pg.inherit_core import _split_hidden

# Frozen factorial defaults (spec §5.2).
W_DICT = 16384
R_ACTIVE = 96
QR_EVERY = 20                      # Grassmannian retraction period (paper App. D)


# ===========================================================================
# The Grassmannian block-sparse featurizer (paper App. D) — signed by construction.
# ===========================================================================
def _batched_orthonormalize_(charts, b, chunk=512):
    """In-place per-block QR retraction of a (d, W) chart matrix with contiguous b-column blocks.
    Each block's b columns are re-orthonormalized in fp64. Chunked over blocks: at d=2048,
    W=16384, b=4 the whole batch is a 268MB fp64 tensor before Q — chunking bounds the peak.
    Fail-loud on a rank-deficient block (a collapsed chart is a finding, not a fallback)."""
    d, W = charts.shape
    assert W % b == 0, f"W={W} not divisible by b={b}"
    assert charts.is_contiguous(), "charts must be contiguous — reshape below must be a VIEW"
    G = W // b
    blocks = charts.reshape(d, G, b)
    for s in range(0, G, chunk):
        sub = blocks[:, s:s + chunk, :].permute(1, 0, 2).double()    # (g, d, b)
        Q, R = torch.linalg.qr(sub)
        rdiag = R.diagonal(dim1=-2, dim2=-1)
        assert float(rdiag.abs().min()) > 1e-9, \
            f"rank-deficient chart block (min |R_ii|={float(rdiag.abs().min())})"
        # QR sign convention: force R_ii > 0 so the retraction is continuous across steps.
        Q = Q * torch.sign(rdiag).unsqueeze(1)                       # (g,1,b) broadcast over d
        blocks[:, s:s + chunk, :] = Q.permute(1, 0, 2).to(charts.dtype)
    return charts


class GrassmannBSF(nn.Module):
    """Block-sparse featurizer on the Grassmannian (paper App. D).

    charts  U ∈ R^{d×W}, contiguous b-column blocks, each block ORTHONORMAL (QR retraction
            every `qr_every` steps — the paper: "projecting every 20 steps suffices").
    encode  z = γ · Uᵀ(x − b_dec)          (tied encoder; SIGNED — App. D: "no more ReLU involved")
            Π_ℓ retains the k_blk blocks of largest ‖z_g‖₂. Selection is PER-SAMPLE, as the
            paper specifies ("Selection is per-sample; batch-level variants in the spirit of
            BatchTopK are a natural extension we leave to future work") — this is the ONE place
            the Grassmannian arm departs from the house BatchTopK convention, on purpose.
    decode  x̂ = b_dec + U z_sparse          (γ already applied in the code, as in Eq. 4)

    γ is a single positive scalar shared across blocks, carried as log γ so it stays positive
    (paper App. D); it compensates the energy lost by tying encoder and decoder.
    b_dec is a house addition (geometric-median init); the paper's decoder has no bias. Signed
    codes are apex-invariant, so this cannot flatter the signed arm — disclosed in §15.

    b=1 reduces to a signed, tied-encoder, unit-norm-atom TopK dictionary (the frontier's
    1-D leg), NOT to the house rectified BatchTopKSAE.
    """

    def __init__(self, activation_dim, dict_size, b, k_blk):
        super().__init__()
        assert dict_size % b == 0, f"dict_size {dict_size} not divisible by b={b}"
        n_blocks = dict_size // b
        assert 1 <= k_blk <= n_blocks, f"k_blk={k_blk} outside [1, {n_blocks}]"
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)
        self.m = int(b)                       # `m` name kept: every block-grain surface reads .m
        self.b = int(b)
        self.n_blocks = int(n_blocks)
        self.k_blk = int(k_blk)
        self.signed = True
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.charts = nn.Parameter(torch.randn(activation_dim, dict_size))
        with torch.no_grad():
            _batched_orthonormalize_(self.charts.data, b)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))       # γ = exp(log_gamma) > 0
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    @property
    def gamma(self):
        return torch.exp(self.log_gamma)

    def block_norms(self, z):
        return z.reshape(z.shape[0], self.n_blocks, self.b).norm(dim=2)

    def encode(self, x, return_active=False, use_threshold=True):
        """PER-SAMPLE Π_ℓ at BOTH training and inference (paper App. D). `use_threshold` is accepted
        for interface compatibility and DELIBERATELY IGNORED.

        Why: the house BatchTopK learns an inference threshold as the smallest selected block norm
        seen during batch-level selection — a valid global cutoff *because* selection is
        batch-level. Under PER-SAMPLE selection that same statistic is the smallest k-th-largest
        norm across rows, which is far below the typical row's cutoff, so a thresholded forward
        admits vastly more than ℓ blocks. Measured on the first factorial run: `mean_block_l0`
        = 3365.7 against `k_blk` = 96, giving held-out FVU 8.75 (reconstruction worse than the
        mean) and a decode that overshoots by ≈ W/d. The paper's Π_ℓ is a hard count; a Grassmannian
        BSF has no threshold to learn."""
        z = self.gamma * ((x - self.b_dec) @ self.charts)            # (n, W), signed
        norms = self.block_norms(z)
        idx = norms.topk(self.k_blk, dim=1, sorted=False).indices
        keep = torch.zeros_like(norms, dtype=torch.bool).scatter_(1, idx, True)
        f = z * keep.repeat_interleave(self.b, dim=1).to(z.dtype)
        if return_active:
            return f, f.abs().sum(0) > 0, z
        return f

    def decode(self, f):
        return (f @ self.charts.transpose(0, 1)) + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat

    @property
    def decoder_weight(self):
        """(d, W) — the block-subspace surface every geometry function consumes."""
        return self.charts


def train_grassmann_bsf(X, b, seed, device, dict_size=W_DICT, r_active=R_ACTIVE, steps=4000,
                        lr=1e-4, batch=2048, warmup=500, qr_every=QR_EVERY, x_scale=1.0):
    """Train ONE Grassmannian BSF. Identical data/steps/batch/LR protocol to train_block_sae
    (the §5 confound control): only the featurizer class moves. Fail-loud on non-finite loss."""
    assert r_active % b == 0, f"r_active {r_active} not divisible by b={b}"
    N, d = X.shape
    k_blk = r_active // b
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ae = GrassmannBSF(d, dict_size, b, k_blk).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_schedule(warmup))
    idx_g = torch.Generator().manual_seed(seed + 1)
    last_loss = None
    for step in range(steps):
        x = X[torch.randint(0, N, (batch,), generator=idx_g)].to(device).float() * float(x_scale)
        if step == 0:
            ae.b_dec.data = _geometric_median(x).to(ae.b_dec.dtype)
            with torch.no_grad():
                # γ init: least-squares gain of the (non-orthogonal) sum of block projections,
                # so reconstruction of the average activation is unbiased at step zero (App. D).
                y = x - ae.b_dec
                f0 = ae.encode(x, use_threshold=False)          # γ=1 at init (log_gamma=0)
                Py = f0 @ ae.charts.transpose(0, 1)
                g0 = float((y * Py).sum() / Py.pow(2).sum().clamp_min(1e-12))
                assert g0 > 0, f"degenerate gamma init ({g0})"
                ae.log_gamma.fill_(math.log(g0))
        f, _active, _z = ae.encode(x, return_active=True, use_threshold=False)
        x_hat = ae.decode(f)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        assert torch.isfinite(loss), f"non-finite loss at step {step} (b={b}, seed={seed})"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        sched.step()
        if (step + 1) % qr_every == 0:
            with torch.no_grad():
                _batched_orthonormalize_(ae.charts.data, b)
        last_loss = float(loss.item())
    with torch.no_grad():                       # final retraction: charts leave training ON the manifold
        _batched_orthonormalize_(ae.charts.data, b)
        # A Grassmannian BSF has NO inference threshold (selection is a hard per-sample Π_ℓ). The
        # buffer is set positive only so `load_bank_any`'s "has a learned threshold" assert — which
        # guards the BatchTopK banks — still passes. It is never read by `encode`.
        ae.threshold.fill_(1.0)
    ae = ae.to("cpu").eval()
    cfg = {"kind": "grassmann", "m": int(b), "b": int(b), "k_blk": int(k_blk),
           "dict_size": int(dict_size), "activation_dim": int(d), "r_active": int(r_active),
           "steps": int(steps), "lr": float(lr), "batch": int(batch), "warmup": int(warmup),
           "seed": int(seed), "n_tokens": int(N), "final_loss": last_loss, "signed": True,
           "x_scale": float(x_scale), "qr_every": int(qr_every),
           "gamma": float(ae.gamma.item()), "threshold": float(ae.threshold.item())}
    return ae, cfg


# ===========================================================================
# √d input normalization (paper §3.1) + a unified bank loader.
# ===========================================================================
def sqrt_d_scale(X, d):
    """s = √d / E‖x‖ over the training rows — the paper's input normalization, as ONE scalar.
    Applied uniformly to every config, so it cannot confound the §5 factorial's moved variable."""
    X = torch.as_tensor(X)
    n = min(100_000, X.shape[0])
    mean_norm = float(X[:n].float().norm(dim=1).mean())
    assert mean_norm > 0, "zero-norm activation block"
    return math.sqrt(float(d)) / mean_norm


class ScaledAE(nn.Module):
    """Presents a featurizer trained on s·x in the ORIGINAL activation units, so every existing
    hook / union / alignment surface consumes it unchanged:  encode(x) := ae.encode(s·x),
    decode(f) := ae.decode(f)/s. Decoder DIRECTIONS are untouched (only their scale), so all
    subspace geometry (union_Q, block_alignment, stable_rank) is invariant to s."""

    def __init__(self, ae, x_scale):
        super().__init__()
        self.ae = ae
        self.x_scale = float(x_scale)
        assert self.x_scale > 0

    # --- structural passthroughs (the surfaces every geometry function reads) ---
    @property
    def m(self):
        return self.ae.m

    @property
    def n_blocks(self):
        return self.ae.n_blocks

    @property
    def k_blk(self):
        return self.ae.k_blk

    @property
    def dict_size(self):
        return self.ae.dict_size

    @property
    def threshold(self):
        return self.ae.threshold

    def block_norms(self, z):
        return self.ae.block_norms(z)

    def encode(self, x, **kw):
        return self.ae.encode(x * self.x_scale, **kw)

    def decode(self, f):
        return self.ae.decode(f) / self.x_scale

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


def decoder_matrix(ae):
    """The (d, W) column-dictionary of ANY featurizer in this program (BlockTopKSAE, GrassmannBSF,
    or a ScaledAE wrapping either) — the single surface the geometry code consumes."""
    inner = ae.ae if isinstance(ae, ScaledAE) else ae
    if isinstance(inner, GrassmannBSF):
        return inner.charts.detach()
    return inner.decoder.weight.detach()


def save_bank_any(ae, cfg, quality, path, data_manifest_sha, x_scale=1.0):
    """Bank a featurizer of either kind. `x_scale` must agree with the trainer's cfg — a mismatch
    would silently splice at the wrong scale, so it RAISES."""
    inner = ae.ae if isinstance(ae, ScaledAE) else ae
    if "x_scale" in cfg:
        assert abs(float(cfg["x_scale"]) - float(x_scale)) < 1e-9, \
            f"x_scale mismatch: cfg {cfg['x_scale']} vs arg {x_scale}"
    if isinstance(ae, ScaledAE):
        assert abs(ae.x_scale - float(x_scale)) < 1e-9, "ScaledAE wraps a different x_scale"
    payload = {"state_dict": inner.state_dict(), "cfg": {**cfg, "x_scale": float(x_scale)},
               "quality": quality, "data_manifest_sha": str(data_manifest_sha)}
    torch.save(payload, path)
    return payload


def load_bank_any(path):
    """Load a BlockTopKSAE or GrassmannBSF bank, re-wrapping the √d scale if one was banked.
    Returns (ae eval CPU — possibly ScaledAE, cfg, quality, manifest_sha)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    kind = cfg.get("kind", "blocktopk")
    if kind == "grassmann":
        inner = GrassmannBSF(cfg["activation_dim"], cfg["dict_size"], cfg["b"], cfg["k_blk"])
    elif kind == "blocktopk":
        from infra.pg.grain_bsf import BlockTopKSAE
        inner = BlockTopKSAE(cfg["activation_dim"], cfg["dict_size"], cfg["m"], cfg["k_blk"],
                             signed=cfg.get("signed", False))
    else:
        raise AssertionError(f"unknown bank kind {kind!r} at {path}")
    inner.load_state_dict(payload["state_dict"])
    inner.eval()
    assert float(inner.threshold.item()) > 0, f"banked dict has no learned threshold ({path})"
    s = float(cfg.get("x_scale", 1.0))
    ae = ScaledAE(inner, s).eval() if s != 1.0 else inner
    return ae, cfg, payload["quality"], payload["data_manifest_sha"]


# ===========================================================================
# MDL at a frozen distortion (paper Eq. 5-6). Selection + description; NEVER a gate.
# ===========================================================================
def rate_bits(variances, delta, smooth=True):
    """Gaussian rate-distortion code length of a spectrum {λ_i} at water level δ (paper Eq. 6):

        smooth (the paper's own evaluation):  Σ_i ½ log₂(1 + λ_i/δ)
        clipped (Eq. 6 as literally written):  Σ_i ½ (log₂(λ_i/δ))_+

    The paper: "We evaluate each water-filling term with the smooth Gaussian rate ½ log₂(1+σ²/δ),
    which agrees with the clipped form wherever σ² ≫ δ but removes the discontinuity an eigenvalue
    would otherwise introduce as it crosses δ." δ is the water level ITSELF, not a total budget."""
    lam = torch.as_tensor(variances, dtype=torch.float64).flatten()
    assert lam.numel() >= 1 and float(lam.min()) >= 0.0, "variances must be non-negative"
    assert float(delta) > 0, "water level δ must be positive"
    if smooth:
        return float((0.5 * torch.log2(1.0 + lam / float(delta))).sum())
    return float(torch.clamp(0.5 * torch.log2(lam / float(delta)), min=0.0).sum())


def log2_binom(n, k):
    """log₂ C(n, ℓ) via lgamma — the support cost of naming which ℓ of G blocks fired."""
    assert 0 <= k <= n, f"C({n},{k}) undefined"
    return float((math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
                 / math.log(2.0))


@torch.no_grad()
def mdl_spectra(ae, X, device="cpu", micro=2048, x_scale=1.0):
    """Collect Eq. 6's three empirical inputs in ONE chunked pass — never materializing the
    (N, W) code matrix (at N=20k, W=16384 that is 1.3GB in fp32):

      p_g            per-block firing rate                       (G,)
      σ²_gj          eigenvalues of cov(z_g | g active)          (G, b)
      λ_a            eigenvalues of cov(r), r = x − x̂            (d,)
      var_ref        tr(cov(x))/d, the distortion reference scale

    At b=1 the per-block covariance is the scalar conditional second moment, so its single
    "eigenvalue" is E[z_g² | active]. For b>1 we accumulate the b×b conditional Gram per block
    and eigendecompose it batched."""
    G, b, d = ae.n_blocks, ae.m, X.shape[1]
    n = X.shape[0]
    fired = torch.zeros(G, dtype=torch.float64)
    gram = torch.zeros(G, b, b, dtype=torch.float64)
    r_sum = torch.zeros(d, dtype=torch.float64)
    r_gram = torch.zeros(d, d, dtype=torch.float64)
    x_sq = torch.zeros(d, dtype=torch.float64)
    x_sum = torch.zeros(d, dtype=torch.float64)
    for s in range(0, n, micro):
        # MDL is computed in the featurizer's OWN (√d-normalized) units: the code, the residual
        # and var_ref must share a scale or the water level δ = frac·var_ref is meaningless.
        # Pass the INNER featurizer with the site's x_scale, never a ScaledAE.
        x = X[s:s + micro].to(device).float() * float(x_scale)
        f = ae.encode(x, use_threshold=True)
        rec = ae.decode(f)
        r = (x - rec).double().cpu()
        r_sum += r.sum(0)
        r_gram += r.transpose(0, 1) @ r
        xd = x.double().cpu()
        x_sum += xd.sum(0)
        x_sq += (xd ** 2).sum(0)
        zb = f.double().cpu().reshape(-1, G, b)                 # (n_i, G, b)
        act = zb.norm(dim=2) > 0                                # (n_i, G)
        fired += act.double().sum(0)
        gram += torch.einsum("ngi,ngj->gij", zb, zb)            # zeros where inactive
    assert float(fired.sum()) > 0, "no block ever fired"
    p_g = fired / n
    cnt = fired.clamp_min(1.0).reshape(G, 1, 1)
    code_eigs = torch.linalg.eigvalsh(gram / cnt).clamp_min(0.0)          # (G, b)
    r_mean = r_sum / n
    r_cov = r_gram / n - torch.outer(r_mean, r_mean)
    residual_eigs = torch.linalg.eigvalsh((r_cov + r_cov.T) / 2).clamp_min(0.0)
    var_ref = float((x_sq / n - (x_sum / n) ** 2).mean())
    return {"block_fire_rates": p_g, "block_code_eigs": code_eigs,
            "residual_eigs": residual_eigs, "var_ref": var_ref, "n_tokens": int(n)}


def grassmann_free_params(n_blocks, block_dim, d):
    """G·b·(d−b): the free parameters of G points on the Grassmannian Gr(b, d) (paper Eq. 6)."""
    return int(n_blocks) * int(block_dim) * (int(d) - int(block_dim))


def free_decoder_params(n_blocks, block_dim, d):
    """G·b·d: an untied/unconstrained decoder (the Vanilla / Block featurizer)."""
    return int(n_blocks) * int(block_dim) * int(d)


def mdl_bits_per_token(*, n_blocks, k_blk, n_tokens, block_fire_rates, block_code_eigs,
                       residual_eigs, delta, dict_free_params, smooth=True):
    """Description length in bits/token at a FROZEN distortion δ — the paper's Eq. 5 / Eq. 6:

        L_δ(x) = log₂C(G, ℓ)                                       [support: which blocks fired]
               + Σ_g p_g Σ_{j≤b} ½ log₂(1 + σ²_gj/δ)               [code, conditional on active]
               + Σ_{a≤d}        ½ log₂(1 + λ_a/δ)                  [residual — TRANSMITTED]
               + ½ · (free dictionary params) · log₂(N) / N        [dictionary, amortized]

      block_fire_rates  p_g, (G,)      — per-block firing rate; Σ_g p_g ≈ ℓ
      block_code_eigs   σ²_gj, (G, b)  — eigenvalues of cov(z_g | block g active)
      residual_eigs     λ_a, (d,)      — eigenvalues of cov(r), r = x − zD
      delta             δ = (1 − R²_target) · var_ref, var_ref = tr(cov(x))/d

    The residual is a TRANSMITTED term, not a distortion budget the residual may bust: every
    config is describable at every δ, and coarser reconstruction simply costs more residual bits.
    """
    p = torch.as_tensor(block_fire_rates, dtype=torch.float64).flatten()
    eigs = torch.as_tensor(block_code_eigs, dtype=torch.float64)
    assert eigs.ndim == 2 and eigs.shape[0] == p.numel(), \
        f"block_code_eigs {tuple(eigs.shape)} vs {p.numel()} blocks"
    assert float(delta) > 0 and int(n_tokens) > 1
    support = log2_binom(int(n_blocks), int(k_blk))
    per_block = (0.5 * torch.log2(1.0 + eigs / float(delta))).sum(dim=1) if smooth \
        else torch.clamp(0.5 * torch.log2(eigs / float(delta)), min=0.0).sum(dim=1)
    code = float((p * per_block).sum())
    residual = rate_bits(residual_eigs, delta, smooth=smooth)
    dictionary = 0.5 * float(dict_free_params) * math.log2(float(n_tokens)) / float(n_tokens)
    return {"total": support + code + residual + dictionary, "support": support, "code": code,
            "residual": residual, "dictionary": dictionary, "delta": float(delta),
            "expected_active_blocks": float(p.sum())}


# ===========================================================================
# Rotated-error null + the distortion-frontier corruptions.
# ===========================================================================
def haar_rotation(d, seed, device="cpu", dtype=torch.float32):
    """A seeded Haar-distributed orthogonal d×d matrix (QR of a Gaussian, sign-fixed so the
    distribution is exactly Haar). Exact: QᵀQ = I to fp round-off."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


def rotate_error(x, x_hat, Q, alpha=1.0):
    """x + α·Q(x̂ − x): at α=1 the per-token error NORM is identical (Q orthogonal) and the error
    DIRECTION destroyed — the §2 first-class control wherever a splice is judged. α scales the
    corruption for the distortion frontier's rotated-error family (Arm I3)."""
    e = (x_hat - x)
    return x + float(alpha) * (e @ Q.transpose(0, 1).to(e.dtype))


def r2_of(x, x_corrupt, x_mean):
    """Realized reconstruction R² = 1 − ‖x' − x‖² / ‖x − x̄‖² over the row block (fp64)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    xc = torch.as_tensor(x_corrupt, dtype=torch.float64)
    mu = torch.as_tensor(x_mean, dtype=torch.float64)
    num = float(((xc - x) ** 2).sum())
    den = float(((x - mu) ** 2).sum())
    assert den > 0, "degenerate activation block (zero variance)"
    return 1.0 - num / den


def quantize_step(x, step):
    """x_q = Δ·round(x/Δ) — the App. G.1 corruption family, verbatim ("we uniformly quantize the
    normalized patch activations per patch, x_q = Δ round(x/Δ), and sweep Δ"). A step-size sweep,
    NOT a min/max range quantizer: no clipping, and the sweep is scale-free once Δ is expressed
    in units of the activation RMS."""
    assert float(step) > 0
    return float(step) * torch.round(x / float(step))


# ---------------------------------------------------------------------------
# PG1 pre-hooks for the corruption families (residual ENTERING layers[L], all positions).
# ---------------------------------------------------------------------------
def make_pre_hook_rotated_splice(ae, Q, alpha=1.0):
    """The rotated-error NULL for a featurizer: splice x + α·Q(x̂(x) − x). α=1 is the null."""
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        flat = hs.reshape(-1, hs.shape[-1]).float()
        rec = ae.decode(ae.encode(flat, use_threshold=True))
        out = rotate_error(flat, rec, Q.to(flat.device, flat.dtype), alpha=alpha)
        return rebuild(out.reshape(hs.shape).to(hs.dtype))
    return hook


def make_pre_hook_quantize(step):
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        flat = hs.reshape(-1, hs.shape[-1]).float()
        return rebuild(quantize_step(flat, step).reshape(hs.shape).to(hs.dtype))
    return hook


def make_pre_hook_gaussian(sigma, seed):
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        flat = hs.reshape(-1, hs.shape[-1]).float()
        g = torch.Generator(device=flat.device).manual_seed(int(seed))
        out = flat + float(sigma) * torch.randn(flat.shape, generator=g, device=flat.device,
                                                dtype=flat.dtype)
        return rebuild(out.reshape(hs.shape).to(hs.dtype))
    return hook


@contextlib.contextmanager
def hook_at(layers, L, hook):
    h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    yield
    h.remove()


# ===========================================================================
# A2 — usage-matched random-atom unions (matched on the nuisance selection used).
# ===========================================================================
def usage_strata(scale, active_mask, n_strata=10):
    """Decile strata of log(scale) over ACTIVE atoms. `scale` = per-latent mean |activation|
    (grain_bsf.block_usage()['latent_mean_act']) — the quantity B*'s causal screening is
    confounded with. Returns (stratum_id per atom, int64, −1 for inactive)."""
    scale = torch.as_tensor(scale, dtype=torch.float64)
    active = torch.as_tensor(active_mask, dtype=torch.bool)
    assert scale.shape == active.shape and bool(active.any()), "bad usage inputs"
    out = torch.full(scale.shape, -1, dtype=torch.long)
    v = torch.log(scale[active].clamp_min(1e-12))
    qs = torch.quantile(v, torch.linspace(0, 1, n_strata + 1, dtype=torch.float64))
    # interior boundaries only → bucketize yields ids in [0, n_strata-1]
    out[active] = torch.bucketize(v, qs[1:-1].contiguous(), right=False)
    return out


def usage_matched_union(bstar, strata, seed, exclude=()):
    """One usage-matched random-atom union: for each B* atom draw a DIFFERENT atom from the same
    usage decile (without replacement). Fail-loud if a stratum cannot supply a partner."""
    g = torch.Generator().manual_seed(int(seed))
    taken = set(int(b) for b in bstar) | set(int(e) for e in exclude)
    picked = []
    for atom in bstar:
        s = int(strata[int(atom)])
        assert s >= 0, f"B* atom {atom} is inactive — cannot usage-match"
        pool = torch.nonzero(strata == s).flatten().tolist()
        pool = [p for p in pool if p not in taken]
        assert pool, f"usage stratum {s} exhausted (|B*|={len(bstar)})"
        pick = pool[int(torch.randint(len(pool), (1,), generator=g))]
        picked.append(pick)
        taken.add(pick)
    return picked


def uniform_random_union(active_atoms, size, seed, exclude=()):
    """The spec's count-matched uniform null (kept alongside the usage-matched one)."""
    g = torch.Generator().manual_seed(int(seed))
    pool = [int(a) for a in active_atoms if int(a) not in set(int(e) for e in exclude)]
    assert len(pool) >= size, f"active pool {len(pool)} < union size {size}"
    perm = torch.randperm(len(pool), generator=g)[:size]
    return [pool[i] for i in perm.tolist()]


# ===========================================================================
# A1 — canonicity-matched surrogate subspaces → the Jaccard null Arm D needs.
# ===========================================================================
def surrogate_at_canonicity(V, target_c, d, seed, tol=2e-3, max_iter=60):
    """A surrogate partner V' of the orthonormal draw V (d,k) whose NORMALIZED canonicity with V
    equals `target_c`. Construction: V'(θ) = orthonormalize(cosθ·V + sinθ·R), R a fixed random
    Gaussian block. c(θ) is monotone ↓ in θ ∈ (0, π/2] — bisected to `tol`.

    This is the null Arm D's label actually needs: it holds the SUBSPACE agreement fixed at the
    observed value and asks whether the ATOM sets still agree more than that alone implies."""
    V = torch.as_tensor(V, dtype=torch.float64)
    d_, k = V.shape
    assert d_ == d, f"V lives in R^{d_}, not R^{d}"
    g = torch.Generator().manual_seed(int(seed))
    R = torch.randn(d, k, generator=g, dtype=torch.float64)
    R = R - V @ (V.transpose(0, 1) @ R)                 # push R into V's complement
    R = orthonormalize(R)

    def c_of(theta):
        Vp = orthonormalize(math.cos(theta) * V + math.sin(theta) * R)
        o, _ = principal_angle_overlap(V, Vp)
        return normalized_canonicity(o, k, d), Vp

    lo, hi = 1e-6, math.pi / 2                            # c(lo)≈1, c(hi)≈0
    c_hi, V_hi = c_of(hi)
    assert c_hi <= target_c + 0.05, f"target_c {target_c} below the θ=π/2 floor {c_hi:.4f}"
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c, Vp = c_of(mid)
        best = (c, Vp)
        if abs(c - target_c) < tol:
            return Vp, c
        if c > target_c:                                  # too aligned → rotate further out
            lo = mid
        else:
            hi = mid
    return best[1], best[0]


def unit_decoder(decoder_weight):
    """Column-normalized fp64 decoder — hoisted out of the per-draw loop (it is a 2048×16384
    fp64 normalization; recomputing it inside a 64-draw null costs more than the null)."""
    Wt = torch.as_tensor(decoder_weight, dtype=torch.float64)
    return Wt / Wt.norm(dim=0, keepdim=True)


def top_aligned_atoms(decoder_weight, V, top_n, already_unit=False):
    """The atom-set readout Arm D compares across draws: the top_n unit-normed decoder columns by
    ‖u_jᵀ V‖² (the m=1 case of grain_bsf.block_alignment, without the per-block loop)."""
    U = torch.as_tensor(decoder_weight, dtype=torch.float64) if already_unit \
        else unit_decoder(decoder_weight)
    Vd = torch.as_tensor(V, dtype=torch.float64)
    score = (U.transpose(0, 1) @ Vd).pow(2).sum(1)
    return torch.topk(score, int(top_n)).indices.tolist()


def canonicity_matched_jaccard_null(decoder_weight, V, target_c, d, top_n, n_draws, seed):
    """The A1 null band: for each draw, build a surrogate partner V' of V at the OBSERVED
    canonicity and score Jaccard(top_aligned(V), top_aligned(V')). Returns (values, realized c's).

    `V` MUST be the same rank the atom sets are read from — canonicity and alignment have to be
    measured on one subspace, or the null is matched to a quantity the readout never used."""
    U = unit_decoder(decoder_weight)
    base = top_aligned_atoms(U, V, top_n, already_unit=True)
    vals = torch.empty(n_draws, dtype=torch.float64)
    cs = []
    for i in range(n_draws):
        Vp, c = surrogate_at_canonicity(V, target_c, d, seed * 7919 + i)
        vals[i] = jaccard(base, top_aligned_atoms(U, Vp, top_n, already_unit=True))
        cs.append(float(c))
    return vals, cs


# ===========================================================================
# Frozen adjudicators (spec §4 as amended, §5.5, §7 as amended).
# Every label reachable — enforced by the dry-run suite.
# ===========================================================================
C_LABELS = ("H_atom-handle", "H_atom-handle-nonselective", "H_capacity-artifact",
            "H_no-atom-handle")


def adjudicate_atom_handle(*, effect_bstar, effect_full, effect_em, effect_rand_uniform_mean,
                           effect_rand_usage_mean, gap_uniform_cilo, gap_usage_cilo, delta_C, rho,
                           bh_pass, dppl_bstar, dppl_full, dppl_rand_band_hi):
    """The frozen §4 table (A2 + A3 amended). Effects are RF drops (1 − RF, dead-floor convention).

      compact = effect(B*) ≥ ρ·effect(full)                       [NON-BINDING, reported]
                AND gap(uniform-random − B*) CI-lo > δ_C
                AND gap(usage-matched − B*) CI-lo > δ_C           [A2: the nuisance-matched null]
                AND the em-random span does NOT reproduce B* (within δ_C)
                AND BH survives
      compact ∧ Δppl(B*) ≤ Δppl(full) + rand_ppl_band  → H_atom-handle
      compact ∧ otherwise                              → H_atom-handle-nonselective   [A3]
      ¬compact ∧ EITHER random band craters (≥ρ·full)  → H_capacity-artifact
      else                                             → H_no-atom-handle
    """
    assert effect_full > 0, "full-read effect ≤ 0 — no dynamic range (UNRUN upstream, not here)"
    em_reproduces = effect_em >= effect_bstar - delta_C
    compact = ((effect_bstar >= rho * effect_full)
               and (gap_uniform_cilo > delta_C) and (gap_usage_cilo > delta_C)
               and (not em_reproduces) and bool(bh_pass))
    if compact:
        selective = dppl_bstar <= dppl_full + dppl_rand_band_hi
        return "H_atom-handle" if selective else "H_atom-handle-nonselective"
    if max(effect_rand_uniform_mean, effect_rand_usage_mean) >= rho * effect_full:
        return "H_capacity-artifact"
    return "H_no-atom-handle"


D_LABELS = ("H_basis-ambiguity", "H_subspace-explains", "H_genuinely-family")


def adjudicate_draws(*, jaccard_cilo, permutation_p95, surrogate_p95):
    """The frozen §7 rule (A1). Ordered, three reachable labels:

      J_cilo > surrogate_p95                    → H_basis-ambiguity   (atoms agree MORE than an
                                                                       equal-canonicity subspace)
      permutation_p95 < J_cilo ≤ surrogate_p95  → H_subspace-explains (shared span explains it)
      J_cilo ≤ permutation_p95                  → H_genuinely-family  (differ at every grain)
    """
    assert surrogate_p95 >= permutation_p95 - 1e-9, \
        (f"surrogate null {surrogate_p95} below the permutation null {permutation_p95} — a "
         "matched-canonicity partner cannot agree LESS than an independent one; check inputs")
    if jaccard_cilo > surrogate_p95:
        return "H_basis-ambiguity"
    if jaccard_cilo > permutation_p95:
        return "H_subspace-explains"
    return "H_genuinely-family"


I_LABELS = ("H_sign-fix", "H_class-boundary", "H_grain-cost")


def adjudicate_sign(*, signed_m4_pass, rectified_m4_pass, any_signed_small_b_pass):
    """The frozen §5.5 rule.
      signed-m4 passes ∧ rectified-m4 fails            → H_sign-fix (rectification was the cause)
      no signed b≤4 config passes at L0                → H_class-boundary
      else                                             → H_grain-cost (block grain costs behavior
                                                          for a reason the code's sign does not
                                                          explain)"""
    if signed_m4_pass and not rectified_m4_pass:
        return "H_sign-fix"
    if not any_signed_small_b_pass:
        return "H_class-boundary"
    return "H_grain-cost"


H_LABELS = ("H_mixture", "H_prior-mismatch", "H_stream-null")


def adjudicate_stream(*, pure_text_passes_text_gate, pure_text_fvu, mixed_fvu, fvu_tol=0.02):
    """The frozen §11.2 rule (amended pre-freeze, A7 — see below).

      pure-text dict passes the text splice gate at L14           → H_mixture
      it fails while reconstructing NO WORSE than the mixed dict  → H_prior-mismatch
      it fails AND reconstructs genuinely worse                   → H_stream-null (undertrained;
                                                                    the anatomy cannot adjudicate)

    A7. The first draft tested `|fvu_text − fvu_mixed| ≤ tol`, a two-sided band. But a dictionary
    trained on pure text should reconstruct text *better* than a 4:1-mixed one. Under the two-sided
    rule, the strongest possible evidence for the BSF authors' language-mismatch conjecture —
    "reconstructs its own stream better than the mixed dictionary does, and still destroys the
    behaviour" — would have been labelled `H_stream-null` ("undertrained"), the exact opposite of
    what it shows. The clause is one-sided: what H_prior-mismatch needs is that reconstruction is
    not the excuse, i.e. `fvu_text ≤ fvu_mixed + tol`.

    Caught by auditing this table BEFORE Arm H ran, after the em-span clause in
    `adjudicate_atom_handle` was found to saturate. Arm H was not frozen and had not executed.
    """
    if pure_text_passes_text_gate:
        return "H_mixture"
    if float(pure_text_fvu) <= float(mixed_fvu) + fvu_tol:
        return "H_prior-mismatch"
    return "H_stream-null"


# ===========================================================================
# Structure statistics (Arm C3 / Arm E): stable rank of an atom set + its Haar band.
# ===========================================================================
def stable_rank(M):
    """‖M‖²_F / ‖M‖²₂ — the H_tiling leg-1 statistic on the B* decoder matrix (d, |B*|)."""
    M = torch.as_tensor(M, dtype=torch.float64)
    fro2 = float((M ** 2).sum())
    assert fro2 > 0, "zero matrix has no stable rank"
    return fro2 / float(torch.linalg.svdvals(M)[0]) ** 2


def stable_rank_haar_band(d, n_cols, n_draws, seed):
    """Matched null for stable_rank: n_cols random UNIT columns in R^d (NOT an orthonormal basis —
    that would have stable rank exactly n_cols and make the band degenerate). The dictionary's
    decoder columns are unit-norm but not mutually orthogonal, so this is the right match.
    A generic such set in high d has stable rank ≈ n_cols; 'tiling' means a value FAR BELOW the
    band, which is reported beside the ≤8 rule rather than replacing it."""
    out = torch.empty(n_draws, dtype=torch.float64)
    for i in range(n_draws):
        g = torch.Generator().manual_seed(int(seed) * 104729 + i)
        M = torch.randn(d, int(n_cols), generator=g, dtype=torch.float64)
        M = M / M.norm(dim=0, keepdim=True)
        out[i] = stable_rank(M)
    return out


def participation_ratio(eigs):
    """(Σλ)² / Σλ² — the paper's C.4 triplet member. 1 for a rank-one block, b when the block's
    b dimensions are used evenly."""
    lam = torch.as_tensor(eigs, dtype=torch.float64).flatten()
    s2 = float((lam ** 2).sum())
    assert s2 > 0, "zero spectrum has no participation ratio"
    return float(lam.sum()) ** 2 / s2


def effective_rank(eigs):
    """exp(spectral entropy) of a PSD spectrum — the third C.4 statistic. 1 for rank-one."""
    lam = torch.as_tensor(eigs, dtype=torch.float64).flatten().clamp_min(0.0)
    tot = float(lam.sum())
    assert tot > 0, "zero spectrum has no effective rank"
    p = lam / tot
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def rank_triplet(eigs):
    """(stable rank, participation ratio, effective rank) of a per-block CODE covariance spectrum.
    stable rank of a covariance = Σλ / λ_max (the ‖·‖²_F/‖·‖²₂ of the underlying matrix)."""
    lam = torch.as_tensor(eigs, dtype=torch.float64).flatten().clamp_min(0.0)
    lmax = float(lam.max())
    assert lmax > 0, "zero spectrum"
    return {"stable_rank": float(lam.sum()) / lmax,
            "participation_ratio": participation_ratio(lam),
            "effective_rank": effective_rank(lam)}


@torch.no_grad()
def shuffled_null_rows(X, seed):
    """The App.-E-style MATCHED-NOISE control: shuffle every activation COORDINATE independently
    across tokens. Per-coordinate marginals (and hence the total variance) are preserved exactly;
    every cross-coordinate correlation — i.e. all the geometry — is destroyed. Rank statistics
    recomputed through the SAME trained featurizer on these rows are the floor a geometry claim
    must beat; they are not an analytic guess."""
    g = torch.Generator().manual_seed(int(seed))
    Y = X.clone()
    n = Y.shape[0]
    for j in range(Y.shape[1]):
        Y[:, j] = Y[torch.randperm(n, generator=g), j]
    return Y


def hungarian_seed_canonicity(W_a, W_b):
    """Mean |cosine| of a maximum-weight one-to-one matching between two dictionaries' unit
    decoder columns (the seed-canonicity statistic). Falls back to a greedy matching if scipy is
    absent — the Haar floor is computed the same way, so the comparison stays like-for-like."""
    A = unit_decoder(W_a)
    B = unit_decoder(W_b)
    C = (A.transpose(0, 1) @ B).abs()                       # (Wa, Wb)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-C.numpy())
        return float(C[r, c].mean())
    except ImportError:
        used = torch.zeros(C.shape[1], dtype=torch.bool)
        tot, k = 0.0, 0
        for i in range(C.shape[0]):
            row = C[i].clone()
            row[used] = -1.0
            j = int(row.argmax())
            used[j] = True
            tot += float(C[i, j])
            k += 1
        return tot / k


# ===========================================================================
# Planted-manifold toy (BSF App. F / the spec §3 signed-vs-rectified diagnostic), weight-free.
# ===========================================================================
def gen_planted_circle(d, n_samples, seed, noise=0.02):
    """A curved 1-D manifold (a circle) embedded in a random 2-plane of R^d. The paper's reading:
    a SIGNED code holds it in ONE 2-D block; a RECTIFIED code cannot (nonneg coefficients span
    only a <180° cone of a block's columns) and must SHATTER it across ≥3 blocks.
    Returns (X (n,d) fp32, U (d,2) fp64 orthonormal — the planted plane)."""
    g = torch.Generator().manual_seed(int(seed))
    U = orthonormalize(torch.randn(d, 2, generator=g, dtype=torch.float64))
    t = torch.rand(n_samples, generator=g, dtype=torch.float64) * (2 * math.pi)
    coords = torch.stack([torch.cos(t), torch.sin(t)], dim=1)              # (n, 2)
    X = coords @ U.transpose(0, 1)
    X += noise * torch.randn(n_samples, d, generator=g, dtype=torch.float64)
    return X.float(), U


def blocks_covering_usage(ae, X, frac=0.90, micro=4096):
    """How many blocks carry `frac` of all per-token block selections on X — the shatter statistic.

    Selection is the PER-ROW top-k_blk of the pre-selection block norms: threshold-free and
    batch-size-free. (The deployment `use_threshold=True` path would make this statistic depend
    on a learned scalar, and on an under-trained featurizer the -1.0 sentinel keeps EVERY block —
    the statistic would be identically n_blocks and the diagnostic vacuous.)"""
    counts = torch.zeros(ae.n_blocks, dtype=torch.float64)
    with torch.inference_mode():
        for s in range(0, X.shape[0], micro):
            x = X[s:s + micro].float()
            _f, _active, code = ae.encode(x, return_active=True, use_threshold=False)
            norms = ae.block_norms(code)                       # (n, n_blocks), pre-selection
            idx = norms.topk(ae.k_blk, dim=1).indices          # per-row, not batch-flattened
            counts += torch.bincount(idx.reshape(-1), minlength=ae.n_blocks).double()
    total = float(counts.sum())
    assert total > 0, "no block ever fired — degenerate featurizer"
    cum = torch.sort(counts, descending=True).values.cumsum(0) / total
    return int((cum >= frac).nonzero()[0].item()) + 1


def b_dec_offset_ratio(ae, X):
    """‖b_dec − mean(X)‖ / std-radius of X. The cone-restriction escape hatch: a RECTIFIED block's
    non-negative cone has apex at b_dec, so pushing b_dec away from the data widens the cone's
    angular coverage of the data without bound. ≈0 → apex inside the data (cone argument bites);
    ≫1 → apex outside (a single block can cover a curved manifold despite rectification)."""
    X = torch.as_tensor(X, dtype=torch.float64)
    mu = X.mean(0)
    radius = float((X - mu).norm(dim=1).mean())
    assert radius > 0, "degenerate data (zero radius)"
    inner = ae.ae if isinstance(ae, ScaledAE) else ae
    return float((inner.b_dec.detach().double() - mu).norm()) / radius
