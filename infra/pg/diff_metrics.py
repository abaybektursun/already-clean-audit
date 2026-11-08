"""W9 Diff-SAE — the two pre-registered readouts (prereg §6) + control builders (§5).

  ρ_novel  (§6.1): fraction of Diff-SAE decoder directions that lie OUTSIDE the span of the
                   pre-existing (before) feature space, beyond the N3 random-delta floor, AND
                   confirmed absent in BEFORE by the §2.2 direct-activation cross-check.
                   Out-of-span energy is operationalised TWO ways (both reported):
                     (i)  projection onto the top-PCA subspace of BEFORE activations;
                     (ii) max cosine to any frozen Gemma-Scope base-feature decoder direction.

  R_KL    (§6.2): behavioural faithfulness. Pure KL math here; the patched after-model forward
                   that produces the logits lives in the job (a residual write-hook at L).

  Controls (§5): make_shuffled_delta (N4), plant_direction (positive control).

Discipline: fail-loud; effect sizes vs the N3/N4 FLOORS, never vs zero (prereg §7). Heavy
imports inside functions.
"""
from __future__ import annotations


# ===========================================================================
# ρ_novel — out-of-span energy of decoder directions.
# ===========================================================================

def before_pca_basis(acts_before, var_threshold=0.95, max_r=512, sample=20000, seed=1337):
    """Top-PCA subspace of BEFORE activations = the 'pre-existing feature space' (method i).
    Returns (U (d, r) orthonormal CPU tensor, info). r = smallest #components with cumulative
    explained variance ≥ var_threshold, capped at max_r.

    Computed via the (d×d) covariance eigendecomposition (eigh) on GPU when available — a full
    SVD of the (n×d) sample on CPU is ~O(n·d²) and far too slow (d=2304, n=20k)."""
    import numpy as np
    import torch
    N, d = acts_before.shape
    n = min(sample, N)
    sel = np.random.default_rng(seed).choice(N, n, replace=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = acts_before[sel].float().to(dev)
    X = X - X.mean(0, keepdim=True)
    C = (X.T @ X) / max(1, (n - 1))                        # (d, d) covariance
    evals, evecs = torch.linalg.eigh(C)                    # ascending eigenvalues
    evals = evals.flip(0).clamp_min(0.0)                   # descending
    evecs = evecs.flip(1)                                  # columns = PCs, descending
    cum = torch.cumsum(evals, 0) / evals.sum().clamp_min(1e-12)
    r = int(torch.searchsorted(cum, torch.tensor(var_threshold, device=dev)).item()) + 1
    r = max(1, min(r, max_r, d))
    U = evecs[:, :r].contiguous().cpu()                    # (d, r) orthonormal columns
    return U, {"r": int(r), "explained_var": float(cum[r-1].item()), "n_sample": int(n),
               "var_threshold": float(var_threshold), "max_r": int(max_r)}


def out_of_span_energy(dirs, U):
    """Out-of-span energy of each UNIT direction vs orthonormal basis U (d, r):
    1 − ‖Uᵀ d‖². dirs: (m, d) unit. Returns (m,) in [0,1] (1 = fully outside the before span)."""
    import torch
    proj = dirs @ U                                        # (m, r)
    in_span = (proj ** 2).sum(1).clamp(0.0, 1.0)
    return (1.0 - in_span)


def max_cosine_to_base(dirs, base_dirs, micro=2048):
    """Method (ii): max |cosine| of each Diff-SAE direction to ANY frozen Gemma-Scope base
    decoder direction. dirs (m, d) unit; base_dirs (B, d). Returns (m,) max-cos. novelty(ii) =
    1 − max_cos (a direction the base dictionary cannot point at is novel)."""
    import torch
    base = torch.nn.functional.normalize(base_dirs.float(), dim=1)     # (B, d)
    out = []
    for s in range(0, dirs.shape[0], micro):
        sims = dirs[s:s+micro].float() @ base.T                        # (micro, B)
        out.append(sims.abs().max(1).values)
    return torch.cat(out, 0)


def novel_flags(out_energy_real, out_energy_n3_floor, crosscheck_survive_mask):
    """A direction is NOVEL iff its out-of-span energy (method i) exceeds the 95th-percentile
    N3 random-delta floor AND it survived the §2.2 direct-activation cross-check (absent in
    before). Returns (bool mask, n_novel, rho_novel)."""
    import numpy as np
    e = np.asarray(out_energy_real, dtype=float)
    floor = float(out_energy_n3_floor)
    cc = np.asarray(crosscheck_survive_mask, dtype=bool)
    mask = (e > floor) & cc
    return mask, int(mask.sum()), float(mask.mean())


def n3_floor_pct95(out_energy_n3):
    """95th percentile of the N3 random-delta out-of-span-energy distribution (the floor a real
    direction must beat — prereg §6.1)."""
    import numpy as np
    return float(np.percentile(np.asarray(out_energy_n3, dtype=float), 95))


# ===========================================================================
# §2.2 direct-activation cross-check — reuse the frozen W4 estimator.
# ===========================================================================

def crosscheck_directions(dirs, acts_before, acts_after, n_perm=1000, seed=1337, fdr_q=0.05,
                          max_candidates=256):
    """Run the frozen delta_crosscoder.direct_activation_crosscheck on Diff-SAE decoder
    directions: a GENUINE novel direction activates on AFTER but NOT on BEFORE (low absence
    ratio, FDR-significant). Returns the full estimator dict + a per-candidate survive mask.
    dirs: (m, d) numpy/torch unit directions (already ranked by the caller; capped to
    max_candidates for permutation cost)."""
    import numpy as np
    from infra.pg.delta_crosscoder import direct_activation_crosscheck
    d = np.asarray(dirs, dtype=np.float32)[:max_candidates]
    cc = direct_activation_crosscheck(
        d, np.asarray(acts_before, dtype=np.float32), np.asarray(acts_after, dtype=np.float32),
        n_perm=n_perm, seed=seed, fdr_q=fdr_q)
    return cc


# ===========================================================================
# R_KL — behavioural faithfulness (pure math; patched forward is in the job).
# ===========================================================================

def mean_kl(logits_p, logits_q):
    """Mean per-position KL(P‖Q) where P=softmax(logits_p), Q=softmax(logits_q). logits_*:
    (n_pos, vocab) float. Uses log_softmax for stability. Returns a scalar (mean over positions)."""
    import torch
    lp = torch.log_softmax(logits_p.float(), dim=-1)
    lq = torch.log_softmax(logits_q.float(), dim=-1)
    p = lp.exp()
    kl = (p * (lp - lq)).sum(-1)                            # (n_pos,)
    return float(kl.mean().item())


def r_kl(kl_baseline_gap, kl_recovered):
    """R_KL = 1 − recovered / baseline_gap (prereg §6.2). baseline_gap = KL(true‖before-patched);
    recovered = KL(true‖(before+recon)-patched). R_KL→1 ⇒ the reconstruction restores the
    behavioural delta; R_KL→0 ⇒ it does not."""
    if kl_baseline_gap <= 0:
        return float("nan")
    return float(1.0 - (kl_recovered / kl_baseline_gap))


def r_kl_bootstrap_ci(kl_baseline_pos, kl_recovered_pos, n_boot=1000, seed=1337, q=(2.5, 97.5)):
    """Bootstrap CI for R_KL by resampling text positions jointly (paired: same resample indices
    for baseline + recovered, since both are measured at the same positions). kl_*_pos: per-text-
    position KL arrays (same length). Returns {point, lo, hi, n}."""
    import numpy as np
    b = np.asarray(kl_baseline_pos, dtype=float); r = np.asarray(kl_recovered_pos, dtype=float)
    n = b.size
    if n == 0 or b.mean() <= 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(n)}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bm = b[idx].mean()
        boots[i] = 1.0 - (r[idx].mean() / bm) if bm > 0 else np.nan
    lo, hi = np.nanpercentile(boots, q)
    return {"point": float(1.0 - r.mean() / b.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


# ===========================================================================
# Controls (§5).
# ===========================================================================

def make_shuffled_delta(acts_before, acts_after, seed=1337):
    """N4 shuffle null: break the before↔after token pairing (permute the after rows). The
    Diff-SAE trained on this delta must collapse ρ_novel/R_KL to the N3 floor (it fits position
    identity, not the rewrite)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(acts_after.shape[0], generator=g)
    return acts_after[perm] - acts_before


def plant_direction(acts_after, seed=1337, alpha=None, frac=0.5, mult=0.5):
    """Positive control (§5): inject a known synthetic unit direction u into a fraction `frac`
    of the AFTER activations (magnitude alpha = `mult` × per-token mean norm if None). The
    Diff-SAE MUST recover u — measured by `recovered_planted_subspace` (the span of a few SAE
    decoder features), not by a single decoder column (a BatchTopK SAE legitimately SPLITS a
    planted direction across several unit-norm columns). Returns (acts_after_planted, u, info)."""
    import torch
    N, d = acts_after.shape
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d, generator=g)
    u = u / u.norm()
    if alpha is None:
        alpha = float(acts_after.float().norm(dim=1).mean().item()) * mult
    planted = acts_after.clone().float()
    n_inject = int(frac * N)
    idx = torch.randperm(N, generator=g)[:n_inject]
    planted[idx] += alpha * u
    return planted, u, {"alpha": float(alpha), "n_inject": int(n_inject), "frac": float(frac)}


def recovered_planted_cosine(decoder_dirs, u):
    """Single-direction readout: max |cosine| between any Diff-SAE decoder direction and the
    planted direction u. Reported for transparency, but NOT the gate — a BatchTopK SAE splits a
    planted direction across several columns, so this UNDER-reads recovery (and degrades as the
    plant grows, since a bigger vector needs more columns). Use recovered_planted_subspace."""
    import torch
    dirs = torch.nn.functional.normalize(decoder_dirs.float(), dim=1)
    uu = (u / u.norm()).float()
    sims = (dirs @ uu).abs()
    return float(sims.max().item()), int(sims.argmax().item())


def recovered_planted_subspace(decoder_dirs, u, topm=8):
    """Positive-control GATE readout: how much of the planted unit direction u lies in the SPAN
    of the top-`topm` Diff-SAE decoder columns (by |cos to u|). Robust to feature-splitting — if
    the SAE represents u as a sum of a few unit-norm columns, their span still captures it.
    Returns (recovery in [0,1], top1_cos). recovery = ‖Qᵀu‖ where Q = orthonormal basis of the
    top-m columns. A genuinely-absent u gives ≈ sqrt(topm/d) ≈ 0.06 (the random floor); a
    recovered u (even split) gives ≈ 1. d=2304, topm=8 → trivial-span floor ~0.06, so the 0.8
    gate is meaningful (13× the floor)."""
    import torch
    dirs = torch.nn.functional.normalize(decoder_dirs.float(), dim=1)   # (M, d)
    uu = (u / u.norm()).float()                                         # (d,)
    cos = (dirs @ uu).abs()
    topm = min(topm, dirs.shape[0])
    idx = torch.topk(cos, topm).indices
    Q, _ = torch.linalg.qr(dirs[idx].T)                                 # (d, topm) orthonormal
    recovery = float(torch.linalg.norm(Q.T @ uu).item())
    return min(1.0, recovery), float(cos.max().item())


# ===========================================================================
# Bootstrap CI helper (prereg §7 — every reported number gets a CI).
# ===========================================================================

def bootstrap_ci(values, n_boot=1000, seed=1337, stat="mean", q=(2.5, 97.5)):
    """Nonparametric bootstrap CI of a per-feature/per-position statistic. values: 1-D array."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    fn = {"mean": np.mean, "median": np.median}[stat]
    boots = np.array([fn(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, q)
    return {"point": float(fn(v)), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


# ===========================================================================
# Discovery↔confirmation firewall (prereg §4 / §9-1c). ONE tested split point.
# ===========================================================================

def enforce_split(held, name="split", **strata):
    """Route EVERY discovery/confirmation carve through here. The "held tensor captured but never
    used" leak hit W9 v1 AND GraftLM E4 (two independent occurrences = systemic), so the split is
    factored into a single fail-loud utility instead of being re-inlined per job.

      held : 1-D bool tensor/array — True marks a SEALED confirmation token.
      **strata : optional named 1-D bool masks further stratifying the confirmation half
                 (e.g. text=~is_img → conf_text_idx). Each is AND-ed with `held`.

    The discovery half is the exact complement of `held` (disjoint + complete by construction);
    the only real failure mode — the one that bit W9 v1 / GraftLM E4 — is an EMPTY half (a held
    set carved but never used), so both halves are asserted non-empty. Returns disc_mask (bool),
    disc_idx, conf_idx, one conf_<name>_idx per stratum, and counts. FAILS LOUD; no silent path."""
    import numpy as np
    held_np = np.asarray(held).astype(bool).ravel()
    n = held_np.size
    disc_np = ~held_np
    n_conf = int(held_np.sum()); n_disc = int(disc_np.sum())
    assert n_conf > 0, f"[{name}] confirmation (held) half EMPTY — the sealed-half firewall is a no-op"
    assert n_disc > 0, f"[{name}] discovery half EMPTY — nothing to fit the dictionary on"
    out = {"disc_mask": disc_np, "disc_idx": np.where(disc_np)[0], "conf_idx": np.where(held_np)[0],
           "n": int(n), "n_disc": n_disc, "n_conf": n_conf}
    for sname, m in strata.items():
        m_np = np.asarray(m).astype(bool).ravel()
        assert m_np.size == n, f"[{name}] stratum '{sname}' size {m_np.size} != held size {n}"
        out[f"conf_{sname}_idx"] = np.where(held_np & m_np)[0]
    return out
