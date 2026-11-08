"""GESTALT Arm F — the flow-field instrument. Probe bank, field evaluation, the P5-exact
integrator with an injectable field, the ridge surrogate ladder, F1 statistics.
Spec: 2026-08-04_gestalt_program_spec.md §1.4/§2 Arm F; journal FR-3 (nulls) + FR-7 (gate).

THE OBJECT. pi0's action expert is a flow-matching vector field v(x, t | obs): lerobot
`PI0Pytorch.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)` (verified
against the lerobot-0.5.1 wheel; the installed version is asserted on-instance). `sample_actions`
integrates x <- x + dt * v with dt = -1/num_steps from t = 1.0 — the reference our own
integrator must reproduce EXACTLY (P5) before any swap is trusted.

FROZEN PROBE RULE. A probe point is (obs_i, x, t) with x = t*z_ij + (1-t)*a_i on the TRAINING
marginal: a_i = the dataset action chunk of obs_i (padded to max_action_dim, exactly
`prepare_action`), z_ij seeded N(0,1) draws, t in T_GRID. This is the distribution the
flow-matching loss itself probes (modeling_pi0.forward: x_t = t*noise + (1-t)*actions), it is
symmetric across models, and it needs no reference model. Field values are compared on the first
`original_action_dim` dims only (training truncates the loss there; padded dims are
unconstrained).

F1 STATISTIC. D_rel(A,B) = rms(vA - vB) / sqrt(0.5*(rms(vA)^2 + rms(vB)^2)) over (probe, chunk,
valid-dim); probe-bootstrap CI. Nulls (FR-3): (a) the CI/half-band must be nonzero; (b) the
within-model adjacent-step drift band max(D_rel(M@s, M@s-512) for M in {U3, U3c}).
H_field-develops at s: CI-lower of D_rel(U3c@s, U3@s) > null (b), with (a) healthy.

F2 SURROGATE LADDER (frozen): ridge v_hat = W phi(x, t, z) with feature blocks
  L0 [1] · L1 [1, t-knots] · L2 [1, t-knots, x] · L3 [+ z16] · L4 [per-knot A_t(x) + z64],
t mapped to the NEAREST knot in T_GRID; z = PCA of the masked-mean prefix embedding, PCA fit on
fit-half probes only. Behavioral gate FR-7: surrogate actions (10-step integration, same noise)
must sit within the action-MSE that the TRUE field loses when integrated at 5 steps vs 10.

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

N_OBS = 64
N_NOISE = 4
T_GRID = (0.9, 0.7, 0.5, 0.3, 0.1)
PROBE_SEED = 20260804
NUM_STEPS_REF = 10
NUM_STEPS_COARSE = 5
RIDGE_LAMBDA = 1e-3
Z_PCA_SEED = 11


# ===========================================================================
# Probe bank construction (manifest-hashed, deterministic).
# ===========================================================================
def probe_noise(n_obs, n_noise, chunk, adim, seed=PROBE_SEED):
    """The frozen z draws, (n_obs, n_noise, chunk, adim) fp32, CPU-deterministic."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(n_obs, n_noise, chunk, adim, generator=g, dtype=torch.float32)


def probe_points(a_ref, z, t):
    """x = t*z + (1-t)*a_ref (broadcast over the noise axis). a_ref (n_obs, chunk, adim),
    z (n_obs, n_noise, chunk, adim) -> (n_obs, n_noise, chunk, adim)."""
    return t * z + (1.0 - t) * a_ref.unsqueeze(1)


def manifest_sha(rows):
    """Content hash of the probe manifest (episode/frame/task identity, state+action bytes)."""
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r["episode_index"]).encode())
        h.update(str(r["frame_index"]).encode())
        h.update(str(r["task"]).encode())
        h.update(r["state"].numpy().tobytes())
        h.update(r["action"].numpy().tobytes())
    return h.hexdigest()[:16]


# ===========================================================================
# Policy bindings (thin; each fact asserted, not assumed).
# ===========================================================================
def load_pi0_policy(path_or_repo, device="cuda"):
    """PI0Policy.from_pretrained -> device, eval. Returns (policy, pre_processor)."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(path_or_repo).to(device).eval()
    pre, _post = make_pre_post_processors(policy.config, pretrained_path=path_or_repo)
    return policy, pre


def prefix_cache(policy, batch):
    """(prefix_pad_masks, past_key_values, state, z_obs) for one preprocessed batch — the obs
    side computed ONCE per probe observation, exactly as `sample_actions` does it."""
    import torch
    from lerobot.policies.pi0 import modeling_pi0 as MP
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = policy.prepare_state(batch)
    # version-agnostic mask helpers: 0.5.1 wheel has the method; the film-era git build has the
    # module-level function (both compute the identical additive 4-D layout)
    prep4d = (model._prepare_attention_masks_4d
              if hasattr(model, "_prepare_attention_masks_4d")
              else MP.prepare_attention_masks_4d)
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        att2d = MP.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att4d = prep4d(att2d)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_kv = model.paligemma_with_expert.forward(
            attention_mask=att4d, position_ids=pos, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
        m = prefix_pad_masks.unsqueeze(-1).to(prefix_embs.dtype)
        z_obs = (prefix_embs * m).sum(1) / m.sum(1).clamp(min=1.0)     # masked-mean (b, width)
    return prefix_pad_masks, past_kv, state, z_obs.float()


def field_eval(policy, prefix_pad_masks, past_kv, state, x_t, t_scalar):
    """v(x_t, t | obs) — one denoise_step, no integration. x_t (b, chunk, adim) fp32;
    returns fp32 (b, chunk, adim)."""
    import torch
    ts = torch.full((x_t.shape[0],), float(t_scalar), dtype=torch.float32, device=x_t.device)
    with torch.no_grad():
        v = policy.model.denoise_step(state=state, prefix_pad_masks=prefix_pad_masks,
                                      past_key_values=past_kv, x_t=x_t, timestep=ts)
    return v.float()


def integrate(policy, prefix_pad_masks, past_kv, state, noise, num_steps, field_fn=None):
    """x <- x + dt*v from t=1.0, dt=-1/num_steps — the sample_actions loop verbatim, with an
    injectable field. field_fn=None uses the TRUE field (P5 must show this path EQUALS
    `model.sample_actions` given the same noise). field_fn(x_t, t)->v swaps the surrogate in."""
    x_t = noise.clone()
    dt = -1.0 / num_steps
    for step in range(num_steps):
        t = 1.0 + step * dt
        if field_fn is None:
            v = field_eval(policy, prefix_pad_masks, past_kv, state, x_t, t)
        else:
            v = field_fn(x_t, t)
        x_t = x_t + dt * v
    return x_t


def expert_param_count(policy):
    """The action-expert parameter count (the Borges denominator): gemma_expert + the action/state
    projections + the action-time MLP. Counted from named_parameters, keys recorded."""
    roots = ("gemma_expert", "state_proj", "action_in_proj", "action_out_proj",
             "action_time_mlp_in", "action_time_mlp_out")
    n, keys = 0, []
    for k, p in policy.model.named_parameters():
        if any(r in k for r in roots):
            n += p.numel()
            keys.append(k)
    assert n > 0, "expert_param_count found no expert parameters — key layout drifted"
    return n, keys


# ===========================================================================
# F1 statistic (pure).
# ===========================================================================
def field_diff(vA, vB, adim_valid, n_boot=2000, seed=7, ref_null=None):
    """D_rel over (probe, chunk, :adim_valid) + probe-bootstrap CI. vA/vB (P, chunk, adim).
    `ref_null` (e.g. the FR-3 adjacent-step drift band): adds p_le_ref = P-hat(bootstrap
    D_rel <= ref_null) — the one-sided achieved significance for 'the contrast clears the
    drift band'. (Named ref_null, NOT ref — the body's per-probe normalizer is `ref`; the
    original name was shadowed and the offline gate caught it, defect D13.)"""
    import numpy as np
    a = np.asarray(vA, dtype=np.float64)[:, :, :adim_valid]
    b = np.asarray(vB, dtype=np.float64)[:, :, :adim_valid]
    assert a.shape == b.shape and a.ndim == 3, f"field shapes {a.shape} vs {b.shape}"
    per = ((a - b) ** 2).mean(axis=(1, 2))                     # (P,) per-probe MSE
    ref = 0.5 * ((a ** 2).mean(axis=(1, 2)) + (b ** 2).mean(axis=(1, 2)))

    def _d(idx):
        den = ref[idx].mean()
        return float(np.sqrt(per[idx].mean() / den)) if den > 0 else float("nan")

    import numpy.random as npr
    rng = npr.default_rng(seed)
    P = per.shape[0]
    point = _d(np.arange(P))
    draws = np.array([_d(rng.integers(0, P, P)) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    assert draws.size > 0.5 * n_boot, "field_diff bootstrap degenerate"
    out = {"d_rel": point, "lo95": float(np.percentile(draws, 2.5)),
           "hi95": float(np.percentile(draws, 97.5)), "n_probes": int(P),
           "rms_a": float(np.sqrt((a ** 2).mean())), "rms_b": float(np.sqrt((b ** 2).mean()))}
    if ref_null is not None:
        out["ref"] = float(ref_null)
        out["p_le_ref"] = float(((draws <= ref_null).sum() + 1) / (draws.size + 1))
    return out


# ===========================================================================
# F2 surrogate ladder (pure ridge; deterministic).
# ===========================================================================
def _nearest_knot(t):
    return min(range(len(T_GRID)), key=lambda i: abs(T_GRID[i] - t))


def features(x_flat, t, z, level):
    """One probe point's feature vector per ladder level. x_flat (Dx,), z (Dz,) numpy."""
    import numpy as np
    k = _nearest_knot(t)
    onehot = np.zeros(len(T_GRID))
    onehot[k] = 1.0
    if level == "L0":
        return np.ones(1)
    if level == "L1":
        return np.concatenate([np.ones(1), onehot])
    if level == "L2":
        return np.concatenate([np.ones(1), onehot, x_flat])
    if level == "L3":
        return np.concatenate([np.ones(1), onehot, x_flat, z[:16]])
    if level == "L4":
        per_knot = np.zeros(len(T_GRID) * x_flat.shape[0])
        per_knot[k * x_flat.shape[0]:(k + 1) * x_flat.shape[0]] = x_flat
        return np.concatenate([np.ones(1), onehot, per_knot, z[:64]])
    raise ValueError(f"unknown ladder level {level!r}")


LADDER = ("L0", "L1", "L2", "L3", "L4")


def fit_ridge(Phi, V, lam=RIDGE_LAMBDA):
    """W = argmin ||Phi W - V||^2 + lam||W||^2, closed form, fp64. Phi (N, F), V (N, Dv)."""
    import numpy as np
    Phi = np.asarray(Phi, np.float64)
    V = np.asarray(V, np.float64)
    F = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(F)
    W = np.linalg.solve(A, Phi.T @ V)
    assert np.isfinite(W).all(), "ridge solution non-finite"
    return W


def surrogate_params(level, Dx, Dv):
    """EXACT parameter count of a ladder rung's W (features x outputs)."""
    F = {"L0": 1, "L1": 1 + len(T_GRID), "L2": 1 + len(T_GRID) + Dx,
         "L3": 1 + len(T_GRID) + Dx + 16,
         "L4": 1 + len(T_GRID) + len(T_GRID) * Dx + 64}[level]
    return F * Dv


def pca_fit(Z, q=64):
    """(mean, components (q, D)) of the obs embeddings via exact eigh — deterministic, no RNG.
    Fit-half only."""
    import numpy as np
    Z = np.asarray(Z, np.float64)
    m = Z.mean(0)
    Zc = Z - m
    w, V = np.linalg.eigh(Zc.T @ Zc)
    order = np.argsort(w)[::-1][:q]
    return m, V[:, order].T


def pca_apply(z, mean, comps):
    import numpy as np
    return (np.asarray(z, np.float64) - mean) @ comps.T
