"""STENCIL core — the fixed-basis adapter (pin the residual-facing factor to a policy basis,
train only the input factor), the 7 policy-basis builders, the composite forgetting score, the
iso-competence forgetting interpolation, the natural-write census wrapper, and `adjudicate_stencil`
(spec §6 frozen verdict order).

The single genuinely NEW component of STENCIL. Everything else — the adapter train loop, the Q/Σ_x
banks, every eval instrument, the paired bootstrap — reuses verified surfaces (`relearn`,
`comwrite_core`, `inherit_pg1`, `ration_core`, `text_eval`, `battery_tasks`, `specter_core`). The
only non-trivial new logic is the fixed-basis pin and the iso-gain interpolation; both are
weight-free and unit-testable on tiny tensors (the §9 dry-run gate).

Frozen conventions (spec §1.2/§2/§6; journal amendments A-STENCIL-1..6):
  * fixed-basis pin (output-side matrices o_proj/down_proj): ΔW = (α/r)·A·B with A = U_policy[:, :r]
    PINNED (a frozen buffer, out_d×r), B (r×in_d) trained from 0 ⇒ zero-delta at step 0. α=r ⇒
    scale 1 (reference_adapter_alpha_scale_convention). ΔW's column space = span(A) = span(U_policy)
    → the residual write is confined to the policy subspace (comwrite_core.energy_fraction == 1).
  * the ONE knob that differs across arms is U_policy; task/optimizer/schedule/rank/seed identical.
  * every null two-sided with a ceiling + a VOID branch (feedback_one_sided_thresholds).

Heavy imports inside functions so the module py_compiles on the control plane (no torch on the
laptop). No try/except hiding — a wrong assumption RAISES (CLAUDE.md).
"""
from __future__ import annotations

import hashlib

# The residual-writing matrices STENCIL adapts (spec §1.2): both write the d=2048 residual stream,
# so both take the SAME per-layer policy basis on their OUTPUT (residual-facing) side.
TARGET_SUFFIXES = ("o_proj", "down_proj")

# The 7 arms (spec §1.2). The first five are fixed-basis (a U_policy); the last two are references
# handled by the trainer (no pinned basis).
FIXED_ARMS = ("susceptibility", "energy", "write", "random", "susceptibility_shuffle", "energy_shuffle")
# which per-layer bank tensor each fixed arm's basis is drawn from ('random' is generated → no source)
ARM_SOURCE = {"susceptibility": "Q_evecs", "susceptibility_shuffle": "Q_evecs",
              "energy": "sigma_evecs", "energy_shuffle": "sigma_evecs",
              "write": "u_wr", "random": None}
REF_ARMS = ("free_lora", "full_ft")
ARMS = FIXED_ARMS + REF_ARMS

RANK_LADDER = (5, 10, 20, 50, 150, 350, 679)   # HANDLE ladder (spec §0)
SEED = 1337


# ===========================================================================
# Policy-basis builders — the (out_d, r) residual-facing basis for ONE layer.
# Only the pair-of-columns differs across arms; the pin/train/scale are identical.
# ===========================================================================
def _orthonormal_check(U, tol=1e-4):
    """Assert U (d, r) has orthonormal columns (the pin + energy_fraction==1 both assume it)."""
    import torch
    U = torch.as_tensor(U, dtype=torch.float64)
    r = U.shape[1]
    gram = U.transpose(0, 1) @ U
    err = float((gram - torch.eye(r, dtype=torch.float64)).abs().max())
    assert err < tol, f"policy basis not orthonormal: max|UᵀU − I| = {err:.2e}"
    return U


def random_orthonormal(d, r, seed):
    """A random orthonormal (d, r) basis (the H_subspace-agnostic null). QR of a seeded Gaussian."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(d, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :r].contiguous()


def build_policy_basis(name, layer_bank, rank, seed=SEED):
    """Return the (out_d=d, rank) residual-facing basis U_policy for one fixed-basis arm at one layer.

    layer_bank keys (per-layer, banked by the measure/bank job):
      Q_evecs   (d, d)  descending-eigenvalue Q eigenbasis (susceptibility / shuffle)
      sigma_evecs (d, d) descending Σ_x eigenbasis (energy)
      u_wr      (d, k)   consensus-write basis (write), k ≥ rank
      d         int      residual width (2048)
    Degenerate/rank-deficient input → the E16 successor rule: fall back to a random orthonormal basis
    for that cell and flag it (blind-safe completion), never silently proceed on a bad basis.
    """
    import torch
    d = int(layer_bank["d"])
    assert 1 <= rank <= d, f"rank {rank} outside [1, {d}]"

    if name == "susceptibility":
        U = torch.as_tensor(layer_bank["Q_evecs"], dtype=torch.float64)[:, :rank]
    elif name == "energy":
        U = torch.as_tensor(layer_bank["sigma_evecs"], dtype=torch.float64)[:, :rank]
    elif name == "write":
        Uw = torch.as_tensor(layer_bank["u_wr"], dtype=torch.float64)
        assert Uw.shape[1] >= rank, f"u_wr has {Uw.shape[1]} cols < rank {rank}"
        U = Uw[:, :rank]
    elif name == "random":
        U = random_orthonormal(d, rank, seed=seed)
    elif name == "susceptibility_shuffle":
        # random `rank` columns of Q's eigvecs (NOT the top-r) — the ORDERING null (spec §1.2/§5).
        Qv = torch.as_tensor(layer_bank["Q_evecs"], dtype=torch.float64)
        import numpy as np
        cols = np.random.default_rng(int(seed)).choice(Qv.shape[1], size=rank, replace=False)
        U = Qv[:, torch.as_tensor(np.sort(cols))]
    elif name == "energy_shuffle":
        # random `rank` columns of Σ_x's eigvecs (NOT the top-r) — the ENERGY ORDERING null (QUORUM
        # gate P3, A-QUORUM-7): holds "energy eigenbasis membership" fixed but destroys the top-r
        # HIGH-VARIANCE ordering. If n* collapses to n*_random, the variance ORDERING is load-bearing
        # and the energy result is not a coordinate/lattice artifact.
        Ev = torch.as_tensor(layer_bank["sigma_evecs"], dtype=torch.float64)
        import numpy as np
        cols = np.random.default_rng(int(seed)).choice(Ev.shape[1], size=rank, replace=False)
        U = Ev[:, torch.as_tensor(np.sort(cols))]
    else:
        raise ValueError(f"unknown fixed-basis arm {name!r} (∈ {FIXED_ARMS})")

    # Q/Σ_x/u_wr columns are eigenvectors/QR columns → orthonormal; verify (guards a bad bank).
    return _orthonormal_check(U).contiguous()


def basis_id(name, layer, rank, seed=SEED):
    """A stable id for a pinned basis (banked with the trained B so ΔW reconstructs exactly)."""
    return f"{name}|L{layer}|r{rank}|s{seed}"


# ===========================================================================
# Fixed-basis adapter — pin A = U_policy, train B (mirrors relearn.make_adapter's forward exactly).
# ===========================================================================
def make_fixed_adapter(base_linear, U_policy, rank, alpha):
    """Wrap a frozen nn.Linear as W·x + (alpha/rank)·A(B(x)) with **A = U_policy[:, :rank] PINNED**
    (a frozen buffer, out_d×rank) and **B trained from 0** (rank×in_d) ⇒ zero delta at step 0. The
    forward is byte-identical in shape/dtype handling to `relearn.make_adapter` (the fp32-master /
    per-forward-cast discipline); only A is frozen and set to the policy basis instead of random.
    Only B requires grad."""
    import torch
    import torch.nn as nn

    class FixedBasisAdapter(nn.Module):
        def __init__(self, base, U, r, a_scale):
            super().__init__()
            out_d, in_d = base.weight.shape
            r = min(r, out_d, in_d)
            assert U.shape == (out_d, r), f"U_policy {tuple(U.shape)} != ({out_d}, {r})"
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            dev = base.weight.device                             # place A/B on the model's device
            # A pinned: a buffer (excluded from the trainable set + optimizer, persisted for exact ΔW).
            self.register_buffer("A", U.to(torch.float32).to(dev).contiguous())
            self.B = nn.Parameter(torch.zeros(r, in_d, device=dev))   # trained; zero-delta at step 0
            self.scale = a_scale / r
            self.rank = r

        def forward(self, x):
            B = self.B.to(x.dtype).transpose(0, 1)               # (in_d, r)
            A = self.A.to(x.dtype).transpose(0, 1)               # (r, out_d)
            return self.base(x) + self.scale * ((x @ B) @ A)

    return FixedBasisAdapter(base_linear, U_policy, rank, alpha)


def fixed_target_names(lm):
    """The residual-writing targets (o_proj / down_proj) across the 18 decoder layers — the STENCIL
    scope (a strict subset of relearn.adapter_target_names; residual writers only)."""
    import torch.nn as nn
    names = sorted(n for n, m in lm.named_modules()
                   if isinstance(m, nn.Linear) and m.weight.ndim == 2
                   and n.rsplit(".", 1)[-1] in TARGET_SUFFIXES)
    assert names, "no o_proj/down_proj targets found"
    return names


def layer_of(name):
    """The decoder layer index embedded in a module name like '...layers.7.self_attn.o_proj'."""
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    assert m, f"cannot find a layer index in {name!r}"
    return int(m.group(1))


def attach_fixed_adapters(lm, arm, banks, rank, alpha, seed=SEED):
    """Replace every o_proj/down_proj with a fixed-basis adapter pinned to that LAYER's policy basis
    for `arm`. `banks` = {layer_index: layer_bank} (per-layer Q_evecs/sigma_evecs/u_wr/d). Freezes
    every existing param first; after attach, exactly the adapter B matrices are trainable.
    Returns {n_modules, trainable_params, frozen_params, names, arm, rank, alpha, basis_ids,
             degenerate:[names that fell back]}."""
    import torch
    assert arm in FIXED_ARMS, f"attach_fixed_adapters is for fixed-basis arms only, got {arm!r}"
    names = fixed_target_names(lm)
    for p in lm.parameters():
        p.requires_grad_(False)
    basis_ids, degenerate = {}, []
    for n in names:
        L = layer_of(n)
        assert L in banks, f"no bank for layer {L}"
        parent_name, _, leaf = n.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        base = getattr(parent, leaf)
        out_d, in_d = base.weight.shape
        r = min(rank, out_d, in_d)
        # per-module seed (deterministic; only random/shuffle arms consume it)
        sub_seed = int(hashlib.sha1(f"{seed}:{arm}:{n}".encode()).hexdigest()[:8], 16)
        # degenerate-cell guard (E16 successor, blind-safe): if the arm's SOURCE bank is non-finite
        # or too small at this rank, fall back to a random orthonormal basis + flag — one bad cell
        # never crashes the fleet, and the flag surfaces it (feedback_signals_that_cannot_fail).
        src_key = ARM_SOURCE[arm]
        src = None if src_key is None else torch.as_tensor(banks[L][src_key])
        degen = src is not None and (not bool(torch.isfinite(src).all()) or src.shape[1] < r)
        if degen:
            U = random_orthonormal(out_d, r, seed=sub_seed)
            degenerate.append(n)
        else:
            U = build_policy_basis(arm, banks[L], r, seed=sub_seed)
        setattr(parent, leaf, make_fixed_adapter(base, U, r, alpha))
        basis_ids[n] = basis_id(arm, L, r, seed=sub_seed)
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lm.parameters() if not p.requires_grad)
    return {"n_modules": len(names), "trainable_params": int(trainable),
            "frozen_params": int(frozen), "names": names, "arm": arm, "rank": rank,
            "alpha": alpha, "basis_ids": basis_ids, "degenerate": degenerate}


# ===========================================================================
# Composite forgetting score — weighted relative drop across the held-out battery.
# ===========================================================================
# Per-component polarity: +1 = higher-is-better (accuracy/CIDEr), -1 = lower-is-better (ppl).
DEFAULT_DIRECTIONS = {"mmlu_broad": +1, "vqav2": +1, "coco_cider": +1, "wikitext_ppl": -1}


def component_forgetting(base_val, adapted_val, direction):
    """Relative degradation of ONE battery component ∈ ℝ (positive = forgot).
    higher-is-better: (base − adapted)/|base|;  lower-is-better: (adapted − base)/|base|."""
    assert direction in (+1, -1)
    denom = abs(float(base_val))
    assert denom > 1e-12, f"degenerate base value {base_val} for forgetting normalization"
    if direction == +1:
        return (float(base_val) - float(adapted_val)) / denom
    return (float(adapted_val) - float(base_val)) / denom


def composite_forgetting(base_metrics, adapted_metrics, weights, directions=None):
    """Weighted composite forgetting over the held-out battery (spec §2). `*_metrics` are dicts
    {component: value}; `weights` {component: w ≥ 0} (frozen on discovery); `directions` polarity.
    Returns {composite, per_component:{c: forget_c}} — composite = Σ w_c·forget_c / Σ w_c."""
    directions = directions or DEFAULT_DIRECTIONS
    comps = sorted(weights)
    assert comps, "empty forgetting-composite weights"
    per = {}
    num, den = 0.0, 0.0
    for c in comps:
        assert c in base_metrics and c in adapted_metrics, f"missing battery component {c!r}"
        assert c in directions, f"no polarity for component {c!r}"
        fc = component_forgetting(base_metrics[c], adapted_metrics[c], directions[c])
        per[c] = fc
        num += float(weights[c]) * fc
        den += float(weights[c])
    assert den > 0, "all-zero composite weights"
    return {"composite": num / den, "per_component": per}


# ===========================================================================
# Iso-competence forgetting interpolation (spec §6, T5) — read forgetting at a common task-gain τ*
# along an arm's OWN (task-gain, forgetting) trajectory.
# ===========================================================================
def interp_forget_at_gain(gains, forgets, tau):
    """Forgetting at task-gain `tau`, linearly interpolated along one arm's own trajectory of
    (gain, forget) samples (checkpoints or ranks). Returns a float, or None if `tau` is outside the
    arm's achieved gain span (the arm cannot reach τ* → excluded from the paired primary, reported
    at its ceiling — spec §2 intercurrent events). Ties in gain → the max-forget branch is taken
    (worst-case, never optimistic). Fail-loud on empty/mismatched inputs."""
    import numpy as np
    g = np.asarray(gains, dtype=np.float64)
    f = np.asarray(forgets, dtype=np.float64)
    assert g.shape == f.shape and g.ndim == 1 and g.size >= 2, "need ≥2 aligned (gain, forget) samples"
    order = np.argsort(g)
    g, f = g[order], f[order]
    if tau < g[0] - 1e-12 or tau > g[-1] + 1e-12:
        return None                                          # outside the arm's reach
    # collapse ties conservatively (same gain → keep the largest forgetting)
    ug, uf = [], []
    i = 0
    while i < g.size:
        j = i
        while j + 1 < g.size and abs(g[j + 1] - g[i]) < 1e-12:
            j += 1
        ug.append(g[i]); uf.append(float(np.max(f[i:j + 1])))
        i = j + 1
    ug, uf = np.asarray(ug), np.asarray(uf)
    return float(np.interp(tau, ug, uf))


def common_tau(arm_gains, quantile=1.0):
    """The frozen τ* = a task-gain level reachable by ALL fixed-basis arms on discovery (spec §6):
    the min over arms of each arm's max achieved gain, scaled by `quantile` (1.0 = the common
    ceiling; discovery may pick a slightly lower fraction for interpolation headroom). `arm_gains`
    = {arm: [gains...]}. Fail-loud if any arm has no positive gain (that arm cannot define τ*)."""
    import numpy as np
    assert arm_gains, "no arms for τ* selection"
    maxes = []
    for a, gs in arm_gains.items():
        gs = np.asarray(gs, dtype=np.float64)
        assert gs.size > 0, f"arm {a!r} has no gain samples"
        maxes.append(float(gs.max()))
    return float(min(maxes) * float(quantile))


# ===========================================================================
# Natural-write census (Arm N) — where does an UNCONSTRAINED ΔW land? (CARRY + CONFLUX tie-in)
# ===========================================================================
def natural_write_census(dW, layer_bank, rank, mtype, seed=SEED):
    """Fraction of an unconstrained ΔW's Frobenius energy captured on its residual-facing side by
    each policy subspace at rank `rank` (comwrite_core.energy_fraction). Returns
    {susceptibility, energy, write, random, chance_floor}. `dW` (out_d, in_d); `mtype` ∈
    {'o_proj','down_proj'} (output side). This is the within-experiment prediction of which
    fixed-basis arm ≈ free-LoRA (the arm whose subspace captures the most natural-ΔW energy)."""
    from infra.pg import comwrite_core as CW
    d = int(layer_bank["d"])
    out = {}
    for name in ("susceptibility", "energy", "write", "random"):
        U = build_policy_basis(name, layer_bank, rank, seed=seed)
        out[name] = float(CW.energy_fraction(dW, U.numpy() if hasattr(U, "numpy") else U, mtype))
    out["chance_floor"] = CW.chance_floor(rank, d)           # Haar floor = rank/d
    return out


# ===========================================================================
# Contrast-range gate (T3) — U_sus vs U_energy must be well below identical span.
# ===========================================================================
def contrast_range(layer_bank, rank):
    """subspace_overlap(U_sus, U_energy) at `rank` (comwrite_core.subspace_overlap ∈ [0,1]; 1 =
    identical span, Haar floor = rank/d). RATION's Q⊥energy predicts this ≈ the floor, NOT 1. The
    in-run re-check of the load-bearing prior (spec §0, T3)."""
    from infra.pg import comwrite_core as CW
    U_sus = build_policy_basis("susceptibility", layer_bank, rank)
    U_en = build_policy_basis("energy", layer_bank, rank)
    ov = float(CW.subspace_overlap(U_sus.numpy() if hasattr(U_sus, "numpy") else U_sus,
                                   U_en.numpy() if hasattr(U_en, "numpy") else U_en))
    return {"overlap": ov, "haar_floor": rank / int(layer_bank["d"]), "rank": int(rank)}


# ===========================================================================
# adjudicate_stencil — the spec §6 frozen verdict order (applied ONCE on confirmation).
# ===========================================================================
def adjudicate_stencil(*, E_adapt, sus_vs_random, sus_vs_shuffle,
                       gain_range_ok, forget_range_ok,
                       random_saturates_frontier, free_strictly_dominates,
                       structure_beats_random, name="stencil"):
    """Frozen branch order (spec §6). Every guard two-sided (feedback_one_sided_thresholds).
      E_adapt        : paired-excess dict (Forgetting(energy) − Forgetting(susceptibility)); floor ci_lo>0.
      sus_vs_random  : paired-excess dict (Forgetting(random) − Forgetting(susceptibility)); floor ci_lo>0.
      sus_vs_shuffle : paired-excess dict (Forgetting(shuffle) − Forgetting(susceptibility)); floor ci_lo>0.
      gain_range_ok  : full-FT reached task-gain > δ_gain on the retained tasks (else no room to learn).
      forget_range_ok: unconstrained adaptation forgot > δ_forget on the battery (else no collateral room).
      random_saturates_frontier : random already ties full-FT's frontier (any subspace adapts identically).
      free_strictly_dominates   : free-LoRA/full-FT strictly dominate all fixed-basis AND no fixed-basis
                                   matches free-LoRA at lower rank (constraining the write helps nowhere).
      structure_beats_random    : some structured arm (sus OR energy) beats random at matched gain.
    Returns {verdict, reason, ...inputs}."""
    r = {"name": name,
         "E_adapt": E_adapt, "sus_vs_random": sus_vs_random, "sus_vs_shuffle": sus_vs_shuffle,
         "gain_range_ok": bool(gain_range_ok), "forget_range_ok": bool(forget_range_ok),
         "random_saturates_frontier": bool(random_saturates_frontier),
         "free_strictly_dominates": bool(free_strictly_dominates),
         "structure_beats_random": bool(structure_beats_random)}

    # 1. VOID-no-range — the benign boundary (a reported truth, not a halt): no room to move.
    if (not gain_range_ok) or (not forget_range_ok):
        r["verdict"] = "VOID-no-range"
        r["reason"] = ("no task-gain range (nothing to learn) and/or no forgetting range (the task "
                       "does not disturb pretraining) at deployable ranks on PG1 — the intervention "
                       "has no room to move; re-pick tasks (spec §1.4).")
        return r
    # 2. H_subspace-agnostic — random ties the structured arms (adaptation is subspace-blind).
    if random_saturates_frontier or not structure_beats_random:
        r["verdict"] = "H_subspace-agnostic"
        r["reason"] = ("random ties the structured subspaces at matched task-gain → adaptation is "
                       "subspace-indifferent; the map neither helps nor hurts (a structural fact).")
        return r
    # 3. H_free-dominates — constraining the write is the wrong lever entirely.
    if free_strictly_dominates:
        r["verdict"] = "H_free-dominates"
        r["reason"] = ("free-LoRA/full-FT strictly dominate every fixed-basis arm on the frontier and "
                       "no fixed-basis arm matches free-LoRA at lower rank → constraining the write "
                       "helps nowhere.")
        return r
    # 4/5. the decisive contrast (guards passed; structure beats random; free does not dominate).
    prescriptive = (E_adapt["ci_lo"] > 0 and sus_vs_random["ci_lo"] > 0 and sus_vs_shuffle["ci_lo"] > 0)
    if prescriptive:
        r["verdict"] = "H_prescriptive"
        r["reason"] = ("adapting in the susceptibility subspace forgets less at matched competence than "
                       "the energy axis (E_adapt ci_lo>0), beats random AND the ordering null "
                       "(shuffle) → the map PRESCRIBES where to adapt; the practical gate is earned.")
    else:
        r["verdict"] = "H_energy-suffices"
        r["reason"] = ("susceptibility ties/loses to energy at matched task-gain (E_adapt ci_lo≤0), or "
                       "its edge vanishes vs random/shuffle → the map is diagnostic-only even in its "
                       "own regime; the boundary of the two-regime law's reach is drawn (a full truth).")
    return r


# ===========================================================================
# Reference-arm attach paths (spec §1.2): free-LoRA (unconstrained low-rank, SAME o/down scope) and
# full-FT (unconstrained full-rank ceiling = the decoder blocks). Neither pins a policy basis.
# ===========================================================================
def attach_free_lora(lm, rank, alpha, seed=SEED):
    """free-LoRA reference: relearn's UNCONSTRAINED low-rank adapter (both A,B trained) on the SAME
    residual-writing scope (o_proj/down_proj) as the fixed-basis arms — a fair 'constrain nothing but
    the scope' reference. Freezes all params first; after attach, exactly the adapter A/B are
    trainable. Returns the attach info (mirrors relearn.attach_adapters' return)."""
    from infra.pg import relearn as RL
    names = fixed_target_names(lm)
    for p in lm.parameters():
        p.requires_grad_(False)
    for n in names:
        parent_name, _, leaf = n.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        base = getattr(parent, leaf)
        sub_seed = int(hashlib.sha1(f"{seed}:free_lora:{n}".encode()).hexdigest()[:8], 16)
        # relearn.make_adapter creates A/B on CPU — move the new adapter onto the model's device.
        setattr(parent, leaf, RL.make_adapter(base, rank, sub_seed, alpha).to(base.weight.device))
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    return {"n_modules": len(names), "trainable_params": int(trainable), "names": names,
            "arm": "free_lora", "rank": rank, "alpha": alpha}


def mark_full_ft_trainable(lm):
    """full-FT ceiling: unfreeze every weight INSIDE the text decoder blocks (name matches
    `layers.<int>.` and carries no vision/image marker) — the residual-writing pathway, a strict
    superset of the fixed-basis o/down scope. The vision tower, multimodal projector, embeddings and
    lm_head stay frozen (a clean text-adaptation ceiling matched to the intervention locus). Returns
    {trainable_params, n_param_tensors}."""
    import re
    for p in lm.parameters():
        p.requires_grad_(False)
    n_tensors = 0
    for name, p in lm.named_parameters():
        if re.search(r"layers\.\d+\.", name) and not any(v in name.lower() for v in ("vision", "image", "siglip")):
            p.requires_grad_(True)
            n_tensors += 1
    assert n_tensors > 0, "full-FT found no decoder-block params to unfreeze"
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    return {"trainable_params": int(trainable), "n_param_tensors": n_tensors}


def detach_adapters(lm):
    """Remove every fixed-basis / free-LoRA adapter, restoring each wrapped nn.Linear (`.base`, which
    was frozen and never modified). The clean way to reset the module tree between cells before a
    state_dict restore (full-FT is undone by the restore; adapters by this). Returns the count."""
    import torch.nn as nn
    replaced = 0
    for name, m in list(lm.named_modules()):
        if hasattr(m, "base") and isinstance(getattr(m, "base"), nn.Linear) and hasattr(m, "B"):
            parent_name, _, leaf = name.rpartition(".")
            parent = lm.get_submodule(parent_name) if parent_name else lm
            setattr(parent, leaf, m.base)
            replaced += 1
    return replaced


def attach_arm(lm, arm, banks, rank, alpha, seed=SEED):
    """Dispatch: attach the adapter/trainable-set for `arm`. Fixed arms pin a per-layer policy basis;
    free_lora is unconstrained low-rank at the o/down scope; full_ft unfreezes the decoder blocks.
    Returns the attach-info dict (always carries 'arm')."""
    if arm in FIXED_ARMS:
        return attach_fixed_adapters(lm, arm, banks, rank, alpha, seed=seed)
    if arm == "free_lora":
        return attach_free_lora(lm, rank, alpha, seed=seed)
    if arm == "full_ft":
        info = mark_full_ft_trainable(lm)
        info["arm"] = "full_ft"
        return info
    raise ValueError(f"unknown arm {arm!r} (∈ {ARMS})")


# ===========================================================================
# Trajectory trainer — relearn.train_adapter_qa's FROZEN optimizer/schedule/CE/seed with checkpoint
# evals (for the iso-gain interpolation, §6/T5) AND the PG1-correct forward.
# Amendment A-STENCIL-8: PG1 is a VLM — the in-distribution forward carries a BLANK gray image (256
# image tokens), IDENTICAL to the eval instrument `text_eval.mmlu_blankimg` (full processor, left-pad,
# logits_to_keep=1, CE on the single answer-letter token). relearn.train_adapter_qa's TEXT-ONLY
# forward is correct for its pure-LM host (gemma-2-2b) but is OOD for PG1 and would mismatch the
# eval forward. Everything else (AdamW + cosine-warmup, seed-determinism, CE target) is verbatim.
# ===========================================================================
def train_with_trajectory(model, proc, device, examples, *, lr, epochs, batch_size,
                          warmup_ratio, seed, max_steps=None, micro_batch=None,
                          checkpoint_steps=(), eval_fn=None, gray=128, post_step=None,
                          aux_loss_fn=None):
    """The relearn recipe (AdamW + cosine-warmup, CE on the answer-letter token, deterministic in
    `seed`) on the PG1 blank-image in-distribution forward (matches `mmlu_blankimg`), with an optional
    `eval_fn(model, step)->dict` called after the optimizer step at each `checkpoint_steps` step.
    Returns the relearn stats dict PLUS 'trajectory': [{step, **eval_fn(...)}].

    `post_step(model, step)->dict|None` (CLEAVE, A-CLEAVE-6) runs after EVERY optimizer step, before
    the next forward — the hook a constrained-training arm needs to project the iterate back onto its
    constraint set (AdamW's elementwise preconditioning does not preserve linear subspaces, so a
    gradient-only projection is not enough). Default None ⇒ this function's behaviour is byte-identical
    to STENCIL's; its returns are collected into 'post_step_log'.

    `aux_loss_fn(model)->scalar tensor` (FULCRUM Leg B) adds a WEIGHT-only regularizer to the task loss
    each optimizer step (once, after the CE micro-batches accumulate, before `opt.step()`): its gradient
    is `backward()`-ed into the params alongside the accumulated CE gradient. Default None ⇒
    byte-identical to STENCIL; the mean penalty value is returned in 'aux_loss_mean'."""
    import math

    import numpy as np
    import torch
    from PIL import Image

    from infra.pg import relearn as RL

    def _cosine_warmup(opt, n_warm, n_total):
        def f(s):
            if s < n_warm:
                return s / max(1, n_warm)
            p = (s - n_warm) / max(1, n_total - n_warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)

    tok = proc.tokenizer
    letters = RL.letter_token_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"                                   # -1 is the real last token (mmlu_blankimg)
    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    img_dtype = next(model.parameters()).dtype

    prompts = [e["prompt"] for e in examples]
    targets = [letters[e["answer_idx"]] for e in examples]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — attach an arm first"
    opt = torch.optim.AdamW(trainable, lr=lr)
    n_batches = len(RL.qa_train_step_batches(examples, batch_size))
    total_steps = max_steps if max_steps is not None else epochs * n_batches
    sched = _cosine_warmup(opt, int(warmup_ratio * total_steps), total_steps)
    g = torch.Generator().manual_seed(int(seed))
    cps = set(int(s) for s in checkpoint_steps)

    model.train()
    step, tokens_seen, loss_tokens = 0, 0, 0
    losses, trajectory, post_log, aux_losses = [], [], [], []
    done = False
    for _ep in range(epochs if max_steps is None else 10 ** 9):
        order = torch.randperm(len(examples), generator=g).tolist()
        for s, e in RL.qa_train_step_batches(examples, batch_size):
            idx = order[s:e]
            loss_tokens += len(idx)
            mb = micro_batch or len(idx)
            step_loss = 0.0
            for ms in range(0, len(idx), mb):
                midx = idx[ms:ms + mb]
                inp = proc(text=[prompts[i] for i in midx], images=[gray_img] * len(midx),
                           return_tensors="pt", padding="longest").to(device)
                inp["pixel_values"] = inp["pixel_values"].to(img_dtype)
                tgt = torch.tensor([targets[i] for i in midx], device=device)
                last = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                             pixel_values=inp["pixel_values"], logits_to_keep=1).logits[:, -1, :]
                loss = (torch.nn.functional.cross_entropy(last.float(), tgt)
                        * (len(midx) / len(idx)))
                assert torch.isfinite(loss), f"non-finite loss at step {step} — HALT"
                loss.backward()
                step_loss += float(loss.item())
                tokens_seen += int(inp["attention_mask"].sum().item())
            if aux_loss_fn is not None:               # FULCRUM Leg B: weight-only regularizer
                aux = aux_loss_fn(model)
                assert torch.isfinite(aux), f"non-finite aux loss at step {step} — HALT"
                aux.backward()
                aux_losses.append(float(aux.item()))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(step_loss)
            step += 1
            if post_step is not None:                 # A-CLEAVE-6: re-project the iterate
                pl = post_step(model, step)
                if pl is not None:
                    post_log.append({"step": step, **pl})
            if step in cps and eval_fn is not None:
                trajectory.append({"step": step, **eval_fn(model, step)})
                model.train()
            if max_steps is not None and step >= max_steps:
                done = True
                break
        if done:
            break
    model.eval()
    tok.padding_side = orig_side
    last_ep = losses[-n_batches:] if len(losses) >= n_batches else losses
    return {"steps": step, "final_loss": losses[-1] if losses else None,
            "mean_loss_last_epoch": sum(last_ep) / len(last_ep) if last_ep else None,
            "tokens_seen": tokens_seen, "loss_tokens": loss_tokens, "n_examples": len(examples),
            "trajectory": trajectory, "post_step_log": post_log,
            "aux_loss_mean": (sum(aux_losses) / len(aux_losses)) if aux_losses else None,
            "recipe": {"lr": lr, "epochs": epochs, "batch_size": batch_size,
                       "warmup_ratio": warmup_ratio, "seed": int(seed),
                       "max_steps": max_steps, "micro_batch": micro_batch}}
