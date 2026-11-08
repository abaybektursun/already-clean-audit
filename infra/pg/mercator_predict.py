"""M-8 — the single-model sink predictor (queue item 8).

Question the instrument serves: coordinate INDICES carry no meaning across families, so only
PROPERTIES generalise — but if the mechanism is right, the sink's ADDRESS is computable from the
BASE MODEL ALONE, with no fine-tune zoo at all.

The mechanism, stated so it can be turned into a score. Adam's update is m̂/(√v̂ + ε). A coordinate
is SUPPRESSED — large gradient in, small motion out — exactly when its gradient is large in
magnitude but INCONSISTENT in direction across batches: then Σ_t g_t (which m̂ tracks) stays small
while Σ_t g_t² (which v̂ tracks) grows. So the predictor needs two per-coordinate statistics and
their product, not magnitude alone:

    mag_j  = mean_t ‖G_t[:, j]‖                                       how much gradient arrives
    cons_j = ‖Σ_t G_t[:, j]‖ / Σ_t ‖G_t[:, j]‖   ∈ [0, 1]             how much of it agrees
    score_j = mag_j · (1 − cons_j)                                    arrives and cancels

`score` is the arm's TREATMENT. Its two named baselines are in the same table (§3):

    activation magnitude  — the standard "outlier feature" address, and the baseline the REFUTER
                            is written against ("predictor <= activation-magnitude baseline");
    random                — the chance floor.

INDEPENDENCE (§7 threats row 'circularity'): every statistic here is computed from a BASE model's
own forward/backward passes. No fine-tune, no consensus Gram, no bank enters the prediction. The
measured banks are used ONLY to score the prediction afterwards, and `heldout_family_cell` keeps
one family entirely out of any threshold choice.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg import darwin_w0 as W0

# The house address-set sizes measured per family, recorded as a row so the open scaling puzzle
# (queue item 8: "39/2048, 46/2048, 108/3072, ~400/4096 — no clean law; mistral is an outlier")
# travels with the prediction rather than living in prose. Values are RECOMPUTED at preflight from
# the census artifacts; this tuple names the cells, never the numbers.
SPARSITY_CELLS = ("gemma-2b", "qwen2.5-3b", "llama-3.2-3b", "mistral-7b", "gemma-7b")


# ---------------------------------------------------------------------------
# Pure statistics (control-plane testable, no torch).
# ---------------------------------------------------------------------------
MAX_ZERO_GRADIENT_FRACTION = 0.5


def consistency(sum_grad_col_norm, sum_col_norm, *, return_counts: bool = False):
    """cons_j = ‖Σ_t G_t[:, j]‖ / Σ_t ‖G_t[:, j]‖, clipped into [0, 1].

    ZERO-GRADIENT COLUMNS ARE REAL. The model runs in bf16, so a column whose gradient is small
    enough underflows to exactly zero at every step, and then this ratio is 0/0 — undefined, not
    zero and not one. Such a column is set to consistency 0 and COUNTED, which is the reading that
    matches the score it feeds: `suppression_score` is mag x (1 - cons), and a column that receives
    no gradient has nothing to cancel and scores 0 through the magnitude factor regardless.

    A LARGE fraction of zero columns means the backward pass is broken rather than the model
    numerically quiet, so that still fails loud. Values above 1 mean the two accumulators saw
    different step counts, which is a bug and is raised rather than clipped away.
    """
    import numpy as np
    a = np.asarray(sum_grad_col_norm, dtype=np.float64)
    b = np.asarray(sum_col_norm, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    zero = b <= 0
    frac = float(zero.mean())
    assert frac <= MAX_ZERO_GRADIENT_FRACTION, (
        f"{zero.sum()}/{b.size} columns ({frac:.1%}) received zero gradient across every step — "
        f"above the {MAX_ZERO_GRADIENT_FRACTION:.0%} ceiling this is a broken backward pass, not "
        f"bf16 underflow")
    r = np.zeros_like(b)
    np.divide(a, b, out=r, where=~zero)
    assert r.max() <= 1.0 + 1e-9, (
        f"consistency {r.max():.6f} > 1 — the summed-gradient and summed-norm accumulators saw "
        f"different step counts")
    r = np.clip(r, 0.0, 1.0)
    return (r, {"n_zero_gradient_columns": int(zero.sum()), "n_columns": int(b.size),
                "zero_fraction": frac}) if return_counts else r


def suppression_score(mag, cons):
    """score_j = mag_j · (1 − cons_j). Large gradient that cancels."""
    import numpy as np
    m = np.asarray(mag, dtype=np.float64)
    c = np.asarray(cons, dtype=np.float64)
    assert m.shape == c.shape, f"shape mismatch {m.shape} vs {c.shape}"
    assert (m >= 0).all(), "negative gradient magnitude"
    return m * (1.0 - c)


def activation_baseline(act_rms):
    """The named incumbent address: per-coordinate activation magnitude (the 'outlier feature'
    list). Returned as a score in the same orientation as `suppression_score` (larger = more
    likely to be in the address) so both go through the identical ranking code."""
    import numpy as np
    a = np.asarray(act_rms, dtype=np.float64)
    assert (a >= 0).all(), "negative activation RMS"
    return a


def random_baseline(n: int, seed: int):
    """The chance floor, drawn once per cell from the cell seed."""
    import numpy as np
    return np.random.default_rng(seed).standard_normal(n)


def predict_address(score, k: int):
    """The predicted address: the top-k coordinates by score. Reuses DARWIN's fixed-k selector so
    the prediction and the measured bank are read with ONE implementation."""
    return W0.top_k_coords(score, k)


def measured_address(U_bank, k: int):
    """The measured address from a bank: the top-k coordinates by leverage ℓ_i = ‖U[i, :]‖².

    This is the ground truth the prediction is scored against, and it comes from the bank artifact
    — never from a number in prose (§0.22).
    """
    from infra.pg import humboldt_core as H
    return W0.top_k_coords(H.leverage(U_bank), k)


def score_prediction(pred_coords, true_coords, n_total: int) -> dict:
    """Overlap of a predicted address with the measured one, with its hypergeometric tail.

    Reuses DARWIN's `overlap_stats`, which returns the Jaccard, the intersection size and the
    log survival function under the null of independent draws — so 'how much overlap' and 'how
    surprising is that much overlap' come from one tested implementation.
    """
    return W0.overlap_stats(pred_coords, true_coords, n_total)


def compare_to_baselines(*, score, act_rms, grad_mag, U_bank, k: int, seed: int) -> dict:
    """The arm's decision table for ONE cell: treatment and THREE baselines, scored identically.

    `act_rms` is the ACTIVATION magnitude — the incumbent address the arm's REFUTER names
    ('predictor <= activation-magnitude baseline'). `grad_mag` is the per-coordinate GRADIENT
    magnitude, which is a strictly harder control and is included because the predictor is
    magnitude x (1 - consistency): without it, a win could be entirely the magnitude factor and
    the sign-consistency term could be doing nothing. Reporting only the named incumbent would
    hide that, so both are in the same table and the primary refuter is evaluated against the
    named one.

    No verdict is returned — that is §6's job, on the pooled cells.
    """
    import numpy as np
    n = int(np.asarray(score).size)
    truth = measured_address(U_bank, k)
    out = {}
    for name, s in (("predictor", score),
                    ("activation_magnitude", activation_baseline(act_rms)),
                    ("gradient_magnitude", activation_baseline(grad_mag)),
                    ("random", random_baseline(n, seed))):
        out[name] = score_prediction(predict_address(s, k), truth, n)
    out["k"] = int(k)
    out["n_total"] = n
    out["beats_activation_baseline"] = bool(
        out["predictor"]["jaccard"] > out["activation_magnitude"]["jaccard"])
    out["beats_gradient_magnitude_baseline"] = bool(
        out["predictor"]["jaccard"] > out["gradient_magnitude"]["jaccard"])
    return out


def heldout_family_cell(families, held: str) -> dict:
    """§7's circularity mitigation, made mechanical: name the family whose banks are excluded from
    every threshold/hyperparameter choice, and refuse a held-out cell that is not in the roster.

    There is exactly one free choice in this predictor — k, the address-set size — and it is fixed
    per cell by the MEASURED bank size of the OTHER families. Recording which family was held is
    what makes the held-out number an out-of-sample number.
    """
    fams = list(families)
    assert held in fams, f"held-out family {held!r} not in roster {fams}"
    assert len(fams) >= 2, "a held-out cell needs at least two families"
    return {"roster": fams, "held_out": held, "fit_on": [f for f in fams if f != held]}


def sparsity_scaling_row(measured: dict) -> list:
    """The open scaling puzzle as a table row per family: |address| vs d, and the ratio.

    `measured` maps family -> {"address_size": int, "d": int}, both READ FROM ARTIFACTS at
    preflight. This function records the ratio and refuses a family that is not in the roster; it
    does not fit a law, because the queue's own note is that there is no clean one.
    """
    rows = []
    for fam, m in sorted(measured.items()):
        assert fam in SPARSITY_CELLS, f"unknown family {fam!r} (roster {SPARSITY_CELLS})"
        a, d = int(m["address_size"]), int(m["d"])
        assert 0 < a <= d, f"{fam}: address size {a} out of range for d={d}"
        rows.append({"family": fam, "address_size": a, "d": d, "fraction": a / d})
    return rows


# ---------------------------------------------------------------------------
# The accumulator (GPU). One backward pass per micro-batch over a FROZEN batch of a base model.
# ---------------------------------------------------------------------------
def accumulate_grad_stats(model, batches, target_keys, *, device: str = "cuda"):
    """Per-column gradient statistics for every named weight, over a frozen batch sequence.

    Returns {key: {"mag": n-vector, "cons": n-vector, "n_steps": int}} where n is the matrix's
    INPUT dimension — the same coordinate axis every input-side bank lives on, so the prediction
    and the ground truth are indexed identically.

    Memory: one fp32 running-sum matrix per target key (≈ the size of those weights). The running
    sum is what makes `cons` computable at all: ‖Σ_t G_t[:, j]‖ is not recoverable from per-step
    column norms.

    `batches` is any iterable of dicts passed straight to the model; the caller freezes it, so the
    SAME tokens are used for every model in the roster and cross-family comparisons are not
    confounded by data.
    """
    import numpy as np
    import torch

    params = dict(model.named_parameters())
    missing = [k for k in target_keys if k not in params]
    assert not missing, f"target keys absent from the model: {missing[:5]}"
    for p in model.parameters():
        p.requires_grad_(False)
    for k in target_keys:
        params[k].requires_grad_(True)

    running_sum = {k: torch.zeros_like(params[k], dtype=torch.float32, device=device)
                   for k in target_keys}
    sum_norm = {k: torch.zeros(params[k].shape[1], dtype=torch.float64, device=device)
                for k in target_keys}
    n_steps = 0
    for batch in batches:
        model.zero_grad(set_to_none=True)
        out = model(**{k: v.to(device) for k, v in batch.items()})
        assert out.loss is not None, "model returned no loss — pass labels in the batch"
        out.loss.backward()
        for k in target_keys:
            g = params[k].grad
            assert g is not None, f"{k} received no gradient"
            g32 = g.to(torch.float32)
            running_sum[k] += g32
            sum_norm[k] += g32.to(torch.float64).pow(2).sum(dim=0).sqrt()
        n_steps += 1
    assert n_steps > 0, "empty batch iterable — the predictor would report a zero address"

    stats = {}
    for k in target_keys:
        s_col = running_sum[k].to(torch.float64).pow(2).sum(dim=0).sqrt().cpu().numpy()
        t_col = sum_norm[k].cpu().numpy()
        stats[k] = {"mag": t_col / n_steps,
                    "cons": consistency(s_col, t_col),
                    "n_steps": int(n_steps)}
        assert np.isfinite(stats[k]["mag"]).all(), f"{k}: non-finite gradient magnitude"
    model.zero_grad(set_to_none=True)
    return stats


def accumulate_activation_rms(model, batches, site_keys, *, device: str = "cuda"):
    """Per-coordinate activation RMS at the named module INPUTS — the activation-magnitude
    baseline's address, measured on the same frozen batches as the gradient statistics so the two
    are not confounded by data.
    """
    import numpy as np
    import torch

    mods = dict(model.named_modules())
    missing = [k for k in site_keys if k not in mods]
    assert not missing, f"site keys absent from the model: {missing[:5]}"
    acc = {k: None for k in site_keys}
    n = {k: 0 for k in site_keys}

    def hook(key):
        def f(_mod, inputs, _out):
            x = inputs[0].detach().to(torch.float64).reshape(-1, inputs[0].shape[-1])
            s = x.pow(2).sum(dim=0)
            acc[key] = s if acc[key] is None else acc[key] + s
            n[key] += int(x.shape[0])
        return f

    handles = [mods[k].register_forward_hook(hook(k)) for k in site_keys]
    with torch.no_grad():
        for batch in batches:
            model(**{k: v.to(device) for k, v in batch.items()})
    for h in handles:
        h.remove()

    out = {}
    for k in site_keys:
        assert acc[k] is not None and n[k] > 0, f"{k} was never invoked — no activations captured"
        out[k] = np.sqrt((acc[k] / n[k]).cpu().numpy())
        assert np.isfinite(out[k]).all(), f"{k}: non-finite activation RMS"
    return out
