"""M-A3 stage 1 — the settling diagnostic (queue A3). Wave 1.

THE MECHANISM, WRITTEN AS A MEASURABLE RATIO. Adam's update is m̂/(√v̂ + ε), where m̂ tracks the MEAN
gradient and v̂ the mean SQUARED gradient. For a coordinate whose per-step gradient has mean μ and
variance σ², the asymptotic ratio is

    |update| ~ |μ| / sqrt(σ² + μ²) = 1 / sqrt(1 + σ²/μ²)

so the single quantity that decides whether a coordinate moves is the NOISE-TO-SIGNAL RATIO
σ²/μ² — call it the settling ratio. A coordinate with a large gradient that agrees with itself has
a small ratio and moves; one with a large gradient that cancels has a large ratio and does not,
however large the gradient is. That is the gradient sink stated in one number, and it is what this
arm measures per coordinate.

    REFUTER (pre-logged): the settling diagnostic does not separate sink coordinates from twins.

Three things are recorded, because the ratio alone would not be a test:

  * the ratio PER ELEMENT of the weight matrix, collapsed to a per-input-coordinate number and
    contrasted between the MEASURED sink coordinates and the full twin battery — the separation is
    read against matched-random subspaces, never against zero.
    PER ELEMENT IS NOT A DETAIL. Adam's second moment is elementwise on the (out, in) matrix, so
    the ratio the mechanism is about is Var(g_ij)/mean(g_ij)^2 for each individual weight.
    Aggregating the gradient into a per-column SUM first and taking the ratio of THAT is a
    different quantity and a systematically flattering one: summing over `out` rows averages the
    per-element noise down by sqrt(out), so a column of noisy, cancelling elements looks
    consistent. The first run of this arm did exactly that (defect M-D18);
  * EPSILON SENSITIVITY. The derivation above drops Adam's ε. If √v̂ is comparable to ε on the
    suppressed coordinates, then ε — not the second moment — is what limits their motion, and the
    mechanism story would be about a hyperparameter. The arm measures where √v̂ sits relative to
    the standard ε grid instead of assuming it is negligible.
  * the SCALAR-DENOMINATOR counterfactual. Replacing the per-coordinate √v̂ with one scalar per
    matrix is what a normalized-SGD step does, and the ratio of the two predicted step sizes is
    how much of the suppression is attributable to the denominator being per-coordinate at all.

Stage 1 is measurement only — no training — so §0.2.7's exact-optimizer rule does not bind here;
the training arm that follows it does bind and says so.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-A3"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
# The ε grid Adam is actually run at in the wild; the arm reports where √v̂ sits against it.
EPS_GRID = (1e-8, 1e-7, 1e-6)


def settling_ratio(sum_g, sum_g2, n_steps: int):
    """σ²/μ² ELEMENTWISE, from the first two moments of the per-step gradient.

    `sum_g` is Σ_t g_t and `sum_g2` is Σ_t g_t², both with the WEIGHT MATRIX'S OWN SHAPE — Adam's
    second moment is elementwise, so the ratio must be too. An element whose mean gradient is
    exactly zero has an UNDEFINED ratio, not a large one; those are counted and masked rather than
    divided by.
    """
    a = np.asarray(sum_g, dtype=np.float64)
    b = np.asarray(sum_g2, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    assert n_steps >= 2, "a variance needs at least two steps"
    mu = a / n_steps
    var = np.maximum(b / n_steps - mu ** 2, 0.0)
    dead = mu == 0.0
    ratio = np.full(mu.shape, np.inf, dtype=np.float64)
    np.divide(var, mu ** 2, out=ratio, where=~dead)
    return {"ratio": ratio, "mean": mu, "var": var, "n_dead": int(dead.sum()),
            "n_coords": int(mu.size), "n_steps": int(n_steps)}


def per_coordinate(ratio_elementwise):
    """Collapse an elementwise ratio to one number per INPUT coordinate (the banks' axis).

    The MEDIAN down the `out` rows, not the mean: the elementwise ratio is heavy-tailed, because
    an element whose mean gradient is near zero produces an enormous ratio, and a mean over that is
    a statement about the tail rather than about the column.
    """
    import numpy as np
    R = np.asarray(ratio_elementwise, dtype=np.float64)
    assert R.ndim == 2, f"expected an (out, in) matrix, got {R.shape}"
    out = np.full(R.shape[1], np.inf, dtype=np.float64)
    for j in range(R.shape[1]):
        col = R[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            out[j] = float(np.median(finite))
    return out


def subspace_mean(values, U):
    """The leverage-weighted mean of a per-coordinate quantity inside a subspace.

    A subspace does not pick out coordinates; it weights them. Weighting by leverage
    ℓ_i = ‖U[i, :]‖² is the same convention every other per-coordinate statistic in this program
    uses, so a sink number and a twin number are commensurable.
    """
    v = np.asarray(values, dtype=np.float64)
    lev = H.leverage(U)
    assert lev.size == v.size, f"leverage {lev.size} != values {v.size}"
    finite = np.isfinite(v)
    w = lev * finite
    assert w.sum() > 0, "every coordinate in this subspace has an undefined ratio"
    return float((v[finite] * lev[finite]).sum() / w.sum())


def epsilon_position(sqrt_v, eps_grid=EPS_GRID) -> dict:
    """Where √v̂ sits relative to Adam's ε. If they are comparable, ε limits the motion, not v̂."""
    s = np.asarray(sqrt_v, dtype=np.float64)
    return {"sqrt_v_median": float(np.median(s)), "sqrt_v_p01": float(np.percentile(s, 1)),
            "sqrt_v_min": float(s.min()),
            "fraction_below_eps": {str(e): float((s < e).mean()) for e in eps_grid},
            "note": ("if a non-trivial fraction of the suppressed coordinates has sqrt(v) below "
                     "epsilon, the denominator that limits their motion is epsilon and the "
                     "mechanism claim would be about a hyperparameter")}


def main():
    ap = argparse.ArgumentParser(description="M-A3 stage 1 — the settling diagnostic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_a3", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    # Stage 1 trains nothing, and §0.2.7 binds the arm the moment it does. Recorded either way.
    M.check_optimizer(ARM, "none (measurement only)", offload_exact_fp32=True)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1
    from infra.aws_pg.jobs.mercator_w0_gpu import _target_keys, _windows

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    keys = _target_keys(model, types, layers)
    texts = _windows(processor, a.n_seq)
    blank = _blank_image(128)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    # Per-coordinate first and second moments of the COLUMN-summed gradient. The column sum is the
    # per-coordinate gradient the optimizer sees for that input dimension, aggregated over outputs.
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for k in keys.values():
        params[k].requires_grad_(True)
    # ELEMENTWISE accumulators, with the weight matrix's own shape. fp32 on the GPU because two
    # fp64 copies of every targeted matrix would not fit, and fp32 is ample for a ratio of second
    # moments — where fp64 sums of the WRONG statistic would not have been.
    s1 = {k: torch.zeros_like(params[k], dtype=torch.float32) for k in keys.values()}
    s2 = {k: torch.zeros_like(params[k], dtype=torch.float32) for k in keys.values()}
    n_steps = 0
    for i, txt in enumerate(texts):
        inp = processor(text=[txt], images=[blank], return_tensors="pt", padding=False).to("cuda")
        model.zero_grad(set_to_none=True)
        o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16), labels=inp["input_ids"])
        o.loss.backward()
        for k in keys.values():
            g = params[k].grad
            assert g is not None, f"{k} received no gradient"
            g32 = g.to(torch.float32)                     # elementwise, as Adam sees it
            s1[k] += g32
            s2[k] += g32 ** 2
        n_steps += 1
        if (i + 1) % 8 == 0:
            print(f"[M-A3] {i + 1}/{len(texts)} windows", flush=True)
    model.zero_grad(set_to_none=True)

    results, sep = {}, []
    for (t, L), key in sorted(keys.items()):
        st = settling_ratio(s1[key].to(torch.float64).cpu().numpy(),
                            s2[key].to(torch.float64).cpu().numpy(), n_steps)
        ratio_coord = per_coordinate(st["ratio"])
        # sqrt(v) per input coordinate, medianed down the column, so the epsilon comparison is
        # against the quantity Adam actually divides by.
        sqrt_v = np.median(np.sqrt(st["var"] + st["mean"] ** 2), axis=0)
        battery = {b: WT.load_bank(bank_dir, t, L, b) for b in FULL_BATTERY}
        cell = {}
        for b, U in battery.items():
            r = subspace_mean(ratio_coord, U)
            cell[b] = r
            at.add("settling_ratio", value=r, layer=L, type=t, bank_id=b, model_id=WT.PG1_REPO,
                   model_sha=base_sha, basis=M.BASIS_COORD,
                   flags={"n_steps": n_steps, "n_dead_elements": st["n_dead"],
                          "n_elements": st["n_coords"],
                          "definition": ("leverage-weighted mean over input coordinates of the "
                                         "column-median ELEMENTWISE Var(g)/mean(g)^2")})
        ep = epsilon_position(sqrt_v)
        at.add("sqrt_v_median", value=ep["sqrt_v_median"], layer=L, type=t,
               model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={**{k2: v2 for k2, v2 in ep.items() if k2 != "note"}})
        # The scalar-denominator counterfactual: one √v per matrix instead of per coordinate.
        scalar_v = float(np.sqrt(np.mean(st["var"] + st["mean"] ** 2)))
        mean_coord = np.median(np.abs(st["mean"]), axis=0)
        step_per_coord = mean_coord / (sqrt_v + EPS_GRID[0])
        step_scalar = mean_coord / (scalar_v + EPS_GRID[0])
        for b, U in battery.items():
            at.add("scalar_over_percoord_step", value=subspace_mean(step_scalar / step_per_coord, U),
                   layer=L, type=t, bank_id=b, model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"scalar_sqrt_v": scalar_v,
                          "meaning": "how much larger a normalized-SGD step would be here"})
        haar = [cell[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
        s = M.surprise(cell["core32"], float(np.mean(haar)), haar)
        sur.add(arm=ARM, expected_ref=expected_ref, cell=f"{t}/L{L}",
                statistic="settling_ratio/core32", s=s)
        sep.append(cell["core32"] / float(np.mean(haar)))
        results[f"{t}/L{L}"] = {"by_bank": cell, "epsilon": ep, "scalar_sqrt_v": scalar_v,
                                "core_over_haar": sep[-1], "n_dead_elements": st["n_dead"]}
        print(f"[M-A3] {t} L{L}: core/haar settling ratio = {sep[-1]:.3f}", flush=True)

    at.check_controls_per_statistic(["settling_ratio", "scalar_over_percoord_step"])
    gates.record("a3_settling_separates_core_from_twins", float(np.median(sep)) > 1.0,
                 median_core_over_haar=float(np.median(sep)), n_cells=len(sep),
                 note="the REFUTER is that the diagnostic does NOT separate them")

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "stage": "settling diagnostic", "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "base_repo": WT.PG1_REPO, "base_sha": base_sha, "types": types,
                  "layers": layers, "n_windows": n_steps, "cells": results,
                  "median_core_over_haar": float(np.median(sep))})
    print("A3_OK " + json.dumps({"rows": info["n_rows"],
                                 "median_core_over_haar": float(np.median(sep))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
