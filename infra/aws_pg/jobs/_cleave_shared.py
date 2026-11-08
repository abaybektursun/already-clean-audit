"""CLEAVE shared job glue — the swap-shell install path (the PROTECT axis's whole machinery), the
natural full-FT trainer, the multimodal forgetting read, and the protective sweep.

Substrate is STENCIL's, verbatim (A-CLEAVE-1): the adaptation task is an MMLU subject trained through
the PG1-correct blank-image forward, and the collateral that matters is the MULTIMODAL pair
{VQAv2, COCO-CIDEr} (A-STENCIL-9: broad-MMLU *transfers* and text-ppl *improves* under MMLU
fine-tuning, so both mask rather than measure collateral). Everything about the roster, splits,
battery and recipe is imported from `_stencil_shared` rather than re-derived.

The NEW glue here is the swap-shell: extract the **language_model** decoder state-dict, project the
natural ΔW off a policy subspace with `comwrite_surgery.build_decoder("residual", …)`, install it, and
read (task-retention, forgetting). The `language_model` scoping is load-bearing and not cosmetic —
PaliGemma's SigLIP vision tower ALSO has `layers.N.self_attn.{q,k,v}_proj`, so a whole-model
state-dict would collide with the text decoder's grid keys inside `grid_of` and silently corrupt the
surgery. `carry_c.py` established this scoping; CLEAVE reuses it, leak-assert included.

Heavy imports inside functions so the module py_compiles on the control plane. No try/except (CLAUDE.md).
"""
from __future__ import annotations

from infra.aws_pg.jobs._stencil_shared import (  # noqa: F401  (substrate reused verbatim, A-CLEAVE-1)
    D,
    N_LAYERS,
    N_PPL_BATTERY,
    RECIPE,
    S3_BUCKET,
    SEED,
    build_roster,
    fetch,
    install_pg,
    load_model,
    make_battery_ctx,
    mmlu_by_subject,
    pinned_pg1_sha,
    reset_model,
    snapshot_base,
    splittable,
    subject_accuracy,
    subject_train_eval,
    train_examples,
)

# The forgetting composite CLEAVE removes: the multimodal collateral, equally weighted (A-STENCIL-9).
MM_COMPONENTS = ("vqav2", "coco_cider")
MM_WEIGHTS = {"vqav2": 1.0, "coco_cider": 1.0}

# Bounded battery for the per-rank sweep (the composite needs only the multimodal pair, so the
# broad-MMLU and wikitext reads are dropped from the inner loop and kept for the natural reference).
N_VQA_SWEEP = 64
N_COCO_SWEEP = 64

# A-CLEAVE-8. The natural update trains the 126-matrix grid ≈ **2.0B params** (the 16384-wide MLP
# dominates), so grads + AdamW state ≈ 12GB — and at STENCIL's micro_batch=8 the ACTIVATIONS blow up
# (8 × ~1000-token 5-shot prompts × a 16384-wide intermediate × 18 layers) and OOM a 48GB L40S.
# `micro_batch` is a pure MEMORY knob: `train_with_trajectory` scales each micro-loss by
# len(micro)/len(batch) and accumulates, so the gradient — and therefore the frozen recipe — is
# mathematically UNCHANGED. Only the memory/compute schedule differs, so this is not a recipe edit.
FT_RECIPE = {**RECIPE, "micro_batch": 2}
GRAD_MICRO_BATCH = 2


def fetch_s3(key, dst):
    """`fetch` with the bucket prefixed. `key` may be a bare `results/...` key or a full `s3://` URI.

    `_grain_shared.fetch` treats any non-`s3://` string as a **local path** and asserts it exists, so a
    bare key fails with a misleading "local artifact missing: results/..." — which is exactly how the
    first discovery launch died. Every CLEAVE job routes S3 reads through here so the prefix cannot be
    forgotten in one job and remembered in another."""
    uri = str(key) if str(key).startswith("s3://") else f"{S3_BUCKET}/{key}"
    return fetch(uri, dst)


# ===========================================================================
# The swap-shell — language-model scoping, install, and the vision-tower leak assert.
# ===========================================================================
def lm_bundle(model):
    """(lm_module, paths, vision_tower_snapshot) — the decoder scope every surgery targets.
    `vt_orig` is the leak reference: a surgery that touches the vision tower is measuring the wrong
    model (carry_c's assert, reused)."""
    from infra.pg.entities import resolve_submodule_paths
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    vt_orig = {k: v.detach().clone().cpu()
               for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
    return lm, paths, vt_orig


def decoder_sd(lm, device=None):
    """A clone of the language decoder's state-dict — `build_decoder`'s base_dec / ft_dec.

    `device=None` → a **CPU** clone (the original contract; the model itself lives on CUDA, so the
    `.to("cpu")` is what makes this a snapshot rather than an alias). `device="cuda"` keeps it beside
    the GPU so the surgery's projections run there (A-CLEAVE-9): at rank 2048 a single install is
    ~15 TFLOP (the 16384-wide MLP), i.e. minutes on 8 vCPU vs sub-second on an L40S. Identical
    arithmetic — `project_delta` casts to fp32 either way and follows its inputs' device. ~5GB per
    copy for PG1's language model.

    The default is CPU **explicitly**: dropping the `.cpu()` silently handed callers CUDA tensors,
    which then met CPU policy bases inside `build_decoder` and raised a device mismatch. Whatever
    device this returns, the caller must build its bases on the SAME one."""
    dev = device or "cpu"
    return {k: v.detach().to(dev).clone() for k, v in lm.state_dict().items()}


def assert_grid(base_lm_sd):
    """The 18×7 architecture firewall: exactly 126 decoder matrices, no vision leakage."""
    from infra.pg import comwrite_surgery as SURG
    grid = SURG.grid_of(base_lm_sd)
    SURG.assert_full_grid(grid, N_LAYERS)
    return grid


def make_installer(model, lm, paths, vt_orig):
    """install(cell_sd, tag) → load a surgered decoder state-dict, asserting a clean load AND that
    the vision tower is untouched (the multimodal-leak gate)."""
    import torch
    dev = next(model.parameters()).device

    def install(cell_sd, tag):
        missing, unexpected = lm.load_state_dict({k: v.to(dev) for k, v in cell_sd.items()},
                                                 strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt_now = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt_now[k].detach().cpu(), vt_orig[k]) for k in vt_orig), \
            f"{tag}: LEAK — vision tower changed"
    return install


def assert_identity_inert(base_lm_sd, banks, seed=SEED, device=None):
    """Harness-inertness (spec §9.1): the `identity` cell must be BITWISE base through the same
    pipeline the treatment uses. If the surgery is not inert, no contrast built on it means anything.

    `device` must match `base_lm_sd`'s: `build_decoder` computes `project_delta(dW, U, t)` BEFORE the
    cell branch (it books the captured energy for every cell, identity included), so a CPU basis
    against a CUDA state-dict raises on device mismatch."""
    import torch

    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    dummy_ft = {k: v.clone() for k, v in base_lm_sd.items()}
    bases = CL.off_bases(banks, "random", 5, seed=seed, n_layers=N_LAYERS, device=device)
    ident, _ = SURG.build_decoder("identity", base_lm_sd, dummy_ft, bases, n_layers=N_LAYERS)
    ok = all(torch.equal(ident[k], base_lm_sd[k]) for k in base_lm_sd)
    assert ok, "identity cell not bitwise base — the surgery pipeline is NOT inert"
    return "bitwise_base_ok"


# ===========================================================================
# Reads: task-gain (subject accuracy) + the multimodal forgetting composite.
# ===========================================================================
def mm_battery(model, proc, device, bctx):
    """The multimodal collateral read {vqav2, coco_cider} — the ONLY components in CLEAVE's
    forgetting composite (A-STENCIL-9). Identical items/order across base and every cell."""
    from infra.pg import inherit_pg1 as IP
    return {"vqav2": IP.eval_pg1(model, proc, bctx["vqa_task"],
                                 bctx["vqa_rows"][:bctx["n_vqa"]], device)["score"],
            "coco_cider": IP.eval_pg1(model, proc, bctx["coco_task"],
                                      bctx["coco_rows"][:bctx["n_coco"]], device)["score"]}


def mm_forgetting(base_mm, adapted_mm):
    """Composite forgetting over the multimodal pair (positive = forgot), via the STENCIL surface."""
    from infra.pg import stencil_core as SC
    return SC.composite_forgetting(base_mm, adapted_mm, MM_WEIGHTS)


def make_reader(model, proc, device, bctx, base_mm, base_subj_acc, eval_hashes):
    """read(tag) → {task_gain, subj_acc, forgetting, per_component, battery} on the CURRENT weights.
    The one read used for the base, the natural update, and every projected cell — identical items,
    identical order (the pairing contract)."""
    def read(tag):
        sa = subject_accuracy(model, proc, device, eval_hashes)
        mm = mm_battery(model, proc, device, bctx)
        comp = mm_forgetting(base_mm, mm)
        return {"tag": tag, "task_gain": sa["acc"] - base_subj_acc, "subj_acc": sa["acc"],
                "forgetting": comp["composite"], "per_component": comp["per_component"],
                "battery": mm}
    return read


# ===========================================================================
# The natural update — an unconstrained full-FT of the decoder blocks.
# ===========================================================================
def train_full_ft(model, proc, device, examples, *, recipe=None, seed=SEED):
    """The NATURAL update: unfreeze the 126 residual-facing projection matrices and fine-tune with
    STENCIL's frozen gentle recipe (A-STENCIL-9) through the PG1-correct blank-image forward
    (A-STENCIL-8). Returns (attach_info, train_stats); the caller snapshots the resulting decoder
    state-dict = ΔW's carrier.

    A-CLEAVE-7: the trainable set is `cleave_core.mark_grid_trainable`, NOT STENCIL's
    `mark_full_ft_trainable` — the latter also trains the per-layer layernorm gains, which
    `build_decoder` drops (non-target keys stay bitwise base), so the `ft` cell could never reproduce
    the trained model and the wiring gate would fail for a reason unrelated to the map. The natural
    update is defined exactly where the projection can act."""
    from infra.pg import cleave_core as CL
    from infra.pg import stencil_core as SC
    recipe = recipe or FT_RECIPE            # A-CLEAVE-8: micro_batch=2 (memory only, same gradient)
    info = CL.mark_grid_trainable(model)
    n_batches = max(1, (len(examples) + recipe["batch_size"] - 1) // recipe["batch_size"])
    total = recipe["epochs"] * n_batches
    stats = SC.train_with_trajectory(
        model, proc, device, examples, lr=recipe["lr"], epochs=recipe["epochs"],
        batch_size=recipe["batch_size"], warmup_ratio=recipe["warmup_ratio"], seed=seed,
        max_steps=total, micro_batch=recipe.get("micro_batch"), checkpoint_steps=(), eval_fn=None)
    return info, {**{k: stats[k] for k in ("steps", "final_loss", "mean_loss_last_epoch",
                                           "tokens_seen")}, "recipe": stats["recipe"]}


# ===========================================================================
# The PROTECT sweep — project the natural ΔW off each policy across the ladder, read the frontier.
# ===========================================================================
def protect_sweep(base_lm_sd, ft_lm_sd, banks, *, install, read, reset, policies, ranks,
                  gain_full, seed=SEED, log=print, device=None, support=None):
    """For each (policy, rank): install `ΔW_off = ΔW − proj_{U_P[:, :r]}ΔW` and read
    (retention, forgetting). Returns [{policy, rank, retention, forgetting, task_gain, …}].

    `gain_full` = the natural update's task-gain (the retention denominator, measured by the caller
    from the `ft` cell — NOT from the trained model in memory, so retention is defined against the
    same swap-shell pipeline every projected cell goes through).

    `reset()` is called before each install so a cell never inherits the previous cell's weights.
    Captured/injected energy per cell comes from `build_decoder`'s own bookkeeping — the projection's
    energy accounting is measured, not assumed."""
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    cells = []
    for policy in policies:
        for r in ranks:
            reset()
            # A-CLEAVE-11: the support is the layer set where THIS rank is certified — identical for
            # the treatment and every null, so the contrast stays matched. `support=None` (no
            # certification supplied) keeps the original whole-stack behaviour.
            if support is None:
                bases = CL.off_bases(banks, policy, int(r), seed=seed, n_layers=N_LAYERS,
                                     device=device)
                excluded, layers_used = (), list(range(N_LAYERS))
            else:
                layers_used = support.get(int(r), [])
                if not layers_used:
                    log(f"[protect] {policy:>22s} r={r:<5d} SKIPPED — no certified layers at this rank")
                    continue
                bases, excluded = CL.off_bases_restricted(banks, policy, int(r), layers_used,
                                                          seed=seed, n_layers=N_LAYERS, device=device)
            cell_sd, book = SURG.build_decoder("residual", base_lm_sd, ft_lm_sd, bases,
                                               excluded=excluded, n_layers=N_LAYERS)
            install(cell_sd, f"{policy}/r{r}")
            del cell_sd, bases
            res = read(f"{policy}/r{r}")
            removed = (book["total_captured_fro2"] / book["total_fro2"]) if book["total_fro2"] > 0 else 0.0
            cells.append({"policy": policy, "rank": int(r),
                          "retention": CL.retention(res["task_gain"], gain_full),
                          "forgetting": res["forgetting"], "task_gain": res["task_gain"],
                          "subj_acc": res["subj_acc"], "per_component": res["per_component"],
                          "removed_energy_fraction": removed,
                          "support_layers": layers_used, "n_support_layers": len(layers_used)})
            log(f"[protect] {policy:>22s} r={r:<5d} L={len(layers_used):>2d} "
                f"retention={cells[-1]['retention']:+.3f} forget={res['forgetting']:+.4f} "
                f"removed_E={removed:.3f}")
    return cells


def alpha_sweep(base_lm_sd, ft_lm_sd, banks, *, install, read, reset, policies, r_star, support,
                gain_full, alphas=None, seed=SEED, log=print, device=None):
    """The PROTECT frontier at a FIXED certified rank, swept by removal STRENGTH α (A-CLEAVE-12).

    For each policy: build the `shared` cell ONCE (the map's component of ΔW at r*, on the certified
    support), then trace α ∈ `alphas` by combining it with the `ft` cell — `W_off(α) = W_ft − α·(W_shared
    − W_base)`. α=0 is the full natural update for EVERY policy (a built-in identity check: all policies
    must coincide there), α=1 is full removal.

    This is cheaper than the rank sweep it replaces — one `build_decoder` per policy plus one shared
    `ft` cell, then pure arithmetic per α — and, unlike a rank sweep, every point sits at a rank where
    `U_sus` is certified-resolved."""
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    alphas = list(alphas or CL.ALPHA_LADDER)
    layers = list(support)
    assert layers, f"no certified support at r*={r_star}"

    # the `ft` cell is policy-independent — build it once
    bases0, excl0 = CL.off_bases_restricted(banks, "random", r_star, layers, seed=seed,
                                            n_layers=N_LAYERS, device=device)
    ft_cell, _ = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases0, excluded=excl0,
                                    n_layers=N_LAYERS)
    del bases0, excl0
    cells = []
    for policy in policies:
        bases, excluded = CL.off_bases_restricted(banks, policy, r_star, layers, seed=seed,
                                                  n_layers=N_LAYERS, device=device)
        shared_cell, book = SURG.build_decoder("shared", base_lm_sd, ft_lm_sd, bases,
                                               excluded=excluded, n_layers=N_LAYERS)
        del bases, excluded
        captured = (book["total_captured_fro2"] / book["total_fro2"]) if book["total_fro2"] > 0 else 0.0
        for a in alphas:
            reset()
            cell_sd = CL.combine_alpha(ft_cell, shared_cell, base_lm_sd, a)
            install(cell_sd, f"{policy}/a{a}")
            del cell_sd
            res = read(f"{policy}/a{a}")
            cells.append({"policy": policy, "rank": int(r_star), "alpha": float(a),
                          "retention": CL.retention(res["task_gain"], gain_full),
                          "forgetting": res["forgetting"], "task_gain": res["task_gain"],
                          "subj_acc": res["subj_acc"], "per_component": res["per_component"],
                          "captured_energy_fraction": captured,
                          "support_layers": layers, "n_support_layers": len(layers)})
            log(f"[alpha] {policy:>22s} a={a:<5.2f} retention={cells[-1]['retention']:+.3f} "
                f"forget={res['forgetting']:+.4f} capturedE={captured:.3f}")
        del shared_cell
    del ft_cell
    return cells


def install_natural(base_lm_sd, ft_lm_sd, banks, *, install, read, reset, seed=SEED, device=None):
    """The retention denominator + the WIRING GATE in one: install the full ΔW through the SAME
    `build_decoder` pipeline (`ft` cell) and read it. Its task-gain is `gain_full`; comparing it to
    the trained model's own read is the spec §9.1 wiring gate (install ≈ natural fine-tune)."""
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    reset()
    bases = CL.off_bases(banks, "random", 5, seed=seed, n_layers=N_LAYERS,
                         device=device)                             # unused by the ft cell
    cell_sd, book = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases, n_layers=N_LAYERS)
    install(cell_sd, "natural/ft")
    del cell_sd, bases
    res = read("natural/ft")
    return res, {"total_fro2": book["total_fro2"]}
