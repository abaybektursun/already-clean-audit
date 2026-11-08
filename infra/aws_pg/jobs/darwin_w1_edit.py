"""DARWIN W1-L / W1-M — durable edits and the reset button. GPU.

  W1-L  Durable edits (§4/W1-L, product 7.2). Place rank-1 closed-form edits into core vs highway
        vs twin coordinates, fine-tune ON TOP, and measure which placements SURVIVE. REFUTER: the
        core placement survives no better than its twins.

  W1-M  Reset-button fine-tune (§4/W1-M, product 7.3). Fine-tune with the update CONFINED to the
        highway subspace, then subtract the coefficients and check the model came back: weight
        distance to base, battery perplexity, and whether the task skill actually left. The named
        incumbent is LoRA-delete, which returns exactly by construction and therefore sets the bar
        that matters — not "does it return" but "does it return at a cost worth paying".

WHY THESE TWO SHARE A MODULE. Both are the same experiment run in opposite directions: W1-L asks
what a subspace makes DURABLE against later training, W1-M asks what it makes REMOVABLE. Sharing
the fine-tune harness means "durable" and "removable" are measured against the same optimizer, the
same task, and the same battery, so the two answers are comparable rather than merely adjacent.

W1-L's editor (§4/W1-L names A5's editor-availability pivot: "if no editor supports gemma within
4h, implement the closed-form rank-1 insert ourselves — it is NEW code with a dry-run gate"). That
is what happens here. The insert is the exact least-squares rank-1 update

    W' = W + (v* - W k) kᵀ / ‖k‖²      =>      W' k = v*   exactly, and W' x = W x for x ⟂ k

with the key k PROJECTED into the placement subspace first, so the edit is placed in that
subspace by construction rather than by hope. `edit_exactness` is recorded on every row: an edit
that did not land is not a datum about durability.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w1_edit --args "--arm W1-M" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

EDIT_TYPE = "o_proj"          # edited cell type: 2048-dim input space, same frame as the banks
RANK_LADDER = (8, 32, 128)    # W1-M's highway restriction ranks. Ratio 4 throughout, which
                              # `check_dose_grid` requires; (4, 32, 128) is ratios 8 and 4
                              # and fails it. This is also §4/W1-D's rank ladder verbatim,
                              # so the two surgery arms are priced on the same grid.
N_EDITS = 10                  # §4/W1-L verbatim
SPEC_EXPECTATIONS = {
    "W1-L": ("core-placed edits survive fine-tuning no better than twin-placed edits",
             "edits placed in core coordinates survive a subsequent fine-tune better than the "
             "same edits placed in highway or twin coordinates"),
    "W1-M": ("coefficient subtraction does not return the model to base, or the task skill "
             "survives the reset",
             "highway-restricted fine-tuning is exactly reversible by coefficient subtraction, "
             "returning both the weights and the battery while removing the task skill"),
}


def least_squares_edit(Wm, K, Vstar):
    """dW = (V* - W K) K⁺ — the JOINT constrained least-squares placement of all edits at once.

    (W + dW) K = W K + (V* - W K) K⁺K = V*, exactly, whenever K has full column rank.

    Not ten independent rank-1 inserts summed, and the difference is not cosmetic. Each rank-1
    insert `(v*_i - W k_i) k_iᵀ/‖k_i‖²` is exact for its OWN key and non-zero on every key not
    orthogonal to it. This arm deliberately places every key inside ONE 32-dim subspace, so ten
    random keys there have pairwise overlap ~1/sqrt(32) ~ 0.18 and each edit corrupts the other
    nine. Summing them measured 20% residual error before a single training step — an "edit
    durability" readout that was mostly edit interference. §4/W1-L says "placed via constrained
    least-squares", and this is what that means.
    """
    import numpy as np
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(Vstar, dtype=np.float64)
    assert K.ndim == 2 and V.shape[1] == K.shape[1], (K.shape, V.shape)
    r = np.linalg.matrix_rank(K)
    assert r == K.shape[1], (
        f"edit keys are rank-deficient ({r} < {K.shape[1]}): the joint solve cannot satisfy all "
        f"of them, and reporting durability on unsatisfiable edits would be meaningless")
    return (V - Wm @ K) @ np.linalg.pinv(K)


def place_key(k, U):
    """The key, projected into the placement subspace: k <- U Uᵀ k, renormalized to ‖k‖."""
    import numpy as np
    k = np.asarray(k, dtype=np.float64)
    p = U @ (U.T @ k)
    nrm = np.linalg.norm(p)
    assert nrm > 1e-8, "key has no component in the placement subspace"
    return p * (np.linalg.norm(k) / nrm)


def edit_batch(rng, Wm, U, n_edits, scale):
    """`n_edits` edits placed in U by ONE least-squares solve, and their (k, v*, base_out) triples.

    Keys are drawn in the ambient space and then projected, so every placement sees keys with the
    SAME ambient statistics — a placement that drew its keys inside its own subspace would be
    comparing subspaces on different data.

    THE TARGET OFFSET IS SIZED BY THE MATRIX, NOT BY ‖W k‖. `bottomsigma32` is by construction the
    base's LOWEST-GAIN input subspace, so ‖W k‖ there is tiny. Sizing the edit as `scale * ‖W k‖`
    installs a microscopic edit in exactly one bank and a large one in the others, and normalising
    the readout by ‖v*‖ then divides by that same tiny number — measured as a 9.5e-2 "failure to
    install" in bottomsigma32 against 4e-3 elsewhere, from bf16 rounding alone. The offset is
    instead `scale * rms(W) * ‖k‖` in a random direction, so edits are the same SIZE in every bank
    and durability is a statement about the subspace rather than about its gain.
    """
    import numpy as np
    m, n = Wm.shape
    assert n_edits <= U.shape[1], (
        f"{n_edits} edits cannot be placed independently in a {U.shape[1]}-dim subspace")
    rms_out = float(np.linalg.norm(Wm)) / (n ** 0.5)          # E‖W x‖ / ‖x‖ for generic x
    K = np.empty((n, n_edits))
    V = np.empty((m, n_edits))
    base = np.empty((m, n_edits))
    for i in range(n_edits):
        k = place_key(rng.standard_normal(n), U)
        u = rng.standard_normal(m)
        u /= np.linalg.norm(u)
        K[:, i] = k
        base[:, i] = Wm @ k
        V[:, i] = base[:, i] + scale * rms_out * float(np.linalg.norm(k)) * u
    dW = least_squares_edit(Wm, K, V)
    pairs = [(K[:, i], V[:, i], base[:, i]) for i in range(n_edits)]
    return dW, pairs


def edit_recovery(Wm, pairs):
    """Mean ‖W k - v*‖ / ‖v* - W0 k‖ — error as a FRACTION OF THE EDIT, not of the output.

    0 means the edit holds exactly; 1 means the cell has returned to where it was before the edit.
    Normalising by the edit's own magnitude is what makes the number comparable across banks whose
    base gains differ by orders of magnitude.
    """
    import numpy as np
    errs = []
    for k, v, base_out in pairs:
        mag = max(float(np.linalg.norm(v - base_out)), 1e-12)
        errs.append(float(np.linalg.norm(Wm @ k - v) / mag))
    return float(np.mean(errs)), errs


def run_w1_l(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha, ctx):
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _ppl_blankimg
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows

    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seed=a.seed)
    train_batches = build_batches(processor, train_rows, "cuda")
    eval_batches = build_batches(processor, eval_rows, "cuda")
    mods = T.trainable_modules(layers, types=(EDIT_TYPE,), layers=ctx["layers"])
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    base_ppl = ctx["base_ppl"]

    seen, results = set(), {}
    for bid in D.bank_ids_for(EDIT_TYPE):
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        pairs_by_cell, applied = {}, 0
        for (t, L), m in mods.items():
            Wm = pristine[(t, L)].cpu().numpy().astype(np.float64)
            U = W.load_bank(bank_dir, t, L, bid)
            dW, pairs = edit_batch(np.random.default_rng(H.cell_seed(a.seed, bid, L)), Wm, U,
                                   a.n_edits, a.edit_scale)
            m.weight.data.copy_(torch.as_tensor(Wm + dW, dtype=m.weight.dtype,
                                                device=m.weight.device))
            pairs_by_cell[(t, L)] = pairs
            applied += len(pairs)

        # Gate 1: did the edit land at all? A durability number computed on an edit that was never
        # installed is a statistic about floating point, not about the subspace.
        pre = {}
        for (t, L), m in mods.items():
            cur = m.weight.detach().to(torch.float32).cpu().numpy().astype(np.float64)
            pre[(t, L)] = edit_recovery(cur, pairs_by_cell[(t, L)])[0]
        exactness = float(np.mean(list(pre.values())))
        ppl_edited = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        atlas.add("edit_exactness", value=exactness, bank_id=bid, dose=float(a.edit_scale),
                  incumbent_id="none", n=applied, model_id=PG1_REPO, model_sha=base_sha,
                  flags={"n_edits": a.n_edits, "per_cell": {f"{k[0]}_L{k[1]}": v
                                                            for k, v in pre.items()},
                         "ppl_edited": ppl_edited, "ppl_base": base_ppl,
                         "battery_cost": ppl_edited - base_ppl,
                         "weight_dtype": "bfloat16",
                         "dtype_floor_note": "bf16 storage sets the floor on this readout (~0.004/edit_scale "
                                             "of the edit); the algebra is exact in fp64",
                         "exactness_max": a.exactness_max,
                         "definition": "mean ‖W k - v*‖ / ‖v* - W0 k‖ immediately after "
                                       "installation — error as a fraction of the EDIT, so banks "
                                       "with different base gains are on one scale"})
        # The floor here is the WEIGHT DTYPE, not the algebra. The insert is exact in fp64, but
        # the model stores bf16 (~8 mantissa bits), so reading the edit back costs ~4e-3 relative
        # error no matter how correct the update is. A 1e-3 gate would therefore fail every
        # placement for a reason that has nothing to do with the science. The gate is set well
        # above the dtype floor and well below "did not install", and the measured value is on the
        # row so a reader can see which regime it sits in.
        assert exactness < a.exactness_max, (
            f"{bid}: rank-1 edits did not install (mean relative error {exactness:.3e} >= "
            f"{a.exactness_max}) — the durability readout would be measuring nothing. The bf16 "
            f"storage floor is ~4e-3; this is far above it.")

        # Fine-tune ON TOP, identically for every placement.
        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        for m in mods.values():
            m.weight.requires_grad_(True)
            params.append(m.weight)
        opt = T.ExactAdam(params, lr=a.lr)
        D.check_optimizer("W1-L", "exact_fp32_adam")
        model.train()
        for step in range(a.steps):
            b = train_batches[step % len(train_batches)]
            loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()

        post = {}
        for (t, L), m in mods.items():
            cur = m.weight.detach().to(torch.float32).cpu().numpy().astype(np.float64)
            post[(t, L)] = edit_recovery(cur, pairs_by_cell[(t, L)])[0]
        survival = float(np.mean(list(post.values())))
        ppl_ft = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        tl = eval_task_loss(model, eval_batches)
        atlas.add("edit_survival_error", value=survival, bank_id=bid, dose=float(a.edit_scale),
                  incumbent_id="twin_placement", n=applied, model_id=PG1_REPO,
                  model_sha=base_sha,
                  flags={"exactness_before_ft": exactness, "steps": a.steps, "lr": a.lr,
                         "ppl_after_ft": ppl_ft, "ppl_base": base_ppl, "task_loss": tl,
                         "per_cell": {f"{k[0]}_L{k[1]}": v for k, v in post.items()},
                         "lower_is_more_durable": True,
                         "definition": "mean ‖W k - v*‖ / ‖v* - W0 k‖ after the fine-tune; 0 = the "
                                       "edit held, 1 = the cell returned to its pre-edit output"})
        results[bid] = {"exactness": exactness, "survival_error": survival, "ppl_ft": ppl_ft,
                        "task_loss": tl}
        seen.add(bid)
        print(f"[W1-L] {bid}: exactness={exactness:.3e} survival_err={survival:.4f} "
              f"ppl={ppl_ft:.4f}", flush=True)
        del opt
        torch.cuda.empty_cache()
        D.s3_sync(out, a.s3_prefix)

    for k, m in mods.items():
        m.weight.data.copy_(pristine[k].to(m.weight.dtype))
    D.check_controls(seen, arm="W1-L")
    core = results["core32"]["survival_error"]
    twin = min(results[t]["survival_error"] for t in H.TWIN_IDS if t in results)
    gates.record("W1L_durability_measured", True, banks=sorted(seen),
                 core_survival_error=core, best_twin_survival_error=twin,
                 refuter_fires=bool(core >= twin))
    return {"results": results, "core_survival_error": core, "best_twin_survival_error": twin,
            "refuter_fires": bool(core >= twin), "n_edits": a.n_edits}


def run_w1_m(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha, ctx):
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _ppl_blankimg
    from infra.aws_pg.jobs.darwin_w1_surgery import highway_basis
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows

    have_wide = len(list(pathlib.Path(bank_dir).glob("*__highway_top256.npy"))) > 0
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seed=a.seed)
    train_batches = build_batches(processor, train_rows, "cuda")
    eval_batches = build_batches(processor, eval_rows, "cuda")
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    mods = T.trainable_modules(layers, types=types, layers=ctx["layers"])
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    base_ppl = ctx["base_ppl"]
    base_task = eval_task_loss(model, eval_batches)
    atlas.add("task_loss", value=base_task, bank_id="none", dose=0.0, incumbent_id="base",
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"arm_phase": "base", "role": "reset reference", "ppl": base_ppl})
    print(f"[W1-M] base ppl={base_ppl:.5f} task={base_task:.5f}", flush=True)

    D.check_dose_grid(list(RANK_LADDER), arm="W1-M", min_doses=2 if a.smoke else 3)
    results = {}
    for r in RANK_LADDER:
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        basis = {k: highway_basis(bank_dir, k[0], k[1], r, have_wide=have_wide) for k in mods}
        proj = T.SubspaceProjector(basis, "cuda")
        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        for m in mods.values():
            m.weight.requires_grad_(True)
            params.append(m.weight)
        opt = T.ExactAdam(params, lr=a.lr)
        D.check_optimizer("W1-M", "exact_fp32_adam")
        model.train()
        for step in range(a.steps):
            b = train_batches[step % len(train_batches)]
            loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
            opt.zero_grad()
            loss.backward()
            proj.apply(mods)                     # confine the UPDATE, not the weight
            opt.step()
        model.eval()
        ppl_ft = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task_ft = eval_task_loss(model, eval_batches)

        # THE RESET. The update was confined to span(U) on the input side, so subtracting its
        # U-coefficients is the exact inverse — dW U Uᵀ == dW when dW already lives there. The
        # residual is therefore also the AUDIT: a non-zero leak means the confinement leaked.
        leak, dist_after = [], []
        for (t, L), m in mods.items():
            dW = (m.weight.detach().to(torch.float32) - pristine[(t, L)]).cpu().numpy().astype(
                np.float64)
            U = basis[(t, L)]
            keep = dW - (dW @ U) @ U.T
            tot = float((dW ** 2).sum())
            leak.append(float((keep ** 2).sum() / tot) if tot > 0 else 0.0)
            reset = pristine[(t, L)].cpu().numpy().astype(np.float64) + keep
            m.weight.data.copy_(torch.as_tensor(reset, dtype=m.weight.dtype,
                                                device=m.weight.device))
            dist_after.append(float(np.sqrt((keep ** 2).sum()) /
                                    max(np.sqrt(tot), 1e-12)) if tot > 0 else 0.0)
        ppl_reset = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task_reset = eval_task_loss(model, eval_batches)

        atlas.add("reset_weight_residual", value=float(np.mean(dist_after)), bank_id="highway",
                  dose=float(r), incumbent_id="lora_delete", n=len(mods), model_id=PG1_REPO,
                  model_sha=base_sha,
                  flags={"rank": r, "mean_leak_fraction": float(np.mean(leak)),
                         "max_leak_fraction": float(np.max(leak)),
                         "definition": "‖dW - dW U Uᵀ‖_F / ‖dW‖_F after coefficient subtraction; "
                                       "0 means the confinement held and the reset is exact"})
        atlas.add("reset_ppl_return", value=ppl_reset - base_ppl, bank_id="highway", dose=float(r),
                  incumbent_id="lora_delete", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"rank": r, "ppl_base": base_ppl, "ppl_after_ft": ppl_ft,
                         "ppl_after_reset": ppl_reset,
                         "ft_battery_cost": ppl_ft - base_ppl,
                         "lora_delete_return": 0.0,
                         "incumbent_note": "LoRA-delete returns exactly by construction (the base "
                                           "weights are never touched); the question this row "
                                           "answers is what an IN-PLACE update costs to undo"})
        atlas.add("reset_task_removal", value=task_reset - base_task, bank_id="highway",
                  dose=float(r), incumbent_id="lora_delete", model_id=PG1_REPO,
                  model_sha=base_sha,
                  flags={"rank": r, "task_base": base_task, "task_after_ft": task_ft,
                         "task_after_reset": task_reset,
                         "task_learned": base_task - task_ft,
                         "fraction_removed": ((task_reset - task_ft) / (base_task - task_ft))
                                             if base_task != task_ft else None,
                         "definition": "task loss after reset minus task loss at base; ~0 means "
                                       "the skill left with the coefficients"})
        results[r] = {"ppl_ft": ppl_ft, "ppl_reset": ppl_reset, "task_ft": task_ft,
                      "task_reset": task_reset, "leak": float(np.mean(leak)),
                      "residual": float(np.mean(dist_after))}
        print(f"[W1-M] r={r}: ppl {base_ppl:.4f} -> {ppl_ft:.4f} -> {ppl_reset:.4f} | "
              f"task {base_task:.4f} -> {task_ft:.4f} -> {task_reset:.4f} | "
              f"leak={np.mean(leak):.3e}", flush=True)
        del opt
        torch.cuda.empty_cache()
        D.s3_sync(out, a.s3_prefix)

    for k, m in mods.items():
        m.weight.data.copy_(pristine[k].to(m.weight.dtype))
    gates.record("W1M_reset_measured", True, ranks=list(RANK_LADDER), base_ppl=base_ppl,
                 base_task=base_task, results={str(k): v for k, v in results.items()})
    return {"base_ppl": base_ppl, "base_task": base_task,
            "results": {str(k): v for k, v in results.items()},
            "incumbent": "lora_delete (exact by construction; the contrast is cost, not return)"}


RUNNERS = {"W1-L": run_w1_l, "W1-M": run_w1_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--n-edits", type=int, default=N_EDITS)
    ap.add_argument("--edit-scale", type=float, default=0.25)
    ap.add_argument("--exactness-max", type=float, default=0.05,
                    help="installation gate for W1-L; the bf16 storage floor is ~4e-3")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        a.steps, a.n_train, a.n_eval, a.n_seq, a.n_edits = 3, 12, 4, 2, 2
        layers = layers[:2]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "layers": layers,
                "types": a.types, "edit_type": EDIT_TYPE, "n_edits": a.n_edits,
                "rank_ladder": list(RANK_LADDER), "steps": a.steps, "lr": a.lr,
                "editor": "own closed-form rank-1 least-squares insert (§4/W1-L's A5 pivot)",
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    dl = _decoder_layers(model)
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    ctx = {"layers": layers, "base_ppl": base_ppl}

    result = RUNNERS[a.arm](model, processor, dl, bank_dir, out, atlas, gates, a, base_sha, ctx)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "base_ppl": base_ppl, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
