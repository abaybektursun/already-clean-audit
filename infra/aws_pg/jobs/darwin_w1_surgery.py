"""DARWIN W1-D / W1-O — weight-surgery arms on the shared ppl harness. GPU.

  W1-D  Delta compression (§4/W1-D). A fine-tune's dW truncated to the universal HIGHWAY basis at
        rank r, against the arm's named incumbent: an equal-rank SVD OF THE MODEL'S OWN DELTA.
        That incumbent is the honest one — it is the best rank-r approximation of this particular
        delta by construction (Eckart-Young), so the universal basis can only win by being
        cheaper to ship, never by being a better fit. REFUTER: universal basis strictly dominated
        at all ranks.
  W1-O  Prune-by-conservedness (§4/W1-O). Rank weights by a doubly-cold score (small magnitude
        AND low consensus-write energy) against magnitude pruning at a matched sparsity ladder.

Both reduce to swapping one effective matrix and running the SAME ppl path W1-N uses and G2
anchored, so their numbers are commensurable with each other and with the quantization arms.

The highway basis at rank up to 256 comes from W0-8's banked `highway_top256` / `core_bottom256`
(that job holds the full eigenbasis anyway); this arm never re-streams the 24-checkpoint cloud.

  pg run infra.aws_pg.jobs.darwin_w1_surgery --args "--arm W1-D" \
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

FT_REPOS = ("google/paligemma-3b-ft-vqav2-224", "google/paligemma-3b-ft-refcoco-seg-224")
RANK_LADDER = (8, 16, 32, 64, 128)          # >=4 doses, log-spaced (§3)
SPARSITY_LADDER = (0.1, 0.2, 0.4, 0.8)      # >=4 doses, log-spaced ratio 2
WIDE_BANK = "highway_top256"

SPEC_EXPECTATIONS = {
    "W1-D": ("universal basis strictly dominated at all ranks",
             "the universal highway basis matches or beats equal-rank SVD-of-own-delta at some "
             "rank"),
    "W1-O": ("doubly-cold ranking no better than magnitude pruning at equal sparsity",
             "pruning by conservedness beats magnitude pruning at matched sparsity"),
}


def vqa_task_nll(model, processor, items, device="cuda"):
    """Teacher-forced NLL of the fine-tune's OWN task outputs — W1-D's correct readout (D19).

    W1-D asks whether a rank-r basis PRESERVES a fine-tune's capability. Scoring that with
    wikitext perplexity measures the opposite thing: on a catastrophic fine-tune, applying LESS
    of the delta improves wikitext, so the universal basis "wins" at low rank precisely because it
    captures less. Task NLL moves the right way — lower means the truncated delta still produces
    the fine-tune's answers.

    Teacher-forced rather than generated: generation adds decoding-policy variance that has
    nothing to do with the basis under test, and the arm compares bases.
    """
    import torch
    tot, ntok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for it in items:
            enc = processor(text=it["prompt"], images=it["image"], suffix=it["answer"],
                            return_tensors="pt").to(device)
            out = model(**enc)
            lab = enc.get("labels")
            assert lab is not None, "processor did not build labels from suffix"
            n = int((lab != -100).sum())
            if n == 0:
                continue
            tot += float(out.loss.item()) * n
            ntok += n
    assert ntok > 0, "task NLL saw no supervised tokens"
    return tot / ntok


# Each fine-tune must be scored on ITS OWN task (defect D22). The first D19 fix scored BOTH
# fine-tunes on VQAv2, and ft-refcoco-seg is a SEGMENTATION model: its full-delta VQA NLL (7.14)
# is WORSE than base (1.70), so the `base - full` normaliser went negative and
# task_recovered_frac read ~1e9. Scoring a segmentation model on VQA measures nothing about its
# capability, and a number that large is a tell, not a result.
TASK_BY_FT = {
    "google/paligemma-3b-ft-vqav2-224": "vqav2",
}

# COVERAGE-IMPOSSIBLE, with the reason and the evidence (defects D19/D22/D23, resolved).
# ft-refcoco-seg's task readout is not constructible from the data available here. Its model card
# states PaliGemma emits "segmentation codewords" — <segXXX> tokens produced by a VQ-VAE over
# masks — so a teacher-forced target must contain them. `lmms-lab/RefCOCO` carries bounding
# boxes, not codewords. Four conventions were tried and all failed the readout-validity gate in
# the same direction (the fine-tune scoring WORSE than base on its own task):
#     D19  wikitext ppl                 — measures retention, moves the wrong way entirely
#     D22  VQAv2                        — a foreign task for a segmentation model
#     D23  "detect" + loc tokens        — the detection convention: base 6.53 -> full 8.80
#          "segment" + loc tokens       — right prefix, truncated target: base 8.63 -> full 9.03
# This is recorded as a coverage gap rather than a fifth guess. W1-D's estimand is answered on
# ft-vqav2, where the gate PASSES (base 1.7047 -> full 0.3788) and the rank sweep is valid.
COVERAGE_IMPOSSIBLE = {
    "google/paligemma-3b-ft-refcoco-seg-224":
        "teacher-forced target requires PaliGemma segmentation codewords (<segXXX>, VQ-VAE over "
        "masks, per the model card); lmms-lab/RefCOCO provides boxes only. Four conventions "
        "tried, all failed the readout-validity gate.",
}


def load_task_items(ft_repo: str, n_items: int, seed: int = 1337):
    """Teacher-forcing items for the fine-tune's OWN task, via battery_tasks' frozen TaskSpecs
    where they exist, so the prompt/target conventions are the house ones rather than re-invented
    (reference_paligemma_suffix_target_space: never prepend a space to a suffix target)."""
    from datasets import load_dataset
    if ft_repo in COVERAGE_IMPOSSIBLE:
        return []          # recorded by the caller as a coverage gap, not scored on a proxy
    kind = TASK_BY_FT.get(ft_repo)
    assert kind, f"no task readout defined for {ft_repo} — refusing to score it on a foreign task"
    if kind == "vqav2":
        ds = load_dataset("merve/vqav2-small", split="validation", streaming=True)
        items = []
        for row in ds:
            a = str(row.get("multiple_choice_answer") or "").strip()
            if not a:
                continue
            items.append({"prompt": f"answer en {row['question']}", "answer": a,
                          "image": row["image"]})
            if len(items) >= n_items:
                break
    else:
        # SEGMENT, not detect (defect D23). battery_tasks' refcoco TaskSpec is the DETECTION
        # convention — prompt "detect {expr}", target four <locXXXX> box tokens. The fine-tune is
        # ft-refcoco-SEG, trained on "segment {expr}" with an output of four loc tokens FOLLOWED
        # by sixteen <segXXX> tokens. Scoring it under the detection prompt made the full
        # fine-tune score WORSE than base on its own task (8.80 vs 6.53), which is impossible for
        # a correct readout and is what exposed the mismatch.
        #
        # SCOPE, stated: the target here is the LOCALISATION component — the four loc tokens that
        # open a segmentation output — under the segment prompt. That is a real part of the
        # fine-tune's capability, not the whole of it; the <segXXX> mask codes are not scored
        # because this dataset does not carry them.
        from infra.pg.battery_tasks import TASKS, _refcoco_expr
        spec = TASKS["refcoco"]
        ds = load_dataset(spec.dataset, split=spec.eval_split, streaming=True)
        items = []
        for row in ds:
            tgt = str(spec.target_fn(row)).strip()
            if not tgt:
                continue
            items.append({"prompt": f"segment {_refcoco_expr(row)}\n", "answer": tgt,
                          "image": spec.image_fn(row)})
            if len(items) >= n_items:
                break
    assert len(items) >= max(4, n_items // 4), f"only {len(items)} {kind} items"
    return items


def truncate_to_basis(dW, U_r):
    """dW projected onto a fixed r-dim INPUT subspace: dW U_r U_rᵀ.

    This is what "shipping the delta in the universal basis" means — the receiver already holds
    U_r, so only the m x r coefficient block travels.
    """
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    U = np.asarray(U_r, dtype=np.float64)
    return (A @ U) @ U.T


def truncate_own_svd(dW, r: int):
    """The incumbent: the best rank-r approximation of THIS delta (Eckart-Young)."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    Uu, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(r, s.size)
    return (Uu[:, :r] * s[:r]) @ Vt[:r]


def ship_cost_values(m: int, n: int, r: int, scheme: str) -> int:
    """Values that must travel per matrix. The universal basis ships only the coefficient block
    because the receiver already has U_r; own-SVD must ship both factors."""
    if scheme == "universal":
        return m * r
    if scheme == "own_svd":
        return r * (m + n)
    raise AssertionError(scheme)


def highway_basis(bank_dir, t, layer, r: int, *, have_wide: bool):
    """The top-r HIGHWAY subspace: the r consensus-MOST-written input directions.

    Both banks are stored in ASCENDING eigenvalue order, so the top-r is the LAST r columns in
    each case. Slicing `[:, :r]` off anti32 would return the r SMALLEST eigenvalues inside the
    top-32 block — the least-written directions of the most-written 32, close to the opposite of
    what this arm ships, and it would have quietly understated the universal basis at every rank
    below 32. (At r = 32 exactly the two slices coincide, which is why W1-O was unaffected.)
    """
    import numpy as np
    if r <= H.BANK_K and not have_wide:
        U = W.load_bank(bank_dir, t, layer, "anti32")
    else:
        U = np.load(pathlib.Path(bank_dir) / W.bank_name(t, layer, WIDE_BANK))
    assert r <= U.shape[1], f"rank {r} exceeds available basis width {U.shape[1]}"
    return np.ascontiguousarray(U[:, -r:])


def doubly_cold_score(W_mat, U_hot):
    """Low |w| AND low consensus-write involvement. Per-weight score, larger = colder = prunable.

    `U_hot` is the highway (consensus-WRITTEN) basis, so a column's hotness is its leverage there;
    a weight is doubly cold when it is small and sits in a column the fine-tune cloud avoids.
    Both factors are rank-normalised to [0,1] so neither silently dominates by having a wider
    numeric range — without that the magnitude term alone decides everything and the arm becomes
    magnitude pruning wearing a different name.
    """
    import numpy as np
    A = np.abs(np.asarray(W_mat, dtype=np.float64))
    hot = H.leverage(U_hot)                       # per-input-coordinate, length n
    def rank01(x):
        order = x.argsort(axis=None).argsort(axis=None).astype(np.float64)
        return (order / max(1, x.size - 1)).reshape(x.shape)
    return (1.0 - rank01(A)) + (1.0 - rank01(np.broadcast_to(hot, A.shape).copy()))


def prune_by_score(W_mat, score, sparsity: float):
    """Zero the `sparsity` fraction of weights with the HIGHEST score (coldest)."""
    import numpy as np
    A = np.asarray(W_mat, dtype=np.float64).copy()
    k = int(round(sparsity * A.size))
    if k <= 0:
        return A
    idx = np.argpartition(score, A.size - k, axis=None)[A.size - k:]
    A.flat[idx] = 0.0
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(SPEC_EXPECTATIONS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="q_proj,o_proj,gate_proj,up_proj,v_proj,k_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANK_LADDER))
    ap.add_argument("--sparsities", default=",".join(str(s) for s in SPARSITY_LADDER))
    ap.add_argument("--ft-repos", default=",".join(FT_REPOS))
    ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--n-task", type=int, default=48,
                    help="VQAv2 items for W1-D's task readout (the D19 fix)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    spars = [float(s) for s in a.sparsities.split(",") if s.strip()]
    ft_repos = [r.strip() for r in a.ft_repos.split(",") if r.strip()]
    if a.smoke:
        types, layers, ranks, spars, ft_repos = (types[:1], layers[:2], ranks[:2], spars[:2],
                                                 ft_repos[:1])
        a.n_seq = 4

    D.check_dose_grid(ranks, arm=a.arm, min_doses=2 if a.smoke else 4)
    D.check_dose_grid(spars, arm=a.arm, min_doses=2 if a.smoke else 4)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers,
                                                       _load_full_pg1, _ppl_blankimg)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    have_wide = len(list(pathlib.Path(bank_dir).glob(f"*__{WIDE_BANK}.npy"))) > 0
    assert have_wide or a.arm != "W1-D" or max(ranks) <= H.BANK_K, (
        f"W1-D needs the {WIDE_BANK} bank for rank>{H.BANK_K}; run W0-8 first "
        f"(darwin_w0_gram --arm W0-8), which banks it")

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "ft_repos": ft_repos,
                "types": types, "layers": layers, "ranks": ranks, "sparsities": spars,
                "n_seq": a.n_seq, "wide_bank": WIDE_BANK, "wide_bank_available": have_wide,
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    dl = _decoder_layers(model)
    mods = {}
    for t in types:
        for L in layers:
            parent = dl[L].self_attn if t.startswith(("q", "k", "v", "o")) else dl[L].mlp
            mods[(t, L)] = getattr(parent, t)
    originals = {k: m.weight.detach().clone() for k, m in mods.items()}

    # W1-D's task readout (defect D19): the arm asks whether a rank-r basis PRESERVES the
    # fine-tune's capability, so it must be scored on the fine-tune's TASK, not on retention.
    # Per-fine-tune task items, loaded inside the fine-tune loop so each is scored on its own
    # task (D22). Base NLL is likewise per-task, since it is the normaliser.
    task_items, base_task = [], float("nan")
    base_res = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
    base_ppl = float(base_res["ppl"])

    atlas.add("base_ppl", value=base_ppl, model_id=PG1_REPO, model_sha=base_sha,
              flags={"n_tokens": base_res["n_tokens"], "n_seq": a.n_seq})
    print(f"[{a.arm}] base ppl {base_ppl:.5f}", flush=True)

    def swap(fn):
        for (t, L), mod in mods.items():
            mod.weight.data.copy_(torch.as_tensor(fn(t, L), dtype=mod.weight.dtype,
                                                  device=mod.weight.device))

    def restore():
        for k, mod in mods.items():
            mod.weight.data.copy_(originals[k])

    def wide_basis(t, L, r):
        return highway_basis(bank_dir, t, L, r, have_wide=have_wide)

    if a.arm == "W1-D":
        for ft_repo in ft_repos:
            ft_sha = api.model_info(ft_repo).sha
            task_items = load_task_items(ft_repo, a.n_task, a.seed)
            if not task_items:
                reason = COVERAGE_IMPOSSIBLE[ft_repo]
                gates.record(f"W1D_task_readout_available__{ft_repo.split('/')[-1]}", False,
                             reason=reason)
                atlas.add("task_readout_coverage_gap", value=1.0, model_id=ft_repo,
                          model_sha=ft_sha, flags={"reason": reason, "arm": "W1-D"})
                print(f"[W1-D] COVERAGE-IMPOSSIBLE {ft_repo}: {reason}", flush=True)
                continue
            base_task = vqa_task_nll(model, processor, task_items)
            atlas.add("base_task_nll", value=base_task, model_id=ft_repo, model_sha=ft_sha,
                      flags={"n_items": len(task_items), "task": TASK_BY_FT[ft_repo],
                             "readout": "teacher-forced NLL on the fine-tune's OWN task (D22)"})
            print(f"[W1-D] {ft_repo.split('/')[-1]} task={TASK_BY_FT[ft_repo]} "
                  f"base_task_nll={base_task:.4f}", flush=True)
            w_ft = W._load_decoder(ft_repo, ft_sha)
            dW = {(t, L): (w_ft[W.matrix_key(t, L)].to(torch.float64).numpy()
                           - originals[(t, L)].to(torch.float64).cpu().numpy())
                  for t in types for L in layers}
            # The FULL delta: the ceiling every truncation is measured against.
            swap(lambda t, L: originals[(t, L)].to(torch.float64).cpu().numpy() + dW[(t, L)])
            full = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
            full_task = vqa_task_nll(model, processor, task_items)
            # READOUT-VALIDITY GATE. A fine-tune must beat the base on its OWN task; if it does
            # not, the readout is measuring the wrong thing and every recovered-fraction below it
            # is nonsense (the D19/D22/D23 family all announced themselves this way, as a ~1e9
            # normaliser). Recorded as VOID-readout rather than published as numbers.
            readout_valid = bool(full_task < base_task)
            gates.record(f"W1D_readout_valid__{ft_repo.split('/')[-1]}", readout_valid,
                         base_task_nll=base_task, full_delta_task_nll=full_task,
                         task=TASK_BY_FT[ft_repo],
                         note=("the fine-tune must score better than base on its own task, or "
                               "the prompt/target convention does not match what it was trained "
                               "to emit"))
            if not readout_valid:
                print(f"[W1-D] VOID-readout for {ft_repo}: full {full_task:.4f} >= "
                      f"base {base_task:.4f}; skipping its rank sweep", flush=True)
                restore()          # the full delta is still swapped in; the NEXT fine-tune would
                del w_ft, dW       # otherwise start from a contaminated model
                continue
            atlas.add("wikitext_ppl", value=full, bank_id="full_delta", model_id=ft_repo,
                      model_sha=ft_sha, flags={"scheme": "full_delta", "base_ppl": base_ppl,
                                               "delta_ppl": full - base_ppl})
            atlas.add("task_nll", value=full_task, bank_id="full_delta", model_id=ft_repo,
                      model_sha=ft_sha,
                      flags={"scheme": "full_delta", "base_task_nll": base_task,
                             "readout": "VQAv2 teacher-forced NLL — the D19-correct readout",
                             "n_items": len(task_items)})
            restore()
            for r in ranks:
                for scheme in ("universal_highway", "own_svd"):
                    def eff(t, L, _r=r, _s=scheme):
                        W0 = originals[(t, L)].to(torch.float64).cpu().numpy()
                        d = (truncate_to_basis(dW[(t, L)], wide_basis(t, L, _r))
                             if _s == "universal_highway" else truncate_own_svd(dW[(t, L)], _r))
                        return W0 + d
                    swap(eff)
                    ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                    task = vqa_task_nll(model, processor, task_items)
                    restore()
                    m0, n0 = originals[(types[0], layers[0])].shape
                    cost = ship_cost_values(int(m0), int(n0), r,
                                            "universal" if scheme == "universal_highway"
                                            else "own_svd")
                    energy = float(np.mean([
                        np.linalg.norm(truncate_to_basis(dW[(t, L)], wide_basis(t, L, r))
                                       if scheme == "universal_highway"
                                       else truncate_own_svd(dW[(t, L)], r))
                        / max(1e-30, np.linalg.norm(dW[(t, L)]))
                        for t in types for L in layers]))
                    atlas.add("task_nll", value=task, bank_id=scheme, dose=float(r),
                              incumbent_id="own_svd", model_id=ft_repo, model_sha=ft_sha,
                              flags={"scheme": scheme, "rank": r, "base_task_nll": base_task,
                                     "full_delta_task_nll": full_task,
                                     "task_recovered_frac": ((base_task - task)
                                                             / max(1e-9, base_task - full_task)),
                                     "ship_values_per_matrix": cost,
                                     "readout": "VQAv2 teacher-forced NLL — the D19-correct readout"})
                    atlas.add("wikitext_ppl", value=ppl, bank_id=scheme, dose=float(r),
                              incumbent_id="own_svd", model_id=ft_repo, model_sha=ft_sha,
                              flags={"scheme": scheme, "rank": r, "base_ppl": base_ppl,
                                     "full_delta_ppl": full, "delta_ppl": ppl - base_ppl,
                                     "ship_values_per_matrix": cost,
                                     "captured_energy_fraction": energy})
                    print(f"[W1-D] {ft_repo.split('/')[-1]} r={r} {scheme}: ppl={ppl:.5f} "
                          f"(full {full:.5f}, base {base_ppl:.5f}, ship {cost})", flush=True)
                D.s3_sync(out, a.s3_prefix)
            del w_ft, dW
    else:
        hot = {(t, L): wide_basis(t, L, min(H.BANK_K, 32)) for t in types for L in layers}
        scores = {k: doubly_cold_score(originals[k].to(torch.float64).cpu().numpy(), hot[k])
                  for k in mods}
        mags = {k: -np.abs(originals[k].to(torch.float64).cpu().numpy()) for k in mods}
        for sp in spars:
            for scheme, sc in (("doubly_cold", scores), ("magnitude", mags)):
                swap(lambda t, L, _sc=sc, _sp=sp: prune_by_score(
                    originals[(t, L)].to(torch.float64).cpu().numpy(), _sc[(t, L)], _sp))
                ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                restore()
                atlas.add("wikitext_ppl", value=ppl, bank_id=scheme, dose=float(sp),
                          incumbent_id="magnitude", model_id=PG1_REPO, model_sha=base_sha,
                          flags={"scheme": scheme, "sparsity": sp, "base_ppl": base_ppl,
                                 "delta_ppl": ppl - base_ppl})
                print(f"[W1-O] sparsity={sp} {scheme}: ppl={ppl:.5f}", flush=True)
            D.s3_sync(out, a.s3_prefix)

    gates.record(f"{a.arm}_completed", True, n_rows=atlas.n_rows)
    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "base_ppl": base_ppl, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} base_ppl={base_ppl:.5f}")


if __name__ == "__main__":
    main()
