"""COMWRITE Arm C — the write-channel test (comwrite spec §6): is U_wr the base's preferred
adaptation channel for NEW learning? One job = one constraint cell of the frozen relearn probe:

  unconstrained    — the reference (must reproduce the ENDOW-R PG1 band, else INSTRUMENT-NULL;
                     that gate is adjudicated downstream on the banked numbers).
  into@0           — every step's gradient projected INTO U_wr(k_C) on the residual-facing side.
  off@0            — projected OFF U_wr(k_C).
  into_random@N    — projected into a seeded equal-k random span (the matched null; ≥3 draws).

Mechanics = the trainscaffold_reach pattern verbatim: PG1 full multimodal model, ONLY the 18×7
decoder matrices trainable, SGD on MMLU letter-CE (text-only forward), per-step in-place grad
surgery — here comwrite_core.project_grad_residual (ONE-sided, into+off partition, tested).
U_wr here is the FULL-24 basis from the Arm-A bank (no LOO: the probe task is outside the
24-FT task family by construction, §6). Readout: held-out BEST-CHECKPOINT blank-image MMLU
(never final-step); train-CE curve reported (TRAINSCAFFOLD identification label).

The ENDOW-R artifact is downloaded + banked verbatim so the analysis step reads the reference
band from the artifact, never from prose (§0 provenance rule).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_c_probe --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/comwrite_c_into --budget-min 300 --args "\
      --cell into@0 --k 16 --anatomy-s3 s3://pg-lineage-769063704696/results/comwrite_a_anatomy/out \
      --endow-r-s3 s3://pg-lineage-769063704696/results/endow_r_r0/out/metrics.json \
      --recipe-binding 'frozen at gate0 from <artifact key path>' \
      --n-train 16000 --steps 1500 --lr 2e-4 --batch 8"

OPEN pre-freeze item (journal 2026-07-05): the §6 instrument gate needs a LIKE-FOR-LIKE
PG1 reference band — endow_r_r0's default artifact has no pg1 init row and uses an
adapter-relearn on a swap-shell (different instrument). Binding is resolved at Gate-0
(either an ENDOW-R pg1 re-run, or the gate re-based on this arm's own unconstrained cell
with the change journaled BEFORE freeze); until then Arm C must not freeze.
"""
import argparse
import json
import pathlib

from infra.pg.gauge_common import PROJ, layer_index, matrix_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", required=True,
                    help="unconstrained | into@0 | off@0 | into_random@<draw>")
    ap.add_argument("--k", type=int, required=True, help="k_C = k*; asserted vs anatomy")
    ap.add_argument("--anatomy-s3", required=True)
    ap.add_argument("--endow-r-s3", required=True,
                    help="the ENDOW-R artifact (reference band; banked verbatim)")
    ap.add_argument("--recipe-binding", required=True,
                    help="REQUIRED disclosure: how this cell's {n_train,steps,lr,batch} bind "
                         "to the ENDOW-R artifact (e.g. 'frozen at gate0 from <key path>' or "
                         "'UNBOUND: <reason>'). Recorded verbatim; a job with no stated "
                         "binding must not run (§6 frozen-config rule + §10.0).")
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=250, help="held-out checkpoints")
    ap.add_argument("--n-per-subject", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mode, _, draw = a.cell.partition("@")
    draw = int(draw or 0)
    assert mode in ("unconstrained", "into", "off", "into_random"), f"unknown cell {a.cell!r}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from datasets import load_dataset
    from infra.pg import comwrite_core as CC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import relearn as Rl
    from infra.pg import text_eval as TE
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import random_basis

    from infra.aws_pg.jobs import _inherit_shared as SH
    anat = json.loads(SH.fetch(f"{a.anatomy_s3}/metrics.json",
                               out / "anatomy_metrics.json").read_text())
    assert int(anat["k_star"]) == a.k, f"--k {a.k} != anatomy k*={anat['k_star']}"
    SH.fetch(a.endow_r_s3, out / "endow_r_metrics.json")           # banked verbatim (§0)
    excluded = {(int(L), t) for L, t in anat["excluded_lt"]}
    grid = [(m["layer"], m["type"]) for m in anat["per_matrix"]]

    rep = {"object": "comwrite_c_probe_v1", "cell": a.cell, "k": a.k, "seed": a.seed,
           "steps": a.steps, "lr": a.lr, "batch": a.batch, "n_train": a.n_train,
           "anatomy_s3": a.anatomy_s3, "endow_r_s3": a.endow_r_s3,
           "recipe_binding": a.recipe_binding,
           "excluded_lt": sorted(excluded), "base_sha": anat["base_sha"]}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    device = "cuda"
    assert torch.cuda.is_available()
    model, proc = IP.load_pg1(sha=anat["base_sha"])
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    tok = proc.tokenizer

    # the trainable 18×7 grid inside the LM (full-weight FT; everything else frozen).
    dkeys = {}
    for k, p in lm.named_parameters():
        if p.ndim == 2 and "layers." in k and any(f".{t}." in k for t in PROJ):
            dkeys[k] = (layer_index(k), matrix_type(k))
    assert len(dkeys) == 18 * 7, f"expected 126 trainable matrices, got {len(dkeys)}"

    # per-key constraint basis (full-24 U_wr from the bank; or the seeded equal-k random
    # span). fp32 on device — the partition must not leak through a bf16 basis. A1-excluded
    # (L,t) carry NO commons: in the into-cells their gradient is ZEROED (no channel there
    # — leaving them free would inflate G(into) toward G(unconstrained) and fake
    # channel-sufficiency); in the off-cell they are unconstrained (off of nothing = g).
    bases, zeroed = {}, set()
    if mode != "unconstrained":
        for k, (L, t) in dkeys.items():
            if (L, t) in excluded:
                if mode in ("into", "into_random"):
                    zeroed.add(k)
                continue
            if mode == "into_random":
                U = random_basis(2048, a.k, 100003 * (draw + 1) + 7 * L + PROJ.index(t),
                                 device, torch.float32)
            else:
                f = SH.fetch(f"{a.anatomy_s3}/scratch/ufull__{t}__L{L}.npy",
                             out / f"bank/ufull__{t}__L{L}.npy")
                U = torch.from_numpy(np.load(f)[:, : a.k].astype(np.float32)).to(device)
            bases[k] = U
    rep["n_constrained"] = len(bases)
    rep["n_zeroed_excluded"] = len(zeroed)

    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    for k in dkeys:
        prm = lm.get_parameter(k)
        prm.requires_grad_(True)
        train_params.append((k, prm))
    model.train()

    # the probe pool (dedup vs the MMLU eval set — the trainscaffold identification hygiene).
    ds = load_dataset(TE.MMLU_REPO, "all")
    poolrows = [dict(r) for r in ds["auxiliary_train"]]
    ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
    ded = Rl.dedup_against_eval(poolrows, ev_hashes)
    sub = Rl.ladder_subset(ded["kept"], a.n_train)
    ex = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    letters = Rl.letter_token_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def _heldout():
        model.eval()
        mcq = TE.mmlu_blankimg(model, proc, device, n_per_subject=a.n_per_subject)
        model.train()
        return {"acc_pct": round(100 * sum(mcq["per_item"]) / len(mcq["per_item"]), 2),
                "n_items": len(mcq["per_item"]), "per_item": mcq["per_item"],
                "per_item_hash": mcq["per_item_hash"]}

    ckpt0 = _heldout()                                # step-0 anchor (the init's own score)
    rep["checkpoints"] = {"0": ckpt0}
    _w()

    opt = torch.optim.SGD([p for _, p in train_params], lr=a.lr, momentum=a.momentum)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(ex), generator=g).tolist()
    di, losses = 0, []
    mode_pg = "into" if mode == "into_random" else mode
    for step in range(1, a.steps + 1):
        idx = [order[(di + j) % len(order)] for j in range(a.batch)]
        di += a.batch
        prompts = [ex[i]["prompt"] for i in idx]
        tgt = torch.tensor([letters[ex[i]["answer_idx"]] for i in idx], device=device)
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        enc.pop("token_type_ids", None)     # PG text-only forward gotcha (auto-loss trap)
        pos = enc["attention_mask"].long().cumsum(-1) - 1
        pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
        last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        loss = torch.nn.functional.cross_entropy(last.float(), tgt)
        assert torch.isfinite(loss), f"non-finite loss step {step} — HALT"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mode != "unconstrained":
            with torch.no_grad():
                for k, prm in train_params:
                    if prm.grad is None:
                        continue
                    if k in zeroed:
                        prm.grad.zero_()
                    elif k in bases:
                        prm.grad.copy_(CC.project_grad_residual(
                            prm.grad, bases[k], dkeys[k][1], mode_pg).to(prm.grad.dtype))
        opt.step()
        losses.append(float(loss.item()))
        if step % a.eval_every == 0 or step == a.steps:
            ck = _heldout()
            rep["checkpoints"][str(step)] = ck
            best = max(rep["checkpoints"].items(), key=lambda kv: kv[1]["acc_pct"])
            rep["best_checkpoint"] = {"step": int(best[0]), "acc_pct": best[1]["acc_pct"]}
            rep["train_ce"] = {"first": losses[0], "last": losses[-1],
                               "curve_every50": losses[::50]}
            _w()
            print(f"[comwrite-c] {a.cell} step {step}: loss={losses[-1]:.4f} "
                  f"heldout={ck['acc_pct']:.2f} best={rep['best_checkpoint']}", flush=True)

    rep["green"] = True
    _w()
    print("CPROBE_OK", json.dumps({"cell": a.cell, "best": rep["best_checkpoint"],
                                   "ce": [losses[0], losses[-1]]}))


if __name__ == "__main__":
    main()
