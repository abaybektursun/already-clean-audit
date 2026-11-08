"""GATE-3 parallel-path attempt (RESEARCH_INDEX §5.4, "the one live route to a measured win").

The boundary theory's PRACTICAL prediction: the install carrier is OUT-OF-SPAN (W15 f_perp;
§5.2 confirms it generation-shared). So a frozen-backbone side-adapter that adds capacity in the
backbone's OUT-OF-SPAN directions should install/recover a capability BETTER than one adding
capacity IN-SPAN — a predictive, novel, practical design win if it holds.

Decisive controlled head-to-head (text-only, cheap): delete MMLU from base gemma-2b
(reinit_lm_layer_ on mid layers), then RECOVER it with a frozen-backbone rank-r adapter whose
FIXED projection directions are:
  * in-span   = top-r left singular vectors of each (deleted) decoder weight  U[:, :r]
  * out-span  = bottom-r left singular vectors                                U[:, -r:]
  * random    = random orthonormal directions (control)
Only B (r×in) is trainable in all three → MATCHED trainable capacity + budget; the ONLY
difference is which span the added capacity lives in. Win = out-span recovers MMLU MCQ
significantly above in-span (and random).

  python -m infra.aws_pg.jobs.gate3_outofspan --out <dir> [--rank 64] [--budget 4000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _proj_A(W_svd, out_d, r, mode, seed):
    """The FIXED adapter directions A [out, r]. mode ∈ {inspan, outspan, random}. W_svd is the
    INTACT pretrained weight (the theory's W_before reference), NOT the deleted/random one."""
    import torch
    if mode in ("inspan", "outspan"):
        U = torch.linalg.svd(W_svd.detach().float(), full_matrices=False).U   # [out, k]
        return (U[:, :r] if mode == "inspan" else U[:, -r:]).contiguous()
    if mode == "random":
        g = torch.Generator().manual_seed(int(seed))
        M = torch.randn(out_d, r, generator=g)
        return (torch.linalg.qr(M).Q if out_d >= r else M / M.norm(dim=0, keepdim=True)).contiguous()
    raise ValueError(mode)


def make_adapter_projected(base_linear, A_fixed, alpha):
    """Frozen base + scale·(x Bᵀ) Aᵀ, A FIXED (precomputed from the intact weight), B trainable
    (B=0 init → zero delta at step 0)."""
    import torch
    import torch.nn as nn
    out_d, in_d = base_linear.weight.shape
    r = A_fixed.shape[1]

    class ProjAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base_linear
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.register_buffer("Afix", A_fixed.to(base_linear.weight.dtype).contiguous())
            self.B = nn.Parameter(torch.zeros(r, in_d))
            self.scale = alpha / r

        def forward(self, x):
            B = self.B.to(x.dtype)
            A = self.Afix.to(x.dtype)
            return self.base(x) + self.scale * ((x @ B.transpose(0, 1)) @ A.transpose(0, 1))

    return ProjAdapter()


def precompute_A(intact_lm, rank):
    """One SVD per decoder linear of the INTACT pretrained weights → cache the top-r (inspan) and
    bottom-r (outspan) left singular directions on CPU. Reused across arms (mode-independent SVD)."""
    import torch
    from infra.pg.relearn import adapter_target_names
    cache = {}
    for n in adapter_target_names(intact_lm):
        W = intact_lm.get_submodule(n).weight.detach().float()
        out_d, in_d = W.shape
        r = min(rank, out_d, in_d)
        U = torch.linalg.svd(W, full_matrices=False).U
        cache[n] = {"inspan": U[:, :r].cpu().contiguous(), "outspan": U[:, -r:].cpu().contiguous()}
    return cache


def attach_projected(lm, A_cache, rank, mode, seed, alpha):
    import hashlib
    from infra.pg.relearn import adapter_target_names
    names = adapter_target_names(lm)
    assert names, "no adapter targets"
    for p in lm.parameters():
        p.requires_grad_(False)
    for n in names:
        parent_name, _, leaf = n.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        base = getattr(parent, leaf)
        out_d, in_d = base.weight.shape
        r = min(rank, out_d, in_d)
        if mode == "random":
            ss = int(hashlib.sha1(f"{seed}:{n}".encode()).hexdigest()[:8], 16)
            A = _proj_A(None, out_d, r, "random", ss)
        else:
            A = A_cache[n][mode]
        setattr(parent, leaf, make_adapter_projected(base, A.to(base.weight.device), alpha))
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    return {"n_modules": len(names), "trainable_params": trainable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--budget", type=int, default=4000, help="MMLU relearn examples")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--reinit-layers", default="6,7,8,9,10", help="mid layers to delete")
    ap.add_argument("--reinit-seed", type=int, default=1337)
    ap.add_argument("--eval-n-per", type=int, default=0, help="0=full MMLU test")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import relearn as R
    from infra.pg.w16_probes import reinit_lm_layer_
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available()
    device = "cuda"
    rank = 8 if a.smoke else a.rank
    budget = 256 if a.smoke else a.budget
    eval_n = 2 if a.smoke else (a.eval_n_per or None)
    reinit_layers = [int(x) for x in a.reinit_layers.split(",")]
    ent = ENTITIES["L0"]                              # gemma-2-2b
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    print(f"[GATE3] rank={rank} budget={budget} reinit={reinit_layers} smoke={a.smoke}", flush=True)

    def fresh_deleted():
        m = AutoModelForCausalLM.from_pretrained(
            ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="eager").eval()
        for L in reinit_layers:
            reinit_lm_layer_(m.model, L, a.reinit_seed)
        return m

    # baselines: intact + deleted (no adapter)
    intact = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    acc_intact = TE.mmlu_accuracy(intact, tok, device, n_per_subject=eval_n)["acc_pct"]
    A_cache = precompute_A(intact.model, rank)        # in/out-span from the INTACT pretrained weights
    del intact; torch.cuda.empty_cache()
    dele = fresh_deleted()
    acc_deleted = TE.mmlu_accuracy(dele, tok, device, n_per_subject=eval_n)["acc_pct"]
    mm_eval_hashes = TE.mmlu_accuracy(dele, tok, device, n_per_subject=2)["per_item_hash"]
    del dele; torch.cuda.empty_cache()
    print(f"[GATE3] intact MMLU={acc_intact:.2f}  deleted MMLU={acc_deleted:.2f}", flush=True)

    # relearn pool: MMLU auxiliary_train, dedup vs eval, fixed ladder subset (shared across arms)
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, mm_eval_hashes)
    sub = R.ladder_subset(ded["kept"], budget)
    examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]

    arms = {}
    for mode in ("inspan", "outspan", "random"):
        m = fresh_deleted()
        info = attach_projected(m.model, A_cache, rank, mode, seed=1337, alpha=a.alpha)
        m.to(device)
        tr = R.train_adapter_qa(m, tok, device, examples, lr=a.lr, epochs=a.epochs,
                                batch_size=8, warmup_ratio=0.03, seed=1337, micro_batch=4)
        acc = TE.mmlu_accuracy(m, tok, device, n_per_subject=eval_n)["acc_pct"]
        recov = ((acc - acc_deleted) / (acc_intact - acc_deleted)
                 if acc_intact > acc_deleted else float("nan"))
        arms[mode] = {"acc_pct": acc, "recovery_frac": recov, "final_loss": tr["final_loss"],
                      "trainable_params": info["trainable_params"]}
        print(f"[GATE3] {mode:7s} MMLU={acc:.2f}  recovery={recov:+.3f}  loss={tr['final_loss']:.3f}", flush=True)
        del m; torch.cuda.empty_cache()

    # verdict: does out-of-span recover above in-span (the boundary-theory practical prediction)?
    os_minus_is = arms["outspan"]["acc_pct"] - arms["inspan"]["acc_pct"]
    os_minus_rand = arms["outspan"]["acc_pct"] - arms["random"]["acc_pct"]
    MARGIN = 1.0                                       # pt; below this is eval noise, not a win
    any_recovered = any(v["recovery_frac"] > 0.05 for v in arms.values())
    if not any_recovered:
        verdict = "UNINFORMATIVE: no arm recovers above floor (deletion too severe for the budget) — no win"
    elif os_minus_is > MARGIN and os_minus_rand > MARGIN:
        verdict = "WIN: out-of-span recovers meaningfully (>1pt) above in-span AND random"
    else:
        verdict = "NO WIN: out-of-span does not beat in-span by a meaningful margin (theory's design edge not practical)"
    result = {"experiment": "gate3_outofspan", "rank": rank, "budget": budget,
              "reinit_layers": reinit_layers, "intact_mmlu": acc_intact, "deleted_mmlu": acc_deleted,
              "arms": arms, "outspan_minus_inspan": os_minus_is, "outspan_minus_random": os_minus_rand,
              "any_arm_recovered": any_recovered,
              "prediction": "boundary theory → out-of-span adapter recovers best",
              "verdict": verdict}
    (out / "gate3_outofspan.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[GATE3] out-span−in-span={os_minus_is:+.2f}  out-span−random={os_minus_rand:+.2f} → {result['verdict']}")


if __name__ == "__main__":
    main()
