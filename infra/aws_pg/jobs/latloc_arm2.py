"""latloc Arm 2 (PRACTICAL, prereg §3.2) — surgical un-forgetting by a LOCALIZED WEIGHT restore.
CONDITIONAL on an Arm-1 localized verdict.

A residual is input-dependent (no storable fixed residual; sourcing one needs the intact model at
inference — the cost to avoid). The DEPLOYABLE intervention is therefore input-independent: copy the
intact decoder weights for ONLY a minimal layer set S (deleted weights everywhere else) and measure
restored blank-image MMLU MCQ. Reported vs a size-matched random-layer weight-restore null (5 seeds).
Honest either way — a null (S large / not better than random) is a full result.

    python -m infra.aws_pg.jobs.latloc_arm2 --restore-layers 13,14,15,16   # (--out from run_job.sh)
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _eval_with_restore(pg_after, pg_before, paths, layer_idxs, proc, device, split_pred, n_per):
    """Snapshot the deleted weights for `layer_idxs`, swap in the INTACT weights there, eval
    blank-image MCQ, then PUT THE DELETED WEIGHTS BACK — so every config (S, each random null) is
    independent. Returns accuracy."""
    import torch
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg import text_eval as TE
    dst = resolve_gemma2_layers(pg_after.get_submodule(paths["language_model"]))
    src = resolve_gemma2_layers(pg_before.get_submodule(paths["language_model"]))
    snap = {L: {k: v.detach().clone() for k, v in dst[L].state_dict().items()} for L in layer_idxs}
    with torch.no_grad():
        for L in layer_idxs:
            dst[L].load_state_dict(src[L].state_dict())           # intact weights in
    acc = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=split_pred)["acc"]
    with torch.no_grad():
        for L in layer_idxs:
            dst[L].load_state_dict(snap[L])                       # deleted weights back
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--restore-layers", default="16", help="comma-sep decoder layers to weight-restore (S)")
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], default="confirmation")
    ap.add_argument("--n-per-subject", type=int, default=0)
    args = ap.parse_args()
    assert args.out, "--out required"
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S = [int(x) for x in args.restore_layers.split(",") if x != ""]

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as Wp
    from infra.pg import latloc as LL
    from infra.aws_pg.jobs.latloc_cell import assert_generation_layers

    assert torch.cuda.is_available(), "arm2 needs GPU"
    device = "cuda"
    n_per = args.n_per_subject or None
    split_pred = Wp.is_confirmation if args.eval_split == "confirmation" else Wp.is_grid_subset

    pg_after, pg_before, proc, paths, _it, _g = H.build_before_after(device, attn_impl="eager")
    n_dec = assert_generation_layers(pg_after.model.language_model, LL.GEMMA2_N_LAYERS)
    assert all(0 <= L < n_dec for L in S), f"restore-layers {S} out of range [0,{n_dec})"

    intact_acc = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, item_filter=split_pred)["acc"]
    deleted_acc = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=n_per, item_filter=split_pred)["acc"]
    print(f"[arm2] intact={intact_acc:.3f} deleted={deleted_acc:.3f} restoring S={S}")

    def rf(acc):
        return LL.rf(acc, deleted_acc, intact_acc)

    restored_acc = _eval_with_restore(pg_after, pg_before, paths, S, proc, device, split_pred, n_per)
    print(f"[arm2] S-restore acc={restored_acc:.3f} RF={rf(restored_acc):.3f}")

    rand_accs = [_eval_with_restore(pg_after, pg_before, paths, list(rs), proc, device, split_pred, n_per)
                 for rs in LL.random_layer_sets(n_dec, len(S))]   # 5 seeds, |S| random layers each
    rand_rf = [rf(a) for a in rand_accs]
    print(f"[arm2] random-|S| restore RF mean={np.mean(rand_rf):.3f} sd={np.std(rand_rf):.3f}")

    # Arm-4 readout-vs-content (no patch): the deletion is MCQ-readout-specific iff the cloze CONTENT
    # is retained in the deleted model while MCQ collapses (prereg §3.4 content axis; the "measure is
    # not the mechanism" dissociation). Patch-under-cloze is a stronger future refinement; this
    # establishes the dissociation EXISTS, which is the load-bearing point.
    clz_int = TE.mmlu_cloze(pg_before, proc.tokenizer, device, processor=proc,
                            n_per_subject=n_per, item_filter=split_pred)["acc"]
    clz_del = TE.mmlu_cloze(pg_after, proc.tokenizer, device, processor=proc,
                            n_per_subject=n_per, item_filter=split_pred)["acc"]
    mcq_ret = deleted_acc / intact_acc if intact_acc > 0 else None
    clz_ret = clz_del / clz_int if clz_int > 0 else None
    arm4 = {"mcq_intact": intact_acc, "mcq_deleted": deleted_acc, "mcq_retention": mcq_ret,
            "cloze_intact": clz_int, "cloze_deleted": clz_del, "cloze_retention": clz_ret,
            "dissociated": bool(clz_ret is not None and mcq_ret is not None and clz_ret > 2 * mcq_ret)}
    print(f"[arm4] MCQ retention={mcq_ret:.3f} CLOZE retention={clz_ret:.3f} dissociated={arm4['dissociated']}")

    row = {"study": "latloc", "arm": "2_weight_restore+4_readout_content", "eval_split": args.eval_split,
           "firewall_n_layers": n_dec, "restore_layers_S": S, "size_S": len(S),
           "intact_acc": intact_acc, "deleted_acc": deleted_acc,
           "restored_acc": restored_acc, "restored_rf": rf(restored_acc),
           "random_restore_accs": rand_accs, "random_restore_rf_mean": float(np.mean(rand_rf)),
           "random_restore_rf_sd": float(np.std(rand_rf)),
           "beats_random": bool(rf(restored_acc) > np.mean(rand_rf) + 2 * np.std(rand_rf)),
           "arm4_readout_vs_content": arm4,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (out / "latloc_arm2.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[arm2] banked; S-restore RF={rf(restored_acc):.3f} vs random "
          f"{np.mean(rand_rf):.3f}±{np.std(rand_rf):.3f} beats_random={row['beats_random']}")


if __name__ == "__main__":
    main()
