"""SAV-B unit runner — ONE N5C ft-task checkpoint (prereg 2026-06-12).

Per unit: extract the Gemma-1 LM decoder from google/paligemma-3b-ft-{task}-224
(PG1-branch extraction path), then
  predictors: L2-to-base (fp32 Frobenius over 2-D mats), LMC barrier on the MMLU
    grid bucket (midpoint shell eval; base acc from frozen controls per-item),
    layer-9 top-16 subspace overlap vs base (64 frozen wikitext-2 prompts);
  outcomes: r=512 adapters, SAV frozen recipe (from controls), n ∈ {1000, 16000},
    seed 1337, confirmation-half eval → S_recover vs frozen BASE/FLOOR arrays.
Shell rebuilt from the kept decoder sd between the two trainings. Fail-loud.

    python -m infra.aws_pg.jobs.savb_unit --out <dir> --controls-s3 <sav_controls.json> \
        --ft-repo google/paligemma-3b-ft-okvqa-224 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

SAVB_NS = (1000, 16000)


def hidden_basis(model, tok, device, prompts):
    """Layer-OVERLAP_LAYER masked token hiddens on the frozen prompts → top-k right basis."""
    import numpy as np
    import torch

    from infra.pg import savb as SBV
    rows = []
    model.eval()
    for i in range(0, len(prompts), 8):
        enc = tok(prompts[i:i + 8], return_tensors="pt", padding=True,
                  truncation=True, max_length=SBV.WIKI_MAX_TOKENS).to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[SBV.OVERLAP_LAYER]
        mask = enc["attention_mask"].bool()
        rows.append(hs[mask].float().cpu().numpy())
    X = np.concatenate(rows, axis=0)
    return SBV.topk_right_basis(X), int(X.shape[0])


def extract_ft_decoder(repo):
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg.entities import resolve_submodule_paths
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    dec = _bare_decoder(lm.state_dict())
    del pg
    return dec, HfApi().model_info(repo).sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--ft-repo", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rank", type=int, default=512)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import savb as SBV
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import boot_s_text, s_text
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder

    assert torch.cuda.is_available()
    device = "cuda"
    task = args.ft_repo.split("/")[-1]
    print(f"[SAVB-UNIT] {task} seed={args.seed} r={args.rank}")

    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "sav_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    fr = C["recipe"]
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]
    conf_hashes = W.mask_by(hashes, hashes, W.is_confirmation)
    b_conf = W.mask_by(b_pi, hashes, W.is_confirmation)
    f_conf = W.mask_by(f_pi, hashes, W.is_confirmation)
    grid_hashes = W.mask_by(hashes, hashes, W.is_grid_subset)
    b_grid = W.mask_by(b_pi, hashes, W.is_grid_subset)
    import numpy as np
    acc_base_grid = float(np.mean(b_grid) * 100)

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)

    # ---- base side: decoder sd + frozen-prompt basis ----
    wiki = load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG)["test"]["text"]
    prompts, prompt_sha = SBV.wiki_prompts(wiki)
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    dec_b = {k: v.cpu() for k, v in _bare_decoder(base.model.state_dict()).items()}
    basis_b, n_tok_b = hidden_basis(base, tok, device, prompts)
    del base
    torch.cuda.empty_cache()

    # ---- unit decoder + weight-space predictors ----
    dec_u, src_sha = extract_ft_decoder(args.ft_repo)
    l2 = SBV.l2_to_base(dec_u, dec_b)
    print(f"[SAVB-UNIT] {task} L2-to-base={l2:.3f}")

    mid = SBV.midpoint_decoder(dec_u, dec_b)
    del dec_b
    m_mid, _ = SB.load_swapback_model(mid, device)
    del mid
    ev_mid = TE.mmlu_choice_logits(m_mid, tok, device, item_filter=W.is_grid_subset)
    assert ev_mid["per_item_hash"] == grid_hashes, "midpoint grid identity drift — HALT"
    del m_mid
    torch.cuda.empty_cache()

    # ---- unit shell: overlap + unprobed bucket acc (LMC ingredient + n0 texture) ----
    model, swap_info = SB.load_swapback_model(dec_u, device)
    basis_u, n_tok_u = hidden_basis(model, tok, device, prompts)
    assert n_tok_u == n_tok_b, "token-count mismatch between base/unit hidden passes"
    overlap = SBV.subspace_overlap(basis_b, basis_u)
    ev_unit = TE.mmlu_choice_logits(model, tok, device, item_filter=W.is_grid_subset)
    assert ev_unit["per_item_hash"] == grid_hashes, "unit grid identity drift — HALT"
    barrier = SBV.lmc_barrier(ev_mid["acc_pct"], acc_base_grid, ev_unit["acc_pct"])
    print(f"[SAVB-UNIT] {task} overlap={overlap:.4f} LMC: mid={ev_mid['acc_pct']:.2f} "
          f"base={acc_base_grid:.2f} unit={ev_unit['acc_pct']:.2f} barrier={barrier:+.2f}")

    # ---- outcomes: train r512 at each n (fresh shell per n), confirmation eval ----
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    outcomes = {}
    for n in SAVB_NS:
        if n != SAVB_NS[0]:
            del model
            torch.cuda.empty_cache()
            model, swap_info = SB.load_swapback_model(dec_u, device)
        arep = R.attach_adapters(model.model, rank=args.rank, seed=args.seed,
                                 alpha=float(args.rank))
        model.to(device)               # adapter masters → cuda (sav_cell pattern)
        sub = R.ladder_subset(ded["kept"], n)
        examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        train_rep = R.train_adapter_qa(model, tok, device, examples, lr=fr["lr"],
                                       epochs=fr["epochs"], batch_size=fr["batch"],
                                       warmup_ratio=fr["warmup"], seed=args.seed,
                                       micro_batch=2)
        ev = TE.mmlu_choice_logits(model, tok, device, item_filter=W.is_confirmation)
        assert ev["per_item_hash"] == conf_hashes, f"{task} n={n}: conf identity drift — HALT"
        S = s_text(ev["per_item"], b_conf, f_conf)
        Sb = boot_s_text(ev["per_item"], b_conf, f_conf)
        outcomes[str(n)] = {"acc_pct": ev["acc_pct"], "S_recover": S,
                            "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                            "train": train_rep, "n_train": len(examples),
                            "adapter": {k: arep[k] for k in
                                        ("n_modules", "trainable_params", "rank", "alpha")},
                            "per_item": ev["per_item"]}
        print(f"[SAVB-UNIT] {task} n={n}: acc={ev['acc_pct']:.2f} S={S:.4f} "
              f"[{Sb[1]:.4f},{Sb[2]:.4f}] CE_last={train_rep['final_loss']:.3f}")

    row = {"unit": task, "ft_repo": args.ft_repo, "source_sha": str(src_sha),
           "kind": "n5c", "seed": args.seed, "rank": args.rank,
           "recipe": fr, "swap_gate": {k: v for k, v in swap_info.items()},
           "predictors": {"l2_base": l2, "lmc_barrier": barrier, "overlap": overlap,
                          "acc_grid_mid": ev_mid["acc_pct"], "acc_grid_unit": ev_unit["acc_pct"],
                          "acc_grid_base_from_controls": acc_base_grid,
                          "prompt_sha": prompt_sha, "n_hidden_tokens": n_tok_u,
                          "overlap_layer": SBV.OVERLAP_LAYER, "overlap_k": SBV.OVERLAP_K},
           "outcomes": outcomes, "dedup": {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]},
           "controls_s3": args.controls_s3, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    (out / f"unit_{task}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[SAVB-UNIT] {task} done")


if __name__ == "__main__":
    main()
