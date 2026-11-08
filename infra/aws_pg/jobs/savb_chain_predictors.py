"""SAV-B chain predictors — PG1/pi0/pi05 L2/LMC/overlap ONLY (outcomes are the
SAV-A frozen confirmation cells; prereg 2026-06-12). Same predictor block as
savb_unit (imported), same frozen prompts, same grid-bucket LMC arithmetic.

    python -m infra.aws_pg.jobs.savb_chain_predictors --out <dir> \
        --controls-s3 <sav_controls.json> [--pi05-sealed-s3 s3://...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--pi05-sealed-s3", default="")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import savb as SBV
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.savb_unit import hidden_basis
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _extract_decoder

    assert torch.cuda.is_available()
    device = "cuda"

    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "sav_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True
    hashes = C["item_hashes"]
    grid_hashes = W.mask_by(hashes, hashes, W.is_grid_subset)
    b_grid = W.mask_by(C["BASE"]["per_item"], hashes, W.is_grid_subset)
    acc_base_grid = float(np.mean(b_grid) * 100)

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    wiki = load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG)["test"]["text"]
    prompts, prompt_sha = SBV.wiki_prompts(wiki)

    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    dec_b = {k: v.cpu() for k, v in _bare_decoder(base.model.state_dict()).items()}
    basis_b, n_tok_b = hidden_basis(base, tok, device, prompts)
    del base
    torch.cuda.empty_cache()

    pi05_local = ""
    if args.pi05_sealed_s3:
        pi05_local = str(out / "pi05_decoder.safetensors")
        subprocess.run(["aws", "s3", "cp", args.pi05_sealed_s3, pi05_local,
                        "--region", "us-east-1"], check=True)

    rows = {}
    for name in ("PG1", "pi0", "pi05"):
        dec_u, src_sha = _extract_decoder(name, device, pi05_local)
        l2 = SBV.l2_to_base(dec_u, dec_b)
        mid = SBV.midpoint_decoder(dec_u, dec_b)
        m_mid, _ = SB.load_swapback_model(mid, device)
        del mid
        ev_mid = TE.mmlu_choice_logits(m_mid, tok, device, item_filter=W.is_grid_subset)
        assert ev_mid["per_item_hash"] == grid_hashes, f"{name} midpoint identity drift — HALT"
        del m_mid
        torch.cuda.empty_cache()
        model, _ = SB.load_swapback_model(dec_u, device)
        del dec_u
        basis_u, n_tok_u = hidden_basis(model, tok, device, prompts)
        assert n_tok_u == n_tok_b, f"{name}: hidden token-count mismatch"
        overlap = SBV.subspace_overlap(basis_b, basis_u)
        ev_unit = TE.mmlu_choice_logits(model, tok, device, item_filter=W.is_grid_subset)
        assert ev_unit["per_item_hash"] == grid_hashes, f"{name} unit identity drift — HALT"
        del model
        torch.cuda.empty_cache()
        barrier = SBV.lmc_barrier(ev_mid["acc_pct"], acc_base_grid, ev_unit["acc_pct"])
        rows[name] = {"unit": name, "kind": "chain", "source_sha": str(src_sha),
                      "predictors": {"l2_base": l2, "lmc_barrier": barrier, "overlap": overlap,
                                     "acc_grid_mid": ev_mid["acc_pct"],
                                     "acc_grid_unit": ev_unit["acc_pct"],
                                     "acc_grid_base_from_controls": acc_base_grid,
                                     "prompt_sha": prompt_sha, "n_hidden_tokens": n_tok_u,
                                     "overlap_layer": SBV.OVERLAP_LAYER,
                                     "overlap_k": SBV.OVERLAP_K}}
        print(f"[SAVB-CHAIN] {name}: L2={l2:.3f} overlap={overlap:.4f} barrier={barrier:+.2f} "
              f"(mid={ev_mid['acc_pct']:.2f} unit={ev_unit['acc_pct']:.2f} base={acc_base_grid:.2f})")

    (out / "chain_predictors.json").write_text(json.dumps(
        {"rows": rows, "controls_s3": args.controls_s3,
         "torch": torch.__version__}, indent=1, default=str))
    print("[SAVB-CHAIN] done")


if __name__ == "__main__":
    main()
