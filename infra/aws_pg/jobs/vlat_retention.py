"""VLAT §9.5 — Phase 1 RETENTION (the trajectory readout). GPU, fail-loud.

For each requested checkpoint, build the runnable Gemma-1 text model via the swap-back shell
(base gemma-2b embed_tokens + lm_head reattached to the checkpoint's EXTRACTED 18×7 decoder + norms),
then measure free-standing reasoning with the SAME fixed instrument: MMLU 5-shot (text_eval.mmlu_accuracy)
and wikitext-2 perplexity (clean_text_perplexity). Emits raw per-checkpoint metrics + per-item hashes;
S_text(c) = (c − FLOOR)/(BASE − FLOOR) and the deletion labels are computed downstream (validated §9.1).

Checkpoints (extraction path):
  base  google/gemma-2b              — the fixed instrument itself (run directly; S_text ref = 1.0 = BASE).
  PG1   google/paligemma-3b-pt-224   — language_model.model decoder, embed_tokens dropped.
  pi0   lerobot/pi0_base             — pi0_port.remap_backbone_lm (embed dropped by the port).
  pi05  lerobot/pi05_base            — sealed decoder from vlat_pi05_extract_gate (S3), or remap fresh.
  floor random decoder              — base decoder matrices replaced by scale-matched Gaussian (FLOOR).

OOD/suppression control: perplexity is the text-shell-appropriate discriminator (MMLU-at-floor WITH
elevated ppl → deletion; MMLU-floor but ppl near-base → suppression). Blank-image (W11) does not apply
to a decoder-only text shell — recorded as a §9.3 refinement.

Firewall: discovery set {base, PG1} (+ floor) vs sealed confirmation set {pi0, pi05} are run as SEPARATE
invocations; the analyst registers the recipe prediction (§9.3) before the {pi0,pi05} invocation.

  pg run infra.aws_pg.jobs.vlat_retention --args "--checkpoints base,PG1,floor" \
      --type g6e.8xlarge,p5.4xlarge --s3-prefix results/vlat_retention_disc --budget-min 180
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
PI05_REPO = "lerobot/pi05_base"


def _bare_decoder(sd):
    """Strip a leading 'model.' and drop embed_tokens/lm_head → pi0_port-style bare decoder keys."""
    out = {}
    for k, v in sd.items():
        if "embed_tokens" in k or "lm_head" in k:
            continue
        out[k[len("model."):] if k.startswith("model.") else k] = v
    return out


def _random_decoder_sd(base_decoder, seed):
    """FLOOR: replace each 2-D decoder matrix with a per-matrix std-matched Gaussian (frozen seed);
    keep 1-D norms at base (randomizing scale params just destroys the forward, not the reasoning)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, w in base_decoder.items():
        if w.ndim == 2:
            std = w.detach().float().std().item()
            out[k] = (torch.randn(tuple(w.shape), generator=g, dtype=torch.float32) * std).to(w.dtype)
        else:
            out[k] = w.detach().clone()
    return out


def _extract_decoder(name, device, pi05_sealed_local):
    """Return (bare_decoder_sd, source_sha) for a checkpoint, or (None, sha) for base (run shell directly)."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg import pi0_port
    from infra.pg.entities import ENTITIES, resolve_submodule_paths
    api = HfApi()

    if name == "base":
        return None, api.model_info(ENTITIES["L0_g1"].repo_id, revision=ENTITIES["L0_g1"].revision or "main").sha
    if name == "PG1":
        repo = ENTITIES["PG1"].repo_id
        pg = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
        lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
        if hasattr(lm, "lm_head") and hasattr(lm, "model"):
            lm = lm.model
        dec = _bare_decoder(lm.state_dict())
        del pg
        return dec, api.model_info(repo).sha
    if name == "pi0":
        repo = ENTITIES["PI0"].repo_id
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        return pi0_port.remap_backbone_lm(sd), api.model_info(repo).sha
    if name == "pi05":
        sha = api.model_info(PI05_REPO, revision="main").sha
        if pi05_sealed_local and pathlib.Path(pi05_sealed_local).exists():
            return {k: v for k, v in load_file(pi05_sealed_local).items()}, sha
        merged = {}
        for f in [x for x in list_repo_files(PI05_REPO) if x.endswith(".safetensors")]:
            merged.update(load_file(hf_hub_download(PI05_REPO, f)))
        return pi0_port.remap_backbone_lm(merged), sha    # prefix-match asserted upstream by the gate
    raise ValueError(f"unknown checkpoint {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", required=True, help="comma list ⊆ {base,PG1,pi0,pi05,floor}")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full test split")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--floor-seed", type=int, default=1337)
    ap.add_argument("--pi05-sealed-s3", default="",
                    help="optional s3:// path to the sealed pi05 decoder from vlat_pi05_extract_gate")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in a.checkpoints.split(",") if c.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "VLAT retention needs GPU (model forwards)."
    device = "cuda"
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)  # fixed-instrument tokenizer
    print(f"[VLAT-RET] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} ckpts={requested}")

    pi05_local = ""
    if "pi05" in requested and a.pi05_sealed_s3:
        pi05_local = str(out / "pi05_backbone_lm_sd.safetensors")
        subprocess.run(["aws", "s3", "cp", a.pi05_sealed_s3, pi05_local, "--region", "us-east-1"], check=True)

    n_per = a.mmlu_n_per_subject or None
    results = {}
    for name in requested:
        if name == "floor":
            base = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16, device_map="cpu")
            dec = _random_decoder_sd(_bare_decoder(base.state_dict()), a.floor_seed)
            del base
            sha = f"random-decoder:seed={a.floor_seed}"
            model, info = SB.load_swapback_model(dec, device)
        else:
            dec, sha = _extract_decoder(name, device, pi05_local)   # base → (None, sha); reals → (dec, sha)
            if name == "base":
                model = AutoModelForCausalLM.from_pretrained(
                    L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16,
                    device_map=device, attn_implementation="eager").eval()
                info = {"path": "base shell direct (instrument reference)"}
            else:
                model, info = SB.load_swapback_model(dec, device)
        mmlu = TE.mmlu_accuracy(model, tok, device, n_per_subject=n_per)
        ppl = TE.clean_text_perplexity(model, tok, device, max_tokens=a.ppl_max_tokens)
        results[name] = {
            "source_sha": sha, "swapback_info": info,
            "mmlu": {k: mmlu[k] for k in ("acc_pct", "n", "n_subjects", "n_shots", "n_per_subject",
                                          "per_subject", "per_item", "per_item_hash", "letter_token_ids")},
            "ppl": {k: ppl[k] for k in ("ppl", "mean_nll", "n_tokens", "window")}}
        print(f"[VLAT-RET] {name:5s} MMLU={mmlu['acc_pct']:.2f} (n={mmlu['n']})  ppl={ppl['ppl']:.3f}")
        del model
        torch.cuda.empty_cache()

    payload = {"torch": torch.__version__, "instrument": L0g1.repo_id, "checkpoints": requested,
               "mmlu_n_per_subject": a.mmlu_n_per_subject, "floor_seed": a.floor_seed, "results": results}
    (out / "retention.json").write_text(json.dumps(payload, indent=2, default=str))
    print("[VLAT-RET] RETENTION_OK", json.dumps({n: round(results[n]["mmlu"]["acc_pct"], 2) for n in results}))


if __name__ == "__main__":
    main()
