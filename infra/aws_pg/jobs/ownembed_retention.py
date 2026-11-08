"""OWN-EMBED retention (RESEARCH_INDEX §5.1) — the decisive disambiguation VLAT left un-run.

VLAT read every checkpoint's decoder under the BASE gemma-2b embed (a fixed instrument) and found
π0/π0.5 at floor — but could not tell *truly-deleted* from *co-adapted-away-from-base-embed*. The
own-embed feasibility gate (ownembed_extract_gate) established that each checkpoint's OWN trained tied
text embedding survives as `lm_head.weight` (vocab 257152) — so the policy actually runs π0-decoder +
π0-own-embed. This job reads each checkpoint's decoder under ITS OWN embedding and computes S_text.

  own ≫ base-embed (and ≫ VLAT's ~0.05) ⇒ co-adapted/preserved (the base-embed deletion was a
                                            readout artifact; reasoning survives under the own interface)
  own ≈ floor (S_text ≤ θ)             ⇒ genuine deletion even under the model's own embedding

Decision rule INHERITED from VLAT @ c9c8773 (recompute from results/vlat/metrics.json; never transcribe):
S_text=(c−FLOOR)/(BASE−FLOOR), θ_forget=0.10. Firewall: Gemma-1 only. Faithful (own-embed was
co-trained with the decoder — not a proxy). NEW: the 257152-vocab own-embed shell builder.

  python -m infra.aws_pg.jobs.ownembed_retention --out <dir> --checkpoints pi0,pi05,base,floor
"""
from __future__ import annotations

import argparse
import json
import pathlib

PI0_REPO = "lerobot/pi0_base"
PI05_REPO = "lerobot/pi05_base"


def _own_head_key(keys):
    cands = [k for k in keys if "lm_head" in k and "gemma_expert" not in k
             and "action_expert" not in k and "vision_tower" not in k]
    assert len(cands) >= 1, f"no backbone lm_head key found (keys e.g. {list(keys)[:4]})"
    return sorted(cands, key=len)[0]


def _extract_own(name, device):
    """Return (bare_decoder_sd, own_head_tensor, sha) for π0/π0.5 — decoder via pi0_port.remap,
    own embed via the tied lm_head."""
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg import pi0_port
    api = HfApi()
    repo = PI0_REPO if name == "pi0" else PI05_REPO
    if name == "pi0":
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
    else:
        sd = {}
        for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
            sd.update(load_file(hf_hub_download(repo, f)))
    dec = pi0_port.remap_backbone_lm(sd)
    head = sd[_own_head_key(sd.keys())]
    return dec, head, api.model_info(repo, revision="main").sha


def _build_ownembed_model(decoder_sd, own_head, device):
    """NEW own-embed shell: a base Gemma-1 (gemma-2b) resized to the own-embed vocab, with the
    checkpoint's OWN tied embed/head attached and its decoder swapped in (vlat_swapback leak gates
    reused). The faithful π0/π0.5 backbone LM as deployed."""
    import torch
    from transformers import AutoModelForCausalLM
    from infra.pg.entities import ENTITIES
    from infra.pg import vlat_swapback as SB

    ent = ENTITIES["L0_g1"]
    m = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    V = int(own_head.shape[0])
    m.resize_token_embeddings(V)
    with torch.no_grad():
        m.get_input_embeddings().weight.copy_(own_head.to(device).to(torch.bfloat16))
    m.tie_weights()                                   # lm_head ← own embed (tied)
    base_sd = dict(m.state_dict())                    # FULL model keys: model.layers… / model.embed_tokens / lm_head
    dec = SB.prefix_decoder_sd({k: v.to(device) for k, v in decoder_sd.items()})  # model.layers…
    new_sd, info = SB.build_swapback_sd(base_sd, dec)  # keep (own) embed/head, swap decoder; leak-gated
    missing, unexpected = m.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, f"own-embed load not strict-clean: {missing} {unexpected}"
    return m, {**info, "vocab": V}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="base,floor,pi0,pi05")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full split (VLAT parity)")
    ap.add_argument("--floor-seed", type=int, default=1337)
    ap.add_argument("--theta", type=float, default=0.10, help="VLAT θ_forget (inherited)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in a.checkpoints.split(",") if c.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _random_decoder_sd

    assert torch.cuda.is_available(), "own-embed retention needs GPU"
    device = "cuda"
    n_per = a.mmlu_n_per_subject or None
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)  # base tokenizer (text rows < 256000)
    print(f"[OWN-RET] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} ckpts={requested} n_per={n_per}")

    acc = {}
    meta = {}
    for name in requested:
        if name == "base":                          # ceiling: base under its OWN (base) embed
            model = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16,
                device_map=device, attn_implementation="eager").eval()
            meta[name] = {"readout": "base-embed (own)"}
        elif name == "floor":                       # random decoder under base embed (chance)
            b = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16, device_map="cpu")
            dec = _random_decoder_sd(_bare_decoder(b.state_dict()), a.floor_seed)
            del b
            model, info = SB.load_swapback_model(dec, device)
            meta[name] = {"readout": "random-decoder + base-embed", **info}
        else:                                       # pi0 / pi05 under their OWN embed
            dec, own_head, sha = _extract_own(name, device)
            model, info = _build_ownembed_model(dec, own_head, device)
            meta[name] = {"readout": "own-embed (tied lm_head)", "source_sha": sha, **info}
            print(f"[OWN-RET] {name}: own-embed shell vocab={info['vocab']} decoder swapped={info['n_decoder_keys_swapped']}")
        mm = TE.mmlu_accuracy(model, tok, device, n_per_subject=n_per)
        assert mm["n"] > 0
        acc[name] = mm["acc_pct"]
        meta[name]["mmlu"] = {k: mm[k] for k in ("acc_pct", "n", "per_item", "per_item_hash")}
        print(f"[OWN-RET] {name:5s} MMLU(own-readout)={mm['acc_pct']:.2f} (n={mm['n']})")
        del model
        torch.cuda.empty_cache()

    # S_text against the in-run base/floor (consistent instrument)
    result = {"torch": torch.__version__, "theta_inherited": a.theta, "acc_pct": acc, "meta": meta,
              "vlat_baseembed_ref": {"base": 40.97, "floor": 24.33, "pi0_S": 0.051, "pi05_S": 0.056}}
    if "base" in acc and "floor" in acc:
        BASE, FLOOR = acc["base"], acc["floor"]
        span = BASE - FLOOR
        result["S_text_ownembed"] = {}
        for name in ("pi0", "pi05"):
            if name in acc:
                S = (acc[name] - FLOOR) / span if span > 0 else float("nan")
                verdict = ("RETAINED-under-own-embed (preserved; base-embed deletion was a readout artifact)"
                           if S > a.theta else
                           "DELETED-under-own-embed (genuine deletion even under the model's own embedding)")
                result["S_text_ownembed"][name] = {"S_text": S, "acc": acc[name],
                                                   "base": BASE, "floor": FLOOR, "verdict": verdict}
                print(f"[OWN-RET] {name} own-embed S_text={S:.3f} (vs VLAT base-embed {result['vlat_baseembed_ref'][name+'_S']}) → {verdict}")
    (out / "ownembed_retention.json").write_text(json.dumps(result, indent=2, default=str))
    print("[OWN-RET] DONE", json.dumps(acc))


if __name__ == "__main__":
    main()
