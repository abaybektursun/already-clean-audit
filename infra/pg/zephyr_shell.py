"""W18 engineered-system shell helpers (Zephyr/Mistral-7B; pure where possible).

  * bare_decoder_sd  — strip a MistralForCausalLM state_dict to bare decoder
    keys (model.* minus embed_tokens), mirroring the Gemma conventions.
  * random_decoder_sd — the W18 FLOOR: every 2-D decoder matrix → scale-matched
    Gaussian (per-tensor std), 1-D norms + embed/head kept (the W11/W14/W17
    UNTRAINED-floor convention; deterministic in seed).
  * load_zephyr — base / RMU / floor loader with strict-clean swap gates for
    the floor (embed/head bitwise-kept; decoder actually randomized).

Probe/adapter scope on the loaded model = `model.model` (the Mistral decoder),
where `relearn.adapter_target_names` / `w16_probes.probe_param_names` apply
unchanged (2-D, embed/lm_head excluded). Heavy imports inside functions.
"""
from __future__ import annotations

ZEPHYR_BASE = "HuggingFaceH4/zephyr-7b-beta"
ZEPHYR_RMU = "cais/Zephyr_RMU"
EMBED_KEYS = ("embed_tokens", "lm_head")


def _is_embed(k: str) -> bool:
    return any(t in k for t in EMBED_KEYS)


def bare_decoder_sd(lm_sd: dict) -> dict:
    """model.*-prefixed keys minus embeddings → bare decoder dict (prefix kept)."""
    return {k: v for k, v in lm_sd.items() if k.startswith("model.") and not _is_embed(k)}


def random_decoder_sd(bare_sd: dict, seed: int) -> dict:
    """Scale-matched Gaussian for every 2-D matrix; 1-D kept. Deterministic."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = {}
    for k in sorted(bare_sd):
        v = bare_sd[k]
        vf = v.detach().cpu().float()
        if vf.ndim >= 2:
            std = vf.std().clamp_min(1e-8)
            out[k] = (torch.randn(vf.shape, generator=g) * std).to(v.dtype)
        else:
            out[k] = v
    return out


def load_zephyr(which: str, device, floor_seed: int = 1337):
    """which ∈ {base, rmu, floor}. floor = base shell with randomized decoder
    (strict-clean load; embed/head bitwise-kept gate; decoder-changed gate).
    Returns (model, info)."""
    import torch
    from transformers import AutoModelForCausalLM

    repo = {"base": ZEPHYR_BASE, "rmu": ZEPHYR_RMU, "floor": ZEPHYR_BASE}[which]
    m = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa").eval()
    info = {"repo": repo, "which": which}
    if which != "floor":
        return m, info
    sd = m.state_dict()                      # live references — clone what we compare
    pre_embed = sd["model.embed_tokens.weight"].detach().cpu().clone()
    ref_key = sorted(k for k in bare_decoder_sd(sd) if sd[k].ndim >= 2)[0]
    pre_ref = sd[ref_key].detach().cpu().clone()
    rnd = random_decoder_sd(bare_decoder_sd(sd), floor_seed)
    missing, unexpected = m.load_state_dict({**sd, **{k: v.to(device) for k, v in rnd.items()}},
                                            strict=True)
    assert not missing and not unexpected, f"floor load not strict-clean: {missing} {unexpected}"
    now = m.state_dict()
    assert torch.equal(now["model.embed_tokens.weight"].detach().cpu(), pre_embed), \
        "floor changed embed_tokens — must be KEPT"
    delta = (now[ref_key].detach().cpu().float() - pre_ref.float()).abs().max().item()
    assert delta > 0, "floor decoder unchanged — randomization did nothing"
    info["floor_seed"] = floor_seed
    info["decoder_max_abs_delta"] = delta
    return m, info
