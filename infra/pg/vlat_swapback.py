"""VLAT swap-back shell — the W11-A decoder-isolation method, re-wired for the Gemma-1 lineage.

The π0/π0.5 port DROPS the tied embed_tokens and has no lm_head (pi0_port docstring; openpi #669),
so a checkpoint's extracted backbone decoder is not directly runnable as a text model. This module
builds a RUNNABLE Gemma-1 text model from an extracted decoder by reattaching the BASE gemma-2b
(Gemma-1, 18L) embed_tokens + lm_head — ONE fixed instrument across all four checkpoints, so any
S_text difference is attributable to the DECODER weights alone (prereg §3).

This INVERTS contrast_a1._lm_swap_state_dict, which keeps PG's extended-vocab embed/head and resets
the DECODER to base. Here the source-of-each-half flips: the embed/head are the FIXED base gemma-2b
and the decoder is the per-checkpoint VARIABLE. The keep/swap logic, the strict-load gate, the
shape-mismatch raise, the partial-swap leak guard and the finite-logits forward check port verbatim.

Base = ENTITIES["L0_g1"] = google/gemma-2b  (Gemma-1, 18L).
NOT  = ENTITIES["L0"]    = google/gemma-2-2b (Gemma-2, 26L — the PaliGemma-2 / W-series base).

Discipline (CLAUDE.md): no try/except hiding; any mismatch RAISES with a trace. Heavy imports live
inside functions so this module py_compiles on a torch-free control plane.
"""
from __future__ import annotations

# Substrings that must NEVER appear in a key entering the runnable text shell — the action expert /
# policy head live in the π0/π0.5 checkpoints and we measure the BACKBONE, never the policy
# (prereg §13:217-218; pi0_port asserts the same on extraction).
ACTION_EXPERT_MARKERS = ("gemma_expert", "action_expert", "expert", "vision_tower")
# A key is part of the fixed instrument (kept from base) iff it names the embedding or output head.
EMBED_KEYS = ("embed_tokens", "lm_head")


def assert_no_action_expert(sd):
    """Fail-loud if any action-expert / policy / vision param leaked into the swap-back state_dict."""
    leaked = sorted(k for k in sd if any(m in k for m in ACTION_EXPERT_MARKERS))
    assert not leaked, f"VLAT swap-back: non-backbone keys leaked into the shell: {leaked[:8]}"


def build_swapback_sd(base_sd, decoder_sd):
    """Build the state_dict to load into a fresh base Gemma-1 shell (google/gemma-2b).

    KEEP base embed_tokens + lm_head (the fixed instrument); REPLACE every other key (decoder layers
    + per-layer norms + final norm) with the checkpoint's EXTRACTED decoder. The result has EXACTLY
    the base shell's keys/shapes, so a subsequent load_state_dict(strict=True) is clean by construction.

    `decoder_sd` keys must live in the SAME namespace as `base_sd` (the caller prefixes the bare
    pi0_port keys with the shell prefix — see `prefix_decoder_sd`). Fail-loud on:
      * a base decoder key absent from the extracted decoder (partial swap → base leaks into contrast),
      * an extracted key the base shell does not expect (un-stripped prefix / stray param),
      * a shape mismatch (architecture misalignment),
      * any action-expert / vision key (compartment leak).
    Returns (new_sd, info).
    """
    assert_no_action_expert(decoder_sd)
    new_sd, kept, swapped, missing = {}, [], [], []
    for k, base_w in base_sd.items():
        if any(e in k for e in EMBED_KEYS):
            new_sd[k] = base_w                     # KEEP base embed/head verbatim (fixed instrument)
            kept.append(k)
            continue
        if k not in decoder_sd:
            missing.append(k)
            continue
        ext_w = decoder_sd[k]
        if tuple(ext_w.shape) != tuple(base_w.shape):
            raise RuntimeError(
                f"VLAT swap-back: shape mismatch on decoder key {k!r}: "
                f"extracted={list(ext_w.shape)} base={list(base_w.shape)}. Architectures must align.")
        new_sd[k] = ext_w                          # SWAP in the per-checkpoint decoder
        swapped.append(k)
    if missing:
        raise RuntimeError(
            f"VLAT swap-back: {len(missing)} base decoder keys absent from the extracted decoder — "
            f"partial swap would leak base weights into the contrast. Examples: {missing[:8]}")
    extra = sorted(set(decoder_sd) - set(swapped))
    if extra:
        raise RuntimeError(
            f"VLAT swap-back: {len(extra)} extracted-decoder keys unused by the base shell "
            f"(prefix not stripped / unexpected param). Examples: {extra[:8]}")
    info = {"n_decoder_keys_swapped": len(swapped),
            "n_embed_keys_kept": len(kept),
            "embed_keys_kept": sorted(kept)}
    return new_sd, info


def prefix_decoder_sd(decoder_sd, prefix="model."):
    """pi0_port.remap_backbone_lm yields BARE GemmaModel keys (`layers.0…`, `norm.weight`). A base
    Gemma-1 loaded as AutoModelForCausalLM nests them under `model.` (and exposes a top-level
    `lm_head`). Re-key the extracted decoder into the shell namespace. Idempotent per key."""
    return {(k if k.startswith(prefix) else prefix + k): v for k, v in decoder_sd.items()}


def load_swapback_model(decoder_sd, device, base_repo=None, base_revision=None,
                        require_decoder_changed=True):
    """Instantiate a fresh base Gemma-1 shell, swap in the extracted decoder, strict-load, and gate:
    0 missing/unexpected, embed/head bitwise-identical to base, decoder actually changed, finite
    logits on a forward. Returns (model, info). The runnable instrument for vlat_retention.

    `decoder_sd` = pi0_port.remap_backbone_lm output (bare keys, embed_tokens already dropped).
    """
    import torch
    from transformers import AutoModelForCausalLM

    from infra.pg.entities import ENTITIES
    ent = ENTITIES["L0_g1"]                          # google/gemma-2b (Gemma-1, 18L)
    repo = base_repo or ent.repo_id
    rev = base_revision if base_revision is not None else ent.revision

    shell = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()  # eager: Gemma attn parity w/ W11
    base_sd = {k: v.detach().clone() for k, v in shell.state_dict().items()}

    dec = prefix_decoder_sd({k: v.to(device) for k, v in decoder_sd.items()})
    new_sd, info = build_swapback_sd(shell.state_dict(), dec)
    missing, unexpected = shell.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, \
        f"VLAT swap-back load not strict-clean: missing={missing} unexpected={unexpected}"

    now = shell.state_dict()
    emb_keys = [k for k in base_sd if any(e in k for e in EMBED_KEYS)]
    for k in emb_keys:
        assert torch.equal(now[k].detach().cpu(), base_sd[k].detach().cpu()), \
            f"VLAT swap-back: base embed/head key {k} changed — must be the fixed instrument."
    if require_decoder_changed:
        dec_keys = [k for k in base_sd if not any(e in k for e in EMBED_KEYS)]
        max_delta = max((now[k].detach().cpu().float() - base_sd[k].detach().cpu().float())
                        .abs().max().item() for k in dec_keys)
        assert max_delta > 0.0, \
            f"VLAT swap-back: decoder is bitwise-equal to base (swap did nothing). max|Δ|={max_delta}"
        info["decoder_max_abs_delta_vs_base"] = max_delta
    return shell, info


# ---------------------------------------------------------------------------
# Scalar instruments shared by the retention job and the §9.1 dry-run gate.
# ---------------------------------------------------------------------------

def s_text(c, base, floor):
    """Retained-reasoning fraction (prereg §2): S_text(c) = (c − FLOOR)/(BASE − FLOOR). Raw, unclamped
    (a value below FLOOR is meaningful). BASE → 1.0, FLOOR → 0.0. Fail-loud on a degenerate band."""
    span = base - floor
    assert span > 0, f"VLAT s_text: BASE ({base}) must exceed FLOOR ({floor}) — degenerate power band."
    return (c - floor) / span


def split_bucket(per_item_hash, frac_discovery=0.5, salt="vlat"):
    """Deterministic discovery/confirmation assignment for one MMLU item (prereg §4.1: sha1-bucket the
    per_item_hash per checkpoint). Stable, salt-namespaced, independent of the per_item_hash's own
    salting. Returns 'discovery' or 'confirmation'."""
    import hashlib
    h = hashlib.sha1(f"{salt}:{per_item_hash}".encode()).hexdigest()
    u = int(h[:13], 16) / float(1 << 52)            # 52 bits → uniform in [0, 1)
    return "discovery" if u < frac_discovery else "confirmation"
