"""Canonical entities + G0 verification — PaliGemma 2 feature-lineage study.

Single source of truth for the §3 entity table of
``research/2026-05-26_paligemma_before_after_interp_spec.md``.

Discipline (CLAUDE.md, spec §3/§9):
  * Do NOT hardcode HF submodule names anywhere else — introspect them here.
  * Do NOT bake checkpoint revisions into prompts — pin them at G0; the live
    commit SHA is resolved by ``verify_entities`` and written to the manifest.
  * Fail loud: a wrong assumption raises with a trace, never a silent fallback.

Module-level imports are kept light (dataclass only) so a *local* entrypoint can
read the registry without a torch/transformers install. Heavy imports live inside
``verify_entities``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    symbol: str
    repo_id: str | None          # None → resolved at G0 (e.g. the SigLIP init)
    role: str
    revision: str | None = None  # pinned commit SHA; filled by verify_entities


# §3 table. Every repo_id below was verified to exist and is GATED — the job
# needs HF_TOKEN set and the Google license accepted, or downloads return 401.
ENTITIES: dict[str, Entity] = {
    # --- PaliGemma-2 program: Gemma-2-2B LM (26 decoder layers), the NEWER generation.
    #     LM init = google/gemma-2-2b (NOT google/gemma-2b — that one-char-shorter id is Gemma-1,
    #     used by the pi-trajectory rows below). Do NOT diff these rows against those (different gen).
    "PG":     Entity("PG",     "google/paligemma2-3b-pt-224",
                     "AFTER — primary subject (pt: isolates the Stage1-2 combination)"),
    "PG_mix": Entity("PG_mix", "google/paligemma2-3b-mix-224",
                     "task-tuned — W6 PT→Mix axis + P1 positive control (mix is task-ready)"),
    "PG_10b": Entity("PG_10b", "google/paligemma2-10b-pt-224",
                     "external-validity replication (§2.5); seed only when needed"),
    "L0":     Entity("L0",     "google/gemma-2-2b",
                     "BEFORE — LM init"),
    "V0":     Entity("V0",     None,
                     "BEFORE — vision (SigLIP) init; resolve at G0 from PG vision_config (HALT)"),
    # --- pi-trajectory study (Gemma-1-2B → PaliGemma-1 → π0); a SEPARATE lineage from the
    #     PaliGemma-2 program above. Gemma-1 is the generation boundary across which the PG2
    #     priors do NOT transfer numerically. SHAs are pinned per-job via HfApi().model_info
    #     directly (NEVER verify_entities — it loads PG2 + Gemma-Scope, the wrong asserts).
    "PG1":    Entity("PG1",    "google/paligemma-3b-pt-224",
                     "FT#1 — PaliGemma-1 (the VLM hop; LM tower = Gemma-1-2B fine-tuned)"),
    "L0_g1":  Entity("L0_g1",  "google/gemma-2b",
                     "FT#0 base — Gemma-1-2B init of PG1's LM (byte-identity UNPROVEN; E0a "
                     "certifies init-family only; v1.0/v1.1 revision pinned in E0a, NOT main)"),
    "PI0":    Entity("PI0",    "lerobot/pi0_base",
                     "FT#2 — π0 backbone (the VLA hop; PyTorch port; expert removed in E1.5)"),
}

# Gemma Scope JumpReLU SAEs for Gemma 2 2B residual stream (W4 LM lens).
# Loaded DIRECTLY from this HF repo, not via sae-lens — sae-lens 6.44.0 pins
# transformer-lens==2.16.1, which conflicts with transformers 5.x. JumpReLU
# params are self-contained (W_enc, W_dec, threshold, b_enc, b_dec). See the
# engineering plan for the rationale.
GEMMA_SCOPE_REPO = "google/gemma-scope-2b-pt-res"

# The three subsystems we diff. Their NESTING varies across transformers versions
# (verified 2026-05-26, transformers 5.x: top-level children are ['lm_head', 'model'];
# the three live under `model.*`). So we never assume a path — resolve_submodule_paths
# walks the tree, records the real dotted path, and HALTs only if genuinely absent.
EXPECTED_SUBMODULES = ("vision_tower", "multi_modal_projector", "language_model")


def resolve_submodule_paths(model) -> dict:
    """Map each subsystem in EXPECTED_SUBMODULES to its real dotted path in this
    model (e.g. ``vision_tower`` → ``model.vision_tower``). Introspected, not
    assumed. Returns the shallowest match per name. HALTs if any is absent —
    this is the G0 structural contract and the canonical extraction path for W1/W3/W4."""
    paths: dict = {}
    for name, _ in model.named_modules():          # top-down → first hit is shallowest
        leaf = name.rsplit(".", 1)[-1]
        if leaf in EXPECTED_SUBMODULES and leaf not in paths:
            paths[leaf] = name
    missing = [n for n in EXPECTED_SUBMODULES if n not in paths]
    if missing:
        top = sorted(dict(model.named_children()))
        raise RuntimeError(
            f"G0 HALT: subsystems {missing} not found anywhere in the module tree. "
            f"Top-level children: {top}. Confirm the real names and fix EXPECTED_SUBMODULES."
        )
    return paths


def verify_entities() -> dict:
    """G0 pre-flight (spec §0.1 step 1 / gate G0).

    Resolves the live commit SHA of every pinned repo, loads PaliGemma 2,
    introspects its real submodule names, and surfaces the vision config so a
    human can confirm the SigLIP-init match. Returns the manifest dict.

    HALTS (raises) on any structural mismatch — that IS the G0 gate.
    """
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, PaliGemmaForConditionalGeneration

    api = HfApi()
    manifest: dict = {"resolved": {}, "submodules": {}, "vision_config": {}, "gemma_scope": {}}

    # 1) Pin live commit SHAs — the §9 reproducibility deposit.
    for sym, ent in ENTITIES.items():
        if ent.repo_id is None:
            continue
        info = api.model_info(ent.repo_id, revision=ent.revision or "main")
        manifest["resolved"][sym] = {"repo_id": ent.repo_id, "sha": info.sha}
        print(f"[G0] {sym:6s} {ent.repo_id}@{info.sha}")

    # 2) Introspect PG2 submodule names — assert the spec's expectation holds.
    cfg = AutoConfig.from_pretrained(ENTITIES["PG"].repo_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    paths = resolve_submodule_paths(model)
    manifest["submodules"] = {
        "paths": paths,
        "top_level": sorted(dict(model.named_children())),
    }
    print(f"[G0] submodule paths: {paths}")

    # 3) Surface the vision init for the SigLIP-match HALT (human G0 sign-off).
    vc = cfg.vision_config
    manifest["vision_config"] = {
        "model_type": getattr(vc, "model_type", None),
        "image_size": getattr(vc, "image_size", None),
        "patch_size": getattr(vc, "patch_size", None),
        "hidden_size": getattr(vc, "hidden_size", None),
        "num_hidden_layers": getattr(vc, "num_hidden_layers", None),
    }
    print(f"[G0] vision_config: {manifest['vision_config']}")

    # 4) Gemma Scope availability (C5 revision-match is a separate human G0 check).
    gs = api.model_info(GEMMA_SCOPE_REPO)
    manifest["gemma_scope"] = {"repo_id": GEMMA_SCOPE_REPO, "sha": gs.sha}

    return manifest
