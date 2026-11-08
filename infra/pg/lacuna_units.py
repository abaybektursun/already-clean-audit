"""LACUNA Arm 3 — the U1–U4 unit constructors (spec §2.3; ledger entry "U3/U4 constructors, NEW").

  U1  lerobot/pi0_base                                     (as published)
  U2  lerobot/pi05_base                                    (as published)
  U3  PG1 backbone + FRESH action expert, π0 architecture   ← state-dict surgery
  U4  random-init VLM + FRESH action expert, π0 architecture ← the floor

U3/U4 are written as ordinary lerobot checkpoint directories (config.json + model.safetensors +
the pre/post-processor jsons copied from π0_base), so `lerobot-train --policy.path=<dir>` and
`lerobot-eval --policy.path=<dir>` consume them with no special-casing anywhere downstream.

Construction discipline:
  * the FRESH expert is whatever `PI0Policy(config)` initialises — lerobot's own initialiser, not a
    hand-rolled Gaussian, so "fresh" means what the library means by it;
  * the backbone is then installed EXPLICITLY (PG1 for U3, an HF random-init PaliGemma for U4)
    rather than relying on whatever `PI0Policy(config)` happens to do for the backbone — the unit's
    identity must not depend on a library default that can change;
  * a key-count + compartment gate runs on every write (the expert must be untouched by the
    backbone install, and vice versa).

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

PI0_REPO = "lerobot/pi0_base"
PG1_REPO = "google/paligemma-3b-pt-224"
# The SAFETENSORS keys are `paligemma_with_expert.…`, but a LIVE `PI0Policy`'s state_dict nests
# them under `model.` (cf. `ctrliface_lerobot.BACKBONE_PATH`). Match on the substring and read the
# prefix off the live object instead of assuming either spelling — F0 caught this the hard way.
PALI_INFIX = "paligemma_with_expert.paligemma."
EXPERT_INFIX = "paligemma_with_expert.gemma_expert."
POLICY_VOCAB = 257152
COPY_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")

# LIBERO's own feature layout, copied from `lerobot/pi05_libero_finetuned` (the checkpoint the N5a
# harness already runs). π0's published config carries the openpi 3-camera layout
# (base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb, state 32), but `HuggingFaceVLA/libero`
# supplies `observation.images.{image,image2}` with an 8-dim state and 7-dim action — so a forward
# on a LIBERO batch raises "All image features are missing from the batch". EVERY unit is written
# with this layout so U1–U4 differ ONLY in backbone weights, which is what the §2.3 estimand needs.
LIBERO_INPUT_FEATURES = {
    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.state": {"type": "STATE", "shape": [8]},
}
LIBERO_OUTPUT_FEATURES = {"action": {"type": "ACTION", "shape": [7]}}


def _canon(k):
    return ".".join(p for p in k.split(".") if p != "model")


def compartments(sd):
    """(paligemma_keys, expert_keys) of a live-or-checkpoint π0 state-dict, spelling-agnostic."""
    pal = [k for k in sd if PALI_INFIX in k]
    exp = [k for k in sd if EXPERT_INFIX in k]
    hint = {"n_keys": len(sd), "expert_like": [k for k in sd if "expert" in k][:6],
            "paligemma_like": [k for k in sd if "paligemma" in k][:3],
            "top_level": sorted({k.split(".")[0] for k in sd})[:12]}
    assert pal, f"no paligemma compartment; {hint}"
    assert exp, f"no gemma_expert compartment; {hint}"
    return pal, exp


def _fill_from(src, policy_sd, emb, tag):
    """{policy_key: tensor} filling the π0 `paligemma` compartment from a source VLM state-dict.

    Uses the SAME compartment-restricted longest-suffix resolver as the ladder shell
    (`lacuna_shell.resolve_keys`): a policy checkpoint carries
    `vision_tower.vision_model.encoder.…` while a shell built by a newer transformers release may
    expose `vision_tower.encoder.…`, and an exact-string lookup fails on that rename even though
    the tensors correspond one-to-one."""
    from infra.pg.lacuna_shell import resolve_keys
    slots = compartments(policy_sd)[0]
    want = {k: _canon(k.split(PALI_INFIX, 1)[1]) for k in slots}
    # VOCAB-WIDTH SLOTS. A live `PI0Policy` carries BOTH `lm_head.weight` and an
    # `…language_model.embed_tokens.weight`, each 257152 rows, while the π0 SAFETENSORS drop
    # embed_tokens (the openpi tied-embedding convention) and PG1's table is 257216. Both slots are
    # therefore filled from PG1's embedding TRUNCATED to the policy's own width — the 64 dropped
    # rows are PG1's `<image>` placeholder + padding, which the π0 architecture does not use (it
    # splices image features rather than routing them through a placeholder token id).
    def _is_vocab(c):
        return c == "lm_head.weight" or c.endswith("embed_tokens.weight")

    non_head = sorted({c for c in want.values() if not _is_vocab(c)})
    res = resolve_keys(set(src), non_head, tag)
    upd = {}
    for k, c in want.items():
        tgt = tuple(policy_sd[k].shape)
        if _is_vocab(c):
            rows = tgt[0]
            assert rows <= emb.shape[0], f"{k}: policy wants {rows} rows, source has {emb.shape[0]}"
            upd[k] = emb[:rows].clone()
        else:
            upd[k] = src[res[c]].clone()
        assert tuple(upd[k].shape) == tgt, f"{k}: source {tuple(upd[k].shape)} vs policy {tgt}"
    assert len(upd) >= 600, f"{tag}: only {len(upd)} backbone slots filled"
    return upd


def pg1_backbone_updates(policy_sd):
    """{policy_key: tensor} installing PG1's VLM into the π0 `paligemma` compartment.

    PG1 ties its head to `embed_tokens` and stores 257216 padded rows, while the policy carries a
    separate 257152-row `lm_head`; the head is therefore PG1's embedding truncated to the policy's
    width (the same 64 pad/`<image>` rows discussed in AMENDMENT-2, here on the OUTPUT side, where
    they are unreachable logits)."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    src = {_canon(k): v.detach().cpu() for k, v in pg.state_dict().items()}
    del pg
    emb = src["language_model.embed_tokens.weight"]
    return _fill_from(src, policy_sd, emb, "pg1_backbone")


def random_backbone_updates(policy_sd, seed):
    """{policy_key: tensor} installing an HF RANDOM-INIT PaliGemma of the identical architecture —
    the U4 floor. Uses `PaliGemmaForConditionalGeneration(config)`, i.e. the model class's own
    initialiser at the real config, not a std-matched shuffle of trained weights."""
    import torch
    from transformers import AutoConfig, PaliGemmaForConditionalGeneration
    torch.manual_seed(int(seed))
    cfg = AutoConfig.from_pretrained(PG1_REPO)
    rnd = PaliGemmaForConditionalGeneration(cfg)
    src = {_canon(k): v.detach().cpu().float() for k, v in rnd.state_dict().items()}
    del rnd
    emb = src["language_model.embed_tokens.weight"]
    return _fill_from(src, policy_sd, emb, "random_backbone")


def build_published(repo, outdir):
    """Write a published policy (π0 or π0.5) to a local checkpoint dir with the LIBERO feature
    layout. Weights are untouched — only the interface is normalised, so U1/U2 differ from U3/U4
    in backbone identity alone."""
    import json
    import pathlib
    import shutil
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in list_repo_files(repo):
        if f.endswith((".safetensors", ".json")) and "/" not in f:
            shutil.copy(hf_hub_download(repo, f), outdir / f)
    cfgp = outdir / "config.json"
    raw = json.loads(cfgp.read_text())
    raw["input_features"] = LIBERO_INPUT_FEATURES
    raw["output_features"] = LIBERO_OUTPUT_FEATURES
    cfgp.write_text(json.dumps(raw, indent=2))
    man = {"kind": f"published:{repo}", "outdir": str(outdir),
           "features": "LIBERO layout (image/image2, state 8, action 7)",
           "sha": HfApi().model_info(repo, revision="main").sha,
           "files": sorted(p.name for p in outdir.iterdir())}
    (outdir / "lacuna_unit_manifest.json").write_text(json.dumps(man, indent=2))
    return man


def build_unit(kind, outdir, seed=0, device="cpu"):
    """Write a lerobot checkpoint directory for U3 (`kind='pg1_backbone'`) or U4 (`kind='random'`).

    Returns the manifest (key counts, compartment gate results, the sha of every source)."""
    import json
    import pathlib
    import shutil

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PI0Config.from_pretrained(PI0_REPO)
    policy = PI0Policy(cfg)                       # lerobot's OWN init — this is the fresh expert
    sd = {k: v.detach().cpu().float() for k, v in policy.state_dict().items()}
    n_expert_before = {k: sd[k].clone() for k in compartments(sd)[1]}

    upd = (pg1_backbone_updates(sd) if kind == "pg1_backbone"
           else random_backbone_updates(sd, seed))
    sd.update(upd)
    # compartment gate: the backbone install must not have touched the expert or the action heads
    for k, v in n_expert_before.items():
        assert torch.equal(sd[k], v), f"backbone install leaked into the expert at {k}"
    missing, unexpected = policy.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"unit load not clean: {missing} {unexpected}"
    policy.save_pretrained(outdir)
    for f in COPY_FILES:
        if not (outdir / f).exists():
            shutil.copy(hf_hub_download(PI0_REPO, f), outdir / f)
    cfgp = outdir / "config.json"
    raw = json.loads(cfgp.read_text())
    raw["input_features"] = LIBERO_INPUT_FEATURES
    raw["output_features"] = LIBERO_OUTPUT_FEATURES
    cfgp.write_text(json.dumps(raw, indent=2))
    man = {"kind": kind, "seed": int(seed), "outdir": str(outdir),
           "features": "LIBERO layout (image/image2, state 8, action 7)",
           "n_backbone_installed": len(upd),
           "n_expert_keys": len(n_expert_before),
           "n_total_keys": len(sd),
           "pi0_sha": HfApi().model_info(PI0_REPO, revision="main").sha,
           "pg1_sha": (HfApi().model_info(PG1_REPO, revision="main").sha
                       if kind == "pg1_backbone" else None),
           "files": sorted(p.name for p in outdir.iterdir())}
    (outdir / "lacuna_unit_manifest.json").write_text(json.dumps(man, indent=2))
    return man
