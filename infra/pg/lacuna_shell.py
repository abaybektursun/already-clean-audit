"""LACUNA shells — the ONE place policy / stitch / floor weights enter the PG1 VLM architecture.

Spec §2.1 rows:
  R1  gemma-2b question-only        (text forward; no image path at all — `textonly` builder)
  R2  STITCH-0                       SigLIP-so400m release + Frobenius-matched RANDOM projector
                                     + base gemma-2b decoder, in the PG1 VLM arch
  R3  PG1 native-in-shell            (CEILING + shell identity)
  R4  π0 re-shell                    policy vision + projector + 18L LM + own tied lm_head
  R5  π0.5 re-shell                  ditto
  R6  matched-random-ΔW              PG1 LM + Gaussian ΔW per-matrix Frobenius-matched to the
                                     **PG1→π0 LM ΔW** (the VLA hop — see AMENDMENT-1) — the FLOOR
  R7  random decoder                 PG1 LM matrices → std-matched Gaussian (scale zero)

Verified key inventory (safetensors headers, 2026-07-27, no weights downloaded):
  lerobot/pi0_base   @25c379b — 777 keys: paligemma.model.vision_tower 437 · multi_modal_projector 2
                                · language_model 163 · paligemma.lm_head [257152,2048] · expert/action
  lerobot/pi05_base  @7de6639 — 812 keys, identical paligemma-side layout
  google/paligemma-3b-pt-224 @35e4f46 — 603 keys; embed_tokens [**257216**, 2048] (padded), the
                                image placeholder token id 257152 lives INSIDE that pad region.

AMENDMENT-2 (own-embed row width): a policy's tied head is 257152 rows, PG1's table is 257216. The
own-embed shell therefore writes the policy head into rows [0:257152] and KEEPS PG1's rows
[257152:257216]. Those rows are the `<image>` placeholder (whose embedding value is overwritten by
projected image features every forward) plus pure padding — i.e. functionally inert, and truncating
instead would delete the image placeholder and break the multimodal forward outright.

No try/except (CLAUDE.md); heavy imports inside functions so this py_compiles on the control plane.
"""
from __future__ import annotations

PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
SIGLIP_REPO = "google/siglip-so400m-patch14-224"
POLICY_REPOS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}
PALI_PREFIX = "paligemma_with_expert.paligemma."

# key-count gate (spec §4.1) — the re-shell is VOID unless every compartment is present.
N_VISION_KEYS = 437
N_PROJ_KEYS = 2
N_LM_KEYS = 163            # 18 layers × 9 + final norm
PG1_VOCAB = 257216         # padded embedding rows in the PG1 checkpoint
POLICY_VOCAB = 257152      # the policies' tied head width


# ===========================================================================
# Key canonicalisation — unifies old (`language_model.model.layers`) and new
# (`model.language_model.layers`) transformers namespaces by dropping the `model`
# path COMPONENTS (never a substring: `vision_model` must survive intact).
# ===========================================================================
def canon(key: str) -> str:
    return ".".join(p for p in key.split(".") if p != "model")


def canon_map(sd):
    out = {}
    for k, v in sd.items():
        c = canon(k)
        assert c not in out, f"canonicalisation collision on {c!r} (from {k!r})"
        out[c] = v
    return out


# ===========================================================================
# Sources.
# ===========================================================================
def policy_full_sd(name):
    """The RAW policy state-dict (every key, all compartments) + the resolved repo sha."""
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    repo = POLICY_REPOS[name]
    sd = {}
    for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
        sd.update(load_file(hf_hub_download(repo, f)))
    sha = HfApi().model_info(repo, revision="main").sha
    return sd, sha


def policy_vlm_sd(name, full_sd=None):
    """{canonical_shell_key: tensor} for the policy's PaliGemma half + the own tied head.

    Returns (vlm_sd_without_head, own_head, counts). Fail-loud on a compartment leak (the action
    expert must never enter the VLM shell) and on the §4.1 key counts."""
    sd = full_sd if full_sd is not None else policy_full_sd(name)[0]
    pali = {k[len(PALI_PREFIX):]: v for k, v in sd.items() if k.startswith(PALI_PREFIX)}
    assert pali, f"{name}: no keys under {PALI_PREFIX!r}"
    leaked = [k for k in pali if "gemma_expert" in k or "action_expert" in k]
    assert not leaked, f"{name}: action-expert keys leaked into the VLM half: {leaked[:5]}"
    c = canon_map(pali)
    head_keys = [k for k in c if k.startswith("lm_head")]
    assert head_keys == ["lm_head.weight"], f"{name}: unexpected head keys {head_keys}"
    own_head = c.pop("lm_head.weight")
    counts = {"vision": sum(1 for k in c if k.startswith("vision_tower")),
              "projector": sum(1 for k in c if k.startswith("multi_modal_projector")),
              "language_model": sum(1 for k in c if k.startswith("language_model")),
              "own_head_shape": list(own_head.shape)}
    assert counts["vision"] == N_VISION_KEYS, f"{name}: {counts['vision']} vision keys ≠ {N_VISION_KEYS}"
    assert counts["projector"] == N_PROJ_KEYS, f"{name}: {counts['projector']} projector keys"
    assert counts["language_model"] == N_LM_KEYS, f"{name}: {counts['language_model']} LM keys"
    assert tuple(own_head.shape) == (POLICY_VOCAB, 2048), f"{name}: own head {tuple(own_head.shape)}"
    assert len(c) == N_VISION_KEYS + N_PROJ_KEYS + N_LM_KEYS, f"{name}: stray keys {len(c)}"
    return c, own_head, counts


def load_pg1(device="cuda", sha=None):
    """PG1 VLM, bf16, EAGER (mandatory — the sdpa prefix-LM bug). REUSED: inherit_pg1.load_pg1."""
    from infra.pg.inherit_pg1 import load_pg1 as _load
    return _load(device=device, sha=sha)


def pg1_baseline_sd(model):
    """A CPU fp-preserving clone of the loaded shell's state-dict, canonicalised."""
    return canon_map({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


# ===========================================================================
# The install path — one strict loader every row goes through.
# ===========================================================================
def _idx(key):
    """The ordered sequence of numeric path components — the layer/block address."""
    return tuple(p for p in key.split(".") if p.isdigit())


def suffix_match(pool, key):
    """Longest suffix of `key` that hits exactly one member of `pool` (no compartment split —
    only for single-compartment pools such as a vision-only state-dict). None if unresolvable.

    GUARD: the match must preserve the numeric address. Without it a suffix short enough to drop
    the layer index silently mis-maps — a pool holding only `…layers.0.self_attn.k_proj.weight`
    would answer the query `…layers.9.self_attn.k_proj.weight` with layer 0, which is exactly the
    kind of wrong-but-plausible install this whole path exists to make impossible."""
    parts = key.split(".")
    for i in range(len(parts)):
        suf = ".".join(parts[i:])
        m = [p for p in pool if (p == suf or p.endswith("." + suf)) and _idx(p) == _idx(key)]
        if len(m) == 1:
            return m[0]
        if len(m) > 1:
            return None
    return None


def resolve_keys(live_canon, update_keys, tag):
    """{update_key: live_canonical_key} — a compartment-restricted longest-suffix match.

    `canon` already unifies the `model.` nesting, but transformers versions also differ in
    INTERNAL depth: a policy checkpoint carries
    `vision_tower.vision_model.encoder.layers.0.…` while a shell built by a newer release may
    expose `vision_tower.encoder.layers.0.…`. Matching on the exact canonical string therefore
    fails on a rename that changes nothing about which tensor is meant.

    The match is restricted to the update key's COMPARTMENT (its first path component:
    `vision_tower` / `multi_modal_projector` / `language_model` / `lm_head`) and then takes the
    LONGEST suffix that resolves to exactly one live key in that compartment. The compartment
    restriction is what makes it safe: without it, the vision tower's
    `…encoder.layers.0.self_attn.k_proj.weight` would collide with the LM's
    `layers.0.self_attn.k_proj.weight`. Ambiguity and non-injectivity both RAISE.
    """
    out, ambiguous, missing = {}, [], []
    for ck in update_keys:
        if ck in live_canon:
            out[ck] = ck
            continue
        parts = ck.split(".")
        root = parts[0]
        cands = [lk for lk in live_canon if lk.split(".")[0] == root]
        hit = None
        for i in range(1, len(parts)):
            suf = ".".join(parts[i:])
            # the numeric address must survive the rename (see `suffix_match`'s guard)
            m = [lk for lk in cands
                 if (lk == suf or lk.endswith("." + suf)) and _idx(lk) == _idx(ck)]
            if len(m) == 1:
                hit = m[0]
                break
            if len(m) > 1:
                ambiguous.append((ck, m[:3]))
                break
        if hit is None:
            missing.append(ck)
        else:
            out[ck] = hit
    assert not ambiguous, f"{tag}: ambiguous key resolution, e.g. {ambiguous[:3]}"
    assert not missing, (
        f"{tag}: {len(missing)} update keys have no shell counterpart, e.g. {missing[:4]}; "
        f"live keys in that compartment: "
        f"{sorted(k for k in live_canon if k.split('.')[0] == missing[0].split('.')[0])[:6]}")
    assert len(set(out.values())) == len(out), \
        f"{tag}: key resolution is not injective — two update keys hit the same shell tensor"
    return out


def install_canon(model, updates, tag, allow_missing=()):
    """Overwrite the shell's parameters from a {canonical_key: tensor} dict, strict.

    Every update key must resolve to exactly one live shell key; shapes must match exactly.
    Returns the number of tensors written. This is the ONLY write path — so a row that silently
    failed to install is impossible."""
    import torch
    live = {canon(k): k for k in model.state_dict()}
    dev = next(model.parameters()).device
    sd = model.state_dict()
    resolved = resolve_keys(set(live), list(updates), tag)
    n, renamed = 0, 0
    with torch.no_grad():
        for ck, v in updates.items():
            lk = resolved[ck]
            renamed += (lk != ck)
            real = live[lk]
            tgt = sd[real]
            assert tuple(tgt.shape) == tuple(v.shape), \
                f"{tag}: shape mismatch on {real}: shell={tuple(tgt.shape)} new={tuple(v.shape)}"
            tgt.copy_(v.to(device=dev, dtype=tgt.dtype))
            n += 1
    touched = set(resolved.values())
    missing = sorted(k for k in live
                     if k not in touched and not any(k.startswith(a) for a in allow_missing))
    return {"n_installed": n, "n_shell_keys": len(live), "n_untouched": len(missing),
            "n_suffix_resolved": renamed, "untouched_examples": missing[:6]}


def build_row_policy(model, name, embed="own", full_sd=None):
    """R4/R5 — install the policy's vision tower + projector + 18L LM into the PG1 VLM shell.

    embed="own" → the policy's tied lm_head becomes embed_tokens rows [0:257152] (AMENDMENT-2);
    embed="pg1" → PG1's embedding table is left untouched (the interface control row R4b/R5b).
    Everything else is identical, so a difference between the two rows IS the embedding interface.
    """
    import torch
    vlm, own_head, counts = policy_vlm_sd(name, full_sd=full_sd)
    info = install_canon(model, vlm, f"{name}/{embed}", allow_missing=("language_model.embed_tokens", "lm_head"))
    emb = model.get_input_embeddings()
    assert emb.weight.shape[0] == PG1_VOCAB, f"shell embed rows {emb.weight.shape[0]} ≠ {PG1_VOCAB}"
    if embed == "own":
        with torch.no_grad():
            emb.weight[:POLICY_VOCAB].copy_(own_head.to(emb.weight.device, emb.weight.dtype))
        model.tie_weights()
        head = model.get_output_embeddings()
        assert head.weight.data_ptr() == emb.weight.data_ptr(), "lm_head not tied to embed after own-embed install"
    else:
        assert embed == "pg1", embed
    info.update(counts)
    info["embed"] = embed
    return info


def build_row_stitch0(model, seed, pg1_sd):
    """R2 — STITCH-0: SigLIP-so400m RELEASE vision tower + a Frobenius-matched RANDOM projector +
    the BASE gemma-2b decoder, inside the PG1 VLM arch, with PG1's embedding table kept as the
    fixed IO instrument.

    AMENDMENT-3 (projector draw): the random projector is Gaussian scaled to PG1's projector
    Frobenius norm rather than default `nn.Linear` init. A default-init projector emits
    near-zero image tokens, which would make the stitch a trivially dead pre-point; norm-matching
    keeps the image path at PG1's scale, which makes the pre-point as STRONG as it can honestly be
    — the conservative direction for T6 (Δ_install). Bias drawn to PG1's bias norm.

    AMENDMENT-4 (stitch IO): PG1's extended-vocab embed/lm_head is kept (the `contrast_a1`
    fixed-instrument convention) rather than resizing gemma-2b's 256000-row table. This retains
    PG1-trained embeddings in the pre-point, which again only INFLATES the pre-point and shrinks
    Δ_install — disclosed as a lower bound on the install, and it is what makes the RefCOCO
    `<loc>` output space available to the stitch row at all.
    """
    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel
    upd = {}
    # --- vision: the SigLIP release ---------------------------------------
    sig = SiglipVisionModel.from_pretrained(SIGLIP_REPO, torch_dtype=torch.bfloat16)
    sig_sd = canon_map({k: v.detach().cpu() for k, v in sig.state_dict().items()})
    want = [k for k in pg1_sd if k.startswith("vision_tower.")]
    assert len(want) == N_VISION_KEYS, f"PG1 shell has {len(want)} vision keys"
    # The release exposes `vision_model.encoder.…` while the shell's tower may be
    # `vision_tower.encoder.…` — same tensors, different nesting depth (see `resolve_keys`).
    pool = set(sig_sd)
    hits = {k: suffix_match(pool, k[len("vision_tower."):]) for k in want}
    miss = [k for k, v in hits.items() if v is None]
    assert not miss, (f"SigLIP release has no counterpart for {len(miss)} PG1 vision keys, e.g. "
                      f"{miss[:4]}; release keys e.g. {sorted(pool)[:5]}")
    assert len(set(hits.values())) == len(hits), "SigLIP key match is not injective"
    for k in want:
        assert tuple(sig_sd[hits[k]].shape) == tuple(pg1_sd[k].shape), \
            f"{k}: release {tuple(sig_sd[hits[k]].shape)} vs shell {tuple(pg1_sd[k].shape)}"
        upd[k] = sig_sd[hits[k]]
    del sig, sig_sd
    # --- projector: Frobenius-matched Gaussian ----------------------------
    g = torch.Generator().manual_seed(int(seed))
    wk, bk = "multi_modal_projector.linear.weight", "multi_modal_projector.linear.bias"
    w0, b0 = pg1_sd[wk].float(), pg1_sd[bk].float()
    zw = torch.randn(tuple(w0.shape), generator=g, dtype=torch.float32)
    upd[wk] = zw * (w0.norm() / zw.norm().clamp_min(1e-30))
    zb = torch.randn(tuple(b0.shape), generator=g, dtype=torch.float32)
    upd[bk] = zb * (b0.norm() / zb.norm().clamp_min(1e-30))
    # --- LM decoder: base gemma-2b ----------------------------------------
    g1 = AutoModelForCausalLM.from_pretrained(GEMMA1_REPO, torch_dtype=torch.bfloat16)
    g1_sd = canon_map({k: v.detach().cpu() for k, v in g1.state_dict().items()})
    del g1
    for k in [x for x in pg1_sd if x.startswith("language_model.") and "embed_tokens" not in x]:
        src = k[len("language_model."):]
        assert src in g1_sd, f"gemma-2b lacks decoder key {src!r} needed by the shell"
        upd[k] = g1_sd[src]
    info = install_canon(model, upd, f"stitch0/s{seed}",
                         allow_missing=("language_model.embed_tokens", "lm_head"))
    info.update({"seed": int(seed), "vision_source": SIGLIP_REPO, "lm_source": GEMMA1_REPO,
                 "projector": "frobenius-matched gaussian (AMENDMENT-3)",
                 "proj_frob_pg1": float(w0.norm()), "proj_frob_new": float(upd[wk].norm())})
    return info


def lm_decoder_of(canon_sd):
    """The bare 2-D+1-D LM decoder of a canonical shell state-dict (no embed/head)."""
    return {k[len("language_model."):]: v for k, v in canon_sd.items()
            if k.startswith("language_model.") and "embed_tokens" not in k}


def build_row_matched_dw(model, pg1_sd, target_lm_sd, seed):
    """R6 — the achievable-damage FLOOR: PG1's LM decoder + a per-matrix Frobenius-matched Gaussian
    ΔW. REUSED: `policy_content_readout.matched_dw_decoder`.

    AMENDMENT-1 (which ΔW is matched): the spec left "matched-random-ΔW on the PG1 LM" without
    naming the hop. It is matched to the **PG1 → π0 LM ΔW** — the VLA hop — because R6 is the floor
    the VLA-hop rows R4/R5 are scored against; matching the gemma→PG1 hop instead would calibrate
    the floor to a different intervention's magnitude and make S uninterpretable.
    """
    from infra.aws_pg.jobs.policy_content_readout import matched_dw_decoder
    base_lm = lm_decoder_of(pg1_sd)
    mdec, frob = matched_dw_decoder(base_lm, target_lm_sd, int(seed))
    upd = {f"language_model.{k}": v for k, v in mdec.items()}
    info = install_canon(model, upd, f"matched_dw/s{seed}",
                         allow_missing=("language_model.embed_tokens", "lm_head", "vision_tower",
                                        "multi_modal_projector"))
    tot = sum(v * v for v in frob.values()) ** 0.5
    info.update({"seed": int(seed), "total_frob_dW": float(tot), "n_matched": len(frob),
                 "matched_to": "PG1→π0 LM ΔW (AMENDMENT-1)"})
    return info


def build_row_random_decoder(model, pg1_sd, seed):
    """R7 — scale zero: PG1's LM 2-D matrices → per-matrix std-matched Gaussian, norms kept.
    REUSED: `vlat_retention._random_decoder_sd`. Reported, NOT the floor (N2 ordering check)."""
    from infra.aws_pg.jobs.vlat_retention import _random_decoder_sd
    rdec = _random_decoder_sd(lm_decoder_of(pg1_sd), int(seed))
    upd = {f"language_model.{k}": v for k, v in rdec.items()}
    info = install_canon(model, upd, f"random_decoder/s{seed}",
                         allow_missing=("language_model.embed_tokens", "lm_head", "vision_tower",
                                        "multi_modal_projector"))
    info["seed"] = int(seed)
    return info


def restore_pg1(model, pg1_sd):
    """Reset the shell to native PG1 (R3) — called before every row so no row inherits another's
    weights (the CLEAVE `reset()` contract)."""
    return install_canon(model, pg1_sd, "restore/pg1")


# ===========================================================================
# The paired {real, gray} forward (spec §2.1, control G).
# ===========================================================================
def gray_task(task, gray=128):
    """A copy of a BTask whose image_fn returns the constant gray-128 image. The prompt, scorer,
    decode length and item order are untouched, so {real, gray} is paired per item by
    construction. `gray=128` is the house blank-image convention (`_ppl_blankimg`/`mmlu_blankimg`)."""
    import dataclasses
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    img = _blank_image(gray)
    return dataclasses.replace(task, image_fn=lambda row: img)


def mm_tasks():
    """The four multimodal battery tasks, REUSED verbatim: `projector_ablation.build_tasks()`
    = {refcoco, textvqa, vqav2, coco_captions} with the July battery's prompts and scorers."""
    from infra.pg.projector_ablation import build_tasks
    return {t.name: t for t in build_tasks()}


# ===========================================================================
# R1 — question-only text forward (no image path at all).
# ===========================================================================
def textonly_scores(task, rows, device, max_new_tokens=None, batch_size=16):
    """R1: greedy-decode `task.prompt_fn(row)` through BASE gemma-2b as a pure text model and score
    with the task's OWN scorer. Bounds the pure language prior — the thing the gray-128 forward
    cannot bound, because a gray image still passes through the projector (spec §8).

    RefCOCO is scored 0 by construction here (gemma-2b has no `<loc>` tokens); that is reported,
    not hidden, and it is why T6 takes max(R2, R1) rather than R1 alone.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA1_REPO)
    m = AutoModelForCausalLM.from_pretrained(GEMMA1_REPO, torch_dtype=torch.bfloat16,
                                             device_map=device, attn_implementation="eager").eval()
    mnt = max_new_tokens or task.max_new_tokens
    per_item, cand, ref = [], {}, {}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [task.prompt_fn(r) for r in batch]
        inp = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            gen = m.generate(**inp, max_new_tokens=mnt, do_sample=False)
        dec = tok.batch_decode(gen[:, inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        for r, pred in zip(batch, dec):
            pred = pred.strip()
            if task.is_corpus:
                i = len(cand)
                cand[i], ref[i] = [pred], task.refs_fn(r)
            else:
                per_item.append(float(task.score_fn(pred, r)))
    del m
    torch.cuda.empty_cache()
    if task.is_corpus:
        return {"corpus": True, "cand": cand, "ref": ref}
    return {"corpus": False, "per_item": per_item}
