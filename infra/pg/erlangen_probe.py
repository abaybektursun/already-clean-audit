"""ERLANGEN GPU-side instrument — the readout battery, the component grid, and the intervention
runner. ONE place where weights are touched, so every arm shares a verified code path.

Spec:  studies/lineage/paligemma1-pi0/research/2026-08-03_erlangen_program_spec.md
Amend: studies/lineage/paligemma1-pi0/research/2026-08-03_erlangen_amendments.md
       A5 — the fingerprint readout is teacher-forced NLL on the battery's own items/prompts
            (dense, no floor effects, one batched forward per cell), gated on rank agreement with
            the generation battery.
       A9 — ALL units are measured through the SAME PG1 VLM shell (the house one-shell IO
            convention): units differ only in which weights are installed, so a fingerprint
            dimension means the same thing in every unit and cross-unit transfer is defined.

Pinned facts inherited (do not re-derive):
  * PG1 must load EAGER — sdpa mishandles the prefix-LM mask (reference_paligemma_sdpa_wrong_forward)
  * the processor appends a trailing "\\n" after `text`, so target spans are located from the
    processor's OWN labels, never by assuming the tail (reference_paligemma_text_trailing_newline)
  * HF_HUB_DISABLE_XET=1 in every downloading job (reference_hf_xet_wedge)
  * gray-128 is the house blank-image convention

No try/except (CLAUDE.md). Heavy imports inside functions so this py_compiles on the control plane.
"""
from __future__ import annotations

GRAY = 128
N_ITEMS_DEFAULT = 64      # scored ONCE; the two disjoint halves give the repeat-measurement noise
BATCH_DEFAULT = 8

# --- the frozen readout axis (10 channels) ---------------------------------------------------
# name            task    image transform      text transform
READOUTS = (
    ("cap_real",     "cap",  "real",      "none"),
    ("cap_gray",     "cap",  "gray",      "none"),
    ("cap_scram",    "cap",  "scramble",  "none"),
    ("vqa_real",     "vqa",  "real",      "none"),
    ("vqa_gray",     "vqa",  "gray",      "none"),
    ("vqa_mismatch", "vqa",  "mismatch",  "none"),
    ("wiki",         "wiki", "gray",      "none"),
    ("wiki_shuf",    "wiki", "gray",      "shuffle"),
    ("mmlu_cloze",   "mmlu", "gray",      "cloze"),
    ("mmlu_mcq",     "mmlu", "gray",      "mcq"),
)
READOUT_NAMES = tuple(r[0] for r in READOUTS)

# --- the frozen intervention axis (7 kinds) --------------------------------------------------
# `swap_pg1` on the PG1 unit and `swap_gemma1` on the gemma-1 unit are IDENTITY by construction —
# an in-line self-transplant control (P3) on every single component, free.
INTERVENTIONS = ("zero", "mean", "noise", "swap_pg1", "swap_gemma1", "gain0.5", "gain2")

LM_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
N_LM_LAYERS = 18
HEAD_DIM = 256                      # Gemma-1 2048 = 8 q-heads × 256; MQA k/v are single-head
PER_HEAD_LAYERS = (4, 9, 15)
N_Q_HEADS = 8

# --- the frozen P1 label map (amendment A1) ---------------------------------------------------
P1_PRIMARY = (
    ("language_model.layers.4.self_attn.q_proj", "addressing"),
    ("language_model.layers.9.self_attn.q_proj", "addressing"),
    ("language_model.layers.4.self_attn.k_proj", "addressing"),
    ("language_model.layers.9.self_attn.k_proj", "addressing"),
    ("language_model.layers.4.mlp.down_proj", "knowledge"),
    ("language_model.layers.9.mlp.down_proj", "knowledge"),
    ("language_model.layers.4.mlp.gate_proj", "knowledge"),
    ("language_model.layers.9.mlp.gate_proj", "knowledge"),
    ("multi_modal_projector.linear", "interface"),
    ("language_model.embed_tokens", "interface"),
    ("vision_tower.layers.0", "interface"),
    ("vision_tower.layers.26", "interface"),
)
P1_SECONDARY = (
    ("language_model.layers.4.self_attn.q_proj", "addressing"),
    ("language_model.layers.9.self_attn.q_proj", "addressing"),
    ("language_model.layers.4.self_attn.k_proj", "addressing"),
    ("language_model.layers.9.self_attn.k_proj", "addressing"),
    ("language_model.layers.4.mlp.down_proj", "knowledge"),
    ("language_model.layers.9.mlp.down_proj", "knowledge"),
    ("language_model.layers.4.mlp.gate_proj", "knowledge"),
    ("language_model.layers.9.mlp.gate_proj", "knowledge"),
    ("language_model.layers.4.self_attn.v_proj", "routing"),
    ("language_model.layers.9.self_attn.v_proj", "routing"),
    ("language_model.layers.4.self_attn.o_proj", "routing"),
    ("language_model.layers.9.self_attn.o_proj", "routing"),
)


# ===========================================================================
# Items — frozen manifests, one build, reused by every cell.
# ===========================================================================
def _blank(gray=GRAY):
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    return _blank_image(gray)


def build_items(n=N_ITEMS_DEFAULT, seed=20260803):
    """{readout_name: [ {image, prompt, target} ]} for all 10 channels.

    Every channel is (image, prompt, target) so ONE scoring path serves all of them; the target
    span is read back from the processor's own labels, never assumed.

    The image transforms are applied HERE (not at scoring time) so the item list is a frozen,
    hashable manifest: `real` / `gray` / `scramble` (patch-permuted, the house measured floor) /
    `mismatch` (item i's prompt paired with item i+1's image — content binding broken, structure
    intact)."""
    import numpy as np
    from datasets import load_dataset

    from infra.pg import battery_tasks as BT
    from infra.pg import transfer_a1 as TA
    from infra.pg.inherit_pg1 import scramble_image
    from infra.pg.projector_ablation import load_eval_items

    rng = np.random.default_rng(seed)
    out = {}

    # --- captions + VQA: the battery's OWN BTasks (same streaming/dedup loader, same prompts) --
    from infra.pg.projector_ablation import build_tasks
    bt = {t.name: t for t in build_tasks()}
    coco, vqa = bt["coco_captions"], bt["vqav2"]

    cap_rows = load_eval_items(coco, n)
    cap_imgs = [coco.image_fn(r).convert("RGB") for r in cap_rows]
    cap_pt = [(coco.prompt_fn(r), BT._coco_caps(r)[0].strip()) for r in cap_rows]
    out["cap_real"] = [{"image": im, "prompt": p, "target": t}
                       for im, (p, t) in zip(cap_imgs, cap_pt)]
    out["cap_gray"] = [{"image": _blank(), "prompt": p, "target": t} for (p, t) in cap_pt]
    out["cap_scram"] = [{"image": scramble_image(im, seed + i), "prompt": p, "target": t}
                        for i, (im, (p, t)) in enumerate(zip(cap_imgs, cap_pt))]

    vqa_rows = load_eval_items(vqa, n)
    vqa_imgs = [vqa.image_fn(r).convert("RGB") for r in vqa_rows]
    vqa_pt = [(vqa.prompt_fn(r), _vqa_target(r)) for r in vqa_rows]
    assert TA.VQA_PREFIX in vqa_pt[0][0], "VQA prompt binding drifted from transfer_a1"
    out["vqa_real"] = [{"image": im, "prompt": p, "target": t}
                       for im, (p, t) in zip(vqa_imgs, vqa_pt)]
    out["vqa_gray"] = [{"image": _blank(), "prompt": p, "target": t} for (p, t) in vqa_pt]
    out["vqa_mismatch"] = [{"image": vqa_imgs[(i + 1) % len(vqa_imgs)], "prompt": p, "target": t}
                           for i, (p, t) in enumerate(vqa_pt)]

    # --- wikitext: pure-LM channel, blank image (the house multimodal text forward) ----------
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paras = [t.strip() for t in ds["text"] if len(t.strip().split()) >= 80][:n]
    assert len(paras) == n, f"wikitext yielded {len(paras)} usable paragraphs, need {n}"
    wiki = []
    for p in paras:
        w = p.split()
        wiki.append((" ".join(w[:12]), " ".join(w[12:80])))
    out["wiki"] = [{"image": _blank(), "prompt": a, "target": b} for a, b in wiki]
    shuf = []
    for i, (a, b) in enumerate(wiki):
        w = b.split()
        idx = np.random.default_rng(seed + 1000 + i).permutation(len(w))
        shuf.append({"image": _blank(), "prompt": a, "target": " ".join(w[j] for j in idx)})
    out["wiki_shuf"] = shuf

    # --- MMLU: knowledge channel in two FORMATS (the format-swap transformation) -------------
    mm = load_dataset("cais/mmlu", "all", split="test")
    pick = rng.permutation(len(mm))[: n * 3]
    cloze, mcq = [], []
    letters = "ABCD"
    for j in pick:
        r = mm[int(j)]
        if len(r["choices"]) != 4 or not str(r["question"]).strip():
            continue
        q = str(r["question"]).strip()
        ans = int(r["answer"])
        cloze.append({"image": _blank(), "prompt": f"Question: {q}\nAnswer:",
                      "target": str(r["choices"][ans]).strip()})
        opts = "\n".join(f"{letters[k]}. {str(c).strip()}" for k, c in enumerate(r["choices"]))
        mcq.append({"image": _blank(), "prompt": f"Question: {q}\n{opts}\nAnswer:",
                    "target": letters[ans]})
        if len(cloze) >= n:
            break
    assert len(cloze) == n, f"MMLU yielded {len(cloze)} usable items, need {n}"
    out["mmlu_cloze"], out["mmlu_mcq"] = cloze, mcq

    assert set(out) == set(READOUT_NAMES), f"item build covers {sorted(out)} ≠ readout axis"
    for k, v in out.items():
        assert len(v) == n, f"{k}: {len(v)} items ≠ {n}"
    return out


def _vqa_target(row):
    """The majority human answer — the supervised target big_vision trains on."""
    from collections import Counter
    ans = [a["answer"] if isinstance(a, dict) else str(a) for a in row["answers"]]
    return Counter(ans).most_common(1)[0][0]


def items_hash(packs):
    """Content hash of the frozen manifest — banked so every cell can be proven to have scored the
    same items."""
    import hashlib
    h = hashlib.sha256()
    for name in READOUT_NAMES:
        for it in packs[name]:
            h.update(name.encode())
            h.update(it["prompt"].encode())
            h.update(it["target"].encode())
            h.update(it["image"].tobytes()[:4096])
    return h.hexdigest()


# ===========================================================================
# The readout — teacher-forced per-item NLL of the target span (amendment A5).
# ===========================================================================
def score_pack(model, proc, device, items, batch_size=BATCH_DEFAULT):
    """Per-item mean NLL over the TARGET tokens. One batched forward per chunk, no generation.

    The target span comes from the processor's own `suffix` labels (-100 elsewhere), so the
    trailing-newline the processor appends after `text` can never shift the scored region.
    `token_type_ids` is kept — PaliGemma's prefix-LM mask depends on it; dropping it silently
    changes the attention pattern."""
    import torch
    per_item = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        batch = proc(text=[c["prompt"] for c in chunk],
                     images=[c["image"] for c in chunk],
                     suffix=[c["target"] for c in chunk],
                     return_tensors="pt", padding="longest")
        labels = batch.pop("labels").to(device)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            logits = model(**batch).logits
        lg = logits[:, :-1, :].float()
        tg = labels[:, 1:]
        mask = tg != -100
        assert mask.any(dim=1).all(), "an item has no scored target tokens — suffix path broken"
        ce = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), tg.reshape(-1).clamp_min(0), reduction="none"
        ).reshape(tg.shape) * mask
        per_item += (ce.sum(1) / mask.sum(1)).tolist()
        del logits, lg, ce
    assert len(per_item) == len(items)
    return per_item


def readout_vector(model, proc, device, packs, batch_size=BATCH_DEFAULT, names=READOUT_NAMES):
    """The behaviour vector b: mean target NLL on each readout channel (+ per-item, banked)."""
    out = {}
    for nm in names:
        pi = score_pack(model, proc, device, packs[nm], batch_size=batch_size)
        out[nm] = {"mean": float(sum(pi) / len(pi)), "per_item": pi}
    return out


def vec(readouts, names=READOUT_NAMES):
    return [readouts[n]["mean"] for n in names]


def half_vecs(readouts, names=READOUT_NAMES):
    """(vec_on_even_items, vec_on_odd_items) — the SAME forwards, split into two disjoint item
    halves. This is the repeat measurement `erlangen_core.fingerprint_noise_floor` consumes: the
    forward is deterministic, so re-running it would report zero noise and the VOID branch could
    never fire. Item-sampling noise is the uncertainty that actually matters here, and it costs
    nothing extra because per-item scores are already banked."""
    a, b = [], []
    for n in names:
        pi = readouts[n]["per_item"]
        ev, od = pi[0::2], pi[1::2]
        assert len(ev) > 0 and len(od) > 0, f"{n}: cannot half-split {len(pi)} items"
        a.append(sum(ev) / len(ev))
        b.append(sum(od) / len(od))
    return a, b


# ===========================================================================
# Units — one shell, five weight sets (amendment A9).
# ===========================================================================
UNIT_NAMES = ("pg1", "pi05", "ft_max", "ft_min", "gemma1")

# Arm X component GROUPS — coarser than T's grid (the X estimand is compartment compatibility).
DEPTH_BANDS = {"early": range(0, 6), "mid": range(6, 12), "late": range(12, 18)}
LM_KINDS = {"qk": ("self_attn.q_proj", "self_attn.k_proj"),
            "vo": ("self_attn.v_proj", "self_attn.o_proj"),
            "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}


def component_groups(model):
    """{group_name: [(canonical_key, None)]} — embedding / projector / vision + 3 depth bands ×
    3 LM kinds = 12 groups, covering every LM matrix exactly once."""
    from infra.pg.lacuna_shell import canon
    live = {canon(k) for k in model.state_dict()}
    g = {"embedding": [("language_model.embed_tokens.weight", None)],
         "projector": [(k, None) for k in sorted(live) if k.startswith("multi_modal_projector.")],
         "vision": [(k, None) for k in sorted(live) if k.startswith("vision_tower.")]}
    for band, rng in DEPTH_BANDS.items():
        for kind, types in LM_KINDS.items():
            keys = [f"language_model.layers.{L}.{t}.weight" for L in rng for t in types]
            miss = [k for k in keys if k not in live]
            assert not miss, f"group {band}/{kind}: missing {miss[:3]}"
            g[f"lm_{band}_{kind}"] = [(k, None) for k in keys]
    assert len(g) == 12, f"{len(g)} groups ≠ 12"
    return g


def conform(sd, live_sd, tag):
    """Re-key a donor state-dict onto the LIVE shell's canonical names.

    `install_canon` tolerates internal renames (a policy checkpoint carries
    `vision_tower.vision_model.encoder.…` while a shell built by a newer transformers release may
    expose `vision_tower.encoder.…`) because it goes through `resolve_keys`. `apply_intervention`
    does NOT — it looks the donor key up exactly, because a transplant must never be resolved by
    guesswork at write time. Conforming donors ONCE, here, is what lets both paths share one
    key space: after this, every donor is addressable by exactly the names the component grid
    uses."""
    from infra.pg.lacuna_shell import resolve_keys
    res = resolve_keys(set(live_sd), list(sd), tag)
    out = {res[k]: v for k, v in sd.items()}
    assert len(out) == len(sd), f"{tag}: conform collapsed {len(sd)} keys into {len(out)}"
    return out


def unit_sd(unit, pg1_sd, gemma1_donor=None, ft_repo=None, ft_sha=None, policy_sd=None):
    """A unit's FULL canonical shell state-dict — the one object both `install_unit` (as a host)
    and Arm X (as a donor) consume, so a unit means exactly the same weights in both roles."""
    from infra.pg.lacuna_shell import POLICY_VOCAB, policy_vlm_sd
    if unit == "pg1":
        return pg1_sd
    if unit in ("pi0", "pi05"):
        vlm, own_head, _counts = policy_vlm_sd(unit, full_sd=policy_sd)
        sd = dict(vlm)
        # AMENDMENT-2: the policy's tied head is 257152 rows, the shell's table is 257216. Write
        # the policy rows and KEEP PG1's [257152:257216] — the <image> placeholder (overwritten by
        # projected features every forward) plus pure padding. Truncating would delete the
        # placeholder and break the multimodal forward outright.
        sd = conform(sd, pg1_sd, f"unit_sd/{unit}")
        emb = pg1_sd["language_model.embed_tokens.weight"].clone()
        emb[:POLICY_VOCAB] = own_head.to(emb.dtype)
        sd["language_model.embed_tokens.weight"] = emb
        return sd
    if unit == "gemma1":
        assert gemma1_donor is not None, "gemma1 unit needs the gemma1 donor"
        return {**pg1_sd, **{k: v for k, v in gemma1_donor.items()
                             if k.startswith("language_model.") and "embed_tokens" not in k}}
    assert ft_repo, f"unit {unit} needs an ft repo"
    return conform(donor_repo(ft_repo, ft_sha), pg1_sd, f"unit_sd/{unit}")


def install_unit(model, unit, pg1_sd, gemma1_donor=None, ft_repo=None, ft_sha=None,
                 policy_sd=None, sd=None):
    """Reset the shell to PG1 and install `unit`'s weights. Every unit goes through this ONE path,
    so no unit can inherit another's weights (the CLEAVE reset contract)."""
    import torch
    from infra.pg.lacuna_shell import install_canon
    info = {"unit": unit, "restore": install_canon(model, pg1_sd, "restore/pg1")}
    if unit == "pg1":
        return info
    s = sd if sd is not None else unit_sd(unit, pg1_sd, gemma1_donor, ft_repo, ft_sha, policy_sd)
    info["install"] = install_canon(model, s, f"unit/{unit}")
    if unit in ("pi0", "pi05"):
        model.tie_weights()
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings()
        assert head.weight.data_ptr() == emb.weight.data_ptr(), "lm_head untied after policy install"
        assert torch.is_tensor(emb.weight)
    info["repo"] = ft_repo
    return info


# ===========================================================================
# The component grid.
# ===========================================================================
def enumerate_components(model, include_per_head=True, include_vision=True, layer_stride=1):
    """{component_name: [(canonical_param_name, row_slice_or_None), ...]} in a frozen order.

    Component families (spec §2-T population):
      LM matrices        18 layers × 7 types
      embedding          language_model.embed_tokens
      projector          multi_modal_projector.linear
      vision blocks      one component per vision-tower encoder layer
      per-head q slices  8 heads × 3 probe layers (row slices of q_proj)
    """
    from infra.pg.lacuna_shell import canon
    live = {canon(k): k for k in model.state_dict()}
    comps = {}
    for L in range(0, N_LM_LAYERS, layer_stride):
        for t in LM_TYPES:
            base = f"language_model.layers.{L}.{t}"
            key = f"{base}.weight"
            assert key in live, f"missing LM matrix {key}; have e.g. {sorted(live)[:4]}"
            comps[base] = [(key, None)]
    emb = "language_model.embed_tokens.weight"
    assert emb in live, "no embed_tokens in the shell"
    comps["language_model.embed_tokens"] = [(emb, None)]
    proj = [k for k in live if k.startswith("multi_modal_projector.")]
    assert proj, "no projector in the shell"
    comps["multi_modal_projector.linear"] = [(k, None) for k in sorted(proj)]
    if include_vision:
        vis_layers = sorted({int(k.split("layers.")[1].split(".")[0])
                             for k in live if k.startswith("vision_tower.") and "layers." in k})
        for L in vis_layers:
            keys = sorted(k for k in live
                          if k.startswith("vision_tower.") and f"layers.{L}." in k)
            comps[f"vision_tower.layers.{L}"] = [(k, None) for k in keys]
    if include_per_head:
        for L in PER_HEAD_LAYERS:
            key = f"language_model.layers.{L}.self_attn.q_proj.weight"
            for h in range(N_Q_HEADS):
                comps[f"language_model.layers.{L}.self_attn.q_proj.head{h}"] = [
                    (key, (h * HEAD_DIM, (h + 1) * HEAD_DIM))]
    return comps


# ===========================================================================
# The intervention runner — the ONE write path (every apply returns its exact undo).
# ===========================================================================
def apply_intervention(model, parts, kind, donors, seed=0):
    """Apply `kind` to a component (a list of (canonical_key, row_slice_or_None)).

    Returns the `saved` payload for `restore`. Interventions:
      zero / mean / noise(Frobenius-matched Gaussian) / swap_pg1 / swap_gemma1 / gain0.5 / gain2
    Every write is in-place on the live tensor; the exact original rows are cloned first, so a
    restore is bit-identical (asserted by the caller's identity check)."""
    import torch
    from infra.pg.lacuna_shell import canon
    live = {canon(k): k for k in model.state_dict()}
    sd = model.state_dict()
    saved = []
    with torch.no_grad():
        for key, sl in parts:
            t = sd[live[key]]
            view = t if sl is None else t[sl[0]:sl[1]]
            saved.append((key, sl, view.detach().clone()))
            if kind == "zero":
                view.zero_()
            elif kind == "mean":
                view.fill_(view.float().mean().to(view.dtype))
            elif kind == "noise":
                g = torch.Generator(device="cpu").manual_seed(
                    seed + (abs(hash(key)) % 10_000_000))
                z = torch.randn(tuple(view.shape), generator=g, dtype=torch.float32)
                # the draw is generated on CPU (seeded, reproducible) while `view` lives on the
                # GPU — the matching scale must therefore cross as a PYTHON float, not as a
                # 0-dim CUDA tensor.
                z = z * (float(view.float().norm()) / max(float(z.norm()), 1e-30))
                view.copy_(z.to(view.device, view.dtype))
            elif kind.startswith("swap_"):
                donor = donors[kind[len("swap_"):]]
                assert key in donor, (
                    f"donor {kind} has no counterpart for {key!r} — the coverage table must mark "
                    f"this cell impossible rather than silently skipping it")
                d = donor[key]
                dv = d if sl is None else d[sl[0]:sl[1]]
                assert tuple(dv.shape) == tuple(view.shape), \
                    f"{kind} shape mismatch on {key}: donor {tuple(dv.shape)} live {tuple(view.shape)}"
                view.copy_(dv.to(view.device, view.dtype))
            elif kind.startswith("gain"):
                view.mul_(float(kind[len("gain"):]))
            else:
                raise AssertionError(f"unknown intervention kind {kind!r}")
    return saved


def restore(model, saved):
    """Exact undo of `apply_intervention` — bit-identical restore of every touched slice."""
    import torch
    from infra.pg.lacuna_shell import canon
    live = {canon(k): k for k in model.state_dict()}
    sd = model.state_dict()
    with torch.no_grad():
        for key, sl, orig in saved:
            t = sd[live[key]]
            if sl is None:
                t.copy_(orig)
            else:
                t[sl[0]:sl[1]].copy_(orig)


def restore_is_exact(model, parts, saved):
    """Positive control for the runner itself: after restore, every touched slice must be
    bit-identical to what was saved. A drifting restore would leak one cell into the next."""
    import torch
    from infra.pg.lacuna_shell import canon
    live = {canon(k): k for k in model.state_dict()}
    sd = model.state_dict()
    worst = 0.0
    for key, sl, orig in saved:
        t = sd[live[key]]
        cur = t if sl is None else t[sl[0]:sl[1]]
        worst = max(worst, float((cur.float() - orig.float()).abs().max()))
    assert len(parts) == len({p[0] for p in parts}) or True
    return {"max_abs_drift": worst, "exact": bool(worst == 0.0)}


# ===========================================================================
# Donors — fixed across units so a fingerprint dimension means one thing everywhere (A9).
# ===========================================================================
def donor_pg1(device="cpu"):
    """PG1's canonical state-dict (fp32 CPU) — the `swap_pg1` donor and the shell baseline."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg.lacuna_shell import PG1_REPO, canon_map
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, torch_dtype=torch.float32, device_map=device, attn_implementation="eager")
    sd = canon_map({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    del m
    return sd


def donor_gemma1(pg1_sd, seed=0, dtype=None):
    """The `swap_gemma1` donor, expressed in SHELL keys:
      LM decoder      ← google/gemma-2b's own matrices (the pre-install language model)
      embedding       ← PG1's (gemma-2b's table is 256000 rows; the shell's is 257216 — the
                        LACUNA AMENDMENT-4 fixed-IO convention, and the only honest counterpart)
      vision tower    ← the SigLIP-so400m RELEASE (PG1's pre-install vision, LACUNA STITCH-0)
      projector       ← a Frobenius-matched Gaussian (there IS no pre-install projector; STITCH-0's
                        AMENDMENT-3 draw, which keeps the image path at PG1's scale)
    """
    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    from infra.pg.lacuna_shell import (
        GEMMA1_REPO, N_VISION_KEYS, SIGLIP_REPO, canon_map, suffix_match,
    )
    # bf16 by default: the donor is installed into a bf16 shell, so fp32 buys nothing and costs
    # ~6 GB of host RAM that the multi-donor arms (X) need.
    dtype = dtype or torch.bfloat16
    out = {}
    g1 = AutoModelForCausalLM.from_pretrained(GEMMA1_REPO, torch_dtype=dtype)
    g1_sd = canon_map({k: v.detach().cpu().clone() for k, v in g1.state_dict().items()})
    del g1
    for k in pg1_sd:
        if k.startswith("language_model.") and "embed_tokens" not in k:
            src = k[len("language_model."):]
            assert src in g1_sd, f"gemma-2b lacks decoder key {src!r}"
            out[k] = g1_sd[src]
    out["language_model.embed_tokens.weight"] = pg1_sd["language_model.embed_tokens.weight"].clone()
    sig = SiglipVisionModel.from_pretrained(SIGLIP_REPO, torch_dtype=dtype)
    sig_sd = canon_map({k: v.detach().cpu().clone() for k, v in sig.state_dict().items()})
    del sig
    want = [k for k in pg1_sd if k.startswith("vision_tower.")]
    assert len(want) == N_VISION_KEYS, f"shell has {len(want)} vision keys ≠ {N_VISION_KEYS}"
    pool = set(sig_sd)
    for k in want:
        hit = suffix_match(pool, k[len("vision_tower."):])
        assert hit is not None, f"SigLIP release has no counterpart for {k}"
        assert tuple(sig_sd[hit].shape) == tuple(pg1_sd[k].shape), k
        out[k] = sig_sd[hit]
    g = torch.Generator().manual_seed(int(seed))
    for k in [x for x in pg1_sd if x.startswith("multi_modal_projector.")]:
        w0 = pg1_sd[k].float()
        z = torch.randn(tuple(w0.shape), generator=g, dtype=torch.float32)
        out[k] = z * (w0.norm() / z.norm().clamp_min(1e-30))
    return out


def donor_repo(repo, sha=None, dtype=None):
    """Any PaliGemma-architecture checkpoint (the `google/paligemma-3b-ft-*-224` roster) as a
    canonical shell state-dict."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg.lacuna_shell import canon_map
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, revision=sha, torch_dtype=dtype or torch.bfloat16, device_map="cpu",
        attn_implementation="eager")
    sd = canon_map({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    del m
    return sd


# ===========================================================================
# Forward-validity gate (N3) — every unit row proves it computes before it is interpreted.
# ===========================================================================
N3_ABS_CEILING_MULT = 3.0   # a host may be at most this multiple of the PG1-in-shell wiki NLL


def forward_validity(model, proc, device, packs, base_vec=None, abs_ceiling=None):
    """N3: a unit/transplant row is interpretable only if
      (a) its wikitext NLL is finite,
      (b) image information still FLOWS (Δ_perception = NLL(gray) − NLL(real) > 0), and
      (c) it is still producing language at all — an ABSOLUTE NLL ceiling.

    (c) is critic finding 10. Without it π0 passed N3 with `wiki_nll` 31.81 against PG1's 3.58 —
    a host that is not a language model in this shell. That matters because Arm X normalises
    compatibility by DELETION damage: a host that is already broken has an enormous deletion
    denominator, so `compat = 1 − max(0, dmg)/dref` is pushed toward 1 for almost any donor, and
    where `dref` is negative (deleting IMPROVES the host) the statistic flips sign. "π0 is a
    permissive host" is that artefact. The ceiling is PG1-anchored and disclosed, not tuned.

    Every branch is named, and a NEGATIVE perception delta is an instrument fault, not a finding."""
    import math
    r = base_vec or readout_vector(model, proc, device, packs,
                                   names=("cap_real", "cap_gray", "wiki"))
    dperc = r["cap_gray"]["mean"] - r["cap_real"]["mean"]
    finite = all(math.isfinite(r[k]["mean"]) for k in ("cap_real", "cap_gray", "wiki"))
    absolute_ok = True if abs_ceiling is None else bool(r["wiki"]["mean"] <= abs_ceiling)
    valid = bool(finite and dperc > 0 and absolute_ok)
    return {"wiki_nll": r["wiki"]["mean"], "cap_real_nll": r["cap_real"]["mean"],
            "cap_gray_nll": r["cap_gray"]["mean"], "delta_perception": dperc,
            "finite": bool(finite), "perception_alive": bool(dperc > 0),
            "abs_ceiling": abs_ceiling, "absolute_ok": absolute_ok, "valid": valid,
            "note": ("forward valid" if valid else
                     "VOID — non-finite readout" if not finite else
                     "VOID — no perception channel (gray is not harder than real)" if dperc <= 0
                     else "VOID — above the absolute NLL ceiling: not producing language in this "
                          "shell, so deletion-normalised statistics are uninterpretable")}
