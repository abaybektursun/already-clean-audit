"""W10 vision-tower activation harvest (prereg §2 treatment axis, §4 data).

Four encoders × two representation points, on the frozen W4/W9 probe images (RefCOCO-testA +
TextVQA — natural images spanning PG task types). All four are ViT patch14 @224 → 256 patch
tokens, so token counts match across encoders (prereg §4 C2 parity).

  encoders : dinov2     — pure-vision floor (self-distilled, NOT text-trained), 768-d
             siglip-base — V0, the public SigLIP that inits PG2's vision_tower, 1152-d
             pg2-vision  — V1, PG2-3b-pt-224's vision_tower, 1152-d
             random      — N3 null: a random-init SigLIP (same config as V0), 1152-d
  points   : pre  — each encoder's last_hidden_state patch tokens
             post — PG2 vision_tower → multi_modal_projector (V1 only; the bridge output. The
                    projector module + its output dim are RESOLVED FROM THE MODEL, never
                    hardcoded — PaliGemma2-3b's LM hidden differs from the Gemma-4 VLA code's).

Discovery↔confirmation split is per IMAGE (avoids same-image patch leakage across the firewall),
using the SAME content-hash rule as diff_harvest (sha1(image)[0] < 4 ≈ 25% held), then expanded
to per-patch and routed through DM.enforce_split at readout time. Each encoder uses its own image
processor on the SAME raw PIL images; V0/V1/random share the SigLIP processor (identical
preprocessing — the load-bearing ρ_novel^vis + Λ comparison), DINOv2 uses its own (different
normalisation, recorded — prereg §4 C1: cross-encoder norm handled at metric time, raw acts kept).

Fail-loud; heavy imports inside functions.
"""
from __future__ import annotations

from infra.pg.contrast_a1 import V0_SIGLIP_REPO
from infra.pg.diff_harvest import MM_PROBES

ENCODERS = ("dinov2", "siglip-base", "pg2-vision", "random")
DINOV2_REPO = "facebook/dinov2-base"          # ViT-B/14 @224 → 256 patch tokens (+CLS, dropped)
SEED = 1337


def load_probe_images(n_per_probe=384, seed=SEED):
    """Stream the frozen MM_PROBES image set → raw RGB PIL images + per-image held mask + a
    content-hashed manifest. Held = sha1(image_bytes)[0] < 4 (≈25%, the prereg §4 sealed half),
    identical to diff_harvest.build_mm_batch so the firewall is consistent across the study."""
    import hashlib
    import numpy as np
    from datasets import load_dataset

    # Dedup by content hash GLOBALLY: the vision tower never sees the prompt, so RefCOCO's
    # multiple referring-expressions per image (and any cross-probe repeat) are pure duplicates
    # that would correlate the SAE training data. Stream until n_per_probe UNIQUE images.
    images, held, tasks, hashes, seen = [], [], [], [], set()
    for probe in MM_PROBES:
        ds = load_dataset(probe["dataset"], split=probe["split"], streaming=True)
        n = 0
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            img = img.convert("RGB")
            digest = hashlib.sha1(img.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            images.append(img)
            held.append(int(digest[0], 16) < 4)
            tasks.append(probe["task"])
            hashes.append(digest)
            n += 1
            if n >= n_per_probe:
                break
        if n < n_per_probe:
            raise RuntimeError(f"{probe['task']}: only {n}/{n_per_probe} UNIQUE images streamed.")

    held = np.array(held, dtype=bool)
    manifest = {
        "probes": [{"task": p["task"], "dataset": p["dataset"], "split": p["split"],
                    "n_images": n_per_probe} for p in MM_PROBES],
        "n_images": len(images), "n_held": int(held.sum()), "seed": seed,
        "content_hashes": hashes, "tasks": tasks,
    }
    return {"images": images, "held": held, "tasks": tasks, "manifest": manifest}


def _load_encoder(name, device):
    """Return (forward_fn, info). forward_fn(pil_batch) -> (pre [B,256,d], post [B,256,Dproj]|None),
    both on `device`. Each encoder is frozen + eval. The PG2 projector (post point) is resolved
    from the live module tree, not assumed."""
    import torch
    from transformers import AutoImageProcessor

    if name == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(DINOV2_REPO, torch_dtype=torch.float32).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(DINOV2_REPO)

        def fwd(pil_batch):
            pv = proc(images=pil_batch, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                pre = model(pixel_values=pv).last_hidden_state[:, 1:, :]   # drop CLS → 256 patches
            return pre.float(), None
        return fwd, {"repo": DINOV2_REPO, "d_pre": int(model.config.hidden_size)}

    if name in ("siglip-base", "random"):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        proc = AutoImageProcessor.from_pretrained(V0_SIGLIP_REPO)
        if name == "siglip-base":
            model = SiglipVisionModel.from_pretrained(V0_SIGLIP_REPO, torch_dtype=torch.float32)
        else:
            torch.manual_seed(SEED)                                        # N3 random-init null
            model = SiglipVisionModel(SiglipVisionConfig.from_pretrained(V0_SIGLIP_REPO))
        model = model.to(device).eval()

        def fwd(pil_batch):
            pv = proc(images=pil_batch, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                pre = model(pixel_values=pv).last_hidden_state                # [B,256,1152]
            return pre.float(), None
        return fwd, {"repo": (V0_SIGLIP_REPO if name == "siglip-base" else f"random-init:{V0_SIGLIP_REPO}"),
                     "d_pre": int(model.config.hidden_size)}

    if name == "pg2-vision":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        from infra.pg.entities import ENTITIES, resolve_submodule_paths
        pg = PaliGemmaForConditionalGeneration.from_pretrained(
            ENTITIES["PG"].repo_id, torch_dtype=torch.float32, device_map=device,
            attn_implementation="sdpa").eval()
        paths = resolve_submodule_paths(pg)
        vt = pg.get_submodule(paths["vision_tower"])
        proj = pg.get_submodule(paths["multi_modal_projector"])
        proc = AutoProcessor.from_pretrained(ENTITIES["PG"].repo_id).image_processor

        def fwd(pil_batch):
            pv = proc(images=pil_batch, return_tensors="pt").pixel_values.to(device).float()
            with torch.inference_mode():
                pre = vt(pixel_values=pv).last_hidden_state                   # [B,256,1152]
                post = proj(pre)                                              # [B,256,Dproj] (resolved)
            return pre.float(), post.float()
        return fwd, {"repo": ENTITIES["PG"].repo_id, "d_pre": int(vt.config.hidden_size),
                     "projector_path": paths["multi_modal_projector"]}

    raise ValueError(f"unknown encoder {name!r}; expected one of {ENCODERS}")


def harvest_encoder(name, images, held_per_image, device, batch_size=32):
    """Push `images` through encoder `name`; return flattened patch activations + per-patch held.

      pre        : float32 [N_patch, d_pre]           (N_patch = n_images × 256)
      post       : float32 [N_patch, Dproj] or None   (only pg2-vision)
      held_patch : bool    [N_patch]                  per-image held expanded to its patches
      image_idx  : int     [N_patch]                  source image index per patch
      n_per_image: int                                patch tokens per image (256)
    """
    import numpy as np
    import torch

    fwd, info = _load_encoder(name, device)
    pre_chunks, post_chunks, n_per = [], [], None
    for s in range(0, len(images), batch_size):
        pre, post = fwd(images[s:s + batch_size])
        b, t, _ = pre.shape
        n_per = t if n_per is None else n_per
        if t != n_per:
            raise RuntimeError(f"{name}: ragged patch count {t} != {n_per} at batch {s}")
        pre_chunks.append(pre.reshape(b * t, -1).cpu())
        if post is not None:
            post_chunks.append(post.reshape(b * t, -1).cpu())

    pre_all = torch.cat(pre_chunks, 0)
    post_all = torch.cat(post_chunks, 0) if post_chunks else None
    held = np.asarray(held_per_image, dtype=bool)
    if held.size != len(images):
        raise RuntimeError(f"held size {held.size} != n_images {len(images)}")
    held_patch = np.repeat(held, n_per)
    image_idx = np.repeat(np.arange(len(images)), n_per)
    return {"encoder": name, "info": info, "pre": pre_all, "post": post_all,
            "held_patch": held_patch, "image_idx": image_idx, "n_per_image": int(n_per),
            "d_pre": int(pre_all.shape[1]), "d_post": (int(post_all.shape[1]) if post_all is not None else None)}
