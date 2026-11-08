"""W10 Λ-locus (headline) + τ (enrichment) — cross-modal text-concept alignment.

Λ (§6.4, the decisive H_already-readable vs H_bridge-installs cut): does text-concept alignment
of vision reps already sit in the encoder, or JUMP at the projector? On the RefCOCO box↔phrase
correspondence set —
  pre  = PG2 vision_tower patch act at the GT-box patch (pre-projector, 1152)
  post = PG2 multi_modal_projector output at that patch (post-projector, 2304)
  text = mean PG2 embed_tokens of the box's referring phrase (the LM text-concept space)
Λ_lift(point) = procrustes aligned-cosine lift over the permuted floor (decoder_alignment).
Λ jump = post_lift − pre_lift. high pre≈post → already-readable; low pre + jump at post →
bridge-installs. Using PG2's OWN pre vs post isolates the PROJECTOR (W1: V0≈V1, ΔW 0.042, so
this ≡ the prereg's SigLIP-base-pre vs PG2-post intent while removing the V0→V1 confound).

τ (§6.1, ENRICHMENT per the pre-freeze amendment — NOT the headline): is SigLIP born more
text-aligned than a pure-vision encoder? siglip-base pre vs dinov2 pre, each aligned to the SAME
text anchor; the statistic is the **SigLIP − DINOv2 DIFFERENTIAL** (never the absolute). The
anchor is the LM's text-token embedding, so this is the CROSS-MODAL (strong) sense; DINOv2 is the
floor controlling for "any vision encoder would show this". metrics.json states which sense.

Metric = decoder_alignment.procrustes_aligned_cosine_boot (train/test split — a learned rotation
cannot be gamed on held-out pairs) + mutual_knn_gallery_curve (C3 gallery control). The §4
discovery↔confirmation firewall is routed through DM.enforce_split (the procrustes internal
train/test split is the second, finer firewall). Fail-loud; heavy imports inside functions.

    python -m infra.aws_pg.jobs.w10_locus --out <dir> [--n-pairs 3000]
"""
from __future__ import annotations

import argparse
import json
import pathlib

GRID = 16                      # 224 / patch14 → 16×16 = 256 patch tokens (all three encoders)
REFCOCO_DATASET = "lmms-lab/RefCOCO"
REFCOCO_SPLITS = ("val", "testA", "testB")
SEED = 1337


def _box_to_patch_index(box_yxyx, grid):
    """Normalized (y0,x0,y1,x1) box → flat row-major patch index of the cell at the box center
    (verbatim from modal_pg_w3._box_to_patch_index)."""
    y0, x0, y1, x1 = box_yxyx
    cy = (y0 + y1) / 2.0; cx = (x0 + x1) / 2.0
    row = min(int(cy * grid), grid - 1)
    col = min(int(cx * grid), grid - 1)
    return row * grid + col


def load_locus_pairs(n_pairs):
    """Stream RefCOCO → unique images + per-pair (unique-image idx, box-patch idx, phrase, held).
    Held = sha1(image)[0] < 4 (≈25%, the §4 sealed half), per image (consistent across its pairs)."""
    import hashlib
    from datasets import load_dataset
    from infra.pg.battery_tasks import _refcoco_expr, _refcoco_gt_box

    images, img_hash_to_idx = [], {}
    pair_img_idx, patch_idx, phrases, held = [], [], [], []
    for split in REFCOCO_SPLITS:
        ds = load_dataset(REFCOCO_DATASET, split=split, streaming=True)
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            img = img.convert("RGB")
            digest = hashlib.sha1(img.tobytes()).hexdigest()
            if digest not in img_hash_to_idx:
                img_hash_to_idx[digest] = len(images)
                images.append(img)
            box = _refcoco_gt_box(row)
            pair_img_idx.append(img_hash_to_idx[digest])
            patch_idx.append(_box_to_patch_index(box, GRID))
            phrases.append(_refcoco_expr(row))
            held.append(int(digest[0], 16) < 4)
            if len(phrases) >= n_pairs:
                break
        if len(phrases) >= n_pairs:
            break
    if len(phrases) < n_pairs:
        raise RuntimeError(f"RefCOCO yielded only {len(phrases)}/{n_pairs} pairs.")
    import numpy as np
    return {"images": images, "pair_img_idx": np.array(pair_img_idx), "patch_idx": np.array(patch_idx),
            "phrases": phrases, "held": np.array(held, dtype=bool),
            "n_unique_images": len(images), "n_pairs": len(phrases)}


def _box_reps(encoder, images, device, batch_size=32, want_post=False):
    """Forward unique `images` through `encoder` (vision_harvest._load_encoder); return all patch
    acts as [n_img, 256, d] (pre) and optionally [n_img,256,d_post] (post). float32 CPU."""
    import torch
    from infra.pg import vision_harvest as V
    fwd, info = V._load_encoder(encoder, device)
    pre_all, post_all, n_per = [], [], None
    for s in range(0, len(images), batch_size):
        pre, post = fwd(images[s:s + batch_size])
        b, t, _ = pre.shape
        n_per = t if n_per is None else n_per
        if t != n_per:
            raise RuntimeError(f"{encoder}: ragged patch count {t} != {n_per}")
        pre_all.append(pre.cpu())
        if want_post:
            if post is None:
                raise RuntimeError(f"{encoder}: post-projector requested but encoder has none")
            post_all.append(post.cpu())
    pre_cat = torch.cat(pre_all, 0)
    post_cat = torch.cat(post_all, 0) if want_post else None
    return pre_cat, post_cat, info, int(n_per)


def _gather_box(all_reps, pair_img_idx, patch_idx):
    """all_reps [n_img,256,d] → per-pair box-patch rep [n_pairs,d] via (image idx, patch idx)."""
    import numpy as np
    arr = all_reps.numpy()
    return arr[pair_img_idx, patch_idx, :].astype(np.float64)   # [n_pairs, d]


def _text_anchor(phrases, device):
    """Mean PG2 embed_tokens rep of each phrase → [n_pairs, d_lm]. The LM text-concept anchor
    (vision↔LM-text alignment = the cross-modal/strong sense). Scale-free downstream (procrustes
    l2-normalizes), so the embed-table scale factor is irrelevant."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from infra.pg.entities import ENTITIES
    proc = AutoProcessor.from_pretrained(ENTITIES["PG"].repo_id)
    tok = proc.tokenizer
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa")
    embed = pg.get_input_embeddings()                          # embed_tokens (vocab, d_lm)
    out = np.empty((len(phrases), embed.embedding_dim), dtype=np.float64)
    with torch.inference_mode():
        for i, ph in enumerate(phrases):
            ids = tok(ph, add_special_tokens=False)["input_ids"]
            if not ids:
                ids = tok(ph if ph.strip() else "object")["input_ids"]
            e = embed(torch.tensor(ids)).mean(0)               # mean over phrase tokens
            out[i] = e.numpy().astype(np.float64)
    del pg
    return out


def _align(vision, text, split, name):
    """Procrustes lift (bootstrap CI) + mutual-kNN gallery curve of vision↔text on the SEALED
    confirmation half (the §4 firewall; procrustes' own train/test split is the finer firewall)."""
    import numpy as np
    from infra.pg import decoder_alignment as DA
    conf = split["conf_idx"]
    Xv = vision[conf]; Xt = text[conf]
    pro = DA.procrustes_aligned_cosine_boot(Xv, Xt, n_boot=100, seed=SEED)
    gal = DA.mutual_knn_gallery_curve(Xv, Xt, gallery_sizes=[1000, min(10000, conf.size)],
                                      k=10, seed=SEED)
    return {"name": name, "n_conf": int(conf.size),
            "procrustes_lift_mean": pro["lift_mean"], "procrustes_lift_ci": pro["lift_ci"],
            "procrustes_matched_cosine": pro["matched_cosine_mean"], "mutual_knn_gallery": gal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=3000)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import diff_metrics as DM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[W10-LOC] device={device} n_pairs={args.n_pairs}")

    pairs = load_locus_pairs(args.n_pairs)
    pii, pidx, held = pairs["pair_img_idx"], pairs["patch_idx"], pairs["held"]
    print(f"[W10-LOC] pairs={pairs['n_pairs']} unique_images={pairs['n_unique_images']} held={int(held.sum())}")
    split = DM.enforce_split(held, name="w10-locus")
    print(f"[W10-LOC] disc={split['n_disc']} conf={split['n_conf']}")

    # text anchor (PG2 embed_tokens of phrases) — the LM text-concept space.
    text = _text_anchor(pairs["phrases"], device)              # [n_pairs, d_lm]

    # Λ: PG2 vision pre vs post (same encoder → isolates the projector).
    pre_all, post_all, pg_info, _ = _box_reps("pg2-vision", pairs["images"], device, want_post=True)
    pre_box = _gather_box(pre_all, pii, pidx)                  # [n_pairs, 1152]
    post_box = _gather_box(post_all, pii, pidx)                # [n_pairs, 2304]
    del pre_all, post_all
    if device == "cuda":
        torch.cuda.empty_cache()
    lam_pre = _align(pre_box, text, split, "pg2_pre_projector")
    lam_post = _align(post_box, text, split, "pg2_post_projector")
    lam_jump = lam_post["procrustes_lift_mean"] - lam_pre["procrustes_lift_mean"]

    # τ (enrichment): siglip-base pre vs dinov2 pre → SigLIP − DINOv2 DIFFERENTIAL.
    sig_pre, _, sig_info, _ = _box_reps("siglip-base", pairs["images"], device)
    sig_box = _gather_box(sig_pre, pii, pidx); del sig_pre
    dino_pre, _, dino_info, _ = _box_reps("dinov2", pairs["images"], device)
    dino_box = _gather_box(dino_pre, pii, pidx); del dino_pre
    if device == "cuda":
        torch.cuda.empty_cache()
    tau_sig = _align(sig_box, text, split, "siglip_base")
    tau_dino = _align(dino_box, text, split, "dinov2_floor")
    tau_diff = tau_sig["procrustes_lift_mean"] - tau_dino["procrustes_lift_mean"]

    # which SENSE does τ support? (amendment: state it in one line, never report the absolute alone)
    tau_sense = ("cross-modal (strong): SigLIP vision concepts align with LM text-token reps "
                 "more than the pure-vision DINOv2 floor" if tau_diff > 0 else
                 "no SigLIP advantage over the DINOv2 floor on this cross-modal probe")

    result = {
        "device": device, "n_pairs": pairs["n_pairs"], "n_unique_images": pairs["n_unique_images"],
        "n_disc": split["n_disc"], "n_conf": split["n_conf"],
        "text_anchor": "mean PG2 embed_tokens of the RefCOCO phrase (LM text-concept space)",
        "encoders": {"pg2-vision": pg_info, "siglip-base": sig_info, "dinov2": dino_info},
        "Lambda_locus": {
            "headline": True, "pre": lam_pre, "post": lam_post,
            "jump_post_minus_pre": float(lam_jump),
            "interpretation": ("bridge-installs: text-alignment jumps at the projector"
                               if lam_jump > 0 else
                               "already-readable: alignment present pre-projector, no projector jump"),
        },
        "tau_enrichment": {
            "headline": False, "note": "ENRICHMENT only (pre-freeze amendment); not the decisive cut",
            "siglip": tau_sig, "dinov2_floor": tau_dino,
            "siglip_minus_dinov2_differential": float(tau_diff),
            "sense_supported": tau_sense,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W10-LOC] Λ pre_lift={lam_pre['procrustes_lift_mean']:.4f} "
          f"post_lift={lam_post['procrustes_lift_mean']:.4f} JUMP={lam_jump:.4f} "
          f"→ {result['Lambda_locus']['interpretation']}")
    print(f"[W10-LOC] τ SigLIP−DINOv2 differential={tau_diff:.4f} → {tau_sense}")
    print(f"[W10-LOC] done → {out}/metrics.json")


if __name__ == "__main__":
    main()
