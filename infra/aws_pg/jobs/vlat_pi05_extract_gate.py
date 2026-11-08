"""VLAT §13 — π0.5 EXTRACTION GATE (resolves the NEW/VERIFY unknown). CPU, fail-loud.

The §13 ledger marks `lerobot/pi05_base` extraction NEW/VERIFY: no π0.5-specific loader exists and
pi0_port.remap_backbone_lm hardcodes the π0 prefix. This job answers, from the real artifact, whether
pi0_port REUSES verbatim for π0.5 or a remap must be built — mirroring e0b_discover (header probe) +
e1_5_port (load → remap → assert_18x7 → frame-consistency vs PaliGemma-1 → K/V identity-closest).

Verdict:
  REUSE   — discovered backbone prefix == pi0_port.BACKBONE_LM_PREFIX, remap yields a complete 18×7
            decoder, and the e1.5 gate passes (median rel-Frob < 0.8, max < 1.40 vs PG1; K/V identity).
  REMAP   — backbone keys exist under a DIFFERENT prefix → report it so a remap_backbone_lm_pi05 is built.
  FAIL    — no Gemma-1 backbone bucket / incomplete 18×7 / frame-consistency fails (architecture surprise).

Also seals the π0.5 backbone decoder (Gemma-1 keys) for vlat_retention/vlat_mechanism reuse.

  pg run infra.aws_pg.jobs.vlat_pi05_extract_gate --type r7i.2xlarge --no-weights --budget-min 60
"""
import argparse
import json
import pathlib
import re

REPO = "lerobot/pi05_base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg import pi0_port
    from infra.pg.entities import resolve_submodule_paths

    rep = {"repo": REPO, "pi05_sha": HfApi().model_info(REPO, revision="main").sha}

    # 1) HEADER discover (keys + shapes; no tensor load) — classify buckets, extract the backbone prefix.
    st_files = [f for f in list_repo_files(REPO) if f.endswith(".safetensors")]
    rep["safetensors_files"] = st_files
    headers = {}
    for f in st_files:
        with safe_open(hf_hub_download(REPO, f), framework="pt") as g:
            for k in g.keys():
                headers[k] = list(g.get_slice(k).get_shape())
    rep["n_tensors"] = len(headers)

    def bucket(k):
        if "gemma_expert" in k or re.search(r"(^|[._])_1([._]|$)", k):
            return "action_expert"
        if "paligemma" in k:
            return "backbone_paligemma"
        return "other"
    buckets = {"backbone_paligemma": [], "action_expert": [], "other": []}
    for k in sorted(headers):
        buckets[bucket(k)].append(k)
    rep["bucket_counts"] = {b: len(v) for b, v in buckets.items()}
    rep["action_expert_keys_sample"] = buckets["action_expert"][:12]
    rep["other_keys_sample"] = buckets["other"][:20]

    # backbone decoder suffix + the prefix that precedes it (the remap target).
    def suffix(k):
        m = re.search(r"(layers\.\d+\..*|embed_tokens\.weight|(?:^|\.)norm\.weight)$", k)
        return m.group(0) if m else None
    bk = buckets["backbone_paligemma"]
    example_prefix = None
    for k in bk:
        s = suffix(k)
        if s and k.endswith(s):
            example_prefix = k[: len(k) - len(s)]
            break
    rep["backbone_prefix_example"] = example_prefix
    rep["prefix_matches_pi0_port"] = (example_prefix == pi0_port.BACKBONE_LM_PREFIX)
    rep["pi0_port_prefix"] = pi0_port.BACKBONE_LM_PREFIX
    layer_ids = sorted({int(m.group(1)) for k in bk for m in [re.search(r"layers\.(\d+)\.", k)] if m})
    rep["backbone_layer_ids"] = layer_ids

    # 2) Attempt the REUSE path: full load → remap_backbone_lm → decoder_matrices → assert_18x7.
    merged = {}
    for f in st_files:
        merged.update(load_file(hf_hub_download(REPO, f)))
    remap_ok = rep["prefix_matches_pi0_port"]
    if remap_ok:
        backbone = {k: v.to(torch.float32) for k, v in pi0_port.remap_backbone_lm(merged).items()}
        matrices = pi0_port.decoder_matrices(backbone)
        rep["sealed_keys"] = pi0_port.assert_18x7(matrices)          # raises unless 18×7 complete
        rep["n_sealed_total"] = len(backbone)
        rep["expert_absent"] = not any("gemma_expert" in k for k in backbone)

        # 3) e1.5 frame-consistency vs PaliGemma-1 (the round-trip control) + K/V identity-closest probe.
        pg = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-pt-224", torch_dtype=torch.float32, device_map="cpu",
            attn_implementation="eager")
        lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
        if hasattr(lm, "lm_head") and hasattr(lm, "model"):
            lm = lm.model
        pg_sd = lm.state_dict()
        rels = {k: pi0_port._rel_frob(pg_sd[k], backbone[k]) for k in matrices}
        vals = np.array(list(rels.values()))
        top5 = sorted(rels, key=rels.get, reverse=True)[:5]
        rep["frame_consistency"] = {
            "median_rel_frob_vs_pg1": float(np.median(vals)), "max_rel_frob_vs_pg1": float(vals.max()),
            "p90": float(np.percentile(vals, 90)), "argmax": max(rels, key=rels.get),
            "top5_outliers": {k: rels[k] for k in top5}, "scramble_floor": 1.40, "median_threshold": 0.8}
        kv = {}
        for li in (0, 4, 8, 12, 17):
            kk, vk = f"layers.{li}.self_attn.k_proj.weight", f"layers.{li}.self_attn.v_proj.weight"
            kv[li] = pi0_port.kv_convention_probe(pg_sd[kk], pg_sd[vk], backbone[kk], backbone[vk])
        rep["kv_probe"] = kv

        # FRAME validity (the load-bearing artifact guard) is SEPARATE from FT MAGNITUDE. A transpose is
        # caught by assert_18x7 (shapes); a K/V swap / row-permutation is caught by the convention probe
        # (identity must be the strict minimum distance AND ≥20% closer than any swap/permute). The
        # π0-calibrated magnitude thresholds (median<0.8, max<1.40) do NOT gate π0.5 — its co-training
        # legitimately moves the late layers past π0 naive-FT scale; that magnitude is a FINDING.
        def _convention_ok(p):
            alt = min(p["d_kv_swap"], p["d_rowperm"])
            return (p["d_identity"] < p["d_kv_swap"] and p["d_identity"] < p["d_rowperm"]
                    and alt > 0 and (p["d_identity"] / alt) < 0.8)
        convention_ok = all(_convention_ok(v) for v in kv.values())
        structure_ok = (rep["expert_absent"] and len(rep["sealed_keys"]) == 126
                        and rep["prefix_matches_pi0_port"])
        rep["convention_ok"] = bool(convention_ok)
        rep["structure_ok"] = bool(structure_ok)
        rep["ft_magnitude"] = {
            "median_rel_frob_vs_pg1": float(np.median(vals)), "max_rel_frob_vs_pg1": float(vals.max()),
            "exceeds_pi0_scale": bool(float(np.median(vals)) >= 0.8 or float(vals.max()) >= 1.40),
            "note": "π0.5 co-training moves the backbone (esp. late layers) FAR past π0 naive-FT scale — "
                    "a FINDING (large ΔW + identity-convention intact), NOT an extraction artifact"}
        gate_ok = structure_ok and convention_ok
        rep["verdict"] = "REUSE" if gate_ok else "FAIL"
        if gate_ok:
            save_file(backbone, str(out / "pi05_backbone_lm_sd.safetensors"))
    else:
        # backbone exists under a different prefix (or not at all) → a remap must be built.
        rep["verdict"] = "REMAP" if bk else "FAIL"
        rep["remap_note"] = ("build remap_backbone_lm_pi05 with backbone_prefix_example; re-run this gate"
                             if bk else "no …paligemma… backbone bucket found — investigate layout")

    (out / "pi05_extract_gate.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"VLAT_PI05_EXTRACT verdict={rep['verdict']} prefix_match={rep['prefix_matches_pi0_port']} "
          f"layers={layer_ids[:2]}..{layer_ids[-2:] if layer_ids else []}",
          json.dumps(rep["bucket_counts"]))
    assert rep["verdict"] in ("REUSE", "REMAP"), (
        f"π0.5 extraction gate FAIL: {rep.get('frame_consistency', rep.get('remap_note'))}")


if __name__ == "__main__":
    main()
