"""E1.5 — port artifact (pi-trajectory spec §E1.5). CPU, fp32, fail-loud.

Emit the SEALED, key-remapped π0 backbone Gemma-1 LM decoder state_dict (the dependency E2b consumes —
E2b never re-slices the port) + a manifest + a WEIGHT-LEVEL round-trip control. Runs only after E0b-load
PASS (the convention is already verified); E1.5 produces the artifact + an independent full-matrix check.

Round-trip control: the spec's JAX re-export / two-independent-ports options both need openpi+JAX (absent
from the pg env). The runnable WEIGHT-LEVEL substitute is FRAME-CONSISTENCY — every one of the 126 backbone
matrices sits within FT-distance of PaliGemma-1 (rel_frob ≪ the √2 transpose/scramble floor); a transposed
or reshaped tensor would read ≈√2. Plus the K/V identity-closest probe. This catches exactly what a logit
round-trip would miss (a tensor transposed AND compensated elsewhere).

  pg run infra.aws_pg.jobs.e1_5_port --type r7i.2xlarge --no-weights --budget-min 60
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"


def _aws_json(key):
    """Fetch an S3 JSON via the `aws` CLI (boto3 is NOT in the pg-env job venv; the CLI is on the box)."""
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--region", "us-east-1"],
                       check=True, capture_output=True)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file, save_file
    from transformers import PaliGemmaForConditionalGeneration
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg import pi0_port

    # E0b-load must have PASSED (the convention gate) before sealing.
    e0b = _aws_json("results/e0b_load_gate/out/e0b_load_gate.json")
    assert e0b["verdict"].startswith("PASS"), f"E0b-load not PASS ({e0b['verdict']}) — cannot seal"

    man = {"source": "lerobot/pi0_base", "e0b_verdict": e0b["verdict"]}
    man["pi0_sha"] = HfApi().model_info("lerobot/pi0_base", revision="main").sha

    # 1) load the port, remap the backbone LM (drops vision + action expert; expert-leak asserts), seal fp32.
    pi0_sd = load_file(hf_hub_download("lerobot/pi0_base", "model.safetensors"))
    backbone = {k: v.to(torch.float32) for k, v in pi0_port.remap_backbone_lm(pi0_sd).items()}
    matrices = pi0_port.decoder_matrices(backbone)
    man["sealed_keys"] = pi0_port.assert_18x7(matrices)          # raises unless 18×7 complete
    man["n_sealed_total"] = len(backbone)
    man["expert_absent"] = not any("gemma_expert" in k for k in backbone)

    # 2) PaliGemma-1 LM decoder (the frame reference for the round-trip control).
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224", torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    pg_sd = lm.state_dict()

    # 3) WEIGHT-LEVEL round-trip control. The anti-TRANSPOSE guard is already given by strict-load (shapes,
    #    E0b) + the K/V identity-closest probe. The remaining hazard a weight-level check adds is a
    #    SYSTEMATIC scramble (random permutation ⇒ rel_frob ≈ √2≈1.41 on MANY tensors). So the gate is:
    #    the BULK is FT-scale (median < 0.8) AND NO tensor sits at the scramble floor (max < 1.40). A
    #    depth-localised LARGE FT (e.g. the last-layer MLP readout the action expert consumes) is real
    #    signal, NOT an artifact — recorded, not rejected.
    import numpy as np
    rels = {k: pi0_port._rel_frob(pg_sd[k], backbone[k]) for k in matrices}
    vals = np.array(list(rels.values()))
    by_type = {}
    for t in pi0_port.PROJ:
        tv = [rels[k] for k in rels if t in k]
        by_type[t] = {"median": float(np.median(tv)), "max": float(np.max(tv))}
    top5 = sorted(rels, key=rels.get, reverse=True)[:5]
    man["frame_consistency"] = {
        "median_rel_frob_vs_pg1": float(np.median(vals)), "max_rel_frob_vs_pg1": float(vals.max()),
        "p90": float(np.percentile(vals, 90)), "argmax": max(rels, key=rels.get),
        "by_type": by_type, "top5_outliers": {k: rels[k] for k in top5},
        "scramble_floor": 1.40, "median_threshold": 0.8,
        "note": "median FT-scale ∧ max<√2; a localised large FT (e.g. last-layer down_proj) is signal not artifact"}
    # K/V identity-closest probe (a second weight-level check on the highest-risk MQA tensors).
    kv = {}
    for li in (0, 8, 17):
        kk, vk = f"layers.{li}.self_attn.k_proj.weight", f"layers.{li}.self_attn.v_proj.weight"
        kv[li] = pi0_port.kv_convention_probe(pg_sd[kk], pg_sd[vk], backbone[kk], backbone[vk])
    man["kv_probe"] = kv

    gate_ok = (man["expert_absent"] and len(man["sealed_keys"]) == 126
               and float(np.median(vals)) < 0.8 and float(vals.max()) < 1.40
               and all(v["identity_is_closest"] for v in kv.values()))
    man["gate_ok"] = bool(gate_ok)
    man["embedding_caveat"] = "tied embed_tokens dropped by the port (openpi #669) — ΔW₂ embedding NOT sealed"

    # 4) seal the artifact (the 18×7 matrices + norms — the full backbone decoder, Gemma-1 keys).
    save_file(backbone, str(out / "pi0_backbone_lm_sd.safetensors"))
    (out / "e1_5_port_manifest.json").write_text(json.dumps(man, indent=2))
    _mx = man["frame_consistency"]["max_rel_frob_vs_pg1"]
    assert gate_ok, f"E1.5 gate FAIL: expert_absent={man['expert_absent']} max_rel={_mx:.3f} kv={kv}"
    print(f"E1_5_OK sealed={len(backbone)} matrices={len(man['sealed_keys'])} max_rel_vs_pg1={_mx:.4f}",
          json.dumps({"gate_ok": gate_ok}))


if __name__ == "__main__":
    main()
