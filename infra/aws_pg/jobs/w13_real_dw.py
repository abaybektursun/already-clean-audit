"""W13 §9-1b — regenerate the real ΔW eff-rank + the N3 matched-random null + freeze the thresholds.

Loads the PG2-3b-pt LM decoder (W_after) and the base gemma-2-2b decoder (W_before), the canonical
w12_controls recipe (no descent / no key-norm — pg.language_model.state_dict() aligns directly with
g2.model.state_dict() → 182 matrices). Computes per-matrix ΔW eff-rank (SVD participation ratio, the
W1 machinery), the per-type / per-depth profile, and the N3 matched-random-ΔW eff-rank anchor. From
the real eff-rank it computes the FROZEN W13 thresholds (recipe frozen in the prereg, value recomputed
here): N_sparse = min(64, 0.1·eff-rank), N_dense = 0.5·eff-rank.

This is a PRE-FREEZE preflight input (it reads ΔW geometry, NOT the behavioral verdict) — it does not
unblind the confirmation half. Weights attached (loads the two models). CPU math (fp32 → SVD); runs on
a g6e box only because the Standard CPU quota is held by an unrelated job (per the GPU-quota directive).

    pg run infra.aws_pg.jobs.w13_real_dw --type g6e.2xlarge --s3-prefix results/w13_real_dw --budget-min 60
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _eff_rank(sv):
    """SVD participation ratio (Σσ)²/Σσ² — the W1/weight_diff eff-rank definition."""
    import torch
    sv = sv.to(torch.float64)
    sv = sv[sv > 0]
    return (sv.sum().item() ** 2) / (sv.pow(2).sum().item()) if sv.numel() else 0.0


def _layer_of(key):
    """Decoder layer index from a key like 'layers.5.self_attn.q_proj.weight' (or with a prefix)."""
    parts = key.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    raise AssertionError(f"no layer index in decoder key {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null-draws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from infra.pg import gain_routing as GR
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    PG, L0 = ENTITIES["PG"], ENTITIES["L0"]
    api = HfApi()
    pg_sha = api.model_info(PG.repo_id).sha
    l0_sha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
    print(f"[W13-DW] PG={PG.repo_id}@{pg_sha} L0={L0.repo_id}@{l0_sha} torch={torch.__version__}")

    # --- load decoders, fp32 CPU (eff-rank is an fp32/fp64 fit, NOT the bf16 graft) ---
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG.repo_id, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    paths = resolve_submodule_paths(pg)
    pg_lm_sd = {k: v.clone() for k, v in pg.get_submodule(paths["language_model"]).state_dict().items()}
    del pg
    g2 = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    l0_sd = {k: v.clone() for k, v in g2.model.state_dict().items()}
    del g2

    matched = GR.matched_decoder_weight_matrices(pg_lm_sd, l0_sd)   # {key:(w_before, w_after)}
    assert len(matched) == 26 * 7, f"expected 182 decoder matrices, got {len(matched)}"

    # --- per-matrix ΔW eff-rank (fp64 SVD) + rel-Frob ---
    per_tensor = []
    for key, (wb, wa) in sorted(matched.items()):
        dW = (wa.to(torch.float64) - wb.to(torch.float64))
        sv = torch.linalg.svdvals(dW)
        per_tensor.append({
            "key": key, "type": GR.fine_type(key), "layer": _layer_of(key),
            "shape": list(dW.shape),
            "rel_frob": (dW.norm() / wb.to(torch.float64).norm().clamp_min(1e-30)).item(),
            "eff_rank": _eff_rank(sv), "max_rank": int(min(dW.shape)),
        })
    eff = [r["eff_rank"] for r in per_tensor]
    eff_rank_mean = sum(eff) / len(eff)

    by_type = {}
    for t in GR.PROJECTION_TYPES:
        rows = [r for r in per_tensor if r["type"] == t]
        by_type[t] = {"n": len(rows),
                      "eff_rank_mean": sum(r["eff_rank"] for r in rows) / len(rows),
                      "rel_frob_mean": sum(r["rel_frob"] for r in rows) / len(rows),
                      "shape": rows[0]["shape"]}
    depth = {}
    for r in per_tensor:
        depth.setdefault(r["layer"], []).append(r["eff_rank"])
    depth_profile = {str(l): sum(v) / len(v) for l, v in sorted(depth.items())}

    # --- N3 matched-random-ΔW eff-rank null (per unique shape, frob-matched Gaussian) ---
    gen = torch.Generator().manual_seed(args.seed)
    shapes = {}
    for key, (wb, wa) in matched.items():
        shapes.setdefault(tuple(wb.shape), []).append((wa.to(torch.float64) - wb.to(torch.float64)).norm().item())
    n3_by_shape = {}
    for shape, frobs in shapes.items():
        frob = sum(frobs) / len(frobs)
        ers = []
        for _ in range(args.n_null_draws):
            z = GR.matched_random_delta(shape, frob, gen)            # fp64 Gaussian, frob-matched
            ers.append(_eff_rank(torch.linalg.svdvals(z)))
        n3_by_shape[str(shape)] = {"n_matrices": len(frobs), "eff_rank_mean": sum(ers) / len(ers)}
    # weight the N3 anchor by how many real matrices share each shape
    n3_eff_rank_mean = sum(v["eff_rank_mean"] * v["n_matrices"] for v in n3_by_shape.values()) \
        / sum(v["n_matrices"] for v in n3_by_shape.values())

    # --- FROZEN thresholds (recipe frozen in prereg §6B; value recomputed here) ---
    N_sparse = min(64, 0.1 * eff_rank_mean)
    N_dense = 0.5 * eff_rank_mean

    summary = {
        "provenance": {"PG": f"{PG.repo_id}@{pg_sha}", "L0": f"{L0.repo_id}@{l0_sha}",
                       "torch": torch.__version__, "eff_rank_def": "participation ratio (Σσ)²/Σσ², fp64 SVD"},
        "lm": {
            "summary": {"n_matrices": len(per_tensor), "eff_rank_mean": eff_rank_mean,
                        "eff_rank_max": max(eff), "eff_rank_min": min(eff),
                        "rel_frob_mean": sum(r["rel_frob"] for r in per_tensor) / len(per_tensor)},
            "by_matrix_type_fine": by_type,
            "depth_profile": depth_profile,
        },
        "n3_null": {"eff_rank_mean": n3_eff_rank_mean, "by_shape": n3_by_shape,
                    "real_as_fraction_of_n3": eff_rank_mean / n3_eff_rank_mean},
        "thresholds_frozen": {
            "recipe": {"N_sparse": "min(64, 0.1*eff_rank)", "N_dense": "0.5*eff_rank", "tau_star": 0.9},
            "eff_rank_used": eff_rank_mean, "N_sparse": N_sparse, "N_dense": N_dense, "tau_star": 0.9,
        },
    }
    (out / "w1_real_dw_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_tensor_detail.json").write_text(json.dumps(per_tensor, indent=2))
    print(f"[W13-DW] eff_rank_mean={eff_rank_mean:.2f} (max={max(eff):.1f} min={min(eff):.1f}) | "
          f"N3 null eff_rank_mean={n3_eff_rank_mean:.2f} | real/N3={eff_rank_mean/n3_eff_rank_mean:.3f}")
    print(f"[W13-DW] by-type eff_rank: " + ", ".join(f"{t}={by_type[t]['eff_rank_mean']:.0f}" for t in GR.PROJECTION_TYPES))
    print(f"[W13-DW] FROZEN thresholds: N_sparse={N_sparse:.1f}  N_dense={N_dense:.1f}  tau*=0.9")
    print(f"[W13-DW] done → w1_real_dw_summary.json")


if __name__ == "__main__":
    main()
