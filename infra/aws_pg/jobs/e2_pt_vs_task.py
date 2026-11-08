"""E2 — PT-vs-task weight decomposition (GraftLM plan §4, Tier-1). EXPLORATORY / DISCOVERY.

Splits the LM weight change into two axes and analyses them on the SAME footing:

  base->pt  (MODALITY INSTALL): Wb = gemma-2-2b decoder, Wa = PG2-3b-pt decoder
  pt->mix   (TASK TUNING):      Wb = PG2-3b-pt decoder,  Wa = PG2-3b-mix decoder

Per LM decoder matrix (q/k/v/o/gate/up/down x 26 layers) compute, with the W8 reconciliation
math (infra.pg.w8_weight_activation, fp64):
  * eff-rank  = SVD participation ratio (Sum sigma)^2 / Sum sigma^2  -> full-rank vs adapter-shaped
  * in-span   = fraction of dW energy inside W_before's top-r output span vs r/out chance
  * rel-Frob  = ||dW||_F / ||W_before||_F  -> magnitude

Headline E2 question: is MODALITY INSTALL the dense, full-rank, in-span rewrite while TASK TUNING
is a thinner / lower-rank adapter on top? -> localises WHERE becoming-a-VLM is expensive (the
GraftLM Q1 "structure / floor"). Both outcomes are publishable.

Fail-fast preflight (research-spec principle 12): base->pt is computed FIRST and its aggregate
eff-rank must reproduce the locked W1 value (~1440/2304, full-rank). A keying/dtype/loading bug
collapses it -> we RAISE before wasting the rest of the run.

CPU only (state_dict math; no GPU, no forward, no datasets). No try/except — fail loud. The three
checkpoints are pre-synced from S3 (HF_HUB_OFFLINE=1). Writes one JSON to --out.

Reuses the EXACT loading + analysis path of infra/modal_pg_w8.py::run_weight so the numbers are
directly comparable to the locked W1/W8 lineage results.
"""
from __future__ import annotations

import argparse
import json
import pathlib

# Frozen E2 config (recorded in the manifest).
LM_MATRIX_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 26                                   # Gemma2-2B decoder depth (contrast_a1: "the 26 Gemma2")
ENERGY_RANKS = (8, 32, 64, 128, 256, 512, 1024)  # in-span energy curve (verbatim W8)
N_RANDOM = 4                                    # Monte-Carlo null draws (analytic r/out is the primary floor)
SEED = 1337
# Locked W1 calibration anchor: base->pt LM eff-rank (full-rank). For the fail-fast cross-check
# ONLY — never quoted as a result (CLAUDE.md baseline rule); E2 recomputes and reports beside it.
W1_LM_EFF_RANK = 1440.3


def _decoder_key(layer: int, mtype: str) -> str:
    """Gemma2 decoder key inside the language_model submodule state_dict."""
    sub = "self_attn" if mtype in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"layers.{layer}.{sub}.{mtype}.weight"


def _load_lm_state_dicts():
    """Load the three LM decoder state_dicts (CPU). base = gemma-2-2b; pt/mix = PG2 language_model.

    Sequential load+extract+free so peak memory ~ one model + the three LM state_dicts.
    """
    import torch
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration

    from infra.pg.contrast_a1 import L0_GEMMA_REPO
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    print(f"[E2] loading base gemma-2-2b ({L0_GEMMA_REPO}) ...", flush=True)
    g2 = AutoModelForCausalLM.from_pretrained(
        L0_GEMMA_REPO, revision=ENTITIES["L0"].revision, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager")
    base = {k: v.clone() for k, v in g2.model.state_dict().items()}
    del g2

    print(f"[E2] loading PG2-pt ({ENTITIES['PG'].repo_id}) ...", flush=True)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa")
    pt_paths = resolve_submodule_paths(pt_model)
    pt = {k: v.clone()
          for k, v in pt_model.get_submodule(pt_paths["language_model"]).state_dict().items()}
    del pt_model

    print(f"[E2] loading PG2-mix ({ENTITIES['PG_mix'].repo_id}) ...", flush=True)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG_mix"].repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa")
    mix_paths = resolve_submodule_paths(mix_model)
    mix = {k: v.clone()
           for k, v in mix_model.get_submodule(mix_paths["language_model"]).state_dict().items()}
    del mix_model

    prov = {"base": L0_GEMMA_REPO, "pt": ENTITIES["PG"].repo_id, "mix": ENTITIES["PG_mix"].repo_id}
    return base, pt, mix, prov


def _analyze_axis(sd_before: dict, sd_after: dict, layers, mtypes) -> dict:
    """Per-matrix eff-rank + in-span energy for one axis; aggregate over matrices."""
    import numpy as np

    from infra.pg import w8_weight_activation as W8

    per_matrix: dict = {}
    prs, rels, e_excess = [], [], []
    for L in layers:
        for mt in mtypes:
            key = _decoder_key(L, mt)
            assert key in sd_before and key in sd_after, \
                f"missing decoder key {key!r} in before/after state_dict"
            Wb = sd_before[key].float().numpy()
            Wa = sd_after[key].float().numpy()
            spec = W8.delta_spectrum(Wb, Wa)
            en = W8.delta_energy_in_base_subspace(
                Wb, Wa, ranks=ENERGY_RANKS, n_random=N_RANDOM, seed=SEED)
            per_matrix[f"L{L}.{mt}"] = {
                "layer": L, "matrix_type": mt, "shape": spec["shape"],
                "participation_ratio": spec["participation_ratio"],
                "participation_ratio_frac_of_maxrank": spec["participation_ratio_frac_of_maxrank"],
                "rel_frob": spec["rel_frob"],
                "energy_top1_frac": spec["energy_top1_frac"],
                "energy_top10_frac": spec["energy_top10_frac"],
                "frac_energy_in_base_topr_real": en["frac_energy_in_base_topr_real"],
                "chance_frac_r_over_out": en["chance_frac_r_over_out"],
                "energy_headline_rank": en["headline_rank"],
                "energy_headline_real": en["headline_real"],
                "energy_headline_chance": en["headline_chance"],
                "energy_headline_excess_over_chance": en["headline_excess_over_chance"],
                "energy_interpretation": en["interpretation"],
            }
            prs.append(spec["participation_ratio"])
            rels.append(spec["rel_frob"])
            e_excess.append(en["headline_excess_over_chance"])
            print(f"[E2] {f'L{L}.{mt}':16s} PR={spec['participation_ratio']:7.1f} "
                  f"relF={spec['rel_frob']:.3f} E_in_base@r{en['headline_rank']}="
                  f"{en['headline_real']:.3f}(chance {en['headline_chance']:.3f})", flush=True)

    prs_a, rels_a, ee_a = np.asarray(prs), np.asarray(rels), np.asarray(e_excess)
    aggregate = {
        "n_matrices": int(prs_a.size),
        "eff_rank_mean": float(prs_a.mean()), "eff_rank_std": float(prs_a.std()),
        "eff_rank_min": float(prs_a.min()), "eff_rank_max": float(prs_a.max()),
        "rel_frob_mean": float(rels_a.mean()), "rel_frob_std": float(rels_a.std()),
        "energy_excess_over_chance_mean": float(ee_a.mean()),
        "energy_excess_over_chance_std": float(ee_a.std()),
    }
    return {"per_matrix": per_matrix, "aggregate": aggregate}


def _headline(base_pt: dict, pt_mix: dict) -> dict:
    """Compare the two axes. Descriptive (exploratory): report the contrast against a
    pre-stated ratio threshold; do not over-claim."""
    bp, pm = base_pt["aggregate"], pt_mix["aggregate"]
    ratio = bp["eff_rank_mean"] / pm["eff_rank_mean"] if pm["eff_rank_mean"] else None
    if ratio is None:
        verdict = "pt->mix eff-rank is zero — degenerate; inspect."
    elif ratio > 1.3:
        verdict = ("MODALITY INSTALL (base->pt) is markedly higher-rank than TASK TUNING "
                   "(pt->mix) -> the dense structural rewrite is becoming-multimodal; task "
                   "tuning is comparatively adapter-shaped.")
    elif ratio < 0.77:
        verdict = ("TASK TUNING (pt->mix) is higher-rank than MODALITY INSTALL (base->pt) -> "
                   "the expensive structural change is the SECOND fine-tune, not install.")
    else:
        verdict = ("Both axes are comparably high-rank -> task tuning is NOT a thin adapter; "
                   "the rewrite is dense at both stages.")
    return {
        "base_pt_eff_rank_mean": bp["eff_rank_mean"],
        "pt_mix_eff_rank_mean": pm["eff_rank_mean"],
        "eff_rank_ratio_install_over_task": ratio,
        "base_pt_rel_frob_mean": bp["rel_frob_mean"],
        "pt_mix_rel_frob_mean": pm["rel_frob_mean"],
        "base_pt_energy_excess_mean": bp["energy_excess_over_chance_mean"],
        "pt_mix_energy_excess_mean": pm["energy_excess_over_chance_mean"],
        "W1_calibration_base_pt_eff_rank": W1_LM_EFF_RANK,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 layer x 2 matrices per axis — wiring check (load+keys+JSON), no preflight assert")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = [12] if a.smoke else list(range(N_LAYERS))
    mtypes = ("q_proj", "gate_proj") if a.smoke else LM_MATRIX_TYPES

    base, pt, mix, prov = _load_lm_state_dicts()

    # base->pt FIRST = fail-fast preflight: its eff-rank must reproduce the locked full-rank W1.
    print("[E2] === axis base->pt (modality install) ===", flush=True)
    base_pt = _analyze_axis(base, pt, layers, mtypes)
    eff = base_pt["aggregate"]["eff_rank_mean"]
    print(f"[E2] base->pt eff_rank_mean={eff:.1f} (locked W1 anchor ~{W1_LM_EFF_RANK})", flush=True)
    if not a.smoke:
        assert eff > 500.0, (
            f"PREFLIGHT FAIL: base->pt eff_rank_mean={eff:.1f} implausibly low vs locked "
            f"W1~{W1_LM_EFF_RANK} -> harness mis-wired (key/dtype/loading bug). Failing loud.")

    print("[E2] === axis pt->mix (task tuning) ===", flush=True)
    pt_mix = _analyze_axis(pt, mix, layers, mtypes)

    head = _headline(base_pt, pt_mix)
    result = {
        "experiment": "E2_pt_vs_task",
        "status": "exploratory",
        "smoke": a.smoke,
        "config": {
            "lm_matrix_types": list(mtypes), "layers": layers, "n_layers": len(layers),
            "energy_ranks": list(ENERGY_RANKS), "n_random_null": N_RANDOM, "seed": SEED,
        },
        "provenance": prov,
        "headline": head,
        "axes": {"base_pt": base_pt, "pt_mix": pt_mix},
    }
    (out / "e2_pt_vs_task.json").write_text(json.dumps(result, indent=2))
    print("E2_OK", json.dumps(head, indent=2), flush=True)


if __name__ == "__main__":
    main()
