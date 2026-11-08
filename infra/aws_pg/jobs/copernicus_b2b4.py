"""CO-B2 (item 23) + CO-B4 (item 25) — the output shadow at L17 and its L0 analogue.

Spec §3 Tier B / §6.3. Weights-only, CPU-minutes. One job because §3 specifies the SAME instrument
for both: only the map and the layer change.

THE QUESTION. L17's writes concentrate into a subspace, leaving a complement unlit. Is that
complement FUNCTION-SIDE geometry — the directions the output map is least sensitive to, so nothing
gradient-relevant ever points there ("the gradient shadow") — or is it unrelated to the readout?

  CO-B2 (L17): shadow = the bottom-32 RIGHT singular subspace of the effective output map,
      M = lm_head · diag(γ_final) with γ = 1 + final_norm.weight (Gemma offset). Folding the final
      RMSNorm gains in is the whole point: lm_head alone is not the map the residual stream sees.
  CO-B4 (L0): shadow = the bottom-32 right singular subspace of embed_tokens — the residual
      directions the embedding barely writes. The multimodal projector is reported beside it as a
      second input path, never merged into it (two objects, one name is the f_perp lesson).

TARGET, both arms: the residual-side writer bank at that layer — o_proj and down_proj ROW banks
merged as the top-32 eigenvectors of Σ U Uᵀ (the same consensus construction mercator_mq2 uses,
because a "shared" subspace of two 32-dim banks is only well posed as their consensus).

TWO NULLS, per §6.3: a MATCH needs cos² above the Haar band AND above the spectrum-preserving
rotation twin's band. The twin Haar-rotates the map's own singular basis, so it keeps the spectrum
and destroys only the orientation — if the two bands agree, that agreement is itself a check that
neither is measuring the map's conditioning by accident.

    pg run infra.aws_pg.jobs.copernicus_b2b4 --args "" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_b2b4 --budget-min 60
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_rowgram as RG

from infra.aws_pg.jobs.invariant_core import PG1_REPO

ARM = "CO-B2"
QUEUE_ITEM = "23"
WAVE = "WF"
K = 32
N_HAAR = 12          # §6.3: Haar band
N_TWIN = 10          # §6.3: >=10 spectrum-preserving rotation draws
WRITER_TYPES = ("o_proj", "down_proj")


def bottom_right_singular(Wmat, k: int):
    """The k directions in the INPUT (residual) space that the map is least sensitive to."""
    A = np.asarray(Wmat, dtype=np.float64)
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    V = np.ascontiguousarray(Vh.T)            # n x r, sigma DESCENDING
    r = V.shape[1]
    assert k <= r, f"k={k} exceeds economy rank {r}"
    return np.ascontiguousarray(V[:, r - k:]), s


def merged_writer_bank(bank_dir, layer: int, k: int = K):
    """The shared residual-side writer subspace: top-k eigenvectors of Σ U Uᵀ over the o_proj and
    down_proj row banks at this layer."""
    Us = []
    for t in WRITER_TYPES:
        U = RG.load_row_bank(bank_dir, t, layer, "core32")
        Us.append(np.asarray(U, dtype=np.float64))
    d = Us[0].shape[0]
    for U in Us:
        assert U.shape[0] == d, f"writer banks disagree on residual dim: {[u.shape for u in Us]}"
    S = np.zeros((d, d), dtype=np.float64)
    for U in Us:
        S += U @ U.T
    w, V = np.linalg.eigh(S)
    return np.ascontiguousarray(V[:, ::-1][:, :k]), {
        "types": list(WRITER_TYPES), "dim": int(d),
        "per_bank_capture": [float(np.linalg.norm(V[:, ::-1][:, :k].T @ U, "fro") ** 2 / U.shape[1])
                             for U in Us],
        "construction": "top-k eigenvectors of sum U U^T over the o_proj/down_proj row core banks"}


def rotation_twin_band(Wmat, target, k: int, n_draws: int, seed: int):
    """SPECTRUM-PRESERVING twin: rotate the map's right singular basis by a Haar rotation, keep the
    singular values, re-read the bottom-k. Destroys orientation, preserves conditioning."""
    A = np.asarray(Wmat, dtype=np.float64)
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    V = np.ascontiguousarray(Vh.T)
    r = V.shape[1]
    out = []
    for d in range(n_draws):
        R = H.haar_subspace(r, r, seed + 1000 * d)      # r x r orthogonal
        Vr = V @ R
        out.append(C.mean_cos2(np.ascontiguousarray(Vr[:, r - k:]), target))
    return out


def cell(name, Wmat, target, k, seed, at, gates, layer, note):
    shadow, sv = bottom_right_singular(Wmat, k)
    cos2 = C.mean_cos2(shadow, target)
    haar = [C.mean_cos2(H.haar_subspace(target.shape[0], k, seed + 7 * d), target)
            for d in range(N_HAAR)]
    hb = C.band(haar)
    tw = rotation_twin_band(Wmat, target, k, N_TWIN, seed + 55)
    tb = C.band(tw)
    match = bool(cos2 > hb["hi"] and cos2 > tb["hi"])
    row = {"cell": name, "layer": layer, "mean_cos2": cos2, "haar_band": hb,
           "rotation_twin_band": tb, "match": match,
           "sigma_min": float(sv.min()), "sigma_max": float(sv.max()),
           "condition_number": float(sv.max() / max(sv.min(), 1e-30)),
           "shape": [int(x) for x in np.shape(Wmat)], "note": note}
    at.add("b2b4_shadow_alignment", value=cos2, layer=layer, n=k,
           flags={k2: v for k2, v in row.items() if k2 not in ("haar_band", "rotation_twin_band")})
    gates.record(f"b2b4_shadow_match__{name}", match, mean_cos2=cos2,
                 haar_band=[hb["lo"], hb["hi"]], rotation_twin_band=[tb["lo"], tb["hi"]],
                 n_haar=N_HAAR, n_twin=N_TWIN, layer=layer,
                 band="MATCH needs cos² above BOTH the Haar band and the spectrum-preserving "
                      "rotation twin band (§6.3); below either -> H-B0",
                 prediction="B2: shadow alignment above Haar on the residual side (§11, mid); "
                            "B4: L0 shadow weaker than L17's (§11, low)")
    print(f"[b2b4] {name}: cos²={cos2:.5f} haar=[{hb['lo']:.5f},{hb['hi']:.5f}] "
          f"twin=[{tb['lo']:.5f},{tb['hi']:.5f}] -> MATCH={match}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration
    from infra.pg.entities import resolve_submodule_paths

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, revision=base_sha, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])

    # THE LOADER SURFACE, resolved rather than assumed (this was an open BIND-AT-FREEZE row).
    # `_lm_decoder_sd` strips down to lm.model and therefore drops lm_head, which is exactly the
    # matrix CO-B2 needs, so it cannot be reused here.
    head = None
    for owner in (lm, model):
        if hasattr(owner, "lm_head"):
            head = owner.lm_head.weight.detach().to(torch.float64).numpy()
            break
    assert head is not None, "lm_head not found on the language model or the top-level model"
    inner = lm.model if hasattr(lm, "model") else lm
    assert hasattr(inner, "norm"), "final RMSNorm (`norm`) not found on the decoder"
    gamma_final = 1.0 + inner.norm.weight.detach().to(torch.float64).numpy()
    embed = inner.embed_tokens.weight.detach().to(torch.float64).numpy()
    mmproj = None
    for attr in ("multi_modal_projector", "multimodal_projector"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            for sub in ("linear", "proj"):
                if hasattr(mod, sub):
                    mmproj = getattr(mod, sub).weight.detach().to(torch.float64).numpy()
                    break
    print(f"[b2b4] lm_head {head.shape} gamma_final {gamma_final.shape} embed {embed.shape} "
          f"mmproj {None if mmproj is None else mmproj.shape}", flush=True)
    del model

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "b2b4_shadow_match__B2_L17_output_shadow",
            "shadow alignment above Haar on the residual side", confidence="mid",
            basis="spec §11 row CO-B2", family="boundary")
    exp.log("CO-B4", "b2b4_shadow_match__B4_L0_embedding_shadow",
            "the L0 shadow is WEAKER than L17's (embedding side messier)", confidence="low",
            basis="spec §11 row CO-B4", family="boundary")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_b2b4", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM))
    gates = C.Gates(out)

    rep = {"object": "copernicus_b2b4_v1", "arms": ["CO-B2", "CO-B4"], "program": C.PROGRAM,
           "queue_items": ["23", "25"], "base_repo": PG1_REPO, "base_sha": base_sha, "k": K,
           "gamma_convention": "gamma = 1 + norm.weight (Gemma family)",
           "provenance": C.provenance(), "utc": C.utc_now(), "cells": []}

    # ---- CO-B2: L17 output shadow -------------------------------------------------------
    target17, info17 = merged_writer_bank(bank_dir, 17)
    rep["merged_writer_L17"] = info17
    M_eff = head * gamma_final[None, :]        # lm_head . diag(gamma_final)
    rep["cells"].append(cell("B2_L17_output_shadow", M_eff, target17, K, a.seed, at, gates, 17,
                             "effective output map = lm_head . diag(1 + final_norm.weight)"))
    # the unfolded map, as a CONTEXT row: it shows what the gamma fold is worth
    rep["cells"].append(cell("B2_L17_output_shadow_unfolded", head, target17, K, a.seed + 3,
                             at, gates, 17, "lm_head WITHOUT the final-norm fold (context row)"))

    # ---- CO-B4: L0 embedding shadow ------------------------------------------------------
    target0, info0 = merged_writer_bank(bank_dir, 0)
    rep["merged_writer_L0"] = info0
    rep["cells"].append(cell("B4_L0_embedding_shadow", embed, target0, K, a.seed + 11,
                             at, gates, 0, "bottom-32 right singular subspace of embed_tokens"))
    if mmproj is not None and mmproj.shape[0] == target0.shape[0]:
        rep["cells"].append(cell("B4_L0_mmprojector_shadow", mmproj.T, target0, K, a.seed + 13,
                                 at, gates, 0,
                                 "multimodal projector, reported BESIDE the embedding and never "
                                 "merged with it"))

    b2 = next(c for c in rep["cells"] if c["cell"] == "B2_L17_output_shadow")
    b4 = next(c for c in rep["cells"] if c["cell"] == "B4_L0_embedding_shadow")
    rep["verdict_B2"] = C.REFUTER_DID_NOT_FIRE if b2["match"] else C.REFUTER_FIRED
    rep["verdict_B4"] = C.REFUTER_DID_NOT_FIRE if b4["match"] else C.REFUTER_FIRED
    rep["b4_weaker_than_b2"] = bool(b4["mean_cos2"] < b2["mean_cos2"])
    gates.record("b4_weaker_than_b2", rep["b4_weaker_than_b2"],
                 b2_cos2=b2["mean_cos2"], b4_cos2=b4["mean_cos2"],
                 band="§11 CO-B4 predicts the L0 shadow is weaker than L17's",
                 prediction="L0 shadow weaker (§11, low)")
    (out / "copernicus_b2b4.json").write_text(json.dumps(rep, indent=2, default=float))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_B2B4_OK B2={rep['verdict_B2']} (cos²={b2['mean_cos2']:.5f}) "
          f"B4={rep['verdict_B4']} (cos²={b4['mean_cos2']:.5f}) "
          f"b4_weaker={rep['b4_weaker_than_b2']}", flush=True)


if __name__ == "__main__":
    main()
