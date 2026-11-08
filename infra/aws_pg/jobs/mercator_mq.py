"""M-Q stage 1 — the gauge gate: fold PG1's norm gains to 1, rotate the residual stream, and prove
the function did not change (queue standing W1-Q). Wave 1.

This is the prerequisite for the arm's science and for two others: §8 puts M-Q before M-11 and
before M-6. If a rotated PG1 is not the SAME FUNCTION as PG1, then every later comparison between
a rotated fine-tune and an unrotated one is a comparison between two different models, and the
causal Adam-basis test means nothing.

WHAT IS ROTATED. Everything that touches the language model's residual stream, enumerated from the
model rather than assumed:

    embed_tokens          writes residual vectors as ROWS      E   -> E R^T
    multi_modal_projector writes image features into the stream W  -> R W
    q/k/v_proj            read the stream                       W  -> W R^T
    o_proj                writes into the stream                W  -> R W
    gate/up_proj          read the stream                       W  -> W R^T
    down_proj             writes into the stream                W  -> R W
    lm_head               reads the final stream                W  -> W R^T
    input_layernorm, post_attention_layernorm, final norm       gains folded to 1 FIRST

THE TIED HEAD HAS TO BE UNTIED. PG1 ties `lm_head` to `embed_tokens`, but the final norm's gain
folds into lm_head's COLUMNS while embed_tokens needs no such fold — and the two then transform
differently under the rotation (reader vs writer of the same tensor). Folding a tied tensor would
silently apply the head's fold to the embedding as well. The job unties first and records that it
did.

FAIL LOUD: no try/except, no synthetic fallback. Both gates can fail, and a failure here is a
finding about the architecture rather than a bug to route around.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_gauge as G
from infra.pg import mercator_program as P

ARM = "M-Q"
ROTATIONS = ("none", "haar", "hadamard", "permutation", "align")


def _reference_logits(model, processor, device, texts, blank):
    """Logits on a fixed batch, in fp32, for the function-preservation comparison."""
    import torch
    out = []
    with torch.inference_mode():
        for t in texts:
            inp = processor(text=[t], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(next(model.parameters()).dtype))
            out.append(o.logits[0].to(torch.float32).cpu().numpy())
    return np.concatenate(out, axis=0)


def _layers(model):
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.entities import resolve_submodule_paths
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    return lm, resolve_gemma2_layers(lm)


def _norm_names(layer) -> list:
    """The norm submodules this layer actually has, by name — read, not assumed."""
    return sorted(n for n, _m in layer.named_children() if "layernorm" in n or "norm" in n)


def _untie_lm_head(model) -> dict:
    """Give lm_head its own storage if it shares it with the embedding."""
    import torch
    head = model.get_submodule("lm_head") if any(
        n == "lm_head" for n, _ in model.named_modules()) else None
    assert head is not None, "no lm_head submodule found"
    emb = None
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            emb = mod
            break
    assert emb is not None, "no embed_tokens submodule found"
    tied = head.weight.data_ptr() == emb.weight.data_ptr()
    if tied:
        head.weight = torch.nn.Parameter(head.weight.detach().clone())
    return {"was_tied": bool(tied), "untied": bool(tied),
            "why": ("the final norm's gain folds into lm_head's columns while the embedding needs "
                    "no such fold, and the two transform differently under the rotation")}


def fold_model(model) -> dict:
    """Fold every foldable RMSNorm gain to exactly 1, in place. Returns the plan + what was done."""
    import torch
    lm, layers = _layers(model)
    plans, folded, sign_stats = [], 0, []
    for li, layer in enumerate(layers):
        plan = G.plan_fold(_norm_names(layer))
        plans.append(plan)
        assert plan["fully_foldable"], (
            f"layer {li}: {plan['consequence']} — this architecture has no exact rotation gauge, "
            f"which is a finding about the model, not a bug to work around")
        for norm_name, consumer_names in plan["foldable"].items():
            norm = getattr(layer, norm_name)
            parent = layer.self_attn if consumer_names[0][0] in "qkvo" else layer.mlp
            mats = [getattr(parent, c).weight.detach().to(torch.float64).cpu().numpy()
                    for c in consumer_names]
            new_w, new_mats, st = G.fold_gain_into(
                norm.weight.detach().to(torch.float64).cpu().numpy(), mats)
            sign_stats.append({"layer": li, "norm": norm_name, **st})
            with torch.no_grad():
                norm.weight.copy_(torch.from_numpy(new_w).to(norm.weight.dtype))
                for c, W in zip(consumer_names, new_mats):
                    getattr(parent, c).weight.copy_(
                        torch.from_numpy(W).to(getattr(parent, c).weight.dtype))
            folded += 1

    # The final norm feeds lm_head.
    final = lm.get_submodule("model.norm") if any(
        n == "model.norm" for n, _ in lm.named_modules()) else lm.get_submodule("norm")
    head = model.get_submodule("lm_head")
    new_w, (Wh,), st_final = G.fold_gain_into(
        final.weight.detach().to(torch.float64).cpu().numpy(),
        [head.weight.detach().to(torch.float64).cpu().numpy()])
    sign_stats.append({"layer": None, "norm": "final", **st_final})
    with torch.no_grad():
        final.weight.copy_(torch.from_numpy(new_w).to(final.weight.dtype))
        head.weight.copy_(torch.from_numpy(Wh).to(head.weight.dtype))
    folded += 1
    neg = sum(r["n_negative_gain"] for r in sign_stats)
    zero = sum(r["n_zero_gain"] for r in sign_stats)
    return {"n_norms_folded": folded, "n_layers": len(layers), "plan": plans[0],
            "architecture": plans[0]["architecture"],
            "n_norms_needing_signed_fold": sum(1 for r in sign_stats
                                               if not r["used_positive_helper"]),
            "n_negative_gain_coords": neg, "n_zero_gain_coords": zero,
            "gain_min": min(r["gain_min"] for r in sign_stats),
            "gain_max": max(r["gain_max"] for r in sign_stats),
            "sign_stats": sign_stats}


def scale_gauge_model(model, c: float) -> int:
    """Apply a CONSTANT-scale gauge to every foldable norm, in place. Mathematically the identity
    on the function; arithmetically it forces the same reassociation the real fold does, which is
    what makes it the right floor to read the fold against."""
    import torch
    lm, layers = _layers(model)
    n = 0
    for layer in layers:
        plan = G.plan_fold(_norm_names(layer))
        for norm_name, consumer_names in plan["foldable"].items():
            norm = getattr(layer, norm_name)
            parent = layer.self_attn if consumer_names[0][0] in "qkvo" else layer.mlp
            mats = [getattr(parent, cn).weight.detach().to(torch.float64).cpu().numpy()
                    for cn in consumer_names]
            new_w, new_mats = G.constant_scale_gauge(
                norm.weight.detach().to(torch.float64).cpu().numpy(), mats, c)
            with torch.no_grad():
                norm.weight.copy_(torch.from_numpy(new_w).to(norm.weight.dtype))
                for cn, W in zip(consumer_names, new_mats):
                    getattr(parent, cn).weight.copy_(
                        torch.from_numpy(W).to(getattr(parent, cn).weight.dtype))
            n += 1
    return n


# The tensor groups the gauge touches. Named so a failing gate can be BISECTED to a group
# instead of guessed at: rotating a strict subset is NOT function-preserving by design, so the
# bisect reads which group's omission or inclusion moves the logits, not whether it passes.
ROTATE_GROUPS = ("attn_readers", "o_proj", "mlp_readers", "down_proj", "embed", "lm_head",
                 "projector")


def rotate_model(model, R, groups=ROTATE_GROUPS) -> dict:
    """Apply the gauge in place, optionally to a SUBSET of tensor groups.

    Returns the per-role counts and the per-group counts. The subset form exists for the bisect:
    the full gauge is the only function-preserving one, so a partial rotation is a diagnostic, and
    the artifact records which groups were included so a partial run can never be mistaken for the
    real thing.
    """
    import torch
    lm, layers = _layers(model)
    sel = tuple(groups)
    for g in sel:
        assert g in ROTATE_GROUPS, f"unknown rotate group {g!r}"
    counts = {"reader": 0, "writer": 0}
    per_group = {g: 0 for g in ROTATE_GROUPS}

    n_bias = {"rotated": 0, "skipped_reader": 0}

    def _apply(mod, role, group):
        if group not in sel:
            return
        with torch.no_grad():
            W = mod.weight.detach().to(torch.float64).cpu().numpy()
            mod.weight.copy_(torch.from_numpy(G.rotate_pair(W, role, R)).to(mod.weight.dtype))
            # A WRITER'S BIAS LIVES ON THE RESIDUAL STREAM and must rotate with it: the module
            # contributes W a + b to the stream, so b -> R b. A reader's bias lives in that
            # module's OUTPUT space, which the gauge does not touch. PaliGemma's multimodal
            # projector is a Linear WITH bias, and leaving it unrotated is exactly what put a
            # deterministic 0.118 error into layer 0's image positions.
            if getattr(mod, "bias", None) is not None:
                if role == "writer":
                    b = mod.bias.detach().to(torch.float64).cpu().numpy()
                    Rb = np.asarray(R, dtype=np.float64) @ b
                    mod.bias.copy_(torch.from_numpy(Rb).to(mod.bias.dtype))
                    n_bias["rotated"] += 1
                else:
                    n_bias["skipped_reader"] += 1
        counts[role] += 1
        per_group[group] += 1

    for layer in layers:
        for t in ("q_proj", "k_proj", "v_proj"):
            _apply(getattr(layer.self_attn, t), "reader", "attn_readers")
        _apply(layer.self_attn.o_proj, "writer", "o_proj")
        for t in ("gate_proj", "up_proj"):
            _apply(getattr(layer.mlp, t), "reader", "mlp_readers")
        _apply(layer.mlp.down_proj, "writer", "down_proj")

    emb = None
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            emb = (name, mod)
            break
    assert emb is not None, "no embed_tokens found"
    _apply(emb[1], "reader", "embed")          # (vocab, d): the residual axis is axis 1
    head = model.get_submodule("lm_head")
    assert head.weight.data_ptr() != emb[1].weight.data_ptr(), (
        "lm_head still shares storage with embed_tokens; rotating one would rotate the other "
        "twice. _untie_lm_head must run first")
    _apply(head, "reader", "lm_head")
    proj = None
    for name, mod in model.named_modules():
        if "multi_modal_projector" in name and hasattr(mod, "weight") and mod.weight.ndim == 2:
            proj = (name, mod)
            break
    assert proj is not None, "no multi_modal_projector weight found"
    _apply(proj[1], "writer", "projector")
    return {**counts, "per_group": per_group, "groups": list(sel), "bias": n_bias,
            "embed_module": emb[0], "projector_module": proj[0]}


def _hidden_states(model, processor, device, texts, blank):
    """Per-layer hidden states on the fixed batch, on CPU in fp64."""
    import torch
    outs = []
    with torch.inference_mode():
        for t in texts:
            inp = processor(text=[t], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(next(model.parameters()).dtype),
                      output_hidden_states=True)
            outs.append([h[0].to(torch.float64).cpu().numpy() for h in o.hidden_states])
    return outs


def locate_break(ref_hidden, rot_hidden, R) -> dict:
    """Find the FIRST layer whose hidden state is not the rotated version of the reference.

    A correct residual-stream gauge makes every hidden state transform as h -> R h. Comparing them
    layer by layer turns "the logits moved" into "the stream stopped being the rotated stream at
    layer k", which is the difference between a diagnosis and a guess: a break at the very first
    entry is the embedding/projector path, and a break at layer k with layer k-1 clean is inside
    that block.
    """
    import numpy as np
    assert len(ref_hidden) == len(rot_hidden), "different batch sizes"
    prof = []
    for bi, (ra, rb) in enumerate(zip(ref_hidden, rot_hidden)):
        assert len(ra) == len(rb), f"batch {bi}: different layer counts"
        for li, (ha, hb) in enumerate(zip(ra, rb)):
            expect = ha @ np.asarray(R, dtype=np.float64).T      # rows are token vectors
            scale = float(np.abs(expect).max()) + 1e-30
            d = float(np.abs(hb - expect).max())
            prof.append({"batch": bi, "layer": li, "max_abs_diff": d,
                         "rel": d / scale, "scale": scale})
    first = next((r for r in prof if r["rel"] > 1e-4), None)
    return {"profile": prof, "first_break": first,
            "n_layers": len(ref_hidden[0]), "n_batch": len(ref_hidden),
            "clean_through_layer": (first["layer"] - 1) if first else len(ref_hidden[0]) - 1}


def main():
    ap = argparse.ArgumentParser(description="M-Q stage 1 — the fold + rotation gauge gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotation", default="hadamard", choices=list(ROTATIONS))
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--locate", action="store_true",
                    help="compare hidden states layer by layer and report the first layer where the stream stops being the rotated stream")
    ap.add_argument("--groups", default=",".join(ROTATE_GROUPS),
                    help="tensor groups to rotate; a strict subset is a BISECT diagnostic and is recorded as such")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id=f"m_q_{a.rotation}", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image

    assert torch.cuda.is_available(), "no CUDA"
    # TF32 MUST BE OFF FOR A GAUGE GATE, and the first runs are the proof.
    #
    # PyTorch runs fp32 matmuls through TF32 on Ampere-and-later by default: ~10 mantissa bits,
    # so ~1e-3 relative. The identity round trip was BIT-EXACT because the weights were unchanged,
    # so the same TF32 computation ran on the same inputs. But a PERMUTATION gauge permutes the
    # weights exactly - no rounding at all - and the gate still failed, localized to layer 0 at
    # 2.5e-4 relative. Exact weights cannot produce that; only a change in ACCUMULATION ORDER can,
    # and permuting the operands is exactly such a change. The error then compounds through 18
    # layers into the 21.6 seen at the logits.
    #
    # So the gauge was never wrong; the arithmetic was. Turning TF32 off makes the fp32 path
    # actually fp32, which is what a 1e-4 logit tolerance presumes in the first place.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    # fp32 on purpose: the gate's tolerance is 1e-4 on logits, and bf16's own rounding is larger
    # than that, so a bf16 model could not distinguish a broken gauge from its own arithmetic.
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32, device_map="cuda",
        attn_implementation="eager").eval()
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    texts = ["The capital of France is Paris, and the capital of Italy is",
             "In 1969 the first humans landed on the",
             "Water boils at one hundred degrees",
             "The quick brown fox jumps over the lazy"][: a.n_seq]

    ref = _reference_logits(model, processor, "cuda", texts, blank)
    ref_hidden = _hidden_states(model, processor, "cuda", texts, blank) if a.locate else None

    # UNTIE FIRST. PG1 ties lm_head to embed_tokens, and every probe below rotates BOTH — on a
    # shared tensor that means rotating the same storage twice, which is a no-op only for the
    # identity and a silent double-rotation for anything else. Doing it before the probes is what
    # makes the probes measure what they claim to.
    untie = _untie_lm_head(model)

    # THE ARITHMETIC NOISE FLOOR, measured before anything is folded. The identity rotation is
    # mathematically a no-op but performs every cast, transpose and matmul the real rotation does,
    # so whatever the logits move by here is what the SURGERY PIPELINE costs in fp32 — the floor
    # both gates below are read against.
    d0 = int(model.config.text_config.hidden_size)
    # (a) the surgery ROUND TRIP: identity rotation, mathematically a no-op, every cast and
    #     transpose the real rotation performs. Measured because it is worth knowing, and because
    #     if it were large it would dominate everything below.
    rotate_model(model, np.eye(d0))
    ident = G.function_preservation(ref, _reference_logits(model, processor, "cuda", texts, blank))
    at.add("gauge_identity_roundtrip_diff", value=ident["max_abs_diff"],
           model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**ident, "what": "identity rotation through the same code path"})
    gates.record("mq_identity_roundtrip_is_bit_exact", ident["max_abs_diff"] == 0.0, **ident)

    # (b) the REASSOCIATION floor, which is what actually bounds the fold: a constant-scale gauge
    #     is unrelated to the model's gains, is the identity on the function, and forces the same
    #     reassociation of the forward pass that folding gamma into the consumers does.
    n_scaled = scale_gauge_model(model, G.REASSOCIATION_PROBE_SCALE)
    floor = G.function_preservation(ref, _reference_logits(model, processor, "cuda", texts, blank))
    at.add("gauge_reassociation_noise_floor", value=floor["max_abs_diff"],
           model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**floor, "what": "constant-scale gauge", "n_norms": n_scaled,
                  "scale": G.REASSOCIATION_PROBE_SCALE, "spec_absolute_tol": G.FOLD_TOL})
    assert floor["max_abs_diff"] > 0, (
        "the constant-scale gauge was bit-exact, so reassociation costs nothing here and the "
        "spec's absolute tolerance is the correct gate; re-run without the noise-floor branch")

    fold = fold_model(model)
    after_fold = _reference_logits(model, processor, "cuda", texts, blank)
    fp_fold = G.function_preservation(ref, after_fold)
    at.add("gauge_fold_max_abs_logit_diff", value=fp_fold["max_abs_diff"],
           model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**fp_fold, **{k: v for k, v in fold.items() if k != "sign_stats"},
                  "untie": untie, "tol": G.FOLD_TOL})
    fold_gate = G.assert_function_preserved(fp_fold, what="fold",
                                            noise_floor=floor["max_abs_diff"])
    gates.require("mq_fold_preserves_function", fold_gate["passed"], **fold_gate)

    d = d0
    if a.rotation == "none":
        R = np.eye(d)
    elif a.rotation == "haar":
        R = np.asarray(M.random_reference_basis(d, a.seed))
    elif a.rotation == "hadamard":
        R = G.hadamard_rotation(d)
    elif a.rotation == "permutation":
        R = G.permutation_rotation(d, a.seed)
    else:
        bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
        R = G.align_rotation(WT.load_bank(bank_dir, "q_proj", 0, "core32"), seed=a.seed)
    at.add("rotation_orthogonality_error", value=G.orthogonality_error(R),
           flags={"rotation": a.rotation, "d": d})

    sel = tuple(g.strip() for g in a.groups.split(",") if g.strip())
    partial = set(sel) != set(ROTATE_GROUPS)
    counts = rotate_model(model, R, sel)
    loc = None
    if a.locate:
        loc = locate_break(ref_hidden, _hidden_states(model, processor, "cuda", texts, blank), R)
        fb = loc["first_break"]
        at.add("gauge_first_break_layer",
               value=float(fb["layer"]) if fb else float(loc["n_layers"]),
               flags={"first_break": fb, "clean_through_layer": loc["clean_through_layer"],
                      "n_layers": loc["n_layers"], "rotation": a.rotation})
        print("LOCATE " + json.dumps({"first_break": fb,
                                      "clean_through_layer": loc["clean_through_layer"],
                                      "n_layers": loc["n_layers"]}), flush=True)
    after_rot = _reference_logits(model, processor, "cuda", texts, blank)
    fp_rot = G.function_preservation(ref, after_rot)
    at.add("gauge_rotation_max_abs_logit_diff", value=fp_rot["max_abs_diff"],
           model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**fp_rot, "rotation": a.rotation, "tensors": counts, "tol": G.FOLD_TOL})
    if partial:
        # A partial gauge is NOT expected to preserve the function; the number is the diagnostic.
        rot_gate = {"gate": f"rotation:{a.rotation}", "partial": True, **fp_rot,
                    "noise_floor": floor["max_abs_diff"],
                    "ratio_to_noise_floor": fp_rot["max_abs_diff"] / floor["max_abs_diff"]}
        gates.record("mq_rotation_preserves_function", False, bisect=True, groups=list(sel),
                     **{k: v for k, v in rot_gate.items() if k != "gate"})
    else:
        rot_gate = G.assert_function_preserved(fp_rot, what=f"rotation:{a.rotation}",
                                               noise_floor=floor["max_abs_diff"])
        gates.require("mq_rotation_preserves_function", rot_gate["passed"],
                      rotation=a.rotation, **rot_gate)

    info = at.to_parquet()
    M.write_json(out, f"{ARM}_gauge_result.json",
                 {"arm": ARM, "stage": "gauge_gate", "rotation": a.rotation,
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(),
                  "utc": M.utc_now(), "atlas": info, "base_repo": WT.PG1_REPO,
                  "base_sha": base_sha, "d_model": d, "untie": untie, "fold": fold,
                  "tf32_disabled": True,
                  "rotated_tensors": counts, "partial_gauge": partial,
                  "locate": (None if loc is None else
                             {k: v for k, v in loc.items() if k != "profile"}),
                  "identity_roundtrip": ident, "noise_floor": floor, "fold_gate": fold_gate, "rotation_gate": rot_gate,
                  "logit_batch": texts})
    print("MQ_GAUGE_OK " + json.dumps({"rotation": a.rotation,
                                       "fold_diff": fp_fold["max_abs_diff"],
                                       "rot_diff": fp_rot["max_abs_diff"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
