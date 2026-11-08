"""DARWIN — the family/scale census instruments (§4 W0-3, W0-4, W0-5, W0-6).

The question these arms exist for: does a consensus-avoided, coordinate-spiky subspace exist
OUTSIDE the Gemma family, and at larger scale? §8 lists "Gemma-only phenomenon" as a threat that
gates the non-Gemma stage-2 plans, so the census must run on bases whose geometry DARWIN has
never seen.

That forces one thing to be generic that HUMBOLDT's A4 hardcoded: A4's config prefilter asserts
gemma-2b's exact geometry (hidden 2048, 18 layers, intermediate 16384, head_dim 256, 1 KV head).
Here the expected geometry is READ OFF THE BASE MODEL'S OWN CONFIG, so the same code censuses
Llama-3.2-3B or Qwen2.5-3B without a per-family branch — and a derivative is rejected for
disagreeing with ITS base, not with Gemma.

Selection is mechanical and logged with rejects, before any projection is computed (§2 external
rosters, §4/W0-3). Nothing here decides anything (§0.1).

FAIL LOUD in the science; candidate ENUMERATION over arbitrary community repos is the one
sanctioned place a broad except is allowed, because an unreadable third-party repo is a
rejection REASON to be recorded, not an arm-ending error. That is the same boundary
humboldt_a4._select and battery_tasks._survey_candidates already draw.
"""
from __future__ import annotations

import json
import pathlib

TAG_FORMS = ("base_model:{}", "base_model:finetune:{}", "base_model:merge:{}")
MIN_FTS_FOR_CENSUS = 12          # §4/W0-3: "bases with >=12 key-aligned full-FT derivatives"
N_EXAMINE_DEFAULT = 80           # hard cap on candidates walked, so an arm cannot run away
ATTN_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TYPES = ("gate_proj", "up_proj", "down_proj")
ALL_TYPES = ATTN_TYPES + MLP_TYPES


def base_geometry(cfg) -> dict:
    """The geometry a derivative must match, read off the BASE's config.

    `text_config` is unwrapped first so a multimodal base (PaliGemma) and a plain causal LM are
    described by the same keys.
    """
    tc = getattr(cfg, "text_config", cfg)
    g = {}
    for k in ("hidden_size", "num_hidden_layers", "intermediate_size", "head_dim",
              "num_attention_heads", "num_key_value_heads"):
        v = getattr(tc, k, None)
        if v is not None:
            g[k] = int(v)
    assert "hidden_size" in g and "num_hidden_layers" in g, \
        f"base config exposes neither hidden_size nor num_hidden_layers: {sorted(vars(tc))[:30]}"
    g["model_type"] = getattr(tc, "model_type", getattr(cfg, "model_type", "?"))
    return g


# The exact filenames THIS ARM'S READER can open — not the wider set transformers accepts.
# Two launches were lost to getting this wrong in opposite directions:
#   D8  a suffix test ("any file ends with .safetensors") admitted
#       `aws-neuron/Mistral-...-cores-2`, whose .safetensors are compiled Neuron graphs. Same
#       class: TensorRT / OpenVINO / GGUF exports and ONNX-only mirrors.
#   D12 the filter then accepted `pytorch_model.bin` while `load_decoder_matrices` — rewritten in
#       D10 to read shards directly, so a vocab-resized derivative cannot abort a census —
#       downloads only `*.safetensors`. `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw3` passed selection
#       and then found no shards to open.
# The invariant both defects violate: THE FILTER MUST ACCEPT EXACTLY WHAT THE LOADER READS.
# `.bin`-only repos are rejected with that stated reason rather than accepted and then failed.
LOADABLE_ENTRYPOINTS = ("model.safetensors", "model.safetensors.index.json")
LEGACY_BIN_ENTRYPOINTS = ("pytorch_model.bin", "pytorch_model.bin.index.json")


def loadable_entrypoint(files) -> bool:
    """True iff the repo carries safetensors DATA this arm's reader can open.

    An index is a POINTER, not data, and the two must not be conflated (defect D13).
    `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw3` ships `model.safetensors.index.json` with no
    `.safetensors` file anywhere — its weights are in `model.uc.bin`, a third-party "ultracompress"
    format — so accepting the index alone selected a repo with nothing to download. The rule is
    therefore: a single-file checkpoint, or an index BACKED BY at least one real shard.
    """
    s = set(files)
    shards = [f for f in s if f.endswith(".safetensors")]
    if "model.safetensors" in s:
        return True
    return "model.safetensors.index.json" in s and bool(shards)


def legacy_bin_only(files) -> bool:
    """True iff the repo has torch .bin weights but no safetensors — a recorded rejection class,
    distinct from 'no weights at all', so the selection log says which it was."""
    s = set(files)
    return any(f in s for f in LEGACY_BIN_ENTRYPOINTS) and not loadable_entrypoint(s)


# Config keys that mark a quantised checkpoint. `quantization_config` is the transformers/BnB/GPTQ
# convention; `quantization` is MLX's. Checking only the former let
# `mlx-community/Qwen2.5-3B-Instruct-4bit` through selection, and its bit-PACKED q_proj —
# (2048, 256) instead of (2048, 2048), 8 values per uint32 — then aborted a census (defect D11).
QUANT_CONFIG_KEYS = ("quantization_config", "quantization")


def quantization_marker(cfg):
    """(is_quantised, description) from the config alone."""
    for key in QUANT_CONFIG_KEYS:
        q = getattr(cfg, key, None)
        if not q:
            continue
        if isinstance(q, dict):
            method = q.get("quant_method") or q.get("bits") or sorted(q)[:3]
        else:
            method = getattr(q, "quant_method", "?")
        return True, f"{key}={method}"
    return False, ""


def cfg_prefilter(cfg, geom: dict):
    """(ok, reason) from the config alone — free, and avoids downloading multi-GB mismatches."""
    is_q, desc = quantization_marker(cfg)
    if is_q:
        return False, f"quantised checkpoint ({desc}) — no dense dW to project"
    tc = getattr(cfg, "text_config", cfg)
    for k, want in geom.items():
        if k == "model_type":
            got = getattr(tc, "model_type", getattr(cfg, "model_type", None))
            if got != want:
                return False, f"model_type={got} != base {want}"
            continue
        got = getattr(tc, k, None)
        if got is None or int(got) != want:
            return False, f"config.{k}={got} != base {want}"
    return True, "config matches the base decoder geometry"


def select_derivatives(api, base_repo: str, geom: dict, *, n_accept: int,
                       n_examine: int = N_EXAMINE_DEFAULT, out=None) -> tuple:
    """The frozen mechanical rule, identical in shape to HUMBOLDT A4's:

      1. query HF for models tagged with base_model {base_repo} across the three tag forms;
      2. dedup, sort by downloads DESCENDING;
      3. walk in order, accept iff the config matches the BASE's geometry and the repo carries
         dense weights;
      4. record the FULL log, accepts and rejects with reasons, plus is_adapter / is_merge flags.

    §8's "PEFT/merge confound" row requires those flags: adapter and merge derivatives are not
    full fine-tunes and their projection fractions are not comparable, so they are rejected from
    the primary rows and the rejection is recorded rather than silent.
    """
    from transformers import AutoConfig
    cands, seen = [], set()
    for form in TAG_FORMS:
        for m in api.list_models(filter=form.format(base_repo), sort="downloads", limit=100):
            if m.id in seen:
                continue
            seen.add(m.id)
            cands.append({"id": m.id, "downloads": int(m.downloads or 0),
                          "tag_form": form.format(base_repo)})
    cands.sort(key=lambda c: -c["downloads"])

    log, accepted = [], []
    for c in cands[:n_examine]:
        if len(accepted) >= n_accept:
            log.append({**c, "decision": "not_examined",
                        "reason": f"quota of {n_accept} already filled"})
            continue
        try:
            files = {x.rfilename for x in api.model_info(c["id"], files_metadata=False).siblings}
            if "adapter_config.json" in files:
                log.append({**c, "decision": "reject", "is_adapter": True,
                            "reason": "adapter-only repo (no full weights) — §8 PEFT confound"})
                continue
            if not loadable_entrypoint(files):
                why = ("legacy torch .bin weights only; this arm's reader opens safetensors"
                       if legacy_bin_only(files) else
                       f"no safetensors entrypoint "
                       f"(has {sorted(f for f in files if f.endswith(('.safetensors', '.bin')))[:4]})")
                log.append({**c, "decision": "reject", "reason": why,
                            "legacy_bin_only": legacy_bin_only(files)})
                continue
            cfg = AutoConfig.from_pretrained(c["id"])
        except Exception as e:      # noqa: BLE001 — enumeration boundary; see module docstring
            log.append({**c, "decision": "reject",
                        "reason": f"unreadable repo: {type(e).__name__}: {str(e)[:200]}"})
            continue
        ok, why = cfg_prefilter(cfg, geom)
        rec = {**c, "model_type": getattr(cfg, "model_type", "?"), "is_adapter": False,
               "is_merge": "merge" in c["tag_form"],
               "ckpt_dtype": str(getattr(cfg, "torch_dtype", None))}
        log.append({**rec, "decision": "accept" if ok else "reject", "reason": why})
        if ok:
            accepted.append(rec)

    rule = {"base_repo": base_repo, "tag_forms": list(TAG_FORMS), "sort": "downloads desc",
            "n_accept": n_accept, "n_examine_cap": n_examine, "geometry": geom,
            "min_for_census": MIN_FTS_FOR_CENSUS}
    payload = {"rule": rule, "n_candidates": len(cands), "log": log,
               "accepted": [r["id"] for r in accepted],
               "census_possible": len(accepted) >= MIN_FTS_FOR_CENSUS}
    if out is not None:
        (pathlib.Path(out) / f"selection_log__{base_repo.replace('/', '_')}.json").write_text(
            json.dumps(payload, indent=2))
    return accepted, payload


def matrix_key(t: str, layer: int) -> str:
    sub = "self_attn" if t in ATTN_TYPES else "mlp"
    return f"layers.{int(layer)}.{sub}.{t}.weight"


# Tensors OUTSIDE the decoder that a candidate must still match for the whole model to LOAD. The
# decoder check below is about the matrices a census reads; this one is about whether the checkpoint
# can be instantiated at all. M-C1 loads each derivative to score perplexity and died 17 of 24
# fine-tunes in on a repo whose vision tower is 768-wide against PG1's 1152 — a mismatch the decoder
# check cannot see, because the decoder was fine (defect M-D54).
LOADABILITY_TAILS = ("multi_modal_projector.linear.weight",)


def decoder_shapes(repo: str, sha: str, types, layers) -> dict:
    """{(type, layer): shape} from the safetensors HEADERS — no weights downloaded.

    Used for the base's reference shapes: the preflight only needs geometry, and pulling a 14 GB
    checkpoint to read six tuples would defeat the point of a header-only preflight.
    """
    from huggingface_hub import get_safetensors_metadata
    meta = get_safetensors_metadata(repo, revision=sha)
    shapes = {}
    for fm in meta.files_metadata.values():
        for name, info in fm.tensors.items():
            shapes[name] = tuple(info.shape)
    assert shapes, f"{repo}: no tensors in safetensors metadata"
    out = {}
    for t in types:
        for L in layers:
            out[(t, L)] = shapes[resolve_key(shapes, t, L)]
    # The loadability tensors are keyed by their STRING tail alongside the (type, layer) tuples, so
    # `validate_candidate` can look them up without a second metadata fetch.
    for tail in LOADABILITY_TAILS:
        hits = [k for k in shapes if k.endswith(tail)]
        if len(hits) == 1:
            out[hits[0]] = shapes[hits[0]]
    return out


def validate_candidate(repo: str, sha: str, base_shapes: dict, types, layers):
    """Verify a candidate's tensors from the SAFETENSORS HEADERS ALONE, before any download.

    `huggingface_hub.get_safetensors_metadata` fetches only the shard headers (kilobytes) and
    returns every tensor's name, shape and dtype. Checking there converts an entire class of
    expensive runtime failures into cheap recorded rejections:

      D8   compiled exports (Neuron/TensorRT/OpenVINO)  -> required key absent
      D11  bit-packed quantisation (MLX 4-bit)          -> shape disagrees with the base
      D12  legacy .bin-only weights                     -> no safetensors metadata at all
      D13  dangling index with no shard                 -> no tensors

    Three consecutive launches were lost to patching those one at a time on the file LISTING,
    which only ever says what a repo is named, never what it contains. This asks the shards.

    Returns (ok, reason).
    """
    from huggingface_hub import get_safetensors_metadata
    try:
        meta = get_safetensors_metadata(repo, revision=sha)
    except Exception as e:      # noqa: BLE001 — enumeration boundary; see module docstring
        return False, f"safetensors metadata unreadable: {type(e).__name__}: {str(e)[:160]}"
    shapes = {}
    for fm in meta.files_metadata.values():
        for name, info in fm.tensors.items():
            shapes[name] = tuple(info.shape)
    if not shapes:
        return False, "no tensors in safetensors metadata (dangling index or non-safetensors weights)"
    for t in types:
        for L in layers:
            tail = matrix_key(t, L)
            hits = _lm_scoped([k for k in shapes if k.endswith(tail)])
            if len(hits) != 1:
                return False, f"expected exactly 1 key ending {tail!r}, found {len(hits)}"
            got, want = shapes[hits[0]], tuple(base_shapes[(t, L)])
            if got != want:
                return False, (f"{tail}: shape {got} != base {want} "
                               f"(undeclared packing/quantisation or reshaped head)")
    for tail in LOADABILITY_TAILS:
        hits = [k for k in shapes if k.endswith(tail)]
        base_hits = [k for k in base_shapes if isinstance(k, str) and k.endswith(tail)]
        if not base_hits:
            continue                      # the base does not carry it; nothing to match against
        want = tuple(base_shapes[base_hits[0]])
        if len(hits) != 1:
            return False, f"expected exactly 1 key ending {tail!r}, found {len(hits)}"
        if shapes[hits[0]] != want:
            return False, (f"{tail}: shape {shapes[hits[0]]} != base {want} — the checkpoint "
                           f"cannot be loaded into the base architecture (different vision tower)")
    return True, f"all {len(types) * len(layers)} decoder matrices present at base shapes"


def _lm_scoped(hits):
    """Drop vision-tower keys when a decoder key is also present.

    PaliGemma's SigLIP tower carries the SAME `layers.N.self_attn.q_proj.weight` tail as the
    language model, so a suffix match returns two keys and every caller that assumed one has
    failed on it — journal M-D6, M-D37, and again at M-D57 when a new call site was added. The
    disambiguation is not a guess: the vision keys are under `vision_tower.`, and a census of a
    language decoder never wants them. Bases without a vision tower return a single hit and are
    untouched.
    """
    if len(hits) <= 1:
        return list(hits)
    lm = [k for k in hits if "vision_tower" not in k and "vision_model" not in k]
    return lm if lm else list(hits)


def preflight_roster(base_repo, base_sha, accepted, shas, types, layers, *, tag="preflight"):
    """Header-validate an accepted roster, returning (kept, log). ONE implementation, because
    three arms need it and the two that had it were the two that did not crash.

    `select_derivatives` accepts on the roster's own criteria — declared base, shape family, recency
    — none of which say whether a checkpoint will LOAD. Every arm that then instantiates the model
    (M-C1 scores perplexity, M-B1-distinct streams weights) can meet a repo whose vision tower is a
    different width, and finds out mid-stream after the earlier models have been downloaded. This
    asks the shard headers first: kilobytes, and the answer is the same one `validate_candidate`
    gives per candidate.
    """
    base_shapes = decoder_shapes(base_repo, base_sha, types, layers)
    kept, log = [], []
    for rec in accepted:
        ok, why = validate_candidate(rec["id"], shas[rec["id"]], base_shapes, types, layers)
        log.append({"id": rec["id"], "ok": ok, "reason": why})
        if ok:
            kept.append(rec)
        else:
            print(f"[{tag}] DROP {rec['id']}: {why}", flush=True)
    print(f"[{tag}] {len(kept)}/{len(accepted)} derivatives passed the header preflight",
          flush=True)
    return kept, log


def resolve_key(available, t: str, layer: int) -> str:
    """Find the checkpoint key holding one decoder matrix.

    Checkpoints disagree on the prefix (`model.layers.…`, `language_model.model.layers.…`, bare
    `layers.…`), so the key is located by SUFFIX match on the unambiguous tail. Fails loud on
    zero or multiple matches rather than guessing — a wrong key here would silently census the
    wrong matrix.
    """
    tail = matrix_key(t, layer)
    hits = _lm_scoped([k for k in available if k.endswith(tail)])
    assert len(hits) == 1, (
        f"expected exactly 1 key ending in {tail!r}, found {len(hits)}: {hits[:4]}")
    return hits[0]


def load_decoder_matrices(repo: str, sha: str, types, layers) -> dict:
    """{(type, layer): fp32 tensor}, read STRAIGHT FROM THE SAFETENSORS SHARDS.

    Deliberately not `AutoModelForCausalLM.from_pretrained`. Instantiating the model pulls in the
    embedding and lm_head, and a fine-tune that resized its vocabulary — routine, and invisible to
    a decoder-geometry prefilter — then aborts the whole census with

        RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error

    even though every decoder matrix this arm reads is perfectly well-formed (journal defect D10).
    Reading the shards directly also skips constructing ~3B parameters of tensors the arm never
    touches, so it is both more robust and considerably faster.

    Fails loud on a missing key: a silently skipped matrix would shrink the census's coverage
    without shrinking its reported cell count.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # A BOUNDED, COUNTED TRANSPORT RETRY WITH A TIMEOUT. A census streams dozens of multi-gigabyte
    # checkpoints and `snapshot_download` will eventually hang rather than fail: the re-census wedged
    # mid-"Fetching 8 files" on a 7B shard and sat with a live instance and a frozen log for five
    # hours, until the reaper. A stall is not an exception, so a retry alone does not help — the
    # per-file timeout (set once in hf_setup) is what turns the hang into a retryable error
    # (M-D97). This is the same
    # system-boundary carve-out `mercator_rowgram` documents; nothing scientific is caught.
    from infra.pg.mercator_core import retry_transport
    d, _n_retry = retry_transport(
        lambda: snapshot_download(repo, revision=sha,
                                  allow_patterns=["*.safetensors", "*.safetensors.index.json"]),
        what=f"snapshot_download({repo})")
    shards = sorted(pathlib.Path(d).glob("*.safetensors"))
    assert shards, f"{repo}: no safetensors shards under {d}"
    index = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                index[k] = shard
    out = {}
    for t in types:
        for L in layers:
            key = resolve_key(index, t, L)
            with safe_open(str(index[key]), framework="pt") as f:
                out[(t, L)] = f.get_tensor(key).to(torch.float32)
    return out


def census_grams(base_repo, base_sha, accepted, shas, types, layers, delete_cache=None):
    """G[(type, layer)] = Σ dW_i^T dW_i / ‖dW_i‖²_F over the accepted derivative cloud, fp64.

    Same estimand and same normalisation as HUMBOLDT's PG1 Gram, so a census number and a house
    number mean the same thing. Streams one checkpoint at a time and deletes it.
    """
    import gc
    import numpy as np
    import torch

    keys = [(t, L) for t in types for L in layers]
    base = load_decoder_matrices(base_repo, base_sha, types, layers)
    n_by = {k: int(base[k].shape[1]) for k in keys}
    G = {k: np.zeros((n_by[k], n_by[k]), dtype=np.float64) for k in keys}
    fro2_by = {k: {} for k in keys}
    n_terms = {k: 0 for k in keys}

    dropped = []
    for i, rec in enumerate(accepted):
        repo = rec["id"]
        print(f"[census] {repo} ({i + 1}/{len(accepted)})", flush=True)
        ft = load_decoder_matrices(repo, shas[repo], types, layers)
        # A shape mismatch here means the config prefilter could not see what the tensors are —
        # a bit-packed quantisation the config did not declare, or a reshaped head. Dropping the
        # DERIVATIVE and recording why is the honest response: aborting throws away every other
        # derivative already streamed, while silently including it would corrupt the Gram. The
        # surviving count is re-checked against MIN_FTS_FOR_CENSUS by the caller, so a roster
        # thinned below the threshold becomes a recorded census-impossible rather than a quiet
        # small-n result (defect D11).
        bad = [k for k in keys if tuple(ft[k].shape) != tuple(base[k].shape)]
        if bad:
            why = (f"tensor shape disagrees with base at {bad[0]}: "
                   f"{tuple(ft[bad[0]].shape)} != {tuple(base[bad[0]].shape)} "
                   f"(undeclared packing/quantisation; {len(bad)}/{len(keys)} cells affected)")
            print(f"[census] DROP {repo}: {why}", flush=True)
            dropped.append({"id": repo, "reason": why})
            del ft
            gc.collect()
            if delete_cache is not None:
                delete_cache(repo)
            continue
        # A SINGLE NON-FINITE DERIVATIVE POISONS EVERY CELL IT TOUCHES. The Gram is a running sum,
        # so one NaN or inf in one delta makes the whole accumulator NaN and the failure surfaces
        # much later, in whatever first tries to use the eigenvectors — the re-census died in
        # `complement_subspace` with 65504 of 65536 basis entries non-finite, which names the
        # symptom and not the repo (defect M-D60). Checked here, where the offending model can be
        # named and dropped like any other rejection.
        bad_cells = [k for k in keys
                     if not torch.isfinite(ft[k].to(torch.float64)).all()]
        if bad_cells:
            why = (f"non-finite weights in {len(bad_cells)}/{len(keys)} cells "
                   f"(e.g. {bad_cells[0]}) — a NaN delta would poison every Gram it enters")
            print(f"[census] DROP {repo}: {why}", flush=True)
            dropped.append({"id": repo, "reason": why})
            del ft
            gc.collect()
            if delete_cache is not None:
                delete_cache(repo)
            continue
        for k in keys:
            A = (ft[k].to(torch.float64) - base[k].to(torch.float64)).numpy()
            fro2 = float((A ** 2).sum())
            if fro2 == 0:
                fro2_by[k][repo] = 0.0          # a frozen matrix: recorded, contributes nothing
                continue
            assert np.isfinite(fro2), f"{repo} {k}: non-finite Frobenius norm {fro2}"
            G[k] += (A.T @ A) / fro2
            fro2_by[k][repo] = fro2
            n_terms[k] += 1
            del A
        del ft
        gc.collect()
        if delete_cache is not None:
            delete_cache(repo)
    return G, base, fro2_by, n_terms, dropped


def spikiness(U) -> dict:
    """Coordinate spikiness of a bank: leverage participation ratio against its Haar reference.

    The matched-ROTATION null and the Haar null coincide for this statistic — rotating a fixed
    subspace by a Haar orthogonal matrix yields a Haar-distributed subspace — so the Haar band is
    the matched null here, and the row says so rather than leaving a reader to wonder which was
    used.
    """
    from infra.pg import humboldt_core as H
    lev = H.leverage(U)
    n, k = U.shape
    return {"participation_ratio": H.participation_ratio(lev),
            "flat_reference": float(n), "haar_expected_leverage": k / n,
            "max_leverage": float(lev.max()), "n": int(n), "k": int(k),
            "null": "matched-rotation == Haar for bank leverage"}
