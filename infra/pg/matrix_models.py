"""MENDELEEV row registry, loaders and binds — contract §3/§5 (infra/pg/MATRIX_CONTRACT.md); spec §0.3
B1–B4, §3.1 (studies/lineage/cross-lineage/research/2026-09-02_mendeleev_spec.md).

ROWS   every row of §3.1 → before/after repo ids (VERIFIED against the HF API on 2026-09-03, metadata only:
       revision sha + config.json fields are in KNOWN; a repo the token cannot read carries
       `access: "403 licence"` and `load_pair` fails loud with that reason), shell, family, floors, the O
       instrument, remote-code / transformers pins, GPU memory for one copy (`gb_one`, the behaviour job
       loads SEQUENTIALLY) and for two copies at batch 4 (`gb_pair`, the graft job), and `s3_cached`
       (True for the repos the weight cache holds — see S3_CACHED, whose authority is the BUCKET
       LISTING, not the tuple; everything else downloads from HF in-job via `hf_files`).
load_pair(row, device, which)  which ∈ {"before", "after", "before-text", "after-pg1"} (str → one member,
       tuple → dict of members). Install rows (1a, 2, 2-10B, 3b) build `before` INSIDE the after-model's
       shell: decoder layers + final norm reset to the base LM, embed/head KEPT from the shell (the C1 /
       CARREL convention of `contrast_a1._lm_swap_state_dict`, anatomy-driven here; Qwen3 → Molmo2 via
       `family_resolvers.qwen3_to_molmo2`). `before-text` = the healthy LM text-only (the §6.5 parity
       member). Policy rows: PG1 shell + `lacuna_shell.build_row_policy` (own head primary, `after-pg1`
       = PG1's table, secondary). Row 5c: gemma-3-4b-pt + the HELMHOLTZ-2 permuted-label adapter MERGED.
snapshot_lm / binds   B1/B3/B4: tokenizer vocab sha256 + shared-prefix, config / safetensors-index
       sha256, per-layer text-decoder ‖ΔW‖/‖W‖ (vision / action expert / MTP excluded by anatomy — only
       `lm_anatomy(model)["layers"]` + final norm + embed + head are compared), embed/head delta on the
       shared vocab rows, `frozen_lm`. Sequential: `snapshot_lm(member, dir)` while `before` is resident,
       `binds(snapshot, after_member)` when `after` is. On an INSTALL row the embed / head / tokenizer
       comparisons are structurally void (`before` is the after-model's own shell) and are reported as
       None beside the real `vs_base` numbers, never as the 0 they would otherwise be. Every `binds` also
       returns `kind_delta`: the row's declared `kind` checked against the movement just measured
       (BENIGN_MAX_*), asserted here and at import over BANKED_DELTA.

Fail-loud; no try/except; heavy imports inside functions (py_compiles without torch/transformers).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

S3_BUCKET = "pg-lineage-769063704696"
H2_ADAPTER_S3 = f"s3://{S3_BUCKET}/results/h2_ladder_skill/out/skill_r2/adapter.pt"
TRANSFORMERS_PIN = "5.9.0"

# ------------------------------------------------------------------ the machine-readable repo facts
# `matrix_repo_facts.json` is a SNAPSHOT OF THE SOURCES, not a second opinion about them: for every repo
# in KNOWN it holds that repo's own `config.json` and `model.safetensors.index.json` fields, read at the
# pinned revision from the S3 weight cache where the bucket holds the repo and from the HF resolve
# endpoint otherwise, plus the bucket's own `weights/hf/hub/` listing. `--refresh-facts` regenerates it;
# NOTHING in it is ever typed by hand.
#
# WHY IT EXISTS. Every registry defect of 2026-09-03/04 has one shape: a value typed here that a
# machine-readable source already knew, with nothing comparing the two. `transformers_pin` 4.57.1 against
# MolmoAct2's own config (5.3.0) cost row 3c-Act 13 cells, twice. `params_b` 4.5 / 4.7 against the banked
# safetensors index totals (4.851 / 5.442) put a wrong number in every artifact's manifest. `S3_CACHED`
# sat at five repos while the bucket held eighteen, so 116 GB of seeded weights went unused. The fix is
# not to retype them more carefully; it is `infra/pg/matrix_audit.py`, which compares every derived field
# against this file and FAILS LOUD, and runs inside `--dryrun`, which gates every launch.
_FACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix_repo_facts.json")
with open(_FACTS_PATH) as _fh:
    FACTS = json.load(_fh)
FACTS_META = FACTS.pop("_meta")

# ------------------------------------------------- kind <-> measured weight movement (spec confound 33)
# A row's `kind` is a CLAIM about how far the fine-tune moved the weights. `binds()['per_layer'][*]
# ['rel_delta']` MEASURES it. Until now the claim and the measurement never met: ctrl-g1 was found
# mislabelled "benign/pt" at ~15x the benign reference, the response was a CODE COMMENT, the row kept its
# kind, and no rel_delta was read by anything anywhere in the repo. Row 3a then repeated the same defect
# on the Qwen family's no-damage control. So the two are now bound, in one rule with two enforcement
# points: `_check_banked_kinds()` runs at IMPORT over every banked measurement (BANKED_DELTA), and
# `binds()` runs the same check on a NEW measurement the moment it is computed.
#
# THE BAND, and exactly how it was chosen. Two banked anchors, both read from
# `studies/lineage/cross-lineage/results/mdl_metrics/<cell>/metrics.json`, key `binds`:
#     ctrl-g2  gemma-2-2b -> gemma-2-2b-it              decoder 0.048972, embed/head 0.032600   <- benign
#     3a       Qwen3-4B-Base -> Qwen3-4B-Instruct-2507  decoder 0.158591, embed/head 0.204083   <- not
# The cut is the GEOMETRIC MEAN of those two, per axis. These are ratios, so the log scale is the natural
# one, and the geometric mean is the point equidistant from the pass case and the fail case on it: the
# data fixes it, not taste. It admits ctrl-g2 with 1.80x (decoder) / 2.50x (embed) headroom and rejects 3a
# by the same two factors. `_dryrun` recomputes both constants from the anchors, so they cannot drift
# away from this derivation.
# WHAT IT WOULD HAVE CAUGHT, had it existed: 3a, at import, at 3.24x the benign decoder movement and
# 6.26x the benign embed movement (its embed moves FURTHER than the checkpoint-change control ctrl-g1's
# 0.1740); and ctrl-g1, at import, at 8.14x / 2.13x over the band. Both were labelled "benign/pt", and
# 3a is the control the §3.1 `3b - 3a` contrast is scored against.
# WHAT IT CANNOT DO, stated: the band is ONE-SIDED and gates the `benign` kind only. A SMALL delta is not
# evidence of benignity — row 5c moves the decoder 0.0145 and destroys MMLU (0.656 -> 0.263, H-2
# skill_r2). This can refute a benign label; it can never confirm one.
BENIGN_MAX_DECODER = 0.0881          # sqrt(0.048972 * 0.158591) = 0.088128
BENIGN_MAX_EMBED = 0.0816            # sqrt(0.032600 * 0.204083) = 0.081567
BANKED_DELTA = {   # row: (mean per-layer decoder rel_delta, embed/head rel_delta or None, source cell)
    "ctrl-g2": (0.048972, 0.032600, "ctrl-g2_S"),
    "ctrl-g1": (0.717512, 0.174031, "ctrl-g1_S"),
    "3a": (0.158591, 0.204083, "3a_S"),
    "5c": (0.014469, 0.000000, "5c_S"),      # embed 0 is REAL here: the H-2 LoRA adapts Linears only
    "2": (0.267343, None, "2_beh_smoke"),    # install row: embed/head structurally undefined (see binds)
}
# The weight cache, DERIVED from the bucket's own `weights/hf/hub/` listing (banked in the facts file by
# `--refresh-facts`; `FACTS_META["bucket_listed"]` says when). This module stays import-light on purpose —
# every weight-free launcher and every job on the GPU box imports it and it must make no network call at
# import — so the listing is banked rather than fetched, and re-read from the live bucket AT USE TIME by
# the two lanes that spend it: `scripts/matrix_launch.py:verify_s3_cache` (AWS, on every `fill`/`go`) and
# `scripts/matrix_vast.py:weights_plan` (Vast, on every plan). Both fail loud on a claim the bucket does
# not hold.
#
# It used to be a hand-written tuple, and it sat stale at five repos while the bucket held eighteen:
# harmless for launching (an absent dir just falls back to the hub) but `s3_cached` is copied into every
# job manifest as provenance, so a stale entry made the manifest record something false, and 116 GB of
# seeded weights were re-downloaded from the hub instead of being used. There is now nothing to keep in
# sync: `aws s3 ls` is the only writer of this name.
S3_CACHED = tuple(sorted(d[len("models--"):].replace("--", "/", 1)
                         for d in FACTS_META["bucket_dirs"]))
HF_ALLOW = ["*.safetensors", "*.json", "*.py", "*.model", "*.txt", "*.tiktoken", "*.jinja"]

# HF API, 2026-09-03 (model_info sha + config.json; NO weights). (sha, architecture, text model_type,
# n_layers, hidden, vocab, tie_word_embeddings, released dtype, remote_code, access).
_K = ("sha", "arch", "model_type", "n_layers", "hidden", "vocab", "tie", "dtype", "remote_code", "access")
KNOWN = {k: dict(zip(_K, v)) for k, v in {
    "google/gemma-2b": ("9cf48e52b224239de00d483ec8eb84fb8d0f3a3a", "GemmaForCausalLM", "gemma", 18, 2048, 256000, True, "bfloat16", False, "ok"),
    "google/gemma-1.1-2b-it": ("d750f5eceb83e978c09e2b3597c2a8784e381022", "GemmaForCausalLM", "gemma", 18, 2048, 256000, True, "bfloat16", False, "ok"),
    "google/paligemma-3b-pt-224": ("35e4f46485b4d07967e7e9935bc3786aad50687c", "PaliGemmaForConditionalGeneration", "gemma", 18, 2048, 257216, True, "float32", False, "ok"),
    "lerobot/pi0_base": ("25c379b52ba2ff8788cab921758a3cc3fe3f77f2", "LeRobot PI0Policy (safetensors)", "gemma", 18, 2048, 257152, True, "float32", False, "ok"),
    "lerobot/pi05_base": ("b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba", "LeRobot PI05Policy (safetensors)", "gemma", 18, 2048, 257152, True, "float32", False, "ok"),
    "google/gemma-2-2b": ("c5ebcd40d208330abc697524c919956e692655cf", "Gemma2ForCausalLM", "gemma2", 26, 2304, 256000, True, "float32", False, "ok"),
    "google/gemma-2-2b-it": ("299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8", "Gemma2ForCausalLM", "gemma2", 26, 2304, 256000, True, "bfloat16", False, "ok"),
    "google/paligemma2-3b-pt-224": ("96eeb174da13ca1a2b247e4d0867436296c36420", "PaliGemmaForConditionalGeneration", "gemma2", 26, 2304, 257216, True, "bfloat16", False, "ok"),
    "google/paligemma2-3b-mix-224": ("8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78", "PaliGemmaForConditionalGeneration", "gemma2", 26, 2304, 257216, True, "bfloat16", False, "ok"),
    "google/gemma-2-9b": ("33c193028431c2fde6c6e51f29e6f17b60cbfac6", "Gemma2ForCausalLM", "gemma2", 42, 3584, 256000, True, "float32", False, "ok"),
    "google/paligemma2-10b-pt-224": ("86f23f048669c9479367fcbf06c76727c2f7dc6f", "PaliGemmaForConditionalGeneration", "gemma2", 42, 3584, 257216, True, "bfloat16", False, "ok"),
    "Qwen/Qwen3-4B-Base": ("906bfd4b4dc7f14ee4320094d8b41684abff8539", "Qwen3ForCausalLM", "qwen3", 36, 2560, 151936, True, "bfloat16", False, "ok"),
    "Qwen/Qwen3-4B-Instruct-2507": ("cdbee75f17c01a7cc42f958dc650907174af0554", "Qwen3ForCausalLM", "qwen3", 36, 2560, 151936, True, "bfloat16", False, "ok"),
    "allenai/Molmo2-4B": ("042abfa7a38879a376cec03d949eff0aefaa0600", "Molmo2ForConditionalGeneration", "molmo2", 36, 2560, 151936, False, "float32", True, "ok"),
    "allenai/Molmo2-ER": ("dab22564403d2607855bb1fffb0721285b445081", "Molmo2ForConditionalGeneration", "molmo2", 36, 2560, 151936, False, "float32", True, "ok"),
    "allenai/MolmoAct2": ("e432d85f6e039edca44afb93c262f3084ab72a9c", "MolmoAct2ForConditionalGeneration", "molmoact2", 36, 2560, 154624, False, "float32", True, "ok"),
    "allenai/MolmoAct2-LIBERO-LeRobot": ("f0c5a9567c2b72faadec901e16e055c8b098c2f5", "LeRobot policy (safetensors, no config.json)", None, None, None, None, None, None, False, "ok"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": ("7872f01b1d1fe23eabc4c98b48bffcef5a386062", "DeepseekV4ForCausalLM", "deepseek_v4", 43, 4096, 129280, False, "bfloat16 + fp8 quant (e4m3, block 128) + fp4 experts", False, "ok"),
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": ("6821d6ad3681a4b137b066b76094fa82ebd0a380", "DeepseekV4ForCausalLM", "deepseek_v4", 43, 4096, 129280, False, "bfloat16 + fp8 quant (e4m3, block 128) + fp4 experts", False, "ok"),
    "google/gemma-3-4b-pt": ("cc012e0a6d0787b4adcc0fa2c4da74402494554d", "Gemma3ForConditionalGeneration", "gemma3", 34, 2560, 262208, True, "bfloat16", False, "ok"),
    "google/gemma-3-4b-it": ("093f9f388b31de276ce2de164bdc2081324b9767", "Gemma3ForConditionalGeneration", "gemma3", 34, 2560, 262208, True, "bfloat16", False, "ok"),
    "google/medgemma-4b-pt": ("1b7b4a7e462da7ab2bc40591ec86ca449edc388a", None, None, None, None, None, None, None, False, "403 licence"),
    "google/gemma-4-31B": ("5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89", "Gemma4ForConditionalGeneration", "gemma4", 60, 5376, 262144, True, "bfloat16", False, "ok"),
    "google/gemma-4-31B-it": ("842da3794eaa0b77d5f08bae87a17459d91ff475", "Gemma4ForConditionalGeneration", "gemma4", 60, 5376, 262144, True, "bfloat16", False, "ok"),
    "HuggingFaceH4/zephyr-7b-beta": ("892b3d7a7b1cf10c7a701c60881cd93df615734c", "MistralForCausalLM", "mistral", 32, 4096, 32000, False, "bfloat16", False, "ok"),
    "cais/Zephyr_RMU": ("70c55b3bf3141a8c24292dec0262b8aea03a0d4a", "MistralForCausalLM", "mistral", 32, 4096, 32000, False, "bfloat16", False, "ok"),
    "meta-llama/Meta-Llama-3-8B-Instruct": ("8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", None, None, None, None, None, None, None, False, "403 licence"),
    "ScaleAI/mhj-llama3-8b-rmu": ("aa68f887c36072c057f19446d474ce5389b79fdf", "LlamaForCausalLM", "llama", 32, 4096, 128256, False, "bfloat16", False, "ok"),
    # INSTRUMENTS E6b (2026-11-04): config.json read with curl from the HF resolve endpoint at this sha (ungated; no token); not in the S3 weight cache,
    # so jobs download it in-job (neurons_harvest.download). Released in float32 (29.19 GB by its safetensors index); the jobs load it in bfloat16.
    "allenai/OLMo-2-1124-7B": ("7df9a82518afdecae4e8c026b27adccc8c1f0032", "Olmo2ForCausalLM", "olmo2", 32, 4096, 100352, False, "float32", False, "ok"),
}.items()}
# gemma-3-4b: config.json nests no vocab_size (262 208 is the tokenizer's, from the Gemma-3 text config default).
# DeepSeek-V4: MTP layers 1 (0731) vs 3 (Vision-Exp); hc_mult 4; 256 routed + 1 shared experts; the Vision-Exp
# config carries vision_* keys under the SAME CausalLM architecture (B2). medgemma / Llama-3-8B-Instruct: the
# HF token returns 403 on config.json (licence not accepted on the account) — rows 5b / 7b blocked until it is.


def _row(before, after, shell, family, kind, O, params_b, install=False, policy=None, attn="eager",
         pin=TRANSFORMERS_PIN, note=None, adapter=None, gated_on=None):
    # `needs_remote_code` is DERIVED, never declared: a repo needs `trust_remote_code` exactly when its
    # own config.json carries an `auto_map`. It was a hand-passed `remote=True` on three rows and it
    # happened to be right; the next row would have had to get it right by hand too. (The values are
    # unchanged — `matrix_audit` asserts the derivation reproduces exactly rows 3b / 3c-ER / 3c-Act.)
    remote = any(FACTS[r]["config"] is not None and FACTS[r]["config"]["auto_map"] for r in (before, after))
    # `floors` used to be here as a dead eighth copy of the scoring floors with a key ("letter") that does
    # not exist in the live table. It had ZERO consumers — `matrix_behaviour` stripped it out of the
    # manifest at :921 — and two definitions of one concept, one of them dead, is how a future edit lands
    # in the wrong place and changes nothing while looking correct. The floors that score every verdict
    # live in `scripts/matrix_verdicts.FLOORS`, and `matrix_audit` now compares that table against the
    # job's own `matrix_behaviour.FLOOR` so the remaining two copies cannot drift.
    gb_one = round(2 * params_b * 1.1 + 8, 1)          # bf16 weights + activations/workspace at the probed batch
    gb_pair = round(4 * params_b * 1.1 + 8, 1)         # two resident copies at batch 4 (graft job)
    return {"before": before, "after": after, "shell": shell, "family": family, "kind": kind, "O": O,
            "install": install, "policy": policy, "attn_impl": attn,
            "needs_remote_code": remote, "transformers_pin": pin, "params_b": params_b,
            "gb_one": gb_one, "gb_pair": gb_pair,
            "s3_cached": {r: (r in S3_CACHED) for r in (before, after)},
            "access": {r: KNOWN[r]["access"] for r in (before, after)},
            "adapter": adapter, "gated_on": gated_on, "note": note}


MOLMO_PIN = "4.57.1"
MOLMO_PIN_NOTE = ("model-card pin transformers==4.57.1 (ROW3 verified the remote code does NOT run on 5.9.0: "
                  "create_causal_mask signature, processor kwargs) — §8b response: ensure_transformers_pin builds "
                  "/opt/pg/venv-tf4.57.1 and re-runs the job under it; manifest.harness_shift records it")

# MolmoAct2 is NOT on the Molmo2 pin, and assuming it was cost 13 cells twice. Read from the banked
# weights rather than guessed: `weights/hf/hub/models--allenai--MolmoAct2/snapshots/e432d85f.../config.json`
# declares `transformers_version: 5.3.0`, while Molmo2-4B declares 4.57.1 and Molmo2-ER 4.56.2. Running
# its major-5 tokenizer config under 4.57.1 fails as
# `AttributeError: 'list' object has no attribute 'keys'` in `SPECIAL_TOKENS_ATTRIBUTES + list(
# special_tokens.keys())` — an error that names neither the version nor the model. The AMI already runs
# 5.9.0, same major as declared, so this pin needs no venv build and no re-exec; if 5.9.0 ever proves too
# far ahead, pin the declared 5.3.0 exactly.
# params_b for the Molmo rows is read from the banked safetensors index `total_size` (all-F32), not from
# a model card: Molmo2-4B / Molmo2-ER = 19.403 GB -> 4.851 B, MolmoAct2 = 21.769 GB -> 5.442 B. The
# registry had 4.5 and 4.7, out by 7.8 % and 15.8 %. Nothing computes on this field, but it is copied
# into every metrics.json manifest as published provenance, so it was wrong in every banked artifact.

MOLMOACT_PIN = "5.9.0"
MOLMOACT_PIN_NOTE = ("MolmoAct2 declares transformers 5.3.0 in its own config.json (its Molmo2 siblings "
                     "declare 4.56/4.57); it runs on the AMI's 5.9.0, same major, with no re-exec.")
ROWS = {
    "1a": _row("google/gemma-2b", "google/paligemma-3b-pt-224", "paligemma", "gemma", "install/pt", "e4 battery (RefCOCO, TextVQA) + VQAv2 cand", 3.0, install=True),
    "1b-pi0": _row("google/paligemma-3b-pt-224", "lerobot/pi0_base", "pi0-pg1", "gemma", "policy", "LIBERO (not this job)", 3.0, policy="pi0"),
    "1b-pi05": _row("google/paligemma-3b-pt-224", "lerobot/pi05_base", "pi0-pg1", "gemma", "policy", "LIBERO (not this job)", 3.0, policy="pi05"),
    "2": _row("google/gemma-2-2b", "google/paligemma2-3b-pt-224", "paligemma", "gemma2", "install/pt", "e4 battery + VQAv2 cand", 3.0, install=True),
    "2-10B": _row("google/gemma-2-9b", "google/paligemma2-10b-pt-224", "paligemma", "gemma2", "install/pt", "e4 battery + VQAv2 cand (DONE C17; behaviour-only row)", 9.7, install=True),
    # 3a is NOT a benign tune. The label follows the MEASUREMENT, not the release note: banked `binds`
    # (3a_S) puts Base -> Instruct-2507 at decoder 0.1586 / embed 0.2041, i.e. 3.24x and 6.26x the benign
    # reference ctrl-g2, with an embed delta larger than the checkpoint-change control ctrl-g1's. 2507 is
    # a later, separately post-trained release, not a light instruction tune of the Base checkpoint it is
    # paired with here. It remains the parent hop of the §3.1 `3b - 3a` contrast; it is NOT the Qwen
    # family's no-damage control, and "benign/pt" asserted that it was.
    "3a": _row("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B-Instruct-2507", "text", "qwen3", "post-train/pt", "—", 4.0, attn="sdpa",
               note="kind measured, not assumed: decoder 0.1586 / embed 0.2041 (3a_S binds) vs ctrl-g2 0.0490 / 0.0326"),
    "3b": _row("Qwen/Qwen3-4B-Instruct-2507", "allenai/Molmo2-4B", "molmo2", "qwen3", "install/post", "N14 vlm_skill_eval", 4.851, install=True, attn="sdpa", pin=MOLMO_PIN, note=MOLMO_PIN_NOTE),
    "3c-ER": _row("allenai/Molmo2-4B", "allenai/Molmo2-ER", "molmo2", "qwen3", "install-refine/post", "N14 vlm_skill_eval", 4.851, attn="sdpa", pin=MOLMO_PIN, note=MOLMO_PIN_NOTE),
    "3c-Act": _row("allenai/Molmo2-ER", "allenai/MolmoAct2", "molmo2", "qwen3", "policy", "N15 LIBERO (not this job)", 5.442, attn="sdpa", pin=MOLMOACT_PIN, note=MOLMOACT_PIN_NOTE),
    "4": _row("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "text", "deepseek_v4", "install/post (MoE, hc_mult 4)", "N14 (gated)", 305.0, attn="sdpa", gated_on="S9-pre",
               note="fp8/fp4 released precision: one copy ≥ 180 GB on a Blackwell GPU; --load-test-only only"),
    "5": _row("google/gemma-3-4b-pt", "google/gemma-3-4b-it", "text", "gemma3", "benign/pt", "—", 4.3),
    "5b": _row("google/gemma-3-4b-pt", "google/medgemma-4b-pt", "text", "gemma3", "domain install/pt", "— (medical skill not measured)", 4.3),
    "5c": _row("google/gemma-3-4b-pt", "google/gemma-3-4b-pt", "text", "gemma3", "synthetic content destruction (C11)", "—", 4.3,
               adapter={"s3": H2_ADAPTER_S3, "run": "h2_ladder_skill/skill_r2", "rank": 16, "alpha": 16.0,
                        "recipe": "permuted-label MMLU auxiliary_train, n=16000, 6 epochs, lr 2e-4, batch 8, seed 0 (its metrics.json)",
                        "banked": "H-2 skill_r2: MCQ 0.656→0.263, cloze 0.509→0.251 on n=570 (metrics.json, not a baseline)"}),
    "6": _row("google/gemma-4-31B", "google/gemma-4-31B-it", "text", "gemma4", "benign/pt", "—", 31.0, note="one H100 80 GB; sequential only"),
    "7a": _row("HuggingFaceH4/zephyr-7b-beta", "cais/Zephyr_RMU", "text", "mistral", "unlearn/post", "MMLU retain (S/K/P on WMDP)", 7.2, attn="sdpa"),
    "7b": _row("meta-llama/Meta-Llama-3-8B-Instruct", "ScaleAI/mhj-llama3-8b-rmu", "text", "llama", "unlearn/post", "MMLU retain (S/K/P on WMDP)", 8.0, attn="sdpa"),
    "ctrl-g2": _row("google/gemma-2-2b", "google/gemma-2-2b-it", "text", "gemma2", "benign/pt", "—", 2.6),
    # ctrl-g1 is a CHECKPOINT-CHANGE control: gemma-1.1 is a separate training run, decoder 0.7175 (max
    # tensor 0.854) against 0.0490 for the real benign tune ctrl-g2. It carried "benign/pt" with only a
    # comment saying so; the label is now the measurement's, and BENIGN_MAX_* enforces it.
    "ctrl-g1": _row("google/gemma-2b", "google/gemma-1.1-2b-it", "text", "gemma", "checkpoint-change/pt", "—", 2.5,
                    note="NOT the benign baseline for install rows — ctrl-g2 is (decoder 0.7175 vs 0.0490, ctrl-g1_S binds)"),
}
SHELL_OF_PARITY = {"paligemma", "molmo2"}          # install rows whose healthy LM has a text-only member


def kind_delta_verdict(row: str, decoder_rel: float, embed_rel=None) -> dict:
    """Does the row's DECLARED kind survive a MEASURED weight movement? (See the BENIGN_MAX_* block.)

    `decoder_rel` = mean per-layer ‖ΔW‖/‖W‖ over the text decoder; `embed_rel` = the embed/head delta on
    the shared vocab rows, or None where that comparison does not exist (install rows — see `binds`).
    Only a `benign` kind is a smallness claim, so only a `benign` kind is gated."""
    kind = ROWS[row]["kind"]
    gated = kind.startswith("benign")
    over = []
    if gated:
        if decoder_rel > BENIGN_MAX_DECODER:
            over.append(f"decoder {decoder_rel:.4f} > {BENIGN_MAX_DECODER} ({decoder_rel / BENIGN_MAX_DECODER:.2f}x)")
        if embed_rel is not None and embed_rel > BENIGN_MAX_EMBED:
            over.append(f"embed/head {embed_rel:.4f} > {BENIGN_MAX_EMBED} ({embed_rel / BENIGN_MAX_EMBED:.2f}x)")
    return {"row": row, "kind": kind, "gated": gated, "decoder_rel": decoder_rel, "embed_rel": embed_rel,
            "band": {"decoder": BENIGN_MAX_DECODER, "embed": BENIGN_MAX_EMBED},
            "consistent": not over,
            "reason": ("not gated: only a `benign` kind claims the weights barely moved" if not gated else
                       "within the benign band" if not over else
                       f"row {row} declares kind={kind!r} but moved " + " and ".join(over) +
                       f" — the benign reference ctrl-g2 is {BANKED_DELTA['ctrl-g2'][0]} / "
                       f"{BANKED_DELTA['ctrl-g2'][1]}. RELABEL THE ROW: the label is the defect, not the "
                       "measurement, and a mislabelled row is a no-damage control that moved.")}


def assert_kind_delta(row: str, decoder_rel: float, embed_rel=None) -> dict:
    v = kind_delta_verdict(row, decoder_rel, embed_rel)
    assert v["consistent"], v["reason"]
    return v


def _check_banked_kinds():
    """IMPORT-TIME gate: every banked measurement against the kind its row declares TODAY. An agent that
    banks a number and pastes it in without touching the label cannot get this module to import — which
    is the enforcement ctrl-g1's comment lacked and 3a then slipped through."""
    for row, (dec, emb, cell) in BANKED_DELTA.items():
        assert row in ROWS, f"BANKED_DELTA names row {row!r}, which is not in ROWS"
        v = kind_delta_verdict(row, dec, emb)
        assert v["consistent"], f"{v['reason']} [banked: mdl_metrics/{cell}/metrics.json, key `binds`]"


_check_banked_kinds()


def row_repos(row: str) -> list:
    r = ROWS[row]
    return sorted({r["before"], r["after"]})


def members(row: str) -> list:
    """The `which` values a behaviour job runs for this row."""
    r = ROWS[row]
    out = ["before", "after"]
    if r["install"] and r["shell"] in SHELL_OF_PARITY:
        out.insert(0, "before-text")
    return out


# resident member -> the key its EXTRA text-shell reading is banked under (`matrix_behaviour --after-text`).
# `before-text` is already taken by the row's bare base LM (a DIFFERENT object: the base model in its own
# architecture), so the shell's own base-decoder reading needs its own name.
TEXT_MEMBER_OF = {"before": "before-shell-text", "after": "after-text"}


def text_members(row: str) -> dict:
    """OPT-IN (`matrix_behaviour --after-text`): {resident member -> banked key} for the extra TEXT-SHELL
    re-readings of this row's members. `{}` where the row has none — a STATEMENT, never a gap.

    WHY (2026-09-04). `members()` loads `before-text` -> `before` -> `after` on an install row, so the two
    arms of the install contrast are read through DIFFERENT forwards: the base LM bare, the shell (base
    decoder and fine-tuned decoder alike) with a blank image. That asymmetry is what `k_shell_parity`
    keeps stamping VOID-shell. Re-reading the two RESIDENT shell members through a plain text forward
    gives a contrast with no image on either side:
      `before-shell-text` vs `after-text`  — same architecture, same processor's tokenizer, same call;
                                             the ONLY difference is the decoder weights (the install).
      `before-text` vs `before-shell-text` — the same base decoder read in its own architecture and read
                                             inside the shell text-only: the VALIDITY control for the
                                             pair above, scored by the frozen §6.5 rule itself
                                             (`matrix_core.k_shell_parity` / `ppl_forward_gate`). It is
                                             not decoration: the shell read text-only put row 2's
                                             `before` ppl at 18558.6 against 7.966 bare, so the text
                                             forward of this wrapper is a suspect instrument until this
                                             control says otherwise.

    The eligibility question is asked of the REGISTRY, not of a list kept here: a row has these readings
    exactly when it has a `before-text` parity member (install into a VLM shell whose healthy LM can be
    read as text). Every other row is skipped for a reason it can state: a text row's `after` member IS
    already read through the text shell (`shell == "text"`), so the re-read would duplicate a banked
    column bit for bit; a policy row's `after` is a π0 backbone in a PG1 shell with no text reading at
    all. A hand-maintained tuple that duplicated this registry (`matrix_behaviour.POLICY_ROWS`) cost 19
    cells on 2026-09-03 — that mistake is not repeated here."""
    return dict(TEXT_MEMBER_OF) if "before-text" in members(row) else {}


# --------------------------------------------------------------------- the O instrument, machine-readable
# `ROWS[row]["O"]` is prose for a human reader. `o_tasks(row)` is the SAME statement as concrete task names
# (the `matrix_items.VISION` / `vlm_skill_eval.TASKS` keys), so a job derives its O columns from ONE place
# instead of re-deriving them from that prose or hard-coding a list of its own (spec §1.2 O, §3.1):
#   PaliGemma-shell rows 1a / 2 / 2-10B  the e4 battery (`matrix_behaviour._o_column`: RefCOCO acc@0.5,
#       TextVQA ANLS+EM, VQAv2 EM) and, beside it, the 4-way candidate-likelihood column of each VQA task.
#   Molmo2-shell rows 3b / 3c-ER         the same three tasks through N14 `infra/pg/vlm_skill_eval.py`.
#   Policy rows 1b-pi0 / 1b-pi05 / 3c-Act   () — their O is LIBERO success, measured by a SEPARATE job.
#   Unlearning rows 7a / 7b              () — their O is the RETAINED MMLU, which IS the S/K columns a
#       recovery/behaviour job already reads on the MMLU eval halves; there is no extra own-skill readout.
#   Text rows 3a / 5 / 5b / 5c / 6 / ctrl-g1 / ctrl-g2   () — no own skill outside the LM.
#   Row 4 (DeepSeek-Vision)              () — its N14 O is gated on S9-pre and has no Forward shell yet (B2).
# RefCOCO is deliberately NOT in the re-measured set: it exists only as a generation readout (no candidate
# form), 200 greedy generations per model per dose is not affordable in a recovery ladder, and no verdict
# reads it. It stays a behaviour-cell column, measured once per row. An empty list is a STATEMENT ("this
# row has no own-skill O here"), never a gap.
O_TASKS = ("vqav2", "textvqa")
O_LIBERO_NOTE = "not measured here (LIBERO success is a separate job)"
O_RETAINED_MMLU_NOTE = "not a separate readout (O := the retained MMLU = the S/K columns)"


def o_tasks(row: str) -> list:
    """The row's O instrument as concrete task names — see the table above. `[]` where O is not a
    per-item vision readout of this pair (policy rows, unlearning rows, text rows, the gated row 4)."""
    r = ROWS[row]
    if r["kind"] == "policy" or r["policy"] or r["shell"] not in SHELL_OF_PARITY:
        return []
    return list(O_TASKS)


# ------------------------------------------------------------------------------------------ files + hashes
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_files(repo: str, revision: str | None = None) -> str:
    """Local snapshot dir: safetensors + configs + tokenizer (+ remote code) ONLY; downloads what is absent
    (HF_HUB_DISABLE_XET=1 is set by the job). The pinned KNOWN sha is used when `revision` is None."""
    from huggingface_hub import snapshot_download
    assert KNOWN[repo]["access"] == "ok", f"{repo}: {KNOWN[repo]['access']} — not loadable with this token"
    return snapshot_download(repo, revision=revision or KNOWN[repo]["sha"], allow_patterns=HF_ALLOW)


def repo_hashes(repo: str, local_dir: str) -> dict:
    """config / safetensors index (or the single-file header) / tokenizer file hashes of a snapshot."""
    out = {"repo": repo, "sha": KNOWN[repo]["sha"]}
    cfg = os.path.join(local_dir, "config.json")
    out["config_sha256"] = sha256_file(cfg) if os.path.exists(cfg) else None
    idx = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        out["safetensors_index_sha256"] = sha256_file(idx)
    else:
        single = [f for f in sorted(os.listdir(local_dir)) if f.endswith(".safetensors")]
        assert single, f"{repo}: no safetensors in {local_dir}"
        with open(os.path.join(local_dir, single[0]), "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            out["safetensors_index_sha256"] = "header:" + hashlib.sha256(f.read(n)).hexdigest()
        out["safetensors_files"] = single
    toks = [f for f in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json")
            if os.path.exists(os.path.join(local_dir, f))]
    out["tokenizer_files_sha256"] = {f: sha256_file(os.path.join(local_dir, f)) for f in toks}
    return out


def vocab_sha256(tok) -> str:
    return hashlib.sha256(json.dumps(sorted(tok.get_vocab().items()), ensure_ascii=False).encode()).hexdigest()


def shared_prefix(tok_a, tok_b) -> dict:
    """Ids 0..k-1 that map to the SAME token string in both vocabularies (B1: Molmo2 shares Qwen3's 151 936)."""
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    ia = {i: t for t, i in va.items()}
    ib = {i: t for t, i in vb.items()}
    k = 0
    while k in ia and k in ib and ia[k] == ib[k]:
        k += 1
    return {"shared_prefix": k, "n_before": len(va), "n_after": len(vb),
            "identical": va == vb, "vocab_sha256_before": vocab_sha256(tok_a), "vocab_sha256_after": vocab_sha256(tok_b)}


# ------------------------------------------------------------------------------------------ loading
def _cls(repo):
    import transformers
    arch = KNOWN[repo]["arch"]
    assert hasattr(transformers, arch), f"{repo}: transformers {transformers.__version__} has no {arch}"
    return getattr(transformers, arch)


def _load_model(repo, device, attn_impl, remote=False, device_map=None):
    import torch
    from transformers import AutoModelForImageTextToText
    local = hf_files(repo)
    kw = dict(dtype=torch.bfloat16, device_map=device_map or device, attn_implementation=attn_impl)
    if remote:
        m = AutoModelForImageTextToText.from_pretrained(local, trust_remote_code=True, **kw)
    else:
        m = _cls(repo).from_pretrained(local, **kw)
    return m.eval(), local


def _tokenizer(local, remote=False):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(local, trust_remote_code=remote)


def _processor(local, remote=False):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(local, trust_remote_code=remote)


def _forward(shell, model, tok_or_proc, device, head=None):
    from infra.pg import matrix_forward as MF
    if shell == "text":
        return MF.TextForward(model, tok_or_proc, device)
    if shell == "paligemma":
        return MF.PaliGemmaForward(model, tok_or_proc, device)
    if shell == "pi0-pg1":
        return MF.Pi0Forward(model, tok_or_proc, head, device)
    if shell == "molmo2":
        return MF.Molmo2Forward(model, tok_or_proc, device)
    raise NotImplementedError(shell)


def _rel_rows(a, b, chunk=8192) -> dict:
    """‖a-b‖/‖b‖ over the first min(rows) rows of two [V, d] tables, chunked in fp32 on CPU (an embedding
    table is gigabytes; never materialise two fp32 copies of one). `b` is the denominator = the base."""
    # Two shapes matter and only one used to be handled. Vocabulary (dim 0) may legitimately differ —
    # that is what `rows` is for. Hidden width (dim 1) may NOT: tables of different width are not
    # comparable at all, so this refuses to produce a number rather than inventing one.
    if int(a.shape[1]) != int(b.shape[1]):
        return {"rel_delta": None, "status": "NOT-COMPARABLE: hidden width differs "
                                             f"(shell {int(a.shape[1])} vs base {int(b.shape[1])})",
                "rows_shell": int(a.shape[0]), "rows_base": int(b.shape[0])}
    # The tail chunk MUST be clamped to `rows`, not to s+chunk. With a larger shell vocabulary
    # `a[s:s+chunk]` returns a full chunk while `b[s:s+chunk]` is clipped at the base's last row, so the
    # final iteration subtracted a [8192, d] block from a [2048, d] one. It killed row 2 — the anchor
    # install row — at `load_before` on 2026-09-04 with "size of tensor a (3264) must match tensor b
    # (2048)". The stub tables in the dry-run have equal row counts, so nothing local could see it.
    rows = min(int(a.shape[0]), int(b.shape[0]))
    d_sq = w_sq = 0.0
    for s in range(0, rows, chunk):
        end = min(s + chunk, rows)
        x = a[s:end].detach().float().cpu()
        y = b[s:end].detach().float().cpu()
        assert x.shape == y.shape, f"_rel_rows block mismatch at {s}: {tuple(x.shape)} vs {tuple(y.shape)}"
        d_sq += float(((x - y) ** 2).sum())
        w_sq += float((y ** 2).sum())
    return {"rows_compared": rows, "rel_delta": _rel(d_sq, w_sq),
            "rows_shell": int(a.shape[0]), "rows_base": int(b.shape[0])}


def install_decoder(shell_model, base_model, tag: str) -> dict:
    """Reset the shell's LM decoder layers + final norm to the base LM's; KEEP the shell's embed/head and
    everything else (the contrast_a1 rule, anatomy-driven so it holds for PaliGemma `layers/norm/embed_tokens`
    AND Molmo2 `blocks/ln_f/wte`). Qwen3 → Molmo2 keys go through family_resolvers.qwen3_to_molmo2.
    Asserts: every decoder key replaced from the base (no partial swap), every base decoder key consumed
    (no remap gap), strict load, decoder actually changed, non-decoder tensors bitwise kept.

    Also measures `embed_head_vs_base` HERE, and only here, because THIS is the one moment both tables are
    resident. Keeping the shell's embed/head is what makes the `before` member's embed identical to the
    `after` member's, so the B3 snapshot delta on an install row is 0 BY CONSTRUCTION — on exactly the rows
    whose fine-tune trained the embedding. The question that 0 pretended to answer ("did the install move
    the embedding?") is answered by this number: base LM's table vs the shell's, on the shared leading vocab
    rows (the base vocab is a prefix of the shell's: gemma 256000 ⊂ PaliGemma 257216, Qwen3 ⊂ Molmo2)."""
    import torch
    from infra.pg.matrix_forward import lm_anatomy
    an_s, an_b = lm_anatomy(shell_model), lm_anatomy(base_model)
    shell_lm = an_s["lm"]
    rel = lambda an, key: an[key][len(an["lm_path"]) + 1:]                 # 'layers' / 'norm' relative to the LM root
    dec_prefix = (rel(an_s, "layers_path") + ".", rel(an_s, "norm_path") + ".")
    base_sd = {k: v.detach().cpu() for k, v in an_b["lm"].state_dict().items()}
    if an_s["family"] in ("molmo2", "molmoact2"):
        from infra.pg.family_resolvers import qwen3_to_molmo2
        base_sd = qwen3_to_molmo2({**base_sd, "lm_head.weight": base_model.lm_head.weight.detach().cpu()})
    base_dec = {k: v for k, v in base_sd.items() if k.startswith(dec_prefix)}
    shell_sd = {k: v.detach().cpu().clone() for k, v in shell_lm.state_dict().items()}   # clone: CPU .cpu() aliases
    dec = [k for k in shell_sd if k.startswith(dec_prefix)]
    missing = [k for k in dec if k not in base_dec]
    assert not missing, f"{tag}: {len(missing)} shell decoder keys absent from the base (partial swap): {missing[:6]}"
    unused = sorted(set(base_dec) - set(dec))
    assert not unused, f"{tag}: {len(unused)} base decoder keys have no shell target (remap gap): {unused[:6]}"
    new_sd = dict(shell_sd)
    for k in dec:
        assert tuple(base_dec[k].shape) == tuple(shell_sd[k].shape), (tag, k, base_dec[k].shape, shell_sd[k].shape)
        new_sd[k] = base_dec[k]
    dev = next(shell_lm.parameters()).device
    res = shell_lm.load_state_dict({k: v.to(dev) for k, v in new_sd.items()}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    now = shell_lm.state_dict()
    max_delta = max((now[k].detach().cpu().float() - shell_sd[k].float()).abs().max().item() for k in dec)
    assert max_delta > 0.0, f"{tag}: decoder unchanged by the swap"
    kept = [k for k in shell_sd if k not in dec]
    for k in kept:
        assert torch.equal(now[k].detach().cpu(), shell_sd[k]), f"{tag}: {k} changed — embed/head must be KEPT"
    ps, pb = _lm_parts(shell_model), _lm_parts(base_model)      # both resident: the only moment they are
    vs_base = {"embed": _rel_rows(ps["embed"], pb["embed"]), "head": _rel_rows(ps["head"], pb["head"])}
    return {"tag": tag, "lm_reset_scope": f"{dec_prefix[0]}* + {dec_prefix[1]}* reset to the base LM; "
                                          f"{len(kept)} non-decoder tensors KEPT (embed/head, shell)",
            "n_decoder_keys_swapped": len(dec), "kept_keys": kept, "decoder_max_abs_delta_vs_shell": max_delta,
            "embed_head_vs_base": vs_base,
            "embed_head_vs_base_note": "shell's kept table vs the BASE LM's, shared leading vocab rows — the "
                                       "measurement the install row's B3 snapshot delta cannot make"}


def fetch_uri(uri: str, dest: str):
    """S3 (instance role) or presigned HTTP (Vast) — helmholtz2_arm._fetch."""
    from infra.aws_pg.jobs.helmholtz2_arm import _fetch
    _fetch(uri, dest)
    assert os.path.getsize(dest) > 0, dest


def merge_adapter(model, adapter_path: str, rank: int, alpha: float, text_mod=None) -> dict:
    """Merge a relearn.RankRAdapter checkpoint (keys `<module>.A` [out, r], `<module>.B` [r, in], relative to
    the module that owns `.layers` — helmholtz2_arm's text_mod) into the base Linears:
    W ← W + (alpha / r)·A·B in fp32, cast back. Every adapter key must be consumed; every relearn target
    Linear must receive one adapter (the H-2 run adapted all 238)."""
    import torch
    from infra.aws_pg.jobs.helmholtz2_arm import _text_module
    from infra.pg.relearn import adapter_target_names
    text_mod = text_mod if text_mod is not None else _text_module(model)
    sd = torch.load(adapter_path, map_location="cpu")
    names = sorted({k[:-2] for k in sd if k.endswith(".A")})
    assert names == sorted({k[:-2] for k in sd if k.endswith(".B")}), "A/B key sets differ"
    assert len(names) * 2 == len(sd), f"stray keys in adapter: {len(sd)} for {len(names)} modules"
    targets = adapter_target_names(text_mod)
    assert names == targets, f"adapter modules ≠ relearn targets: {len(names)} vs {len(targets)}"
    max_delta, scale = 0.0, float(alpha) / int(rank)
    with torch.no_grad():
        for n in names:
            lin = text_mod.get_submodule(n)
            A, B = sd[n + ".A"].float(), sd[n + ".B"].float()
            assert A.shape[1] == rank and B.shape[0] == rank, (n, A.shape, B.shape)
            d = scale * (A @ B)
            assert tuple(d.shape) == tuple(lin.weight.shape), (n, d.shape, lin.weight.shape)
            max_delta = max(max_delta, float(d.abs().max()))
            lin.weight.copy_((lin.weight.float() + d.to(lin.weight.device)).to(lin.weight.dtype))
    return {"n_merged": len(names), "rank": int(rank), "alpha": float(alpha), "scale": scale,
            "adapter_sha256": sha256_file(adapter_path), "max_abs_delta": max_delta}


def load_pair(row: str, device="cuda", which="before", adapter_path=None):
    """One member (str) or several (tuple) of a row. Returns {"fwd", "model", "tok", "proc", "repo",
    "which", "shell", "info", "local"} per member. Sequential loading is the caller's job (free one before
    loading the next); a tuple loads them all resident (graft job only)."""
    if not isinstance(which, str):
        return {w: load_pair(row, device, w, adapter_path) for w in which}
    r = ROWS[row]
    assert which in ("before", "after", "before-text", "after-pg1"), which
    for repo in (r["before"], r["after"]):
        assert KNOWN[repo]["access"] == "ok", f"row {row}: {repo} is {KNOWN[repo]['access']!r} — cannot load"
    if r["gated_on"]:
        assert os.environ.get("MDL_LOAD_TEST") == "1", f"row {row} is gated on {r['gated_on']}: --load-test-only only"
    info = {"row": row, "which": which}
    remote = r["needs_remote_code"]
    if which == "before-text":
        assert r["install"] and r["shell"] in SHELL_OF_PARITY, f"{row}: no text-only parity member"
        m, local = _load_model(r["before"], device, r["attn_impl"])
        tok = _tokenizer(local)
        return {"fwd": _forward("text", m, tok, device), "model": m, "tok": tok, "proc": None, "repo": r["before"],
                "which": which, "shell": "text", "info": info, "local": local}
    if r["policy"]:
        from infra.pg import lacuna_shell as LS
        assert r["shell"] == "pi0-pg1"
        m, local = _load_model(r["before"], device, r["attn_impl"])            # PG1 shell, eager
        proc = _processor(local)
        if which == "before":
            fwd = _forward("paligemma", m, proc, device)
            return {"fwd": fwd, "model": m, "tok": proc.tokenizer, "proc": proc, "repo": r["before"], "which": which,
                    "shell": "paligemma", "info": info, "local": local}
        head = "pg1" if which == "after-pg1" else "own"
        info["install"] = LS.build_row_policy(m, r["policy"], embed=head)
        fwd = _forward("pi0-pg1", m, proc, device, head=head)
        return {"fwd": fwd, "model": m, "tok": proc.tokenizer, "proc": proc, "repo": r["after"], "which": which,
                "shell": "pi0-pg1", "info": info, "local": local}
    assert which != "after-pg1", f"{row}: after-pg1 is a policy-row member"
    if r["adapter"]:
        m, local = _load_model(r["before"], device, r["attn_impl"])
        tok = _tokenizer(local)
        if which == "after":
            path = adapter_path or os.path.join(os.environ.get("MDL_SCRATCH", "/tmp"), "h2_adapter.pt")
            if not os.path.exists(path):
                fetch_uri(r["adapter"]["s3"], path)
            info["adapter"] = {**merge_adapter(m, path, r["adapter"]["rank"], r["adapter"]["alpha"]),
                               "source": r["adapter"]["s3"]}
        return {"fwd": _forward("text", m, tok, device), "model": m, "tok": tok, "proc": None,
                "repo": r["after"] if which == "after" else r["before"], "which": which, "shell": "text",
                "info": info, "local": local}
    if r["install"]:
        m, local = _load_model(r["after"], device, r["attn_impl"], remote=remote)     # the shell
        proc = _processor(local, remote=remote)
        if which == "before":
            base, base_local = _load_model(r["before"], "cpu", r["attn_impl"], device_map="cpu")
            info["install"] = install_decoder(m, base, f"{row}/before")
            # B1 has the same structural defect as B3 on an install row: BOTH members are served by the
            # SHELL's processor, so `binds`'s tokenizer comparison is the shell's vocab against itself and
            # `identical: true` is guaranteed. The base's own tokenizer is right here on disk, so take the
            # real comparison while we have it (this is what makes B1 a test on row 3b, Qwen3 ⊂ Molmo2).
            info["install"]["tokenizer_vs_base"] = shared_prefix(_tokenizer(base_local, remote=remote),
                                                                 proc.tokenizer)
            info["base_local"] = base_local
            del base
        fwd = _forward(r["shell"], m, proc, device)
        return {"fwd": fwd, "model": m, "tok": proc.tokenizer, "proc": proc,
                "repo": r["before"] if which == "before" else r["after"], "which": which, "shell": r["shell"],
                "info": info, "local": local}
    repo = r["before"] if which == "before" else r["after"]
    dm = "auto" if r["gated_on"] else None
    m, local = _load_model(repo, device, r["attn_impl"], remote=remote, device_map=dm)
    if r["shell"] == "text":
        tok = _tokenizer(local, remote=remote)
        return {"fwd": _forward("text", m, tok, device), "model": m, "tok": tok, "proc": None, "repo": repo,
                "which": which, "shell": "text", "info": info, "local": local}
    proc = _processor(local, remote=remote)
    return {"fwd": _forward(r["shell"], m, proc, device), "model": m, "tok": proc.tokenizer, "proc": proc,
            "repo": repo, "which": which, "shell": r["shell"], "info": info, "local": local}


MEMBER_KEYS = ("fwd", "model", "tok", "proc", "repo", "which", "shell", "info", "local")


def assert_member(member: dict, which: str | None = None) -> dict:
    """`load_pair`'s MEMBER CONTRACT, asserted — one DICT per member, never the Forward itself.

    This is the contract, not a convenience. Any stub standing in for `load_pair` in a dry-run MUST
    return members that pass this, and every caller that consumes a member should run it, or a green
    dry-run certifies the harness instead of the code. On 2026-09-03 `matrix_graft._stub_load_pair`
    returned the Forward where this loader returns the member dict, while its docstring claimed to
    mirror the loader: `matrix_graft --dryrun` passed and three GPU packs died at `ctx.attach` with
    `AttributeError: 'dict' object has no attribute 'batch_size'`."""
    assert isinstance(member, dict), f"load_pair returns a member dict, got {type(member).__name__}"
    assert set(member) == set(MEMBER_KEYS), f"member key set differs: {sorted(set(member) ^ set(MEMBER_KEYS))}"
    fwd = member["fwd"]
    for attr in ("model", "shell", "batch_size", "device", "next_token_logits"):
        assert hasattr(fwd, attr), f"member['fwd'] lacks .{attr} — not a contract §3 Forward"
    assert fwd.model is member["model"], "member['model'] must BE member['fwd'].model"
    assert member["tok"] is not None, "member['tok'] is the member's tokenizer (proc.tokenizer on VLM rows)"
    assert isinstance(member["shell"], str) and isinstance(member["info"], dict), (member["shell"], type(member["info"]))
    assert member["which"] in ("before", "after", "before-text", "after-pg1"), member["which"]
    assert which is None or member["which"] == which, (member["which"], which)
    return member


def forward_for(member: dict, model=None):
    """A FRESH contract-§3 Forward over `model` (default: the member's own model) in THIS member's shell,
    with its tokenizer/processor and its device — the same construction `load_pair` used for the member.

    Jobs evaluate models the loader never returned (the recovery job's fair-scratch shell, a graft-back
    donor copy), so the shell ↔ tokenizer/processor binding lives HERE beside `load_pair` instead of being
    re-derived, or guessed, by each job. Each call builds a new Forward: callers set `.batch_size` and
    swap `.model` independently."""
    assert_member(member)
    return _forward(member["shell"], member["model"] if model is None else model,
                    member["proc"] if member["proc"] is not None else member["tok"],
                    member["fwd"].device, head=("pg1" if member["which"] == "after-pg1" else "own"))


def text_forward_for(member: dict):
    """A plain TEXT Forward over an ALREADY-RESIDENT non-text member — its own model, its own tokenizer
    (`proc.tokenizer` on a VLM row), its own device, and NO image (`matrix_behaviour --after-text`).

    Built here, beside `load_pair` and `forward_for`, for the reason those give: the shell ↔
    tokenizer/processor binding lives in ONE place instead of being re-derived by each job. No weights
    move and nothing is loaded — the returned Forward shares `member["model"]`, so the sequential-loading
    rule (never two models resident) is untouched by construction.

    The member must NOT already be a text shell: there the reading exists and re-buying it would bank a
    duplicate of a column under a second name."""
    assert_member(member)
    assert member["shell"] != "text", (
        f"{member['which']}: already a text shell — its text reading IS the banked member, "
        "there is nothing to re-read (see matrix_models.text_members)")
    return _forward("text", member["model"], member["tok"], member["fwd"].device)


def free(member):
    import gc
    import torch
    for k in ("fwd", "model", "proc", "tok"):
        member[k] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------ binds (B1/B3/B4)
def _lm_parts(model) -> dict:
    """{'layers': [(name, tensor)...] per layer, 'norm': [...], 'embed': tensor, 'head': tensor} — the text
    decoder only (vision tower / projector / action expert / MTP are outside `lm_anatomy`)."""
    an = __import__("infra.pg.matrix_forward", fromlist=["lm_anatomy"]).lm_anatomy(model)
    layers = [[(k, v.detach()) for k, v in an["layers"][L].state_dict().items()] for L in range(an["n_layers"])]
    norm = [(k, v.detach()) for k, v in an["norm"].state_dict().items()]
    emb = an["embed"]
    emb_w = next(p for n, p in emb.state_dict().items() if p.ndim == 2)     # Molmo2 wte.embedding, else weight
    return {"layers": layers, "norm": norm, "embed": emb_w.detach(), "head": an["lm_head"].weight.detach(),
            "n_layers": an["n_layers"], "family": an["family"]}


def snapshot_lm(member, snap_dir: str) -> dict:
    """Write the resident model's LM decoder (per layer), final norm, embed and head to disk (own dtype, CPU)
    so the after-model can be compared without both being resident."""
    import torch
    os.makedirs(snap_dir, exist_ok=True)
    parts = _lm_parts(member["model"])
    for L, kv in enumerate(parts["layers"]):
        torch.save({k: v.cpu() for k, v in kv}, os.path.join(snap_dir, f"L{L}.pt"))
    torch.save({"norm": {k: v.cpu() for k, v in parts["norm"]},
                "embed": parts["embed"].cpu(), "head": parts["head"].cpu()},
               os.path.join(snap_dir, "tail.pt"))
    meta = {"dir": snap_dir, "n_layers": parts["n_layers"], "family": parts["family"], "repo": member["repo"],
            "which": member["which"], "row": member["info"]["row"],
            # carried so `binds` knows this `before` was BUILT (install rows) and can say which of its
            # comparisons are structurally void, and report the real numbers measured at build time
            "install": member["info"].get("install"),
            "vocab_sha256": vocab_sha256(member["tok"]),
            "vocab": {i: t for t, i in member["tok"].get_vocab().items()}}
    return meta


def _rel(delta_sq: float, base_sq: float) -> float:
    return float((delta_sq / base_sq) ** 0.5) if base_sq > 0 else float("nan")


def binds(snapshot: dict, after_member) -> dict:
    """B1/B3/B4 for one row from the `before` snapshot and the resident `after` model."""
    import torch
    parts = _lm_parts(after_member["model"])
    assert parts["n_layers"] == snapshot["n_layers"], (parts["n_layers"], snapshot["n_layers"])
    per_layer, frozen_layers = [], []
    for L, kv in enumerate(parts["layers"]):
        snap = torch.load(os.path.join(snapshot["dir"], f"L{L}.pt"), map_location="cpu")
        assert set(snap) == {k for k, _ in kv}, f"layer {L}: key sets differ ({sorted(set(snap) ^ {k for k, _ in kv})[:6]})"
        d_sq = w_sq = 0.0
        worst = ("", 0.0)
        for k, v in kv:
            a = v.float().cpu()
            b = snap[k].float()
            assert a.shape == b.shape, (L, k, a.shape, b.shape)
            dd, ww = float(((a - b) ** 2).sum()), float((b ** 2).sum())
            d_sq += dd
            w_sq += ww
            if ww > 0 and (dd / ww) ** 0.5 > worst[1]:
                worst = (k, (dd / ww) ** 0.5)
        per_layer.append({"layer": L, "rel_delta": _rel(d_sq, w_sq), "max_tensor_rel": worst[1], "max_tensor": worst[0],
                          "exact_zero": d_sq == 0.0})
        frozen_layers.append(d_sq == 0.0)
    tail = torch.load(os.path.join(snapshot["dir"], "tail.pt"), map_location="cpu")
    n_sq = nd_sq = 0.0
    for k, v in parts["norm"]:
        nd_sq += float(((v.float().cpu() - tail["norm"][k].float()) ** 2).sum())
        n_sq += float((tail["norm"][k].float() ** 2).sum())
    tok_a = after_member["tok"]
    va = {i: t for t, i in tok_a.get_vocab().items()}
    vb = snapshot["vocab"]
    k = 0
    while k in va and k in vb and va[k] == vb[k]:
        k += 1
    rows = min(parts["embed"].shape[0], tail["embed"].shape[0], k) if k > 0 else min(parts["embed"].shape[0], tail["embed"].shape[0])
    emb_a, emb_b = parts["embed"][:rows].float().cpu(), tail["embed"][:rows].float()
    head_a, head_b = parts["head"][:rows].float().cpu(), tail["head"][:rows].float()
    emb_d = _rel(float(((emb_a - emb_b) ** 2).sum()), float((emb_b ** 2).sum()))
    head_d = _rel(float(((head_a - head_b) ** 2).sum()), float((head_b ** 2).sum()))
    # ---- the install row's structural zeros. `before` on an install row is not a loaded model: it is the
    # AFTER model's own shell with its decoder reset (`install_decoder`), which KEEPS the shell's embed,
    # head and processor. So on those rows both `emb_d`/`head_d` and `tokenizer.identical` compare the
    # shell against itself — 0 and True BY CONSTRUCTION, on exactly the rows whose fine-tune trained the
    # embedding, which made "the install left the embedding and head alone" unfalsifiable where it is
    # claimed. They are therefore NOT reported as measurements: `rel_delta` is None (None is loud in any
    # arithmetic; 0.0 reads as a result), the refused constant is kept beside it as `snapshot_delta` so
    # the defect stays visible in the record, and `vs_base` carries the real base-LM-vs-shell number
    # `install_decoder` measured while both tables were resident.
    inst = snapshot.get("install")
    void = ("`before` is the after-model's own shell with only its decoder reset, so this compares the "
            "shell against itself — 0/identical by construction, not a measurement (see vs_base)") if inst else None
    emb_vs_base = inst["embed_head_vs_base"]["embed"] if inst else None
    head_vs_base = inst["embed_head_vs_base"]["head"] if inst else None
    out = {"n_layers": parts["n_layers"], "family": parts["family"], "per_layer": per_layer,
           "final_norm_rel_delta": _rel(nd_sq, n_sq), "final_norm_exact_zero": nd_sq == 0.0,
           "embed": {"rows_compared": int(rows), "rel_delta": None if inst else emb_d,
                     "structurally_void": void, "snapshot_delta": emb_d if inst else None,
                     "vs_base": emb_vs_base,
                     "rows_before": int(tail["embed"].shape[0]), "rows_after": int(parts["embed"].shape[0])},
           "head": {"rows_compared": int(rows), "rel_delta": None if inst else head_d,
                    "structurally_void": void, "snapshot_delta": head_d if inst else None,
                    "vs_base": head_vs_base,
                    "tied_after": bool(parts["head"].data_ptr() == parts["embed"].data_ptr())},
           "tokenizer": {"shared_prefix": None if inst else k, "n_before": len(vb), "n_after": len(va),
                         "identical": None if inst else va == vb, "structurally_void": void,
                         "vs_base": inst.get("tokenizer_vs_base") if inst else None,
                         "vocab_sha256_before": snapshot["vocab_sha256"], "vocab_sha256_after": vocab_sha256(tok_a)},
           "frozen_lm": all(frozen_layers) and nd_sq == 0.0,
           "excluded": "vision tower / projector / action expert / MTP / per-layer inputs: outside lm_anatomy"}
    # the measurement meets the label (see BENIGN_MAX_*): a row that declares a benign kind and did not
    # move like one fails HERE, the moment its own number exists, not in a comment written afterwards.
    # `Job.bank` has already written every banked column to metrics.json, so what this costs is the run's
    # tail, not its data — and a mislabelled control's contrast is not worth banking anyway.
    mean_dec = sum(x["rel_delta"] for x in per_layer) / len(per_layer)
    out["kind_delta"] = assert_kind_delta(snapshot["row"], mean_dec,
                                          emb_vs_base["rel_delta"] if inst else emb_d)
    return out


# ------------------------------------------------------------------------------------------ §8b: model-card transformers pin
def transformers_version() -> str:
    """Installed version WITHOUT importing transformers (the job calls the pin check before any import)."""
    from importlib.metadata import version
    return version("transformers")


def _build_pin_venv(venv: str, pin: str):
    """`python -m venv --system-site-packages`, a `.pth` back to THIS interpreter's site-packages, then
    `pip install transformers==<pin>`. check=True: a failed install is fatal.

    The `.pth`, written BEFORE pip runs, is the load-bearing part. `--system-site-packages` inherits the
    BASE interpreter's site dir and never the parent VENV's, and the AWS jobs run under /opt/pg/venv,
    where `bake_env.sh` installs torch, datasets, accelerate and Pillow. So the pin venv saw only
    /usr/lib/python3/dist-packages: no torch, no datasets, no science stack at all. Row 3b
    (Qwen3-4B-Instruct -> Molmo2-4B) built the venv, re-execed correctly, fetched 22 weight files, and
    then died 15 minutes in on `ModuleNotFoundError: No module named 'datasets'` in `build_items` --
    having re-downloaded a numpy/fsspec/safetensors/tqdm set the parent already had. The venv could
    never have loaded a model; the pin mechanism had never run end to end.

    Pointing back at the parent fixes both halves at once. The runtime resolves, and pip now sees the
    parent's packages as already installed, so it changes ONLY what the pin genuinely requires instead
    of shadowing the AMI's pinned scientific stack -- its unbounded fsspec had resolved to 2026.7.0,
    which `datasets==4.8.5` forbids (`fsspec[http]<=2026.2.0`).

    Precedence is what makes it safe: `site.addsitedir` appends a `.pth` path AFTER the site dir holding
    it, so the pin venv's own transformers wins over the parent's 5.9.0 and everything it does not carry
    falls through. The one version the pin forces down is compatible with the AMI by declared contract:
    transformers 4.57.1 wants huggingface-hub<1.0,>=0.34.0 and datasets 4.8.5 wants
    huggingface-hub<2.0,>=0.25.0, so the 0.36.x pip lands satisfies both. pip will not uninstall the
    parent's copies -- they sit outside its environment prefix, the same path `--system-site-packages`
    already relied on."""
    import subprocess
    import sysconfig
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=True)
    py = os.path.join(venv, "bin", "python")
    child_site = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                check=True, capture_output=True, text=True).stdout.strip()
    assert os.path.isdir(child_site), f"pin venv has no site-packages: {child_site!r}"
    # Read the parent's site dirs off its LIVE sys.path, not off a sysconfig scheme name: Debian patches
    # sysconfig with `posix_local`, so `get_paths()['purelib']` can name /usr/local/... rather than the
    # venv the packages are actually in. What the parent can import is the thing we are copying.
    parent_sites = [p for p in sys.path
                    if p.startswith(sys.prefix) and os.path.basename(p) in ("site-packages", "dist-packages")
                    and os.path.isdir(p)]
    if not parent_sites and os.path.isdir(sysconfig.get_paths()["purelib"]):
        parent_sites = [sysconfig.get_paths()["purelib"]]
    assert parent_sites, f"cannot locate the running interpreter's site-packages under {sys.prefix!r}"
    with open(os.path.join(child_site, "_pg_parent_site.pth"), "w") as fh:
        fh.write("\n".join(parent_sites) + "\n")
    print(f"[pin] {child_site} falls through to {parent_sites}", flush=True)
    # `protobuf` and `sentencepiece` ride along because transformers needs them to build a tokenizer
    # from a sentencepiece model, and NEITHER is in the AMI (`bake_env.sh` pins neither), so the `.pth`
    # back to the parent cannot supply them. Row 3c-Act (MolmoAct2) died on exactly this:
    # `ImportError` from `tokenization_utils_base.import_protobuf_decode_error`, 13 cells lost. Rows 3b
    # and 3c-ER survive without them because their tokenizers need no conversion — which is why this
    # surfaced on the third Molmo row rather than the first.
    subprocess.run([py, "-m", "pip", "install", f"transformers=={pin}", "protobuf", "sentencepiece"],
                   check=True)


# The modules the pin venv must import before a job is allowed to spend GPU minutes inside it.
# `datasets` is on this list because its absence is exactly what killed row 3b, and it is not reached
# until `build_items`, long after the weights are on disk.
PIN_VENV_PROBE = ("transformers", "torch", "datasets")


def ensure_transformers_pin(row: str, module: str, argv, running=None, venv_root="/opt/pg", build=None, run=None):
    """§8b frozen response to a model-card transformers pin that differs from the base env (row 3: 4.57.1 vs
    5.9.0). Returns None when the running version already matches the row's pin. Otherwise: build
    `<venv_root>/venv-tf<pin>` once, then run `python -m <module> <argv>` under it as a CHILD process
    (subprocess + check=True rather than execv, so the Vast cell shim's bookkeeping survives — the parent holds
    no model) with MDL_HARNESS_SHIFT set; the child records `manifest.harness_shift`. Returns the shift record
    (the caller must then return without running in-process). Call BEFORE importing transformers.
    Shared by matrix_behaviour / matrix_graft / matrix_recovery."""
    import subprocess
    pin = ROWS[row]["transformers_pin"]
    running = running or transformers_version()
    if pin is None or running == pin:
        return None
    assert os.environ.get("MDL_TF_REEXEC") != "1", \
        f"already re-run under the pin venv, but transformers is {running} (pin {pin}) — venv broken"
    venv = os.path.join(venv_root, f"venv-tf{pin}")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        (build or _build_pin_venv)(venv, pin)
    assert os.path.exists(py), f"pin venv has no interpreter: {py}"
    # Preflight, fail-loud and seconds long: the pin venv must import the science stack BEFORE the job
    # is re-run inside it. Without this the first proof the venv is usable is the job itself, and row 3b
    # spent 15 GPU-minutes fetching Molmo2-4B before reporting that the venv had no `datasets`.
    probe = subprocess.run(
        [py, "-c", f"import {', '.join(PIN_VENV_PROBE)}; import transformers as _t; print(_t.__version__)"],
        capture_output=True, text=True)
    got = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    assert probe.returncode == 0 and got == pin, (
        f"pin venv {venv} cannot run the job: importing {PIN_VENV_PROBE} gave transformers {got!r} "
        f"(want {pin!r}, rc={probe.returncode})\n{probe.stderr.strip()[-2000:]}")
    shift = {"transformers": f"{pin} (model-card pin; base env {running})", "venv": venv, "module": module}
    env = {**os.environ, "MDL_TF_REEXEC": "1", "MDL_HARNESS_SHIFT": json.dumps(shift)}
    print(f"[pin] transformers {running} → {pin}: re-running {module} under {venv}", flush=True)
    (run or (lambda cmd, env: subprocess.run(cmd, check=True, env=env)))([py, "-m", module, *argv], env)
    return shift


# ------------------------------------------------------- refreshing the facts file (network; not at import)
BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "float32": 4}


def _facts_from_config(cfg: dict, index: dict | None) -> dict:
    """The fields `matrix_audit` compares the registry against, read off a repo's OWN config.json and
    safetensors index. Every one of them is the repo's statement about itself; none is a judgement."""
    tc = cfg.get("text_config") or {}
    def pick(*keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
            if k in tc:
                return tc[k]
        return None
    dtype = pick("torch_dtype", "dtype")
    total = ((index or {}).get("metadata") or {}).get("total_size")
    quantized = "quantization_config" in cfg
    # A parameter count is derivable only when the index says how many BYTES the tensors take AND the
    # config says how wide one parameter is. Quantized releases (row 4's DeepSeek: fp8 weights, fp4
    # experts) publish bytes at several widths at once, so `params_b` is NOT derivable there and the
    # audit says so instead of inventing a divisor.
    params_b = None
    if total is not None and not quantized and dtype in BYTES_PER_PARAM:
        params_b = round(total / BYTES_PER_PARAM[dtype] / 1e9, 4)
    # The TEXT model type: `text_config.model_type` on a VLM (trailing "_text" is the sub-config's own
    # suffix, e.g. gemma3_text / molmo2_text), `model_type` on a plain LM. This is what `family` names.
    text_mt = tc.get("model_type") or cfg.get("model_type")
    if isinstance(text_mt, str) and text_mt.endswith("_text"):
        text_mt = text_mt[: -len("_text")]
    return {"transformers_version": cfg.get("transformers_version"),
            "model_type": cfg.get("model_type"), "text_model_type": text_mt,
            "architectures": cfg.get("architectures"), "auto_map": "auto_map" in cfg,
            "n_layers": pick("num_hidden_layers"), "hidden": pick("hidden_size"),
            "vocab": pick("vocab_size"), "tie": pick("tie_word_embeddings"),
            "dtype": dtype, "quantized": quantized,
            "total_size": total, "params_b_from_index": params_b}


def refresh_facts(path: str = None, profile: str = "fuelos", region: str = "us-east-1") -> dict:
    """Regenerate `matrix_repo_facts.json` from the SOURCES. Network; never called at import or in a job.

        python -m infra.pg.matrix_models --refresh-facts

    For every repo in KNOWN: its `config.json` and `model.safetensors.index.json` at the PINNED revision,
    read from the S3 weight cache when the bucket holds the repo and from the HF resolve endpoint
    otherwise, plus the bucket's own `weights/hf/hub/` listing. A repo whose config the token cannot read
    is recorded with `config: null` and its KNOWN access reason — that pairing is itself checked, so
    "403 licence" can never be claimed for a repo whose config in fact reads fine."""
    import datetime
    import subprocess
    import urllib.request
    path = path or _FACTS_PATH
    ls = subprocess.run(["aws", "s3", "ls", f"s3://{S3_BUCKET}/weights/hf/hub/", "--profile", profile,
                         "--region", region], capture_output=True, text=True)
    assert ls.returncode == 0, f"cannot list the weight cache: {ls.stderr.strip()[:300]}"
    dirs = sorted(ln.split("PRE ", 1)[1].rstrip("/") for ln in ls.stdout.splitlines()
                  if " PRE " in ln and "PRE models--" in ln)
    tokp = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                           "--with-decryption", "--profile", profile, "--region", region,
                           "--query", "Parameter.Value", "--output", "text"], capture_output=True, text=True)
    assert tokp.returncode == 0, f"cannot read the HF token: {tokp.stderr.strip()[:200]}"
    token = tokp.stdout.strip()

    def fetch(repo, sha, name):
        d = "models--" + repo.replace("/", "--")
        if d in dirs:
            key = f"weights/hf/hub/{d}/snapshots/{sha}/{name}"
            r = subprocess.run(["aws", "s3", "cp", f"s3://{S3_BUCKET}/{key}", "-", "--profile", profile,
                                "--region", region], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout, f"s3://{S3_BUCKET}/{key}"
        url = f"https://huggingface.co/{repo}/resolve/{sha}/{name}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req) as fh:
                return fh.read().decode(), url
        except urllib.error.HTTPError as e:
            return None, f"{url} -> HTTP {e.code}"

    out = {"_meta": {"generated": datetime.date.today().isoformat(),
                     "by": "python -m infra.pg.matrix_models --refresh-facts",
                     "bucket": S3_BUCKET, "bucket_listed": datetime.date.today().isoformat(),
                     "bucket_dirs": dirs,
                     "sources": "each repo's own config.json + model.safetensors.index.json at the "
                                "KNOWN-pinned revision: the S3 weight cache where the bucket holds the "
                                "repo, the HF resolve endpoint otherwise. Nothing here is hand-typed."}}
    for repo, k in KNOWN.items():
        sha = k["sha"]
        cfg_txt, cfg_src = fetch(repo, sha, "config.json")
        idx_txt, idx_src = fetch(repo, sha, "model.safetensors.index.json")
        rec = {"revision": sha, "config_source": cfg_src, "index_source": idx_src,
               "config_sha256": hashlib.sha256(cfg_txt.encode()).hexdigest() if cfg_txt else None,
               "config": None}
        if cfg_txt is not None:
            rec["config"] = _facts_from_config(json.loads(cfg_txt),
                                               json.loads(idx_txt) if idx_txt else None)
        out[repo] = rec
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    print(f"wrote {path}: {len(KNOWN)} repos, {len(dirs)} cached dirs, "
          f"{sum(1 for r in out.values() if isinstance(r, dict) and r.get('config'))} configs read")
    return out


# ------------------------------------------------------------------------------------------ dry-run
def _dryrun_kind_bands():
    """The kind↔delta band: its derivation, what it admits, and what it would have caught.

    Prints the before/after for the two rows that carried a false `benign/pt`, computed from the banked
    numbers rather than asserted from memory."""
    import math
    g2, a3 = BANKED_DELTA["ctrl-g2"], BANKED_DELTA["3a"]
    assert round(math.sqrt(g2[0] * a3[0]), 4) == BENIGN_MAX_DECODER, "decoder band ≠ its stated derivation"
    assert round(math.sqrt(g2[1] * a3[1]), 4) == BENIGN_MAX_EMBED, "embed band ≠ its stated derivation"
    assert ROWS["ctrl-g2"]["kind"].startswith("benign"), "ctrl-g2 IS the benign reference"
    assert not ROWS["3a"]["kind"].startswith("benign") and not ROWS["ctrl-g1"]["kind"].startswith("benign")
    assert kind_delta_verdict("ctrl-g2", *g2[:2])["consistent"], "the reference must pass its own band"
    # the counterfactual, executed: put the old label back and the module must refuse to build
    for row, (dec, emb, cell) in (("3a", BANKED_DELTA["3a"]), ("ctrl-g1", BANKED_DELTA["ctrl-g1"])):
        real = ROWS[row]["kind"]
        ROWS[row]["kind"] = "benign/pt"
        caught = ""
        try:
            _check_banked_kinds()
        except AssertionError as e:
            caught = str(e)
        finally:
            ROWS[row]["kind"] = real
        assert row in caught and cell in caught and "RELABEL" in caught, caught or "NOT CAUGHT"
        print(f"  kind band: {row} 'benign/pt' -> CAUGHT at import (decoder {dec} = "
              f"{dec / BENIGN_MAX_DECODER:.2f}x band, embed {emb} = {emb / BENIGN_MAX_EMBED:.2f}x); "
              f"now kind={real!r}")
    # one-sided, and said so: 5c destroys MMLU while moving less than the benign reference
    assert BANKED_DELTA["5c"][0] < BANKED_DELTA["ctrl-g2"][0] < BENIGN_MAX_DECODER
    assert kind_delta_verdict("5c", *BANKED_DELTA["5c"][:2])["gated"] is False


def _dryrun():
    import tempfile
    import torch
    from infra.pg.matrix_forward import TextForward, _StubTok, stub_text_model
    # THE DERIVED-VALUE AUDIT (2026-09-04). Every field of this registry that a machine-readable source
    # already knows — each repo's own config.json, its safetensors index, the weight bucket's listing —
    # is compared against that source and FAILS LOUD on divergence, here, in the check that gates every
    # launch. This is the mechanism that ends the class: a new row with a wrong pin, a wrong params_b or
    # a stale cache flag can no longer be committed, only discovered.
    from infra.pg import matrix_audit
    _audit = matrix_audit.audit_derived()
    for e in _audit["escalated"]:
        print(f"  REGISTRY ESCALATION — {e.splitlines()[0]}")
    print(f"  derived-value audit: {_audit['checked_rows']} rows / {_audit['checked_repos']} repos "
          f"against their own sources; {len(_audit['unchecked'])} field(s) the source is silent about "
          f"(reported, never taken as agreement); {len(_audit['escalated'])} escalated "
          f"(python -m infra.pg.matrix_audit --verbose for the full table)")
    for row, r in ROWS.items():
        assert r["before"] in KNOWN and r["after"] in KNOWN, row
        assert r["shell"] in ("text", "paligemma", "molmo2", "pi0-pg1"), row
        assert r["gb_pair"] > r["gb_one"] > 0 and set(r["s3_cached"]) == set(row_repos(row))
        if r["install"]:
            assert KNOWN[r["before"]]["n_layers"] == KNOWN[r["after"]]["n_layers"], row
        for repo in row_repos(row):
            assert KNOWN[repo]["sha"] and len(KNOWN[repo]["sha"]) == 40, repo
    assert sum(1 for r in KNOWN.values() if r["access"] == "ok") == len(KNOWN) - 2
    assert ROWS["5b"]["access"]["google/medgemma-4b-pt"] == "403 licence"
    assert ROWS["7b"]["access"]["meta-llama/Meta-Llama-3-8B-Instruct"] == "403 licence"
    assert members("2") == ["before-text", "before", "after"] and members("5") == ["before", "after"]
    # text_members: derived from members(), so a row's eligibility for the opt-in text-shell re-reading
    # cannot drift from the registry. Install-into-VLM rows have it; every other row is EMPTY, and empty
    # is the statement "this row's `after` is already a text reading, or has none at all".
    assert text_members("2") == {"before": "before-shell-text", "after": "after-text"}
    assert all(text_members(r) == TEXT_MEMBER_OF for r in ("1a", "2", "2-10B", "3b"))
    assert all(text_members(r) == {} for r in ("1b-pi0", "1b-pi05", "3a", "3c-ER", "3c-Act", "4", "5", "5b",
                                               "5c", "6", "7a", "7b", "ctrl-g1", "ctrl-g2"))
    # every row answers, and the answer is the registry's: eligible ⟺ the row has a `before-text` member
    assert {r for r in ROWS if text_members(r)} == {r for r in ROWS if "before-text" in members(r)}
    assert sorted({m for r in ROWS for m in text_members(r)}) == ["after", "before"]
    # o_tasks: the O instrument as task names, every row reachable, [] a statement and not a gap
    assert {r: o_tasks(r) for r in ("1a", "2", "2-10B", "3b", "3c-ER")} == {r: ["vqav2", "textvqa"] for r in
                                                                           ("1a", "2", "2-10B", "3b", "3c-ER")}
    assert all(o_tasks(r) == [] for r in ("1b-pi0", "1b-pi05", "3c-Act", "7a", "7b", "3a", "4", "5", "5b", "5c",
                                          "6", "ctrl-g1", "ctrl-g2"))
    assert set(ROWS) == {r for r in ROWS if o_tasks(r) is not None} and len(ROWS) == 18
    assert all(t in ("refcoco", "vqav2", "textvqa") for t in O_TASKS)
    assert sum(v for r in ROWS.values() for v in r["s3_cached"].values()) > 0
    assert set(k for r in ROWS.values() for k, v in r["s3_cached"].items() if v) <= set(S3_CACHED)
    raised = False
    try:
        load_pair("5b", "cpu", "after")
    except AssertionError as e:
        raised = "403 licence" in str(e)
    assert raised, "a 403 row must fail loud with its reason"
    _dryrun_kind_bands()

    # ---- the REAL loader's return shape, weight-free (the model/tokenizer/processor calls patched out).
    # `assert_member` on every branch a job attaches: plain text row, install row (both members), and the
    # text-only parity member. This is what makes a stub check meaningful — a dry-run that only exercises
    # a stub certifies the harness. 2026-09-03: matrix_graft's stub returned the Forward itself where this
    # loader returns the member dict, `matrix_graft --dryrun` passed, and three GPU packs died in attach.
    import sys as _sys
    me = _sys.modules[__name__]
    saved = {k: getattr(me, k) for k in ("_load_model", "_tokenizer", "_processor", "_forward", "install_decoder")}

    class _Proc:                                   # a processor is the tokenizer holder on the VLM rows
        def __init__(self):
            self.tokenizer = _StubTok()

    class _F:                                      # contract §3 Forward, weight-free
        shell, batch_size, device = "stub", 4, "cpu"

        def __init__(self, model):
            self.model = model

        def next_token_logits(self, prompts):
            raise NotImplementedError

    me._load_model = lambda repo, device, attn_impl, remote=False, device_map=None: (stub_text_model(seed=2), "/local/" + repo)
    me._tokenizer = lambda local, remote=False: _StubTok()
    me._processor = lambda local, remote=False: _Proc()
    me._forward = lambda shell, model, tok_or_proc, device, head=None: _F(model)
    me.install_decoder = lambda shell_model, base_model, tag: {"stub": True}
    try:
        for row, which in (("5", "before"), ("5", "after"), ("2", "before"), ("2", "after"), ("2", "before-text")):
            assert_member(load_pair(row, "cpu", which), which)
        pair = load_pair("5", "cpu", ("before", "after"))
        assert set(pair) == {"before", "after"} and all(assert_member(pair[w], w) for w in pair)
        assert pair["before"]["repo"] == ROWS["5"]["before"] and pair["after"]["repo"] == ROWS["5"]["after"]
        assert "manifest" not in pair and "raw" not in pair, "load_pair returns members only — no top-level keys"
        f0 = forward_for(load_pair("5", "cpu", "before"))            # routes shell/tok/device to _forward
        assert isinstance(f0, _F)
    finally:
        for k, v in saved.items():
            setattr(me, k, v)

    # forward_for with the REAL _forward: a fresh Forward over a model the LOADER NEVER RETURNED (the
    # recovery job's fair-scratch shell / graft-back donor copy), in the member's shell and on its device
    m_own, m_other = stub_text_model(seed=8), stub_text_model(seed=9)
    mem = assert_member({"fwd": TextForward(m_own, _StubTok(), "cpu"), "model": m_own, "tok": _StubTok(),
                         "proc": None, "repo": "stub/x", "which": "before", "shell": "text",
                         "info": {}, "local": "/local"})
    f_own, f_other = forward_for(mem), forward_for(mem, m_other)
    assert isinstance(f_own, TextForward) and f_own.model is m_own and f_own is not mem["fwd"]
    assert f_other.model is m_other and f_other.shell == "text" and f_other.device == "cpu"

    # text_forward_for with the REAL _forward: the SAME resident model read through a plain text shell.
    # The model object must be shared (no second copy resident) and the shell must actually be "text".
    vlm_tok = _StubTok()
    vlm = assert_member({"fwd": _F(m_own), "model": m_own, "tok": vlm_tok, "proc": object(),
                         "repo": "stub/shell", "which": "after", "shell": "paligemma",
                         "info": {}, "local": "/local"})
    ft = text_forward_for(vlm)
    assert isinstance(ft, TextForward) and ft.shell == "text" and ft.device == "cpu"
    assert ft.model is vlm["model"] and ft.tok is vlm_tok, "the text re-read must share the resident model"
    assert ft.next_token_logits(["Answer:"]).shape[0] == 1        # the real TextForward call path runs
    caught = ""
    try:
        text_forward_for(mem)                                     # a text member has nothing to re-read
    except AssertionError as e:
        caught = str(e)
    assert "already a text shell" in caught, caught or "text_forward_for must refuse a text member"

    # binds on stubs: identical → frozen_lm; perturb layer 1 → that layer only; tokenizer shared prefix
    torch.manual_seed(0)
    ma, mb = stub_text_model(seed=1), stub_text_model(seed=1)
    tok = _StubTok()
    tok.get_vocab = lambda: {**{f"t{i}": i for i in range(20)}, **{k: v for k, v in tok._v.items()}}
    mem_b = {"model": ma, "tok": tok, "repo": "stub/before", "which": "before", "info": {"row": "5c"}}
    mem_a = {"model": mb, "tok": tok, "repo": "stub/after", "which": "after", "info": {"row": "5c"}}
    with tempfile.TemporaryDirectory() as d:
        snap = snapshot_lm(mem_b, os.path.join(d, "snap"))
        b0 = binds(snap, mem_a)
        assert b0["frozen_lm"] and all(x["exact_zero"] for x in b0["per_layer"]) and b0["tokenizer"]["identical"]
        assert b0["embed"]["rel_delta"] == 0.0 and b0["head"]["rel_delta"] == 0.0
        assert b0["embed"]["structurally_void"] is None and b0["embed"]["vs_base"] is None   # a REAL zero
        assert b0["kind_delta"]["gated"] is False and b0["kind_delta"]["decoder_rel"] == 0.0
        with torch.no_grad():
            mb.model.layers[1].lin.weight.add_(0.1)
        b1 = binds(snap, mem_a)
        assert not b1["frozen_lm"] and b1["per_layer"][1]["rel_delta"] > 0 and b1["per_layer"][0]["exact_zero"]
        assert b1["per_layer"][1]["max_tensor"] == "lin.weight"
        # the same measurement under a BENIGN label must stop the job where it is made
        mem_b2 = {**mem_b, "info": {"row": "ctrl-g2"}}
        mem_a3 = {**mem_a, "info": {"row": "ctrl-g2"}}
        snap_g2 = snapshot_lm(mem_b2, os.path.join(d, "snap_g2"))
        caught = ""
        try:
            binds(snap_g2, mem_a3)
        except AssertionError as e:
            caught = str(e)
        assert "ctrl-g2" in caught and "RELABEL" in caught, caught or "a benign row that moved must fail loud"
        tok2 = _StubTok()
        tok2.get_vocab = lambda: {**{f"t{i}": i for i in range(20)}, **{k: v for k, v in tok._v.items()}, "extra": 99}
        mem_a2 = {"model": mb, "tok": tok2, "repo": "stub/after", "which": "after", "info": {"row": "5c"}}
        b2 = binds(snap, mem_a2)
        assert b2["tokenizer"]["shared_prefix"] == 20 and not b2["tokenizer"]["identical"]

    # decoder swap on stubs (the contrast_a1 rule, anatomy-driven): decoder keys swapped, embed kept, base consumed
    shell, base = stub_text_model(seed=1), stub_text_model(seed=7)
    emb0 = shell.model.embed_tokens.weight.clone()
    info = install_decoder(shell, base, "stub")
    assert info["n_decoder_keys_swapped"] == len(shell.model.state_dict()) - 1 and info["kept_keys"] == ["embed_tokens.weight"]
    assert torch.equal(shell.model.embed_tokens.weight, emb0)
    assert torch.equal(shell.model.layers[0].lin.weight, base.model.layers[0].lin.weight)

    # ---- the INSTALL row's structural zero, reproduced weight-free and then refused.
    # `before` = this shell with its decoder reset to the base and its OWN embed kept; `after` = the same
    # shell untouched. The B3 embed/head delta between them is 0 whatever the install did to the table.
    true_emb = _rel_rows(shell.model.embed_tokens.weight, base.model.embed_tokens.weight)["rel_delta"]
    assert abs(info["embed_head_vs_base"]["embed"]["rel_delta"] - true_emb) < 1e-9 and true_emb > 0.5
    after_shell = stub_text_model(seed=1)
    with tempfile.TemporaryDirectory() as d:
        snap_i = snapshot_lm({"model": shell, "tok": tok, "repo": "stub/shell", "which": "before",
                              "info": {"row": "2", "install": info}}, os.path.join(d, "snap_i"))
        bi = binds(snap_i, {"model": after_shell, "tok": tok, "repo": "stub/shell", "which": "after",
                            "info": {"row": "2"}})
    assert bi["embed"]["snapshot_delta"] == 0.0 and bi["head"]["snapshot_delta"] == 0.0   # THE DEFECT
    assert bi["embed"]["rel_delta"] is None and bi["head"]["rel_delta"] is None           # no longer a number
    assert bi["tokenizer"]["identical"] is None and bi["tokenizer"]["shared_prefix"] is None
    assert bi["embed"]["structurally_void"] and abs(bi["embed"]["vs_base"]["rel_delta"] - true_emb) < 1e-9
    assert bi["kind_delta"]["embed_rel"] == bi["embed"]["vs_base"]["rel_delta"]           # the honest one
    assert bi["per_layer"][0]["rel_delta"] > 0, "the decoder bind is unaffected and still measures"
    print(f"  install-row binds: snapshot embed/head delta {bi['embed']['snapshot_delta']} is STRUCTURAL "
          f"(shell vs itself) -> rel_delta None; vs_base {true_emb:.4f} over "
          f"{bi['embed']['vs_base']['rows_compared']} rows is the measurement")

    # adapter merge on a stub: W += (α/r)·A·B on every target Linear (stub's Linears live under model.layers)
    m5 = stub_text_model(seed=3)
    from infra.pg.relearn import adapter_target_names
    names = adapter_target_names(m5.model)
    assert names == [f"layers.{i}.lin" for i in range(4)], names
    g = torch.Generator().manual_seed(0)
    sd = {}
    for n in names:
        sd[n + ".A"] = torch.randn(16, 2, generator=g)
        sd[n + ".B"] = torch.randn(2, 16, generator=g)
    w0 = m5.model.layers[2].lin.weight.clone()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "adapter.pt")
        torch.save(sd, p)
        info = merge_adapter(m5, p, 2, 4.0, text_mod=m5.model)
    exp = w0 + 2.0 * (sd["layers.2.lin.A"] @ sd["layers.2.lin.B"])
    assert info["n_merged"] == 4 and info["scale"] == 2.0 and torch.allclose(m5.model.layers[2].lin.weight, exp, atol=1e-6)
    fwd = TextForward(m5, _StubTok(), device="cpu")
    assert fwd.next_token_logits(["Answer:"]).shape == (1, 256)

    # §8b pin response: no-op when matched; build once + child run + shift record when not; loop guard
    calls = {}
    def _shim(venv, reports):
        """A fake `bin/python` that answers the preflight probe with `reports` on stdout. Executable, so
        the REAL preflight subprocess runs against it — an empty file (the old stub) would only have
        proved that the probe was never reached."""
        os.makedirs(os.path.join(venv, "bin"), exist_ok=True)
        p = os.path.join(venv, "bin", "python")
        with open(p, "w") as fh:
            fh.write(f"#!/bin/sh\necho {reports}\n")
        os.chmod(p, 0o755)
    def fake_build(venv, pin):
        _shim(venv, pin)
        calls["build"] = calls.get("build", 0) + 1
    def fake_run(cmd, env):
        calls["cmd"], calls["env"] = cmd, env
    with tempfile.TemporaryDirectory() as d:
        assert ensure_transformers_pin("3a", "m", ["--row", "3a"], running="5.9.0", venv_root=d, build=fake_build, run=fake_run) is None
        sh = ensure_transformers_pin("3b", "m", ["--row", "3b"], running="5.9.0", venv_root=d, build=fake_build, run=fake_run)
        py = os.path.join(d, "venv-tf4.57.1", "bin", "python")
        assert sh["transformers"] == "4.57.1 (model-card pin; base env 5.9.0)" and calls["build"] == 1
        assert calls["cmd"] == [py, "-m", "m", "--row", "3b"] and json.loads(calls["env"]["MDL_HARNESS_SHIFT"]) == sh
        assert calls["env"]["MDL_TF_REEXEC"] == "1" and "MDL_TF_REEXEC" not in os.environ
        ensure_transformers_pin("3c-ER", "m", [], running="5.9.0", venv_root=d, build=fake_build, run=fake_run)
        assert calls["build"] == 1, "venv must be built once"
        assert ensure_transformers_pin("3b", "m", [], running="4.57.1", venv_root=d, build=fake_build, run=fake_run) is None
        os.environ["MDL_TF_REEXEC"] = "1"
        raised = False
        try:
            ensure_transformers_pin("3b", "m", [], running="5.9.0", venv_root=d, build=fake_build, run=fake_run)
        except AssertionError:
            raised = True
        del os.environ["MDL_TF_REEXEC"]
        assert raised, "a re-run that still sees the wrong version must fail loud"
    # The preflight's killing observation: a pin venv that cannot import the science stack (or that
    # carries the wrong transformers) must fail BEFORE the job is re-run in it, never after the weights
    # are on disk. Row 3b's venv had no torch and no datasets and this was found 15 GPU-minutes in.
    with tempfile.TemporaryDirectory() as d:
        for reports, why in ((f"5.9.0\nModuleNotFoundError", "wrong version in the venv"),
                             ("", "probe produced nothing")):
            calls.clear()
            raised = False
            try:
                ensure_transformers_pin("3b", "m", [], running="5.9.0", venv_root=os.path.join(d, why),
                                        build=lambda v, p, _r=reports: _shim(v, _r), run=fake_run)
            except AssertionError:
                raised = True
            assert raised, f"an unusable pin venv must fail loud ({why})"
            assert "cmd" not in calls, f"the job must NOT be re-run in an unusable pin venv ({why})"
    # 3c-Act is deliberately NOT on the Molmo2 pin — its own config.json declares transformers 5.3.0
    # while its siblings declare 4.56/4.57 (see MOLMOACT_PIN). This assert used to lump all three
    # together at 4.57.1, which is precisely the assumption that cost the row 13 cells twice.
    assert {ROWS[r]["transformers_pin"] for r in ("3b", "3c-ER")} == {"4.57.1"}, "Molmo2 rows pin 4.57.1"
    assert ROWS["3c-Act"]["transformers_pin"] == "5.9.0", "MolmoAct2 declares 5.3.0; runs on the AMI's 5.9.0"
    assert ROWS["3a"]["transformers_pin"] == "5.9.0"
    print(f"PASS matrix_models dry-run ({len(ROWS)} rows, {len(KNOWN)} verified repos, 2 blocked by licence; "
          f"binds frozen/perturbed/shared-prefix; kind↔delta band at import and in binds; install-row "
          f"embed/head/tokenizer structural zeros refused, vs_base measured; decoder swap; adapter merge; "
          f"§8b pin re-run)")


if __name__ == "__main__":
    if "--refresh-facts" in sys.argv:
        refresh_facts()
    elif "--audit" in sys.argv:
        from infra.pg import matrix_audit
        matrix_audit.main(sys.argv[1:])
    else:
        _dryrun()
