"""STAGEDIFF roster, loading and the forward condition — spec §2 (studies/stagediff/2026-09-04_SPEC.md).

ROSTER    the seven checkpoints C0..C6 of the Qwen3-4B-Instruct → Molmo2 → MolmoAct2 lineage plus P0
          (`Qwen/Qwen3-4B`, the base the released Neuronpedia lens was fitted on; preflight §9.1 only).
          Every repo carries the HF revision sha it was pinned at (HF API, 2026-09-04), its architecture,
          whether it needs remote code, its transformers pin, and its BOS rule.
BOS rule  `tokenizer(text, add_special_tokens=True)` adds NOTHING on any Qwen2Tokenizer of this lineage
          (asserted at load). The Molmo2/MolmoAct2 processors insert `bos_token_id` = 151645
          (`<|im_end|>`, `processing_molmo2.insert_bos`) before every input, so C1..C6 are run with that
          id prepended (`bos_id`), and C0/P0 — never trained with one — without. Within every adjudicated
          hop (H3, H4, H5, H5′) BOS presence is therefore equal; H1 (C0→C1) differs at position 0 and is
          reported, not adjudicated (§6.2). `--bos-ablation` in the battery measures the other setting.
Forward   `StageForward(TextForward)`: text tokenizer, optional BOS prepend, LEFT padding, content-relative
          `position_ids`, `logits_to_keep=1`; logits are sliced to the shared prefix `V_SHARED` = 151 669
          by every caller before an argmax or a rank.
Loading   mirrors `matrix_models._load_model` (bf16, sdpa, the pinned snapshot only) without touching
          `matrix_models.KNOWN`, whose facts audit spans every other study on this branch; the reused
          pieces (`HF_ALLOW`, `shared_prefix`, `vocab_sha256`, `_build_pin_venv`, `PIN_VENV_PROBE`) are
          imported from there.

Fail-loud; heavy imports inside functions (py_compiles without torch/transformers).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

from infra.pg.matrix_models import (HF_ALLOW, PIN_VENV_PROBE, _build_pin_venv, sha256_file, shared_prefix,
                                    transformers_version, vocab_sha256)

V_SHARED = 151669          # ids 0..151668 map to the SAME STRINGS in every tokenizer of the lineage: Qwen3's 151 643 BPE
                           # tokens + its 26 specials; Molmo2 appends its extras at 151 669+, MolmoAct2 at 151 669+
                           # (<setup_start>). 151 936 is the EMBEDDING row count (rows 151 669..151 935 are unused in
                           # Qwen3) — computed offline 2026-09-04 from the three tokenizer files, banked per job as
                           # `prefix_vocab_sha256` and asserted equal across checkpoints by the verdict.
MOLMO_BOS = 151645         # <|im_end|>: what processing_molmo2.insert_bos inserts (bos_token_id or eos_token_id)
N_LAYERS = 36
D_MODEL = 2560
BAND = tuple(range(12, 32))     # §6.7 block outputs 12..31 of 36 (33 %–89 %)
MOTOR = tuple(range(32, 36))
TF_BASE = "5.9.0"
TF_MOLMO2 = "4.57.1"

_F = ("repo", "sha", "arch", "stage", "pin", "remote", "bos_id")
ROSTER = {k: dict(zip(_F, v)) for k, v in {
    "C0": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "Qwen3ForCausalLM",
           "LLM parent (Molmo2-4B model card names it)", TF_BASE, False, None),
    "C1": ("allenai/Molmo2-4B", "042abfa7a38879a376cec03d949eff0aefaa0600", "Molmo2ForConditionalGeneration",
           "VLM install (SigLIP2 + connector)", TF_MOLMO2, True, MOLMO_BOS),
    "C2": ("allenai/Molmo2-ER", "dab22564403d2607855bb1fffb0721285b445081", "Molmo2ForConditionalGeneration",
           "embodied specialise 20k + rehearse 1.5k", TF_MOLMO2, True, MOLMO_BOS),
    "C3": ("allenai/MolmoAct2-Pretrain", "a05effca9ba36c1177359b42a9d5d7a4568dbe3c", "MolmoAct2ForConditionalGeneration",
           "discrete-action pretrain 200K, +2688 vocab rows", TF_BASE, True, MOLMO_BOS),
    "C4": ("allenai/MolmoAct2", "e432d85f6e039edca44afb93c262f3084ab72a9c", "MolmoAct2ForConditionalGeneration",
           "post-train, action expert, VLM mix, insulation on, 100K", TF_BASE, True, MOLMO_BOS),
    "C5": ("allenai/MolmoAct2-LIBERO", "0d24a92bd1faf321ef497c3bbd5681af97c65aa2", "MolmoAct2ForConditionalGeneration",
           "robot-only, insulation off, head + final norm tuned, 50K", TF_BASE, True, MOLMO_BOS),
    "C6": ("allenai/MolmoAct2-DROID", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb", "MolmoAct2ForConditionalGeneration",
           "robot-only, no language annotations, 100K", TF_BASE, True, MOLMO_BOS),
    "P0": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "Qwen3ForCausalLM",
           "preflight base: the released Neuronpedia lens was fitted on it", TF_BASE, False, None),
}.items()}
HOPS = {"H1": ("C0", "C1"), "H2": ("C1", "C2"), "H3": ("C2", "C3"), "H4": ("C3", "C4"),
        "H5": ("C4", "C5"), "H5p": ("C4", "C6")}
ADJUDICATED_HOPS = ("H3", "H4", "H5", "H5p")
RELEASED_LENS = {"repo": "neuronpedia/jacobian-lens", "sha": "0731326edff4ae730ffc5356fe1a4728c748b3a6",
                 "dir": "qwen3-4b/jlens/Salesforce-wikitext", "file": "Qwen3-4B_jacobian_lens.pt",
                 "base": "Qwen/Qwen3-4B"}


def weights_dir(cid: str) -> str:
    return "models--" + ROSTER[cid]["repo"].replace("/", "--")


# ------------------------------------------------------------------------------------- pin venv
def ensure_pin(cid: str, module: str, argv, venv_root="/opt/pg") -> dict | None:
    """`matrix_models.ensure_transformers_pin` for a ROSTER id: None when the running transformers equals the
    checkpoint's pin; otherwise build `/opt/pg/venv-tf<pin>` once (`_build_pin_venv`), probe it, re-run
    `python -m <module> <argv>` in it as a child and return the shift record (the caller then returns).
    Call BEFORE importing transformers."""
    pin = ROSTER[cid]["pin"]
    running = transformers_version()
    if running == pin:
        return None
    assert os.environ.get("MDL_TF_REEXEC") != "1", \
        f"already re-run under the pin venv, but transformers is {running} (pin {pin}) — venv broken"
    venv = os.path.join(venv_root, f"venv-tf{pin}")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        _build_pin_venv(venv, pin)
    probe = subprocess.run([py, "-c", f"import {', '.join(PIN_VENV_PROBE)}; import transformers as _t; print(_t.__version__)"],
                           capture_output=True, text=True)
    got = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    assert probe.returncode == 0 and got == pin, (
        f"pin venv {venv} cannot run the job: importing {PIN_VENV_PROBE} gave transformers {got!r} "
        f"(want {pin!r}, rc={probe.returncode})\n{probe.stderr.strip()[-2000:]}")
    shift = {"transformers": f"{pin} (model-card pin; base env {running})", "venv": venv, "module": module}
    env = {**os.environ, "MDL_TF_REEXEC": "1", "MDL_HARNESS_SHIFT": json.dumps(shift)}
    print(f"[pin] transformers {running} → {pin}: re-running {module} under {venv}", flush=True)
    subprocess.run([py, "-m", module, *argv], check=True, env=env)
    return shift


# ------------------------------------------------------------------------------------- snapshot + hashes
def snapshot(cid: str) -> str:
    """Local snapshot dir of the PINNED revision: safetensors + configs + tokenizer (+ remote code) only.
    Offline when the S3 cache holds the repo (run_job.sh synced it), a Hub download otherwise."""
    from huggingface_hub import snapshot_download
    r = ROSTER[cid]
    return snapshot_download(r["repo"], revision=r["sha"], allow_patterns=HF_ALLOW)


def repo_hashes(cid: str, local: str) -> dict:
    out = {"repo": ROSTER[cid]["repo"], "sha": ROSTER[cid]["sha"], "local": local}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(local, name)
        out[name] = sha256_file(p) if os.path.exists(p) else None
    assert out["config.json"], f"{cid}: no config.json in {local}"
    return out


def prefix_vocab_sha256(tok, n: int = V_SHARED) -> str:
    """sha256 over the token strings of ids 0..n-1 — the forward-condition parity bind every job banks;
    the verdict asserts it is identical across checkpoints (B1 of the spec's §4)."""
    inv = {i: t for t, i in tok.get_vocab().items()}
    missing = [i for i in range(n) if i not in inv]
    assert not missing, f"vocab lacks {len(missing)} ids below {n}, first {missing[:3]}"
    h = hashlib.sha256()
    for i in range(n):
        h.update(inv[i].encode("utf-8", "surrogatepass")); h.update(b"\x00")
    return h.hexdigest()


# ------------------------------------------------------------------------------------- forward condition
def _text_forward_cls():
    from infra.pg.matrix_forward import TextForward, content_position_ids

    class StageForward(TextForward):
        """`TextForward` + the roster's BOS rule. `_encode`: per-text raw ids (`add_special_tokens=False`; the
        lineage's tokenizers add nothing either way, asserted at load), `bos_id` prepended when set, then LEFT padding built here
        so the BOS sits at the first real position and `content_position_ids` gives it position 0 — exactly
        the Molmo processor's own layout (`insert_bos` at the first valid index)."""
        shell = "text"

        def __init__(self, model, tokenizer, device="cuda", bos_id=None):
            super().__init__(model, tokenizer, device)
            self.bos_id = bos_id

        def encode_rows(self, texts):
            rows = [list(self.tok.encode(t, add_special_tokens=False)) for t in texts]   # raw ids; BOS is OUR rule
            if self.bos_id is not None:
                rows = [[int(self.bos_id)] + r for r in rows]
            return rows

        def _encode(self, texts, images=None):
            import torch
            assert images is None or all(im is None for im in images), "text shell takes no images"
            rows = self.encode_rows(list(texts))
            S = max(len(r) for r in rows)
            pad = int(self.tok.pad_token_id)
            ids = torch.full((len(rows), S), pad, dtype=torch.long)
            am = torch.zeros((len(rows), S), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, S - len(r):] = torch.tensor(r, dtype=torch.long)
                am[i, S - len(r):] = 1
            return {"input_ids": ids.to(self.device), "attention_mask": am.to(self.device)}

        def _call_kwargs(self, enc):
            # use_cache=False explicitly: MolmoAct2's remote code (config 5.3.0) resolves a None to
            # `self.config.use_cache`, which its config lacks under transformers 5.9.0 (W0 smoke-C4/2 died there).
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "position_ids": content_position_ids(enc["attention_mask"]), "use_cache": False}

    return StageForward


def load_ckpt(cid: str, device: str = "cuda", attn: str = "sdpa", batch_size: int = 8) -> dict:
    """One resident checkpoint as a member dict: model (bf16, eval, grads off), tok, fwd (StageForward),
    an (lm_anatomy), bos_id, hashes. Asserts: the anatomy resolves (36 layers, 2560 wide), the tokenizer adds
    no specials of its own, the shared prefix covers V_SHARED ids, `lm_head` has ≥ V_SHARED rows."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    import transformers
    from infra.pg.matrix_forward import lm_anatomy
    r = ROSTER[cid]
    local = snapshot(cid)
    kw = dict(dtype=torch.bfloat16, device_map=device, attn_implementation=attn)
    if r["remote"]:
        model = AutoModelForImageTextToText.from_pretrained(local, trust_remote_code=True, **kw)
    else:
        assert hasattr(transformers, r["arch"]), f"{cid}: transformers {transformers.__version__} has no {r['arch']}"
        model = getattr(transformers, r["arch"]).from_pretrained(local, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=r["remote"])
    an = lm_anatomy(model)
    assert an["n_layers"] == N_LAYERS and len(an["layers"]) == N_LAYERS, an["n_layers"]
    assert int(an["lm_head"].weight.shape[1]) == D_MODEL, an["lm_head"].weight.shape
    assert int(an["lm_head"].weight.shape[0]) >= V_SHARED, an["lm_head"].weight.shape
    probe = "The quick brown fox."
    assert tok.encode(probe, add_special_tokens=True) == tok.encode(probe, add_special_tokens=False), \
        f"{cid}: tokenizer adds specials of its own; the BOS rule assumes it adds none"
    assert tok.pad_token_id is not None, f"{cid}: no pad token"
    if r["bos_id"] is not None:
        assert tok.convert_ids_to_tokens(int(r["bos_id"])) == "<|im_end|>", tok.convert_ids_to_tokens(int(r["bos_id"]))
    fwd = _text_forward_cls()(model, tok, device, bos_id=r["bos_id"])
    fwd.batch_size = batch_size
    hashes = repo_hashes(cid, local)
    hashes["vocab_sha256"] = vocab_sha256(tok)
    hashes["prefix_vocab_sha256"] = prefix_vocab_sha256(tok)
    return {"id": cid, "repo": r["repo"], "sha": r["sha"], "stage": r["stage"], "model": model, "tok": tok,
            "fwd": fwd, "an": an, "bos_id": r["bos_id"], "local": local, "hashes": hashes,
            "dtype": str(model.dtype), "attn": attn, "transformers": transformers.__version__,
            "n_lm_head_rows": int(an["lm_head"].weight.shape[0])}


def assert_shared_prefix(tok_a, tok_b) -> dict:
    sp = shared_prefix(tok_a, tok_b)
    assert sp["shared_prefix"] >= V_SHARED, f"shared prefix {sp['shared_prefix']} < {V_SHARED}"
    return sp


def free(member: dict):
    import gc
    import torch
    for k in ("fwd", "model", "tok", "an"):
        member[k] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------- dry-run
def stub_member(bos_id=None, seed=0):
    """A 4-layer stub member (`matrix_forward.stub_text_model` + `_StubTok`) shaped like a real one so every
    NEW module's dry-run runs the real plumbing weight-free. n_layers=4, d=16, vocab=256."""
    from infra.pg.matrix_forward import _StubTok, lm_anatomy, stub_text_model
    m = stub_text_model(seed=seed)
    tok = _StubTok()
    an = lm_anatomy(m)
    fwd = _text_forward_cls()(m, tok, device="cpu", bos_id=bos_id)
    fwd.batch_size = 4
    return {"id": f"STUB{seed}", "repo": "stub", "sha": "0" * 40, "stage": "stub", "model": m, "tok": tok,
            "fwd": fwd, "an": an, "bos_id": bos_id, "local": None, "hashes": {}, "dtype": "float32",
            "attn": "stub", "transformers": "stub", "n_lm_head_rows": 256}


def _dryrun():
    import torch
    for cid, r in ROSTER.items():
        assert len(r["sha"]) == 40 and r["pin"] in (TF_BASE, TF_MOLMO2), cid
        assert (r["bos_id"] is None) == (cid in ("C0", "P0")), cid
    for h, (p, c) in HOPS.items():
        assert p in ROSTER and c in ROSTER, h
    for h in ADJUDICATED_HOPS:      # BOS presence equal within every adjudicated hop (§2 forward condition)
        p, c = HOPS[h]
        assert (ROSTER[p]["bos_id"] is None) == (ROSTER[c]["bos_id"] is None), h
    assert (ROSTER["C0"]["bos_id"] is None) != (ROSTER["C1"]["bos_id"] is None)   # H1 differs at position 0, stated
    m0 = stub_member(bos_id=None)
    m1 = stub_member(bos_id=1)
    ps = ["Q: 1+1?\nAnswer:", "Q: capital of France? Answer:"]
    e0, e1 = m0["fwd"]._encode(ps), m1["fwd"]._encode(ps)
    assert e0["input_ids"].shape[1] + 1 == e1["input_ids"].shape[1]
    real0 = e0["input_ids"][0][e0["attention_mask"][0] == 1].tolist()
    real1 = e1["input_ids"][0][e1["attention_mask"][0] == 1].tolist()
    assert real1 == [1] + real0, (real0, real1)                     # BOS is exactly the prepend
    assert (e1["attention_mask"][0] == 0).sum() > 0 and e1["attention_mask"][0, -1] == 1   # left pad
    from infra.pg.matrix_forward import content_position_ids
    pos = content_position_ids(e1["attention_mask"])
    assert pos[0][e1["attention_mask"][0] == 1][0].item() == 0                    # BOS at position 0
    lg = m1["fwd"].next_token_logits(ps)
    assert lg.shape == (2, 256) and torch.isfinite(lg).all()
    # batch-composition invariance survives the BOS prepend
    lg_single = m1["fwd"].next_token_logits(ps[:1])
    assert torch.allclose(lg[:1], lg_single, atol=1e-4), (lg[:1] - lg_single).abs().max()
    cl = m1["fwd"].continuation_logprobs([ps[0], ps[0]], ["Paris", "Rome"])
    assert len(cl) == 2 and all(n >= 1 for _, n in cl)
    sha = prefix_vocab_sha256(m0["tok"], n=20)
    assert len(sha) == 64 and sha == prefix_vocab_sha256(m1["tok"], n=20)
    assert weights_dir("C5") == "models--allenai--MolmoAct2-LIBERO"
    print("PASS stagediff_models dry-run (roster, hops, BOS rule, StageForward encode/positions/invariance, prefix sha)")


if __name__ == "__main__":
    _dryrun()
