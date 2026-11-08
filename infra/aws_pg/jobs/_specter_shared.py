"""SPECTER shared job glue — freeze-assert mechanics, parent-artifact pointers, the `specter|`
item split, per-stage gpu_seconds (the §0 binding repair: EVERY driver asserts the field is
non-null before `green`), and the operator-batch builders every GPU arm shares.

Freeze mechanics mirror ATLAS: the control plane checks the per-arm tag
(`scripts/specter_check_freeze.py`, has git) before every launch; the driver half here
recomputes sha256 of the frozen spec copy shipped in the tarball (`specter_spec_frozen.md`)
against `specter_freeze.json` and refuses to run on mismatch. No retroactive tags.

§0.5 amendments discovered at implementation (journal E2, decided pre-freeze):
  A1  bispec_w banked only band METADATA (no Σ_y tensors) — Arm K re-harvests the input Grams
      on its own operator batch with the bispec_w hook recipe (uncentered x.T@x/n, fp64).
  A2  Arm K stiffness sides made concrete: input-space objects (U_unt_qk) read on the Σ_y side;
      residual-space objects (U_unt_residual, U_wr, task footprint) read on the F̄ side
      (spectral completion + direct-quadform subsample). R-K1 headline = U_unt_residual's
      F̄_dc-quantile at the L0 site.
  A3  banked-corpus q(cell) uses the EXPECTED functional over the cell's draw distribution
      (ATLAS cell seeds are not reproducible post-hoc: `hash()` in grammar_cell_ctx is
      PYTHONHASHSEED-dependent); SPECTER's own cells use the REALIZED construction.
  A4  C-1 legs frozen: ppl leg = the 6 UNIQUE coherent vectors (3 benign-disc + 3 mid-disc,
      discovery measurements); vqa leg = all 9 (the ATLAS R5/AM-3 disclosure). C-3 ratios
      recomputed on log(1+Δppl) means for BOTH sites (program damage scale).
  A5  gpu_seconds has NO ATLAS precedent (grep: nothing) — SPECTER introduces it, as §0's
      "spec-promise repair" language implies.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

from infra.aws_pg.jobs._atlas_shared import (  # noqa: F401  (re-exported for specter drivers)
    ATLAS_CORPUS,
    ATLAS_GATE0,
    BISPEC_GATE0,
    BISPEC_K,
    BISPEC_K_SEALED,
    BISPEC_SIGMA,
    BISPEC_V1R_ROWS,
    FUNC_BASE_PPL,
    PPL_TOL,
    SIGMA1_ANCHOR,
    build_v1,
    load_sigma_bank,
)
from infra.aws_pg.jobs._regrain_shared import (  # noqa: F401
    N_PPL_SEQ,
    S3_BUCKET,
    effect_from,
    effect_se,
    fetch,
    install_pg,
    pinned_pg1_sha,
)

# ---- parent artifacts (read-only; every load-bearing number RECOMPUTED from these) ----------
ATLAS_T = f"{S3_BUCKET}/results/atlas_t/out/metrics.json"
ATLAS_B_MASKS = f"{S3_BUCKET}/results/atlas_b/out/gate_masks.pt"
ATLAS_W_RHO = f"{S3_BUCKET}/results/atlas_w/out/rho_bank.pt"
ATLAS_M_FREEZE = f"{S3_BUCKET}/results/atlas_m_fit/out/m_freeze.json"
ATLAS_M_SEALED = f"{S3_BUCKET}/results/atlas_m_sealed/out/metrics.json"
ATLAS_L8_SIGMA = f"{S3_BUCKET}/results/atlas_l8/out/sigma_l8"   # the Gate-0e pipeline's real home
ATLAS_GATE0_JSON = f"{S3_BUCKET}/results/atlas_gate0/out/gate0.json"
COMWRITE_ANATOMY = f"{S3_BUCKET}/results/comwrite_a_anatomy2/out"
REGRAIN_G = f"{S3_BUCKET}/results/regrain_g/out/metrics.json"
SPECTER_GATE0 = f"{S3_BUCKET}/results/specter_gate0/out"
SPECTER_O = f"{S3_BUCKET}/results/specter_o/out"

_JOBS_DIR = pathlib.Path(__file__).resolve().parent
SPEC_FROZEN = _JOBS_DIR / "specter_spec_frozen.md"
FREEZE_MANIFEST = _JOBS_DIR / "specter_freeze.json"

SITES = ("L0", "L8", "L14")
LOSSES = ("ppl", "vqa")
N_OP_PPL = 24            # operator batch A: wikitext windows (12 per half) — frozen
N_OP_VQA = 24            # operator batch A: vqa items (12 per half) — frozen
N_OP_HELD = 12           # held-out batch B per loss — frozen


def assert_frozen(arm):
    """Driver half of freeze-or-refuse (the _atlas_shared.assert_frozen contract, verbatim)."""
    assert FREEZE_MANIFEST.exists(), f"no freeze manifest — arm '{arm}' refuses to launch"
    man = json.loads(FREEZE_MANIFEST.read_text())
    assert arm in man, f"arm '{arm}' has no freeze entry — refuses to launch (no retroactive tags)"
    assert SPEC_FROZEN.exists(), "frozen spec copy missing from the packaged tree"
    h = hashlib.sha256(SPEC_FROZEN.read_bytes()).hexdigest()
    assert h == man[arm]["spec_sha256"], (
        f"spec hash mismatch for arm '{arm}': tree={h[:12]} manifest={man[arm]['spec_sha256'][:12]}")
    return man[arm]


def specter_split(content_hashes, name="specter_vqa"):
    """Fresh `specter|` salt (spec §4.2) — pure sha1 bucketing on item content hashes."""
    import numpy as np

    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"specter|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


# ---------------------------------------------------------------------------
# gpu_seconds — per-stage, asserted non-null before green (§0 binding repair).
# ---------------------------------------------------------------------------
class StageTimer:
    def __init__(self, rep):
        self.rep = rep
        rep["gpu_seconds"] = {}

    def stage(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self._sync()
                self.t0 = time.time()

            def __exit__(self, *e):
                self._sync()
                timer.rep["gpu_seconds"][name] = (
                    timer.rep["gpu_seconds"].get(name, 0.0) + time.time() - self.t0)

            @staticmethod
            def _sync():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        return _Ctx()


def assert_gpu_seconds(rep):
    """The pre-green assert: the field exists, is non-empty, every value non-null/positive."""
    gs = rep.get("gpu_seconds")
    assert gs is not None and len(gs) > 0, "gpu_seconds missing — green refused (§0 mandate)"
    for k, v in gs.items():
        assert v is not None and float(v) >= 0.0, f"gpu_seconds[{k}] null — green refused"


def emit_family(rep, family_name, pvals, labels, q=0.10):
    """BH within a named family WITH the per-comparison p-values emitted (§0 binding repair ii).
    Returns the pass vector; writes rep['families'][family_name]."""
    from infra.pg.atlas_fit import bh_pass
    passed = bh_pass(list(pvals), q=q)
    rep.setdefault("families", {})[family_name] = {
        "q": q, "labels": list(labels), "p_values": [float(p) for p in pvals],
        "bh_pass": [bool(b) for b in passed]}
    return passed


# ---------------------------------------------------------------------------
# Operator batches (spec §4.1) — B=1 forwards, no padding, manifests content-hashed.
# ---------------------------------------------------------------------------
def _find_token_block(hay, needle):
    """First index of `needle` as a contiguous block in `hay`, or None (no exceptions —
    the verified-window data boundary below counts misses instead of raising)."""
    n = needle.shape[0]
    for pos in range(hay.shape[0] - n + 1):
        if bool((hay[pos:pos + n] == needle).all()):
            return pos
    return None


def wikitext_windows(proc, n_windows, window=512, skip=0, gray=128, verify_proc=False):
    """The invcore-FUNC window stream (same corpus walk, same text-block location), returned as
    construction dicts instead of scored. `skip` windows are consumed first (so batch B is
    disjoint from batch A by construction). `verify_proc=True` additionally runs the processor
    and drops windows whose exact token block does not survive the tokenize→decode→retokenize
    round trip (rare deep-corpus boundary effect; every drop counted in the returned dicts'
    shared meta) — needed by consumers that walk far past the ppl gate's first 120 windows."""
    import torch
    from datasets import load_dataset

    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    tok = proc.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blank = _blank_image(gray)
    out = []
    n_done = 0
    n_dropped = 0
    for start in range(0, ids.size(0), window):
        if len(out) >= n_windows:
            break
        chunk_ids = ids[start:start + window]
        if chunk_ids.size(0) < 8:
            break
        window_text = tok.decode(chunk_ids.tolist())
        win_ids = tok(window_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if int(win_ids.shape[0]) < 8:
            continue
        n_done += 1
        if n_done <= skip:
            continue
        if verify_proc:
            inp = proc(text=[window_text], images=[blank], return_tensors="pt", padding=False)
            if _find_token_block(inp["input_ids"][0].cpu(), win_ids) is None:
                n_dropped += 1
                continue
        out.append({"text": window_text, "win_ids": win_ids, "blank": blank,
                    "hash": hashlib.sha1(window_text.encode()).hexdigest(),
                    "n_dropped_unlocatable": n_dropped})
    assert len(out) == n_windows, f"corpus exhausted: {len(out)} < {n_windows}"
    return out


def build_ppl_item(model, proc, device, wdict, install, holder):
    """One ppl-loss operator item (spec §2 instrument identity): blank-image forward, mean-CE
    weights over the located text block, labels for the hessian leg."""
    import torch

    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    inp = proc(text=[wdict["text"]], images=[wdict["blank"]], return_tensors="pt",
               padding=False).to(device)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    in_ids = inp["input_ids"][0]
    win_ids = wdict["win_ids"]
    n_text = int(win_ids.shape[0])
    pos = _locate_text_block(in_ids.cpu(), win_ids)
    T = int(in_ids.shape[0])
    n_t = n_text - 1
    loss_w = torch.zeros(T, dtype=torch.float32)
    loss_w[pos:pos + n_t] = 1.0 / n_t
    labels = torch.full((T,), -100, dtype=torch.long)
    labels[pos:pos + n_t] = win_ids[1:]

    def forward_fn():
        with install():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"]).logits
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": loss_w.to(device),
            "labels": labels, "T": T, "hash": wdict["hash"], "loss": "ppl"}


def majority_answer(row):
    from collections import Counter
    answers = [a["answer"] for a in row["answers"]]
    return Counter(answers).most_common(1)[0][0]


def build_vqa_item(model, proc, device, row, task, install, holder):
    """One vqa-CE operator item: teacher-forced suffix answer (PaliGemma labels convention)."""
    import torch
    img = task.image_fn(row).convert("RGB")
    prompt = task.prompt_fn(row)
    # `majority_answer` reads the VQAv2 answer schema (list of {"answer": str}). TextVQA gives a
    # list of plain strings and COCO-captions gives references, so a task-generic caller crashed
    # with "string indices must be integers" the moment a non-VQAv2 task reached here. Every
    # TaskSpec exposes `refs_fn`, and `task` was already in scope — it simply was not used for the
    # target. The VQAv2 branch is kept byte-identical so existing SPECTER/CARRY numbers reproduce.
    from collections import Counter
    _a = row.get("answers") if hasattr(row, "get") else None
    if isinstance(_a, list) and _a and isinstance(_a[0], dict):
        ans = majority_answer(row)                 # VQAv2 — byte-identical to the original path
    elif isinstance(_a, list) and _a:
        # TextVQA: a list of plain answer strings. `refs_fn` is None for non-corpus tasks, so it
        # cannot be the general answer — the row schema has to be read directly.
        _v = [str(x) for x in _a if str(x).strip()]
        assert _v, f"{task.name}: empty answers list"
        ans = Counter(_v).most_common(1)[0][0]
    else:
        assert task.refs_fn is not None, (
            f"{task.name}: row has no 'answers' and the task exposes no refs_fn — there is no "
            f"teacher-forcing target for this item")
        _r = task.refs_fn(row)
        _r = [x for x in (_r if isinstance(_r, (list, tuple)) else [_r])
              if isinstance(x, str) and x.strip()]
        assert _r, f"{task.name}: refs_fn returned no usable target"
        ans = Counter(_r).most_common(1)[0][0]
    inp = proc(text=[prompt], images=[img], suffix=[ans], return_tensors="pt",
               padding=False).to(device)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    labels = inp["labels"][0].cpu()
    T = int(inp["input_ids"].shape[1])
    live = labels != -100
    n_live = int(live.sum())
    assert n_live >= 1, "teacher-forced answer produced no label tokens"
    # shifted-CE convention: logits at t predict labels[t+1]
    loss_w = torch.zeros(T, dtype=torch.float32)
    shift_labels = torch.full((T,), -100, dtype=torch.long)
    shift_labels[:-1] = labels[1:]
    lw = shift_labels != -100
    loss_w[lw] = 1.0 / int(lw.sum())

    def forward_fn():
        with install():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"]).logits
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": loss_w.to(device),
            "labels": shift_labels, "T": T, "loss": "vqa"}


def batch_manifest(items_meta):
    blob = json.dumps(items_meta, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def safe_log1p_dppl(mean_nll, base_ppl):
    """log(1 + (e^nll − base)) without overflow: exactly nll for nll ≥ 30 (error < 1e-12)."""
    import math
    if mean_nll >= 30.0:
        return float(mean_nll)
    return math.log1p(max(math.exp(mean_nll) - base_ppl, -0.99))


def safe_ppl_gate(model, proc, device, n_seq=N_PPL_SEQ, _cache={}):
    """The blank-image wikitext instrument with a SATURATION-SAFE readout: same windows, same
    text-block location, same shifted-CE as `_ppl_blankimg`, but returns mean_nll and never
    exponentiates an obliterated model into an OverflowError (Arm S's first spike cell measured
    mean NLL > 709 — a legitimate reading the parent instrument cannot represent). Windows are
    built once per (proc id, n_seq) and cached."""
    import contextlib

    import torch
    key = (id(proc), n_seq)
    if key not in _cache:
        _cache[key] = wikitext_windows(proc, n_seq)

    @contextlib.contextmanager
    def _noop():
        yield
    nll_sum, n_tok = 0.0, 0
    for wd in _cache[key]:
        item = build_ppl_item(model, proc, device, wd, _noop, {})
        with torch.no_grad():
            logits = item["forward_fn"]()[0]
        lab = item["labels"].to(logits.device)
        live = lab != -100
        lse = torch.logsumexp(logits[live].float(), dim=-1)
        nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
        nll_sum += float(nll.sum())
        n_tok += int(live.sum())
    mean_nll = nll_sum / n_tok
    import math
    return {"mean_nll": mean_nll, "n_tokens": n_tok,
            "ppl": math.exp(min(mean_nll, 30.0)), "saturated": bool(mean_nll >= 30.0)}


class SpecterCellRunner:
    """The _atlas_shared.CellRunner contract with the saturation-safe ppl leg (mean-NLL readout;
    dead cells are measurements, not crashes). Same record fields plus mean_nll/saturated_ppl."""

    def __init__(self, model, proc, task, items, intact, floor, base_ppl, rep, writer,
                 n_ppl_seq=N_PPL_SEQ):
        self.model, self.proc, self.task = model, proc, task
        self.items, self.intact, self.floor, self.base_ppl = items, intact, floor, base_ppl
        self.rep, self._w, self.n_ppl_seq = rep, writer, n_ppl_seq

    def run(self, name, ctx_factory, *, family, arm, draw=0, skip_vqa=False, **extra):
        from infra.pg import inherit_pg1 as IP
        r = None if skip_vqa else IP.eval_pg1(self.model, self.proc, self.task, self.items,
                                              "cuda", ctx=ctx_factory())
        with ctx_factory():
            pg = safe_ppl_gate(self.model, self.proc, "cuda", n_seq=self.n_ppl_seq)
        rec = {"cell": name, "family": family, "arm": arm, "draw": draw,
               "ppl": pg["ppl"], "mean_nll": pg["mean_nll"],
               "saturated_ppl": pg["saturated"],
               "dppl": pg["ppl"] - self.base_ppl,
               "log1p_dppl": safe_log1p_dppl(pg["mean_nll"], self.base_ppl), **extra}
        if r is not None:
            from infra.aws_pg.jobs._regrain_shared import effect_from, effect_se
            span = self.intact["score"] - self.floor
            assert span > 1e-3, "no dynamic range vs the dead floor — halt"
            rec.update({"vqa": r["score"], "dvqa": r["score"] - self.intact["score"],
                        "effect": effect_from(r["per_item"], self.intact["per_item"], self.floor),
                        "effect_se": effect_se(r["per_item"], self.intact["per_item"], self.floor),
                        "saturated_dead": bool((r["score"] - self.floor) / span < 0.02)})
        self.rep.setdefault("cells", []).append(rec)
        self._w()
        print(f"[specter] {name} draw{draw}: log1p_dppl={rec['log1p_dppl']:+.3f}"
              + (f" dVQA={rec['dvqa']:+.4f}" if r is not None else "")
              + (" SATURATED" if rec["saturated_ppl"] else ""), flush=True)
        return rec


def clean_forward(item, d, device="cuda"):
    """Forward with the injector's delta EXPLICITLY zeroed. The holder is shared across probes,
    so a bare item['forward_fn']() after operator work would silently carry the LAST probe field
    into a 'clean' capture — this is the one sanctioned way to run a clean forward on an
    operator item."""
    import torch
    item["holder"]["delta"] = torch.zeros(item["T"], d, device=device)
    return item["forward_fn"]()


# ---------------------------------------------------------------------------
# The operator bank — the first-class deliverable other programs consume.
# ---------------------------------------------------------------------------
def save_operator_bank(out_path, bank):
    """bank: {(site, mode, loss): {evals, evecs fp32, resid, traces_sb, tr_h2, ...}} plus
    site-level rows. Content-hashed on save."""
    import torch
    torch.save(bank, out_path)
    return hashlib.sha256(pathlib.Path(out_path).read_bytes()).hexdigest()
