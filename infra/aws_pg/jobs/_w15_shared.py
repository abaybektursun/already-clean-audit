"""W15 shared GPU harness — the genuinely NEW instrument the carrier gate needs and the
prior pipeline lacks. e4_capability / eq1 forward in inference_mode; text_eval.mmlu_blankimg
forwards in inference_mode — NONE of them backprop. W15 builds, ONCE (prereg §3, "build the
instrument once"), the per-decoder-matrix gradient ∂loss/∂W_before, then the install and
retention arms project that same gradient into their respective SVD bases.

Everything model-touching lives here (GPU + transformers); the pure projection math lives in
infra/pg/w15_carrier.py (laptop-unit-tested). The jobs (w15_install/retention/preflight) are
thin orchestration over these two.

Reuses the frozen e4/eq1 spine verbatim (build_pair, _decoder_keys, _set_decoder, _generate,
the image-content-hash split guard). No try/except around the science — a wrong assumption
RAISES with a trace (CLAUDE.md).
"""
from __future__ import annotations

import hashlib

from infra.aws_pg.jobs.e4_capability import SEED, _eval_rows, _generate, _per_item_scores
from infra.aws_pg.jobs.e4_compress_curve import _decoder_keys, _set_decoder


# ===========================================================================
# Install corpus (RefCOCO-val, held out from the testA eval by image content-hash).
# Mirrors eq1_trained_adapter._install_items for the single refcoco corpus.
# ===========================================================================

def img_hash(img):
    """32×32 grayscale sha1 — the eq1/e4 content hash (catches the same image across resizes)."""
    return hashlib.sha1(img.convert("L").resize((32, 32)).tobytes()).hexdigest()


def refcoco_install_items(B, n, seed, exclude_hashes):
    """Held-out RefCOCO-val (task, row) install rows whose image-hash is NOT in `exclude_hashes`
    (the testA eval images). Enforces install↔eval disjointness BY CONSTRUCTION (the systemic
    W3/W4/W5 split-bug guard). Logs dropped count — no silent truncation."""
    from datasets import load_dataset
    task = B.TASKS["refcoco"]
    ds = load_dataset(task.dataset, split="val").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    items, hashes, dropped = [], set(), 0
    for i in range(len(ds)):
        r = ds[i]
        h = img_hash(task.image_fn(r))
        if h in exclude_hashes:
            dropped += 1
            continue
        items.append({"t": "refcoco", "row": r})
        hashes.add(h)
    print(f"[w15] refcoco install: kept {len(items)}, dropped {dropped} eval-overlapping (hash-disjoint)",
          flush=True)
    return items, hashes


def split_audit(B, install_hashes, eval_rows, out):
    """HALT if any testA eval image-hash appears in the install corpus (eq1._split_audit verbatim)."""
    import json
    audit = {"install_n_hashes": len(install_hashes), "tasks": {}}
    overlap_total = 0
    for tname, rows in eval_rows.items():
        task = B.TASKS[tname]
        eh = {img_hash(task.image_fn(r)) for r in rows}
        ov = eh & install_hashes
        audit["tasks"][tname] = {"eval_n_hashes": len(eh), "overlap": len(ov)}
        overlap_total += len(ov)
    audit["overlap_total"] = overlap_total
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))
    assert overlap_total == 0, f"SPLIT LEAK: {overlap_total} eval image(s) in install corpus — HALT. {audit}"
    print(f"[w15] split_audit OK: 0 overlap ({len(install_hashes)} install hashes)", flush=True)


# ===========================================================================
# Cached ΔW SVD (the INSTALL basis) — mirrors e4._build_edit_family / eq1._decoder_delta_svd.
# ===========================================================================

def cached_dw_svd(lm_after, lm_before, dkeys, device, q):
    """Truncated randomized SVD of ΔW = W_after − W_before per decoder matrix (q+256 oversample,
    niter=8 — BIT-IDENTICAL to e4/eq1). Returns {key: (U[:,:q], S[:q], V[:,:q])} on CPU. The
    selection grafts and the leverage scores both consume THIS one basis (build once)."""
    import torch
    torch.manual_seed(SEED)
    after_sd = lm_after.state_dict()
    base_sd = lm_before.state_dict()
    svd = {}
    for k in dkeys:
        assert k in base_sd and k in after_sd, f"missing decoder key {k}"
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)
        qq = min(q + 256, min(dW.shape))
        U, S, V = torch.svd_lowrank(dW, q=qq, niter=8)
        svd[k] = (U[:, :q].cpu(), S[:q].cpu(), V[:, :q].cpu())
        del dW, U, S, V
        if device == "cuda":
            torch.cuda.empty_cache()
    return svd


# ===========================================================================
# Selection graft — generalises e4._rankk to an arbitrary per-matrix direction index set.
# ===========================================================================

def selection_graft(base_sd, svd, idx_by_key, dkeys, device):
    """Build the edited-decoder override dict for _set_decoder: for each matrix,
    W = W_before + Σ_{i∈idx} σ_i u_i v_iᵀ over the SELECTED directions idx_by_key[key]
    (a long tensor / list). idx = [0..r-1] ⇒ the energy rank-r graft (≡ e4._rankk). bf16 deploy.
    Uses w15_carrier.graft_from_indices so the selection math is the laptop-tested one."""
    import torch
    from infra.pg import w15_carrier as C
    d = {}
    for key in dkeys:
        U, S, V = svd[key]                                  # CPU fp32 factors
        idx = torch.as_tensor(idx_by_key[key], dtype=torch.long)
        # graft on CPU (base_sd weights are on GPU, the SVD factors are on CPU): keep the
        # fp64 reconstruction off-device, then ship the bf16 result to the model's device.
        W = C.graft_from_indices(base_sd[key].detach().float().cpu(), U, S, V, idx)   # fp64, CPU
        d[key] = W.to(device).to(torch.bfloat16)
    return d


def energy_idx_by_key(svd, dkeys, r):
    """idx_by_key for the ENERGY ordering: top-r by σ = [0..r-1] per matrix (clamped to that
    matrix's available directions). The other orderings come from scored selections in the job."""
    from infra.pg import w15_carrier as C
    return {k: C.energy_indices(r, svd[k][1].shape[0]) for k in dkeys}


# ===========================================================================
# The backward instrument — ∂(task loss)/∂W for each decoder matrix at the model's current weights.
# ===========================================================================

def _refcoco_batches(B, proc, items, train_batch):
    """Encode held-out RefCOCO install items into teacher-forcing batches: proc(text=prompt,
    images=img, suffix=GT-<loc>) → input_ids/attention_mask/pixel_values/labels (labels mask the
    <loc> answer tokens; −100 elsewhere). Mirrors eq1's collate, GT (not teacher) suffix."""
    import torch
    task = B.TASKS["refcoco"]
    batches = []
    for s in range(0, len(items), train_batch):
        chunk = items[s:s + train_batch]
        imgs = [task.image_fn(it["row"]).convert("RGB") for it in chunk]
        prompts = [task.prompt_fn(it["row"]) for it in chunk]
        suffixes = [task.target_fn(it["row"]) for it in chunk]      # GT <loc> string
        enc = proc(text=prompts, images=imgs, suffix=suffixes, return_tensors="pt", padding="longest")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        batches.append(enc)
    return batches


def refcoco_ce(pg, enc, device):
    """−log P(GT <loc> tokens | image, prompt), the RefCOCO decision-token CE, grad-enabled.
    The standard shift-by-one LM loss on the suffix-masked labels (labels != −100). Returns a
    scalar with a backward graph into the decoder weights."""
    import torch
    import torch.nn.functional as F
    ii = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    pv = enc["pixel_values"].to(device)
    labels = enc["labels"].to(device)
    logits = pg(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    assert int(mask.sum()) > 0, "no <loc> answer tokens in batch (suffix/labels broken)"
    return F.cross_entropy(logits[mask].float(), tgt[mask])


def refcoco_ce_whole(pg, enc, device, image_token_index):
    """The WHOLE-sequence LM CE (LoRA-GA honesty arm, §5): predict EVERY text token (prompt text +
    GT <loc> suffix), masking only image tokens + pad. Its gradient is the whole-loss gradient that
    LoRA-GA uses — if the decision-token leverage ties THIS within CI, the novelty collapses to known
    prior art (§6). Grad-enabled scalar."""
    import torch
    import torch.nn.functional as F
    ii = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    pv = enc["pixel_values"].to(device)
    logits = pg(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
    lab = ii.clone()
    lab[ii == image_token_index] = -100                    # image tokens are not predicted
    lab[am == 0] = -100                                    # pad
    tgt = lab[:, 1:]
    mask = tgt != -100
    assert int(mask.sum()) > 0, "no text tokens to predict (whole-loss mask broke)"
    return F.cross_entropy(logits[mask].float(), tgt[mask])


def decoder_weight_grad(pg, paths, dkeys, batches, loss_fn, device):
    """∂(mean loss)/∂W for each MATRIX in dkeys, at the model's CURRENT decoder weights. Enables
    grad ONLY on the 182 decoder weight matrices (everything else frozen → no leak), accumulates
    the gradient over `batches` (mean), reads .grad. Returns ({key: G (out,in) fp64 CPU}, mean_loss).
    Fail-loud on non-finite loss or a missing grad (broken graph)."""
    import torch
    lm = pg.get_submodule(paths["language_model"])
    for p in pg.parameters():
        p.requires_grad_(False)
    params = {}
    for k in dkeys:
        w = lm.get_parameter(k)
        w.requires_grad_(True)
        params[k] = w
    pg.zero_grad(set_to_none=True)
    pg.eval()                                               # no dropout; eval() does NOT disable grad
    nb = len(batches)
    tot = 0.0
    for enc in batches:
        loss = loss_fn(pg, enc, device) / nb
        assert torch.isfinite(loss), f"non-finite W15 loss: {loss.item()}"
        loss.backward()
        tot += float(loss.item())
    grads = {}
    for k, w in params.items():
        assert w.grad is not None, f"no grad for {k} — backward graph did not reach this matrix"
        grads[k] = w.grad.detach().double().cpu()
        w.requires_grad_(False)
    pg.zero_grad(set_to_none=True)
    return grads, tot


# ===========================================================================
# Blank-image MMLU CE (the forgetting signal G_forget) — grad-enabled variant of
# text_eval.mmlu_blankimg, reusing its EXACT prompt construction (BOS/5-shot/left-pad).
# ===========================================================================

def mmlu_blank_batches(processor, n_per_subject, batch_size, device, n_shots=5, gray=128, max_subjects=None):
    """Build blank-image 5-shot MMLU teacher-forcing batches for the forgetting gradient. Reuses
    text_eval._subject_header/_q_block/CHOICES VERBATIM (same item order + left-pad as the scorer).
    Each batch = (enc, gt) where enc has input_ids/attention_mask/pixel_values and gt = the GT
    choice-letter token id per row. Returns (batches, letter_ids)."""
    import numpy as np
    import torch
    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image
    from infra.pg.text_eval import MMLU_REPO, CHOICES, _subject_header, _q_block

    ds = load_dataset(MMLU_REPO, "all")
    shots_by_subj, test_by_subj = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        shots_by_subj[r["subject"]].append(r)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)

    tok = processor.tokenizer
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in CHOICES]
    assert len(set(letter_ids)) == 4, f"choice-letter token ids collide: {letter_ids}"
    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    orig_side = tok.padding_side
    tok.padding_side = "left"

    batches = []
    subjects = sorted(test_by_subj)
    if max_subjects:
        subjects = subjects[:max_subjects]
    for subj in subjects:
        shots = shots_by_subj[subj][:n_shots]
        prefix = _subject_header(subj) + "".join(_q_block(s["question"], s["choices"], s["answer"])
                                                 for s in shots)
        rows = test_by_subj[subj][:n_per_subject] if n_per_subject else test_by_subj[subj]
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            prompts = [prefix + _q_block(r["question"], r["choices"]) for r in chunk]
            enc = processor(text=prompts, images=[gray_img] * len(chunk), return_tensors="pt",
                            padding="longest")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            enc["gt"] = torch.tensor([letter_ids[int(r["answer"])] for r in chunk], dtype=torch.long)
            batches.append(enc)
    tok.padding_side = orig_side
    return batches, letter_ids


def mmlu_blank_ce(pg, enc, device):
    """−log P(GT choice-letter | blank image, 5-shot prompt) — the blank-image MMLU CE, grad-enabled.
    Forward mirrors text_eval.mmlu_blankimg EXACTLY (logits_to_keep=1, last position, left-pad) but
    without inference_mode so the decoder weights get a gradient."""
    import torch
    import torch.nn.functional as F
    last = pg(input_ids=enc["input_ids"].to(device),
              attention_mask=enc["attention_mask"].to(device),
              pixel_values=enc["pixel_values"].to(device),
              logits_to_keep=1).logits[:, -1, :]
    return F.cross_entropy(last.float(), enc["gt"].to(device))


# ===========================================================================
# Cloze-CONTENT MMLU CE (the content signal ∂(cloze-CE)/∂W) — LATMECH §2.3.
# Same blank-image spine as mmlu_blank_*, but the CLOZE format text_eval.mmlu_cloze
# scores: teacher-forced CE on the GOLD CHOICE TEXT (the answer-content the deletion
# is claimed to preserve), not the choice LETTER. Feeds decoder_weight_grad to build
# the content carrier the LATMECH MDE projects out.
# ===========================================================================

def cloze_content_batches(processor, n_per_subject, batch_size, device, n_shots=5,
                          gray=128, max_subjects=None, item_filter=None):
    """Blank-image 5-shot CLOZE teacher-forcing batches for the content gradient. Mirrors
    mmlu_blank_batches' prompt spine (subject header + n_shots dev exemplars, VERBATIM item
    order), but in cloze format: each test item's prompt = header + n_shots `_q_block_cloze`
    exemplars (with the gold answer TEXT) + the test question's `_q_block_cloze` (ends at
    'Answer:'); the SUFFIX is the test item's GOLD CHOICE TEXT, so the processor's `suffix=`
    masks the prefix+image tokens (labels=-100) and the loss lands only on the correct-choice
    continuation. Item selection (subject order + `item_filter` hash predicate = sha1(q[:64])[:12])
    matches text_eval.mmlu_cloze EXACTLY → the carrier is computed on the same population the
    cloze axis scores. Returns a list of proc-encoded batch dicts (input_ids/attention_mask/
    pixel_values/labels). Fail-loud on an empty selection."""
    import hashlib
    import numpy as np
    import torch
    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image
    from infra.pg.text_eval import MMLU_REPO, _subject_header, _q_block_cloze

    ds = load_dataset(MMLU_REPO, "all")
    shots_by_subj, test_by_subj = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        shots_by_subj[r["subject"]].append(r)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)

    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    batches = []
    subjects = sorted(test_by_subj)
    if max_subjects:
        subjects = subjects[:max_subjects]
    for subj in subjects:
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in shots)
        rows = test_by_subj[subj][:n_per_subject] if n_per_subject else test_by_subj[subj]
        if item_filter is not None:
            rows = [r for r in rows if item_filter(
                hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            prompts = [prefix + _q_block_cloze(r["question"]) for r in chunk]
            suffixes = [str(r["choices"][int(r["answer"])]).strip() for r in chunk]
            enc = processor(text=prompts, images=[gray_img] * len(chunk), suffix=suffixes,
                            return_tensors="pt", padding="longest")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            batches.append(enc)
    assert batches, "cloze_content_batches produced no batches (over-restrictive item_filter?)"
    return batches


def cloze_content_ce(pg, enc, device):
    """−log P(gold choice-TEXT tokens | blank image, 5-shot cloze prompt) — the cloze-content
    decision CE, grad-enabled (the loss_fn `decoder_weight_grad` consumes to build the content
    carrier, LATMECH §2.3). Identical shift-by-one, suffix-masked LM loss as `refcoco_ce`: the
    processor's `suffix=` set the labels (prefix+image=-100; gold continuation=targets). Returns
    a scalar with a backward graph into the decoder weights."""
    import torch
    import torch.nn.functional as F
    ii = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    pv = enc["pixel_values"].to(device)
    labels = enc["labels"].to(device)
    logits = pg(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    assert int(mask.sum()) > 0, "no gold-continuation tokens in cloze batch (suffix/labels broken)"
    return F.cross_entropy(logits[mask].float(), tgt[mask])


# ===========================================================================
# LATMECH carrier build (once) + ablation apply — the §2.3 content carrier the
# MDE projects out. Build uses cloze_content_ce; apply uses latmech.project_out.
# ===========================================================================

def latmech_carrier_factors(pg, proc, paths, dkeys, *, svd_q, grad_subjects,
                            grad_per_subject, batch, item_filter, seed, device):
    """Build the EXPENSIVE part ONCE (LATMECH §2.3, "build the instrument once"): the
    per-decoder-matrix gradient G = ∂(cloze-content-CE)/∂W (one backward over `dkeys` via
    decoder_weight_grad + cloze_content_ce on the DISCOVERY split), the top-`svd_q` SVD of each W
    (svd_lowrank, deterministic SEED), and the content-sensitivity scores |u_dᵀ G v_d|
    (w15_carrier.direction_scores). Returns ({key: {U,S,V (cpu fp32), scores (cpu fp64)}},
    mean_grad_loss). `select_carrier_factors` slices this per carrier-k — so a carrier-k SWEEP
    recomputes NEITHER the gradient NOR the SVD (the §6 discovery calibration is cheap)."""
    import torch
    from infra.pg import w15_carrier as Cc
    lm = pg.get_submodule(paths["language_model"])
    batches = cloze_content_batches(proc, grad_per_subject, batch, device,
                                    max_subjects=grad_subjects, item_filter=item_filter)
    grads, gloss = decoder_weight_grad(pg, paths, dkeys, batches, cloze_content_ce, device)
    torch.manual_seed(SEED)
    full = {}
    for k in dkeys:
        W = lm.get_parameter(k).detach().float().to(device)
        qq = min(svd_q, min(W.shape))
        U, S, V = torch.svd_lowrank(W, q=qq, niter=8)          # top-qq directions of W
        scores = Cc.direction_scores(grads[k].to(device), U, V)
        full[k] = {"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), "scores": scores.cpu()}
        del W, U, S, V
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"[latmech] carrier factors over {len(full)} matrices (q={svd_q}), grad_loss={gloss:.4f}",
          flush=True)
    return full, gloss


def select_carrier_factors(full, dkeys, carrier_k, seed):
    """Slice the cached full SVD into the carrier + energy-matched-random factors at ONE carrier-k
    (cheap, no model): the top-`carrier_k` directions by content score (w15_carrier.rank_select)
    and an ENERGY-MATCHED RANDOM subset DISJOINT from them (latmech.energy_matched_random_idx).
    Returns {key: {carrier:(U,S,V), emrand:(U,S,V), removed_E, emrand_E, n_dir, k}} — the carrier.pt
    format apply_carrier consumes."""
    from infra.pg import w15_carrier as Cc
    from infra.pg import latmech as Lm
    factors = {}
    for k in dkeys:
        U, S, V, scores = full[k]["U"], full[k]["S"], full[k]["V"], full[k]["scores"]
        idx = Cc.rank_select(scores, carrier_k)
        removed_E = Lm.removed_energy(S, idx)
        em_idx, em_E = Lm.energy_matched_random_idx(S, removed_E, seed=seed, exclude=idx)
        factors[k] = {
            "carrier": (U[:, idx], S[idx], V[:, idx]),
            "emrand": (U[:, em_idx], S[em_idx], V[:, em_idx]),
            "removed_E": float(removed_E), "emrand_E": float(em_E),
            "n_dir": int(S.numel()), "k": int(idx.numel()),
        }
    return factors


def latmech_carrier(pg, proc, paths, dkeys, *, carrier_k, svd_q, grad_subjects,
                    grad_per_subject, batch, item_filter, seed, device):
    """Convenience: build factors + select at one carrier-k (the non-sweep path; the preflight
    sweeps via latmech_carrier_factors + select_carrier_factors directly)."""
    full, gloss = latmech_carrier_factors(
        pg, proc, paths, dkeys, svd_q=svd_q, grad_subjects=grad_subjects,
        grad_per_subject=grad_per_subject, batch=batch, item_filter=item_filter,
        seed=seed, device=device)
    return select_carrier_factors(full, dkeys, carrier_k, seed), gloss


def apply_carrier(lm, base_sd, dkeys, factors, which, device):
    """Project the cached `which` ∈ {'carrier','emrand'} subspace OUT of every decoder matrix:
    W_abl = W − Σ_i σ_i u_i v_iᵀ over the cached factor directions, then strict-load via
    _set_decoder (bf16 deploy, matching e4/eq1/reprice). `base_sd` = the (fresh, deleted) LM
    state_dict; `lm` MUST be the unmodified deleted decoder (one cell = one fresh build), so the
    subtraction is off the content-preserved weights. Returns the count of overridden matrices."""
    import torch
    from infra.pg import latmech as Lm
    ov = {}
    for k in dkeys:
        U, S, V = factors[k][which]
        U, S, V = U.to(device), S.to(device), V.to(device)
        W = lm.get_parameter(k).detach().float().to(device)
        idx = torch.arange(S.numel(), device=device)
        ov[k] = Lm.project_out_subspace(W, U, S, V, idx).to(device).to(torch.bfloat16)
    _set_decoder(lm, base_sd, ov, device)
    return len(ov)


# ===========================================================================
# Eval (capability-readout path verbatim) — metric + <loc> emission.
# ===========================================================================

def score_refcoco(pg, proc, task, rows, device, batch):
    """(acc@0.5 mean, <loc>-emission fraction) on RefCOCO rows via the frozen _generate path."""
    import numpy as np
    from infra.pg import battery_tasks as BT
    preds = _generate(pg, proc, task, rows, device, batch)
    m = float(np.mean([task.score_fn(p, r) for r, p in zip(rows, preds)]))
    emit = float(sum(BT._decode_loc_box(p) is not None for p in preds) / max(1, len(preds)))
    return m, emit, preds[:12]
