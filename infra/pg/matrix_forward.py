"""MENDELEEV forward shells — contract §3 of infra/pg/MATRIX_CONTRACT.md.

One `Forward` per (model, shell). EVERY behaviour readout goes through these four calls:

  next_token_logits(prompts)            -> (B, V) float32 CPU logits at the last real token
                                           (left-pad; `logits_to_keep=1`; EVERY shell passes content
                                           position_ids, so a score is a property of its prompt and not
                                           of its batch — `PaliGemmaForward._call_kwargs` states where
                                           that departs from text_eval.mmlu_blankimg).
  continuation_logprobs(prompts, conts, sep=" ") -> [(sum_logprob, n_tok)] per (prompt, continuation)
                                           pair, text_eval.mmlu_cloze semantics: full = prompt + sep +
                                           cont.strip() (sep="" after Molmo2's "assistant\n" — ROW3);
                                           continuation located by `_locate_continuation` (shared
                                           prefix + MINIMAL framing suffix, decomposition verified
                                           by exact reconstruction — read its docstring before
                                           touching it); `logits_to_keep=K`
                                           window; float64 sum. Batched across pairs.
  generate(prompts, max_new_tokens)     -> greedy strings (first line, stripped) — the
                                           text_eval._blank_image_generate convention for every shell.
  letter_ids(surfaces, space)           -> one id per surface: the token the model must emit for that
                                           surface IN CONTEXT ("Answer:" + " " + s, or "Answer: (" + s
                                           for space=False) — asserted single-token and distinct.
  letter_token_counts(surfaces, space)  -> the same in-context token count per surface WITHOUT
                                           asserting (the §6.2b tokenizer gate observes it; 0 = the
                                           surface merged into the context and is unreadable).
  Every prompt-taking call accepts `images=` (one PIL image per prompt) for the O column; a text
  shell asserts images is None; VLM shells default to the blank image.
  final_residuals(prompts, layers)      -> {L: (B, d) float16 CPU} last-real-token residual at the
                                           OUTPUT of decoder layer L (forward hook) — the N3 harvest.

Shells: TextForward (any *ForCausalLM, or a VLM class run text-only, e.g. Gemma3/Gemma4
ForConditionalGeneration as HELMHOLTZ-2 did), PaliGemmaForward (blank gray-128 224² image;
input_ids/attention_mask/pixel_values ONLY — no token_type_ids), Molmo2Forward (its remote-code
processor; blank image at the processor's native size; `<|image|>` placeholder prepended),
Pi0Forward (a PaliGemmaForward over the PG1 shell carrying a π0/π0.5 backbone; `head` ∈ {own, pg1}).

Attribute paths verified against transformers v5.9.0 (raw modeling files, tag 22404f79) and the
Molmo2/MolmoAct2 remote code (repo shas 042abfa7 / e432d85f): see `lm_anatomy`.

Fail-loud; no try/except; heavy imports inside functions (py_compiles without torch).
"""
from __future__ import annotations

GRAY = 128
LETTERS = ("A", "B", "C", "D")
MOLMO_IMAGE_PLACEHOLDER = "<|image|>"       # processing_molmo2.IMAGE_PROMPT

# (model_type of the OUTER config) -> (lm_path, layers_attr, norm_attr, embed_attr)
# Text CausalLM classes (v5.9.0): <cls>.model.{layers,norm,embed_tokens}; head <cls>.lm_head.
# VLM CG classes (v5.9.0): <cls>.model.language_model.{layers,norm,embed_tokens}; head <cls>.lm_head.
# Molmo2/MolmoAct2 (remote code): <cls>.model.transformer.{blocks,ln_f,wte}; head <cls>.lm_head.
_ANATOMY = {
    "gemma": ("model", "layers", "norm", "embed_tokens"),
    "gemma2": ("model", "layers", "norm", "embed_tokens"),
    "qwen3": ("model", "layers", "norm", "embed_tokens"),
    "llama": ("model", "layers", "norm", "embed_tokens"),
    "mistral": ("model", "layers", "norm", "embed_tokens"),
    "paligemma": ("model.language_model", "layers", "norm", "embed_tokens"),
    "gemma3": ("model.language_model", "layers", "norm", "embed_tokens"),
    "gemma4": ("model.language_model", "layers", "norm", "embed_tokens"),
    "molmo2": ("model.transformer", "blocks", "ln_f", "wte"),
    "molmoact2": ("model.transformer", "blocks", "ln_f", "wte"),
}


def text_config(config):
    """The text-tower config (nested `text_config` for VLM classes, the config itself for text LMs)."""
    return getattr(config, "text_config", None) or config


def lm_anatomy(model) -> dict:
    """Resolve the language-model decoder anatomy of a loaded model by model_type (table above), then
    ASSERT it against the live module tree (`get_submodule`) and the config's layer count."""
    mt = model.config.model_type
    if mt not in _ANATOMY:
        raise NotImplementedError(f"no anatomy table for model_type={mt!r} ({type(model).__name__})")
    lm_path, layers_attr, norm_attr, embed_attr = _ANATOMY[mt]
    lm = model.get_submodule(lm_path)
    layers = getattr(lm, layers_attr)
    n_cfg = int(text_config(model.config).num_hidden_layers)
    assert len(layers) == n_cfg, f"{mt}: {len(layers)} live layers != config {n_cfg}"
    return {"family": mt, "lm": lm, "lm_path": lm_path,
            "layers": layers, "layers_path": f"{lm_path}.{layers_attr}",
            "norm": getattr(lm, norm_attr), "norm_path": f"{lm_path}.{norm_attr}",
            "embed": getattr(lm, embed_attr), "embed_path": f"{lm_path}.{embed_attr}",
            "lm_head": model.lm_head, "lm_head_path": "lm_head", "n_layers": n_cfg}


def content_position_ids(attention_mask):
    """lm-eval recipe for LEFT padding: RoPE positions start at the first real token (text shells)."""
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def _lcp(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _lcs(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def _locate_continuation(prompt_ids, full_ids, pair=0, text=""):
    """(cont_ids, trailing) for one (bare prompt, prompt+sep+continuation) encoding pair.

    THE SHAPE. A shell frames text as `prompt -> [shared | sep]` and the joined text as
    `full -> [shared | cont | sep]`, where `sep` is the shell's own fixed framing suffix (the
    PaliGemma processor appends one '\\n'; a plain text shell appends nothing, sep = []). So the
    continuation is the block `full` has that `prompt` does not, and `trailing = |sep|` is what
    `text_eval._score_cloze_window` needs to step the scoring window back off the separator.

    WHY NOT longest-common-suffix (the defect this replaces, 2026-09-03). The old locator took
    `s = lcs(prompt, full)` as |sep| and asserted `lcp + s == |prompt|`. `lcs` is GREEDY and
    DATA-DEPENDENT: when a candidate's last token happens to repeat the prompt's last content
    token, the suffix match runs straight past the separator into the shared prefix, the two
    matches OVERLAP, and `lcp + lcs > |prompt|`. Row-2 O hit exactly that on a real VQAv2 item —
    `lcp=268 lcs=2 |prompt|=269`, overlap 1 — and the assertion killed the column. Widening the
    assertion alone would have been worse than the crash: `full[lcp:-lcs]` silently DROPS that
    repeated last token, so one candidate of four would have been scored on n-1 tokens and the
    per-token-mean argmax would have been comparing unlike things.

    THE RULE. `sep` is a property of the SHELL, not of the candidate, so take the MINIMAL suffix
    consistent with the observed prefix match, `trailing = |prompt| - lcp`, and then VERIFY the
    decomposition by exact reconstruction: `prompt[:lcp] + cont + prompt[lcp:] == full`, token for
    token. lcp is safe where lcs is not, because every continuation is `.strip()`ped before it is
    joined, so it can never begin with the whitespace token the framing separator is made of.
    Reconstruction keeps the assertion's protective intent: a continuation that genuinely cannot be
    located — a boundary token that RE-MERGES with the framing, the M-D45/M-D51 trap — leaves a
    tail that does not match and still fails loud. The caller additionally asserts `trailing` is
    constant across the batch, and `cont` is by construction exactly the `|full| - |prompt|` tokens
    the joined encoding adds, so every candidate of an item and every model scores its own
    continuation's own tokens and nothing else.
    """
    p = _lcp(prompt_ids, full_ids)
    trailing = len(prompt_ids) - p
    cont = full_ids[p:len(full_ids) - trailing]
    assert prompt_ids[:p] + cont + prompt_ids[p:] == full_ids, (
        f"continuation not locatable as [shared|cont|sep] (pair {pair}, {text!r}): "
        f"lcp={p} lcs={_lcs(prompt_ids, full_ids)} |prompt|={len(prompt_ids)} |full|={len(full_ids)}")
    assert len(cont) >= 1, f"empty continuation (pair {pair}): {text!r}"
    return cont, trailing


def blank_image(size: int, gray: int = GRAY):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.full((size, size, 3), gray, dtype=np.uint8))


class Forward:
    """Base shell. Subclasses implement `_encode(texts) -> dict` (left-padded, `input_ids` +
    `attention_mask` + whatever the model needs) and `_call_kwargs(enc) -> dict` (the EXACT kwargs
    the model is called with, minus `logits_to_keep`)."""
    shell = "abstract"

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        an = lm_anatomy(model)
        self.family = an["family"]
        self.n_layers = an["n_layers"]
        self.layers = an["layers"]
        self.anatomy_paths = {k: v for k, v in an.items() if k.endswith("_path")}
        self.batch_size = 4                      # util_probe.choose_batch sets this at job start
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    # ---- abstract -------------------------------------------------------------------------
    def _encode(self, texts, images=None):
        raise NotImplementedError

    def _call_kwargs(self, enc):
        raise NotImplementedError

    # ---- shared ---------------------------------------------------------------------------
    def _chunks(self, seq):
        for s in range(0, len(seq), self.batch_size):
            yield seq[s:s + self.batch_size]

    def _last_logits(self, enc):
        import torch
        with torch.inference_mode():
            out = self.model(**self._call_kwargs(enc), logits_to_keep=1)
        return out.logits[:, -1, :]

    def _paired(self, prompts, images):
        prompts = list(prompts)
        if images is None:
            return prompts, [None] * len(prompts)
        images = list(images)
        assert len(images) == len(prompts), (len(images), len(prompts))
        return prompts, images

    def _chunk_pairs(self, prompts, images):
        for s in range(0, len(prompts), self.batch_size):
            yield prompts[s:s + self.batch_size], images[s:s + self.batch_size]

    def next_token_logits(self, prompts, images=None):
        import torch
        outs = []
        prompts, images = self._paired(prompts, images)
        for chunk, imgs in self._chunk_pairs(prompts, images):
            lg = self._last_logits(self._encode(chunk, imgs)).float().cpu()
            assert torch.isfinite(lg).all(), "non-finite next-token logits"
            outs.append(lg)
        return torch.cat(outs, 0)

    def _surface_tokens(self, surface, space):
        """Tokens the surface contributes after its scoring context ("Answer:" for the with-space
        readouts S/F1/F2/F3/F5-letters, "Answer: (" for F4). A tokenizer that merges the surface into the
        context (prefix instability) contributes 0 tokens — unreadable at the next-token position."""
        ctx = "Answer:" if space else "Answer: ("
        base = self.tok.encode(ctx, add_special_tokens=False)
        full = self.tok.encode(ctx + (" " if space else "") + str(surface), add_special_tokens=False)
        if full[:len(base)] != base:
            return []
        return full[len(base):]

    def letter_token_counts(self, surfaces=LETTERS, space=True):
        return [len(self._surface_tokens(s, space)) for s in surfaces]

    def letter_ids(self, surfaces=LETTERS, space=True):
        """One id per surface — the in-context token (Gemma/Qwen/Llama-3: ' A' is one token; Mistral
        SentencePiece splits a bare ' A' into ['▁', 'A'] but emits the single '▁A' after 'Answer:', which
        is what this measures — the same id relearn.letter_token_ids resolves). Asserts single-token +
        distinct; the gate reads `letter_token_counts` first so a failure here is a harness bug."""
        ids = []
        for s in surfaces:
            t = self._surface_tokens(s, space)
            assert len(t) == 1, f"surface {s!r} (space={space}) is {len(t)} tokens in context: {t}"
            ids.append(int(t[0]))
        assert len(set(ids)) == len(ids), f"surface ids collide: {dict(zip(surfaces, ids))}"
        return ids

    def continuation_logprobs(self, prompts, continuations, images=None, sep=" "):
        """text_eval.mmlu_cloze window scoring, batched across (prompt, continuation) pairs; `sep` joins
        prompt and continuation (default " " = the C1 convention; "" for chat-template shells)."""
        import torch
        from infra.pg.text_eval import _score_cloze_window
        prompts, images = self._paired(prompts, images)
        continuations = list(continuations)
        assert len(prompts) == len(continuations), (len(prompts), len(continuations))
        bare = {}
        for i, p in enumerate(prompts):
            key = (p, i if images[i] is not None else None)
            if key not in bare:
                pe = self._encode([p], [images[i]] if images[i] is not None else None)
                am = pe["attention_mask"][0]
                bare[key] = pe["input_ids"][0][int((am == 0).sum()):].tolist()
        pairs = list(range(len(prompts)))
        out = [None] * len(prompts)
        for chunk in self._chunks(pairs):
            fulls = [prompts[i] + sep + continuations[i].strip() for i in chunk]
            imgs = [images[i] for i in chunk]
            enc = self._encode(fulls, imgs if any(x is not None for x in imgs) else None)
            ids, am = enc["input_ids"], enc["attention_mask"]
            real = am.sum(dim=1).tolist()
            cont_ids, ntok, trailing = [], [], None
            for r, i in enumerate(chunk):
                pr = bare[(prompts[i], i if images[i] is not None else None)]
                full_real = ids[r][ids.shape[1] - int(real[r]):].tolist()
                cont, t = _locate_continuation(pr, full_real, i, continuations[i])
                if trailing is None:
                    trailing = t
                assert t == trailing, f"inconsistent shell framing within a batch: {t} vs {trailing}"
                cont_ids.append(cont)
                ntok.append(len(cont))
            K = max(ntok) + trailing + 1
            with torch.inference_mode():
                o = self.model(**self._call_kwargs(enc), logits_to_keep=K)
                lw = torch.log_softmax(o.logits.float(), dim=-1)
            sums = _score_cloze_window(lw, cont_ids, ntok, trailing=trailing)
            assert torch.isfinite(sums).all(), "non-finite continuation logprob"
            for r, i in enumerate(chunk):
                out[i] = (float(sums[r]), int(ntok[r]))
        return out

    def generate(self, prompts, max_new_tokens, images=None, skip_special_tokens=True):
        import torch
        outs = []
        prompts, images = self._paired(prompts, images)
        for chunk, imgs in self._chunk_pairs(prompts, images):
            enc = self._encode(chunk, imgs)
            kw = {k: v for k, v in self._call_kwargs(enc).items() if k != "position_ids"}
            in_len = enc["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = self.model.generate(**kw, max_new_tokens=max_new_tokens, do_sample=False)
            dec = self.tok.batch_decode(gen[:, in_len:], skip_special_tokens=skip_special_tokens)
            outs.extend(d.split("\n")[0].strip() for d in dec)
        assert len(outs) == len(prompts)
        return outs

    def final_residuals(self, prompts, layers):
        """Post-block residual at the last real token for each requested decoder layer index."""
        import torch
        layers = [int(L) for L in layers]
        for L in layers:
            assert 0 <= L < self.n_layers, (L, self.n_layers)
        cap = {}

        def _hook_for(L):
            def h(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                cap[L] = hs[:, -1, :].detach().to(torch.float16).cpu()
            return h

        outs = {L: [] for L in layers}
        for chunk in self._chunks(list(prompts)):
            handles = [self.layers[L].register_forward_hook(_hook_for(L)) for L in layers]
            self._last_logits(self._encode(chunk))
            for hd in handles:
                hd.remove()
            for L in layers:
                assert L in cap, f"hook on layer {L} did not fire"
                outs[L].append(cap.pop(L))
        return {L: torch.cat(v, 0) for L, v in outs.items()}


class TextForward(Forward):
    """Pure text forward: tokenizer(left pad, specials on), content-relative position_ids, exactly
    text_eval.mmlu_accuracy / mmlu_cloze (processor=None)."""
    shell = "text"

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)

    def _encode(self, texts, images=None):
        assert images is None or all(im is None for im in images), "text shell takes no images"
        return self.tok(list(texts), return_tensors="pt", padding=True,
                        add_special_tokens=True).to(self.device)

    def _call_kwargs(self, enc):
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "position_ids": content_position_ids(enc["attention_mask"])}


class PaliGemmaForward(Forward):
    """Blank gray-128 224² image present; the model is called with input_ids / attention_mask /
    pixel_values + CONTENT position_ids (no token_type_ids) + logits_to_keep. The processor appends a
    trailing '\\n' after the text, so the last position predicts after 'Answer:\\n' — that IS the C1
    instrument; parity rule §12.10.

    position_ids ARE passed, and that is a DEPARTURE from `text_eval.mmlu_blankimg` — see `_call_kwargs`."""
    shell = "paligemma"

    def __init__(self, model, processor, device="cuda", gray=GRAY, image_size=224):
        self.proc = processor
        self.gray = gray
        self.blank = blank_image(image_size, gray)
        super().__init__(model, processor.tokenizer, device)

    def _images(self, texts, images):
        if images is None:
            return [self.blank] * len(texts)
        assert len(images) == len(texts)
        return [self.blank if im is None else im.convert("RGB") for im in images]

    def _encode(self, texts, images=None):
        return self.proc(text=list(texts), images=self._images(texts, images),
                         return_tensors="pt", padding="longest").to(self.device)

    def _call_kwargs(self, enc):
        """position_ids are the SHELL's job, not the caller's.

        THE DEFECT (2026-09-03). This sent ids + mask + pixels and no position_ids, so the model fell
        back to arange over the PADDED length. With left padding a sequence's RoPE positions then start
        at the pad width — which is set by the LONGEST prompt in its batch — and the same prompt scored
        differently depending on the company it kept. `matrix_graft.py:54` pins `BATCH = 4` for exactly
        this reason: a correctness property that lived in every caller's memory instead of in the code.
        `content_position_ids` is the same left-pad recipe `TextForward` already used, so the two shells
        now agree and a score is a property of its prompt. `_dryrun_batch_positions` reproduces the old
        drift and pins the new invariance.

        STATED, for the lead — this is a numeric change, not only a bug fix. It moves the shell off
        `text_eval.mmlu_blankimg`'s exact call, which is the C1 instrument the §12.10 preflight compares
        against (row 2 S per item vs the banked CARREL vectors, ≥ 99 % agreement, |Δacc| ≤ 0.01). That
        preflight is the right test of whether the departure matters, and it should be read as such."""
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "pixel_values": enc["pixel_values"].to(self.model.dtype),
                "position_ids": content_position_ids(enc["attention_mask"])}


class Pi0Forward(PaliGemmaForward):
    """PG1 shell carrying a π0 / π0.5 backbone (matrix_models installs it via lacuna_shell); `head`
    records which embedding/head table is in the shell: 'own' (the policy's tied head, primary) or
    'pg1' (PG1's table, secondary)."""
    shell = "pi0-pg1"

    def __init__(self, model, processor, head, device="cuda", gray=GRAY):
        assert head in ("own", "pg1"), head
        self.head = head
        super().__init__(model, processor, device=device, gray=gray)


class Molmo2Forward(Forward):
    """Molmo2 / MolmoAct2 (remote code). Text = '<|image|>' + prompt; the processor expands the
    placeholder into the image-token block, inserts BOS, LEFT-pads (its `insert_bos` contract), and
    returns input_ids / attention_mask / pixel_values / image_token_pooling / image_grids /
    image_num_crops — all of which are passed to the model. Blank gray image at the image
    processor's native `size` (378² for Molmo2-4B), one crop by construction; the vision backbone
    casts pixel_values to its own dtype internally."""
    shell = "molmo2"

    def __init__(self, model, processor, device="cuda", gray=GRAY):
        self.proc = processor
        self.gray = gray
        sz = processor.image_processor.size
        assert sz["height"] == sz["width"], sz
        self.image_size = int(sz["height"])
        self.blank = blank_image(self.image_size, gray)
        self.placeholder = getattr(processor, "image_placeholder_token", MOLMO_IMAGE_PLACEHOLDER)
        super().__init__(model, processor.tokenizer, device)

    def _images(self, texts, images):
        if images is None:
            return [self.blank] * len(texts)
        assert len(images) == len(texts)
        return [self.blank if im is None else im.convert("RGB") for im in images]

    def _encode(self, texts, images=None):
        imgs = self._images(texts, images)
        texts = [self.placeholder + t for t in texts]
        enc = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        return enc.to(self.device)

    def _call_kwargs(self, enc):
        """Same position_ids fix as `PaliGemmaForward._call_kwargs` — read that docstring for the defect.

        UNVERIFIED LOCALLY, stated: Molmo2 is remote code and this laptop may not download it, so that
        `Molmo2ForConditionalGeneration.forward` accepts `position_ids` is asserted from the HF VLM
        contract, not from the file. It is a load-time, fail-loud unknown: row 3b's first forward either
        works or raises a TypeError naming the argument. Nothing silent either way."""
        keys = ("input_ids", "attention_mask", "pixel_values", "image_token_pooling",
                "image_grids", "image_num_crops")
        missing = [k for k in keys if k not in enc]
        assert not missing, f"Molmo2 processor output lacks {missing}; has {sorted(enc.keys())}"
        return {**{k: enc[k] for k in keys},
                "position_ids": content_position_ids(enc["attention_mask"])}


# ===========================================================================================
# dry-run: a weight-free stub model with the v5.9.0 CausalLM anatomy + a char tokenizer.
# ===========================================================================================
class _StubTok:
    """Char-level tokenizer over a fixed alphabet: BOS=1, PAD=0; ' A'..' D', ' 1'..' 4', ' Yes', ' No' are
    single ids (longest match first); `merge_surfaces` makes ' A' unreadable (merges 'Answer: A' into one
    token) so the dry-run can reach the VOID-tokenizer branch."""
    def __init__(self, merge_surfaces=()):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self._v = {" A": 10, " B": 11, " C": 12, " D": 13, " 1": 14, " 2": 15, " 3": 16, " 4": 17,
                   " Yes": 18, " No": 19}
        self._merge = {f"Answer: {m}": 250 + i for i, m in enumerate(merge_surfaces)}

    def _ids(self, text, add_special_tokens):
        out = [1] if add_special_tokens else []
        i = 0
        while i < len(text):
            hit = next((m for m in self._merge if text.startswith(m, i)), None)
            if hit is not None:
                out.append(self._merge[hit]); i += len(hit); continue
            hit = next((v for v in sorted(self._v, key=len, reverse=True) if text.startswith(v, i)), None)
            if hit is not None:
                out.append(self._v[hit]); i += len(hit)
            else:
                out.append(20 + (ord(text[i]) % 200)); i += 1
        return out

    def encode(self, text, add_special_tokens=True):
        return self._ids(text, add_special_tokens)

    def get_vocab(self) -> dict:
        """The HF tokenizer surface `matrix_models.vocab_sha256` / `shared_prefix` (B1) read. The
        char-level ids 20..219 are this alphabet; the merge ids are whatever `merge_surfaces` made."""
        return {**{f"<{i}>": i for i in range(20)}, **dict(self._v), **dict(self._merge),
                **{chr(c): 20 + (c % 200) for c in range(32, 127)}}

    def convert_tokens_to_ids(self, t):
        return -1

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=True):
        import torch
        if isinstance(texts, str):                      # HF accepts a bare string → batch of 1
            texts = [texts]
        rows = [self._ids(t, add_special_tokens) for t in texts]
        S = max(len(r) for r in rows)
        ids = torch.full((len(rows), S), self.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(rows), S), dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, S - len(row):] = torch.tensor(row)
            am[r, S - len(row):] = 1

        class _B(dict):
            def to(self, device):
                return self

            def __getattr__(self, k):               # HF BatchEncoding: enc["x"] and enc.x both work
                return self[k]
        return _B(input_ids=ids, attention_mask=am)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(int(t) - 20 + 32) if int(t) >= 20 else "" for t in row) for row in ids]

    def decode(self, ids, skip_special_tokens=True):
        """EXACT inverse of `_ids` for ordinary characters and the multi-char surfaces, so a
        tokenize → slice → decode → re-tokenize round trip is lossless. `shell_text_perplexity`
        needs that round trip to locate a window's text block inside the shell's own input_ids."""
        inv = {v: k for k, v in self._v.items()}
        return "".join(chr(int(t) - 20) if int(t) >= 20 else inv.get(int(t), "") for t in ids)


def stub_text_model(n_layers=4, d=16, vocab=256, seed=0):
    """A tiny nn.Module with `.config.model_type='llama'`, `.model.layers/.norm/.embed_tokens`,
    `.lm_head`, a `forward(input_ids, attention_mask, position_ids, logits_to_keep)` returning
    `.logits`, and a greedy `.generate`. Each layer returns a tuple (hidden,) like HF decoder layers."""
    import types
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, h, **kw):
            return (torch.tanh(self.lin(h)) + h,)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.config = types.SimpleNamespace(model_type="llama", num_hidden_layers=n_layers,
                                                hidden_size=d, vocab_size=vocab, text_config=None)
            self.dtype = torch.float32

        def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0, **kw):
            h = self.model.embed_tokens(input_ids)
            if position_ids is not None:
                h = h + 1e-3 * position_ids[..., None].float()
            for lyr in self.model.layers:
                h = lyr(h)[0]
            h = self.model.norm(h)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                h = h[:, -logits_to_keep:, :]
            return types.SimpleNamespace(logits=self.lm_head(h))

        def generate(self, input_ids, attention_mask=None, max_new_tokens=4, do_sample=False, **kw):
            ids = input_ids
            for _ in range(max_new_tokens):
                nxt = self.forward(ids, logits_to_keep=1).logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], 1)
            return ids

        def get_submodule(self, path):
            m = self
            for p in path.split("."):
                m = getattr(m, p)
            return m

        def _init_weights(self, module):          # HF PreTrainedModel convention (per-module)
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(0.0, 0.02)
                if getattr(module, "bias", None) is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0); module.bias.data.zero_()

    return M().eval()


def _dryrun():
    import math
    import torch
    m = stub_text_model()
    fwd = TextForward(m, _StubTok(), device="cpu")
    assert fwd.family == "llama" and fwd.n_layers == 4 and fwd.shell == "text"
    ps = ["Q: 1+1?\nAnswer:", "Q: capital of France? Answer:"]
    lg = fwd.next_token_logits(ps)
    assert lg.shape == (2, 256) and lg.dtype == torch.float32
    # batch-composition invariance of the last-token read (left pad + content positions)
    lg1 = fwd.next_token_logits(ps[:1])
    assert torch.allclose(lg[0], lg1[0], atol=1e-5), "left-pad changed the last-token logits"
    assert fwd.letter_token_counts() == [1, 1, 1, 1]
    ids = fwd.letter_ids()
    assert ids == [10, 11, 12, 13], ids
    assert fwd.letter_ids(("1", "2", "3", "4"), True) == [14, 15, 16, 17]
    assert fwd.letter_ids(("Yes", "No"), True) == [18, 19]
    assert fwd.letter_ids(("A", "B", "C", "D"), False) == [20 + ord(c) % 200 for c in "ABCD"]
    assert fwd.letter_token_counts(("Yes", "Maybe"), True) == [1, 6]      # multi-token observed, no assert
    merged = TextForward(m, _StubTok(merge_surfaces=("A",)), device="cpu")
    assert merged.letter_token_counts() == [0, 1, 1, 1], merged.letter_token_counts()
    raised = False
    try:
        merged.letter_ids()
    except AssertionError:
        raised = True
    assert raised, "letter_ids must assert on an unreadable surface"
    raised = False
    try:
        fwd.next_token_logits(ps, images=[object(), object()])
    except AssertionError:
        raised = True
    assert raised, "text shell must refuse images"
    # continuation scoring: sum over the continuation tokens equals a manual gather
    cl = fwd.continuation_logprobs([ps[0], ps[0], ps[1]], ["Paris", "  Rome ", "two"])
    assert len(cl) == 3 and all(n >= 1 for _, n in cl) and all(math.isfinite(s) for s, _ in cl)
    full = ps[0] + " Paris"
    enc = fwd._encode([full])
    with torch.inference_mode():
        lw = torch.log_softmax(m(**fwd._call_kwargs(enc), logits_to_keep=0).logits.float(), -1)
    n = cl[0][1]
    tgt = enc["input_ids"][0, -n:]
    manual = float(sum(lw[0, -n - 1 + j, tgt[j]] for j in range(n)))
    assert abs(manual - cl[0][0]) < 1e-4, (manual, cl[0][0])
    # batching across pairs with different batch sizes gives identical sums
    fwd.batch_size = 1
    cl1 = fwd.continuation_logprobs([ps[0], ps[0], ps[1]], ["Paris", "  Rome ", "two"])
    assert all(abs(a[0] - b[0]) < 1e-4 and a[1] == b[1] for a, b in zip(cl, cl1)), (cl, cl1)
    fwd.batch_size = 4
    cs = fwd.continuation_logprobs([ps[0]], ["Paris"], sep="")             # no separator: 5 chars, not 6
    assert cs[0][1] == cl[0][1] - 1 and cs[0][0] != cl[0][0], (cs, cl[0])
    g = fwd.generate(ps, max_new_tokens=3)
    assert len(g) == 2 and all(isinstance(s, str) for s in g)
    res = fwd.final_residuals(ps, [1, 3])
    assert set(res) == {1, 3} and res[1].shape == (2, 16) and res[1].dtype == torch.float16
    # hooks removed: a second call must not double-register
    res2 = fwd.final_residuals(ps, [1])
    assert torch.equal(res2[1], res[1])
    assert lm_anatomy(m)["layers_path"] == "model.layers"
    _dryrun_cloze_boundary(m)
    _dryrun_batch_positions()
    _dryrun_shell_ppl(m)
    print("PASS matrix_forward dry-run (stub llama anatomy; text shell; cloze window == manual gather; "
          "candidate boundary: lcp/lcs overlap located and scored whole, clean case unchanged, "
          "re-merged boundary still fails loud; "
          "VLM shells pass content position_ids — batch composition moved a score before, not after; "
          "shell-framed ppl locates the text block under an image prefix; no-start-token shell drops "
          "one unscorable target per window)")


def _dryrun_cloze_boundary(m):
    """Regression for the candidate-boundary locator (`_locate_continuation`), weight-free.

    A PaliGemma-shaped stub shell (256 'image' ids + BOS + text + the processor's appended '\\n')
    over the char tokenizer, so the bare prompt is 269 real tokens — the exact field shape. Three
    cases, all scored through the real `continuation_logprobs`:

      A  clean          lcp=268 lcs=1 |prompt|=269   (what mmlu_cloze/R6 always saw: they SUM)
      B  overlap        lcp=268 lcs=2 |prompt|=269   (row-2 O, 2026-09-03: 268+2 > 269) — the
                        candidate's last token repeats the prompt's last content token, so the
                        greedy suffix match runs one token past the separator. The old locator
                        asserted here; a merely-widened assertion would have scored 2 of the 3
                        continuation tokens. This case pins the FULL 3.
      C  unlocatable    the continuation's last token RE-MERGES with the appended '\\n', so no
                        [shared|cont|sep] decomposition exists — must still raise.
    """
    import torch

    class _PGFramedForward(Forward):
        """PaliGemma-shaped stub: 256 'image' ids + BOS + text + the processor's appended '\\n'."""
        shell = "stub-pg"

        def _encode(self, texts, images=None):
            rows = [[9] * 256 + [1] + self.tok.encode(t + "\n", add_special_tokens=False)
                    for t in texts]
            S = max(len(r) for r in rows)
            ids = torch.zeros((len(rows), S), dtype=torch.long)
            am = torch.zeros((len(rows), S), dtype=torch.long)
            for r, row in enumerate(rows):
                ids[r, S - len(row):] = torch.tensor(row)
                am[r, S - len(row):] = 1
            return {"input_ids": ids, "attention_mask": am}

        def _call_kwargs(self, enc):
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    class _NLMergeTok(_StubTok):
        """`_StubTok` plus one merge that SPANS the appended newline: 'z\\n' is a single id."""
        def _ids(self, text, add_special_tokens):
            parts = text.split("z\n")
            out = super()._ids(parts[0], add_special_tokens)
            for q in parts[1:]:
                out.append(230)
                out.extend(super()._ids(q, False))
            return out

    tok = _StubTok()
    fwd = _PGFramedForward(m, tok, device="cpu")
    prompt = "answer en X"                                  # 11 chars -> 256 + BOS + 11 + '\n' = 269
    pr = fwd._encode([prompt])["input_ids"][0].tolist()
    assert len(pr) == 269, len(pr)

    # --- the three (lcp, lcs, |prompt|) shapes, straight off the tokenizer -----------------
    shapes = {}
    for name, cont in (("A_clean", "Paris"), ("B_overlap", "oX")):
        f = fwd._encode([prompt + " " + cont])["input_ids"][0].tolist()
        shapes[name] = (_lcp(pr, f), _lcs(pr, f), len(pr), len(f))
    assert shapes["A_clean"][:3] == (268, 1, 269), shapes["A_clean"]
    assert shapes["B_overlap"][:3] == (268, 2, 269), shapes["B_overlap"]     # 268 + 2 > 269

    # --- case A + case B in ONE batch: each keeps its own, WHOLE continuation --------------
    fwd.batch_size = 4
    got = fwd.continuation_logprobs([prompt, prompt], ["Paris", "oX"])
    assert [n for _, n in got] == [6, 3], got            # " Paris" = 6, " oX" = 3 (not 2)
    # the pre-fix slice `full[lcp:-lcs]` on case B drops the repeated last token — pinned here so
    # nobody "simplifies" the locator back to lcs.
    lcp_b, lcs_b, _, len_b = shapes["B_overlap"]
    assert len_b - lcp_b - lcs_b == 2, "the old lcs slice must be the SHORT one this test rejects"
    # batch composition must not move the numbers
    fwd.batch_size = 1
    one = fwd.continuation_logprobs([prompt, prompt], ["Paris", "oX"])
    assert all(abs(a[0] - b[0]) < 1e-5 and a[1] == b[1] for a, b in zip(got, one)), (got, one)
    fwd.batch_size = 4

    # --- case B against an independent manual gather over exactly those 3 tokens -----------
    enc = fwd._encode([prompt + " oX"])
    with torch.inference_mode():
        lw = torch.log_softmax(m(**fwd._call_kwargs(enc), logits_to_keep=0).logits.float(), -1)
    f = enc["input_ids"][0].tolist()
    want = tok.encode(" oX", add_special_tokens=False)
    assert f[268:271] == want and f[271:] == tok.encode("\n", add_special_tokens=False), (f[268:], want)
    manual = float(sum(lw[0, 267 + j, want[j]] for j in range(3)))
    assert abs(manual - got[1][0]) < 1e-4, (manual, got[1])

    # --- case C: a genuinely unlocatable continuation must still fail loud -----------------
    mf = _PGFramedForward(m, _NLMergeTok(), device="cpu")
    assert len(mf._encode([prompt])["input_ids"][0].tolist()) == 269       # bare prompt unaffected
    raised = ""
    try:
        mf.continuation_logprobs([prompt], ["Xz"])
    except AssertionError as e:
        raised = str(e)
    assert raised.startswith("continuation not locatable as [shared|cont|sep]"), raised or "NO RAISE"

    # --- and the text shell, where the framing suffix is empty (trailing = 0) --------------
    tf = TextForward(m, _StubTok(), device="cpu")
    tp = "Q: a letter?\nAnswer:"
    assert tf.continuation_logprobs([tp], ["yes:"])[0][1] == 5, "text shell: ':' repeat must not bite"
    print("  cloze-boundary regression: A(268,1,269)->6 tok  B(268,2,269)->3 tok (was: assert; "
          "naive widen -> 2)  C re-merged boundary -> raises  text shell trailing=0 -> 5 tok")


def _dryrun_batch_positions():
    """position_ids on the VLM shells: the batch-composition defect, reproduced and then closed.

    The two VLM shells sent no position_ids, so the model fell back to arange over the padded length.
    Under left padding that makes a sequence's positions a function of the WIDEST prompt in its batch,
    and `matrix_graft.py:54` pins `BATCH = 4` to hold that still. Here a stub model does what a real
    forward does when position_ids is None — arange over the padded length — and the SAME prompt is
    scored alone and beside a longer one, through the REAL `_call_kwargs` of both shells, before and
    after. Before: the score moves. After: it does not.
    """
    import torch

    m = stub_text_model(seed=5)
    raw = m.forward

    def abs_pos_forward(input_ids, attention_mask=None, position_ids=None, logits_to_keep=0, **kw):
        if position_ids is None:                      # what an HF forward does: cache_position arange
            position_ids = torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1)
        return raw(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                   logits_to_keep=logits_to_keep)

    m.forward = abs_pos_forward

    class _PGShell(Forward):
        """PaliGemma-shaped stub running the REAL PaliGemmaForward._call_kwargs."""
        shell = "stub-pg-pixels"
        _call_kwargs = PaliGemmaForward._call_kwargs

        def _encode(self, texts, images=None):
            enc = self.tok(list(texts))
            enc["pixel_values"] = torch.zeros(len(texts), 3, 4, 4)
            return enc

    class _PGShellOld(_PGShell):
        """…and the kwargs it sent before the fix."""
        def _call_kwargs(self, enc):
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "pixel_values": enc["pixel_values"].to(self.model.dtype)}

    class _MolmoShell(_PGShell):
        """Molmo2-shaped stub running the REAL Molmo2Forward._call_kwargs (its six processor keys)."""
        shell = "stub-molmo"
        _call_kwargs = Molmo2Forward._call_kwargs

        def _encode(self, texts, images=None):
            enc = _PGShell._encode(self, texts)
            for k in ("image_token_pooling", "image_grids", "image_num_crops"):
                enc[k] = torch.zeros(len(texts), 1)
            return enc

    short = "Q: 2+2?\nAnswer:"
    long = "Q: a considerably longer question, which sets the pad width for the batch\nAnswer:"
    letters = _StubTok()._v          # ' A'..' D' ids — the S readout's four candidates
    ids = [letters[" " + L] for L in LETTERS]

    def p_A(fwd, prompts):
        """p(' A') over the four choice letters for the FIRST prompt — an S-shaped score."""
        lg = fwd.next_token_logits(prompts)[0, ids]
        return float(torch.softmax(lg.float(), -1)[0])

    old, new = _PGShellOld(m, _StubTok(), device="cpu"), _PGShell(m, _StubTok(), device="cpu")
    b_alone, b_batch = p_A(old, [short]), p_A(old, [short, long])
    a_alone, a_batch = p_A(new, [short]), p_A(new, [short, long])
    # TOL: 1e-6, not 0 — batched fp32 matmul is not bitwise stable across batch shapes, so the residual
    # here (~5e-8) is float noise. The pre-fix gap is four orders of magnitude larger and is positional.
    assert abs(b_alone - b_batch) > 1e-3, "the pre-fix kwargs must show the batch-composition drift"
    assert abs(a_alone - a_batch) < 1e-6, (a_alone, a_batch)
    # …and the whole logit vector, not just the one score
    assert torch.allclose(new.next_token_logits([short])[0], new.next_token_logits([short, long])[0], atol=1e-5)
    assert not torch.allclose(old.next_token_logits([short])[0], old.next_token_logits([short, long])[0], atol=1e-4)
    # the Molmo2 shell, same story, and its six processor keys still all go through
    mol = _MolmoShell(m, _StubTok(), device="cpu")
    kw = mol._call_kwargs(mol._encode([short]))
    assert set(kw) == {"input_ids", "attention_mask", "pixel_values", "image_token_pooling",
                       "image_grids", "image_num_crops", "position_ids"}, sorted(kw)
    assert abs(p_A(mol, [short]) - p_A(mol, [short, long])) < 1e-6
    # generate() must NOT receive them (HF builds its own from the cache) — the base class strips them
    assert "position_ids" in new._call_kwargs(new._encode([short]))
    assert len(new.generate([short, long], max_new_tokens=2)) == 2      # would raise if it did not strip
    print(f"  batch positions: p(' A') for {short!r} was {b_alone:.6f} alone / {b_batch:.6f} in a batch "
          f"with a longer prompt (Δ {abs(b_alone - b_batch):.6f}); with content position_ids "
          f"{a_alone:.6f} / {a_batch:.6f} (Δ 0). PaliGemma + Molmo2 shells.")


def _dryrun_shell_ppl(m):
    """`text_eval.shell_text_perplexity` on both shell shapes, weight-free.

    The bug it exists to prevent (2026-09-03): the ppl column ran `model(input_ids=...)` on members
    that live inside a VLM shell, i.e. with no image and none of the processor's framing. Here a
    stub VLM shell reproduces that framing — leading image ids, BOS, and the trailing newline the
    processor appends — and the check is that the readout scores the window's OWN tokens, never the
    image prefix, on both shells."""
    import math

    import torch

    import datasets as _ds
    from infra.pg import text_eval as TE

    class _PrefixForward(Forward):
        """Stub VLM shell: 6 'image' ids + BOS + text + the processor's appended '\\n'."""
        shell = "stub-vlm"

        def _encode(self, texts, images=None):
            rows = [[9] * 6 + [1] + self.tok.encode(t + "\n", add_special_tokens=False) for t in texts]
            S = max(len(r) for r in rows)
            ids = torch.zeros((len(rows), S), dtype=torch.long)
            am = torch.zeros((len(rows), S), dtype=torch.long)
            for r, row in enumerate(rows):
                ids[r, S - len(row):] = torch.tensor(row)
                am[r, S - len(row):] = 1
            return {"input_ids": ids, "attention_mask": am}

        def _call_kwargs(self, enc):
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    class _NoBosTok(_StubTok):
        """Qwen-shaped tokenizer: prepends NO start token, so the text block sits at position 0."""
        def _ids(self, text, add_special_tokens):
            return super()._ids(text, add_special_tokens=False)

    corpus = ("Paris is the capital of France. " * 40) + ("Photosynthesis makes glucose. " * 40)
    orig = _ds.load_dataset
    _ds.load_dataset = lambda *a, **k: {"text": [corpus]}
    try:
        tf = TextForward(m, _StubTok(), device="cpu")
        vf = _PrefixForward(m, _StubTok(), "cpu")
        nf = TextForward(m, _NoBosTok(), device="cpu")
        pt = TE.shell_text_perplexity(tf, window=32, max_tokens=128)
        pv = TE.shell_text_perplexity(vf, window=32, max_tokens=128)
        pn = TE.shell_text_perplexity(nf, window=32, max_tokens=128)
    finally:
        _ds.load_dataset = orig
    assert pt["forward"] == "text shell" and pv["forward"].startswith("stub-vlm shell, blank gray image"), (pt, pv)
    assert pt["n_windows"] == 4 and pv["n_windows"] == 4, (pt["n_windows"], pv["n_windows"])
    # the text shell scores exactly the window's tokens; the VLM shell adds the processor's
    # appended newline (one target per window) and NOTHING from the 6-token image prefix.
    assert pt["n_tokens"] == 128 and pv["n_tokens"] == 128 + 4, (pt["n_tokens"], pv["n_tokens"])
    assert math.isfinite(pt["ppl"]) and math.isfinite(pv["ppl"]) and pt["ppl"] > 1 and pv["ppl"] > 1
    assert pt["protocol"].startswith("shell-framed windows") and pt["protocol"] == pv["protocol"]
    # NO-START-TOKEN shell (row 3a/3b, Qwen): the block starts at 0, so the first target of each
    # window is unscorable and is dropped — one per window, declared in the record. The two
    # start-token shells drop nothing. This is the case that killed 3a_beh on 2026-09-03.
    assert pn["n_windows"] == 4 and pn["n_tokens"] == 128 - 4, (pn["n_windows"], pn["n_tokens"])
    assert pn["dropped_leading_target"] and pn["n_dropped_leading_targets"] == 4, pn["n_dropped_leading_targets"]
    assert not pt["dropped_leading_target"] and not pv["dropped_leading_target"]
    assert pt["n_dropped_leading_targets"] == 0 and pv["n_dropped_leading_targets"] == 0
    assert math.isfinite(pn["ppl"]) and pn["ppl"] > 1
    # a shell whose framing does not contain the window text must FAIL LOUD, never mis-align
    raised = False
    try:
        TE._shell_text_block(_StubTok(), "hello", torch.tensor([9, 9, 1]))
    except AssertionError:
        raised = True
    assert raised, "_shell_text_block must assert when the text block is absent"


if __name__ == "__main__":
    _dryrun()
