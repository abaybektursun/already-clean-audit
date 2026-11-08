"""HUMBOLDT — Phase B shared instrument: the activation projector, the model roster, the corpora.

The projector (§3/Phase B). At each layer, for the residual-stream tensor a bank actually lives in,
record per token  e_U = ‖U Uᵀ h‖² / ‖h‖²  for every bank in the battery. Per-token scalars only.

Site discipline (journal C2). §3 names one site (pre-q/k, post-norm) for every bank. That site is
correct for q/k/v banks — it is exactly the tensor the FUNC ablation acted on, which is what makes
G3 an identification proof — but a gate_proj bank lives in the post-post-attn-LN residual and an
o_proj bank in the attention-head output. Each bank is therefore read at ITS OWN module input:

    q_proj_in     ← q/k/v banks     (= the FUNC site; G3 is unchanged)
    o_proj_in     ← o banks
    gate_proj_in  ← gate/up banks

PG1/PG2-class forwards: EAGER attention only (sdpa mishandles the prefix-LM mask).
gemma-2b text forwards: prepend BOS.
FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import hashlib
import json

from infra.pg import humboldt_core as H

PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"

SITE_MODULE = {"q_proj_in": "q_proj", "o_proj_in": "o_proj", "gate_proj_in": "gate_proj"}
BANK_TYPES_AT_SITE = {"q_proj_in": ("q_proj", "k_proj", "v_proj"),
                      "o_proj_in": ("o_proj",),
                      "gate_proj_in": ("gate_proj", "up_proj")}

# The B1 model roster (§3/B1). model_id → (repo, kind). `kind` selects the loader and the forward.
ROSTER = (
    ("gemma-2b", GEMMA1_REPO, "causal_lm"),
    ("pg1-pt224", PG1_REPO, "paligemma"),
    ("pg1-ft-vqav2", "google/paligemma-3b-ft-vqav2-224", "paligemma"),
    ("pg1-ft-refcoco-seg", "google/paligemma-3b-ft-refcoco-seg-224", "paligemma"),
    ("pi0-reshell", "lerobot/pi0_base", "policy_reshell"),
)

# Corpus candidates, probed before the freeze (the house "introspect, don't assume" pattern from
# battery_tasks.probe_battery_datasets). The probe resolves each slot to the FIRST candidate that
# loads with the named split and field; the resolution is written into the frozen manifest.
CORPUS_CANDIDATES = {
    "wikitext": [{"id": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                  "field": "text"}],
    # 5-way PARALLEL sentences (§3/B1). FLORES-200 is the spec's source and stays first in the
    # ladder; both of its Hub homes are GATED, so the ungated fallbacks preserve the DESIGN
    # (five languages, row-aligned by sentence id, same content across languages) with a different
    # source. Whichever resolves is recorded in the manifest and flagged on every row.
    "parallel": [{"id": "openlanguagedata/flores_plus", "config": "{lang}", "split": "dev",
                  "langs": ["eng_Latn", "zho_Hans", "arb_Arab", "swh_Latn", "rus_Cyrl"],
                  "field": "text", "id_field": "id"},
                 {"id": "facebook/flores", "config": "{lang}", "split": "dev",
                  "langs": ["eng_Latn", "zho_Hans", "arb_Arab", "swh_Latn", "rus_Cyrl"],
                  "field": "sentence", "id_field": None},
                 {"id": "Muennighoff/flores200", "config": "{lang}", "split": "dev",
                  "langs": ["eng_Latn", "zho_Hans", "arb_Arab", "swh_Latn", "rus_Cyrl"],
                  "field": "sentence", "id_field": None},
                 {"id": "facebook/xnli", "config": "{lang}", "split": "validation",
                  "langs": ["en", "zh", "ar", "sw", "ru"],
                  "field": "premise", "id_field": None}],
    "code": [{"id": "codeparrot/codeparrot-clean-valid", "config": None, "split": "train",
              "field": "content"},
             {"id": "bigcode/the-stack-smol", "config": "data/python", "split": "train",
              "field": "content"},
             {"id": "google/code_x_glue_ct_code_to_text", "config": "python", "split": "test",
              "field": "code"}],
    "vqav2": [{"id": "merve/vqav2-small", "config": None, "split": "validation",
               "field": "question", "image_field": "image", "id_field": "question_id"},
              {"id": "merve/vqav2-small", "config": None, "split": "train",
               "field": "question", "image_field": "image", "id_field": "question_id"},
              {"id": "lmms-lab/VQAv2", "config": None, "split": "validation",
               "field": "question", "image_field": "image", "id_field": "question_id"},
              {"id": "HuggingFaceM4/VQAv2", "config": None, "split": "validation",
               "field": "question", "image_field": "image", "id_field": "question_id"}],
}
# Text-field aliases the probe will accept, in order, when a candidate's declared field is absent.
FIELD_ALIASES = ("text", "sentence", "premise", "content", "code", "question", "func_code_string")
REPETITION_PERIODS = (2, 4, 16, 64)


def alignment_hash(token_ids) -> str:
    """sha1 of a token-id sequence. Paired conditions that must be token-identical (real vs gray
    image; the query block across shot counts) are gated on this (§1 kill-table, last row)."""
    return hashlib.sha1(json.dumps([int(t) for t in token_ids]).encode()).hexdigest()[:16]


def content_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The projector.
# ---------------------------------------------------------------------------
class SiteEnergy:
    """forward_pre_hooks on the three site modules of every decoder layer. Per forward, appends a
    (T, n_banks) fp32 array of e_U per (layer, site). Banks are stacked into one matrix per
    (layer, site) so the readout is a single GEMM. Batch size must be 1 — padding would silently
    contaminate per-token rows, and there is no throughput reason to batch a 250k-token sweep."""

    def __init__(self, layers, bank_dir, device, sites=H.SITES_PHASE_B):
        import numpy as np
        import torch
        from infra.pg import humboldt_weights as WT
        self.layers = layers
        self.sites = tuple(sites)
        self.order = {}          # site -> [(type, bank_id), ...]
        self.stack = {}          # (L, site) -> torch (n, nb*32) fp32 on device
        for site in self.sites:
            order = [(t, bid) for t in BANK_TYPES_AT_SITE[site] for bid in H.BANK_IDS]
            self.order[site] = order
            for L in range(len(layers)):
                cols = [WT.load_bank(bank_dir, t, L, bid) for t, bid in order]
                self.stack[(L, site)] = torch.as_tensor(
                    np.concatenate(cols, axis=1), device=device, dtype=torch.float32)
        self.series = {}         # (L, site) -> list of (T, nb) fp32 numpy
        self._handles = []

    def n_banks(self, site):
        return len(self.order[site])

    def _mk(self, L, site):
        B = self.stack[(L, site)]
        nb = self.n_banks(site)

        def pre_hook(_m, args):
            x = args[0]
            assert x.shape[0] == 1, f"SiteEnergy requires batch 1, got {x.shape[0]}"
            Xc = x.reshape(-1, x.shape[-1]).float()          # (T, n) fp32 (§8: fp32 accumulation)
            C = Xc @ B                                       # (T, nb*32)
            num = C.pow(2).reshape(Xc.shape[0], nb, H.BANK_K).sum(-1)
            den = Xc.pow(2).sum(-1, keepdim=True)
            assert bool((den > 0).all()), "zero-norm residual token — e_U undefined"
            self.series[(L, site)].append((num / den).detach().cpu().numpy())
            # record only (return None == no-op)
        return pre_hook

    def reset(self):
        self.series = {(L, s): [] for L in range(len(self.layers)) for s in self.sites}

    def __enter__(self):
        self.reset()
        for L, layer in enumerate(self.layers):
            for site in self.sites:
                mod = getattr(layer.self_attn if site in ("q_proj_in", "o_proj_in") else layer.mlp,
                              SITE_MODULE[site])
                self._handles.append(mod.register_forward_pre_hook(self._mk(L, site)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def collect(self):
        """{(L, site): (T_total, nb) fp32} concatenated over the forwards since the last reset."""
        import numpy as np
        out = {}
        for k, chunks in self.series.items():
            assert chunks, f"no activations captured at {k}"
            out[k] = np.concatenate(chunks, axis=0)
        return out


class SiteSlice:
    """B3's capture: the RAW residual over a token slice, per layer per site, batch 1.
    `set_window(start, length)` must be called before each forward; the slice is the query block
    located by exact token-block match, so Δh across shot counts is elementwise well-defined."""

    def __init__(self, layers, device, sites=H.SITES_PHASE_B):
        self.layers = layers
        self.sites = tuple(sites)
        self.buf = {}
        self._win = None
        self._handles = []

    def set_window(self, start: int, length: int):
        self._win = (int(start), int(length))
        self.buf = {}

    def _mk(self, L, site):
        def pre_hook(_m, args):
            x = args[0]
            assert x.shape[0] == 1, f"SiteSlice requires batch 1, got {x.shape[0]}"
            s, n = self._win
            assert s + n <= x.shape[1], f"window [{s},{s + n}) outside sequence {x.shape[1]}"
            self.buf[(L, site)] = x[0, s:s + n, :].float().detach().cpu().numpy()
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            for site in self.sites:
                mod = getattr(layer.self_attn if site in ("q_proj_in", "o_proj_in") else layer.mlp,
                              SITE_MODULE[site])
                self._handles.append(mod.register_forward_pre_hook(self._mk(L, site)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class BankInject:
    """B4's micro-intervention: add ε·Uz to the site tensor at ONE layer (§3/B4, one layer at a time
    enforced by construction — the class holds exactly one layer). z is a unit-norm random in-bank
    coefficient vector; the added vector is ε·‖h‖-free (absolute), calibrated by the caller."""

    def __init__(self, layer, site, U, z, eps, device):
        import numpy as np
        import torch
        v = np.asarray(U, dtype=np.float64) @ np.asarray(z, dtype=np.float64)
        v = v / np.linalg.norm(v)
        self.v = torch.as_tensor(v, device=device, dtype=torch.float32)
        self.eps = float(eps)
        self.layer, self.site = layer, site
        self._handles = []

    def _mk(self):
        def pre_hook(_m, args):
            x = args[0]
            d = (self.eps * self.v).to(x.dtype)
            return (x + d,) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        mod = getattr(self.layer.self_attn if self.site in ("q_proj_in", "o_proj_in")
                      else self.layer.mlp, SITE_MODULE[self.site])
        self._handles.append(mod.register_forward_pre_hook(self._mk()))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def js_divergence(p, q, eps: float = 1e-12):
    """Mean Jensen-Shannon divergence between two attention maps over the last axis (nats)."""
    import numpy as np
    a = np.asarray(p, dtype=np.float64) + eps
    b = np.asarray(q, dtype=np.float64) + eps
    a = a / a.sum(-1, keepdims=True)
    b = b / b.sum(-1, keepdims=True)
    m = 0.5 * (a + b)
    kl = lambda x, y: (x * (np.log(x) - np.log(y))).sum(-1)    # noqa: E731
    return float((0.5 * kl(a, m) + 0.5 * kl(b, m)).mean())


# ---------------------------------------------------------------------------
# Model roster.
# ---------------------------------------------------------------------------
def load_model(model_id, repo, kind, device="cuda", sha=None):
    """Returns (model, processor_or_tokenizer, decoder_layers, resolved_sha, is_multimodal)."""
    import torch
    from huggingface_hub import HfApi
    api = HfApi()
    if kind == "causal_lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        s = sha or api.model_info(repo).sha
        m = AutoModelForCausalLM.from_pretrained(repo, revision=s, torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 attn_implementation="eager").eval()
        return m, AutoTokenizer.from_pretrained(repo, revision=s), list(m.model.layers), s, False
    if kind == "paligemma":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        from infra.aws_pg.jobs.invariant_core_func import _decoder_layers
        s = sha or api.model_info(repo).sha
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=s, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        return m, AutoProcessor.from_pretrained(repo, revision=s), _decoder_layers(m), s, True
    if kind == "policy_reshell":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        from infra.aws_pg.jobs.invariant_core_func import _decoder_layers
        from infra.pg import lacuna_shell as LS
        pg1_sha = api.model_info(PG1_REPO).sha
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            PG1_REPO, revision=pg1_sha, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        info = LS.build_row_policy(m, "pi0", embed="own")     # LACUNA recipe; embed-size trap inside
        print(f"[hum] pi0 re-shell installed: {json.dumps(info, default=str)}", flush=True)
        s = api.model_info(repo).sha
        proc = AutoProcessor.from_pretrained(PG1_REPO, revision=pg1_sha)
        return m, proc, _decoder_layers(m), f"{s}+shell@{pg1_sha[:8]}", True
    raise AssertionError(f"unknown model kind {kind}")


# ---------------------------------------------------------------------------
# Corpora.
# ---------------------------------------------------------------------------
def _tok_ids(tokenizer, text):
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def build_text_corpora(tokenizer, resolved, n_tokens=50_000, window=512, long_doc_tokens=4096,
                       seed=1337):
    """The five text corpora (§3/B1), as lists of window dicts:
       {corpus_id, item_id, token_class, text}. Token classes label what a later mining pass needs
       to condition on (language, period, position role)."""
    import numpy as np
    from datasets import load_dataset

    out = {}

    # --- wikitext slice + the long document (one article > long_doc_tokens).
    wt = resolved["wikitext"]
    ds = load_dataset(wt["id"], wt["config"], split=wt["split"])
    lines = [t for t in ds[wt.get("resolved_field", wt["field"])] if t.strip()]
    text = "\n\n".join(lines)
    ids = _tok_ids(tokenizer, text)[:n_tokens]
    out["wikitext"] = [{"corpus_id": "wikitext", "item_id": f"w{i}", "token_class": "text",
                        "text": tokenizer.decode(ids[i:i + window].tolist())}
                       for i in range(0, len(ids) - window + 1, window)]
    # The long document (§3/B1: one document, >4k tokens, for the position sweep). wikitext-2-raw
    # marks TOP-LEVEL articles as " = Title = " and SUBsections as " = = Section = = "; splitting on
    # any heading cuts articles into sections, none of which reaches 4k tokens. Split on top-level
    # headings only, then take the LONGEST article — deterministic, and it does not depend on the
    # first long-enough article happening to exist early in the file.
    def _is_article_heading(ln):
        t = ln.strip()
        return t.startswith("=") and t.endswith("=") and not t.startswith("= =")

    articles, cur = [], []
    for ln in lines:
        if _is_article_heading(ln) and cur:
            articles.append("\n".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        articles.append("\n".join(cur))
    assert articles, "wikitext yielded no articles"
    lens = [len(_tok_ids(tokenizer, art)) for art in articles]
    best = int(max(range(len(articles)), key=lambda i: lens[i]))
    assert lens[best] >= 1024, (
        f"longest wikitext article is only {lens[best]} tokens — too short for a position sweep "
        f"(n_articles={len(articles)}, top lengths={sorted(lens)[-5:]})")
    doc_ids = _tok_ids(tokenizer, articles[best])[:long_doc_tokens]
    out["longdoc"] = [{"corpus_id": "longdoc", "item_id": f"doc{best}", "token_class": "longdoc",
                       "text": tokenizer.decode(doc_ids.tolist())}]

    # --- 5-way parallel sentences, paired by sentence id across languages.
    fl = resolved["parallel"]
    langs = fl["langs"]
    field = fl.get("resolved_field", fl["field"])
    per_lang, ids_by_lang = {}, {}
    for lang in langs:
        d = load_dataset(fl["id"], fl["config"].format(lang=lang), split=fl["split"])
        sids = list(range(len(d))) if not fl["id_field"] else [str(x) for x in d[fl["id_field"]]]
        per_lang[lang] = [str(x) for x in d[field]]
        ids_by_lang[lang] = [str(x) for x in sids]
    common = sorted(set.intersection(*[set(v) for v in ids_by_lang.values()]))
    assert len(common) >= 100, f"parallel-corpus shared-id set too small: {len(common)}"
    pos = {lang: {sid: j for j, sid in enumerate(ids_by_lang[lang])} for lang in langs}
    # Some parallel corpora repeat a sentence across rows (XNLI carries each premise three times,
    # once per hypothesis). Keep the FIRST id per distinct reference-language sentence: the pairing
    # is by id, so filtering ids preserves alignment in every language, and repeated sentences would
    # otherwise weight a handful of texts three-fold in every per-language mean.
    ref = langs[0]
    seen, dedup = set(), []
    for sid in common:
        txt = per_lang[ref][pos[ref][sid]]
        if txt in seen:
            continue
        seen.add(txt)
        dedup.append(sid)
    common = dedup
    assert len(common) >= 100, f"parallel corpus has only {len(common)} distinct sentences"
    rows, budget = [], {lang: 0 for lang in langs}
    for sid in common:
        if min(budget.values()) >= n_tokens // len(langs):
            break
        for lang in langs:
            sent = per_lang[lang][pos[lang][sid]]
            budget[lang] += len(_tok_ids(tokenizer, sent))
            rows.append({"corpus_id": "parallel", "item_id": f"s{sid}", "token_class": lang,
                         "text": sent})
    out["parallel"] = rows

    # --- python code slice.
    cd = resolved["code"]
    dsc = load_dataset(cd["id"], cd["config"], split=cd["split"]) if cd["config"] \
        else load_dataset(cd["id"], split=cd["split"])
    # select() BEFORE indexing the column: `dsc[field][:2000]` materialises the WHOLE column first,
    # which for a code corpus is hundreds of megabytes of strings for 2000 rows of use.
    n_take = min(2000, len(dsc))
    ctext = "\n\n".join(str(x) for x in dsc.select(range(n_take))[cd.get("resolved_field",
                                                                         cd["field"])])
    cids = _tok_ids(tokenizer, ctext)[:n_tokens]
    out["code"] = [{"corpus_id": "code", "item_id": f"c{i}", "token_class": "code",
                    "text": tokenizer.decode(cids[i:i + window].tolist())}
                   for i in range(0, len(cids) - window + 1, window)]

    # --- pathological repetition suite (constructed; no dataset). Alphabet = the token ids of a
    # fixed set of common ASCII words, so every model's tokenizer produces the same STRUCTURE.
    rng = np.random.default_rng(seed)
    words = ["the", "cat", "run", "blue", "seven", "table", "quick", "stone", "river", "green",
             "north", "paper", "silver", "window", "orange", "planet", "yellow", "candle",
             "mountain", "printer", "diamond", "kitchen", "elephant", "notebook", "telescope",
             "umbrella", "carpenter", "helicopter", "strawberry", "microscope", "thunderstorm",
             "encyclopedia", "refrigerator", "constellation", "photosynthesis", "bibliography",
             "characteristic", "responsibility", "internationally", "extraordinarily",
             "misunderstanding", "incomprehensible", "counterproductive", "indistinguishable",
             "electroencephalogram", "disproportionately", "uncharacteristically",
             "compartmentalisation", "interdisciplinary", "hyperventilating", "circumnavigate",
             "phosphorescence", "juxtaposition", "onomatopoeia", "kaleidoscope", "labyrinthine",
             "metamorphosis", "nomenclature", "obfuscation", "palindrome", "quintessential",
             "reverberation", "serendipity", "transcendental"]
    rep = []
    for p in REPETITION_PERIODS:
        cycle = words[:p]
        seq = (cycle * (window // p + 1))[:window]
        rep.append({"corpus_id": "repetition", "item_id": f"period{p}",
                    "token_class": f"period{p}", "text": " ".join(seq)})
        shuf = list(seq)
        rng.shuffle(shuf)
        rep.append({"corpus_id": "repetition", "item_id": f"shuffled_period{p}",
                    "token_class": f"shuffled_period{p}", "text": " ".join(shuf)})
    # monotone runs (§3/B1) — two kinds, each with its shuffled control:
    #   numeric   an ascending integer count, the classic monotone token sequence
    #   length    the word list ordered by increasing length (monotone in surface form)
    num = [str(i) for i in range(1, window + 1)]
    lng = sorted((words * (window // len(words) + 1))[:window], key=len)
    for name, seq in (("monotone_numeric", num), ("monotone_length", lng)):
        rep.append({"corpus_id": "repetition", "item_id": name, "token_class": name,
                    "text": " ".join(seq)})
        sh = list(seq)
        rng.shuffle(sh)
        rep.append({"corpus_id": "repetition", "item_id": f"shuffled_{name}",
                    "token_class": f"shuffled_{name}", "text": " ".join(sh)})
    out["repetition"] = rep
    return out


def build_vqav2_items(resolved, n_items=64, seed=1337):
    """VQAv2 image+question items with their gray-128 twin, paired by question id (§3/B1)."""
    from datasets import load_dataset
    v = resolved["vqav2"]
    ds = load_dataset(v["id"], split=v["split"], streaming=True)
    field = v.get("resolved_field", v["field"])
    items = []
    for i, row in enumerate(ds):
        iid = str(row[v["id_field"]]) if v.get("id_field") and v["id_field"] in row else f"row{i}"
        items.append({"item_id": iid, "question": str(row[field]),
                      "image": row[v["image_field"]]})
        if len(items) >= n_items:
            break
    # The item count is a sample size, not a scientific threshold: if the chosen VQAv2 mirror is
    # smaller than REQUESTED, record what it yielded rather than losing the arm. The floor is
    # min(n_items, 8) — a shortfall against a real request of 64 is fatal for the paired contrast,
    # but a smoke that deliberately asks for 4 is not a shortfall at all.
    floor = min(n_items, 8)
    assert len(items) >= floor, \
        f"VQAv2 yielded {len(items)} items, below the floor {floor} for a request of {n_items}"
    if len(items) < n_items:
        print(f"[hum] VQAv2 yielded {len(items)} of {n_items} requested items — recording {len(items)}",
              flush=True)
    return items
