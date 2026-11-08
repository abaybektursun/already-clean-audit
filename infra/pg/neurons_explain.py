"""D1 explain side, N3 (SPEC_D1 §2.2-§2.4; DATA_CONTRACT §3). Laptop only: banked files and LLM calls.

Inputs: a harvest `out/` directory (topk/, stats/, poles/poles.json, poles/vocab.json, manifest.json)
and the corpus directory `corpus/` (tok/<tokname>/tokens.npy, doc_offsets.npy, tokenizer/). Every
context is rendered from tokens.npy at the width its role needs (SPEC §2.1 as amended: 256 tokens
before for the explainer and clusterer, 64 for the judge, 16 after for all). Outputs go under `<out>/explain/`. Each batch file holds one role (and one sample for
the clusterer and the judge), so one answering agent sees one role only.

Subcommands, in pipeline order:
    render            print one pole's contexts (--before tokens before, 16 after, document-bounded)
    prepare           explainer and second-reader requests for a pole list
    collect           -> explanations.jsonl
    prepare-writer    writer requests (sentence only) and writer_job requests for UNREL candidates
    collect-writer    -> texts.jsonl, shuffle.json (same-layer shuffle pole, logged seed), probe_poles.json
    positions         T_15 top-50 positions of S -> positions.npz, positions_poles.json (W3 input)
    prepare-clusters  two clustering samples of each pole's T_15 top-50
    collect-clusters  -> clusters.jsonl (each answer must partition the top-50)
    prepare-judge     2AFC parent tests and positive controls; births from neurons_verdict
    collect-judge     -> judgments.jsonl
    reject            move answers collect refuses to responses/<batch>.rejected.jsonl, so `run` asks
                      for them again (collect -> reject -> run -> collect, at most --max-attempts rounds)
    fork-writer       copy an explain directory without its writer step, to redo the writer (another
                      model) on the same explanations; the source is never written
    remodel           set the model of a role's request lines (all, or only the unanswered ones), so a
                      batch can be resumed on another model; recorded in the manifest
`explain/plans/` maps the opaque clusterer and judge request ids to poles, clusters, decoys and the
shown order; it is never shown to an answering agent.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import string
import unicodedata
import urllib.parse
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from infra.pg import neurons_llm as llm
from infra.pg.neurons_verdict import (GRID_LABELS, HARVEST_FILES, T_CONTROL_BEFORE, file_sha256, grid_cluster_tables,
                                      layer_file, parse_pole, pole_id, pole_stats, pole_vocab, read_json)

PROMPT_DIR = Path(__file__).parent / "neurons_prompts"
TEMPERATURE = 1.0          # every role; see neurons_llm.anthropic_kwargs
MAX_TOKENS = {"explainer": 16000, "second_reader": 16000, "writer": 16000, "writer_job": 8000,
              "clusterer": 16000, "judge": 8000}
SAMPLE_INDEX = {"explainer": 0, "second_reader": 1}
N_EXPLAIN_CONTEXTS = 30    # §2.2
N_TOPK = 50                # §2.1
EXPLAIN_BEFORE = 256       # §2.1 as amended: explainer and clusterer contexts, and the judge's exclusion window
JUDGE_BEFORE = 64          # §2.1: the judge's exemplars and records, so two records fit one call
AFTER = 16                 # §2.1
NOTE_WINDOW = 16           # X7: tokens either side of the marked token searched for encodings
WRITER_ROLES = ("writer", "writer_job")
WRITER_FILES = ("texts.jsonl", "shuffle.json", "probe_poles.json")      # what collect-writer derives
MIN_TEXT_TOKENS = 2        # the probe scores text positions >= 1, so a text needs two tokens
PROMPT_VERSION = "v3"      # W2-D prompt version 3 (§9.2: at most three versions, each logged)
CLUSTERER_PROMPTS = {      # clusterer.md SHA-256 by version (§4.4: tuned on S_gate at W4)
    "v1": "557a806ddfedd79791644740595bd6de51c89682584a3cb45c015a6dd1805d67",
    "v2": "5d8b55cacc3d1a4ffdb1d17e43f2b2d260a5149b99aa2e9a12d088f036e0c1ff",   # 2026-10-21: the partition stated
}
CLUSTERER_PROMPT_VERSION = "v2"
N_JUDGE_EXEMPLARS = 10     # §2.4
N_JUDGE_RECORD = 20        # §2.4
PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
# What may not reach the blind writer from an explanation or a job sentence (SPEC §2.2, B2 of the review):
# pointers at the material the explainer was shown, or at activation values. Digits and the word
# "context" in a description of the job ("B2B", "4.5", "in the context of web styling") are allowed.
LEAK_RES = (("a context marker", re.compile(r"[⟦⟧]")),
            ("an activation value", re.compile(r"\b[agu]\s*=\s*-?\d")),
            ("a reference to a shown context", re.compile(r"\bcontexts?\s*#?\s*\d+", re.IGNORECASE)),
            ("a reference to a shown context", re.compile(r"#\d+")),
            ("a reference to a shown context",
             re.compile(r"\bcontexts?\s+(above|shown|provided|given|listed|supplied|I saw|I was shown)\b", re.IGNORECASE)))


def _strings(n: int) -> dict:
    return {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"}}


def _object(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


EXPLAINER_SCHEMA = _object({
    "explanation": {"type": "string"},
    "jobs": {"type": "array", "minItems": 1, "items": _object({
        "name": {"type": "string"}, "related_to_others": {"type": "boolean"}, "reasoning": {"type": "string"}})},
    "class": {"type": "string", "enum": ["ONE", "TWO-REL", "UNREL", "UNK"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}})
WRITER_SCHEMA = _object({"positives": _strings(5), "near_misses": _strings(5),
                         "distant_concept": {"type": "string"}, "distant": _strings(5),
                         "max": {"type": "string"}})
WRITER_JOB_SCHEMA = _object({"positives": _strings(5)})
CLUSTERER_SCHEMA = _object({
    "clusters": {"type": "array", "minItems": 1, "items": _object({
        "c": {"type": "integer", "minimum": 0}, "name": {"type": "string"}, "echo": {"type": "boolean"},
        "exemplar_ranks": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}}})},
    "pairs": {"type": "array", "items": _object({
        "a": {"type": "integer", "minimum": 0}, "b": {"type": "integer", "minimum": 0},
        "relation": {"type": "string", "enum": ["related", "unrelated"]}})}})
JUDGE_SCHEMA = _object({"answer": {"type": "string", "enum": ["A", "B"]}, "reason": {"type": "string"}})
SCHEMAS = {"explainer": EXPLAINER_SCHEMA, "second_reader": EXPLAINER_SCHEMA, "writer": WRITER_SCHEMA,
           "writer_job": WRITER_JOB_SCHEMA, "clusterer": CLUSTERER_SCHEMA, "judge": JUDGE_SCHEMA}


# --- prompts and the explain manifest --------------------------------------------------------

def load_prompt(role: str) -> tuple[str, str, str]:
    """(system text, user template, SHA-256 of the file) from neurons_prompts/<role>.md."""
    text = (PROMPT_DIR / f"{role}.md").read_text(encoding="utf-8")
    head, sep, user = text.partition("<!-- user -->")
    if not sep or not head.startswith("<!-- system -->"):
        raise ValueError(f"{role}.md: expected '<!-- system -->' ... '<!-- user -->' ...")
    system = head[len("<!-- system -->"):].strip()
    if PLACEHOLDER_RE.search(system):
        raise ValueError(f"{role}.md: the system part has a placeholder")
    return system, user.strip(), llm.sha256_text(text)


def fill(template: str, values: dict) -> str:
    """Replace every {name} in the template; an unknown placeholder or an unused value raises."""
    used = set()

    def sub(m):
        used.add(m.group(1))
        return str(values[m.group(1)])
    out = PLACEHOLDER_RE.sub(sub, template)
    unused = set(values) - used
    if unused:
        raise ValueError(f"values not used by the template: {sorted(unused)}")
    return out


def build_request(role: str, rid: str, model: str, values: dict) -> dict:
    system, user, _ = load_prompt(role)
    return llm.make_request(rid, role, model, TEMPERATURE, MAX_TOKENS[role], system, fill(user, values),
                            SCHEMAS[role])


def ensure_manifest(explain: Path) -> dict:
    """Create explain/manifest.json (started_utc, prompt SHA-256s) or check the prompts against it."""
    hashes = {role: load_prompt(role)[2] for role in llm.ROLES}
    if hashes["second_reader"] != hashes["explainer"]:
        raise ValueError("second_reader.md must equal explainer.md: the second reader is an "
                         "independent sample of the explainer's call")
    path = explain / "manifest.json"
    if path.exists():
        m = read_json(path)
        changed = sorted(r for r in hashes if m["prompts"].get(r) != hashes[r])
        if changed or m["prompt_version"] != PROMPT_VERSION:
            raise ValueError(f"{path}: prompts {changed} changed since {m['started_utc']} (prompt version "
                             f"{m['prompt_version']} -> {PROMPT_VERSION}); use a new explain directory")
        return m
    if hashes["clusterer"] != CLUSTERER_PROMPTS[CLUSTERER_PROMPT_VERSION]:
        raise ValueError(f"clusterer.md has SHA-256 {hashes['clusterer']}, not clusterer prompt "
                         f"{CLUSTERER_PROMPT_VERSION} ({CLUSTERER_PROMPTS[CLUSTERER_PROMPT_VERSION]}); record a new version")
    m = {"started_utc": llm.utc_now(), "prompt_version": PROMPT_VERSION, "clusterer_prompt_version": CLUSTERER_PROMPT_VERSION,
         "prompts": hashes,
         "prompt_dir": str(PROMPT_DIR), "context_before": {"explainer": EXPLAIN_BEFORE, "clusterer": EXPLAIN_BEFORE,
                                                            "judge": JUDGE_BEFORE, "after": AFTER},
         "seeds": {}, "batches": []}
    explain.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1))
    return m


def note_in_manifest(explain: Path, batches: list[dict] = (), seeds: dict | None = None) -> None:
    path = explain / "manifest.json"
    m = read_json(path)
    m["batches"] += list(batches)
    for k, v in (seeds or {}).items():
        if k in m["seeds"] and m["seeds"][k] != v:
            raise ValueError(f"{path}: seed {k} already recorded as {m['seeds'][k]}")
        m["seeds"][k] = v
    path.write_text(json.dumps(m, indent=1))


def record_models(explain: Path) -> None:
    """explain/manifest.json "models_by_role": per role, every model that answered (from the response
    lines) and every model the request lines name, with counts. A mixed batch is recorded as mixed."""
    answered: dict[str, Counter] = {}
    requested: dict[str, Counter] = {}
    for path in sorted((explain / "requests").glob("*.jsonl")):
        for r in llm.read_jsonl(path):
            requested.setdefault(r["role"], Counter())[r["model"]] += 1
        for rid, model in llm.response_models(explain, path.stem).items():
            answered.setdefault(rid.split(":")[0], Counter())[model] += 1
    path = explain / "manifest.json"
    m = read_json(path)
    m["models_by_role"] = {"answered": {k: dict(v) for k, v in sorted(answered.items())},
                           "requested": {k: dict(v) for k, v in sorted(requested.items())}, "utc": llm.utc_now()}
    path.write_text(json.dumps(m, indent=1))


def write_batch(explain: Path, batch: str, requests: list[dict]) -> None:
    llm.write_requests(explain, batch, requests)
    note_in_manifest(explain, [{"batch": batch, "n": len(requests), "utc": llm.utc_now(),
                                "roles": sorted({r["role"] for r in requests}),
                                "models": sorted({r["model"] for r in requests})}])


# --- contexts --------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def corpus(corpus_dir: str, tokname: str):
    """(tokens memmap, doc_offsets, tokenizer) for one tokenizer of the corpus."""
    from transformers import AutoTokenizer
    base = Path(corpus_dir) / "tok" / tokname
    return (np.load(base / "tokens.npy", mmap_mode="r"), np.load(base / "doc_offsets.npy"),
            AutoTokenizer.from_pretrained(str(base / "tokenizer")))


@lru_cache(maxsize=None)
def harvest_corpus(harvest, corpus_dir):
    """corpus() for the tokenizer `harvest` scored with. Refuses a corpus directory whose manifest.json is
    not the one the harvest scored (its SHA-256 against the harvest's corpus_manifest_sha256): token
    positions index that corpus only, and another one renders other text without an error."""
    m = read_json(Path(harvest) / HARVEST_FILES["manifest"])
    got = file_sha256(Path(corpus_dir) / "manifest.json")
    if got != m["corpus_manifest_sha256"]:
        raise ValueError(f"{corpus_dir}/manifest.json has SHA-256 {got}; {harvest} scored the corpus "
                         f"{m['corpus_manifest_sha256']}")
    return corpus(str(corpus_dir), m["tokname"])


def render(tokens, offsets, tokenizer, tok: int, before: int, after: int = AFTER) -> dict:
    """The context of global token `tok`: up to `before` tokens before and `after` after, inside its document."""
    doc = int(np.searchsorted(offsets, tok, side="right") - 1)
    start, end = int(offsets[doc]), int(offsets[doc + 1])
    if not start <= tok < end:
        raise ValueError(f"token {tok} outside the corpus")
    lo, hi = max(start, tok - before), min(end, tok + 1 + after)

    def dec(a, b):
        return tokenizer.decode([int(x) for x in tokens[a:b]])
    return {"doc": doc, "pos": tok - start, "before": dec(lo, tok), "token": dec(tok, tok + 1),
            "after": dec(tok + 1, hi)}


def pole_contexts(harvest, corpus_dir, pole: str, n: int, before: int) -> list[dict]:
    """The pole's top-n contexts in rank order: rank, tok, a, g, u, doc, pos, before, token, after,
    rendered from tokens.npy with the saved tokenizer, `before` tokens before the marked token."""
    layer, neuron, s = parse_pole(pole)
    tk = layer_file(str(harvest), "topk", layer)
    toks = [int(t) for t in tk["tok"][neuron, s][:n] if t >= 0]
    tokens, offsets, tokenizer = harvest_corpus(harvest, corpus_dir)
    return [{"rank": i, "tok": t, "a": float(tk["a"][neuron, s, i]), "g": float(tk["g"][neuron, s, i]),
             "u": float(tk["u"][neuron, s, i]), **render(tokens, offsets, tokenizer, t, before)}
            for i, t in enumerate(toks)]


def one_line(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "↵")


def marked(ctx: dict) -> str:
    return one_line(ctx["before"]) + "⟦" + one_line(ctx["token"]) + "⟧" + one_line(ctx["after"])


def regime(g: float, u: float) -> str:
    return f"g{'+' if g >= 0 else '-'} u{'+' if u >= 0 else '-'}"


# --- X7: decode the noise (SPEC §0c) ---------------------------------------------------------

B64_RE = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")
URL_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){2,}")
ROMAN = {**dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "a b v g d e yo zh z i y k l m n o p r s t u f kh ts ch sh shch \" y ' e yu ya".split(), strict=True)),
         **dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω", "a v g d e z i th i k l m n x o p r s s t y f ch ps o".split(), strict=True)),
         **dict(zip("অআইঈউঊঋএঐওঔ", "a a i i u u ri e oi o ou".split(), strict=True)),
         **{c: r + "a" for c, r in zip("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ",
                                       "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n p ph b bh m y r l sh sh s h".split(),
                                       strict=True)},
         **dict(zip("ািীুূৃেৈোৌ", "a i i u u ri e oi o ou".split(), strict=True)),
         "্": "", "ং": "ng", "ঃ": "h", "ঁ": "n", "ৎ": "t"}
# Script by Unicode block, for these scripts only (the stdlib has no script property). A character
# outside these blocks, such as MICRO SIGN or a modifier letter, belongs to no script here.
SCRIPT_BLOCKS = (("CYRILLIC", 0x0400, 0x052F), ("GREEK", 0x0370, 0x03FF), ("GREEK", 0x1F00, 0x1FFF),
                 ("HEBREW", 0x0590, 0x05FF), ("ARABIC", 0x0600, 0x06FF), ("ARABIC", 0x0750, 0x077F),
                 ("DEVANAGARI", 0x0900, 0x097F), ("BENGALI", 0x0980, 0x09FF), ("THAI", 0x0E00, 0x0E7F),
                 ("HANGUL", 0x1100, 0x11FF), ("HANGUL", 0x3130, 0x318F), ("HANGUL", 0xAC00, 0xD7AF),
                 ("KANA", 0x3040, 0x30FF), ("KANA", 0x31F0, 0x31FF), ("CJK", 0x3400, 0x4DBF),
                 ("CJK", 0x4E00, 0x9FFF), ("CJK", 0xF900, 0xFAFF), ("CJK", 0x20000, 0x2FA1F))
PLAIN = set(string.ascii_letters + string.digits + " \t\n.,;:!?'\"()-/")   # what a decoded line must mostly be
BN_CONSONANTS = set("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ")
BN_SIGNS = set("ািীুূৃেৈোৌ্")              # vowel signs and the virama replace a consonant's inherent "a"
ROMANIZED = ("CYRILLIC", "GREEK", "BENGALI")


def romanize(text: str) -> str:
    out, after_consonant = [], False
    for ch in unicodedata.normalize("NFD", text):
        if ch in BN_SIGNS:
            if after_consonant:
                out[-1] = out[-1][:-1]
            out.append(ROMAN[ch])
            after_consonant = False
        elif ch.lower() in ROMAN:
            out.append(ROMAN[ch.lower()])
            after_consonant = ch in BN_CONSONANTS
        elif unicodedata.category(ch) != "Mn":                 # combining accents are dropped
            out.append(ch)
            after_consonant = False
    return "".join(out)


def readable(text: str) -> bool:
    """Decoded text a reader could use: at least four characters, at least 90% of them ASCII letters,
    digits, spaces or common punctuation. Random identifiers that are valid Base64 decode to bytes
    that fail this."""
    t = text.strip()
    return len(t) >= 4 and sum(c in PLAIN for c in t) >= 0.9 * len(t)


def script_of(ch: str) -> str | None:
    if unicodedata.category(ch)[0] not in "LM":
        return None
    cp = ord(ch)
    return next((name for name, lo, hi in SCRIPT_BLOCKS if lo <= cp <= hi), None)


def base64_text(s: str) -> str | None:
    """The UTF-8 text a Base64 run decodes to, or None. Only runs with mixed case, a digit, + or / or
    padding are tried; the length rule keeps b64decode from raising."""
    core = s.rstrip("=")
    if len(core) % 4 == 1 or not (re.search(r"[0-9+/=]", s) or (re.search(r"[A-Z]", s) and re.search(r"[a-z]", s))):
        return None
    text = base64.b64decode(core + "=" * (-len(core) % 4)).decode("utf-8", errors="replace")
    return text if readable(text) else None


def encoding_notes(text: str) -> list[str]:
    """X7: a "decoded:" line for Base64 or URL-encoded runs that decode to text a reader could use
    (`readable`), and a "transliterated:" line for letters of the scripts in SCRIPT_BLOCKS (romanized
    for Cyrillic, Greek and Bengali; the others are named and marked not transliterated)."""
    notes = []
    decoded = [d for d in (base64_text(m.group()) for m in B64_RE.finditer(text)) if d]
    decoded += [d for d in (urllib.parse.unquote(m.group(), errors="replace") for m in URL_RE.finditer(text)) if readable(d)]
    if decoded:
        notes.append("decoded: " + " | ".join(one_line(d)[:200] for d in decoded))
    scripts = sorted({script_of(c) for c in text} - {None})
    if scripts:
        parts = []
        if set(scripts) & set(ROMANIZED):
            parts.append(f"{one_line(romanize(text))[:300]} ({', '.join(x for x in scripts if x in ROMANIZED)})")
        if set(scripts) - set(ROMANIZED):
            parts.append(f"{', '.join(x for x in scripts if x not in ROMANIZED)} script, not transliterated")
        notes.append("transliterated: " + "; ".join(parts))
    return notes


# --- explainer and second reader -------------------------------------------------------------

def fmt_vocab(items) -> str:
    return ", ".join(f"{json.dumps(t, ensure_ascii=False)} ({s:+.3g})" for t, s in items)


def explainer_values(harvest, corpus_dir, pole: str) -> tuple[dict, bool]:
    """(the explainer prompt's values, whether any context got an X7 decoded or transliterated line)."""
    layer, neuron, s = parse_pole(pole)
    st = pole_stats(harvest, pole)
    man = read_json(Path(harvest) / HARVEST_FILES["manifest"])
    vocab = pole_vocab(read_json(Path(harvest) / HARVEST_FILES["vocab"]), pole)
    ctxs = pole_contexts(harvest, corpus_dir, pole, N_EXPLAIN_CONTEXTS, EXPLAIN_BEFORE)
    tokens, offsets, tokenizer = harvest_corpus(harvest, corpus_dir)
    lines, flagged = [], False
    for c in ctxs:
        lines.append(f"#{c['rank']} a={c['a']:+.4g} ({regime(c['g'], c['u'])}) | {marked(c)}")
        near = render(tokens, offsets, tokenizer, c["tok"], NOTE_WINDOW, NOTE_WINDOW)
        notes = encoding_notes(near["before"] + near["token"] + near["after"])
        flagged = flagged or bool(notes)
        lines += [f"   {note}" for note in notes]
    sc = st["sign_counts"]
    return {"pole": pole, "sign_word": "positive" if s == 0 else "negative", "neuron": neuron,
            "layer": layer, "n_layers": man["n_layers"], "n_tokens_scored": man["n_tokens_scored"],
            "n_pole_tokens": st["n_pole_tokens"], "max_abs": f"{st['max_abs']:.4g}",
            "q99": f"{st['q99']:.4g}", "q999": f"{st['q999']:.4g}", "q9999": f"{st['q9999']:.4g}",
            "density": st["density"], "same_token_frac": f"{st['same_token_frac']:.2f}",
            "regimes": (f"- g>=0, u>=0: {sc[0]}\n- g>=0, u<0: {sc[1]}\n- g<0, u>=0: {sc[2]}\n"
                        f"- g<0, u<0 (double negative): {sc[3]} (share {st['double_negative_share']:.2f})"),
            "promoted": fmt_vocab(vocab["promoted"]), "suppressed": fmt_vocab(vocab["suppressed"]),
            "n_contexts": len(ctxs), "contexts": "\n".join(lines)}, flagged


def cmd_render(a) -> None:
    for ctx in pole_contexts(a.harvest, a.corpus, a.pole, a.n, a.before):
        print(json.dumps({**ctx, "marked": marked(ctx)}, ensure_ascii=False))


def cmd_prepare(a) -> None:
    llm.refuse_opus(a.model, f"{a.batch}: --model", llm.OPUS_ROUTED)   # before anything is written
    explain = Path(a.out) / "explain"
    ensure_manifest(explain)
    poles = read_json(a.poles)
    roles = a.roles.split(",")
    if not set(roles) <= set(SAMPLE_INDEX):
        raise ValueError(f"prepare makes explainer and second_reader requests, got {roles}")
    views = {p: explainer_values(a.harvest, a.corpus, p) for p in poles}
    for role in roles:
        write_batch(explain, f"{a.batch}-{role}",
                    [build_request(role, f"{role}:{p}:{SAMPLE_INDEX[role]}", a.model, views[p][0]) for p in poles])
    llm.write_jsonl_new(explain / "plans" / f"{a.batch}-explainer-view.jsonl",
                        [{"pole": p, "encoding_flag": views[p][1]} for p in poles])


def leak_errors(req: dict, answer: dict) -> list[str]:
    """The explanation and every job sentence reach the blind writer, who must never see the shown
    material or an activation (SPEC §2.2). The check refuses exactly those leaks: the ⟦ ⟧ marker, an
    activation value such as a=12.4 or u = -3, and a reference to a shown context ("context 3",
    "#2", "the contexts shown above"). Digits and the word "context" used to describe the job pass."""
    texts = [("explanation", answer["explanation"])] + [(f"job {k}", j["name"]) for k, j in enumerate(answer["jobs"])]
    return [f"{where} of {req['id'].split(':')[1]} contains {what}: {text!r}"
            for where, text in texts for what, pattern in LEAK_RES if pattern.search(text)]


def same_as_explainer(answers: dict) -> list[tuple[str, str, str]]:
    """(batch, id, reason) for second-reader answers equal (as JSON objects) to the explainer's answer
    for the same pole."""
    first = {rid.split(":")[1]: x for b, xs in answers.items() if b.endswith("-explainer") for rid, x in xs.items()}
    return [(b, rid, "identical to the explainer's answer; the second reader must be an independent sample")
            for b, xs in answers.items() if b.endswith("-second_reader")
            for rid, x in sorted(xs.items()) if first.get(rid.split(":")[1]) == x]


def collected_explainer(explain: Path, batch: str) -> dict[str, dict]:
    """{id: answer} of `<batch>-explainer` as collected earlier: the response lines whose text has the
    response_sha256 calls.jsonl logged for that id. Raises unless every explainer request of the batch
    is logged or declined, so a second reader collected on its own is compared with the whole explainer."""
    b = f"{batch}-explainer"
    calls = explain / "calls.jsonl"
    logged = {r["id"]: r["response_sha256"] for r in llm.read_jsonl(calls)} if calls.exists() else {}
    gave_up = llm.declined(explain, b)
    missing = [r["id"] for r in llm.read_jsonl(llm.requests_path(explain, b)) if r["id"] not in logged and r["id"] not in gave_up]
    if missing:
        raise ValueError(f"the second reader is compared with the collected explainer: {b} is not collected "
                         f"({len(missing)} ids neither in {calls} nor declined, e.g. {missing[:5]}); collect the "
                         f"explainer first or both roles together")
    answers = {}
    for resp in llm.read_jsonl(llm.responses_path(explain, b)):
        if resp["id"] in logged:
            if llm.sha256_text(resp["text"]) != logged[resp["id"]]:
                raise ValueError(f"{llm.responses_path(explain, b)}: {resp['id']} differs from the answer {calls} logged")
            answers[resp["id"]] = json.loads(resp["text"])
    return answers


def role_batches(batch: str, role: str) -> list[str]:
    """The batch files `prepare*` writes for one role: one per sample for the clusterer and judge."""
    if role not in llm.ROLES:
        raise ValueError(f"unknown role {role!r}")
    return [f"{batch}-{role}-s{s}" for s in (0, 1)] if role in ("clusterer", "judge") else [f"{batch}-{role}"]


def writer_texts(rid: str, x: dict) -> list[dict]:
    """The texts.jsonl rows of one writer or writer_job answer."""
    role, pole, tag = rid.split(":")
    if role == "writer":
        rows = [text_row(pole, kind, k, None, t) for kind, key in (("pos", "positives"), ("near", "near_misses"),
                                                                  ("distant", "distant")) for k, t in enumerate(x[key])]
        return rows + [text_row(pole, "max", 0, None, x["max"])]
    j = int(tag[1:])
    return [text_row(pole, "job_pos", 5 * j + i, j, t) for i, t in enumerate(x["positives"])]


def short_texts(req: dict, x: dict, tokenizer) -> list[str]:
    """Writer texts the probe cannot score: fewer than MIN_TEXT_TOKENS tokens under the scored model's
    tokenizer (the probe excludes each text's first token)."""
    out = []
    for row in writer_texts(req["id"], x):
        n = len(tokenizer.encode(row["text"], add_special_tokens=False))
        if n < MIN_TEXT_TOKENS:
            out.append(f"text {row['text_id']} has {n} token(s) under the scored model's tokenizer; "
                       f"the probe needs at least {MIN_TEXT_TOKENS}")
    return out


def validation(explain: Path, batch: str, roles: list[str], tokenizer=None):
    """(batch files, check, cross_check) for `roles`: the one table both collect and reject use, so an
    answer reject moves is exactly an answer collect refuses. Beyond the schema, the explainer and
    second reader are checked for leaks to the writer and for copies, the clusterer for a partition
    of the top-50 with the echo cluster, the writer and writer_job for texts of at least two tokens
    under the scored model's `tokenizer`; the judge has the schema only. A second reader without the
    explainer in `roles` is compared with the explainer answers collected earlier (collected_explainer)."""
    earlier = ({f"{batch}-explainer": collected_explainer(explain, batch)}
               if "second_reader" in roles and "explainer" not in roles else {})
    if set(roles) & set(WRITER_ROLES) and tokenizer is None:
        raise ValueError("writer answers are checked with the scored model's tokenizer: give --harvest and --corpus")
    cluster_plans = (llm.read_jsonl(explain / "plans" / f"{batch}-clusterer.jsonl") if "clusterer" in roles else [])
    plans = {row["id"]: row for row in cluster_plans}

    def check(req, x):
        if req["role"] in ("explainer", "second_reader"):
            return leak_errors(req, x)
        if req["role"] in WRITER_ROLES:
            return short_texts(req, x, tokenizer)
        if req["role"] == "clusterer":
            plan = plans[req["id"]]
            return [f"{plan['pole']} sample {plan['sample']}: {e}"
                    for e in partition_errors(x, plan["n_valid"]) + echo_errors(x, plan["double_negative_ranks"])]
        return []
    return [b for r in roles for b in role_batches(batch, r)], check, lambda out: same_as_explainer({**earlier, **out})


def cmd_reject(a) -> None:
    """Move refused answers out of the responses files so `run` (or the manual agent) answers them again."""
    explain = Path(a.out) / "explain"
    if (a.harvest is None) != (a.corpus is None):
        raise ValueError("--harvest and --corpus go together")
    tokenizer = harvest_corpus(a.harvest, a.corpus)[2] if a.harvest else None
    batches, check, cross_check = validation(explain, a.batch, a.roles.split(","), tokenizer)
    rejected, gave_up = llm.reject_many(explain, batches, a.backend, check, cross_check, a.max_attempts)
    for b in batches:
        ids = rejected.get(b, [])
        print(f"{b}: {len(ids)} rejected: {', '.join(ids)}" if ids else f"{b}: nothing to reject; responses unchanged")
        if gave_up.get(b):
            print(f"{b}: {len(gave_up[b])} declined after {a.max_attempts} attempts (not asked again): {', '.join(gave_up[b])}")


COLLECTED_OUTPUT = {"clusterer": "clusters.jsonl"}   # roles whose collect writes one whole file, derived from responses


def drop_lines(path: Path, ids: set[str]) -> list[dict]:
    """Rewrite a JSONL file without the lines whose id is in `ids` (kept lines keep their bytes); returns the
    dropped rows."""
    raw = [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if line.strip()]
    rows = [json.loads(line) for line in raw]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line for line, r in zip(raw, rows) if r["id"] not in ids), encoding="utf-8")
    os.replace(tmp, path)
    return [r for r in rows if r["id"] in ids]


def cmd_reopen(a) -> None:
    """Move a batch's declined ids of `--roles` back to pending, so `run` asks them again and `reject
    --max-attempts` sets the new cap. The rejected lines stay, so attempts keep counting from where they were.
    Each reopened id is recorded in the manifest under `reopened` with its old reason, attempts and utc.
    A batch already collected is uncollected first, for roles whose collect writes one derived file
    (COLLECTED_OUTPUT): that file is removed and the batch's rows leave calls.jsonl, both recorded, and the
    next collect writes them again from the responses. Refused for other collected roles, and for clusters
    a judge batch was already prepared on."""
    explain = Path(a.out) / "explain"
    roles = a.roles.split(",")
    batches = [b for r in roles for b in role_batches(a.batch, r)]
    reopened = {b: llm.declined(explain, b) for b in batches}
    if not any(reopened.values()):
        print(f"{batches}: nothing declined; nothing changed")
        return
    calls = explain / "calls.jsonl"
    ids = {r["id"] for b in batches for r in llm.read_jsonl(llm.requests_path(explain, b))}
    logged = ids & {r["id"] for r in llm.read_jsonl(calls)} if calls.exists() else set()
    uncollected = None
    if logged:
        other = sorted(set(roles) - set(COLLECTED_OUTPUT))
        if other:
            raise ValueError(f"{a.batch}: {other} already collected; reopen undoes a collect only for {sorted(COLLECTED_OUTPUT)}")
        judge = sorted((explain / "plans").glob("*-judge.jsonl"))
        if judge:
            raise ValueError(f"{judge[0]}: a judge batch was prepared on these clusters; reopening them is refused")
        removed = {}
        for r in roles:
            out = explain / COLLECTED_OUTPUT[r]
            removed[out.name] = file_sha256(out)
            out.unlink()
        uncollected = {"calls_rows_removed": len(drop_lines(calls, logged)), "removed_sha256": removed}
    for b, rows in reopened.items():
        if rows:
            drop_lines(llm.declined_path(explain, b), set(rows))
    m = read_json(explain / "manifest.json")
    m.setdefault("reopened", []).append({
        "utc": llm.utc_now(), "batch": a.batch, "roles": roles, "uncollected": uncollected,
        "ids": {rid: {"batch": b, "role": row["role"], "reason": row["reason"], "finish_reason": row["finish_reason"],
                      "attempts": row["attempts"], "declined_utc": row["utc"]}
                for b, rows in reopened.items() for rid, row in sorted(rows.items())}})
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))
    for b, rows in reopened.items():
        print(f"{b}: {len(rows)} declined ids reopened" + (f": {', '.join(sorted(rows))}" if rows else ""))
    if uncollected:
        print(f"uncollected: removed {sorted(uncollected['removed_sha256'])}, {uncollected['calls_rows_removed']} rows of calls.jsonl")


def print_declined(explain: Path, batches: list[str]) -> None:
    for b in batches:
        for rid, row in sorted(llm.declined(explain, b).items()):
            print(f"declined: {rid}: {llm.decline_text(row)}")


def cmd_collect(a) -> None:
    explain = Path(a.out) / "explain"
    batches, check, cross_check = validation(explain, a.batch, a.roles.split(","))
    answers = llm.collect_many(explain, batches, a.backend, check=check, cross_check=cross_check)
    flag = {r["pole"]: r["encoding_flag"] for r in llm.read_jsonl(explain / "plans" / f"{a.batch}-explainer-view.jsonl")}
    rows = []
    for b, batch in answers.items():
        models = llm.response_models(explain, b)                      # the model each answer came from
        for rid, x in sorted(batch.items()):
            _, pole, sample = rid.split(":")
            rows.append({"pole": pole, "sample": int(sample), "explanation": x["explanation"],
                         "jobs": x["jobs"], "class": x["class"], "confidence": x["confidence"],
                         "encoding_flag": flag[pole], "model": models[rid]})
    for b in batches:                                   # a declined id is an explanation that was not given
        asked = {r["id"]: r["model"] for r in llm.read_jsonl(llm.requests_path(explain, b))}
        for rid, row in sorted(llm.declined(explain, b).items()):
            _, pole, sample = rid.split(":")
            rows.append({"pole": pole, "sample": int(sample), "explanation": "", "jobs": [], "class": "UNK",
                         "confidence": 0.0, "encoding_flag": flag[pole], "model": asked[rid],
                         "declined": llm.decline_text(row)})
    llm.append_jsonl(explain / "explanations.jsonl", rows)
    record_models(explain)
    print_declined(explain, batches)


# --- writer ----------------------------------------------------------------------------------

def cmd_prepare_writer(a) -> None:
    llm.refuse_opus(a.model, f"{a.batch}: --model", llm.OPUS_ROUTED)   # before anything is written
    explain = Path(a.out) / "explain"
    ensure_manifest(explain)
    first = [r for r in llm.read_jsonl(explain / "explanations.jsonl") if r["sample"] == 0 and "declined" not in r]
    write_batch(explain, f"{a.batch}-writer",
                [build_request("writer", f"writer:{r['pole']}:0", a.model, {"explanation": r["explanation"]})
                 for r in first])
    jobs = [build_request("writer_job", f"writer_job:{r['pole']}:j{k}", a.model, {"job": j["name"]})
            for r in first if r["class"] == "UNREL" for k, j in enumerate(r["jobs"])
            if not j["related_to_others"]]
    if jobs:
        write_batch(explain, f"{a.batch}-writer_job", jobs)


def text_row(pole: str, kind: str, k: int, job: int | None, text: str) -> dict:
    return {"text_id": f"{pole}:{kind}:{k}", "pole": pole, "kind": kind, "job": job, "text": text}


def draw_shuffle(harvest, poles: list[str], seed: int) -> dict[str, str]:
    """§5 shuffle: for each pole, a pole drawn uniformly from the same layer, excluding the pole's own
    neuron and dead poles (max|a| = 0, which have no score)."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for pole in sorted(poles):
        layer, neuron, _ = parse_pole(pole)
        max_abs = layer_file(str(harvest), "stats", layer)["max_abs"]
        cands = [pole_id(layer, n, s) for n in range(max_abs.shape[0]) for s in (0, 1)
                 if n != neuron and max_abs[n, s] > 0]
        pairs[pole] = cands[int(rng.integers(len(cands)))]
    return pairs


def cmd_collect_writer(a) -> None:
    explain = Path(a.out) / "explain"
    roles = ["writer"] + (["writer_job"] if llm.requests_path(explain, f"{a.batch}-writer_job").exists() else [])
    batches, check, cross_check = validation(explain, a.batch, roles, harvest_corpus(a.harvest, a.corpus)[2])
    answers = llm.collect_many(explain, batches, a.backend, check=check, cross_check=cross_check)
    # A pole whose writer or writer_job declined gets no texts (the verdict counts it as UNK-N).
    gone = {rid.split(":")[1] for b in batches for rid in llm.declined(explain, b)}
    models = {b: llm.response_models(explain, b) for b in batches}
    texts = [{**row, "model": models[b][rid]} for b in batches for rid, x in sorted(answers[b].items())
             if rid.split(":")[1] not in gone for row in writer_texts(rid, x)]
    poles = sorted({t["pole"] for t in texts})
    # The shuffle is drawn over every pole that reached the writer step, so a decline does not move
    # any other pole's partner.
    writer_poles = sorted({r["id"].split(":")[1] for r in llm.read_jsonl(llm.requests_path(explain, f"{a.batch}-writer"))})
    pairs = draw_shuffle(a.harvest, writer_poles, a.seed)
    llm.write_jsonl_new(explain / "texts.jsonl", texts)
    for name, obj in (("shuffle.json", {"seed": a.seed, "harvest": str(a.harvest), "pairs": pairs}),
                      ("probe_poles.json", sorted(set(poles) | {pairs[p] for p in poles}))):
        if (explain / name).exists():
            raise FileExistsError(f"{explain / name} exists")
        (explain / name).write_text(json.dumps(obj, indent=1))
    note_in_manifest(explain, seeds={"shuffle": a.seed})
    record_models(explain)
    print_declined(explain, batches)




def cmd_remodel(a) -> None:
    """Set the model of the named roles' request lines to --model, so `run` sends them to another model
    (SPEC_D1 §2.2 model policy amendment). With --unanswered-only, lines that already have an answer
    keep their model (the record of who answered); without it, a role with any answered line is
    refused. An Opus id is refused as prepare refuses it. Prints per role the lines changed and the
    answered lines left untouched; the explain manifest records each operation."""
    llm.refuse_opus(a.model, f"{a.batch}: --model", llm.OPUS_ROUTED)
    explain = Path(a.out) / "explain"
    done = []
    for role in a.roles.split(","):
        for batch in role_batches(a.batch, role):
            path = llm.requests_path(explain, batch)
            reqs = llm.read_jsonl(path)
            answered = set(llm.response_models(explain, batch))
            if answered and not a.unanswered_only:
                raise ValueError(f"{batch}: {len(answered)} lines are answered; their model is the record of who "
                                 f"answered them; use --unanswered-only")
            change = [r for r in reqs if r["id"] not in answered]
            old = Counter(r["model"] for r in change)
            for r in change:
                r["model"] = a.model
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs), encoding="utf-8")
            tmp.replace(path)
            done.append({"batch": batch, "role": role, "old_models": dict(old), "new_model": a.model,
                         "changed": len(change), "answered_untouched": len(reqs) - len(change), "utc": llm.utc_now()})
            print(f"{batch}: {len(change)} lines set to {a.model} (from {dict(old)}); "
                  f"{len(reqs) - len(change)} answered lines untouched")
    m = read_json(explain / "manifest.json")
    m["remodels"] = (m["remodels"] if "remodels" in m else []) + done
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))
    record_models(explain)


def cmd_fork_writer(a) -> None:
    """Copy `<source>/explain` to `<out>/explain` without the writer step, so the writer can be redone
    on the same explanations while the source stays as it is. Dropped: the writer and writer_job
    batches' requests, responses, usage, backend and rejected files; their rows in calls.jsonl (else
    collect-writer refuses the ids as already logged); their entries in the manifest's batch list;
    texts.jsonl, shuffle.json, probe_poles.json and the shuffle seed. Everything else is copied byte
    for byte; the manifest records forked_from."""
    src, dst = Path(a.source) / "explain", Path(a.out) / "explain"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    writer_batches = set()
    for path in (src / "requests").glob("*.jsonl"):
        roles = {r["role"] for r in llm.read_jsonl(path)}
        if roles & set(WRITER_ROLES):
            if not roles <= set(WRITER_ROLES):
                raise ValueError(f"{path}: mixes writer and other roles {sorted(roles)}; cannot fork")
            writer_batches.add(path.stem)
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        if rel.parts[0] in ("requests", "responses"):
            if any(rel.name.startswith(f"{b}.") for b in writer_batches):
                continue
        elif rel.as_posix() in WRITER_FILES + ("calls.jsonl", "manifest.json"):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / rel)
    calls = src / "calls.jsonl"
    llm.write_jsonl_new(dst / "calls.jsonl",
                        [r for r in llm.read_jsonl(calls) if r["role"] not in WRITER_ROLES] if calls.exists() else [])
    m = read_json(src / "manifest.json")
    m["batches"] = [b for b in m["batches"] if b["batch"] not in writer_batches]
    m["seeds"] = {k: v for k, v in m["seeds"].items() if k != "shuffle"}
    m["forked_from"] = {"explain": str(src), "utc": llm.utc_now(), "dropped_batches": sorted(writer_batches),
                        "dropped": "the writer step (writer and writer_job batches, their call rows, "
                                   "texts.jsonl, shuffle.json, probe_poles.json, the shuffle seed)"}
    (dst / "manifest.json").write_text(json.dumps(m, indent=1))
    print(f"{dst}: forked from {src} without {sorted(writer_batches)}")


def cmd_fork_clusters(a) -> None:
    """Copy `<source>/explain`'s collected clusters to `<out>/explain` without any judge batch, so another judge
    can be gated on the same clusters (§4.4: the judge is tuned on S_gate at W4). Copied byte for byte:
    positions.npz, positions_poles.json, clusters.jsonl, plans/<b>-clusterer.jsonl and every requests/ and
    responses/ file of the clusterer batches (answers, usage, backend, rejected, declined); calls.jsonl keeps
    its clusterer rows. The manifest is copied without judge batches, judge seeds and judge_gates, gets
    forked_from (source, its manifest SHA-256, utc) and models_by_role recomputed. Judge requests, responses,
    plans, judgments.jsonl and <source>/verdict/ are not copied."""
    src, dst = Path(a.source) / "explain", Path(a.out) / "explain"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    if not (src / "clusters.jsonl").exists():
        raise ValueError(f"{src}: no clusters.jsonl; collect the clusters before forking them")
    batches = set()
    for path in (src / "requests").glob("*.jsonl"):
        roles = {r["role"] for r in llm.read_jsonl(path)}
        if "clusterer" in roles:
            if roles != {"clusterer"}:
                raise ValueError(f"{path}: mixes the clusterer with {sorted(roles - {'clusterer'})}; cannot fork")
            batches.add(path.stem)
    files = [src / name for name in ("positions.npz", "positions_poles.json", "clusters.jsonl")]
    files += sorted((src / "plans").glob("*-clusterer.jsonl"))
    files += [p for d in ("requests", "responses") for p in sorted((src / d).iterdir())
              if any(p.name.startswith(f"{b}.") for b in batches)]
    for path in files:
        (dst / path.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / path.relative_to(src))
    llm.write_jsonl_new(dst / "calls.jsonl", [r for r in llm.read_jsonl(src / "calls.jsonl") if r["role"] == "clusterer"])
    m = read_json(src / "manifest.json")
    dropped = sorted(b["batch"] for b in m["batches"] if "judge" in b["roles"])
    m = {k: v for k, v in m.items() if k != "judge_gates"}
    m["batches"] = [b for b in m["batches"] if "judge" not in b["roles"]]
    m["seeds"] = {k: v for k, v in m["seeds"].items() if not k.startswith("judge:")}
    m["forked_from"] = {"explain": str(src), "manifest_sha256": file_sha256(src / "manifest.json"),
                        "utc": llm.utc_now(), "dropped_batches": dropped,
                        "dropped": "every judge batch (requests, responses, plans, seeds, judge_gates), "
                                   "judgments.jsonl and the source's verdict/"}
    (dst / "manifest.json").write_text(json.dumps(m, indent=1))
    record_models(dst)
    print(f"{dst}: clusters of {src} ({sorted(batches)}), without {dropped}")


# --- clusters --------------------------------------------------------------------------------

def cmd_positions(a) -> None:
    """W3 input: the T_15 top-50 token positions of every pole of S, rank order, -1 padded."""
    explain = Path(a.out) / "explain"
    ensure_manifest(explain)
    poles = read_json(a.poles)
    tok = np.full((len(poles), N_TOPK), -1, dtype=np.int64)
    for i, pole in enumerate(poles):
        layer, neuron, s = parse_pole(pole)
        row = layer_file(str(a.harvest), "topk", layer)["tok"][neuron, s][:N_TOPK]
        tok[i, :len(row)] = row
    for name in ("positions.npz", "positions_poles.json"):
        if (explain / name).exists():
            raise FileExistsError(f"{explain / name} exists")
    np.savez(explain / "positions.npz", tok=tok)
    (explain / "positions_poles.json").write_text(json.dumps(poles))
    m = read_json(explain / "manifest.json")
    m["positions"] = {"poles_file": str(a.poles), "poles_sha256": file_sha256(a.poles), "n_poles": len(poles),
                      "harvest": str(a.harvest), "utc": llm.utc_now()}
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))


def positions(explain: Path) -> dict[str, np.ndarray]:
    tok = np.load(explain / "positions.npz")["tok"]
    return dict(zip(read_json(explain / "positions_poles.json"), tok))


def check_positions(explain: Path, harvest) -> dict[str, np.ndarray]:
    """The exemplar columns are ranks only if `harvest` is the T_15 harvest positions.npz came from.
    When the harvest wrote its own `poles/positions.npz` (DATA_CONTRACT §7), the explain directory's
    positions must equal it and it must equal `topk/` for every pole; otherwise they must equal `topk/`."""
    pos = positions(explain)
    own_path = Path(harvest) / "poles" / "positions.npz"
    own = dict(zip(read_json(Path(harvest) / HARVEST_FILES["poles"]), np.load(own_path)["tok"])) if own_path.exists() else None
    for pole, row in pos.items():
        layer, neuron, s = parse_pole(pole)
        top = layer_file(str(harvest), "topk", layer)["tok"][neuron, s][:N_TOPK]
        if own is not None:
            if pole not in own or not np.array_equal(own[pole], row):
                raise ValueError(f"{own_path}: positions of {pole} differ from {explain / 'positions.npz'}")
            if not np.array_equal(own[pole], top):
                raise ValueError(f"{own_path}: positions of {pole} differ from the same harvest's topk/")
        elif not np.array_equal(top, row):
            raise ValueError(f"{harvest}: top-50 of {pole} differs from {explain / 'positions.npz'}")
    return pos


def opaque_ids(role: str, batch: str, n: int, rng) -> list[str]:
    """`<role>:<batch>:<int>` with the integers a random permutation of 0..n-1: the id says nothing
    about the pole, cluster, decoy or sample it stands for; plans/ holds the mapping."""
    return [f"{role}:{batch}:{int(k)}" for k in rng.permutation(n)]


def cmd_prepare_clusters(a) -> None:
    llm.refuse_opus(a.model, f"{a.batch}: --model", llm.OPUS_ROUTED)   # before anything is written
    explain = Path(a.out) / "explain"
    ensure_manifest(explain)
    pos = check_positions(explain, a.harvest)
    man = read_json(Path(a.harvest) / HARVEST_FILES["manifest"])
    rng = np.random.default_rng(a.seed)
    note_in_manifest(explain, seeds={f"clusterer:{a.batch}": a.seed})
    specs = []
    for pole in pos:
        layer, neuron, s = parse_pole(pole)
        ctxs = pole_contexts(a.harvest, a.corpus, pole, N_TOPK, EXPLAIN_BEFORE)
        max_abs = pole_stats(a.harvest, pole)["max_abs"]
        lines = [f"#{c['rank']} ({abs(c['a']) / max_abs:.2f} of max, {regime(c['g'], c['u'])}) | {marked(c)}"
                 for c in ctxs]
        values = {"pole": pole, "sign_word": "positive" if s == 0 else "negative", "neuron": neuron,
                  "layer": layer, "n_layers": man["n_layers"], "n_contexts": len(ctxs), "last_rank": len(ctxs) - 1,
                  "contexts": "\n".join(lines)}
        double_negative = [c["rank"] for c in ctxs if c["g"] < 0 and c["u"] < 0]
        specs += [(pole, sample, values, len(ctxs), double_negative) for sample in (0, 1)]
    ids = opaque_ids("clusterer", a.batch, len(specs), rng)
    plans, reqs = [], {0: [], 1: []}
    for rid, (pole, sample, values, n_valid, double_negative) in zip(ids, specs):
        reqs[sample].append(build_request("clusterer", rid, a.model, values))
        plans.append({"id": rid, "pole": pole, "sample": sample, "n_valid": n_valid,
                      "double_negative_ranks": double_negative})
    for sample in (0, 1):
        write_batch(explain, f"{a.batch}-clusterer-s{sample}", [reqs[sample][i] for i in rng.permutation(len(reqs[sample]))])
    llm.write_jsonl_new(explain / "plans" / f"{a.batch}-clusterer.jsonl", plans)


def partition_errors(answer: dict, n_valid: int) -> list[str]:
    """A clustering answer must partition ranks 0..n-1, number clusters 0..m-1, and relate every pair once."""
    errors = []
    cs = [c["c"] for c in answer["clusters"]]
    if sorted(cs) != list(range(len(cs))):
        errors.append(f"cluster numbers {sorted(cs)} are not 0..{len(cs) - 1}")
    ranks = sorted(r for c in answer["clusters"] for r in c["exemplar_ranks"])
    if ranks != list(range(n_valid)):
        errors.append(f"exemplar_ranks do not cover 0..{n_valid - 1} exactly once")
    want = {(i, j) for i in range(len(cs)) for j in range(i + 1, len(cs))}
    got = [tuple(sorted((p["a"], p["b"]))) for p in answer["pairs"]]
    if sorted(got) != sorted(want):
        errors.append(f"pairs {sorted(got)} != every pair of clusters once {sorted(want)}")
    return errors


def echo_errors(answer: dict, double_negative: list[int]) -> list[str]:
    """§2.4 as amended: the double-negative contexts (g < 0, u < 0) form one cluster flagged echo,
    and the echo cluster holds exactly those contexts."""
    echoes = [c for c in answer["clusters"] if c["echo"]]
    if len(echoes) > 1:
        return [f"{len(echoes)} clusters flagged echo; at most one"]
    held = sorted(echoes[0]["exemplar_ranks"]) if echoes else []
    if held != sorted(double_negative):
        return [f"the echo cluster holds ranks {held}; the double-negative ranks are {sorted(double_negative)}"]
    return []


def cmd_collect_clusters(a) -> None:
    explain = Path(a.out) / "explain"
    plans = {p["id"]: p for p in llm.read_jsonl(explain / "plans" / f"{a.batch}-clusterer.jsonl")}
    batches, check, cross_check = validation(explain, a.batch, ["clusterer"])
    answers = llm.collect_many(explain, batches, a.backend, check=check, cross_check=cross_check)
    rows = []
    for batch in answers.values():
        for rid, x in answers_in_plan_order(batch, plans):
            rows.append({"pole": plans[rid]["pole"], "sample": plans[rid]["sample"],
                         "clusters": [{"c": c["c"], "name": c["name"], "echo": c["echo"],
                                       "exemplar_ranks": c["exemplar_ranks"]}
                                      for c in sorted(x["clusters"], key=lambda c: c["c"])],
                         "pairs": [[p["a"], p["b"], p["relation"]] for p in x["pairs"]]})
    for b in batches:                       # declined after the attempts cap: the pole leaves Q2 (verdict)
        for rid, row in sorted(llm.declined(explain, b).items()):
            rows.append({"pole": plans[rid]["pole"], "sample": plans[rid]["sample"], "declined": llm.decline_text(row)})
    llm.write_jsonl_new(explain / "clusters.jsonl", rows)
    record_models(explain)
    print_declined(explain, batches)


def answers_in_plan_order(answers: dict, plans: dict) -> list:
    order = {rid: i for i, rid in enumerate(plans)}
    return sorted(answers.items(), key=lambda kv: order[kv[0]])


# --- judge -----------------------------------------------------------------------------------

def layer_token_strings(harvest, corpus_dir, layer: int) -> dict[str, set]:
    """Per pole of the layer: the set of token strings of its top-50 (for the hard decoy)."""
    tokens, _, tokenizer = harvest_corpus(harvest, corpus_dir)
    tk = layer_file(str(harvest), "topk", layer)["tok"]
    cache: dict[int, str] = {}
    out = {}
    for n in range(tk.shape[0]):
        for s in (0, 1):
            ids = [int(tokens[t]) for t in tk[n, s, :N_TOPK] if t >= 0]
            for i in ids:
                if i not in cache:
                    cache[i] = tokenizer.decode([i])
            out[pole_id(layer, n, s)] = {cache[i] for i in ids}
    return out


def exemplar_windows(offsets, exemplars) -> np.ndarray:
    """[k, 2] inclusive token ranges: for each exemplar e, e - 256 .. e + 16 clipped to e's document
    (the window the explainer renders around it). SPEC §2.1 and §2.4 as amended."""
    e = np.asarray(sorted(exemplars), dtype=np.int64)
    doc = np.searchsorted(offsets, e, side="right") - 1
    return np.stack([np.maximum(e - EXPLAIN_BEFORE, offsets[doc]), np.minimum(e + AFTER, offsets[doc + 1] - 1)], axis=1)


def in_windows(tok, windows: np.ndarray) -> np.ndarray:
    """True where a global token index lies inside any window (the -1 padding never does)."""
    t = np.asarray(tok)[..., None]
    return ((t >= windows[:, 0]) & (t <= windows[:, 1])).any(-1)


def record(harvest, corpus_dir, pole: str, windows: np.ndarray, length: int) -> list[str]:
    """A judge record: the pole's top contexts at a checkpoint without any context whose activating
    token lies in an exemplar's window, as marked text with no numbers."""
    return [marked(c) for c in pole_contexts(harvest, corpus_dir, pole, N_TOPK, JUDGE_BEFORE)
            if not in_windows(c["tok"], windows)][:length]


def bullet(lines: list[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


def judge_gate_record(gate: Path, out: Path) -> dict:
    """SPEC §4.4 (W0 amendment) and the W4 order: the gate directory (S_gate, `--gate` equal to `--out`)
    prepares its own judge batch; any other directory (S) only after `<gate>/verdict/judge_gate.json` exists."""
    if gate.resolve() == out.resolve():
        return {"gate_dir": str(gate), "self": True, "utc": llm.utc_now()}
    path = gate / "verdict" / "judge_gate.json"
    if not path.is_file():
        raise ValueError(f"{path} does not exist: the judge validity gate is written on the gate directory before "
                         f"any other directory's judge batch is prepared (SPEC §4.4); run verdict judge-gate first")
    return {"gate_dir": str(gate), "self": False, "judge_gate_sha256": file_sha256(path),
            "judge_gate_created_utc": read_json(path)["created_utc"], "utc": llm.utc_now()}


def cmd_prepare_judge(a) -> None:
    """§2.4 parent test: for every late-born primary cluster, its record at t_before against a random
    same-layer decoy (two draws) and the hard decoy (most shared top-50 token strings at T_15); for
    every pole's main cluster, the positive control at T_14. Every context inside an exemplar's
    rendered window (same document, 64 tokens before to 16 after) is removed from both records;
    both records have the same length. Two judge samples, shown order drawn per call.
    Request ids are a random permutation and each batch file is in random order; plans/ holds the
    mapping and, per request, how many requests of the same cluster its file holds. `--gate` names the
    gate directory (judge_gate_record), recorded in the manifest under judge_gates[batch]."""
    llm.refuse_opus(a.model, f"{a.batch}: --model", llm.OPUS_ROUTED)   # before anything is written
    gate = judge_gate_record(Path(a.gate), Path(a.out))
    explain = Path(a.out) / "explain"
    ensure_manifest(explain)
    grid = a.grid.split(",")
    end = grid[-1]
    pos = check_positions(explain, end)
    tables = grid_cluster_tables(grid, explain)
    rng = np.random.default_rng(a.seed)
    note_in_manifest(explain, seeds={f"judge:{a.batch}": a.seed})
    strings: dict[int, dict] = {}
    specs = []
    for pole in sorted(p for p, t in tables.items() if any(r["is_main"] for r in t)):
        layer, neuron, _ = parse_pole(pole)
        if layer not in strings:                              # poles in id order: one layer at a time
            strings = {layer: layer_token_strings(end, a.corpus, layer)}
        mine = strings[layer][pole]
        by_overlap = sorted((p for p in strings[layer] if parse_pole(p)[1] != neuron),
                            key=lambda p: (-len(mine & strings[layer][p]), p))
        top = pole_contexts(end, a.corpus, pole, N_TOPK, JUDGE_BEFORE)
        items = [(r, r["t_before"], False) for r in tables[pole] if r["status"] == "late_primary"]
        items += [(r, T_CONTROL_BEFORE, True) for r in tables[pole] if r["is_main"]]
        offsets = harvest_corpus(end, a.corpus)[1]
        for r, tb, control in items:
            windows = exemplar_windows(offsets, [int(pos[pole][k]) for k in r["ranks"]])
            exemplars = [marked(top[k]) for k in sorted(r["ranks"])[:N_JUDGE_EXEMPLARS]]
            true_rec = record(grid[tb], a.corpus, pole, windows, N_JUDGE_RECORD)
            if not true_rec:
                raise ValueError(f"{r['cluster']}: no context left in its record at {GRID_LABELS[tb]}")
            tk = layer_file(str(grid[tb]), "topk", layer)["tok"][:, :, :N_TOPK]
            avail = ((tk >= 0) & ~in_windows(tk, windows)).sum(-1)
            eligible = [p for p in by_overlap if avail[parse_pole(p)[1], parse_pole(p)[2]] >= len(true_rec)]
            draws = sorted(eligible)
            picks = rng.choice(len(draws), size=2, replace=False)
            for dtype, dpole, draw in (("random", draws[picks[0]], 0), ("random", draws[picks[1]], 1),
                                       ("hard", eligible[0], 0)):
                recs = {"true": true_rec, "decoy": record(grid[tb], a.corpus, dpole, windows, len(true_rec))}
                for sample in (0, 1):
                    order = ["true", "decoy"] if rng.random() < 0.5 else ["decoy", "true"]
                    values = {"exemplars": bullet(exemplars), "record_a": bullet(recs[order[0]]),
                              "record_b": bullet(recs[order[1]])}
                    specs.append((values, {"cluster": r["cluster"], "control": control, "decoy_type": dtype,
                                           "decoy_pole": dpole, "draw": draw, "t_before": GRID_LABELS[tb],
                                           "sample": sample, "shown_order": order}))
    ids = opaque_ids("judge", a.batch, len(specs), rng)
    per_file = Counter((plan["cluster"], plan["sample"]) for _, plan in specs)
    plans, reqs = [], {0: [], 1: []}
    for rid, (values, plan) in zip(ids, specs):
        reqs[plan["sample"]].append(build_request("judge", rid, a.model, values))
        plans.append({"id": rid, **plan, "cluster_requests_in_file": per_file[(plan["cluster"], plan["sample"])]})
    for sample in (0, 1):
        write_batch(explain, f"{a.batch}-judge-s{sample}", [reqs[sample][i] for i in rng.permutation(len(reqs[sample]))])
    llm.write_jsonl_new(explain / "plans" / f"{a.batch}-judge.jsonl", plans)
    m = read_json(explain / "manifest.json")
    m.setdefault("judge_gates", {})[a.batch] = gate
    (explain / "manifest.json").write_text(json.dumps(m, indent=1))


def cmd_collect_judge(a) -> None:
    explain = Path(a.out) / "explain"
    plans = {p["id"]: p for p in llm.read_jsonl(explain / "plans" / f"{a.batch}-judge.jsonl")}
    batches, check, cross_check = validation(explain, a.batch, ["judge"])
    answers = llm.collect_many(explain, batches, a.backend, check=check, cross_check=cross_check)
    def row(rid):
        p = plans[rid]
        return {"id": rid, **{k: p[k] for k in ("cluster", "control", "decoy_type", "decoy_pole", "draw", "t_before",
                                               "sample", "shown_order")}}
    rows = []
    for batch in answers.values():
        for rid, x in answers_in_plan_order(batch, plans):
            truth = "A" if plans[rid]["shown_order"][0] == "true" else "B"
            rows.append({**row(rid), "answer": x["answer"], "correct": x["answer"] == truth})
    for b in batches:          # declined after the attempts cap: its cluster or control leaves p_u / a_ctrl
        for rid, d in sorted(llm.declined(explain, b).items()):
            rows.append({**row(rid), "answer": None, "correct": None, "declined": llm.decline_text(d)})
    llm.append_jsonl(explain / "judgments.jsonl", rows)
    record_models(explain)
    print_declined(explain, batches)


# --- CLI -------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="neurons_explain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, fn, help, *args):
        p = sub.add_parser(name, help=help)
        p.set_defaults(fn=fn)
        for arg in args:
            kw = {"required": True}
            if arg in ("--seed", "--n", "--before"):
                kw["type"] = int
            if arg == "--backend":
                kw["choices"] = llm.COLLECT_BACKENDS
            p.add_argument(arg, **kw)
        return p

    cmd("render", cmd_render, "print a pole's contexts", "--harvest", "--corpus", "--pole", "--n", "--before")
    cmd("prepare", cmd_prepare, "explainer/second-reader requests",
        "--harvest", "--corpus", "--out", "--batch", "--roles", "--poles", "--model")
    cmd("collect", cmd_collect, "-> explanations.jsonl", "--out", "--batch", "--roles", "--backend")
    cmd("prepare-writer", cmd_prepare_writer, "writer and writer_job requests", "--out", "--batch", "--model")
    cmd("collect-writer", cmd_collect_writer, "-> texts.jsonl, shuffle.json",
        "--out", "--batch", "--backend", "--harvest", "--corpus", "--seed")
    cmd("positions", cmd_positions, "T_15 top-50 positions of S", "--harvest", "--out", "--poles")
    cmd("prepare-clusters", cmd_prepare_clusters, "two clustering samples",
        "--harvest", "--corpus", "--out", "--batch", "--model", "--seed")
    cmd("collect-clusters", cmd_collect_clusters, "-> clusters.jsonl", "--out", "--batch", "--backend")
    cmd("prepare-judge", cmd_prepare_judge, "parent tests and positive controls",
        "--grid", "--corpus", "--out", "--batch", "--model", "--seed", "--gate")
    cmd("collect-judge", cmd_collect_judge, "-> judgments.jsonl", "--out", "--batch", "--backend")
    cmd("fork-writer", cmd_fork_writer, "copy an explain directory without its writer step", "--source", "--out")
    cmd("fork-clusters", cmd_fork_clusters, "copy collected clusters without any judge batch", "--source", "--out")
    rm = cmd("remodel", cmd_remodel, "set the model of a role's request lines", "--out", "--batch", "--roles", "--model")
    rm.add_argument("--unanswered-only", action="store_true", help="change only lines without an answer")
    cmd("reopen", cmd_reopen, "move a batch's declined ids back to pending", "--out", "--batch", "--roles")
    r = cmd("reject", cmd_reject, "move refused answers aside so they are asked again",
            "--out", "--batch", "--roles", "--backend")
    r.add_argument("--max-attempts", type=int, default=3, help="attempts per id before it is declined (default 3)")
    r.add_argument("--harvest", help="with --corpus: the scored model's tokenizer, needed for writer roles")
    r.add_argument("--corpus", help="with --harvest")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
