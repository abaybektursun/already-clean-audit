"""neurons_explain on synthetic banked files (DATA_CONTRACT §1-§3): context rendering, the
prepare -> manual answers -> collect round trip for every role, the gates each collect enforces,
and the judge's records. No downloads, no model.

    uv run --no-project --with pytest --with transformers --with tokenizers --with anthropic \
        --with numpy pytest infra/pg/tests/test_neurons_explain.py -q
"""
from __future__ import annotations

import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from infra.pg import neurons_explain as NE
from infra.pg import neurons_llm as llm
from infra.pg import neurons_verdict as NV
from infra.pg.tests import neurons_fixture as F

POLES = ["L00N0+", "L00N1-", "L01N2+", "L01N3-"]


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank")
    corpus = F.build_corpus(root)
    harvest = F.build_harvest(root / "harvest", corpus, poles=POLES)
    return corpus, harvest


def _explanation(req):
    pole = req["id"].split(":")[1]
    if pole == "L00N0+":
        return {"explanation": "Fires on boats at sea.", "jobs": [
            {"name": "Boats at sea.", "related_to_others": False, "reasoning": "one job"}],
            "class": "ONE", "confidence": 0.8}
    if pole == "L00N1-":
        return {"explanation": "Fires on rain and on bread.", "jobs": [
            {"name": "Rain falling on a town.", "related_to_others": False, "reasoning": "weather"},
            {"name": "Bread in a kitchen.", "related_to_others": False, "reasoning": "food"}],
            "class": "UNREL", "confidence": 0.4}
    return {"explanation": "Fires on birds and their songs.", "jobs": [
        {"name": "Birds.", "related_to_others": True, "reasoning": "animal"},
        {"name": "Bird songs.", "related_to_others": True, "reasoning": "sound of the animal"}],
        "class": "TWO-REL", "confidence": 0.6}


def _writer(req):
    return {"positives": ["the boat sailed"] * 5, "near_misses": ["the boat in a park"] * 5,
            "distant_concept": "bread", "distant": ["the bread on the hill"] * 5, "max": "the ship sails the sea"}


def _prepare_explain(bank, out, batch="d1"):
    corpus, harvest = bank
    (out / "poles.json").write_text(json.dumps(POLES))
    NE.main(["prepare", "--harvest", str(harvest), "--corpus", str(corpus), "--out", str(out), "--batch", batch,
             "--roles", "explainer,second_reader", "--poles", str(out / "poles.json"), "--model", "test-model"])
    return out / "explain"


def test_render_is_document_bounded_and_marks_the_right_token(bank):
    corpus, _ = bank
    tokens, offsets, tok = NE.corpus(str(corpus), F.TOKNAME)
    d = int(np.argmax(np.diff(offsets)))                              # the long document (700 tokens)
    start, end = int(offsets[d]), int(offsets[d + 1])
    near_start = NE.render(tokens, offsets, tok, start + 2, 256)
    assert near_start["doc"] == d and near_start["pos"] == 2
    assert near_start["before"].split() == [tok.decode([int(t)]) for t in tokens[start:start + 2]]
    assert near_start["token"] == tok.decode([int(tokens[start + 2])])
    near_end = NE.render(tokens, offsets, tok, end - 3, 256)
    assert near_end["after"].split() == [tok.decode([int(t)]) for t in tokens[end - 2:end]]
    assert near_end["after"].split()[-1] == F.EOS                     # the document's own EOS, then stop
    for before in (64, 256):
        middle = NE.render(tokens, offsets, tok, start + 300, before)
        assert len(middle["before"].split()) == before and len(middle["after"].split()) == 16
    prev = NE.render(tokens, offsets, tok, start, 256)                 # first token: nothing before it
    assert prev["before"] == "" and prev["pos"] == 0


def test_each_role_renders_at_its_width_from_tokens(bank, tmp_path):
    """SPEC §2.1 as amended: the explainer (and clusterer) see 256 tokens before, the judge 64; the
    harvest's samples/contexts.jsonl (64 before) is not read."""
    corpus, harvest = bank
    tokens, offsets, tok = NE.corpus(str(corpus), F.TOKNAME)
    d = int(np.argmax(np.diff(offsets)))
    deep = int(offsets[d]) + 400
    copy = tmp_path / "h"
    shutil.copytree(harvest, copy)
    tk = NV.load_npz(copy / "topk" / "layer_00.npz")
    tk["tok"][0, 0, 0] = deep
    np.savez(copy / "topk" / "layer_00.npz", **tk)
    (copy / "samples").mkdir()
    (copy / "samples" / "contexts.jsonl").write_text(json.dumps({"pole": "L00N0+", "topk": [{"before": "SAMPLED"}]}) + "\n")
    ctxs = NE.pole_contexts(copy, corpus, "L00N0+", 30, NE.EXPLAIN_BEFORE)
    assert [c["rank"] for c in ctxs] == list(range(30)) and ctxs[0]["tok"] == deep
    assert len(ctxs[0]["before"].split()) == 256 and "SAMPLED" not in ctxs[0]["before"]
    values, flagged = NE.explainer_values(copy, corpus, "L00N0+")
    first = next(x for x in values["contexts"].splitlines() if x.startswith("#0 "))
    assert flagged is False
    assert len(first.split(" | ", 1)[1].split("⟦")[0].split()) == 256
    judge_line = NE.record(copy, corpus, "L00N0+", np.zeros((0, 2), dtype=np.int64), 1)[0]
    assert len(judge_line.split("⟦")[0].split()) == 64


def test_x7_flags_base64_and_other_scripts_in_the_explainer_view(tmp_path):
    """X7 (§0c): a context whose marked token or its sixteen neighbours hold Base64 or a non-Latin
    script gets a "decoded:" or "transliterated:" line; plain contexts get none."""
    import base64
    b64 = base64.b64encode(b"product name").decode()
    corpus = F.build_corpus(tmp_path, n_docs=60, extra_words=(b64, "привет", "প্রিয়", "東京"))
    base = corpus / "tok" / F.TOKNAME
    tokens = np.load(base / "tokens.npy")
    offsets = np.load(base / "doc_offsets.npy")
    vocab = json.loads((base / "tokenizer" / "tokenizer.json").read_text())["model"]["vocab"]
    s1, s2 = int(offsets[10]) + 5, int(offsets[20]) + 5
    tokens[s1 - 2], tokens[s2 + 3], tokens[s2 + 4], tokens[s2 + 5] = vocab[b64], vocab["привет"], vocab["প্রিয়"], vocab["東京"]
    np.save(base / "tokens.npy", tokens)
    harvest = F.build_harvest(tmp_path / "h", corpus, poles=["L00N0+"])
    tk = NV.load_npz(harvest / "topk" / "layer_00.npz")
    tk["tok"][0, 0, :3] = [s1, s2, int(offsets[30]) + 5]
    np.savez(harvest / "topk" / "layer_00.npz", **tk)
    values, flagged = NE.explainer_values(harvest, corpus, "L00N0+")
    assert flagged is True
    lines = values["contexts"].splitlines()
    by_rank, rank = {}, None
    for line in lines:
        if line.startswith("#"):
            rank = int(line[1:].split()[0])
            by_rank[rank] = []
        else:
            by_rank[rank].append(line.strip())
    assert by_rank[0] == ["decoded: product name"]
    assert by_rank[1] == ["transliterated: " + by_rank[1][0].split("transliterated: ")[1]]
    assert "privet" in by_rank[1][0] and "priya" in by_rank[1][0] and "(BENGALI, CYRILLIC)" in by_rank[1][0]
    assert "CJK script, not transliterated" in by_rank[1][0]
    assert by_rank[2] == []
    assert NE.encoding_notes("internationalization getElementById plain words") == []


def test_explainer_round_trip(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)
    reqs = {r: llm.read_jsonl(llm.requests_path(explain, f"d1-{r}")) for r in ("explainer", "second_reader")}
    first = reqs["explainer"][0]
    assert first["id"] == "explainer:L00N0+:0" and reqs["second_reader"][0]["id"] == "second_reader:L00N0+:1"
    assert first["user"] == reqs["second_reader"][0]["user"] and first["temperature"] == 1.0
    assert len(re.findall(r"^#\d+ a=", first["user"], re.M)) == 30
    assert "double negative" in first["user"] and "Output vocabulary, most suppressed" in first["user"]
    assert not NE.PLACEHOLDER_RE.search(first["user"])
    manifest = NV.read_json(explain / "manifest.json")
    assert manifest["prompts"]["explainer"] == llm.sha256_text((NE.PROMPT_DIR / "explainer.md").read_text())
    assert manifest["prompt_version"] == "v3" and manifest["context_before"]["explainer"] == 256
    assert "groups of nouns at all" in first["system"] and "summary motif" in first["system"]
    F.answer_batch(explain, "d1-explainer", _explanation)
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 0.3})
    NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader",
             "--backend", "manual"])
    rows = llm.read_jsonl(explain / "explanations.jsonl")
    assert sorted((r["pole"], r["sample"]) for r in rows) == sorted((p, s) for p in POLES for s in (0, 1))
    assert all(r["encoding_flag"] is False for r in rows)          # X7: the toy corpus has no encodings
    assert len(llm.read_jsonl(explain / "calls.jsonl")) == 8


def test_malformed_explainer_answer_fails_collect_naming_the_id(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)

    def bad(req):
        x = _explanation(req)
        if req["id"] == "explainer:L01N2+:0":
            x["confidence"] = 1.5
        return x
    F.answer_batch(explain, "d1-explainer", bad)
    with pytest.raises(ValueError, match=r"explainer:L01N2\+:0: \$\.confidence: 1\.5 > maximum 1"):
        NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer", "--backend", "manual"])
    assert not (explain / "explanations.jsonl").exists() and not (explain / "calls.jsonl").exists()


@pytest.mark.parametrize("leak,needle", [("fires on ⟦the⟧ token", "a context marker"),
                                         ("strongest in context 3", "a reference to a shown context"),
                                         ("see contexts #2 and #4", "a reference to a shown context"),
                                         ("the contexts shown above", "a reference to a shown context"),
                                         ("peaks at a=12.4", "an activation value"),
                                         ("u = -3", "an activation value")])
def test_explanation_that_would_leak_to_the_writer_fails_collect(bank, tmp_path, leak, needle):
    explain = _prepare_explain(bank, tmp_path)

    def leaky(req):
        x = _explanation(req)
        if req["id"] == "explainer:L01N3-:0":
            x["explanation"] = leak
        return x
    F.answer_batch(explain, "d1-explainer", leaky)
    with pytest.raises(ValueError, match=re.escape(f"explanation of L01N3- contains {needle}")):
        NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer", "--backend", "manual"])


def test_job_sentence_pointing_at_a_shown_context_fails_collect(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)

    def leaky(req):
        x = _explanation(req)
        if req["id"] == "explainer:L00N1-:0":
            x["jobs"][1]["name"] = "Bread, as in context 2."
        return x
    F.answer_batch(explain, "d1-explainer", leaky)
    with pytest.raises(ValueError, match="job 1 of L00N1- contains a reference to a shown context"):
        NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer", "--backend", "manual"])


# The explanation and job sentences refused by the first leak check on the W2-D wave
# (bank/w1/q4b_disc, batch d1, 2026-10-20): descriptions of the job, not leaks.
W2D_D1_REJECTED = [
    "This neuron fires on technical and scientific vocabulary describing mixtures, alloys, and chemical compositions, with a strong preference for the morpheme 'eutectic' or similar compositional-suffix words ('miscible', 'clathrate hydrates', 'mixture', '50/50', '70:30' ratios) in materials-science, metallurgy, and chemistry prose.",
    "Fires on words describing the mixing or compositional nature of substances in materials-science and chemistry text, such as 'eutectic', 'miscible', 'mixture', 'clathrate hydrate', and numeric mixture ratios like '70:30' or '50/50'.",
    "This neuron strongly suppresses when the current token is the start of a common technical/mathematical compound term whose completion is highly predictable from the surrounding context — e.g. the first word of a fixed phrase like \"evaluating expression\", \"value of\", \"solve for x\", \"equation\" — signaling that a formulaic keyword phrase is expected next.",
    "This neuron fires on the midpoint of a citation or reference marker—initials like J.R.R., author surnames, or reference tags such as SSL, CSS3, or SQL—where the model is about to complete a known bibliographic or technical abbreviation rather than the surrounding prose.",
    "Firing at a digit position where a number is being parsed or continued within an arithmetic, percentage, or measurement context.",
    "This neuron fires on the punctuation marks inside legal statute citations and software version numbers — the hyphen in section numbers like 6-3-1005 and the decimal point in versions like 4.5 — i.e., on separator characters inside numeric/legal reference tokens.",
    "This neuron fires on the line breaks and punctuation that separate labeled key-value fields in B2B supplier and manufacturer listing pages (company profiles, product spec sheets and FAQ sections), where the next thing expected is another field label such as country, industry, certification or main markets.",
    "This neuron fires on the word 'styl' or 'styling' and its compounds, especially in the context of web styling (CSS, stylesheets, Stylus) and appearance styling such as hairstyling.",
    "The neuron fires strongly inside random alphanumeric strings (gift-card codes, hashes, tokens, base64 blobs, code/URL fragments), tracking the lowercase/uppercase letter-and-digit character inside a long random string rather than any content word.",
    "This neuron fires on an adjective describing something appetizing or appealing — words like juicy, busty, or picturesque that praise the sensory quality of food, bodies, or scenery right before a noun — and it is most active in code contexts where a collection is being assigned, which is an artifact of the negative pole's coverage rather than the shared job.",
    "This neuron fires on the word fragment that begins a short label or link marker attached to a document description, such as the tail of \"Transcript\", \"JSTOR\", \"Topics\", \"Questions\", \"FB2\", or \"Additional info for\" when a catalog or listing entry transitions into a new linked item.",
    "This neuron detects a rate or quantity expressed per unit of time — a recurring frequency or capacity figure (per day, per month, per year, monthly, annually, daily) where the context states how much of something occurs or is produced over a repeating time period.",
    "Firing on numeric digits and decimal values within arithmetic, unit-conversion, or tabular math contexts where a number is being computed or read off.",
    "This neuron fires when the text is inside the structured 'company/manufacturer product listing' boilerplate of B2B trade pages — the labeled key-value fields (Place of Origin, Brand Name, Certification, Business Type, Main Markets, Year Established) and the factory/about-us/FAQ blocks that follow, especially at the field labels and separators where a new spec field is about to be introduced.",
    "This neuron fires strongly on random alphanumeric identifier strings — redeem codes, database tokens, Base64 blobs, hashes and URLs — and marks the position inside such a high-entropy string where the model expects more arbitrary letters and digits to continue rather than a space or word boundary.",
]


@pytest.mark.parametrize("sentence", W2D_D1_REJECTED)
def test_job_descriptions_with_digits_or_the_word_context_pass_the_leak_check(sentence):
    answer = {"explanation": sentence, "jobs": [{"name": sentence, "related_to_others": False, "reasoning": ""}],
              "class": "ONE", "confidence": 0.5}
    assert NE.leak_errors({"id": "explainer:L00N0+:0"}, answer) == []


def test_second_reader_identical_to_the_explainer_fails(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)
    for role in ("explainer", "second_reader"):
        F.answer_batch(explain, f"d1-{role}", _explanation)            # same answer for every pole
    with pytest.raises(ValueError, match=r"second_reader:L00N0\+:1: identical to the explainer's answer"):
        NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader",
                 "--backend", "manual"])
    assert not (explain / "calls.jsonl").exists()
    one = ["--out", str(tmp_path), "--batch", "d1", "--backend", "manual", "--roles"]
    NE.main(["collect", *one, "explainer"])                          # the explainer alone is fine
    with pytest.raises(ValueError, match=r"second_reader:L00N0\+:1: identical to the explainer's answer"):
        NE.main(["collect", *one, "second_reader"])                  # compared with the collected explainer
    assert len(llm.read_jsonl(explain / "calls.jsonl")) == len(POLES)
    NE.main(["reject", *one, "second_reader"])                       # reject uses the same comparison
    assert len(llm.read_jsonl(llm.rejected_path(explain, "d1-second_reader"))) == len(POLES)
    assert not llm.read_jsonl(llm.responses_path(explain, "d1-second_reader"))


def test_second_reader_collected_after_the_explainer_equals_the_joint_collect(bank, tmp_path):
    """W2-C order: the explainer collected on its own, the second reader later on its own. The outputs
    equal one joint collect's, byte for byte; before the explainer is collected the second reader is refused."""
    joint = tmp_path / "joint"
    joint.mkdir()
    explain = _prepare_explain(bank, joint)
    F.answer_batch(explain, "d1-explainer", _explanation)
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 0.3})
    split = tmp_path / "split"
    shutil.copytree(joint, split)
    NE.main(["collect", "--out", str(joint), "--batch", "d1", "--roles", "explainer,second_reader", "--backend", "manual"])
    one = ["collect", "--out", str(split), "--batch", "d1", "--backend", "manual", "--roles"]
    with pytest.raises(ValueError, match=r"d1-explainer is not collected \(4 ids"):
        NE.main([*one, "second_reader"])
    assert not (split / "explain" / "calls.jsonl").exists()
    NE.main([*one, "explainer"])
    explainer_rows = (split / "explain" / "explanations.jsonl").read_bytes()
    NE.main([*one, "second_reader"])
    for name in ("explanations.jsonl", "calls.jsonl"):
        assert (split / "explain" / name).read_bytes() == (joint / "explain" / name).read_bytes(), name
    assert (split / "explain" / "explanations.jsonl").read_bytes().startswith(explainer_rows)   # appended only

    def models(d):
        m = NV.read_json(d / "explain" / "manifest.json")["models_by_role"]
        return {k: v for k, v in m.items() if k != "utc"}
    assert models(split) == models(joint) and set(models(split)["answered"]) == {"explainer", "second_reader"}
    with pytest.raises(ValueError, match="already in"):
        NE.main([*one, "second_reader"])


def test_a_failed_multi_batch_collect_logs_nothing_and_can_be_rerun(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)
    F.answer_batch(explain, "d1-explainer", _explanation)
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 2.0})
    with pytest.raises(ValueError, match=r"second_reader:L00N0\+:1: \$\.confidence: 2\.0 > maximum 1"):
        NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader",
                 "--backend", "manual"])
    assert not (explain / "calls.jsonl").exists() and not (explain / "explanations.jsonl").exists()
    llm.responses_path(explain, "d1-second_reader").unlink()
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 0.3})
    NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader",
             "--backend", "manual"])
    assert len(llm.read_jsonl(explain / "calls.jsonl")) == 8


VALID = {"explainer": {"explanation": "e", "jobs": [{"name": "n", "related_to_others": False, "reasoning": "r"}],
                       "class": "ONE", "confidence": 0.5},
         "writer": {"positives": ["a"] * 5, "near_misses": ["b"] * 5, "distant_concept": "c",
                    "distant": ["d"] * 5, "max": "m"},
         "writer_job": {"positives": ["a"] * 5},
         "clusterer": {"clusters": [{"c": 0, "name": "n", "echo": False, "exemplar_ranks": [0, 1]}], "pairs": []},
         "judge": {"answer": "A", "reason": "r"}}
INVALID = {"explainer": {**VALID["explainer"], "class": "MANY"},
           "writer": {**VALID["writer"], "positives": ["a"] * 4},
           "writer_job": {"positives": ["a"] * 6},
           "clusterer": {"clusters": [{"c": 0, "name": "n", "exemplar_ranks": [0]}], "pairs": []},  # no echo flag
           "judge": {"answer": "C", "reason": "r"}}


@pytest.mark.parametrize("role", llm.ROLES)
def test_every_role_schema_is_enforced(role):
    key = "explainer" if role == "second_reader" else role
    assert llm.schema_errors(NE.SCHEMAS[role], VALID[key]) == []
    assert llm.schema_errors(NE.SCHEMAS[role], INVALID[key]) != []
    assert llm.schema_errors(NE.SCHEMAS[role], {**VALID[key], "extra": 1}) != []
    system, user, _ = NE.load_prompt(role)
    assert "JSON object" in system and NE.PLACEHOLDER_RE.search(user)


def test_changed_prompt_is_refused(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    shutil.copytree(NE.PROMPT_DIR, prompts)
    monkeypatch.setattr(NE, "PROMPT_DIR", prompts)
    NE.ensure_manifest(tmp_path / "explain")
    (prompts / "judge.md").write_text((prompts / "judge.md").read_text() + "\nextra line\n")
    with pytest.raises(ValueError, match=r"prompts \['judge'\] changed"):
        NE.ensure_manifest(tmp_path / "explain")
    (prompts / "second_reader.md").write_text("<!-- system -->\nx\n<!-- user -->\n{pole}\n")
    with pytest.raises(ValueError, match="second_reader.md must equal explainer.md"):
        NE.ensure_manifest(tmp_path / "explain2")


def test_writer_sees_only_the_sentence_and_texts_carry_a_shuffle(bank, tmp_path):
    corpus, harvest = bank
    explain = _prepare_explain(bank, tmp_path)
    F.answer_batch(explain, "d1-explainer", _explanation)
    NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer", "--backend", "manual"])
    NE.main(["prepare-writer", "--out", str(tmp_path), "--batch", "w1", "--model", "test-model"])
    writer = llm.read_jsonl(llm.requests_path(explain, "w1-writer"))
    assert [r["user"] for r in writer if r["id"] == "writer:L00N0+:0"] == ["Explanation: Fires on boats at sea."]
    jobs = llm.read_jsonl(llm.requests_path(explain, "w1-writer_job"))
    assert [(r["id"], r["user"]) for r in jobs] == [("writer_job:L00N1-:j0", "Job: Rain falling on a town."),
                                                    ("writer_job:L00N1-:j1", "Job: Bread in a kitchen.")]
    for r in writer + jobs:                                           # no context ever reaches the writer
        assert "⟦" not in r["user"] and "⟦" not in r["system"]
    F.answer_batch(explain, "w1-writer", _writer)
    F.answer_batch(explain, "w1-writer_job", lambda r: {"positives": ["rain on the town"] * 5})
    NE.main(["collect-writer", "--out", str(tmp_path), "--batch", "w1", "--backend", "manual",
             "--harvest", str(harvest), "--corpus", str(corpus), "--seed", "7"])
    texts = llm.read_jsonl(explain / "texts.jsonl")
    assert len(texts) == 4 * 16 + 2 * 5
    assert {t["text_id"] for t in texts if t["kind"] == "job_pos"} == {f"L00N1-:job_pos:{k}" for k in range(10)}
    shuffle = NV.read_json(explain / "shuffle.json")
    assert shuffle["seed"] == 7 and NV.read_json(explain / "manifest.json")["seeds"]["shuffle"] == 7
    for pole, other in shuffle["pairs"].items():
        assert NV.parse_pole(other)[0] == NV.parse_pole(pole)[0] and NV.parse_pole(other)[1] != NV.parse_pole(pole)[1]
    assert set(NV.read_json(explain / "probe_poles.json")) == set(POLES) | set(shuffle["pairs"].values())


# --- clusters and judge on a 16-checkpoint grid ----------------------------------------------

S = ["L00N0+", "L00N1-", "L00N2+"]


def design(t, pole):
    """c0 = ranks 0-24 (main, born T_1), c1 = ranks 25-49 (born T_4, late); q999 1, max 10."""
    mag = np.zeros(50)
    if t >= 1:
        mag[:25] = 8.0
    if t >= 4:
        mag[25:] = 5.0
    return mag, 1.0, 10.0


@pytest.fixture(scope="module")
def grid(bank, tmp_path_factory):
    corpus, _ = bank
    root = tmp_path_factory.mktemp("grid")
    dirs = F.build_grid(root, corpus, S, design)
    end = NV.load_npz(dirs[-1] / "topk" / "layer_00.npz")
    end["g"][0, 0] = np.abs(end["g"][0, 0]) + 0.1                    # L00N0+: no double-negative context
    end["g"][2, 0, 30] = -1.0                                         # L00N2+: rank 30 is double negative
    end["u"] = end["a"] / F.silu(end["g"])
    np.savez(dirs[-1] / "topk" / "layer_00.npz", **end)
    # Put L00N0+'s late exemplars (ranks 25-29) at the top of every pole at T_3 (the late cluster's
    # t_before) and its main exemplars (ranks 0-4) at the top of every pole at T_14 (the control's),
    # so each exclusion has something to remove.
    for t, ranks in ((3, slice(25, 30)), (14, slice(0, 5))):
        tk = NV.load_npz(dirs[t] / "topk" / "layer_00.npz")
        tk["tok"][:, :, :5] = end["tok"][0, 0, ranks]
        np.savez(dirs[t] / "topk" / "layer_00.npz", **tk)
    NV.layer_file.cache_clear()
    return dirs


def _clusters(tmp_path, corpus, grid, answer=F.two_halves):
    (tmp_path / "S.json").write_text(json.dumps(S))
    NE.main(["positions", "--harvest", str(grid[-1]), "--out", str(tmp_path), "--poles", str(tmp_path / "S.json")])
    NE.main(["prepare-clusters", "--harvest", str(grid[-1]), "--corpus", str(corpus), "--out", str(tmp_path),
             "--batch", "c1", "--model", "test-model", "--seed", "9"])
    explain = tmp_path / "explain"
    plans = F.plans_by_id(explain, "c1-clusterer")
    for s in (0, 1):
        F.answer_batch(explain, f"c1-clusterer-s{s}", lambda r: answer(plans[r["id"]]))
    return explain, plans


def test_clusterer_prompt_v2_states_the_partition_and_its_example_is_one(bank, grid, tmp_path):
    """Clusterer prompt v2 (W4, 2026-10-21): the shipped example is a complete partition of N = 6 and passes
    the same checks collect-clusters applies; a fixture answer passes them for N = 50; a new directory records
    v2 and the requests carry N and the last rank."""
    system, user, sha = NE.load_prompt("clusterer")
    assert sha == NE.CLUSTERER_PROMPTS["v2"] != NE.CLUSTERER_PROMPTS["v1"]
    example = json.loads(system[system.index('{"clusters"'):])
    assert NE.partition_errors(example, 6) == [] and NE.echo_errors(example, []) == []
    assert "every number from 0 to N-1 exactly once" in system and "at least one number" in system
    corpus, _ = bank
    explain, plans = _clusters(tmp_path, corpus, grid)
    for plan in plans.values():
        x = F.two_halves(plan)
        assert NE.partition_errors(x, plan["n_valid"]) == [] and NE.echo_errors(x, plan["double_negative_ranks"]) == []
    m = NV.read_json(explain / "manifest.json")
    assert m["clusterer_prompt_version"] == "v2" and m["prompts"]["clusterer"] == NE.CLUSTERER_PROMPTS["v2"]
    req = llm.read_jsonl(llm.requests_path(explain, "c1-clusterer-s0"))[0]
    assert "N = 50: its 50 strongest contexts, numbered 0 to 49, strongest first:" in req["user"]
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])


def test_an_older_directory_keeps_its_clusterer_prompt_and_refuses_the_new_one(tmp_path):
    NE.ensure_manifest(tmp_path / "explain")
    path = tmp_path / "explain" / "manifest.json"
    m = NV.read_json(path)
    m["prompts"]["clusterer"] = NE.CLUSTERER_PROMPTS["v1"]               # a directory made before v2
    del m["clusterer_prompt_version"]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=r"prompts \['clusterer'\] changed"):
        NE.ensure_manifest(tmp_path / "explain")
    assert NV.read_json(path)["prompts"]["clusterer"] == NE.CLUSTERER_PROMPTS["v1"]


def test_cluster_answers_must_partition_the_top50(bank, grid, tmp_path):
    corpus, _ = bank

    def bad(plan):
        x = F.two_halves(plan)
        if plan["pole"] == "L00N1-" and plan["sample"] == 0:
            x["clusters"][1]["exemplar_ranks"] = [24] + x["clusters"][1]["exemplar_ranks"]   # rank 24 twice
        return x
    explain, plans = _clusters(tmp_path, corpus, grid, bad)
    reqs = llm.read_jsonl(llm.requests_path(explain, "c1-clusterer-s0"))
    assert len(re.findall(r"^#\d+ \(", reqs[0]["user"], re.M)) == 50
    assert all(re.fullmatch(r"clusterer:c1:\d+", r["id"]) for r in reqs)        # opaque ids (B3)
    assert sorted(int(i.rsplit(":", 1)[1]) for i in plans) == list(range(2 * len(S)))
    with pytest.raises(ValueError, match=r"L00N1- sample 0: exemplar_ranks do not cover 0\.\.49"):
        NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    assert not (explain / "calls.jsonl").exists()


def test_reopen_declined_clusterer_samples_answers_and_collects_them(bank, grid, tmp_path, capsys):
    """A sample declined at the old cap is reopened (after its batch was collected): it leaves the declined
    file, the manifest records the old reason, attempts and utc, the collect outputs are removed and rewritten
    by the next collect, and the attempts keep counting under the new cap."""
    corpus, _ = bank
    bad = {}

    def answer(plan):
        x = F.two_halves(plan)
        if plan["pole"] == "L00N1-" and plan["sample"] == 1:              # rank 24 twice: a format error
            bad["id"] = plan["id"]
            x["clusters"][1]["exemplar_ranks"] = [24] + x["clusters"][1]["exemplar_ranks"]
        return x
    explain, plans = _clusters(tmp_path, corpus, grid, answer)
    rid = bad["id"]
    batch = next(b for b in ("c1-clusterer-s0", "c1-clusterer-s1") if any(r["id"] == rid for r in llm.read_jsonl(llm.requests_path(explain, b))))
    args = ["--out", str(tmp_path), "--batch", "c1", "--roles", "clusterer", "--backend", "manual"]
    NE.main(["reject", *args, "--max-attempts", "1"])                     # declined at the old cap
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    assert any(r.get("declined") for r in llm.read_jsonl(explain / "clusters.jsonl"))
    old = llm.declined(explain, batch)[rid]
    clusters_sha = NV.file_sha256(explain / "clusters.jsonl")
    NE.main(["reopen", "--out", str(tmp_path), "--batch", "c1", "--roles", "clusterer"])
    assert not llm.declined(explain, batch) and not (explain / "clusters.jsonl").exists()
    assert not (explain / "calls.jsonl").read_text().strip()               # every row of the batch left calls.jsonl
    rec = NV.read_json(explain / "manifest.json")["reopened"][0]
    assert rec["ids"][rid] == {"batch": batch, "role": "clusterer", "reason": old["reason"], "finish_reason": None,
                               "attempts": 1, "declined_utc": old["utc"]}
    assert rec["uncollected"] == {"calls_rows_removed": 2 * len(S) - 1, "removed_sha256": {"clusters.jsonl": clusters_sha}}
    with pytest.raises(ValueError, match="missing answer"):              # pending again: collect wants it
        NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    resp = llm.responses_path(explain, batch)                            # asked again: the same format error
    llm.append_jsonl(resp, [{"id": rid, "model": "test-model", "text": json.dumps(answer(plans[rid])), "utc": "2026-10-21T00:00:00Z"}])
    NE.main(["reject", *args, "--max-attempts", "3"])                     # attempt 2 of the new cap of 3
    assert not llm.declined(explain, batch) and len(llm.read_jsonl(llm.rejected_path(explain, batch))) == 2
    llm.append_jsonl(resp, [{"id": rid, "model": "test-model", "text": json.dumps(F.two_halves(plans[rid])),
                             "utc": "2026-10-21T00:00:00Z"}])
    capsys.readouterr()
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    rows = llm.read_jsonl(explain / "clusters.jsonl")
    assert len(rows) == 2 * len(S) and not any(r.get("declined") for r in rows)
    assert len(llm.read_jsonl(explain / "calls.jsonl")) == 2 * len(S) and "declined" not in capsys.readouterr().out
    NE.main(["reopen", "--out", str(tmp_path), "--batch", "c1", "--roles", "clusterer"])   # nothing declined now
    assert "nothing declined; nothing changed" in capsys.readouterr().out
    assert len(NV.read_json(explain / "manifest.json")["reopened"]) == 1


def test_reopen_refuses_clusters_a_judge_batch_was_prepared_on(bank, grid, tmp_path):
    corpus, _ = bank

    def answer(plan):
        x = F.two_halves(plan)
        if plan["pole"] == "L00N0+" and plan["sample"] == 0:
            x["clusters"][0]["exemplar_ranks"] = []                        # an empty cluster: minItems
        return x
    explain, _ = _clusters(tmp_path, corpus, grid, answer)
    NE.main(["reject", "--out", str(tmp_path), "--batch", "c1", "--roles", "clusterer", "--backend", "manual",
             "--max-attempts", "1"])
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(tmp_path),
             "--batch", "j1", "--model", "test-model", "--seed", "3", "--gate", str(tmp_path)])
    before = {p: p.read_bytes() for p in (explain / "clusters.jsonl", explain / "calls.jsonl")}
    with pytest.raises(ValueError, match="a judge batch was prepared on these clusters"):
        NE.main(["reopen", "--out", str(tmp_path), "--batch", "c1", "--roles", "clusterer"])
    assert {p: p.read_bytes() for p in before} == before


def test_echo_cluster_must_hold_exactly_the_double_negative_contexts(bank, grid, tmp_path):
    corpus, _ = bank
    explain, plans = _clusters(tmp_path, corpus, grid, lambda plan: F.two_halves({**plan, "double_negative_ranks": []}))
    dn = {p["pole"]: p["double_negative_ranks"] for p in plans.values()}
    assert dn["L00N0+"] == [] and dn["L00N1-"] == [] and 30 in dn["L00N2+"]
    with pytest.raises(ValueError, match=r"L00N2\+ sample 0: the echo cluster holds ranks \[\]"):
        NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])


def test_judge_records_exclude_exemplars_and_show_no_numbers(bank, grid, tmp_path):
    corpus, _ = bank
    explain, _ = _clusters(tmp_path, corpus, grid)
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    clusters = {(r["pole"], c["c"]): c for r in llm.read_jsonl(explain / "clusters.jsonl") if r["sample"] == 0
                for c in r["clusters"]}
    assert clusters[("L00N2+", 2)]["echo"] and 30 in clusters[("L00N2+", 2)]["exemplar_ranks"]
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(tmp_path),
             "--batch", "j1", "--model", "test-model", "--seed", "3", "--gate", str(tmp_path)])
    plans = F.plans_by_id(explain, "j1-judge")
    reqs = [r for s in (0, 1) for r in llm.read_jsonl(llm.requests_path(explain, f"j1-judge-s{s}"))]
    assert len(reqs) == len(plans) == len(S) * 12                     # (4 random + 2 hard) x (test + control)
    assert all(re.fullmatch(r"judge:j1:\d+", r["id"]) for r in reqs)                # opaque ids (B3)
    assert sorted(int(i.rsplit(":", 1)[1]) for i in plans) == list(range(len(plans)))
    assert {p["cluster_requests_in_file"] for p in plans.values()} == {3}           # B4: shown in plans/
    for s in (0, 1):                                                  # B4: file order is not plan order
        in_file = [r["id"] for r in llm.read_jsonl(llm.requests_path(explain, f"j1-judge-s{s}"))]
        assert in_file != [i for i, p in plans.items() if p["sample"] == s] and sorted(in_file) != in_file
    assert NV.read_json(explain / "manifest.json")["seeds"]["judge:j1"] == 3
    assert {tuple(p["shown_order"]) for p in plans.values()} == {("true", "decoy"), ("decoy", "true")}
    assert not any(p["cluster"].endswith(":c2") for p in plans.values())   # the echo cluster is never judged
    for pole in S:
        late = [p for p in plans.values() if p["cluster"] == f"{pole}:c1"]
        ctrl = [p for p in plans.values() if p["cluster"] == f"{pole}:c0"]
        assert {p["t_before"] for p in late} == {"T_3"} and all(not p["control"] for p in late)
        assert {p["t_before"] for p in ctrl} == {"T_14"} and all(p["control"] for p in ctrl)
        assert sorted(p["decoy_type"] for p in late) == ["hard"] * 2 + ["random"] * 4
    end_tok = NV.layer_file(str(grid[-1]), "topk", 0)["tok"]

    def exemplar_contexts(cluster, t):                             # E_c rendered as the judge would see it
        pole, c = cluster.split(":")
        _, n, s = NV.parse_pole(pole)
        ranks = clusters[(pole, int(c[1:]))]["exemplar_ranks"]
        return {NE.marked(NE.render(*NE.harvest_corpus(grid[t], corpus), int(end_tok[n, s, k]), NE.JUDGE_BEFORE))
                for k in ranks}

    def window_contexts(cluster, t):                               # every top-50 context at t in a window
        pole, c = cluster.split(":")
        _, n, s = NV.parse_pole(pole)
        offsets = NE.harvest_corpus(grid[t], corpus)[1]
        windows = NE.exemplar_windows(offsets, [int(end_tok[n, s, k]) for k in clusters[(pole, int(c[1:]))]["exemplar_ranks"]])
        return {NE.marked(x) for q in (NV.pole_id(0, m, sg) for m in range(6) for sg in (0, 1))
                for x in NE.pole_contexts(grid[t], corpus, q, 50, NE.JUDGE_BEFORE) if NE.in_windows(x["tok"], windows)}
    for r in reqs:
        plan = plans[r["id"]]
        assert not re.search(r"\d", r["user"]), r["id"]               # no numbers anywhere the judge looks
        head, rec_b = r["user"].split("\n\nRecord B:\n")
        exemplars, rec_a = head.split("\n\nRecord A:\n")
        a_lines, b_lines = rec_a.splitlines(), rec_b.splitlines()
        assert len(a_lines) == len(b_lines) == 20
        t = int(plan["t_before"][2:])
        assert not window_contexts(plan["cluster"], t) & {x[2:] for x in a_lines + b_lines}
        assert 1 < len(exemplars.splitlines()) <= 1 + 10
    # each exclusion had something to remove: at T_3 and T_14 every pole's top five are L00N0+ exemplars
    for t, cluster in ((3, "L00N0+:c1"), (14, "L00N0+:c0")):
        for pole in ("L00N0+", "L00N3+"):
            assert {NE.marked(c) for c in NE.pole_contexts(grid[t], corpus, pole, 5, NE.JUDGE_BEFORE)} <= exemplar_contexts(cluster, t)
    for s in (0, 1):                                                  # answer "A" everywhere
        F.answer_batch(explain, f"j1-judge-s{s}", lambda r: {"answer": "A", "reason": "r"})
    NE.main(["collect-judge", "--out", str(tmp_path), "--batch", "j1", "--backend", "manual"])
    for j in llm.read_jsonl(explain / "judgments.jsonl"):
        assert j["correct"] == (j["shown_order"][0] == "true")


def test_judge_gate_through_message_batches_with_a_fake_client(bank, grid, tmp_path, monkeypatch, capsys):
    """The W4 gate on the direct Anthropic path minus the model call: prepare-judge with the bare Sonnet id,
    run-batch with the shipped settings, a fake SDK answering every line correctly except one control line,
    collect-batch, collect-judge with backend anthropic (unchanged), verdict judge-gate. The submission
    carries the judge's exact text and one cache breakpoint per line; the gate records the bare model id."""
    corpus, _ = bank
    explain, _ = _clusters(tmp_path, corpus, grid)
    NE.main(["collect-clusters", "--out", str(tmp_path), "--batch", "c1", "--backend", "manual"])
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(tmp_path),
             "--batch", "j1", "--model", "claude-sonnet-5-5", "--seed", "3", "--gate", str(tmp_path)])
    plans = F.plans_by_id(explain, "j1-judge")
    reqs = {r["id"]: r for s in (0, 1) for r in llm.read_jsonl(llm.requests_path(explain, f"j1-judge-s{s}"))}
    wrong = sorted(i for i, p in plans.items() if p["control"] and p["decoy_type"] == "random")[0]
    created = []

    def results(mb_id):
        cid_to_id = {e["custom_id"]: e["id"] for e in json.loads(
            (llm.submissions_dir(explain) / f"{mb_id}.json").read_text())["requests"]}
        out = []
        for r in created[0][::-1]:
            rid = cid_to_id[r["custom_id"]]
            truth = "A" if plans[rid]["shown_order"][0] == "true" else "B"
            answer = {"A": "B", "B": "A"}[truth] if rid == wrong else truth
            msg = SimpleNamespace(id="msg", model=r["params"]["model"], stop_reason="end_turn", stop_details=None,
                                  usage=SimpleNamespace(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0,
                                                        cache_read_input_tokens=0),
                                  content=[SimpleNamespace(type="text", text=json.dumps({"answer": answer, "reason": "r"}))])
            out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)))
        return out
    batches = SimpleNamespace(create=lambda requests: created.append(list(requests)) or SimpleNamespace(id="msgbatch_1"),
                              retrieve=lambda mb_id: SimpleNamespace(processing_status="ended", request_counts={},
                                                                     ended_at="2026-11-01T00:00:00Z"),
                              results=results)
    monkeypatch.setattr(llm, "anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(batches=batches)))
    settings = str(Path(llm.__file__).parent / "settings" / "anthropic_sonnet55.json")
    llm.main(["run-batch", "--explain", str(explain), "--batches", "j1-judge-s0,j1-judge-s1", "--settings", settings])
    assert len(created[0]) == len(reqs)
    prompts = Counter((r["system"], r["user"]) for r in reqs.values())
    assert 1 in prompts.values() and 2 in prompts.values()             # the shown order makes some lines exact twins
    for r in created[0]:
        p = r["params"]
        rid = json.loads((llm.submissions_dir(explain) / "msgbatch_1.json").read_text())["requests"][int(r["custom_id"][1:])]["id"]
        assert p["model"] == "claude-sonnet-5-5" and p["system"] == reqs[rid]["system"] and "temperature" not in p
        assert "".join(b["text"] for b in p["messages"][0]["content"]) == reqs[rid]["user"]
        assert json.dumps(p).count("cache_control") == (2 if prompts[(reqs[rid]["system"], reqs[rid]["user"])] > 1 else 1)
    llm.main(["collect-batch", "--explain", str(explain)])
    NE.main(["collect-judge", "--out", str(tmp_path), "--batch", "j1", "--backend", "anthropic"])
    NV.main(["judge-gate", "--explain", str(tmp_path), "--out", str(tmp_path)])
    gate = NV.read_json(tmp_path / "verdict" / "judge_gate.json")
    n_ctrl_poles = len({p["cluster"].split(":")[0] for p in plans.values() if p["control"]})
    assert gate["judge"]["models"] == ["claude-sonnet-5-5"] and gate["n_poles"] == n_ctrl_poles
    assert gate["a_ctrl"] == pytest.approx(1 - 0.25 / n_ctrl_poles) and gate["gate"] == "pass"
    assert {c["backend"] for c in llm.read_jsonl(explain / "calls.jsonl") if c["role"] == "judge"} == {"anthropic"}


def test_judge_records_drop_every_context_in_an_exemplar_window(bank, tmp_path):
    """SPEC §2.1 and §2.4 as amended: the window is the explainer's, 256 tokens before an exemplar to
    16 after, in its document. Record contexts 10 after or 100 before an exemplar are removed; 100
    after, 300 before, or just across the document boundary, they stay."""
    corpus, harvest = bank
    tokens, offsets, tok = NE.corpus(str(corpus), F.TOKNAME)
    d = int(np.argmax(np.diff(offsets)))                              # the long document, the corpus's last
    start = int(offsets[d])
    e = start + 500
    after10, after100, before100, before300 = e + 10, e + 100, e - 100, e - 300
    across = start                                                   # first token of the long document ...
    prev_end = start                                                 # ... after an exemplar ending the one before
    copy = tmp_path / "h"
    shutil.copytree(harvest, copy)
    tk = NV.load_npz(copy / "topk" / "layer_00.npz")
    tk["tok"][1, 0, :5] = [after10, after100, before100, before300, across]
    np.savez(copy / "topk" / "layer_00.npz", **tk)

    def shown(tok_index):
        return NE.marked(NE.render(tokens, offsets, tok, tok_index, NE.JUDGE_BEFORE))
    rec = NE.record(copy, corpus, "L00N1+", NE.exemplar_windows(offsets, [e]), 50)
    assert shown(after10) not in rec and shown(before100) not in rec   # inside the window: removed
    assert shown(after100) in rec and shown(before300) in rec           # outside it: kept
    rec = NE.record(copy, corpus, "L00N1+", NE.exemplar_windows(offsets, [prev_end - 2]), 50)
    assert shown(across) in rec                                        # two tokens on, another document: kept
    w = NE.exemplar_windows(offsets, [start + 2, e, prev_end - 2])
    prev_start = int(offsets[d - 1])
    assert w.tolist() == [[max(prev_start, prev_end - 258), prev_end - 1], [start, start + 18], [e - 256, e + 16]]
    assert not NE.in_windows(-1, w)


def test_negative_pole_shows_the_neurons_vocabulary_swapped(bank, tmp_path):
    """vocab.json is per pole and its "-" entry is already the sign flip (JOURNAL 2026-10-25, F41); the negative
    pole's prompt shows the neuron's ("+" entry's) suppressed items as promoted (sign flipped) and its promoted
    items as suppressed."""
    corpus, _ = bank
    harvest = F.build_harvest(tmp_path / "h", corpus, n_layers=4, n_neurons=8, poles=["L03N7-", "L03N7+"])
    vocab = NV.read_json(harvest / "poles" / "vocab.json")["L03N7+"]
    (tmp_path / "p.json").write_text(json.dumps(["L03N7-", "L03N7+"]))
    NE.main(["prepare", "--harvest", str(harvest), "--corpus", str(corpus), "--out", str(tmp_path / "o"),
             "--batch", "v", "--roles", "explainer", "--poles", str(tmp_path / "p.json"), "--model", "test-model"])
    users = {r["id"]: r["user"] for r in llm.read_jsonl(llm.requests_path(tmp_path / "o" / "explain", "v-explainer"))}

    def listed(user, which):
        line = next(x for x in user.splitlines() if x.startswith(f"Output vocabulary, most {which}: "))
        return [(json.loads(m.group(1)), float(m.group(2))) for m in re.finditer(r'("[^"]*") \(([+-][\d.e+-]+)\)', line)]
    neg, pos = users["explainer:L03N7-:0"], users["explainer:L03N7+:0"]
    assert [t for t, _ in listed(neg, "promoted")] == [t for t, _ in vocab["suppressed"]]
    assert [t for t, _ in listed(neg, "suppressed")] == [t for t, _ in vocab["promoted"]]
    assert all(s > 0 for _, s in listed(neg, "promoted")) and all(s < 0 for _, s in listed(neg, "suppressed"))
    assert [t for t, _ in listed(pos, "promoted")] == [t for t, _ in vocab["promoted"]]


def test_check_positions_reads_the_harvests_own_positions_and_requires_agreement(bank, grid, tmp_path):
    """DATA_CONTRACT §7: with T_15's poles/positions.npz present, the explain directory's positions must
    equal it, and it must equal T_15's topk/; without it, topk/ decides."""
    _clusters(tmp_path, bank[0], grid)
    explain = tmp_path / "explain"
    end = tmp_path / "T15"
    shutil.copytree(grid[-1], end)
    assert set(NE.check_positions(explain, end)) == set(S)              # positions.npz == topk == explain
    own = NV.load_npz(end / "poles" / "positions.npz")
    own["tok"][1, 7] += 1                                             # the harvest's file disagrees with topk/
    np.savez(end / "poles" / "positions.npz", **own)
    with pytest.raises(ValueError, match=r"positions of L00N1- differ from .*explain/positions.npz"):
        NE.check_positions(explain, end)
    mine = NV.load_npz(explain / "positions.npz")                     # the same change in the explain directory:
    mine["tok"][1, 7] += 1                                            # it agrees with the harvest's file, which
    np.savez(explain / "positions.npz", **mine)                       # still disagrees with topk/
    with pytest.raises(ValueError, match=r"positions of L00N1- differ from the same harvest's topk/"):
        NE.check_positions(explain, end)
    mine["tok"][1, 7] -= 1
    np.savez(explain / "positions.npz", **mine)
    (end / "poles" / "positions.npz").unlink()                        # absent: topk/ decides
    assert set(NE.check_positions(explain, end)) == set(S)
    tk = NV.load_npz(end / "topk" / "layer_00.npz")
    tk["tok"][2, 0, 3] += 1
    np.savez(end / "topk" / "layer_00.npz", **tk)
    NV.layer_file.cache_clear()
    with pytest.raises(ValueError, match=r"top-50 of L00N2\+ differs from"):
        NE.check_positions(explain, end)


# --- reject: re-asking answers collect refuses -----------------------------------------------

class _Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return io.BytesIO(json.dumps(self.body).encode())

    def __exit__(self, *a):
        return False


def _explainer_endpoint(leaky: dict, asked: list):
    """A stand-in for the API: answers explainer and second_reader requests by pole; a pole in `leaky`
    gets an explanation pointing at a shown context for as many explainer calls as its count says."""
    def urlopen(http, timeout):
        body = json.loads(http.data)
        role = body["response_format"]["json_schema"]["name"]
        pole = re.search(r"^Pole (\S+):", body["messages"][1]["content"], re.M).group(1)
        asked.append(f"{role}:{pole}")
        x = _explanation({"id": f"{role}:{pole}:0", "role": role})
        if role == "second_reader":
            x["confidence"] = 0.3
        if role == "explainer" and leaky.get(pole, 0) > 0:
            leaky[pole] -= 1
            x["explanation"] = "Fires on the word marked in context 2."
        return _Reply({"model": "test-model", "usage": {"prompt_tokens": 9, "completion_tokens": 4},
                       "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(x)}}]})
    return urlopen


def _api_round(explain, monkeypatch, leaky, asked):
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", _explainer_endpoint(leaky, asked))
    settings = {"backend": "openai_compat", "base_url": "https://api.example/v1", "key_env": "NEURONS_API_KEY_TEST",
                "models": {r: "test-model" for r in llm.ROLES}, "temperature": 1.0, "response_format": "json_schema"}
    for role in ("explainer", "second_reader"):
        llm.run_batch(explain, f"d1-{role}", settings, concurrency=4)


def test_reject_moves_the_leaky_answer_and_the_resumed_run_asks_only_for_it(bank, tmp_path, monkeypatch, capsys):
    explain = _prepare_explain(bank, tmp_path)
    leaky, asked = {"L01N3-": 1}, []
    _api_round(explain, monkeypatch, leaky, asked)
    args = ["--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader", "--backend", "openai_compat"]
    with pytest.raises(ValueError, match="explanation of L01N3- contains a reference to a shown context"):
        NE.main(["collect", *args])
    NE.main(["reject", *args])
    out = capsys.readouterr().out
    assert "d1-explainer: 1 rejected: explainer:L01N3-:0" in out and "d1-second_reader: nothing to reject" in out
    rejected = llm.read_jsonl(llm.rejected_path(explain, "d1-explainer"))
    assert [r["id"] for r in rejected] == ["explainer:L01N3-:0"] and "a reference to a shown context" in rejected[0]["reason"]
    assert rejected[0]["rejected_utc"] and "context 2" in rejected[0]["text"]
    assert "explainer:L01N3-:0" not in {r["id"] for r in llm.read_jsonl(llm.responses_path(explain, "d1-explainer"))}
    asked.clear()
    _api_round(explain, monkeypatch, leaky, asked)                     # resume: only the rejected id
    assert asked == ["explainer:L01N3-"]
    usage = [u["id"] for u in llm.read_jsonl(llm.usage_path(explain, "d1-explainer"))]
    assert usage.count("explainer:L01N3-:0") == 2                     # both calls stay in the cost record
    NE.main(["collect", *args])
    assert len(llm.read_jsonl(explain / "explanations.jsonl")) == 8


def test_reject_on_a_clean_batch_changes_nothing(bank, tmp_path, monkeypatch, capsys):
    explain = _prepare_explain(bank, tmp_path)
    _api_round(explain, monkeypatch, {}, [])
    path = llm.responses_path(explain, "d1-explainer")
    rows = path.read_text().splitlines(keepends=True)
    path.write_text("".join(rows[:-1]))                                # one answer missing: not a rejection
    before = {r: llm.responses_path(explain, f"d1-{r}").read_bytes() for r in ("explainer", "second_reader")}
    NE.main(["reject", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader",
             "--backend", "openai_compat"])
    assert capsys.readouterr().out.count("nothing to reject; responses unchanged") == 2
    assert {r: llm.responses_path(explain, f"d1-{r}").read_bytes() for r in before} == before
    assert not llm.rejected_path(explain, "d1-explainer").exists()


def test_reject_declines_at_the_attempts_cap_and_collect_records_the_decline(bank, tmp_path, monkeypatch, capsys):
    explain = _prepare_explain(bank, tmp_path)
    leaky, asked = {"L00N0+": 5}, []                                   # this pole keeps leaking
    args = ["--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader", "--backend", "openai_compat"]
    _api_round(explain, monkeypatch, leaky, asked)
    NE.main(["reject", *args, "--max-attempts", "2"])                  # first rejection
    _api_round(explain, monkeypatch, leaky, asked)
    NE.main(["reject", *args, "--max-attempts", "2"])                  # second: declined, not asked again
    assert "1 declined after 2 attempts (not asked again): explainer:L00N0+:0" in capsys.readouterr().out
    gave_up = llm.declined(explain, "d1-explainer")["explainer:L00N0+:0"]
    assert (gave_up["role"], gave_up["attempts"], gave_up["finish_reason"]) == ("explainer", 2, "stop")
    asked.clear()
    _api_round(explain, monkeypatch, leaky, asked)
    assert asked == []                                                 # run skips the declined id
    NE.main(["collect", *args])
    assert "declined: explainer:L00N0+:0: explainer declined" in capsys.readouterr().out
    rows = {(r["pole"], r["sample"]): r for r in llm.read_jsonl(explain / "explanations.jsonl")}
    assert rows[("L00N0+", 0)]["class"] == "UNK" and rows[("L00N0+", 0)]["declined"].startswith("explainer declined (")
    assert len(rows) == 8 and len(llm.read_jsonl(explain / "calls.jsonl")) == 7


def test_x7_decodes_only_usable_text_and_names_only_real_scripts():
    """A random identifier that is valid Base64 (it decodes to valid UTF-8 like "E짢Wl", as in the W2-D
    d1 requests) gets no line; a Base64 English sentence does; MICRO SIGN and a modifier letter are not
    scripts."""
    import base64
    ident = base64.b64encode("E짢Wl".encode()).decode()
    assert NE.B64_RE.fullmatch(ident) and NE.base64_text(ident) is None
    assert not [n for n in NE.encoding_notes(f"token {ident} end") if n.startswith("decoded")]
    sentence = base64.b64encode(b"the quick brown fox jumps").decode()
    assert NE.encoding_notes(f"payload {sentence} ok") == ["decoded: the quick brown fox jumps"]
    assert NE.script_of("µ") is None and NE.script_of("ʼ") is None and NE.encoding_notes("5 µm wide ʼok") == []
    assert NE.script_of("東") == "CJK" and NE.script_of("ж") == "CYRILLIC" and NE.script_of("ㅎ") == "HANGUL"


def test_fork_writer_redoes_the_writer_on_a_copy_and_leaves_the_source_alone(bank, tmp_path):
    corpus, harvest = bank
    src = tmp_path / "src"
    src.mkdir()
    explain = _prepare_explain(bank, src)
    F.answer_batch(explain, "d1-explainer", _explanation)
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 0.3})
    NE.main(["collect", "--out", str(src), "--batch", "d1", "--roles", "explainer,second_reader", "--backend", "manual"])
    NE.main(["prepare-writer", "--out", str(src), "--batch", "d1", "--model", "test-model"])
    F.answer_batch(explain, "d1-writer", _writer)
    F.answer_batch(explain, "d1-writer_job", lambda r: {"positives": ["rain on the town"] * 5})
    NE.main(["collect-writer", "--out", str(src), "--batch", "d1", "--backend", "manual", "--harvest", str(harvest),
             "--corpus", str(corpus), "--seed", "0"])
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    dst = tmp_path / "dst"
    NE.main(["fork-writer", "--source", str(src), "--out", str(dst)])
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before          # source untouched
    fork = dst / "explain"
    names = {p.relative_to(fork).as_posix() for p in fork.rglob("*") if p.is_file()}
    assert not names & {"texts.jsonl", "shuffle.json", "probe_poles.json"}
    assert not [n for n in names if "d1-writer" in n] and "requests/d1-explainer.jsonl" in names
    assert {r["role"] for r in llm.read_jsonl(fork / "calls.jsonl")} == {"explainer", "second_reader"}
    m = NV.read_json(fork / "manifest.json")
    assert [b["batch"] for b in m["batches"]] == ["d1-explainer", "d1-second_reader"] and "shuffle" not in m["seeds"]
    assert m["forked_from"]["dropped_batches"] == ["d1-writer", "d1-writer_job"]
    assert (fork / "explanations.jsonl").read_bytes() == (explain / "explanations.jsonl").read_bytes()
    NE.main(["prepare-writer", "--out", str(dst), "--batch", "d1", "--model", "other-writer"])
    assert {r["model"] for r in llm.read_jsonl(llm.requests_path(fork, "d1-writer"))} == {"other-writer"}
    F.answer_batch(fork, "d1-writer", _writer, model="other-writer")
    F.answer_batch(fork, "d1-writer_job", lambda r: {"positives": ["rain on the town"] * 5}, model="other-writer")
    NE.main(["collect-writer", "--out", str(dst), "--batch", "d1", "--backend", "manual", "--harvest", str(harvest),
             "--corpus", str(corpus), "--seed", "0"])
    assert len(llm.read_jsonl(fork / "texts.jsonl")) == len(llm.read_jsonl(explain / "texts.jsonl"))
    assert NV.read_json(fork / "shuffle.json")["pairs"] == NV.read_json(explain / "shuffle.json")["pairs"]
    with pytest.raises(FileExistsError):
        NE.main(["fork-writer", "--source", str(src), "--out", str(dst)])


def _writer_endpoint(filtered: dict, asked: list):
    """A stand-in for the API on writer and writer_job requests: the explanation named in `filtered`
    gets finish_reason content_filter for as many writer calls as its count says."""
    def urlopen(http, timeout):
        body = json.loads(http.data)
        role = body["response_format"]["json_schema"]["name"]
        user = body["messages"][1]["content"]
        asked.append(f"{role}:{user}")
        if role == "writer" and filtered.get(user, 0) > 0:
            filtered[user] -= 1
            return _Reply({"model": "test-model", "usage": {"prompt_tokens": 9, "completion_tokens": 0},
                           "choices": [{"finish_reason": "content_filter", "message": {"content": None}}]})
        x = _writer({}) if role == "writer" else {"positives": ["rain on the town"] * 5}
        return _Reply({"model": "test-model", "usage": {"prompt_tokens": 9, "completion_tokens": 40},
                       "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(x)}}]})
    return urlopen


def _writer_dir(bank, tmp_path):
    explain = _prepare_explain(bank, tmp_path)
    F.answer_batch(explain, "d1-explainer", _explanation)
    F.answer_batch(explain, "d1-second_reader", lambda r: {**_explanation(r), "confidence": 0.3})
    NE.main(["collect", "--out", str(tmp_path), "--batch", "d1", "--roles", "explainer,second_reader", "--backend", "manual"])
    NE.main(["prepare-writer", "--out", str(tmp_path), "--batch", "d1", "--model", "test-model"])
    return explain


def _writer_round(explain, monkeypatch, filtered, asked):
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", _writer_endpoint(filtered, asked))
    settings = {"backend": "openai_compat", "base_url": "https://api.example/v1", "key_env": "NEURONS_API_KEY_TEST",
                "models": {r: "test-model" for r in llm.ROLES}, "temperature": 1.0, "response_format": "json_schema"}
    for batch in ("d1-writer", "d1-writer_job"):
        llm.run_batch(explain, batch, settings, concurrency=4)


RAIN = "Explanation: Fires on rain and on bread."                      # the writer request of L00N1-


def test_a_content_filter_once_is_rejected_and_asked_again(bank, tmp_path, monkeypatch):
    explain = _writer_dir(bank, tmp_path)
    filtered, asked = {RAIN: 1}, []
    _writer_round(explain, monkeypatch, filtered, asked)               # the filter does not stop the run
    assert {u["finish_reason"] for u in llm.read_jsonl(llm.usage_path(explain, "d1-writer"))} == {"stop", "content_filter"}
    args = ["--out", str(tmp_path), "--batch", "d1", "--backend", "openai_compat"]
    with pytest.raises(ValueError, match=r"writer:L00N1-:0: text is not JSON"):
        NE.main(["collect-writer", *args, "--harvest", str(bank[1]), "--corpus", str(bank[0]), "--seed", "0"])
    NE.main(["reject", *args, "--roles", "writer,writer_job", "--harvest", str(bank[1]), "--corpus", str(bank[0])])
    asked.clear()
    _writer_round(explain, monkeypatch, filtered, asked)
    assert asked == [f"writer:{RAIN}"]                                 # only the filtered id is asked again
    NE.main(["collect-writer", *args, "--harvest", str(bank[1]), "--corpus", str(bank[0]), "--seed", "0"])
    assert "L00N1-" in {t["pole"] for t in llm.read_jsonl(explain / "texts.jsonl")}
    assert not llm.declined(explain, "d1-writer")


def test_a_content_filter_three_times_is_declined_and_the_pole_is_unk_n(bank, tmp_path, monkeypatch, capsys):
    corpus, harvest = bank
    explain = _writer_dir(bank, tmp_path)
    filtered, asked = {RAIN: 3}, []
    args = ["--out", str(tmp_path), "--batch", "d1", "--backend", "openai_compat"]
    for _ in range(3):
        _writer_round(explain, monkeypatch, filtered, asked)
        NE.main(["reject", *args, "--roles", "writer,writer_job", "--harvest", str(bank[1]), "--corpus", str(bank[0])])
    gave_up = llm.declined(explain, "d1-writer")
    assert list(gave_up) == ["writer:L00N1-:0"] and gave_up["writer:L00N1-:0"]["finish_reason"] == "content_filter"
    assert len(llm.read_jsonl(llm.rejected_path(explain, "d1-writer"))) == 3
    asked.clear()
    _writer_round(explain, monkeypatch, filtered, asked)
    assert asked == []                                                 # run does not ask a declined id
    NE.main(["collect-writer", *args, "--harvest", str(harvest), "--corpus", str(corpus), "--seed", "0"])
    assert "declined: writer:L00N1-:0: writer declined (provider content filter, 3 attempts)" in capsys.readouterr().out
    texts = llm.read_jsonl(explain / "texts.jsonl")
    text_poles = {t["pole"] for t in texts}
    assert "L00N1-" not in text_poles and text_poles == {"L00N0+", "L01N2+", "L01N3-"}
    pairs = NV.read_json(explain / "shuffle.json")["pairs"]
    assert set(pairs) == set(POLES)                                    # drawn over every writer-step pole
    assert NV.read_json(explain / "probe_poles.json") == sorted(text_poles | {pairs[p] for p in text_poles})
    probe = F.write_probe(tmp_path / "probe", texts, NV.read_json(explain / "probe_poles.json"),
                          lambda t, p: (0.5, -0.5))
    recs = {r["pole"]: r for r in NV.q1_records(harvest, probe, explain, 0.3)}
    rain = recs["L00N1-"]
    assert (rain["class"], rain["unk_part"]) == ("UNK", "UNK-N") and "texts" not in rain
    assert rain["unk_reason"] == "writer declined (provider content filter, 3 attempts)"
    q1 = NV.q1_metrics(list(recs.values()), 0.3)
    assert q1["declined"]["n"] == 1 and q1["fractions"]["n"] == 4 and q1["fractions"]["UNK-N"]["k"] >= 1



def test_a_writer_text_of_one_token_is_refused_and_rejected(bank, tmp_path):
    """The probe excludes each text's first token, so a text needs two tokens under the scored model's
    tokenizer; a one-character text is refused by collect-writer, naming its text_id, and moved by reject."""
    corpus, harvest = bank
    explain = _writer_dir(bank, tmp_path)
    F.answer_batch(explain, "d1-writer",
                   lambda r: {**_writer(r), "positives": ["x"] + _writer(r)["positives"][1:]} if r["id"] == "writer:L00N0+:0"
                   else _writer(r))
    F.answer_batch(explain, "d1-writer_job", lambda r: {"positives": ["rain on the town"] * 5})
    args = ["--out", str(tmp_path), "--batch", "d1", "--backend", "manual"]
    with pytest.raises(ValueError, match=r"writer:L00N0\+:0: text L00N0\+:pos:0 has 1 token\(s\) under the scored model's tokenizer"):
        NE.main(["collect-writer", *args, "--harvest", str(harvest), "--corpus", str(corpus), "--seed", "0"])
    with pytest.raises(ValueError, match="give --harvest and --corpus"):
        NE.main(["reject", *args, "--roles", "writer,writer_job"])
    NE.main(["reject", *args, "--roles", "writer,writer_job", "--harvest", str(harvest), "--corpus", str(corpus)])
    assert [r["id"] for r in llm.read_jsonl(llm.rejected_path(explain, "d1-writer"))] == ["writer:L00N0+:0"]



def test_prepare_never_writes_a_request_naming_opus(bank, tmp_path):
    corpus, harvest = bank
    (tmp_path / "poles.json").write_text(json.dumps(POLES))
    base = ["prepare", "--harvest", str(harvest), "--corpus", str(corpus), "--batch", "d9",
            "--roles", "explainer,second_reader", "--poles", str(tmp_path / "poles.json")]
    with pytest.raises(ValueError, match=r"d9: --model: model 'anthropic/claude-opus-5-5': Opus is not called through the API"):
        NE.main([*base, "--out", str(tmp_path / "opus"), "--model", "anthropic/claude-opus-5-5"])
    assert not (tmp_path / "opus").exists()                            # nothing written, not even the manifest
    NE.main([*base, "--out", str(tmp_path / "sonnet"), "--model", "anthropic/claude-sonnet-5-5"])
    explain = tmp_path / "sonnet" / "explain"
    assert {r["model"] for r in llm.read_jsonl(llm.requests_path(explain, "d9-explainer"))} == {"anthropic/claude-sonnet-5-5"}
    F.answer_batch(explain, "d9-explainer", _explanation, model="anthropic/claude-sonnet-5-5")
    F.answer_batch(explain, "d9-second_reader", lambda r: {**_explanation(r), "confidence": 0.3}, model="anthropic/claude-sonnet-5-5")
    NE.main(["collect", "--out", str(tmp_path / "sonnet"), "--batch", "d9", "--roles", "explainer,second_reader",
             "--backend", "manual"])
    with pytest.raises(ValueError, match="Opus is not called through the API"):
        NE.main(["prepare-writer", "--out", str(tmp_path / "sonnet"), "--batch", "d9", "--model", "anthropic/claude-opus-5-5"])
    assert not llm.requests_path(explain, "d9-writer").exists()


def test_a_c1_batch_with_opus_answers_resumes_on_other_models(bank, tmp_path, monkeypatch, capsys):
    """SPEC_D1 §2.2 model policy amendment: banked Opus explainer answers are kept; the rest of the
    explainer runs on model X, the second reader on Flash, the writer on X; nothing answered is sent
    again, no Opus id is sent, and each pole's record names the model that answered it."""
    corpus, harvest = bank
    explain = _prepare_explain(bank, tmp_path, batch="c1")
    opus, x, flash = "anthropic/claude-opus-5-5", "vendor/model-x", "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    path = llm.requests_path(explain, "c1-explainer")                  # the legacy state: requests written for Opus
    reqs = [{**r, "model": opus} for r in llm.read_jsonl(path)]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
    answered = [r for r in reqs if r["id"] in ("explainer:L00N0+:0", "explainer:L00N1-:0")]
    llm.append_jsonl(llm.responses_path(explain, "c1-explainer"), [
        {"id": r["id"], "model": opus, "text": json.dumps(_explanation(r)), "utc": "2026-10-20T16:00:00Z"} for r in answered])
    opus_settings = {"backend": "openai_compat", "base_url": "https://api.example/v1", "key_env": "NEURONS_API_KEY_TEST",
                     "models": {r: opus for r in llm.ROLES}, "temperature": 1.0, "response_format": "json_schema"}
    llm.backend_sidecar_path(explain, "c1-explainer").write_text(json.dumps(
        {"backend": "openai_compat", "settings": opus_settings, "reported_models": [opus], "utc": "2026-10-20T16:00:00Z"}))
    out = ["--out", str(tmp_path), "--batch", "c1"]
    with pytest.raises(ValueError, match="2 lines are answered"):
        NE.main(["remodel", *out, "--roles", "explainer", "--model", x])
    with pytest.raises(ValueError, match="Opus is not called through the API"):
        NE.main(["remodel", *out, "--roles", "explainer", "--model", opus, "--unanswered-only"])
    NE.main(["remodel", *out, "--roles", "explainer", "--model", x, "--unanswered-only"])
    NE.main(["remodel", *out, "--roles", "second_reader", "--model", flash])
    printed = capsys.readouterr().out
    assert f"c1-explainer: 2 lines set to {x} (from {{'{opus}': 2}}); 2 answered lines untouched" in printed
    assert "c1-second_reader: 4 lines set to" in printed and "0 answered lines untouched" in printed
    m = NV.read_json(explain / "manifest.json")
    assert [(r["batch"], r["changed"], r["answered_untouched"]) for r in m["remodels"]] == [("c1-explainer", 2, 2),
                                                                                          ("c1-second_reader", 4, 0)]
    sent = []

    def urlopen(http, timeout):
        body = json.loads(http.data)
        role, user = body["response_format"]["json_schema"]["name"], body["messages"][1]["content"]
        sent.append((role, body["model"]))
        if role in ("explainer", "second_reader"):
            pole = re.search(r"^Pole (\S+):", user, re.M).group(1)
            x_ = {**_explanation({"id": f"{role}:{pole}:0", "role": role}), "confidence": 0.3 if role == "second_reader" else 0.5}
        else:
            x_ = _writer({}) if role == "writer" else {"positives": ["rain on the town"] * 5}
        return _Reply({"model": body["model"], "usage": {"prompt_tokens": 9, "completion_tokens": 9},
                       "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(x_)}}]})
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    mixed = {**opus_settings, "models": {"explainer": x, "writer": x, "writer_job": x,
                                         "second_reader": flash, "clusterer": flash, "judge": flash}}
    settings_file = tmp_path / "mixed.json"
    settings_file.write_text(json.dumps(mixed))
    llm.main(["run-many", "--explain", str(explain), "--batches", "c1-explainer,c1-second_reader",
              "--settings", str(settings_file)])
    assert sorted(sent) == sorted([("explainer", x)] * 2 + [("second_reader", flash)] * 4)   # no Opus, nothing re-sent
    side = NV.read_json(llm.backend_sidecar_path(explain, "c1-explainer"))
    assert side["settings"]["models"]["explainer"] == opus and side["later_settings"][0]["settings"] == mixed
    assert side["reported_models"] == sorted([opus, x])
    NE.main(["collect", *out, "--roles", "explainer,second_reader", "--backend", "openai_compat"])
    rows = {(r["pole"], r["sample"]): r["model"] for r in llm.read_jsonl(explain / "explanations.jsonl")}
    assert rows[("L00N0+", 0)] == opus and rows[("L01N2+", 0)] == x and rows[("L00N0+", 1)] == flash
    m = NV.read_json(explain / "manifest.json")["models_by_role"]
    assert m["answered"]["explainer"] == {opus: 2, x: 2} and m["answered"]["second_reader"] == {flash: 4}
    NE.main(["prepare-writer", *out, "--model", x])
    sent.clear()
    llm.main(["run-many", "--explain", str(explain), "--batches", "c1-writer,c1-writer_job", "--settings", str(settings_file)])
    assert {m_ for _, m_ in sent} == {x}
    NE.main(["collect-writer", *out, "--backend", "openai_compat", "--harvest", str(harvest), "--corpus", str(corpus),
             "--seed", "0"])
    texts = llm.read_jsonl(explain / "texts.jsonl")
    assert {t["model"] for t in texts} == {x}
    probe = F.write_probe(tmp_path / "probe", texts, NV.read_json(explain / "probe_poles.json"), lambda t, p: (0.5, -0.5))
    recs = NV.q1_records(harvest, probe, explain, 0.3)
    q1 = json.loads(json.dumps(NV.q1_metrics(recs, 0.3)))
    by = q1["by_explainer_model"]
    assert set(by) == {opus, x} and by[opus]["n"] == by[x]["n"] == 2
    assert by[opus]["fractions"]["n"] + by[x]["fractions"]["n"] == q1["fractions"]["n"] == 4
    assert q1["kappa_models"] == {"explainer": {opus: 2, x: 2}, "second_reader": {flash: 4}, "cross_model_pairs": 4}
