"""Synthetic banked files that follow DATA_CONTRACT §1-§2, for the neurons_* tests. No downloads:
the tokenizer is a word-level `tokenizers` vocabulary saved with `save_pretrained`. The vocabulary
has no digits, so any digit in a judge prompt would come from the pipeline."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from infra.pg import neurons_llm as llm
from infra.pg.neurons_verdict import GRID_STEPS, file_sha256, parse_pole, pole_id

WORDS = ("the cat sat on mat dog ran to park red blue green boat sea ship sail wind rain cloud sun "
         "tree leaf bird song river stone hill road town bread").split()
EOS = "<eos>"
TOKNAME = "toy"
K = 50


def build_corpus(root: Path, n_docs: int = 400, seed: int = 0, long_doc: int = 700,
                 extra_words: tuple[str, ...] = ()) -> Path:
    """Random word documents of 30-120 tokens, then one document of `long_doc` tokens (so a 256-token
    context can be told from a 64-token one). `extra_words` join the vocabulary but never appear in
    the random text; a test plants them."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    base_vocab = ["<unk>", EOS, *WORDS]
    vocab = {w: i for i, w in enumerate(base_vocab + list(extra_words))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    base = root / "corpus" / "tok" / TOKNAME
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token=EOS).save_pretrained(
        str(base / "tokenizer"))
    rng = np.random.default_rng(seed)
    lengths = [int(rng.integers(30, 120)) for _ in range(n_docs)] + [long_doc]
    docs = [list(rng.integers(2, len(base_vocab), size=n)) + [vocab[EOS]] for n in lengths]
    np.save(base / "tokens.npy", np.concatenate(docs).astype(np.int32))
    np.save(base / "doc_offsets.npy", np.cumsum([0] + [len(d) for d in docs]).astype(np.int64))
    (root / "corpus" / "manifest.json").write_text(json.dumps({"n_docs": n_docs, "seed": seed, "long_doc": long_doc,
                                                                "extra_words": list(extra_words)}))
    return root / "corpus"


def silu(x):
    return x / (1 + np.exp(-x))


def build_harvest(out: Path, corpus: Path, *, n_layers: int = 2, n_neurons: int = 6, seed: int = 0,
                  branch: str = "main", started_utc: str = "2026-10-20T00:00:00Z",
                  poles: list[str] = (), dead: tuple[str, ...] = ()) -> Path:
    """A harvest out/: manifest.json, topk/, stats/, poles/poles.json and poles/vocab.json for `poles`."""
    n_tok = len(np.load(corpus / "tok" / TOKNAME / "tokens.npy"))
    rng = np.random.default_rng(seed)
    for layer in range(n_layers):
        tok = np.full((n_neurons, 2, K), -1, dtype=np.int64)
        a, g, u = (np.full((n_neurons, 2, K), np.nan, dtype=np.float32) for _ in range(3))
        for n in range(n_neurons):
            for s in (0, 1):
                if pole_id(layer, n, s) in dead:
                    continue
                tok[n, s] = rng.choice(n_tok, size=K, replace=False)
                mag = np.sort(rng.uniform(0.5, 10.0, size=K))[::-1]
                gg = rng.normal(1.0, 1.0, size=K)
                gg[np.abs(silu(gg)) < 0.05] = 1.0
                a[n, s] = mag if s == 0 else -mag
                g[n, s], u[n, s] = gg, a[n, s] / silu(gg)
        max_abs = np.nan_to_num(np.abs(a[:, :, 0]))
        valid = tok >= 0
        sign_counts = np.stack([(valid & (g >= 0) & (u >= 0)).sum(-1), (valid & (g >= 0) & (u < 0)).sum(-1),
                                (valid & (g < 0) & (u >= 0)).sum(-1), (valid & (g < 0) & (u < 0)).sum(-1)],
                               axis=-1).astype(np.int16)
        density = np.where(max_abs > 0, 500, 0).astype(np.int64)
        (out / "topk").mkdir(parents=True, exist_ok=True)
        (out / "stats").mkdir(parents=True, exist_ok=True)
        np.savez(out / "topk" / f"layer_{layer:02d}.npz", tok=tok, a=a, g=g, u=u)
        np.savez(out / "stats" / f"layer_{layer:02d}.npz", max_abs=max_abs.astype(np.float32),
                 hist=np.zeros((n_neurons, 2, 400), dtype=np.int32),
                 hist_edges=np.logspace(-6, 4, 401).astype(np.float32),
                 n_pole_tokens=np.full((n_neurons, 2), 5000, dtype=np.int64),
                 q99=(0.05 * max_abs).astype(np.float32), q999=(0.2 * max_abs).astype(np.float32),
                 q9999=(0.5 * max_abs).astype(np.float32), density=density,
                 top_same_token_frac=np.full((n_neurons, 2), 0.1, dtype=np.float32), sign_counts=sign_counts,
                 pos0_share=(np.arange(2 * n_neurons, dtype=np.float32).reshape(n_neurons, 2) / 100))
    (out / "manifest.json").write_text(json.dumps({
        "model_key": "TOY", "repo": "toy/none", "revision": "0" * 40, "branch": branch, "tokname": TOKNAME,
        "tokens_sha256": "f" * 64, "corpus_manifest_sha256": file_sha256(corpus / "manifest.json"),
        "n_layers": n_layers, "width": n_neurons, "hidden": 8, "n_tokens_scored": n_tok,
        "started_utc": started_utc, "finished_utc": started_utc, "topk_K": K}))
    (out / "poles").mkdir(exist_ok=True)
    (out / "poles" / "poles.json").write_text(json.dumps(list(poles)))
    plus = {"promoted": [[w, round(1.0 - 0.05 * i, 3)] for i, w in enumerate(WORDS[:10])],
            "suppressed": [[w, round(-0.5 + 0.02 * i, 3)] for i, w in enumerate(WORDS[10:20])]}
    minus = {"promoted": [[w, -s] for w, s in plus["suppressed"]], "suppressed": [[w, -s] for w, s in plus["promoted"]]}
    # per pole, the "-" entry the flip of the "+" entry, as the harvest writes it (JOURNAL 2026-10-25, F41); both poles of
    # every listed neuron, as in the real files, because pole_vocab checks a "-" entry against its "+" entry
    neurons = sorted({parse_pole(p)[:2] for p in poles})
    (out / "poles" / "vocab.json").write_text(json.dumps({
        pole_id(layer, n, s): (plus, minus)[s] for layer, n in neurons for s in (0, 1)}))
    return out


def set_stats(harvest: Path, layer: int, **fields) -> None:
    """Overwrite stats fields (arrays keyed like the file) in place."""
    path = harvest / "stats" / f"layer_{layer:02d}.npz"
    with np.load(path) as z:
        st = {k: z[k] for k in z.files}
    st.update(fields)
    np.savez(path, **st)


PROBE_RULE = ["score text positions >= 1", "BOS prepended, BOS and text position 0 excluded"]


def write_probe(out: Path, texts: list[dict], poles: list[str], fn, started_utc="2026-10-21T00:00:00Z",
                rule: bool = True) -> Path:
    """A probe-scoring harvest out/: manifest, poles/poles.json and poles/probe/scores.npz with
    max_a/min_a from fn(text_row, pole) -> (max_a, min_a)."""
    vals = np.array([[fn(t, p) for p in poles] for t in texts], dtype=np.float32)   # [T, P, 2]
    (out / "poles" / "probe").mkdir(parents=True, exist_ok=True)
    np.savez(out / "poles" / "probe" / "scores.npz", text_ids=np.array([t["text_id"] for t in texts]),
             max_a=vals[:, :, 0], min_a=vals[:, :, 1],
             argmax_tok=np.zeros(vals.shape[:2], np.int32), argmin_tok=np.zeros(vals.shape[:2], np.int32),
             n_tokens=np.full(len(texts), 12, np.int32), **({"probe_rule": np.array(PROBE_RULE)} if rule else {}))
    (out / "poles" / "poles.json").write_text(json.dumps(poles))
    (out / "manifest.json").write_text(json.dumps({"started_utc": started_utc, "branch": "main", "revision": "0" * 40,
                                                   **({"probe_excluded_positions": "text position 0",
                                                       "probe_rule": PROBE_RULE} if rule else {})}))
    return out


def answer_batch(explain: Path, batch: str, fn, model: str = "test-model") -> None:
    """Act as the manual backend: answer every request of `batch` with fn(request) -> object."""
    llm.write_jsonl_new(llm.responses_path(explain, batch), [
        {"id": r["id"], "model": model, "text": json.dumps(fn(r)), "utc": "2026-10-21T00:00:00Z"}
        for r in llm.read_jsonl(llm.requests_path(explain, batch))])


def build_grid(root: Path, corpus: Path, S: list[str], design, grid_poles: list[str] | None = None) -> list[Path]:
    """16 harvest out/ dirs T_0..T_15 (one layer, six neurons) with poles/exemplar_a.npz.
    design(t, pole) -> (magnitudes [50] in the pole's direction, q999, max_abs) at grid index t.
    `grid_poles` (default S) is the harvest's --poles list: poles/poles.json and the exemplar rows
    follow its order, so it may be a superset of S in any order."""
    grid_poles = list(S) if grid_poles is None else list(grid_poles)
    dirs = []
    for t, step in enumerate(GRID_STEPS):
        h = build_harvest(root / f"T{t:02d}", corpus, n_layers=1, seed=100 + t,
                          branch=f"stage1-step{step}", poles=grid_poles)
        path = h / "stats" / "layer_00.npz"
        with np.load(path) as z:
            st = {k: z[k] for k in z.files}
        a = np.zeros((len(grid_poles), K), dtype=np.float32)
        for i, pole in enumerate(grid_poles):
            _, n, s = parse_pole(pole)
            mag, q999, max_abs = design(t, pole)
            a[i] = mag if s == 0 else -np.asarray(mag)
            st["q999"][n, s], st["max_abs"][n, s] = q999, max_abs
        np.savez(path, **st)
        np.savez(h / "poles" / "exemplar_a.npz", a=a, g=np.ones_like(a), u=a)
        dirs.append(h)
    end_tok = np.load(dirs[-1] / "topk" / "layer_00.npz")["tok"]          # T_15 writes its own positions
    np.savez(dirs[-1] / "poles" / "positions.npz",
             tok=np.stack([end_tok[parse_pole(p)[1], parse_pole(p)[2], :K] for p in grid_poles]))
    return dirs


READOUT_LINES = ["- p_u: upper bound below 0.10", "- b: above the permutation band", "- v: low",
                 "- H_MIG: about zero", "- hard-decoy: far above 0.5"]


def git_journal(root: Path, heading: str, body: list[str] = READOUT_LINES,
                commit_date: str | None = "2026-10-01T00:00:00+00:00") -> Path:
    """`<root>/studies/already-clean/JOURNAL.md` in a git repository at `root`, committed at
    `commit_date` (committer and author date) unless it is None."""
    path = root / "studies" / "already-clean" / "JOURNAL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["# JOURNAL", "", heading, *body, "", "## Later notes", "text", ""]))
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    if commit_date is not None:
        env = {**os.environ, "GIT_COMMITTER_DATE": commit_date, "GIT_AUTHOR_DATE": commit_date}
        cfg = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=/dev/null"]
        subprocess.run(["git", "-C", str(root), "add", str(path)], check=True)
        subprocess.run(["git", "-C", str(root), *cfg, "commit", "-q", "-m", "journal"], check=True, env=env)
    return path


def plans_by_id(explain: Path, name: str) -> dict:
    return {p["id"]: p for p in llm.read_jsonl(explain / "plans" / f"{name}.jsonl")}


def two_halves(plan: dict) -> dict:
    """A clustering answer: ranks 0-24 and 25-49, with the double-negative ranks in an echo cluster."""
    echo = set(plan["double_negative_ranks"])
    groups = [[r for r in range(25) if r not in echo], [r for r in range(25, 50) if r not in echo]]
    clusters = [{"c": i, "name": f"half {i}", "echo": False, "exemplar_ranks": g} for i, g in enumerate(groups)]
    if echo:
        clusters.append({"c": 2, "name": "echo", "echo": True, "exemplar_ranks": sorted(echo)})
    m = len(clusters)
    return {"clusters": clusters,
            "pairs": [{"a": i, "b": j, "relation": "related"} for i in range(m) for j in range(i + 1, m)]}
