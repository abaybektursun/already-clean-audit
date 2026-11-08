"""D1 corpus (N2): SPEC_D1 §4.2 (W0 amendments) and DATA_CONTRACT §1.

Directories. `allenai/dolma3_mix-6T-1025-7B` at one resolved commit; its `data/*` directories (165 at
`2ca900fb`) are sorted and the 50 at indices `floor(i * n_dirs / 50)`, i = 0..49, are taken (the old
`k = floor(n_dirs / 50)` rule left 55).

Shard files. A directory's shard files are its `.jsonl.zst` and `.jsonl.gz` files (the mix holds about
half of each; a name with digits after `.zst` is not a shard file) in lexicographic order. The directory
is read from its first file and, only when that file ends before the directory's budget, on into the next
(the PDF directories' first `.gz` files are 0.11-0.28 MB). Each file is read through `HfFileSystem` (HTTP
range reads) and a zstandard or gzip stream, and reading stops as soon as the budget is filled. The
SHA-256 of exactly the compressed bytes consumed is recorded per file, so the record can be checked
against the first `bytes_consumed` bytes of each file.

Deduplication. A document whose text has the SHA-256 of a document already read (in this directory or
an earlier one) is skipped before selection and counted (`n_duplicates_skipped` per directory); a skipped
copy takes no budget. W0 held 2,364 selected documents with 1,358 distinct texts.

Selection and bands. Selection uses the OLMo-3 tokenizer. A document's count is its token count plus one
EOS. Per directory ("shard" below), with `tokens_per_shard` = 1,000,000, a document belongs to band 1 if
the shard's running count at its start is below 20 %, band 2 below 40 %, band 3 below 100 %; reading
stops at the first document that would start at or past 100 %. Corpus order is all band-1 documents
(shard order, file order), then band 2, then band 3, so band 1 (10M), bands 1-2 (20M) and bands 1-3 (50M)
are prefixes of one token stream under every tokenizer.

Fail-loud: no try/except, no fallback. Hub and transformers imports are inside functions.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pathlib
import time

import numpy as np

DATASET = "allenai/dolma3_mix-6T-1025-7B"
N_DIRS = 50
BAND_FRACS = (0.2, 0.4)          # band 1 ends at 20 % of tokens_per_shard, band 2 at 40 %, band 3 at 100 %
SHARD_SUFFIXES = (".jsonl.zst", ".jsonl.gz")
TOKENIZERS = {                   # tokname -> repo; olmo3 comes from the Hub at a named revision,
    "olmo3": "allenai/Olmo-3-1025-7B",          # the other two from the instance's HF-cache snapshot
    "qwen3": "Qwen/Qwen3-4B-Base",
    "gemma3": "google/gemma-3-4b-pt",
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_snapshot(repo: str) -> tuple[str, str]:
    """(path, commit sha) of the single snapshot of `repo` in the local HF cache (what `pg run --weights`
    synced). Exactly one snapshot must exist."""
    hub = pathlib.Path(os.environ["HF_HUB_CACHE"])
    snaps = sorted((hub / ("models--" + repo.replace("/", "--")) / "snapshots").iterdir())
    assert len(snaps) == 1, f"{repo}: expected one cached snapshot, found {[s.name for s in snaps]}"
    return str(snaps[0]), snaps[0].name


def pick_dirs(dirs: list, n: int = N_DIRS) -> list:
    """`n` directories at evenly spaced indices of the sorted list."""
    dirs = sorted(dirs)
    assert len(dirs) >= n, f"only {len(dirs)} directories, need {n}"
    return [dirs[(i * len(dirs)) // n] for i in range(n)]


def band_of(start: int, tokens_per_shard: int) -> int:
    if start < int(BAND_FRACS[0] * tokens_per_shard):
        return 1
    if start < int(BAND_FRACS[1] * tokens_per_shard):
        return 2
    return 3


class HashingReader:
    """Binary reader that counts and SHA-256-hashes every byte read through it."""

    def __init__(self, f):
        self.f = f
        self.n = 0
        self.h = hashlib.sha256()

    def read(self, size=-1):
        b = self.f.read(size)
        self.n += len(b)
        self.h.update(b)
        return b

    def close(self):
        self.f.close()


def select_docs(lines, encode, tokens_per_shard: int, seen: set) -> tuple[list, int]:
    """([(band, text, olmo_ids)], n_duplicates_skipped) for one shard: documents in file order until the
    running count (tokens + one EOS per document) reaches `tokens_per_shard`, skipping any document whose
    text SHA-256 is in `seen` (which this call extends). Stops reading at that point."""
    out, count, skipped = [], 0, 0
    for line in lines:
        text = json.loads(line)["text"]
        assert isinstance(text, str), f"document without a text string: {line[:200]!r}"
        h = hashlib.sha256(text.encode("utf-8")).digest()
        if h in seen:
            skipped += 1
            continue
        seen.add(h)
        ids = encode(text)
        out.append((band_of(count, tokens_per_shard), text, ids))
        count += len(ids) + 1
        if count >= tokens_per_shard:
            return out, skipped
    raise AssertionError(f"shard ended at {count} tokens, before its budget {tokens_per_shard}")


def decompressed(src, path: str):
    import zstandard
    if path.endswith(".jsonl.zst"):
        return zstandard.ZstdDecompressor().stream_reader(src)
    assert path.endswith(".jsonl.gz"), f"not a shard file: {path}"
    return gzip.GzipFile(fileobj=src, mode="rb")


def read_shard(open_file, paths: list, encode, tokens_per_shard: int, seen: set) -> tuple[list, list, int]:
    """Select one directory's documents from its shard files in order, opening the next file only when
    the previous one ends before the budget. Returns (docs, [{path, bytes_consumed,
    sha256_of_consumed_bytes} per file opened], n_duplicates_skipped)."""
    opened = []

    def lines():
        for path in paths:
            src = HashingReader(open_file(path))
            opened.append((path, src))
            yield from io.TextIOWrapper(decompressed(src, path), encoding="utf-8")

    docs, skipped = select_docs(lines(), encode, tokens_per_shard, seen)
    recs = [{"path": p, "bytes_consumed": s.n, "sha256_of_consumed_bytes": s.h.hexdigest()} for p, s in opened]
    for _, s in opened:
        s.close()
    return docs, recs, skipped


def order_by_band(per_shard: list) -> list:
    """per_shard: [[(band, text, olmo_ids), ...] per shard] -> [(shard_index, band, text, olmo_ids)] in
    corpus order: band 1 of every shard, then band 2, then band 3."""
    return [(s, b, text, ids) for b in (1, 2, 3)
            for s, rows in enumerate(per_shard) for band, text, ids in rows if band == b]


def token_stream(id_lists: list, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Documents concatenated with EOS after each: (tokens int32, doc_offsets int64 of length n_docs+1)."""
    lens = np.array([len(x) + 1 for x in id_lists], dtype=np.int64)
    offsets = np.zeros(len(id_lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    tokens = np.empty(int(offsets[-1]), dtype=np.int32)
    for i, ids in enumerate(id_lists):
        tokens[offsets[i]:offsets[i + 1] - 1] = ids
        tokens[offsets[i + 1] - 1] = eos_id
    return tokens, offsets


def band_bounds(doc_bands: list) -> dict:
    """{band: (doc_start, doc_end)} for a corpus-ordered list of band labels."""
    out = {}
    for b in (1, 2, 3):
        idx = [i for i, x in enumerate(doc_bands) if x == b]
        assert idx and idx == list(range(idx[0], idx[-1] + 1)), f"band {b} is not one contiguous block"
        out[b] = (idx[0], idx[-1] + 1)
    assert out[1][1] == out[2][0] and out[2][1] == out[3][0] and out[3][1] == len(doc_bands)
    return out


def write_tok(out_dir, name: str, tok, id_lists: list) -> dict:
    """tok/<name>/{tokens.npy, doc_offsets.npy, tokenizer/}; returns the arrays and their record."""
    from infra.pg.matrix_models import vocab_sha256
    eos = tok.eos_token_id
    assert eos is not None, f"{name}: tokenizer has no EOS"
    tokens, offsets = token_stream(id_lists, eos)
    d = pathlib.Path(out_dir) / "tok" / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tokens.npy", tokens)
    np.save(d / "doc_offsets.npy", offsets)
    tok.save_pretrained(str(d / "tokenizer"))
    return {"offsets": offsets, "record": {
        "n_tokens": int(tokens.size), "sha256_of_tokens_npy": sha256_file(d / "tokens.npy"),
        "eos_id": int(eos), "bos_id": None if tok.bos_token_id is None else int(tok.bos_token_id),
        "vocab_sha256": vocab_sha256(tok)}}


def load_tok(corpus_dir, tokname: str) -> dict:
    """The harvest's view of a corpus: manifest, its sha256, tokens, doc_offsets, band token bounds."""
    corpus_dir = pathlib.Path(corpus_dir)
    man_path = corpus_dir / "manifest.json"
    man = json.loads(man_path.read_text())
    rec = man["tokenizers"][tokname]
    d = corpus_dir / "tok" / tokname
    got = sha256_file(d / "tokens.npy")
    assert got == rec["sha256_of_tokens_npy"], f"{tokname}: tokens.npy sha256 {got} != manifest"
    tokens = np.load(d / "tokens.npy")
    offsets = np.load(d / "doc_offsets.npy")
    assert tokens.size == rec["n_tokens"] == int(offsets[-1])
    return {"tokname": tokname, "manifest": man, "manifest_sha256": sha256_file(man_path), "tokens": tokens,
            "doc_offsets": offsets, "tokens_sha256": got, "record": rec,
            "band_end": {int(b): v["tok_end"][tokname] for b, v in man["bands"].items()},
            "tokenizer_dir": str(d / "tokenizer")}


def build(out_dir, tokens_per_shard: int, olmo_revision: str, n_dirs: int = N_DIRS) -> dict:
    """Run the whole N2 rule and write `out_dir` (the `corpus/` directory of DATA_CONTRACT §1)."""
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    from importlib.metadata import version
    from transformers import AutoTokenizer
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    commit = api.dataset_info(DATASET).sha
    dirs = [e.path for e in api.list_repo_tree(DATASET, path_in_repo="data", repo_type="dataset",
                                                revision=commit) if isinstance(e, RepoFolder)]
    chosen = pick_dirs(dirs, n_dirs)

    olmo_sha = api.model_info(TOKENIZERS["olmo3"], revision=olmo_revision).sha
    toks = {"olmo3": (AutoTokenizer.from_pretrained(TOKENIZERS["olmo3"], revision=olmo_sha),
                      {"repo": TOKENIZERS["olmo3"], "revision": olmo_sha, "branch": olmo_revision})}
    for name in ("qwen3", "gemma3"):
        path, sha = local_snapshot(TOKENIZERS[name])
        toks[name] = (AutoTokenizer.from_pretrained(path), {"repo": TOKENIZERS[name], "revision": sha})
    olmo = toks["olmo3"][0]

    def encode_olmo(text):
        return olmo(text, add_special_tokens=False)["input_ids"]

    fs = HfFileSystem()
    def open_file(path):
        return fs.open(f"datasets/{DATASET}@{commit}/{path}", "rb")

    per_shard, shard_recs, seen = [], [], set()
    t0 = time.time()
    for d in chosen:
        files = sorted((e for e in api.list_repo_tree(DATASET, path_in_repo=d, repo_type="dataset",
                                                      revision=commit)
                        if isinstance(e, RepoFile) and e.path.endswith(SHARD_SUFFIXES)),
                       key=lambda e: e.path)
        assert files, f"{d}: no {SHARD_SUFFIXES} shard file"
        docs, recs, skipped = read_shard(open_file, [f.path for f in files], encode_olmo, tokens_per_shard, seen)
        meta = {f.path: f for f in files}
        for r in recs:
            r.update(file_size=meta[r["path"]].size,
                     file_lfs_sha256=meta[r["path"]].lfs.sha256 if meta[r["path"]].lfs else None)
        per_shard.append(docs)
        shard_recs.append({"dir": d, "n_files_in_dir": len(files), "files": recs, "n_duplicates_skipped": skipped,
                           "n_docs_selected": {str(b): sum(1 for x in docs if x[0] == b) for b in (1, 2, 3)},
                           "olmo3_tokens_selected": sum(len(x[2]) + 1 for x in docs)})
        print(f"[corpus] {len(shard_recs)}/{len(chosen)} {d}: {len(docs)} docs ({skipped} copies skipped) "
              f"from {len(recs)} file(s), "
              f"{sum(r['bytes_consumed'] for r in recs)} bytes read", flush=True)
    read_s = time.time() - t0

    ordered = order_by_band(per_shard)
    bands = band_bounds([b for _, b, _, _ in ordered])
    with gzip.open(out_dir / "docs.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (s, b, text, _) in enumerate(ordered):
            f.write(json.dumps({"i": i, "shard": shard_recs[s]["dir"], "band": b, "text": text}) + "\n")

    tok_recs, offsets = {}, {}
    for name, (tok, rec) in toks.items():
        ids = ([x[3] for x in ordered] if name == "olmo3" else
               tok([x[2] for x in ordered], add_special_tokens=False)["input_ids"])
        w = write_tok(out_dir, name, tok, ids)
        tok_recs[name] = {**rec, **w["record"]}
        offsets[name] = w["offsets"]
    man = {
        "dataset_repo": DATASET, "dataset_commit": commit, "n_dirs_total": len(dirs),
        "k": len(dirs) / n_dirs,
        "dir_rule": f"sorted data/* directories at indices floor(i * n_dirs_total / {n_dirs}), i = 0..{n_dirs - 1}",
        "dirs": chosen, "shard_rule": f"files ending {SHARD_SUFFIXES} in lexicographic order; the next file "
                                    "is opened only when the previous one ends before the budget",
        "shards": shard_recs, "tokens_per_shard": tokens_per_shard,
        "dedup_rule": "a document whose text SHA-256 was already read (any earlier directory or file) is skipped "
                      "before selection and counted per directory; a copy takes no budget",
        "n_duplicates_skipped": sum(r["n_duplicates_skipped"] for r in shard_recs),
        "band_rule": "per shard, band of a document = where the running OLMo-3 count (tokens + 1 EOS per "
                     "document) stands at its start: < 20 % band 1, < 40 % band 2, else band 3; corpus "
                     "order = band 1 of every shard, then band 2, then band 3",
        "bands": {str(b): {"doc_start": s, "doc_end": e,
                           "tok_start": {n: int(offsets[n][s]) for n in toks},
                           "tok_end": {n: int(offsets[n][e]) for n in toks}} for b, (s, e) in bands.items()},
        "n_docs": len(ordered), "tokenizers": tok_recs,
        "zstandard_version": version("zstandard"), "read_seconds": round(read_s, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man
