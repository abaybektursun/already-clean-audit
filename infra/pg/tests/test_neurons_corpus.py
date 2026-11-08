"""Offline gate for the D1 corpus: band assignment and the streamed shard read, on synthetic shards."""
import gzip
import hashlib
import io
import json

import numpy as np
import pytest
import zstandard

from infra.pg import neurons_corpus as NC


def encode(text):
    return [hash(w) % 50_000 for w in text.split()]


def shard(seed, n_docs, words=(5, 400)):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(20_000)]
    return [json.dumps({"id": f"{seed}-{i}", "text": " ".join(rng.choice(vocab, size=int(rng.integers(*words))))})
            for i in range(n_docs)]


def test_bands_are_nested_prefixes():
    T = 6_000
    shards = [shard(s, 400) for s in range(3)]
    per_shard = [NC.select_docs(lines, encode, T, set())[0] for lines in shards]
    ordered = NC.order_by_band(per_shard)
    bands = NC.band_bounds([b for _, b, _, _ in ordered])
    tokens, offsets = NC.token_stream([x[3] for x in ordered], eos_id=99_999)
    for b, frac in ((1, 0.2), (2, 0.4)):
        # the band-b block = what a standalone selection with budget frac*T takes from every shard, in shard order
        alone = [NC.select_docs(lines, encode, int(frac * T), set())[0] for lines in shards]
        want = [text for rows in alone for _, text, _ in rows]
        block = [x[2] for x in ordered[:bands[b][1]]]
        assert sorted(block) == sorted(want) and len(block) == len(want)
        # it is a prefix of the full token stream
        pre, _ = NC.token_stream([x[3] for x in ordered[:bands[b][1]]], eos_id=99_999)
        assert np.array_equal(tokens[:offsets[bands[b][1]]], pre)
    for rows in per_shard:
        count = 0
        for band, _, ids in rows:
            assert band == NC.band_of(count, T)
            count += len(ids) + 1
        assert count - (len(rows[-1][2]) + 1) < T <= count          # budget exceeded by at most one document
    assert [b for _, b, _, _ in ordered] == sorted(b for _, b, _, _ in ordered)


def compress(lines, path):
    data = ("\n".join(lines) + "\n").encode()
    return zstandard.ZstdCompressor().compress(data) if path.endswith(".zst") else gzip.compress(data)


@pytest.mark.parametrize("path", ["d/shard_0.jsonl.zst", "d/part-0.jsonl.gz"])
def test_streamed_read_hashes_exactly_the_consumed_bytes(path):
    lines = shard(7, 20_000, words=(20, 80))
    raw = compress(lines, path)
    docs, recs, skipped = NC.read_shard(lambda p: io.BytesIO(raw), [path, "d/never_opened.jsonl.zst"], encode, 3_000,
                                        set())
    assert (docs, skipped) == NC.select_docs(iter(lines), encode, 3_000, set())
    assert len(recs) == 1 and recs[0]["path"] == path
    n = recs[0]["bytes_consumed"]
    assert 0 < n < len(raw), (n, len(raw))                          # stopped early, next file never opened
    assert recs[0]["sha256_of_consumed_bytes"] == hashlib.sha256(raw[:n]).hexdigest()


def test_short_first_file_continues_into_the_next():
    a, b = shard(1, 30), shard(2, 2_000)
    files = {"d/a-0000.jsonl.gz": compress(a, ".gz"), "d/shard_0.jsonl.zst": compress(b, ".zst")}
    paths = sorted(files)
    budget = sum(len(encode(json.loads(x)["text"])) + 1 for x in a) + 500
    docs, recs, _ = NC.read_shard(lambda p: io.BytesIO(files[p]), paths, encode, budget, set())
    assert docs == NC.select_docs(iter(a + b), encode, budget, set())[0]
    assert [r["path"] for r in recs] == paths
    assert recs[0]["bytes_consumed"] == len(files[paths[0]])         # the short file was read to its end
    assert recs[0]["sha256_of_consumed_bytes"] == hashlib.sha256(files[paths[0]]).hexdigest()
    n = recs[1]["bytes_consumed"]
    assert 0 < n <= len(files[paths[1]]) and recs[1]["sha256_of_consumed_bytes"] == \
        hashlib.sha256(files[paths[1]][:n]).hexdigest()


def test_duplicates_are_skipped_counted_and_take_no_budget():
    base = shard(3, 300)
    lines = [x for i, x in enumerate(base) for _ in range(1 + i % 3)]       # 1, 2 or 3 copies, adjacent
    budget = 4_000
    seen = set()
    docs, skipped = NC.select_docs(iter(lines), encode, budget, seen)
    want, _ = NC.select_docs(iter(base), encode, budget, set())             # the same stream without copies
    assert [t for _, t, _ in docs] == [t for _, t, _ in want]
    texts = [json.loads(x)["text"] for x in lines]
    n_read = texts.index(docs[-1][1]) + 1
    assert skipped == n_read - len(docs) > 0
    # a later directory skips copies of documents an earlier one took
    docs2, skipped2 = NC.select_docs(iter(base[:5] + shard(4, 300)), encode, budget, seen)
    assert skipped2 == 5 and all(t not in {x for _, x, _ in docs} for _, t, _ in docs2)


def test_pick_dirs_spreads_over_the_sorted_list():
    dirs = [f"data/d{i:03d}" for i in range(140)]
    got = NC.pick_dirs(list(reversed(dirs)))
    assert len(got) == 50 == len(set(got)) and got[0] == dirs[0] and got[-1] == dirs[137]
    assert NC.pick_dirs(dirs[:100]) == dirs[:100:2]
