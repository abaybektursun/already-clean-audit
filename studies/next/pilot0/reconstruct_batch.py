"""Reconstruct the training batch of allenai/Olmo-3-1025-7B stage-1 pretraining for given trainer steps.

Recipe: studies/next/00b_DATA_ORDER.md. No model is run.
Trainer step s (1-based, as the W&B log counts) trains the 0-based batch index k = s - 1 (recipe Step 0).

usage: python -I reconstruct_batch.py --in-dir DIR [--out-dir DIR] [--state-step A] STEP [STEP ...]

--state-step A reads the run config, file list and trainer state of the OLMo-core checkpoint at step A
(default 60000) and asserts its batches_processed equals A. Use one --in-dir per A: cached files keep fixed names.

Writes, per step s: OUT/batch_<s>/instances.jsonl, tokens.npy (uint32, 512 x 8192), meta.json.
Downloads (small inputs, raw range bytes) go to --in-dir. Every check is an assert: any mismatch stops the run.
"""
import argparse
import hashlib
import json
import os
import pickletools
import platform
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tokenizers import Tokenizer

MIX_URL = ("https://raw.githubusercontent.com/allenai/OLMo-core/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/"
           "src/olmo_core/data/mixes/OLMo-mix-0625-official.txt")
CKPT_BASE = "https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step"
TOK_URL = "https://huggingface.co/allenai/Olmo-3-1025-7B/resolve/stage1-step60000"
DATA_BASE = "https://olmo-data.org/"
TOKENIZER_ID = "allenai/dolma3-tokenizer"
UA = {"User-Agent": "pilot0-reconstruct/1.0"}  # the data host answers 403 to the default Python-urllib agent

SEQ, BATCH, ITEM = 8192, 512, 4  # tokens per instance, instances per batch, bytes per uint32 token
WINDOW = SEQ * ITEM  # 32,768 bytes per instance
N_EXPECTED, BATCHES_EXPECTED = 723_872_890, 1_413_814


def http(url, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.headers, r.read()


def cached(url, path):
    if not os.path.exists(path):
        status, _, body = http(url)
        assert status == 200, (url, status)
        with open(path + ".part", "wb") as f:
            f.write(body)
        os.replace(path + ".part", path)
    return path


def head_size(url):
    status, h, _ = http(url, method="HEAD")
    assert status == 200 and h["Accept-Ranges"] == "bytes", (url, status, dict(h))
    return int(h["Content-Length"])


def fetch_window(job):
    url, offset, file_size = job
    last = offset + WINDOW - 1
    status, h, body = http(url, headers={"Range": f"bytes={offset}-{last}"})
    assert status == 206, (url, offset, status)
    assert h["Content-Range"] == f"bytes {offset}-{last}/{file_size}", (url, offset, h["Content-Range"])
    assert len(body) == WINDOW, (url, offset, len(body))
    return body


def read_loader_state(pt_path):
    """The data_loader dict of a torch trainer-state file, read with pickletools. Nothing is unpickled."""
    z = zipfile.ZipFile(pt_path)
    (pkl,) = [n for n in z.namelist() if n.endswith("/data.pkl")]
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops(z.read(pkl))]
    i = ops.index(("BINUNICODE", "data_loader"))
    assert [o for o, _ in ops[i + 1:i + 5]] == ["BINPUT", "EMPTY_DICT", "BINPUT", "MARK"], ops[i + 1:i + 5]
    vals = []
    for name, arg in ops[i + 5:]:
        if name == "SETITEMS":
            break
        if name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
            continue
        assert name in ("BINUNICODE", "SHORT_BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1"), name
        vals.append(arg)
    assert len(vals) % 2 == 0, vals
    return dict(zip(vals[0::2], vals[1::2]))


def fingerprint(fields, basenames, sizes):
    """OLMo-core NumpyDatasetBase.fingerprint (numpy_dataset.py L296-308 at ccf4ef4b)."""
    h = hashlib.sha256()
    h.update(b"class=NumpyFSLDataset")
    for k, v in fields:
        h.update(f"{k}={v},".encode())
    for b, s in zip(basenames, sizes):
        h.update(f"path={b},size={s},".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int, nargs="+", help="trainer steps, 1-based as in the W&B log")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--state-step", type=int, default=60000, help="OLMo-core checkpoint step whose trainer state is read")
    args = ap.parse_args()
    CKPT_URL = f"{CKPT_BASE}{args.state_step}"
    t_start = time.time()
    os.makedirs(args.in_dir, exist_ok=True)
    print(f"python {platform.python_version()}  numpy {np.__version__}  (training pinned numpy<2.0)")

    # Run config, file list and trainer state of the stage-1 run (HF bucket, checkpoint step --state-step).
    cfg = json.load(open(cached(CKPT_URL + "/config.json", os.path.join(args.in_dir, "config.json"))))
    train_paths = open(cached(CKPT_URL + "/data_paths.txt", os.path.join(args.in_dir, "data_paths.txt"))).read().split()
    state = read_loader_state(cached(CKPT_URL + "/train/rank0.pt", os.path.join(args.in_dir, f"rank0_step{args.state_step}.pt")))
    print("trainer state data_loader:", state)
    ds, tk = cfg["dataset"], cfg["dataset"]["tokenizer"]
    assert ds["name"] == "fsl" and ds["sequence_length"] == SEQ and ds["max_target_sequence_length"] == SEQ, ds
    assert "instance_filter_config" not in ds and ds["generate_doc_lengths"] is False, ds
    assert cfg["data_loader"]["global_batch_size"] == SEQ * BATCH, cfg["data_loader"]
    assert state["dataset_type"] == "fsl" and state["sequence_length"] == SEQ, state
    assert state["seed"] == cfg["data_loader"]["seed"] == 34521 and state["epoch"] == 1, state
    assert state["batches_processed"] == state["tokens_processed"] // (SEQ * BATCH), state
    assert state["batches_processed"] == args.state_step, (state["batches_processed"], args.state_step)
    seed, epoch, eos = state["seed"], state["epoch"], tk["eos_token_id"]

    # Step 1: public file list, paired line by line with the training list.
    mix_path = cached(MIX_URL, os.path.join(args.in_dir, "OLMo-mix-0625-official.txt"))
    mix = [line.strip().split(",")[1] for line in open(mix_path, newline="") if line.strip()]
    print("official mix sha256:", hashlib.sha256(open(mix_path, "rb").read()).hexdigest())
    assert len(mix) == len(train_paths) == 1016, (len(mix), len(train_paths))
    rel = [p.split("{TOKENIZER}/", 1)[1] for p in mix]
    top_map = {}
    for r, t in zip(rel, train_paths):
        t_rel = t.split("/allenai/dolma2-tokenizer/", 1)[1]
        assert r.split("/", 1)[1] == t_rel.split("/", 1)[1], (r, t)
        assert top_map.setdefault(t_rel.split("/", 1)[0], r.split("/", 1)[0]) == r.split("/", 1)[0], (r, t)
    assert len(set(top_map.values())) == len(top_map), top_map
    print("directory map, training -> public:", top_map)
    urls = [DATA_BASE + p.replace("{TOKENIZER}", TOKENIZER_ID) for p in mix]

    # Step 2: sizes by HEAD, windows per file.
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(head_size, urls))
    t_head = time.time() - t0
    assert all(s % ITEM == 0 for s in sizes)
    n = np.array([s // WINDOW for s in sizes], dtype=np.int64)
    ends = np.cumsum(n)
    starts = ends - n
    N = int(ends[-1])
    print(f"HEAD on {len(urls)} files: {t_head:.1f} s; total bytes {sum(sizes):,}; instances N = {N:,}")
    assert N == N_EXPECTED and N // BATCH == BATCHES_EXPECTED, N

    # Step 3: fingerprint gate, with the recipe's two failing inputs.
    assert 2 ** 16 <= tk["vocab_size"] < 2 ** 32
    fields = [("vocab_size", tk["vocab_size"]), ("pad_token_id", tk["pad_token_id"]), ("eos_token_id", eos),
              ("dtype", str(np.uint32)), ("max_target_sequence_length", ds["max_target_sequence_length"]),
              ("bos_token_id", tk.get("bos_token_id"))]
    basenames = [os.path.basename(p) for p in train_paths]
    fp = fingerprint(fields, basenames, sizes)
    print("fingerprint recomputed from public sizes:", fp)
    print(f"fingerprint in trainer state step {args.state_step}: ", state["dataset_fingerprint"])
    assert fp == state["dataset_fingerprint"], "fingerprint mismatch"
    bumped = list(sizes)
    bumped[500] += 4
    assert fingerprint(fields, basenames, bumped) != fp
    assert fingerprint([(k, "uint32" if k == "dtype" else v) for k, v in fields], basenames, sizes) != fp

    # Step 6 resolver, checked on the recipe's two worked examples.
    def resolve(inst):
        j = np.searchsorted(ends, inst, side="right")
        local = inst - starts[j]
        assert (local >= 0).all() and (local < n[j]).all()
        return j, local * WINDOW
    j, off = resolve(np.array([0, 400_000_000], dtype=np.int64))
    assert j[0] == 0 and off[0] == 0 and rel[0] == "common_crawl/adult_content/000000.npy"
    assert j[1] == 534 and off[1] == 3_355_651 * WINDOW == 109_957_971_968
    assert rel[534] == "common_crawl/science_math_and_technology/000018.npy"

    # Step 4: the permutation. One uint32 array of N (2.9 GB), shuffled in place; nothing else of size N.
    for s in args.steps:
        assert 1 <= s <= N // BATCH, s
    t0 = time.time()
    G = np.arange(N, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed + epoch)).shuffle(G)
    t_perm = time.time() - t0
    print(f"permutation PCG64({seed} + {epoch}) over {N:,} uint32: {t_perm:.1f} s")
    # Step 5: batch k = G[512k : 512k + 512]; trainer step s -> k = s - 1.
    batch_inst = {s: G[(s - 1) * BATCH:s * BATCH].astype(np.int64) for s in args.steps}
    del G

    tok = Tokenizer.from_file(cached(TOK_URL + "/tokenizer.json", os.path.join(args.in_dir, "tokenizer.json")))
    tok_cfg = json.load(open(cached(TOK_URL + "/tokenizer_config.json", os.path.join(args.in_dir, "tokenizer_config.json"))))
    assert tok.id_to_token(eos) == tok_cfg["eos_token"] == "<|endoftext|>", (tok.id_to_token(eos), tok_cfg["eos_token"])
    assert tok.token_to_id(tok_cfg["pad_token"]) == tk["pad_token_id"], tok_cfg["pad_token"]

    n_top = Counter()
    for r, c in zip(rel, n):
        n_top[r.split("/")[0]] += int(c)
    for s in args.steps:
        inst = batch_inst[s]
        j, offsets = resolve(inst)
        jobs = [(urls[f], int(o), sizes[f]) for f, o in zip(j, offsets)]
        t0 = time.time()
        with ThreadPoolExecutor(16) as ex:
            raw = b"".join(ex.map(fetch_window, jobs))
        t_fetch = time.time() - t0
        assert len(raw) == BATCH * WINDOW
        with open(os.path.join(args.in_dir, f"batch_{s}_ranges.bin"), "wb") as f:
            f.write(raw)
        tokens = np.frombuffer(raw, dtype="<u4").reshape(BATCH, SEQ).astype(np.uint32)
        assert int(tokens.max()) < tk["vocab_size"], int(tokens.max())
        texts = tok.decode_batch([row.tolist() for row in tokens], skip_special_tokens=False)
        n_eos = (tokens == eos).sum(axis=1)
        n_pad = (tokens == tk["pad_token_id"]).sum(axis=1)

        out = os.path.join(args.out_dir, f"batch_{s}")
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "tokens.npy"), tokens)
        with open(os.path.join(out, "instances.jsonl"), "w") as f:
            for b in range(BATCH):
                f.write(json.dumps({"index": b, "instance": int(inst[b]), "file": urls[j[b]],
                                    "offset": int(offsets[b]), "length": WINDOW, "tokens": int(tokens.shape[1]),
                                    "eos": int(n_eos[b]), "text": texts[b][:200]}, ensure_ascii=False) + "\n")
        top = Counter(rel[f].split("/")[0] for f in j)
        sub = Counter(rel[f].rsplit("/", 1)[0] for f in j)
        meta = {
            "trainer_step": s, "batch_index_0based": s - 1,
            "python": platform.python_version(), "numpy": np.__version__,
            "seed": seed, "epoch": epoch, "N": N, "batches_per_epoch": N // BATCH,
            "fingerprint_recomputed": fp, "fingerprint_trainer_state": state["dataset_fingerprint"],
            "trainer_state": CKPT_URL + "/train/rank0.pt", "trainer_state_data_loader": state,
            "tokenizer": TOK_URL + "/tokenizer.json",
            "tokenizer_sha256": hashlib.sha256(open(os.path.join(args.in_dir, "tokenizer.json"), "rb").read()).hexdigest(),
            "range_requests": BATCH, "bytes_fetched": len(raw),
            "seconds": {"head_1016_files": round(t_head, 1), "permutation": round(t_perm, 1), "fetch": round(t_fetch, 1)},
            "token_bytes_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "eos_total": int(n_eos.sum()), "instances_without_eos": int((n_eos == 0).sum()),
            "pad_tokens_total": int(n_pad.sum()), "instances_with_pad": int((n_pad > 0).sum()),
            "distinct_files": int(len(set(j.tolist()))),
            "instances_per_source": {k: {"batch": top[k], "expected_from_corpus_share": round(BATCH * n_top[k] / N, 1)}
                                     for k in sorted(n_top)},
            "instances_per_source_subdir": dict(sorted(sub.items())),
        }
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"step {s} -> batch index {s - 1}: {BATCH} ranges, {len(raw):,} bytes in {t_fetch:.1f} s; "
              f"{meta['distinct_files']} files; EOS {meta['eos_total']}; pad {meta['pad_tokens_total']}; token bytes sha256 {meta['token_bytes_sha256']}")
        print("  instances per source (batch, expected):",
              {k: (v["batch"], v["expected_from_corpus_share"]) for k, v in meta["instances_per_source"].items()})
    print(f"total {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
