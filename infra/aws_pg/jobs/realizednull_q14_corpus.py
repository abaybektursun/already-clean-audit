"""REALIZEDNULL session 2 (N9): the 196-window corpus for the Qwen2.5-14B cell, built off-box (SPEC §3.8
prerequisite 4, §9.2 N9, §9.6 step 3, run R6a).

infra/vast/prepare_corpus.py's recipe with 196 windows instead of 98: wikitext-103-raw-v1 test at the pinned
revision, joined with "\\n", tokenized with the Qwen2.5-14B tokenizer, which has no BOS, so the first 196 x 512 ids
(prepare_corpus.py:38-42). The tokenizer comes from the hub at the commit shackleton:v1-14b baked
(studies/lineage/paligemma1-pi0/research/shackleton_manifest.json → models.Qwen/Qwen2.5-14B.sha); no weights are
read. Windows 0-97 must hash to the corpus GB-3 scored (actstats_flags.json → corpus_sha256), or the job fails.

Writes corpus196.npy (196, 512) int64 and corpus196.json (its file sha256, the array sha256 in prepare_corpus.py:46's
convention, shape, dtype, tokenizer, dataset) to --out. The cell (infra/vast/cells/realizednull_q14.py) reads both
from results/mdl/realizednull_q14_inputs/out/.

    ./infra/aws_pg/pg run infra.aws_pg.jobs.realizednull_q14_corpus --type g6.xlarge --no-weights \\
        --s3-prefix results/mdl/realizednull_q14_inputs --budget-min 20
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from infra.aws_pg.jobs.gutenberg_gb1 import WIKITEXT_REV
from infra.pg.humboldt_core import hf_setup

REPO = "Qwen/Qwen2.5-14B"
REVISION = "97e1e76335b7017d8f67c08a19d103c0504298c9"     # shackleton_manifest.json → models.Qwen/Qwen2.5-14B.sha
N_WINDOWS, WINDOW, N_BAKED = 196, 512, 98
# results/shackleton_r_v1_14b/cells/v1_14b_actstats/actstats_flags.json → corpus_sha256: the 98 x 512 corpus baked into
# shackleton:v1-14b (prepare_corpus.py:46, sha256 of the int64 array's bytes)
GB3_CORPUS98_SHA256 = "e7ab370437e84094776eeedca9c4e09a4ca446fd07038961eced31c947fd304f"
NAME = "corpus196"


def windows_from_ids(ids, n_windows=N_WINDOWS, window=WINDOW):
    """prepare_corpus.py:39-42 (the no-BOS branch): the first n_windows x window ids, as int64 rows."""
    need = n_windows * window
    assert len(ids) >= need, f"corpus too short: {len(ids)} < {need}"
    return np.asarray(ids[:need]).reshape(n_windows, window).astype(np.int64)


def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check_baked(windows, want=GB3_CORPUS98_SHA256):
    """Windows 0-97 are the corpus GB-3 scored, byte for byte."""
    got = array_sha256(windows[:N_BAKED])
    assert got == want, f"windows 0-{N_BAKED - 1} hash {got}, the baked GB-3 corpus is {want}"
    return got


def write_corpus(windows, out, meta):
    """corpus196.npy and corpus196.json in out; returns the JSON record."""
    assert windows.shape == (N_WINDOWS, WINDOW) and windows.dtype == np.int64, (windows.shape, windows.dtype)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    npy = out / f"{NAME}.npy"
    np.save(npy, windows)
    doc = {**meta, "file": npy.name, "file_sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
           "array_sha256": array_sha256(windows), "first98_array_sha256": array_sha256(windows[:N_BAKED]),
           "shape": list(windows.shape), "dtype": str(windows.dtype)}
    (out / f"{NAME}.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hf_setup()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    assert tok.bos_token_id is None, f"{REPO} has a BOS token ({tok.bos_token_id}): prepare_corpus.py's other branch"
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", revision=WIKITEXT_REV)
    text = "\n".join(r["text"] for r in ds)
    ids = tok(text, return_tensors="np").input_ids[0]
    windows = windows_from_ids(ids)
    check_baked(windows)
    doc = write_corpus(windows, a.out, {
        "tokenizer": f"{REPO}@{REVISION}", "dataset": f"Salesforce/wikitext:wikitext-103-raw-v1:test@{WIKITEXT_REV}",
        "recipe": "infra/vast/prepare_corpus.py:38-42 with 196 windows", "n_ids": int(len(ids))})
    print(f"REALIZEDNULL_Q14_CORPUS_OK shape={doc['shape']} file_sha256={doc['file_sha256']} "
          f"first98_array_sha256={doc['first98_array_sha256']}", flush=True)


if __name__ == "__main__":
    main()
