# 00b — Exact stage-1 batches of Olmo-3-1025-7B between two checkpoints

Date 2026-11-05. Read-only feasibility check. No model was run. Downloads: source files, configs, six
trainer-state files of 15–17 KB, 3 KB of token files, 4 KB of one metadata file, one tokenizer file (4.2 MB).

## Answer

**Yes: the 512,000 training instances of steps 60,001–61,000 can be computed from public files and fetched with
HTTP range requests, 512,000 ranges of 32,768 bytes, 16,777,216,000 bytes in total.** The order is a seeded
NumPy permutation over fixed 8,192-token windows of 1,016 public `.npy` files. The run's own trainer state is
public and gives the seed, epoch, batch counter and a dataset fingerprint. The fingerprint recomputed from the
public file sizes matches the training value exactly. One link is not yet proven by a request: that the public
files are byte-identical to the training files (only their sizes are proven). A one-batch GPU loss check against
the public per-step W&B loss settles it (step 60,001: `train/CE loss` = 2.1240015029907227).

## What the stage-1 run used (from its own checkpoint files)

The official script `OLMo-3-1025-7B-pretrain-1.py` names the mix `OLMo_mix_0625_official`
([script L54–67](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py#L54-L67)).
The run itself used a different mix name. The values below are from the run's checkpoint directory
`https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step60000/` (`config.json`,
`data_paths.txt`, `train/rank0.pt`). This is the directory the part-2 script resumes from at step 596,047
([pretrain-2 L104](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-2.py#L104)).

| item | value | source |
|---|---|---|
| run name | `OLMo25` | `config.json` → `run_name` |
| dataset class | `NumpyFSLDataset` (config `name: fsl`) | `config.json` → `dataset` |
| mix | `OLMo-mix-0625`, base `gs://ai2-llm`, tokenizer id `allenai/dolma2-tokenizer` | `config.json` → `dataset` |
| file list | 1,016 paths, identical line by line to `OLMo-mix-0625.txt` | `data_paths.txt`; [mix file](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/mixes/OLMo-mix-0625.txt) |
| sequence length | 8,192; `max_target_sequence_length` 8,192, so `chunk_size` = 1 | `config.json`; [data_loader.py L429–433](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L429-L433) |
| global batch | 4,194,304 tokens = 512 instances | `config.json` → `data_loader` |
| seed, epoch | 34,521, epoch 1 | `train/rank0.pt` → `data_loader` |
| token dtype | uint32 (vocab 100,278 > 65,535) | [numpy_dataset.py L2685–2699](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L2685-L2699) |
| instance filter | not set | `config.json` → `dataset` has no `instance_filter_config` |
| steps skipped by config | none (`steps_to_skip` unset) at steps 1,000, 60,000, 600,000, 1,413,814 | four `config.json` files |
| state at step 60,000 | `global_step` 60,000, `batches_processed` 60,000, `tokens_processed` 251,658,240,000, `world_size` 512, fingerprint `ddc8c024…8a27` | `step60000/train/rank0.pt` |
| state at step 61,000 | `batches_processed` 61,000, same fingerprint, same seed | `step61000/train/rank0.pt` |
| training code | OLMo-core `7ac0eae03d79` (branch `dirkg/Retrofit`), script `src/scripts/train/OLMo2.5-7B.py`, `numpy<2.0` | `config.json` → `launch.git`; [pyproject.toml L16](https://github.com/allenai/OLMo-core/blob/7ac0eae03d79d2b4fe7f9dd5e9b8dbbb7fae7c6e/pyproject.toml#L16) |

The same fingerprint, seed and batch size appear at steps 1,000, 60,000, 600,000 and 1,413,814. Stage 1 is
exactly one epoch: 723,872,890 instances // 512 = 1,413,814 batches, which equals the last stage-1 step in
[`OLMo-3-1025-7B.csv` L1416](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv#L1416).
At step 1,413,814 `batches_processed` is 0 (reset at the end of the epoch).

## Recipe: (step a, step b) → byte ranges → tokens → documents

Line references are to OLMo-core HEAD `ccf4ef4b` (2026-10-23). The same functions at the training commit
`7ac0eae` differ only in error messages and a thread pool in `__getitem__` (diffed class by class).
`data_loader.py`, `numpy_dataset.py` and `utils.py` are byte-identical at the four training commits sampled
(`54f6009d` at step 1,000, `7ac0eae0` at 60,000, `6800ac24` at 600,000, `5878bb48` at 1,413,814).

**Step 0. Steps to batch indices.** The trainer increments `global_step` once per batch
([trainer.py L1692–1693](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/train/trainer.py#L1692-L1693)).
The loader increments `batches_processed` before it yields
([data_loader.py L185–191](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L185-L191)).
So trainer step s trains 0-based batch index s − 1. The checkpoint at step a holds the weights after batch
a − 1. The interval (a, b] = (60,000, 61,000] is batch indices 60,000 … 60,999.

**Step 1. File list in training order.** Training read `gs://ai2-llm/preprocessed/dolma2-0625/v0.1/...`
(`data_paths.txt`); that bucket answers HTTP 403. The public copies are listed in
[`OLMo-mix-0625-official.txt`](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/mixes/OLMo-mix-0625-official.txt)
(1,016 lines, CRLF endings). Line i of the official mix pairs with line i of `data_paths.txt`: all 1,016
basenames agree, and directories map one to one (`all-dressed-snazzy2`→`common_crawl`, `s2pdf`→`olmocr_science_pdfs`,
`stack-edu`→`stack_edu`, `arxiv`→`rpj-proofpile-arxiv`, `wikipedia`→`dolma1_7-wiki-en`). URL of line i:
`https://olmo-data.org/` + path with `{TOKENIZER}` = `allenai/dolma3-tokenizer`
([mixes/__init__.py L98–127](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/mixes/__init__.py#L98-L127)).

**Step 2. Instances per file.** Each file is cut into floor(bytes / (4 × 8,192)) windows of 8,192 tokens. The
tail of each file is dropped. Windows never cross files. Global instance indices are the files' windows
concatenated in file order
([`_sizes_and_offsets` L710–743, `_get_file_size_and_length` L754–772](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L710-L772)).
Sizes come from one HEAD request per file. Result: N = 723,872,890 instances, 23,719,883,626,704 bytes,
4,191,796 tail tokens never trained.

**Step 3. Gate: fingerprint.** The fingerprint is SHA-256 over the class name, six fields and
`path=<basename>,size=<bytes>,` for each file in order
([`fingerprint` L296–308](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L296-L308);
fields at [L616–625](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L616-L625)).
Recomputed from the 1,016 public sizes it is `ddc8c024ea19a0b8f2bbf62c4fd90ab2dd0231d379f8d4e1fb0f14de8ddb8a27`,
equal to the value in `step60000/train/rank0.pt`. Failing inputs: one size changed by 4 bytes, or the dtype
written as `uint32` instead of `<class 'numpy.uint32'>`; both give a different hash. The official dump script
runs the same check ([dump_training_batch.py L200–210](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/dump_training_batch.py#L200-L210)).

**Step 4. Permutation.** `rng = np.random.Generator(np.random.PCG64(seed + epoch))` with seed 34,521 and epoch 1,
then `rng.shuffle(np.arange(N, dtype=np.uint32))`
([`_build_global_indices` L667–690](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L667-L690);
[`get_rng` utils.py L505–506](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/utils.py#L505-L506)).
The array is 2.9 GB. No `global_indices*.npy` is published: the loader writes it to a node-local work directory
(`./dataset-cache` in `config.json`; [L656–665](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L656-L665)).

**Step 5. Batch to instances.** Batch k is `G[512k : 512k + 512]` after truncation to 512 × (N // 512)
([`total_size` L642–649, `__getitem__` L692–718](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L642-L718)).
The data-parallel rank split (`[:, rank::512]`, L720–747) does not change which instances are in the batch.

**Step 6. Instance to byte range.** For instance i, file j is the one with start_j ≤ i < end_j; local
index ℓ = i − start_j; bytes [32,768 ℓ, 32,768 ℓ + 32,767] of file j
([dataset `__getitem__` L668–708, `_read_chunk_from_array` L745–752](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L668-L752);
[`load_array_slice` utils.py L312–330](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/utils.py#L312-L330);
HTTP `Range` in [io.py L646–650](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/io.py#L646-L650)).
Worked examples from the public sizes: instance 0 → `common_crawl/adult_content/000000.npy` bytes 0–32,767;
instance 400,000,000 → file 534 `common_crawl/science_math_and_technology/000018.npy`, ℓ = 3,355,651, bytes
109,957,971,968–109,958,004,735.

**Step 7. Tokens to documents.** Bytes are raw little-endian uint32, no `.npy` header (first bytes
`5200000041060000`, not `\x93NUMPY`). The dataset adds no special tokens
([docstring L551](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/numpy_dataset.py#L551));
documents end with EOS 100257 and there is no BOS (`bos_token_id` unset). Split each window at 100257 and decode
with `allenai/dolma2-tokenizer` (`tokenizer.json`, HF commit `5292e5d6`). The first and last document of a window
are usually partial. Document identity is in the sidecar `<file>.csv.gz`, one row per document:
`start,end,id,src,loc` in token units ([dolma memmap_writer.py](https://github.com/allenai/dolma/blob/main/python/dolma/tokenizer/memmap_writer.py),
`MemmapMetadata` in [data_types.py](https://github.com/allenai/dolma/blob/main/python/dolma/tokenizer/data_types.py)).
The sidecars are gzip, so they are fetched whole: 128,949,804,073 bytes for all 1,016 (HEAD on each). A
1,000-step interval touches an expected 1,015.1 of the 1,016 files, so document IDs for any interval need
nearly all sidecars once.

**Step 8. Gate before trusting the list: one loss on one batch (GPU, not run).** Load HF
`allenai/Olmo-3-1025-7B` revision `stage1-step60000`. Compute the mean next-token cross-entropy over the
512 × 8,191 positions of reconstructed batch 60,000, plain causal attention across documents (stage 1 has
`generate_doc_lengths: false`), with the sliding window set in the revision's
[`config.json`](https://huggingface.co/allenai/Olmo-3-1025-7B/blob/stage1-step60000/config.json)
(`sliding_window` 4,096; `layer_types` starts sliding, sliding, sliding, full). Compare with
the public W&B value for step 60,001: 2.1240015029907227 (run `zckrmllo`, project `ai2-llm/Olmo-3-1025-7B`). A
second pair: checkpoint `stage1-step61000` on batch 61,000 against step 61,001, 2.1173441410064697. Spread
across steps 60,001–61,000: mean 2.0833, sd 0.0248; 2.7 % of steps lie within ±0.003 of 2.1240. A wrong
order or an off-by-one batch therefore lands far from both targets with high probability. The tolerance
must be frozen on repeated forward passes (for example, two attention backends on the same batch) before the
comparison. Cost estimate, not measured: 2 × 7.3e9 × 4.19e6 ≈ 6.1e16 FLOP per batch, about 0.1–0.2 GPU-h on
one L40S (g6e.2xlarge) per batch, plus 14.6 GB of weights per checkpoint.

The core of Steps 2–6 in code (run it on a cloud instance; the 16.8 GB fetch is not laptop work):

```python
import numpy as np                     # numpy<2.0, as pinned at the training commit
SEQ, BATCH, ITEM = 8192, 512, 4
# urls: 1,016 URLs from OLMo-mix-0625-official.txt in file order; sizes: HEAD Content-Length of each
n = np.array([s // (ITEM * SEQ) for s in sizes], dtype=np.int64)
ends = np.cumsum(n); starts = ends - n
N = int(ends[-1]); assert N == 723_872_890 and N // BATCH == 1_413_814
G = np.arange(N, dtype=np.uint32)
np.random.Generator(np.random.PCG64(34521 + 1)).shuffle(G)
def ranges(k):                         # k = 0-based batch index = trainer step - 1
    inst = G[k * BATCH:(k + 1) * BATCH].astype(np.int64)
    j = np.searchsorted(ends, inst, side="right")
    return [(urls[f], int(i - starts[f]) * SEQ * ITEM, SEQ * ITEM) for f, i in zip(j, inst)]
fetch = [r for k in range(60000, 61000) for r in ranges(k)]   # 512,000 (url, offset, length)
```

The official alternative is [`src/scripts/dump_training_batch.py`](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/dump_training_batch.py)
(added 2025-10-14, PR #418). As written it rebuilds the dataset from `config.json`, so it reads `gs://ai2-llm`
(403) and checks paths against `OLMo-mix-0625` (L154–161). It needs two edits: mix `OLMo-mix-0625-official` with
base `https://olmo-data.org`, and the path check replaced by the basename pairing of Step 1. Its `--step` argument
is passed to `data_loader[args.step]` (L229), a 0-based batch index: the batch of trainer step s is `--step s-1`.

## Byte count

| interval | instances | range requests | bytes |
|---|---|---|---|
| 1,000 steps (60,001–61,000) | 512,000 | 512,000 × 32,768 B | 4,194,304 tokens × 1,000 × 4 B = **16,777,216,000 B** (15.625 GiB) |
| 100 steps | 51,200 | 51,200 × 32,768 B | 4,194,304 × 100 × 4 B = **1,677,721,600 B** |
| whole stage-1 corpus, for scale | 723,872,890 | — | 23,719,883,626,704 B; one 1,000-step interval is 0.071 % of it |
| optional document IDs | — | 1,016 whole files | 128,949,804,073 B once, reusable for every interval |

Adjacent instances are rare (512,000 of 723.9 M), so the ranges do not merge.

## Verified by a request vs read from code

### Verified by a request (raw headers saved under the scratchpad path in the last section)

- **Public token file, HEAD**, `https://olmo-data.org/preprocessed/dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/common_crawl/adult_content/000000.npy`:
  ```
  HTTP/2 200
  content-type: application/octet-stream
  content-length: 633595964
  last-modified: Tue, 07 Oct 2025 23:04:42 GMT
  accept-ranges: bytes
  etag: "dc35cfe6c46215c7b7bc6ebd7b128e0d-3"
  server: cloudflare
  ```
- **Same file, `Range: bytes=0-1023`**:
  ```
  HTTP/2 206
  content-length: 1024
  etag: "dc35cfe6c46215c7b7bc6ebd7b128e0d-3"
  content-range: bytes 0-1023/633595964
  accept-ranges: bytes
  ```
  The 256 values are all < 100,278, with EOS 100257 at positions 195, 201, 205; they decode to English web text.
- **All 1,016 public files, HEAD**: 1,016 × `HTTP/2 200`, 1,016 × `accept-ranges: bytes`. Their sizes reproduce
  the training fingerprint (Step 3).
- **Wiki file, `Range: bytes=0-2047`** on `.../dolma1_7-wiki-en/000000.npy`: `HTTP/2 206`,
  `content-range: bytes 0-2047/1618287148`. **Its sidecar, `Range: bytes=0-4095`** on `000000.csv.gz`:
  `HTTP/2 206`, `content-range: bytes 0-4095/8665594`. First sidecar row `0,110,35387594,dolma1_7-wiki-en/00028.gz,8203`.
  The first three document ends in the sidecar (110, 163, 183) equal the EOS positions + 1 in the token bytes.
  Decoded first document: `"Wathone\n\nWathone (, lit. \"Earth\" from Pali \"vasundra\"), born Khin Hla, was a prominent Burmese painter, …"`.
- **Trainer state, HF bucket**, `https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step60000/train/rank0.pt`:
  ```
  HTTP/2 302
  x-linked-size: 15445
  x-xet-hash: 8f1479c659db3d8753e6733e5c693d8a6877d46da6d4c4d1ab58d97fb0ad4e4d
  last-modified: Thu, 24 Jul 2025 04:50:11 GMT
  HTTP/2 200
  accept-ranges: bytes
  content-length: 15445
  ```
  Read with `pickletools` (no unpickling). `config.json` (10,425 B) and `data_paths.txt` (111,470 B) also return
  302 → 200. `train/rank0.pt` returns 302 at steps 1,000, 10,000, 59,000, 61,000, 62,000, 100,000, 596,000,
  596,047, 597,000, 600,000, 1,000,000, 1,413,000, 1,413,814.
- **The CSV's stage-1 URLs do not serve**: `https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage1/step60000/`
  + `config.json`, `train/rank0.pt`, `data_paths.txt`, `model_and_optim/.metadata` → `HTTP/2 404` (also steps 0,
  1,000, 596,000, 1,413,000, 1,413,814). Stage 2 and 3 do serve: `stage2/step1000/train/rank0.pt` 200 (15,381 B),
  `stage3/step11921/train/rank0.pt` 200 (15,317 B).
- **Training path not public**: `https://storage.googleapis.com/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2/adult_content/000000.npy`
  → `HTTP/2 403`; the same path on `olmo-data.org` → `HTTP/2 404`.
- **HF model repo** `allenai/Olmo-3-1025-7B`, revision `stage1-step60000` (API tree): 14 files, HF weights and
  tokenizer only. No data-loader state, no `global_indices`. 1,421 `stage1-*` branches.
- **W&B is publicly readable** through `https://api.wandb.ai/graphql` without a key: run `zckrmllo`
  (`OLMo25-20250723T222053+0000`) has 1,000 distinct steps in 60,001–61,000 with `train/CE loss` and
  `optim/step skipped`. One step in the interval was skipped: 60,780.

### Read from code only (not exercised)

- The permutation (Step 4) was not computed. Its output under numpy 1.26 versus 2.x was not compared.
- Steps 0, 2, 5, 6 were read from code and checked only through the fingerprint and the epoch count
  (N // 512 = 1,413,814).
- Byte-level identity of public and training files: not verified. The fingerprint covers basenames and sizes only.
- The loss check (Step 8) was not run.
- Mapping of HF bucket `OLMo25` to the released model: inferred from equal step and token counts
  (CSV L62–63 versus `rank0.pt`) and the part-2 `load_path`. Weights were not compared.

## Risks

1. **File content.** Sizes match for all 1,016 files; bytes are unproven. The files were renamed between training
   (`dolma2-0625/v0.1`) and release (`dolma3-0625/v0.1-official`). Step 8 decides it.
2. **NumPy version.** Training pinned `numpy<2.0` (training commit `pyproject.toml` L16). Use the same major
   version. A different `Generator.shuffle` stream would change every batch; Step 8 detects it.
3. **Off-by-one.** Trainer step s is batch s − 1. The dump script's `--step` is the batch index. Step 8 with
   neighbors as controls separates the two.
4. **Skipped updates.** `SkipStepAdamW` (sigma factor 6, window 128, `config.json`) multiplies the update by 0
   on outlier steps ([adamw.py L23–44](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/optim/adamw.py#L23-L44)).
   Step 60,780 was skipped: its 512 instances were in a batch but changed no weight. Read
   `optim/step skipped` from W&B for any interval.
5. **Restarts.** The job was preemptible; W&B lists 35 `OLMo25-*` runs. On resume the loader restores
   `batches_processed` ([L451–480](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L451-L480), [L757–777](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py#L757-L777)),
   so a re-run step sees the same batch. Steps 60,001–61,000 lie inside one W&B run (steps about 39,520–168,989).
   Only four of the restart commits were checked for identical data code.
6. **Epoch end and dropped data.** 122 instances after batch 1,413,813 and 4,191,796 file-tail tokens were never
   trained. Nothing else is filtered: no instance filter, no `steps_to_skip`.
7. **Part 1 / part 2 boundary.** Part 1 had `hard_stop` 597,046 steps; part 2 resumed from step 596,047 with a new
   LR schedule (pretrain-1 L99–102, pretrain-2 L85–111). The data order continues (same fingerprint and seed at
   596,047 and 600,000). Which schedule trained steps 596,048–597,046 in the released weights was not resolved;
   several short W&B runs cover that range. This does not touch 60,000–61,000.
8. **Stages 2 and 3.** Different mixes, seeds and possibly dataset classes
   ([README L32–39](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/scripts/official/OLMo3/README.md#L32-L39)).
   The recipe must be redone from that stage's `config.json` and `rank0.pt` (both serve on `olmo-checkpoints.org`
   for stages 2 and 3). The dump script handles only FSL datasets (L144–146).
9. **Hosting.** 512,000 requests per interval to a Cloudflare host; rate limits were not tested. The HF bucket
   files were uploaded 2026-07-23 (`uploadedAt` in the tree API) and could move.

## Alternatives, with the same recipe check

| suite | order recipe | state files public | data public, ranges | bytes per 1,000 steps | status |
|---|---|---|---|---|---|
| **OLMo-2 7B** (`peteish7`) | old trainer `allenai/OLMo`: PCG64(seed + epoch) shuffle of 4,096-token windows ([iterable_dataset.py L91–98](https://github.com/allenai/OLMo/blob/090253dac6688f2532509daa7aa2eb5fae50e956/olmo/data/iterable_dataset.py#L91-L98)); tool [`scripts/inspect_train_data.py`](https://github.com/allenai/OLMo/blob/090253dac6688f2532509daa7aa2eb5fae50e956/scripts/inspect_train_data.py) | `https://olmo-checkpoints.org/ai2-llm/peteish7/step60000-unsharded/config.yaml` 200 (187,137 B), `train.pt` 200 (17,164 B): epoch 0, step 60,000, 61,440,000 examples; seed 6198; 1,024 × 4,096 tokens per step | paths in [OLMo2-7B-stage1.yaml](https://github.com/allenai/OLMo/blob/090253dac6688f2532509daa7aa2eb5fae50e956/configs/official-1124/OLMo2-7B-stage1.yaml) on `olmo-data.org`; training read `/weka/...` (1,122 paths in `config.yaml`); `train_data/global_indices.npy` 404 | 16,777,216,000 B (uint32) | likely feasible; not verified. No fingerprint in `train.pt`, so the size check used here is unavailable. A repetition instance filter is on (loss-masked instances stay in the batch). |
| **OLMo-2-0425-1B-early-training** | not stated | repo `main` has HF weights and tokenizer only | — | — | **no**. The [card](https://huggingface.co/allenai/OLMo-2-0425-1B-early-training) says it is a re-run from step 0 that is "not identical to the original run". No config, seed or loader state is published. |
| **Pythia** | preshuffled data in training order: batch k = rows 1,024k … 1,024k + 1,023 ([batch_viewer.py L42](https://github.com/EleutherAI/pythia/blob/a19eecb807ec2c79a39ebf18108816e6ffffc1d5/utils/batch_viewer.py#L42); [README L8, L49, L222–281](https://github.com/EleutherAI/pythia/blob/a19eecb807ec2c79a39ebf18108816e6ffffc1d5/README.md)) | not needed | `EleutherAI/pile-standard-pythia-preshuffled`: 20 × 30,000,000,000 B + 78,336,000 B = 146,432,000 rows × 2,049 tokens × 2 B = 143,000 × 1,024 rows exactly; HEAD on shard 0: `accept-ranges: bytes` | 1,024,000 × 4,098 B = 4,196,352,000 B, one contiguous range (100 steps: 419,635,200 B) | feasible from documentation; not verified by a loss check. `batch_viewer.py` slices to `(end+1)*1024`, one batch more than the interval. |

OLMoTrace ([arXiv 2504.07096](https://arxiv.org/html/2504.07096v1)) does not expose data order. It indexes the
whole training corpus with infini-gram, re-tokenized with the Llama-2 tokenizer, and returns documents that
contain verbatim spans of a model output. The paper has no notion of training step or checkpoint.

## Corrections to `00_ASSETS.md` (not edited here)

- Line 146 says stage-1 OLMo-core checkpoints are "not verified for stage 1" and that "a per-step document index
  is not a published file". The per-step index is still not a file, but it is computable (above). Stage-1
  OLMo-core checkpoints, with trainer state and `model_and_optim/` shards, are public in the HF bucket
  `allenai/ai2-llm` under `checkpoints/OLMo25/stepN/` (the first 1,000 entries of the step-60,000 listing alone
  hold 10,954,792,031 B of `model_and_optim` shards; the listing is paginated).
- Line 288 ("No data-order index") should point here.

## Evidence files (scratchpad, not in the repo)

Base: `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/`

- `dataorder_in/headers/` — raw HTTP headers for every request quoted above.
- `dataorder_in/listings/official_heads.tsv` — HEAD status, size, `accept-ranges`, `last-modified` for all 1,016 token files; `official_meta_heads.tsv` for the 1,016 sidecars.
- `dataorder_in/ckpt_step{1000,60000,61000,600000,1413814}/` — `config.json`, `train/rank0.pt` (and `data_paths.txt` at 60,000).
- `dataorder_in/wandb/zckrmllo_60001_61001.json` — per-step `train/CE loss` and `optim/step skipped`, steps 60,001–61,002.
- `dataorder_in/tokbytes/` — the 3 KB of token bytes and 4 KB of sidecar bytes.
- `dataorder_scripts/fingerprint.py` (Step 3 with its two failing inputs), `peek_pt.py` (trainer state without unpickling), `decode.py`, `meta_check.py`, `offsets.json` (per-file instance offsets).
