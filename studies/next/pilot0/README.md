# pilot0: offline reconstruction of two stage-1 batches of Olmo-3-1025-7B

Date 2026-11-04. Recipe: `../00b_DATA_ORDER.md`. No model was run. No GPU was used.

**The batch logged at W&B step 60,001 (`train/CE loss` 2.1240015029907227) is in `batch_60001/`, not in
`batch_60000/`.** Directories are named by trainer step, and trainer step s trains the 0-based batch index s − 1.
The dataset fingerprint recomputed from the 1,016 public file sizes equals the value in the trainer state:
`ddc8c024ea19a0b8f2bbf62c4fd90ab2dd0231d379f8d4e1fb0f14de8ddb8a27` on both sides (`run.log`).

## What was run

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd $SP && /usr/bin/time -l $SP/venvs/np1/bin/python -I /Users/abaybektursun/projects/interp_research/studies/next/pilot0/reconstruct_batch.py --in-dir $SP/pilot0_in 60000 60001
```

`--out-dir` defaults to the script's directory, so the outputs are `batch_60000/` and `batch_60001/` here.

- Environment: Python 3.12.12 and NumPy 1.26.4 (`run.log`, line 2; the training commit pins `numpy<2.0`), and
  tokenizers 0.23.3 (`tokenizers.__version__` in the venv `$SP/venvs/np1`).
- Wall time 48.95 s, `/usr/bin/time` real (`run.log`). Phases: 1,016 HEAD requests 3.7 s; permutation 27.6 s;
  range fetch 7.5 s (step 60,000) and 6.5 s (step 60,001) (`batch_*/meta.json` → `seconds`).
- Peak memory: maximum resident set size 2,944,303,104 B; peak memory footprint 3,039,316,776 B (`run.log`).
  The laptop has 25,769,803,776 B of RAM (`sysctl hw.memsize`). The only array of length N is one uint32
  `np.arange(N)`, shuffled in place.
- Bytes fetched per run: 2 × 512 range requests × 32,768 B = 33,554,432 B (`batch_*/meta.json` →
  `bytes_fetched`), plus 1,016 HEAD requests with no body. The raw range bytes are saved as
  `$SP/pilot0_in/batch_60000_ranges.bin` and `batch_60001_ranges.bin` (16,777,216 B each).
- Small inputs, fetched once into `$SP/pilot0_in/` by the script's `cached()` helper and reused (sizes are of
  those files): `config.json` 10,425 B, `data_paths.txt` 111,470 B, `rank0_step60000.pt` 15,445 B (HF bucket
  `allenai/ai2-llm`, `checkpoints/OLMo25/step60000/`); `OLMo-mix-0625-official.txt` 118,078 B (OLMo-core
  `ccf4ef4b`); `tokenizer.json` 7,137,177 B and `tokenizer_config.json` 4,345 B (HF `allenai/Olmo-3-1025-7B`,
  revision `stage1-step60000`). Total 7,396,940 B.
- `run.log` is the second run. A first run with the same code, except for one renamed key in `meta.json` and no
  pad-token count, produced byte-identical `tokens.npy` and `instances.jsonl` (same sha256 as listed below).

## Fingerprint

| side | value | file |
|---|---|---|
| recomputed from public HEAD sizes, training basenames, run config fields | `ddc8c024ea19a0b8f2bbf62c4fd90ab2dd0231d379f8d4e1fb0f14de8ddb8a27` | `run.log`; `batch_*/meta.json` → `fingerprint_recomputed` |
| stored in trainer state `step60000/train/rank0.pt` → `data_loader.dataset_fingerprint` | `ddc8c024ea19a0b8f2bbf62c4fd90ab2dd0231d379f8d4e1fb0f14de8ddb8a27` | `run.log`; `batch_*/meta.json` → `fingerprint_trainer_state` |

The script asserts equality. It also asserts that the recipe's two failing inputs change the hash: one file size
+4 bytes, and the dtype written `uint32` instead of `<class 'numpy.uint32'>`. Both assertions passed.

## Which directory the GPU check uses, and why

| directory | trainer step s | 0-based batch index | W&B `train/CE loss` at step s | file of the W&B value |
|---|---|---|---|---|
| `batch_60001/` | 60,001 | 60,000 | 2.1240015029907227 | `$SP/dataorder_in/wandb/zckrmllo_60001_61001.json` |
| `batch_60000/` | 60,000 | 59,999 | 2.083585500717163 | `$SP/dataorder_in/wandb/zckrmllo_60000.json` |

**Use `batch_60001/`.** Three facts from the training code and the trainer state fix the index:

1. The trainer increments `global_step` before it trains the batch it just drew, then logs that batch's loss at
   the new step (`trainer.py` L1200–1226 at training commit `7ac0eae`). So step s is the s-th batch drawn
   (index s − 1), forwarded with the weights after s − 1 updates.
2. The step-60,000 trainer state has `batches_processed` 60,000 and `tokens_processed` 251,658,240,000 =
   60,000 × 4,194,304 (`run.log`, line 3). On resume, `NumpyFSLDataLoader.load_state_dict` sets
   `batches_processed = tokens_processed // global_batch_size`, and `_get_local_instance_indices` drops the first
   `batches_processed` rows of the 512-column index matrix (`data_loader.py` at `ccf4ef4b`, L720–777). The first
   batch after the checkpoint is index 60,000, logged at step 60,001.
3. W&B logs 2.1240015029907227 at `_step` 60,001.

`batch_60000/` is the neighbor control for a data off-by-one. Losses of different batches spread by about one
per-step sd (0.0248), and the two W&B values differ by 0.0404, which is 1.63 sd. If the index were off by one,
`batch_60000/` would land near 2.1240 and `batch_60001/` would not.

A second off-by-one is in the weights, and it was not checked: HF revision `stage1-step60000` is assumed to
hold the weights after 60,000 updates. If it held the weights after 59,999 updates, `batch_60001/` would move
by the effect of one update on an unseen batch. That is a smaller effect than the batch-to-batch spread.
`batch_60000/` would then reproduce its own W&B value 2.0835855 at the forward-pass tolerance. Under the
assumed naming, it is expected to land below 2.0835855, because update 60,000 trained on it. Whether one
update moves either loss by more than the forward-pass tolerance is not known.

## Batch contents

Instances per top-level source directory. "Expected" is 512 × (that directory's share of the N = 723,872,890
instances) (`batch_*/meta.json` → `instances_per_source`; counts per second-level directory are in
`instances_per_source_subdir`).

| source directory | `batch_60001/` | `batch_60000/` | expected |
|---|---|---|---|
| `common_crawl` | 384 | 393 | 389.5 |
| `dolma1_7-wiki-en` | 1 | 0 | 0.2 |
| `finemath-3plus` | 15 | 13 | 13.1 |
| `olmocr_science_pdfs` | 79 | 64 | 69.5 |
| `rpj-proofpile-arxiv` | 8 | 8 | 4.4 |
| `stack_edu` | 25 | 34 | 35.3 |
| total | 512 | 512 | 512 |

| quantity | `batch_60001/` | `batch_60000/` | file |
|---|---|---|---|
| EOS tokens (100257) | 2,622 | 2,555 | `meta.json` → `eos_total` |
| instances with no EOS | 101 | 91 | `meta.json` → `instances_without_eos` |
| pad tokens (100277) | 41,868 | 19,257 | `meta.json` → `pad_tokens_total` |
| instances containing pad tokens | 9 | 4 | `meta.json` → `instances_with_pad` |
| distinct files touched | 302 | 309 | `meta.json` → `distinct_files` |
| sha256 of the 16,777,216 token bytes | `c389cebe…b80e` | `0341b735…a681` | `meta.json` → `token_bytes_sha256` |
| sha256 of `tokens.npy` (file, with header) | `1e6a4d3b…db53a` | `df409af0…54907` | `shasum -a 256` |

The two batches share no instance (computed from the `instance` fields of the two `instances.jsonl` files).
All pad tokens are in `olmocr_science_pdfs` files. Each of the 13 instances holds one contiguous run. Every
run starts at position 0 or directly after an EOS. Where a run ends inside the window, the next token is an
ordinary token, not EOS. In `batch_60001/` instance 251 is 8,192 pad tokens. In `batch_60000/` instances 267
and 449 are 8,192 pad tokens each. These facts come from an ad hoc read of `tokens.npy`. Why the public files
hold pad runs was not traced further.

## Ten instance beginnings (`batch_60001/instances.jsonl`, indices 0–9)

Decoded with `tokenizer.json` from `allenai/Olmo-3-1025-7B` revision `stage1-step60000`, sha256
`73fd5254…eaecca` (`meta.json` → `tokenizer_sha256`). An ad hoc comparison with the recipe's
`allenai/dolma2-tokenizer` `tokenizer.json` (HF commit `5292e5d6`, `$SP/dataorder_in/tokenizer/tokenizer.json`)
found the same 100,278-entry vocabulary, added tokens, decoder and pre-tokenizer. The 100,000 merges are equal
after normalizing their JSON format (pairs versus space-joined strings). First 200 characters of each decoded
window; a line break in the text is written `\n`.

**0**: instance 513,588,242, `common_crawl/software_development/000019.npy`, byte offset 45,403,013,120, 0 EOS

```
2: https://addons.mozilla.org/en-US/firefox/addon/ft-deepdark/\n\nThe themes don't change the arrangement of the UI.\n\nTry Debian if you need something so rock solid that you can only get security update
```

**1**: instance 290,763,465, `common_crawl/literature/000016.npy`, byte offset 24,895,651,840, 6 EOS

```
 circumstances, of depriving a useful citizen of his life, must be sufficiently painful to such a mind as yours, by the evidence, appears to be; I shall therefore not add to those feelings, by any thi
```

**2**: instance 608,506,823, `olmocr_science_pdfs/science_math_and_technology/000000.npy`, byte offset 116,095,680,512, 0 EOS

```
 ratio.\n\n[0384] Both photodiodes 184 and trigger photodiodes 187 are located in the CMOS circuitry 86 under each hybridization chamber 180. The array of photodiodes combines, along with appropriate el
```

**3**: instance 582,861,839, `olmocr_science_pdfs/entertainment/000004.npy`, byte offset 960,036,864, 0 EOS

```
\n\n\n3. *The Heritage Encyclopedia of Band Music* by William H. Rehrig, edited by Paul E. Bierley (Westerville, Ohio: Integrity Press, 1991) has several helpful articles on band music and history in the
```

**4**: instance 380,394,277, `common_crawl/science_math_and_technology/000013.npy`, byte offset 74,161,979,392, 0 EOS

```
 if they weren't sharing the same MRU. I think that would be a nice thing to say. And I did learn a nice trick, which is to close the sidebar, you don't have to, like, start at the edge of it and push
```

**5**: instance 572,429,324, `finemath-3plus/000006.npy`, byte offset 56,874,500,096, 0 EOS

```
94    1.100   0.271   1.7436   1.3209     0.5006\n3.prog |   0.00164    0.002   0.998   1.0016   1.0007     0.4341\n----------------------------------------------------------------------\n\nlistcoef, perc
```

**6**: instance 335,921,384, `common_crawl/science_math_and_technology/000001.npy`, byte offset 37,974,343,680, 4 EOS

```
2011 MD”) passed by Earth and missed a direct hit by only 7,500 miles. An even closer encounter occurred earlier in 2011 when another small asteroid missed Earth by just 3,400 miles.\n\nAsteroids coming
```

**7**: instance 55,299,036, `common_crawl/electronics_and_hardware/000013.npy`, byte offset 3,140,517,888, 5 EOS

```
 G 9th Gen Gaming Laptop - 16GB DDR4 - NVIDIA GeForce GTX 1660 Ti - 17.3-inch Full HD (1920x1080) IPS-Level Panel, 144Hz, 3ms, 100% sRGB - Windows 10 Home Price in UAE\n\nModel: G731GU-BI7N9\n\nAvailabili
```

**8**: instance 637,522,746, `olmocr_science_pdfs/science_math_and_technology/000004.npy`, byte offset 125,053,272,064, 2 EOS

```
 \Delta Q = T \Delta S - \frac{\tau}{2\beta} \int \left\{ \frac{\partial}{\partial t} \left( \ln \left( \frac{\epsilon}{\epsilon_0} \right) \right) \right\} d\left( \frac{\epsilon}{\epsilon_0} \right)
```

**9**: instance 539,951,669, `common_crawl/sports_and_fitness/000013.npy`, byte offset 4,002,316,288, 8 EOS

```
 6. When the team hits an obstacle, their response is to discuss, disect and disposition it as a group.\n  7. Team members understand and support each other’s individual goals.\n  8. The team communicat
```

## Checks inside `reconstruct_batch.py` (each one is an `assert`)

- Run config: dataset `fsl`, sequence length 8,192, `max_target_sequence_length` 8,192, no instance filter,
  `generate_doc_lengths` false, global batch 4,194,304 tokens; trainer state seed 34,521 = config seed, epoch 1,
  `batches_processed` = `tokens_processed` // 4,194,304.
- File list: 1,016 public paths and 1,016 training paths. Every pair has the same path below the top-level
  directory. The top-level map is one to one: `all-dressed-snazzy2`→`common_crawl`, `arxiv`→`rpj-proofpile-arxiv`,
  `finemath-3plus`→`finemath-3plus`, `s2pdf`→`olmocr_science_pdfs`, `stack-edu`→`stack_edu`,
  `wikipedia`→`dolma1_7-wiki-en` (`run.log`).
- Every HEAD answers 200 with `Accept-Ranges: bytes`. N = 723,872,890 and N // 512 = 1,413,814.
- Fingerprint equality, plus its two failing inputs.
- Resolver against the recipe's worked examples: instance 0 → file 0 `common_crawl/adult_content/000000.npy`,
  byte 0; instance 400,000,000 → file 534 `common_crawl/science_math_and_technology/000018.npy`, byte
  109,957,971,968.
- Every range answers 206 with `Content-Range: bytes a-b/<HEAD size>` exactly, and a body of 32,768 B.
- Token ids < 100,278. The tokenizer maps 100257 to `<|endoftext|>` (= `eos_token` in `tokenizer_config.json`),
  and `pad_token` `<|pad|>` to 100277.

## NumPy 1.26.4 versus 2.4.3

The recipe lists the NumPy major version as an open risk. The script, in its version before the `meta.json` key
rename, was run under NumPy 2.4.3 (venv `$SP/venvs/np2`, Python 3.12.12). It used its own input directory
`$SP/pilot0_in/np2_in/`, which its first attempt filled from the network, and wrote to `$SP/pilot0_in/np2_out/`
(log `$SP/pilot0_in/np2_run.log`). Its `tokens.npy` and `instances.jsonl` for both steps are byte-identical to
the NumPy 1.26.4 outputs here (`cmp`; sha256 as in the table above). So `Generator(PCG64(34522)).shuffle` gives
the same permutation, at least for these 1,024 positions, under both versions. The first NumPy 2.4.3 attempt
stopped at the range fetch with `URLError [Errno 8] nodename nor servname provided` (a DNS failure; log
`$SP/pilot0_in/np2_run_attempt1_dns_error.log`). The re-run succeeded. The script has no retry. A
512,000-range interval will need one.

## GPU loss check (not run)
**Note 2026-11-08 (index worker):** the GPU check ran on 2026-11-05; see `GPU_CHECK.md` (first check) and `GPU_CHECK2.md` (J8, closed in `studies/next/JOURNAL.md` 2026-11-05 16:05 CST).

Run this on one GPU instance (the recipe names g6e.2xlarge, one L40S 48 GB) from this directory. The instance
needs `batch_60001/` and `batch_60000/` (each `tokens.npy` and `meta.json`), torch, and a transformers release that
registers `model_type` `olmo3`. The revision's `config.json` was written by transformers `4.54.0.dev0`
(`$SP/pilot0_in/hf_config_stage1-step60000.json`).

```
python loss_check.py --tokens batch_60001/tokens.npy --revision stage1-step60000 --target 2.1240015029907227 --sd 0.0248 --attn sdpa --out batch_60001/loss_check_sdpa.json
```

`loss_check.py` checks the sha256 of the token bytes against `meta.json`. It loads `allenai/Olmo-3-1025-7B` at
the revision in bfloat16 and forwards each of the 512 rows without an attention mask: causal attention across
documents, with the sliding window of 4,096 and the layer pattern from the revision's config, both asserted. It
sums fp32 cross-entropy over the 8,191 next-token positions of each row and divides by 512 × 8,191 =
4,193,792. It writes the mean, the difference from `--target`, that difference divided by `--sd`, top-1
accuracy, per-row values, and, for the first 32 positions of row 0, the target id with its probability and the
top-1 id with its probability, to the `--out` JSON. Targets
include the pad tokens, because training masked nothing: `get_labels` at `7ac0eae` (`data/utils.py` L541–556)
masks only `label_mask`, `attention_mask` and `instance_mask`, and this run had none of them.

Two more runs separate the forward-pass spread and the off-by-one. Run them first, and freeze the tolerance from
the sdpa − eager difference before reading `diff` of the main run (recipe Step 8):

```
python loss_check.py --tokens batch_60001/tokens.npy --revision stage1-step60000 --target 2.1240015029907227 --sd 0.0248 --attn eager --out batch_60001/loss_check_eager.json
python loss_check.py --tokens batch_60000/tokens.npy --revision stage1-step60000 --target 2.1240015029907227 --sd 0.0248 --attn sdpa --out batch_60000/loss_check_sdpa.json
```

Reference numbers. The sd 0.0248 is the recipe's value. Recomputed as the sample sd of `train/CE loss` over
the 1,000 steps 60,001–61,000, it is 0.024802 (mean 2.083304) (`$SP/dataorder_in/wandb/zckrmllo_60001_61001.json`).
The recipe estimates 0.1–0.2 GPU-h per pass on one L40S (not measured), so the three runs take about 0.3–0.6
GPU-h, plus a 14.6 GB weight download.

Known differences between this forward and the training forward. Their size is not measured.
- Output classes: the training LM head had `vocab_size` 100,352 (`$SP/pilot0_in/config.json` → `model`). The HF
  model has 100,278 (`$SP/pilot0_in/hf_config_stage1-step60000.json`). The 74 extra logits were in the training
  softmax denominator and are absent here.
- Numerics: training used FSDP with bfloat16 parameters and float32 reduction, flash attention, and a compiled
  model (`config.json` → `train_module`). Here it is HF bfloat16 with the chosen `--attn`.

## Not verified here

- That the public files are byte-identical to the training files. Only basenames and sizes are covered, through
  the fingerprint. The GPU check decides this.
- That HF revision `stage1-step60000` equals the OLMo-core checkpoint `OLMo25/step60000`.
- That `loss_check.py` runs. It was only byte-compiled (`python3 -m py_compile`), because no model may run on the
  laptop.

## Files

- `reconstruct_batch.py`: the reconstruction. `python -I reconstruct_batch.py --in-dir DIR [--out-dir DIR] STEP [STEP ...]`.
- `loss_check.py`: the GPU check above. Not run.
- `run.log`: stdout, stderr and `/usr/bin/time -l` of the run.
- `batch_<s>/instances.jsonl`: 512 lines with `index` (row in the batch), `instance` (global instance index),
  `file` (public URL), `offset` (byte offset), `length` (32,768 B), `tokens` (8,192), `eos`, `text` (first 200
  decoded characters).
- `batch_<s>/tokens.npy`: uint32, shape (512, 8192), rows in batch order.
- `batch_<s>/meta.json`: step, index, versions, fingerprints, trainer-state fields, byte counts, timings, hashes,
  per-source counts.
