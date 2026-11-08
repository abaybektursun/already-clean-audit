# Vendored, byte-identical, from Ai2's public repositories (Apache-2.0)

Fetched 2026-09-08 21:08Z on the DEPTHANCHOR seeder instance (`infra/aws_pg/seed_depthanchor_cpu.sh`
stage 2, `curl -fsSL`, log line `REF_OK`), uploaded to
`s3://pg-lineage-769063704696/depthanchor/code/v1/ref/`, and copied here unmodified.

| file | source | sha256 |
|---|---|---|
| `generate_depth_annotation.py` | `https://raw.githubusercontent.com/allenai/molmoact2/main/experiments/scripts/generate_depth_annotation.py` | `5e12c94ddfca31caccb32065edd8061b682befd267d16ee9c994f2e893d9ff45` |
| `reconstruct_from_tokens.py` | `https://raw.githubusercontent.com/allenai/molmoact/main/scripts/reconstruct_from_tokens.py` | `9f2c7ff3a630128159ef551b04696293b77816e9158f6372df6e7f9512f846f0` |

## Why these two files are here

`generate_depth_annotation.py` is the script that WROTE MolmoAct2's depth-token training targets. It
carries a self-contained Depth-Anything-V2 (`vitb`) and the depth VQ-VAE (128 codes, codebook 512,
input 320, downsample 32, `use_norm=False`, `max_value=10.0`), plus `encode_depth_codes`, which is the
exact preprocessing chain — DA2 at 518 → per-frame min-max to uint8 → nearest resize to 320 →
`(x/255 − 0.5)/0.5` → VQ-VAE encoder → a 10×10 code grid. G1 (is the model's depth depth?) compares the
checkpoint's own predicted grid against what this chain produces on the same frame, and G2 builds the
code ladder `d(c)` from `VQVAE.decode`. Reimplementing either would put an untested instrument between
the checkpoint and its own training target.

`reconstruct_from_tokens.py` is the MolmoAct v1 decoder script; it is the source of the VQ-VAE
constructor arguments quoted in `studies/depthanchor/SPEC.md` §2 and is kept for that citation only.

## The two import stubs

`generate_depth_annotation.py` imports `lerobot.utils.constants` (for a dataset root this study never
uses) and `av` (video decoding; our frames are parquet images). `infra/pg/depthanchor_model.py`
`_load_vendored_ref()` installs stub modules for both before executing the file, so the file stays
byte-identical to the released one and the sha256 above keeps meaning something. Its `cv2`, `pyarrow`
and `torch` imports are real and satisfied on the job instance.
