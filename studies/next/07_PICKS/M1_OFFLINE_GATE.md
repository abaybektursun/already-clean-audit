# M1 offline gate

**Go for the first GPU step, the known number K0 (0.26 GPU-h on one L40S), once three laptop steps are
done.** The weight-free dry run ran the job from its entry point through `metrics.json`, the directory
counts, the freeze and the verdict. It printed `M1_DRYRUN_OK` after 110 checks, 0 failed (run of
2026-11-05, log `scratchpad/m1_gate/dryrun_5.log`). The three laptop steps are:
1. commit freeze 1;
2. the S3 existence checks of `M1_SPEC.md` §9.5 S1.3, which could not be run here (no AWS call was allowed);
3. the upload of the five laptop bank files.

The full run is projected at 9.73 GPU-h at the measured L40S rate and 10.15 at the lowest banked L40S rate,
against the ceiling of 12.

`scratchpad/` below means
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/`.
`BANK/` means `/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid/`.

## What was run

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-m1
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.m1_dryrun --hf $SP/m1_gate/hf \
    --grid /Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid \
    --corpus-manifest $SP/m1/corpus_manifest.json --t15-manifest $SP/m1/t15_manifest.json --out $SP/m1_gate/dry
```

- Inputs from earlier sessions: `scratchpad/m1/corpus_manifest.json` (sha256 `0c99a8e8…`) and
  `scratchpad/m1/t15_manifest.json`, both copied from S3 by the M1 asset check (`M1_ASSETS.md` sources).
- Environment: the scratchpad venv `m1_venv`, Python 3.12.12, CPU only. It has torch 2.12.0 and transformers
  5.9.0 (the versions of the banked grid harvest, `scratchpad/m1/t15_manifest.json`), numpy 1.26.4,
  safetensors 0.8.0, huggingface_hub 1.33.0, accelerate 1.15.0 and pyflakes 4.1.0, all from PyPI.
- Wall time 3 min 57 s (`scratchpad/m1_gate/dryrun_5.log`). The check values are in
  `scratchpad/m1_gate/dry/dryrun_summary.json`.
- No model weights were downloaded. The tiny models of section JOB are random tensors made on the laptop.
- pyflakes is clean on the three new modules.

## Fetched from Hugging Face (small text only)

38 files, 7,704,151 bytes in all, under `scratchpad/m1_gate/hf/`, read 2026-11-05:
- `refs.json` (185,024 B, the 1,487 branches);
- `tree_<branch>.json` and `config_<branch>.json` for the start, the three mix branches, `stage1-step1413814`,
  `stage1-step1000000`, `stage1-step560000`, `stage1-step476000`, `stage2-step47684` and `main`;
- `index_<branch>.json` (29,630 B each) for the same branches;
- `tok/` (`tokenizer.json` 7,137,177 B, `tokenizer_config.json`, `special_tokens_map.json`, from the
  math-code-reasoning-web branch);
- `headers/`: the safetensors JSON header only of four shards, read by HTTP range requests. Each was 8 bytes
  of length plus 4,624 to 13,256 bytes of header, with no tensor data.

## Shard identity facts (HF tree API, LFS sha256)

| branch | commit | weight files the index names | sha256 (first 16 hex) |
|---|---|---|---|
| `stage1-step476838` (start) | `31bf90b3211621ce8154e55e7fec5b5eda08e620` | 3 bf16, 14,596,063,712 B | `6005c96a297cb196`, `da953fe9917731e7`, `c5696461572e9e71` |
| `…mix-gen-mc-only-from-2T-ckpt` (G) | `45786899c6c427d17f9291f3d97c7e4fcbd4962b` | 6 fp32, 29,192,085,712 B | `e8e8abbfd63fe4da`, `453c2ae4ebcbdcac`, `982c106e6e5eb7ba`, `3ec0366cc7e8a487`, `e50e3dfb3866c4ff`, `81c6f17da24e3927` |
| `…mix-round5-from-2T-ckpt` (R) | `34e13d5fae4d1aacf4aa019c377a332ed09ff136` | 6 fp32 | `8e8c22d2fe10c0cb`, `89b4cbf3cfcb7704`, `8d2219cefc41a5a4`, `a0e4d0ef57c64ee8`, `f081c89197f1eaef`, `55509870bf443cca` |
| `…mix-math-code-reasoning-web-from-2T-ckpt` (M) | `ab9f4b070e9125de257bfbe31bf8cc777daadfb4` | 6 fp32 | `7c4231e0123f6252`, `fe69ddc329e5e8d5`, `717bb44705fa0d5a`, `dd44c4be40811003`, `9bbe71d55586ba17`, `ebc7c8f5e092adaa` |
| `stage1-step1413814` (T_15, K0 on AWS) | `373bad25002f1624757a73235c5ca844c6375c25` | 3 bf16 | `f256015ffbd26278`, `de327e942a56a61d`, `0262ab848daa983b` |
| `stage1-step1000000` (T_14, K0 on Vast) | `cb286dcdaaf3f92b3c419dd77439078fdebffe53` | 3 bf16 | `2940a4230fddc76b`, `a75b7b4afa9a191f`, `cf359431b10df43f` |
| `main` (the stale set) | `a81bae42db3975be1671e27b9c9a56da1a9f980f` | 3 bf16 | `0490d6668e613a29`, `e127ea479fb6e208`, `f3ddff10052ffe5d` |

The full values are in `infra/pg/m1.py` (`BRANCHES`, `STALE`). Section REAL checks them against the saved
reads. Each mix branch also lists the three `main` shards; their oids equal `main`'s (REAL, three checks). The
shard headers hold only `F32` tensors in the six-shard set and only `BF16` in the three-shard set (REAL, four
checks).

## Results by section

Each line is one `CHECK`. All passed.

### REAL: the registry and the config rewrite on the real files (42 checks)

- The registry matches `refs.json` and the trees for six branches: commit and every weight oid.
  - G-SHARD passes on each branch's own tree.
  - G-SHARD fails on `main`'s three bf16 shards, for every branch.
- The corpus manifest has the registered sha256 `0c99a8e8…`.
- The start's `config.json` equals `REF_CONFIG`.
- For each of the three real mix configs:
  - `rewrite_config` and G-CFG pass with `diff_keys` = `["sliding_window"]`;
  - the as-released config fails G-CFG on `architectures`, `model_type` and `sliding_window`.
- `m1_harvest.load_dir` was run on a snapshot that holds the real M config as a symlink to a blob, the real
  index and the real tokenizer.
  - It wrote a rewritten copy in a new directory.
  - The blob's sha256 was unchanged.
- `AutoConfig` on the copy gives `Olmo3Config`. It differs from the start's parsed config only in
  `sliding_window`.
- `AutoConfig` on the as-released config exits non-zero in a child process with transformers' "does not
  recognize this architecture" error.
- The model built on the meta device from the rewritten config has exactly the index's 355 tensor names and
  7,298,011,136 parameters.
- The seven cast-gate tensor names exist in the index.
- The mix tokenizer's vocabulary sha256 equals the corpus record `2f3c015e…`.
- The banked T_15 manifest and metrics (`scratchpad/m1/t15_manifest.json`, `BANK/1/out/t15_step1413814/metrics.json`)
  pass G-FIT and G-SAME against the registered settings, with `same_diff` empty.

### WIN: the sliding-window difference (2 checks)

A tiny Olmo 3 (4 layers, sdpa) was run on CPU over 2,049 positions.
- Logits are bit-identical for `sliding_window` 4096 and 4097.
- The control window of 1,000 changes them, so the check can see a window effect.

### ROOM: room to fire on control data (5 checks; the 640 banked grid poles, 16 points)

- C by band, the mean code+math share of top-50 slots:
  - T_13: 0.1704, 0.1353, 0.0566, 0.0530;
  - T_15: 0.1875, 0.1389, 0.0600, 0.0444;
  - T_00: 0.3094, 0.2631, 0.2804, 0.2487.
- τ_640 = 0.0134, 0.0025, 0.0035, 0.0098 and κ_640 = 0.195, 0.162, 0.2274, 0.2055; τ < κ in every band.
- `m1.label_lead` ran on the 15 neighbouring banked pairs in both orientations, 120 band labels.
  - SUPPORTED fired 9 times, for example T_01 → T_00 in bands 0–2 and T_03 → T_02 in bands 2–3.
  - REFUTED fired 29 times, for example T_14 → T_15 in band 3 and T_00 → T_01 in bands 0–2.
- G-ID passes on T_15 against itself (I = 0, 0, 0, 0). It fails on its failing input, T_13 in place of the
  twin: I = −0.0171, −0.0036, −0.0034, +0.0086 against the line 0.0006.

### K0 offline: the known number against its bank (10 checks)

- The five bank files exist at their `BANK/` paths with the sha256 registered in `m1.BANK_SHA`:
  - `t15_contexts.jsonl` `a4d46cd8…`
  - `t15_exemplar_a.npz` `84bd59fd…`
  - `t14_exemplar_a.npz` `9fabd78b…`
  - `t14_contexts.jsonl` `692e8585…`
  - `t13_exemplar_a.npz` `7976ca71…`
- 1,535 of the 32,000 T_15 positions (on 439 poles) lie in the first 2^20 tokens.
  - The bank against itself passes.
  - The failing input (T_14's values at those positions) gives a median relative difference of 0.3144 and a
    p99 of 1.5021. It fails against 0.0043 and 0.0222.
- The subset rule:
  - passes on the bank itself;
  - fails on positions + 1 (share 0.038);
  - fails on T_14's own top-50 (share 0.391).

### JOB: the job from its entry point, tiny model, CPU (25 checks)

- The stand-in model is a random Olmo 3: 4 layers, hidden 64, MLP width 96, the real 100,278-token tokenizer.
  The start and T_15 were saved as 3 bf16 shards. The three mix ends were saved as 6 fp32 shards with an
  `olmo2-retrofit` config and `sliding_window` 4097. Each mix "branch" also carries three stale bf16 shards
  of a stand-in `main`.
- The stand-in hub applies `huggingface_hub`'s own `filter_repo_objects` to the allow patterns, with blobs
  and snapshot symlinks as in the real cache.
- The tiny corpus has 10 directories: 3 code+math, 2 PDF, 4 web. It has 163,000 tokens in windows of 128,
  batch 4.
- The bank is a T_15-rule harvest of the stand-in T_15, made through the same `load_model` path. T_14, T_13,
  T_12 and T_00 stand-ins were harvested at T_15's positions.
- Results:
  - `--mode known` passes K0: 1,515 entries, exact share 1.000. The failing input's median relative
    difference is 0.770.
  - `--mode harvest` passes for start_a, start_b, G, R and M, and writes `m1/counts.npz`, `m1/quantiles.npz`
    and `m1/metrics.json`.
  - The harvest's own `metrics.json` was rewritten 9 times per harvest inside the loop (every 40 batches in
    the dry run).
  - The weights are deleted after each harvest.
  - G-CAST on the fp32 ends: a truncating cast differs from the loaded tensors in 48 % to 51 % of entries,
    while round to nearest even matches bit for bit.
  - Each harvest banks one top-50 slot counted as code+math and one counted as other, with decoded context.
  - `--mode refs` writes the six reference count files.
  - start_a and start_b counts are identical.
  - Counts sum to the filled slots.
  - The manifest settings pass G-SAME.
- One finding of the dry run's own building. When the bank was harvested from an in-memory fp32 model cast
  with `.to(torch.bfloat16)`, instead of through `from_pretrained(dtype=bfloat16)`, only 0.387 of the K0
  entries matched exactly. They were still within tolerance. Through the same loader, 1.000 matched. The
  load path changes the numbers in the last bits. The cause was not traced. The real K0 bank was made by the
  same `neurons_harvest.load_model`: the harvest code changed after the grid's commit `17057bd0` only in a
  parser refactor, an added `O2` key and a `qwen3_5` branch of `text_tower` (`git diff 17057bd0 HEAD`).

### FAIL: each gate on its planted failing input (14 checks)

Seven of these ran through the job's entry point in a child process with a planted registry. Each exits
non-zero with the gate's own message:

| gate | planted input | message |
|---|---|---|
| G-REV | registered commit replaced by zeros | `G-REV: snapshot … != registered commit 0000…` |
| G-SHARD | one oid changed | `G-SHARD failed: …` |
| G-SHARD | `main`'s three bf16 shards registered as the mix's | `G-SHARD failed: …` |
| G-CFG | `rewrite` off, so the `olmo2-retrofit` config reaches the gate | `G-CFG failed: … ['architectures', 'model_type', 'sliding_window']` |
| G-CFG | reference width 97 | `G-CFG failed: … ['intermediate_size', 'sliding_window']` |
| G-SAME | registered transformers `0.0.0` | `G-SAME: versions …` |
| inputs | one sha256 changed | `input t15_contexts.jsonl: sha256 …` |

Two more ran through the entry point:
- G-CORPUS: the manifest with one byte appended stops with `G-CORPUS: corpus manifest sha256`.
- K0 against the wrong bank (T_14 values): `K0 failed: … median_rel 2.61`.

Five were direct calls:
- G-CAST passes on a loaded mix end, and its failing input fails.
- G-CAST fails after one entry is changed by 1.0.
- G-FIT fails on `complete` false.
- G-FIT fails on NLL 12.223, banked T_0, outside [2.40, 3.20].
- G-SAME fails on batch 16.

The in-job failing inputs (G-SHARD against `main`'s shards, G-CFG against the as-released config, G-CAST
against the next layer's tensor) are evaluated in every job run and asserted to fail. G0's two built-in
failing inputs (the wrong activation, u × 1.1) failed in every tiny harvest, with 4 of 4 layers over
tolerance (`scratchpad/m1_gate/dry/banks/*/controls.json` → `G0`).

In all, 13 of the 14 gates of `M1_SPEC.md` §4 were shown failing on a planted input. The fourteenth, the
"complete set" branch (a missing harvest gives DOES NOT COUNT), was not exercised.

### VERDICT: freeze and the ordering test (11 checks)

- On the tiny job's own banks, `m1.freeze` and `m1.verdict` run and read every harvest gate. The samples are
  carried. The headline is DOES NOT COUNT because PC_NLL does not fire on random models, which is the right
  answer for random weights.
- On synthetic banks (8 layers × 200 neurons × 2 poles, 10 directories, one binomial draw of 50 slots per pole
  and checkpoint), with lead contrast [mean by band] and bar [by band]:

| case | shifts by band (code+math share) | result |
|---|---|---|
| satisfies the ordering | G −0.03, R +0.02, M +0.06 | H_lead SUPPORTED, H_order SUPPORTED, H_specific SUPPORTED; lead 0.091, 0.092, 0.091, 0.090; bar 0.0134, 0.0146, 0.0085, 0.0142 |
| violates: R above M in band 2 | R +0.10 in band 2 | H_lead SUPPORTED, H_order REFUTED |
| violates: M equals G | all +0.01 | H_lead REFUTED (lead 0.001, −0.000, 0.003, 0.002), H_order REFUTED |
| identity pair differs | start_b +0.02 | DOES NOT COUNT (G_ID false) |
| positive control fails | M worse than G on code+math windows | DOES NOT COUNT (PC_NLL false) |
| ceiling | M +0.6 | H_lead ABOVE_CEILING; "DOES NOT COUNT until inspected" |

- The freeze's room check passes on the synthetic references.
- The failing inputs of G-ID and PC_NLL fail.

### PROJECT: GPU-hours (1 check)

| lane and rates | K0 launch (K0 + references) | one bf16 harvest | one fp32 harvest | instance A (start_a, M, R) | instance B (start_b, G) | total |
|---|---|---|---|---|---|---|
| AWS L40S, T_15's 3,334 tok/s and 234 MB/s | 0.26 | 1.86 | 1.89 | 5.68 | 3.79 | **9.73** |
| AWS L40S, the lowest banked 3,215 tok/s and pilot-0's 143 MB/s | 0.28 | 1.93 | 1.97 | 5.92 | 3.95 | **10.15** |
| Vast RTX PRO 5000, 4,612 tok/s | 0.24 | 1.38 | 1.41 | 4.26 | 2.84 | **7.34** |

The ceiling is 12 GPU-h. The `--budget-min` kill limits (40, 390 and 270 minutes) sum to 11.67 h before boot.
The sources of every rate are in `M1_SPEC.md` §9.7.

## The first GPU step

After freeze 1, the S1.3 existence checks and the S1.4 upload (`M1_SPEC.md` §9.5):

```
aws s3 cp infra/aws_pg/steps/next_m1_k0-01.json s3://pg-lineage-769063704696/results/next_m1/k0-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m1/k0-01/steps.json" \
    --no-weights --type g6e.2xlarge --s3-prefix results/next_m1/k0-01 --budget-min 40 --json
```

**Go.** The no-go conditions are:
- freeze 1 not committed;
- any S1.3 listing empty;
- the inputs prefix not holding the five files.

The job itself stops in seconds, before any download, on a version mismatch (G-SAME) or a wrong input
sha256.

## Spec issues (gaps in `M1_ASSETS.md` and `06_CRITIC.md` M1; not amended here)

1. **Worst.** `M1_ASSETS.md` §4 confound 3 asks for "an exact-text hash check against the training pools …
   before unblinding". The same paragraph says the public `dolma3_dolmino_mix-100B-1025` is the final mix
   only, not the skewed samples, so the check cannot run as written. `M1_SPEC.md` §8 threat 3 states the
   threat and does not gate on it. Whether this blocks the reading "domain, not the very documents" is
   Fable's decision.
2. `06_CRITIC.md` M1 (lines 835–837) refutes on "the between-mix difference … inside the identity pair's spread".
   Per `M1_ASSETS.md` lines 31–42, a floor near zero gives that almost no room. The spec keeps the identity
   pair as the floor (decision 1) and adds the drift reference τ to the bar (§3.2). Fable should confirm this
   is not a reopening of decision 1.
3. `06_CRITIC.md` M1's second refutation counts poles ("does not move more poles toward code and math"). The
   spec registers band means of the per-pole share instead. The pole count is not a registered quantity.
4. `06_CRITIC.md` M1 plans "two adjacent checkpoints of the main stage-2 run as the identity pair" and "about 8
   GPU-h" (lines 827–828). Decision 1 replaced that pair. At measured L40S rates, five harvests are 9.4 GPU-h before K0 and
   the references.
5. `M1_ASSETS.md` change 2 asks for a gate on "the loaded weights hash equal to `main`". The spec checks the
   file sha256 of every downloaded shard against the registered oids and the stale set (G-SHARD), and reads
   tensors back (G-CAST). It does not hash the loaded model, which would add a 14.6 GB pass and no
   information.
6. `M1_ASSETS.md` change 1 says "the harvest window is 2,048 tokens". The forward is 2,049 positions (BOS plus
   2,048). That is still under 4,096; section WIN checks it.
7. `M1_ASSETS.md` §1 table gives the mix config as "same config blob (sha1 `387607fd…`)". That is the sha1 of
   the raw file. The git blob id is `a9df2fe3b454`. This is cosmetic.
8. `M1_ASSETS.md` §6 cites the corpus at `results/neurons/w1/q4b/1/out/corpus/corpus/`. The grid harvests
   read `results/mdl/neurons/w1/q4b/1/out/corpus/corpus` (`steps_grid_{a,b}_mdl.json`), and so did the AWS
   mechanics job of 2026-11-04. The spec uses the `mdl` path. Neither was listed here.
9. The K0 tolerance rests on cross-device numbers quoted in `studies/already-clean/JOURNAL.md` (median
   0.43 %, p99 2.22 %). Their artifact `results/neurons/w3/device_check/t14_1b_vs_1c.json` is on S3 only and
   was not read.
10. The coordinator cited `00_ASSETS.md` lines 22–39 for the grid table. In this branch's copy the table is
    lines 29–46.

## Files created

- `studies/next/07_PICKS/M1_SPEC.md`: the pre-registration.
- `studies/next/07_PICKS/M1_OFFLINE_GATE.md`: this file.
- `infra/pg/m1.py`: the registry, gates, directory counts, references, freeze and verdict (laptop CLI).
- `infra/aws_pg/jobs/m1_harvest.py`: the job and the two loader fixes (config rewrite in a local copy;
  registered shards only; bf16 cast on load, checked by G-CAST).
- `infra/aws_pg/jobs/m1_dryrun.py`: this offline gate.
- `infra/aws_pg/steps/next_m1_k0-01.json`: L1 (K0, references).
- `infra/aws_pg/steps/next_m1_mix-a-01.json`: L2 instance A (start_a, M, R).
- `infra/aws_pg/steps/next_m1_mix-b-01.json`: L2 instance B (start_b, G).

Scratch only, outside the repo: `scratchpad/m1_venv/`, `scratchpad/m1_gate/` (the HF reads, scripts, dry-run
outputs and logs).

---

## K0 record (attempt `k0-01`, 2026-11-05)

Appended by the Opus executor, 2026-11-05. The text above is freeze 1 (tag `m1-freeze-1`, commit `8f34b6629`) and
is unchanged. `B` = `s3://pg-lineage-769063704696`. Every AWS call used `--profile fuelos --region us-east-1`.
Saved copies of the S3 reads are under `scratchpad/k0_checks/`.

### Pre-launch checks (`M1_SPEC.md` §9.1, §9.5 S1), 2026-11-05T16:14Z to 16:17Z

All passed.

| # | check | command or source | result |
|---|---|---|---|
| 1 | worktree at freeze 1, clean | `git rev-parse --short=12 HEAD`; `git status --porcelain --untracked-files=all` | `8f34b66297a6`; no change and no untracked file (so `pg`'s tarball of the working-tree `infra/` equals the commit) |
| 2 | no stray instance | `./infra/aws_pg/pg ps`; `./infra/aws_pg/pg audit` (16:14:30Z) | one instance, `i-03e36825423328376`, job `next_pilot0_loss`, with a reaper; audit "1 running, 1 reapers, 0 orphans, 0 reaper-less". Not an M1 instance. |
| 3 | attempt prefixes empty | `aws s3 ls $B/results/next_m1/` | empty (exit 1) before any upload; `…/k0-01/` empty again just before the steps file was copied |
| 4 | corpus | `aws s3 ls $B/results/mdl/neurons/w1/q4b/1/out/corpus/corpus/` and `…/tok/olmo3/` | `manifest.json` 55,200 B; `tok/olmo3/tokens.npy` 202,163,292 B; `tok/olmo3/doc_offsets.npy` 278,448 B |
| 5 | corpus manifest sha256 | copy of the S3 `manifest.json` (`scratchpad/k0_checks/corpus_manifest_s3.json`) | `0c99a8e86856f1d9…`, equal to `m1.SAME` and to the laptop copy `scratchpad/m1/corpus_manifest.json` |
| 6 | six references: `topk/` | `aws s3 ls $B/results/neurons/w3/grid/<it>/topk/ \| wc -l` | 32 for each of `1/out/t00_step0`, `1b/out/t12_step320000`, `1b/out/t13_step560000`, `1b/out/t14_step1000000`, `1c/out/t14_step1000000`, `1/out/t15_step1413814` |
| 7 | six references: `controls.json`, `manifest.json` | `aws s3 ls …/controls.json`, `…/manifest.json` | present for all six (2,899–2,922 B and 2,998–3,108 B) |
| 8 | six references: corpus identity (asserted again in the `refs` step) | copies in `scratchpad/k0_checks/refman/` | every manifest has `tokens_sha256` `2aca80ca…` and `corpus_manifest_sha256` `0c99a8e8…`, 32 layers. Branch, device, NLL on the first 120,000 tokens: t00 `stage1-step0` L40S 12.2234; t12 `stage1-step320000` RTX PRO 5000 2.8164; t13 `stage1-step560000` RTX PRO 5000 2.7708; t14b `stage1-step1000000` RTX PRO 5000 2.7148; t14c `stage1-step1000000` L40S 2.7147; t15 `stage1-step1413814` L40S 2.6754 |
| 9 | the five laptop bank files before upload | `shasum -a 256` on `BANK/…` | equal to `m1.BANK_SHA` for all five |

### Upload of the five bank files (2026-11-05T16:15:38Z to 16:15:46Z)

Uploaded with `aws s3 cp` to `$B/results/next_m1/inputs/`. Each was checked three ways: the S3 `ContentLength`
equals the laptop size; the copy downloaded back into `scratchpad/k0_checks/inputs_back/` has the registered
sha256; and for the three files below the 8 MB multipart threshold, the S3 ETag equals the laptop md5. The two
`contexts.jsonl` files went up in 2 parts, so their ETags are multipart and are not md5 values.

| S3 name | laptop source (`BANK/`) | bytes | sha256 (laptop = downloaded copy) | ETag | laptop md5 | S3 version id |
|---|---|---|---|---|---|---|
| `t15_contexts.jsonl` | `1/out/t15_step1413814/samples/contexts.jsonl` | 16,502,054 | `a4d46cd837f7a753…` | `6c1182b4…-2` (multipart) | `f7f4612e…` | `DYjL29LZ3qCZCdiMFqEbc2vUg.k95SiI` |
| `t15_exemplar_a.npz` | `1/out/t15_step1413814/poles/exemplar_a.npz` | 213,844 | `84bd59fdaf6b4572…` | `d8462b1b8f43b3a3420dcf160412a7dd` | equal | `1A1MtfTBfZX8M5dzWqEKR8bgc8AwnmdD` |
| `t14_exemplar_a.npz` | `1b/out/t14_step1000000/poles/exemplar_a.npz` | 220,781 | `9fabd78be6d81b5c…` | `c076645621bd22343f0efe41681245b0` | equal | `uDPhb2SSod.7AZZk_hyzl_PkQDqv5nI0` |
| `t14_contexts.jsonl` | `1b/out/t14_step1000000/samples/contexts.jsonl` | 16,440,329 | `692e85854bebb544…` | `e852d881…-2` (multipart) | `0f124b80…` | `5xBMlU08eDvQGxPIBMgyOn_D4LV0fXdE` |
| `t13_exemplar_a.npz` | `1b/out/t13_step560000/poles/exemplar_a.npz` | 226,190 | `7976ca719702c6cd…` | `bfc1b3110b960c98b9945df1083127fd` | equal | `6BnW4j6_g56HP2SV1NIPyudJvuxxSx1M` |

### Launch

`aws s3 cp infra/aws_pg/steps/next_m1_k0-01.json $B/results/next_m1/k0-01/steps.json`, then the first command of
this file with `--detach` added, from the worktree root:
`./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_m1/k0-01/steps.json" --no-weights --type g6e.2xlarge --s3-prefix results/next_m1/k0-01 --budget-min 40 --json --detach`.
Instance `i-047efd8fa80c4254c`, g6e.2xlarge, us-east-1c, EC2 `LaunchTime` 2026-11-05T16:17:34Z, reaper at
17:07:34Z. The attempt is `k0-01`; no earlier K0 attempt exists. Operations: `infra/aws_pg/operations/2026-11-05_next_m1_k0.md`.

### Result: K0 passed, bit-identical (2026-11-05T16:31:38Z)

**K0 passed. All 1,535 re-harvested exemplar activations equal the banked T_15 values bit for bit (median and
p99 relative difference 0.0; tolerance 0.0043 and 0.0222). Every banked top-50 position inside the prefix is in
the prefix run's top-50 (1,535 of 1,535; rule ≥ 0.99). Both failing inputs failed.** STATUS `exit=0`; both
pipeline steps `ok`. Source: `studies/next/m1/banks/k0-01/k0/m1/metrics.json` → `K0` (sha256 `872b2570…`),
copied from `$B/results/next_m1/k0-01/out/k0/m1/metrics.json`.

| rule (`M1_SPEC.md` §6) | value | line | failing input | its value | result |
|---|---|---|---|---|---|
| 1. values, 1,535 entries on 439 poles | median 0.0, p99 0.0, max 0.0; exact share 1.000 | median ≤ 0.0043, p99 ≤ 0.0222 | T_14's banked values (`t14_exemplar_a.npz`) | median 0.31442, p99 1.50210, max 2.43321 | pass; failing input fails |
| 2. subset | 1,535 / 1,535 = 1.000 | ≥ 0.99 | bank positions + 1 | 208 / 1,535 = 0.1355 | pass; failing input fails |
| 3, 4. failing inputs fail | both fail | | | | holds |

The other gates of the `k0` step passed, each with its failing input failing:
- G-SAME: torch `2.12.0+cu130`, transformers `5.9.0`.
- inputs: the five sha256 values.
- G-CORPUS.
- G-REV: snapshot `373bad25…`.
- G-SHARD: 3 bf16 shards; `main`'s stale shards fail.
- G-CFG: `diff_keys` [], `sliding_window` 4096; the retrofit config fails.
- G-CAST: all parameters bf16; 7 of 7 tensors equal the round-to-nearest-even cast; the next-layer tensors fail.
- G0: pass.

**Laptop recount, own code** (`scratchpad/k0_checks/k0_recount.py`, from the files, not from `m1.k0_values_gate`).
It reproduces median 0.0, p99 0.0, exact share 1.0 and the failing input's 0.31442 / 1.50210. The script was
first run on two stand-ins. The bank against itself gave 0.0 and passed. T_14's values as the re-harvest gave
0.31442 / 1.50210 and failed. So it can show a failure.

**Not a copy of the bank.** The re-harvested `poles/exemplar_a.npz` is a different file: sha256 `323e0e8c…`,
20,597 B, against the bank's `84bd59fd…`, 213,844 B. It holds NaN at the 30,465 slots outside the prefix and
numbers at the 1,535 inside (1,535 + 30,465 = 32,000).

**Ten compared entries.** The largest relative difference (all are 0) and nine drawn with `default_rng(0)`:

| pole | slot | position | token | a_bank | a_new | relative difference |
|---|---|---|---|---|---|---|
| L31N10276- | 31 | 105,070 | "azing" | -60.0602 | -60.0602 | 0 |
| L27N10663- | 7 | 807,622 | " face" | -9.07348 | -9.07348 | 0 |
| L21N3982+ | 40 | 879,628 | " sign" | 10.329 | 10.329 | 0 |
| L17N7622- | 9 | 465,267 | " it" | -1.33175 | -1.33175 | 0 |
| L12N727- | 37 | 36,143 | " rewarding" | -0.65045 | -0.65045 | 0 |
| L08N5063+ | 3 | 389,277 | " Hitch" | 0.615606 | 0.615606 | 0 |
| L10N9100- | 1 | 938,714 | " a" | -0.650489 | -0.650489 | 0 |
| L03N10451- | 24 | 1,035,696 | " Bill" | -0.168784 | -0.168784 | 0 |
| L01N9435+ | 42 | 476,576 | " Introduction" | 0.137636 | 0.137636 | 0 |
| L01N369+ | 24 | 702,592 | " forfeiture" | 0.346285 | 0.346285 | 0 |

The job's own two samples ("closest", "farthest") are the same entry, L00N181+ slot 0, token 327,203, a =
0.327806 on both sides. With every difference equal to 0, `argmin` and `argmax` both return the first entry. The
table above holds the samples instead.

**The corpus NLL control equals the bank exactly.** The mean NLL over the first 120,000 tokens is
2.675372037760417 nats per token in the K0 run and in the banked T_15 `controls.json`. This is the known number
named in `09_PLAN.md`'s J7 card ("reproduces 2.675").

**Positions + 1: in-job 0.1355 against offline 0.038.** The two are built differently. The job compares the
shifted positions with the prefix run's top-50, and all 50 of those positions lie inside the prefix: 208 / 1,535.
The offline gate compared them with the bank's global top-50, of which about 2.4 per pole lie inside the
prefix: 58 / 1,535 = 0.0378. Both counts were recomputed here from the two `contexts.jsonl` files. The layer-5
`topk` file was checked to equal the contexts' top-50 for its 20 grid poles. Both values are far below 0.99.

**Measured rates (`studies/next/m1/banks/k0-01/k0/manifest.json`, `m1/download.json`):**
- Harvest: 3,337.75 tokens/s (loop 314.0 s over 1,048,064 tokens); T_15's banked rate is 3,334.
- Download: 76.0 MB/s (14,605,764,269 B in 192.2 s). That is a third of the registered 234 MB/s and below the
  lowest banked 143 MB/s.
- sha256 of the shards 9.2 s; load 51.5 s; final write 137.1 s; snapshot none (K0 has no snapshot band).
- Boot: `LaunchTime` 16:17:34Z to the first `pipeline_status.json` 16:18:49Z, 75 s (the spec assumed 164 s).

### The references step (`refs`, 16:31:58Z to 16:33:05Z)

`gates_pass` true. Each reference manifest's `tokens_sha256` and `corpus_manifest_sha256` equal the registered
values (asserted). Each count file has 50 directories, 5 of them code+math, and its counts sum to its filled
slots.

| ref | branch | device | NLL, first 120k | poles with < 50 filled slots | `counts.npz` sha256 | S3 version id |
|---|---|---|---|---|---|---|
| t00 | `stage1-step0` | L40S | 12.22338 | 0 | `076c074c7987825e…` | `Lphb.x1nTSf8I7m3tqLpmlHgaS_VZNay` |
| t12 | `stage1-step320000` | RTX PRO 5000 | 2.81642 | 0 | `5ef992a8aaa9574d…` | `mxCI7IXspxbf5wmx7F_yfTLbJBRYf7BY` |
| t13 | `stage1-step560000` | RTX PRO 5000 | 2.77081 | 0 | `934c10e5803386bb…` | `BdmH71_5TTxg9Grfo_oa1MEOXJ7q5zOM` |
| t14b | `stage1-step1000000` | RTX PRO 5000 | 2.71476 | 0 | `3eb8a29937c079bf…` | `xmsrJ8r2gTQ.6mCLUrxV5GwdcNUfB.wL` |
| t14c | `stage1-step1000000` | L40S | 2.71470 | 0 | `e91c2469f4acdaf8…` | `KgB44hnr5ZKcbLjIdxQJrAFj9BzRYuGY` |
| t15 | `stage1-step1413814` | L40S | 2.67537 | 2: `L05N10160-` (7), `L07N9578-` (2) | `935c7b710db3de73…` | `NG.1N_U1BU7TvutIgG9P5RA5i0olnVgN` |

The two thin t15 poles have filled slots, so `m1.share` (which asserts filled > 0) holds. Each has a share in
steps of 1/7 or 1/2. No cm_share was computed here. The count files are on the laptop at
`studies/next/m1/banks/refs/<ref>/m1/counts.npz`, untracked (9.5–10.7 MB each). The S3 copies above are on the
versioned, deny-delete bucket.

### Cost and the stop rule (`M1_SPEC.md` §9.7)

- L1 used about 0.27 GPU-h (bounds 0.26 to 0.30): `LaunchTime` 16:17:34Z to `shutting-down`, first seen between 16:33:41Z and 16:35:50Z; STATUS at 16:33:09Z. The
  projection was 0.26. At $2.24 per hour it cost $0.59 to $0.68. With 0.30 in place of 0.27 the total below is 10.05.
- Recomputed with K0's measured 3,337.75 tokens/s and 76.0 MB/s, using the §9.7 per-harvest terms
  (`scratchpad/k0_checks/project.py`, which reproduces the spec's 9.73 and 10.15):
  - bf16 harvest 1.889 h, fp32 harvest 1.959 h;
  - `mix-a-01` 5.85 h, `mix-b-01` 3.89 h;
  - L1 + L2 = **10.02 GPU-h**, below the stop rule's 11.7. No park is needed.
- `mix-a-01`'s kill limit is 390 min = 6.5 h, against 5.85 h projected plus boot. The margin is 0.6 h. If the
  download stays at 76 MB/s, each fp32 download takes 384 s instead of 125 s.

### Not done here

Freeze 2 (`M1_SPEC.md` §9.5 S3, `python -m infra.pg.m1 freeze`) was not run. `thresholds.json` and the
`m1-freeze-2` tag do not exist. No mix harvest was launched.

---

## Freeze 2 (`thresholds.json`, 2026-11-05)

Appended by the Opus executor of the harvests, 2026-11-05. The text above is unchanged. No mix harvest existed
when this was run; only the six reference count files were read.

**Freeze 2 passed: `M1_FREEZE_OK`. The bar is τ in every band (0.0082, 0.0066, 0.0023, 0.0035 in fractions of
top-50 slots), and both the SUPPORTED and the REFUTED branch fire on real reference pairs in every band.**

### The command

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-m1
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.m1 freeze --banks studies/next/m1/banks --out studies/next/m1/thresholds.json
```

- Run 2026-11-05T16:43:13Z to 16:44:06Z (53 s), exit 0, on HEAD `ad8adac8a` (`m1-freeze-1` code; `infra/pg/m1.py`
  unchanged since `8f34b6629`). Log: `scratchpad/k0_checks/freeze2.log`.
- Printed: `M1_FREEZE_OK {"tau": [0.008198923510174372, 0.006556368095930229, 0.0022529069767441845, 0.0035377679869185857], "eps_dev": [2.2597202034929653e-05, 2.8388444767435406e-06, 2.8161337209314796e-05, 1.7487281976737157e-05], "kappa": [0.1832627407340116, 0.17685262990552328, 0.1921298146802326, 0.2021358330305233]}`.
- Output: `studies/next/m1/thresholds.json`, 5,862 B, sha256 `4bb56ded950396cdf5e9a13f1581c315048666b7ed3fdef4982c77a5e737f31d`.
- Inputs: `studies/next/m1/banks/refs/<ref>/m1/counts.npz` (fields `counts`, `filled`, `cm_mask`). Their sha256 were
  read again before the run and equal the K0 record's table and `thresholds.json` → `refs_sha256`. They are the
  files the `refs` step of `k0-01` wrote (S3 version ids in the K0 record). They stay untracked.
- The stop rule (`M1_SPEC.md` §9.7) at the measured 3,337.75 tokens/s and 76.0 MB/s:
  `scratchpad/k0_checks/project.py 3337.75 76.0 0.27` gives mix-a 5.85 h, mix-b 3.89 h, L1 + L2 = 10.02 GPU-h,
  below 11.7. No step is parked.

### The full `thresholds.json`

Every key and value of the file, printed one array or one room pair per line (the file itself uses
`json.dumps(indent=1)`, 378 lines). Bands are layers 0–7, 8–15, 16–23, 24–31. Units: fractions of a pole's
filled top-50 slots.

```
"tau": [0.008198923510174372, 0.006556368095930229, 0.0022529069767441845, 0.0035377679869185857]
"eps_dev": [2.2597202034929653e-05, 2.8388444767435406e-06, 2.8161337209314796e-05, 1.7487281976737157e-05]
"kappa": [0.1832627407340116, 0.17685262990552328, 0.1921298146802326, 0.2021358330305233]
"bar_before_identity": [0.008198923510174372, 0.006556368095930229, 0.0022529069767441845, 0.0035377679869185857]
"C_refs": {
  "t00": [0.3056304505813953, 0.2815319540334303, 0.26641385810319773, 0.2534778115915698]
  "t12": [0.12236770984738371, 0.10999352743459304, 0.07520621366279069, 0.0545030886627907]
  "t13": [0.12645189861918607, 0.11553358920784884, 0.07745912063953488, 0.05635776253633721]
  "t14b": [0.13465082212936044, 0.10897722111191861, 0.0756175054505814, 0.052819994549418624]
  "t14c": [0.13467341933139537, 0.10897438226744187, 0.07564566678779071, 0.05283748183139536]
  "t15": [0.14230716946117114, 0.10467932412790701, 0.07428404342296513, 0.05134197856104651]
}
"room": {
  "pairs": [
    {"as_G": "t00", "as_M": "t12", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t00", "as_M": "t13", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t00", "as_M": "t14b", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t00", "as_M": "t14c", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t00", "as_M": "t15", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t12", "as_M": "t00", "labels": ["ABOVE_CEILING", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t12", "as_M": "t13", "labels": ["REFUTED", "REFUTED", "UNDECIDED", "REFUTED"]}
    {"as_G": "t12", "as_M": "t14b", "labels": ["SUPPORTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t12", "as_M": "t14c", "labels": ["SUPPORTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t12", "as_M": "t15", "labels": ["SUPPORTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t13", "as_M": "t00", "labels": ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t13", "as_M": "t12", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t13", "as_M": "t14b", "labels": ["UNDECIDED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t13", "as_M": "t14c", "labels": ["UNDECIDED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t13", "as_M": "t15", "labels": ["SUPPORTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14b", "as_M": "t00", "labels": ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t14b", "as_M": "t12", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14b", "as_M": "t13", "labels": ["REFUTED", "UNDECIDED", "REFUTED", "UNDECIDED"]}
    {"as_G": "t14b", "as_M": "t14c", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14b", "as_M": "t15", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14c", "as_M": "t00", "labels": ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t14c", "as_M": "t12", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14c", "as_M": "t13", "labels": ["REFUTED", "UNDECIDED", "REFUTED", "UNDECIDED"]}
    {"as_G": "t14c", "as_M": "t14b", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t14c", "as_M": "t15", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t15", "as_M": "t00", "labels": ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t15", "as_M": "t12", "labels": ["REFUTED", "REFUTED", "REFUTED", "UNDECIDED"]}
    {"as_G": "t15", "as_M": "t13", "labels": ["REFUTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"]}
    {"as_G": "t15", "as_M": "t14b", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
    {"as_G": "t15", "as_M": "t14c", "labels": ["REFUTED", "REFUTED", "REFUTED", "REFUTED"]}
  ]
  "supported_reachable": true, "refuted_reachable": true, "tau_below_kappa": true
}
"refs_sha256": {
  "t00": "076c074c7987825e402b099b9aef37c17799e77e91e475dcf45c33da3b8914ed"
  "t12": "5ef992a8aaa9574d9d924bd6e4abf68c8c29c0a2a1ef1b61d8b355d63ee38791"
  "t13": "934c10e5803386bb4e91370d57c93875b91d3271a45ff8bc7139bd8b87c7a4a1"
  "t14b": "3eb8a29937c079bfb4580a1c544158e66c6668398f68f67365ee0b8c6bbf6189"
  "t14c": "e91c2469f4acdaf884c00e4f20a07fbef277b0b53d9bf00631be056283d4a2f7"
  "t15": "935c7b710db3de735c3370c354117c500503a10b0c76e50405ba77ee5f5e435a"
}
```

### Where each threshold comes from

C_r(b) is `m1.band_means(m1.share(counts))` of reference r: the mean over the 176,128 poles of band b of
(slots in the 5 code+math directories) / (filled slots). Every file below is
`studies/next/m1/banks/refs/<r>/m1/counts.npz`, fields `counts` [32, 11008, 2, 50], `filled` [32, 11008, 2] and
`cm_mask` [50].

| threshold | rule (`m1.freeze`) | control data | the pair that sets it, by band 0 / 1 / 2 / 3 |
|---|---|---|---|
| τ(b) | max over `REF_PAIRS` of \|C_j − C_i\| | `t12`, `t13`, `t14b` (RTX PRO 5000), `t14c`, `t15` (L40S); pairs t12–t13, t13–t14b, t14c–t15 | t13–t14b / t13–t14b / t12–t13 / t13–t14b. The other pairs: t12–t13 0.004084, 0.005540, 0.002253, 0.001855; t14c–t15 0.007634, 0.004295, 0.001362, 0.001496 |
| ε_dev(b) | \|C_t14c − C_t14b\| | `t14b`, `t14c`: `stage1-step1000000` harvested on an RTX PRO 5000 and on an L40S | the one pair; 2.3e-5, 2.8e-6, 2.8e-5, 1.7e-5 |
| κ(b) | max over all 30 ordered pairs of the six references of \|C_i − C_j\| | all six, `t00` (random init) included | t00–t12 / t00–t15 / t00–t15 / t00–t15 |
| bar before identity | max(τ, ε_dev) | the two rows above | equals τ in every band (ε_dev is 80 to 2,310 times smaller) |
| G-ID line (verdict) | min_b τ(b) / 4 | τ | 0.0022529 / 4 = 0.000563 (band 2) |
| bar at the verdict | max(τ, ε_dev, \|I\|) | τ above; I from start_a and start_b | written at the verdict |

PC_NLL's floor (max(0.01, \|identity NLL difference\|) nats) and ceiling (2.0 nats) and H_specific's random sets
are registered constants of `infra/pg/m1.py`, not part of `thresholds.json`.

### Room for each branch, on the references (no mix data)

Scripts: `scratchpad/m1_run/freeze2_check.py` and `scratchpad/m1_run/freeze2_between.py`, both calling the frozen
functions of `infra/pg/m1.py`; logs beside them (`freeze2_check.log`, `freeze2_between.log`).

| rule | pass branch | fail branch | does-not-count branch |
|---|---|---|---|
| lead label (`label_lead`, bar = τ, ceiling κ), 30 ordered reference pairs × 4 bands = 120 labels | SUPPORTED 26 times, in every band (8, 6, 6, 6); for example t15 as G and t13 as M in bands 1–3 | REFUTED 85 times, in every band (19, 22, 23, 21); for example t14c as G and t15 as M in all four bands | ABOVE_CEILING ("DOES NOT COUNT until inspected") once: t12 as G, t00 as M, band 0. That mean is the pair that defines κ(0); it exceeds κ by 2.8e-17, a float rounding. So the branch is reachable only by a contrast at least as large as the change from random init to T_12. UNDECIDED 8 times |
| G-ID (max_b \|I(b)\| ≤ 0.000563) | the banked cross-device twin t14b/t14c passes: I = 2.26e-5, −2.8e-6, 2.82e-5, 1.75e-5 | neighbouring checkpoints fail: t14c/t15 max \|I\| 0.00763, t13/t14b 0.00820, t12/t13 0.00554 | its failure is the DOES NOT COUNT branch "harvest not repeatable at the scale of the bar". The registered failing input at the verdict is start_a against t15 |
| between (`label_between`, bar = τ), ordered triples (G, R, M) of t00, t12, t13, t14b, t15 × 4 bands = 240 labels | SUPPORTED 6 times, for example (t12, t14b, t00) in band 0 | REFUTED 126 times, for example (t00, t12, t13) in band 0 | UNDECIDED 108 times. This rule has no does-not-count branch of its own; it is read only in bands where the lead is SUPPORTED |
| τ < κ (freeze assert) | holds in every band: the smallest κ/τ is 22 (band 0) | the CLI refuses to write the file otherwise | — |

The interval is narrow enough that UNDECIDED needs a mean near the bar. On the reference pairs, the 95 %
bootstrap half-width of a band contrast is 0.00024 to 0.00060 (all poles, 88,064 neurons per band), against a
bar of 0.0023 to 0.0082.

**Notes.**
- τ on all poles is not τ on the 640 grid poles. The 640-pole values of freeze 1 (§ROOM) were 0.0134, 0.0025,
  0.0035, 0.0098; on all poles they are 0.0082, 0.0066, 0.0023, 0.0035. The frozen rule uses all poles.
- The banked cross-device twin changes the counts of 58.5 % of poles (`changed_pole_share` 0.5848) while its
  band means move by at most 2.8e-5. A same-type twin is expected to change none (`M1_SPEC.md` §1.4 prior 0.8).
- The author's prior for Δ_MG is +0.01 to +0.06 (`M1_SPEC.md` §1.4). It lies above τ in every band and below κ.
