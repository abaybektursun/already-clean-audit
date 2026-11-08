# N22 offline gate

**The weight-free dry run printed `N22_DRYRUN_OK`: 100 checks, 0 failed** (2026-11-06T15:09:15Z–15:14:26Z; `studies/next/n22/offline_gate/dryrun.log`,
values in `dryrun_summary.json`). It ran the job from its entry point on a tiny random OLMo-3 (8 blocks, hidden 64,
the real tokenizer and vocabulary) in all three launch modes, showed every registered failing input failing,
stopped on every planted failure naming the gate, parked under the stop rule, and produced every label branch on
planted records. The five gates whose quantity a bank holds pass on their pass side and fail on their failing input
at all 16 points (`studies/next/n22/offline_gate/gate_sides.json`). The other gates have no bank; their two sides
were shown on the tiny world only. Go for L1 once the head has reviewed and freeze 1 is tagged.

`SP` below means
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.

## What was run (2026-11-06, worktree `interp_research-n22`, branch `n22`)

```
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n22 items --tok $SP/n22/hf/tok --out studies/next/n22/items.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n22 project > studies/next/n22/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n22/samples/raw_examples.py --tok $SP/n22/hf/tok --out studies/next/n22/samples
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n22/offline_gate/gate_sides.py --tok $SP/n22/hf/tok \
    --tok2 $SP/pilot0_in/tokenizer.json --out studies/next/n22/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n22_dryrun --hf $SP/n22/hf \
    --sides studies/next/n22/offline_gate/gate_sides.json --out $SP/n22/dry
```
CPU torch 2.12.0 and transformers 5.9.0 (the AMI's). `$SP/n22/hf` holds `tok/` (the OLMo-3 tokenizer files of
`main`, sha256 of `tokenizer.json` `73fd5254…`) and `config_stage1-step476838.json`.

## Fetched (small files only, read-only)

- S3: the 16 grid harvests' `controls.json` and `manifest.json` (`results/neurons/w3/grid/{1,1b}/out/tNN_*/`); the T_14
  L40S re-harvest's `controls.json` and `manifest.json` (`grid/1c/out/t14_step1000000/`); M1's T_15 known-number run
  (`results/next_m1/k0-01/out/k0/{controls,manifest}.json`); the corpus manifest
  (`results/mdl/neurons/w1/q4b/1/out/corpus/corpus/manifest.json`).
- Hugging Face: the refs of `allenai/Olmo-3-1025-7B` (one API call, sha256 in `gate_sides.json`); `config.json` of
  `stage1-step{0,1000,2000,3000,5000}` (committed as `studies/next/n22/offline_gate/hf/`; the other 11 configs are
  M2's committed copies in `studies/next/m2/offline_gate/hf/`).
- Laptop: M1's `stage1-step476838` identity pair (`studies/next/m1/banks/start_{a,b}/controls.json`).

## The bank-side gates, both sides at every point (`gate_sides.json`)

| gate | pass side | failing input |
|---|---|---|
| K0 | control pairs: T_15 harvest against M1's K0, both L40S: 0.0; `stage1-step476838` start_a against start_b, both L40S: 0.0; T_14 Blackwell against L40S: 6.1683e-05 (within the coarse tier 0.01 and the reported reference 1.234e-04). The 16 registered values (`n22.K0_BANK`) equal the banked ones. | the neighbouring point's bank, at every point: gap over the tier 3.4 (T_11), 3.9 (T_14), 4.4–7.5 (T_7–T_13 otherwise) at the coarse points; 39,390 (T_15) to 7.7 million (T_0) at the same-device points |
| G-REV | 16 of 16: registered = banked manifest's `revision` = the HF ref today | forty zeros equal none |
| G-CFG | 16 of 16 branch configs pass `m1.config_gate` with no differing key | `model_type` renamed fails at all 16 |
| G-CORPUS | the registered tokens sha256 `2aca80ca…` = all 16 manifests = the corpus record | a zero sha |
| G-TOK | the tokenizer of `main` and of `stage1-step60000` are byte-identical; their vocabulary sha256 `2f3c015e…` = the corpus record's = `items.json`'s; all 1,200 items, 2,400 counterfactuals and 5,056 class prompts re-encode to the registered ids | item 0's ids against item 1's |

K0 per point (the neighbour's gap and its ratio to the tier in force):

| point | tier | neighbour gap | gap / tier |
|---|---|---|---|
| T_0 | 1e-6 | 7.659979 | 7.7e6 |
| T_1 | 1e-6 | 0.627704 | 6.3e5 |
| T_2 | 1e-6 | 0.278543 | 2.8e5 |
| T_3 | 1e-6 | 0.214622 | 2.1e5 |
| T_4 | 1e-6 | 0.154702 | 1.5e5 |
| T_5 | 1e-6 | 0.100189 | 1.0e5 |
| T_6 | 1e-6 | 0.099633 | 1.0e5 |
| T_7 | 0.01 | 0.074955 | 7.5 |
| T_8 | 0.01 | 0.073243 | 7.3 |
| T_9 | 0.01 | 0.043817 | 4.4 |
| T_10 | 0.01 | 0.045643 | 4.6 |
| T_11 | 0.01 | 0.033929 | 3.4 |
| T_12 | 0.01 | 0.045618 | 4.6 |
| T_13 | 0.01 | 0.056044 | 5.6 |
| T_14 | 0.01 | 0.039390 | 3.9 |
| T_15 | 1e-6 | 0.039390 | 3.9e4 |

The degenerate-pole scan of M2 (`studies/next/m2/offline_gate/bank_scan.py`) was not run: N22 reads no pole statistic,
so it has no quantity to check here.

## Gates with no bank, and how each was shown

| gate | why no bank | dry run |
|---|---|---|
| I-GRAD | no lens of these checkpoints exists | tiny world: the repeated pass bit-identical, prompt 1 differs (relative 0.544); a planted noisy repeat stops the job |
| LENS_FIDELITY | no lens | tiny world: cos 0.906 against 0.097 for the rotated residual; planted random matrices in place of J stop the job |
| I-FWD, I-EDIT, G-IC-ID | no fp32 forward of these prompts | tiny world: each 0.0 exactly; planted +0.01, a nonzero zero-edit and shifted operand positions each stop the job |
| the interchange control, PC-ANS, the floors | no forward | VERDICT: planted records fire and do not fire as registered |
| the probes, N25's statistics | no lens, no residuals | tiny world: written for every probe layer and lens variant; SCORE: CKA, the block statistic and the kurtosis on known inputs |

## Results by section

The full list is the log. In short:

- **REAL** (16 checks): `items.json` has the registered sha256 and reproduces byte for byte; the pool's arithmetic,
  distinctness, targets, counterfactuals, partners and class prompts hold for every item; the five bank-side gates
  hold both sides at 16 points; K0's smallest failing-input margin is 3.39 times the tier; the real registry carries
  no dry-run key, the registered revisions, target block 30, sources 0–29, the upper half 16–29, 24 lens prompts.
- **SCORE** (15 checks): the interaction on known numbers and its invariance to a log-partition shift; the lens swap
  exchanges the two coordinates and leaves the orthogonal part unchanged, and is exactly zero with σ the identity;
  the random floors keep the norm per position; ranks with ties; linear CKA (1 on itself and under rotation), the
  job's centred-Gram path equals it; the block statistic; excess kurtosis (Gaussian 0.003, Laplace 3.005); the
  bootstrap; the band rule with ties; the readout rule; the factorizations.
- **JOB** (27 checks): L1 (15:full) exits 0, complete, every gate passes and all 13 failing inputs fail; K0 equals the
  bank written by `neurons_harvest.run` bit for bit (11.535791015625); I-GRAD bit-identical; I-FWD, I-EDIT, G-IC-ID 0.0;
  every record file written; one edit row per (item, v, arm, slot) (4,800 in the test half of 60 items); the
  interchange at sites −1, 0, 4 with both kinds; an item's interaction recomputes from the records; ten raw examples
  with the prompt, top-10 readouts and every arm; the probes; N25's four variants. L2 (accuracy at three points)
  exits 0 with K0 and the counts at each; `plan-lens` on its scan. L3 (n25 + full + full) exits 0; the n25 point has
  the lens, the readout and N25 but no edits.
- **FAIL** (14 checks): G-SAME, INPUTS, G-CORPUS, G-REV, G-CFG, K0, K0 with a tolerance so wide its failing input
  passes ("its failing input passed"), I-GRAD, LENS_FIDELITY, I-FWD, I-EDIT, G-IC-ID, G-TOK: each stops the job naming
  the gate; the stop rule parks T_14 and T_13 and the job ends normally.
- **VERDICT** (27 checks): freeze 2 and the point verdict on the tiny L1 banks (every field, including the freeze half's DISCOVERY values of every arm with their per-item spread; the tiny model reads
  nothing, so LENS_M is "no matched layer" and LENS_U "the positive control did not fire": both branches exercised on
  the real path); the verdict refuses a test file changed after freeze; ABOVE_FLOORS, AT_FLOOR, BELOW_FLOOR, "above the
  random floor but at the shuffle", above the ceiling, fewer than N_MIN; room at 50 items (1 nat 20 of 20, 0.5 nat
  20 of 20, 0.25 nat 14 of 20); the interchange fires and does not; GO, NO-GO at 99, NO-GO without the control;
  H_order SUPPORTED, CENSORED, REFUTED_SAME, REFUTED_CONTRARY, NOT_RUN (twice); the dating arm; N25 LEARNED with
  FORMS_BETWEEN and SINK_DEPENDENT, ARCHITECTURAL with NO_CHANGE, SMOOTH; the massive formation point; plan-lens with
  six, two and no candidates.
- **PROJECT** (1 check): L1 1.05, L2 1.01, L3 5.39 GPU-h at the center rates, 12.29 in all at the high rates; hard kills
  at 1.3 times the high projection (138, 124, 698 min).

## Spec issues found while building the gate (resolved in `N22_SPEC.md` as stated)

1. **The lens's positive control.** A top-1 agreement with the model's output (or with block 30's readout) is 0 on
   a random model and says nothing about the transport. Replaced by the cosine between J_29 h_29 and h_30 against a
   rotated residual (tiny world 0.906 against 0.097; the untransported cosine 0.889 is reported beside it).
2. **DOM_M carries the answer for slot 2.** y = s2 + d, so any class mean of "s2 = v" moves the answer by v − s2
   unless it moves d by the same amount. Resolved by the DoM guard (SPEC §2.6): DOM_M does not count at a band where
   the answer is readable.
3. **The paper's swap positions and layers.** The paper's §A.24.2 swaps every position over the upper half, not each
   slot in its own band; the critic's depth concern still holds because each swap acts where its token is present.
   LENS_U is the paper's protocol; LENS_M the matched-depth test.
4. **No banked forward of any prompt here.** Every N22 quantity other than the five above is new; the dry run is the
   only evidence for those gates' two sides before L1.

## Not verified here, said plainly

- No GPU number exists. The lens rate (78 s per prompt center, 120 s high) is an estimate scaled by FLOPs from N12's
  measured gradient pass; the fp32 forward rate (0.018 s per 63-token prompt at batch 64, 0.040 s high) is an
  estimate. Memory at 7.3 B parameters in fp32 plus the lens vectors is computed, not measured.
- The tiny world checks the code path, the positions, the hooks and every gate. It is not a test of the science: a
  random model answers nothing, so its items were all treated as answered through a dry-run-only registry key
  (`dry_all_answered`, asserted absent from the real registry).
- The tokenizer was checked at `main` and `stage1-step60000` only; the 16 banked harvests each asserted the same
  vocabulary at load. G-TOK checks 20 items at every point on the instance.
- Whether OLMo-3 answers 100 of the 1,200 items at T_15 is unknown; it is the first number L1 produces.
- The papers of §1.5 other than the workspace paper's arithmetic sections were not read in full.

## Files created

| file | what |
|---|---|
| `studies/next/07_PICKS/N22_SPEC.md` | the pre-registration |
| `studies/next/07_PICKS/N22_OFFLINE_GATE.md` | this record |
| `infra/pg/n22.py` | the pool, prompts, scoring, labels, freeze, verdict, plan-lens, projection; CLI |
| `infra/aws_pg/jobs/n22_score.py` | the job |
| `infra/aws_pg/jobs/n22_dryrun.py` | the offline gate |
| `infra/aws_pg/steps/next_n22_l1-01.json`, `next_n22_l2-01.json` | the step lists of L1 and L2 |
| `studies/next/n22/items.json` | the registered pool, class prompts and token ids |
| `studies/next/n22/projection.json` | GPU-hours per launch |
| `studies/next/n22/samples/raw_examples.py`, `raw_examples_zero_gpu.{md,json}` | the ten raw examples, zero-GPU part |
| `studies/next/n22/offline_gate/gate_sides.{py,json,log}` | both sides of the bank-held gates at 16 points |
| `studies/next/n22/offline_gate/hf/config_stage1-step{0,1000,2000,3000,5000}.json` | the five branch configs M2 had not committed |
| `studies/next/n22/offline_gate/dryrun.log`, `dryrun_summary.json` | the dry run's log (progress bars removed) and values |
