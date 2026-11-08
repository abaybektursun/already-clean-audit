# N13: which task is the "MATH 40.0" of Olmo 3 Table 3

**Answer.** Table 3's "MATH" row (40.0 for Olmo 3 7B base) is Minerva MATH on the full 5,000-problem
Hendrycks MATH test split, not MATH-500. Settings: 4 fixed human-written shots in the Minerva
`Problem:`/`Solution:` text format, 4 samples per problem at temperature 0.6 and top-p 0.6, at most 1,024
new tokens, stop at `Problem:` or a blank line. A sample counts as correct under "EM Flex": the answer is
taken by Minerva extraction plus fallbacks, normalized by lm-eval 0.4.3 `normalize_final_answer`, and
matched by sympy equivalence or by Hendrycks `strip_string` equality. The score is pass@1 (fraction of the 4
samples correct) averaged within each of the 7 subjects, then averaged over the 7 subjects without weights.
The olmes task id is **`minerva_math::olmes:n4:v2`**.

No MATH-500 number for the base model is published in any source I could open. For
`allenai/Olmo-3-7B-RL-Zero-Math`, the checkpoint the spec uses, I found no published MATH or MATH-500
number anywhere: the report text, the report tables, the card and its public W&B run give none. Its only
published math numbers are AIME 2024 and 2025. The public W&B project does log MATH-500 for the
**3.1** lineage (`allenai/Olmo-3.1-7B-RL-Zero-Math`): 0.874 to 0.908 exact_match_flex, 0-shot, chat
format, temperature 1.0, 32,768 tokens.

Read-only check, 2026-11-05. No model was run. Sources are tagged [S#] and listed at the end.

## What this means for N13_SPEC

G-KNOWN as written compares the base model's MATH-500 accuracy under the RL-Zero `Answer:` prompt with
40.0 (`N13_SPEC.md` lines 65, 221, 505, 542). Those are two different quantities. They differ on every axis:

| axis | Table 3 MATH (40.0) | N13 base arm |
|---|---|---|
| items | 5,000 (MATH test, 7 subjects) | 500 (MATH-500) |
| prompt | 4-shot Minerva text, no chat template | 0-shot RL-Zero instruction |
| sampling | T 0.6, top-p 0.6, 4 samples | T 0.6, top-p 0.95, seed set A |
| length cap | 1,024 tokens, stop at a blank line | C (16,384 or as frozen) |
| grader | Minerva/Hendrycks EM Flex (lm-eval 0.4.3, sympy) | `math_verify` 0.9.0 |
| aggregation | unweighted mean of 7 subject means | mean over 500 items |

So 40.0 cannot serve as the known number for a MATH-500 run. To use it as a known number, the gate has to
run `minerva_math::olmes:n4:v2` itself (20,000 generations of at most 1,024 tokens). I did not edit the spec.

## Evidence from the report [S1]

All of this holds in both arXiv versions (v1, 2025-12-15, and v2, 2026-04-14). Table 3's 7B column and
Table 46 are the same in both.

1. **Table 3** (7B, "OlmoBaseEval Main suite") has three rows under the Math group: GSM8k 75.5, GSM
   Symbolic 48.6, MATH 40.0. The aggregate "OlmoBaseEval Math" is 54.7.
2. **The aggregate is the plain mean of those three rows.** I recomputed it for all 12 columns of Table 3
   and all 12 of Table 2. The aggregate equals mean(GSM8k, GSM Symbolic, MATH) within ±0.03 every time
   (7B column: 54.70 vs 54.7). If MATH-500 were also in the aggregate, the mean of three rows would not
   reproduce it. Script: `mathtask_scripts/check_avg.py` (path at the end).
3. **Table 46** (Appendix A.4.1, "Base Main Suite") lists two math rows with similar names:
   `Minerva MATH* (2022) | 4 | CoT EM | pass@k | 0.6 | 0.6 | 1024 | 1, 4 (4) | 7`
   `MATH 500* (2022; 2023) | 4 | CoT EM | pass@k | 0.6 | 0.6 | 1024 | 1, 16 (32) | -`
   The task suite in the code (below) contains only the first. No table in the report gives a MATH 500 value
   for any base model.
4. **Table 13** (stage-wise base results) has a column labelled "Minerva". It gives Olmo 3 7B Stage 3
   (6.05T tokens, the long-context end, which is the released base; the card says "7B Model: No merging")
   **39.8**, with the Math aggregate at 54.4. This is a second published evaluation of the same checkpoint
   on the same task, 0.2 below Table 3.
5. **Appendix A.4** says: "All task suites are publicly available at github.com/allenai/olmes#olmo-3-eval-suite."
   Footnote 68 says: "pin to v0.11.0 for all evaluation in the final report" (vLLM).

## The code: olmes [S2]

The quotes come from `allenai/olmes` at HEAD `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e` (2026-03-24).
Every quoted block was added in `7bebf0eb148cd89d1d8a8096505f99d4fc48c209` (2025-10-18, "lift oe-eval ->
olmes"), before the Olmo 3 release. None has changed since. `git diff 7bebf0e HEAD` touches only IFBench
configs and dependency pins.

**The Olmo 3 base math suite.** It contains Minerva MATH and no MATH-500:
https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/configs/task_suites.py#L1050-L1053
```python
TASK_SUITE_CONFIGS["olmo3:base:math"] = {
    "tasks": ["gsm8k::olmo3:n8:v2", "gsm_symbolic_all::olmo3:n8:v2", "minerva_math::olmes:n4:v2"],
    "primary_metric": "macro",
}
```
This is the suite that the olmes README runs for "Run the base main evaluation" (README lines 110–120,
`olmo3:base:math`).

**The Minerva MATH suite.** It covers 7 subjects and uses the macro average:
https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/configs/task_suites.py#L342-L345
```python
TASK_SUITE_CONFIGS["minerva_math::olmes:n4:v2"] = {
    "tasks": [f"minerva_math_{task_type}::olmes:n4:v2" for task_type in MATH_TASK_TYPES],
    "primary_metric": "macro",
}
```
`MATH_TASK_TYPES` = algebra, counting_and_probability, geometry, intermediate_algebra, number_theory,
prealgebra, precalculus (`oe_eval/data/math_task_types.py` lines 1–9).

**The per-subject task.** This block sets the split, the shot count, the sampling and the metric:
https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/configs/tasks.py#L4696-L4714
```python
    TASK_CONFIGS[f"minerva_math_{task_type}::olmes:n4:v2"] = {
        "task_name": f"minerva_math_{task_type}",
        "split": "test",
        "primary_metric": "pass_at_1",
        "num_shots": 4,
        "fewshot_source": "Minerva:MATH:fixed",
        "generation_kwargs": {
            "max_gen_toks": 1024,
            "temperature": 0.6,
            "top_p": 0.6,
            "do_sample": True,
            "stop_sequences": ["Problem:", "\n\n"],
            "repeats": 4,
        },
        "metric_kwargs": {
            "pass_at_ks": [1, 2, 4],
            "maj_at_ks": [1, 2, 4],
        },
    }
```

**Dataset and prompt.** From
https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/tasks/oe_eval_tasks/minerva_math.py#L66-L95
and lines 153–184:
```python
        "dataset_path": "EleutherAI/hendrycks_math",
        ...
        "split": "test",
        "context_kwargs": {
            "use_cot": True,
            "cot_style": "minerva",  # Alternative: "original"
        },
...
            elif self.task_config["context_kwargs"]["cot_style"] == "minerva":
                query = "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"
...
            "answer": normalize_final_answer(remove_boxed(last_boxed_only_string(doc["solution"]))),
```
The test split sizes of `EleutherAI/hendrycks_math` (sha `21a5633873b6a120296cce3e2df9d5550074f4a3`) are
algebra 1,187, counting_and_probability 474, geometry 479, intermediate_algebra 903, number_theory 540,
prealgebra 871 and precalculus 546. The total is 5,000 [S8].

**Few-shot examples.** `FEWSHOT_SOURCES["Minerva:MATH:fixed"]` (`oe_eval/tasks/fewshot_sources.py`
line 2349) holds 4 examples: the domain of √(x−2)/√(5−x), det(AB), Terrell's weights, and a/b. It is the
lm-eval Minerva set with the blank lines inside example 4 removed. `fewshot_examples` returns them in
listed order with no sampling (`base_task.py` lines 583–589). The base model gets no chat template
(`use_chat_format` is set only when the model config sets `chat_model`, `run_eval.py` lines 327–329). The
rendered prompt is (`base_task.py` lines 563–575):
```
Problem:
{shot 1 problem}

Solution: {shot 1 solution, ending "Final Answer: The final answer is $[2,5)$. I hope it is correct."}

Problem:
... (shots 2-4 the same way) ...

Problem:
{test problem}

Solution:
```

**Scoring.** From `minerva_math.py` lines 201–279 and `oe_eval/metrics/metric.py` lines 686–714:
- `exact_match` is the lm-eval Minerva rule. The regex `Final Answer: The final answer is(.*?). I hope it is
  correct.` extracts the answer, `normalize_final_answer` normalizes it, and `is_equiv` compares it. That
  function parses both sides with sympy `parse_latex` and tests `simplify(diff) == 0`; a parse failure counts
  as unequal (lm-eval v0.4.3 `lm_eval/tasks/minerva_math/utils.py` lines 67–81, 144–198, 259–294) [S3].
- `exact_match_flex` takes candidates in this order: the Minerva answer, then the last `\boxed{}`, then the
  text between the last two `$`, then the whole output. Each candidate goes through `normalize_final_answer`.
  A sample passes if any candidate matches the gold by Minerva `is_equiv` or by Hendrycks `is_equiv`
  (`strip_string` equality, `lm_eval/tasks/hendrycks_math/utils.py` lines 36–50) [S3].
- In `PassAtK.process_one_doc` each sample scores `max(exact_match, exact_match_flex)`. Then `pass_at_1` is
  the estimator `1 - C(n-c,1)/C(n,1)` = c/4 for that problem.
- The suite score is `primary_score` = `primary_score_macro`, the mean of the 7 subject pass@1 values
  (`oe_eval/tasks/aggregate_tasks.py` lines 144–151). The micro average is also written, as `_micro`. It
  weights subjects by size (algebra 23.7 %, counting 9.5 %), so it is a different number.
- Pins: `lm_eval==0.4.3` and `antlr4-python3-runtime==4.11` (needed by `parse_latex`), from
  `pyproject.toml` lines 22 and 27; `vllm==0.11.0` with the `gpu` extra, line 69.

**The MATH-500 base config exists but is not in the suite.** It sits at
https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/configs/tasks.py#L3024-L3042:
`"minerva_math_500::olmo3:n32:v2"`, `HuggingFaceH4/MATH-500`, 4 shots `Minerva:MATH:fixed`, T 0.6, top-p
0.6, 1,024 tokens, `"repeats": 32`, `pass_at_ks [1, 2, 4, 8, 16]`. This matches Table 46's MATH 500 row.
No published table reports it.

**Sampling seed.** This config sets no `generation_seeds`, so the 4 samples are unseeded and a rerun will not
match bit for bit. The sampling-only standard error of the macro score, given the 5,000 problems, is at most
0.37 points. That bound uses p(1−p) ≤ 1/4 per problem, 4 samples, and subject weights 1/7. The two
published values for this checkpoint, 40.0 (Table 3) and 39.8 (Table 13), differ by 0.2.

## Other code the cards and report point to

- **olmo-eval** [S12]. The cards link "Evaluation code: https://github.com/allenai/OLMo-Eval". That name now
  resolves to `allenai/olmo-eval`, a different framework created 2026-01-08, after report v1. So it is not
  the code that produced Table 3. Its `olmobase:math` suite has the same composition: `"gsm8k:olmo3base"`,
  `gsm_symb:olmo3base` and `minerva_math:olmo3base` (`src/olmo_eval/evals/suites/olmobase.py` lines
  270–278). Its `minerva_math_<subject>:olmo3base` variant uses `minerva_math_fixed` shots, `max_tokens=1024`,
  `temperature=0.6`, `top_p=0.6`, `stop_sequences=("Problem:", "\n\n")`, `num_samples=4` and
  `PassAtKMetric(k=1)` (`src/olmo_eval/evals/tasks/minerva_math.py` lines 232–255), at commit
  `1810320abc50128ef40cfccb16348e3913a0105a`.
- **OLMo-core** [S13]. The official `OLMo-3-1025-7B-midtrain.py` and `-long-context.py` contain no
  generative math eval config. The report cites OLMo-core for training only.
- **open-instruct** [S10]. `scripts/train/olmo3/7b_rlzero_math.sh` on `main`
  (`2a074525efb00afacb5a1b1c855133512617c66f`) says "This is the later Olmo 3.1 recipe" and sets
  `EVALS="aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero"`. The original
  3.0 snapshot
  (https://github.com/allenai/open-instruct/blob/3af17aacff528025142a54449fcf8ce51a20363c/scripts/train/olmo3/7b_rlzero_math.sh)
  sets `EVALS="aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo"`. Neither script
  evaluates MATH or MATH-500 during training. The `_rlzero` and `_dapo` aliases, and
  `minerva_math_500::hamish_zs_reasoning_rlzero`, are not in public olmes (a search for "rlzero" finds
  nothing). The 3.0 script names a private eval image, `michaeln/oe_eval_rlzero`.

## What the model cards and the blog print

- `allenai/Olmo-3-1025-7B` card (commit `a81bae42db3975be1671e27b9c9a56da1a9f980f`) [S4]. Its evaluation
  table has the column "Olmo 3-Eval Math" = 54.7 and no MATH, GSM8k or GSM Symbolic column. The 2025-11-07
  revision `01b3d9d40ef0` had an OLMo-2-style template table with no MATH column either.
- `allenai/Olmo-3-7B-RL-Zero-Math` (`dee26a4074b88215cd1326472e1ab14d0259d892`) and
  `allenai/Olmo-3.1-7B-RL-Zero-Math` (`364820b002fc8e81d11a81c7fc9dd6b49c83cd05`) [S5, S6]. Neither card has
  an evaluation section, and neither has `model-index` metadata. The 3.0 card says "logs (coming soon)".
  Exact repo names: `allenai/Olmo-3-7B-RL-Zero-Math` (branches `step_100` … `step_1900`, `main`) and
  `allenai/Olmo-3.1-7B-RL-Zero-Math` (branches `step_0100` … `step_2800`, `main`). The 3.0 card's own
  table links `Olmo-3-7B-RLZero-Math`, a name without the hyphen. Other RL-Zero repos: `-Code`, `-IF`,
  `-General`, `-Mix` (3.0) and `Olmo-3.1-7B-RL-Zero-Code`.
- `allenai/Olmo-3-7B-Think` and `allenai/Olmo-3-7B-Instruct` cards [S7]. They print the row name "MATH"
  (Think chain 94.4 / 92.4 / 95.1; Instruct chain 65.1 / 79.6 / 87.3) and do not name the split.
- Ai2 blog [S9]. It has no 7B base table and no RL-Zero numbers; it names "Olmo 3.1 RL Zero 7B Math" only.
  Its 32B base table prints "MATH 43.4" (GSM8k 80.5, GSM Symbolic 61.0), while report Table 2 gives 43.8
  (80.6, 61.2) for the same released 32B model.

## Every published math number for the base and the RL-Zero-Math checkpoints

**Base, `allenai/Olmo-3-1025-7B`**

| source | name printed | task | settings | value |
|---|---|---|---|---|
| report Table 3 (v1, v2) | MATH | Minerva MATH, 5,000, `minerva_math::olmes:n4:v2` | 4-shot, T 0.6, top-p 0.6, 1,024 tok, n = 4, pass@1, macro over 7 | **40.0** |
| report Table 3 | OlmoBaseEval Math | mean of GSM8k 75.5, GSM Symbolic 48.6, MATH 40.0 | `olmo3:base:math` | 54.7 |
| report Table 13, "Olmo 3 7B Stage 3" (6.05T) | Minerva | same task (column name) | Base Main | 39.8 (Math agg. 54.4) |
| report Table 13, "Olmo 3 7B Stage 2" (6T, midtrain end) | Minerva | same | Base Main | 41.4 (Math agg. 59.8) |
| report Table 13, "Olmo 3 7B Stage 1" (5.9T) | Minerva | same | Base Main | 12.2 (Math agg. 23.5) |
| base card | Olmo 3-Eval Math | aggregate | — | 54.7 |
| any source | MATH-500 | — | config `minerva_math_500::olmo3:n32:v2` exists | none published |
| W&B `ai2-llm/Olmo-3-1025-7B`, run `tx9cibv8` (long-context run) | — | — | summary has no math key | none |

**RL-Zero 3.0, `allenai/Olmo-3-7B-RL-Zero-Math`** (the spec's RL model)

| source | task | settings | value |
|---|---|---|---|
| report text and tables | — | — | none |
| report Fig. 24, 38, 39 | AIME 2024, AIME 2025 | pass@1 "bootstrapped average over 32 samples" and pass@32, 32K tokens, T 1.0 | plotted only |
| card | — | — | none |
| W&B `ai2-llm/Olmo-3-7B-RL-Zero`, run `zim8gcyv` ("grpo_17kfilter_olmo3_7b_base", template `olmo_thinker_dapo`) | AIME 2024, `aime:zs_cot_r1::pass_at_32_2024_dapo` | 0-shot, chat, T 1.0, top-p 0.95, 32,768 tok, 32 samples, 30 problems | step 2000: pass@1 0.416 (second step-2000 row: 0.398), pass@32 0.800 |
| same run | AIME 2025, `..._2025_dapo` | same | step 2000: pass@1 0.355 (second row: 0.346), pass@32 0.667 |
| same run | MATH, MATH-500 | — | not logged |

The AIME history of `zim8gcyv`, as pass@1 by step (saved as `hist_aime2024_zim8gcyv.json` and
`hist_aime2025_zim8gcyv.json` in the scratchpad `mathtask_wandb/`):
- 2024: 100: 0.333, 500: 0.346, 1000: 0.375, 1500: 0.417, 1600: 0.430 (maximum), 2000: 0.398 / 0.416.
- 2025: 100: 0.291, 1000: 0.327, 1800: 0.357 (maximum), 2000: 0.346 / 0.355.

It is not established that `zim8gcyv` step 2000 is the released `main`. The open-instruct README names the
3.0 Math run `w0ql4f5r` in `open_instruct_internal`, which I could not open.

**RL-Zero 3.1, `allenai/Olmo-3.1-7B-RL-Zero-Math`** (not the spec's model)

MATH-500 comes from W&B project `ai2-llm/Olmo-3-7B-RL-Zero`, task `minerva_math_500::hamish_zs_reasoning_rlzero`.
The resolved config logged with each value: `dataset_path HuggingFaceH4/MATH-500`, split `test`, 500
instances, `num_shots 0`, `use_chat_format True`, `cot_style plain`, `temperature 1`, `top_p 0.95`,
`max_gen_toks 32768`, one sample, `primary_metric exact_match_flex`. Every row has `exact_match` 0.

| run | the row's `step` field | MATH-500 exact_match_flex |
|---|---|---|
| `ydkgbnai` "olmo3_7b_rlzero_math" (init `Olmo-3-1025-7B` rev `39ff1a89…`) | 400 / 700 / 800 / 1000 / 1100 / 1400 / 1500 / 1600 | 0.874 / 0.900 / 0.880 / 0.908 / 0.898 / 0.894 / 0.890 / 0.890 |
| `kl07u3lf` "olmo3_7b_rlzero_math_restart2k" (init `ydkgbnai` `step_2000`) | 200 / 300 / 400 / 500 / 700 / 800 / 900 | 0.880 / 0.892 / 0.906 / 0.902 / 0.906 / 0.890 / 0.894 |

AIME for the same runs (`..._rlzero` aliases, same sampling as above), as pass@1 / pass@32:
- `ydkgbnai`, 2024: 300 0.318/0.767, 400 0.349/0.800, 1000 0.461/0.733, 1200 0.451/0.800, 1400 0.443/0.767,
  1600 0.457/0.800.
- `ydkgbnai`, 2025: 100 0.268/0.567 … 1600 0.365/0.700, 1900 0.356/0.667.
- `kl07u3lf`, 2024: 300 0.469/0.767 … 800 0.518/0.800, 1000 0.487/0.833.
- `kl07u3lf`, 2025: 100 0.370/0.733 … 800 0.392/0.733, 1000 0.357/0.600.

Report Fig. 39 plots the 3.1 AIME curves; the text says the curves plateau "at a higher score ∼ 50%". It is
not established how the restart's step k maps to the HF branch `step_(2000+k)`, nor which step is 3.1 `main`.

## The olmes task id to run

```bash
# olmes at 7bebf0eb148cd89d1d8a8096505f99d4fc48c209 or 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e (math configs identical)
pip install -e ".[gpu]"      # vllm==0.11.0, lm_eval==0.4.3, antlr4-python3-runtime==4.11
olmes --model allenai/Olmo-3-1025-7B --model-type vllm \
      --task minerva_math::olmes:n4:v2 --output-dir <fresh dir>
```
Read the aggregate task `minerva_math::olmes:n4:v2`, field `primary_score` (= `pass_at_1_macro`), times 100.
The published values are 40.0 (Table 3) and 39.8 (Table 13). Do not read `pass_at_1_micro`. The base
weights at `main` (`a81bae42…`) have the same three safetensors LFS oids (`0490d6668e61`, `e127ea479fb6`,
`f3ddff10052f`) as revision `39ff1a89…`. To run the same suite inside olmo-eval, the equivalent id is
`minerva_math:olmo3base`; it is newer code and was not used for the report.

## Side findings for the spec (not asked; each changes a line of N13_SPEC)

1. **Table 49's "7B RL-Zero" column is the 3.1 recipe.** The column (13,314 prompts, 32 prompts × 8,
   response 16,384) matches the W&B config of the 3.1 run `ydkgbnai`: `allenai/Dolci-RLZero-Math-7B`,
   32 × 8, 16,384, `mask_truncated_completions False`. The 3.0 run `zim8gcyv` logs 16 × 16, response
   length 12,000 and `mask_truncated_completions True`. It trains on
   `saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered` plus
   `saurabh5/MATH_3000_Filtered_olmo_completions_new_template_filtered`, from the init
   `/weka/oe-adapt-default/michaeln/checkpoints/olmo3-7b-base`. This agrees with report A.6.4, which names
   "16k instead of 12k" and "not masking truncated sequences" as 3.1 changes. `N13_SPEC.md` line 61 assigns
   Table 49 to the 3.0 checkpoint. I did not check which MATH split `MATH_3000` was drawn from.
2. **Spec item 6 (Table 15's MATH split).** The olmes `olmo3:adapt` suite has `minerva_math::olmo3:adapt`
   (5,000 problems, 0-shot chat, `"\n\nPresent the answer in LaTex format: \\boxed{Your answer}"`, T 0.6,
   top-p 0.95, `exact_match_flex`, macro) and no MATH-500 task (`task_suites.py` lines 1086–1111,
   `tasks.py` lines 4778–4798). So Table 15's MATH row is most likely Minerva MATH, not MATH 500. This is
   inferred from the suite and was not checked against the runs.
3. The 3.1 card's RLVR section names the Code dataset ("Dolci-RL-Zero-Code-7B … coding queries"). This is a
   copy error in the card.

## Not verified

- The W&B report pages, which are rendered by JavaScript. I read the runs of the public project
  `ai2-llm/Olmo-3-7B-RL-Zero` through the public GraphQL API (8 runs, 3 of them math) instead of the page
  https://wandb.ai/ai2-llm/Olmo-3-7B-RL-Zero/reports/Olmo-3-7B-RL-Zero--VmlldzoxNTM0OTI1Nw.
- `ai2-llm/open_instruct_internal` (run `w0ql4f5r`, named as the 3.0 Math run). The API returns no project
  to an anonymous client.
- The prompt text and answer extraction of the `_rlzero`/`_dapo` aliases. They are not in public olmes. The
  rows show `exact_match` 0 and `exact_match_flex` near 0.89, so the Minerva "Final Answer" pattern never
  fires and some other extraction does. W&B records only the config fields listed above.
- The values inside report Figures 24, 38 and 39, which are plotted only.
- Which training step each RL-Zero-Math `main` is.
- The arXiv HTML version. I read PDF v1 and v2 in full in the evaluation sections and appendix A.4–A.6.

## Sources

- [S1] Olmo 3, arXiv 2512.13961: https://arxiv.org/pdf/2512.13961 (v2, 118 pages) and
  https://arxiv.org/pdf/2512.13961v1 (117 pages). Table 2, Table 3, Table 13, Table 15, Table 16, Table 46,
  Table 49, §3.3, §6, A.4, A.6.4.
- [S2] https://github.com/allenai/olmes at `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`.
- [S3] https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.3/lm_eval/tasks/minerva_math/utils.py and
  `.../hendrycks_math/utils.py`.
- [S4] https://huggingface.co/allenai/Olmo-3-1025-7B (README at `a81bae42…`; commit list and tree API).
- [S5] https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math; [S6] https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math
  (README, model API, refs API).
- [S7] https://huggingface.co/allenai/Olmo-3-7B-Think, https://huggingface.co/allenai/Olmo-3-7B-Instruct.
- [S8] https://datasets-server.huggingface.co/size?dataset=EleutherAI/hendrycks_math.
- [S9] https://allenai.org/blog/olmo3.
- [S10] https://github.com/allenai/open-instruct `scripts/train/olmo3/7b_rlzero_math.sh` and `README.md` at
  `2a074525…`; snapshot at `3af17aac…`; issue #1899 (open).
- [S11] W&B GraphQL https://api.wandb.ai/graphql, project `ai2-llm/Olmo-3-7B-RL-Zero`, runs `zim8gcyv`,
  `ydkgbnai`, `kl07u3lf` (summary, config, sampledHistory).
- [S12] https://github.com/allenai/olmo-eval at `1810320abc50128ef40cfccb16348e3913a0105a`.
- [S13] https://github.com/allenai/OLMo-core `src/scripts/official/OLMo3/` at `ccf4ef4b…`.

Local copies (scratchpad, untrusted downloads, each in its own directory):
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/mathtask_in/olmo3.pdf`
(and `olmo3.txt`), `.../scratchpad/mathtask_v1/olmo3_v1.pdf`, `.../scratchpad/mathtask_olmes/olmes/`,
`.../scratchpad/mathtask_olmoeval/olmo-eval/`, `.../scratchpad/mathtask_lmeval/`, `.../scratchpad/mathtask_hf/`,
`.../scratchpad/mathtask_wandb/` (run JSON and history JSON), `.../scratchpad/mathtask_oi/`,
`.../scratchpad/mathtask_blog/`, `.../scratchpad/mathtask_olmocore/`. Scripts:
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/mathtask_scripts/check_avg.py`
(the Table 2 and Table 3 aggregate recount) and
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/mathtask_scripts/wb_hist.py`
(the W&B history pull).
