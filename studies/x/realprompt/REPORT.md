# REALPROMPT — report

Pre-registration: `studies/x/realprompt/PREREG.md`. Plan: `infra/pg/realprompt_plan.py`. Job: `infra/aws_pg/jobs/realprompt.py`. Scorer: `infra/pg/realprompt_score.py`. Banks: `bank/x/realprompt/<M>/`. Every number below is computed from the banks by the scorer; `results/score.json` holds them all.

**Result.** Frozen verdicts: Q4B 3 NOT SCORABLE; G4 3 REFUTED. Under the exploratory code rule D1 (§1b): 6 REFUTED. The highest push presence of any code neuron at any dose is HTML title attribute (Q4B) at dose 32: 0.20 [0.05, 0.40] against its control's 0.00 [0.00, 0.00].

## 1. Prediction verdicts

Registered: at d* (the smallest positive dose where at least 80% of the push samples are code), construct presence under the push minus under the same-layer random control is at least 0.15. REFUTED when the 95% interval's upper end is below 0.15; SUPPORTED when the gap is at least 0.15 and the lower end is above 0; otherwise NOT DECIDED; NOT SCORABLE when a gate failed, no dose keeps code, or the control's presence at d* is above 0.85.

| model | code neuron | control | d* | code share at d* | push | control | gap [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| Q4B | SQL FOREIGN KEY L11N5238+ | L11N3497+ | — | — | — | — | — | NOT SCORABLE |
| Q4B | Angular L27N2924- | L27N7222- | — | — | — | — | — | NOT SCORABLE |
| Q4B | HTML title attribute L08N3398- | L08N3836- | — | — | — | — | — | NOT SCORABLE |
| G4 | React useEffect L07N10202- | L07N2158- | 2 | 0.90 | 0.00 | 0.05 | -0.05 [-0.15, +0.00] | REFUTED |
| G4 | Python import ... as L06N599+ | L06N3187+ | 2 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |
| G4 | SQL REFERENCES L08N7823+ | L08N6357+ | 2 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |

- SQL FOREIGN KEY (Q4B L11N5238+): NOT SCORABLE — model gate failed: code rule controls.
- Angular (Q4B L27N2924-): NOT SCORABLE — model gate failed: code rule controls.
- HTML title attribute (Q4B L08N3398-): NOT SCORABLE — model gate failed: code rule controls.
- React useEffect (G4 L07N10202-): REFUTED — interval upper end 0.000 < 0.15.
- Python import ... as (G4 L06N599+): REFUTED — interval upper end 0.000 < 0.15.
- SQL REFERENCES (G4 L08N7823+): REFUTED — interval upper end 0.000 < 0.15.

### 1b. Exploratory deviation D1 (not the headline)

Dated 2026-10-26, written after reading Q4B's unsteered code samples: the frozen code rule called 5 of 20 of them prose although all five are Python (import blocks and docstrings carry few of its punctuation characters and no closing fence within 128 tokens). D1 also calls a completion code when at least 2 lines of its first 64 tokens start like a line of code (`^\s*(import |from \S+ import|def |class |return\b|#|//|/\*|SELECT\b|CREATE\b|INSERT\b|ALTER\b|function\b|const |let |var |public |private |<[A-Za-z!/])`). Everything else as registered. Code-rule controls under D1: Q4B code prompts 0.95, prose prompts 0.01; G4 code prompts 1.00, prose prompts 0.09; O7 code prompts —, prose prompts 0.00.

| model | code neuron | d* | code share at d* | push | control | gap [95% CI] | verdict under D1 |
|---|---|---|---|---|---|---|---|
| Q4B | SQL FOREIGN KEY L11N5238+ | 4 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |
| Q4B | Angular L27N2924- | 4 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |
| Q4B | HTML title attribute L08N3398- | 4 | 1.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |
| G4 | React useEffect L07N10202- | 2 | 1.00 | 0.00 | 0.05 | -0.05 [-0.15, +0.00] | REFUTED |
| G4 | Python import ... as L06N599+ | 2 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |
| G4 | SQL REFERENCES L08N7823+ | 2 | 0.95 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | REFUTED |

## 2. Presence table per code neuron and dose

Presence rate over 20 samples (2 prompts × 10) with its 95% bootstrap interval; dose 0 is the unsteered code-prompt samples (shared by every code neuron of the model); gap = push − control, paired by (prompt, sample). Code share = share of samples that are code by the frozen code rule.

### Q4B L11N5238+ — SQL FOREIGN KEY (control L11N3497+; regex `(?i)\bforeign\s+key\b`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 [0.00, 0.00] | — | — | 0.75 | — | 0.95 | — |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.80 | 0.80 | 0.95 | 1.00 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 0.90 | 1.00 | 0.90 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 0.80 | 1.00 | 1.00 |
| 32 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 0.70 | 1.00 | 0.95 |

### Q4B L27N2924- — Angular (control L27N7222-; regex `\bAngular(JS)?\b|@angular/`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 [0.00, 0.00] | — | — | 0.75 | — | 0.95 | — |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 1.00 | 0.95 | 1.00 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 0.90 | 0.95 | 1.00 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.85 | 0.90 | 1.00 | 1.00 |
| 32 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.75 | 0.85 | 0.85 | 1.00 |

### Q4B L08N3398- — HTML title attribute (control L08N3836-; regex `\stitle\s*=\s*["']|\btitle\s+attribute\b|attr\(\s*['"]title`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 [0.00, 0.00] | — | — | 0.75 | — | 0.95 | — |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 0.85 | 1.00 | 1.00 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.80 | 0.85 | 0.95 | 1.00 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 0.90 | 1.00 | 1.00 |
| 32 | 0.20 [0.05, 0.40] | 0.00 [0.00, 0.00] | +0.20 [+0.05, +0.40] | 0.90 | 0.80 | 1.00 | 0.95 |

### G4 L07N10202- — React useEffect (control L07N2158-; regex `\buseEffect\b`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 [0.00, 0.00] | — | — | 0.85 | — | 1.00 | — |
| 2 | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.15] | -0.05 [-0.15, +0.00] | 0.90 | 0.95 | 1.00 | 1.00 |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 1.00 | 1.00 | 1.00 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.70 | 0.90 | 0.95 | 1.00 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.10 | 0.05 | 0.10 | 0.15 |

### G4 L06N599+ — Python import ... as (control L06N3187+; regex `\bimport\s+[\w.]+\s+as\s+\w+|\bfrom\s+[\w.]+\s+import\s+\w+\s+as\s+\w+`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.05 [0.00, 0.15] | — | — | 0.85 | — | 1.00 | — |
| 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 0.95 | 0.95 | 1.00 |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 1.00 | 1.00 | 1.00 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.55 | 0.90 | 0.85 | 0.90 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 1.00 | 0.15 | 1.00 | 0.15 |

### G4 L08N7823+ — SQL REFERENCES (control L08N6357+; regex `(?i)\breferences\s+[\w.`"\[\]]+\s*\(`)

| dose | push | control | gap [95% CI] | code share push | code share control | push (D1) | control (D1) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 [0.00, 0.00] | — | — | 0.85 | — | 1.00 | — |
| 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.95 | 1.00 | 0.95 | 1.00 |
| 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.90 | 0.70 | 0.95 | 0.80 |
| 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.30 | 0.05 | 0.45 | 0.15 |
| 16 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 | 0.15 | 0.00 | 0.20 |

## 3. Instrument checks

| model | identity: max abs logit diff at dose 0 (must be 0) | failing input (dose 1) | realized cos failing input (per-forward hook; must be < 0.95) | step-1 realized cos: cells passing | worst cell median cos | lowest row cos | code rule: dose-0 code prompts (≥ 0.80) | dose-0 prose prompts (≤ 0.20) |
|---|---|---|---|---|---|---|---|---|
| Q4B | 0 | 0.219 | max 0.000 | 84/84 | 0.9996 (L31N5699+, dose 4) | 0.9996 | 0.75 (n 20) | 0.01 (n 80) |
| G4 | 0 | 1.45 | max 0.000 | 96/96 | 0.9985 (L14N6393+, dose 2) | 0.9985 | 0.85 (n 20) | 0.06 (n 80) |
| O7 | 0 | 0.105 | max 0.000 | 12/12 | 0.9989 (L26N6019+, dose 2) | 0.9973 | — (n 0) | 0.00 (n 40) |

Two-instance token identity: not run (one instance per model). Gate failures per model: Q4B: code rule controls; G4: none; O7: none.

## 4. Instance-hours and cost

| model | instance | type | hours | USD |
|---|---|---|---|---|
| Q4B | <instance-id> | g6e.4xlarge | 0.126 | 0.38 |
| G4 | <instance-id> | g6e.4xlarge | 0.113 | 0.34 |
| O7 | <instance-id> | g6e.4xlarge | 0.100 | 0.30 |
| total | | | 0.338 | 1.02 |

Hours: EC2 LaunchTime (describe-instances) to the S3 LastModified of the job's STATUS object; the instance shuts down within about a minute after STATUS, which is not counted. Price: AWS Pricing API, on-demand Linux us-east-1, g6e.4xlarge $3.00424/h (g6e.2xlarge had no capacity at launch). EBS and S3 not counted. Failed launch attempts (InsufficientInstanceCapacity) created no instance.

## 5. Sample-check notes (read after the run)

- The frozen code rule under-calls Python: import blocks and docstrings carry few of its punctuation characters (5 of 20 unsteered Q4B code samples, all Python, read as prose; D1 in §1b). It also calls prose code when the prose contains a fence line (one Q4B news sample opens with two fence lines, then a news paragraph).
- The distinct-bigram loop flag splits on word characters, so repetition inside a word passes it: G4 L26N9472- at dose 8 writes "folk typessss typesss" with ratio 0.90.
