# Inserting "How many neurons have one job" into the walkthrough

Line numbers are in `../index.html` as of commit 8f0bcbfc4 (932 lines).

## Where it goes

VISUALS_PLAN.md P4 gives the place only as the prose it replaces: lines 304, 328, 673 and 702. Of those,
line 702 is where the page reports the audit's counts, so the section goes at **line 702**.

Line 702 sits inside `<section id="after">` (lines 694 to 928), whose parts use `<h3>`. Pasted there as
written, the block is a section with an `<h2>` nested in that section.

## 1. The stylesheet, in `<head>`, after line 10

```html
<link rel="stylesheet" href="one_job/one_job.css">
```

## 2. The section

```html
<section id="count">
  <h2>How many neurons have one job</h2>
  <p>Each dot is one of the 600 neurons tested in a model, placed by the class its test scores give it; drag the strong line to see how many keep one job.</p>
  <div id="onejob"></div>
  <p class="record">Source: the audit’s saved explanations and test-text scores, 600 neurons in each of three models; the page recomputes every class from those scores with the study’s rule.</p>
</section>
```

## 3. The scripts, after line 932

```html
<script src="one_job/data.js"></script>
<script src="one_job/one_job.js"></script>
```

## The prose lines it replaces (VISUALS_PLAN.md P4, "Replaces: 304, 328, 673, 702")

Line 304:

```html
  <p>From preliminary audits of Qwen3-32B, the essay estimates that about 90% of neuron poles have one main job. About 7% have two related jobs, and 2–4% are not yet understood <span class="src">essay, The Short Version</span>. The exact shares depend on its thresholds.</p>
```

Line 328:

```html
  <p><strong>Q1, the audit.</strong> What share of neuron poles has one job, two related jobs, unrelated jobs, or is not understood? It uses 300 random neurons, so 600 poles, in each of Qwen3-4B-Base, Gemma-3-4B-pt and OLMo-3-1025-7B.</p>
```

Line 673:

```html
  <p>In the audit, a not-understood pole says something about the explainer and the test, never about packing <span class="src">spec §0b</span>. The models have 4 to 7 billion parameters, and the corpus is 200 times smaller than the essay’s 10 billion tokens. The study predicts it will not reach the essay’s 90 / 7 / 2–4, and it will examine any gap on the observer’s side first <span class="src">journal 20 Oct 02:58Z</span>.</p>
```

Line 702:

```html
  <p>The audit answered Q1. Counting only poles whose explanation passed the text test, 59.7% of poles in Qwen3-4B-Base have one job, 56.3% in Gemma-3-4B and 51.8% in OLMo-3-7B. Between 32% and 41% of poles are not understood, which almost always means an explanation was written and failed its test. Poles with two unrelated jobs are rare, at most 1.3% <span class="src">already-clean RESULTS.md, Q1 tables</span>. These shares depend on who wrote the explanations. On OLMo, poles explained by Opus 5.5 came out one-job 54.7% of the time and poles explained by Sonnet 5.5 44.8%, and when a second model labelled the same poles its labels agreed with the first only weakly, an agreement statistic of 0.07 to 0.18 where 0 is chance and 1 is perfect. The essay’s 90 / 7 / 2–4 was not reached. The study had predicted it would not be, and it treats the gap as a limit of the explainer and the test before it reads anything into the model.</p>
```

What the picture carries of these lines: the essay's 90% as the line at 540 (304); the four classes and
600 per model (328); "cannot tell is not packing" and the 4 to 7 billion parameter caution, in its caption
(673); the one-job counts 358, 338 and 311, the cannot-tell counts, and the explainer's own count of
525, 520 and 521 (702). It does not carry the essay's 7% and 2–4%, the 200-fold smaller corpus, the
Opus and Sonnet split (54.7% and 44.8%), or the agreement statistic of 0.07 to 0.18.
