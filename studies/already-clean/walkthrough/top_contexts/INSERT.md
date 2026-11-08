# Inserting "What a top context is" into the walkthrough

Line numbers are in `../index.html` as committed at 8f0bcbfc4 (932 lines). Three pastes. Paste from the
bottom up, so that the line numbers above each paste still hold.

## 1. Scripts: after line 932

Line 932 is `<script src="neuron_across_training/slider.js"></script>`. Add after it:

```html
<script src="top_contexts/data.js"></script>
<script src="top_contexts/top_contexts.js"></script>
```

## 2. The section: before line 326

The plan (`VISUALS_PLAN.md`, P1) says the picture replaces lines 318–321 and 509. Lines 318–321 sit inside
the definition list of the aside "Background · neurons, poles, activations", inside section `claim`, where a
section cannot go. So the section goes immediately after section `claim`: before line 326
(`<section id="questions">`), after the blank line 325 that follows `</section>` at line 324.

```html
<section id="topcontexts">
  <h2>What a top context is</h2>
  <p>A neuron’s top contexts are the 50 tokens, out of 20,560,068 scored at the end of training, where it fired hardest, each shown with the text around it.</p>
  <div id="topctx"></div>
  <p class="record">Source: the saved top-50 contexts at the final training point (step 1,413,814), with each context’s position in the corpus; the menu holds the lithium and “got” neurons and 18 drawn at random from the 600 studied.</p>
</section>

```

The lead-in is one sentence. Its number, 20,560,068, is `n_tokens_scored` in the final step's
`metrics.json` (checks in `README.md`).

## 3. The stylesheet: after line 10

Line 10 is `<link rel="stylesheet" href="neuron_across_training/slider.css">`. Add after it:

```html
<link rel="stylesheet" href="top_contexts/top_contexts.css">
```

## The prose the plan says the picture replaces

Quoted from `../index.html`; whether to cut or shorten each is the integrator's call.

- Lines 318–319, the definition "Top context": "The text around one of a pole’s strongest tokens, with
  that token marked: 256 tokens before it and 16 after." The picture shows what a top context is. The 256
  and 16 are what the explainer and clusterer were shown (`DATA_CONTRACT.md` §8); the saved file behind the
  picture holds 64 tokens before and 16 after, and the picture shows the last 180 characters before and the
  first 36 after. If the definition stays, these numbers stay true of the explainer, not of the picture.
- Lines 320–321, the definition "Corpus": "About 50 million tokens of text from the training mix of OLMo-3,
  an open model." This is the run the explanations were written from. The picture's corpus is a different
  run, 20,560,068 scored tokens, and the picture's caption says so. Later lines (500: "A harvest runs a
  model over the corpus") lean on this definition, so cutting it needs a replacement that names both runs.
- Line 509: "Two rented GPUs harvested all 16 checkpoints over one 20.57M-token corpus. The final checkpoint
  ran first ... after 26.9 billed GPU-hours against the 28.5 estimated." The picture carries the first
  sentence's corpus size (20,560,068 scored of 20,570,112 positions; "20.57M" is the positions). The rest
  of the line (GPU-hours, timing) is not in the picture.
