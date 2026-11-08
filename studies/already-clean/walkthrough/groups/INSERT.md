# Inserting "One neuron's 50 contexts, sorted into groups" into the walkthrough

Line numbers are in `../index.html` as committed at 8f0bcbfc4 (932 lines). Three pastes. Paste from the
bottom up, so that the line numbers above each paste still hold.

## 1. Scripts: after line 932

Line 932 is `<script src="neuron_across_training/slider.js"></script>`. Add after it:

```html
<script src="groups/data.js"></script>
<script src="groups/groups.js"></script>
```

## 2. The section: before line 577

The plan (`VISUALS_PLAN.md`, P5) says the picture replaces line 581, line 720 and the counting in line 871.
Line 581 is the first of these, and it sits inside the list of section `readout`, where a section cannot
go. So the section goes immediately before that section: before line 577 (`<section id="readout">`), after
the blank line 576 that follows `</section>` of `running` at line 575.

```html
<section id="groups">
  <h2>One neuron’s 50 contexts, sorted into groups</h2>
  <p>A language model sorted each neuron’s 50 strongest contexts into named groups, and asked a second time it often made a different number: the lithium neuron got 15 groups the first time and 3 the second.</p>
  <div id="ngroups"></div>
  <p class="record">Source: the clustering record, two sortings of each of the 600 neurons by Claude Sonnet 5.5, the same file the verdict read, and the saved top-50 contexts at the end of training.</p>
</section>

```

The lead-in is one sentence. Its numbers are in the checks of `README.md`: 15 and 3 for L03N6372+, and
the two counts differ for 418 of the 600 neurons (equal for 182), which is what "often" stands for.

## 3. The stylesheet: after line 10

Line 10 is `<link rel="stylesheet" href="neuron_across_training/slider.css">`. Add after it:

```html
<link rel="stylesheet" href="groups/groups.css">
```

## The prose the plan says the picture replaces

Quoted from `../index.html`; whether to cut or shorten each is the integrator's call.

- Line 581, step 1 of the readout: "A clustering model groups the pole’s 50 strongest final contexts into
  named clusters, one per candidate job." With the picture above the section, this step can point at it.
- Line 720, the paragraph "A word about words": its second and third sentences say what the picture's
  caption says (the group is what a clustering model produced; it was not asked to check that a group is
  about one thing) and give 109 of 129. The paragraph also switches the page's word from "job" to
  "cluster", which later sections rely on, so that part has to stay.
- Line 871, the counting in "Taken together": "the language model never splits one marked token across
  clusters, ... but most of its clusters mix tokens, so the clusters are not simply token groups." The
  picture lets a reader count the distinct tokens in each group. A caution for whoever edits this line: in
  the record behind the picture, the first sorting puts the same marked-token string into two or more
  groups for 383 of 18,148 (neuron, token) pairs, on 196 of the 600 neurons (for example L01N8975+,
  `' Library'`, 38 contexts in one group and 1 in another). The "never" may come from a different
  definition or subset in the instruments study; it was not checked here.
