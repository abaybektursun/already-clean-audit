# Inserting "One explanation and its test texts" into the walkthrough

Line numbers are in `../index.html` as of commit 8f0bcbfc4 (932 lines; unchanged at 716cc96ed).

## Where it goes

VISUALS_PLAN.md P3 gives the place as the prose it replaces: lines 338–345, 352, 360–382 and Figure 4
(384–411). Most of that is the section "One pass and one fail, from the real data", lines 356 to 411, so the
new section goes at **line 356, in place of lines 356–411**. It keeps that section's id, `example`; no link
on the page points to it. Lines 357–359 are not in the plan's list, but they are the old section's heading and
its sentence introducing the two Qwen3-4B examples, so they go with it.

Lines 338–345 and 352 sit in the section before it, "A test the explanation can fail" (lines 334–354). They are
quoted below; whether they are cut or shortened is the integrator's decision. The picture carries the four roles
of 338–341 (the explanation, the texts written from it alone, the GPU score, the strong line), the pass rule
of 344 and, in its caption, the shuffled-text line of 347–350. It does not carry the class labels of 338 and
352 (one job, UNK-F), the Opus and Sonnet note of 343, or the essay's own rates of 345.

## 1. The stylesheet, in `<head>`, after line 10

```html
<link rel="stylesheet" href="explanation_test/explanation_test.css">
```

## 2. The section, at line 356, replacing lines 356–411

```html
<section id="example">
  <h2>One explanation and the texts that test it</h2>
  <p>An explanation passes only if texts written from it alone fire the neuron and its near misses do not; drag the strong line to see where pass turns to fail.</p>
  <div id="xtest"></div>
  <p class="record">Source: the audit’s saved explanations, test texts and scores for 600 sides of 300 OLMo-3 7B neurons, and the scoring run’s record of the token where each text peaked; the page recomputes pass or fail from those scores with the study’s rule.</p>
</section>
```

## 3. The scripts, after line 932

```html
<script src="explanation_test/data.js"></script>
<script src="explanation_test/explanation_test.js"></script>
```

## Two things on the page that point at what this replaces

- Line 660 says "as the quota neuron was read in the worked example above". The quota neuron, L09N8987− of
  the Qwen3-4B discovery set, leaves the page with lines 360–368; the picture opens on the got neuron instead.
- Figure 4 (lines 384–411) is removed. The page's figures run, in page order, 4, 2, 1, 3, 5, 6, 7, 8
  (lines 385, 431, 465, 512, 724, 776, 819, 874), so without it there is no Figure 4; the host's CSS block
  "tally (figure 4)" at line 226 then styles nothing.

## The prose lines it replaces (VISUALS_PLAN.md P3, "Replaces: 338–345, 352, 360–382, Fig 4")

Lines 338–345:

```html
    <li><strong>The explainer</strong>, a language model, reads a pole’s 30 strongest top contexts. It writes a one-sentence explanation and a class label.</li>
    <li><strong>The blind writer</strong> sees only that sentence. It writes five <em>positives</em> meant to fire the pole and five <em>near misses</em> meant not to. It writes five <em>distant texts</em> on the most unrelated concept it can find. It adds one <em>blind maximum</em>, a sentence meant to fire the pole as hard as possible.</li>
    <li><strong>A GPU runs every text through the model being studied.</strong> A text’s score is its peak activation at the pole, divided by the pole’s strongest activation in the corpus.</li>
    <li><strong>A text is strong</strong> if its score reaches a threshold called θ<sub>s</sub>.</li>
  </ol>
  <p>Each role is a call to a language model, such as Opus 5.5 or Sonnet 5.5, through a paid API.</p>
  <p>An explanation <strong>passes</strong> if at least 3 of 5 positives are strong. At most 2 of 5 near misses and 1 of 5 distant texts may be strong <span class="src">spec §6.1</span>. Near misses test the explanation’s edges. Distant texts test packing, because a packed neuron would also fire on something unrelated.</p>
  <p>The essay’s own version, on Qwen3-32B, found positives strong 76% of the time, near misses 11% and distant texts about 1% <span class="src">essay, Explanations That Could Lose</span>. Its threshold for strong was a fixed 30% of the corpus maximum. This study measures its threshold instead.</p>
```

Line 352:

```html
  <p>A pole’s class is computed, not taken from the explainer. A pole counts as one job only if its explanation passes. A failed explanation makes it <em>UNK-F</em>: not understood, explanation failed.</p>
```

Lines 356–382 (the section's heading and introduction, 356–359, go with it):

```html
<section id="example">
  <h2>One pass and one fail, from the real data</h2>
  <p>Two real poles from the discovery set follow, from Qwen3-4B-Base with the frozen prompt and Opus 5.5 in every role. L09N8987− means layer 9, neuron 8987, negative pole.</p>

  <h3>A pass: L09N8987−</h3>
  <blockquote>Fires on words for apportioning or dividing out shares of something among recipients (quota/quotas, allocation, allocated, apportioned, allotment, distributed, divided among, divvy out to), most strongly on the noun “quota(s)” or “allocation” for production, funding, water, troops or political posts, including pieces inside these words.<span class="who">explainer, Opus 5.5 · labelled one job, confidence 0.75</span></blockquote>
  <ul class="texts">
    <li><span class="kind">positive</span><span>The ministry raised the annual fishing quota for cod, but smaller boats complained their allocation barely covered fuel costs.</span></li>
    <li><span class="kind">near miss</span><span>The ministry raised the annual fishing season for cod, but smaller boats complained their catch barely covered fuel costs.</span></li>
    <li><span class="kind">distant</span><span>The cat’s fur smelled faintly of sun and dust where she had napped by the window.</span></li>
  </ul>
  <p>All five positives were strong, and no near miss or distant text was. The blind maximum scored 1.22, so it fired the pole harder than any token in the 50M-token corpus. It also passed in the Sonnet 5.5 and DeepSeek-V4-Pro rounds, with their own explanations.</p>

  <h3>A fail: L02N3919−</h3>
  <blockquote>In polished, generic, ChatGPT-style informational or SEO prose, this pole fires on the word “and” (sometimes an em dash) that joins two paired abstract modifiers or nouns, as in “cultural and spiritual significance” …<span class="who">explainer, Opus 5.5 · labelled one job, confidence 0.5</span></blockquote>
  <p>SEO prose is text written to rank well in web search.</p>
  <ul class="texts">
    <li><span class="kind">positive</span><span>The temple holds deep cultural and spiritual significance for local communities, attracting visitors from around the world who seek a meaningful connection to the past.</span></li>
    <li><span class="kind">near miss</span><span>The temple holds deep significance for local communities, attracting visitors from around the world who seek a meaningful connection to the past.</span></li>
  </ul>
  <p>All five positives were strong, and so were all five near misses. The rule allows two, so the pole is UNK-F.</p>
  <p>In the Sonnet 5.5 round, the same pole passed with 5, 0 and 0 strong. Sonnet’s explanation named “the connector right after an abstract adjective or noun in polished SEO or encyclopedic prose”. Its near misses kept “and” in plain prose:</p>
  <ul class="texts">
    <li><span class="kind">near miss</span><span>Mix the flour, sugar, and butter in a large bowl, then bake for twenty minutes at 180 degrees.</span></li>
  </ul>
  <p>Opus’s near misses dropped the paired words and kept the polished style, and all five fired. Together, the two rounds suggest the style drives this pole more than the word “and”. That is a reading, not a separate measurement.</p>
  <p>So a failed test judges one explanation and one set of texts. It does not show that the neuron lacks a clear job <span class="src">spec §0b</span>.</p>
```

Lines 384–411, Figure 4 and the section's end: the `<figure class="fig" id="fig4">` block, "Strong texts per
kind, for the two example poles", three rows of squares (L09N8987− Opus 5.5 round 5/0/0 passes; L02N3919− Opus
5.5 round 5/5/0 fails; L02N3919− Sonnet 5.5 round 5/0/0 passes), its caption, and `</section>`.

## Why the picture's examples are not L09N8987− and L02N3919−

Both are poles of the Qwen3-4B discovery set. Their verdict files are on S3 only; none of the local files holds
them (`q1_O7_poles.jsonl`, `q1_Q4B_poles.jsonl` and `q1_G4_poles.jsonl` are the audit sets, and the discovery
set is a separate 60 neurons). The picture is about OLMo-3 7B, as P3 asks. It opens on the got neuron,
L03N7155+, which the page already names (lines 288 and 622) and which passes with 5, 0 and 0 strong, the same
tally as L09N8987−. The second example is the OLMo side whose failure is nearest to L02N3919−'s 5, 5 and 0:
L25N5542+, which fails on its near misses with 5, 3 and 0 (`extract.py` picks it by that rule). The third is
the lithium neuron, L03N6372+ (lines 293 and 622), which passes with 5, 0 and 0.
