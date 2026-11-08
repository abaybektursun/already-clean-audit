# Inserting "The judge's calls" into the walkthrough

Line numbers are in `../index.html` as committed at 8f0bcbfc4 (932 lines). Three pastes. Paste from the bottom up,
so that the line numbers above each paste still hold; the other modules' INSERT.md files use the same base.

## 1. The scripts: after line 932

Line 932 is `<script src="neuron_across_training/slider.js"></script>`. Add after it (after any other module's
scripts already pasted there):

```html
<script src="judge/data.js"></script>
<script src="judge/judge.js"></script>
```

## 2. The section: after line 612, before line 613

The plan (`VISUALS_PLAN.md`, P7) says the picture replaces lines 583–588, 668, 671, 699 and 707–708. Lines 583
and 584 are items of the ordered list in section `readout` (577–611), and lines 586–588 are the paragraphs that
follow it, where a section cannot go. So the section goes right after that section ends: after the blank line 612
that follows `</section>` at line 611, and before line 613, `<section id="meaning">`. The reader then meets the
calls right after the test and its formula are described.

```html
<section id="judge">
  <h2>The judge’s calls</h2>
  <p>A judge saw each late-born group of a neuron’s contexts beside two records from earlier in training, its own neuron’s and a random neuron’s, and picked the one it grew from; each dot is one call, coloured by whether it picked the neuron’s own.</p>
  <div id="judgecalls"></div>
  <p class="record">Source: the judge’s record of 7,352 calls on OLMo-3 7B against a random neuron, and the verdict’s per-group record, which holds the text the judge saw.</p>
</section>

```

The lead-in is one sentence. The section id `judge` and the wrapper id `judgecalls` are not used elsewhere on the
page.

## 3. The stylesheet: after line 10

Line 10 is `<link rel="stylesheet" href="neuron_across_training/slider.css">`. Add after it:

```html
<link rel="stylesheet" href="judge/judge.css">
```

## The prose the plan says the picture replaces

Quoted from `../index.html`; whether to cut or shorten each is the integrator's call. What the picture carries:
the two shares right (81.2% of 4,952 calls on 1,238 late-born groups, 92.4% of 2,400 calls on 600 control
groups), what the judge saw and was asked, the controls, the derived share 26.5% with its range 21 to 32%, a coin
for comparison, and in its caption the easier controls (50.0% against 18.7%) and the warning that a right call is
not proof of descent. What it does not carry: the decision rule (0.10 and 0.20), the validity gate on the tuning
neurons, the trend with birth time, and the within-neuron comparison of 299 against 309.

- Line 583, the parent test: "A judge model sees ten contexts from a late-born cluster, plus two records from the
  checkpoint before its birth. One is the true pole’s 20 strongest contexts. The other is a random same-layer
  pole’s, called the decoy. The records show text only. The judge names the one the cluster grew from." The
  picture's caption and opened call show the same. Two corrections for whoever edits this line, from the records
  (`extract.py`): the judge saw ten contexts in 3,816 of the 7,352 calls and fewer in the rest, because a group
  shows min(10, its size); and the records held 20 contexts in 7,256 calls and fewer in 96, because contexts from
  the group's own passages are removed first and the other record is cut to the same length.
- Line 584, the positive control: "The same test runs on each pole’s main cluster, against the pole’s own record at
  the second-to-last checkpoint. This pair is related by construction. The judge must score at least 0.80 here, on
  the 40 tuning poles, before judging the 600." The picture shows the 600 control groups; the gate on the tuning
  neurons is not in it (line 699 gives 39 tuning poles; the gate record `judge_gate.json` holds 39, one neuron's
  sorting having been declined).
- Line 586: "If a late job grew from the pole’s earlier job, the judge should pick the true record. If it arrived
  unrelated, as packing predicts, the judge is at chance, 50%." The coin button shows chance.
- Lines 587–588, the unrelated share and its formula. The picture's line under the columns derives 26.5% from
  92.4% and 81.2% in words. If these lines are cut, the decision rule at 589–594 still needs the name "unrelated
  share".
- Line 668, "If the result is mixed": the picture carries the 0.265; the rest of the paragraph (birth order, the 17
  neurons, the trend with time) is not in it.
- Line 671, "If the judge fails its control": the picture carries 92.4% on the 600 and, in its caption, that a
  right call is not proof of descent; the gulch call is in the menu ("The Gul neuron: gulch, right with no gul in
  the record"). The instruments story in the middle of the paragraph is not in it.
- Line 699: the picture carries 92.4%, 1,238, 81.2%, 0.265 and its interval, not the 91.0% on the tuning neurons
  or the rule's label.
- Lines 707–708: the picture carries only the easier controls (50.0% against 18.7%, which is the "2.7 times" of
  line 708). The trend with birth time in 707 and the adjusted share 0.21 [0.13, 0.27] and the within-neuron
  comparison in 708 are not in it, so these two lines should stay unless they move elsewhere.
