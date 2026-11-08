# Inserting "When a neuron's groups are born" into the walkthrough

Line numbers are in `../index.html` as committed at 8f0bcbfc4 (932 lines). Three pastes. Paste from the bottom up,
so that the line numbers above each paste still hold; the other modules' INSERT.md files use the same base.

## 1. The scripts: after line 932

Line 932 is `<script src="neuron_across_training/slider.js"></script>`. Add after it (after any other module's
scripts already pasted there):

```html
<script src="births/data.js"></script>
<script src="births/births.js"></script>
```

## 2. The section: after line 612, before line 613

The plan (`VISUALS_PLAN.md`, P6) says the picture replaces lines 582, 617–656, 659, 663–664 and 759–763. Line 582,
the birth rule, is an item of the ordered list in section `readout` (577–611), where a section cannot go, and
617–656 is a figure inside section `meaning`. So the section goes right after section `readout` ends: after the
blank line 612 that follows `</section>` at line 611, and before line 613, `<section id="meaning">`. The reader
then sees births right after the rule that defines them, and before the two neurons of `meaning`.

`../judge/INSERT.md` puts its section at the same place. Put this section first, so that births come before the
judge's calls on the late-born groups: paste it directly before `<section id="judge">` if that is already in.

```html
<section id="births">
  <h2>When a neuron’s groups are born</h2>
  <p>Each row is one group of a neuron’s 50 strongest contexts at the end of training; it is present at a training point when the median activation of its contexts is in the neuron’s top 0.1% there, and born at the first point where it is present and still present at the next.</p>
  <div id="nbirths"></div>
  <p class="record">Source: the verdict’s per-group record for the 600 OLMo-3 7B neurons of the development test, its births per training point, and the saved activations of each group’s contexts at the 16 training points.</p>
</section>

```

The lead-in is one sentence. The section id `births` and the wrapper id `nbirths` are not used elsewhere on the
page (checked by grep on 8f0bcbfc4; `../judge/` uses `judge` and `judgecalls`).

## 3. The stylesheet: after line 10

Line 10 is `<link rel="stylesheet" href="neuron_across_training/slider.css">`. Add after it:

```html
<link rel="stylesheet" href="births/births.css">
```

## The prose the plan says the picture replaces

Quoted from `../index.html`; whether to cut or shorten each is the integrator's call. What the picture carries:
the rule for presence and birth, each group's presence at all 16 points, the birth order inside one neuron, the
count of a neuron's groups born by each point, the births of all 2,454 groups per point, and in its caption that
rows exist only for groups present at the end, that the points are unevenly spaced, and that a birth is a line
crossing. What it does not carry: the judge's calls, which group is strongest at the end, and the records of the
neuron's top 20 before a birth.

- Line 582, step 2 of the readout: "A cluster is born at the first checkpoint where the median activation of its
  contexts is in the pole’s top 0.1% at that checkpoint. It must stay there at the next checkpoint. A cluster born
  after the pole’s first one is late-born." The picture's lead-in states the same rule. The sentence on late-born
  is not in the picture, and the parent test (lines 583–584) needs it, so it should stay.
- Lines 617–656, the figure "The two accounts, drawn from two real OLMo neurons": the picture shows the same two
  neurons (the got neuron in the menu, the lithium neuron by default) from the record, with every group. Its
  arrows and the judge's verdicts are in the judge picture, not here. Two points where the figure differs from the
  record, for whoever keeps it: the figure draws "UC Santa Cruz" as one bar from T5 to T15, but the record has it
  present at T5 to T8, absent at T9 to T11 and present again from T12 (`0000011110001111`, checks below); and it
  draws "lithium" as a bar that "fades after T13", but lithium is not a group of the record (none of the 50
  strongest contexts at the end is lithium), and in `../neuron_across_training/` its top-20 count falls from 19 at
  T12 to 1 at T13.
- Line 659, the got neuron: the four births at steps 5,000, 20,000, 35,000 and 100,000 are in the picture (checked
  below); the token-level description and the judge's 4 of 4 are not.
- Lines 663–664, the lithium neuron: the 15 groups and their births are in the picture, with "UC Santa Cruz" from
  step 8,000 and "Palo Alto" present once at step 60,000 before its birth at step 320,000. The lithium-era top 20
  and the judge's 9 of 12 are not.
- Lines 759–763, "Is anything special about the moment a cluster appears?": the picture's caption carries only the
  conclusion of line 763, that a birth is the crossing of the top-0.1% line and nothing else found marks it. The
  three studies, their divisors and their numbers are not in it, so these lines should stay.
