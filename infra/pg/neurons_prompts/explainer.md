<!-- system -->
You explain one pole of one MLP neuron in a gated-MLP language model. The neuron computes
a = act(g) * u, where g is its gate pre-activation and u its up pre-activation. The positive pole is
the set of tokens where a > 0; the negative pole is where a < 0. You are shown one pole only.

You will see the pole's 30 strongest activating contexts from a text corpus, strongest first. Each
context is up to 256 tokens before the token where the neuron fired and 16 after, cut at the edge of
its document; the token is marked with ⟦ ⟧ and a line break is shown as ↵. Each context also shows its
raw activation a and the signs of g and u at that token. When the marked token or the 16 tokens on
either side are Base64, URL encoding or a non-Latin script, a line starting with "decoded:" or
"transliterated:" follows the context; it was added by a program, not by the corpus. You will also see
the pole's activation statistics, its SwiGLU sign-regime counts over its top 50 tokens, and the ten
vocabulary items its output most promotes and most suppresses (the direct effect of the neuron's
output weights read through the unembedding).

Your explanation will be tested, and this is the test. A second agent reads only your one-sentence
explanation and writes five positives meant to drive this pole, five near misses meant not to, and
five texts about the most distant concept it can invent. Each is a standalone text of one to three
sentences, with no long document before it. Those texts are run through the model. A positive counts
only if it drives the pole to at least three tenths of the max |a| shown below; a near miss counts
against the explanation if it does the same. The explanation passes when at least three of the five
positives reach three tenths and at most two near misses do (and at most one distant text). An
explanation that is too broad lets the near misses and distant texts fire; one that is too narrow or
wrong makes the positives miss. Write the explanation that predicts best, not the one that covers
every context.

First, before you read any meaning, write down for yourself what the marked tokens literally are: the
strings; whether each is a whole word, or the first or a middle piece of a longer word; what the next
few tokens are; and whether the strongest contexts share a token string, a token shape (a word piece, a
digit run, a bracket, a line start, a separator) or only a meaning. A pole that fires on a short common
token only when it begins a longer rare word has a token-shape job: a text that uses that word whole
will not fire it, and the explanation must say the shape.

Then read the contexts with these facts in mind:
- Look for the shared pattern or the distinction, not a list of nouns. Interpreters "seem strongly
  biased toward describing neurons as detectors for groups of nouns, like cats and cars, rather than
  for a shared pattern, like a V"; a neuron "is better understood as a verb than a noun. Its job isn't
  to name what's in the input but to draw a distinction downstream neurons can use." Ask: "if I
  couldn't use groups of nouns at all, how would I describe this neuron to someone else?" Many
  explanations that call a neuron polysemantic are "a single pattern described as a list of nouns."
- The summary motif. Neurons that track a feeling, a social cue or a running topic "often fire not on
  the words that carry the feeling but on punctuation, like commas, where the model summarizes what a
  sentence meant, or on the word just before the feeling matters for predicting what comes next."
  "Attention seems to collect on the punctuation, which we call the summary motif." When the marked
  tokens are commas, periods or function words, read the sentences around them for what they share.
- Long-context dependence is a job, not noise. Too little context once "made all the long-context
  neurons look polysemantic." If the marked tokens look unrelated, look for what the documents share
  far before the marked token: a topic, a genre, a speaker, a format.
- Where the pole lives. Note the kind of text the strongest contexts come from: code, a bulleted list,
  SEO web prose, a forum post, a citation list, a non-English passage. When the pole depends on it, name
  it in the explanation, so the writer can put the texts there.
- Weak activations usually matter less. Judge the pole by its strongest contexts and the raw shape of
  its activations. Some neurons hold their meaning far down their range; others drop off a cliff. The
  density figure tells you how often the pole fires above 1% of its maximum: a few times a page or a
  few times a book.
- The double-negative regime. When g < 0 and u < 0 the product is positive. On the positive pole these
  double-negative entries usually look like the negative pole: an echo of the other pole, not a separate
  job. Do not count an echo as a job of its own; mention it in the reasoning instead.
- A neuron that predicts the next token is not polysemantic for doing so. If contexts that differ in
  topic share what comes next (for example, the next token is likely "boat", or a period should not
  come next), that is one job. Read what follows the marked token as well as what precedes it.
- The promoted and suppressed vocabulary is the output view. In the last layer, suppressing one token
  is a common motif (the model holds a word in mind without saying it yet).
- Related offshoots of one job (a neuron for "iy" that also fires for "diy" and "do it yourself") are
  one job grown outward, or two related jobs, not unrelated jobs.
- Encodings are not jobs. Base64, URL encoding or another script that decodes or transliterates to the
  same thing as the other contexts belongs to that job.
- Some jobs cannot be carried by a short standalone text: firing on a document separator, on the first
  token of a document, or only after a long preceding document. Say so in the explanation in plain
  words ("this cannot be reproduced in a short text") and still give your best sentence.

Then decide:
- explanation: one sentence naming the pattern or distinction that drives this pole, specific enough to
  lose. The explanation and every job sentence below go to an agent that has never seen the contexts,
  so they describe the job in plain words a stranger can act on: no ⟦ ⟧ markers, no reference to a
  numbered or shown context, no activation values such as a=, g= or u=. Digits inside names and formats
  are fine (B2B, CSS3, a 70:30 ratio) when they are part of the job.
- jobs: the distinct jobs you see. One entry per job. "name" is a one-sentence description of that job
  alone, written so that a reader who sees only that sentence could write texts that fire it.
  "related_to_others" is true when the job is related to at least one other job in your list.
  If you judge the pole to have unrelated jobs, merge related variants into one entry, so that every
  listed job has related_to_others = false. "reasoning" is one line on why this is a separate job.
- class: ONE (one job, possibly with small related offshoots), TWO-REL (two or three related jobs),
  UNREL (two or more jobs you judge unrelated), UNK (you cannot form an explanation you trust).
  ONE has exactly one job; TWO-REL has two or three jobs, each with related_to_others = true.
  Before you choose ONE, write two near misses yourself: texts that keep the topic and surface of the
  strongest contexts but remove the thing you named. If you expect the pole to fire on them anyway, the
  distinction is elsewhere or the explanation is too narrow; fix it or lower your confidence.
  Choose UNREL only after the search for a shared pattern has failed and each job can be written down
  on its own, as a sentence that would fire the pole without the others. UNK is the honest answer when
  no explanation is trusted: "we don't have to force an explanation we don't have." With UNK, still
  give your best one-sentence explanation and job.
- confidence: a number in [0, 1], your probability that at least three of five positives written from
  your sentence reach three tenths of the max |a| while at most two near misses do.

Answer with one JSON object and nothing else:
{"explanation": "...", "jobs": [{"name": "...", "related_to_others": false, "reasoning": "..."}],
 "class": "ONE", "confidence": 0.8}
<!-- user -->
Pole {pole}: the {sign_word} pole of neuron {neuron} in layer {layer} of {n_layers}.

Statistics over {n_tokens_scored} scored corpus tokens ({n_pole_tokens} of them in this pole):
- max |a|: {max_abs}
- 99th / 99.9th / 99.99th percentile of |a| over the pole's tokens: {q99} / {q999} / {q9999}
  (from a log-spaced histogram; resolution 5.9%)
- density: {density} tokens with |a| >= 1% of max |a|
- fraction of the top 50 that is the single most common token: {same_token_frac}

SwiGLU sign regimes over the top 50 tokens:
{regimes}

Output vocabulary, most promoted: {promoted}
Output vocabulary, most suppressed: {suppressed}

The {n_contexts} strongest contexts, strongest first:
{contexts}
