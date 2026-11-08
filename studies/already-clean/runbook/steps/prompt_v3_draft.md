# Explainer prompt v3: changes against v2 (draft for the explain agent; final wording after the d2 round)

1. Format sentence (replaces "no ⟦ ⟧ markers, no digits ..., no word "context", and no activation values"):
   "They must describe the job in plain words a stranger can act on: no ⟦ ⟧ markers, no reference to a
   numbered or shown context, and no activation values such as a=, g= or u=. Names and formats may carry
   digits (B2B, CSS3, a ratio like 70:30) when the digits are part of the job."
   Remove "Base64" from the X7 paragraph's rule wording? No: the paragraph describes the decoded: line and is right.
   The leak check (52d90981) already matches this sentence.

2. A required first step, before any meaning is read (the wrong-level failure, L33N7506+):
   "First write down, for yourself, what the marked tokens literally are: the strings, whether each is a whole
   word or the first or a middle piece of a longer word, what the next few tokens are, and whether the
   strongest contexts share a token string, a token shape (a word piece, a digit run, a bracket, a line
   start) or only a meaning. A pole that fires on a short common token only when it begins a longer rare
   word has a token-shape job, and a text that uses the word whole will not fire it; your explanation must say
   the shape."

3. Earn the ONE label (over-claiming: 110 ONE labelled, 23 computed):
   "Before you choose ONE, write two near misses yourself: texts that keep the topic and surface but remove
   the thing you named. If you expect the pole to fire on them anyway, your explanation is too narrow or the
   distinction is elsewhere; fix the explanation or lower your confidence. Your confidence must be your
   probability that at least three of five positives written from your sentence reach three tenths of this
   pole's corpus maximum while at most two near misses do."

4. The strength of the test, stated as a number the explainer can reason about:
   "A positive counts only if it drives the pole to at least three tenths of the maximum you see here, in a
   standalone text of one to three sentences with no long document before it. Explanations whose job needs a
   long document (a running topic, a genre, a speaker) should say what short text can still carry it."

Writer prompt v3 (same version number, measured together):
5. Positives: "one to three sentences; put the thing the explanation names where the explanation says it
   sits (the exact token form, inside the structure named: code, a list, a citation, a URL); when the
   explanation names a token shape or a word piece, use words in which that piece occurs exactly as described."
6. Near misses stay as in v2.
