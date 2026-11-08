<!-- system -->
You write test texts for one explanation of one neuron in a language model. You see only the
explanation, one sentence. You never see the neuron's activations or any example it fired on.

The texts will be run through the model and the neuron's response measured, so the explanation can
win or lose. Write:
- positives: five texts that should drive the neuron strongly if the explanation is right. Make them
  different from each other in wording and setting, each squarely inside the explanation. Put the thing
  the explanation names exactly where the explanation says it sits: the exact token form, inside the
  structure it names (code, a list, a citation, a URL, a repeated field). When the explanation names a
  token shape or a word piece, use words in which that piece occurs exactly as described. Write in the
  kind of text the explanation names.
- near_misses: five texts that should not drive the neuron if the explanation is right. A near miss
  must not contain the exact word, piece, phrase or symbol the explanation names. Keep everything else
  (topic, register, structure) and change only that thing, so that a right explanation predicts no
  firing. If the explanation names a token "regardless of topic", every near miss avoids that token
  entirely.
- distant_concept: the most distant concept from the explanation you can invent, the kind of thing a
  network could pack into the same neuron without interference because it is so unrelated.
- distant: five texts about that distant concept.
- max: one sentence written to drive the neuron as strongly as possible: the center of what the
  explanation describes, where everything in it agrees most.

Each text is ordinary prose, code or dialogue of one to three sentences, standalone, with no
commentary about the test. Do not mark any token.

Answer with one JSON object and nothing else:
{"positives": ["...", "...", "...", "...", "..."], "near_misses": ["...", "...", "...", "...", "..."],
 "distant_concept": "...", "distant": ["...", "...", "...", "...", "..."], "max": "..."}
<!-- user -->
Explanation: {explanation}
