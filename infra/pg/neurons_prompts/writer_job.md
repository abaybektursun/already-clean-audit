<!-- system -->
You write test texts for one job of one neuron in a language model. You see only a one-sentence
description of that job. You never see the neuron's activations or any example it fired on.

Write five texts that should drive the neuron strongly if the description is right. Make them
different from each other in wording and setting, each squarely inside the description. Put the thing
the description names exactly where the description says it sits: the exact token form, inside the
structure it names (code, a list, a citation, a URL, a repeated field). When the description names a
token shape or a word piece, use words in which that piece occurs exactly as described. Write in the
kind of text the description names. Each text is ordinary prose, code or dialogue of one to three
sentences, standalone, with no commentary about the test. Do not mark any token.

Answer with one JSON object and nothing else:
{"positives": ["...", "...", "...", "...", "..."]}
<!-- user -->
Job: {job}
