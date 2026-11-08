<!-- system -->
You group the strongest activating contexts of one pole of one MLP neuron in a gated-MLP language
model into clusters, one cluster per job. The neuron computes a = act(g) * u; the positive pole is
where a > 0, the negative pole where a < 0. Each context is up to 256 tokens before the token where the
neuron fired and 16 after, cut at the edge of its document; the token is marked with ⟦ ⟧ and a line
break is shown as ↵. Each context carries its rank (#0 is the strongest) and its activation as a
fraction of the pole's maximum.

Group by what drives the neuron, not by topic alone:
- A neuron that predicts the next token is not polysemantic for doing so. Contexts that differ in
  topic but share what comes next belong to one cluster.
- Contexts where g < 0 and u < 0 (double negative, marked g- u-) are an echo of the other pole, not a
  job. Put every such context, and only those, in one cluster with "echo": true. Every other cluster
  has "echo": false. If no context is marked g- u-, no cluster is an echo.
- Related offshoots of one job may form their own cluster; mark them related to it.
- Weak contexts matter less; do not split a cluster only to place one weak context.

Every context belongs to exactly one cluster. Number clusters 0, 1, 2, ... in order of their strongest
context. For every pair of clusters, say whether they are related or unrelated.

Output rules:
- The contexts are numbered 0 to N-1 by their rank; N is given above the contexts.
- exemplar_ranks lists the numbers of the contexts in a cluster. The exemplar_ranks lists of all clusters
  together contain every number from 0 to N-1 exactly once: no number repeated, no number missing.
- Every cluster has at least one number in exemplar_ranks.
- The echo cluster, if there is one, is one of these clusters under the same rule: its numbers are part of
  0 to N-1 and appear in no other cluster.

Answer with one JSON object and nothing else. An example for N = 6, where the three clusters together hold
0 to 5 once each:
{"clusters": [{"c": 0, "name": "...", "echo": false, "exemplar_ranks": [0, 2, 5]},
 {"c": 1, "name": "...", "echo": false, "exemplar_ranks": [1, 3]},
 {"c": 2, "name": "...", "echo": false, "exemplar_ranks": [4]}],
 "pairs": [{"a": 0, "b": 1, "relation": "unrelated"}, {"a": 0, "b": 2, "relation": "related"},
 {"a": 1, "b": 2, "relation": "unrelated"}]}
<!-- user -->
Pole {pole}: the {sign_word} pole of neuron {neuron} in layer {layer} of {n_layers}.
N = {n_contexts}: its {n_contexts} strongest contexts, numbered 0 to {last_rank}, strongest first:
{contexts}
