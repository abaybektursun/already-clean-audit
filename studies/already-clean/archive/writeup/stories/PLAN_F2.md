# Plan: validate and deepen the cross-model report (F), 2026-10-24

Two read-only Opus agents, run together (the two-agent limit), each writing one report beside F:

1. **Validator** (`F_validation.md`). Recompute every number in F's nine commonalities and its differences section from the
   raw files; test each instrument-free finding for artifacts: relative-depth definition (layer/(L−1) vs layer/L), binning
   (quartile vs tercile vs quintile), RARE and dead poles in or out, the 1%-of-max density rule against q99 and q999, the
   histogram's 5.9% bins, the position-0 exclusion, sign symmetry; test the loudness→legibility effect inside depth bins and
   inside each explainer; verify every quoted description and pole id; recompute the regex counts with independent regexes;
   define "Qwen's damaged early layers" precisely (dead on one sign, on both, below the rare rule, double-negative share, per
   layer) and say whether "damaged" is the right word. Tier every finding SAFE / SAFE WITH CAVEAT / DO NOT PUBLISH.
2. **Deepener** (`G_cross_model_deep.md`). New analyses that need no explainer, over every neuron of the three models (about
   1.05 million): firing-token type by depth decoded from the top-50 token ids with each model's own tokenizer; per-layer
   curves (density, q999/max, single-token concentration, dead share, position-0 share, double-negative share) aligned by
   relative depth with a cross-model correlation and a null; loudness→legibility within depth and explainer with bootstrap
   intervals; two-pole relations at population scale (top-50 token overlap between a neuron's signs, by depth); the
   final-layer lean in detail; universality of jobs via a matching with a shuffled-pair null and a within-model duplicate
   baseline; literature context for known phenomena (Qwen massive activations, Gemma first-token sinks) kept separate from the
   data and cited. Every claim marked instrument-free or description-based.

Then: a synthesis reads both, resolves conflicts against the files, and the surviving findings go to the write-up with their
caveats. Data: explain bank `bank/w5/verdict/samples/`, `bank/w2c/<cell>/explain/`; harvest bank
`bank/neurons/w1/{q4b,g4,o7_t15}/1/out/<cell>/{stats,topk}/layer_LL.npz` and the tokenizers under
`w1/q4b/1/out/corpus/corpus/tok/{qwen3,gemma3,olmo3}/tokenizer/`. G4 is `google/gemma-3-4b-pt`.
