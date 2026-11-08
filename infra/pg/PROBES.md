# Probe server — request and response formats

`infra/pg/probe_server.py` loads Q4B (Qwen/Qwen3-4B-Base), G4 (google/gemma-3-4b-pt) and O7 (allenai/Olmo-3-1025-7B,
branch stage1-step1413814) once on one GPU and answers JSON requests dropped into a directory. Revisions, residual RMS
and the NEURONSHOW poles come from `studies/neuronshow/plan/plan.json`. All three models in bf16 take about 32 GB; use a
GPU with at least 40 GB, or load fewer with `--models`.

```
# server (repo root on PYTHONPATH; HF_TOKEN in the environment for the Hub download, G4 is gated)
python -m infra.pg.probe_server --requests probe/requests --responses probe/responses [--models Q4B,G4,O7] \
    [--weights Q4B=/path/snapshot_or_hf_cache --weights G4=... --weights O7=...]
# client (any machine that sees the two directories)
python -m infra.pg.probe_client --requests probe/requests --responses probe/responses \
    --json '{"op": "vocab", "model": "G4", "pole": "L18N3736-"}'
```

## Conventions

- Pole: `L<layer>N<neuron><sign>`, e.g. `L07N10202-` (NEURONRW's names). Pushing a negative pole adds −W_down[:, i].
- Dose: residual-RMS units of the pole's layer. The added vector is `dose · RMS_layer · w` with
  `w = sign · W_down[:, i] / ‖W_down[:, i]‖` (fp32), added in fp32 to the decoder layer's output (after the post-MLP norm in
  G4 and O7) and cast back to bf16. Negative doses push the other way. NEURONRW (SPEC A27): Q4B stays fluent at 32; G4 is degraded
  at 16 and mostly not text at 32; O7 loops at 32.
- Positions: absolute token indices of BOS + prompt + generated tokens. BOS is index 0 for G4 and O7; Q4B has none.
  P = prompt length including BOS. `"gen"` = P − 1 and every generated token (NEURONRW's instrument); `"all"` = every
  prompt and generated token; `"first"` = the first non-BOS prompt token; `"last"` = P − 1 only; a list of ints = those
  indices (indices ≥ P land on generated tokens; the first generated token is index P).
- `top5`: `{str, id, logit, p}`, p = softmax of the raw logits at temperature 1 (the sampling temperature is not applied).
- Directory protocol (the vast probe station's, `infra/vast/cells/probe_station.py`): `requests/<rid>.json` is answered
  once into `responses/<rid>.json` (same name; temp file, then rename); requests are never moved or deleted; rid must
  match `[A-Za-z0-9_.-]+`. On the station, `python3 infra/vast/probe.py send request.json` does the naming and the
  waiting; `infra/pg/probe_client.py` does the same on a shared filesystem (it names the file by the request's `"id"`,
  else a new one).
- Unknown fields are rejected. Defaults: `n` 4, `max_new_tokens` 80, `temperature` 0.7, `seed` 20261025. The seed is set with
  `torch.manual_seed(seed)` right before `generate`, so the same request reproduces its samples, and a dose-0 push
  reproduces the unhooked samples token for token.

## Requests

| op | required | optional (default) | returns (`result`) |
|---|---|---|---|
| `steer` | `model`, `pushes: [{pole, dose}, …]`, `prompt` | `n`, `max_new_tokens`, `temperature`, `seed`, `positions` ("gen"), `layer_override` (null) | `samples: [{text, token_ids, n_new_tokens, stopped_eos, bigram_ratio, top5_step1}]`, `top5_prefill`, `pushes: [{pole, dose, layer, layer_injected, rms, alpha_abs}]`, `positions_pushed_prefill`, `n_decode_steps_pushed`, `prompt_tokens` |
| `ablate` | `model`, `poles: [pole, …]`, `prompt` | `n`, `max_new_tokens`, `temperature`, `seed`, `exclude_first` (BOS length) | the steer sample fields, `zeroed: [{pole, n_row_positions_zeroed}]`, `exclude_first` |
| `read` | `model`, `pole`, `text` | — | `tokens: [{i, id, str, a, signed}]`, `max_signed: {i, str, signed}`, `layer`, `neuron`, `sign` |
| `vocab` | `model`, `pole` | `k` (20) | `promoted`, `suppressed` (`{id, str, logit}`), `column_norm`, `banked_top20_attested` (NEURONSHOW poles only), `vocabulary: "full"` |
| `lens` | `model`, `pole`, `dose`, `prompt` | `positions` ("gen") | `layers: [{layer, top5_clean, top5_pushed}]` from the pole's layer to the last, `model_top5_clean`, `model_top5_pushed`, `position`, `positions_pushed` |
| `nll` | `model`, `text` | `pushes` ([]), `positions` ("all"), `ablate` ([]) | `tokens: [{i, str, nll_clean, nll_hooked, delta}]`, `mean_nll_clean`, `mean_nll_hooked`, `mean_delta` |

Details:

- `steer`: several pushes at once; pushes at the same layer add. `layer_override` adds every push's vector (sized by its
  own layer's RMS) at that layer instead. `top5_prefill` is the distribution of the first generated token (shared by all
  samples); `top5_step1` is each sample's distribution at decode step 1 (after its own first token), null when the sample
  ended at its first token. `bigram_ratio` is NEURONRW's distinct-bigram ratio (below 0.5 = repetition loop).
- `ablate`: the NEURONOFF hook (`neuron_off_core.PoleAblator`, mode "pole"): a_i := 0 where sign · a_i > 0, at every
  position ≥ `exclude_first`, prompt and generated tokens alike.
- `read`: a_i is the input of `down_proj` at coordinate i (the neuron's value); the pole fires where `signed` > 0.
- `vocab`: the Arm 0 projection `W_U (gain_final ⊙ w)` in fp32 over the full vocabulary (NEURONRW's judge saw the attested
  vocabulary; `banked_top20_attested` gives that list for the 15 NEURONSHOW poles).
- `lens`: one forward of BOS + prompt with and without the push (default: at the last prompt position only, since there
  is no generation); each decoder output from the pole's layer on, at the last prompt position, through the final norm
  (fp32, the model's gain and epsilon), the unembedding and Gemma's softcap if set. The last layer's lens equals the
  model's own top 5 (checked in fp32 by the tests).
- `nll`: teacher-forced NLL in nats of token i given tokens < i, i from 1 (BOS or Q4B's first token is not scored); text
  limit 2,048 tokens. `delta` = hooked − clean; with no push and no ablation, hooked = clean.

## Responses

`{"id", "ok": true, "op", "request", "result", "elapsed_s", "answered_utc"}`, or on any failure
`{"id", "ok": false, "request" (null if the JSON did not parse), "error", "traceback", "elapsed_s", "answered_utc"}`.
NaN and inf are written as null.

`responses/startup.json` (written before the loop starts): per model `n_layers`, `d_model`, `d_mlp`, `dtype`, `device`,
`bos_id` and `identity` = `{prompt, pole, push_dose0_max_abs_diff, ablator_identity_max_abs_diff,
failing_input_dose1_max_abs_diff, pass}`; pass requires both identity differences exactly 0 and the dose-1 failing input
above 0. `meta` holds the revisions, weight sources, load times, GPU and library versions. If any identity check fails,
the server exits after writing the file.

## Tests

`pytest tests/test_probe_server.py` (tiny random Qwen3, Gemma3 and Olmo3; no download): startup identity; steer (two
poles, negative dose, dose 0 = unhooked generation, same seed = same result, positions gen / last / first / list / all,
layer_override); ablate (zeroed counts, nothing editable = unhooked generation); read (equals `neuron_rw_core.firing_max`);
vocab (equals a direct W_U projection and the column norm); lens (fp32: last layer equals the model, dose 0 equals clean);
nll (equals a plain cross-entropy; dose 0 gives delta 0; a push moves it); the loop on bad JSON, unknown op, unknown field,
missing field, bad pole, unloaded model and out-of-range layer (each answered with ok false and a traceback while the
next good request is answered), requests never moved and answered once, a file name outside the id pattern ignored; the
client round trip through a running loop.
