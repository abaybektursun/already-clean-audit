# Manual backend: answering a requests file with Claude Code subagents

The manual backend is how every D1 verification run is answered today. `neurons_explain.py prepare*`
writes `explain/requests/<batch>.jsonl`; subagents write `explain/responses/<batch>.jsonl`;
`neurons_explain.py collect* --backend manual` validates every answer and logs it to
`explain/calls.jsonl`. Nothing else differs from the API backends.

## What the answering agent does

1. Read `explain/requests/<batch>.jsonl`. Each line is
   `{"id", "role", "model", "temperature", "max_tokens", "system", "user", "schema"}`.
2. For each line, act as the role: follow `system` as the instructions and treat `user` as the input.
   Produce one JSON object that is valid under `schema`.
3. Append one line per request to `explain/responses/<batch>.jsonl`:
   `{"id": <the request id>, "model": <the model id you are running as>, "text": <the JSON object as a string>, "utc": <YYYY-MM-DDTHH:MM:SSZ>}`.
   `text` is the JSON object only: no prose, no code fences. `model` must equal the request's `model`
   and must be the model actually answering; spawn the agent on that model (Opus for `claude-opus-*`).
4. Answer every line. Do not edit the requests file. Do not write any other file.

`collect` fails, listing every id, on a missing answer, an extra or repeated id, a model string
different from the request's, text that is not JSON, or JSON that breaks the schema. It also fails on
role rules: an explanation or job sentence carrying ⟦ ⟧, a digit, the word "context" or an activation
value; a second-reader answer identical to the explainer's; clusters that do not partition the listed
ranks, relate every pair once, or put exactly the g- u- contexts in the one echo cluster. Fix the
named lines in the responses file and run `collect` again. All batches of one collect are validated
before anything is logged, so a failed collect leaves `calls.jsonl` unchanged.

## Redoing refused answers

`collect` writes nothing until every answer passes, and `run` (or the manual agent) only answers ids
that have no response line yet, so a refused answer has to be moved aside before it can be asked again.
The loop is collect → reject → run → collect: `neurons_explain reject --out <OUT> --batch <B> --roles
<r1,r2> --backend <backend> [--max-attempts 3]` applies exactly the checks `collect` applies for those
roles (schema, the role's check, the cross-check), moves every refused answer line from
`responses/<batch>.jsonl` to `responses/<batch>.rejected.jsonl` with the reason and the time, and leaves
missing answers, the usage file and clean batches untouched; `neurons_llm run-many` (or the manual agent)
then answers only the moved ids, and `collect` runs again. The rejection that brings an id to
`--max-attempts` also writes `responses/<batch>.declined.jsonl` (id, role, last reason, the provider's
finish_reason): `run` never asks that id again, and `collect` records it as declined. A declined explainer
or second reader is an explanation not given (the pole is UNK-N); a declined writer or writer_job leaves
the pole without texts (UNK-N, named with the role and the cause in the verdict). A provider decline
(finish_reason `content_filter` or `refusal`) comes back as an empty answer, which collect refuses, so
it goes through the same loop.

## Role isolation

- One agent per batch file, and each batch file holds one role. Explainer and second reader are
  answered by different agents (the second reader is an independent sample). Clusterer samples s0
  and s1 by different agents.
- Judge: one fresh agent per request line. Within a sample file the true record of a cluster appears
  in three requests (two random decoys, one hard decoy) while each decoy record appears once, so an
  agent that sees two judge lines of one cluster can find the true record by repetition
  (`plans/<batch>-judge.jsonl` gives `cluster_requests_in_file` for each request). Request ids are a
  random permutation and the file is in random order, but only one-line-per-agent removes the
  repetition. No agent answers two judge lines. Judge agents append to the same responses file, so
  run them one at a time.
- Writer (`writer`, `writer_job`): one fresh agent per request line as well, so that one pole's
  near misses and distant texts are not shaped by other poles' explanations. Its `user` text is one
  sentence; it must never see contexts, activations, explainer or clusterer requests, or any harvest
  or corpus file.
- No answering agent reads `explain/plans/` (it records which judge record is true),
  `explanations.jsonl`, `texts.jsonl`, `clusters.jsonl`, `judgments.jsonl`, another batch's requests or
  responses, the study spec or the essay. Give each agent only the path of its requests file, the
  path of its responses file, and this runbook.

## Brief to paste into the subagent

> Read `<explain>/requests/<batch>.jsonl` [judge and writer: read only line N]. For each line, follow `system`
> as your instructions and answer `user` as the input. Write one line per request to
> `<explain>/responses/<batch>.jsonl`: `{"id": ..., "model": "<your model id>", "text": "<one JSON
> object valid under the line's schema>", "utc": "<now, UTC>"}`. Write nothing else, read no other
> file, and answer every line.

## 2026-10-21 note: chunked answering of large batches, mixed with API answers

W4 on S (1,200 clusterer requests of about 21k tokens, then about 10,800 judge lines of about 7k) is answered by
Claude Code Sonnet subagents after the API stopped at the provider's user-set limit. Two adaptations:

- **Chunks.** One agent cannot hold a 600-request batch, so `runbook/manual_chunks.py` writes the pending requests of a
  batch in chunks (clusterer 5, judge up to 20) that never hold two requests of one plan group: `pole` for the clusterer
  (so sample s0 and s1 of a pole are still answered by different agents) and `cluster` for the judge (so no agent sees a
  cluster's true record twice, which is what the one-line-per-agent rule above guards against). The chunk files hold
  request rows only; the agent is given the chunk path and the output path and nothing else. `runbook/manual_import.py`
  copies the agent's rows into `responses/<batch>.jsonl` with a fifth key `backend: "manual"`.
- **Mixed batches.** A batch answered partly by an API backend (its sidecar) and partly by agents is collected and
  rejected with `--backend mixed`; `calls.jsonl` records each row's backend. Identity (model id, prompt sha256) is the
  same model; the serving (Claude Code `sonnet` alias, its sampling, no constrained decoding) is what the row's backend
  marks.
