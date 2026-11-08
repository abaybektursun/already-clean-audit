"""D1 LLM calls: one request/response format (DATA_CONTRACT §3) and three backends.

Request line, `explain/requests/<batch>.jsonl`:
    {"id", "role", "model", "temperature", "max_tokens", "system", "user", "schema"}
Response line, `explain/responses/<batch>.jsonl`:
    {"id", "model", "text", "utc"}; `text` is one JSON object valid under the request's schema.
    A row written by a manual agent into a batch an API backend also answered carries "backend": "manual"
    (2026-10-21, W4 on S: the API stopped at the provider limit and Claude Code subagents answer the rest).
Call log, `explain/calls.jsonl`, one line per request/response pair:
    {"id", "role", "model", "backend", "temperature", "prompt_sha256", "response_sha256", "utc"}

Backends:
    manual         a Claude Code subagent answers the requests file and writes the responses file
                   (runbook: neurons_prompts/MANUAL_BACKEND.md).
    mixed          collect/reject only: a batch answered partly by an API backend (its sidecar names it) and
                   partly by manual agents (rows with "backend": "manual"); calls.jsonl logs each row's backend.
    anthropic      Messages API through the `anthropic` SDK; key from ANTHROPIC_API_KEY.
    openai_compat  OpenAI-style chat completions at the settings file's base_url (DeepInfra,
                   OpenRouter, first-party hosts); key from the env var the settings file names.
Every backend ends in the same `collect`, which validates every answer against its request's schema
and raises listing every missing or invalid id. There are no retries: a transport error raises.
The API backends write `responses/<batch>.backend.json` (backend, settings, the model ids the
responses reported) and `responses/<batch>.usage.jsonl` (latency and token counts per call); `collect`
checks the sidecar against the backend it is told, so the logged backend is the one that answered.

CLI (API backends only; the manual backend has no run step):
    python -m infra.pg.neurons_llm run --explain OUT/explain --batch B --settings infra/pg/settings/deepinfra.json
    python -m infra.pg.neurons_llm run-many --explain OUT/explain --batches B1,B2 --settings ... [--concurrency 8]
    python -m infra.pg.neurons_llm run-batch --explain OUT/explain --batches B1,B2 --settings infra/pg/settings/anthropic_sonnet55.json [--dry-run]
    python -m infra.pg.neurons_llm collect-batch --explain OUT/explain
run-batch and collect-batch are the anthropic backend through the Message Batches API: run-batch submits every
pending request of the named batches as Message Batches (split under the 100,000-request and 256 MB limits) and
writes each one's custom_id -> (batch, id) mapping to responses/anthropic_batches/<message batch id>.json;
collect-batch polls every open one until it has ended and writes the same
response, usage and sidecar files `run` writes, so collect and reject are unchanged. A result that is not a
valid answer (errored, expired, canceled, a refusal, another model) goes to responses/<batch>.batch_failures.jsonl
and collect-batch raises after writing everything; the next run-batch asks those ids again. A reply that is not an
answer (refusal, max_tokens) repeated to collect-batch --max-attempts ends in responses/<batch>.declined.jsonl.
The anthropic backend sends bare Anthropic ids (an `anthropic/` prefix is stripped before the call) with one
prompt-cache breakpoint: on the system text, or for the judge after the exemplars (the user text split before
"Record A:"), the prefix a cluster's judge lines share. Usage rows log cache_creation_input_tokens and
cache_read_input_tokens beside prompt_tokens (the uncached remainder) and completion_tokens.
Requests run `--concurrency` at a time; HTTP 429 and 5xx are retried with backoff 2, 4, 8, 16, 32 s
(six attempts), and the attempts are logged per id in the usage file. Answers collect refuses are moved
aside by `reject_many` (CLI: `neurons_explain reject`), after which `run` asks for them again: collect ->
reject -> run -> collect, at most max_attempts rounds per id; an id that reaches max_attempts is
declined (responses/<batch>.declined.jsonl) and not asked again. A provider decline (finish_reason
content_filter or refusal) is recorded as an empty answer, not raised.
The settings file names the backend, the per-role model ids, the temperature and the env var that holds
the key; the key itself is never written, printed or logged. Opus is never called through an API
(SPEC_D1 §2.2 model policy, 2026-10-20): load_settings and run_batch refuse any Opus id before a request
is sent, and make_request refuses a provider-routed one (anthropic/claude-opus...); the manual backend
(Claude Code subagents) stays allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("explainer", "second_reader", "writer", "writer_job", "clusterer", "judge")
BACKENDS = ("manual", "anthropic", "openai_compat")
COLLECT_BACKENDS = BACKENDS + ("mixed",)   # `run` never takes "mixed"; collect and reject do
EFFORTS = ("low", "medium", "high", "xhigh", "max")
REQUEST_KEYS = ("id", "role", "model", "temperature", "max_tokens", "system", "user", "schema")
RESPONSE_KEYS = ("id", "model", "text", "utc")
RESPONSE_KEYS_MANUAL = RESPONSE_KEYS + ("backend",)   # a manual row inside a mixed batch
SCHEMA_KEYWORDS = frozenset({"type", "properties", "required", "additionalProperties", "enum",
                             "items", "minItems", "maxItems", "minimum", "maximum"})
# Structured outputs reject numeric and array-length constraints; `collect` enforces them instead.
API_DROPPED_KEYWORDS = ("minItems", "maxItems", "minimum", "maximum")


def utc_now() -> str:
    """UTC with microseconds, so the order of two artifacts made in the same second is still strict."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_sha256(req: dict) -> str:
    """SHA-256 of the exact prompt the model sees: [system, user] as canonical JSON."""
    return sha256_text(json.dumps([req["system"], req["user"]], ensure_ascii=False))


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_new(path, rows) -> None:
    """Write a new JSONL file; an existing file is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def schema_errors(schema: dict, value, path: str = "$") -> list[str]:
    """Validate `value` against the JSON-schema subset in SCHEMA_KEYWORDS. Returns error strings.
    A schema keyword outside the subset raises, so a schema is never half-checked."""
    unknown = set(schema) - SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"unsupported schema keywords {sorted(unknown)} at {path}")
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return errors + [f"{path}: expected object, got {type(value).__name__}"]
        props = schema.get("properties", {})
        errors += [f"{path}: missing key {k!r}" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected key {k!r}" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errors += schema_errors(sub, value[k], f"{path}.{k}")
    elif kind == "array":
        if not isinstance(value, list):
            return errors + [f"{path}: expected array, got {type(value).__name__}"]
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items > maxItems {schema['maxItems']}")
        for i, item in enumerate(value):
            errors += schema_errors(schema["items"], item, f"{path}[{i}]")
    elif kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
    elif kind in ("integer", "number"):
        ok = isinstance(value, int) if kind == "integer" else isinstance(value, (int, float))
        if not ok or isinstance(value, bool):
            return errors + [f"{path}: expected {kind}, got {type(value).__name__}"]
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    elif kind == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
    elif kind is not None:
        raise ValueError(f"unsupported schema type {kind!r} at {path}")
    return errors


def check_schema(schema: dict, path: str = "$") -> None:
    """Raise if any node of the schema tree uses a keyword outside SCHEMA_KEYWORDS."""
    unknown = set(schema) - SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"unsupported schema keywords {sorted(unknown)} at {path}")
    for name, sub in schema.get("properties", {}).items():
        check_schema(sub, f"{path}.{name}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")


def api_schema(schema: dict) -> dict:
    """The schema sent to an API: the same tree without API_DROPPED_KEYWORDS."""
    out = {}
    for k, v in schema.items():
        if k in API_DROPPED_KEYWORDS:
            continue
        if k == "properties":
            v = {name: api_schema(sub) for name, sub in v.items()}
        elif k == "items":
            v = api_schema(v)
        out[k] = v
    return out


OPUS_RULE = "Opus is not called through the API (SPEC_D1 §2.2 model policy, 2026-10-20); use the manual backend"
OPUS_ROUTED = re.compile(r"^anthropic/claude-opus", re.IGNORECASE)        # a provider-routed Opus id
OPUS_ANY = re.compile(r"^(anthropic[/.])?claude-opus", re.IGNORECASE)     # any Opus id an API could be called with


def refuse_opus(model: str, where: str, pattern=OPUS_ANY) -> None:
    if pattern.match(model):
        raise ValueError(f"{where}: model {model!r}: {OPUS_RULE}")


def make_request(id: str, role: str, model: str, temperature: float, max_tokens: int,
                 system: str, user: str, schema: dict) -> dict:
    """One request line. A provider-routed Opus id (anthropic/claude-opus...) is refused, so no request
    file names it; a bare claude-opus id stays allowed for the manual backend and is refused by `run`."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    refuse_opus(model, id, OPUS_ROUTED)
    if not isinstance(model, str) or not model:
        raise ValueError(f"{id}: model id must be a non-empty string")
    if not id.startswith(role + ":"):
        raise ValueError(f"{id}: id must start with the role {role!r}")
    check_schema(schema)
    return {"id": id, "role": role, "model": model, "temperature": float(temperature),
            "max_tokens": int(max_tokens), "system": system, "user": user, "schema": schema}


def requests_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "requests" / f"{batch}.jsonl"


def responses_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.jsonl"


def backend_sidecar_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.backend.json"


def write_requests(explain_dir, batch: str, requests: list[dict]) -> Path:
    ids = [r["id"] for r in requests]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"batch {batch}: duplicate request ids {dup}")
    for r in requests:
        if tuple(r) != REQUEST_KEYS:
            raise ValueError(f"{r.get('id')}: request keys {tuple(r)} != {REQUEST_KEYS}")
    path = requests_path(explain_dir, batch)
    write_jsonl_new(path, requests)
    return path


def _problem(batch: str, rid: str | None, kind: str, reason: str) -> dict:
    """kind: "batch" (the backend record; asking again cannot fix it), "missing" (no answer yet) or
    "answer" (an answer line that fails; reject moves it so the request is asked again)."""
    return {"batch": batch, "id": rid, "kind": kind, "reason": reason}


def problem_text(p: dict) -> str:
    return p["reason"] if p["id"] is None else f"{p['id']}: {p['reason']}"


def _validate(explain_dir: Path, batch: str, backend: str, check) -> tuple[dict, list[dict], list[dict]]:
    """(answers, call-log rows, problems) for one batch; nothing is written."""
    if backend not in COLLECT_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    problems = []
    sidecar = backend_sidecar_path(explain_dir, batch)
    sidecar_backend = json.loads(sidecar.read_text())["backend"] if sidecar.exists() else None
    if backend == "manual" and sidecar.exists():
        problems.append(_problem(batch, None, "batch", f"{sidecar} exists: batch {batch} was answered by an API backend, not manual"))
    if backend not in ("manual", "mixed"):
        written_by = json.loads(sidecar.read_text())["backend"]
        if written_by != backend:
            problems.append(_problem(batch, None, "batch", f"{sidecar} names backend {written_by!r}, collect was told {backend!r}"))
    requests = read_jsonl(requests_path(explain_dir, batch))
    responses = read_jsonl(responses_path(explain_dir, batch))
    by_id: dict[str, dict] = {}
    row_backend: dict[str, str] = {}   # the backend that answered each row, for the call log
    for resp in responses:
        rid = resp.get("id")
        keys = tuple(sorted(resp))
        if keys == tuple(sorted(RESPONSE_KEYS_MANUAL)) and resp["backend"] == "manual":
            if backend not in ("manual", "mixed"):
                problems.append(_problem(batch, rid, "answer", f"row says backend 'manual', collect was told {backend!r}"))
            row_backend[rid] = "manual"
        elif keys == tuple(sorted(RESPONSE_KEYS)):
            if backend == "mixed" and sidecar_backend is None:
                problems.append(_problem(batch, rid, "answer", "mixed batch: row has no backend key and the batch has no backend sidecar"))
            row_backend[rid] = sidecar_backend if backend == "mixed" else backend
        else:
            problems.append(_problem(batch, rid, "answer", f"response keys {sorted(resp)} != {sorted(RESPONSE_KEYS)} (a manual row in a mixed batch adds backend: manual)"))
            row_backend[rid] = backend
        if rid in by_id:
            problems.append(_problem(batch, rid, "answer", "answered more than once"))
        by_id[rid] = resp
    request_ids = {r["id"] for r in requests}
    problems += [_problem(batch, rid, "answer", f"no such request in batch {batch}") for rid in by_id if rid not in request_ids]
    gave_up = declined(explain_dir, batch)
    problems += [_problem(batch, rid, "answer", "answered after it was declined") for rid in by_id if rid in gave_up]
    answers = {}
    for req in requests:
        rid = req["id"]
        resp = by_id.get(rid)
        if resp is None and rid in gave_up:
            continue
        if resp is None:
            problems.append(_problem(batch, rid, "missing", "missing answer"))
            continue
        if resp.get("model") != req["model"]:
            problems.append(_problem(batch, rid, "answer", f"answered by model {resp.get('model')!r}, request names {req['model']!r}"))
        try:
            obj = json.loads(resp.get("text", ""))
        except json.JSONDecodeError as e:  # recorded as a problem; collect raises below
            problems.append(_problem(batch, rid, "answer", f"text is not JSON ({e.msg} at char {e.pos})"))
            continue
        errs = schema_errors(req["schema"], obj)
        if not errs and check is not None:
            errs = check(req, obj)
        if errs:
            problems.append(_problem(batch, rid, "answer", "; ".join(errs)))
            continue
        answers[rid] = obj
    rows = [{"id": req["id"], "role": req["role"], "model": by_id[req["id"]]["model"], "backend": row_backend[req["id"]],
             "temperature": req["temperature"], "prompt_sha256": prompt_sha256(req),
             "response_sha256": sha256_text(by_id[req["id"]]["text"]), "utc": by_id[req["id"]]["utc"]}
            for req in requests if req["id"] in by_id]
    return answers, rows, problems


def validate_many(explain_dir, batches: list[str], backend: str, check=None, cross_check=None) -> tuple[dict, list, list]:
    """(answers by batch, call-log rows, problems) over several batches; nothing is written.
    `check(request, answer)` returns error strings for rules a JSON schema cannot state;
    `cross_check({batch: answers})` returns (batch, id, reason) for rules across the valid answers."""
    explain_dir = Path(explain_dir)
    out, rows, problems = {}, [], []
    for batch in batches:
        answers, batch_rows, batch_problems = _validate(explain_dir, batch, backend, check)
        out[batch] = answers
        rows += batch_rows
        problems += batch_problems
    if cross_check is not None:
        problems += [_problem(b, rid, "answer", reason) for b, rid, reason in cross_check(out)]
    return out, rows, problems


def collect_many(explain_dir, batches: list[str], backend: str, check=None, cross_check=None) -> dict[str, dict]:
    """Validate every answer of every batch, then log all pairs to calls.jsonl; returns
    {batch: {id: answer object}}. Raises listing every missing, extra, duplicated, model-mismatched
    or schema-invalid answer across all batches, and nothing is logged unless every batch passes."""
    explain_dir = Path(explain_dir)
    out, rows, problems = validate_many(explain_dir, batches, backend, check, cross_check)
    if problems:
        raise ValueError(f"collect {batches}: {len(problems)} problem(s)\n  " + "\n  ".join(map(problem_text, problems)))
    calls = explain_dir / "calls.jsonl"
    logged = {row["id"] for row in read_jsonl(calls)} if calls.exists() else set()
    again = sorted({r["id"] for r in rows} & logged)
    if again:
        raise ValueError(f"collect {batches}: ids already in {calls}: {again[:10]} (n={len(again)})")
    append_jsonl(calls, rows)
    return out


def collect(explain_dir, batch: str, backend: str, check=None) -> dict[str, dict]:
    """collect_many for one batch."""
    return collect_many(explain_dir, [batch], backend, check)[batch]


def rejected_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.rejected.jsonl"


def declined_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.declined.jsonl"


def declined(explain_dir, batch) -> dict[str, dict]:
    """{id: decline row} for one batch: ids rejected --max-attempts times, never asked again."""
    path = declined_path(explain_dir, batch)
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def declined_rows(explain_dir) -> list[dict]:
    """Every decline row of every batch of an explain directory."""
    return [r for p in sorted((Path(explain_dir) / "responses").glob("*.declined.jsonl")) for r in read_jsonl(p)]


PROVIDER_DECLINES = {"content_filter": "provider content filter", "refusal": "provider refusal"}


def decline_text(row: dict) -> str:
    """'<role> declined (<cause>, <n> attempts)' for a decline row."""
    cause = PROVIDER_DECLINES.get(row["finish_reason"], row["reason"])
    return f"{row['role']} declined ({cause}, {row['attempts']} attempts)"


def reject_many(explain_dir, batches: list[str], backend: str, check=None, cross_check=None,
                max_attempts: int = 3) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Move every answer line that collect's validation refuses from responses/<batch>.jsonl to
    responses/<batch>.rejected.jsonl, with the reason and the utc of the rejection, so that `run`
    asks for it again. Missing answers are not rejections. Refuses on a batch-level problem. The
    rejection that brings an id to `max_attempts` also writes responses/<batch>.declined.jsonl (id,
    role, last reason, the provider's finish_reason from the usage file, attempts): the id is not
    asked again and collect treats it as declined. A batch with nothing to reject is left
    byte-identical; kept lines keep their bytes. Returns ({batch: rejected ids}, {batch: declined ids})."""
    explain_dir = Path(explain_dir)
    _, _, problems = validate_many(explain_dir, batches, backend, check, cross_check)
    fixed = [problem_text(p) for p in problems if p["kind"] == "batch"]
    if fixed:
        raise ValueError(f"reject {batches}: asking again cannot fix: {fixed}")
    reasons: dict[str, dict[str, list[str]]] = {}
    for p in problems:
        if p["kind"] == "answer":
            reasons.setdefault(p["batch"], {}).setdefault(p["id"], []).append(p["reason"])
    utc = utc_now()
    gave_up: dict[str, list[str]] = {}
    for batch, by_id in reasons.items():
        path = rejected_path(explain_dir, batch)
        prior = Counter(r["id"] for r in read_jsonl(path)) if path.exists() else Counter()
        roles = {r["id"]: r["role"] for r in read_jsonl(requests_path(explain_dir, batch))}
        usage = usage_path(explain_dir, batch)
        finish = {u["id"]: u.get("finish_reason") for u in read_jsonl(usage)} if usage.exists() else {}
        path = responses_path(explain_dir, batch)
        raw = [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if line.strip()]
        parsed = [json.loads(line) for line in raw]
        append_jsonl(rejected_path(explain_dir, batch),
                     [{**obj, "reason": "; ".join(by_id[obj.get("id")]), "rejected_utc": utc}
                      for obj in parsed if obj.get("id") in by_id])
        done = sorted(rid for rid in by_id if rid in roles and prior[rid] + 1 >= max_attempts)
        append_jsonl(declined_path(explain_dir, batch),
                     [{"id": rid, "role": roles[rid], "reason": "; ".join(by_id[rid]), "finish_reason": finish.get(rid),
                       "attempts": prior[rid] + 1, "utc": utc} for rid in done])
        if done:
            gave_up[batch] = done
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(line for line, obj in zip(raw, parsed) if obj.get("id") not in by_id), encoding="utf-8")
        os.replace(tmp, path)
    return {batch: sorted(by_id) for batch, by_id in reasons.items()}, gave_up


# --- anthropic -------------------------------------------------------------------------------

ANTHROPIC_PREFIX = "anthropic/"         # the provider-routed form (DeepInfra); the Anthropic API takes the bare id
JUDGE_SPLIT = "\n\nRecord A:\n"         # judge user text: exemplars (shared by a cluster's lines) | the two records
CACHE_BREAKPOINT = {"type": "ephemeral"}


def api_model(model: str) -> str:
    """The id sent to the Anthropic API: `anthropic/claude-...` loses its prefix; a bare id is sent as is."""
    return model[len(ANTHROPIC_PREFIX):] if model.startswith(ANTHROPIC_PREFIX) else model


def anthropic_kwargs(req: dict, effort: str, twin: bool = False) -> dict:
    """Messages API arguments. Current Claude models reject sampling parameters, so no temperature
    is sent and only temperature-1.0 requests (the model's own sampling) are accepted. Thinking is left
    adaptive (Sonnet 5.5 rejects disabled thinking); `effort` sets its depth. One prompt-cache breakpoint:
    on the system text, or for the judge on the user text before "Record A:" (the system prompt and the
    exemplars, shared by every judge line of a cluster); the blocks joined are the request's exact text.
    `twin` (the same system and user text is sent again, e.g. clusterer s0/s1 or explainer and second
    reader) adds a second breakpoint at the end of the user text, so the twin reads the whole prompt."""
    if effort not in EFFORTS:
        raise ValueError(f"effort {effort!r} not in {EFFORTS}")
    if req["temperature"] != 1.0:
        raise ValueError(f"{req['id']}: the anthropic backend cannot set temperature "
                         f"{req['temperature']}; only 1.0 requests are answerable")
    model = api_model(req["model"])
    refuse_opus(model, req["id"])
    system, user = req["system"], req["user"]
    if req["role"] == "judge":
        if user.count(JUDGE_SPLIT) != 1:
            raise ValueError(f"{req['id']}: judge user text holds {user.count(JUDGE_SPLIT)} 'Record A:' markers, need 1")
        cut = user.index(JUDGE_SPLIT)
        content = [{"type": "text", "text": user[:cut], "cache_control": CACHE_BREAKPOINT},
                   {"type": "text", "text": user[cut:]}]
    else:
        system = [{"type": "text", "text": system, "cache_control": CACHE_BREAKPOINT}]
        content = [{"type": "text", "text": user}] if twin else user
    if twin:
        content[-1]["cache_control"] = CACHE_BREAKPOINT
    return {"model": model, "max_tokens": req["max_tokens"], "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort,
                              "format": {"type": "json_schema", "schema": api_schema(req["schema"])}}}


def anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)


class Retryable(RuntimeError):
    """HTTP 429 or 5xx: the request may be sent again (call_with_retry)."""


def message_problem(req: dict, msg) -> str | None:
    """Why a Messages API reply cannot be the request's response line, or None: answered by another model
    than the one sent, a stop other than end_turn (a refusal names its category), or not one text block."""
    sent = api_model(req["model"])
    if msg.model != sent:
        return f"answered by model {msg.model!r}, sent {sent!r}"
    if msg.stop_reason != "end_turn":
        category = f" ({msg.stop_details.category})" if msg.stop_reason == "refusal" and msg.stop_details else ""
        return f"stop_reason {msg.stop_reason!r}{category}"
    n_text = sum(b.type == "text" for b in msg.content)
    if n_text != 1:
        return f"{n_text} text blocks in the response"
    return None


def anthropic_usage(rid: str, msg, latency_s: float | None, utc: str) -> dict:
    """One usage row. prompt_tokens is the uncached input; the prompt is prompt_tokens +
    cache_creation_input_tokens + cache_read_input_tokens."""
    u = msg.usage
    return {"id": rid, "model": msg.model, "message_id": msg.id, "latency_s": latency_s, "finish_reason": msg.stop_reason,
            "prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
            "cache_creation_input_tokens": u.cache_creation_input_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens, "utc": utc}


def anthropic_response(req: dict, msg, utc: str) -> dict:
    return {"id": req["id"], "model": msg.model, "text": next(b.text for b in msg.content if b.type == "text"), "utc": utc}


def call_anthropic(client, req: dict, effort: str) -> tuple[dict, dict]:
    import anthropic
    t0 = time.monotonic()
    try:
        msg = client.messages.create(**anthropic_kwargs(req, effort))
    except anthropic.APIStatusError as e:
        if e.status_code == 429 or e.status_code >= 500:
            raise Retryable(f"{req['id']}: HTTP {e.status_code}: {e.message}") from e
        raise
    latency = round(time.monotonic() - t0, 3)
    utc = utc_now()
    usage = {**anthropic_usage(req["id"], msg, latency, utc), "request_id": msg._request_id}
    problem = message_problem(req, msg)
    if problem is not None:
        raise RuntimeError(f"{req['id']}: {problem} (usage {usage})")
    return anthropic_response(req, msg, utc), usage


# --- openai_compat ---------------------------------------------------------------------------

RESPONSE_FORMATS = ("json_schema", "json_object", "none")


def provider_object(order: list[str], quantizations: list[str] | None) -> dict:
    """OpenRouter provider pinning: only the named providers, in order, never a fallback."""
    if not order:
        raise ValueError("provider order must name at least one provider")
    out = {"order": list(order), "allow_fallbacks": False}
    if quantizations:
        out["quantizations"] = list(quantizations)
    return out


def openai_payload(req: dict, provider: dict | None, response_format: str) -> dict:
    """Chat-completions body. `response_format` is the backend's structured-output mode from its
    settings file: json_schema (strict schema), json_object (any JSON object; collect checks the
    schema), or none (the prompt alone asks for JSON)."""
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"response_format {response_format!r} not in {RESPONSE_FORMATS}")
    payload = {"model": req["model"],
               "messages": [{"role": "system", "content": req["system"]},
                            {"role": "user", "content": req["user"]}],
               "temperature": req["temperature"], "max_tokens": req["max_tokens"]}
    if response_format == "json_schema":
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": req["role"], "strict": True,
                                                      "schema": api_schema(req["schema"])}}
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if provider is not None:
        payload["provider"] = provider
    return payload


def call_openai_compat(settings: dict, key: str, req: dict) -> tuple[dict, dict]:
    """One chat-completions call. The key goes only into the Authorization header; an HTTP error is
    re-raised with the server's error body (which never contains the request headers)."""
    url = settings["base_url"].rstrip("/") + "/chat/completions"
    body = json.dumps(openai_payload(req, settings.get("provider"), settings["response_format"])).encode("utf-8")
    http = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(http, timeout=600) as f:
            data = json.load(f)
    except urllib.error.HTTPError as e:
        kind = Retryable if e.code == 429 or e.code >= 500 else RuntimeError
        raise kind(f"{req['id']}: HTTP {e.code} from {url}: {e.read().decode('utf-8', 'replace')[:2000]}") from e
    latency = time.monotonic() - t0
    choice = data["choices"][0]
    reason = choice["finish_reason"]
    if reason not in ("stop", *PROVIDER_DECLINES):
        raise RuntimeError(f"{req['id']}: finish_reason {reason!r} (usage {data.get('usage')})")
    utc = utc_now()
    text = choice["message"]["content"] if reason == "stop" else ""     # a provider decline is an empty answer
    return ({"id": req["id"], "model": data["model"], "text": text, "utc": utc},
            {"id": req["id"], "model": data["model"], "latency_s": round(latency, 3), "finish_reason": reason,
             "prompt_tokens": data["usage"]["prompt_tokens"],
             "completion_tokens": data["usage"]["completion_tokens"], "utc": utc})


# --- run (API backends) ----------------------------------------------------------------------

SETTINGS_KEYS = {"anthropic": {"backend", "effort", "models", "temperature"},
                 "openai_compat": {"backend", "base_url", "key_env", "models", "temperature", "response_format"}}


def load_settings(path) -> dict:
    """A backend settings file (infra/pg/settings/*.json): backend, per-role model ids, temperature,
    and the backend's own fields. The key is never in it, only the name of the env var holding it."""
    settings = json.loads(Path(path).read_text())
    backend = settings.get("backend")
    if backend not in SETTINGS_KEYS:
        raise ValueError(f"{path}: backend {backend!r} is not an API backend")
    missing = SETTINGS_KEYS[backend] - set(settings)
    extra = set(settings) - SETTINGS_KEYS[backend] - {"provider", "notes", "retired"}
    if missing or extra:
        raise ValueError(f"{path}: missing {sorted(missing)}, unknown {sorted(extra)}")
    if "retired" in settings:
        raise ValueError(f"{path}: retired ({settings['retired']})")
    if set(settings["models"]) != set(ROLES):
        raise ValueError(f"{path}: models must name every role {ROLES}")
    for role, model in settings["models"].items():
        refuse_opus(model, f"{path} ({role})")
    return settings


def response_models(explain_dir, batch) -> dict[str, str]:
    """{id: the model id its response line reports} for one batch."""
    path = responses_path(explain_dir, batch)
    return {r["id"]: r["model"] for r in read_jsonl(path)} if path.exists() else {}


def usage_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.usage.jsonl"


RETRY_ATTEMPTS = 6        # per request, on HTTP 429 and 5xx
RETRY_BASE_S = 2.0        # backoff 2, 4, 8, 16, 32 s between attempts


def backoff_sleep(seconds: float) -> None:
    time.sleep(seconds)


def call_with_retry(call, req: dict) -> tuple[dict, dict]:
    """call(req) with up to RETRY_ATTEMPTS attempts on Retryable errors, exponential backoff; the
    attempt count goes into the usage row. Any other error, or the last Retryable, raises."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response, usage = call(req)
        except Retryable as e:
            if attempt == RETRY_ATTEMPTS:
                raise RuntimeError(f"{req['id']}: gave up after {RETRY_ATTEMPTS} attempts: {e}") from e
            backoff_sleep(RETRY_BASE_S * 2 ** (attempt - 1))
            continue
        return response, {**usage, "attempts": attempt}


def run_batch(explain_dir, batch: str, settings: dict, concurrency: int = 8) -> Path:
    """Answer every request of `batch` through the API backend of `settings`, `concurrency` requests
    at a time, appending each response (and its latency, token counts and attempts to
    responses/<batch>.usage.jsonl) as it arrives. Only this thread writes files. Every request's model
    and temperature must equal the settings' for its role, checked on the requests still to be sent
    (answered lines are skipped and never checked or sent again, so a batch can be resumed on another
    model). On any failure, requests not yet started are cancelled, finished answers are kept, and the
    run raises; ids already answered or declined are skipped, so running it again resumes. The sidecar
    responses/<batch>.backend.json holds the first settings, any later settings of a resumed batch
    (later_settings), and the model ids the responses reported."""
    if concurrency < 1:
        raise ValueError(f"concurrency {concurrency} < 1")
    backend = settings["backend"]
    out = responses_path(explain_dir, batch)
    pending = pending_requests(explain_dir, batch, settings)
    sidecar, record = open_sidecar(explain_dir, batch, settings)
    if backend == "anthropic":
        client = anthropic_client()
        call = lambda r: call_anthropic(client, r, settings["effort"])  # noqa: E731
    else:
        key = os.environ[settings["key_env"]]
        call = lambda r: call_openai_compat(settings, key, r)  # noqa: E731
    errors = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call_with_retry, call, req) for req in pending]
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            if fut.exception() is not None:
                errors.append(fut.exception())
                for other in futures:
                    other.cancel()
                continue
            response, usage = fut.result()
            append_jsonl(out, [response])
            append_jsonl(usage_path(explain_dir, batch), [usage])
            note_model(sidecar, record, response["model"])
    if errors:
        raise RuntimeError(f"batch {batch}: {len(errors)} request(s) failed; finished answers are kept "
                           f"and a rerun resumes. First: {errors[0]}") from errors[0]
    return out


def pending_requests(explain_dir, batch: str, settings: dict) -> list[dict]:
    """The requests of `batch` still to be sent: not answered, not declined. Only these are checked:
    answered lines (another model, Opus included) are a record and are never sent again. Every one must
    name the settings' model and temperature for its role, and no Opus id is sent (settings or request)."""
    for role, model in settings["models"].items():                    # before any request is sent
        refuse_opus(model, f"settings ({role})")
    out = responses_path(explain_dir, batch)
    done = {row["id"] for row in read_jsonl(out)} if out.exists() else set()
    done |= set(declined(explain_dir, batch))                          # declined ids are not asked again
    pending = [r for r in read_jsonl(requests_path(explain_dir, batch)) if r["id"] not in done]
    for req in pending:
        refuse_opus(req["model"], req["id"])
    wrong = [r["id"] for r in pending if r["model"] != settings["models"][r["role"]]
             or r["temperature"] != settings["temperature"]]
    if wrong:
        raise ValueError(f"batch {batch}: model or temperature differs from the settings for {wrong[:10]}")
    return pending


def open_sidecar(explain_dir, batch: str, settings: dict) -> tuple[Path, dict]:
    """Write (or extend) responses/<batch>.backend.json before any request is sent: the first settings,
    later settings of a resumed batch, the model ids the responses reported."""
    sidecar = backend_sidecar_path(explain_dir, batch)
    record = {"backend": settings["backend"], "settings": settings, "reported_models": [], "utc": utc_now()}
    if sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["backend"] != settings["backend"]:
            raise ValueError(f"{sidecar} records backend {record['backend']!r}; refusing to mix backends")
        later = record["later_settings"] if "later_settings" in record else []
        if settings != record["settings"] and settings not in [r["settings"] for r in later]:
            record["later_settings"] = later + [{"settings": settings, "utc": utc_now()}]   # a resumed, mixed batch
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(record, indent=1))
    return sidecar, record


def note_model(sidecar: Path, record: dict, model: str) -> None:
    if model not in record["reported_models"]:
        record["reported_models"] = sorted(record["reported_models"] + [model])
        sidecar.write_text(json.dumps(record, indent=1))


# --- Message Batches (anthropic backend) ------------------------------------------------------

CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
BATCH_MAX_REQUESTS = 100_000         # per Message Batch (claude-api skill, python/claude-api/batches.md: "Up to
BATCH_MAX_BYTES = 256_000_000        # 100,000 requests or 256 MB per batch"; 256 MB read as 256e6 bytes, the stricter)
POLL_S = 60.0


def submissions_dir(explain_dir) -> Path:
    return Path(explain_dir) / "responses" / "anthropic_batches"


def open_submissions(explain_dir) -> list[Path]:
    """Submission records (one per Message Batch) that collect-batch has not collected yet."""
    d = submissions_dir(explain_dir)
    return [p for p in sorted(d.glob("*.json")) if "collected_utc" not in json.loads(p.read_text())] if d.exists() else []


def chunk_requests(requests: list[dict], prompts: list[tuple]) -> list[list[int]]:
    """Split request indices into Message Batches under BATCH_MAX_REQUESTS and BATCH_MAX_BYTES (the JSON body
    `{"requests": [...]}`), in order, never between consecutive requests with the same prompt (twins stay in one
    batch, so the second can read the first's cache). A single group above a limit raises."""
    overhead = len(b'{"requests": []}')
    chunks, cur, cur_bytes = [], [], overhead
    k = 0
    while k < len(requests):
        group = [k]
        while group[-1] + 1 < len(requests) and prompts[group[-1] + 1] == prompts[k]:
            group.append(group[-1] + 1)
        size = sum(len(json.dumps(requests[i]).encode("utf-8")) + 2 for i in group)      # + the ", " separator
        if len(group) > BATCH_MAX_REQUESTS or overhead + size > BATCH_MAX_BYTES:
            raise ValueError(f"requests {group[0]}..{group[-1]} share one prompt and alone exceed a batch limit")
        if cur and (len(cur) + len(group) > BATCH_MAX_REQUESTS or cur_bytes + size > BATCH_MAX_BYTES):
            chunks.append(cur)
            cur, cur_bytes = [], overhead
        cur += group
        cur_bytes += size
        k = group[-1] + 1
    return chunks + [cur]


def run_message_batch(explain_dir, batches: list[str], settings: dict, dry_run: bool = False) -> list[str]:
    """Submit every pending request of `batches` (pending_requests) through the anthropic backend's Message
    Batches, with the request shapes of the live path. Requests are ordered by (system, user), so lines that
    share a cached prefix, and identical twins, are submitted one after the other; a request whose system and
    user text occur more than once in the submission gets the twin breakpoint. A submission above a batch limit
    is split into several Message Batches (chunk_requests). custom_ids are q000000, q000001, ... in that order;
    each Message Batch's mapping to (batch, id) is written to responses/anthropic_batches/<message batch id>.json.
    Refused while an earlier submission is not collected, so no id is asked twice at once. Returns the Message
    Batch ids (none on a dry run or when nothing is pending); a dry run builds and checks everything and sends
    nothing."""
    explain_dir = Path(explain_dir)
    if settings["backend"] != "anthropic":
        raise ValueError(f"run-batch is the anthropic backend's; settings name {settings['backend']!r}")
    still_open = open_submissions(explain_dir)
    if still_open:
        raise ValueError(f"{still_open[0]} is not collected yet; run collect-batch first")
    work = sorted(((b, r) for b in batches for r in pending_requests(explain_dir, b, settings)),
                  key=lambda br: (br[1]["system"], br[1]["user"]))
    if not work:
        print(f"{batches}: nothing pending")
        return []
    prompts = [(r["system"], r["user"]) for _, r in work]
    twins = Counter(prompts)                                           # a prompt sent twice: twins
    entries = [{"custom_id": f"q{k:06d}", "batch": b, "id": r["id"]} for k, (b, r) in enumerate(work)]
    requests = [{"custom_id": e["custom_id"], "params": anthropic_kwargs(r, settings["effort"], twin=twins[p] > 1)}
                for e, (_, r), p in zip(entries, work, prompts)]
    bad = [e["custom_id"] for e in entries if not CUSTOM_ID.match(e["custom_id"])]
    if bad:
        raise ValueError(f"cannot submit: custom_ids {bad[:5]} break {CUSTOM_ID.pattern}")
    chunks = chunk_requests(requests, prompts)
    sizes = [len(json.dumps({"requests": [requests[i] for i in c]}).encode("utf-8")) for c in chunks]
    if max(sizes) > BATCH_MAX_BYTES or max(len(c) for c in chunks) > BATCH_MAX_REQUESTS:
        raise ValueError(f"a chunk breaks a batch limit: {[len(c) for c in chunks]} requests, {sizes} bytes")
    models = Counter(p["params"]["model"] for p in requests)
    print(f"{'dry run: ' if dry_run else ''}{len(requests)} requests from {batches} in {len(chunks)} Message Batch(es) of "
          f"{[len(c) for c in chunks]} requests and {sizes} bytes, models {dict(models)}, effort {settings['effort']}")
    if dry_run:
        return []
    for b in batches:
        open_sidecar(explain_dir, b, settings)
    client = anthropic_client()
    ids = []
    for chunk, size in zip(chunks, sizes):
        message_batch = client.messages.batches.create(requests=[requests[i] for i in chunk])
        record = {"message_batch_id": message_batch.id, "batches": batches, "settings": settings,
                  "submitted_utc": utc_now(), "n": len(chunk), "bytes": size, "requests": [entries[i] for i in chunk]}
        path = submissions_dir(explain_dir) / f"{message_batch.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"{path} exists")
        path.write_text(json.dumps(record, indent=1))
        print(f"{message_batch.id}: submitted {len(chunk)} requests; mapping in {path}")
        ids.append(message_batch.id)
    return ids


def poll_sleep(seconds: float) -> None:
    time.sleep(seconds)


def failures_path(explain_dir, batch) -> Path:
    return Path(explain_dir) / "responses" / f"{batch}.batch_failures.jsonl"


def collect_message_batch(explain_dir, record_path: Path, max_attempts: int = 3) -> list[dict]:
    """Poll one submission until processing_status is "ended", then write, per batch, the response lines
    and usage rows the live path writes (usage rows add message_batch_id and custom_id; latency_s is
    None) and the reported model into the sidecar. Results arrive in any order and are keyed by
    custom_id. Every result that is not a valid answer (errored, expired, canceled, or a reply
    message_problem refuses) is a row of responses/<batch>.batch_failures.jsonl; a succeeded reply's
    usage is logged either way (it is billed). A reply that is not an answer (refusal, max_tokens, ...)
    is an attempt: when an id's attempts (such replies plus its rejected answers) reach `max_attempts`,
    it is declined in responses/<batch>.declined.jsonl (reject's row shape) and not asked again.
    Errored, expired and canceled results are not attempts. Nothing is written before every result is
    read. The record gets collected_utc and counts. Returns the failures that were not declined."""
    explain_dir = Path(explain_dir)
    record = json.loads(record_path.read_text())
    mb_id = record["message_batch_id"]
    client = anthropic_client()
    while True:
        status = client.messages.batches.retrieve(mb_id)
        if status.processing_status == "ended":
            break
        print(f"{mb_id}: {status.processing_status} {status.request_counts}", flush=True)
        poll_sleep(POLL_S)
    by_cid = {e["custom_id"]: e for e in record["requests"]}
    reqs = {b: {r["id"]: r for r in read_jsonl(requests_path(explain_dir, b))} for b in record["batches"]}
    answers = {b: [] for b in record["batches"]}
    usage = {b: [] for b in record["batches"]}
    failures = {b: [] for b in record["batches"]}
    seen = set()
    for res in client.messages.batches.results(mb_id):
        if res.custom_id not in by_cid or res.custom_id in seen:
            raise ValueError(f"{mb_id}: result {res.custom_id!r} is unknown or repeated")
        seen.add(res.custom_id)
        e = by_cid[res.custom_id]
        utc = utc_now()
        fail = {"id": e["id"], "custom_id": res.custom_id, "message_batch_id": mb_id, "type": res.result.type, "utc": utc}
        if res.result.type == "succeeded":
            msg = res.result.message
            usage[e["batch"]].append({**anthropic_usage(e["id"], msg, None, utc), "message_batch_id": mb_id,
                                      "custom_id": res.custom_id})
            problem = message_problem(reqs[e["batch"]][e["id"]], msg)
            if problem is None:
                answers[e["batch"]].append(anthropic_response(reqs[e["batch"]][e["id"]], msg, utc))
            else:
                failures[e["batch"]].append({**fail, "reason": problem, "finish_reason": msg.stop_reason})
        elif res.result.type == "errored":
            err = res.result.error.error
            failures[e["batch"]].append({**fail, "reason": f"{err.type}: {err.message}"})
        else:                                                           # expired or canceled
            failures[e["batch"]].append({**fail, "reason": res.result.type})
    for cid in sorted(set(by_cid) - seen):
        e = by_cid[cid]
        failures[e["batch"]].append({"id": e["id"], "custom_id": cid, "message_batch_id": mb_id, "type": "no_result",
                                     "reason": "no result in the batch's results", "utc": utc_now()})
    open_failures, gave_up = [], {}
    for b in record["batches"]:
        append_jsonl(responses_path(explain_dir, b), answers[b])
        append_jsonl(usage_path(explain_dir, b), usage[b])
        if failures[b]:
            append_jsonl(failures_path(explain_dir, b), failures[b])
        sidecar = backend_sidecar_path(explain_dir, b)
        side = json.loads(sidecar.read_text())
        for model in sorted({a["model"] for a in answers[b]}):
            note_model(sidecar, side, model)
        gave_up[b] = decline_at_cap(explain_dir, b, failures[b], reqs[b], max_attempts)
        open_failures += [f for f in failures[b] if f["id"] not in gave_up[b]]
    counts = {b: {"answers": len(answers[b]), "failures": len(failures[b]), "declined": len(gave_up[b])}
              for b in record["batches"]}
    record.update({"collected_utc": utc_now(), "ended_at": str(status.ended_at), "counts": counts})
    record_path.write_text(json.dumps(record, indent=1))
    print(f"{mb_id}: collected {counts}")
    for b in record["batches"]:
        for rid in gave_up[b]:
            print(f"declined: {rid}: {decline_text(declined(explain_dir, b)[rid])}")
    return open_failures


def decline_at_cap(explain_dir, batch: str, failures: list[dict], reqs: dict[str, dict], max_attempts: int) -> list[str]:
    """Decline every id of `failures` whose attempts reach `max_attempts`: replies that were not answers
    (batch_failures rows of type succeeded, this round's included) plus rejected answers. Returns the ids."""
    path = failures_path(explain_dir, batch)
    replies = Counter(f["id"] for f in read_jsonl(path) if f["type"] == "succeeded") if path.exists() else Counter()
    rej = rejected_path(explain_dir, batch)
    rejected = Counter(r["id"] for r in read_jsonl(rej)) if rej.exists() else Counter()
    last = {f["id"]: f for f in failures if f["type"] == "succeeded"}
    done = sorted(rid for rid in last if replies[rid] + rejected[rid] >= max_attempts)
    append_jsonl(declined_path(explain_dir, batch), [
        {"id": rid, "role": reqs[rid]["role"],
         "reason": f"{last[rid]['reason']} on {replies[rid]} of {replies[rid] + rejected[rid]} Message Batches attempts "
                   f"(responses/{batch}.batch_failures.jsonl)",
         "finish_reason": last[rid]["finish_reason"], "attempts": replies[rid] + rejected[rid], "utc": utc_now()}
        for rid in done])
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="neurons_llm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="answer one batch through an API backend")
    run.add_argument("--batch", required=True)
    many = sub.add_parser("run-many", help="answer several batches in sequence (one wave, one command)")
    many.add_argument("--batches", required=True, help="comma-separated batch names")
    mb = sub.add_parser("run-batch", help="submit the pending requests of batches as one Message Batch (anthropic)")
    mb.add_argument("--batches", required=True, help="comma-separated batch names")
    mb.add_argument("--dry-run", action="store_true", help="build and check the submission; send nothing")
    for p in (run, many, mb):
        p.add_argument("--explain", required=True, help="the explain/ directory")
        p.add_argument("--settings", required=True, help="backend settings file, e.g. infra/pg/settings/deepinfra.json")
    for p in (run, many):
        p.add_argument("--concurrency", type=int, default=8, help="requests in flight at once (default 8)")
    cb = sub.add_parser("collect-batch", help="wait for the open Message Batch submissions and write their answers")
    cb.add_argument("--explain", required=True, help="the explain/ directory")
    cb.add_argument("--max-attempts", type=int, default=3, help="replies without an answer before an id is declined (default 3)")
    a = ap.parse_args(argv)
    if a.cmd == "collect-batch":
        paths = open_submissions(a.explain)
        if not paths:
            raise ValueError(f"{submissions_dir(a.explain)}: no open submission")
        failed = [f for path in paths for f in collect_message_batch(Path(a.explain), path, a.max_attempts)]
        if failed:
            raise RuntimeError(f"{len(failed)} request(s) failed, listed in responses/<batch>.batch_failures.jsonl; answers "
                               f"are kept and the next run-batch asks the failed ids again. First: {failed[0]}")
        return
    settings = load_settings(a.settings)
    if a.cmd == "run-batch":
        run_message_batch(Path(a.explain), a.batches.split(","), settings, a.dry_run)
        return
    for batch in (a.batch,) if a.cmd == "run" else a.batches.split(","):
        print(run_batch(Path(a.explain), batch, settings, a.concurrency))


if __name__ == "__main__":
    main()
