"""neurons_llm: the request/response format, schema validation, collect's failure listing, and the
API backends' request construction (no API is called; no keys are read except a fake one).

    uv run --no-project --with pytest --with transformers --with tokenizers --with anthropic \
        --with numpy pytest infra/pg/tests/test_neurons_llm.py -q
"""
from __future__ import annotations

import io
import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from infra.pg import neurons_llm as llm

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer", "items", "p"],
          "properties": {"answer": {"type": "string", "enum": ["A", "B"]},
                         "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}},
                         "p": {"type": "number", "minimum": 0, "maximum": 1}}}


def _req(i, model="m-1", temperature=1.0):
    return llm.make_request(f"judge:L00N{i}+:c0:r0s0", "judge", model, temperature, 100, "sys", f"user {i}", SCHEMA)


def _jreq(i, model="claude-sonnet-5-5", exemplars=None):
    """A judge request in the judge prompt's shape: exemplars, then Record A and Record B."""
    ex = f"- exemplar {i}" if exemplars is None else exemplars
    return llm.make_request(f"judge:L00N{i}+:c0:r0s0", "judge", model, 1.0, 100, "sys",
                            f"Exemplars of the job:\n{ex}\n\nRecord A:\n- rec {i}\n\nRecord B:\n- other", SCHEMA)


def _msg(model, text, stop="end_turn", cache=(5, 11), category=None):
    """A Messages API reply as the SDK returns it (the fields the backend reads)."""
    content = [SimpleNamespace(type="thinking", thinking="")] + ([SimpleNamespace(type="text", text=text)] if text else [])
    return SimpleNamespace(id="msg_01", _request_id="req_01", model=model, stop_reason=stop,
                           stop_details=SimpleNamespace(category=category) if category else None,
                           usage=SimpleNamespace(input_tokens=7, output_tokens=3, cache_creation_input_tokens=cache[0],
                                                 cache_read_input_tokens=cache[1]), content=content)


OK_TEXT = json.dumps({"answer": "B", "items": [3, 4], "p": 1})


def _respond(explain, batch, texts, model="m-1"):
    llm.write_jsonl_new(llm.responses_path(explain, batch),
                        [{"id": rid, "model": model, "text": t, "utc": "2026-10-21T00:00:00Z"} for rid, t in texts])


def test_schema_errors_cover_each_keyword():
    ok = {"answer": "A", "items": [1, 2], "p": 0.5}
    assert llm.schema_errors(SCHEMA, ok) == []
    bad = {"answer": "C", "items": [1, "x", 3], "p": 1.5, "extra": 1}
    errs = " ".join(llm.schema_errors(SCHEMA, bad))
    for needle in ("not in", "maxItems", "expected integer", "maximum", "unexpected key"):
        assert needle in errs
    assert "missing key 'p'" in " ".join(llm.schema_errors(SCHEMA, {"answer": "A", "items": [1, 2]}))
    assert llm.schema_errors({"type": "integer"}, True) != []          # a bool is not an integer
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        llm.check_schema({"type": "object", "properties": {"x": {"type": "string", "pattern": "a"}}})
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        _ = llm.make_request("judge:x:0", "judge", "m", 1.0, 10, "s", "u", {"type": "string", "format": "date"})


def test_api_schema_drops_unsupported_constraints_recursively():
    sent = llm.api_schema(SCHEMA)
    text = json.dumps(sent)
    for k in llm.API_DROPPED_KEYWORDS:
        assert k not in text
    assert sent["properties"]["answer"]["enum"] == ["A", "B"] and sent["additionalProperties"] is False
    assert "minItems" in json.dumps(SCHEMA)                          # the original is untouched


def test_make_request_rejects_bad_role_model_and_id():
    with pytest.raises(ValueError, match="unknown role"):
        llm.make_request("x:1:0", "x", "m", 1.0, 1, "s", "u", SCHEMA)
    with pytest.raises(ValueError, match="model id"):
        llm.make_request("judge:1:0", "judge", "", 1.0, 1, "s", "u", SCHEMA)
    with pytest.raises(ValueError, match="must start with the role"):
        llm.make_request("writer:1:0", "judge", "m", 1.0, 1, "s", "u", SCHEMA)


def test_collect_round_trip_logs_hashes(tmp_path):
    reqs = [_req(i) for i in range(3)]
    llm.write_requests(tmp_path, "b", reqs)
    texts = [(r["id"], json.dumps({"answer": "A", "items": [1, 2], "p": 0.1})) for r in reqs]
    _respond(tmp_path, "b", texts)
    answers = llm.collect(tmp_path, "b", "manual")
    assert set(answers) == {r["id"] for r in reqs}
    calls = llm.read_jsonl(tmp_path / "calls.jsonl")
    assert [c["id"] for c in calls] == [r["id"] for r in reqs]
    assert calls[0]["prompt_sha256"] == llm.sha256_text(json.dumps(["sys", "user 0"]))
    assert calls[0]["response_sha256"] == llm.sha256_text(texts[0][1])
    assert {c["backend"] for c in calls} == {"manual"}
    with pytest.raises(ValueError, match="already in"):
        llm.collect(tmp_path, "b", "manual")                         # never logged twice


def test_collect_lists_every_bad_id_and_logs_nothing(tmp_path):
    reqs = [_req(i) for i in range(5)]
    llm.write_requests(tmp_path, "b", reqs)
    good = json.dumps({"answer": "A", "items": [1, 2], "p": 0.1})
    _respond(tmp_path, "b", [
        (reqs[0]["id"], good),
        (reqs[1]["id"], "not json"),
        (reqs[2]["id"], json.dumps({"answer": "Z", "items": [1, 2], "p": 0.1})),
        (reqs[3]["id"], good),
        (reqs[3]["id"], good),                                      # repeated
        ("judge:extra:c0:r0s0", good)])                              # reqs[4] missing, one extra
    with pytest.raises(ValueError) as e:
        llm.collect(tmp_path, "b", "manual")
    msg = str(e.value)
    for needle in (f"{reqs[1]['id']}: text is not JSON", f"{reqs[2]['id']}: $.answer",
                   f"{reqs[3]['id']}: answered more than once", f"{reqs[4]['id']}: missing answer",
                   "judge:extra:c0:r0s0: no such request"):
        assert needle in msg
    assert not (tmp_path / "calls.jsonl").exists()


def test_collect_rejects_model_mismatch_and_backend_mismatch(tmp_path):
    reqs = [_req(0)]
    llm.write_requests(tmp_path, "b", reqs)
    _respond(tmp_path, "b", [(reqs[0]["id"], json.dumps({"answer": "A", "items": [1, 2], "p": 0}))], model="other")
    with pytest.raises(ValueError, match="answered by model 'other'"):
        llm.collect(tmp_path, "b", "manual")
    with pytest.raises(FileNotFoundError):                           # API backend without its record
        llm.collect(tmp_path, "b", "anthropic")
    llm.backend_sidecar_path(tmp_path, "b").write_text(json.dumps({"backend": "anthropic", "effort": "high"}))
    with pytest.raises(ValueError, match="answered by an API backend"):
        llm.collect(tmp_path, "b", "manual")
    with pytest.raises(ValueError, match="names backend 'anthropic'"):
        llm.collect(tmp_path, "b", "openai_compat")


def test_collect_check_callback_runs_before_logging(tmp_path):
    reqs = [_req(0)]
    llm.write_requests(tmp_path, "b", reqs)
    _respond(tmp_path, "b", [(reqs[0]["id"], json.dumps({"answer": "A", "items": [1, 2], "p": 0}))])
    with pytest.raises(ValueError, match="custom rule"):
        llm.collect(tmp_path, "b", "manual", check=lambda req, x: ["custom rule"])
    assert not (tmp_path / "calls.jsonl").exists()


def test_anthropic_request_construction():
    """Bare id sent (an anthropic/ prefix is stripped); no temperature or thinking field; one cache breakpoint:
    the judge's user text split before "Record A:" with the breakpoint on the exemplars, the system text for
    every other role; the blocks join to the request's exact text. Opus is refused in either form."""
    req = _jreq(0)
    kw = llm.anthropic_kwargs(req, "high")
    assert kw["model"] == "claude-sonnet-5-5" and kw["max_tokens"] == 100 and kw["system"] == "sys"
    head, tail = kw["messages"][0]["content"]
    assert head == {"type": "text", "text": "Exemplars of the job:\n- exemplar 0", "cache_control": {"type": "ephemeral"}}
    assert "cache_control" not in tail and tail["text"].startswith("\n\nRecord A:\n")
    assert head["text"] + tail["text"] == req["user"]
    assert "temperature" not in kw and "thinking" not in kw
    assert kw["output_config"]["effort"] == "high"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": llm.api_schema(SCHEMA)}
    assert llm.anthropic_kwargs(_jreq(0, model="anthropic/claude-sonnet-5-5"), "low")["model"] == "claude-sonnet-5-5"
    clus = llm.make_request("clusterer:c1:0", "clusterer", "claude-haiku-5-5", 1.0, 100, "sys", "contexts", SCHEMA)
    kw = llm.anthropic_kwargs(clus, "low")
    assert kw["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "contexts"}]
    twin = llm.anthropic_kwargs(clus, "low", twin=True)                   # an exact twin reads the whole prompt
    assert twin["messages"][0]["content"] == [{"type": "text", "text": "contexts", "cache_control": {"type": "ephemeral"}}]
    assert json.dumps(twin).count("cache_control") == 2
    jt = llm.anthropic_kwargs(req, "low", twin=True)["messages"][0]["content"]
    assert [b.get("cache_control") for b in jt] == [{"type": "ephemeral"}] * 2 and "".join(b["text"] for b in jt) == req["user"]
    for opus in ("claude-opus-5-5", "anthropic/claude-opus-5-5"):
        with pytest.raises(ValueError, match="Opus is not called"):
            llm.anthropic_kwargs({**req, "model": opus}, "low")
    with pytest.raises(ValueError, match="'Record A:' markers, need 1"):
        llm.anthropic_kwargs(_req(0, model="claude-sonnet-5-5"), "low")
    with pytest.raises(ValueError, match="cannot set temperature"):
        llm.anthropic_kwargs(_jreq(0) | {"temperature": 0.0}, "high")
    with pytest.raises(ValueError, match="effort"):
        llm.anthropic_kwargs(req, "extreme")


def test_openai_compat_request_construction():
    req = _req(0, model="qwen/qwen3-235b")
    provider = llm.provider_object(["fireworks"], ["bf16"])
    assert provider == {"order": ["fireworks"], "allow_fallbacks": False, "quantizations": ["bf16"]}
    p = llm.openai_payload(req, provider, "json_schema")
    assert p["messages"][0] == {"role": "system", "content": "sys"} and p["temperature"] == 1.0
    assert p["response_format"]["json_schema"]["schema"] == llm.api_schema(SCHEMA)
    assert p["provider"]["allow_fallbacks"] is False
    assert "provider" not in llm.openai_payload(req, None, "json_schema")
    assert llm.openai_payload(req, None, "json_object")["response_format"] == {"type": "json_object"}
    assert "response_format" not in llm.openai_payload(req, None, "none")
    with pytest.raises(ValueError, match="response_format"):
        llm.openai_payload(req, None, "xml")
    with pytest.raises(ValueError, match="at least one provider"):
        llm.provider_object([], None)


def settings_for(backend, model, **extra):
    return {"backend": backend, "models": {r: model for r in llm.ROLES}, "temperature": 1.0, **extra}


def test_settings_file_is_checked_and_deepinfra_names_every_role(tmp_path):
    s = llm.load_settings(Path(llm.__file__).parent / "settings" / "deepinfra.json")
    assert s["backend"] == "openai_compat" and s["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert s["key_env"] == "NEURONS_API_KEY_1" and set(s["models"].values()) == {"deepseek-ai/DeepSeek-V4.1-Flash-1017"}
    assert not any("sk-" in str(v) for v in s.values())                # a name, never a key
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"backend": "openai_compat", "models": {}, "temperature": 1.0}))
    with pytest.raises(ValueError, match="missing"):
        llm.load_settings(bad)
    bad.write_text(json.dumps(settings_for("manual", "m")))
    with pytest.raises(ValueError, match="not an API backend"):
        llm.load_settings(bad)


def test_run_batch_through_a_stub_client_then_collect(tmp_path, monkeypatch):
    """The anthropic path end to end with a stub in place of the SDK client: responses and usage
    appended per call, backend and reported model recorded, collect accepts it."""
    reqs = [_jreq(i) for i in range(2)]
    llm.write_requests(tmp_path, "b", reqs)
    sent = []

    def create(**kw):
        sent.append(kw)
        return _msg(kw["model"], OK_TEXT)
    monkeypatch.setattr(llm, "anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    settings = settings_for("anthropic", "claude-sonnet-5-5", effort="medium")
    llm.run_batch(tmp_path, "b", settings)
    assert len(sent) == 2 and all("temperature" not in kw for kw in sent)
    llm.run_batch(tmp_path, "b", settings)                            # resume: nothing left to send
    assert len(sent) == 2
    llm.run_batch(tmp_path, "b", {**settings, "effort": "high"})       # a resumed batch on other settings is recorded
    side = json.loads(llm.backend_sidecar_path(tmp_path, "b").read_text())
    assert side["settings"] == settings and [r["settings"]["effort"] for r in side["later_settings"]] == ["high"]
    with pytest.raises(ValueError, match="refusing to mix backends"):
        llm.run_batch(tmp_path, "b", settings_for("openai_compat", "claude-sonnet-5-5", base_url="https://x",
                                                  key_env="K", response_format="json_object"))
    llm.run_batch(tmp_path, "b", settings_for("anthropic", "claude-haiku-5-5", effort="medium"))
    assert len(sent) == 2                                             # every line answered: nothing checked or sent
    assert json.loads(llm.backend_sidecar_path(tmp_path, "b").read_text())["reported_models"] == ["claude-sonnet-5-5"]
    usage = llm.read_jsonl(llm.usage_path(tmp_path, "b"))
    assert [(u["prompt_tokens"], u["cache_creation_input_tokens"], u["cache_read_input_tokens"], u["request_id"])
            for u in usage] == [(7, 5, 11, "req_01")] * 2
    assert len(llm.collect(tmp_path, "b", "anthropic")) == 2
    assert {c["backend"] for c in llm.read_jsonl(tmp_path / "calls.jsonl")} == {"anthropic"}


def test_a_reply_from_another_model_raises_at_the_call_and_at_collect(tmp_path):
    """Fail loud on the model: the call raises when the reply names another model than the one sent, and
    collect raises when a response line's model differs from its request's."""
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _msg("claude-sonnet-5", OK_TEXT)))
    with pytest.raises(RuntimeError, match="answered by model 'claude-sonnet-5', sent 'claude-sonnet-5-5'"):
        llm.call_anthropic(client, _jreq(0), "low")
    llm.write_requests(tmp_path, "b", [_jreq(0)])
    _respond(tmp_path, "b", [(_jreq(0)["id"], OK_TEXT)], model="claude-sonnet-5")
    llm.backend_sidecar_path(tmp_path, "b").write_text(json.dumps({"backend": "anthropic"}))
    with pytest.raises(ValueError, match="answered by model 'claude-sonnet-5', request names 'claude-sonnet-5-5'"):
        llm.collect(tmp_path, "b", "anthropic")


def test_openai_compat_run_never_writes_or_prints_the_key(tmp_path, monkeypatch, capsys):
    """DeepInfra's path with urlopen stubbed: the key travels only in the Authorization header; it is
    in no file written and nothing printed. An HTTP error carries the server's body."""
    secret = "TESTKEY-not-a-real-key-0123456789"
    monkeypatch.setenv("NEURONS_API_KEY_TEST", secret)
    model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    llm.write_requests(tmp_path, "b", [_req(0, model=model)])
    seen = []

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return io.BytesIO(json.dumps(self.body).encode())

        def __exit__(self, *a):
            return False

    def urlopen(http, timeout):
        seen.append((http.full_url, http.get_header("Authorization"), json.loads(http.data)))
        return Reply({"model": model, "usage": {"prompt_tokens": 11, "completion_tokens": 5},
                      "choices": [{"finish_reason": "stop",
                                   "message": {"content": json.dumps({"answer": "A", "items": [1, 2], "p": 0.5})}}]})
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    settings = settings_for("openai_compat", model, base_url="https://api.deepinfra.com/v1/openai",
                            key_env="NEURONS_API_KEY_TEST", response_format="json_object")
    llm.run_batch(tmp_path, "b", settings)
    url, auth, body = seen[0]
    assert url == "https://api.deepinfra.com/v1/openai/chat/completions" and auth == f"Bearer {secret}"
    assert body["response_format"] == {"type": "json_object"} and secret not in json.dumps(body)
    assert len(llm.collect(tmp_path, "b", "openai_compat")) == 1
    usage = llm.read_jsonl(llm.usage_path(tmp_path, "b"))[0]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["model"]) == (11, 5, model)
    assert all(secret not in f.read_text() for f in tmp_path.rglob("*") if f.is_file())
    out = capsys.readouterr()
    assert secret not in out.out and secret not in out.err

    def refuse(http, timeout):
        raise llm.urllib.error.HTTPError(http.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "unknown field"}'))
    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    llm.write_requests(tmp_path, "c", [_req(1, model=model)])
    with pytest.raises(RuntimeError, match="HTTP 400 .*unknown field") as e:
        llm.run_batch(tmp_path, "c", settings)
    assert secret not in str(e.value)


MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return io.BytesIO(json.dumps(self.body).encode())

    def __exit__(self, *a):
        return False


def stub_endpoint(fail: dict, delays: dict, log: list):
    """urlopen stand-in: the request named in `fail` gets its listed HTTP codes first; `delays` (s)
    make later requests finish before earlier ones."""
    lock = threading.Lock()

    def urlopen(http, timeout):
        body = json.loads(http.data)
        rid = body["messages"][1]["content"]
        with lock:
            codes = fail.get(rid, [])
            code = codes.pop(0) if codes else None
            log.append((rid, code))
        if code is not None:
            raise llm.urllib.error.HTTPError(http.full_url, code, "x", {}, io.BytesIO(b'{"error": "busy"}'))
        time.sleep(delays.get(rid, 0.0))
        return Reply({"model": MODEL, "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                      "choices": [{"finish_reason": "stop",
                                   "message": {"content": json.dumps({"answer": "A", "items": [1, 2], "p": 0.5})}}]})
    return urlopen


def deepinfra_like(tmp_path):
    return settings_for("openai_compat", MODEL, base_url="https://api.deepinfra.com/v1/openai",
                        key_env="NEURONS_API_KEY_TEST", response_format="json_schema")


def batch_of(tmp, name, n):
    reqs = [llm.make_request(f"judge:{name}:{i}", "judge", MODEL, 1.0, 100, "sys", f"user {i}", SCHEMA) for i in range(n)]
    llm.write_requests(tmp, name, reqs)
    return reqs


def test_concurrent_run_handles_out_of_order_answers_and_a_429(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    sleeps = []
    monkeypatch.setattr(llm, "backoff_sleep", sleeps.append)
    settings = deepinfra_like(tmp_path)
    results = {}
    for concurrency, where in ((8, tmp_path / "par"), (1, tmp_path / "seq")):
        reqs = batch_of(where, "b", 10)
        log = []
        delays = {f"user {i}": (10 - i) * 0.01 for i in range(10)}     # later requests answer first
        monkeypatch.setattr(llm.urllib.request, "urlopen", stub_endpoint({"user 3": [429]}, delays, log))
        llm.run_batch(where, "b", settings, concurrency=concurrency)
        order = [r["id"] for r in llm.read_jsonl(llm.responses_path(where, "b"))]
        attempts = {u["id"]: u["attempts"] for u in llm.read_jsonl(llm.usage_path(where, "b"))}
        assert attempts == {r["id"]: 2 if r["id"] == "judge:b:3" else 1 for r in reqs}
        results[concurrency] = (llm.collect(where, "b", "openai_compat"), order,
                                json.loads(llm.backend_sidecar_path(where, "b").read_text())["reported_models"])
    assert sleeps == [2.0, 2.0]                                       # one backoff per run
    assert results[8][0] == results[1][0] and results[8][2] == results[1][2] == [MODEL]
    assert results[8][1] != [f"judge:b:{i}" for i in range(10)]       # answers arrived out of order
    assert results[1][1] == [f"judge:b:{i}" for i in range(10)]       # sequential: retried in place, in order


def test_retry_gives_up_after_six_attempts_naming_the_id_and_keeps_finished_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    sleeps = []
    monkeypatch.setattr(llm, "backoff_sleep", sleeps.append)
    batch_of(tmp_path, "b", 4)
    log = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", stub_endpoint({"user 2": [503] * 6}, {}, log))
    with pytest.raises(RuntimeError, match=r"judge:b:2: gave up after 6 attempts: .*HTTP 503"):
        llm.run_batch(tmp_path, "b", deepinfra_like(tmp_path), concurrency=1)
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
    kept = {r["id"] for r in llm.read_jsonl(llm.responses_path(tmp_path, "b"))}
    assert "judge:b:2" not in kept and {"judge:b:0", "judge:b:1"} <= kept
    batch_of(tmp_path, "c", 1)
    monkeypatch.setattr(llm.urllib.request, "urlopen", stub_endpoint({"user 0": [400]}, {}, log))
    with pytest.raises(RuntimeError, match=r"HTTP 400"):              # not retryable: one attempt, no sleep
        llm.run_batch(tmp_path, "c", deepinfra_like(tmp_path), concurrency=4)
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_run_many_answers_every_batch_in_one_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", stub_endpoint({}, {}, []))
    settings = tmp_path / "s.json"
    settings.write_text(json.dumps(deepinfra_like(tmp_path)))
    for name in ("a", "b"):
        batch_of(tmp_path, name, 3)
    llm.main(["run-many", "--explain", str(tmp_path), "--batches", "a,b", "--settings", str(settings), "--concurrency", "2"])
    assert all(len(llm.collect(tmp_path, name, "openai_compat")) == 3 for name in ("a", "b"))
    assert capsys.readouterr().out.count("responses") == 2


def test_call_anthropic_raises_on_refusal():
    msg = _msg("claude-sonnet-5-5", None, stop="refusal", category="cyber")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))
    with pytest.raises(RuntimeError, match=r"stop_reason 'refusal' \(cyber\)"):
        llm.call_anthropic(client, _jreq(0), "high")


class FakeBatches:
    """messages.batches of the SDK: create keeps the requests, retrieve says in_progress then ended, results
    come back in reverse order with reply(request) deciding each result."""

    def __init__(self, reply):
        self.reply, self.created, self.polls = reply, [], 0

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def retrieve(self, mb_id):
        self.polls += 1
        return SimpleNamespace(processing_status="in_progress" if self.polls % 2 else "ended",
                               request_counts={}, ended_at="2026-11-01T00:00:00Z")

    def results(self, mb_id):
        return [SimpleNamespace(custom_id=r["custom_id"], result=self.reply(r))
                for r in reversed(self.created[int(mb_id.split("_")[1]) - 1])]


def succeeded(r, text=OK_TEXT, **kw):
    return SimpleNamespace(type="succeeded", message=_msg(r["params"]["model"], text, **kw))


def test_message_batch_round_trip_records_every_failure_and_resumes(tmp_path, monkeypatch):
    """run-batch -> collect-batch with a fake SDK: custom_ids map back to (batch, id) through the mapping file,
    twins sit next to each other, results arrive out of order, errored / expired / refused results are
    recorded as failures (the refused reply's usage is logged), collect-batch raises after writing, a second
    run-batch asks only the failed ids, and collect then accepts both batches unchanged."""
    settings = llm.load_settings(SETTINGS_DIR / "anthropic_sonnet55.json")
    a = [_jreq(i) for i in range(4)]
    b = [llm.make_request(r["id"].replace("r0s0", "r0s1"), "judge", r["model"], 1.0, 100, r["system"], r["user"], SCHEMA)
         for r in a]                                                           # identical twins in a second file
    llm.write_requests(tmp_path, "a", a)
    llm.write_requests(tmp_path, "b", b)
    fate = {a[0]["id"]: "errored", a[1]["id"]: "expired", b[2]["id"]: "refusal"}

    def reply(r):
        rid = mapping[r["custom_id"]]
        if fate.get(rid) == "errored":
            return SimpleNamespace(type="errored", error=SimpleNamespace(error=SimpleNamespace(type="api_error", message="boom")))
        if fate.get(rid) == "expired":
            return SimpleNamespace(type="expired")
        if fate.get(rid) == "refusal":
            return succeeded(r, None, stop="refusal", category="general_harms")
        return succeeded(r)
    fake = FakeBatches(reply)
    monkeypatch.setattr(llm, "anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(llm, "poll_sleep", lambda s: None)
    assert llm.run_message_batch(tmp_path, ["a", "b"], settings, dry_run=True) == [] and fake.created == []
    with pytest.raises(ValueError, match="run-batch is the anthropic backend's"):
        llm.run_message_batch(tmp_path, ["a"], settings_for("openai_compat", "claude-sonnet-5-5"))
    mb, = llm.run_message_batch(tmp_path, ["a", "b"], settings)
    record = json.loads((llm.submissions_dir(tmp_path) / f"{mb}.json").read_text())
    mapping = {e["custom_id"]: e["id"] for e in record["requests"]}
    sent = fake.created[0]
    assert all(llm.CUSTOM_ID.match(r["custom_id"]) for r in sent) and len(sent) == 8
    order = [mapping[r["custom_id"]] for r in sent]
    assert all(order[k].replace("s1", "s0") == order[k + 1].replace("s1", "s0") for k in range(0, 8, 2))   # twins adjacent
    assert sent[0]["params"]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(json.dumps(r["params"]).count("cache_control") == 2 for r in sent)      # every line has its twin
    with pytest.raises(ValueError, match="not collected yet"):
        llm.run_message_batch(tmp_path, ["a", "b"], settings)
    with pytest.raises(RuntimeError, match="3 request\\(s\\) failed"):
        llm.main(["collect-batch", "--explain", str(tmp_path)])
    fails = {f["id"]: f for name in ("a", "b") for f in llm.read_jsonl(llm.failures_path(tmp_path, name))}
    assert {k: (f["type"], f["reason"]) for k, f in fails.items()} == {
        a[0]["id"]: ("errored", "api_error: boom"), a[1]["id"]: ("expired", "expired"),
        b[2]["id"]: ("succeeded", "stop_reason 'refusal' (general_harms)")}
    usage = {u["id"]: u for name in ("a", "b") for u in llm.read_jsonl(llm.usage_path(tmp_path, name))}
    assert len(usage) == 6 and usage[b[2]["id"]]["finish_reason"] == "refusal"           # billed, so logged
    assert {(u["message_batch_id"], u["cache_read_input_tokens"], u["latency_s"]) for u in usage.values()} == {(mb, 11, None)}
    with pytest.raises(ValueError, match="missing answer"):
        llm.collect_many(tmp_path, ["a", "b"], "anthropic")
    fate.clear()
    mb2, = llm.run_message_batch(tmp_path, ["a", "b"], settings)
    mapping = {e["custom_id"]: e["id"] for e in json.loads((llm.submissions_dir(tmp_path) / f"{mb2}.json").read_text())["requests"]}
    assert sorted(mapping[r["custom_id"]] for r in fake.created[1]) == sorted([a[0]["id"], a[1]["id"], b[2]["id"]])
    llm.main(["collect-batch", "--explain", str(tmp_path)])
    assert json.loads((llm.submissions_dir(tmp_path) / f"{mb2}.json").read_text())["counts"] == {
        "a": {"answers": 2, "failures": 0, "declined": 0}, "b": {"answers": 1, "failures": 0, "declined": 0}}
    out = llm.collect_many(tmp_path, ["a", "b"], "anthropic")
    assert {k: len(v) for k, v in out.items()} == {"a": 4, "b": 4}
    assert json.loads(llm.backend_sidecar_path(tmp_path, "a").read_text())["reported_models"] == ["claude-sonnet-5-5"]
    assert llm.run_message_batch(tmp_path, ["a", "b"], settings) == []                   # nothing pending


def test_a_batch_failure_repeated_to_the_cap_is_declined_without_a_hand_step(tmp_path, monkeypatch, capsys):
    """A refusal and a max_tokens stop that repeat to --max-attempts end in responses/<batch>.declined.jsonl in
    reject's row shape and are not asked again; an errored result is not an attempt and stays pending; collect
    then accepts the batch with the declines."""
    settings = llm.load_settings(SETTINGS_DIR / "anthropic_sonnet55.json")
    reqs = [_jreq(i) for i in range(4)]
    llm.write_requests(tmp_path, "a", reqs)
    fate = {reqs[0]["id"]: "refusal", reqs[1]["id"]: "max_tokens", reqs[2]["id"]: "errored"}
    mapping = {}

    def reply(r):
        rid = mapping[r["custom_id"]]
        if fate.get(rid) == "errored":
            return SimpleNamespace(type="errored", error=SimpleNamespace(error=SimpleNamespace(type="overloaded_error", message="busy")))
        if fate.get(rid) == "refusal":
            return succeeded(r, None, stop="refusal", category="bio")
        if fate.get(rid) == "max_tokens":
            return succeeded(r, None, stop="max_tokens")
        return succeeded(r)
    fake = FakeBatches(reply)
    monkeypatch.setattr(llm, "anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(llm, "poll_sleep", lambda s: None)
    for round_ in (1, 2):
        mb, = llm.run_message_batch(tmp_path, ["a"], settings)
        mapping.update({e["custom_id"]: e["id"] for e in json.loads((llm.submissions_dir(tmp_path) / f"{mb}.json").read_text())["requests"]})
        with pytest.raises(RuntimeError, match=f"{3 if round_ == 1 else 1} request\\(s\\) failed"):
            llm.main(["collect-batch", "--explain", str(tmp_path), "--max-attempts", "2"])
    gave_up = llm.declined(tmp_path, "a")
    assert sorted(gave_up) == sorted([reqs[0]["id"], reqs[1]["id"]])
    assert all(set(row) == {"id", "role", "reason", "finish_reason", "attempts", "utc"} for row in gave_up.values())
    assert gave_up[reqs[0]["id"]]["reason"] == \
        "stop_reason 'refusal' (bio) on 2 of 2 Message Batches attempts (responses/a.batch_failures.jsonl)"
    assert llm.decline_text(gave_up[reqs[0]["id"]]) == "judge declined (provider refusal, 2 attempts)"
    assert (gave_up[reqs[1]["id"]]["finish_reason"], gave_up[reqs[1]["id"]]["attempts"]) == ("max_tokens", 2)
    assert "declined: " + reqs[0]["id"] in capsys.readouterr().out
    fate.clear()
    mb, = llm.run_message_batch(tmp_path, ["a"], settings)                    # only the errored id is asked again
    assert [e["id"] for e in json.loads((llm.submissions_dir(tmp_path) / f"{mb}.json").read_text())["requests"]] == [reqs[2]["id"]]
    mapping.update({e["custom_id"]: e["id"] for e in json.loads((llm.submissions_dir(tmp_path) / f"{mb}.json").read_text())["requests"]})
    llm.main(["collect-batch", "--explain", str(tmp_path), "--max-attempts", "2"])
    assert sorted(llm.collect(tmp_path, "a", "anthropic")) == sorted([reqs[2]["id"], reqs[3]["id"]])


def test_an_oversized_submission_is_split_into_batches_and_collect_batch_gathers_them(tmp_path, monkeypatch, capsys):
    """Over a batch limit, run-batch submits several Message Batches (one mapping file each), never splitting a
    group of identical prompts; collect-batch collects every one and collect accepts the whole set."""
    settings = llm.load_settings(SETTINGS_DIR / "anthropic_sonnet55.json")
    a = [_jreq(i) for i in range(4)]
    b = [llm.make_request(r["id"].replace("r0s0", "r0s1"), "judge", r["model"], 1.0, 100, r["system"], r["user"], SCHEMA)
         for r in a]
    llm.write_requests(tmp_path, "a", a)
    llm.write_requests(tmp_path, "b", b)
    fake = FakeBatches(succeeded)
    monkeypatch.setattr(llm, "anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(llm, "poll_sleep", lambda s: None)
    one = len(json.dumps({"custom_id": "q000000", "params": llm.anthropic_kwargs(a[0], "low", twin=True)}).encode())
    monkeypatch.setattr(llm, "BATCH_MAX_BYTES", 2 * one + 60)                  # room for one twin pair per batch
    assert llm.run_message_batch(tmp_path, ["a", "b"], settings, dry_run=True) == []
    assert "in 4 Message Batch(es) of [2, 2, 2, 2] requests" in capsys.readouterr().out
    monkeypatch.setattr(llm, "BATCH_MAX_BYTES", 10 ** 9)
    monkeypatch.setattr(llm, "BATCH_MAX_REQUESTS", 5)                          # two twin pairs per batch
    ids = llm.run_message_batch(tmp_path, ["a", "b"], settings)
    assert ids == ["msgbatch_1", "msgbatch_2"] and [len(c) for c in fake.created] == [4, 4]
    records = [json.loads((llm.submissions_dir(tmp_path) / f"{i}.json").read_text()) for i in ids]
    for rec in records:                                                        # each twin pair inside one batch
        assert set(Counter(e["id"].replace("r0s1", "r0s0") for e in rec["requests"]).values()) == {2}
    assert sorted(e["id"] for rec in records for e in rec["requests"]) == sorted(r["id"] for r in a + b)
    llm.main(["collect-batch", "--explain", str(tmp_path)])
    assert llm.open_submissions(tmp_path) == []
    assert {k: len(v) for k, v in llm.collect_many(tmp_path, ["a", "b"], "anthropic").items()} == {"a": 4, "b": 4}
    monkeypatch.setattr(llm, "BATCH_MAX_REQUESTS", 1)                          # a twin pair alone is over the limit
    llm.write_requests(tmp_path, "c", [_jreq(9)])
    llm.write_requests(tmp_path, "d", [llm.make_request("judge:L00N9+:c0:r0s1", "judge", "claude-sonnet-5-5", 1.0, 100,
                                                        "sys", _jreq(9)["user"], SCHEMA)])
    with pytest.raises(ValueError, match="share one prompt and alone exceed a batch limit"):
        llm.run_message_batch(tmp_path, ["c", "d"], settings, dry_run=True)


@pytest.mark.parametrize("name,model,effort", [("anthropic_sonnet55.json", "claude-sonnet-5-5", "low"),
                                               ("anthropic_haiku55.json", "claude-haiku-5-5", "low"),
                                               ("anthropic_sonnet55_medium.json", "claude-sonnet-5-5", "medium"),
                                               ("anthropic_sonnet55_high.json", "claude-sonnet-5-5", "high")])
def test_anthropic_settings_name_bare_ids_and_their_effort(name, model, effort):
    s = llm.load_settings(SETTINGS_DIR / name)
    assert s["backend"] == "anthropic" and s["effort"] == effort and s["temperature"] == 1.0
    assert s["models"] == {r: model for r in llm.ROLES}


SETTINGS_DIR = Path(llm.__file__).parent / "settings"


@pytest.mark.parametrize("name,refusal", [("deepinfra_opus_explainer.json", "Opus is not called through the API"),
                                          ("deepinfra_opus_explainer_writer.json", "retired")])
def test_settings_naming_opus_are_refused(name, refusal):
    with pytest.raises(ValueError, match=refusal) as e:
        llm.load_settings(SETTINGS_DIR / name)
    assert name in str(e.value)


@pytest.mark.parametrize("name,model,roles", [
    ("deepinfra_sonnet_explainer.json", "anthropic/claude-sonnet-5-5", ("explainer", "second_reader")),
    ("deepinfra_flash_second_reader.json", MODEL, llm.ROLES),
    ("deepinfra.json", MODEL, llm.ROLES)])
def test_non_opus_settings_load_and_a_run_refuses_a_batch_prepared_with_another_model(tmp_path, name, model, roles):
    s = llm.load_settings(SETTINGS_DIR / name)
    assert {r: s["models"][r] for r in llm.ROLES} == {r: model if r in roles else MODEL for r in llm.ROLES}
    assert (s["base_url"], s["key_env"], s["temperature"], s["response_format"]) == \
        ("https://api.deepinfra.com/v1/openai", "NEURONS_API_KEY_1", 1.0, "json_schema")
    other = "qwen/other-model"
    llm.write_requests(tmp_path, "d2-explainer", [llm.make_request("explainer:L00N0+:0", "explainer", other, 1.0, 100,
                                                                   "sys", "user", SCHEMA)])
    with pytest.raises(ValueError, match=r"model or temperature differs from the settings for \['explainer:L00N0\+:0'\]"):
        llm.run_batch(tmp_path, "d2-explainer", s)                    # refused before any call or key read


def test_opus_is_never_called_through_an_api(tmp_path, monkeypatch):
    """SPEC_D1 §2.2 model policy (2026-10-20): settings or requests naming Opus are refused before any
    request is sent, on both API backends; a request file never names a provider-routed Opus id; a
    non-Opus Anthropic id (Sonnet) passes."""
    sent = []
    monkeypatch.setenv("NEURONS_API_KEY_TEST", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda http, timeout: sent.append(http) or Reply(
        {"model": json.loads(http.data)["model"], "usage": {"prompt_tokens": 3, "completion_tokens": 2},
         "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answer": "A", "items": [1, 2], "p": 0.5})}}]}))
    with pytest.raises(ValueError, match=r"explainer:L00N0\+:0: model 'anthropic/claude-opus-5-5': Opus is not called"):
        llm.make_request("explainer:L00N0+:0", "explainer", "anthropic/claude-opus-5-5", 1.0, 100, "s", "u", SCHEMA)
    opus = settings_for("openai_compat", "anthropic/Claude-Opus-5-5", base_url="https://api.example/v1",
                        key_env="NEURONS_API_KEY_TEST", response_format="json_schema")
    llm.write_requests(tmp_path, "a", [_req(0, model=MODEL)])
    with pytest.raises(ValueError, match=r"settings \(explainer\): model 'anthropic/Claude-Opus-5-5': Opus is not called"):
        llm.run_batch(tmp_path, "a", opus)
    bare = settings_for("anthropic", "claude-opus-5-5", effort="high")    # the Anthropic API's own id
    llm.write_requests(tmp_path, "b", [_req(0, model="claude-opus-5-5")])  # allowed in a file, for the manual backend
    with pytest.raises(ValueError, match=r"model 'claude-opus-5-5': Opus is not called"):
        llm.run_batch(tmp_path, "b", bare)
    assert sent == [] and not llm.backend_sidecar_path(tmp_path, "b").exists()
    sonnet = settings_for("openai_compat", "anthropic/claude-sonnet-5-5", base_url="https://api.example/v1",
                          key_env="NEURONS_API_KEY_TEST", response_format="json_schema")
    llm.write_requests(tmp_path, "c", [_req(0, model="anthropic/claude-sonnet-5-5")])
    llm.run_batch(tmp_path, "c", sonnet)
    assert len(sent) == 1 and len(llm.collect(tmp_path, "c", "openai_compat")) == 1


def test_a_provider_decline_is_an_empty_answer_and_other_finish_reasons_raise(tmp_path, monkeypatch):
    req = _req(0, model=MODEL)
    for reason in ("content_filter", "refusal"):
        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda http, timeout, r=reason: Reply(
            {"model": MODEL, "usage": {"prompt_tokens": 3, "completion_tokens": 0},
             "choices": [{"finish_reason": r, "message": {"content": None}}]}))
        response, usage = llm.call_openai_compat(deepinfra_like(tmp_path), "k", req)
        assert response["text"] == "" and usage["finish_reason"] == reason
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda http, timeout: Reply(
        {"model": MODEL, "usage": {"prompt_tokens": 3, "completion_tokens": 100},
         "choices": [{"finish_reason": "length", "message": {"content": "{"}}]}))
    with pytest.raises(RuntimeError, match="finish_reason 'length'"):
        llm.call_openai_compat(deepinfra_like(tmp_path), "k", req)
    assert llm.decline_text({"role": "writer", "finish_reason": "content_filter", "reason": "x", "attempts": 3}) == \
        "writer declined (provider content filter, 3 attempts)"


def test_mixed_backend_logs_each_rows_backend_and_refuses_manual_rows_elsewhere(tmp_path):
    reqs = [_req(0), _req(1)]
    llm.write_requests(tmp_path, "b", reqs)
    ok = json.dumps({"answer": "A", "items": [1, 2], "p": 0})
    _respond(tmp_path, "b", [(reqs[0]["id"], ok)])                                   # answered by the API
    llm.backend_sidecar_path(tmp_path, "b").write_text(json.dumps({"backend": "openai_compat"}))
    llm.append_jsonl(llm.responses_path(tmp_path, "b"),
                     [{"id": reqs[1]["id"], "model": "m-1", "text": ok, "utc": "2026-10-21T00:00:00Z", "backend": "manual"}])
    with pytest.raises(ValueError, match="row says backend 'manual'"):            # an API collect refuses the manual row
        llm.collect(tmp_path, "b", "openai_compat")
    out = llm.collect(tmp_path, "b", "mixed")
    assert set(out) == {reqs[0]["id"], reqs[1]["id"]}
    logged = {r["id"]: r["backend"] for r in llm.read_jsonl(tmp_path / "calls.jsonl")}
    assert logged == {reqs[0]["id"]: "openai_compat", reqs[1]["id"]: "manual"}
    # a mixed batch without a sidecar cannot attribute a bare row
    llm.write_requests(tmp_path, "c", [reqs[0]])
    _respond(tmp_path, "c", [(reqs[0]["id"], ok)])
    with pytest.raises(ValueError, match="no backend sidecar"):
        llm.collect(tmp_path, "c", "mixed")
    with pytest.raises(ValueError, match="unknown backend"):
        llm._validate(tmp_path, "c", "nonsense", None)
