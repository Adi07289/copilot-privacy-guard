import os
from privacyguard.llm import ExtractiveProvider, FallbackProvider, get_provider, SYSTEM_PROMPT

def test_extractive_picks_overlapping_sentences():
    p = ExtractiveProvider()
    ctx = ["The offsite is in Goa. Budget is 2 lakh.", "Kubernetes upgrade is scheduled Friday."]
    out = p.answer("when is the kubernetes upgrade", ctx)
    assert "Kubernetes upgrade" in out and "Goa" not in out
    assert p.name == "extractive"

def test_extractive_empty_context():
    assert "no relevant" in ExtractiveProvider().answer("q", []).lower()

class _Boom:
    name = "boom"
    def answer(self, q, c, system=SYSTEM_PROMPT):
        raise RuntimeError("network down")

def test_fallback_switches_and_records():
    f = FallbackProvider(_Boom(), ExtractiveProvider())
    out = f.answer("kubernetes", ["Kubernetes is fine."])
    assert "Kubernetes" in out and f.last_used == "extractive"


def test_extractive_strips_untrusted_wrapper():
    out = ExtractiveProvider().answer("kubernetes upgrade", ['<untrusted_document id="d1">\nKubernetes upgrade is Friday.\n</untrusted_document>'])
    assert "untrusted_document" not in out and "Kubernetes upgrade" in out

import json
import pytest
from privacyguard.llm import (ExtractiveProvider, FallbackProvider, GroqClient, GroqProvider, LLMCache,
                              LLMUnavailable, get_provider, SYSTEM_PROMPT)

class _Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]

class FakeAPI:
    """Stands in for openai.OpenAI: .chat.completions.create(**kw)."""
    def __init__(self, texts=(), errors=()):
        self.texts, self.errors, self.calls = list(texts), list(errors), []
        self.chat = type("Chat", (), {})()
        self.chat.completions = self
    def create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return _Resp(self.texts.pop(0))

class HTTPErr(Exception):
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()

def test_get_provider_without_key_is_extractive():
    assert get_provider().name == "extractive"          # fixture: no key, empty cache path

def test_get_provider_uses_cache_offline(tmp_path):
    cache = LLMCache(tmp_path / "c.json"); cache.put("k", {"text": "x", "model": "m", "latency_ms": 1.0})
    p = get_provider(cache)
    assert isinstance(p, FallbackProvider) and p.name == "groq"

def test_cache_hit_avoids_second_call(tmp_path):
    api = FakeAPI(["hello"])
    c = GroqClient(None, "m", LLMCache(tmp_path / "c.json"), client=api)
    t1, m1 = c.chat("s", "u"); t2, m2 = c.chat("s", "u")
    assert (t1, t2) == ("hello", "hello") and len(api.calls) == 1
    assert m1["cached"] is False and m2["cached"] is True and c.calls == 1
    assert json.loads((tmp_path / "c.json").read_text())       # persisted

def test_retry_on_429_honours_retry_after(tmp_path):
    sleeps = []
    api = FakeAPI(["ok"], errors=[HTTPErr(429, "3")])
    c = GroqClient(None, "m", LLMCache(tmp_path / "c.json"), client=api, sleep=sleeps.append)
    assert c.chat("s", "u")[0] == "ok" and sleeps == [3.0]

def test_non_retryable_error_raises(tmp_path):
    c = GroqClient(None, "m", LLMCache(tmp_path / "c.json"), client=FakeAPI(errors=[HTTPErr(400)]))
    with pytest.raises(HTTPErr):
        c.chat("s", "u")

def test_persistent_429_falls_back(tmp_path):
    api = FakeAPI(errors=[HTTPErr(429)] * 10)
    client = GroqClient(None, "m", LLMCache(tmp_path / "c.json"), client=api, max_retries=2, sleep=lambda s: None)
    p = FallbackProvider(GroqProvider(client), ExtractiveProvider())
    out = p.answer("kubernetes upgrade", ["Kubernetes upgrade is Friday."])
    assert "Kubernetes" in out and p.last_used == "extractive" and len(api.calls) == 3

def test_no_key_no_cache_raises_unavailable(tmp_path):
    with pytest.raises(LLMUnavailable):
        GroqClient(None, "m", LLMCache(tmp_path / "c.json")).chat("s", "u")

def test_corrupt_cache_is_empty(tmp_path):
    p = tmp_path / "c.json"; p.write_text('{"half": {"text": "x"')     # killed mid-write
    cache = LLMCache(p)
    assert len(cache) == 0 and cache.get("half") is None

def test_judge_and_provider_share_one_cache(tmp_path, monkeypatch):
    from privacyguard.llm import get_groq_client, shared_cache
    shared_cache().put("k", {"text": "x", "model": "m", "latency_ms": 1.0})
    assert get_groq_client(judge=True).cache is get_provider()._p.client.cache is shared_cache()

def test_json_mode_sets_response_format(tmp_path):
    api = FakeAPI(['{"a": 1}'])
    GroqClient(None, "m", LLMCache(tmp_path / "c.json"), client=api).chat("s", "u", json_mode=True)
    assert api.calls[0]["response_format"] == {"type": "json_object"} and api.calls[0]["temperature"] == 0
