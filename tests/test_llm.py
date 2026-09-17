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

def test_get_provider_without_key_is_extractive(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    p = get_provider()
    assert p.name == "extractive"

def test_extractive_strips_untrusted_wrapper():
    out = ExtractiveProvider().answer("kubernetes upgrade", ['<untrusted_document id="d1">\nKubernetes upgrade is Friday.\n</untrusted_document>'])
    assert "untrusted_document" not in out and "Kubernetes upgrade" in out
