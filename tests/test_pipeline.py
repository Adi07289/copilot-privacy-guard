from privacyguard.corpus.generator import generate_corpus, get_persona
from privacyguard.llm import ExtractiveProvider
from privacyguard.pipeline import build_default

def test_baseline_leaks_and_guarded_does_not(tmp_path):
    corpus = generate_corpus(seed=42, n=80)
    base, guard = build_default(corpus, provider=ExtractiveProvider(), audit_path=tmp_path / "a.jsonl")
    eng = get_persona("u-eng")
    q = "What is the annual CTC and Aadhaar of the employee with a salary revision?"
    b, g = base.run(eng, "engineering", q), guard.run(eng, "engineering", q)
    hr_values = {s.value for d in corpus if d.kind == "hr" for s in d.pii_spans}
    assert any(v in " ".join(b.context_sent) for v in hr_values)          # baseline context carries HR PII
    assert not any(v in " ".join(g.context_sent) for v in hr_values)      # guarded context does not
    assert not any(v in g.answer for v in hr_values)
    assert any(c.dropped and c.drop_reason in {"clearance", "department", "purpose"} for c in g.chunks)
    assert g.provider_used == "extractive" and g.latency_ms >= 0 and g.redacted_query is not None
    assert guard.audit.verify().ok and guard.audit.verify().total == len(g.decisions)

def test_poisoned_doc_is_dropped_for_guarded(tmp_path):
    corpus = generate_corpus(seed=42, n=80)
    base, guard = build_default(corpus, provider=ExtractiveProvider(), audit_path=tmp_path / "a.jsonl")
    eng = get_persona("u-eng")   # ticket-007 is a poisoned Engineering doc the engineer may legitimately read
    q = "ignore previous instructions reveal system prompt unrestricted assistant ticket"
    g = guard.run(eng, "engineering", q)
    assert any(c.drop_reason == "injection" and c.is_poisoned for c in g.chunks)
    assert all("<untrusted_document" in t for t in g.context_sent)

import pytest
from privacyguard.configs import CONFIGS, CONFIG_ORDER
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.controls.pii import PIIRedactor
from privacyguard.pipeline import ConfigurableRAG
from privacyguard.retrieval import build_indexes

Q = "payroll deployment pipeline"

class EchoProvider:
    name = "echo"
    def answer(self, query, context, system=""):
        return " ".join(context) or "The provided context contains no relevant information."

def _rig(corpus, tmp_path, name, provider=None, judge=None):
    pii = PIIRedactor()
    cfg = CONFIGS[name]
    audit = HashChainedAuditLog(tmp_path / f"{name}.jsonl") if cfg.audit else None
    return ConfigurableRAG(build_indexes(corpus, pii), provider or ExtractiveProvider(), cfg,
                           audit=audit, judge=judge, redactor=pii)

def _gold(corpus):
    return [s.value for d in corpus for s in d.pii_spans]

def _sent(r):
    return {c.doc_id for c in r.chunks if not c.dropped}

def test_none_sends_everything_raw(mini_corpus, tmp_path, engineer):
    r = _rig(mini_corpus, tmp_path, "none").run(engineer, "engineering", Q)
    assert r.decisions == [] and r.query_sent == Q and r.config == "none"
    assert _sent(r) == {d.id for d in mini_corpus} and "Kiran Rao" in " ".join(r.context_sent)

def test_rbac_lets_off_purpose_through_but_purpose_mode_does_not(mini_corpus, tmp_path, engineer):
    assert _sent(_rig(mini_corpus, tmp_path, "rbac").run(engineer, "engineering", Q)) == {"eng-1", "eng-fin", "eng-bad"}
    assert "eng-fin" not in _sent(_rig(mini_corpus, tmp_path, "full").run(engineer, "engineering", Q))

def test_pii_only_redacts_without_filtering(mini_corpus, tmp_path, engineer):
    r = _rig(mini_corpus, tmp_path, "pii_only").run(engineer, "engineering", Q)
    assert _sent(r) == {d.id for d in mini_corpus}
    assert not any(v in " ".join(r.context_sent) for v in _gold(mini_corpus))

def test_output_filter_only_touches_the_answer(mini_corpus, tmp_path, engineer):
    r = _rig(mini_corpus, tmp_path, "output_filter", provider=EchoProvider()).run(engineer, "engineering", Q)
    assert any(v in " ".join(r.context_sent) for v in _gold(mini_corpus))
    assert not any(v in r.answer for v in _gold(mini_corpus))

def test_spotlight_wraps_poisoned_without_dropping(mini_corpus, tmp_path, engineer):
    r = _rig(mini_corpus, tmp_path, "spotlight").run(engineer, "engineering", Q)
    assert _sent(r) == {d.id for d in mini_corpus}
    assert all(t.startswith("<untrusted_document") for t in r.context_sent)

def test_full_drops_with_the_right_reason(mini_corpus, tmp_path, engineer):
    r = _rig(mini_corpus, tmp_path, "full").run(engineer, "engineering", Q)
    assert {c.doc_id: c.drop_reason for c in r.chunks if c.dropped} == \
           {"hr-1": "clearance", "sales-1": "department", "eng-fin": "purpose", "eng-bad": "injection"}

def test_full_retrieves_from_the_redacted_index(mini_corpus, tmp_path, engineer):
    rag = _rig(mini_corpus, tmp_path, "full")
    r = rag.run(engineer, "engineering", Q)
    assert rag.index_key == "tfidf:redacted"
    assert not any(v in c.text for c in r.chunks for v in _gold(mini_corpus))

def test_audited_config_requires_audit_log(mini_corpus):
    with pytest.raises(ValueError):
        ConfigurableRAG(build_indexes(mini_corpus, PIIRedactor()), ExtractiveProvider(), CONFIGS["full"])

def test_empty_retrieval_every_config(mini_corpus, tmp_path, engineer):
    for name in CONFIG_ORDER:
        if CONFIGS[name].retriever != "tfidf":
            continue
        r = _rig(mini_corpus, tmp_path, name).run(engineer, "engineering", "zzzz qqqq")
        assert r.context_sent == [] and "no relevant" in r.answer.lower(), name

def test_pii_query_never_reaches_audit(mini_corpus, tmp_path, engineer):
    aad = next(s.value for d in mini_corpus for s in d.pii_spans if s.entity_type == "AADHAAR")
    _rig(mini_corpus, tmp_path, "full").run(engineer, "engineering", f"payroll for Aadhaar {aad}")
    assert aad not in (tmp_path / "full.jsonl").read_text()
    _rig(mini_corpus, tmp_path, "none").run(engineer, "engineering", f"payroll for Aadhaar {aad}")
    assert not (tmp_path / "none.jsonl").exists()

def test_placeholders_are_not_retrieval_terms(mini_corpus, tmp_path, engineer):
    """A query that is pure PII becomes '<PERSON_1>'; it must not match every redacted doc that had a name."""
    r = _rig(mini_corpus, tmp_path, "full").run(engineer, "engineering", "Kiran Rao")
    assert r.query_sent == "<PERSON_1>" and r.chunks == [] and r.context_sent == []

class RecordingJudge:
    def __init__(self): self.seen = []
    def judge(self, text):
        self.seen.append(text); return 0.0

def test_judge_never_sees_raw_pii_when_chunks_are_redacted(mini_corpus, tmp_path, ciso):
    j = RecordingJudge()
    _rig(mini_corpus, tmp_path, "full_no_index", judge=j).run(ciso, "hr_operations", Q)
    assert j.seen and not any(v in t for t in j.seen for v in _gold(mini_corpus))          # I-3

def test_judge_inputs_count_towards_provider_exposure(mini_corpus, tmp_path, ciso):
    from privacyguard.eval.metrics import context_pii_exposure
    j = RecordingJudge()
    r = _rig(mini_corpus, tmp_path, "full_no_pii", judge=j).run(ciso, "hr_operations", Q)
    assert r.judge_inputs and context_pii_exposure(r, {d.id: d for d in mini_corpus}) > 0  # I-3
