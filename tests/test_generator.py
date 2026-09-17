import json, random
from privacyguard.corpus.generator import generate_corpus, save_corpus, load_corpus, PERSONAS, INJECTION_PAYLOADS
from privacyguard.corpus import indian_pii as ip

def test_corpus_is_deterministic_and_sized():
    a, b = generate_corpus(seed=42, n=80), generate_corpus(seed=42, n=80)
    assert len(a) == 80 and [d.id for d in a] == [d.id for d in b] and a[0].text == b[0].text

def test_kinds_and_poisoned_count():
    docs = generate_corpus(seed=42, n=80)
    kinds = {d.kind for d in docs}
    assert kinds == {"email", "hr", "ticket", "code"}
    assert sum(d.is_poisoned for d in docs) == 4
    assert all(any(p in d.text for p in INJECTION_PAYLOADS) for d in docs if d.is_poisoned)

def test_pii_spans_point_at_real_values():
    for d in generate_corpus(seed=7, n=40):
        for s in d.pii_spans:
            assert d.text[s.start:s.end] == s.value, (d.id, s)
            if s.entity_type == "AADHAAR":
                assert ip.is_valid_aadhaar(s.value)
            if s.entity_type == "PAN":
                assert ip.is_valid_pan(s.value)

def test_hr_docs_are_sensitive_and_tagged():
    docs = generate_corpus(seed=42, n=80)
    hr = [d for d in docs if d.kind == "hr"]
    assert hr and all(d.department == "HR" and d.sensitivity >= 2 and "hr_operations" in d.purposes for d in hr)

def test_save_load_roundtrip(tmp_path):
    docs = generate_corpus(seed=1, n=10)
    p = tmp_path / "c.json"
    save_corpus(docs, p)
    assert load_corpus(p) == docs

def test_personas():
    assert {u.role for u in PERSONAS} >= {"HR Manager", "Engineer", "Finance Analyst", "Intern", "CISO"}
