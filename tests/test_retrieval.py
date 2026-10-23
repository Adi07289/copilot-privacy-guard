from privacyguard.models import Document
from privacyguard.retrieval import TfidfIndex

def _doc(i, text, dept="Sales"):
    return Document(id=f"d{i}", kind="email", department=dept, sensitivity=1, purposes=["general"], title="t", text=text)

def test_search_ranks_relevant_first_and_copies_tags():
    idx = TfidfIndex()
    idx.index([_doc(1, "payroll salary revision april"), _doc(2, "kubernetes deployment pipeline"), _doc(3, "salary slip and ctc")])
    res = idx.search("salary", k=2)
    assert [c.doc_id for c in res] == ["d3", "d1"] or [c.doc_id for c in res] == ["d1", "d3"]
    assert res[0].department == "Sales" and res[0].score > 0

def test_search_before_index_is_empty():
    assert TfidfIndex().search("x") == []

import numpy as np
from privacyguard.controls.pii import PIIRedactor
from privacyguard.retrieval import DenseIndex, build_indexes, redact_documents

def test_indexed_texts_exposed():
    idx = TfidfIndex(); idx.index([_doc(1, "payroll salary")])
    assert idx.indexed_texts == ["t\npayroll salary"]

def test_redact_documents_strips_gold_values_and_keeps_tags(mini_corpus):
    red = redact_documents(mini_corpus, PIIRedactor())
    for d, r in zip(mini_corpus, red):
        assert (r.id, r.department, r.sensitivity, r.purposes, r.is_poisoned, r.canary) == \
               (d.id, d.department, d.sensitivity, d.purposes, d.is_poisoned, d.canary)
        assert r.pii_spans == [] and not any(s.value in r.text for s in d.pii_spans)

def test_build_indexes_raw_and_redacted(mini_corpus):
    idx = build_indexes(mini_corpus, PIIRedactor())
    assert set(idx) == {"tfidf:raw", "tfidf:redacted"}
    assert "Kiran Rao" in "\n".join(idx["tfidf:raw"].indexed_texts)
    assert "Kiran Rao" not in "\n".join(idx["tfidf:redacted"].indexed_texts)
    assert {c.doc_id for c in idx["tfidf:redacted"].search("payroll deployment pipeline")} == {d.id for d in mini_corpus}

def _fake_embed(texts, kind):
    vocab = ["payroll", "deployment", "pipeline", "sales", "salary", "ticket", "cost"]
    return np.array([[t.lower().count(w) + 0.01 for w in vocab] for t in texts], dtype=np.float32)

def test_dense_index_with_injected_embedder(mini_corpus):
    idx = DenseIndex(embedder=_fake_embed); idx.index(mini_corpus)
    res = idx.search("sales pricing", k=2)
    assert res[0].doc_id == "sales-1" and len(res) == 2 and len(idx.indexed_texts) == 5

def test_build_indexes_dense_kind(mini_corpus):
    idx = build_indexes(mini_corpus, PIIRedactor(), kinds=("tfidf", "dense"), dense_embedder=_fake_embed)
    assert set(idx) == {"tfidf:raw", "tfidf:redacted", "dense:raw", "dense:redacted"}

def test_body_detected_values_are_also_redacted_in_the_title():
    from privacyguard.models import Document
    d = Document(id="hr-x", kind="hr", department="HR", sensitivity=3, purposes=["hr_operations"],
                 title="HR record Neel Sathe", text="EMPLOYEE RECORD\nName: Neel Sathe\nEmployee ID: EMP1")
    r = redact_documents([d], PIIRedactor())[0]
    assert "Neel Sathe" not in r.title and "Neel Sathe" not in r.text
