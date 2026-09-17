from privacyguard.controls.access import AccessPolicy
from privacyguard.controls.pii import PIIDetector
from privacyguard.eval.metrics import pii_leakage_rate, oversharing_rate, redaction_precision_recall, latency_delta_ms
from privacyguard.eval.queries import QUERY_SET
from privacyguard.models import Chunk, Document, PIISpan
from privacyguard.pipeline import RAGResult

def _doc(i, text, spans, dept="HR", sens=3):
    return Document(id=f"d{i}", kind="hr", department=dept, sensitivity=sens, purposes=["hr_operations"],
                    title="t", text=text, pii_spans=spans)

def test_query_set_shape():
    assert len(QUERY_SET) == 20
    assert sum(q.targets_poisoned for q in QUERY_SET) == 4 and sum(q.cross_dept_probe for q in QUERY_SET) == 5
    assert len({q.id for q in QUERY_SET}) == 20

def test_pii_leakage_counts_only_retrieved_docs():
    d1 = _doc(1, "Aadhaar 234512345678 PAN ABCPE1234F", [PIISpan(8, 20, "AADHAAR", "234512345678"), PIISpan(25, 35, "PAN", "ABCPE1234F")])
    d2 = _doc(2, "phone +91 9876543210", [PIISpan(6, 20, "IN_MOBILE", "+91 9876543210")])
    by_id = {d.id: d for d in (d1, d2)}
    res = RAGResult("The PAN is ABCPE1234F.", [Chunk("d1", d1.text, 1.0, "HR", 3, ["hr_operations"])], [d1.text], [], 1.0, "extractive")
    assert pii_leakage_rate(res, by_id) == 0.5             # 1 of 2 retrieved values leaked; d2 not retrieved
    assert pii_leakage_rate(RAGResult("x", [], [], [], 1.0, "e"), by_id) is None

def test_oversharing_rate(engineer):
    res = RAGResult("a", [Chunk("d1", "t", 1.0, "HR", 3, ["hr_operations"]),
                          Chunk("d2", "t", 1.0, "Engineering", 1, ["engineering"]),
                          Chunk("d3", "t", 1.0, "HR", 3, ["hr_operations"], dropped=True)], ["t", "t"], [], 1.0, "e")
    assert oversharing_rate(res, engineer, "engineering", AccessPolicy()) == 0.5   # dropped chunks are not "sent"

def test_redaction_pr_on_perfect_and_partial():
    d = _doc(1, "Aadhaar 234512345678 and email a@b.co", [PIISpan(8, 20, "AADHAAR", "234512345678"), PIISpan(31, 37, "EMAIL", "a@b.co")])
    p, r = redaction_precision_recall(PIIDetector(), [d])
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0

def test_latency_delta():
    b = [RAGResult("", [], [], [], 10.0, "e"), RAGResult("", [], [], [], 20.0, "e")]
    g = [RAGResult("", [], [], [], 15.0, "e"), RAGResult("", [], [], [], 30.0, "e")]
    assert latency_delta_ms(b, g) == 7.5
