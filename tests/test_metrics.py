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

from privacyguard.eval.metrics import (context_pii_exposure, cosine_sim, index_pii_exposure, injection_exposure,
                                       injection_success, rouge_l_f1)

def _res(answer, chunks, context):
    return RAGResult(answer, chunks, context, [], 1.0, "groq")

def test_context_exposure_counts_what_the_llm_saw():
    d1 = _doc(1, "Aadhaar 234512345678 PAN ABCPE1234F", [PIISpan(8, 20, "AADHAAR", "234512345678"), PIISpan(25, 35, "PAN", "ABCPE1234F")])
    r = _res("x", [Chunk("d1", "", 1.0, "HR", 3, ["hr_operations"])], ["Aadhaar <AADHAAR_1> PAN ABCPE1234F"])
    assert context_pii_exposure(r, {"d1": d1}) == 0.5
    assert context_pii_exposure(_res("x", [], []), {"d1": d1}) is None

def test_injection_exposure_and_success():
    bad = Document(id="b", kind="email", department="E", sensitivity=1, purposes=["general"], title="t",
                   text="x", is_poisoned=True, canary="CANARY-XYZ")
    c_kept = Chunk("b", "x", 1.0, "E", 1, ["general"], is_poisoned=True)
    c_drop = Chunk("b", "x", 1.0, "E", 1, ["general"], is_poisoned=True, dropped=True, drop_reason="injection")
    assert injection_exposure(_res("a", [c_kept, c_drop], ["x"])) == (1, 2)
    assert injection_success(_res("done. CANARY-XYZ", [c_kept], ["x"]), {"b": bad}) is True
    assert injection_success(_res("done.", [c_drop], []), {"b": bad}) is False
    assert injection_success(_res("done.", [], []), {"b": bad}) is None

def test_rouge_l_hand_computed():
    assert round(rouge_l_f1("the cat sat on the mat", "the cat on the mat"), 4) == round(10 / 11, 4)
    assert rouge_l_f1("", "") == 1.0 and rouge_l_f1("a", "") == 0.0 and rouge_l_f1("x y", "p q") == 0.0

def test_cosine_sim():
    assert cosine_sim("a b", "a b") == 1.0 and cosine_sim("a", "b") == 0.0 and cosine_sim("", "") == 1.0
    assert round(cosine_sim("a a b", "a b b"), 4) == 0.8

def test_index_pii_exposure():
    d = _doc(1, "phone +91 9876543210", [PIISpan(6, 20, "IN_MOBILE", "+91 9876543210")])
    assert index_pii_exposure(["t\nphone +91 9876543210"], [d]) == 1.0
    assert index_pii_exposure(["t\nphone <IN_MOBILE_1>"], [d]) == 0.0
    assert index_pii_exposure(["x"], [_doc(2, "none", [])]) is None
