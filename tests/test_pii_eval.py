import os
import pytest
from privacyguard.eval.heldout_pii import heldout_dataset
from privacyguard.eval.pii_eval import compare_detectors, evaluate_detector, to_markdown
from privacyguard.models import Document, PIISpan

def _doc(text, spans):
    return Document(id="d", kind="x", department="HR", sensitivity=1, purposes=["general"], title="", text=text,
                    pii_spans=[PIISpan(text.index(v), text.index(v) + len(v), t, v) for v, t in spans])

def test_evaluate_detector_exact_counts():
    d = _doc("Call +91 9876543210 or mail a@b.co", [("+91 9876543210", "IN_MOBILE"), ("a@b.co", "EMAIL")])
    def detect(text):   # finds the phone twice (overlapping, same type) and a bogus PERSON
        return [PIISpan(5, 19, "IN_MOBILE", ""), PIISpan(9, 19, "IN_MOBILE", ""), PIISpan(0, 4, "PERSON", "Call")]
    r = evaluate_detector(detect, [d])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)          # overlapping same-type preds merged before scoring
    assert r["per_type"]["EMAIL"]["recall"] == 0.0 and r["per_type"]["IN_MOBILE"]["recall"] == 1.0

def test_heldout_dataset_is_well_formed():
    ds = heldout_dataset()
    assert len(ds) == 32 and sum(1 for d in ds if not d.pii_spans) == 4
    for d in ds:
        for s in d.pii_spans:
            assert d.text[s.start:s.end] == s.value
    assert {s.entity_type for d in ds for s in d.pii_spans} == \
           {"AADHAAR", "PAN", "IN_MOBILE", "IFSC", "UPI_VPA", "CREDIT_CARD", "DOB", "SALARY_INR", "EMAIL", "PERSON"}

def test_compare_runs_ours_without_presidio(monkeypatch):
    import privacyguard.eval.pii_eval as pe
    monkeypatch.setattr(pe, "presidio_available", lambda: False)
    res = compare_detectors({"heldout": heldout_dataset()})
    assert res["detectors"] == ["ours"] and 0 < res["results"]["heldout"]["ours"]["recall"] <= 1
    assert "| IFSC |" in to_markdown(res)

@pytest.mark.skipif(not os.getenv("PG_SLOW_TESTS"), reason="loads Presidio + spaCy; run with PG_SLOW_TESTS=1")
def test_presidio_detector_maps_types():
    pytest.importorskip("presidio_analyzer")
    from privacyguard.eval.pii_eval import presidio_detector
    types = {s.entity_type for s in presidio_detector(False)("Email riya@example.com today")}
    assert "EMAIL" in types
