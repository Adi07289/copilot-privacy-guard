import random
from privacyguard.controls.pii import PIIDetector, PIIRedactor
from privacyguard.corpus import indian_pii as ip
from privacyguard.models import GuardContext, Chunk

rng = random.Random(0)
AAD = ip.generate_aadhaar(rng); PAN = ip.generate_pan(rng); MOB = ip.generate_in_mobile(rng)
IFSC = ip.generate_ifsc(rng); UPI = ip.generate_upi(rng, "Riya Menon"); CARD = ip.generate_card(rng)

def types(text):
    return {s.entity_type for s in PIIDetector().detect(text)}

def test_each_recognizer_positive():
    assert types(f"Aadhaar {AAD}") == {"AADHAAR"}
    assert types(f"PAN {PAN}") == {"PAN"}
    assert types(f"call {MOB}") == {"IN_MOBILE"}
    assert types(f"IFSC {IFSC}") == {"IFSC"}
    assert types(f"pay {UPI}") == {"UPI_VPA"}
    assert types("mail riya@example.com") == {"EMAIL"}
    assert types(f"card {CARD}") == {"CREDIT_CARD"}
    assert types("DOB 14/03/1991") == {"DOB"}
    assert types("CTC ₹12,50,000 per annum") == {"SALARY_INR"}

def test_negatives():
    bad_aadhaar = AAD[:-1] + str((int(AAD[-1]) + 1) % 10)
    assert "AADHAAR" not in types(f"ref {bad_aadhaar}")
    assert types("ticket 1234567890 opened") == set()   # starts with 1 → not a mobile
    assert types("RETRY_LIMIT = 3") == set()

def test_person_detection_any_mode():
    d = PIIDetector()
    assert d.names_mode in {"spacy", "heuristic"}
    assert "PERSON" in {s.entity_type for s in d.detect("Reported by: Mr. Nikhil Bose today")}

def test_overlap_resolution_prefers_longest():
    # an Aadhaar also matches a bare 10-digit mobile substring; only AADHAAR must survive
    spans = PIIDetector().detect(f"id {AAD}")
    assert [s.entity_type for s in spans] == ["AADHAAR"]

def test_redact_replaces_with_typed_placeholders():
    r = PIIRedactor()
    out, mapping = r.redact(f"Aadhaar {AAD} and phone {MOB} and phone {MOB}")
    assert "<AADHAAR_1>" in out and "<IN_MOBILE_1>" in out and AAD not in out and MOB not in out
    assert mapping["<AADHAAR_1>"] == AAD and out.count("<IN_MOBILE_1>") == 2

def test_apply_redacts_query_then_chunks(engineer):
    r = PIIRedactor()
    ctx = GuardContext(user=engineer, purpose="engineering", query=f"what about {PAN}")
    ctx = r.apply(ctx)
    assert ctx.redacted_query == "what about <PAN_1>" and ctx.decisions[-1].control == "pii"
    ctx.chunks = [Chunk("d1", f"email riya@example.com", 0.9, "Engineering", 1, ["general"]),
                  Chunk("d2", "clean text", 0.5, "Engineering", 1, ["general"], dropped=True)]
    ctx = r.apply(ctx)
    assert "<EMAIL_1>" in ctx.chunks[0].text and "redacted:1" in ctx.chunks[0].flags
    assert ctx.chunks[1].text == "clean text"                       # dropped chunks untouched
    assert all("riya@example.com" not in str(d.detail) for d in ctx.decisions)  # never log values
