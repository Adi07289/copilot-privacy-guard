import random
from privacyguard.controls.output import OutputFilter
from privacyguard.corpus import indian_pii as ip

def test_output_filter_redacts_answer_and_logs_no_value():
    aad = ip.generate_aadhaar(random.Random(9))
    out, d = OutputFilter().filter(f"The Aadhaar is {aad}.")
    assert aad not in out and "<AADHAAR_1>" in out
    assert d.control == "output_filter" and d.action == "redact_answer" and aad not in str(d.detail)
