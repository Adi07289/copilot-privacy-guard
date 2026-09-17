import random, re
from privacyguard.corpus import indian_pii as ip

def test_verhoeff_known_vector():
    # 236 -> check digit 3 (standard Verhoeff example)
    assert ip.verhoeff_checksum("236") == 3
    assert ip.is_valid_aadhaar("2363") is False  # too short
    assert ip.verhoeff_validate("2363") is True

def test_generated_aadhaar_is_valid():
    rng = random.Random(1)
    for _ in range(50):
        a = ip.generate_aadhaar(rng)
        assert re.fullmatch(r"[2-9]\d{11}", a) and ip.is_valid_aadhaar(a)
        assert not ip.is_valid_aadhaar(a[:-1] + str((int(a[-1]) + 1) % 10))

def test_pan():
    rng = random.Random(2)
    p = ip.generate_pan(rng)
    assert re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", p) and ip.is_valid_pan(p)
    assert not ip.is_valid_pan("ABCDE123X")

def test_mobile_ifsc_upi_card():
    rng = random.Random(3)
    m = ip.generate_in_mobile(rng)
    assert ip.is_valid_in_mobile(m) and m.startswith("+91 ")
    assert ip.is_valid_in_mobile("9876543210") and not ip.is_valid_in_mobile("1234567890")
    i = ip.generate_ifsc(rng)
    assert ip.is_valid_ifsc(i) and i[4] == "0"
    u = ip.generate_upi(rng, "Riya Menon")
    assert ip.is_valid_upi(u) and "@" in u
    c = ip.generate_card(rng)
    assert ip.luhn_valid(c) and not ip.luhn_valid("4111111111111112")
