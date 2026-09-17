"""Generators + validators for India-specific PII. Validators are reused by the recognizers."""
from __future__ import annotations
import random, re

_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
      [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
      [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_INV = [0,4,3,2,1,5,6,7,8,9]


def verhoeff_checksum(digits: str) -> int:
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][int(d)]]
    return _INV[c]


def verhoeff_validate(number: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(d)]]
    return c == 0


def is_valid_aadhaar(s: str) -> bool:
    s = s.replace(" ", "")
    return bool(re.fullmatch(r"[2-9]\d{11}", s)) and verhoeff_validate(s)


def generate_aadhaar(rng: random.Random) -> str:
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    return body + str(verhoeff_checksum(body))


PAN_RE = re.compile(r"[A-Z]{5}\d{4}[A-Z]")

def is_valid_pan(s: str) -> bool:
    return bool(PAN_RE.fullmatch(s)) and s[3] in "PCHFATBLJG"

def generate_pan(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(3)) + "P" + rng.choice(letters) \
        + "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice(letters)


MOBILE_RE = re.compile(r"(?:\+91[\s-]?|0)?[6-9]\d{9}")

def is_valid_in_mobile(s: str) -> bool:
    return bool(MOBILE_RE.fullmatch(s.strip()))

def generate_in_mobile(rng: random.Random) -> str:
    return "+91 " + str(rng.randint(6, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))


IFSC_RE = re.compile(r"[A-Z]{4}0[A-Z0-9]{6}")
_BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB"]

def is_valid_ifsc(s: str) -> bool:
    return bool(IFSC_RE.fullmatch(s))

def generate_ifsc(rng: random.Random) -> str:
    return rng.choice(_BANKS) + "0" + "".join(str(rng.randint(0, 9)) for _ in range(6))


UPI_RE = re.compile(r"[a-zA-Z0-9._-]{3,}@(?:okaxis|oksbi|okhdfcbank|okicici|ybl|paytm|upi|ibl|axl)\b")
_HANDLES = ["okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "paytm", "upi", "ibl", "axl"]

def is_valid_upi(s: str) -> bool:
    return bool(UPI_RE.fullmatch(s))

def generate_upi(rng: random.Random, name: str) -> str:
    base = re.sub(r"[^a-z]", "", name.lower())[:8] or "user"
    return f"{base}{rng.randint(10, 99)}@{rng.choice(_HANDLES)}"


def luhn_valid(s: str) -> bool:
    digits = [int(c) for c in s.replace(" ", "") if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

def generate_card(rng: random.Random) -> str:
    body = "4" + "".join(str(rng.randint(0, 9)) for _ in range(14))
    for check in range(10):
        if luhn_valid(body + str(check)):
            return body + str(check)
    raise RuntimeError("unreachable")
