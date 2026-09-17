"""Control 1: Presidio-compatible hybrid PII detector (India-first regex registry + optional spaCy NER) and redactor."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable
from privacyguard.controls.base import Control
from privacyguard.corpus import indian_pii as ip
from privacyguard.models import GuardContext, Decision, PIISpan


@dataclass
class Recognizer:
    entity_type: str
    pattern: re.Pattern
    validator: Callable[[str], bool] | None = None
    score: float = 0.8

    def find(self, text: str) -> list[PIISpan]:
        out = []
        for m in self.pattern.finditer(text):
            v = m.group(0)
            if self.validator is None or self.validator(v):
                out.append(PIISpan(m.start(), m.end(), self.entity_type, v))
        return out


class RecognizerRegistry:
    def __init__(self, recognizers: list[Recognizer]):
        self.recognizers = recognizers

    @classmethod
    def default(cls) -> "RecognizerRegistry":
        return cls([
            Recognizer("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)"), ip.is_valid_aadhaar, 0.95),
            Recognizer("PAN", re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"), ip.is_valid_pan, 0.9),
            Recognizer("IN_MOBILE", re.compile(r"(?<![\d@])(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)"), ip.is_valid_in_mobile, 0.7),
            Recognizer("IFSC", re.compile(r"(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])"), ip.is_valid_ifsc, 0.85),
            Recognizer("UPI_VPA", ip.UPI_RE, None, 0.85),
            Recognizer("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), None, 0.9),
            Recognizer("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)"), ip.luhn_valid, 0.85),
            Recognizer("DOB", re.compile(r"(?<!\d)(?:0[1-9]|[12]\d|3[01])[/-](?:0[1-9]|1[0-2])[/-](?:19|20)\d{2}(?!\d)"), None, 0.6),
            Recognizer("SALARY_INR", re.compile(r"(?:₹|Rs\.?|INR)\s?\d[\d,]{3,}(?:\.\d+)?"), None, 0.6),
        ])


_HONORIFIC = re.compile(r"\b(?:Mr|Ms|Mrs|Dr|Shri|Smt)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
_LABELLED = re.compile(r"(?:Name|From|Reported by|Employee|Contact person)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})")


class PIIDetector:
    def __init__(self, registry: RecognizerRegistry | None = None):
        self.registry = registry or RecognizerRegistry.default()
        self._nlp = None
        self.names_mode = "heuristic"
        try:
            import spacy  # optional
            self._nlp = spacy.load("en_core_web_sm")
            self.names_mode = "spacy"
        except Exception:
            self._nlp = None

    def _persons(self, text: str) -> list[PIISpan]:
        spans = []
        if self._nlp is not None:
            for ent in self._nlp(text).ents:
                if ent.label_ == "PERSON":
                    spans.append(PIISpan(ent.start_char, ent.end_char, "PERSON", ent.text))
        for rx in (_HONORIFIC, _LABELLED):
            for m in rx.finditer(text):
                spans.append(PIISpan(m.start(1), m.end(1), "PERSON", m.group(1)))
        return spans

    def detect(self, text: str) -> list[PIISpan]:
        cand = [s for r in self.registry.recognizers for s in r.find(text)] + self._persons(text)
        # resolve overlaps: longest span wins, then earliest
        cand.sort(key=lambda s: (s.start, -(s.end - s.start)))
        out: list[PIISpan] = []
        for s in cand:
            if out and s.start < out[-1].end:
                if (s.end - s.start) > (out[-1].end - out[-1].start):
                    out[-1] = s
                continue
            out.append(s)
        return out


class PIIRedactor(Control):
    name = "pii"

    def __init__(self, detector: PIIDetector | None = None):
        self.detector = detector or PIIDetector()

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        spans = self.detector.detect(text)
        mapping: dict[str, str] = {}
        by_value: dict[tuple[str, str], str] = {}
        counters: Counter = Counter()
        out, cursor = [], 0
        for s in spans:
            key = (s.entity_type, s.value)
            if key not in by_value:
                counters[s.entity_type] += 1
                by_value[key] = f"<{s.entity_type}_{counters[s.entity_type]}>"
                mapping[by_value[key]] = s.value
            out.append(text[cursor:s.start]); out.append(by_value[key]); cursor = s.end
        out.append(text[cursor:])
        return "".join(out), mapping

    @staticmethod
    def _counts(mapping: dict[str, str]) -> dict[str, int]:
        c: Counter = Counter(k.strip("<>").rsplit("_", 1)[0] for k in mapping)
        return dict(c)

    def apply(self, ctx: GuardContext) -> GuardContext:
        if ctx.redacted_query is None:
            ctx.redacted_query, mapping = self.redact(ctx.query)
            ctx.pii_map.update(mapping)
            ctx.decisions.append(Decision(self.name, "redact_query", None, {"entities": self._counts(mapping)}))
            return ctx
        for ch in ctx.chunks:
            if ch.dropped:
                continue
            ch.text, mapping = self.redact(ch.text)
            ctx.pii_map.update(mapping)
            n = len(mapping)
            if n:
                ch.flags.append(f"redacted:{n}")
            ctx.decisions.append(Decision(self.name, "redact_chunk", ch.doc_id, {"entities": self._counts(mapping)}))
        return ctx
