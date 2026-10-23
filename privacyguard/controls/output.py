"""Output-side filtering: redact PII in the generated answer only (analogue of output guardrails)."""
from __future__ import annotations
from privacyguard.controls.pii import PIIRedactor
from privacyguard.models import Decision


class OutputFilter:
    name = "output_filter"

    def __init__(self, redactor: PIIRedactor | None = None):
        self.redactor = redactor or PIIRedactor()

    def filter(self, answer: str) -> tuple[str, Decision]:
        out, mapping = self.redactor.redact(answer)
        return out, Decision(self.name, "redact_answer", None, {"entities": PIIRedactor._counts(mapping)})
