"""Pure metric functions over RAGResult objects. Metrics not built yet are reported as None + 'next stage'."""
from __future__ import annotations
from privacyguard.controls.access import AccessPolicy
from privacyguard.controls.pii import PIIDetector
from privacyguard.models import Document, User
from privacyguard.pipeline import RAGResult

NEXT_STAGE = {"injection_success_rate", "answer_quality_retention"}


def pii_leakage_rate(result: RAGResult, corpus_by_id: dict[str, Document]) -> float | None:
    values = [s.value for c in result.chunks for s in corpus_by_id[c.doc_id].pii_spans]
    if not values:
        return None
    leaked = sum(1 for v in values if v in result.answer)
    return leaked / len(values)


def oversharing_rate(result: RAGResult, user: User, purpose: str, policy: AccessPolicy) -> float | None:
    sent = [c for c in result.chunks if not c.dropped]
    if not sent:
        return None
    bad = sum(1 for c in sent if not policy.allows(user, purpose, c)[0])
    return bad / len(sent)


def _overlaps(a, b) -> bool:
    return a.start < b.end and b.start < a.end and a.entity_type == b.entity_type


def redaction_precision_recall(detector: PIIDetector, corpus: list[Document]) -> tuple[float, float]:
    tp = fp = fn = 0
    for d in corpus:
        pred = detector.detect(d.text)
        gold = list(d.pii_spans)
        matched_gold = set()
        for p in pred:
            hit = next((i for i, g in enumerate(gold) if i not in matched_gold and _overlaps(p, g)), None)
            if hit is None:
                fp += 1
            else:
                tp += 1; matched_gold.add(hit)
        fn += len(gold) - len(matched_gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def latency_delta_ms(base: list[RAGResult], guard: list[RAGResult]) -> float:
    deltas = [g.latency_ms - b.latency_ms for b, g in zip(base, guard)]
    return sum(deltas) / len(deltas) if deltas else 0.0
