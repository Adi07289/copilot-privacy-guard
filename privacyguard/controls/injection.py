"""Control 3: heuristic prompt-injection screener for retrieved chunks. LLM-as-judge is a stubbed next-stage hook."""
from __future__ import annotations
import base64, re
from typing import Protocol
from privacyguard.controls.base import Control
from privacyguard.models import Decision, GuardContext

PATTERNS: list[tuple[str, float]] = [
    (r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions", 1.0),
    (r"you\s+are\s+now\b", 0.6),
    (r"system\s+prompt", 0.5),
    (r"\bdisregard\b", 0.4),
    (r"\b(?:reveal|exfiltrate|leak)\b|\bsend\b.{0,60}\bto\b", 0.5),
    (r"do\s+not\s+tell\s+the\s+user", 0.8),
    (r"assistant\s+must\b", 0.5),
    (r"\bIMPORTANT:", 0.3),
]
_COMPILED = [(re.compile(p, 0 if p.startswith(r"\bIMPORTANT") else re.I), w) for p, w in PATTERNS]
_ZW = re.compile("[​‌‍⁠﻿]")
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_IMPERATIVE = re.compile(r"\b(?:ignore|reveal|send|disregard|reply|output|print|forward|exfiltrate)\b", re.I)
_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def score_text(text: str) -> tuple[float, list[str]]:
    score, hits = 0.0, []
    for rx, w in _COMPILED:
        if rx.search(text):
            score += w; hits.append(rx.pattern)
    if _ZW.search(text):
        score += 0.8; hits.append("zero_width")
    for m in _HTML_COMMENT.finditer(text):
        if _IMPERATIVE.search(m.group(1)):
            score += 0.7; hits.append("html_comment_imperative"); break
    for m in _B64.finditer(text):
        try:
            decoded = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4)).decode("ascii")
        except Exception:
            continue
        if any(rx.search(decoded) for rx, _ in _COMPILED):
            score += 0.9; hits.append("base64_instruction"); break
    return min(1.0, round(score, 3)), hits


class LLMJudge(Protocol):
    def judge(self, text: str) -> float | None: ...


class NullJudge:
    def judge(self, text: str) -> float | None:
        return None


class InjectionScreener(Control):
    name = "injection"

    def __init__(self, threshold: float = 0.5, judge: LLMJudge | None = None):
        self.threshold = threshold
        self.judge = judge or NullJudge()

    def apply(self, ctx: GuardContext) -> GuardContext:
        for ch in ctx.chunks:
            if ch.dropped:
                continue
            score, hits = score_text(ch.text)
            verdict = self.judge.judge(ch.text)
            judge_detail = "not_run (next stage)" if verdict is None else verdict
            if score >= self.threshold:
                ch.dropped, ch.drop_reason = True, "injection"
                ch.flags.append("injection")
                ctx.decisions.append(Decision(self.name, "drop", ch.doc_id,
                                              {"score": score, "hits": hits, "llm_judge": judge_detail}))
            else:
                ch.text = f'<untrusted_document id="{ch.doc_id}">\n{ch.text}\n</untrusted_document>'
                ctx.decisions.append(Decision(self.name, "allow", ch.doc_id,
                                              {"score": score, "hits": hits, "llm_judge": judge_detail}))
        return ctx
