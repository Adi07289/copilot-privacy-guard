"""Baseline and Guarded RAG pipelines sharing the same retriever and LLM so only the control list differs."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from privacyguard.controls.access import PurposeBasedAccessFilter
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.controls.base import Control
from privacyguard.controls.injection import InjectionScreener
from privacyguard.controls.pii import PIIRedactor
from privacyguard.llm import LLMProvider, get_provider
from privacyguard.models import Chunk, Decision, Document, GuardContext, User
from privacyguard.retrieval import Retriever, TfidfIndex


@dataclass
class RAGResult:
    answer: str
    chunks: list[Chunk]
    context_sent: list[str]
    decisions: list[Decision]
    latency_ms: float
    provider_used: str
    redacted_query: str | None = None
    query_sent: str = ""


def _provider_used(p: LLMProvider) -> str:
    return getattr(p, "last_used", p.name)


class BaselineRAG:
    name = "baseline"

    def __init__(self, retriever: Retriever, provider: LLMProvider, k: int = 6):
        self.retriever, self.provider, self.k = retriever, provider, k

    def run(self, user: User, purpose: str, query: str) -> RAGResult:
        t0 = time.perf_counter()
        chunks = self.retriever.search(query, self.k)
        context = [c.text for c in chunks]
        answer = self.provider.answer(query, context)
        return RAGResult(answer, chunks, context, [], (time.perf_counter() - t0) * 1000,
                         _provider_used(self.provider), None, query)


class GuardedRAG:
    name = "guarded"

    def __init__(self, retriever: Retriever, provider: LLMProvider, audit: HashChainedAuditLog,
                 k: int = 6, controls: list[Control] | None = None):
        self.retriever, self.provider, self.audit, self.k = retriever, provider, audit, k
        self.pii = PIIRedactor()
        self.controls = controls if controls is not None else [PurposeBasedAccessFilter(), InjectionScreener()]

    def run(self, user: User, purpose: str, query: str) -> RAGResult:
        t0 = time.perf_counter()
        ctx = GuardContext(user=user, purpose=purpose, query=query)
        ctx = self.pii.apply(ctx)                                  # 1. redact the prompt
        ctx.chunks = self.retriever.search(ctx.redacted_query, self.k)
        for c in self.controls:                                    # 2. access, 3. injection
            ctx = c.apply(ctx)
        ctx = self.pii.apply(ctx)                                  # 1 again. redact surviving chunks
        context = [c.text for c in ctx.chunks if not c.dropped]
        answer = self.provider.answer(ctx.redacted_query, context)
        for d in ctx.decisions:                                    # 4. audit every decision
            self.audit.append({"user": user.id, "purpose": purpose, **d.to_dict()})
        return RAGResult(answer, ctx.chunks, context, ctx.decisions, (time.perf_counter() - t0) * 1000,
                         _provider_used(self.provider), ctx.redacted_query, ctx.redacted_query)


def build_default(corpus: list[Document], provider: LLMProvider | None = None,
                  audit_path: str | Path = "data/audit.jsonl") -> tuple[BaselineRAG, GuardedRAG]:
    retriever = TfidfIndex()
    retriever.index(corpus)
    provider = provider or get_provider()
    return BaselineRAG(retriever, provider), GuardedRAG(retriever, provider, HashChainedAuditLog(audit_path))
