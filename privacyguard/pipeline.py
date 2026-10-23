"""One configurable RAG pipeline. Every comparative configuration shares the retriever and the LLM, so only the
GuardConfig differs between rows of the comparison."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from pathlib import Path
from privacyguard.configs import CONFIGS, GuardConfig
from privacyguard.controls.access import AccessPolicy, PurposeBasedAccessFilter
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.controls.injection import GroqJudge, InjectionScreener, LLMJudge
from privacyguard.controls.output import OutputFilter
from privacyguard.controls.pii import PIIRedactor
from privacyguard.llm import ROOT, LLMProvider, get_groq_client, get_provider
from privacyguard.models import Chunk, Decision, Document, GuardContext, User
from privacyguard.retrieval import Retriever, build_indexes

DEFAULT_AUDIT = ROOT / "data" / "audit.jsonl"
_PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")


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
    config: str = "none"


def _provider_used(p: LLMProvider) -> str:
    return getattr(p, "last_used", p.name)


class ConfigurableRAG:
    def __init__(self, indexes: dict[str, Retriever], provider: LLMProvider, config: GuardConfig,
                 audit: HashChainedAuditLog | None = None, judge: LLMJudge | None = None, k: int = 6,
                 redactor: PIIRedactor | None = None):
        if config.audit and audit is None:
            raise ValueError(f"config {config.name!r} audits decisions but no audit log was given")
        self.indexes, self.provider, self.config, self.audit, self.k = indexes, provider, config, audit, k
        self.pii = redactor or PIIRedactor()
        self.access = PurposeBasedAccessFilter(AccessPolicy(config.access)) if config.access != "none" else None
        self.injection = InjectionScreener(mode=config.injection, judge=judge) if config.injection != "none" else None
        self.output = OutputFilter(self.pii) if config.output_filter else None

    @property
    def index_key(self) -> str:
        return f"{self.config.retriever}:{'redacted' if self.config.redact_index else 'raw'}"

    @property
    def retriever(self) -> Retriever:
        return self.indexes[self.index_key]

    def run(self, user: User, purpose: str, query: str) -> RAGResult:
        c = self.config
        t0 = time.perf_counter()
        ctx = GuardContext(user=user, purpose=purpose, query=query)
        if c.redact_prompt:
            ctx = self.pii.redact_query(ctx)                      # 1. prompt never reaches the index raw
        q = ctx.redacted_query if ctx.redacted_query is not None else query
        retrieval_q = _PLACEHOLDER.sub(" ", q).strip()          # placeholders carry no meaning for retrieval
        ctx.chunks = self.retriever.search(retrieval_q, self.k) if retrieval_q else []
        if self.access:
            ctx = self.access.apply(ctx)                          # 2. clearance, department (+ purpose)
        if self.injection:
            ctx = self.injection.apply(ctx)                       # 3. spotlight / heuristic / + judge
        if c.redact_chunks:
            ctx = self.pii.redact_chunks(ctx)                     # 1 again, on surviving chunks
        context = [ch.text for ch in ctx.chunks if not ch.dropped]
        answer = self.provider.answer(q, context)
        if self.output:
            answer, d = self.output.filter(answer)
            ctx.decisions.append(d)
        if c.audit:
            for d in ctx.decisions:                               # 4. hash-chained audit
                self.audit.append({"user": user.id, "purpose": purpose, "config": c.name, **d.to_dict()})
        return RAGResult(answer, ctx.chunks, context, ctx.decisions, (time.perf_counter() - t0) * 1000,
                         _provider_used(self.provider), ctx.redacted_query, q, c.name)


class BaselineRAG(ConfigurableRAG):
    def __init__(self, indexes, provider, k: int = 6, redactor: PIIRedactor | None = None):
        super().__init__(indexes, provider, CONFIGS["none"], k=k, redactor=redactor)


class GuardedRAG(ConfigurableRAG):
    def __init__(self, indexes, provider, audit, judge=None, k: int = 6, redactor: PIIRedactor | None = None,
                 config: GuardConfig | None = None):
        super().__init__(indexes, provider, config or CONFIGS["full"], audit=audit, judge=judge, k=k, redactor=redactor)


def default_judge() -> LLMJudge | None:
    client = get_groq_client(judge=True)
    return GroqJudge(client) if client else None


def build_default(corpus: list[Document], provider: LLMProvider | None = None,
                  audit_path: str | Path = DEFAULT_AUDIT, judge="auto") -> tuple[BaselineRAG, GuardedRAG]:
    pii = PIIRedactor()
    indexes = build_indexes(corpus, pii)
    provider = provider or get_provider()
    if judge == "auto":
        judge = default_judge()
    return (BaselineRAG(indexes, provider, redactor=pii),
            GuardedRAG(indexes, provider, HashChainedAuditLog(audit_path), judge=judge, redactor=pii))
