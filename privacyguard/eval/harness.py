"""Runs the same query set through Baseline and Guarded pipelines and aggregates the metrics."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from privacyguard.controls.access import AccessPolicy
from privacyguard.corpus.generator import get_persona
from privacyguard.eval.metrics import pii_leakage_rate, oversharing_rate, redaction_precision_recall, latency_delta_ms
from privacyguard.eval.queries import EvalQuery
from privacyguard.llm import LLMProvider
from privacyguard.models import Document
from privacyguard.pipeline import build_default

METRIC_ROWS = [  # key, label, direction
    ("pii_leakage_rate", "PII leakage rate", "lower"),
    ("oversharing_rate", "Over-sharing retrieval rate", "lower"),
    ("injection_success_rate", "Prompt-injection success rate", "lower"),
    ("redaction_precision", "Redaction precision", "higher"),
    ("redaction_recall", "Redaction recall", "higher"),
    ("answer_quality_retention", "Answer-quality retention", "higher"),
    ("added_latency_ms", "Added latency (ms)", "lower"),
]


@dataclass
class EvalReport:
    metrics: dict[str, dict]
    per_query: list[dict]
    n_queries: int
    provider: str
    names_mode: str


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_eval(corpus: list[Document], queries: list[EvalQuery], provider: LLMProvider,
             audit_path: str | Path = "data/audit.jsonl") -> EvalReport:
    base, guard = build_default(corpus, provider=provider, audit_path=audit_path)
    by_id = {d.id: d for d in corpus}
    policy = AccessPolicy()
    b_res, g_res, rows = [], [], []
    for q in queries:
        user = get_persona(q.user_id)
        b, g = base.run(user, q.purpose, q.text), guard.run(user, q.purpose, q.text)
        b_res.append(b); g_res.append(g)
        rows.append({
            "id": q.id, "user": q.user_id, "purpose": q.purpose, "cross_dept_probe": q.cross_dept_probe,
            "targets_poisoned": q.targets_poisoned,
            "baseline": {"pii_leakage": pii_leakage_rate(b, by_id), "oversharing": oversharing_rate(b, user, q.purpose, policy),
                         "latency_ms": round(b.latency_ms, 2), "n_context": len(b.context_sent)},
            "guarded": {"pii_leakage": pii_leakage_rate(g, by_id), "oversharing": oversharing_rate(g, user, q.purpose, policy),
                        "latency_ms": round(g.latency_ms, 2), "n_context": len(g.context_sent),
                        "dropped": [(c.doc_id, c.drop_reason) for c in g.chunks if c.dropped]},
        })
    p, r = redaction_precision_recall(guard.pii.detector, corpus)
    metrics = {
        "pii_leakage_rate": {"baseline": _mean([x["baseline"]["pii_leakage"] for x in rows]), "guarded": _mean([x["guarded"]["pii_leakage"] for x in rows]), "status": "measured"},
        "oversharing_rate": {"baseline": _mean([x["baseline"]["oversharing"] for x in rows]), "guarded": _mean([x["guarded"]["oversharing"] for x in rows]), "status": "measured"},
        "injection_success_rate": {"baseline": None, "guarded": None, "status": "next stage"},
        "redaction_precision": {"baseline": None, "guarded": round(p, 4), "status": "measured"},
        "redaction_recall": {"baseline": None, "guarded": round(r, 4), "status": "measured"},
        "answer_quality_retention": {"baseline": None, "guarded": None, "status": "next stage"},
        "added_latency_ms": {"baseline": 0.0, "guarded": round(latency_delta_ms(b_res, g_res), 2), "status": "measured"},
    }
    return EvalReport(metrics, rows, len(queries), getattr(provider, "last_used", provider.name), guard.pii.detector.names_mode)


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else f"{v}"


def to_markdown(rep: EvalReport) -> str:
    lines = [f"# Evaluation report", "",
             f"Queries: {rep.n_queries} · Provider: `{rep.provider}` · Name detection: `{rep.names_mode}`", "",
             "| Metric | Baseline RAG | Guarded RAG | Better | Status |", "|---|---|---|---|---|"]
    for key, label, direction in METRIC_ROWS:
        m = rep.metrics[key]
        b, g = m["baseline"], m["guarded"]
        if key == "added_latency_ms":
            lines.append(f"| {label} | 0 | {_fmt(g)} ms | {direction} | {m['status']} |")
        else:
            lines.append(f"| {label} | {_fmt(b)} | {_fmt(g)} | {direction} | {m['status']} |")
    lines += ["", "## Per-query", "", "| id | user | purpose | probe | poisoned | base leak | guard leak | base overshare | guard overshare | dropped |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep.per_query:
        lines.append(f"| {r['id']} | {r['user']} | {r['purpose']} | {'✓' if r['cross_dept_probe'] else ''} | {'✓' if r['targets_poisoned'] else ''} | "
                     f"{_fmt(r['baseline']['pii_leakage'])} | {_fmt(r['guarded']['pii_leakage'])} | {_fmt(r['baseline']['oversharing'])} | {_fmt(r['guarded']['oversharing'])} | {len(r['guarded']['dropped'])} |")
    return "\n".join(lines) + "\n"


def save_report(rep: EvalReport, json_path: str | Path, md_path: str | Path) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(asdict(rep), indent=1, ensure_ascii=False))
    Path(md_path).write_text(to_markdown(rep))
