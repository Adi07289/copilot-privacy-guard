"""Comparative evaluation: every configuration runs the same queries over the same corpus, retriever and LLM."""
from __future__ import annotations
import json, statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from privacyguard.configs import CONFIG_ORDER, CONFIGS
from privacyguard.controls.access import AccessPolicy
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.controls.pii import PIIRedactor
from privacyguard.corpus.generator import get_persona
from privacyguard.eval import metrics as M
from privacyguard.eval.queries import EvalQuery
from privacyguard.llm import ROOT, LLMProvider
from privacyguard.models import Document
from privacyguard.pipeline import ConfigurableRAG
from privacyguard.retrieval import build_indexes

METRICS = [
    ("context_pii_exposure", "Context PII exposure (to LLM provider)", "lower"),
    ("pii_leakage_rate", "Answer PII leakage (extractive)", "lower"),
    ("oversharing_rate", "Over-sharing retrieval rate", "lower"),
    ("injection_exposure_rate", "Injection exposure rate", "lower"),
    ("injection_success_rate", "Injection success rate (LLM)", "lower"),
    ("answer_quality_rouge_l", "Answer-quality retention (ROUGE-L)", "higher"),
    ("answer_quality_cosine", "Answer-quality retention (cosine)", "higher"),
    ("index_pii_exposure", "Index PII exposure", "lower"),
    ("added_latency_ms", "Added latency (ms)", "lower"),
]
LLM_KEYS = ("injection_success_rate", "injection_success_n", "injection_success_excluded_fallback",
            "answer_quality_rouge_l", "answer_quality_cosine")


@dataclass
class ComparisonReport:
    provider: str
    configs: list[str]
    aggregates: dict[str, dict]
    runs: list[dict]
    meta: dict = field(default_factory=dict)


def _mean(xs) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(runs: list[dict], configs: list[str], index_exposure: dict[str, float | None]) -> dict[str, dict]:
    ref_lat = {x["query_id"]: x["latency_ms"] for x in runs if x["config"] == "none"}
    out = {}
    for c in configs:
        rows = [x for x in runs if x["config"] == c]
        tot = sum(x["inj_total"] for x in rows)
        measured = [x for x in rows if x["kind"] == "injection" and x["inj_success"] is not None
                    and x["provider_used"] != "extractive"]
        out[c] = {
            "context_pii_exposure": _mean(x["context_exposure"] for x in rows),
            "pii_leakage_rate": _mean(x["pii_leakage"] for x in rows),
            "oversharing_rate": _mean(x["oversharing"] for x in rows),
            "injection_exposure_rate": round(sum(x["inj_exposed"] for x in rows) / tot, 4) if tot else None,
            "injection_success_rate": _mean(1.0 if x["inj_success"] else 0.0 for x in measured),
            "injection_success_n": len(measured),
            "injection_success_excluded_fallback": sum(1 for x in rows if x["kind"] == "injection"
                                                       and x["inj_success"] is not None and x["provider_used"] == "extractive"),
            "answer_quality_rouge_l": _mean(x.get("rouge_l") for x in rows),
            "answer_quality_cosine": _mean(x.get("cosine") for x in rows),
            "answer_quality_n": sum(1 for x in rows if x.get("rouge_l") is not None),
            "index_pii_exposure": index_exposure.get(c),
            # median, not mean: one rate-limit sleep or cold-cache judge call must not define the cost
            "added_latency_ms": round(statistics.median(x["latency_ms"] - ref_lat.get(x["query_id"], x["latency_ms"])
                                                        for x in rows), 2) if rows else None,
            "n_runs": len(rows),
        }
    return out


def _score_quality(runs: list[dict]) -> None:
    """ROUGE-L / cosine vs the `none` answer, only when both answers came from the same provider (no fallback mix)."""
    ref = {x["query_id"]: x for x in runs if x["config"] == "none"}
    for x in runs:
        r = ref.get(x["query_id"])
        if x["kind"] == "utility" and r is not None and r["provider_used"] == x["provider_used"]:
            x["rouge_l"] = round(M.rouge_l_f1(x["answer"], r["answer"]), 4)
            x["cosine"] = round(M.cosine_sim(x["answer"], r["answer"]), 4)


def run_comparison(corpus: list[Document], queries: list[EvalQuery], provider: LLMProvider,
                   configs: list[str] | None = None, judge=None,
                   audit_path: str | Path = ROOT / "data" / "audit_eval.jsonl", kinds: tuple[str, ...] | None = None,
                   dense_embedder=None, k: int = 6, progress=None) -> ComparisonReport:
    configs = [c for c in (configs or CONFIG_ORDER)]
    if "none" not in configs:
        configs.insert(0, "none")
    pii = PIIRedactor()
    indexes = build_indexes(corpus, pii, kinds=tuple(dict.fromkeys(CONFIGS[c].retriever for c in configs)),
                            dense_embedder=dense_embedder)
    audit = HashChainedAuditLog(audit_path)
    pipes = {c: ConfigurableRAG(indexes, provider, CONFIGS[c], audit=audit if CONFIGS[c].audit else None,
                                judge=judge, k=k, redactor=pii) for c in configs}
    by_id, policy = {d.id: d for d in corpus}, AccessPolicy("purpose")
    qs = [q for q in queries if kinds is None or q.kind in kinds]
    runs: list[dict] = []
    for i, q in enumerate(qs, 1):
        if progress:
            progress(f"{i}/{len(qs)} {q.id}")
        user = get_persona(q.user_id)
        for c in configs:
            r = pipes[c].run(user, q.purpose, q.text)
            exposed, total = M.injection_exposure(r)
            runs.append({"config": c, "query_id": q.id, "kind": q.kind, "user": q.user_id, "purpose": q.purpose,
                         "answer": r.answer, "provider_used": r.provider_used, "latency_ms": round(r.latency_ms, 2),
                         "retrieved": [ch.doc_id for ch in r.chunks],
                         "dropped": [[ch.doc_id, ch.drop_reason] for ch in r.chunks if ch.dropped],
                         "pii_leakage": M.pii_leakage_rate(r, by_id), "context_exposure": M.context_pii_exposure(r, by_id),
                         "oversharing": M.oversharing_rate(r, user, q.purpose, policy),
                         "inj_exposed": exposed, "inj_total": total, "inj_success": M.injection_success(r, by_id)})
    _score_quality(runs)
    exposure = {}
    for c in configs:
        v = M.index_pii_exposure(pipes[c].retriever.indexed_texts, corpus)
        exposure[c] = round(v, 4) if v is not None else None
    return ComparisonReport(getattr(provider, "name", "?"), configs, aggregate(runs, configs, exposure), runs,
                            {"k": k, "n_queries": len(qs), "kinds": sorted({q.kind for q in qs})})


def merge_reports(det: ComparisonReport, llm: ComparisonReport | None = None) -> dict:
    """Deterministic metrics come from the extractive pass; LLM-dependent ones from the Groq pass when present."""
    agg = {c: dict(v) for c, v in det.aggregates.items()}
    for c in agg:
        src = llm.aggregates.get(c) if llm else None
        if src:
            for key in LLM_KEYS:
                agg[c][key] = src.get(key)
            agg[c]["answer_pii_leakage_llm"] = src.get("pii_leakage_rate")
            agg[c]["answer_quality_source"] = "llm" if src.get("answer_quality_n") else "extractive"
        else:
            agg[c]["injection_success_rate"] = None
            agg[c]["answer_pii_leakage_llm"] = None
            agg[c]["answer_quality_source"] = "extractive"
    return {"configs": det.configs, "labels": {c: CONFIGS[c].label for c in det.configs},
            "analogues": {c: CONFIGS[c].analogue_of for c in det.configs},
            "groups": {c: CONFIGS[c].group for c in det.configs}, "aggregates": agg,
            "meta": {"deterministic": det.meta, "llm": llm.meta if llm else None,
                     "llm_provider": llm.provider if llm else None}}


def _fmt(key: str, v) -> str:
    if v is None:
        return "—"
    if key == "added_latency_ms":
        return f"{v:+.1f}"
    return f"{v:.1%}"


def to_markdown(merged: dict) -> str:
    agg, labels = merged["aggregates"], merged["labels"]
    head = "| Configuration | " + " | ".join(lbl for _, lbl, _ in METRICS) + " |"
    lines = ["# Comparative evaluation", "",
             "Every row runs the same 36 queries over the same corpus, retriever and LLM; only the defence differs. "
             "Rows other than *ours* are **simplified analogues of a defence family**, not reproductions of the cited papers.",
             "", head, "|" + "---|" * (len(METRICS) + 1)]
    for c in merged["configs"]:
        lines.append(f"| {labels[c]} | " + " | ".join(_fmt(k, agg[c].get(k)) for k, _, _ in METRICS) + " |")
    lines += ["", "Better: " + ", ".join(f"{lbl} → {b}" for _, lbl, b in METRICS), "",
              "## What each row stands in for", "", "| Configuration | Group | Analogue of |", "|---|---|---|"]
    for c in merged["configs"]:
        lines.append(f"| {labels[c]} | {merged['groups'][c]} | {merged['analogues'][c]} |")
    return "\n".join(lines) + "\n"


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def save_report(merged: dict, json_path: str | Path, md_path: str | Path) -> None:
    save_json(merged, json_path)
    Path(md_path).write_text(to_markdown(merged))
