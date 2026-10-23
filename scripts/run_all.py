# scripts/run_all.py
"""Regenerate every result: corpus -> comparison (extractive pass for deterministic metrics, Groq pass for
LLM-dependent ones) -> injection bench -> PII detectors -> compliance. Usage:
  .venv/bin/python scripts/run_all.py            # uses Groq when a key or cache is available
  .venv/bin/python scripts/run_all.py --no-llm   # deterministic only"""
import argparse, importlib.metadata as md, json, platform, sys, time
from privacyguard.compliance import checklist as CK
from privacyguard.compliance.evidence import EvidenceStore
from privacyguard.compliance.report import export_json, render_checklist, render_compliance, render_dpia
from privacyguard.configs import CONFIG_ORDER, CONFIGS
from privacyguard.corpus.generator import generate_corpus, save_corpus
from privacyguard.eval import injection_bench as IB
from privacyguard.eval import pii_eval as PE
from privacyguard.eval.harness import merge_reports, run_comparison, save_json, save_report
from privacyguard.eval.heldout_pii import heldout_dataset
from privacyguard.eval.queries import QUERY_SET
from privacyguard.controls.injection import PromptGuardJudge
from privacyguard.llm import ROOT, ExtractiveProvider, get_prompt_guard_client, get_provider, shared_cache
from privacyguard.pipeline import default_judge
from privacyguard.retrieval import dense_available

R = ROOT / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _v(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    corpus = generate_corpus(42, 80)
    save_corpus(corpus, ROOT / "data" / "corpus.json")
    for name in ("audit_eval.jsonl", "audit_llm.jsonl"):
        (ROOT / "data" / name).unlink(missing_ok=True)
    judge = None if a.no_llm else default_judge()
    configs = [c for c in CONFIG_ORDER if CONFIGS[c].retriever != "dense" or dense_available()]
    log(f"configs: {configs}; judge: {'on' if judge else 'off'}")

    det = run_comparison(corpus, QUERY_SET, ExtractiveProvider(), configs=configs, judge=judge,
                         audit_path=ROOT / "data" / "audit_eval.jsonl", progress=lambda m: log(f"extractive {m}"))
    save_json({"aggregates": det.aggregates, "runs": det.runs}, R / "comparison_extractive.json")

    llm = None
    provider = None if a.no_llm else get_provider()
    if provider is not None and provider.name == "groq":
        llm = run_comparison(corpus, QUERY_SET, provider, configs=configs, judge=judge,
                             audit_path=ROOT / "data" / "audit_llm.jsonl", kinds=("utility", "injection"),
                             progress=lambda m: log(f"llm {m}"))
        save_json({"aggregates": llm.aggregates, "runs": llm.runs}, R / "comparison_llm.json")
        fb = sum(1 for x in llm.runs if x["provider_used"] == "extractive")
        log(f"LLM pass: {len(llm.runs) - fb}/{len(llm.runs)} answers from Groq" + (f" — WARNING {fb} fell back" if fb else ""))
    merged = merge_reports(det, llm)
    save_report(merged, R / "comparison.json", R / "comparison.md")
    log("comparison written")

    pg_client = None if a.no_llm else get_prompt_guard_client()
    guard = PromptGuardJudge(pg_client) if pg_client else None
    bench = IB.run_bench(IB.bench_samples(corpus), judge, guard=guard)
    save_json(bench, R / "injection_bench.json"); (R / "injection_bench.md").write_text(IB.to_markdown(bench))
    log("injection bench written")

    pii = PE.compare_detectors({"corpus": corpus, "heldout": heldout_dataset()})
    save_json(pii, R / "pii_detection.json"); (R / "pii_detection.md").write_text(PE.to_markdown(pii))
    log("PII detection written")

    store = EvidenceStore(merged["aggregates"], bench, pii)
    (R / "dpia.md").write_text("# placeholder\n"); (R / "checklist.md").write_text("# placeholder\n")
    results = CK.run_checks(CK.CheckContext(store, corpus, ROOT / "data" / "audit_eval.jsonl"))
    store.checks = {c.id: r for c, r in results}
    (R / "checklist.md").write_text(render_checklist(results))
    (R / "dpia.md").write_text(render_dpia(store))
    (R / "compliance.md").write_text(render_compliance(store))
    save_json(export_json(store, results), R / "compliance.json")
    log("compliance written")

    cache = shared_cache()
    save_json({"seed": 42, "n_docs": len(corpus), "n_queries": len(QUERY_SET), "configs": configs,
               "dense_available": dense_available(), "judge": bool(judge),
               "llm_provider": llm.provider if llm else None,
               "llm_answer_fallbacks": sum(1 for x in llm.runs if x["provider_used"] == "extractive") if llm else None,
               "judge_unavailable_in_bench": bench["judge_unavailable_count"],
               "groq_model": getattr(getattr(getattr(provider, "_p", None), "client", None), "model", None),
               "judge_model": getattr(getattr(judge, "client", None), "model", None),
               "prompt_guard_model": getattr(pg_client, "model", None),
               "cache_entries": len(cache), "cache_hits": cache.hits, "cache_misses": cache.misses,
               "python": platform.python_version(),
               "packages": {p: _v(p) for p in ("scikit-learn", "spacy", "presidio-analyzer", "fastembed", "openai", "faker")},
               "runtime_s": round(time.time() - t0, 1)}, R / "run_meta.json")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
