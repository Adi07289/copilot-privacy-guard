from privacyguard.corpus.generator import generate_corpus
from privacyguard.eval.harness import aggregate, merge_reports, run_comparison, to_markdown
from privacyguard.eval.queries import QUERY_SET
from privacyguard.llm import ExtractiveProvider

def test_run_comparison_on_the_corpus(tmp_path):
    corpus = generate_corpus(42, 80)
    rep = run_comparison(corpus, QUERY_SET, ExtractiveProvider(), configs=["rbac", "full"], audit_path=tmp_path / "a.jsonl")
    a = rep.aggregates
    assert rep.configs == ["none", "rbac", "full"] and len(rep.runs) == 36 * 3
    assert a["full"]["oversharing_rate"] == 0.0 < a["none"]["oversharing_rate"]
    assert a["rbac"]["oversharing_rate"] <= a["none"]["oversharing_rate"]
    assert a["full"]["context_pii_exposure"] == 0.0 < a["none"]["context_pii_exposure"]
    assert a["full"]["index_pii_exposure"] < a["none"]["index_pii_exposure"]
    assert a["full"]["injection_exposure_rate"] == 0.0 and a["none"]["injection_exposure_rate"] == 1.0
    assert a["none"]["answer_quality_rouge_l"] == 1.0 and a["none"]["added_latency_ms"] == 0.0
    assert a["full"]["injection_success_rate"] is None          # extractive answers cannot obey anything

def _row(config, qid, success, provider):
    return {"config": config, "query_id": qid, "kind": "injection", "provider_used": provider, "latency_ms": 1.0,
            "pii_leakage": None, "context_exposure": None, "oversharing": None, "inj_exposed": 1, "inj_total": 1,
            "inj_success": success}

def test_injection_success_excludes_fallback():
    runs = [_row("none", "a", True, "groq"), _row("none", "b", False, "extractive"), _row("none", "c", False, "groq")]
    agg = aggregate(runs, ["none"], {"none": 0.5})["none"]
    assert agg["injection_success_rate"] == 0.5 and agg["injection_success_n"] == 2
    assert agg["injection_success_excluded_fallback"] == 1

def test_merge_without_llm_and_markdown(tmp_path):
    corpus = generate_corpus(42, 80)
    det = run_comparison(corpus, QUERY_SET[:6], ExtractiveProvider(), configs=["full"], audit_path=tmp_path / "a.jsonl")
    merged = merge_reports(det, None)
    assert merged["aggregates"]["full"]["injection_success_rate"] is None
    assert merged["aggregates"]["full"]["answer_quality_source"] == "extractive"
    md = to_markdown(merged)
    assert "Copilot Privacy Guard (ours)" in md and "simplified analogue" in md
