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

def test_injection_success_counts_only_injection_queries():
    runs = [_row("none", "a", True, "groq"), {**_row("none", "u", True, "groq"), "kind": "utility"}]
    agg = aggregate(runs, ["none"], {"none": 0.5})["none"]
    assert agg["injection_success_n"] == 1                                   # I-1

def test_added_latency_is_the_median_delta():
    runs = [{**_row("none", q, None, "groq"), "latency_ms": 10.0} for q in "abc"] + \
           [{**_row("x", "a", None, "groq"), "latency_ms": 20.0}, {**_row("x", "b", None, "groq"), "latency_ms": 30.0},
            {**_row("x", "c", None, "groq"), "latency_ms": 20010.0}]          # one rate-limit outlier
    assert aggregate(runs, ["none", "x"], {})["x"]["added_latency_ms"] == 20.0  # I-2

def test_answer_quality_skips_fallback_pairs(tmp_path, monkeypatch):
    from privacyguard.eval import harness as H
    rows = [{**_row("none", "u", None, "groq"), "kind": "utility", "answer": "a b"},
            {**_row("full", "u", None, "extractive"), "kind": "utility", "answer": "a b"}]
    H._score_quality(rows)
    assert "rouge_l" not in rows[1]                                             # I-5

def test_leakage_label_names_its_source():
    from privacyguard.eval.harness import METRICS
    assert dict((k, l) for k, l, _ in METRICS)["pii_leakage_rate"].endswith("(extractive)")   # M-4
