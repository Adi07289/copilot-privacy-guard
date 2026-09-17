import json
from privacyguard.corpus.generator import generate_corpus
from privacyguard.eval.harness import run_eval, to_markdown, save_report
from privacyguard.eval.queries import QUERY_SET
from privacyguard.llm import ExtractiveProvider

def test_run_eval_end_to_end(tmp_path):
    rep = run_eval(generate_corpus(42, 80), QUERY_SET, ExtractiveProvider(), tmp_path / "a.jsonl")
    m = rep.metrics
    assert rep.n_queries == 20 and rep.provider == "extractive"
    assert m["oversharing_rate"]["guarded"] == 0.0 and m["oversharing_rate"]["baseline"] > 0
    assert m["pii_leakage_rate"]["guarded"] <= m["pii_leakage_rate"]["baseline"]
    assert 0 < m["redaction_recall"]["guarded"] <= 1.0
    assert m["injection_success_rate"]["status"] == "next stage" and m["injection_success_rate"]["guarded"] is None
    md = to_markdown(rep)
    assert "| PII leakage rate" in md and "next stage" in md
    save_report(rep, tmp_path / "e.json", tmp_path / "e.md")
    assert json.loads((tmp_path / "e.json").read_text())["n_queries"] == 20
