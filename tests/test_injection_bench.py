from collections import Counter
from privacyguard.corpus.generator import generate_corpus
from privacyguard.eval.injection_bench import FAMILIES, HARD_NEGATIVES, PAYLOADS, bench_samples, run_bench, to_markdown

class MarkerJudge:
    """Flags exactly the carrier-wrapped payloads (they all start with 'TICKET #40')."""
    def judge(self, text):
        return 0.9 if text.startswith("TICKET #40") else 0.1

def test_payload_bank_shape():
    assert len(PAYLOADS) == 24 and Counter(f for f, _ in PAYLOADS) == {f: 4 for f in FAMILIES}
    assert len(HARD_NEGATIVES) == 8

def test_samples():
    s = bench_samples(generate_corpus(42, 80))
    assert sum(x.is_injection for x in s) == 24 and len(s) == 24 + 40 + 8
    assert not any(x.family == "clean" and "CANARY" in x.text for x in s)

def test_heuristic_vs_judge():
    b = run_bench(bench_samples(generate_corpus(42, 80)), MarkerJudge())
    assert b["tpr"]["judge"] == 1.0 and b["fpr_all"]["judge"] == 0.0 and b["tpr"]["combined"] == 1.0
    assert b["by_family"]["paraphrased"]["heuristic"] == 0.0 and b["by_family"]["hindi"]["heuristic"] == 0.0
    assert b["by_family"]["direct"]["heuristic"] >= 0.5 and b["fpr_hard"]["heuristic"] >= 0.5
    assert "| paraphrased |" in to_markdown(b)

def test_without_judge_marks_judge_unmeasured():
    b = run_bench(bench_samples(generate_corpus(42, 80)), None)
    assert b["judge_available"] is False and b["tpr"]["judge"] is None and b["tpr"]["combined"] == b["tpr"]["heuristic"]

def test_prompt_guard_column_when_given():
    b = run_bench(bench_samples(generate_corpus(42, 80)), MarkerJudge(), guard=MarkerJudge())
    assert b["tpr"]["prompt_guard"] == 1.0 and "Prompt Guard 2" in to_markdown(b)
    assert "prompt_guard" not in run_bench(bench_samples(generate_corpus(42, 80)), None)["tpr"]
