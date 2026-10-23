"""Quick check: Vanilla RAG vs ours, extractive provider. Full study: scripts/run_all.py"""
from privacyguard.corpus.generator import generate_corpus
from privacyguard.eval.harness import merge_reports, run_comparison, to_markdown
from privacyguard.eval.queries import QUERY_SET
from privacyguard.llm import ExtractiveProvider

if __name__ == "__main__":
    rep = run_comparison(generate_corpus(), QUERY_SET, ExtractiveProvider(), configs=["full"])
    print(to_markdown(merge_reports(rep)))
