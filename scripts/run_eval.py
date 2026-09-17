"""Usage: .venv/bin/python scripts/run_eval.py [--provider extractive|groq] [--corpus data/corpus.json]"""
import argparse, os
from pathlib import Path
from privacyguard.corpus.generator import generate_corpus, load_corpus, save_corpus
from privacyguard.eval.harness import run_eval, save_report, to_markdown
from privacyguard.eval.queries import QUERY_SET
from privacyguard.llm import ExtractiveProvider, get_provider

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="extractive")
    ap.add_argument("--corpus", default="data/corpus.json")
    a = ap.parse_args()
    if not Path(a.corpus).exists():
        print("corpus missing — generating with seed 42"); save_corpus(generate_corpus(), a.corpus)
    corpus = load_corpus(a.corpus)
    provider = ExtractiveProvider() if a.provider == "extractive" else get_provider()
    rep = run_eval(corpus, QUERY_SET, provider, "data/audit.jsonl")
    save_report(rep, "results/eval.json", "results/eval.md")
    print(to_markdown(rep))
