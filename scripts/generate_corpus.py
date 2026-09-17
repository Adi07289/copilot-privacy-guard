"""Usage: .venv/bin/python scripts/generate_corpus.py --seed 42 --n 80 --out data/corpus.json"""
import argparse
from privacyguard.corpus.generator import generate_corpus, save_corpus

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default="data/corpus.json")
    a = ap.parse_args()
    docs = generate_corpus(a.seed, a.n)
    save_corpus(docs, a.out)
    pii = sum(len(d.pii_spans) for d in docs)
    print(f"wrote {len(docs)} docs, {pii} PII spans, {sum(d.is_poisoned for d in docs)} poisoned → {a.out}")
