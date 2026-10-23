"""Retrievers behind one Protocol: sparse TF-IDF and dense bge-small (fastembed). build_indexes() builds a raw
and a redacted (privacy-preserving) index so the guarded pipeline's vector store never holds raw PII."""
from __future__ import annotations
from dataclasses import replace
from typing import Callable, Protocol
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from privacyguard.models import Chunk, Document

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
Embedder = Callable[[list[str], str], np.ndarray]


class Retriever(Protocol):
    indexed_texts: list[str]
    def index(self, docs: list[Document]) -> None: ...
    def search(self, query: str, k: int = 6) -> list[Chunk]: ...


def _doc_text(d: Document) -> str:
    return d.title + "\n" + d.text


class TfidfIndex:
    def __init__(self):
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._docs: list[Document] = []
        self._matrix = None
        self.indexed_texts: list[str] = []

    def index(self, docs: list[Document]) -> None:
        self._docs = list(docs)
        self.indexed_texts = [_doc_text(d) for d in docs]
        self._matrix = self._vec.fit_transform(self.indexed_texts)

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        if self._matrix is None or not self._docs:
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        order = np.argsort(-sims)[:k]
        return [Chunk.from_document(self._docs[i], float(sims[i])) for i in order if sims[i] > 0]


def dense_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except Exception:
        return False


def _fastembed(model_name: str) -> Embedder:
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name)

    def embed(texts: list[str], kind: str) -> np.ndarray:
        it = model.query_embed(texts) if kind == "query" else model.passage_embed(texts)
        return np.array(list(it), dtype=np.float32)
    return embed


class DenseIndex:
    def __init__(self, model_name: str = DENSE_MODEL, embedder: Embedder | None = None):
        self._embed = embedder or _fastembed(model_name)
        self._docs: list[Document] = []
        self._mat: np.ndarray | None = None
        self.indexed_texts: list[str] = []

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def index(self, docs: list[Document]) -> None:
        self._docs = list(docs)
        self.indexed_texts = [_doc_text(d) for d in docs]
        self._mat = self._norm(self._embed(self.indexed_texts, "passage"))

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        if self._mat is None or not self._docs:
            return []
        sims = self._mat @ self._norm(self._embed([query], "query"))[0]
        order = np.argsort(-sims)[:k]
        return [Chunk.from_document(self._docs[i], float(sims[i])) for i in order]


def _apply_mapping(s: str, mapping: dict[str, str]) -> str:
    for placeholder, value in sorted(mapping.items(), key=lambda kv: -len(kv[1])):
        s = s.replace(value, placeholder)
    return s


def redact_documents(docs: list[Document], redactor) -> list[Document]:
    """Redact each document as a whole: values detected in the body are also replaced in the title."""
    out = []
    for d in docs:
        text, mapping = redactor.redact(d.text)
        title, _ = redactor.redact(_apply_mapping(d.title, mapping))
        out.append(replace(d, text=text, title=title, pii_spans=[]))
    return out


def build_indexes(corpus: list[Document], redactor, kinds: tuple[str, ...] = ("tfidf",),
                  dense_embedder: Embedder | None = None) -> dict[str, Retriever]:
    redacted = redact_documents(corpus, redactor)
    out: dict[str, Retriever] = {}
    for kind in kinds:
        for variant, docs in (("raw", corpus), ("redacted", redacted)):
            idx = TfidfIndex() if kind == "tfidf" else DenseIndex(embedder=dense_embedder)
            idx.index(docs)
            out[f"{kind}:{variant}"] = idx
    return out
