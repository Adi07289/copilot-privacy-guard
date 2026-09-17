"""Sparse TF-IDF vector index. Dense embeddings are a drop-in behind the same Protocol."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from privacyguard.models import Document, Chunk


class Retriever(Protocol):
    def index(self, docs: list[Document]) -> None: ...
    def search(self, query: str, k: int = 6) -> list[Chunk]: ...


class TfidfIndex:
    def __init__(self):
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._docs: list[Document] = []
        self._matrix = None

    def index(self, docs: list[Document]) -> None:
        self._docs = list(docs)
        self._matrix = self._vec.fit_transform([d.title + "\n" + d.text for d in docs])

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        if self._matrix is None or not self._docs:
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        order = np.argsort(-sims)[:k]
        return [Chunk.from_document(self._docs[i], float(sims[i])) for i in order if sims[i] > 0]
