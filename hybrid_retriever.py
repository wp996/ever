"""RRF hybrid retrieval for the local Chroma collection.

Score semantics:
- vector mode: Chroma distance, lower is better
- bm25 mode: BM25 score, higher is better
- hybrid mode: RRF score, higher is better
"""
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from rank_bm25 import BM25Okapi

import config_data as config
from keyword_retriever import tokenize
from vector_stores import VectorStoreService


VALID_RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}


def document_key(doc: Document) -> tuple[str, str]:
    """Identify the same stored chunk across Vector and BM25 result lists."""
    return str(doc.metadata.get("source", "")), doc.page_content


def rrf_merge(
    vector_results: Sequence[Document],
    bm25_results: Sequence[Document],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Merge ranked results with reciprocal rank fusion.

    Rank is 1-based and only rank contributes to the score:
    score = 1 / (k + rank). Duplicate documents add both contributions.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    scores: dict[tuple[str, str], float] = {}
    documents: dict[tuple[str, str], Document] = {}

    for rank, doc in enumerate(vector_results, start=1):
        key = document_key(doc)
        documents.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(bm25_results, start=1):
        key = document_key(doc)
        documents.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked = sorted(
        ((documents[key], score) for key, score in scores.items()),
        key=lambda item: (-item[1], document_key(item[0])),
    )
    return ranked


class HybridRetriever:
    """Retriever with vector, bm25 and hybrid modes."""

    def __init__(
        self,
        embedding,
        top_k: int | None = None,
        rrf_k: int | None = None,
        mode: str | None = None,
    ):
        self.vector_service = VectorStoreService(embedding=embedding)
        self.top_k = config.retrieval_top_k if top_k is None else top_k
        self.rrf_k = config.rrf_k if rrf_k is None else rrf_k
        self.mode = self._resolve_mode(mode)
        self.documents = self._load_documents()
        self.bm25 = self._build_bm25()

    def _resolve_mode(self, mode: str | None) -> str:
        selected = (mode or config.retrieval_mode).lower()
        if selected not in VALID_RETRIEVAL_MODES:
            valid = ", ".join(sorted(VALID_RETRIEVAL_MODES))
            raise ValueError(f"mode must be one of: {valid}")
        return selected

    def _load_documents(self) -> list[Document]:
        texts, metadatas = self.vector_service.get_all_chunks()
        texts = [] if texts is None else list(texts)
        metadatas = [] if metadatas is None else list(metadatas)
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas)
        ]

    def _build_bm25(self) -> BM25Okapi | None:
        if not self.documents:
            return None
        tokenized_documents = [tokenize(doc.page_content) for doc in self.documents]
        return BM25Okapi(tokenized_documents)

    def refresh(self) -> None:
        """Reload Chroma chunks and rebuild the BM25 index."""
        self.documents = self._load_documents()
        self.bm25 = self._build_bm25()

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        return self.vector_service.vector_store.similarity_search_with_score(
            query,
            k=top_k,
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )
        return [
            (self.documents[index], float(scores[index]))
            for index in order
            if float(scores[index]) > 0
        ][:top_k]

    def search_with_scores(
        self,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        selected = self._resolve_mode(mode)
        limit = self.top_k if top_k is None else top_k

        if selected == "vector":
            return self._vector_search(query, limit)

        bm25_results = self._bm25_search(query, limit)
        if selected == "bm25":
            return bm25_results

        vector_results = self._vector_search(query, limit)

        merged_all=rrf_merge(
            [doc for doc, _ in vector_results],
            [doc for doc, _ in bm25_results],
            k=self.rrf_k,
        )
        return merged_all[:limit]

    def search(
        self,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        return [
            doc
            for doc, _ in self.search_with_scores(query, mode=mode, top_k=top_k)
        ]

    def get_retriever(self, mode: str | None = None):
        """Return a LangChain Runnable compatible with the existing Chain."""
        selected = self._resolve_mode(mode)

        def _retrieve(query: str) -> list[Document]:
            return self.search(query, mode=selected)

        return RunnableLambda(_retrieve)
