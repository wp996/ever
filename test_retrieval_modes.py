"""Standalone smoke test for RRF and retrieval modes."""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config_data as config
from hybrid_retriever import HybridRetriever, rrf_merge


def test_rrf_formula() -> None:
    doc_a = Document(page_content="A", metadata={"source": "a.txt"})
    doc_b = Document(page_content="B", metadata={"source": "b.txt"})
    doc_c = Document(page_content="C", metadata={"source": "c.txt"})

    merged = rrf_merge([doc_a, doc_b], [doc_b, doc_c], k=60)
    scores = {doc.page_content: score for doc, score in merged}

    expected_b = 1 / (60 + 2) + 1 / (60 + 1)
    assert abs(scores["B"] - expected_b) < 1e-12
    assert [doc.page_content for doc, _ in merged] == ["B", "A", "C"]
    print("[RRF] B = 1/(60+2) + 1/(60+1), formula test passed")


def main() -> None:
    test_rrf_formula()

    retriever = HybridRetriever(
        embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        top_k=config.retrieval_top_k,
        rrf_k=config.rrf_k,
    )
    query = "身高168cm，体重110斤，穿多大码？"

    chain_retriever = retriever.get_retriever("bm25")
    chain_docs = chain_retriever.invoke(query)
    assert isinstance(chain_docs, list)
    assert all(isinstance(doc, Document) for doc in chain_docs)
    print(f"\n[chain] get_retriever('bm25').invoke -> {len(chain_docs)} Document objects")

    for mode in ("bm25", "vector", "hybrid"):
        print(f"\n[{mode}] query={query}")
        try:
            hits = retriever.search_with_scores(query, mode=mode, top_k=3)
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}")
            continue

        if not hits:
            print("  no positive BM25 hit" if mode == "bm25" else "  no hit")
            continue

        for rank, (doc, score) in enumerate(hits, start=1):
            source = doc.metadata.get("source", "<unknown>")
            preview = " ".join(doc.page_content.split())[:60]
            print(f"  {rank}. score={score:.6f} source={source} preview={preview}")


if __name__ == "__main__":
    main()
