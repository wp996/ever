"""RerankerService 独立测试：不接主链路，只验证打分与排序。"""
import time

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config_data as config
from reranker import RerankerService
from vector_stores import VectorStoreService

QUERY = "毛衣起球了怎么办"


def build_candidates():
    """从 Chroma 取 15 条真实 chunk，混入 3 条无关干扰项。"""
    service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
    texts, metadatas = service.get_all_chunks()

    candidates = []
    # 粗排分数故意按取出顺序递减，模拟"粗排没排对"的情况
    for rank, (text, metadata) in enumerate(list(zip(texts, metadatas))[:15]):
        candidates.append((Document(page_content=text, metadata=metadata), 1.0 - rank * 0.01))

    # 干扰项：明显无关，粗排给最低分
    noise = [
        "今天天气不错，适合出去散步。",
        "Python 是一种解释型高级编程语言。",
        "本公司成立于2010年，总部位于上海。",
    ]
    for i, text in enumerate(noise):
        candidates.append((Document(page_content=text, metadata={"source": "干扰项.txt"}), 0.001))

    return candidates


def main():
    candidates = build_candidates()
    print(f"候选数：{len(candidates)}   query：{QUERY}\n")

    reranker = RerankerService()

    # 第一次调用含模型加载（约 30s）
    t0 = time.time()
    results = reranker.rerank(QUERY, candidates, top_n=5)
    first = time.time() - t0

    # 第二次调用才是纯推理耗时
    t0 = time.time()
    results = reranker.rerank(QUERY, candidates, top_n=5)
    pure = time.time() - t0

    print(f"首次（含加载）{first:.2f}s    纯推理 {pure:.2f}s\n")
    print(f"{'#':<3}{'chunk_id':<10}{'source':<18}{'原分数':>8}{'重排分':>9}   预览")
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "?")
        prev = " ".join(r["text"].split())[:26]
        print(f"{i:<3}{r['chunk_id']:<10}{src:<18}{r['original_score']:>8.4f}{r['rerank_score']:>9.4f}   {prev}")

    # ---------- 断言 ----------
    assert len(results) == 5, "返回条数必须等于 top_n"
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "必须按 rerank_score 降序"
    for r in results:
        assert set(r) == {"chunk_id", "text", "original_score", "rerank_score", "metadata"}, "字段缺失"
    assert not [r for r in results if r["metadata"].get("source") == "干扰项.txt"], "干扰项不应进 top5"
    assert all(0.0 <= r["rerank_score"] <= 1.0 for r in results), "normalize=True 时分数应在 0~1"
    print("\n全部断言通过")


if __name__ == "__main__":
    main()
