import json
import time

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config_data as config
from hybrid_retriever import HybridRetriever
from reranker import RerankerService
from test_reranker import QUERY

QUERY = "防晒衣应该怎么洗"
TOP_N = 20

KEY_A = "防晒涂层"
KEY_B = "柔顺剂"

def brief(text,n=40):
    one_line=" ".join(text.split())
    return one_line[:n]+("...." if len(one_line)>n else "")

def mark(text):
    tags=[]
    if KEY_A in text:
        tags.append("A:防晒涂层")
    if KEY_B in text:
        tags.append("B:柔顺剂")
    return "  <<" + "/".join(tags) + ">>" if tags else ""
def main():
    embedding = DashScopeEmbeddings(model=config.embedding_model_name)
    hybrid= HybridRetriever(embedding,mode="hybrid")
    reranker = RerankerService()
    reranker.rerank("预热", [Document(page_content="预热", metadata={"source": "预热"})])

    # ---------- 1. 粗排 20 条 ----------
    candidates = hybrid.search_with_scores(QUERY, mode="hybrid", top_k=TOP_N)
    print(f"\n===== 粗排 Top{TOP_N}（query: {QUERY}）=====")
    for i, (doc, score) in enumerate(candidates, 1):
        src = doc.metadata.get("source", "?")
        print(f"{i:>2}. score={score:.4f}  {src:<20}  {brief(doc.page_content)}{mark(doc.page_content)}")

    # ---------- 2. 重排 20 条 ----------
    reranked = reranker.rerank(QUERY, candidates, top_n=TOP_N)
    print(f"\n===== 重排 Top{TOP_N} =====")
    for i, r in enumerate(reranked, 1):
        src = r["metadata"].get("source", "?")
        # rerank 返回的分数键名视你的 RerankerService 实现而定，常见是 "score" 或 "rerank_score"
        score = r.get("score", r.get("rerank_score", float("nan")))
        print(f"{i:>2}. rerank={score:.4f}  {src:<20}  {brief(r['text'])}{mark(r['text'])}")

    # ---------- 3. 专门盯两个片段：粗排位置 vs 重排位置 + 重排分差 ----------
    def find(cands, key):
        """在候选里找含 key 的第一条，返回 (位置, source, 分数)；找不到返回 None。"""
        for i, c in enumerate(cands, 1):
            if isinstance(c, dict):
                text, src, sc = c["text"], c["metadata"].get("source", "?"), c.get("score", c.get("rerank_score"))
            else:
                doc, sc = c
                text, src = doc.page_content, doc.metadata.get("source", "?")
            if key in text:
                return i, src, sc
        return None

    pos_a_raw = find(candidates, KEY_A)
    pos_b_raw = find(candidates, KEY_B)
    pos_a_re = find(reranked, KEY_A)
    pos_b_re = find(reranked, KEY_B)

    print("\n===== 两个关键片段的位置与分数 =====")
    print(f"A（含“{KEY_A}”）  粗排: {pos_a_raw}   重排: {pos_a_re}")
    print(f"B（含“{KEY_B}”）  粗排: {pos_b_raw}   重排: {pos_b_re}")

    if pos_a_re and pos_b_re:
        sc_a = pos_a_re[2]
        sc_b = pos_b_re[2]
        print(f"\n重排分数：A={sc_a:.4f}   B={sc_b:.4f}   差(A-B)={sc_a - sc_b:+.4f}")
        if sc_a > sc_b:
            print("→ A 分数更高，重排更偏向“防晒涂层”那段")
        elif sc_b > sc_a:
            print("→ B 分数更高，重排更偏向“柔顺剂”那段（注意：这可能是 q27 掉队的原因）")
        else:
            print("→ 两者分数相同")


if __name__ == "__main__":
    main()
