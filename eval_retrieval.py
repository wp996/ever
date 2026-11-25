"""Day5 评测：hybrid 不重排 vs hybrid+rerank，39 题逐题对比。"""
import json
import time

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config_data as config
from hybrid_retriever import HybridRetriever
from reranker import RerankerService

EVAL_FILE = "data/eval_questions.jsonl"
CANDIDATE_K = config.rerank_candidate_k   # 粗排召回数（20）
TOP_K = 5


def load_questions():
    """逐行读 jsonl，每行一个 json 对象。"""
    items = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract(result):

    if isinstance(result, dict):
        return result["text"] ,result["metadata"]["source"]
    else :
        doc,_score=result
        return doc.page_content , doc.metadata["source"]



def hit_position(results, item):
    """返回首次命中的位置（从 1 开始），未命中返回 None。

    命中规则：source 在 expected_sources 里 且 文本包含至少一个 expected_keywords。
    """
    for pos, result in enumerate(results, 1):

        text, source = extract(result)
        source_ok=source in item["expected_sources"]
        keyword_ok = any(kw in text for kw in item["expected_keywords"])
        if source_ok and keyword_ok:
            return pos
    return None


def main():
    questions = load_questions()
    print(f"评测题数：{len(questions)}   粗排召回 {CANDIDATE_K} → 重排取 {TOP_K}\n")

    embedding = DashScopeEmbeddings(model=config.embedding_model_name)
    hybrid = HybridRetriever(embedding, mode="hybrid")
    reranker = RerankerService()

    # 预热：把 30s 模型加载挪到计时之外，否则第一题的重排耗时会虚高
    reranker.rerank("预热", [Document(page_content="预热", metadata={})])

    rows = []
    time_no = 0.0
    time_re = 0.0

    for item in questions:
        question = item["question"]

        # —— 模式 A：hybrid 不重排，直接取 top5 ——
        t0 = time.time()
        results_a = hybrid.search_with_scores(question, mode="hybrid", top_k=TOP_K)
        time_no += time.time() - t0
        pos_a = hit_position(results_a, item)

        # —— 模式 B：hybrid 召回 20 → rerank 取 5 ——
        t0 = time.time()
        candidates = hybrid.search_with_scores(question, mode="hybrid", top_k=CANDIDATE_K)
        results_b = reranker.rerank(question, candidates, top_n=TOP_K)
        time_re += time.time() - t0
        pos_b = hit_position(results_b, item)

        rows.append((item, pos_a, pos_b))
        mark_a = "-" if pos_a is None else str(pos_a)
        mark_b = "-" if pos_b is None else str(pos_b)
        print(f"{item['id']}  {item['type']:<11}  {question[:14]:<16}  不重排:{mark_a:>3}  重排:{mark_b:>3}")

    # ---------- 汇总 ----------
    total = len(rows)
    hit1_no = sum(1 for _, pos_a, _ in rows if pos_a == 1)
    hit1_re = sum(1 for _, _, pos_b in rows if pos_b == 1)
    print(f"hit_rate@1   不重排 {hit1_no}/{total} = {hit1_no / total:.1%}   "
          f"重排 {hit1_re}/{total} = {hit1_re / total:.1%}")
    print(f"平均耗时/题   不重排 {time_no / total:.2f}s   重排 {time_re / total:.2f}s")
    # MRR：命中位置取倒数，未命中记 0，再求平均
    rr_a = [(1 / pos_a if pos_a else 0) for _, pos_a, _ in rows]
    rr_b = [(1 / pos_b if pos_b else 0) for _, _, pos_b in rows]
    mrr_a = sum(rr_a) / total
    mrr_b = sum(rr_b) / total
    print(f"MRR          不重排 {mrr_a:.3f}   重排 {mrr_b:.3f}")

    pos_a_hit = [pos_a for _, pos_a, _ in rows if pos_a is not None]
    pos_b_hit = [pos_b for _, _, pos_b in rows if pos_b is not None]

    avg_pos_a = sum(pos_a_hit) / len(pos_a_hit) if pos_a_hit else 0.0
    avg_pos_b = sum(pos_b_hit) / len(pos_b_hit) if pos_b_hit else 0.0
    print(f"平均命中位置（仅命中题）   不重排 {avg_pos_a:.2f}   重排 {avg_pos_b:.2f}")

    group_stats = {}  # type -> [不重排命中数, 重排命中数, 总数]
    for item, pos_a, pos_b in rows:
        g = item["type"]
        stats = group_stats.setdefault(g, [0, 0, 0])
        stats[2] += 1
        if pos_a is not None:
            stats[0] += 1
        if pos_b is not None:
            stats[1] += 1

    print("\n按 type 分组 hit_rate：")
    for g, (hit_a, hit_b, total_g) in sorted(group_stats.items()):
        rate_a = hit_a / total_g if total_g else 0.0
        rate_b = hit_b / total_g if total_g else 0.0
        print(f"  {g:<11} 不重排 {hit_a}/{total_g} = {rate_a:.1%}   "
              f"重排 {hit_b}/{total_g} = {rate_b:.1%}   差值 {rate_b - rate_a:+.1%}")

    # 位置变化明细：不重排未命中 → 重排命中的题
    improve=[]
    regress=[]
    for item, pos_a, pos_b in rows:
        if pos_a is not None and pos_b is not None and pos_a>pos_b:
            improve.append((item, pos_a, pos_b))
        elif pos_a is not None and (pos_b is  None or pos_b < pos_a):
            regress.append((item, pos_a, pos_b))

    print(f"\n位置变化明细  提升 {len(improve)} 题   回退 {len(regress)} 题")

    if improve:
        print("  提升：")
        for item, pos_a, pos_b in improve:
            print(f"    {item['id']}  {pos_a}→{pos_b}  提升  [{item['type']}]")
    if regress:
        print("  回退：")
        for item, pos_a, pos_b in regress:
            pos_b_str = "-" if pos_b is None else str(pos_b)
            print(f"    {item['id']}  {pos_a}→{pos_b_str}  回退  [{item['type']}]")



if __name__ == "__main__":
    main()
