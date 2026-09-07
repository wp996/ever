import jieba
from rank_bm25 import BM25Okapi


def tokenize(text):
    tokens = jieba.lcut(text)
    return [w for w in tokens if w.strip() and w not in "，。！？：；、()（）"]


class BM25Retriever:
    def __init__(self):
        self.docs = []
        self.bm25 = None

    def build(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(doc) for doc in docs])

    def retrieve(self, query, top_k=5):
        scores = self.bm25.get_scores(tokenize(query))
        rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"chunk": self.docs[i], "score": scores[i]} for i in rank]


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    from vector_stores import VectorStoreService
    import config_data as config

    service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
    docs, _ = service.get_all_chunks()
    retriever = BM25Retriever()
    retriever.build(docs)
    for q in ["针织毛衣如何保养", "尺码怎么选", "颜色如何搭配"]:
        print(q, retriever.retrieve(q, top_k=3))