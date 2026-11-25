import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import hashlib
import config_data as config
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

def make_chunk_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]


class RerankerService:
    def __init__(self,model_path=None):
        self._model=None
        self.model_path = model_path or config.reranker_model_path
    @property
    def model(self):
        if self._model is None:
            self._model = FlagReranker(self.model_path,use_fp16=config.reranker_use_fp16)
        return self._model
    def rerank(self,query,candidates,top_n=None):
        if not candidates :
            return []
        items=[]
        for item in candidates:
            if isinstance(item,tuple):
                doc,original_score=item
            else:
                doc,original_score=item,None
            items.append((doc,original_score))

        pairs = [[query, doc.page_content] for doc,_ in items]
        scores=self.model.compute_score(pairs,normalize=True,batch_size=config.reranker_batch_size, max_length=config.reranker_max_length)
        records=[]
        for (doc,original_score),score in zip(items,scores):
            records.append({
                "chunk_id": make_chunk_id(doc.page_content),
                "text": doc.page_content,
                "original_score": original_score,
                "rerank_score": float(score),
                "metadata": doc.metadata,
            })
        records.sort(key=lambda x: (-x["rerank_score"], -(x["original_score"] or 0.0)))

        limit = config.rerank_top_n if top_n is None else top_n
        return records[:limit]

if __name__ == "__main__":
    docs=[
        Document(page_content="羊毛衫建议冷水手洗，使用中性洗涤剂", metadata={"source": "洗涤养护.txt"}),
        Document(page_content="毛衣起球是纤维摩擦后的正常现象", metadata={"source": "冬季选购养护.txt"}),
        Document(page_content="退货需要在签收后7天内申请", metadata={"source": "退还货政策.txt"}),
        Document(page_content="今天天气不错，适合出去散步", metadata={"source": "冬季选购养护.txt"})
    ]
    svc=RerankerService()
    out=svc.rerank("毛衣怎么洗", docs, top_n=2)
    for r in out:
        print(r["chunk_id"], round(r["rerank_score"], 4), r["text"][:12])