
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


md5_path = "./md5.text"


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# spliter
chunk_size = 400
chunk_overlap = 60
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 0        # 文本分割的阈值

#
similarity_threshold = 1            # 检索返回匹配的文档数量

# retrieval modes
retrieval_mode = "vector"           # vector / bm25 / hybrid
retrieval_top_k = 5
rrf_k = 60

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen-plus"        # 原来写 qwen3-max 会报 Model not exist；可用：qwen-max / qwen-plus

# ---------- reranker（交叉编码器重排） ----------
# 默认 bge-reranker-base：CPU 可跑、约 1.1GB；
# 有 GPU 再换 bge-reranker-v2-m3（multilingual，效果更好但更大更慢）。
# 模型路径优先读环境变量 BGE_RERANKER_PATH，方便换本地目录而不改代码。
reranker_model_path = os.getenv("BGE_RERANKER_PATH", "BAAI/bge-reranker-base")
reranker_batch_size = 16             # CPU 上 16 够用；GPU 可提到 64
reranker_max_length = 512            # query+passage 截断长度，越长越慢
reranker_use_fp16 = False            # 仅 GPU 有效，CPU 保持 False
reranker_device = None               # None=自动（有 cuda 用 cuda，否则 cpu）

# hybrid 候选池与最终条数：先召回 20~30 条，重排后取 5~8 条
rerank_candidate_k = 20              # hybrid 粗排召回条数
rerank_top_n = 5                     # 重排后保留条数
rerank_enabled = False               # 总开关；单次调用仍可用 rerank=True/False 覆盖

# ---------- 评测 ----------
eval_top_k = 5                       # 评测时每个问题看前几条

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
