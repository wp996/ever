from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from hybrid_retriever import HybridRetriever
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from reranker import RerankerService

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        self.hybrid_retriever = HybridRetriever(embedding=self.embedding)


        self.reranker = RerankerService()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是服装电商的客服助手，只能依据下面的参考资料回答问题\n\n"
                           "规则:\n"
                           "1.每条结论后面必须标注来源，格式为 [来源：文件名]\n"
                           "2.参考资料中没有提到的内容，一律不要回答，也不要用常识或外部知识补充\n"
                           "3.如果参考资料完全不涉及用户问题，只回复这一句：参考资料中未涉及该问题。\n" 
                           "参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def _retrieve(self, query: str) -> list[Document]:
        mode=config.retrieval_mode
        use_rerank = config.rerank_enabled
        if not use_rerank:
            doc_score_list=self.hybrid_retriever.search_with_scores(
                query,
                mode=mode,
                top_k=config.retrieval_top_k,
            )
            docs=[]
            for doc,score in doc_score_list:
                new_meta={**doc.metadata,"original_score":score}
                new_doc=Document(
                    page_content=doc.page_content,
                    metadata=new_meta,
                )
                docs.append(new_doc)
            return docs
        else:
            doc_score_candidates=self.hybrid_retriever.search_with_scores(
                query,
                mode=mode,
                top_k=config.rerank_candidate_k
            )
            rerank_result:list[dict]=self.reranker.rerank(
                query=query,
                candidates=doc_score_candidates,
                top_n=config.rerank_top_n
            )
            docs=[]
            for r in rerank_result:
                new_meta={**r["metadata"],
                         "original_score": r["original_score"],
                          "rerank_score": r["rerank_score"]}
                new_doc=Document(
                    page_content=r["text"],
                    metadata=new_meta
                )
                docs.append(new_doc)
            return docs





    def __get_chain(self):
        """获取最终的执行链"""
        retriever=RunnableLambda(self._retrieve)

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for idx,doc in enumerate(docs,1):
                source=doc.metadata.get("source","未知来源")
                formatted_str+=f"[参考文档{idx}]文件名：{source}\n[文档内容]:{doc.page_content}\n\n"


            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template  | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "针织毛衣如何保养？"}, session_config)
    print(res)

