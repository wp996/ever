# 数据摸底

## 1. 基本信息
- 日期：2026-09-02
- 项目：智能客服 RAG
- 数据来源：洗涤养护.txt / 颜色选择.txt / 尺码推荐.txt

## 2. 切块与入库
- Splitter：RecursiveCharacterTextSplitter
- chunk_size：1000
- chunk_overlap：100
- separators：\n\n、\n、.、!、?、。、！、？、空格、空串
- max_split_char_number：1000（小于等于 1000 字不切块，整篇入库）
- Embedding：text-embedding-v4，1024 维
- 向量库：Chroma，集合名 rag，持久化目录 ./chroma_db
- 去重：check_md5，对整篇文本计算 MD5

## 3. 数据统计
- 总块数：6
- 块长：平均 646.3 / 中位 923 / 最短 263 / 最长 997
- 来源分布：洗涤养护.txt 3 块 / 颜色选择.txt 2 块 / 尺码推荐.txt 1 块
- 完全重复块：0

## 4. 发现
1. 知识库只有 3 篇文档、6 个块，量太小，无法支撑有效评测，需扩充到 20-50 篇
2. 当前检索 top-k=1（config.similarity_threshold 实为 k 值），每轮只取 1 个块，召回严重不足
3. 小于 1000 字的文档不切块，尺码推荐.txt 整篇入库
4. 仅支持 .txt，不支持 PDF/Word
5. Prompt 无“不知道就直说”和“引用来源”要求，有幻觉风险
6. 所有用户共用 user_001 会话，历史文件无上限
7. rag.py 中 print_prompt 调试代码未清理
8. metadata 只有 source/create_time/operator，缺 page/heading/chunk_index；operator 写死为“小曹”

## 5. 待办
- [ ] 扩充知识库（退换货 / 物流 / 优惠券 / 支付等主题）
- [ ] top-k 提升到 5，并改名 retriever_top_k
- [ ] 加 BM25（jieba 分词）与混合检索
- [ ] 加重排
- [ ] 接入 RAGAS 评测
- [ ] Prompt 加拒答与引用要求
- [ ] 支持 PDF/Word 解析
- [ ] 多用户会话隔离与历史压缩
- [ ] metadata 补全，移除调试打印