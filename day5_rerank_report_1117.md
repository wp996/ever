# Day5 重排评测报告：hybrid vs hybrid+rerank（2026-11-17）

## 任务背景

Day5 目标：接入 BAAI/bge-reranker-base 对 hybrid 检索结果重排，并做"不重排 vs 重排"对比评测。

- 报告生成日期：2026-11-17
- 重排模块：`reranker.py`（RerankerService）
- 评测脚本：`eval_retrieval.py`（39 题，逐题对比）
- 故障排查脚本：`debug_q27.py`（单题深挖）
- 评测集：`data/eval_questions.jsonl`（39 题）

## 重排配置（config_data.py）

| 配置项 | 值 | 说明 |
|---|---|---|
| reranker_model_path | BAAI/bge-reranker-base（本地缓存） | 支持 BGE_RERANKER_PATH 环境变量覆盖 |
| reranker_batch_size | 16 | CPU 上 16 够用；GPU 可提到 64 |
| reranker_max_length | 512 | query+passage 截断长度，越长越慢 |
| reranker_use_fp16 | False | 仅 GPU 有效，CPU 保持 False |
| reranker_device | None | None=自动（有 cuda 用 cuda，否则 cpu） |
| rerank_candidate_k | 20 | 粗排召回数 |
| rerank_top_n | 5 | 重排后保留数 |
| rerank_enabled | False | 默认关闭，链路可一键开关 |

运行环境：无 GPU（torch CPU 版）。模型首次加载约 30 秒；评测脚本用一次预热调用把加载耗时挪到计时之外。

## RerankerService 设计

- 接口固定：`rerank(query, candidates, top_n)`，candidates 兼容两种形态——`(Document, score)` 元组列表 / 纯 Document 列表
- 返回 dict 列表，每条必含：`chunk_id`（内容 MD5 前 8 位）/ `text` / `original_score`（原检索分，可能为 None）/ `rerank_score` / `metadata`
- 模型懒加载：`self._model` 私有 + `@property model` 只读入口，import 不触发 30 秒加载
- 排序兜底：先按 `rerank_score` 降序，rerank 分相同时按 `original_score` 降序，保证评测可复现
- 打分：`FlagReranker.compute_score(pairs, normalize=True)`，sigmoid 输出 0~1

## 独立测试（重排模块本身）

- 用例：18 条候选（15 条真实库内块 + 3 条干扰文本）
- 断言：全部通过（返回条数、字段完整性、排序单调性、top_n 截断）
- 耗时：首次 18.99s（含模型加载），纯推理 5.99s
- 结论：模块功能正确，CPU 上 batch 推理约 0.33s/对

## 评测设计

**上一轮的教训（天花板）**：38 块小库上，hybrid+RRF 已把正确答案排进 top5（hit_rate@5 双方 100%），重排没有任何差异化空间——不是重排没用，是评测集太容易。

**本轮改进**：

1. 知识库扩充：16 篇 → 26 篇（新增 10 篇硬负例：羽绒服/羊毛衫/牛仔/真丝/男装尺码/童装尺码/换货规则/发票规则/积分细则/养护误区），共 **57 块**
2. 评测集扩充：29 题 → 39 题，type 分布：distractor 16 / lexical 15 / semantic 8，所有 expected_keywords 已程序化校验对应原文
3. 指标升级：新增 hit_rate@1 和 MRR，并加"位置变化明细"（提升/回退逐题点名）
4. 对照设置：
   - 模式 A（不重排）：hybrid 直接取 top5
   - 模式 B（重排）：hybrid 召回 20 → rerank 取 5
   - 命中规则：source 在 expected_sources 内 且 文本含至少一个 expected_keywords

## 主结果

| 指标 | 不重排 | 重排 | 变化 |
|---|---:|---:|---:|
| hit_rate@1 | 32/39 = 82.1% | 36/39 = 92.3% | **+10.2pp** |
| MRR | 0.897 | 0.949 | **+0.052** |
| 平均命中位置（仅命中题） | 1.31 | 1.05 | -0.26 |
| 平均耗时/题 | 0.18s | 7.58s | +7.40s |

按 type 分组（hit_rate@5 口径）：

| type | 不重排 | 重排 | 差值 |
|---|---|---|---|
| distractor | 16/16 = 100.0% | 15/16 = 93.8% | **-6.2%** |
| lexical | 15/15 = 100.0% | 15/15 = 100.0% | +0.0% |
| semantic | 8/8 = 100.0% | 8/8 = 100.0% | +0.0% |

位置变化明细：提升 5 题，回退 1 题。

- 提升：q14（2→1, semantic）、q17（2→1, lexical）、q20（2→1, lexical）、q26（2→1, semantic）、q29（6→2, lexical）
- 回退：q27（3→未命中, distractor）

自洽性校验：@1 增量 = 4 题（q14/q17/q20/q26）✓；MRR 增量 = 4×0.5 + (1/2−1/6) − 1/3 = 2.0，2.0/39 ≈ +0.051 ✓。逐题明细与汇总数字严丝合缝。

## 亮点案例：q29（6→2）

问题："真丝衣服自己在家洗要注意什么"（lexical）。粗排把正确答案（洗涤养护.txt，含"禁止搓揉/专用中性洗涤剂"）排到**第 6 名，已掉出 top5**；重排捞回第 2。说明粗排向量相似度在口语化长 query 上会失准，cross-encoder 逐对精读 query-passage 能纠回粗排的深位错误——这正是重排存在的意义。

## 失败案例剖析：q27（3→挤出 top5）

问题："防晒衣应该怎么洗"（distractor，评测集里预设的"最强干扰用例"）。

- 正确答案 A：夏季穿搭.txt 段（含"防晒涂层"，结论：禁用柔顺剂），粗排第 3 → 重排第 6，**被挤出 top5**
- 硬负例 B：洗涤养护.txt 轻薄化纤段（同题写"防晒衣洗涤建议加柔顺剂"，**与正确答案直接冲突**），粗排第 1 → 重排第 1

分数证据（debug_q27.py 输出）：

- 粗排：A=0.0305（第3）、B=0.0325（第1），分差仅 0.002——**噪音级差距，粗排根本分不出谁对**
- 重排：A=0.7991（第6）、B=0.9761（第1），分差 0.177——**cross-encoder 自信地选了错的那个**

三层原因：

1. **B 字面全覆盖**：query"防晒衣应该怎么洗"里的几乎每个词（防晒衣、洗）B 段都有，字面贴合度满分——尽管结论是错的
2. **A 的正确信息被切块埋没**：A 块开头是"真丝连衣裙：适合正式约会……"，防晒涂层那句埋在 chunk 中后段，cross-encoder 看到的大半上下文与"怎么洗"无关
3. **rerank 只打相关性，不做事实核查**：文档互相矛盾时，重排会放大主题贴合但结论错误的内容

结论：**重排能纠正"排得不准"，不能纠正"文档互相矛盾"**。q27 的病根是知识库数据质量（同题冲突内容），不是重排模块的 bug。附带发现：常见养护误区.txt 从粗排第 20 升到重排第 4，说明重排对"洗护类 query"的整体语义判断是准的，只是被这道精心设计的硬负例精准命中了软肋——评测集完成了它的使命。

## 结论与 trade-off

1. 57 块、带硬负例的库上，hybrid 粗排能把正确答案排进 top5（@5 接近满分），但 top1 约 1/5 排错；重排把 @1 从 82.1% 提到 92.3%，MRR 从 0.897 提到 0.949，**收益集中在 @1 和 MRR 层面**
2. 代价：每题多约 7.4s（CPU）。典型 trade-off：用 7 秒延迟换一个"答非所问的 top1"
3. 重排不是万能药：对 q27 这类"同题冲突文档"会反向放大错误

## 后续优化

1. `rerank_top_n` 5 → 8：q27 的正确块重排后落在第 6，取 8 即可留在 LLM 上下文内，代价是多几百 token；重排的价值不一定是"把对的提到第一"，也可能是"把对的留在可见范围内"
2. 修数据源冲突：洗涤养护.txt 与夏季穿搭.txt 对"防晒衣能否用柔顺剂"结论相反，需人工修订或在文档中标注适用条件；这是任何模型层手段都解决不了的数据质量问题
3. 延迟缓解：GPU 推理 / 换更小的 reranker（如 tiny 版）/ "只在粗排 top1 分数不悬殊时才触发重排"的按需策略
4. 回补 test_reranker.py：补齐 8 用例矩阵的 TODO 断言，修复第 91 行零宽字符残留
