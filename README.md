# Financial-RAG

FinRAG 是一个面向金融年报 / 研报的 RAG 项目

```text
PDF 解析 -> 用 PyMuPDF 提取文本，用 pdfplumber 解析表格，处理表格、多栏/页眉页脚 -> 表格保护切块
-> FAISS 向量索引 -> 三路召回：Dense/BM25/Hybrid 召回
-> bge-reranker-v2-m3 重排 -> Qwen3-8B 证据绑定生成，设定约束：低置信度拒答,生成后引用校验
-> 检索 / 重排 / 生成评测 -> Streamlit Demo
```

最终重排链路定为：

```text
Hybrid Top20 -> bge-reranker-v2-m3 -> Top5
```

## 项目结构

```text
FinRAG/
├── data/
│   ├── pdfs/              # 原始 PDF，自行放入
│   ├── parsed/            # PDF 解析后的 Markdown
│   ├── chunks/            # 切块结果
│   ├── indexes/           # FAISS index 和 metadata
│   └── eval_set/          # 小规模评测集，
├── experiments/           # 评测报告输出
├── src/
│   ├── parsing/           # PDF 解析
│   ├── chunking/          # 表格保护切块
│   ├── indexing/          # FAISS 建索引
│   ├── retrieval/         # Dense/BM25/Hybrid 召回
│   ├── rerank/            # reranker
│   ├── generation/        # Qwen3-8B 证据绑定生成
│   ├── evaluation/        # 召回 / 重排评测
│   └── demo/              # Streamlit 演示
├── requirements.txt
└── README.md
```

## 环境

建议 Python 3.10+

```bash
pip install -r requirements.txt
```

本项目默认模型路径是实验服务器上的本地 HuggingFace cache，可在命令行参数中覆盖：

## 1. PDF 解析

用 PyMuPDF 提取文本，用 pdfplumber 解析表格，处理表格、多栏/页眉页脚

输出为 Markdown，表格会尽量保留为 `[TABLE_START] ... [TABLE_END]`。

## 2. 表格保护切块

选定最终参数为 `chunk_size=1024, overlap=50`，输出 `all_cs1024_ov50.jsonl`：(chunk切块对比的recall在后续)
### Chunk 参数对比摘要
| Chunk 文件 | chunk_size | overlap | Hybrid Recall@3 | Hybrid Recall@5 | Hybrid Recall@10 | elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|
| all_cs512_ov200 | 512 | 200 | 0.55 | 0.60 | 0.7167 | 242.25 |
| all_cs256_ov100 | 256 | 100 | 0.50 | 0.5833 | 0.70 | 306.98 |
| all_cs256_ov200 | 256 | 200 | 0.5167 | 0.60 | 0.6333 | 556.47 |
| all_cs256_ov50 | 256 | 50 | 0.55 | 0.5667 | 0.65 | 236.38 |
| all_cs512_ov100 | 512 | 100 | 0.5333 | 0.6167 | 0.75 | 184.08 |
| all_cs512_ov50 | 512 | 50 | 0.4833 | 0.5167 | 0.60 | 169.12 |
| all_cs1024_ov100 | 1024 | 100 | 0.60 | 0.6667 | 0.7167 | 150.48 |
| all_cs1024_ov50 | 1024 | 50 | 0.6833 | 0.7167 | 0.7502 | 144.76 |
| all_cs1024_ov200 | 1024 | 200 | 0.60 | 0.65 | 0.7167 | 161.79 |
## 3. FAISS 建索引

对最终 chunk 文件 all_cs1024_ov50.jsonl 构建 Flat index

## 4. 基础召回评测
构建评测集：
   - 构建了60条 `(query, ground_truth_doc_id, answer)` 三元组
   - query 分三类：事实型（"宁德时代2025年研发费用是多少？"）、对比型（"宁德时代和比亚迪2025年谁的营业收入更高？"）、汇总型（"十四五”科技创新规划提出了哪些主要量化目标？"）

评测 Dense、BM25、Hybrid 三路召回：
| chunk 文件 | chunk_size | overlap | 召回方案 | Recall@3 | Recall@5 | Recall@10 | elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_cs1024_ov50.jsonl` | 1024 | 50 | 纯向量 | 0.55 | 0.5667 | 0.6333 | 144.76 |
| `all_cs1024_ov50.jsonl` | 1024 | 50 | 纯 BM25 | 0.6333 | 0.65 | 0.7333 | 144.76 |
| `all_cs1024_ov50.jsonl` | 1024 | 50 | 混合召回 | 0.6833 | 0.7167 | 0.7502 | 144.76 |

## 5. FAISS 索引对比实验

固定召回方案为 Dense与BM25混合召回

| 索引类型 | Recall@10 | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) | P50(ms) | P95(ms) |
|----------|-----------:|-------------:|------------:|-------------:|--------:|--------:|
| Flat | 0.7167 | 2533.19 | 0.13 | 237.46 | 865.54 | 1969.57 |
| IVF | 0.6833 | 2515.61 | 69.61 | 241.79 | 860.43 | 1932.65 |
| HNSW | 0.7333 | 2633.99 | 1.66 | 253.25 | 872.48 | 1933.53 |

说明：
- 综合各项指标，选定采用Flat索引。
- 查询延迟统计单条 query 的 Hybrid 召回评测耗时，不包含模型和索引加载时间。
- 构建时间统计 FAISS index 训练 / add 向量耗时，不包含 embedding 生成时间。
- 内存占用使用 FAISS index 序列化大小估算，便于 Flat / IVF / HNSW 横向比较。

## 6. 最终重排评测

在当前 60 条评测集上的最终结果：

| Setting | Recall@5 | Top1 Acc | MRR@5 |
|---|---:|---:|---:|
| Hybrid Candidate Recall@20 | 0.85 | - | - |
| Hybrid Top5 | 0.6833 | 0.4333 | 0.5364 |
| Rerank Top5 | 0.8333 | 0.7 | 0.7528 |
| Delta Rerank - Hybrid | 0.15 | 0.2667 | 0.2164 |

## 7. Qwen3 证据绑定生成

单 query 运行：

```bash
CUDA_VISIBLE_DEVICES=0,7 python src/generation/qwen3_evidence_generator.py \
  --query "宁德时代2025年营业收入是多少？" \
  --max_new_tokens 256 \
  --reranker_batch_size 8 \
  --slow_tokenizer
```

生成层的幻觉缓解策略：

```text
证据绑定 + 低置信度拒答 + 生成后引用校验
```

具体实现：

- 证据绑定：Prompt 中只给 TopK 证据，并要求答案必须引用 `[E1]` 这样的证据编号。
- 低置信度拒答：如果所有 reranker score 都低于阈值，直接返回 `我不确定`。
- 生成后引用校验：生成完成后检查答案里的引用编号是否存在；没有有效引用则拒答。

## 8. Streamlit Demo

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
streamlit run src/demo/streamlit_finrag_demo.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Demo 页面包含：

- 输入 query
- 显示召回文档 / 证据段落
- 显示最终回答
- 显示引用来源和 Citation Check
- 显示是否触发低置信度拒答

## GitHub 说明
实验结果归档在 [docs/results](./docs/results/README.md)，包括：

- FAISS 索引对比
- 多路召回 + 召回评测
- 接入 reranker 后的 recall / Top1 Acc / MRR 变化
