# FinRAG

FinRAG 是一个面向金融年报 / 研报的 RAG 项目，当前整理版只保留实际使用到的链路代码：

```text
PDF 解析 -> 表格保护切块 -> FAISS 向量索引 -> Metadata Routing Hybrid 召回
        -> Metadata-aware bge-reranker-v2-m3 重排 -> Qwen3-8B 证据绑定生成
        -> 检索 / 重排 / 生成评测 -> Streamlit Demo
```

最终重排链路定为：

```text
Hybrid Top20 -> Metadata-aware bge-reranker-v2-m3 -> Top5
```

summary 类型问题使用单独粗筛：

```text
Dense Top500 -> Local Hybrid Top20 -> Metadata-aware Rerank Top5
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

把原始 PDF 放到 `data/pdfs/`，然后运行：

```bash
python src/parsing/hybrid_pdf_parser.py \
  --input_dir data/pdfs \
  --output_dir data/parsed
```

输出为 Markdown，表格会尽量保留为 `[TABLE_START] ... [TABLE_END]`。

## 2. 表格保护切块

默认最终参数为 `chunk_size=1024, overlap=50`，输出 `all_cs1024_ov50.jsonl`：

```bash
python src/chunking/table_aware_chunker.py \
  --input_dir data/parsed \
  --output_dir data/chunks \
  --chunk_size 1024 \
  --overlap 50
```

如果要复现实验中的 3x3 切块对比：

```bash
python src/chunking/table_aware_chunker.py --run_experiments
```

## 3. FAISS 建索引

对最终 chunk 文件建 Flat index：

```bash
python src/indexing/build_faiss.py \
  --chunks_file data/chunks/all_cs1024_ov50.jsonl \
  --output_dir data/indexes \
  --index_type flat
```

会生成类似：

```text
data/indexes/all_cs1024_ov50_flat.faiss
data/indexes/all_cs1024_ov50_flat_metadata.jsonl
```

## 4. 基础召回评测

评测 Metadata Routing 下 Dense、BM25、Hybrid 三路召回：

```bash
python src/evaluation/metadata_retrieval_eval.py \
  --eval_file data/eval_set/eval_set_60.jsonl \
  --chunks_file data/chunks/all_cs1024_ov50.jsonl \
  --faiss_index data/indexes/all_cs1024_ov50_flat.faiss \
  --faiss_metadata data/indexes/all_cs1024_ov50_flat_metadata.jsonl
```

## 5. FAISS 索引对比实验

固定召回方案为 `metadata_retrieval_eval.py` 里的 Hybrid，只比较 FAISS 索引类型：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/evaluation/faiss_index_hybrid_eval.py \
  --chunks_file data/chunks/all_cs1024_ov50.jsonl \
  --eval_file data/eval_set/eval_set_60.jsonl \
  --output_dir data/indexes/faiss_compare
```

默认实验口径与基础召回脚本对齐：

```text
dense_candidate_k = 20
bm25_candidate_k = 20
dense_global_top_k = 300
summary_seed_k = 300
fact_match_mode = all
summary_match_mode = any
summary_min_hits = 1
```

输出报告：

```text
experiments/faiss_index_hybrid_eval.md
experiments/faiss_index_hybrid_eval_details.json
```

报告表格：

| 索引类型 | Recall@10 | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) |
|----------|-----------|-------------|------------|-------------|
| Flat | - | - | - | - |
| IVF | - | - | - | - |
| HNSW | - | - | - | - |

说明：

- 查询延迟统计单条 query 的 Hybrid 召回评测耗时，不包含模型和索引加载时间。
- 构建时间统计 FAISS index 训练 / add 向量耗时，不包含 embedding 生成时间。
- 内存占用使用 FAISS index 序列化大小估算，便于 Flat / IVF / HNSW 横向比较。

## 6. 最终重排评测

最终版重排脚本：

```bash
python src/evaluation/metadata_aware_rerank_final.py \
  --eval_file data/eval_set/eval_set_60.jsonl \
  --chunks_file data/chunks/all_cs1024_ov50.jsonl \
  --faiss_index data/indexes/all_cs1024_ov50_flat.faiss \
  --faiss_metadata data/indexes/all_cs1024_ov50_flat_metadata.jsonl \
  --candidate_k 20 \
  --final_k 5
```

在当前 60 条评测集上的最终结果：

| Setting | Recall@5 | Top1 Acc | MRR@5 |
|---|---:|---:|---:|
| Hybrid Candidate Recall@20 | 0.8500 | - | - |
| Hybrid Top5 | 0.6833 | 0.4333 | 0.5364 |
| Metadata-aware Rerank Top5 | 0.8333 | 0.7000 | 0.7528 |

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
CUDA_VISIBLE_DEVICES=0,7 \
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
