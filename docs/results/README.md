# 实验结果总览

本目录保存可提交到 GitHub 的实验结果文档。`experiments/` 目录仍作为本地运行输出目录，默认不提交；这里的 `docs/results/` 是整理后的结果归档。

原始报告：

- [FAISS 索引对比：Hybrid 召回](./faiss_index_hybrid_eval.md)
- [多路召回 + Chunk 参数召回评测](./metadata_retrieval_eval_batch.md)
- [接入 Metadata-aware Reranker 后的指标变化](./metadata_aware_rerank_final.md)

## 1. FAISS 索引对比

固定召回方案为 `metadata_retrieval_eval.py` 中的 Hybrid，评测 Flat / IVF / HNSW 三种 FAISS 索引。

实验口径：

```text
eval_set = eval_set_60.jsonl
chunks = all_cs1024_ov50.jsonl
recall_k = 10
rank_level = evidence
dense_global_top_k = 300
dense_candidate_k = 20
bm25_candidate_k = 20
hybrid_dense_weight = 0.5
hybrid_bm25_weight = 0.5
```

| 索引类型 | Recall@10 | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) |
|----------|-----------:|-------------:|------------:|-------------:|
| Flat | 0.7167 | 2533.19 | 0.13 | 237.46 |
| IVF | 0.6833 | 2515.61 | 69.61 | 241.79 |
| HNSW | 0.7333 | 2633.99 | 1.66 | 253.25 |

结论：

- HNSW 的证据级 Recall@10 最高，为 `0.7333`。
- Flat 构建最快，构建时间约 `0.13s`，Recall@10 为 `0.7167`。
- IVF 在当前参数下 Recall@10 下降到 `0.6833`，且训练时间明显更高，当前规模下不是最优选择。
- 当前查询延迟主要受 Hybrid 流程、metadata routing、query embedding 和 summary 候选处理影响，不只是 FAISS search 本身。

## 2. 多路召回 + 召回评测

该实验比较不同 chunk 参数下 Dense、BM25、Hybrid 三路召回，指标为证据级 Recall。

### 最终选用配置

最终链路使用：

```text
chunks = all_cs1024_ov50.jsonl
index = all_cs1024_ov50_flat.faiss
```

在该配置下：

| 召回方案 | Recall@3 | Recall@5 | Recall@10 |
|----------|---------:|---------:|----------:|
| Dense | 0.55 | 0.5667 | 0.6333 |
| BM25 | 0.6167 | 0.65 | 0.7333 |
| Hybrid | 0.6333 | 0.6667 | 0.7167 |


- Hybrid 在 Recall@3 和 Recall@5 上最好，说明前排稳定性更强。
- BM25 在 Recall@10 上略高于 Hybrid，说明关键词匹配对财报指标问题非常强。
- 最终选择 `all_cs1024_ov50`，因为它在 Recall@3/5、运行速度和后续 reranker 候选质量之间更均衡。

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

## 3. 接入 Reranker 后的指标变化

最终 rerank 策略：

```text
Hybrid Top20 -> Metadata-aware bge-reranker-v2-m3 -> Top5
```

summary 类型问题使用：

```text
Dense Top500 -> Local Hybrid Top20 -> Metadata-aware Rerank Top5
```

整体结果：

| Setting | Recall@5 | Top1 Acc | MRR@5 |
|---|---:|---:|---:|
| Hybrid Candidate Recall@20 | 0.85 | - | - |
| Hybrid Top5 | 0.6833 | 0.4333 | 0.5364 |
| Metadata-aware Rerank Top5 | 0.8333 | 0.70 | 0.7528 |
| Delta Rerank - Hybrid | +0.15 | +0.2667 | +0.2164 |

按问题类型：

| Question Type | Hybrid Recall@5 | Rerank Recall@5 | Recall Gain | Hybrid Top1 | Rerank Top1 | Top1 Gain |
|---|---:|---:|---:|---:|---:|---:|
| fact | 0.75 | 0.95 | +0.20 | 0.55 | 0.85 | +0.30 |
| compare | 0.55 | 0.70 | +0.15 | 0.30 | 0.55 | +0.25 |
| summary | 0.75 | 0.85 | +0.10 | 0.45 | 0.70 | +0.25 |

结论：

- Candidate Recall@20 为 `0.85`，说明 Hybrid 候选池里已经包含足够多可被 reranker 提升的证据。
- Metadata-aware Reranker 将 Overall Recall@5 从 `0.6833` 提升到 `0.8333`。
- Top1 Acc 从 `0.4333` 提升到 `0.70`，说明 reranker 不只是“捞进 Top5”，也显著改善了首位证据质量。
- compare 类型问题也有提升，但仍是最难类型，主要受两个目标文档都必须命中的约束影响。

