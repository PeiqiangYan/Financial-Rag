#!/usr/bin/env python3
"""
FAISS 向量索引构建脚本 - 支持批量处理

对应项目计划第 1 周：
1. 读取 data/chunks/ 下所有或指定的 jsonl 文件
2. 使用 FlagEmbedding 的 bge-large-zh-v1.5 生成 embedding
3. 使用 FAISS 构建向量索引，支持 Flat/IVF/HNSW
4. 保存：
   - FAISS index 文件
   - chunk metadata 文件
   - indexing 实验记录

重要说明：
- embedding 只使用 chunk["content"]
- metadata 只用于后续过滤、展示、评测，不参与向量化

批量处理示例：
python src/indexing/build_faiss.py --batch

指定多个文件：
python src/indexing/build_faiss.py \
  --chunks_file file1.jsonl file2.jsonl file3.jsonl

指定目录：
python src/indexing/build_faiss.py \
  --chunks_dir data/chunks

跳过已处理的文件（默认）：
python build_faiss.py --batch

强制重新构建所有索引：
python build_faiss.py --batch --force_rebuild
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from FlagEmbedding import FlagModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================
# JSONL 读取
# ============================================

def read_chunks(chunks_file: Path) -> List[Dict]:
    """读取 chunks jsonl"""
    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks 文件不存在: {chunks_file}")

    chunks = []

    with chunks_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {e}")

            if "chunk_id" not in item:
                raise ValueError(f"第 {line_no} 行缺少 chunk_id")

            if "doc_id" not in item:
                raise ValueError(f"第 {line_no} 行缺少 doc_id")

            if "content" not in item:
                raise ValueError(f"第 {line_no} 行缺少 content")

            if not item["content"].strip():
                continue

            chunks.append(item)

    if not chunks:
        raise ValueError(f"没有读取到有效 chunk: {chunks_file}")

    return chunks


def build_texts(chunks: List[Dict]) -> List[str]:
    """
    构造用于 embedding 的文本。

    当前阶段明确只使用 content，不拼接 metadata。
    """
    texts = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        texts.append(content)

    return texts


# ============================================
# Embedding
# ============================================

def load_embedding_model(
    model_name_or_path: str,
    use_fp16: bool = True,
) -> FlagModel:
    """加载 bge embedding 模型"""
    print(f"加载 embedding 模型: {model_name_or_path}")
    
    # 设置离线模式，避免网络超时
    import os
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    model = FlagModel(
        model_name_or_path,
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=use_fp16,
    )

    return model


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """手动 L2 normalize，兼容 FlagEmbedding==1.2.11"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def encode_texts(
    model: FlagModel,
    texts: List[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """批量生成 embeddings"""
    print(f"开始生成 embedding，文本数量: {len(texts)}")
    start_time = time.time()

    # FlagEmbedding==1.2.11 的 FlagModel.encode 不支持 normalize_embeddings 参数
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    if normalize_embeddings:
        embeddings = l2_normalize(embeddings)

    elapsed = time.time() - start_time
    print(f"embedding 完成，耗时: {elapsed:.2f}s")
    print(f"embedding shape: {embeddings.shape}")

    return embeddings


# ============================================
# FAISS Index - 优化版本
# ============================================

def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """
    构建 Flat 索引（精确检索）。
    
    因为 embedding 已经 normalize，所以使用 Inner Product 等价于 cosine similarity。
    """
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"  - Flat 索引，向量数: {index.ntotal}, 维度: {dim}")

    return index


def build_ivf_index(
    embeddings: np.ndarray,
    nlist: Optional[int] = None,
    nprobe: int = 10,
) -> faiss.Index:
    """
    构建 IVF 索引（倒排索引）。
    
    参数:
        nlist: 聚类中心数，默认根据向量数自动设置（sqrt(N) 或 4*sqrt(N)）
        nprobe: 检索时探测的聚类数，默认 10
    """
    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    # 自动设置 nlist
    if nlist is None:
        nlist = int(4 * np.sqrt(n_vectors))  # 经验值：4 * sqrt(N)
        nlist = max(10, min(nlist, 1000))    # 限制范围 10-1000
    
    print(f"  - IVF 参数: nlist={nlist}, nprobe={nprobe} (检索时)")
    
    # 使用 Inner Product 作为距离度量
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    
    # 训练索引
    print(f"  - 训练 IVF 索引...")
    index.train(embeddings)
    
    # 添加向量
    index.add(embeddings)
    
    # 设置检索参数
    index.nprobe = nprobe
    
    print(f"  - IVF 索引构建完成，向量数: {index.ntotal}")

    return index


def build_hnsw_index(
    embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> faiss.Index:
    """
    构建 HNSW 索引（分层可导航小世界图）。
    
    参数:
        m: 每个节点的最大连接数，默认 32（平衡速度和精度）
        ef_construction: 构建时的动态候选列表大小，默认 200（越大构建越慢但精度越高）
        ef_search: 检索时的动态候选列表大小，默认 50（越大检索越慢但召回越高）
    """
    dim = embeddings.shape[1]
    
    print(f"  - HNSW 参数: m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
    
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    
    index.add(embeddings)
    
    print(f"  - HNSW 索引构建完成，向量数: {index.ntotal}")

    return index


def build_ivfpq_index(
    embeddings: np.ndarray,
    nlist: Optional[int] = None,
    m: int = 8,
    nbits: int = 8,
    nprobe: int = 10,
) -> faiss.Index:
    """
    构建 IVF-PQ 索引（倒排索引 + 乘积量化），进一步压缩内存。
    
    适用于大规模向量（>100万）。
    
    参数:
        nlist: 聚类中心数
        m: 子向量数，需要能被向量维度整除
        nbits: 每个子向量的编码位数（2^nbits 个中心）
        nprobe: 检索时探测的聚类数
    """
    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    # 自动设置 nlist
    if nlist is None:
        nlist = int(4 * np.sqrt(n_vectors))
        nlist = max(10, min(nlist, 1000))
    
    # 确保 m 能整除 dim
    if dim % m != 0:
        # 找到最接近的能整除 dim 的 m
        original_m = m
        for m_candidate in [8, 16, 32, 64]:
            if dim % m_candidate == 0:
                m = m_candidate
                break
        print(f"  - 警告: {original_m} 不能整除 {dim}，自动调整为 {m}")
    
    print(f"  - IVF-PQ 参数: nlist={nlist}, m={m}, nbits={nbits}, nprobe={nprobe}")
    
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlist,
        m,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    
    # 训练索引
    print(f"  - 训练 IVF-PQ 索引...")
    index.train(embeddings)
    
    # 添加向量
    index.add(embeddings)
    
    # 设置检索参数
    index.nprobe = nprobe
    
    print(f"  - IVF-PQ 索引构建完成，向量数: {index.ntotal}")

    return index


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    **kwargs,
) -> faiss.Index:
    """根据 index_type 构建 FAISS 索引"""
    index_type = index_type.lower()

    print(f"开始构建 FAISS 索引，类型: {index_type}")
    start_time = time.time()

    if index_type == "flat":
        index = build_flat_index(embeddings)
    elif index_type == "ivf":
        nlist = kwargs.get('nlist', None)
        nprobe = kwargs.get('nprobe', 10)
        index = build_ivf_index(embeddings, nlist=nlist, nprobe=nprobe)
    elif index_type == "hnsw":
        m = kwargs.get('m', 32)
        ef_construction = kwargs.get('ef_construction', 200)
        ef_search = kwargs.get('ef_search', 50)
        index = build_hnsw_index(
            embeddings, 
            m=m, 
            ef_construction=ef_construction,
            ef_search=ef_search
        )
    elif index_type == "ivfpq":
        nlist = kwargs.get('nlist', None)
        m = kwargs.get('pq_m', 8)
        nbits = kwargs.get('pq_nbits', 8)
        nprobe = kwargs.get('nprobe', 10)
        index = build_ivfpq_index(
            embeddings,
            nlist=nlist,
            m=m,
            nbits=nbits,
            nprobe=nprobe
        )
    else:
        raise ValueError(f"不支持的 index_type: {index_type}")

    elapsed = time.time() - start_time

    print(f"FAISS 索引构建完成，耗时: {elapsed:.2f}s")
    print(f"索引向量数: {index.ntotal}")

    return index


# ============================================
# 保存文件
# ============================================

def save_index(index: faiss.Index, output_file: Path) -> None:
    """保存 FAISS index"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_file))
    print(f"FAISS index 已保存: {output_file}")


def save_metadata(chunks: List[Dict], output_file: Path) -> None:
    """
    保存 metadata。

    metadata 的顺序必须和 FAISS index 中向量顺序一致。
    检索时 faiss 返回的 idx 可以直接用 metadata[idx] 找回 chunk。

    本版新增保存：
    - doc_title
    - doc_type
    - source_type
    - company
    - company_short
    - year
    - period
    - source_file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            meta = {
                "faiss_id": idx,

                # 基础 chunk 信息
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "chunk_type": chunk.get("chunk_type"),
                "char_len": chunk.get("char_len"),
                "source_segment_index": chunk.get("source_segment_index"),
                "chunk_size": chunk.get("chunk_size"),
                "overlap": chunk.get("overlap"),

                # 文档级 metadata
                "doc_title": chunk.get("doc_title", ""),
                "doc_type": chunk.get("doc_type", "unknown"),
                "source_type": chunk.get("source_type", "unknown"),
                "company": chunk.get("company", ""),
                "company_short": chunk.get("company_short", ""),
                "year": chunk.get("year", ""),
                "period": chunk.get("period", ""),
                "source_file": chunk.get("source_file", ""),

                # 原始内容，用于展示和证据引用
                "content": chunk.get("content"),
            }

            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"metadata 已保存: {output_file}")


def write_index_report(
    report_dir: Path,
    chunks_file: Path,
    index_file: Path,
    metadata_file: Path,
    model_name_or_path: str,
    index_type: str,
    index_params: Dict,
    chunks: List[Dict],
    embeddings: np.ndarray,
    embed_time: float,
    index_time: float,
) -> None:
    """写入索引构建实验记录"""
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个 chunks 文件生成独立的报告
    stem = chunks_file.stem
    report_file = report_dir / f"indexing_{stem}_{index_type}.md"

    text_chunks = [c for c in chunks if c.get("chunk_type") == "text"]
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]

    doc_ids = sorted(set(c.get("doc_id", "") for c in chunks if c.get("doc_id")))
    doc_types = {}
    source_types = {}

    for c in chunks:
        doc_type = c.get("doc_type", "unknown")
        source_type = c.get("source_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        source_types[source_type] = source_types.get(source_type, 0) + 1

    lines = []
    lines.append(f"# FAISS 索引构建记录 - {stem}")
    lines.append("")
    lines.append("## 输入输出")
    lines.append("")
    lines.append(f"- chunks_file: `{chunks_file}`")
    lines.append(f"- index_file: `{index_file}`")
    lines.append(f"- metadata_file: `{metadata_file}`")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append(f"- embedding_model: `{model_name_or_path}`")
    lines.append(f"- index_type: `{index_type}`")
    if index_params:
        lines.append(f"- index_params: `{index_params}`")
    lines.append(f"- embedding_input: `content only`")
    lines.append(f"- normalize_embeddings: `True`")
    lines.append(f"- similarity: `Inner Product / Cosine Similarity`")
    lines.append("")
    lines.append("## 数据统计")
    lines.append("")
    lines.append(f"- total_chunks: {len(chunks)}")
    lines.append(f"- text_chunks: {len(text_chunks)}")
    lines.append(f"- table_chunks: {len(table_chunks)}")
    lines.append(f"- doc_count: {len(doc_ids)}")
    lines.append(f"- embedding_dim: {embeddings.shape[1]}")
    lines.append("")
    lines.append("## 文档类型分布")
    lines.append("")
    lines.append("| doc_type | chunk_count |")
    lines.append("|---|---:|")
    for k, v in sorted(doc_types.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 来源类型分布")
    lines.append("")
    lines.append("| source_type | chunk_count |")
    lines.append("|---|---:|")
    for k, v in sorted(source_types.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 耗时")
    lines.append("")
    lines.append(f"- embedding_time_sec: {embed_time:.2f}")
    lines.append(f"- index_build_time_sec: {index_time:.2f}")
    lines.append(f"- total_time_sec: {embed_time + index_time:.2f}")
    lines.append("")
    lines.append("## 索引信息")
    lines.append("")
    lines.append(f"- index_size_mb: {index_file.stat().st_size / 1024 / 1024:.2f}")
    lines.append("")
    lines.append("## 召回评测（待补充）")
    lines.append("")
    lines.append("| index_type | Recall@3 | Recall@5 | Recall@10 | 查询延迟(ms) | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    lines.append(f"| {index_type} | ? | ? | ? | ? | - |")
    lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"索引构建报告已保存: {report_file}")


# ============================================
# 核心构建流程
# ============================================

def build_index_pipeline(
    chunks_file: Path,
    output_dir: Path,
    experiment_dir: Path,
    model_name_or_path: str,
    index_type: str,
    batch_size: int,
    use_fp16: bool,
    index_params: Optional[Dict] = None,
) -> None:
    """完整构建流程 - 处理单个文件"""
    
    print(f"\n{'='*80}")
    print(f"处理文件: {chunks_file.name}")
    print(f"{'='*80}")
    
    # 读取数据
    chunks = read_chunks(chunks_file)
    texts = build_texts(chunks)

    print(f"chunks 数量: {len(chunks)}")
    print(f"index_type: {index_type}")
    print(f"embedding_model: {model_name_or_path}")
    print("embedding_input: content only")

    # 加载模型
    model = load_embedding_model(
        model_name_or_path=model_name_or_path,
        use_fp16=use_fp16,
    )

    # 生成 embeddings
    embed_start = time.time()
    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    embed_time = time.time() - embed_start

    # 构建索引
    index_params = index_params or {}
    index_start = time.time()
    index = build_faiss_index(
        embeddings=embeddings,
        index_type=index_type,
        **index_params,
    )
    index_time = time.time() - index_start

    # 保存文件
    stem = chunks_file.stem
    index_file = output_dir / f"{stem}_{index_type}.faiss"
    metadata_file = output_dir / f"{stem}_{index_type}_metadata.jsonl"
    
    save_index(index, index_file)
    save_metadata(chunks, metadata_file)

    # 保存报告
    write_index_report(
        report_dir=experiment_dir,
        chunks_file=chunks_file,
        index_file=index_file,
        metadata_file=metadata_file,
        model_name_or_path=model_name_or_path,
        index_type=index_type,
        index_params=index_params,
        chunks=chunks,
        embeddings=embeddings,
        embed_time=embed_time,
        index_time=index_time,
    )

    print(f"文件 {chunks_file.name} 处理完成")
    print(f"{'='*80}\n")


# ============================================
# 批量处理
# ============================================

def find_jsonl_files(chunks_path: Path) -> List[Path]:
    """查找所有 jsonl 文件"""
    if chunks_path.is_file():
        return [chunks_path]
    elif chunks_path.is_dir():
        jsonl_files = list(chunks_path.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"目录中没有找到 jsonl 文件: {chunks_path}")
        return sorted(jsonl_files)
    else:
        raise ValueError(f"路径不存在: {chunks_path}")


def batch_build(
    chunks_paths: List[Path],
    output_dir: Path,
    experiment_dir: Path,
    model_name_or_path: str,
    index_type: str,
    batch_size: int,
    use_fp16: bool,
    index_params: Optional[Dict] = None,
    force_rebuild: bool = False,
) -> None:
    """批量构建索引，支持跳过已存在的索引文件（除非 force_rebuild=True）"""
    
    # 收集所有需要处理的文件
    all_files = []
    for path in chunks_paths:
        all_files.extend(find_jsonl_files(path))
    
    # 去重
    all_files = sorted(set(all_files))
    
    print(f"\n{'#'*80}")
    print(f"批量处理开始")
    print(f"{'#'*80}")
    print(f"找到 {len(all_files)} 个 jsonl 文件:")
    for f in all_files:
        print(f"  - {f.name}")
    print(f"{'#'*80}\n")
    
    # 统计
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for idx, chunks_file in enumerate(all_files, 1):
        stem = chunks_file.stem
        index_file = output_dir / f"{stem}_{index_type}.faiss"
        
        # 检查是否跳过
        if index_file.exists() and not force_rebuild:
            print(f"\n进度: [{idx}/{len(all_files)}] - 跳过: {chunks_file.name} (索引已存在: {index_file})")
            skipped_count += 1
            continue
        
        try:
            print(f"\n进度: [{idx}/{len(all_files)}]")
            build_index_pipeline(
                chunks_file=chunks_file,
                output_dir=output_dir,
                experiment_dir=experiment_dir,
                model_name_or_path=model_name_or_path,
                index_type=index_type,
                batch_size=batch_size,
                use_fp16=use_fp16,
                index_params=index_params,
            )
            success_count += 1
        except Exception as e:
            print(f"处理失败: {chunks_file.name}")
            print(f"错误信息: {e}")
            fail_count += 1
            # 可选：继续处理下一个文件
            continue
    
    # 总结
    print(f"\n{'#'*80}")
    print(f"批量处理完成")
    print(f"{'#'*80}")
    print(f"成功: {success_count}, 失败: {fail_count}, 跳过: {skipped_count}, 总计: {len(all_files)}")
    print(f"{'#'*80}\n")


# ============================================
# CLI
# ============================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for FinRAG - 支持批量处理")

    # 输入参数组
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--chunks_file",
        type=str,
        nargs="+",
        help="输入 chunks jsonl 文件（可指定多个）",
    )
    input_group.add_argument(
        "--chunks_dir",
        type=str,
        help="输入 chunks 目录（处理目录下所有 .jsonl 文件）",
    )
    input_group.add_argument(
        "--batch",
        action="store_true",
        help="批量模式，处理默认 chunks 目录下的所有 jsonl 文件",
    )

    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/indexes"),
        help="FAISS index 和 metadata 输出目录",
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments"),
        help="实验报告输出目录",
    )

    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/9_data/ypq/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116",
        help="embedding 模型名称或本地路径",
    )

    # 索引参数
    parser.add_argument(
        "--index_type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw", "ivfpq"],
        help="FAISS 索引类型，默认 flat",
    )
    
    # IVF 参数
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="IVF/IVFPQ 聚类中心数，默认自动设置",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=10,
        help="IVF/IVFPQ 检索时探测的聚类数，默认 10",
    )
    
    # HNSW 参数
    parser.add_argument(
        "--hnsw_m",
        type=int,
        default=32,
        help="HNSW 每个节点的最大连接数，默认 32",
    )
    parser.add_argument(
        "--ef_construction",
        type=int,
        default=200,
        help="HNSW 构建时的候选列表大小，默认 200",
    )
    parser.add_argument(
        "--ef_search",
        type=int,
        default=50,
        help="HNSW 检索时的候选列表大小，默认 50",
    )
    
    # IVFPQ 参数
    parser.add_argument(
        "--pq_m",
        type=int,
        default=8,
        help="IVFPQ 子向量数，默认 8",
    )
    parser.add_argument(
        "--pq_nbits",
        type=int,
        default=8,
        help="IVFPQ 每个子向量的编码位数，默认 8",
    )

    # 其他参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="embedding batch size",
    )

    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="关闭 fp16",
    )
    
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重新构建索引，即使索引文件已存在",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 构建索引参数字典
    index_params = {}
    if args.index_type in ["ivf", "ivfpq"]:
        index_params['nprobe'] = args.nprobe
        if args.nlist is not None:
            index_params['nlist'] = args.nlist
    if args.index_type == "hnsw":
        index_params['m'] = args.hnsw_m
        index_params['ef_construction'] = args.ef_construction
        index_params['ef_search'] = args.ef_search
    if args.index_type == "ivfpq":
        index_params['pq_m'] = args.pq_m
        index_params['pq_nbits'] = args.pq_nbits
    
    # 确定要处理的文件列表
    chunks_paths = []
    
    if args.chunks_file:
        chunks_paths = [Path(f) for f in args.chunks_file]
    elif args.chunks_dir:
        chunks_paths = [Path(args.chunks_dir)]
    elif args.batch:
        default_dir = PROJECT_ROOT / "data/chunks"
        chunks_paths = [default_dir]
        print(f"批量模式: 处理目录 {default_dir}")
    else:
        default_file = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
        chunks_paths = [default_file]
        print(f"默认模式: 处理文件 {default_file}")
    
    # 执行批量构建
    batch_build(
        chunks_paths=chunks_paths,
        output_dir=Path(args.output_dir),
        experiment_dir=Path(args.experiment_dir),
        model_name_or_path=args.model_name_or_path,
        index_type=args.index_type,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        index_params=index_params,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
