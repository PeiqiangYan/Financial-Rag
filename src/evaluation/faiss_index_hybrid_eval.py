# -*- coding: utf-8 -*-
"""
FAISS index comparison under the same Hybrid retrieval setting.

Experiment:
- build Flat / IVF / HNSW indexes from the same chunk embeddings
- evaluate each index with Metadata Routing + Hybrid retrieval
- report Recall@K, average query latency, index build time and serialized index size
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_FILE = PROJECT_ROOT / "data/eval_set/eval_set_60.jsonl"
DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/indexes/faiss_compare"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "experiments/faiss_index_hybrid_eval.md"
DEFAULT_OUTPUT_DETAILS = PROJECT_ROOT / "experiments/faiss_index_hybrid_eval_details.json"

DEFAULT_EMBED_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--BAAI--bge-large-zh-v1.5/"
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116"
)

for p in [
    PROJECT_ROOT / "src/indexing",
    PROJECT_ROOT / "src/retrieval",
    PROJECT_ROOT / "src/evaluation",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import faiss  # noqa: E402
from build_faiss import (  # noqa: E402
    build_faiss_index,
    build_texts,
    encode_texts,
    load_embedding_model,
    read_chunks,
    save_index,
    save_metadata,
)
from metadata_retrieval_eval import (  # noqa: E402
    BM25CandidateCache,
    DenseCandidateCache,
    RealEvidenceRetriever,
    eval_one_item,
    read_jsonl,
    recall_at_k,
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def index_size_mb(index: faiss.Index) -> float:
    serialized = faiss.serialize_index(index)
    return float(serialized.nbytes) / 1024 / 1024


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0


def index_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        {
            "name": "Flat",
            "index_type": "flat",
            "params": {},
        },
        {
            "name": "IVF",
            "index_type": "ivf",
            "params": {
                "nlist": args.ivf_nlist if args.ivf_nlist > 0 else None,
                "nprobe": args.ivf_nprobe,
            },
        },
        {
            "name": "HNSW",
            "index_type": "hnsw",
            "params": {
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef_search": args.hnsw_ef_search,
            },
        },
    ]


def build_indexes(
    args: argparse.Namespace,
    chunks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.chunks_file).stem
    built: List[Dict[str, Any]] = []

    for cfg in index_configs(args):
        index_type = cfg["index_type"]
        index_file = output_dir / f"{stem}_{index_type}.faiss"
        metadata_file = output_dir / f"{stem}_{index_type}_metadata.jsonl"

        print("\n" + "=" * 80)
        print(f"构建索引: {cfg['name']}")
        print("=" * 80)

        start = time.perf_counter()
        index = build_faiss_index(
            embeddings=embeddings,
            index_type=index_type,
            **cfg["params"],
        )
        build_time_sec = time.perf_counter() - start

        memory_mb = index_size_mb(index)
        save_index(index, index_file)
        save_metadata(list(chunks), metadata_file)

        built.append(
            {
                "name": cfg["name"],
                "index_type": index_type,
                "params": cfg["params"],
                "index_file": str(index_file),
                "metadata_file": str(metadata_file),
                "build_time_sec": build_time_sec,
                "memory_mb": memory_mb,
                "file_size_mb": file_size_mb(index_file),
            }
        )

        del index

    return built


def evaluate_one_index(
    args: argparse.Namespace,
    index_info: Dict[str, Any],
    eval_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"评测索引: {index_info['name']} | Hybrid Recall@{args.recall_k}")
    print("=" * 80)

    retriever = RealEvidenceRetriever(
        chunks_file=args.chunks_file,
        faiss_index=index_info["index_file"],
        faiss_metadata=index_info["metadata_file"],
        model_name_or_path=args.model_name_or_path,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        use_fp16=not args.no_fp16,
    )

    dense_cache = DenseCandidateCache(
        retriever=retriever,
        global_top_k=args.dense_global_top_k,
    )
    bm25_cache = BM25CandidateCache(retriever)
    use_alias = not args.no_metric_alias

    details: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for i, item in enumerate(eval_items, 1):
        print(f"[{index_info['name']}] [{i}/{len(eval_items)}] {item.get('question_type')} | {item.get('query')}")
        start = time.perf_counter()
        row = call_eval_one_item(
            args=args,
            item=item,
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            use_alias=use_alias,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        row["latency_ms"] = latency_ms
        details.append(row)
        latencies_ms.append(latency_ms)

    rank_key = f"hybrid_{args.rank_level}_rank"
    ranks = [int(row[rank_key]) for row in details]
    recall = recall_at_k(ranks, args.recall_k)
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p50_latency_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    by_type: Dict[str, Dict[str, Any]] = {}
    for qtype in sorted(set(str(x.get("question_type", "")) for x in details)):
        subset = [x for x in details if str(x.get("question_type", "")) == qtype]
        type_ranks = [int(x[rank_key]) for x in subset]
        type_latencies = [float(x["latency_ms"]) for x in subset]
        by_type[qtype] = {
            "count": len(subset),
            f"hybrid_recall@{args.recall_k}": recall_at_k(type_ranks, args.recall_k),
            "avg_latency_ms": float(np.mean(type_latencies)) if type_latencies else 0.0,
        }

    return {
        **index_info,
        "rank_level": args.rank_level,
        f"hybrid_recall@{args.recall_k}": recall,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "total_queries": len(details),
        "by_type": by_type,
        "details": details,
    }


def call_eval_one_item(
    args: argparse.Namespace,
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    use_alias: bool,
) -> Dict[str, Any]:
    kwargs = {
        "item": item,
        "retriever": retriever,
        "dense_cache": dense_cache,
        "bm25_cache": bm25_cache,
        "max_k": args.recall_k,
        "dense_candidate_k": args.dense_candidate_k,
        "bm25_candidate_k": args.bm25_candidate_k,
        "fact_match_mode": args.fact_match_mode,
        "summary_match_mode": args.summary_match_mode,
        "summary_min_hits": args.summary_min_hits,
        "use_alias": use_alias,
    }

    sig = inspect.signature(eval_one_item)
    if "summary_seed_k" in sig.parameters:
        kwargs["summary_seed_k"] = args.summary_seed_k

    return eval_one_item(**kwargs)


def build_report(args: argparse.Namespace, summaries: Sequence[Dict[str, Any]], embedding_time_sec: float) -> str:
    lines: List[str] = []
    lines.append("# FAISS 索引对比：Hybrid 召回")
    lines.append("")
    lines.append("## Eval Config")
    lines.append("")
    lines.append(f"- eval_file: `{args.eval_file}`")
    lines.append(f"- chunks_file: `{args.chunks_file}`")
    lines.append(f"- recall_k: `{args.recall_k}`")
    lines.append(f"- rank_level: `{args.rank_level}`")
    lines.append(f"- dense_global_top_k: `{args.dense_global_top_k}`")
    lines.append(f"- dense_candidate_k: `{args.dense_candidate_k}`")
    lines.append(f"- bm25_candidate_k: `{args.bm25_candidate_k}`")
    lines.append(f"- hybrid_dense_weight: `{args.dense_weight}`")
    lines.append(f"- hybrid_bm25_weight: `{args.bm25_weight}`")
    lines.append(f"- embedding_model: `{args.model_name_or_path}`")
    lines.append(f"- embedding_time_sec: `{embedding_time_sec:.2f}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| 索引类型 | Recall@{} | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) | P50(ms) | P95(ms) |".format(
            args.recall_k
        )
    )
    lines.append("|----------|-----------:|-------------:|------------:|-------------:|--------:|--------:|")
    for item in summaries:
        lines.append(
            f"| {item['name']} | "
            f"{fmt_float(item[f'hybrid_recall@{args.recall_k}'])} | "
            f"{fmt_float(item['avg_latency_ms'], 2)} | "
            f"{fmt_float(item['build_time_sec'], 2)} | "
            f"{fmt_float(item['memory_mb'], 2)} | "
            f"{fmt_float(item['p50_latency_ms'], 2)} | "
            f"{fmt_float(item['p95_latency_ms'], 2)} |"
        )
    lines.append("")
    lines.append("> 查询延迟统计的是单条 query 的 Hybrid 召回评测耗时，不包含模型和索引加载时间。")
    lines.append("> 构建时间统计的是 FAISS index 训练 / add 向量时间，不包含 embedding 生成时间。")
    lines.append("")

    lines.append("## By Question Type")
    lines.append("")
    for item in summaries:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(f"| Type | Count | Recall@{args.recall_k} | Avg Latency(ms) |")
        lines.append("|---|---:|---:|---:|")
        for qtype, row in item["by_type"].items():
            lines.append(
                f"| {qtype} | {row['count']} | "
                f"{fmt_float(row[f'hybrid_recall@{args.recall_k}'])} | "
                f"{fmt_float(row['avg_latency_ms'], 2)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FAISS index types under Hybrid retrieval.")
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--chunks_file", type=str, default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output_report", type=str, default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output_details", type=str, default=str(DEFAULT_OUTPUT_DETAILS))
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_fp16", action="store_true")

    parser.add_argument("--recall_k", type=int, default=10)
    parser.add_argument("--rank_level", choices=["evidence", "doc"], default="evidence")
    parser.add_argument("--dense_global_top_k", type=int, default=300)
    parser.add_argument("--dense_candidate_k", type=int, default=20)
    parser.add_argument("--bm25_candidate_k", type=int, default=20)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    parser.add_argument("--no_metric_alias", action="store_true")

    parser.add_argument("--fact_match_mode", choices=["all", "any", "at_least"], default="all")
    parser.add_argument("--summary_match_mode", choices=["all", "any", "at_least"], default="any")
    parser.add_argument("--summary_min_hits", type=int, default=1)
    parser.add_argument("--summary_seed_k", type=int, default=300)

    parser.add_argument("--ivf_nlist", type=int, default=0, help="0 means auto: 4 * sqrt(N), clipped to [10, 1000].")
    parser.add_argument("--ivf_nprobe", type=int, default=10)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)
    parser.add_argument("--hnsw_ef_search", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_file = Path(args.chunks_file)
    eval_items = read_jsonl(args.eval_file)
    chunks = read_chunks(chunks_file)
    texts = build_texts(chunks)

    print("\n" + "=" * 80)
    print("生成 embedding，一次复用给 Flat / IVF / HNSW")
    print("=" * 80)
    model = load_embedding_model(
        model_name_or_path=args.model_name_or_path,
        use_fp16=not args.no_fp16,
    )
    emb_start = time.perf_counter()
    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    embedding_time_sec = time.perf_counter() - emb_start
    del model

    built_indexes = build_indexes(args=args, chunks=chunks, embeddings=embeddings)
    summaries = [evaluate_one_index(args, info, eval_items) for info in built_indexes]

    report = build_report(args=args, summaries=summaries, embedding_time_sec=embedding_time_sec)
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")

    write_json(
        Path(args.output_details),
        {
            "config": vars(args),
            "embedding_time_sec": embedding_time_sec,
            "summaries": summaries,
        },
    )

    print("\n" + "=" * 100)
    print("FAISS index Hybrid eval done")
    print("=" * 100)
    print(f"report: {args.output_report}")
    print(f"details: {args.output_details}")


if __name__ == "__main__":
    main()
