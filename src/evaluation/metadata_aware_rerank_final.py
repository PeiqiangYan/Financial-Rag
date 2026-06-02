# -*- coding: utf-8 -*-
"""
metadata_aware_rerank_final.py

Final rerank pipeline:
- fact: metadata routing -> Hybrid Top20 -> metadata-aware bge-reranker-v2-m3 -> Top5
- compare: split into two fact subqueries, then run the same final rerank pipeline per subquery
- summary: Dense Top500 -> Local Hybrid Top20 -> metadata-aware bge-reranker-v2-m3 -> Top5

The report keeps Hybrid Top5 and Candidate Recall@20 as references, but the
final selected strategy is Metadata-aware Rerank Top5.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_FILE = PROJECT_ROOT / "data/eval_set/eval_set_60.jsonl"
DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat.faiss"
DEFAULT_FAISS_METADATA = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat_metadata.jsonl"

DEFAULT_EMBED_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--BAAI--bge-large-zh-v1.5/"
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116"
)

DEFAULT_RERANKER_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--BAAI--bge-reranker-v2-m3/"
    "snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)

DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "experiments/metadata_aware_rerank_final.md"
DEFAULT_OUTPUT_DETAILS = PROJECT_ROOT / "experiments/metadata_aware_rerank_final_details.jsonl"


RETRIEVAL_DIR = PROJECT_ROOT / "src/retrieval"
EVALUATION_DIR = PROJECT_ROOT / "src/evaluation"
RERANK_DIR = PROJECT_ROOT / "src/rerank"
for p in [RETRIEVAL_DIR, EVALUATION_DIR, RERANK_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metadata_hybrid_retriever import (  # noqa: E402
    RealEvidenceRetriever,
    parse_query,
    group_candidate_docs,
    get_doc_display_id,
    get_chunk_text,
)

from metadata_retrieval_eval import (  # noqa: E402
    BM25CandidateCache,
    DenseCandidateCache,
    dense_candidates as metadata_dense_candidates,
    hybrid_candidates as metadata_hybrid_candidates,
    find_evidence_rank as metadata_find_evidence_rank,
)

from metadata_aware_reranker import (  # noqa: E402
    BGEReranker,
    build_compare_subqueries,
    metadata_aware_rerank,
    route_candidate_chunks_for_query,
    to_list,
)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error: {path} line {line_no}: {e}") from e
    return rows


def write_jsonl(path: Union[str, Path], rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_final_candidates(
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    query: str,
    question_type: str,
    candidate_chunks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if question_type == "summary":
        coarse_chunks = metadata_dense_candidates(
            dense_cache=dense_cache,
            query=query,
            candidate_chunks=retriever.chunks,
            top_k=args.summary_dense_coarse_k,
        )
        hybrid_topk = metadata_hybrid_candidates(
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            query=query,
            candidate_chunks=coarse_chunks,
            top_k=args.candidate_k,
            dense_candidate_k=args.candidate_k,
            bm25_candidate_k=args.candidate_k,
        )
        return {
            "candidate_chunks": hybrid_topk,
            "pipeline": f"Dense Top{args.summary_dense_coarse_k} -> Local Hybrid Top{args.candidate_k}",
            "coarse_candidate_count": len(coarse_chunks),
            "coarse_candidate_doc_count": len(group_candidate_docs(coarse_chunks)),
        }

    hybrid_topk = metadata_hybrid_candidates(
        retriever=retriever,
        dense_cache=dense_cache,
        bm25_cache=bm25_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=args.candidate_k,
        dense_candidate_k=args.candidate_k,
        bm25_candidate_k=args.candidate_k,
    )
    return {
        "candidate_chunks": hybrid_topk,
        "pipeline": f"Metadata Routing -> Hybrid Top{args.candidate_k}",
    }


def find_rank(
    results: Sequence[Dict[str, Any]],
    gt_docs: Sequence[Any],
    keywords: Sequence[str],
    question_type: str,
    max_k: int,
    args: argparse.Namespace,
) -> int:
    return metadata_find_evidence_rank(
        results=results,
        gt_doc_ids=gt_docs,
        keywords=keywords,
        question_type=question_type,
        max_k=max_k,
        fact_match_mode=args.fact_match_mode,
        summary_match_mode=args.summary_match_mode,
        summary_min_hits=args.summary_min_hits,
        use_alias=not args.no_metric_alias,
        require_all_docs=False,
    )


def eval_fact_or_summary(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    reranker: BGEReranker,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query = item["query"]
    qtype = item.get("question_type", "fact")
    gt_docs = to_list(item.get("ground_truth_doc_id"))
    keywords = [str(x) for x in to_list(item.get("must_contain")) if str(x).strip()]

    routed = route_candidate_chunks_for_query(retriever, query, force_qtype=qtype)
    candidate_info = get_final_candidates(
        retriever=retriever,
        dense_cache=dense_cache,
        bm25_cache=bm25_cache,
        query=query,
        question_type=qtype,
        candidate_chunks=routed["candidate_chunks"],
        args=args,
    )
    candidates = candidate_info["candidate_chunks"]
    hybrid_top5 = candidates[: args.final_k]
    rerank_top5 = metadata_aware_rerank(
        reranker=reranker,
        query=query,
        candidates=candidates,
        top_k=args.final_k,
        batch_size=args.reranker_batch_size,
    )

    return {
        "query": query,
        "question_type": qtype,
        "ground_truth_doc_id": item.get("ground_truth_doc_id"),
        "must_contain": item.get("must_contain", []),
        "parsed": routed["parsed"],
        "pipeline": candidate_info.get("pipeline", ""),
        "candidate_count": len(candidates),
        "candidate_doc_count": len(group_candidate_docs(candidates)),
        "coarse_candidate_count": candidate_info.get("coarse_candidate_count", ""),
        "coarse_candidate_doc_count": candidate_info.get("coarse_candidate_doc_count", ""),
        "candidate_rank": find_rank(candidates, gt_docs, keywords, qtype, args.candidate_k, args),
        "hybrid_top5_rank": find_rank(hybrid_top5, gt_docs, keywords, qtype, args.final_k, args),
        "metadata_aware_rerank_top5_rank": find_rank(rerank_top5, gt_docs, keywords, qtype, args.final_k, args),
        "hybrid_top_doc": get_doc_display_id(hybrid_top5[0]) if hybrid_top5 else "",
        "rerank_top_doc": get_doc_display_id(rerank_top5[0]) if rerank_top5 else "",
        "hybrid_top_chunk_id": hybrid_top5[0].get("chunk_id", "") if hybrid_top5 else "",
        "rerank_top_chunk_id": rerank_top5[0].get("chunk_id", "") if rerank_top5 else "",
    }


def eval_compare(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    reranker: BGEReranker,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query = item["query"]
    gt_docs = [str(x) for x in to_list(item.get("ground_truth_doc_id"))]
    keywords = [str(x) for x in to_list(item.get("must_contain")) if str(x).strip()]
    parsed = parse_query(query)
    parsed["question_type"] = "compare"
    subqueries = build_compare_subqueries(retriever, query)

    candidate_ranks = []
    hybrid_ranks = []
    rerank_ranks = []
    sub_details = []

    for idx, sub in enumerate(subqueries):
        target_doc = gt_docs[idx] if idx < len(gt_docs) else ""
        sub_query = sub["query"]
        candidate_chunks = retriever.route_chunks(sub, fallback_if_empty=True)
        candidate_info = get_final_candidates(
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            query=sub_query,
            question_type="fact",
            candidate_chunks=candidate_chunks,
            args=args,
        )
        candidates = candidate_info["candidate_chunks"]
        hybrid_top5 = candidates[: args.final_k]
        rerank_top5 = metadata_aware_rerank(
            reranker=reranker,
            query=sub_query,
            candidates=candidates,
            top_k=args.final_k,
            batch_size=args.reranker_batch_size,
        )

        candidate_rank = find_rank(candidates, [target_doc], keywords, "fact", args.candidate_k, args)
        hybrid_rank = find_rank(hybrid_top5, [target_doc], keywords, "fact", args.final_k, args)
        rerank_rank = find_rank(rerank_top5, [target_doc], keywords, "fact", args.final_k, args)

        candidate_ranks.append(candidate_rank)
        hybrid_ranks.append(hybrid_rank)
        rerank_ranks.append(rerank_rank)
        sub_details.append(
            {
                "sub_query": sub_query,
                "target_doc": target_doc,
                "parsed": sub,
                "pipeline": candidate_info.get("pipeline", ""),
                "candidate_count": len(candidates),
                "candidate_doc_count": len(group_candidate_docs(candidates)),
                "candidate_rank": candidate_rank,
                "hybrid_top5_rank": hybrid_rank,
                "metadata_aware_rerank_top5_rank": rerank_rank,
            }
        )

    def combine_compare_rank(ranks: Sequence[int]) -> int:
        return max(ranks) if ranks and all(r != -1 for r in ranks) else -1

    return {
        "query": query,
        "question_type": "compare",
        "ground_truth_doc_id": item.get("ground_truth_doc_id"),
        "must_contain": item.get("must_contain", []),
        "parsed": parsed,
        "sub_details": sub_details,
        "candidate_rank": combine_compare_rank(candidate_ranks),
        "hybrid_top5_rank": combine_compare_rank(hybrid_ranks),
        "metadata_aware_rerank_top5_rank": combine_compare_rank(rerank_ranks),
    }


def eval_one_item(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    reranker: BGEReranker,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if item.get("question_type", "fact") == "compare":
        return eval_compare(item, retriever, dense_cache, bm25_cache, reranker, args)
    return eval_fact_or_summary(item, retriever, dense_cache, bm25_cache, reranker, args)


def hit(rank: int, k: int) -> int:
    return int(rank != -1 and rank <= k)


def reciprocal_rank(rank: int, k: int) -> float:
    if rank == -1 or rank > k:
        return 0.0
    return 1.0 / rank


def calc_metrics(details: Sequence[Dict[str, Any]], rank_field: str, k: int) -> Dict[str, float]:
    ranks = [int(d.get(rank_field, -1)) for d in details]
    if not ranks:
        return {"recall": 0.0, "top1_acc": 0.0, "mrr": 0.0}
    return {
        "recall": sum(hit(r, k) for r in ranks) / len(ranks),
        "top1_acc": sum(hit(r, 1) for r in ranks) / len(ranks),
        "mrr": sum(reciprocal_rank(r, k) for r in ranks) / len(ranks),
    }


def fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def metrics_rows(details: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    candidate = calc_metrics(details, "candidate_rank", args.candidate_k)
    hybrid = calc_metrics(details, "hybrid_top5_rank", args.final_k)
    rerank = calc_metrics(details, "metadata_aware_rerank_top5_rank", args.final_k)

    return [
        "| Setting | Recall@5 | Top1 Acc | MRR@5 |",
        "|---|---:|---:|---:|",
        f"| Hybrid Candidate Recall@{args.candidate_k} | {fmt(candidate['recall'])} | - | - |",
        f"| Hybrid Top{args.final_k} | {fmt(hybrid['recall'])} | {fmt(hybrid['top1_acc'])} | {fmt(hybrid['mrr'])} |",
        (
            f"| Metadata-aware Rerank Top{args.final_k} | "
            f"{fmt(rerank['recall'])} | {fmt(rerank['top1_acc'])} | {fmt(rerank['mrr'])} |"
        ),
        (
            "| Delta Rerank - Hybrid | "
            f"{fmt(rerank['recall'] - hybrid['recall'])} | "
            f"{fmt(rerank['top1_acc'] - hybrid['top1_acc'])} | "
            f"{fmt(rerank['mrr'] - hybrid['mrr'])} |"
        ),
    ]


def build_report(details: Sequence[Dict[str, Any]], elapsed_sec: float, args: argparse.Namespace) -> str:
    overall_hybrid = calc_metrics(details, "hybrid_top5_rank", args.final_k)
    overall_rerank = calc_metrics(details, "metadata_aware_rerank_top5_rank", args.final_k)

    lines = []
    lines.append("# Metadata-aware Rerank Final")
    lines.append("")
    lines.append(f"- total_queries: {len(details)}")
    lines.append(f"- elapsed_sec: {elapsed_sec:.2f}")
    lines.append(f"- final_strategy: `Hybrid Top{args.candidate_k} -> Metadata-aware bge-reranker-v2-m3 -> Top{args.final_k}`")
    lines.append(f"- recall_gain: `{fmt(overall_rerank['recall'] - overall_hybrid['recall'])}`")
    lines.append(f"- top1_gain: `{fmt(overall_rerank['top1_acc'] - overall_hybrid['top1_acc'])}`")
    lines.append(f"- mrr_gain: `{fmt(overall_rerank['mrr'] - overall_hybrid['mrr'])}`")
    lines.append("")
    lines.append("## Eval Config")
    lines.append("")
    lines.append(f"- eval_file: `{args.eval_file}`")
    lines.append(f"- chunks_file: `{args.chunks_file}`")
    lines.append(f"- faiss_index: `{args.faiss_index}`")
    lines.append(f"- faiss_metadata: `{args.faiss_metadata}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- final_k: `{args.final_k}`")
    lines.append(f"- dense_global_top_k: `{args.dense_global_top_k}`")
    lines.append(f"- summary_dense_coarse_k: `{args.summary_dense_coarse_k}`")
    lines.append(f"- summary_pipeline: `Dense Top{args.summary_dense_coarse_k} -> Local Hybrid Top{args.candidate_k} -> Metadata-aware Rerank Top{args.final_k}`")
    lines.append(f"- hybrid_dense_weight: `{args.dense_weight}`")
    lines.append(f"- hybrid_bm25_weight: `{args.bm25_weight}`")
    lines.append(f"- use_metric_alias: `{not args.no_metric_alias}`")
    lines.append(f"- reranker_model: `{args.reranker_model}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.extend(metrics_rows(details, args))
    lines.append("")

    by_type = defaultdict(list)
    for d in details:
        by_type[d.get("question_type", "unknown")].append(d)

    lines.append("## By Question Type")
    lines.append("")
    for qtype, rows in by_type.items():
        lines.append(f"### {qtype}")
        lines.append("")
        lines.append(f"- count: {len(rows)}")
        lines.append("")
        lines.extend(metrics_rows(rows, args))
        lines.append("")

    lines.append("## Per-query Rank")
    lines.append("")
    lines.append("| Query | Type | Candidate | Hybrid Top5 | Metadata-aware Rerank Top5 | Ground Truth Doc | Must Contain |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for d in details:
        gt = d.get("ground_truth_doc_id", "")
        if isinstance(gt, list):
            gt = "<br>".join(str(x) for x in gt)
        must = "<br>".join(str(x) for x in to_list(d.get("must_contain", [])))
        lines.append(
            "| "
            + str(d["query"]).replace("|", "\\|")
            + " | "
            + str(d.get("question_type", ""))
            + " | "
            + str(d.get("candidate_rank", -1))
            + " | "
            + str(d.get("hybrid_top5_rank", -1))
            + " | "
            + str(d.get("metadata_aware_rerank_top5_rank", -1))
            + " | "
            + str(gt).replace("|", "\\|")
            + " | "
            + must.replace("|", "\\|")
            + " |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--chunks_file", type=str, default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--faiss_index", type=str, default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss_metadata", type=str, default=str(DEFAULT_FAISS_METADATA))
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--output_report", type=str, default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output_details", type=str, default=str(DEFAULT_OUTPUT_DETAILS))

    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--final_k", type=int, default=5)
    parser.add_argument("--dense_global_top_k", type=int, default=300)
    parser.add_argument("--summary_dense_coarse_k", type=int, default=500)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--bm25_weight", type=float, default=0.5)

    parser.add_argument("--fact_match_mode", type=str, default="all", choices=["all", "any", "at_least"])
    parser.add_argument("--summary_match_mode", type=str, default="any", choices=["all", "any", "at_least"])
    parser.add_argument("--summary_min_hits", type=int, default=1)
    parser.add_argument("--no_metric_alias", action="store_true")
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    eval_items = read_jsonl(args.eval_file)
    retriever = RealEvidenceRetriever(
        chunks_file=args.chunks_file,
        faiss_index=args.faiss_index,
        faiss_metadata=args.faiss_metadata,
        model_name_or_path=args.model_name_or_path,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        use_fp16=not args.no_fp16,
    )
    reranker = BGEReranker(
        model_name_or_path=args.reranker_model,
        use_fp16=not args.no_fp16,
        batch_size=args.reranker_batch_size,
    )
    dense_cache = DenseCandidateCache(
        retriever,
        global_top_k=max(args.dense_global_top_k, args.summary_dense_coarse_k),
    )
    bm25_cache = BM25CandidateCache(retriever)

    details = []
    for i, item in enumerate(eval_items, 1):
        print(f"[final-rerank] [{i}/{len(eval_items)}] {item.get('question_type')} | {item.get('query')}")
        details.append(eval_one_item(item, retriever, dense_cache, bm25_cache, reranker, args))

    elapsed_sec = time.time() - t0
    report = build_report(details, elapsed_sec, args)

    output_report = Path(args.output_report)
    output_details = Path(args.output_details)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_details.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    write_jsonl(output_details, details)

    print("\n" + "=" * 100)
    print("Metadata-aware rerank final done")
    print("=" * 100)
    print(f"report: {output_report}")
    print(f"details: {output_details}")
    print(f"elapsed_sec: {elapsed_sec:.2f}")


if __name__ == "__main__":
    main()



