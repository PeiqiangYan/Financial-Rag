# -*- coding: utf-8 -*-
"""
Metadata-aware reranker utilities used by the final FinRAG pipeline.

This module contains only production-facing helpers:
- route candidate chunks with metadata
- split compare queries into fact subqueries
- run bge-reranker-v2-m3 with doc/page/section metadata in the passage
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_DIR = PROJECT_ROOT / "src/retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from metadata_hybrid_retriever import (  # noqa: E402
    RealEvidenceRetriever,
    parse_query,
    group_candidate_docs,
    get_doc_display_id,
    get_chunk_text,
)


def to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


class BGEReranker:
    def __init__(self, model_name_or_path: str, use_fp16: bool = True, batch_size: int = 32) -> None:
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
        except ImportError as e:
            raise ImportError("FlagEmbedding is required for FlagReranker.") from e

        print(f"Loading reranker: {model_name_or_path}")
        self.model = FlagReranker(model_name_or_path, use_fp16=use_fp16)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [[query, get_chunk_text(x)] for x in candidates]
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)
        if isinstance(scores, float):
            scores = [scores]
        scored = []
        for row, score in zip(candidates, scores):
            out = dict(row)
            out["reranker_score"] = float(score)
            out["score"] = float(score)
            out["rank_source"] = "bge_reranker_v2_m3"
            scored.append(out)
        scored.sort(key=lambda x: x["reranker_score"], reverse=True)
        return scored[:top_k]


def route_candidate_chunks_for_query(
    retriever: RealEvidenceRetriever,
    query: str,
    force_qtype: str = "",
) -> Dict[str, Any]:
    parsed = parse_query(query)
    if force_qtype:
        parsed["question_type"] = force_qtype

    if parsed.get("question_type") == "summary":
        has_routing_signal = bool(parsed.get("company_short") or parsed.get("year"))
        if has_routing_signal:
            chunks = retriever.route_chunks(parsed, fallback_if_empty=True)
        else:
            chunks = retriever.chunks
    else:
        chunks = retriever.route_chunks(parsed, fallback_if_empty=True)

    return {
        "parsed": parsed,
        "candidate_chunks": chunks,
        "candidate_doc_count": len(group_candidate_docs(chunks)),
        "candidate_chunk_count": len(chunks),
    }


def build_compare_subqueries(retriever: RealEvidenceRetriever, query: str) -> List[Dict[str, Any]]:
    parsed = parse_query(query)
    parsed["question_type"] = "compare"
    return retriever.build_compare_subqueries_from_parsed(query, parsed)


def format_rerank_passage(row: Dict[str, Any]) -> str:
    doc = get_doc_display_id(row)
    page = row.get("page", row.get("page_no", row.get("page_index", "")))
    section = row.get("section", row.get("section_title", row.get("title", "")))
    chunk_type = row.get("chunk_type", row.get("type", ""))

    parts = []
    if doc:
        parts.append(f"文档: {doc}")
    if page != "":
        parts.append(f"页码: {page}")
    if section:
        parts.append(f"章节: {section}")
    if chunk_type:
        parts.append(f"类型: {chunk_type}")
    parts.append(f"内容: {get_chunk_text(row)}")
    return "\n".join(str(x) for x in parts if str(x).strip())


def metadata_aware_rerank(
    reranker: BGEReranker,
    query: str,
    candidates: Sequence[Dict[str, Any]],
    top_k: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    pairs = [[query, format_rerank_passage(row)] for row in candidates]
    scores = reranker.model.compute_score(pairs, batch_size=batch_size)
    if isinstance(scores, float):
        scores = [scores]

    scored = []
    for rank, (row, score) in enumerate(zip(candidates, scores), 1):
        out = dict(row)
        out["hybrid_rank"] = rank
        out["reranker_score"] = float(score)
        out["score"] = float(score)
        out["rank_source"] = "metadata_aware_bge_reranker_v2_m3"
        scored.append(out)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored[:top_k]
