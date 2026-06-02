# -*- coding: utf-8 -*-
"""
metadata_retrieval_eval.py

Metadata Routing 场景下的基础召回器对比。

用于填表：

| 召回方案 | Recall@3 | Recall@5 | Recall@10 | 备注 |
|----------|----------|----------|-----------|------|
| 纯向量 Dense | ? | ? | ? | metadata routing 后的向量召回 |
| 纯 BM25 | ? | ? | ? | metadata routing 后的关键词召回 |
| 混合召回 Hybrid | ? | ? | ? | metadata routing 后 Dense + BM25 融合 |

实验口径与最终检索链路的 Direct Top5 保持一致：
- 使用 query 自动解析 company/year/period/metric
- 使用 metadata routing 缩小候选 chunks
- fact/summary 在候选 chunks 内检索
- compare 先拆成两个 fact 子查询，再分别 metadata routing 检索

不使用：
- ground_truth_doc_id 参与检索
- oracle doc-aware
- evidence_signal rerank
- bge-reranker

ground_truth_doc_id / must_contain 只用于最后评测。
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_FILE = PROJECT_ROOT / "data/eval_set/eval_set_60.jsonl"
DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat.faiss"
DEFAULT_FAISS_METADATA = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat_metadata.jsonl"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data/chunks"
DEFAULT_INDEXES_DIR = PROJECT_ROOT / "data/indexes"

DEFAULT_CHUNK_EXPERIMENTS = [
    "all_cs512_ov200",
    "all_cs256_ov100",
    "all_cs256_ov200",
    "all_cs256_ov50",
    "all_cs512_ov100",
    "all_cs512_ov50",
    "all_cs1024_ov100",
    "all_cs1024_ov50",
    "all_cs1024_ov200",
]

DEFAULT_EMBED_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--BAAI--bge-large-zh-v1.5/"
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116"
)

DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "experiments/metadata_retrieval_eval.md"
DEFAULT_OUTPUT_DETAILS = PROJECT_ROOT / "experiments/metadata_retrieval_eval_details.jsonl"
DEFAULT_BATCH_OUTPUT_REPORT = PROJECT_ROOT / "experiments/metadata_retrieval_eval_batch.md"
DEFAULT_BATCH_OUTPUT_DETAILS = PROJECT_ROOT / "experiments/metadata_retrieval_eval_batch_details.jsonl"


# 复用 metadata_hybrid_retriever.py 里的真实 query parsing / metadata routing
RETRIEVAL_DIR = PROJECT_ROOT / "src/retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from metadata_hybrid_retriever import (  # noqa: E402
    RealEvidenceRetriever,
    parse_query,
    group_candidate_docs,
    fuse_dense_bm25,
    dedup_by_chunk_id,
    chunk_key,
    get_doc_display_id,
    get_chunk_text,
    normalize_text,
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
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


def to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


# ============================================================
# Doc matching
# ============================================================

def normalize_doc_id(s: Any) -> str:
    if s is None:
        return ""

    s = str(s).strip()
    s = re.sub(r"\.(pdf|md|txt|jsonl)$", "", s, flags=re.I)

    remove_tokens = [
        "【洞见研报DJyanbao.com】",
        "【深交所】",
        "【上交所】",
        "【上交所科创板】",
    ]
    for token in remove_tokens:
        s = s.replace(token, "")

    s = s.replace("：", "")
    s = s.replace(":", "")
    s = s.replace(" ", "")
    s = s.replace("\u3000", "")

    return s.lower()


def extract_doc_year(s: Any) -> str:
    s = "" if s is None else str(s)
    m = re.search(r"(20\d{2})", s)
    return m.group(1) if m else ""


def extract_doc_period(s: Any) -> str:
    s = "" if s is None else str(s)

    if re.search(r"(半年度报告|中期报告|半年报)", s, flags=re.I):
        return "semiannual"

    if re.search(r"(第一季度报告|一季报|第1季度报告|1季报|q1)", s, flags=re.I):
        return "q1"

    if re.search(r"(第三季度报告|三季报|第3季度报告|3季报|q3)", s, flags=re.I):
        return "q3"

    if re.search(r"(年度报告全文|年度报告|年报)", s, flags=re.I):
        return "annual"

    return ""


def strip_doc_period_words(s: str) -> str:
    patterns = [
        r"20\d{2}年?",
        r"年度报告全文",
        r"半年度报告",
        r"中期报告",
        r"第一季度报告",
        r"第三季度报告",
        r"季度报告",
        r"年度报告",
        r"报告全文",
        r"年报",
        r"半年报",
        r"一季报",
        r"三季报",
    ]

    for p in patterns:
        s = re.sub(p, "", s, flags=re.I)

    return s


def doc_id_matches(pred_doc_id: Any, gt_doc_id: Any) -> bool:
    raw_p = "" if pred_doc_id is None else str(pred_doc_id)
    raw_g = "" if gt_doc_id is None else str(gt_doc_id)

    p = normalize_doc_id(raw_p)
    g = normalize_doc_id(raw_g)

    if not p or not g:
        return False

    if p == g:
        return True

    py = extract_doc_year(raw_p)
    gy = extract_doc_year(raw_g)
    if py and gy and py != gy:
        return False

    pp = extract_doc_period(raw_p)
    gp = extract_doc_period(raw_g)
    if pp and gp and pp != gp:
        return False

    p_core = strip_doc_period_words(p)
    g_core = strip_doc_period_words(g)

    if not p_core or not g_core:
        return False

    if min(len(p_core), len(g_core)) < 4:
        return False

    return p_core == g_core or p_core in g_core or g_core in p_core


def any_doc_id_matches(pred_doc_id: Any, gt_doc_ids: Sequence[Any]) -> bool:
    return any(doc_id_matches(pred_doc_id, gt) for gt in gt_doc_ids)


# ============================================================
# Metric alias evidence matching
# ============================================================

METRIC_ALIASES = {
    "研发费用": ["研发费用", "研发投入", "研发投入金额"],
    "研发投入": ["研发投入", "研发费用", "研发投入金额"],
    "研发投入占营业收入比例": ["研发投入占营业收入比例", "研发投入比例", "研发投入占比"],

    "营业收入": ["营业收入", "主营业务收入", "收入"],
    "主营业务收入": ["主营业务收入", "营业收入", "收入"],

    "归母净利润": ["归母净利润", "归属于上市公司股东的净利润"],
    "归属于上市公司股东的净利润": ["归属于上市公司股东的净利润", "归母净利润"],

    "总资产": ["总资产", "资产总计", "资产总额"],
    "资产总计": ["资产总计", "总资产", "资产总额"],
    "资产负债率": ["资产负债率"],

    "经营活动现金流入": ["经营活动现金流入", "经营活动现金流入小计"],
    "经营活动产生的现金流量净额": [
        "经营活动产生的现金流量净额",
        "经营活动现金流量净额",
        "经营活动产生的现金流量",
    ],

    "毛利率": ["毛利率"],
    "基本每股收益": ["基本每股收益"],
    "保证借款": ["保证借款"],

    "动力电池销量": ["动力电池销量", "动力电池", "销量"],
    "储能电池销量": ["储能电池销量", "储能电池", "销量"],
    "正极材料": ["正极材料"],
    "销量": ["销量"],
    "组件": ["组件"],
    "出货量": ["出货量", "发货量"],
    "风电": ["风电", "风力发电"],
}


def expand_keywords_with_alias(
    keywords: Sequence[str],
    use_alias: bool = True,
) -> List[List[str]]:
    groups: List[List[str]] = []

    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue

        aliases = METRIC_ALIASES.get(kw, [kw]) if use_alias else [kw]

        uniq = []
        for x in aliases:
            if x not in uniq:
                uniq.append(x)

        groups.append(uniq)

    return groups


def keyword_match(
    content: str,
    keywords: Sequence[str],
    mode: str = "all",
    min_hits: int = 1,
    use_alias: bool = True,
) -> bool:
    keywords = [str(x) for x in keywords if str(x).strip()]
    if not keywords:
        return True

    content_norm = normalize_text(content)
    groups = expand_keywords_with_alias(keywords, use_alias=use_alias)

    group_hits = 0

    for group in groups:
        hit_this_group = False

        for kw in group:
            kw_norm = normalize_text(kw)
            if kw_norm and kw_norm in content_norm:
                hit_this_group = True
                break

        if hit_this_group:
            group_hits += 1

    if mode == "all":
        return group_hits == len(groups)

    if mode == "any":
        return group_hits >= 1

    if mode == "at_least":
        return group_hits >= min(min_hits, len(groups))

    raise ValueError(f"Unknown match mode: {mode}")


# ============================================================
# BM25 cache
# ============================================================

class BM25CandidateCache:
    """
    避免每个 query 都重新给同一个候选文档集合构建 BM25。

    metadata routing 后，很多 query 会反复落到同一份年报。
    因此用 candidate doc ids 作为 cache key。
    """

    def __init__(self, retriever: RealEvidenceRetriever) -> None:
        self.retriever = retriever
        self.cache: Dict[str, Dict[str, Any]] = {}

    def make_key(self, chunks: Sequence[Dict[str, Any]]) -> str:
        doc_ids = sorted(set(get_doc_display_id(ch) for ch in chunks))
        return "||".join(doc_ids)

    def search(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        key = self.make_key(chunks)

        if key not in self.cache:
            corpus = [self.retriever.bm25.tokenize(get_chunk_text(ch)) for ch in chunks]
            bm25 = self.retriever.bm25.BM25Okapi(corpus)
            self.cache[key] = {
                "chunks": list(chunks),
                "bm25": bm25,
            }

        obj = self.cache[key]
        bm25 = obj["bm25"]
        cached_chunks = obj["chunks"]

        q_tokens = self.retriever.bm25.tokenize(query)
        scores = bm25.get_scores(q_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            row = dict(cached_chunks[int(idx)])
            row["bm25_score"] = score
            row["score"] = score
            row["rank_source"] = "bm25_metadata"
            results.append(row)

        return results


class DenseCandidateCache:
    """
    Cache global FAISS searches by query.

    A single eval item asks for pure dense and hybrid. Without this cache the
    same query is encoded and searched twice. Compare items also repeat this
    pattern for each subquery.
    """

    def __init__(self, retriever: RealEvidenceRetriever, global_top_k: int) -> None:
        self.retriever = retriever
        self.global_top_k = global_top_k
        self.cache: Dict[str, List[Dict[str, Any]]] = {}

    def search(
        self,
        query: str,
        candidate_chunks: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        if query not in self.cache:
            self.cache[query] = self.retriever.dense.search_global(
                query,
                top_k=self.global_top_k,
            )

        candidate_keys = set(chunk_key(ch) for ch in candidate_chunks)
        candidate_doc_ids = set(get_doc_display_id(ch) for ch in candidate_chunks)

        filtered = []
        for r in self.cache[query]:
            key = chunk_key(r)
            doc_id = get_doc_display_id(r)
            if key in candidate_keys or doc_id in candidate_doc_ids:
                filtered.append(r)

            if len(filtered) >= top_k:
                break

        return filtered


# ============================================================
# Routing / retrieval
# ============================================================

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


def dense_candidates(
    dense_cache: DenseCandidateCache,
    query: str,
    candidate_chunks: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    return dense_cache.search(
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=top_k,
    )


def bm25_candidates(
    bm25_cache: BM25CandidateCache,
    query: str,
    candidate_chunks: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    return bm25_cache.search(
        query=query,
        chunks=candidate_chunks,
        top_k=top_k,
    )


def hybrid_candidates(
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    query: str,
    candidate_chunks: Sequence[Dict[str, Any]],
    top_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
) -> List[Dict[str, Any]]:
    d_res = dense_candidates(
        dense_cache=dense_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=dense_candidate_k,
    )

    b_res = bm25_candidates(
        bm25_cache=bm25_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=bm25_candidate_k,
    )

    fused = fuse_dense_bm25(
        dense_results=d_res,
        bm25_results=b_res,
        dense_weight=retriever.dense_weight,
        bm25_weight=retriever.bm25_weight,
        top_k=max(top_k, dense_candidate_k, bm25_candidate_k),
    )

    return dedup_by_chunk_id(fused)[:top_k]


# ============================================================
# Evidence / doc rank
# ============================================================

def doc_level_hit_at_k(
    results: Sequence[Dict[str, Any]],
    gt_doc_ids: Sequence[Any],
    k: int,
    require_all_docs: bool = False,
) -> bool:
    top = results[:k]
    pred_doc_ids = [get_doc_display_id(r) for r in top]

    if require_all_docs:
        return all(
            any(doc_id_matches(pred, gt) for pred in pred_doc_ids)
            for gt in gt_doc_ids
        )

    return any(
        any_doc_id_matches(pred, gt_doc_ids)
        for pred in pred_doc_ids
    )


def evidence_hit_at_k(
    results: Sequence[Dict[str, Any]],
    gt_doc_ids: Sequence[Any],
    keywords: Sequence[str],
    question_type: str,
    k: int,
    fact_match_mode: str,
    summary_match_mode: str,
    summary_min_hits: int,
    use_alias: bool,
    require_all_docs: bool = False,
) -> bool:
    if not keywords:
        return doc_level_hit_at_k(
            results=results,
            gt_doc_ids=gt_doc_ids,
            k=k,
            require_all_docs=require_all_docs,
        )

    top = results[:k]

    if require_all_docs:
        for gt in gt_doc_ids:
            found_this_doc = False

            for r in top:
                if doc_id_matches(get_doc_display_id(r), gt) and keyword_match(
                    get_chunk_text(r),
                    keywords,
                    mode=fact_match_mode,
                    use_alias=use_alias,
                ):
                    found_this_doc = True
                    break

            if not found_this_doc:
                return False

        return True

    mode = summary_match_mode if question_type == "summary" else fact_match_mode
    min_hits = summary_min_hits if question_type == "summary" else 1

    for r in top:
        if any_doc_id_matches(get_doc_display_id(r), gt_doc_ids) and keyword_match(
            get_chunk_text(r),
            keywords,
            mode=mode,
            min_hits=min_hits,
            use_alias=use_alias,
        ):
            return True

    return False


def find_doc_rank(
    results: Sequence[Dict[str, Any]],
    gt_doc_ids: Sequence[Any],
    max_k: int,
    require_all_docs: bool = False,
) -> int:
    for k in range(1, max_k + 1):
        if doc_level_hit_at_k(
            results=results,
            gt_doc_ids=gt_doc_ids,
            k=k,
            require_all_docs=require_all_docs,
        ):
            return k

    return -1


def find_evidence_rank(
    results: Sequence[Dict[str, Any]],
    gt_doc_ids: Sequence[Any],
    keywords: Sequence[str],
    question_type: str,
    max_k: int,
    fact_match_mode: str,
    summary_match_mode: str,
    summary_min_hits: int,
    use_alias: bool,
    require_all_docs: bool = False,
) -> int:
    for k in range(1, max_k + 1):
        if evidence_hit_at_k(
            results=results,
            gt_doc_ids=gt_doc_ids,
            keywords=keywords,
            question_type=question_type,
            k=k,
            fact_match_mode=fact_match_mode,
            summary_match_mode=summary_match_mode,
            summary_min_hits=summary_min_hits,
            use_alias=use_alias,
            require_all_docs=require_all_docs,
        ):
            return k

    return -1


# ============================================================
# Eval one item
# ============================================================

def eval_fact_or_summary(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    max_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    fact_match_mode: str,
    summary_match_mode: str,
    summary_min_hits: int,
    use_alias: bool,
) -> Dict[str, Any]:
    query = item["query"]
    qtype = item.get("question_type", "fact")
    gt_docs = to_list(item.get("ground_truth_doc_id"))
    keywords = [str(x) for x in to_list(item.get("must_contain")) if str(x).strip()]

    routed = route_candidate_chunks_for_query(
        retriever=retriever,
        query=query,
        force_qtype=qtype,
    )

    candidate_chunks = routed["candidate_chunks"]

    dense_res = dense_candidates(
        dense_cache=dense_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=max_k,
    )

    bm25_res = bm25_candidates(
        bm25_cache=bm25_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=max_k,
    )

    hybrid_res = hybrid_candidates(
        retriever=retriever,
        dense_cache=dense_cache,
        bm25_cache=bm25_cache,
        query=query,
        candidate_chunks=candidate_chunks,
        top_k=max_k,
        dense_candidate_k=dense_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
    )

    output: Dict[str, Any] = {
        "query": query,
        "question_type": qtype,
        "ground_truth_doc_id": item.get("ground_truth_doc_id"),
        "must_contain": item.get("must_contain", []),
        "parsed": routed["parsed"],
        "candidate_doc_count": routed["candidate_doc_count"],
        "candidate_chunk_count": routed["candidate_chunk_count"],
    }

    for method, results in [
        ("dense", dense_res),
        ("bm25", bm25_res),
        ("hybrid", hybrid_res),
    ]:
        output[f"{method}_doc_rank"] = find_doc_rank(
            results=results,
            gt_doc_ids=gt_docs,
            max_k=max_k,
            require_all_docs=False,
        )

        output[f"{method}_evidence_rank"] = find_evidence_rank(
            results=results,
            gt_doc_ids=gt_docs,
            keywords=keywords,
            question_type=qtype,
            max_k=max_k,
            fact_match_mode=fact_match_mode,
            summary_match_mode=summary_match_mode,
            summary_min_hits=summary_min_hits,
            use_alias=use_alias,
            require_all_docs=False,
        )

    return output


def eval_compare(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    max_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    fact_match_mode: str,
    use_alias: bool,
) -> Dict[str, Any]:
    query = item["query"]
    gt_docs = [str(x) for x in to_list(item.get("ground_truth_doc_id"))]
    keywords = [str(x) for x in to_list(item.get("must_contain")) if str(x).strip()]

    parsed = parse_query(query)
    parsed["question_type"] = "compare"

    subqueries = retriever.build_compare_subqueries_from_parsed(query, parsed)

    method_sub_doc_ranks = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
    }

    method_sub_ev_ranks = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
    }

    sub_details = []

    for idx, sub in enumerate(subqueries):
        target_doc = gt_docs[idx] if idx < len(gt_docs) else ""
        sub_query = sub["query"]

        candidate_chunks = retriever.route_chunks(sub, fallback_if_empty=True)

        dense_res = dense_candidates(
            dense_cache=dense_cache,
            query=sub_query,
            candidate_chunks=candidate_chunks,
            top_k=max_k,
        )

        bm25_res = bm25_candidates(
            bm25_cache=bm25_cache,
            query=sub_query,
            candidate_chunks=candidate_chunks,
            top_k=max_k,
        )

        hybrid_res = hybrid_candidates(
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            query=sub_query,
            candidate_chunks=candidate_chunks,
            top_k=max_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
        )

        sub_row = {
            "sub_query": sub_query,
            "target_doc": target_doc,
            "parsed": sub,
            "candidate_doc_count": len(group_candidate_docs(candidate_chunks)),
            "candidate_chunk_count": len(candidate_chunks),
        }

        for method, results in [
            ("dense", dense_res),
            ("bm25", bm25_res),
            ("hybrid", hybrid_res),
        ]:
            doc_rank = find_doc_rank(
                results=results,
                gt_doc_ids=[target_doc],
                max_k=max_k,
                require_all_docs=False,
            )

            ev_rank = find_evidence_rank(
                results=results,
                gt_doc_ids=[target_doc],
                keywords=keywords,
                question_type="fact",
                max_k=max_k,
                fact_match_mode=fact_match_mode,
                summary_match_mode="any",
                summary_min_hits=1,
                use_alias=use_alias,
                require_all_docs=False,
            )

            method_sub_doc_ranks[method].append(doc_rank)
            method_sub_ev_ranks[method].append(ev_rank)

            sub_row[f"{method}_doc_rank"] = doc_rank
            sub_row[f"{method}_evidence_rank"] = ev_rank

        sub_details.append(sub_row)

    output: Dict[str, Any] = {
        "query": query,
        "question_type": "compare",
        "ground_truth_doc_id": item.get("ground_truth_doc_id"),
        "must_contain": item.get("must_contain", []),
        "parsed": parsed,
        "sub_details": sub_details,
    }

    # compare 总 rank：所有子查询都命中，取 max rank；任一失败则 -1
    for method in ["dense", "bm25", "hybrid"]:
        doc_ranks = method_sub_doc_ranks[method]
        ev_ranks = method_sub_ev_ranks[method]

        if doc_ranks and all(r != -1 for r in doc_ranks):
            output[f"{method}_doc_rank"] = max(doc_ranks)
        else:
            output[f"{method}_doc_rank"] = -1

        if ev_ranks and all(r != -1 for r in ev_ranks):
            output[f"{method}_evidence_rank"] = max(ev_ranks)
        else:
            output[f"{method}_evidence_rank"] = -1

    return output


def eval_one_item(
    item: Dict[str, Any],
    retriever: RealEvidenceRetriever,
    dense_cache: DenseCandidateCache,
    bm25_cache: BM25CandidateCache,
    max_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    fact_match_mode: str,
    summary_match_mode: str,
    summary_min_hits: int,
    use_alias: bool,
) -> Dict[str, Any]:
    qtype = item.get("question_type", "fact")

    if qtype == "compare":
        return eval_compare(
            item=item,
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            max_k=max_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            fact_match_mode=fact_match_mode,
            use_alias=use_alias,
        )

    return eval_fact_or_summary(
        item=item,
        retriever=retriever,
        dense_cache=dense_cache,
        bm25_cache=bm25_cache,
        max_k=max_k,
        dense_candidate_k=dense_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        fact_match_mode=fact_match_mode,
        summary_match_mode=summary_match_mode,
        summary_min_hits=summary_min_hits,
        use_alias=use_alias,
    )


# ============================================================
# Metrics / report
# ============================================================

def recall_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 0.0

    return sum(1 for r in ranks if r != -1 and r <= k) / len(ranks)


def reciprocal_rank(rank: int, k: int) -> float:
    if rank == -1 or rank > k:
        return 0.0
    return 1.0 / rank


def mrr_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 0.0

    return sum(reciprocal_rank(r, k) for r in ranks) / len(ranks)


def top1_acc(ranks: Sequence[int]) -> float:
    if not ranks:
        return 0.0

    return sum(1 for r in ranks if r == 1) / len(ranks)


def fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def format_rank(r: int) -> str:
    return str(r) if r != -1 else "-1"


def format_gt_doc(gt: Any) -> str:
    if isinstance(gt, list):
        return "<br>".join(str(x) for x in gt)
    return str(gt)


def format_must_contain(must: Any) -> str:
    return "<br>".join(str(x) for x in to_list(must))


def build_main_table(
    title: str,
    details: Sequence[Dict[str, Any]],
    rank_suffix: str,
    top_k_list: Sequence[int],
) -> List[str]:
    methods = [
        ("dense", "纯向量 Dense", "metadata routing 后向量召回"),
        ("bm25", "纯 BM25", "metadata routing 后关键词召回"),
        ("hybrid", "混合召回 Hybrid", "metadata routing 后 Dense + BM25 融合"),
    ]

    lines = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| 召回方案 | " + " | ".join([f"Recall@{k}" for k in top_k_list]) + " | Top1 Acc | MRR@5 | 备注 |")
    lines.append("|----------|" + "|".join(["---:" for _ in top_k_list]) + "|---:|---:|------|")

    for method, name, note in methods:
        ranks = [int(d[f"{method}_{rank_suffix}"]) for d in details]
        recalls = [fmt(recall_at_k(ranks, k)) for k in top_k_list]
        t1 = fmt(top1_acc(ranks))
        mrr5 = fmt(mrr_at_k(ranks, 5))

        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(recalls)
            + f" | {t1} | {mrr5} | {note} |"
        )

    lines.append("")
    return lines


def build_by_type_table(
    title: str,
    details: Sequence[Dict[str, Any]],
    rank_suffix: str,
    top_k_list: Sequence[int],
) -> List[str]:
    methods = [
        ("dense", "Dense"),
        ("bm25", "BM25"),
        ("hybrid", "Hybrid"),
    ]

    by_type = defaultdict(list)
    for d in details:
        by_type[d.get("question_type", "unknown")].append(d)

    lines = []
    lines.append(f"## {title}")
    lines.append("")

    for qtype, rows in by_type.items():
        lines.append(f"### {qtype}")
        lines.append("")
        lines.append(f"- count: {len(rows)}")
        lines.append("")
        lines.append("| Method | " + " | ".join([f"Recall@{k}" for k in top_k_list]) + " | Top1 Acc | MRR@5 |")
        lines.append("|---|" + "|".join(["---:" for _ in top_k_list]) + "|---:|---:|")

        for method, name in methods:
            ranks = [int(d[f"{method}_{rank_suffix}"]) for d in rows]
            recalls = [fmt(recall_at_k(ranks, k)) for k in top_k_list]
            t1 = fmt(top1_acc(ranks))
            mrr5 = fmt(mrr_at_k(ranks, 5))

            lines.append(
                "| "
                + name
                + " | "
                + " | ".join(recalls)
                + f" | {t1} | {mrr5} |"
            )

        lines.append("")

    return lines


def build_report(
    details: Sequence[Dict[str, Any]],
    elapsed_sec: float,
    args: argparse.Namespace,
) -> str:
    lines = []

    lines.append("# Metadata Routing 基础召回器对比实验结果")
    lines.append("")
    lines.append(f"- total_queries: {len(details)}")
    lines.append(f"- elapsed_sec: {elapsed_sec:.2f}")
    lines.append("")
    lines.append("## Eval Config")
    lines.append("")
    lines.append(f"- eval_file: `{args.eval_file}`")
    lines.append(f"- chunks_file: `{args.chunks_file}`")
    lines.append(f"- faiss_index: `{args.faiss_index}`")
    lines.append(f"- faiss_metadata: `{args.faiss_metadata}`")
    lines.append(f"- model_name_or_path: `{args.model_name_or_path}`")
    lines.append(f"- top_k_list: `{args.top_k_list}`")
    lines.append(f"- dense_candidate_k: `{args.dense_candidate_k}`")
    lines.append(f"- bm25_candidate_k: `{args.bm25_candidate_k}`")
    lines.append(f"- dense_global_top_k: `{args.dense_global_top_k}`")
    lines.append(f"- dense_weight: `{args.dense_weight}`")
    lines.append(f"- bm25_weight: `{args.bm25_weight}`")
    lines.append(f"- fact_match_mode: `{args.fact_match_mode}`")
    lines.append(f"- summary_match_mode: `{args.summary_match_mode}`")
    lines.append(f"- summary_min_hits: `{args.summary_min_hits}`")
    lines.append(f"- use_metric_alias: `{not args.no_metric_alias}`")
    lines.append("")
    lines.append("> 本实验使用真实 metadata routing，不使用 ground_truth_doc_id 参与检索；ground_truth_doc_id 只用于最后评测。")
    lines.append("")

    lines.extend(build_main_table("文档级召回 Doc-level Recall", details, "doc_rank", args.top_k_list))
    lines.extend(build_main_table("证据级召回 Evidence-level Recall", details, "evidence_rank", args.top_k_list))

    lines.extend(build_by_type_table("文档级召回 by Question Type", details, "doc_rank", args.top_k_list))
    lines.extend(build_by_type_table("证据级召回 by Question Type", details, "evidence_rank", args.top_k_list))

    lines.append("## Per-query Rank")
    lines.append("")
    lines.append(
        "| Query | Type | Dense Doc | BM25 Doc | Hybrid Doc | Dense Evidence | BM25 Evidence | Hybrid Evidence | Ground Truth Doc | Must Contain |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")

    for d in details:
        lines.append(
            "| "
            + str(d["query"]).replace("|", "\\|")
            + " | "
            + str(d.get("question_type", ""))
            + " | "
            + format_rank(int(d["dense_doc_rank"]))
            + " | "
            + format_rank(int(d["bm25_doc_rank"]))
            + " | "
            + format_rank(int(d["hybrid_doc_rank"]))
            + " | "
            + format_rank(int(d["dense_evidence_rank"]))
            + " | "
            + format_rank(int(d["bm25_evidence_rank"]))
            + " | "
            + format_rank(int(d["hybrid_evidence_rank"]))
            + " | "
            + format_gt_doc(d.get("ground_truth_doc_id", "")).replace("|", "\\|")
            + " | "
            + format_must_contain(d.get("must_contain", "")).replace("|", "\\|")
            + " |"
        )

    return "\n".join(lines) + "\n"


def parse_chunk_experiment_name(name: str) -> Dict[str, Any]:
    base = Path(str(name)).stem
    m = re.search(r"cs(\d+)_ov(\d+)", base)
    return {
        "experiment": base,
        "chunk_size": int(m.group(1)) if m else "",
        "overlap": int(m.group(2)) if m else "",
    }


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def build_batch_report(
    summaries: Sequence[Dict[str, Any]],
    elapsed_sec: float,
    args: argparse.Namespace,
) -> str:
    methods = [
        ("dense", "纯向量"),
        ("bm25", "纯 BM25"),
        ("hybrid", "混合召回"),
    ]

    lines = []
    lines.append("# 九组 Chunk 参数召回对比实验")
    lines.append("")
    lines.append(f"- total_experiments: {len(summaries)}")
    lines.append(f"- elapsed_sec: {elapsed_sec:.2f}")
    lines.append(f"- eval_file: `{args.eval_file}`")
    lines.append(f"- dense_candidate_k: `{args.dense_candidate_k}`")
    lines.append(f"- bm25_candidate_k: `{args.bm25_candidate_k}`")
    lines.append(f"- dense_global_top_k: `{args.dense_global_top_k}`")
    lines.append("")

    for summary in summaries:
        exp = summary["experiment"]
        lines.append(f"## {exp}")
        lines.append("")
        lines.append(
            f"- chunks_file: `{summary['chunks_file']}`"
        )
        lines.append(
            f"- faiss_index: `{summary['faiss_index']}`"
        )
        lines.append(
            f"- faiss_metadata: `{summary['faiss_metadata']}`"
        )
        lines.append(f"- total_queries: {summary['total_queries']}")
        lines.append(f"- elapsed_sec: {summary['elapsed_sec']:.2f}")
        lines.append("")
        lines.append("| 召回方案 | Recall@3 | Recall@5 | Recall@10 |")
        lines.append("|----------|----------|----------|-----------|")

        for method, name in methods:
            lines.append(
                f"| {name} | "
                f"{fmt(summary[f'{method}_recall@3'])} | "
                f"{fmt(summary[f'{method}_recall@5'])} | "
                f"{fmt(summary[f'{method}_recall@10'])} |"
            )

        lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append(
        "| chunk 文件 | chunk_size | overlap | 召回方案 | Recall@3 | Recall@5 | Recall@10 | elapsed_sec |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")

    for summary in summaries:
        for method, name in methods:
            lines.append(
                f"| `{summary['experiment']}.jsonl` | "
                f"{summary.get('chunk_size', '')} | "
                f"{summary.get('overlap', '')} | "
                f"{name} | "
                f"{fmt(summary[f'{method}_recall@3'])} | "
                f"{fmt(summary[f'{method}_recall@5'])} | "
                f"{fmt(summary[f'{method}_recall@10'])} | "
                f"{summary['elapsed_sec']:.2f} |"
            )

    return "\n".join(lines) + "\n"


# ============================================================
# Main eval
# ============================================================

def evaluate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    t0 = time.time()

    eval_items = read_jsonl(args.eval_file)
    max_k = max(args.top_k_list)

    retriever = RealEvidenceRetriever(
        chunks_file=args.chunks_file,
        faiss_index=args.faiss_index,
        faiss_metadata=args.faiss_metadata,
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

    details = []

    for i, item in enumerate(eval_items, 1):
        print(f"[{i}/{len(eval_items)}] {item.get('question_type')} | {item.get('query')}")

        d = eval_one_item(
            item=item,
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            max_k=max_k,
            dense_candidate_k=args.dense_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            fact_match_mode=args.fact_match_mode,
            summary_match_mode=args.summary_match_mode,
            summary_min_hits=args.summary_min_hits,
            use_alias=use_alias,
        )

        details.append(d)

    elapsed_sec = time.time() - t0

    report = build_report(details, elapsed_sec, args)

    output_report = Path(args.output_report)
    output_details = Path(args.output_details)

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_details.parent.mkdir(parents=True, exist_ok=True)

    output_report.write_text(report, encoding="utf-8")
    write_jsonl(output_details, details)

    print("\n" + "=" * 100)
    print("Metadata Routing 基础召回器对比完成")
    print("=" * 100)
    print(f"report: {output_report}")
    print(f"details: {output_details}")
    print(f"elapsed_sec: {elapsed_sec:.2f}")

    return details


def summarize_experiment(
    experiment: str,
    details: Sequence[Dict[str, Any]],
    elapsed_sec: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    parsed = parse_chunk_experiment_name(experiment)
    summary: Dict[str, Any] = {
        **parsed,
        "chunks_file": args.chunks_file,
        "faiss_index": args.faiss_index,
        "faiss_metadata": args.faiss_metadata,
        "total_queries": len(details),
        "elapsed_sec": elapsed_sec,
    }

    for method in ["dense", "bm25", "hybrid"]:
        ranks = [int(d[f"{method}_evidence_rank"]) for d in details]
        for k in [3, 5, 10]:
            summary[f"{method}_recall@{k}"] = recall_at_k(ranks, k)

    return summary


def run_one_experiment(
    args: argparse.Namespace,
    experiment: str,
    shared_retriever: RealEvidenceRetriever = None,
) -> Dict[str, Any]:
    t0 = time.time()
    eval_items = read_jsonl(args.eval_file)
    max_k = max(args.top_k_list)

    if shared_retriever is None:
        retriever = RealEvidenceRetriever(
            chunks_file=args.chunks_file,
            faiss_index=args.faiss_index,
            faiss_metadata=args.faiss_metadata,
            model_name_or_path=args.model_name_or_path,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            use_fp16=not args.no_fp16,
        )
    else:
        retriever = shared_retriever
        retriever.chunks = read_jsonl(args.chunks_file)
        retriever.dense.index = retriever.dense.faiss.read_index(str(args.faiss_index))
        retriever.dense.metadata = read_jsonl(args.faiss_metadata)
        retriever.dense_weight = args.dense_weight
        retriever.bm25_weight = args.bm25_weight

        if retriever.dense.index.ntotal != len(retriever.dense.metadata):
            raise ValueError(
                "FAISS index vector count does not match metadata rows: "
                f"index.ntotal={retriever.dense.index.ntotal}, "
                f"metadata={len(retriever.dense.metadata)}"
            )

        print(f"加载 chunks: {args.chunks_file}")
        print(f"chunks 条数: {len(retriever.chunks)}")
        print(f"加载 FAISS index: {args.faiss_index}")
        print(f"index.ntotal = {retriever.dense.index.ntotal}")
        print(f"加载 metadata: {args.faiss_metadata}")
        print(f"metadata 条数 = {len(retriever.dense.metadata)}")

    dense_cache = DenseCandidateCache(
        retriever=retriever,
        global_top_k=args.dense_global_top_k,
    )
    bm25_cache = BM25CandidateCache(retriever)
    use_alias = not args.no_metric_alias

    details = []

    for i, item in enumerate(eval_items, 1):
        print(
            f"[{experiment}] [{i}/{len(eval_items)}] "
            f"{item.get('question_type')} | {item.get('query')}"
        )

        d = eval_one_item(
            item=item,
            retriever=retriever,
            dense_cache=dense_cache,
            bm25_cache=bm25_cache,
            max_k=max_k,
            dense_candidate_k=args.dense_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            fact_match_mode=args.fact_match_mode,
            summary_match_mode=args.summary_match_mode,
            summary_min_hits=args.summary_min_hits,
            use_alias=use_alias,
        )
        d["experiment"] = experiment
        details.append(d)

    elapsed_sec = time.time() - t0
    summary = summarize_experiment(
        experiment=experiment,
        details=details,
        elapsed_sec=elapsed_sec,
        args=args,
    )
    summary["details"] = details
    summary["retriever"] = retriever
    return summary


def resolve_batch_experiment_args(
    args: argparse.Namespace,
    experiment: str,
) -> argparse.Namespace:
    exp = Path(experiment).stem
    chunks_file = Path(args.chunks_dir) / f"{exp}.jsonl"
    faiss_index = Path(args.indexes_dir) / f"{exp}_flat.faiss"
    faiss_metadata = Path(args.indexes_dir) / f"{exp}_flat_metadata.jsonl"

    missing = [
        str(p)
        for p in [chunks_file, faiss_index, faiss_metadata]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Experiment {exp} missing required files:\n"
            + "\n".join(f"- {p}" for p in missing)
        )

    output_stem = f"metadata_retrieval_eval_{exp}"
    return clone_args(
        args,
        chunks_file=str(chunks_file),
        faiss_index=str(faiss_index),
        faiss_metadata=str(faiss_metadata),
        output_report=str(Path(args.batch_output_report).parent / f"{output_stem}.md"),
        output_details=str(Path(args.batch_output_details).parent / f"{output_stem}_details.jsonl"),
    )


def evaluate_batch(args: argparse.Namespace) -> None:
    t0 = time.time()
    experiments = args.experiments or DEFAULT_CHUNK_EXPERIMENTS

    summaries = []
    all_details = []
    shared_retriever = None

    for exp in experiments:
        exp_args = resolve_batch_experiment_args(args, exp)
        result = run_one_experiment(
            args=exp_args,
            experiment=Path(exp).stem,
            shared_retriever=shared_retriever,
        )

        if shared_retriever is None:
            shared_retriever = result.pop("retriever")
        else:
            result.pop("retriever", None)

        details = result.pop("details")
        all_details.extend(details)
        summaries.append(result)

    elapsed_sec = time.time() - t0

    output_report = Path(args.batch_output_report)
    output_details = Path(args.batch_output_details)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_details.parent.mkdir(parents=True, exist_ok=True)

    output_report.write_text(
        build_batch_report(summaries, elapsed_sec, args),
        encoding="utf-8",
    )
    write_jsonl(output_details, all_details)

    print("\n" + "=" * 100)
    print("九组 Chunk 参数召回对比实验完成")
    print("=" * 100)
    print(f"batch_report: {output_report}")
    print(f"batch_details: {output_details}")
    print(f"elapsed_sec: {elapsed_sec:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--chunks_file", type=str, default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--faiss_index", type=str, default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss_metadata", type=str, default=str(DEFAULT_FAISS_METADATA))
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunks_dir", type=str, default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--indexes_dir", type=str, default=str(DEFAULT_INDEXES_DIR))

    parser.add_argument("--output_report", type=str, default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output_details", type=str, default=str(DEFAULT_OUTPUT_DETAILS))
    parser.add_argument("--batch_output_report", type=str, default=str(DEFAULT_BATCH_OUTPUT_REPORT))
    parser.add_argument("--batch_output_details", type=str, default=str(DEFAULT_BATCH_OUTPUT_DETAILS))

    parser.add_argument(
        "--batch_chunk_experiments",
        action="store_true",
        help="Run the nine all_cs*_ov* chunk/index experiments and write one summary report.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=DEFAULT_CHUNK_EXPERIMENTS,
        help="Experiment base names for --batch_chunk_experiments.",
    )

    parser.add_argument("--top_k_list", type=int, nargs="+", default=[3, 5, 10])

    parser.add_argument("--dense_candidate_k", type=int, default=20)
    parser.add_argument("--bm25_candidate_k", type=int, default=20)
    parser.add_argument("--dense_global_top_k", type=int, default=300)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--bm25_weight", type=float, default=0.5)

    parser.add_argument(
        "--fact_match_mode",
        type=str,
        default="all",
        choices=["all", "any", "at_least"],
    )

    parser.add_argument(
        "--summary_match_mode",
        type=str,
        default="any",
        choices=["all", "any", "at_least"],
    )

    parser.add_argument("--summary_min_hits", type=int, default=1)

    parser.add_argument(
        "--no_metric_alias",
        action="store_true",
        help="关闭 metric alias evidence matching。默认开启。",
    )

    parser.add_argument("--no_fp16", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_chunk_experiments:
        evaluate_batch(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
