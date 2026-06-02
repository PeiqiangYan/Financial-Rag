# -*- coding: utf-8 -*-
"""
metadata_hybrid_retriever.py

真实检索版 evidence retriever，不依赖 ground_truth_doc_id。

核心流程：
1. 从 query 自动解析 metadata:
   - company_short
   - year
   - period
   - metric
   - question_type

2. fact:
   query -> metadata routing -> 候选文档 chunks -> 文档内 evidence retrieval

3. compare:
   A 和 B 谁的研发费用更高？
   -> A 2025年研发费用是多少？
   -> B 2025年研发费用是多少？
   -> 每个子查询分别 metadata routing + evidence retrieval

4. summary:
   如果能识别 company/year，则 metadata routing
   否则全库 hybrid retrieval

默认路径适配当前项目根目录，可通过命令行参数覆盖。
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat.faiss"
DEFAULT_FAISS_METADATA = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat_metadata.jsonl"
DEFAULT_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--BAAI--bge-large-zh-v1.5/"
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116"
)


# ============================================================
# Company alias
# ============================================================

COMPANY_ALIAS = {
    "宁德时代新能源科技股份有限公司": "宁德时代",
    "宁德时代": "宁德时代",

    "比亚迪股份有限公司": "比亚迪",
    "比亚迪": "比亚迪",

    "重庆长安汽车股份有限公司": "长安汽车",
    "长安汽车": "长安汽车",

    "华域汽车系统股份有限公司": "华域汽车",
    "华域汽车": "华域汽车",

    "宁波均胜电子股份有限公司": "均胜电子",
    "均胜电子": "均胜电子",

    "宁波拓普集团股份有限公司": "拓普集团",
    "拓普集团": "拓普集团",

    "浙江华培动力科技股份有限公司": "华培动力",
    "华培动力": "华培动力",

    "桂林福达股份有限公司": "福达股份",
    "福达股份": "福达股份",

    "厦门厦钨新能源材料股份有限公司": "厦钨新能源",
    "厦钨新能源": "厦钨新能源",

    "阿特斯阳光电力集团股份有限公司": "阿特斯阳光电力",
    "阿特斯阳光电力": "阿特斯阳光电力",
    "阿特斯": "阿特斯阳光电力",

    "芯海科技": "芯海科技",
    "英集芯": "英集芯",
    "科蓝软件": "科蓝软件",
    "东华软件": "东华软件",
    "奇安信": "奇安信",
    "科大讯飞": "科大讯飞",
    "龙芯中科": "龙芯中科",
    "三只松鼠": "三只松鼠",
    "来伊份": "来伊份",
    "阳光乳业": "阳光乳业",
    "凯撒旅业": "凯撒旅业",
    "众信旅游": "众信旅游",
    "君亭酒店": "君亭酒店",
    "浙江新能": "浙江新能",
    "龙源电力": "龙源电力",
    "陕西能源": "陕西能源",
    "新集能源": "新集能源",
    "中信博": "中信博",
    "上声电子": "上声电子",
    "苏州上声电子股份有限公司": "上声电子",
    "合兴汽车电子": "合兴汽车电子",
    "合兴汽车电子股份有限公司": "合兴汽车电子",
}


METRIC_PATTERNS = [
    "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额",
    "经营活动现金流入",
    "主营业务收入",
    "营业收入",
    "研发费用",
    "研发投入",
    "研发投入占营业收入比例",
    "毛利率",
    "资产负债率",
    "总资产",
    "资产总计",
    "归母净利润",
    "基本每股收益",
    "风电发电量",
    "保证借款",
    "动力电池销量",
    "储能电池销量",
    "正极材料销量",
    "组件出货量",
    "汽车销量",
    "销量",
    "出货量",
    "收入",
    "利润",
]


METRIC_ALIASES = {
    "归母净利润": ["归母净利润", "归属于上市公司股东的净利润"],
    "归属于上市公司股东的净利润": ["归属于上市公司股东的净利润", "归母净利润"],
    "研发投入": ["研发投入", "研发费用"],
    "研发费用": ["研发费用", "研发投入"],
    "总资产": ["总资产", "资产总计", "资产总额"],
    "资产总计": ["资产总计", "总资产", "资产总额"],
    "主营业务收入": ["主营业务收入", "营业收入"],
    "营业收入": ["营业收入", "主营业务收入", "收入"],
    "经营活动现金流入": ["经营活动现金流入", "经营活动现金流量流入"],
    "经营活动产生的现金流量净额": [
        "经营活动产生的现金流量净额",
        "经营活动现金流量净额",
        "经营活动产生的现金流量",
    ],
    "保证借款": ["保证借款"],
    "毛利率": ["毛利率"],
    "资产负债率": ["资产负债率"],
    "基本每股收益": ["基本每股收益"],
    "风电发电量": ["风电", "发电量"],
    "动力电池销量": ["动力电池销量", "动力电池", "销量"],
    "储能电池销量": ["储能电池销量", "储能电池", "销量"],
    "组件出货量": ["组件", "出货量"],
    "正极材料销量": ["正极材料", "销量"],
}


# ============================================================
# IO / normalize
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


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def normalize_doc_string(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\.(pdf|md|txt|jsonl)$", "", s, flags=re.I)
    s = s.replace("【洞见研报DJyanbao.com】", "")
    s = s.replace("【深交所】", "")
    s = s.replace("【上交所】", "")
    s = s.replace("【上交所科创板】", "")
    s = s.replace("：", "")
    s = s.replace(":", "")
    s = s.replace(" ", "")
    s = s.replace("\u3000", "")
    return s.lower()


def get_doc_display_id(row: Dict[str, Any]) -> str:
    return (
        row.get("doc_id")
        or row.get("doc_title")
        or row.get("source_file")
        or ""
    )


def get_chunk_text(row: Dict[str, Any]) -> str:
    return row.get("content", "") or ""


def dedup_by_chunk_id(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = str(r.get("chunk_id") or r.get("retrieval_index") or id(r))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ============================================================
# Query understanding
# ============================================================

def extract_year(query: str) -> str:
    m = re.search(r"(20\d{2})", query)
    return m.group(1) if m else ""


def extract_period(query: str) -> str:
    if re.search(r"(半年度|中期|上半年|半年报|半年度报告)", query):
        return "semiannual"

    if re.search(r"(第一季度|一季度|第1季度|1季度|Q1|q1)", query):
        return "q1"

    if re.search(r"(第三季度|三季度|第3季度|3季度|Q3|q3)", query):
        return "q3"

    if re.search(r"(年度|全年|年报|年度报告)", query):
        return "annual"

    # 财报问法里出现 2025年xxx，一般默认年度报告
    if re.search(r"20\d{2}年", query):
        return "annual"

    return ""


def extract_metric(query: str) -> str:
    for m in METRIC_PATTERNS:
        if m in query:
            return m
    return ""


def extract_companies(query: str) -> List[str]:
    hits: List[Tuple[int, str]] = []
    for alias, short in COMPANY_ALIAS.items():
        pos = query.find(alias)
        if pos != -1:
            hits.append((pos, short))

    hits.sort(key=lambda x: x[0])

    out = []
    for _, short in hits:
        if short not in out:
            out.append(short)

    return out


def infer_question_type(query: str, companies: Sequence[str]) -> str:
    compare_markers = ["谁更高", "哪个高", "谁的", "相比", "对比", "vs", "VS", "和"]
    if len(companies) >= 2 and any(m in query for m in compare_markers):
        return "compare"

    summary_markers = ["总结", "概括", "如何描述", "怎么看待", "有哪些", "主要", "判断", "趋势", "支持方向"]
    if any(m in query for m in summary_markers):
        return "summary"

    return "fact"


def parse_query(query: str) -> Dict[str, Any]:
    companies = extract_companies(query)
    year = extract_year(query)
    period = extract_period(query)
    metric = extract_metric(query)
    qtype = infer_question_type(query, companies)

    return {
        "query": query,
        "companies": companies,
        "company_short": companies[0] if companies else "",
        "year": year,
        "period": period,
        "metric": metric,
        "question_type": qtype,
    }


def metric_keywords(metric: str) -> List[str]:
    if not metric:
        return []
    return METRIC_ALIASES.get(metric, [metric])


# ============================================================
# Metadata matching
# ============================================================

def chunk_company_value(chunk: Dict[str, Any]) -> str:
    return (
        chunk.get("company_short")
        or chunk.get("company")
        or chunk.get("doc_title")
        or chunk.get("doc_id")
        or chunk.get("source_file")
        or ""
    )


def chunk_year_value(chunk: Dict[str, Any]) -> str:
    if chunk.get("year"):
        return str(chunk.get("year"))
    text = " ".join([
        str(chunk.get("doc_id", "")),
        str(chunk.get("doc_title", "")),
        str(chunk.get("source_file", "")),
    ])
    m = re.search(r"(20\d{2})", text)
    return m.group(1) if m else ""


def chunk_period_value(chunk: Dict[str, Any]) -> str:
    if chunk.get("period"):
        return str(chunk.get("period"))

    text = " ".join([
        str(chunk.get("doc_id", "")),
        str(chunk.get("doc_title", "")),
        str(chunk.get("source_file", "")),
    ])

    if re.search(r"(半年度报告|中期报告|半年报)", text):
        return "semiannual"
    if re.search(r"(第一季度报告|一季报|第1季度报告|1季报|q1)", text, flags=re.I):
        return "q1"
    if re.search(r"(第三季度报告|三季报|第3季度报告|3季报|q3)", text, flags=re.I):
        return "q3"
    if re.search(r"(年度报告全文|年度报告|年报)", text):
        return "annual"

    return ""


def company_matches(chunk: Dict[str, Any], company_short: str) -> bool:
    if not company_short:
        return True

    target = normalize_text(company_short)
    val = normalize_text(chunk_company_value(chunk))
    doc = normalize_text(get_doc_display_id(chunk))
    source = normalize_text(chunk.get("source_file", ""))

    return target in val or target in doc or target in source


def year_matches(chunk: Dict[str, Any], year: str) -> bool:
    if not year:
        return True
    return str(chunk_year_value(chunk)) == str(year)


def period_matches(chunk: Dict[str, Any], period: str) -> bool:
    if not period:
        return True

    cp = chunk_period_value(chunk)

    # 如果 chunk 没有 period 信息，不强杀，避免误过滤；
    # 但如果有，就必须一致。
    if not cp:
        return True

    return cp == period


def metadata_match(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    company_short = filters.get("company_short", "")
    year = filters.get("year", "")
    period = filters.get("period", "")

    if company_short and not company_matches(chunk, company_short):
        return False

    if year and not year_matches(chunk, year):
        return False

    if period and not period_matches(chunk, period):
        return False

    return True


def filter_chunks_by_metadata(
    chunks: Sequence[Dict[str, Any]],
    filters: Dict[str, Any],
    fallback_if_empty: bool = True,
) -> List[Dict[str, Any]]:
    filtered = [ch for ch in chunks if metadata_match(ch, filters)]

    # 如果 period 太严格导致空，放宽 period
    if not filtered and filters.get("period"):
        loose_filters = dict(filters)
        loose_filters["period"] = ""
        filtered = [ch for ch in chunks if metadata_match(ch, loose_filters)]

    # 如果仍然空，按 company/year 失败，返回全库兜底
    if not filtered and fallback_if_empty:
        return list(chunks)

    return filtered


def group_candidate_docs(chunks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        doc_id = get_doc_display_id(ch)
        if not doc_id:
            doc_id = str(ch.get("source_file", "unknown"))
        docs.setdefault(doc_id, []).append(ch)
    return docs


# ============================================================
# Dense / BM25 / Hybrid
# ============================================================

class DenseSearcher:
    def __init__(
        self,
        faiss_index_path: Union[str, Path],
        metadata_path: Union[str, Path],
        model_name_or_path: Union[str, Path],
        use_fp16: bool = True,
        query_instruction: Optional[str] = None,
    ) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError("faiss 未安装，请先安装 faiss-cpu 或 faiss-gpu。") from e

        try:
            from FlagEmbedding import FlagModel  # type: ignore
        except ImportError as e:
            raise ImportError("FlagEmbedding 未安装，请先安装 FlagEmbedding。") from e

        self.faiss = faiss
        self.index = faiss.read_index(str(faiss_index_path))
        self.metadata = read_jsonl(metadata_path)
        self.query_instruction = query_instruction

        print(f"加载 FAISS index: {faiss_index_path}")
        print(f"index.ntotal = {self.index.ntotal}")
        print(f"加载 metadata: {metadata_path}")
        print(f"metadata 条数 = {len(self.metadata)}")
        print(f"加载 embedding 模型: {model_name_or_path}")

        self.model = FlagModel(
            str(model_name_or_path),
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16,
        )

    @staticmethod
    def l2_normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype="float32")
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return x / norm

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query])
        emb = np.asarray(emb, dtype="float32")
        emb = self.l2_normalize(emb)
        return emb

    def search_global(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        q_emb = self.encode_query(query)
        scores, ids = self.index.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = dict(self.metadata[idx])
            row["dense_score"] = float(score)
            row["score"] = float(score)
            row["retrieval_index"] = int(idx)
            row["rank_source"] = "dense"
            results.append(row)
        return results


class BM25LocalSearcher:
    def __init__(self) -> None:
        try:
            import jieba  # type: ignore
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as e:
            raise ImportError("请先安装 jieba 和 rank_bm25。") from e

        self.jieba = jieba
        self.BM25Okapi = BM25Okapi

    def tokenize(self, text: str) -> List[str]:
        text = text or ""
        text = text.replace("\n", " ")

        special_tokens = re.findall(
            r"\d[\d,，\.]*%?|\d[\d,，\.]*(?:亿元|万元|千元|元|GWh|GW|MW|兆瓦时|万千瓦时|万辆|万吨)",
            text,
            flags=re.I,
        )

        words = list(self.jieba.cut(text))
        tokens = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            tokens.append(w)

        tokens.extend(special_tokens)
        return tokens

    def search(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        corpus = [self.tokenize(get_chunk_text(ch)) for ch in chunks]
        bm25 = self.BM25Okapi(corpus)
        q_tokens = self.tokenize(query)
        scores = bm25.get_scores(q_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            row = dict(chunks[int(idx)])
            row["bm25_score"] = score
            row["score"] = score
            row["rank_source"] = "bm25_local"
            results.append(row)

        return results


def min_max_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}

    vals = list(score_map.values())
    mn, mx = min(vals), max(vals)

    if math.isclose(mx, mn):
        return {k: 1.0 for k in score_map}

    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def chunk_key(row: Dict[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("retrieval_index") or id(row))


def fuse_dense_bm25(
    dense_results: Sequence[Dict[str, Any]],
    bm25_results: Sequence[Dict[str, Any]],
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    dense_scores: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}

    for r in dense_results:
        key = chunk_key(r)
        rows[key] = dict(r)
        dense_scores[key] = float(r.get("dense_score", r.get("score", 0.0)))

    for r in bm25_results:
        key = chunk_key(r)
        if key not in rows:
            rows[key] = dict(r)
        else:
            rows[key].update({k: v for k, v in r.items() if k not in rows[key]})
        bm25_scores[key] = float(r.get("bm25_score", r.get("score", 0.0)))

    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize(bm25_scores)

    fused: List[Dict[str, Any]] = []
    for key, row in rows.items():
        dn = dense_norm.get(key, 0.0)
        bn = bm25_norm.get(key, 0.0)
        score = dense_weight * dn + bm25_weight * bn

        out = dict(row)
        out["dense_norm"] = dn
        out["bm25_norm"] = bn
        out["hybrid_score"] = score
        out["score"] = score
        out["rank_source"] = "hybrid"
        fused.append(out)

    fused.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), reverse=True)
    return fused[:top_k]


# ============================================================
# Evidence ranking inside candidate chunks
# ============================================================

def evidence_keyword_score(content: str, keywords: Sequence[str]) -> float:
    content_norm = normalize_text(content)
    score = 0.0

    for kw in keywords:
        kw_norm = normalize_text(kw)
        if not kw_norm:
            continue

        if kw_norm in content_norm:
            score += 10.0

    return score


def query_token_score(query: str, content: str) -> float:
    content_norm = normalize_text(content)
    query_norm = normalize_text(query)

    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9%\.]+", query_norm)

    score = 0.0
    for tok in tokens:
        if len(tok) >= 2 and tok in content_norm:
            score += 1.0

    return score


def rerank_by_evidence_signal(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    keywords: Sequence[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    scored = []

    for ch in chunks:
        content = get_chunk_text(ch)
        score = 0.0
        score += evidence_keyword_score(content, keywords)
        score += query_token_score(query, content)

        # 保留原 retrieval score 的轻微影响
        score += 0.1 * float(ch.get("score", 0.0))

        if score > 0:
            row = dict(ch)
            row["evidence_score"] = score
            row["score"] = score
            scored.append(row)

    scored.sort(key=lambda x: x.get("evidence_score", x.get("score", 0.0)), reverse=True)
    return scored[:top_k]


def has_any_keyword(content: str, keywords: Sequence[str]) -> bool:
    content_norm = normalize_text(content)
    return any(normalize_text(kw) in content_norm for kw in keywords if normalize_text(kw))


def has_all_keyword_groups(content: str, keywords: Sequence[str]) -> bool:
    content_norm = normalize_text(content)
    for kw in keywords:
        kw_norm = normalize_text(kw)
        if not kw_norm:
            continue
        if kw_norm not in content_norm:
            return False
    return True


# ============================================================
# Real retriever
# ============================================================

class RealEvidenceRetriever:
    def __init__(
        self,
        chunks_file: Union[str, Path],
        faiss_index: Union[str, Path],
        faiss_metadata: Union[str, Path],
        model_name_or_path: Union[str, Path],
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_fp16: bool = True,
    ) -> None:
        self.chunks = read_jsonl(chunks_file)
        self.dense = DenseSearcher(
            faiss_index_path=faiss_index,
            metadata_path=faiss_metadata,
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
        )
        self.bm25 = BM25LocalSearcher()

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        print(f"加载 chunks: {chunks_file}")
        print(f"chunks 条数: {len(self.chunks)}")

    # --------------------------------------------------------
    # metadata routing
    # --------------------------------------------------------

    def route_chunks(
        self,
        parsed: Dict[str, Any],
        fallback_if_empty: bool = True,
    ) -> List[Dict[str, Any]]:
        filters = {
            "company_short": parsed.get("company_short", ""),
            "year": parsed.get("year", ""),
            "period": parsed.get("period", ""),
        }

        routed = filter_chunks_by_metadata(
            self.chunks,
            filters=filters,
            fallback_if_empty=fallback_if_empty,
        )

        return routed

    def route_docs(
        self,
        parsed: Dict[str, Any],
        fallback_if_empty: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        routed_chunks = self.route_chunks(parsed, fallback_if_empty=fallback_if_empty)
        return group_candidate_docs(routed_chunks)

    # --------------------------------------------------------
    # dense candidate after metadata routing
    # --------------------------------------------------------

    def dense_search_with_metadata_filter(
        self,
        query: str,
        candidate_chunks: Sequence[Dict[str, Any]],
        global_top_k: int = 200,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        FAISS 本身是全库索引。
        这里做法：
        1. 先全库 dense 检索 global_top_k
        2. 再按 candidate_chunks 的 chunk_id / doc_id 过滤
        """
        if not candidate_chunks:
            return []

        candidate_keys = set(chunk_key(ch) for ch in candidate_chunks)
        candidate_doc_ids = set(get_doc_display_id(ch) for ch in candidate_chunks)

        global_results = self.dense.search_global(query, top_k=global_top_k)

        filtered = []
        for r in global_results:
            key = chunk_key(r)
            doc_id = get_doc_display_id(r)
            if key in candidate_keys or doc_id in candidate_doc_ids:
                filtered.append(r)

        return filtered[:top_k]

    # --------------------------------------------------------
    # evidence retrieval in candidate chunks
    # --------------------------------------------------------

    def retrieve_evidence_from_chunks(
        self,
        query: str,
        candidate_chunks: Sequence[Dict[str, Any]],
        metric: str = "",
        top_k: int = 10,
        dense_candidate_k: int = 50,
        bm25_candidate_k: int = 50,
        dense_global_top_k: int = 300,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        keywords = metric_keywords(metric)

        bm25_results = self.bm25.search(
            query=query,
            chunks=candidate_chunks,
            top_k=bm25_candidate_k,
        )

        dense_results = self.dense_search_with_metadata_filter(
            query=query,
            candidate_chunks=candidate_chunks,
            global_top_k=dense_global_top_k,
            top_k=dense_candidate_k,
        )

        if use_hybrid:
            retrieval_results = fuse_dense_bm25(
                dense_results=dense_results,
                bm25_results=bm25_results,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                top_k=max(dense_candidate_k, bm25_candidate_k),
            )
        else:
            retrieval_results = dedup_by_chunk_id(list(bm25_results) + list(dense_results))

        # 再用 evidence signal 重排
        reranked = rerank_by_evidence_signal(
            query=query,
            chunks=retrieval_results,
            keywords=keywords,
            top_k=top_k,
        )

        # 如果 rerank 后为空，用 BM25 兜底
        if not reranked:
            reranked = bm25_results[:top_k]

        return reranked[:top_k]

    # --------------------------------------------------------
    # fact
    # --------------------------------------------------------

    def retrieve_fact(
        self,
        query: str,
        top_k: int = 10,
        dense_candidate_k: int = 50,
        bm25_candidate_k: int = 50,
        dense_global_top_k: int = 300,
    ) -> Dict[str, Any]:
        parsed = parse_query(query)
        parsed["question_type"] = "fact"

        candidate_chunks = self.route_chunks(parsed, fallback_if_empty=True)

        evidence = self.retrieve_evidence_from_chunks(
            query=query,
            candidate_chunks=candidate_chunks,
            metric=parsed.get("metric", ""),
            top_k=top_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            dense_global_top_k=dense_global_top_k,
            use_hybrid=True,
        )

        return {
            "query": query,
            "question_type": "fact",
            "parsed": parsed,
            "candidate_chunk_count": len(candidate_chunks),
            "candidate_doc_count": len(group_candidate_docs(candidate_chunks)),
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # compare
    # --------------------------------------------------------

    def build_compare_subqueries_from_parsed(self, query: str, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        companies = parsed.get("companies", [])
        year = parsed.get("year", "")
        period = parsed.get("period", "")
        metric = parsed.get("metric", "")

        if len(companies) < 2:
            return []

        subqueries = []
        for company in companies[:2]:
            if year and metric:
                subq = f"{company}{year}年{metric}是多少？"
            elif metric:
                subq = f"{company}{metric}是多少？"
            else:
                subq = f"{company} {query}"

            subqueries.append(
                {
                    "company_short": company,
                    "year": year,
                    "period": period,
                    "metric": metric,
                    "query": subq,
                    "question_type": "fact",
                }
            )

        return subqueries

    def retrieve_compare(
        self,
        query: str,
        top_k_each: int = 5,
        dense_candidate_k: int = 50,
        bm25_candidate_k: int = 50,
        dense_global_top_k: int = 300,
    ) -> Dict[str, Any]:
        parsed = parse_query(query)
        parsed["question_type"] = "compare"

        subqueries = self.build_compare_subqueries_from_parsed(query, parsed)

        sub_results = []
        for sub in subqueries:
            candidate_chunks = self.route_chunks(sub, fallback_if_empty=True)

            evidence = self.retrieve_evidence_from_chunks(
                query=sub["query"],
                candidate_chunks=candidate_chunks,
                metric=sub.get("metric", ""),
                top_k=top_k_each,
                dense_candidate_k=dense_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                dense_global_top_k=dense_global_top_k,
                use_hybrid=True,
            )

            sub_results.append(
                {
                    "sub_query": sub["query"],
                    "parsed": sub,
                    "candidate_chunk_count": len(candidate_chunks),
                    "candidate_doc_count": len(group_candidate_docs(candidate_chunks)),
                    "evidence": evidence,
                }
            )

        return {
            "query": query,
            "question_type": "compare",
            "parsed": parsed,
            "sub_results": sub_results,
        }

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    def retrieve_summary(
        self,
        query: str,
        top_k: int = 10,
        dense_candidate_k: int = 80,
        bm25_candidate_k: int = 80,
        dense_global_top_k: int = 500,
    ) -> Dict[str, Any]:
        parsed = parse_query(query)
        parsed["question_type"] = "summary"

        # summary 如果识别到公司/year，则做弱 metadata routing；
        # 如果识别不到，则走全库。
        has_routing_signal = bool(parsed.get("company_short") or parsed.get("year"))

        if has_routing_signal:
            candidate_chunks = self.route_chunks(parsed, fallback_if_empty=True)
        else:
            candidate_chunks = self.chunks

        evidence = self.retrieve_evidence_from_chunks(
            query=query,
            candidate_chunks=candidate_chunks,
            metric=parsed.get("metric", ""),
            top_k=top_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            dense_global_top_k=dense_global_top_k,
            use_hybrid=True,
        )

        return {
            "query": query,
            "question_type": "summary",
            "parsed": parsed,
            "candidate_chunk_count": len(candidate_chunks),
            "candidate_doc_count": len(group_candidate_docs(candidate_chunks)),
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # auto
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_k_each: int = 5,
        dense_candidate_k: int = 50,
        bm25_candidate_k: int = 50,
        dense_global_top_k: int = 300,
    ) -> Dict[str, Any]:
        parsed = parse_query(query)
        qtype = parsed.get("question_type", "fact")

        if qtype == "compare":
            return self.retrieve_compare(
                query=query,
                top_k_each=top_k_each,
                dense_candidate_k=dense_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                dense_global_top_k=dense_global_top_k,
            )

        if qtype == "summary":
            return self.retrieve_summary(
                query=query,
                top_k=top_k,
                dense_candidate_k=dense_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                dense_global_top_k=max(dense_global_top_k, 500),
            )

        return self.retrieve_fact(
            query=query,
            top_k=top_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            dense_global_top_k=dense_global_top_k,
        )


# ============================================================
# Display
# ============================================================

def short_text(text: str, max_len: int = 350) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def print_evidence_list(evidence: Sequence[Dict[str, Any]]) -> None:
    for i, r in enumerate(evidence, 1):
        print("-" * 100)
        print(f"Rank {i}")
        print(f"doc_id: {get_doc_display_id(r)}")
        print(f"chunk_id: {r.get('chunk_id', '')}")
        print(f"score: {r.get('score', '')}")
        print(f"rank_source: {r.get('rank_source', '')}")
        print(short_text(get_chunk_text(r)))


def print_result(result: Dict[str, Any]) -> None:
    print("=" * 100)
    print(f"Query: {result.get('query')}")
    print(f"Question Type: {result.get('question_type')}")
    print(f"Parsed: {json.dumps(result.get('parsed', {}), ensure_ascii=False)}")
    print("=" * 100)

    if result.get("question_type") == "compare":
        for sub in result.get("sub_results", []):
            print("\n" + "=" * 100)
            print(f"Sub Query: {sub.get('sub_query')}")
            print(f"Parsed: {json.dumps(sub.get('parsed', {}), ensure_ascii=False)}")
            print(f"candidate_doc_count: {sub.get('candidate_doc_count')}")
            print(f"candidate_chunk_count: {sub.get('candidate_chunk_count')}")
            print_evidence_list(sub.get("evidence", []))
        return

    print(f"candidate_doc_count: {result.get('candidate_doc_count')}")
    print(f"candidate_chunk_count: {result.get('candidate_chunk_count')}")
    print_evidence_list(result.get("evidence", []))


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks_file", type=str, default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--faiss_index", type=str, default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss_metadata", type=str, default=str(DEFAULT_FAISS_METADATA))
    parser.add_argument("--model_name_or_path", type=str, default=str(DEFAULT_MODEL))

    parser.add_argument("--query", type=str, required=True)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_k_each", type=int, default=5)

    parser.add_argument("--dense_candidate_k", type=int, default=50)
    parser.add_argument("--bm25_candidate_k", type=int, default=50)
    parser.add_argument("--dense_global_top_k", type=int, default=300)

    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--bm25_weight", type=float, default=0.5)

    parser.add_argument("--no_fp16", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.time()

    retriever = RealEvidenceRetriever(
        chunks_file=args.chunks_file,
        faiss_index=args.faiss_index,
        faiss_metadata=args.faiss_metadata,
        model_name_or_path=args.model_name_or_path,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        use_fp16=not args.no_fp16,
    )

    result = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        top_k_each=args.top_k_each,
        dense_candidate_k=args.dense_candidate_k,
        bm25_candidate_k=args.bm25_candidate_k,
        dense_global_top_k=args.dense_global_top_k,
    )

    print_result(result)

    print("\n" + "=" * 100)
    print(f"elapsed_sec: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
