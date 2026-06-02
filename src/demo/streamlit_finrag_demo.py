# -*- coding: utf-8 -*-
"""
Streamlit demo for FinRAG.

Run on the server, for example:

CUDA_VISIBLE_DEVICES=0,7 streamlit run /9_data/ypq/FinRAG/src/demo/streamlit_finrag_demo.py \
  --server.address 0.0.0.0 \
  --server.port 8501
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [
    PROJECT_ROOT / "src/generation",
    PROJECT_ROOT / "src/evaluation",
    PROJECT_ROOT / "src/retrieval",
    PROJECT_ROOT / "src/rerank",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwen3_evidence_generator import (  # noqa: E402
    DEFAULT_CHUNKS_FILE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_FAISS_INDEX,
    DEFAULT_FAISS_METADATA,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_RERANKER_MODEL,
    FinRAGGenerationPipeline,
)


SAMPLE_QUERIES = [
    "宁德时代2025年营业收入是多少？",
    "宁德时代和比亚迪2025年谁的研发费用更高？",
    "福达股份2025年年度报告中如何描述公司的主营业务？",
    "绿岛风年报中对新风系统行业竞争格局的判断是什么？",
]


def build_args(settings: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        chunks_file=settings["chunks_file"],
        faiss_index=settings["faiss_index"],
        faiss_metadata=settings["faiss_metadata"],
        embed_model=settings["embed_model"],
        reranker_model=settings["reranker_model"],
        generator_model=settings["generator_model"],
        candidate_k=settings["candidate_k"],
        final_k=settings["final_k"],
        compare_top_k_each=settings["compare_top_k_each"],
        max_evidences=settings["max_evidences"],
        dense_global_top_k=settings["dense_global_top_k"],
        summary_dense_coarse_k=settings["summary_dense_coarse_k"],
        dense_weight=settings["dense_weight"],
        bm25_weight=settings["bm25_weight"],
        min_rerank_score=settings["min_rerank_score"],
        max_evidence_chars=settings["max_evidence_chars"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        reranker_batch_size=settings["reranker_batch_size"],
        no_fp16=settings["no_fp16"],
        slow_tokenizer=settings["slow_tokenizer"],
    )


@st.cache_resource(show_spinner="正在加载检索、重排和生成模型...")
def load_pipeline(settings_json: str) -> FinRAGGenerationPipeline:
    settings = json.loads(settings_json)
    args = build_args(settings)
    return FinRAGGenerationPipeline(args)


def show_evidence(ev: Dict[str, Any]) -> None:
    title_parts = [f"[{ev.get('id', '')}]", ev.get("doc_id", "")]
    page = ev.get("page", "")
    if page != "":
        title_parts.append(f"page={page}")
    score = ev.get("score")
    if score is not None:
        title_parts.append(f"score={float(score):.4f}")

    with st.expander(" | ".join(str(x) for x in title_parts if str(x)), expanded=False):
        meta_cols = st.columns([1, 1, 2])
        meta_cols[0].caption("Chunk ID")
        meta_cols[0].code(str(ev.get("chunk_id", "")) or "-", language=None)
        meta_cols[1].caption("Hybrid Rank")
        meta_cols[1].code(str(ev.get("hybrid_rank", "")) or "-", language=None)
        meta_cols[2].caption("Source Doc")
        meta_cols[2].code(str(ev.get("doc_id", "")) or "-", language=None)
        st.markdown(str(ev.get("text", "")))


def show_result(result: Dict[str, Any], elapsed: float) -> None:
    refused = bool(result.get("refused"))
    answer = str(result.get("answer", "")).strip()

    st.subheader("最终回答")
    if refused:
        st.warning(answer or "我不确定")
    else:
        st.success(answer)

    st.caption(f"耗时: {elapsed:.2f}s")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Question Type", str(result.get("question_type", "")))
    c2.metric("Evidence Count", len(result.get("evidences", [])))
    c3.metric("Refused", str(refused))
    max_score = None
    if result.get("evidences"):
        max_score = max(float(ev.get("score", 0.0)) for ev in result["evidences"])
    c4.metric("Max Score", "-" if max_score is None else f"{max_score:.4f}")

    st.subheader("召回文档 / 证据段落")
    st.caption(str(result.get("pipeline", "")))
    for ev in result.get("evidences", []):
        show_evidence(ev)

    st.subheader("幻觉控制")
    check = result.get("citation_check", {})
    refusal_reason = result.get("refusal_reason", {})
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Evidence Binding", "passed" if check.get("has_valid_citation") or refused else "failed")
    hc2.metric("Low-confidence Refusal", "triggered" if refused and refusal_reason else "not triggered")
    hc3.metric("Citation Verification", "passed" if check.get("passed") else "failed")

    with st.expander("Citation Check JSON", expanded=False):
        st.json(check)
    if refusal_reason:
        with st.expander("Refusal Reason JSON", expanded=False):
            st.json(refusal_reason)
    with st.expander("Raw Result JSON", expanded=False):
        st.json(result)


def sidebar_settings() -> Dict[str, Any]:
    st.sidebar.header("模型与索引")
    chunks_file = st.sidebar.text_input("chunks_file", str(DEFAULT_CHUNKS_FILE))
    faiss_index = st.sidebar.text_input("faiss_index", str(DEFAULT_FAISS_INDEX))
    faiss_metadata = st.sidebar.text_input("faiss_metadata", str(DEFAULT_FAISS_METADATA))
    embed_model = st.sidebar.text_input("embed_model", str(DEFAULT_EMBED_MODEL))
    reranker_model = st.sidebar.text_input("reranker_model", str(DEFAULT_RERANKER_MODEL))
    generator_model = st.sidebar.text_input("generator_model", str(DEFAULT_GENERATOR_MODEL))

    st.sidebar.header("检索与重排")
    candidate_k = st.sidebar.slider("Hybrid candidate_k", 5, 100, 20, 5)
    final_k = st.sidebar.slider("Final evidence top_k", 1, 10, 5, 1)
    compare_top_k_each = st.sidebar.slider("Compare top_k_each", 1, 5, 3, 1)
    max_evidences = st.sidebar.slider("Max evidences", 1, 10, 8, 1)
    dense_global_top_k = st.sidebar.slider("Dense global top_k", 50, 1000, 300, 50)
    summary_dense_coarse_k = st.sidebar.slider("Summary dense coarse_k", 100, 1500, 500, 100)
    dense_weight = st.sidebar.slider("Hybrid dense weight", 0.0, 1.0, 0.5, 0.05)
    bm25_weight = st.sidebar.slider("Hybrid BM25 weight", 0.0, 1.0, 0.5, 0.05)
    min_rerank_score = st.sidebar.number_input("低置信拒答阈值", value=-5.0, step=0.5)

    st.sidebar.header("生成")
    max_evidence_chars = st.sidebar.slider("每条证据最大字符数", 200, 2000, 900, 100)
    max_new_tokens = st.sidebar.slider("max_new_tokens", 64, 1024, 256, 64)
    temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.1, 0.05)
    top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.8, 0.05)
    reranker_batch_size = st.sidebar.slider("reranker_batch_size", 1, 64, 8, 1)
    slow_tokenizer = st.sidebar.checkbox("slow_tokenizer", value=True)
    no_fp16 = st.sidebar.checkbox("no_fp16", value=False)

    return {
        "chunks_file": chunks_file,
        "faiss_index": faiss_index,
        "faiss_metadata": faiss_metadata,
        "embed_model": embed_model,
        "reranker_model": reranker_model,
        "generator_model": generator_model,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "compare_top_k_each": compare_top_k_each,
        "max_evidences": max_evidences,
        "dense_global_top_k": dense_global_top_k,
        "summary_dense_coarse_k": summary_dense_coarse_k,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "min_rerank_score": min_rerank_score,
        "max_evidence_chars": max_evidence_chars,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reranker_batch_size": reranker_batch_size,
        "slow_tokenizer": slow_tokenizer,
        "no_fp16": no_fp16,
    }


def main() -> None:
    st.set_page_config(page_title="FinRAG Demo", page_icon="📊", layout="wide")

    st.title("FinRAG 金融研报问答 Demo")
    st.caption("输入 query，展示 Metadata-aware Rerank 召回证据、Qwen3 生成答案、引用来源与校验结果。")

    settings = sidebar_settings()
    settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True)

    if st.sidebar.button("清空模型缓存"):
        load_pipeline.clear()
        st.sidebar.success("缓存已清空，下一次提问会重新加载模型。")

    st.subheader("提问")
    selected = st.selectbox("示例问题", [""] + SAMPLE_QUERIES)
    default_query = selected or "宁德时代2025年营业收入是多少？"
    query = st.text_area("Query", value=default_query, height=90)

    run = st.button("运行问答", type="primary")
    if not run:
        st.info("点击“运行问答”后会加载模型并生成答案。首次加载会比较慢。")
        return

    if not query.strip():
        st.warning("请输入 query。")
        return

    with st.spinner("正在检索、重排并生成答案..."):
        t0 = time.time()
        pipeline = load_pipeline(settings_json)
        result = pipeline.answer(query.strip())
        elapsed = time.time() - t0

    show_result(result, elapsed)


if __name__ == "__main__":
    main()
