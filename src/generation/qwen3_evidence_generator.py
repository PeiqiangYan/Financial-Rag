# -*- coding: utf-8 -*-
"""
qwen3_evidence_generator.py

Generation layer for FinRAG.

Pipeline:
1. Retrieve evidence with the final Metadata-aware Rerank pipeline:
   Hybrid Top20 -> metadata-aware bge-reranker-v2-m3 -> Top5
   summary: Dense Top500 -> Local Hybrid Top20 -> metadata-aware rerank Top5
2. Refuse low-confidence queries when all retrieved evidence scores are below
   a configurable threshold.
3. Generate with Qwen3-8B using evidence-bound prompting.
4. Verify citations after generation. If the answer does not cite valid
   evidence ids, return "我不确定" instead of an unsupported answer.

Hallucination mitigation:
- evidence binding
- low-confidence refusal
- post-generation citation verification
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data/chunks/all_cs1024_ov50.jsonl"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat.faiss"
DEFAULT_FAISS_METADATA = PROJECT_ROOT / "data/indexes/all_cs1024_ov50_flat_metadata.jsonl"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "experiments/qwen3_generation_results.jsonl"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "experiments/qwen3_generation_results.md"

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
DEFAULT_GENERATOR_MODEL = (
    "/9_data/ypq/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)


for p in [
    PROJECT_ROOT / "src/retrieval",
    PROJECT_ROOT / "src/evaluation",
    PROJECT_ROOT / "src/rerank",
]:
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
)
from metadata_aware_reranker import (  # noqa: E402
    BGEReranker,
    build_compare_subqueries,
    metadata_aware_rerank,
    route_candidate_chunks_for_query,
)


REFUSAL_TEXT = "我不确定"


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


def truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def evidence_page(row: Dict[str, Any]) -> str:
    page = row.get("page", row.get("page_no", row.get("page_index", "")))
    return "" if page is None else str(page)


def evidence_chunk_id(row: Dict[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("id") or row.get("retrieval_index") or "")


def dedup_evidences(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = (
            evidence_chunk_id(row)
            or f"{get_doc_display_id(row)}::{evidence_page(row)}::{get_chunk_text(row)[:80]}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def format_evidence(row: Dict[str, Any], idx: int, max_chars: int) -> Dict[str, Any]:
    return {
        "id": f"E{idx}",
        "doc_id": get_doc_display_id(row),
        "page": evidence_page(row),
        "chunk_id": evidence_chunk_id(row),
        "score": float(row.get("reranker_score", row.get("score", 0.0)) or 0.0),
        "hybrid_rank": row.get("hybrid_rank", ""),
        "text": truncate_text(get_chunk_text(row), max_chars=max_chars),
    }


def render_evidence_block(evidences: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for ev in evidences:
        meta = f"来源: {ev['doc_id']}"
        if ev.get("page") != "":
            meta += f", 页码: {ev['page']}"
        if ev.get("chunk_id"):
            meta += f", chunk_id: {ev['chunk_id']}"
        blocks.append(
            f"[{ev['id']}]\n"
            f"{meta}\n"
            f"证据内容: {ev['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, evidences: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence_block = render_evidence_block(evidences)
    system = (
        "你是一个金融研报和财报问答助手。必须严格基于给定证据回答。"
        "禁止使用证据外信息，禁止编造数字、年份、公司名称或结论。"
        "每个关键结论后必须标注证据编号，例如 [E1]、[E2]。"
        "如果证据不足以回答问题，只回答“我不确定”。"
    )
    user = (
        f"问题：{query}\n\n"
        f"证据：\n{evidence_block}\n\n"
        "请基于上面的证据回答问题。要求：\n"
        "1. 答案必须简洁、直接。\n"
        "2. 涉及数字、指标、公司对比或总结性结论时，句末必须引用证据编号。\n"
        "3. 如果证据无法支持答案，只回答“我不确定”。\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_citations(answer: str) -> List[str]:
    found = re.findall(r"\[(E\d+)\]", answer)
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def validate_citations(answer: str, evidences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_ids = {ev["id"] for ev in evidences}
    citations = extract_citations(answer)
    invalid = [x for x in citations if x not in valid_ids]
    is_refusal = answer.strip().startswith(REFUSAL_TEXT)
    return {
        "is_refusal": is_refusal,
        "citations": citations,
        "invalid_citations": invalid,
        "has_valid_citation": any(x in valid_ids for x in citations),
        "passed": is_refusal or (bool(citations) and not invalid and any(x in valid_ids for x in citations)),
    }


def apply_citation_guard(answer: str, evidences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    check = validate_citations(answer, evidences)
    if check["passed"]:
        return {
            "answer": answer.strip(),
            "citation_check": check,
            "post_check_action": "pass",
        }
    return {
        "answer": REFUSAL_TEXT,
        "citation_check": check,
        "post_check_action": "refuse_invalid_or_missing_citation",
    }


class Qwen3Generator:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        tokenizer_use_fast: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("需要安装 transformers 和 torch 才能加载 Qwen3-8B。") from e

        print(f"加载生成模型: {model_name_or_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_fast=tokenizer_use_fast,
            )
        except Exception as e:
            if not tokenizer_use_fast:
                raise
            print(
                "Fast tokenizer 加载失败，尝试 use_fast=False 慢 tokenizer。"
                f"原始错误: {type(e).__name__}: {e}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_fast=False,
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError as e:
            msg = str(e)
            if "qwen3" in msg.lower() and "does not recognize" in msg.lower():
                raise RuntimeError(
                    "当前 transformers 版本不支持 Qwen3。请在 finrag 环境中升级：\n"
                    "python -m pip install -U \"transformers>=4.51.0\" "
                    "\"tokenizers>=0.21.0\" accelerate safetensors\n"
                    "如果服务器不能直连 PyPI，可以加镜像源，例如：\n"
                    "python -m pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple "
                    "\"transformers>=4.51.0\" \"tokenizers>=0.21.0\" accelerate safetensors"
                ) from e
            raise
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.torch = torch

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs.input_ids.shape[-1]:]
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer.strip()


class FinRAGGenerationPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.retriever = RealEvidenceRetriever(
            chunks_file=args.chunks_file,
            faiss_index=args.faiss_index,
            faiss_metadata=args.faiss_metadata,
            model_name_or_path=args.embed_model,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            use_fp16=not args.no_fp16,
        )
        self.reranker = BGEReranker(
            model_name_or_path=args.reranker_model,
            use_fp16=not args.no_fp16,
            batch_size=args.reranker_batch_size,
        )
        self.dense_cache = DenseCandidateCache(
            self.retriever,
            global_top_k=max(args.dense_global_top_k, args.summary_dense_coarse_k),
        )
        self.bm25_cache = BM25CandidateCache(self.retriever)
        self.generator = Qwen3Generator(
            model_name_or_path=args.generator_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_use_fast=not args.slow_tokenizer,
        )

    def _hybrid_candidates(
        self,
        query: str,
        question_type: str,
        candidate_chunks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if question_type == "summary":
            coarse_chunks = metadata_dense_candidates(
                dense_cache=self.dense_cache,
                query=query,
                candidate_chunks=self.retriever.chunks,
                top_k=self.args.summary_dense_coarse_k,
            )
            hybrid_topk = metadata_hybrid_candidates(
                retriever=self.retriever,
                dense_cache=self.dense_cache,
                bm25_cache=self.bm25_cache,
                query=query,
                candidate_chunks=coarse_chunks,
                top_k=self.args.candidate_k,
                dense_candidate_k=self.args.candidate_k,
                bm25_candidate_k=self.args.candidate_k,
            )
            return {
                "candidates": hybrid_topk,
                "pipeline": f"Dense Top{self.args.summary_dense_coarse_k} -> Local Hybrid Top{self.args.candidate_k}",
                "coarse_candidate_count": len(coarse_chunks),
                "coarse_candidate_doc_count": len(group_candidate_docs(coarse_chunks)),
            }

        hybrid_topk = metadata_hybrid_candidates(
            retriever=self.retriever,
            dense_cache=self.dense_cache,
            bm25_cache=self.bm25_cache,
            query=query,
            candidate_chunks=candidate_chunks,
            top_k=self.args.candidate_k,
            dense_candidate_k=self.args.candidate_k,
            bm25_candidate_k=self.args.candidate_k,
        )
        return {
            "candidates": hybrid_topk,
            "pipeline": f"Metadata Routing -> Hybrid Top{self.args.candidate_k}",
        }

    def retrieve_evidence(self, query: str) -> Dict[str, Any]:
        parsed = parse_query(query)
        qtype = parsed.get("question_type", "fact")

        if qtype == "compare":
            return self._retrieve_compare(query)
        return self._retrieve_fact_or_summary(query, qtype)

    def _retrieve_fact_or_summary(self, query: str, qtype: str) -> Dict[str, Any]:
        routed = route_candidate_chunks_for_query(self.retriever, query, force_qtype=qtype)
        candidate_info = self._hybrid_candidates(
            query=query,
            question_type=qtype,
            candidate_chunks=routed["candidate_chunks"],
        )
        rerank_topk = metadata_aware_rerank(
            reranker=self.reranker,
            query=query,
            candidates=candidate_info["candidates"],
            top_k=self.args.final_k,
            batch_size=self.args.reranker_batch_size,
        )
        evidences = [
            format_evidence(row, idx=i + 1, max_chars=self.args.max_evidence_chars)
            for i, row in enumerate(rerank_topk)
        ]
        return {
            "query": query,
            "question_type": qtype,
            "parsed": routed["parsed"],
            "pipeline": candidate_info["pipeline"],
            "candidate_count": len(candidate_info["candidates"]),
            "candidate_doc_count": len(group_candidate_docs(candidate_info["candidates"])),
            "evidences": evidences,
        }

    def _retrieve_compare(self, query: str) -> Dict[str, Any]:
        parsed = parse_query(query)
        parsed["question_type"] = "compare"
        subqueries = build_compare_subqueries(self.retriever, query)

        all_rows: List[Dict[str, Any]] = []
        sub_details = []
        per_sub_top_k = max(1, self.args.compare_top_k_each)

        for sub in subqueries:
            sub_query = sub["query"]
            candidate_chunks = self.retriever.route_chunks(sub, fallback_if_empty=True)
            candidate_info = self._hybrid_candidates(
                query=sub_query,
                question_type="fact",
                candidate_chunks=candidate_chunks,
            )
            rerank_rows = metadata_aware_rerank(
                reranker=self.reranker,
                query=sub_query,
                candidates=candidate_info["candidates"],
                top_k=per_sub_top_k,
                batch_size=self.args.reranker_batch_size,
            )
            all_rows.extend(rerank_rows)
            sub_details.append(
                {
                    "sub_query": sub_query,
                    "parsed": sub,
                    "pipeline": candidate_info["pipeline"],
                    "candidate_count": len(candidate_info["candidates"]),
                    "selected_count": len(rerank_rows),
                }
            )

        merged = dedup_evidences(all_rows)[: self.args.max_evidences]
        evidences = [
            format_evidence(row, idx=i + 1, max_chars=self.args.max_evidence_chars)
            for i, row in enumerate(merged)
        ]
        return {
            "query": query,
            "question_type": "compare",
            "parsed": parsed,
            "pipeline": f"Compare Decomposition -> Hybrid Top{self.args.candidate_k} -> Metadata-aware Rerank",
            "sub_details": sub_details,
            "candidate_count": sum(x["candidate_count"] for x in sub_details),
            "candidate_doc_count": len(group_candidate_docs(merged)),
            "evidences": evidences,
        }

    def should_refuse(self, evidences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidences:
            return {
                "refuse": True,
                "reason": "no_evidence",
                "max_score": None,
            }
        scores = [float(ev.get("score", 0.0)) for ev in evidences]
        max_score = max(scores)
        if all(score < self.args.min_rerank_score for score in scores):
            return {
                "refuse": True,
                "reason": "low_confidence_all_scores_below_threshold",
                "max_score": max_score,
                "threshold": self.args.min_rerank_score,
            }
        return {
            "refuse": False,
            "reason": "",
            "max_score": max_score,
            "threshold": self.args.min_rerank_score,
        }

    def answer(self, query: str) -> Dict[str, Any]:
        retrieval = self.retrieve_evidence(query)
        evidences = retrieval["evidences"]
        refusal = self.should_refuse(evidences)

        if refusal["refuse"]:
            answer = REFUSAL_TEXT
            citation_check = validate_citations(answer, evidences)
            return {
                **retrieval,
                "answer": answer,
                "raw_answer": answer,
                "refused": True,
                "refusal_reason": refusal,
                "citation_check": citation_check,
                "post_check_action": "refuse_before_generation",
            }

        messages = build_prompt(query, evidences)
        raw_answer = self.generator.generate(messages)
        guarded = apply_citation_guard(raw_answer, evidences)
        return {
            **retrieval,
            "answer": guarded["answer"],
            "raw_answer": raw_answer,
            "refused": guarded["answer"] == REFUSAL_TEXT,
            "refusal_reason": refusal if guarded["answer"] == REFUSAL_TEXT else {},
            "citation_check": guarded["citation_check"],
            "post_check_action": guarded["post_check_action"],
        }


def result_to_markdown(row: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"## {row['query']}")
    lines.append("")
    lines.append(f"- question_type: `{row.get('question_type', '')}`")
    lines.append(f"- pipeline: `{row.get('pipeline', '')}`")
    lines.append(f"- refused: `{row.get('refused')}`")
    if row.get("refusal_reason"):
        lines.append(f"- refusal_reason: `{row.get('refusal_reason')}`")
    lines.append("")
    lines.append("### Answer")
    lines.append("")
    lines.append(str(row.get("answer", "")))
    lines.append("")
    lines.append("### Evidences")
    lines.append("")
    for ev in row.get("evidences", []):
        lines.append(
            f"- [{ev['id']}] doc=`{ev['doc_id']}`, page=`{ev['page']}`, "
            f"chunk_id=`{ev['chunk_id']}`, score=`{ev['score']:.4f}`"
        )
        lines.append(f"  - {ev['text']}")
    lines.append("")
    lines.append("### Citation Check")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(row.get("citation_check", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def load_queries(args: argparse.Namespace) -> List[str]:
    if args.query:
        return [args.query]
    if args.query_file:
        rows = read_jsonl(args.query_file)
        return [str(row.get("query", "")).strip() for row in rows if str(row.get("query", "")).strip()]
    raise ValueError("请传入 --query 或 --query_file。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--query_file", type=str, default="")
    parser.add_argument("--output_jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--output_md", type=str, default=str(DEFAULT_OUTPUT_MD))

    parser.add_argument("--chunks_file", type=str, default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--faiss_index", type=str, default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss_metadata", type=str, default=str(DEFAULT_FAISS_METADATA))
    parser.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--generator_model", type=str, default=DEFAULT_GENERATOR_MODEL)

    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--final_k", type=int, default=5)
    parser.add_argument("--compare_top_k_each", type=int, default=3)
    parser.add_argument("--max_evidences", type=int, default=8)
    parser.add_argument("--dense_global_top_k", type=int, default=300)
    parser.add_argument("--summary_dense_coarse_k", type=int, default=500)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    parser.add_argument(
        "--min_rerank_score",
        type=float,
        default=-5.0,
        help="若所有证据 reranker_score 都低于该阈值，则拒答。",
    )

    parser.add_argument("--max_evidence_chars", type=int, default=900)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument(
        "--slow_tokenizer",
        action="store_true",
        help="强制使用 use_fast=False，规避旧 tokenizers 解析 Qwen tokenizer.json 失败的问题。",
    )
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args)
    pipeline = FinRAGGenerationPipeline(args)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"[generation] [{i}/{len(queries)}] {query}")
        results.append(pipeline.answer(query))

    write_jsonl(args.output_jsonl, results)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(result_to_markdown(row) for row in results), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Qwen3 evidence-bound generation done")
    print("=" * 100)
    print(f"jsonl: {args.output_jsonl}")
    print(f"markdown: {args.output_md}")


if __name__ == "__main__":
    main()



