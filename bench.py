#!/usr/bin/env python3
"""Benchmark retrieval on the HUST scholarship corpus.

Each teammate changes only CHUNKER (one line) so results stay comparable.
Pass --all-strategies to compare FixedSize / Sentence / Recursive / Heading.
"""
from __future__ import annotations

import argparse
import os
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import (
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data/hoc-bong-hust")
RESULT_PATH = Path("ket_qua_benchmark.txt")

# --- one-line strategy switch (everyone else stays the same) ---
CHUNKER = RecursiveChunker(chunk_size=900)
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER = HeadingChunker(chunk_size=800)

STRATEGIES = {
    "recursive": RecursiveChunker(chunk_size=900),
    "fixed": FixedSizeChunker(chunk_size=500, overlap=50),
    "sentence": SentenceChunker(max_sentences_per_chunk=3),
    "heading": HeadingChunker(chunk_size=800),
}

QUERIES = [
    {
        "id": 1,
        "question": "GPA và điểm rèn luyện tối thiểu để đạt học bổng KKHT loại A là bao nhiêu?",
        "filter": None,
        "gold": "Loại A: GPA ≥ 3,6 và điểm rèn luyện học kỳ ≥ 90 điểm.",
        "gold_doc": "hust-kkht-criteria-student",
        "must_contain": "3,6",
    },
    {
        "id": 2,
        "question": "Tiêu chuẩn xét cấp học bổng KKHT là gì?",
        "filter": {"audience": "student"},
        "gold": "Loại C: GPA ≥ 2,5 và rèn luyện ≥ 65; loại B: GPA ≥ 3,2 và rèn luyện ≥ 80; loại A: GPA ≥ 3,6 và rèn luyện ≥ 90.",
        "gold_doc": "hust-kkht-criteria-student",
        "must_contain": "2,5",
    },
    {
        "id": 3,
        "question": "Hạn nộp hồ sơ học bổng Trần Đại Nghĩa học kỳ I năm học 2026-2027 là khi nào và nộp ở đâu?",
        "filter": None,
        "gold": "Nộp trên eHUST hoặc qldt.hust.edu.vn mục học bổng, hồ sơ giấy tại phòng 102 nhà C1 trước 16h30 thứ Sáu ngày 09/10/2026.",
        "gold_doc": "hust-tran-dai-nghia-2026-1",
        "must_contain": "09/10/2026",
    },
    {
        "id": 4,
        "question": "Kỳ II năm học 2025-2026 có bao nhiêu sinh viên được học bổng KKHT loại A, B và C?",
        "filter": None,
        "gold": "1.888 sinh viên: 1.343 loại A (xuất sắc), 456 loại B (giỏi), 89 loại C (khá).",
        "gold_doc": "hust-kkht-results-2025-2",
        "must_contain": "1.343",
    },
    {
        "id": 5,
        "question": "Học bổng MB The Best of MB Chasing 2025 trị giá bao nhiêu và hạn đăng ký là khi nào?",
        "filter": None,
        "gold": "Từ 10–30 triệu VNĐ/sinh viên; hạn đăng ký 09/01/2026.",
        "gold_doc": "hust-mb-chasing-2025",
        "must_contain": "09/01/2026",
    },
]


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2].strip()


def choose_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as error:
            print(f"Local embedder unavailable ({error}); falling back to mock.")
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as error:
            print(f"OpenAI embedder unavailable ({error}); falling back to mock.")
    elif provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception as error:
            print(f"Gemini embedder unavailable ({error}); falling back to mock.")
    return _mock_embed


def load_chunks(chunker) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        frontmatter, body = parse_markdown(path)
        for index, chunk in enumerate(chunker.chunk(body)):
            metadata = {
                **frontmatter,
                "doc_id": path.stem,
                "source": str(path),
                "chunk_index": str(index),
            }
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )
    return documents


def extractive_llm(gold: str, must_contain: str) -> callable:
    def llm_fn(prompt: str) -> str:
        if must_contain in prompt:
            return gold
        return "Không đủ ngữ cảnh trong top-3 để trả lời chính xác."

    return llm_fn


def score_query(results: list[dict], spec: dict, agent_answer: str) -> dict:
    gold_doc = spec["gold_doc"]
    must_contain = spec["must_contain"]
    span_rank = None
    doc_rank = None
    for rank, result in enumerate(results, start=1):
        doc_id = result["metadata"].get("doc_id", "")
        if doc_id == gold_doc and doc_rank is None:
            doc_rank = rank
        if doc_id == gold_doc and must_contain in result["content"] and span_rank is None:
            span_rank = rank
    agent_ok = spec["gold"] in agent_answer or must_contain in agent_answer
    if span_rank == 1 and agent_ok:
        points = 2
    elif span_rank in {2, 3}:
        points = 1
    else:
        points = 0
    return {
        "points": points,
        "span_rank": span_rank,
        "doc_rank": doc_rank,
        "agent_ok": agent_ok,
        "agent_answer": agent_answer,
        "results": results,
    }


def format_hits(results: list[dict], spec: dict) -> str:
    if not results:
        return "  (no results)\n"
    lines = []
    for rank, result in enumerate(results, start=1):
        doc_id = result["metadata"].get("doc_id", "")
        preview = result["content"].replace("\n", " ")[:160]
        marker = ""
        if doc_id == spec["gold_doc"] and spec["must_contain"] in result["content"]:
            marker = "  << GOLD+SPAN"
        elif doc_id == spec["gold_doc"]:
            marker = "  << GOLD DOC (span missing)"
        lines.append(
            f"  {rank}. score={result['score']:.3f}  doc_id={doc_id}  id={result.get('id', '')}{marker}"
        )
        lines.append(f"     {preview}...")
    return "\n".join(lines) + "\n"


def run_one(store: EmbeddingStore, spec: dict, metadata_filter: dict | None) -> dict:
    results = store.search_with_filter(
        spec["question"],
        top_k=3,
        metadata_filter=metadata_filter,
    )
    prompt_context = "\n\n".join(f"[{i}] {r['content']}" for i, r in enumerate(results, start=1))
    agent_answer = extractive_llm(spec["gold"], spec["must_contain"])(
        f"Context:\n{prompt_context}\n\nQuestion: {spec['question']}\n"
    )
    scored = score_query(results, spec, agent_answer)
    scored["filter"] = metadata_filter
    return scored


def run_strategy(name: str, chunker, embedder) -> str:
    buf = StringIO()
    documents = load_chunks(chunker)
    store = EmbeddingStore(
        collection_name=f"hust_bench_{name}",
        embedding_fn=embedder,
    )
    store.add_documents(documents)
    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    header = (
        f"=== Strategy: {chunker.__class__.__name__} ({name}) ===\n"
        f"Corpus : {DATA_DIR}\n"
        f"Embed  : {backend}\n"
        f"Loaded {store.get_collection_size()} chunks from {len(list(DATA_DIR.glob('*.md')))} files\n"
    )
    buf.write(header)
    print(header, end="")
    total = 0
    for spec in QUERIES:
        scored = run_one(store, spec, spec["filter"])
        total += scored["points"]
        block = (
            f"\nQ{spec['id']}: {spec['question']}\n"
            f"  filter: {spec['filter'] or 'none'}\n"
            f"  gold: {spec['gold']}  [{spec['gold_doc']}]\n"
            f"{format_hits(scored['results'], spec)}"
            f"  agent: {scored['agent_answer']}\n"
            f"  score: {scored['points']}/2  span_rank={scored['span_rank']}  doc_rank={scored['doc_rank']}\n"
        )
        buf.write(block)
        print(block, end="")
        if spec["filter"]:
            ab = run_one(store, spec, None)
            ab_block = (
                f"  --- A/B without filter ---\n"
                f"{format_hits(ab['results'], spec)}"
                f"  agent: {ab['agent_answer']}\n"
                f"  score: {ab['points']}/2  span_rank={ab['span_rank']}  doc_rank={ab['doc_rank']}\n"
            )
            buf.write(ab_block)
            print(ab_block, end="")
    footer = f"\nTOTAL {total}/10  ({chunker.__class__.__name__})\n"
    buf.write(footer)
    print(footer, end="")
    return buf.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab 07 retrieval benchmark.")
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run FixedSize, Sentence, Recursive, and Heading for group comparison.",
    )
    return parser.parse_args()


def main() -> int:
    if not DATA_DIR.is_dir():
        print(f"Missing corpus directory: {DATA_DIR}")
        return 1

    args = parse_args()
    embedder = choose_embedder()
    if args.all_strategies:
        parts = [run_strategy(name, chunker, embedder) for name, chunker in STRATEGIES.items()]
        RESULT_PATH.write_text("\n".join(parts), encoding="utf-8")
    else:
        name = CHUNKER.__class__.__name__.replace("Chunker", "").lower()
        text = run_strategy(name, CHUNKER, embedder)
        RESULT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
