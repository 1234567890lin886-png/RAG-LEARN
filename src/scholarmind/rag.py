from pathlib import Path
from typing import List

from .chunking import DocumentChunk, chunk_documents, read_text_files
from .embeddings import Embedder
from .index import InMemoryVectorIndex, SearchResult


def build_index(
    docs_dir: Path,
    embedder: Embedder,
    chunk_size: int = 500,
    overlap: int = 80,
) -> InMemoryVectorIndex:
    documents = read_text_files(docs_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    vectors = embedder.encode(chunk.text for chunk in chunks)
    return InMemoryVectorIndex(chunks, vectors)


def retrieve(
    question: str,
    index: InMemoryVectorIndex,
    embedder: Embedder,
    top_k: int = 3,
) -> List[SearchResult]:
    query_vector = embedder.encode([question])
    return index.search(query_vector, top_k=top_k)


def compose_grounded_answer(question: str, results: List[SearchResult]) -> str:
    """Create a source-grounded answer without using a generative model."""

    if not results:
        return "没有检索到相关片段。"

    lines = [
        f"问题：{question}",
        "",
        "基于当前资料，最相关的证据如下：",
    ]

    for rank, result in enumerate(results, start=1):
        snippet = _shorten(result.chunk.text)
        lines.append(
            f"{rank}. [{result.chunk.source}] 相似度 {result.score:.3f}\n   {snippet}"
        )

    lines.extend(
        [
            "",
            "学习提示：这个版本先展示可追溯证据。下一步可以把这些片段交给生成模型，让它组织成更自然的回答。",
        ]
    )
    return "\n".join(lines)


def explain_chunks(chunks: List[DocumentChunk]) -> str:
    return "\n".join(
        f"- {chunk.id}: {chunk.start}-{chunk.end}, {len(chunk.text)} chars"
        for chunk in chunks
    )


def _shorten(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
