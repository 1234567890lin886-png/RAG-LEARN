from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DocumentChunk:
    """A small searchable piece of a source document."""

    id: str
    source: str
    text: str
    start: int
    end: int


def read_text_files(directory: Path) -> List[tuple[str, str]]:
    """Read Markdown and text files from a directory."""

    supported_suffixes = {".md", ".txt"}
    documents: List[tuple[str, str]] = []

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            documents.append((path.name, path.read_text(encoding="utf-8")))

    return documents


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[DocumentChunk]:
    """Split one document into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: List[DocumentChunk] = []
    cursor = 0
    chunk_id = 0
    step = chunk_size - overlap

    while cursor < len(clean_text):
        end = min(cursor + chunk_size, len(clean_text))
        piece = clean_text[cursor:end].strip()

        if piece:
            chunks.append(
                DocumentChunk(
                    id=f"{source}:{chunk_id}",
                    source=source,
                    text=piece,
                    start=cursor,
                    end=end,
                )
            )
            chunk_id += 1

        if end == len(clean_text):
            break
        cursor += step

    return chunks


def chunk_documents(
    documents: Iterable[tuple[str, str]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[DocumentChunk]:
    """Split many documents into chunks."""

    chunks: List[DocumentChunk] = []
    for source, text in documents:
        chunks.extend(chunk_text(text, source, chunk_size=chunk_size, overlap=overlap))
    return chunks
