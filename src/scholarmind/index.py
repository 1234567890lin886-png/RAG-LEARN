from dataclasses import dataclass
from typing import List

import numpy as np

from .chunking import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class InMemoryVectorIndex:
    """A tiny vector index for learning and small demos."""

    def __init__(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        if len(self.chunks) == 0:
            return []
        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        scores = self.vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in top_indices
        ]
