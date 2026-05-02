import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    """Common interface for all text embedding backends."""

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Return one normalized vector for each input text."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class HashingEmbedder:
    """A dependency-light fallback that demonstrates the retrieval pipeline."""

    dimensions: int = 256

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in _simple_tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                index = int(digest, 16) % self.dimensions
                vector[index] += 1.0
            rows.append(vector)

        if not rows:
            return np.empty((0, self.dimensions), dtype=np.float32)

        return _normalize(np.vstack(rows))


@dataclass
class TransformerEmbedder:
    """Use a Hugging Face Transformer encoder to create sentence embeddings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = False
    cache_dir: Optional[Path] = None
    max_length: int = 256

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            cache_dir=cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            cache_dir=cache_dir,
        )
        self._model.eval()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            hidden_size = getattr(self._model.config, "hidden_size", 0)
            return np.empty((0, hidden_size), dtype=np.float32)

        with self._torch.no_grad():
            encoded = self._tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            output = self._model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            vectors = summed / counts

        return _normalize(vectors.cpu().numpy().astype(np.float32))


def _simple_tokens(text: str) -> list[str]:
    lower_text = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", lower_text)
    cjk_tokens = [char for char in lower_text if "\u4e00" <= char <= "\u9fff"]
    return latin_tokens + cjk_tokens
