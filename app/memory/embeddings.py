from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_text(text) for text in texts])

    @property
    def dim(self) -> int:
        raise NotImplementedError


class NoOpEmbedder(Embedder):
    def embed_text(self, text: str) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    @property
    def dim(self) -> int:
        return 0


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._dim = 0
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text or ""])
        return vectors[0] if len(vectors) else np.zeros(self.dim, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        cleaned = [text or "" for text in texts]
        vectors = model.encode(
            cleaned,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dim(self) -> int:
        if self._dim:
            return self._dim
        if self._model is None:
            return 0
        self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._dim


class TfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self._dim = 0

    def embed_text(self, text: str) -> np.ndarray:
        query_vec, _ = self.embed_with_corpus(text or "", [])
        return query_vec

    def embed_with_corpus(self, query: str, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        combined = [query or ""] + [item or "" for item in corpus]
        matrix = vectorizer.fit_transform(combined)
        dense = matrix.toarray().astype(np.float32)
        self._dim = dense.shape[1] if dense.ndim == 2 else 0
        query_vec = dense[0] if len(dense) else np.zeros(0, dtype=np.float32)
        corpus_vecs = dense[1:] if len(dense) > 1 else np.zeros((0, self._dim), dtype=np.float32)
        return query_vec, corpus_vecs

    @property
    def dim(self) -> int:
        return self._dim


def build_embedder(backend: str, model_name: str) -> Embedder:
    normalized = (backend or "").strip().lower()
    if normalized in {"sentence_transformers", "sentence-transformers"}:
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.warning("sentence-transformers not available; falling back to TF-IDF.")
            return _fallback_tfidf()
        return SentenceTransformersEmbedder(model_name)
    if normalized == "tfidf":
        return _fallback_tfidf()
    logger.warning("Unknown embedding backend '%s'; using TF-IDF.", backend)
    return _fallback_tfidf()


def _fallback_tfidf() -> Embedder:
    if importlib.util.find_spec("sklearn") is None:
        logger.warning("scikit-learn not available; memory retrieval disabled.")
        return NoOpEmbedder()
    return TfidfEmbedder()


def serialize_embedding(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_embedding(payload: bytes) -> np.ndarray:
    if not payload:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(payload, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def batch_cosine_similarity(query: np.ndarray, items: np.ndarray) -> list[float]:
    if items.size == 0 or query.size == 0:
        return []
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return [0.0 for _ in range(items.shape[0])]
    item_norms = np.linalg.norm(items, axis=1)
    scores: list[float] = []
    for idx, item_norm in enumerate(item_norms):
        if item_norm == 0.0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(query, items[idx]) / (query_norm * item_norm)))
    return scores


def normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    return normalized.translate(
        {
            ord("\u00e7"): "c",
            ord("\u011f"): "g",
            ord("\u0131"): "i",
            ord("\u00f6"): "o",
            ord("\u015f"): "s",
            ord("\u00fc"): "u",
        }
    )
