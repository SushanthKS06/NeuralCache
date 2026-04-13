from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neural_cache.config import EncoderConfig, EmbeddingModel

class QueryEncoder:
    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()
        self._model = None
        self._dimension = self.config.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = self.config.model_name.value
        self._model = SentenceTransformer(
            model_name,
            device=self.config.device,
        )
        self._model.max_seq_length = self.config.max_seq_length
        self._dimension = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        queries: str | list[str],
        normalize: bool | None = None,
    ) -> NDArray[np.float32]:
        if isinstance(queries, str):
            queries = [queries]

        normalize = normalize if normalize is not None else self.config.normalize

        embeddings = self.model.encode(
            queries,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )

        return embeddings.astype(np.float32)

    def encode_single(self, query: str) -> NDArray[np.float32]:
        result = self.encode(query)
        return result[0]

    def similarity(
        self,
        query1: str,
        query2: str,
    ) -> float:
        emb1 = self.encode_single(query1)
        emb2 = self.encode_single(query2)
        return float(np.dot(emb1, emb2))

    def get_model_info(self) -> dict:
        return {
            "model_name": self.config.model_name.value,
            "embedding_dimension": self._dimension,
            "max_seq_length": self.config.max_seq_length,
            "device": self.config.device,
            "pooling": self.config.pooling_strategy,
            "normalize": self.config.normalize,
        }

    def warmup(self) -> None:
        _ = self.encode("__warmup_query__")
