# embeddings/dense_embedder.py
# WHY THIS EXISTS: Embeds text chunks using multilingual-e5-large.
# Mandatory prefixes: "passage: " at ingestion, "query: " at retrieval.
# Missing these prefixes degrades similarity scores by 0.05-0.15.
# batch_size=16 — safe for RTX 3050 6GB VRAM. LAW 4.

import torch
import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

settings = get_settings()


class DenseEmbedder:
    """
    Wraps multilingual-e5-large for SmartDocs ingestion and retrieval.

    Two modes:
        embed_passages() — for PDF chunks at ingestion (prefix: "passage: ")
        embed_query()    — for user queries at retrieval (prefix: "query: ")

    Never call model.encode() directly outside this class.
    The prefix requirement is enforced here — not the caller's responsibility.
    """

    def __init__(self):
        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        self.batch_size = settings.embedding_batch_size  # 16 for RTX 3050

        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """
        Lazy loads model on first use.
        Checks CUDA availability — falls back to CPU with warning if needed.
        """
        if self._model is not None:
            return self._model

        # Verify CUDA available if cuda requested
        if self.device == "cuda" and not torch.cuda.is_available():
            print(
                "WARNING: CUDA requested but not available. "
                "Falling back to CPU. Embedding will be slower."
            )
            self.device = "cpu"

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        return self._model
    
    
    async def warmup(self) -> None:
        """Pre-loads model so first real request is not slow."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.embed_passages(["warmup"]))
    

    @property
    def model(self) -> SentenceTransformer:
        return self._load_model()

    def embed_passages(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embeds document chunks for ingestion into pgvector.
        Adds mandatory "passage: " prefix to every text.

        Args:
            texts: List of chunk texts (without any prefix)
            show_progress: Show tqdm progress bar for large batches

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        if not texts:
            return np.array([])

        # MANDATORY: "passage: " prefix — this is model-specific behavior
        # multilingual-e5-large was trained with these prefixes
        # Wrong: model.encode(["भूमि अधिग्रहण"])
        # Right: model.encode(["passage: भूमि अधिग्रहण"])
        prefixed = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embeds a single user query for retrieval.
        Adds mandatory "query: " prefix.

        Args:
            query_text: Raw user query (no prefix)

        Returns:
            numpy array of shape (1024,)
        """
        # MANDATORY: "query: " prefix at retrieval time
        prefixed = f"query: {query_text}"

        embedding = self.model.encode(
            [prefixed],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding[0]

    def embed_passages_batched(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """
        Embeds large lists of passages in batches.
        Use this for full document ingestion (hundreds of chunks).
        Returns list of individual embeddings (not stacked).

        Args:
            texts: List of chunk texts
            show_progress: Show progress bar (recommended for large docs)

        Returns:
            List of numpy arrays, one per text
        """
        embeddings = self.embed_passages(texts, show_progress=show_progress)
        return [embeddings[i] for i in range(len(embeddings))]


# Module-level singleton — import this in ingestion_worker and retriever
_embedder_instance: DenseEmbedder | None = None


def get_embedder() -> DenseEmbedder:
    """
    Returns singleton DenseEmbedder instance.
    Model loads once, reused for all calls in the process lifetime.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = DenseEmbedder()
    return _embedder_instance