# embeddings/sparse_embedder.py
# WHY THIS EXISTS: BM25 sparse index for exact-match retrieval.
# Dense vectors miss exact keyword matches (section numbers, names).
# BM25 catches them. Combined with dense = hybrid retrieval. LAW 5.
# Indic normalization MUST run before BM25 indexing for Hindi text.

import re
import numpy as np
from rank_bm25 import BM25Okapi
from langdetect import detect, LangDetectException

from ingestion.indic_preprocessing import (
    indic_preprocessing_pipeline,
    ALL_INDIC_LANGUAGES,
)


def tokenize_for_bm25(text: str, lang_code: str = "en") -> list[str]:
    """
    Tokenizes text for BM25 indexing.
    For Indic languages: runs full normalization pipeline first.
    For Latin: standard whitespace + punctuation tokenization.

    This is why BM25 works for Hindi in SmartDocs —
    indic_normalize runs before tokenization, ensuring
    Unicode variants of the same word map to the same token.

    Args:
        text: Text to tokenize
        lang_code: ISO language code

    Returns:
        List of tokens
    """
    if not text:
        return []

    # For Indic languages — normalize first, then tokenize
    if lang_code in ALL_INDIC_LANGUAGES:
        result = indic_preprocessing_pipeline(
            text=text,
            lang_code=lang_code,
            return_sentences=False,
        )
        normalized_text = result["cleaned_text"]
    else:
        normalized_text = text.lower()

    # Remove punctuation and split on whitespace
    normalized_text = re.sub(r"[^\w\s]", " ", normalized_text)
    tokens = normalized_text.split()

    # Remove empty tokens and single characters (except Indic scripts)
    tokens = [
        t for t in tokens
        if len(t) > 1 or any(
            "\u0900" <= c <= "\u0d7f" for c in t
        )
    ]

    return tokens


class SparseEmbedder:
    """
    BM25 sparse index for a single document's chunks.
    One SparseEmbedder instance per document per ingestion session.
    Rebuilt from stored chunks on retrieval.

    Works alongside DenseEmbedder in hybrid retrieval.
    Dense weight: 0.7 | Sparse weight: 0.3 (from retrieval_config.yaml)
    """

    def __init__(self, lang_code: str = "en"):
        self.lang_code = lang_code
        self.bm25: BM25Okapi | None = None
        self.corpus_chunks: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[str]) -> None:
        """
        Builds BM25 index from list of text chunks.
        Called during ingestion after chunking and indic preprocessing.
        Indic normalization runs inside tokenize_for_bm25.

        Args:
            chunks: List of preprocessed text chunks
        """
        if not chunks:
            return

        self.corpus_chunks = chunks
        self.tokenized_corpus = [
            tokenize_for_bm25(chunk, self.lang_code)
            for chunk in chunks
        ]

        # Filter out empty tokenized chunks
        valid_indices = [
            i for i, tokens in enumerate(self.tokenized_corpus)
            if tokens
        ]

        if not valid_indices:
            return

        valid_corpus = [self.tokenized_corpus[i] for i in valid_indices]
        self.bm25 = BM25Okapi(valid_corpus)
        self._valid_indices = valid_indices

    def get_scores(self, query: str) -> np.ndarray:
        """
        Returns BM25 scores for all chunks against query.
        Query normalized using same pipeline as corpus.

        Args:
            query: User query text

        Returns:
            numpy array of BM25 scores, one per chunk
        """
        if self.bm25 is None or not self.corpus_chunks:
            return np.zeros(len(self.corpus_chunks))

        query_tokens = tokenize_for_bm25(query, self.lang_code)
        if not query_tokens:
            return np.zeros(len(self.corpus_chunks))

        raw_scores = self.bm25.get_scores(query_tokens)

        # Map back to full corpus size
        full_scores = np.zeros(len(self.corpus_chunks))
        for score_idx, corpus_idx in enumerate(self._valid_indices):
            if score_idx < len(raw_scores):
                full_scores[corpus_idx] = raw_scores[score_idx]

        return full_scores

    def get_top_n(
        self,
        query: str,
        n: int = 20,
    ) -> list[tuple[int, float, str]]:
        """
        Returns top-N chunks by BM25 score.

        Args:
            query: User query
            n: Number of results to return

        Returns:
            List of (chunk_index, score, chunk_text) tuples
        """
        scores = self.get_scores(query)

        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    int(idx),
                    float(scores[idx]),
                    self.corpus_chunks[idx],
                ))

        return results