# embeddings/embedding_validator.py
# WHY THIS EXISTS: Validates that multilingual-e5-large correctly places
# Hindi and English text about the same concept in the same vector space.
# If this fails, retrieval is broken at the foundation. Run before anything else.
# Target: cosine similarity > 0.85 between Hindi and English equivalents.

import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_settings

settings = get_settings()


def validate_cross_language_similarity(
    threshold: float = 0.85,
    verbose: bool = True
) -> tuple[float, bool]:
    """
    Validates that multilingual-e5-large embeds Hindi and English
    equivalents close together in vector space.

    Returns:
        tuple: (similarity_score, passed)

    Raises:
        AssertionError: if similarity < threshold
    """
    if verbose:
        print(f"Loading {settings.embedding_model}...")
        print(f"Device: {settings.embedding_device}")
        print("")

    model = SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device
    )

    # Test pairs — same concept in Hindi and English
    # MANDATORY: "passage: " prefix required by multilingual-e5-large
    test_pairs = [
        {
            "hindi": "passage: भूमि अधिग्रहण मुआवजा",
            "english": "passage: land acquisition compensation",
            "concept": "land acquisition compensation"
        },
        {
            "hindi": "passage: जीएसटी कर चालान",
            "english": "passage: GST tax invoice",
            "concept": "GST tax invoice"
        },
        {
            "hindi": "passage: बीमा पॉलिसी दावा",
            "english": "passage: insurance policy claim",
            "concept": "insurance policy claim"
        },
    ]

    scores = []
    all_passed = True

    for pair in test_pairs:
        hindi_emb = model.encode([pair["hindi"]])
        english_emb = model.encode([pair["english"]])
        sim = cosine_similarity(hindi_emb, english_emb)[0][0]
        scores.append(sim)
        passed = sim > threshold

        if not passed:
            all_passed = False

        if verbose:
            status = "✅" if passed else "❌"
            print(f"{status} Concept: {pair['concept']}")
            print(f"   Similarity: {sim:.4f} (threshold: {threshold})")
            print("")

    avg_score = sum(scores) / len(scores)

    if verbose:
        print(f"Average cross-language similarity: {avg_score:.4f}")
        print("")
        if all_passed:
            print("✅ VALIDATION PASSED — Safe to proceed.")
        else:
            print("❌ VALIDATION FAILED — Do not proceed.")
            print("   Fix: delete ~/.cache/huggingface and re-download model.")

    return avg_score, all_passed


def run_validation_gate() -> None:
    """
    Hard gate — raises SystemExit if validation fails.
    Used in CI pipelines and startup checks.
    """
    score, passed = validate_cross_language_similarity()

    assert passed, (
        f"Embedding validation FAILED (score: {score:.4f}, "
        f"threshold: 0.85). Do not proceed."
    )


if __name__ == "__main__":
    score, passed = validate_cross_language_similarity(verbose=True)
    if not passed:
        sys.exit(1)