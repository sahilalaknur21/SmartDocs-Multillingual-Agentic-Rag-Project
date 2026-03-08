# smartdocs/validate_embeddings.py
#
# WHY THIS EXISTS:
# multilingual-e5-large must place Hindi and English text about the
# same concept close together in vector space. If it doesn't, retrieval
# fails silently — Hindi queries return wrong or empty results.
# This test proves the model works correctly before we build anything else.
# Run this ONCE before Part 1. If it fails, do not write any other code.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading multilingual-e5-large model...")
print("First run: downloads ~1.1GB from HuggingFace. This takes 2-5 minutes.")
print("Subsequent runs: loads from local cache in seconds.")
print("")

model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Model loaded. Running cross-language similarity test...")
print("")

# MANDATORY: multilingual-e5-large requires "passage: " prefix at ingestion
# Without this prefix, similarity scores DROP by 0.05-0.15
# This is not optional — it is how the model was trained
hindi = model.encode(["passage: भूमि अधिग्रहण मुआवजा"])      # land acquisition compensation
english = model.encode(["passage: land acquisition compensation"])

sim = cosine_similarity(hindi, english)[0][0]

print(f"Hindi text:   भूमि अधिग्रहण मुआवजा")
print(f"English text: land acquisition compensation")
print(f"")
print(f"Cross-language similarity: {sim:.4f}")
print("")

if sim > 0.85:
    print("✅ PASSED — Embeddings place Hindi and English in the same vector space.")
    print("✅ Retrieval pipeline is validated. Safe to proceed to Part 1.")
else:
    print(f"❌ FAILED — Similarity is {sim:.4f}, must be above 0.85.")
    print("❌ DO NOT PROCEED. Debug model loading first.")
    print("   Most likely cause: model downloaded incorrectly.")
    print("   Fix: delete the .cache/huggingface folder and re-run.")

assert sim > 0.85, f"VALIDATION FAILED: {sim:.4f} — do not proceed to Part 1"