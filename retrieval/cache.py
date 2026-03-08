# retrieval/cache.py
# WHY THIS EXISTS: Semantic query cache via aioredis.
# Cosine similarity 0.95 threshold for cache hit.
# Zero re-embedding, zero re-retrieval on repeat queries. LAW 12.

import json
import hashlib
import logging
import time
from typing import Optional
from functools import lru_cache

import numpy as np
import redis.asyncio as aioredis

from config.settings import get_settings
from embeddings.dense_embedder import get_embedder

settings = get_settings()
logger = logging.getLogger(__name__)

CACHE_PREFIX = "smartdocs:cache"
CACHE_EMB_PREFIX = "smartdocs:query_emb"
CACHE_TTL = 3600  # 1 hour


@lru_cache(maxsize=1)
def _get_redis_url() -> str:
    return settings.redis_url


async def _get_redis() -> aioredis.Redis:
    """Returns aioredis client. Non-singleton — aioredis manages pooling."""
    return await aioredis.from_url(
        _get_redis_url(),
        encoding="utf-8",
        decode_responses=True,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a, b) / norm)


def _query_hash(user_id: str, query: str) -> str:
    return hashlib.sha256(f"{user_id}:{query}".encode()).hexdigest()[:16]


async def get_cached_response(
    query: str,
    user_id: str,
    similarity_threshold: float = 0.95,
) -> Optional[dict]:
    """
    Checks aioredis for a semantically similar cached response.
    Embeds current query, compares against stored query embeddings.
    Cache is per-user — never cross-user.

    Args:
        query: Current query text
        user_id: Current user (cache is per-user)
        similarity_threshold: Cosine threshold (default: 0.95)

    Returns:
        Cached response dict with cache_hit=True, or None
    """
    start = time.perf_counter()

    try:
        redis = await _get_redis()
        embedder = get_embedder()
        current_emb = embedder.embed_query(query)

        pattern = f"{CACHE_EMB_PREFIX}:{user_id}:*"
        keys = await redis.keys(pattern)

        for emb_key in keys:
            cached_json = await redis.get(emb_key)
            if not cached_json:
                continue

            cached_emb = np.array(json.loads(cached_json))
            similarity = _cosine_similarity(current_emb, cached_emb)

            if similarity >= similarity_threshold:
                query_hash = emb_key.split(":")[-1]
                response_key = f"{CACHE_PREFIX}:{user_id}:{query_hash}"
                cached = await redis.get(response_key)

                if cached:
                    result = json.loads(cached)
                    result["cache_hit"] = True
                    result["cache_similarity"] = round(similarity, 4)
                    latency = round((time.perf_counter() - start) * 1000, 2)
                    logger.info(
                        "Cache hit",
                        extra={
                            "user_id": user_id,
                            "similarity": similarity,
                            "latency_ms": latency,
                        },
                    )
                    return result

        return None

    except Exception as e:
        # Cache failure is non-fatal — log and continue
        logger.warning("Cache read failed — proceeding without cache", extra={"error": str(e)})
        return None


async def set_cached_response(
    query: str,
    user_id: str,
    response: dict,
) -> bool:
    """
    Stores query embedding + response in aioredis.
    TTL: 1 hour. Per-user namespace.

    Args:
        query: User query
        user_id: Current user
        response: Full response dict

    Returns:
        True if stored successfully
    """
    try:
        redis = await _get_redis()
        embedder = get_embedder()

        q_hash = _query_hash(user_id, query)
        embedding = embedder.embed_query(query)

        # Store embedding for future similarity checks
        await redis.setex(
            f"{CACHE_EMB_PREFIX}:{user_id}:{q_hash}",
            CACHE_TTL,
            json.dumps(embedding.tolist()),
        )

        # Store response
        response["cache_hit"] = False
        await redis.setex(
            f"{CACHE_PREFIX}:{user_id}:{q_hash}",
            CACHE_TTL,
            json.dumps(response, ensure_ascii=False),
        )

        logger.debug("Response cached", extra={"user_id": user_id, "hash": q_hash})
        return True

    except Exception as e:
        logger.warning("Cache write failed", extra={"error": str(e)})
        return False


async def invalidate_user_cache(user_id: str) -> int:
    """
    Invalidates all cache entries for a user.
    Called when user uploads a new document. LAW 12.

    Returns:
        Number of keys deleted
    """
    try:
        redis = await _get_redis()
        patterns = [
            f"{CACHE_PREFIX}:{user_id}:*",
            f"{CACHE_EMB_PREFIX}:{user_id}:*",
        ]
        total = 0
        for pattern in patterns:
            keys = await redis.keys(pattern)
            if keys:
                await redis.delete(*keys)
                total += len(keys)

        logger.info("User cache invalidated", extra={"user_id": user_id, "keys_deleted": total})
        return total

    except Exception as e:
        logger.error("Cache invalidation failed", extra={"error": str(e), "user_id": user_id})
        return 0