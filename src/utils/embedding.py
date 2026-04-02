"""
Embedding utilities for the fraud detection agent.

Provides helpers to build a canonical text representation of a transaction
(for Bedrock embedding) and to compute cosine similarity without numpy so
the module stays lightweight inside a Lambda deployment package.
"""

import math


def build_transaction_text(transaction: dict) -> str:
    """Build a normalised text representation of a transaction for embedding.

    The resulting string is passed directly to the embedding model, so the
    field order and labels are intentionally stable — changing them would
    shift embedding space and invalidate stored vectors.

    Args:
        transaction: Raw or sanitized transaction dict.  Missing fields are
                     replaced with the literal string "unknown" so that the
                     embedding always has the same structure.

    Returns:
        A single-line string suitable for passing to an embedding API.
    """
    amount: float | int | str = transaction.get("amount", "unknown")
    merchant_category: str = str(transaction.get("merchant_category", "unknown"))
    merchant_city: str = str(transaction.get("merchant_city", "unknown"))
    merchant_country: str = str(transaction.get("merchant_country", "unknown"))
    channel: str = str(transaction.get("channel", "unknown"))
    local_hour: str = str(transaction.get("local_hour", "unknown"))
    user_id: str = str(transaction.get("user_id", "unknown"))

    return (
        f"amount:{amount} "
        f"category:{merchant_category} "
        f"city:{merchant_city} "
        f"country:{merchant_country} "
        f"channel:{channel} "
        f"hour:{local_hour} "
        f"user:{user_id}"
    )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the cosine similarity between two dense vectors.

    Uses only the standard-library ``math`` module so the function works
    inside AWS Lambda without any numeric library in the deployment package.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.  Must have the same length as vec_a.

    Returns:
        Cosine similarity in [-1.0, 1.0].  Returns 0.0 if either vector has
        zero magnitude (avoids division by zero).

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot: float = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a: float = math.sqrt(sum(a * a for a in vec_a))
    mag_b: float = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)
