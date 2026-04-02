"""
src/utils — shared utilities for the fraud detection agent.

Re-exports the most commonly used symbols so callers can import directly
from `src.utils` without knowing which submodule each lives in.

Example:
    from src.utils import build_transaction_text, cosine_similarity
    from src.utils import enrich_geoip, calculate_local_hour
    from src.utils import sanitize_transaction, sanitize_llm_input
    from src.utils import FraudMetrics
"""

from src.utils.embedding import build_transaction_text, cosine_similarity
from src.utils.geoip import calculate_local_hour, enrich_geoip
from src.utils.metrics import FraudMetrics
from src.utils.sanitizer import sanitize_llm_input, sanitize_transaction

__all__ = [
    # embedding
    "build_transaction_text",
    "cosine_similarity",
    # geoip
    "enrich_geoip",
    "calculate_local_hour",
    # sanitizer
    "sanitize_transaction",
    "sanitize_llm_input",
    # metrics
    "FraudMetrics",
]
