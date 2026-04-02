"""
In-memory mock implementations of DynamoDB, OpenSearch, and Redis clients.

Each class matches the public interface of its production counterpart so that
SentinelService, FraudAnalystService, and AMLSpecialistService can be
constructed with these mocks in tests without any modification.

Design choices:
- InMemoryDynamoDBClient stores items in plain dicts keyed by (PK) or (PK, SK).
  Methods that query by GSI (verdict, userId, status) perform linear scans of
  the in-memory dict — acceptable because test datasets are small.
- InMemoryOpenSearchClient computes exact cosine similarity rather than
  approximate HNSW, which is fine for correctness tests.
- InMemoryRedisClient is a simple dict; TTL semantics are not enforced because
  E2E tests run in a single process without wall-clock sleeping.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# InMemoryDynamoDBClient
# ---------------------------------------------------------------------------


class InMemoryDynamoDBClient:
    """Pure Python stand-in for DynamoDBClient.

    Stores all items in ``self._tables``, a dict of table name -> item store.
    Each item store is a dict keyed by a tuple (PK,) or (PK, SK) depending on
    whether the item carries an SK field.

    The interface exactly mirrors src/clients/dynamodb_client.py.
    """

    def __init__(self) -> None:
        # {table_name: {item_key_tuple: item_dict}}
        self._tables: dict[str, dict[tuple, dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store(self, table_name: str) -> dict[tuple, dict]:
        """Return (creating if absent) the item store for *table_name*."""
        if table_name not in self._tables:
            self._tables[table_name] = {}
        return self._tables[table_name]

    @staticmethod
    def _item_key(item: dict) -> tuple:
        """Derive the storage key from an item's PK (and SK if present)."""
        pk = item.get("PK", "")
        sk = item.get("SK")
        return (pk, sk) if sk is not None else (pk,)

    def _put_item(self, table_name: str, item: dict) -> None:
        self._store(table_name)[self._item_key(item)] = item

    def _get_item(self, table_name: str, pk: str, sk: str | None = None) -> dict | None:
        key = (pk, sk) if sk is not None else (pk,)
        return self._store(table_name).get(key)

    def _all_items(self, table_name: str) -> list[dict]:
        return list(self._store(table_name).values())

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    def get_persona(self, table_name: str, user_id: str) -> dict | None:
        """Return the latest persona for *user_id*, or None.

        Mirrors the production query: PK=USER#<user_id>, SK=VERSION#<n>.
        Scans all items with the matching PK and returns the one with the
        lexicographically largest SK (latest version).
        """
        prefix_pk = f"USER#{user_id}"
        candidates = [
            item
            for item in self._all_items(table_name)
            if item.get("PK") == prefix_pk
            and str(item.get("SK", "")).startswith("VERSION#")
        ]
        if not candidates:
            return None
        # Sort descending by SK; the highest version string wins.
        candidates.sort(key=lambda i: str(i.get("SK", "")), reverse=True)
        return candidates[0]

    def put_persona(self, table_name: str, persona: dict) -> None:
        """Store a persona item.  Caller must set PK and SK."""
        self._put_item(table_name, persona)

    # ------------------------------------------------------------------
    # Fraud decision
    # ------------------------------------------------------------------

    def put_decision(self, table_name: str, decision: dict) -> None:
        """Store a fraud decision item."""
        self._put_item(table_name, decision)

    def get_decisions_by_verdict(
        self,
        table_name: str,
        verdict: str,
        since_timestamp: str,
        index_name: str = "verdict-timestamp-index",
    ) -> list[dict]:
        """Return decisions matching *verdict* since *since_timestamp*.

        The in-memory implementation ignores *index_name* and performs a
        linear scan, filtering on the ``verdict`` and ``timestamp`` fields.
        """
        results = [
            item
            for item in self._all_items(table_name)
            if item.get("verdict") == verdict
            and str(item.get("timestamp", "")) >= since_timestamp
        ]
        results.sort(key=lambda i: str(i.get("timestamp", "")))
        return results

    # ------------------------------------------------------------------
    # Fraud pattern
    # ------------------------------------------------------------------

    def get_pattern(self, table_name: str, pattern_name: str) -> dict | None:
        """Fetch a single pattern by name."""
        return self._get_item(table_name, f"PATTERN#{pattern_name}")

    def put_pattern(self, table_name: str, pattern: dict) -> None:
        """Store a fraud pattern item."""
        self._put_item(table_name, pattern)

    def scan_patterns(self, table_name: str, status: str = "ACTIVE") -> list[dict]:
        """Return all patterns whose 'status' field equals *status*."""
        return [
            item
            for item in self._all_items(table_name)
            if item.get("status") == status
        ]

    # ------------------------------------------------------------------
    # AML risk score
    # ------------------------------------------------------------------

    def get_aml_risk_score(self, table_name: str, user_id: str) -> dict | None:
        """Fetch the AML risk score record for *user_id*, or None."""
        return self._get_item(table_name, f"AML#{user_id}")

    def put_aml_risk_score(self, table_name: str, score: dict) -> None:
        """Store an AML risk score record."""
        self._put_item(table_name, score)

    # ------------------------------------------------------------------
    # Investigation case
    # ------------------------------------------------------------------

    def get_investigation_case(self, table_name: str, case_id: str) -> dict | None:
        """Fetch an investigation case by *case_id*, or None."""
        return self._get_item(table_name, f"CASE#{case_id}")

    def put_investigation_case(self, table_name: str, case: dict) -> None:
        """Store an investigation case item."""
        self._put_item(table_name, case)

    def query_investigations_by_user(
        self,
        table_name: str,
        user_id: str,
        status: str | None = None,
        index_name: str = "userId-status-index",
    ) -> list[dict]:
        """Return all investigation cases for *user_id*, optionally filtered by status."""
        results = [
            item
            for item in self._all_items(table_name)
            if item.get("userId") == user_id
            and (status is None or item.get("status") == status)
        ]
        return results

    def query_open_investigations(
        self,
        table_name: str,
        index_name: str = "status-openedAt-index",
    ) -> list[dict]:
        """Return all cases with status=OPEN, sorted ascending by openedAt."""
        results = [
            item
            for item in self._all_items(table_name)
            if item.get("status") == "OPEN"
        ]
        results.sort(key=lambda i: str(i.get("opened_at", "")))
        return results


# ---------------------------------------------------------------------------
# InMemoryOpenSearchClient
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero norm to avoid division by zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryOpenSearchClient:
    """Pure Python stand-in for OpenSearchClient.

    Stores indexed documents as a list of (doc_id, vector, metadata) tuples.
    kNN search performs exact cosine similarity against all stored vectors and
    returns the top-k results by score, mirroring the production cosine-space
    HNSW index.

    The interface exactly mirrors src/clients/opensearch_client.py.
    """

    def __init__(self) -> None:
        # Each entry: {"id": str, "vector": list[float], "metadata": dict}
        self._index: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index_if_not_exists(
        self,
        index_name: str,
        dimension: int = 1536,
    ) -> None:
        """No-op in the mock — the in-memory list accepts any dimension."""
        pass

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_vector(
        self,
        index_name: str,
        doc_id: str,
        vector: list[float],
        metadata: dict,
    ) -> None:
        """Append (or overwrite) a document in the in-memory index.

        If a document with the same *doc_id* already exists it is replaced,
        matching the upsert semantics of the production client.
        """
        # Remove any existing entry with this doc_id (upsert semantics).
        self._index = [e for e in self._index if e["id"] != doc_id]
        self._index.append({"id": doc_id, "vector": vector, "metadata": metadata})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def knn_search(
        self,
        index_name: str,
        vector: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Return the top-k most similar documents by cosine similarity.

        Each result dict matches the production format:
            {"id": str, "score": float, "metadata": dict}
        """
        if not self._index:
            return []

        scored = [
            {
                "id": entry["id"],
                "score": _cosine_similarity(vector, entry["vector"]),
                "metadata": entry["metadata"],
            }
            for entry in self._index
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# InMemoryRedisClient
# ---------------------------------------------------------------------------


class InMemoryRedisClient:
    """Pure Python stand-in for RedisClient.

    Backed by a plain dict.  TTL semantics are not enforced — keys persist for
    the lifetime of the object.  Rate-limit checks always return True (allow)
    so tests are not gated by artificial rate limits.

    The interface exactly mirrors src/clients/redis_client.py.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persona cache
    # ------------------------------------------------------------------

    def get_persona_cache(self, user_id: str) -> dict | None:
        """Return the cached persona for *user_id*, or None on cache miss."""
        return self._cache.get(f"persona:{user_id}")

    def set_persona_cache(
        self,
        user_id: str,
        persona: dict,
        ttl: int = 3600,
    ) -> None:
        """Cache *persona* for *user_id*. TTL is accepted but not enforced."""
        self._cache[f"persona:{user_id}"] = persona

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(
        self,
        user_id: str,
        window_seconds: int = 60,
        max_requests: int = 10,
    ) -> bool:
        """Always returns True — E2E tests are not rate-limited."""
        return True

    # ------------------------------------------------------------------
    # Pattern cache
    # ------------------------------------------------------------------

    def get_pattern_cache(self) -> list[dict] | None:
        """Return the cached pattern list, or None on cache miss."""
        return self._cache.get("patterns:all")

    def set_pattern_cache(
        self,
        patterns: list[dict],
        ttl: int = 86400,
    ) -> None:
        """Cache *patterns*. TTL is accepted but not enforced."""
        self._cache["patterns:all"] = patterns
