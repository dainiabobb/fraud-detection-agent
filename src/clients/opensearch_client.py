"""
OpenSearchClient — thin wrapper around an opensearch-py OpenSearch client.

Constructor injection allows tests to replace the client with a mock without
patching the opensearch_py module globally.

Index convention used by this service:
  - Vectors stored under the field name "embedding".
  - Metadata fields are stored as top-level document attributes alongside
    "embedding", so callers do not need to nest data under a sub-key.
  - kNN index uses the nmslib engine with cosine similarity to match
    Amazon Bedrock Titan Embeddings V2 output (1536-dim, cosine-comparable).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Wraps an opensearch-py OpenSearch instance for vector search operations."""

    def __init__(self, opensearch_client: Any) -> None:
        """
        Args:
            opensearch_client: An initialised opensearch_py.OpenSearch instance.
                               Injected so tests can supply a mock.
        """
        self._client = opensearch_client

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index_if_not_exists(
        self,
        index_name: str,
        dimension: int = 1536,
    ) -> None:
        """Create a kNN-enabled index if it does not already exist.

        Uses the nmslib HNSW engine with cosine space_type, which is
        compatible with Amazon Titan Embeddings V2 vectors.

        Args:
            index_name: OpenSearch index name.
            dimension:  Vector dimension (default 1536 for Titan Embeddings V2).
        """
        try:
            if self._client.indices.exists(index=index_name):
                logger.debug("OpenSearch index already exists", extra={"index": index_name})
                return

            body: dict[str, Any] = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 512,
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "nmslib",
                                "space_type": "cosinesimil",
                                "parameters": {
                                    "ef_construction": 512,
                                    "m": 16,
                                },
                            },
                        }
                    }
                },
            }
            self._client.indices.create(index=index_name, body=body)
            logger.info("Created OpenSearch kNN index", extra={"index": index_name})
        except Exception:
            logger.exception(
                "OpenSearch create_index_if_not_exists failed",
                extra={"index": index_name},
            )
            raise

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
        """Index a document that combines an embedding vector with arbitrary metadata.

        The document is stored as:
            { "embedding": [...], **metadata }

        Existing documents with the same *doc_id* are overwritten (upsert
        semantics via the index API).

        Args:
            index_name: Target index.
            doc_id:     Unique document identifier.
            vector:     Embedding vector (must match index dimension).
            metadata:   Arbitrary key/value pairs stored alongside the vector.
        """
        try:
            body: dict[str, Any] = {"embedding": vector, **metadata}
            self._client.index(
                index=index_name,
                id=doc_id,
                body=body,
                refresh=False,  # async refresh — callers should not rely on immediate visibility
            )
            logger.debug(
                "Indexed vector document",
                extra={"index": index_name, "doc_id": doc_id},
            )
        except Exception:
            logger.exception(
                "OpenSearch index_vector failed",
                extra={"index": index_name, "doc_id": doc_id},
            )
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def knn_search(
        self,
        index_name: str,
        vector: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Perform approximate nearest-neighbour search and return the top-k results.

        Each result dict contains:
            - "id":     OpenSearch document ID
            - "score":  Cosine similarity score returned by OpenSearch
            - "metadata": All document fields except "embedding"

        Args:
            index_name: Index to search.
            vector:     Query embedding vector.
            k:          Number of nearest neighbours to return (default 5).

        Returns:
            List of result dicts sorted by descending score (highest similarity first).
        """
        try:
            query: dict[str, Any] = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": vector,
                            "k": k,
                        }
                    }
                },
                # Exclude the raw vector from the response to keep payloads small.
                "_source": {"excludes": ["embedding"]},
            }
            response = self._client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            results: list[dict] = []
            for hit in hits:
                results.append(
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "metadata": hit.get("_source", {}),
                    }
                )
            return results
        except Exception:
            logger.exception(
                "OpenSearch knn_search failed",
                extra={"index": index_name, "k": k},
            )
            raise
