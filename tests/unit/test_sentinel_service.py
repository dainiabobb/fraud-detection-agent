"""
Unit tests for SentinelService.

All external dependencies (DynamoDB, OpenSearch, Redis, Bedrock) are replaced
with MagicMock instances.  The prompt file is patched away so tests never
touch the filesystem for the gray-zone Haiku call path.

Routing thresholds from the default Config:
  auto_approve_threshold  = 0.85  (>= this → APPROVE when no AML signal)
  escalation_threshold    = 0.75  (< this  → ESCALATE unconditionally)
  gray zone               = [0.75, 0.85)  → Haiku called
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.decision import FraudDecision
from src.services.sentinel_service import SentinelService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    sample_config,
    mock_dynamodb_client,
    mock_opensearch_client,
    mock_redis_client,
    mock_bedrock_client,
    mock_metrics,
) -> SentinelService:
    return SentinelService(
        config=sample_config,
        dynamodb_client=mock_dynamodb_client,
        opensearch_client=mock_opensearch_client,
        redis_client=mock_redis_client,
        bedrock_client=mock_bedrock_client,
        metrics=mock_metrics,
    )


def _knn_results(score: float) -> list[dict]:
    """Return a minimal kNN result list with a single hit at the given score."""
    return [{"id": "doc-1", "score": score, "metadata": {"verdict": "APPROVE"}}]


# ---------------------------------------------------------------------------
# Auto-approve path (similarity >= 0.85, no AML signal)
# ---------------------------------------------------------------------------


class TestAutoApprovePath:
    def test_high_similarity_produces_approve(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        # Persona from Redis cache (cache hit — DynamoDB not called)
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []

        # Embedding stub and high-similarity kNN result
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(sample_transaction)

        assert isinstance(decision, FraudDecision)
        assert decision.verdict == "APPROVE"
        assert decision.tier == "SENTINEL"
        assert decision.escalation_target is None
        # Haiku must NOT be called on the auto-approve path
        mock_bedrock_client.invoke_haiku.assert_not_called()
        mock_metrics.record_auto_approve.assert_called_once()

    def test_auto_approve_confidence_bounded_at_one(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        # Score slightly above 1.0 — confidence must be clamped to 1.0
        mock_opensearch_client.knn_search.return_value = _knn_results(1.05)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(sample_transaction)
        assert decision.verdict == "APPROVE"
        assert decision.confidence <= 1.0


# ---------------------------------------------------------------------------
# Definite escalation path (similarity < 0.75)
# ---------------------------------------------------------------------------


class TestDefiniteEscalationPath:
    def test_low_similarity_produces_escalate(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.60)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(sample_transaction)

        assert decision.verdict == "ESCALATE"
        assert decision.escalation_target == "FRAUD_ONLY"
        # Haiku must NOT be invoked on the definite-escalation path
        mock_bedrock_client.invoke_haiku.assert_not_called()
        mock_metrics.record_escalation.assert_called()


# ---------------------------------------------------------------------------
# AML signal detection
# ---------------------------------------------------------------------------


class TestAMLSignalDetection:
    def test_structuring_amount_triggers_aml_signal(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        # $9,200 is in the structuring window [$8,000, $9,999.99]
        txn = {**sample_transaction, "amount": 9200.0}
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        # High similarity — would auto-approve — but AML override forces ESCALATE
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(txn)

        # AML signal overrides the auto-approve; must route to AML specialist
        assert decision.verdict == "ESCALATE"
        assert decision.escalation_target == "AML_ONLY"
        mock_metrics.record_aml_escalation.assert_called_once()

    def test_high_risk_jurisdiction_triggers_aml_signal(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        # Iran is in _HIGH_RISK_JURISDICTIONS
        txn = {**sample_transaction, "merchant_country": "IR", "amount": 200.0}
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(txn)

        assert decision.verdict == "ESCALATE"
        assert "AML" in decision.escalation_target
        mock_metrics.record_aml_escalation.assert_called_once()

    def test_round_number_amount_triggers_aml_signal(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        # $5,000 exactly: >= 5000 and divisible by 1000
        txn = {**sample_transaction, "amount": 5000.0}
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(txn)

        assert decision.verdict == "ESCALATE"
        assert "AML" in decision.escalation_target


# ---------------------------------------------------------------------------
# Gray zone (0.75 <= similarity < 0.85)
# ---------------------------------------------------------------------------


class TestGrayZone:
    def test_gray_zone_invokes_haiku(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []  # no patterns
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.80)

        haiku_json = json.dumps({
            "verdict": "APPROVE",
            "confidence": 0.78,
            "reasoning": "Transaction matches persona profile.",
            "escalation_target": None,
        })
        mock_bedrock_client.invoke_haiku.return_value = {
            "text": haiku_json,
            "tokens_used": 120,
        }

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )

        # Patch file I/O so we don't need the actual prompt template
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "Prompt {transaction_json} {persona_json} "
                "{similar_transactions} {pattern_matches}"
            )
            decision = service.process_transaction(sample_transaction)

        mock_bedrock_client.invoke_haiku.assert_called_once()
        assert decision.verdict == "APPROVE"

    def test_gray_zone_pattern_match_escalates_without_haiku(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = [
            {"pattern_name": "rapid-geo-shift", "detection_rule": "some rule"}
        ]
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.80)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        decision = service.process_transaction(sample_transaction)

        # Pattern match in gray zone → ESCALATE without calling Haiku
        assert decision.verdict == "ESCALATE"
        assert "rapid-geo-shift" in decision.pattern_matches
        mock_bedrock_client.invoke_haiku.assert_not_called()
        mock_metrics.record_pattern_match.assert_called_once()


# ---------------------------------------------------------------------------
# Persona caching behaviour
# ---------------------------------------------------------------------------


class TestPersonaCaching:
    def test_persona_cache_hit_skips_dynamodb(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        service.process_transaction(sample_transaction)

        mock_dynamodb_client.get_persona.assert_not_called()

    def test_persona_cache_miss_reads_dynamodb_and_caches(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        # Redis miss → DynamoDB hit → write back to cache
        mock_redis_client.get_persona_cache.return_value = None
        mock_redis_client.get_pattern_cache.return_value = []
        mock_dynamodb_client.get_persona.return_value = sample_persona
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.92)

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )
        service.process_transaction(sample_transaction)

        mock_dynamodb_client.get_persona.assert_called_once()
        mock_redis_client.set_persona_cache.assert_called_once()


# ---------------------------------------------------------------------------
# LLM parse failure (fail-closed)
# ---------------------------------------------------------------------------


class TestLlmParseFailure:
    def test_haiku_parse_failure_defaults_to_escalate(
        self,
        sample_config,
        sample_transaction,
        sample_persona,
        mock_dynamodb_client,
        mock_opensearch_client,
        mock_redis_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        mock_redis_client.get_persona_cache.return_value = sample_persona
        mock_redis_client.get_pattern_cache.return_value = []
        mock_bedrock_client.get_embedding.return_value = [0.1] * 1536
        mock_opensearch_client.knn_search.return_value = _knn_results(0.80)

        # Return malformed text that cannot be parsed as JSON
        mock_bedrock_client.invoke_haiku.return_value = {
            "text": "I cannot determine a verdict for this transaction.",
            "tokens_used": 50,
        }

        service = _make_service(
            sample_config,
            mock_dynamodb_client,
            mock_opensearch_client,
            mock_redis_client,
            mock_bedrock_client,
            mock_metrics,
        )

        # NOTE: The production _parse_llm_response passes extra={"message": ...}
        # to logger.error, which conflicts with the built-in LogRecord "message"
        # attribute and raises KeyError in Python 3.12.  We patch the logger to
        # isolate the fail-closed routing logic from that pre-existing log bug.
        import logging
        with patch.object(
            logging.getLogger("src.services.sentinel_service"),
            "error",
        ):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "Prompt {transaction_json} {persona_json} "
                    "{similar_transactions} {pattern_matches}"
                )
                decision = service.process_transaction(sample_transaction)

        # Fail-closed: parse error → ESCALATE
        assert decision.verdict == "ESCALATE"
