"""
Shared pytest fixtures for the fraud detection agent test suite.

All service-level fixtures return MagicMock instances so individual tests can
configure return values without reaching real AWS endpoints.  The Config
fixture uses known-safe sentinel values that exercise no environment reads.
"""

from unittest.mock import MagicMock

import pytest

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.lambda_client import LambdaInvokeClient
from src.clients.opensearch_client import OpenSearchClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.utils.metrics import FraudMetrics


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transaction() -> dict:
    """Realistic inbound transaction payload as it would arrive from Kinesis."""
    return {
        "transaction_id": "txn-abc123",
        "user_id": "user-7890",
        "amount": 145.50,
        "currency": "USD",
        "merchant_name": "Whole Foods Market",
        "merchant_category": "Groceries",
        "merchant_city": "Dallas",
        "merchant_country": "US",
        "channel": "in-store",
        "timestamp": "2026-03-31T14:05:00Z",
        "ip_address": "192.168.1.100",
        "device_id": "device-xyz",
        "card_last_four": "4321",
    }


@pytest.fixture
def sample_persona() -> dict:
    """Behavioral persona matching the README JSON schema exactly."""
    return {
        "user_id": "user-7890",
        "geo_footprint": [
            {"city": "Dallas", "state": "TX", "country": "US", "frequency": 0.65}
        ],
        "category_anchors": [
            {
                "category": "Groceries",
                "avg_amount": 120.0,
                "frequency": "weekly",
                "std_deviation": 35.0,
            }
        ],
        "velocity": {
            "daily_txn_count": 4.2,
            "daily_spend_amount": 185.0,
            "max_single_txn": 450.0,
            "hourly_burst_limit": 3,
        },
        "anomaly_history": {
            "false_positive_triggers": ["travel-to-miami"],
            "confirmed_fraud_count": 0,
        },
        "ip_footprint": [
            {
                "region": "Dallas-Fort Worth, TX",
                "frequency": 0.65,
                "typical_asn": "AS7018 AT&T",
            }
        ],
        "temporal_profile": {
            "active_hours": [7, 22],
            "peak_hour": 12,
            "weekend_ratio": 0.28,
            "timezone_estimate": "America/Chicago",
        },
        "aml_profile": {
            "deposit_pattern": {"avg_amount": 850.0, "pct_near_threshold": 0.02},
            "transfer_pattern": {
                "avg_count": 3,
                "unique_counterparties": 12,
                "pct_high_risk_jurisdictions": 0.0,
            },
            "round_trip_score": 0.0,
            "economic_profile_match": True,
            "typology_flags": [],
        },
        "version": "v1",
    }


@pytest.fixture
def sample_config() -> Config:
    """Config instance with test-safe sentinel values — no env reads."""
    return Config(
        dynamo_table_personas="FraudPersonas",
        dynamo_table_decisions="FraudDecisions",
        dynamo_table_patterns="FraudPatterns",
        dynamo_table_aml_risk="AMLRiskScores",
        dynamo_table_investigations="InvestigationCases",
        opensearch_endpoint="test-endpoint",
        opensearch_index="fraud-vectors",
        redis_host="localhost",
        redis_port=6379,
        bedrock_region="us-east-1",
        fraud_analyst_function_name="fraud-analyst",
        aml_specialist_function_name="aml-specialist",
        persona_cache_ttl=3600,
        embedding_dimension=1536,
        sentinel_model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        sonnet_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        titan_embedding_model_id="amazon.titan-embed-text-v2:0",
        auto_approve_threshold=0.85,
        escalation_threshold=0.75,
        aml_investigation_threshold=50,
        aml_compliance_threshold=80,
    )


# ---------------------------------------------------------------------------
# Client mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dynamodb_client() -> MagicMock:
    """MagicMock that stands in for DynamoDBClient."""
    return MagicMock(spec=DynamoDBClient)


@pytest.fixture
def mock_opensearch_client() -> MagicMock:
    """MagicMock that stands in for OpenSearchClient."""
    return MagicMock(spec=OpenSearchClient)


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """MagicMock that stands in for RedisClient."""
    return MagicMock(spec=RedisClient)


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """MagicMock that stands in for BedrockClient."""
    return MagicMock(spec=BedrockClient)


@pytest.fixture
def mock_lambda_client() -> MagicMock:
    """MagicMock that stands in for LambdaInvokeClient."""
    return MagicMock(spec=LambdaInvokeClient)


@pytest.fixture
def mock_metrics() -> MagicMock:
    """MagicMock that stands in for FraudMetrics."""
    return MagicMock(spec=FraudMetrics)
