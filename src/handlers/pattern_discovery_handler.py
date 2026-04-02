"""
Pattern discovery handler — EventBridge scheduled trigger, daily at 06:00 UTC.

Responsibility:
  - Call PatternDiscoveryService.discover_patterns(), which mines recent
    fraud decisions for emergent patterns using Claude Sonnet.
  - Log a structured summary of how many patterns were discovered, refined,
    and retired.
  - Return the summary dict (available in EventBridge execution logs).

All AWS clients are initialised at module level for Lambda container reuse.
"""

import json
import logging
import os
from typing import Any

import boto3
import redis

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.services.pattern_discovery_service import PatternDiscoveryService
from src.utils.metrics import FraudMetrics

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level initialisation (cold start)
# ---------------------------------------------------------------------------
try:
    config = Config.from_env()

    _boto3_dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    _boto3_bedrock = boto3.client("bedrock-runtime", region_name=config.bedrock_region)
    _boto3_cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    _redis_raw = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
    )

    dynamo_client = DynamoDBClient(_boto3_dynamodb)
    bedrock_client = BedrockClient(_boto3_bedrock)
    redis_client = RedisClient(_redis_raw)
    metrics = FraudMetrics(_boto3_cloudwatch)

    pattern_discovery_service = PatternDiscoveryService(
        config=config,
        dynamo_client=dynamo_client,
        bedrock_client=bedrock_client,
        redis_client=redis_client,
        metrics=metrics,
    )

    logger.info(json.dumps({"message": "Pattern discovery handler initialised (cold start)"}))

except Exception as _init_err:
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during pattern discovery handler module initialisation",
                "errorType": type(_init_err).__name__,
                "detail": str(_init_err),
            }
        )
    )
    raise


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for daily fraud pattern discovery.

    Triggered by an EventBridge rule at 06:00 UTC every day.  The *event*
    payload is the standard EventBridge scheduled-event structure but the
    handler does not use any of its fields.

    Args:
        event:   EventBridge scheduled event dict (fields unused).
        context: Lambda context object (unused but required by signature).

    Returns:
        Summary dict with keys:
          - "patternsDiscovered": int — new patterns written to DynamoDB
          - "patternsRefined":    int — existing patterns updated
          - "patternsRetired":    int — patterns marked RETIRED
          - "status":             "ok" | "error"

    Raises:
        Exception: Propagated so EventBridge can record the execution as failed
                   and trigger alerting via CloudWatch alarms.
    """
    logger.info(json.dumps({"message": "Pattern discovery handler invoked"}))

    try:
        # The service returns a summary dict; the exact schema is service-defined
        # but callers can depend on the three count keys documented above.
        result: dict = pattern_discovery_service.discover_patterns()

        patterns_discovered: int = result.get("patternsDiscovered", 0)
        patterns_refined: int = result.get("patternsRefined", 0)
        patterns_retired: int = result.get("patternsRetired", 0)

        # Publish individual CloudWatch metrics for each category so dashboards
        # and alarms can track discovery health independently.
        for _ in range(patterns_discovered):
            metrics.record_new_pattern()
        for _ in range(patterns_retired):
            metrics.record_pattern_retired()

        logger.info(
            json.dumps(
                {
                    "message": "Pattern discovery run complete",
                    "patternsDiscovered": patterns_discovered,
                    "patternsRefined": patterns_refined,
                    "patternsRetired": patterns_retired,
                }
            )
        )

        return {
            "status": "ok",
            "patternsDiscovered": patterns_discovered,
            "patternsRefined": patterns_refined,
            "patternsRetired": patterns_retired,
        }

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "Pattern discovery run failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            exc_info=True,
        )
        raise
