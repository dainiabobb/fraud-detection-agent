"""
Sentinel handler — Kinesis trigger (batch size 10).

Responsibility:
  - Decode each Kinesis record's base64 payload into a raw transaction dict.
  - Call SentinelService.process_transaction() for fast Haiku-based triage.
  - If the decision verdict is ESCALATE, build the escalation payload and
    invoke the SwarmOrchestratorService asynchronously (fire-and-forget via
    the swarm_orchestrator Lambda).
  - Return a batchItemFailures list so Kinesis partial-batch failure handling
    retries only the records that errored.

All AWS clients are initialised at module level (outside the handler function)
so they survive across Lambda warm invocations.
"""

import base64
import json
import logging
import os
import time
from typing import Any

import boto3
import redis
from opensearchpy import OpenSearch, RequestsHttpConnection

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.lambda_client import LambdaInvokeClient
from src.clients.opensearch_client import OpenSearchClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.services.sentinel_service import SentinelService
from src.services.swarm_orchestrator_service import SwarmOrchestratorService
from src.utils.metrics import FraudMetrics

# ---------------------------------------------------------------------------
# Structured logging — emit JSON-compatible records for CloudWatch Logs
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level initialisation — runs once per Lambda container (cold start)
# ---------------------------------------------------------------------------
try:
    config = Config.from_env()

    _boto3_dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    _boto3_bedrock = boto3.client("bedrock-runtime", region_name=config.bedrock_region)
    _boto3_lambda = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    _boto3_cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    _redis_raw = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
    )

    _opensearch_raw = OpenSearch(
        hosts=[{"host": config.opensearch_endpoint, "port": 443}],
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    dynamo_client = DynamoDBClient(_boto3_dynamodb)
    opensearch_client = OpenSearchClient(_opensearch_raw)
    redis_client = RedisClient(_redis_raw)
    bedrock_client = BedrockClient(_boto3_bedrock)
    lambda_invoke_client = LambdaInvokeClient(_boto3_lambda)
    metrics = FraudMetrics(_boto3_cloudwatch)

    sentinel_service = SentinelService(
        config=config,
        dynamo_client=dynamo_client,
        opensearch_client=opensearch_client,
        redis_client=redis_client,
        bedrock_client=bedrock_client,
        metrics=metrics,
    )

    swarm_orchestrator_service = SwarmOrchestratorService(
        config=config,
        lambda_invoke_client=lambda_invoke_client,
    )

    logger.info(json.dumps({"message": "Sentinel handler initialised (cold start)"}))

except Exception as _init_err:
    # Re-raise so Lambda fails fast with a clear error rather than a cryptic
    # AttributeError on the first invocation.
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during Sentinel handler module initialisation",
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
    """Lambda entry point for Kinesis-triggered fraud triage.

    Args:
        event:   Kinesis event dict with a "Records" list.
        context: Lambda context object (unused but required by signature).

    Returns:
        {"batchItemFailures": [...]} so the Kinesis trigger retries only
        failed records rather than the entire batch.
    """
    records: list[dict] = event.get("Records", [])
    batch_item_failures: list[dict[str, str]] = []

    logger.info(
        json.dumps(
            {
                "message": "Sentinel handler invoked",
                "batchSize": len(records),
            }
        )
    )

    for record in records:
        sequence_number: str = record["kinesis"]["sequenceNumber"]
        start_ms = int(time.time() * 1000)

        try:
            # ------------------------------------------------------------------
            # 1. Decode the Kinesis record payload
            # ------------------------------------------------------------------
            raw_data: bytes = base64.b64decode(record["kinesis"]["data"])
            transaction_dict: dict = json.loads(raw_data)
            transaction_id: str = transaction_dict.get("transaction_id", "UNKNOWN")
            user_id: str = transaction_dict.get("user_id", "UNKNOWN")

            logger.info(
                json.dumps(
                    {
                        "message": "Processing transaction",
                        "transactionId": transaction_id,
                        "userId": user_id,
                        "sequenceNumber": sequence_number,
                    }
                )
            )

            # ------------------------------------------------------------------
            # 2. Run Sentinel triage
            # ------------------------------------------------------------------
            decision = sentinel_service.process_transaction(transaction_dict)

            latency_ms = int(time.time() * 1000) - start_ms

            logger.info(
                json.dumps(
                    {
                        "message": "Sentinel decision recorded",
                        "transactionId": transaction_id,
                        "userId": user_id,
                        "verdict": decision.verdict,
                        "confidence": decision.confidence,
                        "tier": decision.tier,
                        "latencyMs": latency_ms,
                    }
                )
            )

            # ------------------------------------------------------------------
            # 3. Escalate to swarm orchestrator when required
            # ------------------------------------------------------------------
            if decision.verdict == "ESCALATE":
                escalation_payload: dict = {
                    "transaction": transaction_dict,
                    # persona / similar_transactions are resolved inside the
                    # orchestrator; pass None here so it fetches them fresh.
                    "persona": None,
                    "similar_transactions": [],
                    "escalation_target": decision.escalation_target,
                    "aml_signals": [],
                    "escalation_context": {
                        "sentinel_reasoning": decision.reasoning,
                        "sentinel_confidence": decision.confidence,
                        "pattern_matches": decision.pattern_matches,
                    },
                }

                swarm_orchestrator_service.orchestrate(escalation_payload)

                logger.info(
                    json.dumps(
                        {
                            "message": "Escalation dispatched to swarm orchestrator",
                            "transactionId": transaction_id,
                            "userId": user_id,
                            "escalationTarget": decision.escalation_target,
                        }
                    )
                )

        except Exception as exc:
            # ------------------------------------------------------------------
            # 4. Record the failure — do NOT re-raise so the loop continues
            # ------------------------------------------------------------------
            logger.error(
                json.dumps(
                    {
                        "message": "Failed to process Kinesis record",
                        "errorType": type(exc).__name__,
                        "error": str(exc),
                        "sequenceNumber": sequence_number,
                    }
                ),
                exc_info=True,
            )
            # Signal to Kinesis that this specific record should be retried.
            batch_item_failures.append({"itemIdentifier": sequence_number})

    logger.info(
        json.dumps(
            {
                "message": "Sentinel batch complete",
                "totalRecords": len(records),
                "failures": len(batch_item_failures),
            }
        )
    )

    return {"batchItemFailures": batch_item_failures}
