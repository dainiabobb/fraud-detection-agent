"""
Fraud analyst handler — synchronous invocation from the swarm orchestrator.

Responsibility:
  - Accept a payload carrying a transaction, persona, similar transactions,
    and escalation context.
  - Call FraudAnalystService.analyze() which uses Claude Sonnet for deep-dive
    fraud classification.
  - Return the serialised FraudDecision so the orchestrator can combine it
    with the AML specialist's output.

All AWS clients are initialised at module level for Lambda container reuse.
"""

import json
import logging
import os
from typing import Any

import boto3

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.opensearch_client import OpenSearchClient
from src.config import Config
from src.services.fraud_analyst_service import FraudAnalystService
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
    _boto3_opensearch_s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    dynamo_client = DynamoDBClient(_boto3_dynamodb)
    bedrock_client = BedrockClient(_boto3_bedrock)
    metrics = FraudMetrics(_boto3_cloudwatch)

    # OpenSearch is used by the fraud analyst for similar-transaction retrieval.
    # Import here so the error message is clear if the package is missing.
    from opensearchpy import OpenSearch, RequestsHttpConnection  # noqa: E402 (deferred for clarity)

    _opensearch_raw = OpenSearch(
        hosts=[{"host": config.opensearch_endpoint, "port": 443}],
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    opensearch_client = OpenSearchClient(_opensearch_raw)

    fraud_analyst_service = FraudAnalystService(
        config=config,
        dynamo_client=dynamo_client,
        opensearch_client=opensearch_client,
        bedrock_client=bedrock_client,
        metrics=metrics,
    )

    logger.info(json.dumps({"message": "Fraud analyst handler initialised (cold start)"}))

except Exception as _init_err:
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during fraud analyst handler module initialisation",
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
    """Lambda entry point for deep-dive fraud analysis.

    Expected event shape (all from the swarm orchestrator):
      {
        "transaction_dict":       <raw transaction dict>,
        "persona_dict":           <BehavioralPersona dict | None>,
        "similar_transactions":   [<transaction dict>, ...],
        "escalation_context":     { "sentinel_reasoning": ..., ... }
      }

    Args:
        event:   Payload dict (see above).
        context: Lambda context object (unused but required by signature).

    Returns:
        Serialised FraudDecision dict (model_dump()) so the orchestrator can
        deserialise it back into a FraudDecision.

    Raises:
        Exception: Propagated so the orchestrator's invoke_sync() raises a
                   RuntimeError and can handle the failure gracefully.
    """
    transaction_dict: dict = event.get("transaction_dict", {})
    persona_dict: dict | None = event.get("persona_dict")
    similar_transactions: list[dict] = event.get("similar_transactions", [])
    escalation_context: dict = event.get("escalation_context", {})

    transaction_id: str = transaction_dict.get("transaction_id", "UNKNOWN")
    user_id: str = transaction_dict.get("user_id", "UNKNOWN")

    logger.info(
        json.dumps(
            {
                "message": "Fraud analyst handler invoked",
                "transactionId": transaction_id,
                "userId": user_id,
                "hasSimilarTransactions": len(similar_transactions) > 0,
                "hasPersona": persona_dict is not None,
            }
        )
    )

    try:
        decision = fraud_analyst_service.analyze(
            transaction_dict=transaction_dict,
            persona_dict=persona_dict,
            similar_transactions=similar_transactions,
            escalation_context=escalation_context,
        )

        result: dict = decision.model_dump()

        logger.info(
            json.dumps(
                {
                    "message": "Fraud analyst decision produced",
                    "transactionId": transaction_id,
                    "userId": user_id,
                    "verdict": decision.verdict,
                    "confidence": decision.confidence,
                    "tokensUsed": decision.tokens_used,
                }
            )
        )

        return result

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "Fraud analyst analysis failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "transactionId": transaction_id,
                    "userId": user_id,
                }
            ),
            exc_info=True,
        )
        raise
