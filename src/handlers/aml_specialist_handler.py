"""
AML specialist handler — synchronous invocation from the swarm orchestrator.

Responsibility:
  - Accept a payload with transaction, persona, optional user history, optional
    current AML score, and optional existing investigation case.
  - If user_history or current_aml_score are absent from the payload, fetch
    them from DynamoDB before delegating to AMLSpecialistService.analyze().
  - Return the serialised result dict for the orchestrator to combine with the
    fraud analyst's output.

All AWS clients are initialised at module level for Lambda container reuse.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import boto3

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.config import Config
from src.services.aml_specialist_service import AMLSpecialistService
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

    dynamo_client = DynamoDBClient(_boto3_dynamodb)
    bedrock_client = BedrockClient(_boto3_bedrock)
    metrics = FraudMetrics(_boto3_cloudwatch)

    aml_specialist_service = AMLSpecialistService(
        config=config,
        dynamo_client=dynamo_client,
        bedrock_client=bedrock_client,
        metrics=metrics,
    )

    logger.info(json.dumps({"message": "AML specialist handler initialised (cold start)"}))

except Exception as _init_err:
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during AML specialist handler module initialisation",
                "errorType": type(_init_err).__name__,
                "detail": str(_init_err),
            }
        )
    )
    raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_recent_decisions(user_id: str, months: int = 6) -> list[dict]:
    """Query DynamoDB for the user's decision history over the past *months* months.

    Uses the verdict-timestamp GSI queried for all verdicts by iterating the
    known verdict values so we get the full history regardless of outcome.

    Args:
        user_id: The user whose history to fetch.
        months:  How many months back to look (default 6).

    Returns:
        List of decision dicts, may be empty if no history exists.
    """
    since_dt = datetime.now(tz=timezone.utc) - timedelta(days=months * 30)
    since_iso: str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # The decisions table stores records keyed by transactionId, but the
    # verdict-timestamp-index GSI lets us filter by verdict + timestamp.
    # We scan all known verdict values to get the complete user history.
    verdicts = ["APPROVE", "BLOCK", "ESCALATE"]
    all_decisions: list[dict] = []

    for verdict in verdicts:
        try:
            items = dynamo_client.get_decisions_by_verdict(
                table_name=config.dynamo_table_decisions,
                verdict=verdict,
                since_timestamp=since_iso,
            )
            # Filter server-side by user_id (GSI does not include user_id as a key).
            user_items = [d for d in items if d.get("user_id") == user_id]
            all_decisions.extend(user_items)
        except Exception as exc:
            # Log and continue — a partial history is better than no analysis.
            logger.warning(
                json.dumps(
                    {
                        "message": "Failed to fetch decision history for verdict",
                        "verdict": verdict,
                        "userId": user_id,
                        "errorType": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            )

    # Sort ascending by timestamp for chronological analysis.
    all_decisions.sort(key=lambda d: d.get("timestamp", ""))
    return all_decisions


def _fetch_aml_score(user_id: str) -> dict | None:
    """Fetch the current AML risk score from DynamoDB, or None if absent.

    Args:
        user_id: The user whose AML score to retrieve.

    Returns:
        AML risk score dict, or None if no record exists.
    """
    try:
        return dynamo_client.get_aml_risk_score(
            table_name=config.dynamo_table_aml_risk,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "message": "Failed to fetch AML risk score — proceeding without it",
                    "userId": user_id,
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            )
        )
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for AML risk analysis.

    Expected event shape (all from the swarm orchestrator):
      {
        "transaction_dict":   <raw transaction dict>,
        "persona_dict":       <BehavioralPersona dict | None>,
        "user_history":       [<decision dict>, ...] | None,   # fetched if absent
        "current_aml_score":  <AMLRiskScore dict | None>,      # fetched if absent
        "existing_case":      <InvestigationCase dict | None>
      }

    If "user_history" is None or not present the handler queries the decisions
    table for the past 6 months automatically.  Similarly, if "current_aml_score"
    is None or absent it is fetched from the AML risk table.

    Args:
        event:   Payload dict (see above).
        context: Lambda context object (unused but required by signature).

    Returns:
        Result dict produced by AMLSpecialistService.analyze() — structure is
        service-defined but always includes at minimum "aml_risk_score" and
        "recommendation" keys.

    Raises:
        Exception: Propagated so the orchestrator's invoke_sync() raises and
                   can handle the failure gracefully.
    """
    transaction_dict: dict = event.get("transaction_dict", {})
    persona_dict: dict | None = event.get("persona_dict")
    existing_case: dict | None = event.get("existing_case")

    transaction_id: str = transaction_dict.get("transaction_id", "UNKNOWN")
    user_id: str = transaction_dict.get("user_id", "UNKNOWN")

    logger.info(
        json.dumps(
            {
                "message": "AML specialist handler invoked",
                "transactionId": transaction_id,
                "userId": user_id,
            }
        )
    )

    try:
        # ------------------------------------------------------------------
        # Resolve user_history — fetch from DynamoDB if caller did not supply it
        # ------------------------------------------------------------------
        user_history: list[dict] | None = event.get("user_history")
        if user_history is None:
            logger.info(
                json.dumps(
                    {
                        "message": "user_history not supplied — fetching 6-month history from DynamoDB",
                        "userId": user_id,
                    }
                )
            )
            user_history = _fetch_recent_decisions(user_id, months=6)

        # ------------------------------------------------------------------
        # Resolve current_aml_score — fetch from DynamoDB if not supplied
        # ------------------------------------------------------------------
        current_aml_score: dict | None = event.get("current_aml_score")
        if current_aml_score is None:
            logger.info(
                json.dumps(
                    {
                        "message": "current_aml_score not supplied — fetching from DynamoDB",
                        "userId": user_id,
                    }
                )
            )
            current_aml_score = _fetch_aml_score(user_id)

        # ------------------------------------------------------------------
        # Delegate to service
        # ------------------------------------------------------------------
        result: dict = aml_specialist_service.analyze(
            transaction_dict=transaction_dict,
            persona_dict=persona_dict,
            user_history=user_history,
            current_aml_score=current_aml_score,
            existing_case=existing_case,
        )

        logger.info(
            json.dumps(
                {
                    "message": "AML specialist analysis complete",
                    "transactionId": transaction_id,
                    "userId": user_id,
                    "recommendation": result.get("recommendation"),
                }
            )
        )

        return result

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "AML specialist analysis failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "transactionId": transaction_id,
                    "userId": user_id,
                }
            ),
            exc_info=True,
        )
        raise
