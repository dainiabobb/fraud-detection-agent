"""
Swarm orchestrator handler — asynchronous invocation from Sentinel.

Responsibility:
  - Accept the escalation payload that Sentinel fires asynchronously.
  - Delegate to SwarmOrchestratorService.orchestrate(), which fans out to the
    FraudAnalyst and/or AMLSpecialist Lambda functions and synthesises their
    results into a final decision.
  - Return a success status dict (ignored by Sentinel because the invocation
    was fire-and-forget, but available for synchronous test invocations).

All AWS clients are initialised at module level for Lambda container reuse.
"""

import json
import logging
import os
from typing import Any

import boto3

from src.clients.lambda_client import LambdaInvokeClient
from src.config import Config
from src.services.swarm_orchestrator_service import SwarmOrchestratorService

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

    _boto3_lambda = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    lambda_invoke_client = LambdaInvokeClient(_boto3_lambda)

    swarm_orchestrator_service = SwarmOrchestratorService(
        config=config,
        lambda_invoke_client=lambda_invoke_client,
    )

    logger.info(json.dumps({"message": "Swarm orchestrator handler initialised (cold start)"}))

except Exception as _init_err:
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during swarm orchestrator handler module initialisation",
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
    """Lambda entry point for the swarm orchestration step.

    The event is the escalation payload dict produced by the Sentinel handler:
      {
        "transaction":        <raw transaction dict>,
        "persona":            <BehavioralPersona dict | None>,
        "similar_transactions": [...],
        "escalation_target":  "FRAUD_ONLY" | "AML_ONLY" | "BOTH",
        "aml_signals":        [...],
        "escalation_context": { "sentinel_reasoning": ..., ... }
      }

    Args:
        event:   Escalation payload dict (see above).
        context: Lambda context object (unused but required by signature).

    Returns:
        {"status": "ok", "transactionId": <id>} on success.

    Raises:
        Exception: Propagated so Lambda marks the invocation as failed and the
                   async retry policy kicks in (if configured).
    """
    transaction_id: str = (event.get("transaction") or {}).get("transaction_id", "UNKNOWN")
    user_id: str = (event.get("transaction") or {}).get("user_id", "UNKNOWN")

    logger.info(
        json.dumps(
            {
                "message": "Swarm orchestrator handler invoked",
                "transactionId": transaction_id,
                "userId": user_id,
                "escalationTarget": event.get("escalation_target"),
            }
        )
    )

    try:
        swarm_orchestrator_service.orchestrate(event)

        logger.info(
            json.dumps(
                {
                    "message": "Swarm orchestration complete",
                    "transactionId": transaction_id,
                    "userId": user_id,
                }
            )
        )

        return {"status": "ok", "transactionId": transaction_id}

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "Swarm orchestration failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "transactionId": transaction_id,
                    "userId": user_id,
                }
            ),
            exc_info=True,
        )
        # Re-raise so Lambda records the failure and the async retry policy
        # (MaximumRetryAttempts on the event source mapping) can retry.
        raise
