"""
Archaeologist handler — EventBridge scheduled trigger, weekly on Sunday at 03:00 UTC.

Responsibility:
  - Determine which users to process: use the list provided in the event, or
    fall back to scanning all known users from the DynamoDB personas table.
  - Build a fetch_history_fn closure that queries the DynamoDB decisions table
    for a given user's 24-month decision history.  This keeps the service layer
    free of direct DynamoDB references while still supporting injection of any
    callable in tests.
  - Call ArchaeologistService.run_weekly_batch(user_ids, fetch_history_fn) to
    rebuild behavioral personas from historical data using Claude Sonnet.
  - Return a batch summary dict.

All AWS clients are initialised at module level for Lambda container reuse.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

import boto3
import redis

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.services.archaeologist_service import ArchaeologistService
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

    archaeologist_service = ArchaeologistService(
        config=config,
        dynamo_client=dynamo_client,
        bedrock_client=bedrock_client,
        redis_client=redis_client,
        metrics=metrics,
    )

    logger.info(json.dumps({"message": "Archaeologist handler initialised (cold start)"}))

except Exception as _init_err:
    logger.exception(
        json.dumps(
            {
                "message": "Fatal error during archaeologist handler module initialisation",
                "errorType": type(_init_err).__name__,
                "detail": str(_init_err),
            }
        )
    )
    raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _scan_all_user_ids() -> list[str]:
    """Scan the personas table to collect all distinct user IDs.

    Iterates pages automatically.  Each item's PK has the form "USER#<sub>",
    so we strip the prefix to extract the bare user ID.

    Returns:
        Deduplicated list of user IDs known to the system.
    """
    table = dynamo_client._resource.Table(config.dynamo_table_personas)
    user_ids: list[str] = []
    scan_kwargs: dict[str, Any] = {
        # Only fetch the PK — avoids reading the full persona payload.
        "ProjectionExpression": "PK",
    }

    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            pk: str = item.get("PK", "")
            if pk.startswith("USER#"):
                user_ids.append(pk[len("USER#"):])
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    # Deduplicate in case of multiple VERSION# entries per user.
    return list(dict.fromkeys(user_ids))


def _build_fetch_history_fn(months: int = 24) -> Callable[[str], list[dict]]:
    """Return a callable that fetches a user's decision history for *months* months.

    The returned function is passed to ArchaeologistService so it can retrieve
    per-user history without coupling to a specific data source.  In tests,
    a mock callable can be injected instead.

    Args:
        months: Window size in months (default 24 for the weekly archaeology run).

    Returns:
        Callable[[str], list[dict]] — accepts user_id, returns sorted decision list.
    """
    since_dt = datetime.now(tz=timezone.utc) - timedelta(days=months * 30)
    since_iso: str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    verdicts = ["APPROVE", "BLOCK", "ESCALATE"]

    def fetch_history(user_id: str) -> list[dict]:
        all_decisions: list[dict] = []
        for verdict in verdicts:
            try:
                items = dynamo_client.get_decisions_by_verdict(
                    table_name=config.dynamo_table_decisions,
                    verdict=verdict,
                    since_timestamp=since_iso,
                )
                # The GSI is verdict + timestamp — filter by user_id in memory.
                user_items = [d for d in items if d.get("user_id") == user_id]
                all_decisions.extend(user_items)
            except Exception as exc:
                logger.warning(
                    json.dumps(
                        {
                            "message": "fetch_history: failed for one verdict — continuing",
                            "userId": user_id,
                            "verdict": verdict,
                            "errorType": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                )
        # Chronological order for the persona-building model.
        all_decisions.sort(key=lambda d: d.get("timestamp", ""))
        return all_decisions

    return fetch_history


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for weekly persona archaeology.

    Triggered by an EventBridge rule every Sunday at 03:00 UTC.

    The *event* may contain an optional "user_ids" list to restrict processing
    to a specific cohort (useful for backfills or targeted reruns).  When the
    list is absent the handler scans the personas table for all known users.

    Args:
        event:   EventBridge scheduled event dict.
                 Optional key: "user_ids" (list[str]).
        context: Lambda context object (unused but required by signature).

    Returns:
        Batch summary dict with at minimum:
          - "usersProcessed": int
          - "usersErrored":   int
          - "status":         "ok" | "partial" | "error"

    Raises:
        Exception: Propagated so EventBridge records the execution as failed.
    """
    logger.info(json.dumps({"message": "Archaeologist handler invoked"}))

    try:
        # ------------------------------------------------------------------
        # 1. Determine user list
        # ------------------------------------------------------------------
        user_ids: list[str] | None = event.get("user_ids")

        if user_ids:
            logger.info(
                json.dumps(
                    {
                        "message": "Processing explicit user list from event",
                        "userCount": len(user_ids),
                    }
                )
            )
        else:
            logger.info(
                json.dumps({"message": "No user_ids in event — scanning personas table"})
            )
            user_ids = _scan_all_user_ids()
            logger.info(
                json.dumps(
                    {
                        "message": "Personas table scan complete",
                        "userCount": len(user_ids),
                    }
                )
            )

        if not user_ids:
            logger.warning(json.dumps({"message": "No users found — skipping archaeology run"}))
            return {"status": "ok", "usersProcessed": 0, "usersErrored": 0}

        # ------------------------------------------------------------------
        # 2. Build the history-fetch callable for the service layer
        # ------------------------------------------------------------------
        fetch_history_fn = _build_fetch_history_fn(months=24)

        # ------------------------------------------------------------------
        # 3. Run the batch
        # ------------------------------------------------------------------
        summary: dict = archaeologist_service.run_weekly_batch(
            user_ids=user_ids,
            fetch_history_fn=fetch_history_fn,
        )

        users_processed: int = summary.get("usersProcessed", 0)
        users_errored: int = summary.get("usersErrored", 0)
        status: str = "partial" if users_errored > 0 else "ok"

        logger.info(
            json.dumps(
                {
                    "message": "Archaeology batch complete",
                    "usersProcessed": users_processed,
                    "usersErrored": users_errored,
                    "status": status,
                }
            )
        )

        return {
            "status": status,
            "usersProcessed": users_processed,
            "usersErrored": users_errored,
        }

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "Archaeology batch failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            exc_info=True,
        )
        raise
