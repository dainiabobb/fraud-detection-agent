"""
PatternDiscoveryService — offline pattern synthesis for the Sentinel cache.

Runs on an EventBridge daily schedule.  It:
  1. Queries the FraudDecisions GSI for BLOCK decisions from the last 7 days.
  2. Fetches all currently ACTIVE patterns from DynamoDB.
  3. Prompts Sonnet to cluster the BLOCK decisions and produce a JSON array of
     pattern actions (ADD / REFINE / RETIRE).
  4. Applies each action to DynamoDB (create, update, or soft-delete).
  5. Invalidates the Redis pattern cache so the Sentinel picks up the changes
     on the next warm request.
  6. Publishes CloudWatch metrics (new patterns added, retired).

Pattern lifecycle:
  ACTIVE  — loaded into the Sentinel's pattern cache; matched against transactions.
  RETIRED — soft-deleted by setting status="RETIRED"; never deleted from DynamoDB.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.models.pattern import FraudPattern
from src.utils.metrics import FraudMetrics

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from *text*."""
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        raise ValueError(f"No JSON object or array found in LLM response: {text!r}")

    if obj_start == -1:
        start, end_char = arr_start, "]"
    elif arr_start == -1:
        start, end_char = obj_start, "}"
    else:
        start = min(obj_start, arr_start)
        end_char = "}" if start == obj_start else "]"

    end = text.rfind(end_char)
    if end == -1 or end < start:
        raise ValueError(
            f"No closing '{end_char}' found in LLM response: {text!r}"
        )
    return text[start : end + 1]


class PatternDiscoveryService:
    """Offline agent that discovers and maintains the Sentinel's fraud patterns."""

    def __init__(
        self,
        config: Config,
        dynamodb_client: DynamoDBClient,
        bedrock_client: BedrockClient,
        redis_client: RedisClient,
        metrics: FraudMetrics,
    ) -> None:
        self._config = config
        self._dynamo = dynamodb_client
        self._bedrock = bedrock_client
        self._redis = redis_client
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_patterns(self) -> list[dict]:
        """Run the full pattern discovery cycle.

        Returns:
            List of action dicts taken during this run. Each dict has the shape:
            {
                "action":        "ADD" | "REFINE" | "RETIRE",
                "pattern_name":  str,
                "status":        "ok" | "error",
                "detail":        str,   # human-readable summary
            }
        """
        now = datetime.now(timezone.utc)
        since_iso = (now - timedelta(days=7)).isoformat()

        # Step 1 — fetch recent BLOCK decisions from the GSI.
        recent_blocks = self._fetch_recent_blocks(since_iso)
        if not recent_blocks:
            logger.info(
                "No BLOCK decisions in the last 7 days — skipping pattern discovery"
            )
            return []

        # Step 2 — fetch current ACTIVE patterns.
        existing_patterns = self._dynamo.scan_patterns(
            table_name=self._config.dynamo_table_patterns,
            status="ACTIVE",
        )

        logger.info(
            "Starting pattern discovery",
            extra={
                "block_count": len(recent_blocks),
                "existing_pattern_count": len(existing_patterns),
            },
        )

        # Step 3 — invoke Sonnet for cluster analysis.
        prompt = self._load_prompt(
            "pattern_discovery",
            recent_blocks_json=json.dumps(recent_blocks, default=str),
            existing_patterns_json=json.dumps(existing_patterns, default=str),
        )
        sonnet_response = self._bedrock.invoke_sonnet(prompt, max_tokens=4096)
        tokens_used: int = sonnet_response.get("tokens_used", 0)
        self._metrics.record_token_spend(tokens_used)

        # Step 4 — parse the action array.
        actions: list[dict] = self._parse_pattern_actions(
            sonnet_response.get("text", "")
        )

        # Step 5 — apply each action to DynamoDB.
        results: list[dict] = []
        for action in actions:
            result = self._apply_action(action, now)
            results.append(result)

        # Step 6 — invalidate Redis pattern cache so Sentinel picks up changes.
        try:
            # Write an empty list to invalidate; next Sentinel request will
            # rebuild the cache from DynamoDB on cache miss.
            self._redis.set_pattern_cache([], ttl=1)
            logger.info("Pattern cache invalidated in Redis")
        except Exception:
            logger.warning(
                "Failed to invalidate pattern cache in Redis — stale cache may persist",
                exc_info=True,
            )

        logger.info(
            "Pattern discovery cycle complete",
            extra={
                "actions_taken": len(results),
                "tokens_used": tokens_used,
            },
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_recent_blocks(self, since_iso: str) -> list[dict]:
        """Query the verdict-timestamp GSI for BLOCK decisions since *since_iso*."""
        try:
            return self._dynamo.get_decisions_by_verdict(
                table_name=self._config.dynamo_table_decisions,
                verdict="BLOCK",
                since_timestamp=since_iso,
            )
        except Exception:
            logger.error(
                "Failed to fetch recent BLOCK decisions from DynamoDB",
                extra={
                    "errorType": "DynamoDBQueryFailure",
                    "since": since_iso,
                },
                exc_info=True,
            )
            return []

    def _parse_pattern_actions(self, text: str) -> list[dict]:
        """Parse the JSON array of pattern actions from the model response.

        Returns an empty list on parse failure so the cycle degrades gracefully.
        """
        try:
            json_str = _extract_json(text)
            result = json.loads(json_str)
            if not isinstance(result, list):
                raise ValueError(
                    f"Expected a JSON array from pattern_discovery prompt, "
                    f"got {type(result).__name__}"
                )
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse pattern_discovery LLM response — no actions taken",
                extra={
                    "errorType": type(exc).__name__,
                    "error_detail": str(exc),
                    "raw_text": text[:500],
                },
            )
            return []

    def _apply_action(self, action: dict, now: datetime) -> dict:
        """Apply a single pattern action dict to DynamoDB.

        Returns a result summary dict suitable for the caller's action log.
        """
        action_type: str = action.get("action", "")
        pattern_name: str = action.get("pattern_name", "")

        if not action_type or not pattern_name:
            logger.warning(
                "Skipping malformed action — missing 'action' or 'pattern_name'",
                extra={"raw_action": action},
            )
            return {
                "action": action_type,
                "pattern_name": pattern_name,
                "status": "error",
                "detail": "Malformed action: missing required fields.",
            }

        try:
            if action_type == "ADD":
                return self._apply_add(action, now)
            elif action_type == "REFINE":
                return self._apply_refine(action)
            elif action_type == "RETIRE":
                return self._apply_retire(action)
            else:
                logger.warning(
                    "Unknown pattern action type — skipping",
                    extra={"action_type": action_type, "pattern_name": pattern_name},
                )
                return {
                    "action": action_type,
                    "pattern_name": pattern_name,
                    "status": "error",
                    "detail": f"Unknown action type: '{action_type}'",
                }
        except Exception as exc:
            logger.error(
                "Failed to apply pattern action",
                extra={
                    "errorType": type(exc).__name__,
                    "action_type": action_type,
                    "pattern_name": pattern_name,
                    "error_detail": str(exc),
                },
                exc_info=True,
            )
            return {
                "action": action_type,
                "pattern_name": pattern_name,
                "status": "error",
                "detail": str(exc),
            }

    def _apply_add(self, action: dict, now: datetime) -> dict:
        """Create a new ACTIVE FraudPattern in DynamoDB."""
        pattern_name: str = action["pattern_name"]
        new_pattern = FraudPattern(
            pattern_name=pattern_name,
            description=action.get("description", ""),
            detection_rule=action.get("detection_rule", ""),
            precision=float(action.get("precision", 0.0)),
            recall=None,
            sample_transaction_ids=action.get("sample_transaction_ids", []),
            discovered_at=now.isoformat(),
            status="ACTIVE",
        )
        pattern_item = new_pattern.model_dump()
        pattern_item["PK"] = f"PATTERN#{pattern_name}"

        self._dynamo.put_pattern(
            table_name=self._config.dynamo_table_patterns,
            pattern=pattern_item,
        )
        self._metrics.record_new_pattern()
        logger.info(
            "New fraud pattern added",
            extra={
                "pattern_name": pattern_name,
                "precision": new_pattern.precision,
            },
        )
        return {
            "action": "ADD",
            "pattern_name": pattern_name,
            "status": "ok",
            "detail": f"Pattern '{pattern_name}' created (precision={new_pattern.precision:.2f}).",
        }

    def _apply_refine(self, action: dict) -> dict:
        """Update the detection_rule of an existing pattern."""
        pattern_name: str = action["pattern_name"]
        existing = self._dynamo.get_pattern(
            table_name=self._config.dynamo_table_patterns,
            pattern_name=pattern_name,
        )
        if existing is None:
            logger.warning(
                "REFINE action targeting unknown pattern — skipping",
                extra={"pattern_name": pattern_name},
            )
            return {
                "action": "REFINE",
                "pattern_name": pattern_name,
                "status": "error",
                "detail": f"Pattern '{pattern_name}' not found; cannot refine.",
            }

        suggested_rule = action.get("suggested_rule_change")
        if suggested_rule:
            existing["detection_rule"] = suggested_rule

        reason = action.get("reason", "")
        # Append a note about the refinement into the pattern's description.
        existing["description"] = (
            f"{existing.get('description', '')} [REFINED: {reason}]"
        ).strip()

        self._dynamo.put_pattern(
            table_name=self._config.dynamo_table_patterns,
            pattern=existing,
        )
        logger.info(
            "Fraud pattern refined",
            extra={"pattern_name": pattern_name, "reason": reason},
        )
        return {
            "action": "REFINE",
            "pattern_name": pattern_name,
            "status": "ok",
            "detail": f"Pattern '{pattern_name}' refined: {reason}",
        }

    def _apply_retire(self, action: dict) -> dict:
        """Soft-delete a pattern by setting its status to RETIRED."""
        pattern_name: str = action["pattern_name"]
        existing = self._dynamo.get_pattern(
            table_name=self._config.dynamo_table_patterns,
            pattern_name=pattern_name,
        )
        if existing is None:
            logger.warning(
                "RETIRE action targeting unknown pattern — skipping",
                extra={"pattern_name": pattern_name},
            )
            return {
                "action": "RETIRE",
                "pattern_name": pattern_name,
                "status": "error",
                "detail": f"Pattern '{pattern_name}' not found; cannot retire.",
            }

        reason = action.get("reason", "")
        existing["status"] = "RETIRED"
        existing["description"] = (
            f"{existing.get('description', '')} [RETIRED: {reason}]"
        ).strip()

        self._dynamo.put_pattern(
            table_name=self._config.dynamo_table_patterns,
            pattern=existing,
        )
        self._metrics.record_pattern_retired()
        logger.info(
            "Fraud pattern retired",
            extra={"pattern_name": pattern_name, "reason": reason},
        )
        return {
            "action": "RETIRE",
            "pattern_name": pattern_name,
            "status": "ok",
            "detail": f"Pattern '{pattern_name}' retired: {reason}",
        }

    def _load_prompt(self, template_name: str, **kwargs: object) -> str:
        """Load and format a prompt template from the prompts/ directory."""
        path = os.path.join(_PROMPTS_DIR, f"{template_name}.txt")
        with open(path, encoding="utf-8") as fh:
            template = fh.read()
        return template.format(**kwargs)
